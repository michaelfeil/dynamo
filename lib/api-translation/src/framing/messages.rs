//! Anthropic Messages framing. The whole multi-iteration ReAct loop is one assistant message whose
//! content blocks span every iteration, so server-tool activity rides native `tool_use` /
//! `tool_result` blocks and *is* carried into a following request.
//!
//! Block-framing (thinking->text->tool_use close order, `signature_delta` on thinking close) adapted
//! from basetenlabs/dynamo @ 68dec805 (Apache-2.0):
//! https://github.com/basetenlabs/dynamo/blob/68dec805e22e66851dc0bf2a8faba7ce0665b00d/lib/llm/src/protocols/anthropic/stream_converter.rs
//! Output types: vendored `dynamo_protocols::types::anthropic`.
//! SPDX-License-Identifier: Apache-2.0.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use serde_json::Value;

use dynamo_protocols::types::anthropic::{
    AnthropicDelta, AnthropicErrorBody, AnthropicErrorResponse, AnthropicMessageDeltaBody,
    AnthropicMessageResponse, AnthropicResponseContentBlock, AnthropicStopReason,
    AnthropicStreamEvent, AnthropicUsage,
};
use dynamo_protocols::types::{
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestToolMessageContent,
    CompletionUsage, FinishReason, ReasoningContent,
};

use super::{
    BufferedResponse, CompletedIteration, OpenIterationScope, ProtocolEnvelope, StagedIteration,
    StreamFraming,
};
use crate::baseten_response_extension::{
    BasetenFrame, BasetenResponseExtension, IterationScope, ServerToolCallOutcome,
};
use crate::coding_adapter::{CodingAdapter, ToolCallStatus, ToolCallToRender, ToolResultToRender};
use crate::model::{ServerToolCall, Termination, ToolCall, ToolInvocation, ToolOutput};
use crate::wire::{next_id_seq, sse_frame, to_json_string};
use crate::{CcMessage, SemanticChunk};

/// The engine emits no cryptographic thinking signature, and the Anthropic shape requires the field.
const ERASED_SIGNATURE: &str = "erased";
const MESSAGE_OBJECT: &str = "message";
const ASSISTANT_ROLE: &str = "assistant";

/// Monomorphizes [`BasetenResponseExtension`] on Messages' own usage shape — confined here since nothing
/// outside this module renders Messages.
type MessagesExtension = BasetenResponseExtension<AnthropicUsage>;

pub(super) struct MessagesEnvelope {
    coding_adapter: Option<Box<dyn CodingAdapter>>,
}

impl MessagesEnvelope {
    pub(super) fn new(coding_adapter: Option<Box<dyn CodingAdapter>>) -> Self {
        Self { coding_adapter }
    }
}

impl ProtocolEnvelope for MessagesEnvelope {
    fn stream_framing(self: Box<Self>, model: String) -> Box<dyn StreamFraming> {
        Box::new(MessagesFraming::new(model, self.coding_adapter))
    }

    fn buffered_body(&self, response: &BufferedResponse<'_>) -> String {
        let body = AnthropicMessageResponse {
            id: messages_id(),
            object_type: MESSAGE_OBJECT.to_string(),
            role: ASSISTANT_ROLE.to_string(),
            content: transcript_blocks(
                response.transcript,
                self.coding_adapter.as_deref(),
                &response.failed_server_tool_call_ids(),
            ),
            model: response.model.to_string(),
            stop_reason: Some(anthropic_stop_reason(response.termination)),
            stop_sequence: None,
            usage: anthropic_usage(response.usage),
        };
        let extension = messages_extension(
            response.iterations,
            response.request_server_tool_calls,
            response.termination,
        );
        to_json_string(&BasetenFrame {
            body: &body,
            baseten: (!extension.is_empty()).then_some(&extension),
        })
    }
}

/// Anthropic's error body, the shape both the buffered error and the `error` SSE event carry.
pub(super) fn error_json(message: &str) -> String {
    to_json_string(&AnthropicErrorResponse {
        object_type: "error".to_string(),
        error: AnthropicErrorBody {
            error_type: "api_error".to_string(),
            message: message.to_string(),
        },
    })
}

/// The transcript only ever holds arguments the parser already proved to be JSON, so a parse failure
/// here is a history-fold bug: report it loudly and ship an empty object rather than a `null` input,
/// which no Anthropic client can distinguish from a tool that takes none.
fn tool_call_input(call_id: &str, arguments: &str) -> serde_json::Value {
    serde_json::from_str(arguments).unwrap_or_else(|error| {
        debug_assert!(false, "transcript tool args must parse: {error}");
        tracing::error!("tool call `{call_id}` has unparseable arguments in history: {error}");
        serde_json::json!({})
    })
}

/// A tool result as the structure the provider returned. Two encodings arrive here: the transcript
/// flattens a result to text, and an all-text MCP result is already one joined string
/// ([`crate::tools::providers::mcp`]), so the JSON a provider wrote reaches both framings wrapped in
/// a string. Unwrapped once here so an adapter reads structure, per
/// [`ToolResultToRender::content`]. A payload that was never JSON stays a string.
fn structured_content(content: &Value) -> Cow<'_, Value> {
    match content {
        Value::String(text) => serde_json::from_str(text)
            .map(Cow::Owned)
            .unwrap_or(Cow::Borrowed(content)),
        _ => Cow::Borrowed(content),
    }
}

/// The loop transcript as Anthropic content blocks, in transcript order — which is the order the model
/// itself saw, so echoing this body back reproduces its prefix. A `tool_use` block is not necessarily
/// adjacent to its `tool_result`: an iteration that mixes server and client calls declares both in one
/// assistant message, and only the server one is answered here.
fn transcript_blocks(
    transcript: &[CcMessage],
    coding_adapter: Option<&dyn CodingAdapter>,
    failed_server_tool_call_ids: &HashSet<&str>,
) -> Vec<AnthropicResponseContentBlock> {
    let mut blocks = Vec::new();
    let mut rendered_calls: HashMap<&str, &str> = HashMap::new();
    for message in transcript {
        match message {
            CcMessage::Assistant(assistant) => {
                if let Some(ReasoningContent::Text(reasoning)) = &assistant.reasoning_content {
                    blocks.push(AnthropicResponseContentBlock::Thinking {
                        thinking: reasoning.clone(),
                        signature: ERASED_SIGNATURE.to_string(),
                    });
                }
                if let Some(ChatCompletionRequestAssistantMessageContent::Text(text)) =
                    &assistant.content
                {
                    blocks.push(AnthropicResponseContentBlock::Text {
                        text: text.clone(),
                        citations: None,
                    });
                }
                for call in assistant.tool_calls.iter().flatten() {
                    if let Some(block) = adapter_call_block(
                        coding_adapter,
                        &ToolCallToRender {
                            tool_name: &call.function.name,
                            id: &call.id,
                            args: &call.function.arguments,
                            status: if failed_server_tool_call_ids.contains(call.id.as_str()) {
                                ToolCallStatus::Failed
                            } else {
                                ToolCallStatus::Completed
                            },
                        },
                    ) {
                        rendered_calls.insert(&call.id, &call.function.name);
                        blocks.push(block);
                        continue;
                    }
                    blocks.push(AnthropicResponseContentBlock::ToolUse {
                        id: call.id.clone(),
                        name: call.function.name.clone(),
                        input: tool_call_input(&call.id, &call.function.arguments),
                    });
                }
            }
            CcMessage::Tool(tool) => {
                let ChatCompletionRequestToolMessageContent::Text(content) = &tool.content else {
                    continue;
                };
                let raw = Value::String(content.clone());
                if let Some(block) =
                    rendered_calls
                        .get(tool.tool_call_id.as_str())
                        .and_then(|tool_name| {
                            adapter_result_block(
                                coding_adapter,
                                &ToolResultToRender {
                                    tool_name,
                                    id: &tool.tool_call_id,
                                    content: &raw,
                                    is_error: failed_server_tool_call_ids
                                        .contains(tool.tool_call_id.as_str()),
                                },
                            )
                        })
                {
                    blocks.push(block);
                    continue;
                }
                blocks.push(AnthropicResponseContentBlock::ToolResult {
                    tool_use_id: tool.tool_call_id.clone(),
                    content: content.clone(),
                    is_error: failed_server_tool_call_ids
                        .contains(tool.tool_call_id.as_str())
                        .then_some(true),
                });
            }
            // The loop appends nothing else; the client's own prefix is not part of the transcript.
            _ => {}
        }
    }
    blocks
}

fn tool_result_block(call: &ToolCall, output: &ToolOutput) -> AnthropicResponseContentBlock {
    AnthropicResponseContentBlock::ToolResult {
        tool_use_id: call.id.clone(),
        content: output.text(),
        is_error: output.is_error().then_some(true),
    }
}

/// The adapter's block for a call, or `None` when there is no adapter or it did not expand this
/// call. The buffered and streamed framings differ only in where they read the call from, so the
/// trait-call shape lives here once rather than at each of their sites.
fn adapter_call_block(
    adapter: Option<&dyn CodingAdapter>,
    call: &ToolCallToRender<'_>,
) -> Option<AnthropicResponseContentBlock> {
    Some(rendered_block(adapter?.render_tool_call(call)?.item))
}

/// The adapter's block for a resolved call's result. Same contract as [`adapter_call_block`].
fn adapter_result_block(
    adapter: Option<&dyn CodingAdapter>,
    result: &ToolResultToRender<'_>,
) -> Option<AnthropicResponseContentBlock> {
    let adapter = adapter?;
    let structured = structured_content(result.content);
    Some(rendered_block(adapter.render_tool_result(
        &ToolResultToRender {
            content: &structured,
            ..*result
        },
    )?))
}

/// The adapter's block for a call, as the arguments a streamed block's delta carries.
fn adapter_call_args(
    adapter: Option<&dyn CodingAdapter>,
    call: &ToolCallToRender<'_>,
) -> Option<String> {
    adapter?
        .render_tool_call(call)?
        .item
        .get("input")
        .map(Value::to_string)
}

/// The adapter's own object as this protocol's typed block. Total by construction: the block enum's
/// untagged catch-all takes any shape the typed variants reject, so an unrecognised block
/// round-trips rather than being lost.
fn rendered_block(item: Value) -> AnthropicResponseContentBlock {
    match serde_json::from_value(item.clone()) {
        Ok(block) => block,
        Err(_) => AnthropicResponseContentBlock::Other(item),
    }
}

/// Synthetic Anthropic Messages response id (`msg_<seq>`).
fn messages_id() -> String {
    format!("msg_{}", next_id_seq())
}

/// Which content block is currently open. Tool blocks open and close inline, so they can never be
/// open — which is why this is not the crate's `ContentKind`.
#[derive(PartialEq)]
enum OpenBlock {
    None,
    Thinking(u32),
    Text(u32),
}

struct MessagesFraming {
    model: String,
    message_id: String,
    started: bool,
    next_block_index: u32,
    open: OpenBlock,
    /// Buffered to ride the next extras-preserving frame (see [`event_frame`]).
    pending_baseten: MessagesExtension,
    open_iteration_scope: OpenIterationScope<AnthropicUsage>,
    coding_adapter: Option<Box<dyn CodingAdapter>>,
}

impl MessagesFraming {
    fn new(model: String, coding_adapter: Option<Box<dyn CodingAdapter>>) -> Self {
        Self {
            model,
            message_id: messages_id(),
            started: false,
            next_block_index: 0,
            open: OpenBlock::None,
            pending_baseten: MessagesExtension::default(),
            open_iteration_scope: OpenIterationScope::default(),
            coding_adapter,
        }
    }

    fn rendered_call_block(&self, call: &ToolCall) -> Option<AnthropicResponseContentBlock> {
        adapter_call_block(
            self.coding_adapter.as_deref(),
            &ToolCallToRender {
                tool_name: &call.name,
                id: &call.id,
                args: &call.raw_args,
                status: ToolCallStatus::InProgress,
            },
        )
    }

    fn rendered_call_args(&self, call: &ToolCall) -> Option<String> {
        adapter_call_args(
            self.coding_adapter.as_deref(),
            &ToolCallToRender {
                tool_name: &call.name,
                id: &call.id,
                args: &call.raw_args,
                status: ToolCallStatus::Completed,
            },
        )
    }

    fn rendered_result_block(
        &self,
        call: &ToolCall,
        output: &ToolOutput,
    ) -> Option<AnthropicResponseContentBlock> {
        adapter_result_block(
            self.coding_adapter.as_deref(),
            &ToolResultToRender {
                tool_name: &call.name,
                id: &call.id,
                content: &output.content,
                is_error: output.is_error(),
            },
        )
    }

    fn ensure_started(&mut self, frames: &mut Vec<String>) {
        if self.started {
            return;
        }
        self.started = true;
        let message = AnthropicMessageResponse {
            id: self.message_id.clone(),
            object_type: MESSAGE_OBJECT.to_string(),
            role: ASSISTANT_ROLE.to_string(),
            content: vec![],
            model: self.model.clone(),
            stop_reason: None,
            stop_sequence: None,
            usage: AnthropicUsage::default(),
        };
        frames.push(self.event_frame(AnthropicStreamEvent::MessageStart { message }));
    }

    /// Open a content block at the next index, returning it. Caller records `self.open` for
    /// streaming blocks; tool blocks close inline so they don't.
    fn open_block(
        &mut self,
        frames: &mut Vec<String>,
        content_block: AnthropicResponseContentBlock,
    ) -> u32 {
        let index = self.next_block_index;
        self.next_block_index += 1;
        frames.push(self.event_frame(AnthropicStreamEvent::ContentBlockStart {
            index,
            content_block,
        }));
        index
    }

    fn block_stop(&mut self, index: u32) -> String {
        self.event_frame(AnthropicStreamEvent::ContentBlockStop { index })
    }

    fn block_delta(&mut self, index: u32, delta: AnthropicDelta) -> String {
        self.event_frame(AnthropicStreamEvent::ContentBlockDelta { index, delta })
    }

    /// Close the currently open streaming block (thinking closes with a placeholder
    /// `signature_delta` per the Anthropic spec).
    fn close_open(&mut self, frames: &mut Vec<String>) {
        match std::mem::replace(&mut self.open, OpenBlock::None) {
            OpenBlock::None => {}
            OpenBlock::Thinking(index) => {
                frames.push(self.block_delta(
                    index,
                    AnthropicDelta::SignatureDelta {
                        signature: ERASED_SIGNATURE.to_string(),
                    },
                ));
                frames.push(self.block_stop(index));
            }
            OpenBlock::Text(index) => frames.push(self.block_stop(index)),
        }
    }

    /// Build one SSE frame, flushing `pending_baseten` onto the next extras-preserving event. The SDK
    /// rebuilds `*_stop` events from its snapshot and drops extra fields, so nothing may ride those.
    fn event_frame(&mut self, event: AnthropicStreamEvent) -> String {
        let name = event_name(&event);
        let baseten = if preserves_extra_fields(&event) {
            std::mem::take(&mut self.pending_baseten)
        } else {
            MessagesExtension::default()
        };
        sse_frame(
            Some(name),
            &to_json_string(&BasetenFrame {
                body: &event,
                baseten: (!baseten.is_empty()).then_some(&baseten),
            }),
        )
    }
}

impl StreamFraming for MessagesFraming {
    fn on_chunk(&mut self, chunk: &SemanticChunk) -> Vec<String> {
        let mut frames = Vec::new();
        match chunk {
            SemanticChunk::ThinkingDelta(thinking) => {
                self.ensure_started(&mut frames);
                let index = match self.open {
                    OpenBlock::Thinking(index) => index,
                    OpenBlock::None | OpenBlock::Text(_) => {
                        self.close_open(&mut frames);
                        let index = self.open_block(
                            &mut frames,
                            AnthropicResponseContentBlock::Thinking {
                                thinking: String::new(),
                                signature: String::new(),
                            },
                        );
                        self.open = OpenBlock::Thinking(index);
                        index
                    }
                };
                frames.push(self.block_delta(
                    index,
                    AnthropicDelta::ThinkingDelta {
                        thinking: thinking.clone(),
                    },
                ));
            }
            SemanticChunk::TextDelta(text) => {
                self.ensure_started(&mut frames);
                let index = match self.open {
                    OpenBlock::Text(index) => index,
                    OpenBlock::None | OpenBlock::Thinking(_) => {
                        self.close_open(&mut frames);
                        let index = self.open_block(
                            &mut frames,
                            AnthropicResponseContentBlock::Text {
                                text: String::new(),
                                citations: None,
                            },
                        );
                        self.open = OpenBlock::Text(index);
                        index
                    }
                };
                frames.push(
                    self.block_delta(index, AnthropicDelta::TextDelta { text: text.clone() }),
                );
            }
            SemanticChunk::ToolCall(call) => {
                self.ensure_started(&mut frames);
                self.close_open(&mut frames);
                // One block: start + one complete-args delta + stop, so no stop precedes the args
                // (no orphan-close). `input` starts empty — the SDK builds it from the delta.
                let block = self.rendered_call_block(call).unwrap_or_else(|| {
                    AnthropicResponseContentBlock::ToolUse {
                        id: call.id.clone(),
                        name: call.name.clone(),
                        input: serde_json::json!({}),
                    }
                });
                let partial_json = self
                    .rendered_call_args(call)
                    .unwrap_or_else(|| call.args.to_string());
                let index = self.open_block(&mut frames, block);
                frames
                    .push(self.block_delta(index, AnthropicDelta::InputJsonDelta { partial_json }));
                frames.push(self.block_stop(index));
            }
            SemanticChunk::Usage(_) | SemanticChunk::Stop { .. } => {}
        }
        frames
    }

    /// `tool_result` blocks only: the transcript needs no `baseten` copy, since these blocks *are*
    /// what a client echoes back.
    fn emit_completed_iteration(&mut self, iteration: &CompletedIteration<'_>) -> Vec<String> {
        let mut frames = Vec::new();
        if iteration.invocations.is_empty() {
            return frames;
        }
        self.ensure_started(&mut frames);
        self.close_open(&mut frames);
        for ToolInvocation {
            server_call,
            output,
        } in iteration.invocations
        {
            let call = &server_call.call;
            let block = self
                .rendered_result_block(call, output)
                .unwrap_or_else(|| tool_result_block(call, output));
            let index = self.open_block(&mut frames, block);
            frames.push(self.block_stop(index));
        }
        frames
    }

    fn emit_client_tool_calls(&mut self, _calls: &[ToolCall]) -> Vec<String> {
        // Already streamed as `tool_use` blocks by `on_chunk`.
        Vec::new()
    }

    fn emit_dispatched_server_tool(
        &mut self,
        _iteration: u32,
        _server_call: &ServerToolCall,
    ) -> Vec<String> {
        // Ditto: the native `tool_use` block already went out, with the same id and arguments.
        Vec::new()
    }

    fn stage_iteration(&mut self, staged: StagedIteration<'_>) -> Vec<String> {
        // Unreachable while the loop stages after draining a call, which always starts the message and
        // always emits an extras-preserving frame before the next one.
        if !self.started {
            tracing::warn!("iteration scope dropped before message_start (call cadence changed?)");
            return Vec::new();
        }
        self.pending_baseten.iterations.extend(
            self.open_iteration_scope
                .stage(staged.scope(anthropic_usage)),
        );
        Vec::new()
    }

    fn finish(
        &mut self,
        termination: Termination,
        usage: &CompletionUsage,
        server_tool_calls: &[ServerToolCallOutcome],
    ) -> Vec<String> {
        let mut frames = Vec::new();
        // Mandatory even for a call carrying only a finish_reason, or the client holds a committed
        // 200 with no message to close.
        self.ensure_started(&mut frames);
        self.close_open(&mut frames);
        self.pending_baseten
            .iterations
            .extend(self.open_iteration_scope.close());
        self.pending_baseten.merge(MessagesExtension {
            request: termination.request_scope(server_tool_calls),
            ..MessagesExtension::default()
        });
        frames.push(self.event_frame(AnthropicStreamEvent::MessageDelta {
            delta: AnthropicMessageDeltaBody {
                stop_reason: Some(anthropic_stop_reason(termination)),
                stop_sequence: None,
            },
            usage: anthropic_usage(usage),
        }));
        frames.push(self.event_frame(AnthropicStreamEvent::MessageStop {}));
        frames
    }

    // No code slot: Anthropic's `error.type` is a closed vocabulary, not ours to extend.
    fn error_sse_frame(&mut self, _error_code: Option<&str>, message: &str) -> String {
        sse_frame(Some("error"), &error_json(message))
    }
}

fn messages_extension(
    iterations: &[IterationScope<CompletionUsage>],
    request_server_tool_calls: &[ServerToolCallOutcome],
    termination: Termination,
) -> MessagesExtension {
    MessagesExtension {
        iterations: iterations
            .iter()
            .cloned()
            .map(|iteration| iteration.into_usage_only(anthropic_usage))
            .collect(),
        request: termination.request_scope(request_server_tool_calls),
    }
}

/// The cap's closest fit is `pause_turn`, Anthropic's own value for a server-tool turn suspended with
/// the model still wanting to continue; its client contract — resend the message as-is to resume — is
/// exactly ours, where `tool_use` would ask the client to execute calls TB already ran.
fn anthropic_stop_reason(termination: Termination) -> AnthropicStopReason {
    match termination {
        Termination::ReactCapExhausted => AnthropicStopReason::PauseTurn,
        Termination::Model(FinishReason::ToolCalls | FinishReason::FunctionCall) => {
            AnthropicStopReason::ToolUse
        }
        Termination::Model(FinishReason::Length) => AnthropicStopReason::MaxTokens,
        Termination::Model(FinishReason::Stop) => AnthropicStopReason::EndTurn,
        Termination::Model(FinishReason::ContentFilter) => AnthropicStopReason::Refusal,
    }
}

/// Anthropic's token buckets are disjoint — `input_tokens` EXCLUDES cache reads, where CC's
/// `prompt_tokens` includes them. SEG bills the sum of the three buckets, so emitting the CC total
/// here charges every cache-read token twice (`go/shared-endpoints-gateway/pkg/token-counting`).
fn anthropic_usage(usage: &CompletionUsage) -> AnthropicUsage {
    let cache_read_input_tokens = usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|details| details.cached_tokens)
        // Clamped: a backend over-reporting cache reads would underflow the subtraction below.
        .map(|cached| cached.min(usage.prompt_tokens));
    AnthropicUsage {
        input_tokens: usage.prompt_tokens - cache_read_input_tokens.unwrap_or(0),
        output_tokens: usage.completion_tokens,
        // OpenAI-compatible backends do not report cache-write counts. Emit an explicit 0 (not
        // absent) so downstream metering never interprets a missing field as "the whole prompt
        // was written to cache".
        cache_creation_input_tokens: Some(0),
        cache_read_input_tokens,
    }
}

/// The SSE `event:` name, which must match the body's `type`.
fn event_name(event: &AnthropicStreamEvent) -> &'static str {
    match event {
        AnthropicStreamEvent::MessageStart { .. } => "message_start",
        AnthropicStreamEvent::ContentBlockStart { .. } => "content_block_start",
        AnthropicStreamEvent::ContentBlockDelta { .. } => "content_block_delta",
        AnthropicStreamEvent::ContentBlockStop { .. } => "content_block_stop",
        AnthropicStreamEvent::MessageDelta { .. } => "message_delta",
        AnthropicStreamEvent::MessageStop {} => "message_stop",
        AnthropicStreamEvent::Ping {} => "ping",
        AnthropicStreamEvent::Error { .. } => "error",
    }
}

/// Whether the Anthropic SDK forwards unknown fields on this event intact. It rebuilds the `*_stop`
/// and `ping` events from its own snapshot, so anything extra on those is lost.
fn preserves_extra_fields(event: &AnthropicStreamEvent) -> bool {
    match event {
        AnthropicStreamEvent::MessageStart { .. }
        | AnthropicStreamEvent::ContentBlockStart { .. }
        | AnthropicStreamEvent::ContentBlockDelta { .. }
        | AnthropicStreamEvent::MessageDelta { .. } => true,
        AnthropicStreamEvent::ContentBlockStop { .. }
        | AnthropicStreamEvent::MessageStop {}
        | AnthropicStreamEvent::Ping {}
        | AnthropicStreamEvent::Error { .. } => false,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn buffered_web_search_uses_anthropic_blocks_and_drops_page_text() {
        use crate::test_utils::FakeMessagesSearchAdapter;

        let name = "baseten__provider__search";
        let adapter = FakeMessagesSearchAdapter { tool_name: name };
        let transcript = vec![
            serde_json::from_value(serde_json::json!({
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": name, "arguments": "{\"search_queries\":[\"rust\"]}"}
                }]
            }))
            .unwrap(),
            serde_json::from_value(serde_json::json!({
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "{\"payload\":{\"results\":[{\"url\":\"https://example.com/rust\",\"title\":\"Rust\",\"text\":\"hidden\"}]}}"
            }))
            .unwrap(),
        ];
        let blocks = serde_json::to_value(transcript_blocks(
            &transcript,
            Some(&adapter),
            &HashSet::new(),
        ))
        .unwrap();
        assert_eq!(blocks[0]["type"], "server_tool_use");
        assert_eq!(blocks[0]["name"], "web_search");
        assert_eq!(blocks[0]["id"], "srvtoolu_call_1");
        assert_eq!(blocks[0]["input"], serde_json::json!({"query": "rust"}));
        assert_eq!(blocks[1]["type"], "web_search_tool_result");
        assert_eq!(blocks[1]["tool_use_id"], blocks[0]["id"]);
        assert_eq!(
            blocks[1]["content"],
            serde_json::json!([{
                "type": "web_search_result",
                "title": "Rust",
                "url": "https://example.com/rust"
            }])
        );
        assert!(blocks[1]["content"][0].get("text").is_none());
    }

    /// The failed-call path: `is_error` in the transcript reaches both the call's status and the
    /// result block, and neither is exercised by the success test above.
    #[test]
    fn buffered_failed_web_search_renders_the_error_result_block() {
        use crate::test_utils::FakeMessagesSearchAdapter;

        let name = "baseten__provider__search";
        let adapter = FakeMessagesSearchAdapter { tool_name: name };
        let transcript = vec![
            serde_json::from_value(serde_json::json!({
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": name, "arguments": "{\"search_queries\":[\"rust\"]}"}
                }]
            }))
            .unwrap(),
            serde_json::from_value(serde_json::json!({
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "upstream refused"
            }))
            .unwrap(),
        ];
        let failed = HashSet::from(["call_1"]);
        let blocks =
            serde_json::to_value(transcript_blocks(&transcript, Some(&adapter), &failed)).unwrap();
        assert_eq!(blocks[0]["type"], "server_tool_use");
        assert_eq!(blocks[1]["type"], "web_search_tool_result");
        assert_eq!(
            blocks[1]["content"],
            serde_json::json!({
                "type": "web_search_tool_result_error",
                "error_code": "unavailable"
            })
        );
    }

    /// The generic-block fallback is load-bearing: an adapter is present but this call is not one
    /// it expanded, so it must render as a plain `tool_use` under the provider's own name.
    #[test]
    fn buffered_non_search_call_falls_back_to_the_generic_block() {
        use crate::test_utils::FakeMessagesSearchAdapter;

        let adapter = FakeMessagesSearchAdapter {
            tool_name: "baseten__provider__search",
        };
        let transcript = vec![
            serde_json::from_value(serde_json::json!({
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_9",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"berlin\"}"}
                }]
            }))
            .unwrap(),
            serde_json::from_value(serde_json::json!({
                "role": "tool", "tool_call_id": "call_9", "content": "sunny"
            }))
            .unwrap(),
        ];
        let blocks = serde_json::to_value(transcript_blocks(
            &transcript,
            Some(&adapter),
            &HashSet::new(),
        ))
        .unwrap();
        assert_eq!(blocks[0]["type"], "tool_use");
        assert_eq!(blocks[0]["name"], "get_weather");
        assert_eq!(blocks[0]["id"], "call_9");
        assert_eq!(blocks[1]["type"], "tool_result");
        assert_eq!(blocks[1]["tool_use_id"], "call_9");
    }

    /// Both degrade silently by design, so a test is the only witness: a result that was never JSON
    /// yields no citations, and args that do not parse fall back to the generic block.
    #[test]
    fn buffered_unparseable_content_and_args_degrade_without_inventing_data() {
        use crate::test_utils::FakeMessagesSearchAdapter;

        let name = "baseten__provider__search";
        let adapter = FakeMessagesSearchAdapter { tool_name: name };
        let transcript = vec![
            serde_json::from_value(serde_json::json!({
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": name, "arguments": "{\"search_queries\":[\"rust\"]}"}
                }]
            }))
            .unwrap(),
            serde_json::from_value(serde_json::json!({
                "role": "tool", "tool_call_id": "call_1", "content": "not json at all"
            }))
            .unwrap(),
        ];
        let blocks = serde_json::to_value(transcript_blocks(
            &transcript,
            Some(&adapter),
            &HashSet::new(),
        ))
        .unwrap();
        assert_eq!(blocks[1]["type"], "web_search_tool_result");
        assert_eq!(blocks[1]["content"], serde_json::json!([]));

        // Args that do not parse decline the adapter's block, which is what leaves the generic one
        // standing. Asserted on the adapter directly: reaching it through the transcript would trip
        // `tool_call_input`'s debug assert first, since unparseable history args are a fold bug.
        assert!(
            adapter
                .render_tool_call(&ToolCallToRender {
                    tool_name: name,
                    id: "call_2",
                    args: "{not json",
                    status: ToolCallStatus::Completed,
                })
                .is_none()
        );
    }

    /// The SSE `event:` line and the body's own `type` must agree, or the Anthropic SDK ignores the
    /// event. `event_name` restates the vendored enum's serde tags, so this pins the two together:
    /// a re-vendor that renames a variant fails here instead of silently shipping a mismatch.
    #[test]
    fn event_name_matches_the_serialized_type_for_every_variant() {
        let message = AnthropicMessageResponse {
            id: String::new(),
            object_type: MESSAGE_OBJECT.to_string(),
            role: ASSISTANT_ROLE.to_string(),
            content: vec![],
            model: String::new(),
            stop_reason: None,
            stop_sequence: None,
            usage: AnthropicUsage::default(),
        };
        let every_variant = [
            AnthropicStreamEvent::MessageStart { message },
            AnthropicStreamEvent::ContentBlockStart {
                index: 0,
                content_block: AnthropicResponseContentBlock::Text {
                    text: String::new(),
                    citations: None,
                },
            },
            AnthropicStreamEvent::ContentBlockDelta {
                index: 0,
                delta: AnthropicDelta::TextDelta {
                    text: String::new(),
                },
            },
            AnthropicStreamEvent::ContentBlockStop { index: 0 },
            AnthropicStreamEvent::MessageDelta {
                delta: AnthropicMessageDeltaBody {
                    stop_reason: None,
                    stop_sequence: None,
                },
                usage: AnthropicUsage::default(),
            },
            AnthropicStreamEvent::MessageStop {},
            AnthropicStreamEvent::Ping {},
        ];
        for event in every_variant {
            let body: serde_json::Value = serde_json::from_str(&to_json_string(&event)).unwrap();
            assert_eq!(body["type"], event_name(&event), "body: {body}");
        }
    }

    use dynamo_protocols::types::PromptTokensDetails;

    fn cc_usage(prompt_tokens: u32, cached_tokens: Option<u32>) -> CompletionUsage {
        CompletionUsage {
            prompt_tokens,
            completion_tokens: 7,
            total_tokens: prompt_tokens + 7,
            prompt_tokens_details: cached_tokens.map(|cached| PromptTokensDetails {
                cached_tokens: Some(cached),
                audio_tokens: None,
            }),
            completion_tokens_details: None,
        }
    }

    #[test]
    fn input_tokens_excludes_cache_reads() {
        let usage = anthropic_usage(&cc_usage(1000, Some(900)));
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.cache_read_input_tokens, Some(900));
    }

    /// Explicit `0`, never absent: downstream metering must not read a missing field as "the whole
    /// prompt was written to cache".
    #[test]
    fn cache_creation_input_tokens_is_explicit_zero() {
        let usage = anthropic_usage(&cc_usage(1000, Some(900)));
        assert_eq!(usage.cache_creation_input_tokens, Some(0));
        let wire = serde_json::to_value(&usage).unwrap();
        assert_eq!(wire["cache_creation_input_tokens"], 0, "{wire}");
    }

    #[test]
    fn input_tokens_is_the_cc_total_when_nothing_was_cached() {
        for cached_tokens in [None, Some(0)] {
            let usage = anthropic_usage(&cc_usage(1000, cached_tokens));
            assert_eq!(usage.input_tokens, 1000, "cached_tokens: {cached_tokens:?}");
        }
    }

    /// A backend over-reporting cache reads must not wrap the subtraction into a huge `input_tokens`.
    #[test]
    fn cache_reads_above_the_prompt_total_are_clamped() {
        let usage = anthropic_usage(&cc_usage(100, Some(1000)));
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.cache_read_input_tokens, Some(100));
    }
}
