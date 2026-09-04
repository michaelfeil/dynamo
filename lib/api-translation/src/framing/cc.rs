//! ChatCompletions framing — the canonical internal protocol, so this edge is near-identity.
//!
//! Every wire shape here is a vendored `dynamo_protocols` type (patch 0002 gives the response types
//! the serialization differences an emitter needs). Nothing here is restated locally.
//!
//! Server-tool activity has no native CC shape, so it rides `baseten.iterations` (see
//! `docs/architecture.md` "Protocol capability split"). `continuation_messages` carries the loop
//! transcript as CC request-messages a client appends, which Messages gets natively from its blocks.

use dynamo_protocols::types::{
    ChatChoice, ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallChunk, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionResponseMessage, ChatCompletionStreamResponseDelta, CompletionUsage,
    CreateChatCompletionResponse, CreateChatCompletionStreamResponse, FinishReason, FunctionCall,
    FunctionCallStream, FunctionType, ReasoningContent, Role,
};

use super::{
    BufferedResponse, CompletedIteration, ProtocolEnvelope, StagedIteration, StreamFraming,
    openai_error_sse_frame,
};
use crate::baseten_response_extension::{
    BasetenFrame, BasetenResponseExtension, IterationScope, ServerToolCallOutcome,
    ServerToolCallRecord,
};
use crate::model::{ErrorClass, ServerToolCall, Termination, ToolCall};
use crate::util::unix_secs;
use crate::wire::{next_id_seq, sse_frame, to_json_string};
use crate::{CcMessage, SemanticChunk};

const CHUNK_OBJECT: &str = "chat.completion.chunk";

/// Monomorphizes [`BasetenResponseExtension`] on CC's own usage shape — confined here since nothing outside
/// this module renders CC.
type CcExtension = BasetenResponseExtension<CompletionUsage>;

pub(super) struct CcEnvelope;

impl ProtocolEnvelope for CcEnvelope {
    fn stream_framing(self: Box<Self>, model: String) -> Box<dyn StreamFraming> {
        Box::new(CcFraming::new(model))
    }

    fn buffered_body(&self, response: &BufferedResponse<'_>) -> String {
        let created = created_now();
        // One `content` field, so every iteration's text concatenates: CC cannot separate them.
        let text = concatenated(response.transcript, assistant_text);
        let reasoning = concatenated(response.transcript, assistant_reasoning);
        let tool_calls: Vec<ChatCompletionMessageToolCall> = response
            .client_tool_calls
            .iter()
            .map(cc_tool_call)
            .collect();
        #[expect(
            deprecated,
            reason = "vendored struct literal must name `function_call`"
        )]
        let message = ChatCompletionResponseMessage {
            role: Role::Assistant,
            // `null`, not omitted, on a tool-only answer — the OpenAI shape.
            content: (!text.is_empty() || tool_calls.is_empty())
                .then_some(ChatCompletionMessageContent::Text(text)),
            reasoning_content: (!reasoning.is_empty()).then_some(reasoning),
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            refusal: None,
            function_call: None,
            audio: None,
        };
        let body = CreateChatCompletionResponse {
            id: cc_id(created),
            object: "chat.completion".to_string(),
            created,
            model: response.model.to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message,
                finish_reason: Some(cc_finish_reason(response.termination)),
                logprobs: None,
            }],
            usage: Some(response.usage.clone()),
            service_tier: None,
            system_fingerprint: None,
        };
        let extension = CcExtension {
            iterations: response.iterations.to_vec(),
            request: response
                .termination
                .request_scope(response.request_server_tool_calls),
        };
        to_json_string(&BasetenFrame {
            body: &body,
            baseten: (!extension.is_empty()).then_some(&extension),
        })
    }
}

/// The cap's closest fit is `length` — same contract for the client, an answer cut short with no tool
/// calls to run — where `tool_calls` would demand a round TB already ran and `stop` would claim a
/// complete answer.
fn cc_finish_reason(termination: Termination) -> FinishReason {
    match termination {
        Termination::Model(stop) => stop,
        Termination::ReactCapExhausted => FinishReason::Length,
    }
}

fn concatenated(transcript: &[CcMessage], pick: impl Fn(&CcMessage) -> Option<&str>) -> String {
    transcript.iter().filter_map(pick).collect()
}

fn assistant_text(message: &CcMessage) -> Option<&str> {
    match message {
        CcMessage::Assistant(assistant) => match assistant.content.as_ref()? {
            ChatCompletionRequestAssistantMessageContent::Text(text) => Some(text),
            // The loop only ever builds `Text`; an `Array` here would be a history-fold bug.
            ChatCompletionRequestAssistantMessageContent::Array(_) => None,
        },
        _ => None,
    }
}

fn assistant_reasoning(message: &CcMessage) -> Option<&str> {
    match message {
        CcMessage::Assistant(assistant) => match assistant.reasoning_content.as_ref()? {
            ReasoningContent::Text(reasoning) => Some(reasoning),
            ReasoningContent::Segments(_) => None,
        },
        _ => None,
    }
}

/// The model's verbatim argument bytes, never a reserialize — byte-exact cache prefix.
fn cc_tool_call(call: &ToolCall) -> ChatCompletionMessageToolCall {
    ChatCompletionMessageToolCall {
        id: call.id.clone(),
        r#type: FunctionType::Function,
        function: FunctionCall {
            name: call.name.clone(),
            arguments: call.raw_args.clone(),
        },
    }
}

/// Seconds since the epoch as the OpenAI `created` field types it. Wraps in 2106; matching the
/// schema's width beats widening it here.
fn created_now() -> u32 {
    unix_secs() as u32
}

/// Synthetic ChatCompletions response id (`chatcmpl-<created>-<seq>`). The client SDK surfaces it
/// but does not parse the format.
fn cc_id(created: u32) -> String {
    format!("chatcmpl-{created}-{}", next_id_seq())
}

struct CcFraming {
    id: String,
    created: u32,
    model: String,
    role_sent: bool,
    /// Monotonic across the response: client tools stream one delta each, so the SDK-accumulated
    /// `tool_calls[].index` can't be local to a single emit.
    next_tool_call_index: u32,
}

impl CcFraming {
    fn new(model: String) -> Self {
        let created = created_now();
        Self {
            id: cc_id(created),
            created,
            model,
            role_sent: false,
            next_tool_call_index: 0,
        }
    }

    /// `role` rides the response's first delta only; every later one omits it.
    fn take_role(&mut self) -> Option<Role> {
        (!std::mem::replace(&mut self.role_sent, true)).then_some(Role::Assistant)
    }

    fn empty_delta(&mut self) -> ChatCompletionStreamResponseDelta {
        ChatCompletionStreamResponseDelta {
            role: self.take_role(),
            content: None,
            reasoning_content: None,
            tool_calls: None,
            refusal: None,
            function_call: None,
        }
    }

    /// The terminal chunk's delta, which OpenAI leaves empty. `role` never rides it: a client that
    /// opens its assistant message on `delta.role` would open one at the very end.
    fn terminal_delta(&self) -> ChatCompletionStreamResponseDelta {
        ChatCompletionStreamResponseDelta {
            role: None,
            content: None,
            reasoning_content: None,
            tool_calls: None,
            refusal: None,
            function_call: None,
        }
    }

    fn delta_frame(&mut self, delta: ChatCompletionStreamResponseDelta) -> String {
        self.chunk_frame(
            vec![ChatChoiceStream {
                index: 0,
                delta,
                finish_reason: None,
                logprobs: None,
            }],
            None,
            None,
        )
    }

    /// One iteration scope on a frame of its own, which carries `choices: []` so no client SDK folds
    /// it into the message.
    fn iteration_frame(&self, iteration: IterationScope<CompletionUsage>) -> String {
        self.chunk_frame(
            Vec::new(),
            None,
            Some(&CcExtension {
                iterations: vec![iteration],
                request: None,
            }),
        )
    }

    fn chunk_frame(
        &self,
        choices: Vec<ChatChoiceStream>,
        usage: Option<&CompletionUsage>,
        baseten: Option<&CcExtension>,
    ) -> String {
        let chunk = CreateChatCompletionStreamResponse {
            id: self.id.clone(),
            object: CHUNK_OBJECT.to_string(),
            created: self.created,
            model: self.model.clone(),
            choices,
            usage: usage.cloned(),
            service_tier: None,
            system_fingerprint: None,
        };
        sse_frame(
            None,
            &to_json_string(&BasetenFrame {
                body: &chunk,
                baseten,
            }),
        )
    }
}

impl StreamFraming for CcFraming {
    fn on_chunk(&mut self, chunk: &SemanticChunk) -> Vec<String> {
        let (content, reasoning_content) = match chunk {
            SemanticChunk::TextDelta(text) => (Some(text.clone()), None),
            SemanticChunk::ThinkingDelta(thinking) => (None, Some(thinking.clone())),
            // Surfaced elsewhere: emit_completed_iteration / emit_client_tool_calls, emit_iteration_usage / finish.
            SemanticChunk::ToolCall(_) | SemanticChunk::Usage(_) | SemanticChunk::Stop { .. } => {
                return Vec::new();
            }
        };
        let delta = ChatCompletionStreamResponseDelta {
            content: content.map(ChatCompletionMessageContent::Text),
            reasoning_content,
            ..self.empty_delta()
        };
        vec![self.delta_frame(delta)]
    }

    /// One frame for the whole iteration, never one per call: a client appending
    /// `continuation_messages` in arrival order must never receive a call without its result.
    fn emit_completed_iteration(&mut self, iteration: &CompletedIteration<'_>) -> Vec<String> {
        vec![self.iteration_frame(IterationScope {
            server_tool_calls: iteration.server_tool_calls(),
            continuation_messages: iteration.continuation_messages.to_vec(),
            ..IterationScope::at(iteration.index)
        })]
    }

    /// Native `tool_calls`. Streamed deltas need an integer `index` per call (the SDK accumulates
    /// by it), monotonic across the separate emits the loop makes.
    fn emit_client_tool_calls(&mut self, calls: &[ToolCall]) -> Vec<String> {
        if calls.is_empty() {
            return Vec::new();
        }
        let indexed: Vec<ChatCompletionMessageToolCallChunk> = calls
            .iter()
            .map(|call| {
                let index = self.next_tool_call_index;
                self.next_tool_call_index += 1;
                ChatCompletionMessageToolCallChunk {
                    index,
                    id: Some(call.id.clone()),
                    r#type: Some(FunctionType::Function),
                    function: Some(FunctionCallStream {
                        name: Some(call.name.clone()),
                        arguments: Some(call.raw_args.clone()),
                    }),
                }
            })
            .collect();
        let delta = ChatCompletionStreamResponseDelta {
            tool_calls: Some(indexed),
            ..self.empty_delta()
        };
        vec![self.delta_frame(delta)]
    }

    fn emit_dispatched_server_tool(
        &mut self,
        iteration: u32,
        server_call: &ServerToolCall,
    ) -> Vec<String> {
        vec![self.iteration_frame(IterationScope {
            server_tool_calls: vec![ServerToolCallRecord::dispatched(server_call)],
            ..IterationScope::at(iteration)
        })]
    }

    fn stage_iteration(&mut self, staged: StagedIteration<'_>) -> Vec<String> {
        vec![self.iteration_frame(staged.scope(CompletionUsage::clone))]
    }

    fn finish(
        &mut self,
        termination: Termination,
        usage: &CompletionUsage,
        server_tool_calls: &[ServerToolCallOutcome],
    ) -> Vec<String> {
        let mut frames = Vec::new();
        // A response with no content at all (the ReAct cap can end one) still owes the client the
        // opening role delta OpenAI always sends, on its own chunk ahead of the terminal one.
        if !self.role_sent {
            let opening = self.empty_delta();
            frames.push(self.delta_frame(opening));
        }
        let extension = CcExtension {
            request: termination.request_scope(server_tool_calls),
            ..CcExtension::default()
        };
        frames.push(self.chunk_frame(
            vec![ChatChoiceStream {
                index: 0,
                delta: self.terminal_delta(),
                finish_reason: Some(cc_finish_reason(termination)),
                logprobs: None,
            }],
            Some(usage),
            (!extension.is_empty()).then_some(&extension),
        ));
        frames.push(sse_frame(None, "[DONE]"));
        frames
    }

    fn error_sse_frame(
        &mut self,
        class: ErrorClass,
        error_code: Option<&str>,
        message: &str,
    ) -> String {
        openai_error_sse_frame(class, error_code, message)
    }
}
