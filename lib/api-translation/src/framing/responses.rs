//! OpenAI Responses framing (stateless: no `store`/`previous_response_id`/`conversation`).
//!
//! `input`/`output` items give Responses a native transcript container Messages has and CC does
//! not (see `docs/architecture.md` "Protocol capability split"): a client that echoes `output` back as
//! its next `input` replays the whole loop transcript, tool calls and reasoning included, with no
//! `baseten` side channel needed for continuation.
//!
//! Item choice: a client-executed tool call is `function_call` (the real output-item enum has no
//! `function_call_output` — only the client, on its next request, answers one as an *input* item).
//! A server tool TB already resolved has no such open/closed pair to lean on, so it rides `mcp_call`
//! instead: the one output-item type OpenAI defines for "the platform ran this and already has a
//! result", matching a remote MCP tool's own same-turn call+result shape. TB parses its own wire
//! format, so no real MCP server or `Tool::Mcp` declaration is needed for this to round-trip.
//!
//! SSE carrier: verified against the installed `openai` Python SDK's stream accumulator
//! (`openai/lib/streaming/responses/_responses.py`). Every event type preserves an unknown
//! top-level field except `response.output_text.{delta,done}`, `response.function_call_arguments.delta`,
//! and `response.completed` — the last is rebuilt from `event.response` alone (nothing else copied
//! over), and the same rebuild is what a client's `get_final_response()` reads its whole output list
//! from — so the terminal event's `response.output` must carry every item this stream produced, not
//! an empty placeholder, and `baseten` rides inside that `response` body rather than top-level.
//! Every other event carries `baseten` top-level, same as CC/Messages.

use std::collections::HashMap;

use dynamo_protocols::types::responses::{
    B10ReasoningEffort, ErrorObject, FunctionToolCall, IncompleteDetails, InputTokenDetails,
    Instructions, MCPToolCall, MCPToolCallStatus, OutputContent, OutputItem, OutputMessage,
    OutputMessageContent, OutputStatus, OutputTextContent, OutputTokenDetails,
    PromptCacheRetention, Reasoning, ReasoningItem, ReasoningItemContent, ReasoningTextContent,
    Response, ResponseTextParam, ResponseUsage, ServiceTier, Status,
    TextResponseFormatConfiguration, Tool, ToolChoiceOptions, ToolChoiceParam, Truncation,
};
use dynamo_protocols::types::{CompletionUsage, FinishReason};
use serde_json::{Value, json};

use super::{
    BufferedResponse, CompletedIteration, OpenIterationScope, ProtocolEnvelope, StagedIteration,
    StreamFraming,
};
use crate::baseten_response_extension::{
    BasetenFrame, BasetenResponseExtension, IterationScope, ServerToolCallOutcome,
};
use crate::coding_adapter::{CodingAdapter, RenderedToolCall, ToolCallStatus, ToolCallToRender};
use crate::model::{
    BackendError, ErrorClass, ServerToolCall, Termination, ToolCall, ToolInvocation,
};
use crate::util::unix_secs;
use crate::wire::{next_id_seq, sse_frame, to_json_string, to_json_value};
use crate::{CcMessage, SemanticChunk};

/// Confined here: nothing outside this module renders Responses.
type ResponsesExtension = BasetenResponseExtension<ResponseUsage>;

/// The Responses request's echo-back params: what the terminal envelope repeats to the client, per
/// the OpenResponses schema (spec defaults applied at render time, mirroring the fork's
/// `make_response`). Built from the raw request body by [`ResponsesParams::from_body`]; the other
/// protocols pass `Default::default()`.
#[derive(Debug, Default, Clone)]
pub struct ResponsesParams {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_output_tokens: Option<u32>,
    pub parallel_tool_calls: Option<bool>,
    pub store: Option<bool>,
    /// Kept raw: a `baseten__*` selection entry is not a typed `Tool`; entries that don't parse as
    /// one are left out of the echo.
    pub tools: Vec<Value>,
    pub tool_choice: Option<ToolChoiceParam>,
    pub instructions: Option<String>,
    pub reasoning: Option<Reasoning>,
    pub text: Option<ResponseTextParam>,
    pub service_tier: Option<ServiceTier>,
    pub truncation: Option<Truncation>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub prompt_cache_key: Option<String>,
    pub prompt_cache_retention: Option<PromptCacheRetention>,
    pub safety_identifier: Option<String>,
    /// Echoed verbatim: the spec puts the request's `metadata` on the response body.
    pub metadata: Option<HashMap<String, String>>,
    /// Echoed as sent; the spec default (0) applies only when the request omitted it.
    pub top_logprobs: Option<u8>,
}

impl ResponsesParams {
    /// Lenient projection of the raw Responses request body: a field that doesn't parse is echoed
    /// as absent rather than failing the request — validation belongs to ingress adaptation, not
    /// the echo.
    pub fn from_body(body: &Value) -> Self {
        fn field<T: serde::de::DeserializeOwned>(body: &Value, name: &str) -> Option<T> {
            body.get(name)
                .cloned()
                .and_then(|value| serde_json::from_value(value).ok())
        }
        Self {
            temperature: field(body, "temperature"),
            top_p: field(body, "top_p"),
            max_output_tokens: field(body, "max_output_tokens"),
            parallel_tool_calls: field(body, "parallel_tool_calls"),
            store: field(body, "store"),
            tools: field(body, "tools").unwrap_or_default(),
            tool_choice: field(body, "tool_choice"),
            instructions: field(body, "instructions"),
            // Projected a field at a time, so one half cannot discard the other: an effort
            // the echo type cannot spell must not take `summary` with it, and `summary` is
            // what decides whether the reasoning item is emitted at all.
            reasoning: body
                .get("reasoning")
                .filter(|value| value.is_object())
                .map(|reasoning| Reasoning {
                    effort: field::<B10ReasoningEffort>(reasoning, "effort")
                        .as_ref()
                        .and_then(B10ReasoningEffort::to_async_openai),
                    summary: field(reasoning, "summary"),
                }),
            text: field(body, "text"),
            service_tier: field(body, "service_tier"),
            truncation: field(body, "truncation"),
            presence_penalty: field(body, "presence_penalty"),
            frequency_penalty: field(body, "frequency_penalty"),
            prompt_cache_key: field(body, "prompt_cache_key"),
            prompt_cache_retention: field(body, "prompt_cache_retention"),
            safety_identifier: field(body, "safety_identifier"),
            metadata: field(body, "metadata"),
            top_logprobs: field(body, "top_logprobs"),
        }
    }

    /// The typed tools the terminal envelope echoes, spec-normalized (`strict` defaults to true on
    /// function tools, matching the fork's `normalize_tools`).
    fn normalized_tools(&self) -> Vec<Tool> {
        self.tools
            .iter()
            .filter_map(|entry| serde_json::from_value::<Tool>(entry.clone()).ok())
            .map(|tool| match tool {
                Tool::Function(mut ft) => {
                    if ft.strict.is_none() {
                        ft.strict = Some(true);
                    }
                    Tool::Function(ft)
                }
                other => other,
            })
            .collect()
    }
}

pub(super) struct ResponsesEnvelope {
    coding_adapter: Option<Box<dyn CodingAdapter>>,
    params: ResponsesParams,
}

impl ResponsesEnvelope {
    pub(super) fn new(
        coding_adapter: Option<Box<dyn CodingAdapter>>,
        params: ResponsesParams,
    ) -> Self {
        Self {
            coding_adapter,
            params,
        }
    }
}

impl ProtocolEnvelope for ResponsesEnvelope {
    fn stream_framing(self: Box<Self>, model: String) -> Box<dyn StreamFraming> {
        Box::new(ResponsesFraming::new(
            model,
            self.coding_adapter,
            self.params,
        ))
    }

    fn buffered_body(&self, response: &BufferedResponse<'_>) -> String {
        let (status, incomplete_details) = response_status(response.termination);
        let (output, rendered_calls) = transcript_output(response, self.coding_adapter.as_deref());
        let body = make_response(
            &self.params,
            responses_id(),
            response.model.to_string(),
            output,
            status,
            incomplete_details,
            None,
            Some(responses_usage(response.usage)),
        );
        let extension = responses_extension(
            response.iterations,
            response.request_server_tool_calls,
            response.termination,
        );
        let mut body = finished_body(self.coding_adapter.as_deref(), body, &rendered_calls);
        if let Value::Object(object) = &mut body {
            patch_response_for_spec(object, &self.params);
        }
        to_json_string(&BasetenFrame {
            body: &body,
            baseten: (!extension.is_empty()).then_some(&extension),
        })
    }
}

fn responses_id() -> String {
    format!("resp_{}", next_id_seq())
}

/// Synthetic item ids carry OpenAI's kind prefixes (`msg_`/`rs_`): some agent tooling
/// pattern-matches item kinds by prefix. Function/mcp items reuse the model's own call id instead.
fn message_item_id() -> String {
    format!("msg_{}", next_id_seq())
}

fn reasoning_item_id() -> String {
    format!("rs_{}", next_id_seq())
}

/// Every SSE event type this framing emits, single-sourced: the wire literal exists once, and
/// [`Self::preserves_extra_fields`] stays exhaustive.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ResponsesEvent {
    Created,
    InProgress,
    OutputItemAdded,
    ContentPartAdded,
    ReasoningTextDelta,
    ReasoningTextDone,
    OutputTextDelta,
    OutputTextDone,
    ContentPartDone,
    OutputItemDone,
    FunctionCallArgumentsDelta,
    FunctionCallArgumentsDone,
    McpCallCompleted,
    McpCallFailed,
    Completed,
    Incomplete,
    Failed,
}

impl ResponsesEvent {
    fn as_str(self) -> &'static str {
        match self {
            Self::Created => "response.created",
            Self::InProgress => "response.in_progress",
            Self::OutputItemAdded => "response.output_item.added",
            Self::ContentPartAdded => "response.content_part.added",
            Self::ReasoningTextDelta => "response.reasoning_text.delta",
            Self::ReasoningTextDone => "response.reasoning_text.done",
            Self::OutputTextDelta => "response.output_text.delta",
            Self::OutputTextDone => "response.output_text.done",
            Self::ContentPartDone => "response.content_part.done",
            Self::OutputItemDone => "response.output_item.done",
            Self::FunctionCallArgumentsDelta => "response.function_call_arguments.delta",
            Self::FunctionCallArgumentsDone => "response.function_call_arguments.done",
            Self::McpCallCompleted => "response.mcp_call.completed",
            Self::McpCallFailed => "response.mcp_call.failed",
            Self::Completed => "response.completed",
            Self::Incomplete => "response.incomplete",
            Self::Failed => "response.failed",
        }
    }

    /// Whether the `openai` SDK's stream accumulator forwards unknown top-level fields on this
    /// event intact. It rebuilds these three from their own typed sub-fields with nothing else
    /// copied over — see the module doc comment.
    fn preserves_extra_fields(self) -> bool {
        !matches!(
            self,
            Self::OutputTextDelta | Self::OutputTextDone | Self::FunctionCallArgumentsDelta
        )
    }
}

/// The response envelope, echoing request params with the spec's defaults for omitted fields —
/// mirrors the fork's `make_response` (lib/llm responses/stream_converter.rs). `error` is set only
/// by the terminal-failure path.
#[allow(clippy::too_many_arguments)]
fn make_response(
    params: &ResponsesParams,
    id: String,
    model: String,
    output: Vec<OutputItem>,
    status: Status,
    incomplete_details: Option<IncompleteDetails>,
    error: Option<ErrorObject>,
    usage: Option<ResponseUsage>,
) -> Response {
    Response {
        background: Some(false),
        billing: None,
        conversation: None,
        created_at: unix_secs(),
        completed_at: matches!(status, Status::Completed).then(unix_secs),
        error,
        id,
        incomplete_details,
        instructions: params.instructions.clone().map(Instructions::Text),
        max_output_tokens: params.max_output_tokens,
        metadata: Some(params.metadata.clone().unwrap_or_default()),
        model,
        object: "response".to_string(),
        output,
        parallel_tool_calls: params.parallel_tool_calls.or(Some(true)),
        previous_response_id: None,
        prompt: None,
        prompt_cache_key: params.prompt_cache_key.clone(),
        prompt_cache_retention: params.prompt_cache_retention,
        reasoning: params.reasoning.clone(),
        safety_identifier: params.safety_identifier.clone(),
        service_tier: Some(params.service_tier.unwrap_or(ServiceTier::Auto)),
        status,
        temperature: params.temperature.or(Some(1.0)),
        text: Some(params.text.clone().unwrap_or(ResponseTextParam {
            format: TextResponseFormatConfiguration::Text,
            verbosity: None,
        })),
        tool_choice: params
            .tool_choice
            .clone()
            .or(Some(ToolChoiceParam::Mode(ToolChoiceOptions::Auto))),
        tools: Some(params.normalized_tools()),
        top_logprobs: Some(params.top_logprobs.unwrap_or(0)),
        top_p: params.top_p.or(Some(1.0)),
        truncation: Some(params.truncation.unwrap_or(Truncation::Disabled)),
        usage,
    }
}

/// Patch a serialized `Response` object to satisfy the OpenResponses schema — mirrors the fork's
/// `patch_response_for_spec` (lib/llm responses/mod.rs):
///  1. spec-nullable-required fields forced present-as-null,
///  2. `presence_penalty`/`frequency_penalty`/`store` injected (absent from the typed `Response`),
///  3. `usage.input_tokens_details.cache_write_tokens` defaulted to 0 — openai-python >= 2.53
///     types it as required, and the engine reports no cache writes.
fn patch_response_for_spec(object: &mut serde_json::Map<String, Value>, params: &ResponsesParams) {
    for key in dynamo_protocols::types::responses::SPEC_NULLABLE_REQUIRED_RESPONSE_FIELDS {
        object.entry(*key).or_insert(Value::Null);
    }
    object.insert(
        "presence_penalty".into(),
        json!(params.presence_penalty.unwrap_or(0.0)),
    );
    object.insert(
        "frequency_penalty".into(),
        json!(params.frequency_penalty.unwrap_or(0.0)),
    );
    object.insert("store".into(), json!(params.store.unwrap_or(false)));
    if let Some(Value::Object(usage)) = object.get_mut("usage")
        && let Some(Value::Object(details)) = usage.get_mut("input_tokens_details")
    {
        details.entry("cache_write_tokens").or_insert(json!(0));
    }
}

/// Model truncations map to the spec's own `incomplete_details.reason` literals. The cap's nearest
/// fit is also `incomplete`, but its reason is TB's own name: unlike CC's `finish_reason` /
/// Messages' `stop_reason`, `reason` is a free string on the wire (the SDK types it leniently), so
/// TB states the real cause instead of picking a spec value that means something else.
fn response_status(termination: Termination) -> (Status, Option<IncompleteDetails>) {
    let incomplete = |reason: &str| {
        (
            Status::Incomplete,
            Some(IncompleteDetails {
                reason: reason.to_string(),
            }),
        )
    };
    match termination {
        Termination::Model(FinishReason::Length) => incomplete("max_output_tokens"),
        Termination::Model(FinishReason::ContentFilter) => incomplete("content_filter"),
        // The react cap is TB's own budget, not a truncated model turn: `completed` (tool-bank's
        // decision) — Codex auto-retries any `incomplete`, re-running the whole loop into the same
        // deterministic cap. The cause is stated in `baseten.request.termination_reason`.
        Termination::Model(
            FinishReason::Stop | FinishReason::ToolCalls | FinishReason::FunctionCall,
        )
        | Termination::ReactCapExhausted => (Status::Completed, None),
    }
}

fn finished_body(
    adapter: Option<&dyn CodingAdapter>,
    body: impl serde::Serialize,
    rendered: &HashMap<String, Value>,
) -> Value {
    let mut body = to_json_value(&body);
    if let Some(adapter) = adapter {
        body = adapter.splice_rendered_calls(body, rendered);
    }
    body
}

/// The loop transcript as output items, in transcript order — the order the model itself saw, so
/// echoing this back as the next request's `input` reproduces its prefix. Mirrors `messages.rs`'s
/// `transcript_blocks`: every tool call the model made appears as `function_call`/`mcp_call`
/// regardless of who answers it; a call the loop never answers (the client's to run) stays a plain
/// `function_call`, and a server tool's `mcp_call` carries its result in the same item.
fn transcript_output(
    response: &BufferedResponse<'_>,
    coding_adapter: Option<&dyn CodingAdapter>,
) -> (Vec<OutputItem>, HashMap<String, Value>) {
    let failed_server_tool_call_ids = response.failed_server_tool_call_ids();
    let server_tool_providers = response.server_tool_providers();
    let transcript = response.transcript;
    let mut output = Vec::new();
    let mut open_call_slots: HashMap<String, usize> = HashMap::new();
    let mut rendered_calls = HashMap::new();
    for message in transcript {
        match message {
            CcMessage::Assistant(assistant) => {
                if let Some(dynamo_protocols::types::ReasoningContent::Text(reasoning)) =
                    &assistant.reasoning_content
                {
                    output.push(reasoning_item(
                        reasoning_item_id(),
                        reasoning,
                        OutputStatus::Completed,
                    ));
                }
                if let Some(
                    dynamo_protocols::types::ChatCompletionRequestAssistantMessageContent::Text(
                        text,
                    ),
                ) = &assistant.content
                {
                    output.push(message_item(
                        message_item_id(),
                        text,
                        OutputStatus::Completed,
                    ));
                }
                for call in assistant.tool_calls.iter().flatten() {
                    let rendered = coding_adapter.and_then(|adapter| {
                        render_with_slot(
                            adapter,
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
                        )
                    });
                    if let Some((rendered, slot)) = rendered {
                        output.push(slot);
                        rendered_calls.insert(call.id.clone(), rendered.item);
                        continue;
                    }
                    let (name, namespace) = call_identity(coding_adapter, &call.function.name);
                    output.push(function_call_item(
                        &call.id,
                        &name,
                        namespace.as_deref(),
                        &call.function.arguments,
                        OutputStatus::Completed,
                    ));
                    open_call_slots.insert(call.id.clone(), output.len() - 1);
                }
            }
            CcMessage::Tool(tool) => {
                let dynamo_protocols::types::ChatCompletionRequestToolMessageContent::Text(content) =
                    &tool.content
                else {
                    continue;
                };
                // Replace in place so item order still matches the model's.
                if let Some(&slot) = open_call_slots.get(&tool.tool_call_id) {
                    let OutputItem::FunctionCall(placeholder) = &output[slot] else {
                        debug_assert!(
                            false,
                            "slot for `{}` is not a function_call",
                            tool.tool_call_id
                        );
                        tracing::error!(
                            event_name = "responses.slot_not_function_call",
                            "Slot for `{}` is not a function_call",
                            tool.tool_call_id
                        );
                        continue;
                    };
                    // A tool result answering a dispatched server tool re-renders as an mcp_call;
                    // the provider label comes from the loop's own call records, never from
                    // parsing the tool's name (the crate knows no tool namespace).
                    let Some(&provider) = server_tool_providers.get(placeholder.name.as_str())
                    else {
                        continue;
                    };
                    let (provider, name, arguments) = (
                        provider.to_string(),
                        placeholder.name.clone(),
                        placeholder.arguments.clone(),
                    );
                    output[slot] = mcp_call_item(
                        &tool.tool_call_id,
                        &name,
                        &provider,
                        &arguments,
                        content,
                        failed_server_tool_call_ids.contains(tool.tool_call_id.as_str()),
                    );
                }
            }
            _ => {}
        }
    }
    (output, rendered_calls)
}

fn reasoning_item(id: String, text: &str, status: OutputStatus) -> OutputItem {
    OutputItem::Reasoning(ReasoningItem {
        id: Some(id),
        summary: Vec::new(),
        content: Some(vec![ReasoningItemContent::ReasoningText(
            ReasoningTextContent {
                text: text.to_string(),
            },
        )]),
        encrypted_content: None,
        status: Some(status),
    })
}

fn message_item(id: String, text: &str, status: OutputStatus) -> OutputItem {
    OutputItem::Message(OutputMessage {
        content: vec![OutputMessageContent::OutputText(OutputTextContent {
            annotations: Vec::new(),
            logprobs: Some(Vec::new()),
            text: text.to_string(),
        })],
        id,
        // `AssistantRole` is single-variant, so the default is provably `assistant`.
        role: Default::default(),
        phase: None,
        status,
    })
}

/// The wire name a call carried, split into the `(name, namespace)` the client dispatches on:
/// a namespaced tool resolves to its declaration, everything else passes as itself.
fn call_identity(
    coding_adapter: Option<&dyn CodingAdapter>,
    tool_name: &str,
) -> (String, Option<String>) {
    coding_adapter
        .and_then(|adapter| adapter.resolve_tool_identity(tool_name))
        .map_or_else(
            || (tool_name.to_string(), None),
            |identity| (identity.name, Some(identity.namespace)),
        )
}

/// The model's verbatim argument bytes, never a reserialize — byte-exact cache prefix, same
/// invariant `cc_tool_call` (in `cc.rs`) keeps for ChatCompletions. The namespace is the client's
/// own declaration for the tool, restamped so a registry keyed on `(name, namespace)` resolves.
fn function_call_item(
    id: &str,
    name: &str,
    namespace: Option<&str>,
    raw_args: &str,
    status: OutputStatus,
) -> OutputItem {
    OutputItem::FunctionCall(FunctionToolCall {
        arguments: raw_args.to_string(),
        call_id: id.to_string(),
        namespace: namespace.map(str::to_string),
        name: name.to_string(),
        id: Some(id.to_string()),
        status: Some(status),
    })
}

fn item_call_id(item: &OutputItem) -> Option<&str> {
    match item {
        OutputItem::FunctionCall(call) => Some(&call.call_id),
        OutputItem::McpCall(call) => Some(&call.id),
        OutputItem::WebSearchCall(call) => Some(&call.id),
        _ => None,
    }
}

fn mcp_call_item(
    id: &str,
    name: &str,
    provider: &str,
    raw_args: &str,
    output: &str,
    is_error: bool,
) -> OutputItem {
    OutputItem::McpCall(MCPToolCall {
        arguments: raw_args.to_string(),
        id: id.to_string(),
        name: name.to_string(),
        server_label: provider.to_string(),
        approval_request_id: None,
        error: is_error.then(|| output.to_string()),
        output: (!is_error).then(|| output.to_string()),
        status: Some(if is_error {
            MCPToolCallStatus::Failed
        } else {
            MCPToolCallStatus::Completed
        }),
    })
}

/// The slot is parsed back out of the client's own object, so the two cannot disagree, and a call
/// that cannot be recorded is declined here rather than half-emitted.
fn render_with_slot(
    adapter: &dyn CodingAdapter,
    call: &ToolCallToRender<'_>,
) -> Option<(RenderedToolCall, OutputItem)> {
    let rendered = adapter.render_tool_call(call)?;
    serde_json::from_value(rendered.item.clone())
        .inspect_err(|error| {
            tracing::error!(
                event_name = "responses.render_invalid_item",
                "Rendered tool call `{}` is not a Responses output item: {error}",
                call.id
            );
        })
        .ok()
        .map(|slot| (rendered, slot))
}

fn responses_usage(usage: &CompletionUsage) -> ResponseUsage {
    ResponseUsage {
        input_tokens: usage.prompt_tokens,
        input_tokens_details: InputTokenDetails {
            cached_tokens: usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|details| details.cached_tokens)
                .unwrap_or(0),
        },
        output_tokens: usage.completion_tokens,
        output_tokens_details: OutputTokenDetails {
            reasoning_tokens: usage
                .completion_tokens_details
                .as_ref()
                .and_then(|details| details.reasoning_tokens)
                .unwrap_or(0),
        },
        total_tokens: usage.total_tokens,
    }
}

fn responses_extension(
    iterations: &[IterationScope<CompletionUsage>],
    request_server_tool_calls: &[ServerToolCallOutcome],
    termination: Termination,
) -> ResponsesExtension {
    ResponsesExtension {
        iterations: iterations
            .iter()
            .cloned()
            .map(|iteration| iteration.into_usage_only(responses_usage))
            .collect(),
        request: termination.request_scope(request_server_tool_calls),
    }
}

/// Which output item is mid-stream, with the text accumulated so far (needed on close: the
/// item-level `.done` events re-send the whole item, and the accumulator drops extras from
/// `output_text.done` so `baseten` can never ride there anyway). Function/mcp calls open and close
/// inline within one `on_chunk`/`emit_completed_iteration` call, so they never appear here.
struct OpenContentItem {
    kind: OpenContentKind,
    output_index: u32,
    item_id: String,
    text: String,
}

/// Reasoning vs. text is the only per-kind variation in the open/delta/close cycle; everything
/// else (`ResponsesEvent`s, part and item constructors) dispatches off this.
#[derive(Clone, Copy, PartialEq, Eq)]
enum OpenContentKind {
    Reasoning,
    Text,
}

impl OpenContentKind {
    fn item(self, item_id: String, text: &str, status: OutputStatus) -> OutputItem {
        match self {
            Self::Reasoning => reasoning_item(item_id, text, status),
            Self::Text => message_item(item_id, text, status),
        }
    }

    fn part(self, text: &str) -> OutputContent {
        match self {
            Self::Reasoning => OutputContent::ReasoningText(ReasoningTextContent {
                text: text.to_string(),
            }),
            Self::Text => OutputContent::OutputText(OutputTextContent {
                annotations: Vec::new(),
                logprobs: Some(Vec::new()),
                text: text.to_string(),
            }),
        }
    }

    fn mint_item_id(self) -> String {
        match self {
            Self::Reasoning => reasoning_item_id(),
            Self::Text => message_item_id(),
        }
    }

    fn delta_event(self) -> ResponsesEvent {
        match self {
            Self::Reasoning => ResponsesEvent::ReasoningTextDelta,
            Self::Text => ResponsesEvent::OutputTextDelta,
        }
    }

    fn text_done_event(self) -> ResponsesEvent {
        match self {
            Self::Reasoning => ResponsesEvent::ReasoningTextDone,
            Self::Text => ResponsesEvent::OutputTextDone,
        }
    }
}

struct ResponsesFraming {
    model: String,
    response_id: String,
    started: bool,
    next_sequence_number: u64,
    next_output_index: u32,
    open_content_item: Option<OpenContentItem>,
    /// Every item completed so far, in order — the terminal envelope event's `response.output`
    /// needs the full list (a client reading only `response.completed` builds its final object
    /// from that field alone, not from the deltas it may have skipped).
    output: Vec<OutputItem>,
    /// A function call's `output_index`, from dispatch until its iteration completes (server tool)
    /// or the response ends (client tool, never looked up again).
    open_calls: HashMap<String, u32>,
    pending_baseten: ResponsesExtension,
    open_iteration_scope: OpenIterationScope<ResponseUsage>,
    coding_adapter: Option<Box<dyn CodingAdapter>>,
    rendered_calls: HashMap<String, Value>,
    params: ResponsesParams,
    /// The last usage the stream carried, for the terminal-failure envelope (the regular `finish`
    /// receives the loop's cumulative usage instead).
    usage: Option<CompletionUsage>,
}

impl ResponsesFraming {
    fn new(
        model: String,
        coding_adapter: Option<Box<dyn CodingAdapter>>,
        params: ResponsesParams,
    ) -> Self {
        Self {
            model,
            response_id: responses_id(),
            started: false,
            next_sequence_number: 0,
            next_output_index: 0,
            open_content_item: None,
            output: Vec::new(),
            open_calls: HashMap::new(),
            pending_baseten: ResponsesExtension::default(),
            open_iteration_scope: OpenIterationScope::default(),
            coding_adapter,
            rendered_calls: HashMap::new(),
            params,
            usage: None,
        }
    }

    fn make_response(
        &self,
        output: Vec<OutputItem>,
        status: Status,
        incomplete_details: Option<IncompleteDetails>,
        error: Option<ErrorObject>,
        usage: Option<ResponseUsage>,
    ) -> Response {
        make_response(
            &self.params,
            self.response_id.clone(),
            self.model.clone(),
            output,
            status,
            incomplete_details,
            error,
            usage,
        )
    }

    fn take_sequence_number(&mut self) -> u64 {
        let sequence_number = self.next_sequence_number;
        self.next_sequence_number += 1;
        sequence_number
    }

    fn ensure_started(&mut self, frames: &mut Vec<String>) {
        if self.started {
            return;
        }
        self.started = true;
        let response = self.make_response(Vec::new(), Status::InProgress, None, None, None);
        frames.push(self.envelope_frame(ResponsesEvent::Created, response.clone(), None));
        frames.push(self.envelope_frame(ResponsesEvent::InProgress, response, None));
    }

    /// One top-level (non-envelope) frame: `baseten` rides as a sibling of `type`, flushing whatever
    /// is pending — every event type preserves it except the three named in the module doc comment,
    /// none of which TB ever attaches extension data to.
    fn event_frame(&mut self, event: ResponsesEvent, body: serde_json::Value) -> String {
        self.named_event_frame(event.as_str(), event.preserves_extra_fields(), body)
    }

    fn named_event_frame(
        &mut self,
        event: &str,
        preserves_extra_fields: bool,
        mut body: serde_json::Value,
    ) -> String {
        if preserves_extra_fields && !self.pending_baseten.is_empty() {
            let baseten = std::mem::take(&mut self.pending_baseten);
            body["baseten"] = to_json_value(&baseten);
        }
        body["type"] = event.into();
        body["sequence_number"] = self.take_sequence_number().into();
        sse_frame(Some(event), &to_json_string(&body))
    }

    /// The one envelope shape (`response.created`/`.in_progress`/`.completed`/`.incomplete`), all
    /// `{type, sequence_number, response}`. `baseten` rides inside `response`, not top-level: the
    /// accumulator rebuilds `response.completed` from `event.response` alone, dropping any
    /// top-level sibling.
    fn envelope_frame(
        &mut self,
        event: ResponsesEvent,
        response: impl serde::Serialize,
        baseten: Option<ResponsesExtension>,
    ) -> String {
        let baseten = baseten.unwrap_or_default();
        // Every embedded `response` object is spec-patched, mirroring the fork's `make_sse_event`.
        let mut response = to_json_value(&response);
        if let Value::Object(object) = &mut response {
            patch_response_for_spec(object, &self.params);
        }
        sse_frame(
            Some(event.as_str()),
            &to_json_string(&serde_json::json!({
                "type": event.as_str(),
                "sequence_number": self.take_sequence_number(),
                "response": BasetenFrame {
                    body: &response,
                    baseten: (!baseten.is_empty()).then_some(&baseten),
                },
            })),
        )
    }

    /// The item's id is the caller's to mint — the same id must appear on every event of its lifecycle.
    fn open_item(&mut self, frames: &mut Vec<String>, item: &impl serde::Serialize) -> u32 {
        let output_index = self.next_output_index;
        self.next_output_index += 1;
        frames.push(self.event_frame(
            ResponsesEvent::OutputItemAdded,
            serde_json::json!({"output_index": output_index, "item": item}),
        ));
        output_index
    }

    fn rendered_call(
        &self,
        call: &ToolCall,
        status: ToolCallStatus,
    ) -> Option<(RenderedToolCall, OutputItem)> {
        render_with_slot(
            self.coding_adapter.as_deref()?,
            &ToolCallToRender {
                tool_name: &call.name,
                id: &call.id,
                args: &call.raw_args,
                status,
            },
        )
    }

    fn emit_lifecycle(
        &mut self,
        frames: &mut Vec<String>,
        output_index: u32,
        id: &str,
        events: &[&'static str],
    ) {
        for event in events {
            frames.push(self.named_event_frame(
                event,
                true,
                json!({"output_index": output_index, "item_id": id}),
            ));
        }
    }

    fn emit_item_done(
        &mut self,
        frames: &mut Vec<String>,
        output_index: u32,
        item: &impl serde::Serialize,
    ) {
        frames.push(self.event_frame(
            ResponsesEvent::OutputItemDone,
            serde_json::json!({"output_index": output_index, "item": item}),
        ));
    }

    /// A brand-new item's `output_item.done`: emit it, then record it (feeds the terminal
    /// envelope's `response.output`). A server tool's later resolution replaces this recorded copy
    /// in place via [`Self::replace_recorded_item_done`] rather than appending a second one.
    fn record_item_done(&mut self, frames: &mut Vec<String>, output_index: u32, item: OutputItem) {
        self.emit_item_done(frames, output_index, &item);
        self.output.push(item);
    }

    /// A previously recorded item resolving to its final shape (the `function_call` placeholder
    /// becoming an `mcp_call`): emit the new `output_item.done` a live client sees, and replace the
    /// recorded copy so the terminal envelope's `response.output` doesn't carry both.
    fn replace_recorded_item_done(
        &mut self,
        frames: &mut Vec<String>,
        output_index: u32,
        item: OutputItem,
    ) {
        self.emit_item_done(frames, output_index, &item);
        self.replace_recorded_item(item);
    }

    fn replace_recorded_item(&mut self, item: OutputItem) {
        let call_id = item_call_id(&item).map(str::to_string);
        let slot = call_id.as_deref().and_then(|call_id| {
            self.output
                .iter_mut()
                .find(|recorded| item_call_id(recorded) == Some(call_id))
        });
        match slot {
            Some(slot) => *slot = item,
            None => {
                debug_assert!(
                    false,
                    "resolved item {call_id:?} has no recorded placeholder"
                );
                tracing::error!(
                    event_name = "responses.placeholder_missing",
                    "Resolved item {call_id:?} has no recorded placeholder"
                );
                self.output.push(item);
            }
        }
    }

    fn close_open(&mut self, frames: &mut Vec<String>) {
        self.close_open_with(frames, OutputStatus::Completed);
    }

    /// The backend-error paths close an in-flight item as `incomplete`/`in_progress`-terminal per
    /// the fork's `collect_output`; the regular finish keeps `completed`.
    fn close_open_with(&mut self, frames: &mut Vec<String>, status: OutputStatus) {
        let Some(OpenContentItem {
            kind,
            output_index,
            item_id,
            text,
        }) = self.open_content_item.take()
        else {
            return;
        };
        let mut text_done_body = serde_json::json!({
            "output_index": output_index, "item_id": item_id, "content_index": 0, "text": text,
        });
        if kind == OpenContentKind::Text {
            text_done_body["logprobs"] = serde_json::json!([]);
        }
        frames.push(self.event_frame(kind.text_done_event(), text_done_body));
        frames.push(self.event_frame(
            ResponsesEvent::ContentPartDone,
            serde_json::json!({"output_index": output_index, "item_id": item_id, "content_index": 0, "part": kind.part(&text)}),
        ));
        self.record_item_done(frames, output_index, kind.item(item_id, &text, status));
    }

    fn content_delta(&mut self, kind: OpenContentKind, delta: &str, frames: &mut Vec<String>) {
        self.ensure_started(frames);
        if self.open_content_item.as_ref().map(|open| open.kind) != Some(kind) {
            self.close_open(frames);
            let item_id = kind.mint_item_id();
            let output_index = self.open_item(
                frames,
                &kind.item(item_id.clone(), "", OutputStatus::InProgress),
            );
            frames.push(self.event_frame(
                ResponsesEvent::ContentPartAdded,
                serde_json::json!({"output_index": output_index, "item_id": item_id, "content_index": 0, "part": kind.part("")}),
            ));
            self.open_content_item = Some(OpenContentItem {
                kind,
                output_index,
                item_id,
                text: String::new(),
            });
        }
        let open_content_item = self.open_content_item.as_mut().expect("opened above");
        open_content_item.text.push_str(delta);
        let (output_index, item_id) = (
            open_content_item.output_index,
            open_content_item.item_id.clone(),
        );
        let mut delta_body = serde_json::json!({"output_index": output_index, "item_id": item_id, "content_index": 0, "delta": delta});
        // The SDK's typed `output_text.delta` event declares `logprobs` (reasoning deltas don't).
        if kind == OpenContentKind::Text {
            delta_body["logprobs"] = serde_json::json!([]);
        }
        frames.push(self.event_frame(kind.delta_event(), delta_body));
    }
}

impl StreamFraming for ResponsesFraming {
    fn on_chunk(&mut self, chunk: &SemanticChunk) -> Vec<String> {
        let mut frames = Vec::new();
        match chunk {
            SemanticChunk::ThinkingDelta(delta) => {
                self.content_delta(OpenContentKind::Reasoning, delta, &mut frames);
            }
            SemanticChunk::TextDelta(delta) => {
                self.content_delta(OpenContentKind::Text, delta, &mut frames);
            }
            SemanticChunk::ToolCall(call) => {
                self.ensure_started(&mut frames);
                self.close_open(&mut frames);
                if let Some((rendered, slot)) = self.rendered_call(call, ToolCallStatus::InProgress)
                {
                    let output_index = self.open_item(&mut frames, &rendered.item);
                    self.emit_lifecycle(
                        &mut frames,
                        output_index,
                        &call.id,
                        &rendered.lifecycle_events,
                    );
                    self.output.push(slot);
                    self.rendered_calls.insert(call.id.clone(), rendered.item);
                    self.open_calls.insert(call.id.clone(), output_index);
                    return frames;
                }
                // A server tool's result later replaces this via `emit_completed_iteration`;
                // the item id is `call.id` on every event.
                let (name, namespace) = call_identity(self.coding_adapter.as_deref(), &call.name);
                let output_index = self.open_item(
                    &mut frames,
                    &function_call_item(
                        &call.id,
                        &name,
                        namespace.as_deref(),
                        "",
                        OutputStatus::InProgress,
                    ),
                );
                frames.push(self.event_frame(
                    ResponsesEvent::FunctionCallArgumentsDelta,
                    serde_json::json!({"output_index": output_index, "item_id": call.id, "delta": call.raw_args}),
                ));
                frames.push(self.event_frame(
                    ResponsesEvent::FunctionCallArgumentsDone,
                    serde_json::json!({"output_index": output_index, "item_id": call.id, "name": &name, "arguments": call.raw_args}),
                ));
                self.record_item_done(
                    &mut frames,
                    output_index,
                    function_call_item(
                        &call.id,
                        &name,
                        namespace.as_deref(),
                        &call.raw_args,
                        OutputStatus::Completed,
                    ),
                );
                self.open_calls.insert(call.id.clone(), output_index);
            }
            // Buffered for the terminal-failure envelope only; the regular finish gets the loop's
            // cumulative usage as an argument.
            SemanticChunk::Usage(usage) => self.usage = Some(usage.clone()),
            SemanticChunk::Stop { .. } => {}
        }
        frames
    }

    /// Server-tool results only: resolves the recorded `function_call` placeholder into its
    /// `mcp_call` at the same `output_index`.
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
            let Some(output_index) = self.open_calls.remove(&call.id) else {
                debug_assert!(false, "result for undispatched call `{}`", call.id);
                tracing::error!(
                    event_name = "responses.undispatched_result",
                    "Result for undispatched call `{}`",
                    call.id
                );
                continue;
            };
            let resolved = if output.is_error() {
                ToolCallStatus::Failed
            } else {
                ToolCallStatus::Completed
            };
            if let Some((rendered, slot)) = self.rendered_call(call, resolved) {
                self.emit_lifecycle(
                    &mut frames,
                    output_index,
                    &call.id,
                    &rendered.lifecycle_events,
                );
                self.emit_item_done(&mut frames, output_index, &rendered.item);
                self.replace_recorded_item(slot);
                self.rendered_calls.insert(call.id.clone(), rendered.item);
                continue;
            }
            let text = output.text();
            frames.push(self.event_frame(
                if output.is_error() {
                    ResponsesEvent::McpCallFailed
                } else {
                    ResponsesEvent::McpCallCompleted
                },
                serde_json::json!({"output_index": output_index, "item_id": call.id}),
            ));
            let item = mcp_call_item(
                &call.id,
                &call.name,
                &server_call.provider,
                &call.raw_args,
                &text,
                output.is_error(),
            );
            self.replace_recorded_item_done(&mut frames, output_index, item);
        }
        frames
    }

    fn emit_client_tool_calls(&mut self, _calls: &[ToolCall]) -> Vec<String> {
        // Already streamed as `function_call` items by `on_chunk`.
        Vec::new()
    }

    fn emit_dispatched_server_tool(
        &mut self,
        _iteration: u32,
        _server_call: &ServerToolCall,
    ) -> Vec<String> {
        // Ditto: the native `function_call` placeholder already went out, same id and arguments.
        Vec::new()
    }

    fn stage_iteration(&mut self, staged: StagedIteration<'_>) -> Vec<String> {
        if !self.started {
            tracing::warn!(
                event_name = "framing.iteration_scope_dropped",
                "iteration scope dropped before response.created (call cadence changed?)"
            );
            return Vec::new();
        }
        self.pending_baseten.iterations.extend(
            self.open_iteration_scope
                .stage(staged.scope(responses_usage)),
        );
        Vec::new()
    }

    fn close_iteration(&mut self) {
        self.pending_baseten
            .iterations
            .extend(self.open_iteration_scope.close());
    }

    fn finish(
        &mut self,
        termination: Termination,
        usage: &CompletionUsage,
        server_tool_calls: &[ServerToolCallOutcome],
    ) -> Vec<String> {
        let mut frames = Vec::new();
        self.ensure_started(&mut frames);
        self.close_open(&mut frames);
        self.pending_baseten.merge(ResponsesExtension {
            request: termination.request_scope(server_tool_calls),
            ..ResponsesExtension::default()
        });
        let (status, incomplete_details) = response_status(termination);
        let terminal_event = match status {
            Status::Completed => ResponsesEvent::Completed,
            Status::Incomplete => ResponsesEvent::Incomplete,
            Status::Failed | Status::InProgress | Status::Cancelled | Status::Queued => {
                unreachable!("response_status never returns this status")
            }
        };
        let output = std::mem::take(&mut self.output);
        let response = self.make_response(
            output,
            status,
            incomplete_details,
            None,
            Some(responses_usage(usage)),
        );
        self.pending_baseten
            .iterations
            .extend(self.open_iteration_scope.close());
        let baseten = std::mem::take(&mut self.pending_baseten);
        let response = finished_body(
            self.coding_adapter.as_deref(),
            response,
            &self.rendered_calls,
        );
        frames.push(self.envelope_frame(terminal_event, response, Some(baseten)));
        frames
    }

    /// Terminal backend failure, mirroring the fork's `emit_error_events`
    /// (lib/llm responses/stream_converter.rs):
    /// - a truncation-shaped error ("cutoff by max_tokens") is a `length` finish the worker
    ///   mispresented as a failure — re-present the spec-correct `response.incomplete`
    ///   (`incomplete_details.reason: "max_output_tokens"`) so clients keep the partial output;
    /// - genuine failures emit `response.failed` carrying whatever output streamed, with a typed
    ///   `error.code` string: 429 -> `rate_limit_exceeded`, prompt-overflow phrasings ->
    ///   `context_length_exceeded` (the exact string codex classifies on), else `server_error`.
    fn finish_with_backend_error(&mut self, error: &BackendError) -> Vec<String> {
        if error.is_truncation() {
            let mut frames = Vec::new();
            self.ensure_started(&mut frames);
            self.close_open_with(&mut frames, OutputStatus::Incomplete);
            let usage = self.usage.take().unwrap_or_default();
            frames.extend(self.finish(Termination::Model(FinishReason::Length), &usage, &[]));
            return frames;
        }
        let mut frames = Vec::new();
        self.ensure_started(&mut frames);
        self.close_open_with(&mut frames, OutputStatus::Incomplete);
        let error_object = ErrorObject {
            code: if error.http_status == Some(429) {
                "rate_limit_exceeded".to_string()
            } else if error.is_context_overflow() {
                "context_length_exceeded".to_string()
            } else {
                "server_error".to_string()
            },
            message: error.message.clone(),
        };
        let output = std::mem::take(&mut self.output);
        let usage = self.usage.take().map(|usage| responses_usage(&usage));
        let response = self.make_response(output, Status::Failed, None, Some(error_object), usage);
        self.pending_baseten
            .iterations
            .extend(self.open_iteration_scope.close());
        let baseten = std::mem::take(&mut self.pending_baseten);
        let response = finished_body(
            self.coding_adapter.as_deref(),
            response,
            &self.rendered_calls,
        );
        frames.push(self.envelope_frame(ResponsesEvent::Failed, response, Some(baseten)));
        frames
    }

    /// `response.failed` — the terminal envelope SDK accumulators and Codex read failure from
    /// (`get_final_response` / `incomplete_details` parsing), where a bare `error` event would make
    /// the `openai` SDK raise mid-iteration and drop the accumulated output. `_class` has no slot:
    /// the Responses error object is `{code, message}` only.
    fn error_sse_frame(
        &mut self,
        _class: ErrorClass,
        error_code: Option<&str>,
        message: &str,
    ) -> String {
        // One concatenated SSE payload: the trait returns a single String; frames concatenate legally.
        let mut frames = Vec::new();
        self.ensure_started(&mut frames);
        self.close_open(&mut frames);
        // `error_code` is total on projected errors; "error" is a never-expected backstop.
        debug_assert!(error_code.is_some(), "projected error without error_code");
        let error_object = ErrorObject {
            code: error_code.unwrap_or("error").to_string(),
            message: message.to_string(),
        };
        let output = std::mem::take(&mut self.output);
        let response = self.make_response(output, Status::Failed, None, Some(error_object), None);
        let baseten = std::mem::take(&mut self.pending_baseten);
        let response = finished_body(
            self.coding_adapter.as_deref(),
            response,
            &self.rendered_calls,
        );
        frames.push(self.envelope_frame(ResponsesEvent::Failed, response, Some(baseten)));
        frames.concat()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn react_cap_completes_rather_than_incomplete() {
        let (status, details) = response_status(Termination::ReactCapExhausted);
        assert_eq!(status, Status::Completed);
        assert!(details.is_none());
    }

    /// A length-truncated model turn is `incomplete` with the spec literal, not `completed`.
    #[test]
    fn length_termination_is_incomplete_with_spec_reason() {
        let (status, details) = response_status(Termination::Model(FinishReason::Length));
        assert_eq!(status, Status::Incomplete);
        assert_eq!(details.unwrap().reason, "max_output_tokens");
        let (status, _) = response_status(Termination::Model(FinishReason::Stop));
        assert_eq!(status, Status::Completed);
    }

    #[test]
    fn responses_usage_maps_cache_reads_into_input_token_details() {
        let usage = CompletionUsage {
            prompt_tokens: 1000,
            completion_tokens: 50,
            total_tokens: 1050,
            prompt_tokens_details: Some(dynamo_protocols::types::PromptTokensDetails {
                cached_tokens: Some(900),
                audio_tokens: None,
            }),
            completion_tokens_details: None,
        };
        let mapped = responses_usage(&usage);
        assert_eq!(mapped.input_tokens, 1000);
        assert_eq!(mapped.input_tokens_details.cached_tokens, 900);
        assert_eq!(mapped.output_tokens_details.reasoning_tokens, 0);
    }

    fn frame_body(frame: &str) -> serde_json::Value {
        let (event, data) = frame.split_once('\n').unwrap();
        let name = event.strip_prefix("event: ").unwrap();
        let body: serde_json::Value =
            serde_json::from_str(data.strip_prefix("data: ").unwrap().trim_end()).unwrap();
        assert_eq!(body["type"], name);
        body
    }

    /// Resolves one hoisted name back to its `(name, namespace)` declaration; every other call
    /// passes through as itself (default `resolve_tool_identity`).
    struct NamespaceStampAdapter;
    impl CodingAdapter for NamespaceStampAdapter {
        fn render_tool_call(&self, _call: &ToolCallToRender<'_>) -> Option<RenderedToolCall> {
            None
        }
        fn resolve_tool_identity(
            &self,
            tool_name: &str,
        ) -> Option<crate::coding_adapter::ToolIdentity> {
            (tool_name == "multi_agent_v1__spawn_agent").then(|| {
                crate::coding_adapter::ToolIdentity {
                    name: "spawn_agent".to_string(),
                    namespace: "multi_agent_v1".to_string(),
                }
            })
        }
    }

    /// A namespaced tool's streamed call is stamped back to the client's `(name, namespace)`
    /// registry key, not the hoisted wire name codex dispatched it under.
    #[test]
    fn responses_stamps_a_namespaced_call_with_its_declaration() {
        let mut framing = ResponsesFraming::new(
            "m".to_string(),
            Some(Box::new(NamespaceStampAdapter)),
            ResponsesParams::default(),
        );
        let frames = framing.on_chunk(&SemanticChunk::ToolCall(crate::model::ToolCall {
            id: "call_1".to_string(),
            name: "multi_agent_v1__spawn_agent".to_string(),
            args: json!({}),
            raw_args: "{}".to_string(),
        }));
        let added = frames
            .iter()
            .map(|frame| frame_body(frame))
            .find(|body| body["type"] == "response.output_item.added")
            .expect("an output_item.added frame");
        assert_eq!(added["item"]["type"], "function_call");
        assert_eq!(added["item"]["name"], "spawn_agent");
        assert_eq!(added["item"]["namespace"], "multi_agent_v1");
    }

    #[test]
    fn reasoning_items_carry_the_reasoning_text_discriminator() {
        let mut framing = ResponsesFraming::new("m".to_string(), None, ResponsesParams::default());
        let frames = framing.on_chunk(&SemanticChunk::ThinkingDelta("thinking".into()));
        let added = frames
            .iter()
            .map(|frame| frame_body(frame))
            .find(|body| body["type"] == "response.output_item.added")
            .expect("an output_item.added frame");
        assert_eq!(added["item"]["type"], "reasoning");
        assert_eq!(added["item"]["content"][0]["type"], "reasoning_text");
    }

    /// The other applier: `finished_body` tags the terminal envelope's `response.output`, which a
    /// client parses on replay even when it saw every streaming frame.
    #[test]
    fn terminal_envelope_reasoning_carries_the_reasoning_text_discriminator() {
        let mut framing = ResponsesFraming::new("m".to_string(), None, ResponsesParams::default());
        let mut frames = framing.on_chunk(&SemanticChunk::ThinkingDelta("thinking".into()));
        frames.extend(framing.finish(
            Termination::Model(dynamo_protocols::types::FinishReason::Stop),
            &CompletionUsage::default(),
            &[],
        ));
        let terminal = frames
            .iter()
            .map(|frame| frame_body(frame))
            .find(|body| body["type"] == "response.completed")
            .expect("a response.completed frame");
        let reasoning = terminal["response"]["output"]
            .as_array()
            .expect("output items")
            .iter()
            .find(|item| item["type"] == "reasoning")
            .expect("a reasoning output item");
        assert_eq!(reasoning["content"][0]["type"], "reasoning_text");
        assert_eq!(reasoning["content"][0]["text"], "thinking");
    }

    /// One item id per item, on every event of its lifecycle — a client matching deltas to items
    /// by `item_id` (not `output_index`) must see them attach.
    #[test]
    fn item_id_is_stable_across_added_deltas_done_and_envelope() {
        let mut framing = ResponsesFraming::new("m".to_string(), None, ResponsesParams::default());
        let mut frames = Vec::new();
        frames.extend(framing.on_chunk(&SemanticChunk::TextDelta("hi".to_string())));
        frames.extend(framing.finish(
            Termination::Model(dynamo_protocols::types::FinishReason::Stop),
            &CompletionUsage::default(),
            &[],
        ));
        let bodies: Vec<serde_json::Value> = frames.iter().map(|f| frame_body(f)).collect();
        let id_of = |event: &str, field: &str| {
            let body = bodies
                .iter()
                .find(|b| b["type"] == event)
                .unwrap_or_else(|| panic!("no {event}"));
            match field {
                "item" => body["item"]["id"].as_str().unwrap().to_string(),
                _ => body[field].as_str().unwrap().to_string(),
            }
        };
        let added_id = id_of("response.output_item.added", "item");
        assert_eq!(id_of("response.content_part.added", "item_id"), added_id);
        assert_eq!(id_of("response.output_text.delta", "item_id"), added_id);
        assert_eq!(id_of("response.output_text.done", "item_id"), added_id);
        assert_eq!(id_of("response.output_item.done", "item"), added_id);
        let envelope = bodies.last().unwrap();
        assert_eq!(envelope["type"], "response.completed");
        assert_eq!(
            envelope["response"]["output"][0]["id"].as_str().unwrap(),
            added_id
        );
        // The streaming placeholder is in progress; only the close marks it completed.
        let added = bodies
            .iter()
            .find(|b| b["type"] == "response.output_item.added")
            .unwrap();
        assert_eq!(added["item"]["status"], "in_progress");
        let done = bodies
            .iter()
            .find(|b| b["type"] == "response.output_item.done")
            .unwrap();
        assert_eq!(done["item"]["status"], "completed");
    }

    /// Continues `sequence_number` from the frames already sent (shape rationale on the impl).
    #[test]
    fn error_sse_frame_is_a_response_failed_envelope_with_a_live_sequence_number() {
        let mut framing = ResponsesFraming::new("m".to_string(), None, ResponsesParams::default());
        let frames_before_error = framing.on_chunk(&SemanticChunk::TextDelta("hi".into()));
        let error_payload = framing.error_sse_frame(
            ErrorClass::Internal,
            Some("model_streamed_error"),
            "upstream unavailable",
        );
        let terminal = error_payload
            .split("\n\n")
            .filter(|frame| !frame.is_empty())
            .last()
            .unwrap()
            .to_string()
            + "\n\n";
        assert!(
            terminal.starts_with("event: response.failed\n"),
            "{terminal}"
        );
        let body = frame_body(&terminal);
        assert_eq!(body["type"], "response.failed");
        assert_eq!(body["response"]["status"], "failed");
        assert_eq!(body["response"]["error"]["code"], "model_streamed_error");
        assert_eq!(body["response"]["error"]["message"], "upstream unavailable");
        assert_eq!(
            body["response"]["output"][0]["content"][0]["text"], "hi",
            "the open item's partial text rides the failed envelope"
        );
        assert!(
            body["sequence_number"].as_u64().unwrap() > frames_before_error.len() as u64,
            "sequence continues past the close frames"
        );
    }
}

#[cfg(test)]
mod graft_tests {
    use super::*;

    /// The echo is built from the raw body, so it must accept every effort the ingress accepts.
    /// Parsing it into a type that cannot spell one drops the whole block leniently, taking
    /// `summary` with it and withholding the reasoning item the client asked for.
    #[test]
    fn echoed_reasoning_survives_an_effort_the_upstream_enum_cannot_spell() {
        let params = ResponsesParams::from_body(&json!({
            "reasoning": {"effort": "max", "summary": "auto"}
        }));
        let reasoning = params.reasoning.expect("the block must survive");
        assert_eq!(
            reasoning.effort,
            Some(dynamo_protocols::types::ReasoningEffort::Xhigh),
            "`max` reports as the strongest level the echo type has"
        );
        assert_eq!(
            reasoning.summary,
            Some(dynamo_protocols::types::responses::ReasoningSummary::Auto)
        );

        let off_ladder = ResponsesParams::from_body(&json!({
            "reasoning": {"effort": "turbo", "summary": "auto"}
        }))
        .reasoning
        .expect("the block must survive an unknown effort too");
        assert_eq!(off_ladder.effort, None, "nothing to echo for a non-level");
        assert_eq!(
            off_ladder.summary,
            Some(dynamo_protocols::types::responses::ReasoningSummary::Auto)
        );

        // The halves are independent in both directions: a summary this type
        // cannot spell must not discard the effort either.
        let bad_summary = ResponsesParams::from_body(&json!({
            "reasoning": {"effort": "high", "summary": "verbose"}
        }))
        .reasoning
        .expect("the block must survive an unknown summary");
        assert_eq!(
            bad_summary.effort,
            Some(dynamo_protocols::types::ReasoningEffort::High)
        );
        assert_eq!(bad_summary.summary, None);
    }

    fn params() -> ResponsesParams {
        ResponsesParams::from_body(&json!({
            "model": "m",
            "input": "hi",
            "temperature": 0.5,
            "store": true,
            "presence_penalty": 0.75,
            "frequency_penalty": 0.25,
            "top_logprobs": 5,
            "metadata": {"job": "x"},
            "service_tier": "auto",
            "tools": [{"type": "function", "name": "get_weather", "parameters": {"type": "object"}}],
        }))
    }

    fn frame_body(frame: &str) -> Value {
        let (event, data) = frame.split_once('\n').unwrap();
        let name = event.strip_prefix("event: ").unwrap();
        let body: Value =
            serde_json::from_str(data.strip_prefix("data: ").unwrap().trim_end()).unwrap();
        assert_eq!(body["type"], name);
        body
    }

    /// The terminal envelope echoes request params with the spec's defaults for omitted fields and
    /// the spec patch (present-as-null required fields, `presence_penalty`/`frequency_penalty`/
    /// `store` injection, `usage.input_tokens_details.cache_write_tokens` defaulted to 0 for
    /// openai-python >= 2.53 typed clients).
    #[test]
    fn terminal_envelope_echoes_params_with_spec_defaults_and_patch() {
        let mut framing = ResponsesFraming::new("m".to_string(), None, params());
        let mut frames = framing.on_chunk(&SemanticChunk::TextDelta("hi".into()));
        frames.extend(framing.finish(
            Termination::Model(FinishReason::Stop),
            &CompletionUsage::default(),
            &[],
        ));
        let terminal = frames
            .iter()
            .map(|frame| frame_body(frame))
            .find(|body| body["type"] == "response.completed")
            .expect("a response.completed frame");
        let response = &terminal["response"];
        assert_eq!(response["temperature"], 0.5, "param echoed");
        assert_eq!(response["top_p"], 1.0, "spec default");
        assert_eq!(response["tool_choice"], "auto", "spec default");
        assert_eq!(response["truncation"], "disabled", "spec default");
        assert_eq!(
            response["top_logprobs"], 5,
            "param echoed, not the spec default"
        );
        assert_eq!(
            response["metadata"],
            json!({"job": "x"}),
            "param echoed verbatim"
        );
        assert_eq!(response["service_tier"], "auto", "param echoed");
        assert_eq!(response["store"], true, "injected from the request");
        assert_eq!(
            response["presence_penalty"], 0.75,
            "injected from the request"
        );
        assert_eq!(
            response["frequency_penalty"], 0.25,
            "injected from the request"
        );
        assert_eq!(response["tools"][0]["name"], "get_weather");
        assert_eq!(response["tools"][0]["strict"], true, "normalized");
        for key in dynamo_protocols::types::responses::SPEC_NULLABLE_REQUIRED_RESPONSE_FIELDS {
            assert!(
                response.get(*key).is_some(),
                "spec-nullable-required field {key} must be present"
            );
        }
        assert_eq!(
            response["usage"]["input_tokens_details"]["cache_write_tokens"], 0,
            "cache_write_tokens must be explicit"
        );
    }

    /// A terminal backend error emits `response.failed` carrying whatever output streamed, with the
    /// typed `error.code` string clients classify on.
    #[test]
    fn backend_error_finishes_as_response_failed_with_partial_output_and_typed_code() {
        let cases = [
            (Some(429), "anything", "rate_limit_exceeded"),
            (
                Some(400),
                "Input length 100 exceeds the maximum allowed input length of 10 tokens.",
                "context_length_exceeded",
            ),
            (
                Some(400),
                "maximum context length is 8192 tokens",
                "context_length_exceeded",
            ),
            (Some(500), "boom", "server_error"),
        ];
        for (http_status, message, expected_code) in cases {
            let mut framing =
                ResponsesFraming::new("m".to_string(), None, ResponsesParams::default());
            framing.on_chunk(&SemanticChunk::TextDelta("partial".into()));
            let frames = framing.finish_with_backend_error(&BackendError {
                message: message.to_string(),
                http_status,
                error_code: None,
            });
            let terminal = frames
                .iter()
                .map(|frame| frame_body(frame))
                .find(|body| body["type"] == "response.failed")
                .unwrap_or_else(|| panic!("{message}: no response.failed frame"));
            let response = &terminal["response"];
            assert_eq!(response["status"], "failed");
            assert_eq!(response["error"]["code"], expected_code, "{message}");
            assert_eq!(response["error"]["message"], message);
            assert_eq!(
                response["output"][0]["content"][0]["text"], "partial",
                "the streamed partial output must ride the failure envelope"
            );
            assert_eq!(
                response["output"][0]["status"], "incomplete",
                "the in-flight item closes incomplete, not completed"
            );
        }
    }

    /// A truncation-shaped backend error ("cutoff by max_tokens") is a length finish the worker
    /// mispresented as a failure: re-presented as the spec incomplete shape, never `failed`.
    #[test]
    fn truncation_shaped_backend_error_rescues_to_incomplete() {
        let mut framing = ResponsesFraming::new("m".to_string(), None, ResponsesParams::default());
        framing.on_chunk(&SemanticChunk::TextDelta("partial".into()));
        let frames = framing.finish_with_backend_error(&BackendError {
            message: "model error: Tool calls cutoff by max_tokens.".to_string(),
            http_status: Some(400),
            error_code: None,
        });
        let terminal = frame_body(frames.last().unwrap());
        assert_eq!(terminal["type"], "response.incomplete");
        assert_eq!(
            terminal["response"]["incomplete_details"]["reason"],
            "max_output_tokens"
        );
        assert!(terminal["response"]["error"].is_null(), "not a failure");
        assert_eq!(
            terminal["response"]["output"][0]["content"][0]["text"],
            "partial"
        );
        assert_eq!(terminal["response"]["output"][0]["status"], "incomplete");
    }
}
