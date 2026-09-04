//! Request edge: client request bytes -> canonical typed [`CcRequest`] (+ server-tool claims).
//!
//! Two protocol flavors converge on one typed CC request. Both normalize as JSON first —
//! server-tool entry handling *must* run pre-typing (a `baseten__*` tool entry isn't a valid CC
//! tool) — then finalize once via `serde_json::from_value::<CcRequest>`, so everything downstream
//! (accumulator, `build_next_request`, predict body) is typed + validated.
//!
//! - CC-native: near-identity — route `baseten__*` tools through the hooks.
//! - Messages: parse onto the shared Anthropic types, then translate blocks to CC, splitting an
//!   echoed assistant turn back at each `tool_result` so the replayed CC prefix is byte-identical to
//!   what the model saw. A block kind with no CC translation is refused, never dropped.
//!
//! Server-tool selection is not decided here: every server-tool-shaped `tools[]` entry goes to the
//! request's [`IngressHooks`], which claims, drops, or rejects it (see [`crate::hooks`]).
//!
//! Ported (block->CC translation) from basetenlabs/dynamo @ 68dec805 (Apache-2.0), via tool-bank's
//! `api_translation/request.rs`:
//!  - Anthropic->CC fan-in (`UnifiedRequest` `TryFrom`):
//!    https://github.com/basetenlabs/dynamo/blob/68dec805e22e66851dc0bf2a8faba7ce0665b00d/lib/llm/src/protocols/unified.rs
//!  - `convert_user_blocks` / `convert_assistant_blocks`:
//!    https://github.com/basetenlabs/dynamo/blob/68dec805e22e66851dc0bf2a8faba7ce0665b00d/lib/llm/src/protocols/anthropic/types.rs
//!
//! SPDX-License-Identifier: Apache-2.0.

use std::num::NonZeroU32;

use http::HeaderMap;
use serde::Deserialize;
use serde_json::{Value, json};

use dynamo_protocols::types::anthropic::{
    AnthropicContentBlock, AnthropicImageSource, AnthropicMessage, AnthropicMessageContent,
    AnthropicRole, AnthropicTool, AnthropicToolChoice, AnthropicToolChoiceMode, ThinkingConfig,
    ToolResultContent, ToolResultContentBlock,
};
use dynamo_protocols::types::responses::{
    EasyInputContent, FunctionCallOutput, InputContent, InputItem, InputParam, InputRole, Item,
    MessageItem, Reasoning as ResponsesReasoning, ReasoningItemContent, ReasoningSummary,
    ResponseTextParam, Role as ResponsesRole, SummaryPart, TextResponseFormatConfiguration,
    Tool as ResponsesTool, ToolChoiceFunction, ToolChoiceOptions, ToolChoiceParam,
};
use dynamo_protocols::types::{
    ChatCompletionNamedToolChoice, ChatCompletionRequestMessageContentPartImage,
    ChatCompletionRequestMessageContentPartText, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestToolMessageContent,
    ChatCompletionRequestToolMessageContentPart, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, ChatCompletionStreamOptions, ChatCompletionTool,
    ChatCompletionToolChoiceOption, ChatCompletionToolType, FunctionName, FunctionObject, ImageUrl,
    Stop,
};

use crate::coding_adapter::CodingAdapter;
use crate::framing::ResponsesParams;
use crate::history::{AssistantMessageBuffer, tool_result_message};
use crate::hooks::{
    IngressHooks, REACT_ITERATIONS_MAX, REACT_ITERATIONS_MIN, ToolDisposition,
    is_reserved_tool_type,
};
use crate::model::{ReactCapSource, RequestRejection, ToolCall};
use crate::{
    CcMessage, CcRequest, ClientProtocol, IMAGE_SOURCE_TYPE_BASE64, RESERVED_TOOL_PREFIX,
    SERVER_TOOL_USE_ID_PREFIX,
};
use dynamo_protocols::types::CreateChatCompletionRequest;

/// The reserved `baseten` request-body object — Baseten's whole extension namespace on top of the
/// OpenAI/Anthropic request schemas. Absent means all defaults.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct BasetenRequestExtension {
    #[serde(default)]
    pub tool_settings: ToolSettings,
}

/// `baseten.tool_settings`: per-request server-tool knobs. Which server tools run is chosen by
/// `tools` entries carrying the reserved prefix, not here.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolSettings {
    /// `NonZero`: a `0` cap is meaningless, so it is a 400 rather than a value picked for the client.
    pub max_react_iterations: Option<NonZeroU32>,
    pub max_tool_calls_per_iteration: Option<NonZeroU32>,
}

/// The server tools the hooks claimed for one request, in `tools` order. Construction proves the
/// names are distinct, so this doubles as a loop's server-vs-client discriminator.
#[derive(Debug, Default)]
pub struct ServerToolClaims(Vec<String>);

impl ServerToolClaims {
    /// A duplicate is caller-supplied (one `tools` entry yields one claim) and would advertise
    /// two identical function tools — refuse it instead of forwarding a malformed tool list.
    pub fn new(names: Vec<String>) -> Result<Self, String> {
        let mut seen = std::collections::HashSet::with_capacity(names.len());
        if let Some(duplicate) = names.iter().find(|name| !seen.insert(name.as_str())) {
            return Err(format!("duplicate server tool selection `{duplicate}`"));
        }
        Ok(Self(names))
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The claimed name behind a tool name the model called, or `None` for a client tool.
    pub fn selected(&self, name: &str) -> Option<&str> {
        self.0
            .iter()
            .find(|claimed| claimed.as_str() == name)
            .map(String::as_str)
    }

    pub fn joined(&self) -> String {
        self.0.join(", ")
    }
}

/// The adapted request: the typed CC template (messages included) + everything a caller's loop
/// needs that isn't in the request body.
pub struct AdaptedRequest {
    /// Canonical CC template; `messages`/`stream` are re-set per iteration by [`build_next_request`].
    pub request: CcRequest,
    /// What the hooks claimed; empty under [`crate::hooks::DropServerTools`].
    pub server_tool_claims: ServerToolClaims,
    pub max_react_iterations: NonZeroU32,
    pub react_cap_source: ReactCapSource,
    pub max_tool_calls_per_iteration: NonZeroU32,
    /// Whole-request server-tool budget (OpenAI Responses `max_tool_calls`); `None` = unbudgeted.
    pub max_tool_calls: Option<NonZeroU32>,
    /// Client-facing egress: `true` streams SSE, `false` buffers one response. The model call
    /// is always streamed regardless.
    pub should_stream: bool,
}

/// Everything ingress adaptation produced. The adapter rides alongside the request rather than
/// inside it: it is a runtime object, and [`AdaptedRequest`] is the payload a loop sends.
pub struct AdaptedIngress {
    pub request: AdaptedRequest,
    pub coding_adapter: Option<Box<dyn CodingAdapter>>,
    /// The Responses request's echo-back params, for
    /// [`ClientProtocol::envelope`](crate::ClientProtocol) on the Responses protocol. `None` on
    /// the other protocols.
    pub responses_params: Option<ResponsesParams>,
}

/// Tools removed by a `Drop` disposition, for the tool_choice degrade below.
#[derive(Default)]
struct DroppedTools {
    names: Vec<String>,
    any: bool,
}

impl DroppedTools {
    fn record(&mut self, protocol: ClientProtocol, entry: &Value) {
        self.any = true;
        let name = match protocol {
            ClientProtocol::ChatCompletions => entry
                .get("function")
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str),
            ClientProtocol::Messages | ClientProtocol::Responses => {
                entry.get("name").and_then(Value::as_str)
            }
        };
        let kind = entry.get("type").and_then(Value::as_str).unwrap_or("");
        tracing::warn!(
            tool_type = kind,
            tool_name = name.unwrap_or(""),
            "dropping server-tool-shaped tool: nothing on this endpoint executes it"
        );
        if let Some(name) = name {
            self.names.push(name.to_string());
        }
        // A nameless selection entry (`{"type": "baseten__…"}`, a hosted-tool shape) is addressed
        // by its type; record that too so a tool_choice naming it degrades instead of reaching the
        // model as a nonexistent tool.
        if !kind.is_empty() {
            self.names.push(kind.to_string());
        }
    }
}

pub fn adapt_request(
    body: &[u8],
    protocol: ClientProtocol,
    headers: &HeaderMap,
    hooks: &mut dyn IngressHooks,
) -> Result<AdaptedIngress, RequestRejection> {
    let mut body_json: Value = serde_json::from_slice(body)
        .map_err(|e| RequestRejection::malformed(format!("invalid JSON body: {e}")))?;

    let limits = hooks.limits();
    // Removed pre-adaptation: it must never reach the model, and the Messages catch-all would forward it.
    let baseten_extension = body_json
        .as_object_mut()
        .and_then(|fields| fields.remove("baseten"))
        .map(serde_json::from_value::<BasetenRequestExtension>)
        .transpose()
        .map_err(|e| {
            RequestRejection::malformed(format!("invalid `baseten` request extension: {e}"))
        })?
        .unwrap_or_default();
    let tool_settings = baseten_extension.tool_settings;
    let max_tool_calls_per_iteration = tool_calls_within_ceiling(
        limits.max_tool_calls_per_iteration,
        tool_settings.max_tool_calls_per_iteration,
    )?;
    // The one place a protocol picks its coding client; everything that client needs — the body
    // rewrite here, the rendering later — rides what it returns.
    let ingress_rewrite = hooks.rewrite_ingress(protocol, headers, &mut body_json)?;
    let responses_params = matches!(protocol, ClientProtocol::Responses)
        .then(|| ResponsesParams::from_body(&body_json));
    let mut dropped = DroppedTools::default();
    let (mut request, server_tool_claims, max_tool_calls) = match protocol {
        ClientProtocol::ChatCompletions => adapt_cc(hooks, &mut dropped, body_json)?,
        ClientProtocol::Messages => adapt_messages(hooks, &mut dropped, body_json)?,
        ClientProtocol::Responses => adapt_responses(hooks, &mut dropped, body_json)?,
    };
    // A tool_choice naming a dropped tool demands a call nothing can answer; degrade it to auto
    // (standard-dynamo behavior) rather than trapping the request.
    if let Some(ChatCompletionToolChoiceOption::Named(named)) = &request.tool_choice
        && dropped.names.contains(&named.function.name)
    {
        tracing::warn!(
            tool_name = %named.function.name,
            "tool_choice names a dropped server tool; degrading to auto"
        );
        request.tool_choice = Some(ChatCompletionToolChoiceOption::Auto);
    }
    // After adaptation: a single iteration can only ever end on a dispatched call, so a request that
    // can reach a server tool needs two.
    let iterations_floor =
        if can_dispatch_server_tool(request.tool_choice.as_ref(), &server_tool_claims) {
            REACT_ITERATIONS_MIN
        } else {
            NonZeroU32::MIN
        };
    let (max_react_iterations, react_cap_source) = match (
        ingress_rewrite
            .as_ref()
            .and_then(|ingress_rewrite| ingress_rewrite.max_react_iterations),
        tool_settings.max_react_iterations,
    ) {
        // No principled winner to pick, and picking one quietly costs a debugging afternoon.
        (Some(_), Some(_)) => {
            return Err(RequestRejection::malformed(format!(
                "a client search tool's max_uses and baseten.tool_settings.max_react_iterations \
                 both bound the ReAct loop; send one, not both (max_react_iterations accepts \
                 {iterations_floor}..={REACT_ITERATIONS_MAX})"
            )));
        }
        // Clamped, not refused: `max_uses` counts searches, an axis whose bounds the client had no
        // way to aim at ours, so a value outside them is not a caller mistake to report.
        (Some(translated), None) => {
            let clamped = translated.clamp(iterations_floor, REACT_ITERATIONS_MAX);
            let react_cap_source = if clamped == REACT_ITERATIONS_MAX {
                ReactCapSource::ServiceCeiling
            } else {
                ReactCapSource::Request
            };
            (clamped, react_cap_source)
        }
        (None, requested) => react_iterations_within_bounds(
            limits.default_react_iterations,
            requested,
            iterations_floor,
        )?,
    };

    // Model backends 400 on `tool_choice` without `tools`, and the openai/codex SDKs send
    // `tool_choice: "auto"` unconditionally even on tool-less calls — drop the no-op choice.
    if request.tools.as_ref().is_none_or(Vec::is_empty) {
        request.tool_choice = match request.tool_choice.take() {
            None
            | Some(ChatCompletionToolChoiceOption::Auto | ChatCompletionToolChoiceOption::None) => {
                None
            }
            // Dropping the request's only tools left a demanding tool_choice with nothing to
            // demand: omit it entirely — the drop already logged what happened.
            Some(_) if dropped.any => None,
            Some(
                demanding @ (ChatCompletionToolChoiceOption::Required
                | ChatCompletionToolChoiceOption::Named(_)),
            ) => {
                // Bounded: `Named` carries a client-supplied tool name in its `Debug`.
                let demanding_debug = crate::util::truncate(&format!("{demanding:?}"), 80);
                return Err(RequestRejection::malformed(format!(
                    "tool_choice {demanding_debug} requires a tool call, but the request declares no tools"
                )));
            }
        };
    }

    // The parser folds one choice per model call, so anything above 1 would silently return a
    // single-choice body for a multi-choice request.
    if request.n.is_some_and(|n| n > 1) {
        return Err(RequestRejection::Unsupported(
            "n > 1 is not supported with server tools".to_string(),
        ));
    }
    // Match SDK default.
    let should_stream = request.stream.unwrap_or(false);

    tracing::info!(
        target: "http",
        protocol = ?protocol,
        server_tools = %server_tool_claims.joined(),
        max_react_iterations,
        messages = request.messages.len(),
        "ingress adapted",
    );

    Ok(AdaptedIngress {
        request: AdaptedRequest {
            request,
            server_tool_claims,
            max_react_iterations,
            react_cap_source,
            max_tool_calls_per_iteration,
            max_tool_calls,
            should_stream,
        },
        coding_adapter: ingress_rewrite.map(|ingress_rewrite| ingress_rewrite.adapter),
        responses_params,
    })
}

/// Render one iteration's predict body from the template + current history, forcing streaming with
/// per-chunk usage. A client `required`/named `tool_choice` reaches the first model call only, or the
/// loop would be trapped in calls that never answer.
pub fn build_next_request(
    template: &CcRequest,
    messages: &[CcMessage],
    is_first_iteration: bool,
) -> CcRequest {
    let mut next_request = template.clone();
    next_request.messages = messages.to_vec();
    if !is_first_iteration {
        next_request.tool_choice = None;
    }
    next_request.stream = Some(true);
    next_request.stream_options = Some(ChatCompletionStreamOptions {
        include_usage: true,
        continuous_usage_stats: false,
    });
    next_request
}

// --- ChatCompletions: near-identity -----------------------------------------

/// The fields of a CC `tools` entry that ingress routing needs; the entry itself is forwarded
/// verbatim (client tool) or replaced (server tool selection).
#[derive(Deserialize)]
struct CcToolEntryHead {
    #[serde(rename = "type")]
    tool_type: Option<String>,
    function: Option<CcToolFunctionHead>,
}

#[derive(Deserialize)]
struct CcToolFunctionHead {
    name: Option<String>,
}

/// CC client is already canonical: route `baseten__*` tool entries through the hooks (a claim
/// becomes a function tool) and guard reserved-namespace spoofing. Messages are forwarded
/// untouched — CC carries no server-tool history back into a following request.
fn adapt_cc(
    hooks: &mut dyn IngressHooks,
    dropped: &mut DroppedTools,
    mut cc_body: Value,
) -> Result<(CcRequest, ServerToolClaims, Option<NonZeroU32>), RequestRejection> {
    let mut claimed_names = Vec::new();
    if let Some(entries) = cc_body.get_mut("tools").and_then(Value::as_array_mut) {
        let mut cc_tools = Vec::with_capacity(entries.len());
        for tool_entry in std::mem::take(entries) {
            let tool_entry_head = CcToolEntryHead::deserialize(&tool_entry)
                .map_err(|e| RequestRejection::malformed(format!("invalid tool entry: {e}")))?;
            if is_reserved_tool_type(tool_entry_head.tool_type.as_deref()) {
                match hooks.on_server_tool(ClientProtocol::ChatCompletions, &tool_entry) {
                    // Back to JSON: a client tool entry is forwarded verbatim, so the array stays
                    // untyped until the whole body is finalized below.
                    ToolDisposition::Claim(claim) => {
                        cc_tools.push(json!(claim.tool));
                        claimed_names.push(claim.name);
                    }
                    ToolDisposition::Drop => {
                        dropped.record(ClientProtocol::ChatCompletions, &tool_entry);
                    }
                    ToolDisposition::Reject(rejection) => return Err(rejection),
                }
            } else {
                let client_tool_name = tool_entry_head.function.and_then(|function| function.name);
                reject_reserved_client_tool(client_tool_name.as_deref())?;
                cc_tools.push(tool_entry);
            }
        }
        *entries = cc_tools;
    }
    let request: CcRequest = serde_json::from_value(cc_body).map_err(|e| {
        RequestRejection::malformed(format!("not a valid ChatCompletions body: {e}"))
    })?;
    // `None`: only Responses models `max_tool_calls`.
    Ok((
        request,
        ServerToolClaims::new(claimed_names).map_err(RequestRejection::malformed)?,
        None,
    ))
}

// --- Anthropic Messages: parse + translate ----------------------------------

/// Anthropic Messages request. Messages, content blocks, tools and `tool_choice` are the vendored
/// types; only the envelope is local, for two reasons the vendored
/// `AnthropicCreateMessageRequest` cannot serve:
/// - `unmodeled` catch-all: TB is a proxy, so an Anthropic field it does not translate must still
///   reach the model server. The vendored type has no catch-all, so a re-sync that models a new
///   field would silently start dropping it.
/// - `tools` stays `Vec<Value>`: a `baseten__*` selection entry carries no `name`, which the
///   vendored `AnthropicTool` requires. Client entries are typed as `AnthropicTool` per entry below.
#[derive(Deserialize)]
struct MessagesRequest {
    model: String,
    max_tokens: Option<u32>,
    #[serde(default)]
    messages: Vec<AnthropicMessage>,
    /// String or text-block array; `flatten_text` handles both (the vendored `SystemContent`
    /// deserializer that does the same is private to that crate).
    system: Option<Value>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stop_sequences: Option<Vec<String>>,
    /// Modeled because TB *reads* it to choose streaming vs buffered egress; leaving it to
    /// `unmodeled` would forward it to the model while TB itself saw no stream request.
    stream: Option<bool>,
    thinking: Option<ThinkingConfig>,
    output_config: Option<MessagesOutputConfig>,
    #[serde(default)]
    tools: Vec<Value>,
    tool_choice: Option<Value>,
    /// Modeled only to refuse them: both request Anthropic-hosted execution (remote MCP servers,
    /// code-execution containers) that nothing on the model's CC endpoint provides, so forwarding via
    /// `unmodeled` would silently run the request without the asked-for capability.
    mcp_servers: Option<Value>,
    container: Option<Value>,
    /// Every field TB doesn't model. Forwarded verbatim into the CC body so the model server (via
    /// dynamo) is the boundary that supports or rejects it — a new Anthropic field is never dropped
    /// *here*, but an Anthropic-spelled key is unrecognized on the model's CC endpoint (see
    /// tool-bank's `docs/protocol.md` (monorepo `rust/tool-bank/docs/protocol.md`) "Unmodeled request fields").
    #[serde(flatten)]
    unmodeled: serde_json::Map<String, Value>,
}

/// Translate the whole Messages request into the canonical typed CC request.
fn adapt_messages(
    hooks: &mut dyn IngressHooks,
    dropped: &mut DroppedTools,
    body: Value,
) -> Result<(CcRequest, ServerToolClaims, Option<NonZeroU32>), RequestRejection> {
    let messages_request: MessagesRequest = serde_json::from_value(body)
        .map_err(|e| RequestRejection::Malformed(format!("invalid Messages request: {e}")))?;
    if messages_request.mcp_servers.is_some() {
        return Err(RequestRejection::Unsupported(
            "`mcp_servers` is not supported: tool-bank does not connect to caller-supplied MCP \
             servers; use `baseten__*` server tools"
                .to_string(),
        ));
    }
    if messages_request.container.is_some() {
        return Err(RequestRejection::Unsupported(
            "`container` is not supported: tool-bank runs no code-execution containers".to_string(),
        ));
    }

    let mut cc_messages: Vec<CcMessage> = Vec::new();
    // An empty or null `system` carries no instruction; a `{"role":"system","content":""}` in the
    // replayed prefix would be TB adding a message the caller did not send.
    if let Some(system) = messages_request
        .system
        .as_ref()
        .map(flatten_text)
        .filter(|system| !system.is_empty())
    {
        cc_messages.push(text_message(&AnthropicRole::System, system)?);
    }
    for message in &messages_request.messages {
        translate_message(message, &mut cc_messages)?;
    }

    let (cc_tools, claimed_names) = expand_declared_tools(
        hooks,
        dropped,
        ClientProtocol::Messages,
        messages_request.tools,
        is_anthropic_server_tool_shaped,
        client_function_tool,
    )?;

    let (tool_choice, parallel_tool_calls) = messages_request
        .tool_choice
        .as_ref()
        .map(translate_tool_choice)
        .transpose()?
        .unzip();
    let (reasoning_effort, response_format) =
        messages_output_config(messages_request.output_config)?;
    let request = CcRequest {
        inner: CreateChatCompletionRequest {
            model: messages_request.model,
            messages: cc_messages,
            max_completion_tokens: messages_request.max_tokens,
            temperature: messages_request.temperature,
            top_p: messages_request.top_p,
            stop: messages_request.stop_sequences.map(Stop::StringArray),
            stream: messages_request.stream,
            tools: (!cc_tools.is_empty()).then_some(cc_tools),
            tool_choice,
            parallel_tool_calls: parallel_tool_calls.flatten(),
            reasoning_effort,
            response_format,
            ..CreateChatCompletionRequest::default()
        },
        // TODO(BT-16120): these reach the model's CC endpoint, not dynamo's Messages->CC path, so an
        // Anthropic-shaped field (e.g. cache_control) may not be interpreted there.
        unmodeled: unmodeled_fields(messages_request.unmodeled, messages_request.thinking)
            .map_err(RequestRejection::Malformed)?,
    };
    // `None`: only Responses models `max_tool_calls`.
    Ok((
        request,
        ServerToolClaims::new(claimed_names).map_err(RequestRejection::malformed)?,
        None,
    ))
}

/// Anthropic `thinking`, parsed once into states that can hold what is legal for them: only
/// `Enabled` takes a manual budget — Anthropic rejects `budget_tokens` with the other two modes,
/// and `adaptive` (the only mode on Anthropic 4.7+ models) scopes depth via `output_config.effort`.
enum MessagesThinking {
    Enabled { budget_tokens: Option<u32> },
    Adaptive,
    Disabled,
}

fn parse_messages_thinking(thinking: ThinkingConfig) -> Result<MessagesThinking, String> {
    match (thinking.thinking_type.as_str(), thinking.budget_tokens) {
        ("enabled", budget_tokens) => Ok(MessagesThinking::Enabled { budget_tokens }),
        ("adaptive", None) => Ok(MessagesThinking::Adaptive),
        ("disabled", None) => Ok(MessagesThinking::Disabled),
        ("adaptive" | "disabled", Some(_)) => Err(format!(
            "`thinking.budget_tokens` is only valid with `type: \"enabled\"`, not with {:?}",
            thinking.thinking_type
        )),
        (other, _) => Err(format!("unsupported thinking type {other:?}")),
    }
}

/// The request's untranslated fields, plus the reasoning enablement CC carries as a vendor field.
///
/// Anthropic `thinking` -> `chat_template_kwargs.enable_thinking` (Kimi K2.x gates the
/// `reasoning_content` channel on it) — `disabled` translates too, or an explicit opt-out would
/// silently vanish and leave the model default in charge. `budget_tokens` -> `thinking_budget` is
/// best-effort: an OpenAI endpoint has no native budget field and Kimi honors it only loosely.
fn unmodeled_fields(
    mut unmodeled: serde_json::Map<String, Value>,
    thinking: Option<ThinkingConfig>,
) -> Result<serde_json::Map<String, Value>, String> {
    let Some(thinking) = thinking else {
        return Ok(unmodeled);
    };
    let mut translated_kwargs = serde_json::Map::new();
    match parse_messages_thinking(thinking)? {
        MessagesThinking::Enabled { budget_tokens } => {
            translated_kwargs.insert("enable_thinking".to_string(), json!(true));
            if let Some(budget_tokens) = budget_tokens {
                translated_kwargs.insert("thinking_budget".to_string(), json!(budget_tokens));
            }
        }
        MessagesThinking::Adaptive => {
            translated_kwargs.insert("enable_thinking".to_string(), json!(true));
        }
        MessagesThinking::Disabled => {
            translated_kwargs.insert("enable_thinking".to_string(), json!(false));
        }
    }
    // The client's own kwargs win per key (more specific), but must not erase the rest of the translation.
    match unmodeled
        .entry("chat_template_kwargs")
        .or_insert_with(|| Value::Object(serde_json::Map::new()))
    {
        Value::Object(client_kwargs) => {
            for (key, value) in translated_kwargs {
                client_kwargs.entry(key).or_insert(value);
            }
        }
        _ => return Err("`chat_template_kwargs` must be a JSON object".to_string()),
    }
    Ok(unmodeled)
}

/// Anthropic `output_config`: `effort` -> CC `reasoning_effort` (low|medium|high|max, plus the
/// `xhigh` Claude Code itself sends; both land on CC's `xhigh`), `format`
/// (structured output) -> CC `response_format`.
#[derive(Deserialize)]
struct MessagesOutputConfig {
    effort: Option<String>,
    format: Option<MessagesOutputFormat>,
    #[serde(flatten)]
    unmodeled: serde_json::Map<String, Value>,
}

#[derive(Deserialize)]
struct MessagesOutputFormat {
    #[serde(rename = "type")]
    format_type: String,
    schema: Option<Value>,
    #[serde(flatten)]
    unmodeled: serde_json::Map<String, Value>,
}

fn messages_output_config(
    output_config: Option<MessagesOutputConfig>,
) -> Result<
    (
        Option<dynamo_protocols::types::ReasoningEffort>,
        Option<dynamo_protocols::types::ResponseFormat>,
    ),
    RequestRejection,
> {
    use dynamo_protocols::types::ReasoningEffort;
    let Some(output_config) = output_config else {
        return Ok((None, None));
    };
    if let Some(unknown_key) = output_config.unmodeled.keys().next() {
        return Err(RequestRejection::Unsupported(format!(
            "unsupported `output_config` field {unknown_key:?}"
        )));
    }
    let reasoning_effort = output_config
        .effort
        .map(|effort| match effort.as_str() {
            "low" => Ok(ReasoningEffort::Low),
            "medium" => Ok(ReasoningEffort::Medium),
            "high" => Ok(ReasoningEffort::High),
            // Anthropic documents `max`; Claude Code sends CC's own spelling.
            "max" | "xhigh" => Ok(ReasoningEffort::Xhigh),
            other => Err(RequestRejection::Unsupported(format!(
                "unsupported `output_config.effort` {other:?}"
            ))),
        })
        .transpose()?;
    let response_format = output_config
        .format
        .map(messages_response_format)
        .transpose()?;
    Ok((reasoning_effort, response_format))
}

/// Anthropic `output_config.format` -> CC `response_format`: the same JSON-schema payload, plus
/// the `name` CC requires and Anthropic has no field for, and `strict` — Anthropic's structured
/// output guarantees conformance, which on the model's CC endpoint is exactly the strict mode.
fn messages_response_format(
    format: MessagesOutputFormat,
) -> Result<dynamo_protocols::types::ResponseFormat, RequestRejection> {
    if format.format_type != "json_schema" {
        return Err(RequestRejection::Unsupported(format!(
            "unsupported `output_config.format.type` {:?}: only \"json_schema\"",
            format.format_type
        )));
    }
    if let Some(unknown_key) = format.unmodeled.keys().next() {
        return Err(RequestRejection::Unsupported(format!(
            "unsupported `output_config.format` field {unknown_key:?}"
        )));
    }
    let Some(schema) = format.schema else {
        return Err(RequestRejection::malformed(
            "`output_config.format` of type \"json_schema\" requires `schema`",
        ));
    };
    Ok(dynamo_protocols::types::ResponseFormat::JsonSchema {
        json_schema: dynamo_protocols::types::ResponseFormatJsonSchema {
            name: "output".to_string(),
            description: None,
            schema,
            strict: Some(true),
        },
    })
}

/// A caller-executed Anthropic tool definition -> a CC function tool. An Anthropic *native*
/// server-tool entry (a versioned `type` like `web_search_20250305`) is refused: only Anthropic runs
/// those, so advertising it to the model would offer a tool nobody can execute.
fn client_function_tool(tool_entry: Value) -> Result<ChatCompletionTool, RequestRejection> {
    let client_tool: AnthropicTool = serde_json::from_value(tool_entry)
        .map_err(|e| RequestRejection::malformed(format!("invalid tool definition: {e}")))?;
    if let Some(tool_type) = client_tool.tool_type.as_deref().filter(|t| *t != "custom") {
        // Malformed, not Unsupported: only Anthropic can execute these, so it is the caller
        // pointing an Anthropic-hosted tool at a non-Anthropic endpoint, not a TB gap to close.
        return Err(RequestRejection::malformed(format!(
            "tool {:?} has type {tool_type:?}: Anthropic-native server tools are not supported; \
             use a `{RESERVED_TOOL_PREFIX}*` type for a Baseten server tool",
            client_tool.name
        )));
    }
    reject_reserved_client_tool(Some(&client_tool.name))?;
    Ok(function_tool(
        &client_tool.name,
        client_tool.description.as_deref().unwrap_or_default(),
        client_tool.input_schema.unwrap_or_else(|| json!({})),
    ))
}

/// Anthropic `tool_choice` -> CC. `any`->`required`, `tool`->a named function choice; `auto`/`none`
/// map through, `disable_parallel_tool_use: true` -> `parallel_tool_calls: false`. Anything TB
/// cannot carry over is refused, never dropped: a caller that asked for a behavior must not be told
/// 200 while the request ran without it.
fn translate_tool_choice(
    tool_choice: &Value,
) -> Result<(ChatCompletionToolChoiceOption, Option<bool>), RequestRejection> {
    let tool_choice = AnthropicToolChoice::deserialize(tool_choice)
        .map_err(|e| RequestRejection::Malformed(format!("unsupported tool_choice: {e}")))?;
    // Untagged, so a `{type: tool}` with no `name` decodes as Simple rather than failing above.
    let (mode, name, disable_parallel_tool_use) = match &tool_choice {
        AnthropicToolChoice::Named(named) => (
            &named.choice_type,
            Some(named.name.as_str()),
            named.disable_parallel_tool_use,
        ),
        AnthropicToolChoice::Simple(simple) => {
            (&simple.choice_type, None, simple.disable_parallel_tool_use)
        }
    };
    // Only an explicit `true` emits the CC field; both defaults mean "parallel allowed".
    let parallel_tool_calls = (disable_parallel_tool_use == Some(true)).then_some(false);
    let translated = match (mode, name) {
        (AnthropicToolChoiceMode::Auto, _) => ChatCompletionToolChoiceOption::Auto,
        (AnthropicToolChoiceMode::Any, _) => ChatCompletionToolChoiceOption::Required,
        (AnthropicToolChoiceMode::None, _) => ChatCompletionToolChoiceOption::None,
        (AnthropicToolChoiceMode::Tool, Some(name)) => {
            ChatCompletionToolChoiceOption::Named(ChatCompletionNamedToolChoice {
                r#type: ChatCompletionToolType::Function,
                function: FunctionName {
                    name: name.to_string(),
                },
            })
        }
        (AnthropicToolChoiceMode::Tool, None) => {
            return Err(RequestRejection::malformed(
                "tool_choice type `tool` requires `name`",
            ));
        }
    };
    Ok((translated, parallel_tool_calls))
}

/// Translate one Anthropic message into one or more CC messages. Key case: an echoed assistant turn
/// (TB presents the whole server-tool loop as one Messages turn) splits back at each `tool_result`
/// into `[assistant(tool_calls), tool(result), assistant(text)]`, byte-identical to what the model saw —
/// which holds by construction, since the assistant messages come from the same
/// [`AssistantMessageBuffer`] the loop appends with.
///
/// `server_tool_use` / `web_search_tool_result` are the same two shapes under Anthropic's
/// server-tool names, so they fold into the same buffer. Absorbed here rather than in a coding
/// adapter: they are Messages protocol blocks, and a caller replaying a turn it got from Anthropic
/// itself is entitled to send them whether or not it declared a coding client.
fn translate_message(
    message: &AnthropicMessage,
    cc_messages: &mut Vec<CcMessage>,
) -> Result<(), RequestRejection> {
    let blocks = match &message.content {
        AnthropicMessageContent::Text { content } => {
            cc_messages.push(text_message(&message.role, content.clone())?);
            return Ok(());
        }
        AnthropicMessageContent::Blocks { content } => content,
    };

    if message.role == AnthropicRole::Assistant {
        let mut turn = AssistantMessageBuffer::default();
        for block in blocks {
            match block {
                AnthropicContentBlock::Text { text, .. } => turn.text.push_str(text),
                AnthropicContentBlock::Thinking { thinking, .. } => {
                    turn.thinking.push_str(thinking);
                }
                AnthropicContentBlock::ToolUse {
                    id, name, input, ..
                } => turn.tool_calls.push(echoed_tool_call(id, name, input)),
                AnthropicContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    ..
                } => {
                    turn.flush_into(cc_messages);
                    cc_messages.push(echoed_tool_result(tool_use_id, content.as_ref())?);
                }
                AnthropicContentBlock::ServerToolUse { id, name, input } => turn.tool_calls.push(
                    echoed_tool_call(&replayed_server_tool_use_id(id), name, input),
                ),
                AnthropicContentBlock::WebSearchToolResult {
                    tool_use_id,
                    content,
                } => {
                    turn.flush_into(cc_messages);
                    cc_messages.push(
                        echoed_server_tool_result(tool_use_id, content)
                            .map_err(RequestRejection::malformed)?,
                    );
                }
                unsupported => return Err(unsupported_block("assistant", unsupported)),
            }
        }
        turn.flush_into(cc_messages);
        return Ok(());
    }

    // Text keeps its place before a following tool_result (byte-identical CC replay).
    let mut text = String::new();
    for block in blocks {
        match block {
            AnthropicContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                if !text.is_empty() {
                    cc_messages.push(text_message(&message.role, std::mem::take(&mut text))?);
                }
                cc_messages.push(echoed_tool_result(tool_use_id, content.as_ref())?);
            }
            AnthropicContentBlock::Text {
                text: block_text, ..
            } => text.push_str(block_text),
            unsupported => return Err(unsupported_block("user", unsupported)),
        }
    }
    if !text.is_empty() {
        cc_messages.push(text_message(&message.role, text)?);
    }
    Ok(())
}

/// Appended to a rendered body, so it is always the last message and rides the request only, never
/// the accumulated history a client replays.
///
/// `user` because open-weight templates carry no mid-conversation system role, and those accepting
/// `system` in `messages[]` often hoist it to the prompt's top, rewriting the warmed prefix.
#[must_use]
pub fn with_steering_prompt(mut request: CcRequest, steering_prompt: &str) -> CcRequest {
    request.messages.push(user_message(steering_prompt));
    request
}

fn user_message(text: &str) -> CcMessage {
    CcMessage::User(ChatCompletionRequestUserMessage {
        content: ChatCompletionRequestUserMessageContent::Text(text.to_owned()),
        name: None,
    })
}

/// A plain-text message in the CC role matching the Anthropic one. `system` inside `messages[]` is a
/// client compatibility shape (Anthropic's own field is top-level) and keeps its role.
fn text_message(role: &AnthropicRole, text: String) -> Result<CcMessage, RequestRejection> {
    Ok(match role {
        AnthropicRole::User => CcMessage::User(ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Text(text),
            name: None,
        }),
        AnthropicRole::System => CcMessage::System(ChatCompletionRequestSystemMessage {
            content: Some(ChatCompletionRequestSystemMessageContent::Text(text)),
            name: None,
            tools: None,
        }),
        // An assistant message reaches this only as `content: "..."`, with no blocks to split.
        AnthropicRole::Assistant => CcMessage::Assistant(
            AssistantMessageBuffer {
                text,
                ..AssistantMessageBuffer::default()
            }
            .into_assistant_message()
            .ok_or_else(|| RequestRejection::malformed("assistant message has empty content"))?,
        ),
    })
}

/// An echoed `tool_use` block back into the CC tool call TB originally sent.
fn echoed_tool_call(id: &str, name: &str, input: &Value) -> ToolCall {
    // Explicit-null input -> `{}` (a literal "null" arg string breaks cache replay).
    let args = if input.is_null() {
        json!({})
    } else {
        input.clone()
    };
    ToolCall {
        id: id.to_string(),
        name: name.to_string(),
        raw_args: args.to_string(),
        args,
    }
}

/// Anthropic's server-tool result carries structured JSON rather than a `tool_result`'s text or
/// blocks, and the blocks the adapter mints hold only citation pairs — small enough to replay to the
/// model verbatim.
fn echoed_server_tool_result(tool_use_id: &str, content: &Value) -> Result<CcMessage, String> {
    if tool_use_id.is_empty() {
        return Err("web_search_tool_result block has an empty tool_use_id".to_string());
    }
    Ok(tool_result_message(
        replayed_server_tool_use_id(tool_use_id),
        ChatCompletionRequestToolMessageContent::Text(match content {
            Value::String(text) => text.clone(),
            other => other.to_string(),
        }),
    ))
}

/// The id as the model issued it: the adapter prefixes on the way out, so a replayed call resolves
/// to its original call id and the pair still matches.
fn replayed_server_tool_use_id(id: &str) -> String {
    id.strip_prefix(SERVER_TOOL_USE_ID_PREFIX)
        .unwrap_or(id)
        .to_string()
}

fn echoed_tool_result(
    tool_use_id: &str,
    content: Option<&ToolResultContent>,
) -> Result<CcMessage, RequestRejection> {
    // The vendored block type makes `tool_use_id` mandatory; empty is still a value TB must not
    // invent a substitute for.
    if tool_use_id.is_empty() {
        return Err(RequestRejection::malformed(
            "tool_result block has an empty tool_use_id",
        ));
    }
    Ok(tool_result_message(
        tool_use_id.to_string(),
        content
            .map(tool_result_content)
            .transpose()?
            .unwrap_or_default(),
    ))
}

fn tool_result_content(
    content: &ToolResultContent,
) -> Result<ChatCompletionRequestToolMessageContent, RequestRejection> {
    let blocks = match content {
        ToolResultContent::Text(text) => {
            return Ok(ChatCompletionRequestToolMessageContent::Text(text.clone()));
        }
        ToolResultContent::Blocks(blocks) => blocks,
    };
    let mut text = String::new();
    let mut parts: Vec<ChatCompletionRequestToolMessageContentPart> = Vec::new();
    let mut has_image = false;
    for block in blocks {
        match block {
            ToolResultContentBlock::Text { text: block_text } => {
                text.push_str(block_text);
                parts.push(ChatCompletionRequestToolMessageContentPart::Text(
                    ChatCompletionRequestMessageContentPartText {
                        text: block_text.clone(),
                    },
                ));
            }
            ToolResultContentBlock::Image { source } => {
                has_image = true;
                parts.push(ChatCompletionRequestToolMessageContentPart::ImageUrl(
                    image_url_part(source)?,
                ));
            }
            ToolResultContentBlock::Document(doc) => {
                let Some(block_text) = doc.text() else {
                    return Err(RequestRejection::Unsupported(
                        "unsupported tool_result document block: no text source".to_string(),
                    ));
                };
                text.push_str(&block_text);
                parts.push(ChatCompletionRequestToolMessageContentPart::Text(
                    ChatCompletionRequestMessageContentPartText { text: block_text },
                ));
            }
            ToolResultContentBlock::SearchResult(result) => {
                let Some(block_text) = result.text() else {
                    return Err(RequestRejection::Unsupported(
                        "unsupported tool_result search_result block: no text content".to_string(),
                    ));
                };
                text.push_str(&block_text);
                parts.push(ChatCompletionRequestToolMessageContentPart::Text(
                    ChatCompletionRequestMessageContentPartText { text: block_text },
                ));
            }
            ToolResultContentBlock::Other(value) => {
                return Err(RequestRejection::Unsupported(format!(
                    "unsupported tool_result content block type {:?}",
                    value
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or("(no type)")
                )));
            }
        }
    }
    Ok(if has_image {
        ChatCompletionRequestToolMessageContent::Array(parts)
    } else {
        ChatCompletionRequestToolMessageContent::Text(text)
    })
}

fn image_url_part(
    source: &AnthropicImageSource,
) -> Result<ChatCompletionRequestMessageContentPartImage, RequestRejection> {
    if source.source_type != IMAGE_SOURCE_TYPE_BASE64 {
        return Err(RequestRejection::Unsupported(format!(
            "unsupported image source type {:?}; only base64 is supported",
            source.source_type
        )));
    }
    let url = url::Url::parse(&format!(
        "data:{};base64,{}",
        source.media_type, source.data
    ))
    .map_err(|e| RequestRejection::malformed(format!("invalid image data URI: {e}")))?;
    Ok(ChatCompletionRequestMessageContentPartImage {
        image_url: ImageUrl {
            url,
            detail: None,
            uuid: None,
        },
    })
}

/// A content block kind TB has no CC translation for in this role. Refused, never dropped:
/// discarding request content would have the model answer about input the caller did not send.
/// `image` is the live case — representable in CC, just not translated yet.
/// `thinking`/`tool_use`/`server_tool_use`/`web_search_tool_result` land here from the *user* branch
/// only (assistant-only blocks in a user message — a caller mistake, named as such, not an internal
/// error).
fn unsupported_block(role: &str, block: &AnthropicContentBlock) -> RequestRejection {
    let kind = match block {
        AnthropicContentBlock::Image { .. } => "image",
        AnthropicContentBlock::RedactedThinking { .. } => "redacted_thinking",
        AnthropicContentBlock::ServerToolUse { .. } => "server_tool_use",
        AnthropicContentBlock::WebSearchToolResult { .. } => "web_search_tool_result",
        AnthropicContentBlock::Thinking { .. } => "thinking",
        AnthropicContentBlock::ToolUse { .. } => "tool_use",
        // Anything the vendored types don't model at all keeps whatever `type` the client sent.
        AnthropicContentBlock::Other(value) => value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("(no type)"),
        // Translated in every role; listed so a new vendored variant fails the build here.
        AnthropicContentBlock::Text { .. } | AnthropicContentBlock::ToolResult { .. } => {
            return RequestRejection::Unsupported(
                "internal error: translated block reported as unsupported".to_string(),
            );
        }
    };
    RequestRejection::Unsupported(format!("unsupported {role} content block type {kind:?}"))
}

// --- OpenAI Responses: parse + translate (stateless subset) ----------------

/// Responses request. `input`/`tool_choice` are the vendored types (typed directly: unlike
/// Anthropic's `tool_choice`, `ToolChoiceParam` deserializes unambiguously); `tools` stays
/// `Vec<Value>` for the same reason as the other two protocols — a `baseten__*` selection entry
/// has no `name`, which `Tool::Function` requires. See `unmodeled` on `MessagesRequest` above for
/// why the envelope is local rather than the vendored `CreateResponse`, and tool-bank's `docs/protocol.md` (monorepo `rust/tool-bank/docs/protocol.md`)
/// "Unmodeled request fields" for what forwarding does NOT promise: an untranslated
/// Responses-spelled key reaches the model's CC endpoint unrecognized.
#[derive(Deserialize)]
struct ResponsesRequest {
    model: String,
    #[serde(default)]
    input: InputParam,
    instructions: Option<String>,
    max_output_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stream: Option<bool>,
    #[serde(default)]
    tools: Vec<Value>,
    tool_choice: Option<ToolChoiceParam>,
    /// Translated, not ridden through: their CC counterparts are spelled differently
    /// (`response_format`, `reasoning_effort`), so an unmodeled pass-through would silently drop
    /// the requested behavior at the model's CC endpoint.
    text: Option<ResponseTextParam>,
    reasoning: Option<ResponsesReasoning>,
    /// TB enforces the whole-request server-tool budget itself. Modeled so it can never ride
    /// `unmodeled` to the model's CC endpoint.
    #[serde(default, deserialize_with = "responses_max_tool_calls")]
    max_tool_calls: Option<NonZeroU32>,
    /// Stateless-only guards: TB persists nothing, so any of these present is refused rather than
    /// silently ignored (a body that 200s today must not change meaning once one of these starts
    /// mattering).
    store: Option<bool>,
    previous_response_id: Option<String>,
    conversation: Option<Value>,
    background: Option<bool>,
    prompt: Option<Value>,
    #[serde(flatten)]
    unmodeled: serde_json::Map<String, Value>,
}

/// Serde reports struct-field errors without the field's name; the budget's message must carry it.
fn responses_max_tool_calls<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<NonZeroU32>, D::Error> {
    Option::<NonZeroU32>::deserialize(deserializer)
        .map_err(|e| serde::de::Error::custom(format!("invalid `max_tool_calls`: {e}")))
}

fn adapt_responses(
    hooks: &mut dyn IngressHooks,
    dropped: &mut DroppedTools,
    body: Value,
) -> Result<(CcRequest, ServerToolClaims, Option<NonZeroU32>), RequestRejection> {
    let responses_request: ResponsesRequest = serde_json::from_value(body)
        .map_err(|e| RequestRejection::Malformed(format!("invalid Responses request: {e}")))?;
    reject_stateful_fields(&responses_request)?;

    let mut cc_messages: Vec<CcMessage> = Vec::new();
    if let Some(instructions) = responses_request
        .instructions
        .as_ref()
        .filter(|text| !text.is_empty())
    {
        cc_messages.push(CcMessage::System(ChatCompletionRequestSystemMessage {
            content: Some(ChatCompletionRequestSystemMessageContent::Text(
                instructions.clone(),
            )),
            name: None,
            tools: None,
        }));
    }
    match responses_request.input {
        InputParam::Text(text) if !text.is_empty() => {
            cc_messages.push(CcMessage::User(ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text(text),
                name: None,
            }));
        }
        InputParam::Text(_) => {}
        InputParam::Items(items) => {
            translate_input_items(&items, &mut cc_messages).map_err(RequestRejection::malformed)?
        }
    }

    let (cc_tools, claimed_names) = expand_declared_tools(
        hooks,
        dropped,
        ClientProtocol::Responses,
        responses_request.tools,
        is_responses_server_tool_shaped,
        responses_client_function_tool,
    )?;

    let request = CcRequest {
        inner: CreateChatCompletionRequest {
            model: responses_request.model,
            messages: cc_messages,
            max_completion_tokens: responses_request.max_output_tokens,
            temperature: responses_request.temperature,
            top_p: responses_request.top_p,
            stream: responses_request.stream,
            tool_choice: match responses_request
                .tool_choice
                .as_ref()
                .map(translate_responses_tool_choice)
            {
                None => None,
                Some(Ok(choice)) => Some(choice),
                // Standard dynamo cannot honor a hosted-tool `tool_choice` (the tool was dropped);
                // degrade to auto like the Messages path does. tool-bank keeps the 400.
                Some(Err(reason)) if hooks.degrades_unsupported_tool_choice() => {
                    tracing::warn!(%reason, "tool_choice degraded to auto");
                    Some(ChatCompletionToolChoiceOption::Auto)
                }
                Some(Err(reason)) => return Err(RequestRejection::Unsupported(reason)),
            },
            tools: (!cc_tools.is_empty()).then_some(cc_tools),
            response_format: responses_request
                .text
                .map(cc_response_format)
                .transpose()
                .map_err(RequestRejection::Unsupported)?,
            reasoning_effort: responses_request
                .reasoning
                .map(cc_reasoning_effort)
                .transpose()
                .map_err(RequestRejection::Unsupported)?
                .flatten(),
            ..CreateChatCompletionRequest::default()
        },
        unmodeled: responses_request.unmodeled,
    };
    Ok((
        request,
        ServerToolClaims::new(claimed_names).map_err(RequestRejection::malformed)?,
        responses_request.max_tool_calls,
    ))
}

/// Responses `text.format` -> CC `response_format`: same schema payload, differently nested.
/// `verbosity` has no CC counterpart and is refused, never dropped.
fn cc_response_format(
    text: ResponseTextParam,
) -> Result<dynamo_protocols::types::ResponseFormat, String> {
    if text.verbosity.is_some() {
        return Err("`text.verbosity` is not supported".to_string());
    }
    Ok(match text.format {
        TextResponseFormatConfiguration::Text => dynamo_protocols::types::ResponseFormat::Text,
        TextResponseFormatConfiguration::JsonObject => {
            dynamo_protocols::types::ResponseFormat::JsonObject
        }
        TextResponseFormatConfiguration::JsonSchema(json_schema) => {
            dynamo_protocols::types::ResponseFormat::JsonSchema { json_schema }
        }
    })
}

/// Responses `reasoning.effort` -> CC `reasoning_effort` (one shared enum, different field name).
/// `summary` has no CC counterpart: TB renders reasoning as item content, not summary parts, so
/// `auto` is satisfied by that, and `concise`/`detailed` are refused rather than silently dropped.
fn cc_reasoning_effort(
    reasoning: ResponsesReasoning,
) -> Result<Option<dynamo_protocols::types::ReasoningEffort>, String> {
    match reasoning.summary {
        None | Some(ReasoningSummary::Auto) => {}
        Some(demanded) => {
            return Err(format!(
                "`reasoning.summary` {demanded:?} is not supported: tool-bank streams reasoning as \
                 item content, not as summary parts. Set `model_reasoning_summary = \"none\"`."
            ));
        }
    }
    Ok(reasoning.effort)
}

/// Stateless-only stance: TB persists no response, so continuing from one would silently produce a
/// self-contained answer instead of the multi-turn behavior the caller asked for.
fn reject_stateful_fields(request: &ResponsesRequest) -> Result<(), RequestRejection> {
    if request.store == Some(true) {
        return Err(RequestRejection::Unsupported(
            "`store: true` is not supported: tool-bank is stateless".to_string(),
        ));
    }
    if request.previous_response_id.is_some() {
        return Err(RequestRejection::Unsupported(
            "`previous_response_id` is not supported: tool-bank is stateless, echo the previous \
             response's `output` back as `input` instead"
                .to_string(),
        ));
    }
    if request.conversation.is_some() {
        return Err(RequestRejection::Unsupported(
            "`conversation` is not supported: tool-bank is stateless".to_string(),
        ));
    }
    if request.background == Some(true) {
        return Err(RequestRejection::Unsupported(
            "`background: true` is not supported: tool-bank is stateless".to_string(),
        ));
    }
    if request.prompt.is_some() {
        return Err(RequestRejection::Unsupported(
            "`prompt` is not supported: tool-bank stores no prompt templates, send the full \
             prompt as `input`"
                .to_string(),
        ));
    }
    Ok(())
}

/// Translate the whole `input` item list into canonical CC messages. Items arrive flat (no
/// per-turn wrapper the way Anthropic messages group blocks), so one [`AssistantMessageBuffer`]
/// accumulates across a run of assistant-owned items (reasoning, text, open tool calls) and flushes
/// whenever a non-assistant item interrupts it — same split-at-boundary technique
/// [`translate_message`] uses per Anthropic message, just flattened over the whole list.
fn translate_input_items(
    items: &[InputItem],
    cc_messages: &mut Vec<CcMessage>,
) -> Result<(), String> {
    let mut turn = AssistantMessageBuffer::default();
    for item in items {
        match item {
            InputItem::EasyMessage(easy) => {
                turn.flush_into(cc_messages);
                cc_messages.push(easy_message(easy)?);
            }
            InputItem::Item(Item::Message(MessageItem::Input(message))) => {
                turn.flush_into(cc_messages);
                cc_messages.push(input_message(message)?);
            }
            InputItem::Item(Item::Message(MessageItem::Output(message))) => {
                for content in &message.content {
                    match content {
                        dynamo_protocols::types::responses::InputOutputMessageContent::OutputText(text) => {
                            turn.text.push_str(&text.text);
                        }
                        dynamo_protocols::types::responses::InputOutputMessageContent::Refusal(_) => {
                            return Err("unsupported input item: assistant refusal content".to_string());
                        }
                    }
                }
            }
            InputItem::Item(Item::Reasoning(reasoning)) => {
                for ReasoningItemContent::ReasoningText(content) in
                    reasoning.content.iter().flatten()
                {
                    turn.thinking.push_str(&content.text);
                }
                // OpenAI/Codex transcripts carry `summary` (TB emits `content`); fold both.
                for SummaryPart::SummaryText(summary) in &reasoning.summary {
                    turn.thinking.push_str(&summary.text);
                }
            }
            InputItem::Item(Item::FunctionCall(call)) => {
                turn.tool_calls.push(responses_echoed_tool_call(
                    &call.call_id,
                    &call.name,
                    &call.arguments,
                )?);
            }
            InputItem::Item(Item::McpCall(call)) => {
                turn.tool_calls.push(responses_echoed_tool_call(
                    &call.id,
                    &call.name,
                    &call.arguments,
                )?);
                turn.flush_into(cc_messages);
                let result = if let Some(error) = &call.error {
                    error.clone()
                } else {
                    call.output.clone().unwrap_or_default()
                };
                cc_messages.push(tool_result_message(
                    call.id.clone(),
                    ChatCompletionRequestToolMessageContent::Text(result),
                ));
            }
            InputItem::Item(Item::FunctionCallOutput(output)) => {
                turn.flush_into(cc_messages);
                cc_messages.push(tool_result_message(
                    output.call_id.clone(),
                    ChatCompletionRequestToolMessageContent::Text(function_call_output_text(
                        &output.output,
                    )?),
                ));
            }
            InputItem::ItemReference(_) => {
                return Err(
                    "`item_reference` input items are not supported: tool-bank is stateless, it \
                     has no stored item to resolve one against"
                        .to_string(),
                );
            }
            InputItem::Item(other) => {
                // Bounded: the item embeds arbitrary client content, whole-`Debug` echoes it back.
                let item_debug = crate::util::truncate(&format!("{other:?}"), 120);
                return Err(format!("unsupported input item type: {item_debug}"));
            }
        }
    }
    turn.flush_into(cc_messages);
    Ok(())
}

fn easy_message(
    easy: &dynamo_protocols::types::responses::EasyInputMessage,
) -> Result<CcMessage, String> {
    let text = match &easy.content {
        EasyInputContent::Text(text) => text.clone(),
        EasyInputContent::ContentList(parts) => flatten_input_content(parts)?,
    };
    responses_role_message(easy.role, text)
}

fn input_message(
    message: &dynamo_protocols::types::responses::InputMessage,
) -> Result<CcMessage, String> {
    let text = flatten_input_content(&message.content)?;
    let role = match message.role {
        InputRole::User => ResponsesRole::User,
        InputRole::System => ResponsesRole::System,
        InputRole::Developer => ResponsesRole::Developer,
    };
    responses_role_message(role, text)
}

/// `developer` (Responses' name for the same instruction-channel role CC/Anthropic call `system`)
/// maps onto CC's `system` role — there is no separate CC role for it.
fn responses_role_message(role: ResponsesRole, text: String) -> Result<CcMessage, String> {
    Ok(match role {
        ResponsesRole::User => CcMessage::User(ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Text(text),
            name: None,
        }),
        ResponsesRole::System | ResponsesRole::Developer => {
            CcMessage::System(ChatCompletionRequestSystemMessage {
                content: Some(ChatCompletionRequestSystemMessageContent::Text(text)),
                name: None,
                tools: None,
            })
        }
        ResponsesRole::Assistant => CcMessage::Assistant(
            AssistantMessageBuffer {
                text,
                ..AssistantMessageBuffer::default()
            }
            .into_assistant_message()
            .ok_or("assistant message has empty content")?,
        ),
    })
}

fn unsupported_input_content_part(kind: &str) -> String {
    format!("unsupported input content part type: {kind}")
}

/// Text-only content parts, concatenated. An image/file part is refused, never dropped — same
/// stance `unsupported_block` takes for an untranslatable Anthropic block.
fn flatten_input_content(parts: &[InputContent]) -> Result<String, String> {
    let mut text = String::new();
    for part in parts {
        match part {
            InputContent::InputText(part) => text.push_str(&part.text),
            InputContent::InputImage(_) => {
                return Err(unsupported_input_content_part("input_image"));
            }
            InputContent::InputFile(_) => return Err(unsupported_input_content_part("input_file")),
        }
    }
    Ok(text)
}

/// A `function_call_output`'s output as the single text form the history append shares: a
/// structured part list is accepted only if every part is text.
fn function_call_output_text(output: &FunctionCallOutput) -> Result<String, String> {
    use dynamo_protocols::types::responses::UpstreamInputContent;
    match output {
        FunctionCallOutput::Text(text) => Ok(text.clone()),
        // Carries upstream's original `InputContent`, not the Dynamo-relaxed shadow — same variants.
        FunctionCallOutput::Content(parts) => parts
            .iter()
            .map(|part| match part {
                UpstreamInputContent::InputText(part) => Ok(part.text.clone()),
                UpstreamInputContent::InputImage(_) => {
                    Err(unsupported_input_content_part("input_image"))
                }
                UpstreamInputContent::InputFile(_) => {
                    Err(unsupported_input_content_part("input_file"))
                }
            })
            .collect(),
    }
}

/// An echoed `function_call`/`mcp_call` item back into the CC tool call TB originally sent. Unlike
/// Anthropic's `input: Value`, the Responses arguments are already the model's verbatim JSON
/// string, so there is no reserialize step to preserve byte-exactness for.
fn responses_echoed_tool_call(id: &str, name: &str, raw_args: &str) -> Result<ToolCall, String> {
    let args = serde_json::from_str(raw_args).map_err(|e| {
        format!("echoed call `{id}` has non-JSON `arguments` (the model never produces those): {e}")
    })?;
    Ok(ToolCall {
        id: id.to_string(),
        name: name.to_string(),
        raw_args: raw_args.to_string(),
        args,
    })
}

/// A caller-executed Responses tool definition -> a CC function tool. Any other declared tool
/// (`Tool::Mcp`, `Tool::WebSearch`, …) is refused: only the platform that natively runs it can, and
/// TB has no execution path for OpenAI's own built-in/remote tools.
fn responses_client_function_tool(
    tool_entry: Value,
) -> Result<ChatCompletionTool, RequestRejection> {
    let tool: ResponsesTool = serde_json::from_value(tool_entry)
        .map_err(|e| RequestRejection::malformed(format!("invalid tool definition: {e}")))?;
    let ResponsesTool::Function(function) = tool else {
        // Bounded: the entry embeds the client's full tool definition, whole-`Debug` echoes it back.
        let tool_debug = crate::util::truncate(&format!("{tool:?}"), 80);
        return Err(RequestRejection::Unsupported(format!(
            "unsupported tool type {tool_debug}: tool-bank only runs client function tools and \
             its own `{RESERVED_TOOL_PREFIX}*` server tools"
        )));
    };
    reject_reserved_client_tool(Some(&function.name))?;
    let mut cc_tool = function_tool(
        &function.name,
        function.description.as_deref().unwrap_or_default(),
        function.parameters.unwrap_or_else(|| json!({})),
    );
    cc_tool.function.strict = function.strict;
    Ok(cc_tool)
}

/// Responses' `tool_choice` is already function-call-shaped, so this is a narrower mapping than
/// Anthropic's `translate_tool_choice`: no `any`/`tool` naming mismatch to bridge.
fn translate_responses_tool_choice(
    tool_choice: &ToolChoiceParam,
) -> Result<ChatCompletionToolChoiceOption, String> {
    Ok(match tool_choice {
        ToolChoiceParam::Mode(ToolChoiceOptions::Auto) => ChatCompletionToolChoiceOption::Auto,
        ToolChoiceParam::Mode(ToolChoiceOptions::None) => ChatCompletionToolChoiceOption::None,
        ToolChoiceParam::Mode(ToolChoiceOptions::Required) => {
            ChatCompletionToolChoiceOption::Required
        }
        ToolChoiceParam::Function(ToolChoiceFunction { name }) => {
            ChatCompletionToolChoiceOption::Named(ChatCompletionNamedToolChoice {
                r#type: ChatCompletionToolType::Function,
                function: FunctionName { name: name.clone() },
            })
        }
        other => {
            return Err(format!(
                "unsupported tool_choice {other:?}: only client function tools and this \
                 endpoint's own server tools can be chosen, and the model chooses freely among them"
            ));
        }
    })
}

// --- Shared helpers ---------------------------------------------------------

/// Split a typed protocol's declared tools: a server-tool-shaped entry (reserved `baseten__*`
/// type, or the protocol's own hosted-tool shapes) goes to the hooks; anything else translates via
/// the protocol's client-tool constructor. (CC keeps its own loop — it forwards client entries
/// verbatim as untyped JSON instead of translating them.)
fn expand_declared_tools(
    hooks: &mut dyn IngressHooks,
    dropped: &mut DroppedTools,
    protocol: ClientProtocol,
    tool_entries: Vec<Value>,
    is_server_tool_shaped: fn(&Value) -> bool,
    client_function_tool: fn(Value) -> Result<ChatCompletionTool, RequestRejection>,
) -> Result<(Vec<ChatCompletionTool>, Vec<String>), RequestRejection> {
    let mut cc_tools = Vec::with_capacity(tool_entries.len());
    let mut claimed_names = Vec::new();
    for tool_entry in tool_entries {
        let tool_type = tool_entry.get("type").and_then(Value::as_str);
        if is_reserved_tool_type(tool_type) || is_server_tool_shaped(&tool_entry) {
            match hooks.on_server_tool(protocol, &tool_entry) {
                ToolDisposition::Claim(claim) => {
                    cc_tools.push(claim.tool);
                    claimed_names.push(claim.name);
                }
                ToolDisposition::Drop => dropped.record(protocol, &tool_entry),
                ToolDisposition::Reject(rejection) => return Err(rejection),
            }
        } else {
            cc_tools.push(client_function_tool(tool_entry)?);
        }
    }
    Ok((cc_tools, claimed_names))
}

/// An Anthropic tool entry only Anthropic (or a hook) can execute: any versioned non-`custom`
/// `type` (`web_search_20250305`, `code_execution_*`, ...). A plain caller-executed tool has
/// `type: "custom"` or no `type` at all.
fn is_anthropic_server_tool_shaped(entry: &Value) -> bool {
    entry
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| kind != "custom")
}

/// A Responses tool entry naming an OpenAI-hosted execution surface. Anything else non-`function`
/// (`custom`, `namespace`, ...) still reaches the client-tool constructor, whose refusal names it.
fn is_responses_server_tool_shaped(entry: &Value) -> bool {
    entry
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| {
            kind.starts_with("web_search")
                || matches!(
                    kind,
                    "mcp"
                        | "file_search"
                        | "code_interpreter"
                        | "computer_use_preview"
                        | "image_generation"
                        | "local_shell"
                )
        })
}

fn function_tool(name: &str, description: &str, parameters: Value) -> ChatCompletionTool {
    ChatCompletionTool {
        r#type: ChatCompletionToolType::Function,
        function: FunctionObject {
            name: name.to_string(),
            description: Some(description.to_string()),
            parameters: Some(parameters),
            strict: None,
        },
    }
}

fn reject_reserved_client_tool(name: Option<&str>) -> Result<(), RequestRejection> {
    match name {
        Some(name) if name.starts_with(RESERVED_TOOL_PREFIX) => Err(RequestRejection::malformed(
            format!("client tool name {name:?} uses reserved namespace `{RESERVED_TOOL_PREFIX}`"),
        )),
        _ => Ok(()),
    }
}

/// Flatten Anthropic string|block-array content to plain text (the CC system/tool message form).
fn flatten_text(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .map(|block| {
                block
                    .get("text")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .or_else(|| block.as_str().map(str::to_string))
                    // A non-text block (e.g. structured tool output) is preserved as JSON, not dropped.
                    .unwrap_or_else(|| block.to_string())
            })
            .collect::<Vec<_>>()
            .join(""),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// The config value is a ceiling a request may lower; beyond it is a 400, since a silent clamp
/// yields a truncated answer that reads as complete.
fn tool_calls_within_ceiling(
    max_tool_calls_per_iteration: NonZeroU32,
    requested: Option<NonZeroU32>,
) -> Result<NonZeroU32, RequestRejection> {
    let Some(requested) = requested else {
        return Ok(max_tool_calls_per_iteration);
    };
    if requested > max_tool_calls_per_iteration {
        return Err(RequestRejection::malformed(format!(
            "baseten.tool_settings.max_tool_calls_per_iteration must be <= {max_tool_calls_per_iteration} (got {requested})"
        )));
    }
    Ok(requested)
}

/// Whether the request can reach a server tool at all: one was claimed and calling isn't forbidden.
pub fn can_dispatch_server_tool(
    tool_choice: Option<&ChatCompletionToolChoiceOption>,
    server_tool_claims: &ServerToolClaims,
) -> bool {
    !server_tool_claims.is_empty()
        && !matches!(tool_choice, Some(ChatCompletionToolChoiceOption::None))
}

/// A 400 outside the bounds, never a clamp: clamping up spends a budget the caller did not ask for,
/// clamping down yields a truncated answer that reads as complete. Only ever the caller's own
/// `baseten.tool_settings.max_react_iterations`, so the message names the field they wrote; a bound
/// translated from another axis is clamped at the call site instead.
fn react_iterations_within_bounds(
    num_default_react_iterations: NonZeroU32,
    requested: Option<NonZeroU32>,
    iterations_floor: NonZeroU32,
) -> Result<(NonZeroU32, ReactCapSource), RequestRejection> {
    if requested
        .is_some_and(|requested| !(iterations_floor..=REACT_ITERATIONS_MAX).contains(&requested))
    {
        return Err(RequestRejection::malformed(format!(
            "baseten.tool_settings.max_react_iterations must be within \
             {iterations_floor}..={REACT_ITERATIONS_MAX} (got {})",
            requested.expect("checked just above")
        )));
    }
    let effective = requested.unwrap_or(num_default_react_iterations);
    let cap_source = if effective == REACT_ITERATIONS_MAX {
        ReactCapSource::ServiceCeiling
    } else if requested.is_some() {
        ReactCapSource::Request
    } else {
        ReactCapSource::ServerDefault
    };
    Ok((effective, cap_source))
}

#[cfg(test)]
#[path = "request_test.rs"]
mod tests;
