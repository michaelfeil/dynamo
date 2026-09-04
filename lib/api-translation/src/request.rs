//! Request edge: client request bytes -> canonical typed [`CcRequest`] (+ server-tool claims).
//!
//! Two protocol flavors converge on one typed CC request. Both normalize as JSON first —
//! server-tool entry handling *must* run pre-typing (a hosted-tool entry isn't a valid CC
//! tool) — then finalize once via `serde_json::from_value::<CcRequest>`, so everything downstream
//! (accumulator, `build_next_request`, predict body) is typed + validated.
//!
//! - CC-native: near-identity — route non-`function` tool entries through the hooks.
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
    AnthropicRole, AnthropicTool, AnthropicToolChoice, AnthropicToolChoiceMode, DocumentBlock,
    DocumentSource, SearchResultBlock, ThinkingConfig, ToolResultContent, ToolResultContentBlock,
};
use dynamo_protocols::types::responses::{
    AgentMessageInputContent, AgentMessageItemParam, EasyInputContent, FunctionCallOutput,
    InputContent, InputItem, InputRole, Item, MessageItem, ReasoningItemContent, ResponseTextParam,
    Role as ResponsesRole, ServiceTier as ResponsesServiceTier, SummaryPart,
    TextResponseFormatConfiguration, Tool as ResponsesTool, ToolChoiceFunction, ToolChoiceOptions,
    ToolChoiceParam,
};
use dynamo_protocols::types::{
    ChatCompletionNamedToolChoice, ChatCompletionRequestMessageContentPartImage,
    ChatCompletionRequestMessageContentPartText, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestToolMessageContent,
    ChatCompletionRequestToolMessageContentPart, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
    ChatCompletionStreamOptions, ChatCompletionTool, ChatCompletionToolChoiceOption,
    ChatCompletionToolType, CreateChatCompletionRequest, FunctionName, FunctionObject, ImageUrl,
    ServiceTier as CcServiceTier, Stop,
};

use crate::coding_adapter::CodingAdapter;
use crate::framing::ResponsesParams;
use crate::history::{AssistantMessageBuffer, tool_result_message};
use crate::hooks::{IngressHooks, ToolDisposition};
use crate::loss::{Loss, LossKind, Losses};
use crate::model::{ReactCapSource, RequestRejection, ToolCall};
use crate::{
    CcMessage, CcRequest, ClientProtocol, IMAGE_SOURCE_TYPE_BASE64, SERVER_TOOL_USE_ID_PREFIX,
};

/// The reserved `baseten` request-body object — Baseten's whole extension namespace on top of the
/// OpenAI/Anthropic request schemas. Absent means all defaults.
#[derive(Debug, Default, Deserialize)]
struct BasetenRequestExtension {
    #[serde(default)]
    pub tool_settings: ToolSettings,
    /// Members this crate doesn't model. The `baseten` object must never
    /// reach the model and the pre-adaptation strip has already removed it,
    /// so unknown members are warned about and dropped — never a 400 (fork
    /// deployments historically carried them harmlessly).
    #[serde(flatten)]
    pub unmodeled: serde_json::Map<String, Value>,
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
#[derive(Debug)]
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
    /// What adaptation did not carry through (each also handed to
    /// [`IngressHooks::on_loss`] and counted on the ingress stage line). Empty is the common case.
    pub losses: Vec<Loss>,
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

impl std::fmt::Debug for AdaptedIngress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `CodingAdapter` is a runtime object with no Debug bound; presence suffices.
        f.debug_struct("AdaptedIngress")
            .field("request", &self.request)
            .field("coding_adapter", &self.coding_adapter.is_some())
            .field("responses_params", &self.responses_params)
            .finish()
    }
}

/// Tools removed by a `Drop` disposition, for the tool_choice degrade below.
#[derive(Default)]
struct DroppedTools {
    names: Vec<String>,
    any: bool,
    /// Everything adaptation dropped, skipped, or degraded — server tools included.
    losses: Losses,
}

impl DroppedTools {
    fn record(&mut self, protocol: ClientProtocol, entry: &Value) {
        self.any = true;
        let position = self.names.len();
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
        self.losses.record(
            LossKind::ServerToolDropped,
            format!("tools[{position}]"),
            format!(
                "dropped server-tool-shaped tool type={kind:?} name={:?}: nothing on this endpoint executes it",
                name.unwrap_or("")
            ),
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
    let body_json: Value = serde_json::from_slice(body)
        .map_err(|e| RequestRejection::malformed(format!("invalid JSON body: {e}")))?;
    adapt_request_json(body_json, protocol, headers, hooks)
}

/// [`adapt_request`] over an already-parsed body. A frontend that has parsed the client's bytes
/// once (to validate them) hands the parsed object straight in, so this layer sees exactly what
/// the client sent — never a typed struct re-serialized through its own (possibly lossy) `Serialize`.
pub fn adapt_request_json(
    body_json: Value,
    protocol: ClientProtocol,
    headers: &HeaderMap,
    hooks: &mut dyn IngressHooks,
) -> Result<AdaptedIngress, RequestRejection> {
    let started = std::time::Instant::now();
    let result = adapt_request_json_inner(body_json, protocol, headers, hooks);
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    // The canonical per-stage line: one per request for the ingress stage, structured and
    // joinable on the caller's request-id span field. Model-supplied strings stay out of it;
    // `outcome`/`error_class`/loss kinds are closed vocabularies.
    match &result {
        Ok(adapted) => tracing::info!(
            target: "http",
            event_name = "stage.ingress",
            stage = "ingress",
            protocol = ?protocol,
            outcome = "ok",
            elapsed_ms,
            model = %adapted.request.request.model,
            messages = adapted.request.request.messages.len(),
            server_tools = %adapted.request.server_tool_claims.joined(),
            max_react_iterations = adapted.request.max_react_iterations.get(),
            losses = adapted.request.losses.len(),
            loss_kinds = %crate::loss::count_by_kind(&adapted.request.losses)
                .iter()
                .map(|(kind, n)| format!("{kind}={n}"))
                .collect::<Vec<_>>()
                .join(","),
            "ingress adapted"
        ),
        Err(rejection) => tracing::info!(
            target: "http",
            event_name = "stage.ingress",
            stage = "ingress",
            protocol = ?protocol,
            outcome = "rejected",
            error_class = rejection.class_label(),
            status = rejection.status().as_u16(),
            elapsed_ms,
            "ingress rejected: {}",
            rejection.detail()
        ),
    }
    result
}

fn adapt_request_json_inner(
    mut body_json: Value,
    protocol: ClientProtocol,
    headers: &HeaderMap,
    hooks: &mut dyn IngressHooks,
) -> Result<AdaptedIngress, RequestRejection> {
    let limits = hooks.limits();
    let mut dropped = DroppedTools::default();
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
    // tool-bank's original guard: a top-level `tool_settings` is a nesting mistake, and letting it
    // ride the catch-all would forward it to the model while the caller's caps silently default.
    if body_json
        .as_object()
        .is_some_and(|fields| fields.contains_key("tool_settings"))
    {
        return Err(RequestRejection::malformed(
            "`tool_settings` must be nested as `baseten.tool_settings`",
        ));
    }
    for key in baseten_extension.unmodeled.keys() {
        dropped.losses.record(
            LossKind::ExtensionFieldDropped,
            format!("baseten.{key}"),
            "`baseten` request extension member this crate does not model; dropped",
        );
    }
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
        dropped.losses.record(
            LossKind::ToolChoiceDegraded,
            "tool_choice",
            format!(
                "tool_choice named dropped server tool {:?}; degraded to auto",
                named.function.name
            ),
        );
        request.tool_choice = Some(ChatCompletionToolChoiceOption::Auto);
    }
    // After adaptation: a single iteration can only ever end on a dispatched call, so a request that
    // can reach a server tool needs two.
    debug_assert!(
        limits.server_tool_iterations_floor <= limits.max_react_iterations
            && limits.default_react_iterations <= limits.max_react_iterations,
        "IngressLimits out of order: {limits:?}"
    );
    let ceiling = limits.max_react_iterations;
    let iterations_floor =
        if can_dispatch_server_tool(request.tool_choice.as_ref(), &server_tool_claims) {
            limits.server_tool_iterations_floor
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
                 {iterations_floor}..={ceiling})"
            )));
        }
        // Clamped, not refused: `max_uses` counts searches, an axis whose bounds the client had no
        // way to aim at ours, so a value outside them is not a caller mistake to report.
        (Some(translated), None) => {
            let clamped = translated.clamp(iterations_floor, ceiling);
            let react_cap_source = if clamped == ceiling {
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
            ceiling,
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

    let losses = dropped.losses.into_vec();
    for loss in &losses {
        hooks.on_loss(loss);
    }

    Ok(AdaptedIngress {
        request: AdaptedRequest {
            request,
            server_tool_claims,
            max_react_iterations,
            react_cap_source,
            max_tool_calls_per_iteration,
            max_tool_calls,
            should_stream,
            losses,
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

/// The one field of a CC `tools` entry that ingress routing needs; the entry itself is forwarded
/// verbatim (client tool) or replaced (server tool claim).
#[derive(Deserialize)]
struct CcToolEntryHead {
    #[serde(rename = "type")]
    tool_type: Option<String>,
}

/// CC client is already canonical: route non-`function` tool entries through the hooks (a claim
/// becomes a function tool; the default drops them with a warning). Messages are forwarded
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
            // Shape-only routing: a CC tool entry whose `type` is not `function` is a hosted
            // tool no engine can execute (`web_search_preview`, a tool-bank selection, ...).
            if tool_entry_head
                .tool_type
                .as_deref()
                .is_some_and(|kind| kind != "function")
            {
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
                cc_tools.push(tool_entry);
            }
        }
        *entries = cc_tools;
    }
    let request: CcRequest = parse_cc_body(cc_body)?;
    // `None`: only Responses models `max_tool_calls`.
    Ok((
        request,
        ServerToolClaims::new(claimed_names).map_err(RequestRejection::malformed)?,
        None,
    ))
}

/// A typed parse of client JSON. Routed through `serde_path_to_error` so the rejection names the
/// offending member: serde omits the field name, and an untagged enum discards the inner error.
/// `enclosing_member` is `json`'s path within the body, empty for the body itself.
fn parse_client_json<T: serde::de::DeserializeOwned>(
    protocol: ClientProtocol,
    enclosing_member: &str,
    json: Value,
) -> Result<T, RequestRejection> {
    let protocol = match protocol {
        ClientProtocol::ChatCompletions => "ChatCompletions",
        ClientProtocol::Messages => "Messages",
        ClientProtocol::Responses => "Responses",
    };
    serde_path_to_error::deserialize(json).map_err(|e| {
        // serde_path_to_error reports "." for a failure at the root of what it was handed.
        let inner_member = match e.path().to_string() {
            root if root.is_empty() || root == "." => String::new(),
            member if enclosing_member.is_empty() => member,
            member => format!(".{member}"),
        };
        let member = format!("{enclosing_member}{inner_member}");
        let cause = e.into_inner();
        if member.is_empty() {
            RequestRejection::malformed(format!("invalid {protocol} request: {cause}"))
        } else {
            RequestRejection::malformed(format!(
                "invalid {protocol} request at `{member}`: {cause}"
            ))
        }
    })
}

/// `CcRequest` flattens the typed request and the unmodeled catch-all, and serde buffers flattened
/// content, so a path-tracking parse of it reports the root. On failure only, the typed wire struct
/// is parsed once more on its own to name the member; a success there means the failure came from
/// the catch-all, and the original error stands.
fn parse_cc_body(cc_body: Value) -> Result<CcRequest, RequestRejection> {
    match <CcRequest as Deserialize>::deserialize(&cc_body) {
        Ok(request) => Ok(request),
        Err(flattened) => {
            parse_client_body::<dynamo_protocols::types::CreateChatCompletionRequest>(
                ClientProtocol::ChatCompletions,
                cc_body,
            )?;
            Err(RequestRejection::malformed(format!(
                "invalid ChatCompletions request: {flattened}"
            )))
        }
    }
}

fn parse_client_body<T: serde::de::DeserializeOwned>(
    protocol: ClientProtocol,
    body: Value,
) -> Result<T, RequestRejection> {
    parse_client_json(protocol, "", body)
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
    /// Anthropic `auto|standard_only`: a capacity-tier hint with no CC meaning. Dropped, never
    /// forwarded — on the CC wire `service_tier` is a different enum and the reparse would 400.
    service_tier: Option<Value>,
    /// Anthropic's top-level automatic-caching `cache_control`. Dropped, never translated: the
    /// fork's own `cache_control` extension on the CC wrapper is a list of ranges with different
    /// semantics, and the Anthropic object shape 400s its parser.
    cache_control: Option<Value>,
    /// Every other field this layer doesn't model. Only the fork's CC extension surface
    /// ([`CC_EXTENSION_KEYS`]) rides through onto the CC body; any other Anthropic-spelled key is
    /// unknown on the model's CC endpoint and is dropped with a warning (standard-dynamo behavior).
    #[serde(flatten)]
    unmodeled: serde_json::Map<String, Value>,
}

/// Top-level request keys with a typed home on the fork's Chat Completions wrapper
/// (`NvCreateChatCompletionRequest`: its `baseten_ext`, `common`, `nvext` and template-args
/// fields). A Messages client may carry these untranslated onto the CC body; every other
/// unmodeled key is dropped. `top_k` is Anthropic's own sampling field, which the fork's CC
/// wrapper accepts under the same name.
pub const CC_EXTENSION_KEYS: &[&str] = &[
    "chat_template_kwargs",
    "chat_template_args",
    "nvext",
    "allowed_worker_ids",
    "decode_cache_control",
    "dynamic_temperature",
    "priority",
    "reasoning",
    "thinking_token_budget",
    "mocker_config",
    "media_io_kwargs",
    "return_tokens_as_token_ids",
    "ignore_eos",
    "min_tokens",
    "top_k",
    "min_p",
    "repetition_penalty",
];

/// Standard-dynamo drop: an Anthropic-dialect key with no CC meaning (Claude Code's
/// `context_management`, `metadata` remainders, ...) must not reach the CC body, where the
/// deployment's strict reparse would 400 the whole request. Warned, so a silently ignored
/// capability is visible in the logs.
fn drop_non_extension_fields(unmodeled: &mut serde_json::Map<String, Value>, losses: &mut Losses) {
    let dropped: Vec<String> = unmodeled
        .keys()
        .filter(|key| !CC_EXTENSION_KEYS.contains(&key.as_str()))
        .cloned()
        .collect();
    if dropped.is_empty() {
        return;
    }
    for key in &dropped {
        unmodeled.remove(key);
        losses.record(
            LossKind::RequestFieldDropped,
            key.as_str(),
            "Messages request field with no Chat Completions meaning; dropped",
        );
    }
}

/// Translate the whole Messages request into the canonical typed CC request.
fn adapt_messages(
    hooks: &mut dyn IngressHooks,
    dropped: &mut DroppedTools,
    body: Value,
) -> Result<(CcRequest, ServerToolClaims, Option<NonZeroU32>), RequestRejection> {
    let messages_request: MessagesRequest = parse_client_body(ClientProtocol::Messages, body)?;
    // Anthropic-hosted features this endpoint cannot honor: tool-bank refuses them, standard
    // dynamo (whose previous converter ignored them) drops them with a warning.
    for (field, present) in [
        ("mcp_servers", messages_request.mcp_servers.is_some()),
        ("container", messages_request.container.is_some()),
    ] {
        if !present {
            continue;
        }
        if hooks.rejects_unsupported_messages_features() {
            return Err(RequestRejection::Unsupported(format!(
                "`{field}` is not supported: this endpoint connects to no caller-supplied MCP \
                 servers and runs no code-execution containers"
            )));
        }
        dropped.losses.record(
            LossKind::RequestFieldDropped,
            field,
            "Anthropic-hosted feature this endpoint cannot honor; dropped",
        );
    }
    // Anthropic ranges: temperature and top_p are 0..1 on this wire (the OpenAI surfaces allow
    // temperature up to 2). Fork conformance floor — Anthropic itself 400s these.
    if let Some(t) = messages_request.temperature
        && (!t.is_finite() || !(0.0..=1.0).contains(&t))
    {
        return Err(RequestRejection::malformed(format!(
            "temperature must be between 0 and 1, got {t}"
        )));
    }
    if let Some(p) = messages_request.top_p
        && (!p.is_finite() || !(0.0..=1.0).contains(&p))
    {
        return Err(RequestRejection::malformed(format!(
            "top_p must be between 0 and 1, got {p}"
        )));
    }
    // Anthropic bounds the manual thinking budget: >= 1024 and strictly less than max_tokens.
    // (Budget with a non-`enabled` mode is refused in `parse_messages_thinking`.)
    if let Some(thinking) = &messages_request.thinking
        && thinking.thinking_type == "enabled"
        && let Some(budget) = thinking.budget_tokens
    {
        if budget < 1024 {
            return Err(RequestRejection::malformed(format!(
                "thinking.budget_tokens must be at least 1024, got {budget}"
            )));
        }
        // Anthropic requires `max_tokens`; the typed request keeps it optional (the deployment's
        // template may supply it), so the upper bound applies only when the client sent one.
        if let Some(max_tokens) = messages_request.max_tokens
            && budget >= max_tokens
        {
            return Err(RequestRejection::malformed(format!(
                "thinking.budget_tokens ({budget}) must be less than max_tokens ({max_tokens})"
            )));
        }
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
    let strict = hooks.rejects_unsupported_messages_features();
    for (message_index, message) in messages_request.messages.iter().enumerate() {
        translate_message(
            message,
            message_index,
            &mut cc_messages,
            strict,
            &mut dropped.losses,
        )?;
    }
    // Anthropic prefill semantics: a trailing assistant message asks the model to CONTINUE it, not
    // answer fresh. Mark the lowered trailing CC assistant message `partial` (the Kimi-style
    // prefill marker on the fork's assistant type); the render layer continues the final message.
    // Skipped when the assistant turn ended on a tool_result (the trailing CC message is `tool`,
    // not an open assistant channel to continue).
    if messages_request
        .messages
        .last()
        .is_some_and(|message| message.role == AnthropicRole::Assistant)
        && let Some(CcMessage::Assistant(assistant)) = cc_messages.last_mut()
    {
        assistant.partial = Some(true);
    }
    // Post-normalization (after `srvtoolu_` prefix stripping) so replayed server-tool pairs still
    // match; over the translated CC messages so it covers the whole transcript.
    validate_tool_results_have_tool_use(&cc_messages)?;

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
    if messages_request.service_tier.is_some() {
        dropped.losses.record(
            LossKind::RequestFieldDropped,
            "service_tier",
            "Anthropic service_tier has no Chat Completions meaning; dropped",
        );
    }
    if messages_request.cache_control.is_some() {
        dropped.losses.record(
            LossKind::RequestFieldDropped,
            "cache_control",
            "Anthropic top-level cache_control is not translated to the fork's range-based cache_control extension; dropped",
        );
    }
    let mut unmodeled = unmodeled_fields(messages_request.unmodeled, messages_request.thinking)
        .map_err(RequestRejection::Malformed)?;
    // `user` / `metadata.user_id` lift onto the modeled field first, then whatever is left that
    // the CC wrapper has no home for is dropped.
    let user = take_body_user(&mut unmodeled);
    drop_non_extension_fields(&mut unmodeled, &mut dropped.losses);
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
            user,
            ..CreateChatCompletionRequest::default()
        },
        unmodeled,
    };
    // `None`: only Responses models `max_tool_calls`.
    Ok((
        request,
        ServerToolClaims::new(claimed_names).map_err(RequestRejection::malformed)?,
        None,
    ))
}

/// Downstream keys sticky routing off a body `user` (Anthropic spelling: `metadata.user_id`), so
/// both spellings move onto the modeled field — left in the map they'd serialize a second,
/// driftable `user` key. An out-of-spec top-level `user` wins: downstream reads exactly it.
fn take_body_user(unmodeled: &mut serde_json::Map<String, Value>) -> Option<String> {
    if let Some(client_user) = take_string(unmodeled, "user") {
        return Some(client_user);
    }
    let Some(Value::Object(metadata)) = unmodeled.get_mut("metadata") else {
        return None;
    };
    let user_id = take_string(metadata, "user_id");
    if metadata.is_empty() {
        unmodeled.remove("metadata");
    }
    user_id
}

/// Removes `key` only when it holds a string; any other shape stays in place, since a non-string
/// identity is nothing we can translate onto a typed field.
fn take_string(fields: &mut serde_json::Map<String, Value>, key: &str) -> Option<String> {
    match fields.remove(key) {
        Some(Value::String(value)) => Some(value),
        Some(untranslatable) => {
            fields.insert(key.to_owned(), untranslatable);
            None
        }
        None => None,
    }
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
fn client_function_tool(tool_entry: Value) -> Result<Vec<ChatCompletionTool>, RequestRejection> {
    let client_tool: AnthropicTool = serde_json::from_value(tool_entry)
        .map_err(|e| RequestRejection::malformed(format!("invalid tool definition: {e}")))?;
    if let Some(tool_type) = client_tool.tool_type.as_deref().filter(|t| *t != "custom") {
        // Malformed, not Unsupported: only Anthropic can execute these, so it is the caller
        // pointing an Anthropic-hosted tool at a non-Anthropic endpoint, not a TB gap to close.
        return Err(RequestRejection::malformed(format!(
            "tool {:?} has type {tool_type:?}: Anthropic-native server tools are not supported \
             on this endpoint",
            client_tool.name
        )));
    }
    // Anthropic requires input_schema on client tools; a tool without one has no callable shape to
    // advertise, so it is refused rather than defaulted (fork validation floor).
    let Some(input_schema) = client_tool.input_schema else {
        return Err(RequestRejection::malformed(format!(
            "tool {:?} is missing input_schema (required for client tools)",
            client_tool.name
        )));
    };
    Ok(vec![function_tool(
        &client_tool.name,
        client_tool.description.as_deref().unwrap_or_default(),
        input_schema,
    )])
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
    message_index: usize,
    cc_messages: &mut Vec<CcMessage>,
    strict_blocks: bool,
    losses: &mut Losses,
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
        for (block_index, block) in blocks.iter().enumerate() {
            match block {
                AnthropicContentBlock::Text { text, .. } => {
                    if !turn.text.is_empty() {
                        turn.text.push('\n');
                    }
                    turn.text.push_str(text);
                }
                AnthropicContentBlock::Thinking { thinking, .. } => {
                    turn.append_thinking_block(thinking);
                }
                AnthropicContentBlock::ToolUse {
                    id, name, input, ..
                } => {
                    turn.close_thinking_segment();
                    turn.tool_calls.push(echoed_tool_call(id, name, input));
                }
                AnthropicContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    ..
                } => {
                    turn.flush_into(cc_messages);
                    cc_messages.push(echoed_tool_result(
                        tool_use_id,
                        content.as_ref(),
                        &format!("messages[{message_index}].content[{block_index}]"),
                        losses,
                    )?);
                }
                AnthropicContentBlock::ServerToolUse { id, name, input } => {
                    turn.close_thinking_segment();
                    turn.tool_calls.push(echoed_tool_call(
                        &replayed_server_tool_use_id(id),
                        name,
                        input,
                    ));
                }
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
                // Fork parity: encrypted reasoning has no CC slot; skipped rather than refused —
                // agent clients (Claude Code) replay these blocks verbatim, so refusing would
                // fail real session replays.
                AnthropicContentBlock::RedactedThinking { .. } => {
                    losses.record(
                        LossKind::ContentBlockSkipped,
                        format!("messages[{message_index}].content[{block_index}]"),
                        "assistant redacted_thinking block skipped: encrypted, no Chat Completions representation",
                    );
                }
                // Fork parity: unknown assistant-authored blocks are model output echoed back;
                // skip rather than 400 so a new upstream block type never breaks replay.
                AnthropicContentBlock::Other(value) => {
                    let block_type = value
                        .get("type")
                        .and_then(|t| t.as_str())
                        .unwrap_or("(no type)");
                    losses.record(
                        LossKind::ContentBlockSkipped,
                        format!("messages[{message_index}].content[{block_index}]"),
                        format!("unknown assistant content block type={block_type:?} skipped"),
                    );
                }
                unsupported => return Err(unsupported_block("assistant", unsupported)),
            }
        }
        turn.flush_into(cc_messages);
        return Ok(());
    }

    // Text keeps its place before a following tool_result (byte-identical CC replay). An image
    // switches the pending run to multimodal parts (CC data-URI image parts) — user role only.
    let mut text = String::new();
    let mut parts: Vec<ChatCompletionRequestUserMessageContentPart> = Vec::new();
    let mut has_image = false;
    for (block_index, block) in blocks.iter().enumerate() {
        match block {
            AnthropicContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                flush_user_run(
                    &message.role,
                    &mut text,
                    &mut parts,
                    &mut has_image,
                    cc_messages,
                )?;
                cc_messages.push(echoed_tool_result(
                    tool_use_id,
                    content.as_ref(),
                    &format!("messages[{message_index}].content[{block_index}]"),
                    losses,
                )?);
            }
            AnthropicContentBlock::Text {
                text: block_text, ..
            } => {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(block_text);
                parts.push(ChatCompletionRequestUserMessageContentPart::Text(
                    ChatCompletionRequestMessageContentPartText {
                        text: block_text.clone(),
                    },
                ));
            }
            AnthropicContentBlock::Image { source } if message.role == AnthropicRole::User => {
                has_image = true;
                parts.push(ChatCompletionRequestUserMessageContentPart::ImageUrl(
                    image_url_part(source)?,
                ));
            }
            // Fork parity: a user `document` block (PDF/text attachment) has no CC translation on
            // this stack — the model cannot consume it — so it is dropped with a warning rather
            // than failing the request; the text beside it still reaches the model.
            AnthropicContentBlock::Other(value) if is_document_block(value) => {
                losses.record(
                    LossKind::ContentBlockSkipped,
                    format!("messages[{message_index}].content[{block_index}]"),
                    "user document block dropped: no Chat Completions representation on this stack",
                );
            }
            unsupported if strict_blocks => {
                return Err(unsupported_block(role_name(&message.role), unsupported));
            }
            // Standard dynamo (the previous converter's behavior): a block with no Chat
            // Completions translation is skipped with a warning; the rest of the message still
            // reaches the model. tool-bank refuses instead (`rejects_unsupported_messages_features`).
            unsupported => {
                let refusal = unsupported_block(role_name(&message.role), unsupported);
                losses.record(
                    LossKind::ContentBlockSkipped,
                    format!("messages[{message_index}].content[{block_index}]"),
                    format!(
                        "untranslatable user content block skipped: {}",
                        refusal.detail()
                    ),
                );
            }
        }
    }
    flush_user_run(
        &message.role,
        &mut text,
        &mut parts,
        &mut has_image,
        cc_messages,
    )?;
    Ok(())
}

fn is_document_block(value: &Value) -> bool {
    value.get("type").and_then(Value::as_str) == Some("document")
}

fn role_name(role: &AnthropicRole) -> &'static str {
    match role {
        AnthropicRole::User => "user",
        AnthropicRole::System => "system",
        AnthropicRole::Assistant => "assistant",
    }
}

/// Flush a pending run of user/system text (and, for the user role, image parts) as one CC
/// message. All-text collapses to the plain text form; any image emits the multimodal part array.
fn flush_user_run(
    role: &AnthropicRole,
    text: &mut String,
    parts: &mut Vec<ChatCompletionRequestUserMessageContentPart>,
    has_image: &mut bool,
    cc_messages: &mut Vec<CcMessage>,
) -> Result<(), RequestRejection> {
    let run_parts = std::mem::take(parts);
    let run_text = std::mem::take(text);
    if std::mem::take(has_image) {
        cc_messages.push(CcMessage::User(ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Array(run_parts),
            name: None,
        }));
    } else if !run_text.is_empty() {
        cc_messages.push(text_message(role, run_text)?);
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
    field: &str,
    losses: &mut Losses,
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
            .map(|content| tool_result_content(content, field, losses))
            .transpose()?
            .unwrap_or_default(),
    ))
}

fn tool_result_content(
    content: &ToolResultContent,
    field: &str,
    losses: &mut Losses,
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
                // Fork parity (#559): a document with no text representation degrades to a short
                // placeholder keeping the title/source pointer visible, never a request error —
                // Claude Code replays these inside tool results.
                let block_text = doc.text().unwrap_or_else(|| document_placeholder(doc));
                text.push_str(&block_text);
                parts.push(ChatCompletionRequestToolMessageContentPart::Text(
                    ChatCompletionRequestMessageContentPartText { text: block_text },
                ));
            }
            ToolResultContentBlock::SearchResult(result) => {
                let block_text = result
                    .text()
                    .unwrap_or_else(|| search_result_placeholder(result));
                text.push_str(&block_text);
                parts.push(ChatCompletionRequestToolMessageContentPart::Text(
                    ChatCompletionRequestMessageContentPartText { text: block_text },
                ));
            }
            ToolResultContentBlock::Other(value) => {
                // Fork parity: carried as coerced text, not refused — small blocks pass through
                // as compact JSON so the model keeps the pointers, oversized ones become a
                // one-line placeholder so base64 payloads can't balloon the prompt.
                let block_text = coerce_unknown_block(value, field, losses);
                text.push_str(&block_text);
                parts.push(ChatCompletionRequestToolMessageContentPart::Text(
                    ChatCompletionRequestMessageContentPartText { text: block_text },
                ));
            }
        }
    }
    Ok(if has_image {
        ChatCompletionRequestToolMessageContent::Array(parts)
    } else {
        ChatCompletionRequestToolMessageContent::Text(text)
    })
}

/// Ceiling for passing an unknown no-text block through as raw JSON. Above this the block becomes
/// a one-line placeholder so oversized payloads (e.g. base64 documents) can't balloon the prompt
/// (fork #559 semantics).
const UNKNOWN_BLOCK_JSON_LIMIT: usize = 1024;

/// Text stand-in for an unknown tool_result block: compact JSON when small, a one-line placeholder
/// above the size ceiling. Carried rather than skipped — silently dropping request content is the
/// failure mode this layer exists to end.
fn coerce_unknown_block(value: &Value, field: &str, losses: &mut Losses) -> String {
    let block_type = value
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("<missing>");
    let json = value.to_string();
    if json.len() <= UNKNOWN_BLOCK_JSON_LIMIT {
        tracing::debug!(
            "coercing unknown Anthropic tool_result content block to JSON text: type={block_type}"
        );
        json
    } else {
        losses.record(
            LossKind::ToolResultBlockOmitted,
            field,
            format!(
                "oversized unknown tool_result block type={block_type:?} bytes={} replaced by a placeholder",
                json.len()
            ),
        );
        format!(
            "[unsupported {block_type} tool_result block omitted ({} bytes)]",
            json.len()
        )
    }
}

/// Placeholder for `document` blocks with no text representation — keeps the title/source pointer
/// visible to the model (fork #559 semantics).
fn document_placeholder(doc: &DocumentBlock) -> String {
    let title = doc.title.as_deref().unwrap_or("untitled");
    match &doc.source {
        DocumentSource::Url { url } => format!("[document \"{title}\": {url}]"),
        DocumentSource::Base64 { media_type, data } => format!(
            "[document \"{title}\" omitted: {media_type}, {} bytes base64]",
            data.len()
        ),
        // Text / Content sources always have a text representation.
        _ => format!("[document \"{title}\"]"),
    }
}

/// Placeholder for `search_result` blocks with no text content — keeps the source/title pointer
/// visible to the model.
fn search_result_placeholder(result: &SearchResultBlock) -> String {
    let title = result.title.as_deref().unwrap_or("untitled");
    match result.source.as_deref() {
        Some(source) => format!("[search result \"{title}\": {source}]"),
        None => format!("[search result \"{title}\"]"),
    }
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

/// `input`, items still raw. `InputParam` and `InputItem` are both untagged, so one typed parse of
/// the array reports only that the array matched no variant; per-item parsing keeps the index.
#[derive(Deserialize)]
#[serde(untagged)]
enum ResponsesInput {
    Text(String),
    Items(Vec<Value>),
}

impl Default for ResponsesInput {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

/// Responses `reasoning`, local rather than the upstream struct so `effort` resolves through the
/// fork's alias table (`REASONING_EFFORT_ALIASES`, `max` -> `xhigh` by default) exactly like chat
/// `reasoning_effort`: upstream's enum has no `max`, which DeepSeek V4 / GLM clients send, and the
/// serve-side policy maps `xhigh` back to the model's native tier. One vocabulary across ingresses.
#[derive(Deserialize)]
struct ResponsesReasoning {
    #[serde(default, deserialize_with = "responses_reasoning_effort")]
    effort: Option<dynamo_protocols::types::ReasoningEffort>,
    #[serde(default)]
    summary: Option<Value>,
}

fn responses_reasoning_effort<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<dynamo_protocols::types::ReasoningEffort>, D::Error> {
    match Option::<String>::deserialize(deserializer)? {
        None => Ok(None),
        Some(effort) => dynamo_protocols::types::parse_reasoning_effort(effort)
            .map(Some)
            .map_err(|e| serde::de::Error::custom(format!("invalid `reasoning.effort`: {e}"))),
    }
}

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
    input: ResponsesInput,
    instructions: Option<String>,
    max_output_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stream: Option<bool>,
    /// Modeled (not ridden through `unmodeled`) so the affinity resolver sees it on `CcRequest.user`
    /// like on the other two ingress paths; the serialized body is identical either way.
    user: Option<String>,
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
    #[serde(default, deserialize_with = "responses_service_tier")]
    service_tier: Option<ResponsesServiceTier>,
    include: Option<Vec<String>>,
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

fn responses_service_tier<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<ResponsesServiceTier>, D::Error> {
    Option::<ResponsesServiceTier>::deserialize(deserializer)
        .map_err(|e| serde::de::Error::custom(format!("invalid `service_tier`: {e}")))
}

fn adapt_responses(
    hooks: &mut dyn IngressHooks,
    dropped: &mut DroppedTools,
    body: Value,
) -> Result<(CcRequest, ServerToolClaims, Option<NonZeroU32>), RequestRejection> {
    let responses_request: ResponsesRequest = parse_client_body(ClientProtocol::Responses, body)?;
    reject_stateful_fields(&responses_request)?;
    check_includes(hooks, &responses_request, &mut dropped.losses)?;
    // OpenAI ranges: temperature 0..2, top_p 0..1 (fork conformance floor — OpenAI 400s these).
    if let Some(t) = responses_request.temperature
        && (!t.is_finite() || !(0.0..=2.0).contains(&t))
    {
        return Err(RequestRejection::malformed(format!(
            "temperature must be between 0 and 2, got {t}"
        )));
    }
    if let Some(p) = responses_request.top_p
        && (!p.is_finite() || !(0.0..=1.0).contains(&p))
    {
        return Err(RequestRejection::malformed(format!(
            "top_p must be between 0 and 1, got {p}"
        )));
    }
    // Fork parity: an out-of-range top_logprobs is refused up front (OpenAI's hosted API 400s at
    // 21), not silently clamped or forwarded for the engine to mangle. The value itself rides the
    // ordered unmodeled passthrough onto the CC body.
    if let Some(value) = responses_request.unmodeled.get("top_logprobs")
        && !value.is_null()
        && value.as_u64().is_none_or(|n| n > 20)
    {
        return Err(RequestRejection::malformed(format!(
            "top_logprobs must be an integer between 0 and 20, got {value}"
        )));
    }

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
    // Tool declarations come from top-level `tools` plus any `additional_tools` input items
    // (codex Responses-Lite framing); top-level entries come first so they win a name clash.
    let mut tool_entries = responses_request.tools;
    match responses_request.input {
        ResponsesInput::Text(text) if !text.is_empty() => {
            cc_messages.push(CcMessage::User(ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text(text),
                name: None,
            }));
        }
        ResponsesInput::Text(_) => {}
        ResponsesInput::Items(items) => {
            let items = items
                .into_iter()
                .enumerate()
                .map(|(index, item)| {
                    parse_client_json(ClientProtocol::Responses, &format!("input[{index}]"), item)
                })
                .collect::<Result<Vec<InputItem>, _>>()?;
            let additional_tools =
                translate_input_items(&items, &mut cc_messages, &mut dropped.losses)
                    .map_err(RequestRejection::malformed)?;
            tool_entries.extend(additional_tools);
        }
    }
    // OpenAI 400s a function_call_output whose call_id has no preceding function_call — same
    // orphan rule as the Messages surface, over the translated CC messages (item order preserved).
    validate_tool_results_have_tool_use(&cc_messages)?;

    let (cc_tools, claimed_names) = expand_declared_tools(
        hooks,
        dropped,
        ClientProtocol::Responses,
        tool_entries,
        is_responses_server_tool_shaped,
        responses_client_function_tool,
    )?;
    let cc_tools = dedupe_tools_by_name(cc_tools, &mut dropped.losses);

    let request = CcRequest {
        inner: CreateChatCompletionRequest {
            model: responses_request.model,
            messages: cc_messages,
            max_completion_tokens: responses_request.max_output_tokens,
            temperature: responses_request.temperature,
            top_p: responses_request.top_p,
            stream: responses_request.stream,
            service_tier: responses_request.service_tier.map(cc_service_tier),
            user: responses_request.user,
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
                    dropped.losses.record(
                        LossKind::ToolChoiceDegraded,
                        "tool_choice",
                        format!("tool_choice degraded to auto: {reason}"),
                    );
                    Some(ChatCompletionToolChoiceOption::Auto)
                }
                Some(Err(reason)) => return Err(RequestRejection::Unsupported(reason)),
            },
            tools: (!cc_tools.is_empty()).then_some(cc_tools),
            store: responses_request.store,
            response_format: responses_request
                .text
                .map(cc_response_format)
                .transpose()
                .map_err(RequestRejection::Unsupported)?
                .flatten(),
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
/// `verbosity` has no CC counterpart and is refused, never dropped. Plain `text` is the spec
/// default and lowers to no `response_format` at all (fork parity: an explicit no-op constraint
/// must not perturb engines that branch on the field's presence).
fn cc_response_format(
    text: ResponseTextParam,
) -> Result<Option<dynamo_protocols::types::ResponseFormat>, String> {
    if text.verbosity.is_some() {
        return Err("`text.verbosity` is not supported".to_string());
    }
    Ok(match text.format {
        TextResponseFormatConfiguration::Text => None,
        TextResponseFormatConfiguration::JsonObject => {
            Some(dynamo_protocols::types::ResponseFormat::JsonObject)
        }
        TextResponseFormatConfiguration::JsonSchema(json_schema) => {
            Some(dynamo_protocols::types::ResponseFormat::JsonSchema { json_schema })
        }
    })
}

/// Responses `reasoning.effort` -> CC `reasoning_effort` (one shared enum, different field name).
/// `summary` has no CC counterpart and is not forwarded; every value (`auto`/`concise`/`detailed`)
/// is accepted — the egress side decides how much reasoning to render (the fork gates its
/// reasoning output item on the summary being requested at all), so refusing a level here would
/// 400 a request the response layer can satisfy.
fn cc_reasoning_effort(
    reasoning: ResponsesReasoning,
) -> Result<Option<dynamo_protocols::types::ReasoningEffort>, String> {
    if let Some(summary) = reasoning.summary {
        tracing::debug!(
            ?summary,
            "reasoning.summary accepted; rendering is decided at egress"
        );
    }
    Ok(reasoning.effort)
}

/// Responses `service_tier` -> CC `service_tier`, one-to-one: `auto` (let the platform pick a tier)
/// and `default` (explicitly the default tier) are distinct requests, so neither is folded into the
/// other — the fork's previous converter kept them apart too, and the response echoes what was sent.
fn cc_service_tier(tier: ResponsesServiceTier) -> CcServiceTier {
    match tier {
        ResponsesServiceTier::Auto => CcServiceTier::Auto,
        ResponsesServiceTier::Default => CcServiceTier::Default,
        ResponsesServiceTier::Flex => CcServiceTier::Flex,
        ResponsesServiceTier::Scale => CcServiceTier::Scale,
        ResponsesServiceTier::Priority => CcServiceTier::Priority,
    }
}

/// Stateless-only stance: TB persists no response, so continuing from one would silently produce a
/// self-contained answer instead of the multi-turn behavior the caller asked for.
fn reject_stateful_fields(request: &ResponsesRequest) -> Result<(), RequestRejection> {
    // `store` is accepted and carried onto the CC body (OpenAI SDKs default it to true); nothing
    // here persists anything, the caller's layer decides what storing means.
    if request.previous_response_id.is_some() {
        return Err(RequestRejection::Unsupported(
            "`previous_response_id` is not supported: this endpoint is stateless, echo the previous \
             response's `output` back as `input` instead"
                .to_string(),
        ));
    }
    if request.conversation.is_some() {
        return Err(RequestRejection::Unsupported(
            "`conversation` is not supported: this endpoint is stateless".to_string(),
        ));
    }
    if request.background == Some(true) {
        return Err(RequestRejection::Unsupported(
            "`background: true` is not supported: this endpoint is stateless".to_string(),
        ));
    }
    if request.prompt.is_some() {
        return Err(RequestRejection::Unsupported(
            "`prompt` is not supported: this endpoint stores no prompt templates, send the full \
             prompt as `input`"
                .to_string(),
        ));
    }
    Ok(())
}

/// Shared with the codex adapter, which strips this include before the gate sees it.
pub const ENCRYPTED_REASONING_INCLUDE: &str = "reasoning.encrypted_content";

/// `include` entries never reach the CC body (the field is modeled, not forwarded); the only one
/// with a decision attached is the encrypted-reasoning include, which the hooks may refuse
/// (tool-bank) or, by default, let through with a warning — Codex sends it on every request and
/// nothing on this stack produces encrypted content, so the answer is simply "none included".
fn check_includes(
    hooks: &dyn IngressHooks,
    request: &ResponsesRequest,
    losses: &mut Losses,
) -> Result<(), RequestRejection> {
    let asks_encrypted_reasoning = request
        .include
        .as_ref()
        .is_some_and(|includes| includes.iter().any(|i| i == ENCRYPTED_REASONING_INCLUDE));
    if !asks_encrypted_reasoning {
        return Ok(());
    }
    if hooks.rejects_encrypted_reasoning_include() {
        return Err(RequestRejection::Unsupported(format!(
            "`include: \"{ENCRYPTED_REASONING_INCLUDE}\"` is not supported: this endpoint never \
             emits encrypted reasoning content"
        )));
    }
    losses.record(
        LossKind::IncludeIgnored,
        "include",
        format!("`{ENCRYPTED_REASONING_INCLUDE}` ignored: no encrypted reasoning content is produced on this stack"),
    );
    Ok(())
}

/// Translate the whole `input` item list into canonical CC messages. Items arrive flat (no
/// per-turn wrapper the way Anthropic messages group blocks), so one [`AssistantMessageBuffer`]
/// accumulates across a run of assistant-owned items (reasoning, text, open tool calls) and flushes
/// whenever a non-assistant item interrupts it — same split-at-boundary technique
/// [`translate_message`] uses per Anthropic message, just flattened over the whole list.
///
/// Returns the tool declarations carried on `additional_tools` items, for the caller to merge
/// into the declared tool list (they are declarations, not transcript, so they leave the message
/// stream and do not break an open assistant turn).
fn translate_input_items(
    items: &[InputItem],
    cc_messages: &mut Vec<CcMessage>,
    losses: &mut Losses,
) -> Result<Vec<Value>, String> {
    let mut turn = AssistantMessageBuffer::default();
    let mut additional_tools = Vec::new();
    for (item_index, item) in items.iter().enumerate() {
        match item {
            InputItem::Item(Item::AdditionalTools(additional)) => {
                additional_tools.extend(additional.tools.iter().cloned());
            }
            // An assistant-shaped easy message coalesces into the open assistant turn; an
            // explicit message item pins the turn boundary even when its text is empty, so
            // strict-alternation templates never see adjacent user turns merge.
            InputItem::EasyMessage(easy) if easy.role == ResponsesRole::Assistant => {
                match &easy.content {
                    EasyInputContent::Text(text) => turn.text.push_str(text),
                    EasyInputContent::ContentList(parts) => {
                        turn.text.push_str(&flatten_input_content(parts)?);
                    }
                }
                turn.explicit_content = true;
            }
            InputItem::EasyMessage(easy) => {
                turn.flush_into(cc_messages);
                cc_messages.push(easy_message(easy)?);
            }
            InputItem::Item(Item::Message(MessageItem::Input(message))) => {
                turn.flush_into(cc_messages);
                cc_messages.push(input_message(message)?);
            }
            InputItem::Item(Item::Message(MessageItem::Output(message))) => {
                // Explicit assistant message item: even empty text keeps the turn boundary.
                turn.explicit_content = true;
                for content in &message.content {
                    match content {
                        dynamo_protocols::types::responses::InputOutputMessageContent::OutputText(text) => {
                            turn.text.push_str(&text.text);
                        }
                        // Refusal text folds into the assistant's content so templates render it
                        // like normal output — the model keeps visibility into what it refused.
                        dynamo_protocols::types::responses::InputOutputMessageContent::Refusal(refusal) => {
                            turn.text.push_str(&refusal.refusal);
                        }
                    }
                }
            }
            InputItem::Item(Item::Reasoning(reasoning)) => {
                for ReasoningItemContent::ReasoningText(content) in
                    reasoning.content.iter().flatten()
                {
                    turn.append_thinking_delta(&content.text);
                }
                // OpenAI/Codex transcripts carry `summary` (TB emits `content`); fold both.
                for SummaryPart::SummaryText(summary) in &reasoning.summary {
                    turn.append_thinking_delta(&summary.text);
                }
            }
            InputItem::Item(Item::FunctionCall(call)) => {
                // A namespaced echo maps back onto the flattened name the model saw
                // (`{namespace}__{name}`), matching the declaration-side flattening.
                let chat_name = match call.namespace.as_deref().filter(|ns| !ns.is_empty()) {
                    Some(namespace) => format!("{namespace}__{}", call.name),
                    None => call.name.clone(),
                };
                // Reasoning replayed after this call belongs to the next segment (the fork's
                // `ReasoningContent::Segments` shape), exactly as the Messages path does.
                turn.close_thinking_segment();
                turn.tool_calls.push(responses_echoed_tool_call(
                    &call.call_id,
                    &chat_name,
                    &call.arguments,
                )?);
            }
            InputItem::Item(Item::McpCall(call)) => {
                turn.close_thinking_segment();
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
            InputItem::Item(Item::AgentMessage(message)) => {
                turn.flush_into(cc_messages);
                match plaintext_agent_message(message) {
                    Some(text) => {
                        cc_messages.push(responses_role_message(
                            ResponsesRole::Assistant,
                            format!("Agent message from {}:\n{text}", message.author),
                        )?);
                    }
                    None => losses.record(
                        LossKind::AgentMessageDropped,
                        format!("input[{item_index}]"),
                        "replayed agent message dropped on ingress (encrypted or empty)",
                    ),
                }
            }
            InputItem::Item(Item::Compaction(_)) => losses.record(
                LossKind::ReplayItemIgnored,
                format!("input[{item_index}]"),
                "replayed Responses compaction item ignored on ingress",
            ),
            InputItem::ItemReference(_) => {
                return Err(
                    "`item_reference` input items are not supported: this endpoint is stateless, it \
                     has no stored item to resolve one against"
                        .to_string(),
                );
            }
            InputItem::Item(other) => {
                // Fork parity: unknown/untranslatable replayed items (computer-use outputs, new
                // upstream item kinds) are skipped, not refused — they are transcript echoes, and
                // failing the request would break agent replay. The open assistant turn still
                // flushes so semantically distinct turns don't coalesce across the gap.
                turn.flush_into(cc_messages);
                // Bounded: the item embeds arbitrary client content, whole-`Debug` echoes it back.
                let item_debug = crate::util::truncate(&format!("{other:?}"), 120);
                losses.record(
                    LossKind::InputItemSkipped,
                    format!("input[{item_index}]"),
                    format!("unsupported input item skipped: {item_debug}"),
                );
            }
        }
    }
    turn.flush_into(cc_messages);
    Ok(additional_tools)
}

/// One CC tool per function name, first declaration wins: a tool declared both top-level and on an
/// `additional_tools` item must not be advertised twice (model backends reject duplicate names).
fn dedupe_tools_by_name(
    tools: Vec<ChatCompletionTool>,
    losses: &mut Losses,
) -> Vec<ChatCompletionTool> {
    let mut seen = std::collections::HashSet::with_capacity(tools.len());
    tools
        .into_iter()
        .filter(|tool| {
            let fresh = seen.insert(tool.function.name.clone());
            if !fresh {
                losses.record(
                    LossKind::DuplicateToolDropped,
                    "tools",
                    format!(
                        "duplicate tool declaration {:?} dropped (first declaration kept)",
                        tool.function.name
                    ),
                );
            }
            fresh
        })
        .collect()
}

/// An agent message's plaintext, `None` when any part is encrypted or the text is empty — the
/// same drop rule codex applies when it renders one for a model.
fn plaintext_agent_message(message: &AgentMessageItemParam) -> Option<String> {
    let mut parts = Vec::with_capacity(message.content.len());
    for part in &message.content {
        let AgentMessageInputContent::InputText { text } = part else {
            return None;
        };
        parts.push(text.as_str());
    }
    let text = parts.join("\n");
    (!text.trim().is_empty()).then_some(text)
}

fn easy_message(
    easy: &dynamo_protocols::types::responses::EasyInputMessage,
) -> Result<CcMessage, String> {
    if let (ResponsesRole::User, EasyInputContent::ContentList(parts)) = (easy.role, &easy.content)
    {
        // User content keeps its image parts (multimodal CC content).
        return Ok(CcMessage::User(ChatCompletionRequestUserMessage {
            content: user_input_content(parts)?,
            name: None,
        }));
    }
    let text = match &easy.content {
        EasyInputContent::Text(text) => text.clone(),
        EasyInputContent::ContentList(parts) => flatten_input_content(parts)?,
    };
    responses_role_message(easy.role, text)
}

fn input_message(
    message: &dynamo_protocols::types::responses::InputMessage,
) -> Result<CcMessage, String> {
    let role = match message.role {
        InputRole::User => ResponsesRole::User,
        InputRole::System => ResponsesRole::System,
        InputRole::Developer => ResponsesRole::Developer,
    };
    if role == ResponsesRole::User {
        // User content keeps its image parts (multimodal CC content).
        return Ok(CcMessage::User(ChatCompletionRequestUserMessage {
            content: user_input_content(&message.content)?,
            name: None,
        }));
    }
    responses_role_message(role, flatten_input_content(&message.content)?)
}

/// User `input` content parts -> CC user content: all-text collapses to the plain text form, an
/// `input_image` switches to the multimodal part array (fork parity — images must survive to the
/// chat request, not be flattened away). Files still have no CC translation and are refused.
fn user_input_content(
    parts: &[InputContent],
) -> Result<ChatCompletionRequestUserMessageContent, String> {
    let mut text = String::new();
    let mut cc_parts: Vec<ChatCompletionRequestUserMessageContentPart> = Vec::new();
    let mut has_image = false;
    for part in parts {
        match part {
            InputContent::InputText(part) => {
                text.push_str(&part.text);
                cc_parts.push(ChatCompletionRequestUserMessageContentPart::Text(
                    ChatCompletionRequestMessageContentPartText {
                        text: part.text.clone(),
                    },
                ));
            }
            InputContent::InputImage(image) => {
                has_image = true;
                cc_parts.push(ChatCompletionRequestUserMessageContentPart::ImageUrl(
                    responses_image_part(image)?,
                ));
            }
            InputContent::InputFile(_) => return Err(unsupported_input_content_part("input_file")),
        }
    }
    Ok(if has_image {
        ChatCompletionRequestUserMessageContent::Array(cc_parts)
    } else {
        ChatCompletionRequestUserMessageContent::Text(text)
    })
}

/// A Responses `input_image` -> a CC image part. `file_id` references cannot be resolved here;
/// only a URL (https or data URI) translates.
fn responses_image_part(
    image: &dynamo_protocols::types::responses::InputImageContent,
) -> Result<ChatCompletionRequestMessageContentPartImage, String> {
    let Some(image_url) = image.image_url.as_deref() else {
        return Err(
            "input_image without `image_url` is not supported (file_id references cannot be \
             resolved here)"
                .to_string(),
        );
    };
    let url = url::Url::parse(image_url).map_err(|e| format!("invalid image_url: {e}"))?;
    Ok(ChatCompletionRequestMessageContentPartImage {
        image_url: ImageUrl {
            url,
            detail: Some(image.detail.clone()),
            uuid: None,
        },
    })
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
) -> Result<Vec<ChatCompletionTool>, RequestRejection> {
    let tool: ResponsesTool = serde_json::from_value(tool_entry)
        .map_err(|e| RequestRejection::malformed(format!("invalid tool definition: {e}")))?;
    match tool {
        ResponsesTool::Function(function) => {
            let mut cc_tool = function_tool(
                &function.name,
                function.description.as_deref().unwrap_or_default(),
                function.parameters.unwrap_or_else(|| json!({})),
            );
            cc_tool.function.strict = function.strict;
            Ok(vec![cc_tool])
        }
        // A namespace group flattens to `{namespace}__{name}` function tools (Codex declares its
        // MCP apps this way): namespaces exist so member names can overlap between groups, and
        // the egress side maps emitted calls back to the wire (name, namespace) pair against the
        // request's declarations.
        ResponsesTool::Namespace(group) => {
            let mut members = Vec::with_capacity(group.tools.len());
            for member in &group.tools {
                let dynamo_protocols::types::responses::NamespaceToolParamTool::Function(function) =
                    member
                else {
                    return Err(RequestRejection::Unsupported(format!(
                        "unsupported namespace tool member type \"custom\" in group {:?}",
                        group.name
                    )));
                };
                let flat_name = format!("{}__{}", group.name, function.name);
                let mut cc_tool = function_tool(
                    &flat_name,
                    function.description.as_deref().unwrap_or_default(),
                    function.parameters.clone().unwrap_or_else(|| json!({})),
                );
                cc_tool.function.strict = function.strict;
                members.push(cc_tool);
            }
            Ok(members)
        }
        other => {
            // Bounded: the entry embeds the client's full tool definition, whole-`Debug` echoes it back.
            let tool_debug = crate::util::truncate(&format!("{other:?}"), 80);
            Err(RequestRejection::Unsupported(format!(
                "unsupported tool type {tool_debug}: only client function tools (and the server \
                 tools this endpoint claims) are accepted"
            )))
        }
    }
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

/// A tool message must answer a preceding assistant tool call — an orphan means the client
/// corrupted its transcript, and forwarding it renders a nonsensical conversation (both Anthropic
/// and OpenAI 400 the equivalents). Runs over the *translated* CC messages, after any
/// `srvtoolu_`-prefix normalization, so replayed server-tool pairs still match.
fn validate_tool_results_have_tool_use(messages: &[CcMessage]) -> Result<(), RequestRejection> {
    let mut seen_ids: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for message in messages {
        match message {
            CcMessage::Assistant(assistant) => {
                for tool_call in assistant.tool_calls.iter().flatten() {
                    seen_ids.insert(tool_call.id.as_str());
                }
            }
            CcMessage::Tool(tool) => {
                if !seen_ids.contains(tool.tool_call_id.as_str()) {
                    return Err(RequestRejection::malformed(format!(
                        "tool result references call id \"{}\" with no preceding tool call",
                        tool.tool_call_id
                    )));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Split a typed protocol's declared tools: a server-tool-shaped entry (the protocol's own
/// hosted-tool shapes; any non-client `type`) goes to the hooks; anything else translates via
/// the protocol's client-tool constructor. (CC keeps its own loop — it forwards client entries
/// verbatim as untyped JSON instead of translating them.)
fn expand_declared_tools(
    hooks: &mut dyn IngressHooks,
    dropped: &mut DroppedTools,
    protocol: ClientProtocol,
    tool_entries: Vec<Value>,
    is_server_tool_shaped: fn(&Value) -> bool,
    client_function_tool: fn(Value) -> Result<Vec<ChatCompletionTool>, RequestRejection>,
) -> Result<(Vec<ChatCompletionTool>, Vec<String>), RequestRejection> {
    let mut cc_tools = Vec::with_capacity(tool_entries.len());
    let mut claimed_names = Vec::new();
    for tool_entry in tool_entries {
        if is_server_tool_shaped(&tool_entry) {
            match hooks.on_server_tool(protocol, &tool_entry) {
                ToolDisposition::Claim(claim) => {
                    cc_tools.push(claim.tool);
                    claimed_names.push(claim.name);
                }
                ToolDisposition::Drop => dropped.record(protocol, &tool_entry),
                ToolDisposition::Reject(rejection) => return Err(rejection),
            }
        } else {
            cc_tools.extend(client_function_tool(tool_entry)?);
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

/// A Responses tool entry that is not a client-executable shape: anything but a `function`, a
/// `custom` tool, or a `namespace` group (all three translate into client function tools) goes to
/// the hooks — OpenAI-hosted surfaces (`web_search*`, `mcp`, `file_search`, ...) and any
/// consumer-defined selection type alike. Shape only; the crate knows no tool namespace.
fn is_responses_server_tool_shaped(entry: &Value) -> bool {
    entry
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| !matches!(kind, "function" | "custom" | "namespace"))
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
            // Fork separator ruling (decided 2026-09-03): adjacent blocks join with "\n", the
            // shape the deployed Messages converter always produced (prompt-cache stable).
            .join("\n"),
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
    ceiling: NonZeroU32,
) -> Result<(NonZeroU32, ReactCapSource), RequestRejection> {
    if requested.is_some_and(|requested| !(iterations_floor..=ceiling).contains(&requested)) {
        return Err(RequestRejection::malformed(format!(
            "baseten.tool_settings.max_react_iterations must be within \
             {iterations_floor}..={ceiling} (got {})",
            requested.expect("checked just above")
        )));
    }
    let effective = requested.unwrap_or(num_default_react_iterations);
    let cap_source = if effective == ceiling {
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
#[allow(clippy::unwrap_used)]
mod tests;
