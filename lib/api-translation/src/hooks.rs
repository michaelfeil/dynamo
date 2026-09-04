//! The ingress hook seam: where server-tool selection plugs in without this crate executing
//! anything. [`crate::request::adapt_request`] hands every server-tool-shaped `tools[]` entry to
//! the request's [`IngressHooks`], which may claim it (tool-bank resolving its own selection
//! against its registry), drop it (standard dynamo), or reject the request. The hooks also supply
//! the ReAct-cap defaults tool-bank sourced from its `ReactLoopSettings`, and the coding-adapter
//! body rewrite that ran inline in tool-bank's `adapt_request`.

use std::num::NonZeroU32;

use http::HeaderMap;
use serde_json::Value;

use dynamo_protocols::types::ChatCompletionTool;

use crate::ClientProtocol;
use crate::coding_adapter::CodingAdapter;
use crate::loss::Loss;
use crate::model::RequestRejection;

/// The ReAct-loop bounds the consumer's loop enforces. The crate parses `baseten.tool_settings`
/// (it is wire) and validates the request against these; what a request may ask for is the
/// consumer's policy, so the bounds live here on the hook, not as crate constants.
#[derive(Debug, Clone, Copy)]
pub struct IngressLimits {
    /// Iteration cap for a request that names none. Must lie within
    /// `server_tool_iterations_floor..=max_react_iterations`.
    pub default_react_iterations: NonZeroU32,
    /// Hard ceiling on a single request's ReAct iterations, regardless of what it asks for.
    pub max_react_iterations: NonZeroU32,
    /// Floor for a request that can dispatch a server tool: the call and the answer are separate
    /// model calls, so one iteration could only ever end on a dispatched call. Requests that cannot
    /// reach a server tool are bounded by 1.
    pub server_tool_iterations_floor: NonZeroU32,
    /// Ceiling on `baseten.tool_settings.max_tool_calls_per_iteration`, and the value used when the
    /// request names none.
    pub max_tool_calls_per_iteration: NonZeroU32,
}

impl Default for IngressLimits {
    /// tool-bank's shipped `ReactLoopSettings` defaults; a plain dynamo deployment (no server
    /// tools) never reaches the floor and only uses the ceiling to bound a client-supplied cap.
    fn default() -> Self {
        Self {
            default_react_iterations: NonZeroU32::new(20).expect("20 > 0"),
            max_react_iterations: NonZeroU32::new(20).expect("20 > 0"),
            server_tool_iterations_floor: NonZeroU32::new(2).expect("2 > 0"),
            max_tool_calls_per_iteration: NonZeroU32::new(10).expect("10 > 0"),
        }
    }
}

/// A server tool the hooks claimed: the entry leaves the client tool list, `tool` is advertised to
/// the model in its place, and the claim is recorded on
/// [`crate::request::AdaptedRequest::server_tool_claims`].
pub struct ClaimedServerTool {
    /// The function-tool name the model calls this tool by (for tool-bank, the qualified name).
    /// Also what a `tool_choice` naming the claim resolves against.
    pub name: String,
    /// The CC function tool declared to the model in place of the claimed entry.
    pub tool: ChatCompletionTool,
}

/// What the hooks decided about one server-tool-shaped `tools[]` entry.
pub enum ToolDisposition {
    /// The caller will fulfill this tool itself.
    Claim(ClaimedServerTool),
    /// Remove the entry from the tool list; the crate logs the drop and degrades any `tool_choice`
    /// naming it to `auto`.
    Drop,
    /// Refuse the whole request.
    Reject(RequestRejection),
}

/// What rewriting the ingress produced: the adapter that renders coding-client items back, and any
/// loop bound the client declared on its own tool that the baseline request shape cannot carry.
pub struct IngressRewrite {
    pub adapter: Box<dyn CodingAdapter>,
    pub max_react_iterations: Option<NonZeroU32>,
}

/// Per-request ingress extension points. `adapt_request` drives these; the default impls are
/// standard-dynamo behavior (no server tools, no coding adapter, shipped cap defaults).
pub trait IngressHooks {
    /// Cap defaults for requests that name none.
    fn limits(&self) -> IngressLimits {
        IngressLimits::default()
    }

    /// Decide one server-tool-shaped `tools[]` entry — anything that is not a client-executable
    /// function tool: a Chat Completions or Responses entry with a non-`function` `type`, or an
    /// Anthropic entry with a non-`custom` `type`. Routing is by shape only: the crate knows the
    /// protocols' hosted-tool shapes, not any consumer's tool namespace (tool-bank recognises its
    /// own `baseten__*` selections inside its implementation). `entry` is the client's raw JSON.
    fn on_server_tool(&mut self, protocol: ClientProtocol, entry: &Value) -> ToolDisposition {
        let _ = (protocol, entry);
        ToolDisposition::Drop
    }

    /// Whether a `tool_choice` naming a tool shape this endpoint cannot run (a hosted/server tool
    /// under the default drop, an unsupported Responses shape) degrades to `auto` with a warning —
    /// standard-dynamo behavior — or rejects the request. tool-bank keeps its stricter 400.
    fn degrades_unsupported_tool_choice(&self) -> bool {
        true
    }

    /// Whether a Responses `include: ["reasoning.encrypted_content"]` is refused (tool-bank's
    /// stance: it never emits encrypted reasoning, so a caller asking for it is told so). Standard
    /// dynamo accepts it — the Codex CLI sends it by default — and drops the entry with a warning,
    /// since no encrypted content is produced either way.
    fn rejects_encrypted_reasoning_include(&self) -> bool {
        false
    }

    /// Whether Messages features this endpoint cannot honor are refused (tool-bank's stance) or
    /// dropped with a warning (standard dynamo, whose previous converter skipped them): content
    /// blocks with no Chat Completions translation in a user message, and the Anthropic-hosted
    /// `mcp_servers` / `container` request fields.
    fn rejects_unsupported_messages_features(&self) -> bool {
        false
    }

    /// One [`Loss`] the request survived — something adaptation dropped, skipped, degraded, or
    /// folded. Every loss is also on [`crate::request::AdaptedRequest::losses`] and logged at WARN
    /// by the crate; this hook is for the consumer's counter (tool-bank labels one metric by
    /// [`crate::LossKind::as_label`]). Default: nothing beyond the crate's own log.
    fn on_loss(&mut self, loss: &Loss) {
        let _ = loss;
    }

    /// One shot at rewriting the raw body before typed parsing — the coding-adapter seam (tool-bank
    /// expands a coding client's `web_search` tool into provider tools here). Returning an adapter
    /// makes egress render that client's own item shapes.
    fn rewrite_ingress(
        &mut self,
        protocol: ClientProtocol,
        headers: &HeaderMap,
        body: &mut Value,
    ) -> Result<Option<IngressRewrite>, RequestRejection> {
        let _ = (protocol, headers, body);
        Ok(None)
    }
}

/// Standard-dynamo ingress: every server-tool-shaped entry is dropped (with a warning at the call
/// site), no coding adapter, shipped cap defaults.
pub struct DropServerTools;

impl IngressHooks for DropServerTools {}
