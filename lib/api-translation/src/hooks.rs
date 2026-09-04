//! The ingress hook seam: where server-tool selection plugs in without this crate executing
//! anything. [`crate::request::adapt_request`] hands every server-tool-shaped `tools[]` entry to
//! the request's [`IngressHooks`], which may claim it (tool-bank resolving a `baseten__*` selection
//! against its registry), drop it (standard dynamo), or reject the request. The hooks also supply
//! the ReAct-cap defaults tool-bank sourced from its `ReactLoopSettings`, and the coding-adapter
//! body rewrite that ran inline in tool-bank's `adapt_request`.

use std::num::NonZeroU32;

use http::HeaderMap;
use serde_json::Value;

use dynamo_protocols::types::ChatCompletionTool;

use crate::coding_adapter::CodingAdapter;
use crate::model::RequestRejection;
use crate::{ClientProtocol, RESERVED_TOOL_PREFIX};

/// Hard ceiling on a single request's ReAct iterations, regardless of hook defaults.
pub const REACT_ITERATIONS_MAX: NonZeroU32 = NonZeroU32::new(20).expect("20 > 0");
/// Floor for a request that can dispatch a server tool: the call and the answer are separate model
/// calls, so one iteration could only ever end on a dispatched call. Others are bounded by 1.
pub const REACT_ITERATIONS_MIN: NonZeroU32 = NonZeroU32::new(2).expect("2 > 0");

/// The ReAct-cap defaults a request falls back to when its own `baseten.tool_settings` names none —
/// what tool-bank configured via `ReactLoopSettings`.
#[derive(Debug, Clone, Copy)]
pub struct IngressLimits {
    /// Iteration cap for a request that names none. Must lie within
    /// [`REACT_ITERATIONS_MIN`]..=[`REACT_ITERATIONS_MAX`].
    pub default_react_iterations: NonZeroU32,
    /// Ceiling on `baseten.tool_settings.max_tool_calls_per_iteration`, and the value used when the
    /// request names none.
    pub max_tool_calls_per_iteration: NonZeroU32,
}

impl Default for IngressLimits {
    /// tool-bank's shipped `ReactLoopSettings` defaults.
    fn default() -> Self {
        Self {
            default_react_iterations: REACT_ITERATIONS_MAX,
            max_tool_calls_per_iteration: NonZeroU32::new(10).expect("10 > 0"),
        }
    }
}

/// A server tool the hooks claimed: the entry leaves the client tool list, `tool` is advertised to
/// the model in its place, and the claim is recorded on
/// [`crate::request::AdaptedRequest::server_tool_claims`].
pub struct ClaimedServerTool {
    /// The function-tool name the model calls this tool by (for a `baseten__*` selection, the
    /// qualified name). Also what a `tool_choice` naming the claim resolves against.
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

    /// Decide one server-tool-shaped `tools[]` entry: a reserved `baseten__*`-typed entry (any
    /// protocol), an Anthropic-native server tool (a versioned non-`custom` `type`), or a Responses
    /// non-function tool (`web_search`, `mcp`, ...). `entry` is the client's raw JSON.
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

/// Whether a `tools[]` entry's `type` selects a Baseten server tool (`baseten__*`).
pub(crate) fn is_reserved_tool_type(tool_type: Option<&str>) -> bool {
    tool_type.is_some_and(|kind| kind.starts_with(RESERVED_TOOL_PREFIX))
}

/// The `<provider>` segment of a reserved `baseten__<provider>__<tool>` name, for wire records that
/// label the provider. `None` when the name is not reserved-shaped.
pub(crate) fn reserved_tool_provider(name: &str) -> Option<&str> {
    let (provider, tool) = name.strip_prefix(RESERVED_TOOL_PREFIX)?.split_once("__")?;
    (!provider.is_empty() && !tool.is_empty()).then_some(provider)
}
