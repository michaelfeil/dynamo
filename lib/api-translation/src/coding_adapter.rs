//! The contract a coding client's adapter implements, and nothing else: this module names no
//! client. Ingress is one body rewrite before any typed parsing (behind
//! [`crate::hooks::IngressHooks::rewrite_ingress`]); egress is the client's own item JSON plus the
//! events that carry it. The client adapters themselves (Claude Code, Codex) live with the caller
//! that runs their web-search machinery — this crate only renders what an adapter returns.

use std::collections::HashMap;

use serde_json::Value;

pub enum ToolCallStatus {
    InProgress,
    Completed,
    Failed,
}

pub struct ToolCallToRender<'a> {
    pub tool_name: &'a str,
    pub id: &'a str,
    pub args: &'a str,
    pub status: ToolCallStatus,
}

pub struct ToolResultToRender<'a> {
    pub tool_name: &'a str,
    pub id: &'a str,
    /// The provider's structure, never JSON inside a string. A provider reaches the framings in
    /// either encoding, so the framings' shared render seam unwraps it before an adapter sees it.
    pub content: &'a Value,
    pub is_error: bool,
}

pub struct RenderedToolCall {
    pub item: Value,
    /// Adapter events carry an item, so they preserve extra top-level fields.
    pub lifecycle_events: Vec<&'static str>,
}

/// The declaration a namespaced tool was hoisted from: the `(name, namespace)` pair the client's
/// registry dispatches on.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolIdentity {
    pub name: String,
    pub namespace: String,
}

pub trait CodingAdapter: Send + Sync {
    /// `None` for a call this adapter did not expand: it renders the protocol's usual way.
    fn render_tool_call(&self, call: &ToolCallToRender<'_>) -> Option<RenderedToolCall>;
    /// The client's own object for a resolved call's result, for a protocol that surfaces one
    /// separately from the call. Default `None`: the call item is the whole story.
    fn render_tool_result(&self, _result: &ToolResultToRender<'_>) -> Option<Value> {
        None
    }
    /// Replace each recorded typed slot in a finished body with the item rendered for it, by id.
    /// Default identity: only a protocol whose typed body cannot carry the client's shape splices.
    fn splice_rendered_calls(&self, body: Value, _rendered: &HashMap<String, Value>) -> Value {
        body
    }
    /// The declaration a namespaced tool was hoisted from, so egress can stamp the call item
    /// back into the client's registry key. Default `None`: the call's own name passes through.
    fn resolve_tool_identity(&self, _tool_name: &str) -> Option<ToolIdentity> {
        None
    }
}
