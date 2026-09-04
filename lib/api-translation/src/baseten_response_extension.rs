//! The `baseten` response extension: everything TB adds on top of the stock client protocol, under
//! one key on a frame of the client's own protocol. This is a published wire schema — documented
//! today in tool-bank's `rust/tool-bank/docs/protocol.md` in the monorepo
//! (<https://github.com/basetenlabs/baseten/blob/master/rust/tool-bank/docs/protocol.md>); the wire-contract half of that doc moves next to this crate as a follow-up — rendered
//! by [`super::framing`] and carried identically on both protocols.

use dynamo_protocols::types::CompletionUsage;
use serde::Serialize;

use crate::CcMessage;
use crate::model::{ServerToolCall, ServerToolCallStatus, Termination, ToolOutput};

/// Independent fields per scope rather than a one-of, so a frame carrying both loses neither.
/// `Usage` is the client protocol's own usage type; every other field is identical across protocols
/// and modes.
#[derive(Serialize)]
pub struct BasetenResponseExtension<Usage: Serialize> {
    /// Plural in both modes: one entry on a streaming frame, all of them on a buffered body.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub iterations: Vec<IterationScope<Usage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<RequestScope>,
}

/// Hand-written rather than `#[derive(Default)]`: the derive adds a spurious `Usage: Default`
/// bound (neither field needs one — an empty `Vec` and `None` require nothing of `Usage`), which
/// `ResponseUsage` (no `Default` impl upstream) would otherwise fail.
impl<Usage: Serialize> Default for BasetenResponseExtension<Usage> {
    fn default() -> Self {
        Self {
            iterations: Vec::new(),
            request: None,
        }
    }
}

impl<Usage: Serialize> BasetenResponseExtension<Usage> {
    pub fn is_empty(&self) -> bool {
        self.iterations.is_empty() && self.request.is_none()
    }

    /// Fold `other` in, so a frame that has to carry two scopes at once loses neither.
    pub fn merge(&mut self, other: Self) {
        self.iterations.extend(other.iterations);
        if let Some(request) = other.request {
            self.request = Some(request);
        }
    }
}

/// One model call inside the ReAct loop.
#[derive(Serialize, Clone)]
pub struct IterationScope<Usage: Serialize> {
    pub index: u32,
    /// This call's *instant* usage — never the canonical `usage` path, so SEG never meters it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// CC only, twice per call (dispatch, then completion) so a client can render a running call.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub server_tool_calls: Vec<ServerToolCallRecord>,
    /// CC only: the messages a client appends to continue from this iteration. Messages carries the
    /// same thing natively, as content blocks.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub continuation_messages: Vec<CcMessage>,
    /// No schema: a client must not branch on these.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub debug_msg: Vec<String>,
}

impl<Usage: Serialize> IterationScope<Usage> {
    /// Nothing but the index: a streaming frame carries only what that moment knows, so each emitter
    /// fills in its own field with struct-update syntax.
    pub fn at(index: u32) -> Self {
        Self {
            index,
            usage: None,
            server_tool_calls: Vec::new(),
            continuation_messages: Vec::new(),
            debug_msg: Vec::new(),
        }
    }
}

impl IterationScope<CompletionUsage> {
    /// For a protocol that carries the loop's calls and history natively: repeating them under
    /// `baseten` would double the wire.
    pub fn into_usage_only<Usage: Serialize>(
        self,
        convert_usage: impl Fn(&CompletionUsage) -> Usage,
    ) -> IterationScope<Usage> {
        IterationScope {
            index: self.index,
            usage: self.usage.as_ref().map(convert_usage),
            server_tool_calls: Vec::new(),
            continuation_messages: Vec::new(),
            debug_msg: self.debug_msg.clone(),
        }
    }
}

/// What [`IterationScope::debug_msg`] carries when the loop appended its steering message.
pub fn steering_appended_note(steering_prompt: &str) -> String {
    format!("appended steering message: {steering_prompt}")
}

/// One server-tool call, keyed by the model's own `id` so a client merges the dispatch and completion
/// records. No result: `continuation_messages` already carries it, once.
#[derive(Serialize, Clone)]
pub struct ServerToolCallRecord {
    pub id: String,
    /// The provider half of the qualified name, same split as the metric label.
    pub provider: String,
    pub name: String,
    pub arguments: String,
    /// `null` until the call completes, so a client never has to read meaning into a missing field.
    pub is_error: Option<bool>,
    /// Terminal state, absent while running. `is_error` above stays as its coarse boolean twin —
    /// the shape CC clients already parse.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<ServerToolCallStatus>,
    /// The reached-processing billing rule's verdict; absent while running.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub billable: Option<bool>,
    /// What the provider reports this call charging against; `None` while running or unreported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sku: Option<String>,
}

impl ServerToolCallRecord {
    pub fn dispatched(server_call: &ServerToolCall) -> Self {
        Self {
            id: server_call.call.id.clone(),
            provider: server_call.provider.clone(),
            name: server_call.call.name.clone(),
            arguments: server_call.call.raw_args.clone(),
            is_error: None,
            status: None,
            billable: None,
            sku: None,
        }
    }

    pub fn completed(server_call: &ServerToolCall, output: &ToolOutput) -> Self {
        Self {
            is_error: Some(output.is_error()),
            status: Some(output.status),
            billable: Some(output.billable),
            sku: output.sku.clone(),
            ..Self::dispatched(server_call)
        }
    }
}

/// The whole client request. Present only when TB has something the stock protocol cannot say.
#[derive(Serialize)]
pub struct RequestScope {
    /// TB-owned terminations only: the model's own reason is already on the native stop field, and
    /// repeating it would make a client parse two vocabularies for one fact.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub termination_reason: Option<TerminationReason>,
    /// Every server-tool call the loop answered — executed or TB-refused (`is_error: true`, no
    /// `sku`) — in dispatch order across iterations.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub server_tool_calls: Vec<ServerToolCallOutcome>,
}

impl Termination {
    pub fn request_scope(
        self,
        server_tool_calls: &[ServerToolCallOutcome],
    ) -> Option<RequestScope> {
        let termination_reason = match self {
            Self::Model(_) => None,
            Self::ReactCapExhausted => Some(TerminationReason::MaxReactIterationsReached),
        };
        (termination_reason.is_some() || !server_tool_calls.is_empty()).then(|| RequestScope {
            termination_reason,
            server_tool_calls: server_tool_calls.to_vec(),
        })
    }
}

/// One answered server-tool call, usage-record slim: its terminal state and what it charges
/// against, never the content — arguments and results live in the per-iteration records / native
/// blocks.
#[derive(Serialize, Clone)]
pub struct ServerToolCallOutcome {
    pub id: String,
    /// The provider half of the qualified name, same split as the metric label.
    pub provider: String,
    pub name: String,
    pub status: ServerToolCallStatus,
    /// The reached-processing billing rule's verdict (see [`ToolOutput::billable`]).
    pub billable: bool,
    /// What the provider reports this call charging against; `None` when unreported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sku: Option<String>,
}

impl ServerToolCallOutcome {
    pub fn completed(server_call: &ServerToolCall, output: &ToolOutput) -> Self {
        Self {
            id: server_call.call.id.clone(),
            provider: server_call.provider.clone(),
            name: server_call.call.name.clone(),
            status: output.status,
            billable: output.billable,
            sku: output.sku.clone(),
        }
    }
}

/// A termination TB owns, named exactly. Never the model's reason.
#[derive(Serialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum TerminationReason {
    MaxReactIterationsReached,
}

/// One frame body: the client protocol's own fields at the top level, plus TB's optional side channel
/// alongside them. One type for both protocols, so the `baseten` key attaches identically on each.
///
/// Top level rather than inside `usage`: neither SDK's stream accumulator preserves either position
/// into its final object, so a streaming client reads the frame as it passes in both cases, and top
/// level is the one anchor that works on every frame.
#[derive(Serialize)]
pub struct BasetenFrame<'a, Body: Serialize, Usage: Serialize> {
    #[serde(flatten)]
    pub body: &'a Body,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseten: Option<&'a BasetenResponseExtension<Usage>>,
}
