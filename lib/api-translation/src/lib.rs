//! API-translation layer. ChatCompletions is the canonical internal hub protocol: the client
//! protocol is translated to CC at the request edge ([`request`]) and back at the response edge
//! ([`sse_emitter`]). The model's CC stream is decoded once at the single boundary source
//! ([`sse_parser::SseParser`]) into [`SemanticChunk`]s that downstream consumers fold — nothing
//! re-derives those boundaries.
//!
//! Ported from tool-bank's `api_translation` module (Apache-2.0). Server tools are represented but
//! never executed here: ingress selection plugs in through [`hooks::IngressHooks`] (the shipped
//! [`hooks::DropServerTools`] is standard-dynamo behavior — drop them), and the egress framings
//! render whatever server-tool activity the caller's loop reports.

pub mod baseten_response_extension;
pub mod client_egress;
pub mod coding_adapter;
pub mod framing;
pub mod history;
pub mod hooks;
pub mod loss;
pub mod model;
pub mod request;
pub mod sse_emitter;
pub mod sse_parser;

pub(crate) mod util;
pub(crate) mod wire;

#[cfg(test)]
mod dynamo_conformance_tests;
#[cfg(test)]
pub(crate) mod test_utils;

use crate::model::ToolCall;

/// What adaptation did not carry through, and the closed set of kinds a counter can be labeled by.
pub use crate::loss::{Loss, LossKind};
/// The client-facing error class, re-exported so callers can select it without reaching into
/// [`model`]. Each protocol renders it as its own `error.type` word on the response edge.
pub use crate::model::ErrorClass;

/// The canonical internal wire types (the fork's `dynamo-protocols`). `Cc*` names track the
/// DoR vocabulary — the client protocol is translated to these at the request edge and back at the
/// response edge; everything between speaks ChatCompletions.
/// The protocol crate this crate's public API is expressed in. Consumers should reach the wire
/// types through this re-export instead of a second `dynamo-protocols` dependency, so the two can
/// never drift onto different revisions.
pub use dynamo_protocols;
pub use dynamo_protocols::types::{ChatCompletionRequestMessage as CcMessage, FinishReason};

/// The canonical ChatCompletions request: the fork's wire type plus the ordered passthrough of
/// every request field it does not model (`chat_template_kwargs`, `guided_*`, provider sampling
/// extras). The catch-all lives HERE, on the translation layer's own type, not on the shared wire
/// type: a `#[serde(flatten)]` map on `CreateChatCompletionRequest` itself would make serde
/// deserialize the struct through its map path, and every wrapper that flattens it alongside
/// its own catch-all (dynamo's `NvCreateChatCompletionRequest`) would then see every key —
/// `model`, `messages`, ... — as unknown.
///
/// `unmodeled` is a `serde_json::Map` (ordered under `preserve_order`), NOT a HashMap, so the
/// re-serialized field order is turn-stable for the KV-cache prefix. Derefs to the wire type so
/// the modeled fields read as its own.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CcRequest {
    #[serde(flatten)]
    pub inner: dynamo_protocols::types::CreateChatCompletionRequest,
    #[serde(flatten)]
    pub unmodeled: serde_json::Map<String, serde_json::Value>,
}

impl std::ops::Deref for CcRequest {
    type Target = dynamo_protocols::types::CreateChatCompletionRequest;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for CcRequest {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl From<dynamo_protocols::types::CreateChatCompletionRequest> for CcRequest {
    fn from(inner: dynamo_protocols::types::CreateChatCompletionRequest) -> Self {
        Self {
            inner,
            unmodeled: serde_json::Map::new(),
        }
    }
}

#[cfg(test)]
mod cc_request_tests {
    use super::CcRequest;

    /// The passthrough contract that used to sit on the wire type: unknown request fields survive
    /// a parse/serialize round trip, in the client's order, alongside the modeled fields.
    #[test]
    fn unmodeled_request_fields_round_trip_in_client_order() {
        let input = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"enable_thinking": true},
            "guided_json": {"type": "object"},
            "zeta_first": 1,
            "alpha_second": 2
        });
        let request: CcRequest = serde_json::from_value(input.clone()).unwrap();

        assert_eq!(request.model, "test-model");
        assert_eq!(request.messages.len(), 1);
        assert_eq!(
            request.unmodeled.keys().collect::<Vec<_>>(),
            [
                "chat_template_kwargs",
                "guided_json",
                "zeta_first",
                "alpha_second"
            ]
        );

        let output = serde_json::to_value(&request).unwrap();
        assert_eq!(output["model"], "test-model");
        assert_eq!(
            output["chat_template_kwargs"],
            input["chat_template_kwargs"]
        );
        assert_eq!(output["guided_json"], input["guided_json"]);
        assert_eq!(output["zeta_first"], 1);
        assert_eq!(output["alpha_second"], 2);
        // No key is emitted twice: the modeled fields are not mirrored into the catch-all.
        let text = serde_json::to_string(&request).unwrap();
        assert_eq!(text.matches("\"model\"").count(), 1, "{text}");
    }
}

/// Anthropic's own id shape for a server-executed tool use, which its clients match results on.
/// Minted on egress by a Messages coding adapter and stripped back off on ingress, so a replayed
/// turn resolves to the same call id the model issued.
pub const SERVER_TOOL_USE_ID_PREFIX: &str = "srvtoolu_";

/// Anthropic's discriminator value for a base64-inline image source — the only image source kind
/// translatable to a CC data-URI part.
pub(crate) const IMAGE_SOURCE_TYPE_BASE64: &str = "base64";

/// Channel carrying encoded SSE frames from the loop/emitter to the HTTP egress.
pub type SseFrameTx = tokio::sync::mpsc::Sender<String>;

/// External protocol the client speaks. Internal/model edge is always ChatCompletions; `Messages`
/// clients are translated at both edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientProtocol {
    ChatCompletions,
    Messages,
    Responses,
}

/// Reported per raw model delta by the parser ([`sse_parser::SseDataYield`]); the axis a loop's
/// phase timeline keys off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentKind {
    Thinking,
    Text,
    ToolCall,
}

/// One finished semantic unit off the model's ChatCompletions stream, drawn once at the single
/// boundary source ([`sse_parser::SseParser`]). Text and thinking stream through as deltas; a tool
/// call is held until its args parse and surfaces once, complete.
#[derive(Debug, Clone)]
pub enum SemanticChunk {
    TextDelta(String),
    ThinkingDelta(String),
    /// A complete tool call: valid-JSON args, assembled by the parser.
    ToolCall(ToolCall),
    /// One model call's usage (we request `stream_options.include_usage`).
    Usage(dynamo_protocols::types::CompletionUsage),
    Stop {
        finish_reason: dynamo_protocols::types::FinishReason,
    },
}
