//! Crate-owned data vocabulary. Minimal mirrors of the tool-bank types the representation code
//! needs, so the crate stands alone: callers that execute server tools (tool-bank) re-export these
//! and layer their execution machinery on top.
//!
//! Ported from tool-bank `src/model.rs` (Apache-2.0), trimmed to the data the translation layer
//! reads — no cancellation, telemetry, or provider machinery.

use serde_json::Value;

/// A complete model tool call, args already proven to parse.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: Value,
    /// The model's verbatim args bytes, replayed unchanged into history. A parse->reserialize
    /// roundtrip (`args.to_string()`) can differ from the model's exact bytes and break the model's
    /// KV-cache prefix on replay; this preserves them.
    pub raw_args: String,
}

/// A model tool call the caller resolved to a server tool it executes itself. The crate only
/// renders these (extension records, `mcp_call` items); dispatching them is the caller's.
#[derive(Debug, Clone)]
pub struct ServerToolCall {
    pub call: ToolCall,
    /// The provider label wire records carry (`mcp_call.server_label`, the extension's
    /// `provider` field) — for a `baseten__<provider>__<tool>` claim, the `<provider>` segment.
    pub provider: String,
}

/// One server-tool call and what it produced — the loop's unit of tool activity. The call carries
/// the only copy of the tool's id and name, so the result can never disagree with the call it
/// answers.
#[derive(Debug, Clone)]
pub struct ToolInvocation {
    pub server_call: ServerToolCall,
    pub output: ToolOutput,
}

/// One server-tool call's terminal state, as the request transcript spells it. `Refused` never
/// reached a provider: the caller bounced the call before dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServerToolCallStatus {
    Succeeded,
    Failed,
    Refused,
}

/// What the provider reported for a call's charge, as the caller's loop validated it. Data only:
/// how a report is obtained and priced is the caller's (tool-bank's) policy.
#[derive(Debug, Clone, PartialEq)]
pub enum UsageReport {
    /// No usage block came back.
    Unreported,
    /// The provider's SKU (`None` when it reported none) and charge quantity.
    Reported { sku: Option<String>, quantity: f64 },
    /// A block was present but cannot be represented (fractional or garbage quantity); the wire
    /// renders such a call as not billable rather than re-pricing it.
    Unsupported,
}

impl UsageReport {
    pub fn sku(&self) -> Option<&str> {
        match self {
            Self::Reported { sku, .. } => sku.as_deref(),
            _ => None,
        }
    }

    pub fn quantity(&self) -> Option<f64> {
        match self {
            Self::Reported { quantity, .. } => Some(*quantity),
            _ => None,
        }
    }
}

/// The caller's billing verdict on a completed call, shared verbatim by the wire transcript and the
/// caller's billing records. `usage_expected`: the provider was configured to report usage, so
/// `Unreported` from it is a contract violation rather than a legitimate fallback.
#[derive(Debug, Clone, PartialEq)]
pub struct BillingVerdict {
    pub billable: bool,
    pub usage: UsageReport,
    pub usage_expected: bool,
}

impl BillingVerdict {
    /// A call that bills at the default quantity with no provider report — the common test shape.
    pub fn billable_unreported() -> Self {
        Self {
            billable: true,
            usage: UsageReport::Unreported,
            usage_expected: false,
        }
    }

    /// A call that does not bill (failed, refused).
    pub fn not_billable() -> Self {
        Self {
            billable: false,
            usage: UsageReport::Unreported,
            usage_expected: false,
        }
    }
}

/// A server tool's structured result, as the caller's loop produced it. A non-`Succeeded` status
/// has its `content` passed to the model verbatim to self-correct.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub content: Value,
    pub status: ServerToolCallStatus,
    /// The caller's billing verdict, echoed on the wire transcript (see
    /// [`crate::baseten_extension::WireVerdict`]).
    pub verdict: BillingVerdict,
}

impl ToolOutput {
    /// The model-facing error flag: everything short of `Succeeded`.
    pub fn is_error(&self) -> bool {
        self.status != ServerToolCallStatus::Succeeded
    }

    /// Must match byte-identically across the history append and any egress echo (KV-cache prefix).
    pub fn text(&self) -> String {
        match &self.content {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        }
    }
}

/// The client-facing remedy class of a failed request — the fault axis beside the HTTP `status`.
/// Each protocol renders it as its own `error.type` vocabulary; clients' retry wrappers act on that
/// word, so it must never claim a server fault for a caller mistake or the reverse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    InvalidRequest,
    Authentication,
    Permission,
    NotFound,
    RequestTooLarge,
    RateLimited,
    Overloaded,
    Internal,
}

impl ErrorClass {
    /// For statuses the crate did not author (a mirrored upstream response, a pre-projection guard):
    /// the class the status implies. Callers that know better pick their class directly.
    pub fn from_status(status: http::StatusCode) -> Self {
        // 529 is Baseten's non-IANA overload status (see `predict::client`).
        match status.as_u16() {
            401 => Self::Authentication,
            403 => Self::Permission,
            404 => Self::NotFound,
            413 => Self::RequestTooLarge,
            429 => Self::RateLimited,
            529 => Self::Overloaded,
            _ if status.is_client_error() => Self::InvalidRequest,
            _ => Self::Internal,
        }
    }
}

/// The translation layer's own terminal errors — what the parser and the frame channel can fail
/// with. Callers with a wider error vocabulary (tool-bank) wrap these.
#[derive(Debug, thiserror::Error)]
pub enum TranslationError {
    /// The model backend answered with a client-error status mid-stream; its status and message
    /// carry through (wrapped in the client's error envelope, never raw bytes) — a 4xx is the
    /// caller's to fix.
    #[error("model responded {status}")]
    UpstreamResponse {
        status: http::StatusCode,
        body: String,
    },
    /// The model stream was unusable — degradation with no status to mirror. `error_code` is a
    /// bounded label naming which check failed.
    #[error("model unavailable: {detail}")]
    UpstreamUnavailable {
        detail: String,
        error_code: &'static str,
    },
    /// Client closed the response stream; the caller emits no error frame.
    #[error("client disconnected")]
    ClientDisconnected,
}

/// Why ingress adaptation refused a request. Both are a 400 with the detail passed through; they
/// differ in who owns the fix.
#[derive(Debug, thiserror::Error)]
pub enum RequestRejection {
    /// The caller's body violates the contract.
    #[error("{0}")]
    Malformed(String),
    /// Valid in the client protocol, but this layer has no translation for it — ours to close.
    #[error("{0}")]
    Unsupported(String),
}

impl RequestRejection {
    pub fn detail(&self) -> &str {
        match self {
            Self::Malformed(detail) | Self::Unsupported(detail) => detail,
        }
    }

    pub fn malformed(detail: impl Into<String>) -> Self {
        Self::Malformed(detail.into())
    }
}

/// Why a request stopped early. `ReactCapExhausted` is the caller's loop cutting off a request the
/// model wanted to continue; every client stop enum is closed, so each protocol maps it to the
/// closest-fit value and `baseten.request.termination_reason` names the real cause alongside it.
/// Both are HTTP 200 successes.
#[derive(Debug, Clone, Copy)]
pub enum Termination {
    Model(crate::FinishReason),
    ReactCapExhausted,
}

/// Which setting fixed a request's ReAct iteration cap; the caller's loop telemetry reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReactCapSource {
    /// The hooks' default; the request named no cap. Raisable per request.
    ServerDefault,
    /// `baseten.tool_settings.max_react_iterations`, below the ceiling. Raisable further.
    Request,
    /// The cap equals [`crate::hooks::REACT_ITERATIONS_MAX`], whichever setting produced it: the
    /// only case no caller can raise.
    ServiceCeiling,
}

impl ReactCapSource {
    /// Stable label for a `react_cap_exhausted` disposition's `error_code`.
    pub fn error_code(self) -> &'static str {
        match self {
            Self::ServerDefault => "react_cap_server_default",
            Self::Request => "react_cap_request",
            Self::ServiceCeiling => "react_cap_service_ceiling",
        }
    }
}

/// Backend error detail for a stream that ended in failure — what the terminal error framing
/// branches on (see `StreamFraming::finish_with_backend_error`).
#[derive(Debug, Clone)]
pub struct BackendError {
    pub message: String,
    /// The upstream's HTTP status when it reported one (a mid-stream error frame's `code`).
    pub http_status: Option<u16>,
    /// The translation layer's own diagnostic code, carried on the CC/Messages error frames.
    pub error_code: Option<String>,
}

impl BackendError {
    /// A truncation-shaped backend error is a `length` finish the worker mispresented as a failure
    /// (legacy chat processors raise "Tool calls cutoff by max_tokens." instead of finishing the
    /// turn); the Responses framing re-presents it as the spec-correct incomplete shape.
    pub fn is_truncation(&self) -> bool {
        self.message
            .to_ascii_lowercase()
            .contains("cutoff by max_tokens")
    }

    /// Prompt-overflow rejections: the Baseten chat processor's
    /// "Input length N exceeds the maximum allowed input length of M tokens."
    /// and the OpenAI-style "maximum context length" phrasing.
    pub fn is_context_overflow(&self) -> bool {
        let message = self.message.to_ascii_lowercase();
        message.contains("exceeds the maximum allowed input length")
            || message.contains("maximum context length")
    }
}

impl From<&TranslationError> for BackendError {
    fn from(error: &TranslationError) -> Self {
        match error {
            TranslationError::UpstreamResponse { status, body } => Self {
                message: body.clone(),
                http_status: Some(status.as_u16()),
                error_code: None,
            },
            TranslationError::UpstreamUnavailable { detail, error_code } => Self {
                message: detail.clone(),
                http_status: None,
                error_code: Some((*error_code).to_string()),
            },
            TranslationError::ClientDisconnected => Self {
                message: "client disconnected".to_string(),
                http_status: None,
                error_code: None,
            },
        }
    }
}
