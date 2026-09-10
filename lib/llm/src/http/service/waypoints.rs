// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Internal inspection invokes the production router with a private request extension.
//! The requested boundary is returned; `preserve_intermediates` additionally keeps every earlier
//! boundary the same run passed through, so one call yields the whole request path instead of one
//! artifact per call. The collector is request-local — it lives on the extension and dies with the
//! request — not a process-wide artifact registry.
//! The listener receives the same router value as serving, including its middleware layers.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use axum::{
    Extension, Json, Router,
    body::Body,
    extract::State,
    http::Request,
    response::{IntoResponse, Response},
    routing::post,
};
use dynamo_runtime::{
    engine::{AsyncEngineContextProvider, ResponseStream},
    pipeline::{AsyncEngine, Context, ManyOut, SingleIn},
    protocols::annotated::Annotated,
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, value::RawValue};
use tokio_util::sync::CancellationToken;
use tower::ServiceExt;

use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use crate::protocols::unified::{AnthropicContext, ResponsesContext};
use crate::types::openai::chat_completions::OpenAIChatCompletionsStreamingEngine;

const KEY: &str = "waypoints";
pub const MAX_BYTES: usize = 4 * 1024 * 1024;
pub type Hook = Arc<dyn AsyncEngine<SingleIn<Value>, ManyOut<Annotated<Value>>, anyhow::Error>>;
pub type HookSlot = Arc<OnceLock<Hook>>;

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Protocol {
    Chat,
    Messages,
    Responses,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Stage {
    Canonical,
    Render,
    Tokenize,
    EngineRequest,
    EngineOutput,
    ChatStream,
    Client,
    #[serde(skip_deserializing)]
    Error,
}

#[derive(Clone)]
pub(crate) struct Inspection {
    stop_after: Stage,
    engine_output: Option<Vec<Map<String, Value>>>,
    hook: Option<Hook>,
    preserve_intermediates: bool,
    /// Boundaries reached before `stop_after`. Cloning an `Inspection` shares this, so a checkpoint
    /// reached inside the handler is readable back in `inspect`.
    intermediates: Arc<Mutex<BTreeMap<Stage, Value>>>,
}

impl Inspection {
    /// Whether this run keeps `stage`. The requested boundary is returned as the artifact itself,
    /// so it is never also an intermediate.
    fn wants(&self, stage: Stage) -> bool {
        self.preserve_intermediates && stage < self.stop_after
    }

    fn record(&self, stage: Stage, value: Value) -> Result<(), super::error::HttpError> {
        let mut intermediates = self
            .intermediates
            .lock()
            .map_err(|_| invalid_hook("artifact collector poisoned"))?;
        if intermediates.contains_key(&stage) {
            return Err(invalid_hook(format!("repeated {stage:?} artifact")));
        }
        intermediates.insert(stage, value);
        let size = serde_json::to_vec(&*intermediates)
            .map_err(|err| invalid_hook(err.to_string()))?
            .len();
        if size > MAX_BYTES {
            intermediates.remove(&stage);
            return Err(super::error::HttpError {
                code: 413,
                message: "Waypoints capture exceeds byte limit".into(),
            });
        }
        Ok(())
    }

    fn take(&self) -> BTreeMap<Stage, Value> {
        self.intermediates
            .lock()
            .map(|mut intermediates| std::mem::take(&mut *intermediates))
            .unwrap_or_default()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InspectRequest {
    protocol: Protocol,
    // The real handler must see the original fields, including ones it rejects or preserves.
    request: Box<RawValue>,
    stop_after: Stage,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    #[serde(default)]
    engine_output: Option<Vec<Map<String, Value>>>,
    /// Also return every boundary before `stop_after` that this run passed through.
    #[serde(default)]
    preserve_intermediates: bool,
    #[serde(default = "default_timeout")]
    timeout_ms: u64,
}

#[derive(Serialize)]
struct Artifact<T> {
    stage: Stage,
    value: T,
}

/// An artifact plus the boundaries the same run passed through before it.
#[derive(Serialize)]
struct Inspected<T> {
    #[serde(flatten)]
    artifact: T,
    #[serde(skip_serializing_if = "Option::is_none")]
    intermediates: Option<BTreeMap<Stage, Value>>,
}

#[derive(Serialize)]
struct HookRequest<'a> {
    request: &'a NvCreateChatCompletionRequest,
    stop_after: Stage,
    engine_output: &'a Option<Vec<Map<String, Value>>>,
    max_bytes: usize,
    preserve_intermediates: bool,
}

#[derive(Serialize)]
#[serde(untagged)]
pub(crate) enum CanonicalContext<'a> {
    Chat {},
    Messages {
        api_context: &'a Option<AnthropicContext>,
        losses: &'a Option<Vec<b10_dynamo_api_translation::Loss>>,
        prompt_injected_reasoning: bool,
    },
    Responses {
        api_context: &'a Option<ResponsesContext>,
        losses: &'a Option<Vec<b10_dynamo_api_translation::Loss>>,
        preserve_omitted_max_tokens: bool,
    },
}

#[derive(Serialize)]
struct CanonicalValue<'a> {
    request: &'a NvCreateChatCompletionRequest,
    context: CanonicalContext<'a>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct EngineRequestArtifact {
    // These models belong to the Python processor, not the native Rust token engine.
    request: Map<String, Value>,
    routing_constraints: Option<Map<String, Value>>,
    routing_priority: Option<Map<String, Value>>,
    do_not_queue: bool,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ChatChunk {
    #[serde(flatten)]
    inner: dynamo_protocols::types::CreateChatCompletionStreamResponse,
    #[serde(skip_serializing_if = "Option::is_none")]
    nvext: Option<Value>,
    // Python may expose this backend-owned metadata alongside the production CC chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_cache_metrics: Option<Map<String, Value>>,
}

#[derive(Deserialize, Serialize)]
#[serde(
    tag = "stage",
    content = "value",
    rename_all = "snake_case",
    deny_unknown_fields
)]
enum HookArtifact {
    Render(Option<String>),
    Tokenize(Vec<u32>),
    EngineRequest(EngineRequestArtifact),
    EngineOutput(Vec<Map<String, Value>>),
    ChatStream(Vec<ChatChunk>),
}

impl HookArtifact {
    fn stage(&self) -> Stage {
        match self {
            Self::Render(_) => Stage::Render,
            Self::Tokenize(_) => Stage::Tokenize,
            Self::EngineRequest(_) => Stage::EngineRequest,
            Self::EngineOutput(_) => Stage::EngineOutput,
            Self::ChatStream(_) => Stage::ChatStream,
        }
    }

    /// The payload without its stage tag, for the intermediates map.
    fn into_value(self) -> Result<Value, serde_json::Error> {
        let mut tagged = serde_json::to_value(self)?;
        Ok(tagged["value"].take())
    }
}

#[derive(Serialize)]
#[serde(untagged)]
enum ClientBody {
    // Already serialized by the real protocol writer; do not reinterpret its schema.
    Json(Box<RawValue>),
    Stream(String),
}

#[derive(Serialize)]
struct ClientValue {
    status: u16,
    content_type: String,
    body: ClientBody,
}

#[derive(Serialize)]
struct FailureValue {
    requested_stage: Stage,
    #[serde(flatten)]
    response: ClientValue,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    capture_truncated: bool,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    error: &'a str,
}

fn default_timeout() -> u64 {
    30_000
}

pub(crate) fn attach<T: Send + Sync + 'static>(
    request: &mut Context<T>,
    inspection: Option<Extension<Inspection>>,
) {
    if let Some(Extension(inspection)) = inspection {
        request.insert(KEY, inspection);
    }
}

pub(crate) fn canonical(
    request: &Context<NvCreateChatCompletionRequest>,
    context: CanonicalContext<'_>,
) -> Option<Response> {
    let inspection = request.get::<Inspection>(KEY).ok()?;
    if inspection.stop_after != Stage::Canonical && !inspection.wants(Stage::Canonical) {
        return None;
    }
    let value = CanonicalValue {
        request: request.content(),
        context,
    };
    if inspection.stop_after != Stage::Canonical {
        // Recorded rather than returned: canonical never reaches the hook, so this is the only
        // place it can be preserved.
        let result = serde_json::to_value(&value)
            .map_err(|err| invalid_hook(err.to_string()))
            .and_then(|value| inspection.record(Stage::Canonical, value));
        if let Err(err) = result {
            return Some(error(err.code, &err.message));
        }
        return None;
    }
    Some(
        Json(Artifact {
            stage: Stage::Canonical,
            value,
        })
        .into_response(),
    )
}

pub(crate) fn active<T: Send + Sync + 'static>(request: &Context<T>) -> bool {
    request.get::<Inspection>(KEY).is_ok()
}

pub(crate) enum Generated {
    Stopped(Response),
    Stream(ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>),
}

/// The normal engine is untouched unless this request came from the internal router.
pub(crate) async fn generate(
    engine: OpenAIChatCompletionsStreamingEngine,
    request: Context<NvCreateChatCompletionRequest>,
) -> anyhow::Result<Generated> {
    let Ok(inspection) = request.get::<Inspection>(KEY) else {
        return Ok(Generated::Stream(engine.generate(request).await?));
    };
    let hook = inspection
        .hook
        .as_ref()
        .ok_or_else(|| super::error::HttpError {
            code: 400,
            message: "this frontend has no Waypoints Python hook".into(),
        })?;
    let context = request.context();
    let payload = serde_json::to_value(HookRequest {
        request: request.content(),
        stop_after: inspection.stop_after,
        engine_output: &inspection.engine_output,
        max_bytes: MAX_BYTES,
        preserve_intermediates: inspection.preserve_intermediates,
    })
    .map_err(|err| invalid_hook(format!("cannot serialize request: {err}")))?;
    let payload = request.map(|_| payload);
    let mut stream = hook.generate(payload).await?;
    let mut chunks = Vec::new();
    let mut stopped = None;
    let mut bytes = 0;
    while let Some(item) = stream.next().await {
        if let Some((message, status)) = super::openai::extract_backend_error_if_present(&item) {
            return Err(super::error::HttpError {
                code: status.as_u16(),
                message,
            }
            .into());
        }
        let value = item.data.ok_or_else(|| invalid_hook("missing result"))?;
        bytes += serde_json::to_vec(&value)?.len();
        if bytes > MAX_BYTES {
            return Err(super::error::HttpError {
                code: 413,
                message: "Waypoints capture exceeds byte limit".into(),
            }
            .into());
        }
        // The hook stream is drained even once the requested boundary has arrived: preserved
        // intermediates may follow it, and the python generator finishes cleanly.
        if value.get("stage").is_some() {
            let artifact: HookArtifact = serde_json::from_value(value).map_err(|err| {
                invalid_hook(format!(
                    "expected {:?} artifact: {err}",
                    inspection.stop_after
                ))
            })?;
            let stage = artifact.stage();
            if stage == inspection.stop_after {
                if matches!(artifact, HookArtifact::Render(None)) {
                    return Err(invalid_hook("requested render artifact has no text").into());
                }
                if stopped.is_some() || !chunks.is_empty() {
                    return Err(invalid_hook(format!("repeated {stage:?} artifact")).into());
                }
                stopped = Some(artifact);
                continue;
            }
            if !inspection.wants(stage) {
                return Err(invalid_hook(format!(
                    "expected {:?}, received {:?}",
                    inspection.stop_after, stage
                ))
                .into());
            }
            let value = artifact
                .into_value()
                .map_err(|err| invalid_hook(format!("cannot preserve {stage:?}: {err}")))?;
            inspection.record(stage, value)?;
            continue;
        }
        if inspection.stop_after != Stage::Client {
            return Err(invalid_hook(format!(
                "expected {:?} artifact, received a chat chunk",
                inspection.stop_after
            ))
            .into());
        }
        let chunk: ChatChunk = serde_json::from_value(value)
            .map_err(|err| invalid_hook(format!("expected client chat chunk: {err}")))?;
        chunks.push(Annotated::from_data(NvCreateChatCompletionStreamResponse {
            inner: chunk.inner,
            nvext: chunk.nvext,
        }));
    }
    if let Some(artifact) = stopped {
        return Ok(Generated::Stopped(Json(artifact).into_response()));
    }
    if inspection.stop_after != Stage::Client {
        return Err(invalid_hook(format!(
            "requested {:?} boundary was not reached",
            inspection.stop_after
        ))
        .into());
    }
    Ok(Generated::Stream(ResponseStream::new(
        Box::pin(futures::stream::iter(chunks)),
        context,
    )))
}

pub fn router(production: Router, hook: HookSlot) -> Router {
    Router::new()
        .route("/v1/waypoints", post(inspect))
        .layer(axum::extract::DefaultBodyLimit::max(MAX_BYTES))
        .with_state((production, hook))
}

async fn inspect(
    State((production, hook)): State<(Router, HookSlot)>,
    Json(spec): Json<InspectRequest>,
) -> Response {
    let path = match spec.protocol {
        Protocol::Chat => "/v1/chat/completions",
        Protocol::Messages => "/v1/messages",
        Protocol::Responses => "/v1/responses",
    };
    if !(1..=300_000).contains(&spec.timeout_ms) {
        return error(400, "timeout_ms must be between 1 and 300000");
    }
    let mut request = Request::builder().method("POST").uri(path);
    // We construct a new JSON message, not a byte-for-byte HTTP replay.
    let connection_headers: Vec<_> = spec
        .headers
        .iter()
        .filter(|(name, _)| name.eq_ignore_ascii_case("connection"))
        .flat_map(|(_, value)| {
            value
                .split(',')
                .map(|name| name.trim().to_ascii_lowercase())
        })
        .collect();
    for (name, value) in spec.headers {
        let lower = name.to_ascii_lowercase();
        if lower.starts_with("content-")
            || connection_headers.contains(&lower)
            || matches!(
                lower.as_str(),
                "host"
                    | "connection"
                    | "keep-alive"
                    | "proxy-authenticate"
                    | "proxy-authorization"
                    | "te"
                    | "trailer"
                    | "transfer-encoding"
                    | "upgrade"
            )
        {
            continue;
        }
        request = request.header(name, value);
    }
    request = request.header("content-type", "application/json");
    let Ok(mut request) = request.body(Body::from(spec.request.get().to_owned())) else {
        return error(400, "invalid captured headers");
    };
    let inspection = Inspection {
        stop_after: spec.stop_after,
        engine_output: spec.engine_output,
        hook: hook.get().cloned(),
        preserve_intermediates: spec.preserve_intermediates,
        intermediates: Default::default(),
    };
    // Kept here too: the collector is shared with the clone the request carries, so boundaries
    // recorded inside the handler are readable once it returns.
    let collector = inspection.clone();
    request.extensions_mut().insert(inspection);
    // Bound both handler execution (including preprocessing) and streamed response collection.
    let result = tokio::time::timeout(Duration::from_millis(spec.timeout_ms), async move {
        let response = production
            .oneshot(request)
            .await
            .expect("Router is infallible");
        let status = response.status();
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_owned();
        let mut body = response.into_body().into_data_stream();
        let mut bytes = Vec::new();
        while let Some(chunk) = body.next().await {
            let chunk = chunk.map_err(|_| (502, "Failed to collect frontend response body"))?;
            if chunk.len() > MAX_BYTES - bytes.len() {
                return Err((413, "Waypoints result exceeds capture limit"));
            }
            bytes.extend_from_slice(&chunk);
        }
        Ok::<_, (u16, &str)>((status, content_type, bytes))
    })
    .await;
    let (status, content_type, bytes) = match result {
        Ok(Ok(response)) => response,
        Ok(Err((status, message))) => return failed(status, message, &collector),
        Err(_) => return failed(408, "Waypoints request timed out", &collector),
    };
    if !status.is_success() {
        // Preserve the production error without interpreting each protocol's error schema.
        let body = serde_json::from_slice(&bytes)
            .map(ClientBody::Json)
            .unwrap_or_else(|_| ClientBody::Stream(String::from_utf8_lossy(&bytes).into_owned()));
        return failure(
            ClientValue {
                status: status.as_u16(),
                content_type,
                body,
            },
            &collector,
        );
    }
    if spec.stop_after != Stage::Client {
        let artifact: Value = match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(_) => return failed(502, "invalid frontend JSON", &collector),
        };
        return inspected(artifact, &collector);
    }
    let body = if content_type.starts_with("application/json") {
        match serde_json::from_slice(&bytes) {
            Ok(value) => ClientBody::Json(value),
            Err(_) => return failed(502, "invalid frontend JSON", &collector),
        }
    } else {
        match String::from_utf8(bytes.to_vec()) {
            Ok(value) => ClientBody::Stream(value),
            Err(_) => return failed(502, "invalid frontend UTF-8", &collector),
        }
    };
    inspected(
        Artifact {
            stage: Stage::Client,
            value: ClientValue {
                status: status.as_u16(),
                content_type,
                body,
            },
        },
        &collector,
    )
}

fn inspected<T: Serialize>(artifact: T, collector: &Inspection) -> Response {
    let result = Inspected {
        artifact,
        intermediates: collector.preserve_intermediates.then(|| collector.take()),
    };
    match serde_json::to_vec(&result) {
        Ok(bytes) if bytes.len() <= MAX_BYTES => json_bytes(bytes),
        Ok(_) => failure_with_intermediates(
            diagnostic_error(413, "Waypoints capture exceeds byte limit"),
            collector.stop_after,
            result.intermediates,
        ),
        Err(_) => failure_with_intermediates(
            diagnostic_error(502, "invalid frontend JSON"),
            collector.stop_after,
            result.intermediates,
        ),
    }
}

fn diagnostic_error(status: u16, message: &str) -> ClientValue {
    ClientValue {
        status,
        content_type: "application/json".into(),
        body: ClientBody::Json(
            serde_json::value::to_raw_value(&ErrorBody { error: message })
                .expect("serializable error message"),
        ),
    }
}

fn failed(status: u16, message: &str, collector: &Inspection) -> Response {
    failure(diagnostic_error(status, message), collector)
}

fn failure(response: ClientValue, collector: &Inspection) -> Response {
    failure_with_intermediates(
        response,
        collector.stop_after,
        collector.preserve_intermediates.then(|| collector.take()),
    )
}

fn failure_with_intermediates(
    response: ClientValue,
    requested_stage: Stage,
    intermediates: Option<BTreeMap<Stage, Value>>,
) -> Response {
    let mut result = Inspected {
        artifact: Artifact {
            stage: Stage::Error,
            value: FailureValue {
                requested_stage,
                response,
                capture_truncated: false,
            },
        },
        intermediates,
    };
    loop {
        let bytes = serde_json::to_vec(&result).expect("serializable captured JSON");
        if bytes.len() <= MAX_BYTES {
            return json_bytes(bytes);
        }
        result.artifact.value.capture_truncated = true;
        // Keep the longest complete prefix that fits, reserving room for the terminal error.
        if result
            .intermediates
            .as_mut()
            .and_then(BTreeMap::pop_last)
            .is_none()
        {
            result.artifact.value.response = diagnostic_error(
                result.artifact.value.response.status,
                "Original error body omitted: Waypoints capture exceeds byte limit",
            );
        }
    }
}

fn json_bytes(bytes: Vec<u8>) -> Response {
    ([("content-type", "application/json")], bytes).into_response()
}

fn error(status: u16, message: &str) -> Response {
    (
        axum::http::StatusCode::from_u16(status).expect("constant status"),
        Json(ErrorBody { error: message }),
    )
        .into_response()
}

fn invalid_hook(message: impl Into<String>) -> super::error::HttpError {
    super::error::HttpError {
        code: 502,
        message: format!("Waypoints hook: {}", message.into()),
    }
}

async fn bind_listener(host: &str, port: u16) -> std::io::Result<tokio::net::TcpListener> {
    tokio::net::TcpListener::bind((host, port)).await
}

pub(crate) fn spawn(production: Router, hook: HookSlot, host: String, cancel: CancellationToken) {
    use dynamo_runtime::config::{env_is_truthy, environment_names::llm};
    if env_is_truthy(llm::DYN_WAYPOINTS_DISABLE) {
        return;
    }
    tokio::spawn(async move {
        let port = std::env::var(llm::DYN_WAYPOINTS_PORT).unwrap_or_else(|_| "9192".into());
        let result = async {
            let port: u16 = port.parse()?;
            let listener = bind_listener(&host, port).await?;
            tracing::info!(host, port, "Waypoints internal listener started");
            axum::serve(listener, router(production, hook))
                .with_graceful_shutdown(cancel.cancelled_owned())
                .await?;
            anyhow::Ok(())
        }
        .await;
        if let Err(error) = result {
            tracing::error!(%error, "Waypoints listener failed");
        }
    });
}

#[cfg(test)]
mod tests;
