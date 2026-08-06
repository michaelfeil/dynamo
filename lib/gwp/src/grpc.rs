// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native Envoy gRPC scheduling and lifecycle services.
//!
//! Envoy buffers the OpenAI request body and sends an
//! `envoy.service.auth.v3.CheckRequest`. This adapter performs the same GWP
//! scheduling operation, then returns
//! request-header mutations for the selected workload authority. Request and
//! response streaming remain owned by Envoy.

use std::time::Instant;

use envoy_types::ext_authz::v3::pb::{
    Authorization, AuthorizationServer, CheckRequest, CheckResponse, DeniedHttpResponse,
    HeaderAppendAction, HeaderValue as EnvoyHeaderValue, HeaderValueOption, HttpResponse,
    HttpStatus, OkHttpResponse,
};
use envoy_types::pb::google::rpc::{Code as RpcCode, Status as RpcStatus};
use http::{HeaderValue, StatusCode};
use tonic::{Request, Response, Status};
use tonic_health::ServingStatus;
use tonic_health::pb::health_server::HealthServer;
use tonic_health::server::{HealthReporter, HealthService};

pub mod pb {
    tonic::include_proto!("dynamo.gwp.v1");
}

use pb::lifecycle_server::{Lifecycle, LifecycleServer};
use pb::{RequestFinishedRequest, ResponseStartedRequest};

use crate::control::{ControlError, ControlState, ResolvedSchedule, resolve_schedule};
use crate::session::{client_session_id, headers};

const GRPC_MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;
const MAX_BODY_BYTES: usize = 50 * 1024 * 1024;

const INTERNAL_REQUEST_ID_HEADER: &str = "x-gwp-internal-request-id";
const AUTHORITY_HEADER: &str = "x-gwp-authority";
const ENDPOINT_HEADER: &str = "x-gwp-endpoint-id";
const MODEL_HEADER: &str = "x-gwp-model";
const SESSION_HEADER: &str = "x-gwp-session-id";
const STICKY_HEADER: &str = "x-gwp-sticky";
const AFFINE_WORKER_ID_HEADER: &str = "x-gwp-affine-worker-id";
/// Trusted ingress override for the model used by GWP routing. The Wasm
/// request-header cleanup removes it before the request reaches the workload.
const ROUTING_MODEL_ID_HEADER: &str = "x-gwp-dynamo-model-id";

#[derive(Clone)]
pub struct GrpcAuthorization {
    state: ControlState,
}

#[derive(Clone)]
pub struct GrpcLifecycle {
    state: ControlState,
}

impl GrpcLifecycle {
    pub(crate) fn new(state: ControlState) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl Lifecycle for GrpcLifecycle {
    async fn response_started(
        &self,
        request: Request<ResponseStartedRequest>,
    ) -> Result<Response<()>, Status> {
        let started = Instant::now();
        let request = request.into_inner();
        if request.request_id.is_empty() {
            self.state.core.router.metrics().observe_lifecycle_rpc(
                "response_started",
                "invalid_argument",
                started.elapsed(),
            );
            return Err(Status::invalid_argument("request_id must not be empty"));
        }
        let upstream_status = u16::try_from(request.upstream_status)
            .ok()
            .filter(|status| (100..=599).contains(status))
            .ok_or_else(|| {
                self.state.core.router.metrics().observe_lifecycle_rpc(
                    "response_started",
                    "invalid_argument",
                    started.elapsed(),
                );
                Status::invalid_argument("upstream_status must be between 100 and 599")
            })?;
        self.state
            .core
            .response_started(
                &request.request_id,
                request.actual_worker_id,
                upstream_status,
            )
            .await;
        self.state.core.router.metrics().observe_lifecycle_rpc(
            "response_started",
            "ok",
            started.elapsed(),
        );
        Ok(Response::new(()))
    }

    async fn request_finished(
        &self,
        request: Request<RequestFinishedRequest>,
    ) -> Result<Response<()>, Status> {
        let started = Instant::now();
        let request = request.into_inner();
        if request.request_id.is_empty() {
            self.state.core.router.metrics().observe_lifecycle_rpc(
                "request_finished",
                "invalid_argument",
                started.elapsed(),
            );
            return Err(Status::invalid_argument("request_id must not be empty"));
        }
        let reason = request.finish_reason.as_deref().unwrap_or("unspecified");
        self.state
            .core
            .request_finished(&request.request_id, reason)
            .await;
        self.state.core.router.metrics().observe_lifecycle_rpc(
            "request_finished",
            "ok",
            started.elapsed(),
        );
        Ok(Response::new(()))
    }
}

impl GrpcAuthorization {
    pub(crate) fn new(state: ControlState) -> Self {
        Self { state }
    }

    async fn check_inner(&self, request: CheckRequest) -> Result<CheckResponse, ControlError> {
        let http = request
            .attributes
            .and_then(|attributes| attributes.request)
            .and_then(|request| request.http)
            .ok_or_else(|| ControlError::new(StatusCode::BAD_REQUEST, "missing HTTP attributes"))?;

        // Envoy's gRPC HttpRequest.id is an internal numeric stream id, not
        // the generated/client x-request-id used for tracing. Preserve the
        // established transaction prefix by preferring the explicit header.
        let request_id = http
            .headers
            .get("x-request-id")
            .and_then(|value| nonempty(value))
            .or_else(|| nonempty(&http.id))
            .ok_or_else(|| ControlError::new(StatusCode::BAD_REQUEST, "missing request id"))?
            .to_owned();
        let original_path = nonempty(&http.path)
            .ok_or_else(|| ControlError::new(StatusCode::BAD_REQUEST, "missing request path"))?
            .to_owned();
        let body_bytes = if http.raw_body.is_empty() {
            http.body.as_bytes()
        } else {
            http.raw_body.as_slice()
        };
        if body_bytes.len() > MAX_BODY_BYTES {
            return Err(ControlError::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                "request body exceeds 50 MiB limit",
            ));
        }
        let body = serde_json::from_slice(body_bytes).map_err(|error| {
            ControlError::new(
                StatusCode::BAD_REQUEST,
                format!("invalid JSON request body: {error}"),
            )
        })?;
        let session_id = client_session_id(|name| http.headers.get(name).map(String::as_str));
        let routing_model_id = http
            .headers
            .get(ROUTING_MODEL_ID_HEADER)
            .and_then(|value| nonempty(value))
            .map(ToOwned::to_owned);
        let routing_requirements = http
            .headers
            .get(headers::ROUTING_REQUIREMENTS)
            .map(ToOwned::to_owned);

        let resolved = resolve_schedule(
            &self.state,
            request_id,
            original_path,
            session_id,
            routing_model_id,
            routing_requirements,
            body,
        )
        .await?;
        self.success_response(resolved).await
    }

    async fn success_response(
        &self,
        resolved: ResolvedSchedule,
    ) -> Result<CheckResponse, ControlError> {
        let cleanup_request_id = resolved.request_id.clone();
        let mut headers = Vec::with_capacity(7);
        for (name, value) in [
            (INTERNAL_REQUEST_ID_HEADER, resolved.request_id),
            (AUTHORITY_HEADER, resolved.authority),
            (ENDPOINT_HEADER, resolved.endpoint_id),
            (MODEL_HEADER, resolved.model),
            (SESSION_HEADER, resolved.session_id),
            (STICKY_HEADER, resolved.sticky.to_string()),
        ] {
            if let Err(message) = validate_header_value(&value) {
                self.state
                    .core
                    .request_finished(&cleanup_request_id, "invalid_ext_authz_response_header")
                    .await;
                return Err(ControlError::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    message,
                ));
            }
            headers.push(overwrite_header(name, value));
        }
        if let Some(worker_id) = resolved.affine_worker_id {
            headers.push(overwrite_header(
                AFFINE_WORKER_ID_HEADER,
                worker_id.to_string(),
            ));
        }

        // Never trust a client-supplied affinity hint. Envoy removes the
        // incoming value before applying the authoritative header above.
        let mut headers_to_remove = vec![AFFINE_WORKER_ID_HEADER.to_owned()];
        if resolved.api_key.is_empty() {
            headers_to_remove.push("authorization".to_owned());
        } else {
            let authorization = format!("Bearer {}", resolved.api_key);
            if let Err(message) = validate_header_value(&authorization) {
                self.state
                    .core
                    .request_finished(&cleanup_request_id, "invalid_ext_authz_response_header")
                    .await;
                return Err(ControlError::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    message,
                ));
            }
            headers.push(overwrite_header("authorization", authorization));
        }

        Ok(CheckResponse {
            status: Some(rpc_status(RpcCode::Ok, "")),
            http_response: Some(HttpResponse::OkResponse(OkHttpResponse {
                headers,
                headers_to_remove,
                ..Default::default()
            })),
            ..Default::default()
        })
    }
}

#[tonic::async_trait]
impl Authorization for GrpcAuthorization {
    async fn check(
        &self,
        request: Request<CheckRequest>,
    ) -> Result<Response<CheckResponse>, Status> {
        let started = Instant::now();
        let body_bytes = request
            .get_ref()
            .attributes
            .as_ref()
            .and_then(|attributes| attributes.request.as_ref())
            .and_then(|request| request.http.as_ref())
            .map(|http| {
                if http.raw_body.is_empty() {
                    http.body.len()
                } else {
                    http.raw_body.len()
                }
            });
        let response = match self.check_inner(request.into_inner()).await {
            Ok(response) => response,
            Err(error) => denied_response(error),
        };
        let result = response_result(&response);
        self.state.core.router.metrics().observe_schedule_service(
            "ext_authz",
            result,
            started.elapsed(),
            body_bytes.map(|bytes| bytes as u64),
        );
        Ok(Response::new(response))
    }
}

pub(crate) fn authorization_server(state: ControlState) -> AuthorizationServer<GrpcAuthorization> {
    AuthorizationServer::new(GrpcAuthorization::new(state))
        .max_decoding_message_size(GRPC_MAX_MESSAGE_BYTES)
}

pub(crate) fn lifecycle_server(state: ControlState) -> LifecycleServer<GrpcLifecycle> {
    LifecycleServer::new(GrpcLifecycle::new(state))
}

pub(crate) async fn health_server(
    lifecycle: crate::lifecycle::Lifecycle,
) -> (HealthServer<HealthService>, tokio::task::JoinHandle<()>) {
    let reporter = HealthReporter::new();
    publish_health(&reporter, &lifecycle).await;
    let service = HealthServer::new(HealthService::from_health_reporter(reporter.clone()));
    let task = tokio::spawn(async move {
        let mut revision = lifecycle.subscribe();
        loop {
            match lifecycle.next_time_transition() {
                Some(deadline) => {
                    tokio::select! {
                        changed = revision.changed() => {
                            if changed.is_err() {
                                return;
                            }
                        }
                        _ = tokio::time::sleep_until(deadline) => {}
                    }
                }
                None => {
                    if revision.changed().await.is_err() {
                        return;
                    }
                }
            }
            publish_health(&reporter, &lifecycle).await;
        }
    });
    (service, task)
}

async fn publish_health(reporter: &HealthReporter, lifecycle: &crate::lifecycle::Lifecycle) {
    reporter
        .set_service_status("gwp-liveness", ServingStatus::Serving)
        .await;
    reporter
        .set_service_status(
            "gwp-readiness",
            if lifecycle.is_ready() {
                ServingStatus::Serving
            } else {
                ServingStatus::NotServing
            },
        )
        .await;
}

fn nonempty(value: &str) -> Option<&str> {
    (!value.is_empty()).then_some(value)
}

fn validate_header_value(value: &str) -> Result<(), &'static str> {
    HeaderValue::from_str(value)
        .map(|_| ())
        .map_err(|_| "scheduled route contains an invalid HTTP header value")
}

fn overwrite_header(key: impl Into<String>, value: impl Into<String>) -> HeaderValueOption {
    #[allow(deprecated)]
    HeaderValueOption {
        header: Some(EnvoyHeaderValue {
            key: key.into(),
            value: value.into(),
            raw_value: Vec::new(),
        }),
        append: None,
        append_action: HeaderAppendAction::OverwriteIfExistsOrAdd.into(),
        keep_empty_value: false,
    }
}

fn denied_response(error: ControlError) -> CheckResponse {
    let code = match error.status.as_u16() {
        400 | 413 => RpcCode::InvalidArgument,
        409 => RpcCode::AlreadyExists,
        429 => RpcCode::ResourceExhausted,
        400..=499 => RpcCode::InvalidArgument,
        500..=599 => RpcCode::Unavailable,
        _ => RpcCode::Unknown,
    };
    CheckResponse {
        status: Some(rpc_status(code, error.message.clone())),
        http_response: Some(HttpResponse::DeniedResponse(DeniedHttpResponse {
            status: Some(HttpStatus {
                code: i32::from(error.status.as_u16()),
            }),
            headers: vec![overwrite_header(
                "content-type",
                "text/plain; charset=utf-8",
            )],
            body: error.message,
        })),
        ..Default::default()
    }
}

fn rpc_status(code: RpcCode, message: impl Into<String>) -> RpcStatus {
    RpcStatus {
        code: code.into(),
        message: message.into(),
        details: Vec::new(),
    }
}

fn response_result(response: &CheckResponse) -> &'static str {
    match &response.http_response {
        Some(HttpResponse::OkResponse(_)) => "success",
        Some(HttpResponse::DeniedResponse(denied)) => match denied
            .status
            .as_ref()
            .map(|status| status.code)
            .unwrap_or(403)
        {
            429 | 503 | 529 => "overloaded",
            400..=499 => "client_error",
            _ => "internal_error",
        },
        Some(HttpResponse::ErrorResponse(_)) | None => "internal_error",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, HashMap};
    use std::sync::Arc;
    use std::time::Duration;

    use dynamo_llm::local_model::runtime_config::ModelRuntimeConfig;
    use envoy_types::pb::envoy::service::auth::v3::{AttributeContext, attribute_context};
    use prost::Message;
    use tonic_health::pb::HealthCheckRequest;
    use tonic_health::pb::health_server::Health;

    use crate::config::{EndpointConfig, EndpointId, GwpConfig, ModelRoute, RoutingConfig};
    use crate::core::GwpCore;
    use crate::session::InMemoryAffinityStore;
    use crate::topology::{TopologySnapshot, TopologyStore, TopologyWorker};

    async fn test_state() -> ControlState {
        let endpoint = EndpointId("test-endpoint".into());
        let config = GwpConfig {
            endpoints: BTreeMap::from([(
                endpoint.clone(),
                EndpointConfig {
                    ingress_url: url::Url::parse("http://example.test/v1").unwrap(),
                    api_key: String::new(),
                    planner_url: url::Url::parse("http://planner.test/deep/health").unwrap(),
                    planner_api_key: None,
                    properties: Default::default(),
                },
            )]),
            routes: vec![ModelRoute {
                models: vec!["test-model".into()],
                endpoints: vec![endpoint.clone()],
            }],
            routing: RoutingConfig {
                block_size: 4,
                ..Default::default()
            },
            ..Default::default()
        };
        let (router, workers_tx) = crate::router::GwpRouter::new_process_local(
            config.routing.block_size,
            config.routing.approx_indexer_ttl_secs,
        )
        .await
        .unwrap();
        let topology = TopologyStore::new(
            TopologySnapshot::from_config(
                &config,
                HashMap::from([(
                    10,
                    TopologyWorker {
                        endpoint,
                        runtime: ModelRuntimeConfig::default(),
                        observed_load: None,
                    },
                )]),
            )
            .unwrap(),
        );
        workers_tx
            .send(HashMap::from([(10, ModelRuntimeConfig::default())]))
            .unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        ControlState {
            core: GwpCore::new(
                router,
                Arc::new(InMemoryAffinityStore::new()),
                topology,
                Arc::new(config),
            ),
            lifecycle: crate::lifecycle::Lifecycle::ready_for_tests(),
        }
    }

    fn check_request(body: serde_json::Value, headers: HashMap<String, String>) -> CheckRequest {
        CheckRequest {
            attributes: Some(AttributeContext {
                request: Some(attribute_context::Request {
                    http: Some(attribute_context::HttpRequest {
                        id: "stream-1".into(),
                        method: "POST".into(),
                        headers,
                        path: "/v1/chat/completions".into(),
                        host: "gwp.test".into(),
                        scheme: "http".into(),
                        raw_body: serde_json::to_vec(&body).unwrap(),
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            }),
        }
    }

    #[tokio::test]
    async fn routing_model_header_overrides_body_model() {
        let state = test_state().await;
        let service = GrpcAuthorization::new(state.clone());
        let response = service
            .check_inner(check_request(
                serde_json::json!({
                    "model": "not-routed",
                    "messages": [{"role": "user", "content": "hello"}],
                }),
                HashMap::from([
                    ("x-request-id".into(), "override-rid".into()),
                    (ROUTING_MODEL_ID_HEADER.into(), "test-model".into()),
                ]),
            ))
            .await
            .unwrap();
        let HttpResponse::OkResponse(ok) = response.http_response.unwrap() else {
            panic!("expected successful ext-authz response");
        };
        let headers: HashMap<_, _> = ok
            .headers
            .iter()
            .filter_map(|option| option.header.as_ref())
            .map(|header| (header.key.as_str(), header.value.as_str()))
            .collect();
        assert_eq!(headers.get(MODEL_HEADER), Some(&"test-model"));
        let request_id = *headers.get(INTERNAL_REQUEST_ID_HEADER).unwrap();
        state
            .core
            .request_finished(request_id, "test_cleanup")
            .await;
    }

    #[tokio::test]
    async fn missing_or_unknown_model_is_a_client_error() {
        let service = GrpcAuthorization::new(test_state().await);
        for body in [
            serde_json::json!({"messages": []}),
            serde_json::json!({"model": 42, "messages": []}),
            serde_json::json!({"model": "not-routed", "messages": []}),
        ] {
            let error = service
                .check_inner(check_request(
                    body,
                    HashMap::from([("x-request-id".into(), "invalid-model-rid".into())]),
                ))
                .await
                .unwrap_err();
            assert_eq!(error.status, StatusCode::BAD_REQUEST);
            assert!(error.message.contains("model"));
        }
    }

    #[tokio::test]
    async fn affine_worker_header_is_emitted_only_for_an_affinity_hit() {
        let service = GrpcAuthorization::new(test_state().await);
        for (affine_worker_id, expected) in [(None, None), (Some(42), Some("42"))] {
            let response = service
                .success_response(ResolvedSchedule {
                    request_id: "header-test".into(),
                    authority: "example.test:80".into(),
                    endpoint_id: "test-endpoint".into(),
                    model: "test-model".into(),
                    session_id: "session".into(),
                    sticky: affine_worker_id.is_some(),
                    affine_worker_id,
                    api_key: String::new(),
                })
                .await
                .unwrap();
            let HttpResponse::OkResponse(ok) = response.http_response.unwrap() else {
                panic!("expected successful ext-authz response");
            };
            assert!(
                ok.headers_to_remove
                    .iter()
                    .any(|header| header == AFFINE_WORKER_ID_HEADER)
            );
            let actual = ok
                .headers
                .iter()
                .filter_map(|option| option.header.as_ref())
                .find(|header| header.key == AFFINE_WORKER_ID_HEADER)
                .map(|header| header.value.as_str());
            assert_eq!(actual, expected);
        }
    }

    #[tokio::test]
    async fn lifecycle_rpcs_are_idempotent_for_unknown_requests() {
        let service = GrpcLifecycle::new(test_state().await);
        service
            .response_started(Request::new(ResponseStartedRequest {
                request_id: "unknown".into(),
                actual_worker_id: Some(10),
                upstream_status: 200,
            }))
            .await
            .unwrap();
        service
            .request_finished(Request::new(RequestFinishedRequest {
                request_id: "unknown".into(),
                finish_reason: Some("complete".into()),
            }))
            .await
            .unwrap();

        let error = service
            .request_finished(Request::new(RequestFinishedRequest {
                request_id: String::new(),
                finish_reason: None,
            }))
            .await
            .unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);

        let error = service
            .response_started(Request::new(ResponseStartedRequest {
                request_id: "unknown".into(),
                actual_worker_id: None,
                upstream_status: 0,
            }))
            .await
            .unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
    }

    #[tokio::test(start_paused = true)]
    async fn grpc_health_tracks_warmup_and_drain() {
        let lifecycle = crate::lifecycle::Lifecycle::starting(1, Duration::from_secs(2));
        lifecycle.update_routing(1);
        let reporter = HealthReporter::new();
        publish_health(&reporter, &lifecycle).await;
        let service = HealthService::from_health_reporter(reporter.clone());

        let status = service
            .check(Request::new(HealthCheckRequest {
                service: "gwp-readiness".into(),
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(status.status, 2); // NOT_SERVING

        tokio::time::advance(Duration::from_secs(2)).await;
        publish_health(&reporter, &lifecycle).await;
        let status = service
            .check(Request::new(HealthCheckRequest {
                service: "gwp-readiness".into(),
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(status.status, 1); // SERVING

        lifecycle.begin_draining();
        publish_health(&reporter, &lifecycle).await;
        let readiness = service
            .check(Request::new(HealthCheckRequest {
                service: "gwp-readiness".into(),
            }))
            .await
            .unwrap()
            .into_inner();
        let liveness = service
            .check(Request::new(HealthCheckRequest {
                service: "gwp-liveness".into(),
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(readiness.status, 2);
        assert_eq!(liveness.status, 1);
    }

    #[test]
    fn queued_lifecycle_event_uses_rpc_message_encoding() {
        let event = pb::QueuedLifecycleEvent {
            event: Some(pb::queued_lifecycle_event::Event::ResponseStarted(
                ResponseStartedRequest {
                    request_id: "rid".into(),
                    actual_worker_id: Some(42),
                    upstream_status: 429,
                },
            )),
        };
        let decoded = pb::QueuedLifecycleEvent::decode(event.encode_to_vec().as_slice()).unwrap();
        assert_eq!(decoded, event);
    }

    #[tokio::test]
    async fn tonic_server_serves_schedule_lifecycle_and_health() {
        let state = test_state().await;
        let core = state.core.clone();
        let (health, health_task) = health_server(state.lifecycle.clone()).await;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
        let server = tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(authorization_server(state.clone()))
                .add_service(lifecycle_server(state))
                .add_service(health)
                .serve_with_incoming(incoming),
        );
        let endpoint = format!("http://{address}");
        let channel = tonic::transport::Endpoint::from_shared(endpoint)
            .unwrap()
            .connect()
            .await
            .unwrap();

        let mut health_client = tonic_health::pb::health_client::HealthClient::new(channel.clone());
        let readiness = health_client
            .check(HealthCheckRequest {
                service: "gwp-readiness".into(),
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(readiness.status, 1);

        let body = serde_json::to_vec(&serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
        }))
        .unwrap();
        let check = CheckRequest {
            attributes: Some(AttributeContext {
                request: Some(attribute_context::Request {
                    http: Some(attribute_context::HttpRequest {
                        id: "stream-1".into(),
                        method: "POST".into(),
                        headers: HashMap::from([("x-request-id".into(), "external-rid".into())]),
                        path: "/v1/chat/completions".into(),
                        host: "gwp.test".into(),
                        scheme: "http".into(),
                        raw_body: body,
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            }),
        };
        let mut auth_client = envoy_types::pb::envoy::service::auth::v3::authorization_client::AuthorizationClient::new(channel.clone());
        let response = auth_client.check(check).await.unwrap().into_inner();
        let HttpResponse::OkResponse(ok) = response.http_response.unwrap() else {
            panic!("expected successful ext-authz response");
        };
        let request_id = ok
            .headers
            .iter()
            .filter_map(|option| option.header.as_ref())
            .find(|header| header.key == INTERNAL_REQUEST_ID_HEADER)
            .unwrap()
            .value
            .clone();
        assert_eq!(core.inflight_len(), 1);

        let mut lifecycle_client = pb::lifecycle_client::LifecycleClient::new(channel);
        lifecycle_client
            .response_started(ResponseStartedRequest {
                request_id: request_id.clone(),
                actual_worker_id: Some(10),
                upstream_status: 200,
            })
            .await
            .unwrap();
        lifecycle_client
            .request_finished(RequestFinishedRequest {
                request_id,
                finish_reason: Some("complete".into()),
            })
            .await
            .unwrap();
        assert_eq!(core.inflight_len(), 0);

        health_task.abort();
        server.abort();
    }
}
