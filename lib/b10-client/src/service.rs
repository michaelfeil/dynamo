// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Low-overhead HTTP service for generation coordination.
//!
//! Python may own process startup and supervision, but request decoding,
//! coordination, and response framing remain entirely in Rust.

use crate::protocol::codec::{decode_new_request, denial_frame, trace_context_from_wire};
use crate::protocol::{
    self, GenerationResponseFrameV1, NewRequestV1, ProtocolErrorV1, generation_response_frame_v1,
};
use crate::{
    DisaggregationStrategy, GenerationCoordinatorClient, GenerationOptions, GenerationOutcome,
    RequestContext,
};
use anyhow::{Context, Result};
use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{Response, StatusCode, header};
use axum::routing::{get, post};
use dynamo_runtime::pipeline::AsyncEngineContext;
use dynamo_runtime::pipeline::context::Controller;
use futures::{StreamExt, stream};
use prost::Message;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

const MAX_REQUEST_BYTES: usize = 512 * 1024 * 1024;
pub const COORDINATE_PATH: &str = "/v1/coordinate";
pub const HEALTH_PATH: &str = "/health";

pub struct GenerationCoordinatorService {
    coordinator: Arc<dyn GenerationCoordinatorClient>,
    strategy: DisaggregationStrategy,
}

impl GenerationCoordinatorService {
    pub fn new(
        coordinator: Arc<dyn GenerationCoordinatorClient>,
        strategy: DisaggregationStrategy,
    ) -> Self {
        Self {
            coordinator,
            strategy,
        }
    }

    pub async fn start(
        self: Arc<Self>,
        address: SocketAddr,
    ) -> Result<RunningGenerationCoordinatorService> {
        let listener = tokio::net::TcpListener::bind(address)
            .await
            .with_context(|| {
                format!("failed to bind generation coordinator service to {address}")
            })?;
        let local_addr = listener.local_addr()?;
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let app = Router::new()
            .route(COORDINATE_PATH, post(coordinate))
            .route(HEALTH_PATH, get(health))
            .layer(DefaultBodyLimit::max(MAX_REQUEST_BYTES))
            .with_state(self);
        let task = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = shutdown_rx.await;
                })
                .await
                .context("generation coordinator HTTP service failed")
        });
        Ok(RunningGenerationCoordinatorService {
            local_addr,
            shutdown_tx: Some(shutdown_tx),
            task: Some(task),
        })
    }

    async fn generate(&self, mut request: NewRequestV1) -> Result<GenerationOutcome> {
        protocol::validate_request(&request)?;
        let request_id = request.request_id.clone();
        let trace_context = request.trace_context.take();
        let metadata = std::mem::take(&mut request.metadata);
        let (generation_request, enable_potential_loads_next_check) =
            decode_new_request(request_id.clone(), request, self.strategy)?;
        let inner: Arc<dyn AsyncEngineContext> = Arc::new(Controller::new(request_id));
        let trace_context = trace_context.map(trace_context_from_wire);
        self.coordinator
            .generate(
                RequestContext::new(inner, trace_context, metadata),
                generation_request,
                GenerationOptions {
                    enable_potential_loads_next_check,
                    ..Default::default()
                },
            )
            .await
    }
}

pub struct RunningGenerationCoordinatorService {
    local_addr: SocketAddr,
    shutdown_tx: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<Result<()>>>,
}

impl RunningGenerationCoordinatorService {
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    pub fn endpoint_url(&self) -> String {
        format!("http://{}{COORDINATE_PATH}", self.local_addr)
    }

    pub async fn shutdown(mut self) -> Result<()> {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
        self.task
            .take()
            .expect("running service owns its task")
            .await
            .context("service task failed")?
    }
}

impl Drop for RunningGenerationCoordinatorService {
    fn drop(&mut self) {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
    }
}

async fn health() -> StatusCode {
    StatusCode::OK
}

async fn coordinate(
    State(service): State<Arc<GenerationCoordinatorService>>,
    body: Bytes,
) -> Response<Body> {
    let request = match NewRequestV1::decode(body) {
        Ok(request) => request,
        Err(error) => return text_response(StatusCode::BAD_REQUEST, error.to_string()),
    };
    match service.generate(request).await {
        Ok(GenerationOutcome::Denied(denied)) => protobuf_response(Body::from(
            protocol::encode_response_frame(&denial_frame(denied)),
        )),
        Ok(GenerationOutcome::Connected(generated)) => {
            let admission = GenerationResponseFrameV1 {
                frame: Some(generation_response_frame_v1::Frame::Admission(
                    generated.admission.into(),
                )),
            };
            let first = stream::once(async move {
                Ok::<_, Infallible>(Bytes::from(protocol::encode_response_frame(&admission)))
            });
            let chunks = generated.stream.map(|chunk| {
                let frame = match rmp_serde::to_vec_named(&chunk) {
                    Ok(chunk) => GenerationResponseFrameV1 {
                        frame: Some(generation_response_frame_v1::Frame::ChunkMsgpack(chunk)),
                    },
                    Err(error) => GenerationResponseFrameV1 {
                        frame: Some(generation_response_frame_v1::Frame::Error(
                            ProtocolErrorV1 {
                                message: format!("failed to encode generation chunk: {error}"),
                            },
                        )),
                    },
                };
                Ok::<_, Infallible>(Bytes::from(protocol::encode_response_frame(&frame)))
            });
            protobuf_response(Body::from_stream(first.chain(chunks)))
        }
        Err(error) => text_response(StatusCode::BAD_REQUEST, error.to_string()),
    }
}

fn protobuf_response(body: Body) -> Response<Body> {
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, protocol::RESPONSE_CONTENT_TYPE)
        .body(body)
        .expect("static response is valid")
}

fn text_response(status: StatusCode, message: String) -> Response<Body> {
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
        .body(Body::from(message))
        .expect("static response is valid")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::codec::map_value;
    use crate::{GeneratedRequest, GenerationAdmission, RemoteGenerationCoordinator};
    use crate::{GenerationRequest, RouterRequestNew};
    use dynamo_runtime::logging::DistributedTraceContext;
    use dynamo_runtime::pipeline::{EngineStream, ResponseStream};
    use dynamo_runtime::protocols::annotated::Annotated;
    use futures::future::BoxFuture;
    use rmpv::Value;
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    struct Capture {
        request_id: String,
        trace_id: String,
        metadata: BTreeMap<String, String>,
        request: GenerationRequest,
    }

    #[derive(Default)]
    struct CapturingCoordinator {
        capture: Mutex<Option<Capture>>,
    }

    impl GenerationCoordinatorClient for CapturingCoordinator {
        fn generate(
            &self,
            context: RequestContext,
            request: GenerationRequest,
            _options: GenerationOptions,
        ) -> BoxFuture<'_, Result<GenerationOutcome>> {
            Box::pin(async move {
                *self.capture.lock().unwrap() = Some(Capture {
                    request_id: context.id().to_string(),
                    trace_id: context.trace_context().unwrap().trace_id.clone(),
                    metadata: context.metadata_snapshot(),
                    request,
                });
                let response = Value::Map(vec![(Value::from("token"), Value::from(42))]);
                let output = stream::iter([Annotated::from_data(response)]);
                let stream: EngineStream<Annotated<Value>> =
                    ResponseStream::new(Box::pin(output), context.inner());
                Ok(GenerationOutcome::Connected(GeneratedRequest {
                    stream,
                    admission: GenerationAdmission {
                        estimated_overlap_tokens: 32,
                        best_overlap_blocks: 2,
                        prefill_worker_id: 7,
                        prefill_dp_rank: 1,
                        decode_worker_id: None,
                        decode_dp_rank: None,
                    },
                }))
            })
        }
    }

    #[tokio::test]
    async fn rust_service_round_trips_remote_client_without_python_request_handling() {
        let coordinator = Arc::new(CapturingCoordinator::default());
        let service = Arc::new(GenerationCoordinatorService::new(
            coordinator.clone(),
            DisaggregationStrategy::Aggregated,
        ));
        let running = service.start("127.0.0.1:0".parse().unwrap()).await.unwrap();
        let base_url = format!("http://{}", running.local_addr());
        assert_eq!(
            reqwest::get(format!("{base_url}{HEALTH_PATH}"))
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );

        let remote =
            RemoteGenerationCoordinator::new(format!("{base_url}{COORDINATE_PATH}")).unwrap();
        let inner: Arc<dyn AsyncEngineContext> =
            Arc::new(Controller::new("request-service-1".to_string()));
        let trace: DistributedTraceContext = serde_json::from_value(serde_json::json!({
            "trace_id": "0123456789abcdef0123456789abcdef",
            "span_id": "0123456789abcdef"
        }))
        .unwrap();
        let context = RequestContext::new(
            inner,
            Some(trace),
            BTreeMap::from([("tenant".to_string(), "acme".to_string())]),
        );
        let worker_request = serde_json::from_value(serde_json::json!({
            "model": "fast-model",
            "sampling_params": {"temperature": 0.2},
            "lora": "adapter-a",
            "user": "session-a"
        }))
        .unwrap();
        let outcome = remote
            .generate(
                context,
                GenerationRequest {
                    routing_request: RouterRequestNew {
                        tokens: vec![1, 2, 3],
                        do_not_queue: true,
                        ..Default::default()
                    },
                    primary_worker_request: worker_request,
                    decode_worker_request: None,
                },
                GenerationOptions::default(),
            )
            .await
            .unwrap();
        let GenerationOutcome::Connected(mut generated) = outcome else {
            panic!("expected connected generation")
        };
        assert_eq!(generated.admission.prefill_worker_id, 7);
        assert_eq!(
            map_value(
                generated
                    .stream
                    .next()
                    .await
                    .unwrap()
                    .data
                    .as_ref()
                    .unwrap(),
                "token"
            )
            .and_then(Value::as_i64),
            Some(42)
        );
        assert!(generated.stream.next().await.is_none());

        let capture = coordinator.capture.lock().unwrap().take().unwrap();
        assert_eq!(capture.request_id, "request-service-1");
        assert_eq!(capture.trace_id, "0123456789abcdef0123456789abcdef");
        assert_eq!(
            capture.metadata.get("tenant").map(String::as_str),
            Some("acme")
        );
        assert_eq!(capture.request.routing_request.tokens, vec![1, 2, 3]);
        assert!(capture.request.routing_request.do_not_queue);
        assert_eq!(
            map_value(&capture.request.primary_worker_request, "streaming")
                .and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(
            map_value(&capture.request.primary_worker_request, "user").and_then(Value::as_str),
            Some("session-a")
        );
        assert_eq!(
            map_value(&capture.request.primary_worker_request, "model").and_then(Value::as_str),
            Some("fast-model")
        );
        assert_eq!(
            map_value(&capture.request.primary_worker_request, "lora").and_then(Value::as_str),
            Some("adapter-a")
        );
        running.shutdown().await.unwrap();
    }
}
