// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::protocol::codec::encode_request;
use crate::protocol::{self, GenerationResponseFrameV1, generation_response_frame_v1};
use crate::{
    CancellationPolicy, DeniedGenerationRequest, DeniedRequest, GeneratedRequest,
    GenerationAdmission, GenerationCoordinatorClient, GenerationOptions, GenerationOutcome,
    GenerationRequest, RequestContext, RouteOptions,
};
use anyhow::{Context, Result, bail};
use dynamo_runtime::pipeline::{EngineStream, ResponseStream};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::future::BoxFuture;
use futures::{Stream, StreamExt};
use prost::Message;
use reqwest::{Client, Url};
use rmpv::Value;
use std::pin::Pin;
use std::sync::Arc;
use tokio_util::codec::{FramedRead, LengthDelimitedCodec};
use tokio_util::io::StreamReader;

type FramedResponse = Pin<Box<dyn Stream<Item = Result<GenerationResponseFrameV1>> + Send>>;

/// Generation coordinator client that POSTs protobuf to one exact HTTP URL.
pub struct RemoteGenerationCoordinator {
    endpoint: Url,
    client: Client,
}

impl RemoteGenerationCoordinator {
    pub fn new(endpoint: impl AsRef<str>) -> Result<Self> {
        let endpoint = Url::parse(endpoint.as_ref()).context("invalid generation service URL")?;
        if !matches!(endpoint.scheme(), "http" | "https") {
            bail!("generation service URL must use http or https");
        }
        Ok(Self {
            endpoint,
            client: Client::new(),
        })
    }

    async fn generate_remote(
        &self,
        context: RequestContext,
        request: GenerationRequest,
        options: GenerationOptions,
    ) -> Result<GenerationOutcome> {
        validate_remote_options(&options)?;
        let wire_request = encode_request(&context, request, &options)?;
        let request_context = context.inner();
        let send = self
            .client
            .post(self.endpoint.clone())
            .header(
                reqwest::header::CONTENT_TYPE,
                protocol::REQUEST_CONTENT_TYPE,
            )
            .header(reqwest::header::ACCEPT, protocol::RESPONSE_CONTENT_TYPE)
            .body(wire_request.encode_to_vec())
            .send();
        let response = tokio::select! {
            response = send => response.context("generation service request failed")?,
            _ = request_context.stopped() => return Ok(cancelled_outcome()),
            _ = request_context.killed() => return Ok(cancelled_outcome()),
        };

        if !response.status().is_success() {
            let status = response.status();
            let detail = response
                .text()
                .await
                .unwrap_or_else(|_| "response body unavailable".to_string());
            bail!("generation service returned HTTP {status}: {detail}");
        }

        let mut reader = response_frames(response);
        let first = tokio::select! {
            frame = reader.next() => frame.transpose()?,
            _ = request_context.stopped() => return Ok(cancelled_outcome()),
            _ = request_context.killed() => return Ok(cancelled_outcome()),
        }
        .context("generation service ended before an admission or denial")?;
        match first.frame {
            Some(generation_response_frame_v1::Frame::Admission(admission)) => {
                connected_outcome(context, admission.into(), reader)
            }
            Some(generation_response_frame_v1::Frame::Denied(denied)) => {
                Ok(GenerationOutcome::Denied(denied.try_into()?))
            }
            Some(generation_response_frame_v1::Frame::Error(error)) => {
                bail!("generation service protocol error: {}", error.message)
            }
            Some(generation_response_frame_v1::Frame::ChunkMsgpack(_)) => {
                bail!("generation service sent a chunk before admission")
            }
            None => bail!("generation service sent an empty response frame"),
        }
    }
}

fn cancelled_outcome() -> GenerationOutcome {
    GenerationOutcome::Denied(DeniedGenerationRequest {
        denied: DeniedRequest::Cancelled(),
        admission: None,
    })
}

impl GenerationCoordinatorClient for RemoteGenerationCoordinator {
    fn generate(
        &self,
        context: RequestContext,
        request: GenerationRequest,
        options: GenerationOptions,
    ) -> BoxFuture<'_, Result<GenerationOutcome>> {
        Box::pin(self.generate_remote(context, request, options))
    }
}

fn validate_remote_options(options: &GenerationOptions) -> Result<()> {
    for (name, route) in [("primary", &options.primary), ("decode", &options.decode)] {
        if !route.require_available.is_empty()
            || route.potential_loads_check.is_some()
            || route.cancellation != CancellationPolicy::Cancellable
            || route.max_reroutes != RouteOptions::default().max_reroutes
            || route.tracing_enabled
            || route.wait_for_first_response
            || route.phase.is_some()
        {
            bail!(
                "remote generation does not accept service-local {name} RouteOptions; configure them on the generation service"
            );
        }
    }
    Ok(())
}

fn connected_outcome(
    context: RequestContext,
    admission: GenerationAdmission,
    mut reader: FramedResponse,
) -> Result<GenerationOutcome> {
    let engine_context = context.inner();
    let cancellation = Arc::clone(&engine_context);
    let output = async_stream::stream! {
        loop {
            let frame = tokio::select! {
                frame = reader.next() => frame,
                _ = cancellation.stopped() => break,
                _ = cancellation.killed() => break,
            };
            match frame {
                Some(Ok(frame)) => match frame.frame {
                    Some(generation_response_frame_v1::Frame::ChunkMsgpack(chunk)) => {
                        match rmp_serde::from_slice::<Annotated<Value>>(&chunk) {
                            Ok(chunk) => yield chunk,
                            Err(error) => {
                                yield Annotated::from_error(format!(
                                    "invalid generation service chunk: {error}"
                                ));
                                break;
                            }
                        }
                    }
                    Some(generation_response_frame_v1::Frame::Error(error)) => {
                        yield Annotated::from_error(error.message);
                        break;
                    }
                    Some(generation_response_frame_v1::Frame::Admission(_)) => {
                        yield Annotated::from_error(
                            "generation service sent more than one admission",
                        );
                        break;
                    }
                    Some(generation_response_frame_v1::Frame::Denied(_)) => {
                        yield Annotated::from_error(
                            "generation service denied an already-admitted stream",
                        );
                        break;
                    }
                    None => {
                        yield Annotated::from_error(
                            "generation service sent an empty response frame",
                        );
                        break;
                    }
                },
                None => break,
                Some(Err(error)) => {
                    yield Annotated::from_error(format!(
                        "generation service response failed: {error}"
                    ));
                    break;
                }
            }
        }
    };
    let stream: EngineStream<Annotated<Value>> =
        ResponseStream::new(Box::pin(output), engine_context);
    Ok(GenerationOutcome::Connected(GeneratedRequest {
        stream,
        admission,
    }))
}

fn response_frames(response: reqwest::Response) -> FramedResponse {
    let body = response
        .bytes_stream()
        .map(|chunk| chunk.map_err(std::io::Error::other));
    let codec = LengthDelimitedCodec::builder()
        .max_frame_length(protocol::MAX_FRAME_BYTES)
        .new_codec();
    FramedRead::new(StreamReader::new(body), codec)
        .map(|frame| {
            GenerationResponseFrameV1::decode(frame?)
                .context("invalid generation service protobuf frame")
        })
        .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::codec::map_value;
    use crate::protocol::{AdmissionV1, NewRequestV1};
    use axum::Router;
    use axum::body::{Body, Bytes};
    use axum::extract::State;
    use axum::http::{Response, StatusCode, header};
    use axum::routing::post;
    use dynamo_kv_router::protocols::{BlockExtraInfo, BlockMmObjectInfo, RoutingConstraints};
    use dynamo_runtime::logging::DistributedTraceContext;
    use dynamo_runtime::pipeline::AsyncEngineContext;
    use dynamo_runtime::pipeline::context::Controller;
    use futures::StreamExt;
    use std::collections::{BTreeMap, HashSet};
    use std::convert::Infallible;
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct TestServerState {
        captured: Arc<Mutex<Option<NewRequestV1>>>,
        response: Vec<u8>,
    }

    async fn coordinate(State(state): State<TestServerState>, body: Bytes) -> Response<Body> {
        let request = NewRequestV1::decode(body).expect("valid request protobuf");
        protocol::validate_request(&request).expect("semantically valid request");
        *state.captured.lock().unwrap() = Some(request);

        // Deliberately split both the frame header and payload. This exercises
        // the real streaming decoder rather than relying on HTTP chunk boundaries.
        let response = state.response;
        let split_at = [2, 11, response.len()];
        let mut start = 0;
        let chunks = split_at.into_iter().map(move |end| {
            let chunk = Bytes::copy_from_slice(&response[start..end]);
            start = end;
            Ok::<_, Infallible>(chunk)
        });
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, protocol::RESPONSE_CONTENT_TYPE)
            .body(Body::from_stream(futures::stream::iter(chunks)))
            .unwrap()
    }

    fn value(json: serde_json::Value) -> Value {
        serde_json::from_value(json).unwrap()
    }

    #[tokio::test]
    async fn remote_client_sends_typed_request_and_streams_framed_response() {
        let first_chunk = Annotated::from_data(value(serde_json::json!({"token": 17})));
        let second_chunk = Annotated::from_data(value(serde_json::json!({"finish": true})));
        let frames = [
            GenerationResponseFrameV1 {
                frame: Some(generation_response_frame_v1::Frame::Admission(
                    AdmissionV1 {
                        estimated_overlap_tokens: 64,
                        best_overlap_blocks: 3,
                        prefill_worker_id: 10,
                        prefill_dp_rank: 1,
                        decode_worker_id: Some(20),
                        decode_dp_rank: Some(2),
                    },
                )),
            },
            GenerationResponseFrameV1 {
                frame: Some(generation_response_frame_v1::Frame::ChunkMsgpack(
                    rmp_serde::to_vec_named(&first_chunk).unwrap(),
                )),
            },
            GenerationResponseFrameV1 {
                frame: Some(generation_response_frame_v1::Frame::ChunkMsgpack(
                    rmp_serde::to_vec_named(&second_chunk).unwrap(),
                )),
            },
        ];
        let response = frames
            .iter()
            .flat_map(protocol::encode_response_frame)
            .collect::<Vec<_>>();
        let captured = Arc::new(Mutex::new(None));
        let app = Router::new()
            .route("/v1/coordinate", post(coordinate))
            .with_state(TestServerState {
                captured: Arc::clone(&captured),
                response,
            });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let coordinator =
            RemoteGenerationCoordinator::new(format!("http://{address}/v1/coordinate")).unwrap();
        let inner: Arc<dyn AsyncEngineContext> =
            Arc::new(Controller::new("request-123".to_string()));
        let trace_context: DistributedTraceContext = serde_json::from_value(serde_json::json!({
            "trace_id": "0123456789abcdef0123456789abcdef",
            "span_id": "0123456789abcdef",
            "parent_id": "fedcba9876543210",
            "tracestate": "vendor=value",
            "x_request_id": "external-123",
            "request_id": "trace-request-123"
        }))
        .unwrap();
        let context = RequestContext::new(
            inner,
            Some(trace_context),
            BTreeMap::from([("tenant".to_string(), "acme".to_string())]),
        );
        let worker_request = value(serde_json::json!({
            "model": "test-model",
            "streaming": true,
            "sampling_params": {"temperature": 0.25, "max_tokens": 8},
            "dynamic_temperature_rules": [{"after": 4, "temperature": 0.1}],
            "lora": "adapter-a",
            "cache_salt": "salt-a",
            "priority": -4,
            "service_tier": "priority",
            "user": "session-a",
            "mm_args": {
                "mm_kwargs": ["image-kwargs"],
                "media_token_id": 32000,
                "mm_hashes": ["image-hash"],
                "mm_positions": [2],
                "mm_lengths": [4],
                "mm_native_inputs": [{"type": "image_url", "url": "https://example.invalid/image"}]
            }
        }));
        let routing_request = crate::RouterRequestNew {
            tokens: vec![1, 2, 3, 4],
            block_mm_infos: Some(vec![
                Some(BlockExtraInfo {
                    mm_objects: vec![BlockMmObjectInfo {
                        mm_hash: 99,
                        offsets: vec![(2, 4)],
                    }],
                }),
                None,
            ]),
            routing_constraints: RoutingConstraints {
                required_taints: HashSet::from(["gpu".to_string()]),
                preferred_taints: [("zone-a".to_string(), 0.75)].into_iter().collect(),
            },
            allowed_worker_ids: Some(HashSet::from([7, 5])),
            priority_jump: 1.5,
            priority_load_shed_percent: 25,
            do_not_queue: true,
        };
        let outcome = coordinator
            .generate(
                context,
                GenerationRequest {
                    routing_request,
                    primary_worker_request: worker_request,
                    decode_worker_request: None,
                },
                GenerationOptions {
                    enable_potential_loads_next_check: true,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let GenerationOutcome::Connected(mut generated) = outcome else {
            panic!("expected an admitted request")
        };
        assert_eq!(generated.admission.prefill_worker_id, 10);
        assert_eq!(generated.admission.decode_worker_id, Some(20));
        assert_eq!(
            generated.stream.next().await.unwrap().data,
            first_chunk.data
        );
        assert_eq!(
            generated.stream.next().await.unwrap().data,
            second_chunk.data
        );
        assert!(generated.stream.next().await.is_none());

        let captured = captured.lock().unwrap().take().unwrap();
        assert_eq!(captured.request_id, "request-123");
        assert_eq!(
            captured
                .trace_context
                .as_ref()
                .unwrap()
                .x_request_id
                .as_deref(),
            Some("external-123")
        );
        assert_eq!(
            captured.metadata.get("tenant").map(String::as_str),
            Some("acme")
        );
        let request = captured;
        assert_eq!(request.model, "test-model");
        assert_eq!(request.tokens, vec![1, 2, 3, 4]);
        assert_eq!(request.lora.as_deref(), Some("adapter-a"));
        assert!(request.enable_potential_loads_next_check);
        let routing = request.routing.unwrap();
        assert_eq!(routing.session_id.as_deref(), Some("session-a"));
        assert!(routing.do_not_queue);
        assert_eq!(routing.allowed_worker_ids, vec![5, 7]);
        assert_eq!(request.mm_routing_args.unwrap().blocks.len(), 2);
        assert_eq!(request.mm_payloads.unwrap().hashes, vec!["image-hash"]);
        let sampling: Value = rmp_serde::from_slice(&request.sampling_msgpack).unwrap();
        assert_eq!(
            map_value(&sampling, "sampling_params")
                .and_then(|params| map_value(params, "max_tokens"))
                .and_then(Value::as_u64),
            Some(8)
        );

        server.abort();
    }
}
