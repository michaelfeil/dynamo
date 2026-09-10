use super::super::{anthropic, openai, service_v2::HttpService};
use super::*;
use crate::{model_card::ModelDeploymentCard, request_template::RequestTemplate};
use dynamo_runtime::pipeline::async_trait;
use serde_json::json;
use std::sync::Mutex;

#[tokio::test]
async fn listener_respects_configured_host() {
    for host in ["127.0.0.1", "127.0.0.2", "0.0.0.0"] {
        let listener = bind_listener(host, 0).await.unwrap();
        assert_eq!(listener.local_addr().unwrap().ip().to_string(), host);
    }
}

#[derive(Default)]
struct Engine {
    requests: Mutex<Vec<Value>>,
}

fn chunk() -> Value {
    json!({"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1,
        "model": "test", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hello"},
        "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}})
}

fn artifact_value(stage: &str) -> Value {
    match stage {
        "render" => json!("hi"),
        "tokenize" => json!([1, 2, 3]),
        "engine_request" => {
            json!({"request":{"sampling_params":{}},"routing_constraints":null,"routing_priority":null,"do_not_queue":false})
        }
        "engine_output" => json!([{"outputs":[{"token_ids_diff":[42]}]}]),
        "chat_stream" => json!([chunk()]),
        _ => panic!("unsupported test artifact {stage}"),
    }
}

#[test]
fn typed_chat_chunks_preserve_known_metadata_and_reject_unknown_fields() {
    let mut input = chunk();
    input["kv_cache_metrics"] = json!({"num_reused_blocks": 3});
    let typed: ChatChunk = serde_json::from_value(input.clone()).unwrap();
    assert_eq!(serde_json::to_value(typed).unwrap(), input);
    input["unexpected_extension"] = json!(true);
    assert!(serde_json::from_value::<ChatChunk>(input).is_err());
}

#[tokio::test]
async fn ingress_and_client_json_are_not_reconstructed() {
    let expected = r#"{ "unknown":1e2, "unknown":2, "nested": { "z":1,"a":2 } }"#;
    let production = Router::new().route(
        path("chat"),
        post(move |body: String| async move {
            assert_eq!(body, expected);
            ([("content-type", "application/json")], expected)
        }),
    );
    let response = router(production, HookSlot::default())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/waypoints")
                .header("content-type", "application/json")
                .body(Body::from(format!(
                    r#"{{"protocol":"chat","request":{expected},"stop_after":"client"}}"#
                )))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body = axum::body::to_bytes(response.into_body(), MAX_BYTES)
        .await
        .unwrap();
    assert!(std::str::from_utf8(&body).unwrap().contains(expected));
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        anyhow::Error,
    > for Engine
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateChatCompletionRequest>,
    ) -> anyhow::Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>> {
        self.requests
            .lock()
            .unwrap()
            .push(serde_json::to_value(request.content())?);
        Ok(ResponseStream::new(
            Box::pin(futures::stream::iter([Annotated::from_data(
                serde_json::from_value(chunk())?,
            )])),
            request.context(),
        ))
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<Value>, ManyOut<Annotated<Value>>, anyhow::Error> for Engine {
    async fn generate(
        &self,
        request: SingleIn<Value>,
    ) -> anyhow::Result<ManyOut<Annotated<Value>>> {
        self.requests
            .lock()
            .unwrap()
            .push(request.content().clone());
        let stop_after = request.content()["stop_after"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        let mut items = Vec::new();
        if request.content()["preserve_intermediates"] == json!(true) {
            for stage in [
                "render",
                "tokenize",
                "engine_request",
                "engine_output",
                "chat_stream",
            ] {
                if stage == stop_after {
                    break;
                }
                items.push(Annotated::from_data(
                    json!({"stage": stage, "value": artifact_value(stage)}),
                ));
            }
        }
        items.push(Annotated::from_data(if stop_after == "client" {
            chunk()
        } else {
            json!({"stage": stop_after, "value": artifact_value(&stop_after)})
        }));
        Ok(ResponseStream::new(
            Box::pin(futures::stream::iter(items)),
            request.context(),
        ))
    }
}

fn setup() -> (Router, Router, Arc<Engine>, Arc<Engine>) {
    let (production, inspection, engine, hook, _) = setup_with_metrics();
    (production, inspection, engine, hook)
}

fn setup_with_metrics() -> (
    Router,
    Router,
    Arc<Engine>,
    Arc<Engine>,
    Arc<super::super::metrics::Metrics>,
) {
    let service = HttpService::builder()
        .enable_chat_endpoints(true)
        .enable_responses_endpoints(true)
        .enable_anthropic_endpoints(true)
        .build()
        .unwrap();
    let state = service.state_clone();
    let metrics = state.metrics_clone();
    let engine = Arc::new(Engine::default());
    let card = ModelDeploymentCard::with_name_only("test");
    state
        .manager()
        .add_chat_completions_model("test", card.mdcsum(), engine.clone())
        .unwrap();
    let template = Some(RequestTemplate {
        model: "test".into(),
        temperature: 0.25,
        max_completion_tokens: 128,
    });
    let production = openai::chat_completions_router(state.clone(), template.clone(), None)
        .1
        .merge(openai::responses_router(state.clone(), template.clone(), None).1)
        .merge(anthropic::anthropic_messages_router(state, template, None).1);
    let hook = Arc::new(Engine::default());
    let slot = HookSlot::default();
    assert!(slot.set(hook.clone()).is_ok());
    (
        production.clone(),
        router(production, slot),
        engine,
        hook,
        metrics,
    )
}

#[tokio::test]
async fn early_boundaries_are_successful_and_preserve_hook_contract() {
    use super::super::metrics::{Endpoint, ErrorType, RequestType, Status};
    for (protocol, endpoint) in [
        ("chat", Endpoint::ChatCompletions),
        ("responses", Endpoint::Responses),
        ("messages", Endpoint::AnthropicMessages),
    ] {
        for stage in [
            "canonical",
            "render",
            "tokenize",
            "engine_request",
            "engine_output",
            "chat_stream",
        ] {
            let (_, inspection, engine, hook, metrics) = setup_with_metrics();
            let registry = prometheus::Registry::new();
            metrics.register(&registry).unwrap();
            let output = json!([{"outputs": [{"token_ids_diff": [42], "finish_reason": "stop"}]}]);
            let (status, body) = send(
                inspection,
                "/v1/waypoints",
                json!({
                    "protocol": protocol, "request": input(protocol, false),
                    "stop_after": stage, "engine_output": output,
                }),
            )
            .await;
            assert_eq!(status, 200, "{protocol}/{stage}: {body}");
            let artifact: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(artifact["stage"], stage);
            assert!(engine.requests.lock().unwrap().is_empty());
            let calls = hook.requests.lock().unwrap();
            if stage == "canonical" {
                assert!(calls.is_empty());
            } else {
                assert_eq!(artifact["value"], artifact_value(stage));
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].as_object().unwrap().len(), 5);
                assert_eq!(calls[0]["stop_after"], stage);
                assert_eq!(calls[0]["engine_output"], output);
                assert_eq!(calls[0]["max_bytes"], MAX_BYTES);
                assert_eq!(calls[0]["request"]["model"], "test");
                assert_eq!(calls[0]["request"]["messages"][0]["content"], "hi");
            }
            assert_eq!(
                metrics.get_request_counter(
                    "test",
                    &endpoint,
                    &RequestType::Unary,
                    &Status::Success,
                    &ErrorType::None
                ),
                1,
                "{protocol}/{stage}"
            );
            assert_eq!(
                metrics.get_request_counter(
                    "test",
                    &endpoint,
                    &RequestType::Unary,
                    &Status::Error,
                    &ErrorType::Internal
                ),
                0,
                "{protocol}/{stage}"
            );
            for family in registry.gather() {
                for metric in family.get_metric() {
                    if family.get_field_type() == prometheus::proto::MetricType::GAUGE {
                        assert_eq!(
                            metric.get_gauge().value(),
                            0.0,
                            "{} leaked after {protocol}/{stage}",
                            family.name()
                        );
                    }
                }
            }
        }
    }
}

struct ResultHook(Vec<Value>);

struct FailingHook(u16);

#[async_trait]
impl AsyncEngine<SingleIn<Value>, ManyOut<Annotated<Value>>, anyhow::Error> for FailingHook {
    async fn generate(
        &self,
        request: SingleIn<Value>,
    ) -> anyhow::Result<ManyOut<Annotated<Value>>> {
        Ok(ResponseStream::new(
            Box::pin(futures::stream::iter([
                Annotated::from_data(json!({"stage":"render","value":"prompt before error"})),
                Annotated::from_data(json!({"stage":"tokenize","value":[1,2,3]})),
                Annotated::from_error(
                    json!({"code":self.0,"message":"processor failed"}).to_string(),
                ),
            ])),
            request.context(),
        ))
    }
}

#[tokio::test]
async fn processor_errors_keep_completed_boundaries_for_all_protocols() {
    for protocol in ["chat", "responses", "messages"] {
        for code in [400, 500] {
            let (production, _, engine, _) = setup();
            let hook = HookSlot::default();
            assert!(hook.set(Arc::new(FailingHook(code))).is_ok());
            let (status, body) = send(
                router(production, hook),
                "/v1/waypoints",
                json!({
                    "protocol":protocol,"request":input(protocol,false),
                    "stop_after":"client","preserve_intermediates":true
                }),
            )
            .await;
            let result = assert_failure(status, &body, code, "client");
            assert!(
                result["value"]["body"]
                    .to_string()
                    .contains("processor failed")
            );
            assert_eq!(
                result["intermediates"]["canonical"]["request"]["model"],
                "test"
            );
            assert_eq!(result["intermediates"]["render"], "prompt before error");
            assert_eq!(result["intermediates"]["tokenize"], json!([1, 2, 3]));
            assert_eq!(result["intermediates"].as_object().unwrap().len(), 3);
            assert!(engine.requests.lock().unwrap().is_empty());
        }
    }
}

#[tokio::test]
async fn handler_errors_keep_the_original_body_without_claiming_a_completed_stage() {
    for protocol in ["chat", "responses", "messages"] {
        let (production, inspection, engine, hook) = setup();
        let raw = json!(true);
        let (original_status, original_body) = send(production, path(protocol), raw.clone()).await;
        assert!(original_status >= 400);
        let (status, body) = send(inspection, "/v1/waypoints", json!({
            "protocol":protocol,"request":raw,"stop_after":"canonical","preserve_intermediates":true
        })).await;
        let result = assert_failure(status, &body, original_status, "canonical");
        assert_eq!(
            result["value"]["body"],
            serde_json::from_str::<Value>(&original_body).unwrap_or_else(|_| json!(original_body))
        );
        assert_eq!(result["intermediates"], json!({}));
        assert!(engine.requests.lock().unwrap().is_empty());
        assert!(hook.requests.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn preserved_fused_render_is_null_but_later_or_duplicate_hook_stages_fail() {
    for (items, expected_status) in [
        (
            vec![
                json!({"stage":"render","value":null}),
                json!({"stage":"tokenize","value":[1]}),
            ],
            None,
        ),
        (
            vec![
                json!({"stage":"engine_output","value":[]}),
                json!({"stage":"tokenize","value":[1]}),
            ],
            Some(502),
        ),
        (
            vec![
                json!({"stage":"render","value":"one"}),
                json!({"stage":"render","value":"two"}),
            ],
            Some(502),
        ),
    ] {
        let (production, _, engine, _) = setup();
        let hook = HookSlot::default();
        assert!(hook.set(Arc::new(ResultHook(items))).is_ok());
        let (status, body) = send(
            router(production, hook),
            "/v1/waypoints",
            json!({
                "protocol":"chat","request":input("chat",false),
                "stop_after":"tokenize","preserve_intermediates":true
            }),
        )
        .await;
        if let Some(code) = expected_status {
            assert_failure(status, &body, code, "tokenize");
        } else {
            let result: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(status, 200);
            assert_eq!(result["stage"], "tokenize");
            assert_eq!(result["intermediates"]["render"], Value::Null);
        }
        assert!(engine.requests.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn error_capture_is_bounded_and_reports_truncation() {
    let intermediates = BTreeMap::from([
        (Stage::Canonical, json!({"request":{}})),
        (Stage::Render, json!("x".repeat(MAX_BYTES))),
    ]);
    let response = failure_with_intermediates(
        diagnostic_error(400, "bad input"),
        Stage::Client,
        Some(intermediates),
    );
    let status = response.status().as_u16();
    let bytes = axum::body::to_bytes(response.into_body(), MAX_BYTES)
        .await
        .unwrap();
    let body = String::from_utf8(bytes.to_vec()).unwrap();
    let result = assert_failure(status, &body, 400, "client");
    assert_eq!(result["value"]["capture_truncated"], true);
    assert_eq!(result["intermediates"], json!({"canonical":{"request":{}}}));
}

#[tokio::test]
async fn serialized_envelope_limit_and_body_transport_errors_are_distinct() {
    let escaped = Router::new().route(
        path("chat"),
        post(|| async { "\"".repeat(MAX_BYTES / 2 + 1) }),
    );
    let (status, body) = send(
        router(escaped, HookSlot::default()),
        "/v1/waypoints",
        json!({
            "protocol":"chat", "request":{}, "stop_after":"client"
        }),
    )
    .await;
    assert!(body.len() <= MAX_BYTES);
    assert_failure(status, &body, 413, "client");

    let broken = Router::new().route(
        path("chat"),
        post(|| async {
            Body::from_stream(futures::stream::iter([Err::<String, _>(
                std::io::Error::other("read failed"),
            )]))
        }),
    );
    let (status, body) = send(
        router(broken, HookSlot::default()),
        "/v1/waypoints",
        json!({
            "protocol":"chat", "request":{}, "stop_after":"client"
        }),
    )
    .await;
    assert_failure(status, &body, 502, "client");
}

#[async_trait]
impl AsyncEngine<SingleIn<Value>, ManyOut<Annotated<Value>>, anyhow::Error> for ResultHook {
    async fn generate(
        &self,
        request: SingleIn<Value>,
    ) -> anyhow::Result<ManyOut<Annotated<Value>>> {
        Ok(ResponseStream::new(
            Box::pin(futures::stream::iter(
                self.0.clone().into_iter().map(Annotated::from_data),
            )),
            request.context(),
        ))
    }
}

#[tokio::test]
async fn malformed_hook_results_do_not_fall_through_to_generation() {
    for (stage, results) in [
        (
            "tokenize",
            vec![json!({"stage": "render", "value": "wrong"})],
        ),
        ("tokenize", vec![chunk()]),
        ("tokenize", vec![]),
        ("tokenize", vec![json!({"stage":"tokenize"})]),
        (
            "tokenize",
            vec![json!({"stage":"tokenize","value":"not tokens"})],
        ),
        ("tokenize", vec![json!({"stage":"tokenize","value":[-1]})]),
        (
            "tokenize",
            vec![json!({"stage":"tokenize","value":[4294967296_u64]})],
        ),
        ("render", vec![json!({"stage":"render","value":[1]})]),
        (
            "engine_request",
            vec![json!({"stage":"engine_request","value":{"request":[]}})],
        ),
        (
            "engine_output",
            vec![json!({"stage":"engine_output","value":[1]})],
        ),
        (
            "chat_stream",
            vec![json!({"stage":"chat_stream","value":[{}]})],
        ),
        ("client", vec![json!({"choices": []})]),
        (
            "client",
            vec![chunk(), json!({"stage": "client", "value": "mixed"})],
        ),
    ] {
        let (production, _, engine, _) = setup();
        let slot = HookSlot::default();
        assert!(slot.set(Arc::new(ResultHook(results))).is_ok());
        let (status, body) = send(
            router(production, slot),
            "/v1/waypoints",
            json!({
                "protocol": "chat", "request": input("chat", false), "stop_after": stage,
            }),
        )
        .await;
        assert_failure(status, &body, 502, stage);
        assert!(body.contains("Waypoints hook:"), "{stage}: {body}");
        assert!(engine.requests.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn public_envelope_is_validated_before_calling_the_hook() {
    for spec in [
        json!({"protocol":"chat","request":{},"stop_after":"error"}),
        json!({"protocol":"unknown","request":{},"stop_after":"canonical"}),
        json!({"protocol":"chat","request":{},"stop_after":"unknown"}),
        json!({"protocol": "chat", "request": {}}),
        json!({"protocol": "chat", "stop_after": "client"}),
        json!({"protocol": "chat", "request": {}, "stop_after": "client", "max_bytes": 1}),
        json!({"protocol": "chat", "request": {}, "stop_after": "client", "engine_output": {}}),
        json!({"protocol": "chat", "request": {}, "stop_after": "client", "engine_output": [1]}),
    ] {
        let (_, inspection, engine, hook) = setup();
        assert_eq!(send(inspection, "/v1/waypoints", spec).await.0, 422);
        assert!(engine.requests.lock().unwrap().is_empty());
        assert!(hook.requests.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn replay_rebuilds_http_framing_but_preserves_application_headers() {
    let production = Router::new().route(
        path("chat"),
        post(
            |headers: axum::http::HeaderMap, Json(body): Json<Value>| async move {
                assert_eq!(headers.get("content-type").unwrap(), "application/json");
                assert_eq!(headers.get("x-baseten-request-id").unwrap(), "captured-id");
                for name in [
                    "host",
                    "content-length",
                    "content-encoding",
                    "connection",
                    "x-hop",
                    "transfer-encoding",
                    "upgrade",
                ] {
                    assert!(!headers.contains_key(name), "forwarded stale {name}");
                }
                Json(body)
            },
        ),
    );
    let raw = input("chat", false);
    let (status, body) = send(router(production, HookSlot::default()), "/v1/waypoints", json!({
        "protocol": "chat", "request": raw, "stop_after": "client",
        "headers": {"Host": "old-host", "Content-Length": "999", "Content-Type": "text/plain", "Content-Encoding": "gzip", "Connection": "X-Hop", "X-Hop": "secret", "Transfer-Encoding": "chunked", "Upgrade": "websocket", "x-baseten-request-id": "captured-id"},
    })).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["value"]["body"],
        raw
    );
}

fn input(protocol: &str, stream: bool) -> Value {
    match protocol {
        "responses" => {
            json!({"model": "test", "input": "hi", "stream": stream})
        }
        "messages" => {
            json!({"model": "test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 128, "stream": stream})
        }
        _ => {
            json!({"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": stream})
        }
    }
}

fn path(protocol: &str) -> &'static str {
    match protocol {
        "responses" => "/v1/responses",
        "messages" => "/v1/messages",
        _ => "/v1/chat/completions",
    }
}

async fn send(router: Router, path: &str, value: Value) -> (u16, String) {
    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .header("x-request-id", "same-id")
                .body(Body::from(value.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status().as_u16();
    let bytes = axum::body::to_bytes(response.into_body(), MAX_BYTES)
        .await
        .unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

#[tokio::test]
async fn canonical_matches_actual_engine_input_for_all_protocols() {
    for protocol in ["chat", "responses", "messages"] {
        let (production, inspection, engine, hook) = setup();
        let raw = input(protocol, false);
        let (status, body) = send(production, path(protocol), raw.clone()).await;
        assert_eq!(status, 200, "{protocol}: {body}");
        let (status, body) = send(
            inspection,
            "/v1/waypoints",
            json!({
                "protocol": protocol, "request": raw, "stop_after": "canonical",
            }),
        )
        .await;
        assert_eq!(status, 200, "{protocol}: {body}");
        let artifact: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(artifact["stage"], "canonical");
        assert_eq!(
            artifact["value"]["request"],
            engine.requests.lock().unwrap()[0],
            "{protocol}"
        );
        assert_eq!(
            engine.requests.lock().unwrap().len(),
            1,
            "inspection called live engine"
        );
        assert!(hook.requests.lock().unwrap().is_empty());
    }
}

// IDs/timestamps are intentionally freshly generated by production converters.
fn normalize(value: &mut Value) {
    match value {
        Value::Object(map) => {
            for key in ["id", "created", "created_at", "completed_at", "item_id"] {
                map.remove(key);
            }
            for value in map.values_mut() {
                normalize(value);
            }
        }
        Value::Array(values) => values.iter_mut().for_each(normalize),
        _ => {}
    }
}

fn normalized_body(body: &str, stream: bool) -> Value {
    if stream {
        Value::Array(
            body.lines()
                .filter_map(|line| {
                    line.strip_prefix("data: ").map(|data| {
                        let mut value = serde_json::from_str(data).unwrap_or_else(|_| json!(data));
                        normalize(&mut value);
                        value
                    })
                })
                .collect(),
        )
    } else {
        let mut value = serde_json::from_str(body).unwrap();
        normalize(&mut value);
        value
    }
}

#[tokio::test]
async fn client_uses_real_unary_and_stream_converters() {
    for protocol in ["chat", "responses", "messages"] {
        for stream in [false, true] {
            let (production, inspection, engine, hook) = setup();
            let raw = input(protocol, stream);
            let (status, expected) = send(production, path(protocol), raw.clone()).await;
            assert_eq!(status, 200, "{protocol}: {expected}");
            let (status, body) = send(
                inspection,
                "/v1/waypoints",
                json!({
                    "protocol": protocol, "request": raw, "stop_after": "client",
                }),
            )
            .await;
            assert_eq!(status, 200, "{protocol}: {body}");
            let actual: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(actual["stage"], "client");
            let actual_body = if stream {
                actual["value"]["body"].as_str().unwrap().to_owned()
            } else {
                actual["value"]["body"].to_string()
            };
            assert_eq!(
                normalized_body(&actual_body, stream),
                normalized_body(&expected, stream),
                "{protocol}, stream={stream}"
            );
            assert_eq!(engine.requests.lock().unwrap().len(), 1);
            assert_eq!(hook.requests.lock().unwrap().len(), 1);
        }
    }
}

#[tokio::test]
async fn missing_hook_never_falls_through_to_live_generation() {
    let (production, _, engine, _) = setup();
    let (status, body) = send(
        router(production, HookSlot::default()),
        "/v1/waypoints",
        json!({
            "protocol": "chat", "request": input("chat", false), "stop_after": "tokenize",
        }),
    )
    .await;
    assert_failure(status, &body, 400, "tokenize");
    assert!(engine.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn internal_route_is_not_on_public_router() {
    let (production, _, _, _) = setup();
    assert_eq!(send(production, "/v1/waypoints", json!({})).await.0, 404);
}

#[tokio::test]
async fn metadata_cannot_activate_internal_hook() {
    let (production, _, engine, hook) = setup();
    let response = production
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path("chat"))
                .header("content-type", "application/json")
                .header("x-dynamo-meta-waypoints", r#"{"stop_after":"tokenize"}"#)
                .body(Body::from(input("chat", false).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(engine.requests.lock().unwrap().len(), 1);
    assert!(hook.requests.lock().unwrap().is_empty());
}

struct WaitingHook {
    started: Arc<tokio::sync::Notify>,
    stopped: Arc<tokio::sync::Notify>,
}

#[async_trait]
impl AsyncEngine<SingleIn<Value>, ManyOut<Annotated<Value>>, anyhow::Error> for WaitingHook {
    async fn generate(
        &self,
        request: SingleIn<Value>,
    ) -> anyhow::Result<ManyOut<Annotated<Value>>> {
        let context = request.context();
        let stopped = self.stopped.clone();
        tokio::spawn(async move {
            tokio::select! { _ = context.stopped() => {}, _ = context.killed() => {} }
            stopped.notify_one();
        });
        self.started.notify_one();
        futures::future::pending().await
    }
}

#[tokio::test]
async fn timeout_and_disconnect_cancel_before_first_engine_item() {
    for disconnect in [false, true] {
        let (production, _, engine, _) = setup();
        let started = Arc::new(tokio::sync::Notify::new());
        let stopped = Arc::new(tokio::sync::Notify::new());
        let slot = HookSlot::default();
        assert!(
            slot.set(Arc::new(WaitingHook {
                started: started.clone(),
                stopped: stopped.clone()
            }))
            .is_ok()
        );
        let task = tokio::spawn(send(
            router(production, slot),
            "/v1/waypoints",
            json!({
                "protocol":"chat", "request":input("chat", false), "stop_after":"tokenize",
                "preserve_intermediates":true,
                "timeout_ms": if disconnect { 30000 } else { 20 },
            }),
        ));
        tokio::time::timeout(Duration::from_secs(1), started.notified())
            .await
            .unwrap();
        if disconnect {
            task.abort();
            assert!(task.await.unwrap_err().is_cancelled());
        } else {
            let (status, body) = task.await.unwrap();
            let result = assert_failure(status, &body, 408, "tokenize");
            assert_eq!(
                result["intermediates"]["canonical"]["request"]["model"],
                "test"
            );
        }
        tokio::time::timeout(Duration::from_secs(1), stopped.notified())
            .await
            .unwrap();
        assert!(engine.requests.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn production_validation_errors_are_preserved() {
    let (production, inspection, engine, hook) = setup();
    let raw = json!({"model":"test", "input":"hi", "previous_response_id":"resp_prev"});
    let expected = send(production, path("responses"), raw.clone()).await;
    assert_eq!(expected.0, 501);
    let (status, body) = send(
        inspection,
        "/v1/waypoints",
        json!({
            "protocol":"responses", "request":raw, "stop_after":"canonical",
        }),
    )
    .await;
    let result = assert_failure(status, &body, expected.0, "canonical");
    assert_eq!(
        result["value"]["body"],
        serde_json::from_str::<Value>(&expected.1).unwrap()
    );
    assert!(engine.requests.lock().unwrap().is_empty());
    assert!(hook.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn inspection_does_not_consume_live_request_routing_metadata() {
    use super::super::baseten::{
        WorkerResponseMetadata, get_or_create_context_id, publish_worker_response_metadata,
        take_worker_response_metadata,
    };
    for protocol in ["chat", "responses", "messages"] {
        let (_, inspection, _, _) = setup();
        let id = format!("waypoints-isolation-{protocol}");
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("x-baseten-request-id", id.parse().unwrap());
        let context_id = get_or_create_context_id(&headers);
        let worker = WorkerResponseMetadata {
            prefill_worker_id: None,
            prefill_dp_rank: None,
            decode_worker_id: Some(42),
            decode_dp_rank: Some(0),
        };
        publish_worker_response_metadata(context_id.clone(), worker);
        let (status, body) = send(
            inspection,
            "/v1/waypoints",
            json!({
                "protocol":protocol, "request":input(protocol, false), "stop_after":"client",
                "headers": {"x-baseten-request-id": id},
            }),
        )
        .await;
        assert_eq!(status, 200, "{body}");
        assert_eq!(take_worker_response_metadata(&context_id), Some(worker));
    }
}

#[tokio::test]
async fn capture_bound_and_deadline_include_response_body() {
    let large = Router::new().route(path("chat"), post(|| async { "x".repeat(MAX_BYTES + 1) }));
    let spec = json!({"protocol":"chat", "request":{}, "stop_after":"client", "timeout_ms":10});
    let (status, body) = send(
        router(large, HookSlot::default()),
        "/v1/waypoints",
        spec.clone(),
    )
    .await;
    assert_failure(status, &body, 413, "client");
    let slow = Router::new().route(
        path("chat"),
        post(|| async {
            Body::from_stream(futures::stream::pending::<Result<String, std::io::Error>>())
        }),
    );
    let (status, body) = send(router(slow, HookSlot::default()), "/v1/waypoints", spec).await;
    assert_failure(status, &body, 408, "client");
}

fn assert_failure(status: u16, body: &str, original_status: u16, requested: &str) -> Value {
    assert_eq!(status, 200, "{body}");
    let result: Value = serde_json::from_str(body).unwrap();
    assert_eq!(result["stage"], "error", "{body}");
    assert_eq!(result["value"]["status"], original_status, "{body}");
    assert_eq!(result["value"]["requested_stage"], requested);
    result
}

/// One call carries the whole request path: the requested boundary as the artifact, every earlier
/// boundary under `intermediates` — including `canonical`, which the handler records because it
/// never reaches the hook. Keys are in pipeline order, not alphabetical.
#[tokio::test]
async fn preserve_intermediates_returns_every_earlier_boundary_in_one_call() {
    for protocol in ["chat", "responses", "messages"] {
        let (_, inspection, _, hook) = setup();
        let (status, body) = send(
            inspection,
            "/v1/waypoints",
            json!({
                "protocol": protocol, "request": input(protocol, false),
                "stop_after": "client", "preserve_intermediates": true,
            }),
        )
        .await;
        assert_eq!(status, 200, "{protocol}: {body}");
        let artifact: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(artifact["stage"], "client");

        let intermediates = &artifact["intermediates"];
        assert_eq!(
            intermediates["canonical"]["request"]["model"], "test",
            "{protocol}: canonical is recorded by the handler, not the hook: {body}"
        );
        for stage in [
            "render",
            "tokenize",
            "engine_request",
            "engine_output",
            "chat_stream",
        ] {
            assert_eq!(
                intermediates[stage],
                artifact_value(stage),
                "{protocol}/{stage}: {body}"
            );
        }
        assert_eq!(
            intermediates
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect::<Vec<_>>(),
            [
                "canonical",
                "render",
                "tokenize",
                "engine_request",
                "engine_output",
                "chat_stream"
            ],
            "{protocol}: {body}"
        );
        assert_eq!(
            hook.requests.lock().unwrap()[0]["preserve_intermediates"],
            true
        );
    }
}

/// The requested boundary is the artifact and is never repeated as an intermediate; boundaries
/// after it never ran, so they cannot appear.
#[tokio::test]
async fn preserve_intermediates_stops_at_the_requested_boundary() {
    let (_, inspection, _, _) = setup();
    let (status, body) = send(
        inspection,
        "/v1/waypoints",
        json!({
            "protocol": "chat", "request": input("chat", false),
            "stop_after": "tokenize", "preserve_intermediates": true,
        }),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let artifact: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(artifact["stage"], "tokenize");
    assert_eq!(artifact["value"], artifact_value("tokenize"));
    assert_eq!(
        artifact["intermediates"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<Vec<_>>(),
        ["canonical", "render"],
        "{body}"
    );
}

/// Default runs keep the shape they had: one artifact, no `intermediates` key.
#[tokio::test]
async fn omitting_preserve_intermediates_keeps_the_single_artifact_shape() {
    for stage in ["canonical", "tokenize", "client"] {
        let (_, inspection, _, _) = setup();
        let (status, body) = send(
            inspection,
            "/v1/waypoints",
            json!({
                "protocol": "chat", "request": input("chat", false), "stop_after": stage,
            }),
        )
        .await;
        assert_eq!(status, 200, "{stage}: {body}");
        let artifact: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(artifact["stage"], stage);
        assert!(artifact.get("intermediates").is_none(), "{stage}: {body}");
    }
}
