use super::coordinator::{
    RouteAndConnectOutcome, RouteSource, RouterGuardClient, route_and_connect, route_request,
    shield_route_and_connect,
};
use super::guard::{ROUTER_GUARD_CLEANUP_GRACE_PERIOD, RouterRequestGuard};
use super::stream_with_optional_prefill_mark;
use super::types::{
    AdmittedRequestTimings, DeniedRequest, MinReplicaAvailable, PotentialLoadsCheckData,
    PreflightInputs, RouterRequestNew,
};
use crate::context;
use anyhow::Result;
use dynamo_kv_router::protocols::{
    BlockExtraInfo, PotentialLoad as RsPotentialLoad, RouterBackpressureReason, RouterRequest,
    RouterResponse as RsRouterResponse, RoutingConstraints,
};
use dynamo_runtime::pipeline::context::Controller;
use dynamo_runtime::pipeline::{
    AsyncEngineContext, AsyncEngineContextProvider, EngineStream, ResponseStream, async_trait,
    context::Context as RsContext,
};
use dynamo_runtime::protocols::annotated::Annotated as RsAnnotated;
use futures::StreamExt;
use futures::stream;
use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Build an `rmpv::Value` from `jv!` syntax. The request plane
/// carries `rmpv::Value` (not `rmpv::Value`), but `jv!` is
/// the most ergonomic way to build test fixtures, so this macro bridges the two
/// via a JSON-level round-trip (both types impl Serialize + Deserialize).
macro_rules! jv {
    ($($x:tt)*) => { serde_json::from_value::<rmpv::Value>(serde_json::json!($($x)*)).unwrap() };
}

/// Convert a `serde_json::Value` to `rmpv::Value` for test fixtures.
fn jv_value(v: serde_json::Value) -> rmpv::Value {
    serde_json::from_value::<rmpv::Value>(v).expect("json -> rmpv round-trip")
}

const TEST_BLOCK_SIZE: u32 = 32;

/// When Yes, the fake's `direct()` short-circuits with
/// `Err("cancelled by context stop")` when the request context is
/// already stopped or killed -- so the routing-phase / setup-phase
/// cancellation paths surface an explicit cancellation error rather
/// than the scripted `responses` queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CancelRespect {
    No,
    Yes,
}

/// Per-`direct()` observability record: the fake always logs the raw
/// `(instance_id, payload)` tuple into `calls` at entry, but only
/// pushes a `DetailedCall` (with `completed: true`) when `direct()`
/// actually finishes. This separates the two observable cases the
/// cancelled vs. detached-setup contrast relies on: the legacy `calls`
/// log captures the ATTEMPT, while `detailed_calls` captures the
/// COMPLETED work -- a shielded inner that continues after the outer
/// task is aborted still completes (and pushes a `DetailedCall`), an
/// inline-open cancelled by task-drop does not.
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct DetailedCall {
    instance_id: u64,
    method: String,
    observed_pause: bool,
    completed: bool,
}

/// Upgraded `RouterGuardClient` fake that supports the full
/// `route_and_connect` lifecycle: scripted `New` / `Backpressure` /
/// `PotentialLoads` responses, method-aware fixed acknowledgements for
/// `mark_free` / `mark_prefill` (without consuming the scripted queue,
/// so cleanup callbacks never starve the route direct), mid-flight
/// instance-set mutation via `remove_instance` / `remove_available`,
/// deferred "go down" semantics via `auto_remove_on_error`, an
/// artificial `open_delay`, optional multi-chunk stream emission
/// (with in-band `take_while(!is_stopped && !is_killed)` truncation
/// for the stream-cancellation test), and toggled
/// `respect_cancel` so a stopped request context short-circuits
/// `direct()` to `Err("cancelled by context stop")`.
struct RouterGuardClientForTesting {
    endpoint_id: String,
    available_instance_ids: Mutex<Vec<u64>>,
    instance_ids: Mutex<Vec<u64>>,
    responses: Mutex<VecDeque<Result<RsRouterResponse, String>>>,
    stream_chunks_queue: Mutex<Option<VecDeque<Vec<rmpv::Value>>>>,
    respect_cancel: Mutex<CancelRespect>,
    auto_remove_on_error: AtomicBool,
    stream_items_polled: Arc<AtomicUsize>,
    open_delay: Mutex<Duration>,
    first_response_delay: Mutex<Duration>,
    prefill_callback_delay: Mutex<Duration>,
    mark_free_callback_delay: Mutex<Duration>,
    route_contexts: Mutex<Vec<Arc<dyn dynamo_runtime::pipeline::AsyncEngineContext>>>,
    calls: Mutex<Vec<(u64, rmpv::Value)>>,
    detailed_calls: Mutex<Vec<DetailedCall>>,
}

impl RouterGuardClientForTesting {
    /// Backward-compatible 3-arg constructor used by the legacy
    /// route_request tokio tests: defaults `respect_cancel=No`,
    /// `auto_remove_on_error=false`, `open_delay=Duration::ZERO`,
    /// `stream_chunks_queue=None`, empty `calls` / `detailed_calls`.
    fn new(
        available_instance_ids: Vec<u64>,
        instance_ids: Vec<u64>,
        responses: Vec<Result<RsRouterResponse, String>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            endpoint_id: "test.router".to_string(),
            available_instance_ids: Mutex::new(available_instance_ids),
            instance_ids: Mutex::new(instance_ids),
            responses: Mutex::new(responses.into()),
            stream_chunks_queue: Mutex::new(None),
            respect_cancel: Mutex::new(CancelRespect::No),
            auto_remove_on_error: AtomicBool::new(false),
            stream_items_polled: Arc::new(AtomicUsize::new(0)),
            open_delay: Mutex::new(Duration::ZERO),
            first_response_delay: Mutex::new(Duration::ZERO),
            prefill_callback_delay: Mutex::new(Duration::ZERO),
            mark_free_callback_delay: Mutex::new(Duration::ZERO),
            route_contexts: Mutex::new(Vec::new()),
            calls: Mutex::new(Vec::new()),
            detailed_calls: Mutex::new(Vec::new()),
        })
    }

    fn set_respect_cancel(&self, mode: CancelRespect) {
        *self.respect_cancel.lock().unwrap() = mode;
    }
    fn set_auto_remove_on_error(&self, on: bool) {
        self.auto_remove_on_error.store(on, Ordering::Release);
    }
    fn set_open_delay(&self, delay: Duration) {
        *self.open_delay.lock().unwrap() = delay;
    }
    fn set_first_response_delay(&self, delay: Duration) {
        *self.first_response_delay.lock().unwrap() = delay;
    }
    fn set_prefill_callback_delay(&self, delay: Duration) {
        *self.prefill_callback_delay.lock().unwrap() = delay;
    }
    fn set_mark_free_callback_delay(&self, delay: Duration) {
        *self.mark_free_callback_delay.lock().unwrap() = delay;
    }
    fn set_stream_chunks(&self, chunks: Vec<Vec<rmpv::Value>>) {
        *self.stream_chunks_queue.lock().unwrap() = Some(chunks.into());
    }

    /// Remove `id` from both `instance_ids` and `available_instance_ids`
    /// -- simulates the worker instance being de-registered from the
    /// informer mid-flight (the proactive stale check inside
    /// `connect_worker` then sees the worker as absent and reroutes).
    fn remove_instance(&self, id: u64) {
        self.instance_ids.lock().unwrap().retain(|&x| x != id);
        self.available_instance_ids
            .lock()
            .unwrap()
            .retain(|&x| x != id);
    }
    /// Remove `id` only from `available_instance_ids` -- simulates a
    /// component going down (zero available replicas) while remaining
    /// registered (the post-route required-available re-check then
    /// yields `RequiredComponentsDown`).
    fn remove_available(&self, id: u64) {
        self.available_instance_ids
            .lock()
            .unwrap()
            .retain(|&x| x != id);
    }

    fn calls(&self) -> Vec<(u64, rmpv::Value)> {
        self.calls.lock().unwrap().clone()
    }
    fn detailed_calls(&self) -> Vec<DetailedCall> {
        self.detailed_calls.lock().unwrap().clone()
    }
    fn method_call_count(&self, method: &str) -> usize {
        self.detailed_calls
            .lock()
            .unwrap()
            .iter()
            .filter(|c| c.method == method)
            .count()
    }
    fn completed_direct_count(&self) -> usize {
        self.detailed_calls
            .lock()
            .unwrap()
            .iter()
            .filter(|c| c.completed)
            .count()
    }
    fn stream_items_polled_count(&self) -> usize {
        self.stream_items_polled.load(Ordering::Acquire)
    }
    fn route_contexts(&self) -> Vec<Arc<dyn dynamo_runtime::pipeline::AsyncEngineContext>> {
        self.route_contexts.lock().unwrap().clone()
    }
}

#[async_trait]
impl RouterGuardClient for RouterGuardClientForTesting {
    fn endpoint_id(&self) -> String {
        self.endpoint_id.clone()
    }

    fn available_instance_ids(&self) -> Vec<u64> {
        self.available_instance_ids.lock().unwrap().clone()
    }

    fn instance_ids(&self) -> Vec<u64> {
        self.instance_ids.lock().unwrap().clone()
    }

    async fn direct(
        &self,
        request: RsContext<rmpv::Value>,
        instance_id: u64,
    ) -> Result<EngineStream<RsAnnotated<rmpv::Value>>> {
        let data = request.content().clone();
        let context = request.context();
        let method = data["method"].as_str().unwrap_or("").to_string();
        let ctx_stopped_or_killed = context.is_stopped() || context.is_killed();

        // Legacy log: every direct() attempt is recorded so tests can
        // assert the ATTEMPT happened, even if direct() is subsequently
        // cancelled before pushing a `DetailedCall`.
        self.calls.lock().unwrap().push((instance_id, data.clone()));
        if method != "mark_free" && method != "mark_prefill" {
            self.route_contexts.lock().unwrap().push(context.clone());
        }

        // Method-aware fixed acknowledgements for cleanup callbacks:
        // mark_free / mark_prefill MUST NOT consume the scripted
        // responses queue (or a subsequent route direct on the same
        // shared fake would starve). The wire form carries the
        // success tag the cleanup task checks for.
        if method == "mark_free" || method == "mark_prefill" {
            let callback_delay = if method == "mark_prefill" {
                *self.prefill_callback_delay.lock().unwrap()
            } else {
                *self.mark_free_callback_delay.lock().unwrap()
            };
            if callback_delay > Duration::ZERO {
                tokio::time::sleep(callback_delay).await;
            }
            let resp = if method == "mark_free" {
                RsRouterResponse::FreeMarked { success: true }
            } else {
                RsRouterResponse::PrefillMarked { success: true }
            };
            let data = jv_value(serde_json::to_value(&resp)?);
            let stream = stream::iter(vec![RsAnnotated::from_data(data)]);
            let stream: EngineStream<RsAnnotated<rmpv::Value>> =
                ResponseStream::new(Box::pin(stream), context);
            self.detailed_calls.lock().unwrap().push(DetailedCall {
                instance_id,
                method,
                observed_pause: callback_delay > Duration::ZERO,
                completed: true,
            });
            return Ok(stream);
        }

        // Cancellation short-circuit: when the request context is
        // stopped or killed at entry, return an explicit cancellation
        // error -- mirroring how a real worker honouring the linked
        // request ctx would respond.
        if *self.respect_cancel.lock().unwrap() == CancelRespect::Yes && ctx_stopped_or_killed {
            if self.auto_remove_on_error.load(Ordering::Acquire) {
                self.remove_instance(instance_id);
            }
            self.detailed_calls.lock().unwrap().push(DetailedCall {
                instance_id,
                method,
                observed_pause: false,
                completed: false,
            });
            return Err(anyhow::anyhow!("cancelled by context stop"));
        }

        // Artificial open delay (the setup-phase work) -- a shielded
        // detached-setup open continues through this even if the outer
        // task is aborted; an inline cancellable-setup open gets
        // dropped at this await.
        let open_delay = *self.open_delay.lock().unwrap();
        if open_delay > Duration::ZERO {
            tokio::time::sleep(open_delay).await;
        }

        // Multi-chunk stream mode: each direct() consumes one inner
        // Vec and emits it as a stream that truncates itself via an
        // in-band `take_while(!is_stopped && !is_killed)` filter so the
        // stream-cancellation test sees a clean cut at the chunk the
        // parent stop lands on. This branch is used by the WORKER
        // role fake; the ROUTER role fake leaves
        // `stream_chunks_queue=None` and falls through to the legacy
        // single-chunk scripted response.
        if let Some(chunks_queue) = self.stream_chunks_queue.lock().unwrap().as_mut()
            && let Some(chunks) = chunks_queue.pop_front()
        {
            let annotated: Vec<RsAnnotated<rmpv::Value>> =
                chunks.into_iter().map(RsAnnotated::from_data).collect();
            let ctx_for_filter = context.clone();
            let stream_items_polled = self.stream_items_polled.clone();
            let stream = stream::iter(annotated)
                .inspect(move |_| {
                    stream_items_polled.fetch_add(1, Ordering::AcqRel);
                })
                .take_while(move |_| {
                    let stop = ctx_for_filter.is_stopped() || ctx_for_filter.is_killed();
                    std::future::ready(!stop)
                });
            let stream: EngineStream<RsAnnotated<rmpv::Value>> =
                ResponseStream::new(Box::pin(stream), context);
            self.detailed_calls.lock().unwrap().push(DetailedCall {
                instance_id,
                method,
                observed_pause: true,
                completed: true,
            });
            return Ok(stream);
        }

        // Legacy single-chunk: pop one scripted response. Ok ->
        // stream; Err -> Err (optionally removing the instance, used to
        // simulate a worker that goes down mid-flight so the reactive
        // stale check then sees the worker as absent).
        let response = self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| Err("missing scripted router response".to_string()));
        let stream_result: Result<EngineStream<RsAnnotated<rmpv::Value>>> = match response {
            Ok(resp) => {
                let data = jv_value(serde_json::to_value(&resp)?);
                let first_response_delay = *self.first_response_delay.lock().unwrap();
                let stream: std::pin::Pin<
                    Box<dyn futures::Stream<Item = RsAnnotated<rmpv::Value>> + Send>,
                > = if first_response_delay > Duration::ZERO {
                    Box::pin(stream::once(async move {
                        tokio::time::sleep(first_response_delay).await;
                        RsAnnotated::from_data(data)
                    }))
                } else {
                    Box::pin(stream::iter(vec![RsAnnotated::from_data(data)]))
                };
                Ok(ResponseStream::new(stream, context))
            }
            Err(err) => {
                if self.auto_remove_on_error.load(Ordering::Acquire) {
                    self.remove_instance(instance_id);
                }
                Err(anyhow::anyhow!(err))
            }
        };

        let completed = stream_result.is_ok();
        self.detailed_calls.lock().unwrap().push(DetailedCall {
            instance_id,
            method,
            observed_pause: true,
            completed,
        });
        stream_result
    }
}

fn new_response() -> Result<RsRouterResponse, String> {
    Ok(RsRouterResponse::New {
        worker_id: 1,
        dp_rank: 0,
        overlap_blocks: 0,
        best_overlap_blocks: 0,
        dp_strict_rank: false,
    })
}

fn free_marked_response() -> Result<RsRouterResponse, String> {
    Ok(RsRouterResponse::FreeMarked { success: true })
}

async fn wait_for_call_count(client: &RouterGuardClientForTesting, count: usize) {
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if client.calls.lock().unwrap().len() >= count {
                return;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("timed out waiting for router guard client calls");
}

async fn route(
    router: Arc<RouterGuardClientForTesting>,
    request: rmpv::Value,
    request_id: &str,
    require: Vec<MinReplicaAvailable>,
    notify_timeout: Duration,
) -> (RouterRequestGuard, RouteSource) {
    route_request(
        router,
        Arc::new(request),
        request_id.to_string(),
        None,
        require,
        notify_timeout,
        false,
        true,
    )
    .await
    .map(|(guard, source, _timings)| (guard, source))
    .unwrap()
}

#[tokio::test]
async fn no_router_instances_returns_router_backpressure() {
    let router = RouterGuardClientForTesting::new(vec![], vec![], vec![]);

    let (guard, source) = route(
        router.clone(),
        jv!({"method": "new", "tokens": [1]}),
        "req-no-router",
        vec![],
        Duration::from_secs(60),
    )
    .await;

    assert!(matches!(source, RouteSource::RouterBackpressure));
    assert!(!guard.routed());
    assert!(guard.backpressure_reason().contains("do_not_queue"));
    assert!(router.calls().is_empty());
}

#[tokio::test]
async fn min_replica_preflight_returns_required_down_without_routing() {
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![]);
    let required = RouterGuardClientForTesting::new(vec![], vec![], vec![]);

    let (guard, source) = route(
        router.clone(),
        jv!({"method": "new", "tokens": [1]}),
        "req-preflight",
        vec![MinReplicaAvailable {
            name: "prefillworker".to_string(),
            router: required,
        }],
        Duration::from_secs(60),
    )
    .await;

    assert!(matches!(source, RouteSource::RequiredDown { name } if name == "prefillworker"));
    assert!(!guard.routed());
    assert!(guard.backpressure_reason().contains("do_not_queue"));
    assert!(router.calls().is_empty());
}

#[tokio::test]
async fn drop_sends_mark_free_without_mark_prefill() {
    let router = RouterGuardClientForTesting::new(
        vec![7],
        vec![7],
        vec![new_response(), free_marked_response()],
    );

    let (guard, source) = route(
        router.clone(),
        jv!({"method": "new", "tokens": [1]}),
        "req-drop",
        vec![],
        Duration::from_secs(60),
    )
    .await;

    assert!(matches!(source, RouteSource::Routed { worker_id: 1 }));
    assert!(guard.routed());
    drop(guard);

    wait_for_call_count(&router, 2).await;
    let calls = router.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[1].0, 7);
    assert_eq!(calls[1].1["method"].as_str(), Some("mark_free"));
}

#[tokio::test]
async fn notify_timeout_sends_mark_free_and_exits() {
    let router = RouterGuardClientForTesting::new(
        vec![7],
        vec![7],
        vec![new_response(), free_marked_response()],
    );

    let (guard, source) = route(
        router.clone(),
        jv!({"method": "new", "tokens": [1]}),
        "req-timeout",
        vec![],
        Duration::from_millis(5),
    )
    .await;

    assert!(matches!(source, RouteSource::Routed { worker_id: 1 }));
    assert!(guard.routed());
    wait_for_call_count(&router, 2).await;
    let calls = router.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[1].1["method"].as_str(), Some("mark_free"));

    drop(guard);
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(router.calls().len(), 2);
}

#[test]
fn router_request_new_defaults_minimal_wire() {
    let value = RouterRequestNew::default()
        .into_routing_request_value()
        .expect("build ok");

    assert_eq!(value["method"].as_str(), Some("new"));
    assert_eq!(value["tokens"], jv!([]));
    // defaults are skipped on the wire
    assert!(value["block_mm_infos"].is_nil());
    assert!(value["routing_constraints"].is_nil());
    assert!(value["priority_jump"].is_nil());
    assert!(value["priority_load_shed_percent"].is_nil());
    assert!(value["do_not_queue"].is_nil());

    let parsed: RouterRequest =
        serde_json::from_value(serde_json::to_value(&value).unwrap()).expect("round-trips");
    match parsed {
        RouterRequest::New {
            tokens,
            do_not_queue,
            priority_jump,
            ..
        } => {
            assert!(tokens.is_empty());
            assert!(!do_not_queue);
            assert_eq!(priority_jump, 0.0);
        }
        _ => panic!("expected New"),
    }
}

#[test]
fn router_request_new_priority_fields_round_trip() {
    let req = RouterRequestNew {
        tokens: vec![1, 2, 3],
        block_mm_infos: None,
        routing_constraints: RoutingConstraints::default(),
        priority_jump: 0.5,
        priority_load_shed_percent: 10,
        do_not_queue: true,
    };
    let value = req.into_routing_request_value().expect("build ok");

    assert_eq!(value["method"].as_str(), Some("new"));
    assert_eq!(value["tokens"], jv!([1, 2, 3]));
    assert_eq!(value["priority_jump"].as_f64(), Some(0.5));
    assert_eq!(value["priority_load_shed_percent"].as_i64(), Some(10));
    assert_eq!(value["do_not_queue"].as_bool(), Some(true));

    match serde_json::from_value::<RouterRequest>(serde_json::to_value(&value).unwrap())
        .expect("round-trips")
    {
        RouterRequest::New {
            tokens,
            do_not_queue,
            priority_jump,
            priority_load_shed_percent,
            ..
        } => {
            assert_eq!(tokens, vec![1, 2, 3]);
            assert!(do_not_queue);
            assert_eq!(priority_jump, 0.5);
            assert_eq!(priority_load_shed_percent, 10);
        }
        _ => panic!("expected New"),
    }
}

#[test]
fn router_request_new_block_mm_infos_carried() {
    let infos: Vec<Option<BlockExtraInfo>> = serde_json::from_value(
        serde_json::json!([{"mm_objects": [{"mm_hash": 22, "offsets": [[0, 1]]}]}]),
    )
    .expect("block_mm_infos deserializes");
    let req = RouterRequestNew {
        tokens: vec![1, 2, 3],
        block_mm_infos: Some(infos),
        routing_constraints: RoutingConstraints::default(),
        priority_jump: 0.0,
        priority_load_shed_percent: 0,
        do_not_queue: false,
    };
    let value = req.into_routing_request_value().expect("build ok");

    assert_eq!(value["method"].as_str(), Some("new"));
    // block_mm_infos round-trips through the wire tagged payload.
    match serde_json::from_value::<RouterRequest>(serde_json::to_value(&value).unwrap())
        .expect("round-trips")
    {
        RouterRequest::New { block_mm_infos, .. } => {
            let infos = block_mm_infos.expect("block_mm_infos present");
            assert_eq!(infos.len(), 1);
            assert_eq!(infos[0].as_ref().unwrap().mm_objects[0].mm_hash, 22);
        }
        _ => panic!("expected New"),
    }
}

#[test]
fn router_request_new_routing_constraints_non_default_round_trip() {
    let mut required_taints = std::collections::HashSet::new();
    required_taints.insert("gpu".to_string());
    let rc = RoutingConstraints {
        required_taints,
        preferred_taints: std::collections::HashMap::new(),
    };
    let req = RouterRequestNew {
        tokens: vec![1],
        block_mm_infos: None,
        routing_constraints: rc,
        priority_jump: 0.0,
        priority_load_shed_percent: 0,
        do_not_queue: false,
    };
    let value = req.into_routing_request_value().expect("build ok");

    assert_eq!(value["method"].as_str(), Some("new"));
    assert!(!value["routing_constraints"].is_nil());
    match serde_json::from_value::<RouterRequest>(serde_json::to_value(&value).unwrap())
        .expect("round-trips")
    {
        RouterRequest::New {
            routing_constraints,
            ..
        } => {
            assert!(routing_constraints.required_taints.contains("gpu"));
        }
        _ => panic!("expected New"),
    }
}

// ----- end-to-end `route_and_connect` test scaffolding -----

/// Build a non-Python `context::Context` for tests: a `Controller` is
/// the underlying `Arc<dyn AsyncEngineContext>` (the same machinery the
/// bindings use under GIL -- `Controller::new` + `Context::new`), so
/// `route_and_connect`/`connect_worker`/`create_request_context` can
/// link a child to it, propagate stop_generating, and observe
/// `is_stopped()`/`is_killed()` from the fake's `direct()` body.
fn build_test_context(id: &str) -> context::Context {
    let inner: Arc<dyn AsyncEngineContext> = Arc::new(Controller::new(id.to_string()));
    context::Context::new(inner, None, None, BTreeMap::new())
}

fn make_routing_request() -> Arc<rmpv::Value> {
    Arc::new(
        RouterRequestNew::default()
            .into_routing_request_value()
            .expect("default routing request builds"),
    )
}

fn make_worker_request() -> rmpv::Value {
    jv!({"method": "generate", "prompt": "hello"})
}

fn route_response_new(worker_id: u64) -> Result<RsRouterResponse, String> {
    Ok(RsRouterResponse::New {
        worker_id,
        dp_rank: 0,
        overlap_blocks: 0,
        best_overlap_blocks: 0,
        dp_strict_rank: false,
    })
}

fn backpressure_response(
    reason: RouterBackpressureReason,
    queued_isl_tokens: usize,
    max_queued_isl_tokens: Option<usize>,
) -> Result<RsRouterResponse, String> {
    Ok(RsRouterResponse::Backpressure {
        reason,
        queued_isl_tokens,
        max_queued_isl_tokens,
    })
}

/// Single-worker PotentialLoads scripted response used by the preflight tests.
fn potential_loads_response(
    potential_prefill_tokens: usize,
    potential_decode_blocks: usize,
    pending_count: usize,
    pending_isl_tokens: usize,
) -> Result<RsRouterResponse, String> {
    Ok(RsRouterResponse::PotentialLoads {
        loads: vec![RsPotentialLoad {
            worker_id: 1,
            dp_rank: 0,
            potential_prefill_tokens,
            potential_decode_blocks,
            active_requests: 0,
        }],
        pending_count,
        pending_isl_tokens,
    })
}

fn potential_loads_response_for_workers(
    loads: Vec<(usize, usize)>,
    pending_count: usize,
    pending_isl_tokens: usize,
) -> Result<RsRouterResponse, String> {
    Ok(RsRouterResponse::PotentialLoads {
        loads: loads
            .into_iter()
            .enumerate()
            .map(
                |(idx, (potential_prefill_tokens, potential_decode_blocks))| RsPotentialLoad {
                    worker_id: idx as u64 + 1,
                    dp_rank: idx as u32,
                    potential_prefill_tokens,
                    potential_decode_blocks,
                    active_requests: 0,
                },
            )
            .collect(),
        pending_count,
        pending_isl_tokens,
    })
}

/// Test driver: cast both role fakes to `Arc<dyn RouterGuardClient>` and
/// forward to the real `route_and_connect` so a test exercises the
/// production routing -> connect lifecycle end-to-end.
#[allow(clippy::too_many_arguments)]
async fn connect(
    router: Arc<RouterGuardClientForTesting>,
    worker: Arc<RouterGuardClientForTesting>,
    routing_request: Arc<rmpv::Value>,
    request_id: &str,
    context: context::Context,
    require: Vec<MinReplicaAvailable>,
    preflight_inputs: Option<PreflightInputs>,
    worker_request: rmpv::Value,
    max_reroutes: u64,
    allow_cancel_routing: bool,
    allow_cancel_setup: bool,
    notify_timeout: Duration,
) -> Result<RouteAndConnectOutcome> {
    route_and_connect(
        router as Arc<dyn RouterGuardClient>,
        worker as Arc<dyn RouterGuardClient>,
        routing_request,
        request_id.to_string(),
        context,
        require,
        preflight_inputs,
        worker_request,
        TEST_BLOCK_SIZE,
        max_reroutes,
        allow_cancel_routing,
        allow_cancel_setup,
        false,
        notify_timeout,
        false,
        None,
    )
    .await
}

async fn wait_for_method_call_count(
    client: &RouterGuardClientForTesting,
    method: &str,
    count: usize,
    timeout: Duration,
) {
    let method_owned = method.to_string();
    tokio::time::timeout(timeout, async {
        loop {
            if client.method_call_count(method) >= count {
                return;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    })
    .await
    .unwrap_or_else(|_| {
        panic!(
            "timed out waiting for method={} count={}",
            method_owned, count
        )
    });
}

async fn wait_for_completion_count(
    client: &RouterGuardClientForTesting,
    count: usize,
    timeout: Duration,
) {
    tokio::time::timeout(timeout, async {
        loop {
            if client.completed_direct_count() >= count {
                return;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("timed out waiting for completion count={}", count));
}

async fn wait_for_stream_items_polled(
    client: &RouterGuardClientForTesting,
    count: usize,
    timeout: Duration,
) {
    tokio::time::timeout(timeout, async {
        loop {
            if client.stream_items_polled_count() >= count {
                return;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("timed out waiting for stream item poll count={}", count));
}

/// Pull one item off a `Connected` stream and assert it equals `expected`.
/// The `EngineStream` is a `Pin<Box<dyn Stream + Send>>`; `StreamExt::next`
/// polls it once. Used by the stream-cancellation test.
async fn take_one_from_stream(
    stream: &mut EngineStream<RsAnnotated<rmpv::Value>>,
) -> Option<RsAnnotated<rmpv::Value>> {
    stream.as_mut().next().await
}

fn assert_connected_timing_splits(timings: &AdmittedRequestTimings) {
    assert!(
        timings.routing_stream_connect_duration <= timings.routing_new_duration,
        "routing stream connect should be a subspan of routing new: {timings:?}"
    );
    assert!(
        timings.worker_stream_connect_duration <= timings.worker_connect_duration,
        "worker stream connect should be a subspan of worker setup: {timings:?}"
    );
}

// ----- end-to-end `route_and_connect` scenarios -----

#[tokio::test]
async fn route_and_connect_happy_router_response_inject_and_mark_free_on_drop() {
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![route_response_new(1)]);
    let worker = RouterGuardClientForTesting::new(vec![1], vec![1], vec![route_response_new(1)]);
    let context = build_test_context("test-happy-route");

    let outcome = connect(
        router.clone(),
        worker.clone(),
        make_routing_request(),
        "req-happy",
        context.clone(),
        Vec::new(),
        None,
        make_worker_request(),
        0,
        true,
        true,
        Duration::from_secs(60),
    )
    .await
    .expect("happy route");

    match outcome {
        RouteAndConnectOutcome::Connected { worker_id, .. } => assert_eq!(worker_id, 1),
        other => panic!("expected Connected, got {:?}", other),
    }

    // The route direct happened on the router fake; the generation open
    // happened on the worker fake. The injected `router_response` field
    // carries `worker_id: 1`.
    assert_eq!(router.method_call_count("new"), 1);
    assert_eq!(worker.method_call_count("generate"), 1);
    let worker_call = &worker.calls()[0];
    assert_eq!(
        worker_call.1["router_response"]["worker_id"].as_i64(),
        Some(1)
    );

    // Cleanup on drop fires `mark_free` on the ROUTER fake (the guard's
    // `router` field is the routing router). The worker fake never sees
    // `mark_free`.
    drop(outcome);
    wait_for_method_call_count(&router, "mark_free", 1, Duration::from_secs(2)).await;
    assert_eq!(router.method_call_count("mark_free"), 1);
    assert_eq!(worker.method_call_count("mark_free"), 0);
}

#[tokio::test]
async fn route_and_connect_wait_for_first_response_omits_sentinel() {
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![route_response_new(1)]);
    let worker = RouterGuardClientForTesting::new(vec![], vec![1], Vec::new());
    worker.set_stream_chunks(vec![vec![
        jv!({"drop_this_message": true, "internal_healthy": true}),
        jv!({"chunk": 1}),
        jv!({"chunk": 2}),
    ]]);
    let context = build_test_context("test-swallow-first-event");

    let outcome = route_and_connect(
        router.clone() as Arc<dyn RouterGuardClient>,
        worker.clone() as Arc<dyn RouterGuardClient>,
        make_routing_request(),
        "req-swallow-first-event".to_string(),
        context,
        Vec::new(),
        None,
        make_worker_request(),
        TEST_BLOCK_SIZE,
        0,
        true,
        true,
        true,
        Duration::from_secs(60),
        false,
        None,
    )
    .await
    .expect("connect");

    let (guard, worker_id, mut stream, timings) = match outcome {
        RouteAndConnectOutcome::Connected {
            guard,
            worker_id,
            stream,
            timings,
        } => (guard, worker_id, stream, timings),
        other => panic!("expected Connected, got {:?}", other),
    };

    assert_eq!(worker_id, 1);
    assert_connected_timing_splits(&timings);
    assert!(
        timings.worker_first_response_duration.is_none(),
        "sentinel first event should not record first-response timing: {timings:?}"
    );
    assert!(
        timings.worker_sentinel_event_duration.is_some(),
        "sentinel first event should record sentinel timing: {timings:?}"
    );
    assert_eq!(
        worker.stream_items_polled_count(),
        1,
        "setup should consume exactly the readiness event before returning"
    );

    let first_visible = take_one_from_stream(&mut stream)
        .await
        .expect("remaining stream should contain first visible item");
    assert_eq!(first_visible.data, Some(jv!({"chunk": 1})));

    let second_visible = take_one_from_stream(&mut stream)
        .await
        .expect("remaining stream should contain second visible item");
    assert_eq!(second_visible.data, Some(jv!({"chunk": 2})));

    assert_eq!(worker.stream_items_polled_count(), 3);
    drop(stream);
    drop(guard);
    wait_for_method_call_count(&router, "mark_free", 1, Duration::from_secs(2)).await;
}

#[tokio::test]
async fn route_and_connect_wait_for_first_response_replays_non_sentinel_item() {
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![route_response_new(1)]);
    let worker = RouterGuardClientForTesting::new(vec![], vec![1], Vec::new());
    worker.set_stream_chunks(vec![vec![jv!({"chunk": 1}), jv!({"chunk": 2})]]);
    let context = build_test_context("test-swallow-first-event-non-health");

    let outcome = route_and_connect(
        router.clone() as Arc<dyn RouterGuardClient>,
        worker.clone() as Arc<dyn RouterGuardClient>,
        make_routing_request(),
        "req-swallow-first-event-non-health".to_string(),
        context,
        Vec::new(),
        None,
        make_worker_request(),
        TEST_BLOCK_SIZE,
        0,
        true,
        true,
        true,
        Duration::from_secs(60),
        false,
        None,
    )
    .await
    .expect("connect");

    let (guard, worker_id, mut stream, timings) = match outcome {
        RouteAndConnectOutcome::Connected {
            guard,
            worker_id,
            stream,
            timings,
        } => (guard, worker_id, stream, timings),
        other => panic!("expected Connected, got {:?}", other),
    };

    assert_eq!(worker_id, 1);
    assert_connected_timing_splits(&timings);
    assert!(
        timings.worker_first_response_duration.is_some(),
        "non-sentinel first event should record first-response timing: {timings:?}"
    );
    assert!(
        timings.worker_sentinel_event_duration.is_none(),
        "non-sentinel first event should not record sentinel timing: {timings:?}"
    );
    assert_eq!(
        worker.stream_items_polled_count(),
        1,
        "setup should inspect exactly the first event before returning"
    );

    let first_visible = take_one_from_stream(&mut stream)
        .await
        .expect("first non-health item should be replayed");
    assert_eq!(first_visible.data, Some(jv!({"chunk": 1})));
    assert_eq!(
        worker.stream_items_polled_count(),
        1,
        "replayed first item should not poll the worker stream again"
    );

    let second_visible = take_one_from_stream(&mut stream)
        .await
        .expect("remaining stream should contain second item");
    assert_eq!(second_visible.data, Some(jv!({"chunk": 2})));

    drop(stream);
    drop(guard);
    wait_for_method_call_count(&router, "mark_free", 1, Duration::from_secs(2)).await;
}

#[tokio::test]
async fn stream_prefill_mark_skips_internal_event_and_marks_first_real_item() {
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![route_response_new(1)]);
    let worker = RouterGuardClientForTesting::new(vec![], vec![1], Vec::new());
    worker.set_stream_chunks(vec![vec![
        jv!({"drop_this_message": true, "internal_healthy": true}),
        jv!({"chunk": 1}),
    ]]);
    let context = build_test_context("test-prefill-mark-first-real-item");

    let outcome = connect(
        router.clone(),
        worker,
        make_routing_request(),
        "req-prefill-mark-first-real-item",
        context,
        Vec::new(),
        None,
        make_worker_request(),
        0,
        true,
        true,
        Duration::from_secs(60),
    )
    .await
    .expect("connect");

    let (guard, mut stream) = match outcome {
        RouteAndConnectOutcome::Connected { guard, stream, .. } => (Arc::new(guard), stream),
        other => panic!("expected Connected, got {:?}", other),
    };
    stream = stream_with_optional_prefill_mark(stream, Arc::clone(&guard), true);

    let internal = take_one_from_stream(&mut stream)
        .await
        .expect("internal event should still be forwarded without first-event mutation");
    assert_eq!(
        internal.data,
        Some(jv!({"drop_this_message": true, "internal_healthy": true}))
    );
    assert_eq!(router.method_call_count("mark_prefill"), 0);

    let first_real = take_one_from_stream(&mut stream)
        .await
        .expect("real worker item should follow internal event");
    assert_eq!(first_real.data, Some(jv!({"chunk": 1})));
    wait_for_method_call_count(&router, "mark_prefill", 1, Duration::from_secs(2)).await;

    drop(stream);
    drop(guard);
    wait_for_method_call_count(&router, "mark_free", 1, Duration::from_secs(2)).await;
}

#[tokio::test]
async fn route_and_connect_wait_for_first_response_uses_sentinel_behavior() {
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![route_response_new(1)]);
    let worker = RouterGuardClientForTesting::new(vec![], vec![1], Vec::new());
    worker.set_stream_chunks(vec![vec![
        jv!({"drop_this_message": true, "internal_healthy": true}),
        jv!({"chunk": 1}),
    ]]);
    let context = build_test_context("test-wait-and-return-first-event-sentinel");

    let outcome = route_and_connect(
        router.clone() as Arc<dyn RouterGuardClient>,
        worker.clone() as Arc<dyn RouterGuardClient>,
        make_routing_request(),
        "req-wait-and-return-first-event-sentinel".to_string(),
        context,
        Vec::new(),
        None,
        make_worker_request(),
        TEST_BLOCK_SIZE,
        0,
        true,
        true,
        true,
        Duration::from_secs(60),
        false,
        None,
    )
    .await
    .expect("connect");

    let (guard, _worker_id, mut stream, timings) = match outcome {
        RouteAndConnectOutcome::Connected {
            guard,
            worker_id,
            stream,
            timings,
        } => (guard, worker_id, stream, timings),
        other => panic!("expected Connected, got {:?}", other),
    };

    assert_connected_timing_splits(&timings);
    assert!(
        timings.worker_first_response_duration.is_none(),
        "sentinel first event should not record first-response timing: {timings:?}"
    );
    assert!(
        timings.worker_sentinel_event_duration.is_some(),
        "sentinel first event should record sentinel timing: {timings:?}"
    );
    assert_eq!(
        worker.stream_items_polled_count(),
        1,
        "setup should wait for exactly the first event before returning"
    );

    let first_visible = take_one_from_stream(&mut stream)
        .await
        .expect("sentinel should be swallowed before first visible item");
    assert_eq!(first_visible.data, Some(jv!({"chunk": 1})));

    drop(stream);
    drop(guard);
    wait_for_method_call_count(&router, "mark_free", 1, Duration::from_secs(2)).await;
}

#[tokio::test]
async fn route_and_connect_wait_for_first_response_failure_returns_denied() {
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![route_response_new(1)]);
    let worker = RouterGuardClientForTesting::new(vec![], vec![1], Vec::new());
    worker.set_stream_chunks(vec![vec![]]);
    let context = build_test_context("test-swallow-first-event-denied");

    let outcome = route_and_connect(
        router.clone() as Arc<dyn RouterGuardClient>,
        worker.clone() as Arc<dyn RouterGuardClient>,
        make_routing_request(),
        "req-swallow-first-event-denied".to_string(),
        context,
        Vec::new(),
        None,
        make_worker_request(),
        TEST_BLOCK_SIZE,
        0,
        true,
        true,
        true,
        Duration::from_secs(60),
        false,
        None,
    )
    .await
    .expect("first event failure should be a denial, not a raised error");

    match outcome {
        RouteAndConnectOutcome::Denied(DeniedRequest::FirstWorkerEventFailed { error }) => {
            assert!(
                error.contains("worker stream ended before first event"),
                "unexpected error: {error}"
            );
        }
        other => panic!("expected Denied(FirstWorkerEventFailed), got {:?}", other),
    }

    wait_for_method_call_count(&router, "mark_free", 1, Duration::from_secs(2)).await;
    assert_eq!(worker.method_call_count("generate"), 1);
    assert_eq!(router.method_call_count("mark_free"), 1);
}

#[tokio::test]
async fn route_and_connect_proactive_stale_reroutes_then_connects() {
    // First route returns worker 1 -- but the worker fake's instance set
    // only contains 2, so the proactive stale check fires before any
    // `direct()` on the worker. Second route returns worker 2 and the
    // connect succeeds.
    let router = RouterGuardClientForTesting::new(
        vec![7],
        vec![7],
        vec![route_response_new(1), route_response_new(2)],
    );
    let worker = RouterGuardClientForTesting::new(vec![], vec![2], vec![route_response_new(2)]);
    let context = build_test_context("test-proactive-stale");

    let started = Instant::now();
    let outcome = connect(
        router.clone(),
        worker.clone(),
        make_routing_request(),
        "req-proactive-stale",
        context,
        Vec::new(),
        None,
        make_worker_request(),
        1,
        true,
        true,
        Duration::from_secs(60),
    )
    .await
    .expect("connect");
    assert!(started.elapsed() >= ROUTER_GUARD_CLEANUP_GRACE_PERIOD);

    match &outcome {
        RouteAndConnectOutcome::Connected {
            worker_id, timings, ..
        } => {
            assert_eq!(*worker_id, 2);
            assert_eq!(timings.stale_reroutes, 1);
            assert!(timings.routing_new_duration >= ROUTER_GUARD_CLEANUP_GRACE_PERIOD);
            assert!(timings.worker_connect_duration < timings.routing_new_duration);
        }
        other => panic!("expected Connected, got {:?}", other),
    }

    // Two route attempts reached the router fake; the first guard was
    // dropped inside `connect_worker` (proactive stale frees the guard),
    // firing one mark_free; the second guard is dropped here after the
    // Connected outcome, firing another mark_free.
    assert_eq!(router.method_call_count("new"), 2);
    assert_eq!(worker.method_call_count("generate"), 1);
    // The payload is moved into each attempt and handed back on the stale
    // pre-check, not cloned: the retry must deliver the original fields
    // intact with exactly one `router_response` entry (attempt 2's).
    let worker_call = &worker.calls()[0];
    assert_eq!(worker_call.1["prompt"].as_str(), Some("hello"));
    assert_eq!(
        worker_call.1["router_response"]["worker_id"].as_i64(),
        Some(2)
    );
    let router_response_entries = match &worker_call.1 {
        rmpv::Value::Map(map) => map
            .iter()
            .filter(|(k, _)| k.as_str() == Some("router_response"))
            .count(),
        _ => panic!("worker request should be a map"),
    };
    assert_eq!(router_response_entries, 1);
    drop(outcome);
    wait_for_method_call_count(&router, "mark_free", 2, Duration::from_secs(2)).await;
    assert_eq!(router.method_call_count("mark_free"), 2);
}

#[tokio::test]
async fn route_and_connect_stale_loop_exhausted_returns_next_router_unreachable() {
    // Router keeps choosing worker 1; worker fake has NO registered
    // instances, so every route is proactively stale. With max_reroutes=1
    // the loop runs (initial + 1 reroute) = 2 attempts, exhausting the
    // bound and returning Denied::NextRouterUnreachable
    // {"stale route loop exhausted"}.
    let router = RouterGuardClientForTesting::new(
        vec![7],
        vec![7],
        vec![route_response_new(1), route_response_new(1)],
    );
    let worker = RouterGuardClientForTesting::new(vec![], vec![], vec![]);
    let context = build_test_context("test-stale-exhausted");

    let outcome = connect(
        router.clone(),
        worker.clone(),
        make_routing_request(),
        "req-stale-exhausted",
        context,
        Vec::new(),
        None,
        make_worker_request(),
        1,
        true,
        true,
        Duration::from_secs(60),
    )
    .await
    .expect("denied, not raised");

    match outcome {
        RouteAndConnectOutcome::Denied(DeniedRequest::NextRouterUnreachable { error }) => {
            assert_eq!(error, "stale route loop exhausted");
        }
        other => panic!("expected Denied(NextRouterUnreachable), got {:?}", other),
    }

    // Both stale guards were dropped inside connect_worker, both firing
    // mark_free on the router fake.
    assert_eq!(router.method_call_count("new"), 2);
    assert_eq!(worker.method_call_count("generate"), 0);
    wait_for_method_call_count(&router, "mark_free", 2, Duration::from_secs(2)).await;
    assert_eq!(router.method_call_count("mark_free"), 2);
}

#[tokio::test]
async fn route_and_connect_reactive_stale_reroutes_then_connects() {
    // Router returns worker 1; the worker fake instance set contains
    // {1, 2} so the proactive check passes; but the worker's first direct
    // returns Err and auto-removes worker 1 -- so the reactive stale
    // check inside connect_worker fires (worker now absent) and the loop
    // reroutes. The first attempt's payload copy was consumed by the
    // failed open, so the retry copies the base payload again. Second
    // route returns worker 2 and (since worker 2 is still in the
    // instance set) connect succeeds.
    let router = RouterGuardClientForTesting::new(
        vec![7],
        vec![7],
        vec![route_response_new(1), route_response_new(2)],
    );
    let worker = RouterGuardClientForTesting::new(
        vec![],
        vec![1, 2],
        vec![Err("worker_open_error".to_string()), route_response_new(2)],
    );
    worker.set_auto_remove_on_error(true);
    let context = build_test_context("test-reactive-stale");

    let started = Instant::now();
    let outcome = connect(
        router.clone(),
        worker.clone(),
        make_routing_request(),
        "req-reactive-stale",
        context,
        Vec::new(),
        None,
        make_worker_request(),
        1,
        true,
        true,
        Duration::from_secs(60),
    )
    .await
    .expect("connect");
    assert!(started.elapsed() >= ROUTER_GUARD_CLEANUP_GRACE_PERIOD);

    match &outcome {
        RouteAndConnectOutcome::Connected {
            worker_id, timings, ..
        } => {
            assert_eq!(*worker_id, 2);
            assert_eq!(timings.stale_reroutes, 1);
            assert!(timings.routing_new_duration >= ROUTER_GUARD_CLEANUP_GRACE_PERIOD);
            assert!(timings.worker_connect_duration < timings.routing_new_duration);
        }
        other => panic!("expected Connected, got {:?}", other),
    }

    assert_eq!(router.method_call_count("new"), 2);
    // First worker direct errored (incomplete); second succeeded.
    assert_eq!(worker.completed_direct_count(), 1);
    // The retry's payload is a fresh copy of the untouched base: original
    // fields intact, exactly one `router_response` entry, attempt 2's id.
    let retry_call = &worker.calls()[1];
    assert_eq!(retry_call.1["prompt"].as_str(), Some("hello"));
    assert_eq!(
        retry_call.1["router_response"]["worker_id"].as_i64(),
        Some(2)
    );
    let router_response_entries = match &retry_call.1 {
        rmpv::Value::Map(map) => map
            .iter()
            .filter(|(k, _)| k.as_str() == Some("router_response"))
            .count(),
        _ => panic!("worker request should be a map"),
    };
    assert_eq!(router_response_entries, 1);
    drop(outcome);
    wait_for_method_call_count(&router, "mark_free", 2, Duration::from_secs(2)).await;
    assert_eq!(router.method_call_count("mark_free"), 2);
}

#[tokio::test]
async fn route_and_connect_non_stale_open_error_raises_and_frees_guard() {
    // worker fake's first direct errors, but auto_remove_on_error is OFF,
    // so the worker remains in the instance set and connect_worker takes
    // the `Other(err)` arm (not `StaleWorker`), which route_and_connect
    // propagates as `Err` -- the test caller sees a panic if connect()
    // returned Ok. After the raise, the armed guard's drop fires
    // mark_free.
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![route_response_new(1)]);
    let worker = RouterGuardClientForTesting::new(
        vec![],
        vec![1],
        vec![Err("non_stale_open_error".to_string())],
    );
    let context = build_test_context("test-non-stale-raise");

    let err = connect(
        router.clone(),
        worker.clone(),
        make_routing_request(),
        "req-non-stale",
        context,
        Vec::new(),
        None,
        make_worker_request(),
        0,
        true,
        true,
        Duration::from_secs(60),
    )
    .await
    .expect_err("non-stale open failure raises");

    assert!(
        err.to_string().contains("non_stale_open_error"),
        "error should carry the worker's open error, got: {}",
        err
    );

    wait_for_method_call_count(&router, "mark_free", 1, Duration::from_secs(2)).await;
    assert_eq!(router.method_call_count("mark_free"), 1);
    assert_eq!(worker.method_call_count("generate"), 1);
    assert_eq!(worker.completed_direct_count(), 0);
}

#[tokio::test]
async fn route_and_connect_router_backpressure_returns_denied() {
    let router = RouterGuardClientForTesting::new(
        vec![7],
        vec![7],
        vec![backpressure_response(
            RouterBackpressureReason::DoNotQueue,
            10,
            Some(100),
        )],
    );
    let worker = RouterGuardClientForTesting::new(vec![], vec![], vec![]);
    let context = build_test_context("test-backpressure");

    let outcome = connect(
        router.clone(),
        worker.clone(),
        make_routing_request(),
        "req-bp",
        context,
        Vec::new(),
        None,
        make_worker_request(),
        0,
        true,
        true,
        Duration::from_secs(60),
    )
    .await
    .expect("denied, not raised");

    match outcome {
        RouteAndConnectOutcome::Denied(DeniedRequest::RouterBackpressure {
            reason,
            queued_isl_tokens,
            max_queued_isl_tokens,
        }) => {
            assert_eq!(reason, "do_not_queue");
            assert_eq!(queued_isl_tokens, 10);
            assert_eq!(max_queued_isl_tokens, Some(100));
        }
        other => panic!("expected Denied(RouterBackpressure), got {:?}", other),
    }

    // Unarmed guard -- cleanup task was NOT spawned, so no mark_free.
    assert_eq!(router.method_call_count("mark_free"), 0);
    assert_eq!(worker.method_call_count("generate"), 0);
}

#[tokio::test]
async fn route_and_connect_router_queue_backpressure_reason_is_preserved() {
    let router = RouterGuardClientForTesting::new(
        vec![7],
        vec![7],
        vec![backpressure_response(
            RouterBackpressureReason::MaxQueuedIslTokensExceeded,
            512,
            Some(512),
        )],
    );
    let worker = RouterGuardClientForTesting::new(vec![], vec![], vec![]);
    let context = build_test_context("test-router-queue-backpressure");

    let outcome = connect(
        router.clone(),
        worker.clone(),
        make_routing_request(),
        "req-router-queue-bp",
        context,
        Vec::new(),
        None,
        make_worker_request(),
        0,
        true,
        true,
        Duration::from_secs(60),
    )
    .await
    .expect("denied, not raised");

    match outcome {
        RouteAndConnectOutcome::Denied(DeniedRequest::RouterBackpressure {
            reason,
            queued_isl_tokens,
            max_queued_isl_tokens,
        }) => {
            assert_eq!(reason, "max_queued_isl_tokens_exceeded");
            assert_eq!(queued_isl_tokens, 512);
            assert_eq!(max_queued_isl_tokens, Some(512));
        }
        other => panic!("expected Denied(RouterBackpressure), got {:?}", other),
    }

    assert_eq!(router.method_call_count("mark_free"), 0);
    assert_eq!(worker.method_call_count("generate"), 0);
}

#[tokio::test]
async fn route_and_connect_cancellable_routing_frees_late_new_after_parent_stop() {
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![route_response_new(1)]);
    router.set_open_delay(Duration::from_millis(200));
    let worker = RouterGuardClientForTesting::new(vec![], vec![1], vec![route_response_new(1)]);
    let context = build_test_context("test-route-late-new-cancelled");

    let router_for_task = router.clone();
    let worker_for_task = worker.clone();
    let context_for_task = context.clone();
    let task = tokio::spawn(async move {
        connect(
            router_for_task,
            worker_for_task,
            make_routing_request(),
            "req-route-late-new-cancelled",
            context_for_task,
            Vec::new(),
            None,
            make_worker_request(),
            0,
            true,
            true,
            Duration::from_secs(60),
        )
        .await
    });

    wait_for_call_count(&router, 1).await;
    context.inner().stop_generating();

    let outcome = task
        .await
        .expect("task should not panic")
        .expect("cancelled denial should not raise");
    assert!(matches!(
        outcome,
        RouteAndConnectOutcome::Denied(DeniedRequest::Cancelled())
    ));
    wait_for_method_call_count(&router, "mark_free", 1, Duration::from_secs(2)).await;
    assert_eq!(router.method_call_count("mark_free"), 1);
    assert_eq!(worker.method_call_count("generate"), 0);
}

#[tokio::test]
async fn route_and_connect_cancellable_setup_frees_late_stream_after_parent_stop() {
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![route_response_new(1)]);
    let worker = RouterGuardClientForTesting::new(vec![], vec![1], vec![route_response_new(1)]);
    worker.set_open_delay(Duration::from_millis(200));
    let context = build_test_context("test-setup-late-stream-cancelled");

    let router_for_task = router.clone();
    let worker_for_task = worker.clone();
    let context_for_task = context.clone();
    let task = tokio::spawn(async move {
        connect(
            router_for_task,
            worker_for_task,
            make_routing_request(),
            "req-setup-late-stream-cancelled",
            context_for_task,
            Vec::new(),
            None,
            make_worker_request(),
            0,
            true,
            true,
            Duration::from_secs(60),
        )
        .await
    });

    wait_for_call_count(&worker, 1).await;
    context.inner().stop_generating();

    let outcome = task
        .await
        .expect("task should not panic")
        .expect("cancelled denial should not raise");
    assert!(matches!(
        outcome,
        RouteAndConnectOutcome::Denied(DeniedRequest::Cancelled())
    ));
    wait_for_method_call_count(&router, "mark_free", 1, Duration::from_secs(2)).await;
    assert_eq!(router.method_call_count("mark_free"), 1);
    assert_eq!(worker.method_call_count("generate"), 1);
}

#[tokio::test]
async fn route_and_connect_detached_setup_ignores_already_stopped_parent() {
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![route_response_new(1)]);
    router.set_respect_cancel(CancelRespect::Yes);
    let worker = RouterGuardClientForTesting::new(vec![], vec![1], vec![route_response_new(1)]);
    worker.set_respect_cancel(CancelRespect::Yes);
    let context = build_test_context("test-detached-setup-stopped-parent");
    context.inner().stop_generating();

    let outcome = connect(
        router.clone(),
        worker.clone(),
        make_routing_request(),
        "req-detached-setup-stopped-parent",
        context,
        Vec::new(),
        None,
        make_worker_request(),
        0,
        false,
        false,
        Duration::from_secs(60),
    )
    .await
    .expect("detached route/setup should ignore already stopped parent");

    match &outcome {
        RouteAndConnectOutcome::Connected { worker_id, .. } => assert_eq!(*worker_id, 1),
        other => panic!("expected Connected, got {:?}", other),
    }

    let route_contexts = router.route_contexts();
    assert_eq!(route_contexts.len(), 1);
    assert!(!route_contexts[0].is_stopped());
    let worker_contexts = worker.route_contexts();
    assert_eq!(worker_contexts.len(), 1);
    assert!(!worker_contexts[0].is_stopped());

    drop(outcome);
    wait_for_method_call_count(&router, "mark_free", 1, Duration::from_secs(2)).await;
}

#[tokio::test]
async fn route_request_parent_kill_kills_detached_route_context_when_cancellable() {
    async fn wait_for_context_killed(
        context: Arc<dyn dynamo_runtime::pipeline::AsyncEngineContext>,
        timeout: Duration,
    ) {
        tokio::time::timeout(timeout, async {
            loop {
                if context.is_killed() {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        })
        .await
        .expect("timed out waiting for context to be killed");
    }

    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![route_response_new(1)]);
    router.set_open_delay(Duration::from_millis(200));
    let context = build_test_context("test-route-parent-kill");

    let route_task = tokio::spawn(route_request(
        router.clone(),
        make_routing_request(),
        "req-route-parent-kill".to_string(),
        Some(context.clone()),
        Vec::new(),
        Duration::from_secs(60),
        false,
        true,
    ));

    wait_for_call_count(&router, 1).await;
    let route_contexts = router.route_contexts();
    assert_eq!(route_contexts.len(), 1);
    assert!(!route_contexts[0].is_killed());

    context
        .inner()
        .kill_with_reason(Some("test_parent_context_killed"));
    wait_for_context_killed(route_contexts[0].clone(), Duration::from_secs(1)).await;

    let (guard, source, _timings) = route_task
        .await
        .expect("route task should not panic")
        .expect("fake route still returns its scripted response");
    assert!(matches!(source, RouteSource::Routed { worker_id: 1 }));
    drop(guard);
    wait_for_method_call_count(&router, "mark_free", 1, Duration::from_secs(2)).await;
}

#[tokio::test(start_paused = true)]
async fn route_request_success_aborts_parent_cancellation_forwarder() {
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![route_response_new(1)]);
    let context = build_test_context("test-route-forwarder-abort");

    let (guard, source, _timings) = route_request(
        router.clone(),
        make_routing_request(),
        "req-route-forwarder-abort".to_string(),
        Some(context),
        Vec::new(),
        Duration::from_secs(60),
        false,
        true,
    )
    .await
    .expect("route should succeed");
    assert!(matches!(source, RouteSource::Routed { worker_id: 1 }));

    let route_contexts = router.route_contexts();
    assert_eq!(route_contexts.len(), 1);
    assert!(!route_contexts[0].is_killed());

    tokio::time::sleep(Duration::from_secs(592)).await;
    assert!(
        !route_contexts[0].is_killed(),
        "successful route should abort its cancellation forwarder"
    );

    drop(guard);
    wait_for_method_call_count(&router, "mark_free", 1, Duration::from_secs(2)).await;
}

#[tokio::test(start_paused = true)]
async fn route_and_connect_first_response_timeout_kills_route_context_and_backpressures() {
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![route_response_new(1)]);
    router.set_first_response_delay(Duration::from_secs(600));
    let worker = RouterGuardClientForTesting::new(vec![], vec![1], vec![route_response_new(1)]);
    let context = build_test_context("test-first-response-timeout");

    let outcome = connect(
        router.clone(),
        worker.clone(),
        make_routing_request(),
        "req-first-response-timeout",
        context,
        Vec::new(),
        None,
        make_worker_request(),
        0,
        true,
        true,
        Duration::from_secs(60),
    )
    .await
    .expect("timeout maps to router backpressure, not an error");

    match outcome {
        RouteAndConnectOutcome::Denied(DeniedRequest::RouterBackpressure {
            reason,
            queued_isl_tokens,
            max_queued_isl_tokens,
        }) => {
            assert_eq!(reason, "do_not_queue");
            assert_eq!(queued_isl_tokens, 0);
            assert_eq!(max_queued_isl_tokens, None);
        }
        other => panic!("expected Denied(RouterBackpressure), got {:?}", other),
    }

    wait_for_method_call_count(&router, "mark_free", 1, Duration::from_secs(2)).await;
    assert_eq!(router.method_call_count("mark_free"), 1);
    assert_eq!(worker.method_call_count("generate"), 0);

    let route_contexts = router.route_contexts();
    assert_eq!(route_contexts.len(), 1);
    assert!(
        route_contexts[0].is_killed(),
        "timeout should kill the detached route context"
    );
}

#[tokio::test]
async fn route_and_connect_require_available_goes_down_post_route_returns_denied() {
    // The required component is up at t=0 (preflight passes), goes down
    // DURING the route (router fake open_delay=30ms gives the window),
    // then the post-route re-check inside `route_once` fires
    // `RequiredComponentsDown`. The armed guard is dropped -- mark_free
    // fires -- and the worker's connect never happens.
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![route_response_new(1)]);
    router.set_open_delay(Duration::from_millis(30));
    let worker = RouterGuardClientForTesting::new(vec![], vec![1], vec![route_response_new(1)]);
    let required = RouterGuardClientForTesting::new(vec![7], vec![7], vec![]);
    let context = build_test_context("test-require-down");

    let required_clone = required.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(10)).await;
        required_clone.remove_available(7);
    });

    let outcome = connect(
        router.clone(),
        worker.clone(),
        make_routing_request(),
        "req-post-route-down",
        context,
        vec![MinReplicaAvailable {
            name: "prefillworker".to_string(),
            router: required,
        }],
        None,
        make_worker_request(),
        0,
        true,
        true,
        Duration::from_secs(60),
    )
    .await
    .expect("denied, not raised");

    match outcome {
        RouteAndConnectOutcome::Denied(DeniedRequest::RequiredComponentsDown { name }) => {
            assert_eq!(name, "prefillworker");
        }
        other => panic!("expected Denied(RequiredComponentsDown), got {:?}", other),
    }

    // Route reached the router; worker connect never happened; guard drop
    // fires mark_free.
    assert_eq!(router.method_call_count("new"), 1);
    assert_eq!(router.method_call_count("mark_free"), 0);
    assert_eq!(worker.method_call_count("generate"), 0);
    wait_for_method_call_count(&router, "mark_free", 1, Duration::from_secs(2)).await;
    assert_eq!(router.method_call_count("mark_free"), 1);
}

#[tokio::test]
async fn route_and_connect_preflight_overflow_returns_next_router_backpressure() {
    // Single-worker p50 prefill tokens (10_000) exceed the threshold (1_000)
    // -> evaluate_potential_loads returns Some(NextRouterBackpressureInfo).
    // The potential-loads preflight runs before the route, so the route request
    // is never sent and there is no routed guard to free.
    // The other thresholds are set high so only prefill trips.
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![route_response_new(1)]);
    let worker = RouterGuardClientForTesting::new(vec![], vec![1], vec![route_response_new(1)]);
    let next = RouterGuardClientForTesting::new(
        vec![9],
        vec![9],
        vec![potential_loads_response(10_000, 1_000, 50, 5_000)],
    );
    let context = build_test_context("test-preflight-overflow");

    let preflight_inputs = PreflightInputs {
        check: PotentialLoadsCheckData {
            router: next.clone(),
            queue_depth_threshold: 100,
            prefill_tokens_threshold: 1_000,
            decode_tokens_threshold: 100_000,
            load_percentile: 0.5,
        },
        tokens: vec![1],
        block_mm_infos: None,
    };

    let outcome = connect(
        router.clone(),
        worker.clone(),
        make_routing_request(),
        "req-pf-overflow",
        context,
        Vec::new(),
        Some(preflight_inputs),
        make_worker_request(),
        0,
        true,
        true,
        Duration::from_secs(60),
    )
    .await
    .expect("denied, not raised");

    match outcome {
        RouteAndConnectOutcome::Denied(DeniedRequest::NextRouterBackpressure {
            queue_depth,
            pending_isl_tokens,
            total_prefill_tokens,
            total_decode_blocks,
        }) => {
            assert_eq!(queue_depth, 50);
            assert_eq!(pending_isl_tokens, 5_000);
            assert_eq!(total_prefill_tokens, 10_000);
            assert_eq!(total_decode_blocks, 1_000);
        }
        other => panic!("expected Denied(NextRouterBackpressure), got {:?}", other),
    }

    // Preflight ran on next; route and worker were never reached, so no routed
    // guard exists and no mark_free is sent.
    assert_eq!(next.method_call_count("potential_loads"), 1);
    assert_eq!(router.method_call_count("new"), 0);
    assert_eq!(router.method_call_count("mark_free"), 0);
    assert_eq!(worker.method_call_count("generate"), 0);
}

#[tokio::test]
async fn route_and_connect_preflight_uses_configured_load_percentile_not_sum() {
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![route_response_new(1)]);
    let worker = RouterGuardClientForTesting::new(vec![], vec![1], vec![route_response_new(1)]);
    let next = RouterGuardClientForTesting::new(
        vec![9],
        vec![9],
        vec![potential_loads_response_for_workers(
            vec![(10, 1), (20, 2), (1_000, 100)],
            0,
            0,
        )],
    );
    let context = build_test_context("test-preflight-percentile");

    let preflight_inputs = PreflightInputs {
        check: PotentialLoadsCheckData {
            router: next.clone(),
            queue_depth_threshold: 0,
            prefill_tokens_threshold: 500,
            decode_tokens_threshold: 500,
            load_percentile: 0.5,
        },
        tokens: vec![1],
        block_mm_infos: None,
    };

    let outcome = connect(
        router.clone(),
        worker.clone(),
        make_routing_request(),
        "req-pf-percentile",
        context,
        Vec::new(),
        Some(preflight_inputs),
        make_worker_request(),
        0,
        true,
        true,
        Duration::from_secs(60),
    )
    .await
    .expect("ok");

    match outcome {
        RouteAndConnectOutcome::Connected { worker_id, .. } => assert_eq!(worker_id, 1),
        other => panic!(
            "expected Connected because p50 load passes, got {:?}",
            other
        ),
    }

    assert_eq!(next.method_call_count("potential_loads"), 1);
    assert_eq!(worker.method_call_count("generate"), 1);
    drop(outcome);
    wait_for_method_call_count(&router, "mark_free", 1, Duration::from_secs(2)).await;
}

#[tokio::test]
async fn route_and_connect_preflight_decode_threshold_accepts_tokens() {
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![route_response_new(1)]);
    let worker = RouterGuardClientForTesting::new(vec![], vec![1], vec![route_response_new(1)]);
    let next = RouterGuardClientForTesting::new(
        vec![9],
        vec![9],
        vec![potential_loads_response(1, 2, 0, 0)],
    );
    let context = build_test_context("test-preflight-decode-token-threshold");

    let preflight_inputs = PreflightInputs {
        check: PotentialLoadsCheckData {
            router: next.clone(),
            queue_depth_threshold: 0,
            prefill_tokens_threshold: 100,
            // 33 tokens at block size 32 ceil-converts to 2 blocks, so a
            // reported load of 2 decode blocks is within threshold.
            decode_tokens_threshold: TEST_BLOCK_SIZE as usize + 1,
            load_percentile: 0.5,
        },
        tokens: vec![1],
        block_mm_infos: None,
    };

    let outcome = connect(
        router.clone(),
        worker.clone(),
        make_routing_request(),
        "req-pf-decode-token-threshold",
        context,
        Vec::new(),
        Some(preflight_inputs),
        make_worker_request(),
        0,
        true,
        true,
        Duration::from_secs(60),
    )
    .await
    .expect("ok");

    match outcome {
        RouteAndConnectOutcome::Connected { worker_id, .. } => assert_eq!(worker_id, 1),
        other => panic!("expected Connected, got {:?}", other),
    }

    assert_eq!(next.method_call_count("potential_loads"), 1);
    assert_eq!(worker.method_call_count("generate"), 1);
    drop(outcome);
    wait_for_method_call_count(&router, "mark_free", 1, Duration::from_secs(2)).await;
}

#[tokio::test]
async fn route_and_connect_preflight_thresholds_zero_disables_preflight() {
    // All thresholds zero disables every dimension -- even though the
    // preflight reports heavy load. The preflight returns Ok(None), the
    // route succeeds, and the worker connect happens -> Connected{1}.
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![route_response_new(1)]);
    let worker = RouterGuardClientForTesting::new(vec![], vec![1], vec![route_response_new(1)]);
    let next = RouterGuardClientForTesting::new(
        vec![9],
        vec![9],
        vec![potential_loads_response(
            usize::MAX / 4,
            1_000,
            usize::MAX / 4,
            5_000,
        )],
    );
    let context = build_test_context("test-preflight-zero");

    let preflight_inputs = PreflightInputs {
        check: PotentialLoadsCheckData {
            router: next.clone(),
            queue_depth_threshold: 0,
            prefill_tokens_threshold: 0,
            decode_tokens_threshold: 0,
            load_percentile: 0.5,
        },
        tokens: vec![1],
        block_mm_infos: None,
    };

    let outcome = connect(
        router.clone(),
        worker.clone(),
        make_routing_request(),
        "req-pf-zero",
        context,
        Vec::new(),
        Some(preflight_inputs),
        make_worker_request(),
        0,
        true,
        true,
        Duration::from_secs(60),
    )
    .await
    .expect("ok");

    match outcome {
        RouteAndConnectOutcome::Connected { worker_id, .. } => assert_eq!(worker_id, 1),
        other => panic!("expected Connected, got {:?}", other),
    }

    // Preflight DID run (so the disabled-threshold path is covered);
    // the route succeeded; the worker's direct opened.
    assert_eq!(next.method_call_count("potential_loads"), 1);
    assert_eq!(worker.method_call_count("generate"), 1);
    drop(outcome);
    wait_for_method_call_count(&router, "mark_free", 1, Duration::from_secs(2)).await;
}

#[tokio::test(start_paused = true)]
async fn route_and_connect_preflight_follows_cancellable_routing_policy() {
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![route_response_new(1)]);
    let worker = RouterGuardClientForTesting::new(vec![], vec![1], vec![route_response_new(1)]);
    let next = RouterGuardClientForTesting::new(
        vec![9],
        vec![9],
        vec![potential_loads_response(1, 1, 0, 0)],
    );
    next.set_respect_cancel(CancelRespect::Yes);
    let context = build_test_context("test-preflight-cancellable-routing");
    context.inner().stop_generating();

    let preflight_inputs = PreflightInputs {
        check: PotentialLoadsCheckData {
            router: next.clone(),
            queue_depth_threshold: 100,
            prefill_tokens_threshold: 100,
            decode_tokens_threshold: 100,
            load_percentile: 0.5,
        },
        tokens: vec![1],
        block_mm_infos: None,
    };

    let outcome = connect(
        router.clone(),
        worker.clone(),
        make_routing_request(),
        "req-pf-cancellable-routing",
        context,
        Vec::new(),
        Some(preflight_inputs),
        make_worker_request(),
        0,
        true,
        true,
        Duration::from_secs(60),
    )
    .await
    .expect("denied, not raised");

    assert!(matches!(
        outcome,
        RouteAndConnectOutcome::Denied(DeniedRequest::Cancelled())
    ));

    assert_eq!(next.calls().len(), 0);
    assert_eq!(next.method_call_count("potential_loads"), 0);
    assert_eq!(router.method_call_count("new"), 0);
    assert_eq!(worker.method_call_count("generate"), 0);
    let preflight_contexts = next.route_contexts();
    assert!(preflight_contexts.is_empty());
}

#[tokio::test]
async fn route_and_connect_preflight_detaches_when_routing_cancellation_disabled() {
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![route_response_new(1)]);
    router.set_respect_cancel(CancelRespect::Yes);
    let worker = RouterGuardClientForTesting::new(vec![], vec![1], vec![route_response_new(1)]);
    let next = RouterGuardClientForTesting::new(
        vec![9],
        vec![9],
        vec![potential_loads_response(1, 1, 0, 0)],
    );
    next.set_respect_cancel(CancelRespect::Yes);
    let context = build_test_context("test-preflight-detached-routing");
    context.inner().stop_generating();

    let preflight_inputs = PreflightInputs {
        check: PotentialLoadsCheckData {
            router: next.clone(),
            queue_depth_threshold: 100,
            prefill_tokens_threshold: 100,
            decode_tokens_threshold: 100,
            load_percentile: 0.5,
        },
        tokens: vec![1],
        block_mm_infos: None,
    };

    let outcome = connect(
        router.clone(),
        worker.clone(),
        make_routing_request(),
        "req-pf-detached-routing",
        context,
        Vec::new(),
        Some(preflight_inputs),
        make_worker_request(),
        0,
        false,
        false,
        Duration::from_secs(60),
    )
    .await
    .expect("ok");

    match &outcome {
        RouteAndConnectOutcome::Connected { worker_id, .. } => assert_eq!(*worker_id, 1),
        other => panic!("expected Connected, got {:?}", other),
    }

    assert_eq!(next.method_call_count("potential_loads"), 1);
    assert_eq!(router.method_call_count("new"), 1);
    assert_eq!(worker.method_call_count("generate"), 1);
    let preflight_contexts = next.route_contexts();
    assert_eq!(preflight_contexts.len(), 1);
    assert!(!preflight_contexts[0].is_stopped());
    assert!(!preflight_contexts[0].is_killed());
    let route_contexts = router.route_contexts();
    assert_eq!(route_contexts.len(), 1);
    assert!(!route_contexts[0].is_stopped());
    assert!(!route_contexts[0].is_killed());

    drop(outcome);
    wait_for_method_call_count(&router, "mark_free", 1, Duration::from_secs(2)).await;
}

#[tokio::test]
async fn route_and_connect_preflight_unreachable_returns_next_router_unreachable() {
    // Preflight router fake has NO available instances -> query_potential_loads
    // bails with "no router instances available...". Because preflight is
    // sequential, the route request is never sent.
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![route_response_new(1)]);
    let worker = RouterGuardClientForTesting::new(vec![], vec![1], vec![route_response_new(1)]);
    let next = RouterGuardClientForTesting::new(vec![], vec![], vec![]);
    let context = build_test_context("test-pf-unreachable");

    let preflight_inputs = PreflightInputs {
        check: PotentialLoadsCheckData {
            router: next.clone(),
            queue_depth_threshold: 0,
            prefill_tokens_threshold: 1_000_000,
            decode_tokens_threshold: 16_000_000,
            load_percentile: 0.5,
        },
        tokens: vec![1],
        block_mm_infos: None,
    };

    let outcome = connect(
        router.clone(),
        worker.clone(),
        make_routing_request(),
        "req-pf-unreachable",
        context,
        Vec::new(),
        Some(preflight_inputs),
        make_worker_request(),
        0,
        true,
        true,
        Duration::from_secs(60),
    )
    .await
    .expect("denied, not raised");

    match outcome {
        RouteAndConnectOutcome::Denied(DeniedRequest::NextRouterUnreachable { error }) => {
            assert!(
                error.contains("no router instances available"),
                "error: {}",
                error
            );
        }
        other => panic!("expected Denied(NextRouterUnreachable), got {:?}", other),
    }

    // Preflight was attempted but never made a direct() call (empty
    // instance set bails before the loop); route and worker were never called.
    assert_eq!(next.method_call_count("potential_loads"), 0);
    assert_eq!(router.method_call_count("new"), 0);
    assert_eq!(router.method_call_count("mark_free"), 0);
    assert_eq!(worker.method_call_count("generate"), 0);
}

#[tokio::test(start_paused = true)]
async fn route_and_connect_cancellable_routing_checks_stopped_context_before_direct() {
    // Stop the parent context BEFORE awaiting route_and_connect. Cancellable
    // routing now checks the parent context before making a router direct()
    // call, so cancellation is classified as Cancelled instead of as a router
    // reachability failure.
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![route_response_new(1)]);
    router.set_respect_cancel(CancelRespect::Yes);
    let worker = RouterGuardClientForTesting::new(vec![], vec![1], vec![route_response_new(1)]);
    let context = build_test_context("test-routing-cancel");
    context.inner().stop_generating();

    let outcome = connect(
        router.clone(),
        worker.clone(),
        make_routing_request(),
        "req-routing-cancel",
        context,
        Vec::new(),
        None,
        make_worker_request(),
        0,
        true,
        true,
        Duration::from_secs(60),
    )
    .await
    .expect("denied, not raised");

    assert!(matches!(
        outcome,
        RouteAndConnectOutcome::Denied(DeniedRequest::Cancelled())
    ));

    // The policy-gated boundary check rejects before calling the router.
    assert_eq!(router.calls().len(), 0);
    assert_eq!(router.completed_direct_count(), 0);
    assert_eq!(router.method_call_count("mark_free"), 0);
    assert_eq!(worker.method_call_count("generate"), 0);
    let route_contexts = router.route_contexts();
    assert!(route_contexts.is_empty());
}

#[tokio::test(start_paused = true)]
async fn route_and_connect_post_admit_stream_eof_fires_mark_free_then_denies_unreachable() {
    // Race Site 12 regression: `router.direct()` returns Ok(stream) (the
    // router ADMITTED the request internally), but the stream ends before
    // yielding any data -- so `first_stream_response` returns
    // `Err("router response stream ended before data")`. The provisional
    // guard requests `mark_free`, kills the detached route context, and waits
    // for cleanup before the retry can reuse the same request id. Both
    // ROUTER_GUARD_ATTEMPTS=2 attempts hit this path; the loop exhausts and
    // `route_request` returns Err, which `route_once` (no preflight, no
    // require_during) maps to `Denied::NextRouterUnreachable`.
    //
    // Asserts `mark_free == 2` -- one per attempt's cleanup task. This is
    // the post-admit error path (Race Site 12); the pre-admit error path
    // (`router.direct()` returns Err) still `dismiss`es with `mark_free == 0`
    // -- see `route_and_connect_routing_cancelled_in_band_returns_denied_next_router_unreachable`.
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], Vec::new());
    router.set_mark_free_callback_delay(Duration::from_millis(250));
    // Empty chunk-Vec per direct() -> stream::iter(vec![]) -> first
    // stream.next() is None -> first_stream_response errors. One entry
    // per direct() so the chunks_queue survives both attempts.
    router.set_stream_chunks(vec![Vec::new(), Vec::new()]);
    let worker = RouterGuardClientForTesting::new(vec![], vec![1], vec![route_response_new(1)]);
    let context = build_test_context("test-post-admit-eof");

    let outcome = connect(
        router.clone(),
        worker.clone(),
        make_routing_request(),
        "req-post-admit-eof",
        context,
        Vec::new(),
        None,
        make_worker_request(),
        0,
        true,
        true,
        Duration::from_secs(60),
    )
    .await
    .expect("denied, not raised");

    match outcome {
        RouteAndConnectOutcome::Denied(DeniedRequest::NextRouterUnreachable { error }) => {
            assert!(
                error.contains("router response stream ended before data"),
                "error: {}",
                error
            );
        }
        other => panic!("expected Denied(NextRouterUnreachable), got {:?}", other),
    }

    // Both attempts: direct() returned Ok with an empty stream (completed),
    // the route context was killed, and mark_free completed before the next
    // route attempt reused the same request id.
    wait_for_method_call_count(&router, "mark_free", 2, Duration::from_secs(2)).await;
    assert_eq!(router.completed_direct_count(), 4);
    assert_eq!(router.method_call_count("mark_free"), 2);
    assert_eq!(worker.method_call_count("generate"), 0);
    let methods: Vec<String> = router
        .detailed_calls()
        .iter()
        .map(|c| c.method.clone())
        .collect();
    assert_eq!(
        methods,
        vec![
            "new".to_string(),
            "mark_free".to_string(),
            "new".to_string(),
            "mark_free".to_string()
        ]
    );
    let route_contexts = router.route_contexts();
    assert_eq!(route_contexts.len(), 2);
    assert!(route_contexts.iter().all(|ctx| ctx.is_killed()));
}

#[tokio::test]
async fn route_and_connect_unexpected_response_variant_fires_mark_free_then_denies_protocol_error()
{
    // Medium regression: `router.direct()` returns Ok(stream) and the first
    // stream item decodes successfully -- but the variant is NOT a clean
    // admit (`New`) or clean denial (`Backpressure`). Instead the router
    // returns `FreeMarked` (which is the reply shape for a `mark_free`
    // request, not a `new` request). Admission state is ambiguous -- the
    // router's contract is broken. `route_request` fails closed: requests
    // mark_free, drops the provisional guard, returns an unarmed placeholder
    // guard with
    // `RouteSource::ProtocolError { received }`. `route_once` maps that to
    // `Denied::ProtocolError { received }`. Mirrors the stricter
    // `potential_loads` handling (`PotentialLoadsError::ProtocolError`
    // -> `DeniedRequest::ProtocolError`).
    //
    // Only the FIRST attempt's instance hits this path: route_request
    // `return`s immediately on the unexpected variant, so only one
    // mark_free fires.
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![free_marked_response()]);
    let worker = RouterGuardClientForTesting::new(vec![], vec![1], vec![route_response_new(1)]);
    let context = build_test_context("test-unexpected-variant");

    let outcome = connect(
        router.clone(),
        worker.clone(),
        make_routing_request(),
        "req-unexpected-variant",
        context,
        Vec::new(),
        None,
        make_worker_request(),
        0,
        true,
        true,
        Duration::from_secs(60),
    )
    .await
    .expect("denied, not raised");

    match outcome {
        RouteAndConnectOutcome::Denied(DeniedRequest::ProtocolError { received }) => {
            assert!(
                received.contains("free_marked"),
                "received should contain the variant name (snake_case serde tag): {}",
                received
            );
        }
        other => panic!("expected Denied(ProtocolError), got {:?}", other),
    }

    // One route direct() call (method="new", completed) plus one mark_free
    // callback (method="mark_free", completed) = 2 entries with
    // `completed=true`. The provisional guard requested mark_free before
    // drop; the placeholder guard is unarmed so its drop is a no-op (no
    // second mark_free).
    wait_for_method_call_count(&router, "mark_free", 1, Duration::from_secs(2)).await;
    assert_eq!(router.completed_direct_count(), 2);
    assert_eq!(router.method_call_count("mark_free"), 1);
    assert_eq!(worker.method_call_count("generate"), 0);
}

#[tokio::test]
async fn route_and_connect_stream_cancelled_in_band_truncates_stream() {
    // Multi-chunk worker stream: chunks [c1, c2, c3]. take_while forwards
    // while the linked request context is live; once the parent context
    // is stopped, the child's stop_generating propagates, the next
    // take_while poll yields false, the stream ends.
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![route_response_new(1)]);
    let worker = RouterGuardClientForTesting::new(vec![], vec![1], Vec::new());
    worker.set_stream_chunks(vec![
        vec![jv!({"chunk": 1})],
        vec![jv!({"chunk": 2})],
        vec![jv!({"chunk": 3})],
    ]);
    let context = build_test_context("test-stream-cancel");

    let outcome = connect(
        router.clone(),
        worker.clone(),
        make_routing_request(),
        "req-stream-cancel",
        context.clone(),
        Vec::new(),
        None,
        make_worker_request(),
        0,
        true,
        true,
        Duration::from_secs(60),
    )
    .await
    .expect("ok");

    let (mut stream, guard, _worker_id) = match outcome {
        RouteAndConnectOutcome::Connected {
            stream,
            guard,
            worker_id,
            ..
        } => (stream, guard, worker_id),
        other => panic!("expected Connected, got {:?}", other),
    };

    // Pull the first chunk before stopping -- take_while's predicate
    // returns true so chunk 1 forwards.
    let first = take_one_from_stream(&mut stream).await;
    assert!(first.is_some(), "first chunk should arrive");
    assert_eq!(first.unwrap().data, Some(jv!({"chunk": 1})));

    // Stop the parent now; the linked child's controller propagates
    // stop_generating synchronously, so the next take_while poll
    // returns false -> stream ends.
    context.inner().stop_generating();
    let next = tokio::time::timeout(
        Duration::from_millis(100),
        take_one_from_stream(&mut stream),
    )
    .await
    .expect("next poll returns promptly");
    assert!(next.is_none(), "stream should truncate after stop");

    // Guard drop cleanup fires mark_free on the router fake.
    drop(stream);
    drop(guard);
    wait_for_method_call_count(&router, "mark_free", 1, Duration::from_secs(2)).await;
}

#[tokio::test]
async fn route_and_connect_lifecycle_mark_prefill_then_mark_free_order() {
    // Drive the lifecycle explicitly: route succeeds (mark "new"),
    // the test calls mark_prefill (mark "mark_prefill"), then mark_free
    // (mark "mark_free"). On drop there is NO second mark_free because
    // the cleanup task transitioned to cleanup_done=true after the
    // explicit mark_free. The router fake's detailed_calls preserves
    // the call order so the test asserts the exact lifecycle sequence.
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![route_response_new(1)]);
    let worker = RouterGuardClientForTesting::new(vec![], vec![1], vec![route_response_new(1)]);
    let context = build_test_context("test-lifecycle");

    let outcome = connect(
        router.clone(),
        worker.clone(),
        make_routing_request(),
        "req-lifecycle",
        context,
        Vec::new(),
        None,
        make_worker_request(),
        0,
        true,
        true,
        Duration::from_secs(60),
    )
    .await
    .expect("ok");

    let (guard, _worker_id, _stream) = match outcome {
        RouteAndConnectOutcome::Connected {
            guard,
            worker_id,
            stream,
            ..
        } => (guard, worker_id, stream),
        other => panic!("expected Connected, got {:?}", other),
    };

    // mark_prefill -> cleanup task wakes, sends mark_prefill. Order=[new, mark_prefill].
    guard.mark_prefill();
    wait_for_method_call_count(&router, "mark_prefill", 1, Duration::from_secs(2)).await;

    // mark_free -> cleanup task wakes, sends mark_free, sets cleanup_done.
    guard.mark_free();
    wait_for_method_call_count(&router, "mark_free", 1, Duration::from_secs(2)).await;

    // Now drop the guard -- Drop sees cleanup_done=true and is a no-op:
    // no second mark_free. Sleep a beat so a stray mark_free would
    // have arrived if it were ever going to fire.
    drop(guard);
    tokio::time::sleep(Duration::from_millis(50)).await;

    let methods: Vec<String> = router
        .detailed_calls()
        .iter()
        .map(|c| c.method.clone())
        .collect();
    assert_eq!(
        methods,
        vec![
            "new".to_string(),
            "mark_prefill".to_string(),
            "mark_free".to_string()
        ],
        "lifecycle method order, got {:?}",
        methods
    );
}

#[tokio::test(start_paused = true)]
async fn route_and_connect_slow_mark_free_callback_still_completes_once() {
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![route_response_new(1)]);
    router.set_mark_free_callback_delay(
        ROUTER_GUARD_CLEANUP_GRACE_PERIOD + Duration::from_millis(100),
    );
    let worker = RouterGuardClientForTesting::new(vec![], vec![1], vec![route_response_new(1)]);
    let context = build_test_context("test-slow-mark-free");

    let outcome = connect(
        router.clone(),
        worker.clone(),
        make_routing_request(),
        "req-slow-mark-free",
        context,
        Vec::new(),
        None,
        make_worker_request(),
        0,
        true,
        true,
        Duration::from_secs(60),
    )
    .await
    .expect("ok");

    let (guard, _worker_id, _stream) = match outcome {
        RouteAndConnectOutcome::Connected {
            guard,
            worker_id,
            stream,
            ..
        } => (guard, worker_id, stream),
        other => panic!("expected Connected, got {:?}", other),
    };

    guard.mark_free();
    wait_for_method_call_count(&router, "mark_free", 1, Duration::from_secs(2)).await;
    drop(guard);
    tokio::time::sleep(Duration::from_millis(10)).await;

    let methods: Vec<String> = router
        .detailed_calls()
        .iter()
        .map(|c| c.method.clone())
        .collect();
    assert_eq!(methods, vec!["new".to_string(), "mark_free".to_string()]);
    assert_eq!(router.method_call_count("mark_free"), 1);
}

#[tokio::test]
async fn route_and_connect_mark_free_preempts_in_flight_mark_prefill() {
    // A slow/stuck mark_prefill callback must not delay freeing the router slot.
    // The cleanup task should abandon the prefill future as soon as mark_free is
    // requested and send mark_free immediately.
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![route_response_new(1)]);
    router.set_prefill_callback_delay(Duration::from_secs(5));
    let worker = RouterGuardClientForTesting::new(vec![], vec![1], vec![route_response_new(1)]);
    let context = build_test_context("test-prefill-preempted-by-free");

    let outcome = connect(
        router.clone(),
        worker.clone(),
        make_routing_request(),
        "req-prefill-preempted-by-free",
        context,
        Vec::new(),
        None,
        make_worker_request(),
        0,
        true,
        true,
        Duration::from_secs(60),
    )
    .await
    .expect("ok");

    let (guard, _worker_id, _stream) = match outcome {
        RouteAndConnectOutcome::Connected {
            guard,
            worker_id,
            stream,
            ..
        } => (guard, worker_id, stream),
        other => panic!("expected Connected, got {:?}", other),
    };

    guard.mark_prefill();
    wait_for_call_count(&router, 2).await;
    assert_eq!(router.calls()[1].1["method"].as_str(), Some("mark_prefill"));

    guard.mark_free();
    wait_for_method_call_count(&router, "mark_free", 1, Duration::from_millis(250)).await;
    drop(guard);
    tokio::time::sleep(Duration::from_millis(50)).await;

    let completed_methods: Vec<String> = router
        .detailed_calls()
        .iter()
        .map(|c| c.method.clone())
        .collect();
    assert_eq!(
        completed_methods,
        vec!["new".to_string(), "mark_free".to_string()],
        "completed lifecycle methods, got {:?}",
        completed_methods
    );
    assert_eq!(router.method_call_count("mark_prefill"), 0);
    assert_eq!(router.method_call_count("mark_free"), 1);
}

#[tokio::test]
async fn route_and_connect_detached_setup_completes_after_outer_abort() {
    // allow_cancel_setup=false -> connect_worker wraps the worker open
    // in shield_to_completion: a tokio::spawn runs the worker's direct()
    // to completion regardless of the outer task's lifetime. The outer
    // await is on the oneshot Receiver; when we abort the outer task
    // the Receiver is dropped, but the spawned open_fut keeps running
    // (its 200ms open_delay) and then completes -- pushing to
    // detailed_calls with completed=true. Meanwhile the guard (still
    // in connect_worker's frame) is dropped along with the outer task,
    // its Drop fires mark_free on the router fake.
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![route_response_new(1)]);
    let worker = RouterGuardClientForTesting::new(vec![], vec![1], vec![route_response_new(1)]);
    worker.set_open_delay(Duration::from_millis(200));
    let context = build_test_context("test-detached-setup");

    let router_for_task = router.clone();
    let worker_for_task = worker.clone();
    let routing_request = make_routing_request();
    let worker_request = make_worker_request();
    let context_for_task = context.clone();
    let task = tokio::spawn(async move {
        connect(
            router_for_task,
            worker_for_task,
            routing_request,
            "req-detached-setup",
            context_for_task,
            Vec::new(),
            None,
            worker_request,
            0,
            true,
            false,
            Duration::from_secs(60),
        )
        .await
    });

    // Wait for the route direct to fire (router fake open_delay=0), so
    // the guard is armed and connect_worker has entered the shielded
    // open. The 50ms sleep is well within the worker's 200ms open_delay
    // so the outer await is still pending.
    wait_for_method_call_count(&router, "new", 1, Duration::from_secs(2)).await;
    tokio::time::sleep(Duration::from_millis(50)).await;

    task.abort();
    let _ = task.await;

    // Shielded inner continues to completion DESPITE the abort -- the
    // worker's direct() finishes and pushes detailed_calls{completed:true}.
    wait_for_completion_count(&worker, 1, Duration::from_secs(2)).await;
    assert_eq!(worker.method_call_count("generate"), 1);
    assert_eq!(worker.completed_direct_count(), 1);

    // Guard drop fired (synchronously with the outer task drop) and the
    // always-detached cleanup task then sent mark_free on the router
    // fake. The worker's generation stream was wrapped by the shielded
    // inner's oneshot Sender; once the inner finishes the stream is
    // dropped (no consumer) and the guard is already gone.
    wait_for_method_call_count(&router, "mark_free", 1, Duration::from_secs(2)).await;
    assert_eq!(router.method_call_count("mark_free"), 1);
}

#[tokio::test]
async fn shield_route_and_connect_no_taker_drains_connected_worker_stream() {
    // Regression for the route/setup -> stream handoff: when the Python
    // awaitable is cancelled while the shielded route_and_connect loop is still
    // pending, the oneshot receiver is dropped. If the background loop later
    // returns Connected, the no-taker path must consume the worker stream
    // instead of dropping it at the handoff boundary.
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![route_response_new(1)]);
    let worker = RouterGuardClientForTesting::new(vec![], vec![1], Vec::new());
    worker.set_open_delay(Duration::from_millis(200));
    worker.set_stream_chunks(vec![vec![
        jv!({"chunk": 1}),
        jv!({"chunk": 2}),
        jv!({"chunk": 3}),
    ]]);
    let context = build_test_context("test-shield-route-no-taker-drain");

    let route_fut = route_and_connect(
        router.clone() as Arc<dyn RouterGuardClient>,
        worker.clone() as Arc<dyn RouterGuardClient>,
        make_routing_request(),
        "req-shield-route-no-taker".to_string(),
        context,
        Vec::new(),
        None,
        make_worker_request(),
        TEST_BLOCK_SIZE,
        0,
        false,
        false,
        false,
        Duration::from_secs(60),
        false,
        None,
    );
    let task = tokio::spawn(async move { shield_route_and_connect(route_fut).await });

    wait_for_method_call_count(&router, "new", 1, Duration::from_secs(2)).await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    task.abort();
    let _ = task.await;

    wait_for_completion_count(&worker, 1, Duration::from_secs(2)).await;
    wait_for_stream_items_polled(&worker, 3, Duration::from_secs(2)).await;
    wait_for_method_call_count(&router, "mark_free", 1, Duration::from_secs(2)).await;
    assert_eq!(worker.stream_items_polled_count(), 3);
    assert_eq!(router.method_call_count("mark_free"), 1);
}

#[tokio::test]
async fn route_and_connect_cancellable_setup_drops_on_outer_abort() {
    // The contrast against the detached case: allow_cancel_setup=true
    // inlines the open await in connect_worker. Aborting the outer task
    // drops the inline `open_fut.await`, which drops the worker's
    // direct() future mid-`tokio::time::sleep(open_delay)` -- the sleep
    // is cancelled and the post-sleep `detailed_calls` push never runs.
    // The legacy `calls` log still shows the attempt (recorded at
    // direct() entry); the detailed log does NOT show a completion.
    // Guard drop fires mark_free as in the detached case.
    let router = RouterGuardClientForTesting::new(vec![7], vec![7], vec![route_response_new(1)]);
    let worker = RouterGuardClientForTesting::new(vec![], vec![1], vec![route_response_new(1)]);
    worker.set_open_delay(Duration::from_millis(200));
    let context = build_test_context("test-cancellable-setup");

    let router_for_task = router.clone();
    let worker_for_task = worker.clone();
    let routing_request = make_routing_request();
    let worker_request = make_worker_request();
    let context_for_task = context.clone();
    let task = tokio::spawn(async move {
        connect(
            router_for_task,
            worker_for_task,
            routing_request,
            "req-cancellable-setup",
            context_for_task,
            Vec::new(),
            None,
            worker_request,
            0,
            true,
            true,
            Duration::from_secs(60),
        )
        .await
    });

    wait_for_method_call_count(&router, "new", 1, Duration::from_secs(2)).await;
    tokio::time::sleep(Duration::from_millis(50)).await;

    task.abort();
    let _ = task.await;

    // Wait well past the 200ms open_delay: if the inlined open had
    // continued (which it must NOT) we'd see a completion now. The
    // direct() call entry was logged in legacy `calls` (1 entry for
    // generate) but detailed_calls stays empty.
    tokio::time::sleep(Duration::from_millis(250)).await;
    assert_eq!(worker.completed_direct_count(), 0);
    assert_eq!(worker.method_call_count("generate"), 0);
    assert_eq!(worker.calls().len(), 1);

    // Guard drop -> mark_free fires on the router fake regardless of
    // setup cancellation mode.
    wait_for_method_call_count(&router, "mark_free", 1, Duration::from_secs(2)).await;
    assert_eq!(router.method_call_count("mark_free"), 1);
}

/// The routing wire used to be built by round-tripping through `serde_json`.
/// It now serializes straight into `rmpv`, so pin the two against each other:
/// the request plane is shared with older peers and the bytes must not move.
#[test]
fn router_request_new_matches_legacy_json_roundtrip() {
    let cases = vec![
        RouterRequestNew::default(),
        RouterRequestNew {
            tokens: vec![0, 1, 127, 128, 255, 256, 65_535, 65_536, 151_643],
            priority_jump: 1.5,
            priority_load_shed_percent: 42,
            do_not_queue: true,
            ..Default::default()
        },
    ];

    for req in cases {
        let legacy: rmpv::Value = serde_json::from_value(
            serde_json::to_value(RouterRequest::from(req.clone())).expect("to_value"),
        )
        .expect("from_value");
        let direct = req
            .into_routing_request_value()
            .expect("direct rmpv conversion");

        assert_eq!(direct, legacy, "rmpv wire value changed");
    }
}
