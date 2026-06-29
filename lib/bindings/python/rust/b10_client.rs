//! Router-worker coordinator: route a KV-router `new` request, then generate
//! on the routed worker with detached guard cleanup on cancellation.
//!
//! The module is split across four submodules:
//! - [`types`] holds the PyO3 pyclasses (`PyRouterRequestNew`,
//!   `CancellationPolicy`, `DeniedRequest`, `AdmittedRequest`,
//!   `RouterCoordinatorPotentialLoadsCheck`) plus the
//!   cross-module wire structs (`RouterRequestNew`, `PreflightInputs`,
//!   `MinReplicaAvailable`, `PotentialLoadsCheckData`,
//!   `NextRouterBackpressureInfo`).
//! - [`guard`] holds the per-request lifecycle guard (`mark_prefill` /
//!   `mark_free`, the detached cleanup task on drop) plus the
//!   `ROUTER_GUARD_*` timeouts.
//! - [`coordinator`] is the algorithmic core: the `RouterGuardClient` trait
//!   and `JsonRouterGuardClient` adapter, the `route_request` /
//!   `route_and_connect` loop, the `route_once` / `connect_worker` helpers,
//!   the preflight `query_potential_loads` / `evaluate_potential_loads`, and
//!   the shielded-phase drivers (`shield_to_completion`,
//!   `shield_route_and_connect`, `shield_stream_to_completion`).
//! - [`tests`] is the `#[cfg(test)]` suite covering the legacy `route_request`
//!   and full `route_and_connect` lifecycle (the in-file fake and sync
//!   `RouterRequestNew` round-trip tests).
//!
//! The root file declares the `RouterWorkerCoordinator` pyclass and its
//! `route_and_worker` shim: it borrows the user's `PyRouterRequestNew`
//! under the GIL, lifts the routing knobs and preflight inputs into
//! plain `Send` values, then drives the `coordinator::route_and_connect`
//! loop with a per-phase shield selected by the caller's `CancellationPolicy`.
mod coordinator;
mod guard;
mod types;

#[cfg(test)]
mod tests;

// Re-export the pyclasses registered in `lib.rs::add_class::<...>` so they
// resolve as `crate::b10_client::Foo` (the crate-root path lib.rs expects).
// Pyclasses are `pub(crate)` in `types` (and `pub(crate)` here on
// `RouterWorkerCoordinator`); glob re-export would miss them.
pub(crate) use types::{
    AdmittedRequest, CancellationPolicy, DeniedRequest, PyRouterRequestNew,
    RouterCoordinatorPotentialLoadsCheck,
};

use crate::llm::local_model::RoutingConstraints as PyRoutingConstraints;
use crate::{AsyncResponseStream, Client, context, process_stream, to_pyerr};
use dynamo_kv_router::protocols::{BlockExtraInfo, RoutingConstraints};
use dynamo_runtime::pipeline::{EngineStream, ResponseStream};
use dynamo_runtime::protocols::annotated::Annotated as RsAnnotated;
use futures::StreamExt;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;

use coordinator::{
    JsonRouterGuardClient, RouteAndConnectOutcome, RouteSource, RouterGuardClient,
    route_and_connect, shield_route_and_connect, shield_stream_to_completion,
    should_drop_first_worker_event,
};
use guard::{ROUTER_GUARD_NOTIFY_TIMEOUT, RouterRequestGuard};
use types::{MinReplicaAvailable, PotentialLoadsCheckData, PreflightInputs, RouterRequestNew};

pub(crate) const DROP_THIS_MESSAGE_KEY: &str = "drop_this_message";

fn stream_with_optional_prefill_mark(
    stream: EngineStream<RsAnnotated<serde_json::Value>>,
    guard: Arc<RouterRequestGuard>,
    mark_prefill_on_response: bool,
) -> EngineStream<RsAnnotated<serde_json::Value>> {
    if !mark_prefill_on_response {
        return stream;
    }

    let stream_context = stream.context();
    let mut marked = false;
    let stream = stream.map(move |response| {
        if !marked
            && !response.is_error()
            && response.data.is_some()
            && !should_drop_first_worker_event(&response)
        {
            guard.mark_prefill();
            marked = true;
        }
        response
    });
    ResponseStream::new(Box::pin(stream), stream_context)
}

fn attach_worker_stream_to_parent_context(
    stream: &EngineStream<RsAnnotated<serde_json::Value>>,
    parent: &context::Context,
) {
    let parent_inner = parent.inner();
    let stream_context = stream.context();
    parent_inner.link_child(stream_context.clone());
    if parent_inner.is_killed() {
        stream_context.kill_with_reason(Some("parent_context_already_killed"));
    } else if parent_inner.is_stopped() {
        stream_context.stop_generating_with_reason(Some("parent_context_already_stopped"));
    }
}

#[pyclass]
pub(crate) struct RouterWorkerCoordinator {
    router: Client,
    worker: Client,
    block_size: u32,
}

#[pymethods]
impl RouterWorkerCoordinator {
    /// Build a coordinator that routes via `router_client` and generates via
    /// `worker_client`. Both are existing `Client` instances. `block_size` must
    /// match the KV router's block size so per-request overlap blocks can be
    /// interpreted consistently.
    #[new]
    #[pyo3(signature = (router_client, worker_client, block_size=32))]
    fn new(router_client: Client, worker_client: Client, block_size: u32) -> PyResult<Self> {
        if block_size == 0 {
            return Err(PyValueError::new_err("block_size must be positive"));
        }

        Ok(Self {
            router: router_client,
            worker: worker_client,
            block_size,
        })
    }

    /// KV router block size in tokens.
    fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Route a KV-router `new` request, then generate on the routed worker.
    ///
    /// `routing_kwargs` is a [`PyRouterRequestNew`] pyclass -- the REQUIRED,
    /// single source of truth for the six `RouterRequest::New` wire-body fields
    /// (`tokens`, `block_mm_infos`, `routing_constraints`, `priority_jump`,
    /// `priority_load_shed_percent`, `do_not_queue`). The first-class `tokens`
    /// and `block_mm_infos` arguments are GONE; both live on the pyclass.
    /// `worker_args` is the body sent to the worker for generation; on a
    /// successful route the decoded `RouterResponse::New` is added to it under
    /// the `router_response` field — carrying `worker_id`, `overlap_blocks`
    /// (potential cache hit), and `dp_rank` / `dp_strict_rank` (the dp-rank
    /// instruction for the worker) — so the worker (or a further forwarder)
    /// receives the routing decision.
    ///
    /// `cancellation` (a [`CancellationPolicy`], default `Cancellable`) selects
    /// which of three phases — the `route_and_connect` loop (`routing`), each
    /// per-attempt `direct()` worker-stream open (`setup`), and the worker
    /// generation stream hand-back (`stream`) — may be aborted by a Python
    /// cancellation of this awaitable in-band via the linked request context.
    /// A detached phase runs to completion regardless; an armed guard produced
    /// by an abandoned phase is dropped (which fires the always-detached
    /// `mark_free` cleanup task) when the phase finishes, so the router is
    /// never orphaned. Cancellation is NEVER a tokio task-drop — it propagates
    /// through `context.is_stopped() || is_killed()` to the underlying
    /// `direct()` open and the worker stream, and the coordinator also checks
    /// the context at policy-cancellable phase boundaries. If cancellation wins
    /// after the router admitted a request, the guard requests `mark_free`
    /// before returning `DeniedRequest.Cancelled`. Routing and setup are
    /// INDEPENDENT axes; see [`CancellationPolicy`] for the full matrix.
    ///
    /// When `potential_loads_next_check` is given, a *potential loads* preflight
    /// queries the *downstream* `client` it carries (another router, e.g. the
    /// next router in a disagg-prefill topology -- distinct from the routing
    /// router) for worker potential loads before the route request is sent (on
    /// the first attempt only -- a stale-route reroute does not change the
    /// downstream router's loads) and denies the request when the configured
    /// load percentile exceeds the thresholds. This is deliberately sequential
    /// so a preflight denial does not leave a newly routed request to free. The
    /// preflight is part of the `routing` phase, so it is shielded when
    /// `cancellation.allow_cancel_routing()` is false.
    /// `tracing_enabled=true` emits route/preflight step breadcrumbs with whether
    /// a trace context is available; slow potential-load checks still warn
    /// regardless of this flag. When `wait_for_first_response=true`, worker
    /// setup waits for the first non-error item from the returned
    /// worker stream and drops it only when it carries the drop-message
    /// sentinel key `dynamo._core.B10_DROP_THIS_MESSAGE_KEY`. Otherwise,
    /// setup prepends the item back onto the returned stream. This lets a
    /// component emit a readiness event so the awaitable does not complete
    /// until the worker has produced data.
    /// When `mark_prefill_on_response=true`, Rust calls `mark_prefill()` on
    /// the routed guard as soon as the worker stream produces its first
    /// non-error, non-sentinel data item. Manual Python `mark_prefill()` calls
    /// remain supported; this just makes that call optional without waiting for
    /// Python to consume the stream.
    ///
    /// Returns a [`AdmittedRequest`] on a successful route (with the worker
    /// generation stream, lifecycle guard, setup timing/reroute accessors, and
    /// chosen `worker_id`) or a
    /// [`DeniedRequest`] when the router is backpressured, a `require_available`
    /// component is down, the preflight overflows, the preflight cannot reach
    /// the router, policy-allowed cancellation wins, the stale-route reroute
    /// loop is exhausted, or `wait_for_first_response` cannot read the first
    /// worker stream item — never raising in those cases. Discriminate in Python with
    /// `isinstance(result, AdmittedRequest)` / `isinstance(result, DeniedRequest)`
    /// (and `isinstance(result, DeniedRequest.<Variant>)` for the denial reason);
    /// first-event failures are returned as
    /// `DeniedRequest.FirstWorkerEventFailed`, not raised.
    /// On a successful route, `response_stream()` yields the worker generation
    /// tokens, timing/reroute accessors report route/connect setup seconds and
    /// stale reroutes, and `mark_prefill()` / `mark_free()` drive the
    /// KV-lifecycle callbacks to the router. A non-stale worker-open failure (or a
    /// non-object `worker_args`) IS raised, not returned as a `DeniedRequest`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (context, routing_kwargs, worker_args=None, require_available=None, potential_loads_next_check=None, annotated=false, cancellation=CancellationPolicy::Cancellable, max_reroutes=1, tracing_enabled=false, wait_for_first_response=false, mark_prefill_on_response=false))]
    fn route_and_worker<'p>(
        &self,
        py: Python<'p>,
        context: context::Context,
        routing_kwargs: Py<PyRouterRequestNew>,
        worker_args: Option<PyObject>,
        require_available: Option<Vec<Client>>,
        potential_loads_next_check: Option<PyObject>,
        annotated: Option<bool>,
        cancellation: CancellationPolicy,
        max_reroutes: u64,
        tracing_enabled: bool,
        wait_for_first_response: bool,
        mark_prefill_on_response: bool,
    ) -> PyResult<Bound<'p, PyAny>> {
        let annotated = annotated.unwrap_or(false);
        let allow_cancel_routing = cancellation.allow_cancel_routing();
        let allow_cancel_setup = cancellation.allow_cancel_setup();
        let allow_cancel_stream = cancellation.allow_cancel_stream();

        // Extract the PyRouterRequestNew under the GIL; the async block runs
        // without it. `tokens`, `block_mm_infos`, and `routing_constraints` all
        // live on the pyclass now -- there is no `routing_kwargs` dict, no
        // first-class `tokens`, and no first-class `block_mm_infos` argument.
        let borrowed = routing_kwargs.bind(py).borrow();
        let tokens: Vec<u32> = borrowed.tokens.clone();
        let block_mm_infos_py: Option<PyObject> = borrowed.block_mm_infos.clone();
        let routing_constraints_py: Option<Py<PyRoutingConstraints>> =
            borrowed.routing_constraints.clone();
        let priority_jump: f64 = borrowed.priority_jump;
        let priority_load_shed_percent: u8 = borrowed.priority_load_shed_percent;
        let do_not_queue: bool = borrowed.do_not_queue;
        drop(borrowed);

        // `block_mm_infos` is held loosely as an `Optional[Any]` on the pyclass;
        // pythonize to a serde_json::Value and deserialize to the typed wire
        // form (same path as before), so the route (`New`) and the preflight
        // (`PotentialLoads`) share the same overlap-aware metadata.
        let block_mm_infos_typed: Option<Vec<Option<BlockExtraInfo>>> = match block_mm_infos_py {
            Some(mm) => {
                let value = pythonize::depythonize(&mm.into_bound(py))?;
                Some(serde_json::from_value(value).map_err(to_pyerr)?)
            }
            None => None,
        };

        // `routing_constraints` is the local-model pyclass
        // (`crate::llm::local_model::RoutingConstraints`, aliased here as
        // `PyRoutingConstraints`); `None` means the default (empty) wire
        // constraints. Convert via its bidirectional `From` impl.
        let routing_constraints_wire: RoutingConstraints = match routing_constraints_py {
            Some(rc) => {
                let rc_ref = rc.bind(py).borrow();
                let rc_cloned: PyRoutingConstraints = rc_ref.clone();
                RoutingConstraints::from(rc_cloned)
            }
            None => RoutingConstraints::default(),
        };

        // Extract the preflight check into a plain `Send` struct while we hold
        // the GIL. `router` is the downstream `client`'s router (the router the
        // preflight queries, distinct from the routing router). The
        // `block_mm_infos` conditioning the preflight comes from the pyclass
        // field (see above), not the check itself.
        let next_check: Option<PotentialLoadsCheckData> = match potential_loads_next_check {
            Some(obj) => {
                let bound = obj.into_bound(py);
                let check = bound
                    .downcast::<RouterCoordinatorPotentialLoadsCheck>()
                    .map_err(|_| {
                        PyTypeError::new_err(
                            "potential_loads_next_check must be a RouterCoordinatorPotentialLoadsCheck",
                        )
                    })?;
                let borrowed = check.borrow();
                Some(PotentialLoadsCheckData {
                    router: Arc::new(JsonRouterGuardClient::new(borrowed.client.router.clone())),
                    queue_depth_threshold: borrowed.queue_depth_threshold,
                    prefill_tokens_threshold: borrowed.prefill_tokens_threshold,
                    decode_blocks_threshold: borrowed.decode_blocks_threshold,
                    load_percentile: borrowed.load_percentile,
                })
            }
            None => None,
        };

        // The preflight needs the tokens too; clone now -- the routing request
        // below will move `tokens` into `RouterRequestNew`.
        let tokens_for_check: Option<Vec<u32>> = next_check.as_ref().map(|_| tokens.clone());

        let req = RouterRequestNew {
            tokens,
            block_mm_infos: block_mm_infos_typed.clone(),
            routing_constraints: routing_constraints_wire,
            priority_jump,
            priority_load_shed_percent,
            do_not_queue,
        };
        let routing_request = req.into_routing_request_value().map_err(to_pyerr)?;

        let worker_request: serde_json::Value = match worker_args {
            Some(wa) => pythonize::depythonize(&wa.into_bound(py))?,
            None => serde_json::Value::Object(Default::default()),
        };

        let require: Vec<MinReplicaAvailable> = require_available
            .unwrap_or_default()
            .into_iter()
            .map(|client| MinReplicaAvailable {
                name: client.endpoint.id().to_string(),
                router: Arc::new(JsonRouterGuardClient::new(client.router)),
            })
            .collect();

        // The preflight runs only on the first route attempt (a stale-route
        // reroute does not change the downstream router's reported loads), so
        // capture its inputs here; `route_and_connect` takes them by value and
        // `take`s once.
        let preflight_inputs: Option<PreflightInputs> = next_check.map(|check| PreflightInputs {
            check,
            tokens: tokens_for_check.unwrap_or_default(),
            block_mm_infos: block_mm_infos_typed,
        });

        let request_id = context.inner().id().to_string();
        let router_router = self.router.router.clone();
        let worker_router = self.worker.router.clone();
        let block_size = self.block_size;

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            // `worker_args` must be a JSON object so the per-route
            // `RouterResponse::New` can be injected under `router_response`.
            if !matches!(worker_request, serde_json::Value::Object(_)) {
                return Err(PyValueError::new_err(
                    "worker_args must be a JSON object so the router response can be added as the `router_response` field",
                ));
            }

            let router_guard_client: Arc<dyn RouterGuardClient> =
                Arc::new(JsonRouterGuardClient::new(router_router));
            let worker_guard_client: Arc<dyn RouterGuardClient> =
                Arc::new(JsonRouterGuardClient::new(worker_router));
            let parent_context_for_stream = context.clone();

            // --- Route + connect phase (routing shield) ---
            // The whole loop -- `route_once` running the `require_available`
            // check, then (on the first attempt only) the sequential
            // `potential_loads_next_check` preflight, then the `new` route and
            // post-route `require_available` re-check, then `connect_worker`
            // opening the worker stream and re-routing on a stale worker up to
            // `max_reroutes` -- is the `routing` phase. When the caller disallows
            // cancellation during routing it is detached via `shield_route_and_connect`
            // so a Python cancellation cannot abort it; an armed guard abandoned
            // by a cancelled shield is handled inside the detached task: if the
            // loop only reached an armed guard, dropping it fires the cleanup
            // task -> mark_free; if it reached a connected worker stream, the
            // no-taker path drains that stream before dropping the guard. A
            // non-stale open failure propagates from the loop as `Err` and is
            // raised (not a denial).
            let loop_fut = route_and_connect(
                router_guard_client,
                worker_guard_client,
                routing_request,
                request_id,
                context,
                require,
                preflight_inputs,
                worker_request,
                block_size,
                max_reroutes,
                allow_cancel_routing,
                allow_cancel_setup,
                wait_for_first_response,
                ROUTER_GUARD_NOTIFY_TIMEOUT,
                tracing_enabled,
            );
            let outcome = if allow_cancel_routing {
                loop_fut.await.map_err(to_pyerr)?
            } else {
                shield_route_and_connect(loop_fut).await.map_err(to_pyerr)?
            };

            let (guard, worker_id, stream, timings) = match outcome {
                RouteAndConnectOutcome::Denied(denied) => {
                    return Python::with_gil(|py| denied.into_py_any(py));
                }
                RouteAndConnectOutcome::Connected {
                    guard,
                    worker_id,
                    stream,
                    timings,
                } => (guard, worker_id, stream, timings),
            };

            if allow_cancel_stream && !allow_cancel_setup {
                attach_worker_stream_to_parent_context(&stream, &parent_context_for_stream);
            }

            // The guard is wrapped in an `Arc` shared with the background
            // stream-drain task so `mark_free` (fired on `Drop`) is deferred
            // until both the `AdmittedRequest` AND the drain task part with
            // their `Arc` clone -- a caller that takes `response_stream()` then
            // drops the admit object does not prematurely free the routed
            // request while the worker stream is still being consumed.
            let guard = Arc::new(guard);

            // --- Consume the worker stream ---
            // When the caller disallows cancellation during the stream, the
            // stream + guard hand-back is shielded from Python awaitable
            // cancellation; on no taker the detached task drains the worker
            // generation to completion and then drops the guard -> mark_free.
            // Once the `AsyncResponseStream` is returned, though, that stream
            // owns consumption: dropping it closes the receiver, `process_stream`
            // stops on send failure, and the upstream worker stream is dropped.
            // The caller drives `mark_free` itself when it receives the guard.
            // By default it also drives `mark_prefill`; when
            // `mark_prefill_on_response` is enabled, the Rust forwarding
            // task marks prefill when the worker stream produces the first
            // non-error, non-sentinel data item.
            let (guard, stream) = if allow_cancel_stream {
                let (tx, rx) = tokio::sync::mpsc::channel(32);
                let guard_for_drain = Arc::clone(&guard);
                let stream = stream_with_optional_prefill_mark(
                    stream,
                    Arc::clone(&guard),
                    mark_prefill_on_response,
                );
                // Clone `tx` for the `closed()` future; the original `tx` is
                // moved into `process_stream` below. `Sender::closed()`
                // resolves once ALL `Receiver`s are gone (i.e. the Python
                // `AsyncResponseStream` was dropped) and is cancel-safe, so
                // it can race the drain without leaking a sender slot.
                let tx_for_closed = tx.clone();
                tokio::spawn(async move {
                    let mut drain = tokio::spawn(process_stream(stream, tx));
                    let closed = tx_for_closed.closed();
                    tokio::pin!(closed);
                    // Race the drain against consumer-drop: when the Python
                    // `AsyncResponseStream` (the mpsc receiver) is dropped
                    // while the upstream worker is idle, `process_stream`
                    // blocks on `stream.next().await` until the next upstream
                    // item arrives -- so the guard's `mark_free` (fired on
                    // Drop after `process_stream` returns) is delayed by up
                    // to the next upstream item / stream end. `tx.closed()`
                    // completes as soon as all receivers go away, so aborting
                    // the drain here surfaces the consumer-drop promptly:
                    // the drain task is aborted, its held `tx` (and the
                    // upstream `stream`) drop, then `drop(guard_for_drain)`
                    // fires `mark_free` now rather than later.
                    tokio::select! {
                        _ = &mut drain => {}
                        _ = &mut closed => {
                            drain.abort();
                            let _ = drain.await;
                        }
                    }
                    drop(guard_for_drain);
                });
                (guard, AsyncResponseStream::new(rx, annotated))
            } else {
                let stream = stream_with_optional_prefill_mark(
                    stream,
                    Arc::clone(&guard),
                    mark_prefill_on_response,
                );
                let (guard, _source, rx) =
                    shield_stream_to_completion(stream, guard, RouteSource::Routed { worker_id })
                        .await
                        .map_err(to_pyerr)?;
                (guard, AsyncResponseStream::new(rx, annotated))
            };

            let admitted = AdmittedRequest::new(guard, stream, timings, block_size);
            Python::with_gil(|py| admitted.into_py_any(py))
        })
    }
}
