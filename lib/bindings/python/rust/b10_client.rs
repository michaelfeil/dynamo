//! PyO3 adapter for [`dynamo_b10_client::RouterWorkerCoordinator`].
//!
//! Routing, admission, worker connection, cancellation, and guard cleanup live
//! in `dynamo-b10-client`. This module converts Python inputs, records routed
//! worker metadata, and exposes the returned stream as an async Python object.
mod startup;
mod types;
use startup::{CoordinatorClient, CoordinatorStartup};

// Re-export the pyclasses registered in `lib.rs::add_class::<...>` so they
// resolve as `crate::b10_client::Foo` (the crate-root path lib.rs expects).
// Pyclasses are `pub(crate)` in `types` (and `pub(crate)` here on
// `RouterWorkerCoordinator`); glob re-export would miss them.
use types::RouterWorkerPhaseArg;
pub(crate) use types::{
    AdmittedRequest, CancellationPolicy, DeniedGenerationRequest, DeniedRequest, GeneratedRequest,
    PyRouterRequestNew, PyRouterWorkerPhase, RouterCoordinatorPotentialLoadsCheck,
};

use crate::llm::local_model::RoutingConstraints as PyRoutingConstraints;
use crate::{AsyncResponseStream, Client, context, process_stream, to_pyerr};
use dynamo_b10_client::{
    CancellationPolicy as CoreCancellationPolicy,
    DisaggregationStrategy as CoreDisaggregationStrategy,
    GenerationCoordinator as CoreGenerationCoordinator,
    GenerationCoordinatorClient as CoreGenerationCoordinatorClient,
    GenerationCoordinatorService as CoreGenerationCoordinatorService, GenerationOptions,
    GenerationOutcome as CoreGenerationOutcome, GenerationRequest, JsonRouterGuardClient,
    MinReplicaAvailable, PotentialLoadsCheck, PrefillMarkTiming as CorePrefillMarkTiming,
    RemoteGenerationCoordinator, RequestContext, RouteAndConnectOutcome, RouteOptions,
    RouterRequestGuard, RouterRequestNew, RouterWorkerCoordinator as CoreRouterWorkerCoordinator,
    RouterWorkerPhase as CoreRouterWorkerPhase,
    RunningGenerationCoordinatorService as CoreRunningGenerationCoordinatorService,
    stream_with_optional_prefill_mark,
};
use dynamo_kv_router::protocols::{BlockExtraInfo, RoutingConstraints};
use dynamo_runtime::pipeline::EngineStream;
use dynamo_runtime::protocols::annotated::Annotated as RsAnnotated;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::collections::BTreeMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

pub(crate) use dynamo_b10_client::DROP_THIS_MESSAGE_KEY;

async fn shield_stream_to_completion(
    stream: EngineStream<RsAnnotated<rmpv::Value>>,
    guard: Arc<RouterRequestGuard>,
) -> anyhow::Result<(
    Arc<RouterRequestGuard>,
    tokio::sync::mpsc::Receiver<RsAnnotated<PyObject>>,
)> {
    let (output_tx, output_rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let (stream_tx, stream_rx) = tokio::sync::mpsc::channel(32);
        let guard_for_drain = Arc::clone(&guard);
        tokio::spawn(async move {
            process_stream(stream, stream_tx).await;
            drop(guard_for_drain);
        });
        if let Err((guard, mut stream_rx)) = output_tx.send((guard, stream_rx)) {
            while stream_rx.recv().await.is_some() {}
            drop(guard);
        }
    });
    output_rx.await.map_err(|_| {
        anyhow::anyhow!("detached stream task ended without a receiver (caller cancelled)")
    })
}

fn enum_value(value: Option<&Bound<'_, PyAny>>) -> PyResult<Option<String>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    if let Ok(value) = value.extract::<String>() {
        return Ok(Some(value));
    }
    value
        .getattr("value")
        .and_then(|value| value.extract::<String>())
        .map(Some)
        .map_err(|_| PyTypeError::new_err("expected a string or enum with a string .value"))
}

fn extract_router_request(
    py: Python<'_>,
    routing_kwargs: &Py<PyRouterRequestNew>,
) -> PyResult<RouterRequestNew> {
    let borrowed = routing_kwargs.bind(py).borrow();
    let tokens = borrowed.tokens.clone();
    let block_mm_infos_py = borrowed.block_mm_infos.clone();
    let routing_constraints_py = borrowed.routing_constraints.clone();
    let allowed_worker_ids = borrowed.allowed_worker_ids.clone();
    let priority_jump = borrowed.priority_jump;
    let priority_load_shed_percent = borrowed.priority_load_shed_percent;
    let do_not_queue = borrowed.do_not_queue;
    drop(borrowed);

    let block_mm_infos = match block_mm_infos_py {
        Some(mm) => {
            let value = pythonize::depythonize(&mm.into_bound(py))?;
            Some(serde_json::from_value(value).map_err(to_pyerr)?)
        }
        None => None,
    };
    let routing_constraints = match routing_constraints_py {
        Some(rc) => {
            let rc = rc.bind(py).borrow().clone();
            RoutingConstraints::from(rc)
        }
        None => RoutingConstraints::default(),
    };

    Ok(RouterRequestNew {
        tokens,
        block_mm_infos,
        routing_constraints,
        allowed_worker_ids,
        priority_jump,
        priority_load_shed_percent,
        do_not_queue,
    })
}

fn generation_python_stream(
    stream: EngineStream<RsAnnotated<rmpv::Value>>,
    annotated: bool,
) -> AsyncResponseStream {
    let (tx, rx) = tokio::sync::mpsc::channel(32);
    let tx_for_closed = tx.clone();
    tokio::spawn(async move {
        let mut drain = tokio::spawn(process_stream(stream, tx));
        tokio::select! {
            _ = &mut drain => {}
            _ = tx_for_closed.closed() => {
                drain.abort();
                let _ = drain.await;
            }
        }
    });
    AsyncResponseStream::new(rx, annotated)
}

/// Python adapter for the language-neutral Rust generation coordinator.
///
/// `generate()` is awaitable and returns either `GeneratedRequest` or a
/// `DeniedGenerationRequest`, which wraps the typed denial and any completed
/// prefill admission. Request-model serialization remains outside this class:
/// callers pass the already-msgpackable worker dictionaries and a
/// `PyRouterRequestNew`.
#[pyclass]
pub(crate) struct GenerationCoordinator {
    startup: Option<Arc<CoordinatorStartup>>,
    inner: Arc<dyn CoreGenerationCoordinatorClient>,
    service: Option<Arc<CoreGenerationCoordinatorService>>,
    running_service: Arc<tokio::sync::Mutex<Option<CoreRunningGenerationCoordinatorService>>>,
}

#[pymethods]
impl GenerationCoordinator {
    #[new]
    #[pyo3(signature = (
        *,
        primary_worker_client,
        primary_router_client,
        next_worker_client=None,
        next_router_client=None,
        disaggregation_strategy=None,
        model_name,
        kv_block_size,
        disagg_request_id_machine_id=None,
        prefill_mark_timing=None,
        runtime=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        primary_worker_client: &Bound<'_, PyAny>,
        primary_router_client: &Bound<'_, PyAny>,
        next_worker_client: Option<&Bound<'_, PyAny>>,
        next_router_client: Option<&Bound<'_, PyAny>>,
        disaggregation_strategy: Option<&Bound<'_, PyAny>>,
        model_name: String,
        kv_block_size: u32,
        disagg_request_id_machine_id: Option<u64>,
        prefill_mark_timing: Option<&Bound<'_, PyAny>>,
        runtime: Option<&crate::DistributedRuntime>,
    ) -> PyResult<Self> {
        let _ = model_name;
        let strategy = match enum_value(disaggregation_strategy)?.as_deref() {
            None | Some("aggregated" | "prefill_and_decode") => {
                CoreDisaggregationStrategy::Aggregated
            }
            Some("prefill_first") => CoreDisaggregationStrategy::PrefillFirst,
            Some(value) => {
                return Err(PyValueError::new_err(format!(
                    "unsupported disaggregation strategy: {value}"
                )));
            }
        };
        let mark_timing = match enum_value(prefill_mark_timing)?.as_deref() {
            None | Some("after_prefill_compute") => CorePrefillMarkTiming::AfterPrefillCompute,
            Some("after_transfer") => CorePrefillMarkTiming::AfterTransfer,
            Some(value) => {
                return Err(PyValueError::new_err(format!(
                    "unsupported prefill mark timing: {value}"
                )));
            }
        };
        if strategy == CoreDisaggregationStrategy::PrefillFirst
            && (next_worker_client.is_none() || next_router_client.is_none())
        {
            return Err(PyValueError::new_err(
                "next_worker_client and next_router_client are required for disaggregated generation",
            ));
        }
        if kv_block_size == 0 {
            return Err(PyValueError::new_err("kv_block_size must be positive"));
        }
        let machine_id = disagg_request_id_machine_id
            .or_else(|| runtime.map(|runtime| runtime.inner().connection_id()))
            .ok_or_else(|| {
                PyValueError::new_err("disagg_request_id_machine_id is required without runtime")
            })?;
        let startup = Arc::new(CoordinatorStartup {
            _runtime: runtime.map(|runtime| Arc::new(runtime.inner().clone())),
            primary_worker: CoordinatorClient::parse(primary_worker_client, runtime)?,
            primary_router: CoordinatorClient::parse(primary_router_client, runtime)?,
            next_worker: next_worker_client
                .map(|client| CoordinatorClient::parse(client, runtime))
                .transpose()?,
            next_router: next_router_client
                .map(|client| CoordinatorClient::parse(client, runtime))
                .transpose()?,
            strategy,
            mark_timing,
            block_size: kv_block_size,
            machine_id,
            ready: tokio::sync::OnceCell::new(),
        });
        let inner: Arc<dyn CoreGenerationCoordinatorClient> = startup.clone();
        let service = Arc::new(CoreGenerationCoordinatorService::new(
            Arc::clone(&inner),
            strategy,
        ));
        Ok(Self {
            startup: Some(startup),
            inner,
            service: Some(service),
            running_service: Arc::new(tokio::sync::Mutex::new(None)),
        })
    }

    /// Connect to exactly one named remote generation coordinator backend.
    ///
    /// The returned object exposes the same `generate()` method as the local
    /// constructor. Endpoint discovery and multi-endpoint selection are left
    /// to a future client implementation.
    #[classmethod]
    fn remote(_cls: &Bound<'_, PyType>, mut backends: BTreeMap<String, String>) -> PyResult<Self> {
        if backends.len() != 1 {
            return Err(PyValueError::new_err(
                "remote generation currently requires exactly one named backend",
            ));
        }
        let (name, url) = backends.pop_first().expect("length checked above");
        if name.is_empty() {
            return Err(PyValueError::new_err(
                "remote generation backend name cannot be empty",
            ));
        }
        let inner = RemoteGenerationCoordinator::new(url).map_err(to_pyerr)?;
        Ok(Self {
            startup: None,
            inner: Arc::new(inner),
            service: None,
            running_service: Arc::new(tokio::sync::Mutex::new(None)),
        })
    }

    #[getter]
    fn is_client(&self) -> bool {
        self.service.is_none()
    }

    #[getter]
    fn is_server(&self) -> bool {
        self.service.is_some()
    }

    /// Resolve endpoint strings once. Also performed by serve() and generate().
    fn start<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let startup = self.startup.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            if let Some(startup) = startup {
                startup.start().await.map_err(to_pyerr)?;
            }
            Ok(())
        })
    }

    /// Bind the native HTTP entry point in the background. Direct `generate()`
    /// calls remain available on the same coordinator.
    #[pyo3(signature = (host="0.0.0.0", port=8080))]
    fn serve<'p>(&self, py: Python<'p>, host: &str, port: u16) -> PyResult<Bound<'p, PyAny>> {
        let service = self.service.as_ref().cloned().ok_or_else(|| {
            PyValueError::new_err("a remote generation coordinator is client-only")
        })?;
        let host = host
            .parse::<IpAddr>()
            .map_err(|error| PyValueError::new_err(format!("invalid bind host: {error}")))?;
        let address = SocketAddr::new(host, port);
        let running = Arc::clone(&self.running_service);
        let startup = self.startup.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            if let Some(startup) = startup {
                startup.start().await.map_err(to_pyerr)?;
            }
            let mut slot = running.lock().await;
            if slot.is_some() {
                return Err(PyValueError::new_err(
                    "generation coordinator server is already running",
                ));
            }
            let server = service.start(address).await.map_err(to_pyerr)?;
            let endpoint_url = server.endpoint_url();
            *slot = Some(server);
            Ok(endpoint_url)
        })
    }

    fn shutdown<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let running = Arc::clone(&self.running_service);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let server = running.lock().await.take();
            if let Some(server) = server {
                server.shutdown().await.map_err(to_pyerr)?;
            }
            Ok(())
        })
    }

    /// Coordinate aggregate or prefill-first generation entirely in Rust.
    ///
    /// Aggregate routing/setup/streaming is cancellable. In prefill-first mode,
    /// cancellation is allowed through the prefill response, disabled while
    /// the handoff is routed and connected to decode, then enabled again for
    /// the decode stream.
    #[pyo3(signature = (
        context,
        routing_kwargs,
        worker_args,
        decode_worker_args=None,
        enable_potential_loads_next_check=false,
        annotated=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn generate<'p>(
        &self,
        py: Python<'p>,
        context: context::Context,
        routing_kwargs: Py<PyRouterRequestNew>,
        worker_args: PyObject,
        decode_worker_args: Option<PyObject>,
        enable_potential_loads_next_check: bool,
        annotated: bool,
    ) -> PyResult<Bound<'p, PyAny>> {
        let routing_request = extract_router_request(py, &routing_kwargs)?;
        let primary_worker_request = pythonize::depythonize(&worker_args.into_bound(py))?;
        let decode_worker_request = decode_worker_args
            .map(|args| pythonize::depythonize(&args.into_bound(py)))
            .transpose()?;
        let core_context = RequestContext::new(
            context.inner(),
            context.trace_context().cloned(),
            context.metadata_snapshot(),
        );
        let coordinator = Arc::clone(&self.inner);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let outcome = coordinator
                .generate(
                    core_context,
                    GenerationRequest {
                        routing_request,
                        primary_worker_request,
                        decode_worker_request,
                    },
                    GenerationOptions {
                        enable_potential_loads_next_check,
                        ..Default::default()
                    },
                )
                .await
                .map_err(to_pyerr)?;
            match outcome {
                CoreGenerationOutcome::Denied(denied) => {
                    if let Some(admission) = denied.admission {
                        context.record_prefill_worker(
                            admission.prefill_worker_id,
                            admission.prefill_dp_rank,
                        );
                    }
                    Python::with_gil(|py| {
                        DeniedGenerationRequest::new(
                            DeniedRequest::from(denied.denied),
                            denied.admission,
                        )
                        .into_py_any(py)
                    })
                }
                CoreGenerationOutcome::Connected(generated) => {
                    context.record_prefill_worker(
                        generated.admission.prefill_worker_id,
                        generated.admission.prefill_dp_rank,
                    );
                    if let (Some(worker_id), Some(dp_rank)) = (
                        generated.admission.decode_worker_id,
                        generated.admission.decode_dp_rank,
                    ) {
                        context.record_decode_worker(worker_id, dp_rank);
                    }
                    let stream = generation_python_stream(generated.stream, annotated);
                    Python::with_gil(|py| {
                        GeneratedRequest::new(stream, generated.admission).into_py_any(py)
                    })
                }
            }
        })
    }
}

#[pyclass]
pub(crate) struct RouterWorkerCoordinator {
    inner: Arc<CoreRouterWorkerCoordinator>,
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

        let inner = CoreRouterWorkerCoordinator::from_push_routers(
            router_client.router,
            worker_client.router,
            block_size,
        )
        .map_err(to_pyerr)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// KV router block size in tokens.
    fn block_size(&self) -> u32 {
        self.inner.block_size()
    }

    /// Route a KV-router `new` request, then generate on the routed worker.
    ///
    /// `routing_kwargs` is a [`PyRouterRequestNew`] pyclass -- the REQUIRED,
    /// single source of truth for the seven `RouterRequest::New` wire-body fields
    /// (`tokens`, `block_mm_infos`, `routing_constraints`, `allowed_worker_ids`, `priority_jump`,
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
    /// When provided, `phase` is a [`PyRouterWorkerPhase`] (or its exact string
    /// representation). Aggregate and decode
    /// variants record the selected worker as the serving/decode worker;
    /// prefill variants record it as the prefill worker. A later phase of the
    /// same kind overwrites the earlier attribution.
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
    #[pyo3(signature = (context, routing_kwargs, worker_args=None, require_available=None, potential_loads_next_check=None, annotated=false, cancellation=CancellationPolicy::Cancellable, max_reroutes=1, tracing_enabled=false, wait_for_first_response=false, mark_prefill_on_response=false, phase=None))]
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
        phase: Option<RouterWorkerPhaseArg>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let annotated = annotated.unwrap_or(false);
        let core_cancellation = CoreCancellationPolicy::from(cancellation);
        let allow_cancel_stream = core_cancellation.allow_cancel_stream();

        // Extract the PyRouterRequestNew under the GIL; the async block runs
        // without it. `tokens`, `block_mm_infos`, and `routing_constraints` all
        // live on the pyclass now -- there is no `routing_kwargs` dict, no
        // first-class `tokens`, and no first-class `block_mm_infos` argument.
        let borrowed = routing_kwargs.bind(py).borrow();
        let tokens: Vec<u32> = borrowed.tokens.clone();
        let block_mm_infos_py: Option<PyObject> = borrowed.block_mm_infos.clone();
        let routing_constraints_py: Option<Py<PyRoutingConstraints>> =
            borrowed.routing_constraints.clone();
        let allowed_worker_ids = borrowed.allowed_worker_ids.clone();
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
        let next_check: Option<PotentialLoadsCheck> = match potential_loads_next_check {
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
                Some(PotentialLoadsCheck {
                    router: Arc::new(JsonRouterGuardClient::new(borrowed.client.router.clone())),
                    queue_depth_threshold: borrowed.queue_depth_threshold,
                    prefill_tokens_threshold: borrowed.prefill_tokens_threshold,
                    decode_tokens_threshold: borrowed.decode_tokens_threshold,
                    load_percentile: borrowed.load_percentile,
                })
            }
            None => None,
        };

        let routing_request = RouterRequestNew {
            tokens,
            block_mm_infos: block_mm_infos_typed,
            routing_constraints: routing_constraints_wire,
            allowed_worker_ids,
            priority_jump,
            priority_load_shed_percent,
            do_not_queue,
        };
        let worker_request: rmpv::Value = match worker_args {
            Some(wa) => pythonize::depythonize(&wa.into_bound(py))?,
            None => rmpv::Value::Map(Vec::new()),
        };

        let require: Vec<MinReplicaAvailable> = require_available
            .unwrap_or_default()
            .into_iter()
            .map(|client| MinReplicaAvailable {
                name: client.endpoint.id().to_string(),
                router: Arc::new(JsonRouterGuardClient::new(client.router)),
            })
            .collect();

        let block_size = self.inner.block_size();
        let coordinator = Arc::clone(&self.inner);
        let phase = phase.map(|phase| phase.0);
        let core_phase = phase.map(CoreRouterWorkerPhase::from);
        let core_context = RequestContext::new(
            context.inner(),
            context.trace_context().cloned(),
            context.metadata_snapshot(),
        );
        // Sample after all synchronous PyO3 conversions and immediately before
        // creating the async routing future.
        let frontend_overhead_duration = context
            .milliseconds_since_request_start()?
            .map(Duration::from_millis);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let parent_context_for_stream = context.clone();
            if !matches!(worker_request, rmpv::Value::Map(_)) {
                return Err(PyValueError::new_err(
                    "worker_args must be a JSON object so the router response can be added as the `router_response` field",
                ));
            }
            let outcome = coordinator
                .route_and_worker(
                    core_context,
                    routing_request,
                    worker_request,
                    RouteOptions {
                        require_available: require,
                        potential_loads_check: next_check,
                        cancellation: core_cancellation,
                        max_reroutes,
                        tracing_enabled,
                        wait_for_first_response,
                        phase: core_phase,
                    },
                )
                .await
                .map_err(to_pyerr)?;

            let (guard, stream, timings) = match outcome {
                RouteAndConnectOutcome::Denied(denied) => {
                    let denied = DeniedRequest::from(denied);
                    return Python::with_gil(|py| denied.into_py_any(py));
                }
                RouteAndConnectOutcome::Connected {
                    guard,
                    worker_id: _,
                    stream,
                    timings,
                } => (guard, stream, timings),
            };

            if let (Some(phase), Some((worker_id, dp_rank))) = (phase, guard.routed_worker_info()) {
                match phase {
                    PyRouterWorkerPhase::Agg => {
                        parent_context_for_stream.record_prefill_worker(worker_id, dp_rank);
                        parent_context_for_stream.record_decode_worker(worker_id, dp_rank);
                    }
                    PyRouterWorkerPhase::DecodeFirst | PyRouterWorkerPhase::DecodeSecond => {
                        parent_context_for_stream.record_decode_worker(worker_id, dp_rank);
                    }
                    PyRouterWorkerPhase::PrefillFirst | PyRouterWorkerPhase::PrefillSecond => {
                        parent_context_for_stream.record_prefill_worker(worker_id, dp_rank);
                    }
                }
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
                let (guard, rx) = shield_stream_to_completion(stream, guard)
                    .await
                    .map_err(to_pyerr)?;
                (guard, AsyncResponseStream::new(rx, annotated))
            };

            let admitted = AdmittedRequest::new(
                guard,
                stream,
                timings,
                block_size,
                frontend_overhead_duration,
            );
            Python::with_gil(|py| admitted.into_py_any(py))
        })
    }
}
