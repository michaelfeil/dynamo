// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python-facing request, option, denial, and admission wrappers for the
//! language-neutral types in `dynamo-b10-client`.

use crate::AsyncResponseStream;
use crate::llm::local_model::RoutingConstraints as PyRoutingConstraints;
use crate::tokens::extract_list_or_numpy_u32;
use dynamo_b10_client::{
    AdmittedRequestTimings, CancellationPolicy as CoreCancellationPolicy,
    DeniedRequest as CoreDeniedRequest, GenerationAdmission, RouterRequestGuard,
    RouterWorkerPhase as CoreRouterWorkerPhase,
};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

/// Cancellation policy for a [`super::RouterWorkerCoordinator::route_and_worker`]
/// call. Selects which of the three phases — routing (the `route_and_connect`
/// loop), per-attempt worker-stream setup (the `direct()` open), and the worker
/// generation stream hand-back — may be aborted by a Python cancellation of
/// the awaiting awaitable (in-band, via the linked request context). A detached
/// phase runs to completion regardless; the armed guard produced by an
/// abandoned detached phase is still dropped (and the always-detached
/// `mark_free` cleanup task fired) when the phase finishes, so the router is
/// never left behind un-prefilled.
///
/// Cancelling a phase means propagating `context.is_stopped() || is_killed()`
/// in-band to the underlying `direct()` open and the worker generation stream;
/// it is NEVER a tokio task-drop. Routing and setup are independent axes, so
/// the three booleans ([`Self::allow_cancel_routing`] /
/// [`Self::allow_cancel_setup`] / [`Self::allow_cancel_stream`]) are derived
/// explicitly from the chosen variant via a `match`, NOT from a single global
/// toggle.
#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CancellationPolicy {
    /// Routing, setup, and the worker stream are all cancellable in-band via
    /// the request context. Default.
    Cancellable,
    /// Detach routing and setup (the `route_and_connect` loop, including each
    /// per-attempt `direct()` open) to completion; once the worker stream is
    /// connected and handed back, re-allow cancellation of the stream. Use when
    /// callers cannot tolerate a mid-open abort but DO want to drop the
    /// downstream decode stream on cancellation.
    DetachToWorkerStreamConnected,
    /// Fully detached: routing, setup, and the stream all run to completion;
    /// nothing the caller does can short-circuit any phase. Use for an
    /// unsupervised fire-and-forget path.
    FullyDetached,
    /// Cancellable routing and setup; once the worker stream is in hand,
    /// detach the decode stream to completion so a slow consumer cannot abort
    /// the worker's prefill/decode.
    CancellableUntilWorkerThenDetach,
    /// Cancellable routing and worker stream; detach ONLY each per-attempt
    /// worker-stream setup (`direct()` open + hidden-state transfer) to
    /// completion. The per-attempt dispatch is never revoked mid-flight, but
    /// the route can be cancelled before it lands and the decode stream can
    /// be cancelled after the worker stream is connected. Matches the classic
    /// `allow_cancellation_during_worker_stream_setup=False`,
    /// `allow_cancellation_during_routing=True`,
    /// `allow_cancellation_during_worker_stream=True` triple.
    DetachSetupOnly,
}

impl From<CancellationPolicy> for CoreCancellationPolicy {
    fn from(value: CancellationPolicy) -> Self {
        match value {
            CancellationPolicy::Cancellable => Self::Cancellable,
            CancellationPolicy::DetachToWorkerStreamConnected => {
                Self::DetachToWorkerStreamConnected
            }
            CancellationPolicy::FullyDetached => Self::FullyDetached,
            CancellationPolicy::CancellableUntilWorkerThenDetach => {
                Self::CancellableUntilWorkerThenDetach
            }
            CancellationPolicy::DetachSetupOnly => Self::DetachSetupOnly,
        }
    }
}

/// Logical routing phase for worker attribution.
///
/// This is intentionally a closed Python enum rather than a free-form string:
/// an unknown phase is rejected by PyO3 at the protocol boundary instead of
/// silently attributing a worker to the wrong half of a disaggregated request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[pyclass(name = "RouterWorkerPhase", eq, eq_int)]
pub(crate) enum PyRouterWorkerPhase {
    Agg,
    DecodeFirst,
    PrefillFirst,
    DecodeSecond,
    PrefillSecond,
}

impl PyRouterWorkerPhase {
    pub(super) fn parse(value: &str) -> Option<Self> {
        match value {
            "agg" => Some(Self::Agg),
            "decode_first" => Some(Self::DecodeFirst),
            "prefill_first" => Some(Self::PrefillFirst),
            "decode_second" => Some(Self::DecodeSecond),
            "prefill_second" => Some(Self::PrefillSecond),
            _ => None,
        }
    }

    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Agg => "agg",
            Self::DecodeFirst => "decode_first",
            Self::PrefillFirst => "prefill_first",
            Self::DecodeSecond => "decode_second",
            Self::PrefillSecond => "prefill_second",
        }
    }
}

impl From<PyRouterWorkerPhase> for CoreRouterWorkerPhase {
    fn from(value: PyRouterWorkerPhase) -> Self {
        match value {
            PyRouterWorkerPhase::Agg => Self::Agg,
            PyRouterWorkerPhase::DecodeFirst => Self::DecodeFirst,
            PyRouterWorkerPhase::PrefillFirst => Self::PrefillFirst,
            PyRouterWorkerPhase::DecodeSecond => Self::DecodeSecond,
            PyRouterWorkerPhase::PrefillSecond => Self::PrefillSecond,
        }
    }
}

#[pymethods]
impl PyRouterWorkerPhase {
    fn __str__(&self) -> &'static str {
        self.as_str()
    }
}

#[derive(Debug)]
pub(super) struct RouterWorkerPhaseArg(pub(super) PyRouterWorkerPhase);

impl<'py> FromPyObject<'py> for RouterWorkerPhaseArg {
    fn extract_bound(value: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(phase) = value.extract::<PyRouterWorkerPhase>() {
            return Ok(Self(phase));
        }
        if let Ok(phase) = value.extract::<String>() {
            return PyRouterWorkerPhase::parse(&phase).map(Self).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "invalid RouterWorkerPhase {phase:?}; expected one of: \
                     agg, decode_first, prefill_first, decode_second, prefill_second"
                ))
            });
        }
        Err(PyTypeError::new_err(
            "phase must be a RouterWorkerPhase or one of its exact string values",
        ))
    }
}

#[cfg(test)]
mod router_worker_phase_tests {
    use super::*;

    #[test]
    fn exact_string_values_parse() {
        for (value, phase) in [
            ("agg", PyRouterWorkerPhase::Agg),
            ("decode_first", PyRouterWorkerPhase::DecodeFirst),
            ("prefill_first", PyRouterWorkerPhase::PrefillFirst),
            ("decode_second", PyRouterWorkerPhase::DecodeSecond),
            ("prefill_second", PyRouterWorkerPhase::PrefillSecond),
        ] {
            assert_eq!(PyRouterWorkerPhase::parse(value), Some(phase));
            assert_eq!(phase.as_str(), value);
        }
    }

    #[test]
    fn unknown_string_is_rejected() {
        assert_eq!(PyRouterWorkerPhase::parse("prefill"), None);
        assert_eq!(PyRouterWorkerPhase::parse("unknown"), None);
    }
}

/// Python-side carrier of the seven [`RouterRequest::New`] wire-body fields
/// (minus the `method` tag, supplied by the coordinator). Sent as the
/// REQUIRED `routing_kwargs` argument to
/// [`super::RouterWorkerCoordinator::route_and_worker`]; it is the single source of
/// truth for the routing inputs (the first-class `tokens` and `block_mm_infos`
/// arguments are GONE — both live on this pyclass now).
///
/// `block_mm_infos` is held loosely as `Optional[Any]` on the Python side and
/// converted to the typed wire `Vec<Option<BlockExtraInfo>>` at the Rust
/// boundary under the GIL (via pythonize + serde). `routing_constraints` is the
/// local-model pyclass `RoutingConstraints` (`crate::llm::local_model::
/// RoutingConstraints`, aliased in this module as `PyRoutingConstraints`);
/// `None` means the default (empty) constraints. `tokens` defaults to the
/// empty list.
///
/// `tokens` accepts either a Python sequence of ints or a NumPy `uint32` /
/// `int64` array (see [`crate::tokens::extract_list_or_numpy_u32`]) -- the
/// frontend keeps prompt tokens in a `uint32` buffer, and materializing that
/// buffer into a `list[int]` just to cross this boundary costs one Python
/// `int` object per token. The stored field stays a `Vec<u32>`, so the getter
/// still hands back a `list[int]`.
#[pyclass]
#[derive(Debug, Clone, Default)]
pub(crate) struct PyRouterRequestNew {
    #[pyo3(get)]
    pub(super) tokens: Vec<u32>,
    #[pyo3(get, set)]
    pub(super) block_mm_infos: Option<PyObject>,
    #[pyo3(get, set)]
    pub(super) routing_constraints: Option<Py<PyRoutingConstraints>>,
    #[pyo3(get, set)]
    pub(super) allowed_worker_ids: Option<HashSet<u64>>,
    #[pyo3(get, set)]
    pub(super) priority_jump: f64,
    #[pyo3(get, set)]
    pub(super) priority_load_shed_percent: u8,
    #[pyo3(get, set)]
    pub(super) do_not_queue: bool,
}

#[pymethods]
impl PyRouterRequestNew {
    #[new]
    #[pyo3(signature = (
        tokens,
        block_mm_infos = None,
        routing_constraints = None,
        allowed_worker_ids = None,
        priority_jump = 0.0,
        priority_load_shed_percent = 0,
        do_not_queue = false,
    ))]
    fn new(
        tokens: &Bound<'_, PyAny>,
        block_mm_infos: Option<PyObject>,
        routing_constraints: Option<Py<PyRoutingConstraints>>,
        allowed_worker_ids: Option<HashSet<u64>>,
        priority_jump: f64,
        priority_load_shed_percent: u8,
        do_not_queue: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            tokens: extract_list_or_numpy_u32(tokens)?,
            block_mm_infos,
            routing_constraints,
            allowed_worker_ids,
            priority_jump,
            priority_load_shed_percent,
            do_not_queue,
        })
    }

    /// Accepts the same inputs as the constructor's `tokens`: a Python
    /// sequence of ints or a NumPy `uint32`/`int64` array.
    #[setter]
    fn set_tokens(&mut self, tokens: &Bound<'_, PyAny>) -> PyResult<()> {
        self.tokens = extract_list_or_numpy_u32(tokens)?;
        Ok(())
    }
}

/// Why a [`super::RouterWorkerCoordinator::route_and_worker`] call was denied. One of
/// these is returned (never raised) instead of a `AdmittedRequest` when the
/// router is backpressured, a `require_available` component is down,
/// policy-allowed cancellation wins at a phase boundary, or
/// `wait_for_first_response` could not read a first worker event.
///
/// In Python this is a typed enum: discriminate with `isinstance(result,
/// DeniedRequest.<Variant>)` and read fields as attributes, e.g.
/// `result.name` on `DeniedRequest.RequiredComponentsDown`.
#[pyclass]
#[derive(Debug, Clone)]
pub(crate) enum DeniedRequest {
    /// The KV router itself returned backpressure (or had no router instances up).
    RouterBackpressure {
        /// The router's backpressure reason name (snake_case), e.g. `do_not_queue`.
        reason: String,
        /// ISL tokens the router reports as currently queued.
        queued_isl_tokens: usize,
        /// The configured cap on queued ISL tokens, when known.
        max_queued_isl_tokens: Option<usize>,
    },
    /// A `require_available` component had zero replicas available.
    RequiredComponentsDown {
        /// The name of the down component (its endpoint id).
        name: String,
    },
    /// The selected worker could not be reached, including exhausted stale reroutes.
    NextRouterUnreachable {
        /// The routing or worker connection error.
        error: String,
    },
    /// The router returned an unexpected response variant.
    ProtocolError {
        /// Debug representation of the unexpected `RouterResponse` variant
        /// received from the downstream router, for diagnostics.
        received: String,
    },
    /// The request context was stopped or killed at a phase boundary where the
    /// selected [`CancellationPolicy`] allows cancellation. If the router had
    /// already admitted the request, the coordinator requested `mark_free`
    /// before returning this denial.
    Cancelled(),
    /// `wait_for_first_response` waited for the routed worker stream's first
    /// event, but the stream ended or produced an error before that event could
    /// be handled.
    FirstWorkerEventFailed {
        /// The error encountered while waiting for the first worker stream
        /// event.
        error: String,
    },
}

impl From<CoreDeniedRequest> for DeniedRequest {
    fn from(value: CoreDeniedRequest) -> Self {
        match value {
            CoreDeniedRequest::RouterBackpressure {
                reason,
                queued_isl_tokens,
                max_queued_isl_tokens,
            } => Self::RouterBackpressure {
                reason,
                queued_isl_tokens,
                max_queued_isl_tokens,
            },
            CoreDeniedRequest::RequiredComponentsDown { name } => {
                Self::RequiredComponentsDown { name }
            }
            CoreDeniedRequest::NextRouterUnreachable { error } => {
                Self::NextRouterUnreachable { error }
            }
            CoreDeniedRequest::ProtocolError { received } => Self::ProtocolError { received },
            CoreDeniedRequest::Cancelled() => Self::Cancelled(),
            CoreDeniedRequest::FirstWorkerEventFailed { error } => {
                Self::FirstWorkerEventFailed { error }
            }
        }
    }
}

/// A coordinated generation denial. Unlike a router-only `DeniedRequest`, it
/// may retain admission metadata for a prefill leg that completed before the
/// decode leg was denied.
#[pyclass]
pub(crate) struct DeniedGenerationRequest {
    denied: DeniedRequest,
    admission: Option<GenerationAdmission>,
}

impl DeniedGenerationRequest {
    pub(super) fn new(denied: DeniedRequest, admission: Option<GenerationAdmission>) -> Self {
        Self { denied, admission }
    }
}

#[pymethods]
impl DeniedGenerationRequest {
    /// The denial reason as a typed `DeniedRequest.<Variant>` instance.
    ///
    /// Must go through `into_py_any`, not `Py::new`: for a pyo3 complex enum,
    /// `Py::new` produces an instance of the *base* class only (the pyo3 guide
    /// documents the two as inconsistent), so `isinstance(denied,
    /// DeniedRequest.RouterBackpressure)` was always False and every denial
    /// -- including plain router backpressure -- surfaced to callers as the
    /// generic "framework denied request with unknown response" 500 instead
    /// of a 429. `route_and_worker` already returns denials via `into_py_any`;
    /// this keeps the coordinated path consistent with it. The downcast back
    /// to `Py<DeniedRequest>` keeps the Rust signature (and the `_core.pyi`
    /// stub, `-> DeniedRequest`) honest: every variant class extends the base.
    fn denied_request(&self, py: Python<'_>) -> PyResult<Py<DeniedRequest>> {
        let variant = self.denied.clone().into_py_any(py)?.into_bound(py);
        Ok(variant.downcast_into::<DeniedRequest>()?.unbind())
    }

    fn estimated_overlap_tokens(&self) -> Option<u64> {
        self.admission
            .map(|admission| admission.estimated_overlap_tokens)
    }

    fn b10_best_overlap_blocks(&self) -> Option<u64> {
        self.admission
            .map(|admission| admission.best_overlap_blocks)
    }

    fn prefill_worker_id(&self) -> Option<u64> {
        self.admission.map(|admission| admission.prefill_worker_id)
    }

    fn prefill_dp_rank(&self) -> Option<u32> {
        self.admission.map(|admission| admission.prefill_dp_rank)
    }
}

/// Outcome of a [`super::RouterWorkerCoordinator::route_and_worker`] call. Returned
/// only when the route succeeded; carries the lifecycle guard (`mark_prefill`
/// / `mark_free`), the worker generation stream, the estimated cached-token
/// overlap, route/connect setup timings, and stale-reroute count. A denial is a
/// [`DeniedRequest`] instead. The chosen `worker_id` is not surfaced to Python
/// (it lives on the guard for cleanup).
#[pyclass]
pub(crate) struct AdmittedRequest {
    pub(super) guard: Arc<RouterRequestGuard>,
    pub(super) stream: std::sync::Mutex<Option<AsyncResponseStream>>,
    pub(super) timings: AdmittedRequestTimings,
    pub(super) block_size: u32,
    pub(super) frontend_overhead_duration: Option<Duration>,
}

impl AdmittedRequest {
    /// Pack an armed guard and the worker generation stream into the
    /// Python-facing admit object. The guard is wrapped in an `Arc` shared
    /// with the background stream-drain task so `mark_free` (fired on `Drop`)
    /// is deferred until both the `AdmittedRequest` AND the drain task part
    /// with their `Arc` clone -- a caller that takes `response_stream()` then
    /// drops the admit object does not prematurely free the routed request
    /// while the worker stream is still being consumed.
    pub(super) fn new(
        guard: Arc<RouterRequestGuard>,
        stream: AsyncResponseStream,
        timings: AdmittedRequestTimings,
        block_size: u32,
        frontend_overhead_duration: Option<Duration>,
    ) -> Self {
        Self {
            guard,
            stream: std::sync::Mutex::new(Some(stream)),
            timings,
            block_size,
            frontend_overhead_duration,
        }
    }
}

/// A complete aggregate or prefill-first generation admitted by the Rust
/// generation coordinator. Unlike `AdmittedRequest`, all router guards are
/// owned inside the stream because the coordinator may span two worker pools.
#[pyclass]
pub(crate) struct GeneratedRequest {
    pub(super) stream: std::sync::Mutex<Option<AsyncResponseStream>>,
    pub(super) admission: GenerationAdmission,
}

impl GeneratedRequest {
    pub(super) fn new(stream: AsyncResponseStream, admission: GenerationAdmission) -> Self {
        Self {
            stream: std::sync::Mutex::new(Some(stream)),
            admission,
        }
    }
}

#[pymethods]
impl GeneratedRequest {
    fn estimated_overlap_tokens(&self) -> u64 {
        self.admission.estimated_overlap_tokens
    }

    fn b10_best_overlap_blocks(&self) -> u64 {
        self.admission.best_overlap_blocks
    }

    fn prefill_worker_id(&self) -> u64 {
        self.admission.prefill_worker_id
    }

    fn prefill_dp_rank(&self) -> u32 {
        self.admission.prefill_dp_rank
    }

    fn decode_worker_id(&self) -> Option<u64> {
        self.admission.decode_worker_id
    }

    fn decode_dp_rank(&self) -> Option<u32> {
        self.admission.decode_dp_rank
    }

    /// The coordinated response stream. May be called only once.
    fn response_stream<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let stream = self.stream.lock().unwrap().take().ok_or_else(|| {
            PyValueError::new_err("no response stream available: already consumed")
        })?;
        Ok(Bound::new(py, stream)?.into_any())
    }
}

#[pymethods]
impl AdmittedRequest {
    /// Estimated cached-token overlap for the chosen worker on this request,
    /// derived from the router response and the coordinator block size.
    fn estimated_overlap_tokens(&self) -> u64 {
        self.guard.estimated_overlap_tokens(self.block_size)
    }

    /// Max device-tier overlap across all candidate workers, in blocks.
    /// `0` on routers that predate the field.
    fn b10_best_overlap_blocks(&self) -> u64 {
        self.guard.b10_best_overlap_blocks()
    }

    /// Seconds from HTTP context creation through synchronous Python-to-Rust
    /// request conversion, sampled before the async routing future is created.
    fn frontend_overhead_duration_seconds(&self) -> Option<f64> {
        self.frontend_overhead_duration
            .map(|duration| duration.as_secs_f64())
    }

    /// Seconds from entering route/connect setup to the successful KV-router
    /// `new` response used for this admitted worker. Includes availability checks and
    /// stale-route reroute work before the final accepted route.
    fn routing_new_duration_seconds(&self) -> f64 {
        self.timings.routing_new_duration.as_secs_f64()
    }

    /// Seconds spent opening the KV-router stream, excluding the first router
    /// response wait.
    fn routing_stream_connect_duration_seconds(&self) -> f64 {
        self.timings.routing_stream_connect_duration.as_secs_f64()
    }

    /// Seconds from the successful KV-router `new` response to completed worker
    /// setup. If `wait_for_first_response` was enabled, this includes waiting
    /// for the first worker stream event.
    fn worker_connect_duration_seconds(&self) -> f64 {
        self.timings.worker_connect_duration.as_secs_f64()
    }

    /// Seconds spent opening the worker stream, excluding any optional first
    /// worker stream event wait.
    fn worker_stream_connect_duration_seconds(&self) -> f64 {
        self.timings.worker_stream_connect_duration.as_secs_f64()
    }

    /// Seconds spent waiting for a non-sentinel first worker stream event during
    /// setup. `None` means `wait_for_first_response` was disabled or the first
    /// event was a drop-message sentinel.
    fn worker_first_response_duration_seconds(&self) -> Option<f64> {
        self.timings
            .worker_first_response_duration
            .map(|duration| duration.as_secs_f64())
    }

    /// Seconds spent waiting for a drop-message sentinel as the first worker
    /// stream event during setup.
    /// `None` means `wait_for_first_response` was disabled.
    fn worker_sentinel_event_duration_seconds(&self) -> Option<f64> {
        self.timings
            .worker_sentinel_event_duration
            .map(|duration| duration.as_secs_f64())
    }

    /// Number of stale-route reroutes before this request was admitted. `0`
    /// means the first route connected to its worker.
    fn stale_reroutes(&self) -> u64 {
        self.timings.stale_reroutes
    }

    /// The worker generation stream. May be called only once. Raises
    /// `ValueError` if already consumed.
    fn response_stream<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let stream = self.stream.lock().unwrap().take().ok_or_else(|| {
            PyValueError::new_err("no response stream available: already consumed")
        })?;
        Ok(Bound::new(py, stream)?.into_any())
    }

    /// Mark the routed request's KV blocks as prefilled on the worker. No-op
    /// when the route did not arm the guard.
    fn mark_prefill(&self) {
        self.guard.mark_prefill();
    }

    /// Free the routed request's KV blocks on the worker. Also performed
    /// automatically on drop, so calling this is optional (it lets the caller
    /// free eagerly instead of waiting for GC).
    fn mark_free(&self) {
        self.guard.mark_free();
    }
}
