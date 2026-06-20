// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pyclasses and cross-submodule data carriers for the
//! `b10_client::RouterWorkerCoordinator` lifecycle.
//!
//! This submodule holds:
//!  * the Python-facing pyclasses ([`PyRouterRequestNew`], [`CancellationPolicy`],
//!    [`DeniedRequest`], [`RouterCoordinatorPotentialLoadsCheck`], [`AdmittedRequest`]);
//!  * the wire-mirror [`RouterRequestNew`] + its conversion to the wire
//!    [`RouterRequest::New`];
//!  * the plain `Send` data carriers the binding shim (root `b10_client.rs`)
//!    builds under the GIL and hands to the async coordinator: [`MinReplicaAvailable`],
//!    [`PotentialLoadsCheckData`], [`PreflightInputs`], [`NextRouterBackpressureInfo`].
//!
//! Cross-submodule items use `pub(super)`; pyclasses use `pub(crate)` so `lib.rs`
//! can register them as `crate::b10_client::Foo`.

use crate::llm::local_model::RoutingConstraints as PyRoutingConstraints;
use crate::{AsyncResponseStream, Client};
use anyhow::Result;
use dynamo_kv_router::protocols::{BlockExtraInfo, RouterRequest, RoutingConstraints};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

use super::coordinator::RouterGuardClient;
use super::guard::RouterRequestGuard;

/// The payload of a `RouterRequest::New` routing request, minus the `method`
/// tag (supplied by the coordinator). Built from the typed fields of the
/// Python-facing [`PyRouterRequestNew`] pyclass under the GIL, this tag-less
/// mirror is converted to the wire [`RouterRequest::New`] via the [`From`]
/// impl below. The pyclass is the single source of truth for the routing
/// inputs (no `routing_kwargs` dict, no first-class `tokens`/`block_mm_infos`
/// override arguments), so this mirror is no longer serde-deserialized from a
/// Python dict; it is constructed directly and converted to the wire enum.
#[derive(Debug, Clone, Default)]
pub(super) struct RouterRequestNew {
    pub(super) tokens: Vec<u32>,
    pub(super) block_mm_infos: Option<Vec<Option<BlockExtraInfo>>>,
    pub(super) routing_constraints: RoutingConstraints,
    pub(super) priority_jump: f64,
    pub(super) priority_load_shed_percent: u8,
    pub(super) do_not_queue: bool,
}

/// Canonical conversion from the tag-less kwargs mirror into the wire
/// [`RouterRequest::New`] variant. Enum variants have no field-spread syntax in
/// Rust, so the per-field copy lives in this single `From` impl rather than at
/// every call site.
impl From<RouterRequestNew> for RouterRequest {
    fn from(req: RouterRequestNew) -> Self {
        RouterRequest::New {
            tokens: req.tokens,
            block_mm_infos: req.block_mm_infos,
            routing_constraints: req.routing_constraints,
            priority_jump: req.priority_jump,
            priority_load_shed_percent: req.priority_load_shed_percent,
            do_not_queue: req.do_not_queue,
        }
    }
}

impl RouterRequestNew {
    /// Build the wire body for a `new` routing request from the typed fields.
    /// Every field is set by the caller from the [`PyRouterRequestNew`] pyclass
    /// under the GIL; no serde-deserialized defaults are layered in.
    pub(super) fn into_routing_request_value(self) -> Result<serde_json::Value> {
        Ok(serde_json::to_value(RouterRequest::from(self))?)
    }
}

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

impl CancellationPolicy {
    /// Whether the `route_and_connect` loop (route + required-available check
    /// + first-attempt preflight + per-attempt worker setup shield) may be
    ///   cancelled in-band via the request context.
    pub(super) fn allow_cancel_routing(self) -> bool {
        match self {
            Self::Cancellable => true,
            Self::DetachToWorkerStreamConnected => false,
            Self::FullyDetached => false,
            Self::CancellableUntilWorkerThenDetach => true,
            Self::DetachSetupOnly => true,
        }
    }

    /// Whether each per-attempt `direct()` worker-stream open may be cancelled
    /// in-band. Independent from [`Self::allow_cancel_routing`] so a caller can
    /// allow routing cancellation while protecting the open.
    pub(super) fn allow_cancel_setup(self) -> bool {
        match self {
            Self::Cancellable => true,
            Self::DetachToWorkerStreamConnected => false,
            Self::FullyDetached => false,
            Self::CancellableUntilWorkerThenDetach => true,
            Self::DetachSetupOnly => false,
        }
    }

    /// Whether the worker generation stream hand-back may be cancelled after
    /// the open hands the stream back.
    pub(super) fn allow_cancel_stream(self) -> bool {
        match self {
            Self::Cancellable => true,
            Self::DetachToWorkerStreamConnected => true,
            Self::FullyDetached => false,
            Self::CancellableUntilWorkerThenDetach => false,
            Self::DetachSetupOnly => true,
        }
    }
}

/// Python-side carrier of the six [`RouterRequest::New`] wire-body fields
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
#[pyclass]
#[derive(Debug, Clone, Default)]
pub(crate) struct PyRouterRequestNew {
    #[pyo3(get, set)]
    pub(super) tokens: Vec<u32>,
    #[pyo3(get, set)]
    pub(super) block_mm_infos: Option<PyObject>,
    #[pyo3(get, set)]
    pub(super) routing_constraints: Option<Py<PyRoutingConstraints>>,
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
        priority_jump = 0.0,
        priority_load_shed_percent = 0,
        do_not_queue = false,
    ))]
    fn new(
        tokens: Vec<u32>,
        block_mm_infos: Option<PyObject>,
        routing_constraints: Option<Py<PyRoutingConstraints>>,
        priority_jump: f64,
        priority_load_shed_percent: u8,
        do_not_queue: bool,
    ) -> Self {
        Self {
            tokens,
            block_mm_infos,
            routing_constraints,
            priority_jump,
            priority_load_shed_percent,
            do_not_queue,
        }
    }
}

#[derive(Clone)]
pub(super) struct MinReplicaAvailable {
    pub(super) name: String,
    pub(super) router: Arc<dyn RouterGuardClient>,
}

/// Plain, `Send` view of a [`RouterCoordinatorPotentialLoadsCheck`] extracted
/// under the GIL so the async block can run without holding it. `router` is the
/// downstream `client`'s router wrapped as a [`RouterGuardClient`]; the
/// preflight queries *it* (not the routing router).
pub(super) struct PotentialLoadsCheckData {
    pub(super) router: Arc<dyn RouterGuardClient>,
    pub(super) queue_depth_threshold: usize,
    pub(super) prefill_tokens_threshold: usize,
    pub(super) decode_blocks_threshold: usize,
}

/// Fields captured when the next-router preflight finds the aggregated router
/// loads exceed the configured thresholds. Carried on
/// `DeniedRequest::NextRouterBackpressure`.
pub(super) struct NextRouterBackpressureInfo {
    pub(super) queue_depth: usize,
    pub(super) pending_isl_tokens: usize,
    pub(super) total_prefill_tokens: usize,
    pub(super) total_decode_blocks: usize,
}

/// Inputs for the next-router potential-loads preflight, captured up front so
/// the `route_and_connect` loop can run the preflight exactly once (on the first
/// attempt) and skip it on stale-route reroutes -- the downstream router's loads
/// do not change because a routed worker turned out to be stale, so re-querying
/// is wasteful.
pub(super) struct PreflightInputs {
    pub(super) check: PotentialLoadsCheckData,
    pub(super) tokens: Vec<u32>,
    pub(super) block_mm_infos: Option<Vec<Option<BlockExtraInfo>>>,
}

/// Why a [`super::RouterWorkerCoordinator::route_and_worker`] call was denied. One of
/// these is returned (never raised) instead of a `AdmittedRequest` when the
/// router is backpressured, a `require_available` component is down, the
/// optional `potential_loads_next_check` preflight found the next request
/// would overfill the router, or that preflight could not reach the router.
///
/// In Python this is a typed enum: discriminate with `isinstance(result,
/// DeniedRequest.<Variant>)` and read fields as attributes, e.g.
/// `result.name` on `DeniedRequest.RequiredComponentsDown`.
#[pyclass]
#[derive(Debug)]
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
    /// The `potential_loads_next_check` preflight found the aggregated router
    /// loads would exceed the configured thresholds.
    NextRouterBackpressure {
        /// Router-level pending queue depth (`pending_count`).
        queue_depth: usize,
        /// ISL tokens the router reports as currently queued.
        pending_isl_tokens: usize,
        /// Sum of `potential_prefill_tokens` across workers.
        total_prefill_tokens: usize,
        /// Sum of `potential_decode_blocks` across workers.
        total_decode_blocks: usize,
    },
    /// The `potential_loads_next_check` preflight could not reach the router.
    NextRouterUnreachable {
        /// The error encountered while querying the router.
        error: String,
    },
    /// The `potential_loads_next_check` preflight received an unexpected
    /// router response (not `RouterResponse::PotentialLoads`): a
    /// wrong-protocol shape such as `Backpressure` or `New`, or an
    /// older/unknown variant. The preflight fails closed -- the request is
    /// denied rather than passing the overload check unvalidated.
    ProtocolError {
        /// Debug representation of the unexpected `RouterResponse` variant
        /// received from the downstream router, for diagnostics.
        received: String,
    },
}

/// Required preflight passed to [`super::RouterWorkerCoordinator::route_and_worker`]:
/// before routing, the coordinator queries the *downstream* `client` (another
/// router further along the pipeline, e.g. the next router in a
/// disagg-prefill topology) for the *potential loads* of all its workers (the
/// `potential_loads` method) and denies the request when the aggregated loads
/// would exceed the configured thresholds -- so a request is not routed onward
/// to an already-overloaded downstream router. A threshold of `0` disables that
/// dimension (no limit). Prefill is summed in tokens, decode in BLOCKS (no
/// `block_size` conversion), and `queue_depth` is the router-level
/// `pending_count`.
///
/// The `client` is REQUIRED: it is the downstream router whose loads are checked
/// (this is distinct from the routing router the coordinator routes through).
/// The overlap-aware `block_mm_infos` conditioning the reported loads is passed
/// as a first-class argument to `route_and_worker` (and is shared by the route
/// and the preflight), not on this check. Defaults:
/// `queue_depth_threshold=0` (disabled), `prefill_tokens_threshold=1_000_000`,
/// `decode_blocks_threshold=16_000_000`.
#[pyclass]
pub(crate) struct RouterCoordinatorPotentialLoadsCheck {
    /// Downstream router `Client` whose potential loads are checked ahead of
    /// routing, to ensure it is not already overloaded.
    #[pyo3(get, set)]
    pub(super) client: Client,
    #[pyo3(get, set)]
    pub(super) queue_depth_threshold: usize,
    #[pyo3(get, set)]
    pub(super) prefill_tokens_threshold: usize,
    #[pyo3(get, set)]
    pub(super) decode_blocks_threshold: usize,
}

#[pymethods]
impl RouterCoordinatorPotentialLoadsCheck {
    #[new]
    #[pyo3(signature = (
        client,
        queue_depth_threshold = 0,
        prefill_tokens_threshold = 1_000_000,
        decode_blocks_threshold = 16_000_000,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        client: Client,
        queue_depth_threshold: usize,
        prefill_tokens_threshold: usize,
        decode_blocks_threshold: usize,
    ) -> Self {
        Self {
            client,
            queue_depth_threshold,
            prefill_tokens_threshold,
            decode_blocks_threshold,
        }
    }
}

/// Outcome of a [`super::RouterWorkerCoordinator::route_and_worker`] call. Returned
/// only when the route succeeded; carries the lifecycle guard (`mark_prefill`
/// / `mark_free`), the worker generation stream, and the router's reported
/// `overlap_blocks`. A denial is a [`DeniedRequest`] instead. The chosen
/// `worker_id` is not surfaced to Python (it lives on the guard for cleanup).
#[pyclass]
pub(crate) struct AdmittedRequest {
    pub(super) guard: Arc<RouterRequestGuard>,
    pub(super) stream: std::sync::Mutex<Option<AsyncResponseStream>>,
}

impl AdmittedRequest {
    /// Pack an armed guard and the worker generation stream into the
    /// Python-facing admit object. The guard is wrapped in an `Arc` shared
    /// with the background stream-drain task so `mark_free` (fired on `Drop`)
    /// is deferred until both the `AdmittedRequest` AND the drain task part
    /// with their `Arc` clone -- a caller that takes `response_stream()` then
    /// drops the admit object does not prematurely free the routed request
    /// while the worker stream is still being consumed.
    pub(super) fn new(guard: Arc<RouterRequestGuard>, stream: AsyncResponseStream) -> Self {
        Self {
            guard,
            stream: std::sync::Mutex::new(Some(stream)),
        }
    }
}

#[pymethods]
impl AdmittedRequest {
    /// The router's rounded effective cached blocks (approximate KV-cache hit,
    /// in BLOCKS) the router reported for the chosen worker on this request:
    /// how many KV blocks the worker likely already holds for these tokens.
    /// `0` when the route did not arm the guard. Use to derive a per-request hit
    /// rate against the request's decode block count.
    fn overlap_blocks(&self) -> u32 {
        self.guard.overlap_blocks()
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
