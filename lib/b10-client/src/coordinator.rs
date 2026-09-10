// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Routing and worker-connect orchestration for the
//! `b10_client::RouterWorkerCoordinator` lifecycle.
//!
//! Holds the [`RouterGuardClient`] trait (+ the production
//! [`JsonRouterGuardClient`] impl), the `route_request` /
//! `route_once` / `connect_worker` / `route_and_connect` core, the
//! `shield_to_completion` / `shield_stream_to_completion` detach helpers, and
//! the next-router `potential_loads` preflight
//! (`query_potential_loads` / `evaluate_potential_loads`). Internal routing
//! enums (`RouteSource` / `RouteOnceOutcome` / `OpenResult` /
//! `RouteAndConnectOutcome`) live here too.
//!
//! Cross-submodule items use `pub(super)` so the root `b10_client` module (its
//! `RouterWorkerCoordinator` shim) and the sibling `b10_client::tests` module
//! can reach them; purely internal items stay private.

use crate::context::RequestContext;
use anyhow::Result;
use dynamo_kv_router::protocols::{
    BlockExtraInfo, RouterBackpressureReason, RouterRequest, RouterResponse as RsRouterResponse,
};
use dynamo_llm::discovery::{RuntimeConfigWatch, runtime_config_watch};
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::pipeline::{
    AsyncEngineContextProvider, EngineStream, PushRouter, ResponseStream, async_trait,
    context::Context as RsContext,
};
use dynamo_runtime::prelude::DistributedRuntimeProvider;
use dynamo_runtime::protocols::annotated::Annotated as RsAnnotated;
use futures::StreamExt;
use rand::Rng;
use serde::{Serialize, de::DeserializeOwned};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tracing::Instrument;

/// Convert any `Serialize` into an `rmpv::Value`. Used for routing metadata
/// (`RouterRequest` / `RouterResponse`) that originates from typed wire structs
/// but must flow through the `rmpv::Value`-typed request plane alongside the
/// user payload. Goes through msgpack rather than `serde_json`: a `new`
/// request carries the whole prompt, and the JSON hop rebuilt every token as a
/// `serde_json::Value` before rebuilding it again as an `rmpv::Value`.
/// `to_vec_named` is required -- `rmpv::ext::to_value` emits the compact
/// representation (structs as arrays, enums as `[index, payload]`), which is
/// not what the request plane sends.
fn to_rmpv_value<T: Serialize>(value: &T) -> Result<rmpv::Value> {
    let bytes = rmp_serde::to_vec_named(value)?;
    Ok(rmpv::decode::read_value(&mut bytes.as_slice())?)
}

/// Decode a `Deserialize` type from an `rmpv::Value`.
/// Used to recover the typed `RouterResponse` from the wire `rmpv::Value`.
fn from_rmpv_value<T: DeserializeOwned>(value: &rmpv::Value) -> Result<T> {
    let mut bytes = Vec::new();
    rmpv::encode::write_value(&mut bytes, value)?;
    Ok(rmp_serde::from_slice(&bytes)?)
}

use super::DROP_THIS_MESSAGE_KEY;
use super::guard::{
    ROUTER_GUARD_ATTEMPTS, ROUTER_GUARD_CALLBACK_TIMEOUT, ROUTER_GUARD_CLEANUP_GRACE_PERIOD,
    ROUTER_GUARD_NOTIFY_TIMEOUT, ROUTER_GUARD_RETRY_DELAY, RouterRequestGuard,
};
use super::payload_copy::PayloadCopy;
use super::types::{
    AdmittedRequestTimings, DeniedRequest, MinReplicaAvailable, NextRouterBackpressureInfo,
    PotentialLoadsCheck, PreflightInputs, RouteOptions, RouterRequestNew,
};

/// JSON-typed push router used to talk to KV router instances.
///
/// On v1.2.0 the Python `Client` pyclass holds a `PushRouter<rmpv::Value,
/// RsAnnotated<rmpv::Value>>` plus a separate `endpoint` handle; this
/// alias names that router type used throughout the b10_client coordinator.
pub type JsonPushRouter = PushRouter<rmpv::Value, RsAnnotated<rmpv::Value>>;

const POTENTIAL_LOADS_SLOW_LOG_THRESHOLD: Duration = Duration::from_secs(1);
const ROUTE_STREAM_OPEN_SLOW_LOG_THRESHOLD: Duration = Duration::from_secs(1);
const ROUTE_FIRST_RESPONSE_SLOW_LOG_THRESHOLD: Duration = Duration::from_secs(1);
const ROUTE_FIRST_RESPONSE_TIMEOUT: Duration = Duration::from_secs(590);
const DURATION_LOG_MS_PRECISION: f64 = 1_000.0;

/// High-level B10 client that routes a request and opens the selected worker.
pub struct RouterWorkerCoordinator {
    router: Arc<dyn RouterGuardClient>,
    worker: Arc<dyn RouterGuardClient>,
    block_size: u32,
}

impl RouterWorkerCoordinator {
    async fn query_loads(
        &self,
        request: crate::protocol::BidRequestV1,
    ) -> Result<RsRouterResponse> {
        request.validate()?;
        let instances = available_router_instance_ids(self.router.as_ref());
        let instance = instances
            .first()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("no router instances available for potential loads"))?;
        // The existing router RPC only models tokens/MM; salt remains
        // on the bid wire contract for future router support.
        let request = to_rmpv_value(&RouterRequest::PotentialLoads {
            tokens: request.tokens.into(),
            block_mm_infos: request
                .mm_routing_args
                .map(crate::protocol::codec::mm_routing_args_from_wire)
                .transpose()?,
            allow_short_caching: false,
        })?;
        let response = tokio::time::timeout(Duration::from_secs(5), async {
            let stream = self
                .router
                .direct(RsContext::new(request), instance)
                .await?;
            first_stream_response(stream).await
        })
        .await
        .map_err(|_| anyhow::anyhow!("potential loads router query timed out"))??;
        Ok(response.response)
    }

    pub(crate) async fn potential_loads(
        &self,
        request: crate::protocol::BidRequestV1,
    ) -> Result<Vec<dynamo_kv_router::scheduling::PotentialLoad>> {
        let RsRouterResponse::PotentialLoads { loads, .. } = self.query_loads(request).await?
        else {
            anyhow::bail!("unexpected router response to bid query");
        };
        anyhow::ensure!(!loads.is_empty(), "no eligible workers for bid");
        Ok(loads)
    }

    pub async fn worker_loads(&self, mode: crate::WorkerMode) -> Result<Vec<crate::WorkerLoad>> {
        let RsRouterResponse::PotentialLoads { loads, .. } = self
            .query_loads(crate::protocol::BidRequestV1 {
                tokens: vec![0],
                ..Default::default()
            })
            .await?
        else {
            anyhow::bail!("unexpected router response to worker loads query");
        };
        let mut workers = std::collections::BTreeMap::new();
        for load in loads {
            let worker = workers.entry(load.worker_id).or_insert(crate::WorkerLoad {
                worker_id: load.worker_id,
                disaggregation_mode: mode,
                potential_prefill_tokens: 0,
                potential_decode_blocks: 0,
                active_requests: 0,
            });
            worker.potential_prefill_tokens += if load.potential_prefill_tokens == 1 {
                0
            } else {
                load.potential_prefill_tokens
            };
            worker.potential_decode_blocks += load.potential_decode_blocks;
            worker.active_requests += load.active_requests;
        }
        Ok(workers.into_values().collect())
    }

    pub(crate) fn router(&self) -> Arc<dyn RouterGuardClient> {
        Arc::clone(&self.router)
    }

    pub fn new(
        router: Arc<dyn RouterGuardClient>,
        worker: Arc<dyn RouterGuardClient>,
        block_size: u32,
    ) -> Result<Self> {
        if block_size == 0 {
            anyhow::bail!("block_size must be positive");
        }
        Ok(Self {
            router,
            worker,
            block_size,
        })
    }

    pub fn from_push_routers(
        router: JsonPushRouter,
        worker: JsonPushRouter,
        block_size: u32,
    ) -> Result<Self> {
        let runtime_configs = spawn_runtime_config_watch(&worker.client.endpoint);
        Self::new(
            Arc::new(JsonRouterGuardClient::new(router)),
            Arc::new(JsonRouterGuardClient {
                router: worker,
                runtime_configs: Some(runtime_configs),
            }),
            block_size,
        )
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    pub async fn route_and_worker(
        &self,
        context: RequestContext,
        routing_request: RouterRequestNew,
        worker_request: rmpv::Value,
        options: RouteOptions,
    ) -> Result<RouteAndConnectOutcome> {
        if !matches!(worker_request, rmpv::Value::Map(_)) {
            anyhow::bail!(
                "worker_args must be a JSON object so the router response can be added as the `router_response` field"
            );
        }

        let preflight_inputs = options.potential_loads_check.map(|check| PreflightInputs {
            tokens: routing_request.tokens.clone(),
            block_mm_infos: routing_request.block_mm_infos.clone(),
            check,
        });
        let routing_request = Arc::new(routing_request.into_routing_request_value()?);
        let request_id = context.id().to_string();
        let phase = options.phase.map(|phase| phase.as_str().to_string());
        let allow_cancel_routing = options.cancellation.allow_cancel_routing();
        let allow_cancel_setup = options.cancellation.allow_cancel_setup();
        let allow_cancel_stream = options.cancellation.allow_cancel_stream();
        let parent_context_for_stream = context.clone();
        let loop_fut = route_and_connect(
            Arc::clone(&self.router),
            Arc::clone(&self.worker),
            routing_request,
            request_id,
            context,
            options.require_available,
            preflight_inputs,
            worker_request,
            self.block_size,
            options.max_reroutes,
            allow_cancel_routing,
            allow_cancel_setup,
            options.wait_for_first_response,
            ROUTER_GUARD_NOTIFY_TIMEOUT,
            options.tracing_enabled,
            phase,
        );
        let outcome = if allow_cancel_routing {
            loop_fut.await
        } else {
            shield_route_and_connect(loop_fut).await
        }?;

        if allow_cancel_stream
            && !allow_cancel_setup
            && let RouteAndConnectOutcome::Connected { stream, .. } = &outcome
        {
            attach_worker_stream_to_parent_context(stream, &parent_context_for_stream);
        }

        Ok(outcome)
    }
}

pub(super) fn duration_ms_for_log(duration: Duration) -> f64 {
    let duration_ms = duration.as_secs_f64() * 1000.0;
    (duration_ms * DURATION_LOG_MS_PRECISION).round() / DURATION_LOG_MS_PRECISION
}

fn denied_request_kind(denied: &DeniedRequest) -> String {
    match denied {
        DeniedRequest::RouterBackpressure { reason, .. } => {
            format!("router_backpressure.{reason}")
        }
        DeniedRequest::RequiredComponentsDown { .. } => "required_components_down".to_string(),
        DeniedRequest::NextRouterBackpressure { .. } => "next_router_backpressure".to_string(),
        DeniedRequest::NextRouterUnreachable { .. } => "next_router_unreachable".to_string(),
        DeniedRequest::ProtocolError { .. } => "protocol_error".to_string(),
        DeniedRequest::Cancelled() => "cancelled".to_string(),
        DeniedRequest::FirstWorkerEventFailed { .. } => "first_worker_event_failed".to_string(),
    }
}

fn log_route_and_connect_denied(
    request_id: &str,
    phase: Option<&str>,
    worker_id: Option<u64>,
    stale_reroutes: u64,
    denied: &DeniedRequest,
) {
    let worker_id = worker_id.map(|worker_id| worker_id.to_string());
    tracing::info!(
        request_id = %request_id,
        phase = phase.unwrap_or("unknown"),
        worker_id = worker_id.as_deref(),
        stale_reroutes,
        denied_kind = %denied_request_kind(denied),
        denied = ?denied,
        unified_model_logs = true,
        "route_and_connect denied"
    );
}

fn create_detached_router_request_context(
    request: rmpv::Value,
    parent_ctx: &Option<RequestContext>,
    request_id: &str,
    follow_parent_cancellation: bool,
) -> (RsContext<rmpv::Value>, Option<tokio::task::JoinHandle<()>>) {
    let request_ctx = RsContext::with_id_and_metadata(
        request,
        request_id.to_string(),
        parent_ctx
            .as_ref()
            .map(|ctx| ctx.metadata_snapshot())
            .unwrap_or_default(),
    );

    let cancellation_forwarder = if follow_parent_cancellation && let Some(parent_ctx) = parent_ctx
    {
        let parent = parent_ctx.inner();
        let route_context = request_ctx.context();
        if parent.is_killed() {
            route_context.kill_with_reason(Some("parent_context_already_killed"));
            None
        } else if parent.is_stopped() {
            route_context.stop_generating_with_reason(Some("parent_context_already_stopped"));
            None
        } else {
            let parent_for_kill = parent.clone();
            let parent_for_stop = parent.clone();
            let route_for_parent = route_context.clone();
            let route_for_kill = route_context.clone();
            let route_for_stop = route_context.clone();
            let route_for_timeout = route_context.clone();
            let request_id = request_id.to_string();
            Some(tokio::spawn(async move {
                tokio::select! {
                    biased;
                    _ = parent_for_kill.killed() => {
                        route_for_parent.kill_with_reason(Some("parent_context_killed"));
                    }
                    _ = parent_for_stop.stopped() => {
                        if parent_for_stop.is_killed() {
                            route_for_parent.kill_with_reason(Some("parent_context_killed"));
                        } else {
                            route_for_parent.stop_generating_with_reason(Some("parent_context_stopped"));
                        }
                    }
                    _ = route_for_kill.killed() => {}
                    _ = route_for_stop.stopped() => {}
                    _ = tokio::time::sleep(ROUTE_FIRST_RESPONSE_TIMEOUT + Duration::from_secs(1)) => {
                        tracing::debug!(
                            request_id = %request_id,
                            timeout_secs = ROUTE_FIRST_RESPONSE_TIMEOUT.as_secs(),
                            "detached route context cancellation forwarder expired"
                        );
                        route_for_timeout.kill_with_reason(Some("route_context_forwarder_timeout"));
                    }
                }
            }))
        }
    } else {
        None
    };

    (request_ctx, cancellation_forwarder)
}

fn abort_cancellation_forwarder(forwarder: &mut Option<tokio::task::JoinHandle<()>>) {
    if let Some(forwarder) = forwarder.take() {
        forwarder.abort();
    }
}

fn trace_context_available(context: &Option<RequestContext>) -> bool {
    context
        .as_ref()
        .and_then(|context| context.trace_context())
        .is_some()
}

fn log_route_step(
    enabled: bool,
    context: &Option<RequestContext>,
    request_id: &str,
    step: &'static str,
) {
    if !enabled {
        return;
    }
    tracing::debug!(
        request_id = %request_id,
        step,
        trace_context_available = trace_context_available(context),
        "b10 route_and_worker step"
    );
}

fn cancellation_denial_for_context(context: &RequestContext, allow: bool) -> Option<DeniedRequest> {
    if !allow {
        return None;
    }

    let inner = context.inner();
    if inner.is_killed() || inner.is_stopped() {
        Some(DeniedRequest::Cancelled())
    } else {
        None
    }
}

fn cancellation_denial_for_optional_context(
    context: &Option<RequestContext>,
    allow: bool,
) -> Option<DeniedRequest> {
    context
        .as_ref()
        .and_then(|ctx| cancellation_denial_for_context(ctx, allow))
}

fn create_worker_request_context(
    request: rmpv::Value,
    parent_ctx: &RequestContext,
    follow_parent_during_setup: bool,
) -> RsContext<rmpv::Value> {
    if follow_parent_during_setup {
        let child_ctx = RsContext::with_id_and_metadata(
            request,
            parent_ctx.id().to_string(),
            parent_ctx.metadata_snapshot(),
        );
        parent_ctx.inner().link_child(child_ctx.context());
        if parent_ctx.inner().is_stopped() || parent_ctx.inner().is_killed() {
            child_ctx
                .context()
                .stop_generating_with_reason(Some("parent_context_already_stopped_or_killed"));
        }
        child_ctx
    } else {
        RsContext::with_id_and_metadata(
            request,
            parent_ctx.inner().id().to_string(),
            parent_ctx.metadata_snapshot(),
        )
    }
}

fn attach_worker_stream_to_parent_context(
    stream: &EngineStream<RsAnnotated<rmpv::Value>>,
    parent: &RequestContext,
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

/// Why a [`route_request`] call resolved the way it did. The coordinator maps
/// these onto `DeniedRequest::RouterBackpressure` /
/// `DeniedRequest::RequiredComponentsDown` /
/// `DeniedRequest::ProtocolError`.
pub enum RouteSource {
    /// Route succeeded; carries the KV router's chosen worker id for generation.
    Routed { worker_id: u64 },
    /// The router itself returned backpressure (or no router instances were up).
    RouterBackpressure,
    /// A `require_available` component had zero replicas available; carries the
    /// component's name so the coordinator can surface it on a `DeniedRequest`.
    RequiredDown { name: String },
    /// The router replied with a variant that is not a clean admit (`New`) or
    /// clean denial (`Backpressure`) for a `new` request (e.g. `PrefillMarked`,
    /// `FreeMarked`, `PotentialLoads`). Admission state is ambiguous, so
    /// `route_request` fails closed -- it requests `mark_free`, drops the
    /// provisional guard, and surfaces a
    /// [`DeniedRequest::ProtocolError`] to the caller. Mirrors the stricter
    /// `potential_loads` handling (`PotentialLoadsError::ProtocolError`).
    ProtocolError { received: String },
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct RouteRequestTimings {
    pub(super) stream_connect_duration: Duration,
}

pub(super) struct RouterStreamResponse {
    data: rmpv::Value,
    pub(super) response: RsRouterResponse,
}

#[async_trait]
pub trait RouterGuardClient: Send + Sync {
    fn endpoint_id(&self) -> String;

    fn available_instance_ids(&self) -> Vec<u64>;

    fn instance_ids(&self) -> Vec<u64>;

    fn stable_routing_id(&self, worker_id: u64) -> Option<String>;

    async fn direct(
        &self,
        request: RsContext<rmpv::Value>,
        instance_id: u64,
    ) -> Result<EngineStream<RsAnnotated<rmpv::Value>>>;
}

#[derive(Clone)]
pub struct JsonRouterGuardClient {
    router: JsonPushRouter,
    runtime_configs: Option<Arc<OnceLock<RuntimeConfigWatch>>>,
}

impl JsonRouterGuardClient {
    pub fn new(router: JsonPushRouter) -> Self {
        Self {
            router,
            runtime_configs: None,
        }
    }
}

/// Non-blocking; lookups return `None` until the watch is established.
fn spawn_runtime_config_watch(endpoint: &Endpoint) -> Arc<OnceLock<RuntimeConfigWatch>> {
    let endpoint = endpoint.clone();
    let slot = Arc::new(OnceLock::new());
    let slot_for_task = Arc::clone(&slot);
    endpoint.drt().runtime().primary().spawn(async move {
        match runtime_config_watch(&endpoint).await {
            Ok(watch) => {
                let _ = slot_for_task.set(watch);
            }
            Err(err) => tracing::warn!(
                endpoint = %endpoint.id(),
                error = %err,
                "stable_routing_id lookup unavailable: runtime config watch failed"
            ),
        }
    });
    slot
}

#[async_trait]
impl RouterGuardClient for JsonRouterGuardClient {
    fn endpoint_id(&self) -> String {
        self.router.client.endpoint.id().to_string()
    }

    fn available_instance_ids(&self) -> Vec<u64> {
        self.router.client.instance_ids_avail().to_vec()
    }

    fn instance_ids(&self) -> Vec<u64> {
        self.router.client.instance_ids()
    }

    fn stable_routing_id(&self, worker_id: u64) -> Option<String> {
        self.runtime_configs
            .as_ref()?
            .get()?
            .borrow()
            .get(&worker_id)?
            .stable_routing_id
            .clone()
    }

    async fn direct(
        &self,
        request: RsContext<rmpv::Value>,
        instance_id: u64,
    ) -> Result<EngineStream<RsAnnotated<rmpv::Value>>> {
        self.router.direct(request, instance_id).await
    }
}

/// Route a KV router `new` request and return a guard plus the outcome source.
///
/// `router` is the routing-decision client (its `PushRouter` reaches the KV
/// router service). `require_min1_replica_available` lists additional
/// components that must have at least one available replica as a preflight;
/// any of those with zero replicas short-circuits to
/// [`RouteSource::RequiredDown`] without routing.
#[allow(clippy::too_many_arguments)]
pub async fn route_request(
    router: Arc<dyn RouterGuardClient>,
    request: Arc<rmpv::Value>,
    request_id: String,
    context: Option<RequestContext>,
    require_min1_replica_available: Vec<MinReplicaAvailable>,
    notify_timeout: Duration,
    tracing_enabled: bool,
    allow_cancel_routing: bool,
) -> Result<(RouterRequestGuard, RouteSource, RouteRequestTimings)> {
    if let Some((response, name)) =
        min_replica_available_backpressure(&require_min1_replica_available)?
    {
        tracing::info!(
            request_id = %request_id,
            replica_name = %name,
            response = ?response.response,
            "route_request returning preflight backpressure (required component down)"
        );
        let guard = RouterRequestGuard::new(
            router,
            request_id,
            0,
            response.data,
            response.response,
            false,
            notify_timeout,
        );
        return Ok((
            guard,
            RouteSource::RequiredDown { name },
            RouteRequestTimings::default(),
        ));
    }

    let instance_ids = available_router_instance_ids(router.as_ref());
    if instance_ids.is_empty() {
        let response = router_backpressure_response()?;
        tracing::info!(
            request_id = %request_id,
            endpoint = %router.endpoint_id(),
            response = ?response.response,
            "route_request returning backpressure because no router instances are available"
        );
        let guard = RouterRequestGuard::new(
            router,
            request_id,
            0,
            response.data,
            response.response,
            false,
            notify_timeout,
        );
        return Ok((
            guard,
            RouteSource::RouterBackpressure,
            RouteRequestTimings::default(),
        ));
    }

    let mut last_error = None;
    for attempt in 0..ROUTER_GUARD_ATTEMPTS {
        for (instance_index, &instance_id) in instance_ids.iter().enumerate() {
            let has_more_route_attempts =
                attempt + 1 < ROUTER_GUARD_ATTEMPTS || instance_index + 1 < instance_ids.len();
            // RouterGuardClient::direct takes RsContext<rmpv::Value> by value,
            // so the payload is materialized here and only here. Everything
            // above this point passes the Arc.
            let (request_ctx, mut cancellation_forwarder) = create_detached_router_request_context(
                (*request).clone(),
                &context,
                &request_id,
                allow_cancel_routing,
            );
            let route_context = request_ctx.context();
            let span = context
                .as_ref()
                .map(|context| context.direct_span("route_request", instance_id))
                .unwrap_or_else(tracing::Span::none);

            // Provisional guard armed BEFORE the router `direct`: if the
            // outer future is cancelled after the router admitted the
            // request internally but before we observe a response, the
            // (always-detached) cleanup task fires `mark_free` so the
            // router's slot is reclaimed. On `direct` returning a clean Err
            // (router denied) `dismiss` stops the cleanup task without
            // sending `mark_free`; on success `commit` installs the
            // response and the cleanup task remains armed (for `New`) or
            // exits immediately (for `Backpressure`). For an unexpected
            // variant (not `New` / `Backpressure`) `route_request` fails
            // closed -- it requests `mark_free`, drops the provisional guard,
            // and surfaces a `RouteSource::ProtocolError`.
            let provisional_guard = RouterRequestGuard::new_provisional(
                router.clone(),
                request_id.clone(),
                instance_id,
                notify_timeout,
            );

            // Stage 1: open the router stream. An `Err` HERE means the
            // router never admitted the request (clean denial or in-band
            // context cancel): `dismiss` the provisional guard so the
            // cleanup task exits WITHOUT sending `mark_free`. The detached
            // route context is still killed and, when another route attempt
            // remains, the retry backs off so any queued router coroutine can
            // observe cancellation before the same request id is reused.
            let stream_open_started = Instant::now();
            let stream = match router
                .direct(request_ctx, instance_id)
                .instrument(span)
                .await
            {
                Ok(stream) => {
                    let elapsed = stream_open_started.elapsed();
                    if elapsed >= ROUTE_STREAM_OPEN_SLOW_LOG_THRESHOLD {
                        tracing::warn!(
                            request_id = %request_id,
                            router_instance_id = instance_id,
                            attempt = attempt + 1,
                            attempts = ROUTER_GUARD_ATTEMPTS,
                            elapsed_ms = elapsed.as_millis(),
                            threshold_ms = ROUTE_STREAM_OPEN_SLOW_LOG_THRESHOLD.as_millis(),
                            "route_request router stream open exceeded expected latency"
                        );
                    } else if tracing_enabled {
                        tracing::debug!(
                            request_id = %request_id,
                            router_instance_id = instance_id,
                            attempt = attempt + 1,
                            attempts = ROUTER_GUARD_ATTEMPTS,
                            elapsed_ms = elapsed.as_millis(),
                            "route_request router stream opened"
                        );
                    }
                    (stream, elapsed)
                }
                Err(err) => {
                    let elapsed = stream_open_started.elapsed();
                    route_context.kill_with_reason(Some("router_direct_failed"));
                    abort_cancellation_forwarder(&mut cancellation_forwarder);
                    provisional_guard.dismiss();
                    last_error = Some(err.to_string());
                    tracing::warn!(
                        request_id = %request_id,
                        router_instance_id = instance_id,
                        attempt = attempt + 1,
                        attempts = ROUTER_GUARD_ATTEMPTS,
                        elapsed_ms = elapsed.as_millis(),
                        error = %err,
                        "route_request router.direct failed (no admission)"
                    );
                    if has_more_route_attempts {
                        tokio::time::sleep(ROUTER_GUARD_CLEANUP_GRACE_PERIOD).await;
                    }
                    continue;
                }
            };
            let (stream, stream_connect_duration) = stream;

            // Stage 2: read the first stream item. The router has now
            // admitted the request internally, so an `Err` here (stream
            // ended before data, decode failure, malformed JSON) is a
            // POST-ADMISSION error: the slot may be reserved on the router.
            // Request `mark_free` before dropping the provisional guard. The
            // cleanup task sends `mark_free` asynchronously. The router's
            // `ActiveSequencesMultiWorker::free` tolerates spurious
            // `mark_free` for unknown request_ids (idempotent
            // `RequestNotFound` arm at
            // lib/kv-router/src/sequences/multi_worker.rs:482 logs at
            // debug and returns Ok). A first-response timeout additionally
            // kills the detached route context so the router coroutine is
            // cancelled instead of only freeing scheduler state.
            let first_response_started = Instant::now();
            let router_response = match tokio::time::timeout(
                ROUTE_FIRST_RESPONSE_TIMEOUT,
                first_stream_response(stream),
            )
            .await
            {
                Ok(Ok(response)) => {
                    let elapsed = first_response_started.elapsed();
                    if elapsed >= ROUTE_FIRST_RESPONSE_SLOW_LOG_THRESHOLD {
                        tracing::warn!(
                            request_id = %request_id,
                            router_instance_id = instance_id,
                            attempt = attempt + 1,
                            attempts = ROUTER_GUARD_ATTEMPTS,
                            elapsed_ms = elapsed.as_millis(),
                            threshold_ms = ROUTE_FIRST_RESPONSE_SLOW_LOG_THRESHOLD.as_millis(),
                            "route_request first router response exceeded expected latency"
                        );
                    } else if tracing_enabled {
                        tracing::debug!(
                            request_id = %request_id,
                            router_instance_id = instance_id,
                            attempt = attempt + 1,
                            attempts = ROUTER_GUARD_ATTEMPTS,
                            elapsed_ms = elapsed.as_millis(),
                            "route_request first router response received"
                        );
                    }
                    response
                }
                Ok(Err(err)) => {
                    last_error = Some(err.to_string());
                    tracing::warn!(
                        request_id = %request_id,
                        router_instance_id = instance_id,
                        attempt = attempt + 1,
                        attempts = ROUTER_GUARD_ATTEMPTS,
                        elapsed_ms = first_response_started.elapsed().as_millis(),
                        error = %err,
                        "route_request post-admission first_stream_response failed; \
                         freeing provisional guard and killing detached route context"
                    );
                    provisional_guard.mark_free();
                    route_context.kill_with_reason(Some("router_first_response_failed"));
                    abort_cancellation_forwarder(&mut cancellation_forwarder);
                    provisional_guard
                        .wait_for_cleanup(ROUTER_GUARD_CLEANUP_GRACE_PERIOD)
                        .await;
                    drop(provisional_guard);
                    continue;
                }
                Err(_) => {
                    tracing::warn!(
                        request_id = %request_id,
                        router_instance_id = instance_id,
                        attempt = attempt + 1,
                        attempts = ROUTER_GUARD_ATTEMPTS,
                        timeout_secs = ROUTE_FIRST_RESPONSE_TIMEOUT.as_secs(),
                        elapsed_ms = first_response_started.elapsed().as_millis(),
                        "route_request timed out waiting for first router response; \
                         freeing provisional guard, killing detached route context, \
                         and returning router backpressure"
                    );
                    provisional_guard.mark_free();
                    route_context.kill_with_reason(Some("router_first_response_timeout"));
                    abort_cancellation_forwarder(&mut cancellation_forwarder);
                    provisional_guard
                        .wait_for_cleanup(ROUTER_GUARD_CLEANUP_GRACE_PERIOD)
                        .await;
                    drop(provisional_guard);
                    let response = router_backpressure_response()?;
                    let guard = RouterRequestGuard::new(
                        router,
                        request_id,
                        instance_id,
                        response.data,
                        response.response,
                        false,
                        notify_timeout,
                    );
                    return Ok((
                        guard,
                        RouteSource::RouterBackpressure,
                        RouteRequestTimings {
                            stream_connect_duration,
                        },
                    ));
                }
            };
            abort_cancellation_forwarder(&mut cancellation_forwarder);

            // Stage 3: dispatch on the decoded router response variant.
            //   `New` => router admitted; commit armed (cleanup stays armed
            //            so a later `Drop` fires `mark_free` once the request
            //            is consumed).
            //   `Backpressure` => clean denial; commit unarmed (cleanup task
            //            exits without sending `mark_free`).
            //   other => protocol error: fail closed. Drop the provisional
            //            guard after requesting `mark_free`, then
            //            return an unarmed placeholder guard via the same
            //            `router_backpressure_response` helper used by the
            //            no-instances early-return. `route_once` drops the
            //            placeholder on its `ProtocolError` arm without
            //            touching `new_response`/`backpressure_fields`.
            if !matches!(
                &router_response.response,
                RsRouterResponse::New { .. } | RsRouterResponse::Backpressure { .. }
            ) {
                let received = serde_json::to_string(&router_response.response)
                    .unwrap_or_else(|_| format!("{:?}", router_response.response));
                tracing::warn!(
                    request_id = %request_id,
                    router_instance_id = instance_id,
                    response = ?router_response.response,
                    "route_request got unexpected router response variant \
                     (expected New or Backpressure); failing closed -- freeing \
                     provisional guard"
                );
                provisional_guard.mark_free();
                drop(provisional_guard);
                // `router` and `request_id` are moved (not cloned) because
                // the immediately-following `return Ok(...)` is the final
                // use of both in this stage; any non-`return` path here
                // `continue`s the loop via the `Err` arm without
                // touching them.
                let placeholder = match router_backpressure_response() {
                    Ok(p) => RouterRequestGuard::new(
                        router,
                        request_id,
                        instance_id,
                        p.data,
                        p.response,
                        false,
                        notify_timeout,
                    ),
                    Err(e) => {
                        last_error = Some(e.to_string());
                        continue;
                    }
                };
                return Ok((
                    placeholder,
                    RouteSource::ProtocolError { received },
                    RouteRequestTimings {
                        stream_connect_duration,
                    },
                ));
            }

            let RouterStreamResponse { data, response } = router_response;
            let (worker_id, armed) = match &response {
                RsRouterResponse::New { worker_id, .. } => (Some(*worker_id), true),
                RsRouterResponse::Backpressure { .. } => {
                    tracing::info!(
                        request_id = %request_id,
                        router_instance_id = instance_id,
                        response = ?response,
                        "route_request returning unarmed (backpressure) response"
                    );
                    (None, false)
                }
                // Unreachable: the `if !matches!` filter above already
                // returned for any variant that is not `New` or
                // `Backpressure`. Returned as `unreachable!` so a future
                // variant added to `RsRouterResponse` that bypasses the
                // filter surfaces here as a panic instead of silently
                // mapping onto `Backpressure` (unarmed) semantics.
                _ => unreachable!(
                    "route_request stage-3 filter ruled out non-New/Backpressure variants"
                ),
            };
            let source = match worker_id {
                Some(worker_id) => RouteSource::Routed { worker_id },
                None => RouteSource::RouterBackpressure,
            };
            let guard = provisional_guard.commit(data, response, armed);
            return Ok((
                guard,
                source,
                RouteRequestTimings {
                    stream_connect_duration,
                },
            ));
        }
    }

    Err(anyhow::anyhow!(
        "failed to route request through any KV router{}",
        last_error.map(|err| format!(": {err}")).unwrap_or_default()
    ))
}

fn rotated_instance_ids(mut instance_ids: Vec<u64>) -> Vec<u64> {
    if instance_ids.len() > 1 {
        let offset = (rand::rng().random::<u64>() as usize) % instance_ids.len();
        instance_ids.rotate_left(offset);
    }
    instance_ids
}

fn available_router_instance_ids(router: &dyn RouterGuardClient) -> Vec<u64> {
    rotated_instance_ids(router.available_instance_ids())
}

fn min_replica_available_backpressure(
    requirements: &[MinReplicaAvailable],
) -> Result<Option<(RouterStreamResponse, String)>> {
    for requirement in requirements {
        if requirement.router.available_instance_ids().is_empty() {
            tracing::info!(
                replica_name = %requirement.name,
                endpoint = %requirement.router.endpoint_id(),
                "route_request preflight found no required replicas available"
            );
            return Ok(Some((
                router_backpressure_response()?,
                requirement.name.clone(),
            )));
        }
    }

    Ok(None)
}

/// Return the name of the first `require_available` component with zero
/// available replicas, or `None` when all have at least one replica. Unlike
/// [`min_replica_available_backpressure`] this builds no synthetic guard
/// response -- the route's own guard carries any post-route denial -- so the
/// route phase can check before and after routing without constructing
/// throwaway guards.
fn required_down_name(requirements: &[MinReplicaAvailable]) -> Option<String> {
    for requirement in requirements {
        if requirement.router.available_instance_ids().is_empty() {
            tracing::info!(
                replica_name = %requirement.name,
                endpoint = %requirement.router.endpoint_id(),
                "required-available check found no replicas available"
            );
            return Some(requirement.name.clone());
        }
    }
    None
}

fn router_backpressure_response() -> Result<RouterStreamResponse> {
    let response = RsRouterResponse::Backpressure {
        reason: RouterBackpressureReason::DoNotQueue,
        queued_isl_tokens: 0,
        max_queued_isl_tokens: None,
    };
    let data = to_rmpv_value(&response)?;
    Ok(RouterStreamResponse { data, response })
}

pub(super) fn callback_router_instance_ids(
    router: &dyn RouterGuardClient,
    preferred_instance_id: u64,
) -> Vec<u64> {
    let mut remaining: Vec<u64> = router
        .instance_ids()
        .into_iter()
        .filter(|instance_id| *instance_id != preferred_instance_id)
        .collect();
    remaining = rotated_instance_ids(remaining);

    let mut instance_ids = Vec::with_capacity(remaining.len() + 1);
    instance_ids.push(preferred_instance_id);
    instance_ids.extend(remaining);
    instance_ids
}

pub(super) async fn first_stream_response(
    mut stream: EngineStream<RsAnnotated<rmpv::Value>>,
) -> Result<RouterStreamResponse> {
    let response = stream
        .next()
        .await
        .ok_or_else(|| anyhow::anyhow!("router response stream ended before data"))?
        .ok()
        .map_err(|e| anyhow::anyhow!(e))?;

    let data = response
        .data
        .ok_or_else(|| anyhow::anyhow!("router response did not contain data"))?;
    let router_response = from_rmpv_value(&data)
        .map_err(|err| anyhow::anyhow!("failed to decode router response {data}: {err}"))?;

    Ok(RouterStreamResponse {
        data,
        response: router_response,
    })
}

/// Run `fut` to completion on a detached background task so a Python
/// cancellation of the awaiting future cannot abort it. If the caller is
/// dropped before receiving (no taker), `fut` still runs to completion and its
/// output is dropped — for an armed guard, that drop fires the
/// (always-detached) cleanup task, i.e. `mark_free`. The caller converts both
/// the shield error (no taker) and the inner future's own error via
/// `.map_err(to_pyerr)?.map_err(to_pyerr)?` when the output is itself a
/// `Result`.
pub async fn shield_to_completion<F, T>(fut: F) -> Result<T, anyhow::Error>
where
    F: std::future::Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    let (otx, orx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let _ = otx.send(fut.await);
    });
    orx.await
        .map_err(|_| anyhow::anyhow!("detached task ended without a receiver (caller cancelled)"))
}

async fn drain_worker_stream_to_completion(
    stream: EngineStream<RsAnnotated<rmpv::Value>>,
    guard: &RouterRequestGuard,
) {
    let mut stream = stream;
    let mut prefill_marked = false;
    while let Some(response) = stream.next().await {
        let is_error = response.is_error();
        if !prefill_marked
            && !is_error
            && response.data.is_some()
            && !should_drop_first_worker_event(&response)
        {
            guard.mark_prefill();
            prefill_marked = true;
        }
        if is_error {
            break;
        }
    }
}

/// Error from the `potential_loads` preflight: either the downstream router
/// could not be reached ([`PotentialLoadsError::Unreachable`]) or it replied
/// with an unexpected `RouterResponse` variant that is not `PotentialLoads`
/// ([`PotentialLoadsError::ProtocolError`]). The coordinator maps both to a
/// [`DeniedRequest`] -- `NextRouterUnreachable` and `ProtocolError`
/// respectively -- so the preflight never silently passes an unexpected
/// condition.
enum PotentialLoadsError {
    Unreachable { error: String },
    ProtocolError { received: String },
}

fn potential_loads_outcome(
    result: &Result<Option<NextRouterBackpressureInfo>, PotentialLoadsError>,
) -> &'static str {
    match result {
        Ok(Some(_)) => "backpressure",
        Ok(None) => "pass",
        Err(PotentialLoadsError::Unreachable { .. }) => "unreachable",
        Err(PotentialLoadsError::ProtocolError { .. }) => "protocol_error",
    }
}

/// Query the downstream `client`'s router (carried on `check.router`) for
/// potential loads (the `potential_loads` method) and evaluate the response
/// against `check`. Returns `Ok(None)` when the preflight passes; `Ok(Some(info))`
/// when the selected load percentile exceeds the configured thresholds (next-router
/// backpressure); `Err(PotentialLoadsError::Unreachable)` when the query
/// itself could not be performed (the coordinator maps this to a
/// `DeniedRequest::NextRouterUnreachable`); `Err(PotentialLoadsError::ProtocolError)`
/// when the decoded response is not `RouterResponse::PotentialLoads` (the
/// coordinator maps this to a `DeniedRequest::ProtocolError`) -- the preflight
/// fails closed rather than treating an unexpected/wrong-protocol shape as a
/// silent pass.
///
/// A threshold of `0` disables that dimension (no limit). Prefill and decode
/// thresholds are measured in tokens; decode is converted to blocks with the
/// coordinator block size before comparing with router-reported
/// `potential_decode_blocks`. `queue_depth` is the router-level `pending_count`.
#[allow(clippy::too_many_arguments)]
async fn query_potential_loads(
    tokens: Vec<u32>,
    block_mm_infos: Option<Vec<Option<BlockExtraInfo>>>,
    context: Option<RequestContext>,
    request_id: &str,
    check: &PotentialLoadsCheck,
    block_size: u32,
    tracing_enabled: bool,
    allow_cancel_routing: bool,
) -> Result<Option<NextRouterBackpressureInfo>, PotentialLoadsError> {
    let started = Instant::now();
    if tracing_enabled {
        tracing::info!(
            request_id = %request_id,
            endpoint = %check.router.endpoint_id(),
            trace_context_available = trace_context_available(&context),
            "query_potential_loads started"
        );
    }

    let result = 'query: {
        let instance_ids = available_router_instance_ids(check.router.as_ref());
        if instance_ids.is_empty() {
            break 'query Err(PotentialLoadsError::Unreachable {
                error: "no router instances available for potential loads query".to_string(),
            });
        }
        let request = RouterRequest::PotentialLoads {
            tokens: tokens.into(),
            block_mm_infos,
            // no caching for this query.
            allow_short_caching: false,
        };
        let request_value =
            match to_rmpv_value(&request).map_err(|e| PotentialLoadsError::Unreachable {
                error: format!("failed to encode potential loads request: {e}"),
            }) {
                Ok(value) => value,
                Err(err) => break 'query Err(err),
            };

        let mut last_error: Option<String> = None;
        for attempt in 0..ROUTER_GUARD_ATTEMPTS {
            if attempt > 0 {
                tokio::time::sleep(ROUTER_GUARD_RETRY_DELAY).await;
            }

            for &instance_id in &instance_ids {
                let (request_ctx, mut cancellation_forwarder) =
                    create_detached_router_request_context(
                        request_value.clone(),
                        &context,
                        request_id,
                        allow_cancel_routing,
                    );
                let span = context
                    .as_ref()
                    .map(|ctx| ctx.direct_span("query_potential_loads", instance_id))
                    .unwrap_or_else(tracing::Span::none);

                let result = async {
                    let stream = check.router.direct(request_ctx, instance_id).await?;
                    if tracing_enabled {
                        tracing::debug!(
                            request_id = %request_id,
                            router_instance_id = instance_id,
                            attempt = attempt + 1,
                            attempts = ROUTER_GUARD_ATTEMPTS,
                            "query_potential_loads router stream opened"
                        );
                    }
                    first_stream_response(stream).await
                }
                .instrument(span);
                tokio::pin!(result);
                let result = tokio::select! {
                    result = &mut result => result,
                    _ = tokio::time::sleep(POTENTIAL_LOADS_SLOW_LOG_THRESHOLD) => {
                        tracing::warn!(
                            request_id = %request_id,
                            router_instance_id = instance_id,
                            attempt = attempt + 1,
                            attempts = ROUTER_GUARD_ATTEMPTS,
                            threshold_ms = POTENTIAL_LOADS_SLOW_LOG_THRESHOLD.as_millis(),
                            "query_potential_loads attempt still pending after expected latency"
                        );
                        result.await
                    }
                };
                abort_cancellation_forwarder(&mut cancellation_forwarder);

                match result {
                    Ok(router_stream_response) => {
                        break 'query evaluate_potential_loads(
                            &router_stream_response.response,
                            check,
                            block_size,
                        );
                    }
                    Err(err) => {
                        last_error = Some(err.to_string());
                        tracing::warn!(
                            request_id = %request_id,
                            router_instance_id = instance_id,
                            attempt = attempt + 1,
                            attempts = ROUTER_GUARD_ATTEMPTS,
                            error = %err,
                            "query_potential_loads attempt failed",
                        );
                    }
                }
            }
        }

        Err(PotentialLoadsError::Unreachable {
            error: format!(
                "failed to query potential loads through any KV router{}",
                last_error.map(|err| format!(": {err}")).unwrap_or_default()
            ),
        })
    };

    let elapsed = started.elapsed();
    let outcome = potential_loads_outcome(&result);
    if elapsed >= POTENTIAL_LOADS_SLOW_LOG_THRESHOLD {
        tracing::warn!(
            request_id = %request_id,
            endpoint = %check.router.endpoint_id(),
            elapsed_ms = elapsed.as_millis(),
            threshold_ms = POTENTIAL_LOADS_SLOW_LOG_THRESHOLD.as_millis(),
            outcome,
            "query_potential_loads exceeded expected latency"
        );
    } else if tracing_enabled {
        tracing::debug!(
            request_id = %request_id,
            endpoint = %check.router.endpoint_id(),
            elapsed_ms = elapsed.as_millis(),
            outcome,
            "query_potential_loads completed"
        );
    }
    result
}

/// Evaluate a decoded `RouterResponse::PotentialLoads` against the check's
/// thresholds. Returns `Ok(Some(info))` when any enabled threshold is exceeded
/// (next-router backpressure) and `Ok(None)` when the response is
/// `PotentialLoads` but no threshold is exceeded. Returns
/// `Err(PotentialLoadsError::ProtocolError)` when the response is NOT
/// `PotentialLoads` -- the preflight fails closed: a `Backpressure`, `New`, or
/// older/wrong-protocol shape is surfaced as a denial rather than silently
/// passing the overload check.
fn evaluate_potential_loads(
    response: &RsRouterResponse,
    check: &PotentialLoadsCheck,
    block_size: u32,
) -> Result<Option<NextRouterBackpressureInfo>, PotentialLoadsError> {
    let RsRouterResponse::PotentialLoads {
        loads,
        pending_count,
        pending_isl_tokens,
    } = response
    else {
        return Err(PotentialLoadsError::ProtocolError {
            received: format!("{response:?}"),
        });
    };
    let prefill_tokens = percentile_load(
        loads.iter().map(|l| l.potential_prefill_tokens),
        check.load_percentile,
    );
    let decode_blocks = percentile_load(
        loads.iter().map(|l| l.potential_decode_blocks),
        check.load_percentile,
    );
    let queue_depth = *pending_count;
    let pending_isl = *pending_isl_tokens;
    let decode_blocks_threshold =
        decode_tokens_threshold_to_blocks(check.decode_tokens_threshold, block_size);

    let prefill_exceeded =
        check.prefill_tokens_threshold != 0 && prefill_tokens > check.prefill_tokens_threshold;
    let decode_exceeded = decode_blocks_threshold != 0 && decode_blocks > decode_blocks_threshold;
    let queue_exceeded =
        check.queue_depth_threshold != 0 && queue_depth > check.queue_depth_threshold;

    if prefill_exceeded || decode_exceeded || queue_exceeded {
        Ok(Some(NextRouterBackpressureInfo {
            queue_depth,
            pending_isl_tokens: pending_isl,
            prefill_tokens,
            decode_blocks,
        }))
    } else {
        Ok(None)
    }
}

fn decode_tokens_threshold_to_blocks(decode_tokens_threshold: usize, block_size: u32) -> usize {
    if decode_tokens_threshold == 0 {
        return 0;
    }

    let block_size = block_size as usize;
    debug_assert_ne!(block_size, 0, "block_size must be positive");
    if block_size == 0 {
        return decode_tokens_threshold;
    }

    ((decode_tokens_threshold - 1) / block_size) + 1
}

fn percentile_load(values: impl Iterator<Item = usize>, percentile: f64) -> usize {
    let mut values = values.collect::<Vec<_>>();
    let percentile = normalize_load_percentile(percentile);
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    if values.len() == 1 {
        return values[0];
    }

    let rank = percentile * (values.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        return values[lower];
    }

    let weight = rank - lower as f64;
    let interpolated = values[lower] as f64 * (1.0 - weight) + values[upper] as f64 * weight;
    interpolated.ceil() as usize
}

fn normalize_load_percentile(percentile: f64) -> f64 {
    if !percentile.is_finite() {
        tracing::warn!(
            load_percentile = ?percentile,
            normalized_load_percentile = 0.5,
            "potential_loads load_percentile must be finite and between 0.0 and 1.0; using p50"
        );
        return 0.5;
    }

    if !(0.0..=1.0).contains(&percentile) {
        let normalized = percentile.clamp(0.0, 1.0);
        tracing::warn!(
            load_percentile = percentile,
            normalized_load_percentile = normalized,
            "potential_loads load_percentile must be between 0.0 and 1.0; clamping"
        );
        return normalized;
    }

    percentile
}

/// Render a [`RouterBackpressureReason`] as its snake_case reason name.
fn reason_to_string(reason: &RouterBackpressureReason) -> &'static str {
    match reason {
        RouterBackpressureReason::MaxQueuedIslTokensExceeded => "max_queued_isl_tokens_exceeded",
        RouterBackpressureReason::DoNotQueue => "do_not_queue",
    }
}

/// Outcome of one `route_once` attempt: either ready to connect to the chosen
/// worker (armed guard + worker id) or a denial. Every armed-but-denied path
/// drops the guard before returning so the router is never left behind
/// un-prefilled.
enum RouteOnceOutcome {
    Route {
        guard: RouterRequestGuard,
        worker_id: u64,
        timings: RouteRequestTimings,
    },
    Denied(DeniedRequest),
}

/// One attempt of the route phase: check `require_available`, run the optional
/// next-router potential-loads preflight, then issue the KV-router `new` request
/// only if those pre-route checks pass. The potential-loads check is deliberately
/// sequential: a downstream overload denial must not leave a fresh routed request
/// sitting in the routing router's scheduler while this coordinator races to free
/// it. A short post-route `require_available` re-check still runs because a
/// component can go down while the route is in flight. Every armed-but-denied
/// path frees the guard first.
#[allow(clippy::too_many_arguments)]
async fn route_once(
    router_guard_client: Arc<dyn RouterGuardClient>,
    routing_request: Arc<rmpv::Value>,
    request_id: String,
    context: Option<RequestContext>,
    require: Vec<MinReplicaAvailable>,
    preflight: Option<PreflightInputs>,
    block_size: u32,
    notify_timeout: Duration,
    tracing_enabled: bool,
    allow_cancel_routing: bool,
) -> RouteOnceOutcome {
    log_route_step(tracing_enabled, &context, &request_id, "route_once_start");

    if let Some(denied) = cancellation_denial_for_optional_context(&context, allow_cancel_routing) {
        tracing::info!(
            request_id = %request_id,
            "route_once denied before routing because context was cancelled"
        );
        return RouteOnceOutcome::Denied(denied);
    }

    // Short-circuit when a `require_available` component is already down before
    // any downstream network call.
    if let Some(name) = required_down_name(&require) {
        tracing::info!(
            request_id = %request_id,
            replica_name = %name,
            "route_once denied before routing because a required component is down"
        );
        return RouteOnceOutcome::Denied(DeniedRequest::RequiredComponentsDown { name });
    }

    if let Some(pf) = preflight {
        let PreflightInputs {
            check,
            tokens,
            block_mm_infos,
        } = pf;
        log_route_step(
            tracing_enabled,
            &context,
            &request_id,
            "potential_loads_preflight_start",
        );
        let preflight_result = query_potential_loads(
            tokens,
            block_mm_infos,
            context.clone(),
            &request_id,
            &check,
            block_size,
            tracing_enabled,
            allow_cancel_routing,
        )
        .await;

        if let Some(denied) =
            cancellation_denial_for_optional_context(&context, allow_cancel_routing)
        {
            tracing::info!(
                request_id = %request_id,
                "route_once denied after potential_loads because context was cancelled"
            );
            return RouteOnceOutcome::Denied(denied);
        }

        match preflight_result {
            Ok(Some(info)) => {
                tracing::info!(
                    request_id = %request_id,
                    queue_depth = info.queue_depth,
                    pending_isl_tokens = info.pending_isl_tokens,
                    total_prefill_tokens = info.prefill_tokens,
                    total_decode_blocks = info.decode_blocks,
                    "route_once denied before routing by potential_loads preflight"
                );
                return RouteOnceOutcome::Denied(DeniedRequest::NextRouterBackpressure {
                    queue_depth: info.queue_depth,
                    pending_isl_tokens: info.pending_isl_tokens,
                    total_prefill_tokens: info.prefill_tokens,
                    total_decode_blocks: info.decode_blocks,
                });
            }
            Ok(None) => {
                log_route_step(
                    tracing_enabled,
                    &context,
                    &request_id,
                    "potential_loads_preflight_passed",
                );
            }
            Err(PotentialLoadsError::ProtocolError { received }) => {
                tracing::warn!(
                    request_id = %request_id,
                    received = %received,
                    "route_once denied before routing by malformed potential_loads response"
                );
                return RouteOnceOutcome::Denied(DeniedRequest::ProtocolError { received });
            }
            Err(PotentialLoadsError::Unreachable { error }) => {
                tracing::warn!(
                    request_id = %request_id,
                    error = %error,
                    "route_once denied before routing because potential_loads was unreachable"
                );
                return RouteOnceOutcome::Denied(DeniedRequest::NextRouterUnreachable { error });
            }
        }
    }

    log_route_step(
        tracing_enabled,
        &context,
        &request_id,
        "route_request_start",
    );
    let route_res = route_request(
        router_guard_client,
        routing_request,
        request_id.clone(),
        context.clone(),
        Vec::new(),
        notify_timeout,
        tracing_enabled,
        allow_cancel_routing,
    )
    .await;

    match route_res {
        Ok((guard, source, timings)) => {
            if let Some(denied) =
                cancellation_denial_for_optional_context(&context, allow_cancel_routing)
            {
                guard.mark_free();
                drop(guard);
                tracing::info!(
                    request_id = %request_id,
                    "route_once denied after route_request because context was cancelled"
                );
                return RouteOnceOutcome::Denied(denied);
            }

            match source {
                RouteSource::Routed { worker_id } => {
                    log_route_step(
                        tracing_enabled,
                        &context,
                        &request_id,
                        "route_request_admitted",
                    );
                    // Post-route required re-check: a component may have gone down while
                    // the route was in flight.
                    if let Some(name) = required_down_name(&require) {
                        guard.mark_free();
                        drop(guard);
                        return RouteOnceOutcome::Denied(DeniedRequest::RequiredComponentsDown {
                            name,
                        });
                    }
                    RouteOnceOutcome::Route {
                        guard,
                        worker_id,
                        timings,
                    }
                }
                RouteSource::RouterBackpressure => {
                    let (reason, queued_isl_tokens, max_queued_isl_tokens) = guard
                        .backpressure_fields()
                        .unwrap_or((RouterBackpressureReason::DoNotQueue, 0, None));
                    drop(guard);
                    RouteOnceOutcome::Denied(DeniedRequest::RouterBackpressure {
                        reason: reason_to_string(&reason).to_string(),
                        queued_isl_tokens,
                        max_queued_isl_tokens,
                    })
                }
                // Unreachable in production: route_request is called with an empty
                // require list, so it can never return RequiredDown. Defended anyway.
                RouteSource::RequiredDown { name } => {
                    drop(guard);
                    RouteOnceOutcome::Denied(DeniedRequest::RequiredComponentsDown { name })
                }
                // The router replied with a variant that is not a clean admit
                // (`New`) or clean denial (`Backpressure`) for a `new` request.
                // `route_request` already failed closed (requested mark_free, dropped
                // the provisional guard) and returned an unarmed placeholder guard.
                // Surface the protocol error to the caller.
                RouteSource::ProtocolError { received } => {
                    drop(guard);
                    RouteOnceOutcome::Denied(DeniedRequest::ProtocolError { received })
                }
            }
        }
        Err(route_err) => {
            if let Some(denied) =
                cancellation_denial_for_optional_context(&context, allow_cancel_routing)
            {
                return RouteOnceOutcome::Denied(denied);
            }
            RouteOnceOutcome::Denied(DeniedRequest::NextRouterUnreachable {
                error: route_err.to_string(),
            })
        }
    }
}

/// Outcome of one [`connect_worker`] attempt: the worker stream opened (armed
/// guard handed back to pack into `AdmittedRequest`), the routed worker is no
/// longer present (`Stale` -- retry the route), setup reached a typed denial
/// (`Denied`), or the open failed for any other reason (`Other` -- raise). The
/// guard is moved INTO `open_fut` so an outer cancellation during a shielded open
/// does not drop the guard
/// mid-setup (which would race `mark_free` with the worker open completing);
/// instead the guard is dropped inside the shielded task (or inside
/// `open_fut`'s frame on cancel) so `mark_free` fires AFTER the open future
/// resolves. The `Stale` arms additionally call `mark_free` +
/// `wait_for_cleanup` before dropping so the stale-retry loop does not
/// re-route while the prior `mark_free` is still in-flight.
enum OpenResult {
    Ok {
        guard: RouterRequestGuard,
        worker_id: u64,
        stream: EngineStream<RsAnnotated<rmpv::Value>>,
        timings: WorkerConnectTimings,
    },
    Stale {
        /// `Some` on the proactive pre-check (staged copy never resolved,
        /// reusable by the retry); `None` post-open (consumed by `direct()`).
        payload: Option<PayloadCopy>,
    },
    Denied(DeniedRequest),
    Other(anyhow::Error),
}

#[derive(Debug, Clone, Copy, Default)]
struct WorkerConnectTimings {
    stream_connect_duration: Duration,
    first_response_duration: Option<Duration>,
    sentinel_event_duration: Option<Duration>,
}

async fn wait_for_first_worker_event(
    stream: &mut EngineStream<RsAnnotated<rmpv::Value>>,
    worker_id: u64,
) -> Result<RsAnnotated<rmpv::Value>> {
    let first = stream
        .next()
        .await
        .ok_or_else(|| anyhow::anyhow!("worker stream ended before first event"))?;

    let first = first
        .ok()
        .map_err(|err| anyhow::anyhow!("worker stream first event was an error: {err}"))?;

    tracing::debug!(
        worker_id = %worker_id,
        "connect_worker: observed first worker stream event during setup"
    );
    Ok(first)
}

pub fn should_drop_first_worker_event(event: &RsAnnotated<rmpv::Value>) -> bool {
    event
        .data
        .as_ref()
        .map(|data| data[DROP_THIS_MESSAGE_KEY].as_bool())
        .and_then(|value| value)
        .unwrap_or(false)
}

/// Mark prefill on the first real worker response while preserving the stream.
pub fn stream_with_optional_prefill_mark(
    stream: EngineStream<RsAnnotated<rmpv::Value>>,
    guard: Arc<RouterRequestGuard>,
    mark_prefill_on_response: bool,
) -> EngineStream<RsAnnotated<rmpv::Value>> {
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

fn prepend_first_worker_event(
    first: RsAnnotated<rmpv::Value>,
    stream: EngineStream<RsAnnotated<rmpv::Value>>,
) -> EngineStream<RsAnnotated<rmpv::Value>> {
    let stream_context = stream.context();
    let replay_stream = futures::stream::once(async move { first }).chain(stream);
    ResponseStream::new(Box::pin(replay_stream), stream_context)
}

/// Open the worker generation stream on the routed `worker_id`. A route is
/// *stale* when the worker is absent from the worker informer's instance set
/// (its etcd entry was removed after the router chose it); that is detected
/// both proactively (before the open, to skip the network round-trip) and
/// reactively (when `.direct()` returns an error and the worker has since
/// vanished). A stale route returns [`OpenResult::Stale`] so the loop
/// re-routes, carrying the staged `payload` back when the pre-check fired
/// before it was resolved. Any other open error returns [`OpenResult::Other`] so it is
/// raised. The armed `guard` is moved INTO `open_fut` (the proactive pre-check
/// alone does not move the guard because it returns before `open_fut` is
/// constructed) so an outer cancellation during a shielded open drops the
/// guard inside the shielded task AFTER the open future resolves -- avoiding
/// a race where `mark_free` fires while the worker open is still in-flight
/// and could succeed. The `Stale` arms call `mark_free` +
/// `wait_for_cleanup(ROUTER_GUARD_CALLBACK_TIMEOUT)` before dropping so the
/// stale-retry loop does not re-route while the prior `mark_free` is still
/// in-flight.
///
/// When `allow_cancel_setup` is false the open is detached via
/// [`shield_to_completion`] so a Python cancellation cannot abort the setup,
/// payload-copy resolution and `router_response` injection included.
/// When `wait_for_first_response` is true, setup includes waiting for the first
/// non-error worker stream event. The event is dropped only when it carries the
/// drop-message sentinel; otherwise it is prepended back onto the returned stream.
#[allow(clippy::too_many_arguments)]
async fn connect_worker(
    worker_guard_client: Arc<dyn RouterGuardClient>,
    worker_id: u64,
    guard: RouterRequestGuard,
    payload: PayloadCopy,
    request_id: String,
    phase: Option<String>,
    context: RequestContext,
    allow_cancel_setup: bool,
    wait_for_first_response: bool,
) -> OpenResult {
    if !worker_guard_client.instance_ids().contains(&worker_id) {
        tracing::info!(
            worker_id = %worker_id,
            "connect_worker: routed worker not in worker instance set (stale route)"
        );
        // Stale pre-check (before any open): free the guard now and wait for
        // the cleanup task to finish so the subsequent re-route does not race
        // the prior mark_free. The unresolved copy goes back for the retry.
        guard.mark_free();
        guard.wait_for_cleanup(ROUTER_GUARD_CALLBACK_TIMEOUT).await;
        drop(guard);
        return OpenResult::Stale {
            payload: Some(payload),
        };
    }

    let span = context.direct_span("route_and_worker", worker_id);
    let open_ctx = context.clone();
    let wgc = worker_guard_client.clone();
    // The guard is MOVED into `open_fut` so an outer cancellation during a
    // shielded open drops the guard inside the shielded task AFTER the open
    // future resolves. The proactive stale pre-check above returns before
    // `open_fut` is constructed, so it does not move the guard.
    let open_fut = async move {
        // Payload resolution is part of setup: it must sit inside the shield
        // so a detached-setup policy admits no cancellation point between
        // route admission and the worker open.
        let mut worker_request = match payload.finish(&request_id, phase.as_deref()).await {
            Ok(value) => value,
            Err(err) => {
                drop(guard);
                return OpenResult::Other(err);
            }
        };
        // Inject the per-route `RouterResponse::New` so the worker (or a
        // further forwarder) sees the routing decision.
        match &mut worker_request {
            rmpv::Value::Map(map) => {
                map.retain(|(k, _)| k.as_str() != Some("router_response"));
                map.push((
                    rmpv::Value::from("router_response"),
                    guard.new_response().clone(),
                ));
            }
            _ => {
                drop(guard);
                return OpenResult::Other(anyhow::anyhow!(
                    "worker_args must be a JSON object so the router response can be added as the `router_response` field"
                ));
            }
        }
        let worker_request_ctx =
            create_worker_request_context(worker_request, &open_ctx, allow_cancel_setup);
        let stream_connect_started = Instant::now();
        let stream_result = wgc
            .direct(worker_request_ctx, worker_id)
            .instrument(span)
            .await;
        let mut timings = WorkerConnectTimings {
            stream_connect_duration: stream_connect_started.elapsed(),
            first_response_duration: None,
            sentinel_event_duration: None,
        };
        match stream_result {
            Ok(mut stream) => {
                if wait_for_first_response {
                    let first_response_started = Instant::now();
                    let first = match wait_for_first_worker_event(&mut stream, worker_id).await {
                        Ok(first) => first,
                        Err(err) => {
                            let stable_routing_id = wgc.stable_routing_id(worker_id);
                            tracing::warn!(
                                worker_id = %worker_id,
                                stable_routing_id = stable_routing_id.as_deref().unwrap_or("unavailable"),
                                error = %err,
                                "connect_worker: failed while waiting for first worker stream event"
                            );
                            drop(guard);
                            return OpenResult::Denied(DeniedRequest::FirstWorkerEventFailed {
                                error: err.to_string(),
                            });
                        }
                    };
                    let first_event_duration = first_response_started.elapsed();

                    if should_drop_first_worker_event(&first) {
                        timings.sentinel_event_duration = Some(first_event_duration);
                        tracing::debug!(
                            worker_id = %worker_id,
                            "connect_worker: swallowed first worker stream sentinel"
                        );
                    } else {
                        timings.first_response_duration = Some(first_event_duration);
                        stream = prepend_first_worker_event(first, stream);
                    }
                }
                OpenResult::Ok {
                    guard,
                    worker_id,
                    stream,
                    timings,
                }
            }
            Err(err) => {
                let stable_routing_id = wgc.stable_routing_id(worker_id);
                if wgc.instance_ids().contains(&worker_id) {
                    tracing::warn!(
                        worker_id = %worker_id,
                        stable_routing_id = stable_routing_id.as_deref().unwrap_or("unavailable"),
                        error = %err,
                        "connect_worker: worker open failed (non-stale)"
                    );
                    // No re-route happens on `Other`; drop the guard without
                    // waiting for cleanup so the caller's error path is not
                    // blocked on the (best-effort) mark_free round-trip.
                    drop(guard);
                    OpenResult::Other(err)
                } else {
                    tracing::info!(
                        worker_id = %worker_id,
                        stable_routing_id = stable_routing_id.as_deref().unwrap_or("unavailable"),
                        error = %err,
                        "connect_worker: open failed and worker now absent (stale route)"
                    );
                    // Stale post-open: free + wait as above. The resolved
                    // copy is gone; the retry stages afresh.
                    guard.mark_free();
                    guard.wait_for_cleanup(ROUTER_GUARD_CALLBACK_TIMEOUT).await;
                    drop(guard);
                    OpenResult::Stale { payload: None }
                }
            }
        }
    };

    if allow_cancel_setup {
        open_fut.await
    } else {
        // `shield_to_completion` returns Err only when the spawn itself fails
        // (no-taker path is unreachable: the detached task runs to completion
        // and its output is dropped, firing mark_free via the guard's Drop).
        // Defensively treat any shield error as `Other` so it is raised.
        match shield_to_completion(open_fut).await {
            Ok(inner) => inner,
            Err(shield_err) => OpenResult::Other(shield_err),
        }
    }
}

/// Outcome of the `route_and_connect` loop: either the worker stream opened
/// (ready to hand back as an `AdmittedRequest`) or a denial. A non-stale open
/// failure propagates as `Err` from the loop and is raised by the caller.
#[allow(clippy::large_enum_variant)]
pub enum RouteAndConnectOutcome {
    Connected {
        guard: RouterRequestGuard,
        worker_id: u64,
        stream: EngineStream<RsAnnotated<rmpv::Value>>,
        timings: AdmittedRequestTimings,
    },
    Denied(DeniedRequest),
}

impl std::fmt::Debug for RouteAndConnectOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RouteAndConnectOutcome::Connected { worker_id, .. } => f
                .debug_struct("Connected")
                .field("worker_id", worker_id)
                .finish_non_exhaustive(),
            RouteAndConnectOutcome::Denied(d) => f.debug_tuple("Denied").field(d).finish(),
        }
    }
}

/// Run the route+connect loop to completion on a detached task so a Python
/// cancellation of the awaiting future cannot abort routing/setup. Unlike the
/// generic [`shield_to_completion`], this helper owns the no-taker case for a
/// successful worker connection: when the awaiting side is gone before the
/// `Connected` outcome can be handed back, it drains the connected worker stream
/// before dropping the guard. That keeps a detached request from being aborted
/// exactly at the route/setup -> stream handoff.
pub async fn shield_route_and_connect<F>(fut: F) -> Result<RouteAndConnectOutcome>
where
    F: std::future::Future<Output = Result<RouteAndConnectOutcome>> + Send + 'static,
{
    let (otx, orx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        if let Err(result) = otx.send(fut.await) {
            match result {
                Ok(RouteAndConnectOutcome::Connected {
                    guard,
                    worker_id,
                    stream,
                    ..
                }) => {
                    tracing::info!(
                        worker_id = %worker_id,
                        "route_and_connect shield completed after caller cancelled; \
                         draining connected worker stream before freeing guard"
                    );
                    drain_worker_stream_to_completion(stream, &guard).await;
                    drop(guard);
                }
                Ok(RouteAndConnectOutcome::Denied(_)) => {}
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        "route_and_connect shield failed after caller cancelled"
                    );
                }
            }
        }
    });
    orx.await.map_err(|_| {
        anyhow::anyhow!("detached route task ended without a receiver (caller cancelled)")
    })?
}

/// The route + connect lifecycle engine: repeatedly `route_once` then
/// `connect_worker` until either the worker stream opens or the route is
/// denied. A stale route (the chosen worker is no longer present) re-routes,
/// bounded by `max_reroutes` (the initial attempt plus up to `max_reroutes`
/// reroutes); exhausting the bound returns
/// `DeniedRequest::NextRouterUnreachable { "stale route loop exhausted" }`.
/// The preflight runs only on the first attempt (a stale reroute does not change
/// the downstream router's reported loads). Each attempt sends a copy of
/// `worker_request` with the per-route `RouterResponse::New` injected under
/// the `router_response` field so the worker sees `worker_id` / `dp_rank` /
/// `overlap_blocks` / `dp_strict_rank`; see `payload_copy.rs` for when that
/// copy runs and what a denied route costs. `block_size` must match the routed KV
/// router so the admitted log can derive token-level overlap estimates from
/// `overlap_blocks`. The whole loop is run under the routing cancellation shield
/// by the caller.
#[allow(clippy::too_many_arguments)]
pub async fn route_and_connect(
    router_guard_client: Arc<dyn RouterGuardClient>,
    worker_guard_client: Arc<dyn RouterGuardClient>,
    routing_request: Arc<rmpv::Value>,
    request_id: String,
    context: RequestContext,
    require: Vec<MinReplicaAvailable>,
    mut preflight_inputs: Option<PreflightInputs>,
    worker_request: rmpv::Value,
    block_size: u32,
    max_reroutes: u64,
    allow_cancel_routing: bool,
    allow_cancel_setup: bool,
    wait_for_first_response: bool,
    notify_timeout: Duration,
    tracing_enabled: bool,
    phase: Option<String>,
) -> Result<RouteAndConnectOutcome> {
    let started = Instant::now();
    let mut attempt: u64 = 0;
    // The base payload is never mutated; `spare` holds a staged copy handed
    // back by a stale pre-check so that retry does not stage again.
    let worker_request = Arc::new(worker_request);
    let mut spare: Option<PayloadCopy> = None;
    loop {
        let preflight = if attempt == 0 {
            preflight_inputs.take()
        } else {
            None
        };
        // Stage the payload copy, then route; it resolves inside
        // `connect_worker`'s setup once the route is usable, and a denial
        // abandons it without waiting (policy in payload_copy.rs).
        let copy = spare
            .take()
            .unwrap_or_else(|| PayloadCopy::stage(Arc::clone(&worker_request)));
        let route_outcome = route_once(
            router_guard_client.clone(),
            routing_request.clone(),
            request_id.clone(),
            Some(context.clone()),
            require.clone(),
            preflight,
            block_size,
            notify_timeout,
            tracing_enabled,
            allow_cancel_routing,
        )
        .await;
        let routing_new_returned_at = Instant::now();

        let (guard, worker_id, route_timings) = match route_outcome {
            RouteOnceOutcome::Denied(denied) => {
                copy.abandon();
                log_route_and_connect_denied(&request_id, phase.as_deref(), None, attempt, &denied);
                return Ok(RouteAndConnectOutcome::Denied(denied));
            }
            RouteOnceOutcome::Route {
                guard,
                worker_id,
                timings,
            } => (guard, worker_id, timings),
        };

        match connect_worker(
            worker_guard_client.clone(),
            worker_id,
            guard,
            copy,
            request_id.clone(),
            phase.clone(),
            context.clone(),
            allow_cancel_setup,
            wait_for_first_response,
        )
        .await
        {
            OpenResult::Ok {
                guard,
                worker_id,
                stream,
                timings: connect_timings,
            } => {
                if let Some(denied) = cancellation_denial_for_context(&context, allow_cancel_setup)
                {
                    guard.mark_free();
                    drop(stream);
                    drop(guard);
                    tracing::info!(
                        request_id = %request_id,
                        worker_id = %worker_id,
                        "route_and_connect denied after worker setup because context was cancelled"
                    );
                    log_route_and_connect_denied(
                        &request_id,
                        phase.as_deref(),
                        Some(worker_id),
                        attempt,
                        &denied,
                    );
                    return Ok(RouteAndConnectOutcome::Denied(denied));
                }

                let worker_connected_at = Instant::now();
                let estimated_overlap_tokens = guard.estimated_overlap_tokens(block_size);
                let timings = AdmittedRequestTimings {
                    routing_new_duration: routing_new_returned_at.duration_since(started),
                    routing_stream_connect_duration: route_timings.stream_connect_duration,
                    worker_stream_connect_duration: connect_timings.stream_connect_duration,
                    worker_first_response_duration: connect_timings.first_response_duration,
                    worker_sentinel_event_duration: connect_timings.sentinel_event_duration,
                    worker_connect_duration: worker_connected_at
                        .duration_since(routing_new_returned_at),
                    stale_reroutes: attempt,
                };
                tracing::info!(
                    request_id = %request_id,
                    worker_id = %worker_id,
                    phase = phase.as_deref().unwrap_or("unknown"),
                    stale_reroutes = attempt,
                    routing_new_ms = duration_ms_for_log(timings.routing_new_duration),
                    routing_connect_ms = duration_ms_for_log(timings.routing_stream_connect_duration),
                    worker_connect_ms = duration_ms_for_log(timings.worker_stream_connect_duration),
                    worker_first_ms = timings
                        .worker_first_response_duration
                        .map(duration_ms_for_log),
                    worker_sentinel_ms = timings
                        .worker_sentinel_event_duration
                        .map(duration_ms_for_log),
                    worker_setup_ms = duration_ms_for_log(timings.worker_connect_duration),
                    estimated_overlap_tokens,
                    "route_and_connect admitted"
                );
                return Ok(RouteAndConnectOutcome::Connected {
                    guard,
                    worker_id,
                    stream,
                    timings,
                });
            }
            OpenResult::Stale { payload: returned } => {
                // The guard was freed + waited-for-cleanup inside `connect_worker`.
                if let Some(denied) = cancellation_denial_for_context(&context, allow_cancel_setup)
                    .or_else(|| cancellation_denial_for_context(&context, allow_cancel_routing))
                {
                    log_route_and_connect_denied(
                        &request_id,
                        phase.as_deref(),
                        Some(worker_id),
                        attempt,
                        &denied,
                    );
                    return Ok(RouteAndConnectOutcome::Denied(denied));
                }
                if attempt >= max_reroutes {
                    let denied = DeniedRequest::NextRouterUnreachable {
                        error: "stale route loop exhausted".to_string(),
                    };
                    log_route_and_connect_denied(
                        &request_id,
                        phase.as_deref(),
                        Some(worker_id),
                        attempt,
                        &denied,
                    );
                    return Ok(RouteAndConnectOutcome::Denied(denied));
                }
                // Pre-check stale hands the staged copy back; post-open
                // stale consumed it and the retry stages afresh.
                spare = returned;
                // The guard cleanup wait above observes the mark_free task reaching
                // terminal state. Keep a short grace period before reusing the
                // request id for a fresh route so router-side free processing is
                // much more likely to be visible to the next attempt.
                tokio::time::sleep(ROUTER_GUARD_CLEANUP_GRACE_PERIOD).await;
                attempt += 1;
                continue;
            }
            OpenResult::Denied(denied) => {
                if let Some(cancelled) =
                    cancellation_denial_for_context(&context, allow_cancel_setup)
                {
                    log_route_and_connect_denied(
                        &request_id,
                        phase.as_deref(),
                        Some(worker_id),
                        attempt,
                        &cancelled,
                    );
                    return Ok(RouteAndConnectOutcome::Denied(cancelled));
                }
                log_route_and_connect_denied(
                    &request_id,
                    phase.as_deref(),
                    Some(worker_id),
                    attempt,
                    &denied,
                );
                return Ok(RouteAndConnectOutcome::Denied(denied));
            }
            OpenResult::Other(err) => {
                if let Some(cancelled) =
                    cancellation_denial_for_context(&context, allow_cancel_setup)
                {
                    log_route_and_connect_denied(
                        &request_id,
                        phase.as_deref(),
                        Some(worker_id),
                        attempt,
                        &cancelled,
                    );
                    return Ok(RouteAndConnectOutcome::Denied(cancelled));
                }
                return Err(err);
            }
        }
    }
}
