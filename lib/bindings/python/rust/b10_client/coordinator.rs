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

use crate::{context, create_request_context, get_span_for_direct_context, process_stream};
use anyhow::Result;
use dynamo_kv_router::protocols::{
    BlockExtraInfo, RouterBackpressureReason, RouterRequest, RouterResponse as RsRouterResponse,
};
use dynamo_runtime::pipeline::{
    AsyncEngineContextProvider, EngineStream, PushRouter, ResponseStream, async_trait,
    context::Context as RsContext,
};
use dynamo_runtime::protocols::annotated::Annotated as RsAnnotated;
use futures::StreamExt;
use pyo3::PyObject;
use rand::Rng;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::Instrument;

use super::DROP_THIS_MESSAGE_KEY;
use super::guard::{
    ROUTER_GUARD_ATTEMPTS, ROUTER_GUARD_CALLBACK_TIMEOUT, ROUTER_GUARD_CLEANUP_GRACE_PERIOD,
    ROUTER_GUARD_RETRY_DELAY, RouterRequestGuard,
};
use super::types::{
    AdmittedRequestTimings, DeniedRequest, MinReplicaAvailable, NextRouterBackpressureInfo,
    PotentialLoadsCheckData, PreflightInputs,
};

/// JSON-typed push router used to talk to KV router instances.
///
/// On v1.2.0 the Python `Client` pyclass holds a `PushRouter<serde_json::Value,
/// RsAnnotated<serde_json::Value>>` plus a separate `endpoint` handle; this
/// alias names that router type used throughout the b10_client coordinator.
type JsonPushRouter = PushRouter<serde_json::Value, RsAnnotated<serde_json::Value>>;

const POTENTIAL_LOADS_SLOW_LOG_THRESHOLD: Duration = Duration::from_secs(1);
const ROUTE_STREAM_OPEN_SLOW_LOG_THRESHOLD: Duration = Duration::from_secs(1);
const ROUTE_FIRST_RESPONSE_SLOW_LOG_THRESHOLD: Duration = Duration::from_secs(1);
const ROUTE_FIRST_RESPONSE_TIMEOUT: Duration = Duration::from_secs(590);
const DURATION_LOG_MS_PRECISION: f64 = 1_000.0;

fn duration_ms_for_log(duration: Duration) -> f64 {
    let duration_ms = duration.as_secs_f64() * 1000.0;
    (duration_ms * DURATION_LOG_MS_PRECISION).round() / DURATION_LOG_MS_PRECISION
}

fn create_detached_router_request_context(
    request: serde_json::Value,
    parent_ctx: &Option<context::Context>,
    request_id: &str,
    follow_parent_cancellation: bool,
) -> (
    RsContext<serde_json::Value>,
    Option<tokio::task::JoinHandle<()>>,
) {
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

fn trace_context_available(context: &Option<context::Context>) -> bool {
    context
        .as_ref()
        .and_then(|context| context.trace_context())
        .is_some()
}

fn log_route_step(
    enabled: bool,
    context: &Option<context::Context>,
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

fn cancellation_denial_for_context(
    context: &context::Context,
    allow: bool,
) -> Option<DeniedRequest> {
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
    context: &Option<context::Context>,
    allow: bool,
) -> Option<DeniedRequest> {
    context
        .as_ref()
        .and_then(|ctx| cancellation_denial_for_context(ctx, allow))
}

fn create_worker_request_context(
    request: serde_json::Value,
    parent_ctx: &context::Context,
    follow_parent_during_setup: bool,
) -> RsContext<serde_json::Value> {
    if follow_parent_during_setup {
        create_request_context(request, &Some(parent_ctx.clone()))
    } else {
        RsContext::with_id_and_metadata(
            request,
            parent_ctx.inner().id().to_string(),
            parent_ctx.metadata_snapshot(),
        )
    }
}

/// Why a [`route_request`] call resolved the way it did. The coordinator maps
/// these onto `DeniedRequest::RouterBackpressure` /
/// `DeniedRequest::RequiredComponentsDown` /
/// `DeniedRequest::ProtocolError`.
pub(super) enum RouteSource {
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

pub(super) struct RouterStreamResponse {
    data: serde_json::Value,
    pub(super) response: RsRouterResponse,
}

#[async_trait]
pub(super) trait RouterGuardClient: Send + Sync {
    fn endpoint_id(&self) -> String;

    fn available_instance_ids(&self) -> Vec<u64>;

    fn instance_ids(&self) -> Vec<u64>;

    async fn direct(
        &self,
        request: RsContext<serde_json::Value>,
        instance_id: u64,
    ) -> Result<EngineStream<RsAnnotated<serde_json::Value>>>;
}

#[derive(Clone)]
pub(super) struct JsonRouterGuardClient {
    router: JsonPushRouter,
}

impl JsonRouterGuardClient {
    pub(super) fn new(router: JsonPushRouter) -> Self {
        Self { router }
    }
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

    async fn direct(
        &self,
        request: RsContext<serde_json::Value>,
        instance_id: u64,
    ) -> Result<EngineStream<RsAnnotated<serde_json::Value>>> {
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
pub(super) async fn route_request(
    router: Arc<dyn RouterGuardClient>,
    request: serde_json::Value,
    request_id: String,
    context: Option<context::Context>,
    require_min1_replica_available: Vec<MinReplicaAvailable>,
    notify_timeout: Duration,
    tracing_enabled: bool,
    allow_cancel_routing: bool,
) -> Result<(RouterRequestGuard, RouteSource)> {
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
        return Ok((guard, RouteSource::RequiredDown { name }));
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
        return Ok((guard, RouteSource::RouterBackpressure));
    }

    let mut last_error = None;
    for attempt in 0..ROUTER_GUARD_ATTEMPTS {
        for (instance_index, &instance_id) in instance_ids.iter().enumerate() {
            let has_more_route_attempts =
                attempt + 1 < ROUTER_GUARD_ATTEMPTS || instance_index + 1 < instance_ids.len();
            let (request_ctx, mut cancellation_forwarder) = create_detached_router_request_context(
                request.clone(),
                &context,
                &request_id,
                allow_cancel_routing,
            );
            let route_context = request_ctx.context();
            let span = context
                .as_ref()
                .map(|context| {
                    get_span_for_direct_context(context, "route_request", &instance_id.to_string())
                })
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
                    stream
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
                    return Ok((guard, RouteSource::RouterBackpressure));
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
                return Ok((placeholder, RouteSource::ProtocolError { received }));
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
            return Ok((guard, source));
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
    let data = serde_json::to_value(&response)?;
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
    mut stream: EngineStream<RsAnnotated<serde_json::Value>>,
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
    let router_response = serde_json::from_value(data.clone())
        .map_err(|err| anyhow::anyhow!("failed to decode router response {data}: {err}))"))?;

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
pub(super) async fn shield_to_completion<F, T>(fut: F) -> Result<T, anyhow::Error>
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
    stream: EngineStream<RsAnnotated<serde_json::Value>>,
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

/// Spawn [`process_stream`] for `stream` on a detached task and hand back the
/// channel receiver plus the `guard` and `source` via a oneshot, so a Python
/// cancellation cannot abort the worker generation hand-back. If the caller is
/// dropped before receiving (no taker), the task drains the worker generation
/// to completion (so the server-side work is not interrupted mid-prefill) and
/// then drops the guard, which fires `mark_free`. The caller drives
/// `mark_prefill` / `mark_free` itself once it receives the guard. After the
/// receiver has been handed back, dropping it is treated as consumer
/// cancellation: [`process_stream`] exits on send failure and drops the
/// upstream worker stream.
pub(super) async fn shield_stream_to_completion(
    stream: EngineStream<RsAnnotated<serde_json::Value>>,
    guard: Arc<RouterRequestGuard>,
    source: RouteSource,
) -> Result<
    (
        Arc<RouterRequestGuard>,
        RouteSource,
        tokio::sync::mpsc::Receiver<RsAnnotated<PyObject>>,
    ),
    anyhow::Error,
> {
    let (otx, orx) = tokio::sync::oneshot::channel::<(
        Arc<RouterRequestGuard>,
        RouteSource,
        tokio::sync::mpsc::Receiver<RsAnnotated<PyObject>>,
    )>();
    tokio::spawn(async move {
        let (tx, rx) = tokio::sync::mpsc::channel::<RsAnnotated<PyObject>>(32);
        let guard_for_drain = Arc::clone(&guard);
        tokio::spawn(async move {
            process_stream(stream, tx).await;
            drop(guard_for_drain);
        });
        // No taker: drain the worker generation to completion, then the
        // returned guard drops -> cleanup -> mark_free.
        if let Err((mut _guard, _source, mut rx)) = otx.send((guard, source, rx)) {
            while rx.recv().await.is_some() {}
            drop(_guard);
        }
    });
    orx.await.map_err(|_| {
        anyhow::anyhow!("detached stream task ended without a receiver (caller cancelled)")
    })
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
    context: Option<context::Context>,
    request_id: &str,
    check: &PotentialLoadsCheckData,
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
            tokens,
            block_mm_infos,
        };
        let request_value =
            match serde_json::to_value(&request).map_err(|e| PotentialLoadsError::Unreachable {
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
                    .map(|ctx| {
                        get_span_for_direct_context(
                            ctx,
                            "query_potential_loads",
                            &instance_id.to_string(),
                        )
                    })
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
    check: &PotentialLoadsCheckData,
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

/// Render a [`RouterBackpressureReason`] as its snake_case reason name (the
/// `serde`-serialised form), falling back to the `Debug` form on error.
fn reason_to_string(reason: &RouterBackpressureReason) -> String {
    serde_json::to_value(reason)
        .ok()
        .and_then(|v| v.as_str().map(|s| s.to_string()))
        .unwrap_or_else(|| format!("{reason:?}"))
}

/// Outcome of one `route_once` attempt: either ready to connect to the chosen
/// worker (armed guard + worker id) or a denial. Every armed-but-denied path
/// drops the guard before returning so the router is never left behind
/// un-prefilled.
enum RouteOnceOutcome {
    Route {
        guard: RouterRequestGuard,
        worker_id: u64,
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
    routing_request: serde_json::Value,
    request_id: String,
    context: Option<context::Context>,
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
        Ok((guard, source)) => {
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
                    RouteOnceOutcome::Route { guard, worker_id }
                }
                RouteSource::RouterBackpressure => {
                    let (reason, queued_isl_tokens, max_queued_isl_tokens) = guard
                        .backpressure_fields()
                        .unwrap_or((RouterBackpressureReason::DoNotQueue, 0, None));
                    drop(guard);
                    RouteOnceOutcome::Denied(DeniedRequest::RouterBackpressure {
                        reason: reason_to_string(&reason),
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
        stream: EngineStream<RsAnnotated<serde_json::Value>>,
    },
    Stale,
    Denied(DeniedRequest),
    Other(anyhow::Error),
}

async fn wait_for_first_worker_event(
    stream: &mut EngineStream<RsAnnotated<serde_json::Value>>,
    worker_id: u64,
) -> Result<RsAnnotated<serde_json::Value>> {
    let first = stream
        .next()
        .await
        .ok_or_else(|| anyhow::anyhow!("worker stream ended before first event"))?;

    let first = first
        .ok()
        .map_err(|err| anyhow::anyhow!("worker stream first event was an error: {err}"))?;

    tracing::debug!(
        worker_id,
        "connect_worker: observed first worker stream event during setup"
    );
    Ok(first)
}

pub(super) fn should_drop_first_worker_event(event: &RsAnnotated<serde_json::Value>) -> bool {
    event
        .data
        .as_ref()
        .and_then(|data| data.get(DROP_THIS_MESSAGE_KEY))
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
}

fn prepend_first_worker_event(
    first: RsAnnotated<serde_json::Value>,
    stream: EngineStream<RsAnnotated<serde_json::Value>>,
) -> EngineStream<RsAnnotated<serde_json::Value>> {
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
/// re-routes; any other open error returns [`OpenResult::Other`] so it is
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
/// [`shield_to_completion`] so a Python cancellation cannot abort the setup.
/// When `wait_for_first_response` is true, setup includes waiting for the first
/// non-error worker stream event. The event is dropped only when it carries the
/// drop-message sentinel; otherwise it is prepended back onto the returned stream.
async fn connect_worker(
    worker_guard_client: Arc<dyn RouterGuardClient>,
    worker_id: u64,
    guard: RouterRequestGuard,
    worker_request: serde_json::Value,
    context: context::Context,
    allow_cancel_setup: bool,
    wait_for_first_response: bool,
) -> OpenResult {
    if !worker_guard_client.instance_ids().contains(&worker_id) {
        tracing::info!(
            worker_id,
            "connect_worker: routed worker not in worker instance set (stale route)"
        );
        // Stale pre-check (before any open): free the guard now and wait for
        // the cleanup task to finish so the subsequent re-route does not race
        // the prior mark_free.
        guard.mark_free();
        guard.wait_for_cleanup(ROUTER_GUARD_CALLBACK_TIMEOUT).await;
        drop(guard);
        return OpenResult::Stale;
    }

    let span = get_span_for_direct_context(&context, "route_and_worker", &worker_id.to_string());
    let open_ctx = context.clone();
    let wgc = worker_guard_client.clone();
    // The guard is MOVED into `open_fut` so an outer cancellation during a
    // shielded open drops the guard inside the shielded task AFTER the open
    // future resolves. The proactive stale pre-check above returns before
    // `open_fut` is constructed, so it does not move the guard.
    let open_fut = async move {
        let worker_request_ctx =
            create_worker_request_context(worker_request, &open_ctx, allow_cancel_setup);
        let stream_result = wgc
            .direct(worker_request_ctx, worker_id)
            .instrument(span)
            .await;
        match stream_result {
            Ok(mut stream) => {
                if wait_for_first_response {
                    let first = match wait_for_first_worker_event(&mut stream, worker_id).await {
                        Ok(first) => first,
                        Err(err) => {
                            tracing::warn!(
                                worker_id,
                                error = %err,
                                "connect_worker: failed while waiting for first worker stream event"
                            );
                            drop(guard);
                            return OpenResult::Denied(DeniedRequest::FirstWorkerEventFailed {
                                error: err.to_string(),
                            });
                        }
                    };

                    if should_drop_first_worker_event(&first) {
                        tracing::debug!(
                            worker_id,
                            "connect_worker: swallowed first worker stream sentinel"
                        );
                    } else {
                        stream = prepend_first_worker_event(first, stream);
                    }
                }
                OpenResult::Ok {
                    guard,
                    worker_id,
                    stream,
                }
            }
            Err(err) => {
                if wgc.instance_ids().contains(&worker_id) {
                    tracing::warn!(
                        worker_id,
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
                        worker_id,
                        error = %err,
                        "connect_worker: open failed and worker now absent (stale route)"
                    );
                    // Stale post-open: free + wait so the re-route does not
                    // race the prior mark_free.
                    guard.mark_free();
                    guard.wait_for_cleanup(ROUTER_GUARD_CALLBACK_TIMEOUT).await;
                    drop(guard);
                    OpenResult::Stale
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
pub(super) enum RouteAndConnectOutcome {
    Connected {
        guard: RouterRequestGuard,
        worker_id: u64,
        stream: EngineStream<RsAnnotated<serde_json::Value>>,
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
pub(super) async fn shield_route_and_connect<F>(fut: F) -> Result<RouteAndConnectOutcome>
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
                        worker_id,
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
/// the downstream router's reported loads). Each attempt injects the per-route
/// `RouterResponse::New` into a fresh `worker_request` clone under the
/// `router_response` field so the worker sees `worker_id` / `dp_rank` /
/// `overlap_blocks` / `dp_strict_rank`. `block_size` must match the routed KV
/// router so the admitted log can derive token-level overlap estimates from
/// `overlap_blocks`. The whole loop is run under the routing cancellation shield
/// by the caller.
#[allow(clippy::too_many_arguments)]
pub(super) async fn route_and_connect(
    router_guard_client: Arc<dyn RouterGuardClient>,
    worker_guard_client: Arc<dyn RouterGuardClient>,
    routing_request: serde_json::Value,
    request_id: String,
    context: context::Context,
    require: Vec<MinReplicaAvailable>,
    mut preflight_inputs: Option<PreflightInputs>,
    worker_request: serde_json::Value,
    block_size: u32,
    max_reroutes: u64,
    allow_cancel_routing: bool,
    allow_cancel_setup: bool,
    wait_for_first_response: bool,
    notify_timeout: Duration,
    tracing_enabled: bool,
) -> Result<RouteAndConnectOutcome> {
    let started = Instant::now();
    let mut attempt: u64 = 0;
    loop {
        let preflight = if attempt == 0 {
            preflight_inputs.take()
        } else {
            None
        };
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

        let (guard, worker_id) = match route_outcome {
            RouteOnceOutcome::Denied(denied) => {
                return Ok(RouteAndConnectOutcome::Denied(denied));
            }
            RouteOnceOutcome::Route { guard, worker_id } => (guard, worker_id),
        };

        // Inject the per-route `RouterResponse::New` into a fresh worker-args
        // clone so the worker (or a further forwarder) sees the routing decision.
        let mut req = worker_request.clone();
        match &mut req {
            serde_json::Value::Object(map) => {
                map.insert("router_response".to_string(), guard.new_response().clone());
            }
            _ => {
                drop(guard);
                return Err(anyhow::anyhow!(
                    "worker_args must be a JSON object so the router response can be added as the `router_response` field"
                ));
            }
        }

        match connect_worker(
            worker_guard_client.clone(),
            worker_id,
            guard,
            req,
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
            } => {
                if let Some(denied) = cancellation_denial_for_context(&context, allow_cancel_setup)
                {
                    guard.mark_free();
                    drop(stream);
                    drop(guard);
                    tracing::info!(
                        request_id = %request_id,
                        worker_id,
                        "route_and_connect denied after worker setup because context was cancelled"
                    );
                    return Ok(RouteAndConnectOutcome::Denied(denied));
                }

                let worker_connected_at = Instant::now();
                let estimated_overlap_tokens = guard.estimated_overlap_tokens(block_size);
                let timings = AdmittedRequestTimings {
                    routing_new_duration: routing_new_returned_at.duration_since(started),
                    worker_connect_duration: worker_connected_at
                        .duration_since(routing_new_returned_at),
                    stale_reroutes: attempt,
                };
                tracing::info!(
                    request_id = %request_id,
                    worker_id,
                    stale_reroutes = attempt,
                    routing_new_duration_ms = duration_ms_for_log(timings.routing_new_duration),
                    worker_connect_duration_ms = duration_ms_for_log(timings.worker_connect_duration),
                    estimated_overlap_tokens,
                    unified_logs = true,
                    "route_and_connect admitted"
                );
                return Ok(RouteAndConnectOutcome::Connected {
                    guard,
                    worker_id,
                    stream,
                    timings,
                });
            }
            OpenResult::Stale => {
                // The guard was freed + waited-for-cleanup inside `connect_worker`.
                if let Some(denied) = cancellation_denial_for_context(&context, allow_cancel_setup)
                    .or_else(|| cancellation_denial_for_context(&context, allow_cancel_routing))
                {
                    return Ok(RouteAndConnectOutcome::Denied(denied));
                }
                if attempt >= max_reroutes {
                    return Ok(RouteAndConnectOutcome::Denied(
                        DeniedRequest::NextRouterUnreachable {
                            error: "stale route loop exhausted".to_string(),
                        },
                    ));
                }
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
                    return Ok(RouteAndConnectOutcome::Denied(cancelled));
                }
                return Ok(RouteAndConnectOutcome::Denied(denied));
            }
            OpenResult::Other(err) => {
                if let Some(cancelled) =
                    cancellation_denial_for_context(&context, allow_cancel_setup)
                {
                    return Ok(RouteAndConnectOutcome::Denied(cancelled));
                }
                return Err(err);
            }
        }
    }
}
