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
    EngineStream, PushRouter, async_trait, context::Context as RsContext,
};
use dynamo_runtime::protocols::annotated::Annotated as RsAnnotated;
use futures::StreamExt;
use pyo3::PyObject;
use rand::Rng;
use std::sync::Arc;
use std::time::Duration;
use tracing::Instrument;

use super::guard::{
    ROUTER_GUARD_ATTEMPTS, ROUTER_GUARD_CALLBACK_TIMEOUT, ROUTER_GUARD_RETRY_DELAY,
    RouterRequestGuard,
};
use super::types::{
    DeniedRequest, MinReplicaAvailable, NextRouterBackpressureInfo, PotentialLoadsCheckData,
    PreflightInputs,
};

/// JSON-typed push router used to talk to KV router instances.
///
/// On v1.2.0 the Python `Client` pyclass holds a `PushRouter<serde_json::Value,
/// RsAnnotated<serde_json::Value>>` plus a separate `endpoint` handle; this
/// alias names that router type used throughout the b10_client coordinator.
type JsonPushRouter = PushRouter<serde_json::Value, RsAnnotated<serde_json::Value>>;

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
    /// `route_request` fails closed -- it drops the provisional guard (whose
    /// cleanup task fires `mark_free` asynchronously) and surfaces a
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
pub(super) async fn route_request(
    router: Arc<dyn RouterGuardClient>,
    request: serde_json::Value,
    request_id: String,
    context: Option<context::Context>,
    require_min1_replica_available: Vec<MinReplicaAvailable>,
    notify_timeout: Duration,
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
        if attempt > 0 {
            tokio::time::sleep(ROUTER_GUARD_RETRY_DELAY).await;
        }

        for &instance_id in &instance_ids {
            let request_ctx = create_request_context(request.clone(), &context);
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
            // closed -- it drops the provisional guard (firing `mark_free`
            // via the cleanup task) and surfaces a `RouteSource::ProtocolError`.
            let provisional_guard = RouterRequestGuard::new_provisional(
                router.clone(),
                request_id.clone(),
                instance_id,
                notify_timeout,
            );

            // Stage 1: open the router stream. An `Err` HERE means the
            // router never admitted the request (clean denial or in-band
            // context cancel): `dismiss` the provisional guard so the
            // cleanup task exits WITHOUT sending `mark_free`. Preserves the
            // `route_and_connect_routing_cancelled_in_band_returns_denied_next_router_unreachable`
            // regression at tests.rs:1251 (asserts `mark_free == 0` after a
            // context-stop on `direct`).
            let stream = match router
                .direct(request_ctx, instance_id)
                .instrument(span)
                .await
            {
                Ok(stream) => stream,
                Err(err) => {
                    provisional_guard.dismiss();
                    last_error = Some(err.to_string());
                    tracing::warn!(
                        request_id = %request_id,
                        router_instance_id = instance_id,
                        attempt = attempt + 1,
                        attempts = ROUTER_GUARD_ATTEMPTS,
                        error = %err,
                        "route_request router.direct failed (no admission)"
                    );
                    continue;
                }
            };

            // Stage 2: read the first stream item. The router has now
            // admitted the request internally, so an `Err` here (stream
            // ended before data, decode failure, malformed JSON) is a
            // POST-ADMISSION error: the slot may be reserved on the router.
            // Drop the provisional guard WITHOUT dismissing it -- the
            // cleanup task fires `mark_free` asynchronously. The router's
            // `ActiveSequencesMultiWorker::free` tolerates spurious
            // `mark_free` for unknown request_ids (idempotent
            // `RequestNotFound` arm at
            // lib/kv-router/src/sequences/multi_worker.rs:482 logs at
            // debug and returns Ok).
            let router_response = match first_stream_response(stream).await {
                Ok(response) => response,
                Err(err) => {
                    last_error = Some(err.to_string());
                    tracing::warn!(
                        request_id = %request_id,
                        router_instance_id = instance_id,
                        attempt = attempt + 1,
                        attempts = ROUTER_GUARD_ATTEMPTS,
                        error = %err,
                        "route_request post-admission first_stream_response failed; \
                         dropping provisional guard (cleanup task fires mark_free)"
                    );
                    drop(provisional_guard);
                    continue;
                }
            };

            // Stage 3: dispatch on the decoded router response variant.
            //   `New` => router admitted; commit armed (cleanup stays armed
            //            so a later `Drop` fires `mark_free` once the request
            //            is consumed).
            //   `Backpressure` => clean denial; commit unarmed (cleanup task
            //            exits without sending `mark_free`).
            //   other => protocol error: fail closed. Drop the provisional
            //            guard (armed cleanup task fires `mark_free`), then
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
                     (expected New or Backpressure); failing closed -- dropping \
                     provisional guard (cleanup task fires mark_free)"
                );
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
/// response -- the route's own guard carries any denial -- so the redesign can
/// run the required-available check *in parallel* with the route and again as a
/// post-route re-check without constructing throwaway guards.
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
        if !prefill_marked && !is_error {
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

/// Query the downstream `client`'s router (carried on `check.router`) for
/// potential loads (the `potential_loads` method) and evaluate the response
/// against `check`. Returns `Ok(None)` when the preflight passes; `Ok(Some(info))`
/// when the aggregated loads exceed the configured thresholds (next-router
/// backpressure); `Err(PotentialLoadsError::Unreachable)` when the query
/// itself could not be performed (the coordinator maps this to a
/// `DeniedRequest::NextRouterUnreachable`); `Err(PotentialLoadsError::ProtocolError)`
/// when the decoded response is not `RouterResponse::PotentialLoads` (the
/// coordinator maps this to a `DeniedRequest::ProtocolError`) -- the preflight
/// fails closed rather than treating an unexpected/wrong-protocol shape as a
/// silent pass.
///
/// A threshold of `0` disables that dimension (no limit). Prefill is summed in
/// tokens, decode in BLOCKS (no `block_size` conversion), and `queue_depth` is
/// the router-level `pending_count`.
async fn query_potential_loads(
    tokens: Vec<u32>,
    block_mm_infos: Option<Vec<Option<BlockExtraInfo>>>,
    context: Option<context::Context>,
    request_id: &str,
    check: &PotentialLoadsCheckData,
) -> Result<Option<NextRouterBackpressureInfo>, PotentialLoadsError> {
    let instance_ids = available_router_instance_ids(check.router.as_ref());
    if instance_ids.is_empty() {
        return Err(PotentialLoadsError::Unreachable {
            error: "no router instances available for potential loads query".to_string(),
        });
    }
    let request = RouterRequest::PotentialLoads {
        tokens,
        block_mm_infos,
    };
    let request_value =
        serde_json::to_value(&request).map_err(|e| PotentialLoadsError::Unreachable {
            error: format!("failed to encode potential loads request: {e}"),
        })?;

    let mut last_error: Option<String> = None;
    for attempt in 0..ROUTER_GUARD_ATTEMPTS {
        if attempt > 0 {
            tokio::time::sleep(ROUTER_GUARD_RETRY_DELAY).await;
        }

        for &instance_id in &instance_ids {
            let request_ctx = create_request_context(request_value.clone(), &context);
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
                first_stream_response(stream).await
            }
            .instrument(span)
            .await;

            match result {
                Ok(router_stream_response) => {
                    return evaluate_potential_loads(&router_stream_response.response, check);
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
    let total_prefill: usize = loads.iter().map(|l| l.potential_prefill_tokens).sum();
    let total_decode: usize = loads.iter().map(|l| l.potential_decode_blocks).sum();
    let queue_depth = *pending_count;
    let pending_isl = *pending_isl_tokens;

    let prefill_exceeded =
        check.prefill_tokens_threshold != 0 && total_prefill > check.prefill_tokens_threshold;
    let decode_exceeded =
        check.decode_blocks_threshold != 0 && total_decode > check.decode_blocks_threshold;
    let queue_exceeded =
        check.queue_depth_threshold != 0 && queue_depth > check.queue_depth_threshold;

    if prefill_exceeded || decode_exceeded || queue_exceeded {
        Ok(Some(NextRouterBackpressureInfo {
            queue_depth,
            pending_isl_tokens: pending_isl,
            total_prefill_tokens: total_prefill,
            total_decode_blocks: total_decode,
        }))
    } else {
        Ok(None)
    }
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

/// One attempt of the route phase: run the `new` route, the required-available
/// check, and (on the first attempt only) the next-router potential-loads
/// preflight in *parallel*; merge their results; then re-check
/// required-available after the route completes (a component can go down while
/// the route is in flight). Every armed-but-denied path frees the guard first.
///
/// Merge precedence when the route armed a guard and more than one denier fires
/// is `RequiredComponentsDown` > `NextRouterBackpressure` >
/// `NextRouterUnreachable`. [`route_request`] is invoked with an empty `require`
/// list so it does *not* re-run the required-available preflight inline -- the
/// parallel check here is the authoritative one (it is the cheap local
/// informer lookup that the post-route re-check repeats); the preflight is run
/// only on the first attempt because a stale reroute does not change the
/// downstream router's reported loads.
async fn route_once(
    router_guard_client: Arc<dyn RouterGuardClient>,
    routing_request: serde_json::Value,
    request_id: String,
    context: Option<context::Context>,
    require: Vec<MinReplicaAvailable>,
    preflight: Option<PreflightInputs>,
    notify_timeout: Duration,
) -> RouteOnceOutcome {
    let require_during = required_down_name(&require);

    // Short-circuit when a `require_available` component is already down at
    // the start of the attempt: skipping both the route and the parallel
    // preflight (the preflight is a downstream network call that would be
    // wasted because the route will be denied regardless of the loads).
    if let Some(name) = require_during {
        return RouteOnceOutcome::Denied(DeniedRequest::RequiredComponentsDown { name });
    }

    let route_fut = route_request(
        router_guard_client,
        routing_request,
        request_id.clone(),
        context.clone(),
        Vec::new(),
        notify_timeout,
    );
    let preflight_fut = async move {
        if let Some(pf) = preflight {
            match query_potential_loads(
                pf.tokens,
                pf.block_mm_infos,
                context.clone(),
                &request_id,
                &pf.check,
            )
            .await
            {
                Ok(Some(info)) => Some(Ok(Some(info))),
                Ok(None) => Some(Ok(None)),
                Err(err) => Some(Err(err)),
            }
        } else {
            None
        }
    };

    let (route_res, preflight_res) = tokio::join!(route_fut, preflight_fut);

    match route_res {
        Ok((guard, RouteSource::Routed { worker_id })) => {
            if let Some(name) = require_during {
                drop(guard);
                return RouteOnceOutcome::Denied(DeniedRequest::RequiredComponentsDown { name });
            }
            if let Some(Ok(Some(info))) = preflight_res {
                drop(guard);
                return RouteOnceOutcome::Denied(DeniedRequest::NextRouterBackpressure {
                    queue_depth: info.queue_depth,
                    pending_isl_tokens: info.pending_isl_tokens,
                    total_prefill_tokens: info.total_prefill_tokens,
                    total_decode_blocks: info.total_decode_blocks,
                });
            }
            if let Some(Err(PotentialLoadsError::ProtocolError { received })) = preflight_res {
                drop(guard);
                return RouteOnceOutcome::Denied(DeniedRequest::ProtocolError { received });
            }
            if let Some(Err(PotentialLoadsError::Unreachable { error })) = preflight_res {
                drop(guard);
                return RouteOnceOutcome::Denied(DeniedRequest::NextRouterUnreachable { error });
            }
            // Post-route required re-check (step 4): a component may have gone
            // down between the parallel check and the route completing.
            if let Some(name) = required_down_name(&require) {
                drop(guard);
                return RouteOnceOutcome::Denied(DeniedRequest::RequiredComponentsDown { name });
            }
            RouteOnceOutcome::Route { guard, worker_id }
        }
        Ok((guard, RouteSource::RouterBackpressure)) => {
            if let Some(name) = require_during {
                drop(guard);
                return RouteOnceOutcome::Denied(DeniedRequest::RequiredComponentsDown { name });
            }
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
        Ok((_guard, RouteSource::RequiredDown { name })) => {
            drop(_guard);
            RouteOnceOutcome::Denied(DeniedRequest::RequiredComponentsDown { name })
        }
        // The router replied with a variant that is not a clean admit
        // (`New`) or clean denial (`Backpressure`) for a `new` request.
        // `route_request` already failed closed (dropped the provisional
        // guard so the cleanup task fires `mark_free` asynchronously) and
        // returned an unarmed placeholder guard. Surface the protocol error
        // to the caller; overrides any preflight result (which would be
        // inconsistent with the live router's malformed reply).
        Ok((_guard, RouteSource::ProtocolError { received })) => {
            drop(_guard);
            RouteOnceOutcome::Denied(DeniedRequest::ProtocolError { received })
        }
        Err(route_err) => {
            if let Some(name) = require_during {
                return RouteOnceOutcome::Denied(DeniedRequest::RequiredComponentsDown { name });
            }
            if let Some(Ok(Some(info))) = preflight_res {
                return RouteOnceOutcome::Denied(DeniedRequest::NextRouterBackpressure {
                    queue_depth: info.queue_depth,
                    pending_isl_tokens: info.pending_isl_tokens,
                    total_prefill_tokens: info.total_prefill_tokens,
                    total_decode_blocks: info.total_decode_blocks,
                });
            }
            if let Some(Err(PotentialLoadsError::ProtocolError { received })) = preflight_res {
                return RouteOnceOutcome::Denied(DeniedRequest::ProtocolError { received });
            }
            let error = if let Some(Err(PotentialLoadsError::Unreachable { error })) = preflight_res
            {
                error
            } else {
                route_err.to_string()
            };
            RouteOnceOutcome::Denied(DeniedRequest::NextRouterUnreachable { error })
        }
    }
}

/// Outcome of one [`connect_worker`] attempt: the worker stream opened (armed
/// guard handed back to pack into `AdmittedRequest`), the routed worker is no
/// longer present (`Stale` -- retry the route), or the open failed for any
/// other reason (`Other` -- raise). The guard is moved INTO `open_fut` so an
/// outer cancellation during a shielded open does not drop the guard
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
    Other(anyhow::Error),
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
async fn connect_worker(
    worker_guard_client: Arc<dyn RouterGuardClient>,
    worker_id: u64,
    guard: RouterRequestGuard,
    worker_request: serde_json::Value,
    context: context::Context,
    allow_cancel_setup: bool,
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
        let worker_request_ctx = create_request_context(worker_request, &Some(open_ctx));
        let stream_result = wgc
            .direct(worker_request_ctx, worker_id)
            .instrument(span)
            .await;
        match stream_result {
            Ok(stream) => OpenResult::Ok {
                guard,
                worker_id,
                stream,
            },
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
/// `overlap_blocks` / `dp_strict_rank`. The whole loop is run under the routing
/// cancellation shield by the caller.
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
    max_reroutes: u64,
    allow_cancel_setup: bool,
    notify_timeout: Duration,
) -> Result<RouteAndConnectOutcome> {
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
            notify_timeout,
        )
        .await;

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
        )
        .await
        {
            OpenResult::Ok {
                guard,
                worker_id,
                stream,
            } => {
                return Ok(RouteAndConnectOutcome::Connected {
                    guard,
                    worker_id,
                    stream,
                });
            }
            OpenResult::Stale => {
                // The guard was freed + waited-for-cleanup inside `connect_worker`.
                if attempt >= max_reroutes {
                    return Ok(RouteAndConnectOutcome::Denied(
                        DeniedRequest::NextRouterUnreachable {
                            error: "stale route loop exhausted".to_string(),
                        },
                    ));
                }
                attempt += 1;
                continue;
            }
            OpenResult::Other(err) => return Err(err),
        }
    }
}
