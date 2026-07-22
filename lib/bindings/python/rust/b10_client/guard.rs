// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The KV-router request guard and its `mark_prefill` / `mark_free` lifecycle.
//!
//! [`RouterRequestGuard`] owns a routed KV router request and spawns an
//! always-detached cleanup task on creation when armed (i.e. the route
//! succeeded). The cleanup task sends `mark_free` (and at most one
//! `mark_prefill`) to the router, freeing the request on drop or when
//! `mark_free` is requested. A free request preempts in-flight prefill marking
//! best-effort so router-slot cleanup is not stuck behind `mark_prefill`
//! retries/timeouts. Held internally by the Python [`super::types::AdmittedRequest`].
//!
//! Also holds the [`ROUTER_GUARD_ATTEMPTS`] / [`ROUTER_GUARD_RETRY_DELAY`] /
//! [`ROUTER_GUARD_CLEANUP_GRACE_PERIOD`] /
//! [`ROUTER_GUARD_NOTIFY_TIMEOUT`] timeouts used across the `b10_client`
//! submodules.

use anyhow::Result;
use dynamo_kv_router::protocols::{RouterBackpressureReason, RouterResponse as RsRouterResponse};
use dynamo_runtime::pipeline::context::Context as RsContext;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::{Mutex, Notify, oneshot};
use tracing::Instrument;

use super::coordinator::{RouterGuardClient, callback_router_instance_ids, first_stream_response};

pub(super) const ROUTER_GUARD_ATTEMPTS: usize = 2;
pub(super) const ROUTER_GUARD_RETRY_DELAY: Duration = Duration::from_millis(50);
pub(super) const ROUTER_GUARD_CLEANUP_GRACE_PERIOD: Duration = Duration::from_millis(500);
pub(super) const ROUTER_GUARD_NOTIFY_TIMEOUT: Duration = Duration::from_secs(10 * 60);
/// Per-attempt cap on a single `mark_prefill` / `mark_free` callback to an
/// instance of the KV router. When the cap fires the in-flight future is
/// aborted, the attempt is recorded as an error so the outer
/// [`ROUTER_GUARD_ATTEMPTS`] retry loop can re-try (or move on), and a warning
/// is logged. Also bounds the caller's wait-for-cleanup in the stale-route
/// retry path of [`super::coordinator::connect_worker`].
pub(super) const ROUTER_GUARD_CALLBACK_TIMEOUT: Duration = Duration::from_secs(15);

#[derive(Clone, Copy)]
pub(super) enum GuardMark {
    Prefill,
    Free,
}

enum GuardMarkSendResult {
    Sent,
    PreemptedByFree,
}

impl GuardMark {
    fn request_method(self) -> &'static str {
        match self {
            Self::Prefill => "mark_prefill",
            Self::Free => "mark_free",
        }
    }

    fn success_from(self, response: &RsRouterResponse) -> Option<bool> {
        match (self, response) {
            (Self::Prefill, RsRouterResponse::PrefillMarked { success }) => Some(*success),
            (Self::Free, RsRouterResponse::FreeMarked { success }) => Some(*success),
            _ => None,
        }
    }
}

fn guard_free_requested(state: &RouterRequestGuardState) -> bool {
    state.free_requested.load(Ordering::Acquire) || state.dropped.load(Ordering::Acquire)
}

fn request_once(flag: &AtomicBool, notify: &Notify) {
    if flag.load(Ordering::Relaxed) {
        return;
    }
    if !flag.swap(true, Ordering::Relaxed) {
        notify.notify_one();
    }
}

async fn wait_for_free_request(state: &RouterRequestGuardState) {
    loop {
        if guard_free_requested(state) {
            return;
        }
        state.notify.notified().await;
    }
}

fn flatten_guard_callback_timeout(
    method: &str,
    result: std::result::Result<Result<()>, tokio::time::error::Elapsed>,
) -> Result<()> {
    match result {
        Ok(r) => r,
        Err(_) => Err(anyhow::anyhow!(
            "router callback {method} timed out after {}s",
            ROUTER_GUARD_CALLBACK_TIMEOUT.as_secs()
        )),
    }
}

pub(super) struct RouterRequestGuardState {
    router: Arc<dyn RouterGuardClient>,
    request_id: String,
    preferred_instance_id: u64,
    prefill_requested: AtomicBool,
    free_requested: AtomicBool,
    dropped: AtomicBool,
    cleanup_done: AtomicBool,
    notify_timeout: Duration,
    notify: Notify,
    /// Receiver for the cleanup task's completion signal. Taken (once) by
    /// [`RouterRequestGuard::wait_for_cleanup`]; `None` after the first waiter.
    cleanup_done_rx: Mutex<Option<oneshot::Receiver<()>>>,
}

/// Owns a routed KV router request and its `mark_prefill` / `mark_free`
/// lifecycle callbacks. Spawns a cleanup task on creation when `armed`
/// (i.e. the route succeeded); the task frees the request on drop or when
/// `mark_free` is requested. Held internally by the Python `AdmittedRequest`.
///
/// Two construction modes:
/// - [`RouterRequestGuard::new`] builds a guard with a known router response
///   (used for synthesised backpressure / `RequiredDown` outcomes). When
///   `armed=false` no cleanup task is spawned.
/// - [`RouterRequestGuard::new_provisional`] arms a cleanup task BEFORE the
///   router's `direct()` reply is observed, so a cancellation between
///   admit-on-router and first-response still reclaims the router's slot via
///   `mark_free`. The caller MUST convert the provisional guard with
///   [`RouterRequestGuard::commit`] (response ready) or
///   [`RouterRequestGuard::dismiss`] (router denied) before dropping, so the
///   cleanup task's behaviour is well-defined.
pub(super) struct RouterRequestGuard {
    state: Arc<RouterRequestGuardState>,
    /// Raw JSON of the router response. `None` on a provisional guard until
    /// [`Self::commit`] installs it.
    new_response: Option<rmpv::Value>,
    /// Decoded router response. `None` on a provisional guard until
    /// [`Self::commit`] installs it.
    response: Option<RsRouterResponse>,
    armed: bool,
}

impl RouterRequestGuard {
    /// Build a guard with a known router response. Used for synthesised
    /// outcomes (`RequiredDown`, no-instances `RouterBackpressure`) that bypass
    /// `router.direct()` and so cannot race a cancellation with admission.
    /// When `armed=false` no cleanup task is spawned.
    pub(super) fn new(
        router: Arc<dyn RouterGuardClient>,
        request_id: String,
        preferred_instance_id: u64,
        new_response: rmpv::Value,
        response: RsRouterResponse,
        armed: bool,
        notify_timeout: Duration,
    ) -> Self {
        Self::construct(
            router,
            request_id,
            preferred_instance_id,
            Some(new_response),
            Some(response),
            armed,
            notify_timeout,
        )
    }

    /// Build a provisional armed guard BEFORE calling `router.direct()`. The
    /// cleanup task is spawned immediately so a cancellation between
    /// admit-on-router and first-response still reclaims the router's slot
    /// via `mark_free`. The cleanup task's behaviour is well-defined under
    /// three exit pathways from the caller:
    /// - [`Self::dismiss`] for a pre-admission clean error or in-band
    ///   cancel -- stops the cleanup task without sending `mark_free`
    ///   (admission never happened, so `mark_free` would be wasted traffic).
    /// - [`Self::commit`] for a clean admit (`New`) or clean denial
    ///   (`Backpressure`); installs the response and either keeps the
    ///   cleanup task armed (New, awaiting a later `Drop` once the request
    ///   is consumed) or signals it to exit without `mark_free` (Backpressure).
    /// - `drop` without `commit`/`dismiss` for an ambiguous post-admission
    ///   state -- Race Site 12 fail-closed semantics. `Drop` sets `dropped`
    ///   and `free_requested` and notifies the cleanup task, which fires
    ///   `mark_free` asynchronously. Spurious `mark_free` is benign because
    ///   the router's `free` tolerates unknown `request_id`s idempotently.
    pub(super) fn new_provisional(
        router: Arc<dyn RouterGuardClient>,
        request_id: String,
        preferred_instance_id: u64,
        notify_timeout: Duration,
    ) -> Self {
        Self::construct(
            router,
            request_id,
            preferred_instance_id,
            None,
            None,
            true,
            notify_timeout,
        )
    }

    fn construct(
        router: Arc<dyn RouterGuardClient>,
        request_id: String,
        preferred_instance_id: u64,
        new_response: Option<rmpv::Value>,
        response: Option<RsRouterResponse>,
        armed: bool,
        notify_timeout: Duration,
    ) -> Self {
        let (cleanup_done_tx, cleanup_done_rx) = oneshot::channel();
        let state = Arc::new(RouterRequestGuardState {
            router,
            request_id,
            preferred_instance_id,
            prefill_requested: AtomicBool::new(false),
            free_requested: AtomicBool::new(false),
            dropped: AtomicBool::new(false),
            cleanup_done: AtomicBool::new(!armed),
            notify_timeout,
            notify: Notify::new(),
            cleanup_done_rx: Mutex::new(Some(cleanup_done_rx)),
        });

        if armed {
            tokio::spawn(router_request_guard_cleanup(state.clone(), cleanup_done_tx));
        }

        Self {
            state,
            new_response,
            response,
            armed,
        }
    }

    /// Install a router response into a provisional guard. On `armed=true`
    /// (router admitted: `New` response) the cleanup task remains armed and
    /// `Drop` will fire `mark_free`. On `armed=false` (router returned
    /// backpressure: no admission) the cleanup task is signalled to exit
    /// without sending `mark_free`.
    pub(super) fn commit(
        mut self,
        new_response: rmpv::Value,
        response: RsRouterResponse,
        armed: bool,
    ) -> Self {
        self.new_response = Some(new_response);
        self.response = Some(response);
        self.armed = armed;
        if !armed {
            self.state.cleanup_done.store(true, Ordering::Release);
            self.state.notify.notify_one();
        }
        self
    }

    /// Dismiss a provisional guard WITHOUT sending `mark_free`. Use when the
    /// router returned a clean error or an in-band cancellation: no admission
    /// happened, so `mark_free` would be wasted traffic. Sets `cleanup_done`
    /// and notifies the cleanup task so it exits on its next loop iteration;
    /// `Drop` is then a no-op because `armed=false && cleanup_done=true`.
    pub(super) fn dismiss(mut self) {
        self.armed = false;
        self.state.cleanup_done.store(true, Ordering::Release);
        self.state.notify.notify_one();
    }

    /// The raw JSON of the initial router response, for surfacing to Python.
    pub(super) fn new_response(&self) -> &rmpv::Value {
        self.new_response
            .as_ref()
            .expect("RouterRequestGuard response accessed before commit")
    }

    /// True when the route succeeded without router backpressure.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn routed(&self) -> bool {
        matches!(self.response.as_ref(), Some(RsRouterResponse::New { .. }))
    }

    /// Serialised router backpressure reason, or an empty string when routed.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn backpressure_reason(&self) -> String {
        match self.response.as_ref() {
            Some(RsRouterResponse::Backpressure { .. }) => {
                serde_json::to_string(self.response.as_ref().expect("checked Some above"))
                    .unwrap_or_else(|_| {
                        format!("{:?}", self.response.as_ref().expect("checked Some above"))
                    })
            }
            _ => String::new(),
        }
    }

    /// When the router returned backpressure, the decoded backpressure fields
    /// (`reason`, queued ISL tokens, optional max queued ISL tokens); `None`
    /// when the route succeeded.
    pub(super) fn backpressure_fields(
        &self,
    ) -> Option<(RouterBackpressureReason, usize, Option<usize>)> {
        if let Some(RsRouterResponse::Backpressure {
            reason,
            queued_isl_tokens,
            max_queued_isl_tokens,
        }) = self.response.as_ref()
        {
            Some((reason.clone(), *queued_isl_tokens, *max_queued_isl_tokens))
        } else {
            None
        }
    }

    /// Estimated cached-token overlap the router reported for a routed
    /// [`RouterResponse::New`], derived from router-native `overlap_blocks` and
    /// the coordinator block size.
    /// `0` when the route did not arm the guard (the response is not `New`).
    pub(super) fn estimated_overlap_tokens(&self, block_size: u32) -> u64 {
        if let Some(RsRouterResponse::New { overlap_blocks, .. }) = self.response.as_ref() {
            u64::from(*overlap_blocks) * u64::from(block_size)
        } else {
            0
        }
    }

    /// Request that the cleanup task marks prefill complete.
    pub(super) fn mark_prefill(&self) {
        if !self.armed {
            return;
        }
        request_once(&self.state.prefill_requested, &self.state.notify);
    }

    /// Request that the cleanup task frees the request.
    pub(super) fn mark_free(&self) {
        if !self.armed {
            return;
        }
        request_once(&self.state.free_requested, &self.state.notify);
    }

    /// Wait until the cleanup task has reached its terminal `cleanup_done`
    /// state (i.e. `mark_free` has been sent, or the guard was never armed),
    /// bounded by `timeout`. When the bound fires the cleanup task may still
    /// be in-flight -- a warning is logged and this method returns anyway so
    /// the caller (e.g. a stale-route retry loop) is not blocked indefinitely.
    /// At most one waiter is supported per guard; subsequent callers fall
    /// back to a single `cleanup_done` flag re-check.
    pub(super) async fn wait_for_cleanup(&self, timeout: Duration) {
        if self.state.cleanup_done.load(Ordering::Acquire) {
            return;
        }
        let rx = self.state.cleanup_done_rx.lock().await.take();
        let Some(rx) = rx else {
            if !self.state.cleanup_done.load(Ordering::Acquire) {
                tracing::warn!(
                    request_id = %self.state.request_id,
                    "router request guard cleanup signal already consumed; mark_free completion uncertain"
                );
            }
            return;
        };
        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(())) => {}
            Ok(Err(_)) => {
                tracing::warn!(
                    request_id = %self.state.request_id,
                    "router request guard cleanup task exited without signaling; mark_free completion uncertain"
                );
            }
            Err(_) => {
                tracing::warn!(
                    request_id = %self.state.request_id,
                    timeout_ms = timeout.as_millis(),
                    "router request guard cleanup did not complete within timeout; proceeding anyway"
                );
            }
        }
    }
}

impl Drop for RouterRequestGuard {
    fn drop(&mut self) {
        if self.armed && !self.state.cleanup_done.load(Ordering::Acquire) {
            let prefill_requested = self.state.prefill_requested.load(Ordering::Acquire);
            let free_requested = self.state.free_requested.load(Ordering::Acquire);
            if free_requested {
                tracing::debug!(
                    request_id = %self.state.request_id,
                    endpoint = %self.state.router.endpoint_id(),
                    preferred_router_instance_id = self.state.preferred_instance_id,
                    prefill_requested,
                    free_requested,
                    "router request guard dropped after explicit free request; ensuring cleanup"
                );
            } else {
                tracing::warn!(
                    request_id = %self.state.request_id,
                    endpoint = %self.state.router.endpoint_id(),
                    preferred_router_instance_id = self.state.preferred_instance_id,
                    prefill_requested,
                    free_requested,
                    "router request guard dropped while still armed; requesting mark_free"
                );
            }
            self.state.dropped.store(true, Ordering::Release);
            self.state.free_requested.store(true, Ordering::Release);
            self.state.notify.notify_one();
        }
    }
}

async fn send_router_guard_mark(
    state: &RouterRequestGuardState,
    mark: GuardMark,
) -> Result<GuardMarkSendResult> {
    let instance_ids =
        callback_router_instance_ids(state.router.as_ref(), state.preferred_instance_id);
    let method = mark.request_method();
    let preemptible = matches!(mark, GuardMark::Prefill);
    let request: rmpv::Value = serde_json::from_value(serde_json::json!({
        "method": method,
        "request_id": state.request_id.clone(),
    }))?;
    let mut last_error = None;

    for attempt in 0..ROUTER_GUARD_ATTEMPTS {
        if preemptible && guard_free_requested(state) {
            return Ok(GuardMarkSendResult::PreemptedByFree);
        }

        if attempt > 0 {
            if preemptible {
                tokio::select! {
                    _ = tokio::time::sleep(ROUTER_GUARD_RETRY_DELAY) => {}
                    _ = wait_for_free_request(state) => {
                        return Ok(GuardMarkSendResult::PreemptedByFree);
                    }
                }
            } else {
                tokio::time::sleep(ROUTER_GUARD_RETRY_DELAY).await;
            }
        }

        for &instance_id in &instance_ids {
            if preemptible && guard_free_requested(state) {
                return Ok(GuardMarkSendResult::PreemptedByFree);
            }

            let request_ctx = RsContext::with_id_and_metadata(
                request.clone(),
                state.request_id.clone(),
                Default::default(),
            );
            let span = tracing::info_span!(
                "kv_router.router_request_guard_callback",
                request_id = %state.request_id,
                router_instance_id = instance_id,
                preferred_router_instance_id = state.preferred_instance_id,
                attempt = attempt + 1,
                attempts = ROUTER_GUARD_ATTEMPTS,
                method,
            );

            let result = async {
                let stream = state.router.direct(request_ctx, instance_id).await?;
                let router_response = first_stream_response(stream).await?;
                match mark.success_from(&router_response.response) {
                    Some(true) => Ok(()),
                    Some(false) => Err(anyhow::anyhow!(
                        "router callback {method} returned unsuccessful response: {:?}",
                        router_response.response
                    )),
                    None => Err(anyhow::anyhow!(
                        "router callback {method} returned unexpected response: {:?}",
                        router_response.response
                    )),
                }
            }
            .instrument(span);

            // Bound each per-instance callback attempt with the existing hard
            // timeout, but warn earlier when cleanup callbacks exceed the
            // grace period that route_request waits before reusing the same
            // request_id.
            let started = tokio::time::Instant::now();
            let hard_timeout = tokio::time::timeout(ROUTER_GUARD_CALLBACK_TIMEOUT, result);
            tokio::pin!(hard_timeout);
            let slow_log = tokio::time::sleep(ROUTER_GUARD_CLEANUP_GRACE_PERIOD);
            tokio::pin!(slow_log);
            let mut slow_logged = false;
            let result = loop {
                if preemptible {
                    tokio::select! {
                        result = &mut hard_timeout => {
                            break flatten_guard_callback_timeout(method, result);
                        }
                        _ = wait_for_free_request(state) => {
                            return Ok(GuardMarkSendResult::PreemptedByFree);
                        }
                        _ = &mut slow_log, if !slow_logged => {
                            slow_logged = true;
                            tracing::warn!(
                                request_id = %state.request_id,
                                method,
                                router_instance_id = instance_id,
                                preferred_router_instance_id = state.preferred_instance_id,
                                attempt = attempt + 1,
                                attempts = ROUTER_GUARD_ATTEMPTS,
                                elapsed_ms = started.elapsed().as_millis(),
                                threshold_ms = ROUTER_GUARD_CLEANUP_GRACE_PERIOD.as_millis(),
                                "router request guard callback still pending after cleanup grace period"
                            );
                        }
                    }
                } else {
                    tokio::select! {
                        result = &mut hard_timeout => {
                            break flatten_guard_callback_timeout(method, result);
                        }
                        _ = &mut slow_log, if !slow_logged => {
                            slow_logged = true;
                            tracing::warn!(
                                request_id = %state.request_id,
                                method,
                                router_instance_id = instance_id,
                                preferred_router_instance_id = state.preferred_instance_id,
                                attempt = attempt + 1,
                                attempts = ROUTER_GUARD_ATTEMPTS,
                                elapsed_ms = started.elapsed().as_millis(),
                                threshold_ms = ROUTER_GUARD_CLEANUP_GRACE_PERIOD.as_millis(),
                                "router request guard callback still pending after cleanup grace period"
                            );
                        }
                    }
                }
            };

            match result {
                Ok(()) => {
                    if instance_id != state.preferred_instance_id {
                        tracing::warn!(
                            request_id = %state.request_id,
                            method,
                            preferred_router_instance_id = state.preferred_instance_id,
                            fallback_router_instance_id = instance_id,
                            "router request guard callback succeeded via fallback router"
                        );
                    }
                    return Ok(GuardMarkSendResult::Sent);
                }
                Err(err) => {
                    last_error = Some(err.to_string());
                    tracing::warn!(
                        request_id = %state.request_id,
                        method,
                        router_instance_id = instance_id,
                        attempt = attempt + 1,
                        attempts = ROUTER_GUARD_ATTEMPTS,
                        error = %err,
                        "router request guard callback failed"
                    );
                }
            }
        }
    }

    Err(anyhow::anyhow!(
        "router callback {method} failed for request {}{}",
        state.request_id,
        last_error.map(|err| format!(": {err}")).unwrap_or_default()
    ))
}

async fn router_request_guard_cleanup(
    state: Arc<RouterRequestGuardState>,
    send_done_tx: oneshot::Sender<()>,
) {
    let mut prefill_sent = false;
    let mut prefill_attempted = false;

    loop {
        if state.cleanup_done.load(Ordering::Acquire) {
            break;
        }

        if guard_free_requested(&state) {
            let free_result = send_router_guard_mark(&state, GuardMark::Free).await;
            if let Err(err) = free_result {
                tracing::error!(
                    request_id = %state.request_id,
                    error = %err,
                    "router request guard failed to free request"
                );
            }
            state.cleanup_done.store(true, Ordering::Release);
            break;
        }

        if state.prefill_requested.load(Ordering::Acquire) && !prefill_sent && !prefill_attempted {
            prefill_attempted = true;
            prefill_sent = matches!(
                send_router_guard_mark(&state, GuardMark::Prefill).await,
                Ok(GuardMarkSendResult::Sent)
            );
            continue;
        }

        if tokio::time::timeout(state.notify_timeout, state.notify.notified())
            .await
            .is_err()
        {
            tracing::warn!(
                request_id = %state.request_id,
                timeout_secs = state.notify_timeout.as_secs(),
                "router request guard cleanup timed out waiting for notify; freeing request"
            );
            state.free_requested.store(true, Ordering::Release);
        }
    }

    // Signal any stale-retry waiter that the cleanup task has reached its
    // terminal state (best-effort -- the receiver may have already timed out
    // or been dropped).
    let _ = send_done_tx.send(());
}
