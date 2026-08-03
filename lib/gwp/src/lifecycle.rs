// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process lifecycle and readiness state.
//!
//! Liveness only means that the gRPC server can make progress. Readiness is stricter:
//! the reflector must have completed a planner reconciliation, every configured
//! route must have a live endpoint, at least one worker must be routable, and a
//! joining replica must have observed the configured peer warm-up window.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use tokio::sync::watch;

#[derive(Debug)]
struct State {
    planner_reconciled: bool,
    routable_workers: usize,
    configured_routes_covered: bool,
    peer_replicas_at_startup: usize,
    peer_warmup_until: Option<tokio::time::Instant>,
    draining: bool,
}

/// Shared lifecycle state used by the reflector, gRPC probes, and shutdown
/// coordinator.
#[derive(Clone, Debug)]
pub struct Lifecycle {
    inner: Arc<RwLock<State>>,
    revision: watch::Sender<u64>,
}

/// Snapshot used to publish gRPC health and explain readiness in logs/tests.
#[derive(Clone, Debug)]
pub struct HealthReport {
    pub ready: bool,
    pub live: bool,
    pub phase: &'static str,
    pub reason: &'static str,
    pub planner_reconciled: bool,
    pub routable_workers: usize,
    pub configured_routes_covered: bool,
    pub peer_replicas_at_startup: usize,
    pub peer_warmup_remaining_secs: u64,
    pub draining: bool,
}

impl Lifecycle {
    /// Create a starting lifecycle. Peer warm-up is applied only when another
    /// replica was already registered when this process joined.
    pub fn starting(peer_replicas: usize, peer_warmup: Duration) -> Self {
        let peer_warmup_until = (peer_replicas > 0 && !peer_warmup.is_zero())
            .then(|| tokio::time::Instant::now() + peer_warmup);
        Self {
            inner: Arc::new(RwLock::new(State {
                planner_reconciled: false,
                routable_workers: 0,
                configured_routes_covered: false,
                peer_replicas_at_startup: peer_replicas,
                peer_warmup_until,
                draining: false,
            })),
            revision: watch::channel(0).0,
        }
    }

    /// Test helper for manually assembled proxy state.
    #[cfg(test)]
    pub(crate) fn ready_for_tests() -> Self {
        let lifecycle = Self::starting(0, Duration::ZERO);
        lifecycle.update_routing(1, true);
        lifecycle
    }

    /// Publish the result of a complete reflector cycle.
    pub fn update_routing(&self, routable_workers: usize, configured_routes_covered: bool) {
        {
            let mut state = self.inner.write();
            state.planner_reconciled = true;
            state.routable_workers = routable_workers;
            state.configured_routes_covered = configured_routes_covered;
        }
        self.notify_changed();
    }

    /// Stop admitting new schedules. Lifecycle completion events remain
    /// accepted so existing Envoy requests can release their bookings.
    pub fn begin_draining(&self) {
        self.inner.write().draining = true;
        self.notify_changed();
    }

    pub fn is_ready(&self) -> bool {
        self.report().ready
    }

    #[allow(dead_code)]
    pub(crate) fn subscribe(&self) -> watch::Receiver<u64> {
        self.revision.subscribe()
    }

    #[allow(dead_code)]
    pub(crate) fn next_time_transition(&self) -> Option<tokio::time::Instant> {
        self.inner
            .read()
            .peer_warmup_until
            .filter(|deadline| *deadline > tokio::time::Instant::now())
    }

    fn notify_changed(&self) {
        self.revision.send_modify(|revision| {
            *revision = revision.wrapping_add(1);
        });
    }

    pub fn report(&self) -> HealthReport {
        let state = self.inner.read();
        let now = tokio::time::Instant::now();
        let warmup_remaining = state
            .peer_warmup_until
            .and_then(|deadline| deadline.checked_duration_since(now))
            .unwrap_or_default();
        let warmup_remaining_secs = if warmup_remaining.is_zero() {
            0
        } else {
            warmup_remaining
                .as_secs()
                .saturating_add(u64::from(warmup_remaining.subsec_nanos() > 0))
        };

        let (phase, reason) = if state.draining {
            ("draining", "process is draining existing requests")
        } else if !state.planner_reconciled {
            ("starting", "waiting for initial planner reconciliation")
        } else if state.routable_workers == 0 {
            ("not_ready", "no planner-observed worker is routable")
        } else if !state.configured_routes_covered {
            (
                "not_ready",
                "one or more configured routes have no live endpoint",
            )
        } else if warmup_remaining_secs > 0 {
            (
                "warming",
                "receiving replica events before accepting traffic",
            )
        } else {
            ("ready", "ready")
        };
        let ready = phase == "ready";

        HealthReport {
            ready,
            live: true,
            phase,
            reason,
            planner_reconciled: state.planner_reconciled,
            routable_workers: state.routable_workers,
            configured_routes_covered: state.configured_routes_covered,
            peer_replicas_at_startup: state.peer_replicas_at_startup,
            peer_warmup_remaining_secs: warmup_remaining_secs,
            draining: state.draining,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(start_paused = true)]
    async fn readiness_requires_reconciliation_route_coverage_and_warmup() {
        let lifecycle = Lifecycle::starting(2, Duration::from_secs(90));
        assert_eq!(lifecycle.report().phase, "starting");

        lifecycle.update_routing(4, true);
        let warming = lifecycle.report();
        assert_eq!(warming.phase, "warming");
        assert_eq!(warming.peer_warmup_remaining_secs, 90);

        tokio::time::advance(Duration::from_secs(89)).await;
        assert!(!lifecycle.is_ready());
        tokio::time::advance(Duration::from_secs(1)).await;
        assert!(lifecycle.is_ready());

        lifecycle.update_routing(4, false);
        assert_eq!(lifecycle.report().phase, "not_ready");
        lifecycle.update_routing(0, true);
        assert_eq!(lifecycle.report().phase, "not_ready");
    }

    #[tokio::test]
    async fn draining_revokes_readiness_but_not_liveness() {
        let lifecycle = Lifecycle::ready_for_tests();
        assert!(lifecycle.is_ready());
        lifecycle.begin_draining();
        let report = lifecycle.report();
        assert!(!report.ready);
        assert!(report.live);
        assert_eq!(report.phase, "draining");
    }
}
