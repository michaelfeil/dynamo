//! The worker payload copy for `route_and_connect`: a plain
//! `rmpv::Value::clone` on the blocking pool; only *when* it runs varies.
//! Small payloads copy eagerly, hidden behind the routing RPC (a denied
//! route discards at most a few ms of background work). Large payloads
//! (multi-hundred-MB `mm_kwargs`) copy only after admission — their copy
//! dwarfs the RPC, so overlapping would save a few ms while risking the
//! full copy as waste on every denied or backpressured request.

use super::coordinator::duration_ms_for_log;
use anyhow::Result;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Eager-copy cutoff: ~a few ms of clone at page-fault-bound throughput
/// (~2 GB/s) — about what the routing RPC hides.
const EAGER_COPY_MAX_BYTES: usize = 8 << 20;

/// Log the payload copy when it kept the critical path waiting (beyond what
/// the routing RPC hid) for longer than this.
pub(super) const PAYLOAD_COPY_UNHIDDEN_LOG_THRESHOLD: Duration = Duration::from_millis(40);

/// Bytes in `Binary` / `String` / `Ext` leaves, for the slow-copy log line.
/// Runs on the blocking pool next to the copy itself.
fn value_payload_bytes(value: &rmpv::Value) -> usize {
    match value {
        rmpv::Value::Binary(bytes) => bytes.len(),
        rmpv::Value::String(s) => s.as_bytes().len(),
        rmpv::Value::Ext(_, bytes) => bytes.len(),
        rmpv::Value::Array(items) => items.iter().map(value_payload_bytes).sum(),
        rmpv::Value::Map(entries) => entries
            .iter()
            .map(|(key, val)| value_payload_bytes(key) + value_payload_bytes(val))
            .sum(),
        _ => 0,
    }
}

/// Whether a deep clone of `value` stays within `budget` bytes, charging
/// leaf bytes AND per-element container storage. Containers are charged in
/// O(1) before descending and the walk stops once the budget goes negative,
/// so oversized payloads are classified in O(1) — safe on the executor.
fn clone_cost_within(value: &rmpv::Value, budget: &mut isize) -> bool {
    match value {
        rmpv::Value::Binary(bytes) => *budget -= bytes.len() as isize,
        rmpv::Value::String(s) => *budget -= s.as_bytes().len() as isize,
        rmpv::Value::Ext(_, bytes) => *budget -= bytes.len() as isize,
        rmpv::Value::Array(items) => {
            *budget -= (items.len() * size_of::<rmpv::Value>()) as isize;
            if *budget < 0 {
                return false;
            }
            if !items.iter().all(|item| clone_cost_within(item, budget)) {
                return false;
            }
        }
        rmpv::Value::Map(entries) => {
            *budget -= (entries.len() * size_of::<(rmpv::Value, rmpv::Value)>()) as isize;
            if *budget < 0 {
                return false;
            }
            if !entries
                .iter()
                .all(|(key, val)| clone_cost_within(key, budget) && clone_cost_within(val, budget))
            {
                return false;
            }
        }
        _ => {}
    }
    *budget >= 0
}

/// A worker payload copy staged for one routing attempt: resolved via
/// [`PayloadCopy::finish`] inside the worker-setup shield once the route is
/// usable, or dropped via [`PayloadCopy::abandon`] on a denied route.
pub(super) enum PayloadCopy {
    /// Small payload, copying since before the routing RPC was sent.
    Eager {
        handle: tokio::task::JoinHandle<(rmpv::Value, usize, Duration)>,
    },
    /// Large payload; the copy starts in `finish`, after admission.
    Deferred { base: Arc<rmpv::Value> },
}

impl PayloadCopy {
    pub(super) fn stage(base: Arc<rmpv::Value>) -> Self {
        let mut budget = EAGER_COPY_MAX_BYTES as isize;
        if clone_cost_within(&base, &mut budget) {
            let handle = tokio::task::spawn_blocking(move || {
                let started = Instant::now();
                let payload_bytes = value_payload_bytes(&base);
                ((*base).clone(), payload_bytes, started.elapsed())
            });
            Self::Eager { handle }
        } else {
            Self::Deferred { base }
        }
    }

    /// Abandon without waiting: `Deferred` has done no work; an `Eager`
    /// task finishes its small copy in the background and is discarded.
    pub(super) fn abandon(self) {}

    /// Produce the copy. Runs after the routing RPC returned, so time spent
    /// here is exactly the copy latency routing did NOT hide; logged above
    /// [`PAYLOAD_COPY_UNHIDDEN_LOG_THRESHOLD`].
    pub(super) async fn finish(self, request_id: &str, phase: Option<&str>) -> Result<rmpv::Value> {
        let wait_started = Instant::now();
        let (value, payload_bytes, copy_duration) = match self {
            Self::Eager { handle } => handle
                .await
                .map_err(|err| anyhow::anyhow!("worker payload copy task failed: {err}"))?,
            Self::Deferred { base } => {
                let (value, payload_bytes) = tokio::task::spawn_blocking(move || {
                    ((*base).clone(), value_payload_bytes(&base))
                })
                .await
                .map_err(|err| anyhow::anyhow!("worker payload copy task failed: {err}"))?;
                (value, payload_bytes, wait_started.elapsed())
            }
        };
        let unhidden = wait_started.elapsed();
        if unhidden >= PAYLOAD_COPY_UNHIDDEN_LOG_THRESHOLD {
            tracing::info!(
                request_id = %request_id,
                phase = phase.unwrap_or("unknown"),
                unhidden_ms = duration_ms_for_log(unhidden),
                copy_ms = duration_ms_for_log(copy_duration),
                payload_mb = payload_bytes as f64 / 1e6,
                "route_and_connect: worker payload copy exceeded 40ms of unhidden latency"
            );
        }
        Ok(value)
    }
}
