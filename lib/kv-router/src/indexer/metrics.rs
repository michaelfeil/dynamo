// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "runtime-protocols")]
use std::sync::Arc;
#[cfg(all(feature = "metrics", feature = "runtime-protocols"))]
use std::sync::OnceLock;
#[cfg(feature = "bench")]
use std::sync::atomic::AtomicU64;

#[cfg(feature = "runtime-protocols")]
use dynamo_runtime::component::Component;
#[cfg(all(feature = "metrics", feature = "runtime-protocols"))]
use dynamo_runtime::metrics::MetricsHierarchy;
#[cfg(feature = "metrics")]
use prometheus::{Counter, CounterVec, IntCounter, IntCounterVec, Opts};

use crate::protocols::{KvCacheEventData, KvCacheEventError};

/// Lightweight, `Copy` discriminant for [`KvCacheEventData`].
///
/// Extracted before the event is moved into `apply_event()`, then passed to
/// [`PreBoundEventCounters::inc`] so the compiler enforces exhaustiveness
/// without requiring a clone of the full event payload.
///
/// `Display` produces the Prometheus label value (`"stored"`, `"removed"`,
/// `"cleared"`), so this enum is also the single source of truth for the
/// `event_type` label — replacing the former `get_event_type()` helper.
#[derive(Debug, Clone, Copy)]
pub enum EventKind {
    Stored,
    Removed,
    Cleared,
}

impl EventKind {
    pub fn of(data: &KvCacheEventData) -> Self {
        match data {
            KvCacheEventData::Stored(_) => Self::Stored,
            KvCacheEventData::Removed(_) => Self::Removed,
            KvCacheEventData::Cleared => Self::Cleared,
        }
    }
}

impl std::fmt::Display for EventKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stored => f.write_str(METRIC_EVENT_STORED),
            Self::Removed => f.write_str(METRIC_EVENT_REMOVED),
            Self::Cleared => f.write_str(METRIC_EVENT_CLEARED),
        }
    }
}

/// Lightweight, `Copy` discriminant for KV event warnings.
#[derive(Debug, Clone, Copy)]
pub enum EventWarningKind {
    DuplicateStore,
}

impl std::fmt::Display for EventWarningKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateStore => f.write_str(METRIC_WARNING_DUPLICATE_STORE),
        }
    }
}

#[cfg(feature = "bench")]
pub(crate) struct ShardedIndexerCounters {
    pub(crate) find_match_dispatches: AtomicU64,
    pub(crate) find_match_early_returns: AtomicU64,
    pub(crate) anchor_installs: AtomicU64,
    pub(crate) anchor_reuses: AtomicU64,
    pub(crate) remove_broadcasts: AtomicU64,
}

#[cfg(feature = "bench")]
impl ShardedIndexerCounters {
    fn new() -> Self {
        Self {
            find_match_dispatches: AtomicU64::new(0),
            find_match_early_returns: AtomicU64::new(0),
            anchor_installs: AtomicU64::new(0),
            anchor_reuses: AtomicU64::new(0),
            remove_broadcasts: AtomicU64::new(0),
        }
    }
}

#[cfg(feature = "bench")]
pub(crate) struct ShardedIndexerTiming {
    pub(crate) calls: AtomicU64,
    pub(crate) routing_ns: AtomicU64,
    pub(crate) shard_ns: AtomicU64,
}

#[cfg(feature = "bench")]
impl ShardedIndexerTiming {
    fn new() -> Self {
        Self {
            calls: AtomicU64::new(0),
            routing_ns: AtomicU64::new(0),
            shard_ns: AtomicU64::new(0),
        }
    }
}

#[cfg(feature = "bench")]
pub(crate) struct ShardedIndexerMetrics {
    pub(crate) counters: ShardedIndexerCounters,
    pub(crate) timing: ShardedIndexerTiming,
}

#[cfg(feature = "bench")]
impl ShardedIndexerMetrics {
    pub(crate) fn new() -> Self {
        Self {
            counters: ShardedIndexerCounters::new(),
            timing: ShardedIndexerTiming::new(),
        }
    }
}

/// Metrics for the KV Indexer.
#[derive(Clone)]
#[cfg_attr(not(feature = "metrics"), derive(Default))]
pub struct KvIndexerMetrics {
    /// Counter of events applied.
    #[cfg(feature = "metrics")]
    pub kv_cache_events_applied: IntCounterVec,
    /// Counter of suspicious-but-valid KV events.
    #[cfg(feature = "metrics")]
    pub kv_cache_event_warnings: IntCounterVec,
    /// Count of indexer operations by type (event applies, remove_worker,
    /// find_matches).
    #[cfg(feature = "metrics")]
    pub indexer_ops_count: IntCounterVec,
    /// Total seconds spent in indexer operations by type; divide by
    /// `kv_indexer_ops_count` for mean latency.
    #[cfg(feature = "metrics")]
    pub indexer_ops_latency: CounterVec,
}

/// Metric status labels.
pub const METRIC_STATUS_OK: &str = "ok";
pub const METRIC_STATUS_PARENT_NOT_FOUND: &str = "parent_block_not_found";
pub const METRIC_STATUS_BLOCK_NOT_FOUND: &str = "block_not_found";
pub const METRIC_STATUS_INVALID_BLOCK: &str = "invalid_block";

/// Metric event labels.
pub const METRIC_EVENT_STORED: &str = "stored";
pub const METRIC_EVENT_REMOVED: &str = "removed";
pub const METRIC_EVENT_CLEARED: &str = "cleared";

/// Metric warning labels.
pub const METRIC_WARNING_DUPLICATE_STORE: &str = "duplicate_store";

/// Metric operation labels for indexer operation tracking. Event applies use
/// the [`EventKind`] labels (`stored`/`removed`/`cleared`).
pub const METRIC_OP_REMOVE_WORKER: &str = "remove_worker";
pub const METRIC_OP_FIND_MATCHES: &str = "find_matches";

/// Metric name for KV cache events applied counter.
#[cfg(all(feature = "metrics", feature = "runtime-protocols"))]
const KV_CACHE_EVENTS_APPLIED_SUFFIX: &str = "kv_cache_events_applied";
#[cfg(feature = "metrics")]
const KV_CACHE_EVENTS_APPLIED_NAME: &str = "dynamo_kvrouter_kv_cache_events_applied";
#[cfg(all(feature = "metrics", feature = "runtime-protocols"))]
const KV_CACHE_EVENT_WARNINGS_SUFFIX: &str = "kv_cache_event_warnings";
#[cfg(feature = "metrics")]
const KV_CACHE_EVENT_WARNINGS_NAME: &str = "dynamo_kvrouter_kv_cache_event_warnings";
/// Metric name for KV indexer operations count.
#[cfg(all(feature = "metrics", feature = "runtime-protocols"))]
const KV_INDEXER_OPS_COUNT_SUFFIX: &str = "kv_indexer_ops_count";
#[cfg(feature = "metrics")]
const KV_INDEXER_OPS_COUNT_NAME: &str = "dynamo_kvrouter_kv_indexer_ops_count";
/// Metric name for KV indexer operations latency.
#[cfg(all(feature = "metrics", feature = "runtime-protocols"))]
const KV_INDEXER_OPS_LATENCY_SUFFIX: &str = "kv_indexer_ops_latency_total";
#[cfg(feature = "metrics")]
const KV_INDEXER_OPS_LATENCY_NAME: &str = "dynamo_kvrouter_kv_indexer_ops_latency_total";

#[cfg(all(feature = "metrics", feature = "runtime-protocols"))]
static KV_INDEXER_METRICS: OnceLock<Arc<KvIndexerMetrics>> = OnceLock::new();

impl KvIndexerMetrics {
    #[cfg(all(feature = "metrics", feature = "runtime-protocols"))]
    fn new(
        kv_cache_events_applied: IntCounterVec,
        kv_cache_event_warnings: IntCounterVec,
        indexer_ops_count: IntCounterVec,
        indexer_ops_latency: CounterVec,
    ) -> Self {
        Self {
            kv_cache_events_applied,
            kv_cache_event_warnings,
            indexer_ops_count,
            indexer_ops_latency,
        }
    }

    /// Creates a new KvIndexerMetrics from a Component, memoizing the result in
    /// KV_INDEXER_METRICS to avoid duplicate registration issues.
    #[cfg(feature = "runtime-protocols")]
    pub fn from_component(component: &Component) -> Arc<Self> {
        #[cfg(feature = "metrics")]
        {
            KV_INDEXER_METRICS
                .get_or_init(|| {
                    let try_create = || -> anyhow::Result<Self> {
                        let kv_cache_events_applied = component.metrics().create_intcountervec(
                            KV_CACHE_EVENTS_APPLIED_SUFFIX,
                            "Total number of KV cache events applied to index",
                            &["event_type", "status"],
                            &[],
                        )?;
                        let kv_cache_event_warnings = component.metrics().create_intcountervec(
                            KV_CACHE_EVENT_WARNINGS_SUFFIX,
                            "Total number of suspicious KV cache events seen by the router indexer",
                            &["warning_kind"],
                            &[],
                        )?;
                        let indexer_ops_count = component.metrics().create_intcountervec(
                            KV_INDEXER_OPS_COUNT_SUFFIX,
                            "Total count of KV indexer operations by type",
                            &["operation"],
                            &[],
                        )?;
                        let indexer_ops_latency = component.metrics().create_countervec(
                            KV_INDEXER_OPS_LATENCY_SUFFIX,
                            "Total duration of KV indexer operations in seconds",
                            &["operation"],
                            &[],
                        )?;
                        Ok(Self::new(
                            kv_cache_events_applied,
                            kv_cache_event_warnings,
                            indexer_ops_count,
                            indexer_ops_latency,
                        ))
                    };

                    match try_create() {
                        Ok(metrics) => Arc::new(metrics),
                        Err(e) => {
                            tracing::warn!("Failed to create kv indexer metrics from component: {}. Using unregistered metrics as fallback.", e);
                            Arc::new(Self::new_unregistered())
                        }
                    }
                })
                .clone()
        }

        #[cfg(not(feature = "metrics"))]
        {
            let _ = component;
            Arc::new(Self::new_unregistered())
        }
    }

    /// Creates a new KvIndexerMetrics which is not registered with a MetricsRegistry.
    /// This may be used for tests or as a fallback for when a MetricsRegistry is not available / has errored.
    #[cfg(feature = "metrics")]
    pub fn new_unregistered() -> Self {
        Self {
            kv_cache_events_applied: IntCounterVec::new(
                Opts::new(
                    KV_CACHE_EVENTS_APPLIED_NAME,
                    "Total number of KV cache events applied to index",
                ),
                &["event_type", "status"],
            )
            .unwrap(),
            kv_cache_event_warnings: IntCounterVec::new(
                Opts::new(
                    KV_CACHE_EVENT_WARNINGS_NAME,
                    "Total number of suspicious KV cache events seen by the router indexer",
                ),
                &["warning_kind"],
            )
            .unwrap(),
            indexer_ops_count: IntCounterVec::new(
                Opts::new(
                    KV_INDEXER_OPS_COUNT_NAME,
                    "Total count of KV indexer operations by type",
                ),
                &["operation"],
            )
            .unwrap(),
            indexer_ops_latency: CounterVec::new(
                Opts::new(
                    KV_INDEXER_OPS_LATENCY_NAME,
                    "Total duration of KV indexer operations in seconds",
                ),
                &["operation"],
            )
            .unwrap(),
        }
    }

    /// Creates a no-op metrics instance when Prometheus support is disabled.
    #[cfg(not(feature = "metrics"))]
    pub fn new_unregistered() -> Self {
        Self::default()
    }

    pub fn increment_event_applied(
        &self,
        event_type: &'static str,
        result: Result<(), KvCacheEventError>,
    ) {
        #[cfg(feature = "metrics")]
        {
            match result {
                Ok(_) => {
                    self.kv_cache_events_applied
                        .with_label_values(&[event_type, METRIC_STATUS_OK])
                        .inc_by(1);
                }
                Err(e) => {
                    let error_label = match e {
                        KvCacheEventError::ParentBlockNotFound => METRIC_STATUS_PARENT_NOT_FOUND,
                        KvCacheEventError::BlockNotFound => METRIC_STATUS_BLOCK_NOT_FOUND,
                        KvCacheEventError::InvalidBlockSequence => METRIC_STATUS_INVALID_BLOCK,
                    };
                    self.kv_cache_events_applied
                        .with_label_values(&[event_type, error_label])
                        .inc_by(1);
                }
            }
        }
        #[cfg(not(feature = "metrics"))]
        let _ = (self, event_type, result);
    }

    /// Record one indexer operation (count + total seconds) under `operation`.
    ///
    /// For hot worker-loop recording prefer the prebound
    /// [`PreBoundEventCounters::observe_event_op`] /
    /// [`PreBoundEventCounters::observe_remove_worker_op`]; this method does a
    /// `with_label_values` lookup and is intended for per-request paths such as
    /// `find_matches`.
    pub fn increment_indexer_op(&self, operation: &str, duration: std::time::Duration) {
        #[cfg(feature = "metrics")]
        {
            self.indexer_ops_count.with_label_values(&[operation]).inc();
            self.indexer_ops_latency
                .with_label_values(&[operation])
                .inc_by(duration.as_secs_f64());
        }
        #[cfg(not(feature = "metrics"))]
        let _ = (self, operation, duration);
    }

    pub fn increment_event_warning(&self, warning_kind: &'static str) {
        #[cfg(feature = "metrics")]
        {
            self.kv_cache_event_warnings
                .with_label_values(&[warning_kind])
                .inc_by(1);
        }
        #[cfg(not(feature = "metrics"))]
        let _ = (self, warning_kind);
    }

    /// Pre-resolve all `IntCounter` handles for the finite (event_type, status) label space.
    /// Call this once per worker thread at startup, then use
    /// [`PreBoundEventCounters::inc`] in the hot loop to avoid the
    /// `with_label_values` hashmap lookup on every event.
    pub fn prebind(&self) -> PreBoundEventCounters {
        PreBoundEventCounters::new(self)
    }
}

/// Pre-resolved `IntCounter` handles for every (event_type, status) combination.
///
/// Created once per worker thread via [`KvIndexerMetrics::prebind`], then used in
/// the event processing loop with a direct `.inc()` call instead of the
/// `IntCounterVec::with_label_values()` hashmap lookup.
pub struct PreBoundEventCounters {
    #[cfg(feature = "metrics")]
    inner: PreBoundMetricCounters,
}

#[cfg(feature = "metrics")]
struct PreBoundMetricCounters {
    stored: ResultCounters,
    removed: ResultCounters,
    cleared: ResultCounters,
    duplicate_store_warning: IntCounter,
    stored_op: PreBoundOpCounters,
    removed_op: PreBoundOpCounters,
    cleared_op: PreBoundOpCounters,
    remove_worker_op: PreBoundOpCounters,
}

/// Pre-resolved count + latency handles for one `operation` label value of the
/// `kv_indexer_ops_*` pair.
#[cfg(feature = "metrics")]
struct PreBoundOpCounters {
    count: IntCounter,
    latency_seconds: Counter,
}

#[cfg(feature = "metrics")]
impl PreBoundOpCounters {
    fn new(counts: &IntCounterVec, latencies: &CounterVec, operation: &'static str) -> Self {
        Self {
            count: counts.with_label_values(&[operation]),
            latency_seconds: latencies.with_label_values(&[operation]),
        }
    }

    fn observe(&self, elapsed: std::time::Duration) {
        self.count.inc();
        self.latency_seconds.inc_by(elapsed.as_secs_f64());
    }
}

#[cfg(feature = "metrics")]
struct ResultCounters {
    ok: IntCounter,
    parent_not_found: IntCounter,
    block_not_found: IntCounter,
    invalid_block: IntCounter,
}

#[cfg(feature = "metrics")]
impl ResultCounters {
    fn new(counters: &IntCounterVec, event_type: &'static str) -> Self {
        Self {
            ok: counters.with_label_values(&[event_type, METRIC_STATUS_OK]),
            parent_not_found: counters
                .with_label_values(&[event_type, METRIC_STATUS_PARENT_NOT_FOUND]),
            block_not_found: counters
                .with_label_values(&[event_type, METRIC_STATUS_BLOCK_NOT_FOUND]),
            invalid_block: counters.with_label_values(&[event_type, METRIC_STATUS_INVALID_BLOCK]),
        }
    }

    fn for_result(&self, result: Result<(), KvCacheEventError>) -> &IntCounter {
        match result {
            Ok(()) => &self.ok,
            Err(KvCacheEventError::ParentBlockNotFound) => &self.parent_not_found,
            Err(KvCacheEventError::BlockNotFound) => &self.block_not_found,
            Err(KvCacheEventError::InvalidBlockSequence) => &self.invalid_block,
        }
    }
}

impl PreBoundEventCounters {
    fn new(metrics: &KvIndexerMetrics) -> Self {
        #[cfg(feature = "metrics")]
        {
            let cv = &metrics.kv_cache_events_applied;
            let warnings = &metrics.kv_cache_event_warnings;
            let ops = &metrics.indexer_ops_count;
            let latencies = &metrics.indexer_ops_latency;
            Self {
                inner: PreBoundMetricCounters {
                    stored: ResultCounters::new(cv, METRIC_EVENT_STORED),
                    removed: ResultCounters::new(cv, METRIC_EVENT_REMOVED),
                    cleared: ResultCounters::new(cv, METRIC_EVENT_CLEARED),
                    duplicate_store_warning: warnings
                        .with_label_values(&[METRIC_WARNING_DUPLICATE_STORE]),
                    stored_op: PreBoundOpCounters::new(ops, latencies, METRIC_EVENT_STORED),
                    removed_op: PreBoundOpCounters::new(ops, latencies, METRIC_EVENT_REMOVED),
                    cleared_op: PreBoundOpCounters::new(ops, latencies, METRIC_EVENT_CLEARED),
                    remove_worker_op: PreBoundOpCounters::new(
                        ops,
                        latencies,
                        METRIC_OP_REMOVE_WORKER,
                    ),
                },
            }
        }
        #[cfg(not(feature = "metrics"))]
        {
            let _ = metrics;
            Self {}
        }
    }

    /// Increment the pre-resolved counter for the given event kind and result.
    ///
    /// Takes [`EventKind`] (a `Copy` discriminant) instead of a string label,
    /// so the compiler enforces exhaustiveness — a new [`EventKind`] or
    /// [`KvCacheEventError`] variant will produce a compile error here.
    pub fn inc(&self, kind: EventKind, result: Result<(), KvCacheEventError>) {
        #[cfg(feature = "metrics")]
        {
            let counters = match kind {
                EventKind::Stored => &self.inner.stored,
                EventKind::Removed => &self.inner.removed,
                EventKind::Cleared => &self.inner.cleared,
            };
            counters.for_result(result).inc();
        }
        #[cfg(not(feature = "metrics"))]
        let _ = (self, kind, result);
    }

    pub fn inc_warning(&self, kind: EventWarningKind) {
        #[cfg(feature = "metrics")]
        {
            let counter = match kind {
                EventWarningKind::DuplicateStore => &self.inner.duplicate_store_warning,
            };
            counter.inc();
        }
        #[cfg(not(feature = "metrics"))]
        let _ = (self, kind);
    }

    /// Record one `kv_indexer_ops_*` observation for an event apply, using the
    /// pre-resolved handles for the event kind's `operation` label.
    pub fn observe_event_op(&self, kind: EventKind, elapsed: std::time::Duration) {
        #[cfg(feature = "metrics")]
        {
            let op = match kind {
                EventKind::Stored => &self.inner.stored_op,
                EventKind::Removed => &self.inner.removed_op,
                EventKind::Cleared => &self.inner.cleared_op,
            };
            op.observe(elapsed);
        }
        #[cfg(not(feature = "metrics"))]
        let _ = (self, kind, elapsed);
    }

    /// Record one `kv_indexer_ops_*` observation for a worker removal
    /// (covers both whole-worker and per-DP-rank removals).
    pub fn observe_remove_worker_op(&self, elapsed: std::time::Duration) {
        #[cfg(feature = "metrics")]
        self.inner.remove_worker_op.observe(elapsed);
        #[cfg(not(feature = "metrics"))]
        let _ = (self, elapsed);
    }
}
