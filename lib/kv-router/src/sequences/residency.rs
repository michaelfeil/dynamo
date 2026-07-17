// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use dynamo_tokens::SequenceHash;
use lru::LruCache;
use rustc_hash::FxBuildHasher;
use tokio::time::Instant;

const DEFAULT_RESIDENCY_CAPACITY_BLOCKS: u64 = 50_000;
// LruCache stores a hash table plus linked-list nodes, so this cap is a CPU-router
// memory budget, not a device KV-cache size mirror.
const MAX_RESIDENCY_CAPACITY_BLOCKS: u64 = 200_000;
// Keep query-time boundary sampling bounded for all-worker scheduler scoring.
const LRU_SAMPLE_PROBE_LIMIT: usize = 64;
const LARGE_RESIDENCY_TRIM_BLOCKS: u64 = 10_000;

/// Tiny non-zero eviction cost reported for a tracked worker when no eviction
/// is predicted, so the worker appears in `eviction_costs` (and the routing
/// log's `rec` term is non-zero) whenever residency scoring is enabled.
const RESIDENCY_TRACKED_EPSILON: f64 = 0.0001;

type ResidencyCache = LruCache<SequenceHash, Instant, FxBuildHasher>;

#[derive(Debug)]
pub(crate) struct WorkerResidency {
    capacity_blocks: u64,
    blocks: ResidencyCache,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvictionPressure {
    pub resident_blocks: u64,
    pub capacity_blocks: Option<u64>,
    pub new_blocks: u64,
    pub would_evict_blocks: u64,
    /// Exact age of the oldest resident block that would be evicted.
    pub oldest_evicted_age: Option<Duration>,
    /// Exact age near the eviction boundary when it is cheap to probe; otherwise
    /// a bounded oldest-side sample.
    pub youngest_evicted_age: Option<Duration>,
    /// Estimated as `would_evict_blocks * average(endpoint recency costs)`.
    pub eviction_cost: f64,
}

impl WorkerResidency {
    pub(crate) fn new(capacity_blocks: Option<u64>) -> Self {
        let (requested_capacity, tracked_capacity) = tracked_capacity(capacity_blocks);
        warn_if_capacity_capped(requested_capacity, tracked_capacity);
        Self {
            capacity_blocks: tracked_capacity,
            blocks: LruCache::unbounded_with_hasher(FxBuildHasher),
        }
    }

    pub(crate) fn set_capacity(&mut self, capacity_blocks: Option<u64>) -> bool {
        let (requested_capacity, tracked_capacity) = tracked_capacity(capacity_blocks);
        if self.capacity_blocks == tracked_capacity {
            return false;
        }

        warn_if_capacity_capped(requested_capacity, tracked_capacity);

        self.capacity_blocks = tracked_capacity;
        let trimmed_blocks = self.trim_to_capacity();
        if trimmed_blocks >= LARGE_RESIDENCY_TRIM_BLOCKS {
            tracing::warn!(
                trimmed_blocks,
                tracked_capacity_blocks = tracked_capacity,
                "router residency capacity change trimmed a large number of blocks"
            );
        }
        true
    }

    pub(crate) fn touch_sequence_hashes(&mut self, sequence_hashes: &[SequenceHash], now: Instant) {
        if sequence_hashes.is_empty() {
            return;
        }

        for &sequence_hash in sequence_hashes {
            self.blocks.put(sequence_hash, now);
        }

        self.trim_to_capacity();
    }

    pub(crate) fn eviction_pressure_for_request_at(
        &self,
        estimated_cached_blocks: u64,
        request_blocks: u64,
        half_life: Duration,
        now: Instant,
    ) -> EvictionPressure {
        let estimated_new_blocks =
            request_blocks.saturating_sub(estimated_cached_blocks.min(request_blocks));
        self.eviction_pressure_for_estimated_new_blocks_at(estimated_new_blocks, half_life, now)
    }

    fn eviction_pressure_for_estimated_new_blocks_at(
        &self,
        estimated_new_blocks: u64,
        half_life: Duration,
        now: Instant,
    ) -> EvictionPressure {
        let resident_blocks = self.blocks.len() as u64;
        let capacity = self.capacity_blocks;
        let capacity_blocks = Some(capacity);

        let target_evictions = resident_blocks
            .saturating_add(estimated_new_blocks)
            .saturating_sub(capacity);
        let would_evict_blocks = target_evictions.min(resident_blocks);
        if would_evict_blocks == 0 {
            return EvictionPressure {
                resident_blocks,
                capacity_blocks,
                new_blocks: estimated_new_blocks,
                would_evict_blocks,
                oldest_evicted_age: None,
                youngest_evicted_age: None,
                eviction_cost: with_tracked_epsilon(0.0, capacity),
            };
        }

        let oldest_evicted_age = self
            .blocks
            .peek_lru()
            .map(|(_, last_access)| now.saturating_duration_since(*last_access));
        let youngest_evicted_age =
            self.sample_live_lru_age(would_evict_blocks.saturating_sub(1), now);
        let eviction_cost = estimate_eviction_cost(
            oldest_evicted_age,
            youngest_evicted_age,
            would_evict_blocks,
            half_life,
        );

        EvictionPressure {
            resident_blocks,
            capacity_blocks,
            new_blocks: estimated_new_blocks,
            would_evict_blocks,
            oldest_evicted_age,
            youngest_evicted_age,
            eviction_cost: with_tracked_epsilon(eviction_cost, capacity),
        }
    }

    #[cfg(test)]
    pub(crate) fn test_state(&self) -> (Option<u64>, u64) {
        (Some(self.capacity_blocks), self.blocks.len() as u64)
    }

    fn trim_to_capacity(&mut self) -> u64 {
        let mut trimmed_blocks = 0;
        while self.blocks.len() as u64 > self.capacity_blocks {
            if self.blocks.pop_lru().is_none() {
                break;
            }
            trimmed_blocks += 1;
        }
        trimmed_blocks
    }

    fn sample_live_lru_age(&self, live_rank: u64, now: Instant) -> Option<Duration> {
        let rank_from_oldest = live_rank as usize;
        let resident_blocks = self.blocks.len();
        if resident_blocks == 0 || rank_from_oldest >= resident_blocks {
            return None;
        }

        let last_access = if rank_from_oldest < LRU_SAMPLE_PROBE_LIMIT {
            self.blocks
                .iter()
                .rev()
                .nth(rank_from_oldest)
                .map(|(_, last_access)| *last_access)
        } else {
            let distance_from_mru = resident_blocks - rank_from_oldest - 1;
            if distance_from_mru < LRU_SAMPLE_PROBE_LIMIT {
                self.blocks
                    .iter()
                    .nth(distance_from_mru)
                    .map(|(_, last_access)| *last_access)
            } else {
                // The true boundary is too deep to probe cheaply. Use an oldest-side
                // sample instead of the MRU side to avoid over-penalizing this worker.
                self.blocks
                    .iter()
                    .rev()
                    .nth(LRU_SAMPLE_PROBE_LIMIT - 1)
                    .map(|(_, last_access)| *last_access)
            }
        }
        .or_else(|| self.blocks.peek_lru().map(|(_, last_access)| *last_access))?;
        Some(now.saturating_duration_since(last_access))
    }
}

impl EvictionPressure {
    pub(crate) fn empty(estimated_new_blocks: u64) -> Self {
        Self {
            resident_blocks: 0,
            capacity_blocks: None,
            new_blocks: estimated_new_blocks,
            would_evict_blocks: 0,
            oldest_evicted_age: None,
            youngest_evicted_age: None,
            eviction_cost: 0.0,
        }
    }
}

fn tracked_capacity(capacity_blocks: Option<u64>) -> (u64, u64) {
    let requested_capacity = capacity_blocks
        .filter(|capacity| *capacity > 0)
        .unwrap_or(DEFAULT_RESIDENCY_CAPACITY_BLOCKS);
    (
        requested_capacity,
        requested_capacity.min(MAX_RESIDENCY_CAPACITY_BLOCKS),
    )
}

fn warn_if_capacity_capped(requested_capacity: u64, tracked_capacity: u64) {
    if requested_capacity > tracked_capacity {
        tracing::warn!(
            requested_capacity_blocks = requested_capacity,
            tracked_capacity_blocks = tracked_capacity,
            "router residency capacity exceeds tracker budget; using capped capacity estimate"
        );
    }
}

fn with_tracked_epsilon(cost: f64, capacity: u64) -> f64 {
    if capacity > 0 {
        cost.max(RESIDENCY_TRACKED_EPSILON)
    } else {
        cost
    }
}

fn recency_cost(age: Duration, half_life: Duration) -> f64 {
    if half_life.is_zero() {
        return 1.0;
    }
    0.5_f64.powf(age.as_secs_f64() / half_life.as_secs_f64())
}

fn estimate_eviction_cost(
    oldest_evicted_age: Option<Duration>,
    youngest_evicted_age: Option<Duration>,
    would_evict_blocks: u64,
    half_life: Duration,
) -> f64 {
    if would_evict_blocks == 0 {
        return 0.0;
    }

    match (oldest_evicted_age, youngest_evicted_age) {
        (Some(oldest), Some(youngest)) => {
            let average_endpoint_cost =
                (recency_cost(oldest, half_life) + recency_cost(youngest, half_life)) / 2.0;
            would_evict_blocks as f64 * average_endpoint_cost
        }
        (Some(age), None) | (None, Some(age)) => {
            would_evict_blocks as f64 * recency_cost(age, half_life)
        }
        (None, None) => would_evict_blocks as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caps_lru_and_deduplicates_touches() {
        let mut residency = WorkerResidency::new(Some(2));
        let now = Instant::now();

        residency.touch_sequence_hashes(&[100, 200, 100], now);
        assert_eq!(residency.test_state().1, 2);

        residency.touch_sequence_hashes(&[300], now);
        assert_eq!(residency.test_state(), (Some(2), 2));
    }

    #[test]
    fn unknown_capacity_uses_default_budget() {
        let mut residency = WorkerResidency::new(None);

        residency.touch_sequence_hashes(&[100, 200], Instant::now());

        assert_eq!(
            residency.test_state(),
            (Some(DEFAULT_RESIDENCY_CAPACITY_BLOCKS), 2)
        );
    }

    #[test]
    fn eviction_pressure_decays_with_half_life() {
        let mut residency = WorkerResidency::new(Some(1));
        let now = Instant::now();
        residency.touch_sequence_hashes(&[100], now);

        let pressure = residency.eviction_pressure_for_estimated_new_blocks_at(
            1,
            Duration::from_secs(10),
            now + Duration::from_secs(10),
        );

        assert_eq!(pressure.resident_blocks, 1);
        assert_eq!(pressure.capacity_blocks, Some(1));
        assert_eq!(pressure.new_blocks, 1);
        assert_eq!(pressure.would_evict_blocks, 1);
        assert_eq!(pressure.oldest_evicted_age, Some(Duration::from_secs(10)));
        assert_eq!(pressure.youngest_evicted_age, Some(Duration::from_secs(10)));
        assert!((pressure.eviction_cost - 0.5).abs() < 0.000_001);
    }

    #[test]
    fn eviction_pressure_uses_estimated_cached_blocks() {
        let mut residency = WorkerResidency::new(Some(2));
        let now = Instant::now();
        residency.touch_sequence_hashes(&[100], now);
        residency.touch_sequence_hashes(&[200], now + Duration::from_secs(5));

        let pressure = residency.eviction_pressure_for_request_at(
            3,
            4,
            Duration::from_secs(60),
            now + Duration::from_secs(10),
        );

        assert_eq!(pressure.resident_blocks, 2);
        assert_eq!(pressure.new_blocks, 1);
        assert_eq!(pressure.would_evict_blocks, 1);
        assert_eq!(pressure.oldest_evicted_age, Some(Duration::from_secs(10)));
        assert_eq!(pressure.youngest_evicted_age, Some(Duration::from_secs(10)));
    }

    #[test]
    fn eviction_pressure_discounts_estimated_resident_request_blocks() {
        let mut residency = WorkerResidency::new(Some(80));
        let now = Instant::now();
        let resident_hashes: Vec<_> = (0..50).collect();
        residency.touch_sequence_hashes(&resident_hashes, now);

        let pressure =
            residency.eviction_pressure_for_request_at(50, 100, Duration::from_secs(60), now);

        assert_eq!(pressure.resident_blocks, 50);
        assert_eq!(pressure.new_blocks, 50);
        assert_eq!(pressure.would_evict_blocks, 20);
    }

    #[test]
    fn eviction_pressure_clamps_estimated_cached_blocks_to_request_blocks() {
        let mut residency = WorkerResidency::new(Some(8));
        let now = Instant::now();
        residency.touch_sequence_hashes(&[100, 200, 300, 400, 500], now);

        let pressure =
            residency.eviction_pressure_for_request_at(20, 10, Duration::from_secs(60), now);

        assert_eq!(pressure.new_blocks, 0);
        assert_eq!(pressure.would_evict_blocks, 0);
    }

    #[test]
    fn eviction_pressure_estimates_cost_from_eviction_endpoints() {
        let mut residency = WorkerResidency::new(Some(100));
        let now = Instant::now();
        for hash in 0..100 {
            residency.touch_sequence_hashes(&[hash], now + Duration::from_secs(hash));
        }

        let pressure = residency.eviction_pressure_for_estimated_new_blocks_at(
            100,
            Duration::from_secs(10),
            now + Duration::from_secs(100),
        );

        let expected_cost = 100.0
            * (recency_cost(Duration::from_secs(100), Duration::from_secs(10))
                + recency_cost(Duration::from_secs(1), Duration::from_secs(10)))
            / 2.0;

        assert_eq!(pressure.resident_blocks, 100);
        assert_eq!(pressure.new_blocks, 100);
        assert_eq!(pressure.would_evict_blocks, 100);
        assert_eq!(pressure.oldest_evicted_age, Some(Duration::from_secs(100)));
        assert_eq!(pressure.youngest_evicted_age, Some(Duration::from_secs(1)));
        assert!((pressure.eviction_cost - expected_cost).abs() < 0.000_001);
    }
}
