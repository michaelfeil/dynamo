// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! B10 scoring and observed/local load fusion.
//!
//! The topology provider supplies a delayed, cache-blind observed baseline.
//! The scheduler supplies immediate cache-aware local load. For each request,
//! B10 snapshots both inputs and the hot-reloaded routing configuration once,
//! then applies the signed local change since the observation anchor.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use dynamo_kv_router::protocols::{WorkerConfigLike, WorkerId, WorkerWithDpRank};
use dynamo_kv_router::scheduling::{IslStats, SchedulingRequest};
use dynamo_llm::kv_router::b10hotreloadablecm;
use dynamo_llm::local_model::runtime_config::ModelRuntimeConfig;
use parking_lot::RwLock;

use crate::config::LoadBalancingPolicy;

pub(crate) const MIN_ROUTER_TEMPERATURE: f64 = 1e-12;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ObservedWorkerLoad {
    pub prefill_tokens: usize,
    pub decode_blocks: usize,
    pub active_requests: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LocalLoadAnchor {
    pub prefill_tokens: usize,
    pub decode_blocks: usize,
    pub active_requests: usize,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ObservedLoadSnapshot {
    pub(crate) loads: HashMap<WorkerId, ObservedWorkerLoad>,
    pub(crate) anchors: HashMap<WorkerId, LocalLoadAnchor>,
}

#[derive(Clone, Default)]
pub struct ObservedLoadStore {
    inner: Arc<RwLock<ObservedLoadSnapshot>>,
}

impl ObservedLoadStore {
    pub fn replace(&self, loads: HashMap<WorkerId, ObservedWorkerLoad>) {
        self.replace_refreshed(loads, &HashMap::new());
    }

    pub(crate) fn replace_refreshed(
        &self,
        loads: HashMap<WorkerId, ObservedWorkerLoad>,
        refreshed_anchors: &HashMap<WorkerId, LocalLoadAnchor>,
    ) {
        let mut snapshot = self.inner.write();
        snapshot
            .anchors
            .retain(|worker_id, _| loads.contains_key(worker_id));
        snapshot
            .anchors
            .extend(refreshed_anchors.iter().map(|(id, anchor)| (*id, *anchor)));
        snapshot.loads = loads;
    }

    pub(crate) fn snapshot(&self) -> ObservedLoadSnapshot {
        self.inner.read().clone()
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ScoreBreakdown {
    pub(crate) logit: f64,
    pub(crate) prefill_blocks: f64,
    pub(crate) decode_blocks: f64,
    pub(crate) active_requests: f64,
    pub(crate) cache_miss_tokens: usize,
    pub(crate) residency_cost: f64,
    pub(crate) isl_penalty: f64,
}

pub(crate) struct B10Scorer {
    observed: ObservedLoadStore,
}

pub(crate) struct PreparedScoring {
    pub(crate) request: SchedulingRequest,
    observed_loads: HashMap<WorkerId, ObservedWorkerLoad>,
    anchors: HashMap<WorkerId, LocalLoadAnchor>,
    routing: b10hotreloadablecm::B10RoutingConfig,
    policy: LoadBalancingPolicy,
    block_size: u32,
}

impl B10Scorer {
    pub(crate) fn new(observed: ObservedLoadStore) -> Self {
        Self { observed }
    }

    pub(crate) fn prepare(
        &self,
        request: &SchedulingRequest,
        policy: LoadBalancingPolicy,
        block_size: u32,
    ) -> PreparedScoring {
        let observed = self.observed.snapshot();
        let locality_enabled = request
            .router_config_override
            .as_ref()
            .and_then(|config| config.overlap_score_credit)
            != Some(0.0);
        let request = SchedulingRequest {
            maybe_request_id: request.maybe_request_id.clone(),
            token_seq: request.token_seq.clone(),
            isl_tokens: request.isl_tokens,
            lora_name: request.lora_name.clone(),
            expected_output_tokens: request.expected_output_tokens,
            pinned_worker: request.pinned_worker,
            allowed_worker_ids: request.allowed_worker_ids.clone(),
            routing_constraints: request.routing_constraints.clone(),
            router_config_override: request.router_config_override.clone(),
            track_prefill_tokens: request.track_prefill_tokens,
            priority_jump: request.priority_jump,
            priority_load_shed_percent: request.priority_load_shed_percent,
            do_not_queue: request.do_not_queue,
            tier_overlap_blocks: if locality_enabled {
                request.tier_overlap_blocks.clone()
            } else {
                Default::default()
            },
            effective_overlap_blocks: if locality_enabled {
                request.effective_overlap_blocks.clone()
            } else {
                Default::default()
            },
            effective_cached_tokens: if locality_enabled {
                request.effective_cached_tokens.clone()
            } else {
                Default::default()
            },
            shared_cache_hits: locality_enabled
                .then(|| request.shared_cache_hits.clone())
                .flatten(),
            decode_blocks: request.decode_blocks.clone(),
            prefill_tokens: request.prefill_tokens.clone(),
            active_requests: request.active_requests.clone(),
            active_request_isl_stats: request.active_request_isl_stats.clone(),
            eviction_costs: request.eviction_costs.clone(),
            update_states: request.update_states,
            resp_tx: None,
        };
        Self::prepared(request, observed, policy, block_size)
    }

    fn prepared(
        request: SchedulingRequest,
        observed: ObservedLoadSnapshot,
        policy: LoadBalancingPolicy,
        block_size: u32,
    ) -> PreparedScoring {
        let routing = b10hotreloadablecm::get_config().get().routing;
        PreparedScoring {
            request,
            observed_loads: observed.loads,
            anchors: observed.anchors,
            routing,
            policy,
            block_size,
        }
    }

    pub(crate) fn residency_eviction_half_life(&self) -> Option<Duration> {
        let routing = b10hotreloadablecm::get_config().get().routing;
        (routing.router_residency_eviction_cost > 0.0)
            .then(|| Duration::from_secs_f64(routing.router_residency_half_life))
    }
}

impl PreparedScoring {
    pub(crate) fn temperature(&self) -> f64 {
        let temperature = self
            .policy
            .temperature
            .or_else(|| {
                self.request
                    .router_config_override
                    .as_ref()
                    .and_then(|config| config.router_temperature)
            })
            .unwrap_or(self.routing.router_temperature);
        if temperature.is_finite() && temperature > 0.0 {
            temperature.max(MIN_ROUTER_TEMPERATURE)
        } else {
            MIN_ROUTER_TEMPERATURE
        }
    }

    pub(crate) fn score(
        &self,
        workers: &HashMap<WorkerId, ModelRuntimeConfig>,
        worker: WorkerWithDpRank,
    ) -> ScoreBreakdown {
        let prefill_weight = self
            .policy
            .prefill_weight
            .or_else(|| {
                self.request
                    .router_config_override
                    .as_ref()
                    .and_then(|config| config.prefill_load_scale)
            })
            .unwrap_or(self.routing.router_overlap_score_weight);
        let observed_weight = self.policy.planner_weight.unwrap_or(1.0);
        let local_weight = self.policy.local_weight.unwrap_or(1.0);
        let observed = (worker.dp_rank == 0)
            .then(|| self.observed_loads.get(&worker.worker_id))
            .flatten()
            .copied()
            .unwrap_or_default();
        let anchor = (worker.dp_rank == 0)
            .then(|| self.anchors.get(&worker.worker_id))
            .flatten()
            .copied()
            .unwrap_or_default();
        let local_prefill = self.request.prefill_tokens_for(worker);
        let prefill_blocks = fuse_load(
            observed.prefill_tokens,
            local_prefill,
            anchor.prefill_tokens,
            observed_weight,
            local_weight,
        ) / self.block_size as f64;
        let local_decode = self
            .request
            .decode_blocks
            .get(&worker)
            .copied()
            .unwrap_or(prefill_blocks.floor() as usize);
        let decode_blocks = fuse_load(
            observed.decode_blocks,
            local_decode,
            anchor.decode_blocks,
            observed_weight,
            local_weight,
        );
        let cache_miss_tokens = if self.request.isl_tokens <= self.routing.router_cache_miss_min_isl
        {
            self.request.isl_tokens
        } else {
            self.request
                .isl_tokens
                .saturating_sub(self.request.effective_cached_tokens_for(worker))
        };
        let active_requests_worker = fuse_load(
            observed.active_requests,
            self.request.active_requests_for(worker),
            anchor.active_requests,
            observed_weight,
            local_weight,
        );
        let active_requests = active_requests_worker
            * (1.0 - self.routing.router_active_request_dp_blend)
            + mean_active_requests(workers, self, worker, observed_weight, local_weight)
                * self.routing.router_active_request_dp_blend;
        let residency_cost =
            self.routing.router_residency_eviction_cost * self.request.eviction_cost_for(worker);
        let isl_penalty = active_request_isl_penalty(
            &self.request,
            worker,
            self.routing.router_active_request_isl_mismatch_penalty,
            self.routing.router_active_request_isl_penalty_ramp,
        );
        let logit = prefill_weight * prefill_blocks
            + self
                .policy
                .decode_weight
                .unwrap_or(self.routing.router_decode_block_weight)
                * decode_blocks
            + self
                .policy
                .active_request_weight
                .unwrap_or(self.routing.router_active_request_weight)
                * active_requests
            + self
                .policy
                .cache_miss_weight
                .unwrap_or(self.routing.router_cache_miss_weight)
                * cache_miss_tokens as f64
            + residency_cost
            + isl_penalty;
        ScoreBreakdown {
            logit,
            prefill_blocks,
            decode_blocks,
            active_requests,
            cache_miss_tokens,
            residency_cost,
            isl_penalty,
        }
    }
}

/// Apply the signed local change since an external observation, while never
/// scoring below the current local view.
pub(crate) fn fuse_load(
    observed: usize,
    local: usize,
    anchor: usize,
    observed_weight: f64,
    local_weight: f64,
) -> f64 {
    let observed_component = observed_weight * observed as f64;
    let local_delta = local_weight * (local as f64 - anchor as f64);
    (observed_component + local_delta).max(local_weight * local as f64)
}

fn mean_active_requests(
    workers: &HashMap<WorkerId, ModelRuntimeConfig>,
    scoring: &PreparedScoring,
    worker: WorkerWithDpRank,
    observed_weight: f64,
    local_weight: f64,
) -> f64 {
    let Some(config) = workers.get(&worker.worker_id) else {
        return 0.0;
    };
    let start = config.data_parallel_start_rank();
    let end = start.saturating_add(config.data_parallel_size());
    if start == end {
        return 0.0;
    }
    (start..end)
        .map(|rank| {
            let candidate = WorkerWithDpRank::new(worker.worker_id, rank);
            let observed = (rank == 0)
                .then(|| scoring.observed_loads.get(&worker.worker_id))
                .flatten()
                .copied()
                .unwrap_or_default();
            let anchor = (rank == 0)
                .then(|| scoring.anchors.get(&worker.worker_id))
                .flatten()
                .copied()
                .unwrap_or_default();
            fuse_load(
                observed.active_requests,
                scoring.request.active_requests_for(candidate),
                anchor.active_requests,
                observed_weight,
                local_weight,
            )
        })
        .sum::<f64>()
        / (end - start) as f64
}

fn active_request_isl_penalty(
    request: &SchedulingRequest,
    worker: WorkerWithDpRank,
    mismatch_penalty: f64,
    ramp: (f64, f64),
) -> f64 {
    let (start, full) = ramp;
    if mismatch_penalty <= 0.0
        || !mismatch_penalty.is_finite()
        || start < 0.0
        || !start.is_finite()
        || full <= start
        || !full.is_finite()
    {
        return 0.0;
    }
    let Some(stats) = request.active_request_isl_stats.as_ref() else {
        return 0.0;
    };
    let factor = |tokens: f64| ((tokens - start) / (full - start)).clamp(0.0, 1.0);
    let penalty = |stats: &IslStats| {
        if stats.count == 0 || !stats.mean.is_finite() || !stats.stddev.is_finite() {
            return 0.0;
        }
        mismatch_penalty
            * factor((request.isl_tokens as f64 - stats.mean.max(0.0)).abs())
                .max(factor(stats.stddev.max(0.0)))
    };
    let worker_penalty = stats
        .by_worker_id
        .get(&worker.worker_id)
        .map(penalty)
        .unwrap_or(0.0);
    let rank_penalty = stats
        .by_worker_with_dp_rank
        .as_ref()
        .and_then(|by_rank| by_rank.get(&worker))
        .map(penalty)
        .unwrap_or(0.0);
    worker_penalty.max(rank_penalty)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observed_baseline_applies_signed_local_delta() {
        assert_eq!(fuse_load(100, 80, 80, 1.0, 1.0), 100.0);
        assert_eq!(fuse_load(100, 120, 80, 1.0, 1.0), 140.0);
        assert_eq!(fuse_load(20, 80, 80, 1.0, 1.0), 80.0);
        assert_eq!(fuse_load(100, 40, 80, 1.0, 1.0), 60.0);
    }

    #[test]
    fn source_weights_can_isolate_observed_or_local_load() {
        assert_eq!(fuse_load(100, 30, 20, 1.0, 0.0), 100.0);
        assert_eq!(fuse_load(100, 30, 20, 0.0, 1.0), 30.0);
    }
}
