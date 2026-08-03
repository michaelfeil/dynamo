// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GWP-owned worker scoring.
//!
//! Planner load is a delayed, cache-blind observation. The local scheduler is
//! immediate and cache-aware. The default score starts from the planner
//! snapshot and applies the signed change in GWP's local load since that
//! refresh boundary. This avoids double-counting while allowing both new
//! bookings and completions to affect routing before the next poll.

use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use dynamo_kv_router::protocols::{
    WorkerConfigLike, WorkerId, WorkerSelectionResult, WorkerWithDpRank,
};
use dynamo_kv_router::scheduling::{
    IslStats, KvSchedulerError, RoutingEligibility, SchedulingRequest, WorkerEligibilityError,
};
use dynamo_kv_router::selector::{WorkerSelector, softmax_sample};
use dynamo_llm::kv_router::b10hotreloadablecm;
use dynamo_llm::local_model::runtime_config::ModelRuntimeConfig;
use parking_lot::RwLock;

use crate::config::LoadBalancingPolicy;

const MIN_ROUTER_TEMPERATURE: f64 = 1e-12;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RemoteWorkerLoad {
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
pub(crate) struct RemoteLoadSnapshot {
    pub(crate) loads: HashMap<WorkerId, RemoteWorkerLoad>,
    pub(crate) anchors: HashMap<WorkerId, LocalLoadAnchor>,
}

#[derive(Clone, Default)]
pub struct RemoteLoadStore {
    inner: Arc<RwLock<RemoteLoadSnapshot>>,
}

impl RemoteLoadStore {
    pub fn replace(&self, loads: HashMap<WorkerId, RemoteWorkerLoad>) {
        self.replace_refreshed(loads, &HashMap::new());
    }

    pub(crate) fn replace_refreshed(
        &self,
        loads: HashMap<WorkerId, RemoteWorkerLoad>,
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

    pub(crate) fn snapshot(&self) -> RemoteLoadSnapshot {
        self.inner.read().clone()
    }
}

/// Applies the signed cache-aware local delta to the delayed cache-blind
/// planner baseline, while never scoring below the current local view.
#[cfg(test)]
fn reconcile_delayed_load(planner: usize, local: usize, anchor: usize) -> usize {
    planner.saturating_sub(anchor).saturating_add(local)
}

fn locality_enabled(request: &SchedulingRequest) -> bool {
    request
        .router_config_override
        .as_ref()
        .and_then(|config| config.overlap_score_credit)
        != Some(0.0)
}

#[derive(Debug, Clone, Copy)]
struct GwpScore {
    logit: f64,
    prefill_blocks: f64,
    decode_blocks: f64,
    active_requests: f64,
    cache_miss_tokens: usize,
    residency_cost: f64,
    isl_penalty: f64,
}

pub struct GwpWorkerSelector {
    remote: RemoteLoadStore,
    policies: SelectionPolicyStore,
    last_log_time_ms: AtomicU64,
}

#[derive(Clone, Default)]
pub(crate) struct SelectionPolicyStore {
    inner: Arc<dashmap::DashMap<String, LoadBalancingPolicy>>,
}

impl SelectionPolicyStore {
    pub(crate) fn install(
        &self,
        request_id: &str,
        policy: LoadBalancingPolicy,
    ) -> SelectionPolicyGuard {
        self.inner.insert(request_id.to_owned(), policy);
        SelectionPolicyGuard {
            store: self.clone(),
            request_id: request_id.to_owned(),
        }
    }

    pub(crate) fn remove(&self, request_id: &str) {
        self.inner.remove(request_id);
    }

    fn get(&self, request_id: Option<&str>) -> LoadBalancingPolicy {
        request_id
            .and_then(|id| self.inner.get(id).map(|entry| *entry.value()))
            .unwrap_or_default()
    }
}

pub(crate) struct SelectionPolicyGuard {
    store: SelectionPolicyStore,
    request_id: String,
}

impl Drop for SelectionPolicyGuard {
    fn drop(&mut self) {
        self.store.remove(&self.request_id);
    }
}

struct ScoringRequest {
    request: SchedulingRequest,
    planner_loads: HashMap<WorkerId, RemoteWorkerLoad>,
    anchors: HashMap<WorkerId, LocalLoadAnchor>,
}

impl GwpWorkerSelector {
    pub fn new(remote: RemoteLoadStore, policies: SelectionPolicyStore) -> Self {
        Self {
            remote,
            policies,
            last_log_time_ms: AtomicU64::new(0),
        }
    }

    fn scoring_request(&self, request: &SchedulingRequest) -> ScoringRequest {
        let remote = self.remote.snapshot();
        let locality_enabled = locality_enabled(request);

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
        ScoringRequest {
            request,
            planner_loads: remote.loads,
            anchors: remote.anchors,
        }
    }

    fn should_log(&self, worker_count: usize) -> bool {
        let interval = if worker_count > 5 {
            selection_log_interval_ms()
        } else {
            100
        };
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let previous = self.last_log_time_ms.load(Ordering::Relaxed);
        now.saturating_sub(previous) >= interval
            && self
                .last_log_time_ms
                .compare_exchange(previous, now, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
    }
}

/// Throttle interval for per-worker scoring logs when the pool is large.
/// Read once from the same setting as the B10 selector and default to 2000ms.
fn selection_log_interval_ms() -> u64 {
    static INTERVAL: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *INTERVAL.get_or_init(|| {
        std::env::var("B10_KV_ROUTER_SELECTION_LOG_INTERVAL_MS")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|&value| value > 0)
            .unwrap_or(2_000)
    })
}

fn weighted_load(
    planner: usize,
    local: usize,
    anchor: usize,
    planner_weight: f64,
    local_weight: f64,
) -> f64 {
    let planner_component = planner_weight * planner as f64;
    let local_delta = local_weight * (local as f64 - anchor as f64);
    (planner_component + local_delta).max(local_weight * local as f64)
}

fn mean_active_requests<C: WorkerConfigLike>(
    workers: &HashMap<WorkerId, C>,
    scoring: &ScoringRequest,
    worker: WorkerWithDpRank,
    planner_weight: f64,
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
            let planner = (rank == 0)
                .then(|| scoring.planner_loads.get(&worker.worker_id))
                .flatten()
                .copied()
                .unwrap_or_default();
            let anchor = (rank == 0)
                .then(|| scoring.anchors.get(&worker.worker_id))
                .flatten()
                .copied()
                .unwrap_or_default();
            weighted_load(
                planner.active_requests,
                scoring.request.active_requests_for(candidate),
                anchor.active_requests,
                planner_weight,
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

fn score_worker<C: WorkerConfigLike>(
    workers: &HashMap<WorkerId, C>,
    scoring: &ScoringRequest,
    worker: WorkerWithDpRank,
    block_size: u32,
    policy: LoadBalancingPolicy,
) -> GwpScore {
    let request = &scoring.request;
    let config = b10hotreloadablecm::get_config().get();
    let routing = &config.routing;
    let prefill_weight = policy
        .prefill_weight
        .or_else(|| {
            request
                .router_config_override
                .as_ref()
                .and_then(|config| config.prefill_load_scale)
        })
        .unwrap_or(routing.router_overlap_score_weight);
    let planner_weight = policy.planner_weight.unwrap_or(1.0);
    let local_weight = policy.local_weight.unwrap_or(1.0);
    let planner = (worker.dp_rank == 0)
        .then(|| scoring.planner_loads.get(&worker.worker_id))
        .flatten()
        .copied()
        .unwrap_or_default();
    let anchor = (worker.dp_rank == 0)
        .then(|| scoring.anchors.get(&worker.worker_id))
        .flatten()
        .copied()
        .unwrap_or_default();
    let local_prefill = request.prefill_tokens_for(worker);
    let prefill_blocks = weighted_load(
        planner.prefill_tokens,
        local_prefill,
        anchor.prefill_tokens,
        planner_weight,
        local_weight,
    ) / block_size as f64;
    let local_decode = request
        .decode_blocks
        .get(&worker)
        .copied()
        .unwrap_or(prefill_blocks.floor() as usize);
    let decode_blocks = weighted_load(
        planner.decode_blocks,
        local_decode,
        anchor.decode_blocks,
        planner_weight,
        local_weight,
    );
    let cache_miss_tokens = if request.isl_tokens <= routing.router_cache_miss_min_isl {
        request.isl_tokens
    } else {
        request
            .isl_tokens
            .saturating_sub(request.effective_cached_tokens_for(worker))
    };
    let active_requests_worker = weighted_load(
        planner.active_requests,
        request.active_requests_for(worker),
        anchor.active_requests,
        planner_weight,
        local_weight,
    );
    let active_requests = active_requests_worker * (1.0 - routing.router_active_request_dp_blend)
        + mean_active_requests(workers, scoring, worker, planner_weight, local_weight)
            * routing.router_active_request_dp_blend;
    let residency_cost = routing.router_residency_eviction_cost * request.eviction_cost_for(worker);
    let isl_penalty = active_request_isl_penalty(
        request,
        worker,
        routing.router_active_request_isl_mismatch_penalty,
        routing.router_active_request_isl_penalty_ramp,
    );
    let logit = prefill_weight * prefill_blocks
        + policy
            .decode_weight
            .unwrap_or(routing.router_decode_block_weight)
            * decode_blocks
        + policy
            .active_request_weight
            .unwrap_or(routing.router_active_request_weight)
            * active_requests
        + policy
            .cache_miss_weight
            .unwrap_or(routing.router_cache_miss_weight)
            * cache_miss_tokens as f64
        + residency_cost
        + isl_penalty;
    GwpScore {
        logit,
        prefill_blocks,
        decode_blocks,
        active_requests,
        cache_miss_tokens,
        residency_cost,
        isl_penalty,
    }
}

fn strict_dp_rank(
    scores: impl IntoIterator<Item = (WorkerWithDpRank, f64)>,
    selected: WorkerWithDpRank,
) -> bool {
    let mut selected_score = None;
    let mut worst_alternative = f64::NEG_INFINITY;
    for (worker, score) in scores {
        if worker.worker_id != selected.worker_id {
            continue;
        }
        if worker == selected {
            selected_score = Some(score);
        } else {
            worst_alternative = worst_alternative.max(score);
        }
    }
    let Some(selected_score) = selected_score else {
        return false;
    };
    selected_score.is_finite()
        && worst_alternative.is_finite()
        && worst_alternative - selected_score > 0.05 * selected_score.abs().max(1e-9)
}

impl WorkerSelector<ModelRuntimeConfig> for GwpWorkerSelector {
    fn residency_eviction_half_life(&self) -> Option<Duration> {
        let config = b10hotreloadablecm::get_config().get();
        (config.routing.router_residency_eviction_cost > 0.0)
            .then(|| Duration::from_secs_f64(config.routing.router_residency_half_life))
    }

    fn select_worker(
        &self,
        workers: &HashMap<WorkerId, ModelRuntimeConfig>,
        request: &SchedulingRequest,
        eligibility: RoutingEligibility<'_>,
        block_size: u32,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        assert!(request.isl_tokens > 0);
        if eligibility.pinned_worker().is_none()
            && !eligibility.has_eligible_worker(workers.iter().map(|(&id, config)| (id, config)))
        {
            if eligibility.has_eligible_worker_ignoring_overload(
                workers.iter().map(|(&id, config)| (id, config)),
            ) {
                return Err(KvSchedulerError::AllEligibleWorkersOverloaded);
            }
            return Err(KvSchedulerError::NoEndpoints);
        }

        let policy = self.policies.get(request.maybe_request_id.as_deref());
        let scoring = self.scoring_request(request);
        let request = &scoring.request;
        let request_blocks = request.request_blocks(block_size);
        let verbose = self.should_log(workers.len());
        let config = b10hotreloadablecm::get_config().get();
        let temperature = policy
            .temperature
            .or_else(|| {
                request
                    .router_config_override
                    .as_ref()
                    .and_then(|config| config.router_temperature)
            })
            .unwrap_or(config.routing.router_temperature);
        let temperature = if temperature.is_finite() && temperature > 0.0 {
            temperature.max(MIN_ROUTER_TEMPERATURE)
        } else {
            MIN_ROUTER_TEMPERATURE
        };

        if let Some(worker) = eligibility.pinned_worker() {
            match eligibility.validate_worker_rank(workers, worker) {
                Ok(_) => {}
                Err(WorkerEligibilityError::WorkerNotAllowed { .. }) => {
                    return Err(KvSchedulerError::PinnedWorkerNotAllowed {
                        worker_id: worker.worker_id,
                    });
                }
                Err(WorkerEligibilityError::WorkerOverloaded { .. }) => {
                    return Err(KvSchedulerError::PinnedWorkerOverloaded {
                        worker_id: worker.worker_id,
                    });
                }
                Err(_) => return Err(KvSchedulerError::NoEndpoints),
            }
            let score = score_worker(workers, &scoring, worker, block_size, policy);
            log_score(verbose, worker, score, 1.0);
            return Ok(WorkerSelectionResult {
                worker,
                required_blocks: request_blocks,
                effective_overlap_blocks: request.effective_overlap_blocks_for(worker),
                cached_tokens: request.effective_cached_tokens_for(worker),
                dp_strict_rank: true,
            });
        }

        let mut scores = HashMap::new();
        eligibility.for_each_eligible_worker_rank(workers, |worker, worker_config| {
            let score = score_worker(workers, &scoring, worker, block_size, policy);
            let taint_multiplier = request
                .routing_constraints
                .preferred_taint_multiplier(worker_config.taints())
                .unwrap_or(1.0);
            log_score(verbose, worker, score, taint_multiplier);
            scores.insert(worker, (score.logit + 1.0) * taint_multiplier);
        });
        if scores.is_empty() {
            return Err(KvSchedulerError::NoEndpoints);
        }
        let (worker, _score) = softmax_sample(&scores, temperature);
        if verbose {
            tracing::info!(
                worker_id = worker.worker_id,
                dp_rank = ?worker.dp_rank,
                cached_tokens = request.effective_cached_tokens_for(worker),
                "GwpWorkerSelector selected worker"
            );
        }
        Ok(WorkerSelectionResult {
            worker,
            required_blocks: request_blocks,
            effective_overlap_blocks: request.effective_overlap_blocks_for(worker),
            cached_tokens: request.effective_cached_tokens_for(worker),
            dp_strict_rank: strict_dp_rank(scores, worker),
        })
    }
}

fn log_score(verbose: bool, worker: WorkerWithDpRank, score: GwpScore, taint_multiplier: f64) {
    if !verbose {
        return;
    }
    tracing::info!(
        worker_id = worker.worker_id,
        dp_rank = ?worker.dp_rank,
        logit = score.logit,
        taint_multiplier,
        prefill_blocks = score.prefill_blocks,
        decode_blocks = score.decode_blocks,
        active_requests = score.active_requests,
        cache_miss_tokens = score.cache_miss_tokens,
        residency_cost = score.residency_cost,
        isl_penalty = score.isl_penalty,
        "GWP worker score"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::config::RouterConfigOverride;
    use dynamo_kv_router::protocols::RoutingConstraints;
    use dynamo_kv_router::scheduling::TierOverlapBlocks;

    fn request(request_id: &str) -> SchedulingRequest {
        SchedulingRequest {
            maybe_request_id: Some(request_id.into()),
            token_seq: None,
            isl_tokens: 64,
            lora_name: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            router_config_override: None,
            track_prefill_tokens: true,
            priority_jump: 0.0,
            priority_load_shed_percent: 0,
            do_not_queue: true,
            tier_overlap_blocks: TierOverlapBlocks::default(),
            effective_overlap_blocks: HashMap::new(),
            effective_cached_tokens: HashMap::new(),
            shared_cache_hits: None,
            decode_blocks: Default::default(),
            prefill_tokens: Default::default(),
            active_requests: Default::default(),
            active_request_isl_stats: None,
            eviction_costs: HashMap::new(),
            update_states: false,
            resp_tx: None,
        }
    }

    #[test]
    fn delayed_planner_baseline_applies_signed_local_delta() {
        assert_eq!(reconcile_delayed_load(100, 80, 80), 100);
        assert_eq!(reconcile_delayed_load(100, 120, 80), 140);
        assert_eq!(reconcile_delayed_load(20, 80, 80), 80);
        assert_eq!(reconcile_delayed_load(100, 40, 80), 60);
    }

    #[test]
    fn planner_load_steers_away_from_loaded_worker() {
        let remote = RemoteLoadStore::default();
        remote.replace(HashMap::from([(
            1,
            RemoteWorkerLoad {
                decode_blocks: 100,
                ..Default::default()
            },
        )]));
        let selector = GwpWorkerSelector::new(remote, SelectionPolicyStore::default());
        let workers = HashMap::from([
            (1, ModelRuntimeConfig::default()),
            (2, ModelRuntimeConfig::default()),
        ]);
        let request = request("planner-load");
        let selected = selector
            .select_worker(&workers, &request, request.eligibility(), 64)
            .expect("select");
        assert_eq!(selected.worker.worker_id, 2);
    }

    #[test]
    fn trie_disabled_scrubs_locality_fields() {
        let selector =
            GwpWorkerSelector::new(RemoteLoadStore::default(), SelectionPolicyStore::default());
        let worker = WorkerWithDpRank::from_worker_id(1);
        let mut request = request("no-trie");
        request.effective_overlap_blocks.insert(worker, 8.0);
        request.effective_cached_tokens.insert(worker, 32);
        request.router_config_override = Some(RouterConfigOverride {
            overlap_score_credit: Some(0.0),
            ..Default::default()
        });

        let scored = selector.scoring_request(&request);
        assert!(scored.request.effective_overlap_blocks.is_empty());
        assert!(scored.request.effective_cached_tokens.is_empty());
    }

    #[test]
    fn request_policy_can_choose_planner_or_local_signal() {
        let remote = RemoteLoadStore::default();
        remote.replace_refreshed(
            HashMap::from([(
                1,
                RemoteWorkerLoad {
                    decode_blocks: 100,
                    ..Default::default()
                },
            )]),
            &HashMap::from([(
                1,
                LocalLoadAnchor {
                    decode_blocks: 20,
                    ..Default::default()
                },
            )]),
        );
        let policies = SelectionPolicyStore::default();
        let selector = GwpWorkerSelector::new(remote, policies.clone());
        let worker = WorkerWithDpRank::from_worker_id(1);
        let other_worker = WorkerWithDpRank::from_worker_id(2);
        let mut request = request("weighted");
        request.decode_blocks.insert(worker, 0);
        request.decode_blocks.insert(other_worker, 30);
        let workers = HashMap::from([
            (1, ModelRuntimeConfig::default()),
            (2, ModelRuntimeConfig::default()),
        ]);

        let planner_guard = policies.install(
            "weighted",
            LoadBalancingPolicy {
                planner_weight: Some(1.0),
                local_weight: Some(0.0),
                prefill_weight: Some(0.0),
                decode_weight: Some(1.0),
                active_request_weight: Some(0.0),
                cache_miss_weight: Some(0.0),
                ..Default::default()
            },
        );
        let planner_only = selector
            .select_worker(&workers, &request, request.eligibility(), 1)
            .expect("planner-only selection");
        assert_eq!(planner_only.worker, other_worker);
        drop(planner_guard);

        let _local_guard = policies.install(
            "weighted",
            LoadBalancingPolicy {
                planner_weight: Some(0.0),
                local_weight: Some(1.0),
                prefill_weight: Some(0.0),
                decode_weight: Some(1.0),
                active_request_weight: Some(0.0),
                cache_miss_weight: Some(0.0),
                ..Default::default()
            },
        );
        let local_only = selector
            .select_worker(&workers, &request, request.eligibility(), 1)
            .expect("local-only selection");
        assert_eq!(local_only.worker, worker);
    }
}
