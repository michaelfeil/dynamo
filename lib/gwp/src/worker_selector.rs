// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Eligibility, sampling, and result construction around GWP's B10 scorer.

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
    KvSchedulerError, RoutingEligibility, SchedulingRequest, WorkerEligibilityError,
};
use dynamo_kv_router::selector::{WorkerSelector, softmax_sample};
use dynamo_llm::local_model::runtime_config::ModelRuntimeConfig;

use crate::config::LoadBalancingPolicy;
use crate::scoring::{B10Scorer, ObservedLoadStore, ScoreBreakdown};
#[cfg(test)]
use crate::scoring::{LocalLoadAnchor, ObservedWorkerLoad};

pub struct GwpWorkerSelector {
    scorer: B10Scorer,
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

impl GwpWorkerSelector {
    pub fn new(observed: ObservedLoadStore, policies: SelectionPolicyStore) -> Self {
        Self {
            scorer: B10Scorer::new(observed),
            policies,
            last_log_time_ms: AtomicU64::new(0),
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
        self.scorer.residency_eviction_half_life()
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
        let scoring = self.scorer.prepare(request, policy, block_size);
        let request = &scoring.request;
        let request_blocks = request.request_blocks(block_size);
        let verbose = self.should_log(workers.len());
        let temperature = scoring.temperature();

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
            let score = scoring.score(workers, worker);
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
            let score = scoring.score(workers, worker);
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

fn log_score(
    verbose: bool,
    worker: WorkerWithDpRank,
    score: ScoreBreakdown,
    taint_multiplier: f64,
) {
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
    fn planner_load_steers_away_from_loaded_worker() {
        let observed = ObservedLoadStore::default();
        observed.replace(HashMap::from([(
            1,
            ObservedWorkerLoad {
                decode_blocks: 100,
                ..Default::default()
            },
        )]));
        let selector = GwpWorkerSelector::new(observed, SelectionPolicyStore::default());
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
        let selector = GwpWorkerSelector::new(
            ObservedLoadStore::default(),
            SelectionPolicyStore::default(),
        );
        let worker = WorkerWithDpRank::from_worker_id(1);
        let mut request = request("no-trie");
        request.effective_overlap_blocks.insert(worker, 8.0);
        request.effective_cached_tokens.insert(worker, 32);
        request.router_config_override = Some(RouterConfigOverride {
            overlap_score_credit: Some(0.0),
            ..Default::default()
        });

        let scored = selector
            .scorer
            .prepare(&request, LoadBalancingPolicy::default(), 64);
        assert!(scored.request.effective_overlap_blocks.is_empty());
        assert!(scored.request.effective_cached_tokens.is_empty());
    }

    #[test]
    fn request_policy_can_choose_planner_or_local_signal() {
        let observed = ObservedLoadStore::default();
        observed.replace_refreshed(
            HashMap::from([(
                1,
                ObservedWorkerLoad {
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
        let selector = GwpWorkerSelector::new(observed, policies.clone());
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
