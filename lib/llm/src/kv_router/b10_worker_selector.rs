// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! B10 Worker Selector with hot-reloadable configuration
//!
//! Adapts upstream's `DefaultWorkerSelector` cost function with
//! absolute cache-miss weighting and short-request bypass, sourcing
//! tuning knobs from the hot-reloadable config manager.

use crate::kv_router::b10hotreloadablecm;
use crate::local_model::runtime_config::ModelRuntimeConfig;
use dynamo_kv_router::protocols::{
    WorkerConfigLike, WorkerId, WorkerSelectionResult, WorkerWithDpRank,
};
use dynamo_kv_router::scheduling::{
    KvSchedulerError, RoutingEligibility, SchedulingRequest, WorkerEligibilityError,
};
use dynamo_kv_router::selector::{WorkerSelector, softmax_sample};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Returns whether DP routing should be strict for the already-selected DP rank.
///
/// Lower scores are better. Strict routing is used when the selected rank is
/// materially better than the worst same-worker alternative; otherwise routing
/// can relax the rank with limited expected impact.
fn b10_filter_dp_score(
    all_scores: impl IntoIterator<Item = (WorkerWithDpRank, f64)>,
    worker: WorkerWithDpRank,
) -> bool {
    const DP_ROUTING_THRESHOLD_PCT: f64 = 0.05;

    let mut selected_score = None;
    let mut worst_alternative = f64::NEG_INFINITY;

    for (candidate, score) in all_scores {
        if candidate.worker_id != worker.worker_id {
            continue;
        }
        if candidate == worker {
            selected_score = Some(score);
        } else {
            worst_alternative = worst_alternative.max(score);
        }
    }

    let Some(selected_score) = selected_score else {
        return false;
    };
    if !selected_score.is_finite() || !worst_alternative.is_finite() {
        return false;
    }

    let denominator = selected_score.abs().max(1e-9);
    let advantage_vs_worst = worst_alternative - selected_score;

    advantage_vs_worst > DP_ROUTING_THRESHOLD_PCT * denominator
}

/// B10 Worker Selector that uses hot-reloadable configuration.
#[derive(Debug)]
pub struct B10WorkerSelector {
    last_log_time_ms: AtomicU64,
}

impl Default for B10WorkerSelector {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy)]
struct B10Score {
    logit: f64,
    potential_prefill_block: f64,
    decode_block: f64,
    active_requests: f64,
    active_request_dp_blend: f64,
    cache_miss_absolute_tokens: usize,
}

impl B10WorkerSelector {
    pub fn new() -> Self {
        Self {
            last_log_time_ms: AtomicU64::new(0),
        }
    }

    /// Check if we should print this loop iteration (rate-limited based on worker count).
    /// If workers > 5, print at most once every 2000ms, else once every 100ms.
    fn should_print_this_loop(&self, num_workers: usize) -> bool {
        let log_interval_ms = if num_workers > 5 { 2000 } else { 100 };

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let last_log = self.last_log_time_ms.load(Ordering::Relaxed);

        if now_ms.saturating_sub(last_log) < log_interval_ms {
            return false;
        }

        self.last_log_time_ms
            .compare_exchange(last_log, now_ms, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
    }
}

fn mean_active_requests_for_worker<C: WorkerConfigLike>(
    workers: &HashMap<WorkerId, C>,
    request: &SchedulingRequest,
    worker: WorkerWithDpRank,
) -> f64 {
    let Some(config) = workers.get(&worker.worker_id) else {
        return 0.0;
    };
    let dp_start = config.data_parallel_start_rank();
    let dp_end = dp_start.saturating_add(config.data_parallel_size());
    if dp_start == dp_end {
        return 0.0;
    }

    let mut total_active_requests = 0usize;
    let mut total_dp_workers = 0usize;
    for dp_rank in dp_start..dp_end {
        let worker_with_dp = WorkerWithDpRank::new(worker.worker_id, dp_rank);
        total_active_requests += request.active_requests_for(worker_with_dp);
        total_dp_workers += 1;
    }

    total_active_requests as f64 / total_dp_workers as f64
}

#[allow(clippy::too_many_arguments)]
fn score_worker<C: WorkerConfigLike>(
    workers: &HashMap<WorkerId, C>,
    request: &SchedulingRequest,
    worker: WorkerWithDpRank,
    block_size: u32,
    overlap_weight: f64,
    cache_miss_weight: f64,
    cache_miss_min_isl: usize,
    active_request_weight: f64,
    active_request_dp_blend: f64,
) -> B10Score {
    let prefill_token = request.prefill_tokens_for(worker);
    let potential_prefill_block = (prefill_token as f64) / (block_size as f64);
    let decode_block_fallback = potential_prefill_block.floor() as usize;
    let decode_block = request
        .decode_blocks
        .get(&worker)
        .copied()
        .unwrap_or(decode_block_fallback) as f64;

    // Absolute cache-miss tokens: ISL minus device-resident cache hit tokens.
    // For short requests below `cache_miss_min_isl`, treat the request as a
    // full miss so load balancing dominates over cache locality.
    let cache_miss_absolute_tokens: usize = if request.isl_tokens <= cache_miss_min_isl {
        request.isl_tokens
    } else {
        request
            .isl_tokens
            .saturating_sub(request.effective_cached_tokens_for(worker))
    };

    let active_requests_worker = request.active_requests_for(worker) as f64;
    let active_requests = active_requests_worker * (1.0 - active_request_dp_blend)
        + mean_active_requests_for_worker(workers, request, worker) * active_request_dp_blend;

    let logit = overlap_weight * potential_prefill_block
        + decode_block
        + active_request_weight * active_requests
        + cache_miss_weight * (cache_miss_absolute_tokens as f64);

    B10Score {
        logit,
        potential_prefill_block,
        decode_block,
        active_requests,
        active_request_dp_blend,
        cache_miss_absolute_tokens,
    }
}

impl WorkerSelector<ModelRuntimeConfig> for B10WorkerSelector {
    fn select_worker(
        &self,
        workers: &HashMap<WorkerId, ModelRuntimeConfig>,
        request: &SchedulingRequest,
        eligibility: RoutingEligibility<'_>,
        block_size: u32,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        assert!(request.isl_tokens > 0);

        let pinned_worker = eligibility.pinned_worker();

        if pinned_worker.is_none()
            && !eligibility.has_eligible_worker(
                workers
                    .iter()
                    .map(|(&worker_id, config)| (worker_id, config)),
            )
        {
            if eligibility.has_eligible_worker_ignoring_overload(
                workers
                    .iter()
                    .map(|(&worker_id, config)| (worker_id, config)),
            ) {
                return Err(KvSchedulerError::AllEligibleWorkersOverloaded);
            }

            return Err(KvSchedulerError::NoEndpoints);
        }

        let request_blocks = request.request_blocks(block_size);
        let hot_reloadable_config = b10hotreloadablecm::get_config().get();
        let verbose = self.should_print_this_loop(workers.len());

        let overlap_weight = request
            .router_config_override
            .as_ref()
            .and_then(|cfg| cfg.prefill_load_scale)
            .unwrap_or(hot_reloadable_config.routing.router_overlap_score_weight);

        let cache_miss_weight = hot_reloadable_config.routing.router_cache_miss_weight;
        let cache_miss_min_isl = hot_reloadable_config.routing.router_cache_miss_min_isl;
        let active_request_weight = hot_reloadable_config.routing.router_active_request_weight;
        let active_request_dp_blend = hot_reloadable_config.routing.router_active_request_dp_blend;
        let temperature = request
            .router_config_override
            .as_ref()
            .and_then(|cfg| cfg.router_temperature)
            .unwrap_or(hot_reloadable_config.routing.router_temperature);

        let score_worker = |worker: WorkerWithDpRank| -> B10Score {
            score_worker(
                workers,
                request,
                worker,
                block_size,
                overlap_weight,
                cache_miss_weight,
                cache_miss_min_isl,
                active_request_weight,
                active_request_dp_blend,
            )
        };

        if let Some(worker) = pinned_worker {
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

            let score = score_worker(worker);
            let effective_overlap_blocks = request.effective_overlap_blocks_for(worker);
            let cached_tokens = request.effective_cached_tokens_for(worker);
            if verbose {
                tracing::info!(
                    "worker_id={} dp={:?} logit={:.3} | ow={:.2}*ppf={:.2} + db={:.2} + arw={:.2}*ar={:.2}(dpb={:.2}) + cmw={:.2}*cm={}",
                    worker.worker_id,
                    worker.dp_rank,
                    score.logit,
                    overlap_weight,
                    score.potential_prefill_block,
                    score.decode_block,
                    active_request_weight,
                    score.active_requests,
                    score.active_request_dp_blend,
                    cache_miss_weight,
                    score.cache_miss_absolute_tokens
                );
            }

            tracing::info!(
                "B10WorkerSelector selected pinned worker: worker_id={} dp_rank={:?}, logit: {:.3}, effective cached blocks: {:.2}",
                worker.worker_id,
                worker.dp_rank,
                score.logit,
                effective_overlap_blocks,
            );

            return Ok(WorkerSelectionResult {
                worker,
                required_blocks: request_blocks,
                effective_overlap_blocks,
                cached_tokens,
                dp_strict_rank: true,
            });
        }

        let mut worker_logits: HashMap<WorkerWithDpRank, f64> = HashMap::default();
        eligibility.for_each_eligible_worker_rank(workers, |worker, _| {
            let score = score_worker(worker);
            if verbose {
                tracing::info!(
                    "worker_id={} dp={:?} logit={:.3} | ow={:.2}*ppf={:.2} + db={:.2} + arw={:.2}*ar={:.2}(dpb={:.2}) + cmw={:.2}*cm={}",
                    worker.worker_id,
                    worker.dp_rank,
                    score.logit,
                    overlap_weight,
                    score.potential_prefill_block,
                    score.decode_block,
                    active_request_weight,
                    score.active_requests,
                    score.active_request_dp_blend,
                    cache_miss_weight,
                    score.cache_miss_absolute_tokens
                );
            }
            worker_logits.insert(worker, score.logit);
        });

        if worker_logits.is_empty() {
            return Err(KvSchedulerError::NoEndpoints);
        }

        let (best_worker, best_logit) = if temperature == 0.0 {
            let min_logit = worker_logits
                .values()
                .copied()
                .fold(f64::INFINITY, f64::min);
            worker_logits
                .iter()
                .filter(|(_, logit)| **logit == min_logit)
                .min_by_key(|(worker, _)| (worker.worker_id, worker.dp_rank))
                .map(|(worker, logit)| (*worker, *logit))
                .expect("worker_logits non-empty")
        } else {
            softmax_sample(&worker_logits, temperature)
        };
        let effective_overlap_blocks = request.effective_overlap_blocks_for(best_worker);
        let cached_tokens = request.effective_cached_tokens_for(best_worker);

        tracing::info!(
            "B10WorkerSelector selected worker: worker_id={} dp_rank={:?}, logit: {:.3}, effective cached blocks: {:.2}",
            best_worker.worker_id,
            best_worker.dp_rank,
            best_logit,
            effective_overlap_blocks,
        );

        Ok(WorkerSelectionResult {
            worker: best_worker,
            required_blocks: request_blocks,
            effective_overlap_blocks,
            cached_tokens,
            dp_strict_rank: b10_filter_dp_score(
                worker_logits
                    .iter()
                    .map(|(worker, logit)| (*worker, *logit)),
                best_worker,
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::RouterConfigOverride;

    #[test]
    fn test_b10_worker_selector_creation() {
        let _selector = B10WorkerSelector::new();
    }

    fn test_worker_config(dp_start: u32, dp_size: u32) -> ModelRuntimeConfig {
        ModelRuntimeConfig {
            data_parallel_start_rank: dp_start,
            data_parallel_size: dp_size,
            ..Default::default()
        }
    }

    fn base_request(isl_tokens: usize) -> SchedulingRequest {
        SchedulingRequest {
            maybe_request_id: Some("test".into()),
            token_seq: None,
            isl_tokens,
            tier_overlap_blocks: Default::default(),
            effective_overlap_blocks: HashMap::default(),
            effective_cached_tokens: HashMap::default(),
            decode_blocks: Default::default(),
            prefill_tokens: Default::default(),
            active_requests: HashMap::new(),
            track_prefill_tokens: true,
            router_config_override: None,
            update_states: false,
            lora_name: None,
            priority_jump: 0.0,
            priority_load_shed_percent: 0,
            do_not_queue: false,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: dynamo_kv_router::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: None,
        }
    }

    #[test]
    fn score_uses_token_level_cache_miss_not_block_rounding() {
        let worker = WorkerWithDpRank::new(1, 0);
        let workers = HashMap::from([(worker.worker_id, test_worker_config(0, 1))]);
        let mut request = base_request(65);
        request.effective_cached_tokens.insert(worker, 63);
        request.prefill_tokens.insert(worker, 0);

        let score = score_worker(&workers, &request, worker, 64, 0.0, 1.0, 0, 0.0, 2.0 / 3.0);

        assert_eq!(score.cache_miss_absolute_tokens, 2);
        assert_eq!(score.logit, 2.0);
    }

    #[test]
    fn score_blends_selected_dp_and_worker_mean_active_requests() {
        let worker0 = WorkerWithDpRank::new(1, 0);
        let worker1 = WorkerWithDpRank::new(1, 1);
        let workers = HashMap::from([(1, test_worker_config(0, 2))]);
        let mut request = base_request(128);
        request.active_requests.insert(worker0, 9);
        request.active_requests.insert(worker1, 3);
        request.prefill_tokens.insert(worker0, 0);

        let score = score_worker(&workers, &request, worker0, 64, 0.0, 0.0, 0, 1.0, 2.0 / 3.0);

        assert!((score.active_requests - (9.0 / 3.0 + 6.0 * 2.0 / 3.0)).abs() < 1e-9);
        assert!((score.logit - score.active_requests).abs() < 1e-9);
    }

    #[test]
    fn zero_temperature_selection_is_deterministic_without_tree_size_tiebreak() {
        let selector = B10WorkerSelector::new();
        let workers = HashMap::from([
            (30, test_worker_config(0, 1)),
            (10, test_worker_config(0, 1)),
            (20, test_worker_config(0, 1)),
        ]);
        let mut request = base_request(128);
        request.router_config_override = Some(RouterConfigOverride {
            router_temperature: Some(0.0),
            ..Default::default()
        });

        let result = selector
            .select_worker(&workers, &request, request.eligibility(), 64)
            .unwrap();

        assert_eq!(result.worker, WorkerWithDpRank::new(10, 0));
    }

    #[test]
    fn dp_strict_score_restored_for_same_worker_rank_advantage() {
        let selected = WorkerWithDpRank::new(1, 0);
        let scores = [
            (selected, 100.0),
            (WorkerWithDpRank::new(1, 1), 106.0),
            (WorkerWithDpRank::new(2, 0), 10.0),
        ];

        assert!(b10_filter_dp_score(scores, selected));

        let relaxed_scores = [
            (selected, 100.0),
            (WorkerWithDpRank::new(1, 1), 104.0),
            (WorkerWithDpRank::new(2, 0), 10.0),
        ];

        assert!(!b10_filter_dp_score(relaxed_scores, selected));
    }
}
