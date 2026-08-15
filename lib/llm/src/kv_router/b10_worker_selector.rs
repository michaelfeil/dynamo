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
    IslStats, KvSchedulerError, RoutingEligibility, SchedulingRequest, WorkerEligibilityError,
};
use dynamo_kv_router::selector::{WorkerSelector, softmax_sample};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
    decode_block_weight: f64,
    active_requests: f64,
    active_request_dp_blend: f64,
    cache_miss_absolute_tokens: usize,
    residency_eviction_cost: f64,
    active_request_isl_penalty: f64,
}

impl B10WorkerSelector {
    pub fn new() -> Self {
        Self {
            last_log_time_ms: AtomicU64::new(0),
        }
    }

    /// Check if we should print this loop iteration (rate-limited based on worker count).
    /// If workers > 5, print at most once every `B10_KV_ROUTER_SELECTION_LOG_INTERVAL_MS`
    /// (default 2000ms), else once every 100ms. The interval is read once at first use.
    fn should_print_this_loop(&self, num_workers: usize) -> bool {
        let log_interval_ms = if num_workers > 5 {
            selection_log_interval_ms()
        } else {
            100
        };

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

/// Throttle interval for the per-worker scoring log when the pool is large
/// (`workers > 5`). Read once from `B10_KV_ROUTER_SELECTION_LOG_INTERVAL_MS`
/// and cached for the process lifetime; defaults to 2000ms.
fn selection_log_interval_ms() -> u64 {
    static INTERVAL: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *INTERVAL.get_or_init(|| {
        std::env::var("B10_KV_ROUTER_SELECTION_LOG_INTERVAL_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&v| v > 0)
            .unwrap_or(2000)
    })
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

fn active_request_isl_penalty(
    request: &SchedulingRequest,
    worker: WorkerWithDpRank,
    mismatch_penalty: f64,
    penalty_ramp: (f64, f64),
) -> f64 {
    let (penalty_start_tokens, penalty_full_tokens) = penalty_ramp;
    if mismatch_penalty <= 0.0
        || !mismatch_penalty.is_finite()
        || penalty_start_tokens < 0.0
        || !penalty_start_tokens.is_finite()
        || penalty_full_tokens <= penalty_start_tokens
        || !penalty_full_tokens.is_finite()
    {
        return 0.0;
    }
    let Some(stats) = request.active_request_isl_stats.as_ref() else {
        return 0.0;
    };

    let ramp_width = penalty_full_tokens - penalty_start_tokens;
    let factor_for = |tokens: f64| ((tokens - penalty_start_tokens) / ramp_width).clamp(0.0, 1.0);

    let penalty_for = |stats: &IslStats| -> f64 {
        if stats.count == 0 || !stats.mean.is_finite() || !stats.stddev.is_finite() {
            return 0.0;
        }

        let incoming_isl = request.isl_tokens as f64;
        let center = stats.mean.max(0.0);
        let distance = (incoming_isl - center).abs();
        let factor = factor_for(distance).max(factor_for(stats.stddev.max(0.0)));

        mismatch_penalty * factor
    };

    let mut penalty: f64 = stats
        .by_worker_id
        .get(&worker.worker_id)
        .map(penalty_for)
        .unwrap_or(0.0);

    if let Some(rank_stats) = stats
        .by_worker_with_dp_rank
        .as_ref()
        .and_then(|stats| stats.get(&worker))
    {
        penalty = penalty.max(penalty_for(rank_stats));
    }

    penalty
}

#[allow(clippy::too_many_arguments)]
fn score_worker<C: WorkerConfigLike>(
    workers: &HashMap<WorkerId, C>,
    request: &SchedulingRequest,
    worker: WorkerWithDpRank,
    block_size: u32,
    overlap_weight: f64,
    decode_block_weight: f64,
    cache_miss_weight: f64,
    cache_miss_min_isl: usize,
    active_request_weight: f64,
    active_request_dp_blend: f64,
    residency_eviction_cost_weight: f64,
    active_request_isl_mismatch_penalty: f64,
    active_request_isl_penalty_ramp: (f64, f64),
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
    let residency_eviction_cost =
        residency_eviction_cost_weight * request.eviction_cost_for(worker);
    let active_request_isl_penalty = active_request_isl_penalty(
        request,
        worker,
        active_request_isl_mismatch_penalty,
        active_request_isl_penalty_ramp,
    );

    let logit = overlap_weight * potential_prefill_block
        + decode_block_weight * decode_block
        + active_request_weight * active_requests
        + cache_miss_weight * (cache_miss_absolute_tokens as f64)
        + residency_eviction_cost
        + active_request_isl_penalty;

    B10Score {
        logit,
        potential_prefill_block,
        decode_block,
        decode_block_weight,
        active_requests,
        active_request_dp_blend,
        cache_miss_absolute_tokens,
        residency_eviction_cost,
        active_request_isl_penalty,
    }
}

impl WorkerSelector<ModelRuntimeConfig> for B10WorkerSelector {
    fn residency_eviction_half_life(&self) -> Option<Duration> {
        let config = b10hotreloadablecm::get_config().get();
        let routing = &config.routing;
        (routing.router_residency_eviction_cost > 0.0)
            .then(|| Duration::from_secs_f64(routing.router_residency_half_life))
    }

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

        let decode_block_weight = hot_reloadable_config.routing.router_decode_block_weight;
        let cache_miss_weight = hot_reloadable_config.routing.router_cache_miss_weight;
        let cache_miss_min_isl = hot_reloadable_config.routing.router_cache_miss_min_isl;
        let session_affinity_score_multiplier = hot_reloadable_config
            .routing
            .router_session_affinity_score_multiplier;
        let active_request_weight = hot_reloadable_config.routing.router_active_request_weight;
        let active_request_dp_blend = hot_reloadable_config.routing.router_active_request_dp_blend;
        let residency_eviction_cost_weight =
            hot_reloadable_config.routing.router_residency_eviction_cost;
        let active_request_isl_mismatch_penalty = hot_reloadable_config
            .routing
            .router_active_request_isl_mismatch_penalty;
        let active_request_isl_penalty_ramp = hot_reloadable_config
            .routing
            .router_active_request_isl_penalty_ramp;
        let temperature = b10hotreloadablecm::sanitize_router_temperature(
            request
                .router_config_override
                .as_ref()
                .and_then(|cfg| cfg.router_temperature)
                .unwrap_or(hot_reloadable_config.routing.router_temperature),
        );

        let score_worker = |worker: WorkerWithDpRank| -> B10Score {
            score_worker(
                workers,
                request,
                worker,
                block_size,
                overlap_weight,
                decode_block_weight,
                cache_miss_weight,
                cache_miss_min_isl,
                active_request_weight,
                active_request_dp_blend,
                residency_eviction_cost_weight,
                active_request_isl_mismatch_penalty,
                active_request_isl_penalty_ramp,
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
                    "worker_id={} dp={:?} logit={:.3} | ow={:.2}*ppf={:.2} + dbw={:.2}*db={:.2} + arw={:.2}*ar={:.2}(dpb={:.2}) + cmw={:.2}*cm={} + rec={:.3} + islp={:.3}",
                    worker.worker_id,
                    worker.dp_rank,
                    score.logit,
                    overlap_weight,
                    score.potential_prefill_block,
                    score.decode_block_weight,
                    score.decode_block,
                    active_request_weight,
                    score.active_requests,
                    score.active_request_dp_blend,
                    cache_miss_weight,
                    score.cache_miss_absolute_tokens,
                    score.residency_eviction_cost,
                    score.active_request_isl_penalty
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
        eligibility.for_each_eligible_worker_rank(workers, |worker, config| {
            let score = score_worker(worker);
            let preferred_taint_multiplier = request
                .routing_constraints
                .preferred_taint_multiplier(config.taints())
                .unwrap_or(1.0);
            let session_affinity_multiplier = if request.preferred_worker == Some(worker) {
                session_affinity_score_multiplier
            } else {
                1.0
            };
            let weighted_logit =
                (score.logit + 1.0) * preferred_taint_multiplier * session_affinity_multiplier;
            if verbose {
                tracing::info!(
                    "worker_id={} dp={:?} logit={:.3} (base={:.3} * ptm={:.3} * affinity={:.3}) | ow={:.2}*ppf={:.2} + dbw={:.2}*db={:.2} + arw={:.2}*ar={:.2}(dpb={:.2}) + cmw={:.2}*cm={} + rec={:.3} + islp={:.3}",
                    worker.worker_id,
                    worker.dp_rank,
                    weighted_logit,
                    score.logit,
                    preferred_taint_multiplier,
                    session_affinity_multiplier,
                    overlap_weight,
                    score.potential_prefill_block,
                    score.decode_block_weight,
                    score.decode_block,
                    active_request_weight,
                    score.active_requests,
                    score.active_request_dp_blend,
                    cache_miss_weight,
                    score.cache_miss_absolute_tokens,
                    score.residency_eviction_cost,
                    score.active_request_isl_penalty
                );
            }
            worker_logits.insert(worker, weighted_logit);
        });

        if worker_logits.is_empty() {
            return Err(KvSchedulerError::NoEndpoints);
        }

        let (best_worker, best_logit) = softmax_sample(&worker_logits, temperature);
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
            active_request_isl_stats: None,
            eviction_costs: HashMap::new(),
            track_prefill_tokens: true,
            router_config_override: None,
            update_states: false,
            lora_name: None,
            priority_jump: 0.0,
            priority_load_shed_percent: 0,
            do_not_queue: false,
            expected_output_tokens: None,
            pinned_worker: None,
            preferred_worker: None,
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

        let score = score_worker(
            &workers,
            &request,
            worker,
            64,
            0.0,
            1.0,
            1.0,
            0,
            0.0,
            2.0 / 3.0,
            0.0,
            0.0,
            (2048.0, 32_768.0),
        );

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

        let score = score_worker(
            &workers,
            &request,
            worker0,
            64,
            0.0,
            1.0,
            0.0,
            0,
            1.0,
            2.0 / 3.0,
            0.0,
            0.0,
            (2048.0, 32_768.0),
        );

        assert!((score.active_requests - (9.0 / 3.0 + 6.0 * 2.0 / 3.0)).abs() < 1e-9);
        assert!((score.logit - score.active_requests).abs() < 1e-9);
    }

    #[test]
    fn decode_block_weight_scales_decode_term_independently_of_prefill() {
        let worker = WorkerWithDpRank::new(1, 0);
        let workers = HashMap::from([(worker.worker_id, test_worker_config(0, 1))]);
        let mut request = base_request(256);
        // Force a known decode-block count of 4 (256 ISL / 64 block_size, no
        // cached tokens, no prefill_tokens entry so prefill term is 0).
        request.decode_blocks.insert(worker, 4);
        request.prefill_tokens.insert(worker, 0);

        let score_unit = score_worker(
            &workers,
            &request,
            worker,
            64,
            0.0,
            1.0,
            0.0,
            0,
            0.0,
            2.0 / 3.0,
            0.0,
            0.0,
            (2048.0, 32_768.0),
        );
        let score_double = score_worker(
            &workers,
            &request,
            worker,
            64,
            0.0,
            2.0,
            0.0,
            0,
            0.0,
            2.0 / 3.0,
            0.0,
            0.0,
            (2048.0, 32_768.0),
        );

        assert_eq!(score_unit.decode_block, 4.0);
        assert_eq!(score_unit.logit, 4.0);
        assert_eq!(score_double.decode_block_weight, 2.0);
        assert_eq!(score_double.logit, 8.0);
        // Prefill term (overlap_weight * potential_prefill_block) stays 0 in
        // both, proving the decode weight is applied independently.
        assert_eq!(score_unit.potential_prefill_block, 0.0);
    }

    #[test]
    fn decode_block_weight_zero_suppresses_decode_term() {
        let worker = WorkerWithDpRank::new(1, 0);
        let workers = HashMap::from([(worker.worker_id, test_worker_config(0, 1))]);
        let mut request = base_request(256);
        request.decode_blocks.insert(worker, 4);
        request.prefill_tokens.insert(worker, 0);

        let score = score_worker(
            &workers,
            &request,
            worker,
            64,
            1.0,
            0.0,
            0.0,
            0,
            0.0,
            2.0 / 3.0,
            0.0,
            0.0,
            (2048.0, 32_768.0),
        );

        assert_eq!(score.decode_block, 4.0);
        assert_eq!(score.logit, 0.0);
    }

    fn tainted_worker_config(taints: &[&str]) -> ModelRuntimeConfig {
        ModelRuntimeConfig {
            taints: taints.iter().map(|taint| taint.to_string()).collect(),
            ..test_worker_config(0, 1)
        }
    }

    #[test]
    fn preferred_taints_bias_b10_toward_matching_worker() {
        let selector = B10WorkerSelector::new();
        let workers = HashMap::from([
            (10, tainted_worker_config(&["b10_worker_pool=default"])),
            (20, tainted_worker_config(&["b10_worker_pool=fast"])),
        ]);
        let mut request = base_request(128);
        request.router_config_override = Some(RouterConfigOverride {
            prefill_load_scale: Some(1.0),
            router_temperature: Some(0.0),
            ..Default::default()
        });
        request.routing_constraints.preferred_taints =
            HashMap::from([("b10_worker_pool=fast".to_string(), 0.85)]);

        let result = selector
            .select_worker(&workers, &request, request.eligibility(), 64)
            .unwrap();

        assert_eq!(result.worker, WorkerWithDpRank::new(20, 0));
    }

    #[test]
    fn negative_preferred_taints_bias_b10_away_from_matching_worker() {
        let selector = B10WorkerSelector::new();
        let workers = HashMap::from([
            (10, tainted_worker_config(&["b10_worker_pool=fast"])),
            (20, tainted_worker_config(&["b10_worker_pool=default"])),
        ]);
        let mut request = base_request(128);
        request.router_config_override = Some(RouterConfigOverride {
            prefill_load_scale: Some(1.0),
            router_temperature: Some(0.0),
            ..Default::default()
        });
        request.routing_constraints.preferred_taints =
            HashMap::from([("b10_worker_pool=fast".to_string(), -0.85)]);

        let result = selector
            .select_worker(&workers, &request, request.eligibility(), 64)
            .unwrap();

        assert_eq!(result.worker, WorkerWithDpRank::new(20, 0));
    }

    #[test]
    fn session_affinity_halves_score_for_exact_worker_and_dp_rank() {
        let selector = B10WorkerSelector::new();
        let worker_rank_0 = WorkerWithDpRank::new(10, 0);
        let worker_rank_1 = WorkerWithDpRank::new(10, 1);
        let workers = HashMap::from([(10, test_worker_config(0, 2))]);
        let mut request = base_request(128);
        request.router_config_override = Some(RouterConfigOverride {
            prefill_load_scale: Some(1.0),
            router_temperature: Some(0.0),
            ..Default::default()
        });
        request.prefill_tokens.insert(worker_rank_0, 640);
        request.prefill_tokens.insert(worker_rank_1, 384);
        request.preferred_worker = Some(worker_rank_0);

        let result = selector
            .select_worker(&workers, &request, request.eligibility(), 64)
            .unwrap();

        assert_eq!(result.worker, worker_rank_0);
    }

    #[test]
    fn zero_temperature_is_floored_and_does_not_tiebreak_by_worker_id() {
        let selector = B10WorkerSelector::new();
        // All workers identical -> all logits tie. A temperature of 0 is now
        // floored to 1e-12, so ties go through softmax_sample (uniform random)
        // instead of the old lowest-worker_id tiebreak.
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

        let mut seen = std::collections::HashSet::new();
        for _ in 0..100 {
            let result = selector
                .select_worker(&workers, &request, request.eligibility(), 64)
                .unwrap();
            seen.insert(result.worker);
        }
        // With 3 tied workers over 100 trials, uniform-random tiebreaking must
        // produce more than one distinct winner. This guards against regressing
        // back to a deterministic lowest-worker_id (u64) tiebreak.
        assert!(
            seen.len() > 1,
            "temperature=0 should not deterministically pick the lowest worker_id; saw {seen:?}"
        );
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
