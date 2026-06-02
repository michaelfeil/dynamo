// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! B10 Worker Selector with hot-reloadable configuration
//!
//! Adapts upstream's `DefaultWorkerSelector` cost function with
//! absolute cache-miss weighting and short-request bypass, sourcing
//! tuning knobs from the hot-reloadable config manager.
//!
//! DP-rank-aware load balancing has been intentionally dropped: upstream's
//! selector handles DP-rank fan-out via `WorkerConfigLike::data_parallel_size`,
//! so this selector simply iterates DP ranks like upstream does and picks the
//! lowest-logit worker. No softmax / no active-request blending across DP ranks.

use crate::kv_router::b10hotreloadablecm;
use crate::local_model::runtime_config::ModelRuntimeConfig;
use dynamo_kv_router::protocols::{WorkerId, WorkerSelectionResult};
use dynamo_kv_router::scheduling::{
    KvSchedulerError, RoutingEligibility, SchedulingRequest, WorkerEligibilityError,
};
use dynamo_kv_router::selector::WorkerSelector;
use std::collections::HashMap;

/// B10 Worker Selector that uses hot-reloadable configuration.
#[derive(Debug, Default)]
pub struct B10WorkerSelector;

impl B10WorkerSelector {
    pub fn new() -> Self {
        Self
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
        let isl = request.isl_tokens;

        let hot_reloadable_config = b10hotreloadablecm::get_config().get();

        let overlap_weight = request
            .router_config_override
            .as_ref()
            .and_then(|cfg| cfg.prefill_load_scale)
            .unwrap_or(hot_reloadable_config.routing.router_overlap_score_weight);

        let cache_miss_weight = hot_reloadable_config.routing.router_cache_miss_weight;
        let cache_miss_min_isl = hot_reloadable_config.routing.router_cache_miss_min_isl;

        let score_worker = |worker: dynamo_kv_router::protocols::WorkerWithDpRank| -> f64 {
            let effective_overlap_blocks = request.effective_overlap_blocks_for(worker);
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
            let cache_miss_absolute_tokens: usize = if isl <= cache_miss_min_isl {
                isl
            } else {
                let overlap_tokens =
                    (effective_overlap_blocks.max(0.0) * (block_size as f64)) as usize;
                isl.saturating_sub(overlap_tokens)
            };

            overlap_weight * potential_prefill_block
                + decode_block
                + cache_miss_weight * (cache_miss_absolute_tokens as f64)
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

            let logit = score_worker(worker);
            let effective_overlap_blocks = request.effective_overlap_blocks_for(worker);
            let cached_tokens = request.effective_cached_tokens_for(worker);

            tracing::info!(
                "B10WorkerSelector selected pinned worker: worker_id={} dp_rank={:?}, logit: {:.3}, effective cached blocks: {:.2}",
                worker.worker_id,
                worker.dp_rank,
                logit,
                effective_overlap_blocks,
            );

            return Ok(WorkerSelectionResult {
                worker,
                required_blocks: request_blocks,
                effective_overlap_blocks,
                cached_tokens,
            });
        }

        let mut best_worker = None;
        let mut best_logit = f64::INFINITY;
        eligibility.for_each_eligible_worker_rank(workers, |worker, _| {
            let score = score_worker(worker);
            if score < best_logit {
                best_logit = score;
                best_worker = Some(worker);
            }
        });

        let best_worker = best_worker.ok_or(KvSchedulerError::NoEndpoints)?;
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
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_b10_worker_selector_creation() {
        let _selector = B10WorkerSelector::new();
    }
}
