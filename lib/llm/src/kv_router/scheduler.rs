// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::protocols::{LocalBlockHash, RouterBackpressureReason, SharedCacheHits};
pub use dynamo_kv_router::scheduling::overlap_refresh::{
    NoopOverlapScoresRefresh, OverlapScoresRefresh, RefreshedOverlap,
};
pub use dynamo_kv_router::scheduling::policy::RouterSchedulingPolicy;
pub use dynamo_kv_router::scheduling::{
    KvSchedulerError, LocalScheduler, OverloadedWorkerProvider, PotentialLoad, SchedulingRequest,
    SchedulingResponse, TierOverlapBlocks,
};
pub use dynamo_kv_router::selector::DefaultWorkerSelector;
use dynamo_kv_router::selector::WorkerSelector as WorkerSelectorTrait;

use super::b10hotreloadablecm;
use super::metrics::ROUTER_QUEUE_METRICS;
use super::sequence::{
    RuntimeSequencePublisher, SequenceError, SequenceRequest, create_multi_worker_sequences,
};
use crate::discovery::RuntimeConfigWatch;
use crate::local_model::runtime_config::ModelRuntimeConfig;
use anyhow::Result;
use dynamo_kv_router::{
    PrefillLoadEstimator,
    config::{KvRouterConfig, RouterConfigOverride},
    protocols::{RoutingConstraints, WorkerId, WorkerWithDpRank},
};
use dynamo_runtime::component::Component;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_tokens::SequenceHash;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

/// Publish the growth of the queue's cancelled-booking count since the last sync
/// as counter increments.
fn b10_sync_cancelled_requests(worker_type: &str, current: usize, last: &mut usize) {
    let delta = current.saturating_sub(*last);
    *last = current;
    ROUTER_QUEUE_METRICS.b10_inc_cancelled_requests(worker_type, delta as u64);
}

pub struct KvScheduler<Sel = DefaultWorkerSelector, RF = NoopOverlapScoresRefresh>
where
    Sel: WorkerSelectorTrait<ModelRuntimeConfig>,
    RF: OverlapScoresRefresh,
{
    inner: Arc<
        LocalScheduler<
            RuntimeSequencePublisher,
            ModelRuntimeConfig,
            RouterSchedulingPolicy,
            Sel,
            RF,
        >,
    >,
}

impl<Sel, RF> KvScheduler<Sel, RF>
where
    Sel: WorkerSelectorTrait<ModelRuntimeConfig> + Send + Sync + 'static,
    RF: OverlapScoresRefresh + Send + Sync + 'static,
{
    /// Start the scheduler, optionally wiring an [`OverlapScoresRefresh`] into the queue so
    /// long-waiting requests can be re-scored at dequeue time.
    #[expect(clippy::too_many_arguments)]
    pub async fn start(
        component: Component,
        block_size: u32,
        workers_with_configs: RuntimeConfigWatch,
        selector: Sel,
        kv_router_config: &KvRouterConfig,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        overlap_scores_refresh: Option<Arc<RF>>,
        overloaded_worker_provider: Option<OverloadedWorkerProvider>,
        worker_type: &'static str,
    ) -> Result<Self, KvSchedulerError> {
        let initial_workers: HashMap<WorkerId, ModelRuntimeConfig> =
            workers_with_configs.borrow().clone();

        let router_id = component.drt().discovery().instance_id();
        let slots = create_multi_worker_sequences(
            component.clone(),
            block_size as usize,
            initial_workers,
            kv_router_config.router_replica_sync,
            router_id,
            worker_type,
        )
        .await
        .map_err(|e| KvSchedulerError::InitFailed(e.to_string()))?;

        let watch_worker_configs = !kv_router_config.skip_initial_worker_wait;
        if !watch_worker_configs {
            tracing::info!("skipping discovery-based worker monitoring");
        }

        let policy = RouterSchedulingPolicy::new(kv_router_config.router_queue_policy);
        tracing::info!(
            "Router queue policy: {}",
            kv_router_config.router_queue_policy
        );

        let inner = Arc::new(LocalScheduler::new_with_overlap_refresh(
            slots,
            workers_with_configs.clone(),
            kv_router_config.router_queue_threshold,
            kv_router_config
                .router_queue_by_incoming_missing_isl
                .clone(),
            block_size,
            selector,
            policy,
            prefill_load_estimator,
            overlap_scores_refresh,
            overloaded_worker_provider,
            kv_router_config.router_track_residency,
            kv_router_config.router_track_active_request_isl,
            kv_router_config.router_queue_recheck_interval(),
            kv_router_config.router_track_prefill_tokens,
            component.drt().child_token(),
            worker_type,
            watch_worker_configs,
        ));

        let metrics_scheduler = Arc::clone(&inner);
        let metrics_cancel_token = component.drt().child_token();
        let mut queue_updates = inner.subscribe_queue_updates();
        tokio::spawn(async move {
            let mut recheck_interval = tokio::time::interval(Duration::from_secs(60));
            let mut hot_reload_interval = tokio::time::interval(Duration::from_secs(10));
            hot_reload_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            // Starting from 0 credits cancellations that happened between
            // queue start and this task's first sync.
            let mut last_cancelled_requests = 0;
            let sync_scheduler = Arc::clone(&metrics_scheduler);
            let mut sync_queue_metrics = move || {
                ROUTER_QUEUE_METRICS.set_pending(worker_type, sync_scheduler.pending_count());
                ROUTER_QUEUE_METRICS
                    .set_pending_isl_tokens(worker_type, sync_scheduler.pending_isl_tokens());
                b10_sync_cancelled_requests(
                    worker_type,
                    sync_scheduler.b10_cancelled_requests_count(),
                    &mut last_cancelled_requests,
                );
                use std::sync::atomic::Ordering::Relaxed;
                let eval = sync_scheduler.b10_eval_gauges();
                ROUTER_QUEUE_METRICS.b10_set_gate_evaluation(
                    worker_type,
                    "prefill_busy",
                    eval.prefill_threshold_tokens.load(Relaxed),
                    eval.prefill_evaluated_tokens.load(Relaxed),
                );
                ROUTER_QUEUE_METRICS.b10_set_gate_evaluation(
                    worker_type,
                    "decode_tokens",
                    dynamo_kv_router::scheduling::queue::router_queue_threshold_decode_tokens(),
                    eval.decode_evaluated_tokens.load(Relaxed),
                );
                let tier_evaluated: Vec<u64> = eval
                    .isl_evaluated_tokens_per_tier
                    .iter()
                    .map(|slot| slot.load(Relaxed))
                    .collect();
                ROUTER_QUEUE_METRICS.b10_set_isl_tokens_tiers(
                    worker_type,
                    &sync_scheduler.b10_isl_tier_caps(),
                    &tier_evaluated,
                );
            };
            sync_queue_metrics();

            loop {
                tokio::select! {
                    _ = metrics_cancel_token.cancelled() => break,
                    result = queue_updates.changed() => {
                        if result.is_err() {
                            break;
                        }
                        sync_queue_metrics();
                    }
                    _ = recheck_interval.tick() => {
                        sync_queue_metrics();
                    }
                    _ = hot_reload_interval.tick() => {
                        if let Some(v) = b10hotreloadablecm::get_router_queue_threshold() {
                            let threshold = if v > 0.0 { Some(v) } else { None };
                            metrics_scheduler.update_router_queue_threshold(threshold).await;
                        }
                        metrics_scheduler.reconfigure_residency_capacities();
                        // Sync every tick, not only on threshold changes: the
                        // gate/eval gauges are exported only by this closure,
                        // and the 60s recheck alone leaves them a minute
                        // stale on dashboards.
                        sync_queue_metrics();
                    }
                }
            }
        });

        Ok(Self { inner })
    }

    #[expect(clippy::too_many_arguments)]
    pub async fn schedule(
        &self,
        maybe_request_id: Option<String>,
        isl_tokens: usize,
        token_seq: Option<Vec<SequenceHash>>,
        tier_overlap_blocks: TierOverlapBlocks,
        effective_overlap_blocks: HashMap<dynamo_kv_router::protocols::WorkerWithDpRank, f64>,
        effective_cached_tokens: HashMap<dynamo_kv_router::protocols::WorkerWithDpRank, usize>,
        router_config_override: Option<&RouterConfigOverride>,
        update_states: bool,
        lora_name: Option<String>,
        priority_jump: f64,
        priority_load_shed_percent: u8,
        do_not_queue: bool,
        expected_output_tokens: Option<u32>,
        pinned_worker: Option<WorkerWithDpRank>,
        preferred_worker: Option<WorkerWithDpRank>,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
        routing_constraints: RoutingConstraints,
        shared_cache_hits: Option<SharedCacheHits>,
    ) -> Result<SchedulingResponse, KvSchedulerError> {
        self.schedule_with_block_hashes(
            maybe_request_id,
            isl_tokens,
            token_seq,
            None,
            tier_overlap_blocks,
            effective_overlap_blocks,
            effective_cached_tokens,
            router_config_override,
            update_states,
            lora_name,
            priority_jump,
            priority_load_shed_percent,
            do_not_queue,
            expected_output_tokens,
            pinned_worker,
            preferred_worker,
            allowed_worker_ids,
            routing_constraints,
            shared_cache_hits,
        )
        .await
    }

    /// Like [`schedule`](Self::schedule) but forwards the block hashes used to compute the
    /// initial overlap scores. Required to enable dequeue-time overlap refresh; ignored if
    /// the scheduler was not constructed with an [`OverlapScoresRefresh`].
    #[expect(clippy::too_many_arguments)]
    pub async fn schedule_with_block_hashes(
        &self,
        maybe_request_id: Option<String>,
        isl_tokens: usize,
        token_seq: Option<Vec<SequenceHash>>,
        block_hashes: Option<Vec<LocalBlockHash>>,
        tier_overlap_blocks: TierOverlapBlocks,
        effective_overlap_blocks: HashMap<dynamo_kv_router::protocols::WorkerWithDpRank, f64>,
        effective_cached_tokens: HashMap<dynamo_kv_router::protocols::WorkerWithDpRank, usize>,
        router_config_override: Option<&RouterConfigOverride>,
        update_states: bool,
        lora_name: Option<String>,
        priority_jump: f64,
        priority_load_shed_percent: u8,
        do_not_queue: bool,
        expected_output_tokens: Option<u32>,
        pinned_worker: Option<WorkerWithDpRank>,
        preferred_worker: Option<WorkerWithDpRank>,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
        routing_constraints: RoutingConstraints,
        shared_cache_hits: Option<SharedCacheHits>,
    ) -> Result<SchedulingResponse, KvSchedulerError> {
        let response = self
            .inner
            .schedule_with_block_hashes(
                maybe_request_id,
                isl_tokens,
                token_seq,
                block_hashes,
                tier_overlap_blocks,
                effective_overlap_blocks,
                effective_cached_tokens,
                router_config_override,
                update_states,
                lora_name,
                priority_jump,
                priority_load_shed_percent,
                do_not_queue,
                expected_output_tokens,
                pinned_worker,
                preferred_worker,
                allowed_worker_ids,
                routing_constraints,
                shared_cache_hits,
            )
            .await;
        if let Err(KvSchedulerError::Backpressure { reason, .. }) = &response {
            ROUTER_QUEUE_METRICS
                .inc_backpressure(self.worker_type(), router_backpressure_reason_label(reason));
        }
        ROUTER_QUEUE_METRICS.set_pending(self.worker_type(), self.pending_count());
        ROUTER_QUEUE_METRICS.set_pending_isl_tokens(self.worker_type(), self.pending_isl_tokens());
        response
    }

    pub fn register_workers(&self, worker_ids: &HashSet<WorkerId>) {
        self.inner.register_workers(worker_ids);
    }

    pub async fn add_request(&self, req: SequenceRequest) -> Result<(), SequenceError> {
        self.inner.add_request(req).await
    }

    pub async fn mark_prefill_completed(&self, request_id: &str) -> Result<(), SequenceError> {
        self.inner.mark_prefill_completed(request_id).await?;
        ROUTER_QUEUE_METRICS.set_pending(self.worker_type(), self.pending_count());
        ROUTER_QUEUE_METRICS.set_pending_isl_tokens(self.worker_type(), self.pending_isl_tokens());
        Ok(())
    }

    pub async fn free(&self, request_id: &str) -> Result<(), SequenceError> {
        self.inner.free(request_id).await?;
        ROUTER_QUEUE_METRICS.set_pending(self.worker_type(), self.pending_count());
        ROUTER_QUEUE_METRICS.set_pending_isl_tokens(self.worker_type(), self.pending_isl_tokens());
        Ok(())
    }

    pub fn pending_count(&self) -> usize {
        self.inner.pending_count()
    }

    pub fn pending_isl_tokens(&self) -> usize {
        self.inner.pending_isl_tokens()
    }

    pub fn worker_type(&self) -> &'static str {
        self.inner.worker_type()
    }

    pub fn add_output_block(
        &self,
        request_id: &str,
        decay_fraction: Option<f64>,
    ) -> Result<(), SequenceError> {
        self.inner.add_output_block(request_id, decay_fraction)
    }

    pub fn get_potential_loads(
        &self,
        token_seq: Option<Vec<SequenceHash>>,
        isl_tokens: usize,
        effective_cached_tokens: HashMap<dynamo_kv_router::protocols::WorkerWithDpRank, usize>,
        track_prefill_tokens: bool,
    ) -> Vec<PotentialLoad> {
        self.inner.get_potential_loads(
            token_seq,
            isl_tokens,
            effective_cached_tokens,
            track_prefill_tokens,
        )
    }

    pub fn get_active_lora_counts(&self) -> HashMap<String, usize> {
        self.inner.get_active_lora_counts()
    }

    pub fn supports_overlap_refresh(&self) -> bool {
        self.inner.supports_overlap_refresh()
    }
}

fn router_backpressure_reason_label(reason: &RouterBackpressureReason) -> &'static str {
    match reason {
        RouterBackpressureReason::MaxQueuedIslTokensExceeded => "max_queued_isl_tokens_exceeded",
        RouterBackpressureReason::DoNotQueue => "do_not_queue",
    }
}
