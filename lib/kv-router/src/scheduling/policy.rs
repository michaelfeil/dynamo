// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::time::Duration;

use super::config::RouterQueuePolicy;
use super::types::SchedulingContext;
use crate::protocols::WorkerConfigLike;
use ordered_float::OrderedFloat;
/// Pluggable scheduling policy that determines queue ordering.
/// Monomorphized for zero-cost inlining on the hot comparison path.
///
/// Higher key = higher priority (natural max-heap ordering).
pub trait SchedulingPolicy: Send + Sync + 'static {
    /// Priority key stored in each queue entry.
    type Key: Ord + Eq + Clone + Send + 'static;

    /// Compute priority key at enqueue time.
    fn enqueue_key<C: WorkerConfigLike>(
        &self,
        arrival_offset: Duration,
        ctx: SchedulingContext<'_, C>,
    ) -> Self::Key;

    /// Recompute priority key during update(). Default: return old key unchanged.
    fn rekey<C: WorkerConfigLike>(
        &self,
        _now: Duration,
        old_key: &Self::Key,
        _ctx: SchedulingContext<'_, C>,
    ) -> Self::Key {
        old_key.clone()
    }

    /// When true, queue rebuilds heap via rekey() on each update() call.
    /// When false (default), rekey path is compiled out entirely.
    const DYNAMIC: bool = false;

    /// Runtime equivalent of [`Self::DYNAMIC`] for enum-dispatched policies.
    fn is_dynamic(&self) -> bool {
        Self::DYNAMIC
    }
}

/// FCFS with priority bumps: key = priority_jump - arrival_offset.
/// Earlier arrival or higher priority_jump produces a higher key, scheduled first.
///
/// Optimizes for tail TTFT — no request waits longer than necessary,
/// since ordering is purely by (adjusted) arrival time.
pub struct FcfsPolicy;

impl SchedulingPolicy for FcfsPolicy {
    type Key = OrderedFloat<f64>;

    fn enqueue_key<C: WorkerConfigLike>(
        &self,
        arrival_offset: Duration,
        ctx: SchedulingContext<'_, C>,
    ) -> Self::Key {
        OrderedFloat(ctx.request().priority_jump.max(0.0) - arrival_offset.as_secs_f64())
    }
}

/// LCFS with priority bumps: key = priority_jump + arrival_offset.
/// Later arrival or higher priority_jump produces a higher key, scheduled first.
///
/// This intentionally favors newer arrivals under saturation and is mainly useful
/// for policy comparison experiments.
pub struct LcfsPolicy;

impl SchedulingPolicy for LcfsPolicy {
    type Key = OrderedFloat<f64>;

    fn enqueue_key<C: WorkerConfigLike>(
        &self,
        arrival_offset: Duration,
        ctx: SchedulingContext<'_, C>,
    ) -> Self::Key {
        OrderedFloat(ctx.request().priority_jump.max(0.0) + arrival_offset.as_secs_f64())
    }
}

/// Weighted Shortest Processing Time (Smith's rule):
/// key = (1 + priority_jump) / new_tokens, where new_tokens estimates the
/// actual prefill cost by subtracting the effective KV cache overlap from ISL.
/// Unpinned requests use the best available overlap. Pinned requests use only
/// the overlap for their exact target worker so queue ordering matches routing.
///
/// Optimizes for average TTFT — minimizes total weighted completion time
/// (Smith 1956). Short or high-priority requests are scheduled before
/// long low-priority ones, reducing mean latency across the batch.
pub struct WsptPolicy;

impl SchedulingPolicy for WsptPolicy {
    type Key = OrderedFloat<f64>;

    fn enqueue_key<C: WorkerConfigLike>(
        &self,
        _arrival_offset: Duration,
        ctx: SchedulingContext<'_, C>,
    ) -> Self::Key {
        let weight = 1.0 + ctx.request().priority_jump.max(0.0);
        let new_tokens = ctx.best_effective_prefill_tokens().max(1);
        OrderedFloat(weight / new_tokens as f64)
    }
}

const B10_FAIR_WSPT_BLEND_SECS: f64 = 15.0;
const B10_FAIR_WSPT_CREDIT_SCALE_SECS: f64 = 15.0;
const B10_FAIR_WSPT_REFERENCE_TOKENS: f64 = 1024.0;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct B10FairWsptKey {
    score: OrderedFloat<f64>,
    arrival_offset: Duration,
}

impl Ord for B10FairWsptKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .cmp(&other.score)
            // Stable FCFS tie-break: earlier arrival wins.
            .then_with(|| other.arrival_offset.cmp(&self.arrival_offset))
    }
}

impl PartialOrd for B10FairWsptKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// B10 fair WSPT hybrid:
/// FCFS with a bounded WSPT credit that linearly fades to zero over 15s.
pub struct B10FairWsptPolicy;

impl B10FairWsptPolicy {
    fn key<C: WorkerConfigLike>(
        now: Duration,
        arrival_offset: Duration,
        ctx: SchedulingContext<'_, C>,
    ) -> B10FairWsptKey {
        let age_secs = now.saturating_sub(arrival_offset).as_secs_f64().max(0.0);
        let wspt_weight = (1.0 - age_secs / B10_FAIR_WSPT_BLEND_SECS).clamp(0.0, 1.0);
        let priority_jump = ctx.request().priority_jump.max(0.0);
        let effective_new_tokens = ctx.best_effective_prefill_tokens().max(1) as f64;
        let wspt_credit = B10_FAIR_WSPT_CREDIT_SCALE_SECS * B10_FAIR_WSPT_REFERENCE_TOKENS
            / (B10_FAIR_WSPT_REFERENCE_TOKENS + effective_new_tokens);
        let score = priority_jump - arrival_offset.as_secs_f64() + wspt_weight * wspt_credit;

        B10FairWsptKey {
            score: OrderedFloat(score),
            arrival_offset,
        }
    }
}

impl SchedulingPolicy for B10FairWsptPolicy {
    type Key = B10FairWsptKey;

    const DYNAMIC: bool = true;

    fn enqueue_key<C: WorkerConfigLike>(
        &self,
        arrival_offset: Duration,
        ctx: SchedulingContext<'_, C>,
    ) -> Self::Key {
        Self::key(arrival_offset, arrival_offset, ctx)
    }

    fn rekey<C: WorkerConfigLike>(
        &self,
        now: Duration,
        old_key: &Self::Key,
        ctx: SchedulingContext<'_, C>,
    ) -> Self::Key {
        Self::key(now, old_key.arrival_offset, ctx)
    }
}

/// Runtime-dispatched scheduling policy selected via configuration.
/// Delegates to the concrete policy variant; the branch is fully predictable
/// since the variant is fixed at queue construction time.
pub enum RouterSchedulingPolicy {
    Fcfs(FcfsPolicy),
    Lcfs(LcfsPolicy),
    Wspt(WsptPolicy),
    B10FairWspt(B10FairWsptPolicy),
}

impl RouterSchedulingPolicy {
    pub fn new(kind: RouterQueuePolicy) -> Self {
        match kind {
            RouterQueuePolicy::Fcfs => Self::Fcfs(FcfsPolicy),
            RouterQueuePolicy::Lcfs => Self::Lcfs(LcfsPolicy),
            RouterQueuePolicy::Wspt => Self::Wspt(WsptPolicy),
            RouterQueuePolicy::B10FairWspt => Self::B10FairWspt(B10FairWsptPolicy),
        }
    }
}

impl SchedulingPolicy for RouterSchedulingPolicy {
    type Key = RouterSchedulingKey;

    const DYNAMIC: bool = true;

    fn is_dynamic(&self) -> bool {
        matches!(self, Self::B10FairWspt(_))
    }

    fn enqueue_key<C: WorkerConfigLike>(
        &self,
        arrival_offset: Duration,
        ctx: SchedulingContext<'_, C>,
    ) -> Self::Key {
        match self {
            Self::Fcfs(p) => RouterSchedulingKey::Scalar(p.enqueue_key(arrival_offset, ctx)),
            Self::Lcfs(p) => RouterSchedulingKey::Scalar(p.enqueue_key(arrival_offset, ctx)),
            Self::Wspt(p) => RouterSchedulingKey::Scalar(p.enqueue_key(arrival_offset, ctx)),
            Self::B10FairWspt(p) => {
                RouterSchedulingKey::B10FairWspt(p.enqueue_key(arrival_offset, ctx))
            }
        }
    }

    fn rekey<C: WorkerConfigLike>(
        &self,
        now: Duration,
        old_key: &Self::Key,
        ctx: SchedulingContext<'_, C>,
    ) -> Self::Key {
        match (self, old_key) {
            (Self::B10FairWspt(p), RouterSchedulingKey::B10FairWspt(key)) => {
                RouterSchedulingKey::B10FairWspt(p.rekey(now, key, ctx))
            }
            _ => old_key.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RouterSchedulingKey {
    Scalar(OrderedFloat<f64>),
    B10FairWspt(B10FairWsptKey),
}

impl Ord for RouterSchedulingKey {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Scalar(a), Self::Scalar(b)) => a.cmp(b),
            (Self::B10FairWspt(a), Self::B10FairWspt(b)) => a.cmp(b),
            (Self::Scalar(_), Self::B10FairWspt(_)) => Ordering::Less,
            (Self::B10FairWspt(_), Self::Scalar(_)) => Ordering::Greater,
        }
    }
}

impl PartialOrd for RouterSchedulingKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rustc_hash::FxHashMap;

    use super::*;
    use crate::SchedulingRequest;
    use crate::protocols::{OverlapScores, WorkerWithDpRank};
    use crate::test_utils::SimpleWorkerConfig;

    fn workers_for_request(request: &SchedulingRequest) -> HashMap<u64, SimpleWorkerConfig> {
        let mut workers = HashMap::new();

        for worker in request.effective_cached_tokens.keys() {
            workers.entry(worker.worker_id).or_default();
        }

        if let Some(worker) = request.pinned_worker {
            workers.entry(worker.worker_id).or_default();
        }

        if let Some(allowed_worker_ids) = request.allowed_worker_ids.as_ref() {
            for &worker_id in allowed_worker_ids {
                workers.entry(worker_id).or_default();
            }
        }

        workers
    }

    fn enqueue_key<P: SchedulingPolicy>(
        policy: &P,
        arrival_offset: Duration,
        request: &SchedulingRequest,
    ) -> P::Key {
        let workers = workers_for_request(request);
        policy.enqueue_key(arrival_offset, SchedulingContext::new(request, &workers))
    }

    fn request_with(
        isl_tokens: usize,
        priority_jump: f64,
        overlaps: OverlapScores,
    ) -> SchedulingRequest {
        let effective_overlap_blocks = overlaps
            .scores
            .iter()
            .map(|(worker, overlap)| (*worker, *overlap as f64))
            .collect();
        let effective_cached_tokens = overlaps
            .scores
            .iter()
            .map(|(worker, overlap)| (*worker, *overlap as usize * 16))
            .collect();
        SchedulingRequest {
            maybe_request_id: None,
            token_seq: None,
            isl_tokens,
            tier_overlap_blocks: Default::default(),
            effective_overlap_blocks,
            effective_cached_tokens,
            decode_blocks: FxHashMap::default(),
            prefill_tokens: FxHashMap::default(),
            active_requests: HashMap::new(),
            track_prefill_tokens: true,
            router_config_override: None,
            update_states: false,
            lora_name: None,
            priority_jump,
            priority_load_shed_percent: 0,
            do_not_queue: false,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: None,
        }
    }

    fn overlaps_from(scores: &[(u64, u32)]) -> OverlapScores {
        let mut map = FxHashMap::default();
        for &(worker_id, score) in scores {
            map.insert(WorkerWithDpRank::new(worker_id, 0), score);
        }
        OverlapScores {
            scores: map,
            frequencies: vec![],
        }
    }

    // ---- FCFS policy tests ----

    #[test]
    fn fcfs_earlier_arrival_scheduled_first() {
        let policy = FcfsPolicy;
        let req = request_with(512, 0.0, OverlapScores::default());
        let early = enqueue_key(&policy, Duration::from_secs(1), &req);
        let late = enqueue_key(&policy, Duration::from_secs(10), &req);
        assert!(early > late, "earlier arrival should have higher key");
    }

    #[test]
    fn fcfs_priority_jump_promotes() {
        let policy = FcfsPolicy;
        // Both arrive at the same wall-clock offset (10s), but one has priority.
        let normal = request_with(512, 0.0, OverlapScores::default());
        let boosted = request_with(512, 100.0, OverlapScores::default());
        let t = Duration::from_secs(10);
        let key_normal = enqueue_key(&policy, t, &normal);
        let key_boosted = enqueue_key(&policy, t, &boosted);
        assert!(
            key_boosted > key_normal,
            "priority_jump should produce a higher key"
        );
    }

    #[test]
    fn fcfs_priority_jump_beats_earlier_arrival() {
        let policy = FcfsPolicy;
        // Request A arrived at t=0 with no priority.
        // Request B arrived at t=5 with priority_jump=50s.
        // B should be scheduled first despite arriving later.
        let a = request_with(512, 0.0, OverlapScores::default());
        let b = request_with(512, 50.0, OverlapScores::default());
        let key_a = enqueue_key(&policy, Duration::from_secs(0), &a);
        let key_b = enqueue_key(&policy, Duration::from_secs(5), &b);
        assert!(key_b > key_a);
    }

    #[test]
    fn lcfs_later_arrival_scheduled_first() {
        let policy = LcfsPolicy;
        let req = request_with(512, 0.0, OverlapScores::default());
        let early = enqueue_key(&policy, Duration::from_secs(1), &req);
        let late = enqueue_key(&policy, Duration::from_secs(10), &req);
        assert!(late > early, "later arrival should have higher key");
    }

    #[test]
    fn lcfs_priority_jump_promotes() {
        let policy = LcfsPolicy;
        let normal = request_with(512, 0.0, OverlapScores::default());
        let boosted = request_with(512, 100.0, OverlapScores::default());
        let t = Duration::from_secs(10);
        let key_normal = enqueue_key(&policy, t, &normal);
        let key_boosted = enqueue_key(&policy, t, &boosted);
        assert!(
            key_boosted > key_normal,
            "priority_jump should produce a higher key"
        );
    }

    #[test]
    fn router_scheduling_policy_matches_fcfs_and_lcfs_ordering() {
        let req = request_with(512, 0.0, OverlapScores::default());
        let early = Duration::from_secs(1);
        let late = Duration::from_secs(10);

        let fcfs = RouterSchedulingPolicy::new(RouterQueuePolicy::Fcfs);
        assert!(enqueue_key(&fcfs, early, &req) > enqueue_key(&fcfs, late, &req));

        let lcfs = RouterSchedulingPolicy::new(RouterQueuePolicy::Lcfs);
        assert!(enqueue_key(&lcfs, late, &req) > enqueue_key(&lcfs, early, &req));
    }

    #[test]
    fn router_scheduling_policy_dynamic_only_for_b10_fair_wspt() {
        assert!(!RouterSchedulingPolicy::new(RouterQueuePolicy::Fcfs).is_dynamic());
        assert!(!RouterSchedulingPolicy::new(RouterQueuePolicy::Lcfs).is_dynamic());
        assert!(!RouterSchedulingPolicy::new(RouterQueuePolicy::Wspt).is_dynamic());
        assert!(RouterSchedulingPolicy::new(RouterQueuePolicy::B10FairWspt).is_dynamic());
    }

    // ---- B10 fair WSPT policy tests ----

    #[test]
    fn b10_fair_wspt_shorter_request_gets_early_credit() {
        let policy = B10FairWsptPolicy;
        let short = request_with(100, 0.0, OverlapScores::default());
        let long = request_with(10_000, 0.0, OverlapScores::default());

        assert!(
            enqueue_key(&policy, Duration::ZERO, &short)
                > enqueue_key(&policy, Duration::ZERO, &long),
            "fresh queued requests should be WSPT-like"
        );
    }

    #[test]
    fn b10_fair_wspt_short_newer_request_can_beat_early_longer_request() {
        let policy = B10FairWsptPolicy;
        let old_long = request_with(10_000, 0.0, OverlapScores::default());
        let newer_short = request_with(100, 0.0, OverlapScores::default());

        assert!(
            enqueue_key(&policy, Duration::from_secs(5), &newer_short)
                > enqueue_key(&policy, Duration::ZERO, &old_long),
            "bounded WSPT credit should allow early short-request promotion"
        );
    }

    #[test]
    fn b10_fair_wspt_ages_into_fcfs_after_15_seconds() {
        let policy = B10FairWsptPolicy;
        let old_long = request_with(10_000, 0.0, OverlapScores::default());
        let newer_short = request_with(100, 0.0, OverlapScores::default());

        let old_key = enqueue_key(&policy, Duration::ZERO, &old_long);
        let newer_key = enqueue_key(&policy, Duration::from_secs(5), &newer_short);
        assert!(newer_key > old_key);

        let old_workers = workers_for_request(&old_long);
        let newer_workers = workers_for_request(&newer_short);
        let aged_old_key = policy.rekey(
            Duration::from_secs(15),
            &old_key,
            SchedulingContext::new(&old_long, &old_workers),
        );
        let aged_newer_key = policy.rekey(
            Duration::from_secs(15),
            &newer_key,
            SchedulingContext::new(&newer_short, &newer_workers),
        );

        assert!(
            aged_old_key > aged_newer_key,
            "a 15s-old request should fall back to FCFS ahead of newer requests"
        );
    }

    #[test]
    fn b10_fair_wspt_priority_jump_promotes() {
        let policy = B10FairWsptPolicy;
        let normal = request_with(512, 0.0, OverlapScores::default());
        let boosted = request_with(512, 5.0, OverlapScores::default());

        assert!(
            enqueue_key(&policy, Duration::from_secs(3), &boosted)
                > enqueue_key(&policy, Duration::from_secs(3), &normal),
            "priority_jump should remain additive"
        );
    }

    #[test]
    fn b10_fair_wspt_overlap_increases_early_credit() {
        let policy = B10FairWsptPolicy;
        let no_cache = request_with(1024, 0.0, OverlapScores::default());
        let cached = request_with(1024, 0.0, overlaps_from(&[(0, 60)]));

        assert!(
            enqueue_key(&policy, Duration::ZERO, &cached)
                > enqueue_key(&policy, Duration::ZERO, &no_cache),
            "effective cache overlap should increase early WSPT credit"
        );
    }

    // ---- WSPT policy tests ----

    #[test]
    fn wspt_shorter_request_scheduled_first() {
        let policy = WsptPolicy;
        let short = request_with(100, 0.0, OverlapScores::default());
        let long = request_with(1000, 0.0, OverlapScores::default());
        let t = Duration::ZERO;
        assert!(
            enqueue_key(&policy, t, &short) > enqueue_key(&policy, t, &long),
            "shorter request should have higher key"
        );
    }

    #[test]
    fn wspt_overlap_reduces_effective_cost() {
        let policy = WsptPolicy;
        // Both 1024 ISL tokens, but one has 60 blocks cached (960 tokens).
        let no_cache = request_with(1024, 0.0, OverlapScores::default());
        let cached = request_with(1024, 0.0, overlaps_from(&[(0, 60)]));
        let t = Duration::ZERO;
        let key_no_cache = enqueue_key(&policy, t, &no_cache);
        let key_cached = enqueue_key(&policy, t, &cached);
        assert!(
            key_cached > key_no_cache,
            "request with overlap should have higher key (fewer new tokens)"
        );
    }

    #[test]
    fn wspt_overlap_applies_when_prefill_tracking_disabled() {
        let policy = WsptPolicy;
        let mut req = request_with(1024, 0.0, overlaps_from(&[(0, 60)]));
        req.track_prefill_tokens = false;

        let key = enqueue_key(&policy, Duration::ZERO, &req);
        let expected = OrderedFloat(1.0 / 64.0);
        assert_eq!(key, expected);
    }

    #[test]
    fn wspt_priority_promotes() {
        let policy = WsptPolicy;
        let normal = request_with(512, 0.0, OverlapScores::default());
        let boosted = request_with(512, 5.0, OverlapScores::default());
        let t = Duration::ZERO;
        assert!(
            enqueue_key(&policy, t, &boosted) > enqueue_key(&policy, t, &normal),
            "priority_jump should increase key"
        );
    }

    #[test]
    fn wspt_uses_max_overlap() {
        let policy = WsptPolicy;
        // 4 workers with overlaps [10, 20, 50, 60]. max = 60.
        // new_tokens = 1024 - 60*16 = 64
        let req = request_with(
            1024,
            0.0,
            overlaps_from(&[(0, 10), (1, 20), (2, 50), (3, 60)]),
        );
        let key = enqueue_key(&policy, Duration::ZERO, &req);
        let expected = OrderedFloat(1.0 / 64.0);
        assert_eq!(key, expected);
    }

    #[test]
    fn wspt_uses_pinned_worker_overlap_when_present() {
        let policy = WsptPolicy;
        let mut req = request_with(1024, 0.0, overlaps_from(&[(0, 60), (1, 1)]));
        req.pinned_worker = Some(WorkerWithDpRank::new(1, 0));

        let key = enqueue_key(&policy, Duration::ZERO, &req);
        let expected = OrderedFloat(1.0 / 1008.0);
        assert_eq!(key, expected);
    }

    #[test]
    fn wspt_missing_pinned_overlap_uses_zero() {
        let policy = WsptPolicy;
        let mut req = request_with(1024, 0.0, overlaps_from(&[(0, 60)]));
        req.pinned_worker = Some(WorkerWithDpRank::new(1, 0));

        let key = enqueue_key(&policy, Duration::ZERO, &req);
        let expected = OrderedFloat(1.0 / 1024.0);
        assert_eq!(key, expected);
    }

    #[test]
    fn wspt_no_overlap_falls_back_to_isl() {
        let policy = WsptPolicy;
        let req = request_with(512, 0.0, OverlapScores::default());
        let key = enqueue_key(&policy, Duration::ZERO, &req);
        let expected = OrderedFloat(1.0 / 512.0);
        assert_eq!(key, expected);
    }

    #[test]
    fn wspt_full_overlap_clamps_to_one() {
        let policy = WsptPolicy;
        // 512 tokens, 64 blocks cached = 1024 cached tokens > ISL → saturating_sub → 0 → max(1)
        let req = request_with(512, 0.0, overlaps_from(&[(0, 64)]));
        let key = enqueue_key(&policy, Duration::ZERO, &req);
        let expected = OrderedFloat(1.0 / 1.0);
        assert_eq!(key, expected);
    }

    #[test]
    fn wspt_required_taints_ignore_incompatible_overlap() {
        let policy = WsptPolicy;
        let mut req = request_with(1024, 0.0, overlaps_from(&[(0, 60), (1, 1)]));
        req.routing_constraints.required_taints =
            std::collections::HashSet::from(["mdc-b".to_string()]);

        let mut workers = workers_for_request(&req);
        workers.get_mut(&0).unwrap().taints =
            std::collections::HashSet::from(["mdc-a".to_string()]);
        workers.get_mut(&1).unwrap().taints =
            std::collections::HashSet::from(["mdc-b".to_string()]);

        let key = policy.enqueue_key(Duration::ZERO, SchedulingContext::new(&req, &workers));
        let expected = OrderedFloat(1.0 / 1008.0);
        assert_eq!(key, expected);
    }
}
