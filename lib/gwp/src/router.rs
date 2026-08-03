// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The GWP approximate router: a `KvRouter` using etcd discovery and the ZMQ
//! event plane for cross-replica active-sequence synchronization.
//!
//! This is the critical-path reuse from the design doc (Decision 1), resolved
//! more simply than the `MockDiscovery` sketch: `RuntimeConfigWatch` is just a
//! `watch::Receiver<HashMap<WorkerId, ModelRuntimeConfig>>`, so the reflector
//! owns the matching `watch::Sender` and feeds planner worker snapshots
//! directly. The worker feed remains local because every replica reflects the
//! same configured endpoints and planners, while the `DistributedRuntime` uses etcd to
//! discover peer GWP routers and ZMQ to exchange request lifecycle events.
//!
//! Config choices (see `gwp_kv_router_config`):
//! - `use_kv_events: false` — the primary indexer is a local prune-TTL'd radix
//!   tree populated by `record_routing_decision` (the "approximate indexer");
//!   remote endpoints never publish KV events to us.
//! - `skip_initial_worker_wait: false` — REQUIRED for the reflector feed: this
//!   flag doubles as "watch worker configs" in `KvScheduler::start`
//!   (`scheduler.rs:94`); setting it to true would freeze the worker set at
//!   construction time. With `DYN_ROUTER_MIN_INITIAL_WORKERS` unset the
//!   constructor does not block on an empty feed.
//! - `router_snapshot_threshold: None` — snapshots go to the NATS object
//!   store, which GWP does not have.
//! - `router_queue_threshold: None` — GWP never parks requests; downstream
//!   endpoints do their own queueing.
//! - `router_replica_sync: true` — replicas exchange add/prefill/free events
//!   over ZMQ so every selector sees global GWP in-flight load.

use std::collections::HashMap;
use std::sync::Arc;

use dynamo_kv_router::config::{KvRouterConfig, RouterConfigOverride};
use dynamo_kv_router::protocols::{
    RoutingConstraints, TokensWithHashes, WorkerId, WorkerWithDpRank,
};
use dynamo_llm::kv_router::{
    ACTIVE_SEQUENCES_SUBJECT, FindBestMatchOutcome, KvRouter,
    metrics::register_global_metrics_with_component,
};
use dynamo_llm::local_model::runtime_config::ModelRuntimeConfig;
use dynamo_runtime::discovery::{
    DiscoveryInstance, DiscoveryQuery, EventChannelQuery, EventTransportKind,
};
use dynamo_runtime::distributed::{DiscoveryBackend, DistributedConfig, RequestPlaneMode};
use dynamo_runtime::metrics::MetricsHierarchy;
use dynamo_runtime::storage::kv;
use dynamo_runtime::{DistributedRuntime, Runtime};
use tokio::sync::watch;

use crate::{
    config::ModelStagePolicy,
    metrics::GwpMetrics,
    worker_selector::{GwpWorkerSelector, LocalLoadAnchor, SelectionPolicyStore},
};

pub use crate::worker_selector::{RemoteLoadStore, RemoteWorkerLoad};

/// The sender half of the worker feed. Owned by the reflector: every planner
/// poll publishes a full `HashMap<WorkerId, ModelRuntimeConfig>` snapshot of
/// the live planner workers.
pub type WorkerConfigSender = watch::Sender<HashMap<WorkerId, ModelRuntimeConfig>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouteSelection {
    pub worker: WorkerWithDpRank,
    pub cached_tokens: usize,
}

/// Worker type label used for GWP-tier metrics.
const GWP_WORKER_TYPE: &str = "decode";

fn gwp_kv_router_config(ttl_secs: u64) -> KvRouterConfig {
    KvRouterConfig {
        use_kv_events: false,
        skip_initial_worker_wait: false,
        router_snapshot_threshold: None,
        router_queue_threshold: None,
        router_replica_sync: true,
        router_temperature: 0.0,
        router_ttl_secs: ttl_secs as f64,
        ..Default::default()
    }
}

/// GWP replicas discover each other's event channels through etcd and exchange
/// active-sequence events directly over ZMQ. `ETCD_ENDPOINTS` and the standard
/// etcd auth environment variables configure the discovery connection.
fn gwp_distributed_config() -> DistributedConfig {
    DistributedConfig {
        discovery_backend: DiscoveryBackend::KvStore(kv::Selector::Etcd(Box::default())),
        nats_config: None,
        request_plane: RequestPlaneMode::Tcp,
        event_transport_kind: EventTransportKind::Zmq,
    }
}

/// A `KvRouter` plus the pieces GWP needs to keep it alive, exposing only the
/// narrow lifecycle the proxy uses (design doc "Approximate router lifecycle"):
///
/// | proxy event                  | call                        |
/// |------------------------------|-----------------------------|
/// | fall-through decision        | [`GwpRouter::pick`]         |
/// | sticky hit                   | [`GwpRouter::add_request`]  |
/// | upstream response headers    | [`GwpRouter::mark_prefill_completed`] |
/// | stream end / abort / error   | [`GwpRouter::free`]         |
pub struct GwpRouter {
    kv: KvRouter<GwpWorkerSelector>,
    remote_loads: RemoteLoadStore,
    selection_policies: SelectionPolicyStore,
    metrics: GwpMetrics,
    block_size: u32,
    /// Keeps the shared-discovery runtime (and its cancellation token, which
    /// the scheduler/indexer background tasks are children of) alive.
    _drt: DistributedRuntime,
}

impl GwpRouter {
    /// Build the router and the worker feed. Must be called from within a
    /// tokio runtime. The returned [`WorkerConfigSender`] is handed to the
    /// reflector; workers only become routable once a snapshot is sent.
    pub async fn new(
        block_size: u32,
        approx_indexer_ttl_secs: u64,
    ) -> anyhow::Result<(Arc<Self>, WorkerConfigSender)> {
        Self::new_with_distributed_config(
            block_size,
            approx_indexer_ttl_secs,
            gwp_distributed_config(),
        )
        .await
    }

    #[cfg(test)]
    pub(crate) async fn new_process_local(
        block_size: u32,
        approx_indexer_ttl_secs: u64,
    ) -> anyhow::Result<(Arc<Self>, WorkerConfigSender)> {
        Self::new_with_distributed_config(
            block_size,
            approx_indexer_ttl_secs,
            DistributedConfig::process_local(),
        )
        .await
    }

    async fn new_with_distributed_config(
        block_size: u32,
        approx_indexer_ttl_secs: u64,
        distributed_config: DistributedConfig,
    ) -> anyhow::Result<(Arc<Self>, WorkerConfigSender)> {
        let runtime = Runtime::from_current()?;
        let drt = DistributedRuntime::new(runtime, distributed_config).await?;
        let namespace = drt.namespace("gwp")?;
        let component = namespace.component("router")?;
        register_global_metrics_with_component(&component);
        let endpoint = component.endpoint("egress");
        let metrics = GwpMetrics::from_endpoint(&endpoint)?;
        let client = endpoint.client().await?;

        let (tx, rx) = watch::channel(HashMap::new());
        let config = gwp_kv_router_config(approx_indexer_ttl_secs);
        let remote_loads = RemoteLoadStore::default();
        let selection_policies = SelectionPolicyStore::default();
        let selector = GwpWorkerSelector::new(remote_loads.clone(), selection_policies.clone());

        let kv = KvRouter::new(
            endpoint,
            client,
            rx,
            block_size,
            selector,
            Some(config),
            None,
            GWP_WORKER_TYPE,
            None,
            false,
            None,
        )
        .await?;

        Ok((
            Arc::new(Self {
                kv,
                remote_loads,
                selection_policies,
                metrics,
                block_size,
                _drt: drt,
            }),
            tx,
        ))
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    pub fn metrics(&self) -> &GwpMetrics {
        &self.metrics
    }

    pub fn prometheus_metrics(&self) -> anyhow::Result<String> {
        self._drt.metrics().prometheus_expfmt()
    }

    /// Number of other GWP router publishers registered for replica-sync at
    /// this instant. The readiness warm-up uses this startup snapshot.
    pub async fn replica_peer_count(&self) -> anyhow::Result<usize> {
        let own_instance_id = self._drt.connection_id();
        let query = DiscoveryQuery::EventChannels(EventChannelQuery::topic(
            "gwp",
            "router",
            ACTIVE_SEQUENCES_SUBJECT,
        ));
        let peers = self._drt.discovery().list(query).await?;
        Ok(peers
            .into_iter()
            .filter_map(|instance| match instance {
                DiscoveryInstance::EventChannel { instance_id, .. } => Some(instance_id),
                _ => None,
            })
            .filter(|instance_id| *instance_id != own_instance_id)
            .collect::<std::collections::HashSet<_>>()
            .len())
    }

    /// Cancel background router tasks and immediately remove this replica's
    /// discovery registrations instead of waiting for etcd lease expiry.
    pub fn shutdown(&self) {
        self._drt.shutdown();
    }

    /// Replace the planner-sourced baseline used by the selector. For workers
    /// refreshed by this planner result, capture the scheduler's local view at
    /// this exact boundary. Later selections apply the signed local change
    /// after that anchor; publications for other endpoints preserve it.
    pub async fn replace_remote_loads(
        &self,
        loads: HashMap<WorkerId, RemoteWorkerLoad>,
        refreshed_workers: &std::collections::HashSet<WorkerId>,
    ) -> anyhow::Result<()> {
        if refreshed_workers.is_empty() {
            self.remote_loads.replace(loads);
            return Ok(());
        }
        let local = self.kv.get_potential_loads(&[], None, None, None).await?;
        let mut anchors: HashMap<_, _> = refreshed_workers
            .iter()
            .copied()
            .map(|worker_id| (worker_id, LocalLoadAnchor::default()))
            .collect();
        for load in local {
            if let Some(anchor) = anchors.get_mut(&load.worker_id) {
                *anchor = LocalLoadAnchor {
                    prefill_tokens: load.potential_prefill_tokens,
                    decode_blocks: load.potential_decode_blocks,
                    active_requests: load.active_requests,
                };
            }
        }
        self.remote_loads.replace_refreshed(loads, &anchors);
        Ok(())
    }

    pub fn remote_load_store(&self) -> RemoteLoadStore {
        self.remote_loads.clone()
    }

    /// Fall-through decision: score all live workers and provisionally
    /// register the request in the scheduler (`update_states=true` — do NOT
    /// also call [`Self::add_request`], that would double-count). The
    /// approximate index is updated only after the endpoint reports the worker
    /// that actually served the request.
    ///
    /// `rid` is the per-request id (NOT the session id — concurrent requests
    /// in one session must not collide in the slot tracker).
    pub async fn pick(
        &self,
        rid: &str,
        tokens: &[u32],
        allowed_worker_ids: Option<std::collections::HashSet<WorkerId>>,
        policy: ModelStagePolicy,
    ) -> anyhow::Result<RouteSelection> {
        self.book(rid, tokens, allowed_worker_ids, policy).await
    }

    // TODO(gwp-provisional-booking): The per-model `trie` stage controls the
    // approximate cache heuristic only. Active-prefix tracking remains part of
    // the generic scheduler. If GWP later needs hashless provisional scheduler
    // slots, add that behind a reviewed Dynamo API: keep scalar request/prefill
    // accounting, install real hashes after HTTP 200, and make the replica
    // transition ordered or atomic. Do not use random hashes; they substantially
    // over-account shared prefixes at high concurrency.

    /// Book a sticky request on its live affine worker while still querying
    /// the indexer for this turn's approximate cached-token count.
    pub async fn book_pinned(
        &self,
        rid: &str,
        tokens: &[u32],
        worker: WorkerWithDpRank,
        policy: ModelStagePolicy,
    ) -> anyhow::Result<RouteSelection> {
        let cached_tokens = if policy.trie {
            (self
                .kv
                .get_overlap_blocks(tokens, None, worker, None)
                .await? as usize)
                .saturating_mul(self.block_size as usize)
                .min(tokens.len())
        } else {
            0
        };
        self.add_request(rid, tokens, cached_tokens, worker).await;
        Ok(RouteSelection {
            worker,
            cached_tokens,
        })
    }

    async fn book(
        &self,
        rid: &str,
        tokens: &[u32],
        allowed_worker_ids: Option<std::collections::HashSet<WorkerId>>,
        policy: ModelStagePolicy,
    ) -> anyhow::Result<RouteSelection> {
        let _policy_guard = self.selection_policies.install(rid, policy.load_balancing);
        let request_override = RouterConfigOverride {
            overlap_score_credit: (!policy.trie).then_some(0.0),
            ..Default::default()
        };
        let result = self
            .kv
            .find_best_match_details(
                Some(rid),
                tokens,
                None,
                Some(&request_override),
                true, // update_states: registers the scheduler slot under `rid`
                false,
                None,
                0.0,
                0,
                false,
                None,
                None,
                allowed_worker_ids,
                RoutingConstraints::default(),
            )
            .await?;
        match result {
            FindBestMatchOutcome::Routed {
                worker,
                cached_tokens,
                ..
            } => Ok(RouteSelection {
                worker,
                cached_tokens: if policy.trie {
                    cached_tokens.min(tokens.len())
                } else {
                    0
                },
            }),
            FindBestMatchOutcome::Backpressure {
                reason,
                queued_isl_tokens,
                max_queued_isl_tokens,
            } => anyhow::bail!(
                "router backpressure: {reason:?} (queued_isl_tokens={queued_isl_tokens}, max_queued_isl_tokens={max_queued_isl_tokens:?})"
            ),
        }
    }

    async fn add_request(
        &self,
        rid: &str,
        tokens: &[u32],
        cached_tokens: usize,
        worker: WorkerWithDpRank,
    ) {
        self.kv
            .add_request(
                rid.to_string(),
                tokens,
                None,
                cached_tokens.min(tokens.len()),
                None,
                worker,
                None,
                None,
            )
            .await;
    }

    /// Replace a provisional booking with the worker reported by the endpoint.
    ///
    /// Replacement uses a distinct request ID. Replica-sync events are
    /// delivered independently, so a delayed `Free` for the provisional ID
    /// can never erase the `AddRequest` for the confirmed worker.
    pub async fn rebook_request(
        &self,
        provisional_rid: &str,
        confirmed_rid: &str,
        tokens: &[u32],
        cached_tokens: usize,
        worker: WorkerWithDpRank,
    ) {
        self.free(provisional_rid).await;
        self.add_request(confirmed_rid, tokens, cached_tokens, worker)
            .await;
    }

    /// Attribute the request prefix to the worker that actually served it.
    pub async fn record_routing_decision(
        &self,
        rid: &str,
        tokens: &[u32],
        worker: WorkerWithDpRank,
    ) {
        let tokens_with_hashes = TokensWithHashes::new(tokens.to_vec(), self.block_size);
        if let Err(error) = self
            .kv
            .record_routing_decision(tokens_with_hashes, worker)
            .await
        {
            tracing::warn!(rid, %error, "failed to record confirmed routing decision");
        }
    }

    /// Upstream response headers: prefill is done, decode load begins.
    pub async fn mark_prefill_completed(&self, rid: &str) {
        if let Err(error) = self.kv.mark_prefill_completed(rid).await {
            tracing::debug!(rid, %error, "mark_prefill_completed on unknown slot");
        }
    }

    /// Stream end / client abort / upstream error: release the in-flight slot.
    /// Without this the load model monotonically inflates.
    pub async fn free(&self, rid: &str) {
        if let Err(error) = self.kv.free(rid).await {
            tracing::debug!(rid, %error, "free on unknown slot");
        }
    }

    #[cfg(test)]
    pub(crate) async fn potential_loads(
        &self,
        tokens: &[u32],
    ) -> Vec<dynamo_kv_router::scheduling::PotentialLoad> {
        self.kv
            .get_potential_loads(tokens, None, None, None)
            .await
            .expect("query potential loads")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_runtime_uses_etcd_discovery_and_zmq() {
        let config = gwp_distributed_config();
        match config.discovery_backend {
            DiscoveryBackend::KvStore(kv::Selector::Etcd(options)) => {
                assert!(options.attach_lease);
            }
            other => panic!("expected etcd discovery, got {other:?}"),
        }
        assert_eq!(config.event_transport_kind, EventTransportKind::Zmq);
        assert_eq!(config.request_plane, RequestPlaneMode::Tcp);
        assert!(config.nats_config.is_none());
    }

    #[test]
    fn production_router_enables_replica_sync() {
        assert!(gwp_kv_router_config(120).router_replica_sync);
    }

    /// The de-risk test from the design doc: a `KvRouter` constructed with no
    /// etcd and no NATS, fed workers through a bare watch channel, must route,
    /// prefer warm prefixes, and run the full slot lifecycle.
    #[tokio::test]
    async fn kv_router_without_etcd_or_nats() {
        let block_size = 4;
        let (router, tx) = GwpRouter::new_process_local(block_size, 120)
            .await
            .expect("construct GwpRouter");

        // Reflector-style feed: two planner-observed workers appear.
        let mut workers = HashMap::new();
        workers.insert(1u64, ModelRuntimeConfig::default());
        workers.insert(2u64, ModelRuntimeConfig::default());
        tx.send(workers).expect("feed workers");

        // The scheduler ingests the watch asynchronously; give it a beat.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let tokens: Vec<u32> = (0..32).collect();

        // Fall-through: routes to one of the fed workers.
        let first = router
            .pick("rid-1", &tokens, None, ModelStagePolicy::default())
            .await
            .expect("pick");
        assert!(
            first.worker.worker_id == 1 || first.worker.worker_id == 2,
            "routed to unknown worker {}",
            first.worker.worker_id
        );
        router
            .record_routing_decision("rid-1", &tokens, first.worker)
            .await;
        router.mark_prefill_completed("rid-1").await;
        router.free("rid-1").await;

        // Same prefix, no load anywhere: overlap credit must win — the second
        // request lands on the same worker the indexer recorded.
        let second = router
            .pick("rid-2", &tokens, None, ModelStagePolicy::default())
            .await
            .expect("pick warm");
        assert_eq!(
            first.worker.worker_id, second.worker.worker_id,
            "warm prefix did not stick to the recorded worker"
        );
        router.free("rid-2").await;

        // Sticky-path accounting also estimates this turn's cache hit.
        let sticky = router
            .book_pinned("rid-3", &tokens, first.worker, ModelStagePolicy::default())
            .await
            .expect("book sticky worker");
        assert_eq!(sticky.worker, first.worker);
        assert!(sticky.cached_tokens > 0);
        router.mark_prefill_completed("rid-3").await;
        router.free("rid-3").await;
    }

    #[tokio::test]
    async fn trie_disabled_ignores_warm_overlap_but_keeps_scheduler_load_tracking() {
        let (router, tx) = GwpRouter::new_process_local(4, 120)
            .await
            .expect("construct GwpRouter");
        tx.send(HashMap::from([(1u64, ModelRuntimeConfig::default())]))
            .expect("feed worker");
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let tokens: Vec<u32> = (0..32).collect();
        let worker = WorkerWithDpRank::from_worker_id(1);
        router
            .record_routing_decision("warm", &tokens, worker)
            .await;

        let selection = router
            .pick(
                "trie-disabled",
                &tokens,
                None,
                ModelStagePolicy {
                    affinity: true,
                    trie: false,
                    ..Default::default()
                },
            )
            .await
            .expect("route without trie");
        assert_eq!(selection.cached_tokens, 0);

        let load = router
            .potential_loads(&[])
            .await
            .into_iter()
            .find(|load| load.worker_id == worker.worker_id)
            .expect("worker load");
        assert_eq!(load.active_requests, 1);
        assert!(load.potential_decode_blocks > 0);

        router.free("trie-disabled").await;
    }

    /// Workers fed AFTER construction must become routable (the reflector
    /// starts polling only after the router exists) — guards the
    /// `skip_initial_worker_wait=false` requirement.
    #[tokio::test]
    async fn workers_fed_after_construction_are_routable() {
        let (router, tx) = GwpRouter::new_process_local(4, 120)
            .await
            .expect("construct GwpRouter");

        let mut workers = HashMap::new();
        workers.insert(7u64, ModelRuntimeConfig::default());
        tx.send(workers).expect("feed worker");
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let tokens: Vec<u32> = (100..116).collect();
        let picked = router
            .pick("rid-late", &tokens, None, ModelStagePolicy::default())
            .await
            .expect("pick");
        assert_eq!(picked.worker.worker_id, 7);
        router.free("rid-late").await;
    }

    #[test]
    fn unrelated_remote_publication_preserves_worker_anchor() {
        let store = RemoteLoadStore::default();
        let loads = HashMap::from([
            (1, RemoteWorkerLoad::default()),
            (2, RemoteWorkerLoad::default()),
        ]);
        store.replace_refreshed(
            loads.clone(),
            &HashMap::from([
                (
                    1,
                    LocalLoadAnchor {
                        active_requests: 5,
                        ..Default::default()
                    },
                ),
                (
                    2,
                    LocalLoadAnchor {
                        active_requests: 7,
                        ..Default::default()
                    },
                ),
            ]),
        );

        // Only endpoint/worker 2 received a new planner snapshot. Worker 1's
        // anchor must survive instead of being reset by a global generation.
        store.replace_refreshed(
            loads,
            &HashMap::from([(
                2,
                LocalLoadAnchor {
                    active_requests: 9,
                    ..Default::default()
                },
            )]),
        );
        let snapshot = store.snapshot();
        assert_eq!(snapshot.anchors[&1].active_requests, 5);
        assert_eq!(snapshot.anchors[&2].active_requests, 9);
    }
}
