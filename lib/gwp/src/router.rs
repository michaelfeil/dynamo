// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The GWP approximate router: a `KvRouter` using etcd discovery and a
//! configurable event plane for cross-replica active-sequence synchronization.
//!
//! This is the critical-path reuse from the design doc (Decision 1), resolved
//! more simply than the `MockDiscovery` sketch: `RuntimeConfigWatch` is just a
//! `watch::Receiver<HashMap<WorkerId, ModelRuntimeConfig>>`, so the topology
//! registry owns one sender and one `KvRouter` per canonical model and feeds
//! each only its eligible workers. The feeds remain local because every
//! replica observes the same topology, while one shared `DistributedRuntime`
//! uses etcd to discover peer model routers and ZMQ or NATS to exchange
//! model-scoped request lifecycle events.
//!
//! Config choices (see `gwp_kv_router_config`):
//! - `use_kv_events: false` — the primary indexer is a local prune-TTL'd radix
//!   tree populated by `record_routing_decision` (the "approximate indexer");
//!   remote endpoints never publish KV events to us.
//! - `skip_initial_worker_wait: false` — REQUIRED for the topology feed: this
//!   flag doubles as "watch worker configs" in `KvScheduler::start`
//!   (`scheduler.rs:94`); setting it to true would freeze the worker set at
//!   construction time. With `DYN_ROUTER_MIN_INITIAL_WORKERS` unset the
//!   constructor does not block on an empty feed.
//! - `router_snapshot_threshold: None` — snapshots go to the NATS object
//!   store, which GWP does not have.
//! - `router_queue_threshold: None` — GWP never parks requests; downstream
//!   endpoints do their own queueing.
//! - `router_replica_sync: true` — replicas exchange add/prefill/free events
//!   over the configured event plane so every selector sees global GWP
//!   in-flight load.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use dashmap::DashMap;
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
use dynamo_runtime::slug::Slug;
use dynamo_runtime::storage::kv;
use dynamo_runtime::{DistributedRuntime, Runtime};
use serde::{Deserialize, Serialize};
use tokio::sync::watch;

use crate::{
    config::{EndpointId, ModelStagePolicy},
    metrics::GwpMetrics,
    scoring::LocalLoadAnchor,
    worker_selector::{GwpWorkerSelector, SelectionPolicyStore},
};

pub use crate::scoring::{ObservedLoadStore, ObservedWorkerLoad};

/// The sender half of the worker feed. Owned by the topology controller, which
/// publishes a full `HashMap<WorkerId, ModelRuntimeConfig>` generation.
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

/// GWP keeps etcd as its discovery plane while allowing replica events to use
/// either direct ZMQ (the backwards-compatible default) or NATS Core pub-sub.
/// `DYN_EVENT_PLANE=nats` selects NATS and `NATS_SERVER` configures its server.
fn gwp_distributed_config() -> anyhow::Result<DistributedConfig> {
    let event_transport_kind = match std::env::var(
        dynamo_runtime::config::environment_names::event_plane::DYN_EVENT_PLANE,
    ) {
        Ok(value) if value == "nats" => EventTransportKind::Nats,
        Ok(value) if value == "zmq" || value.is_empty() => EventTransportKind::Zmq,
        Err(std::env::VarError::NotPresent) => EventTransportKind::Zmq,
        Ok(value) => anyhow::bail!(
            "invalid DYN_EVENT_PLANE value '{value}'; valid values are 'nats' and 'zmq'"
        ),
        Err(std::env::VarError::NotUnicode(_)) => {
            anyhow::bail!("DYN_EVENT_PLANE must contain valid Unicode")
        }
    };

    Ok(gwp_distributed_config_for(event_transport_kind))
}

fn gwp_distributed_config_for(event_transport_kind: EventTransportKind) -> DistributedConfig {
    DistributedConfig {
        discovery_backend: DiscoveryBackend::KvStore(kv::Selector::Etcd(Box::default())),
        nats_config: (event_transport_kind == EventTransportKind::Nats).then(Default::default),
        request_plane: RequestPlaneMode::Tcp,
        event_transport_kind,
    }
}

/// Canonical model identity used to select one isolated scheduler and event
/// channel. Request aliases resolve to this ID before entering the registry.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct OracleVersionId(String);

impl OracleVersionId {
    pub fn new(value: impl Into<String>) -> anyhow::Result<Self> {
        let value = value.into();
        anyhow::ensure!(
            !value.trim().is_empty(),
            "oracle version ID must not be empty"
        );
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn transport_key(&self) -> String {
        if Slug::try_from(self.as_str()).is_ok() {
            self.0.clone()
        } else {
            Slug::slugify_unique(self.as_str()).to_string()
        }
    }
}

impl std::fmt::Display for OracleVersionId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Process-wide registry of independent schedulers, keyed by canonical model.
/// Aliases are resolved by the topology before this boundary.
pub struct GwpRouterRegistry {
    routers: DashMap<OracleVersionId, Arc<GwpRouter>>,
    /// Removed routers are quiesced and retained for safe reuse. `KvRouter`
    /// cancellation is tied to the shared runtime, so dropping one model
    /// router would also cancel its siblings until the underlying router
    /// supports scoped shutdown.
    dormant: DashMap<OracleVersionId, Arc<GwpRouter>>,
    create_lock: tokio::sync::Mutex<()>,
    drt: DistributedRuntime,
    metrics: GwpMetrics,
    block_size: u32,
    approx_indexer_ttl_secs: u64,
}

impl GwpRouterRegistry {
    pub async fn new(block_size: u32, approx_indexer_ttl_secs: u64) -> anyhow::Result<Arc<Self>> {
        Self::new_with_distributed_config(
            block_size,
            approx_indexer_ttl_secs,
            gwp_distributed_config()?,
        )
        .await
    }

    #[cfg(test)]
    pub(crate) async fn new_process_local(
        block_size: u32,
        approx_indexer_ttl_secs: u64,
    ) -> anyhow::Result<Arc<Self>> {
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
    ) -> anyhow::Result<Arc<Self>> {
        let runtime = Runtime::from_current()?;
        let drt = DistributedRuntime::new(runtime, distributed_config).await?;
        let namespace = drt.namespace("gwp")?;
        let metrics_component = namespace.component("router")?;
        register_global_metrics_with_component(&metrics_component);
        let metrics = GwpMetrics::from_endpoint(&metrics_component.endpoint("egress"))?;
        let registry = Arc::new(Self {
            routers: DashMap::new(),
            dormant: DashMap::new(),
            create_lock: tokio::sync::Mutex::new(()),
            drt,
            metrics,
            block_size,
            approx_indexer_ttl_secs,
        });
        Ok(registry)
    }

    async fn ensure_model(
        &self,
        oracle_version_id: &OracleVersionId,
    ) -> anyhow::Result<Arc<GwpRouter>> {
        if let Some(registered) = self.routers.get(oracle_version_id) {
            return Ok(registered.clone());
        }
        let _guard = self.create_lock.lock().await;
        if let Some(registered) = self.routers.get(oracle_version_id) {
            return Ok(registered.clone());
        }
        if let Some((_, router)) = self.dormant.remove(oracle_version_id) {
            tracing::info!(%oracle_version_id, "reactivated dormant oracle-version router");
            self.routers
                .insert(oracle_version_id.clone(), router.clone());
            return Ok(router);
        }
        let router = GwpRouter::new_for_model(
            self.drt.clone(),
            oracle_version_id.clone(),
            self.block_size,
            self.approx_indexer_ttl_secs,
        )
        .await?;
        tracing::info!(
            %oracle_version_id,
            event_component = %oracle_version_id.transport_key(),
            "created oracle-version router"
        );
        self.routers
            .insert(oracle_version_id.clone(), router.clone());
        Ok(router)
    }

    pub fn model_router(&self, oracle_version_id: &OracleVersionId) -> Option<Arc<GwpRouter>> {
        self.routers
            .get(oracle_version_id)
            .map(|router| router.clone())
    }

    /// Create and feed every scheduler in a complete topology generation.
    /// Removal happens only after the topology snapshot becomes visible so a
    /// request can never observe an old route without its router.
    pub async fn reconcile_topology(
        &self,
        topology: &crate::topology::TopologySnapshot,
        refreshed_workers: &HashSet<WorkerId>,
    ) -> anyhow::Result<()> {
        for model in topology.models.keys() {
            self.ensure_model(&OracleVersionId::new(model.clone())?)
                .await?;
        }

        for model in topology.models.keys() {
            let model = OracleVersionId::new(model.clone())?;
            let router = self
                .routers
                .get(&model)
                .expect("model router created during reconciliation")
                .clone();
            let eligible_endpoints = topology
                .models
                .get(model.as_str())
                .map(|binding| &binding.endpoints)
                .expect("topology model exists");
            let workers: HashMap<_, _> = topology
                .workers
                .iter()
                .filter(|(_, worker)| eligible_endpoints.contains(&worker.endpoint))
                .map(|(worker_id, worker)| (*worker_id, worker.runtime.clone()))
                .collect();
            let loads: HashMap<_, _> = topology
                .workers
                .iter()
                .filter(|(_, worker)| eligible_endpoints.contains(&worker.endpoint))
                .filter_map(|(worker_id, worker)| {
                    worker.observed_load.map(|load| (*worker_id, load))
                })
                .collect();
            let refreshed: HashSet<_> = refreshed_workers
                .iter()
                .copied()
                .filter(|worker_id| workers.contains_key(worker_id))
                .collect();
            router.replace_observed_loads(loads, &refreshed).await?;
            router
                .workers_tx
                .send(workers)
                .map_err(|_| anyhow::anyhow!("worker feed closed for oracle version {model}"))?;
            let mut per_endpoint: HashMap<EndpointId, usize> = eligible_endpoints
                .iter()
                .cloned()
                .map(|endpoint| (endpoint, 0))
                .collect();
            for worker in topology
                .workers
                .values()
                .filter(|worker| eligible_endpoints.contains(&worker.endpoint))
            {
                *per_endpoint.entry(worker.endpoint.clone()).or_default() += 1;
            }
            router.metrics.replace_scheduler_live_workers(per_endpoint);
        }
        Ok(())
    }

    /// Stop admitting removed models after their topology disappears. The
    /// router is quiesced and moved to a dormant cache; in-flight
    /// lifecycle calls retain their `Arc` and can still release local state.
    pub fn retire_absent(&self, topology: &crate::topology::TopologySnapshot) {
        let active: HashSet<_> = topology
            .models
            .keys()
            .filter_map(|model| OracleVersionId::new(model.clone()).ok())
            .collect();
        let removed: Vec<_> = self
            .routers
            .iter()
            .filter(|router| !active.contains(router.key()))
            .map(|router| router.key().clone())
            .collect();
        for oracle_version_id in removed {
            if let Some((_, router)) = self.routers.remove(&oracle_version_id) {
                let _ = router.workers_tx.send(HashMap::new());
                router.observed_loads.replace(HashMap::new());
                router
                    .metrics
                    .replace_scheduler_live_workers(HashMap::new());
                tracing::info!(%oracle_version_id, "moved oracle-version router to dormant cache");
                self.dormant.insert(oracle_version_id, router);
            }
        }
    }

    pub fn metrics(&self) -> &GwpMetrics {
        &self.metrics
    }

    #[cfg(feature = "server")]
    pub(crate) fn register_info_route(&self, topology: crate::topology::TopologyStore) {
        let callback: dynamo_runtime::engine_routes::EngineRouteCallback = Arc::new(move |_| {
            let topology = topology.clone();
            Box::pin(async move { Ok(serde_json::to_value(topology.load().info())?) })
        });
        self.drt.engine_routes().register("info", callback);
    }

    pub fn prometheus_metrics(&self) -> anyhow::Result<String> {
        self.drt.metrics().prometheus_expfmt()
    }

    pub async fn replica_peer_count(&self) -> anyhow::Result<usize> {
        let own_instance_id = self.drt.connection_id();
        let mut peers = HashSet::new();
        let routers: Vec<_> = self.routers.iter().map(|router| router.clone()).collect();
        for router in routers {
            peers.extend(router.replica_peer_ids().await?);
        }
        peers.remove(&own_instance_id);
        Ok(peers.len())
    }

    pub fn shutdown(&self) {
        self.drt.shutdown();
    }

    #[cfg(test)]
    pub(crate) async fn potential_loads(
        &self,
        tokens: &[u32],
    ) -> Vec<dynamo_kv_router::scheduling::PotentialLoad> {
        assert_eq!(self.routers.len(), 1, "test helper requires one model");
        let model = self
            .routers
            .iter()
            .next()
            .expect("model router")
            .key()
            .clone();
        self.potential_loads_for_model(&model, tokens).await
    }

    #[cfg(test)]
    pub(crate) async fn potential_loads_for_model(
        &self,
        oracle_version_id: &OracleVersionId,
        tokens: &[u32],
    ) -> Vec<dynamo_kv_router::scheduling::PotentialLoad> {
        self.model_router(oracle_version_id)
            .expect("model router")
            .potential_loads(tokens)
            .await
    }
}

/// A `KvRouter` plus the pieces GWP needs to keep it alive, exposing only the
/// narrow lifecycle the proxy uses (design doc "Approximate router lifecycle"):
///
/// | proxy event                  | call                        |
/// |------------------------------|-----------------------------|
/// | fall-through decision        | [`GwpRouter::pick`]         |
/// | sticky hit                   | `GwpRouter::add_request`    |
/// | upstream response headers    | [`GwpRouter::mark_prefill_completed`] |
/// | stream end / abort / error   | [`GwpRouter::free`]         |
pub struct GwpRouter {
    kv: KvRouter<GwpWorkerSelector>,
    observed_loads: ObservedLoadStore,
    selection_policies: SelectionPolicyStore,
    block_size: u32,
    component_name: String,
    workers_tx: WorkerConfigSender,
    metrics: GwpMetrics,
    /// Keeps the shared-discovery runtime (and its cancellation token, which
    /// the scheduler/indexer background tasks are children of) alive.
    _drt: DistributedRuntime,
}

impl GwpRouter {
    async fn new_for_model(
        drt: DistributedRuntime,
        oracle_version_id: OracleVersionId,
        block_size: u32,
        approx_indexer_ttl_secs: u64,
    ) -> anyhow::Result<Arc<Self>> {
        let namespace = drt.namespace("gwp")?;
        let component_name = oracle_version_id.transport_key();
        let component = namespace.component(&component_name)?;
        let endpoint = component.endpoint("egress");
        let metrics = GwpMetrics::from_endpoint(&endpoint)?;
        let client = endpoint.client().await?;

        let (tx, rx) = watch::channel(HashMap::new());
        let config = gwp_kv_router_config(approx_indexer_ttl_secs);
        let observed_loads = ObservedLoadStore::default();
        let selection_policies = SelectionPolicyStore::default();
        let selector = GwpWorkerSelector::new(observed_loads.clone(), selection_policies.clone());

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
            None,
        )
        .await?;

        Ok(Arc::new(Self {
            kv,
            observed_loads,
            selection_policies,
            block_size,
            component_name,
            workers_tx: tx,
            metrics,
            _drt: drt,
        }))
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    pub fn metrics(&self) -> &GwpMetrics {
        &self.metrics
    }

    async fn replica_peer_ids(&self) -> anyhow::Result<HashSet<u64>> {
        let query = DiscoveryQuery::EventChannels(EventChannelQuery::topic(
            "gwp",
            self.component_name.clone(),
            ACTIVE_SEQUENCES_SUBJECT,
        ));
        let peers = self._drt.discovery().list(query).await?;
        Ok(peers
            .into_iter()
            .filter_map(|instance| match instance {
                DiscoveryInstance::EventChannel { instance_id, .. } => Some(instance_id),
                _ => None,
            })
            .collect())
    }

    /// Replace the provider-observed baseline used by the selector. For workers
    /// refreshed by this update, capture the scheduler's local view at
    /// this exact boundary. Later selections apply the signed local change
    /// after that anchor; publications for other endpoints preserve it.
    pub async fn replace_observed_loads(
        &self,
        loads: HashMap<WorkerId, ObservedWorkerLoad>,
        refreshed_workers: &std::collections::HashSet<WorkerId>,
    ) -> anyhow::Result<()> {
        if refreshed_workers.is_empty() {
            self.observed_loads.replace(loads);
            return Ok(());
        }
        // Discounted, deliberately: these anchors are subtracted from
        // placement-path locals (which apply the discounts) in fuse_load.
        let local = self
            .kv
            .get_potential_loads(&[], None, None, None, true)
            .await?;
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
        self.observed_loads.replace_refreshed(loads, &anchors);
        Ok(())
    }

    /// Fall-through decision: score all live workers and provisionally
    /// register the request in the scheduler (`update_states=true` — do NOT
    /// also call `Self::add_request`, that would double-count). The
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
            .get_potential_loads(tokens, None, None, None, true)
            .await
            .expect("query potential loads")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelTokenizationConfig;
    use crate::topology::{ModelBinding, ProfileId, ResolvedModelProfile, TopologySnapshot};

    fn topology(models: &[&str]) -> TopologySnapshot {
        let mut snapshot = TopologySnapshot::default();
        for model in models {
            let profile = ProfileId((*model).to_string());
            snapshot.models.insert(
                (*model).to_string(),
                ModelBinding {
                    endpoints: HashSet::new(),
                    profile: profile.clone(),
                },
            );
            snapshot.profiles.insert(
                profile,
                ResolvedModelProfile {
                    stages: ModelStagePolicy::default(),
                    tokenization: ModelTokenizationConfig::Pseudo,
                },
            );
        }
        snapshot
    }

    async fn process_local_model_router(
        block_size: u32,
    ) -> (Arc<GwpRouterRegistry>, Arc<GwpRouter>, WorkerConfigSender) {
        let registry = GwpRouterRegistry::new_process_local(block_size, 120)
            .await
            .expect("construct model router registry");
        let model = OracleVersionId::new("model").expect("valid oracle version ID");
        let router = registry.ensure_model(&model).await.expect("model router");
        (registry, router.clone(), router.workers_tx.clone())
    }

    #[test]
    fn production_runtime_uses_etcd_discovery_and_selected_event_plane() {
        let config = gwp_distributed_config_for(EventTransportKind::Zmq);
        match config.discovery_backend {
            DiscoveryBackend::KvStore(kv::Selector::Etcd(options)) => {
                assert!(options.attach_lease);
            }
            other => panic!("expected etcd discovery, got {other:?}"),
        }
        assert_eq!(config.event_transport_kind, EventTransportKind::Zmq);
        assert_eq!(config.request_plane, RequestPlaneMode::Tcp);
        assert!(config.nats_config.is_none());

        let config = gwp_distributed_config_for(EventTransportKind::Nats);
        assert_eq!(config.event_transport_kind, EventTransportKind::Nats);
        assert_eq!(config.request_plane, RequestPlaneMode::Tcp);
        assert!(config.nats_config.is_some());
    }

    #[test]
    fn production_router_enables_replica_sync() {
        assert!(gwp_kv_router_config(120).router_replica_sync);
    }

    #[test]
    fn model_event_components_are_safe_stable_and_collision_resistant() {
        let transport_key = |value| OracleVersionId::new(value).unwrap().transport_key();
        assert_eq!(transport_key("composer-2-5"), "composer-2-5");
        let slash = transport_key("org/model-v1");
        assert_eq!(slash, transport_key("org/model-v1"));
        assert!(
            slash
                .chars()
                .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
        );
        assert_ne!(slash, transport_key("org.model-v1"));
        assert_ne!(transport_key("Model"), transport_key("model"));
    }

    #[tokio::test]
    async fn registry_starts_empty_and_reconciles_active_and_dormant_models() {
        let registry = GwpRouterRegistry::new_process_local(4, 120)
            .await
            .expect("construct empty registry");
        assert!(registry.routers.is_empty());
        assert!(registry.dormant.is_empty());

        let both = topology(&["alpha", "beta"]);
        registry
            .reconcile_topology(&both, &HashSet::new())
            .await
            .expect("create configured routers");
        assert_eq!(registry.routers.len(), 2);

        let beta_id = OracleVersionId::new("beta").unwrap();
        let beta = registry.model_router(&beta_id).expect("beta router");
        let alpha_only = topology(&["alpha"]);
        registry
            .reconcile_topology(&alpha_only, &HashSet::new())
            .await
            .expect("reconcile surviving router");
        registry.retire_absent(&alpha_only);
        assert_eq!(registry.routers.len(), 1);
        assert_eq!(registry.dormant.len(), 1);
        assert!(beta.workers_tx.borrow().is_empty());

        registry
            .reconcile_topology(&both, &HashSet::new())
            .await
            .expect("reactivate removed router");
        assert!(Arc::ptr_eq(
            &beta,
            &registry.model_router(&beta_id).expect("reactivated beta")
        ));
        assert_eq!(registry.routers.len(), 2);
        assert!(registry.dormant.is_empty());
    }

    /// The de-risk test from the design doc: a `KvRouter` constructed with no
    /// etcd and no NATS, fed workers through a bare watch channel, must route,
    /// prefer warm prefixes, and run the full slot lifecycle.
    #[tokio::test]
    async fn kv_router_without_etcd_or_nats() {
        let block_size = 4;
        let (_registry, router, tx) = process_local_model_router(block_size).await;

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
        let (_registry, router, tx) = process_local_model_router(4).await;
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
        let (_registry, router, tx) = process_local_model_router(4).await;

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
    fn unrelated_observed_publication_preserves_worker_anchor() {
        let store = ObservedLoadStore::default();
        let loads = HashMap::from([
            (1, ObservedWorkerLoad::default()),
            (2, ObservedWorkerLoad::default()),
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
