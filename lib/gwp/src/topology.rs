// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical routing topology and the publication boundary between topology
//! producers (currently the ConfigMap/planner reflector) and request routing.
//!
//! Producers publish complete snapshots. The controller validates and fans a
//! snapshot out to the scheduler worker watch, observed-load store, readiness,
//! metrics, and the request-path [`TopologyStore`]. Acquisition-specific
//! failure handling stays in the producer.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use arc_swap::ArcSwap;
use dynamo_kv_router::protocols::WorkerId;
use dynamo_llm::local_model::runtime_config::ModelRuntimeConfig;
use parking_lot::Mutex;
#[cfg(feature = "server")]
use serde::Serialize;
use tokio::sync::mpsc;

use crate::config::{
    EndpointConfig, EndpointId, GwpConfig, ModelStagePolicy, ModelTokenizationConfig,
    RoutingRequirements, properties_satisfy_routing_requirements,
};
use crate::lifecycle::Lifecycle;
use crate::metrics::GwpMetrics;
use crate::router::{GwpRouterRegistry, ObservedLoadStore, ObservedWorkerLoad, WorkerConfigSender};

/// Data-plane endpoint fields. Planner connection details are intentionally
/// absent: they belong to the current topology producer, not routing state.
#[derive(Clone, Debug)]
pub struct RoutableEndpoint {
    pub ingress_url: url::Url,
    pub api_key: String,
    pub properties: BTreeMap<String, std::collections::BTreeSet<String>>,
}

impl From<&EndpointConfig> for RoutableEndpoint {
    fn from(endpoint: &EndpointConfig) -> Self {
        Self {
            ingress_url: endpoint.ingress_url.clone(),
            api_key: endpoint.api_key.clone(),
            properties: endpoint.properties.clone(),
        }
    }
}

/// Internal profile key. The current adapter creates one profile per
/// canonical model; a future topology producer may share one across models.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ProfileId(pub String);

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedModelProfile {
    pub stages: ModelStagePolicy,
    pub tokenization: ModelTokenizationConfig,
}

#[derive(Clone, Debug)]
pub struct ModelBinding {
    pub endpoints: HashSet<EndpointId>,
    pub profile: ProfileId,
}

/// One externally observed scheduler worker.
#[derive(Clone, Debug)]
pub struct TopologyWorker {
    pub endpoint: EndpointId,
    pub runtime: ModelRuntimeConfig,
    pub observed_load: Option<ObservedWorkerLoad>,
}

#[cfg(feature = "server")]
#[derive(Debug, Eq, PartialEq, Serialize)]
pub(crate) struct OracleInfo {
    oracle_version_id: String,
    replicas: usize,
}

#[cfg(feature = "server")]
#[derive(Debug, Eq, PartialEq, Serialize)]
pub(crate) struct GwpInfo {
    oracles: Vec<OracleInfo>,
}

/// Complete, immutable routing generation consumed by a scheduling call.
#[derive(Clone, Debug, Default)]
pub struct TopologySnapshot {
    pub endpoints: BTreeMap<EndpointId, RoutableEndpoint>,
    pub aliases: BTreeMap<String, String>,
    pub models: HashMap<String, ModelBinding>,
    pub profiles: HashMap<ProfileId, ResolvedModelProfile>,
    pub workers: HashMap<WorkerId, TopologyWorker>,
}

impl TopologySnapshot {
    /// Normalize the existing ConfigMap schema and a producer-owned worker
    /// view into the source-neutral topology used by routing.
    pub fn from_config(
        config: &GwpConfig,
        workers: HashMap<WorkerId, TopologyWorker>,
    ) -> anyhow::Result<Self> {
        let endpoints = config
            .endpoints
            .iter()
            .map(|(id, endpoint)| (id.clone(), RoutableEndpoint::from(endpoint)))
            .collect();
        let mut models: HashMap<String, ModelBinding> = HashMap::new();
        let mut profiles = HashMap::new();
        for route in &config.routes {
            for configured_model in &route.models {
                let canonical = config.canonical_model(configured_model).to_string();
                let profile = ProfileId(canonical.clone());
                models
                    .entry(canonical.clone())
                    .or_insert_with(|| ModelBinding {
                        endpoints: HashSet::new(),
                        profile: profile.clone(),
                    })
                    .endpoints
                    .extend(route.endpoints.iter().cloned());
                profiles
                    .entry(profile)
                    .or_insert_with(|| ResolvedModelProfile {
                        stages: config.model_policy(&canonical),
                        tokenization: config
                            .tokenization
                            .models
                            .get(&canonical)
                            .cloned()
                            .unwrap_or(ModelTokenizationConfig::Pseudo),
                    });
            }
        }
        let snapshot = Self {
            endpoints,
            aliases: config.served_alias_model_map.clone(),
            models,
            profiles,
            workers,
        };
        snapshot.validate()?;
        Ok(snapshot)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        for (alias, canonical) in &self.aliases {
            anyhow::ensure!(
                self.models.contains_key(canonical),
                "model alias {alias} references unknown canonical model {canonical}"
            );
        }
        let mut endpoint_models: HashMap<&EndpointId, &str> = HashMap::new();
        for (model, binding) in &self.models {
            anyhow::ensure!(
                self.profiles.contains_key(&binding.profile),
                "model {model} references unknown profile {}",
                binding.profile.0
            );
            for endpoint in &binding.endpoints {
                anyhow::ensure!(
                    self.endpoints.contains_key(endpoint),
                    "model {model} references unknown endpoint {}",
                    endpoint.0
                );
                if let Some(existing) = endpoint_models.insert(endpoint, model) {
                    anyhow::ensure!(
                        existing == model,
                        "endpoint {} is shared by canonical models {existing} and {model}; model schedulers require distinct endpoint worker pools",
                        endpoint.0
                    );
                }
            }
        }
        for (worker_id, worker) in &self.workers {
            anyhow::ensure!(
                self.endpoints.contains_key(&worker.endpoint),
                "worker {worker_id} references unknown endpoint {}",
                worker.endpoint.0
            );
            anyhow::ensure!(
                worker.runtime.data_parallel_size > 0,
                "worker {worker_id} has zero data_parallel_size"
            );
        }
        Ok(())
    }

    pub fn canonical_model<'a>(&'a self, requested: &'a str) -> &'a str {
        self.aliases
            .get(requested)
            .map(String::as_str)
            .unwrap_or(requested)
    }

    pub fn model_binding(&self, requested: &str) -> Option<(String, &ModelBinding)> {
        let canonical = self.canonical_model(requested);
        self.models
            .get(canonical)
            .map(|binding| (canonical.to_string(), binding))
    }

    pub fn profile(&self, binding: &ModelBinding) -> Option<&ResolvedModelProfile> {
        self.profiles.get(&binding.profile)
    }

    pub fn routing_candidates(&self, requirements: &RoutingRequirements) -> HashSet<EndpointId> {
        self.endpoints
            .iter()
            .filter(|(_, endpoint)| {
                properties_satisfy_routing_requirements(&endpoint.properties, requirements)
            })
            .map(|(endpoint, _)| endpoint.clone())
            .collect()
    }

    pub fn workers_in_endpoints(&self, endpoints: &HashSet<EndpointId>) -> HashSet<WorkerId> {
        self.workers
            .iter()
            .filter(|(_, worker)| endpoints.contains(&worker.endpoint))
            .map(|(worker_id, _)| *worker_id)
            .collect()
    }

    pub fn worker_endpoint(&self, worker_id: WorkerId) -> Option<&EndpointId> {
        self.workers.get(&worker_id).map(|worker| &worker.endpoint)
    }

    pub fn endpoint(&self, endpoint: &EndpointId) -> Option<&RoutableEndpoint> {
        self.endpoints.get(endpoint)
    }

    pub fn is_alive(&self, worker_id: WorkerId) -> bool {
        self.workers.contains_key(&worker_id)
    }

    #[cfg(feature = "server")]
    pub(crate) fn info(&self) -> GwpInfo {
        let mut oracles: Vec<_> = self
            .models
            .iter()
            .map(|(oracle_version_id, binding)| OracleInfo {
                oracle_version_id: oracle_version_id.clone(),
                replicas: self
                    .workers
                    .values()
                    .filter(|worker| binding.endpoints.contains(&worker.endpoint))
                    .count(),
            })
            .collect();
        oracles
            .sort_unstable_by(|left, right| left.oracle_version_id.cmp(&right.oracle_version_id));
        GwpInfo { oracles }
    }

    fn worker_configs(&self) -> HashMap<WorkerId, ModelRuntimeConfig> {
        self.workers
            .iter()
            .map(|(id, worker)| (*id, worker.runtime.clone()))
            .collect()
    }

    fn observed_loads(&self) -> HashMap<WorkerId, ObservedWorkerLoad> {
        self.workers
            .iter()
            .filter_map(|(id, worker)| worker.observed_load.map(|load| (*id, load)))
            .collect()
    }
}

/// Atomic request-path view. The small mutex serializes authoritative replaces
/// with positive worker confirmations from successful downstream responses.
#[derive(Clone)]
pub struct TopologyStore {
    inner: Arc<ArcSwap<TopologySnapshot>>,
    update: Arc<Mutex<()>>,
}

impl TopologyStore {
    pub fn new(initial: TopologySnapshot) -> Self {
        Self {
            inner: Arc::new(ArcSwap::from_pointee(initial)),
            update: Arc::new(Mutex::new(())),
        }
    }

    pub fn load(&self) -> Arc<TopologySnapshot> {
        self.inner.load_full()
    }

    pub fn replace(&self, snapshot: TopologySnapshot) {
        let _guard = self.update.lock();
        self.inner.store(Arc::new(snapshot));
    }

    /// Record positive liveness evidence from the endpoint response. The next
    /// provider snapshot remains authoritative and may remove this worker.
    pub fn confirm_worker(&self, endpoint: &EndpointId, worker_id: WorkerId) -> anyhow::Result<()> {
        let _guard = self.update.lock();
        let current = self.inner.load_full();
        if let Some(worker) = current.workers.get(&worker_id) {
            anyhow::ensure!(
                &worker.endpoint == endpoint,
                "worker {worker_id} is advertised by both {} and {}",
                worker.endpoint.0,
                endpoint.0
            );
            return Ok(());
        }
        anyhow::ensure!(
            current.endpoints.contains_key(endpoint),
            "unknown endpoint {}",
            endpoint.0
        );
        let mut next = (*current).clone();
        next.workers.insert(
            worker_id,
            TopologyWorker {
                endpoint: endpoint.clone(),
                runtime: ModelRuntimeConfig::default(),
                observed_load: None,
            },
        );
        self.inner.store(Arc::new(next));
        Ok(())
    }
}

/// Complete producer update. `refreshed_workers` identifies external load
/// observations whose local scheduler anchors must be recaptured.
pub struct TopologyUpdate {
    pub snapshot: TopologySnapshot,
    pub refreshed_workers: HashSet<WorkerId>,
}

pub type TopologyUpdateSender = mpsc::Sender<TopologyUpdate>;

/// Validates and publishes source-neutral topology updates.
pub struct TopologyController {
    store: TopologyStore,
    workers_tx: Option<WorkerConfigSender>,
    observed_loads: Option<ObservedLoadStore>,
    model_routers: Option<Arc<GwpRouterRegistry>>,
    lifecycle: Option<Lifecycle>,
    metrics: Option<GwpMetrics>,
}

impl TopologyController {
    pub fn new(
        store: TopologyStore,
        workers_tx: WorkerConfigSender,
        observed_loads: ObservedLoadStore,
    ) -> Self {
        Self {
            store,
            workers_tx: Some(workers_tx),
            observed_loads: Some(observed_loads),
            model_routers: None,
            lifecycle: None,
            metrics: None,
        }
    }

    pub fn model_scoped(store: TopologyStore, routers: Arc<GwpRouterRegistry>) -> Self {
        Self {
            store,
            workers_tx: None,
            observed_loads: None,
            model_routers: Some(routers),
            lifecycle: None,
            metrics: None,
        }
    }

    pub fn with_lifecycle(mut self, lifecycle: Lifecycle) -> Self {
        self.lifecycle = Some(lifecycle);
        self
    }

    pub fn with_metrics(mut self, metrics: GwpMetrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    async fn publish(&self, update: TopologyUpdate) -> anyhow::Result<()> {
        update.snapshot.validate()?;
        if let Some(routers) = &self.model_routers {
            routers
                .reconcile_topology(&update.snapshot, &update.refreshed_workers)
                .await?;
        } else {
            self.observed_loads
                .as_ref()
                .expect("legacy observed load store")
                .replace(update.snapshot.observed_loads());
        }
        if let Some(workers_tx) = &self.workers_tx {
            workers_tx
                .send(update.snapshot.worker_configs())
                .map_err(|_| anyhow::anyhow!("router worker feed closed"))?;
        }
        if let Some(metrics) = &self.metrics {
            let mut per_endpoint: HashMap<EndpointId, usize> = update
                .snapshot
                .endpoints
                .keys()
                .cloned()
                .map(|endpoint| (endpoint, 0))
                .collect();
            for worker in update.snapshot.workers.values() {
                *per_endpoint.entry(worker.endpoint.clone()).or_default() += 1;
            }
            metrics.replace_scheduler_live_workers(per_endpoint);
        }
        if let Some(lifecycle) = &self.lifecycle {
            lifecycle.update_routing(update.snapshot.workers.len());
        }
        self.store.replace(update.snapshot.clone());
        if let Some(routers) = &self.model_routers {
            routers.retire_absent(&update.snapshot);
        }
        Ok(())
    }

    pub fn spawn(self, mut updates: mpsc::Receiver<TopologyUpdate>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            while let Some(update) = updates.recv().await {
                if let Err(error) = self.publish(update).await {
                    tracing::error!(%error, "rejected topology update; retaining last valid snapshot");
                    if self
                        .workers_tx
                        .as_ref()
                        .is_some_and(WorkerConfigSender::is_closed)
                    {
                        return;
                    }
                }
            }
            tracing::error!("topology producer exited; controller stopping");
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ModelRoute, RoutingConfig};

    fn config() -> GwpConfig {
        GwpConfig {
            endpoints: BTreeMap::from([(
                EndpointId("a".into()),
                EndpointConfig {
                    ingress_url: url::Url::parse("http://a.example/v1").unwrap(),
                    api_key: String::new(),
                    planner_url: url::Url::parse("http://planner.example/deep/health").unwrap(),
                    planner_api_key: None,
                    properties: Default::default(),
                },
            )]),
            routes: vec![ModelRoute {
                models: vec!["model".into()],
                endpoints: vec![EndpointId("a".into())],
            }],
            routing: RoutingConfig {
                block_size: 4,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn config_normalizes_profiles_routes_and_workers() {
        let mut runtime = ModelRuntimeConfig::default();
        runtime.taints.insert("zone-a".into());
        let snapshot = TopologySnapshot::from_config(
            &config(),
            HashMap::from([(
                7,
                TopologyWorker {
                    endpoint: EndpointId("a".into()),
                    runtime,
                    observed_load: Some(ObservedWorkerLoad {
                        prefill_tokens: 10,
                        decode_blocks: 2,
                        active_requests: 1,
                    }),
                },
            )]),
        )
        .unwrap();
        let (_, binding) = snapshot.model_binding("model").unwrap();
        assert_eq!(
            snapshot.profile(binding).unwrap().stages,
            ModelStagePolicy::default()
        );
        assert!(snapshot.workers[&7].runtime.taints.contains("zone-a"));
        assert_eq!(snapshot.worker_endpoint(7), Some(&EndpointId("a".into())));
    }

    #[cfg(feature = "server")]
    #[test]
    fn info_lists_oracles_and_live_replica_counts() {
        let mut config = config();
        config.endpoints.insert(
            EndpointId("b".into()),
            EndpointConfig {
                ingress_url: url::Url::parse("http://b.example/v1").unwrap(),
                api_key: String::new(),
                planner_url: url::Url::parse("http://planner-b.example/deep/health").unwrap(),
                planner_api_key: None,
                properties: Default::default(),
            },
        );
        config.routes.push(ModelRoute {
            models: vec!["empty-model".into()],
            endpoints: vec![EndpointId("b".into())],
        });
        let snapshot = TopologySnapshot::from_config(
            &config,
            HashMap::from([
                (
                    7,
                    TopologyWorker {
                        endpoint: EndpointId("a".into()),
                        runtime: ModelRuntimeConfig::default(),
                        observed_load: None,
                    },
                ),
                (
                    8,
                    TopologyWorker {
                        endpoint: EndpointId("a".into()),
                        runtime: ModelRuntimeConfig::default(),
                        observed_load: None,
                    },
                ),
            ]),
        )
        .unwrap();

        let info = snapshot.info();
        assert_eq!(
            info,
            GwpInfo {
                oracles: vec![
                    OracleInfo {
                        oracle_version_id: "empty-model".into(),
                        replicas: 0,
                    },
                    OracleInfo {
                        oracle_version_id: "model".into(),
                        replicas: 2,
                    },
                ],
            }
        );
        assert_eq!(
            serde_json::to_value(info).unwrap(),
            serde_json::json!({
                "oracles": [
                    {"oracle_version_id": "empty-model", "replicas": 0},
                    {"oracle_version_id": "model", "replicas": 2}
                ]
            })
        );
    }

    #[test]
    fn rejects_worker_owned_by_unknown_endpoint() {
        let error = TopologySnapshot::from_config(
            &config(),
            HashMap::from([(
                7,
                TopologyWorker {
                    endpoint: EndpointId("missing".into()),
                    runtime: ModelRuntimeConfig::default(),
                    observed_load: None,
                },
            )]),
        )
        .unwrap_err();
        assert!(error.to_string().contains("unknown endpoint"));
    }

    #[test]
    fn rejects_endpoint_worker_pool_shared_by_canonical_models() {
        let mut config = config();
        config.routes.push(ModelRoute {
            models: vec!["other-model".into()],
            endpoints: vec![EndpointId("a".into())],
        });
        let error = TopologySnapshot::from_config(&config, HashMap::new()).unwrap_err();
        assert!(error.to_string().contains("shared by canonical models"));
    }

    #[test]
    fn confirmation_is_provisional_and_cross_endpoint_safe() {
        let config = config();
        let store =
            TopologyStore::new(TopologySnapshot::from_config(&config, HashMap::new()).unwrap());
        store.confirm_worker(&EndpointId("a".into()), 9).unwrap();
        assert!(store.load().is_alive(9));
        let authoritative = TopologySnapshot::from_config(&config, HashMap::new()).unwrap();
        store.replace(authoritative);
        assert!(!store.load().is_alive(9));
    }

    #[tokio::test]
    async fn controller_publishes_taints_and_retains_last_valid_snapshot() {
        let config = config();
        let initial = TopologySnapshot::from_config(&config, HashMap::new()).unwrap();
        let store = TopologyStore::new(initial);
        let (workers_tx, workers_rx) = tokio::sync::watch::channel(HashMap::new());
        let controller =
            TopologyController::new(store.clone(), workers_tx, ObservedLoadStore::default());
        let mut runtime = ModelRuntimeConfig::default();
        runtime.taints.insert("zone-a".into());
        let valid = TopologySnapshot::from_config(
            &config,
            HashMap::from([(
                7,
                TopologyWorker {
                    endpoint: EndpointId("a".into()),
                    runtime,
                    observed_load: None,
                },
            )]),
        )
        .unwrap();
        controller
            .publish(TopologyUpdate {
                snapshot: valid,
                refreshed_workers: HashSet::from([7]),
            })
            .await
            .unwrap();
        assert!(workers_rx.borrow()[&7].taints.contains("zone-a"));

        let mut invalid = (*store.load()).clone();
        invalid.profiles.clear();
        assert!(
            controller
                .publish(TopologyUpdate {
                    snapshot: invalid,
                    refreshed_workers: HashSet::new(),
                })
                .await
                .is_err()
        );
        assert!(store.load().is_alive(7));
    }
}
