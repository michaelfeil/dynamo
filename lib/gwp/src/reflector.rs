// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The reflector. See design doc "Reflector loop" and Decision 1.
//!
//! Every ~1s, polls each endpoint deployment's planner service. Its
//! `GET /deep/health` returns the internal worker list
//! with per-worker load in `detailed_load_data` (keyed by u64 worker id, with
//! prefill/decode token counts, block counts, request counts, and the
//! disaggregation role). One poll cycle drives three things:
//!
//! 1. **Endpoint liveness** — a binding is honored iff its deployment still
//!    has at least one planner-observed worker.
//! 2. **Endpoint ownership** — every planner worker maps back to its
//!    independently routable ingress.
//! 3. **The router feed** — a full `HashMap<WorkerId, ModelRuntimeConfig>`
//!    snapshot of live planner workers.
//!
//! Failure semantics: **last-known-good + staleness grace, per endpoint**. A
//! usable poll (200 with `detailed_load_data` present) reconciles that
//! endpoint immediately; a failed poll (non-200 / timeout / parse error /
//! `detailed_load_data: null`, which the planner returns while its router is
//! unreachable) holds that endpoint's last-known-good. Only after an endpoint's
//! planner has been unusable for the configured staleness-grace window is its
//! endpoint evicted (turning its sticks into unsticks). An explicit
//! `healthy: false` is authoritative and cordons the endpoint immediately. One
//! endpoint's planner blip never disturbs another endpoint.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use bytes::BytesMut;
use dynamo_llm::local_model::runtime_config::ModelRuntimeConfig;
use futures::stream::{FuturesUnordered, StreamExt};
use parking_lot::RwLock;

use crate::config::{ConfigStore, EndpointConfig, EndpointId};
use crate::identity::EndpointTable;
use crate::lifecycle::Lifecycle;
use crate::metrics::GwpMetrics;
use crate::models::ModelCatalog;
use crate::router::{GwpRouter, RemoteLoadStore, RemoteWorkerLoad, WorkerConfigSender};

/// How often the reflector polls the planners.
pub const POLL_INTERVAL: Duration = Duration::from_secs(1);

/// Per-poll HTTP timeout. The configured staleness grace should be longer so
/// several attempts fit inside one grace window.
const POLL_TIMEOUT: Duration = Duration::from_secs(15);

/// Planner responses are operator-controlled but arrive over the network.
/// Bound buffering so a broken planner cannot exhaust the GWP process.
const MAX_PLANNER_BODY_BYTES: usize = 50 * 1024 * 1024;

/// Per-worker load from the planner's `detailed_load_data`.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct DetailedLoadData {
    pub num_prefill_tokens: i64,
    pub num_decode_tokens: i64,
    pub num_decode_blocks: i64,
    pub num_requests: i64,
    /// Disaggregation role: `prefill_and_decode` | `prefill` | `decode`.
    pub role: String,
}

/// The subset of the planner's `GET /deep/health` response GWP consumes
/// (`DeepHealthResponse` in `planner_common.py`); unknown fields ignored.
#[derive(Debug, serde::Deserialize)]
pub struct DeepHealthResponse {
    pub healthy: bool,
    /// Worker list keyed by u64 worker id (JSON object keys arrive as
    /// strings; serde_json parses them into u64). `None` while the planner
    /// cannot reach its router — treated as a failed poll, not an empty
    /// endpoint.
    #[serde(default)]
    pub detailed_load_data: Option<HashMap<u64, DetailedLoadData>>,
}

/// The set of currently alive planner `WorkerId`s. The session resolver consults
/// this to decide stick vs. unstick (design doc "Stick/unstick" mental model).
#[derive(Clone, Default)]
pub struct LivenessSet {
    inner: Arc<RwLock<HashSet<u64>>>,
}

impl LivenessSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// True iff `worker_id` is currently advertised by a live endpoint.
    /// This is the stick/unstick predicate.
    pub fn is_alive(&self, worker_id: u64) -> bool {
        self.inner.read().contains(&worker_id)
    }

    pub fn replace(&self, live: HashSet<u64>) {
        *self.inner.write() = live;
    }

    pub fn insert(&self, worker_id: u64) {
        self.inner.write().insert(worker_id);
    }

    pub fn clear(&self) {
        self.inner.write().clear();
    }

    pub fn len(&self) -> usize {
        self.inner.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.read().is_empty()
    }
}

/// Grace-window state machine, one per endpoint. Time is injected so tests can
/// drive it with paused tokio time.
struct PollState {
    /// When this endpoint's planner last answered usably (or the loop started —
    /// a planner down from the very start still gets one grace window).
    last_good: tokio::time::Instant,
    /// Whether the grace window already expired and this endpoint was cleared,
    /// so a long outage clears exactly once.
    cleared: bool,
}

impl PollState {
    fn new(now: tokio::time::Instant) -> Self {
        Self {
            last_good: now,
            cleared: false,
        }
    }

    fn on_success(&mut self, now: tokio::time::Instant) {
        self.last_good = now;
        self.cleared = false;
    }

    /// Returns true iff this failure crosses the grace boundary and the
    /// caller must evict the cluster's workers (exactly once per outage).
    fn on_failure(&mut self, now: tokio::time::Instant, grace: Duration) -> bool {
        if self.cleared || now.duration_since(self.last_good) <= grace {
            return false;
        }
        self.cleared = true;
        true
    }
}

/// Per-endpoint reflector state: the grace machine plus the last-known-good
/// worker and load snapshots.
struct EndpointState {
    poll: PollState,
    last_good_workers: HashSet<u64>,
    last_good_loads: HashMap<u64, RemoteWorkerLoad>,
}

/// Background task polling every endpoint planner and reflecting deployment
/// candidates into the local router's world. Construct with the
/// [`EndpointTable`] shared with the proxy and the [`WorkerConfigSender`]
/// returned by [`GwpRouter::new`](crate::router::GwpRouter::new).
pub struct Reflector {
    config: ConfigStore,
    http: reqwest::Client,
    table: EndpointTable,
    liveness: LivenessSet,
    models: ModelCatalog,
    remote_loads: RemoteLoadStore,
    load_anchor_router: Option<Arc<GwpRouter>>,
    workers_tx: WorkerConfigSender,
    endpoints: HashMap<EndpointId, EndpointState>,
    lifecycle: Option<Lifecycle>,
    metrics: Option<GwpMetrics>,
}

impl Reflector {
    pub fn new(
        config: impl Into<ConfigStore>,
        table: EndpointTable,
        workers_tx: WorkerConfigSender,
    ) -> Self {
        Self::with_state(
            config,
            table,
            workers_tx,
            ModelCatalog::new(),
            RemoteLoadStore::default(),
        )
    }

    pub fn with_state(
        config: impl Into<ConfigStore>,
        table: EndpointTable,
        workers_tx: WorkerConfigSender,
        models: ModelCatalog,
        remote_loads: RemoteLoadStore,
    ) -> Self {
        let config = config.into();
        let current = config.load();
        let now = tokio::time::Instant::now();
        let endpoints = current
            .endpoints
            .keys()
            .map(|endpoint_id| {
                (
                    endpoint_id.clone(),
                    EndpointState {
                        poll: PollState::new(now),
                        last_good_workers: HashSet::new(),
                        last_good_loads: HashMap::new(),
                    },
                )
            })
            .collect();
        models.replace_from_config(&current);
        let http = reqwest::Client::builder()
            .timeout(POLL_TIMEOUT)
            .build()
            .expect("reqwest client");
        Self {
            config,
            http,
            table,
            liveness: LivenessSet::new(),
            models,
            remote_loads,
            load_anchor_router: None,
            workers_tx,
            endpoints,
            lifecycle: None,
            metrics: None,
        }
    }

    /// Report each complete published routing view into process readiness.
    pub fn with_lifecycle(mut self, lifecycle: Lifecycle) -> Self {
        self.lifecycle = Some(lifecycle);
        self
    }

    pub fn with_metrics(mut self, metrics: GwpMetrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Capture local scheduler load at each successful planner snapshot. This
    /// is installed by the production builder; reflector-only tests can keep
    /// using the store without constructing a full router.
    pub fn with_load_anchor_router(mut self, router: Arc<GwpRouter>) -> Self {
        self.load_anchor_router = Some(router);
        self
    }

    /// The liveness set the session resolver consults for stick/unstick.
    pub fn liveness(&self) -> LivenessSet {
        self.liveness.clone()
    }

    pub fn models(&self) -> ModelCatalog {
        self.models.clone()
    }

    fn reconcile_config(&mut self, config: &crate::config::GwpConfig) {
        let configured: HashSet<EndpointId> = config.endpoints.keys().cloned().collect();
        let removed: Vec<EndpointId> = self
            .endpoints
            .keys()
            .filter(|endpoint_id| !configured.contains(*endpoint_id))
            .cloned()
            .collect();
        for endpoint_id in removed {
            self.endpoints.remove(&endpoint_id);
            tracing::info!(
                endpoint = %endpoint_id.0,
                "removed endpoint from live configuration"
            );
        }

        let now = tokio::time::Instant::now();
        for endpoint_id in config.endpoints.keys() {
            if !self.endpoints.contains_key(endpoint_id) {
                self.endpoints.insert(
                    endpoint_id.clone(),
                    EndpointState {
                        poll: PollState::new(now),
                        last_good_workers: HashSet::new(),
                        last_good_loads: HashMap::new(),
                    },
                );
                tracing::info!(
                    endpoint = %endpoint_id.0,
                    "added endpoint from live configuration"
                );
            }
        }
        self.models.replace_from_config(config);
    }

    /// Feed every planner worker as a scheduler candidate and retain its
    /// owning ingress. The endpoint's local router may select a different
    /// worker; response-header reconciliation corrects that provisional
    /// booking in [`crate::core::GwpCore::response_started`].
    fn apply_endpoint_workers(
        &mut self,
        endpoint_id: &EndpointId,
        workers: HashMap<u64, DetailedLoadData>,
    ) {
        let mut live = HashSet::new();
        let mut loads = HashMap::new();
        for (worker_id, load) in workers {
            if let Err(error) = self.table.upsert_worker(worker_id, endpoint_id.clone()) {
                tracing::error!(endpoint = %endpoint_id.0, worker_id, %error);
                continue;
            }
            live.insert(worker_id);
            loads.insert(
                worker_id,
                RemoteWorkerLoad {
                    prefill_tokens: load.num_prefill_tokens.max(0) as usize,
                    decode_blocks: load.num_decode_blocks.max(0) as usize,
                    active_requests: load.num_requests.max(0) as usize,
                },
            );
        }
        if let Some(state) = self.endpoints.get_mut(endpoint_id) {
            state.last_good_workers = live;
            state.last_good_loads = loads;
        }
    }

    /// Rebuild the global view (liveness, endpoint table, router feed) as the
    /// union of all endpoints' last-known-good candidates.
    async fn publish(&self, refreshed_workers: &HashSet<u64>) {
        let mut live = HashSet::new();
        for state in self.endpoints.values() {
            live.extend(state.last_good_workers.iter().copied());
        }
        let configs: HashMap<u64, ModelRuntimeConfig> = live
            .iter()
            .map(|&gwp_id| (gwp_id, ModelRuntimeConfig::default()))
            .collect();
        let loads: HashMap<u64, RemoteWorkerLoad> = self
            .endpoints
            .values()
            .flat_map(|state| state.last_good_loads.iter().map(|(id, load)| (*id, *load)))
            .collect();
        let routable_workers = live.len();
        if let Some(metrics) = &self.metrics {
            metrics.replace_scheduler_live_workers(
                self.endpoints
                    .iter()
                    .map(|(endpoint, state)| (endpoint.clone(), state.last_good_workers.len()))
                    .collect(),
            );
        }

        self.table.retain(&live);
        self.liveness.replace(live);
        if let Some(router) = &self.load_anchor_router {
            if let Err(error) = router
                .replace_remote_loads(loads.clone(), refreshed_workers)
                .await
            {
                tracing::warn!(%error, "failed to capture local load anchors; publishing planner loads with existing anchors");
                self.remote_loads.replace(loads);
            }
        } else {
            self.remote_loads.replace(loads);
        }
        if let Some(lifecycle) = &self.lifecycle {
            let config = self.config.load();
            let configured_routes_covered = config.routes.iter().all(|route| {
                route.models.iter().all(|model| {
                    route.endpoints.iter().any(|endpoint_id| {
                        self.endpoints.get(endpoint_id).is_some_and(|state| {
                            !state.last_good_workers.is_empty()
                                && self.models.endpoint_serves(endpoint_id, model)
                        })
                    })
                })
            });
            lifecycle.update_routing(routable_workers, configured_routes_covered);
        }
        // send() fails only when the router is gone; the loop exits via the
        // JoinHandle then anyway.
        let _ = self.workers_tx.send(configs);
    }

    async fn fetch(
        http: reqwest::Client,
        endpoint: EndpointConfig,
    ) -> anyhow::Result<DeepHealthResponse> {
        let mut request = http.get(endpoint.planner_url.clone());
        if !endpoint.planner_api_key().is_empty() {
            request = request.bearer_auth(endpoint.planner_api_key());
        }
        let response = request.send().await?.error_for_status()?;
        if response
            .content_length()
            .is_some_and(|length| length > MAX_PLANNER_BODY_BYTES as u64)
        {
            anyhow::bail!(
                "planner response exceeds {} byte limit",
                MAX_PLANNER_BODY_BYTES
            );
        }

        let mut body = BytesMut::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            if body.len().saturating_add(chunk.len()) > MAX_PLANNER_BODY_BYTES {
                anyhow::bail!(
                    "planner response exceeds {} byte limit",
                    MAX_PLANNER_BODY_BYTES
                );
            }
            body.extend_from_slice(&chunk);
        }
        Ok(serde_json::from_slice(&body)?)
    }

    async fn apply_fetch_result(
        &mut self,
        endpoint_id: EndpointId,
        fetched: anyhow::Result<DeepHealthResponse>,
        grace: Duration,
    ) {
        // `healthy: false` is an authoritative planner cordon and removes the
        // endpoint immediately. Missing load data and request failures are
        // ambiguous, so those retain last-known-good through grace.
        let workers = match fetched {
            Ok(DeepHealthResponse { healthy: false, .. }) => {
                tracing::warn!(
                    endpoint = %endpoint_id.0,
                    "planner reported unhealthy; cordoning endpoint immediately"
                );
                if let Some(state) = self.endpoints.get_mut(&endpoint_id) {
                    state.last_good_workers.clear();
                    state.last_good_loads.clear();
                }
                self.publish(&HashSet::new()).await;
                return;
            }
            Ok(DeepHealthResponse {
                healthy: true,
                detailed_load_data: Some(workers),
            }) => Some(workers),
            Ok(response) => {
                tracing::warn!(
                    endpoint = %endpoint_id.0,
                    has_detailed_load_data = response.detailed_load_data.is_some(),
                    "planner reported unusable; holding last-known-good worker set"
                );
                None
            }
            Err(error) => {
                tracing::warn!(
                    endpoint = %endpoint_id.0,
                    %error,
                    "planner poll failed; holding last-known-good worker set"
                );
                None
            }
        };

        let now = tokio::time::Instant::now();
        let mut refreshed_workers = HashSet::new();
        match workers {
            Some(workers) => {
                refreshed_workers.extend(workers.keys().copied());
                if let Some(state) = self.endpoints.get_mut(&endpoint_id) {
                    state.poll.on_success(now);
                }
                self.apply_endpoint_workers(&endpoint_id, workers);
            }
            None => {
                let should_evict = self
                    .endpoints
                    .get_mut(&endpoint_id)
                    .is_some_and(|state| state.poll.on_failure(now, grace));
                if should_evict {
                    tracing::error!(
                        endpoint = %endpoint_id.0,
                        grace_secs = grace.as_secs(),
                        "planner unusable past staleness grace; evicting endpoint (its sessions unstick)"
                    );
                    if let Some(state) = self.endpoints.get_mut(&endpoint_id) {
                        state.last_good_workers = HashSet::new();
                        state.last_good_loads = HashMap::new();
                    }
                }
            }
        }
        self.publish(&refreshed_workers).await;
    }

    /// Poll every endpoint once, publishing each result as soon as it resolves.
    /// Used by focused tests; production uses independent endpoint cadences in
    /// [`Self::run`].
    #[cfg(test)]
    async fn poll_once(&mut self) {
        let current = self.config.load();
        self.reconcile_config(&current);
        // Publish configuration removals even when no planners remain.
        self.publish(&HashSet::new()).await;
        let endpoints: Vec<(EndpointId, EndpointConfig)> = current
            .endpoints
            .iter()
            .map(|(id, endpoint)| (id.clone(), endpoint.clone()))
            .collect();
        let grace = Duration::from_secs(current.routing.planner_staleness_grace_secs);
        let mut fetches = endpoints
            .into_iter()
            .map(|(endpoint_id, endpoint)| {
                let http = self.http.clone();
                async move {
                    let fetched = Self::fetch(http, endpoint).await;
                    (endpoint_id, fetched)
                }
            })
            .collect::<FuturesUnordered<_>>();

        while let Some((endpoint_id, fetched)) = fetches.next().await {
            self.apply_fetch_result(endpoint_id, fetched, grace).await;
        }
    }

    async fn run(&mut self) {
        let mut ticker = tokio::time::interval(POLL_INTERVAL);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut in_flight = HashSet::new();
        let mut fetches = FuturesUnordered::new();

        loop {
            tokio::select! {
                _ = ticker.tick() => {
                    if self.workers_tx.is_closed() {
                        tracing::info!("router dropped; reflector exiting");
                        return;
                    }
                    let current = self.config.load();
                    self.reconcile_config(&current);
                    self.publish(&HashSet::new()).await;
                    in_flight.retain(|endpoint_id| current.endpoints.contains_key(endpoint_id));
                    for (endpoint_id, endpoint) in &current.endpoints {
                        if !in_flight.insert(endpoint_id.clone()) {
                            continue;
                        }
                        let endpoint_id = endpoint_id.clone();
                        let endpoint = endpoint.clone();
                        let planner_url = endpoint.planner_url.clone();
                        let http = self.http.clone();
                        fetches.push(async move {
                            let fetched = Self::fetch(http, endpoint).await;
                            (endpoint_id, planner_url, fetched)
                        });
                    }
                }
                Some((endpoint_id, planner_url, fetched)) = fetches.next(), if !fetches.is_empty() => {
                    in_flight.remove(&endpoint_id);
                    let current = self.config.load();
                    let Some(endpoint) = current.endpoints.get(&endpoint_id) else {
                        continue;
                    };
                    if endpoint.planner_url != planner_url {
                        tracing::debug!(endpoint = %endpoint_id.0, "discarding result from replaced planner URL");
                        continue;
                    }
                    let grace = Duration::from_secs(current.routing.planner_staleness_grace_secs);
                    self.apply_fetch_result(endpoint_id, fetched, grace).await;
                }
            }
        }
    }

    /// Spawn the poll loop. The task ends when the [`WorkerConfigSender`]'s
    /// receiver (the router) is dropped.
    pub fn spawn(mut self) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move { self.run().await })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{EndpointId, GwpConfig, ModelRoute};
    use crate::lifecycle::Lifecycle;
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn eid(s: &str) -> EndpointId {
        EndpointId(s.to_string())
    }

    fn endpoint_config() -> EndpointConfig {
        EndpointConfig {
            ingress_url: url::Url::parse("http://localhost:0/v1").unwrap(),
            api_key: "ck".into(),
            planner_url: url::Url::parse("http://localhost:0/deep/health").unwrap(),
            planner_api_key: None,
            properties: Default::default(),
        }
    }

    fn test_config(endpoints: &[&str]) -> GwpConfig {
        GwpConfig {
            endpoints: endpoints
                .iter()
                .map(|name| (eid(name), endpoint_config()))
                .collect::<BTreeMap<_, _>>(),
            ..Default::default()
        }
    }

    fn make_reflector(endpoints: &[&str]) -> (Reflector, crate::router::WorkerConfigSender) {
        let (tx, _rx) = tokio::sync::watch::channel(HashMap::new());
        let reflector = Reflector::new(test_config(endpoints), EndpointTable::new(), tx.clone());
        (reflector, tx)
    }

    async fn spawn_planner(
        body: impl Into<String>,
        delay: Duration,
        content_length: Option<usize>,
    ) -> url::Url {
        let body = body.into();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            let _ = socket.read(&mut request).await;
            tokio::time::sleep(delay).await;
            let length = content_length.unwrap_or(body.len());
            let headers = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {length}\r\nconnection: close\r\n\r\n"
            );
            socket.write_all(headers.as_bytes()).await.unwrap();
            socket.write_all(body.as_bytes()).await.unwrap();
        });
        url::Url::parse(&format!("http://{addr}/deep/health")).unwrap()
    }

    async fn spawn_repeating_planner(
        body: &'static str,
        delay: Duration,
        hits: Arc<AtomicUsize>,
    ) -> url::Url {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                let hits = hits.clone();
                tokio::spawn(async move {
                    let mut request = [0_u8; 4096];
                    let _ = socket.read(&mut request).await;
                    hits.fetch_add(1, Ordering::Relaxed);
                    tokio::time::sleep(delay).await;
                    let headers = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                        body.len()
                    );
                    socket.write_all(headers.as_bytes()).await.unwrap();
                    socket.write_all(body.as_bytes()).await.unwrap();
                });
            }
        });
        url::Url::parse(&format!("http://{addr}/deep/health")).unwrap()
    }

    async fn apply(reflector: &mut Reflector, endpoint: &str, workers: &[u64]) {
        reflector.apply_endpoint_workers(
            &eid(endpoint),
            workers
                .iter()
                .map(|worker| {
                    (
                        *worker,
                        DetailedLoadData {
                            num_prefill_tokens: 0,
                            num_decode_tokens: 0,
                            num_decode_blocks: 0,
                            num_requests: 0,
                            role: "prefill_and_decode".into(),
                        },
                    )
                })
                .collect(),
        );
        if let Some(state) = reflector.endpoints.get_mut(&eid(endpoint)) {
            state.poll.on_success(tokio::time::Instant::now());
        }
        reflector.publish(&workers.iter().copied().collect()).await;
    }

    #[tokio::test]
    async fn apply_feeds_every_planner_worker_with_endpoint_ownership() {
        let (mut reflector, tx) = make_reflector(&["us-east", "eu-west"]);
        let mut rx = tx.subscribe();

        apply(&mut reflector, "us-east", &[11, 12]).await;
        apply(&mut reflector, "eu-west", &[22, 23]).await;

        for worker in [11, 12, 22, 23] {
            assert!(reflector.liveness().is_alive(worker));
        }
        assert_eq!(reflector.table.get(11), Some(eid("us-east")));
        assert_eq!(reflector.table.get(22), Some(eid("eu-west")));

        let configs = rx.borrow_and_update().clone();
        assert_eq!(configs.len(), 4);
        assert!(
            [11, 12, 22, 23]
                .into_iter()
                .all(|worker| configs.contains_key(&worker))
        );
    }

    #[tokio::test]
    async fn endpoint_remains_live_while_any_internal_worker_exists() {
        let (mut reflector, _tx) = make_reflector(&["us-east"]);
        apply(&mut reflector, "us-east", &[1, 2]).await;
        assert!(reflector.liveness().is_alive(1));
        assert!(reflector.liveness().is_alive(2));

        apply(&mut reflector, "us-east", &[2]).await;
        assert!(!reflector.liveness().is_alive(1));
        assert!(reflector.liveness().is_alive(2));

        apply(&mut reflector, "us-east", &[]).await;
        assert!(!reflector.liveness().is_alive(2));
        assert!(reflector.table.get(1).is_none());
        assert!(reflector.table.get(2).is_none());
    }

    #[tokio::test]
    async fn readiness_tracks_worker_and_configured_route_coverage() {
        let mut config = test_config(&["us-east", "eu-west"]);
        config.routes = vec![
            ModelRoute {
                models: vec!["model-a".into()],
                endpoints: vec![eid("us-east")],
            },
            ModelRoute {
                models: vec!["model-b".into()],
                endpoints: vec![eid("eu-west")],
            },
        ];
        let (tx, _rx) = tokio::sync::watch::channel(HashMap::new());
        let lifecycle = Lifecycle::starting(0, Duration::ZERO);
        let mut reflector =
            Reflector::new(config, EndpointTable::new(), tx).with_lifecycle(lifecycle.clone());

        apply(&mut reflector, "us-east", &[1]).await;
        let missing_route = lifecycle.report();
        assert_eq!(missing_route.routable_workers, 1);
        assert!(!missing_route.configured_routes_covered);
        assert!(!missing_route.ready);

        apply(&mut reflector, "eu-west", &[2]).await;
        assert!(lifecycle.is_ready());

        apply(&mut reflector, "us-east", &[]).await;
        let removed = lifecycle.report();
        assert_eq!(removed.routable_workers, 1);
        assert!(!removed.configured_routes_covered);
        assert!(!removed.ready);
    }

    #[tokio::test]
    async fn endpoint_load_preserves_each_planner_worker_tuple() {
        let (mut reflector, _tx) = make_reflector(&["us-east"]);
        reflector.apply_endpoint_workers(
            &eid("us-east"),
            HashMap::from([
                (
                    1,
                    DetailedLoadData {
                        num_prefill_tokens: 640,
                        num_decode_tokens: 0,
                        num_decode_blocks: 20,
                        num_requests: 2,
                        role: "prefill_and_decode".into(),
                    },
                ),
                (
                    2,
                    DetailedLoadData {
                        num_prefill_tokens: 64,
                        num_decode_tokens: 0,
                        num_decode_blocks: 1,
                        num_requests: 1,
                        role: "prefill_and_decode".into(),
                    },
                ),
            ]),
        );
        let state = &reflector.endpoints[&eid("us-east")];
        assert_eq!(
            state.last_good_loads[&1],
            RemoteWorkerLoad {
                prefill_tokens: 640,
                decode_blocks: 20,
                active_requests: 2,
            }
        );
        assert_eq!(
            state.last_good_loads[&2],
            RemoteWorkerLoad {
                prefill_tokens: 64,
                decode_blocks: 1,
                active_requests: 1,
            }
        );
    }

    #[tokio::test]
    async fn one_endpoints_outage_does_not_disturb_another() {
        let (mut reflector, _tx) = make_reflector(&["us-east", "eu-west"]);
        apply(&mut reflector, "us-east", &[1]).await;
        apply(&mut reflector, "eu-west", &[2]).await;
        let (us, eu) = (1, 2);

        let grace = Duration::from_secs(30);
        let late = tokio::time::Instant::now() + grace + Duration::from_secs(1);
        let state = reflector.endpoints.get_mut(&eid("eu-west")).unwrap();
        assert!(state.poll.on_failure(late, grace));
        state.last_good_workers = HashSet::new();
        reflector.publish(&HashSet::new()).await;

        assert!(reflector.liveness().is_alive(us));
        assert!(!reflector.liveness().is_alive(eu));
        assert!(reflector.table.get(us).is_some());
        assert!(reflector.table.get(eu).is_none());
    }

    #[tokio::test]
    async fn config_reload_adds_and_removes_endpoints_without_restart() {
        let store = ConfigStore::new(test_config(&["us-east", "eu-west"]));
        let (tx, _rx) = tokio::sync::watch::channel(HashMap::new());
        let mut reflector = Reflector::new(store.clone(), EndpointTable::new(), tx.clone());
        apply(&mut reflector, "us-east", &[1]).await;
        apply(&mut reflector, "eu-west", &[2]).await;

        store.replace(test_config(&["us-east"]));
        let next = store.load();
        reflector.reconcile_config(&next);
        reflector.publish(&HashSet::new()).await;

        assert!(reflector.liveness().is_alive(1));
        assert!(!reflector.liveness().is_alive(2));
        assert!(!reflector.endpoints.contains_key(&eid("eu-west")));

        store.replace(test_config(&["us-east", "ap-south"]));
        let next = store.load();
        reflector.reconcile_config(&next);
        assert!(reflector.endpoints.contains_key(&eid("ap-south")));
    }

    #[tokio::test]
    async fn unhealthy_planner_cordons_endpoint_immediately() {
        let planner = spawn_planner(
            r#"{"healthy":false,"detailed_load_data":{"1":{"num_prefill_tokens":0,"num_decode_tokens":0,"num_decode_blocks":0,"num_requests":0,"role":"prefill_and_decode"}}}"#,
            Duration::ZERO,
            None,
        )
        .await;
        let (mut reflector, _tx) = make_reflector(&["us-east"]);
        apply(&mut reflector, "us-east", &[1]).await;
        let mut config = test_config(&["us-east"]);
        config
            .endpoints
            .get_mut(&eid("us-east"))
            .unwrap()
            .planner_url = planner;
        reflector.config.replace(config);

        reflector.poll_once().await;

        assert!(!reflector.liveness().is_alive(1));
        assert!(reflector.table.get(1).is_none());
    }

    #[tokio::test]
    async fn healthy_planner_publishes_without_waiting_for_slow_peer() {
        let fast = spawn_planner(
            r#"{"healthy":true,"detailed_load_data":{"1":{"num_prefill_tokens":0,"num_decode_tokens":0,"num_decode_blocks":0,"num_requests":0,"role":"prefill_and_decode"}}}"#,
            Duration::ZERO,
            None,
        )
        .await;
        let slow = spawn_planner(
            r#"{"healthy":true,"detailed_load_data":{"2":{"num_prefill_tokens":0,"num_decode_tokens":0,"num_decode_blocks":0,"num_requests":0,"role":"prefill_and_decode"}}}"#,
            Duration::from_secs(1),
            None,
        )
        .await;
        let mut config = test_config(&["fast", "slow"]);
        config.endpoints.get_mut(&eid("fast")).unwrap().planner_url = fast;
        config.endpoints.get_mut(&eid("slow")).unwrap().planner_url = slow;
        let (tx, mut rx) = tokio::sync::watch::channel(HashMap::new());
        let mut reflector = Reflector::new(config, EndpointTable::new(), tx);

        let poll = tokio::spawn(async move { reflector.poll_once().await });
        tokio::time::timeout(Duration::from_millis(500), async {
            loop {
                rx.changed().await.unwrap();
                if rx.borrow_and_update().contains_key(&1) {
                    break;
                }
            }
        })
        .await
        .expect("fast planner must publish before the slow planner resolves");
        poll.await.unwrap();
        assert!(rx.borrow().contains_key(&2));
    }

    #[tokio::test]
    async fn slow_planner_does_not_throttle_healthy_planner_cadence() {
        const FAST_BODY: &str = r#"{"healthy":true,"detailed_load_data":{"1":{"num_prefill_tokens":0,"num_decode_tokens":0,"num_decode_blocks":0,"num_requests":0,"role":"prefill_and_decode"}}}"#;
        const SLOW_BODY: &str = r#"{"healthy":true,"detailed_load_data":{"2":{"num_prefill_tokens":0,"num_decode_tokens":0,"num_decode_blocks":0,"num_requests":0,"role":"prefill_and_decode"}}}"#;
        let fast_hits = Arc::new(AtomicUsize::new(0));
        let slow_hits = Arc::new(AtomicUsize::new(0));
        let fast = spawn_repeating_planner(FAST_BODY, Duration::ZERO, fast_hits.clone()).await;
        let slow =
            spawn_repeating_planner(SLOW_BODY, Duration::from_secs(5), slow_hits.clone()).await;
        let mut config = test_config(&["fast", "slow"]);
        config.endpoints.get_mut(&eid("fast")).unwrap().planner_url = fast;
        config.endpoints.get_mut(&eid("slow")).unwrap().planner_url = slow;
        let (tx, rx) = tokio::sync::watch::channel(HashMap::new());
        let handle = Reflector::new(config, EndpointTable::new(), tx).spawn();

        tokio::time::sleep(Duration::from_millis(2_400)).await;
        assert!(
            fast_hits.load(Ordering::Relaxed) >= 3,
            "healthy endpoint was not polled on its own one-second cadence"
        );
        assert_eq!(
            slow_hits.load(Ordering::Relaxed),
            1,
            "a hanging endpoint should have at most one in-flight poll"
        );
        drop(rx);
        handle.abort();
    }

    #[tokio::test]
    async fn removed_endpoint_discards_its_inflight_planner_result() {
        const BODY: &str = r#"{"healthy":true,"detailed_load_data":{"9":{"num_prefill_tokens":0,"num_decode_tokens":0,"num_decode_blocks":0,"num_requests":0,"role":"prefill_and_decode"}}}"#;
        let hits = Arc::new(AtomicUsize::new(0));
        let planner =
            spawn_repeating_planner(BODY, Duration::from_millis(1_500), hits.clone()).await;
        let mut config = test_config(&["removed"]);
        config
            .endpoints
            .get_mut(&eid("removed"))
            .unwrap()
            .planner_url = planner;
        let store = ConfigStore::new(config);
        let (tx, rx) = tokio::sync::watch::channel(HashMap::new());
        let reflector = Reflector::new(store.clone(), EndpointTable::new(), tx);
        let liveness = reflector.liveness();
        let handle = reflector.spawn();

        let deadline = tokio::time::Instant::now() + Duration::from_secs(1);
        while hits.load(Ordering::Relaxed) == 0 && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(hits.load(Ordering::Relaxed), 1);
        store.replace(test_config(&[]));
        tokio::time::sleep(Duration::from_millis(1_700)).await;
        assert!(
            !liveness.is_alive(9),
            "removed endpoint was resurrected by an old planner response"
        );
        drop(rx);
        handle.abort();
    }

    #[tokio::test]
    async fn planner_content_length_over_limit_is_rejected() {
        let planner = spawn_planner("", Duration::ZERO, Some(MAX_PLANNER_BODY_BYTES + 1)).await;
        let mut endpoint = endpoint_config();
        endpoint.planner_url = planner;

        let error = Reflector::fetch(reqwest::Client::new(), endpoint)
            .await
            .unwrap_err();

        assert!(error.to_string().contains("exceeds"));
    }

    #[tokio::test(start_paused = true)]
    async fn grace_window_holds_then_clears_exactly_once() {
        let mut state = PollState::new(tokio::time::Instant::now());
        let grace = Duration::from_secs(15);

        // Failures inside the grace window: hold last-known-good.
        tokio::time::advance(Duration::from_secs(2)).await;
        assert!(!state.on_failure(tokio::time::Instant::now(), grace));
        tokio::time::advance(Duration::from_secs(10)).await;
        assert!(!state.on_failure(tokio::time::Instant::now(), grace));

        // Crossing the boundary clears — exactly once.
        tokio::time::advance(Duration::from_secs(5)).await;
        assert!(state.on_failure(tokio::time::Instant::now(), grace));
        tokio::time::advance(Duration::from_secs(60)).await;
        assert!(
            !state.on_failure(tokio::time::Instant::now(), grace),
            "clear only once per outage"
        );

        // Recovery re-arms the grace window.
        state.on_success(tokio::time::Instant::now());
        tokio::time::advance(Duration::from_secs(2)).await;
        assert!(!state.on_failure(tokio::time::Instant::now(), grace));
        tokio::time::advance(Duration::from_secs(20)).await;
        assert!(state.on_failure(tokio::time::Instant::now(), grace));
    }

    #[test]
    fn deep_health_response_parses_planner_wire_format() {
        // Mirrors planner_common.py's DeepHealthResponse: detailed_load_data
        // is dict[int, DetailedLoadData] — JSON object keys arrive as strings.
        let raw = r#"{
            "num_healthy_workers": 2,
            "router_healthy": true,
            "healthy": true,
            "capacity": 2000,
            "consumption": 400,
            "hardware_scaling_factor": 1.0,
            "internal_potential_loads": null,
            "detailed_load_data": {
                "42": {"num_prefill_tokens": 128, "num_decode_tokens": 4096,
                        "num_decode_blocks": 64, "num_requests": 3,
                        "role": "prefill_and_decode"},
                "7":  {"num_prefill_tokens": 0, "num_decode_tokens": 0,
                        "num_decode_blocks": 0, "num_requests": 0,
                        "role": "decode"}
            }
        }"#;
        let parsed: DeepHealthResponse = serde_json::from_str(raw).unwrap();
        assert!(parsed.healthy);
        let workers = parsed.detailed_load_data.unwrap();
        assert_eq!(workers.len(), 2);
        assert_eq!(workers[&42].num_requests, 3);
        assert_eq!(workers[&42].role, "prefill_and_decode");
        assert_eq!(workers[&7].role, "decode");
    }

    #[test]
    fn null_detailed_load_data_is_not_an_empty_cluster() {
        // The planner returns detailed_load_data: null while its router is
        // unreachable — must be treated as a failed poll (hold), never as
        // "cluster has zero workers" (evict).
        let raw = r#"{"healthy": false, "detailed_load_data": null}"#;
        let parsed: DeepHealthResponse = serde_json::from_str(raw).unwrap();
        assert!(parsed.detailed_load_data.is_none());

        let empty = r#"{"healthy": true, "detailed_load_data": {}}"#;
        let parsed: DeepHealthResponse = serde_json::from_str(empty).unwrap();
        assert_eq!(parsed.detailed_load_data.unwrap().len(), 0);
    }
}
