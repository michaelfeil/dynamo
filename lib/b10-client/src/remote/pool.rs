// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use baseten_configmap::{CoordinatorAffinityConfig, GenerationCoordinatorConfig};
use dynamo_llm::{
    protocols::common::extensions::SessionAffinityId,
    session_affinity::{AffinityAcquire, AffinityCoordinator},
};
use dynamo_runtime::{
    component::{Component, Namespace},
    traits::DistributedRuntimeProvider,
};
use std::{collections::BTreeMap, sync::RwLock, time::Duration};
use tokio::sync::{OnceCell, watch};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

pub const AFFINITY_TOPIC: &str = "generation_coordinator_affinity_events";
const POLL_INTERVAL: Duration = Duration::from_secs(5);
// Allow a polling interval plus a slow peer's bounded probe before expiring healthy rows.
const MAX_SNAPSHOT_AGE: Duration = Duration::from_secs(15);

struct Backend {
    endpoint: Url,
    loads: Vec<crate::WorkerLoad>,
    observed: Instant,
}

impl Backend {
    fn is_current(&self, endpoint: &Url) -> bool {
        &self.endpoint == endpoint && self.observed.elapsed() < MAX_SNAPSHOT_AGE
    }

    fn has_affinity(&self, worker_id: u64, allowed: &[u64]) -> bool {
        self.observed.elapsed() < MAX_SNAPSHOT_AGE
            && self.eligible(allowed)
            && (allowed.is_empty() || allowed.contains(&worker_id))
            && self
                .affinity_workers()
                .any(|load| load.worker_id == worker_id)
    }

    fn affinity_workers(&self) -> impl Iterator<Item = &crate::WorkerLoad> {
        self.loads
            .iter()
            .filter(|load| load.disaggregation_mode != crate::WorkerMode::Decode)
    }

    fn eligible(&self, allowed: &[u64]) -> bool {
        let mut prefill = false;
        let mut decode = false;
        for load in self
            .loads
            .iter()
            .filter(|load| allowed.is_empty() || allowed.contains(&load.worker_id))
        {
            match load.disaggregation_mode {
                crate::WorkerMode::PrefillAndDecode => return true,
                crate::WorkerMode::Prefill => prefill = true,
                crate::WorkerMode::Decode => decode = true,
            }
        }
        prefill && decode
    }
}

struct PoolState {
    backends: RwLock<BTreeMap<String, Backend>>,
    workers: watch::Sender<Vec<u64>>,
    affinity: Option<AffinityCoordinator>,
}

impl PoolState {
    fn retain_live(&self, remotes: &mut BTreeMap<String, Url>, allowed: &[u64]) {
        let backends = self.backends.read().unwrap();
        remotes.retain(|name, endpoint| {
            backends
                .get(name)
                .is_some_and(|backend| backend.is_current(endpoint) && backend.eligible(allowed))
        });
    }
}

#[derive(Clone)]
struct RemoteConfig {
    reader: ConfigReader,
    default_remote: Option<BTreeMap<String, String>>,
}

impl RemoteConfig {
    fn snapshot(&self) -> Result<GenerationCoordinatorConfig> {
        let mut config = self.reader.snapshot().generation_coordinator.clone();
        if config.remotes.is_none() {
            config.remotes = self.default_remote.clone();
        }
        config.validate()?;
        Ok(config)
    }
}

pub(super) struct RemotePool {
    config: RemoteConfig,
    client: Client,
    affinity: Option<(CoordinatorAffinityConfig, Component)>,
    ready: OnceCell<Arc<PoolState>>,
    cancel: CancellationToken,
}

impl Drop for RemotePool {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

impl RemotePool {
    pub(super) async fn bid(
        &self,
        request: protocol::BidRequestV1,
    ) -> Result<protocol::BidResponseV1> {
        let mut remotes = self.remotes()?;
        if remotes.len() > 1 {
            self.state().await?.retain_live(&mut remotes, &[]);
        }
        Ok(self.best_bid(&remotes, request).await?.1)
    }

    async fn best_bid(
        &self,
        remotes: &BTreeMap<String, Url>,
        request: protocol::BidRequestV1,
    ) -> Result<(Url, protocol::BidResponseV1)> {
        request.validate()?;
        let bids = futures::future::join_all(remotes.iter().map(|(name, endpoint)| {
            let request = request.clone();
            async move {
                match fetch_bid(&self.client, endpoint, request).await {
                    Ok(bid) => Some((name, endpoint, bid)),
                    Err(_) => {
                        tracing::warn!(backend = name, "coordinator bid failed");
                        None
                    }
                }
            }
        }))
        .await;
        let (_, endpoint, bid) = bids
            .into_iter()
            .flatten()
            .min_by(|left, right| {
                left.2
                    .score()
                    .cmp(&right.2.score())
                    .then_with(|| {
                        (left.0.as_str() != "default").cmp(&(right.0.as_str() != "default"))
                    })
                    .then_with(|| left.0.cmp(right.0))
            })
            .context("no usable remote coordinator bids")?;
        Ok((endpoint.clone(), bid))
    }

    pub(super) fn new(
        config: ConfigReader,
        client: Client,
        namespace: Option<&Namespace>,
        default_remote: Option<String>,
    ) -> Result<Self> {
        let config = RemoteConfig {
            reader: config,
            default_remote: default_remote.map(|url| BTreeMap::from([("default".into(), url)])),
        };
        let snapshot = config.snapshot()?;
        snapshot
            .remotes
            .as_ref()
            .context("remote coordinator requires remotes")?;
        let affinity = snapshot
            .affinity
            .as_ref()
            .map(|settings| {
                let component = namespace
                    .context("coordinator affinity requires a runtime namespace")?
                    .component("coordinator_clients")?;
                Ok::<_, anyhow::Error>((settings.clone(), component))
            })
            .transpose()?;
        Ok(Self {
            config,
            client,
            affinity,
            ready: OnceCell::new(),
            cancel: namespace.map_or_else(CancellationToken::new, |namespace| {
                namespace.drt().child_token()
            }),
        })
    }

    fn remotes(&self) -> Result<BTreeMap<String, Url>> {
        let snapshot = self.config.snapshot()?;
        snapshot
            .remotes
            .as_ref()
            .context("remote coordinator requires remotes; restart to switch to local mode")?
            .iter()
            .map(|(name, url)| Ok((name.clone(), Url::parse(url)?)))
            .collect()
    }

    fn pooled(&self, count: usize) -> bool {
        count > 1 || self.affinity.is_some()
    }

    pub(super) async fn start(&self) -> Result<()> {
        anyhow::ensure!(
            !self.cancel.is_cancelled(),
            "coordinator runtime is shut down"
        );
        if self.pooled(self.remotes()?.len()) {
            self.state().await?;
        }
        Ok(())
    }

    async fn state(&self) -> Result<&Arc<PoolState>> {
        self.ready
            .get_or_try_init(|| async {
                let (workers, worker_ids) = watch::channel(Vec::new());
                let affinity = self
                    .affinity
                    .as_ref()
                    .map(|(settings, _)| {
                        AffinityCoordinator::new(Duration::from_secs(settings.ttl_secs))
                    })
                    .transpose()?;
                let state = Arc::new(PoolState {
                    backends: RwLock::new(BTreeMap::new()),
                    workers,
                    affinity,
                });
                let mut interval = tokio::time::interval(POLL_INTERVAL);
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                interval.tick().await;
                refresh(&state, &self.config, &self.client).await;
                if let (Some(affinity), Some((_, component))) = (&state.affinity, &self.affinity) {
                    affinity
                        .enable_replica_sync_for_pool(component, AFFINITY_TOPIC, worker_ids)
                        .await?;
                }
                let weak = Arc::downgrade(&state);
                let config = self.config.clone();
                let client = self.client.clone();
                let cancel = self.cancel.clone();
                tokio::spawn(async move {
                    loop {
                        tokio::select! {
                            _ = cancel.cancelled() => return,
                            _ = interval.tick() => {}
                        }
                        let Some(state) = weak.upgrade() else {
                            return;
                        };
                        tokio::select! {
                            _ = cancel.cancelled() => return,
                            _ = refresh(&state, &config, &client) => {}
                        }
                    }
                });
                Ok(state)
            })
            .await
    }

    pub(super) async fn select(
        &self,
        session: Option<&str>,
        request: protocol::BidRequestV1,
        allowed: &[u64],
        context: &RequestContext,
    ) -> Result<(Url, Option<AffinityAcquire>)> {
        anyhow::ensure!(
            !self.cancel.is_cancelled(),
            "coordinator runtime is shut down"
        );
        request.validate()?;
        let remotes = self.remotes()?;
        if !self.pooled(remotes.len()) {
            return Ok((
                remotes.into_values().next().expect("validated singleton"),
                None,
            ));
        }
        let state = self.state().await?;
        let affinity = match (&state.affinity, session.filter(|id| !id.is_empty())) {
            (Some(store), Some(session)) => Some(
                store
                    .acquire_with_context(
                        &SessionAffinityId::new(session),
                        None,
                        context.inner().as_ref(),
                    )
                    .await?,
            ),
            _ => None,
        };
        // Re-read after awaiting an in-flight first admission: remotes may have changed.
        let mut remotes = self.remotes()?;
        if remotes.len() == 1 {
            // No destination search, but keep the selection so admission still
            // updates affinity and the stream retains its lease.
            return Ok((remotes.into_values().next().expect("singleton"), affinity));
        }
        let affinity_worker_id = affinity
            .as_ref()
            .and_then(AffinityAcquire::target)
            .map(|target| target.worker_id)
            .filter(|id| allowed.is_empty() || allowed.contains(id));
        // Inventory establishes liveness and capacity; prompt-specific bids
        // rank only those candidates. Restrictions apply before affinity too.
        state.retain_live(&mut remotes, allowed);
        {
            let backends = state.backends.read().unwrap();
            if let Some(target) = affinity_worker_id {
                let mut matches = backends.iter().filter(|(name, backend)| {
                    remotes
                        .get(*name)
                        .is_some_and(|url| url == &backend.endpoint)
                        && backend.has_affinity(target, allowed)
                });
                if let Some((_, backend)) = matches.next()
                    && matches.next().is_none()
                {
                    return Ok((backend.endpoint.clone(), affinity));
                }
            }
        }
        let (endpoint, _) = self.best_bid(&remotes, request).await?;
        Ok((endpoint, affinity))
    }

    pub(super) async fn worker_loads(&self) -> Result<Vec<crate::WorkerLoad>> {
        let remotes = self.remotes()?;
        if !self.pooled(remotes.len()) {
            return fetch_worker_loads(
                &self.client,
                remotes
                    .values()
                    .next()
                    .context("no remote coordinators configured")?,
            )
            .await;
        }
        let state = self.state().await?;
        let backends = state.backends.read().unwrap();
        let mut loads = Vec::new();
        for (name, url) in remotes {
            if let Some(backend) = backends
                .get(&name)
                .filter(|backend| backend.is_current(&url))
            {
                loads.extend(backend.loads.iter().cloned());
            }
        }
        Ok(loads)
    }
}

async fn refresh(state: &PoolState, config: &RemoteConfig, client: &Client) {
    let mut backends = BTreeMap::new();
    if let Ok(snapshot) = config.snapshot()
        && let Some(remotes) = snapshot.remotes.as_ref()
    {
        let results = futures::future::join_all(remotes.iter().map(|(name, url)| async move {
            let result = async {
                let endpoint = Url::parse(url)?;
                let loads = fetch_worker_loads(client, &endpoint).await?;
                Ok::<_, anyhow::Error>(Backend {
                    endpoint,
                    loads,
                    observed: Instant::now(),
                })
            }
            .await;
            (name, result)
        }))
        .await;
        for (name, result) in results {
            match result {
                Ok(backend) => {
                    backends.insert(name.clone(), backend);
                }
                Err(_) => tracing::warn!(backend = name, "coordinator worker loads probe failed"),
            }
        }
    }
    let mut ids: Vec<_> = backends
        .values()
        .flat_map(|backend| backend.affinity_workers().map(|load| load.worker_id))
        .collect();
    ids.sort_unstable();
    ids.dedup();
    *state.backends.write().unwrap() = backends;
    state.workers.send_replace(ids);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{WorkerLoad, WorkerMode::*};

    #[test]
    fn affinity_requires_complete_allowed_capacity_and_prefill_membership() {
        let row = |worker_id, disaggregation_mode| WorkerLoad {
            worker_id,
            disaggregation_mode,
            potential_prefill_tokens: 0,
            potential_decode_blocks: 0,
            active_requests: 0,
        };
        for (loads, allowed, expected) in [
            (vec![row(7, Decode)], vec![], false),
            (vec![row(7, Prefill)], vec![], false),
            (vec![row(7, PrefillAndDecode)], vec![], true),
            (vec![row(7, Prefill), row(8, Decode)], vec![], true),
            (vec![row(8, Prefill), row(7, Decode)], vec![], false),
            (vec![row(7, PrefillAndDecode)], vec![9], false),
            (vec![row(7, Prefill), row(8, Decode)], vec![7, 9], false),
            (vec![row(7, Prefill), row(8, Decode)], vec![7, 8], true),
        ] {
            let mut backend = Backend {
                endpoint: Url::parse("http://coordinator/v1/coordinate").unwrap(),
                loads,
                observed: Instant::now(),
            };
            assert_eq!(backend.has_affinity(7, &allowed), expected);
            assert!(backend.is_current(&backend.endpoint));
            assert!(!backend.is_current(&Url::parse("http://replacement/v1/coordinate").unwrap()));
            assert!(
                backend
                    .affinity_workers()
                    .all(|load| load.disaggregation_mode != Decode)
            );
            backend.observed -= MAX_SNAPSHOT_AGE;
            assert!(!backend.is_current(&backend.endpoint));
            assert!(!backend.has_affinity(7, &allowed));
        }
    }
}
