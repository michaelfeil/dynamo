// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{
    DeniedGenerationRequest, DeniedRequest, DisaggregationStrategy, GenerationCoordinator,
    GenerationCoordinatorClient, GenerationCoordinatorService, GenerationOptions,
    GenerationOutcome, GenerationRequest, JsonPushRouter, PrefillMarkTiming,
    RemoteGenerationCoordinator, RequestContext, RouterWorkerCoordinator,
};
use anyhow::{Context, Result, ensure};
use baseten_configmap::{ConfigReader, GenerationCoordinatorConfig};
use dynamo_runtime::pipeline::network::egress::push_router::RouterMode;
use dynamo_runtime::{DistributedRuntime, component::Endpoint};
use std::{collections::BTreeMap, net::SocketAddr, sync::Arc};
use tokio::sync::OnceCell;

/// A connected router or an endpoint to connect lazily.
pub enum CoordinatorClient {
    Connected(JsonPushRouter),
    Endpoint(Endpoint),
}

impl CoordinatorClient {
    async fn connect(&self) -> anyhow::Result<JsonPushRouter> {
        match self {
            Self::Connected(client) => Ok(client.clone()),
            Self::Endpoint(endpoint) => {
                let client = endpoint.client().await?;
                JsonPushRouter::from_client(client, RouterMode::RoundRobin).await
            }
        }
    }
}

/// Local router/worker topology, independent of HTTP configuration.
pub struct LocalCoordinatorOptions {
    pub primary_worker: CoordinatorClient,
    pub primary_router: CoordinatorClient,
    pub next_worker: Option<CoordinatorClient>,
    pub next_router: Option<CoordinatorClient>,
    pub strategy: DisaggregationStrategy,
    pub mark_timing: PrefillMarkTiming,
    pub block_size: u32,
    pub machine_id: u64,
}

struct LocalCoordinator {
    options: LocalCoordinatorOptions,
    ready: OnceCell<GenerationCoordinator>,
}

impl LocalCoordinator {
    pub async fn start(&self) -> anyhow::Result<&GenerationCoordinator> {
        self.ready
            .get_or_try_init(|| async {
                let primary = Arc::new(RouterWorkerCoordinator::from_push_routers(
                    self.options.primary_router.connect().await?,
                    self.options.primary_worker.connect().await?,
                    self.options.block_size,
                )?);
                let next = match (&self.options.next_router, &self.options.next_worker) {
                    (Some(router), Some(worker)) => {
                        Some(Arc::new(RouterWorkerCoordinator::from_push_routers(
                            router.connect().await?,
                            worker.connect().await?,
                            self.options.block_size,
                        )?))
                    }
                    _ => None,
                };
                GenerationCoordinator::new(
                    primary,
                    next,
                    self.options.strategy,
                    self.options.mark_timing,
                    self.options.machine_id,
                )
            })
            .await
    }
}

impl GenerationCoordinatorClient for LocalCoordinator {
    fn bid(
        &self,
        request: crate::protocol::BidRequestV1,
    ) -> futures::future::BoxFuture<'_, Result<crate::protocol::BidResponseV1>> {
        Box::pin(async { self.start().await?.bid(request).await })
    }
    fn worker_loads(&self) -> futures::future::BoxFuture<'_, Result<Vec<crate::WorkerLoad>>> {
        Box::pin(async { self.start().await?.worker_loads().await })
    }
    fn generate(
        &self,
        context: RequestContext,
        request: GenerationRequest,
        options: GenerationOptions,
    ) -> futures::future::BoxFuture<'_, anyhow::Result<GenerationOutcome>> {
        Box::pin(async move {
            let inner = context.inner();
            let cancelled = || {
                Ok(GenerationOutcome::Denied(DeniedGenerationRequest {
                    denied: DeniedRequest::Cancelled(),
                    admission: None,
                }))
            };
            let coordinator = tokio::select! {
                biased;
                _ = inner.stopped() => return cancelled(),
                _ = inner.killed() => return cancelled(),
                result = self.start() => result?,
            };
            coordinator.generate(context, request, options).await
        })
    }
}

/// Startup-selected client and optional runtime-owned HTTP listener.
pub struct GenerationCoordinatorRuntime {
    client: Arc<dyn GenerationCoordinatorClient>,
    local: Option<Arc<LocalCoordinator>>,
    listener: Option<Listener>,
}

struct Listener {
    runtime: DistributedRuntime,
    address: Option<SocketAddr>,
    strategy: DisaggregationStrategy,
    started: OnceCell<Option<String>>,
}

impl GenerationCoordinatorRuntime {
    pub fn bid(
        &self,
        request: crate::protocol::BidRequestV1,
    ) -> futures::future::BoxFuture<'_, Result<crate::protocol::BidResponseV1>> {
        self.client.bid(request)
    }
    pub fn worker_loads(&self) -> futures::future::BoxFuture<'_, Result<Vec<crate::WorkerLoad>>> {
        self.client.worker_loads()
    }
    pub fn new(
        runtime: DistributedRuntime,
        options: LocalCoordinatorOptions,
        config: ConfigReader,
        namespace: Option<String>,
        is_client_force: Option<bool>,
    ) -> Result<Self> {
        let default_remote = (is_client_force != Some(false))
            .then(|| std::env::var("DYNAMO_GENERATION_COORDINATOR_URL").ok())
            .flatten();
        Self::with_default_remote(
            runtime,
            options,
            config,
            namespace,
            is_client_force,
            default_remote,
        )
    }

    fn with_default_remote(
        runtime: DistributedRuntime,
        options: LocalCoordinatorOptions,
        config: ConfigReader,
        namespace: Option<String>,
        is_client_force: Option<bool>,
        default_remote: Option<String>,
    ) -> Result<Self> {
        ensure!(options.block_size > 0, "kv_block_size must be positive");
        ensure!(
            options.strategy != DisaggregationStrategy::PrefillFirst
                || (options.next_worker.is_some() && options.next_router.is_some()),
            "next_worker_client and next_router_client are required for disaggregated generation"
        );
        let snapshot = config.snapshot();
        let settings = &snapshot.generation_coordinator;
        let listener = Some(Listener {
            runtime: runtime.clone(),
            address: settings.listen_address(),
            strategy: options.strategy,
            started: OnceCell::new(),
        });
        let (client, local): (Arc<dyn GenerationCoordinatorClient>, _) =
            if is_client_force.unwrap_or(settings.remotes.is_some()) {
                ensure!(
                    settings.remotes.is_some() || default_remote.is_some(),
                    "remote coordinator mode requires remotes"
                );
                let namespace = runtime
                    .namespace(namespace.context(
                        "namespace is required for configured remote coordinator mode",
                    )?)?;
                (
                    Arc::new(RemoteGenerationCoordinator::from_runtime_config(
                        config,
                        Some(&namespace),
                        default_remote,
                    )?),
                    None,
                )
            } else {
                let local = Arc::new(LocalCoordinator {
                    options,
                    ready: OnceCell::new(),
                });
                (local.clone(), Some(local))
            };
        Ok(Self {
            client,
            local,
            listener,
        })
    }

    pub fn remote(backends: BTreeMap<String, String>) -> Result<Self> {
        let settings = GenerationCoordinatorConfig {
            remotes: Some(backends),
            ..Default::default()
        };
        settings.validate()?;
        Ok(Self {
            client: Arc::new(RemoteGenerationCoordinator::from_config(
                ConfigReader::in_memory(baseten_configmap::UnifiedConfig {
                    generation_coordinator: settings,
                    ..Default::default()
                }),
            )?),
            local: None,
            listener: None,
        })
    }

    pub fn generate(
        &self,
        context: RequestContext,
        request: GenerationRequest,
        options: GenerationOptions,
    ) -> futures::future::BoxFuture<'_, Result<GenerationOutcome>> {
        self.client.generate(context, request, options)
    }

    pub fn is_client(&self) -> bool {
        self.local.is_none()
    }

    pub fn is_server(&self) -> bool {
        self.listener.as_ref().is_some_and(|listener| {
            listener.started.get().is_some_and(Option::is_some)
                && !listener.runtime.child_token().is_cancelled()
        })
    }

    pub async fn start(&self) -> Result<Option<String>> {
        let Some(listener) = &self.listener else {
            self.client.start().await?;
            return Ok(None);
        };
        let shutdown = listener.runtime.child_token();
        ensure!(!shutdown.is_cancelled(), "runtime is shut down");
        listener
            .started
            .get_or_try_init(|| async {
                // Keep transports alive until HTTP requests finish draining.
                let guard = listener.runtime.register_graceful_task();
                if let Some(local) = &self.local {
                    tokio::select! {
                        biased;
                        _ = shutdown.cancelled() => anyhow::bail!("runtime is shut down"),
                        result = local.start() => { result?; }
                    }
                } else {
                    tokio::select! {
                        biased;
                        _ = shutdown.cancelled() => anyhow::bail!("runtime is shut down"),
                        result = self.client.start() => { result?; }
                    }
                }
                let Some(address) = listener.address else {
                    return Ok(None);
                };
                let server = Arc::new(GenerationCoordinatorService::new(
                    self.client.clone(),
                    listener.strategy,
                ))
                .start(address)
                .await?;
                let url = server.endpoint_url();
                tokio::spawn(async move {
                    let _guard = guard;
                    shutdown.cancelled().await;
                    if let Err(error) = server.shutdown().await {
                        tracing::error!(%error, "generation coordinator HTTP shutdown failed");
                    }
                });
                Ok(Some(url))
            })
            .await
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use baseten_configmap::UnifiedConfig;
    use dynamo_runtime::{Runtime, distributed::DistributedConfig};

    #[tokio::test]
    async fn startup_fixes_mode_and_listener_until_runtime_shutdown() {
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            let runtime = DistributedRuntime::new(
                Runtime::from_current().unwrap(),
                DistributedConfig::process_local(),
            )
            .await
            .unwrap();
            let endpoint = runtime
                .namespace("coordinator-test")
                .unwrap()
                .component("worker")
                .unwrap()
                .endpoint("generate");
            let options = || LocalCoordinatorOptions {
                primary_worker: CoordinatorClient::Endpoint(endpoint.clone()),
                primary_router: CoordinatorClient::Endpoint(endpoint.clone()),
                next_worker: None,
                next_router: None,
                strategy: DisaggregationStrategy::Aggregated,
                mark_timing: PrefillMarkTiming::AfterPrefillCompute,
                block_size: 32,
                machine_id: 1,
            };
            let reader = ConfigReader::in_memory(UnifiedConfig::default());
            let new = |force, namespace, default_remote| {
                GenerationCoordinatorRuntime::with_default_remote(
                    runtime.clone(),
                    options(),
                    reader.clone(),
                    namespace,
                    force,
                    default_remote,
                )
            };
            let local = new(None, None, Some("http://default/v1/coordinate".into())).unwrap();
            assert!(!local.is_client());
            assert!(!new(None, None, Some("invalid".into())).unwrap().is_client());
            assert!(
                !new(Some(false), None, Some("invalid".into()))
                    .unwrap()
                    .is_client()
            );
            let namespace = || Some("coordinator-test".into());
            assert!(new(Some(true), namespace(), Some("invalid".into())).is_err());
            assert!(
                new(
                    Some(true),
                    namespace(),
                    Some("http://default/v1/coordinate".into())
                )
                .unwrap()
                .is_client()
            );
            assert_eq!(
                new(Some(true), namespace(), None)
                    .err()
                    .unwrap()
                    .to_string(),
                "remote coordinator mode requires remotes"
            );
            let mut config = UnifiedConfig::default();
            config.generation_coordinator.port = Some(0);
            config.generation_coordinator.host = "127.0.0.1".parse().unwrap();
            config.generation_coordinator.remotes = Some(BTreeMap::from([(
                "default".into(),
                "http://127.0.0.1:1/v1/coordinate".into(),
            )]));
            config.generation_coordinator.affinity =
                Some(baseten_configmap::CoordinatorAffinityConfig { ttl_secs: 60 });
            reader.replace(config);
            assert!(new(None, None, None).is_err());
            assert!(
                new(None, namespace(), Some("invalid".into()))
                    .unwrap()
                    .is_client()
            );
            let forced_local = new(Some(false), None, None).unwrap();
            assert!(!forced_local.is_client());
            assert!(forced_local.start().await.unwrap().is_some());
            assert!(forced_local.is_server());
            let remote = new(Some(true), namespace(), None).unwrap();
            assert!(!local.is_client());
            assert_eq!(local.start().await.unwrap(), None);
            assert!(!local.is_server());
            assert!(remote.is_client());
            assert!(!remote.is_server());
            let url = remote.start().await.unwrap().unwrap();
            assert!(remote.is_server());
            reader.replace(UnifiedConfig::default());
            assert_eq!(remote.start().await.unwrap().as_deref(), Some(url.as_str()));
            assert!(remote.is_client());
            assert!(!forced_local.is_client());
            let health = url.replace("/v1/coordinate", "/health");
            assert!(reqwest::get(health).await.unwrap().status().is_success());
            runtime.shutdown();
            runtime.child_token().cancelled().await;
            assert!(!remote.is_server());
            assert!(remote.start().await.is_err());
        })
        .await
        .expect("coordinator startup timed out");
    }
}
