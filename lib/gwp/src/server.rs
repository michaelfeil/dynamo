// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Production GWP control-plane wiring.

use std::sync::Arc;
use std::time::Duration;

use crate::config::{ConfigStore, GwpConfig, SessionBackendConfig};
use crate::core::GwpCore;
use crate::lifecycle::Lifecycle;
use crate::reflector::Reflector;
use crate::session::{AffinityStore, InstrumentedAffinityStore};
use crate::topology::{TopologyController, TopologySnapshot, TopologyStore};

/// Absorb bursts when several endpoint polls resolve while the controller is
/// capturing scheduler load anchors.
const TOPOLOGY_UPDATE_BUFFER: usize = 16;

pub struct BuiltServer {
    pub topology_handle: tokio::task::JoinHandle<()>,
    pub health_task: tokio::task::JoinHandle<()>,
    pub core: GwpCore,
    pub lifecycle: Lifecycle,
    pub authorization_service:
        envoy_types::ext_authz::v3::pb::AuthorizationServer<crate::grpc::GrpcAuthorization>,
    pub lifecycle_service:
        crate::grpc::pb::lifecycle_server::LifecycleServer<crate::grpc::GrpcLifecycle>,
    pub health_service:
        tonic_health::pb::health_server::HealthServer<tonic_health::server::HealthService>,
}

pub async fn build(config: GwpConfig) -> anyhow::Result<BuiltServer> {
    build_inner(ConfigStore::new(config), None).await
}

pub async fn build_with_reload(
    config: GwpConfig,
    path: std::path::PathBuf,
) -> anyhow::Result<BuiltServer> {
    build_inner(ConfigStore::new(config), Some(path)).await
}

async fn build_inner(
    config: ConfigStore,
    reload_path: Option<std::path::PathBuf>,
) -> anyhow::Result<BuiltServer> {
    let initial = config.load();
    let tokenizers = crate::tokens::TokenizerRegistry::from_config(&initial)?;
    let (router, workers_tx) = crate::router::GwpRouter::new(
        initial.routing.block_size,
        initial.routing.approx_indexer_ttl_secs,
    )
    .await?;
    let peer_replicas = router.replica_peer_count().await?;
    let lifecycle = Lifecycle::starting(
        peer_replicas,
        Duration::from_secs(initial.routing.replica_warmup_secs),
    );
    if peer_replicas > 0 {
        tracing::info!(
            peer_replicas,
            warmup_secs = initial.routing.replica_warmup_secs,
            "peer replicas found; holding readiness while replica events converge"
        );
    }

    let topology = TopologyStore::new(TopologySnapshot::from_config(&initial, Default::default())?);
    let (updates_tx, updates_rx) = tokio::sync::mpsc::channel(TOPOLOGY_UPDATE_BUFFER);
    let controller =
        TopologyController::new(topology.clone(), workers_tx, router.observed_load_store())
            .with_metrics(router.metrics().clone())
            .with_router(router.clone())
            .with_lifecycle(lifecycle.clone());
    let mut controller_handle = controller.spawn(updates_rx);
    let mut reflector_handle = Reflector::new(config.clone(), updates_tx)
        .with_metrics(router.metrics().clone())
        .spawn();
    // Either side exiting tears down the whole topology pipeline.
    let topology_handle = tokio::spawn(async move {
        tokio::select! {
            _ = &mut reflector_handle => controller_handle.abort(),
            _ = &mut controller_handle => reflector_handle.abort(),
        }
    });

    let affinity = Arc::new(InstrumentedAffinityStore::new(
        make_affinity(&initial).await?,
        router.metrics().clone(),
    ));
    let core = GwpCore::with_tokenizers(router, affinity, topology, config.clone(), tokenizers);
    core.spawn_janitor();
    if let Some(path) = reload_path {
        config.spawn_file_reloader(path, Duration::from_secs(1));
    }

    let control_state = crate::control::ControlState {
        core: core.clone(),
        lifecycle: lifecycle.clone(),
    };
    let (health_service, health_task) = crate::grpc::health_server(lifecycle.clone()).await;
    Ok(BuiltServer {
        topology_handle,
        health_task,
        core,
        lifecycle,
        authorization_service: crate::grpc::authorization_server(control_state.clone()),
        lifecycle_service: crate::grpc::lifecycle_server(control_state),
        health_service,
    })
}

async fn make_affinity(config: &GwpConfig) -> anyhow::Result<Arc<dyn AffinityStore>> {
    match config.session.resolved_backend()? {
        SessionBackendConfig::Etcd { endpoints } => {
            #[cfg(feature = "etcd")]
            {
                let store = crate::session::EtcdAffinityStore::connect(endpoints.clone())
                    .await
                    .map_err(|error| {
                        anyhow::anyhow!(
                            "failed to initialize configured etcd session affinity at \
                             {endpoints:?}: {error}"
                        )
                    })?;
                tracing::info!(?endpoints, "session affinity: etcd");
                Ok(Arc::new(store))
            }
            #[cfg(not(feature = "etcd"))]
            {
                let _ = endpoints;
                anyhow::bail!(
                    "configured etcd session affinity requires the dynamo-gwp 'etcd' feature"
                )
            }
        }
        SessionBackendConfig::Redis { url, key_prefix } => {
            #[cfg(feature = "redis")]
            {
                let store = crate::session::RedisAffinityStore::connect(&url, key_prefix)
                    .await
                    .map_err(|error| {
                        anyhow::anyhow!("failed to initialize configured Redis affinity: {error}")
                    })?;
                tracing::info!("session affinity: redis");
                Ok(Arc::new(store))
            }
            #[cfg(not(feature = "redis"))]
            {
                let _ = (url, key_prefix);
                anyhow::bail!(
                    "configured Redis session affinity requires the dynamo-gwp 'redis' feature"
                )
            }
        }
    }
}
