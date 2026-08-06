// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Global Workload Plane entrypoint. Requires the `server` feature.
//!
//! Serves Envoy scheduling, lifecycle notifications, and standard health over
//! gRPC on `DYN_GWP_GRPC_PORT` (default 8091).

use dynamo_gwp::GwpConfig;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dynamo_runtime::logging::init();

    let config = GwpConfig::load()?;
    tracing::info!(
        endpoints = config.endpoints.len(),
        routes = config.routes.len(),
        "starting Global Workload Plane"
    );

    let shutdown_grace = Duration::from_secs(config.routing.shutdown_grace_secs);
    let built = dynamo_gwp::server::build_with_reload(config, GwpConfig::config_path()).await?;

    let grpc_port: u16 = std::env::var("DYN_GWP_GRPC_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8091);
    let grpc_address = ([0, 0, 0, 0], grpc_port).into();
    tracing::info!(grpc_port, "GWP gRPC control plane listening");

    let server_shutdown = CancellationToken::new();
    let shutdown_task = tokio::spawn(shutdown_coordinator(
        built.lifecycle.clone(),
        built.core.clone(),
        shutdown_grace,
        server_shutdown.clone(),
    ));
    let grpc_graceful = server_shutdown.clone();
    let grpc_served = tonic::transport::Server::builder()
        .add_service(built.authorization_service)
        .add_service(built.lifecycle_service)
        .add_service(built.health_service)
        .serve_with_shutdown(grpc_address, async move { grpc_graceful.cancelled().await });
    tokio::pin!(grpc_served);
    let mut topology_handle = built.topology_handle;

    let result = tokio::select! {
        grpc_served = &mut grpc_served => {
            server_shutdown.cancel();
            grpc_served.map_err(anyhow::Error::from)
        },
        _ = &mut topology_handle => {
            tracing::error!("topology pipeline exited unexpectedly");
            built.lifecycle.begin_draining();
            built.core.force_finish_all("topology_exit").await;
            server_shutdown.cancel();
            (&mut grpc_served).await.map_err(anyhow::Error::from)?;
            anyhow::bail!("topology pipeline exited unexpectedly")
        },
    };

    shutdown_task.abort();
    built.health_task.abort();
    built.lifecycle.begin_draining();
    built.core.force_finish_all("server_exit").await;
    built.core.router.shutdown();
    result
}

async fn shutdown_coordinator(
    lifecycle: dynamo_gwp::lifecycle::Lifecycle,
    core: dynamo_gwp::core::GwpCore,
    grace: Duration,
    server_shutdown: CancellationToken,
) {
    if let Err(error) = shutdown_signal().await {
        tracing::error!(%error, "failed to install shutdown signal handler");
        return;
    }

    lifecycle.begin_draining();
    tracing::info!(
        inflight = core.inflight_len(),
        grace_secs = grace.as_secs(),
        "shutdown signal received; readiness revoked and new schedules disabled"
    );

    let deadline = tokio::time::Instant::now() + grace;
    while core.inflight_len() > 0 && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let remaining = core.inflight_len();
    if remaining > 0 {
        tracing::warn!(
            remaining,
            "graceful shutdown deadline reached; force-freeing scheduler bookings"
        );
        core.force_finish_all("shutdown_timeout").await;
    }
    server_shutdown.cancel();
}

async fn shutdown_signal() -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        tokio::select! {
            result = tokio::signal::ctrl_c() => result?,
            _ = terminate.recv() => {},
        }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await?;

    Ok(())
}
