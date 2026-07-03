// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Python entrypoint for the B10 KV router.
//!
//! Wraps upstream's `KvRouter` with two B10-specific affordances:
//! * the `B10WorkerSelector` (hot-reloadable cost function), or upstream's
//!   `DefaultWorkerSelector` for parity testing
//! * a max-active-routers gate that delays bringing the router up until
//!   fewer than N peers are already serving traffic in the same component
//!
//! The `algo_selector="Python"` (pluggable Python selector) and
//! `dp_strict_rank` plumbing from the v1.0.0 fork have been dropped:
//! production deployments only use `B10`, and upstream's selector trait
//! handles DP fan-out internally.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use anyhow::Result;
use pyo3::prelude::*;
use rand::Rng;

use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::indexer::KvIndexerMetrics;
use dynamo_kv_router::selector::WorkerSelector;
use dynamo_llm::b10_health::{register_runtime_cancel_token, set_health};
use dynamo_llm::kv_router::{
    KvRouter,
    b10_worker_selector::B10WorkerSelector,
    b10hotreloadablecm::{get_router_active_replicas, set_log_no_changes, validate_config},
    metrics::{RouterRequestMetrics, register_global_metrics_with_component},
    scheduler::DefaultWorkerSelector,
};
use dynamo_llm::local_model::runtime_config::ModelRuntimeConfig;
use dynamo_runtime::{
    DistributedRuntime, Runtime, Worker,
    component::Component,
    config::{self, environment_names::logging::otlp as env_otlp},
    metrics::MetricsHierarchy,
    pipeline::network::Ingress,
};

use super::entrypoint::KvRouterConfig as PyKvRouterConfig;

const MAX_WAIT_SECONDS: u64 = 10 * 365 * 24 * 3600; // 10 years

enum AlgoSelector {
    Default,
    B10,
}

struct Args {
    namespace: String,
    component_to_route: String,
    router_component_name: String,
    // maximum time to wait before forcing activation
    max_wait_seconds: u64,
    block_size: u32,
    kv_router_config: KvRouterConfig,
    kv_router_metrics_port: u16,
    algo_selector: AlgoSelector,
}

/// Simple HTTP server for serving /metrics for a component.
struct ComponentMetricsServer {
    component: Component,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

impl ComponentMetricsServer {
    fn new(component: Component) -> Self {
        Self {
            component,
            shutdown_tx: None,
        }
    }

    async fn start(&mut self, port: u16) -> Result<()> {
        use axum::{Router, routing::get};
        use std::net::SocketAddr;
        use tokio::net::TcpListener;

        let axum_router = Router::new().route(
            "/metrics",
            get({
                let component = self.component.clone();
                move || async move {
                    match component.metrics().prometheus_expfmt() {
                        Ok(metrics) => metrics,
                        Err(e) => {
                            tracing::error!("Failed to get metrics from runtime: {}", e);
                            format!("# ERROR: Failed to get metrics: {}", e)
                        }
                    }
                }
            }),
        );

        let socket_addr = SocketAddr::new(
            std::net::IpAddr::V4(std::net::Ipv4Addr::new(0, 0, 0, 0)),
            port,
        );

        let (tx, rx) = tokio::sync::oneshot::channel();
        self.shutdown_tx = Some(tx);

        let tcp_listener = TcpListener::bind(socket_addr).await.map_err(|e| {
            anyhow::anyhow!(
                "Failed to bind to address {}: {}. The port may be in use.",
                socket_addr,
                e
            )
        })?;

        tracing::info!("Starting Runtime metrics server on {}", socket_addr);

        tokio::spawn(async move {
            axum::serve(tcp_listener, axum_router)
                .with_graceful_shutdown(async {
                    rx.await.ok();
                })
                .await
                .map_err(|e| {
                    tracing::error!("Runtime metrics server error: {}", e);
                })
                .unwrap();
        });

        Ok(())
    }

    fn stop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }
}

impl Drop for ComponentMetricsServer {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Guard that periodically sets health to true and sets it to false on drop.
struct TaskAbortGuard {
    task: tokio::task::JoinHandle<()>,
}

impl Drop for TaskAbortGuard {
    fn drop(&mut self) {
        set_health(false, "router task aborted or dropped", None);
        self.task.abort();
    }
}

async fn run_with_selector<Sel>(
    runtime: DistributedRuntime,
    component_worker: Component,
    component_router: Component,
    selector: Sel,
    args: Args,
) -> Result<()>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + Sync + 'static,
{
    register_runtime_cancel_token(runtime.primary_token());

    // Memoize metrics with the router component so they appear on the router's
    // metrics port (ComponentMetricsServer). Both use OnceLock; whoever calls
    // first wins, so register on `component_router` before `KvRouter::new()`
    // registers them on the worker component.
    let _ = KvIndexerMetrics::from_component(&component_router);
    RouterRequestMetrics::from_component(&component_router);
    register_global_metrics_with_component(&component_router);

    tracing::info!(
        "KvRouter starting with configuration: {:?}",
        args.kv_router_config
    );

    let endpoint = component_worker.endpoint("generate");
    let client = endpoint.client().await?;

    let start_serving_complete: Arc<(AtomicBool, AtomicBool)> =
        Arc::new((AtomicBool::new(false), AtomicBool::new(false)));

    // Start the health heartbeat before blocking on worker discovery.
    // KvRouter::new blocks until at least one worker registers, which can take
    // 10+ minutes for large models. The heartbeat signals that the router process
    // is alive so the deployment health probe doesn't time out.
    let _abort_guard = {
        let start_serving_complete = start_serving_complete.clone();
        let task = tokio::spawn(async move {
            loop {
                let started = start_serving_complete.0.load(Ordering::SeqCst);
                let serving = start_serving_complete.1.load(Ordering::SeqCst);
                // Signal K8s readiness as soon as startup is complete, regardless of whether
                // this replica has been elected to serve yet. A replica may remain in standby
                // for an extended period; withholding readiness would block rolling promotions.
                if started || serving {
                    set_health(true, "router health heartbeat", None);
                }
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        });
        TaskAbortGuard { task }
    };

    let workers_with_configs = dynamo_llm::discovery::runtime_config_watch(&endpoint).await?;

    let component_name = component_worker.name().to_lowercase();
    let is_prefill =
        component_name.contains("prefill") || !args.kv_router_config.router_track_active_blocks;
    let worker_type = if is_prefill {
        dynamo_llm::discovery::WORKER_TYPE_PREFILL
    } else {
        dynamo_llm::discovery::WORKER_TYPE_DECODE
    };
    let disable_snapshots_in_primary = args.kv_router_config.router_disable_snapshots_in_primary;

    // Start the metrics server using the router component's metrics registry, before router start.
    let mut metrics_server = ComponentMetricsServer::new(component_router.clone());
    metrics_server.start(args.kv_router_metrics_port).await?;

    let kv_router = Arc::new(
        KvRouter::new(
            endpoint,
            client,
            workers_with_configs,
            args.block_size,
            selector,
            Some(args.kv_router_config),
            None,
            worker_type,
            None,
            false,
            None,
        )
        .await?,
    );
    let router = Ingress::for_engine(kv_router.clone())?;

    // now startup is complete, but not yet serving
    start_serving_complete.0.store(true, Ordering::SeqCst);

    // Wait until we have less than router_active_replicas active.
    // This value is hot-reloaded from the router config map, so routers can
    // become active as the config changes without restarting the process.
    let start_time = std::time::Instant::now();
    loop {
        let jitter = rand::rng().random_range(0..100);
        tokio::time::sleep(std::time::Duration::from_millis(jitter)).await;
        let active_workers: Option<usize> = get_active_components(&component_worker).await;
        if let Some(active_workers_some) = active_workers
            && active_workers_some == 0
        {
            tracing::info!(
                "No active worker instances found, proceeding to start router immediately"
            );
            let jitter = rand::rng().random_range(0..1000);
            tokio::time::sleep(std::time::Duration::from_millis(jitter)).await;
            if get_active_components(&component_router).await == Some(0) {
                tracing::info!(
                    "No active router instances found, proceeding to start router immediately"
                );
                break;
            }
        }
        let sleep_time = {
            let elapsed = start_time.elapsed().as_secs();
            // Cap max_wait_seconds at 4 hours for the heuristic so very large
            // values (e.g. 10y) don't produce absurd back-off spans.
            let capped_max_wait = std::cmp::min(args.max_wait_seconds, 4 * 3600);
            let max_wait_seconds_half = std::cmp::max(capped_max_wait / 2, 1);
            let wait_time_secs = if elapsed >= max_wait_seconds_half {
                0.5
            } else {
                7.0 - (6.0 * elapsed as f64 / max_wait_seconds_half as f64)
            };
            (wait_time_secs.max(0.5) * 1000.0) as u64
        };
        tokio::time::sleep(std::time::Duration::from_millis(sleep_time)).await;

        let router_active_replicas = get_router_active_replicas();
        let active_routers = match get_active_components(&component_router).await {
            Some(count) => count,
            None => router_active_replicas + 1, // force wait
        };

        if active_routers < router_active_replicas {
            tracing::info!(
                "Current active routers: {}/{}, proceeding",
                active_routers,
                router_active_replicas
            );
            break;
        }
        if start_time.elapsed().as_secs() > args.max_wait_seconds {
            tracing::warn!(
                "Waited for more than {} seconds, proceeding anyway",
                args.max_wait_seconds
            );
            break;
        }

        tracing::info!(
            "Maximum active routers reached ({}/{}), waiting to become active... (waiting since {}s)",
            active_routers,
            router_active_replicas,
            start_time.elapsed().as_secs()
        );
    }

    if disable_snapshots_in_primary {
        kv_router.disable_snapshots();
    }

    tracing::info!("Starting router service...");
    // now serving
    start_serving_complete.1.store(true, Ordering::SeqCst);

    component_router
        .endpoint("generate")
        .endpoint_builder()
        .handler(router)
        .start()
        .await?;

    set_health(false, "router shutting down gracefully", None);

    Ok(())
}

async fn app(runtime: Runtime, args: Args) -> Result<()> {
    let runtime = DistributedRuntime::from_settings(runtime)
        .await
        .map_err(|e| {
            tracing::error!("Failed to connect to distributed runtime: {}", e);
            e
        })?;

    let component_worker = runtime
        .namespace(&args.namespace)?
        .component(&args.component_to_route)?;

    let component_router = runtime
        .namespace(&args.namespace)?
        .component(&args.router_component_name)?;

    match args.algo_selector {
        AlgoSelector::Default => {
            let worker_type = if component_worker.name().to_lowercase().contains("prefill")
                || !args.kv_router_config.router_track_active_blocks
            {
                dynamo_llm::discovery::WORKER_TYPE_PREFILL
            } else {
                dynamo_llm::discovery::WORKER_TYPE_DECODE
            };
            let selector =
                DefaultWorkerSelector::new(Some(args.kv_router_config.clone()), worker_type);
            run_with_selector(runtime, component_worker, component_router, selector, args).await
        }
        AlgoSelector::B10 => {
            let selector = B10WorkerSelector::new();
            run_with_selector(runtime, component_worker, component_router, selector, args).await
        }
    }
}

async fn get_active_components(component: &Component) -> Option<usize> {
    const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
    match tokio::time::timeout(TIMEOUT, component.list_instances()).await {
        Ok(Ok(instances)) => Some(instances.len()),
        Ok(Err(e)) => {
            tracing::warn!(
                "component.list_instances failed for component '{}': {}",
                component.name(),
                e
            );
            None
        }
        Err(e) => {
            tracing::warn!(
                "component.list_instances timed out for component '{}': '{}'",
                component.name(),
                e
            );
            None
        }
    }
}

#[pyfunction]
#[pyo3(signature = (
    namespace="dynamo".to_string(),
    component_to_route="TensorRTLLMWorker".to_string(),
    block_size=32,
    router_component_name="Router".to_string(),
    max_active_routers=None,
    max_wait_seconds=MAX_WAIT_SECONDS,
    kv_router_config=None,
    kv_router_metrics_port=9091,
    algo_selector="B10".to_string(),
))]
#[allow(clippy::too_many_arguments)]
pub fn start_router(
    py: Python<'_>,
    namespace: String,
    component_to_route: String,
    block_size: u32,
    router_component_name: String,
    max_active_routers: Option<usize>,
    max_wait_seconds: u64,
    kv_router_config: Option<PyKvRouterConfig>,
    kv_router_metrics_port: u16,
    algo_selector: String,
) -> PyResult<()> {
    if max_active_routers.is_some() {
        let warning = "start_router(max_active_routers=...) is deprecated and has no effect; set router_active_replicas in the router config map instead";
        eprintln!("Warning: {}", warning);
        tracing::warn!("{}", warning);
    }

    py.allow_threads(|| {
        let kv_router_config = kv_router_config
            .map(|config| config.inner())
            .unwrap_or_default();

        let algo_selector = match algo_selector.as_str() {
            "Default" => AlgoSelector::Default,
            "B10" => AlgoSelector::B10,
            other => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid algo_selector: {} (expected 'Default' or 'B10')",
                    other
                )));
            }
        };

        let args = Args {
            namespace,
            component_to_route,
            router_component_name,
            block_size,
            max_wait_seconds,
            kv_router_config,
            kv_router_metrics_port,
            algo_selector,
        };

        set_log_no_changes(true);
        if !validate_config() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid B10 routing configuration",
            ));
        }

        let worker = Worker::from_settings()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // `_core` defers Rust logging initialization when OTEL export is
        // enabled because the OTLP batch exporter needs a Tokio runtime.  This
        // router path creates a Rust Worker directly, so initialize logging now
        // that the Worker runtime exists and before router startup emits Rust
        // tracing events.
        if config::env_is_truthy(env_otlp::OTEL_EXPORT_ENABLED) {
            worker.runtime().secondary().block_on(async {
                dynamo_runtime::logging::init();
            });
            if !tracing::dispatcher::has_been_set() {
                eprintln!(
                    "ERROR: OTEL_EXPORT_ENABLED=1 but no tracing subscriber \
                     installed before `start_router` startup. Router telemetry \
                     (spans, logs) will be SILENT."
                );
            }
        }

        worker
            .execute(move |rt| async move { app(rt, args).await })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    })
}
