use super::*;
use dynamo_runtime::component::Endpoint;

pub(super) enum CoordinatorClient {
    Connected(dynamo_b10_client::JsonPushRouter),
    Endpoint(Endpoint),
}

impl CoordinatorClient {
    pub(super) fn parse(
        value: &Bound<'_, PyAny>,
        runtime: Option<&crate::DistributedRuntime>,
    ) -> PyResult<Self> {
        if let Ok(client) = value.extract::<Client>() {
            return Ok(Self::Connected(client.router));
        }
        let path = value.extract::<String>().map_err(|_| {
            PyTypeError::new_err("coordinator clients must be Client objects or endpoint strings")
        })?;
        let runtime = runtime
            .ok_or_else(|| PyValueError::new_err("runtime is required for endpoint strings"))?;
        Ok(Self::Endpoint(runtime.endpoint(path)?.inner))
    }

    async fn connect(&self) -> anyhow::Result<dynamo_b10_client::JsonPushRouter> {
        match self {
            Self::Connected(client) => Ok(client.clone()),
            Self::Endpoint(endpoint) => {
                let client = endpoint.client().await?;
                dynamo_b10_client::JsonPushRouter::from_client(
                    client,
                    crate::RsRouterMode::RoundRobin,
                )
                .await
            }
        }
    }
}

pub(super) struct CoordinatorStartup {
    pub _runtime: Option<Arc<dynamo_runtime::DistributedRuntime>>,
    pub primary_worker: CoordinatorClient,
    pub primary_router: CoordinatorClient,
    pub next_worker: Option<CoordinatorClient>,
    pub next_router: Option<CoordinatorClient>,
    pub strategy: CoreDisaggregationStrategy,
    pub mark_timing: CorePrefillMarkTiming,
    pub block_size: u32,
    pub machine_id: u64,
    pub ready: tokio::sync::OnceCell<CoreGenerationCoordinator>,
}

impl CoordinatorStartup {
    pub async fn start(&self) -> anyhow::Result<&CoreGenerationCoordinator> {
        self.ready
            .get_or_try_init(|| async {
                let primary = Arc::new(CoreRouterWorkerCoordinator::from_push_routers(
                    self.primary_router.connect().await?,
                    self.primary_worker.connect().await?,
                    self.block_size,
                )?);
                let next = match (&self.next_router, &self.next_worker) {
                    (Some(router), Some(worker)) => {
                        Some(Arc::new(CoreRouterWorkerCoordinator::from_push_routers(
                            router.connect().await?,
                            worker.connect().await?,
                            self.block_size,
                        )?))
                    }
                    _ => None,
                };
                CoreGenerationCoordinator::new(
                    primary,
                    next,
                    self.strategy,
                    self.mark_timing,
                    self.machine_id,
                )
            })
            .await
    }
}

impl CoreGenerationCoordinatorClient for CoordinatorStartup {
    fn generate(
        &self,
        context: RequestContext,
        request: GenerationRequest,
        options: GenerationOptions,
    ) -> futures::future::BoxFuture<'_, anyhow::Result<CoreGenerationOutcome>> {
        Box::pin(async move {
            let inner = context.inner();
            let cancelled = || {
                Ok(CoreGenerationOutcome::Denied(
                    dynamo_b10_client::DeniedGenerationRequest {
                        denied: dynamo_b10_client::DeniedRequest::Cancelled(),
                        admission: None,
                    },
                ))
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
