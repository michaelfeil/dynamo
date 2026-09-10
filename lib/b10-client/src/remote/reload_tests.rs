// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{
    DisaggregationStrategy, GeneratedRequest, GenerationAdmission, GenerationCoordinatorService,
    RouterRequestNew, RunningGenerationCoordinatorService,
};
use baseten_configmap::UnifiedConfig;
use dynamo_runtime::pipeline::{AsyncEngineContext, ResponseStream, context::Controller};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::StreamExt;
use rmpv::Value;
use std::collections::BTreeMap;
use std::time::Duration;
use tokio::sync::Notify;

#[derive(Default)]
struct Backend {
    id: u64,
    release: Arc<Notify>,
    closed: Arc<Notify>,
}

struct Closed(Arc<Notify>);
impl Drop for Closed {
    fn drop(&mut self) {
        self.0.notify_one();
    }
}

impl GenerationCoordinatorClient for Backend {
    fn generate(
        &self,
        context: RequestContext,
        _: GenerationRequest,
        _: GenerationOptions,
    ) -> BoxFuture<'_, Result<GenerationOutcome>> {
        Box::pin(async move {
            let release = self.release.clone();
            let closed = Closed(self.closed.clone());
            let id = self.id;
            let output = async_stream::stream! {
                let _closed = closed;
                release.notified().await;
                yield Annotated::from_data(Value::from(id));
            };
            Ok(GenerationOutcome::Connected(GeneratedRequest {
                admission: GenerationAdmission {
                    prefill_worker_id: id,
                    ..Default::default()
                },
                stream: ResponseStream::new(Box::pin(output), context.inner()),
            }))
        })
    }
}

async fn request(
    client: &dyn GenerationCoordinatorClient,
) -> (Arc<dyn AsyncEngineContext>, GeneratedRequest) {
    let context: Arc<dyn AsyncEngineContext> = Arc::new(Controller::new("reload-test".into()));
    let result = client
        .generate(
            RequestContext::new(context.clone(), None, BTreeMap::new()),
            GenerationRequest {
                routing_request: RouterRequestNew {
                    tokens: vec![1, 2, 3],
                    ..Default::default()
                },
                primary_worker_request: Value::Map(vec![
                    ("model".into(), "test".into()),
                    ("sampling_params".into(), Value::Map(vec![])),
                ]),
                decode_worker_request: None,
            },
            GenerationOptions::default(),
        )
        .await
        .unwrap();
    let GenerationOutcome::Connected(generated) = result else {
        panic!("expected admission")
    };
    (context, generated)
}

async fn start(
    client: Arc<dyn GenerationCoordinatorClient>,
) -> RunningGenerationCoordinatorService {
    Arc::new(GenerationCoordinatorService::new(
        client,
        DisaggregationStrategy::Aggregated,
    ))
    .start("127.0.0.1:0".parse().unwrap())
    .await
    .unwrap()
}

fn set_remote(reader: &ConfigReader, url: Option<String>) {
    let mut config = UnifiedConfig::default();
    config.generation_coordinator.remotes =
        url.map(|url| BTreeMap::from([("default".into(), url)]));
    reader.replace(config);
}

#[tokio::test]
async fn relay_reloads_remote_endpoints_without_changing_mode_or_active_streams() {
    tokio::time::timeout(Duration::from_secs(10), async {
        let backend = |id| {
            Arc::new(Backend {
                id,
                ..Default::default()
            })
        };
        let first = backend(2);
        let second = backend(3);
        let first_server = start(first.clone()).await;
        let second_server = start(second.clone()).await;
        let reader = ConfigReader::in_memory(UnifiedConfig::default());
        set_remote(&reader, Some(first_server.endpoint_url()));
        let remote = Arc::new(RemoteGenerationCoordinator::from_config(reader.clone()).unwrap());
        let relay_server = start(remote).await;
        let relay = RemoteGenerationCoordinator::new(relay_server.endpoint_url()).unwrap();
        let (_, mut old) = request(&relay).await;
        assert_eq!(old.admission.prefill_worker_id, 2);

        set_remote(&reader, Some(second_server.endpoint_url()));
        let (context, mut new) = request(&relay).await;
        assert_eq!(new.admission.prefill_worker_id, 3);
        first.release.notify_one();
        assert_eq!(old.stream.next().await.unwrap().data, Some(Value::from(2)));
        assert!(old.stream.next().await.is_none());
        context.stop_generating();
        assert!(new.stream.next().await.is_none());
        drop(new);
        second.closed.notified().await;

        set_remote(&reader, None);
        assert!(configured_endpoint(&reader).is_err());
        relay_server.shutdown().await.unwrap();
        first_server.shutdown().await.unwrap();
        second_server.shutdown().await.unwrap();
    })
    .await
    .expect("relay test timed out");
}
