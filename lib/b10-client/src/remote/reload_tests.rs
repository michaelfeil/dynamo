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
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;
use tokio::sync::Notify;

#[derive(Default)]
struct Backend {
    id: u64,
    release: Arc<Notify>,
    closed: Arc<Notify>,
    load: AtomicUsize,
    unavailable: AtomicBool,
    deny: AtomicBool,
    bids: AtomicUsize,
    decode_tokens: AtomicUsize,
    session: std::sync::Mutex<Option<String>>,
    stall_bid: bool,
}

struct Closed(Arc<Notify>);
impl Drop for Closed {
    fn drop(&mut self) {
        self.0.notify_one();
    }
}

impl GenerationCoordinatorClient for Backend {
    fn bid(
        &self,
        request: protocol::BidRequestV1,
    ) -> BoxFuture<'_, Result<protocol::BidResponseV1>> {
        Box::pin(async move {
            self.bids.fetch_add(1, Ordering::Relaxed);
            if self.stall_bid {
                self.release.notified().await;
            }
            request.validate()?;
            anyhow::ensure!(!self.unavailable.load(Ordering::Relaxed), "unavailable");
            Ok(protocol::BidResponseV1 {
                affinity: request.affinity_worker_id == Some(self.id),
                prefill_tokens: self.load.load(Ordering::Relaxed) as u64,
                decode_tokens: self.decode_tokens.load(Ordering::Relaxed) as u64,
            })
        })
    }
    fn worker_loads(&self) -> BoxFuture<'_, Result<Vec<crate::WorkerLoad>>> {
        Box::pin(async {
            anyhow::ensure!(!self.unavailable.load(Ordering::Relaxed), "unavailable");
            Ok(vec![crate::WorkerLoad {
                worker_id: self.id,
                disaggregation_mode: crate::WorkerMode::PrefillAndDecode,
                potential_prefill_tokens: 0,
                potential_decode_blocks: 0,
                active_requests: self.load.load(Ordering::Relaxed),
            }])
        })
    }
    fn generate(
        &self,
        context: RequestContext,
        request: GenerationRequest,
        _: GenerationOptions,
    ) -> BoxFuture<'_, Result<GenerationOutcome>> {
        Box::pin(async move {
            *self.session.lock().unwrap() = context
                .metadata_snapshot()
                .get(dynamo_llm::protocols::common::extensions::SESSION_AFFINITY_CONTEXT_KEY)
                .cloned();
            anyhow::ensure!(
                request
                    .routing_request
                    .allowed_worker_ids
                    .as_ref()
                    .is_none_or(|ids| ids.contains(&self.id)),
                "ineligible worker"
            );
            if self.deny.load(Ordering::Relaxed) {
                return Ok(cancelled_outcome());
            }
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
    let (context, result) = session_request(client, "").await;
    let GenerationOutcome::Connected(generated) = result else {
        panic!("expected admission")
    };
    (context, generated)
}

async fn session_request(
    client: &dyn GenerationCoordinatorClient,
    session: &str,
) -> (Arc<dyn AsyncEngineContext>, GenerationOutcome) {
    restricted_request(client, session, &[]).await
}

async fn restricted_request(
    client: &dyn GenerationCoordinatorClient,
    session: &str,
    allowed: &[u64],
) -> (Arc<dyn AsyncEngineContext>, GenerationOutcome) {
    let context: Arc<dyn AsyncEngineContext> = Arc::new(Controller::new("reload-test".into()));
    let result = client
        .generate(
            RequestContext::new(
                context.clone(),
                None,
                BTreeMap::from([(
                    dynamo_llm::protocols::common::extensions::SESSION_AFFINITY_CONTEXT_KEY.into(),
                    session.into(),
                )]),
            ),
            GenerationRequest {
                routing_request: RouterRequestNew {
                    tokens: vec![1, 2, 3],
                    allowed_worker_ids: (!allowed.is_empty())
                        .then(|| allowed.iter().copied().collect()),
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
    (context, result)
}

fn admitted_id(outcome: GenerationOutcome) -> u64 {
    let GenerationOutcome::Connected(generated) = outcome else {
        panic!("expected admission")
    };
    generated.admission.prefill_worker_id
}

#[tokio::test]
async fn remote_pool_keeps_live_affinity_and_rebinds_after_worker_loss() {
    use baseten_configmap::CoordinatorAffinityConfig;
    use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};

    tokio::time::timeout(Duration::from_secs(25), async {
        let first = Arc::new(Backend {
            id: 2,
            ..Default::default()
        });
        let second = Arc::new(Backend {
            id: 3,
            load: AtomicUsize::new(10),
            ..Default::default()
        });
        let first_server = start(first.clone()).await;
        let second_server = start(second.clone()).await;
        let runtime = DistributedRuntime::new(
            Runtime::from_current().unwrap(),
            DistributedConfig::process_local(),
        )
        .await
        .unwrap();
        let mut config = UnifiedConfig::default();
        config.generation_coordinator.remotes = Some(BTreeMap::from([(
            "default".into(),
            first_server.endpoint_url(),
        )]));
        config.generation_coordinator.affinity = Some(CoordinatorAffinityConfig { ttl_secs: 60 });
        let reader = ConfigReader::in_memory(config.clone());
        let namespace = runtime.namespace("coordinator-affinity-test").unwrap();
        let remote = RemoteGenerationCoordinator::from_runtime_config(
            reader.clone(),
            Some(&namespace),
            None,
        )
        .unwrap();
        remote.start().await.unwrap();
        let bid_counts = || {
            (
                first.bids.load(Ordering::Relaxed),
                second.bids.load(Ordering::Relaxed),
            )
        };
        let (_, active) = session_request(&remote, "sticky").await;
        assert!(
            matches!(&active, GenerationOutcome::Connected(g) if g.admission.prefill_worker_id == 2)
        );
        assert_eq!(bid_counts(), (0, 0));

        // Collect affinity before a second destination exists. Adding a cheaper
        // remote must retain that binding; new sessions bid once it is polled live.
        config
            .generation_coordinator
            .remotes
            .as_mut()
            .unwrap()
            .insert("canary".into(), second_server.endpoint_url());
        reader.replace(config.clone());
        first.load.store(15, Ordering::Relaxed);
        assert_eq!(admitted_id(session_request(&remote, "sticky").await.1), 2);
        assert_eq!(bid_counts(), (0, 0));
        while !remote
            .worker_loads()
            .await
            .unwrap()
            .iter()
            .any(|load| load.worker_id == 3)
        {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert_eq!(
            admitted_id(session_request(&remote, "new-after-expansion").await.1),
            3
        );
        assert_eq!(bid_counts(), (1, 1));

        // Same scorer for direct bids and generation fallback: affinity halves
        // the cost, and decode tokens contribute one tenth of their count.
        let mut bid = protocol::BidRequestV1 {
            tokens: vec![1, 2, 3],
            affinity_worker_id: Some(2),
            ..Default::default()
        };
        assert!(remote.bid(bid.clone()).await.unwrap().affinity);
        bid.affinity_worker_id = None;
        assert_eq!(remote.bid(bid.clone()).await.unwrap().prefill_tokens, 10);
        second.decode_tokens.store(100, Ordering::Relaxed);
        assert_eq!(remote.bid(bid).await.unwrap().prefill_tokens, 15);
        first.load.store(0, Ordering::Relaxed);
        second.decode_tokens.store(0, Ordering::Relaxed);
        let topic = dynamo_runtime::discovery::DiscoveryQuery::EventChannels(
            dynamo_runtime::discovery::EventChannelQuery::topic(
                namespace.name(),
                "coordinator_clients",
                pool::AFFINITY_TOPIC,
            ),
        );
        assert_eq!(runtime.discovery().list(topic).await.unwrap().len(), 1);
        assert_eq!(
            admitted_id(session_request(&remote, "restricted").await.1),
            2
        );
        assert_eq!(
            admitted_id(restricted_request(&remote, "restricted", &[3]).await.1),
            3
        );
        assert_eq!(
            admitted_id(restricted_request(&remote, "restricted-cold", &[3]).await.1),
            3
        );
        let before = bid_counts();
        assert_eq!(admitted_id(session_request(&remote, "sticky").await.1), 2);
        assert_eq!(bid_counts(), before, "live affinity must bypass every bid");

        // A denial must release the initializing slot without inventing an admission.
        first.deny.store(true, Ordering::Relaxed);
        assert!(matches!(
            session_request(&remote, "denied").await.1,
            GenerationOutcome::Denied(_)
        ));
        first.deny.store(false, Ordering::Relaxed);
        first.load.store(20, Ordering::Relaxed);
        second.load.store(0, Ordering::Relaxed);
        let before = bid_counts();
        assert_eq!(
            admitted_id(session_request(&remote, "fresh-bid").await.1),
            3
        );
        assert_eq!(bid_counts(), (before.0 + 1, before.1 + 1));
        loop {
            let loads = remote.worker_loads().await.unwrap();
            if loads
                .iter()
                .any(|load| load.worker_id == 2 && load.active_requests == 20)
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert_eq!(admitted_id(session_request(&remote, "sticky").await.1), 2);
        assert_eq!(admitted_id(session_request(&remote, "denied").await.1), 3);
        assert_eq!(admitted_id(session_request(&remote, "cold").await.1), 3);

        first.unavailable.store(true, Ordering::Relaxed);
        while remote
            .worker_loads()
            .await
            .unwrap()
            .iter()
            .any(|load| load.worker_id == 2)
        {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert_eq!(remote.worker_loads().await.unwrap()[0].worker_id, 3);
        let before = bid_counts();
        assert_eq!(admitted_id(session_request(&remote, "sticky").await.1), 3);
        assert_eq!(
            admitted_id(session_request(&remote, "after-loss").await.1),
            3
        );
        remote
            .bid(protocol::BidRequestV1 {
                tokens: vec![1],
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(bid_counts(), (before.0, before.1 + 3));
        // Endpoint removal applies immediately, even before the next probe.
        config
            .generation_coordinator
            .remotes
            .as_mut()
            .unwrap()
            .remove("canary");
        reader.replace(config);
        let RemoteEndpoints::Pool(pool) = &remote.endpoints else {
            unreachable!()
        };
        let context = RequestContext::new(
            Arc::new(Controller::new("removed".into())),
            None,
            BTreeMap::new(),
        );
        let before = bid_counts();
        // A singleton does not need a live inventory or a bid, but must still
        // record the actual admission, replacing the old canary binding.
        assert_eq!(admitted_id(session_request(&remote, "sticky").await.1), 2);
        let (endpoint, affinity) = pool
            .select(
                Some("sticky"),
                protocol::BidRequestV1 {
                    tokens: vec![1],
                    ..Default::default()
                },
                &[],
                &context,
            )
            .await
            .unwrap();
        assert_eq!(endpoint.as_str(), first_server.endpoint_url());
        assert_eq!(affinity.unwrap().target().unwrap().worker_id, 2);
        assert_eq!(bid_counts(), before);
        assert_eq!(first.session.lock().unwrap().as_deref(), Some("sticky"));
        // The earlier stream still belongs to the original backend.
        first.release.notify_one();
        let GenerationOutcome::Connected(mut active) = active else {
            unreachable!()
        };
        assert_eq!(
            active.stream.next().await.unwrap().data,
            Some(Value::from(2))
        );
        drop(active);
        drop(remote);
        first_server.shutdown().await.unwrap();
        second_server.shutdown().await.unwrap();
        runtime.shutdown();
    })
    .await
    .expect("remote pool test timed out");
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

#[tokio::test]
async fn relay_preserves_healthy_bid_when_an_option_times_out() {
    use prost::Message;
    tokio::time::timeout(Duration::from_secs(8), async {
        let healthy = Arc::new(Backend {
            load: AtomicUsize::new(7),
            ..Default::default()
        });
        let stalled = Arc::new(Backend {
            stall_bid: true,
            ..Default::default()
        });
        let healthy_server = start(healthy.clone()).await;
        let stalled_server = start(stalled.clone()).await;
        let mut config = UnifiedConfig::default();
        config.generation_coordinator.remotes = Some(BTreeMap::from([
            ("healthy".into(), healthy_server.endpoint_url()),
            ("stalled".into(), stalled_server.endpoint_url()),
        ]));
        let relay = start(Arc::new(
            RemoteGenerationCoordinator::from_config(ConfigReader::in_memory(config)).unwrap(),
        ))
        .await;
        // A relay aggregation may take slightly longer than one option's 5s
        // budget. Do not impose another equal deadline on the whole response.
        let response = Client::new()
            .post(relay.endpoint_url().replace("coordinate", "bid"))
            .body(
                protocol::BidRequestV1 {
                    tokens: vec![1],
                    ..Default::default()
                }
                .encode_to_vec(),
            )
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap();
        let bid = protocol::BidResponseV1::decode(response.bytes().await.unwrap()).unwrap();
        assert_eq!(bid.prefill_tokens, 7);
        assert_eq!(healthy.bids.load(Ordering::Relaxed), 1);
        assert_eq!(stalled.bids.load(Ordering::Relaxed), 1);
        relay.shutdown().await.unwrap();
        healthy_server.shutdown().await.unwrap();
        stalled.release.notify_one();
        stalled_server.shutdown().await.unwrap();
    })
    .await
    .expect("relay did not finish after the per-option timeout");
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
                unavailable: AtomicBool::new(true),
                ..Default::default()
            })
        };
        let first = backend(2);
        let second = backend(3);
        let first_server = start(first.clone()).await;
        let second_server = start(second.clone()).await;
        let reader = ConfigReader::in_memory(UnifiedConfig::default());
        let remote = Arc::new(
            RemoteGenerationCoordinator::from_runtime_config(
                reader.clone(),
                None,
                Some(first_server.endpoint_url()),
            )
            .unwrap(),
        );
        let relay_server = start(remote).await;
        let relay = RemoteGenerationCoordinator::new(relay_server.endpoint_url()).unwrap();
        let explicit = crate::GenerationCoordinatorRuntime::remote(BTreeMap::from([(
            "default".into(),
            relay_server.endpoint_url(),
        )]))
        .unwrap();
        explicit.start().await.unwrap();
        let context = RequestContext::new(
            Arc::new(Controller::new("explicit-singleton".into())),
            None,
            BTreeMap::from([(
                dynamo_llm::protocols::common::extensions::SESSION_AFFINITY_CONTEXT_KEY.into(),
                "single-session".into(),
            )]),
        );
        assert_eq!(
            admitted_id(
                explicit
                    .generate(
                        context,
                        GenerationRequest {
                            routing_request: RouterRequestNew {
                                tokens: vec![1],
                                ..Default::default()
                            },
                            primary_worker_request: Value::Map(vec![
                                ("model".into(), "test".into()),
                                ("sampling_params".into(), Value::Map(vec![])),
                            ]),
                            decode_worker_request: None,
                        },
                        GenerationOptions::default()
                    )
                    .await
                    .unwrap()
            ),
            2
        );
        assert_eq!(
            first.session.lock().unwrap().as_deref(),
            Some("single-session")
        );
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
        assert_eq!(request(&relay).await.1.admission.prefill_worker_id, 2);
        assert!(RemoteGenerationCoordinator::from_config(reader.clone()).is_err());
        assert_eq!(first.bids.load(Ordering::Relaxed), 0);
        assert_eq!(second.bids.load(Ordering::Relaxed), 0);
        relay_server.shutdown().await.unwrap();
        first_server.shutdown().await.unwrap();
        second_server.shutdown().await.unwrap();
    })
    .await
    .expect("relay test timed out");
}
