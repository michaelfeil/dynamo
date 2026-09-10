// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::{
    protocols::common::extensions::SessionAffinityId,
    session_affinity::{AffinityCoordinator, AffinityTarget},
};
use dynamo_runtime::{
    DistributedRuntime, Runtime,
    discovery::{DiscoveryQuery, EventChannelQuery},
    distributed::DistributedConfig,
};
use std::time::Duration;
use tokio::sync::watch;

#[tokio::test]
async fn replicas_sync_raw_worker_ids_on_the_coordinator_topic() {
    use dynamo_runtime::{
        distributed::DiscoveryBackend, pipeline::context::Controller, storage::kv,
    };
    let directory = tempfile::tempdir().unwrap();
    let config = || DistributedConfig {
        discovery_backend: DiscoveryBackend::KvStore(kv::Selector::File(directory.path().into())),
        ..DistributedConfig::process_local()
    };
    let drt = DistributedRuntime::new(Runtime::from_current().unwrap(), config())
        .await
        .unwrap();
    let namespace = "coordinator-affinity-sync";
    let component = drt
        .namespace(namespace)
        .unwrap()
        .component("coordinator_clients")
        .unwrap();
    let store = AffinityCoordinator::new(Duration::from_secs(60)).unwrap();
    let (workers, ids) = watch::channel(vec![10]);
    let topic = super::pool::AFFINITY_TOPIC;
    store
        .enable_replica_sync_for_pool(&component, topic, ids)
        .await
        .unwrap();
    let peer_runtime = DistributedRuntime::new(Runtime::from_current().unwrap(), config())
        .await
        .unwrap();
    let peer_component = peer_runtime
        .namespace(namespace)
        .unwrap()
        .component("coordinator_clients")
        .unwrap();
    let peer = AffinityCoordinator::new(Duration::from_secs(60)).unwrap();
    peer.enable_replica_sync_for_pool(&peer_component, topic, workers.subscribe())
        .await
        .unwrap();
    let context = Controller::new("affinity-test".into());
    let session = SessionAffinityId::new("session");
    workers.send_replace(vec![11]);
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            peer.acquire_with_context(&session, None, &context)
                .await
                .unwrap()
                .complete_selection(AffinityTarget {
                    worker_id: 11,
                    dp_rank: None,
                })
                .unwrap();
            if store.query_target(&session, None).unwrap().is_some() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("custom-topic update was not replicated");
    assert_eq!(
        store
            .query_target(&session, None)
            .unwrap()
            .unwrap()
            .worker_id,
        11
    );
    let default_topic = DiscoveryQuery::EventChannels(EventChannelQuery::topic(
        component.namespace().name(),
        component.name(),
        "session_affinity_events",
    ));
    assert!(
        drt.discovery()
            .list(default_topic)
            .await
            .unwrap()
            .is_empty()
    );
    drop(store);
    drop(peer);
    peer_runtime.shutdown();
    drt.shutdown();
}
