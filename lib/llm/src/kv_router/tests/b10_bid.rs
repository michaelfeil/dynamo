// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::collections::HashMap;

use dynamo_kv_router::protocols::RouterRequest;
use dynamo_runtime::pipeline::{AsyncEngine, SingleIn};

use crate::{
    kv_router::{BasetenWorkerSelector, b10_worker_selector::B10WorkerSelector},
    protocols::common::extensions::SESSION_AFFINITY_CONTEXT_KEY,
    session_affinity::AffinityCoordinator,
};

fn request_context(id: &str, method: &str) -> SingleIn<RouterRequest> {
    let body: RouterRequest = serde_json::from_value(serde_json::json!({
        "method": method, "tokens": [11, 12, 21],
    }))
    .unwrap();
    let mut request = dynamo_runtime::pipeline::Context::with_id_and_metadata(
        body,
        id.to_string(),
        Default::default(),
    );
    request.insert_metadata(SESSION_AFFINITY_CONTEXT_KEY, "bid-session");
    request
}

async fn response_body<Sel>(
    router: &KvRouter<Sel>,
    request: SingleIn<RouterRequest>,
) -> RouterResponse
where
    Sel: dynamo_kv_router::selector::WorkerSelector<ModelRuntimeConfig> + Send + Sync + 'static,
{
    use futures::StreamExt;
    router
        .generate(request)
        .await
        .unwrap()
        .next()
        .await
        .unwrap()
        .data
        .unwrap()
}

#[tokio::test(start_paused = true)]
async fn bid_affinity_missing_bound_expired_and_normal_booking() {
    let affinity = AffinityCoordinator::new(std::time::Duration::from_secs(10)).unwrap();
    let session = crate::protocols::common::extensions::SessionAffinityId::new("bid-session");
    let router = make_test_router(BasetenWorkerSelector::B10(B10WorkerSelector::new()), None)
        .await
        .with_session_affinity_coordinator(affinity.clone());
    assert!(matches!(
        response_body(&router, request_context("live", "bid")).await,
        RouterResponse::Bid {
            worker_id: 0 | 1,
            dp_rank: 0,
            prefill_blocks,
            decode_blocks: 0,
        } if prefill_blocks == 3.0 / f64::from(router.block_size)
    ));
    assert_eq!(affinity.query_target(&session, None).unwrap(), None);
    assert!(router.affinity_leases.is_empty());
    // An ordinary New still acquires affinity and books active state.
    assert!(matches!(
        response_body(&router, request_context("live", "new")).await,
        RouterResponse::New { .. }
    ));
    assert!(router.affinity_leases.contains_key("live"));
    let active = || {
        router
            .scheduler
            .get_potential_loads(None, 0, HashMap::new(), false, false)
    };
    let active_count = || {
        active()
            .iter()
            .map(|load| load.active_requests)
            .sum::<usize>()
    };
    assert_eq!(active_count(), 1);
    // Repeated bids with the same ID must not release the live booking.
    for _ in 0..3 {
        response_body(&router, request_context("live", "bid")).await;
    }
    assert!(router.affinity_leases.contains_key("live"));
    assert_eq!(active_count(), 1);
    assert_eq!(router.pending_count(), 0);
    router.free("live").await.unwrap();
    assert_eq!(active_count(), 0);
    tokio::time::advance(std::time::Duration::from_secs(9)).await;
    response_body(&router, request_context("idle-bid", "bid")).await;
    tokio::time::advance(std::time::Duration::from_secs(2)).await;
    assert_eq!(
        affinity.query_target(&session, None).unwrap(),
        None,
        "bid must not refresh TTL"
    );
    assert!(router.affinity_leases.is_empty());
    assert_eq!(active_count(), 0);
}
