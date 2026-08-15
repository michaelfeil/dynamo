// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Duration};

use dynamo_runtime::{engine::AsyncEngineContext, error::ErrorType, pipeline::context::Controller};

use super::{
    AffinityAcquire, AffinityCoordinator, AffinityTarget, coordinator::ReplicaApplyOutcome,
};
use crate::protocols::common::extensions::SessionAffinityId;

fn session_id() -> SessionAffinityId {
    SessionAffinityId::new("session-1")
}

fn target(worker_id: u64, dp_rank: Option<u32>) -> AffinityTarget {
    AffinityTarget { worker_id, dp_rank }
}

fn coordinator() -> AffinityCoordinator {
    AffinityCoordinator::new(Duration::from_secs(10)).unwrap()
}

async fn assert_binding_expires_after_refreshed_ttl(coordinator: &AffinityCoordinator) {
    tokio::time::advance(Duration::from_secs(9)).await;
    assert_eq!(
        coordinator.query_target(&session_id(), None).unwrap(),
        Some(target(7, Some(0)))
    );
    tokio::time::advance(Duration::from_secs(2)).await;
    assert_eq!(coordinator.query_target(&session_id(), None).unwrap(), None);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_initialization_is_atomic() {
    let coordinator = coordinator();
    let first = coordinator.acquire(&session_id(), None).await.unwrap();
    let AffinityAcquire::Initialize(first) = first else {
        panic!("first request must initialize");
    };

    let waiter_coordinator = coordinator.clone();
    let waiter = tokio::spawn(async move { waiter_coordinator.acquire(&session_id(), None).await });
    coordinator.wait_for_initializing_waiter().await;
    assert!(!waiter.is_finished());

    let first_lease = first.commit(target(7, Some(0))).unwrap();
    let second = waiter.await.unwrap().unwrap();
    let AffinityAcquire::Bound {
        target: second_target,
        lease: second_lease,
    } = second
    else {
        panic!("waiter must acquire the committed binding");
    };
    assert_eq!(second_target, target(7, Some(0)));
    drop(first_lease);
    drop(second_lease);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_initializer_cancellation_wakes_waiter() {
    let coordinator = coordinator();
    let first = coordinator.acquire(&session_id(), None).await.unwrap();
    let AffinityAcquire::Initialize(first) = first else {
        panic!("first request must initialize");
    };

    let waiter_coordinator = coordinator.clone();
    let waiter = tokio::spawn(async move { waiter_coordinator.acquire(&session_id(), None).await });
    coordinator.wait_for_initializing_waiter().await;
    drop(first);

    let next = waiter.await.unwrap().unwrap();
    assert!(matches!(&next, AffinityAcquire::Initialize(_)));
    drop(next);
    assert_eq!(coordinator.entry_count(), 0);
    assert!(matches!(
        coordinator.acquire(&session_id(), None).await.unwrap(),
        AffinityAcquire::Initialize(_)
    ));
}

#[tokio::test(start_paused = true)]
async fn session_affinity_wait_stops_when_request_is_cancelled() {
    let coordinator = coordinator();
    let first = coordinator.acquire(&session_id(), None).await.unwrap();
    let AffinityAcquire::Initialize(first) = first else {
        panic!("first request must initialize");
    };

    let context = Arc::new(Controller::default());
    let waiter_context = context.clone();
    let waiter_coordinator = coordinator.clone();
    let waiter = tokio::spawn(async move {
        waiter_coordinator
            .acquire_with_context(&session_id(), None, waiter_context.as_ref())
            .await
    });
    coordinator.wait_for_initializing_waiter().await;
    context.stop();

    let Err(error) = waiter.await.unwrap() else {
        panic!("cancelled waiter must return an error");
    };
    assert!(dynamo_runtime::error::match_error_chain(
        error.as_ref(),
        &[ErrorType::Cancelled],
        &[]
    ));
    drop(first);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_validates_worker_and_rank_contract() {
    let coordinator = coordinator();
    let AffinityAcquire::Initialize(initializer) = coordinator
        .acquire(&session_id(), Some(target(7, None)))
        .await
        .unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(target(7, None)).unwrap());

    assert!(
        coordinator
            .acquire(&session_id(), Some(target(8, None)))
            .await
            .is_err()
    );
    assert!(
        coordinator
            .acquire(&session_id(), Some(target(7, Some(0))))
            .await
            .is_err()
    );
    assert!(
        coordinator
            .acquire(&session_id(), Some(target(7, None)))
            .await
            .is_ok()
    );
}

#[tokio::test(start_paused = true)]
async fn session_affinity_failed_bound_operation_invalidates_binding() {
    let coordinator = coordinator();
    let AffinityAcquire::Initialize(initializer) =
        coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(target(7, Some(0))).unwrap());

    let operation = coordinator.acquire(&session_id(), None).await.unwrap();
    assert_eq!(operation.target(), Some(target(7, Some(0))));
    operation.invalidate();

    assert_eq!(coordinator.query_target(&session_id(), None).unwrap(), None);
    assert_eq!(coordinator.entry_count(), 0);
    assert!(matches!(
        coordinator.acquire(&session_id(), None).await.unwrap(),
        AffinityAcquire::Initialize(_)
    ));
}

#[tokio::test(start_paused = true)]
async fn session_affinity_bound_lease_drop_refreshes_idle_ttl() {
    let coordinator = coordinator();
    let AffinityAcquire::Initialize(initializer) =
        coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(target(7, Some(0))).unwrap());

    tokio::time::advance(Duration::from_secs(9)).await;
    let AffinityAcquire::Bound { lease, .. } =
        coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("continuation must acquire the binding");
    };
    tokio::time::advance(Duration::from_secs(2)).await;
    tokio::task::yield_now().await;
    assert_eq!(
        coordinator.query_target(&session_id(), None).unwrap(),
        Some(target(7, Some(0)))
    );
    drop(lease);

    assert_binding_expires_after_refreshed_ttl(&coordinator).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_query_is_read_only() {
    let coordinator = coordinator();
    assert_eq!(coordinator.query_target(&session_id(), None).unwrap(), None);
    assert_eq!(coordinator.entry_count(), 0);

    let initializing = coordinator.acquire(&session_id(), None).await.unwrap();
    assert_eq!(coordinator.query_target(&session_id(), None).unwrap(), None);
    assert_eq!(coordinator.entry_count(), 1);
    drop(initializing);
    assert_eq!(coordinator.entry_count(), 0);

    let AffinityAcquire::Initialize(initializer) =
        coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(target(7, Some(0))).unwrap());
    assert_eq!(
        coordinator.query_target(&session_id(), None).unwrap(),
        Some(target(7, Some(0)))
    );
    coordinator.expire_for_test(&session_id());
    assert_eq!(coordinator.query_target(&session_id(), None).unwrap(), None);
    assert_eq!(coordinator.entry_count(), 1);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_reaper_removes_idle_entries_and_stops_on_drop() {
    let coordinator = coordinator();
    let cancellation = coordinator.cancellation_token();
    let AffinityAcquire::Initialize(initializer) =
        coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(target(7, Some(0))).unwrap());

    coordinator.wait_for_reaper().await;
    tokio::time::advance(Duration::from_secs(10)).await;
    tokio::task::yield_now().await;
    assert_eq!(coordinator.entry_count(), 0);

    drop(coordinator);
    cancellation.cancelled().await;
}

#[test]
fn session_affinity_rejects_invalid_ttl_before_starting_reaper() {
    for ttl in [
        Duration::ZERO,
        Duration::from_secs(super::MAX_SESSION_AFFINITY_TTL_SECS + 1),
    ] {
        let Err(error) = AffinityCoordinator::new(ttl) else {
            panic!("invalid TTL must fail coordinator construction");
        };
        assert!(dynamo_runtime::error::match_error_chain(
            error.as_ref(),
            &[dynamo_runtime::error::ErrorType::InvalidArgument],
            &[]
        ));
        assert!(error.to_string().contains("session affinity TTL"));
    }
}

#[tokio::test(start_paused = true)]
async fn session_affinity_enforces_id_and_entry_limits() {
    let coordinator = AffinityCoordinator::with_test_limits(1, 8);
    let oversized = SessionAffinityId::new("123456789");
    let Err(error) = coordinator.acquire(&oversized, None).await else {
        panic!("oversized session ID must fail");
    };
    assert!(dynamo_runtime::error::match_error_chain(
        error.as_ref(),
        &[ErrorType::InvalidArgument],
        &[]
    ));
    assert_eq!(coordinator.entry_count(), 0);

    let first_id = SessionAffinityId::new("first");
    let first = coordinator.acquire(&first_id, None).await.unwrap();
    let second_id = SessionAffinityId::new("second");
    let Err(error) = coordinator.acquire(&second_id, None).await else {
        panic!("entry limit must reject a second session");
    };
    assert!(dynamo_runtime::error::match_error_chain(
        error.as_ref(),
        &[ErrorType::ResourceExhausted],
        &[]
    ));

    drop(first);
    assert_eq!(coordinator.entry_count(), 0);
    assert!(matches!(
        coordinator.acquire(&second_id, None).await.unwrap(),
        AffinityAcquire::Initialize(_)
    ));
}

#[tokio::test(start_paused = true)]
async fn session_affinity_replica_newer_rebind_converges() {
    let coordinator = coordinator();
    let local_target = target(7, Some(0));
    let conflicting_target = target(8, Some(1));

    assert_eq!(
        coordinator.apply_replica_version_for_test("replicated", local_target, 1, 1, 10),
        ReplicaApplyOutcome::Inserted
    );
    assert_eq!(
        coordinator
            .query_target(&SessionAffinityId::new("replicated"), None)
            .unwrap(),
        Some(local_target)
    );
    assert_eq!(
        coordinator.apply_replica_version_for_test("replicated", conflicting_target, 1, 2, 20),
        ReplicaApplyOutcome::ReplacedNewer
    );
    assert_eq!(
        coordinator.apply_replica_version_for_test("replicated", local_target, 1, 1, 10),
        ReplicaApplyOutcome::IgnoredStale
    );
    assert_eq!(
        coordinator
            .query_target(&SessionAffinityId::new("replicated"), None)
            .unwrap(),
        Some(conflicting_target)
    );
}

#[tokio::test(start_paused = true)]
async fn session_affinity_replica_restart_epoch_supersedes_old_counter() {
    let coordinator = coordinator();
    let old_target = target(7, Some(0));
    let restarted_target = target(8, Some(1));

    assert_eq!(
        coordinator.apply_replica_version_for_test("replicated", old_target, 100, 50, 10),
        ReplicaApplyOutcome::Inserted
    );
    assert_eq!(
        coordinator.apply_replica_version_for_test("replicated", restarted_target, 200, 1, 20),
        ReplicaApplyOutcome::ReplacedNewer
    );
    assert_eq!(
        coordinator.apply_replica_version_for_test("replicated", old_target, 100, 51, 10),
        ReplicaApplyOutcome::IgnoredStale
    );
    assert_eq!(
        coordinator
            .query_target(&SessionAffinityId::new("replicated"), None)
            .unwrap(),
        Some(restarted_target)
    );
}

#[tokio::test(start_paused = true)]
async fn session_affinity_replica_duplicate_refreshes_local_ttl() {
    let coordinator = coordinator();
    let replicated_id = SessionAffinityId::new("replicated");
    let replicated_target = target(7, Some(0));

    assert_eq!(
        coordinator.apply_replica_update_for_test("replicated", replicated_target),
        ReplicaApplyOutcome::Inserted
    );
    tokio::time::advance(Duration::from_secs(9)).await;
    assert_eq!(
        coordinator.apply_replica_update_for_test("replicated", replicated_target),
        ReplicaApplyOutcome::Refreshed
    );
    tokio::time::advance(Duration::from_secs(2)).await;
    assert_eq!(
        coordinator.query_target(&replicated_id, None).unwrap(),
        Some(replicated_target)
    );
    tokio::time::advance(Duration::from_secs(9)).await;
    assert_eq!(
        coordinator.query_target(&replicated_id, None).unwrap(),
        None
    );
}

#[tokio::test(start_paused = true)]
async fn session_affinity_replica_ignores_initializing_sessions() {
    let coordinator = coordinator();
    let initializing = coordinator.acquire(&session_id(), None).await.unwrap();

    assert_eq!(
        coordinator.apply_replica_update_for_test(session_id().as_str(), target(7, Some(0))),
        ReplicaApplyOutcome::IgnoredInitializing
    );
    assert_eq!(coordinator.query_target(&session_id(), None).unwrap(), None);
    drop(initializing);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_replica_enforces_id_and_entry_limits() {
    let coordinator = AffinityCoordinator::with_test_limits(1, 8);

    assert_eq!(
        coordinator.apply_replica_update_for_test("123456789", target(7, Some(0))),
        ReplicaApplyOutcome::RejectedSessionId
    );
    assert_eq!(
        coordinator.apply_replica_update_for_test("first", target(7, Some(0))),
        ReplicaApplyOutcome::Inserted
    );
    assert_eq!(
        coordinator.apply_replica_update_for_test("second", target(8, Some(0))),
        ReplicaApplyOutcome::RejectedCapacity
    );
    assert_eq!(coordinator.entry_count(), 1);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_publishes_after_selection_and_lease_completion() {
    let coordinator = coordinator();
    let mut updates = coordinator.enable_test_replica(99, 4);
    let selected_target = target(7, Some(0));
    let operation = coordinator.acquire(&session_id(), None).await.unwrap();
    let lease = operation
        .complete_selection(selected_target)
        .unwrap()
        .unwrap();

    let after_dispatch = updates.recv().await.unwrap();
    assert_eq!(after_dispatch.session_id, session_id().as_str());
    assert_eq!(after_dispatch.worker_id, selected_target.worker_id);
    assert_eq!(after_dispatch.dp_rank, selected_target.dp_rank);
    assert_eq!(after_dispatch.router_id, 99);

    assert!(updates.try_recv().is_err());
    drop(lease);
    let after_completion = updates.recv().await.unwrap();
    assert_eq!(after_completion, after_dispatch);
    assert!(updates.try_recv().is_err());
}

#[tokio::test(start_paused = true)]
async fn session_affinity_local_rebind_publishes_newer_version() {
    let origin = coordinator();
    let mut updates = origin.enable_test_replica(99, 8);
    let replica = coordinator();
    let original_target = target(7, Some(0));
    let rebound_target = target(8, Some(1));

    let lease = origin
        .acquire(&session_id(), None)
        .await
        .unwrap()
        .complete_selection(original_target)
        .unwrap()
        .unwrap();
    let original = updates.recv().await.unwrap();
    assert_eq!(
        replica.apply_replica_message_for_test(original.clone()),
        ReplicaApplyOutcome::Inserted
    );
    drop(lease);
    updates.recv().await.unwrap();

    let rebound_lease = origin
        .acquire(&session_id(), None)
        .await
        .unwrap()
        .complete_selection(rebound_target)
        .unwrap()
        .unwrap();
    let rebound = updates.recv().await.unwrap();
    assert!(rebound.version_counter > original.version_counter);
    assert_eq!(
        replica.apply_replica_message_for_test(rebound),
        ReplicaApplyOutcome::ReplacedNewer
    );
    assert_eq!(
        replica.apply_replica_message_for_test(original),
        ReplicaApplyOutcome::IgnoredStale
    );
    assert_eq!(
        replica.query_target(&session_id(), None).unwrap(),
        Some(rebound_target)
    );
    drop(rebound_lease);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_concurrent_rebind_keeps_first_newer_binding() {
    let coordinator = coordinator();
    let original_target = target(7, Some(0));
    let first_rebind_target = target(8, Some(1));
    let second_rebind_target = target(9, Some(2));

    let initial_lease = coordinator
        .acquire(&session_id(), None)
        .await
        .unwrap()
        .complete_selection(original_target)
        .unwrap()
        .unwrap();
    drop(initial_lease);

    let first = coordinator.acquire(&session_id(), None).await.unwrap();
    let second = coordinator.acquire(&session_id(), None).await.unwrap();
    let first_lease = first
        .complete_selection(first_rebind_target)
        .unwrap()
        .unwrap();
    assert!(
        second
            .complete_selection(second_rebind_target)
            .unwrap()
            .is_none()
    );
    assert_eq!(
        coordinator.query_target(&session_id(), None).unwrap(),
        Some(first_rebind_target)
    );
    drop(first_lease);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_completion_restores_expired_remote_binding() {
    let origin = coordinator();
    let mut updates = origin.enable_test_replica(99, 4);
    let replica = coordinator();
    let replicated_target = target(7, Some(0));
    let operation = origin.acquire(&session_id(), None).await.unwrap();
    let lease = operation
        .complete_selection(replicated_target)
        .unwrap()
        .unwrap();

    let after_dispatch = updates.recv().await.unwrap();
    assert_eq!(
        replica.apply_replica_message_for_test(after_dispatch),
        ReplicaApplyOutcome::Inserted
    );
    tokio::time::advance(Duration::from_secs(11)).await;
    assert_eq!(replica.query_target(&session_id(), None).unwrap(), None);

    drop(lease);
    let after_completion = updates.recv().await.unwrap();
    assert_eq!(
        replica.apply_replica_message_for_test(after_completion),
        ReplicaApplyOutcome::ReplacedExpired
    );
    assert_eq!(
        replica.query_target(&session_id(), None).unwrap(),
        Some(replicated_target)
    );
}
