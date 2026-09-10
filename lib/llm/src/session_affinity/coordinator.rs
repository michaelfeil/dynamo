// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    sync::{
        Arc, Mutex, OnceLock, Weak,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use dashmap::{DashMap, mapref::entry::Entry};
use dynamo_runtime::{
    engine::AsyncEngineContext,
    error::{DynamoError, ErrorType},
    pipeline::Error,
};
use tokio::{sync::Notify, time::Instant};
use tokio_util::sync::CancellationToken;

use super::{
    MAX_SESSION_AFFINITY_ENTRIES, MAX_SESSION_AFFINITY_ID_BYTES, MAX_SESSION_AFFINITY_TTL_SECS,
    replica_sync::{ReplicaSyncRuntime, SessionAffinityUpdate},
};
use crate::protocols::common::extensions::SessionAffinityId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffinityTarget {
    pub worker_id: u64,
    pub dp_rank: Option<u32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct AffinityVersion {
    pub epoch: u64,
    pub counter: u64,
    pub router_id: u64,
}

struct AffinityVersionClock {
    epoch: u64,
    counter: u64,
}

enum AffinityEntry {
    Initializing {
        revision: u64,
        notify: Arc<Notify>,
    },
    Bound {
        target: AffinityTarget,
        version: AffinityVersion,
        revision: u64,
        active_leases: usize,
        idle_deadline: Instant,
    },
}

pub(super) struct AffinityCoordinatorInner {
    entries: DashMap<String, AffinityEntry>,
    ttl: Duration,
    max_entries: usize,
    max_session_id_bytes: usize,
    entry_count: AtomicUsize,
    next_revision: AtomicU64,
    version_clock: Mutex<AffinityVersionClock>,
    cancel: CancellationToken,
    replica: OnceLock<ReplicaSyncRuntime>,
    #[cfg(test)]
    reaper_started: Arc<Notify>,
    #[cfg(test)]
    waiter_observed: Arc<Notify>,
}

impl Drop for AffinityCoordinatorInner {
    fn drop(&mut self) {
        self.cancel.cancel();
        if let Some(replica) = self.replica.get_mut() {
            replica.shutdown_now();
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ReplicaApplyOutcome {
    Inserted,
    Refreshed,
    ReplacedExpired,
    ReplacedNewer,
    IgnoredInitializing,
    IgnoredStale,
    RejectedSessionId,
    RejectedCapacity,
}

#[derive(Clone)]
pub struct AffinityCoordinator {
    inner: Arc<AffinityCoordinatorInner>,
}

impl AffinityCoordinator {
    /// Synchronize a caller-owned worker pool on its own event topic.
    pub async fn enable_replica_sync_for_pool(
        &self,
        component: &dynamo_runtime::component::Component,
        topic: &str,
        workers: tokio::sync::watch::Receiver<Vec<u64>>,
    ) -> Result<(), Error> {
        let replica = ReplicaSyncRuntime::start(
            component,
            topic,
            workers,
            Arc::downgrade(&self.inner),
            &self.inner.cancel,
        )
        .await?;
        self.inner
            .replica
            .set(replica)
            .map_err(|_| anyhow::anyhow!("session affinity replica sync already enabled"))
    }

    pub fn new(ttl: Duration) -> Result<Self, Error> {
        Self::new_with_limits(
            ttl,
            MAX_SESSION_AFFINITY_ENTRIES,
            MAX_SESSION_AFFINITY_ID_BYTES,
        )
    }

    fn new_with_limits(
        ttl: Duration,
        max_entries: usize,
        max_session_id_bytes: usize,
    ) -> Result<Self, Error> {
        if !(Duration::from_secs(1)..=Duration::from_secs(MAX_SESSION_AFFINITY_TTL_SECS))
            .contains(&ttl)
        {
            return Err(invalid_argument(format!(
                "session affinity TTL must be between 1 and {MAX_SESSION_AFFINITY_TTL_SECS} seconds"
            )));
        }
        let inner = Arc::new(AffinityCoordinatorInner {
            entries: DashMap::new(),
            ttl,
            max_entries,
            max_session_id_bytes,
            entry_count: AtomicUsize::new(0),
            next_revision: AtomicU64::new(1),
            version_clock: Mutex::new(AffinityVersionClock {
                epoch: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64,
                counter: 0,
            }),
            cancel: CancellationToken::new(),
            replica: OnceLock::new(),
            #[cfg(test)]
            reaper_started: Arc::new(Notify::new()),
            #[cfg(test)]
            waiter_observed: Arc::new(Notify::new()),
        });
        Self::spawn_reaper(&inner);
        tracing::info!(
            ttl_secs = ttl.as_secs(),
            max_entries,
            "session affinity enabled"
        );
        Ok(Self { inner })
    }

    fn spawn_reaper(inner: &Arc<AffinityCoordinatorInner>) {
        let weak = Arc::downgrade(inner);
        let cancel = inner.cancel.clone();
        let period = inner.ttl.min(Duration::from_secs(30));
        #[cfg(test)]
        let reaper_started = inner.reaper_started.clone();
        tokio::spawn(async move {
            #[cfg(test)]
            reaper_started.notify_one();
            loop {
                tokio::select! {
                    _ = cancel.cancelled() => return,
                    _ = tokio::time::sleep(period) => {}
                }
                let Some(inner) = weak.upgrade() else {
                    return;
                };
                let now = Instant::now();
                let mut removed = 0;
                inner.entries.retain(|_, entry| {
                    let retain = !matches!(
                        entry,
                        AffinityEntry::Bound {
                            active_leases: 0,
                            idle_deadline,
                            ..
                        } if *idle_deadline <= now
                    );
                    removed += usize::from(!retain);
                    retain
                });
                inner.entry_count.fetch_sub(removed, Ordering::Relaxed);
            }
        });
    }

    pub(crate) async fn enable_replica_sync(
        &self,
        client: dynamo_runtime::component::Client,
    ) -> Result<(), Error> {
        self.enable_replica_sync_for_pool(
            client.endpoint.component(),
            super::replica_sync::SESSION_AFFINITY_SUBJECT,
            client.instance_avail_watcher(),
        )
        .await
    }

    #[cfg(test)]
    pub(crate) async fn acquire(
        &self,
        session_id: &SessionAffinityId,
        requested_target: Option<AffinityTarget>,
    ) -> Result<AffinityAcquire, Error> {
        self.acquire_inner(session_id, requested_target, None).await
    }

    pub async fn acquire_with_context(
        &self,
        session_id: &SessionAffinityId,
        requested_target: Option<AffinityTarget>,
        request_context: &dyn AsyncEngineContext,
    ) -> Result<AffinityAcquire, Error> {
        self.acquire_inner(session_id, requested_target, Some(request_context))
            .await
    }

    async fn acquire_inner(
        &self,
        session_id: &SessionAffinityId,
        requested_target: Option<AffinityTarget>,
        request_context: Option<&dyn AsyncEngineContext>,
    ) -> Result<AffinityAcquire, Error> {
        self.validate_session_id(session_id)?;
        let session_id = session_id.as_str().to_string();

        loop {
            let now = Instant::now();
            match self.inner.entries.entry(session_id.clone()) {
                Entry::Vacant(entry) => {
                    self.reserve_entry()?;
                    tracing::debug!(
                        session_id = %session_id,
                        "session affinity miss: new session, pinning after worker selection"
                    );
                    return Ok(AffinityAcquire::Initialize(entry.insert_initializing(
                        &self.inner,
                        session_id,
                        requested_target,
                    )));
                }
                Entry::Occupied(mut entry) => match entry.get_mut() {
                    AffinityEntry::Initializing { notify, .. } => {
                        #[cfg(test)]
                        self.inner.waiter_observed.notify_one();
                        let notified = notify.clone().notified_owned();
                        tokio::pin!(notified);
                        notified.as_mut().enable();
                        drop(entry);
                        if let Some(context) = request_context {
                            tokio::select! {
                                biased;
                                _ = context.stopped() => {
                                    return Err(cancelled(context.id()));
                                }
                                _ = context.killed() => {
                                    return Err(cancelled(context.id()));
                                }
                                _ = notified => {}
                            }
                        } else {
                            notified.await;
                        }
                    }
                    AffinityEntry::Bound {
                        target: _,
                        revision,
                        active_leases,
                        idle_deadline,
                        ..
                    } if *active_leases == 0 && *idle_deadline <= now => {
                        tracing::debug!(
                            session_id = %session_id,
                            "session affinity miss: pin expired (idle past TTL), re-selecting worker"
                        );
                        let revision = self.inner.next_revision.fetch_add(1, Ordering::Relaxed);
                        let notify = Arc::new(Notify::new());
                        *entry.get_mut() = AffinityEntry::Initializing {
                            revision,
                            notify: notify.clone(),
                        };
                        drop(entry);
                        return Ok(AffinityAcquire::Initialize(AffinityInitialization {
                            coordinator: Arc::downgrade(&self.inner),
                            session_id,
                            revision,
                            notify,
                            requested_target,
                            active: true,
                        }));
                    }
                    AffinityEntry::Bound {
                        target,
                        version: _,
                        revision,
                        active_leases,
                        ..
                    } => {
                        validate_bound_target(&session_id, *target, requested_target)?;
                        tracing::debug!(
                            session_id = %session_id,
                            worker_id = target.worker_id,
                            dp_rank = ?target.dp_rank,
                            active_leases = *active_leases + 1,
                            "session affinity hit: reusing pinned worker"
                        );
                        *active_leases += 1;
                        let lease = AffinityLease {
                            coordinator: Arc::downgrade(&self.inner),
                            session_id,
                            revision: *revision,
                            active: true,
                        };
                        return Ok(AffinityAcquire::Bound {
                            target: *target,
                            lease,
                        });
                    }
                },
            }
        }
    }

    pub fn query_target(
        &self,
        session_id: &SessionAffinityId,
        requested_target: Option<AffinityTarget>,
    ) -> Result<Option<AffinityTarget>, Error> {
        self.validate_session_id(session_id)?;
        let Some(entry) = self.inner.entries.get(session_id.as_str()) else {
            return Ok(None);
        };
        let AffinityEntry::Bound {
            target,
            active_leases,
            idle_deadline,
            ..
        } = entry.value()
        else {
            return Ok(None);
        };
        if *active_leases == 0 && *idle_deadline <= Instant::now() {
            return Ok(None);
        }
        validate_bound_target(session_id.as_str(), *target, requested_target)?;
        tracing::debug!(
            session_id = %session_id.as_str(),
            worker_id = target.worker_id,
            dp_rank = ?target.dp_rank,
            "session affinity hit: reusing pinned worker"
        );

        Ok(Some(*target))
    }

    #[cfg(test)]
    pub(super) fn entry_count(&self) -> usize {
        self.inner.entry_count.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(super) fn cancellation_token(&self) -> CancellationToken {
        self.inner.cancel.clone()
    }

    #[cfg(test)]
    pub(super) async fn wait_for_reaper(&self) {
        self.inner.reaper_started.notified().await;
    }

    #[cfg(test)]
    pub(super) async fn wait_for_initializing_waiter(&self) {
        self.inner.waiter_observed.notified().await;
    }

    #[cfg(test)]
    pub(super) fn expire_for_test(&self, session_id: &SessionAffinityId) {
        let Some(mut entry) = self.inner.entries.get_mut(session_id.as_str()) else {
            panic!("session affinity entry missing");
        };
        let AffinityEntry::Bound {
            active_leases,
            idle_deadline,
            ..
        } = entry.value_mut()
        else {
            panic!("session affinity entry is not bound");
        };
        assert_eq!(*active_leases, 0);
        *idle_deadline = Instant::now();
    }

    #[cfg(test)]
    pub(super) fn with_test_limits(max_entries: usize, max_session_id_bytes: usize) -> Self {
        Self::new_with_limits(Duration::from_secs(10), max_entries, max_session_id_bytes).unwrap()
    }

    #[cfg(test)]
    pub(super) fn enable_test_replica(
        &self,
        router_id: u64,
        capacity: usize,
    ) -> tokio::sync::mpsc::Receiver<SessionAffinityUpdate> {
        let (replica, rx) = ReplicaSyncRuntime::for_test(router_id, capacity);
        self.inner
            .replica
            .set(replica)
            .unwrap_or_else(|_| panic!("session affinity test replica already enabled"));
        rx
    }

    #[cfg(test)]
    pub(super) fn apply_replica_update_for_test(
        &self,
        session_id: impl Into<String>,
        target: AffinityTarget,
    ) -> ReplicaApplyOutcome {
        let version = self.inner.next_affinity_version();
        let update = SessionAffinityUpdate {
            session_id: session_id.into(),
            worker_id: target.worker_id,
            dp_rank: target.dp_rank,
            router_id: version.router_id,
            version_epoch: version.epoch,
            version_counter: version.counter,
            version_router_id: version.router_id,
        };
        self.inner.apply_replica_update(update)
    }

    #[cfg(test)]
    pub(super) fn apply_replica_version_for_test(
        &self,
        session_id: impl Into<String>,
        target: AffinityTarget,
        epoch: u64,
        counter: u64,
        router_id: u64,
    ) -> ReplicaApplyOutcome {
        let update = SessionAffinityUpdate {
            session_id: session_id.into(),
            worker_id: target.worker_id,
            dp_rank: target.dp_rank,
            router_id,
            version_epoch: epoch,
            version_counter: counter,
            version_router_id: router_id,
        };
        self.inner.apply_replica_update(update)
    }

    #[cfg(test)]
    pub(super) fn apply_replica_message_for_test(
        &self,
        update: SessionAffinityUpdate,
    ) -> ReplicaApplyOutcome {
        self.inner.apply_replica_update(update)
    }

    fn validate_session_id(&self, session_id: &SessionAffinityId) -> Result<(), Error> {
        if session_id.as_str().len() > self.inner.max_session_id_bytes {
            return Err(invalid_argument(format!(
                "session affinity ID must not exceed {} bytes",
                self.inner.max_session_id_bytes
            )));
        }
        Ok(())
    }

    fn reserve_entry(&self) -> Result<(), Error> {
        self.inner
            .reserve_entry()
            .then_some(())
            .ok_or_else(|| resource_exhausted("session affinity entry limit reached"))
    }
}

impl AffinityCoordinatorInner {
    fn reserve_entry(&self) -> bool {
        self.entry_count
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
                (count < self.max_entries).then_some(count + 1)
            })
            .is_ok()
    }

    fn next_affinity_version(&self) -> AffinityVersion {
        let mut clock = self
            .version_clock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        clock.counter += 1;
        AffinityVersion {
            epoch: clock.epoch,
            counter: clock.counter,
            router_id: self
                .replica
                .get()
                .map(ReplicaSyncRuntime::router_id)
                .unwrap_or_default(),
        }
    }

    fn observe_affinity_version(&self, version: AffinityVersion) {
        let mut clock = self
            .version_clock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if version.epoch > clock.epoch {
            clock.epoch = version.epoch;
            clock.counter = version.counter;
        } else if version.epoch == clock.epoch {
            clock.counter = clock.counter.max(version.counter);
        }
    }

    fn publish_replica_update(
        &self,
        session_id: &str,
        target: AffinityTarget,
        version: AffinityVersion,
    ) {
        if let Some(replica) = self.replica.get() {
            replica.publish(session_id, target, version);
        }
    }

    pub(super) fn apply_replica_update(
        &self,
        update: SessionAffinityUpdate,
    ) -> ReplicaApplyOutcome {
        let SessionAffinityUpdate {
            session_id,
            worker_id,
            dp_rank,
            router_id,
            version_epoch,
            version_counter,
            version_router_id,
        } = update;
        if session_id.len() > self.max_session_id_bytes {
            return ReplicaApplyOutcome::RejectedSessionId;
        }
        let target = AffinityTarget { worker_id, dp_rank };
        let version = AffinityVersion {
            epoch: version_epoch,
            counter: version_counter,
            router_id: if version_counter == 0 {
                router_id
            } else {
                version_router_id
            },
        };
        self.observe_affinity_version(version);

        let now = Instant::now();
        match self.entries.entry(session_id) {
            Entry::Vacant(entry) => {
                if !self.reserve_entry() {
                    return ReplicaApplyOutcome::RejectedCapacity;
                }
                let revision = self.next_revision.fetch_add(1, Ordering::Relaxed);
                entry.insert(AffinityEntry::Bound {
                    target,
                    version,
                    revision,
                    active_leases: 0,
                    idle_deadline: now + self.ttl,
                });
                ReplicaApplyOutcome::Inserted
            }
            Entry::Occupied(mut entry) => match entry.get_mut() {
                AffinityEntry::Initializing { .. } => ReplicaApplyOutcome::IgnoredInitializing,
                AffinityEntry::Bound {
                    version: existing_version,
                    active_leases,
                    idle_deadline,
                    ..
                } if *active_leases == 0
                    && *idle_deadline <= now
                    && version >= *existing_version =>
                {
                    let revision = self.next_revision.fetch_add(1, Ordering::Relaxed);
                    *entry.get_mut() = AffinityEntry::Bound {
                        target,
                        version,
                        revision,
                        active_leases: 0,
                        idle_deadline: now + self.ttl,
                    };
                    ReplicaApplyOutcome::ReplacedExpired
                }
                AffinityEntry::Bound {
                    target: existing,
                    version: existing_version,
                    idle_deadline,
                    ..
                } if *existing == target && *existing_version <= version => {
                    *existing_version = version;
                    *idle_deadline = now + self.ttl;
                    ReplicaApplyOutcome::Refreshed
                }
                AffinityEntry::Bound {
                    version: existing_version,
                    ..
                } if version > *existing_version => {
                    let revision = self.next_revision.fetch_add(1, Ordering::Relaxed);
                    *entry.get_mut() = AffinityEntry::Bound {
                        target,
                        version,
                        revision,
                        active_leases: 0,
                        idle_deadline: now + self.ttl,
                    };
                    ReplicaApplyOutcome::ReplacedNewer
                }
                AffinityEntry::Bound { .. } => ReplicaApplyOutcome::IgnoredStale,
            },
        }
    }
}

trait VacantEntryExt {
    fn insert_initializing(
        self,
        inner: &Arc<AffinityCoordinatorInner>,
        session_id: String,
        requested_target: Option<AffinityTarget>,
    ) -> AffinityInitialization;
}

impl<'a> VacantEntryExt for dashmap::mapref::entry::VacantEntry<'a, String, AffinityEntry> {
    fn insert_initializing(
        self,
        inner: &Arc<AffinityCoordinatorInner>,
        session_id: String,
        requested_target: Option<AffinityTarget>,
    ) -> AffinityInitialization {
        let revision = inner.next_revision.fetch_add(1, Ordering::Relaxed);
        let notify = Arc::new(Notify::new());
        self.insert(AffinityEntry::Initializing {
            revision,
            notify: notify.clone(),
        });
        AffinityInitialization {
            coordinator: Arc::downgrade(inner),
            session_id,
            revision,
            notify,
            requested_target,
            active: true,
        }
    }
}

pub enum AffinityAcquire {
    Initialize(AffinityInitialization),
    Bound {
        target: AffinityTarget,
        lease: AffinityLease,
    },
}

impl AffinityAcquire {
    pub fn target(&self) -> Option<AffinityTarget> {
        match self {
            Self::Initialize(_) => None,
            Self::Bound { target, .. } => Some(*target),
        }
    }

    pub(crate) fn invalidate(self) {
        if let Self::Bound { mut lease, .. } = self {
            lease.invalidate();
        }
    }

    pub fn complete_selection(
        self,
        selected_target: AffinityTarget,
    ) -> Result<Option<AffinityLease>, Error> {
        match self {
            Self::Initialize(initialization) => {
                let lease = initialization.commit(selected_target)?;
                lease.publish(selected_target);
                Ok(Some(lease))
            }
            Self::Bound { target, lease } => {
                if target.worker_id != selected_target.worker_id
                    || target
                        .dp_rank
                        .is_some_and(|rank| Some(rank) != selected_target.dp_rank)
                {
                    let Some(lease) = lease.rebind(selected_target) else {
                        return Ok(None);
                    };
                    lease.publish(selected_target);
                    Ok(Some(lease))
                } else {
                    lease.publish(target);
                    Ok(Some(lease))
                }
            }
        }
    }
}

pub struct AffinityInitialization {
    coordinator: Weak<AffinityCoordinatorInner>,
    session_id: String,
    revision: u64,
    notify: Arc<Notify>,
    requested_target: Option<AffinityTarget>,
    active: bool,
}

impl AffinityInitialization {
    pub(crate) fn commit(mut self, target: AffinityTarget) -> Result<AffinityLease, Error> {
        validate_bound_target(&self.session_id, target, self.requested_target)?;
        let Some(inner) = self.coordinator.upgrade() else {
            return Err(anyhow::anyhow!("session affinity coordinator dropped"));
        };
        let Some(mut entry) = inner.entries.get_mut(&self.session_id) else {
            return Err(invalid_argument(
                "session affinity initialization was cancelled",
            ));
        };
        if !matches!(
            entry.value(),
            AffinityEntry::Initializing { revision, .. } if *revision == self.revision
        ) {
            return Err(invalid_argument("session affinity initialization changed"));
        }
        *entry = AffinityEntry::Bound {
            target,
            version: inner.next_affinity_version(),
            revision: self.revision,
            active_leases: 1,
            idle_deadline: Instant::now() + inner.ttl,
        };
        drop(entry);
        self.active = false;
        self.notify.notify_waiters();
        Ok(AffinityLease {
            coordinator: Arc::downgrade(&inner),
            session_id: self.session_id.clone(),
            revision: self.revision,
            active: true,
        })
    }
}

impl Drop for AffinityInitialization {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let Some(inner) = self.coordinator.upgrade() else {
            return;
        };
        let removed = inner.entries.remove_if(&self.session_id, |_, entry| {
            matches!(
                entry,
                AffinityEntry::Initializing { revision, .. } if *revision == self.revision
            )
        });
        if removed.is_some() {
            inner.entry_count.fetch_sub(1, Ordering::Relaxed);
        }
        self.notify.notify_waiters();
    }
}

/// Keeps a local binding alive until the admitted stream is dropped.
pub struct AffinityLease {
    coordinator: Weak<AffinityCoordinatorInner>,
    session_id: String,
    revision: u64,
    active: bool,
}

impl AffinityLease {
    fn publish(&self, target: AffinityTarget) {
        let Some(inner) = self.coordinator.upgrade() else {
            return;
        };
        let Some(entry) = inner.entries.get(&self.session_id) else {
            return;
        };
        let AffinityEntry::Bound {
            target: current,
            version,
            revision,
            ..
        } = entry.value()
        else {
            return;
        };
        if *revision == self.revision && *current == target {
            inner.publish_replica_update(&self.session_id, target, *version);
        }
    }

    fn rebind(mut self, target: AffinityTarget) -> Option<Self> {
        let inner = self.coordinator.upgrade()?;
        let mut entry = inner.entries.get_mut(&self.session_id)?;
        if !matches!(
            entry.value(),
            AffinityEntry::Bound { revision, .. } if *revision == self.revision
        ) {
            return None;
        }
        let revision = inner.next_revision.fetch_add(1, Ordering::Relaxed);
        *entry = AffinityEntry::Bound {
            target,
            version: inner.next_affinity_version(),
            revision,
            active_leases: 1,
            idle_deadline: Instant::now() + inner.ttl,
        };
        drop(entry);
        self.active = false;
        Some(Self {
            coordinator: Arc::downgrade(&inner),
            session_id: self.session_id.clone(),
            revision,
            active: true,
        })
    }

    fn release(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        let Some(inner) = self.coordinator.upgrade() else {
            return;
        };
        let (target, version) = {
            let Some(mut entry) = inner.entries.get_mut(&self.session_id) else {
                return;
            };
            let AffinityEntry::Bound {
                target,
                version,
                revision,
                active_leases,
                idle_deadline,
            } = entry.value_mut()
            else {
                return;
            };
            if *revision != self.revision || *active_leases == 0 {
                return;
            }
            *active_leases -= 1;
            *idle_deadline = Instant::now() + inner.ttl;
            (*target, *version)
        };
        inner.publish_replica_update(&self.session_id, target, version);
    }

    fn invalidate(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        let Some(inner) = self.coordinator.upgrade() else {
            return;
        };
        let removed = inner.entries.remove_if(&self.session_id, |_, entry| {
            matches!(
                entry,
                AffinityEntry::Bound { revision, .. } if *revision == self.revision
            )
        });
        if removed.is_some() {
            inner.entry_count.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl Drop for AffinityLease {
    fn drop(&mut self) {
        self.release();
    }
}

fn validate_bound_target(
    session_id: &str,
    bound: AffinityTarget,
    requested: Option<AffinityTarget>,
) -> Result<(), Error> {
    let Some(requested) = requested else {
        return Ok(());
    };
    if bound.worker_id != requested.worker_id {
        return Err(invalid_argument(format!(
            "session {session_id} is bound to worker {}, not {}",
            bound.worker_id, requested.worker_id
        )));
    }
    match (bound.dp_rank, requested.dp_rank) {
        (Some(bound), Some(requested)) if bound != requested => Err(invalid_argument(format!(
            "session {session_id} is bound to DP rank {bound}, not {requested}"
        ))),
        (None, Some(requested)) => Err(invalid_argument(format!(
            "session {session_id} has worker-only affinity and cannot add DP rank {requested}"
        ))),
        _ => Ok(()),
    }
}

pub(crate) fn invalid_argument(message: impl Into<String>) -> Error {
    DynamoError::builder()
        .error_type(ErrorType::InvalidArgument)
        .message(message.into())
        .build()
        .into()
}

fn resource_exhausted(message: impl Into<String>) -> Error {
    DynamoError::builder()
        .error_type(ErrorType::ResourceExhausted)
        .message(message.into())
        .build()
        .into()
}

fn cancelled(context_id: &str) -> Error {
    DynamoError::builder()
        .error_type(ErrorType::Cancelled)
        .message(format!(
            "request {context_id} was cancelled while waiting for session affinity"
        ))
        .build()
        .into()
}
