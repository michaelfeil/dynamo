// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::poll_fn;
use std::sync::Arc;
use std::task::Poll;

use rustc_hash::FxHashMap;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use super::multi_worker::{
    ActiveSequencesMultiWorker, ReplicaWorkerPolicy, SequencePublisher, SequenceSubscriber,
};
use super::prompt_registry::WorkerLoadSnapshot;
use crate::protocols::{
    ActiveSequenceEvent, ActiveSequenceEventData, MAX_REPLICA_BATCH_DURATION,
    MAX_REPLICA_BATCH_EVENTS, WorkerWithDpRank,
};

#[derive(Default)]
struct ReplicaBatchEffects {
    worker_loads: FxHashMap<WorkerWithDpRank, PendingLoadPublication>,
    wake_scheduler: bool,
    cleanup_prompt_trie: bool,
}

struct PendingLoadPublication {
    latest_load: WorkerLoadSnapshot,
    publish: bool,
}

impl ReplicaBatchEffects {
    fn record_worker_load(
        &mut self,
        worker: WorkerWithDpRank,
        load: WorkerLoadSnapshot,
        publish: bool,
    ) {
        self.worker_loads
            .entry(worker)
            .and_modify(|pending| {
                pending.latest_load = load;
                pending.publish |= publish;
            })
            .or_insert(PendingLoadPublication {
                latest_load: load,
                publish,
            });
    }
}

impl<P: SequencePublisher + 'static> ActiveSequencesMultiWorker<P> {
    /// Apply one decoded replica-sync batch and flush its deferred effects once.
    pub fn apply_replica_batch(&self, events: Vec<ActiveSequenceEvent>) {
        self.apply_sourced_replica_batch(None, events);
    }

    /// Apply a decoded replica-sync batch attributed to one event publisher.
    pub fn apply_replica_batch_from_source(
        &self,
        publisher_id: u64,
        events: Vec<ActiveSequenceEvent>,
    ) {
        self.apply_sourced_replica_batch(Some(publisher_id), events);
    }

    fn apply_sourced_replica_batch(
        &self,
        publisher_id: Option<u64>,
        events: Vec<ActiveSequenceEvent>,
    ) {
        let mut effects = ReplicaBatchEffects::default();
        for mut event in events {
            event.publisher_id = publisher_id;
            self.apply_replica_event(event, &mut effects);
        }
        self.flush_replica_batch_effects(&mut effects);
    }

    /// Release scheduler state owned by a publisher that discovery has expired.
    pub fn replica_source_removed(&self, publisher_id: u64) {
        self.dead_replica_sources.insert(publisher_id);

        // Tombstone before scanning so a late AddRequest cannot race with eviction.
        let request_ids = self.request_index.requests_for_replica_source(publisher_id);
        let mut effects = ReplicaBatchEffects::default();
        let decay_now = Instant::now();
        for request_id in &request_ids {
            self.apply_replica_free(request_id, decay_now, &mut effects);
        }
        self.flush_replica_batch_effects(&mut effects);

        if !request_ids.is_empty() {
            tracing::warn!(
                publisher_id,
                evicted_requests = request_ids.len(),
                "Evicted scheduler state owned by dead router replica publisher"
            );
        }
    }

    /// Spawn a background task that subscribes to replica-sync events from peer routers
    /// and applies them to the local state.
    pub fn start_replica_sync<S: SequenceSubscriber + 'static>(
        self: &Arc<Self>,
        subscriber: S,
        cancel_token: CancellationToken,
    ) {
        let this = Arc::clone(self);
        tokio::spawn(async move {
            if let Err(error) = this.run_replica_sync(subscriber, cancel_token).await {
                tracing::error!("Error in active sequences events subscription: {error}");
            }
        });
    }

    pub(super) async fn run_replica_sync<S: SequenceSubscriber>(
        &self,
        mut subscriber: S,
        cancel_token: CancellationToken,
    ) -> anyhow::Result<()> {
        let mut effects = ReplicaBatchEffects::default();

        loop {
            let result = tokio::select! {
                result = subscriber.next_event() => result,
                _ = cancel_token.cancelled() => {
                    tracing::debug!("Subscription task cancelled");
                    break;
                }
            };

            let Some(result) = result else {
                break;
            };
            let Ok(event) = result else {
                tracing::error!(
                    "Error receiving active sequence event: {}",
                    result.unwrap_err()
                );
                continue;
            };

            let batch_start = Instant::now();
            let mut batch_events = 0;
            let mut exit_after_flush = false;
            let mut yield_after_flush = false;
            let mut next_event = Some(event);

            loop {
                let event = next_event
                    .take()
                    .expect("replica batch event must be present");
                self.apply_replica_event(event, &mut effects);
                batch_events += 1;

                if cancel_token.is_cancelled() {
                    exit_after_flush = true;
                    break;
                }

                if batch_events >= MAX_REPLICA_BATCH_EVENTS
                    || batch_start.elapsed() >= MAX_REPLICA_BATCH_DURATION
                {
                    yield_after_flush = true;
                    break;
                }

                match poll_fn(|cx| Poll::Ready(subscriber.poll_next_event(cx))).await {
                    Poll::Ready(Some(Ok(event))) => next_event = Some(event),
                    Poll::Ready(Some(Err(error))) => {
                        tracing::error!("Error receiving active sequence event: {error}");
                        break;
                    }
                    Poll::Ready(None) => {
                        exit_after_flush = true;
                        break;
                    }
                    Poll::Pending => break,
                }
            }

            self.flush_replica_batch_effects(&mut effects);

            if exit_after_flush {
                break;
            }
            if yield_after_flush {
                tokio::task::yield_now().await;
            }
        }

        Ok(())
    }

    fn apply_replica_event(&self, event: ActiveSequenceEvent, effects: &mut ReplicaBatchEffects) {
        let ActiveSequenceEvent {
            request_id,
            worker: event_worker,
            data,
            router_id,
            lora_name,
            publisher_id,
        } = event;

        if router_id == self.router_id {
            return;
        }

        if publisher_id.is_some_and(|id| self.dead_replica_sources.contains(&id)) {
            return;
        }

        // ActiveSequenceEvent does not carry prompt-load decay timestamps yet.
        // Peer routers still approximate decay anchoring with local receive time.
        let decay_now = Instant::now();

        match data {
            ActiveSequenceEventData::AddRequest {
                token_sequence,
                track_prefill_tokens,
                expected_output_tokens,
                prefill_load_hint,
            } => {
                if self.replica_worker_policy == ReplicaWorkerPolicy::LazyRegister {
                    self.ensure_worker_registered(event_worker);
                }
                let table = self.workers.read();
                let Some(&idx) = table.index.get(&event_worker) else {
                    tracing::debug!(
                        worker = ?event_worker,
                        "Dropping replica AddRequest for unregistered worker"
                    );
                    return;
                };

                self.request_index.set_request_from_replica_source(
                    request_id.clone(),
                    event_worker,
                    lora_name,
                    publisher_id,
                );
                let (expired_request_ids, load) = {
                    let slot = &table.slots[idx];
                    let mut seq = slot.sequences.write();
                    let outcome = seq.add_request_with_prefill_tracking(
                        request_id,
                        token_sequence,
                        expected_output_tokens,
                        track_prefill_tokens,
                        prefill_load_hint,
                        decay_now,
                    );
                    let load = seq.worker_load_snapshot();
                    self.prompt_registry
                        .apply_membership_delta_and_load_without_cleanup(
                            event_worker,
                            outcome.membership_delta,
                            load,
                        );
                    (outcome.expired_request_ids, load)
                };
                drop(table);
                self.request_index
                    .remove_requests(expired_request_ids.iter());

                // Close the only insertion race with replica_source_removed: if discovery
                // expired this publisher while AddRequest was being applied, release it now.
                if let Some(publisher_id) = publisher_id
                    && self.dead_replica_sources.contains(&publisher_id)
                {
                    self.replica_source_removed(publisher_id);
                    return;
                }
                effects.record_worker_load(event_worker, load, true);
                effects.cleanup_prompt_trie = true;
            }
            ActiveSequenceEventData::Free => {
                self.apply_replica_free(&request_id, decay_now, effects);
            }
            ActiveSequenceEventData::MarkPrefillCompleted => {
                let Some(worker) = self.request_index.worker_for(&request_id) else {
                    return;
                };
                let table = self.workers.read();
                let Some(&idx) = table.index.get(&worker) else {
                    return;
                };
                let load = {
                    let mut seq = table.slots[idx].sequences.write();
                    seq.mark_prefill_completed(&request_id, decay_now);
                    let load = seq.worker_load_snapshot();
                    self.prompt_registry.replace_worker_load_state(worker, load);
                    load
                };
                drop(table);
                effects.record_worker_load(worker, load, false);
                effects.wake_scheduler = true;
            }
        }
    }

    fn apply_replica_free(
        &self,
        request_id: &String,
        decay_now: Instant,
        effects: &mut ReplicaBatchEffects,
    ) {
        let Some(worker) = self.request_index.remove_request(request_id) else {
            return;
        };
        let table = self.workers.read();
        let Some(&idx) = table.index.get(&worker) else {
            return;
        };
        let load = {
            let slot = &table.slots[idx];
            let mut seq = slot.sequences.write();
            let delta = seq.free(request_id, decay_now);
            let load = seq.worker_load_snapshot();
            self.prompt_registry
                .apply_membership_delta_and_load_without_cleanup(worker, delta, load);
            load
        };
        drop(table);
        effects.record_worker_load(worker, load, true);
        effects.wake_scheduler = true;
        effects.cleanup_prompt_trie = true;
    }

    fn flush_replica_batch_effects(&self, effects: &mut ReplicaBatchEffects) {
        let decay_now = Instant::now();
        let mut active_loads = Vec::with_capacity(effects.worker_loads.len());
        for (worker, pending) in effects.worker_loads.drain() {
            if pending.publish {
                active_loads.push(self.observe_worker_load_snapshot(
                    worker,
                    pending.latest_load,
                    decay_now,
                ));
            }
        }
        if !active_loads.is_empty() {
            self.publisher.publish_load_batch(active_loads);
        }

        if std::mem::take(&mut effects.wake_scheduler) {
            self.notify_remote_state_update();
        }
        if std::mem::take(&mut effects.cleanup_prompt_trie) {
            self.prompt_registry.maybe_cleanup();
        }
    }
}
