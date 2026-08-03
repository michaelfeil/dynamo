// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mapping from planner-observed `WorkerId`s to routable endpoints.
//!
//! GWP can choose a deployment ingress but the endpoint's local router chooses
//! the internal worker. GWP nevertheless schedules the planner-observed
//! workers: after the response arrives, `x-baseten-dyn-worker-id` corrects a
//! provisional choice when the local router served the request elsewhere.

use std::sync::Arc;

use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use dynamo_kv_router::protocols::WorkerId;

use crate::config::EndpointId;

/// Reverse mapping from scheduler worker ID to routable endpoint ID.
#[derive(Clone, Default)]
pub struct EndpointTable {
    by_worker: Arc<DashMap<WorkerId, EndpointId>>,
}

impl EndpointTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Associate a planner-observed worker with the ingress endpoint that owns
    /// it. Worker IDs must be globally unique across configured endpoints.
    pub fn upsert_worker(
        &self,
        worker_id: WorkerId,
        endpoint_id: EndpointId,
    ) -> anyhow::Result<()> {
        match self.by_worker.entry(worker_id) {
            Entry::Occupied(existing) => {
                anyhow::ensure!(
                    existing.get() == &endpoint_id,
                    "worker {worker_id} is advertised by both {} and {}",
                    existing.get().0,
                    endpoint_id.0
                );
            }
            Entry::Vacant(entry) => {
                entry.insert(endpoint_id);
            }
        }
        Ok(())
    }

    pub fn get(&self, worker_id: WorkerId) -> Option<EndpointId> {
        self.by_worker.get(&worker_id).map(|entry| entry.clone())
    }

    pub fn retain(&self, keep: &std::collections::HashSet<WorkerId>) {
        self.by_worker
            .retain(|worker_id, _| keep.contains(worker_id));
    }

    pub fn workers_in_endpoints(
        &self,
        endpoints: &std::collections::HashSet<EndpointId>,
    ) -> std::collections::HashSet<WorkerId> {
        self.by_worker
            .iter()
            .filter(|entry| endpoints.contains(entry.value()))
            .map(|entry| *entry.key())
            .collect()
    }

    pub fn len(&self) -> usize {
        self.by_worker.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_worker.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_table_roundtrip_and_retain() {
        let table = EndpointTable::new();
        table.upsert_worker(10, EndpointId("a".into())).unwrap();
        table.upsert_worker(20, EndpointId("b".into())).unwrap();
        let (a, b) = (10, 20);
        assert_eq!(table.get(a), Some(EndpointId("a".into())));

        table.retain(&std::collections::HashSet::from([b]));
        assert!(table.get(a).is_none());
        assert_eq!(table.get(b), Some(EndpointId("b".into())));
    }

    #[test]
    fn endpoint_filter_returns_scheduler_ids() {
        let table = EndpointTable::new();
        table.upsert_worker(10, EndpointId("a".into())).unwrap();
        table.upsert_worker(20, EndpointId("b".into())).unwrap();
        assert_eq!(
            table.workers_in_endpoints(&std::collections::HashSet::from([EndpointId("a".into())])),
            std::collections::HashSet::from([10])
        );
    }

    #[test]
    fn rejects_worker_advertised_by_two_endpoints() {
        let table = EndpointTable::new();
        table.upsert_worker(10, EndpointId("a".into())).unwrap();
        assert!(table.upsert_worker(10, EndpointId("b".into())).is_err());
    }
}
