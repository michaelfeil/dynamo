// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dashmap::{DashMap, mapref::entry::Entry};
use std::collections::HashMap;
use std::hash::Hash;

use super::single::RequestId;
use crate::protocols::{WorkerId, WorkerWithDpRank};
use crate::scheduling::{ActiveRequestIslStats, IslStats};

#[derive(Debug, Clone, Copy, Default)]
struct IslAccumulator {
    count: usize,
    sum: f64,
    sum_sq: f64,
}

impl IslAccumulator {
    fn add(&mut self, isl: usize) {
        let isl = isl as f64;
        self.count += 1;
        self.sum += isl;
        self.sum_sq += isl * isl;
    }

    fn remove(&mut self, isl: usize) {
        let isl = isl as f64;
        self.count -= 1;
        self.sum -= isl;
        self.sum_sq -= isl * isl;
    }

    fn snapshot(&self) -> Option<IslStats> {
        if self.count == 0 {
            return None;
        }

        let count = self.count as f64;
        let mean = self.sum / count;
        let mean_sq = self.sum_sq / count;
        let variance = (mean_sq - mean * mean).max(0.0);
        Some(IslStats {
            count: self.count,
            mean,
            stddev: variance.sqrt(),
        })
    }
}

#[derive(Debug, Default)]
pub(super) struct RequestIndex {
    request_to_worker: DashMap<RequestId, WorkerWithDpRank>,
    request_to_lora: DashMap<RequestId, String>,
    /// ISL per active request. Absent means the request should not contribute
    /// to active-request ISL stats.
    request_to_isl: DashMap<RequestId, usize>,
    isl_stats_by_worker: DashMap<WorkerWithDpRank, IslAccumulator>,
    isl_stats_by_worker_id: DashMap<WorkerId, IslAccumulator>,
    track_worker_rank_isl: bool,
}

impl RequestIndex {
    pub(super) fn new(track_worker_rank_isl: bool) -> Self {
        Self {
            track_worker_rank_isl,
            ..Default::default()
        }
    }

    pub(super) fn try_insert_request(
        &self,
        request_id: RequestId,
        worker: WorkerWithDpRank,
        lora_name: Option<String>,
    ) -> Result<(), WorkerWithDpRank> {
        match self.request_to_worker.entry(request_id.clone()) {
            Entry::Occupied(entry) => Err(*entry.get()),
            Entry::Vacant(entry) => {
                entry.insert(worker);
                if let Some(lora_name) = lora_name {
                    self.request_to_lora.insert(request_id, lora_name);
                }
                Ok(())
            }
        }
    }

    pub(super) fn set_request(
        &self,
        request_id: RequestId,
        worker: WorkerWithDpRank,
        lora_name: Option<String>,
    ) {
        let previous_worker = self.request_to_worker.insert(request_id.clone(), worker);
        let previous_isl = self.request_to_isl.remove(&request_id).map(|(_, isl)| isl);
        if let (Some(previous_worker), Some(previous_isl)) = (previous_worker, previous_isl) {
            self.remove_isl(previous_worker, previous_isl);
        }
        if let Some(lora_name) = lora_name {
            self.request_to_lora.insert(request_id, lora_name);
        } else {
            self.request_to_lora.remove(&request_id);
        }
    }

    pub(super) fn set_request_isl(
        &self,
        request_id: RequestId,
        worker: WorkerWithDpRank,
        isl: usize,
    ) {
        self.request_to_isl.insert(request_id, isl);
        self.add_isl(worker, isl);
    }

    pub(super) fn worker_for(&self, request_id: &RequestId) -> Option<WorkerWithDpRank> {
        self.request_to_worker.get(request_id).map(|entry| *entry)
    }

    pub(super) fn lora_for(&self, request_id: &RequestId) -> Option<String> {
        self.request_to_lora
            .get(request_id)
            .map(|entry| entry.value().clone())
    }

    pub(super) fn remove_request(&self, request_id: &RequestId) -> Option<WorkerWithDpRank> {
        let worker = self
            .request_to_worker
            .remove(request_id)
            .map(|(_request_id, worker)| worker);
        self.request_to_lora.remove(request_id);
        if let (Some(worker), Some((_request_id, isl))) =
            (worker, self.request_to_isl.remove(request_id))
        {
            self.remove_isl(worker, isl);
        }
        worker
    }

    pub(super) fn remove_requests<'a>(&self, request_ids: impl IntoIterator<Item = &'a RequestId>) {
        for request_id in request_ids {
            self.remove_request(request_id);
        }
    }

    pub(super) fn remove_worker_requests(&self, worker: WorkerWithDpRank) -> Vec<RequestId> {
        let request_ids: Vec<_> = self
            .request_to_worker
            .iter()
            .filter(|entry| *entry.value() == worker)
            .map(|entry| entry.key().clone())
            .collect();
        self.remove_requests(request_ids.iter());
        request_ids
    }

    pub(super) fn active_lora_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for entry in self.request_to_lora.iter() {
            let lora_name = entry.value().clone();
            *counts.entry(lora_name).or_insert(0) += 1;
        }
        counts
    }

    pub(super) fn active_request_counts(&self) -> HashMap<WorkerWithDpRank, usize> {
        let mut counts = HashMap::new();
        for entry in self.request_to_worker.iter() {
            *counts.entry(*entry.value()).or_insert(0) += 1;
        }
        counts
    }

    /// Mean/stddev of ISL tokens over active requests, grouped by rank views.
    /// Stats are maintained incrementally on lifecycle changes, so
    /// this snapshot scales with active ranks rather than active requests.
    pub(super) fn active_request_isl_stats(&self) -> ActiveRequestIslStats {
        ActiveRequestIslStats {
            by_worker_with_dp_rank: self
                .track_worker_rank_isl
                .then(|| snapshot_isl_stats(&self.isl_stats_by_worker)),
            by_worker_id: snapshot_isl_stats(&self.isl_stats_by_worker_id),
        }
    }

    fn add_isl(&self, worker: WorkerWithDpRank, isl: usize) {
        if self.track_worker_rank_isl {
            self.isl_stats_by_worker.entry(worker).or_default().add(isl);
        }
        self.isl_stats_by_worker_id
            .entry(worker.worker_id)
            .or_default()
            .add(isl);
    }

    fn remove_isl(&self, worker: WorkerWithDpRank, isl: usize) {
        if self.track_worker_rank_isl {
            remove_isl_stat(&self.isl_stats_by_worker, worker, isl);
        }
        remove_isl_stat(&self.isl_stats_by_worker_id, worker.worker_id, isl);
    }

    #[cfg(any(test, feature = "bench"))]
    pub(super) fn is_empty(&self) -> bool {
        self.request_to_worker.is_empty()
            && self.request_to_lora.is_empty()
            && self.request_to_isl.is_empty()
            && self.isl_stats_by_worker.is_empty()
            && self.isl_stats_by_worker_id.is_empty()
    }

    #[cfg(any(test, feature = "bench"))]
    pub(super) fn worker_len(&self) -> usize {
        self.request_to_worker.len()
    }
}

fn snapshot_isl_stats<K>(map: &DashMap<K, IslAccumulator>) -> HashMap<K, IslStats>
where
    K: Copy + Eq + Hash,
{
    map.iter()
        .filter_map(|entry| entry.value().snapshot().map(|stats| (*entry.key(), stats)))
        .collect()
}

fn remove_isl_stat<K>(map: &DashMap<K, IslAccumulator>, key: K, isl: usize)
where
    K: Copy + Eq + Hash,
{
    if let Entry::Occupied(mut entry) = map.entry(key) {
        entry.get_mut().remove(isl);
        if entry.get().count == 0 {
            entry.remove_entry();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_insert_returns_existing_worker() {
        let index = RequestIndex::default();
        let worker = WorkerWithDpRank::new(1, 0);

        index
            .try_insert_request("req-1".to_string(), worker, Some("adapter".to_string()))
            .unwrap();
        assert_eq!(
            index.try_insert_request("req-1".to_string(), WorkerWithDpRank::new(2, 0), None),
            Err(worker)
        );
        assert_eq!(index.worker_for(&"req-1".to_string()), Some(worker));
        assert_eq!(
            index.lora_for(&"req-1".to_string()),
            Some("adapter".to_string())
        );
    }

    #[test]
    fn remove_request_is_idempotent() {
        let index = RequestIndex::default();
        let worker = WorkerWithDpRank::new(1, 0);
        let request_id = "req-1".to_string();

        index.set_request(request_id.clone(), worker, Some("adapter".to_string()));
        assert_eq!(index.remove_request(&request_id), Some(worker));
        assert_eq!(index.remove_request(&request_id), None);
        assert!(index.is_empty());
    }

    #[test]
    fn set_request_without_lora_clears_stale_lora_mapping() {
        let index = RequestIndex::default();
        let request_id = "req-1".to_string();

        index.set_request(
            request_id.clone(),
            WorkerWithDpRank::new(1, 0),
            Some("adapter".to_string()),
        );
        index.set_request(request_id.clone(), WorkerWithDpRank::new(2, 0), None);

        assert_eq!(
            index.worker_for(&request_id),
            Some(WorkerWithDpRank::new(2, 0))
        );
        assert_eq!(index.lora_for(&request_id), None);
    }

    #[test]
    fn remove_worker_requests_clears_both_maps() {
        let index = RequestIndex::default();
        let worker_a = WorkerWithDpRank::new(1, 0);
        let worker_b = WorkerWithDpRank::new(2, 0);
        index.set_request("req-a".to_string(), worker_a, Some("adapter-a".to_string()));
        index.set_request("req-b".to_string(), worker_b, Some("adapter-b".to_string()));
        index.set_request("req-c".to_string(), worker_a, None);

        let mut removed = index.remove_worker_requests(worker_a);
        removed.sort();
        assert_eq!(removed, vec!["req-a".to_string(), "req-c".to_string()]);
        assert_eq!(index.worker_for(&"req-b".to_string()), Some(worker_b));
        assert_eq!(
            index.active_lora_counts(),
            HashMap::from([("adapter-b".to_string(), 1)])
        );
    }
}
