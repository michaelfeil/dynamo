// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Mutex;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pythonize::pythonize;

use dynamo_kv_router::protocols::{
    DpRank, WorkerId, WorkerSelectionResult as RsWorkerSelectionResult, WorkerWithDpRank,
};
use dynamo_kv_router::scheduling::{
    KvSchedulerError as RsKvSchedulerError, RoutingEligibility, SchedulingRequest,
};
use dynamo_kv_router::selector::WorkerSelector;
use dynamo_llm::local_model::runtime_config::ModelRuntimeConfig;

#[pyclass]
#[derive(Clone)]
pub struct PySchedulingRequest {
    #[pyo3(get)]
    pub request_id: Option<String>,
    #[pyo3(get)]
    pub isl_tokens: usize,
    #[pyo3(get)]
    pub block_size: u32,
    #[pyo3(get)]
    pub pinned_worker: Option<(WorkerId, DpRank)>,
    #[pyo3(get)]
    pub allowed_worker_ids: Option<Vec<WorkerId>>,
    #[pyo3(get)]
    pub overlaps: Py<PyAny>,
    #[pyo3(get)]
    pub effective_overlap_blocks: Py<PyAny>,
    #[pyo3(get)]
    pub effective_cached_tokens: Py<PyAny>,
    #[pyo3(get)]
    pub decode_blocks: Py<PyAny>,
    #[pyo3(get)]
    pub prefill_tokens: Py<PyAny>,
}

#[pyclass]
#[derive(Clone)]
pub struct PyWorkerSelectionResult {
    #[pyo3(get)]
    pub worker_id: WorkerId,
    #[pyo3(get)]
    pub dp_rank: DpRank,
}

#[pymethods]
impl PyWorkerSelectionResult {
    #[new]
    fn new(worker_id: WorkerId, dp_rank: DpRank) -> Self {
        Self { worker_id, dp_rank }
    }
}

pub struct PythonWorkerSelector {
    python_worker_selector: Mutex<Py<PyAny>>,
}

impl PythonWorkerSelector {
    pub fn new(python_worker_selector: Py<PyAny>) -> Self {
        Self {
            python_worker_selector: Mutex::new(python_worker_selector),
        }
    }
}

impl WorkerSelector<ModelRuntimeConfig> for PythonWorkerSelector {
    fn select_worker(
        &self,
        workers: &HashMap<WorkerId, ModelRuntimeConfig>,
        request: &SchedulingRequest,
        eligibility: RoutingEligibility<'_>,
        block_size: u32,
    ) -> Result<RsWorkerSelectionResult, RsKvSchedulerError> {
        if workers.is_empty() {
            return Err(RsKvSchedulerError::NoEndpoints);
        }

        let start_time = std::time::Instant::now();
        let python_result = Python::with_gil(|py| {
            let py_workers = PyDict::new(py);
            for (worker_id, config) in workers {
                let config_value = pythonize(py, config)
                    .map_err(|e| format!("Failed to serialize worker config: {e}"))?;
                py_workers
                    .set_item(worker_id, config_value)
                    .map_err(|e| format!("Failed to set worker in dict: {e}"))?;
            }

            let decode_blocks = request
                .decode_blocks
                .iter()
                .map(|(worker, value)| ((worker.worker_id, worker.dp_rank), *value))
                .collect::<HashMap<(WorkerId, DpRank), usize>>();
            let prefill_tokens = request
                .prefill_tokens
                .iter()
                .map(|(worker, value)| ((worker.worker_id, worker.dp_rank), *value))
                .collect::<HashMap<(WorkerId, DpRank), usize>>();
            let effective_overlap_blocks = request
                .effective_overlap_blocks
                .iter()
                .map(|(worker, value)| ((worker.worker_id, worker.dp_rank), *value))
                .collect::<HashMap<(WorkerId, DpRank), f64>>();
            let effective_cached_tokens = request
                .effective_cached_tokens
                .iter()
                .map(|(worker, value)| ((worker.worker_id, worker.dp_rank), *value))
                .collect::<HashMap<(WorkerId, DpRank), usize>>();

            let py_request = PySchedulingRequest {
                request_id: request.maybe_request_id.clone(),
                isl_tokens: request.isl_tokens,
                block_size,
                pinned_worker: request
                    .pinned_worker
                    .map(|worker| (worker.worker_id, worker.dp_rank)),
                allowed_worker_ids: request
                    .allowed_worker_ids
                    .as_ref()
                    .map(|ids| ids.iter().copied().collect()),
                overlaps: pythonize(py, &effective_overlap_blocks)
                    .map_err(|e| format!("Failed to serialize overlaps: {e}"))?
                    .unbind(),
                effective_overlap_blocks: pythonize(py, &effective_overlap_blocks)
                    .map_err(|e| format!("Failed to serialize effective_overlap_blocks: {e}"))?
                    .unbind(),
                effective_cached_tokens: pythonize(py, &effective_cached_tokens)
                    .map_err(|e| format!("Failed to serialize effective_cached_tokens: {e}"))?
                    .unbind(),
                decode_blocks: pythonize(py, &decode_blocks)
                    .map_err(|e| format!("Failed to serialize decode_blocks: {e}"))?
                    .unbind(),
                prefill_tokens: pythonize(py, &prefill_tokens)
                    .map_err(|e| format!("Failed to serialize prefill_tokens: {e}"))?
                    .unbind(),
            };

            let selector = self
                .python_worker_selector
                .lock()
                .map_err(|_| "Python worker selector mutex poisoned".to_string())?;
            let result = selector
                .call1(py, (py_workers, py_request))
                .map_err(|e| format!("Python worker selector failed: {e}"))?;

            result
                .extract::<PyWorkerSelectionResult>(py)
                .map_err(|e| format!("Python selector must return PyWorkerSelectionResult: {e}"))
        });

        if start_time.elapsed() > std::time::Duration::from_millis(2) {
            tracing::info!(
                "Python worker selector took {:?} to execute",
                start_time.elapsed()
            );
        }

        let py_result = python_result.map_err(|err| {
            tracing::error!("Python worker selector error: {err}");
            RsKvSchedulerError::NoEndpoints
        })?;

        let selected_worker = WorkerWithDpRank::new(py_result.worker_id, py_result.dp_rank);
        eligibility
            .validate_worker_rank(workers, selected_worker)
            .map_err(|err| RsKvSchedulerError::InitFailed(err.to_string()))?;

        let required_blocks = request.isl_tokens.div_ceil(block_size as usize) as u64;
        let effective_overlap_blocks = request.effective_overlap_blocks_for(selected_worker);
        let cached_tokens = request.effective_cached_tokens_for(selected_worker);

        Ok(RsWorkerSelectionResult {
            worker: selected_worker,
            required_blocks,
            effective_overlap_blocks,
            cached_tokens,
            dp_strict_rank: true,
        })
    }
}
