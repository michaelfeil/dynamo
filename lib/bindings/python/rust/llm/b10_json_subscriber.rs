// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::*;
use crate::Endpoint;
use llm_rs::json_subscriber::JsonSubscriber;
use pyo3::exceptions::{PyRuntimeError, PyStopAsyncIteration};
use pyo3_async_runtimes::tokio::future_into_py;
use std::sync::Arc;

/// Python-visible wrapper that is an *async iterator* over a JsonSubscriber stream.
/// Provides python with JSON strings, nothing fancy.
#[pyclass]
pub struct JsonSubscriberIter {
    subscriber: Arc<tokio::sync::Mutex<JsonSubscriber>>,
}

#[pymethods]
impl JsonSubscriberIter {
    #[new]
    fn new(endpoint: Endpoint, topic: &str) -> PyResult<Self> {
        let runtime = pyo3_async_runtimes::tokio::get_runtime();
        let component = endpoint.inner.component().clone();

        // Block on the async constructor of JsonSubscriber
        let subscriber = runtime.block_on(async {
            JsonSubscriber::new(component, topic, None)
                .await
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to create JsonSubscriber: {}", e))
                })
        })?;

        Ok(JsonSubscriberIter {
            subscriber: Arc::new(tokio::sync::Mutex::new(subscriber)),
        })
    }

    /// Make it usable in `async for`.
    fn __aiter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    /// Each await pulls exactly one JSON string from the Rust side.
    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let subscriber = self.subscriber.clone();
        future_into_py(py, async move {
            let next_value = {
                let mut subscriber_guard = subscriber.lock().await;
                subscriber_guard.next().await
            };
            match next_value {
                Some(json_value) => {
                    // Convert JsonValue to JSON string
                    let json_string = serde_json::to_string(&json_value).map_err(|e| {
                        PyRuntimeError::new_err(format!("Failed to serialize JSON: {}", e))
                    })?;
                    Ok(json_string)
                }
                None => Err(PyStopAsyncIteration::new_err("Stream ended")),
            }
        })
    }

    fn next_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.__anext__(py)
    }

    /// Shutdown the subscriber
    fn shutdown(&self) {
        let runtime = pyo3_async_runtimes::tokio::get_runtime();
        let subscriber = self.subscriber.clone();

        runtime.block_on(async {
            let mut guard = subscriber.lock().await;
            guard.shutdown();
        });
    }
}

impl Drop for JsonSubscriberIter {
    fn drop(&mut self) {
        self.shutdown();
    }
}
