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
use llm_rs::json_publisher::JsonPublisher;
use pyo3::exceptions::PyRuntimeError;
use serde_json::Value as JsonValue;

#[pyclass]
pub struct PyJsonPublisher {
    inner: JsonPublisher,
}

#[pymethods]
impl PyJsonPublisher {
    #[new]
    fn new(endpoint: Endpoint) -> PyResult<Self> {
        let runtime = pyo3_async_runtimes::tokio::get_runtime();

        // Extract component from endpoint (v1.0.0: Component is no longer exposed to Python)
        let component = endpoint.inner.component().clone();
        // Block on the async constructor of JsonPublisher
        let publisher = runtime.block_on(async { JsonPublisher::new(component, None) });

        Ok(PyJsonPublisher { inner: publisher })
    }

    /// Publish a JSON event using a string payload
    #[pyo3(signature = (json_str, topic))]
    fn publish(&self, json_str: &str, topic: &str) -> PyResult<()> {
        let runtime = pyo3_async_runtimes::tokio::get_runtime();

        let json_value: JsonValue = serde_json::from_str(json_str)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid JSON structure: {}", e)))?;

        // Execute the async publish method and wait for its completion
        runtime.block_on(async {
            self.inner
                .publish(topic, json_value)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
        })
    }
}
