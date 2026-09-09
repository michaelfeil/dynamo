// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use baseten_mm_client::{HttpEncoderConfig, MultiModalClient};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::to_pyerr;

pyo3::create_exception!(
    dynamo._core,
    EncoderHttpError,
    pyo3::exceptions::PyRuntimeError
);

fn encoder_error(error: anyhow::Error) -> PyErr {
    if let Some(status) = baseten_mm_client::http_error_status(&error) {
        Python::with_gil(|py| {
            let exception = EncoderHttpError::new_err(error.to_string());
            if let Err(error) = exception.value(py).setattr("status_code", status) {
                return error;
            }
            exception
        })
    } else {
        to_pyerr(error)
    }
}

#[pyclass]
#[derive(Clone)]
pub(crate) struct MultiModalEncoderClient {
    transport: MultiModalClient,
}

#[pymethods]
impl MultiModalEncoderClient {
    #[new]
    #[pyo3(signature = (*, url, api_key=None, cache_urls=Vec::new(), proxy=None, request_timeout_s=300.0, max_retries=1, max_concurrent_requests=64))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        url: String,
        api_key: Option<String>,
        cache_urls: Vec<String>,
        proxy: Option<String>,
        request_timeout_s: f64,
        max_retries: u32,
        max_concurrent_requests: usize,
    ) -> PyResult<Self> {
        let transport = MultiModalClient::new(HttpEncoderConfig {
            url,
            api_key: api_key.unwrap_or_default(),
            cache_urls,
            proxy,
            request_timeout_s,
            max_retries,
            max_concurrent_requests,
        })
        .map_err(to_pyerr)?;
        Ok(Self { transport })
    }

    fn call_batch<'py>(
        &self,
        py: Python<'py>,
        requests: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let requests: Vec<serde_json::Value> = pythonize::depythonize(requests)?;
        let transport = self.transport.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let batch = transport
                .call_batch(requests)
                .await
                .map_err(encoder_error)?;
            Python::with_gil(|py| {
                let result = PyDict::new(py);
                result.set_item("data", pythonize::pythonize(py, &batch.responses)?)?;
                result.set_item("individual_request_times", batch.individual_request_times)?;
                result.set_item("total_time", batch.total_time)?;
                result.set_item("num_cached", batch.num_cached)?;
                result.set_item("num_shared", batch.num_shared)?;
                Ok(result.unbind())
            })
        })
    }
}
