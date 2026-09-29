// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Mutex;

use baseten_mm_client::{HttpEncoderConfig, MultiModalClient, ProductionBdnProxyRequired};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use crate::to_pyerr;

const EMBEDDING_ALREADY_SENT: &str = "MediaEmbedding was already sent to a worker";

/// Encoder `mm_kwargs` bytes held in Rust. Python sees only the length; the
/// bytes move into the worker request without a copy. Cannot be pickled or
/// deep-copied, so neither can objects that hold one.
#[pyclass(frozen)]
pub(crate) struct MediaEmbedding {
    bytes: Mutex<Option<Vec<u8>>>,
    len: usize,
}

impl MediaEmbedding {
    fn from_vec(bytes: Vec<u8>) -> Self {
        let len = bytes.len();
        Self {
            bytes: Mutex::new(Some(bytes)),
            len,
        }
    }

    fn is_sent(&self) -> bool {
        self.bytes.lock().unwrap().is_none()
    }

    fn take(&self) -> PyResult<Vec<u8>> {
        self.bytes
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| PyValueError::new_err(EMBEDDING_ALREADY_SENT))
    }
}

#[pymethods]
impl MediaEmbedding {
    #[new]
    fn new(data: &[u8]) -> Self {
        Self::from_vec(data.to_vec())
    }

    fn __len__(&self) -> usize {
        self.len
    }

    fn __repr__(&self) -> String {
        let state = if self.is_sent() { ", sent" } else { "" };
        format!("MediaEmbedding({} bytes{state})", self.len)
    }

    /// Copy of the payload as Python bytes.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.bytes.lock().unwrap();
        let bytes = bytes
            .as_deref()
            .ok_or_else(|| PyValueError::new_err(EMBEDDING_ALREADY_SENT))?;
        Ok(PyBytes::new(py, bytes))
    }
}

/// The value under string `key` in an `rmpv` map; `None` if `value` is not a map
/// or has no such key.
fn map_get_mut<'a>(value: &'a mut rmpv::Value, key: &str) -> Option<&'a mut rmpv::Value> {
    let rmpv::Value::Map(entries) = value else {
        return None;
    };
    entries
        .iter_mut()
        .find_map(|(candidate, value)| (candidate.as_str() == Some(key)).then_some(value))
}

/// Moves a binary `mm_response.mm_kwargs` out of an encoder response envelope.
fn take_mm_kwargs(response: &mut rmpv::Value) -> Option<Vec<u8>> {
    let kwargs = map_get_mut(map_get_mut(response, "mm_response")?, "mm_kwargs")?;
    match std::mem::replace(kwargs, rmpv::Value::Nil) {
        rmpv::Value::Binary(bytes) => Some(bytes),
        other => {
            *kwargs = other;
            None
        }
    }
}

/// Writes `blobs` into `mm_args.mm_kwargs[index]` of a worker request.
fn fill_mm_kwargs(request: &mut rmpv::Value, blobs: Vec<(usize, Vec<u8>)>) -> anyhow::Result<()> {
    let Some(rmpv::Value::Array(kwargs)) =
        map_get_mut(request, "mm_args").and_then(|args| map_get_mut(args, "mm_kwargs"))
    else {
        anyhow::bail!("worker_args.mm_args.mm_kwargs must be a list");
    };
    for (index, bytes) in blobs {
        let slot = kwargs
            .get_mut(index)
            .ok_or_else(|| anyhow::anyhow!("mm_kwargs index {index} out of range"))?;
        *slot = rmpv::Value::Binary(bytes);
    }
    Ok(())
}

type MmKwargsList<'py> = (Bound<'py, PyDict>, Bound<'py, PyDict>, Bound<'py, PyList>);

/// `(args, args["mm_args"], args["mm_args"]["mm_kwargs"])` when that shape is present.
fn mm_kwargs_list<'py>(args: &Bound<'py, PyAny>) -> PyResult<Option<MmKwargsList<'py>>> {
    let Ok(dict) = args.downcast::<PyDict>() else {
        return Ok(None);
    };
    let Some(mm_args) = dict.get_item("mm_args")? else {
        return Ok(None);
    };
    let Ok(mm_args) = mm_args.downcast_into::<PyDict>() else {
        return Ok(None);
    };
    let Some(kwargs) = mm_args.get_item("mm_kwargs")? else {
        return Ok(None);
    };
    let Ok(kwargs) = kwargs.downcast_into::<PyList>() else {
        return Ok(None);
    };
    Ok(Some((dict.clone(), mm_args, kwargs)))
}

/// `depythonize` for worker args that moves the bytes of each [`MediaEmbedding`]
/// in `mm_args.mm_kwargs` into the request. Does not modify the caller's dicts.
pub(crate) fn depythonize_worker_args(args: &Bound<'_, PyAny>) -> PyResult<rmpv::Value> {
    let py = args.py();
    let Some((dict, mm_args, kwargs)) = mm_kwargs_list(args)? else {
        return Ok(pythonize::depythonize(args)?);
    };
    let placeholders = PyList::empty(py);
    let mut embeddings = Vec::new();
    for (index, item) in kwargs.iter().enumerate() {
        match item.downcast_into::<MediaEmbedding>() {
            Ok(embedding) => {
                embeddings.push((index, embedding));
                placeholders.append(py.None())?;
            }
            Err(err) => placeholders.append(err.into_inner())?,
        }
    }
    if embeddings.is_empty() {
        return Ok(pythonize::depythonize(args)?);
    }
    let mm_args = mm_args.copy()?;
    mm_args.set_item("mm_kwargs", placeholders)?;
    let args = dict.copy()?;
    args.set_item("mm_args", mm_args)?;
    let mut request: rmpv::Value = pythonize::depythonize(&args)?;
    // Check all handles before taking any, so an error leaves them intact. The GIL
    // is held, so nothing can take one in between.
    if embeddings
        .iter()
        .any(|(_, embedding)| embedding.get().is_sent())
    {
        return Err(PyValueError::new_err(EMBEDDING_ALREADY_SENT));
    }
    let blobs = embeddings
        .into_iter()
        .map(|(index, embedding)| Ok((index, embedding.get().take()?)))
        .collect::<PyResult<Vec<_>>>()?;
    fill_mm_kwargs(&mut request, blobs).map_err(to_pyerr)?;
    Ok(request)
}

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
    #[pyo3(signature = (*, url, api_key, cache_urls=Vec::new(), proxy=None, request_timeout_s=300.0, max_retries=1, max_concurrent_requests=64))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        url: String,
        api_key: String,
        cache_urls: Vec<String>,
        proxy: Option<String>,
        request_timeout_s: f64,
        max_retries: u32,
        max_concurrent_requests: usize,
    ) -> PyResult<Self> {
        let transport = MultiModalClient::new(HttpEncoderConfig {
            url,
            api_key,
            cache_urls,
            proxy,
            request_timeout_s,
            max_retries,
            max_concurrent_requests,
        })
        .map_err(|error| {
            if error.downcast_ref::<ProductionBdnProxyRequired>().is_some() {
                PyValueError::new_err(error.to_string())
            } else {
                to_pyerr(error)
            }
        })?;
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
            let mut batch = transport
                .call_batch(requests)
                .await
                .map_err(encoder_error)?;
            let embeddings: Vec<_> = batch.responses.iter_mut().map(take_mm_kwargs).collect();
            Python::with_gil(|py| {
                let data = pythonize::pythonize(py, &batch.responses)?;
                for (index, bytes) in embeddings.into_iter().enumerate() {
                    if let Some(bytes) = bytes {
                        data.get_item(index)?
                            .get_item("mm_response")?
                            .set_item("mm_kwargs", Py::new(py, MediaEmbedding::from_vec(bytes))?)?;
                    }
                }
                let result = PyDict::new(py);
                result.set_item("data", data)?;
                result.set_item("individual_request_times", batch.individual_request_times)?;
                result.set_item("total_time", batch.total_time)?;
                result.set_item("num_cached", batch.num_cached)?;
                result.set_item("num_shared", batch.num_shared)?;
                Ok(result.unbind())
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(entries: Vec<(&str, rmpv::Value)>) -> rmpv::Value {
        rmpv::Value::Map(entries.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }

    #[test]
    fn take_mm_kwargs_moves_binary_and_keeps_other_payloads() {
        let mut response = map(vec![(
            "mm_response",
            map(vec![
                ("mm_hash", "h".into()),
                ("mm_kwargs", rmpv::Value::Binary(vec![1, 2, 3])),
            ]),
        )]);
        assert_eq!(take_mm_kwargs(&mut response), Some(vec![1, 2, 3]));
        assert_eq!(response["mm_response"]["mm_kwargs"], rmpv::Value::Nil);

        let mut base64 = map(vec![(
            "mm_response",
            map(vec![("mm_kwargs", "AAEC".into())]),
        )]);
        assert_eq!(take_mm_kwargs(&mut base64), None);
        assert_eq!(base64["mm_response"]["mm_kwargs"].as_str(), Some("AAEC"));

        let mut error = map(vec![("success", false.into())]);
        assert_eq!(take_mm_kwargs(&mut error), None);
    }

    #[test]
    fn fill_mm_kwargs_writes_binaries_at_indices() {
        let mut request = map(vec![(
            "mm_args",
            map(vec![(
                "mm_kwargs",
                rmpv::Value::Array(vec![rmpv::Value::Nil, "AAEC".into(), rmpv::Value::Nil]),
            )]),
        )]);
        fill_mm_kwargs(&mut request, vec![(0, vec![1]), (2, vec![2])]).unwrap();
        assert_eq!(
            request["mm_args"]["mm_kwargs"],
            rmpv::Value::Array(vec![
                rmpv::Value::Binary(vec![1]),
                "AAEC".into(),
                rmpv::Value::Binary(vec![2]),
            ])
        );
        assert!(fill_mm_kwargs(&mut request, vec![(3, vec![3])]).is_err());
        assert!(fill_mm_kwargs(&mut map(vec![]), vec![(0, vec![1])]).is_err());
    }
}
