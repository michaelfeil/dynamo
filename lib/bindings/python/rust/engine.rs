// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::{Error, Result};
use futures::{StreamExt as FuturesStreamExt, stream};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3::{PyAny, PyErr};
use pyo3_async_runtimes::TaskLocals;
use pythonize::{depythonize, pythonize};
pub use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::error::{BackendError, DynamoError, ErrorType};
use dynamo_runtime::logging::get_distributed_tracing_context;
pub use dynamo_runtime::{
    pipeline::{AsyncEngine, AsyncEngineContextProvider, Data, ManyOut, ResponseStream, SingleIn},
    protocols::{annotated::Annotated, maybe_error::MaybeError},
};

use super::context::{Context, callable_accepts_kwarg};
use super::errors::py_exception_to_backend_error;

/// Add bindings from this crate to the provided module
pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PythonAsyncEngine>()?;
    Ok(())
}
// todos:
// - [ ] enable context cancellation
//   - this will likely require a change to the function signature python calling arguments
// - [ ] other `AsyncEngine` implementations will have a similar pattern, i.e. one AsyncEngine
//       implementation per struct

/// Rust/Python bridge that maps to the [`AsyncEngine`] trait
///
/// Currently this is only implemented for the [`SingleIn`] and [`ManyOut`] types; however,
/// more [`AsyncEngine`] implementations can be added in the future.
///
/// For the [`SingleIn`] and [`ManyOut`] case, this implementation will take a Python async
/// generator and convert it to a Rust async stream.
///
/// ```python
/// class ComputeEngine:
///     def __init__(self):
///         self.compute_engine = make_compute_engine()
///
///     def generate(self, request):
///         async generator():
///            async for output in self.compute_engine.generate(request):
///                yield output
///         return generator()
///
/// def main():
///     loop = asyncio.create_event_loop()
///     compute_engine = ComputeEngine()
///     engine = PythonAsyncEngine(compute_engine.generate, loop)
///     service = RustService()
///     service.add_engine("model_name", engine)
///     loop.run_until_complete(service.run())
/// ```
#[pyclass]
#[derive(Clone)]
pub struct PythonAsyncEngine(PythonServerStreamingEngine);

impl PythonAsyncEngine {
    pub fn set_logging_label(&mut self, label: impl AsRef<str>) {
        self.0.set_logging_label(label);
    }
}

#[pymethods]
impl PythonAsyncEngine {
    /// Create a new instance of the PythonAsyncEngine
    ///
    /// # Arguments
    /// - `generator`: a Python async generator that will be used to generate responses
    /// - `event_loop`: the Python event loop that will be used to run the generator
    ///
    /// Note: In Rust land, the request and the response are both concrete; however, in
    /// Python land, the request and response not strongly typed, meaning the generator
    /// could accept a different type of request or return a different type of response
    /// and we would not know until runtime.
    #[new]
    pub fn new(generator: PyObject, event_loop: PyObject) -> PyResult<Self> {
        let cancel_token = CancellationToken::new();
        Ok(PythonAsyncEngine(PythonServerStreamingEngine::new(
            cancel_token,
            Arc::new(generator),
            Arc::new(event_loop),
        )))
    }

    pub fn block_until_stream_item(&mut self, enabled: bool) {
        self.0.block_until_stream_item(enabled);
    }

    #[pyo3(name = "set_logging_label")]
    pub fn set_logging_label_py(&mut self, label: String) {
        self.set_logging_label(label);
    }
}

#[async_trait::async_trait]
impl<Req, Resp> AsyncEngine<SingleIn<Req>, ManyOut<Annotated<Resp>>, Error> for PythonAsyncEngine
where
    Req: Data + Serialize,
    Resp: Data + for<'de> Deserialize<'de>,
{
    async fn generate(&self, request: SingleIn<Req>) -> Result<ManyOut<Annotated<Resp>>, Error> {
        self.0.generate(request).await
    }
}

#[derive(Clone)]
pub struct PythonServerStreamingEngine {
    _cancel_token: CancellationToken,
    generator: Arc<PyObject>,
    event_loop: Arc<PyObject>,
    has_context: bool,
    block_until_stream_item: bool,
    logging_label: String,
}

impl PythonServerStreamingEngine {
    pub fn new(
        cancel_token: CancellationToken,
        generator: Arc<PyObject>,
        event_loop: Arc<PyObject>,
    ) -> Self {
        let has_context = Python::with_gil(|py| {
            let callable = generator.bind(py);
            callable_accepts_kwarg(py, callable, "context").unwrap_or(false)
        });

        PythonServerStreamingEngine {
            _cancel_token: cancel_token,
            generator,
            event_loop,
            has_context,
            block_until_stream_item: false,
            logging_label: "Worker".to_string(),
        }
    }

    pub fn block_until_stream_item(&mut self, enabled: bool) {
        self.block_until_stream_item = enabled;
    }

    pub fn set_logging_label(&mut self, logging_label: impl AsRef<str>) {
        self.logging_label = logging_label.as_ref().to_string();
    }
}

#[derive(Debug, thiserror::Error)]
enum ResponseProcessingError {
    #[error("dynamo error")]
    Dynamo(DynamoError),

    #[error("deserialize error: {0}")]
    Deserialize(String),

    #[error("gil offload error: {0}")]
    Offload(String),
}

#[async_trait::async_trait]
impl<Req, Resp> AsyncEngine<SingleIn<Req>, ManyOut<Annotated<Resp>>, Error>
    for PythonServerStreamingEngine
where
    Req: Data + Serialize,
    Resp: Data + for<'de> Deserialize<'de>,
{
    async fn generate(&self, request: SingleIn<Req>) -> Result<ManyOut<Annotated<Resp>>, Error> {
        // Create a context
        let (request, context) = request.transfer(());
        let ctx = context.context();

        let id = context.id().to_string();
        tracing::trace!("processing request: {}", id);

        // Capture current trace context
        let current_trace_context = get_distributed_tracing_context();

        // Clone the PyObject to move into the thread

        // Create a channel to communicate between the Python thread and the Rust async context
        let (tx, rx) = mpsc::channel::<Annotated<Resp>>(128);

        let generator = self.generator.clone();
        let event_loop = self.event_loop.clone();
        let ctx_python = ctx.clone();
        let has_context = self.has_context;
        let metadata = context.metadata().clone();
        let logging_label = self.logging_label.clone();

        // Acquiring the GIL is similar to acquiring a standard lock/mutex
        // Performing this in an tokio async task could block the thread for an undefined amount of time
        // To avoid this, we spawn a blocking task to acquire the GIL and perform the operations needed
        // while holding the GIL.
        //
        // Under low GIL contention, we wouldn't need to do this.
        // However, under high GIL contention, this can lead to significant performance degradation.
        //
        // Since we cannot predict the GIL contention, we will always use the blocking task and pay the
        // cost. The Python GIL is the gift that keeps on giving -- performance hits...
        let stream = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| {
                let py_request = pythonize(py, &request)?;
                let id = ctx_python.id().to_string();

                // Create context with trace information
                let py_ctx = Py::new(
                    py,
                    Context::new(ctx_python.clone(), current_trace_context, None, metadata),
                )?;

                let gen_result = if has_context {
                    // Pass context as a kwarg
                    let kwarg = PyDict::new(py);
                    kwarg.set_item("context", &py_ctx)?;
                    generator.call(py, (py_request,), Some(&kwarg))
                } else {
                    // Legacy: No `context` arg
                    generator.call1(py, (py_request,))
                }?;
                tracing::info!(
                    unified_model_logs = true,
                    "{}: Processing request {id}",
                    logging_label
                );

                let locals = TaskLocals::new(event_loop.bind(py).clone());
                pyo3_async_runtimes::tokio::into_stream_with_locals_v1(
                    locals,
                    gen_result.into_bound(py),
                )
            })
        })
        .await??;

        // process the stream
        // any error thrown in the stream will be caught and complete the processing task
        // errors are captured by a task that is watching the processing task
        // the error will be emitted as an annotated error
        let request_id = id.clone();
        let mut stream = Box::pin(stream);

        let stream = if self.block_until_stream_item {
            let first_item = match FuturesStreamExt::next(&mut stream).await {
                Some(Ok(item)) => item,
                Some(Err(e)) => {
                    if e.to_string().contains("CancelledError") {
                        tracing::info!(
                            request_id,
                            "Python async generator stream cancelled before first iteration"
                        );
                    } else {
                        tracing::warn!(
                            unified_model_logs = true,
                            request_id,
                            "Python exception occurred before finish of first iteration: {}",
                            e
                        );
                    }
                    return Err(Error::new(e));
                }
                None => {
                    tracing::warn!(
                        request_id,
                        "Python async generator stream ended before processing started"
                    );
                    return Err(Error::new(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "Python async generator stream ended before processing started",
                    )));
                }
            };
            stream::once(async move { Ok(first_item) })
                .chain(stream)
                .boxed()
        } else {
            stream
        };

        tokio::spawn(async move {
            tracing::debug!(
                request_id,
                "starting task to process python async generator stream"
            );

            let mut stream = stream;
            let mut count = 0;

            while let Some(item) = FuturesStreamExt::next(&mut stream).await {
                count += 1;
                tracing::trace!(
                    request_id,
                    "processing the {}th item from python async generator",
                    count
                );

                let mut done = false;

                let response = match process_item::<Resp>(item).await {
                    Ok(response) => response,
                    Err(e) => {
                        done = true;

                        match e {
                            ResponseProcessingError::Deserialize(e) => {
                                // tell the python async generator to stop generating
                                // right now, this is impossible as we are not passing the context to the python async generator
                                // todo: add task-local context to the python async generator
                                ctx.stop_generating();
                                Annotated::from_error(format!(
                                    "critical error: invalid response object from python async generator; application-logic-mismatch: {}",
                                    e
                                ))
                            }
                            ResponseProcessingError::Dynamo(dynamo_err) => {
                                Annotated::from_err(dynamo_err)
                            }
                            ResponseProcessingError::Offload(e) => Annotated::from_error(format!(
                                "critical error: failed to offload the python async generator to a new thread: {}",
                                e
                            )),
                        }
                    }
                };

                if tx.send(response).await.is_err() {
                    tracing::trace!(
                        request_id,
                        "error forwarding annotated response to channel; channel is closed"
                    );
                    break;
                }

                if done {
                    tracing::debug!(
                        request_id,
                        "early termination of python async generator stream task"
                    );
                    break;
                }
            }

            tracing::debug!(
                request_id,
                "finished processing python async generator stream"
            );
        });

        let stream = ReceiverStream::new(rx);

        Ok(ResponseStream::new(Box::pin(stream), context.context()))
    }
}

async fn process_item<Resp>(
    item: Result<Py<PyAny>, PyErr>,
) -> Result<Annotated<Resp>, ResponseProcessingError>
where
    Resp: Data + for<'de> Deserialize<'de>,
{
    let item = item.map_err(|e| {
        Python::with_gil(|py| {
            e.display(py);

            // Check if the Python exception is a Dynamo error type.
            // Wrap as Backend* since this is the backend engine context.
            if let Some((backend_err, message)) = py_exception_to_backend_error(py, &e) {
                return ResponseProcessingError::Dynamo(
                    DynamoError::builder()
                        .error_type(ErrorType::Backend(backend_err))
                        .message(message)
                        .build(),
                );
            }

            // GeneratorExit from Python's generator protocol (e.g., GC closing
            // a generator) is treated as an engine shutdown.
            if e.is_instance_of::<pyo3::exceptions::PyGeneratorExit>(py) {
                return ResponseProcessingError::Dynamo(
                    DynamoError::builder()
                        .error_type(ErrorType::Backend(BackendError::EngineShutdown))
                        .message("engine shutting down")
                        .build(),
                );
            }

            // Map well-known Python exceptions to specific Backend error types.
            // Order matters: check subclasses before their parents
            // (e.g., ConnectionRefusedError before ConnectionError).
            let backend_err = if e.is_instance_of::<pyo3::exceptions::PyValueError>(py)
                || e.is_instance_of::<pyo3::exceptions::PyTypeError>(py)
            {
                BackendError::InvalidArgument
            } else if e.is_instance_of::<pyo3::exceptions::PyTimeoutError>(py) {
                BackendError::ConnectionTimeout
            } else if e.is_instance_of::<pyo3::exceptions::PyConnectionRefusedError>(py) {
                BackendError::CannotConnect
            } else if e.is_instance_of::<pyo3::exceptions::PyConnectionResetError>(py)
                || e.is_instance_of::<pyo3::exceptions::PyBrokenPipeError>(py)
                || e.is_instance_of::<pyo3::exceptions::PyConnectionError>(py)
            {
                BackendError::Disconnected
            } else if e.is_instance_of::<pyo3::exceptions::asyncio::CancelledError>(py) {
                BackendError::Cancelled
            } else {
                BackendError::Unknown
            };

            ResponseProcessingError::Dynamo(
                DynamoError::builder()
                    .error_type(ErrorType::Backend(backend_err))
                    .message(e.to_string())
                    .build(),
            )
        })
    })?;
    let response = tokio::task::spawn_blocking(move || {
        Python::with_gil(|py| depythonize::<Resp>(&item.into_bound(py)))
    })
    .await
    .map_err(|e| ResponseProcessingError::Offload(e.to_string()))?
    .map_err(|e| ResponseProcessingError::Deserialize(e.to_string()))?;

    let response = Annotated::from_data(response);

    Ok(response)
}
