// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use baseten_parsers::{Call, Event, Tool, UnifiedStream, request_init};
use parking_lot::Mutex;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

pyo3::create_exception!(dynamo._core, ParserStreamError, PyRuntimeError);

#[pyclass(name = "ParserToolCall", module = "dynamo._core", frozen, get_all)]
#[derive(Clone)]
pub struct PyToolCall {
    tool_index: usize,
    id: Option<String>,
    name: Option<String>,
    arguments: String,
    complete: bool,
}

impl From<Call> for PyToolCall {
    fn from(call: Call) -> Self {
        Self {
            tool_index: call.tool_index,
            id: call.id,
            name: call.name,
            arguments: call.arguments,
            complete: call.complete,
        }
    }
}

#[pyclass(name = "ToolParseOutput", module = "dynamo._core", frozen, get_all)]
pub struct PyToolOutput {
    normal_text: String,
    calls: Vec<PyToolCall>,
}

#[pyclass(name = "ParserEvent", module = "dynamo._core", frozen, get_all)]
#[derive(Clone)]
pub struct PyEvent {
    kind: &'static str,
    text: Option<String>,
    call: Option<PyToolCall>,
}

impl From<Event> for PyEvent {
    fn from(event: Event) -> Self {
        match event {
            Event::Text(text) => Self {
                kind: "text",
                text: Some(text),
                call: None,
            },
            Event::Reasoning(text) => Self {
                kind: "reasoning",
                text: Some(text),
                call: None,
            },
            Event::ToolCall(call) => Self {
                kind: "tool_call",
                text: None,
                call: Some(call.into()),
            },
        }
    }
}

fn tools_from_python(tools: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<Tool>> {
    tools.map_or_else(
        || Ok(Vec::new()),
        |value| pythonize::depythonize(value).map_err(|e| PyValueError::new_err(e.to_string())),
    )
}

fn value_error(error: anyhow::Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn stream_error(py: Python<'_>, error: anyhow::Error, events: Vec<PyEvent>) -> PyErr {
    let exception = ParserStreamError::new_err(error.to_string());
    // Already committed events cannot be discarded when a later input fragment fails.
    if let Err(error) = exception.value(py).setattr("events", events) {
        return error;
    }
    exception
}

#[pyclass(name = "ToolCallStream", module = "dynamo._core")]
pub struct PyToolStream;

#[pymethods]
impl PyToolStream {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Self> {
        Err(PyRuntimeError::new_err(
            "ToolCallStream was removed; use UnifiedParserStream",
        ))
    }
}

#[pyclass(name = "UnifiedParserStream", module = "dynamo._core")]
pub struct PyUnifiedStream {
    inner: Mutex<UnifiedStream>,
}

#[pymethods]
impl PyUnifiedStream {
    #[new]
    #[pyo3(signature = (family, tools=None, *, prompt_token_ids=None, starting_state="none", tool_output_mode="native", named_tool=None, invalid_guided_payload="reject", backend="dynamo", tokenizer_path=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        family: String,
        tools: Option<&Bound<'_, PyAny>>,
        prompt_token_ids: Option<Vec<u32>>,
        starting_state: &str,
        tool_output_mode: &str,
        named_tool: Option<String>,
        invalid_guided_payload: &str,
        backend: &str,
        tokenizer_path: Option<String>,
    ) -> PyResult<Self> {
        let tools = tools_from_python(tools)?;
        let init = request_init(
            prompt_token_ids.unwrap_or_default(),
            starting_state,
            tool_output_mode,
            named_tool,
            invalid_guided_payload,
        )
        .map_err(value_error)?;
        let inner = py
            .allow_threads(|| {
                UnifiedStream::new_with_backend(
                    backend,
                    &family,
                    &tools,
                    init,
                    tokenizer_path.as_deref().map(std::path::Path::new),
                )
            })
            .map_err(value_error)?;
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    #[getter]
    fn preserve_special_tokens(&self) -> bool {
        self.inner.lock().preserve_special_tokens()
    }

    fn step(&self, py: Python<'_>, text: String) -> PyResult<Vec<PyEvent>> {
        self.advance(py, Some(&text))
    }

    fn finish(&self, py: Python<'_>) -> PyResult<Vec<PyEvent>> {
        self.advance(py, None)
    }
}

impl PyUnifiedStream {
    fn advance(&self, py: Python<'_>, text: Option<&str>) -> PyResult<Vec<PyEvent>> {
        py.allow_threads(|| self.inner.lock().advance(text))
            .map(|events| events.into_iter().map(Into::into).collect())
            .map_err(|error| {
                stream_error(
                    py,
                    error.error,
                    error.events.into_iter().map(Into::into).collect(),
                )
            })
    }
}

pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyToolCall>()?;
    m.add_class::<PyToolOutput>()?;
    m.add_class::<PyEvent>()?;
    m.add_class::<PyToolStream>()?;
    m.add_class::<PyUnifiedStream>()?;
    m.add("ParserStreamError", m.py().get_type::<ParserStreamError>())?;
    m.add(
        "VLLM_UNIFIED_PARSER_FAMILIES",
        baseten_parsers::vllm::FAMILIES.to_vec(),
    )?;
    m.add(
        "VLLM_PARSER_UPSTREAM_REVISION",
        baseten_parsers::vllm::UPSTREAM_REVISION,
    )?;
    m.add(
        "UNIFIED_PARSER_FAMILIES",
        baseten_parsers::unified_parser_families(),
    )?;
    m.add(
        "PARSER_UPSTREAM_REVISION",
        baseten_parsers::UPSTREAM_REVISION,
    )?;
    Ok(())
}
