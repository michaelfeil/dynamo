// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use baseten_parsers::{
    Call, Event, Tool, ToolParserInput, ToolStream, UnifiedStream, request_init,
};
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
pub struct PyToolStream {
    inner: Mutex<ToolStream>,
}

#[pymethods]
impl PyToolStream {
    #[new]
    #[pyo3(signature = (family, tools=None, *, backend="dynamo"))]
    fn new(
        py: Python<'_>,
        family: String,
        tools: Option<&Bound<'_, PyAny>>,
        backend: &str,
    ) -> PyResult<Self> {
        let tools = tools_from_python(tools)?;
        let inner = py
            .allow_threads(|| ToolStream::new(backend, &family, &tools))
            .map_err(value_error)?;
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    #[getter]
    fn preserve_special_tokens(&self) -> bool {
        self.inner.lock().preserve_special_tokens()
    }

    #[getter]
    fn prefers_tokens(&self) -> bool {
        self.inner.lock().prefers_tokens()
    }

    #[getter]
    fn completion_semantics(&self) -> &'static str {
        self.inner.lock().completion_semantics()
    }

    fn step(&self, py: Python<'_>, text: String) -> PyResult<PyToolOutput> {
        self.advance(py, Some(ToolParserInput::Text(&text)))
    }

    fn step_tokens(&self, py: Python<'_>, token_ids: Vec<u32>) -> PyResult<PyToolOutput> {
        self.advance(py, Some(ToolParserInput::Tokens(&token_ids)))
    }

    fn finish(&self, py: Python<'_>) -> PyResult<PyToolOutput> {
        self.advance(py, None)
    }
}

impl PyToolStream {
    fn advance(
        &self,
        py: Python<'_>,
        input: Option<ToolParserInput<'_>>,
    ) -> PyResult<PyToolOutput> {
        py.allow_threads(|| self.inner.lock().advance(input))
            .map(|output| PyToolOutput {
                normal_text: output.normal_text,
                calls: output.calls.into_iter().map(Into::into).collect(),
            })
            .map_err(|error| {
                stream_error(
                    py,
                    error.error,
                    error.events.into_iter().map(Into::into).collect(),
                )
            })
    }
}

#[pyclass(name = "UnifiedParserStream", module = "dynamo._core")]
pub struct PyUnifiedStream {
    inner: Mutex<UnifiedStream>,
}

#[pymethods]
impl PyUnifiedStream {
    #[new]
    #[pyo3(signature = (family, tools=None, *, prompt_token_ids=None, starting_state="none", tool_output_mode="native", named_tool=None, invalid_guided_payload="reject"))]
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
            .allow_threads(|| UnifiedStream::new(&family, &tools, init))
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

#[pyclass(
    name = "ReasoningParseOutput",
    module = "dynamo._core",
    frozen,
    get_all
)]
pub struct PyReasoningOutput {
    normal_text: String,
    reasoning_text: String,
}

impl From<baseten_parsers::ReasoningOutput> for PyReasoningOutput {
    fn from(output: baseten_parsers::ReasoningOutput) -> Self {
        Self {
            normal_text: output.normal_text,
            reasoning_text: output.reasoning_text,
        }
    }
}

#[pyclass(name = "ReasoningParserStream", module = "dynamo._core")]
pub struct PyReasoningStream {
    inner: Mutex<baseten_parsers::ReasoningStream>,
}

#[pymethods]
impl PyReasoningStream {
    #[new]
    #[pyo3(signature = (family, *, in_reasoning=None))]
    fn new(py: Python<'_>, family: String, in_reasoning: Option<bool>) -> PyResult<Self> {
        let parser = py
            .allow_threads(|| baseten_parsers::ReasoningStream::new(&family, in_reasoning))
            .map_err(value_error)?;
        Ok(Self {
            inner: Mutex::new(parser),
        })
    }

    #[pyo3(signature = (text, token_ids=None))]
    fn step(
        &self,
        py: Python<'_>,
        text: &str,
        token_ids: Option<Vec<u32>>,
    ) -> PyResult<PyReasoningOutput> {
        py.allow_threads(|| {
            self.inner
                .lock()
                .step(text, token_ids.as_deref().unwrap_or_default())
        })
        .map(Into::into)
        .map_err(|error| stream_error(py, error, Vec::new()))
    }

    fn finish(&self, py: Python<'_>) -> PyResult<PyReasoningOutput> {
        py.allow_threads(|| self.inner.lock().finish())
            .map(Into::into)
            .map_err(|error| stream_error(py, error, Vec::new()))
    }
}

pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyReasoningOutput>()?;
    m.add_class::<PyReasoningStream>()?;
    let mut families = baseten_parsers::reasoning_parser_families();
    families.sort_unstable();
    m.add("REASONING_PARSER_FAMILIES", families)?;
    m.add_class::<PyToolCall>()?;
    m.add_class::<PyToolOutput>()?;
    m.add_class::<PyEvent>()?;
    m.add_class::<PyToolStream>()?;
    m.add_class::<PyUnifiedStream>()?;
    m.add("ParserStreamError", m.py().get_type::<ParserStreamError>())?;
    m.add(
        "VLLM_TOOL_PARSER_FAMILIES",
        baseten_parsers::vllm::FAMILIES.to_vec(),
    )?;
    m.add(
        "VLLM_PARSER_UPSTREAM_REVISION",
        baseten_parsers::vllm::UPSTREAM_REVISION,
    )?;
    m.add(
        "TOOL_PARSER_FAMILIES",
        baseten_parsers::REGISTERED_FAMILIES.to_vec(),
    )?;
    m.add(
        "UNIFIED_PARSER_FAMILIES",
        baseten_parsers::REGISTERED_UNIFIED_FAMILIES.to_vec(),
    )?;
    m.add(
        "PARSER_UPSTREAM_REVISION",
        baseten_parsers::UPSTREAM_REVISION,
    )?;
    Ok(())
}
