// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Request-scoped lifecycle guards around the pinned upstream parser registries.

use anyhow::{Result, bail, ensure};
pub use dynamo_parsers_v2 as upstream;
pub use upstream::{
    InvalidGuidedPayloadPolicy, REGISTERED_FAMILIES, REGISTERED_UNIFIED_FAMILIES, Tool,
    ToolParseResult, ToolParser, ToolParserInput, UnifiedParser, UnifiedParserInit,
    UnifiedParserOutput, UnifiedParserStartingState, UnifiedToolOutputMode,
};

pub const UPSTREAM_REVISION: &str = "23b402787dab4c1a859c07488cce42927362db3e";

/// Normalized call delta, including any identifier supplied by the backend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Call {
    pub tool_index: usize,
    pub id: Option<String>,
    pub name: Option<String>,
    pub arguments: String,
    pub complete: bool,
}

impl Call {
    fn new(call: upstream::ToolCallDelta, id: Option<&str>) -> Self {
        Self {
            tool_index: call.tool_index,
            id: id.map(str::to_owned),
            name: call.name,
            arguments: call.arguments,
            complete: call.complete,
        }
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct ToolOutput {
    pub normal_text: String,
    pub calls: Vec<Call>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    Text(String),
    Reasoning(String),
    ToolCall(Call),
}

/// Events committed before a parsing failure must remain visible to the caller.
#[derive(Debug)]
pub struct StreamError {
    pub error: anyhow::Error,
    pub events: Vec<Event>,
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)
    }
}

impl std::error::Error for StreamError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.error.as_ref())
    }
}

/// Shared validation for string-based callers, including language bindings.
pub fn request_init(
    prompt_token_ids: Vec<u32>,
    starting_state: &str,
    tool_output_mode: &str,
    named_tool: Option<String>,
    invalid_guided_payload: &str,
) -> Result<UnifiedParserInit> {
    let starting_state = match starting_state {
        "none" => UnifiedParserStartingState::None,
        "reasoning" => UnifiedParserStartingState::Reasoning,
        "response" => UnifiedParserStartingState::Response,
        _ => bail!("starting_state must be none, reasoning, or response"),
    };
    let tool_output_mode = match tool_output_mode {
        "native" if named_tool.is_none() => UnifiedToolOutputMode::Native,
        "guided_json" => UnifiedToolOutputMode::GuidedJson { named_tool },
        _ => bail!("use native without named_tool, or guided_json"),
    };
    let invalid_guided_payload = match invalid_guided_payload {
        "reject" => InvalidGuidedPayloadPolicy::Reject,
        "recover_as_text" => InvalidGuidedPayloadPolicy::RecoverAsText,
        "stream_best_effort" => InvalidGuidedPayloadPolicy::StreamBestEffort,
        _ => bail!("unknown invalid_guided_payload policy"),
    };
    Ok(UnifiedParserInit {
        prompt_token_ids,
        starting_state,
        tool_output_mode,
        invalid_guided_payload,
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum InputMode {
    Text,
    Tokens,
}

pub struct ToolCallStream {
    parser: Box<dyn upstream::ToolParser>,
    mode: Option<InputMode>,
    closed: bool,
}

impl ToolCallStream {
    pub fn new(family: &str, tools: &[Tool]) -> Result<Self> {
        Ok(Self::from_parser(upstream::create_tool_parser_for_family(
            family, tools,
        )?))
    }

    /// Wrap another backend implementing the peer-shaped `ToolParser` contract.
    pub fn from_parser(parser: Box<dyn ToolParser>) -> Self {
        Self {
            parser,
            mode: None,
            closed: false,
        }
    }

    pub fn preserve_special_tokens(&self) -> bool {
        self.parser.preserve_special_tokens()
    }

    pub fn prefers_tokens(&self) -> bool {
        self.parser.prefers_tokens()
    }

    pub fn tool_call_id(&self, index: usize) -> Option<&str> {
        self.parser.tool_call_id(index)
    }

    pub fn step(&mut self, input: ToolParserInput<'_>) -> Result<ToolParseResult> {
        ensure!(!self.closed, "parser stream is closed");
        let mode = match input {
            ToolParserInput::Text(_) => InputMode::Text,
            ToolParserInput::Tokens(_) => {
                // Upstream's default push_tokens silently returns no output.
                ensure!(
                    self.prefers_tokens(),
                    "this parser does not support token input"
                );
                InputMode::Tokens
            }
        };
        ensure!(
            self.mode.is_none_or(|previous| previous == mode),
            "cannot mix text and token input"
        );
        self.mode = Some(mode);
        let result = self.parser.push_input(input);
        if result.is_err() {
            self.closed = true;
        }
        result
    }

    pub fn finish(&mut self) -> Result<ToolParseResult> {
        ensure!(!self.closed, "parser stream is closed");
        self.closed = true;
        self.parser.finish()
    }

    /// Advance or finalize (`None`), resolving model-supplied IDs in Rust.
    pub fn advance(&mut self, input: Option<ToolParserInput<'_>>) -> Result<ToolOutput> {
        let output = match input {
            Some(input) => self.step(input)?,
            None => self.finish()?,
        };
        Ok(ToolOutput {
            normal_text: output.normal_text,
            calls: output
                .calls
                .into_iter()
                .map(|call| {
                    let index = call.tool_index;
                    Call::new(call, self.tool_call_id(index))
                })
                .collect(),
        })
    }
}

pub struct UnifiedStream {
    parser: Box<dyn upstream::UnifiedParser>,
    closed: bool,
}

impl UnifiedStream {
    pub fn new(family: &str, tools: &[Tool], init: UnifiedParserInit) -> Result<Self> {
        Self::from_parser(
            upstream::create_unified_parser_for_family(family, tools)?,
            init,
        )
    }

    /// Wrap another backend without changing lifecycle or binding code.
    pub fn from_parser(
        mut parser: Box<dyn UnifiedParser>,
        init: UnifiedParserInit,
    ) -> Result<Self> {
        parser.initialize_request(init)?;
        Ok(Self {
            parser,
            closed: false,
        })
    }

    pub fn preserve_special_tokens(&self) -> bool {
        self.parser.preserve_special_tokens()
    }

    pub fn tool_call_id(&self, index: usize) -> Option<&str> {
        self.parser.tool_call_id(index)
    }

    /// Committed events remain in `output` even if upstream fails later in this step.
    pub fn step(&mut self, text: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        ensure!(!self.closed, "parser stream is closed");
        let result = self.parser.parse_into(text, output);
        if result.is_err() {
            self.closed = true;
        }
        result
    }

    pub fn finish(&mut self) -> Result<UnifiedParserOutput> {
        ensure!(!self.closed, "parser stream is closed");
        self.closed = true;
        self.parser.finish()
    }

    /// Advance or finalize (`None`), retaining event order and partial failures.
    pub fn advance(&mut self, text: Option<&str>) -> std::result::Result<Vec<Event>, StreamError> {
        let mut output = UnifiedParserOutput::default();
        let result = match text {
            Some(text) => self.step(text, &mut output),
            None => self.finish().map(|tail| output = tail),
        };
        let events = output
            .events
            .into_iter()
            .map(|event| match event {
                upstream::UnifiedParserEvent::Text(text) => Event::Text(text),
                upstream::UnifiedParserEvent::Reasoning(text) => Event::Reasoning(text),
                upstream::UnifiedParserEvent::ToolCall(call) => {
                    let index = call.tool_index;
                    Event::ToolCall(Call::new(call, self.tool_call_id(index)))
                }
            })
            .collect();
        match result {
            Ok(()) => Ok(events),
            Err(error) => Err(StreamError { error, events }),
        }
    }
}

pub use dynamo_parsers::reasoning::{
    ParserResult as ReasoningOutput, ReasoningParser,
    get_available_reasoning_parsers as reasoning_parser_families,
};

/// Standalone reasoning extraction using Dynamo's existing model grammars.
/// One instance belongs to one response choice; finalization is terminal.
pub struct ReasoningStream {
    parser: Box<dyn ReasoningParser>,
    closed: bool,
}

impl ReasoningStream {
    /// `None` retains the model default; `Some` overrides prompt reasoning state.
    pub fn new(family: &str, in_reasoning: Option<bool>) -> Result<Self> {
        let family = family.to_lowercase();
        ensure!(
            reasoning_parser_families().contains(&family.as_str()),
            "unknown reasoning parser: {family}"
        );
        let parser =
            dynamo_parsers::reasoning::ReasoningParserType::get_reasoning_parser_from_name(&family);
        Ok(Self::from_parser(Box::new(parser), in_reasoning))
    }

    pub fn from_parser(mut parser: Box<dyn ReasoningParser>, in_reasoning: Option<bool>) -> Self {
        if let Some(state) = in_reasoning {
            parser.set_in_reasoning(state);
        }
        Self {
            parser,
            closed: false,
        }
    }

    /// Text and its corresponding token IDs describe the same incremental chunk.
    pub fn step(&mut self, text: &str, token_ids: &[u32]) -> Result<ReasoningOutput> {
        ensure!(!self.closed, "parser stream is closed");
        Ok(self
            .parser
            .parse_reasoning_streaming_incremental(text, token_ids))
    }

    pub fn finish(&mut self) -> Result<ReasoningOutput> {
        ensure!(!self.closed, "parser stream is closed");
        self.closed = true;
        Ok(self.parser.finish_reasoning_stream())
    }
}
