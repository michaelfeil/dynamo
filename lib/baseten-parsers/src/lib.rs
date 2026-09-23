// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Request-scoped lifecycle guards around the pinned upstream parser registries.

mod harmony;
pub mod vllm;

use anyhow::{Result, bail, ensure};
pub use dynamo_parsers_v2 as upstream;
pub use upstream::{
    InvalidGuidedPayloadPolicy, REGISTERED_UNIFIED_FAMILIES, Tool, UnifiedParser,
    UnifiedParserInit, UnifiedParserOutput, UnifiedParserStartingState, UnifiedToolOutputMode,
};

pub const UPSTREAM_REVISION: &str = "bb20dd01b6257cff9b739be2b18e7a444980d04a";

/// Families accepted by the Dynamo backend, including local v1 adapters.
pub fn unified_parser_families() -> Vec<&'static str> {
    REGISTERED_UNIFIED_FAMILIES
        .iter()
        .chain(harmony::FAMILIES)
        .copied()
        .collect()
}

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

struct DynamoStream {
    parser: Box<dyn upstream::UnifiedParser>,
    closed: bool,
}

impl DynamoStream {
    pub fn new(family: &str, tools: &[Tool], init: UnifiedParserInit) -> Result<Self> {
        let parser: Box<dyn UnifiedParser> = if harmony::FAMILIES.contains(&family) {
            Box::new(harmony::HarmonyParser::new(tools)?)
        } else {
            upstream::create_unified_parser_for_family(family, tools)?
        };
        Self::from_parser(parser, init)
    }

    /// Wrap a parser implementing Dynamo's unified trait.
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

enum Backend {
    Dynamo(DynamoStream),
    Vllm(vllm::VllmUnifiedStream),
}

/// One ordered reasoning, text, and tool-call stream for either Rust backend.
pub struct UnifiedStream {
    backend: Backend,
}

impl UnifiedStream {
    pub fn new(family: &str, tools: &[Tool], init: UnifiedParserInit) -> Result<Self> {
        Ok(Self {
            backend: Backend::Dynamo(DynamoStream::new(family, tools, init)?),
        })
    }

    pub fn new_with_backend(
        backend: &str,
        family: &str,
        tools: &[Tool],
        init: UnifiedParserInit,
        tokenizer_path: Option<&std::path::Path>,
    ) -> Result<Self> {
        match backend {
            "dynamo" => {
                ensure!(
                    tokenizer_path.is_none(),
                    "tokenizer_path is only used by vLLM"
                );
                Self::new(family, tools, init)
            }
            "vllm" => {
                let path = tokenizer_path
                    .ok_or_else(|| anyhow::anyhow!("vLLM requires tokenizer_path"))?;
                Ok(Self {
                    backend: Backend::Vllm(vllm::VllmUnifiedStream::new(
                        family, tools, init, path,
                    )?),
                })
            }
            _ => bail!("unknown unified parser backend: {backend}"),
        }
    }

    pub fn from_parser(parser: Box<dyn UnifiedParser>, init: UnifiedParserInit) -> Result<Self> {
        Ok(Self {
            backend: Backend::Dynamo(DynamoStream::from_parser(parser, init)?),
        })
    }

    pub fn preserve_special_tokens(&self) -> bool {
        match &self.backend {
            Backend::Dynamo(parser) => parser.preserve_special_tokens(),
            Backend::Vllm(parser) => parser.preserve_special_tokens(),
        }
    }

    pub fn tool_call_id(&self, index: usize) -> Option<&str> {
        match &self.backend {
            Backend::Dynamo(parser) => parser.tool_call_id(index),
            Backend::Vllm(parser) => parser.tool_call_id(index),
        }
    }

    pub fn advance(&mut self, text: Option<&str>) -> std::result::Result<Vec<Event>, StreamError> {
        match &mut self.backend {
            Backend::Dynamo(parser) => parser.advance(text),
            Backend::Vllm(parser) => parser.advance(text),
        }
    }
}
