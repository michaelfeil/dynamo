// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use crate::{
    Event, StreamError, Tool, ToolCallStream, ToolOutput, ToolParserInput, vllm::VllmToolStream,
};
use anyhow::{Result, bail};

/// Backend selection for the tool-only API. Dynamo remains the default in bindings.
pub enum ToolStream {
    Dynamo(ToolCallStream),
    Vllm(VllmToolStream),
}

impl ToolStream {
    pub fn new(backend: &str, family: &str, tools: &[Tool]) -> Result<Self> {
        match backend {
            "dynamo" => Ok(Self::Dynamo(ToolCallStream::new(family, tools)?)),
            "vllm" => Ok(Self::Vllm(VllmToolStream::new(family, tools)?)),
            _ => bail!("unknown tool parser backend: {backend}"),
        }
    }

    pub fn preserve_special_tokens(&self) -> bool {
        match self {
            Self::Dynamo(p) => p.preserve_special_tokens(),
            Self::Vllm(p) => p.preserve_special_tokens(),
        }
    }

    pub fn prefers_tokens(&self) -> bool {
        match self {
            Self::Dynamo(p) => p.prefers_tokens(),
            Self::Vllm(_) => false,
        }
    }

    pub fn completion_semantics(&self) -> &'static str {
        match self {
            Self::Dynamo(_) => "native",
            Self::Vllm(p) => p.completion_semantics(),
        }
    }

    pub fn advance(
        &mut self,
        input: Option<ToolParserInput<'_>>,
    ) -> std::result::Result<ToolOutput, StreamError> {
        match self {
            Self::Dynamo(parser) => parser.advance(input).map_err(|error| StreamError {
                error,
                events: vec![],
            }),
            Self::Vllm(parser) => {
                let text = match input {
                    Some(ToolParserInput::Text(text)) => Some(text),
                    None => None,
                    Some(ToolParserInput::Tokens(_)) => {
                        return Err(StreamError {
                            error: anyhow::anyhow!("vLLM tool parsers do not support token input"),
                            events: vec![],
                        });
                    }
                };
                let mut result = ToolOutput::default();
                for event in parser.advance(text)? {
                    match event {
                        Event::Text(text) => result.normal_text.push_str(&text),
                        Event::ToolCall(call) => result.calls.push(call),
                        Event::Reasoning(_) => {
                            unreachable!("tool-only parsers do not emit reasoning")
                        }
                    }
                }
                Ok(result)
            }
        }
    }
}
