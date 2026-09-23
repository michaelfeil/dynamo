// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Adapter for vLLM's native unified reasoning and tool parsers.
use std::{path::Path, sync::Arc};

use anyhow::{Result, ensure};
use vllm_parser::unified as v;
use vllm_tokenizer::{DecodedText, HuggingFaceTokenizer};

use crate::{
    Call, Event, InvalidGuidedPayloadPolicy, StreamError, Tool, UnifiedParserInit,
    UnifiedParserStartingState, UnifiedToolOutputMode,
};

// Keep in sync with both vllm-parser and vllm-tokenizer in Cargo.toml.
pub const UPSTREAM_REVISION: &str = "f84325c48c0acc1e3703103788c5f2976e719762";
pub const FAMILIES: &[&str] = &["gemma4", "hy_v3", "hy_v4", "inkling", "kimi_k3"];

pub(crate) struct VllmUnifiedStream {
    parser: Box<dyn v::UnifiedParser>,
    open_call: Option<(usize, Option<String>)>,
    closed: bool,
}

impl VllmUnifiedStream {
    pub fn new(
        family: &str,
        tools: &[Tool],
        init: UnifiedParserInit,
        tokenizer_path: &Path,
    ) -> Result<Self> {
        ensure!(
            FAMILIES.contains(&family),
            "unknown vLLM unified parser family: {family}"
        );
        ensure!(
            init.starting_state == UnifiedParserStartingState::None,
            "vLLM does not support starting_state"
        );
        ensure!(
            matches!(init.tool_output_mode, UnifiedToolOutputMode::Native),
            "vLLM does not support guided_json"
        );
        ensure!(
            init.invalid_guided_payload == InvalidGuidedPayloadPolicy::Reject,
            "vLLM does not support invalid_guided_payload"
        );
        let tokenizer = Arc::new(HuggingFaceTokenizer::new(tokenizer_path)?);
        let tools: Vec<_> = tools
            .iter()
            .map(|tool| vllm_parser::tool::Tool {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.parameters.clone(),
                strict: tool.strict,
            })
            .collect();
        let mut parser = match family {
            "gemma4" => v::Gemma4UnifiedParser::create(&tools, tokenizer)?,
            "hy_v3" => v::HyV3UnifiedParser::create(&tools, tokenizer)?,
            "hy_v4" => v::HyV4UnifiedParser::create(&tools, tokenizer)?,
            "inkling" => v::InklingUnifiedParser::create(&tools, tokenizer)?,
            "kimi_k3" => v::KimiK3UnifiedParser::create(&tools, tokenizer)?,
            _ => unreachable!("family validated against FAMILIES"),
        };
        parser.initialize(&init.prompt_token_ids)?;
        Ok(Self {
            parser,
            open_call: None,
            closed: false,
        })
    }

    pub fn preserve_special_tokens(&self) -> bool {
        self.parser.preserve_special_tokens()
    }
    pub fn tool_call_id(&self, index: usize) -> Option<&str> {
        self.parser.tool_call_id(index)
    }

    fn close_call(&mut self, events: &mut Vec<Event>) {
        if let Some((tool_index, id)) = self.open_call.take() {
            events.push(Event::ToolCall(Call {
                tool_index,
                id,
                name: None,
                arguments: String::new(),
                complete: true,
            }));
        }
    }

    fn project(&mut self, output: v::UnifiedParserOutput) -> Vec<Event> {
        let mut events = Vec::new();
        for event in output.events {
            match event {
                v::UnifiedParserEvent::Text(text) => {
                    self.close_call(&mut events);
                    events.push(Event::Text(text));
                }
                v::UnifiedParserEvent::Reasoning(text) => {
                    self.close_call(&mut events);
                    events.push(Event::Reasoning(text.text));
                }
                v::UnifiedParserEvent::ToolCall(delta) => {
                    if self
                        .open_call
                        .as_ref()
                        .is_some_and(|(index, _)| *index != delta.tool_index)
                    {
                        self.close_call(&mut events);
                    }
                    let id = self
                        .parser
                        .tool_call_id(delta.tool_index)
                        .map(str::to_owned);
                    self.open_call = Some((delta.tool_index, id.clone()));
                    events.push(Event::ToolCall(Call {
                        tool_index: delta.tool_index,
                        id,
                        name: delta.name,
                        arguments: delta.arguments,
                        complete: false,
                    }));
                }
            }
        }
        events
    }

    pub fn advance(&mut self, text: Option<&str>) -> std::result::Result<Vec<Event>, StreamError> {
        if self.closed {
            return Err(StreamError {
                error: anyhow::anyhow!("parser stream is closed"),
                events: vec![],
            });
        }
        let mut output = v::UnifiedParserOutput::default();
        let result = match text {
            Some(text) => self
                .parser
                .parse_into(DecodedText::unattributed(text), &mut output),
            None => {
                self.closed = true;
                self.parser.finish().map(|tail| output = tail)
            }
        };
        let mut events = self.project(output);
        if let Err(error) = result {
            self.closed = true;
            return Err(StreamError {
                error: error.into(),
                events,
            });
        }
        if text.is_none() {
            self.close_call(&mut events);
        }
        Ok(events)
    }
}

use v::UnifiedParser as _;
