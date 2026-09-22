// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! vLLM tool parsing with request-local lifecycle and ordered error output.
use anyhow::{Result, bail};
use vllm_parser::tool as v;

use crate::{Call, Event, StreamError, Tool};

// Keep in sync with the vllm-parser Git revision in Cargo.toml.
pub const UPSTREAM_REVISION: &str = "f84325c48c0acc1e3703103788c5f2976e719762";

type Factory = fn(&[v::Tool]) -> v::Result<Box<dyn v::ToolParser>>;
struct Family {
    name: &'static str,
    create: Factory,
    whole_call: bool,
}

macro_rules! families {
    ($($name:literal => $parser:ident, $whole:literal;)*) => {
        pub const FAMILIES: &[&str] = &[$($name),*];
        const REGISTRY: &[Family] = &[$(Family {
            name: $name,
            create: <v::$parser as v::ToolParser>::create,
            whole_call: $whole,
        }),*];
    };
}

families! {
    "deepseek_v3" => DeepSeekV3ToolParser, false;
    "deepseek_v31" => DeepSeekV31ToolParser, false;
    "deepseek_v32" => DeepSeekV32ToolParser, true;
    "deepseek_v4" => DeepSeekV4ToolParser, true;
    "deepseek_v41" => DeepSeekV41ToolParser, true;
    "glm45" => Glm45MoeToolParser, true;
    "glm47" => Glm47MoeToolParser, true;
    "qwen3_coder" => Qwen3CoderToolParser, true;
    "minimax_m2" => MinimaxM2ToolParser, true;
    "minimax_m3" => MinimaxM3ToolParser, true;
    "mimo" => MiMoToolParser, true;
    "kimi_k2" => KimiK2ToolParser, false;
    "qwen3_xml" => Qwen3XmlToolParser, false;
    "llama3_json" => Llama3JsonToolParser, false;
    "granite4" => Granite4ToolParser, false;
    "hermes" => HermesToolParser, false;
    "internlm2" => Internlm2ToolParser, false;
    "mistral" => MistralToolParser, false;
    "phi4_mini_json" => Phi4MiniJsonToolParser, false;
    "seed_oss" => SeedOssToolParser, true;
}

pub struct VllmToolStream {
    parser: Box<dyn v::ToolParser>,
    whole_call: bool,
    open_call: Option<(usize, Option<String>)>,
    closed: bool,
}

impl VllmToolStream {
    pub fn new(family: &str, tools: &[Tool]) -> Result<Self> {
        let Some(family) = REGISTRY.iter().find(|entry| entry.name == family) else {
            bail!("unknown vLLM tool parser family: {family}");
        };
        let tools: Vec<_> = tools
            .iter()
            .map(|tool| v::Tool {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.parameters.clone(),
                strict: tool.strict,
            })
            .collect();
        Ok(Self {
            parser: (family.create)(&tools)?,
            whole_call: family.whole_call,
            open_call: None,
            closed: false,
        })
    }

    pub fn preserve_special_tokens(&self) -> bool {
        self.parser.preserve_special_tokens()
    }

    /// Whole-call parsers report native completion. Other families report closure
    /// at the next call, visible text, or successful EOF, like vLLM's chat assembler.
    pub fn completion_semantics(&self) -> &'static str {
        if self.whole_call {
            "native"
        } else {
            "stream_boundary"
        }
    }

    fn close_call(&mut self, events: &mut Vec<Event>) {
        if let Some((index, id)) = self.open_call.take() {
            events.push(Event::ToolCall(Call {
                tool_index: index,
                id,
                name: None,
                arguments: String::new(),
                complete: true,
            }));
        }
    }

    fn project(&mut self, output: v::ToolParserOutput, events: &mut Vec<Event>) {
        for event in output.events {
            match event {
                v::ToolParserEvent::Text(text) => {
                    self.close_call(events);
                    events.push(Event::Text(text));
                }
                v::ToolParserEvent::ToolCall(delta) => {
                    if self
                        .open_call
                        .as_ref()
                        .is_some_and(|(index, _)| *index != delta.tool_index)
                    {
                        self.close_call(events);
                    }
                    if !self.whole_call {
                        self.open_call = Some((
                            delta.tool_index,
                            self.parser
                                .tool_call_id(delta.tool_index)
                                .map(str::to_owned),
                        ));
                    }
                    events.push(Event::ToolCall(Call {
                        tool_index: delta.tool_index,
                        id: self
                            .parser
                            .tool_call_id(delta.tool_index)
                            .map(str::to_owned),
                        name: delta.name,
                        arguments: delta.arguments,
                        complete: self.whole_call,
                    }));
                }
            }
        }
    }

    /// Errors are terminal and retain events committed before the failure.
    pub fn advance(&mut self, text: Option<&str>) -> std::result::Result<Vec<Event>, StreamError> {
        if self.closed {
            return Err(StreamError {
                error: anyhow::anyhow!("parser stream is closed"),
                events: vec![],
            });
        }
        let mut output = v::ToolParserOutput::default();
        let result = match text {
            Some(text) => self.parser.parse_into(text, &mut output),
            None => {
                self.closed = true;
                self.parser.finish().map(|tail| output = tail)
            }
        };
        let mut events = Vec::new();
        self.project(output, &mut events);
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
