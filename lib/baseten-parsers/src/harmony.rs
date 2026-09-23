// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Adapt v1's gpt-oss reasoning/tool parsers to the ordered unified contract.

use std::sync::OnceLock;

use anyhow::{Result, anyhow, ensure};
use dynamo_parsers_v1::{GptOssReasoningParser, ParserResult, ReasoningParser};

use crate::{
    Tool, UnifiedParser, UnifiedParserInit, UnifiedParserOutput, UnifiedParserStartingState,
    UnifiedToolOutputMode, upstream::ToolCallDelta,
};

pub const FAMILIES: &[&str] = &["harmony", "gpt_oss", "gpt-oss"];

// HF tokenizers and openai-harmony use different spellings for the same token IDs.
const MARKER_ALIASES: &[(&str, &str)] = &[
    ("<|im_start|>", "<|start|>"),
    ("<|im_end|>", "<|end|>"),
    ("<|im_sep|>", "<|message|>"),
    ("<|meta_sep|>", "<|channel|>"),
    ("<|meta_start|>", "<|constrain|>"),
    ("<|fim_suffix|>", "<|return|>"),
    ("<|ghissue|>", "<|call|>"),
];

pub(crate) struct HarmonyParser {
    reasoning: GptOssReasoningParser,
    tools: Vec<dynamo_parsers_v1::ToolDefinition>,
    call_ids: Vec<String>,
    message: String,
    marker_prefix: String,
}

fn initialize_encoding() -> Result<()> {
    static READY: OnceLock<Result<()>> = OnceLock::new();
    READY.get_or_init(|| {
        // v1's tool encoding loader uses spawn_blocking. Warm it once outside
        // the caller's runtime; subsequent parsing only reads the cached encoding.
        std::thread::spawn(|| {
            tokio::runtime::Builder::new_current_thread().enable_all().build()?
                .block_on(dynamo_parsers_v1::tool_calling::harmony::harmony_parser::get_harmony_encoding())
                .as_ref().map(|_| ()).map_err(|error| anyhow!("Harmony encoding: {error}"))
        }).join().map_err(|_| anyhow!("Harmony encoding loader panicked"))?
    }).as_ref().map(|_| ()).map_err(|error| anyhow!("{error}"))
}

impl HarmonyParser {
    pub fn new(tools: &[Tool]) -> Result<Self> {
        initialize_encoding()?;
        Ok(Self {
            reasoning: GptOssReasoningParser::new()?,
            tools: tools
                .iter()
                .map(|tool| dynamo_parsers_v1::ToolDefinition {
                    name: tool.name.clone(),
                    parameters: Some(tool.parameters.clone()),
                    strict: tool.strict,
                })
                .collect(),
            call_ids: Vec::new(),
            message: String::new(),
            marker_prefix: String::new(),
        })
    }

    fn emit(&mut self, result: ParserResult, output: &mut UnifiedParserOutput) -> Result<()> {
        output.push_reasoning(result.reasoning_text);
        if result.normal_text.is_empty() {
            return Ok(());
        }
        if !dynamo_parsers_v1::detect_tool_call_start(&result.normal_text, Some("harmony"))? {
            output.push_text(result.normal_text);
            return Ok(());
        }
        // A recipient may precede channel metadata; v1's reasoning handoff
        // starts at the channel, so tool parsing needs the original message.
        let (calls, text) =
            futures::executor::block_on(dynamo_parsers_v1::detect_and_parse_tool_call(
                self.message
                    .strip_prefix("<|start|>assistant")
                    .unwrap_or(&self.message),
                Some("harmony"),
                Some(&self.tools),
            ))?;
        for call in calls {
            let index = self.call_ids.len();
            self.call_ids.push(call.id);
            output.push_call(ToolCallDelta {
                tool_index: index,
                name: Some(call.function.name),
                arguments: call.function.arguments,
                complete: true,
            });
        }
        if let Some(text) = text {
            output.push_text(text);
        }
        Ok(())
    }
}

impl UnifiedParser for HarmonyParser {
    fn initialize_request(&mut self, init: UnifiedParserInit) -> Result<()> {
        ensure!(
            init.tool_output_mode == UnifiedToolOutputMode::Native,
            "Harmony v1 supports native tool output, not guided_json"
        );
        self.reasoning = GptOssReasoningParser::new()?;
        self.call_ids.clear();
        self.message.clear();
        self.marker_prefix.clear();
        let prefix = match init.starting_state {
            UnifiedParserStartingState::Reasoning => "<|channel|>analysis<|message|>",
            UnifiedParserStartingState::Response => "<|channel|>final<|message|>",
            UnifiedParserStartingState::None => "",
        };
        if !prefix.is_empty() {
            self.reasoning
                .parse_reasoning_streaming_incremental(prefix, &[]);
        } else if !init.prompt_token_ids.is_empty() {
            // gpt-oss's start token is 200006; only the trailing generation
            // message establishes parser state, never earlier conversation turns.
            let at = init
                .prompt_token_ids
                .iter()
                .rposition(|id| *id == 200006)
                .ok_or_else(|| anyhow!("Harmony prompt has no message-start token"))?;
            let encoding = futures::executor::block_on(
                dynamo_parsers_v1::tool_calling::harmony::harmony_parser::get_harmony_encoding(),
            )
            .as_ref()
            .map_err(|error| anyhow!("Harmony encoding: {error}"))?;
            let suffix = encoding
                .tokenizer()
                .decode_utf8(&init.prompt_token_ids[at + 1..])?;
            let suffix = suffix.strip_prefix("assistant").ok_or_else(|| {
                anyhow!("Harmony prompt must end with an assistant generation prefix")
            })?;
            self.reasoning
                .parse_reasoning_streaming_incremental(suffix, &[]);
            self.message.push_str(suffix);
        }
        Ok(())
    }

    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        let mut input = std::mem::take(&mut self.marker_prefix);
        input.push_str(delta);
        let hold = MARKER_ALIASES
            .iter()
            .flat_map(|(alias, _)| {
                (1..alias.len()).filter(|length| input.ends_with(&alias[..*length]))
            })
            .max()
            .unwrap_or(0);
        self.marker_prefix = input.split_off(input.len() - hold);
        for (alias, canonical) in MARKER_ALIASES {
            input = input.replace(alias, canonical);
        }
        // v1 returns separate text/reasoning fields. Advancing through one
        // delimiter at a time prevents a chunk spanning channels from reordering them.
        for piece in input.split_inclusive('>') {
            self.message.push_str(piece);
            let result = self
                .reasoning
                .parse_reasoning_streaming_incremental(piece, &[]);
            self.emit(result, output)?;
            if ["<|end|>", "<|return|>", "<|call|>"]
                .iter()
                .any(|marker| self.message.ends_with(marker))
            {
                self.message.clear();
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        let mut output = UnifiedParserOutput::default();
        let pending = std::mem::take(&mut self.marker_prefix);
        self.message.push_str(&pending);
        let result = self
            .reasoning
            .parse_reasoning_streaming_incremental(&pending, &[]);
        self.emit(result, &mut output)?;
        let result = self.reasoning.finish_reasoning_stream();
        self.emit(result, &mut output)?;
        Ok(output)
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn tool_call_id(&self, index: usize) -> Option<&str> {
        self.call_ids.get(index).map(String::as_str)
    }
}
