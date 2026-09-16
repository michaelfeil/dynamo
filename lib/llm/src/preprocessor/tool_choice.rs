// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tool-choice guided decoding policy for OpenAI chat requests.

use crate::preprocessor::{OpenAIPreprocessor, PreprocessedRequest};
use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use crate::protocols::openai::tools::get_json_schema_from_tools;

use dynamo_parsers::tool_calling::{ToolChoice, ToolDefinition};
use dynamo_protocols::types::{ChatCompletionTool, ChatCompletionToolChoiceOption, ResponseFormat};
use dynamo_runtime::error::{DynamoError, ErrorType};

fn invalid_argument(message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::InvalidArgument)
        .message(message)
        .build()
}

impl OpenAIPreprocessor {
    /// Apply guided decoding for OpenAI tool-choice requests.
    ///
    /// Structural tags are preferred when enabled and supported by the configured
    /// tool-call parser. Forced tool-choice requests fall back to the legacy
    /// JSON-schema constraint when structural tags are not applied.
    pub(super) fn apply_tool_choice_guided_decoding(
        &self,
        request: &NvCreateChatCompletionRequest,
        common_request: &mut PreprocessedRequest,
        prompt_injected_reasoning: bool,
    ) -> Result<bool, DynamoError> {
        let tool_choice = request
            .inner
            .tool_choice
            .as_ref()
            .unwrap_or(&ChatCompletionToolChoiceOption::Auto);
        let tools = request.inner.tools.as_deref().unwrap_or(&[]);
        let is_forced_tool_choice = matches!(
            tool_choice,
            ChatCompletionToolChoiceOption::Required | ChatCompletionToolChoiceOption::Named(_)
        );
        let has_explicit_guided_decoding = has_explicit_guided_decoding(request);
        let has_response_format_constraint = has_response_format_constraint(request);

        if is_forced_tool_choice && has_explicit_guided_decoding {
            return Err(invalid_argument(concat!(
                "guided decoding cannot be used in the same request as ",
                "tool_choice=\"required\" or a named tool_choice.",
            )));
        }

        // For non-forced tool choice, explicit guided decoding and response_format
        // constrain assistant content, so tool-choice guided decoding stays inactive.
        let has_assistant_constraint =
            has_explicit_guided_decoding || has_response_format_constraint;
        if !is_forced_tool_choice && has_assistant_constraint {
            return Ok(false);
        }

        if is_forced_tool_choice
            && has_response_format_constraint
            && let Some(gd) = common_request.sampling_options.guided_decoding.as_mut()
        {
            // OpenAI `response_format` applies to assistant content, not tool calls.
            gd.json = None;
        }

        // Preserve the model-specific grammar installed by the chat processor.
        if common_request
            .sampling_options
            .guided_decoding
            .as_ref()
            .is_some_and(|gd| gd.structural_tag.is_some())
        {
            // Keep forced/named tool-choice jailing and marker-parser fallback.
            return Ok(false);
        }

        if self.apply_tool_choice_structural_tag(
            &convert_tool_choice(tool_choice),
            &convert_tools(tools),
            request.inner.parallel_tool_calls,
            prompt_injected_reasoning,
            common_request,
        )? {
            return Ok(true);
        }

        match get_json_schema_from_tools(Some(tool_choice), Some(tools)) {
            Ok(Some(schema)) => {
                let gd = common_request
                    .sampling_options
                    .guided_decoding
                    .get_or_insert_default();
                gd.json = Some(schema);
            }
            Ok(None) => {}
            Err(err) => {
                return Err(invalid_argument(err.to_string()));
            }
        }

        // Auto/None requests can reach here when neither structural tags nor a
        // tool-choice JSON fallback were needed.
        Ok(false)
    }
}

fn has_explicit_guided_decoding(request: &NvCreateChatCompletionRequest) -> bool {
    request.common.guided_json.is_some()
        || request.common.guided_regex.is_some()
        || request
            .common
            .guided_choice
            .as_ref()
            .is_some_and(|v| !v.is_empty())
        || request.common.guided_grammar.is_some()
}

fn has_response_format_constraint(request: &NvCreateChatCompletionRequest) -> bool {
    request
        .inner
        .response_format
        .as_ref()
        .is_some_and(|format| !matches!(format, ResponseFormat::Text))
}

fn convert_tool_choice(tool_choice: &ChatCompletionToolChoiceOption) -> ToolChoice {
    match tool_choice {
        ChatCompletionToolChoiceOption::None => ToolChoice::None,
        ChatCompletionToolChoiceOption::Auto => ToolChoice::Auto,
        ChatCompletionToolChoiceOption::Required => ToolChoice::Required,
        ChatCompletionToolChoiceOption::Named(named) => {
            ToolChoice::Named(named.function.name.clone())
        }
    }
}

fn convert_tools(tools: &[ChatCompletionTool]) -> Vec<ToolDefinition> {
    tools
        .iter()
        .map(|tool| ToolDefinition {
            name: tool.function.name.clone(),
            parameters: tool.function.parameters.clone(),
            strict: tool.function.strict,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local_model::runtime_config::StructuralTagMode;
    use crate::model_card::ModelDeploymentCard;
    use crate::protocols::common::GuidedDecodingOptions;
    use serde_json::json;

    #[test]
    fn preserves_processor_structural_tag_and_tool_choice_jail() {
        for mode in [StructuralTagMode::Off, StructuralTagMode::On] {
            let mut mdc = ModelDeploymentCard::load_from_disk(
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/data/sample-models/mock-llama-3.1-8b-instruct"
                ),
                None,
            )
            .unwrap();
            mdc.runtime_config.structural_tag_mode = mode;
            mdc.runtime_config.tool_call_parser = Some("hermes".into());
            let preprocessor = OpenAIPreprocessor::new(mdc).unwrap();

            for choice in [
                json!("required"),
                json!({"type": "function", "function": {"name": "weather"}}),
            ] {
                let request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
                    "model": "test",
                    "messages": [{"role": "user", "content": "Weather?"}],
                    "tools": [{"type": "function", "function": {
                        "name": "weather", "parameters": {"type": "object", "properties": {}}
                    }}],
                    "tool_choice": choice,
                }))
                .unwrap();
                let mut common: PreprocessedRequest = serde_json::from_value(json!({
                    "model": "test", "token_ids": [1]
                }))
                .unwrap();
                let tag = json!({
                    "type": "structural_tag",
                    "format": {"type": "tag", "begin": "<tool_call>",
                               "content": {"type": "json_schema", "json_schema": {"type": "object"}},
                               "end": "</tool_call>"}
                });
                common.sampling_options.guided_decoding = Some(GuidedDecodingOptions {
                    structural_tag: Some(tag.clone()),
                    ..Default::default()
                });

                let bypass_jail = preprocessor
                    .apply_tool_choice_guided_decoding(&request, &mut common, false)
                    .unwrap();
                assert!(
                    !bypass_jail,
                    "processor grammar must retain forced-tool jailing"
                );
                let gd = common.sampling_options.guided_decoding.as_ref().unwrap();
                assert_eq!(gd.structural_tag.as_ref(), Some(&tag));
                assert!(
                    gd.json.is_none(),
                    "must not layer generic JSON over the processor grammar"
                );

                // Without a processor grammar, retain the configured fallback.
                common.sampling_options.guided_decoding = None;
                let bypass_jail = preprocessor
                    .apply_tool_choice_guided_decoding(&request, &mut common, false)
                    .unwrap();
                let gd = common.sampling_options.guided_decoding.as_ref().unwrap();
                assert_eq!(bypass_jail, mode == StructuralTagMode::On);
                assert_eq!(gd.structural_tag.is_some(), mode == StructuralTagMode::On);
                assert_eq!(gd.json.is_some(), mode == StructuralTagMode::Off);
            }
        }
    }
}
