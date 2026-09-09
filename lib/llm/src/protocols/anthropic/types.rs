// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Anthropic Messages API conversion logic.
//!
//! Pure protocol types live in `dynamo_protocols::types::anthropic`.
//! This module provides bidirectional conversion to/from the internal
//! chat completions format used by the Dynamo engine.

// Re-export all pure Anthropic protocol types so existing `use crate::protocols::anthropic::*`
// continues to work throughout dynamo-llm.
pub use dynamo_protocols::types::anthropic::*;

use dynamo_protocols::types::CompletionUsage;
use uuid::Uuid;

use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
};

/// Canonicalize a `/v1/messages` request body — the JSON exactly as the
/// client sent it — into the engine-bound Chat Completions request.
///
/// Goes through the shared api-translation crate (tool-bank's ingress).
/// Server-tool-shaped tools are dropped with a warning inside the crate
/// (standard-dynamo behavior); the client's `thinking` config lands on the
/// adapted body's own `thinking` and `thinking_token_budget` fields. The handler threads the parsed body here rather than re-serializing
/// its typed `AnthropicCreateMessageRequest`: the typed struct is a lossy
/// projection (joined system blocks, typed tool definitions, ...), and what
/// the canonicalizer sees must be what the client wrote.
pub fn anthropic_body_to_chat_request(
    body: serde_json::Value,
) -> anyhow::Result<NvCreateChatCompletionRequest> {
    canonicalize_anthropic_body(body).map(|canonical| canonical.request)
}

/// b10: [`anthropic_body_to_chat_request`] plus the typed loss record, timed
/// and logged as the `canonicalize` stage.
pub fn canonicalize_anthropic_body(
    body: serde_json::Value,
) -> anyhow::Result<crate::protocols::unified::Canonicalized> {
    let started = std::time::Instant::now();
    let result = canonicalize_anthropic_body_inner(body);
    crate::protocols::unified::b10_log_canonicalize_stage("messages", started, &result);
    result
}

fn canonicalize_anthropic_body_inner(
    body: serde_json::Value,
) -> anyhow::Result<crate::protocols::unified::Canonicalized> {
    let adapted = b10_dynamo_api_translation::request::adapt_request_json(
        body,
        b10_dynamo_api_translation::ClientProtocol::Messages,
        &http::HeaderMap::new(),
        &mut b10_dynamo_api_translation::hooks::DropServerTools,
    )
    .map_err(anyhow::Error::new)?;
    let losses = adapted.request.losses;
    let lowered = adapted.request.request;
    // Wire-edge re-parse: serializing the adapted CC body and reading it
    // back as the Nv wrapper distributes the extension keys (thinking,
    // nvext, chat_template_kwargs, cache_control, ...) into their typed
    // homes exactly as if the deployment had received the bytes over
    // HTTP. Residual keys — fields neither the wire types nor the extension
    // surface model — land in the wrapper's `unsupported_fields` catch-all
    // (warned during validation, never serialized), so they cannot ride the
    // engine-bound body: strict engine-side parsers (the python harness's
    // extra=forbid pydantic models) 400 on them.
    let mut nv: NvCreateChatCompletionRequest =
        serde_json::from_value(serde_json::to_value(&lowered)?)?;
    // The messages handler always streams internally.
    nv.inner.stream = Some(true);
    nv.inner.stream_options = Some(dynamo_protocols::types::ChatCompletionStreamOptions {
        include_usage: true,
        continuous_usage_stats: false,
    });
    Ok(crate::protocols::unified::Canonicalized {
        request: nv,
        losses,
    })
}

/// Typed-struct entry for callers that no longer hold the client's bytes.
/// Re-serializes the struct in Anthropic's wire shapes (see
/// `SystemContent`'s `Serialize`) and canonicalizes that; the HTTP handler
/// uses [`anthropic_body_to_chat_request`] on the original body instead.
impl TryFrom<AnthropicCreateMessageRequest> for NvCreateChatCompletionRequest {
    type Error = anyhow::Error;

    fn try_from(req: AnthropicCreateMessageRequest) -> Result<Self, Self::Error> {
        anthropic_body_to_chat_request(serde_json::to_value(&req)?)
    }
}

pub(crate) fn new_tool_use_id() -> String {
    format!("toolu_{}", Uuid::new_v4().simple())
}

/// Convert Dynamo's OpenAI-compatible usage into Anthropic's non-overlapping
/// input-token accounting.
///
/// Dynamo backends report `prompt_tokens` as the complete prompt and
/// `cached_tokens` as a subset of it. Anthropic reports the cached subset
/// separately, so `input_tokens` must exclude it.
pub(super) fn completion_usage_to_anthropic(usage: &CompletionUsage) -> AnthropicUsage {
    let cache_read_input_tokens = usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|details| details.cached_tokens)
        // A backend must not be able to produce an Anthropic usage breakdown
        // whose cached subset exceeds the complete prompt.
        .map(|tokens| tokens.min(usage.prompt_tokens))
        .filter(|&tokens| tokens > 0);

    AnthropicUsage {
        input_tokens: usage
            .prompt_tokens
            .saturating_sub(cache_read_input_tokens.unwrap_or(0)),
        output_tokens: usage.completion_tokens,
        // OpenAI-compatible backends do not report cache-write counts. Emit an
        // explicit 0 (not absent) so downstream metering never interprets a
        // missing field as "the whole prompt was written to cache".
        cache_creation_input_tokens: Some(0),
        cache_read_input_tokens,
    }
}

/// Convert a completed chat completion response into an Anthropic Messages response.
pub fn chat_completion_to_anthropic_response(
    chat_resp: NvCreateChatCompletionResponse,
    model: &str,
    api_context: Option<&crate::protocols::unified::AnthropicContext>,
) -> AnthropicMessageResponse {
    let _ = api_context; // Available for future enrichment (service_tier, etc.)
    let msg_id = format!("msg_{}", Uuid::new_v4().simple());

    let choice = chat_resp.inner.choices.into_iter().next();
    let mut content = Vec::new();
    let mut stop_reason = None;

    if let Some(choice) = choice {
        // Map finish_reason
        stop_reason = choice.finish_reason.map(|fr| match fr {
            dynamo_protocols::types::FinishReason::Stop => AnthropicStopReason::EndTurn,
            dynamo_protocols::types::FinishReason::Length => AnthropicStopReason::MaxTokens,
            dynamo_protocols::types::FinishReason::ToolCalls => AnthropicStopReason::ToolUse,
            dynamo_protocols::types::FinishReason::ContentFilter => AnthropicStopReason::EndTurn,
            dynamo_protocols::types::FinishReason::FunctionCall => AnthropicStopReason::ToolUse,
        });

        // Extract tool calls
        if let Some(tool_calls) = choice.message.tool_calls {
            for tc in tool_calls {
                let input: serde_json::Value =
                    serde_json::from_str(&tc.function.arguments).unwrap_or(serde_json::json!({}));
                content.push(AnthropicResponseContentBlock::ToolUse {
                    id: new_tool_use_id(),
                    name: tc.function.name,
                    input,
                });
            }
        }

        // Extract reasoning content (from --dyn-reasoning-parser, e.g. qwen3).
        // The backend strips <think>...</think> from the text and surfaces it
        // as reasoning_content on the message. Map this to a Thinking block
        // so clients see proper extended thinking in the Anthropic response.
        if let Some(thinking) = choice.message.reasoning_content.filter(|t| !t.is_empty()) {
            content.insert(
                0,
                AnthropicResponseContentBlock::Thinking {
                    thinking,
                    signature: String::new(),
                },
            );
        }

        // Extract text content
        let text = match choice.message.content {
            Some(dynamo_protocols::types::ChatCompletionMessageContent::Text(t)) => Some(t),
            Some(dynamo_protocols::types::ChatCompletionMessageContent::Parts(_)) => {
                tracing::warn!(
                    "Multimodal (Parts) content in chat completion response replaced with placeholder text in Anthropic conversion."
                );
                Some("[multimodal content]".to_string())
            }
            None => None,
        };
        if let Some(text) = text {
            // Text goes after thinking block (if any)
            content.push(AnthropicResponseContentBlock::Text {
                text,
                citations: None,
            });
        }
    }

    // Ensure there's at least one content block
    if content.is_empty() {
        content.push(AnthropicResponseContentBlock::Text {
            text: String::new(),
            citations: None,
        });
    }

    // Map usage through the same protocol conversion used by the streaming path.
    let usage = chat_resp
        .inner
        .usage
        .as_ref()
        .map(completion_usage_to_anthropic)
        .unwrap_or_else(|| AnthropicUsage {
            cache_creation_input_tokens: Some(0),
            ..Default::default()
        });

    AnthropicMessageResponse {
        id: msg_id,
        object_type: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: model.to_string(),
        stop_reason,
        stop_sequence: None,
        usage,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unknown Anthropic root fields (e.g. Claude Code's `context_management`)
    /// must not reach the engine-bound body (strict engine-side parsers 400 on
    /// extras). The shared canonicalizer drops them at ingress with a warning,
    /// so they land neither on the body nor in `unsupported_fields`.
    #[test]
    fn unknown_root_fields_are_held_out_of_the_engine_body() {
        let chat = anthropic_body_to_chat_request(serde_json::json!({
            "model": "m", "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}
        }))
        .unwrap();
        assert!(
            chat.unsupported_fields.is_empty(),
            "dropped at ingress, not carried: {:?}",
            chat.unsupported_fields
        );
        let body = serde_json::to_value(&chat).unwrap();
        assert!(body.get("context_management").is_none());
    }
    use dynamo_protocols::types::{
        ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageContent,
        ChatCompletionRequestToolMessageContent, ChatCompletionRequestUserMessageContent,
        ChatCompletionRequestUserMessageContentPart, ChatCompletionToolChoiceOption,
        ReasoningContent,
    };

    #[test]
    fn test_simple_user_message_conversion() {
        let req = AnthropicCreateMessageRequest {
            model: "test-model".into(),
            max_tokens: 100,
            messages: vec![AnthropicMessage {
                role: AnthropicRole::User,
                content: AnthropicMessageContent::Text {
                    content: "Hello!".into(),
                },
            }],
            system: None,
            temperature: Some(0.7),
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            metadata: None,
            tools: None,
            tool_choice: None,
            cache_control: None,
            thinking: None,
            service_tier: None,
            container: None,
            output_config: None,
            unmodeled: Default::default(),
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert_eq!(chat_req.inner.model, "test-model");
        assert_eq!(chat_req.inner.max_completion_tokens, Some(100));
        assert_eq!(chat_req.inner.temperature, Some(0.7));
        assert_eq!(chat_req.inner.messages.len(), 1);

        match &chat_req.inner.messages[0] {
            ChatCompletionRequestMessage::User(u) => match &u.content {
                ChatCompletionRequestUserMessageContent::Text(t) => {
                    assert_eq!(t, "Hello!");
                }
                _ => panic!("expected text content"),
            },
            _ => panic!("expected user message"),
        }
    }

    #[test]
    fn test_system_message_prepended() {
        let req = AnthropicCreateMessageRequest {
            model: "test-model".into(),
            max_tokens: 100,
            messages: vec![AnthropicMessage {
                role: AnthropicRole::User,
                content: AnthropicMessageContent::Text {
                    content: "Hi".into(),
                },
            }],
            system: Some(SystemContent {
                text: "You are helpful.".into(),
                cache_control: None,
            }),
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            metadata: None,
            tools: None,
            tool_choice: None,
            cache_control: None,
            thinking: None,
            service_tier: None,
            container: None,
            output_config: None,
            unmodeled: Default::default(),
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert_eq!(chat_req.inner.messages.len(), 2);
        assert_eq!(
            system_text(&chat_req),
            "You are helpful.",
            "the system prompt must reach the model as its text, not as a serialized struct"
        );
        assert!(matches!(
            &chat_req.inner.messages[1],
            ChatCompletionRequestMessage::User(_)
        ));
    }

    /// The text of the leading CC system message.
    fn system_text(chat_req: &NvCreateChatCompletionRequest) -> &str {
        match &chat_req.inner.messages[0] {
            ChatCompletionRequestMessage::System(system) => match &system.content {
                Some(ChatCompletionRequestSystemMessageContent::Text(text)) => text,
                other => panic!("expected text system content, got {other:?}"),
            },
            other => panic!("expected a system message first, got {other:?}"),
        }
    }

    fn body_with_system(system: serde_json::Value) -> NvCreateChatCompletionRequest {
        anthropic_body_to_chat_request(serde_json::json!({
            "model": "test-model",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "system": system,
        }))
        .unwrap()
    }

    /// Regression: the string form used to reach the model as `{"text":"..."}`
    /// because the typed `SystemContent` was re-serialized as its own struct.
    #[test]
    fn system_string_body_reaches_the_model_verbatim() {
        let chat_req = body_with_system(serde_json::json!("You are helpful."));
        assert_eq!(system_text(&chat_req), "You are helpful.");
        assert_eq!(chat_req.inner.messages.len(), 2);
    }

    #[test]
    fn system_block_array_body_flattens_to_its_text() {
        let chat_req = body_with_system(serde_json::json!([
            {"type": "text", "text": "You are helpful."},
            {"type": "text", "text": " Be terse."},
        ]));
        // Separator ruling (2026-09-03): adjacent blocks join with "\n", the deployed
        // converter's shape (prompt-cache prefix for Claude Code's two-block system array).
        assert_eq!(system_text(&chat_req), "You are helpful.\n Be terse.");
    }

    #[test]
    fn system_block_with_cache_control_keeps_only_its_text() {
        let chat_req = body_with_system(serde_json::json!([
            {"type": "text", "text": "Cached preamble.", "cache_control": {"type": "ephemeral"}},
        ]));
        assert_eq!(system_text(&chat_req), "Cached preamble.");
        let body = serde_json::to_value(&chat_req).unwrap();
        assert!(
            body["messages"][0].get("cache_control").is_none(),
            "block-level cache_control does not leak onto the CC message"
        );
        // The typed round-trip (callers holding only the struct) agrees.
        let req: AnthropicCreateMessageRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model", "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "system": [{"type": "text", "text": "Cached preamble.", "cache_control": {"type": "ephemeral"}}],
        }))
        .unwrap();
        let typed: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert_eq!(system_text(&typed), "Cached preamble.");
    }

    #[test]
    fn test_message_level_system_role_conversion() {
        let json = r#"{
            "model": "test-model",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Hi"},
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "Keep answers short."},
                        {"type": "text", "text": "Use the available shell."}
                    ]
                },
                {"role": "user", "content": "List files"}
            ]
        }"#;

        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req.messages[1].role, AnthropicRole::System));

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert_eq!(chat_req.inner.messages.len(), 3);
        assert!(matches!(
            &chat_req.inner.messages[0],
            ChatCompletionRequestMessage::User(_)
        ));
        match &chat_req.inner.messages[1] {
            ChatCompletionRequestMessage::System(system) => match &system.content {
                Some(ChatCompletionRequestSystemMessageContent::Text(text)) => {
                    // Separator ruling (2026-09-03): adjacent text blocks join with "\n" (the
                    // fork's converter always did; tool-bank's "" was not adopted).
                    assert_eq!(text, "Keep answers short.\nUse the available shell.");
                }
                other => panic!("expected text content, got {other:?}"),
            },
            other => panic!("expected system message, got {other:?}"),
        }
        assert!(matches!(
            &chat_req.inner.messages[2],
            ChatCompletionRequestMessage::User(_)
        ));
    }

    #[test]
    fn test_tool_use_blocks_conversion() {
        let req = AnthropicCreateMessageRequest {
            model: "test-model".into(),
            max_tokens: 100,
            messages: vec![
                AnthropicMessage {
                    role: AnthropicRole::User,
                    content: AnthropicMessageContent::Text {
                        content: "What's the weather?".into(),
                    },
                },
                AnthropicMessage {
                    role: AnthropicRole::Assistant,
                    content: AnthropicMessageContent::Blocks {
                        content: vec![AnthropicContentBlock::ToolUse {
                            id: "tool_123".into(),
                            name: "get_weather".into(),
                            input: serde_json::json!({"location": "SF"}),
                            cache_control: None,
                        }],
                    },
                },
                AnthropicMessage {
                    role: AnthropicRole::User,
                    content: AnthropicMessageContent::Blocks {
                        content: vec![AnthropicContentBlock::ToolResult {
                            tool_use_id: "tool_123".into(),
                            content: Some(ToolResultContent::Text("72F and sunny".into())),
                            is_error: None,
                            cache_control: None,
                        }],
                    },
                },
            ],
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            metadata: None,
            tools: None,
            tool_choice: None,
            cache_control: None,
            thinking: None,
            service_tier: None,
            container: None,
            output_config: None,
            unmodeled: Default::default(),
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert_eq!(chat_req.inner.messages.len(), 3);
        assert!(matches!(
            &chat_req.inner.messages[0],
            ChatCompletionRequestMessage::User(_)
        ));
        assert!(matches!(
            &chat_req.inner.messages[1],
            ChatCompletionRequestMessage::Assistant(_)
        ));
        assert!(matches!(
            &chat_req.inner.messages[2],
            ChatCompletionRequestMessage::Tool(_)
        ));
    }

    #[test]
    fn test_stop_sequences_conversion() {
        let req = AnthropicCreateMessageRequest {
            model: "test-model".into(),
            max_tokens: 100,
            messages: vec![AnthropicMessage {
                role: AnthropicRole::User,
                content: AnthropicMessageContent::Text {
                    content: "Hi".into(),
                },
            }],
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: Some(vec!["STOP".into(), "END".into()]),
            stream: false,
            metadata: None,
            tools: None,
            tool_choice: None,
            cache_control: None,
            thinking: None,
            service_tier: None,
            container: None,
            output_config: None,
            unmodeled: Default::default(),
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert!(chat_req.inner.stop.is_some());
    }

    #[test]
    fn test_tools_conversion() {
        let req = AnthropicCreateMessageRequest {
            model: "test-model".into(),
            max_tokens: 100,
            messages: vec![AnthropicMessage {
                role: AnthropicRole::User,
                content: AnthropicMessageContent::Text {
                    content: "Hi".into(),
                },
            }],
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            metadata: None,
            tools: Some(vec![AnthropicTool {
                name: "get_weather".into(),
                tool_type: None,
                description: Some("Get weather info".into()),
                input_schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                })),
                cache_control: None,
                defer_loading: None,
            }]),
            tool_choice: Some(AnthropicToolChoice::Simple(AnthropicToolChoiceSimple {
                choice_type: AnthropicToolChoiceMode::Auto,
                disable_parallel_tool_use: None,
            })),
            cache_control: None,
            thinking: None,
            service_tier: None,
            container: None,
            output_config: None,
            unmodeled: Default::default(),
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert!(chat_req.inner.tools.is_some());
        let tools = chat_req.inner.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "get_weather");
        assert!(matches!(
            chat_req.inner.tool_choice,
            Some(ChatCompletionToolChoiceOption::Auto)
        ));
    }

    /// Claude Code declares its WebSearch server tool (no input_schema) and a
    /// tool_choice; server tools are filtered during conversion, and a
    /// tool_choice without tools is rejected downstream with
    /// 400 "When using `tool_choice`, `tools` must be set". Both must be
    /// dropped so the model can answer in text.
    #[test]
    fn test_server_tools_only_drops_tools_and_tool_choice() {
        let req = AnthropicCreateMessageRequest {
            model: "test-model".into(),
            max_tokens: 100,
            messages: vec![AnthropicMessage {
                role: AnthropicRole::User,
                content: AnthropicMessageContent::Text {
                    content: "Search the web".into(),
                },
            }],
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            metadata: None,
            tools: Some(vec![AnthropicTool {
                name: "web_search".into(),
                tool_type: Some("web_search_20250305".into()),
                description: None,
                input_schema: None,
                cache_control: None,
                defer_loading: None,
            }]),
            tool_choice: Some(AnthropicToolChoice::Simple(AnthropicToolChoiceSimple {
                choice_type: AnthropicToolChoiceMode::Any,
                disable_parallel_tool_use: None,
            })),
            cache_control: None,
            thinking: None,
            service_tier: None,
            container: None,
            output_config: None,
            unmodeled: Default::default(),
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert!(chat_req.inner.tools.is_none());
        assert!(chat_req.inner.tool_choice.is_none());
    }

    /// A named tool_choice pointing at a filtered server tool degrades to
    /// auto when function tools remain, instead of naming a tool the worker
    /// never saw.
    #[test]
    fn test_named_choice_for_filtered_tool_degrades_to_auto() {
        let req = AnthropicCreateMessageRequest {
            model: "test-model".into(),
            max_tokens: 100,
            messages: vec![AnthropicMessage {
                role: AnthropicRole::User,
                content: AnthropicMessageContent::Text {
                    content: "Search the web".into(),
                },
            }],
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            metadata: None,
            tools: Some(vec![
                AnthropicTool {
                    name: "web_search".into(),
                    tool_type: Some("web_search_20250305".into()),
                    description: None,
                    input_schema: None,
                    cache_control: None,
                    defer_loading: None,
                },
                AnthropicTool {
                    name: "get_weather".into(),
                    tool_type: None,
                    description: None,
                    input_schema: Some(serde_json::json!({"type": "object"})),
                    cache_control: None,
                    defer_loading: None,
                },
            ]),
            tool_choice: Some(AnthropicToolChoice::Named(AnthropicToolChoiceNamed {
                choice_type: AnthropicToolChoiceMode::Tool,
                name: "web_search".into(),
                disable_parallel_tool_use: None,
            })),
            cache_control: None,
            thinking: None,
            service_tier: None,
            container: None,
            output_config: None,
            unmodeled: Default::default(),
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert_eq!(chat_req.inner.tools.as_ref().unwrap().len(), 1);
        assert!(matches!(
            chat_req.inner.tool_choice,
            Some(ChatCompletionToolChoiceOption::Auto)
        ));
    }

    #[allow(deprecated)]
    #[test]
    fn test_chat_completion_to_anthropic_response() {
        let chat_resp = NvCreateChatCompletionResponse {
            inner: dynamo_protocols::types::CreateChatCompletionResponse {
                id: "chatcmpl-xyz".into(),
                choices: vec![dynamo_protocols::types::ChatChoice {
                    index: 0,
                    message: dynamo_protocols::types::ChatCompletionResponseMessage {
                        content: Some(dynamo_protocols::types::ChatCompletionMessageContent::Text(
                            "Hello!".to_string(),
                        )),
                        refusal: None,
                        tool_calls: None,
                        role: dynamo_protocols::types::Role::Assistant,
                        function_call: None,
                        audio: None,
                        reasoning_content: None,
                    },
                    finish_reason: Some(dynamo_protocols::types::FinishReason::Stop),
                    logprobs: None,
                }],
                created: 1726000000,
                model: "test-model".into(),
                service_tier: None,
                system_fingerprint: None,
                object: "chat.completion".to_string(),
                usage: Some(dynamo_protocols::types::CompletionUsage {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                    total_tokens: 15,
                    prompt_tokens_details: None,
                    completion_tokens_details: None,
                }),
            },
            nvext: None,
        };

        let response = chat_completion_to_anthropic_response(chat_resp, "test-model", None);
        assert!(response.id.starts_with("msg_"));
        assert_eq!(response.object_type, "message");
        assert_eq!(response.role, "assistant");
        assert_eq!(response.model, "test-model");
        assert_eq!(response.stop_reason, Some(AnthropicStopReason::EndTurn));
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 5);
        assert_eq!(response.content.len(), 1);
        match &response.content[0] {
            AnthropicResponseContentBlock::Text { text, .. } => {
                assert_eq!(text, "Hello!");
            }
            _ => panic!("expected text block"),
        }
    }

    #[allow(deprecated)]
    #[test]
    fn test_anthropic_response_input_tokens_excludes_cached() {
        // OpenAI prompt_tokens is the total (12) and already includes the
        // cached tokens (11). Anthropic input_tokens must report only the
        // uncached portion (12 - 11 = 1), with cache_read reported separately.
        let chat_resp = NvCreateChatCompletionResponse {
            inner: dynamo_protocols::types::CreateChatCompletionResponse {
                id: "chatcmpl-cache".into(),
                choices: vec![dynamo_protocols::types::ChatChoice {
                    index: 0,
                    message: dynamo_protocols::types::ChatCompletionResponseMessage {
                        content: Some(dynamo_protocols::types::ChatCompletionMessageContent::Text(
                            "Hi!".to_string(),
                        )),
                        refusal: None,
                        tool_calls: None,
                        role: dynamo_protocols::types::Role::Assistant,
                        function_call: None,
                        audio: None,
                        reasoning_content: None,
                    },
                    finish_reason: Some(dynamo_protocols::types::FinishReason::Stop),
                    logprobs: None,
                }],
                created: 1726000000,
                model: "test-model".into(),
                service_tier: None,
                system_fingerprint: None,
                object: "chat.completion".to_string(),
                usage: Some(dynamo_protocols::types::CompletionUsage {
                    prompt_tokens: 12,
                    completion_tokens: 5,
                    total_tokens: 17,
                    prompt_tokens_details: Some(dynamo_protocols::types::PromptTokensDetails {
                        audio_tokens: None,
                        cached_tokens: Some(11),
                    }),
                    completion_tokens_details: None,
                }),
            },
            nvext: None,
        };

        let response = chat_completion_to_anthropic_response(chat_resp, "test-model", None);
        assert_eq!(response.usage.input_tokens, 1);
        assert_eq!(response.usage.cache_read_input_tokens, Some(11));
        assert_eq!(response.usage.cache_creation_input_tokens, Some(0));
        assert_eq!(response.usage.output_tokens, 5);
    }

    #[test]
    fn test_anthropic_usage_clamps_cached_tokens_to_prompt_tokens() {
        let usage = CompletionUsage {
            prompt_tokens: 12,
            completion_tokens: 5,
            total_tokens: 17,
            prompt_tokens_details: Some(dynamo_protocols::types::PromptTokensDetails {
                audio_tokens: None,
                cached_tokens: Some(20),
            }),
            completion_tokens_details: None,
        };

        let usage = completion_usage_to_anthropic(&usage);
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.cache_read_input_tokens, Some(12));
        assert_eq!(usage.cache_creation_input_tokens, Some(0));
        assert_eq!(usage.output_tokens, 5);
    }

    #[allow(deprecated)]
    #[test]
    fn test_tool_use_id_is_rewritten_to_toolu_prefix() {
        let chat_resp = NvCreateChatCompletionResponse {
            inner: dynamo_protocols::types::CreateChatCompletionResponse {
                id: "chatcmpl-xyz".into(),
                choices: vec![dynamo_protocols::types::ChatChoice {
                    index: 0,
                    message: dynamo_protocols::types::ChatCompletionResponseMessage {
                        content: None,
                        refusal: None,
                        tool_calls: Some(vec![
                            dynamo_protocols::types::ChatCompletionMessageToolCall {
                                id: "chatcmpl-tool-DEADBEEF".into(),
                                r#type: dynamo_protocols::types::FunctionType::Function,
                                function: dynamo_protocols::types::FunctionCall {
                                    name: "get_weather".into(),
                                    arguments: r#"{"location":"SF"}"#.into(),
                                },
                            },
                        ]),
                        role: dynamo_protocols::types::Role::Assistant,
                        function_call: None,
                        audio: None,
                        reasoning_content: None,
                    },
                    finish_reason: Some(dynamo_protocols::types::FinishReason::ToolCalls),
                    logprobs: None,
                }],
                created: 1726000000,
                model: "test-model".into(),
                service_tier: None,
                system_fingerprint: None,
                object: "chat.completion".to_string(),
                usage: Some(dynamo_protocols::types::CompletionUsage {
                    prompt_tokens: 20,
                    completion_tokens: 10,
                    total_tokens: 30,
                    prompt_tokens_details: None,
                    completion_tokens_details: None,
                }),
            },
            nvext: None,
        };

        let response = chat_completion_to_anthropic_response(chat_resp, "test-model", None);
        let (id, name) = response
            .content
            .iter()
            .find_map(|block| match block {
                AnthropicResponseContentBlock::ToolUse { id, name, .. } => {
                    Some((id.clone(), name.clone()))
                }
                _ => None,
            })
            .expect("expected a tool_use content block");
        assert!(
            id.starts_with("toolu_"),
            "tool_use.id must start with toolu_, got {id}"
        );
        assert!(
            !id.contains("chatcmpl-tool-"),
            "upstream id format must not leak, got {id}"
        );
        assert_eq!(name, "get_weather");
    }

    #[test]
    fn test_deserialize_simple_message() {
        let json =
            r#"{"model":"test","max_tokens":100,"messages":[{"role":"user","content":"Hello"}]}"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "test");
        assert_eq!(req.max_tokens, 100);
        assert_eq!(req.messages.len(), 1);
    }

    #[test]
    fn test_deserialize_content_blocks() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "tool_result", "tool_use_id": "tool_1", "content": "result text"}
                ]
            }]
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.messages.len(), 1);
        match &req.messages[0].content {
            AnthropicMessageContent::Blocks { content } => {
                assert_eq!(content.len(), 2);
            }
            _ => panic!("expected blocks content"),
        }
    }

    #[test]
    fn test_deserialize_thinking_block() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me reason about this...", "signature": "sig123"},
                    {"type": "text", "text": "Here is my answer."}
                ]
            }]
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        match &req.messages[0].content {
            AnthropicMessageContent::Blocks { content } => {
                assert_eq!(content.len(), 2);
                match &content[0] {
                    AnthropicContentBlock::Thinking {
                        thinking,
                        signature,
                        ..
                    } => {
                        assert_eq!(thinking, "Let me reason about this...");
                        assert_eq!(signature, "sig123");
                    }
                    other => panic!("expected Thinking, got {other:?}"),
                }
            }
            _ => panic!("expected blocks content"),
        }
    }

    #[test]
    fn test_thinking_block_becomes_reasoning_content() {
        let req = AnthropicCreateMessageRequest {
            model: "test-model".into(),
            max_tokens: 100,
            messages: vec![AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: AnthropicMessageContent::Blocks {
                    content: vec![
                        AnthropicContentBlock::Thinking {
                            thinking: "I should think...".into(),
                            signature: "sig".into(),
                            cache_control: None,
                        },
                        AnthropicContentBlock::Text {
                            text: "Answer".into(),
                            citations: None,
                            cache_control: None,
                        },
                    ],
                },
            }],
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            metadata: None,
            tools: None,
            tool_choice: None,
            cache_control: None,
            thinking: None,
            service_tier: None,
            container: None,
            output_config: None,
            unmodeled: Default::default(),
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        match &chat_req.inner.messages[0] {
            ChatCompletionRequestMessage::Assistant(a) => {
                assert_eq!(
                    a.reasoning_content,
                    Some(ReasoningContent::Text("I should think...".into()))
                );
                match &a.content {
                    Some(ChatCompletionRequestAssistantMessageContent::Text(t)) => {
                        assert_eq!(t, "Answer");
                    }
                    other => panic!("expected text content, got {other:?}"),
                }
            }
            other => panic!("expected assistant message, got {other:?}"),
        }
    }

    #[test]
    fn test_known_and_unknown_block_types() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "server_tool_use", "id": "stu_1", "name": "web_search", "input": {}},
                    {"type": "redacted_thinking", "data": "encrypted"},
                    {"type": "web_search_tool_result", "tool_use_id": "stu_1", "content": [{"type": "web_search_result", "url": "https://example.com"}]},
                    {"type": "future_block_type", "some_field": 42},
                    {"type": "text", "text": "world"}
                ]
            }]
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        match &req.messages[0].content {
            AnthropicMessageContent::Blocks { content } => {
                assert_eq!(content.len(), 6);
                assert!(matches!(&content[0], AnthropicContentBlock::Text { .. }));
                assert!(matches!(
                    &content[1],
                    AnthropicContentBlock::ServerToolUse { name, .. } if name == "web_search"
                ));
                assert!(matches!(
                    &content[2],
                    AnthropicContentBlock::RedactedThinking { data } if data == "encrypted"
                ));
                assert!(matches!(
                    &content[3],
                    AnthropicContentBlock::WebSearchToolResult { tool_use_id, .. } if tool_use_id == "stu_1"
                ));
                // Truly unknown types still fall through to Other with full JSON preserved
                assert!(matches!(
                    &content[4],
                    AnthropicContentBlock::Other(v) if v.get("type").and_then(|t| t.as_str()) == Some("future_block_type")
                ));
                assert!(matches!(&content[5], AnthropicContentBlock::Text { .. }));
            }
            _ => panic!("expected blocks content"),
        }

        // Conversion should succeed — server_tool_use becomes a tool call,
        // redacted_thinking and web_search_tool_result are preserved gracefully
        let chat_req: NvCreateChatCompletionRequest = AnthropicCreateMessageRequest {
            model: "test".into(),
            max_tokens: 100,
            messages: req.messages,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            metadata: None,
            tools: None,
            tool_choice: None,
            cache_control: None,
            thinking: None,
            service_tier: None,
            container: None,
            output_config: None,
            unmodeled: Default::default(),
        }
        .try_into()
        .unwrap();
        // CC-pivot: tool-bank semantics — the echoed turn splits back at the
        // web_search_tool_result into [assistant(text+tool_call), tool(result),
        // assistant(trailing text)], byte-identical to the CC sequence the model saw;
        // redacted_thinking and unknown blocks are skipped, never refused.
        assert_eq!(chat_req.inner.messages.len(), 3);
        match &chat_req.inner.messages[0] {
            ChatCompletionRequestMessage::Assistant(a) => {
                assert!(a.tool_calls.is_some());
                let tc = a.tool_calls.as_ref().unwrap();
                assert_eq!(tc.len(), 1);
                assert_eq!(tc[0].function.name, "web_search");
            }
            other => panic!("expected assistant, got {other:?}"),
        }
        match &chat_req.inner.messages[1] {
            ChatCompletionRequestMessage::Tool(t) => assert_eq!(t.tool_call_id, "stu_1"),
            other => panic!("expected tool result, got {other:?}"),
        }
        assert!(matches!(
            &chat_req.inner.messages[2],
            ChatCompletionRequestMessage::Assistant(_)
        ));
    }

    #[test]
    fn test_tool_result_string_content() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "simple text"}
                ]
            }]
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        match &req.messages[0].content {
            AnthropicMessageContent::Blocks { content } => match &content[0] {
                AnthropicContentBlock::ToolResult { content, .. } => {
                    let text = content.clone().unwrap().into_text();
                    assert_eq!(text, "simple text");
                }
                other => panic!("expected ToolResult, got {other:?}"),
            },
            _ => panic!("expected blocks"),
        }
    }

    #[test]
    fn test_tool_result_array_content() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "f", "input": {}}
                ]
            }, {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": [
                        {"type": "text", "text": "line 1"},
                        {"type": "text", "text": "line 2"}
                    ]}
                ]
            }]
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        match &req.messages[1].content {
            AnthropicMessageContent::Blocks { content } => match &content[0] {
                AnthropicContentBlock::ToolResult { content, .. } => {
                    let text = content.clone().unwrap().into_text();
                    assert_eq!(text, "line 1line 2");
                }
                other => panic!("expected ToolResult, got {other:?}"),
            },
            _ => panic!("expected blocks"),
        }
    }

    #[test]
    fn test_tool_result_text_only_converts_to_text_content() {
        // Text-only tool results must keep the flat `Text` content shape
        // (backwards-compatible with non-multimodal backends).
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "f", "input": {}}
                ]
            }, {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": [
                        {"type": "text", "text": "line 1"},
                        {"type": "text", "text": "line 2"}
                    ]}
                ]
            }]
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        match &chat_req.inner.messages[1] {
            ChatCompletionRequestMessage::Tool(tool) => {
                assert_eq!(tool.tool_call_id, "t1");
                match &tool.content {
                    ChatCompletionRequestToolMessageContent::Text(text) => {
                        assert_eq!(text, "line 1line 2");
                    }
                    other => panic!("expected Text content, got {other:?}"),
                }
            }
            other => panic!("expected tool message, got {other:?}"),
        }
    }

    #[test]
    fn test_tool_result_image_preserved() {
        // Regression: images inside tool_result content arrays were silently
        // dropped (flattened to text-only), so the model never saw images
        // delivered via tool_result (e.g. Claude Code Read/screenshot tools).
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "f", "input": {}}
                ]
            }, {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": [
                        {"type": "text", "text": "screenshot taken"},
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "aGVsbG8="
                        }}
                    ]}
                ]
            }]
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert_eq!(chat_req.inner.messages.len(), 2);
        match &chat_req.inner.messages[1] {
            ChatCompletionRequestMessage::Tool(tool) => {
                assert_eq!(tool.tool_call_id, "t1");
                match &tool.content {
                    ChatCompletionRequestToolMessageContent::Array(parts) => {
                        assert_eq!(parts.len(), 2);
                        match &parts[0] {
                            ChatCompletionRequestUserMessageContentPart::Text(t) => {
                                assert_eq!(t.text, "screenshot taken");
                            }
                            other => panic!("expected text part, got {other:?}"),
                        }
                        match &parts[1] {
                            ChatCompletionRequestUserMessageContentPart::ImageUrl(img) => {
                                assert_eq!(
                                    img.image_url.url.as_str(),
                                    "data:image/png;base64,aGVsbG8="
                                );
                            }
                            other => panic!("expected image_url part, got {other:?}"),
                        }
                    }
                    other => panic!("expected Array content, got {other:?}"),
                }
            }
            other => panic!("expected tool message, got {other:?}"),
        }
    }

    /// Deserialize a `tool_result.content` payload the way the endpoint does
    /// and run it through the conversion.
    fn convert_tool_result_json(
        content: serde_json::Value,
    ) -> Result<ChatCompletionRequestToolMessageContent, anyhow::Error> {
        // Through the full conversion path (the shared crate owns the
        // tool_result normalization now).
        let req: AnthropicCreateMessageRequest = serde_json::from_value(serde_json::json!({
            "model": "m", "max_tokens": 10,
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "f", "input": {}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": content}
                ]}
            ]
        }))
        .unwrap();
        let chat: NvCreateChatCompletionRequest = req.try_into()?;
        chat.inner
            .messages
            .into_iter()
            .find_map(|m| match m {
                ChatCompletionRequestMessage::Tool(t) => Some(t.content),
                _ => None,
            })
            .ok_or_else(|| anyhow::anyhow!("no tool message produced"))
    }

    fn expect_text(content: serde_json::Value) -> String {
        match convert_tool_result_json(content).unwrap() {
            ChatCompletionRequestToolMessageContent::Text(text) => text,
            other => panic!("expected Text content, got {other:?}"),
        }
    }

    #[test]
    fn test_tool_result_tool_reference_blocks_coerced_to_json_text() {
        // Verbatim Claude Code ToolSearch result: content is entirely
        // tool_reference blocks. Must not fail the request; small unknown
        // blocks pass through as JSON text so the model keeps the pointers.
        let text = expect_text(serde_json::json!([
            {"type": "tool_reference", "tool_name": "mcp__slack__read_thread"},
            {"type": "tool_reference", "tool_name": "mcp__slack__read_channel"},
        ]));
        assert!(text.contains("mcp__slack__read_thread"), "text: {text}");
        assert!(text.contains("mcp__slack__read_channel"), "text: {text}");
        assert!(text.contains("tool_reference"), "text: {text}");
    }

    #[test]
    fn test_tool_result_oversized_unknown_block_becomes_placeholder() {
        // Unknown blocks above the JSON pass-through limit must not balloon
        // the prompt: replaced by a one-line placeholder naming the type.
        let big = "x".repeat(2048); // 2x the crate's UNKNOWN_BLOCK_JSON_LIMIT
        let text = expect_text(serde_json::json!([
            {"type": "mystery_blob", "payload": big},
        ]));
        assert!(
            text.starts_with("[unsupported mystery_blob"),
            "text: {text}"
        );
        assert!(text.len() < 200, "placeholder should be short: {text}");
    }

    #[test]
    fn test_tool_result_base64_document_becomes_placeholder() {
        // A base64 document has no text representation: replaced by a short
        // placeholder, the surrounding text blocks survive, and the request
        // does not fail.
        let text = expect_text(serde_json::json!([
            {"type": "text", "text": "before "},
            {
                "type": "document",
                "title": "report",
                "source": {"type": "base64", "media_type": "application/pdf", "data": "aGVsbG8="},
            },
            {"type": "text", "text": " after"},
        ]));
        assert!(text.starts_with("before "), "text: {text}");
        assert!(text.ends_with(" after"), "text: {text}");
        assert!(
            text.contains("[document \"report\" omitted: application/pdf"),
            "text: {text}"
        );
    }

    #[test]
    fn test_tool_result_url_document_keeps_url_pointer() {
        let text = expect_text(serde_json::json!([{
            "type": "document",
            "title": "spec",
            "source": {"type": "url", "url": "https://example.com/spec.pdf"},
        }]));
        assert_eq!(text, "[document \"spec\": https://example.com/spec.pdf]");
    }

    #[test]
    fn test_tool_result_text_document_is_flattened() {
        let text = expect_text(serde_json::json!([{
            "type": "document",
            "title": "notes",
            "source": {"type": "text", "media_type": "text/plain", "data": "doc body"},
        }]));
        assert_eq!(text, "doc body");
    }

    #[test]
    fn test_tool_result_content_document_is_flattened() {
        let text = expect_text(serde_json::json!([{
            "type": "document",
            "source": {"type": "content", "content": [
                {"type": "text", "text": "part one "},
                {"type": "text", "text": "part two"},
            ]},
        }]));
        assert_eq!(text, "part one part two");
    }

    #[test]
    fn test_tool_result_search_result_is_flattened() {
        let text = expect_text(serde_json::json!([
            {
                "type": "search_result",
                "source": "https://example.com/doc",
                "title": "Example",
                "content": [{"type": "text", "text": "search hit text"}],
                "citations": {"enabled": true},
            },
            {"type": "text", "text": " and trailing text"},
        ]));
        assert_eq!(text, "search hit text and trailing text");
    }

    #[test]
    fn test_tool_result_malformed_document_is_coerced_not_rejected() {
        // A document whose shape we don't understand degrades to a coerced
        // unknown block (JSON text), never a request-level error.
        let text = expect_text(serde_json::json!([
            {"type": "document", "source": {"type": "mystery"}},
            {"type": "text", "text": "still here"},
        ]));
        assert!(text.ends_with("still here"), "text: {text}");
        assert!(text.contains("mystery"), "text: {text}");
    }

    #[test]
    fn test_count_tokens_estimate() {
        let req = AnthropicCountTokensRequest {
            model: "test".into(),
            messages: vec![AnthropicMessage {
                role: AnthropicRole::User,
                content: AnthropicMessageContent::Text {
                    content: "Hello, world! This is a test message.".into(),
                },
            }],
            system: Some(SystemContent {
                text: "You are helpful.".into(),
                cache_control: None,
            }),
            tools: None,
        };

        let tokens = req.estimate_tokens();
        assert!(tokens > 0, "should estimate non-zero tokens");
        // "Hello, world! This is a test message." (37) + "You are helpful." (16) + role (4) = 57 / 3 = 19
        assert_eq!(tokens, 19);
    }

    // --- ReasoningContent enum tests ---

    fn make_req(blocks: Vec<AnthropicContentBlock>) -> ChatCompletionRequestAssistantMessage {
        let req = AnthropicCreateMessageRequest {
            model: "test-model".into(),
            max_tokens: 100,
            messages: vec![AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: AnthropicMessageContent::Blocks { content: blocks },
            }],
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            metadata: None,
            tools: None,
            tool_choice: None,
            cache_control: None,
            thinking: None,
            service_tier: None,
            container: None,
            output_config: None,
            unmodeled: Default::default(),
        };
        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        match chat_req.inner.messages.into_iter().next().unwrap() {
            ChatCompletionRequestMessage::Assistant(a) => a,
            other => panic!("expected assistant, got {other:?}"),
        }
    }

    fn tool_use(id: &str) -> AnthropicContentBlock {
        AnthropicContentBlock::ToolUse {
            id: id.into(),
            name: "fn".into(),
            input: serde_json::json!({}),
            cache_control: None,
        }
    }

    fn thinking(text: &str) -> AnthropicContentBlock {
        AnthropicContentBlock::Thinking {
            thinking: text.into(),
            signature: "sig".into(),
            cache_control: None,
        }
    }

    #[test]
    fn test_interleaved_thinking_and_tool_calls() {
        // [Thinking("A"), ToolUse("t1"), Thinking("B"), ToolUse("t2")]
        // segments = ["A", "B", ""] (trailing empty), tool_calls = [t1, t2]
        let msg = make_req(vec![
            thinking("A"),
            tool_use("t1"),
            thinking("B"),
            tool_use("t2"),
        ]);

        let segs = msg
            .reasoning_content
            .as_ref()
            .expect("reasoning_content should be set")
            .segments()
            .expect("should be Segments variant");
        assert_eq!(segs.len(), 3); // tool_calls.len() + 1
        assert_eq!(segs[0], "A");
        assert_eq!(segs[1], "B");
        assert_eq!(segs[2], ""); // no trailing reasoning

        assert_eq!(
            msg.reasoning_content.as_ref().unwrap().to_flat_string(),
            "A\nB"
        );

        let tcs = msg.tool_calls.as_ref().expect("tool_calls should be set");
        assert_eq!(tcs.len(), 2);
        assert_eq!(tcs[0].id, "t1");
        assert_eq!(tcs[1].id, "t2");
    }

    #[test]
    fn test_trailing_reasoning_preserved_in_segments() {
        // [Thinking("A"), ToolUse("t1"), Thinking("B")]
        // segments = ["A", "B"], trailing reasoning "B" must appear in segments[1]
        let msg = make_req(vec![thinking("A"), tool_use("t1"), thinking("B")]);

        let segs = msg
            .reasoning_content
            .as_ref()
            .expect("reasoning_content should be set")
            .segments()
            .expect("should be Segments variant");
        assert_eq!(segs.len(), 2); // 1 tool call + 1 trailing
        assert_eq!(segs[0], "A");
        assert_eq!(segs[1], "B"); // trailing reasoning preserved

        assert_eq!(
            msg.reasoning_content.as_ref().unwrap().to_flat_string(),
            "A\nB"
        );
    }

    #[test]
    fn test_tool_use_before_thinking() {
        // [ToolUse("t1"), Thinking("A"), ToolUse("t2")]
        // segments = ["", "A", ""] — empty first segment, reasoning before t2
        let msg = make_req(vec![tool_use("t1"), thinking("A"), tool_use("t2")]);

        let segs = msg
            .reasoning_content
            .as_ref()
            .expect("reasoning_content should be set")
            .segments()
            .expect("should be Segments variant");
        assert_eq!(segs.len(), 3);
        assert_eq!(segs[0], ""); // no reasoning before t1
        assert_eq!(segs[1], "A");
        assert_eq!(segs[2], ""); // no trailing

        assert_eq!(
            msg.reasoning_content.as_ref().unwrap().to_flat_string(),
            "A"
        );
    }

    #[test]
    fn test_all_thinking_then_all_tools() {
        // [Thinking("A"), Thinking("B"), ToolUse("t1"), ToolUse("t2")]
        // segments = ["A\nB", "", ""] — all reasoning before first tool
        let msg = make_req(vec![
            thinking("A"),
            thinking("B"),
            tool_use("t1"),
            tool_use("t2"),
        ]);

        let segs = msg
            .reasoning_content
            .as_ref()
            .expect("reasoning_content should be set")
            .segments()
            .expect("should be Segments variant");
        assert_eq!(segs.len(), 3);
        assert_eq!(segs[0], "A\nB");
        assert_eq!(segs[1], "");
        assert_eq!(segs[2], "");

        assert_eq!(
            msg.reasoning_content.as_ref().unwrap().to_flat_string(),
            "A\nB"
        );
    }

    #[test]
    fn test_tool_calls_no_thinking_produces_no_segments() {
        // [ToolUse("t1"), ToolUse("t2")] — all empty segments → reasoning_content = None
        let msg = make_req(vec![tool_use("t1"), tool_use("t2")]);

        assert!(
            msg.reasoning_content.is_none(),
            "no reasoning means no reasoning_content"
        );
    }

    #[test]
    fn test_thinking_only_no_tools_produces_text_variant() {
        // [Thinking("A"), Text("answer")] — no tool calls → ReasoningContent::Text
        let msg = make_req(vec![
            thinking("A"),
            AnthropicContentBlock::Text {
                text: "answer".into(),
                citations: None,
                cache_control: None,
            },
        ]);

        assert_eq!(
            msg.reasoning_content,
            Some(ReasoningContent::Text("A".into()))
        );
        assert!(msg.reasoning_content.as_ref().unwrap().segments().is_none());
        assert!(matches!(
            msg.content,
            Some(ChatCompletionRequestAssistantMessageContent::Text(ref t)) if t == "answer"
        ));
    }

    #[test]
    fn test_single_thinking_then_single_tool() {
        // [Thinking("reason"), ToolUse("t1")] → Segments(["reason", ""])
        let msg = make_req(vec![thinking("reason"), tool_use("t1")]);

        let segs = msg
            .reasoning_content
            .as_ref()
            .expect("reasoning_content should be set")
            .segments()
            .expect("should be Segments variant");
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0], "reason");
        assert_eq!(segs[1], "");

        assert_eq!(
            msg.reasoning_content.as_ref().unwrap().to_flat_string(),
            "reason"
        );
    }

    // Regression test for the KV-cache flattening bug.
    //
    // OLD CODE: `convert_assistant_blocks` concatenated all thinking blocks into a
    // single flat string — `reasoning_content = Text("A\nB")`.  A chat template
    // given only that string can only reconstruct:
    //
    //     <think>A\nB</think> <call>t1</call> <call>t2</call>
    //
    // That token sequence diverges from what the model originally generated at the
    // very first `</think>`, so the KV cache misses on every multi-tool exchange.
    //
    // NEW CODE: `convert_assistant_blocks` produces `Segments(["A", "B", ""])` so a
    // template that understands segments can reconstruct byte-for-byte:
    //
    //     <think>A</think> <call>t1</call> <think>B</think> <call>t2</call>
    //
    // This test fails on the old code because the old code returns `Text("A\nB")` and
    // `.segments()` returns `None`, causing the `expect` below to panic.
    #[test]
    fn test_interleaved_reasoning_not_flattened_regression() {
        let msg = make_req(vec![
            thinking("A"),
            tool_use("t1"),
            thinking("B"),
            tool_use("t2"),
        ]);

        // Must be Segments, not Text.  Text("A\nB") is the old (broken) behaviour:
        // it loses which reasoning block preceded which tool call.
        assert!(
            !matches!(msg.reasoning_content, Some(ReasoningContent::Text(_))),
            "reasoning_content must NOT be flat Text when tool calls are interleaved; \
             Text loses positional info and forces a KV cache miss on every multi-tool turn"
        );

        let segs = msg
            .reasoning_content
            .as_ref()
            .expect("reasoning_content should be set")
            .segments()
            .expect(
                "must be Segments so a chat template can reconstruct \
                 <think>A</think><call>t1</call><think>B</think><call>t2</call> \
                 rather than front-loading all reasoning before all calls",
            );

        // segs[i] precedes tool_calls[i] — the invariant a template relies on
        assert_eq!(segs[0], "A", "reasoning before t1");
        assert_eq!(segs[1], "B", "reasoning before t2");
        assert_eq!(segs[2], "", "no trailing reasoning");

        let tools = msg.tool_calls.as_ref().unwrap();
        assert_eq!(tools[0].id, "t1");
        assert_eq!(tools[1].id, "t2");
    }

    #[test]
    fn test_per_block_cache_control_deserialization() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "World"}
                ]
            }]
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        match &req.messages[0].content {
            AnthropicMessageContent::Blocks { content } => {
                match &content[0] {
                    AnthropicContentBlock::Text { cache_control, .. } => {
                        assert!(cache_control.is_some());
                    }
                    other => panic!("expected Text, got {other:?}"),
                }
                match &content[1] {
                    AnthropicContentBlock::Text { cache_control, .. } => {
                        assert!(cache_control.is_none());
                    }
                    other => panic!("expected Text, got {other:?}"),
                }
            }
            _ => panic!("expected blocks"),
        }
    }

    #[test]
    fn test_system_string_no_cache_control() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}],
            "system": "You are helpful."
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        let system = req.system.as_ref().unwrap();
        assert_eq!(system.text, "You are helpful.");
        assert!(system.cache_control.is_none());
    }

    #[test]
    fn test_text_block_with_citations() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "According to the document...",
                        "citations": [
                            {"type": "char_location", "cited_text": "relevant text", "document_index": 0, "start_char_index": 0, "end_char_index": 13}
                        ]
                    }
                ]
            }]
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        match &req.messages[0].content {
            AnthropicMessageContent::Blocks { content } => match &content[0] {
                AnthropicContentBlock::Text { citations, .. } => {
                    assert!(citations.is_some());
                    let cites = citations.as_ref().unwrap();
                    assert_eq!(cites.len(), 1);
                    assert_eq!(cites[0]["type"], "char_location");
                }
                other => panic!("expected Text, got {other:?}"),
            },
            _ => panic!("expected blocks"),
        }
    }

    #[test]
    fn test_redacted_thinking_block() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "visible reasoning", "signature": "sig1"},
                    {"type": "redacted_thinking", "data": "base64-encrypted-data"},
                    {"type": "text", "text": "Final answer"}
                ]
            }]
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        match &req.messages[0].content {
            AnthropicMessageContent::Blocks { content } => {
                assert_eq!(content.len(), 3);
                assert!(matches!(
                    &content[0],
                    AnthropicContentBlock::Thinking { .. }
                ));
                match &content[1] {
                    AnthropicContentBlock::RedactedThinking { data } => {
                        assert_eq!(data, "base64-encrypted-data");
                    }
                    other => panic!("expected RedactedThinking, got {other:?}"),
                }
                assert!(matches!(&content[2], AnthropicContentBlock::Text { .. }));
            }
            _ => panic!("expected blocks"),
        }
    }

    #[test]
    fn test_server_tool_use_and_web_search_result() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "server_tool_use", "id": "stu_1", "name": "web_search", "input": {"query": "rust programming"}},
                    {"type": "web_search_tool_result", "tool_use_id": "stu_1", "content": [{"type": "web_search_result", "url": "https://www.rust-lang.org", "title": "Rust"}]},
                    {"type": "text", "text": "Based on my search..."}
                ]
            }]
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        match &req.messages[0].content {
            AnthropicMessageContent::Blocks { content } => {
                assert_eq!(content.len(), 3);
                match &content[0] {
                    AnthropicContentBlock::ServerToolUse { id, name, input } => {
                        assert_eq!(id, "stu_1");
                        assert_eq!(name, "web_search");
                        assert_eq!(input["query"], "rust programming");
                    }
                    other => panic!("expected ServerToolUse, got {other:?}"),
                }
                match &content[1] {
                    AnthropicContentBlock::WebSearchToolResult {
                        tool_use_id,
                        content,
                    } => {
                        assert_eq!(tool_use_id, "stu_1");
                        assert!(content.is_array());
                    }
                    other => panic!("expected WebSearchToolResult, got {other:?}"),
                }
            }
            _ => panic!("expected blocks"),
        }

        // ServerToolUse should convert to a tool call
        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        match &chat_req.inner.messages[0] {
            ChatCompletionRequestMessage::Assistant(a) => {
                let tc = a.tool_calls.as_ref().expect("should have tool calls");
                assert_eq!(tc.len(), 1);
                assert_eq!(tc[0].id, "stu_1");
                assert_eq!(tc[0].function.name, "web_search");
            }
            other => panic!("expected assistant, got {other:?}"),
        }
    }

    #[test]
    fn test_thinking_config_deserialization() {
        let json = r#"{
            "model": "test",
            "max_tokens": 16000,
            "messages": [{"role": "user", "content": "Solve this step by step"}],
            "thinking": {"type": "enabled", "budget_tokens": 10000}
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        let thinking = req.thinking.as_ref().expect("thinking should be set");
        assert_eq!(thinking.thinking_type, "enabled");
        assert_eq!(thinking.budget_tokens, Some(10000));
    }

    #[test]
    fn test_thinking_config_disabled() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {"type": "disabled"}
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        let thinking = req.thinking.as_ref().expect("thinking should be set");
        assert_eq!(thinking.thinking_type, "disabled");
        assert!(thinking.budget_tokens.is_none());
    }

    #[test]
    fn test_disable_parallel_tool_use() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [{"name": "get_weather", "description": "Get weather", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": true}
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        match &req.tool_choice {
            Some(AnthropicToolChoice::Simple(s)) => {
                assert_eq!(s.choice_type, AnthropicToolChoiceMode::Auto);
                assert_eq!(s.disable_parallel_tool_use, Some(true));
            }
            other => panic!("expected Simple tool choice, got {other:?}"),
        }
    }

    // --- Image passthrough tests ---

    #[test]
    fn test_image_block_becomes_multimodal_content() {
        let req = AnthropicCreateMessageRequest {
            model: "test-model".into(),
            max_tokens: 100,
            messages: vec![AnthropicMessage {
                role: AnthropicRole::User,
                content: AnthropicMessageContent::Blocks {
                    content: vec![
                        AnthropicContentBlock::Text {
                            text: "What is in this image?".into(),
                            citations: None,
                            cache_control: None,
                        },
                        AnthropicContentBlock::Image {
                            source: AnthropicImageSource {
                                source_type: "base64".into(),
                                media_type: "image/png".into(),
                                data: "iVBORw0KGgo=".into(), // tiny valid-ish base64
                            },
                        },
                    ],
                },
            }],
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            metadata: None,
            tools: None,
            tool_choice: None,
            cache_control: None,
            thinking: None,
            service_tier: None,
            container: None,
            output_config: None,
            unmodeled: Default::default(),
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert_eq!(chat_req.inner.messages.len(), 1);

        match &chat_req.inner.messages[0] {
            ChatCompletionRequestMessage::User(u) => match &u.content {
                ChatCompletionRequestUserMessageContent::Array(parts) => {
                    assert_eq!(parts.len(), 2);
                    // First part: text
                    match &parts[0] {
                        ChatCompletionRequestUserMessageContentPart::Text(t) => {
                            assert_eq!(t.text, "What is in this image?");
                        }
                        other => panic!("expected text part, got {other:?}"),
                    }
                    // Second part: image with data URI
                    match &parts[1] {
                        ChatCompletionRequestUserMessageContentPart::ImageUrl(img) => {
                            let url_str = img.image_url.url.to_string();
                            assert!(
                                url_str.starts_with("data:image/png;base64,"),
                                "expected data URI, got: {url_str}"
                            );
                            assert!(url_str.contains("iVBORw0KGgo="));
                        }
                        other => panic!("expected image_url part, got {other:?}"),
                    }
                }
                other => panic!("expected Array content, got {other:?}"),
            },
            other => panic!("expected user message, got {other:?}"),
        }
    }

    #[test]
    fn test_pure_text_stays_text_format() {
        // Verify backwards compatibility: pure text messages don't use Array format.
        let req = AnthropicCreateMessageRequest {
            model: "test-model".into(),
            max_tokens: 100,
            messages: vec![AnthropicMessage {
                role: AnthropicRole::User,
                content: AnthropicMessageContent::Blocks {
                    content: vec![
                        AnthropicContentBlock::Text {
                            text: "Hello ".into(),
                            citations: None,
                            cache_control: None,
                        },
                        AnthropicContentBlock::Text {
                            text: "world".into(),
                            citations: None,
                            cache_control: None,
                        },
                    ],
                },
            }],
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            metadata: None,
            tools: None,
            tool_choice: None,
            cache_control: None,
            thinking: None,
            service_tier: None,
            container: None,
            output_config: None,
            unmodeled: Default::default(),
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        match &chat_req.inner.messages[0] {
            ChatCompletionRequestMessage::User(u) => match &u.content {
                ChatCompletionRequestUserMessageContent::Text(t) => {
                    // Adjacent user text blocks join with "\n" (separator ruling 2026-09-03).
                    assert_eq!(t, "Hello \nworld");
                }
                other => panic!("expected Text content (not Array), got {other:?}"),
            },
            other => panic!("expected user message, got {other:?}"),
        }
    }

    #[test]
    fn test_image_with_tool_result_flush() {
        // Image + text should flush as Array before tool_result becomes a Tool message.
        let req = AnthropicCreateMessageRequest {
            model: "test-model".into(),
            max_tokens: 100,
            messages: vec![
                AnthropicMessage {
                    role: AnthropicRole::User,
                    content: AnthropicMessageContent::Text {
                        content: "What's the weather?".into(),
                    },
                },
                AnthropicMessage {
                    role: AnthropicRole::Assistant,
                    content: AnthropicMessageContent::Blocks {
                        content: vec![AnthropicContentBlock::ToolUse {
                            id: "tool_1".into(),
                            name: "screenshot".into(),
                            input: serde_json::json!({}),
                            cache_control: None,
                        }],
                    },
                },
                AnthropicMessage {
                    role: AnthropicRole::User,
                    content: AnthropicMessageContent::Blocks {
                        content: vec![
                            AnthropicContentBlock::Image {
                                source: AnthropicImageSource {
                                    source_type: "base64".into(),
                                    media_type: "image/jpeg".into(),
                                    data: "/9j/4AAQ".into(),
                                },
                            },
                            AnthropicContentBlock::ToolResult {
                                tool_use_id: "tool_1".into(),
                                content: Some(ToolResultContent::Text("screenshot taken".into())),
                                is_error: None,
                                cache_control: None,
                            },
                        ],
                    },
                },
            ],
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            metadata: None,
            tools: None,
            tool_choice: None,
            cache_control: None,
            thinking: None,
            service_tier: None,
            container: None,
            output_config: None,
            unmodeled: Default::default(),
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        // user("What's the weather?"), assistant(tool_use), user(image), tool("screenshot taken")
        assert_eq!(chat_req.inner.messages.len(), 4);

        // Third message: user with image (Array format, flushed before tool_result)
        match &chat_req.inner.messages[2] {
            ChatCompletionRequestMessage::User(u) => match &u.content {
                ChatCompletionRequestUserMessageContent::Array(parts) => {
                    assert_eq!(parts.len(), 1);
                    assert!(matches!(
                        &parts[0],
                        ChatCompletionRequestUserMessageContentPart::ImageUrl(_)
                    ));
                }
                other => panic!("expected Array content for image, got {other:?}"),
            },
            other => panic!("expected user message, got {other:?}"),
        }

        // Fourth message: tool result
        assert!(matches!(
            &chat_req.inner.messages[3],
            ChatCompletionRequestMessage::Tool(_)
        ));
    }
}
