// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Converts a stream of chat completion SSE chunks into Anthropic Messages API SSE events.
//!
//! The event sequence follows the Anthropic streaming spec:
//! `message_start` -> `content_block_start` -> N x `content_block_delta` ->
//! `content_block_stop` -> `message_delta` -> `message_stop`

use std::collections::HashSet;

use axum::response::sse::Event;
use dynamo_protocols::types::ChatCompletionMessageContent;
use uuid::Uuid;

use super::types::{
    AnthropicDelta, AnthropicErrorBody, AnthropicMessageDeltaBody, AnthropicMessageResponse,
    AnthropicResponseContentBlock, AnthropicStopReason, AnthropicStreamEvent, AnthropicUsage,
    completion_usage_to_anthropic, matched_stop_sequence, new_tool_use_id,
};
use crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;
use crate::protocols::unified::AnthropicContext;

/// State machine that converts a chat completion stream into Anthropic SSE events.
pub struct AnthropicStreamConverter {
    model: String,
    message_id: String,
    /// Preserved Anthropic-specific request context for faithful response reconstruction.
    api_context: Option<AnthropicContext>,
    // Thinking/reasoning tracking
    thinking_block_started: bool,
    thinking_block_closed: bool,
    thinking_block_index: u32,
    // Text tracking
    text_block_started: bool,
    text_block_closed: bool,
    text_block_index: u32,
    // Text arriving between a tool header and its final arguments must wait
    // until that tool block closes. Keep the original text delta boundaries.
    pending_text: Vec<String>,
    usage: AnthropicUsage,
    // Tool call tracking
    tool_call_states: Vec<ToolCallState>,
    tool_calls_sent: HashSet<String>,
    // Block index counter
    next_block_index: u32,
    // Stop reason
    finish_reason: Option<dynamo_protocols::types::FinishReason>,
    matched_stop: Option<String>,
    message_start_emitted: bool,
}

struct ToolCallState {
    id: String,
    name: String,
    accumulated_args: String,
    block_index: u32,
    started: bool,
    /// Set when `content_block_stop` has already been emitted inline
    /// (complete tool call detected mid-stream). Prevents duplicate stop in `emit_end_events()`.
    stopped: bool,
}

impl AnthropicStreamConverter {
    pub fn new(model: String) -> Self {
        Self {
            model,
            message_id: format!("msg_{}", Uuid::new_v4().simple()),
            api_context: None,
            thinking_block_started: false,
            thinking_block_closed: false,
            thinking_block_index: 0,
            text_block_started: false,
            text_block_closed: false,
            text_block_index: 0,
            pending_text: Vec::new(),
            usage: AnthropicUsage {
                cache_creation_input_tokens: Some(0),
                cache_read_input_tokens: Some(0),
                ..Default::default()
            },
            tool_call_states: Vec::new(),
            tool_calls_sent: HashSet::new(),
            next_block_index: 0,
            finish_reason: None,
            matched_stop: None,
            message_start_emitted: false,
        }
    }

    /// Create a converter seeded with the original Anthropic request context.
    /// This allows the response stream to carry forward metadata that was lost
    /// during the Anthropic-to-OpenAI request conversion.
    pub fn with_context(model: String, context: AnthropicContext) -> Self {
        let mut converter = Self::new(model);
        converter.api_context = Some(context);
        converter
    }

    /// Emit once, with authoritative input tokens or the zero fallback.
    /// The full recorded usage is kept separately for final reconciliation.
    fn emit_start_events_with<T>(
        &mut self,
        make_event: impl Fn(&str, &AnthropicStreamEvent) -> T,
        input_tokens: u32,
    ) -> Vec<T> {
        if self.message_start_emitted {
            return Vec::new();
        }
        self.message_start_emitted = true;
        // TODO: When AnthropicMessageResponse gains a `service_tier` field,
        // populate it from `self.api_context` (if the original request specified one).
        let message = AnthropicMessageResponse {
            id: self.message_id.clone(),
            object_type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![],
            model: self.model.clone(),
            stop_reason: None,
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens,
                ..self.usage.clone()
            },
        };

        let event = AnthropicStreamEvent::MessageStart { message };
        vec![make_event("message_start", &event)]
    }

    /// Process a single chat completion stream chunk and return zero or more SSE events.
    pub fn process_chunk(
        &mut self,
        chunk: &NvCreateChatCompletionStreamResponse,
    ) -> Vec<Result<Event, anyhow::Error>> {
        self.process_chunk_with(chunk, make_sse_event)
    }

    fn process_chunk_with<T>(
        &mut self,
        chunk: &NvCreateChatCompletionStreamResponse,
        make_event: impl Fn(&str, &AnthropicStreamEvent) -> T,
    ) -> Vec<T> {
        let mut initial_input_tokens = 0;
        if let Some(usage) = &chunk.inner.usage {
            self.usage = completion_usage_to_anthropic(usage);
            // Continuous usage may know prompt length before the backend reports
            // cache hits. Only an explicit cached count (including zero) is authoritative.
            if usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|details| details.cached_tokens)
                .is_some()
            {
                initial_input_tokens = self.usage.input_tokens;
            }
        }
        let mut events = self.emit_start_events_with(&make_event, initial_input_tokens);
        // Metadata can arrive separately from the finish reason (including on
        // a choices-empty usage chunk). Only retain request-owned stop strings.
        if let Some(matched) =
            matched_stop_sequence(chunk.nvext.as_ref(), self.api_context.as_ref())
        {
            self.matched_stop = Some(matched);
        }

        for choice in &chunk.inner.choices {
            let delta = &choice.delta;

            // Track finish reason
            if let Some(ref fr) = choice.finish_reason {
                self.finish_reason = Some(*fr);
            }

            // Handle reasoning/thinking content deltas
            if let Some(ref reasoning) = delta.reasoning_content
                && !reasoning.is_empty()
            {
                // Emit content_block_start on first thinking token
                if !self.thinking_block_started {
                    self.thinking_block_started = true;
                    self.thinking_block_index = self.next_block_index;
                    self.next_block_index += 1;

                    let block_start = AnthropicStreamEvent::ContentBlockStart {
                        index: self.thinking_block_index,
                        content_block: AnthropicResponseContentBlock::Thinking {
                            thinking: String::new(),
                            signature: String::new(),
                        },
                    };
                    events.push(make_event("content_block_start", &block_start));
                }

                // Emit thinking delta
                let block_delta = AnthropicStreamEvent::ContentBlockDelta {
                    index: self.thinking_block_index,
                    delta: AnthropicDelta::ThinkingDelta {
                        thinking: reasoning.clone(),
                    },
                };
                events.push(make_event("content_block_delta", &block_delta));
            }

            // Handle text content deltas
            let content_text = match &delta.content {
                Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                _ => None,
            };

            if let Some(text) = content_text
                && !text.is_empty()
            {
                // Close thinking block before text starts (Anthropic spec: thinking → text → tool_use)
                if self.thinking_block_started && !self.thinking_block_closed {
                    self.thinking_block_closed = true;
                    // Emit signature delta to close the thinking block.
                    // The engine doesn't produce Anthropic-style cryptographic signatures,
                    // so we use "erased" (the standard placeholder per the Anthropic spec).
                    // When `api_context` is available and the original request had
                    // `thinking.thinking_type == "enabled"`, this is expected — the backend
                    // simply doesn't generate real signatures. If/when the backend starts
                    // returning real signatures, we can use the context to validate or
                    // pass them through instead of hardcoding "erased".
                    let sig_delta = AnthropicStreamEvent::ContentBlockDelta {
                        index: self.thinking_block_index,
                        delta: AnthropicDelta::SignatureDelta {
                            signature: "erased".to_string(),
                        },
                    };
                    events.push(make_event("content_block_delta", &sig_delta));

                    let block_stop = AnthropicStreamEvent::ContentBlockStop {
                        index: self.thinking_block_index,
                    };
                    events.push(make_event("content_block_stop", &block_stop));
                }

                if self
                    .tool_call_states
                    .iter()
                    .any(|tc| tc.started && !tc.stopped)
                {
                    self.pending_text.push(text.to_string());
                } else {
                    self.emit_text(text, &mut events, &make_event);
                }
            }

            // Handle tool call deltas
            if let Some(tool_calls) = &delta.tool_calls {
                // Close thinking block before tool blocks (if text never appeared)
                if self.thinking_block_started && !self.thinking_block_closed {
                    self.thinking_block_closed = true;
                    let sig_delta = AnthropicStreamEvent::ContentBlockDelta {
                        index: self.thinking_block_index,
                        delta: AnthropicDelta::SignatureDelta {
                            signature: "erased".to_string(),
                        },
                    };
                    events.push(make_event("content_block_delta", &sig_delta));
                    let block_stop = AnthropicStreamEvent::ContentBlockStop {
                        index: self.thinking_block_index,
                    };
                    events.push(make_event("content_block_stop", &block_stop));
                }

                // Close the text block before opening any tool blocks.
                // Anthropic streaming spec requires each block to be closed
                // (content_block_stop) before the next block starts.
                if self.text_block_started && !self.text_block_closed {
                    self.text_block_closed = true;
                    let block_stop = AnthropicStreamEvent::ContentBlockStop {
                        index: self.text_block_index,
                    };
                    events.push(make_event("content_block_stop", &block_stop));
                }

                for tc in tool_calls {
                    let tc_index = tc.index as usize;

                    // Ensure we have state for this tool call index
                    while self.tool_call_states.len() <= tc_index {
                        let block_index = self.next_block_index;
                        self.next_block_index += 1;
                        self.tool_call_states.push(ToolCallState {
                            id: String::new(),
                            name: String::new(),
                            accumulated_args: String::new(),
                            block_index,
                            started: false,
                            stopped: false,
                        });
                    }

                    if tc.id.is_some() && self.tool_call_states[tc_index].id.is_empty() {
                        self.tool_call_states[tc_index].id = new_tool_use_id();
                    }
                    if let Some(func) = &tc.function {
                        if let Some(name) = &func.name {
                            self.tool_call_states[tc_index].name = name.clone();
                        }
                        if let Some(args) = &func.arguments {
                            // Emit content_block_start on first delta for this tool call
                            if !self.tool_call_states[tc_index].started {
                                let tc_id = self.tool_call_states[tc_index].id.clone();

                                // Dedup guard: skip if we've already emitted this tool call ID
                                if !tc_id.is_empty() && self.tool_calls_sent.contains(&tc_id) {
                                    continue;
                                }

                                self.tool_call_states[tc_index].started = true;
                                let block_index = self.tool_call_states[tc_index].block_index;
                                let tc_name = self.tool_call_states[tc_index].name.clone();

                                if !tc_id.is_empty() {
                                    self.tool_calls_sent.insert(tc_id.clone());
                                }

                                let block_start = AnthropicStreamEvent::ContentBlockStart {
                                    index: block_index,
                                    content_block: AnthropicResponseContentBlock::ToolUse {
                                        id: tc_id,
                                        name: tc_name,
                                        input: serde_json::json!({}),
                                    },
                                };
                                events.push(make_event("content_block_start", &block_start));
                            }

                            self.tool_call_states[tc_index]
                                .accumulated_args
                                .push_str(args);

                            let block_index = self.tool_call_states[tc_index].block_index;
                            let block_delta = AnthropicStreamEvent::ContentBlockDelta {
                                index: block_index,
                                delta: AnthropicDelta::InputJsonDelta {
                                    partial_json: args.clone(),
                                },
                            };
                            events.push(make_event("content_block_delta", &block_delta));

                            // Emit content_block_stop immediately if the tool call's
                            // arguments have been fully accumulated and parse as valid
                            // JSON. Backends that emit a complete tool call in one
                            // chunk (e.g. trtllm with `id+name+args` packed) close the
                            // block here on the same chunk. Backends that stream args
                            // incrementally (e.g. `minimax_m2` parser, which dribbles
                            // `""`, `{"file_path":...`, `, "content":...`, `}` across
                            // multiple chunks) do not close until `accumulated_args`
                            // parses, then `emit_end_events` finalizes on stream end.
                            //
                            // Without the JSON-parse guard, the very first chunk —
                            // which carries `id` and `name` but only an empty/prefix
                            // `arguments` — would emit `content_block_stop` before any
                            // `input_json_delta` carrying real arguments arrived.
                            // Anthropic SSE consumers (Claude Code, Anthropic SDK)
                            // close the block on `content_block_stop` and discard
                            // subsequent deltas as orphans, leaving callers with
                            // `tool_use.input == {}` and a deterministic
                            // `InputValidationError` retry loop.
                            if !self.tool_call_states[tc_index].id.is_empty()
                                && !self.tool_call_states[tc_index].name.is_empty()
                                && !self.tool_call_states[tc_index].stopped
                                && serde_json::from_str::<serde_json::Value>(
                                    &self.tool_call_states[tc_index].accumulated_args,
                                )
                                .is_ok()
                            {
                                self.tool_call_states[tc_index].stopped = true;
                                let block_stop =
                                    AnthropicStreamEvent::ContentBlockStop { index: block_index };
                                events.push(make_event("content_block_stop", &block_stop));
                            }
                        }
                    }
                }
            }
            if self
                .tool_call_states
                .iter()
                .all(|tc| !tc.started || tc.stopped)
            {
                self.flush_pending_text(&mut events, &make_event);
            }
        }

        events
    }

    fn emit_text<T>(
        &mut self,
        text: &str,
        events: &mut Vec<T>,
        make_event: &impl Fn(&str, &AnthropicStreamEvent) -> T,
    ) {
        if !self.text_block_started || self.text_block_closed {
            self.text_block_started = true;
            self.text_block_closed = false;
            self.text_block_index = self.next_block_index;
            self.next_block_index += 1;
            events.push(make_event(
                "content_block_start",
                &AnthropicStreamEvent::ContentBlockStart {
                    index: self.text_block_index,
                    content_block: AnthropicResponseContentBlock::Text {
                        text: String::new(),
                        citations: None,
                    },
                },
            ));
        }
        events.push(make_event(
            "content_block_delta",
            &AnthropicStreamEvent::ContentBlockDelta {
                index: self.text_block_index,
                delta: AnthropicDelta::TextDelta {
                    text: text.to_string(),
                },
            },
        ));
    }

    fn flush_pending_text<T>(
        &mut self,
        events: &mut Vec<T>,
        make_event: &impl Fn(&str, &AnthropicStreamEvent) -> T,
    ) {
        for text in std::mem::take(&mut self.pending_text) {
            self.emit_text(&text, events, make_event);
        }
    }

    /// Emit the final events when the stream ends.
    pub fn emit_end_events(&mut self) -> Vec<Result<Event, anyhow::Error>> {
        self.emit_end_events_with(make_sse_event)
    }

    fn emit_end_events_with<T>(
        &mut self,
        make_event: impl Fn(&str, &AnthropicStreamEvent) -> T,
    ) -> Vec<T> {
        // An empty successful stream must still open before it closes.
        let mut events = self.emit_start_events_with(&make_event, 0);

        // Close thinking block if started and not already closed mid-stream
        if self.thinking_block_started && !self.thinking_block_closed {
            self.thinking_block_closed = true;
            let sig_delta = AnthropicStreamEvent::ContentBlockDelta {
                index: self.thinking_block_index,
                delta: AnthropicDelta::SignatureDelta {
                    signature: "erased".to_string(),
                },
            };
            events.push(make_event("content_block_delta", &sig_delta));
            let block_stop = AnthropicStreamEvent::ContentBlockStop {
                index: self.thinking_block_index,
            };
            events.push(make_event("content_block_stop", &block_stop));
        }

        // Finish tools before releasing text that arrived while a tool was open.
        for tc in &mut self.tool_call_states {
            if tc.started && !tc.stopped {
                tc.stopped = true;
                let block_stop = AnthropicStreamEvent::ContentBlockStop {
                    index: tc.block_index,
                };
                events.push(make_event("content_block_stop", &block_stop));
            }
        }
        self.flush_pending_text(&mut events, &make_event);

        if self.text_block_started && !self.text_block_closed {
            self.text_block_closed = true;
            let block_stop = AnthropicStreamEvent::ContentBlockStop {
                index: self.text_block_index,
            };
            events.push(make_event("content_block_stop", &block_stop));
        }

        // Emit message_delta with stop_reason and real token usage from engine
        use dynamo_protocols::types::FinishReason;
        let stop_reason = self.finish_reason.map(|reason| match reason {
            FinishReason::Stop if self.matched_stop.is_some() => AnthropicStopReason::StopSequence,
            FinishReason::Stop | FinishReason::ContentFilter => AnthropicStopReason::EndTurn,
            FinishReason::Length => AnthropicStopReason::MaxTokens,
            FinishReason::ToolCalls | FinishReason::FunctionCall => AnthropicStopReason::ToolUse,
        });
        let stop_sequence = if stop_reason == Some(AnthropicStopReason::StopSequence) {
            self.matched_stop.clone()
        } else {
            None
        };
        let message_delta = AnthropicStreamEvent::MessageDelta {
            delta: AnthropicMessageDeltaBody {
                stop_reason,
                stop_sequence,
            },
            usage: self.usage.clone(),
        };
        events.push(make_event("message_delta", &message_delta));

        // Emit message_stop
        let message_stop = AnthropicStreamEvent::MessageStop {};
        events.push(make_event("message_stop", &message_stop));

        events
    }

    /// Emit error events when the stream ends due to a backend error.
    pub fn emit_error_events(&mut self) -> Vec<Result<Event, anyhow::Error>> {
        let error_event = AnthropicStreamEvent::Error {
            error: AnthropicErrorBody {
                error_type: "api_error".to_string(),
                message: "An internal error occurred during generation.".to_string(),
            },
        };
        vec![make_sse_event("error", &error_event)]
    }
}

fn make_sse_event(event_type: &str, event: &AnthropicStreamEvent) -> Result<Event, anyhow::Error> {
    let data = serde_json::to_string(event)?;
    Ok(Event::default().event(event_type).data(data))
}

/// A tagged event for testing: the event type string paired with the
/// serialized stream event. This avoids needing to parse `axum::sse::Event`
/// (which doesn't implement `Display`).
#[cfg(test)]
#[derive(Debug)]
struct TaggedEvent {
    event_type: String,
    data: AnthropicStreamEvent,
}

#[cfg(test)]
fn make_tagged_event(event_type: &str, event: &AnthropicStreamEvent) -> TaggedEvent {
    TaggedEvent {
        event_type: event_type.to_string(),
        data: event.clone(),
    }
}

#[cfg(test)]
impl AnthropicStreamConverter {
    // Exercise the same event generation as the production SSE writer.
    fn process_chunk_tagged(
        &mut self,
        chunk: &NvCreateChatCompletionStreamResponse,
    ) -> Vec<TaggedEvent> {
        self.process_chunk_with(chunk, make_tagged_event)
    }

    fn emit_end_events_tagged(&mut self) -> Vec<TaggedEvent> {
        self.emit_end_events_with(make_tagged_event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_protocols::types::{
        ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionMessageToolCallChunk,
        ChatCompletionStreamResponseDelta, CompletionUsage, FunctionCallStream, FunctionType,
    };

    fn text_chunk(text: &str) -> NvCreateChatCompletionStreamResponse {
        #[allow(deprecated)]
        NvCreateChatCompletionStreamResponse {
            inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
                id: "chat-1".into(),
                choices: vec![ChatChoiceStream {
                    index: 0,
                    delta: ChatCompletionStreamResponseDelta {
                        content: Some(ChatCompletionMessageContent::Text(text.into())),
                        function_call: None,
                        tool_calls: None,
                        role: None,
                        refusal: None,
                        reasoning_content: None,
                    },
                    finish_reason: None,
                    logprobs: None,
                }],
                created: 0,
                model: "test".into(),
                service_tier: None,
                system_fingerprint: None,
                object: "chat.completion.chunk".into(),
                usage: None,
            },
            nvext: None,
        }
    }

    fn tool_call_chunk(
        tc_index: u32,
        id: Option<&str>,
        name: Option<&str>,
        args: Option<&str>,
    ) -> NvCreateChatCompletionStreamResponse {
        #[allow(deprecated)]
        NvCreateChatCompletionStreamResponse {
            inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
                id: "chat-1".into(),
                choices: vec![ChatChoiceStream {
                    index: 0,
                    delta: ChatCompletionStreamResponseDelta {
                        content: None,
                        function_call: None,
                        tool_calls: Some(vec![ChatCompletionMessageToolCallChunk {
                            index: tc_index,
                            id: id.map(String::from),
                            r#type: Some(FunctionType::Function),
                            function: Some(FunctionCallStream {
                                name: name.map(String::from),
                                arguments: args.map(String::from),
                            }),
                        }]),
                        role: None,
                        refusal: None,
                        reasoning_content: None,
                    },
                    finish_reason: None,
                    logprobs: None,
                }],
                created: 0,
                model: "test".into(),
                service_tier: None,
                system_fingerprint: None,
                object: "chat.completion.chunk".into(),
                usage: None,
            },
            nvext: None,
        }
    }

    /// Real Nemotron replay: whitespace, tool header, whitespace, then arguments.
    /// The second text delta must not target the text block closed by the header.
    #[test]
    fn test_text_between_tool_header_and_arguments_gets_a_new_block() {
        let mut conv = AnthropicStreamConverter::new("nemotron".into());
        let mut events = conv.process_chunk_tagged(&text_chunk("\n"));
        events.extend(conv.process_chunk_tagged(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("get_weather"),
            Some(""),
        )));
        let pending = conv.process_chunk_tagged(&text_chunk("\n"));
        assert!(
            pending.is_empty(),
            "text waits for the open tool block to close"
        );
        events.extend(conv.process_chunk_tagged(&tool_call_chunk(
            0,
            None,
            None,
            Some(r#"{"city":"Tokyo"}"#),
        )));
        events.extend(conv.emit_end_events_tagged());
        assert_text_tool_text_blocks(&events, r#"{"city":"Tokyo"}"#);
    }

    #[test]
    fn test_text_after_tool_and_text_pending_at_stream_end_are_preserved() {
        for args in [r#"{"city":"Tokyo"}"#, r#"{"city":"#] {
            let mut conv = AnthropicStreamConverter::new("nemotron".into());
            let mut events = conv.process_chunk_tagged(&text_chunk("\n"));
            events.extend(conv.process_chunk_tagged(&tool_call_chunk(
                0,
                Some("call-1"),
                Some("get_weather"),
                Some(args),
            )));
            events.extend(conv.process_chunk_tagged(&text_chunk("\n")));
            events.extend(conv.emit_end_events_tagged());
            assert_text_tool_text_blocks(&events, args);
        }
    }

    fn assert_text_tool_text_blocks(events: &[TaggedEvent], expected_args: &str) {
        let mut active = None;
        let mut texts = Vec::new();
        let mut arguments = String::new();
        let mut started = Vec::new();
        for event in events {
            match &event.data {
                AnthropicStreamEvent::ContentBlockStart { index, .. } => {
                    assert!(
                        active.is_none(),
                        "previous block must close before next start"
                    );
                    active = Some(*index);
                    started.push(*index);
                }
                AnthropicStreamEvent::ContentBlockDelta { index, delta } => {
                    assert_eq!(active, Some(*index), "delta must target the open block");
                    match delta {
                        AnthropicDelta::TextDelta { text } => texts.push((*index, text.as_str())),
                        AnthropicDelta::InputJsonDelta { partial_json } => {
                            arguments.push_str(partial_json)
                        }
                        _ => panic!("unexpected delta: {delta:?}"),
                    }
                }
                AnthropicStreamEvent::ContentBlockStop { index } => {
                    assert_eq!(
                        active.take(),
                        Some(*index),
                        "stop must close the open block once"
                    );
                }
                _ => assert!(active.is_none(), "all blocks close before message end"),
            }
        }
        assert_eq!(started, vec![0, 1, 2]);
        assert_eq!(texts, vec![(0, "\n"), (2, "\n")]);
        assert_eq!(arguments, expected_args);
        assert!(active.is_none());
    }

    fn event_types(events: &[TaggedEvent]) -> Vec<&str> {
        events.iter().map(|e| e.event_type.as_str()).collect()
    }

    /// A chunk carrying engine usage (typically the final chunk).
    fn usage_chunk(
        prompt_tokens: u32,
        cached_tokens: Option<u32>,
        completion_tokens: u32,
    ) -> NvCreateChatCompletionStreamResponse {
        #[allow(deprecated)]
        NvCreateChatCompletionStreamResponse {
            inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
                id: "chat-1".into(),
                choices: vec![],
                created: 0,
                model: "test".into(),
                service_tier: None,
                system_fingerprint: None,
                object: "chat.completion.chunk".into(),
                usage: Some(dynamo_protocols::types::CompletionUsage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                    prompt_tokens_details: cached_tokens.map(|c| {
                        dynamo_protocols::types::PromptTokensDetails {
                            audio_tokens: None,
                            cached_tokens: Some(c),
                        }
                    }),
                    completion_tokens_details: None,
                }),
            },
            nvext: None,
        }
    }

    /// Streaming usage starts at zero, then updates to the engine's total
    /// prompt tokens minus its cached-token count.
    #[test]
    fn test_streaming_input_tokens_reconciled_from_engine_usage() {
        let mut conv = AnthropicStreamConverter::new("test-model".into());

        assert_eq!(conv.usage.input_tokens, 0);

        // Exercise the production chunk path rather than its tagged test mirror.
        let events = conv.process_chunk(&usage_chunk(12, Some(11), 5));
        assert_eq!(
            events.len(),
            1,
            "usage-only first chunk emits message_start but no content events"
        );
        assert_eq!(conv.usage.input_tokens, 1);
        assert_eq!(conv.usage.cache_read_input_tokens, Some(11));
        assert_eq!(conv.usage.cache_creation_input_tokens, Some(0));
        assert_eq!(conv.usage.output_tokens, 5);

        let delta = conv.emit_end_events_tagged();
        let message_delta = delta
            .iter()
            .find(|e| e.event_type == "message_delta")
            .expect("message_delta present");
        match &message_delta.data {
            AnthropicStreamEvent::MessageDelta { usage, .. } => {
                assert_eq!(usage.input_tokens, 1);
                assert_eq!(usage.cache_read_input_tokens, Some(11));
                assert_eq!(usage.cache_creation_input_tokens, Some(0));
                assert_eq!(usage.output_tokens, 5);
            }
            other => panic!("expected MessageDelta, got {other:?}"),
        }
    }

    /// MP-1550: check the actual SSE wire, including metadata arriving on a
    /// separate choices-empty chunk before or after the finish reason.
    #[tokio::test]
    async fn test_production_sse_matched_stop() {
        use dynamo_protocols::types::FinishReason;
        for (finish, matched, expected, sequence) in [
            (
                FinishReason::Stop,
                Some("END"),
                "stop_sequence",
                Some("END"),
            ),
            (FinishReason::Stop, Some("<|return|>"), "end_turn", None),
            (FinishReason::Stop, None, "end_turn", None),
            (FinishReason::ToolCalls, Some("END"), "tool_use", None),
            (FinishReason::FunctionCall, Some("END"), "tool_use", None),
            (FinishReason::Length, Some("END"), "max_tokens", None),
            (FinishReason::ContentFilter, Some("END"), "end_turn", None),
        ] {
            for metadata_first in [false, true] {
                let mut conv = AnthropicStreamConverter::with_context(
                    "m".into(),
                    AnthropicContext {
                        stop_sequences: vec!["END".into()],
                        ..Default::default()
                    },
                );
                let mut terminal = text_chunk("");
                terminal.inner.choices[0].finish_reason = Some(finish);
                let mut metadata = usage_chunk(12, Some(11), 5);
                metadata.nvext = Some(serde_json::json!({"matched_stop": matched}));
                let chunks = if metadata_first {
                    [&metadata, &terminal]
                } else {
                    [&terminal, &metadata]
                };
                let mut events = conv.process_chunk(chunks[0]);
                events.extend(conv.process_chunk(chunks[1]));
                events.extend(conv.emit_end_events());
                let frames = sse_frames(events).await;
                let delta = &frames.iter().find(|(n, _)| n == "message_delta").unwrap().1["delta"];
                assert_eq!(
                    delta["stop_reason"], expected,
                    "{finish:?}, {matched:?}, metadata_first={metadata_first}"
                );
                assert_eq!(delta["stop_sequence"], serde_json::json!(sequence));
            }
        }
    }

    #[tokio::test]
    async fn test_backend_error_does_not_emit_successful_end() {
        for with_content in [false, true] {
            let mut conv = AnthropicStreamConverter::new("m".into());
            let mut events = if with_content {
                conv.process_chunk(&text_chunk("Hi"))
            } else {
                vec![]
            };
            events.extend(conv.emit_error_events());
            let frames = sse_frames(events).await;
            assert_eq!(frames.last().unwrap().0, "error");
            assert!(
                !frames
                    .iter()
                    .any(|(n, _)| n == "message_delta" || n == "message_stop")
            );
        }
    }

    /// Parse an SSE body into `(event name, data JSON)` frames.
    async fn sse_frames(
        events: Vec<Result<Event, anyhow::Error>>,
    ) -> Vec<(String, serde_json::Value)> {
        use axum::response::IntoResponse;
        let events: Vec<Result<Event, std::convert::Infallible>> =
            events.into_iter().map(|e| Ok(e.expect("event"))).collect();
        let response = axum::response::sse::Sse::new(futures::stream::iter(events)).into_response();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let text = String::from_utf8(body.to_vec()).expect("utf8");
        text.split("\n\n")
            .filter(|frame| !frame.trim().is_empty())
            .map(|frame| {
                let name = frame
                    .lines()
                    .find_map(|l| l.strip_prefix("event: "))
                    .expect("event name")
                    .to_string();
                let data = frame
                    .lines()
                    .find_map(|l| l.strip_prefix("data: "))
                    .expect("data");
                (name, serde_json::from_str(data).expect("json"))
            })
            .collect()
    }

    /// MP-1654, production path: the first frame actually written to the SSE
    /// body is `message_start`, its usage comes from the first chunk (so the
    /// start event must be built after that chunk's usage is recorded), it is
    /// written exactly once, and both cache fields are present on the wire.
    #[tokio::test]
    async fn test_production_sse_message_start_carries_first_chunk_usage() {
        let mut conv = AnthropicStreamConverter::new("test-model".into());

        let mut first = text_chunk("OK");
        first.inner.usage = Some(CompletionUsage {
            prompt_tokens: 7125,
            completion_tokens: 1,
            total_tokens: 7126,
            prompt_tokens_details: Some(dynamo_protocols::types::PromptTokensDetails {
                audio_tokens: None,
                cached_tokens: Some(7100),
            }),
            completion_tokens_details: None,
        });
        let mut events = conv.process_chunk(&first);
        events.extend(conv.process_chunk(&text_chunk(".")));
        events.extend(conv.process_chunk(&usage_chunk(7125, Some(7100), 3)));
        events.extend(conv.emit_end_events());

        let frames = sse_frames(events).await;
        let names: Vec<&str> = frames.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        let start = &frames[0].1["message"]["usage"];
        assert_eq!(start["input_tokens"], 25);
        assert_eq!(start["cache_read_input_tokens"], 7100);
        assert_eq!(start["cache_creation_input_tokens"], 0);
        assert_eq!(start["output_tokens"], 1);
        let end_usage = &frames.iter().find(|(n, _)| n == "message_delta").unwrap().1["usage"];
        assert_eq!(end_usage["output_tokens"], 3);
        assert_eq!(end_usage["input_tokens"], 25);
        assert_eq!(end_usage["cache_read_input_tokens"], 7100);
    }

    /// Exercise the real generator: it emits provisional prompt usage before
    /// token backends report cache metadata, often only on the terminal chunk.
    #[tokio::test]
    async fn test_generator_to_converter_initial_usage_requires_cache_metadata() {
        use crate::protocols::common::llm_backend::{BackendOutput, FinishReason};
        use crate::protocols::openai::{
            DeltaGeneratorExt, chat_completions::NvCreateChatCompletionRequest,
        };
        use dynamo_protocols::types::PromptTokensDetails;

        let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m", "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "stream_options": {"include_usage": true, "continuous_usage_stats": true}
        }))
        .unwrap();
        for (first_cached, final_cached, expected_start_input) in [
            (None, 7100, 0),
            (Some(None), 7100, 0),
            (None, 0, 0),
            (Some(Some(0)), 0, 7125),
            (Some(Some(7100)), 7100, 25),
        ] {
            let mut generator = request.response_generator("cache-usage".into());
            generator.update_isl(7125);
            let mut first: BackendOutput = serde_json::from_value(serde_json::json!({
                "token_ids": [1], "tokens": ["OK"], "text": "OK", "index": 0
            }))
            .unwrap();
            first.completion_usage = first_cached.map(|cached_tokens| CompletionUsage {
                prompt_tokens_details: Some(PromptTokensDetails {
                    audio_tokens: None,
                    cached_tokens,
                }),
                ..usage_chunk(7125, None, 1).inner.usage.unwrap()
            });
            let first_chunk = generator.choice_from_postprocessor(first.clone()).unwrap();
            assert_eq!(
                first_chunk.inner.usage.as_ref().unwrap().prompt_tokens,
                7125
            );
            let mut conv = AnthropicStreamConverter::new("m".into());
            let mut frames = sse_frames(conv.process_chunk(&first_chunk)).await;
            assert_eq!(
                frames
                    .iter()
                    .map(|(name, _)| name.as_str())
                    .collect::<Vec<_>>(),
                [
                    "message_start",
                    "content_block_start",
                    "content_block_delta"
                ],
                "unknown cache usage must not buffer content"
            );
            let start = &frames[0].1["message"]["usage"];
            assert_eq!(
                start["input_tokens"], expected_start_input,
                "{first_cached:?}"
            );
            assert_eq!(
                start["cache_read_input_tokens"],
                first_cached.flatten().unwrap_or(0)
            );
            assert_eq!(start["cache_creation_input_tokens"], 0);
            assert_eq!(start["output_tokens"], 1);

            let last = BackendOutput {
                token_ids: vec![2],
                tokens: vec![Some(".".into())],
                text: Some(".".into()),
                finish_reason: Some(FinishReason::Stop),
                completion_usage: usage_chunk(7125, Some(final_cached), 2).inner.usage,
                ..first
            };
            let last_chunk = generator.choice_from_postprocessor(last).unwrap();
            let mut events = conv.process_chunk(&last_chunk);
            events.extend(conv.process_chunk(&generator.create_usage_chunk()));
            events.extend(conv.emit_end_events());
            frames.extend(sse_frames(events).await);
            let end = &frames
                .iter()
                .find(|(name, _)| name == "message_delta")
                .unwrap()
                .1["usage"];
            assert_eq!(end["input_tokens"], 7125 - final_cached);
            assert_eq!(end["cache_read_input_tokens"], final_cached);
            assert_eq!(end["output_tokens"], 2);
            assert_eq!(
                frames
                    .iter()
                    .filter(|(name, _)| name == "message_start")
                    .count(),
                1
            );
        }
    }

    /// Missing cache metadata uses a zero start, but still reconciles at end;
    /// `cache_read_input_tokens: 0` is explicit on both events.
    #[tokio::test]
    async fn test_production_sse_cold_prompt_writes_explicit_zero_cache_read() {
        let mut conv = AnthropicStreamConverter::new("test-model".into());
        let mut events = conv.process_chunk(&usage_chunk(7039, None, 1));
        events.extend(conv.emit_end_events());

        let frames = sse_frames(events).await;
        let start = &frames[0].1;
        assert_eq!(start["type"], "message_start");
        assert_eq!(start["message"]["usage"]["input_tokens"], 0);
        assert_eq!(start["message"]["usage"]["cache_read_input_tokens"], 0);
        let delta = frames
            .iter()
            .find(|(n, _)| n == "message_delta")
            .map(|(_, v)| v)
            .expect("message_delta");
        assert_eq!(delta["usage"]["input_tokens"], 7039);
        assert_eq!(delta["usage"]["cache_read_input_tokens"], 0);
        assert_eq!(
            frames.iter().filter(|(n, _)| n == "message_start").count(),
            1
        );
    }

    /// A stream that ends without any chunk still opens with `message_start`.
    #[test]
    fn test_empty_stream_emits_message_start_at_end() {
        let mut conv = AnthropicStreamConverter::new("test-model".into());
        let end = conv.emit_end_events_tagged();
        assert_eq!(
            event_types(&end),
            vec!["message_start", "message_delta", "message_stop"]
        );
    }

    /// Backends that only report usage on the final chunk still get a leading
    /// `message_start` (zero counts, explicit cache fields) on the first chunk.
    #[test]
    fn test_message_start_without_first_chunk_usage_still_leads() {
        let mut conv = AnthropicStreamConverter::new("test-model".into());
        let events = conv.process_chunk_tagged(&text_chunk("Hi"));
        let usage = match &events[0].data {
            AnthropicStreamEvent::MessageStart { message } => &message.usage,
            _ => panic!("message_start must lead"),
        };
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.cache_read_input_tokens, Some(0));
        assert_eq!(usage.cache_creation_input_tokens, Some(0));
    }

    /// Regression test: text block must be closed (content_block_stop)
    /// before the tool_use block starts (content_block_start).
    ///
    /// Without this fix, the text block stop was batched at the end,
    /// causing Claude Code's streaming parser to receive out-of-order
    /// events and fail to execute tool calls ("Error editing file").
    #[test]
    fn test_text_block_stops_before_tool_block_starts() {
        let mut conv = AnthropicStreamConverter::new("test-model".into());
        // Keep these assertions focused on content-block ordering.
        let _ = conv.emit_start_events_with(make_tagged_event, 0);

        // Stream some text
        let text_events = conv.process_chunk_tagged(&text_chunk("I'll edit the file."));
        assert_eq!(
            event_types(&text_events),
            vec!["content_block_start", "content_block_delta"]
        );

        // Stream a tool call — text block must close first
        let tool_events = conv.process_chunk_tagged(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("Edit"),
            Some("{\"file_path\":\"/tmp/test.txt\"}"),
        ));

        assert_eq!(
            event_types(&tool_events),
            vec![
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
            ],
            "text block must be closed before tool block starts; complete tool call stopped inline"
        );

        // Verify indices: stop=0 (text), start=1 (tool)
        match &tool_events[0].data {
            AnthropicStreamEvent::ContentBlockStop { index } => assert_eq!(*index, 0),
            other => panic!("expected ContentBlockStop, got {other:?}"),
        }
        match &tool_events[1].data {
            AnthropicStreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                assert_eq!(*index, 1);
                match content_block {
                    AnthropicResponseContentBlock::ToolUse { id, name, .. } => {
                        assert!(id.starts_with("toolu_"));
                        assert!(!id.contains("call-1"));
                        assert_eq!(name, "Edit");
                    }
                    other => panic!("expected ToolUse, got {other:?}"),
                }
            }
            other => panic!("expected ContentBlockStart, got {other:?}"),
        }

        // End events should NOT duplicate either stop (both already emitted inline)
        let end_events = conv.emit_end_events_tagged();
        assert_eq!(
            event_types(&end_events),
            vec!["message_delta", "message_stop"],
            "no block stops in end events (both text and tool already closed inline)"
        );
    }

    #[test]
    fn test_streaming_tool_use_id_is_rewritten_to_toolu_prefix() {
        let mut conv = AnthropicStreamConverter::new("test-model".into());
        let events = conv.process_chunk_tagged(&tool_call_chunk(
            0,
            Some("chatcmpl-tool-DEADBEEF"),
            Some("Edit"),
            Some("{\"file_path\":\"/tmp/test.txt\"}"),
        ));

        let id = events
            .iter()
            .find_map(|event| match &event.data {
                AnthropicStreamEvent::ContentBlockStart {
                    content_block: AnthropicResponseContentBlock::ToolUse { id, .. },
                    ..
                } => Some(id.clone()),
                _ => None,
            })
            .expect("expected tool_use content block start");
        assert!(
            id.starts_with("toolu_"),
            "tool_use.id must start with toolu_, got {id}"
        );
        assert!(
            !id.contains("chatcmpl-tool-"),
            "upstream id format must not leak, got {id}"
        );
    }

    /// Tool-only response (no preceding text): no spurious stop events.
    #[test]
    fn test_tool_only_response_no_text_block() {
        let mut conv = AnthropicStreamConverter::new("test-model".into());
        // Keep these assertions focused on content-block ordering.
        let _ = conv.emit_start_events_with(make_tagged_event, 0);

        let tool_events = conv.process_chunk_tagged(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("Read"),
            Some("{\"path\":\"/tmp/test.txt\"}"),
        ));
        assert_eq!(
            event_types(&tool_events),
            vec![
                "content_block_start",
                "content_block_delta",
                "content_block_stop"
            ],
            "complete tool call emits stop inline"
        );

        let end_events = conv.emit_end_events_tagged();
        assert_eq!(
            event_types(&end_events),
            vec!["message_delta", "message_stop"],
            "no block stop in end events (already stopped inline)"
        );
    }

    /// Text-only response: stop emitted in end events (no early close).
    #[test]
    fn test_text_only_response_stop_in_end_events() {
        let mut conv = AnthropicStreamConverter::new("test-model".into());

        conv.process_chunk_tagged(&text_chunk("Hello world"));

        let end_events = conv.emit_end_events_tagged();
        assert_eq!(
            event_types(&end_events),
            vec!["content_block_stop", "message_delta", "message_stop"]
        );
        match &end_events[0].data {
            AnthropicStreamEvent::ContentBlockStop { index } => assert_eq!(*index, 0),
            other => panic!("expected text stop at index 0, got {other:?}"),
        }
    }

    fn reasoning_chunk(text: &str) -> NvCreateChatCompletionStreamResponse {
        #[allow(deprecated)]
        NvCreateChatCompletionStreamResponse {
            inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
                id: "chat-1".into(),
                choices: vec![ChatChoiceStream {
                    index: 0,
                    delta: ChatCompletionStreamResponseDelta {
                        content: None,
                        function_call: None,
                        tool_calls: None,
                        role: None,
                        refusal: None,
                        reasoning_content: Some(text.into()),
                    },
                    finish_reason: None,
                    logprobs: None,
                }],
                created: 0,
                model: "test".into(),
                service_tier: None,
                system_fingerprint: None,
                object: "chat.completion.chunk".into(),
                usage: None,
            },
            nvext: None,
        }
    }

    /// Full reasoning flow: thinking → text → tool_use.
    /// Verifies block ordering (thinking=0, text=1, tool=2) and that each
    /// block is properly closed before the next one starts.
    #[test]
    fn test_thinking_text_then_tool_call() {
        let mut conv = AnthropicStreamConverter::new("test-model".into());
        // Keep these assertions focused on content-block ordering.
        let _ = conv.emit_start_events_with(make_tagged_event, 0);

        // 1. Reasoning tokens → thinking block starts
        let ev = conv.process_chunk_tagged(&reasoning_chunk("Let me think..."));
        assert_eq!(
            event_types(&ev),
            vec!["content_block_start", "content_block_delta"]
        );
        assert!(matches!(
            &ev[0].data,
            AnthropicStreamEvent::ContentBlockStart {
                index: 0,
                content_block: AnthropicResponseContentBlock::Thinking { .. }
            }
        ));

        // 2. Text arrives → thinking block closes (signature + stop), text block opens
        let ev = conv.process_chunk_tagged(&text_chunk("Hello!"));
        assert_eq!(
            event_types(&ev),
            vec![
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta"
            ]
        );
        assert!(matches!(
            &ev[1].data,
            AnthropicStreamEvent::ContentBlockStop { index: 0 }
        ));
        assert!(matches!(
            &ev[2].data,
            AnthropicStreamEvent::ContentBlockStart { index: 1, .. }
        ));

        // 3. Tool call → text block closes, tool block opens at index 2.
        //    Because the tool call arrives complete (id + name + args in one
        //    chunk), inline dispatch also emits content_block_stop immediately.
        let ev = conv.process_chunk_tagged(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("Read"),
            Some("{\"path\":\"/tmp/test.txt\"}"),
        ));
        assert_eq!(
            event_types(&ev),
            vec![
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop"
            ]
        );
        assert!(matches!(
            &ev[0].data,
            AnthropicStreamEvent::ContentBlockStop { index: 1 }
        ));
        assert!(matches!(
            &ev[1].data,
            AnthropicStreamEvent::ContentBlockStart { index: 2, .. }
        ));
    }

    /// Thinking-only response (no text/tool follows): thinking block closed in end events.
    #[test]
    fn test_thinking_only_closed_in_end_events() {
        let mut conv = AnthropicStreamConverter::new("test-model".into());
        conv.process_chunk_tagged(&reasoning_chunk("Deep thought..."));

        let ev = conv.emit_end_events_tagged();
        assert_eq!(
            event_types(&ev),
            vec![
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
    }

    /// Multiple tool calls: each gets inline content_block_stop.
    #[test]
    fn test_multiple_tool_calls_each_stopped_inline() {
        let mut conv = AnthropicStreamConverter::new("test-model".into());
        // Keep these assertions focused on content-block ordering.
        let _ = conv.emit_start_events_with(make_tagged_event, 0);

        let events1 = conv.process_chunk_tagged(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("Read"),
            Some("{\"path\":\"/tmp/a.txt\"}"),
        ));
        assert_eq!(
            event_types(&events1),
            vec![
                "content_block_start",
                "content_block_delta",
                "content_block_stop"
            ],
            "first tool call closed inline"
        );

        let events2 = conv.process_chunk_tagged(&tool_call_chunk(
            1,
            Some("call-2"),
            Some("Write"),
            Some("{\"path\":\"/tmp/b.txt\"}"),
        ));
        assert_eq!(
            event_types(&events2),
            vec![
                "content_block_start",
                "content_block_delta",
                "content_block_stop"
            ],
            "second tool call closed inline"
        );

        // End events: no block stops (both already closed)
        let end_events = conv.emit_end_events_tagged();
        assert_eq!(
            event_types(&end_events),
            vec!["message_delta", "message_stop"],
            "no block stops in end events"
        );
    }

    /// Verify that `with_context` stores the context and produces the same
    /// event structure as `new` — the context is carried for future enrichment.
    #[test]
    fn test_with_context_preserves_context() {
        use crate::protocols::unified::AnthropicContext;

        let ctx = AnthropicContext {
            service_tier: Some("priority".to_string()),
            ..Default::default()
        };
        let mut conv = AnthropicStreamConverter::with_context("test-model".into(), ctx);
        let _ = conv.emit_start_events_with(make_tagged_event, 0);
        assert!(conv.api_context.is_some());
        assert_eq!(
            conv.api_context.as_ref().unwrap().service_tier.as_deref(),
            Some("priority")
        );

        // Should produce the same events as a regular converter
        let ev = conv.process_chunk_tagged(&text_chunk("Hello"));
        assert_eq!(
            event_types(&ev),
            vec!["content_block_start", "content_block_delta"]
        );

        let end = conv.emit_end_events_tagged();
        assert_eq!(
            event_types(&end),
            vec!["content_block_stop", "message_delta", "message_stop"]
        );
    }

    /// Regression: tool_use args streamed across multiple chunks must NOT close
    /// the content block until the accumulated arguments parse as valid JSON.
    ///
    /// Reproduces the on-the-wire SSE order observed against MiniMax-M2.5 with
    /// the `minimax_m2` tool-call parser (Claude Code session
    /// `~/.claude/projects/-workspaces-demo-failure/23742d15-…`):
    ///
    ///   chunk 1: id=call-1, name=Write, args=""             → start + delta(empty)
    ///   chunk 2: args=`{"file_path":"whoami.txt"`           → delta
    ///   chunk 3: args=`, "content":"MiniMaxAI/MiniMax-M2.5"`→ delta
    ///   chunk 4: args=`}`                                   → delta + stop (JSON now parses)
    ///
    /// Without the JSON-parse guard the inline stop fired on chunk 1, and
    /// chunks 2-4 emitted orphan deltas that Anthropic SSE consumers
    /// (Claude Code, Anthropic SDK) discard, surfacing as `tool_use.input == {}`
    /// and a deterministic `InputValidationError: required parameter X is missing`
    /// loop on every Claude Code Write/Bash/Edit invocation.
    #[test]
    fn test_streamed_tool_args_close_only_when_json_complete() {
        let mut conv = AnthropicStreamConverter::new("test-model".into());
        // Keep these assertions focused on content-block ordering.
        let _ = conv.emit_start_events_with(make_tagged_event, 0);

        // Chunk 1: id + name + empty args prefix. Block opens, empty delta
        // emitted, but block must NOT close (args don't parse yet).
        let ev1 =
            conv.process_chunk_tagged(&tool_call_chunk(0, Some("call-1"), Some("Write"), Some("")));
        assert_eq!(
            event_types(&ev1),
            vec!["content_block_start", "content_block_delta"],
            "first chunk: open block + empty delta, no premature stop"
        );

        // Chunk 2: partial args, still not parseable.
        let ev2 = conv.process_chunk_tagged(&tool_call_chunk(
            0,
            None,
            None,
            Some("{\"file_path\":\"whoami.txt\""),
        ));
        assert_eq!(
            event_types(&ev2),
            vec!["content_block_delta"],
            "partial-args chunk: delta only, no stop"
        );

        // Chunk 3: more partial args, still not parseable.
        let ev3 = conv.process_chunk_tagged(&tool_call_chunk(
            0,
            None,
            None,
            Some(", \"content\":\"MiniMaxAI/MiniMax-M2.5\""),
        ));
        assert_eq!(
            event_types(&ev3),
            vec!["content_block_delta"],
            "still-partial-args chunk: delta only, no stop"
        );

        // Chunk 4: closing brace. Accumulated args now parse as valid JSON,
        // so the inline-stop fires on this chunk.
        let ev4 = conv.process_chunk_tagged(&tool_call_chunk(0, None, None, Some("}")));
        assert_eq!(
            event_types(&ev4),
            vec!["content_block_delta", "content_block_stop"],
            "completing-args chunk: delta + inline stop (JSON now parses)"
        );

        // End events: no leftover block stop (already closed inline on chunk 4).
        let end_events = conv.emit_end_events_tagged();
        assert_eq!(
            event_types(&end_events),
            vec!["message_delta", "message_stop"],
            "no leftover block stop in end events"
        );
    }

    /// Regression: streamed tool_use whose args never parse (e.g. truncated by
    /// `max_tokens`) must close in `emit_end_events`, not be left dangling.
    #[test]
    fn test_streamed_tool_args_unclosed_finalized_in_end_events() {
        let mut conv = AnthropicStreamConverter::new("test-model".into());
        // Keep these assertions focused on content-block ordering.
        let _ = conv.emit_start_events_with(make_tagged_event, 0);

        let ev1 =
            conv.process_chunk_tagged(&tool_call_chunk(0, Some("call-1"), Some("Write"), Some("")));
        assert_eq!(
            event_types(&ev1),
            vec!["content_block_start", "content_block_delta"],
            "open block + empty delta, no inline stop"
        );

        let ev2 = conv.process_chunk_tagged(&tool_call_chunk(
            0,
            None,
            None,
            Some("{\"file_path\":\"truncated"),
        ));
        assert_eq!(
            event_types(&ev2),
            vec!["content_block_delta"],
            "partial-args chunk: delta only"
        );

        // Stream ends with args still incomplete. emit_end_events must close
        // the block so consumers see a well-formed (if argument-empty) tool_use
        // rather than a dangling content block.
        let end_events = conv.emit_end_events_tagged();
        assert_eq!(
            event_types(&end_events),
            vec!["content_block_stop", "message_delta", "message_stop"],
            "end events close the dangling tool block"
        );
    }

    /// Regression: full minimax-m2 Claude Code session captured on the wire.
    ///
    /// The OpenAI-format chunks below mirror what came back from the engine
    /// when Claude Code asked
    ///     user:      "What model are you?"
    ///     assistant: "I am MiniMaxAI/MiniMax-M2.5."
    ///     user:      "write that to whoami.txt"
    /// against the production deployment on 2026-05-08. The pre-fix Anthropic
    /// SSE re-emission closed the tool_use block on the first (empty-args)
    /// delta and orphaned the four following `input_json_delta` events,
    /// surfacing as `tool_use.input == {}` and a deterministic
    /// `InputValidationError: required parameter file_path/content is missing`
    /// retry loop in Claude Code (see Slack thread `1777841579.980589` and
    /// session `~/.claude/projects/-workspaces-demo-failure/23742d15-…`).
    ///
    /// Asserts the full event sequence emitted by the converter:
    ///
    ///     content_block_start (thinking, idx=0)
    ///     content_block_delta * N (thinking_delta)
    ///     content_block_delta (signature_delta) + content_block_stop (idx=0)
    ///     content_block_start (text, idx=1) + content_block_delta + content_block_stop
    ///     content_block_start (tool_use, idx=2)
    ///     content_block_delta (input_json_delta, partial_json="")
    ///     content_block_delta (input_json_delta, partial_json="{\"file_path\":\"whoami.txt\"")
    ///     content_block_delta (input_json_delta, partial_json=", \"content\":\"MiniMaxAI/MiniMax-M2.5\"")
    ///     content_block_delta (input_json_delta, partial_json="}") + content_block_stop (idx=2)
    ///     message_delta (stop_reason=tool_use) + message_stop
    ///
    /// Critically: every `input_json_delta` for index 2 must arrive *before*
    /// `content_block_stop` for index 2.
    #[test]
    fn test_minimax_m2_claude_code_session_replay() {
        let mut conv = AnthropicStreamConverter::new("MiniMaxAI/MiniMax-M2.5".into());

        // 1. Thinking block: a few reasoning tokens, then text starts which
        //    forces the thinking block closed (signature_delta + stop).
        let mut events: Vec<TaggedEvent> = Vec::new();
        events.extend(conv.process_chunk_tagged(&reasoning_chunk("The user wants me ")));
        events.extend(conv.process_chunk_tagged(&reasoning_chunk("to write to whoami.txt.")));

        // 2. Text content (Claude Code session captured "\n\n\n" between
        //    thinking and the tool call — mirror that here).
        events.extend(conv.process_chunk_tagged(&text_chunk("\n\n\n")));

        // 3. Tool call: minimax_m2 parser streams arguments incrementally.
        //    chunk a: id + name + empty args prefix.
        //    chunk b..d: args dribble in across three more chunks.
        events.extend(conv.process_chunk_tagged(&tool_call_chunk(
            0,
            Some("chatcmpl-tool-x"),
            Some("Write"),
            Some(""),
        )));
        events.extend(conv.process_chunk_tagged(&tool_call_chunk(
            0,
            None,
            None,
            Some("{\"file_path\":\"whoami.txt\""),
        )));
        events.extend(conv.process_chunk_tagged(&tool_call_chunk(
            0,
            None,
            None,
            Some(", \"content\":\"MiniMaxAI/MiniMax-M2.5\""),
        )));
        events.extend(conv.process_chunk_tagged(&tool_call_chunk(0, None, None, Some("}"))));

        // 4. End-of-stream finalization.
        events.extend(conv.emit_end_events_tagged());

        // Locate every event for the tool_use block (index 2) and assert the
        // input_json_delta events all precede content_block_stop. This is the
        // exact ordering invariant the pre-fix code violated.
        let tool_block_events: Vec<&TaggedEvent> = events
            .iter()
            .filter(|e| {
                matches!(
                    &e.data,
                    AnthropicStreamEvent::ContentBlockStart { index: 2, .. }
                        | AnthropicStreamEvent::ContentBlockDelta { index: 2, .. }
                        | AnthropicStreamEvent::ContentBlockStop { index: 2, .. }
                )
            })
            .collect();

        let stop_pos = tool_block_events
            .iter()
            .position(|e| matches!(e.data, AnthropicStreamEvent::ContentBlockStop { .. }))
            .expect("tool_use block must be closed");

        // No event for index 2 may follow the content_block_stop.
        assert_eq!(
            stop_pos,
            tool_block_events.len() - 1,
            "content_block_stop for tool_use (idx=2) must be the last event for that block; \
             pre-fix code emitted input_json_delta events after stop, which Anthropic SSE \
             consumers (Claude Code, Anthropic SDK) discard as orphans → tool_use.input == {{}}"
        );

        // Concrete assertion on the tool block's event types:
        //   start, delta(""), delta("{...}"), delta(", ..."), delta("}"), stop
        let kinds: Vec<&str> = tool_block_events
            .iter()
            .map(|e| e.event_type.as_str())
            .collect();
        assert_eq!(
            kinds,
            vec![
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
            ],
            "tool_use block must receive 4 input_json_delta events (\"\", \
             \"{{\\\"file_path\\\":...\", \", \\\"content\\\":...\", \"}}\") \
             before close"
        );

        // The reconstructed JSON should match what Claude Code expects.
        let mut accumulated = String::new();
        for e in &tool_block_events {
            if let AnthropicStreamEvent::ContentBlockDelta {
                delta: AnthropicDelta::InputJsonDelta { partial_json },
                ..
            } = &e.data
            {
                accumulated.push_str(partial_json);
            }
        }
        assert_eq!(
            accumulated, r#"{"file_path":"whoami.txt", "content":"MiniMaxAI/MiniMax-M2.5"}"#,
            "concatenated input_json_delta payloads must reconstruct the full tool args"
        );
        let parsed: serde_json::Value =
            serde_json::from_str(&accumulated).expect("accumulated args must be valid JSON");
        assert_eq!(parsed["file_path"], "whoami.txt");
        assert_eq!(parsed["content"], "MiniMaxAI/MiniMax-M2.5");

        // Sanity: the stream as a whole ends cleanly with message_delta +
        // message_stop and no leftover open blocks.
        let last_two: Vec<&str> = events
            .iter()
            .rev()
            .take(2)
            .map(|e| e.event_type.as_str())
            .collect();
        assert_eq!(last_two, vec!["message_stop", "message_delta"]);
    }
}
