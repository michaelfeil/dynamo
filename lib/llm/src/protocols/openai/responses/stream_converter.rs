// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Converts a stream of chat completion SSE chunks into Responses API SSE events.
//!
//! The event sequence follows the OpenAI Responses API streaming spec:
//! `response.created` -> `response.in_progress` -> `response.output_item.added` ->
//! `response.content_part.added` -> N x `response.output_text.delta` ->
//! `response.output_text.done` -> `response.content_part.done` ->
//! `response.output_item.done` -> `response.completed` -> `[DONE]`

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::response::sse::Event;
use dynamo_protocols::types::responses::{
    AssistantRole, ErrorObject, FunctionToolCall, IncompleteDetails, InputTokenDetails,
    Instructions, OutputContent, OutputItem, OutputMessage, OutputMessageContent, OutputStatus,
    OutputTextContent, OutputTokenDetails, ReasoningItem, Response, ResponseCompletedEvent,
    ResponseContentPartAddedEvent, ResponseContentPartDoneEvent, ResponseCreatedEvent,
    ResponseFailedEvent, ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent, ResponseInProgressEvent, ResponseIncompleteEvent,
    ResponseOutputItemAddedEvent, ResponseOutputItemDoneEvent,
    ResponseReasoningSummaryPartAddedEvent, ResponseReasoningSummaryPartDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent, ResponseReasoningSummaryTextDoneEvent,
    ResponseStreamEvent, ResponseTextDeltaEvent, ResponseTextDoneEvent, ResponseTextParam,
    ResponseUsage, ServiceTier, Status, SummaryPart, SummaryTextContent,
    TextResponseFormatConfiguration, ToolChoiceOptions, ToolChoiceParam, Truncation,
};
use uuid::Uuid;

use dynamo_protocols::types::{ChatCompletionMessageContent, FinishReason};

use super::ResponseParams;
use crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;
use crate::protocols::unified::ResponsesContext;

/// State machine that converts a chat completion stream into Responses API events.
pub struct ResponseStreamConverter {
    response_id: String,
    model: String,
    params: ResponseParams,
    /// Preserved Responses API-specific request context for faithful response reconstruction.
    api_context: Option<ResponsesContext>,
    created_at: u64,
    sequence_number: u64,
    // Text message tracking
    message_item_id: String,
    message_started: bool,
    message_output_index: u32,
    accumulated_text: String,
    reasoning_item_id: String,
    reasoning_started: bool,
    reasoning_done: bool,
    reasoning_output_index: u32,
    reasoning_output_status: Option<OutputStatus>,
    accumulated_reasoning: String,
    // Function call tracking
    function_call_items: Vec<FunctionCallState>,
    // Output index counter
    next_output_index: u32,
    // Usage stats from the backend's final chunk
    usage: Option<ResponseUsage>,
    output_limit_reached: bool,
}

struct FunctionCallState {
    item_id: String,
    call_id: String,
    name: String,
    accumulated_args: String,
    output_index: u32,
    started: bool,
    /// Set when done/item_done events have already been emitted
    /// (on `finish_reason`). Prevents duplicate in `emit_end_events()`.
    done: bool,
}

impl ResponseStreamConverter {
    pub fn new(model: String, params: ResponseParams) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            response_id: format!("resp_{}", Uuid::new_v4().simple()),
            model,
            params,
            api_context: None,
            created_at,
            sequence_number: 0,
            message_item_id: format!("msg_{}", Uuid::new_v4().simple()),
            message_started: false,
            message_output_index: 0,
            accumulated_text: String::new(),
            reasoning_item_id: format!("rs_{}", Uuid::new_v4().simple()),
            reasoning_started: false,
            reasoning_done: false,
            reasoning_output_index: 0,
            reasoning_output_status: None,
            accumulated_reasoning: String::new(),
            function_call_items: Vec::new(),
            next_output_index: 0,
            usage: None,
            output_limit_reached: false,
        }
    }

    pub fn with_context(model: String, params: ResponseParams, context: ResponsesContext) -> Self {
        let mut converter = Self::new(model, params);
        converter.api_context = Some(context);
        converter
    }

    fn next_seq(&mut self) -> u64 {
        let seq = self.sequence_number;
        self.sequence_number += 1;
        seq
    }

    /// Wire identity — (name, namespace) — for a model-emitted tool name.
    /// Namespace members are declared to the worker under mangled
    /// `{namespace}__{name}` flat names (names may overlap between groups);
    /// emitted function_call items must carry the member's original name and
    /// its group's namespace, since codex dispatches on the exact pair.
    fn resolve_tool_identity(&self, chat_name: &str) -> (String, Option<String>) {
        super::resolve_tool_identity(self.params.tools.as_deref(), chat_name)
    }

    fn make_response(&self, status: Status, output: Vec<OutputItem>) -> Response {
        let is_incomplete = status == Status::Incomplete;
        let completed_at = if status == Status::Completed {
            Some(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            )
        } else {
            None
        };
        Response {
            id: self.response_id.clone(),
            object: "response".to_string(),
            created_at: self.created_at,
            completed_at,
            status,
            model: self.model.clone(),
            output,
            // Echo request params with spec-required defaults for omitted fields
            background: Some(false),
            metadata: Some(HashMap::new()),
            parallel_tool_calls: self.params.parallel_tool_calls.or(Some(true)),
            temperature: self.params.temperature.or(Some(1.0)),
            text: Some(self.params.text.clone().unwrap_or(ResponseTextParam {
                format: TextResponseFormatConfiguration::Text,
                verbosity: None,
            })),
            tool_choice: self
                .params
                .tool_choice
                .clone()
                .or(Some(ToolChoiceParam::Mode(ToolChoiceOptions::Auto))),
            tools: Some(
                self.params
                    .tools
                    .clone()
                    .map(super::normalize_tools)
                    .unwrap_or_default(),
            ),
            top_p: self.params.top_p.or(Some(1.0)),
            truncation: Some(self.params.truncation.unwrap_or(Truncation::Disabled)),
            // Nullable required fields
            billing: None,
            conversation: None,
            error: None,
            incomplete_details: is_incomplete.then(|| IncompleteDetails {
                reason: "max_output_tokens".to_string(),
            }),
            instructions: self.params.instructions.clone().map(Instructions::Text),
            max_output_tokens: self.params.max_output_tokens,
            previous_response_id: self
                .api_context
                .as_ref()
                .and_then(|ctx| ctx.previous_response_id.clone()),
            prompt: None,
            prompt_cache_key: self.params.prompt_cache_key.clone(),
            prompt_cache_retention: self.params.prompt_cache_retention,
            reasoning: self.params.reasoning.clone(),
            safety_identifier: self.params.safety_identifier.clone(),
            service_tier: Some(self.params.service_tier.unwrap_or(ServiceTier::Auto)),
            top_logprobs: Some(0),
            usage: self.usage.clone(),
        }
    }

    /// Emit the initial lifecycle events: created + in_progress.
    pub fn emit_start_events(&mut self) -> Vec<Result<Event, anyhow::Error>> {
        let mut events = Vec::with_capacity(2);

        let created = ResponseStreamEvent::ResponseCreated(ResponseCreatedEvent {
            sequence_number: self.next_seq(),
            response: self.make_response(Status::InProgress, vec![]),
        });
        events.push(self.make_sse_event(&created));

        let in_progress = ResponseStreamEvent::ResponseInProgress(ResponseInProgressEvent {
            sequence_number: self.next_seq(),
            response: self.make_response(Status::InProgress, vec![]),
        });
        events.push(self.make_sse_event(&in_progress));

        events
    }

    /// Process a single chat completion stream chunk and return zero or more SSE events.
    pub fn process_chunk(
        &mut self,
        chunk: &NvCreateChatCompletionStreamResponse,
    ) -> Vec<Result<Event, anyhow::Error>> {
        let mut events = Vec::new();

        // Capture usage stats from the final chunk (sent when stream_options.include_usage=true)
        if let Some(ref u) = chunk.inner.usage {
            self.usage = Some(ResponseUsage {
                input_tokens: u.prompt_tokens,
                input_tokens_details: InputTokenDetails {
                    cached_tokens: u
                        .prompt_tokens_details
                        .as_ref()
                        .and_then(|d| d.cached_tokens)
                        .unwrap_or(0),
                },
                output_tokens: u.completion_tokens,
                output_tokens_details: OutputTokenDetails {
                    reasoning_tokens: u
                        .completion_tokens_details
                        .as_ref()
                        .and_then(|d| d.reasoning_tokens)
                        .unwrap_or(0),
                },
                total_tokens: u.total_tokens,
            });
        }

        for choice in &chunk.inner.choices {
            let delta = &choice.delta;

            if choice.finish_reason == Some(FinishReason::Length) {
                self.output_limit_reached = true;
            }

            if let Some(reasoning) = delta.reasoning_content.as_deref()
                && !reasoning.is_empty()
                && !self.reasoning_done
                && self.params.reasoning_summary_requested()
            {
                self.accumulated_reasoning.push_str(reasoning);
                if !self.reasoning_started {
                    self.reasoning_started = true;
                    self.reasoning_output_index = self.next_output_index;
                    let output_index = self.reasoning_output_index;
                    self.next_output_index += 1;

                    let item_added = ResponseStreamEvent::ResponseOutputItemAdded(
                        ResponseOutputItemAddedEvent {
                            sequence_number: self.next_seq(),
                            output_index,
                            item: OutputItem::Reasoning(ReasoningItem {
                                id: self.reasoning_item_id.clone(),
                                summary: vec![],
                                content: None,
                                encrypted_content: None,
                                status: Some(OutputStatus::InProgress),
                            }),
                        },
                    );
                    events.push(self.make_sse_event(&item_added));

                    let part_added = ResponseStreamEvent::ResponseReasoningSummaryPartAdded(
                        ResponseReasoningSummaryPartAddedEvent {
                            sequence_number: self.next_seq(),
                            item_id: self.reasoning_item_id.clone(),
                            output_index,
                            summary_index: 0,
                            part: SummaryPart::SummaryText(SummaryTextContent {
                                text: String::new(),
                            }),
                        },
                    );
                    events.push(self.make_sse_event(&part_added));
                }

                let reasoning_delta = ResponseStreamEvent::ResponseReasoningSummaryTextDelta(
                    ResponseReasoningSummaryTextDeltaEvent {
                        sequence_number: self.next_seq(),
                        item_id: self.reasoning_item_id.clone(),
                        output_index: self.reasoning_output_index,
                        summary_index: 0,
                        delta: reasoning.to_string(),
                    },
                );
                events.push(self.make_sse_event(&reasoning_delta));
            }

            // Handle text content deltas — extract text from the enum
            let content_text = match &delta.content {
                Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                Some(ChatCompletionMessageContent::Parts(_)) => {
                    // Multimodal streaming not yet supported
                    None
                }
                None => None,
            };
            if let Some(content) = content_text
                && !content.is_empty()
            {
                self.append_reasoning_done_events(&mut events, OutputStatus::Completed);

                // Emit output_item.added + content_part.added on first text
                if !self.message_started {
                    self.message_started = true;
                    self.message_output_index = self.next_output_index;
                    let output_index = self.message_output_index;
                    self.next_output_index += 1;

                    let item_added = ResponseStreamEvent::ResponseOutputItemAdded(
                        ResponseOutputItemAddedEvent {
                            sequence_number: self.next_seq(),
                            output_index,
                            item: OutputItem::Message(OutputMessage {
                                id: self.message_item_id.clone(),
                                content: vec![],
                                role: AssistantRole::Assistant,
                                phase: None,
                                status: OutputStatus::InProgress,
                            }),
                        },
                    );
                    events.push(self.make_sse_event(&item_added));

                    let part_added = ResponseStreamEvent::ResponseContentPartAdded(
                        ResponseContentPartAddedEvent {
                            sequence_number: self.next_seq(),
                            item_id: self.message_item_id.clone(),
                            output_index,
                            content_index: 0,
                            part: OutputContent::OutputText(OutputTextContent {
                                text: String::new(),
                                annotations: vec![],
                                logprobs: Some(vec![]),
                            }),
                        },
                    );
                    events.push(self.make_sse_event(&part_added));
                }

                // Emit text delta
                self.accumulated_text.push_str(content);
                let text_delta =
                    ResponseStreamEvent::ResponseOutputTextDelta(ResponseTextDeltaEvent {
                        sequence_number: self.next_seq(),
                        item_id: self.message_item_id.clone(),
                        output_index: self.message_output_index,
                        content_index: 0,
                        delta: content.to_string(),
                        logprobs: Some(vec![]),
                    });
                events.push(self.make_sse_event(&text_delta));
            }

            // Handle tool call deltas
            if let Some(tool_calls) = &delta.tool_calls {
                if !tool_calls.is_empty() {
                    self.append_reasoning_done_events(&mut events, OutputStatus::Completed);
                }
                for tc in tool_calls {
                    let tc_index = tc.index as usize;

                    // Start a new function call if we haven't seen this index
                    while self.function_call_items.len() <= tc_index {
                        let output_index = self.next_output_index;
                        self.next_output_index += 1;
                        self.function_call_items.push(FunctionCallState {
                            item_id: format!("fc_{}", Uuid::new_v4().simple()),
                            call_id: String::new(),
                            name: String::new(),
                            accumulated_args: String::new(),
                            output_index,
                            started: false,
                            done: false,
                        });
                    }

                    // Update call_id and name if provided
                    if let Some(id) = &tc.id {
                        self.function_call_items[tc_index].call_id = id.clone();
                    }
                    if let Some(func) = &tc.function {
                        if let Some(name) = &func.name {
                            self.function_call_items[tc_index].name = name.clone();
                        }
                        if let Some(args) = &func.arguments {
                            // Emit output_item.added on first delta for this function call
                            if !self.function_call_items[tc_index].started {
                                self.function_call_items[tc_index].started = true;
                                let item_id = self.function_call_items[tc_index].item_id.clone();
                                let call_id = self.function_call_items[tc_index].call_id.clone();
                                let fc_name = self.function_call_items[tc_index].name.clone();
                                let output_index = self.function_call_items[tc_index].output_index;
                                let seq = self.next_seq();
                                let (wire_name, namespace) = self.resolve_tool_identity(&fc_name);
                                let item_added = ResponseStreamEvent::ResponseOutputItemAdded(
                                    ResponseOutputItemAddedEvent {
                                        sequence_number: seq,
                                        output_index,
                                        item: OutputItem::FunctionCall(FunctionToolCall {
                                            id: Some(item_id),
                                            call_id,
                                            namespace,
                                            name: wire_name,
                                            arguments: String::new(),
                                            status: Some(OutputStatus::InProgress),
                                        }),
                                    },
                                );
                                events.push(self.make_sse_event(&item_added));
                            }

                            self.function_call_items[tc_index]
                                .accumulated_args
                                .push_str(args);
                            let output_index = self.function_call_items[tc_index].output_index;

                            let item_id = self.function_call_items[tc_index].item_id.clone();
                            let seq = self.next_seq();
                            let args_delta =
                                ResponseStreamEvent::ResponseFunctionCallArgumentsDelta(
                                    ResponseFunctionCallArgumentsDeltaEvent {
                                        sequence_number: seq,
                                        item_id,
                                        output_index,
                                        delta: args.clone(),
                                    },
                                );
                            events.push(self.make_sse_event(&args_delta));
                        }
                    }
                }
            }

            // A finish_reason marks the end of the generation for this choice, so
            // every open tool-call argument stream is complete. Close them here so
            // done events carry the fully concatenated arguments. Backends that
            // fragment arguments across chunks (with id+name only on the first
            // fragment) make any earlier "looks complete" heuristic unsafe — done
            // events must wait for finish_reason (or stream end, in
            // `emit_end_events`).
            if choice.finish_reason.is_some() {
                for idx in 0..self.function_call_items.len() {
                    events.extend(self.close_function_call(idx));
                }
            }
        }

        events
    }

    /// Emit `function_call_arguments.done` + `output_item.done` for the function
    /// call at `idx`, if it has started streaming and is not yet closed.
    fn close_function_call(&mut self, idx: usize) -> Vec<Result<Event, anyhow::Error>> {
        let mut events = Vec::new();
        {
            let fc = &self.function_call_items[idx];
            if !fc.started || fc.done {
                return events;
            }
        }
        self.function_call_items[idx].done = true;
        // Truncated turns (finish_reason=length) mark the tool call incomplete.
        let output_status = self.output_status();
        let fc = &self.function_call_items[idx];
        let (item_id, call_id, fc_name, fc_args, output_index) = (
            fc.item_id.clone(),
            fc.call_id.clone(),
            fc.name.clone(),
            fc.accumulated_args.clone(),
            fc.output_index,
        );

        let (wire_name, namespace) = self.resolve_tool_identity(&fc_name);
        let args_done = ResponseStreamEvent::ResponseFunctionCallArgumentsDone(
            ResponseFunctionCallArgumentsDoneEvent {
                sequence_number: self.next_seq(),
                item_id: item_id.clone(),
                output_index,
                arguments: fc_args.clone(),
                name: Some(wire_name.clone()),
            },
        );
        events.push(self.make_sse_event(&args_done));

        let item_done = ResponseStreamEvent::ResponseOutputItemDone(ResponseOutputItemDoneEvent {
            sequence_number: self.next_seq(),
            output_index,
            item: OutputItem::FunctionCall(FunctionToolCall {
                id: Some(item_id),
                call_id,
                namespace,
                name: wire_name,
                arguments: fc_args,
                status: Some(output_status),
            }),
        });
        events.push(self.make_sse_event(&item_done));

        events
    }

    fn append_reasoning_done_events(
        &mut self,
        events: &mut Vec<Result<Event, anyhow::Error>>,
        output_status: OutputStatus,
    ) {
        if self.reasoning_done {
            return;
        }
        self.reasoning_done = true;
        if !self.reasoning_started {
            return;
        }
        self.reasoning_output_status = Some(output_status);

        // On truncation the reasoning stream was cut off before it could
        // conclude. Preserve the partial summary (OpenAI does the same) but
        // append an ellipsis so clients render a visible "truncated here"
        // marker, and emit it as a trailing delta so streaming consumers —
        // which already rendered the earlier deltas — pick it up too.
        if output_status == OutputStatus::Incomplete && !self.accumulated_reasoning.is_empty() {
            self.accumulated_reasoning.push_str("...");
            let ellipsis_delta = ResponseStreamEvent::ResponseReasoningSummaryTextDelta(
                ResponseReasoningSummaryTextDeltaEvent {
                    sequence_number: self.next_seq(),
                    item_id: self.reasoning_item_id.clone(),
                    output_index: self.reasoning_output_index,
                    summary_index: 0,
                    delta: "...".to_string(),
                },
            );
            events.push(self.make_sse_event(&ellipsis_delta));
        }

        let text_done = ResponseStreamEvent::ResponseReasoningSummaryTextDone(
            ResponseReasoningSummaryTextDoneEvent {
                sequence_number: self.next_seq(),
                item_id: self.reasoning_item_id.clone(),
                output_index: self.reasoning_output_index,
                summary_index: 0,
                text: self.accumulated_reasoning.clone(),
            },
        );
        events.push(self.make_sse_event(&text_done));

        let summary = SummaryPart::SummaryText(SummaryTextContent {
            text: self.accumulated_reasoning.clone(),
        });
        let part_done = ResponseStreamEvent::ResponseReasoningSummaryPartDone(
            ResponseReasoningSummaryPartDoneEvent {
                sequence_number: self.next_seq(),
                item_id: self.reasoning_item_id.clone(),
                output_index: self.reasoning_output_index,
                summary_index: 0,
                part: summary.clone(),
            },
        );
        events.push(self.make_sse_event(&part_done));

        let item_done = ResponseStreamEvent::ResponseOutputItemDone(ResponseOutputItemDoneEvent {
            sequence_number: self.next_seq(),
            output_index: self.reasoning_output_index,
            item: OutputItem::Reasoning(ReasoningItem {
                id: self.reasoning_item_id.clone(),
                summary: vec![summary],
                content: None,
                encrypted_content: None,
                status: Some(output_status),
            }),
        });
        events.push(self.make_sse_event(&item_done));
    }

    fn completed_output(&self) -> Vec<OutputItem> {
        self.collect_output(self.output_status())
    }

    fn collect_output(&self, output_status: OutputStatus) -> Vec<OutputItem> {
        let mut output = Vec::new();
        if self.reasoning_started {
            output.push((
                self.reasoning_output_index,
                OutputItem::Reasoning(ReasoningItem {
                    id: self.reasoning_item_id.clone(),
                    summary: vec![SummaryPart::SummaryText(SummaryTextContent {
                        text: self.accumulated_reasoning.clone(),
                    })],
                    content: None,
                    encrypted_content: None,
                    status: Some(self.reasoning_output_status.unwrap_or(output_status)),
                }),
            ));
        }
        if self.message_started {
            output.push((
                self.message_output_index,
                OutputItem::Message(OutputMessage {
                    id: self.message_item_id.clone(),
                    content: vec![OutputMessageContent::OutputText(OutputTextContent {
                        text: self.accumulated_text.clone(),
                        annotations: vec![],
                        logprobs: Some(vec![]),
                    })],
                    role: AssistantRole::Assistant,
                    phase: None,
                    status: output_status,
                }),
            ));
        }
        for function_call in &self.function_call_items {
            if function_call.started {
                let (wire_name, namespace) = self.resolve_tool_identity(&function_call.name);
                output.push((
                    function_call.output_index,
                    OutputItem::FunctionCall(FunctionToolCall {
                        id: Some(function_call.item_id.clone()),
                        call_id: function_call.call_id.clone(),
                        namespace,
                        name: wire_name,
                        arguments: function_call.accumulated_args.clone(),
                        status: Some(output_status),
                    }),
                ));
            }
        }
        output.sort_unstable_by_key(|(output_index, _)| *output_index);
        output.into_iter().map(|(_, item)| item).collect()
    }

    fn output_status(&self) -> OutputStatus {
        if self.output_limit_reached {
            OutputStatus::Incomplete
        } else {
            OutputStatus::Completed
        }
    }

    fn terminal_status(&self) -> Status {
        if self.output_limit_reached {
            Status::Incomplete
        } else {
            Status::Completed
        }
    }

    /// Emit the final events when the stream ends: done events + completed.
    pub fn emit_end_events(&mut self) -> Vec<Result<Event, anyhow::Error>> {
        let mut events = Vec::new();

        let output_status = self.output_status();
        self.append_reasoning_done_events(&mut events, output_status);

        // Close text message if it was started
        if self.message_started {
            let text_done = ResponseStreamEvent::ResponseOutputTextDone(ResponseTextDoneEvent {
                sequence_number: self.next_seq(),
                item_id: self.message_item_id.clone(),
                output_index: self.message_output_index,
                content_index: 0,
                text: self.accumulated_text.clone(),
                logprobs: Some(vec![]),
            });
            events.push(self.make_sse_event(&text_done));

            let part_done =
                ResponseStreamEvent::ResponseContentPartDone(ResponseContentPartDoneEvent {
                    sequence_number: self.next_seq(),
                    item_id: self.message_item_id.clone(),
                    output_index: self.message_output_index,
                    content_index: 0,
                    part: OutputContent::OutputText(OutputTextContent {
                        text: self.accumulated_text.clone(),
                        annotations: vec![],
                        logprobs: Some(vec![]),
                    }),
                });
            events.push(self.make_sse_event(&part_done));

            let item_done =
                ResponseStreamEvent::ResponseOutputItemDone(ResponseOutputItemDoneEvent {
                    sequence_number: self.next_seq(),
                    output_index: self.message_output_index,
                    item: OutputItem::Message(OutputMessage {
                        id: self.message_item_id.clone(),
                        content: vec![OutputMessageContent::OutputText(OutputTextContent {
                            text: self.accumulated_text.clone(),
                            annotations: vec![],
                            logprobs: Some(vec![]),
                        })],
                        role: AssistantRole::Assistant,
                        phase: None,
                        status: output_status,
                    }),
                });
            events.push(self.make_sse_event(&item_done));
        }

        // Close any function call items not already closed on finish_reason
        // (e.g. streams that end without a finish_reason chunk).
        for idx in 0..self.function_call_items.len() {
            events.extend(self.close_function_call(idx));
        }

        // Emit the terminal event from accumulated state.
        let terminal_status = self.terminal_status();
        let response = self.make_response(terminal_status.clone(), self.completed_output());
        let terminal = if terminal_status == Status::Incomplete {
            ResponseStreamEvent::ResponseIncomplete(ResponseIncompleteEvent {
                sequence_number: self.next_seq(),
                response,
            })
        } else {
            ResponseStreamEvent::ResponseCompleted(ResponseCompletedEvent {
                sequence_number: self.next_seq(),
                response,
            })
        };
        events.push(self.make_sse_event(&terminal));

        events
    }

    /// Emit error events when the stream ends due to a backend error.
    ///
    /// A truncation-shaped backend error is a `length` finish the worker
    /// mispresented as a failure (legacy chat processors raise "Tool calls
    /// cutoff by max_tokens." instead of finishing the turn); surface the
    /// spec-correct incomplete shape so clients keep the partial tool call.
    /// Genuine failures keep `response.failed`, but carry the error detail
    /// and whatever output had streamed instead of an empty response.
    pub fn emit_error_events(
        &mut self,
        error: Option<BackendError>,
    ) -> Vec<Result<Event, anyhow::Error>> {
        if error.as_ref().is_some_and(BackendError::is_truncation) {
            self.output_limit_reached = true;
            return self.emit_end_events();
        }

        let mut events = Vec::new();

        let output = self.collect_output(OutputStatus::Incomplete);
        let mut response = self.make_response(Status::Failed, output);
        response.error = Some(match error {
            Some(error) => ErrorObject {
                code: if error.http_status == 429 {
                    "rate_limit_exceeded".to_string()
                } else if error.is_context_overflow() {
                    // The exact string OpenAI clients classify on: codex, for
                    // one, matches `error.code == "context_length_exceeded"`
                    // verbatim and presents its clean out-of-context-room
                    // handling instead of a raw failure dump.
                    "context_length_exceeded".to_string()
                } else {
                    "server_error".to_string()
                },
                message: error.message,
            },
            None => ErrorObject {
                code: "server_error".to_string(),
                message: "The model backend returned an error before the response completed."
                    .to_string(),
            },
        });

        let failed = ResponseStreamEvent::ResponseFailed(ResponseFailedEvent {
            sequence_number: self.next_seq(),
            response,
        });
        events.push(self.make_sse_event(&failed));

        events
    }
}

/// Backend error detail captured from an `event: error` annotation mid-stream.
#[derive(Debug, Clone)]
pub struct BackendError {
    pub message: String,
    pub http_status: u16,
}

impl BackendError {
    fn is_truncation(&self) -> bool {
        self.message
            .to_ascii_lowercase()
            .contains("cutoff by max_tokens")
    }

    /// Prompt-overflow rejections: the Baseten chat processor's
    /// "Input length N exceeds the maximum allowed input length of M tokens."
    /// and the OpenAI-style "maximum context length" phrasing.
    fn is_context_overflow(&self) -> bool {
        let message = self.message.to_ascii_lowercase();
        message.contains("exceeds the maximum allowed input length")
            || message.contains("maximum context length")
    }
}

impl ResponseStreamConverter {
    /// Serialize a stream event, patching any embedded `response` object to
    /// satisfy the OpenResponses schema. Takes `&self` so spec-required
    /// sampling params can be sourced from the originating request via
    /// `self.params` rather than hardcoded at each emit site.
    fn make_sse_event(&self, event: &ResponseStreamEvent) -> Result<Event, anyhow::Error> {
        let event_type = get_event_type(event);
        let mut value = serde_json::to_value(event)?;
        if let serde_json::Value::Object(ref mut obj) = value
            && let Some(serde_json::Value::Object(inner)) = obj.get_mut("response")
        {
            super::patch_response_for_spec(
                inner,
                self.params.presence_penalty.unwrap_or(0.0),
                self.params.frequency_penalty.unwrap_or(0.0),
                self.params.store.unwrap_or(false),
            );
        }
        let data = serde_json::to_string(&value)?;
        Ok(Event::default().event(event_type).data(data))
    }
}

fn get_event_type(event: &ResponseStreamEvent) -> &'static str {
    match event {
        ResponseStreamEvent::ResponseCreated(_) => "response.created",
        ResponseStreamEvent::ResponseInProgress(_) => "response.in_progress",
        ResponseStreamEvent::ResponseCompleted(_) => "response.completed",
        ResponseStreamEvent::ResponseFailed(_) => "response.failed",
        ResponseStreamEvent::ResponseIncomplete(_) => "response.incomplete",
        ResponseStreamEvent::ResponseQueued(_) => "response.queued",
        ResponseStreamEvent::ResponseOutputItemAdded(_) => "response.output_item.added",
        ResponseStreamEvent::ResponseOutputItemDone(_) => "response.output_item.done",
        ResponseStreamEvent::ResponseContentPartAdded(_) => "response.content_part.added",
        ResponseStreamEvent::ResponseContentPartDone(_) => "response.content_part.done",
        ResponseStreamEvent::ResponseOutputTextDelta(_) => "response.output_text.delta",
        ResponseStreamEvent::ResponseOutputTextDone(_) => "response.output_text.done",
        ResponseStreamEvent::ResponseRefusalDelta(_) => "response.refusal.delta",
        ResponseStreamEvent::ResponseRefusalDone(_) => "response.refusal.done",
        ResponseStreamEvent::ResponseFunctionCallArgumentsDelta(_) => {
            "response.function_call_arguments.delta"
        }
        ResponseStreamEvent::ResponseFunctionCallArgumentsDone(_) => {
            "response.function_call_arguments.done"
        }
        ResponseStreamEvent::ResponseFileSearchCallInProgress(_) => {
            "response.file_search_call.in_progress"
        }
        ResponseStreamEvent::ResponseFileSearchCallSearching(_) => {
            "response.file_search_call.searching"
        }
        ResponseStreamEvent::ResponseFileSearchCallCompleted(_) => {
            "response.file_search_call.completed"
        }
        ResponseStreamEvent::ResponseWebSearchCallInProgress(_) => {
            "response.web_search_call.in_progress"
        }
        ResponseStreamEvent::ResponseWebSearchCallSearching(_) => {
            "response.web_search_call.searching"
        }
        ResponseStreamEvent::ResponseWebSearchCallCompleted(_) => {
            "response.web_search_call.completed"
        }
        ResponseStreamEvent::ResponseReasoningSummaryPartAdded(_) => {
            "response.reasoning_summary_part.added"
        }
        ResponseStreamEvent::ResponseReasoningSummaryPartDone(_) => {
            "response.reasoning_summary_part.done"
        }
        ResponseStreamEvent::ResponseReasoningSummaryTextDelta(_) => {
            "response.reasoning_summary_text.delta"
        }
        ResponseStreamEvent::ResponseReasoningSummaryTextDone(_) => {
            "response.reasoning_summary_text.done"
        }
        ResponseStreamEvent::ResponseReasoningTextDelta(_) => "response.reasoning_text.delta",
        ResponseStreamEvent::ResponseReasoningTextDone(_) => "response.reasoning_text.done",
        ResponseStreamEvent::ResponseImageGenerationCallCompleted(_) => {
            "response.image_generation_call.completed"
        }
        ResponseStreamEvent::ResponseImageGenerationCallGenerating(_) => {
            "response.image_generation_call.generating"
        }
        ResponseStreamEvent::ResponseImageGenerationCallInProgress(_) => {
            "response.image_generation_call.in_progress"
        }
        ResponseStreamEvent::ResponseImageGenerationCallPartialImage(_) => {
            "response.image_generation_call.partial_image"
        }
        ResponseStreamEvent::ResponseMCPCallArgumentsDelta(_) => {
            "response.mcp_call_arguments.delta"
        }
        ResponseStreamEvent::ResponseMCPCallArgumentsDone(_) => "response.mcp_call_arguments.done",
        ResponseStreamEvent::ResponseMCPCallCompleted(_) => "response.mcp_call.completed",
        ResponseStreamEvent::ResponseMCPCallFailed(_) => "response.mcp_call.failed",
        ResponseStreamEvent::ResponseMCPCallInProgress(_) => "response.mcp_call.in_progress",
        ResponseStreamEvent::ResponseMCPListToolsCompleted(_) => {
            "response.mcp_list_tools.completed"
        }
        ResponseStreamEvent::ResponseMCPListToolsFailed(_) => "response.mcp_list_tools.failed",
        ResponseStreamEvent::ResponseMCPListToolsInProgress(_) => {
            "response.mcp_list_tools.in_progress"
        }
        ResponseStreamEvent::ResponseCodeInterpreterCallInProgress(_) => {
            "response.code_interpreter_call.in_progress"
        }
        ResponseStreamEvent::ResponseCodeInterpreterCallInterpreting(_) => {
            "response.code_interpreter_call.interpreting"
        }
        ResponseStreamEvent::ResponseCodeInterpreterCallCompleted(_) => {
            "response.code_interpreter_call.completed"
        }
        ResponseStreamEvent::ResponseCodeInterpreterCallCodeDelta(_) => {
            "response.code_interpreter_call_code.delta"
        }
        ResponseStreamEvent::ResponseCodeInterpreterCallCodeDone(_) => {
            "response.code_interpreter_call_code.done"
        }
        ResponseStreamEvent::ResponseOutputTextAnnotationAdded(_) => {
            "response.output_text.annotation.added"
        }
        ResponseStreamEvent::ResponseCustomToolCallInputDelta(_) => {
            "response.custom_tool_call_input.delta"
        }
        ResponseStreamEvent::ResponseCustomToolCallInputDone(_) => {
            "response.custom_tool_call_input.done"
        }
        ResponseStreamEvent::ResponseError(_) => "error",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::unified::ResponsesContext;
    use dynamo_protocols::types::{
        ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionMessageToolCallChunk,
        ChatCompletionStreamResponseDelta, FunctionCallStream, FunctionType,
    };

    fn default_params() -> ResponseParams {
        ResponseParams::default()
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

    fn finish_chunk(reason: FinishReason) -> NvCreateChatCompletionStreamResponse {
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
                        reasoning_content: None,
                    },
                    finish_reason: Some(reason),
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

    fn with_finish_reason(
        mut chunk: NvCreateChatCompletionStreamResponse,
        reason: FinishReason,
    ) -> NvCreateChatCompletionStreamResponse {
        chunk.inner.choices[0].finish_reason = Some(reason);
        chunk
    }

    /// Extract the SSE event type from a Result<Event, _>.
    fn event_type(event: &Result<Event, anyhow::Error>) -> String {
        let debug = format!("{:?}", event.as_ref().unwrap());
        // Event debug format: Event { ... event: "response.xxx" ... }
        // Parse the event type from the serialized SSE data
        if let Some(start) = debug.find("event: ") {
            let rest = &debug[start + 7..];
            if let Some(end) = rest.find("\\n") {
                return rest[..end].to_string();
            }
        }
        "unknown".to_string()
    }

    fn event_types(events: &[Result<Event, anyhow::Error>]) -> Vec<String> {
        events.iter().map(event_type).collect()
    }

    /// Extract the SSE `data:` JSON payload from a Result<Event, _>.
    fn event_data(event: &Result<Event, anyhow::Error>) -> serde_json::Value {
        let debug = format!("{:?}", event.as_ref().unwrap());
        let start = debug.find("data: ").expect("event has data") + 6;
        let rest = &debug[start..];
        let end = rest.find("\\n").unwrap_or(rest.len());
        // The debug output is a Rust string literal, so unescape it via serde.
        let raw: String =
            serde_json::from_str(&format!("\"{}\"", &rest[..end])).expect("unescape event data");
        serde_json::from_str(&raw).expect("event data is JSON")
    }

    /// Collect (event_type, data) pairs.
    fn typed_events(events: &[Result<Event, anyhow::Error>]) -> Vec<(String, serde_json::Value)> {
        events
            .iter()
            .map(|e| (event_type(e), event_data(e)))
            .collect()
    }

    #[test]
    fn b10_length_finish_reason_marks_open_tool_call_incomplete() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());
        let _ = conv.process_chunk(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("get_weather"),
            Some("{\"city\":\"SF"),
        ));
        // The merged close-on-finish_reason path (GLM fragmented-args fix)
        // closes the open tool call inline. On a `length` finish this matches
        // OpenAI verbatim: it emits `function_call_arguments.done` with the
        // partial args + `output_item.done` with an `incomplete` item status,
        // then surfaces truncation via the terminal `response.incomplete`.
        let finish_events = conv.process_chunk(&finish_chunk(FinishReason::Length));
        let finish_types = event_types(&finish_events);
        assert!(
            finish_types.contains(&"response.function_call_arguments.done".to_string()),
            "args.done emitted on truncation: {finish_types:?}"
        );
        assert!(
            finish_types.contains(&"response.output_item.done".to_string()),
            "output_item.done emitted on truncation: {finish_types:?}"
        );

        let end_events = conv.emit_end_events();
        assert_eq!(
            event_types(&end_events).last().map(String::as_str),
            Some("response.incomplete")
        );

        let response = conv.make_response(conv.terminal_status(), conv.completed_output());
        assert_eq!(response.status, Status::Incomplete);
        assert_eq!(
            response
                .incomplete_details
                .as_ref()
                .map(|details| details.reason.as_str()),
            Some("max_output_tokens")
        );
        let OutputItem::FunctionCall(call) = &response.output[0] else {
            panic!("expected function call output");
        };
        // Partial args are preserved and the item is marked incomplete — the
        // exact shape OpenAI emits for a tool call truncated mid-arguments.
        assert_eq!(call.status, Some(OutputStatus::Incomplete));
    }

    #[test]
    fn b10_length_finish_reason_emits_incomplete_terminal_response() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());
        let _ = conv.process_chunk(&text_chunk("partial"));
        let _ = conv.process_chunk(&finish_chunk(FinishReason::Length));

        let end_events = conv.emit_end_events();
        assert_eq!(
            event_types(&end_events).last().map(String::as_str),
            Some("response.incomplete")
        );

        let response = conv.make_response(conv.terminal_status(), conv.completed_output());
        assert_eq!(response.status, Status::Incomplete);
        assert_eq!(
            response
                .incomplete_details
                .as_ref()
                .map(|details| details.reason.as_str()),
            Some("max_output_tokens")
        );
        assert_eq!(response.completed_at, None);
        let OutputItem::Message(message) = &response.output[0] else {
            panic!("expected message output");
        };
        assert_eq!(message.status, OutputStatus::Incomplete);
    }

    #[test]
    fn b10_length_finish_reason_marks_reasoning_item_incomplete() {
        use dynamo_protocols::types::responses::{Reasoning, ReasoningSummary};

        let params = ResponseParams {
            reasoning: Some(Reasoning {
                effort: None,
                summary: Some(ReasoningSummary::Auto),
            }),
            ..default_params()
        };
        let mut conv = ResponseStreamConverter::new("test-model".into(), params);
        let _ = conv.process_chunk(&reasoning_chunk("partial"));
        let _ = conv.process_chunk(&finish_chunk(FinishReason::Length));
        let _ = conv.emit_end_events();

        let response = conv.make_response(conv.terminal_status(), conv.completed_output());
        let OutputItem::Reasoning(reasoning) = &response.output[0] else {
            panic!("expected reasoning output");
        };
        assert_eq!(reasoning.status, Some(OutputStatus::Incomplete));
        // Truncated reasoning keeps its partial summary with a trailing ellipsis
        // so clients can see the stream was cut off mid-thought.
        let SummaryPart::SummaryText(summary) = &reasoning.summary[0];
        assert_eq!(summary.text, "partial...");
    }

    #[test]
    fn b10_completed_reasoning_stays_complete_when_text_is_truncated() {
        use dynamo_protocols::types::responses::{Reasoning, ReasoningSummary};

        let params = ResponseParams {
            reasoning: Some(Reasoning {
                effort: None,
                summary: Some(ReasoningSummary::Auto),
            }),
            ..default_params()
        };
        let mut conv = ResponseStreamConverter::new("test-model".into(), params);
        let _ = conv.process_chunk(&reasoning_chunk("complete reasoning"));
        let _ = conv.process_chunk(&text_chunk("partial answer"));
        let _ = conv.process_chunk(&finish_chunk(FinishReason::Length));
        let _ = conv.emit_end_events();

        let response = conv.make_response(conv.terminal_status(), conv.completed_output());
        assert_eq!(response.status, Status::Incomplete);
        let OutputItem::Reasoning(reasoning) = &response.output[0] else {
            panic!("expected reasoning output");
        };
        assert_eq!(reasoning.status, Some(OutputStatus::Completed));
        // Reasoning concluded normally (answer text followed) — no ellipsis is
        // appended; only the message that followed is truncated.
        let SummaryPart::SummaryText(summary) = &reasoning.summary[0];
        assert_eq!(summary.text, "complete reasoning");
        let OutputItem::Message(message) = &response.output[1] else {
            panic!("expected message output");
        };
        assert_eq!(message.status, OutputStatus::Incomplete);
    }

    #[test]
    fn b10_same_chunk_text_and_length_complete_reasoning_only() {
        use dynamo_protocols::types::responses::{Reasoning, ReasoningSummary};

        let params = ResponseParams {
            reasoning: Some(Reasoning {
                effort: None,
                summary: Some(ReasoningSummary::Auto),
            }),
            ..default_params()
        };
        let mut conv = ResponseStreamConverter::new("test-model".into(), params);
        let _ = conv.process_chunk(&reasoning_chunk("complete reasoning"));

        let events = conv.process_chunk(&with_finish_reason(
            text_chunk("partial answer"),
            FinishReason::Length,
        ));

        assert_eq!(
            event_types(&events),
            vec![
                "response.reasoning_summary_text.done".to_string(),
                "response.reasoning_summary_part.done".to_string(),
                "response.output_item.done".to_string(),
                "response.output_item.added".to_string(),
                "response.content_part.added".to_string(),
                "response.output_text.delta".to_string(),
            ]
        );
        assert_eq!(conv.reasoning_output_status, Some(OutputStatus::Completed));

        let _ = conv.emit_end_events();
        let response = conv.make_response(conv.terminal_status(), conv.completed_output());
        assert_eq!(response.status, Status::Incomplete);
        let OutputItem::Reasoning(reasoning) = &response.output[0] else {
            panic!("expected reasoning output");
        };
        assert_eq!(reasoning.status, Some(OutputStatus::Completed));
        let OutputItem::Message(message) = &response.output[1] else {
            panic!("expected message output");
        };
        assert_eq!(message.status, OutputStatus::Incomplete);
    }

    #[test]
    fn b10_same_chunk_tool_call_and_length_complete_reasoning_only() {
        use dynamo_protocols::types::responses::{Reasoning, ReasoningSummary};

        let params = ResponseParams {
            reasoning: Some(Reasoning {
                effort: None,
                summary: Some(ReasoningSummary::Auto),
            }),
            ..default_params()
        };
        let mut conv = ResponseStreamConverter::new("test-model".into(), params);
        let _ = conv.process_chunk(&reasoning_chunk("complete reasoning"));

        let _ = conv.process_chunk(&with_finish_reason(
            tool_call_chunk(
                0,
                Some("call-1"),
                Some("get_weather"),
                Some("{\"city\":\"SF"),
            ),
            FinishReason::Length,
        ));

        assert_eq!(conv.reasoning_output_status, Some(OutputStatus::Completed));
        let _ = conv.emit_end_events();
        let response = conv.make_response(conv.terminal_status(), conv.completed_output());
        assert_eq!(response.status, Status::Incomplete);
        let OutputItem::Reasoning(reasoning) = &response.output[0] else {
            panic!("expected reasoning output");
        };
        assert_eq!(reasoning.status, Some(OutputStatus::Completed));
        let OutputItem::FunctionCall(function_call) = &response.output[1] else {
            panic!("expected function call output");
        };
        assert_eq!(function_call.status, Some(OutputStatus::Incomplete));
    }

    #[test]
    fn b10_requested_reasoning_summary_streams_complete_event_sequence() {
        use dynamo_protocols::types::responses::{Reasoning, ReasoningSummary};

        let params = ResponseParams {
            reasoning: Some(Reasoning {
                effort: None,
                summary: Some(ReasoningSummary::Auto),
            }),
            ..default_params()
        };
        let mut conv = ResponseStreamConverter::new("test-model".into(), params);

        let reasoning_events = conv.process_chunk(&reasoning_chunk("thinking"));
        assert_eq!(
            event_types(&reasoning_events),
            vec![
                "response.output_item.added".to_string(),
                "response.reasoning_summary_part.added".to_string(),
                "response.reasoning_summary_text.delta".to_string(),
            ]
        );

        let text_events = conv.process_chunk(&text_chunk("answer"));
        assert_eq!(
            event_types(&text_events),
            vec![
                "response.reasoning_summary_text.done".to_string(),
                "response.reasoning_summary_part.done".to_string(),
                "response.output_item.done".to_string(),
                "response.output_item.added".to_string(),
                "response.content_part.added".to_string(),
                "response.output_text.delta".to_string(),
            ]
        );

        let response = conv.make_response(Status::Completed, conv.completed_output());
        assert_eq!(response.output.len(), 2);
        let OutputItem::Reasoning(reasoning) = &response.output[0] else {
            panic!("expected reasoning output before message");
        };
        assert_eq!(
            reasoning.summary,
            vec![SummaryPart::SummaryText(SummaryTextContent {
                text: "thinking".to_string(),
            })]
        );
        assert!(matches!(response.output[1], OutputItem::Message(_)));
    }

    #[test]
    fn b10_reasoning_without_requested_summary_emits_no_events() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());

        let events = conv.process_chunk(&reasoning_chunk("private reasoning"));

        assert!(events.is_empty());
        assert!(conv.completed_output().is_empty());
    }

    #[test]
    fn b10_reasoning_summary_ignores_updates_after_completion() {
        use dynamo_protocols::types::responses::{Reasoning, ReasoningSummary};

        let params = ResponseParams {
            reasoning: Some(Reasoning {
                effort: None,
                summary: Some(ReasoningSummary::Auto),
            }),
            ..default_params()
        };
        let mut conv = ResponseStreamConverter::new("test-model".into(), params);

        let _ = conv.process_chunk(&reasoning_chunk("summary"));
        let _ = conv.process_chunk(&text_chunk("answer"));
        let late_events = conv.process_chunk(&reasoning_chunk(" must not be appended"));

        assert!(late_events.is_empty());
        let output = conv.completed_output();
        let OutputItem::Reasoning(reasoning) = &output[0] else {
            panic!("expected reasoning output");
        };
        assert_eq!(
            reasoning.summary,
            vec![SummaryPart::SummaryText(SummaryTextContent {
                text: "summary".to_string(),
            })]
        );
    }

    #[test]
    fn b10_reasoning_summary_finishes_before_tool_call() {
        use dynamo_protocols::types::responses::{Reasoning, ReasoningSummary};

        let params = ResponseParams {
            reasoning: Some(Reasoning {
                effort: None,
                summary: Some(ReasoningSummary::Auto),
            }),
            ..default_params()
        };
        let mut conv = ResponseStreamConverter::new("test-model".into(), params);

        let _ = conv.process_chunk(&reasoning_chunk("summary"));
        let tool_events = conv.process_chunk(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("get_time"),
            Some("{}"),
        ));
        assert_eq!(
            &event_types(&tool_events)[..3],
            [
                "response.reasoning_summary_text.done".to_string(),
                "response.reasoning_summary_part.done".to_string(),
                "response.output_item.done".to_string(),
            ]
        );

        let late_events = conv.process_chunk(&reasoning_chunk(" must not be appended"));
        assert!(late_events.is_empty());
    }

    #[test]
    fn b10_reasoning_summary_does_not_start_after_visible_output() {
        use dynamo_protocols::types::responses::{Reasoning, ReasoningSummary};

        let params = ResponseParams {
            reasoning: Some(Reasoning {
                effort: None,
                summary: Some(ReasoningSummary::Auto),
            }),
            ..default_params()
        };
        let mut conv = ResponseStreamConverter::new("test-model".into(), params);

        let _ = conv.process_chunk(&text_chunk("answer"));
        let late_events = conv.process_chunk(&reasoning_chunk("out of order"));

        assert!(late_events.is_empty());
        assert!(
            conv.completed_output()
                .iter()
                .all(|item| !matches!(item, OutputItem::Reasoning(_)))
        );
    }

    /// Tool call done events fire on finish_reason with the full arguments,
    /// and are not duplicated by the end events.
    #[test]
    fn b10_tool_call_done_on_finish_reason() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());
        let _ = conv.emit_start_events(); // consume start events

        let events = conv.process_chunk(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("get_weather"),
            Some("{\"city\":\"SF\"}"),
        ));

        let types = event_types(&events);
        assert!(
            types.contains(&"response.output_item.added".to_string()),
            "should emit output_item.added: {types:?}"
        );
        assert!(
            types.contains(&"response.function_call_arguments.delta".to_string()),
            "should emit args delta: {types:?}"
        );
        assert!(
            !types.contains(&"response.function_call_arguments.done".to_string()),
            "done must wait for finish_reason: {types:?}"
        );

        let finish_events = conv.process_chunk(&finish_chunk(
            dynamo_protocols::types::FinishReason::ToolCalls,
        ));
        let finish_typed = typed_events(&finish_events);
        let args_done = finish_typed
            .iter()
            .find(|(t, _)| t == "response.function_call_arguments.done")
            .expect("args done on finish_reason");
        assert_eq!(args_done.1["arguments"], "{\"city\":\"SF\"}");
        assert!(
            finish_typed
                .iter()
                .any(|(t, _)| t == "response.output_item.done"),
            "output_item.done on finish_reason"
        );

        // End events should NOT duplicate the done events
        let end_types = event_types(&conv.emit_end_events());
        assert!(
            !end_types.contains(&"response.function_call_arguments.done".to_string()),
            "done should not be duplicated in end events: {end_types:?}"
        );
        assert!(
            !end_types.contains(&"response.output_item.done".to_string()),
            "output_item.done for the tool should not appear in end events: {end_types:?}"
        );

        let response = conv.make_response(conv.terminal_status(), conv.completed_output());
        assert_eq!(response.status, Status::Completed);
        let OutputItem::FunctionCall(call) = &response.output[0] else {
            panic!("expected function call output");
        };
        assert_eq!(call.status, Some(OutputStatus::Completed));
    }

    /// Regression test for GLM-style argument fragmentation: id + name arrive on
    /// the first chunk together with only the first argument fragment, and the
    /// rest of the arguments stream in later chunks. Done events must carry the
    /// fully concatenated arguments, not just the first fragment.
    #[test]
    fn b10_fragmented_args_done_carries_full_arguments() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());
        let _ = conv.emit_start_events();

        let mut all_events = Vec::new();
        all_events.extend(conv.process_chunk(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("shell"),
            Some("{"),
        )));
        all_events.extend(conv.process_chunk(&tool_call_chunk(
            0,
            None,
            None,
            Some("\"command\":"),
        )));
        all_events.extend(conv.process_chunk(&tool_call_chunk(
            0,
            None,
            None,
            Some("[\"ls\",\"-la\"]}"),
        )));

        // No done events until finish_reason
        let types = event_types(&all_events);
        assert!(
            !types.contains(&"response.function_call_arguments.done".to_string()),
            "no done while args still streaming: {types:?}"
        );
        assert_eq!(
            types
                .iter()
                .filter(|t| *t == "response.function_call_arguments.delta")
                .count(),
            3,
            "each fragment is a delta: {types:?}"
        );

        let finish_events = conv.process_chunk(&finish_chunk(
            dynamo_protocols::types::FinishReason::ToolCalls,
        ));
        let finish_typed = typed_events(&finish_events);

        let full_args = "{\"command\":[\"ls\",\"-la\"]}";
        let args_done = finish_typed
            .iter()
            .find(|(t, _)| t == "response.function_call_arguments.done")
            .expect("args done present");
        assert_eq!(args_done.1["arguments"], full_args);

        let item_done = finish_typed
            .iter()
            .find(|(t, _)| t == "response.output_item.done")
            .expect("item done present");
        assert_eq!(item_done.1["item"]["arguments"], full_args);
        assert_eq!(item_done.1["item"]["call_id"], "call-1");
        assert_eq!(item_done.1["item"]["name"], "shell");

        // The final response.completed output must also carry the full args.
        let end_events = conv.emit_end_events();
        let end_typed = typed_events(&end_events);
        let completed = end_typed
            .iter()
            .find(|(t, _)| t == "response.completed")
            .expect("completed present");
        assert_eq!(completed.1["response"]["output"][0]["arguments"], full_args);
    }

    /// Streams that end without a finish_reason chunk still get exactly one
    /// pair of done events, with full arguments, from the end events.
    #[test]
    fn b10_stream_end_without_finish_reason_closes_tool_call() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());
        let _ = conv.emit_start_events();

        let _ = conv.process_chunk(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("shell"),
            Some("{"),
        ));
        let _ = conv.process_chunk(&tool_call_chunk(0, None, None, Some("\"a\":1}")));

        let end_typed = typed_events(&conv.emit_end_events());
        let done_count = end_typed
            .iter()
            .filter(|(t, _)| t == "response.function_call_arguments.done")
            .count();
        assert_eq!(done_count, 1, "exactly one args done: {end_typed:?}");
        let args_done = end_typed
            .iter()
            .find(|(t, _)| t == "response.function_call_arguments.done")
            .unwrap();
        assert_eq!(args_done.1["arguments"], "{\"a\":1}");
        assert!(
            end_typed.iter().any(|(t, _)| t == "response.completed"),
            "completed present: {end_typed:?}"
        );
    }

    /// Multiple tool calls are all closed on finish_reason.
    #[test]
    fn b10_multiple_tool_calls_closed_on_finish_reason() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());
        let _ = conv.emit_start_events();

        let _ = conv.process_chunk(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("get_weather"),
            Some("{\"city\":\"SF\"}"),
        ));
        let _ = conv.process_chunk(&tool_call_chunk(
            1,
            Some("call-2"),
            Some("get_time"),
            Some("{\"tz\":\"PST\"}"),
        ));

        let finish_types = event_types(&conv.process_chunk(&finish_chunk(
            dynamo_protocols::types::FinishReason::ToolCalls,
        )));
        assert_eq!(
            finish_types
                .iter()
                .filter(|t| *t == "response.function_call_arguments.done")
                .count(),
            2,
            "both tool calls closed on finish_reason: {finish_types:?}"
        );

        // End events should have no function call done events
        let end_types = event_types(&conv.emit_end_events());
        let fc_done_count = end_types
            .iter()
            .filter(|t| *t == "response.function_call_arguments.done")
            .count();
        assert_eq!(
            fc_done_count, 0,
            "no function_call_arguments.done in end events: {end_types:?}"
        );
    }

    /// Text-only response: no tool-related events at all.
    #[test]
    fn test_text_only_response_no_tool_events() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());
        let _ = conv.emit_start_events();

        let events = conv.process_chunk(&text_chunk("Hello world"));
        let types = event_types(&events);
        assert!(
            !types.contains(&"response.function_call_arguments.done".to_string()),
            "no tool events in text-only: {types:?}"
        );

        let end_events = conv.emit_end_events();
        let end_types = event_types(&end_events);
        assert!(
            end_types.contains(&"response.output_text.done".to_string()),
            "text done in end events: {end_types:?}"
        );
        assert!(
            end_types.contains(&"response.completed".to_string()),
            "completed in end events: {end_types:?}"
        );
    }

    /// Text followed by tool call: both handled correctly.
    #[test]
    fn test_text_then_tool_call() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());
        let _ = conv.emit_start_events();

        let text_events = conv.process_chunk(&text_chunk("Let me check that."));
        let text_types = event_types(&text_events);
        assert!(
            text_types.contains(&"response.output_item.added".to_string()),
            "text message started: {text_types:?}"
        );

        let tool_events = conv.process_chunk(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("search"),
            Some("{\"q\":\"rust\"}"),
        ));
        let tool_types = event_types(&tool_events);
        assert!(
            tool_types.contains(&"response.output_item.added".to_string()),
            "tool call item added after text: {tool_types:?}"
        );

        let finish_types = event_types(&conv.process_chunk(&finish_chunk(
            dynamo_protocols::types::FinishReason::ToolCalls,
        )));
        assert!(
            finish_types.contains(&"response.function_call_arguments.done".to_string()),
            "tool call done on finish_reason after text: {finish_types:?}"
        );
        assert!(
            finish_types.contains(&"response.output_item.done".to_string()),
            "output_item.done on finish_reason after text: {finish_types:?}"
        );
    }

    /// Verify that `with_context` populates `previous_response_id`
    /// in the generated Response objects.
    #[test]
    fn test_with_context_enriches_response() {
        let ctx = ResponsesContext {
            previous_response_id: Some("resp_prev_123".to_string()),
            store: true,
            ..Default::default()
        };
        let params = ResponseParams::default();
        let mut conv = ResponseStreamConverter::with_context("test-model".into(), params, ctx);

        // Process one text chunk so there's output
        let _ = conv.emit_start_events();
        let _ = conv.process_chunk(&text_chunk("Hello"));
        let _end_events = conv.emit_end_events();

        let response = conv.make_response(Status::Completed, vec![]);
        assert_eq!(
            response.previous_response_id.as_deref(),
            Some("resp_prev_123")
        );
    }

    /// Without context, previous_response_id is None.
    #[test]
    fn test_without_context_defaults() {
        let params = ResponseParams::default();
        let conv = ResponseStreamConverter::new("test-model".into(), params);

        let response = conv.make_response(Status::Completed, vec![]);
        assert_eq!(response.previous_response_id, None);
    }

    #[test]
    fn test_stream_response_echoes_parallel_tool_calls() {
        let params = ResponseParams {
            parallel_tool_calls: Some(false),
            ..Default::default()
        };
        let conv = ResponseStreamConverter::new("test-model".into(), params);

        let response = conv.make_response(Status::Completed, vec![]);
        assert_eq!(response.parallel_tool_calls, Some(false));
    }

    /// A legacy chat processor raises "Tool calls cutoff by max_tokens."
    /// instead of finishing the turn with `finish_reason=length`, which
    /// arrives here as a backend error with no length finish ever seen.
    /// It must be presented as spec-correct truncation — done events with
    /// the partial args, item status `incomplete`, terminal
    /// `response.incomplete` with reason `max_output_tokens` — and never
    /// as `response.failed`.
    #[test]
    fn b10_backend_cutoff_error_presented_as_incomplete() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());
        let _ = conv.emit_start_events();
        let _ = conv.process_chunk(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("run_command"),
            Some("{\"cmd\":\"ls -"),
        ));

        let events = conv.emit_error_events(Some(BackendError {
            message: "Tool calls cutoff by max_tokens.".to_string(),
            http_status: 400,
        }));
        let types = event_types(&events);

        assert!(
            !types.contains(&"response.failed".to_string()),
            "cutoff must not surface as failure: {types:?}"
        );
        assert!(
            types.contains(&"response.function_call_arguments.done".to_string()),
            "args.done with partial args: {types:?}"
        );
        assert!(
            types.contains(&"response.output_item.done".to_string()),
            "output_item.done for the truncated call: {types:?}"
        );
        assert_eq!(
            types.last().map(String::as_str),
            Some("response.incomplete")
        );

        let (_, terminal) = typed_events(&events).pop().unwrap();
        let response = &terminal["response"];
        assert_eq!(response["status"], "incomplete");
        assert_eq!(
            response["incomplete_details"]["reason"],
            "max_output_tokens"
        );
        let output = response["output"].as_array().unwrap();
        assert!(!output.is_empty(), "partial output preserved");
        assert_eq!(output[0]["type"], "function_call");
        assert_eq!(output[0]["status"], "incomplete");
        assert_eq!(output[0]["arguments"], "{\"cmd\":\"ls -");
    }

    /// A genuine backend failure keeps `response.failed`, but the terminal
    /// response must carry the error detail and any partial output instead
    /// of `output: []` with `error: null`.
    #[test]
    fn b10_backend_error_carries_detail_and_partial_output() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());
        let _ = conv.emit_start_events();
        let _ = conv.process_chunk(&text_chunk("partial answer"));

        let events = conv.emit_error_events(Some(BackendError {
            message: "engine worker crashed".to_string(),
            http_status: 500,
        }));
        let typed = typed_events(&events);
        assert_eq!(typed.len(), 1);
        let (event_type, data) = &typed[0];
        assert_eq!(event_type, "response.failed");

        let response = &data["response"];
        assert_eq!(response["status"], "failed");
        assert_eq!(response["error"]["code"], "server_error");
        assert_eq!(response["error"]["message"], "engine worker crashed");
        let output = response["output"].as_array().unwrap();
        assert_eq!(output[0]["status"], "incomplete");
        assert_eq!(output[0]["content"][0]["text"], "partial answer");
    }

    /// A tool declared inside a `{"type": "namespace"}` group must have its
    /// namespace echoed on every emitted function_call item — codex
    /// dispatches on the exact (name, namespace) pair with no fallback, so
    /// a stripped namespace makes the call undispatchable client-side.
    #[test]
    fn b10_namespaced_tool_call_echoes_namespace() {
        use dynamo_protocols::types::responses::{
            FunctionToolParam, NamespaceToolParam, NamespaceToolParamTool, Tool,
        };

        let params = ResponseParams {
            tools: Some(vec![Tool::Namespace(NamespaceToolParam {
                name: "mcp__codex_apps__gmail".into(),
                description: "Gmail tools".into(),
                tools: vec![NamespaceToolParamTool::Function(FunctionToolParam {
                    name: "get_recent_emails".into(),
                    ..Default::default()
                })],
            })]),
            ..Default::default()
        };
        let mut conv = ResponseStreamConverter::new("test-model".into(), params);
        let _ = conv.emit_start_events();
        // The worker was given the mangled flat name (namespaces exist so
        // member names can overlap between groups), so that is what the
        // model emits; the wire items carry the original (name, namespace).
        let events = conv.process_chunk(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("mcp__codex_apps__gmail__get_recent_emails"),
            Some("{}"),
        ));
        let added = typed_events(&events)
            .into_iter()
            .find(|(t, _)| t == "response.output_item.added")
            .expect("item added");
        assert_eq!(added.1["item"]["namespace"], "mcp__codex_apps__gmail");
        assert_eq!(added.1["item"]["name"], "get_recent_emails");

        let finish_events = conv.process_chunk(&finish_chunk(FinishReason::ToolCalls));
        let typed = typed_events(&finish_events);
        let done = typed
            .iter()
            .find(|(t, _)| t == "response.output_item.done")
            .expect("item done");
        assert_eq!(done.1["item"]["namespace"], "mcp__codex_apps__gmail");
        assert_eq!(done.1["item"]["name"], "get_recent_emails");
        let args_done = typed
            .iter()
            .find(|(t, _)| t == "response.function_call_arguments.done")
            .expect("args done");
        assert_eq!(args_done.1["name"], "get_recent_emails");

        let response = conv.make_response(Status::Completed, conv.completed_output());
        let OutputItem::FunctionCall(call) = &response.output[0] else {
            panic!("expected function call output");
        };
        assert_eq!(call.namespace.as_deref(), Some("mcp__codex_apps__gmail"));
        assert_eq!(call.name, "get_recent_emails");
    }

    /// Plain function tools (no namespace group) keep namespace absent.
    #[test]
    fn b10_plain_tool_call_has_no_namespace() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());
        let _ = conv.emit_start_events();
        let _ = conv.process_chunk(&tool_call_chunk(
            0,
            Some("call-1"),
            Some("get_weather"),
            Some("{}"),
        ));
        let finish_events = conv.process_chunk(&finish_chunk(FinishReason::ToolCalls));
        let done = typed_events(&finish_events)
            .into_iter()
            .find(|(t, _)| t == "response.output_item.done")
            .expect("item done");
        assert!(done.1["item"]["namespace"].is_null());
    }

    /// Prompt-overflow backend errors must carry the exact code string
    /// OpenAI clients classify on. Codex matches
    /// `response.error.code == "context_length_exceeded"` verbatim (its only
    /// context-overflow detector) and presents its out-of-context-room
    /// handling; anything else surfaces as a raw non-retryable failure.
    #[test]
    fn b10_prompt_overflow_error_carries_openai_code() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());
        let _ = conv.emit_start_events();

        let events = conv.emit_error_events(Some(BackendError {
            message:
                "Input length 345123 exceeds the maximum allowed input length of 131040 tokens."
                    .to_string(),
            http_status: 400,
        }));
        let typed = typed_events(&events);
        assert_eq!(typed.len(), 1);
        let (event_type, data) = &typed[0];
        assert_eq!(event_type, "response.failed");
        assert_eq!(data["response"]["error"]["code"], "context_length_exceeded");
        assert!(
            data["response"]["error"]["message"]
                .as_str()
                .unwrap()
                .contains("exceeds the maximum allowed input length")
        );
    }

    /// Even with no captured detail, `response.failed` must populate `error`
    /// rather than emit the spec-violating `error: null`.
    #[test]
    fn b10_backend_error_without_detail_still_populates_error() {
        let mut conv = ResponseStreamConverter::new("test-model".into(), default_params());
        let _ = conv.emit_start_events();

        let events = conv.emit_error_events(None);
        let typed = typed_events(&events);
        assert_eq!(typed.len(), 1);
        let (event_type, data) = &typed[0];
        assert_eq!(event_type, "response.failed");
        assert_eq!(data["response"]["error"]["code"], "server_error");
        assert!(
            data["response"]["error"]["message"]
                .as_str()
                .is_some_and(|m| !m.is_empty())
        );
    }
}
