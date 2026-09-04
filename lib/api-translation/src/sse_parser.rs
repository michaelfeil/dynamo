//! The single boundary source: model ChatCompletions SSE -> [`SemanticChunk`]. Folds the CC wire
//! decode (one `data:` payload = one `chat.completion.chunk`) and the per-call accumulation +
//! tool-call boundary decisions into one place, so no downstream re-derives boundaries.
//!
//! Ported (copy-adapt onto our types) from basetenlabs/dynamo @ 68dec805 (Apache-2.0):
//!  - boundary logic (`AnthropicStreamConverter::process_chunk`, inline-close-on-JSON-complete):
//!    https://github.com/basetenlabs/dynamo/blob/68dec805e22e66851dc0bf2a8faba7ce0665b00d/lib/llm/src/protocols/anthropic/stream_converter.rs
//!  - per-turn fold (`DeltaAggregator`):
//!    https://github.com/basetenlabs/dynamo/blob/68dec805e22e66851dc0bf2a8faba7ce0665b00d/lib/llm/src/protocols/openai/chat_completions/aggregator.rs
//!
//! SPDX-License-Identifier: Apache-2.0.
//!
//! Text and thinking surface as deltas immediately. A tool call is held until its accumulated args
//! parse as a complete JSON value, then surfaces once as a complete [`ToolCall`] (eager, so the
//! loop can start executing before stream end); an unfinished call is finalized at stream close.
//! The model's tool id is preserved verbatim (cache-coherent replay), never minted.

use std::collections::HashMap;

use dynamo_protocols::types::{
    ChatCompletionMessageContent, ChatCompletionMessageToolCallChunk,
    ChatCompletionResponseContentPart, CompletionUsage, CreateChatCompletionStreamResponse,
    FinishReason,
};
use serde_json::Value;

use crate::SemanticChunk;
use crate::model::{ToolCall, TranslationError};
use crate::util::truncate;

/// A mid-stream `{"error":{...}}` frame, which the backend serializes in-band because its headers
/// are already sent. A 4xx `code` is the caller's to act on (MAPI answers a tool-call cutoff by
/// `max_tokens` with 400), so its status and message carry through; anything else is degradation
/// with no status to mirror.
fn mid_stream_error_frame(error_frame: &Value, message: &str) -> TranslationError {
    let frame_code = error_frame.get("code").and_then(Value::as_u64);
    let client_error_status = frame_code
        .and_then(|code| u16::try_from(code).ok())
        .and_then(|code| http::StatusCode::from_u16(code).ok())
        .filter(http::StatusCode::is_client_error);
    match client_error_status {
        Some(status) => TranslationError::UpstreamResponse {
            status,
            body: format!("model error: {message}"),
        },
        // May embed request content, and there is no status to hand the caller: dropped entirely.
        None => model_unavailable(
            "model_streamed_error",
            format!("model streamed a mid-stream error frame (code {frame_code:?})"),
        ),
    }
}

/// Every stream-decode failure is upstream (model) degradation with no status to mirror.
fn model_unavailable(error_code: &'static str, detail: String) -> TranslationError {
    TranslationError::UpstreamUnavailable { detail, error_code }
}

const DONE: &str = "[DONE]";

type ChunkResult = Result<SemanticChunk, TranslationError>;

/// One parser per model turn: tool indexing is the model's per-response CC index, which restarts
/// each model call.
#[derive(Default)]
pub struct SseParser {
    tools: Vec<ToolCallAccumulator>,
    /// CC `tool_calls[].index` -> position in `tools`.
    pos_by_index: HashMap<u32, usize>,
    usage: Option<CompletionUsage>,
    finish_reason: Option<FinishReason>,
    saw_output: bool,
}

#[derive(Default)]
struct ToolCallAccumulator {
    id: String,
    name: String,
    args: String,
    args_depth: JsonContainerDepth,
    emitted: bool,
}

/// Container depth of the args accumulated so far, string- and escape-aware. The cheap per-chunk
/// completeness gate: without it, `try_complete` would run a full JSON parse over the whole
/// accumulated args on every chunk — O(n²) in argument length.
#[derive(Default)]
struct JsonContainerDepth {
    depth: u32,
    in_string: bool,
    escaped: bool,
    seen_container: bool,
}

impl JsonContainerDepth {
    fn feed(&mut self, appended: &str) {
        // Byte scan is UTF-8-safe: multi-byte code points never contain ASCII bytes.
        for byte in appended.bytes() {
            if self.escaped {
                self.escaped = false;
            } else if self.in_string {
                match byte {
                    b'\\' => self.escaped = true,
                    b'"' => self.in_string = false,
                    _ => {}
                }
            } else {
                match byte {
                    b'"' => self.in_string = true,
                    b'{' | b'[' => {
                        self.depth += 1;
                        self.seen_container = true;
                    }
                    b'}' | b']' => self.depth = self.depth.saturating_sub(1),
                    _ => {}
                }
            }
        }
    }

    fn is_balanced_container(&self) -> bool {
        self.seen_container && self.depth == 0 && !self.in_string
    }
}

impl SseParser {
    /// Decode + fold one SSE `data:` payload. `[DONE]` yields nothing (the stream ends when the
    /// byte source closes). A decode failure yields one model-unavailable error with a truncated
    /// body (a raw upstream body in a client error frame is info-leak + noise).
    pub fn push_and_yield(&mut self, data: &str) -> Vec<ChunkResult> {
        if data.trim() == DONE {
            return Vec::new();
        }
        let chunk: CreateChatCompletionStreamResponse = match serde_json::from_str(data) {
            Ok(c) => c,
            Err(e) => {
                // The model backend streams a mid-stream `{"error":{...}}` frame on its own
                // failures; surface its message, not a decode error.
                // TODO(BT-16216): a `max_tokens` tool-call cutoff may be better projected as a
                // length termination than as an error at all.
                if let Ok(frame) = serde_json::from_str::<Value>(data)
                    && let Some(error_frame) = frame.get("error")
                    && let Some(message) = error_frame.get("message").and_then(Value::as_str)
                {
                    return vec![Err(mid_stream_error_frame(error_frame, message))];
                }
                return vec![Err(model_unavailable(
                    "model_chunk_undecodable",
                    format!(
                        "chat.completion.chunk decode failed: {e}; data: {}",
                        truncate(data, 256)
                    ),
                ))];
            }
        };
        self.push_chunk(chunk)
    }

    fn push_chunk(&mut self, chunk: CreateChatCompletionStreamResponse) -> Vec<ChunkResult> {
        // Usage may ride the finish chunk or a trailing choices-empty chunk; buffer for `flush_and_yield`.
        if let Some(usage) = chunk.usage {
            self.usage = Some(usage);
        }
        let mut out = Vec::new();
        let Some(choice) = chunk.choices.into_iter().next() else {
            return out;
        };
        let delta = choice.delta;

        if let Some(reasoning) = delta.reasoning_content.filter(|s| !s.is_empty()) {
            self.saw_output = true;
            out.push(Ok(SemanticChunk::ThinkingDelta(reasoning)));
        }
        if let Some(text) = delta
            .content
            .and_then(content_text)
            .filter(|s| !s.is_empty())
        {
            self.saw_output = true;
            out.push(Ok(SemanticChunk::TextDelta(text)));
        }
        for call in delta.tool_calls.into_iter().flatten() {
            self.saw_output = true;
            self.accumulate_tool_call(call, &mut out);
        }
        if let Some(finish) = choice.finish_reason {
            self.finish_reason = Some(finish);
        }
        out
    }

    /// The model may split `id`/`name`/`arguments` across chunks (Kimi/MAPI), so each is optional
    /// and merged by index; a name-only chunk is not dropped.
    fn accumulate_tool_call(
        &mut self,
        call: ChatCompletionMessageToolCallChunk,
        out: &mut Vec<ChunkResult>,
    ) {
        // Safety net beside the JSON-parse trigger: a new tool index means any earlier call's args
        // are done — flush the complete ones now (`try_complete` only emits parseable calls, never
        // fabricates an incomplete one).
        if !self.pos_by_index.contains_key(&call.index) {
            for pos in 0..self.tools.len() {
                if let Some(complete) = self.try_complete(pos) {
                    out.push(complete);
                }
            }
        }
        let pos = *self.pos_by_index.entry(call.index).or_insert_with(|| {
            self.tools.push(ToolCallAccumulator::default());
            self.tools.len() - 1
        });
        if self.tools[pos].emitted {
            // Args frozen at dispatch: drop late chunks so echoed history == what ran. A
            // non-whitespace late chunk is spurious model output past a complete value; meter it.
            let (name, args) = call
                .function
                .map(|f| (f.name, f.arguments))
                .unwrap_or((None, None));
            let spurious = call.id.is_some()
                || name.is_some()
                || args.as_deref().is_some_and(|a| !a.trim().is_empty());
            if spurious {
                // Per-chunk on a dribbling model, so debug; callers wanting a drop metric wrap
                // the parser (tool-bank counts these in its own observability layer).
                tracing::debug!(
                    tool_id = %self.tools[pos].id,
                    "dropping tool-call content after eager dispatch (spurious past a complete value)"
                );
            }
            return;
        }
        if let Some(id) = call.id.filter(|s| !s.is_empty()) {
            self.tools[pos].id = id;
        }
        if let Some(func) = call.function {
            if let Some(name) = func.name.filter(|s| !s.is_empty()) {
                self.tools[pos].name = name;
            }
            if let Some(args) = func.arguments {
                self.tools[pos].args_depth.feed(&args);
                self.tools[pos].args.push_str(&args);
            }
        }
        if let Some(complete) = self.try_complete(pos) {
            out.push(complete);
        }
    }

    /// Only containers (`{…}`/`[…]`) dispatch eagerly: a bare scalar parses while still growing
    /// (`42` before `42.5`) and would truncate, so scalars wait for `flush_and_yield`.
    fn try_complete(&mut self, pos: usize) -> Option<ChunkResult> {
        let tool = &self.tools[pos];
        let args = tool.args.trim_start();
        if tool.emitted
            || tool.id.is_empty()
            || tool.name.is_empty()
            || !(args.starts_with('{') || args.starts_with('['))
            // Balanced-depth gate first: the full parse runs at most once per call, not per chunk.
            || !tool.args_depth.is_balanced_container()
            || serde_json::from_str::<Value>(args).is_err()
        {
            return None;
        }
        Some(self.finalize(pos))
    }

    /// Empty args -> `{}`. Non-empty unparseable args mean the model cut the call off mid-arguments
    /// (e.g. `max_tokens`) — fail loud (the backend errors on this too; fabricating `{}` would be
    /// worse), with a clean message, not raw serde text.
    fn finalize(&mut self, pos: usize) -> ChunkResult {
        self.tools[pos].emitted = true;
        // Args are frozen at dispatch (late chunks are dropped), so hollowing the buffer is safe.
        let accumulated_args = std::mem::take(&mut self.tools[pos].args);
        let tool = &self.tools[pos];
        let raw_args = if accumulated_args.trim().is_empty() {
            "{}".to_string()
        } else {
            accumulated_args
        };
        let args: Value = serde_json::from_str(&raw_args).map_err(|_| {
            model_unavailable(
                "truncated_tool_call",
                format!(
                    "model truncated tool call `{}` (incomplete arguments)",
                    tool.name
                ),
            )
        })?;
        Ok(SemanticChunk::ToolCall(ToolCall {
            id: tool.id.clone(),
            name: tool.name.clone(),
            args,
            raw_args,
        }))
    }

    /// MAPI/Kimi omits the terminal `finish_reason` on `max_tokens` mid-token; synthesize `length`
    /// so a truncated-but-real call degrades cleanly. A stream with no output at all is a genuine
    /// upstream failure.
    pub fn flush_and_yield(&mut self) -> Vec<ChunkResult> {
        let mut out = Vec::new();
        for pos in 0..self.tools.len() {
            if self.tools[pos].emitted {
                continue;
            }
            // A tool accumulator that never got an id+name "opener" (reordered/corrupt tool stream)
            // can't be dispatched; fail loud rather than surface a nameless ToolCall downstream.
            if self.tools[pos].id.is_empty() || self.tools[pos].name.is_empty() {
                self.tools[pos].emitted = true;
                out.push(Err(model_unavailable(
                    "malformed_tool_call",
                    "model produced a tool call with no id/name".into(),
                )));
                continue;
            }
            out.push(self.finalize(pos));
        }
        if let Some(usage) = self.usage.take() {
            out.push(Ok(SemanticChunk::Usage(usage)));
        }
        let finish_reason = match self.finish_reason {
            Some(s) => s,
            None if self.saw_output => {
                tracing::warn!(
                    "upstream closed without a finish_reason after output; treating as length-truncated"
                );
                FinishReason::Length
            }
            None => {
                out.push(Err(model_unavailable(
                    "stream_truncated",
                    "model stream closed before producing any output".into(),
                )));
                return out;
            }
        };
        out.push(Ok(SemanticChunk::Stop { finish_reason }));
        out
    }
}

/// CC content is a flat string or multimodal parts; we serve text models, so concatenate text
/// parts and ignore the rest (image/video/audio parts appear on *input*, not the model's stream).
fn content_text(content: ChatCompletionMessageContent) -> Option<String> {
    match content {
        ChatCompletionMessageContent::Text(s) => Some(s),
        ChatCompletionMessageContent::Parts(parts) => Some(
            parts
                .into_iter()
                .filter_map(|part| match part {
                    ChatCompletionResponseContentPart::Text(text_part) => Some(text_part.text),
                    _ => None,
                })
                .collect(),
        ),
    }
}

#[cfg(test)]
#[path = "sse_parser_test.rs"]
mod tests;
