//! Shared test fixtures for the API-translation layer: parser fixtures (`sse_parser_test.rs` +
//! `dynamo_conformance_tests/aggregator.rs`) that build CC `chat.completion.chunk` payloads and
//! classify [`SemanticChunk`]s, plus the emitter harness (`sse_emitter_test.rs` +
//! `dynamo_conformance_tests/stream_converter.rs`) that drives [`SseEmitter`] and parses the SSE
//! frames it renders.
#![allow(clippy::unwrap_used)]

use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::coding_adapter::{
    CodingAdapter, RenderedToolCall, ToolCallStatus, ToolCallToRender, ToolResultToRender,
};
use crate::model::{ServerToolCall, ToolCall};
use crate::sse_emitter::SseEmitter;
use crate::sse_parser::SseParser;
use crate::{ClientProtocol, SemanticChunk};

pub(crate) fn chunk(delta: Value, finish: Option<&str>) -> String {
    json!({
        "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    })
    .to_string()
}

pub(crate) fn tool_delta(
    index: u32,
    id: Option<&str>,
    name: Option<&str>,
    args: Option<&str>,
) -> Value {
    let mut func = serde_json::Map::new();
    if let Some(n) = name {
        func.insert("name".into(), json!(n));
    }
    if let Some(a) = args {
        func.insert("arguments".into(), json!(a));
    }
    let mut call = serde_json::Map::new();
    call.insert("index".into(), json!(index));
    if let Some(i) = id {
        call.insert("id".into(), json!(i));
    }
    call.insert("type".into(), json!("function"));
    call.insert("function".into(), Value::Object(func));
    json!({ "tool_calls": [call] })
}

pub(crate) fn collect(parser: &mut SseParser, datas: &[String]) -> Vec<SemanticChunk> {
    let mut out = Vec::new();
    for d in datas {
        for r in parser.push_and_yield(d) {
            out.push(r.expect("push error"));
        }
    }
    for r in parser.flush_and_yield() {
        out.push(r.expect("finish error"));
    }
    out
}

pub(crate) fn kinds(chunks: &[SemanticChunk]) -> Vec<&'static str> {
    chunks
        .iter()
        .map(|c| match c {
            SemanticChunk::TextDelta(_) => "text",
            SemanticChunk::ThinkingDelta(_) => "thinking",
            SemanticChunk::ToolCall(_) => "tool_call",
            SemanticChunk::Usage(_) => "usage",
            SemanticChunk::Stop { .. } => "stop",
        })
        .collect()
}

pub(crate) fn stop_reason(chunks: &[SemanticChunk]) -> Option<String> {
    chunks.iter().find_map(|c| match c {
        SemanticChunk::Stop { finish_reason } => Some(format!("{finish_reason:?}")),
        _ => None,
    })
}

pub(crate) fn tool(chunks: &[SemanticChunk]) -> &ToolCall {
    tools(chunks).into_iter().next().expect("no tool call")
}

pub(crate) fn tools(chunks: &[SemanticChunk]) -> Vec<&ToolCall> {
    chunks
        .iter()
        .filter_map(|c| match c {
            SemanticChunk::ToolCall(t) => Some(t),
            _ => None,
        })
        .collect()
}

pub(crate) fn text(chunks: &[SemanticChunk]) -> String {
    chunks
        .iter()
        .filter_map(|c| match c {
            SemanticChunk::TextDelta(t) => Some(t.as_str()),
            _ => None,
        })
        .collect()
}

pub(crate) fn tool_call(id: &str, name: &str, args: Value) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        raw_args: args.to_string(),
        args,
    }
}

/// One parsed SSE frame: the `event:` name (Messages) and the `data:` JSON (or the `[DONE]` marker).
pub(crate) struct Frame {
    pub(crate) event: Option<String>,
    pub(crate) data: Value,
    pub(crate) done: bool,
}

pub(crate) fn parse(raw: &str) -> Frame {
    let mut event = None;
    let mut data_line = "";
    for line in raw.lines() {
        if let Some(e) = line.strip_prefix("event: ") {
            event = Some(e.to_string());
        } else if let Some(d) = line.strip_prefix("data: ") {
            data_line = d;
        }
    }
    if data_line == "[DONE]" {
        return Frame {
            event,
            data: Value::Null,
            done: true,
        };
    }
    Frame {
        event,
        data: serde_json::from_str(data_line).unwrap(),
        done: false,
    }
}

/// Drive `body` against a fresh emitter, then drain all emitted frames (parsed).
pub(crate) async fn drive<F, Fut>(protocol: ClientProtocol, body: F) -> Vec<Frame>
where
    F: FnOnce(SseEmitter) -> Fut,
    Fut: std::future::Future<Output = SseEmitter>,
{
    drive_with_coding_adapter(protocol, None, body).await
}

pub(crate) async fn drive_with_coding_adapter<F, Fut>(
    protocol: ClientProtocol,
    coding_adapter: Option<Box<dyn CodingAdapter>>,
    body: F,
) -> Vec<Frame>
where
    F: FnOnce(SseEmitter) -> Fut,
    Fut: std::future::Future<Output = SseEmitter>,
{
    let (tx, mut rx) = mpsc::channel::<String>(256);
    let framing = protocol
        .envelope(coding_adapter, Default::default())
        .stream_framing("m".into());
    let emitter = SseEmitter::new(tx, framing);
    let emitter = body(emitter).await;
    drop(emitter); // drops the tx clone so the receiver closes
    let mut frames = Vec::new();
    while let Some(f) = rx.recv().await {
        frames.push(parse(&f));
    }
    frames
}

pub(crate) fn event_names(frames: &[Frame]) -> Vec<String> {
    frames
        .iter()
        .map(|f| {
            if f.done {
                "[DONE]".into()
            } else {
                f.event.clone().unwrap_or_else(|| "data".into())
            }
        })
        .collect()
}

pub(crate) fn server_tool_call(call: ToolCall) -> ServerToolCall {
    let provider = crate::hooks::reserved_tool_provider(&call.name)
        .expect("fixture tool name is qualified")
        .to_string();
    ServerToolCall { call, provider }
}

// --- fake coding adapters ----------------------------------------------------
//
// The real Claude Code / Codex adapters live with the caller that runs their web-search machinery
// (tool-bank); these minimal stand-ins reproduce the item shapes the framings' adapter bridges
// must carry, so the seam stays covered without the execution-side crates.

/// Messages-flavored search adapter: renders `server_tool_use` / `web_search_tool_result` blocks
/// for one bound tool name, extracting citation pairs from a `{"payload":{"results":[...]}}`
/// result shape (page text dropped).
pub(crate) struct FakeMessagesSearchAdapter {
    pub tool_name: &'static str,
}

impl CodingAdapter for FakeMessagesSearchAdapter {
    fn render_tool_call(&self, call: &ToolCallToRender<'_>) -> Option<RenderedToolCall> {
        if call.tool_name != self.tool_name {
            return None;
        }
        let args: Value = serde_json::from_str(call.args).ok()?;
        let query = args.get("search_queries")?.get(0)?.clone();
        let input = match call.status {
            // The streamed block starts empty; the SDK builds `input` from the delta.
            ToolCallStatus::InProgress => json!({}),
            ToolCallStatus::Completed | ToolCallStatus::Failed => json!({"query": query}),
        };
        Some(RenderedToolCall {
            item: json!({
                "type": "server_tool_use",
                "id": format!("srvtoolu_{}", call.id),
                "name": "web_search",
                "input": input,
            }),
            lifecycle_events: Vec::new(),
        })
    }

    fn render_tool_result(&self, result: &ToolResultToRender<'_>) -> Option<Value> {
        if result.tool_name != self.tool_name {
            return None;
        }
        let content = if result.is_error {
            json!({"type": "web_search_tool_result_error", "error_code": "unavailable"})
        } else {
            let citations: Vec<Value> = result
                .content
                .get("payload")
                .and_then(|payload| payload.get("results"))
                .and_then(Value::as_array)
                .map(|entries| {
                    entries
                        .iter()
                        .filter_map(|entry| {
                            Some(json!({
                                "type": "web_search_result",
                                "title": entry.get("title")?,
                                "url": entry.get("url")?,
                            }))
                        })
                        .collect()
                })
                .unwrap_or_default();
            json!(citations)
        };
        Some(json!({
            "type": "web_search_tool_result",
            "tool_use_id": format!("srvtoolu_{}", result.id),
            "content": content,
        }))
    }
}

/// Responses-flavored search adapter: renders `web_search_call` items with Codex-shaped `action`
/// (a `queries` array the typed slot cannot hold) plus the lifecycle events, and splices rendered
/// items back into a finished body by id.
pub(crate) struct FakeResponsesSearchAdapter {
    pub tool_name: &'static str,
}

impl CodingAdapter for FakeResponsesSearchAdapter {
    fn render_tool_call(&self, call: &ToolCallToRender<'_>) -> Option<RenderedToolCall> {
        if call.tool_name != self.tool_name {
            return None;
        }
        let args: Value = serde_json::from_str(call.args).ok()?;
        let queries = args
            .get("search_queries")
            .cloned()
            .or_else(|| args.get("q").map(|q| json!([q])))?;
        let (status, lifecycle_events) = match call.status {
            ToolCallStatus::InProgress => (
                "in_progress",
                vec![
                    "response.web_search_call.in_progress",
                    "response.web_search_call.searching",
                ],
            ),
            ToolCallStatus::Completed => ("completed", vec!["response.web_search_call.completed"]),
            ToolCallStatus::Failed => ("failed", vec!["response.web_search_call.completed"]),
        };
        Some(RenderedToolCall {
            item: json!({
                "type": "web_search_call",
                "id": call.id,
                "status": status,
                "action": {"type": "search", "queries": queries, "query": queries.get(0)},
            }),
            lifecycle_events,
        })
    }

    fn splice_rendered_calls(
        &self,
        mut body: Value,
        rendered: &std::collections::HashMap<String, Value>,
    ) -> Value {
        if let Some(items) = body.get_mut("output").and_then(Value::as_array_mut) {
            for item in items {
                if let Some(replacement) = item
                    .get("id")
                    .and_then(Value::as_str)
                    .and_then(|id| rendered.get(id))
                {
                    *item = replacement.clone();
                }
            }
        }
        body
    }
}
