//! Ports of `lib/llm/src/protocols/anthropic/stream_converter.rs` (`AnthropicStreamConverter`):
//! parser-side boundary decisions (`SseParser`) and Anthropic block *framing* (`SseEmitter`).
//! Oracle: basetenlabs/dynamo @ 68dec805 (Apache-2.0). SPDX-License-Identifier: Apache-2.0.

use serde_json::json;

use super::{chunk, collect, kinds, stop_reason, tool_delta, tools};
use crate::model::Termination;
use crate::sse_parser::SseParser;
use crate::test_utils::{drive, event_names, tool_call};
use crate::{ClientProtocol, SemanticChunk};
use dynamo_protocols::types::FinishReason;

// --- parser-side boundary (SseParser) ---------------------------------------

/// upstream: stream_converter.rs::test_minimax_m2_claude_code_session_replay
/// ADAPT (parser side): the full on-the-wire MiniMax-M2.5 session — thinking, text, then a tool
/// call whose args dribble across four chunks. The parser holds the call until its args parse, then
/// surfaces ONE complete `ToolCall`. (Dynamo's converter streams N `input_json_delta` events; our
/// emitter renders the one complete call as a single delta — see `framing/messages.rs` header + `sse_emitter_test.rs`.)
#[test]
fn minimax_m2_session_replay() {
    let mut parser = SseParser::default();
    let out = collect(
        &mut parser,
        &[
            chunk(json!({"reasoning_content": "The user wants me "}), None),
            chunk(
                json!({"reasoning_content": "to write to whoami.txt."}),
                None,
            ),
            chunk(json!({"content": "\n\n\n"}), None),
            chunk(
                tool_delta(0, Some("chatcmpl-tool-x"), Some("Write"), Some("")),
                None,
            ),
            chunk(
                tool_delta(0, None, None, Some(r#"{"file_path":"whoami.txt""#)),
                None,
            ),
            chunk(
                tool_delta(
                    0,
                    None,
                    None,
                    Some(r#", "content":"MiniMaxAI/MiniMax-M2.5""#),
                ),
                None,
            ),
            chunk(tool_delta(0, None, None, Some("}")), None),
        ],
    );
    assert_eq!(
        kinds(&out),
        vec!["thinking", "thinking", "text", "tool_call", "stop"]
    );
    let tool_calls = tools(&out);
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(
        tool_calls[0].id, "chatcmpl-tool-x",
        "model id preserved verbatim"
    );
    assert_eq!(tool_calls[0].name, "Write");
    assert_eq!(
        tool_calls[0].args,
        json!({"file_path": "whoami.txt", "content": "MiniMaxAI/MiniMax-M2.5"})
    );
    assert_eq!(
        tool_calls[0].raw_args, r#"{"file_path":"whoami.txt", "content":"MiniMaxAI/MiniMax-M2.5"}"#,
        "verbatim arg bytes (spacing preserved for KV-cache prefix fidelity)"
    );
    // No terminal finish_reason in the captured stream -> synthesized Length.
    assert_eq!(stop_reason(&out).as_deref(), Some("Length"));
}

// --- emitter block framing (SseEmitter) -------------------------------------
// (Already-covered framing tests live as self-rolled units in sse_emitter_test.rs:
// test_text_block_stops_before_tool_block_starts / test_tool_only_response_no_text_block /
// test_thinking_text_then_tool_call -> messages_thinking_text_tool_block_ordering &
// messages_tool_use_stop_is_last_for_block_no_orphan.)

/// upstream: stream_converter.rs::test_text_only_response_stop_in_end_events
/// PORT: a text-only turn's block is closed at finish (not early), then message_delta/message_stop.
#[tokio::test]
async fn conformance_text_only_stop_in_end_events() {
    let usage = serde_json::from_value(
        json!({"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
    )
    .unwrap();
    let frames = drive(ClientProtocol::Messages, |mut e| async move {
        e.push(&SemanticChunk::TextDelta("Hello world".into()))
            .await
            .unwrap();
        e.finish(Termination::Model(FinishReason::Stop), &usage)
            .await
            .unwrap();
        e
    })
    .await;
    assert_eq!(
        event_names(&frames),
        vec![
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    // Text block stop rides finish, at index 0.
    assert_eq!(frames[3].data["index"], 0);
}

/// upstream: stream_converter.rs::test_thinking_only_closed_in_end_events
/// PORT: a thinking-only turn closes at finish with signature_delta + stop before message_delta.
#[tokio::test]
async fn conformance_thinking_only_closed_in_finish() {
    let usage = serde_json::from_value(
        json!({"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
    )
    .unwrap();
    let frames = drive(ClientProtocol::Messages, |mut e| async move {
        e.push(&SemanticChunk::ThinkingDelta("Deep thought...".into()))
            .await
            .unwrap();
        e.finish(Termination::Model(FinishReason::Stop), &usage)
            .await
            .unwrap();
        e
    })
    .await;
    assert_eq!(
        event_names(&frames),
        vec![
            "message_start",
            "content_block_start", // thinking idx0
            "content_block_delta", // thinking_delta
            "content_block_delta", // signature_delta (close thinking)
            "content_block_stop",  // thinking stop
            "message_delta",
            "message_stop",
        ]
    );
    assert_eq!(frames[3].data["delta"]["type"], "signature_delta");
}

/// upstream: stream_converter.rs::test_multiple_tool_calls_each_stopped_inline
/// PORT: each complete tool call opens + closes its own block inline; finish adds no leftover stop.
#[tokio::test]
async fn conformance_multiple_tool_calls_each_stopped_inline() {
    let usage = serde_json::from_value(
        json!({"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
    )
    .unwrap();
    let frames = drive(ClientProtocol::Messages, |mut e| async move {
        e.push(&SemanticChunk::ToolCall(tool_call(
            "call-1",
            "Read",
            json!({"path": "/tmp/a.txt"}),
        )))
        .await
        .unwrap();
        e.push(&SemanticChunk::ToolCall(tool_call(
            "call-2",
            "Write",
            json!({"path": "/tmp/b.txt"}),
        )))
        .await
        .unwrap();
        e.finish(Termination::Model(FinishReason::ToolCalls), &usage)
            .await
            .unwrap();
        e
    })
    .await;
    assert_eq!(
        event_names(&frames),
        vec![
            "message_start",
            "content_block_start", // tool call-1 idx0
            "content_block_delta",
            "content_block_stop",
            "content_block_start", // tool call-2 idx1
            "content_block_delta",
            "content_block_stop",
            "message_delta", // finish: no leftover block stop
            "message_stop",
        ]
    );
    assert_eq!(frames[1].data["index"], 0);
    assert_eq!(frames[4].data["index"], 1);
}

/// upstream: stream_converter.rs::test_streaming_tool_use_id_is_rewritten_to_toolu_prefix
/// ADAPT: dynamo rewrites the tool_use id to a `toolu_` prefix. We preserve the model's id verbatim
/// (cache-coherent replay, see sse_parser.rs / framing/messages.rs) — no rewrite, no synthetic prefix.
#[tokio::test]
async fn conformance_tool_use_id_preserved_not_rewritten() {
    let frames = drive(ClientProtocol::Messages, |mut e| async move {
        e.push(&SemanticChunk::ToolCall(tool_call(
            "chatcmpl-tool-DEADBEEF",
            "Edit",
            json!({"file_path": "/tmp/test.txt"}),
        )))
        .await
        .unwrap();
        e
    })
    .await;
    let start = frames
        .iter()
        .find(|frame| frame.data["content_block"]["type"] == json!("tool_use"))
        .unwrap();
    assert_eq!(
        start.data["content_block"]["id"], "chatcmpl-tool-DEADBEEF",
        "model id preserved verbatim (divergence: dynamo rewrites to toolu_)"
    );
    assert!(
        !start.data["content_block"]["id"]
            .as_str()
            .unwrap()
            .starts_with("toolu_")
    );
}
