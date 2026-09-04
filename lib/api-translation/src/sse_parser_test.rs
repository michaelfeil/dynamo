//! `SseParser` conformance. Boundary/accumulation scenarios ported from dynamo
//! `stream_converter.rs` tests, asserting the semantic-chunk sequence (block *framing* is the
//! emitter's job, tested there). Plus TB regressions: split-chunk merge, eager dispatch, truncation.

use super::*;
use crate::SemanticChunk;
use crate::test_utils::*;
use serde_json::json;

#[test]
fn text_then_complete_tool_call() {
    let mut p = SseParser::default();
    let out = collect(
        &mut p,
        &[
            chunk(json!({"content": "I'll edit."}), None),
            chunk(
                tool_delta(0, Some("call-1"), Some("Edit"), Some(r#"{"path":"/x"}"#)),
                None,
            ),
            chunk(json!({}), Some("tool_calls")),
        ],
    );
    assert_eq!(kinds(&out), vec!["text", "tool_call", "stop"]);
    assert_eq!(tool(&out).id, "call-1");
    assert_eq!(tool(&out).args, json!({"path": "/x"}));
}

#[test]
fn tool_only_complete_in_one_chunk() {
    let mut p = SseParser::default();
    let out = collect(
        &mut p,
        &[chunk(
            tool_delta(0, Some("call-1"), Some("Read"), Some(r#"{"p":"/x"}"#)),
            Some("tool_calls"),
        )],
    );
    assert_eq!(kinds(&out), vec!["tool_call", "stop"]);
}

#[test]
fn text_only_stops_cleanly() {
    let mut p = SseParser::default();
    let out = collect(&mut p, &[chunk(json!({"content": "hi"}), Some("stop"))]);
    assert_eq!(kinds(&out), vec!["text", "stop"]);
}

#[test]
fn thinking_text_then_tool() {
    let mut p = SseParser::default();
    let out = collect(
        &mut p,
        &[
            chunk(json!({"reasoning_content": "hmm"}), None),
            chunk(json!({"content": "ok"}), None),
            chunk(
                tool_delta(0, Some("c1"), Some("Read"), Some(r#"{"p":"/x"}"#)),
                Some("tool_calls"),
            ),
        ],
    );
    assert_eq!(kinds(&out), vec!["thinking", "text", "tool_call", "stop"]);
}

/// The MiniMax regression at the parser layer: a tool call whose args dribble across chunks must
/// surface exactly once, only when the accumulated args parse — never as an early/empty call.
#[test]
fn streamed_args_surface_once_when_json_completes() {
    let mut p = SseParser::default();
    let mut out = Vec::new();
    out.extend(p.push_and_yield(&chunk(
        tool_delta(0, Some("call-1"), Some("Write"), Some("")),
        None,
    )));
    assert!(out.is_empty(), "no tool_call before args parse");
    out.extend(p.push_and_yield(&chunk(
        tool_delta(0, None, None, Some(r#"{"file_path":"w.txt""#)),
        None,
    )));
    out.extend(p.push_and_yield(&chunk(
        tool_delta(0, None, None, Some(r#", "content":"M2.5""#)),
        None,
    )));
    assert!(
        out.iter()
            .all(|r| !matches!(r.as_ref().unwrap(), SemanticChunk::ToolCall(_)))
    );
    out.extend(p.push_and_yield(&chunk(
        tool_delta(0, None, None, Some("}")),
        Some("tool_calls"),
    )));
    let completed: Vec<_> = out
        .iter()
        .filter(|r| matches!(r.as_ref().unwrap(), SemanticChunk::ToolCall(_)))
        .collect();
    assert_eq!(completed.len(), 1, "exactly one complete ToolCall");
    let fin = p.flush_and_yield();
    let t = match completed[0].as_ref().unwrap() {
        SemanticChunk::ToolCall(t) => t,
        _ => unreachable!(),
    };
    assert_eq!(t.args, json!({"file_path": "w.txt", "content": "M2.5"}));
    assert!(
        fin.iter()
            .all(|r| !matches!(r.as_ref().unwrap(), SemanticChunk::ToolCall(_)))
    );
}

/// `}`/`{` inside a string argument, split across chunks, plus an escaped quote: the eager-dispatch
/// gate is string- and escape-aware, so it waits for the real container to close.
#[test]
fn braces_inside_a_string_argument_do_not_dispatch_early() {
    let mut p = SseParser::default();
    let mut out = Vec::new();
    out.extend(p.push_and_yield(&chunk(
        tool_delta(0, Some("call-1"), Some("Grep"), Some(r#"{"pattern":"a}"#)),
        None,
    )));
    out.extend(p.push_and_yield(&chunk(tool_delta(0, None, None, Some("b{c")), None)));
    out.extend(p.push_and_yield(&chunk(tool_delta(0, None, None, Some(r#"\"q\""#)), None)));
    assert!(
        out.iter()
            .all(|r| !matches!(r.as_ref().unwrap(), SemanticChunk::ToolCall(_))),
        "dispatched while the string was still open"
    );
    out.extend(p.push_and_yield(&chunk(
        tool_delta(0, None, None, Some(r#""}"#)),
        Some("tool_calls"),
    )));
    let calls: Vec<_> = out
        .iter()
        .filter(|r| matches!(r.as_ref().unwrap(), SemanticChunk::ToolCall(_)))
        .collect();
    assert_eq!(calls.len(), 1, "exactly one complete ToolCall");
    let SemanticChunk::ToolCall(t) = calls[0].as_ref().unwrap() else {
        unreachable!()
    };
    assert_eq!(t.args, json!({"pattern": r#"a}b{c"q""#}));
}

/// Split id/name/args across chunks (Kimi/MAPI): a name-only chunk must not be dropped.
#[test]
fn split_id_name_args_merge_by_index() {
    let mut p = SseParser::default();
    let out = collect(
        &mut p,
        &[
            chunk(tool_delta(0, Some("c1"), None, None), None),
            chunk(tool_delta(0, None, Some("Search"), None), None),
            chunk(
                tool_delta(0, None, None, Some(r#"{"q":"x"}"#)),
                Some("tool_calls"),
            ),
        ],
    );
    let t = tool(&out);
    assert_eq!(t.id, "c1");
    assert_eq!(t.name, "Search");
    assert_eq!(t.args, json!({"q": "x"}));
}

#[test]
fn preserves_model_tool_id() {
    let mut p = SseParser::default();
    let out = collect(
        &mut p,
        &[chunk(
            tool_delta(0, Some("chatcmpl-tool-abc"), Some("F"), Some("{}")),
            Some("tool_calls"),
        )],
    );
    assert_eq!(
        tool(&out).id,
        "chatcmpl-tool-abc",
        "model id preserved, not minted"
    );
}

#[test]
fn multiple_tool_calls_each_complete() {
    let mut p = SseParser::default();
    let out = collect(
        &mut p,
        &[
            chunk(
                tool_delta(0, Some("a"), Some("R"), Some(r#"{"p":"/a"}"#)),
                None,
            ),
            chunk(
                tool_delta(1, Some("b"), Some("W"), Some(r#"{"p":"/b"}"#)),
                Some("tool_calls"),
            ),
        ],
    );
    assert_eq!(kinds(&out), vec!["tool_call", "tool_call", "stop"]);
}

/// Truncated tool args (never parse) at stream end -> hard error (the call is genuinely broken).
#[test]
fn incomplete_args_at_finish_errors() {
    let mut p = SseParser::default();
    p.push_and_yield(&chunk(
        tool_delta(0, Some("c1"), Some("Write"), Some("")),
        None,
    ));
    p.push_and_yield(&chunk(
        tool_delta(0, None, None, Some(r#"{"file_path":"trunc"#)),
        None,
    ));
    let fin = p.flush_and_yield();
    assert!(
        fin.iter()
            .any(|r| matches!(r, Err(TranslationError::UpstreamUnavailable { .. }))),
        "truncated args must error at finish (model cut-off = upstream)"
    );
}

#[test]
fn no_output_is_upstream_error() {
    let mut p = SseParser::default();
    let fin = p.flush_and_yield();
    assert!(matches!(
        fin.last(),
        Some(Err(TranslationError::UpstreamUnavailable { .. }))
    ));
}

#[test]
fn missing_finish_reason_after_output_synthesizes_length() {
    let mut p = SseParser::default();
    p.push_and_yield(&chunk(json!({"content": "partial"}), None));
    let fin = p.flush_and_yield();
    assert!(matches!(
        fin.last(),
        Some(Ok(SemanticChunk::Stop {
            finish_reason: FinishReason::Length
        }))
    ));
}

#[test]
fn usage_folds_into_finish() {
    let mut p = SseParser::default();
    p.push_and_yield(&chunk(json!({"content": "hi"}), Some("stop")));
    p.push_and_yield(
        &json!({
            "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        })
        .to_string(),
    );
    let fin = p.flush_and_yield();
    let usage = fin.iter().find_map(|r| match r {
        Ok(SemanticChunk::Usage(u)) => Some(u),
        _ => None,
    });
    assert_eq!(usage.map(|u| u.completion_tokens), Some(7));
}

#[test]
fn malformed_chunk_truncates_in_error() {
    let mut p = SseParser::default();
    let junk = format!("{{not json {}", "x".repeat(500));
    let out = p.push_and_yield(&junk);
    match &out[0] {
        Err(TranslationError::UpstreamUnavailable { detail: msg, .. }) => {
            assert!(msg.contains("decode failed"));
            assert!(msg.len() < 400, "body must be truncated in the error");
        }
        other => panic!("expected Inference error, got {other:?}"),
    }
}

#[test]
fn done_sentinel_yields_nothing() {
    let mut p = SseParser::default();
    assert!(p.push_and_yield("[DONE]").is_empty());
}

/// Byte-exact args: the ToolCall carries the model's verbatim arg string (spaces preserved), not a
/// compact reserialize — the KV-cache prefix depends on it.
#[test]
fn preserves_raw_arg_bytes() {
    let mut p = SseParser::default();
    let spaced = r#"{"q": "x",  "n": 1}"#;
    let out = collect(
        &mut p,
        &[chunk(
            tool_delta(0, Some("c1"), Some("S"), Some(spaced)),
            Some("tool_calls"),
        )],
    );
    let t = tool(&out);
    assert_eq!(
        t.raw_args, spaced,
        "raw_args must be the model's verbatim bytes"
    );
    assert_eq!(
        t.args,
        json!({"q": "x", "n": 1}),
        "parsed args for execution"
    );
}

/// A 4xx `code` on the frame is the caller's to act on, so it keeps the upstream status and the
/// upstream message instead of collapsing to "unavailable, please retry".
#[test]
fn coded_client_error_chunk_keeps_status_and_message() {
    let mut p = SseParser::default();
    let out =
        p.push_and_yield(r#"{"error":{"message":"Tool calls cutoff by max_tokens.","code":400}}"#);
    match &out[0] {
        Err(TranslationError::UpstreamResponse { status, body, .. }) => {
            assert_eq!(*status, http::StatusCode::BAD_REQUEST);
            assert!(body.contains("Tool calls cutoff"), "body: {body}");
        }
        other => panic!("expected UpstreamResponse, got {other:?}"),
    }
}

/// A 5xx `code` is upstream degradation, where the retry advice is right.
#[test]
fn coded_server_error_chunk_stays_unavailable() {
    let mut p = SseParser::default();
    let out = p.push_and_yield(r#"{"error":{"message":"framework error","code":500}}"#);
    assert!(matches!(
        &out[0],
        Err(TranslationError::UpstreamUnavailable { .. })
    ));
}

/// An uncoded mid-stream `{"error":{...}}` frame fails as an Upstream error, not a raw decode
/// failure — and its message, which may echo request content, stays out of the detail (no status
/// means the caller gets the generic line anyway).
#[test]
fn uncoded_error_chunk_is_unavailable_without_its_message() {
    let mut p = SseParser::default();
    let out = p.push_and_yield(
        r#"{"error":{"message":"Tool calls cutoff by max_tokens","type":"internal_server_error"}}"#,
    );
    match &out[0] {
        Err(TranslationError::UpstreamUnavailable { detail, .. }) => {
            assert!(!detail.contains("Tool calls cutoff"), "{detail}");
            assert!(detail.contains("mid-stream error frame"), "{detail}");
        }
        other => panic!("expected Upstream error, got {other:?}"),
    }
}
