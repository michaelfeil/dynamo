//! Ports of `lib/llm/src/protocols/openai/chat_completions/aggregator.rs` (`DeltaAggregator`) — the
//! direct reference behavior for `SseParser` (CC delta stream -> aggregated tool calls / text / finish).
//! Source: basetenlabs/dynamo @ 68dec805 (Apache-2.0). SPDX-License-Identifier: Apache-2.0.

use serde_json::json;

use super::{chunk, collect, kinds, stop_reason, text, tool_delta, tools};
use crate::model::TranslationError;
use crate::sse_parser::SseParser;

/// upstream: aggregator.rs::test_issue_8640_split_tool_call_arguments_reconstructed
/// PORT: a single tool call split as name-only chunk then two arg fragments must reconstruct the
/// full arguments, not drop the fragments.
#[test]
fn split_tool_call_args_reconstructed() {
    let mut p = SseParser::default();
    let out = collect(
        &mut p,
        &[
            chunk(tool_delta(0, Some("tc1"), Some("get_weather"), None), None),
            chunk(tool_delta(0, None, None, Some(r#"{"city":"#)), None),
            chunk(
                tool_delta(0, None, None, Some(r#""Tokyo"}"#)),
                Some("tool_calls"),
            ),
        ],
    );
    let ts = tools(&out);
    assert_eq!(ts.len(), 1, "exactly one reconstructed tool call");
    assert_eq!(ts[0].id, "tc1");
    assert_eq!(ts[0].name, "get_weather");
    assert_eq!(ts[0].args, json!({"city": "Tokyo"}));
    assert_eq!(ts[0].raw_args, r#"{"city":"Tokyo"}"#);
}

/// upstream: aggregator.rs::test_parallel_tool_calls_interleaved_chunks_aggregate_independently
/// PORT: two parallel calls (index 0/1) whose chunks interleave must aggregate independently.
#[test]
fn parallel_tool_calls_interleaved_aggregate_independently() {
    let mut p = SseParser::default();
    let out = collect(
        &mut p,
        &[
            chunk(tool_delta(0, Some("tc0"), Some("get_weather"), None), None),
            chunk(tool_delta(1, Some("tc1"), Some("get_time"), None), None),
            chunk(tool_delta(0, None, None, Some(r#"{"city":"#)), None),
            chunk(tool_delta(1, None, None, Some(r#"{"tz":"#)), None),
            chunk(tool_delta(0, None, None, Some(r#""Tokyo"}"#)), None),
            chunk(
                tool_delta(1, None, None, Some(r#""JST"}"#)),
                Some("tool_calls"),
            ),
        ],
    );
    let ts = tools(&out);
    assert_eq!(ts.len(), 2, "both parallel tool calls surface");
    assert_eq!(ts[0].id, "tc0");
    assert_eq!(ts[0].name, "get_weather");
    assert_eq!(ts[0].args, json!({"city": "Tokyo"}));
    assert_eq!(ts[1].id, "tc1");
    assert_eq!(ts[1].name, "get_time");
    assert_eq!(ts[1].args, json!({"tz": "JST"}));
}

/// upstream: aggregator.rs::test_fragment_only_chunks_without_opener_drop_cleanly
/// ADAPT: dynamo silently drops an args-only call that never carried id/name (tool_calls stays
/// None). We fail loud instead — a nameless tool call can't be dispatched, so `finish()` raises an
/// `Upstream` error rather than surface a malformed ToolCall.
#[test]
fn fragment_only_chunks_without_opener_errors() {
    let mut p = SseParser::default();
    for r in p
        .push_and_yield(&chunk(
            tool_delta(0, None, None, Some(r#"{"orphaned":true}"#)),
            Some("stop"),
        ))
        .chunks
    {
        assert!(
            r.is_ok(),
            "no eager dispatch/error mid-stream (id/name absent)"
        );
    }
    let fin = p.flush_and_yield();
    assert!(
        fin.iter()
            .any(|r| matches!(r, Err(TranslationError::UpstreamUnavailable { .. }))),
        "args-only tool call with no id/name fails loud at finish"
    );
}

/// upstream: aggregator.rs::test_multiple_deltas_same_choice
/// PORT: two text deltas + a terminal finish. We surface per-delta (downstream concatenates); the
/// concatenation must match and the model's finish_reason passes through.
#[test]
fn multiple_deltas_same_choice_surface_in_order() {
    let mut p = SseParser::default();
    let out = collect(
        &mut p,
        &[
            chunk(json!({"content": "Hello,"}), None),
            chunk(json!({"content": " world!"}), Some("stop")),
        ],
    );
    assert_eq!(kinds(&out), vec!["text", "text", "stop"]);
    assert_eq!(text(&out), "Hello, world!");
    assert_eq!(stop_reason(&out).as_deref(), Some("Stop"));
}

/// upstream: aggregator.rs::test_preserves_intermediate_whitespace_chunks
/// PORT: a whitespace-only content delta between tokens must be preserved (not trimmed away).
#[test]
fn preserves_intermediate_whitespace_delta() {
    let mut p = SseParser::default();
    let out = collect(
        &mut p,
        &[
            chunk(json!({"content": "Hello"}), None),
            chunk(json!({"content": " "}), None),
            chunk(json!({"content": "world"}), Some("stop")),
        ],
    );
    assert_eq!(
        kinds(&out),
        vec!["text", "text", "text", "stop"],
        "the whitespace-only delta is not dropped"
    );
    assert_eq!(text(&out), "Hello world");
}

/// upstream: aggregator.rs::test_empty_tool_calls_preserves_original_finish_reason
/// PORT: an empty `tool_calls` array must not fabricate a call, and the finish_reason is preserved.
#[test]
fn empty_tool_calls_preserves_finish_reason() {
    let mut p = SseParser::default();
    let out = collect(
        &mut p,
        &[chunk(
            json!({"content": "resp", "tool_calls": []}),
            Some("length"),
        )],
    );
    assert_eq!(kinds(&out), vec!["text", "stop"], "no phantom tool call");
    assert_eq!(stop_reason(&out).as_deref(), Some("Length"));
}

/// upstream: aggregator.rs::test_tool_calling_finish_reason_override_from_stop (and _from_length,
/// _from_stop_alternative)
/// ADAPT: dynamo rewrites finish_reason to ToolCalls whenever tool calls are present. We pass the
/// model's finish_reason through verbatim — a complete tool call with finish=Stop stays Stop.
#[test]
fn finish_reason_passthrough_not_overridden_by_tool_calls() {
    let mut p = SseParser::default();
    let out = collect(
        &mut p,
        &[
            chunk(json!({"content": "I'll check the weather."}), None),
            chunk(
                tool_delta(
                    0,
                    Some("call-1"),
                    Some("get_weather"),
                    Some(r#"{"location":"New York"}"#),
                ),
                Some("stop"),
            ),
        ],
    );
    assert_eq!(tools(&out).len(), 1);
    assert_eq!(
        stop_reason(&out).as_deref(),
        Some("Stop"),
        "finish_reason is NOT overridden to ToolCalls (divergence from dynamo)"
    );
}

/// upstream: aggregator.rs::test_tool_calling_finish_reason_override_from_none
/// ADAPT: dynamo sets finish_reason=ToolCalls when it is absent and tool calls are present. We
/// synthesize Length on a missing terminal finish_reason after output (tool call counts as output).
#[test]
fn tool_call_missing_finish_reason_synthesizes_length() {
    let mut p = SseParser::default();
    let out = collect(
        &mut p,
        &[chunk(
            tool_delta(
                0,
                Some("call-1"),
                Some("calculate"),
                Some(r#"{"expr":"2+2"}"#),
            ),
            None,
        )],
    );
    assert_eq!(tools(&out).len(), 1);
    assert_eq!(
        stop_reason(&out).as_deref(),
        Some("Length"),
        "missing finish_reason after output -> synthesized Length (not dynamo's ToolCalls)"
    );
}
