// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use baseten_parsers::{
    Event, Tool, ToolParserInput, ToolStream,
    vllm::{FAMILIES, VllmToolStream},
};
use serde_json::json;

fn tools() -> Vec<Tool> {
    vec![Tool {
        name: "weather".into(),
        description: None,
        strict: None,
        parameters: json!({"type":"object","properties":{"city":{"type":"string"}}}),
    }]
}
const KIMI_START: &str = "<|tool_calls_section_begin|><|tool_call_begin|>functions.weather:0<|tool_call_argument_begin|>";
const KIMI_END: &str = "<|tool_call_end|><|tool_calls_section_end|>";

#[test]
fn every_family_has_request_local_lifecycle() {
    for family in FAMILIES {
        let mut p = ToolStream::new("vllm", family, &tools()).unwrap();
        assert!(!p.prefers_tokens());
        assert!(p.advance(Some(ToolParserInput::Tokens(&[1]))).is_err());
        assert_eq!(
            p.advance(Some(ToolParserInput::Text("hello")))
                .unwrap()
                .normal_text,
            "hello",
            "{family}"
        );
        assert!(p.advance(None).is_ok(), "{family}");
        assert!(p.advance(None).is_err());
        assert!(p.advance(Some(ToolParserInput::Text("late"))).is_err());
    }
    assert!(ToolStream::new("unknown", "glm47", &[]).is_err());
    assert!(ToolStream::new("vllm", "unknown", &[]).is_err());
}

#[test]
fn whole_call_families_preserve_arguments_at_every_unicode_split() {
    let cases = [
        (
            "glm47",
            "<tool_call>weather<arg_key>city</arg_key><arg_value>café 🚀 &amp;</arg_value></tool_call>",
        ),
        (
            "qwen3_coder",
            "<tool_call>\n<function=weather>\n<parameter=city>café 🚀 &amp;</parameter>\n</function>\n</tool_call>",
        ),
        (
            "deepseek_v4",
            "<｜DSML｜tool_calls><｜DSML｜invoke name=\"weather\"><｜DSML｜parameter name=\"city\" string=\"true\">café 🚀 &amp;</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>",
        ),
        (
            "minimax_m2",
            "<minimax:tool_call><invoke name=\"weather\"><parameter name=\"city\">café 🚀 &amp;</parameter></invoke></minimax:tool_call>",
        ),
        (
            "glm45",
            "<tool_call>weather\n<arg_key>city</arg_key>\n<arg_value>café 🚀 &amp;</arg_value>\n</tool_call>",
        ),
        (
            "deepseek_v32",
            "<｜DSML｜function_calls><｜DSML｜invoke name=\"weather\"><｜DSML｜parameter name=\"city\" string=\"true\">café 🚀 &amp;</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜function_calls>",
        ),
        (
            "deepseek_v41",
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"weather\"><｜DSML｜ parameter name=\"city\" string=\"true\">café 🚀 &amp;</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>",
        ),
        (
            "minimax_m3",
            "]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"weather\">]<]minimax[>[<city>café 🚀 &amp;]<]minimax[>[</city>]<]minimax[>[</invoke>]<]minimax[>[</tool_call>",
        ),
        (
            "mimo",
            "<tool_call><function=weather><parameter=city>café 🚀 &amp;</parameter></function></tool_call>",
        ),
        (
            "seed_oss",
            "<seed:tool_call>\n<function=weather>\n<parameter=city>café 🚀 &amp;</parameter>\n</function>\n</seed:tool_call>",
        ),
    ];
    for (family, input) in cases {
        for split in input.char_indices().map(|(i, _)| i).chain([input.len()]) {
            let mut p = ToolStream::new("vllm", family, &tools()).unwrap();
            assert_eq!(p.completion_semantics(), "native");
            let mut calls = vec![];
            for chunk in [&input[..split], "", &input[split..]] {
                calls.extend(p.advance(Some(ToolParserInput::Text(chunk))).unwrap().calls);
            }
            calls.extend(p.advance(None).unwrap().calls);
            assert_eq!(calls.len(), 1, "{family} split {split}");
            assert!(calls[0].complete);
            assert_eq!(calls[0].name.as_deref(), Some("weather"));
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&calls[0].arguments).unwrap(),
                json!({"city":"café 🚀 &amp;"}),
                "{family}"
            );
        }
    }
}

#[test]
fn unfinished_kimi_arguments_stream_and_eof_closure_retains_id() {
    let mut p = ToolStream::new("vllm", "kimi_k2", &tools()).unwrap();
    assert_eq!(p.completion_semantics(), "stream_boundary");
    let start = p.advance(Some(ToolParserInput::Text(KIMI_START))).unwrap();
    assert_eq!(start.calls[0].name.as_deref(), Some("weather"));
    for text in ["{\"city\":", "\"Paris\"", "}"] {
        let next = p.advance(Some(ToolParserInput::Text(text))).unwrap();
        assert_eq!(next.calls.len(), 1);
        assert_eq!(next.calls[0].arguments, text);
        assert!(!next.calls[0].complete);
    }
    assert!(
        p.advance(Some(ToolParserInput::Text(KIMI_END)))
            .unwrap()
            .calls
            .is_empty()
    );
    let tail = p.advance(None).unwrap();
    assert_eq!(tail.calls.len(), 1);
    assert!(tail.calls[0].complete);
    assert_eq!(tail.calls[0].id.as_deref(), Some("functions.weather:0"));
}

#[test]
fn truncated_kimi_is_not_marked_complete_and_errors_are_terminal() {
    let mut p = ToolStream::new("vllm", "kimi_k2", &tools()).unwrap();
    p.advance(Some(ToolParserInput::Text(&format!(
        "{KIMI_START}{{\"city\":"
    ))))
    .unwrap();
    let error = p.advance(None).unwrap_err();
    assert!(error.events.is_empty());
    assert!(p.advance(None).is_err());
}

#[test]
fn committed_events_survive_a_later_error_in_the_same_chunk() {
    let mut p = VllmToolStream::new("qwen3_coder", &tools()).unwrap();
    let input = "prefix<tool_call>\n<function=weather>\n<parameter=city>Paris</parameter>\n</function>\n</tool_call>\n<tool_call>\n<bad>\n</tool_call>";
    let error = p.advance(Some(input)).unwrap_err();
    assert!(matches!(&error.events[0], Event::Text(text) if text == "prefix"));
    assert!(error.events.iter().any(|event| matches!(event, Event::ToolCall(call) if call.complete && call.name.as_deref() == Some("weather"))));
    assert!(p.advance(Some("late")).unwrap_err().events.is_empty());
}

#[test]
fn kimi_closes_previous_call_before_starting_the_next() {
    let mut p = VllmToolStream::new("kimi_k2", &tools()).unwrap();
    let first = format!("{KIMI_START}{{}}<|tool_call_end|>");
    p.advance(Some(&first)).unwrap();
    let events = p
        .advance(Some(
            "<|tool_call_begin|>functions.weather:1<|tool_call_argument_begin|>{}",
        ))
        .unwrap();
    assert!(
        matches!(&events[0], Event::ToolCall(call) if call.tool_index == 0 && call.complete && call.id.as_deref() == Some("functions.weather:0"))
    );
    assert!(
        matches!(&events[1], Event::ToolCall(call) if call.tool_index == 1 && !call.complete && call.name.as_deref() == Some("weather"))
    );
    p.advance(Some(KIMI_END)).unwrap();
    let tail = p.advance(None).unwrap();
    assert!(matches!(&tail[0], Event::ToolCall(call) if call.tool_index == 1 && call.complete));
}

#[test]
fn independent_requests_do_not_share_partial_state() {
    let mut first = ToolStream::new("vllm", "glm47", &tools()).unwrap();
    let mut second = ToolStream::new("vllm", "glm47", &tools()).unwrap();
    assert!(
        first
            .advance(Some(ToolParserInput::Text("<tool_call>wea")))
            .unwrap()
            .calls
            .is_empty()
    );
    assert_eq!(
        second
            .advance(Some(ToolParserInput::Text("plain")))
            .unwrap()
            .normal_text,
        "plain"
    );
    assert_eq!(
        first
            .advance(Some(ToolParserInput::Text("ther</tool_call>")))
            .unwrap()
            .calls[0]
            .name
            .as_deref(),
        Some("weather")
    );
    assert!(second.advance(None).unwrap().calls.is_empty());
}
