// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use baseten_parsers::{
    Event, Tool, UnifiedParserInit, UnifiedParserOutput, UnifiedStream, request_init,
    unified_parser_families, upstream,
};
use serde_json::json;

fn tools() -> Vec<Tool> {
    vec![Tool {
        name: "weather".into(),
        description: None,
        parameters: json!({"type":"object", "properties":{"city":{"type":"string"}}}),
        strict: None,
    }]
}

#[test]
fn registered_families_use_the_unified_lifecycle() {
    for family in unified_parser_families() {
        let mut stream =
            UnifiedStream::new(family, &tools(), UnifiedParserInit::default()).unwrap();
        stream.advance(Some("hello")).unwrap();
        stream.advance(None).unwrap();
        assert!(stream.advance(Some("late")).is_err(), "{family}");
    }
    assert!(UnifiedStream::new("missing", &[], UnifiedParserInit::default()).is_err());
}

#[test]
fn reasoning_text_and_tool_calls_share_one_ordered_stream() {
    let mut stream = UnifiedStream::new("qwen3", &tools(), UnifiedParserInit::default()).unwrap();
    let mut events = stream.advance(Some(
        "<think>Check weather.</think>Looking up. <tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>Sunny."
    )).unwrap();
    events.extend(stream.advance(None).unwrap());
    assert_eq!(events[0], Event::Reasoning("Check weather.".into()));
    assert_eq!(events[1], Event::Text("Looking up. ".into()));
    assert_eq!(events.last(), Some(&Event::Text("Sunny.".into())));
    let calls: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            Event::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name.as_deref(), Some("weather"));
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&calls[0].arguments).unwrap(),
        json!({"city":"Paris"})
    );
    assert!(calls[0].complete);
}

#[test]
fn guided_named_call_uses_the_same_stream() {
    let init = request_init(
        vec![],
        "none",
        "guided_json",
        Some("weather".into()),
        "reject",
    )
    .unwrap();
    let mut stream = UnifiedStream::new("qwen3", &tools(), init).unwrap();
    let mut events = stream.advance(Some(r#"{"city":"Paris"}"#)).unwrap();
    events.extend(stream.advance(None).unwrap());
    let [Event::ToolCall(call)] = events.as_slice() else {
        panic!("expected one call")
    };
    assert_eq!(call.name.as_deref(), Some("weather"));
    assert!(call.complete);
}

#[test]
fn partial_errors_keep_committed_events_and_close_the_stream() {
    struct FailingParser;
    impl upstream::UnifiedParser for FailingParser {
        fn parse_into(&mut self, _: &str, output: &mut UnifiedParserOutput) -> anyhow::Result<()> {
            output.push_text("committed");
            anyhow::bail!("injected failure")
        }

        fn finish(&mut self) -> anyhow::Result<UnifiedParserOutput> {
            Ok(UnifiedParserOutput::default())
        }
    }
    let mut stream =
        UnifiedStream::from_parser(Box::new(FailingParser), UnifiedParserInit::default()).unwrap();
    let error = stream.advance(Some("input")).unwrap_err();
    assert_eq!(error.events, vec![Event::Text("committed".into())]);
    assert!(
        stream
            .advance(Some("retry"))
            .unwrap_err()
            .to_string()
            .contains("closed")
    );
}

#[test]
fn invalid_configuration_is_rejected_in_rust() {
    assert!(request_init(vec![], "invalid", "native", None, "reject").is_err());
    assert!(request_init(vec![], "none", "native", Some("weather".into()), "reject").is_err());
    assert!(request_init(vec![], "none", "native", None, "invalid").is_err());
}

#[test]
fn harmony_v1_preserves_order_at_every_character_boundary() {
    let text = "<|channel|>analysis<|message|>Check 🌤️.<|end|>\
        <|start|>assistant<|channel|>commentary<|message|>Looking up.<|end|>\
        <|start|>assistant to=functions.weather<|channel|>commentary<|message|>{\"city\":\"Paris\"}<|call|>\
        <|start|>assistant<|channel|>analysis<|message|>Checked.<|end|>\
        <|start|>assistant<|channel|>final<|message|>Sunny.<|return|>";
    fn parse(parts: &[&str]) -> Vec<upstream::UnifiedEvent> {
        let mut parser =
            UnifiedStream::new("harmony", &tools(), UnifiedParserInit::default()).unwrap();
        assert!(parser.preserve_special_tokens());
        let mut events = Vec::new();
        for part in parts {
            events.extend(parser.advance(Some(part)).unwrap());
        }
        events.extend(parser.advance(None).unwrap());
        let mut output = UnifiedParserOutput::default();
        for event in events {
            match event {
                Event::Text(text) => output.push_text(text),
                Event::Reasoning(text) => output.push_reasoning(text),
                Event::ToolCall(call) => {
                    assert!(call.complete);
                    assert_eq!(call.tool_index, 0);
                    assert_eq!(parser.tool_call_id(0), call.id.as_deref());
                    assert!(call.id.is_some());
                    output.push_call(upstream::ToolCallDelta {
                        tool_index: call.tool_index,
                        name: call.name,
                        arguments: call.arguments,
                        complete: call.complete,
                    });
                }
            }
        }
        upstream::assemble(&output.events)
    }
    let expected = parse(&[text]);
    assert!(matches!(
        &expected[..],
        [
            upstream::UnifiedEvent::Reasoning { .. },
            upstream::UnifiedEvent::Text { .. },
            upstream::UnifiedEvent::ToolCall { .. },
            upstream::UnifiedEvent::Reasoning { .. },
            upstream::UnifiedEvent::Text { .. }
        ]
    ));
    let hf_text = text
        .replace("<|channel|>", "<|meta_sep|>")
        .replace("<|message|>", "<|im_sep|>")
        .replace("<|start|>", "<|im_start|>")
        .replace("<|end|>", "<|im_end|>")
        .replace("<|call|>", "<|ghissue|>")
        .replace("<|return|>", "<|fim_suffix|>");
    for text in [text, hf_text.as_str()] {
        for (at, _) in text.char_indices() {
            assert_eq!(parse(&[&text[..at], &text[at..]]), expected, "split {at}");
        }
        let pieces: Vec<_> = text
            .char_indices()
            .map(|(at, c)| &text[at..at + c.len_utf8()])
            .collect();
        assert_eq!(parse(&pieces), expected);
    }
}

#[test]
fn harmony_v1_initialization_and_unfinished_calls() {
    let prompt = upstream::encode_harmony(
        "<|start|>user<|message|>Hi<|end|><|start|>assistant<|channel|>analysis<|message|>",
    )
    .unwrap();
    let mut stream =
        UnifiedStream::new("gpt_oss", &tools(), UnifiedParserInit::native(&prompt)).unwrap();
    assert_eq!(
        stream.advance(Some("Thinking")).unwrap(),
        vec![Event::Reasoning("Thinking".into())]
    );
    stream.advance(Some("<|end|><|start|>assistant to=functions.weather<|channel|>commentary<|message|>{\"city\":\"Par")).unwrap();
    assert!(stream.advance(None).unwrap().is_empty());
    assert!(stream.advance(Some("late")).is_err());

    let init = request_init(vec![], "response", "native", None, "reject").unwrap();
    let mut stream = UnifiedStream::new("gpt-oss", &tools(), init).unwrap();
    assert_eq!(
        stream.advance(Some("Hello")).unwrap(),
        vec![Event::Text("Hello".into())]
    );
    let init = request_init(vec![], "none", "guided_json", None, "reject").unwrap();
    assert!(UnifiedStream::new("harmony", &tools(), init).is_err());
}
