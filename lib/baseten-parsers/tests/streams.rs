// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use baseten_parsers::{
    Event, REGISTERED_UNIFIED_FAMILIES, Tool, UnifiedParserInit, UnifiedParserOutput,
    UnifiedStream, request_init, upstream,
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
    for family in REGISTERED_UNIFIED_FAMILIES {
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
