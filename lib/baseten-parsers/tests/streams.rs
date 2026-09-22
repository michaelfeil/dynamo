// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use baseten_parsers::{
    REGISTERED_FAMILIES, REGISTERED_UNIFIED_FAMILIES, Tool, ToolCallStream, ToolParseResult,
    ToolParserInput, UnifiedParserInit, UnifiedParserOutput, UnifiedStream, upstream,
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

const GLM: &str = "Before <tool_call>weather<arg_key>city</arg_key><arg_value>café 🚀</arg_value></tool_call> after";

#[test]
fn every_upstream_tool_family_is_available() {
    for family in REGISTERED_FAMILIES {
        let mut wrapper = ToolCallStream::new(family, &tools()).unwrap();
        let mut reference = upstream::create_tool_parser_for_family(family, &tools()).unwrap();
        assert_eq!(
            wrapper.prefers_tokens(),
            reference.prefers_tokens(),
            "{family}"
        );
        assert_eq!(
            wrapper.preserve_special_tokens(),
            reference.preserve_special_tokens(),
            "{family}"
        );
        for chunk in ["", "hello", " café 🚀", ""] {
            assert_eq!(
                wrapper.step(ToolParserInput::Text(chunk)).unwrap(),
                reference.push(chunk).unwrap(),
                "{family}"
            );
        }
        assert_eq!(
            wrapper.finish().unwrap(),
            reference.finish().unwrap(),
            "{family}"
        );
        assert!(wrapper.step(ToolParserInput::Text("late")).is_err());
        assert!(wrapper.finish().is_err());
    }
}

#[test]
fn every_upstream_unified_family_is_available() {
    for family in REGISTERED_UNIFIED_FAMILIES {
        let mut wrapper =
            UnifiedStream::new(family, &tools(), UnifiedParserInit::default()).unwrap();
        let mut reference = upstream::create_unified_parser_for_family(family, &tools()).unwrap();
        reference
            .initialize_request(UnifiedParserInit::default())
            .unwrap();
        for chunk in ["", "hello", " café 🚀", ""] {
            let mut actual = UnifiedParserOutput::default();
            let mut expected = UnifiedParserOutput::default();
            wrapper.step(chunk, &mut actual).unwrap();
            reference.parse_into(chunk, &mut expected).unwrap();
            assert_eq!(actual, expected, "{family}");
        }
        assert_eq!(
            wrapper.finish().unwrap(),
            reference.finish().unwrap(),
            "{family}"
        );
        assert!(
            wrapper
                .step("late", &mut UnifiedParserOutput::default())
                .is_err()
        );
        assert!(wrapper.finish().is_err());
    }
}

#[test]
fn glm_all_unicode_split_positions_and_empty_chunks() {
    for split in GLM.char_indices().map(|(i, _)| i).chain([GLM.len()]) {
        let mut parser = ToolCallStream::new("glm47", &tools()).unwrap();
        let mut result = ToolParseResult::default();
        for chunk in [&GLM[..split], "", &GLM[split..], ""] {
            result.append(parser.step(ToolParserInput::Text(chunk)).unwrap());
        }
        result.append(parser.finish().unwrap());
        let result = result.coalesce_calls();
        assert_eq!(result.normal_text, "Before  after", "split {split}");
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("weather"));
        assert_eq!(result.calls[0].arguments, r#"{"city":"café 🚀"}"#);
        assert!(result.calls[0].complete);
    }
}

#[test]
fn independent_choices_and_truncated_glm_call() {
    let mut a = ToolCallStream::new("glm47", &tools()).unwrap();
    let mut b = ToolCallStream::new("glm47", &tools()).unwrap();
    a.step(ToolParserInput::Text(
        "<tool_call>weather<arg_key>city</arg_key>",
    ))
    .unwrap();
    let output = b.step(ToolParserInput::Text(GLM)).unwrap();
    assert_eq!(output.calls.len(), 1);
    assert_eq!(output.calls[0].tool_index, 0);
    let tail = a.finish().unwrap();
    assert!(tail.calls.is_empty());
    assert!(tail.normal_text.is_empty());
}

#[test]
fn invalid_family_and_unsupported_token_input_fail_loudly() {
    assert!(ToolCallStream::new("missing", &[]).is_err());
    assert!(UnifiedStream::new("missing", &[], UnifiedParserInit::default()).is_err());
    let mut parser = ToolCallStream::new("glm47", &tools()).unwrap();
    assert!(parser.step(ToolParserInput::Tokens(&[1])).is_err());
    // Rejected input does not advance the stream.
    assert_eq!(
        parser.step(ToolParserInput::Text(GLM)).unwrap().calls.len(),
        1
    );
}

#[test]
fn harmony_token_input_and_mixed_input_rejection() {
    let mut parser = ToolCallStream::new("harmony", &tools()).unwrap();
    let mut reference = upstream::create_tool_parser_for_family("harmony", &tools()).unwrap();
    let ids = upstream::encode_harmony(
        "<|channel|>commentary to=functions.weather <|constrain|>json<|message|>{\"city\":\"Paris\"}<|call|>",
    )
    .unwrap();
    let mut output = ToolParseResult::default();
    for chunk in ids.chunks(1) {
        let result = parser.step(ToolParserInput::Tokens(chunk)).unwrap();
        assert_eq!(result, reference.push_tokens(chunk).unwrap());
        output.append(result);
    }
    assert!(parser.step(ToolParserInput::Text("mixed")).is_err());
    let tail = parser.finish().unwrap();
    assert_eq!(tail, reference.finish().unwrap());
    output.append(tail);
    assert!(!output.calls.is_empty());
}

#[test]
fn unified_reasoning_content_order() {
    let mut parser = UnifiedStream::new("qwen3", &tools(), UnifiedParserInit::default()).unwrap();
    let mut output = UnifiedParserOutput::default();
    for chunk in ["<thi", "nk>reason</think>", "answer"] {
        parser.step(chunk, &mut output).unwrap();
    }
    output.append(&mut parser.finish().unwrap());
    assert_eq!(
        output.assembled(),
        vec![
            upstream::UnifiedEvent::Reasoning {
                text: "reason".into()
            },
            upstream::UnifiedEvent::Text {
                text: "answer".into()
            },
        ]
    );
}

#[test]
fn unified_guided_named_call() {
    let init = UnifiedParserInit {
        tool_output_mode: upstream::UnifiedToolOutputMode::GuidedJson {
            named_tool: Some("weather".into()),
        },
        ..Default::default()
    };
    let mut parser = UnifiedStream::new("qwen3", &tools(), init).unwrap();
    let mut output = UnifiedParserOutput::default();
    parser.step(r#"{"city":"Paris"}"#, &mut output).unwrap();
    output.append(&mut parser.finish().unwrap());
    assert_eq!(
        output.assembled(),
        vec![upstream::UnifiedEvent::ToolCall {
            name: "weather".into(),
            arguments: json!({"city":"Paris"}),
        }]
    );
}

#[test]
fn committed_events_survive_errors_and_failed_streams_close() {
    struct FailingParser;
    impl upstream::UnifiedParser for FailingParser {
        fn parse_into(&mut self, _: &str, output: &mut UnifiedParserOutput) -> anyhow::Result<()> {
            output.push_text("committed");
            anyhow::bail!("injected parser failure")
        }
        fn finish(&mut self) -> anyhow::Result<UnifiedParserOutput> {
            Ok(UnifiedParserOutput::default())
        }
    }
    let mut parser =
        UnifiedStream::from_parser(Box::new(FailingParser), UnifiedParserInit::default()).unwrap();
    let failure = parser.advance(Some("input")).unwrap_err();
    assert_eq!(
        failure.events,
        vec![baseten_parsers::Event::Text("committed".into())]
    );
    assert!(
        parser
            .step("retry", &mut UnifiedParserOutput::default())
            .unwrap_err()
            .to_string()
            .contains("closed")
    );
    assert!(parser.finish().is_err());
}

#[test]
fn alternative_tool_backend_and_normalized_ids() {
    struct OtherBackend;
    impl baseten_parsers::ToolParser for OtherBackend {
        fn create(_: &[Tool]) -> anyhow::Result<Box<dyn baseten_parsers::ToolParser>> {
            Ok(Box::new(Self))
        }
        fn push(&mut self, _: &str) -> anyhow::Result<ToolParseResult> {
            Ok(ToolParseResult {
                normal_text: "text".into(),
                calls: vec![upstream::ToolCallDelta {
                    tool_index: 0,
                    name: Some("weather".into()),
                    arguments: "{}".into(),
                    complete: true,
                }],
            })
        }
        fn tool_call_id(&self, _: usize) -> Option<&str> {
            Some("native-id")
        }
    }
    let mut parser = ToolCallStream::from_parser(Box::new(OtherBackend));
    let output = parser
        .advance(Some(ToolParserInput::Text("input")))
        .unwrap();
    assert_eq!(output.normal_text, "text");
    assert_eq!(output.calls[0].id.as_deref(), Some("native-id"));
    assert!(output.calls[0].complete);
    assert!(parser.advance(None).unwrap().calls.is_empty());
    assert!(parser.advance(None).is_err());
}

#[test]
fn configuration_validation_is_rust_owned() {
    use baseten_parsers::request_init;
    let config = request_init(
        vec![1],
        "reasoning",
        "guided_json",
        Some("weather".into()),
        "reject",
    )
    .unwrap();
    assert_eq!(config.prompt_token_ids, vec![1]);
    assert_eq!(
        config.starting_state,
        baseten_parsers::UnifiedParserStartingState::Reasoning
    );
    assert_eq!(
        config.tool_output_mode,
        baseten_parsers::UnifiedToolOutputMode::GuidedJson {
            named_tool: Some("weather".into())
        }
    );
    assert!(request_init(vec![], "invalid", "native", None, "reject").is_err());
    assert!(request_init(vec![], "none", "native", Some("weather".into()), "reject").is_err());
    assert!(request_init(vec![], "none", "native", None, "invalid").is_err());
}

#[test]
fn pinned_upstream_fixes_survive_the_tool_adapter() {
    let cases = [
        (
            "deepseek_v4",
            json!({"type": "string"}),
            "<think>checking</think><｜DSML｜tool_calls><｜DSML｜invoke name=\"inspect\"><｜DSML｜parameter name=\"value\" string=\"true\">  café\n</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>",
            "<think>checking</think>",
            json!({"value": "  café\n"}),
        ),
        (
            "glm47",
            json!({"allOf": [{"type": ["integer", "string"]}, {"type": "integer"}]}),
            "<tool_call>inspect<arg_key>value</arg_key><arg_value>42</arg_value></tool_call>",
            "",
            json!({"value": 42}),
        ),
    ];
    for (family, schema, input, normal_text, arguments) in cases {
        let tools = [Tool {
            name: "inspect".into(),
            description: None,
            parameters: json!({"type": "object", "properties": {"value": schema}}),
            strict: None,
        }];
        for character_chunks in [false, true] {
            let mut parser = ToolCallStream::new(family, &tools).unwrap();
            let mut result = ToolParseResult::default();
            if character_chunks {
                for c in input.chars() {
                    result.append(parser.step(ToolParserInput::Text(&c.to_string())).unwrap());
                }
            } else {
                result.append(parser.step(ToolParserInput::Text(input)).unwrap());
            }
            result.append(parser.finish().unwrap());
            let result = result.coalesce_calls();
            assert_eq!(result.normal_text, normal_text, "{family}");
            assert_eq!(result.calls.len(), 1, "{family}");
            assert_eq!(result.calls[0].name.as_deref(), Some("inspect"));
            assert!(result.calls[0].complete);
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&result.calls[0].arguments).unwrap(),
                arguments,
                "{family}"
            );
        }
    }
}
