use serde_json::json;

use super::*;
use crate::SemanticChunk;
use crate::model::{ServerToolCallStatus, ToolOutput};
use crate::model::{ToolCall, ToolInvocation};

fn tc(id: &str, name: &str, args: serde_json::Value) -> SemanticChunk {
    SemanticChunk::ToolCall(ToolCall {
        id: id.into(),
        name: name.into(),
        raw_args: args.to_string(),
        args,
    })
}

/// A successful server-tool invocation answering the call with id `call_id`.
fn invocation(call_id: &str, content: serde_json::Value) -> ToolInvocation {
    ToolInvocation {
        server_call: crate::test_utils::server_tool_call(ToolCall {
            id: call_id.into(),
            name: "baseten__stub__Read".into(),
            args: json!({}),
            raw_args: "{}".into(),
        }),
        output: ToolOutput {
            content,
            status: ServerToolCallStatus::Succeeded,
            billable: true,
            sku: None,
        },
    }
}

/// Serialize a message to JSON for shape assertions (the wire form fed back to the model).
fn as_json(message: &CcMessage) -> serde_json::Value {
    serde_json::to_value(message).unwrap()
}

#[test]
fn commits_assistant_turn_with_thinking_text_and_tool_calls() {
    let mut accumulator = MessageHistoryAccumulator::new(vec![]);
    accumulator.push(&SemanticChunk::ThinkingDelta("let me ".into()));
    accumulator.push(&SemanticChunk::ThinkingDelta("think".into()));
    accumulator.push(&SemanticChunk::TextDelta("answer".into()));
    accumulator.push(&tc("call-1", "Search", json!({"q": "x"})));
    accumulator.commit_assistant_message();
    assert_eq!(accumulator.messages().len(), 1);

    let message = as_json(&accumulator.messages()[0]);
    assert_eq!(message["role"], "assistant");
    assert_eq!(message["content"], "answer");
    assert_eq!(message["reasoning_content"], "let me think");
    assert_eq!(message["tool_calls"][0]["id"], "call-1");
    assert_eq!(message["tool_calls"][0]["type"], "function");
    assert_eq!(message["tool_calls"][0]["function"]["name"], "Search");
    // Args serialized as a JSON string (CC wire form).
    assert_eq!(
        message["tool_calls"][0]["function"]["arguments"],
        r#"{"q":"x"}"#
    );
}

#[test]
fn tool_only_turn_has_null_content() {
    let mut accumulator = MessageHistoryAccumulator::new(vec![]);
    accumulator.push(&tc("c1", "Read", json!({"p": "/x"})));
    accumulator.commit_assistant_message();
    let message = as_json(&accumulator.messages()[0]);
    // content omitted (skip_serializing_if None) — no `"content"` key, not `null`.
    assert!(
        message.get("content").is_none(),
        "tool-only turn omits content"
    );
    assert!(message.get("reasoning_content").is_none());
}

#[test]
fn empty_turn_commits_nothing() {
    let mut accumulator = MessageHistoryAccumulator::new(vec![]);
    accumulator.commit_assistant_message();
    assert_eq!(accumulator.messages().len(), 0);
}

#[test]
fn tool_results_append_as_tool_messages_in_order() {
    let mut accumulator = MessageHistoryAccumulator::new(vec![]);
    accumulator.push(&tc("c1", "Read", json!({})));
    accumulator.commit_assistant_message();
    accumulator.append_tool_results(&[
        invocation("c1", json!("file body")),
        invocation("c2", json!({"n": 1})),
    ]);
    let msgs = accumulator.messages();
    assert_eq!(msgs.len(), 3);
    let r0 = as_json(&msgs[1]);
    assert_eq!(r0["role"], "tool");
    assert_eq!(r0["tool_call_id"], "c1");
    assert_eq!(r0["content"], "file body");
    // Structured content rendered to a string (must match the history/echo edge).
    let r1 = as_json(&msgs[2]);
    assert_eq!(r1["content"], r#"{"n":1}"#);
}

/// An iteration's messages start at its assistant message, not at the results that answer it: a
/// client replaying a `tool` message with no matching `assistant.tool_calls[].id` gets a 400 on its
/// next request.
#[test]
fn iteration_messages_carry_the_whole_call_result_pair() {
    let mut accumulator = MessageHistoryAccumulator::new(vec![]);
    accumulator.push(&tc("c1", "Read", json!({})));
    accumulator.commit_assistant_message();
    accumulator.append_tool_results(&[invocation("c1", json!("file body"))]);

    let replayable: Vec<serde_json::Value> = accumulator
        .iteration_messages()
        .iter()
        .map(as_json)
        .collect();
    assert_eq!(replayable.len(), 2);
    assert_eq!(replayable[0]["role"], "assistant");
    assert_eq!(replayable[0]["tool_calls"][0]["id"], "c1");
    assert_eq!(replayable[1]["role"], "tool");
    assert_eq!(replayable[1]["tool_call_id"], "c1");

    // A client appends every frame's messages, so a leaked earlier pair duplicates in its prefix.
    accumulator.push(&tc("c2", "Read", json!({})));
    accumulator.commit_assistant_message();
    accumulator.append_tool_results(&[invocation("c2", json!("other body"))]);
    let second: Vec<serde_json::Value> = accumulator
        .iteration_messages()
        .iter()
        .map(as_json)
        .collect();
    assert_eq!(second.len(), 2);
    assert_eq!(second[0]["tool_calls"][0]["id"], "c2");
    assert_eq!(second[1]["tool_call_id"], "c2");
}

/// A tool call whose verbatim `raw_args` carry irregular whitespace a compact reserialize mangles.
fn tc_raw(id: &str, name: &str, raw_args: &str) -> SemanticChunk {
    SemanticChunk::ToolCall(ToolCall {
        id: id.into(),
        name: name.into(),
        args: serde_json::from_str(raw_args).expect("valid json args"),
        raw_args: raw_args.into(),
    })
}

/// History replay must be append-only and byte-stable: once a turn is committed, later turns'
/// prefix serializes byte-identically. This is the request-assembly side of the KV-cache-prefix
/// contract (the render oracle, tests/render_prefix_equivalence.rs, proves it survives templating).
#[test]
fn replay_is_append_only_and_byte_stable() {
    let serialize = |accumulator: &MessageHistoryAccumulator| {
        accumulator
            .messages()
            .iter()
            .map(|m| serde_json::to_string(m).expect("serialize"))
            .collect::<Vec<_>>()
    };

    let mut accumulator = MessageHistoryAccumulator::new(vec![
        serde_json::from_value(json!({"role": "system", "content": "sys"})).expect("system"),
        serde_json::from_value(json!({"role": "user", "content": "hi"})).expect("user"),
    ]);

    // Turn 1: thinking + a tool call with whitespace a compact reserialize would drop.
    let weird_args = r#"{"city": "São Paulo" ,"units":  "metric"}"#;
    accumulator.push(&SemanticChunk::ThinkingDelta("weighing options".into()));
    accumulator.push(&tc_raw("call-1", "get_weather", weird_args));
    accumulator.commit_assistant_message();
    let after_turn1 = serialize(&accumulator);

    accumulator.append_tool_results(&[invocation("call-1", json!("18C foggy"))]);

    // Turn 2: another thinking + tool call appended on top.
    accumulator.push(&SemanticChunk::ThinkingDelta("now the time".into()));
    accumulator.push(&tc_raw(
        "call-2",
        "get_time",
        r#"{"tz":"America/Sao_Paulo"}"#,
    ));
    accumulator.commit_assistant_message();
    let after_turn2 = serialize(&accumulator);

    assert!(after_turn2.len() > after_turn1.len(), "turn 2 only appends");
    assert_eq!(
        after_turn2[..after_turn1.len()],
        after_turn1[..],
        "committed prefix serializes byte-identically across turns"
    );

    // raw_args survives commit verbatim (guards the byte-exact cache prefix on replay).
    let assistant1 = as_json(&accumulator.messages()[2]);
    assert_eq!(
        assistant1["tool_calls"][0]["function"]["arguments"], weird_args,
        "model's verbatim arg bytes are replayed unchanged, not a compact reserialize"
    );
}

#[test]
fn loop_turns_append_after_the_client_prefix() {
    let client_msg: CcMessage =
        serde_json::from_value(json!({"role": "user", "content": "hi"})).unwrap();
    let mut accumulator = MessageHistoryAccumulator::new(vec![client_msg]);
    accumulator.push(&SemanticChunk::TextDelta("hello".into()));
    accumulator.commit_assistant_message();
    assert_eq!(accumulator.messages().len(), 2);
    assert_eq!(as_json(&accumulator.messages()[1])["content"], "hello");
}
