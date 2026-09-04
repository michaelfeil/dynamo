//! b10: Baseten-owned regression tests for the Responses -> Chat Completions conversion, kept out
//! of the upstream file so fork rebases never conflict on test hunks.

use super::*;
use dynamo_protocols::types::ChatCompletionRequestMessage;

/// Codex multi-agent `agent_message` and Responses-Lite `additional_tools` input items parse
/// (they used to fail the whole request as unknown variants). At this layer they are SKIPPED
/// by the converter with a debug log — the agent turn and the tool declarations do not reach
/// the chat template. `additional_tools` gains a real consumer in the shared crate; this pins
/// the interim behaviour so it is a stated choice, not an accident.
#[test]
fn agent_message_and_additional_tools_items_parse_and_are_skipped() {
    let req: NvCreateResponse = serde_json::from_value(serde_json::json!({
        "model": "m",
        "input": [
            {"type": "agent_message", "author": "planner", "recipient": "coder",
             "content": [{"type": "input_text", "text": "plan"}]},
            {"type": "additional_tools", "role": "developer",
             "tools": [{"type": "function", "name": "lookup", "parameters": {"type": "object"}}]},
            {"role": "user", "content": "hi"}
        ]
    }))
    .unwrap();
    let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
    assert_eq!(
        chat_req.inner.messages.len(),
        1,
        "only the user turn survives"
    );
    assert!(matches!(
        chat_req.inner.messages[0],
        ChatCompletionRequestMessage::User(_)
    ));
    assert!(
        chat_req.inner.tools.is_none(),
        "additional_tools are not declared here"
    );
}
