//! b10: Baseten-owned regression tests for the Responses -> Chat Completions conversion, kept out
//! of the upstream file so fork rebases never conflict on test hunks.

use super::*;
use dynamo_protocols::types::ChatCompletionRequestMessage;

/// Through the shared crate, Codex multi-agent `agent_message` and Responses-Lite
/// `additional_tools` input items are honored: the agent message becomes an assistant turn and
/// the declared tools reach the CC tool list (D1 only parsed and skipped them).
#[test]
fn agent_message_and_additional_tools_items_are_honored() {
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
    assert_eq!(chat_req.inner.messages.len(), 2, "agent turn + user turn");
    assert!(matches!(
        chat_req.inner.messages[0],
        ChatCompletionRequestMessage::Assistant(_)
    ));
    let tools = serde_json::to_value(chat_req.inner.tools.as_ref().expect("declared")).unwrap();
    assert_eq!(tools[0]["function"]["name"], "lookup");
}
