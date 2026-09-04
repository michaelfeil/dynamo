use dynamo_protocols::types::{ChatCompletionRequestToolMessageContent, FinishReason};
use serde_json::{Value, json};

use crate::coding_adapter::CodingAdapter;
use crate::framing::CompletedIteration;
use crate::history::tool_result_message;
use crate::model::{ServerToolCallStatus, ToolOutput};
use crate::model::{Termination, ToolInvocation};
use crate::test_utils::{
    FakeMessagesSearchAdapter, FakeResponsesSearchAdapter, drive, drive_with_coding_adapter,
    event_names, server_tool_call, tool_call,
};
use crate::{ClientProtocol, SemanticChunk};

/// One completed server-tool invocation: the model's call plus a successful text result. The name is
/// qualified as the registry produces it, which the display record splits the provider out of.
fn invocation(id: &str, tool: &str, content: serde_json::Value) -> ToolInvocation {
    ToolInvocation {
        server_call: server_tool_call(tool_call(
            id,
            &format!("baseten__parallel__{tool}"),
            json!({}),
        )),
        output: ToolOutput {
            content,
            status: ServerToolCallStatus::Succeeded,
            billable: true,
            sku: None,
        },
    }
}

// --- ChatCompletions --------------------------------------------------------

#[tokio::test]
async fn cc_text_and_thinking_deltas_role_once() {
    let frames = drive(ClientProtocol::ChatCompletions, |mut e| async move {
        e.push(&SemanticChunk::ThinkingDelta("hmm".into()))
            .await
            .unwrap();
        e.push(&SemanticChunk::TextDelta("hi".into()))
            .await
            .unwrap();
        e
    })
    .await;
    assert_eq!(frames.len(), 2);
    assert_eq!(
        frames[0].data["choices"][0]["delta"]["reasoning_content"],
        "hmm"
    );
    assert_eq!(frames[0].data["choices"][0]["delta"]["role"], "assistant"); // role on first only
    assert_eq!(frames[1].data["choices"][0]["delta"]["content"], "hi");
    assert!(frames[1].data["choices"][0]["delta"].get("role").is_none());
}

#[tokio::test]
async fn cc_tool_call_is_hidden() {
    let frames = drive(ClientProtocol::ChatCompletions, |mut e| async move {
        e.push(&SemanticChunk::ToolCall(tool_call(
            "c1",
            "ws",
            json!({"q": "x"}),
        )))
        .await
        .unwrap();
        e
    })
    .await;
    assert!(
        frames.is_empty(),
        "server tool calls are hidden on the CC default channel"
    );
}

#[tokio::test]
async fn cc_client_tool_calls_are_native() {
    let frames = drive(ClientProtocol::ChatCompletions, |mut e| async move {
        e.emit_client_tool_calls(&[tool_call("c1", "ws", json!({"q": "x"}))])
            .await
            .unwrap();
        e
    })
    .await;
    let call = &frames[0].data["choices"][0]["delta"]["tool_calls"][0];
    assert_eq!(call["index"], 0);
    assert_eq!(call["id"], "c1");
    assert_eq!(call["function"]["name"], "ws");
    assert_eq!(call["function"]["arguments"], r#"{"q":"x"}"#);
}

#[tokio::test]
async fn cc_client_tools_stream_inline_with_running_index() {
    // The loop emits each client tool as it completes (one call apiece); indices stay monotonic
    // across the separate emits so the SDK accumulates them into distinct tool_calls.
    let frames = drive(ClientProtocol::ChatCompletions, |mut e| async move {
        e.emit_client_tool_calls(&[tool_call("c1", "ws", json!({"q": "x"}))])
            .await
            .unwrap();
        e.emit_client_tool_calls(&[tool_call("c2", "calc", json!({"n": 2}))])
            .await
            .unwrap();
        e
    })
    .await;
    assert_eq!(
        frames[0].data["choices"][0]["delta"]["tool_calls"][0]["index"],
        0
    );
    assert_eq!(
        frames[0].data["choices"][0]["delta"]["tool_calls"][0]["id"],
        "c1"
    );
    assert_eq!(
        frames[1].data["choices"][0]["delta"]["tool_calls"][0]["index"],
        1
    );
    assert_eq!(
        frames[1].data["choices"][0]["delta"]["tool_calls"][0]["id"],
        "c2"
    );
}

#[tokio::test]
async fn cc_iteration_usage_is_side_channel_no_canonical_usage() {
    let usage = serde_json::from_value(
        json!({"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}),
    )
    .unwrap();
    let frames = drive(ClientProtocol::ChatCompletions, |mut e| async move {
        e.emit_iteration_usage(0, &usage).await.unwrap();
        e
    })
    .await;
    assert_eq!(frames[0].data["choices"].as_array().unwrap().len(), 0); // choices:[]
    assert!(
        frames[0].data.get("usage").is_none(),
        "must not touch canonical usage"
    );
    let iteration = &frames[0].data["baseten"]["iterations"][0];
    assert_eq!(iteration["index"], 0);
    assert_eq!(iteration["usage"]["completion_tokens"], 5);
}

/// The terminal chunk is finish + usage only: tool activity rode the `iterations` frames, so nothing
/// here duplicates it.
#[tokio::test]
async fn cc_finish_emits_usage_and_done_without_tool_history() {
    let usage = serde_json::from_value(
        json!({"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}),
    )
    .unwrap();
    let frames = drive(ClientProtocol::ChatCompletions, |mut e| async move {
        e.push(&SemanticChunk::TextDelta("done".into()))
            .await
            .unwrap();
        e.finish(Termination::Model(FinishReason::Stop), &usage)
            .await
            .unwrap();
        e
    })
    .await;
    assert_eq!(event_names(&frames), vec!["data", "data", "[DONE]"]);
    let fin = &frames[1].data;
    assert_eq!(fin["choices"][0]["finish_reason"], "stop");
    assert_eq!(fin["usage"]["completion_tokens"], 20);
    assert!(fin["choices"][0]["delta"].get("baseten").is_none());
}

/// The terminal chunk carries both, so a client reads the reason on the frame it reads
/// `finish_reason` from. A capped request can stream no content at all, and even then the opening
/// `role` delta OpenAI always sends rides its own chunk ahead of the terminal one.
#[tokio::test]
async fn cc_react_cap_finishes_as_length_with_termination_event() {
    let usage = serde_json::from_value(
        json!({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}),
    )
    .unwrap();
    let frames = drive(ClientProtocol::ChatCompletions, |mut e| async move {
        e.finish(Termination::ReactCapExhausted, &usage)
            .await
            .unwrap();
        e
    })
    .await;
    assert_eq!(frames[0].data["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(frames[0].data["choices"][0]["finish_reason"], Value::Null);
    let fin = &frames[1].data;
    assert_eq!(fin["choices"][0]["finish_reason"], "length");
    assert_eq!(
        fin["choices"][0]["delta"],
        json!({}),
        "the terminal delta stays empty, so `role` never arrives last"
    );
    assert_eq!(
        fin["baseten"]["request"]["termination_reason"],
        "max_react_iterations_reached"
    );
}

/// A CC client's only window onto the loop's hidden server-tool turns. One frame for the whole phase,
/// so a client appending `messages` can never hold a call without its result.
#[tokio::test]
async fn cc_server_tool_activity_rides_one_iteration_entry() {
    let frames = drive(ClientProtocol::ChatCompletions, |mut e| async move {
        e.emit_completed_iteration(&CompletedIteration {
            index: 3,
            invocations: &[invocation("c1", "ws", json!("RESULT"))],
            continuation_messages: &[tool_result_message(
                "c1".to_string(),
                ChatCompletionRequestToolMessageContent::Text("RESULT".to_string()),
            )],
        })
        .await
        .unwrap();
        e
    })
    .await;
    assert_eq!(frames.len(), 1);
    let iteration = &frames[0].data["baseten"]["iterations"][0];
    assert_eq!(iteration["index"], 3);
    assert_eq!(iteration["server_tool_calls"][0]["id"], "c1");
    assert_eq!(iteration["server_tool_calls"][0]["is_error"], false);
    // The result is carried once, as the replayable `tool` message.
    assert!(iteration["server_tool_calls"][0].get("content").is_none());
    assert_eq!(iteration["continuation_messages"][0]["role"], "tool");
    assert_eq!(iteration["continuation_messages"][0]["tool_call_id"], "c1");
    assert_eq!(iteration["continuation_messages"][0]["content"], "RESULT");
    // Display-only: never on the canonical channel the SDK accumulates into the message.
    assert_eq!(frames[0].data["choices"].as_array().unwrap().len(), 0);
}

/// A CC client can render the call while it runs: the dispatch frame carries the arguments, with
/// `is_error` explicitly `null` until the result lands.
#[tokio::test]
async fn cc_dispatched_server_tool_shows_arguments_before_the_result() {
    let frames = drive(ClientProtocol::ChatCompletions, |mut e| async move {
        e.emit_dispatched_server_tool(
            1,
            &server_tool_call(tool_call("c1", "baseten__parallel__ws", json!({"q": "x"}))),
        )
        .await
        .unwrap();
        e
    })
    .await;
    let call = &frames[0].data["baseten"]["iterations"][0]["server_tool_calls"][0];
    assert_eq!(call["id"], "c1");
    assert_eq!(call["provider"], "parallel");
    assert_eq!(call["arguments"], r#"{"q":"x"}"#);
    assert_eq!(call["is_error"], Value::Null, "still running");
}

/// Messages streams the same call as a native `tool_use` block, so a `baseten` copy would duplicate it.
#[tokio::test]
async fn messages_dispatched_server_tool_emits_no_extension() {
    let frames = drive(ClientProtocol::Messages, |mut e| async move {
        e.emit_dispatched_server_tool(
            1,
            &server_tool_call(tool_call("c1", "baseten__parallel__ws", json!({"q": "x"}))),
        )
        .await
        .unwrap();
        e
    })
    .await;
    assert!(frames.is_empty());
}

/// Messages says the same thing in native blocks, so repeating it under `baseten` would double every
/// result on the wire.
#[tokio::test]
async fn messages_omits_the_replayable_history() {
    let frames = drive(ClientProtocol::Messages, |mut e| async move {
        e.emit_completed_iteration(&CompletedIteration {
            index: 0,
            invocations: &[invocation("c1", "ws", json!("RESULT"))],
            continuation_messages: &[tool_result_message(
                "c1".to_string(),
                ChatCompletionRequestToolMessageContent::Text("RESULT".to_string()),
            )],
        })
        .await
        .unwrap();
        e
    })
    .await;
    assert!(frames.iter().all(|frame| frame.data["baseten"].is_null()));
}

// --- Anthropic Messages -----------------------------------------------------

#[tokio::test]
async fn messages_thinking_text_tool_block_ordering() {
    let usage = serde_json::from_value(
        json!({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}),
    )
    .unwrap();
    let frames = drive(ClientProtocol::Messages, |mut e| async move {
        e.push(&SemanticChunk::ThinkingDelta("plan".into()))
            .await
            .unwrap();
        e.push(&SemanticChunk::TextDelta("ok".into()))
            .await
            .unwrap();
        e.push(&SemanticChunk::ToolCall(tool_call(
            "c1",
            "Read",
            json!({"p": "/x"}),
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
            "content_block_start", // thinking idx0
            "content_block_delta", // thinking_delta
            "content_block_delta", // signature_delta (close thinking)
            "content_block_stop",  // thinking stop
            "content_block_start", // text idx1
            "content_block_delta", // text_delta
            "content_block_stop",  // text stop (before tool)
            "content_block_start", // tool_use idx2
            "content_block_delta", // input_json_delta (complete args)
            "content_block_stop",  // tool stop (inline)
            "message_delta",
            "message_stop",
        ],
    );
    // Block indices monotonic; tool_use carries the model id + complete args in one delta.
    assert_eq!(frames[8].data["index"], 2);
    assert_eq!(frames[8].data["content_block"]["type"], "tool_use");
    assert_eq!(frames[8].data["content_block"]["id"], "c1");
    assert_eq!(frames[9].data["delta"]["partial_json"], r#"{"p":"/x"}"#);
    assert_eq!(frames[11].data["delta"]["stop_reason"], "tool_use");
}

/// It must ride `message_delta`, the same frame as `stop_reason`: the SDK drops extras on
/// `message_stop`, so a frame of its own would be invisible.
#[tokio::test]
async fn messages_react_cap_finishes_as_pause_turn_with_termination_event() {
    let usage = serde_json::from_value(
        json!({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}),
    )
    .unwrap();
    let frames = drive(ClientProtocol::Messages, |mut e| async move {
        e.finish(Termination::ReactCapExhausted, &usage)
            .await
            .unwrap();
        e
    })
    .await;
    let delta = frames
        .iter()
        .find(|frame| frame.event.as_deref() == Some("message_delta"))
        .unwrap();
    assert_eq!(delta.data["delta"]["stop_reason"], "pause_turn");
    assert_eq!(
        delta.data["baseten"]["request"]["termination_reason"],
        "max_react_iterations_reached"
    );
}

/// A filtered turn is a refusal, not a clean stop: `end_turn` would read as the model choosing to
/// finish, hiding the filter from the caller.
#[tokio::test]
async fn messages_content_filter_finishes_as_refusal() {
    let usage = serde_json::from_value(
        json!({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}),
    )
    .unwrap();
    let frames = drive(ClientProtocol::Messages, |mut e| async move {
        e.finish(Termination::Model(FinishReason::ContentFilter), &usage)
            .await
            .unwrap();
        e
    })
    .await;
    let delta = frames
        .iter()
        .find(|frame| frame.event.as_deref() == Some("message_delta"))
        .unwrap();
    assert_eq!(delta.data["delta"]["stop_reason"], "refusal");
}

#[tokio::test]
async fn messages_tool_use_stop_is_last_for_block_no_orphan() {
    let frames = drive(ClientProtocol::Messages, |mut e| async move {
        e.push(&SemanticChunk::ToolCall(tool_call(
            "c1",
            "W",
            json!({"file": "x", "content": "y"}),
        )))
        .await
        .unwrap();
        e
    })
    .await;
    // start, one input_json_delta (whole args), stop — nothing after stop for the block.
    assert_eq!(
        event_names(&frames),
        vec![
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop"
        ]
    );
    assert_eq!(
        frames[2].data["delta"]["partial_json"],
        r#"{"file":"x","content":"y"}"#
    );
}

#[tokio::test]
async fn messages_tool_results_render_as_blocks() {
    let frames = drive(ClientProtocol::Messages, |mut e| async move {
        e.push(&SemanticChunk::ToolCall(tool_call("c1", "R", json!({}))))
            .await
            .unwrap();
        e.emit_completed_iteration(&CompletedIteration {
            index: 0,
            invocations: &[invocation("c1", "R", json!("RESULT"))],
            continuation_messages: &[],
        })
        .await
        .unwrap();
        e
    })
    .await;
    // message_start, tool_use(start/delta/stop), tool_result(start/stop)
    let result_start = frames
        .iter()
        .find(|f| f.data["content_block"]["type"] == json!("tool_result"))
        .unwrap();
    assert_eq!(result_start.data["content_block"]["tool_use_id"], "c1");
    assert_eq!(result_start.data["content_block"]["content"], "RESULT");
}

#[tokio::test]
async fn messages_second_iteration_opens_new_monotonic_blocks() {
    let usage = serde_json::from_value(
        json!({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}),
    )
    .unwrap();
    let frames = drive(ClientProtocol::Messages, |mut e| async move {
        // Iteration 1: tool call (idx0), results (idx1).
        e.push(&SemanticChunk::ToolCall(tool_call("c1", "R", json!({}))))
            .await
            .unwrap();
        e.emit_completed_iteration(&CompletedIteration {
            index: 0,
            invocations: &[invocation("c1", "R", json!("R1"))],
            continuation_messages: &[],
        })
        .await
        .unwrap();
        // Iteration 2: text answer (idx2).
        e.push(&SemanticChunk::TextDelta("final".into()))
            .await
            .unwrap();
        e.finish(Termination::Model(FinishReason::Stop), &usage)
            .await
            .unwrap();
        e
    })
    .await;
    // The final text block must open at a fresh index (2), not collide with iteration 1's blocks.
    let text_start = frames
        .iter()
        .find(|f| f.data["content_block"]["type"] == json!("text"))
        .unwrap();
    assert_eq!(
        text_start.data["index"], 2,
        "second turn's block index stays monotonic"
    );
}

/// An iteration's scope is held until its index closes (here: at finish), so a client accumulating
/// `iterations` sees exactly one entry per index — the buffered body's shape.
#[tokio::test]
async fn messages_iteration_scope_arrives_once_and_merged() {
    let usage = serde_json::from_value(
        json!({"prompt_tokens": 7, "completion_tokens": 9, "total_tokens": 16}),
    )
    .unwrap();
    let frames = drive(ClientProtocol::Messages, |mut e| async move {
        e.push(&SemanticChunk::TextDelta("hi".into()))
            .await
            .unwrap(); // starts the message
        e.emit_iteration_debug_msg(0, "steered").await.unwrap();
        e.push(&SemanticChunk::TextDelta(" there".into()))
            .await
            .unwrap(); // a preserved delta must NOT flush the still-open iteration
        e.emit_iteration_usage(0, &usage).await.unwrap();
        e.finish(Termination::Model(FinishReason::Stop), &usage)
            .await
            .unwrap();
        e
    })
    .await;
    let carrying: Vec<_> = frames
        .iter()
        .filter(|f| f.data["baseten"]["iterations"].is_array())
        .collect();
    assert_eq!(carrying.len(), 1, "one frame carries the iteration entry");
    let iterations = carrying[0].data["baseten"]["iterations"]
        .as_array()
        .unwrap();
    assert_eq!(iterations.len(), 1, "index 0 appears exactly once");
    assert_eq!(iterations[0]["index"], 0);
    assert_eq!(iterations[0]["usage"]["output_tokens"], 9);
    assert_eq!(iterations[0]["debug_msg"][0], "steered");
}

/// The open scope flushes when the next iteration starts staging, not only at finish.
#[tokio::test]
async fn messages_iteration_scope_flushes_on_index_advance() {
    let usage = serde_json::from_value(
        json!({"prompt_tokens": 7, "completion_tokens": 9, "total_tokens": 16}),
    )
    .unwrap();
    let frames = drive(ClientProtocol::Messages, |mut e| async move {
        e.push(&SemanticChunk::TextDelta("hi".into()))
            .await
            .unwrap();
        e.emit_iteration_usage(0, &usage).await.unwrap();
        e.emit_iteration_debug_msg(1, "steered").await.unwrap();
        e.push(&SemanticChunk::TextDelta(" there".into()))
            .await
            .unwrap();
        e.finish(Termination::Model(FinishReason::Stop), &usage)
            .await
            .unwrap();
        e
    })
    .await;
    let carrying: Vec<_> = frames
        .iter()
        .filter(|f| f.data["baseten"]["iterations"].is_array())
        .collect();
    assert_eq!(
        carrying.len(),
        2,
        "iteration 0 flushes mid-stream, 1 at finish"
    );
    assert_eq!(carrying[0].data["baseten"]["iterations"][0]["index"], 0);
    assert_eq!(carrying[1].data["baseten"]["iterations"][0]["index"], 1);
    assert_eq!(
        carrying[1].data["baseten"]["iterations"][0]["debug_msg"][0],
        "steered"
    );
}

#[tokio::test]
async fn messages_finish_without_content_still_emits_envelope() {
    let usage = serde_json::from_value(
        json!({"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1}),
    )
    .unwrap();
    let frames = drive(ClientProtocol::Messages, |mut e| async move {
        e.finish(Termination::Model(FinishReason::Stop), &usage)
            .await
            .unwrap();
        e
    })
    .await;
    assert_eq!(
        event_names(&frames),
        vec!["message_start", "message_delta", "message_stop"],
        "empty model output still gets a well-formed envelope"
    );
}

// --- Responses ---------------------------------------------------------------

/// The `openai` SDK's stream accumulator rebuilds `response.output_text.delta`/`.done` and
/// `response.function_call_arguments.delta` from their own typed fields, dropping any top-level
/// sibling — so pending usage must skip those and ride the next event that survives intact.
#[tokio::test]
async fn responses_iteration_usage_skips_dropped_events_and_rides_the_next_preserved_one() {
    let usage = serde_json::from_value(
        json!({"prompt_tokens": 7, "completion_tokens": 9, "total_tokens": 16}),
    )
    .unwrap();
    let frames = drive(ClientProtocol::Responses, |mut e| async move {
        e.push(&SemanticChunk::TextDelta("hi".into()))
            .await
            .unwrap(); // opens the item
        e.emit_iteration_usage(0, &usage).await.unwrap(); // buffered: the open event is dropped
        e.push(&SemanticChunk::TextDelta(" there".into()))
            .await
            .unwrap(); // another dropped delta: still buffered
        e.finish(Termination::Model(FinishReason::Stop), &usage)
            .await
            .unwrap(); // closes the item: content_part.done is the first surviving event
        e
    })
    .await;
    let dropped = ["response.output_text.delta", "response.output_text.done"];
    for frame in &frames {
        if dropped.contains(&frame.data["type"].as_str().unwrap()) {
            assert!(
                frame.data.get("baseten").is_none(),
                "usage must never ride a dropped event: {}",
                frame.data
            );
        }
    }
    // The iteration is held open until finish, so its one merged entry rides the terminal
    // `response.completed` — whose `baseten` sits inside `response`.
    let with_usage = frames
        .iter()
        .find(|f| f.data["response"]["baseten"]["iterations"].is_array())
        .expect("the iteration entry rides the terminal frame");
    assert_eq!(with_usage.data["type"], "response.completed");
    assert_eq!(
        with_usage.data["response"]["baseten"]["iterations"][0]["usage"]["output_tokens"],
        9
    );
}

/// A server tool's dispatch (`function_call` placeholder) and resolution (`mcp_call`, call+result
/// in one item) both surface as `output_item.added`/`.done` — no `baseten` needed to see either.
#[tokio::test]
async fn responses_without_coding_adapter_preserves_mcp_call_behavior() {
    let call = tool_call("c1", "baseten__parallel__web_search", json!({"q": "x"}));
    let frames = drive(ClientProtocol::Responses, |mut e| async move {
        e.push(&SemanticChunk::ToolCall(call.clone()))
            .await
            .unwrap();
        e.emit_completed_iteration(&CompletedIteration {
            index: 0,
            invocations: &[ToolInvocation {
                server_call: server_tool_call(call),
                output: ToolOutput {
                    content: json!("RESULT"),
                    status: ServerToolCallStatus::Succeeded,
                    billable: true,
                    sku: None,
                },
            }],
            continuation_messages: &[],
        })
        .await
        .unwrap();
        e
    })
    .await;
    let added = frames
        .iter()
        .find(|f| f.data["type"] == "response.output_item.added")
        .unwrap();
    assert_eq!(added.data["item"]["type"], "function_call");
    let resolved = frames
        .iter()
        .rev()
        .find(|f| {
            f.data["type"] == "response.output_item.done" && f.data["item"]["type"] == "mcp_call"
        })
        .expect("the placeholder resolves to an mcp_call item");
    assert_eq!(resolved.data["item"]["id"], "c1");
    assert_eq!(resolved.data["item"]["output"], "RESULT");
}

#[tokio::test]
async fn responses_failed_server_tool_resolves_to_failed_mcp_call() {
    let call = tool_call("c1", "baseten__parallel__web_search", json!({"q": "x"}));
    let frames = drive(ClientProtocol::Responses, |mut e| async move {
        e.push(&SemanticChunk::ToolCall(call.clone()))
            .await
            .unwrap();
        e.emit_completed_iteration(&CompletedIteration {
            index: 0,
            invocations: &[ToolInvocation {
                server_call: server_tool_call(call),
                output: ToolOutput {
                    content: json!("tool execution failed: connect timeout"),
                    status: ServerToolCallStatus::Failed,
                    billable: true,
                    sku: None,
                },
            }],
            continuation_messages: &[],
        })
        .await
        .unwrap();
        e
    })
    .await;
    assert!(
        frames
            .iter()
            .any(|f| f.data["type"] == "response.mcp_call.failed")
    );
    let resolved = frames
        .iter()
        .rev()
        .find(|f| {
            f.data["type"] == "response.output_item.done" && f.data["item"]["type"] == "mcp_call"
        })
        .unwrap();
    assert_eq!(resolved.data["item"]["status"], "failed");
    assert_eq!(
        resolved.data["item"]["error"],
        "tool execution failed: connect timeout"
    );
}

#[tokio::test]
async fn responses_finish_without_content_still_emits_envelope() {
    let usage = serde_json::from_value(
        json!({"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1}),
    )
    .unwrap();
    let frames = drive(ClientProtocol::Responses, |mut e| async move {
        e.finish(Termination::Model(FinishReason::Stop), &usage)
            .await
            .unwrap();
        e
    })
    .await;
    let types: Vec<&str> = frames
        .iter()
        .map(|f| f.data["type"].as_str().unwrap())
        .collect();
    assert_eq!(
        types,
        vec![
            "response.created",
            "response.in_progress",
            "response.completed"
        ],
        "empty model output still gets a well-formed envelope"
    );
}

fn codex_adapter(tool_name: &'static str) -> Box<dyn CodingAdapter> {
    Box::new(FakeResponsesSearchAdapter { tool_name })
}

fn claude_code_adapter(tool_name: &'static str) -> Box<dyn CodingAdapter> {
    Box::new(FakeMessagesSearchAdapter { tool_name })
}

#[tokio::test]
async fn messages_coding_adapter_streams_native_web_search_with_stable_indices() {
    let name = "baseten__provider__search";
    let call = tool_call("call_1", name, json!({"search_queries": ["rust"]}));
    let frames = drive_with_coding_adapter(
        ClientProtocol::Messages,
        Some(claude_code_adapter(name)),
        |mut e| async move {
            e.push(&SemanticChunk::ToolCall(call.clone()))
                .await
                .unwrap();
            e.emit_completed_iteration(&CompletedIteration {
                index: 0,
                invocations: &[ToolInvocation {
                    server_call: server_tool_call(call),
                    output: ToolOutput {
                        // An all-text MCP result reaches the framing as one joined string, so the
                        // provider's JSON arrives wrapped. Rendering must still find the citations.
                        content: json!(
                            r#"{"payload":{"results":[{"title":"Rust","url":"https://example.com/rust","text":"hidden"}]}}"#
                        ),
                        status: ServerToolCallStatus::Succeeded,
                billable: true,
                        sku: None,
                    },
                }],
                continuation_messages: &[],
            })
            .await
            .unwrap();
            e
        },
    )
    .await;
    let call_start = frames
        .iter()
        .find(|frame| frame.data["content_block"]["type"] == "server_tool_use")
        .unwrap();
    let call_index = call_start.data["index"].as_u64().unwrap();
    assert_eq!(call_start.data["content_block"]["id"], "srvtoolu_call_1");
    assert_eq!(call_start.data["content_block"]["name"], "web_search");
    assert_eq!(call_start.data["content_block"]["input"], json!({}));
    let call_delta = frames
        .iter()
        .find(|frame| {
            frame.event.as_deref() == Some("content_block_delta")
                && frame.data["index"] == call_index
        })
        .unwrap();
    assert_eq!(call_delta.data["delta"]["type"], "input_json_delta");
    assert_eq!(
        call_delta.data["delta"]["partial_json"],
        "{\"query\":\"rust\"}"
    );
    assert!(frames.iter().any(|frame| {
        frame.event.as_deref() == Some("content_block_stop") && frame.data["index"] == call_index
    }));
    let result_start = frames
        .iter()
        .find(|frame| frame.data["content_block"]["type"] == "web_search_tool_result")
        .unwrap();
    let result_index = result_start.data["index"].as_u64().unwrap();
    assert_eq!(
        result_start.data["content_block"]["tool_use_id"],
        "srvtoolu_call_1"
    );
    assert_eq!(
        result_start.data["content_block"]["content"],
        json!([{
            "type": "web_search_result",
            "title": "Rust",
            "url": "https://example.com/rust"
        }])
    );
    assert!(frames.iter().any(|frame| {
        frame.event.as_deref() == Some("content_block_stop") && frame.data["index"] == result_index
    }));
}

/// The streamed failure path: a provider error renders the error result block rather than an empty
/// citation list, which is the only thing distinguishing "search failed" from "search found nothing".
#[tokio::test]
async fn messages_coding_adapter_streams_the_error_result_block_for_a_failed_call() {
    let name = "baseten__provider__search";
    let call = tool_call("call_1", name, json!({"search_queries": ["rust"]}));
    let frames = drive_with_coding_adapter(
        ClientProtocol::Messages,
        Some(claude_code_adapter(name)),
        |mut e| async move {
            e.push(&SemanticChunk::ToolCall(call.clone()))
                .await
                .unwrap();
            e.emit_completed_iteration(&CompletedIteration {
                index: 0,
                invocations: &[ToolInvocation {
                    server_call: server_tool_call(call),
                    output: ToolOutput {
                        content: json!({"error": "upstream refused"}),
                        status: ServerToolCallStatus::Failed,
                        billable: true,
                        sku: None,
                    },
                }],
                continuation_messages: &[],
            })
            .await
            .unwrap();
            e
        },
    )
    .await;
    let result_start = frames
        .iter()
        .find(|frame| frame.data["content_block"]["type"] == "web_search_tool_result")
        .expect("a failed server tool still renders its Anthropic result block");
    assert_eq!(
        result_start.data["content_block"]["tool_use_id"],
        "srvtoolu_call_1"
    );
    assert_eq!(
        result_start.data["content_block"]["content"],
        json!({"type": "web_search_tool_result_error", "error_code": "unavailable"})
    );
}

#[tokio::test]
async fn responses_coding_adapter_emits_web_search_call_lifecycle_with_stable_id() {
    let name = "baseten__parallel__web_search";
    let call = tool_call(
        "ws_1",
        name,
        json!({
            "objective": "find Rust release",
            "search_queries": ["latest Rust stable", "Rust release notes"]
        }),
    );
    let adapter = codex_adapter(name);
    let frames = drive_with_coding_adapter(
        ClientProtocol::Responses,
        Some(adapter),
        |mut e| async move {
            e.push(&SemanticChunk::ToolCall(call.clone()))
                .await
                .unwrap();
            e.emit_completed_iteration(&CompletedIteration {
                index: 0,
                invocations: &[ToolInvocation {
                    server_call: server_tool_call(call),
                    output: ToolOutput {
                        content: json!({"results": ["hidden"]}),
                        status: ServerToolCallStatus::Succeeded,
                        billable: true,
                        sku: None,
                    },
                }],
                continuation_messages: &[],
            })
            .await
            .unwrap();
            e.push(&SemanticChunk::TextDelta("Rust is current.".into()))
                .await
                .unwrap();
            e.finish(
                Termination::Model(FinishReason::Stop),
                &dynamo_protocols::types::CompletionUsage::default(),
            )
            .await
            .unwrap();
            e
        },
    )
    .await;
    let types = frames
        .iter()
        .map(|frame| frame.data["type"].as_str().unwrap())
        .collect::<Vec<_>>();
    let added_index = types
        .iter()
        .position(|event| *event == "response.output_item.added")
        .unwrap();
    let in_progress_index = types
        .iter()
        .position(|event| *event == "response.web_search_call.in_progress")
        .unwrap();
    let searching_index = types
        .iter()
        .position(|event| *event == "response.web_search_call.searching")
        .unwrap();
    let completed_index = types
        .iter()
        .position(|event| *event == "response.web_search_call.completed")
        .unwrap();
    let done_index = types
        .iter()
        .position(|event| *event == "response.output_item.done")
        .unwrap();
    assert!(
        added_index < in_progress_index
            && in_progress_index < searching_index
            && searching_index < completed_index
            && completed_index < done_index
    );
    let added = &frames[added_index].data["item"];
    let done = &frames[done_index].data["item"];
    let terminal = &frames.last().unwrap().data["response"]["output"][0];
    for item in [added, done, terminal] {
        assert_eq!(item["id"], "ws_1");
        assert_eq!(item["type"], "web_search_call");
        assert_eq!(
            item["action"]["queries"],
            json!(["latest Rust stable", "Rust release notes"])
        );
        assert_eq!(item["action"]["query"], "latest Rust stable");
        assert!(item["action"].get("sources").is_none());
        assert!(item.get("results").is_none());
        assert!(item.get("annotations").is_none());
    }
    assert_eq!(added["status"], "in_progress");
    assert_eq!(done["status"], "completed");
    assert_eq!(terminal["status"], "completed");
    assert!(frames.iter().all(|frame| {
        frame.data["item"]["type"] != "function_call" || frame.data["item"]["name"] != "web_search"
    }));
}

#[tokio::test]
async fn responses_coding_adapter_records_an_unclosed_rendered_call_in_the_terminal_envelope() {
    let name = "baseten__parallel__web_search";
    let call = tool_call(
        "ws_1",
        name,
        json!({"search_queries": ["latest Rust stable"]}),
    );
    let adapter = codex_adapter(name);
    let frames = drive_with_coding_adapter(
        ClientProtocol::Responses,
        Some(adapter),
        |mut e| async move {
            e.push(&SemanticChunk::ToolCall(call.clone()))
                .await
                .unwrap();
            e.finish(
                Termination::Model(FinishReason::Stop),
                &dynamo_protocols::types::CompletionUsage::default(),
            )
            .await
            .unwrap();
            e
        },
    )
    .await;
    let added = frames
        .iter()
        .find(|frame| frame.data["type"] == "response.output_item.added")
        .expect("the client was told the item exists");
    assert_eq!(added.data["item"]["id"], "ws_1");
    let terminal = &frames.last().unwrap().data["response"]["output"][0];
    assert_eq!(
        terminal["type"], "web_search_call",
        "an item announced with output_item.added must survive into response.output, got {terminal}"
    );
    assert_eq!(terminal["id"], "ws_1");
    assert_eq!(terminal["status"], "in_progress", "it never completed");
    assert_eq!(
        terminal["action"]["queries"],
        json!(["latest Rust stable"]),
        "and it carries the shape the client was shown, not the typed slot's singular query"
    );
}
#[tokio::test]
async fn responses_coding_adapter_marks_a_failed_search_failed() {
    let name = "baseten__parallel__web_search";
    let call = tool_call("ws_err", name, json!({"search_queries": ["down"]}));
    let frames = drive_with_coding_adapter(
        ClientProtocol::Responses,
        Some(codex_adapter(name)),
        |mut e| async move {
            e.push(&SemanticChunk::ToolCall(call.clone()))
                .await
                .unwrap();
            e.emit_completed_iteration(&CompletedIteration {
                index: 0,
                invocations: &[ToolInvocation {
                    server_call: server_tool_call(call),
                    output: ToolOutput {
                        content: json!("provider unavailable"),
                        status: ServerToolCallStatus::Failed,
                        billable: true,
                        sku: None,
                    },
                }],
                continuation_messages: &[],
            })
            .await
            .unwrap();
            e.finish(
                Termination::Model(FinishReason::Stop),
                &dynamo_protocols::types::CompletionUsage::default(),
            )
            .await
            .unwrap();
            e
        },
    )
    .await;
    let done = frames
        .iter()
        .find(|frame| frame.data["type"] == "response.output_item.done")
        .expect("the failed call still closes its item");
    assert_eq!(done.data["item"]["type"], "web_search_call");
    assert_eq!(done.data["item"]["status"], "failed");
    let terminal = &frames.last().unwrap().data["response"]["output"][0];
    assert_eq!(terminal["status"], "failed", "and the envelope agrees");
}

// Dropped with the Codex adapter: `responses_coding_adapter_renders_fetch_as_open_page` tested
// the adapter's own `open_page` action rendering, which lives with tool-bank's execution machinery
// and has no framing-seam expression beyond what the tests above already cover.

#[tokio::test]
async fn responses_react_cap_names_the_real_reason_directly() {
    let usage = serde_json::from_value(
        json!({"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1}),
    )
    .unwrap();
    let frames = drive(ClientProtocol::Responses, |mut e| async move {
        e.finish(Termination::ReactCapExhausted, &usage)
            .await
            .unwrap();
        e
    })
    .await;
    let terminal = frames.last().unwrap();
    assert_eq!(terminal.data["type"], "response.completed");
    assert!(terminal.data["response"]["incomplete_details"].is_null());
    assert_eq!(
        terminal.data["response"]["baseten"]["request"]["termination_reason"],
        "max_react_iterations_reached"
    );
}
