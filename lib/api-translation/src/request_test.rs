use super::*;
use crate::coding_adapter::{CodingAdapter, RenderedToolCall, ToolCallToRender};
use crate::hooks::{ClaimedServerTool, DropServerTools, IngressLimits, IngressRewrite};
use crate::loss::{Loss, LossKind, Losses};

/// tool-bank's selection namespace, as its hooks recognise it; the crate itself knows no prefix.
const TEST_SERVER_TOOL_PREFIX: &str = "baseten__";
const REACT_ITERATIONS_MAX: std::num::NonZeroU32 = std::num::NonZeroU32::new(20).unwrap();
const REACT_ITERATIONS_MIN: std::num::NonZeroU32 = std::num::NonZeroU32::new(2).unwrap();
use crate::{CcMessage, CcRequest};

/// Test hooks with the shape tool-bank's real impl takes: a fixed registry of claimable
/// `baseten__*` tools, tool-bank-style rejections for everything else server-tool-shaped, and
/// configurable cap defaults.
struct StubHooks {
    limits: IngressLimits,
    offered: Vec<&'static str>,
}

impl StubHooks {
    fn empty() -> Self {
        Self {
            limits: IngressLimits {
                // Under the ceiling, so a request can be seen raising it too.
                default_react_iterations: NonZeroU32::new(5).unwrap(),
                max_react_iterations: REACT_ITERATIONS_MAX,
                server_tool_iterations_floor: REACT_ITERATIONS_MIN,
                max_tool_calls_per_iteration: NonZeroU32::new(10).unwrap(),
            },
            offered: Vec::new(),
        }
    }

    fn with_tools(offered: &[&'static str]) -> Self {
        Self {
            offered: offered.to_vec(),
            ..Self::empty()
        }
    }

    fn claim(&self, qualified: &str) -> ClaimedServerTool {
        ClaimedServerTool {
            name: qualified.to_string(),
            tool: serde_json::from_value(json!({
                "type": "function",
                "function": {"name": qualified, "description": "", "parameters": {"type": "object"}},
            }))
            .unwrap(),
        }
    }
}

impl IngressHooks for StubHooks {
    fn limits(&self) -> IngressLimits {
        self.limits
    }

    // tool-bank's stance for the tests that predate the seam: refuse what cannot be honored.
    fn rejects_unsupported_messages_features(&self) -> bool {
        true
    }

    fn on_server_tool(&mut self, _protocol: ClientProtocol, entry: &Value) -> ToolDisposition {
        let kind = entry
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if !kind.starts_with(TEST_SERVER_TOOL_PREFIX) {
            // tool-bank's stance: only its own reserved namespace is claimable; a vendor-hosted
            // tool aimed at this endpoint is refused rather than dropped.
            return ToolDisposition::Reject(RequestRejection::malformed(format!(
                "unsupported tool type {kind:?}: vendor-hosted server tools are not supported"
            )));
        }
        // Strict selection parse: nothing but `type` is honored on a selection entry.
        if let Some(extra) = entry
            .as_object()
            .and_then(|object| object.keys().find(|key| *key != "type"))
        {
            return ToolDisposition::Reject(RequestRejection::malformed(format!(
                "invalid server tool selection: unknown field `{extra}`"
            )));
        }
        if !self.offered.contains(&kind.as_str()) {
            return ToolDisposition::Reject(RequestRejection::malformed(format!(
                "unknown server tool `{kind}`"
            )));
        }
        ToolDisposition::Claim(self.claim(&kind))
    }
}

fn test_hooks() -> StubHooks {
    StubHooks::empty()
}

/// Old-signature shim: most tests predate the hooks seam and use the default stub hooks.
fn adapt(
    body: &[u8],
    protocol: ClientProtocol,
    headers: &HeaderMap,
) -> Result<AdaptedIngress, RequestRejection> {
    adapt_request(body, protocol, headers, &mut test_hooks())
}

/// Most tests only assert on the adapted request; the adapter has its own tests.
fn request_only(
    body: &[u8],
    protocol: ClientProtocol,
    headers: &HeaderMap,
) -> Result<AdaptedRequest, RequestRejection> {
    adapt(body, protocol, headers).map(|i| i.request)
}

fn anthropic_message(v: Value) -> AnthropicMessage {
    serde_json::from_value(v).unwrap()
}

/// The CC messages one Anthropic message translates to, in the wire form fed back to the model.
fn translated(message: &AnthropicMessage) -> Vec<Value> {
    let mut cc_messages = Vec::new();
    translate_message(message, 0, &mut cc_messages, false, &mut Losses::default()).unwrap();
    cc_messages
        .iter()
        .map(|m| serde_json::to_value(m).unwrap())
        .collect()
}

#[test]
fn translates_tool_use_assistant_to_cc_tool_calls() {
    let msg = anthropic_message(json!({"role": "assistant", "content": [
        {"type": "thinking", "thinking": "let me", "signature": ""},
        {"type": "text", "text": "calling"},
        {"type": "tool_use", "id": "c1", "name": "ws", "input": {"q": "x"}},
    ]}));
    let out = translated(&msg);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0]["content"], json!("calling"));
    // Fork Segments semantics: reasoning interleaved with tool calls keeps its position
    // (segment i precedes call i, plus one trailing segment).
    assert_eq!(out[0]["reasoning_content"], json!(["let me", ""]));
    assert_eq!(out[0]["tool_calls"][0]["id"], json!("c1"));
    assert_eq!(
        out[0]["tool_calls"][0]["function"]["arguments"],
        json!(r#"{"q":"x"}"#)
    );
}

#[test]
fn splits_echoed_assistant_turn_into_cc_sequence() {
    let msg = anthropic_message(json!({"role": "assistant", "content": [
        {"type": "thinking", "thinking": "let me search", "signature": ""},
        {"type": "tool_use", "id": "c1", "name": "ws", "input": {"q": "x"}},
        {"type": "tool_result", "tool_use_id": "c1", "content": "RESULT"},
        {"type": "text", "text": "Final answer."},
    ]}));
    let out = translated(&msg);
    assert_eq!(out.len(), 3);
    assert_eq!(out[0]["tool_calls"][0]["id"], json!("c1"));
    assert_eq!(out[0]["content"], Value::Null);
    assert_eq!(
        out[1],
        json!({"role": "tool", "tool_call_id": "c1", "content": "RESULT"})
    );
    assert_eq!(
        out[2],
        json!({"role": "assistant", "content": "Final answer."})
    );
}

#[test]
fn translates_tool_result_user_to_cc_tool_message() {
    let msg = anthropic_message(json!({"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "c1", "content": "result text"}
    ]}));
    let out = translated(&msg);
    assert_eq!(
        out,
        vec![json!({"role": "tool", "tool_call_id": "c1", "content": "result text"})]
    );
}

#[test]
fn tool_use_null_input_becomes_empty_object() {
    // Explicit null input must not serialize to the literal "null" arg string (breaks cache replay).
    let msg = anthropic_message(json!({"role": "assistant", "content": [
        {"type": "tool_use", "id": "c1", "name": "ws", "input": null},
    ]}));
    let out = translated(&msg);
    assert_eq!(
        out[0]["tool_calls"][0]["function"]["arguments"],
        json!("{}")
    );
}

#[test]
fn user_text_before_tool_result_preserves_order() {
    // Text preceding a tool_result must stay before it (byte-identical CC prefix replay).
    let msg = anthropic_message(json!({"role": "user", "content": [
        {"type": "text", "text": "here is the context"},
        {"type": "tool_result", "tool_use_id": "c1", "content": "R"},
    ]}));
    let out = translated(&msg);
    assert_eq!(
        out[0],
        json!({"role": "user", "content": "here is the context"})
    );
    assert_eq!(
        out[1],
        json!({"role": "tool", "tool_call_id": "c1", "content": "R"})
    );
}

/// A `tool_result` with no `tool_use_id` is refused at parse: TB would otherwise have to invent the
/// `tool_call_id` its CC translation requires.
#[test]
fn tool_result_without_tool_use_id_fails_loud() {
    let msg = json!({"role": "user", "content": [{"type": "tool_result", "content": "R"}]});
    assert!(serde_json::from_value::<AnthropicMessage>(msg).is_err());
}

/// Both are a 400, but only `Unsupported` is a gap in our adapter — a misclassification blames the
/// wrong side for the refusal.
#[test]
fn rejection_separates_our_gaps_from_a_malformed_body() {
    let reject = |body: Value, protocol| {
        adapt(
            &serde_json::to_vec(&body).unwrap(),
            protocol,
            &HeaderMap::new(),
        )
        .expect_err("must be refused")
    };
    assert!(matches!(
        reject(
            json!({"model": "m", "baseten": {"nope": {}}}),
            ClientProtocol::ChatCompletions
        ),
        RequestRejection::Malformed(_)
    ));
    // CC allows `n`, and a thinking block is a valid Messages block (just not in a user
    // message): TB simply cannot carry either.
    assert!(matches!(
        reject(
            json!({"model": "m", "n": 2, "messages": []}),
            ClientProtocol::ChatCompletions
        ),
        RequestRejection::Unsupported(_)
    ));
    assert!(matches!(
        reject(
            json!({"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": [
                {"type": "thinking", "thinking": "hm", "signature": ""},
            ]}]}),
            ClientProtocol::Messages
        ),
        RequestRejection::Unsupported(_)
    ));
}

#[test]
fn tool_result_text_blocks_stay_flat_text() {
    let msg = anthropic_message(json!({"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": [
            {"type": "text", "text": "line 1"},
            {"type": "text", "text": "line 2"},
        ]},
    ]}));
    let out = translated(&msg);
    assert_eq!(
        out,
        vec![json!({"role": "tool", "tool_call_id": "t1", "content": "line 1line 2"})]
    );
}

#[test]
fn tool_result_image_block_becomes_cc_image_part() {
    let msg = anthropic_message(json!({"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": [
            {"type": "text", "text": "screenshot taken"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}},
        ]},
    ]}));
    let out = translated(&msg);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0]["role"], json!("tool"));
    assert_eq!(out[0]["tool_call_id"], json!("t1"));
    let content = &out[0]["content"];
    assert_eq!(content.as_array().unwrap().len(), 2);
    assert_eq!(
        content[0],
        json!({"type": "text", "text": "screenshot taken"})
    );
    assert_eq!(content[1]["type"], json!("image_url"));
    assert_eq!(
        content[1]["image_url"]["url"],
        json!("data:image/png;base64,aGVsbG8=")
    );
}

#[test]
fn tool_result_non_base64_image_source_is_refused() {
    let msg = anthropic_message(json!({"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "c1", "content": [
            {"type": "image", "source": {"type": "url", "media_type": "image/png", "data": "https://x/y.png"}},
        ]},
    ]}));
    let mut cc_messages = Vec::new();
    let refusal = translate_message(&msg, 0, &mut cc_messages, false, &mut Losses::default())
        .err()
        .unwrap();
    assert!(
        matches!(refusal, RequestRejection::Unsupported(_)),
        "refusal: {refusal}"
    );
    assert!(
        refusal.detail().contains("image source type"),
        "refusal: {refusal}"
    );
}

/// Fork parity: an unknown/unparseable tool_result block is carried as coerced JSON text, never a
/// request error — agent clients replay these verbatim, so refusing would fail real sessions.
#[test]
fn tool_result_unknown_block_is_coerced_to_json_text() {
    let msg = anthropic_message(json!({"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "c1", "content": [{"type": "document", "title": "x"}]},
    ]}));
    let out = translated(&msg);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0]["role"], json!("tool"));
    let text = out[0]["content"].as_str().unwrap();
    assert!(text.contains("document"), "coerced text: {text}");
    assert!(text.contains("\"x\""), "coerced text: {text}");
}

#[test]
fn malformed_content_fails_loud() {
    let msg = json!({"role": "user", "content": {"unexpected": "object"}});
    assert!(serde_json::from_value::<AnthropicMessage>(msg).is_err());
}

#[test]
fn build_next_request_forces_stream_replaces_messages_preserves_template() {
    // Template with tools + a vendor field (rides `unmodeled`) + client tool_choice.
    let template: CcRequest = serde_json::from_value(json!({
        "model": "m",
        "messages": [{"role": "user", "content": "old"}],
        "tools": [{"type": "function", "function": {"name": "f"}}],
        "tool_choice": "auto",
        "temperature": 0.5,
        "chat_template_kwargs": {"enable_thinking": true},
    }))
    .unwrap();
    let new_msgs: Vec<CcMessage> = vec![
        serde_json::from_value(json!({"role": "user", "content": "new"})).unwrap(),
        serde_json::from_value(json!({"role": "assistant", "content": "reply"})).unwrap(),
    ];
    let req = build_next_request(&template, &new_msgs, true);
    let v = serde_json::to_value(&req).unwrap();
    assert_eq!(v["stream"], true);
    assert_eq!(v["stream_options"]["include_usage"], true);
    assert_eq!(v["messages"].as_array().unwrap().len(), 2);
    assert_eq!(v["messages"][0]["content"], "new");
    // Template fields preserved, including tool_choice (untouched by design) and vendor passthrough.
    assert_eq!(v["tool_choice"], "auto");
    assert_eq!(v["temperature"], 0.5);
    assert_eq!(v["chat_template_kwargs"]["enable_thinking"], true);
}

/// Passthrough-proxy contract: arbitrary vendor fields (not just `chat_template_kwargs`), including
/// nested ones, ride the `unmodeled` catch-all through `build_next_request` with values intact.
/// dynamo drops these on egress; a proxy must re-emit them verbatim.
#[test]
fn build_next_request_preserves_arbitrary_vendor_fields() {
    let template: CcRequest = serde_json::from_value(json!({
        "model": "m",
        "messages": [{"role": "user", "content": "old"}],
        "guided_json": {"type": "object", "properties": {"a": {"type": "string"}}},
        "guided_regex": "[0-9]+",
        "chat_template_kwargs": {"enable_thinking": true, "thinking_budget": 2048},
        "some_future_field": {"nested": [1, 2, {"k": "v"}]},
    }))
    .unwrap();
    let new_msgs: Vec<CcMessage> =
        vec![serde_json::from_value(json!({"role": "user", "content": "new"})).unwrap()];
    let v = serde_json::to_value(build_next_request(&template, &new_msgs, true)).unwrap();
    assert_eq!(v["guided_json"]["properties"]["a"]["type"], "string");
    assert_eq!(v["guided_regex"], "[0-9]+");
    assert_eq!(v["chat_template_kwargs"]["thinking_budget"], 2048);
    assert_eq!(v["some_future_field"]["nested"][2]["k"], "v");
}

#[test]
fn messages_tool_choice_translated_and_unmodeled_forwarded() {
    let body = serde_json::to_vec(&json!({
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "any"},
        // `top_k` has a typed home on the fork's CC wrapper, so it rides through.
        "top_k": 5,
        // `metadata.user_id` is lifted onto the modeled `user` field (sticky-routing key), and
        // the now-empty `metadata` is dropped rather than serializing a second driftable copy.
        "metadata": {"user_id": "u1"},
    }))
    .unwrap();
    let adapted = request_only(&body, ClientProtocol::Messages, &HeaderMap::new()).unwrap();
    let cc = serde_json::to_value(&adapted.request).unwrap();
    assert_eq!(cc["tool_choice"], "required");
    assert_eq!(cc["top_k"], 5);
    assert_eq!(cc["user"], "u1");
    assert_eq!(cc["metadata"], Value::Null);
}

/// Anthropic-only top-level fields must not reach the CC body: the deployment's strict reparse
/// 400s on them (Claude Code's `context_management` did exactly this live). They are dropped —
/// standard-dynamo behavior — while the fork's own extension keys and `baseten` still work.
#[test]
fn messages_anthropic_only_top_level_fields_are_dropped_not_forwarded() {
    for (key, value) in [
        ("service_tier", json!("standard_only")),
        ("cache_control", json!({"type": "ephemeral"})),
        (
            "context_management",
            json!({"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}),
        ),
        ("metadata", json!({"user_id": "u1", "session": "s9"})),
    ] {
        let adapted = adapt_messages_body(json!({key: value}))
            .unwrap_or_else(|e| panic!("{key} must adapt, got {e:?}"));
        let cc = serde_json::to_value(&adapted.request).unwrap();
        assert!(cc.get(key).is_none(), "{key} leaked onto the CC body: {cc}");
    }
    // The fork's CC extension surface still rides through, and `baseten` is still honored.
    let adapted = adapt_messages_body(json!({
        "top_k": 7,
        "priority": {"level": 1},
        "chat_template_kwargs": {"enable_thinking": true},
        "nvext": {"ignore_eos": true},
        "baseten": {"tool_settings": {"max_react_iterations": 3}},
    }))
    .unwrap();
    assert_eq!(adapted.max_react_iterations.get(), 3);
    let cc = serde_json::to_value(&adapted.request).unwrap();
    assert_eq!(cc["top_k"], 7);
    assert_eq!(cc["priority"]["level"], 1);
    assert_eq!(cc["chat_template_kwargs"]["enable_thinking"], true);
    assert_eq!(cc["nvext"]["ignore_eos"], true);
    assert!(cc.get("baseten").is_none());
}

#[test]
fn tool_choice_shape_translation() {
    let translated = |shape: Value| json!(translate_tool_choice(&shape).unwrap().0);
    assert_eq!(translated(json!({"type": "auto"})), json!("auto"));
    assert_eq!(translated(json!({"type": "any"})), json!("required"));
    assert_eq!(translated(json!({"type": "none"})), json!("none"));
    assert_eq!(
        translated(json!({"type": "tool", "name": "ws"})),
        json!({"type": "function", "function": {"name": "ws"}})
    );
    assert!(translate_tool_choice(&json!({"type": "tool"})).is_err());
    assert!(translate_tool_choice(&json!({"type": "bogus"})).is_err());
}

#[test]
fn build_next_request_applies_tool_choice_only_on_the_first_iteration() {
    let template: CcRequest = serde_json::from_value(json!({
        "model": "m",
        "messages": [{"role": "user", "content": "old"}],
        "tool_choice": "required",
    }))
    .unwrap();
    let msgs: Vec<CcMessage> =
        vec![serde_json::from_value(json!({"role": "user", "content": "new"})).unwrap()];
    let first = serde_json::to_value(build_next_request(&template, &msgs, true)).unwrap();
    assert_eq!(first["tool_choice"], "required");
    let later = serde_json::to_value(build_next_request(&template, &msgs, false)).unwrap();
    assert!(
        later.get("tool_choice").is_none() || later["tool_choice"].is_null(),
        "tool_choice must not force a call after the first iteration"
    );
}

/// Nothing but the appended message changes: `tools` stays attached and no `tool_choice` is invented.
#[test]
fn build_next_request_appends_the_steering_message_last() {
    let template: CcRequest = serde_json::from_value(json!({
        "model": "m",
        "messages": [{"role": "user", "content": "old"}],
        "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}],
    }))
    .unwrap();
    let msgs: Vec<CcMessage> =
        vec![serde_json::from_value(json!({"role": "user", "content": "new"})).unwrap()];

    let last_iteration = serde_json::to_value(with_steering_prompt(
        build_next_request(&template, &msgs, false),
        "answer now, do not call search",
    ))
    .unwrap();
    let messages = last_iteration["messages"].as_array().unwrap();
    assert_eq!(messages.len(), msgs.len() + 1);
    assert_eq!(
        messages[messages.len() - 1],
        json!({"role": "user", "content": "answer now, do not call search"})
    );
    assert!(
        last_iteration.get("tool_choice").is_none(),
        "{last_iteration}"
    );
    assert_eq!(
        last_iteration["tools"],
        serde_json::to_value(&template.tools).unwrap()
    );
}

#[test]
fn baseten_tool_settings_lowers_caps_never_raises() {
    let request_with = |baseten: Option<Value>| {
        let mut body = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
        if let (Some(fields), Some(baseten)) = (body.as_object_mut(), baseten) {
            fields.insert("baseten".into(), baseten);
        }
        adapt(
            &serde_json::to_vec(&body).unwrap(),
            ClientProtocol::ChatCompletions,
            &HeaderMap::new(),
        )
        .map(|ingress| ingress.request)
    };
    let default = request_with(None).unwrap();
    assert_eq!(default.max_react_iterations.get(), 5);
    assert_eq!(default.max_tool_calls_per_iteration.get(), 10);

    let lowered = request_with(Some(json!({
        "tool_settings": {"max_react_iterations": 3, "max_tool_calls_per_iteration": 4}
    })))
    .unwrap();
    assert_eq!(lowered.max_react_iterations.get(), 3);
    assert_eq!(lowered.max_tool_calls_per_iteration.get(), 4);

    // A request may raise as well as lower it: the config value is a default, not a cap.
    let raised =
        request_with(Some(json!({"tool_settings": {"max_react_iterations": 12}}))).unwrap();
    assert_eq!(raised.max_react_iterations.get(), 12);
    assert_eq!(
        request_with(Some(
            json!({"tool_settings": {"max_react_iterations": REACT_ITERATIONS_MAX.get()}})
        ))
        .unwrap()
        .max_react_iterations
        .get(),
        REACT_ITERATIONS_MAX.get()
    );

    // A 400, not a clamp: a truncated answer is indistinguishable from a complete one.
    let refused = request_with(Some(
        json!({"tool_settings": {"max_react_iterations": REACT_ITERATIONS_MAX.get() + 1}}),
    ))
    .expect_err("above the ceiling must be refused, not clamped");
    assert!(
        matches!(refused, RequestRejection::Malformed(ref detail) if detail.contains("max_react_iterations")),
        "{refused:?}"
    );

    // Which setting fixed the cap, so `react_cap_exhausted` can name the caller or the deployment.
    assert_eq!(default.react_cap_source, ReactCapSource::ServerDefault);
    assert_eq!(lowered.react_cap_source, ReactCapSource::Request);
    assert_eq!(raised.react_cap_source, ReactCapSource::Request);
    // At the ceiling the caller has nothing left to raise, whichever setting put it there.
    assert_eq!(
        request_with(Some(
            json!({"tool_settings": {"max_react_iterations": REACT_ITERATIONS_MAX.get()}})
        ))
        .unwrap()
        .react_cap_source,
        ReactCapSource::ServiceCeiling
    );

    // A 400, not a clamp: a truncated answer is indistinguishable from a complete one.
    let refused = request_with(Some(
        json!({"tool_settings": {"max_tool_calls_per_iteration": 11}}),
    ))
    .expect_err("above the ceiling must be refused, not clamped");
    assert!(
        matches!(refused, RequestRejection::Malformed(ref detail) if detail.contains("max_tool_calls_per_iteration")),
        "{refused}"
    );
    // A zero cap is meaningless: refused, not silently turned into some value TB chose.
    assert!(request_with(Some(json!({"tool_settings": {"max_react_iterations": 0}}))).is_err());
}

/// The floor is [`REACT_ITERATIONS_MIN`] only for a request that can dispatch a server tool, whose
/// single iteration could dispatch a call but never answer from its result — the call and the answer
/// are separate model calls. A request that cannot dispatch one is complete in a single iteration.
#[test]
fn single_iteration_refused_only_when_a_server_tool_can_be_called() {
    let adapt_with = |tools: Value, tool_choice: Option<&str>, iterations: u32| {
        let mut body = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": tools,
            "baseten": {"tool_settings": {"max_react_iterations": iterations}},
        });
        if let (Some(fields), Some(tool_choice)) = (body.as_object_mut(), tool_choice) {
            fields.insert("tool_choice".into(), json!(tool_choice));
        }
        adapt_request(
            &serde_json::to_vec(&body).unwrap(),
            ClientProtocol::ChatCompletions,
            &HeaderMap::new(),
            &mut StubHooks::with_tools(&["baseten__web__search"]),
        )
        .map(|ingress| ingress.request)
    };
    let server_tool = json!([{"type": "baseten__web__search"}]);
    let client_tool =
        json!([{"type": "function", "function": {"name": "ls", "parameters": {"type": "object"}}}]);

    let refused = adapt_with(server_tool.clone(), None, 1)
        .expect_err("a single iteration cannot answer from a server-tool result");
    assert!(
        refused.detail().contains("max_react_iterations"),
        "{refused}"
    );
    assert_eq!(
        adapt_with(server_tool.clone(), None, REACT_ITERATIONS_MIN.get())
            .unwrap()
            .max_react_iterations,
        REACT_ITERATIONS_MIN
    );

    // `none` forbids calling, so the selection cannot be reached.
    assert_eq!(
        adapt_with(server_tool, Some("none"), 1)
            .unwrap()
            .max_react_iterations
            .get(),
        1
    );
    assert_eq!(
        adapt_with(client_tool, None, 1)
            .unwrap()
            .max_react_iterations
            .get(),
        1
    );
    assert_eq!(
        adapt_with(json!([]), None, 1)
            .unwrap()
            .max_react_iterations
            .get(),
        1
    );
}

/// Unknown members of the `baseten` namespace are warned about and dropped, never a 400 — fork
/// deployments historically carried extension members here harmlessly (CC-pivot leniency). The
/// `tool_settings` object itself stays strict: a misspelled cap knob is a 400, not a silently
/// ignored setting. `config` on a tool selection is refused by the same mechanism.
#[test]
fn baseten_extension_unknown_members_dropped_but_tool_settings_strict() {
    // Unknown sibling of tool_settings: accepted, dropped with a warn, defaults apply.
    let adapted = adapt(
        &serde_json::to_vec(&json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "baseten": {"tool_setting": {}, "future_knob": 7},
        }))
        .unwrap(),
        ClientProtocol::ChatCompletions,
        &HeaderMap::new(),
    )
    .expect("unknown `baseten` members must not refuse the request");
    assert!(adapted.request.server_tool_claims.is_empty());

    let refuse = |body: Value| {
        adapt(
            &serde_json::to_vec(&body).unwrap(),
            ClientProtocol::ChatCompletions,
            &HeaderMap::new(),
        )
        .expect_err("must be refused")
        .detail()
        .to_string()
    };
    // Inside tool_settings, strictness holds.
    assert!(
        refuse(json!({"model": "m", "baseten": {"tool_settings": {"max_iterations": 2}}}))
            .contains("baseten")
    );
    assert!(
        refuse(json!({
            "model": "m",
            "tools": [{"type": "baseten__web__search", "config": {"depth": 2}}],
        }))
        .contains("server tool selection")
    );
}

/// The `baseten` extension is TB's own: it must never ride through to the model backend.
#[test]
fn baseten_extension_is_stripped_from_the_forwarded_body() {
    for protocol in [ClientProtocol::ChatCompletions, ClientProtocol::Messages] {
        let body = serde_json::to_vec(&json!({
            "model": "m",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "baseten": {"tool_settings": {"max_react_iterations": 2}},
        }))
        .unwrap();
        let adapted = request_only(&body, protocol, &HeaderMap::new()).unwrap();
        let cc = serde_json::to_value(&adapted.request).unwrap();
        assert!(cc.get("baseten").is_none(), "leaked on {protocol:?}");
    }
}

#[test]
fn duplicate_server_tool_selection_is_refused() {
    let names = vec![
        "baseten__web__search".to_string(),
        "baseten__web__search".to_string(),
    ];
    assert!(
        ServerToolClaims::new(names)
            .unwrap_err()
            .contains("duplicate")
    );
}

/// A user block with no CC translation: standard dynamo skips it with a warning (the previous
/// converter's behavior — the rest of the message still reaches the model); tool-bank's strict
/// hooks refuse it, since a silently discarded block would have the model answer about input the
/// caller never sent. (`image` now translates, so an assistant-only block stands in here.)
#[test]
fn untranslatable_user_block_is_skipped_by_default_and_refused_under_strict_hooks() {
    let msg = anthropic_message(json!({"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "thinking", "thinking": "hm", "signature": ""},
    ]}));
    let mut lenient = Vec::new();
    translate_message(&msg, 0, &mut lenient, false, &mut Losses::default()).unwrap();
    assert_eq!(lenient.len(), 1);
    assert_eq!(
        serde_json::to_value(&lenient[0]).unwrap()["content"],
        "what is this"
    );
    let err = translate_message(&msg, 0, &mut Vec::new(), true, &mut Losses::default())
        .expect_err("strict: refused");
    assert!(err.detail().contains("thinking"), "{err}");
}

/// Fork separator ruling: adjacent text blocks — in `system`, a user message, and an assistant
/// message — join with "\n", the shape the deployed converter always produced. Claude Code sends
/// a two-block system array on every request, so this is the prompt-cache prefix.
#[test]
fn adjacent_text_blocks_join_with_newline() {
    let adapted = adapt_messages_body(json!({
        "system": [{"type": "text", "text": "You are Claude."}, {"type": "text", "text": "Be brief."}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
            {"role": "user", "content": "go"}
        ]
    }))
    .unwrap();
    let out = serde_json::to_value(&adapted.request.messages).unwrap();
    assert_eq!(out[0]["content"], "You are Claude.\nBe brief.");
    assert_eq!(out[1]["content"], "one\ntwo");
    assert_eq!(out[2]["content"], "a\nb");
}

/// `mcp_servers` / `container` (Anthropic-hosted features): dropped with a warning by default,
/// refused under tool-bank's strict hooks.
#[test]
fn mcp_servers_dropped_by_default_refused_under_strict_hooks() {
    let body = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}],
        "mcp_servers": [{"type": "url", "url": "https://x", "name": "x"}]});
    let ok = adapt_request(
        &serde_json::to_vec(&body).unwrap(),
        ClientProtocol::Messages,
        &HeaderMap::new(),
        &mut crate::hooks::DropServerTools,
    );
    assert!(ok.is_ok(), "{ok:?}");
    struct Strict;
    impl IngressHooks for Strict {
        fn rejects_unsupported_messages_features(&self) -> bool {
            true
        }
    }
    let err = adapt_request(
        &serde_json::to_vec(&body).unwrap(),
        ClientProtocol::Messages,
        &HeaderMap::new(),
        &mut Strict,
    )
    .expect_err("strict hooks refuse mcp_servers");
    assert!(err.detail().contains("mcp_servers"), "{err:?}");
}

/// A user `image` block translates to a CC multimodal image part (fork parity — Claude Code
/// pastes screenshots into user content).
#[test]
fn user_image_block_becomes_cc_image_part() {
    let msg = anthropic_message(json!({"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBOR"}},
    ]}));
    let out = translated(&msg);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0]["role"], json!("user"));
    let parts = out[0]["content"].as_array().unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0], json!({"type": "text", "text": "what is this"}));
    assert_eq!(parts[1]["type"], json!("image_url"));
    assert_eq!(
        parts[1]["image_url"]["url"],
        json!("data:image/png;base64,iVBOR")
    );
}

/// An Anthropic-native server tool (versioned `type`) only Anthropic can execute: refused, so the
/// model is never offered a tool nobody will run. A plain/`custom` client tool passes through.
#[test]
fn anthropic_native_server_tool_entry_is_refused() {
    let native = client_function_tool(json!({"type": "web_search_20250305", "name": "web_search"}))
        .expect_err("native server tool must be refused");
    // Malformed, not Unsupported: nothing for TB to close, the caller aimed it at the wrong vendor.
    assert!(
        matches!(native, RequestRejection::Malformed(ref detail) if detail.contains("web_search_20250305")),
        "{native:?}"
    );

    for entry in [
        json!({"name": "get_weather", "input_schema": {"type": "object"}}),
        json!({"type": "custom", "name": "get_weather", "input_schema": {"type": "object"}}),
    ] {
        let cc_tools = json!(client_function_tool(entry).unwrap());
        assert_eq!(cc_tools[0]["type"], "function");
        assert_eq!(cc_tools[0]["function"]["name"], "get_weather");
        assert_eq!(cc_tools[0]["function"]["parameters"]["type"], "object");
    }
}

/// A user `document` block (Claude Code attaches PDFs this way) is dropped with a warning, not
/// refused: the model cannot consume it on this stack, and the text beside it must still arrive.
#[test]
fn user_document_block_is_dropped_not_refused() {
    let msg = anthropic_message(json!({"role": "user", "content": [
        {"type": "document", "title": "spec.pdf",
         "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="}},
        {"type": "text", "text": "summarize this"},
    ]}));
    let out = translated(&msg);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0]["role"], "user");
    assert_eq!(out[0]["content"], "summarize this");

    let adapted = adapt_messages_body(json!({"messages": [{"role": "user", "content": [
        {"type": "document", "title": "spec.pdf", "source": {"type": "text", "media_type": "text/plain", "data": "x"}},
        {"type": "text", "text": "hi"},
    ]}]}))
    .expect("document blocks must not 400 the request");
    assert_eq!(adapted.request.messages.len(), 1);
}

/// A `system` role inside `messages[]` is a client compatibility shape; it must keep its role rather
/// than collapse into `user`.
#[test]
fn system_role_in_messages_keeps_its_role() {
    let msg = anthropic_message(json!({"role": "system", "content": "be terse"}));
    let out = translated(&msg);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0]["role"], "system");
    assert_eq!(out[0]["content"], "be terse");
}

#[test]
fn tool_choice_named_without_name_is_refused() {
    assert!(
        translate_tool_choice(&json!({"type": "tool"}))
            .unwrap_err()
            .detail()
            .contains("requires `name`")
    );
}

/// `disable_parallel_tool_use: true` -> `parallel_tool_calls: false`; the two defaults ("parallel
/// allowed") emit no CC field, so the model backend's own default stays in charge.
#[test]
fn tool_choice_disable_parallel_maps_to_cc_parallel_tool_calls() {
    let (_, parallel_tool_calls) =
        translate_tool_choice(&json!({"type": "auto", "disable_parallel_tool_use": true})).unwrap();
    assert_eq!(parallel_tool_calls, Some(false));
    let (_, parallel_tool_calls) = translate_tool_choice(&json!({"type": "auto"})).unwrap();
    assert_eq!(parallel_tool_calls, None);
    let adapted = adapt_messages_body(
        json!({"tool_choice": {"type": "auto", "disable_parallel_tool_use": true}}),
    )
    .unwrap();
    assert_eq!(adapted.request.parallel_tool_calls, Some(false));
}

// --- OpenAI Responses ---------------------------------------------------------

fn responses_input(items: Value) -> Vec<Value> {
    let items: Vec<InputItem> = serde_json::from_value(items).unwrap();
    let mut cc_messages = Vec::new();
    translate_input_items(&items, &mut cc_messages, &mut Losses::default()).unwrap();
    cc_messages
        .iter()
        .map(|m| serde_json::to_value(m).unwrap())
        .collect()
}

fn adapt_messages_body(extra: Value) -> Result<AdaptedRequest, RequestRejection> {
    // 4096: large enough that a thinking budget fixture (2048) stays under max_tokens now that
    // the Anthropic budget bounds are enforced.
    let mut body = json!({
        "model": "m",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "hi"}],
    });
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    adapt(
        &serde_json::to_vec(&body).unwrap(),
        ClientProtocol::Messages,
        &HeaderMap::new(),
    )
    .map(|ingress| ingress.request)
}

#[test]
fn messages_thinking_enabled_with_budget_translates_to_kwargs() {
    let adapted =
        adapt_messages_body(json!({"thinking": {"type": "enabled", "budget_tokens": 2048}}))
            .unwrap();
    let cc = serde_json::to_value(&adapted.request).unwrap();
    assert_eq!(cc["chat_template_kwargs"]["enable_thinking"], true);
    assert_eq!(cc["chat_template_kwargs"]["thinking_budget"], 2048);
}

/// `adaptive` is the only thinking mode on Anthropic 4.7+ models, so refusing it locks out current
/// Anthropic-SDK clients. It carries no budget; depth comes from `output_config.effort`.
#[test]
fn messages_thinking_adaptive_enables_thinking_without_budget() {
    let adapted = adapt_messages_body(json!({"thinking": {"type": "adaptive"}})).unwrap();
    let cc = serde_json::to_value(&adapted.request).unwrap();
    assert_eq!(cc["chat_template_kwargs"]["enable_thinking"], true);
    assert!(cc["chat_template_kwargs"].get("thinking_budget").is_none());
}

#[test]
fn messages_thinking_disabled_translates_to_kwargs() {
    let adapted = adapt_messages_body(json!({"thinking": {"type": "disabled"}})).unwrap();
    let cc = serde_json::to_value(&adapted.request).unwrap();
    assert_eq!(cc["chat_template_kwargs"]["enable_thinking"], false);
}

#[test]
fn messages_thinking_unknown_type_is_refused() {
    let err = adapt_messages_body(json!({"thinking": {"type": "turbo"}}))
        .err()
        .unwrap();
    assert!(err.detail().contains("unsupported thinking type"), "{err}");
}

#[test]
fn messages_thinking_budget_outside_enabled_is_refused() {
    for mode in ["adaptive", "disabled"] {
        let err = adapt_messages_body(json!({"thinking": {"type": mode, "budget_tokens": 2048}}))
            .err()
            .unwrap();
        assert!(err.detail().contains("budget_tokens"), "{mode}: {err}");
    }
}

/// The client's own `chat_template_kwargs` merges per key with the thinking translation — the
/// client wins on collision, but its presence must not erase the rest of the translation.
#[test]
fn messages_thinking_merges_per_key_with_client_kwargs() {
    let adapted = adapt_messages_body(json!({
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "chat_template_kwargs": {"enable_thinking": false, "custom_flag": 1},
    }))
    .unwrap();
    let cc = serde_json::to_value(&adapted.request).unwrap();
    assert_eq!(cc["chat_template_kwargs"]["enable_thinking"], false);
    assert_eq!(cc["chat_template_kwargs"]["custom_flag"], 1);
    assert_eq!(cc["chat_template_kwargs"]["thinking_budget"], 2048);
}

#[test]
fn messages_output_config_effort_translates_to_reasoning_effort() {
    // `xhigh` is what Claude Code sends; without it every Claude Code turn 400s.
    for (effort, cc_effort) in [("low", "low"), ("max", "xhigh"), ("xhigh", "xhigh")] {
        let adapted = adapt_messages_body(json!({"output_config": {"effort": effort}})).unwrap();
        let cc = serde_json::to_value(&adapted.request).unwrap();
        assert_eq!(cc["reasoning_effort"], cc_effort, "effort {effort}");
    }
}

/// Both ask for Anthropic-hosted execution nothing on the model's CC endpoint provides; forwarding them
/// via `unmodeled` would 200 while silently running without the capability.
#[test]
fn messages_anthropic_hosted_execution_fields_are_refused() {
    let mcp_rejection = adapt_messages_body(
        json!({"mcp_servers": [{"type": "url", "url": "https://mcp.example.com", "name": "ex"}]}),
    )
    .expect_err("mcp_servers must be refused");
    assert!(
        matches!(mcp_rejection, RequestRejection::Unsupported(ref detail) if detail.contains("mcp_servers")),
        "{mcp_rejection:?}"
    );
    let container_rejection =
        adapt_messages_body(json!({"container": "cont_1"})).expect_err("container must be refused");
    assert!(
        matches!(container_rejection, RequestRejection::Unsupported(ref detail) if detail.contains("container")),
        "{container_rejection:?}"
    );
}

#[test]
fn messages_output_config_format_translates_to_cc_response_format() {
    let schema = json!({"type": "object", "properties": {"answer": {"type": "string"}}});
    let adapted = adapt_messages_body(
        json!({"output_config": {"format": {"type": "json_schema", "schema": schema}}}),
    )
    .unwrap();
    let response_format = serde_json::to_value(&adapted.request.response_format).unwrap();
    assert_eq!(response_format["type"], "json_schema");
    assert_eq!(response_format["json_schema"]["schema"], schema);
    assert_eq!(response_format["json_schema"]["strict"], true);

    // A schema-less json_schema and an untranslatable type are refused, never dropped.
    let missing_schema =
        adapt_messages_body(json!({"output_config": {"format": {"type": "json_schema"}}}))
            .err()
            .unwrap();
    assert!(missing_schema.detail().contains("requires `schema`"));
    let unknown_type =
        adapt_messages_body(json!({"output_config": {"format": {"type": "grammar"}}}))
            .err()
            .unwrap();
    assert!(unknown_type.detail().contains("format.type"));
}

/// The payload Claude Code sends on its conversation-title call, once per interactive session:
/// `effort` and `format` together in one `output_config`. Refusing it leaves sessions untitled.
#[test]
fn messages_output_config_claude_code_title_call_translates() {
    let adapted = adapt_messages_body(json!({"output_config": {"effort": "high", "format": {
        "type": "json_schema",
        "schema": {"type": "object", "properties": {"title": {"type": "string"}}},
    }}}))
    .unwrap();
    let cc = serde_json::to_value(&adapted.request).unwrap();
    assert_eq!(cc["reasoning_effort"], "high");
    assert_eq!(cc["response_format"]["type"], "json_schema");
    assert_eq!(
        cc["response_format"]["json_schema"]["schema"]["properties"]["title"]["type"],
        "string"
    );
}

#[test]
fn adapt_messages_missing_model_is_refused() {
    let body = serde_json::to_vec(
        &json!({"max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}),
    )
    .unwrap();
    let err = adapt(&body, ClientProtocol::Messages, &HeaderMap::new())
        .err()
        .unwrap();
    assert!(err.detail().contains("missing field `model`"), "{err}");
}

#[test]
fn adapt_responses_missing_model_is_refused() {
    let body = serde_json::to_vec(&json!({"input": "hi"})).unwrap();
    let err = adapt(&body, ClientProtocol::Responses, &HeaderMap::new())
        .err()
        .unwrap();
    assert!(err.detail().contains("missing field `model`"), "{err}");
}

#[test]
fn adapt_responses_non_json_echoed_arguments_are_refused() {
    let err = responses_echoed_tool_call("call_1", "get_weather", "{not json")
        .err()
        .unwrap();
    assert!(err.contains("non-JSON `arguments`"), "{err}");
}

/// The openai/codex SDKs send `tool_choice: "auto"` unconditionally, even with no tools declared —
/// forwarded as-is, model backends 400 with "When using `tool_choice`, `tools` must be set."
#[test]
fn adapt_responses_max_tool_calls_is_the_loop_budget_not_a_model_field() {
    let body =
        serde_json::to_vec(&json!({"model": "m", "input": "hi", "max_tool_calls": 3})).unwrap();
    let adapted = request_only(&body, ClientProtocol::Responses, &HeaderMap::new()).unwrap();
    assert_eq!(adapted.max_tool_calls, NonZeroU32::new(3));
    // Enforced by TB, so it must not ride `unmodeled` to the model's CC endpoint.
    let cc_body = serde_json::to_value(&adapted.request).unwrap();
    assert!(cc_body.get("max_tool_calls").is_none(), "{cc_body}");

    let zero =
        serde_json::to_vec(&json!({"model": "m", "input": "hi", "max_tool_calls": 0})).unwrap();
    let refused = adapt(&zero, ClientProtocol::Responses, &HeaderMap::new())
        .expect_err("a zero budget must be refused, not treated as unbudgeted");
    assert!(refused.detail().contains("max_tool_calls"), "{refused}");
}

#[test]
fn adapt_responses_drops_auto_tool_choice_without_tools() {
    let body =
        serde_json::to_vec(&json!({"model": "m", "input": "hi", "tool_choice": "auto"})).unwrap();
    let adapted = request_only(&body, ClientProtocol::Responses, &HeaderMap::new()).unwrap();
    assert!(adapted.request.tool_choice.is_none());
    assert!(adapted.request.tools.is_none());
}

#[test]
fn adapt_responses_refuses_demanding_tool_choice_without_tools() {
    let body = serde_json::to_vec(&json!({"model": "m", "input": "hi", "tool_choice": "required"}))
        .unwrap();
    let err = adapt(&body, ClientProtocol::Responses, &HeaderMap::new())
        .err()
        .unwrap();
    assert!(err.detail().contains("declares no tools"), "{err}");
}

#[test]
fn adapt_responses_plain_string_input_becomes_one_user_message() {
    let body = serde_json::to_vec(&json!({"model": "m", "input": "hi"})).unwrap();
    let adapted = request_only(&body, ClientProtocol::Responses, &HeaderMap::new()).unwrap();
    let cc = serde_json::to_value(&adapted.request).unwrap();
    assert_eq!(cc["messages"], json!([{"role": "user", "content": "hi"}]));
}

#[test]
fn adapt_responses_instructions_become_a_leading_system_message() {
    let body = serde_json::to_vec(&json!({
        "model": "m",
        "instructions": "be terse",
        "input": "hi",
    }))
    .unwrap();
    let adapted = request_only(&body, ClientProtocol::Responses, &HeaderMap::new()).unwrap();
    let cc = serde_json::to_value(&adapted.request).unwrap();
    assert_eq!(
        cc["messages"][0],
        json!({"role": "system", "content": "be terse"})
    );
    assert_eq!(cc["messages"][1]["content"], "hi");
}

/// Deliberate break: `ReasoningItemContent` is tagged as of async-openai 0.38, so untagged content
/// is refused rather than defaulted. Nothing depends on the old shape yet, and defaulting it would
/// have to stay forever.
#[test]
fn adapt_responses_refuses_reasoning_content_without_the_tag() {
    let items: Result<Vec<InputItem>, _> = serde_json::from_value(json!([
        {"type": "reasoning", "id": "r1", "summary": [], "content": [{"text": "let me check"}]},
    ]));
    assert!(items.is_err());
}

/// A whole assistant turn — reasoning, text, and an open (client) tool call — flattened across
/// separate input items, the way a client echoes a prior response's `output` back as `input`.
#[test]
fn adapt_responses_reassembles_a_flattened_assistant_turn() {
    let out = responses_input(json!([
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "weather?"}]},
        {"type": "reasoning", "id": "r1", "summary": [], "content": [{"type": "reasoning_text", "text": "let me check"}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "checking"}]},
        {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": "{\"city\":\"SF\"}"},
    ]));
    assert_eq!(out.len(), 2);
    assert_eq!(out[0]["role"], "user");
    assert_eq!(out[1]["role"], "assistant");
    assert_eq!(out[1]["content"], "checking");
    // Fork Segments semantics (same as the Messages path): the replayed call closes the reasoning
    // segment, so segment 0 precedes call 0 and one empty trailing segment follows it.
    assert_eq!(out[1]["reasoning_content"], json!(["let me check", ""]));
    assert_eq!(out[1]["tool_calls"][0]["id"], "c1");
    assert_eq!(
        out[1]["tool_calls"][0]["function"]["arguments"],
        r#"{"city":"SF"}"#
    );
}

/// `function_call_output` (a client answering its own open call) splits the turn just like an
/// Anthropic `tool_result` does — same technique, flattened item list instead of nested blocks.
#[test]
fn adapt_responses_function_call_output_splits_the_turn() {
    let out = responses_input(json!([
        {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "72F"},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "It's 72F."}]},
    ]));
    assert_eq!(out.len(), 3);
    assert_eq!(out[0]["tool_calls"][0]["id"], "c1");
    assert_eq!(
        out[1],
        json!({"role": "tool", "tool_call_id": "c1", "content": "72F"})
    );
    assert_eq!(out[2]["content"], "It's 72F.");
}

/// A server tool's `mcp_call` carries its own result — one item, already resolved — so it becomes
/// a CC tool call *and* its answering `tool` message in the same step, with no separate output item
/// to wait for.
#[test]
fn adapt_responses_mcp_call_carries_its_own_result() {
    let out = responses_input(json!([
        {
            "type": "mcp_call",
            "id": "c1",
            "name": "baseten__stub__search",
            "server_label": "stub",
            "arguments": "{\"q\":\"x\"}",
            "output": "RESULT",
        },
    ]));
    assert_eq!(out.len(), 2);
    assert_eq!(out[0]["tool_calls"][0]["id"], "c1");
    assert_eq!(
        out[0]["tool_calls"][0]["function"]["arguments"],
        r#"{"q":"x"}"#
    );
    assert_eq!(
        out[1],
        json!({"role": "tool", "tool_call_id": "c1", "content": "RESULT"})
    );
}

#[test]
fn adapt_responses_item_reference_is_refused() {
    let items: Vec<InputItem> = serde_json::from_value(json!([{"id": "resp_123_item_0"}])).unwrap();
    let err = translate_input_items(&items, &mut Vec::new(), &mut Losses::default())
        .err()
        .unwrap();
    assert!(err.contains("item_reference"), "{err}");
}

#[test]
fn adapt_responses_stateful_fields_are_refused() {
    // `store` is accepted and carried onto the CC body (OpenAI SDKs default it to true).
    let body = serde_json::to_vec(&json!({"model": "m", "input": "hi", "store": true})).unwrap();
    let adapted = request_only(&body, ClientProtocol::Responses, &HeaderMap::new()).unwrap();
    assert_eq!(adapted.request.store, Some(true));

    for field in [
        "previous_response_id",
        "conversation",
        "background",
        "prompt",
    ] {
        let value = if field == "previous_response_id" {
            json!("resp_1")
        } else if field == "conversation" {
            json!({"id": "conv_1"})
        } else if field == "prompt" {
            json!({"id": "pmpt_1"})
        } else {
            json!(true)
        };
        let mut body = json!({"model": "m", "input": "hi"});
        body[field] = value;
        let body = serde_json::to_vec(&body).unwrap();
        let Err(err) = adapt(&body, ClientProtocol::Responses, &HeaderMap::new()) else {
            panic!("{field} must be refused");
        };
        assert!(
            err.detail().contains(field) || err.detail().contains("stateless"),
            "{field}: {err}"
        );
    }
}

#[test]
fn adapt_responses_tool_choice_shape_translation() {
    let translated = |tool_choice: Value| {
        let parsed: ToolChoiceParam = serde_json::from_value(tool_choice).unwrap();
        json!(translate_responses_tool_choice(&parsed).unwrap())
    };
    assert_eq!(translated(json!("auto")), json!("auto"));
    assert_eq!(translated(json!("required")), json!("required"));
    assert_eq!(translated(json!("none")), json!("none"));
    assert_eq!(
        translated(json!({"type": "function", "name": "get_weather"})),
        json!({"type": "function", "function": {"name": "get_weather"}})
    );
}

#[test]
fn adapt_responses_translates_text_format_and_reasoning_effort() {
    let body = serde_json::to_vec(&json!({
        "model": "m",
        "input": "hi",
        "text": {"format": {"type": "json_schema", "name": "out", "schema": {"type": "object"}}},
        "reasoning": {"effort": "low"},
    }))
    .unwrap();
    let adapted = request_only(&body, ClientProtocol::Responses, &HeaderMap::new())
        .ok()
        .unwrap();
    let cc = serde_json::to_value(&adapted.request).unwrap();
    assert_eq!(cc["response_format"]["type"], "json_schema");
    assert_eq!(cc["response_format"]["json_schema"]["name"], "out");
    assert_eq!(cc["reasoning_effort"], "low");
    // Translated, not ridden through: the Responses spellings must not reach the model.
    assert!(cc.get("text").is_none());
    assert!(cc.get("reasoning").is_none());
}

#[test]
fn adapt_responses_refuses_untranslatable_text_fields() {
    let err = cc_response_format(
        serde_json::from_value(json!({"format": {"type": "text"}, "verbosity": "low"})).unwrap(),
    )
    .err()
    .unwrap();
    assert!(err.contains("text.verbosity"), "{err}");
}

/// Every `reasoning.summary` level is accepted (Codex sends `auto`; other SDK clients send
/// `concise`/`detailed`): the summary is an egress rendering decision, never forwarded to the model,
/// while the effort beside it still reaches the model.
#[test]
fn adapt_responses_accepts_every_reasoning_summary_and_keeps_the_effort() {
    for summary in ["auto", "concise", "detailed"] {
        let effort = cc_reasoning_effort(
            serde_json::from_value(json!({"summary": summary, "effort": "high"})).unwrap(),
        )
        .unwrap_or_else(|e| panic!("summary {summary} must be accepted: {e}"));
        assert_eq!(
            serde_json::to_value(effort).unwrap(),
            json!("high"),
            "effort must survive an accepted summary"
        );
        let adapted =
            adapt_responses_body(json!({"reasoning": {"summary": summary, "effort": "low"}}))
                .unwrap();
        let cc = serde_json::to_value(&adapted.request).unwrap();
        assert_eq!(cc["reasoning_effort"], "low");
        assert!(cc.get("reasoning").is_none(), "summary is not forwarded");
    }
}

#[test]
fn adapt_responses_reasoning_item_summary_is_folded() {
    let out = responses_input(json!([
        {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "recap"}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}]},
    ]));
    assert_eq!(out[0]["reasoning_content"], "recap");
}

#[test]
fn adapt_responses_client_tool_strict_flag_is_forwarded() {
    let cc_tools = responses_client_function_tool(json!({
        "type": "function", "name": "f", "strict": true,
        "parameters": {"type": "object"},
    }))
    .unwrap();
    assert_eq!(cc_tools.len(), 1);
    assert_eq!(cc_tools[0].function.strict, Some(true));
}

#[test]
fn adapt_responses_client_function_tool_and_server_selection() {
    let body = serde_json::to_vec(&json!({
        "model": "m",
        "input": "hi",
        "tools": [
            {"type": "function", "name": "get_weather", "parameters": {"type": "object"}},
        ],
    }))
    .unwrap();
    let adapted = request_only(&body, ClientProtocol::Responses, &HeaderMap::new()).unwrap();
    let cc = serde_json::to_value(&adapted.request).unwrap();
    assert_eq!(cc["tools"][0]["type"], "function");
    assert_eq!(cc["tools"][0]["function"]["name"], "get_weather");
}

#[test]
fn adapt_responses_non_function_tool_is_refused() {
    let err = responses_client_function_tool(json!({"type": "web_search"}))
        .err()
        .unwrap();
    assert!(err.detail().contains("WebSearch"), "{err}");
}

fn web_search_request(tool: Value) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "model": "m",
        "input": "find it",
        "tools": [tool],
    }))
    .unwrap()
}

fn messages_web_search_request(max_uses: u32) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "model": "m",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "find it"}],
        "tools": [{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": max_uses,
        }],
        "tool_choice": {"type": "tool", "name": "web_search"},
    }))
    .unwrap()
}

// --- the coding-adapter rewrite seam -----------------------------------------
//
// tool-bank's Claude Code / Codex adapters plug in through `IngressHooks::rewrite_ingress`; these
// tests drive the seam with a fake rewrite mirroring the shape of tool-bank's: a client
// `web_search`-flavored tool expands into `baseten__*` selections, `max_uses` translates to an
// iteration bound, and a forced tool_choice is repointed at the first expanded tool.
//
// Dropped with the adapters themselves (provider/execution behavior with no seam expression):
// provider-priority resolution and its rejections, SearchToolBindings' second-search-tool refusal,
// the Anthropic web_search filter (`allowed_domains`/`blocked_domains`/`user_location`) refusals,
// Codex namespace flattening, and replayed `web_search_call` absorption.

struct NoopAdapter;

impl CodingAdapter for NoopAdapter {
    fn render_tool_call(&self, _call: &ToolCallToRender<'_>) -> Option<RenderedToolCall> {
        None
    }
}

struct FakeWebSearchRewrite {
    inner: StubHooks,
}

impl FakeWebSearchRewrite {
    fn new() -> Self {
        Self {
            inner: StubHooks::with_tools(&[
                "baseten__selected__search_web",
                "baseten__selected__fetch_page",
            ]),
        }
    }
}

impl IngressHooks for FakeWebSearchRewrite {
    fn limits(&self) -> IngressLimits {
        self.inner.limits()
    }

    fn on_server_tool(&mut self, protocol: ClientProtocol, entry: &Value) -> ToolDisposition {
        self.inner.on_server_tool(protocol, entry)
    }

    fn rewrite_ingress(
        &mut self,
        _protocol: ClientProtocol,
        _headers: &HeaderMap,
        body: &mut Value,
    ) -> Result<Option<IngressRewrite>, RequestRejection> {
        let Some(tools) = body.get_mut("tools").and_then(Value::as_array_mut) else {
            return Ok(None);
        };
        let Some(position) = tools.iter().position(|tool| {
            tool.get("type")
                .and_then(Value::as_str)
                .is_some_and(|kind| kind.starts_with("web_search"))
        }) else {
            return Ok(None);
        };
        let declared = tools.remove(position);
        let max_react_iterations = match declared.get("max_uses") {
            None | Some(Value::Null) => None,
            Some(value) => Some(
                value
                    .as_u64()
                    .and_then(|uses| u32::try_from(uses).ok())
                    .and_then(NonZeroU32::new)
                    .ok_or_else(|| {
                        RequestRejection::malformed("`max_uses` must be a positive integer")
                    })?,
            ),
        };
        tools.insert(position, json!({"type": "baseten__selected__search_web"}));
        tools.insert(
            position + 1,
            json!({"type": "baseten__selected__fetch_page"}),
        );
        if body
            .get("tool_choice")
            .and_then(|tool_choice| tool_choice.get("name"))
            .is_some()
        {
            body["tool_choice"] = json!({"type": "tool", "name": "baseten__selected__search_web"});
        }
        Ok(Some(IngressRewrite {
            adapter: Box::new(NoopAdapter),
            max_react_iterations,
        }))
    }
}

#[test]
fn rewrite_ingress_expands_tools_repoints_choice_and_bounds_iterations() {
    let AdaptedIngress {
        request: adapted,
        coding_adapter,
        ..
    } = adapt_request(
        &messages_web_search_request(8),
        ClientProtocol::Messages,
        &HeaderMap::new(),
        &mut FakeWebSearchRewrite::new(),
    )
    .unwrap();
    let cc = serde_json::to_value(&adapted.request).unwrap();
    assert_eq!(
        cc["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|tool| tool["function"]["name"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec![
            "baseten__selected__search_web",
            "baseten__selected__fetch_page"
        ]
    );
    assert_eq!(
        cc["tool_choice"]["function"]["name"],
        "baseten__selected__search_web"
    );
    assert_eq!(adapted.max_react_iterations.get(), 8);
    assert_eq!(
        adapted.server_tool_claims.joined(),
        "baseten__selected__search_web, baseten__selected__fetch_page"
    );
    assert!(coding_adapter.is_some());
}

/// Without a claiming/rewriting hook, the stub hooks keep tool-bank's stance: a vendor-hosted tool
/// aimed at this endpoint is refused. (The shipped [`DropServerTools`] default drops it instead —
/// pinned below.)
#[test]
fn messages_hosted_web_search_without_a_claiming_hook_is_refused() {
    let err = adapt(
        &messages_web_search_request(8),
        ClientProtocol::Messages,
        &HeaderMap::new(),
    )
    .err()
    .unwrap();
    assert!(err.detail().contains("web_search_20250305"), "{err}");
}

#[test]
fn rewrite_bound_outside_the_react_bounds_is_clamped_not_refused() {
    for (max_uses, expected) in [
        (1, REACT_ITERATIONS_MIN.get()),
        (REACT_ITERATIONS_MAX.get() + 1, REACT_ITERATIONS_MAX.get()),
        (8, 8),
    ] {
        let adapted = adapt_request(
            &messages_web_search_request(max_uses),
            ClientProtocol::Messages,
            &HeaderMap::new(),
            &mut FakeWebSearchRewrite::new(),
        )
        .map(|ingress| ingress.request)
        .unwrap_or_else(|e| panic!("max_uses {max_uses} is a legal Anthropic value: {e}"));
        assert_eq!(
            adapted.max_react_iterations.get(),
            expected,
            "max_uses {max_uses}"
        );
    }
}

#[test]
fn rewrite_bound_rejects_zero_and_reads_null_as_absent() {
    let adapt_with = |max_uses: Value| {
        let mut body: Value = serde_json::from_slice(&messages_web_search_request(8)).unwrap();
        body["tools"][0]["max_uses"] = max_uses;
        adapt_request(
            &serde_json::to_vec(&body).unwrap(),
            ClientProtocol::Messages,
            &HeaderMap::new(),
            &mut FakeWebSearchRewrite::new(),
        )
        .map(|ingress| ingress.request)
    };
    // Zero searches is not a budget the loop can serve.
    let err = adapt_with(json!(0)).expect_err("max_uses 0 is not a runnable budget");
    assert!(err.detail().contains("max_uses"), "{err}");
    // Null is the field unset, so the hooks' default stands rather than a 400.
    let defaulted = adapt_with(Value::Null).expect("an explicit null is the field being absent");
    assert_eq!(
        defaulted.max_react_iterations,
        test_hooks().limits().default_react_iterations
    );
}

#[test]
fn rewrite_bound_conflicting_with_tool_settings_is_refused() {
    let mut body: Value = serde_json::from_slice(&messages_web_search_request(8)).unwrap();
    body["baseten"] = json!({"tool_settings": {"max_react_iterations": 4}});
    let err = adapt_request(
        &serde_json::to_vec(&body).unwrap(),
        ClientProtocol::Messages,
        &HeaderMap::new(),
        &mut FakeWebSearchRewrite::new(),
    )
    .expect_err("two bounds on one loop must not be resolved by a silent pick");
    assert!(err.detail().contains("send one, not both"), "{err}");
}

// --- the shipped DropServerTools default (standard dynamo) --------------------

#[test]
fn drop_server_tools_drops_hosted_tools_and_degrades_a_naming_tool_choice() {
    let body = serde_json::to_vec(&json!({
        "model": "m",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "find it"}],
        "tools": [
            {"type": "web_search_20250305", "name": "web_search"},
            {"name": "get_weather", "input_schema": {"type": "object"}},
        ],
        "tool_choice": {"type": "tool", "name": "web_search"},
    }))
    .unwrap();
    let adapted = adapt_request(
        &body,
        ClientProtocol::Messages,
        &HeaderMap::new(),
        &mut DropServerTools,
    )
    .unwrap()
    .request;
    let cc = serde_json::to_value(&adapted.request).unwrap();
    assert_eq!(cc["tools"].as_array().unwrap().len(), 1);
    assert_eq!(cc["tools"][0]["function"]["name"], "get_weather");
    assert_eq!(
        cc["tool_choice"], "auto",
        "a tool_choice naming the dropped tool degrades to auto"
    );
    assert!(adapted.server_tool_claims.is_empty());
}

#[test]
fn drop_server_tools_omits_tool_choice_when_no_tools_survive() {
    let adapted = adapt_request(
        &messages_web_search_request(8),
        ClientProtocol::Messages,
        &HeaderMap::new(),
        &mut DropServerTools,
    )
    .unwrap()
    .request;
    assert!(adapted.request.tools.is_none());
    assert!(adapted.request.tool_choice.is_none());
    assert!(adapted.server_tool_claims.is_empty());
}

#[test]
fn drop_server_tools_drops_reserved_cc_selections() {
    let body = serde_json::to_vec(&json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"type": "baseten__web__search"},
            {"type": "function", "function": {"name": "ls", "parameters": {"type": "object"}}},
        ],
    }))
    .unwrap();
    let adapted = adapt_request(
        &body,
        ClientProtocol::ChatCompletions,
        &HeaderMap::new(),
        &mut DropServerTools,
    )
    .unwrap()
    .request;
    let cc = serde_json::to_value(&adapted.request).unwrap();
    assert_eq!(cc["tools"].as_array().unwrap().len(), 1);
    assert_eq!(cc["tools"][0]["function"]["name"], "ls");
    assert!(adapted.server_tool_claims.is_empty());
}

/// The adapter mints the `srvtoolu_` prefix on egress; ingress strips it so a replayed call
/// resolves to the id the model issued and the prefix can never double up.
#[test]
fn replayed_server_tool_use_id_round_trips_with_exactly_one_prefix() {
    let msg = anthropic_message(json!({"role": "assistant", "content": [
        {"type": "server_tool_use", "id": "srvtoolu_c1", "name": "web_search",
         "input": {"query": "rust"}},
    ]}));
    let cc_id = translated(&msg)[0]["tool_calls"][0]["id"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(cc_id, "c1");
    assert!(
        !cc_id.starts_with("srvtoolu_"),
        "ingress must strip the prefix so an egress adapter never doubles it: {cc_id}"
    );
}

#[test]
fn translates_replayed_server_tool_blocks_to_cc_call_and_result() {
    let msg = anthropic_message(json!({"role": "assistant", "content": [
        {"type": "server_tool_use", "id": "srvtoolu_c1", "name": "web_search",
         "input": {"query": "rust"}},
        {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_c1", "content": [
            {"type": "web_search_result", "title": "Rust", "url": "https://example.com/rust"}
        ]},
    ]}));
    let out = translated(&msg);
    assert_eq!(out.len(), 2);
    // The prefix is the adapter's, minted on egress: the model only ever issued `c1`.
    assert_eq!(out[0]["tool_calls"][0]["id"], json!("c1"));
    assert_eq!(
        out[0]["tool_calls"][0]["function"]["name"],
        json!("web_search")
    );
    assert_eq!(out[1]["role"], json!("tool"));
    assert_eq!(out[1]["tool_call_id"], json!("c1"));
    assert!(
        out[1]["content"]
            .as_str()
            .unwrap()
            .contains("https://example.com/rust")
    );
}

#[test]
fn messages_replayed_server_tool_blocks_are_accepted_without_a_coding_adapter() {
    let body = serde_json::to_vec(&json!({
        "model": "m",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "find it"},
            {"role": "assistant", "content": [
                {"type": "server_tool_use", "id": "srvtoolu_c1", "name": "web_search",
                 "input": {"query": "rust"}},
                {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_c1", "content": [
                    {"type": "web_search_result", "title": "Rust", "url": "https://example.com/rust"}
                ]},
            ]},
        ],
    }))
    .unwrap();
    let adapted = request_only(&body, ClientProtocol::Messages, &HeaderMap::new())
        .expect("a Messages caller may replay the blocks Anthropic's own protocol defines");
    let cc = serde_json::to_value(&adapted.request).unwrap();
    assert_eq!(cc["messages"][1]["tool_calls"][0]["id"], "c1");
    assert_eq!(cc["messages"][2]["role"], "tool");
}

/// A Responses hosted-tool shape (`web_search`) routes to the hooks; the stub hooks keep
/// tool-bank's refusal. (The Codex rewrite that used to absorb it lives with tool-bank now,
/// expressed through `rewrite_ingress` — see the fake-rewrite tests above.)
#[test]
fn responses_hosted_web_search_routes_to_the_hooks() {
    let err = adapt(
        &web_search_request(json!({"type": "web_search"})),
        ClientProtocol::Responses,
        &HeaderMap::new(),
    )
    .expect_err("web_search without a claiming hook must keep the rejection");
    assert!(err.detail().contains("web_search"), "{err}");

    let adapted = adapt_request(
        &web_search_request(json!({"type": "web_search"})),
        ClientProtocol::Responses,
        &HeaderMap::new(),
        &mut DropServerTools,
    )
    .unwrap()
    .request;
    assert!(adapted.request.tools.is_none());
    assert!(adapted.server_tool_claims.is_empty());
}

/// A replayed `web_search_call` input item has no CC translation here: skipped (fork parity — a
/// transcript echo must not fail the request), not refused. tool-bank's Codex path still strips
/// them in its `rewrite_ingress` before typed parsing, so it never reaches this skip.
#[test]
fn responses_replayed_web_search_call_items_are_skipped() {
    let body = serde_json::to_vec(&json!({
        "model": "m",
        "input": [
            {
                "id": "ws_1",
                "type": "web_search_call",
                "status": "completed",
                "action": {
                    "type": "search",
                    "queries": ["first", "second"],
                    "query": "first"
                }
            },
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        ]
    }))
    .unwrap();
    let adapted = request_only(&body, ClientProtocol::Responses, &HeaderMap::new())
        .expect("web_search_call replay is skipped, not refused");
    assert_eq!(adapted.request.messages.len(), 1);
}

/// The refusal truncates the echoed tool definition, which is client-controlled text: the pad sweep
/// walks the cut across every UTF-8 byte offset so a char-boundary split would surface here.
#[test]
fn adapt_responses_refusal_truncates_non_ascii_tool_definition() {
    for pad in 0..8 {
        let description = format!("{}{}", "a".repeat(pad), "é".repeat(200));
        let err = responses_client_function_tool(json!({
            "type": "custom", "name": "crm", "description": description,
            "format": {"type": "text"},
        }))
        .err()
        .unwrap();
        assert!(
            err.detail().contains("unsupported tool type"),
            "pad {pad}: {err}"
        );
        assert!(
            err.detail().len() < 400,
            "pad {pad}: unbounded echo, {} bytes",
            err.detail().len()
        );
    }
}

// --- Fork validation floor (CC-pivot: suite-validated ingress conformance, kept over TB) --------

fn adapt_responses_body(extra: Value) -> Result<AdaptedRequest, RequestRejection> {
    let mut body = json!({"model": "m", "input": "hi"});
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    request_only(
        &serde_json::to_vec(&body).unwrap(),
        ClientProtocol::Responses,
        &HeaderMap::new(),
    )
}

/// `service_tier` maps one-to-one: `auto` (platform picks) and `default` (explicitly the default
/// tier) are different requests, and the response echoes whichever was sent, so folding one into the
/// other would silently change routing and make the echo lie.
#[test]
fn adapt_responses_service_tier_maps_one_to_one() {
    for tier in ["auto", "default", "flex", "scale", "priority"] {
        let adapted = adapt_responses_body(json!({"service_tier": tier})).unwrap();
        assert_eq!(
            json!(adapted.request.service_tier),
            json!(tier),
            "service_tier `{tier}` must reach CC unchanged"
        );
    }
    let adapted = adapt_responses_body(json!({})).unwrap();
    assert!(
        adapted.request.service_tier.is_none(),
        "absent stays absent"
    );
}

/// Responses replay segments reasoning around tool calls exactly like the Messages path: segment i
/// precedes call i, so `[reasoning, function_call, reasoning]` re-renders byte-exactly for a
/// segments-aware template (KV-cache prefix) instead of collapsing to one flat string.
#[test]
fn adapt_responses_replay_segments_reasoning_around_tool_calls() {
    let out = responses_input(json!([
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]},
        {"type": "reasoning", "id": "r1", "summary": [], "content": [{"type": "reasoning_text", "text": "first"}]},
        {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": "{}"},
        {"type": "reasoning", "id": "r2", "summary": [], "content": [{"type": "reasoning_text", "text": "second"}]},
        {"type": "function_call_output", "call_id": "c1", "output": "42"},
    ]));
    assert_eq!(out[1]["role"], "assistant");
    assert_eq!(out[1]["reasoning_content"], json!(["first", "second"]));
    assert_eq!(out[1]["tool_calls"][0]["id"], "c1");
    assert_eq!(out[2]["role"], "tool");
}

/// Codex declares MCP apps as `namespace` tool groups. Members flatten to `{ns}__{name}` function
/// tools on the declaration side, and a replayed `function_call` carrying `namespace` refolds onto
/// the same flat name — otherwise every echoed call would be an orphan for the tool it names.
#[test]
fn adapt_responses_namespace_group_flattens_and_echo_refolds() {
    let adapted = adapt_responses_body(json!({
        "tools": [{"type": "namespace", "name": "crm", "description": "CRM tools", "tools": [
            {"type": "function", "name": "lookup", "description": "look up", "parameters": {"type": "object"}}
        ]}],
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]},
            {"type": "function_call", "call_id": "c1", "name": "lookup", "namespace": "crm", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"}
        ]
    }))
    .unwrap();
    let tools = serde_json::to_value(adapted.request.tools.as_ref().unwrap()).unwrap();
    assert_eq!(tools[0]["function"]["name"], "crm__lookup");
    let messages = serde_json::to_value(&adapted.request.messages).unwrap();
    assert_eq!(
        messages[1]["tool_calls"][0]["function"]["name"],
        "crm__lookup"
    );
}

/// `top_logprobs` outside 0..=20 is refused up front (OpenAI 400s at 21); 20 rides through.
#[test]
fn adapt_responses_top_logprobs_bounds() {
    assert!(adapt_responses_body(json!({"top_logprobs": 20})).is_ok());
    let err = adapt_responses_body(json!({"top_logprobs": 21})).unwrap_err();
    assert!(matches!(err, RequestRejection::Malformed(_)), "{err:?}");
    let err = adapt_responses_body(json!({"top_logprobs": "5"})).unwrap_err();
    assert!(matches!(err, RequestRejection::Malformed(_)), "{err:?}");
}

/// An explicit but empty assistant message item survives as a turn boundary (`content: ""`), so two
/// user turns around it do not merge under a strict-alternation template; a refusal part folds
/// into the assistant text.
#[test]
fn adapt_responses_empty_assistant_turn_and_refusal_keep_boundaries() {
    let out = responses_input(json!([
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "one"}]},
        {"type": "message", "role": "assistant", "content": []},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "two"}]},
        {"type": "message", "role": "assistant", "content": [{"type": "refusal", "refusal": "no"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "three"}]},
    ]));
    assert_eq!(out.len(), 5);
    assert_eq!(out[1]["role"], "assistant");
    assert_eq!(out[1]["content"], "");
    assert_eq!(out[3]["content"], "no");
}

/// The thinking-budget upper bound needs a `max_tokens` to compare against; a request that leaves
/// `max_tokens` to the deployment template is accepted, the lower bound still applies.
#[test]
fn adapt_messages_thinking_budget_without_max_tokens_is_accepted() {
    let mut body = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled", "budget_tokens": 2048}});
    let ok = request_only(
        &serde_json::to_vec(&body).unwrap(),
        ClientProtocol::Messages,
        &HeaderMap::new(),
    );
    assert!(ok.is_ok(), "{ok:?}");
    body["thinking"]["budget_tokens"] = json!(512);
    let err = request_only(
        &serde_json::to_vec(&body).unwrap(),
        ClientProtocol::Messages,
        &HeaderMap::new(),
    );
    assert!(
        matches!(err, Err(RequestRejection::Malformed(_))),
        "{err:?}"
    );
}

/// Codex Responses-Lite framing declares tools on an `additional_tools` input item instead of (or
/// in addition to) top-level `tools`. Those declarations must reach the CC tool list — otherwise a
/// `tool_choice: "required"` beside them 400s as "requires a tool call, but no tools" — and a name
/// declared in both places is advertised once (top-level first).
#[test]
fn responses_additional_tools_item_declares_cc_tools() {
    let adapted = adapt_responses_body(json!({
        "input": [
            {"type": "additional_tools", "role": "developer", "tools": [
                {"type": "function", "name": "lookup", "parameters": {"type": "object"}},
                {"type": "function", "name": "shared", "description": "from item",
                 "parameters": {"type": "object"}},
            ]},
            {"role": "user", "content": "hi"},
        ],
        "tools": [
            {"type": "function", "name": "shared", "description": "top-level",
             "parameters": {"type": "object"}},
        ],
        "tool_choice": "required",
    }))
    .expect("additional_tools declarations satisfy tool_choice: required");
    let cc = serde_json::to_value(&adapted.request).unwrap();
    assert_eq!(cc["tool_choice"], "required");
    let names: Vec<&str> = cc["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["function"]["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        ["shared", "lookup"],
        "top-level first, duplicate dropped"
    );
    assert_eq!(
        cc["tools"][0]["function"]["description"], "top-level",
        "the top-level declaration wins the clash"
    );
    // The item itself is a declaration, not transcript: only the user message remains.
    assert_eq!(cc["messages"].as_array().unwrap().len(), 1);
    assert_eq!(cc["messages"][0]["role"], "user");
}

/// Codex's default request carries `include: ["reasoning.encrypted_content"]`. Standard dynamo
/// (the default hooks) accepts it — the include is dropped with a warning, never forwarded — while
/// hooks that opt in (tool-bank) still refuse it.
#[test]
fn responses_encrypted_reasoning_include_is_dropped_unless_hooks_reject_it() {
    let body = serde_json::to_vec(&json!({
        "model": "m", "input": "hi",
        "include": ["reasoning.encrypted_content"],
    }))
    .unwrap();
    let adapted = adapt_request(
        &body,
        ClientProtocol::Responses,
        &HeaderMap::new(),
        &mut DropServerTools,
    )
    .expect("default hooks accept the Codex default include");
    let cc = serde_json::to_value(&adapted.request.request).unwrap();
    assert!(
        cc.get("include").is_none(),
        "include never reaches the CC body"
    );

    struct RejectingHooks;
    impl IngressHooks for RejectingHooks {
        fn rejects_encrypted_reasoning_include(&self) -> bool {
            true
        }
    }
    let err = adapt_request(
        &body,
        ClientProtocol::Responses,
        &HeaderMap::new(),
        &mut RejectingHooks,
    )
    .expect_err("opted-in hooks keep tool-bank's refusal");
    assert!(
        err.detail().contains("reasoning.encrypted_content"),
        "{err:?}"
    );
}

/// Anthropic bounds temperature at 1; the OpenAI surfaces allow up to 2. The same value must be a
/// 400 on one wire and valid on the other.
#[test]
fn temperature_ranges_differ_per_wire() {
    let err = adapt_messages_body(json!({"temperature": 1.5}))
        .expect_err("1.5 is outside Anthropic's 0..=1 range");
    assert!(err.detail().contains("between 0 and 1"), "{err}");

    adapt_responses_body(json!({"temperature": 1.5})).expect("1.5 is inside OpenAI's 0..=2 range");

    let err = adapt_responses_body(json!({"temperature": 2.1}))
        .expect_err("2.1 is outside OpenAI's 0..=2 range");
    assert!(err.detail().contains("between 0 and 2"), "{err}");
}

#[test]
fn top_p_out_of_range_is_refused_on_both_wires() {
    for build in [adapt_messages_body, adapt_responses_body] {
        let err = build(json!({"top_p": 1.5})).expect_err("top_p is 0..=1 on both wires");
        assert!(
            err.detail().contains("top_p must be between 0 and 1"),
            "{err}"
        );
        build(json!({"top_p": 0.9})).expect("in-range top_p must pass");
    }
}

#[test]
fn messages_thinking_budget_below_floor_is_refused() {
    let err = adapt_messages_body(json!({"thinking": {"type": "enabled", "budget_tokens": 100}}))
        .expect_err("Anthropic floors the manual budget at 1024");
    assert!(err.detail().contains("at least 1024"), "{err}");
}

#[test]
fn messages_thinking_budget_at_or_above_max_tokens_is_refused() {
    // The fixture's max_tokens is 4096; the budget must be strictly below it.
    let err = adapt_messages_body(json!({"thinking": {"type": "enabled", "budget_tokens": 4096}}))
        .expect_err("budget >= max_tokens leaves no room for output");
    assert!(err.detail().contains("less than max_tokens"), "{err}");
}

#[test]
fn messages_tool_without_input_schema_is_refused() {
    let err = adapt_messages_body(json!({"tools": [{"name": "x"}]}))
        .expect_err("a client tool without input_schema has no callable shape");
    assert!(err.detail().contains("input_schema"), "{err}");
}

/// An orphan tool_result (no preceding tool_use with that id) is a corrupted transcript: 400, not
/// a nonsensical forward. The matched pair on the same body shape passes.
#[test]
fn messages_orphan_tool_result_is_refused_and_matched_pair_passes() {
    let orphan = json!({"messages": [
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c-orphan", "content": "R"}
        ]},
    ]});
    let err = adapt_messages_body(orphan).expect_err("orphan tool_result must 400");
    assert!(err.detail().contains("no preceding tool call"), "{err}");
    assert!(err.detail().contains("c-orphan"), "{err}");

    let matched = json!({"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "c1", "name": "f", "input": {}}
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "R"}
        ]},
    ]});
    adapt_messages_body(matched).expect("a tool_result answering its tool_use must pass");
}

#[test]
fn responses_orphan_function_call_output_is_refused() {
    let err = adapt_responses_body(json!({"input": [
        {"type": "function_call_output", "call_id": "c9", "output": "x"}
    ]}))
    .expect_err("orphan function_call_output must 400");
    assert!(err.detail().contains("no preceding tool call"), "{err}");
    assert!(err.detail().contains("c9"), "{err}");
}

/// Anthropic prefill: a trailing assistant message means "continue this message". The lowered
/// trailing CC assistant message carries `partial: true`; a non-trailing assistant does not.
#[test]
fn messages_trailing_assistant_marks_prefill_partial() {
    let adapted = adapt_messages_body(json!({"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "The answer is"},
    ]}))
    .unwrap();
    let cc = serde_json::to_value(&adapted.request).unwrap();
    let messages = cc["messages"].as_array().unwrap();
    assert_eq!(messages.last().unwrap()["partial"], json!(true));

    // Not trailing: the assistant turn is history, not a prefill to continue.
    let adapted = adapt_messages_body(json!({"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "The answer is"},
        {"role": "user", "content": "go on"},
    ]}))
    .unwrap();
    let cc = serde_json::to_value(&adapted.request).unwrap();
    for message in cc["messages"].as_array().unwrap() {
        assert!(
            message.get("partial").is_none(),
            "no message may carry partial: {message}"
        );
    }
}

/// tool-bank's original nesting guard, restored: a top-level `tool_settings` is a 400 naming the
/// right home, not a silently defaulted cap plus an unknown key forwarded to the model.
#[test]
fn top_level_tool_settings_is_a_nesting_error_on_every_protocol() {
    for protocol in [
        ClientProtocol::ChatCompletions,
        ClientProtocol::Messages,
        ClientProtocol::Responses,
    ] {
        let body = json!({
            "model": "m",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "input": "hi",
            "tool_settings": {"max_react_iterations": 3},
        });
        let err = adapt(
            &serde_json::to_vec(&body).unwrap(),
            protocol,
            &HeaderMap::new(),
        )
        .expect_err("top-level tool_settings must be refused");
        assert!(
            matches!(err, RequestRejection::Malformed(_)),
            "{protocol:?}: {err}"
        );
        assert!(
            err.detail().contains("baseten.tool_settings"),
            "{protocol:?}: {err}"
        );
    }
    // Nested correctly it is honored, not refused.
    let ok = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "baseten": {"tool_settings": {"max_react_iterations": 3}},
    });
    adapt(
        &serde_json::to_vec(&ok).unwrap(),
        ClientProtocol::ChatCompletions,
        &HeaderMap::new(),
    )
    .expect("nested tool_settings is valid");
}

/// `input[]` is an untagged enum, so serde alone reports only "data did not match any variant of
/// untagged enum InputParam" — no index, no field. Three prod callers hit that message and could not
/// act on it.
#[test]
fn a_malformed_input_item_rejection_names_its_index() {
    let chat_completions_shaped_content_part = json!({
        "input": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    });
    let rejection = adapt_responses_body(chat_completions_shaped_content_part)
        .expect_err("a `text` content part is not a Responses input part");
    let RequestRejection::Malformed(detail) = &rejection else {
        panic!("{rejection:?}");
    };
    assert!(detail.contains("input[0]"), "{detail}");
}

#[test]
fn malformed_member_rejections_name_the_member_on_every_ingress() {
    let messages = adapt_messages_body(json!({"max_tokens": "ten"}))
        .expect_err("a string `max_tokens` must be refused");
    assert!(
        matches!(messages, RequestRejection::Malformed(ref detail) if detail.contains("max_tokens")),
        "{messages:?}"
    );
    let cc_body = serde_json::to_vec(&json!({
        "model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": "hot",
    }))
    .unwrap();
    let chat_completions =
        request_only(&cc_body, ClientProtocol::ChatCompletions, &HeaderMap::new())
            .expect_err("a string `temperature` must be refused");
    let detail = format!("{chat_completions:?}");
    assert!(detail.contains("temperature"), "{detail}");
    let responses = adapt_responses_body(json!({"reasoning": {"effort": "sideways"}}))
        .expect_err("an unknown effort must be refused");
    assert!(
        matches!(responses, RequestRejection::Malformed(ref detail) if detail.contains("reasoning.effort")),
        "{responses:?}"
    );
}

/// `serde_path_to_error` renders a path without a leading dot, so composition must add exactly one
/// separator — never zero (`input[0]content`) and never two (`input[0]..content`).
#[test]
fn a_composed_member_path_carries_one_separator() {
    let nested_failure = json!({"model": "m", "input": "hi", "reasoning": {"effort": 5}});
    let rejection: Result<super::ResponsesRequest, _> =
        super::parse_client_json(ClientProtocol::Responses, "input[0]", nested_failure);
    let RequestRejection::Malformed(detail) = rejection.err().expect("a numeric effort is refused")
    else {
        panic!("expected a malformed rejection");
    };
    assert!(detail.contains("`input[0].reasoning.effort`"), "{detail}");
}

/// Responses `reasoning.effort` goes through the fork's alias table like chat `reasoning_effort`:
/// `max` (DeepSeek V4 / GLM clients) canonicalizes to `xhigh` instead of failing the upstream enum.
/// main-v1.2.0's own `CreateResponse` accepted it; the crate must not regress that.
#[test]
fn responses_reasoning_effort_max_canonicalizes_like_chat() {
    let adapted = adapt_responses_body(json!({"reasoning": {"effort": "max"}}))
        .expect("`max` is a documented client spelling");
    let cc = serde_json::to_value(&adapted.request).unwrap();
    assert_eq!(cc["reasoning_effort"], "xhigh");
}

/// Every drop/degrade/skip on the ingress path is a typed [`Loss`] on the adapted request — the
/// answer to "what did this request lose becoming Chat Completions?" as data, not log grep.
#[test]
fn losses_record_server_tool_drop_tool_choice_degrade_and_field_drops() {
    let body = serde_json::to_vec(&json!({
        "model": "m", "max_tokens": 64,
        "messages": [{"role": "user", "content": "ZEBRA_CONTENT_MARKER"}],
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        "tool_choice": {"type": "tool", "name": "web_search"},
        "cache_control": {"type": "ephemeral"},
        "service_tier": "standard_only",
        "baseten": {"not_a_member": 1},
    }))
    .unwrap();
    let mut hooks = DropServerTools;
    let adapted = adapt_request(
        &body,
        ClientProtocol::Messages,
        &HeaderMap::new(),
        &mut hooks,
    )
    .expect("drops are non-fatal")
    .request;
    let kinds: Vec<(LossKind, usize)> = crate::loss::count_by_kind(&adapted.losses);
    assert_eq!(
        kinds,
        vec![
            (LossKind::ServerToolDropped, 1),
            (LossKind::ToolChoiceDegraded, 1),
            (LossKind::RequestFieldDropped, 2),
            (LossKind::ExtensionFieldDropped, 1),
        ],
        "{:?}",
        adapted.losses
    );
    let fields: Vec<&str> = adapted
        .losses
        .iter()
        .map(|loss| loss.field.as_str())
        .collect();
    assert!(fields.contains(&"tools[0]"), "{fields:?}");
    assert!(fields.contains(&"tool_choice"), "{fields:?}");
    assert!(fields.contains(&"cache_control"), "{fields:?}");
    assert!(fields.contains(&"service_tier"), "{fields:?}");
    assert!(fields.contains(&"baseten.not_a_member"), "{fields:?}");
    for loss in &adapted.losses {
        assert!(
            !loss.detail.contains("ZEBRA_CONTENT_MARKER"),
            "detail must not carry message content: {loss:?}"
        );
    }
}

/// A clean request records nothing: the empty list is the common case a counter must not inflate.
#[test]
fn a_fully_translatable_request_records_no_losses() {
    let adapted = adapt_messages_body(json!({})).unwrap();
    assert!(adapted.losses.is_empty(), "{:?}", adapted.losses);
}

/// The consumer's counter hook sees exactly the recorded list, in order.
#[test]
fn every_loss_reaches_the_on_loss_hook() {
    #[derive(Default)]
    struct Counting(Vec<Loss>);
    impl IngressHooks for Counting {
        fn on_loss(&mut self, loss: &Loss) {
            self.0.push(loss.clone());
        }
    }
    let body = serde_json::to_vec(&json!({
        "model": "m",
        "input": [
            {"type": "compaction", "encrypted_content": "..."},
            {"role": "user", "content": "hi"},
        ],
        "include": ["reasoning.encrypted_content"],
    }))
    .unwrap();
    let mut hooks = Counting::default();
    let adapted = adapt_request(
        &body,
        ClientProtocol::Responses,
        &HeaderMap::new(),
        &mut hooks,
    )
    .unwrap()
    .request;
    assert_eq!(hooks.0, adapted.losses);
    assert_eq!(
        crate::loss::count_by_kind(&adapted.losses),
        vec![
            (LossKind::ReplayItemIgnored, 1),
            (LossKind::IncludeIgnored, 1)
        ],
        "{:?}",
        adapted.losses
    );
    let ignored = adapted
        .losses
        .iter()
        .find(|loss| loss.kind == LossKind::ReplayItemIgnored)
        .unwrap();
    assert_eq!(ignored.field, "input[0]");
}

/// Skipped Messages content blocks name their position so the customer log line can be joined back
/// to the request the client sent.
#[test]
fn skipped_content_blocks_name_message_and_block_index() {
    let adapted = adapt_messages_body(json!({
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "ok"},
                {"type": "redacted_thinking", "data": "xxx"},
            ]},
            {"role": "user", "content": [
                {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "d"}},
                {"type": "text", "text": "and?"},
            ]},
        ],
    }))
    .unwrap();
    let fields: Vec<&str> = adapted
        .losses
        .iter()
        .map(|loss| loss.field.as_str())
        .collect();
    assert_eq!(
        fields,
        vec!["messages[1].content[1]", "messages[2].content[0]"],
        "{:?}",
        adapted.losses
    );
    assert!(
        adapted
            .losses
            .iter()
            .all(|loss| loss.kind == LossKind::ContentBlockSkipped)
    );
}

/// Routing is by shape only: a Chat Completions `tools[]` entry whose `type` is not `function`
/// (an OpenAI-hosted `web_search_preview`, a tool-bank selection, ...) reaches the hooks; the
/// standard-dynamo default drops it with a recorded loss instead of forwarding it to an engine
/// that cannot execute it. (Previously only reserved-typed entries were routed and any other
/// non-function entry was forwarded verbatim.)
#[test]
fn cc_non_function_tool_entries_route_to_the_hooks_and_drop_by_default() {
    let body = serde_json::to_vec(&json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"type": "web_search_preview"},
            {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}},
        ],
    }))
    .unwrap();
    let mut hooks = DropServerTools;
    let adapted = adapt_request(
        &body,
        ClientProtocol::ChatCompletions,
        &HeaderMap::new(),
        &mut hooks,
    )
    .unwrap()
    .request;
    let tools = serde_json::to_value(adapted.request.tools.as_ref().unwrap()).unwrap();
    assert_eq!(tools.as_array().unwrap().len(), 1, "{tools}");
    assert_eq!(tools[0]["function"]["name"], "lookup");
    assert_eq!(
        crate::loss::count_by_kind(&adapted.losses),
        vec![(LossKind::ServerToolDropped, 1)]
    );
}

/// The crate knows no tool namespace: a client tool named like a consumer's server-tool selection
/// is an ordinary function tool here. Guarding a consumer's namespace against spoofing is the
/// consumer's job at dispatch time (tool-bank refuses a model call to an unclaimed `baseten__*`
/// name in its loop).
#[test]
fn client_tools_named_like_a_consumer_namespace_are_ordinary_function_tools() {
    let adapted = adapt_messages_body(json!({
        "tools": [{"name": "baseten__acme__lookup", "input_schema": {"type": "object"}}],
    }))
    .unwrap();
    let tools = serde_json::to_value(adapted.request.tools.as_ref().unwrap()).unwrap();
    assert_eq!(tools[0]["function"]["name"], "baseten__acme__lookup");
    assert!(adapted.losses.is_empty(), "{:?}", adapted.losses);
}
