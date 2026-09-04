//! Render-equivalence oracle: proves tool-bank's multi-turn ReAct request assembly renders (at
//! dynamo) to a byte-identical token/prompt PREFIX across turns — the KV-cache-prefix contract.
//!
//! Drives the REAL request edge — `SseParser` -> `MessageHistoryAccumulator` ->
//! `build_next_request` — then renders each turn's request with the in-workspace, runtime-free
//! dynamo prompt renderer (`dynamo-renderer`, TEST-ONLY dev-dep) and asserts append-only prefix stability
//! on the rendered bytes. This is the render-side complement to the fast, ungated request-assembly
//! units in `history_test.rs`.
//!
//! Oracle: basetenlabs/dynamo `lib/renderer` @ 68dec805 (Apache-2.0). The Qwen3 fixture and the
//! append-only contract mirror that crate's own `test_qwen3_thinking_append_only_across_tool_use_turn`
//! (lib/renderer/src/template/oai.rs @ rev).
//!
//! Gated behind the `render-conformance` feature (mod decl in `mod.rs`) so the fast test path
//! never runs the minijinja renderer. Run: `cargo test --package b10-dynamo-api-translation
//! --features render-conformance`.
//!
//! SPDX-License-Identifier: Apache-2.0.
#![allow(clippy::unwrap_used)]

use std::collections::HashMap;

use dynamo_renderer::{ChatTemplate, ContextMixins, OAIChatLikeRequest, PromptFormatter};
use minijinja::value::Value as JinjaValue;
use serde_json::{Value, json};

use crate::history::MessageHistoryAccumulator;
use crate::model::{BillingVerdict, ServerToolCallStatus, ToolOutput, UsageReport};
use crate::model::{ToolCall, ToolInvocation};
use crate::request::build_next_request;
use crate::sse_parser::SseParser;
use crate::{CcMessage, CcRequest, SemanticChunk};

/// Real Qwen3-4B-Thinking-2507 chat template, vendored from dynamo (see fixture `_provenance`).
/// Branches on `tool_call.arguments is string` (renders arg bytes verbatim) and references
/// `reasoning_content` — the two properties the byte-exact cache prefix depends on.
const QWEN3_FIXTURE: &str = include_str!("data/qwen3_4b_thinking_tokenizer_config.json");

/// The Qwen3-Thinking generation prompt the template appends when `add_generation_prompt` is true.
/// A turn's rendered request ends here; the next turn reproduces everything BEFORE it and appends.
const GEN_PROMPT: &str = "<|im_start|>assistant\n<think>\n";

// --- render harness ---------------------------------------------------------

/// Models dynamo's render entry for a forwarded tool-bank request. Delegates every render input to
/// the bare `CreateChatCompletionRequest` impl, except `chat_template_args`: tool-bank forwards
/// `chat_template_kwargs` in the `unmodeled` passthrough, and dynamo's `NvCreateChatCompletionRequest`
/// threads it into the template context via its `chat_template_args` field (serde alias
/// `chat_template_kwargs`, lib/llm/src/protocols/openai/chat_completions.rs @ 68dec805). Reproduce
/// that threading so the oracle renders what dynamo would.
struct DynamoRenderReq<'a> {
    inner: &'a CcRequest,
    args: Option<HashMap<String, Value>>,
}

impl<'a> DynamoRenderReq<'a> {
    fn new(inner: &'a CcRequest) -> Self {
        let args = inner
            .unmodeled
            .get("chat_template_kwargs")
            .and_then(Value::as_object)
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect());
        Self { inner, args }
    }
}

impl OAIChatLikeRequest for DynamoRenderReq<'_> {
    fn model(&self) -> String {
        self.inner.model()
    }
    fn messages(&self) -> JinjaValue {
        self.inner.messages()
    }
    fn tools(&self) -> Option<JinjaValue> {
        self.inner.tools()
    }
    fn tool_choice(&self) -> Option<JinjaValue> {
        self.inner.tool_choice()
    }
    fn should_add_generation_prompt(&self) -> bool {
        self.inner.should_add_generation_prompt()
    }
    fn chat_template_args(&self) -> Option<&HashMap<String, Value>> {
        self.args.as_ref()
    }
}

fn formatter_from(fixture: &str) -> PromptFormatter {
    let config: ChatTemplate =
        serde_json::from_str(fixture).expect("parse tokenizer_config fixture");
    PromptFormatter::from_parts(config, ContextMixins::default(), false).expect("build formatter")
}

fn render(formatter: &PromptFormatter, req: &CcRequest) -> String {
    let PromptFormatter::OAI(f) = formatter;
    f.render(&DynamoRenderReq::new(req)).expect("render prompt")
}

fn strip_gen_prompt(rendered: &str) -> &str {
    rendered
        .strip_suffix(GEN_PROMPT)
        .expect("rendered request ends with the generation prompt")
}

/// Report the first byte divergence for a readable failure (mirrors the upstream oracle test).
fn assert_prefix(whole: &str, prefix: &str, label: &str) {
    if whole.starts_with(prefix) {
        return;
    }
    let div = prefix
        .as_bytes()
        .iter()
        .zip(whole.as_bytes())
        .position(|(prefix_byte, whole_byte)| prefix_byte != whole_byte)
        .unwrap_or_else(|| prefix.len().min(whole.len()));
    let lo = div.saturating_sub(40);
    panic!(
        "{label}: NOT an append-only prefix-extension; diverges at byte {div}\n  \
         prefix ends: ...{}|{}\n  \
         whole   has: ...{}|{}",
        String::from_utf8_lossy(&prefix.as_bytes()[lo..div]),
        String::from_utf8_lossy(&prefix.as_bytes()[div..(div + 60).min(prefix.len())]),
        String::from_utf8_lossy(&whole.as_bytes()[lo..div]),
        String::from_utf8_lossy(&whole.as_bytes()[div..(div + 60).min(whole.len())]),
    );
}

// --- ReAct scenario driven through the real request edge --------------------

fn chunk(delta: Value, finish: Option<&str>) -> String {
    json!({
        "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    })
    .to_string()
}

fn tool_delta(id: &str, name: &str, args: &str) -> Value {
    json!({"tool_calls": [{
        "index": 0, "id": id, "type": "function",
        "function": {"name": name, "arguments": args},
    }]})
}

/// One model turn: parse chunks through `SseParser`, fold into the accumulator, commit.
fn drive_turn(accumulator: &mut MessageHistoryAccumulator, sse_payloads: &[String]) {
    let mut parser = SseParser::default();
    let mut chunks: Vec<SemanticChunk> = Vec::new();
    for payload in sse_payloads {
        for chunk in parser.push_and_yield(payload).chunks {
            chunks.push(chunk.expect("parser push"));
        }
    }
    for chunk in parser.flush_and_yield() {
        chunks.push(chunk.expect("parser finish"));
    }
    for chunk in &chunks {
        accumulator.push(chunk);
    }
    let before = accumulator.messages().len();
    accumulator.commit_assistant_message();
    assert_eq!(
        accumulator.messages().len(),
        before + 1,
        "model turn produced one assistant message"
    );
}

fn tools_json() -> Value {
    json!([
        {"type": "function", "function": {
            "name": "get_weather",
            "description": "Current weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        }},
        {"type": "function", "function": {
            "name": "get_time",
            "description": "Current local time for a tz.",
            "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}, "required": ["tz"]}
        }}
    ])
}

/// The model's turn-1 tool-call argument bytes: deliberately irregular whitespace + a non-ASCII
/// value, so a naive parse->reserialize (`serde_json` compact) would NOT reproduce them. If these
/// survive into the rendered prefix verbatim, the `raw_args` byte-exactness contract holds.
const WEIRD_ARGS: &str = r#"{"city": "São Paulo" ,"units":  "metric"}"#;

fn seed_template() -> CcRequest {
    serde_json::from_value(json!({
        "model": "qwen3-thinking",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Weather then local time in São Paulo?"},
        ],
        "tools": tools_json(),
    }))
    .expect("seed template request")
}

/// Build the three successive predict requests of a 2-tool-call ReAct loop, driving the real edge.
/// Returns `[req_turn1, req_turn2, req_turn3]` — the bodies sent before each model turn.
fn react_requests(template: &CcRequest) -> Vec<CcRequest> {
    let seed: Vec<CcMessage> = template.messages.clone();
    let mut accumulator = MessageHistoryAccumulator::new(seed);
    let mut requests = vec![build_next_request(template, accumulator.messages(), true)];

    drive_turn(
        &mut accumulator,
        &[
            chunk(
                json!({"reasoning_content": "I'll check the weather first."}),
                None,
            ),
            chunk(
                tool_delta("call-weather", "baseten__weather__get_weather", WEIRD_ARGS),
                Some("tool_calls"),
            ),
        ],
    );
    accumulator.append_tool_results(&[invocation(
        "call-weather",
        "baseten__weather__get_weather",
        json!("18C, foggy"),
    )]);
    requests.push(build_next_request(template, accumulator.messages(), false));

    drive_turn(
        &mut accumulator,
        &[
            chunk(json!({"reasoning_content": "Now the local time."}), None),
            chunk(
                tool_delta(
                    "call-time",
                    "baseten__clock__get_time",
                    r#"{"tz":"America/Sao_Paulo"}"#,
                ),
                Some("tool_calls"),
            ),
        ],
    );
    accumulator.append_tool_results(&[invocation(
        "call-time",
        "baseten__clock__get_time",
        json!("14:03"),
    )]);
    requests.push(build_next_request(template, accumulator.messages(), false));

    requests
}

/// A successful server-tool invocation answering `call_id`.
fn invocation(call_id: &str, name: &str, content: serde_json::Value) -> ToolInvocation {
    ToolInvocation {
        server_call: crate::test_utils::server_tool_call(ToolCall {
            id: call_id.into(),
            name: name.into(),
            args: json!({}),
            raw_args: "{}".into(),
        }),
        output: ToolOutput {
            content,
            status: ServerToolCallStatus::Succeeded,
            verdict: BillingVerdict::billable_unreported(),
        },
    }
}

// --- Test A: render-equivalence oracle --------------------------------------

#[test]
fn react_requests_render_to_append_only_prefix() {
    let formatter = formatter_from(QWEN3_FIXTURE);
    let template = seed_template();
    let requests = react_requests(&template);

    let renders: Vec<String> = requests
        .iter()
        .map(|request| render(&formatter, request))
        .collect();

    // Each turn reproduces the previous turn's rendered history (everything before its generation
    // prompt) byte-for-byte, then appends. This is the KV-cache-prefix contract.
    assert_prefix(
        &renders[1],
        strip_gen_prompt(&renders[0]),
        "turn 2 vs turn 1",
    );
    assert_prefix(
        &renders[2],
        strip_gen_prompt(&renders[1]),
        "turn 3 vs turn 2",
    );

    // The model's verbatim arg bytes reach the rendered prefix unchanged (guards `raw_args`); a
    // compact reserialize would collapse the whitespace and never match.
    let compact = serde_json::to_string(
        &serde_json::from_str::<Value>(WEIRD_ARGS).expect("weird args parse"),
    )
    .expect("compact");
    assert_ne!(compact, WEIRD_ARGS, "fixture args must be non-canonical");
    assert!(
        renders[1].contains(WEIRD_ARGS) && renders[2].contains(WEIRD_ARGS),
        "verbatim tool-call arguments must appear in the rendered prefix"
    );
}

// --- Test B: settle preserve_order ------------------------------------------

/// Purpose-built probe template (NOT a model template): consumes two `chat_template_kwargs` by name
/// and echoes the messages, so any order-sensitivity in the passthrough would surface in the bytes.
/// `think` renders through an `if` rather than the bool itself: minijinja formats booleans the way
/// Python Jinja2 does (`True`), and the property under test is that the kwarg reached the context.
const ORDER_PROBE_TEMPLATE: &str = r#"{%- for m in messages %}{{ m.role }}={{ m.content }};{% endfor %}think={% if enable_thinking | default(false) %}on{% else %}off{% endif %};lang={{ lang | default("none") }};budget={{ thinking_budget | default("none") }}"#;

fn probe_formatter() -> PromptFormatter {
    formatter_from(&json!({ "chat_template": ORDER_PROBE_TEMPLATE }).to_string())
}

#[test]
fn passthrough_field_order_does_not_change_render() {
    let formatter = probe_formatter();

    // Identical requests; the `chat_template_kwargs` map and sibling `unmodeled` fields differ only
    // in key ORDER (preserve_order keeps JSON insertion order in `unmodeled`).
    let req_a: CcRequest = serde_json::from_value(json!({
        "model": "probe",
        "messages": [{"role": "user", "content": "hi"}],
        "chat_template_kwargs": {"enable_thinking": true, "lang": "en"},
        "guided_json": {"x": 1, "y": 2},
    }))
    .expect("req a");
    let req_b: CcRequest = serde_json::from_value(json!({
        "model": "probe",
        "messages": [{"role": "user", "content": "hi"}],
        "chat_template_kwargs": {"lang": "en", "enable_thinking": true},
        "guided_json": {"y": 2, "x": 1},
    }))
    .expect("req b");

    // VERDICT: field order does NOT affect the render. The renderer consumes kwargs as named
    // context variables and messages/tools by structure, never by JSON key order — so the
    // `preserve_order` guarantee on `unmodeled` is not load-bearing for the render/cache-prefix
    // contract (it only stabilizes the forwarded request JSON bytes).
    assert_eq!(
        render(&formatter, &req_a),
        render(&formatter, &req_b),
        "render must be invariant to passthrough field order"
    );
}

// --- Test C: fork extensions render -----------------------------------------

#[test]
fn fork_extensions_reach_render_without_corruption() {
    let formatter = probe_formatter();

    let with_thinking: CcRequest = serde_json::from_value(json!({
        "model": "probe",
        "messages": [{"role": "user", "content": "hi"}],
        "chat_template_kwargs": {"enable_thinking": true, "lang": "de"},
    }))
    .expect("with thinking");
    let without: CcRequest = serde_json::from_value(json!({
        "model": "probe",
        "messages": [{"role": "user", "content": "hi"}],
    }))
    .expect("without");

    // `chat_template_kwargs.enable_thinking` is threaded into the template context (fork extension
    // survives the round-trip); absent it, the same template renders the false branch.
    let rendered = render(&formatter, &with_thinking);
    assert!(
        rendered.contains("think=on") && rendered.contains("lang=de"),
        "chat_template_kwargs must reach the template context, got: {rendered}"
    );
    assert!(
        render(&formatter, &without).contains("think=off"),
        "absent enable_thinking renders the false branch"
    );

    // `thinking_budget` (Anthropic `budget_tokens` -> `chat_template_kwargs`) threads through too.
    let with_budget: CcRequest = serde_json::from_value(json!({
        "model": "probe",
        "messages": [{"role": "user", "content": "hi"}],
        "chat_template_kwargs": {"enable_thinking": true, "thinking_budget": 2048},
    }))
    .expect("with budget");
    assert!(
        render(&formatter, &with_budget).contains("budget=2048"),
        "thinking_budget must reach the template context"
    );

    // An unmodeled passthrough field that the template ignores must not corrupt the render.
    let mut with_extra = with_thinking;
    with_extra
        .unmodeled
        .insert("guided_regex".into(), json!("[0-9]+"));
    assert_eq!(
        render(&formatter, &with_extra),
        rendered,
        "unmodeled passthrough of a template-irrelevant field must not change the render"
    );
}
