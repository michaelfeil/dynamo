//! b10: Baseten-owned regression tests for the Chat Completions wrapper, kept out of the upstream
//! file so fork rebases never conflict on test hunks.

use super::*;
use serde_json::json;

/// Regression: the wrapper flattens the wire request alongside its own
/// `unsupported_fields` catch-all. If the wire type carried a flattened
/// catch-all of its own, serde would take the struct's map path and no
/// key would be consumed, so `model`/`messages` would land in
/// `unsupported_fields` (warning on every request, 400ing under
/// `DYN_ALLOW_UNSUPPORTED_FIELDS=false`) and re-serialization would emit
/// every key twice.
#[test]
fn modeled_fields_never_land_in_unsupported_fields() {
    let request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.5,
        "chat_template_kwargs": {"enable_thinking": true},
        "nvext": {"ignore_eos": true},
        "min_tokens": 3,
        "priority": {"level": 1},
    }))
    .unwrap();
    assert!(
        request.unsupported_fields.is_empty(),
        "unsupported_fields: {:?}",
        request.unsupported_fields.keys().collect::<Vec<_>>()
    );
    assert_eq!(request.inner.model, "m");
    assert_eq!(request.inner.temperature, Some(0.5));
    assert_eq!(
        request.chat_template_args.as_ref().unwrap()["enable_thinking"],
        json!(true)
    );
    assert_eq!(request.common.min_tokens, Some(3));
    assert!(request.baseten_ext.priority.is_some());
    assert!(request.nvext.is_some());

    let text = serde_json::to_string(&request).unwrap();
    for key in ["\"model\"", "\"messages\"", "\"temperature\"", "\"nvext\""] {
        assert_eq!(text.matches(key).count(), 1, "{key} duplicated in {text}");
    }
}

/// Only genuinely unknown keys reach `unsupported_fields`, and they never
/// serialize back onto the engine-bound body.
#[test]
fn unknown_fields_are_held_in_unsupported_fields_only() {
    let request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "context_management": {"edits": []},
    }))
    .unwrap();
    assert_eq!(
        request.unsupported_fields.keys().collect::<Vec<_>>(),
        ["context_management"]
    );
    let body = serde_json::to_value(&request).unwrap();
    assert!(body.get("context_management").is_none());
}
