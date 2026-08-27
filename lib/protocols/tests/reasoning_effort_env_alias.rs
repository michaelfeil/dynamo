// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Integration test (own process) so the `REASONING_EFFORT_ALIASES` env var can
// seed the process-lifetime alias cache before any parse runs — proving the
// configurable override reaches BOTH request surfaces, not just chat. Keep this
// file to a single #[test]: the alias table is a OnceLock, so a second test in
// the same binary would race the initialization.

use dynamo_protocols::types::responses::CreateResponse;
use dynamo_protocols::types::{CreateChatCompletionRequest, ReasoningEffort};

#[test]
fn env_alias_override_applies_to_chat_and_responses() {
    // SAFETY: set before any thread reads the environment; this test binary is
    // single-threaded at this point and owns the process.
    unsafe {
        std::env::set_var(
            "REASONING_EFFORT_ALIASES",
            r#"{"max":"low","turbo":"high"}"#,
        );
    }

    // Responses surface: custom override (max→low, NOT the default max→xhigh)
    // proves the env table — not a hardcoded map — is what remaps.
    let responses_req: CreateResponse = serde_json::from_value(serde_json::json!({
        "input": "hi",
        "reasoning": {"effort": "max"}
    }))
    .unwrap();
    assert_eq!(
        responses_req.reasoning.unwrap().effort,
        Some(ReasoningEffort::Low)
    );

    // Same table serves a non-default alias key on responses.
    let responses_req: CreateResponse = serde_json::from_value(serde_json::json!({
        "input": "hi",
        "reasoning": {"effort": "turbo"}
    }))
    .unwrap();
    assert_eq!(
        responses_req.reasoning.unwrap().effort,
        Some(ReasoningEffort::High)
    );

    // Chat surface: identical behaviour from the identical table.
    let chat_req: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "max"
    }))
    .unwrap();
    assert_eq!(chat_req.reasoning_effort, Some(ReasoningEffort::Low));
}
