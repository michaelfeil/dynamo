// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use baseten_parsers::{UnifiedStream, request_init, vllm};

fn init() -> baseten_parsers::UnifiedParserInit {
    request_init(vec![], "none", "native", None, "reject").unwrap()
}

#[test]
fn vllm_requires_a_tokenizer_and_native_unified_family() {
    assert_eq!(
        vllm::FAMILIES,
        &["gemma4", "hy_v3", "hy_v4", "inkling", "kimi_k3"]
    );
    let error = UnifiedStream::new_with_backend("vllm", "gemma4", &[], init(), None)
        .err()
        .expect("missing tokenizer should fail");
    assert!(error.to_string().contains("tokenizer_path"));
    let error = UnifiedStream::new_with_backend("typo", "gemma4", &[], init(), None)
        .err()
        .expect("unknown backend should fail");
    assert!(error.to_string().contains("backend"));
    let error = UnifiedStream::new_with_backend(
        "vllm",
        "glm47",
        &[],
        init(),
        Some(std::path::Path::new("/nonexistent/tokenizer.json")),
    )
    .err()
    .expect("tool-only family should fail");
    assert!(
        error
            .to_string()
            .contains("unknown vLLM unified parser family")
    );

    let reasoning_start = request_init(vec![], "reasoning", "native", None, "reject").unwrap();
    let error = UnifiedStream::new_with_backend(
        "vllm",
        "gemma4",
        &[],
        reasoning_start,
        Some(std::path::Path::new("/nonexistent/tokenizer.json")),
    )
    .err()
    .expect("unsupported starting state should fail");
    assert!(error.to_string().contains("starting_state"));
}

#[test]
fn native_vllm_unified_parser_loads_tokenizer_and_streams_text() {
    let mut tokenizer: serde_json::Value = serde_json::from_str(include_str!(
        "../../tokenizers/tests/data/minimal-bpe/tokenizer.json"
    ))
    .unwrap();
    tokenizer["added_tokens"] = serde_json::json!([
        {"id": 23, "content": "<|channel>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 24, "content": "<channel|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
    ]);
    let path = std::env::temp_dir().join(format!(
        "baseten-parser-{}-tokenizer.json",
        std::process::id()
    ));
    std::fs::write(&path, serde_json::to_vec(&tokenizer).unwrap()).unwrap();
    let result = (|| {
        let mut parser =
            UnifiedStream::new_with_backend("vllm", "gemma4", &[], init(), Some(&path))?;
        let mut events = parser.advance(Some("<|channel>thought\nNeed weather.<|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}<tool_call|>"))?;
        events.extend(parser.advance(None)?);
        Ok::<_, anyhow::Error>(events)
    })();
    std::fs::remove_file(&path).unwrap();
    let events = result.unwrap();
    assert_eq!(events.len(), 3);
    assert_eq!(
        events[0],
        baseten_parsers::Event::Reasoning("Need weather.".into())
    );
    let baseten_parsers::Event::ToolCall(call) = &events[1] else {
        panic!("expected call")
    };
    assert_eq!(call.name.as_deref(), Some("get_weather"));
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&call.arguments).unwrap(),
        serde_json::json!({"location":"Paris"})
    );
    assert!(!call.complete);
    let baseten_parsers::Event::ToolCall(closed) = &events[2] else {
        panic!("expected closure")
    };
    assert!(closed.complete);
}
