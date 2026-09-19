// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use baseten_parsers::{ReasoningStream, reasoning_parser_families};

#[test]
fn delimiters_and_unicode_are_chunk_invariant() {
    for (family, input, initial) in [
        ("qwen3", "<think>café 杭州</think>answer", None),
        ("deepseek_v4", "café 杭州</think>answer", Some(true)),
        ("deepseek_r1", "café 杭州</think>answer", None),
        ("kimi", "◁think▷café 杭州◁/think▷answer", None),
        ("mistral", "[THINK]café 杭州[/THINK]answer", None),
    ] {
        for split in input.char_indices().map(|(i, _)| i).chain([input.len()]) {
            let mut parser = ReasoningStream::new(family, initial).unwrap();
            let mut normal = String::new();
            let mut reasoning = String::new();
            for chunk in [&input[..split], "", &input[split..]] {
                let out = parser.step(chunk, &[]).unwrap();
                normal.push_str(&out.normal_text);
                reasoning.push_str(&out.reasoning_text);
            }
            let tail = parser.finish().unwrap();
            normal.push_str(&tail.normal_text);
            reasoning.push_str(&tail.reasoning_text);
            assert_eq!(normal, "answer", "{family} split {split}");
            assert_eq!(reasoning, "café 杭州", "{family} split {split}");
        }
    }
}

#[test]
fn eof_lifecycle_and_independent_choices() {
    let mut first = ReasoningStream::new("qwen3", None).unwrap();
    let mut second = ReasoningStream::new("deepseek_r1", Some(false)).unwrap();
    assert_eq!(
        first
            .step("<think>reason</thi", &[])
            .unwrap()
            .reasoning_text,
        "reason"
    );
    assert_eq!(second.step("answer", &[]).unwrap().normal_text, "answer");
    assert_eq!(first.finish().unwrap().reasoning_text, "</thi");
    assert!(first.finish().is_err());
    assert!(first.step("late", &[]).is_err());
    second.finish().unwrap();
    assert!(ReasoningStream::new("typo", None).is_err());
    assert!(ReasoningStream::new("QWEN3", None).is_ok());
}

#[test]
fn every_registered_family_is_available() {
    for family in reasoning_parser_families() {
        ReasoningStream::new(family, None)
            .unwrap()
            .finish()
            .unwrap();
    }
}
