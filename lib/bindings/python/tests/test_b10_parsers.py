# SPDX-FileCopyrightText: Copyright (c) 2026 Baseten Labs, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Binding smoke checks; parser behavior and registry coverage are tested in Rust."""

import pytest

from dynamo.parsers import (
    TOOL_PARSER_FAMILIES,
    UNIFIED_PARSER_FAMILIES,
    ParserStreamError,
    ToolCallStream,
    UnifiedParserStream,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


def test_tool_binding():
    assert "glm47" in TOOL_PARSER_FAMILIES
    parser = ToolCallStream(
        "glm47", [{"name": "weather", "parameters": {"type": "object"}}]
    )
    assert not parser.prefers_tokens
    output = parser.step("<tool_call>weather</tool_call>")
    assert output.normal_text == ""
    (call,) = output.calls
    assert (call.tool_index, call.id, call.name, call.arguments, call.complete) == (
        0,
        None,
        "weather",
        "{}",
        True,
    )
    parser.finish()
    with pytest.raises(ParserStreamError, match="closed"):
        parser.step("late")


def test_unified_binding():
    assert "qwen3" in UNIFIED_PARSER_FAMILIES
    parser = UnifiedParserStream("qwen3")
    events = parser.step("<think>reason</think>answer") + parser.finish()
    assert [(event.kind, event.text, event.call) for event in events] == [
        ("reasoning", "reason", None),
        ("text", "answer", None),
    ]
    with pytest.raises(ParserStreamError, match="closed") as error:
        parser.finish()
    assert error.value.events == []


def test_invalid_configuration():
    with pytest.raises(ValueError):
        ToolCallStream("missing")
    with pytest.raises(ValueError):
        ToolCallStream("glm47", [{"parameters": {}}])
    with pytest.raises(ValueError):
        UnifiedParserStream("qwen3", starting_state="invalid")


def test_reasoning_binding():
    from dynamo.parsers import REASONING_PARSER_FAMILIES, ReasoningParserStream

    assert "deepseek_v4" in REASONING_PARSER_FAMILIES
    parser = ReasoningParserStream("deepseek_v4", in_reasoning=True)
    outputs = [parser.step(ch) for ch in "café 杭州</think>answer"]
    outputs.append(parser.finish())
    assert "".join(out.reasoning_text for out in outputs) == "café 杭州"
    assert "".join(out.normal_text for out in outputs) == "answer"
    with pytest.raises(ParserStreamError, match="closed"):
        parser.finish()
    with pytest.raises(ParserStreamError, match="closed"):
        parser.step("late")
    with pytest.raises(ValueError, match="unknown reasoning parser"):
        ReasoningParserStream("typo")


def test_reasoning_eof_and_request_isolation():
    from dynamo.parsers import ReasoningParserStream

    reasoning = ReasoningParserStream("qwen3")
    plain = ReasoningParserStream("deepseek_r1", in_reasoning=False)
    assert reasoning.step("<think>reason</thi").reasoning_text == "reason"
    assert plain.step("answer", token_ids=[]).normal_text == "answer"
    assert reasoning.finish().reasoning_text == "</thi"
    assert plain.finish().reasoning_text == ""


def test_vllm_tool_binding():
    from dynamo.parsers import VLLM_PARSER_UPSTREAM_REVISION, VLLM_TOOL_PARSER_FAMILIES

    assert len(VLLM_PARSER_UPSTREAM_REVISION) == 40
    assert "glm47" in VLLM_TOOL_PARSER_FAMILIES
    parser = ToolCallStream("glm47", backend="vllm")
    assert parser.completion_semantics == "native"
    assert not parser.prefers_tokens
    with pytest.raises(ParserStreamError, match="token input"):
        parser.step_tokens([1])
    (call,) = parser.step("<tool_call>weather</tool_call>").calls
    assert (call.name, call.arguments, call.complete) == ("weather", "{}", True)
    parser.finish()
    with pytest.raises(ParserStreamError, match="closed"):
        parser.finish()
    with pytest.raises(ValueError, match="backend"):
        ToolCallStream("glm47", backend="typo")


def test_vllm_streams_unfinished_arguments():
    parser = ToolCallStream("kimi_k2", backend="vllm")
    assert parser.completion_semantics == "stream_boundary"
    start = parser.step(
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.weather:0"
        "<|tool_call_argument_begin|>"
    )
    assert start.calls[0].name == "weather"
    (call,) = parser.step('{"city":').calls
    assert call.arguments == '{"city":'
    assert not call.complete
    parser.step('"Paris"}<|tool_call_end|><|tool_calls_section_end|>')
    (end,) = parser.finish().calls
    assert end.complete
    assert end.id == "functions.weather:0"


def test_vllm_partial_error_events():
    parser = ToolCallStream("qwen3_coder", backend="vllm")
    with pytest.raises(ParserStreamError) as caught:
        parser.step("visible<tool_call>\n<bad>\n</tool_call>")
    assert [(e.kind, e.text) for e in caught.value.events] == [("text", "visible")]
    with pytest.raises(ParserStreamError, match="closed"):
        parser.step("late")
