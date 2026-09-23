# SPDX-FileCopyrightText: Copyright (c) 2026 Baseten Labs, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Binding smoke checks; parser behavior and registry coverage are tested in Rust."""

import pytest

from dynamo.parsers import (
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


def test_unified_reasoning_and_tool_call_binding():
    parser = UnifiedParserStream(
        "qwen3", [{"name": "weather", "parameters": {"type": "object"}}]
    )
    events = (
        parser.step(
            "<think>Check.</think>Looking up. "
            "<tool_call><function=weather></function></tool_call>Done."
        )
        + parser.finish()
    )
    assert [(event.kind, event.text) for event in events] == [
        ("reasoning", "Check."),
        ("text", "Looking up. "),
        ("tool_call", None),
        ("text", "Done."),
    ]
    (call,) = [event.call for event in events if event.kind == "tool_call"]
    assert (call.tool_index, call.name, call.arguments, call.complete) == (
        0,
        "weather",
        "{}",
        True,
    )


def test_invalid_configuration():
    with pytest.raises(ValueError):
        UnifiedParserStream("missing")
    with pytest.raises(ValueError):
        UnifiedParserStream("qwen3", [{"parameters": {}}])
    with pytest.raises(ValueError):
        UnifiedParserStream("qwen3", starting_state="invalid")


def test_removed_tool_stream_name_remains_importable():
    from dynamo._core import ToolCallStream as CoreToolCallStream

    with pytest.raises(RuntimeError, match="use UnifiedParserStream"):
        ToolCallStream("harmony")
    with pytest.raises(RuntimeError, match="use UnifiedParserStream"):
        CoreToolCallStream("harmony")


def test_vllm_unified_requires_tokenizer_path():
    from dynamo.parsers import VLLM_UNIFIED_PARSER_FAMILIES

    assert "gemma4" in VLLM_UNIFIED_PARSER_FAMILIES
    with pytest.raises(ValueError, match="tokenizer_path"):
        UnifiedParserStream("gemma4", backend="vllm")
