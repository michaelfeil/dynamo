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
