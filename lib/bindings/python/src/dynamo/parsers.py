# SPDX-FileCopyrightText: Copyright (c) 2026 Baseten Labs, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Request-scoped unified Rust parsers, loaded when used."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dynamo._core import (
        PARSER_UPSTREAM_REVISION,
        UNIFIED_PARSER_FAMILIES,
        VLLM_PARSER_UPSTREAM_REVISION,
        VLLM_UNIFIED_PARSER_FAMILIES,
        ParserEvent,
        ParserStreamError,
        ParserToolCall,
        ToolParseOutput,
        UnifiedParserStream,
    )


class ToolCallStream:
    """Compatibility import for the removed tool-only stream."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("ToolCallStream was removed; use UnifiedParserStream")


__all__ = [
    "PARSER_UPSTREAM_REVISION",
    "UNIFIED_PARSER_FAMILIES",
    "VLLM_PARSER_UPSTREAM_REVISION",
    "VLLM_UNIFIED_PARSER_FAMILIES",
    "ParserEvent",
    "ParserStreamError",
    "ParserToolCall",
    "ToolCallStream",
    "ToolParseOutput",
    "UnifiedParserStream",
]


def __getattr__(name: str) -> Any:
    if name in __all__ and name != "ToolCallStream":
        from dynamo import _core

        return getattr(_core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
