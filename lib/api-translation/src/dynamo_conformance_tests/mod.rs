//! Conformance suite: dynamo's upstream aggregator test *vectors* ported onto our own parser
//! ([`SseParser`]), asserting parity of the semantic outcome (not verbatim event
//! bytes — dynamo's aggregator is re-implemented as our parser, not linked).
//!
//! Source: basetenlabs/dynamo @ 68dec805 (Apache-2.0). Files here mirror dynamo's source basenames
//! for 1:1 correlation (they intentionally break our `_test.rs` convention — the special role of
//! this dir):
//!  - `aggregator.rs` <- lib/llm/src/protocols/openai/chat_completions/aggregator.rs
//!
//! SPDX-License-Identifier: Apache-2.0.
//!
//! SKIPPED upstream tests (dynamo-specific behavior that is explicitly NOT our contract):
//!  - test_multiple_deltas_merge_nvext_fields — `nvext` is dropped from our vendored types.
//!  - test_multiple_choices — we serve single-choice; the parser reads `choices[0]` only.
//!  - test_tool_calling_finish_reason_override_* — dynamo rewrites finish_reason to ToolCalls when
//!    tool calls are present; we pass the model's finish_reason through verbatim.
//!  - test_parses_aggregated_tool_call_text_into_tool_calls /
//!    test_preserves_non_tool_content_when_parsing_aggregated_tool_calls — dynamo's hermes text
//!    tool-call jail-parsing (`ParsingOptions`); we read structured `tool_calls` only.
//!  - test_harmony_* — Harmony marker analysis/drop; not our contract.
//!  - test_reasoning_only_response_serializes_content_key_as_null — a dynamo serialization quirk.
//!
//! Upstream tests already covered by `sse_parser_test.rs` (not re-ported here): test_empty_stream,
//! test_single_delta, test_no_tool_calling_preserves_original_finish_reason, test_tool_calling_output.
#![allow(clippy::unwrap_used)]

use crate::test_utils::*;

mod aggregator;
#[cfg(feature = "render-conformance")]
mod oai;
mod test_stream_converter;
