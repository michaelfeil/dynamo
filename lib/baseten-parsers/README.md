# Baseten parsers

Pure Rust lifecycle adapters around public `dynamo-parsers-v2` 0.6.3, pinned to
`ai-dynamo/frontend-crates` revision `bb20dd01b6257cff9b739be2b18e7a444980d04a`.
No Python dependency in this crate. Python bindings ship in the Dynamo wheel as
`dynamo.parsers`; no separate Python package is required.

The Git revision selects the source; the version is an additional consistency
check, not a crates.io fallback. This pin includes frontend-crates #247, #248,
and #253. Advance the manifest revision and `UPSTREAM_REVISION` together, then
update both the workspace and Python-binding lockfiles. New upstream commits
can be selected before a crates.io release; use their declared crate version.

Rust owns parser creation, configuration validation (`request_init`), lifecycle,
call-ID normalization, and ordered events including partial errors. The bindings
only convert Python values, release the GIL, and translate Rust results/errors.
`ToolCallStream::advance` and `UnifiedStream::advance` are the same Rust entry
points used by Python; `None` finalizes the stream.

Alternative backends implement the re-exported peer-shaped `ToolParser` or
`UnifiedParser` contracts and enter through `from_parser(Box<dyn ...>)`. They
receive the same lifecycle and result normalization without requiring upstream
registry mutation or changes to the binding. Named Python construction selects
the built-in upstream registry; additional named backends require Rust factory
integration, not Python callbacks.

## Python

```python
from dynamo.parsers import ToolCallStream, TOOL_PARSER_FAMILIES

parser = ToolCallStream("glm47", tools=[
    {"name": "weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}
    }}
])
output = parser.step(delta_text)
# output.normal_text; output.calls: tool_index, id, name, arguments, complete
tail = parser.finish()
```

Tool definitions use the flat upstream shape, not OpenAI's `function` wrapper.
Create one parser per response choice. Concatenate argument fragments by tool
index; do not parse each fragment as complete JSON. Model-supplied IDs are
preserved when upstream exposes them; `None` leaves ID generation to the caller.

All upstream tool families are selected through its registry, without a second
dispatch table: `harmony`, `harmony_text`, `deepseek_v4`, `qwen3_coder`,
`muse_glimmer`, `minimax_m2`, `minimax_m3`, `gemma4`, `glm47`, `kimi_k2`, `kimi_k3`.
Read `TOOL_PARSER_FAMILIES` for the authoritative list in the installed wheel.

Check `preserve_special_tokens` before configuring decoding. Harmony advertises
`prefers_tokens` and accepts `step_tokens(ids)` as well as `step(text)`; do not mix
input representations within a response. Token input on text-only parsers raises
instead of invoking upstream's no-op default.

## Ordered reasoning and tool events

```python
from dynamo.parsers import UnifiedParserStream, UNIFIED_PARSER_FAMILIES

parser = UnifiedParserStream("qwen3", tools=tools, starting_state="none")
events = parser.step(delta_text)
events += parser.finish()
# event.kind: text | reasoning | tool_call
# event.text for text/reasoning; event.call for tool_call
```

Unified families: `deepseek_v4`, `deepseek_v41`, `gemma4`, `qwen3`, `qwen3_coder`,
`muse_glimmer`, `kimi_k2`, `kimi_k3`, `kimi-k3` (aliases included). Not every tool
family has an upstream unified implementation; unavailable names fail explicitly.
Events retain upstream order. Apply answer-only text stops only to visible text,
not reasoning or tool arguments.

Initialization accepts `prompt_token_ids`, `starting_state` (`none`, `reasoning`,
`response`), `tool_output_mode` (`native`, `guided_json`), optional `named_tool`
in guided mode, and `invalid_guided_payload` (`reject`, `recover_as_text`,
`stream_best_effort`). Support is family-dependent; upstream rejects unsupported
initialization. Policy remains a caller decision.

## Lifecycle and compatibility

`finish()` is terminal; subsequent steps or finishes raise. Upstream parse errors
also close the stream. Invalid input representation is rejected before advancing
state. `ParserStreamError.events` preserves any unified events committed before
an error in that step; the caller must handle them explicitly rather than retrying
the same input. No parser silently falls back to the old implementation.

The wrapper preserves upstream grammar behavior, including GLM's complete-block
buffering and suppression of incomplete calls at EOF. It does not map model output
to HTTP errors, generate missing call IDs, enforce request tool-choice policy, or
change routing/cancellation. Existing response paths are unchanged: selecting this
backend requires model-specific integration and parity testing.

Rust parsing releases the GIL; Python output objects are materialized once per
step. No per-step JSON envelope is serialized. No performance improvement is
claimed without complete binding benchmarks.

## Verification

`cargo test -p baseten-parsers` checks registry coverage, upstream adapter parity,
GLM Unicode partitions, EOF, independent choices, token input, and lifecycle.
`lib/bindings/python/tests/test_b10_parsers.py` checks the installed binding.
Model grammar conformance remains in upstream; these adapter tests are not proof
of compatibility with every existing Baseten model configuration.

## Standalone reasoning extraction

`ReasoningParserStream` exposes the existing in-tree `dynamo-parsers` reasoning
registry through the same Python module. It reuses those implementations directly;
no copied grammar or additional parser state machine is needed. Its families are
listed in `REASONING_PARSER_FAMILIES` and are independent of the pinned v2 tool and
unified registries (and their `PARSER_UPSTREAM_REVISION`).

```python
from dynamo.parsers import ReasoningParserStream

# The prompt already contains <think>.
parser = ReasoningParserStream("deepseek_v4", in_reasoning=True)
first = parser.step("Let me check.</thi")
second = parser.step("nk>The answer is 42.")
tail = parser.finish()
reasoning_content = "".join(x.reasoning_text for x in (first, second, tail))
content = "".join(x.normal_text for x in (first, second, tail))
```

Outputs are deltas with delimiter tokens removed. Keep special tokens in decoded
input so the parser can recognize them. Call `finish()` to flush incomplete
markers at EOF, and create one parser per response choice. `in_reasoning=None`
retains the model default; a boolean invokes the existing parser's initial-state
override (support is family-dependent). Unknown families raise `ValueError`.
`step(text, token_ids=...)` optionally passes the corresponding chunk's token IDs
for token-aware families. Rust parsing releases the GIL.

Feed `normal_text` into a separate tool parser when using a sequential pipeline.
Do not apply this again to reasoning already extracted by `UnifiedParserStream`.
This API makes the backend available to Python callers; it does not automatically
replace SGLang/vLLM frontend parser selection.
