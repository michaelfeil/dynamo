# Baseten parsers

`baseten-parsers` provides one request-scoped Rust stream for ordered visible
text, reasoning, and tool-call events. Python exposes it as
`dynamo.parsers.UnifiedParserStream`. Construct one stream per response choice,
call `step` for each decoded text delta, and call `finish` at end of stream.

The `backend` argument selects the Rust parser implementation:

- `dynamo` (default) uses `dynamo-parsers-v2` 0.6.3 from frontend-crates revision
  `bb20dd01b6257cff9b739be2b18e7a444980d04a`. Available families are in
  `UNIFIED_PARSER_FAMILIES`.
- `vllm` uses vLLM's native unified Rust parsers at revision
  `f84325c48c0acc1e3703103788c5f2976e719762`. Available families are in
  `VLLM_UNIFIED_PARSER_FAMILIES`. Supply a local `tokenizer.json` path with
  `tokenizer_path`; vLLM uses it to resolve model markers and prompt state.
  Its native unified families are `gemma4`, `hy_v3`, `hy_v4`, `inkling`, and
  `kimi_k3`. vLLM currently accepts `prompt_token_ids` and native tool output;
  the Dynamo-specific `starting_state`, guided JSON, and invalid-payload
  policies are rejected for this backend.

```python
from dynamo.parsers import UnifiedParserStream

parser = UnifiedParserStream("qwen3", tools=tools)
# For vLLM: UnifiedParserStream("gemma4", tools=tools,
#                               backend="vllm", tokenizer_path="/model/tokenizer.json")
events = parser.step(delta_text)
events += parser.finish()
# event.kind: text | reasoning | tool_call
# event.text for text/reasoning; event.call for tool_call
```

Tool definitions use the flat upstream shape, with `name`, `parameters`, and
optional `description` and `strict`. Tool argument fragments are ordered and
must be concatenated by call index. A `complete` call delta marks closure.
`preserve_special_tokens` indicates whether the decoder must retain marker
text. `ParserStreamError.events` contains events committed before an error;
errors and `finish()` close the stream.

`ToolCallStream` remains importable from `dynamo.parsers` and `dynamo._core`
for compatibility, but construction raises `RuntimeError`. The standalone
`ReasoningParserStream` has been removed. Importing `dynamo.parsers` itself
does not load the native extension; the extension loads when a live parser,
constant, or exception is requested.

Run `cargo test -p baseten-parsers` for the Rust adapter checks. The Python
binding smoke checks are in `lib/bindings/python/tests/test_b10_parsers.py`.
