# Deterministic mocker output

`mocker_config.output_text` is an optional field in the existing `BasetenExt.mocker_config` map.
It contains raw generated text, including the model family's reasoning and tool markers.

1. The frontend encodes it with the deployed tokenizer, preserving special tokens,
   without applying a chat template or adding BOS/EOS tokens.
2. Internal requests carry `mocker_config.output_token_ids`; the worker never tokenizes text.
3. The mocker emits these IDs on its normal scheduler signals. The existing frontend
   detokenizer, parser, and HTTP/SSE writers process the output.

Omitting the field preserves random generation. An empty string returns no tokens.
Output ends with `stop` at the end of the supplied sequence, or `length` when
`max_tokens` truncates it. Supplied output overrides synthetic reasoning markers and
Baseten's `force_max_tokens` benchmark setting. Normal frontend stop handling still applies.

The Baseten disaggregated mocker suppresses prefill tokens; decode emits the entire
sequence. The same request field is forwarded to both workers. No replay offset is needed.
GPU workers do not consume this field.

Example for a deployment configured with the Qwen parser and a declared `weather` tool:

```json
{
  "model": "deployed-model",
  "messages": [{"role": "user", "content": "Weather in Paris?"}],
  "tools": [{"type": "function", "function": {
    "name": "weather", "parameters": {"type": "object", "properties": {
      "city": {"type": "string"}
    }, "required": ["city"]}
  }}],
  "max_tokens": 256,
  "stream": true,
  "mocker_config": {
    "output_text": "<think>Look up weather.</think><tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>"
  }
}
```

Tests should check both streaming and non-streaming responses, reasoning isolation,
tool names/arguments, usage, truncation, and empty output. Text fixtures exercise the
deployed tokenizer/parser family; recorded token fixtures remain necessary for exact
engine chunk boundaries or tokenizations that text cannot reproduce.

For the Baseten deployment smoke test, configure `tool_call_parser: qwen3` and
`tool_call_parser_source: dynamo`, then run from the MP project:

```bash
DYNCI_MOCKER_PARSER_FAMILY=qwen3 DYNCI_REPLICA_URL=http://localhost:8000 \
  python -m pytest tests/end2end/test_tool_calling.py -k mocker_supplied_tool_output
```

Install a Dynamo runtime containing the new fields and mocker implementation on both
frontend and workers before enabling this test. Older runtimes may ignore unknown fields.
