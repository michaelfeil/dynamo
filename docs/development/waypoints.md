# Inspect the real request path with Waypoints

Waypoints runs a request through the normal HTTP handler and returns one selected
boundary. It is an internal debugging endpoint, not another inference protocol or
a general-purpose pipeline runner. Chat Completions, Responses, and Messages use
the same validation, defaults, translation, and response writers as public requests.
Both listeners use the same router value, including its middleware layers.

The listener starts with `HttpService` on port **9192**. Set `DYN_WAYPOINTS_PORT`
to change it or `DYN_WAYPOINTS_DISABLE=1` to disable it. It uses the service's configured
host (default `0.0.0.0` for pod-network access), without authentication. Keep it off public Services and
gateways; a containerPort declaration alone is not an access-control boundary.
Only use it from a trusted internal network or through a port-forward. A failed
diagnostic bind is logged and does not take down inference.
The listener starts only if Chat, Responses, or Messages is enabled at service
startup. Metrics-only services (including kserve's HTTP sidecar) do not open it.
Rust-preprocessor frontends still support `canonical`; later boundaries require
the Python hook and otherwise fail before engine dispatch. An explicit loopback
service host is respected. Pre-bound main listeners do not override the configured host.
Disable it on hosts where pod/direct-network callers are not trusted. Container
ports do not enforce isolation; use network policy/firewalls to restrict callers.
Automated callers resolve frontend pod IPs (requiring namespace pod-list access),
handle pod replacement, and may pin a replica for a comparison; do not expect a Service.

```sh
curl http://localhost:9192/v1/waypoints -H 'Content-Type: application/json' -d '{
  "protocol": "chat",
  "request": {"model": "my-model", "messages": [{"role":"user","content":"Hello"}]},
  "stop_after": "render",
  "timeout_ms": 30000
}'
```

`protocol` is `chat`, `responses`, or `messages`. The nested `request` JSON is
forwarded unchanged to the production handler; its normal deserialization still
determines which fields are retained. This is not byte-for-byte HTTP replay.
Optional `headers` contains single-valued application headers.
Hop-by-hop, connection-nominated, Host and Content-* headers are discarded because
the adapter constructs a new JSON message. Strip captured
`X-Baseten-Request-Rate-Limit-Percentage` and `X-Baseten-Token-Rate-Limit-Percentage`
when sampling requests: the production check is read-only (no quota debit)
but may reject stale captured usage with 429. The diagnostic call's
own headers are not treated as captured headers. Successful responses contain
`{"stage": "...", "value": ...}`. Once the envelope is valid, a processing error
is a terminal `error` artifact with HTTP 200, not a failed inspection call.

Set `preserve_intermediates` to also return every boundary before `stop_after`
that the run passed through, keyed by stage under `intermediates` in pipeline
order:

```jsonc
{
  "stage": "client", "value": {"status": 200, "content_type": "...", "body": {}},
  "intermediates": {
    "canonical": {"request": {}, "context": {}},
    "render": "...", "tokenize": [1, 2], "engine_request": {},
    "engine_output": [], "chat_stream": []
  }
}
```

One call then carries the whole request path instead of one artifact per call,
which matters when the later boundaries must describe the same generation as the
earlier ones: separate calls each rerun ingress and preprocessing, and any call
that reaches the engine samples again. Boundaries after `stop_after` never ran
and are absent; the requested boundary is the artifact and is not repeated.
`render` on a fused token renderer is preserved as `null` rather than failing a
run that did not ask to stop there. Preserving `engine_output` buffers the engine
stream before the parsers see it — postprocessing is chunk-by-chunk either way,
so the parsed result is unchanged, but this is a further reason not to read
Waypoints timings as latency. Everything preserved counts against the same 4 MiB
capture bound, so a large request may need a nearer `stop_after`.

### Failed runs

```json
{
  "stage": "error",
  "value": {
    "requested_stage": "client",
    "status": 400,
    "content_type": "application/json",
    "body": {"message": "Invalid request"}
  },
  "intermediates": {
    "canonical": {"request": {}, "context": {}},
    "render": "prompt before failure",
    "tokenize": [1, 2]
  }
}
```

The original handler status and body are data in `value`; JSON error bodies are
not reinterpreted as a second error schema. A text error body remains a string.
`requested_stage` names the intended exit, **not** the exact operation that failed.
Only completed boundaries received before the error appear in `intermediates`;
no later work runs and there is no retry or live-generation fallback. The map is
omitted when `preserve_intermediates` is false. Python flushes completed artifacts
before an ordinary exception crosses the existing error bridge.

Run deadlines (status 408), capture limits (413), and malformed hook results (502)
also produce terminal error artifacts. On overflow, the response keeps the longest
complete intermediate prefix that fits and sets `value.capture_truncated: true`
if artifacts or the original error body must be omitted. The **entire serialized
response**, including its envelope, is limited to 4 MiB. Cancellation still stops
generation; a timeout can only include artifacts already received by Rust, not
unflushed Python-local state. A disconnected caller receives no result.

Invalid Waypoints envelopes/controls are still HTTP 4xx (including an oversized
input body); they never start an inspected run. `error` is an output state, not
a valid `stop_after`. An error encoded inside an HTTP-200 SSE body remains stream
content in the `client` artifact, matching the production writer.

Protocol/stage controls and Rust-owned artifacts use Serde types. Unknown control
values or malformed public envelopes return 422. Malformed Python hook results
produce an error artifact with status 502 and a `Waypoints hook:` message naming the expected stage or chunk;
they never fall back to live generation. Render artifacts contain text, tokenize
artifacts contain unsigned 32-bit IDs, and chat-stream artifacts use the production
Chat Completions chunk schema. Engine objects remain opaque Python-owned JSON.
Client bodies retain the production writer's JSON or UTF-8 stream text.

Live worker-routing response headers are not collected or published by diagnostics;
replaying a captured ID must not consume or overwrite a live request's header data.

| `stop_after` | Returned value | Work not run |
| --- | --- | --- |
| `canonical` | Engine-bound Chat Completions request and protocol context/losses | Python preprocessing and generation |
| `render` | Actual rendered prompt text | Tokenization and everything after it |
| `tokenize` | Actual prompt token IDs | Remaining preprocessing and generation |
| `engine_request` | Fully prepared engine request and routing arguments | Coordinator dispatch and postprocessing |
| `engine_output` | Raw engine-response objects | Chat parsing and client conversion |
| `chat_stream` | Actual Chat Completions chunks | Protocol response conversion |
| `client` | Status, content type, and actual JSON body or SSE text | Nothing; runs the complete path |

Boundaries after `canonical` need the served Python integration registered through
`HttpService.set_waypoints_hook`. Without it, requests fail before invoking a live
engine. The adapter calls the ordinary `FrontendProcessor.infer`; a private
request-local field selects an early exit. External headers/JSON cannot install
that field. There is no process-wide capture registry or request-ID lookup.

## Replay

Add `engine_output` containing a previous raw engine-output artifact's `value`.
This replaces only coordinator generation. The original request still passes
through real ingress and preprocessing; the real postprocessor and protocol
writers process the replay. A configured tokenizer, model configuration, and
schema compiler are still needed. GPU generation is not. Omitting `engine_output`
allows real generation when the selected boundary reaches it.

There is deliberately no arbitrary `start_at` graph, alternate reconstruction
implementation, token injection, or artifact-history store. Fused token renderers
(Harmony and Kimi K3) reject `render` because they have no intermediate string;
use `tokenize`. `/v1/completions` and native Rust token-engine inspection after
`canonical` are not supported by this Python adapter.

## Limits and effects

Input and capture are bounded to 4 MiB. `timeout_ms` defaults to 30 seconds and is
limited to 1–300000 ms; it covers handler execution and response collection.
Cancellation follows the production connection monitor. Work which does not
cooperate with cancellation cannot be forcibly interrupted mid-operation.

This is buffered diagnostics, not a latency benchmark. It is not side-effect-free:
live generation can route to workers, preprocessing may fetch media, and production
metrics and debug logs still run. The Python request-completion/accounting logger
is suppressed for inspected requests; processor metrics emit normally, including
replay token/cache counts and timings from the diagnostic run.
Rust request/loss/response metrics and worker-side effects still run; successful
early stops count as successful requests, not internal errors. Probes therefore
affect per-model success counters and request-duration histograms; a canary
organization does not isolate those model-only SLI labels. This is not an
end-to-end customer billing-isolation guarantee. Use an explicitly provisioned
canary identity and verify downstream accounting before canarying production.
Large multimodal artifacts are not a stable
replay format; encoded media is omitted from the engine-request view. Artifacts
may contain sensitive prompt/response data; do not publish them indiscriminately.
