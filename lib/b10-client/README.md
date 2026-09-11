# Dynamo B10 client

`dynamo-b10-client` implements Baseten's route, admission, worker-connect, and
KV-lifecycle protocol without depending on Python or PyO3.

The high-level Rust entry point is `RouterWorkerCoordinator`:

```rust,ignore
use dynamo_b10_client::{
    RequestContext, RouteOptions, RouterRequestNew, RouterWorkerCoordinator,
};

let client = RouterWorkerCoordinator::from_push_routers(
    router_push_router,
    worker_push_router,
    32,
)?;

let outcome = client
    .route_and_worker(
        RequestContext::new(context, trace_context, metadata),
        RouterRequestNew {
            tokens,
            ..Default::default()
        },
        worker_payload,
        RouteOptions::default(),
    )
    .await?;
```

Complete aggregate or prefill-first generation is owned by
`GenerationCoordinator`. Bindings provide already-serialized primary and
decode worker maps; the coordinator routes both legs, carries the prefill
handoff into decode, merges topology constraints, suppresses the decode
bootstrap, and owns both router guards for the lifetime of the returned stream.
If decode routing is denied after prefill admission, the denied outcome retains
the prefill worker and overlap metadata for failure-path observability.

`RemoteGenerationCoordinator` implements the same
`GenerationCoordinatorClient` interface for one exact HTTP URL. It sends a
protobuf `NewRequest` to `/v1/coordinate` and consumes a length-delimited stream
of protobuf response frames. Python callers select it without changing their
generation call site:

```python
coordinator = dynamo.GenerationCoordinator.remote(
    {"default": "http://generation-coordinator:8080/v1/coordinate"}
)
result = await coordinator.generate(
    context,
    routing_kwargs,
    worker_args,
    decode_worker_args,
)
```

The matching server is also implemented in Rust and merely lifecycle-managed
through PyO3. Python starts it during normal container initialization; it does
not decode protobufs or handle generation requests:

```python
local = dynamo.GenerationCoordinator(
    runtime=runtime,
    primary_worker_client="namespace.worker.generate",
    primary_router_client="namespace.router.generate",
    model_name=model_name,
    kv_block_size=32,
)
endpoint_url = await local.start()
# With a configured port, GET /health and POST /v1/coordinate are served by Rust.
# Runtime shutdown stops the listener; no coordinator context manager needed.
```

The local constructor requires `runtime` as a keyword argument, including when
passing explicit clients. Direct generation initializes clients lazily; `start()`
initializes the configured backend and optional HTTP listener. The listener stays
alive even if the Python coordinator handle is dropped, and runtime shutdown
initiates graceful HTTP shutdown. HTTP listening requires a runtime; there is no
separate coordinator shutdown method.

`GenerationCoordinatorRuntime` owns client initialization and HTTP lifecycle in
Rust. Python wraps it without owning startup or shutdown state. Rust callers use
the same `start()`, `generate()`, `is_client()`, and `is_server()` methods.

Local/remote mode and listener settings are fixed at construction. Only the
`remotes` endpoint map reloads. Rust callers can use
`RemoteGenerationCoordinator::from_config(reader)` for reloadable endpoints;
the HTTP client retains its connection pool across updates.

The configured constructor captures `DYNAMO_DEFAULT_GENERATION_COORDINATOR_URL` as its
default remote (`default`); its presence does not enable remote mode or imply
that a service is live. Explicit configmap `remotes` take precedence, including
after reload; removing them restores the captured URL. Explicit `.remote(...)`
clients do not use this environment fallback. Configured affinity still requires
explicit `remotes` in the configmap.

The constructor's `is_client_force: bool | None = None` overrides mode selection:
`False` keeps orchestration local and ignores the environment URL; `True` requires
remotes or an environment URL; `None` selects remote mode only when configmap
`remotes` are configured, otherwise local even when the environment URL is set.
In Rust, pass `Option<bool>` to `GenerationCoordinatorRuntime::new`. This does not
change the config reader or HTTP listener settings.

HTTP is disabled by default. Enable it in the mounted config:

```yaml
b10_generation_coordinator_config:
  port: 8080
```

`GET /v1/worker_loads` returns a flat JSON array of worker loads. Disaggregated
prefill and decode pools are queried concurrently and concatenated, retaining
`disaggregation_mode` (`prefill`, `decode`, or `prefill_and_decode`) on each row.
DP ranks are summed per worker within each pool; identical worker IDs across
pools remain separate rows. Like Baseten deep health, an idle one-token probe is
normalized to zero per rank.

```json
[{"worker_id":42,"disaggregation_mode":"prefill","potential_prefill_tokens":128,"potential_decode_blocks":8,"active_requests":2}]
```

An unavailable local router pool fails the request. Multi-remote relays return
the union of current worker snapshots, omitting unavailable/stale remotes (an
empty array when none are current), without filtering out incomplete disaggregated pools.
Queries are bounded to five seconds. Remote clients and HTTP relays forward to
the sibling `worker_loads` endpoint using the same HTTP connection pool and
reloadable remote URL as generation. Rust callers use `worker_loads()` on the
coordinator runtime or client. This endpoint shares the listener's trusted-network
access requirements; no new port or Python handler is introduced.

`POST /v1/bid` queries potential loads for a prompt and its cache identity. Both bodies are
unary `application/x-protobuf`: `BidRequest { tokens, mm_routing_args, cache_salt, session_id }`
and `BidResponse { affinity, prefill_tokens, decode_tokens }`, without generation stream framing. Rust callers use
`client.bid(request).await?` or `coordinator.bid(request).await?` with `BidRequestV1`.
Tokens must be nonempty; malformed/empty requests return 400. Each outgoing remote
bid and each local router query has its own five-second timeout. Options run
concurrently; timed-out/failed remote options are excluded and the best successful
bid is returned. If none succeeds, the response is 503. There is no competing
five-second timeout around the whole pool; allow response/transport overhead when
calling a relay. A relay used as another pool's option must still respond within
that caller's five-second option budget.

The coordinator returns its best aggregate worker or prefill/decode pair using
`(1 - 0.5 * affinity) * (prefill_tokens + 0.1 * decode_tokens)`. For now, local bids
always return `affinity=false`. Callers supply only `session_id`, the same key used
by `/v1/coordinate`; affinity decisions belong to the downstream coordinator, not
the caller or relay. The key is forwarded through relays and router request metadata
for future affinity resolution, without any affinity lookup or update during bidding.
Each DP rank
is a candidate; decode blocks are converted using the decode pool's block size.
Disaggregated routers are queried concurrently and both must have workers. The pair's
estimate combines prefill-pool prefill tokens with decode-pool decode tokens.
Bids send only `PotentialLoads`, with short caching disabled: no admission, worker
execution, capacity reservation, or affinity update. Tokens and MM routing info affect
the estimate. `cache_salt` is carried through relays but does not yet affect the
unchanged router RPC/cache. LoRA, worker restrictions, and MM payloads are not bid fields.

The latency-sensitive path stays entirely native: Hyper receives the body,
Prost decodes it, the Rust coordinator routes it, and Hyper streams framed
responses. There is no Uvicorn/FastAPI server and no per-request PyO3 crossing.

With one named remote, generation forwards directly without a destination search or
bid. Session metadata is still forwarded, and configured affinity still records
admissions and retains stream leases. Without configured affinity, singleton startup
does not initialize worker-load polling either.
Multi-backend pools reuse a live session-affinity assignment verified by five-second
worker-load polls. Both affinity and bidding require a current snapshot for the
configured URL with aggregate capacity or both prefill and decode workers.
New/recovered remotes enter selection after a successful poll. Otherwise they bid
concurrently against these live candidates,
discard failed bids, and choose the lowest score above. Ties prefer `default`, then
backend name. Generation worker restrictions prefilter remotes using polled membership.
Cold selection uses fresh bids, not the background load values. A failed bid does
not change inventory liveness; inventory probes maintain that independently. The configmap-backed
native runtime can enable Dynamo's synchronized session affinity store; see the
[coordinator configuration](../baseten-configmap/README.md#generation-coordinator-listener)
for the affinity scope, selection policy, and reload behavior. Call `start()` to
initialize polling before serving requests. The single-endpoint constructor retains
its direct HTTP path.

The wire schema (compiled by Prost during the build) is
[`proto/generation_coordinator.proto`](proto/generation_coordinator.proto).
Tokens and multimodal payloads travel separately from `worker_msgpack`, an
opaque MessagePack map containing all remaining worker kwargs. The codec
preserves unknown keys and binary values; the worker owns validation of model,
sampling, and backend options. The general map must exclude `tokens` and
`mm_args`. The coordinator reconstructs `tokens: {tokens: [...]}`, sets the
request ID, and attaches multimodal payloads only to the primary worker request.
Generation coordination supplies the routing response and disaggregation state.

Session identity and cache salt are projected into routing metadata for bids
while their original worker fields remain in the general map. Request IDs,
tokens, routing, and the worker blob are required at the protocol boundary.
This schema is breaking: clients and coordinators must be upgraded together.

`RouterGuardClient` is the transport seam for custom clients and deterministic
tests. `JsonRouterGuardClient` adapts Dynamo's JSON `PushRouter`.

The Python extension in `lib/bindings/python` only converts Python values,
adapts the response stream, and exposes the Rust admission result as PyO3
classes. Changes to routing, reroute, cancellation, and guard cleanup
belong in this crate.
