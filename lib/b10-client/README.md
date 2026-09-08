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
    primary_worker_client=worker_client,
    primary_router_client=router_client,
    model_name=model_name,
    kv_block_size=32,
    disagg_request_id_machine_id=0,
)
endpoint_url = await local.serve(host="0.0.0.0", port=8080)
# GET /health and POST /v1/coordinate are now served by Rust/Hyper/Axum.
await local.shutdown()
```

The Baseten deployment starts this listener by default; `port: null` disables
HTTP while retaining direct binding calls.

The latency-sensitive path stays entirely native: Hyper receives the body,
Prost decodes it, the Rust coordinator routes it, and Hyper streams framed
responses. There is no Uvicorn/FastAPI server and no per-request PyO3 crossing.

The remote constructor deliberately accepts a named endpoint map but currently
requires exactly one entry. This preserves the configuration surface for the
future multi-endpoint load/session-aware selector without adding a lookup to
today's single-endpoint request path.

The wire schema (compiled by Prost during the build) is
[`proto/generation_coordinator.proto`](proto/generation_coordinator.proto).
Request IDs, models, tokens, routing, and sampling are semantically
required and are checked at the protocol boundary. LoRA, multimodal data,
session/cache affinity, and trace extensions remain optional. Sampling is one
required MessagePack map because its backend-specific shape evolves
independently; routing, LoRA, multimodal, trace, and admission data are typed
protobuf fields.
Worker requests always enable streaming; the prefill worker applies its own
phase-specific override. `routing.session_id` carries the worker's `user` value
once, rather than duplicating it in the request body.
Endpoint probing and selection are not part of the single-endpoint protocol.

`RouterGuardClient` is the transport seam for custom clients and deterministic
tests. `JsonRouterGuardClient` adapts Dynamo's JSON `PushRouter`.

The Python extension in `lib/bindings/python` only converts Python values,
adapts the response stream, and exposes the Rust admission result as PyO3
classes. Changes to routing, preflight, reroute, cancellation, and guard cleanup
belong in this crate.
