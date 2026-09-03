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

`RouterGuardClient` is the transport seam for custom clients and deterministic
tests. `JsonRouterGuardClient` adapts Dynamo's JSON `PushRouter`.

The Python extension in `lib/bindings/python` only converts Python values,
adapts the response stream, and exposes the Rust admission result as PyO3
classes. Changes to routing, preflight, reroute, cancellation, and guard cleanup
belong in this crate.
