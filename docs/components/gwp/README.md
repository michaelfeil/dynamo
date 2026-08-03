---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Global Workload Plane
---

# Global Workload Plane

The Global Workload Plane (GWP) routes OpenAI-compatible requests across
multiple Dynamo ingress endpoints. It combines endpoint-local planner state
with stable session affinity and a shared scheduler event plane.

GWP selects a routable ingress endpoint, not an individual worker behind that
endpoint. It schedules the workers advertised by each endpoint's planner and
maps the provisional worker to its owning ingress. The selected endpoint's
local Dynamo router remains responsible for the final worker choice.

> GWP is currently an experimental proof of concept. The configuration and
> control APIs may change.

## Capabilities

- Routes multiple model families to different endpoint pools.
- Uses hot-reloaded model routes as the authoritative external model catalog.
- Aggregates `/deep/health` planner state into endpoint-level load signals.
- Preserves session affinity in shared etcd or Redis using planner worker IDs.
- Filters endpoints by Alyx's dimensioned
  `x-baseten-model-apis-routing-requirements` contract.
- Hot-reloads endpoint and route configuration.
- Synchronizes scheduler lifecycle events between GWP replicas over ZMQ, with
  replica discovery through etcd.
- Provides Envoy gRPC ext-authz scheduling with Rust Wasm lifecycle hooks.

## Prerequisites

- A Rust toolchain capable of building the Dynamo workspace.
- An etcd server reachable by all GWP replicas.
- One or more Dynamo ingress endpoints exposing the OpenAI-compatible APIs.
  Successful responses must include
  `x-baseten-dyn-worker-id`.
- A planner `/deep/health` endpoint for each configured ingress endpoint.
- Envoy with gRPC ext-authz, Proxy-Wasm, and dynamic forward proxy support when
  using the supplied adapter.
- `curl` and, optionally, `jq` for the examples below.

All URLs are accessed directly from the GWP process. In Kubernetes, use service
DNS names that resolve from the GWP pod or development environment; `kubectl`
port forwarding is not required.

## Build from source

From the Dynamo repository root:

```console
cargo build -p dynamo-gwp --features server,etcd,redis --bin dynamo-gwp
```

Run the unit tests with:

```console
cargo test -p dynamo-gwp --features server,etcd,redis
```

The ignored etcd integration tests can use a local server:

```console
DYN_GWP_TEST_ETCD=http://127.0.0.1:2379 \
  cargo test -p dynamo-gwp --features server,etcd \
    session::etcd_store::tests::etcd_roundtrip -- --ignored
```

The ignored Redis integration test uses `DYN_GWP_TEST_REDIS`:

```console
DYN_GWP_TEST_REDIS=redis://127.0.0.1:6379/ \
  cargo test -p dynamo-gwp --features server,redis \
    session::redis_store::tests::redis_roundtrip_and_expiry -- --ignored
```

## Start etcd locally

The following starts a single-node development server:

```console
etcd --name gwp-local \
  --data-dir=/tmp/gwp-etcd \
  --listen-client-urls=http://127.0.0.1:2379 \
  --advertise-client-urls=http://127.0.0.1:2379 \
  --listen-peer-urls=http://127.0.0.1:2380 \
  --initial-advertise-peer-urls=http://127.0.0.1:2380 \
  --initial-cluster=gwp-local=http://127.0.0.1:2380
```

An existing etcd cluster is also suitable. Set `ETCD_ENDPOINTS` to a
comma-separated list of endpoints. The discovery connection honors the standard
Dynamo `ETCD_AUTH_*` authentication and TLS environment variables.

## Configure endpoints and routes

Create a configuration file such as `/tmp/gwp.yaml`:

```yaml
endpoints:
  kimi-k2-a:
    ingress_url: http://kimi-k2-a.dynamo.svc.cluster.local/v1
    planner_url: http://kimi-k2-a-planner.dynamo.svc.cluster.local/deep/health

  glm-4-7-a:
    ingress_url: http://glm-4-7-a.dynamo.svc.cluster.local/v1
    planner_url: http://glm-4-7-a-planner.dynamo.svc.cluster.local/deep/health
    properties:
      region: [us]
      compliance: [hippa]

  glm-4-7-b:
    ingress_url: http://glm-4-7-b.dynamo.svc.cluster.local/v1
    planner_url: http://glm-4-7-b-planner.dynamo.svc.cluster.local/deep/health
    properties:
      region: [canada]
      compliance: [standard]

routes:
  - models: [kimi-k2]
    endpoints: [kimi-k2-a]
  - models: [glm-4.7]
    endpoints: [glm-4-7-a, glm-4-7-b]

served_alias_model_map:
  glm-4.7-preview: glm-4.7

session:
  ttl_secs: 600
  prompt_hash_fallback:
    token_position: 100000
  backend:
    type: etcd
    endpoints: [http://127.0.0.1:2379]

routing:
  pseudo_stride: 4
  block_size: 32
  planner_staleness_grace_secs: 30
  replica_warmup_secs: 90
  shutdown_grace_secs: 30
  approx_indexer_ttl_secs: 120
```

`ingress_url` must use the exact path `/v1`; GWP rejects other paths during
configuration validation because Envoy does not rewrite the inference path. A route constrains its listed models
to the referenced endpoint pool. Every routable external model name must be
listed in a route. Update the ConfigMap to add or remove a model alias; GWP
hot-reloads the catalog without calling downstream `/v1/models`.

`served_alias_model_map` maps a public, legacy, or deployment-advertised name
to exactly one canonical routing model. GWP uses the canonical model for
endpoint eligibility, tokenizer selection, prompt-derived affinity, and
metrics, while forwarding the original request body unchanged. GWP's
configured routes accept both canonical names and aliases. Aliases may remain
in `routes.models`; every member is canonicalized before the endpoint sets are
merged, so deployment-specific served names form one load-balancing group.
Alias chains are rejected.

When no recognized session header or string-valued OpenAI `user` is present,
`session.prompt_hash_fallback.token_position` derives affinity from the four
consecutive rolling routing hashes ending at that prompt position. The key is
model-scoped and stable as turns append beyond the cutoff. Shorter prompts
continue to receive a minted session ID. The position counts real tokens for
models with real tokenization and pseudo tokens otherwise.

`x-baseten-model-apis-routing-requirements` is a JSON object encoded as an
HTTP header value. Dimensions AND together and values within one `required`
list OR together:

```console
curl -H 'x-baseten-model-apis-routing-requirements: {"compliance":{"required":["hippa"]},"region":{"required":["us","canada"]}}' ...
```

This requires `hippa` compliance and either the `us` or `canada` region. The
filter is additive to model routes and liveness. A missing header or `{}` adds
no constraint, so headerless traffic can use any otherwise eligible endpoint.
Invalid JSON returns 400. A valid requirement with no live matching endpoint
returns 503. `preferred` arrays are accepted for schema compatibility but are
not scored yet. The Envoy adapter fails closed instead of using its default
endpoint whenever this header is present and scheduling fails.

For planner state, an HTTP error, malformed response, or
`detailed_load_data: null` retains the last usable snapshot for
`planner_staleness_grace_secs`. A successful response with an empty
`detailed_load_data` map immediately leaves the endpoint with no routable
candidate. An explicit `healthy: false` is an authoritative cordon and also
removes the endpoint immediately.

GWP watches the configuration file and atomically applies valid endpoint,
route, and endpoint-property changes. Add or remove an endpoint and
its route references in the same file update. Existing sticky bindings are
rechecked against the current routing requirements on every request.
`routing.block_size`, discovery through `ETCD_ENDPOINTS`, and the session etcd
connection are initialized at process startup.
`routing.replica_warmup_secs` and `routing.shutdown_grace_secs` are also
startup-only; restart replicas when changing those settings.

## Probes, replica convergence, and shutdown

GWP implements the standard gRPC health protocol on port 8091. Service
`gwp-liveness` remains serving while the process and Tonic server can make
progress. Service `gwp-readiness` becomes serving only after the first complete
planner reconciliation, at least one worker is routable, every configured
route has a live endpoint, and any replica warm-up has elapsed.

When another GWP replica is already registered in etcd at startup, the new
replica subscribes to ZMQ lifecycle events and remains unready for
`replica_warmup_secs` (90 seconds by default). A first replica does not incur
that delay. Planner polling continues during warm-up, so delayed load converges
from planner ground truth while add/prefill/free events from other GWP replicas
are accumulated.

ZMQ is live pub/sub, not a replay log. The warm-up establishes a bounded
observation window; it cannot recover a prefix lifecycle event that completed
before the new replica subscribed. This is why readiness combines the warm-up
with current planner load rather than claiming a complete historical snapshot.

Probe GWP directly with Kubernetes gRPC probes. Envoy has separate HTTP probes,
so pod readiness covers both containers:

```yaml
startupProbe:
  grpc:
    port: 8091
    service: gwp-liveness
  periodSeconds: 2
  failureThreshold: 30
readinessProbe:
  grpc:
    port: 8091
    service: gwp-readiness
  periodSeconds: 2
  failureThreshold: 2
livenessProbe:
  grpc:
    port: 8091
    service: gwp-liveness
  periodSeconds: 10
  failureThreshold: 3
```

On SIGTERM or Ctrl-C, GWP first marks `gwp-readiness` not serving and rejects
new schedules. It continues accepting lifecycle RPCs for existing Envoy
requests for up to `shutdown_grace_secs`. At the deadline it explicitly frees
any remaining scheduler bookings, then stops gRPC and unregisters its ZMQ
publisher from etcd. Set the pod
`terminationGracePeriodSeconds` greater than `shutdown_grace_secs` (for example,
45 seconds for the 30-second default).

## How request selection works

GWP balances a non-sticky request in five stages:

1. Parse `x-baseten-model-apis-routing-requirements` and retain endpoints
   satisfying every required routing dimension.
2. Filter to live endpoints configured for the requested model.
3. Honor a shared affinity binding when that worker's endpoint is still live
   and model-eligible.
4. Otherwise, run the existing Dynamo selector using the replica-local
   approximate prefix index, planner-observed baseline load, and GWP in-flight
   load synchronized across replicas.
5. Forward to the selected ingress, whose local Dynamo router chooses the
   internal worker. On a successful response,
   `x-baseten-dyn-worker-id` identifies the actual worker.

When step 3 finds a live affinity binding, GWP also forwards
`x-gwp-affine-worker-id: <worker-id>` to the selected deployment. The header
reports the worker confirmed for that session on a previous request; it is
absent for newly minted sessions, affinity misses, and normal scored
selections. It is a routing hint for the deployment, not a claim about which
worker will serve the current request.

The initial scheduler booking is provisional. If the reported worker differs,
GWP frees the provisional booking, re-books the same workload on the reported
worker, and attributes the prefix and session binding to that worker. Planner
worker IDs survive GWP config reloads and restarts. If a worker or endpoint is
removed, its bindings become stale and those sessions fall through to normal
selection; other bindings remain intact.

A sticky binding is always rechecked against the current request's required
properties. If the bound endpoint lacks one, the request unsticks and returns
to normal eligible-endpoint scoring.

## Run one replica

```console
DYN_GWP_CONFIG_PATH=/tmp/gwp.yaml \
DYN_GWP_GRPC_PORT=8091 \
DYN_SYSTEM_PORT=9090 \
ETCD_ENDPOINTS=http://127.0.0.1:2379 \
  target/debug/dynamo-gwp
```

The process serves scheduling, lifecycle, and standard health over gRPC on
port 8091, plus `:9090/metrics` from the standard Dynamo system server. Client
inference traffic always enters through Envoy.

The GWP metrics include routed and denied request counters plus
`dynamo_component_gwp_time_to_first_byte_seconds` (provisional booking to
upstream headers) and `dynamo_component_gwp_request_duration_seconds`
(provisional booking through final scheduler free). Routed counters and histograms carry `model`, `routed_endpoint`, and
sanitized `downstream_authority` (`host:port`) labels. Response metrics add
`outcome`: `success`, `overloaded` (429/503/529), `client_error` (other 4xx),
or `upstream_error` (all other non-200 and no-response failures).
`dynamo_component_gwp_request_outcomes_total` records the terminal category
after final free, including transport failures that never produced response
headers. `dynamo_component_gwp_affinity_lookups_total{backend,outcome}`
distinguishes `hit`, `miss`, `worker_unavailable`, `ineligible`, and
`backend_error`.
`dynamo_component_gwp_affinity_operation_duration_seconds{backend,operation,result}`
provides per-operation backend latency with bounded `success`, `error`,
`timeout`, and `bulkhead_rejected` results. Scheduling waits only on
`operation="peek"`; writes and removals run outside the request-critical lookup
path. Session and worker IDs are intentionally excluded from affinity labels.

Cluster-level scheduler gauges aggregate this replica's request ownership:

- `dynamo_component_gwp_scheduler_inflight_requests{routed_endpoint,aggregation}` exposes
  `total` and `per_live_worker`.
- `dynamo_component_gwp_scheduler_inflight_tokens{routed_endpoint,cache_status,aggregation}`
  exposes total input tokens plus the scheduler's approximate `cached` and
  `uncached` split, both as totals and per live planner worker.
- `dynamo_component_gwp_scheduler_live_workers{routed_endpoint}` exposes the normalization
  denominator.

Sum `aggregation="total"` across GWP replicas. Because every replica observes
the same planner worker set, use `max` for
`dynamo_component_gwp_scheduler_live_workers`; summing
the `per_live_worker` series across replicas yields the deployment-wide
per-worker load.

The same scrape also includes the generic scheduler state registered by
`KvRouter`: `dynamo_frontend_worker_active_decode_blocks`,
`dynamo_frontend_worker_active_prefill_tokens`,
`dynamo_frontend_worker_active_requests`,
`dynamo_frontend_worker_active_prefill_requests`,
`dynamo_frontend_worker_active_decode_requests`, and
`dynamo_frontend_router_queue_*`. Worker gauges are labeled by downstream
planner `worker_id`, `dp_rank`, and `worker_type`.

Verify the metrics server:

```console
curl -fsS http://127.0.0.1:9090/metrics
```

Send a non-streaming request through Envoy:

```console
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-session-id: setup-check' \
  -d '{"model":"glm-4.7","messages":[{"role":"user","content":"Reply with OK"}],"stream":false}'
```

Use `curl -N` and set `"stream":true` to test streaming.

## Run multiple replicas

Start every replica with the same configuration and `ETCD_ENDPOINTS`, but a
different gRPC and metrics port when sharing a host network:

```console
DYN_GWP_CONFIG_PATH=/tmp/gwp.yaml \
DYN_GWP_GRPC_PORT=8092 \
DYN_SYSTEM_PORT=9091 \
ETCD_ENDPOINTS=http://127.0.0.1:2379 \
  target/debug/dynamo-gwp
```

GWP enables router replica synchronization and uses ZMQ for its scheduler event
plane. etcd discovers the GWP router event channels; it does not carry the
scheduler event payloads. Each replica dynamically binds a ZMQ TCP publisher
and advertises a reachable host or pod address in etcd. Network policy and
firewall rules must therefore allow peer-to-peer TCP connections to the
advertised ports.

Startup logs should include `discovery_backend=etcd` and `event_plane=Zmq`.
Peer discovery is visible through `Discovery Added` and
`Connecting to new ZMQ publisher` messages.

Replica sync shares scheduler lifecycle events such as request admission,
prefill completion, and request release. The approximate radix KV index remains
replica-local, and each replica independently refreshes configured model routes
and planner state. The core in-flight request table is also local, so all
lifecycle calls for one request must continue to reach the same GWP control
replica. Put another way, replica sync complements request pinning; it does not
replace it.

Session affinity is separate from replica sync. All replicas consult the
configured session etcd store, so a session continues to select the same
endpoint while that endpoint remains eligible. When an endpoint disappears,
normal selection reassigns only sessions that were bound to that endpoint.

## Use the Envoy adapter

The repository includes a development Envoy configuration under
[`deploy/gwp/poc`](../../../deploy/gwp/poc/README.md). From the repository root:

```console
envoy --mode validate -c deploy/gwp/poc/envoy.yaml
envoy -c deploy/gwp/poc/envoy.yaml
```

It listens on `0.0.0.0:8080`, schedules and sends lifecycle events through
separate gRPC pools targeting `127.0.0.1:8091`, and dynamically forwards to the
selected authority. Local clients use `127.0.0.1:8080`.

For a production multi-replica deployment, the adapter or its upstream load
balancer must pin every scheduler lifecycle call for a request to the same GWP
replica.

Verify the adapter. The public listener intentionally returns 404 for
`/v1/models`; configured routes are authoritative:

```console
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-session-id: envoy-check' \
  -d '{"model":"glm-4.7","messages":[{"role":"user","content":"Reply with OK"}],"stream":false}'
```

## Deploy on Kubernetes

The Kubernetes base under
[`deploy/gwp/kubernetes`](../../../deploy/gwp/kubernetes/README.md) runs two
replicas with Envoy and GWP in the same pod. It includes the workload Service,
metrics port, readiness/liveness probes, rolling-update policy, pod disruption
budget, ConfigMap hot reload, and graceful-termination timings.

Render it from `deploy/gwp/kustomization.yaml` after setting the GWP and
approved Envoy images. The public Service exposes Envoy, not GWP's gRPC
control port. See the Kubernetes README for etcd and ZMQ network requirements.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| GWP cannot initialize discovery | Confirm `ETCD_ENDPOINTS`, credentials, TLS settings, and network reachability. |
| A model has no routable endpoint | Check the configured route, model spelling, planner payload usability, and planner staleness. |
| Replicas do not synchronize | Confirm the same etcd cluster/namespace is used and that advertised ZMQ TCP ports are reachable between replicas. |
| Envoy returns `502` or `503` | Check GWP health, ext-authz outcomes, and the selected ingress authority. |
| A session changes endpoint unexpectedly | Send a stable `x-session-id`, confirm the response succeeded with `x-baseten-dyn-worker-id`, and check the shared session backend. |

Session affinity is configured independently from replica discovery:

```yaml
session:
  ttl_secs: 600
  prompt_hash_fallback:
    token_position: 100000
  backend:
    type: redis
    url: rediss://user:password@redis.example.com:6379/
    key_prefix: "gwp:affinity:"
```

Redis uses native key expiry and an automatically reconnecting multiplexed
connection. `redis://` and TLS `rediss://` URLs are supported. Alternatively,
select `type: etcd` with an `endpoints` array; legacy
`session.etcd_endpoints` remains accepted. GWP must be built with the matching
`redis` or `etcd` feature, and startup fails if the selected client cannot
initialize. Runtime backend errors are affinity misses and requests fall
through to normal KV/load-aware routing; GWP never creates replica-local
fallback affinity. Backend selection is startup-only and requires a pod
restart; `session.ttl_secs` and `session.prompt_hash_fallback` can hot-reload.

## Related documentation

- [Global Workload Plane design](../../design-docs/global-workload-plane-design.md)
- [GWP crate overview](../../../lib/gwp/README.md)
- [Composer live-cluster PoC](../../../deploy/gwp/poc/README.md)
