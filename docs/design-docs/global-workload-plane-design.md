---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Global Workload Plane Design
---

Architecture and design decisions for the **Global Workload Plane** (GWP): a
multi-replica, session-aware **external router** that places OpenAI-style
requests across independently routable *endpoints* (not internal workers) by reusing the
`dynamo-kv-router` crate's approximate indexer and selector primitives.

**Envoy is the data plane; GWP is the control plane.** Envoy terminates the
client connection, buffers/inspects the request, asks GWP where to send it,
proxies the bytes, and reports lifecycle events back. GWP only makes placement
decisions and maintains live endpoint state — it never touches the response
stream.

## Overview

Today the Dynamo KV Router picks the best *worker within a single deployment*
using prefix-overlap scoring against a radix-tree indexer populated by KV
events. The Global Workload Plane consumes an immutable topology snapshot and
maintains one isolated router/scheduler per canonical `oracle_version_id`. It
feeds each router only that oracle version's workers, provisionally picks a
worker, then returns that worker's owning endpoint authority to Envoy. The
current topology producer combines configured routes/endpoints with every
endpoint planner's worker and load observations. The selected
endpoint's own router remains responsible for the final internal-worker
choice. GWP reconciles that provisional booking from the worker ID returned in
the response. GWP is deployed as its own crate / component with multiple
replicas; each Envoy instance talks to its router replica over a persistent
local connection.

Stick/unstick, with the approximate router as the fallback and the
background truth-keeper:

The mental model for when a session sticks vs. falls through:

- **Stick** — honor the session iff `(session→worker binding exists in the
  configured affinity store)
  AND (that worker and its endpoint are still alive and serve the requested
  model)` in one topology snapshot. Route directly to that worker's
  endpoint; do not re-run `find_best_match` for the provisional decision.
- **Unstick** — if the bound worker or endpoint is gone or does not serve the
  requested model,
  binding is stale: dishonor it and fall through to the approximate router.
- **Everything else** (no session id, or no binding) → the approximate router:
  approximate-tokenize, filter by model, `find_best_match`, and map the
  provisional worker to an endpoint.

On a sticky hit, the optimistic-affinity path can return the bound worker
without waiting for tokenization when a tokenization permit is available. It
then tokenizes and books the request into that oracle version's scheduler in a
background task. If no permit is available, the same work is performed
synchronously before returning. The planner poll (Decision 4) remains the real
load source and corrects local accounting within ~1s. Prefix ownership is
recorded only after `ResponseStarted`, when the actual worker is known.

Both paths drive the scheduler lifecycle off **Envoy's stream events**:
`mark_prefill_completed` on `ResponseStarted` (first SSE response headers),
`free` on `RequestFinished` (stream close, reset, timeout, or client
disconnect).

Every successful remote response carries the internal Dynamo `WorkerId` that
actually served it in `x-baseten-dyn-worker-id`. If it differs from the
provisional choice, GWP frees the provisional scheduler booking and re-books
the same workload under a distinct accounting request ID on the actual worker.
Prefix ownership and session affinity are then attached to the actual worker,
while the same topology snapshot supplies the only addressable egress authority.

A background **endpoint reflector** polls **each endpoint's planner service**
(`planner_common.py`, one per dynamo graph) every ~1s — its `deep/health`
already returns a worker list with per-worker load (`detailed_load_data`,
keyed by u64 worker id). GWP publishes every worker with its load and owning
endpoint. An empty usable response makes the endpoint
unroutable immediately; transport, status, parse, and `null` failures retain
the last-known-good observation for a configurable grace period. This makes
the locally hosted `KvRouter` compare workers across endpoints while preserving
the endpoint-level egress control boundary.

A note on the resulting "two places" tradeoff: the routing decision now lives in
both the configured affinity store (Redis in production) and the approximate
router (for load modeling, block level awareness routing). This is deliberate —
the affinity store answers
"where did this session go last, and is that endpoint alive?" with a fast
shared-store read; the approximate router answers "given load and prefix
overlap, where *should* a new request go?". They are kept in sync via the
lifecycle interface below.

## Deployment shape: Envoy data plane + external router

```
                         ┌──────────────────────────┐
                         │ External router (GWP)    │
                         │                          │
                         │ - cache-aware routing    │
                         │ - session placement      │
                         │ - endpoint load estimates│
                         └────────────▲─────────────┘
                                      │
                   route decision + lifecycle events
                                      │
Client ───────────────► Envoy ────────┴──────────► Selected endpoint
                         │                              │
                         │                              ▼
                         │                         Actual worker
                         │                         e.g. worker 42
                         │
                         └──── streams SSE response back to client
```

### Request lifecycle

1. Client sends an LLM request to Envoy.
2. Envoy generates a request ID and buffers the request body.
3. Envoy's native gRPC ext-authz filter calls GWP. The `CheckRequest` carries
   the original OpenAI JSON in `raw_body` and the request path and headers in
   standard HTTP attributes:
   ```
   x-request-id: <request id>
   path: /v1/chat/completions
   raw_body: <original OpenAI JSON>
   ```
4. GWP checks affinity, cache state, and current load, then returns ext-authz
   request-header mutations:
   ```
   x-gwp-internal-request-id: <request id>:<uuid>
   x-gwp-authority: cluster-b.example:443
   x-gwp-endpoint-id: cluster-b
   x-gwp-session-id: <resolved or minted session id>
   x-gwp-sticky: false
   ```
5. Envoy proxies the original request to that authority.
6. When the first SSE response arrives, Envoy observes
   `status: 200`, required `x-baseten-dyn-worker-id: 42`, other response headers,
   and asynchronously reports it with a unary gRPC lifecycle call:
   ```
   ResponseStarted {
     request_id: <request id>,
     actual_worker_id: 42,          # required on successful responses
     status: 200
   }
   ```
7. The router reconciles and advances its worker lifecycle:
   - if worker 42 differs from the provisional worker, free the provisional
     booking and add the same workload to worker 42 under a distinct accounting
     request ID
   - prefill for the confirmed booking is finished (`mark_prefill_completed`)
   - active decode accounting begins
   - on success, record prefix ownership and bind the session to worker 42
8. When the SSE stream closes, resets, times out, or the client disconnects,
   Envoy sends another unary gRPC lifecycle call:
   ```
   RequestFinished {
     request_id: <request id>,
     reason: complete               # reset | timeout | client_disconnect
   }
   ```
9. The router releases the request's scheduler slot (`free`) and drops its
   in-flight state.

### Envoy filter hooks

Native ext-authz owns scheduling; the Rust Wasm filter owns lifecycle and
header hygiene:

```
ext_authz Check   → buffer the body, schedule through gRPC, and install route headers
request headers  → strip GWP-internal routing headers and capture scheduling metadata
response headers → observe status + worker ID, enqueue ResponseStarted, and
                   inject x-session-id / x-routed-endpoint
onLog()           → enqueue RequestFinished after the stream terminates
```

Envoy keeps HTTP, TLS, connection pooling, backpressure, retries, and SSE
streaming. The external router only makes placement decisions and maintains
live endpoint state.

### Envoy integration: gRPC scheduling and lifecycle

Scheduling uses Envoy's native gRPC ext-authz filter and GWP's Tonic service.
It is the only control call on the request critical path. The Rust Wasm filter
observes response headers and stream teardown, then asynchronously sends unary
`ResponseStarted` and `RequestFinished` gRPC calls through a VM-root shared
queue. GWP's per-request state machine accepts lifecycle events in either order
and reconciles a finish that arrives before response headers.

### Per-request state and replica pinning

Between schedule and `RequestFinished` the router keeps an
**in-flight entry** per request id:
`{session_id, provisional/confirmed decision, endpoint_id, tokens,
accounting_request_id}`. `ResponseStarted` needs these fields to correct the
booking; `RequestFinished` frees the current accounting request ID.

- **Replica pinning:** the in-flight table is per-replica, so all three
  lifecycle calls for one request must hit the same router replica. Each Envoy
  instance uses a persistent connection to one router replica (sidecar or 1:1
  pairing), which gives this for free. Cross-replica in-flight state is future
  work (session *affinity* is already cross-replica via the configured store).
- **Lost `RequestFinished`:** if Envoy crashes or the event is dropped, the
  slot would leak. A janitor sweeps in-flight entries older than a generous
  bound (default 600s) and `free`s them; `ActiveSequencesMultiWorker`'s
  periodic force-expiry is a second safety net.
- **Router unavailable:** the data plane **fails closed** — on a schedule
  timeout/error, return the control error or 503. There is no static endpoint
  bypass, so model and routing constraints cannot be silently violated. GWP's state
  self-heals: the load model misses those requests, but the planner poll
  re-anchors it within ~1s.

### Baseten integration: the shared-endpoints-gateway is the data plane

At Baseten the fronting proxy is not Envoy — it is the
**shared-endpoints-gateway** (`go/shared-endpoints-gateway`), a custom Go
reverse proxy (`gorilla/mux` + `httputil.ReverseProxy`). GWP exposes the gRPC
scheduling and lifecycle protocol used by Envoy, while `GwpCore` remains
transport-independent. Another data plane can use that protocol or a thin
adapter that invokes the same three core operations from its own seams:

| lifecycle call     | gateway seam |
|--------------------|--------------|
| schedule | `resolveEndpoint` (`pkg/server/server.go:562`) — called per request after authz/rate-limit, today returns the single static endpoint (or an org override); a candidate chosen by GWP slots in here. Request metadata/body is already parsed upstream in the middleware chain. |
| `ResponseStarted`  | `httputil.ReverseProxy.ModifyResponse` — first upstream response headers, incl. `x-baseten-dyn-worker-id`. |
| `RequestFinished`  | response-body close (wrap the proxied body; covers completion, error, and client disconnect). |

What the gateway already has, and how GWP maps onto it:

- **Model mapping** (`shared-endpoints-model-mapping` ConfigMap):
  `models[].routing.clusterLocal.{serviceName, namespace}` resolves to
  `http://<serviceName>.<namespace>.svc.cluster.local` — the dynamo graph's
  frontend Service (port 80 → 8000, no path rewriting). A model can already
  have **multiple deployments** (org overrides, e.g. a default and a hi-TPM
  deployment of the same model); the Django-fed dynamic routes even carry a
  `candidates` array that is currently collapsed to one entry. **GWP's job in
  this world is choosing among candidates per request** — today that choice
  is static (org override) or nonexistent.
- **GWP's "cluster" ≈ one dynamo graph** (`modelSlug`): each graph has its
  own frontend Service and its own planner. In GWP's ConfigMap that means
  `ingress_url: http://<modelSlug>.dynamo.svc.cluster.local/v1` and
  `planner_url: http://<modelSlug>-planner.dynamo.svc.cluster.local/deep/health`.
  Cross-*cluster* (multi-workload-plane) routing needs new exposure — every
  Service involved is ClusterIP today — so v1 runs GWP per workload plane
  beside the gateway, picking among that plane's graphs.
- **No session/cache awareness exists in the gateway today** (its only
  per-request routing input is the org id), so GWP is additive: a
  `globalRouterMiddleware` or a widened `resolveEndpoint` consults GWP and
  falls open to the current static resolution on timeout/error.

### Planner reachability (per graph)

The planner is already a k8s Service on the classic path:
`<modelSlug>-planner.<namespace>.svc.cluster.local`, **port 80 → 8008**
(`helm/charts/baseten-dynamo-model/templates/service_planner.yaml`). Two
consumers already poll it exactly this way (the MCM/LSM poller and the
llm-operator poller hit `/deep/health` and `/desired_scale`), so GWP's
reflector follows an established pattern.

**Deployment gap:** on the DynamoGraphDeployment path
(`dynamoGraphDeployment.enabled=true`, used by the model-APIs clusters) the
chart renders **no planner Service** — only the frontend gets one. Closing
that gap is a small helm change (render the planner Service on the DGD
branch too). Port numbers have also drifted across deployment generations
(80→8008 vs a stale 80→8080 sample), so `planner_url` is explicit per-endpoint
config rather than a derived convention.

## Design Goals

1. **Endpoint-level KV-aware routing** — pick the independently routable
   endpoint with the most observed prefix overlap for a new session, not just
   round-robin.
2. **Session affinity across turns** — stick to the last-routed endpoint iff it
   is still alive and serves the requested model; otherwise unstick and fall
   through to the approximate router. Stickiness is a fast bypass for the
   *decision*, not a bias inside scoring.
3. **Multi-replica, stateless-safe** — any replica can schedule any *session*
   (bindings live in a shared Redis or etcd store, not process memory).
   Production uses Redis for affinity and reserves etcd for router-replica
   discovery. Per-request in-flight state is
   replica-local; Envoy pins one request's lifecycle events to the replica
   that scheduled it.
4. **Reuse, don't fork** — drive selection through the existing
   `dynamo-kv-router` `Indexer` + `WorkerSelector` + `KvRouterConfig`; do not
   reimplement scoring.
5. **No remote discovery registration required** — endpoints expose an OpenAI
   ingress and planner; GWP feeds each planner's internal workers locally and
   maps them to the owning endpoint.
6. **LLM-agnostic load modeling** — the approximate indexer must work without a
   real tokenizer and without real KV events, so it can run anywhere.
7. **Graceful degradation** — affinity read errors become misses and failed
   affinity writes are dropped, so requests continue through normal routing
   without replica-local session truth. Replica events use ZMQ by default or
   NATS Core when `DYN_EVENT_PLANE=nats`; discovery still requires etcd. If an
   endpoint's planner stops answering, hold that endpoint's
   last-known-good through a grace window (other endpoints unaffected); if the
   router is down, Envoy fails closed rather than bypassing model and routing
   constraints.
8. **Data plane owned by Envoy** — TLS, pooling, backpressure, retries, and
   SSE streaming stay in Envoy; GWP never proxies response bytes in
   production.

## External contracts

### Planner worker list (already shipped)

There is no global planner — each dynamo graph runs its own **planner
service** (`planner_common.py`, fastapi on `PLANNER_PORT`, default 8008), and
its `GET /deep/health` **already returns what GWP needs**:
`DeepHealthResponse.detailed_load_data: dict[worker_id(u64) -> DetailedLoadData]`
with per-worker `num_prefill_tokens`, `num_decode_tokens`,
`num_decode_blocks`, `num_requests`, and the disaggregation `role`
(`prefill_and_decode` | `prefill` | `decode`). The field was added exactly for
this ("next-generation alyx … detect if certain replicas have died and correct
their LB model").

The reflector polls **one planner per configured endpoint**. The endpoint id
comes from the config entry, not the response. Planner worker IDs become GWP
scheduler identities and map back to that endpoint. The remaining work is
contractual:

- **Stabilize `detailed_load_data`** as a consumed API (it is currently
  described as dashboard-facing), or add a dedicated `/workers` alias.
- **Reachability:** `http://<modelSlug>-planner.<ns>.svc.cluster.local/deep/health`
  (Service port 80 → 8008) on the classic path — see "Planner reachability".
  The DynamoGraphDeployment chart path renders no planner Service yet (small
  helm fix); cross-workload-plane exposure is needed only when GWP runs
  centrally.
- **Cadence:** GWP polls at ~1s; the planner already serves a ~1s-cached view
  of router potential loads, so the poll adds negligible load.
- **Failure shape:** the planner returns `detailed_load_data: null` while its
  own router is unreachable — GWP treats that as a failed poll (hold
  last-known-good), never as an empty endpoint.

### Required served-worker response header

On every successful response the Dynamo frontend sets
`x-baseten-dyn-worker-id` to the decimal worker ID that served the request.
Envoy forwards it and includes it as `ResponseStarted.actual_worker_id`.

- The ID must match a worker owned by the selected endpoint. A worker that has
  not reached the next planner snapshot may be learned from the response.
- The header corrects scheduler load, prefix ownership, and session affinity.
- The endpoint remains the egress authority; GWP does not attempt to address
  the internal worker directly.
- A missing or cross-endpoint ID is a contract violation. GWP logs it and keeps
  the provisional booking rather than corrupting another endpoint's state.

## Approximate router lifecycle

The scheduler and indexer operate on planner worker IDs, with an endpoint
ownership table providing the routable authority.

- **Scheduler** (in-flight load, for new-session placement): the provisional
  sequence is registered by exactly one path. A fall-through uses
  `find_best_match(update_states=true)`; a sticky hit uses `add_request`
  directly. A correction first frees that provisional booking and then adds a
  replacement under a different accounting request ID.
- **Correction:** at `ResponseStarted`, free the provisional booking at least
  once and, when the reported worker differs, add the same workload under a
  distinct accounting request ID on the reported worker. Distinct IDs make
  independently delivered ZMQ `Free` and `AddRequest` events order-safe.
- **Indexer** (prefix-overlap, for KV-aware scoring): populated at
  `ResponseStarted` by `record_routing_decision` for the confirmed worker.

| GWP action                         | `KvRouter` call                | When |
|------------------------------------|--------------------------------|------|
| Register sequence (scheduler) — fall-through | `find_best_match(update_states=true)` | schedule, fall-through only; do NOT also `add_request` (double-count) |
| Register sequence (scheduler) — stick | `add_request(request_id, tokens, …, bound)` | sticky hit only; optimistically in the background when permitted, otherwise during schedule |
| Correct provisional booking       | `free(provisional_id)` then `add_request(confirmed_id, ..., actual)` | ResponseStarted, when actual differs |
| Populate indexer                  | `record_routing_decision(tokens_with_hashes, actual)` | ResponseStarted, on success |
| Mark prefill done                 | `mark_prefill_completed(accounting_id)` | ResponseStarted, after correction |
| Confirm worker affinity           | affinity-store `put(sid, actual, ttl)` | ResponseStarted, on success status |
| Release scheduler slot            | `free(accounting_id)`         | RequestFinished |

Two consequences follow:

1. **The decision is in two places with the same identity.** The configured
   affinity store holds the confirmed worker binding; the scheduler holds
   worker in-flight load for new-session scheduling.
2. **Sticks avoid scoring.** A stick bypasses `find_best_match`; it pays only
   the affinity lookup synchronously when optimistic accounting is available;
   tokenization and `add_request` then run in the background. The planner poll
   remains the baseline remote load source. Response correction
   keeps locally observed lifecycle load honest between polls.

### Request lifecycle: free on RequestFinished (load-model leak fix)

`free(request_id)` must be called for **every request when its stream ends**.
Without this, completed requests stay in `ActiveSequencesMultiWorker` forever, so
the load model monotonically inflates and routing degrades within minutes.
This applies to both sticks (which `add_request`-ed) and fall-throughs (which
`find_best_match`-registered).

Envoy Wasm's `onLog()` is the single choke point that covers **all** stream
endings — completion, reset, timeout, and client disconnect — so
`RequestFinished` is the one signal `free` hangs off:

| Envoy event                              | `KvRouter` call          |
|------------------------------------------|--------------------------|
| `encodeHeaders()` → `ResponseStarted` | `mark_prefill_completed(request_id)` |
| `onLog()` → `RequestFinished(reason)` | `free(request_id)` — releases the in-flight slot |

Marking prefill at **response headers** (rather than first body byte) also
covers non-streaming responses: headers always arrive, so the earlier "known
weakness" of unmarked prefill for `stream=false` requests disappears. For SSE
the two moments are near-identical (headers are sent when the first event is
ready).

Safety nets for a lost `RequestFinished` (Envoy crash, dropped event): the
router's in-flight janitor (bounded age, default 600s) and the scheduler's
periodic force-expiry.

## Architecture

### Components

```
            ┌────────────────────────────────────────────────────────┐
            │            External router (GWP, N replicas)           │
            │                                                        │
 Envoy ────▶│  ┌──────────┐   ┌─────────────┐   ┌──────────────┐    │
 gRPC       │  │ Tonic    │──▶│  session    │──▶│ per-oracle   │    │
 control    │  │ services │   │  resolver   │   │ router       │    │
 calls      │  │          │   │ (Redis or   │   │ registry     │    │
            │  │          │   │   etcd)     │   │ (KvRouter +  │    │
            │  │ schedule │   │             │   │  approx      │    │
            │  │ started  │   └─────────────┘   │  indexer +   │    │
            │  │ finished │                     │  selector)   │    │
            │  └──────────┘                     └──────▲───────┘    │
            │                 ┌─────────────┐   ┌──────┴───────┐    │
            │                 │  current    │──▶│  topology    │    │
            │                 │  producer   │   │  controller  │    │
            │                 │ (reflector) │   │ + snapshots  │    │
            │                 └──────▲──────┘   └──────────────┘    │
            │                        │ 1s poll, per endpoint         │
            │                        │ each graph's planner          │
            │                        │ deep/health: workers + load   │
            └────────────────────────────────────────────────────────┘

 Envoy ── proxies bytes ──▶ selected endpoint ingress (authority from the
                            ext-authz header mutations; TLS/pooling/retries in Envoy)
```

- **Envoy adapters** — Tonic gRPC services for ext-authz scheduling,
  asynchronous Wasm lifecycle delivery, and standard liveness/readiness
  health checks. Envoy remains the only inference data plane.
- **session resolver** — implements GWP's async `AffinityStore` trait with an
  operator-selected etcd or Redis backend. `InMemoryAffinityStore` is compiled
  only for tests. The store holds the last confirmed planner `WorkerId`.
- **router registry** — dynamically maintains one `KvRouter<Sel>` per canonical
  `oracle_version_id`, isolating scheduler state and active-sequence events
  between routable models. Routers absent from the latest topology are retired.
  The registry uses etcd discovery and either the default ZMQ event plane or
  NATS Core selected by `DYN_EVENT_PLANE`. `router_replica_sync=true` shares
  add/prefill/free lifecycle events across GWP replicas, so each selector sees
  global GWP in-flight load for its oracle version.
  `use_kv_events=false` retains a replica-local prune-TTL'd radix indexer
  populated by `record_routing_decision`; worker configs arrive through a
  plain `watch` channel owned by the topology controller.
- **topology controller** — validates complete, source-neutral topology
  generations, atomically replaces the request-path snapshot, and fans the
  same generation out to the router worker feed, observed-load fusion,
  readiness, and metrics. Structurally invalid updates retain the last valid
  generation.
- **readiness** — after warmup, GWP becomes ready when the reconciled topology
  contains at least one live worker anywhere. A configured model with zero
  workers does not make the whole multi-model GWP deployment unready; requests
  for that model still fail routing normally.
- **current topology producer (reflector)** — background task. Reads a ConfigMap (path from
  `DYN_GWP_CONFIG_PATH`, default `/configs/gwp.yaml`) with separate
  `endpoints`, `routes`, `session`, and `routing` sections. Every 1s it polls
  every endpoint planner concurrently and emits complete topology generations.
  It preserves each planner worker's coherent load tuple. A failed poll
  (non-200/timeout/`detailed_load_data: null`) holds that endpoint's
  last-known-good for `routing.planner_staleness_grace_secs` (default 30);
  an empty but usable worker set evicts it immediately.
- **live configuration** — the ConfigMap-mounted YAML is
  polled and atomically hot-reloaded. Adding or removing an endpoint reconciles
  its producer state and routing eligibility
  without restarting GWP. Configured routes are the authoritative external
  model catalog because a downstream `/v1/models` response may be unavailable
  or expose internal IDs instead of client-facing aliases. Every routable
  model names its endpoint candidates explicitly. GWP does not depend on or
  expose a `/v1/models` control-plane diagnostic. The system port does expose
  `/info`, listing each loaded `oracle_version_id` and its current live replica
  count; credentials and planner URLs are not included.
- **authority mapping** — the chosen `WorkerId` maps through the request's
  immutable topology snapshot to its endpoint and `ingress_url`; GWP returns
  that URL's host:port and configured
  endpoint name as ext-authz header mutations. Envoy owns the actual egress:
  TLS, pooling, retries, streaming.

The normalized configuration shape is:

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
  ttl_secs: 1800
  prompt_hash_fallback:
    token_position: 100000
  backend:
    type: redis
    url: redis://redis.gwp.svc.cluster.local:6379/
    key_prefix: "gwp:affinity:"

routing:
  pseudo_stride: 4
  block_size: 32
  planner_staleness_grace_secs: 30

tokenization:
  models:
    glm-4.7:
      mode: real
      directory: /opt/dynamo-gwp/tokenizers/glm4.7
```

`served_alias_model_map` is applied to request names and every
`routes.models` member before route candidate sets are built. Multiple
deployment-advertised names can therefore remain in one route while routing,
tokenization, affinity, and metrics collapse to one canonical model. The
original OpenAI request body is forwarded unchanged.

Endpoint routing properties are dimensioned hard constraints applied before
model and load scoring. Alyx stamps
`x-baseten-model-apis-routing-requirements`, whose value is a JSON object such
as `{"region":{"required":["us"]}}`. Dimensions AND together; alternatives
within one dimension's `required` list OR together. A missing header or `{}`
adds no constraint. Malformed JSON returns 400, and valid requirements with no
live match return 503. `preferred` is accepted but not scored yet. Envoy must
fail closed whenever this header is present and scheduling fails; its normal
default-endpoint fallback is forbidden on the constrained path.

### Topology identity mapping

`KvRouter` and `WorkerSelector` operate on `WorkerId` (u64) and
`WorkerWithDpRank` and on a `HashMap<WorkerId, ModelRuntimeConfig>`. The
topology controller projects those inputs from one generation without changing
the crate; the current producer uses planner IDs directly:

| GWP concept        | `dynamo-kv-router` type                  |
|--------------------|------------------------------------------|
| endpoint           | configured egress authority owning one or more workers |
| planner worker     | native `WorkerId`, with `dp_rank: 0`, mapped to its endpoint |
| session binding    | last confirmed planner worker; honored iff live and model-eligible |
| response echo      | `x-routed-endpoint: <endpoint_id>` and required `x-baseten-dyn-worker-id` |
| `ModelRuntimeConfig` | synthesized once per live planner worker |

The topology snapshot maps every worker ID back to the configured endpoint.
Worker IDs are stable across GWP polls and restarts, so bindings survive as
long as the current provider continues to advertise that worker.

### Data Flow

```
1. Envoy: client POST /v1/chat/completions (maybe with x-session-id and
   x-baseten-model-apis-routing-requirements)
   ext-authz buffers the body and carries the generated request ID
2. Envoy -> GWP: gRPC ext-authz Check with HTTP attributes and the original
   OpenAI body in raw_body
3. GWP check(rid, headers, path, body):
   a. model = served_alias_model_map.get(body["model"]) ?? body["model"]
   b. sid = first_known_session_header(headers) ?? body["user"]
            ?? four_hash_prompt_prefix_at_configured_cutoff
            ?? mint base10-<uuid>
   c. select the isolated router/scheduler for canonical model (oracle_version_id)
   d. eligible = live workers whose endpoints satisfy every routing dimension,
                 serve the model, and belong to its configured route
   e. bound = affinity.peek(sid)            // last confirmed worker, may be None
   f. if bound.is_some_and(|w| eligible.contains(w)):    // STICK
          decision = bound                   // no find_best_match
          tokens = tokenize_for_model(model, body)
                   // Optimistic path may return first, then do this in background.
          kv_router.add_request(rid, tokens, bound)      // scheduler load, keyed by rid
          sticky = true
      else:                                  // UNSTICK or new session
          tokens = tokenize_for_model(model, body)
          decision = kv_router.find_best_match(
              tokens, context_id=rid, update_states=true, eligible)
          sticky = false                     // scheduler already updated; no separate add_request
   g. endpoint = endpoint_table.endpoint_of(decision)
   h. inflight[rid] = {
          sid, decision, endpoint.id, tokens, accounting_rid: rid
      }
   i. return ext-authz header mutations for authority, endpoint, session, and stickiness
4. Envoy proxies the request to `authority` (TLS/pooling/retries in Envoy)
5. Envoy encodeHeaders (first SSE response):
   inject x-session-id / x-routed-endpoint into the client response,
   forward x-baseten-dyn-worker-id through,
   -> GWP: ResponseStarted { rid, endpoint_id, actual_worker_id, status }
6. GWP response_started(rid, actual_worker_id, status):
   a. if status != 200:
          drop inflight[rid]
          kv_router.free(accounting_rid)                 // no confirmed booking
          record denied metric; return
   b. validate actual_worker_id belongs to endpoint
   c. if actual_worker_id != decision:
          kv_router.free(accounting_rid)                 // at least once
          accounting_rid = rid + ":confirmed:" + actual_worker_id
          kv_router.add_request(accounting_rid, tokens, actual_worker_id)
          decision = actual_worker_id
   d. kv_router.mark_prefill_completed(accounting_rid)
   e. kv_router.record_routing_decision(tokens, decision)
   f. affinity.put(sid, decision, ttl)                   // bind confirmed worker
7. Envoy streams SSE to the client
8. Envoy Wasm onLog (close | reset | timeout | disconnect):
   -> GWP: RequestFinished { rid, reason }
9. GWP request_finished(rid):
   entry = drop inflight[rid]; kv_router.free(entry.accounting_rid)
```

**Key discipline:** `rid` keys the in-flight table and provisional scheduler
slot; `sid` keys affinity only. A correction uses a distinct derived
accounting ID, so reordered replica-sync events cannot let a provisional
`Free` erase the confirmed `AddRequest`. Two concurrent requests in one
session get distinct IDs. Both sticky and fall-through paths record confirmed
prefix ownership at `ResponseStarted`.

## Design Decisions

### Decision 1: Reuse `KvRouter` with a local planner-worker feed

**Context:** `KvRouter::new` takes a `RuntimeConfigWatch` — in production wired
via `runtime_config_watch` (`lib/llm/src/discovery/runtime_configs.rs:24`),
which joins instance availability + `ModelDeploymentCard.runtime_config` from
the local etcd-backed discovery plane. Remote clusters are not in local etcd.

**Options Considered:**
1. *Fork the selector* — reimplement cluster-level scoring outside the crate.
   Pros: no discovery coupling. Cons: duplicates scoring logic, drifts.
2. *Register remote workers in real etcd* — write `DiscoverySpec::Endpoint` +
   `DiscoverySpec::Model` into `KVStoreDiscovery` with leases. Pros: zero new
   code paths. Cons: pollutes the shared etcd with foreign-cluster state; lease
   churn at 1s cadence; cross-cluster coupling.
3. *Mock discovery in-process* — build a `SharedMockRegistry`-backed
   `MockDiscovery` (`lib/runtime/src/discovery/mock.rs:15`) and feed it from
   the reflector.
4. *Feed the watch channel directly* — `RuntimeConfigWatch` is just
   `watch::Receiver<HashMap<WorkerId, ModelRuntimeConfig>>`; GWP's topology
   controller owns the `watch::Sender` and publishes a full generation.

**Decision:** Option 4 for endpoint configuration, combined with shared router
lifecycle synchronization. Remote endpoints are not registered in discovery;
each replica feeds provider-observed worker IDs and endpoint ownership locally.
The GWP router registry uses etcd discovery and a configurable event plane so
`router_replica_sync` can mirror each oracle version's active-sequence load
across replicas. `DYN_EVENT_PLANE` selects `zmq` (the default) or `nats`.

**Implementation notes (hard-won):**
- `skip_initial_worker_wait` **must stay false**: it doubles as "watch worker
  configs" in `KvScheduler::start` (`scheduler.rs:94`); true freezes the worker
  set at construction, breaking the reflector feed. With
  `DYN_ROUTER_MIN_INITIAL_WORKERS` unset the constructor does not block.
- `router_snapshot_threshold: None`,
  `router_queue_threshold: None` (no queueing at the routing tier),
  `router_replica_sync: true` (add/prefill/free events over the selected event
  plane).
- Discovery uses the standard `ETCD_ENDPOINTS` and etcd authentication
  environment variables. `DYN_EVENT_PLANE=nats` uses NATS Core pub-sub;
  otherwise GWP uses direct ZMQ.

**Consequences:** GWP runs its own `KvRouter` view of workers across remote
deployments. It still routes only to ingress endpoints; intra-deployment
routing stays the job of each endpoint's own router, and response reconciliation
corrects the provisional GWP worker choice.

### Decision 2: shared etcd or Redis for multi-replica session affinity

**Context:** GWP's async `AffinityStore` trait has a test-only
`InMemoryAffinityStore`. Production affinity requires a shared backend. The
production currently uses Redis with a 1800s TTL. The backend remains
selectable for other deployments.

**Options Considered:**
1. *etcd* — already in the stack as the discovery plane (`KVStoreDiscovery`,
   `etcd-client` is a workspace dep). Pros: no new datastore to run/secure;
   leases give TTL for free; reuses existing client/credentials. Cons: raft
   consensus per write (~5–15ms p99) and lease bookkeeping; not built for very
   high write QPS.
2. *Redis* — purpose-built for short-TTL key lookups at high QPS. Pros: atomic
   `SET ... EX`, native expiration, reconnecting multiplexed client. Cons:
   another datastore to operate and secure.
3. *Sticky header only* — let the client always send `x-session-id` and a
   `x-endpoint-id`; no server state. Cons: clients won't; defeats the "mint on
   the fly" goal.

**Decision:** Keep the store selectable. `EtcdAffinityStore` lives behind the
`etcd` feature and uses etcd **leases** for TTL. `RedisAffinityStore` lives
behind the `redis` feature and uses Redis key expiry. The standard image
includes both; `session.backend.type` chooses one at runtime. Existing
`session.etcd_endpoints` configuration remains compatible. A configured client
must initialize successfully. At runtime, read errors are affinity misses and
failed writes are logged and dropped; normal routing continues without
creating divergent replica-local bindings.

**Scale envelope:** Redis is required for production affinity so affinity
traffic remains isolated from etcd discovery. The etcd implementation remains
available for smaller or non-production deployments.

**etcd usage rules (what keeps it cheap):**
- **Leases expire, never keepalive.** Grant a lease with the TTL, write the
  binding, and let it expire. No keepalive traffic — one write per `put`.
- **Serializable (stale/local) reads for `peek`.** A stale binding is harmless:
  it may name an evicted endpoint, which the liveness check catches and unsticks.
  This keeps the hot-path `peek` off the consensus path (sub-ms, local).
- **`put` happens at `ResponseStarted`** — off the request-scheduling path, so
  raft write latency never delays a placement decision.
- **Blast-radius:** the key prefix is dedicated to GWP affinity
  (`gwp/affinity/<sid>`) so it is isolable from discovery keys; monitor etcd
  write load. If affinity writes meaningfully perturb discovery, move affinity
  to a dedicated etcd cluster or Redis.

**Redis usage rules:**
- `peek` is a plain `GET` and never refreshes TTL.
- `put` is one atomic `SET key value EX ttl`; the last confirmed worker written
  for a session wins.
- The default prefix is `gwp:affinity:` and can be overridden to isolate
  environments.
- `ConnectionManager` automatically reconnects; `redis://` and TLS
  `rediss://` URLs are accepted.

**Semantics:** the store holds the last confirmed planner
`WorkerWithDpRank` per session. The read is a **stickiness check, not a scoring
bias**: `peek(sid)` returns the bound worker; GWP honors it iff the reflector
reports it alive and its endpoint still serves the request model and satisfies
the current routing requirements. Otherwise it ignores the binding and falls
through to the approximate router. `peek` (no TTL refresh) is used on the
request path so a failed routing attempt does not extend affinity; `put` is
only called on a successful upstream status carrying
`x-baseten-dyn-worker-id`. The last confirmed worker written for a session
wins.

**Consequences:** A binding whose worker or endpoint has disappeared is not
eagerly deleted from the backend. It is ignored on the next `peek`, then
overwritten by the next successful response or removed by TTL expiry. This
keeps the hot path to one backend read regardless of churn.

### Decision 3: Per-model real or pseudo tokenization

**Context:** `KvRouter::find_best_match_details` takes `tokens: &[u32]`
(`lib/llm/src/kv_router.rs:478`) and hashes them via
`compute_block_hash_for_seq`. We do not have the real tokenizer for arbitrary
remote models, and even when we do, applying the exact chat template (tool
calls, reasoning, multimodal) is brittle.

**Options Considered:**
1. *Full preprocessor* — run `OpenAIPreprocessor` with a real `PromptFormatter`
   per model. Pros: exact. Cons: needs every remote model's
   `tokenizer_config.json` + chat template; tool-call preprocessing diverges
   across backends; expensive on the hot path.
2. *Predefined processors per model name* — ship a static registry
   `model_name -> PromptFormatter` (mirroring `deepseek_formatter_for`,
   `preprocessor/prompt.rs:190`), falling back to tier 3 for unknown models.
3. *Pseudo-tokenization* — byte heuristic: pack each 4-byte chunk of the
   rendered string into one `u32` (little-endian, zero-padded tail) to produce
   a `Vec<u32>` of the right *shape* (~bytes/4 ≈ real token count on English
   text); block hashes still collide consistently for the same prefix, which
   is all the indexer needs for *relative* scoring. (Chunk-packing, not
   every-Nth-byte sampling: sampling leaves 3 of 4 bytes invisible, which
   manufactures false prefix overlap between genuinely different prompts.
   Packing is lossless at the same cost and token count.)

**Decision:** Tokenization is an explicit per-model policy, with pseudo as the
default for every configured model without an override:
1. `mode: real` points to a directory containing `tokenizer.json`,
   `chat_template.jinja`, and `tokenizer_config.json`. GWP loads the
   `basetenkenizer` tokenizer and compiles the template once at startup. Chat
   requests render the exact template before encoding; completion prompts use
   the real tokenizer without a template.
2. `mode: pseudo`, or no entry for the model, uses deterministic 4-byte chunk
   packing.

The indexer only needs *consistent* hashes for prefix comparison, so
pseudo-tokenization is sound for relative ranking even though absolute token
counts are wrong. GWP records the tier in request-path diagnostics.

**Consequences:** Routing quality degrades gracefully with model novelty.
Real-tokenizer bundles are portable filesystem units and do not require model
downloads at runtime. A broken explicit bundle fails startup rather than
silently mixing token spaces. Tokenization policy is startup-only: ConfigMap
reloads retain the existing policy until all replicas are restarted.

### Decision 4: LLM-agnostic load model from the 1s poll

**Context:** Real `ActiveLoad` (active decode blocks, kv used/total) arrives
over NATS `kv_metrics` from real engines. Remote clusters don't publish to our
NATS. But each cluster's planner `deep/health` already returns **real
per-worker load** in `detailed_load_data`: `num_prefill_tokens`,
`num_decode_tokens`, `num_decode_blocks`, `num_requests`, and the
disaggregation `role` — sourced from the cluster's own router's potential
loads (~1s-cached).

**Decision:** The current producer parses `detailed_load_data` and preserves each
worker's coherent load tuple. Each new planner generation anchors GWP's current
local load. Selection uses the delayed planner tuple plus only local growth
since that anchor, never less than the current local view. This prevents a
request visible in both sources from being counted twice while covering work
that started after the delayed snapshot. GWP then delegates to the existing
fixed B10 scoring module, mixing prefix overlap and reconciled load without
changing the generic KV router. The downstream response corrects the
provisional choice when the endpoint's local router selects another worker.

**Consequences:** Load balancing is predictive until response headers arrive;
confirmed bookings then make subsequent load and prefix decisions reflect the
worker that actually served the request.

### Decision 5: Use existing selector balancing

**Context:** Prefix overlap should be balanced against endpoint load.

**Decision:** Keep one concrete B10 scorer, separate from worker eligibility
and sampling, after GWP reconciles provider-observed load with local lifecycle
deltas. Its hot-reloadable `b10_routing_config` controls
the overlap/prefill weight, decode-block weight, prefill/decode token
discounts, active-request weight, and temperature. A hard per-endpoint
admission cap is not part of this PoC and can be added later if production
traces show it is necessary.

### Decision 6: Mint a session id and echo the selected endpoint

**Context:** Clients may not send `x-session-id` on turn 1. OpenAI requests
already have an optional stable `user` field, and the selected endpoint is
useful for observability.

**Decision:** Resolve the session ID in precedence order: explicit
`x-session-id`; vendor-neutral `x-session-affinity`; Codex `session-id`; Claude Code
`x-claude-code-session-id`; OpenCode `x-parent-session-id`; Claude Code
`x-claude-code-agent-id` then `x-claude-code-parent-agent-id`; string-valued
OpenAI `user`; four consecutive model-scoped rolling prompt hashes ending at
the configured cutoff; then minted `base10-<uuid>`. The prompt-hash fallback
is disabled when no cutoff is configured and is unavailable until the prompt
reaches that position.
The ext-authz decision carries that `session_id` and `endpoint_id`; the Envoy
filters inject both into the client response:
- `x-session-id: <sid>` — clients that echo it get stickiness next turn;
  clients that ignore it force GWP to mint a new id each turn (correct, just no
  cross-turn affinity).
- `x-routed-endpoint: <endpoint_id>` — the independently routable ingress GWP
  selected for this turn.

**Consequences:** The affinity store only ever holds confirmed decisions. A
client that round-trips `x-session-id` gets sticky, prefix-warm routing as long
as the bound worker and its endpoint stay alive and eligible for the requested
model.

### Decision 7: Reconcile provisional placement from the served-worker header

**Context:** GWP selects a planner worker but can route only to that worker's
endpoint. The endpoint's local router may choose a different worker.

**Decision:** Every successful response provides
`x-baseten-dyn-worker-id`. Envoy sends it in
`ResponseStarted.actual_worker_id`. When it differs from the provisional
choice, GWP frees the provisional booking and adds the same workload to the
reported worker under a distinct accounting request ID. It records prefix
ownership and affinity on the reported worker.

**Consequences:** GWP's scheduler converges immediately instead of waiting for
the next planner poll. The endpoint remains the addressable unit, and distinct
provisional/confirmed request IDs make replica-sync event reordering safe.

### Decision 8: external data plane, GWP as an external router

**Context:** Combining placement with an inline reverse proxy would make GWP
own TLS, connection pooling, backpressure, retries, and streaming edge cases.
A hardened data plane already fronts inference traffic.

**Options Considered:**
1. *Inline Rust reverse proxy* — one process does everything. Pros: no extra
   hop or per-request state handoff. Cons: owns the entire data-plane
   hardening surface and creates another proxy tier to operate.
2. *External data plane + GWP external router* — the fronting proxy buffers
   the request, calls GWP for the placement decision, proxies the bytes, and
   reports lifecycle events (gRPC scheduling decision,
   `ResponseStarted`, `RequestFinished`). Pros: TLS/pooling/backpressure/
   retries/SSE stay where they already are; GWP shrinks to a pure decision
   service. Cons: one extra local RTT on the request path;
   per-request in-flight state on the router with replica pinning; data-plane
   integration to build (Envoy filter or gateway hook).
3. *Custom Go rewrite of the routing core* — port the placement logic into
   the gateway itself. Pros: no callout. Cons: forks the routing core away
   from `dynamo-kv-router` (Rust) — exactly the "reuse, don't fork" goal this
   design exists to protect.

**Decision:** Option 2, with a transport-agnostic core behind two adapters:
- **Envoy**: native gRPC ext-authz performs scheduling; the Rust Wasm filter's
  response-header and `onLog` hooks map onto prefill and free.
- **shared-endpoints-gateway** (the concrete Baseten integration, see
  "Baseten integration"): `resolveEndpoint` / `ModifyResponse` / body-close
  map onto the same three calls.

**Consequences:**
- The scheduling core is transport-agnostic; Tonic adapters are thin shells
  over it and Envoy is required for end-to-end inference.
- GWP holds per-request in-flight state between `schedule` and
  `request_finished` (session + selected endpoint), with a
  janitor for lost `RequestFinished` events and replica pinning provided by
  the data plane's persistent connection (see "Deployment shape").
- `ResponseStarted` marks prefill at the first upstream response headers,
  which also covers non-streaming responses.

## Algorithms

### Per-model tokenization

```
fn approx_tokens(req, model_name) -> (Vec<u32>, Tier):
    if let Some(bundle) = configured_real_tokenizer(model_name):
        if req.messages:
            text = bundle.chat_template.render(req.messages, request_kwargs)
            return (bundle.tokenizer.encode(text), Tier::ExactTemplate)
        return (bundle.tokenizer.encode(req.prompt), Tier::TokenizerNoTemplate)
    bytes = render_plain(req.messages or req.prompt).as_bytes()
    toks = bytes.chunks(PSEUDO_STRIDE).map(pack_le_u32_zero_padded).collect()
    return (toks, Tier::Pseudo)
```

### Topology pipeline and current planner producer

Routing consumes one immutable `TopologySnapshot` containing routable endpoint
metadata, canonical model bindings, resolved model profiles, worker ownership,
worker runtime configuration/taints, and observed load. A topology controller
validates each complete update and publishes the same generation to request
routing, the scheduler worker watch, load fusion, readiness, and metrics.

Exactly one producer is authoritative for a GWP process. Today that producer
combines the hot-reloaded ConfigMap with planner observations. A future
database or Kafka adapter can emit the same complete snapshot without adding
source-specific behavior to request routing. Protocol failure and replay
semantics remain producer-owned; the controller retains its last valid
snapshot when an update is structurally invalid.

Endpoint properties and worker taints remain distinct. Endpoint properties
apply hard deployment-level routing requirements before scheduling. Worker
taints travel in `ModelRuntimeConfig` and participate in the router's required
and preferred worker-level constraints. The current planner payload does not
include runtime configs, so its adapter emits the same default, untainted
worker configs as before.

The current planner producer keeps a **last-known-good** endpoint observation
and a configurable **staleness grace** (default 30s) **per endpoint**, so a
planner blip is a non-event, not a mass re-route. The naive "clear liveness on
a failed poll" would unstick sessions during a transient outage. Instead:

- a **usable non-empty poll** (200 with `detailed_load_data` present) reconciles
  the endpoint immediately and publishes every planner worker;
- a **usable empty poll** evicts the endpoint immediately because the planner
  has authoritatively reported no routable workers;
- a **failed poll** (non-200 / timeout / parse error / `detailed_load_data:
  null`, which the planner returns while its own router is unreachable) holds
  that endpoint's last-known-good and only counts down its grace window. An
  endpoint is evicted (its sticks → unsticks) once its planner
  has been unusable for the whole grace period.

```
GRACE = config.routing.planner_staleness_grace_secs  # default 30
state[e] = { last_good: now, cleared: false, workers: {} }

every 1s, all endpoints concurrently:
  resp = http_get(e.planner_url, auth=e.planner_api_key)
  usable = resp.is_200 and resp.detailed_load_data is not null
  if usable:
      if resp.detailed_load_data.is_empty():
          state[e].workers = {}           # authoritative empty
      else:
          state[e].workers = resp.workers.map((wid, load) => {
              endpoint_table.upsert(wid, e.endpoint_id)
              return synthesize(wid, load)
          })
      state[e].last_good = now; state[e].cleared = false
  else:
      if !state[e].cleared and now - state[e].last_good > GRACE:
          state[e].workers = {}           # evict this endpoint only
          state[e].cleared = true
      # else: hold last-known-good — a blip is a non-event

# after all endpoints: publish the worker union
candidates = state[*].workers.flatten()
endpoint_table.retain(candidates.ids)
liveness_set = candidates.ids
workers_watch.send(runtime_configs(candidates))
load_feed.replace(candidates.loads)
```

Consequences of the per-endpoint grace window:

- A short planner blip never evicts an endpoint or re-routes a session —
  `peek` keeps returning the bound worker, sticks stay stuck, and `put` is not
  churned.
- Endpoint isolation: one planner outage evicts only that endpoint; every
  other endpoint's sticks and feed entries are untouched.
- There is no divergence between consecutive usable polls: each reconciles the
  endpoint to a fresh per-worker snapshot, so load self-corrects on the next
  successful poll.
- Only a sustained transport/API outage (≥ GRACE) flips an endpoint into
  "planner down": its liveness entry clears, its sticks unstick on their next
  `peek`, and sessions fall through to another eligible endpoint
  until the planner returns.

### Core lifecycle (transport-agnostic; called by the gRPC adapters)

```
fn schedule(rid, headers, path, body) -> ResolvedRoute:
    model = served_alias_model_map.get(body["model"]) ?? body["model"]
    sid = first_known_session_header(headers) ?? body["user"]
          ?? four_hash_prompt_prefix_at_configured_cutoff
          ?? mint "base10-<uuid>"
    tokens = tokenize_for_model(model, body)           // real opt-in or pseudo default
    eligible = live workers whose endpoints satisfy every routing dimension,
               serve model, and intersect its configured route
    bound = affinity.peek(sid)                         // last confirmed worker
    if let Some(w) = bound and eligible.contains(w):   // STICK
        decision = w; sticky = true
        kv_router.add_request(rid, tokens, decision)   // scheduler load, keyed by rid
    else:                                              // UNSTICK or NEW
        decision = kv_router.find_best_match(
            tokens, context_id=rid, update_states=true, eligible)
        sticky = false                                 // no separate add_request (double-count)
    endpoint = endpoint_table.endpoint_of(decision)
    inflight[rid] = {
        sid, decision, endpoint.id, tokens, accounting_rid: rid,
        booked_at: monotonic_now()
    }
    return { authority: endpoint.authority, endpoint_id: endpoint.id,
             session_id: sid, sticky }

fn response_started(rid, endpoint_id?, actual_worker_id?, status):
    entry = inflight[rid] or return                   // unknown rid: warn, no-op
    metrics.observe_ttft(monotonic_now() - entry.booked_at)
    if status != 200:
        inflight.remove(rid)
        kv_router.free(entry.accounting_rid)          // no confirmed booking
        metrics.observe_e2e(monotonic_now() - entry.booked_at)
        metrics.denied(endpoint_id, status)
        return
    actual = validate_owner(actual_worker_id, entry.endpoint_id)
    if actual != entry.decision:
        kv_router.free(entry.accounting_rid)          // idempotent, at least once
        confirmed_rid = rid + ":confirmed:" + actual
        kv_router.add_request(confirmed_rid, entry.tokens, actual)
        entry.decision = actual
        entry.accounting_rid = confirmed_rid
    kv_router.mark_prefill_completed(entry.accounting_rid)
    kv_router.record_routing_decision(entry.tokens, entry.decision)
    affinity.put(entry.sid, entry.decision, ttl)      // bind confirmed worker

fn request_finished(rid, reason):
    entry = inflight.remove(rid)
    kv_router.free(entry.accounting_rid)  // RELEASES confirmed/provisional slot
    metrics.observe_e2e(monotonic_now() - entry.booked_at)

janitor (every 60s):            // lost RequestFinished (Envoy crash, dropped event)
    for rid, entry in inflight where age(entry) > 600s:
        request_finished(rid, "janitor")
```

Routed counters and both lifecycle histograms use the bounded labels
`model`, `routed_endpoint`, and sanitized `downstream_authority` (`host:port`).
The raw ingress URL is deliberately not a label: paths, queries, and
credentials must never enter Prometheus, and the stable endpoint ID survives
URL changes. Response metrics additionally classify `outcome` into exactly
four values: `success` (HTTP 200), `overloaded` (429/503/529), `client_error`
(other 4xx), and `upstream_error` (all other non-200 responses plus
no-response transport failures). The exact HTTP `status` remains on response
counters. `gwp_request_outcomes_total` is incremented once after final free,
so no-response failures and janitor expirations are countable without inferring
totals from histogram buckets.

**Key discipline:** `rid` (Envoy's per-request id) keys the in-flight table and
the provisional scheduler booking; `sid` keys affinity only. A corrected
booking uses a distinct derived accounting ID, preventing a reordered
provisional `Free` from deleting the confirmed `AddRequest`. Two concurrent
requests in one session get distinct IDs and never collide. At
`ResponseStarted` the session is bound to the confirmed worker.

## Performance Considerations

- **Schedule callout** (the only on-path hop): one local RTT Envoy→router.
  Stick: affinity-store `peek` + in-memory liveness check. With an available
  tokenization permit, optimistic affinity returns the decision before
  tokenization and `add_request`, which continue in the background; permit
  saturation falls back to synchronous accounting. Pseudo mode is µs-scale;
  real mode is expected to dominate the synchronous path.
  Fall-through adds one indexer query + selector pass (sub-ms). Measure the
  model bundles in production before setting a universal p99 target.
- **`ResponseStarted` / `RequestFinished`**: fire-and-forget off the data
  path; the affinity `put` rides on `ResponseStarted` and never delays the
  client stream.
- **In-flight table**: one small entry per active request (tokens ≈ bytes/4 of
  the prompt), dropped at `RequestFinished`; janitor bounds leakage at 600s.
- **Reflector cost**: one planner poll/s per endpoint. Bounded by planner
  fan-out, not request rate. Runs off the request path.
- **Memory**: the approx indexer holds per-worker block hashes with a TTL
  (`PruneConfig::ttl`, default 120s). With pseudo-tokenization the hash space
  is small, so the radix tree stays compact.
- **Affinity cost**: one store read per affinity-enabled schedule. Production
  Redis keeps this traffic separate from etcd discovery; the stick path does
  only a `peek` and never refreshes TTL. Monitor Redis latency and key count at
  the configured 1800s retention.

## Future Work

- Evaluate streaming gRPC scheduling only if overlapping affinity lookup with
  request-body delivery produces a measured critical-path improvement.
- Replace the 1s poll with a watch/stream where the planner supports it
  (e.g. SSE on `list_workers_url`).
- Add database or Kafka topology producers that emit the same complete
  snapshot contract; select exactly one authoritative producer per process.
- Support controlled tokenizer-bundle rollouts without requiring a full
  replica restart while preserving one prefix-hash space during the rollout.
- Share the GWP approximate indexer state across replicas (today each replica
  builds its own from the traffic it schedules; the existing
  `standalone_indexer` service could host a shared one).
- Cross-cluster KV migration hints: when a session's prefix is warm on cluster
  A but A is overloaded, emit a hint to migrate rather than just re-route.
- Direct streaming `ActiveLoad` ingestion from clusters that expose a metrics
  endpoint, reducing the current planner-poll staleness.

## References

- KV router crate: `lib/kv-router/src/lib.rs`
- Approximate (pruning) indexer: `lib/kv-router/src/indexer/pruning.rs`,
  re-exported `approx` at `lib/kv-router/src/lib.rs:23`
- High-level `KvRouter`: `lib/llm/src/kv_router.rs`
- `AffinityStore` trait (llm, sync ancestor of GWP's async trait):
  `lib/llm/src/kv_router/sticky/router.rs:45`
- `runtime_config_watch` (worker join): `lib/llm/src/discovery/runtime_configs.rs:24`
- `DistributedConfig`, etcd discovery, and ZMQ/NATS event planes:
  `lib/runtime/src/distributed.rs`, `lib/runtime/src/transports/event_plane/`
- `basetenkenizer`: configured real-tokenizer implementation in
  `lib/gwp/src/tokens.rs`
- Preprocessor + chat-template glue: `lib/llm/src/preprocessor.rs`,
  `lib/llm/src/preprocessor/prompt.rs:190` (`deepseek_formatter_for`)
- Hot-reloadable config pattern: `lib/llm/src/kv_router/b10hotreloadablecm.rs`
- Standalone HTTP indexer service: `lib/kv-router/src/standalone_indexer/`
- Baseten planner service (per-graph, `deep/health` + `detailed_load_data`):
  `baseten_dynamo/cache_aware_routing_trtllm/src/common/planner_common.py`
- Dynamo model helm chart (frontend + planner Services):
  `helm/charts/baseten-dynamo-model/templates/{service.yaml,service_planner.yaml}`
- Shared-endpoints-gateway (Go data plane, model mapping, per-request hook):
  `go/shared-endpoints-gateway/pkg/server/server.go` (`resolveEndpoint`),
  `helm/charts/shared-endpoints-gateway/templates/configmap_model_mapping.yaml`
- Envoy ext_proc / HTTP filter hooks: https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/ext_proc_filter
- Router design doc: `docs/design-docs/router-design.md`
- Component design template: `docs/templates/component-design.md`
