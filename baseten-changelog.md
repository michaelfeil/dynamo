# Baseten Dynamo Fork Patch Changelog

Source branch: `main-v1.0.0`

Comparison base: `origin/upstream/v1.0.0`

Target rebase base reviewed: `bed9f269312151481cd67a8d21b70e0f52424c2b`
from `https://github.com/ai-dynamo/dynamo`.

Branch head covered:

```text
c9bf0c4ed fix: Make router active replicas hot configurable (#251)
```

This file is the long-term patch ledger for rebasing the Baseten Dynamo fork
onto a newer upstream release such as `main-v1.2.0`.

The entries below are intentionally grouped by replay unit, not by original PR
or commit. Each group should be treated as one functional area to audit, port,
test, and either keep or drop during an upgrade.

Status values:

- `keep`: likely Baseten-specific or still required.
- `review-upstream`: check the target upstream release before replaying.
- `upstream-sync`: local branch intentionally copied an upstream/main change.
- `mixed`: contains both Baseten-specific work and changes that may already be
  upstream.
- `redesign`: the old behavior may still matter, but the target branch already
  changed the subsystem enough that the old patch should not be replayed
  directly.
- `drop`: do not preserve the old fork behavior on the target branch.
- `reverted`: historical context only; do not replay as-is.

## Target Base Validation

The SHA `bed9f269312151481cd67a8d21b70e0f52424c2b` is a valid upstream Dynamo
commit:

```text
bed9f269312151481cd67a8d21b70e0f52424c2b feat(mocker): add AIC forward-pass engine perf shim (#10150)
```

I could not fetch an upstream branch or tag literally named `main-v1.2.0` or
`v1.2.0`. The precise plan should therefore be: create a local Baseten
`main-v1.2.0` branch from the upstream SHA above.

```bash
git fetch https://github.com/ai-dynamo/dynamo bed9f269312151481cd67a8d21b70e0f52424c2b
git switch -c main-v1.2.0 bed9f269312151481cd67a8d21b70e0f52424c2b
```

## Target Decision Summary

| Patch | Target decision | Reason |
| --- | --- | --- |
| `PATCH-001` CI/container | Keep, retarget branch names | Fork image publishing and CI still need Baseten-specific wiring. |
| `PATCH-002` runtime/shutdown | Keep behavior, port manually | Shutdown/drain invariants still matter; target code has moved. |
| `PATCH-003` NATS/JetStream/discovery | Mostly discard or minimize | Target has P2P standalone-indexer recovery; NATS is less central. Keep JSON pub/sub, but drop Python JetStream object storage because offload is no longer used. |
| `PATCH-004` router core/queueing | Redesign, do not replay old queue stack | Target already has tiered ISL queue config and P2P recovery. Preserve only missing Baseten policy knobs after testing. |
| `PATCH-005` metrics/tracing | Keep | Added metrics should be preserved across versions. |
| `PATCH-006` protocols | Mixed: follow v1.1 by default, test exceptions | Target still rejects `stream_options` without streaming, and that is acceptable. Target has modern tool/reasoning parsing; port only Baseten/client-facing gaps found by tests. |
| `PATCH-007` Python APIs | Keep relevant API behavior | Holding streams until first token for Python webserver engines remains useful; restore the custom Python worker selector because routing still has the trait hook. |
| `PATCH-008` model/parser/vLLM | Mostly review-upstream | Target already has broad parser/reasoning support; do not replay old version bumps blindly. |
| `PATCH-009` validation/limits | Mixed | TCP limit is configurable upstream, but default is 32 MiB at target SHA. Preserve 256 MiB by env/config or carry a default change. |
| `PATCH-010` logging | Keep structured logging | Preserve request correlation and structured fields; avoid old noisy level churn. |
| `PATCH-011` upstream syncs | Drop as direct patches | Use only as audit hints; many behaviors are already represented upstream. |
| `PATCH-012` reverted experiments | Drop | Historical context only. |

## Alignment with `main-v1.1.0`

The refreshed Baseten `main-v1.1.0` branch follows the same strategy this
ledger recommends: advance to upstream first, then replay Baseten changes as a
short series of logical commits instead of preserving the old raw commit
sequence. After upstream commit `cc5b2cd29` (`chore: bump version references to
v1.1.0`), the Baseten replay is distilled into these groups:

| `main-v1.1.0` replay commit | Ledger group |
| --- | --- |
| `9e5806c82` CI/fork-survival workflows, build script, version stamping | `PATCH-001` |
| `14e94a0f9` runtime resilience, lifecycle ordering, transport hardening | `PATCH-002`, `PATCH-009`, `PATCH-010` |
| `ce5dfa1c9` OTel exporter env handling | `PATCH-005`, `PATCH-010` |
| `1de204080` B10 worker selector, hot-reloadable config, snapshot toggle | `PATCH-004`, `PATCH-005` |
| `a3f024b40` hot-reloadable queue threshold, queue metrics, `respond` result | `PATCH-004`, `PATCH-005` |
| `6f36f18d4` OpenAI protocol extensions, `baseten_ext`, B10 health/rate-limit | `PATCH-006` |
| `d493fef1d` Anthropic protocol conformance | `PATCH-006` |
| `9adbfdc64` B10 router and service pipeline Python bindings | `PATCH-007` |
| `9cfeeb5e2` `JsonPublisher` / `JsonSubscriberIter` Python bindings | `PATCH-003`, `PATCH-007` |
| `e127637c2` dispatchable BIS Dynamo Image Push workflow | `PATCH-001` |

What is the same:

- The replay unit is a behavior area, not a historical commit.
- CI/image publishing is carried as fork-specific infrastructure.
- Runtime resilience, observability, router policy, protocol compatibility, and
  Python API surfaces remain the major Baseten-owned areas.
- The branch keeps source commit intent alive while producing a cleaner patch
  queue.

What differs for `main-v1.2.0`:

- The target SHA already has P2P standalone-indexer recovery and tiered ISL
  queueing, so the `main-v1.1.0` router/NATS replay should be treated as audit
  evidence first, not copied wholesale.
- The target has newer tool/reasoning parsing and first-token APIs, so the
  OpenAI/Anthropic/Python patches should be test-driven deltas rather than a
  direct port of the v1.1 commits.
- The TCP message-size default remains a concrete Baseten delta: target
  upstream is configurable but defaults to 32 MiB, while Baseten wants 256 MiB.
- The `465795594` WIP data snapshot on `main-v1.1.0` is not a model for the
  long-term patch ledger; preserve it only if those profiler artifacts are
  intentionally required.

## Recommended First Wave for `main-v1.2.0`

For the target SHA reviewed here, the maintainable patch queue should be much
smaller than the historical commit list. Start with concrete Baseten deltas and
use tests to prove whether broader subsystems still need work:

1. Carry the small TCP default patch from `PATCH-009`: target upstream exposes
   `DYN_TCP_MAX_MESSAGE_SIZE`, but defaults to 32 MiB. Baseten should default
   to 256 MiB unless deployment config is guaranteed to set the env var.
2. Port `PATCH-001` CI/container changes that are still branch- and
   registry-specific. Retarget branch literals to `main-v1.2.0`.
3. Preserve structured logging and request correlation from `PATCH-010`, but
   only add fields or level changes that are missing on the target.
4. Build a metric inventory for `PATCH-005` and port missing metric names or
   labels that dashboards depend on.
5. Test target runtime shutdown/drain behavior before porting `PATCH-002`; keep
   only missing invariants.
6. Follow the v1.1 protocol decisions by default, then test protocol and
   Python first-token behavior before porting `PATCH-006` or `PATCH-007`; the
   target already has newer tool/reasoning parsing and first-token APIs.
7. Treat `PATCH-003` and most of `PATCH-004` as audit-only initially. The
   target has P2P recovery and tiered ISL queueing, so only carry forward
   Baseten-specific knobs that remain absent after testing.

## Historical Ledger Order

1. `PATCH-001`: Fork CI, release, and container build infrastructure
2. `PATCH-002`: Runtime lifecycle, transport resilience, and graceful shutdown
3. `PATCH-003`: NATS, JetStream, and discovery compatibility
4. `PATCH-004`: Baseten router core, hot reload, scheduling, and queueing
5. `PATCH-005`: Router metrics, tracing, and observability
6. `PATCH-006`: OpenAI, Anthropic, and Baseten protocol compatibility
7. `PATCH-007`: Python bindings and service-facing APIs
8. `PATCH-008`: Model, parser, vLLM, and multimodal compatibility
9. `PATCH-009`: Validation, limits, and operational compatibility knobs
10. `PATCH-010`: Logging policy and production signal cleanup
11. `PATCH-011`: Upstream syncs and likely already-upstream patches
12. `PATCH-012`: Reverted or do-not-replay experiments

Use this order for reading the ledger and understanding dependencies. For the
actual `main-v1.2.0` replay, use the smaller first-wave order above and only
expand into these groups when tests show a concrete gap.

## PATCH-001: Fork CI, Release, and Container Build Infrastructure

Status: `keep`

Source commits:

- `f2076864a` chore: CI cleanup, build environment, and infrastructure setup
- `396e5a58b` chore: skip fern docs release-version job on fork
- `24c32bf91` ci: target main-v1.0.0 instead of main for pre-merge push triggers (#161)
- `88319e629` ci: add post-merge build for framework=none image (#162)
- `2546311a1` ci: use depot runner for post-merge build (#163)
- `1309b86da` feat(container): stamp dynamo version into image (#215)
- `0e3ac0d0e` ci: add dispatchable BIS Dynamo Image Push workflow (#252)
- `b76d509d2` container: skip stale vllm hotfix on newer versions

Purpose:

Make upstream Dynamo build, test, and image workflows usable for the Baseten
fork and release branches. This includes removing irrelevant upstream workflows,
retargeting CI to release branch names, adding post-merge image builds, stamping
image versions, adding BIS image push dispatch, and avoiding stale container
hotfixes on newer dependency versions.

1.0 -> 1.1 behavior:

The v1.1 upgrade kept this as fork-owned infrastructure. Commit `9e5806c82`
removed upstream-only workflows, retargeted CI to `main-v1.1.0`, restored the
fork `container/build.sh`, added `.version-base` and `tools/version-stamp.sh`,
and kept container-template changes needed for Baseten image builds. Commit
`e127637c2` then added the dispatchable BIS Dynamo Image Push workflow.

For v1.2, follow the v1.1 strategy: keep the fork CI/image/versioning layer and
retarget all branch literals to `main-v1.2.0`. Re-audit one-off v1.1 fixes such
as LFS fixture removal, stale dependency workarounds, and clippy workarounds
instead of carrying them mechanically.

v1.2 implementation note:

The fork CI/image layer was replayed as a branch-retargeted infrastructure
patch. The branch carries Baseten workflow filtering, post-merge image build
wiring, BIS image push dispatch, `.version-base`, `tools/version-stamp.sh`,
`container/build.sh`, and container template version stamping. Upstream-only
workflow churn was not preserved as a Baseten patch unless it affected fork
builds.

The image tag resolver now also uses `.version-base` as the first fallback when
the checkout has no merged semver tag or release branch. This keeps v1.2 branch
builds on `v1.2.x.dev.<sha>-<suffix>` instead of falling back to
`v0.0.1.dev.<sha>-<suffix>`.

A required `Baseten Changelog Check` GitHub Actions workflow now runs on every
PR targeting `main-v1.2.0` and fails if `baseten-changelog.md` is not modified.
PRs that legitimately do not need a changelog entry can bypass the check by
applying the `skip-changelog` label. This makes the patch ledger an enforced
artifact of the fork rather than a documentation convention.

The post-merge `framework=none` image workflow now builds native `amd64` and
`arm64` images on Depot runners, pushes arch-specific tags, and publishes the
unsuffixed tag as a multi-arch manifest. The workflow does not publish a mutable
`latest` tag.

Replay notes:

Port this first so the new branch has a working CI and image path. Retarget all
branch-name literals, for example from `main-v1.0.0` to `main-v1.2.0`. Do not
blindly replay old dependency workarounds; keep only the ones still needed by
the target container stack.

## PATCH-002: Runtime Lifecycle, Transport Resilience, and Graceful Shutdown

Status: `keep`

Source commits:

- `4a97a52bf` feat: runtime resilience and transport hardening
- `eda65f1ba` fix: bump instance-down log levels to info for production observability
- `5f0d465aa` fix: spawn health heartbeat before blocking on worker discovery (#165)
- `a36209650` fix(runtime): prevent NATS/ETCD teardown during HTTP request drain (#184)
- `98ee65835` Merge pull request #247 from basetenlabs/trid/graceful-dyn10
- `6f49732b7` runtime: drain inflight requests before stopping push endpoints
- `e1574723e` runtime: unpublish draining endpoints from discovery
- `59a6c1582` network host parameter (makes dev on vultr more consistent)
- `0bf331f55` Merge pull request #244 from basetenlabs/aracharl/network-host-param

Purpose:

Harden runtime lifecycle behavior for production serving. This group adds
structured context stop/kill reasons, graceful request draining, health
heartbeat startup ordering, push endpoint shutdown behavior, discovery
unpublication while draining, runtime dependency lifetime fixes during HTTP
drain, network host configurability, and less noisy instance-down reporting.

1.0 -> 1.1 behavior:

The v1.1 upgrade kept runtime hardening in `14e94a0f9`. That replay preserved
graceful drain behavior, stop/kill reason APIs, shutdown ordering fixes, health
heartbeat startup ordering, etcd reconnect/watch recovery, NATS
NoResponders/auto-resubscribe handling, the 256 MiB TCP framing default,
worker-pool default bumps, and selected runtime log cleanup. Where upstream
v1.1 already had shared TCP max-message-size helpers, the fork dropped older
duplicated helper code and kept upstream structure.

For v1.2, split this decision. Keep the 256 MiB TCP default. Test target
startup, drain, endpoint unpublication, and shutdown behavior before porting
lifecycle code. Do not preserve NATS recovery pieces as an objective; the
v1.2 direction follows the no-NATS/P2P recovery path unless a concrete
non-P2P production gap is found.

v1.2 implementation note:

The concrete v1.2 runtime replays are intentionally narrow so far: the TCP
request-plane max message default is restored to 256 MiB, and target shutdown
lifecycle logs now carry the `unified_model_logs` marker at the lifecycle points
that still exist. The broader v1.0/v1.1 NATS recovery stack is not a replay
objective for v1.2. Additional drain/unpublication behavior should only be
ported after a target-specific failing test or production gap is identified.

The Python `dynamo_worker` decorator also accepts the Baseten
`register_shutdown` compatibility keyword again and wires it to SIGINT/SIGTERM
runtime shutdown hooks. This is a narrow binding compatibility restore rather
than a broader NATS lifecycle replay.

Runtime shutdown Phase 2 now has a Baseten-owned safety cap around the
graceful endpoint drain wait. If `tracker.wait_for_completion()` does not
finish, the runtime logs the remaining graceful endpoint count and proceeds to
Phase 3 teardown instead of hanging forever behind a deadlocked in-flight
request. The cap is controlled by `DYN_RUNTIME_GRACEFUL_SHUTDOWN_TIMEOUT_SECS`;
Baseten uses a 4 minute default (`240` seconds), not the 15 minute value discussed for the upstream proposal in dyn1.3+.

Replay notes:

Port behavior, not necessarily implementation. Upstream may have refactored
runtime ownership, endpoint lifecycle, or discovery. The important invariant is:
stop advertising before shutdown, allow in-flight work to drain, keep required
runtime dependencies alive during drain, and expose enough lifecycle signal to
debug shutdowns.

Validation:

Exercise startup health, request drain, endpoint unpublication, NATS/ETCD
lifetime during drain, and forced shutdown timeout behavior.

## PATCH-003: NATS, JetStream, and Discovery Compatibility

Status: `redesign`

Source commits:

- `3d09914c7` Merge pull request #160 from basetenlabs/blarson/backport-release-0.6.0
- `b7ce2ca07` feat(nats): expose JetStream object store to Python for large embedding offload
- `481029400` revive lost commits
- `5ed0b1400` fix config stomping issue
- `a45d0585f` fix: regenerate python bindings Cargo.lock for --locked CI
- `aea5b6af7` fix: add nats_client accessor and update Cargo.lock
- `a1f8300a9` nats getter
- `9d34e2fee` dedupe
- `102c7b07b` fix: pin cudarc to =0.19.3 to fix cuda.rs compilation
- `d75aaad45` fix(kv-router): prevent router NATS consumers from deleting each other on simultaneous startup (#173)
- `45c78f781` fix(kv-router): restore etcd-based alive registry for orphan cleanup (#179)
- `bfd4d0772` feat: sync upstream PR #143 and #146 - snapshot metrics + NATS stream config (#149)

Purpose:

Keep discovery, NATS, and JetStream behavior stable for Baseten deployments.
This includes JetStream object store exposure, NATS client accessors, stream
configuration update behavior, default retention tuning, consumer startup race
fixes, and etcd-based alive registry behavior for orphan cleanup.

1.0 -> 1.1 behavior:

The v1.1 upgrade mostly dropped NATS-centric router recovery and JetStream
recovery patches under the no-NATS direction. The important exception was
`9cfeeb5e2`: v1.1 restored `JsonPublisher` and `JsonSubscriberIter` after those
APIs were rewritten on the upstream EventPlane abstraction, making them
transport-agnostic rather than NATS-only.

For v1.2, keep the v1.1 no-NATS decision. Preserve only transport-agnostic
JSON pub/sub behavior that current Baseten code still uses. Do not restore
router recovery mechanisms whose only purpose was NATS/JetStream correctness if
target P2P/EventPlane paths cover the need.

Replay notes:

This group has mixed provenance. Some changes were backports or upstream syncs,
so the target upstream release may already contain them. Audit first, then port
only remaining Baseten requirements. Python JetStream object-store access is not
a remaining requirement for v1.2 because Baseten no longer performs JetStream
offloading.

Editorial notes:

NATS may be increasingly less needed because the primary recovery mechanism is
moving toward Local Router Trie peer-to-peer recovery.

Target assessment:

The target SHA has standalone-indexer P2P recovery paths, including tests that
launch a second indexer with `--peers` and pre-seeded `--workers`. That validates
the editorial note: do not replay NATS-centric recovery work by default. Drop
`b7ce2ca07` for v1.2: the Python JetStream bucket/object-store API was only
needed for large embedding offload, and Baseten no longer uses JetStream
offloading. Keep the existing EventPlane-backed JSON publisher/subscriber
bindings, but do not add new Python NATS bucket storage.

v1.2 implementation note:

The transport-agnostic JSON publisher/subscriber Python bindings were restored
through the current EventPlane-backed implementation. NATS bucket/object-store
Python access was explicitly not restored because JetStream offloading is no
longer used.

Validation:

Run router startup with multiple instances and verify P2P/EventPlane recovery
paths. Do not run or add Python object-store offload tests unless JetStream
offloading is reintroduced as a product requirement.

## PATCH-004: Baseten Router Core, Hot Reload, Scheduling, and Queueing

Status: `redesign`

Source commits:

- `e6a947a52` feat: KV router enhancements for production serving
- `055e6a1d7` fix: address PR review comments on KV router
- `6a0dd058c` Merge pull request #180 from basetenlabs/blarson/best_overlap_blocks
- `21c31b138` wip
- `2e571f9d3` field order
- `2448517f2` fix: populate active request counts before selection (#185)
- `a996149fc` Merge pull request #196 from basetenlabs/mf/fix-dp-routing
- `51b6fb545` fix dp routing stats
- `618215bb4` Merge pull request #212 from basetenlabs/blarson/flexible_queue_algo
- `2fbc428b9` feat(router): export queue metrics from router, add queue wait logging
- `964b49307` feat(router): refresh overlap scores at dequeue time
- `3a51601b0` refactor(router): pluggable queue admission policy
- `1d8715e4d` fix(kv-router): enable metrics feature by default
- `d9b41f280` feat(router): hot-reload router_queue_threshold via B10 config
- `c78ff5755` refactor(router): rate-limit queue threshold hot-reload to every 10s
- `8afa24f06` fix(tests): replace Some(0.0) threshold with Some(f64::EPSILON) in queue tests
- `f910229fa` refactor(router): review fixes for hot-reload threshold
- `a7f06652d` chore(router): remove logging from queue hot path
- `0a8e69660` Merge pull request #218 from basetenlabs/blarson/router-queue
- `7f2ce7561` fix(kv-router): prevent ghost sequences and ensure expiry triggers queue drain
- `55ecf1995` refactor(kv-router): respond() returns Result<(), RespondError> instead of bool
- `3f9c998c5` kv-router: cap pending queue depth to prevent mark_free deadlock
- `de0b0d1da` kv-router: hot-reload queue threshold only when explicitly configured
- `70d7bba89` Merge pull request #219 from basetenlabs/blarson/router-queueing
- `d852cdedc` kv-router: fix TCP starvation of mark_free and add queue timeout safety net
- `c09131020` kv-router: add instrumentation logs for queue starvation diagnosis
- `c8e21dff1` kv-router: remove debug-only logs, keep prod-observable fix signals
- `96d4adc4f` kv-router: remove queue wait timeout (covered by pending_cancellations)
- `c2df02e4d` kv-router: reduce TCP pool defaults, plug pending_cancellations leak
- `61b4d1cb8` kv-router: log cleanup - suppress tcp client-drop warn, info for slot cancellation
- `c7c708b91` kv-router: downgrade noisy warn logs to info
- `b52a41cb7` kv-router: remove noisy mark_free not-in-tracker log
- `729a81cd6` kv-router: log cleanup - mark_free info, remove pool utilization, simplify queue-full msg
- `c9bf0c4ed` fix: Make router active replicas hot configurable (#251)

Purpose:

This is the main Baseten router patch. It adds production serving behavior:
lazy worker removal, Baseten worker selection, hot-reloadable router config,
snapshot controls, best-overlap response fields, active request counts before
selection, DP routing stats, pluggable queue admission, queue metrics, overlap
refresh at dequeue time, queue threshold hot reload, ghost sequence prevention,
pending queue caps, caller disconnect handling, TCP starvation fixes, and active
replica hot configuration.

1.0 -> 1.1 behavior:

The v1.1 upgrade split router work into `1de204080` and `a3f024b40`. It kept a
simplified `B10WorkerSelector`, hot-reloadable B10 config, runtime DP/TP sizing
from that config, `PotentialLoads`, queue-threshold hot reload, queue metrics,
and `respond() -> Result` cleanup. It deliberately dropped Python selector
support, `PyWorkerSelectionResult`, `dp_strict_rank`, DP-heavy active-request
scoring, softmax sampling, and old NATS request-plane changes.

For v1.2, follow v1.1 on keeping B10 as the maintained Rust selector, but
diverge from v1.1 by restoring arbitrary Python worker selector callbacks.
The target scheduler still routes through the
`WorkerSelector<ModelRuntimeConfig>` trait, so the clean v1.2 shape is a custom
trait-object selector variant rather than Python-specific logic in the router.
`RouterConfig(..., algo_selector="B10")` selects the Rust `B10WorkerSelector`;
`RouterConfig(..., algo_selector="Python", python_worker_selector=...)` selects
the Python bridge. Preserve `dp_strict_rank` through the router
request/response/scheduling path so B10 and Python callbacks can request strict
DP-rank routing. Still follow v1.1 on dropping NATS recovery and avoid replaying
the old queue stack where target tiered ISL queueing already provides the
queueing mechanism. DP routing and routing policy are hard to test, so preserve
the production policy where it is already wired through B10.

v1.2 implementation note:

Followed v1.1 for `router_queue_threshold` hot reload, adapted to the target's
new actor-based router queue instead of copying the old `RwLock` queue shape.
The B10 config map now accepts root and override-group `router_queue_threshold`
values; the scheduler polls the hot-reloadable B10 config every 10 seconds and
updates the queue actor without restart. Positive values enable queueing at the
new threshold, while `0` or `None` disables queueing. When queueing is disabled
after requests are already pending, the actor drains them immediately so the
target queue cannot strand requests behind a now-disabled threshold.

The current v1.2 branch restores arbitrary Python worker selectors after review.
This intentionally differs from v1.1. The implementation keeps the bridge in a
separate Python-binding module and connects it to the Rust router through the
same `WorkerSelector<ModelRuntimeConfig>` trait used by `DefaultWorkerSelector`
and `B10WorkerSelector`. The callback receives the worker map plus a
`PySchedulingRequest` and returns `PyWorkerSelectionResult`; Rust validates the
chosen worker/rank against routing eligibility before returning a strict-DP
selection result.

The standalone B10 `start_router` binding was restored in the v1.1 shape rather
than as Python router logic: Python cheaply reexports `dynamo._core.start_router`,
and the Rust binding starts the current `KvRouter` with either
`B10WorkerSelector` or `DefaultWorkerSelector`. The B10 config map again owns
`router_active_replicas` at the root and override-group levels so router
activation can be hot configured without a Python-side implementation.
The v1.0 `router_disable_snapshots_in_primary` knob was also restored: the
Python/Rust config accepts the flag, and the active B10 router disables its
JetStream snapshot loop after winning the active-router gate.

The v1.2 local-indexer worker-query recovery path now honors
`skip_initial_worker_wait`: by default, the B10 router waits for the initial
worker-query KV recovery to complete before registering the serving `generate`
endpoint. This is intentionally scoped to the local-indexer/event-plane path;
the deprecated JetStream recovery path is left unchanged.

The router bookkeeping protocol now accepts a `request_id` payload override for
`MarkPrefill`, matching the existing `MarkFree` behavior when the transport
context id cannot be used. `PotentialLoads` responses also report the current
router queue backlog through `pending_count` and `pending_isl_tokens`, so callers
can inspect both worker load projections and queued-work pressure in one request.
Each `PotentialLoad` row also carries the worker's `active_requests` count,
matching the v1.0 bookkeeping surface used by autoscaling consumers.

The B10 selector heuristic must stay compatible with v1.0 for a seamless router
upgrade. The v1.2 port therefore restores the v1.0 active-request term, DP
strict-rank decision, softmax temperature sampling, and throttled score
breakdown logs while adapting the cache-hit input to the target scheduler's
`effective_cached_tokens` model. v1.0 did not distinguish cache tiers for this
heuristic: every cache hit received full routing credit. With host/disk
cache-hit weights set to `1.0` and no active queue/load, the selector preserves
that behavior: prefill load is based on all cache hits at full credit, decode
load is unchanged, and cache-miss tokens are computed from the same effective
cached-token signal. This is needed so deployments can turn on the v1.2 router
without changing worker placement behavior except where the new tiered-cache
weights or eligibility constraints are intentionally configured.

Taint-pool routing note:

Add `Endpoint.list_endpoint_taints(only_live=True)` to the Python binding. It
should return a same-endpoint MDC taint snapshot as `dict[worker_id,
set[taint]]` by reading `DiscoveryQuery::EndpointModels`, deserializing each
`ModelDeploymentCard`, and extracting `runtime_config.taints`. With
`only_live=True`, filter the snapshot through a direct
`DiscoveryQuery::Endpoint` instance list, not a newly created endpoint client.
Startup code can use this primitive before `register_model` to choose and
advertise a pool taint such as `fast` or `slow`.

The active token-load discounting from the Baseten router is also required for
v1.2. Upstream Dynamo does not currently carry this behavior, but without it the
router over-penalizes workers that already have active decode/prefill load and
under-emphasizes the incremental load from the next request. This is a major
routing flaw for agentic workloads, where repeated short requests need the new
request's potential load to dominate the placement decision while existing
worker load is discounted by the hot-reloadable B10 prefill/decode factors.
The discount acts only on the current load, not potential load. Potential load
is more important because it is about to be added.

Heuristic and selector parity note:

- `softmax_sample` accepts any worker-logit map that can be iterated as
  `(&WorkerWithDpRank, &f64)`, instead of requiring `FxHashMap`.
- The B10 worker selector can use the standard `HashMap` for its local logits
  map, so `dynamo-llm` does not need to depend on `rustc-hash` for this path.
- The zero-temperature selection path stays allocation-free; only the softmax
  sampling path collects entries.

Replay notes:

Treat this as one coherent router subsystem port. Do not cherry-pick the commits
one by one unless the target upstream code is very close. Start by porting the
configuration model, then worker selection, then queue/admission behavior, then
bug fixes, then metrics/logging surfaces. Preserve final behavior and ignore
intermediate churn.

Editorial notes:

Because the main recovery path is moving to peer-to-peer mode, the old
NATS-backed router recovery patches are expected to be less relevant. Router
queueing also has tiered ISL queues upstream. The old DP routing stats fix does
not need to be replayed as a standalone patch if actual values can come from
MDC/configmap.

Target assessment:

The target SHA already has `router_queue_by_incoming_missing_isl`,
`router_queue_policy`, and P2P recovery coverage. Do not replay the old router
queue stack wholesale. Re-test target upstream behavior first, then port only
missing Baseten requirements such as specific hot-reload knobs, active-replica
configuration, or selection/response fields that are still absent. Do not carry
the DP routing stats patch unless target deployments prove they cannot source
the same values through MDC/configmap.

Current replay delta for Baseten `main-v1.2.0`:

When replaying this PR from Dynamo upstream onto Baseten `main-v1.2.0`, treat
the selector and queue changes as one compatibility unit. The target branch
already contains the v1.2 scheduler shape and Baseten selector parity work, so
the replay should preserve that shape and add only the request fields and
admission behavior described below.

- `SchedulingRequest` carries an `active_requests` snapshot populated at
  admission before invoking the selector. This intentionally matches the v1.0
  cost model, where active requests were counted by scanning the active
  request-to-worker map and materializing a per-worker map before selection.
  The cost is still an O(active requests) scan plus a fresh map allocation, but
  it is not a new regression relative to the Baseten v1.0 router behavior.
- `B10WorkerSelector` keeps the active-request blend hot-reloadable through
  `router_active_request_dp_blend`. The default is `2/3` DP-wide mean active
  requests across the worker's DP ranks and `1/3` selected DP-rank active
  requests. Non-finite values fall back to the default and out-of-range values
  are clamped with error-level logs. The score also retains the absolute
  cache-miss token term and the short-request full-miss bypass.
- `softmax_sample` is public and generic over worker-logit maps that iterate as
  `(&WorkerWithDpRank, &f64)`. That keeps the shared scheduler helper usable by
  both the core router and the B10 selector without forcing `dynamo-llm` to
  carry a `rustc-hash` dependency solely for this path.
- The B10 selector uses `HashMap<WorkerWithDpRank, f64>` for its local logits
  map. Zero-temperature selection remains deterministic by choosing the lowest
  logit and breaking ties by `(worker_id, dp_rank)` without allocating an
  additional entries vector. Non-zero temperature still delegates to softmax
  sampling.
- The selector keeps the v1.0 strict-DP decision: strict rank is returned when
  the selected rank has more than a 5% score advantage over the worst
  same-worker alternative.

Queue admission fields to replay:

- Add `priority_load_shed_percent: u8` to `RouterRequest::New`, with
  `#[serde(default, skip_serializing_if = "is_default_priority_load_shed_percent")]`
  and default `0`. Thread the value through `KvRouter::find_best_match_details`,
  `KvRouter::find_best_match`, `KvRouterScheduler`, `LocalScheduler`, and into
  `SchedulingRequest`.
- `priority_load_shed_percent` is only meaningful together with
  `priority_jump > 0.0`. The queue's `tier_cap_for_request` should leave the
  cap unchanged unless both are set. When both are set, compute the boosted cap
  as `cap + cap * priority_load_shed_percent / 100`, using saturating arithmetic.
- The cap being boosted is the existing tiered pending-ISL rejection cap from
  `router_queue_by_incoming_missing_isl`, selected from the request's effective
  cache-miss tokens and the registered worker count. This is a load-shed grace
  margin for priority requests: it allows a priority request to enter a slightly
  fuller pending queue before returning `MaxQueuedIslTokensExceeded`.
- `priority_load_shed_percent` does not change queue ordering. Queue ordering
  still comes from `priority_jump` through the queue policy's enqueue key. The
  percent only changes the rejection threshold used when the queue is already
  active and pending-ISL caps are configured.
- Backpressure responses for cap rejection should report the cap that was
  actually applied to that request. For a priority request this means
  `max_queued_isl_tokens` can be the boosted cap, not the base tier cap.
- Add `do_not_queue: bool` to `RouterRequest::New`, with
  `#[serde(default, skip_serializing_if = "is_false")]` and default `false`.
  Thread the value through the same scheduler path into `SchedulingRequest`.
- `do_not_queue` preserves queue-by-default behavior. If the field is omitted,
  the router should behave exactly as before and may park the request in the
  pending queue.
- When `do_not_queue: true`, only reject at the point where the queue would
  otherwise park the request because queueing is enabled and all eligible
  workers are prefill-busy. Do not reject requests that can be scheduled
  immediately. Do not change pinned-worker or allow-list eligibility checks.
- The `do_not_queue` rejection is surfaced as
  `RouterBackpressureReason::DoNotQueue` with the current
  `queued_isl_tokens` and no max cap. This keeps it distinguishable from
  `MaxQueuedIslTokensExceeded` for callers and metrics.
- Existing OpenAI/preprocessed request paths can pass `do_not_queue: false`
  unless the replay also adds a higher-level request hint. Mocker replay paths
  should also pass `false` to preserve prior behavior.

Compatibility and API notes:

- `RouterRequest::New` remains backwards-compatible for older JSON clients:
  omitted `priority_jump`, `priority_load_shed_percent`, and `do_not_queue`
  deserialize to `0.0`, `0`, and `false` respectively.
- Default serialization omits `priority_jump`, `priority_load_shed_percent`,
  and `do_not_queue`, so request JSON remains compact and older wire payloads
  are not churned.
- `RouterBackpressureReason` is now part of the behavior contract for queue
  opt-out. Downstream clients that match reasons should tolerate the new
  `do_not_queue` snake-case value.

Conflict/rebase notes:

- If `b10_worker_selector.rs` conflicts during replay, keep the
  `B10WorkerSelector` implementation with active-request scoring, softmax
  temperature sampling, throttled score logs, and the `(worker_id, dp_rank)`
  zero-temperature tie-break. Do not revert to the simpler upstream-style
  selector that omits active-request scoring.
- If duplicate `active_requests_for` methods appear in
  `scheduling/types.rs`, keep a single method. Some target branches may already
  have this helper from the selector parity replay.
- If `SchedulingRequest` construction fails after adding fields, update all
  request literals and helper constructors with `priority_load_shed_percent: 0`
  and `do_not_queue: false` unless the test is specifically exercising those
  fields.

Validation:

Run router unit tests plus manual or integration coverage for: hot config
reload, queue threshold changes, cancellation, caller disconnect, queue-full
behavior, `mark_free`/`add_request` races, DP routing, active replica changes,
and best-overlap response fields.

Focused validation from the current replay:

- `cargo fmt -- --check`
- `cargo test -p dynamo-kv-router test_router_request_new_do_not_queue_defaults_to_false --lib`
- `cargo test -p dynamo-kv-router test_do_not_queue_backpressures_instead_of_queueing --lib`
- `cargo test -p dynamo-llm b10_worker_selector --lib`

## PATCH-005: Router Metrics, Tracing, and Observability

Status: `keep`

Source commits:

- `40b1a7aed` feat: add kv_router.select_worker tracing event with routing metrics (#153)
- `3723174f5` refactor(kv-router): migrate select_worker span event to span attributes (#182)
- `05ecb882f` fix: restore missing router and indexer metrics on ComponentMetricsServer (#169)
- `36133139e` Merge pull request #223 from basetenlabs/blarson/router_metrics
- `d17293eeb` feat(kv-router): add orphan-expired metric, fix indexer ops gaps
- `2b5248db0` trim: drop speculative instrumentation
- `b0c38598e` cleanup: store metrics on ConcurrentRadixTree, rename to force_expired_request_count
- `8e575bd55` rename: force_expired_request_count -> force_expired_requests
- `bd3531d98` remove redundant section header above KvRouterReliabilityMetrics
- `ce3a100d8` cargo fmt
- `24c048a47` fix(dynamo): Make OTel log exporter opt-in to prevent BatchLogProcessor errors (#157)
- `316432aa1` fix(dynamo): Fall back to OTEL_EXPORTER_OTLP_ENDPOINT for trace export (#171)

Purpose:

Preserve Baseten's production observability surface. This includes router
selection span attributes, router and indexer metrics restoration, indexer ops
metric coverage, orphan/force-expired reliability metrics, and OTel defaults
that avoid noisy log exporter errors while honoring standard trace endpoint
configuration.

1.0 -> 1.1 behavior:

The v1.1 upgrade kept only concrete observability surfaces. Router/indexer
Prometheus constants landed in `1de204080`, queue wait metrics landed in
`a3f024b40`, and OTel exporter environment behavior landed in `ce5dfa1c9`.
Speculative or unconsumed instrumentation was not preserved.

For v1.2, follow that filtering rule. Preserve metric names, labels, and trace
fields that dashboards or production debugging depend on. Add a metric inventory
before replaying code, and avoid carrying metrics that have no known consumer.

v1.2 implementation note:

The OTel exporter environment behavior from v1.1 was restored: trace export
falls back from `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` to the standard
`OTEL_EXPORTER_OTLP_ENDPOINT`, and log export remains opt-in through a dedicated
logs endpoint so enabling tracing does not create noisy BatchLogProcessor
errors. Router queue and selector observability are carried through the B10 and
queue-threshold replay slices; no separate speculative metrics patch was added.

The standalone B10 router path registers the global worker-load and router queue
metrics against its component metrics registry so the explicit router metrics
port exposes the same production counters and gauges expected by v1.1-era
dashboards.

The standalone B10 router path also initializes deferred Rust logging when
`OTEL_EXPORT_ENABLED=1`. `_core` defers `dynamo_runtime::logging::init()` under
OTel until a Tokio runtime exists, and the B10 router creates a Rust `Worker`
directly instead of constructing a Python `DistributedRuntime`. Initializing
logging after `Worker::from_settings()` preserves Rust `tracing` startup logs
and OTLP trace export for router/distributed startup.

Worker-based KV recovery now has a lightweight `RecoveryProcessLogger` that
reports aggregate restore progress, recovered event counts, and the final
initial-recovery completion summary used by router startup gating.

Editorial notes:

All metrics that Baseten added should be preserved across versions.


Replay notes:

Port after PATCH-004 so metric sources exist. Keep final metric names and labels
stable unless dashboards are updated at the same time. Check target upstream for
equivalent OTel defaults before replaying the OTel pieces.

Validation:

Confirm Prometheus scrape output, dashboard metric names, router selection trace
attributes, and OTel behavior with and without explicit log exporter settings.

## PATCH-006: OpenAI, Anthropic, and Baseten Protocol Compatibility

Status: `mixed`

Source commits:

- `5a08107e1` feat: OpenAI protocol extensions and HTTP service enhancements
- `da9d51d0e` fix: address PR review comments on HTTP service and publisher
- `402f5b47f` fix: silently ignore stream_options when stream is not true (#168)
- `863caa9b2` fix(dynamo): Replace strict unknown parameter rejection with warn-and-ignore (#174)
- `ccc31c543` fix(aggregator): accumulate tool call arguments across incremental streaming chunks (#188)
- `603bd5e09` fix: remove double reasoning parse from Anthropic streaming handler (#193)
- `2829d154b` Merge pull request #199 from basetenlabs/trid/anthropic-thinking-v1.0.0
- `0b2ff53fa` dyn1.0.0 mirror changes ant thinking, optional reasoning, move thinking from chat.rs to baseten
- `6d828bb89` dyn1.0.0 mirror changes ant thinking, optional reasoning, move thinking from chat.rs to baseten
- `f1bec62b2` Merge pull request #201 from basetenlabs/mf/v1.0-logprobs-dynamic-temperature
- `fce8920c3` backport logprobs token ids and typed dynamic temperature
- `06f715be8` small patches
- `550606875` small patches
- `bc6fb7455` Merge pull request #203 from basetenlabs/mf/v1.0-b10-extension-required
- `788b55c90` added stuff around extnesion
- `9b279389a` small patches
- `fa3f07ad1` fix tests
- `25ce09585` fix(anthropic): [DONE] leak, toolu_ id prefix, and engine-error status (#208)
- `01841858f` feat(async-openai): OpenAI protocol compatibility fixes for agentic workloads (#221)
- `3e42842b0` feat(async-openai): widen ReasoningEffort for DeepSeek V4 (#220)
- `d104200bf` fix(anthropic): gate inline tool_use stop on parseable accumulated args (#232)
- `d637b3f39` Adding reasoning block for openai requests (#243)

Purpose:

Preserve Baseten and OpenAI/Anthropic compatibility behavior at the HTTP and
protocol layers. This includes Baseten request/billing/priority extensions,
rate limiting, health checks, endpoint activation controls, tolerant validation,
`stream_options` tolerance, grammar response formats, reasoning fields,
Anthropic thinking blocks, dynamic temperature, logprobs token IDs, streamed
tool-call argument accumulation, Anthropic stream correctness, async-openai type
compatibility, and DeepSeek V4 reasoning effort support.

1.0 -> 1.1 behavior:

The v1.1 upgrade kept the Baseten protocol surface in `6f36f18d4`: `baseten_ext`
fields, B10 health, B10 rate limiting, selective endpoint activation, required
Baseten extension validation, and warn-and-ignore handling for unsupported
fields. It dropped old `async-openai` fork edits, old aggregator tool-call merge
patches, grammar/structural response format variants, token-id/logprobs response
variants, and request-id flavoring that no longer matched upstream. Commit
`d493fef1d` kept Anthropic conformance fixes: no OpenAI `[DONE]` sentinel on
Anthropic streams, `toolu_` IDs, and backend status passthrough.

For v1.2, follow v1.1 more closely by default. Preserve `baseten_ext`, health,
rate limiting, and required extension validation if those are still
client-facing. Prefer target/upstream protocol behavior for old
`stream_options`, parser, grammar, logprobs, and response-shape differences.
Reopen only specific dropped behaviors that fail client or compatibility tests.

Replay notes:

This group must be audited against target upstream before porting. Many of
these are likely upstreamable protocol fixes or may already exist in modified
form. Preserve Baseten extension behavior and client-facing response shapes;
drop exact copies of generic compatibility fixes if upstream already implements
them.

Editorial notes:

`stream_options` tolerance does not need to be preserved. Tool-calling
aggregation is brittle and should be tested; ideally preserve any Baseten fixes
that still fail against the target upstream implementation.

Target assessment:

The target SHA still validates that `stream_options` is only allowed when
`stream=true`; accept that upstream behavior and drop the old tolerant
`stream_options` patch. The target also has a much newer frontend parsing stack
for reasoning and tool calls, including buffered post-reasoning tool text and
parity fixtures. Run the brittle tool-calling tests against target first, then
port only failing cases.

v1.2 implementation note:

Followed v1.1 for the B10 health and rate-limit subset. The branch now carries
the standalone B10 health heartbeat/poison state, the `/health_file` route, the
header-driven B10 rate limiter, and Python functions `set_health`,
`is_healthy`, `set_poisoned`, and `set_rate_limit_level`. The rate limiter was
attached to the same endpoint classes v1.1 gated first: completions, chat
completions, and embeddings. The target already has selective endpoint
activation through `HttpService.enable_endpoint(...)`, with Python tests using
it to turn chat on explicitly; do not replay old endpoint activation code unless
new endpoint-specific tests fail. Broader protocol work remains open only for
response-shape or tolerance decisions not already covered by `baseten_ext` and
Anthropic conformance slices.

Baseten owns its own `context_id` format, built at HTTP ingress in the fork-only
file `lib/llm/src/http/service/b10_context_id.rs`. The format is
`{org_namespace}--{b10_request_id}--{model_version_id}[--{extras}]`:

- `org_namespace`: `X-Baseten-Org-Namespace`, falling back to
  `X-Baseten-Billing-Org-Id` (beefeater sets the billing-org header on the
  direct BIS route), else `none`.
- `b10_request_id`: `X-Baseten-Request-Id`, truncated at the first `:` (SEG may
  append a `:cf-ray:user-id` suffix); a UUID when absent.
- `model_version_id`: `X-Baseten-Model-APIs-Version-Id`, falling back to
  `X-Baseten-Model-Version-ID`, else `none`.
- `extras` (optional 4th segment): non-empty `cf_ray`/`user_id` from
  `X-Baseten-Customer-Request-Context`, joined with `:` and sanitized so it
  carries no `--`. Opaque pass-through; present-but-empty headers are treated as
  absent so fallbacks fire.

Workers parse `Context::id()` by splitting on `--` into 3 or 4 parts (the 4th,
`extras`, is ignored); `none` maps back to an empty string. The format is
backward-compatible with the older 3-part `org--request--model_version` build
the gemma image still pins, since that image cannot be rebuilt.

The goal of this format is log correlation: emit the full id as the `context_id`
log field wherever a request is logged, and the narrow `b10_request_id` segment
alongside it, so frontend and worker logs can be joined on a request.

The next v1.2 slice restored the typed root-level `baseten_ext` fields for chat
and completion requests: `b10_cache_control`, `baseten`, `dynamic_temperature`,
and `thinking`. This follows v1.1's "validate-or-400, never silently drop"
decision for Baseten-maintained fields: invalid `dynamic_temperature` keys,
values, or `thinking.type` variants fail during deserialization. Unlike v1.1,
this slice intentionally kept the target branch's stricter unknown-field
validation instead of changing all unknown parameters to warn-and-ignore; later
validation kept that stricter target behavior.

The Anthropic conformance slice follows v1.1 commit `d493fef1d` closely because
all three behaviors are externally visible API compatibility requirements:
Anthropic streams no longer receive the OpenAI-only `[DONE]` sentinel,
Anthropic `tool_use.id` values are minted as `toolu_<uuid>` instead of exposing
backend/OpenAI tool-call IDs, and Anthropic backend errors preserve the backend
HTTP status while rewriting `max_completion_tokens` wording to `max_tokens`.
This intentionally does not reopen unrelated v1.0 protocol tolerance patches.

Cherry-picked `d104200bf` (#232) onto v1.2.0 as #296: gate the inline Anthropic
`content_block_stop` for `tool_use` on the accumulated `input_json_delta` args
parsing as a complete JSON value. Without this, incremental-parser backends
(`glm47`, `minimax_m2`, `kimi25`, `qwen3_coder`) emit `content_block_stop` after
the first delta, so Anthropic SDK consumers (Claude Code) discard the trailing
deltas and see `tool_use.input == {}`, looping on `InputValidationError`. Drop
this patch once the equivalent guard lands upstream in `ai-dynamo/dynamo`'s
`stream_converter.rs` (track via upstream port of #232).

Validated the remaining PATCH-006 tolerance/parser decisions on v1.2. Keep the
old warn-and-ignore unknown-field patch dropped: target tests still reject
unsupported chat/completion fields, and that is the intended stricter behavior
unless a client test proves otherwise. Keep the old non-streaming
`stream_options` tolerance dropped as documented above. Keep the old
incremental tool-call aggregation patch dropped as a separate replay item: the
target's current streaming parser suite passes across GPT-OSS/Harmony,
DeepSeek, Kimi, Qwen, and Nemotron captures. The GPT-OSS tests require the
`openai_harmony` tokenizer vocab to be cached or downloadable; without network
access they fail before exercising Dynamo behavior.

Also: carry `ErrorMessage::from_http_error` 5xx pass-through (BIS-165/#272) to each future fork — it allows 529 (site overloaded) to reach clients instead of being squashed to 500.
Also should include error classsification for metrics, so that we can observe it as site-overloaded etc in metrics.

The response-only null-omission patch should be preserved on v1.2. The target
already omits absent `Choice.logprobs` and `Choice.finish_reason` through
upstream `async-openai`, but Dynamo's local chat response types and completion
response wrapper still need explicit `skip_serializing_if = "Option::is_none"`
on optional response fields. This avoids streaming chunks like
`function_call: null`, `tool_calls: null`, `refusal: null`, `usage: null`,
`service_tier: null`, and `system_fingerprint: null` while leaving request-side
serialization behaviour unchanged.

Validation:

Run HTTP service tests for OpenAI chat/completions, Anthropic streaming,
tool-calling, reasoning/thinking fields, logprobs, dynamic temperature,
Baseten extension validation, `stream_options`, and unknown-parameter handling.

## PATCH-007: Python Bindings and Service-Facing APIs

Status: `keep`

Source commits:

- `72c26adb8` feat: Python bindings, examples, and service pipeline

Purpose:

Expose Baseten runtime, router, KV cache, JSON pub/sub, HTTP service controls,
and decorator behavior through Python bindings and stubs. Adds router examples
and an OpenAI service pipeline example.

1.0 -> 1.1 behavior:

The v1.1 upgrade kept B10 router and service pipeline bindings in `9adbfdc64`.
It added B10 health/rate-limit Python wrappers, context binding additions,
shutdown decorator arguments, service pipeline examples, and shared-memory
monitor examples. It dropped Python selector support and
`PyWorkerSelectionResult` along with the old Python selector example. Commit
`9cfeeb5e2` later restored JSON publisher/subscriber bindings after they became
EventPlane-backed and safe to use without NATS.

For v1.2, restore the Python selector surface despite the v1.1 drop. The reason
is that the target router still has a clean selector trait boundary, and the
product requirement is to support `algo_selector="Python"` for arbitrary Python
selection logic. Keep it separate from `entrypoint.rs`: the binding entrypoint
only parses `algo_selector`, while the Python callback adapter lives in its own
module and implements `WorkerSelector<ModelRuntimeConfig>`. Preserve B10 router
bindings through `RouterConfig(..., algo_selector="B10")`, preserve JSON
pub/sub if planner or routing code still depends on it, but keep Python
JetStream object storage dropped because Baseten no longer performs JetStream
offloading. Keep first-token webserver behavior as a design requirement, but
port through the target frontend/backend APIs rather than copying old binding
code.

Replay notes:

Replay after the Rust APIs from PATCH-002, PATCH-003, PATCH-004, and PATCH-006
exist on the target branch. Regenerate or manually update `_core.pyi`,
`Cargo.lock`, and examples after API conflicts are resolved.

Editorial notes:

The Python side requires holding the stream until the first token for the
webserver. The current implementation does this for webserver Python engines
only. This design likely makes sense in future Dynamo versions too.

Target assessment:

Keep this behavior as a design requirement, but port it through the target
frontend/backend API shape instead of copying the old bindings mechanically.
The target has rewritten frontend processors, so the implementation point may
move.

v1.2 implementation note:

Followed v1.1 for the HTTP Python-engine first-yield gate. `HttpAsyncEngine`
now enables `PythonAsyncEngine.block_until_stream_item(true)` by default, so
HTTP service registration waits for the Python generator's first item before
returning the Rust response stream. This preserves the webserver/disaggregated
decode invariant without restoring the v1.1-dropped Python worker selector
abstraction. B10 routing remains exposed through `RouterConfig(...,
algo_selector="B10")`, and JSON publisher/subscriber bindings were already
restored in the earlier PATCH-003/PATCH-007 overlap slice.

Validation:

Compile Python bindings, import `dynamo.runtime`, validate type stubs, and run
or smoke-test the router and OpenAI service examples.

## PATCH-008: Model, Parser, vLLM, and Multimodal Compatibility

Status: `mixed`

Source commits:

- `a7da3a65d` chore: bump vLLM to 0.19.0 for Dynamo fork compatibility (#194)
- `650305287` feat(vllm): bump fork to 0.20.0 for DeepSeek V4 + KV-router group_idx filter (#214)
- `860351213` chore(frontend): Add Gemma 4 parser support + Test Cases (#8852) (#222)
- `40a9b2b63` Merge pull request #235 from basetenlabs/dyo/gemma4-default-thinking-off
- `f098125d3` fix(preprocessor): default Gemma 4 reasoning OFF when chat_template_args omits flag
- `04c67b9b2` Merge pull request #164 from basetenlabs/fix/remove-missing-lfs-video
- `6e22e7a30` fix(llm): remove missing LFS video fixture

Purpose:

Keep model support aligned with Baseten requirements. This includes vLLM fork
compatibility, DeepSeek V4 support, KV-router group index filtering, Gemma 4
tool-calling and reasoning parsers, Gemma 4 default reasoning behavior, and
test fixture cleanup for missing LFS media.

1.0 -> 1.1 behavior:

The v1.1 upgrade mostly accepted upstream parser, model, and recipe movement
instead of replaying old version bumps as Baseten patches. Gemma/parser and
tool-call related work appeared primarily in upstream commits before the
Baseten replay stack, not as a separate Baseten port. The v1.1 CI/container
commit also carried a narrow missing-LFS-media cleanup where needed.

For v1.2, follow the same rule: do not replay dependency or parser version
bumps blindly. Carry only parser/model behavior that target tests or the
Baseten support matrix prove is missing.

Replay notes:

Do not replay version numbers blindly. Choose the target vLLM version from the
new Dynamo branch and Baseten support matrix, then port only remaining
compatibility behavior. Check whether upstream already has Gemma 4 parser
support and whether the missing LFS fixture is still referenced.

v1.2 implementation note:

The target already includes broad Gemma 4 parser and chat-template support, so
the old parser-support commits are not replayed wholesale. Keep the narrow
Baseten v1.1 decision from `f098125d3`: Gemma 4 reasoning parsing is disabled
unless `chat_template_args.enable_thinking` is explicitly true. This matches the
Gemma 4 template behavior, where omitting the flag does not emit reasoning
channel markers, and prevents the parser from running in a mode that can only
fall through. The hyphen alias `gemma-4` receives the same treatment.

Validation:

Run parser tests, Gemma 4 chat template tests, vLLM integration smoke tests, and
any multimodal tests that previously depended on the removed fixture.

## PATCH-009: Validation, Limits, and Operational Compatibility Knobs

Status: `mixed`

Source commits:

- `52a13feeb` Merge pull request #197 from basetenlabs/mf/warning-disable-option
- `43ad6e23a` disable warning flag
- `b6817ba7d` Merge pull request #205 from basetenlabs/mf/add-256mb-limit-tcp
- `5ddff62f4` add 256 mb limit
- `9ea89daa1` add 256 mb limit

Purpose:

Preserve operational knobs that keep Baseten deployments compatible with real
traffic: warning suppression or disablement behavior and a 256 MB TCP payload
limit for large payloads.

1.0 -> 1.1 behavior:

The v1.1 upgrade kept the 256 MiB TCP framing default inside `14e94a0f9` even
though upstream exposed shared max-message-size plumbing. It also kept Baseten
extension validation behavior inside `6f36f18d4`: fork-maintained extension
fields should validate explicitly instead of being silently ignored.

For v1.2, keep both decisions where the related features remain. Change the
target TCP default from 32 MiB to 256 MiB unless all deployments are guaranteed
to set `DYN_TCP_MAX_MESSAGE_SIZE`. Preserve required-field validation for any
`baseten_ext` fields that are carried forward.

Editorial notes:

The TCP limit work was contributed upstream, but the target SHA still needs an
explicit decision on the default size used by Baseten deployments.

Target assessment:

The target SHA has configurable TCP max message size via
`DYN_TCP_MAX_MESSAGE_SIZE`, but its default is `32 * 1024 * 1024`, while
`main-v1.0.0` carries `256 * 1024 * 1024`. If Baseten needs 256 MiB by default,
carry a small default-value patch unless deployment config is guaranteed to set
the environment variable everywhere.

Editorial notes:

Prefer a tiny code fix that defaults to 256 MiB.

v1.2 implementation note:

The TCP default was already restored as a tiny 256 MiB default-value patch. The
B10 config-map warning suppression knob from v1.1 was also restored:
`B10_CONFIGMAP_DISABLE_WARNING=1` or `true` now suppresses warnings for absent,
unreadable, unparsable, or failed hot-reload B10 router config maps while
leaving the default-config fallback behavior unchanged.


Replay notes:

Inspect the new upstream configuration surfaces before copying these changes.
If upstream exposes cleaner options for warning behavior or transport limits,
map Baseten defaults onto those instead.

Validation:

Test payloads near and above the expected TCP limit, warning-flag behavior, and
failure modes for oversized payloads.

## PATCH-010: Logging Policy and Production Signal Cleanup

Status: `keep`

Source commits:

- `f05fff4ca` Merge pull request #191 from basetenlabs/blarson/dyn1_error_log_debug
- `f2284d808` debug(logging): add request_id to shutdown noise and downgrade log levels
- `a12782956` fix(context): suppress duplicate stop/kill logs for child context propagation
- `7f66d2b5e` fix(context): suppress duplicate stop/kill logs for child context propagation
- `fa42386ca` fix(etcd): downgrade watch task exit logs from error/warn to info
- `0bdbc29b3` fix(logging): downgrade channel-closed log to info (not debug)
- `6c0a14499` fix(process_stream): exit immediately on context stop, add lifecycle logs
- `87597be69` restore warn
- `b51d61315` Merge pull request #195 from basetenlabs/blarson/log_improvements
- `8a9312239` BIS logging improvements: dynamo
- `86ec76b88` Add unified_model_logs to graceful shutdown lifecycle logs
- `8b72cb8aa` argh

Purpose:

Keep production logs useful under high-volume serving. This reduces duplicate
or misleading cancellation, context, etcd, channel, stream, queue, and shutdown
logs while adding request IDs and Baseten unified model log fields where they
help operations.

1.0 -> 1.1 behavior:

The v1.1 upgrade kept logging changes only where they had operational value.
`14e94a0f9` carried selected runtime log cleanup and instance-down signal
changes, `ce5dfa1c9` made OTel log export opt-in and trace endpoint fallback
spec-compliant, and `6f36f18d4` preserved Baseten structured model/request log
fields. It did not preserve every historical log-level tweak.

For v1.2, follow v1.1: keep structured request/model fields and OTel behavior,
then port only log-level changes tied to known false alarms or missing
debugging signal.

Editorial notes:

Keep structured logging.

Target assessment:

Keep structured request correlation and Baseten log fields. The target already
has request-id propagation and frontend tracing tests, so port only missing
structured fields and avoid replaying old log-level churn without evidence.

v1.2 implementation note:

Restored the v1.1 `unified_model_logs = true` marker for the current
`Runtime::shutdown()` lifecycle points: shutdown initiation and the final phase
where backend service connections are disconnected. Signal-handler shutdown
logs already carried the marker on the target branch, and the old
`Runtime::initiate_shutdown()` hunk no longer applies because that API is not
present in the target. The broader context reason/log-level churn remains
unported unless a current false alarm or missing operational signal is proven.


Replay notes:

Replay selectively after functional runtime and router patches are in place.
Avoid preserving every historical log-level tweak if upstream changed the log
site. Preserve the Baseten log schema fields and the intent: fewer false alarms,
better correlation, and enough lifecycle signal for debugging.

Validation:

Review logs during normal startup, graceful shutdown, cancellation, etcd watch
restart, client disconnect, and queue pressure.

## PATCH-011: Upstream Syncs and Likely Already-Upstream Patches

Status: `drop`

Source commits:

- `bfd4d0772` feat: sync upstream PR #143 and #146 - snapshot metrics + NATS stream config (#149)
- `40b1a7aed` feat: add kv_router.select_worker tracing event with routing metrics (#153)
- `402f5b47f` fix: silently ignore stream_options when stream is not true (#168)
- `603bd5e09` fix: remove double reasoning parse from Anthropic streaming handler (#193)
- `01841858f` feat(async-openai): OpenAI protocol compatibility fixes for agentic workloads (#221)
- `860351213` chore(frontend): Add Gemma 4 parser support + Test Cases (#8852) (#222)

Purpose:

Track changes that are especially likely to have landed upstream exactly or in
modified form. These commits are also listed under their functional patch groups
above because they affect replay behavior.

1.0 -> 1.1 behavior:

The v1.1 upgrade repeatedly used upstream equivalence as a reason to drop old
fork hunks. Examples include accepting upstream router queue structure, parser
changes, async-openai/protocol restructuring, and indexer/router metric work
where the target branch had already absorbed the behavior in a different shape.

For v1.2, keep this section as audit evidence only. Do not replay these commits
directly; use them to verify whether each functional patch has already been
satisfied by the target SHA.

Replay notes:

Before replaying onto a newer upstream tag, search upstream history for these
behaviors by commit title, PR number, changed file, and code behavior. If the
target upstream release already has the behavior, mark the corresponding
functional item as satisfied and do not replay the Baseten commit.

Useful commands:

```bash
TARGET_UPSTREAM=bed9f269312151481cd67a8d21b70e0f52424c2b
git cherry -v "$TARGET_UPSTREAM" main-v1.0.0
git log "$TARGET_UPSTREAM" --grep '<distinctive title words>'
git log "$TARGET_UPSTREAM" -- <path>
git blame "$TARGET_UPSTREAM" -- <path>
```

## PATCH-012: Reverted or Do-Not-Replay Experiments

Status: `reverted`

Source commits:

- `0fe453f52` Merge pull request #234 from basetenlabs/dyo/kvrouter-gemma4-accept-all-groups
- `67eee7334` fix(kv-router): accept all kv_cache_group_id events for hybrid-attention models
- `59b0132bd` Revert "fix(mp): accept all kv_cache_group_id events for hybrid-attention models" (#234)
- `df55350e3` Revert "Merge pull request #234 from basetenlabs/dyo/kvrouter-gemma4-accept-all-groups"

Purpose:

Record the attempted hybrid-attention KV cache group handling change and its
revert so it is not accidentally replayed from history.

1.0 -> 1.1 behavior:

The v1.1 branch also contains `465795594`, a WIP data snapshot that removed
profiler `.npz` artifacts before a dev-box restart. That commit is not a stable
fork-upgrade decision and should not be treated as part of the replay pattern.
The reverted hybrid-attention KV cache group experiment remained historical
context only.

For v1.2, drop these by default. Preserve profiler artifacts or redesign
hybrid-attention KV behavior only if there is an explicit current requirement.

Replay notes:

Do not replay as-is. If hybrid-attention KV cache group behavior is still
needed, redesign and validate it against the target upstream router and model
support code.
