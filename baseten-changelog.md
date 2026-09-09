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

Document guidelines:

- Prefer updating an existing patch section when a change belongs to an
  already-covered replay unit.
- Add a new patch section only for a substantial standalone PR, ideally around
  500 LOC or larger, or for a change that creates a distinct future rebase
  decision.
- If in doubt, keep the changelog entry local to the relevant existing section
  and reference these document guidelines instead of adding a new section.

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
- `7a6c0cb81` chore(container): bump bundled NATS to v2.14.4 and etcd to v3.7.1 (CVE remediation; server pins only, client crates unchanged)
- Current PR: build(mocker): enable KVBM offload in runtime wheel
- Current PR: build: upgrade NIXL stack to 1.4.0

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

The runtime image now builds NIXL 1.4.0 with UCX 1.22 and replaces TRT-LLM's
bundled NIXL and UCX libraries as one stack. Rust and Python bindings use the
same NIXL version across framework variants. KVBM workers explicitly create
the UCX backend before registering device memory, since POSIX does not support
NIXL 1.4 `VRAM_SEG` registrations.

A required `Baseten Changelog Check` GitHub Actions workflow now runs on every
PR targeting `main-v1.2.0` and fails if `baseten-changelog.md` is not modified.
PRs that legitimately do not need a changelog entry can bypass the check by
applying the `skip-changelog` label. This makes the patch ledger an enforced
artifact of the fork rather than a documentation convention.

A required `Upstream Frontend Crates Check` GitHub Actions workflow now runs on
PRs targeting `main-v1.2.0`. If the PR changes frontend, OpenAI protocol,
tool-call parsing, reasoning parsing, renderer, or related parity-test files,
the PR description must link an upstream `ai-dynamo/frontend-crates` issue/PR,
link an upstream `ai-dynamo/dynamo` issue/PR, or explicitly explain why the
change is Baseten-only and cannot be useful to any other Dynamo user. The check
also instructs authors to make upstream issues implementation-ready with a
minimal sanitized JSON payload. Dynamo agents working from proprietary or
customer-derived data are strongly encouraged to open an upstream issue with a
slightly anonymized but still reproducible case that preserves the relevant
format, fields, parser markers, and failure shape while replacing real content
with dummy equivalents. The guidance also warns authors to avoid customer
prompts, model outputs, tenant/model IDs, API keys, request IDs, URLs, headers,
logs, and other customer or internal data.

The post-merge `framework=none` image workflow now builds native `amd64` and
`arm64` images on Depot runners, pushes arch-specific tags, and publishes the
unsuffixed tag as a multi-arch manifest. The workflow does not publish a mutable
`latest` tag.

The standard `ai-dynamo-runtime` wheel now enables the existing
`mocker-kvbm-offload` Cargo feature in both media-FFmpeg and non-media builds.
This keeps G2 host-offload simulation available to downstream mocker images
without rebuilding native bindings in each application image. The feature only
changes behavior when mocker G2 capacity and bandwidth arguments are supplied.

The dispatchable BIS Dynamo Image Push workflow now passes `--no-tag-latest`,
so registry-dispatched builds from arbitrary refs no longer move the shared
`latest-none` tag; only the post-merge build path publishes moving tags.
`container/build.sh` additionally stamps `org.opencontainers.image.revision`
and `co.baseten.dynamo-sha` OCI labels so image→SHA provenance is readable
from the registry config blob without pulling the image.

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

Frontend client disconnects stop the request context by default instead of
killing it. Baseten can opt back into the hard-kill behavior by setting
`DYN_CLIENT_DISCONNECT_BEHAVIOR=kill`. Stopping lets
engines continue their cancellation path long enough to emit billing information
for cancelled streams and run cleanup/GC for in-flight prefill-decode requests,
including cases where decode invalidates an RDMA transfer. Unknown values panic
rather than silently selecting a behavior.

Runtime shutdown Phase 2 now has a Baseten-owned safety cap around the
graceful endpoint drain wait. If `tracker.wait_for_completion()` does not
finish, the runtime logs the remaining graceful endpoint count and proceeds to
Phase 3 teardown instead of hanging forever behind a deadlocked in-flight
request. The cap is controlled by `DYN_RUNTIME_GRACEFUL_SHUTDOWN_TIMEOUT_SECS`;
Baseten uses a 4 minute default (`240` seconds), not the 15 minute value discussed for the upstream proposal in dyn1.3+.

The primary etcd lease TTL is raised from 10 seconds to 30 seconds on v1.2.
This gives the lease keep-alive loop more tolerance for transient runtime,
scheduler, or network stalls before the process lease expires and the runtime
is cancelled. The keep-alive cadence still derives from the TTL returned by
etcd, so this only changes the requested lease grant duration.

Mirrored upstream PR `ai-dynamo/dynamo#11146` for v1.2 etcd watch recovery.
After an etcd watch reconnects, the watcher emits an authoritative full-prefix
`Resync` snapshot, and stateful consumers (`KvCache`, `TypedPrefixWatcher`,
storage-backed discovery) rebuild or diff local state from that snapshot. This
prevents stale lease-bound discovery entries from surviving missed delete
events during etcd reconnects.

The frontend HTTP service now registers with the graceful-shutdown tracker for
the lifetime of its serve+drain future (`HttpService::run` holds a
`GracefulTaskGuard`, basetenlabs/dynamo#461). Before this, only worker
orchestrators registered, so on a frontend Phase 2 was empty and Phase 3
cancelled the primary token milliseconds after SIGTERM — tearing down the
KV-store discovery watch while axum was still draining multi-minute streams.
The dying frontend then saw an empty instance list, treated a healthy router as
unreachable (`Instance not found and no other instances available`), and failed
in-flight requests with HTTP 529 `model_unavailable`; the router-side request
guard's `mark_free` failed on the same path, leaking queued-ISL admission until
the router's 300 s stale-sequence expiry. Observed on every frontend HPA
scale-down under long-context traffic (fde wdld27k, 2026-07-17). With the
guard, Phase 2 holds until in-flight HTTP requests finish, bounded by
`DYN_RUNTIME_GRACEFUL_SHUTDOWN_TIMEOUT_SECS` — deployments must size that
variable (and `terminationGracePeriodSeconds`) above the longest expected
stream; the Baseten model-values 90 s pin is too low for long-context serving.
Validated by killing both frontend pods with four in-flight ~600K-token
requests: all drained to HTTP 200, shutdown held 83 s/161 s (previously 0.6 ms
and 529s). Known follow-ups, not yet ported: the discovery `endpoint_watcher`
still publishes an empty instance list on local cancellation, and the router
guard leak deserves a liveness lease instead of relying on the 300 s expiry.

Replay notes:

Port behavior, not necessarily implementation. Upstream may have refactored
runtime ownership, endpoint lifecycle, or discovery. The important invariant is:
stop advertising before shutdown, allow in-flight work to drain, keep required
runtime dependencies alive during drain, and expose enough lifecycle signal to
debug shutdowns.

Validation:

Exercise startup health, request drain, endpoint unpublication, NATS/ETCD
lifetime during drain, and forced shutdown timeout behavior.

Backported upstream PR `ai-dynamo/dynamo#10437` (`73903bdc807323c0b14dbb4ddb7c79b7da72d3d8`, `perf(runtime): add request-plane msgpack payload codec`) for v1.2. New request-plane sends default to msgpack when
`DYN_REQUEST_PLANE_CODEC` is unset. Incoming control messages that omit the
`payload_codec` field are still decoded as JSON for compatibility with older
clients; set `DYN_REQUEST_PLANE_CODEC=json` to force JSON sends.

The Dynamo fork is adjusted to send Python `bytes` natively through the
request pipeline using msgpack as the default codec, instead of
base64-encoding binary payloads. This mirrors upstream
`ai-dynamo/dynamo#12015`, which swaps the Python `Client` request-plane
intermediate and the server ingress from `serde_json::Value` to
`rmpv::Value` so that msgpack's native `Binary` type round-trips as
`PyBytes` end-to-end. Both the fork and upstream are considered aligned
when the fork contains some version of `ai-dynamo/dynamo#12015` and a
Python `bytes` field survives the full client→router→worker→client
round-trip as `bytes` (not base64, not an int array).

Added `DYN_ENABLE_FAULT_INJECTION` to opt into PushRouter request-path
fault-injection handling. It defaults off so transient transport/backend
failures do not quarantine remotes unless explicitly enabled; Baseten avoids
calling `report_instance_down` by default.

Adapted upstream `ai-dynamo/dynamo#14159` (`57aac94525dcf8d3085ef6d7d7b25529c5587ab5`):
direct dispatch with fault detection disabled checks the borrowed live discovery
table without cloning instances and collecting IDs. The fault-enabled routing
snapshot path is unchanged; do not substitute a reconciled snapshot for live
discovery when replaying this change.

The worker ingress's terminal `complete_final` publish failure is classified
like the mid-stream data-path failure: when the request context is already
stopped/killed (client disconnect, or an upstream early break such as the
stop-word tool-call cutoff under `parallel_tool_calls=false`), the failed
final send logs at DEBUG instead of ERROR — the peer dropped the receiver and
`handle_writer` exits without draining, so the final frame is undeliverable
by construction and the receiver already treats "stream closed while stopped"
as a clean end. Genuine transport failures (context still live) keep the
ERROR. The `PUBLISH_FINAL` error counter stays unconditional for dashboard
continuity. Observed at ~1/min on tool-call-heavy Kimi serving as pure log
noise; likely upstreamable.

The TCP request-plane writer now discards frames whose caller already abandoned
the request and dropped its response receiver, instead of delivering them
minutes later to a caller that no longer exists (basetenlabs/dynamo#763). Late
delivery made the KV router book scheduler state that nothing freed until the
600 s stale-request reaper.

The TCP request sender ports upstream ai-dynamo/dynamo#10519 (commit
`867f530414d380599c0b9a317d39d848e2364094`): split header/payload frames,
small-chunk coalescing, and byte-bounded vectored writes remove large payload
copies without changing the wire protocol. Preserve the abandoned-request
check from #763 when replaying this port. Message-size limits are unchanged;
this replaces the need for the retained send-scratch fix proposed in #761.

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
- `a75d34383` feat(kv-router): floor router temperature and make B10 log throttle configurable
- `ebf5d4d47` b10: bump router stale-request expiry (orphan timeout) from 5 to 10 minutes (#550)

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

This branch implements only the session-affinity subset needed by the standalone
Baseten B10 router: header-based affinity through `X-Dynamo-Session-ID`,
synchronization between router replicas, and soft affinity scoring in B10.
Generic push and prefill routers are unchanged. The existing body-level
`nvext.session_control` implementation is not part of this port and remains
unchanged. The subset is ported where the upstream code applies cleanly and
reimplemented where the v1.2 architecture differs, based on these three PRs:
[header-based session affinity #10875](https://github.com/ai-dynamo/dynamo/pull/10875),
[session-affinity replica synchronization #11750](https://github.com/ai-dynamo/dynamo/pull/11750),
and [soft session-affinity preference #12804](https://github.com/ai-dynamo/dynamo/pull/12804).
The resulting interface is opt-in for standalone Python router consumers through
`start_router(router_config=RouterConfig(..., session_affinity_ttl_secs=...))`;
omitting the TTL disables affinity. The TTL remains owned by the outer
`RouterConfig`, matching upstream.
When enabled, the exact affinity `(worker_id, dp_rank)` receives a `0.5` B10
score multiplier while remaining subject to normal eligibility and overload
checks.
There is no hard/soft mode flag in this backport: setting the TTL enables soft
header-based affinity, while omitting it disables the feature. A later Dynamo
version may provide a superset of this interface without changing that behavior.

Followed v1.1 for `router_queue_threshold` hot reload, adapted to the target's
new actor-based router queue instead of copying the old `RwLock` queue shape.
The B10 config map now accepts root and override-group `router_queue_threshold`
values; the scheduler polls the hot-reloadable B10 config every 10 seconds and
updates the queue actor without restart. Positive values enable queueing at the
new threshold, while `0` or `None` disables queueing. When queueing is disabled
after requests are already pending, the actor drains them immediately so the
target queue cannot strand requests behind a now-disabled threshold.

The router queue also has an internal `DYN_ROUTER_QUEUE_BUSY_FRACTIONAL`
admission knob for large replica counts. By default, queue admission preserves
the exact all-eligible-workers-busy behavior. When the env var is enabled, the
queue starts once busy eligible workers reach `floor(0.99 * N)` above 16
eligible workers, `floor(0.98 * N)` above 64, and `floor(0.97 * N)` above 200.
This lets large deployments begin queueing before a literal P100 worker-busy
condition, while keeping pinned-worker and small-deployment behavior exact.

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

Worker recovery snapshots include device, host-pinned, and disk cache state.
Failed tier dumps return errors instead of caching incomplete snapshots.
Lower-tier removal batches apply valid removals even when other hashes are
missing or duplicated. Preserve these invariants when rebasing (#758).

Worker-query recovery selectively ports two behaviors from upstream
[ai-dynamo/dynamo#13053](https://github.com/ai-dynamo/dynamo/pull/13053): explicit
`Error` responses use the existing bounded exponential-backoff retries, preserving
indexed state and the cursor on exhaustion; regression coverage verifies that
ahead-of-watermark snapshots converge with duplicate live-tail replay and removals
across device, host-pinned, and disk tiers. No residency-domain or wire changes
are required for these ports.

The router bookkeeping protocol now accepts a `request_id` payload override for
`MarkPrefill`, matching the existing `MarkFree` behavior when the transport
context id cannot be used. `PotentialLoads` responses also report the current
router queue backlog through `pending_count` and `pending_isl_tokens`, so callers
can inspect both worker load projections and queued-work pressure in one request.
Each `PotentialLoad` row also carries the worker's `active_requests` count,
matching the v1.0 bookkeeping surface used by autoscaling consumers.
`PotentialLoads` requests now also accept `allow_short_caching`, an opt-in
500 ms router-side cache for repeated probes with fewer than 32 input tokens.
Callers that set this flag avoid repeatedly recomputing the same load snapshot
for short capacity probes. The cache is implemented in
`lib/llm/src/kv_router/b10_potential_loads_cache.rs`.

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

The discounting is scoped to the placement projection only. The projection
threads an `apply_discounts` flag (restoring the v1.0-era carve-out from
`2d78ce32b` that the scheduling-framework rewrite dropped): the queue admit
path passes `true`, while the `PotentialLoads` RPC and the Python
`get_potential_loads` binding pass `false` and report raw token/block counts.
This matters because the planner's autoscaling signal reads that RPC; with the
discount applied it was understated by the configured factor (5x at the shipped
`0.2`). One deliberate exception: GWP local-load anchors are captured with
`apply_discounts=true` because `fuse_load` subtracts them from discounted
placement-path locals, so both sides must be measured the same way. Relatedly,
`HotReloadableConfig::new` publishes the initial config's discounts and queue
threshold to the scheduler atomics even when the config file is missing or
invalid; previously the atomics stayed at their `1.0` static init on the
default-config path while `get_config()` reported the serde defaults.

Heuristic and selector parity note:

- `softmax_sample` accepts any worker-logit map that can be iterated as
  `(&WorkerWithDpRank, &f64)`, instead of requiring `FxHashMap`.
- The B10 worker selector can use the standard `HashMap` for its local logits
  map, so `dynamo-llm` does not need to depend on `rustc-hash` for this path.
- The zero-temperature selection path stays allocation-free; only the softmax
  sampling path collects entries.
- `router_temperature` is floored to `1e-12` when set to `0`, negative, or
  non-finite (sanitized at config load, env default, and the per-request
  override site). A temperature of `0` previously broke exact logit ties by
  the lowest `worker_id` (u64), which is endpoint-correlated and caused a
  self-reinforcing traffic imbalance in multi-endpoint deployments. The floor
  keeps selection effectively deterministic on the min logit while routing
  ties through `softmax_sample` (uniform random). The dead `worker_id`
  tiebreak branch was removed from `B10WorkerSelector`.
- The per-worker scoring log throttle interval is configurable via
  `B10_KV_ROUTER_SELECTION_LOG_INTERVAL_MS` (read once via `OnceLock`, default
  `2000ms`), replacing the hardcoded `2000ms` for pools larger than 5 workers.

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

The `KvEventPublisher.local_indexer_endpoint` exposure is only a local replay
of upstream ai-dynamo/dynamo#11498, which gives Python shutdown hooks access to
worker-local KV query endpoints so they can deregister before runtime teardown.
If the target base already contains that upstream PR, register the exposed
worker-local KV query endpoints with `phase="early"` instead of carrying a
separate endpoint-exposure patch.

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
- Keep this scheduler PR in the v1.2 replay to preserve ISL tracking at both
  scopes: per worker and per DP rank. The scheduler now records running mean
  and variance for each worker, and for each worker/rank pair when DP is
  active, so rank-local versus worker-global ISL pressure remains observable.

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
- `4d01df101` redefines the unused `b10-fair-wspt` queue policy as a static
  FCFS policy with bounded missing-prefill credit instead of a dynamic aging
  WSPT policy. The enqueue score is `priority_jump - arrival_offset + credit`,
  where credit is `15s` at `0` missing prefill tokens, linearly fades to `0s`
  at `8192` missing prefill tokens, and remains `0s` beyond that. This keeps
  the queue non-dynamic, preserves FCFS pressure behavior, and still promotes
  requests that should have better TTFT because little prefill work remains.
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

Opt-in router residency tracking:

- Adds per-worker `WorkerResidency` LRU (`SequenceHash -> Instant`) gated by
  `router_track_residency`. Off by default.
- Touch is taken before `slot.sequences.write()` and `configure_residency`
  trims per-worker outside the outer `workers.write()` lock.
- Exposes `eviction_pressure_for_new_blocks_at` for selector use.
- Capacity hard-capped at 200k blocks per worker (CPU-router memory budget,
  not a device KV-cache mirror).

Residency eviction cost wired into `B10WorkerSelector` (#348):

- Adds hot-reloadable `router_residency_eviction_cost` (weight, default 0 = off)
  and `router_residency_half_life` (recency decay, default 120s), with env
  overrides and finite/non-negative sanitization.
- `admit_one` computes per-worker eviction cost only when residency tracking is
  on **and** the weight is `> 0`; the result is added to the `B10WorkerSelector`
  logit (`+ rec` term), penalizing workers that would evict recently-used
  blocks. Default behavior is unchanged.
- Cost path acquires the multi-worker residency read lock once per eligible
  worker per request; gated off by default. A batched single-lock query is a
  follow-up if the weight is enabled at scale.

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

Upstream sync note:

- ai-dynamo/dynamo#10887 (fix: cancel `RouterRequest::New` while waiting for
  KV router scheduler admission, closes ai-dynamo/dynamo#10878) has been
  merged upstream and is folded into this branch. The replay races scheduler
  selection against `ctx.context().stopped()` / `killed()` and calls
  `self.free(&context_id)` to release scheduler state before returning a
  `Cancelled` error.
- Queued-request cancellation needs both ai-dynamo/dynamo#10887 (above) and
  ai-dynamo/dynamo#10331 (the `response_is_closed()` booking guard in
  `book_and_respond`, which the fork base predates). Preserve the focused
  scheduler queue regressions for cancelled pending requests and response
  delivery rollback.

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
- `7f7bdd3b4` fix(mocker): publish planner load metrics
- `290dc296a` refactor(mocker): scope planner metrics to engine args

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

The mocker runtime publishes ActiveLoad and forward-pass metrics under the
externally visible worker component, matching the vLLM and TRT-LLM worker
identity used by routers and planners. Its scheduler snapshots also expose
running and waiting requests, KV block usage, and per-iteration token counts to
the Python metrics callback so Baseten's planner receives the same detailed
worker-load schema from mocker deployments as it does from real engines.

Worker-based KV recovery now has a lightweight `RecoveryProcessLogger` that
reports aggregate restore progress, recovered event counts, and the final
initial-recovery completion summary used by router startup gating.

`best_overlap_blocks` was restored to `RouterResponse::New` (originally added
by `4f8c728ef` / PR #180 on the v1.0 branch; dropped in the v1.2 rebase). It is
the maximum device-tier overlap across all candidate workers, computed in
`find_best_match_details` as a max over the per-request
`tier_overlap_blocks.device` map the routing path already builds — no extra
indexer queries or allocation. The field is `#[serde(default)]`, so mixed
router/frontend versions stay wire-compatible, and it also carries
`FindBestMatchOutcome::Routed::best_overlap_blocks`. The b10_client
`AdmittedRequest` exposes it to Python as `b10_best_overlap_blocks()`. Consumed
by the frontend's `llm_kv_cache_best_prefix_hit_rate` and
`llm_kv_cache_hit_rate_efficiency` metrics; dropping it during a rebase kills
those dashboards.

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
- `de9d4ba04` fix(frontend): correct Anthropic /v1/messages input_tokens accounting
  (port of upstream #11030 `2a29f6d65f`; Baseten deviation: always emit
  `cache_creation_input_tokens: 0`)
- `08000db` fix(protocols): accept image_url detail=original (#693; original is
  in OpenAI's current vision API and Baseten's Model APIs docs; no backend
  consumes detail today, so all values are no-ops; upstreamable)

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
completions, and embeddings, and later extended to Responses and Anthropic
Messages/count_tokens. `service_tier: flex` gets a B10-specific `+1.0`
rate-limit surcharge with a kill switch; all other tiers remain on the
default/non-flex path. The target already has selective endpoint activation
through `HttpService.enable_endpoint(...)`, with Python tests using it to turn
chat on explicitly; do not replay old endpoint activation code unless new
endpoint-specific tests fail. Broader protocol work remains open only for
response-shape or tolerance decisions not already covered by `baseten_ext` and
Anthropic conformance slices.

The Python HTTP frontend now returns B10 route attribution in four
`x-baseten-dyn-*` OpenAI and Anthropic response headers. The selected worker ID
and DP rank are recorded in request-scoped `Context.metadata` according to the
`RouterWorkerPhase` enum (or its backward-compatible exact string values);
unknown phases fail at the binding boundary.

The v1.2 health replay also ties readiness to runtime lifetime. The B10 router
registers the `DistributedRuntime` primary cancellation token with B10 health,
and `/health_file` returns unhealthy once that token is cancelled even if the
last heartbeat is still fresh. The router keeps the 60s initial heartbeat grace
period before publishing health, so the endpoint remains unhealthy during
startup unless another caller explicitly sets health.

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

A later v1.2 slice (#308, port of #307) added the per-request `thinking_token_budget` field to `BasetenExt`, flattened at the request root like `reasoning` and `dynamic_temperature`. Without it the frontend's `warn_unsupported_fields` validator stripped the field before it reached the worker, so clients could not dynamically bound runaway reasoning (e.g. Qwen3.5/3.6, GLM-5.2). The worker maps it onto `vllm.SamplingParams.thinking_token_budget`, which forces the reasoning-end token (`</think>`) once the `<think>` block reaches the cap. Omitted -> `None`, so a worker/BIS-config default still applies.

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

Current v1.2 OpenAI compatibility gap: the protocol layer now accepts
multimodal tool-message content so real OpenAI-compatible clients are not
rejected at JSON deserialization time, but this is only the ingress piece.
The corresponding upstream protocol support was merged in
[`ai-dynamo/frontend-crates#143`](https://github.com/ai-dynamo/frontend-crates/pull/143)
and released in `dynamo-protocols` 5.0.0.
Follow-up work should implement the two remaining pieces explicitly: preserve
and validate the relevant `tool_choice` behavior through the Dynamo/OpenAI
pipeline, and add processor/backend handling for image content carried in tool
messages rather than merely preserving it in the typed request.

The v1.2 HTTP compatibility work also keeps rejected 400 requests observable:
OpenAI and Anthropic JSON-deserialization failures are converted from Axum 422
responses into API-compatible 400 responses in middleware, and the middleware
can emit structured `unified_logs` entries with the rejection reason and serde
message before returning the 400.

Restored the v1.0 logprobs `token_id` response field on v1.2.0 via #302:
`ChatCompletionTokenLogprob` and `ChatChoiceLogprobs` are now defined locally in
`lib/protocols/src/types/chat.rs` (shadowing the `async-openai` re-exports) with
an added `token_id: Option<u32>` that serializes only when present, and
`DeltaGenerator::create_logprobs` populates it from the backend `token_ids`
already threaded through `lib/llm/src/protocols/common.rs` and the chat
aggregator/jail/HTTP paths. This re-adds the Baseten logprobs `token_id` surface
from commit `fce8920c3` that was dropped on the v1.1 follow-default in this section.
Drop once upstream `async-openai` exposes an equivalent `token_id` field on
`ChatCompletionTokenLogprob`.

The response-only null-omission patch should be preserved on v1.2. The target
already omits absent `Choice.logprobs` and `Choice.finish_reason` through
upstream `async-openai`, but Dynamo's local chat response types and completion
response wrapper still need explicit `skip_serializing_if = "Option::is_none"`
on optional response fields. This avoids streaming chunks like
`function_call: null`, `tool_calls: null`, `refusal: null`, `usage: null`,
`service_tier: null`, and `system_fingerprint: null` while leaving request-side
serialization behaviour unchanged.

Wire-type groundwork for the shared normalization crate (BLS, PR stack D1):
`async-openai` 0.34 -> 0.41.3 (tagged `reasoning_text` content parts — a
parse-strictness increase, untagged parts now 400; `ReasoningItem.id:
Option`, lib/llm wraps its ids in `Some`); `AnthropicCreateMessageRequest`
gains an ordered `unmodeled` passthrough and `AnthropicTool` gains
`defer_loading`; `AnthropicMessageContent` gets a hand-written `Deserialize`
so malformed `content` 400s name the actual problem; `SystemContent`
documents itself as the lossy typed view it is and serializes back in
Anthropic's own wire shapes; Responses input accepts codex `agent_message`
and `additional_tools` items — at this slice the lib/llm converter SKIPS both
(previously the request 400d as an unknown variant), the shared crate later
declares `additional_tools` as tools; `ChatCompletionResponseMessage.
reasoning_content` is omitted when absent (OpenAI defines no such field);
`serde_json` gains `preserve_order` so passthrough fields re-serialize in
client order. Already on main-v1.2.0 and NOT changed here: non-terminal
`finish_reason` omission, string `error.code`, `document`/`search_result`
tool_result blocks. Baseten-specific; not upstreamable.

The `reasoning_effort` field on chat completion requests is normalized through a process-wide alias map before deserialization. Defaults: `"max"` → `"xhigh"`. Override at runtime by setting the `REASONING_EFFORT_ALIASES` env var to a JSON object (e.g. `REASONING_EFFORT_ALIASES='{"max":"xhigh","minimum":"low"}'`); parsed once on first use, silently falls back to the hardcoded defaults if absent or unparseable.

The same alias map applies to `reasoning.effort` on `/v1/responses` requests (`CreateResponse.reasoning` custom deserializer): without it, `{"reasoning": {"effort": "max"}}` was a deserialization 400 on the Responses API while the identical effort succeeded on chat completions. Serve-side reasoning policies map `xhigh` back to the model-native `max` tier, so DeepSeek V4 / GLM clients get identical effort behavior on both APIs (PR #574).

The request-side assistant message accepts `reasoning` as a serde alias for `reasoning_content`, so prior-turn reasoning sent under either wire name (OpenRouter/newer-vLLM `reasoning` or DeepSeek/vLLM-legacy `reasoning_content`) deserializes into the canonical field and re-renders into the chat template.

`BasetenExt.mocker_config` per-request passthrough field: a free-form
`Option<HashMap<String, Value>>` for per-request overrides consumed only by
the CPU mocker backend (e.g. speedup ratios for replay timing); GPU engines
ignore it. Serde-optional (omitted when `None`) and counted by `is_empty()` so
a request carrying only this key isn't dropped by the flattened field's
`skip_serializing_if`. Wire format only; preprocessor forwarding and mocker
consumption land separately. Baseten-specific; not upstreamable.

Added `chat_template_args` (alias `chat_template_kwargs`) to `BasetenExt` so it is available on all request paths that flatten `BasetenExt` — including `/v1/chat/completions` and `/v1/responses`. The `TryFrom<NvCreateResponse>` conversion now forwards `baseten_ext.chat_template_args` instead of hard-coding `None`, so callers of `/v1/responses` can pass a custom chat-template context through to the worker.

Added optional `BasetenExt.allowed_worker_ids` filtering through `RoutingHints`
and `RouterRequest::New`, with the same `Set[int]` input exposed by the B10
Python client. Omission preserves existing routing behavior.

Owned `ChatCompletionRequestSystemMessage` (content optional, opaque `tools` passthrough) and added `partial: Option<bool>` on the owned assistant message so Moonshot Kimi K3 conformance traffic deserializes: K3 sends system messages carrying only a dynamic `tools` list (no `content`, which upstream rejects with 400 "missing field `content`") and assistant prefill turns marked `partial: true` (#496). Implemented in `lib/protocols/src/types/chat.rs` (owned struct + `partial` field) with `From` bridges in `impls.rs` and call-site updates in `lib/llm` (anthropic, responses) + tests; both fields forward opaquely to the worker. Drop once upstream `async-openai` relaxes `content` and accepts the `tools`/`partial` keys.

The Anthropic `/v1/messages` request conversion drops `tool_choice` when no declared tool survives tool conversion (server tools like `web_search` have no `input_schema` and are filtered), and degrades a named `tool_choice` pointing at a filtered tool to `auto` when function tools remain. Previously the inconsistent converted request hit the worker's `400 "When using tool_choice, tools must be set"` — Claude Code on Baseten backends triggered this whenever its WebSearch server tool was invoked, surfacing an API error mid-turn instead of a text answer.

The Responses API stream converter (`lib/llm/src/protocols/openai/responses/stream_converter.rs`) closes streamed function-call items on the choice's `finish_reason` (or stream end) instead of on the first argument delta that carries `id`+`name`, so `function_call_arguments.done` / `output_item.done` carry the fully concatenated arguments for backends that fragment arguments across chunks (GLM-5.2; stock Codex consumes only the done items) (#544).

`/v1/responses` maps `finish_reason=length` to `status:"incomplete"` with `incomplete_details.reason="max_output_tokens"` (and `completed_at:null`), marking the truncated output item incomplete, on both the non-streaming and streaming paths; the stream ends with a `response.incomplete` terminal event instead of `response.completed`. Streamed reasoning is surfaced via the `response.reasoning_summary_part/text.*` lifecycle, gated on the request setting `reasoning.summary` (kept private otherwise); a reasoning item stays `completed` when the model produced an answer/tool call and is marked `incomplete` only when truncation landed mid-reasoning. Backport of upstream ai-dynamo #12182 (incomplete-on-truncation) and #12183 (streamed reasoning), rebased onto the finish_reason-gated tool-call close above (stacked on #544).

Anthropic `tool_result.content` accepts the documented non-text/image block types instead of rejecting the request: `document` and `search_result` are typed variants (`DocumentBlock`/`SearchResultBlock` in `lib/protocols/src/types/anthropic.rs`) whose text payloads are flattened into the converted tool message, and blocks with no text representation (base64/url documents, `tool_reference` from Claude Code's tool-search, unknown future types) are skipped with a warning. Replaces the hard 400 introduced by #510 ("unsupported Anthropic tool_result content block"), which broke agent frameworks attaching citations/RAG results or tool references to tool output on `/v1/messages` (GLM 5.2 Model APIs customer regression, 2026-08-08). A malformed document/search_result degrades to a skipped block, never a request-level error (#559).

On the streamed `/v1/responses` path, a mid-stream backend `event: error` annotation no longer degrades to `response.failed` with `output: []` and `error: null`. The error message/status are extracted (same `extract_backend_error_if_present` logic as chat completions) and carried on `response.failed.error` (`server_error`, or `rate_limit_exceeded` on 429), with whatever output had streamed included and marked `incomplete`. A truncation-shaped error — the Baseten chat processor's legacy `"Tool calls cutoff by max_tokens."` raise — is instead presented as spec-correct truncation: the converter sets the output-limit state and emits the #547 incomplete sequence (`function_call_arguments.done` with partial args, `output_item.done` `status=incomplete`, terminal `response.incomplete` `reason=max_output_tokens`). Baseten-specific compatibility shim for workers that predate the monorepo fix removing that raise (#553).

Prompt-overflow backend errors ("Input length N exceeds the maximum allowed input length of M tokens." / OpenAI's "maximum context length" phrasing) carry `error.code = "context_length_exceeded"` on the streamed `response.failed` — the exact string OpenAI clients classify on (codex's only context-overflow detector is `error.code == "context_length_exceeded"`, an SSE-path string match; its HTTP-400 path is a non-retryable raw-text failure) (#553).

`/v1/responses` drops `tool_choice` when no declared tool survives conversion to chat completions. Codex hardcodes `tool_choice: "auto"` on every request and sends `tools: []` on auxiliary turns — its auto-compaction turn in particular — so forwarding the choice tripped the worker's `400 "When using tool_choice, tools must be set"` and killed long sessions at the exact moment they tried to free context (compaction retries every error 5× then fails the turn) (#553).

`{"type": "namespace"}` tool groups on `/v1/responses` requests (codex sends these for MCP/app tool groupings; emission is on by default via its `namespace_tools` provider capability) round-trip: `convert_tools` flattens member function tools into the chat-completions tool list under collision-proof `{namespace}__{name}` names (namespaces exist so member names can overlap between groups), and emitted `function_call` items map back to the wire (name, namespace) pair via exact-match lookup against the declarations (`resolve_tool_identity`) on the streaming (`output_item.added`/`.done`, `function_call_arguments.done`, terminal output) and non-streaming paths; replayed namespaced calls in input history render under the same mangled name so the model sees its history named consistently with the tool list. Codex dispatches on the exact (name, namespace) pair with no fallback, so a stripped namespace makes the call undispatchable client-side. Freeform/custom members of a namespace are not forwarded (no constrained-decoding contract across the chat bridge) (#553).

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

Also add Python `Context.detached(id)` for trace-preserving disaggregated handoff.
It creates a fresh cancellation controller while carrying metadata, trace context,
and the captured `engine.generate` span. This is needed because decode requests
detach cancellation ownership after prefill without splitting TRT-LLM/Honeycomb
backend spans from the original Dynamo request. Drop after upstream has this API.

Also expose `b10_health.register_runtime(runtime)` so Python-owned runtime
startup paths can attach the same runtime cancellation token that the Rust B10
router registers automatically.

OpenAI and Anthropic HTTP contexts stamp `dynamo.request_start` as Unix epoch
milliseconds. Python `Context.get_milliseconds_since_request_start()` exposes
the elapsed time.

v1.2 b10_client portability note:

The admission implementation now lives in the workspace crate
`lib/b10-client` (`dynamo-b10-client`). Its high-level
`RouterWorkerCoordinator` is usable directly from Rust; the Python extension
converts PyO3 inputs and adapts the returned worker stream while sharing the
same routing, preflight, reroute, cancellation, and guard-cleanup state machine.
The lifecycle tests moved with the implementation so the language-neutral
crate owns its behavioral contract.

The fork-carried `lib/bindings/python/rust/b10_client/` module exposes the
admission lifecycle used by `RouterConfig(..., algo_selector="B10")`. It is a
self-contained unit added by the fork; the upstream at the target SHA does not
ship an equivalent. The module deliberately keeps all router-facing policy,
guard arming, and post-admission failure handling in a single crate-local
surface so it can be lifted forward without touching the rest of the Python
bindings.

`route_request` (in `coordinator.rs`) is the function that owns the
post-admission fail-closed invariant. It is split into three explicit stages:
Stage 1 `router.direct()` failure dismisses the provisional guard before
admission (no `mark_free`); Stage 2 post-admission stream error drops the
provisional guard without dismissing, so the armed cleanup task fires
`mark_free` asynchronously; Stage 3 unexpected response variant (neither
`New` nor `Backpressure`) fails closed via the new
`RouteSource::ProtocolError { received }` variant, which `route_once` maps to
`DeniedRequest::ProtocolError { received }`. The added behavior corresponds to
fork commit `e06cc1308` (Race Site 12 + unexpected-variant regression); tests
`route_and_connect_post_admit_stream_eof_fires_mark_free_then_denies_unreachable`
and
`route_and_connect_unexpected_response_variant_fires_mark_free_then_denies_protocol_error`
guard the invariant.

v1.2 worker payload copy staging (PRs #511/#512/#513): the reroute loop no
longer deep-copies the worker payload per attempt. `#511` builds the routing
wire via msgpack instead of a `serde_json` round-trip, `#512` passes the
routing request as `Arc<rmpv::Value>`, and `#513` adds
`b10_client/payload_copy.rs`: each attempt's payload copy is a plain
`rmpv::Value::clone` on the blocking pool, staged before the routing RPC for
small payloads (hidden behind the round-trip) and after admission for large
ones (rejected large-multimodal requests do no copy work), resolved inside
`connect_worker`'s setup shield so `DetachSetupOnly` gains no new
cancellation point, handed back unresolved on a proactive stale for the
retry to reuse, and logged (`unhidden_ms` / `copy_ms` / `payload_mb`) when
it keeps the critical path waiting beyond 40ms. Replay as part of the 1:1
module copy; the eager cutoff lives in `EAGER_COPY_MAX_BYTES`.

v1.2 packed routing tokens (`DYN_ROUTER_SEND_PACKED_TOKENS`): the `tokens`
field of `RouterRequest::New` / `PotentialLoads` (kv-router `protocols.rs`,
`TokenBlob`) deserializes from either a packed little-endian u32 byte blob or
the legacy integer array — routers accept both, always. Clients send the
packed blob by default, replacing ~3ms of per-element msgpack encode/decode
per 100k-token route with a memcpy (wire size unchanged). JSON codecs and a
non-msgpack `DYN_REQUEST_PLANE_CODEC` always use the integer array
(`is_human_readable`); `DYN_ROUTER_SEND_PACKED_TOKENS` is an opt-out kill
switch, where any non-truthy value reverts the sender without a redeploy.

Guidance for future versions: prefer copying this module forward 1:1 from
the prior fork release (or from a future upstream equivalent, when one exists)
and avoid introducing bespoke Baseten-only changes inside `b10_client/`. The
goal is to keep the module maintainable 1:1 with future Dynamo releases so an
upgrade can be a near-verbatim copy plus test refresh, not a hand-port. If a
behavior change is unavoidable, prefer adding it at a trait boundary the
upstream client already exposes (e.g. the `Router` trait, `GuardMark`, or the
`route_once` -> `DeniedRequest` mapping) rather than inside the
admission/cleanup state machine itself.

v1.2 NumPy token-id inputs:

The token-id arguments on the routing entry points (`PyRouterRequestNew`'s
constructor and `tokens` setter, `KvRouter.generate` / `best_worker` /
`get_potential_loads` / `get_overlap_scores`, and the free function
`compute_block_hash_for_seq`) take `&Bound<'_, PyAny>` and go through
`crate::tokens::extract_list_or_numpy_u32` instead of extracting a `Vec<u32>`
directly. That accepts a NumPy `uint32`/`int64` array in addition to the
previous `list[int]`, so the Baseten frontend can keep prompt tokens in the
`uint32` buffer the tokenizer produced rather than materializing one Python
`int` object per token (~29 ms for a 1M-token prompt) just to cross the
binding. `list[int]` behavior is unchanged -- it is the last branch of the
same helper. The extraction order and error strings mirror
`extract_list_or_numpy_u32` in the `llm-runtime-metrics` bindings
(`bindings/python/rust/lib.rs`) so both extensions accept the same inputs.
This adds a `numpy` crate dependency to `lib/bindings/python/Cargo.toml`,
which must be kept in lockstep with the pyo3 minor version.

v1.2 Rust generation coordinator:

`dynamo-b10-client::GenerationCoordinator` owns aggregate and prefill-first
generation above `RouterWorkerCoordinator`. It carries the prefill handoff into
decode, merges topology constraints, suppresses the decode bootstrap, and keeps
both router guards alive through the returned stream. The Python extension
only converts serialized worker maps, exposes admission metadata, and adapts
the response stream.

Preserve error annotations before filtering metadata-only messages (#768):
prefill failures retain their original error, bootstrap failures are forwarded
without treating them as successful transfer, and decode errors terminate the
stream after one error item. This restores the legacy Python adapter's
error-before-data handling. Cancellation shielding and bootstrap EOF behavior
remain unchanged; keep the error/cleanup regression coverage when replaying.

Aggregate routing, worker setup, and streaming follow parent cancellation.
Prefill-first follows parent cancellation through the prefill response, shields
the prefill-to-decode routing and connection handoff, then links the decode
stream back to the parent cancellation context after the connection exists.
Keep the cancellation and guard-lifecycle tests with the Rust implementation
when replaying this API.

Return a typed denial variant from `DeniedGenerationRequest.denied_request()`
(#780): the Python extension built the reason with `Py::new` on the
`DeniedRequest` pyo3 complex enum, which produces an instance of the base class
only (pyo3 documents `Py::new` and `.into_pyobject` as inconsistent for complex
enums). Callers discriminating with `isinstance(denied,
DeniedRequest.RouterBackpressure)` therefore never matched, and every denial on
the coordinated path -- router backpressure included -- surfaced as a generic
500 instead of a 429. The accessor now converts with `into_py_any` (as
`route_and_worker` already did) and downcasts back to `Py<DeniedRequest>` so
the Rust signature and the `_core.pyi` stub stay honest. When replaying: never
construct a complex-enum pyclass for Python with `Py::new`; add a binding-level
test that asserts `isinstance(denied_request(), DeniedRequest.<Variant>)` for
every variant, since the harness-side unit tests use a fake denial object and
did not catch this.

v1.2 remote generation coordinator:

The generation coordinator now has a common Rust client trait with local and
remote implementations. The remote path uses a versioned, typed protobuf over
HTTP; sampling is the only extensible MessagePack request section. A native
Hyper/Axum service wraps the local coordinator and exposes `/health` plus the
streaming `/v1/coordinate` endpoint. Python controls service startup and
shutdown through PyO3 but is not present on the request path. The Python remote
constructor accepts a named endpoint map and requires exactly one backend for
now, preserving the API shape for future parallel potential-load probes and
session-aware multi-endpoint selection. Block size and disaggregated request-ID
machine identity remain service-local configuration rather than wire fields.

Replay the protobuf schema, native client/service, PyO3 lifecycle bindings, and
the broad client/server streaming tests together. Before claiming the intended
sub-2 ms proxy overhead, run an optimized p50/p99 benchmark in the integration
image; correctness tests alone do not establish the latency target.

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
Low priority: keep/extend startup logging so runtime defaults used are visible (discovery backend, event plane, request plane).

v1.2 follow-up: the per-request `request received` / `request completed`
events in `lib/runtime/src/pipeline/network/ingress/push_handler.rs`
(inline at `handle_payload` admission and the `RequestMetricsGuard::drop`
impl) were downgraded from `info!` to `debug!`. They fired on every request
and masked more useful INFO-level signal in production logs. Request
correlation is preserved through the existing `request_id` field and
metrics; only the standalone lifecycle events moved to debug.

v1.2 marker-name fix: several Rust log sites carried a misspelled
`unified_logs = true` marker that the log pipeline never matched; they were
renamed to `unified_model_logs = true`. When rebasing or editing log sites,
preserve the `unified_model_logs` / `skip_unified_model_logs` field settings
exactly as spelled — the pipeline matches only these names, so a renamed or
dropped marker silently changes customer-facing log visibility.

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

## PATCH-013: B10 Router Startup Readiness Gate Fix

Status: `keep`

Source commits:

- Current PR: fix: b10 router startup wrong gates

Purpose:

Fix the health/readiness signaling in the B10 KV router so that Kubernetes
sees the router as ready as soon as startup completes (`start_serving_complete.0`
is set), not only when the router is actively serving traffic. A router replica
may wait indefinitely in standby before being elected; withholding K8s readiness
during that wait would block rolling promotions (e.g. 2 active / 4 total pods,
1 max unavailable).

Also fixes a bug where `set_health` was called with 2 arguments instead of the
required 3 (`healthy`, `reason`, `timeout`).

Replay notes:

Apply to any branch that carries the two-stage `(started, serving)` gate in
`lib/bindings/python/rust/llm/b10_router.rs`. The heartbeat loop must gate on
`started || serving`, not on `serving` alone.

## PATCH-014: CI Image Build Overhaul (Depot, sccache, Multi-Arch, Runtime Target)

Status: `keep`

Source commits:

- Current PR: feat: overhaul CI image builds — Depot builders, sccache, multi-arch, runtime target
- Current PR: ci: keep torch/CUDA in a cached layer, drop resolve-image-tag job (#495)
- Current PR: ci: dedupe image-push workflows (post-merge calls bis via workflow_call)

Purpose:

Cut bis-dynamo-image-push from ~20 min flat to time proportional to the change
(identical rerun ~3–4 min, test-only ~3.6 min, minor Rust change ~9 min
multi-arch). Build-infra only; no runtime behavior changes.

- sccache via Depot Cache (WebDAV) using the runner-provided token; S3 fallback
  retained; loud warnings replace silent degradation.
- Depot remote builders replace registry cache-from/to; cargo target cache
  mounts persist across runs.
- bis-dynamo-image-push builds amd64+arm64 natively and publishes a manifest
  list; CI images use `--target runtime` (consumers only need the wheelhouse).
- lld links Rust; Rust test sources excluded from the docker build context.
- dynamo_runtime template: environment layers ordered before wheel_builder-
  derived layers; wheel install takes requirements as `--constraint`.
- dynamo_runtime template: nixl wheels (and their torch/CUDA ~6 GB dependency
  stack) install in the source-independent section for all targets, keeping
  that layer cached instead of re-installed/re-pushed on every source change.
- resolve-image-tag jobs removed from post-merge-build and
  bis-dynamo-image-push; each job computes the deterministic tag inline with
  `git rev-parse --short=9` (pinned so shallow/full checkouts agree), via the
  `.github/actions/resolve-image-tag` composite action.
- post-merge-build.yml is a thin `workflow_call` caller of
  bis-dynamo-image-push.yml (the pipelines were duplicates); post-merge builds
  thereby gain the sccache-verification step, S3 sccache fallback, job
  timeout, lfs skip, and image-tag artifact that previously existed only on
  the dispatched path.

Replay notes:

Entirely Baseten-specific CI/build plumbing across `.github/workflows/`
(bis-dynamo-image-push, post-merge-build), `container/build.sh`,
`container/use-sccache.sh`, `container/templates/`, and `.dockerignore`.
On upstream rebase, replay wholesale; only expect conflicts where upstream
touches the same templates (wheel_builder, dynamo_runtime, args).

## PATCH-015: B10 Preferred-Taint Scoring

Status: `keep`

Source commits:

- Current PR: feat: apply preferred taints in B10 selector

Purpose:

Apply request-level `preferred_taints` to the B10 worker selector's cost before
choosing a worker. Positive weights bias traffic toward matching workers and
negative weights bias traffic away, while `required_taints` remain the hard
eligibility filter.

Replay notes:

Preserve this behavior while B10 remains a Baseten-specific selector. If B10 is
replaced by an upstream selector, verify that the replacement applies
`RoutingConstraints::preferred_taint_multiplier` during worker scoring.

## PATCH-016: Client.instances() Snapshot API Backport

Status: `upstream-sync`

Source commits:

- Current PR: feat(bindings): expose Client.instances() with instance/transport
  snapshot (#11617)

Purpose:

Backport of upstream `ai-dynamo/dynamo#11617` (merged upstream as
`c2a0c0fb8`). Exposes `Client.instances()` to Python: a structured snapshot of
an endpoint's currently-registered instances (`instance_id`, `namespace`,
`component`, `endpoint`, `device_type`, and `transport {kind, address}`),
mirroring the runtime `Instance`/`TransportType` model. Lets a worker discover
peer node addresses (e.g. the startup RDMA connectivity precheck) via a dynamo
API instead of reading the discovery/etcd registry layout directly.

Replay notes:

Already upstream as of `c2a0c0fb8`; drop this patch when rebasing onto any
release that contains it. One local deviation: the backported test file omits
`test_python_request_plane_plain_annotated_error_and_malformed_frames`, which
covers `_dynamo_annotated` response unwrapping that does not exist in the v1.2
runtime.

## PATCH-017: KVBM on_rewind — trim slot state after speculative-decoding rewind

Status: `mixed`

Required when KVBM connector is used. If this branch is deployed with
`kv_connector_config` enabled, the following patches must be present:

- **Rewind API** (`on_rewind` / `rewind_device_blocks`): trims stale freed
  block ids and token sequence after speculative-decoding rewind. Without
  this, `get_finished` hangs when Eagle rejects draft tokens crossing a
  block boundary.
- **Failed onboard recovery** (`recover_failed_load`): separates transfer
  completion from success, rewinds connector state to the native device
  prefix, and suppresses one external rematch so normal prefill recomputes the
  missing suffix.
- **Terminal prefill boundary**: accepts the scheduler's first decode token in
  `num_scheduled_tokens` without requiring a device block for KV that is never
  materialized.

Source commits:

- `7a4e938f2` feat(kvbm): add on_rewind to trim slot state after specdec rewind
- `5b549f17c` fix(kvbm): record on_rewind in recorder and add rewind_device_blocks tests
- `a91abfd63` style: apply cargo fmt formatting
- `107b81364` fix(kvbm): guard on_rewind against missing slots
- `139a41ecc` fix(kvbm): truncate token sequence on rewind
- Current PR: fix(kvbm): recompute prefixes after failed asynchronous loads
- Current follow-up: fix(kvbm): accept sparse TensorRT-LLM connector hashes
- Current follow-up: fix(kvbm): handle the terminal prefill token boundary

Upstream PRs:

- `ai-dynamo/dynamo#11736` — upstream contribution
- `NVIDIA/TensorRT-LLM#16455` — TRT-LLM counterpart
- `NVIDIA/TensorRT-LLM#16448` — upstream issue

Purpose:

Fix a hang in the KV cache connector when used with Eagle speculative
decoding. When rejected draft tokens cause `rewindKVCache` to free blocks
crossing a block boundary, the connector's per-request `block_ids` list was
never trimmed, retaining stale freed block ids. This caused `get_finished`'s
cross-rank `mpi_allgather` + `set.intersection` to never complete.

Changes:

- `ExternallyManagedDeviceSlot` trait: new `rewind_device_blocks` method
- `VllmConnectorSlot.rewind_device_blocks`: trims `device_blocks` to
  `live_block_ids`, clamps `current_position` / `evaluated_blocks` /
  `offload_terminated_at_block`, clears `stored_block_priorities` for freed
  blocks, and truncates `TokenBlockSequence` to `current_position` to prevent
  rejected draft tokens from corrupting block hashes
- `Leader` trait (both `trtllm_leader.rs` and `leader.rs`): new `on_rewind`
  method with `has_slot` guard (matching `request_finished` pattern)
- `PyTrtllmKvConnectorLeader` / `PyKvConnectorLeader`: exposed as pymethod
- `KvConnectorLeaderRecorder`: records `OnRewind` action to the recording
  channel (matching `UpdateStateAfterAlloc` pattern)
- Python wrappers: `DynamoKVBMConnectorLeader.on_rewind` (trtllm) and vLLM
  `KvConnectorLeader.on_rewind` delegate to the Rust leader
- The TensorRT-LLM adapter preserves an absent incremental block-hash update
  as `None`, matching the Rust binding's optional external hash-chain contract
- `VllmConnectorSlot.apply_scheduler_output` recognizes the transition beyond
  the known token sequence before validating device-block coverage, preventing
  a one-token boundary panic when the scheduler begins decoding
- `_core.pyi`: type stub added
- Rust unit tests: covers `[0,1,2] -> rewind [0,1] -> append [3]` cycle and
  no-op-when-growing case

Replay notes:

The TRT-LLM side (`basetenlabs/trt-llm#199`) adds the Python `on_rewind` hook
in `kv_cache_connector.py`, calls it from `resource_manager.py` after each
`rewind_kv_cache` (guarded by `not self.is_draft` so only the target KV
manager notifies the connector), and blocks non-linear-tree specdec +
connector with `NotImplementedError` in `py_executor_creator.py`. If
upstream merges the TRT-LLM PR, the `py_executor_creator.py` guard can be
relaxed once external connectors implement `on_rewind`. The sequence
truncation in `rewind_device_blocks` is needed because without it, rejected
draft tokens remain in `self.sequence` and the next
`apply_scheduler_output` appends new tokens after them, corrupting block
  hashes used for offload/save.

For asynchronous onboard failures, keep the scheduler's failure bit until the
worker has observed completion, then drain failed request IDs separately from
the normal completion list. `recover_failed_load` preserves native device
blocks, clears host/disk match provenance, rewinds the computed position, and
prevents the same external cache entry from immediately matching again. The
paired TensorRT-LLM change performs TP-wide failure consensus and invokes this
hook before resuming the request. This behavior is request-granular because
the TensorRT-LLM connector API does not expose individual invalid block IDs.

## PATCH-018: B10 Residency Capacity Tracking for On-the-Fly Discovery

Status: `keep`

Source commits:

- Current PR: fix B10 residency capacity tracking
- `056f78ba2` fix(router): initialize residency for externally registered workers

Purpose:

Residency setup previously only configured workers present in the worker
table at `configure_residency` time. Workers discovered on the fly — lazily
from a request or replica-sync event, or via the EPP/allowed-worker external
registration path — got a slot but no residency tracker until a later config
path happened to run. This was most visible on passive active/passive routers
at high RPS, where events can surface a worker before the worker-config watch
has supplied its capacity.

Changes:

- Lazily registered workers (request/replica-sync path) now get a residency
  tracker immediately when residency tracking is enabled, using default
  capacity until the worker config is observed.
- Externally registered workers (`register_external_workers`) get the same
  default-capacity residency initialization for each newly added slot when
  residency tracking is enabled, so the EPP path is no longer a gap.
- The B10 hot-reload tick reconfigures residency capacities from the current
  worker-config snapshot, so capacity changes and newly observed worker configs
  apply without waiting for topology churn. Per-worker `set_capacity`
  short-circuits when the effective capacity is unchanged, so the LRU is only
  trimmed when a capacity number actually changes.
- `engine_metrics_total_kv_blocks_override` is parsed from the B10
  hot-reloadable config as a root-level optional integer and stored in a global
  atomic override; `ModelRuntimeConfig::total_kv_blocks()` prefers the override
  before the worker MDC value. `0` is rejected and ignored.

Replay notes:

Discovery now works on the fly for residency: any worker registration path
(lazy, external, or reconcile) ends up with a residency tracker when
`router_track_residency` is on, and the periodic hot-reload tick closes the
gap between default and configured capacity. No per-message LRU work is done
when capacities are unchanged.

## PATCH-019: Router Observability — Request-State Gauges, Queue-Gate Gauges, Lifecycle Counters, kv_indexer_ops

Status: `keep`

Source commits:

- Current PR: feat: router observability (#453); includes the indexer half of
  the closed #409 (`a8aadf36f`), itself the v1.2 port of v1.0 #223.

Purpose:

Make the look-aside router's admission behavior observable: per-worker
request-state, how close each queue admission gate is to engaging, and
lifecycle counters for force-expiry and pre-admission cancellation. Restores
the v1.0 `kv_indexer_ops` metrics lost in the v1.0 -> v1.2 upgrade.

Metrics to preserve across fork upgrades (final names; live-verified on the
2p/2d disagg harness):

- `dynamo_frontend_worker_active_requests{worker_id,dp_rank,worker_type}`
- `dynamo_frontend_worker_active_prefill_requests{...}` — booked, not yet
  mark_prefill
- `dynamo_frontend_worker_active_decode_requests{...}` — derived
  active - prefill
- `dynamo_frontend_router_queue_gate_threshold_tokens{worker_type,gate}` —
  gate in {prefill_busy, decode_tokens} (gate=isl_cap removed, superseded by
  the per-tier isl_tokens pair below)
- `dynamo_frontend_router_queue_isl_tokens_threshold{worker_type,missing_isl_floor}`
  / `dynamo_frontend_router_queue_isl_tokens_last_evaluated{...}` — per
  missing-ISL tier (keyed by the tier's floor), PER-WORKER so values never
  move with fleet size: configured max_queue_depth, and pending-ISL / live
  workers at that tier's last cap evaluation (enforcement scales by worker
  count; priority bonus excluded); replaces the former gate=isl_cap series
- `dynamo_frontend_router_queue_gate_last_evaluated_tokens{worker_type,gate}`
  — gate in {prefill_busy, decode_tokens}; snapshots of each gate's LAST
  admission evaluation
- `dynamo_frontend_router_force_expired_requests_total{worker_type}`
- `dynamo_frontend_router_queue_cancelled_requests_total{worker_type}`
- `dynamo_component_kv_indexer_ops_count{operation}` /
  `dynamo_component_kv_indexer_ops_latency_total{operation}` — operation in
  {stored, removed, cleared, remove_worker, find_matches}; the
  `dynamo_component_` prefix is literal (identity is the `dynamo_component`
  label); only exported by routers that run a KV indexer
- Upstream companions these join on dashboards (do not rename):
  `dynamo_frontend_worker_active_decode_blocks`, `_active_prefill_tokens`,
  `dynamo_frontend_router_queue_pending_requests`, `_pending_isl_tokens`,
  `_backpressure_total{reason}`, `dynamo_component_kv_cache_events_applied`.

Key types/functions to preserve (fork-added surface):

- `WorkerLoadObservation` struct + `SequencePublisher::observe_load` /
  `b10_observe_force_expired_requests` / `b10_observe_worker_removed`
  (default no-ops; the removal hook prunes departed workers' gauge series
  from the look-aside router, where `KvWorkerMonitor` never runs) —
  `lib/kv-router/src/sequences/multi_worker.rs`; sunk into Prometheus by
  `RuntimeSequencePublisher` in `lib/llm/src/kv_router/sequence.rs`
- `WorkerLoadMetrics::observe` (derives active_decode_requests) +
  `WorkerLoadMetrics::b10_remove_series`,
  `RouterSequenceMetrics` + `b10_register_router_sequence_metrics`,
  `RouterQueueMetrics::{b10_inc_cancelled_requests, b10_set_gate_evaluation,
  b10_set_gate_threshold}` — `lib/llm/src/kv_router/metrics.rs`
- `B10QueueEvalGauges` + `SchedulerQueue::b10_eval_gauges` +
  `b10_record_prefill_evaluation` (min-visited capture in all three
  prefill-busy paths), `b10_prune_cancelled_pending` + `cancelled_requests`
  atomic, pub `router_queue_threshold_decode_tokens()` —
  `lib/kv-router/src/scheduling/queue.rs`
- Metrics sync task additions (gate gauges + `b10_sync_cancelled_requests`)
  in `lib/llm/src/kv_router/scheduler.rs`; stale-series cleanup on worker
  removal in `lib/llm/src/discovery/worker_monitor.rs`
- `KvIndexerMetrics::{indexer_ops_count, indexer_ops_latency}` + prebound
  counters in the SyncIndexer worker loops —
  `lib/kv-router/src/indexer/metrics.rs`, `concurrent_radix_tree*.rs`,
  `thread_pool.rs`

Replay notes:

Regression tests to keep: `test_worker_load_metrics_pef`,
`test_router_queue_metrics_pef` (exact-name PEF assertions),
`test_eval_gauges_capture_last_admission_evaluation`. Known caveats carried
deliberately: the cancelled counter only fires when a parked entry's response
channel drops (cross-process cancels do not deliver that signal today; see
PR #452); gate last-evaluated is exact in the default all-workers-busy mode
and may understate under fractional busy mode (>16 workers); counters are
absent until first event (no zero-series materialization — use
`or vector(0)`).

## PATCH-020: Mocker admission cache truth reporting (port of upstream #12711)

Status: `upstream-sync`

Source commits:

- Current PR: feat(mocker): report admission cache truth as first-chunk completion_usage

Purpose:

Port of upstream `ai-dynamo/dynamo#12711` (feat(mocker): report admission cache
truth as first-chunk completion_usage). The vLLM-mode mocker scheduler computes
each request's admission-time cached-prefix tokens (`PrefillCost.cached_tokens`,
post-eviction truth) but never reported it: every stream chunk carried
`completion_usage: None`, so cache-hit surfaces fell back to the KV router's
radix estimate. This change carries the scheduler's truth out on the stream:

- `OutputSignal` gains `cached_tokens: Option<usize>` (serde default +
  skip-if-none, so serialized replay artifacts stay compatible), set once on the
  request's first output signal via `VllmRequestState::take_cached_tokens_for_signal`
  (a preempted request re-probing a cache warmed by its own blocks keeps the
  original count).
- Captured in `schedule_request` alongside `AdmissionEvent.reused_input_tokens`
  from the same local, so on first admission the two never disagree.
- `lib/llm/src/mocker.rs` relays it as `LLMEngineOutput.completion_usage` on the
  first chunk and repeats cumulative totals on the final chunk (OpenAI streaming
  convention). sglang mode reports `None` explicitly.

Replay notes:

Port of upstream `ai-dynamo/dynamo#12711` (open upstream as of this writing);
drop this patch when rebasing onto any release that contains it. Field names,
helper names, and semantics mirror the upstream PR verbatim so the upgrade
merges cleanly; only the surrounding v1.2.0 structure differs (no `rejected`
field on `OutputSignal`, no `already_complete` emission path). The fork's
relay test is a new `lib/llm/src/mocker.rs` `mod tests` that drives
`MockEngine::generate` directly with a wired scheduler channel (upstream
modified its existing test module instead); expect a merge conflict there and
reconcile the two shapes.

Additional upstream mocker sync:

- Backports `ai-dynamo/dynamo#13483` at `d63567331a5fc4fbf3d3c1c51990684362429e47`.
  ZMQ KV-event batches use named MessagePack encoding so optional `medium` and
  `group_idx` fields cannot shift under positional encoding. The v1.2 sink
  serializes batches inline, so the backport adds a small encoding helper while
  preserving the upstream router-wire regression coverage for device stored,
  device removed, and host-pinned stored events. Drop this adaptation when the
  fork advances to an upstream release containing #13483.
- Backports the compiled AIC `EngineHandle` prediction path present in upstream
  commit `7645809841d5b8bb0225bbf351f917883bb0f9bd`. The v1.2 wrapper previously
  walked Python model operations and queried the performance database for each
  changing scheduler context. The backport compiles the model once, dispatches
  prefill and decode predictions to the Rust handle, and retains the Python path
  as a compatibility fallback with an operational opt-out. Drop this adaptation
  when the fork advances to a release containing the upstream implementation.

## PATCH-021: KVBM logical tier-residency event consolidation

Status: `keep`

Source commits:

- Current PR: feat(kvbm): add Baseten tier-dedup event mode

Purpose:

Add the opt-in `baseten_dedup` KV-event consolidator mode. It tracks G1, G2,
and G3 residency as an internal bit mask but exposes one synthetic Device
presence to the router. The first reachable residency publishes `BlockStored`,
intermediate tier stores and removals are suppressed, and the last residency
publishes `BlockRemoved`. Canonical engine metadata bridges KVBM events without
rehashing MLA token spans. Parent loss withdraws descendants leaf-first, while
parent restoration replays still-resident descendants parent-first.

Replay notes:

Keep this as a separate mode until production churn testing establishes it as
the default. Preserve the external-hash-to-canonical-sequence mapping: KVBM's
published hash is the bridge to authoritative engine metadata and must not be
used as the tracker's token-derived internal key.

## GWP Control Plane (global-routing)

Status: `keep` — Baseten-specific control plane; not upstream.

Replay unit covering the GWP Envoy data plane and its loopback control API,
rebased onto `main-v1.2.0`. Native Envoy `ext_authz` owns body buffering and
the synchronous schedule callout (`/gwp/v1/ext-authz`); a Rust Proxy-Wasm
module (`lib/gwp-envoy-filter`) dispatches response-started and request-finished
asynchronously via a VM-root shared queue so neither event blocks the client
stream. The legacy Lua adapter (`deploy/gwp/poc/gwp.lua`) is retained as a
fallback but is no longer bundled by the Kustomization.

Core lifecycle (`lib/gwp/src/core.rs`) keeps per-request mutex-protected state;
optimistic affinity hits defer tokenization and pinned booking into a bounded
4-job background pool, falling back to synchronous scheduling when saturated.
Finish-before-response retains the booking until `mark_prefill_completed` runs,
then frees; a 5 s terminal grace reconciles out-of-order HTTP/2 control calls.

Per-worker `LocalLoadAnchor` (`lib/gwp/src/router.rs`) replaces the
generation-gated anchor reset so a publication for one endpoint no longer
resets another's baseline; the reflector polls each endpoint on its own cadence
(`lib/gwp/src/reflector.rs`). Endpoint ingress URLs must be plaintext `http://`
ending in `/v1`; `api_key` is ascii-validated (CRLF guard) and, when empty,
client `authorization` is explicitly removed rather than inherited.

How to replay: copy the lib/gwp-* crates and the design docs.


## Document Guidelines Reminder

Before adding another top-level patch section, check the document guidelines at
the top of this file. New sections should generally represent a substantial
standalone PR, ideally around 500 LOC or larger, or a distinct future rebase
decision; smaller follow-ups should be folded into the existing relevant
section.

## CC pivot: shared `lib/api-translation` crate (`b10-dynamo-api-translation`)

Chat Completions + `baseten_ext` is the canonical normalization contract
(team decision 2026-08-27). New crate `lib/api-translation` (stack D2),
ported from tool-bank's `api_translation` module (baseten master @
a63c05ca16): `adapt_request` ingress for CC/Messages/Responses, `SseParser`
(CC SSE -> SemanticChunk), `MessageHistoryAccumulator`, the three egress
framers driven by `SseEmitter`/`BufferedEgress`, and tool-bank's
dynamo_conformance_tests as the crate suite (render-equivalence oracle behind
the off-by-default `render-conformance` feature). Server-tool execution stays
in tool-bank behind the `IngressHooks` seam (server-tool claims, react caps,
coding-adapter selection); the `DropServerTools` default is standard-dynamo
behavior (drop hosted tools with a warning, degrade a `tool_choice` naming a
dropped tool to auto). The ordered `unmodeled` catch-all lives on the crate's
own `CcRequest { inner, unmodeled }` wrapper, not on the shared wire type
(a flattened map there makes the Nv wrappers' flatten-inside-flatten collect
every key into `unsupported_fields`). Port deviations from tool-bank that
were reverted on review: the ReAct-cap termination is `completed` again
(tool-bank's anti-retry decision; Codex retries any `incomplete`), and the
in-flight iteration scope is held (`OpenIterationScope`) so a streamed
client sees one `iterations[]` entry per index like the buffered body.
Standard-dynamo leniency the seam adds: a `tool_choice` naming a dropped or
unsupported tool shape degrades to `auto` with a warning
(`IngressHooks::degrades_unsupported_tool_choice`, tool-bank keeps its 400),
and nameless server-tool selection entries are recorded by type so the
degrade catches them. Inherited tool-bank semantics carried as-is (decided
2026-09-03): adjacent text blocks and system blocks join with "\n" — see D6
— unknown user content blocks are skipped by default and 400 only behind
tool-bank's hooks (D4), and unmodeled passthrough keys are last-wins over
the translated body. Nothing in lib/llm uses the crate yet (stack D6).
Baseten-specific; not upstreamable.
every key into `unsupported_fields`). Nothing in lib/llm uses the crate yet
(stack D6). Baseten-specific; not upstreamable.

Egress conformance grafts over the shared crate's tool-bank framing (stack
D5): status-aware public error framing (OpenAI/Anthropic `error.type` by HTTP
status class — 429 rate_limit, 4xx invalid_request/authentication/permission/
not_found/request_too_large, 529 overloaded, 5xx api_error; Responses stream
failures as `response.failed` with string codes plus the "cutoff by
max_tokens" -> `response.incomplete` rescue); the Responses envelope echoes
`metadata`, `top_logprobs`, `presence_penalty`, `frequency_penalty`,
`service_tier` as sent (spec defaults only when omitted);
`cache_write_tokens` in Responses usage and explicit
`cache_creation_input_tokens: 0` in Anthropic usage; process-unique synthetic
ids; mid-stream `error.code` parsed as number or decimal string; the
`CodingAdapter` tool-identity seam the Responses framer resolves through;
non-terminal chat chunks pinned to OMIT `finish_reason`; streamed
`baseten.iterations[]` scopes released when their iteration closes
(`StreamFraming::close_iteration`) instead of one iteration late (ported from
tool-bank, baseten #27828); server-tool call records and outcomes carry the
billing verdict as tool-bank ships it (`billable`, `sku`, `quantity` present
only when the call bills; `ToolOutput.verdict` mirrors `BillingVerdict` /
`UsageReport` as data, baseten #27386). Baseten-specific.

Anthropic's 1024 floor on `thinking.budget_tokens` is not enforced. That floor is
a fact about their models; the engines here cap reasoning at whatever token count
they are given, so a smaller budget is a request they can serve and refusing it
turns a working parameter into a 400. The upper bound stays: a budget at or above
`max_tokens` leaves no room for an answer, so reasoning consumes the completion
and the client gets `content: null` with `finish_reason: "length"`.

Ingress edges on the shared crate (stack D4) — the validation floor and
parity fixes live agent traffic and the bx suites taught us, each with a test
in `request_test.rs`: Messages `temperature`/`top_p` 0..1, thinking budget
< max_tokens, client-tool `input_schema` required, orphan
`tool_result` 400; Responses `temperature` 0..2, orphan `function_call_output`
400; a trailing assistant turn sets CC `partial: true` (prefill); Anthropic-only
top-level fields (`service_tier`, `cache_control`, `context_management`, ...)
dropped with a warning instead of forwarded (`CC_EXTENSION_KEYS` names the
fork extension surface that rides through); Responses accepts
`include: ["reasoning.encrypted_content"]` by default (hook-selectable
refusal for tool-bank) and every `reasoning.summary` value, declares
`additional_tools` items as CC tools, and 501s `conversation` with the other
stateful fields; `service_tier` maps one-to-one (`auto` is not `default`);
interleaved thinking/tool replay emits `ReasoningContent::Segments`; user
image blocks translate to CC image parts; user `document` blocks and unknown
assistant blocks/items drop with a warning instead of 400ing; Codex
`namespace` tool groups flatten to `{ns}__{name}`; `store` is carried; empty
assistant turns and refusal parts survive as turn boundaries;
`top_logprobs > 20` is refused; top-level `tool_settings` nesting guard
restored. Accepted tool-bank deltas pinned with `// CC-pivot: tool-bank
semantics`. Review follow-ups: Responses replay now closes the reasoning
segment at every replayed `function_call`/`mcp_call` like the Messages path
(a `[reasoning, function_call, reasoning]` transcript re-renders byte-exactly);
the thinking-budget upper bound applies only when the client sent
`max_tokens` (the deployment template may supply it); Codex namespace
flattening, `top_logprobs` bounds, and empty-turn/refusal boundaries are
pinned by tests. Also in this slice: `take_body_user`, unknown `baseten`
extension members warn+drop, plain `text.format` lowers to no
`response_format`, `pub adapt_request_json`, `content_kind` removal, `store`
accepted. Separator ruling (2026-09-03): adjacent text blocks in `system`,
user, and assistant messages join with "\n" — the deployed converter's
shape — not tool-bank's "". Standard-dynamo leniency behind
`IngressHooks::rejects_unsupported_messages_features` (default false;
tool-bank sets true): untranslatable user content blocks are skipped with a
warning, and `mcp_servers` / `container` are dropped with a warning, where
tool-bank 400s. Ingress rejections name the offending JSON member
(`serde_path_to_error`; Responses `input[i]` parsed per index) and Responses
`reasoning.effort` resolves through the fork's `REASONING_EFFORT_ALIASES`
table like chat `reasoning_effort` (`max` -> `xhigh`, no longer a 400; ported
from tool-bank #27635/#27828). The crate carries no consumer tool namespace
(review, Marius): server-tool routing is by shape only (CC/Responses
non-`function` `type`, Anthropic non-`custom` `type` -> the hooks; a plain
deployment drops with a recorded loss — wire-visible: a CC `web_search_preview`
entry used to be forwarded verbatim), `RESERVED_TOOL_PREFIX` /
`reserved_tool_provider` / the reserved-name client-tool guard move to
tool-bank (guarded at dispatch), the Responses framer takes the provider label
from the loop's call records, ReAct bounds (`max_react_iterations`,
`server_tool_iterations_floor`) come from `IngressHooks::limits()` instead of
crate constants, and the steering-note wording leaves the schema module. Every
drop, skip, degrade, or fold on the
ingress path is now a typed `Loss { kind, field, detail }` on
`AdaptedRequest.losses` (closed `LossKind` vocabulary, labels safe as metric
labels; `field` is the JSON path, never message content), handed to the new
`IngressHooks::on_loss` (replaces `on_non_fatal`/`NonFatalCondition`) and
logged once under `event_name = "ingress.loss"`; `adapt_request_json` emits
the canonical per-stage line `stage.ingress` (elapsed_ms, outcome,
error_class/status on rejection, model, losses, loss_kinds), replacing
`predict.ingress_adapted`. Baseten-specific.

## CC pivot: frontends on `b10-dynamo-api-translation` (ingress)

The Messages and Responses handlers in lib/llm canonicalize through the shared
crate (stack D6): they take the client's own JSON (`Json<Value>`, typed view
parsed alongside — never a re-serialized struct, which had sent the system
prompt to the model as `{"text": ...}`), call `adapt_request_json` with
`DropServerTools`, and hand the canonical CC body to the existing wire-edge
re-parse into the Nv wrapper (stream forcing, chat_template_kwargs
distribution, residual-unmodeled drain into `unsupported_fields` preserved).
Conversion rejections map to 400; the deliberate stateful-field 501s stay in
`validate_response_unsupported_fields`. The old lib/llm Messages->CC and
Responses->CC converters are deleted. The switchover is INGRESS-ONLY: lib/llm
keeps its Anthropic and Responses stream converters for egress (the crate's
framing is consumed by tool-bank); routing frontend egress through the crate is
the named follow-up. `ResponseParams` echoes `metadata`, `top_logprobs`, and
the penalties (projected from the raw body) so the frontend envelope agrees
with the crate's. Template defaults: the template's `model` still applies
before canonicalization; its `temperature` / `max_completion_tokens` apply to
the canonical CC request AFTER the Anthropic wire validation, so an
OpenAI-ranged template temperature no longer 400s templated `/v1/messages`
requests for a value the client never sent. Disclosed global wire change:
`ErrorMessage.code` serializes as a string on every OpenAI-shaped endpoint
including `/v1/chat/completions`.
The handlers count the crate's typed ingress losses on
`{prefix}_b10_ingress_losses_total{model,endpoint,kind}` (the request still
succeeds, so this counter is where a quiet degradation becomes visible) and
the canonicalizers emit the per-stage line `stage.canonicalize` (elapsed_ms,
outcome, model, losses, loss_kinds) around the crate's own `stage.ingress`.
Validated live on FDE GLM-5.2 across the bx
behavior suites and real Claude Code / Codex sessions. Baseten-specific; not
upstreamable.

Two ingress corrections from replaying 1,000 requests across 12 production
configs against the pre-pivot converters. Adjacent Responses `input_text` parts
join with `"\n"` (`flatten_input_content`, the all-text branch of
`user_input_content`, and `function_call_output_text`): they were concatenated
with no separator, where the deleted converter kept the part list and every chat
template rendered a newline between parts, so the last word of one part ran into
the first word of the next on 22 production requests per model. The Messages
side already applied that separator (`flatten_text`), so the two ingress paths
had disagreed. And both handlers take the client's body as raw bytes
(`http::service::RawJson`) and parse the typed view off those bytes through
`serde_path_to_error` (`http::service::deserialize_body`), restoring the field
path and the position the axum `Json<T>` rejection carried:
"messages[6].role: unknown variant `tool`, ... at line 1 column 64" rather than
"unknown variant `tool`". `RawJson` replaces `Json<serde_json::Value>` in those
two handlers and keeps its content-type and syntax-error rejections unchanged.
