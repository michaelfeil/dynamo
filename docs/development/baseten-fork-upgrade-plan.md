# Baseten Fork Upgrade and Patch Changelog Plan

## Goal

Maintain the Baseten Dynamo fork as a documented patch stack on top of upstream
Dynamo releases.

The immediate source branch is `main-v1.0.0`. The upstream repository is
`https://github.com/ai-dynamo/dynamo`. The desired future workflow is:

1. Start from a clean upstream release branch, for example `main-v1.2.0`.
2. Review every Baseten-specific change that still matters.
3. Reapply those changes deliberately, line by line or patch by patch.
4. Preserve a changelog that explains why each patch exists, what it changes,
   and whether upstream already accepted the same behavior.

The output of this process should be a `CHANGELOG.md`-style patch ledger, not
just a Git history dump. Git commits are evidence; the changelog is the durable
source of intent.

Current reviewed target for the next rebase:

```text
bed9f269312151481cd67a8d21b70e0f52424c2b feat(mocker): add AIC forward-pass engine perf shim (#10150)
```

As of review, there was no fetchable upstream branch or tag literally named
`main-v1.2.0` or `v1.2.0`. Use the SHA above as the upstream base and create the
local Baseten branch name from it.

## Current Repository Facts

Observed on `2026-06-01`:

- Source branch reviewed: `main-v1.0.0`
- Local target branch created: `main-v1.2.0` at
  `bed9f269312151481cd67a8d21b70e0f52424c2b`
- Current fork remote: `origin -> https://github.com/basetenlabs/dynamo`
- Source branch head reviewed:
  `c9bf0c4ed fix: Make router active replicas hot configurable (#251)`
- Local upstream-like ref already present: `origin/upstream/v1.0.0`
- Local `v1.0.0` tag points at `72c26adb8`, which appears to be part of the
  Baseten patch stack, not necessarily the canonical upstream release tag.

Because the local tag may not be the true upstream tag, prefer explicit upstream
refs such as `upstream/v1.0.0` after adding the upstream remote, or the existing
`origin/upstream/v1.0.0` tracking branch if that branch is known to mirror
upstream.

Initial divergence from `origin/upstream/v1.0.0` to `main-v1.0.0`:

```bash
git rev-list --left-right --count origin/upstream/v1.0.0...main-v1.0.0
# 9 161
```

That means the local comparison currently sees 9 commits only on the upstream
side of the symmetric range and 161 commits only on the Baseten fork side. This
count is a starting point, not the final patch count, because merge commits,
fixup commits, reverts, upstreamed changes, and squashed changes need to be
collapsed into logical entries.

## `main-v1.1.0` Strategy Reference

The current `origin/main-v1.1.0` branch demonstrates the intended upgrade
style. It advances through upstream commits to `cc5b2cd29` (`chore: bump
version references to v1.1.0`), then applies Baseten changes as a compact
logical patch series:

- `9e5806c82`: CI/fork-survival workflows, `container/build.sh`, version
  stamping, and repo hygiene.
- `14e94a0f9`: runtime resilience, lifecycle ordering, transport hardening, and
  TCP default behavior.
- `ce5dfa1c9`: OTel exporter environment handling.
- `1de204080`: B10 worker selector, hot-reloadable config, and snapshot toggle.
- `a3f024b40`: hot-reloadable router queue threshold, queue metrics, and
  `respond` result plumbing.
- `6f36f18d4`: OpenAI protocol extensions, `baseten_ext`, B10 health, and
  rate-limit behavior.
- `d493fef1d`: Anthropic protocol conformance.
- `9adbfdc64`: B10 router and service pipeline Python bindings.
- `9cfeeb5e2`: restored JSON publisher/subscriber Python bindings.
- `e127637c2`: dispatchable BIS Dynamo Image Push workflow.

That branch is the closest precedent for `main-v1.2.0`: replay logical behavior
groups after the upstream base, not the full old commit stream. The difference
is that `main-v1.2.0` should be more selective. The reviewed target SHA already
contains newer router recovery, tiered queueing, parser/tool-call handling, and
first-token infrastructure, so the v1.1 replay commits should be treated as
evidence and tests before they are treated as patches to port.

Current `main-v1.2.0` replay status: the B10 health-file and header rate-limit
portion of `6f36f18d4` has been ported because it is client/platform-facing and
not replaced by upstream. The typed root-level `baseten_ext` request fields
have also been restored for chat/completions, with v1.1-style
validate-or-error behavior for invalid Baseten-maintained fields. Anthropic
conformance from `d493fef1d` has also been ported because it is direct API
surface area: no OpenAI `[DONE]` marker on Anthropic streams, Anthropic-native
`toolu_` IDs, and backend status passthrough for Anthropic errors. The remaining
Python HTTP-engine first-yield gate from `9adbfdc64` has been restored through
the target `HttpAsyncEngine` shape. B10 `router_queue_threshold` hot reload from
`a3f024b40` has been restored through the target actor-based router queue rather
than the old v1.1 queue lock. The v1.1 B10 config-map warning suppression flag
has also been restored. Runtime shutdown lifecycle logs now carry the v1.1
`unified_model_logs` marker where the target still has matching lifecycle
points. The Gemma 4 parser stack from PATCH-008 was not replayed wholesale
because the target already has broad Gemma 4 support, but the v1.1
default-thinking-off decision was kept: `gemma4` and `gemma-4` disable reasoning
parsing unless `chat_template_args.enable_thinking` is explicitly true.
Selective endpoint activation is already present in the target Python
`HttpService.enable_endpoint(...)` path, so it is treated as satisfied rather
than replayed. Arbitrary Python worker selector callbacks remain dropped, in
line with v1.1; Python users select the maintained Baseten policy through
`RouterConfig(..., algo_selector="B10")`. Python JetStream object-store access
from PATCH-003 remains dropped because Baseten no longer performs JetStream
offloading. The remaining protocol work should stay split into smaller
follow-up decisions for parser, response-shape, or tolerance compatibility only
if tests prove the target still regressed a Baseten client expectation. The
v1.1 warn-and-ignore behavior for generic unknown fields was not ported in the
`baseten_ext` slice; keep that as a separate client-test decision.

## Definitions

- **Upstream base**: The clean Dynamo release or commit that Baseten originally
  forked from, for example `upstream/v1.0.0`.
- **Fork branch**: The Baseten release branch, for example `main-v1.0.0`.
- **Patch ledger**: The markdown changelog that records Baseten logical patches,
  their source commits, rationale, touched files, tests, and upstream status.
- **Logical patch**: A coherent behavior change. It may be one commit, several
  commits, a squashed merge, or a cleaned-up rewrite of historical commits.
- **Upstreamed patch**: A Baseten behavior that upstream accepted exactly or in
  modified form. These should usually not be replayed blindly on a newer
  upstream version.

## One-Time Setup

Add the canonical upstream remote if it is not already configured:

```bash
git remote add upstream https://github.com/ai-dynamo/dynamo
git fetch upstream --tags
git fetch origin
```

Record the exact refs used for each upgrade attempt:

```bash
git rev-parse upstream/v1.0.0
git rev-parse origin/main-v1.0.0
git merge-base upstream/v1.0.0 origin/main-v1.0.0
```

If the repository does not have an upstream branch named `v1.0.0`, use the exact
upstream tag or commit SHA instead. The planning documents and changelog should
always store SHAs, not only branch names.

## Phase 1: Build the Raw Commit Inventory

Create a working directory for generated upgrade notes:

```bash
mkdir -p upgrade-notes/v1.0.0
```

Capture the fork-only commits with merge context:

```bash
git log --first-parent --reverse --oneline \
  upstream/v1.0.0..origin/main-v1.0.0 \
  > upgrade-notes/v1.0.0/first-parent.txt
```

Capture all fork-only non-equivalent commits. This removes commits whose patch
ID is already equivalent across the two sides of the range:

```bash
git log --reverse --cherry-pick --right-only --no-merges --oneline \
  upstream/v1.0.0...origin/main-v1.0.0 \
  > upgrade-notes/v1.0.0/fork-only-nonmerge.txt
```

Capture changed file summaries:

```bash
git diff --stat upstream/v1.0.0..origin/main-v1.0.0 \
  > upgrade-notes/v1.0.0/diffstat.txt

git diff --name-status upstream/v1.0.0..origin/main-v1.0.0 \
  > upgrade-notes/v1.0.0/name-status.txt
```

Capture patch-ID equivalence data for later upstreamed/squashed detection:

```bash
git cherry -v upstream/v1.0.0 origin/main-v1.0.0 \
  > upgrade-notes/v1.0.0/git-cherry.txt
```

Interpretation:

- `+` means Git does not see an equivalent patch upstream.
- `-` means Git sees an equivalent patch already present upstream.

The `-` entries are strong upstreamed candidates. The `+` entries still require
manual review because upstream may have accepted a modified version that does
not produce the same patch ID.

## Phase 2: Collapse Commits Into Logical Patches

Do not turn every commit into a changelog entry. Instead, group commits by
behavior and ownership area.

Suggested categories:

- CI and release infrastructure
- Runtime resilience and graceful shutdown
- Router and KV router behavior
- OpenAI and Anthropic protocol compatibility
- Python bindings and user-facing APIs
- Container and dependency version changes
- Observability, tracing, logging, and metrics
- Baseten-specific configuration or platform integration

Useful commands while grouping:

```bash
git show --stat <commit>
git show --name-status <commit>
git show --find-renames <commit>
git log --oneline --ancestry-path <base>..<merge_commit>
```

For merge commits, inspect the merge branch rather than only the merge commit:

```bash
git log --oneline --reverse <merge_commit>^1..<merge_commit>^2
git diff --stat <merge_commit>^1..<merge_commit>
git diff <merge_commit>^1..<merge_commit>
```

Each logical patch should get one changelog entry with all contributing commits
listed under `Source commits`.

## Phase 3: Detect Patches Already Accepted Upstream

Use several checks. No single command is sufficient.

### Exact or Near-Exact Patch Match

```bash
git cherry -v upstream/v1.0.0 origin/main-v1.0.0
```

Entries marked `-` are likely exact patch matches.

### Rewritten or Squashed Match

Compare the logical patch branch against a newer upstream release or target SHA:

```bash
TARGET_UPSTREAM=bed9f269312151481cd67a8d21b70e0f52424c2b
git range-diff upstream/v1.0.0..origin/main-v1.0.0 \
  "$TARGET_UPSTREAM"..baseten/replay-v1.2.0
```

This comparison becomes useful after a replay branch exists. Before replay, use
the target SHA directly for path-level evidence:

```bash
git log "$TARGET_UPSTREAM" -- <path>
git blame "$TARGET_UPSTREAM" -- <path>
git grep -n '<distinctive_behavior_or_config_name>' "$TARGET_UPSTREAM" -- <path>
```

When looking for whether upstream absorbed a specific change:

```bash
git log --all --grep '<distinctive words from commit title>'
git log "$TARGET_UPSTREAM" -- <path>
git blame "$TARGET_UPSTREAM" -- <path>
```

Also search by behavior, not just commit title. Upstream may accept the same
idea with a different implementation or commit message.

### Manual Status Labels

Every changelog entry should have one of these statuses:

- `keep`: Still Baseten-specific and should be replayed.
- `upstreamed-exact`: Equivalent patch exists upstream; do not replay.
- `upstreamed-modified`: Upstream has the behavior, but implementation differs;
  verify before dropping.
- `obsolete`: No longer needed because upstream architecture changed or the
  product requirement changed.
- `needs-redesign`: Still needed, but should be reimplemented for the newer
  upstream code.
- `unknown`: Not reviewed yet.

## Patch Ledger Format

Create or maintain a file such as:

```text
baseten-changelog.md
```

Recommended entry template:

```markdown
## PATCH-0001: Short behavior name

Status: keep
Area: router
Introduced on: main-v1.0.0
Source commits:
- c9bf0c4ed fix: Make router active replicas hot configurable (#251)

Purpose:
Explain the operational or product reason this patch exists.

Behavior:
Describe the externally observable behavior, not just code movement.

Files touched:
- path/to/file.rs
- path/to/test.rs

Replay notes:
Describe how to reapply this on a future upstream branch. Include conflicts,
ordering dependencies, and any code that should be rewritten instead of copied.

Upstream status:
Unknown / exact / modified / obsolete. Include upstream commit or PR links when
known.

Validation:
List tests, manual checks, or production signals that prove the patch still
works.
```

Use stable patch IDs such as `PATCH-0001` so future branches can refer to the
same logical patch even when commit SHAs change.

## Phase 4: Prepare the Next Upgrade Branch

For the reviewed `main-v1.2.0` target:

```bash
git fetch https://github.com/ai-dynamo/dynamo bed9f269312151481cd67a8d21b70e0f52424c2b
git switch -c main-v1.2.0 bed9f269312151481cd67a8d21b70e0f52424c2b
```

Create a replay branch:

```bash
git switch -c baseten/replay-v1.2.0
```

Replay patches in ledger order, not raw chronological commit order. Ledger order
should respect dependencies. For example, configuration plumbing should precede
features that depend on it, and metrics infrastructure should precede metrics
emission changes.

For each patch:

1. Read the ledger entry.
2. Check whether the target upstream SHA already has the behavior.
3. If still needed, apply the smallest coherent change.
4. Run targeted tests.
5. Update the ledger with replay notes and the new commit SHA.

Use `git cherry-pick` only when the patch is still structurally compatible:

```bash
git cherry-pick -x <old_commit>
```

If the surrounding code changed materially, prefer a manual reimplementation and
record the old commit under `Source commits`.

## Phase 5: Produce the Human Changelog

After review, distill the patch ledger into a shorter changelog for humans:

```text
baseten-changelog.md
```

This file should contain:

- Patch ID
- Short title
- Source commit IDs
- Status
- One-paragraph purpose
- Replay decision for the next upstream version

The human changelog should not contain full diffs. Store diffs separately if
needed:

```bash
git format-patch --output-directory upgrade-notes/v1.0.0/patches \
  upstream/v1.0.0..origin/main-v1.0.0
```

## Recommended Review Order for `main-v1.2.0`

Based on the target SHA review, start with small concrete deltas and use tests
to decide whether larger historical subsystems still need replay:

1. TCP message-size default: target has `DYN_TCP_MAX_MESSAGE_SIZE` but defaults
   to 32 MiB; Baseten likely wants a tiny 256 MiB default patch.
2. CI, release, and image infrastructure that references Baseten branches,
   registries, runners, or image stamping.
3. Structured logging fields and request correlation that remain absent on the
   target.
4. Router/indexer metric names and labels required by existing dashboards.
5. Runtime shutdown/drain invariants, after testing target behavior.
6. Protocol compatibility tests for tool calling, reasoning, Anthropic streams,
   and Baseten extensions; drop the old non-streaming `stream_options`
   tolerance unless product requirements change.
7. Python service-facing API and first-token behavior, after verifying the
   target's existing `notify_first_token` path.
8. Router and NATS recovery work only where target P2P recovery, tiered ISL
   queueing, and MDC/configmap routing data do not satisfy Baseten needs.
9. Model/parser/vLLM deltas only when the target support matrix or parser tests
   show a concrete gap.

This order should keep the replay branch small. Most old router queueing,
NATS-centric recovery, upstream sync, and reverted experiment commits should be
used as audit evidence rather than replay candidates.

## Acceptance Criteria

The fork management work is complete when:

- Every fork-only logical patch from `main-v1.0.0` has a patch ledger entry.
- Every entry has a status other than `unknown`.
- Every `keep` or `needs-redesign` entry has replay notes for the next upstream
  target.
- Every `upstreamed-exact` or `upstreamed-modified` entry includes the upstream
  commit or PR evidence.
- The future branch, for example `main-v1.2.0`, can be rebuilt from upstream plus
  the ledger without depending on undocumented historical context.
