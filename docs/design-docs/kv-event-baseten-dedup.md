# `baseten_dedup` KV event consolidation mode

## Goal

Keep the router's trie aligned with a worker's logical KV availability while a
block moves between device (G1), host (G2), and disk (G3) cache. The router does
not need to know which tier contains the block: every available block is
published as one synthetic device-cache residency.

The existing `dedup` and `passthrough` modes remain unchanged. Deployments opt
in with:

```text
DYN_KVBM_KV_EVENTS_CONSOLIDATOR_MODE=baseten_dedup
```

## State model

Each canonical sequence hash has one record containing its canonical store
metadata, parent sequence hash, and a producer-specific residency mask:

```text
ENGINE_DEVICE = 0b0001
KVBM_DEVICE   = 0b0010
KVBM_HOST     = 0b0100
KVBM_DISK     = 0b1000
```

Input stores and removals always update this internal mask. They produce a
router event only when aggregate availability changes:

```text
000 -> nonzero: publish one BlockStored event
nonzero -> nonzero: publish nothing
nonzero -> 000: publish one BlockRemoved event
```

For example, `G1 store -> G2 store -> G1 remove -> G2 remove` produces exactly
one router store followed by one router removal. All output events use the
device medium; the four residency bits remain private implementation details.

This first version targets base-model TRT-LLM/KVBM traffic. LoRA, multimodal,
and Eagle identity reconciliation remain outside this Baseten-specific tracker;
the existing generic event path is unchanged. In particular, KVBM does not yet
carry the adapter identity required to reconcile a KVBM-first LoRA handoff.
If a later engine event supplies a different external hash or richer identity
metadata for an already-advertised KVBM record, the tracker explicitly removes
the old subtree before re-advertising it parent-first under the new identity.
It never silently changes the hash used by a prior router event.

The canonical store metadata is retained until the last residency disappears.
KVBM events identify their device, host, or disk residency through their
storage tier, while engine events identify engine-device residency. Repeated
stores and removals for an already-set or already-cleared bit are idempotent.

## Parent and handoff correctness

A block can be advertised only after its parent is advertised. Physically
resident children whose parent is absent remain recorded but hidden. When the
parent returns, reachable descendants are re-announced in parent-first order.
This prevents inserting a child into the trie without its lineage.

When the final confirmed residency disappears, the tracker publishes REMOVE
immediately and prunes the record. A later KVBM STORE is self-contained: its
tokens and parent hash recreate canonical metadata and publish a fresh STORE
once its parent is available. This can produce a truthful REMOVE/STORE pair
during an asynchronous handoff, but it never keeps stale residency advertised
and needs no timeout-based metadata retention.

## Implementation plan

1. Add `baseten_dedup` to Rust and Python mode parsing.
2. Add a dedicated tracker using producer-specific device/host/disk residency
   bits and canonical metadata.
3. Preserve the existing ZMQ/NATS wire format and publish all output as device
   residency.
4. Track parent/child relationships and recompute visible descendants after a
   parent transition.
5. Add transition metrics and INFO/DEBUG diagnostics for suppressed tier
   changes, aggregate stores/removals, and orphan re-announcements.
6. Recreate pruned canonical metadata from late KVBM stores.

## Validation

Unit tests cover every tier order, duplicate transitions, G1-to-G2 handoff,
last-tier eviction, child-before-parent arrival, parent eviction/reappearance,
late lower-tier store after engine removal, external-hash migration, clear-all,
and unchanged behavior for the existing modes. Integration testing will compare
worker physical residency against router matches during sustained offload and
eviction churn.
