---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: KVBM G4 NVMe RAID Plan
---

# KVBM G4 NVMe RAID-Backed Storage Agent Plan

**Status:** Draft plan

This document proposes a first `G4` implementation for KVBM using NVMe RAID-backed storage agents. The design intentionally favors a simple cache architecture over a fully distributed storage system:

- one deterministic owner per block
- no redundancy
- no background repair
- worker-to-agent direct query and fetch
- event plane used only as optional fallback or metadata hinting, not as the primary lookup path
- pinned-host staging on the storage agent for remote transfers
- NIXL over UCX as the recommended remote transfer mechanism

## Overview

The current KVBM implementation actively supports `G1/G2/G3` flows. `G4` exists in the type system and design docs, but not yet as a completed runtime path. This plan defines a practical first `G4` path that works on local high-throughput storage nodes without requiring a shared filesystem or a distributed metadata mesh.

The key idea is:

1. Workers derive `sequence_hash` locally from token blocks.
2. Workers compute the owning `G4` storage agent from a deterministic hash over the active agent set.
3. Workers query the owner directly for exact block existence.
4. On hit, workers fetch the block payload from that owner and onboard it locally.
5. On miss or transfer failure, workers treat the block as a cache miss and recompute.

## Design Goals

1. **Simple correctness model** - `G4` is a cache, not a source-of-truth store. Misses are acceptable and fall back to recompute.
2. **High-throughput local storage** - Each storage node uses a local NVMe RAID volume for strong read bandwidth and simpler capacity management.
3. **Deterministic routing** - Workers can compute block ownership locally without a global per-block metadata service.
4. **Low coordination overhead** - No cross-agent lookup gossip or background replica repair in the hot path.
5. **Incremental integration** - The first version should fit the existing KVBM architecture without requiring a full rework of `OffloadManager`.

## Non-Goals

- No block redundancy in the first version
- No background repair
- No prefix-tree or radix-tree lookup as a primary requirement
- No event-plane dependence for correctness
- No shared POSIX mount as the main data path
- No attempt to make `G4` a strongly consistent distributed database

## Architecture

### Components

- **Inference Worker**: Computes block hashes locally, queries/fetches from the owner, and onboards blocks into local KVBM tiers.
- **G4 Storage Agent**: Owns a shard of blocks, stores payloads on local NVMe RAID storage, maintains local metadata, and serves query/fetch/put APIs.
- **Discovery Plane**: Publishes the live set of storage agents and their endpoints.
- **Optional Event Plane Subscriber**: Consumes block lifecycle events to warm metadata or support fallback observability workflows.

### Data Model

Each block is treated as an immutable object keyed by `sequence_hash`.

Suggested local metadata record:

```rust
struct G4BlockMeta {
    sequence_hash: u64,
    block_hash: u64,
    size_bytes: usize,
    checksum: Option<[u8; 32]>,
    disk_path: String,
    last_access_unix_ms: u64,
    lease_until_unix_ms: Option<u64>,
    block_size_tokens: usize,
    model_signature: String,
}
```

Only exact-block lookup is required for the first version. A local `HashMap<SequenceHash, G4BlockMeta>` or embedded KV store is sufficient.

## Hashing and Block Identity

KVBM already has two relevant hash concepts:

- **`BlockHash`**: content-only hash of the tokens in a single block
- **`SequenceHash`**: parent-aware chained hash used as the actual reusable block identity

Current implementation details:

- `compute_hash_v2()` uses `xxh3_64_with_seed(...)`
- `block_hash = hash(tokens, salt_hash)`
- first block: `sequence_hash = block_hash`
- subsequent blocks: `sequence_hash = hash([parent_sequence_hash, block_hash], salt_hash)`

For `G4`, the primary lookup key should be `sequence_hash`.

`block_hash` may still be stored in metadata for debugging, validation, or future secondary indexing, but it should not replace `sequence_hash` as the ownership and lookup key.

Checksums are separate from identity. They should be treated as transfer-validation metadata, not as the canonical KVBM block key.

## Routing and Ownership

### Ownership Rule

Each block has exactly one `G4` owner:

```text
owner = rendezvous_hash(sequence_hash, active_storage_agents)
```

Rendezvous hashing is preferred because it is simple to compute locally and handles membership changes cleanly.

### Discovery Dependency

Workers do not need a global block index, but they do need:

- the current live storage-agent membership set
- endpoint addresses for those agents
- a ring or membership epoch to reason about topology changes

Discovery is therefore responsible for agent liveness and endpoint publication, not for per-block lookup.

## Storage Layout on NVMe RAID

Each storage agent writes blocks to its own local NVMe RAID volume. The simplest initial layout is:

- hash-based subdirectories to avoid oversized single directories
- immutable block payload files
- separate metadata store for lookup

Example:

```text
/nvme-raid/kvbm-g4/ab/cd/<sequence_hash>.blk
/nvme-raid/kvbm-g4/ef/01/<sequence_hash>.blk
```

The agent should treat the RAID volume as a single fast local store rather than managing per-disk block placement itself.

## APIs

The first version should expose block-centric APIs, not request-centric ones.

Suggested interface:

```rust
trait G4StorageAgent {
    async fn query_blocks(&self, hashes: Vec<u64>) -> Result<Vec<G4BlockHit>>;
    async fn fetch_blocks(&self, hashes: Vec<u64>) -> Result<Vec<G4BlockPayload>>;
    async fn put_blocks(&self, blocks: Vec<G4PutBlock>) -> Result<()>;
}
```

Suggested types:

```rust
struct G4BlockHit {
    sequence_hash: u64,
    size_bytes: usize,
    checksum: Option<[u8; 32]>,
}

struct G4PutBlock {
    sequence_hash: u64,
    block_hash: u64,
    size_bytes: usize,
    bytes: bytes::Bytes,
    checksum: Option<[u8; 32]>,
}
```

The API intentionally does not include `has_request()`. Requests are transient scheduler concepts; reusable cache state is block-based.

## Transfer Plan

### Recommended Remote Transfer Path

For the first version, remote block transfer should use:

- **Control plane:** direct RPC-style query and fetch requests to the owning storage agent
- **Data plane:** NIXL over UCX
- **Staging model:** storage agent reads from NVMe RAID into pinned host memory, then transfers to the worker

The recommended read path is:

```text
NVMe RAID file -> pinned host buffer on storage agent -> NIXL/UCX transfer -> worker host/device staging -> local onboard
```

This is intentionally conservative.

The first version should not depend on direct remote-disk-to-device transfer. Pinned-host staging is the simpler and safer starting point.

### Why This Transfer Path

- It aligns with the current transfer direction already favored in Dynamo and TensorRT-LLM runtime docs.
- It avoids coupling the first `G4` implementation to remote GDS assumptions.
- It keeps the storage agent responsible for reading local RAID-backed payloads.
- It allows workers to keep using familiar host-to-device onboard paths locally.

## Data Flow

### Write Path

When a worker produces a registered block that should be materialized in `G4`:

1. Worker computes `sequence_hash`.
2. Worker computes the owning storage agent from discovery membership.
3. Worker sends `put_blocks()` to that owner.
4. Owner persists the payload to local NVMe RAID storage and records metadata locally.
5. On success, the block is available in `G4`.
6. On failure, the block is simply not cached remotely.

This write path is best-effort. Since `G4` is a cache, write failure does not affect inference correctness.

### Read Path

When a worker wants to reuse blocks from `G4`:

1. Worker derives candidate `sequence_hash` values locally from token blocks.
2. Worker computes the owner for each hash.
3. Worker sends `query_blocks()` to the owner.
4. For hits, worker sends `fetch_blocks()`.
5. The storage agent reads the block from NVMe RAID into pinned host memory.
6. The storage agent transfers the payload using NIXL/UCX.
7. Worker writes fetched payloads into host or device staging and onboards them into local KVBM pools.
8. On miss or transfer failure, the worker recomputes the block locally.

### Event Plane Usage

The event plane is not required for correctness in this design.

It may still be used for:

- metrics
- observability
- warming a local metadata cache
- offline reconciliation or debugging

But the primary lookup path is direct worker-to-owner query.

## Failure Model

### Storage-Agent Unavailability

If the owning storage agent is unavailable:

- `query_blocks()` fails or times out
- worker treats the block as a cache miss
- worker recomputes locally

No retry to alternate owners is needed because the first version uses a single owner and no redundancy.

### Transfer Failure

If `fetch_blocks()` fails in transit:

- worker discards the failed transfer
- worker treats the block as a cache miss
- worker recomputes locally

The system should not mark the block as successfully fetched unless the full payload length and checksum are validated.

### Write Failure

If `put_blocks()` fails:

- the block remains absent from `G4`
- no repair is attempted
- later reads fall back to recompute

## Metadata Store Choice

The first version does not require a distributed metadata database.

Recommended choices:

- in-memory `HashMap` for initial bring-up
- local embedded KV store later if restart persistence is needed

Examples of acceptable local metadata stores:

- `HashMap`
- `hashbrown`
- `sled`
- `SQLite`
- `RocksDB`
- `LMDB`

The metadata store is local to each storage agent because ownership is deterministic.

## Why Not Use a Distributed DB as the Payload Store

The first version should avoid storing actual KV payload bytes in a database because that typically introduces:

- write amplification
- compaction pressure
- worse large-blob handling
- unnecessary transactional overhead

Payload bytes should live on local NVMe-backed files as immutable blobs. The metadata database should only index them.

## Integration Plan

### Phase 1: Design and Interface

- Define the `G4` storage-agent API
- Define worker-to-agent message types
- Define hashing and ownership rules
- Define local disk layout and metadata schema
- Define the pinned-host transfer path and checksum validation rules

### Phase 2: Storage Agent Bring-Up

- Register storage-agent endpoints in discovery
- Add local metadata index
- Add `query_blocks()`, `fetch_blocks()`, and `put_blocks()`
- Persist block payloads to NVMe RAID-backed local storage
- Add pinned-host buffer management for fetch responses

### Phase 3: Worker Integration

- Derive `sequence_hash` locally in the worker-side path
- Add owner routing based on discovery membership
- Add direct query/fetch before local recompute
- Onboard fetched blocks into local KVBM tiers

### Phase 4: Policy and Observability

- Decide when blocks should be written to `G4`
- Add metrics for query hit rate, fetch latency, put latency, and transfer failures
- Optionally subscribe to event-plane updates as a secondary metadata source

## Open Questions

- Should `put_blocks()` happen immediately on block registration, or only after an offload threshold is reached?
- Should workers always fetch into host pinned memory first, or should the interface allow later direct device-target transfer?
- How should membership churn be handled during long fetches: strict epoch check, or best-effort with retry?
- Should `G4` materialization be tied to KVBM offload policy, or managed by a separate backend policy layer?

## Future Work

- Add redundancy and alternate-owner reads
- Add optional asynchronous write replication
- Add prefix-aware lookup if exact-block probing proves too expensive
- Add local block compaction and disk-space-aware eviction
- Add direct device-target remote transfer when transport and capability checks are mature enough

## References

- [KVBM Design](./kvbm-design.md)
- [Discovery Plane](./discovery-plane.md)
- [Distributed Runtime](./distributed-runtime.md)
- [TensorRT-LLM KV Cache Transfer](../backends/trtllm/trtllm-kv-cache-transfer.md)
