// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Baseten-specific cache-event consolidation.
//!
//! Collapses engine-device and KVBM device/host/disk residency into one
//! router-visible logical residency while preserving ordered trie updates.

use std::collections::{HashMap, HashSet};

use super::tracker::{
    CacheStatusTracker, ConsolidatedEvent, EventSource, RemoveEventInput, SequenceHash,
    StorageTier, StoreEventInput, compute_local_block_hash_with_lora, compute_sequence_hash,
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ResidencyMask(u8);

impl ResidencyMask {
    // Engine and KVBM device events are independent producers. Keep their
    // presence separate so either producer can remove its registration without
    // hiding a block that is still registered by the other.
    const ENGINE_DEVICE: Self = Self(0b0001);
    const KVBM_DEVICE: Self = Self(0b0010);
    const KVBM_HOST: Self = Self(0b0100);
    const KVBM_DISK: Self = Self(0b1000);

    fn for_event(source: EventSource, tier: Option<StorageTier>) -> Self {
        if source != EventSource::Kvbm {
            return Self::ENGINE_DEVICE;
        }

        match tier.unwrap_or(StorageTier::HostPinned) {
            StorageTier::Device => Self::KVBM_DEVICE,
            StorageTier::HostPinned => Self::KVBM_HOST,
            StorageTier::Disk => Self::KVBM_DISK,
        }
    }

    fn insert(&mut self, other: Self) -> bool {
        let previous = self.0;
        self.0 |= other.0;
        previous != self.0
    }

    fn remove(&mut self, other: Self) -> bool {
        let previous = self.0;
        self.0 &= !other.0;
        previous != self.0
    }

    fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// Canonical cache status used by Baseten deployments.
///
/// Engine device, KVBM device, host, and disk are private worker-side residency
/// details. The router sees a single synthetic Device residency for a block
/// while any bit remains set.
#[derive(Debug, Clone, Default)]
struct BasetenBlockMetadata {
    residency: ResidencyMask,
    parent_sequence_hash: Option<SequenceHash>,
    canonical_store: Option<ConsolidatedEvent>,
    advertised: bool,
}

/// Deduplicates tier transitions into one logical router residency.
///
/// Unlike [`DedupCacheStatusTracker`], this tracker deliberately keys source
/// state by producer and storage tier. This allows either device producer's
/// removal to be suppressed while the same canonical block remains registered
/// by the other producer or in KVBM's host/disk cache.
#[derive(Debug, Default)]
pub struct BasetenDedupCacheStatusTracker {
    blocks: HashMap<SequenceHash, BasetenBlockMetadata>,
    hash_mapping: HashMap<String, SequenceHash>,
    /// Stores are normally parent-first, but retain a child if either event
    /// source presents it before its canonical parent.
    pending_children: HashMap<String, Vec<StoreEventInput>>,
    children: HashMap<SequenceHash, HashSet<SequenceHash>>,
    event_queue: Vec<ConsolidatedEvent>,
}

impl BasetenDedupCacheStatusTracker {
    pub fn new() -> Self {
        Self::default()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn handle_store(
        &mut self,
        block_hash: String,
        source: EventSource,
        token_ids: Vec<u32>,
        parent_hash: Option<String>,
        block_size: usize,
        lora_name: Option<String>,
        tier: Option<StorageTier>,
        data_parallel_rank: Option<i32>,
    ) -> bool {
        CacheStatusTracker::handle_store(
            self,
            StoreEventInput {
                block_hash,
                source,
                token_ids,
                parent_hash,
                block_size,
                lora_name,
                tier,
                data_parallel_rank,
            },
        )
    }

    pub fn handle_remove(
        &mut self,
        block_hash: &str,
        source: EventSource,
        tier: Option<StorageTier>,
    ) -> bool {
        CacheStatusTracker::handle_remove(
            self,
            RemoveEventInput {
                block_hash: block_hash.to_string(),
                source,
                tier,
            },
        )
    }

    fn resolve_parent(&self, parent_hash: Option<&str>) -> Option<SequenceHash> {
        parent_hash.and_then(|hash| self.hash_mapping.get(hash).copied())
    }

    fn sequence_hash_for_store(
        &self,
        token_ids: &[u32],
        parent_sequence_hash: Option<SequenceHash>,
        lora_name: Option<&str>,
    ) -> SequenceHash {
        compute_sequence_hash(
            parent_sequence_hash,
            compute_local_block_hash_with_lora(token_ids, lora_name),
        )
    }

    fn store_identity_differs(
        current: Option<&ConsolidatedEvent>,
        replacement: &ConsolidatedEvent,
    ) -> bool {
        match (current, replacement) {
            (
                Some(ConsolidatedEvent::Store {
                    block_hash: current_hash,
                    token_ids: current_tokens,
                    block_size: current_block_size,
                    lora_name: current_lora,
                    ..
                }),
                ConsolidatedEvent::Store {
                    block_hash: replacement_hash,
                    token_ids: replacement_tokens,
                    block_size: replacement_block_size,
                    lora_name: replacement_lora,
                    ..
                },
            ) => {
                current_hash != replacement_hash
                    || current_tokens != replacement_tokens
                    || current_block_size != replacement_block_size
                    || current_lora != replacement_lora
            }
            (None, _) => true,
            _ => true,
        }
    }

    fn should_advertise(&self, sequence_hash: SequenceHash) -> bool {
        let Some(block) = self.blocks.get(&sequence_hash) else {
            return false;
        };

        !block.residency.is_empty()
            && block.canonical_store.is_some()
            && block.parent_sequence_hash.is_none_or(|parent| {
                self.blocks
                    .get(&parent)
                    .is_some_and(|metadata| metadata.advertised)
            })
    }

    fn queue_store(&mut self, sequence_hash: SequenceHash) {
        let parent_block_hash = self
            .blocks
            .get(&sequence_hash)
            .and_then(|block| block.parent_sequence_hash)
            .and_then(|parent| self.blocks.get(&parent))
            .and_then(|parent| parent.canonical_store.as_ref())
            .and_then(|event| match event {
                ConsolidatedEvent::Store { block_hash, .. } => Some(block_hash.clone()),
                _ => None,
            });

        let Some(mut event) = self
            .blocks
            .get(&sequence_hash)
            .and_then(|block| block.canonical_store.clone())
        else {
            return;
        };

        if let ConsolidatedEvent::Store {
            parent_hash,
            source,
            tier,
            ..
        } = &mut event
        {
            *parent_hash = parent_block_hash;
            *source = "baseten_dedup".to_string();
            *tier = Some(StorageTier::Device);
        }
        self.event_queue.push(event);
    }

    fn queue_remove(&mut self, sequence_hash: SequenceHash) {
        let block_hash = self
            .blocks
            .get(&sequence_hash)
            .and_then(|block| block.canonical_store.as_ref())
            .and_then(|event| match event {
                ConsolidatedEvent::Store { block_hash, .. } => Some(block_hash.clone()),
                _ => None,
            });

        if let Some(block_hash) = block_hash {
            self.event_queue.push(ConsolidatedEvent::Remove {
                block_hash,
                source: "baseten_dedup".to_string(),
                tier: Some(StorageTier::Device),
            });
        }
    }

    fn advertise_subtree(&mut self, sequence_hash: SequenceHash) {
        if !self.should_advertise(sequence_hash) {
            return;
        }

        let already_advertised = self
            .blocks
            .get(&sequence_hash)
            .is_some_and(|block| block.advertised);
        if !already_advertised {
            self.queue_store(sequence_hash);
            if let Some(block) = self.blocks.get_mut(&sequence_hash) {
                block.advertised = true;
            }
        }

        let children = self
            .children
            .get(&sequence_hash)
            .cloned()
            .unwrap_or_default();
        for child in children {
            self.advertise_subtree(child);
        }
    }

    fn hide_subtree(&mut self, sequence_hash: SequenceHash) {
        let children = self
            .children
            .get(&sequence_hash)
            .cloned()
            .unwrap_or_default();
        for child in children {
            self.hide_subtree(child);
        }

        let advertised = self
            .blocks
            .get(&sequence_hash)
            .is_some_and(|block| block.advertised);
        if advertised {
            self.queue_remove(sequence_hash);
            if let Some(block) = self.blocks.get_mut(&sequence_hash) {
                block.advertised = false;
            }
        }
    }

    fn refresh_visibility(&mut self, sequence_hash: SequenceHash) {
        if self.should_advertise(sequence_hash) {
            self.advertise_subtree(sequence_hash);
        } else {
            self.hide_subtree(sequence_hash);
        }
    }

    fn prune_absent_leaf(&mut self, mut sequence_hash: SequenceHash) {
        loop {
            let can_prune = self.blocks.get(&sequence_hash).is_some_and(|block| {
                block.residency.is_empty()
                    && !block.advertised
                    && self
                        .children
                        .get(&sequence_hash)
                        .is_none_or(HashSet::is_empty)
            });
            if !can_prune {
                return;
            }

            let parent = self
                .blocks
                .remove(&sequence_hash)
                .and_then(|block| block.parent_sequence_hash);
            self.hash_mapping
                .retain(|_, mapped_sequence_hash| *mapped_sequence_hash != sequence_hash);
            self.children.remove(&sequence_hash);

            let Some(parent) = parent else {
                return;
            };
            if let Some(siblings) = self.children.get_mut(&parent) {
                siblings.remove(&sequence_hash);
            }
            sequence_hash = parent;
        }
    }

    #[cfg(test)]
    pub(super) fn residency(&self, block_hash: &str) -> Option<u8> {
        let sequence_hash = self.hash_mapping.get(block_hash)?;
        self.blocks
            .get(sequence_hash)
            .map(|block| block.residency.0)
    }

    #[cfg(test)]
    pub(super) fn metadata_is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

impl CacheStatusTracker for BasetenDedupCacheStatusTracker {
    fn handle_store(&mut self, event: StoreEventInput) -> bool {
        if let Some(parent_hash) = event.parent_hash.as_ref()
            && !self.hash_mapping.contains_key(parent_hash)
        {
            tracing::debug!(
                block_hash = event.block_hash,
                parent_hash,
                source = ?event.source,
                "baseten_dedup deferred child until its parent arrives"
            );
            self.pending_children
                .entry(parent_hash.clone())
                .or_default()
                .push(event);
            return false;
        }

        let StoreEventInput {
            block_hash,
            source,
            token_ids,
            parent_hash,
            block_size,
            lora_name,
            tier,
            data_parallel_rank: _,
        } = event;

        let residency = ResidencyMask::for_event(source, tier);
        let previous_events = self.event_queue.len();

        let parent_sequence_hash = self.resolve_parent(parent_hash.as_deref());
        // KVBM normally publishes the engine-facing external hash. Reuse an
        // existing alias when present so a KVBM-first record can be enriched by
        // the later engine event even when LoRA identity was unavailable to
        // KVBM's event manager.
        let sequence_hash = self
            .hash_mapping
            .get(&block_hash)
            .copied()
            .unwrap_or_else(|| {
                self.sequence_hash_for_store(&token_ids, parent_sequence_hash, lora_name.as_deref())
            });

        let replacement_store = ConsolidatedEvent::Store {
            block_hash: block_hash.clone(),
            parent_hash,
            token_ids,
            block_size,
            lora_name,
            source: source.to_str().to_string(),
            tier: Some(StorageTier::Device),
        };
        let replace_canonical = self
            .blocks
            .get(&sequence_hash)
            .is_none_or(|block| block.canonical_store.is_none())
            || source != EventSource::Kvbm;
        let migrate_advertised_identity = replace_canonical
            && self.blocks.get(&sequence_hash).is_some_and(|block| {
                block.advertised
                    && Self::store_identity_differs(
                        block.canonical_store.as_ref(),
                        &replacement_store,
                    )
            });

        // Withdraw the old identity (and its children) before changing any
        // router-visible hash or LoRA metadata. refresh_visibility below will
        // then re-advertise parent-first using the replacement identity.
        if migrate_advertised_identity {
            self.hide_subtree(sequence_hash);
        }

        self.hash_mapping.insert(block_hash.clone(), sequence_hash);
        let block = self.blocks.entry(sequence_hash).or_default();
        block.residency.insert(residency);

        if block.parent_sequence_hash != parent_sequence_hash {
            if let Some(previous_parent) = block.parent_sequence_hash
                && let Some(children) = self.children.get_mut(&previous_parent)
            {
                children.remove(&sequence_hash);
            }
            block.parent_sequence_hash = parent_sequence_hash;
        }

        // A KVBM store carries enough block and lineage metadata to recreate a
        // base-model record after the engine copy has already been removed.
        // Prefer engine metadata when both producers are present because it may
        // add identity fields such as the LoRA name.
        if replace_canonical {
            block.canonical_store = Some(replacement_store);
        }

        if let Some(parent) = parent_sequence_hash {
            self.children
                .entry(parent)
                .or_default()
                .insert(sequence_hash);
        }

        self.refresh_visibility(sequence_hash);

        let pending_children = self
            .pending_children
            .remove(&block_hash)
            .unwrap_or_default();
        for child in pending_children {
            CacheStatusTracker::handle_store(self, child);
        }

        tracing::debug!(
            sequence_hash,
            block_hash,
            source = ?source,
            tier = ?tier,
            residency = self.blocks[&sequence_hash].residency.0,
            advertised = self.blocks[&sequence_hash].advertised,
            "baseten_dedup processed cache store"
        );

        self.event_queue.len() != previous_events
    }

    fn handle_remove(&mut self, event: RemoveEventInput) -> bool {
        let RemoveEventInput {
            block_hash,
            source,
            tier,
        } = event;
        if !self.hash_mapping.contains_key(&block_hash) {
            let mut removed_pending = false;
            self.pending_children.retain(|_, children| {
                let previous_len = children.len();
                children.retain(|child| {
                    child.block_hash != block_hash || child.source != source || child.tier != tier
                });
                removed_pending |= children.len() != previous_len;
                !children.is_empty()
            });
            if removed_pending {
                tracing::debug!(
                    block_hash,
                    source = ?source,
                    "baseten_dedup cancelled deferred child on removal"
                );
                return false;
            }
        }

        let sequence_hash = self.hash_mapping.get(&block_hash).copied();
        let Some(sequence_hash) = sequence_hash else {
            tracing::warn!(
                block_hash,
                source = ?source,
                tier = ?tier,
                "baseten_dedup ignored removal for unknown block"
            );
            return false;
        };

        let residency = ResidencyMask::for_event(source, tier);
        let Some(block) = self.blocks.get_mut(&sequence_hash) else {
            return false;
        };
        if !block.residency.remove(residency) {
            tracing::debug!(
                sequence_hash,
                block_hash,
                source = ?source,
                tier = ?tier,
                "baseten_dedup ignored duplicate cache removal"
            );
            return false;
        }

        let previous_events = self.event_queue.len();
        self.refresh_visibility(sequence_hash);

        tracing::debug!(
            sequence_hash,
            block_hash,
            source = ?source,
            tier = ?tier,
            residency = self.blocks[&sequence_hash].residency.0,
            advertised = self.blocks[&sequence_hash].advertised,
            "baseten_dedup processed cache removal"
        );

        let emitted = self.event_queue.len() != previous_events;
        self.prune_absent_leaf(sequence_hash);
        emitted
    }

    fn handle_clear_all(&mut self) {
        self.blocks.clear();
        self.hash_mapping.clear();
        self.pending_children.clear();
        self.children.clear();
        self.event_queue.clear();
        self.event_queue.push(ConsolidatedEvent::ClearAll);
    }

    fn drain_events(&mut self) -> Vec<ConsolidatedEvent> {
        std::mem::take(&mut self.event_queue)
    }

    fn num_blocks(&self) -> usize {
        self.blocks
            .values()
            .filter(|block| !block.residency.is_empty())
            .count()
    }
}
