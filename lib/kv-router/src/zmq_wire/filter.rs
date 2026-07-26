// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use rustc_hash::FxHashMap;
use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;

use crate::protocols::{BlockExtraInfo, CacheGroupClass};

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(super) enum KvCacheEventTrailingField {
    GroupIdx(u32),
    KvCacheSpecKind(KvCacheSpecKind),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(super) enum BlockStoredTrailingField {
    Common(KvCacheEventTrailingField),
    BlockMmInfos(Vec<Option<BlockExtraInfo>>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvCacheSpecKind {
    FullAttention,
    MlaAttention,
    SlidingWindow,
    SlidingWindowMla,
    Mamba,
    ChunkedLocalAttention,
    SinkFullAttention,
    EncoderOnlyAttention,
    CrossAttention,
    Unknown,
}

impl KvCacheSpecKind {
    pub(crate) fn from_wire(value: &str) -> Self {
        match value {
            "full_attention" => Self::FullAttention,
            "mla_attention" => Self::MlaAttention,
            "sliding_window" => Self::SlidingWindow,
            "sliding_window_mla" => Self::SlidingWindowMla,
            "mamba" => Self::Mamba,
            "chunked_local_attention" => Self::ChunkedLocalAttention,
            "sink_full_attention" => Self::SinkFullAttention,
            "encoder_only_attention" => Self::EncoderOnlyAttention,
            "cross_attention" => Self::CrossAttention,
            unknown => {
                tracing::warn!(
                    kv_cache_spec_kind = unknown,
                    "Unknown KV cache spec kind; treating KV event as non-main"
                );
                Self::Unknown
            }
        }
    }

    pub(crate) fn as_wire(self) -> &'static str {
        match self {
            Self::FullAttention => "full_attention",
            Self::MlaAttention => "mla_attention",
            Self::SlidingWindow => "sliding_window",
            Self::SlidingWindowMla => "sliding_window_mla",
            Self::Mamba => "mamba",
            Self::ChunkedLocalAttention => "chunked_local_attention",
            Self::SinkFullAttention => "sink_full_attention",
            Self::EncoderOnlyAttention => "encoder_only_attention",
            Self::CrossAttention => "cross_attention",
            Self::Unknown => "unknown",
        }
    }
}

/// How the router indexes KV events for one [`KvCacheSpecKind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexStrategy {
    /// Index into the attention (radix) block index.
    Attention,
    /// Index as recurrent-state checkpoints. No kind maps to this yet;
    /// admitting a recurrent kind is a registry change, not a filter edit.
    Recurrent,
    /// Drop the event before indexing.
    Drop,
}

/// Maps each [`KvCacheSpecKind`] to an [`IndexStrategy`].
///
/// The default map reproduces the historical main-attention allow-list:
/// `FullAttention`, `MlaAttention`, and `SinkFullAttention` index as
/// attention; every other kind (including `Unknown` and unregistered kinds)
/// drops. Adding or removing an indexable kind is a registration change via
/// [`IndexStrategyRegistry::with_strategy`].
#[derive(Debug, Clone)]
pub struct IndexStrategyRegistry {
    strategies: FxHashMap<KvCacheSpecKind, IndexStrategy>,
}

impl Default for IndexStrategyRegistry {
    fn default() -> Self {
        let mut strategies = FxHashMap::default();
        for kind in [
            KvCacheSpecKind::FullAttention,
            KvCacheSpecKind::MlaAttention,
            KvCacheSpecKind::SinkFullAttention,
        ] {
            strategies.insert(kind, IndexStrategy::Attention);
        }
        Self { strategies }
    }
}

impl IndexStrategyRegistry {
    /// The strategy for `kind`; unregistered kinds drop.
    pub fn strategy(&self, kind: KvCacheSpecKind) -> IndexStrategy {
        self.strategies
            .get(&kind)
            .copied()
            .unwrap_or(IndexStrategy::Drop)
    }

    /// Register or override the strategy for one kind.
    pub fn with_strategy(mut self, kind: KvCacheSpecKind, strategy: IndexStrategy) -> Self {
        self.strategies.insert(kind, strategy);
        self
    }

    /// The cache-group class stamped on events of `kind`, `None` for dropped
    /// kinds.
    pub(crate) fn cache_group(&self, kind: KvCacheSpecKind) -> Option<CacheGroupClass> {
        match self.strategy(kind) {
            IndexStrategy::Attention => Some(CacheGroupClass::Attention),
            IndexStrategy::Recurrent => Some(CacheGroupClass::Recurrent),
            IndexStrategy::Drop => None,
        }
    }
}

impl Serialize for KvCacheSpecKind {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_wire())
    }
}

impl<'de> Deserialize<'de> for KvCacheSpecKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(Self::from_wire(&value))
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct KvCacheEventMetadata {
    pub(crate) group_idx: Option<u32>,
    pub(crate) kv_cache_spec_kind: Option<KvCacheSpecKind>,
    pub(crate) kv_cache_spec_sliding_window: Option<u32>,
    /// Event-declared block size in tokens; only `BlockStored` carries it.
    pub(crate) block_size: Option<u32>,
}

impl KvCacheEventMetadata {
    pub(super) fn record_trailing(&mut self, trailing: KvCacheEventTrailingField) {
        match trailing {
            KvCacheEventTrailingField::GroupIdx(value) => {
                if self.group_idx.is_none() {
                    self.group_idx = Some(value);
                } else if self.kv_cache_spec_sliding_window.is_none() {
                    self.kv_cache_spec_sliding_window = Some(value);
                }
            }
            KvCacheEventTrailingField::KvCacheSpecKind(kind) => {
                self.kv_cache_spec_kind = Some(kind);
            }
        }
    }
}
