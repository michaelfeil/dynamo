// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use anyhow::Result;
use dynamo_kv_router::{
    protocols::{
        BlockExtraInfo, BlockHashOptions, RouterResponse, RoutingConstraints, WorkerId,
        compute_block_hash_for_seq,
    },
    scheduling::SchedulingRequest,
    selector::WorkerSelector,
};

use super::{
    KvRouter, map_scheduler_error, query_tiered_matches,
    scheduler_inputs::tier_overlap_blocks_from_tiered_matches,
};
use crate::local_model::runtime_config::ModelRuntimeConfig;

impl<Sel> KvRouter<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + Sync + 'static,
{
    /// Select a candidate and return its block costs without admission or booking.
    pub(super) async fn bid(
        &self,
        tokens: &[u32],
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
        routing_constraints: RoutingConstraints,
    ) -> Result<RouterResponse> {
        anyhow::ensure!(!tokens.is_empty(), "cannot bid on an empty token sequence");
        let hash_options = BlockHashOptions {
            block_mm_infos,
            lora_name: None,
            is_eagle: Some(self.is_eagle),
        };
        let block_hashes = compute_block_hash_for_seq(tokens, self.block_size, hash_options);
        let token_seq = self.kv_router_config.compute_seq_hashes_for_tracking(
            tokens,
            self.block_size,
            None,
            hash_options,
            Some(&block_hashes),
        );
        let lookup = query_tiered_matches(
            &self.indexer,
            self.shared_cache.as_deref(),
            tokens,
            self.block_size,
            block_hashes,
            false,
        )
        .await?;
        let estimates = self.cache_hit_estimates_from_tiered_matches(&lookup.tiered_matches);
        let bid = self
            .scheduler
            .probe(SchedulingRequest {
                token_seq,
                isl_tokens: tokens.len(),
                allowed_worker_ids,
                routing_constraints,
                track_prefill_tokens: self.kv_router_config.track_prefill_tokens(None),
                tier_overlap_blocks: tier_overlap_blocks_from_tiered_matches(
                    &lookup.tiered_matches,
                ),
                effective_overlap_blocks: estimates.effective_overlap_blocks,
                effective_cached_tokens: estimates.cached_tokens,
                shared_cache_hits: lookup.shared_cache_hits,
                ..Default::default()
            })
            .await
            .map_err(map_scheduler_error)?;
        Ok(RouterResponse::Bid {
            worker_id: bid.worker.worker_id,
            dp_rank: bid.worker.dp_rank,
            prefill_blocks: bid.prefill_blocks,
            decode_blocks: bid.decode_blocks,
        })
    }
}
