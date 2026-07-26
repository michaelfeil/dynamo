// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::{Duration, Instant};

use dynamo_kv_router::protocols::{BlockExtraInfo, RouterResponse};
use parking_lot::Mutex;
use rand::Rng;

const TTL: Duration = Duration::from_millis(500);
const TTL_JITTER: Duration = Duration::from_millis(100);
const MAX_TOKENS: usize = 32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct B10PotentialLoadsCacheKey {
    tokens: Vec<u32>,
    block_mm_infos: Option<Vec<Option<BlockExtraInfo>>>,
}

#[derive(Clone, Debug)]
struct B10PotentialLoadsCacheEntry {
    key: B10PotentialLoadsCacheKey,
    expires_at: Instant,
    response: RouterResponse,
}

#[derive(Debug, Default)]
pub(super) struct B10PotentialLoadsCache {
    entry: Mutex<Option<B10PotentialLoadsCacheEntry>>,
}

impl B10PotentialLoadsCache {
    pub(super) fn key(
        tokens: &[u32],
        block_mm_infos: &Option<Vec<Option<BlockExtraInfo>>>,
        allow_short_caching: bool,
    ) -> Option<B10PotentialLoadsCacheKey> {
        (allow_short_caching && tokens.len() < MAX_TOKENS).then(|| B10PotentialLoadsCacheKey {
            tokens: tokens.to_vec(),
            block_mm_infos: block_mm_infos.clone(),
        })
    }

    pub(super) fn get(
        &self,
        key: &B10PotentialLoadsCacheKey,
        now: Instant,
    ) -> Option<RouterResponse> {
        let mut entry = self.entry.lock();
        let current = entry.as_ref()?;
        if current.expires_at > now && current.key == *key {
            return Some(current.response.clone());
        }
        if current.expires_at <= now {
            *entry = None;
        }
        None
    }

    pub(super) fn put(
        &self,
        key: B10PotentialLoadsCacheKey,
        response: RouterResponse,
        now: Instant,
    ) {
        let ttl = ttl_with_jitter();
        let mut entry = self.entry.lock();
        *entry = Some(B10PotentialLoadsCacheEntry {
            key,
            response,
            expires_at: now + ttl,
        });
    }
}

fn ttl_with_jitter() -> Duration {
    // expectation duration = TTL + 0.05 * TTL_JITTER + TTL_JITTER / 2
    let mut rng = rand::rng();
    // avoid thundering herd on cache expiration, by making most caches expire slightly earlier.
    let bias = if rng.random_bool(0.05) {
        TTL_JITTER
    } else {
        Duration::ZERO
    };
    TTL + bias + Duration::from_millis(rng.random_range(1..=TTL_JITTER.as_millis() as u64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::{protocols::BlockMmObjectInfo, scheduling::PotentialLoad};

    #[test]
    fn cache_key_requires_opt_in_and_short_tokens() {
        let short_tokens: Vec<u32> = (0..31).collect();
        let long_tokens: Vec<u32> = (0..32).collect();
        let block_mm_infos = Some(vec![Some(BlockExtraInfo {
            mm_objects: vec![BlockMmObjectInfo {
                mm_hash: 7,
                offsets: vec![(0, 1)],
            }],
        })]);

        assert!(B10PotentialLoadsCache::key(&short_tokens, &block_mm_infos, false).is_none());
        assert!(B10PotentialLoadsCache::key(&long_tokens, &block_mm_infos, true).is_none());

        let key = B10PotentialLoadsCache::key(&short_tokens, &block_mm_infos, true)
            .expect("short opted-in request should be cacheable");

        assert_eq!(key.tokens, short_tokens);
        assert_eq!(key.block_mm_infos, block_mm_infos);
    }

    #[test]
    fn cache_returns_previous_response_until_expiry() {
        let cache = B10PotentialLoadsCache::default();
        let key = B10PotentialLoadsCacheKey {
            tokens: vec![1],
            block_mm_infos: None,
        };
        let response = RouterResponse::PotentialLoads {
            loads: vec![PotentialLoad {
                worker_id: 1,
                dp_rank: 0,
                potential_prefill_tokens: 11,
                potential_decode_blocks: 2,
                active_requests: 3,
            }],
            pending_count: 7,
            pending_isl_tokens: 13,
        };
        let now = Instant::now();

        cache.put(key.clone(), response, now);

        let cached = cache
            .get(&key, now + Duration::from_millis(100))
            .expect("fresh cache entry should be returned");
        assert!(matches!(
            cached,
            RouterResponse::PotentialLoads {
                loads,
                pending_count: 7,
                pending_isl_tokens: 13,
            } if loads.len() == 1
                && loads[0].worker_id == 1
                && loads[0].potential_prefill_tokens == 11
        ));

        assert!(
            cache
                .get(
                    &B10PotentialLoadsCacheKey {
                        tokens: vec![2],
                        block_mm_infos: None,
                    },
                    now + Duration::from_millis(100),
                )
                .is_none()
        );
        assert!(
            cache
                .get(&key, now + TTL + 2 * TTL_JITTER + Duration::from_millis(1))
                .is_none()
        );
    }
}
