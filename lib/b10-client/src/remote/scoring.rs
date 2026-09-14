// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use baseten_configmap::bid_weight_ratio;

use crate::protocol::BidResponseV1;

pub(super) trait BidScorer {
    fn score(&self, bid: &BidResponseV1) -> u128;
}

pub(super) struct WeightedBidScorer {
    prefill_weight: u128,
    decode_weight: u128,
    affinity_weight: u128,
    unmatched_weight: u128,
}

impl WeightedBidScorer {
    pub(super) fn new(decode_token_weight: f64, affinity_multiplier: f64) -> Self {
        let (decode_weight, prefill_weight) = bid_weight_ratio(decode_token_weight);
        let (affinity_weight, unmatched_weight) = bid_weight_ratio(affinity_multiplier);
        Self {
            prefill_weight,
            decode_weight,
            affinity_weight,
            unmatched_weight,
        }
    }
}

impl BidScorer for WeightedBidScorer {
    fn score(&self, bid: &BidResponseV1) -> u128 {
        (self.prefill_weight * u128::from(bid.prefill_tokens)
            + self.decode_weight * u128::from(bid.decode_tokens))
            * if bid.affinity {
                self.affinity_weight
            } else {
                self.unmatched_weight
            }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_scores_match_original_u128_formula() {
        let scorer = WeightedBidScorer::new(0.1, 0.5);
        for prefill_tokens in [0, 1, (1 << 53), (1 << 53) + 1, u64::MAX] {
            for decode_tokens in [0, 1, (1 << 53), (1 << 53) + 1, u64::MAX] {
                for affinity in [false, true] {
                    let bid = BidResponseV1 {
                        affinity,
                        prefill_tokens,
                        decode_tokens,
                    };
                    let original = (if affinity { 1 } else { 2 })
                        * (10 * u128::from(prefill_tokens) + u128::from(decode_tokens));
                    assert_eq!(scorer.score(&bid), original);
                }
            }
        }
        assert_eq!(
            scorer.score(&BidResponseV1 {
                affinity: false,
                prefill_tokens: 0,
                decode_tokens: 6
            }),
            scorer.score(&BidResponseV1 {
                affinity: true,
                prefill_tokens: 1,
                decode_tokens: 2
            }),
        );
    }

    #[test]
    fn custom_and_zero_weights_use_integer_ratios() {
        for (decode, multiplier, prefill_tokens, decode_tokens, affinity, expected) in [
            (0.2, 0.8, 10, 20, false, 350),
            (0.2, 0.8, 10, 20, true, 280),
            (0.0, 0.5, 10, u64::MAX, false, 20),
            (0.0, 0.5, 10, u64::MAX, true, 10),
            (0.2, 0.0, 10, 20, true, 0),
            (0.0003, 0.125, 1, 1, true, 10003),
            (0.0003, 0.125, 1, 1, false, 80024),
            (
                429496.7295,
                429496.7295,
                u64::MAX,
                u64::MAX,
                false,
                u128::from(u64::MAX) * (2000 + 858_993_459) * 2000,
            ),
            (
                429496.7295,
                429496.7295,
                u64::MAX,
                u64::MAX,
                true,
                u128::from(u64::MAX) * (2000 + 858_993_459) * 858_993_459,
            ),
        ] {
            let scorer = WeightedBidScorer::new(decode, multiplier);
            assert_eq!(
                scorer.score(&BidResponseV1 {
                    affinity,
                    prefill_tokens,
                    decode_tokens
                }),
                expected
            );
        }
    }
}
