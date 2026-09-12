// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result, ensure};

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
    pub(super) fn new(decode_token_weight: f64, affinity_multiplier: f64) -> Result<Self> {
        let (decode_weight, prefill_weight) =
            ratio(decode_token_weight).context("invalid bid_decode_token_weight")?;
        let (affinity_weight, unmatched_weight) =
            ratio(affinity_multiplier).context("invalid bid_affinity_multiplier")?;
        ensure!(
            (prefill_weight + decode_weight)
                .checked_mul(u128::from(u64::MAX))
                .and_then(|score| score.checked_mul(affinity_weight.max(unmatched_weight)))
                .is_some(),
            "bid weights would overflow u128 scores"
        );
        Ok(Self {
            prefill_weight,
            decode_weight,
            affinity_weight,
            unmatched_weight,
        })
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

fn ratio(value: f64) -> Result<(u128, u128)> {
    const SCALE: u128 = 10_000;
    let scaled = (value * SCALE as f64).round();
    ensure!(
        value.is_finite()
            && value >= 0.0
            && scaled <= f64::from(u32::MAX)
            && scaled / SCALE as f64 == value,
        "bid weights must be between 0 and 429496.7295 with at most four decimal places"
    );
    let numerator = scaled as u128;
    let (mut a, mut b) = (numerator, SCALE);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    Ok((numerator / a, SCALE / a))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_scores_match_original_u128_formula() {
        let scorer = WeightedBidScorer::new(0.1, 0.5).unwrap();
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
        ] {
            let scorer = WeightedBidScorer::new(decode, multiplier).unwrap();
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

    #[test]
    fn invalid_precision_range_and_overflow_are_rejected() {
        for value in [-1.0, f64::NAN, f64::INFINITY, 0.12345, 429496.7296] {
            assert!(WeightedBidScorer::new(value, 0.5).is_err());
            assert!(WeightedBidScorer::new(0.1, value).is_err());
        }
        assert!(WeightedBidScorer::new(429496.7293, 429496.7293).is_err());
        let boundary = WeightedBidScorer::new(429496.7295, 429496.7295).unwrap();
        for affinity in [false, true] {
            assert!(
                boundary.score(&BidResponseV1 {
                    affinity,
                    prefill_tokens: u64::MAX,
                    decode_tokens: u64::MAX,
                }) > 0
            );
        }
    }
}
