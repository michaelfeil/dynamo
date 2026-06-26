// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod block_tracker;
pub mod multi_worker;
mod prefill_tracker;
mod prompt_membership_trie;
mod prompt_registry;
mod request_maps;
mod residency;
pub mod single;
mod topology;

use std::sync::atomic::{AtomicU64, Ordering};

pub use multi_worker::*;
pub use prefill_tracker::PrefillTokenDeltas;
pub use residency::EvictionPressure;
pub use single::*;

static PREFILL_TOKEN_DISCOUNT: AtomicU64 = AtomicU64::new(1.0f64.to_bits());
static DECODE_TOKEN_DISCOUNT: AtomicU64 = AtomicU64::new(1.0f64.to_bits());

pub fn set_token_load_discounts(prefill_discount: f64, decode_discount: f64) {
    let prefill_bits = prefill_discount.to_bits();
    let decode_bits = decode_discount.to_bits();
    let previous_prefill_bits = PREFILL_TOKEN_DISCOUNT.swap(prefill_bits, Ordering::Relaxed);
    let previous_decode_bits = DECODE_TOKEN_DISCOUNT.swap(decode_bits, Ordering::Relaxed);

    if previous_prefill_bits != prefill_bits || previous_decode_bits != decode_bits {
        tracing::info!(
            previous_prefill_discount = f64::from_bits(previous_prefill_bits),
            prefill_discount,
            previous_decode_discount = f64::from_bits(previous_decode_bits),
            decode_discount,
            "token load discounts changed"
        );
    }
}

pub(crate) fn token_load_discounts() -> (f64, f64) {
    (
        f64::from_bits(PREFILL_TOKEN_DISCOUNT.load(Ordering::Relaxed)),
        f64::from_bits(DECODE_TOKEN_DISCOUNT.load(Ordering::Relaxed)),
    )
}
