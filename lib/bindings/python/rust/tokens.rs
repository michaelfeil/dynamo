// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared token-id extraction for the Python bindings.
//!
//! Token-id arguments on the routing entry points are typed as `&Bound<'_,
//! PyAny>` rather than a concrete `Vec<u32>` so callers can hand us a NumPy
//! array (the buffer the tokenizer already produced) instead of a Python
//! `list[int]`. Materializing `Encoding.ids` into a list costs one Python
//! `int` object per token -- around 29 ms for a 1M-token prompt, and at 200k
//! tokens the list materialization is ~2.7x the cost of the tokenization
//! itself -- so the frontend wants to keep prompt tokens in a `uint32` array
//! end to end.
//!
//! The extraction order below mirrors `extract_list_or_numpy_u32` in the
//! `llm-runtime-metrics` Python bindings (`bindings/python/rust/lib.rs`) so
//! the two extensions accept exactly the same inputs and raise the same
//! message on a bad one.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Extract token ids from a NumPy `uint32`/`int64` array or a Python
/// sequence of ints.
///
/// A contiguous `uint32` array is the fast path: the values are copied
/// straight out of the buffer with no Python object churn. `int64` arrays are
/// range-checked per element. Anything else falls back to the pre-existing
/// `Vec<u32>` extraction, so `list[int]` behaves exactly as it did before.
pub(crate) fn extract_list_or_numpy_u32(token_ids: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    if let Ok(arr) = token_ids.extract::<PyReadonlyArray1<'_, u32>>() {
        return Ok(match arr.as_slice() {
            Ok(values) => values.to_vec(),
            Err(_) => arr.as_array().iter().copied().collect(),
        });
    }

    if let Ok(arr) = token_ids.extract::<PyReadonlyArray1<'_, i64>>() {
        let values = arr.as_array();
        let mut out = Vec::with_capacity(values.len());
        for &v in &values {
            let u = u32::try_from(v)
                .map_err(|_| PyValueError::new_err("numpy int64 values must fit in u32"))?;
            out.push(u);
        }
        return Ok(out);
    }

    if let Ok(ids) = token_ids.extract::<Vec<u32>>() {
        return Ok(ids);
    }
    Err(PyValueError::new_err(
        "token_ids must be list[int] or numpy.ndarray (uint32/int64)",
    ))
}
