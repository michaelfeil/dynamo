// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-wide configuration-loader logging controls.

use std::sync::LazyLock;
use std::sync::atomic::{AtomicBool, Ordering};

static LOG_NO_CHANGES: AtomicBool = AtomicBool::new(false);
static WARNINGS_DISABLED: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("B10_CONFIGMAP_DISABLE_WARNING")
        .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
});

pub fn set_log_no_changes(enabled: bool) {
    LOG_NO_CHANGES.store(enabled, Ordering::Relaxed);
}

pub(super) fn log_no_changes() -> bool {
    LOG_NO_CHANGES.load(Ordering::Relaxed)
}

pub(super) fn warnings_disabled() -> bool {
    *WARNINGS_DISABLED
}
