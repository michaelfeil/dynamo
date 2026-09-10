// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resolve a reader once at construction; take a fresh immutable snapshot when using it.
//! File loading never changes state in a consumer crate.

mod config;
mod logging;
mod reader;
mod registry;

pub use config::{
    B10RoutingConfig, CoordinatorAffinityConfig, GenerationCoordinatorConfig, LLMRuntimeConfig,
    UnifiedConfig, sanitize_router_temperature,
};
pub use logging::set_log_no_changes;
pub use reader::ConfigReader;
pub use registry::{
    FileSource, ReaderRegistry, current_reader, process_reader, reader_for, try_current_reader,
    with_reader,
};
