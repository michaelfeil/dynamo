// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Baseten B10 admission client.
//!
//! This crate owns the router-to-worker lifecycle independently of any language
//! binding. Rust callers use [`RouterWorkerCoordinator`]; Python bindings adapt
//! their native objects to the same request and outcome types.

mod context;
mod coordinator;
mod guard;
mod payload_copy;
mod types;

#[cfg(test)]
mod tests;

pub use context::RequestContext;
pub use coordinator::{
    JsonPushRouter, JsonRouterGuardClient, RouteAndConnectOutcome, RouterGuardClient,
    RouterWorkerCoordinator, stream_with_optional_prefill_mark,
};
pub use guard::RouterRequestGuard;
pub use types::{
    AdmittedRequestTimings, CancellationPolicy, DeniedRequest, MinReplicaAvailable,
    PotentialLoadsCheck, RouteOptions, RouterRequestNew, RouterWorkerPhase,
};

/// Worker-stream item key used for an internal readiness event that should not
/// be exposed to the caller.
pub const DROP_THIS_MESSAGE_KEY: &str = "drop_this_message";
