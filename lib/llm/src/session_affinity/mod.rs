// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod coordinator;
mod replica_sync;

use std::time::Duration;

use dynamo_runtime::{component::Client, pipeline::Error};

pub(crate) use coordinator::{AffinityAcquire, AffinityLease};
pub use coordinator::{AffinityCoordinator, AffinityTarget};

pub const MAX_SESSION_AFFINITY_TTL_SECS: u64 = 31_536_000;
pub const MAX_SESSION_AFFINITY_ENTRIES: usize = 262_144;
pub const MAX_SESSION_AFFINITY_ID_BYTES: usize = 256;

pub(crate) async fn create_affinity_coordinator(
    ttl: Option<Duration>,
    client: Client,
) -> Result<Option<AffinityCoordinator>, Error> {
    let Some(ttl) = ttl else {
        return Ok(None);
    };
    let coordinator = AffinityCoordinator::new(ttl)?;
    coordinator.enable_replica_sync(client).await?;
    Ok(Some(coordinator))
}

#[cfg(test)]
mod tests;
