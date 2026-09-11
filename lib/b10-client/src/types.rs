// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::RouterGuardClient;
use anyhow::Result;
use dynamo_kv_router::protocols::{BlockExtraInfo, RouterRequest, RoutingConstraints};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

/// Typed body of a B10 `new` routing request.
#[derive(Debug, Clone, Default)]
pub struct RouterRequestNew {
    pub tokens: Vec<u32>,
    pub block_mm_infos: Option<Vec<Option<BlockExtraInfo>>>,
    pub routing_constraints: RoutingConstraints,
    pub allowed_worker_ids: Option<HashSet<u64>>,
    pub priority_jump: f64,
    pub priority_load_shed_percent: u8,
    pub do_not_queue: bool,
}

impl From<RouterRequestNew> for RouterRequest {
    fn from(req: RouterRequestNew) -> Self {
        Self::New {
            tokens: req.tokens.into(),
            block_mm_infos: req.block_mm_infos,
            routing_constraints: req.routing_constraints,
            allowed_worker_ids: req.allowed_worker_ids,
            priority_jump: req.priority_jump,
            priority_load_shed_percent: req.priority_load_shed_percent,
            do_not_queue: req.do_not_queue,
        }
    }
}

impl RouterRequestNew {
    pub fn into_routing_request_value(self) -> Result<rmpv::Value> {
        let bytes = rmp_serde::to_vec_named(&RouterRequest::from(self))?;
        Ok(rmpv::decode::read_value(&mut bytes.as_slice())?)
    }
}

/// Cancellation behavior for routing, worker setup, and response streaming.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CancellationPolicy {
    #[default]
    Cancellable,
    DetachToWorkerStreamConnected,
    FullyDetached,
    CancellableUntilWorkerThenDetach,
    DetachSetupOnly,
}

impl CancellationPolicy {
    pub fn allow_cancel_routing(self) -> bool {
        matches!(
            self,
            Self::Cancellable | Self::CancellableUntilWorkerThenDetach | Self::DetachSetupOnly
        )
    }

    pub fn allow_cancel_setup(self) -> bool {
        matches!(
            self,
            Self::Cancellable | Self::CancellableUntilWorkerThenDetach
        )
    }

    pub fn allow_cancel_stream(self) -> bool {
        matches!(
            self,
            Self::Cancellable | Self::DetachToWorkerStreamConnected | Self::DetachSetupOnly
        )
    }
}

/// Worker role used for request attribution and admission logs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterWorkerPhase {
    Agg,
    DecodeFirst,
    PrefillFirst,
    DecodeSecond,
    PrefillSecond,
}

impl RouterWorkerPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Agg => "agg",
            Self::DecodeFirst => "decode_first",
            Self::PrefillFirst => "prefill_first",
            Self::DecodeSecond => "decode_second",
            Self::PrefillSecond => "prefill_second",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "agg" => Some(Self::Agg),
            "decode_first" => Some(Self::DecodeFirst),
            "prefill_first" => Some(Self::PrefillFirst),
            "decode_second" => Some(Self::DecodeSecond),
            "prefill_second" => Some(Self::PrefillSecond),
            _ => None,
        }
    }
}

#[derive(Clone)]
pub struct MinReplicaAvailable {
    pub name: String,
    pub router: Arc<dyn RouterGuardClient>,
}

/// Per-call behavior for [`crate::RouterWorkerCoordinator::route_and_worker`].
pub struct RouteOptions {
    pub require_available: Vec<MinReplicaAvailable>,
    pub cancellation: CancellationPolicy,
    pub max_reroutes: u64,
    pub tracing_enabled: bool,
    pub wait_for_first_response: bool,
    pub phase: Option<RouterWorkerPhase>,
}

impl Default for RouteOptions {
    fn default() -> Self {
        Self {
            require_available: Vec::new(),
            cancellation: CancellationPolicy::default(),
            max_reroutes: 1,
            tracing_enabled: false,
            wait_for_first_response: false,
            phase: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AdmittedRequestTimings {
    pub routing_new_duration: Duration,
    pub routing_stream_connect_duration: Duration,
    pub worker_stream_connect_duration: Duration,
    pub worker_first_response_duration: Option<Duration>,
    pub worker_sentinel_event_duration: Option<Duration>,
    pub worker_connect_duration: Duration,
    pub stale_reroutes: u64,
}

#[derive(Debug)]
pub enum DeniedRequest {
    RouterBackpressure {
        reason: String,
        queued_isl_tokens: usize,
        max_queued_isl_tokens: Option<usize>,
    },
    RequiredComponentsDown {
        name: String,
    },
    NextRouterUnreachable {
        error: String,
    },
    ProtocolError {
        received: String,
    },
    Cancelled(),
    FirstWorkerEventFailed {
        error: String,
    },
}
