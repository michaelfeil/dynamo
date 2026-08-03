// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-independent scheduling resolution shared by the gRPC adapters.

#![cfg(feature = "server")]

use http::StatusCode;

use crate::core::{GwpCore, ScheduleError, ScheduleRequest};
use crate::lifecycle::Lifecycle;

#[derive(Clone)]
pub(crate) struct ControlState {
    pub core: GwpCore,
    pub lifecycle: Lifecycle,
}

/// Step 4: the placement decision.
#[derive(Debug)]
pub(crate) struct ResolvedSchedule {
    /// GWP-owned scheduler transaction ID. It extends the caller's request ID
    /// with a locally minted UUID and must be used for both lifecycle calls.
    pub request_id: String,
    /// `host:port` of the selected endpoint ingress — what Envoy sets as the
    /// upstream authority.
    pub authority: String,
    pub endpoint_id: String,
    /// Canonical model after resolving public and legacy aliases.
    pub model: String,
    /// Session id used this turn (header, OpenAI `user`, or minted fallback).
    /// The filter injects it as `x-session-id` on the client response.
    pub session_id: String,
    pub sticky: bool,
    /// Worker confirmed on a previous request and recovered from affinity.
    pub affine_worker_id: Option<u64>,
    pub api_key: String,
}

#[derive(Debug)]
pub(crate) struct ControlError {
    pub status: StatusCode,
    pub message: String,
}

impl ControlError {
    pub(crate) fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

impl From<ScheduleError> for ControlError {
    fn from(error: ScheduleError) -> Self {
        let status = match &error {
            ScheduleError::Tokenization(_) => StatusCode::BAD_REQUEST,
            ScheduleError::RoutingRequirements(_) => StatusCode::BAD_REQUEST,
            ScheduleError::InvalidModel(_) => StatusCode::BAD_REQUEST,
            ScheduleError::NoRoutableEndpoint(_) => StatusCode::SERVICE_UNAVAILABLE,
            ScheduleError::UnknownEndpoint(_) => StatusCode::BAD_GATEWAY,
            ScheduleError::DuplicateRequestId(_) => StatusCode::CONFLICT,
            ScheduleError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self::new(status, error.to_string())
    }
}

pub(crate) async fn resolve_schedule(
    state: &ControlState,
    request_id: String,
    original_path: String,
    session_id: Option<String>,
    routing_model_id: Option<String>,
    routing_requirements: Option<String>,
    body: serde_json::Value,
) -> Result<ResolvedSchedule, ControlError> {
    if !state.lifecycle.is_ready() {
        return Err(ControlError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            state.lifecycle.report().reason,
        ));
    }
    let transaction_id = format!("{request_id}:{}", uuid::Uuid::new_v4());
    let scheduled = state
        .core
        .schedule(ScheduleRequest {
            rid: transaction_id.clone(),
            session_id,
            routing_model_id,
            routing_requirements_header: routing_requirements,
            request_path: original_path,
            body,
        })
        .await
        .map_err(ControlError::from)?;
    let ingress = &scheduled.endpoint.ingress_url;
    let authority = match (ingress.host_str(), ingress.port_or_known_default()) {
        (Some(host), Some(port)) if host.contains(':') => format!("[{host}]:{port}"),
        (Some(host), Some(port)) => format!("{host}:{port}"),
        (Some(host), None) if host.contains(':') => format!("[{host}]"),
        (Some(host), None) => host.to_string(),
        _ => String::new(),
    };
    Ok(ResolvedSchedule {
        request_id: transaction_id,
        authority,
        endpoint_id: scheduled.endpoint_id.0.clone(),
        model: scheduled.model,
        session_id: scheduled.session_id,
        sticky: scheduled.sticky,
        affine_worker_id: scheduled.affine_worker_id,
        api_key: scheduled.endpoint.api_key.clone(),
    })
}
