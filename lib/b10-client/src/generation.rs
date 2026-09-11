// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! High-level generation coordination built on [`RouterWorkerCoordinator`].
//!
//! This module owns the language-neutral parts of the generation coordinator:
//! aggregate and prefill-first routing, the prefill/decode handoff, router
//! guard lifetimes, decode bootstrap suppression, topology constraints, and
//! admission metadata. Model-specific request validation and serialization
//! stay at the language binding and are passed here as opaque MessagePack maps.

use crate::{
    CancellationPolicy, RequestContext, RouteAndConnectOutcome, RouteOptions, RouterRequestNew,
    RouterWorkerCoordinator, RouterWorkerPhase, stream_with_optional_prefill_mark,
};
use anyhow::{Context, Result, bail};
use dynamo_kv_router::protocols::RoutingConstraints;
use dynamo_runtime::pipeline::{EngineStream, ResponseStream};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::protocols::maybe_error::MaybeError;
use futures::StreamExt;
use futures::future::BoxFuture;
use rmpv::Value;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const MIN_GLOBAL_DISAGG_REQUEST_ID: u64 = 1 << 42;
const MACHINE_ID_BITS: u32 = 10;
const COUNTER_BITS: u32 = 12;
static DISAGG_REQUEST_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generation topology implemented by [`GenerationCoordinator`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DisaggregationStrategy {
    /// One worker performs prefill and decode.
    #[default]
    Aggregated,
    /// The primary pool performs prefill and the next pool performs decode.
    PrefillFirst,
}

/// When the prefill router should stop counting prefill work in flight.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PrefillMarkTiming {
    /// Mark on the first prefill response while KV transfer remains in flight.
    #[default]
    AfterPrefillCompute,
    /// Mark only after the decode worker's bootstrap proves transfer completed.
    AfterTransfer,
}

/// Opaque worker payloads plus the typed routing request for one generation.
///
/// `decode_worker_request` is required for `PrefillFirst`. Keeping it separate
/// lets a binding omit large multimodal blobs from the decode payload before
/// entering Rust. Rust injects the prefill handoff into it.
pub struct GenerationRequest {
    pub routing_request: RouterRequestNew,
    pub primary_worker_request: Value,
    pub decode_worker_request: Option<Value>,
}

/// Per-request routing behavior. Phase, first-response, tracing, and reroute
/// settings owned by the generation state machine are normalized internally.
#[derive(Default)]
pub struct GenerationOptions {
    pub primary: RouteOptions,
    pub decode: RouteOptions,
}

/// Admission metadata from the prefill-bearing route.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GenerationAdmission {
    pub estimated_overlap_tokens: u64,
    pub best_overlap_blocks: u64,
    pub prefill_worker_id: u64,
    pub prefill_dp_rank: u32,
    /// `None` when a terminal/error prefill response does not open decode.
    pub decode_worker_id: Option<u64>,
    pub decode_dp_rank: Option<u32>,
}

/// A successfully connected generation stream.
pub struct GeneratedRequest {
    pub stream: EngineStream<Annotated<Value>>,
    pub admission: GenerationAdmission,
}

/// A generation denial, optionally carrying admission metadata from a
/// prefill leg that completed before a downstream decode denial.
#[derive(Debug)]
pub struct DeniedGenerationRequest {
    pub denied: crate::DeniedRequest,
    pub admission: Option<GenerationAdmission>,
}

/// Generation either connects all required worker legs or returns the typed
/// denial produced by routing. Transport/protocol failures are returned as
/// `Err` from [`GenerationCoordinator::generate`].
pub enum GenerationOutcome {
    Connected(GeneratedRequest),
    Denied(DeniedGenerationRequest),
}

impl std::fmt::Debug for GenerationOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Connected(request) => f
                .debug_struct("Connected")
                .field("admission", &request.admission)
                .finish_non_exhaustive(),
            Self::Denied(denied) => f.debug_tuple("Denied").field(denied).finish(),
        }
    }
}

/// Coordinates complete generation requests over one or two routed pools.
pub struct GenerationCoordinator {
    primary: Arc<RouterWorkerCoordinator>,
    next: Option<Arc<RouterWorkerCoordinator>>,
    strategy: DisaggregationStrategy,
    prefill_mark_timing: PrefillMarkTiming,
    disagg_request_id_machine_id: u64,
}

impl GenerationCoordinator {
    /// Return the best eligible worker (or prefill/decode pair), without admission.
    pub async fn bid(
        &self,
        request: crate::protocol::BidRequestV1,
    ) -> Result<crate::protocol::BidResponseV1> {
        match self.strategy {
            DisaggregationStrategy::Aggregated => self.primary.bid(request).await,
            DisaggregationStrategy::PrefillFirst => {
                let next = self
                    .next
                    .as_ref()
                    .expect("validated disaggregated topology");
                let (prefill, decode) =
                    tokio::try_join!(self.primary.bid(request.clone()), next.bid(request))?;
                Ok(crate::protocol::BidResponseV1 {
                    decode_tokens: decode.decode_tokens,
                    ..prefill
                })
            }
        }
    }

    pub async fn worker_loads(&self) -> Result<Vec<WorkerLoad>> {
        match self.strategy {
            DisaggregationStrategy::Aggregated => {
                self.primary
                    .worker_loads(WorkerMode::PrefillAndDecode)
                    .await
            }
            DisaggregationStrategy::PrefillFirst => {
                let next = self
                    .next
                    .as_ref()
                    .expect("validated disaggregated topology");
                let (mut prefill, decode) = tokio::try_join!(
                    self.primary.worker_loads(WorkerMode::Prefill),
                    next.worker_loads(WorkerMode::Decode),
                )?;
                prefill.extend(decode);
                Ok(prefill)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        primary: Arc<RouterWorkerCoordinator>,
        next: Option<Arc<RouterWorkerCoordinator>>,
        strategy: DisaggregationStrategy,
        prefill_mark_timing: PrefillMarkTiming,
        disagg_request_id_machine_id: u64,
    ) -> Result<Self> {
        if strategy == DisaggregationStrategy::PrefillFirst && next.is_none() {
            bail!("next coordinator is required for prefill-first generation");
        }
        Ok(Self {
            primary,
            next,
            strategy,
            prefill_mark_timing,
            disagg_request_id_machine_id,
        })
    }

    pub async fn generate(
        &self,
        context: RequestContext,
        request: GenerationRequest,
        options: GenerationOptions,
    ) -> Result<GenerationOutcome> {
        validate_generation_request(&request, self.strategy)?;
        match self.strategy {
            DisaggregationStrategy::Aggregated => {
                self.generate_aggregated(context, request, options.primary)
                    .await
            }
            DisaggregationStrategy::PrefillFirst => {
                self.generate_prefill_first(context, request, options).await
            }
        }
    }

    async fn generate_aggregated(
        &self,
        context: RequestContext,
        mut request: GenerationRequest,
        mut options: RouteOptions,
    ) -> Result<GenerationOutcome> {
        set_map_field(
            &mut request.primary_worker_request,
            "disaggregation_mode",
            Value::from("prefill_and_decode"),
        )?;
        options.wait_for_first_response = true;
        options.tracing_enabled = true;
        options.phase = Some(RouterWorkerPhase::Agg);
        // Aggregate work follows the request lifetime in every phase.
        options.cancellation = CancellationPolicy::Cancellable;
        let outcome = self
            .primary
            .route_and_worker(
                context.clone(),
                request.routing_request,
                request.primary_worker_request,
                options,
            )
            .await?;
        let RouteAndConnectOutcome::Connected {
            guard,
            worker_id,
            stream,
            ..
        } = outcome
        else {
            let RouteAndConnectOutcome::Denied(denied) = outcome else {
                unreachable!()
            };
            return Ok(GenerationOutcome::Denied(DeniedGenerationRequest {
                denied,
                admission: None,
            }));
        };

        let admission = admission_from_guard(&guard, self.primary.block_size(), worker_id, true);
        let guard = Arc::new(guard);
        let stream = stream_with_optional_prefill_mark(stream, Arc::clone(&guard), true);
        let output = async_stream::stream! {
            let _guard = guard;
            futures::pin_mut!(stream);
            while let Some(item) = stream.next().await {
                yield item;
            }
        };
        Ok(GenerationOutcome::Connected(GeneratedRequest {
            stream: ResponseStream::new(Box::pin(output), context.inner()),
            admission,
        }))
    }

    async fn generate_prefill_first(
        &self,
        context: RequestContext,
        mut request: GenerationRequest,
        mut options: GenerationOptions,
    ) -> Result<GenerationOutcome> {
        let mut decode_worker_request = request
            .decode_worker_request
            .take()
            .context("decode_worker_request is required for prefill-first generation")?;
        set_map_field(
            &mut request.primary_worker_request,
            "disaggregation_mode",
            Value::from("prefill"),
        )?;
        options.primary.wait_for_first_response = true;
        options.primary.tracing_enabled = true;
        options.primary.phase = Some(RouterWorkerPhase::PrefillFirst);
        // Cancellation is allowed through routing and worker setup. Once the
        // prefill payload is sent the worker may pin KV for our context; the
        // prefill stream is detached so a client disconnect cannot strand that
        // staging before the decode leg takes over.
        options.primary.cancellation = CancellationPolicy::CancellableUntilWorkerThenDetach;

        let prefill = self
            .primary
            .route_and_worker(
                context.clone(),
                request.routing_request.clone(),
                request.primary_worker_request,
                options.primary,
            )
            .await?;
        let RouteAndConnectOutcome::Connected {
            guard: prefill_guard,
            worker_id: prefill_worker_id,
            mut stream,
            ..
        } = prefill
        else {
            let RouteAndConnectOutcome::Denied(denied) = prefill else {
                unreachable!()
            };
            return Ok(GenerationOutcome::Denied(DeniedGenerationRequest {
                denied,
                admission: None,
            }));
        };

        let mut prefill_response = next_visible_data(&mut stream, "prefill").await?;
        let admission = admission_from_guard(
            &prefill_guard,
            self.primary.block_size(),
            prefill_worker_id,
            false,
        );
        if self.prefill_mark_timing == PrefillMarkTiming::AfterPrefillCompute {
            prefill_guard.mark_prefill();
        }

        let handoff =
            prepare_handoff_response(&mut prefill_response, self.disagg_request_id_machine_id)?;
        let Some(handoff) = handoff else {
            // Error/terminal prefill responses have no decode leg. The Python
            // coordinator marks prefill on these paths regardless of timing.
            prefill_guard.mark_prefill();
            let output = async_stream::stream! {
                let _guard = prefill_guard;
                yield Annotated::from_data(prefill_response);
            };
            return Ok(GenerationOutcome::Connected(GeneratedRequest {
                stream: ResponseStream::new(Box::pin(output), context.inner()),
                admission,
            }));
        };

        set_map_field(
            &mut decode_worker_request,
            "disaggregation_mode",
            Value::from("decode"),
        )?;
        set_map_field(&mut decode_worker_request, "disaggregated_params", handoff)?;
        merge_response_routing_constraints(&prefill_response, &mut request.routing_request)?;

        options.decode.wait_for_first_response = false;
        options.decode.tracing_enabled = true;
        options.decode.max_reroutes = 2;
        options.decode.phase = Some(RouterWorkerPhase::DecodeSecond);
        // Once prefill produced a handoff, abandoning decode routing/setup can
        // strand transferred KV state. RouterWorkerCoordinator shields those
        // phases to completion under this policy, then links the connected
        // decode stream back to the parent so normal client cancellation is
        // restored after the critical handoff window.
        options.decode.cancellation = CancellationPolicy::DetachToWorkerStreamConnected;
        // GenerationCoordinatorV2 intentionally applies request priority only
        // to the first routed component. Allowed workers and do_not_queue are
        // request-wide and remain on the cloned request.
        request.routing_request.priority_jump = 0.0;
        request.routing_request.priority_load_shed_percent = 0;
        let decode = self
            .next
            .as_ref()
            .expect("constructor validates next coordinator")
            .route_and_worker(
                context.clone(),
                request.routing_request,
                decode_worker_request,
                options.decode,
            )
            .await?;
        let RouteAndConnectOutcome::Connected {
            guard: decode_guard,
            worker_id: decode_worker_id,
            stream: mut decode_stream,
            ..
        } = decode
        else {
            let RouteAndConnectOutcome::Denied(denied) = decode else {
                unreachable!()
            };
            return Ok(GenerationOutcome::Denied(DeniedGenerationRequest {
                denied,
                admission: Some(admission),
            }));
        };

        let (decode_worker_id, decode_dp_rank) = decode_guard
            .routed_worker_info()
            .unwrap_or((decode_worker_id, 0));
        let mut admission = admission;
        admission.decode_worker_id = Some(decode_worker_id);
        admission.decode_dp_rank = Some(decode_dp_rank);

        // Yield the prefill handoff immediately. Bootstrap consumption occurs
        // on the next poll, preserving time-to-first-token behavior.
        let output = async_stream::stream! {
            let prefill_guard = prefill_guard;
            let _decode_guard = decode_guard;
            yield Annotated::from_data(prefill_response);

            match next_visible_annotated(&mut decode_stream).await {
                Some(error) if error.is_error() => {
                    yield error;
                    return;
                }
                Some(_bootstrap) => {
                    // Decode readiness proves KV transfer completed. The two
                    // guard operations are idempotent for both mark timings.
                    prefill_guard.mark_free();
                    prefill_guard.mark_prefill();
                }
                None => {
                    yield Annotated::from_error(
                        "decode bootstrap stream ended before a response was produced",
                    );
                    return;
                }
            }

            while let Some(item) = decode_stream.next().await {
                let is_error = item.is_error();
                if is_error || item.data.is_some() {
                    yield item;
                }
                if is_error {
                    return;
                }
            }
        };
        Ok(GenerationOutcome::Connected(GeneratedRequest {
            stream: ResponseStream::new(Box::pin(output), context.inner()),
            admission,
        }))
    }
}

/// Common interface implemented by in-process and remote generation clients.
///
/// Requests are owned so bindings can choose the implementation once during
/// construction and use one object-safe API for every generation.
pub trait GenerationCoordinatorClient: Send + Sync {
    fn bid(
        &self,
        _request: crate::protocol::BidRequestV1,
    ) -> BoxFuture<'_, Result<crate::protocol::BidResponseV1>> {
        Box::pin(async { bail!("bidding is not supported by this coordinator") })
    }
    fn start(&self) -> BoxFuture<'_, Result<()>> {
        Box::pin(futures::future::ready(Ok(())))
    }
    fn worker_loads(&self) -> BoxFuture<'_, Result<Vec<WorkerLoad>>> {
        Box::pin(async { bail!("worker loads are not supported by this coordinator") })
    }
    fn generate(
        &self,
        context: RequestContext,
        request: GenerationRequest,
        options: GenerationOptions,
    ) -> BoxFuture<'_, Result<GenerationOutcome>>;
}

impl GenerationCoordinatorClient for GenerationCoordinator {
    fn bid(
        &self,
        request: crate::protocol::BidRequestV1,
    ) -> BoxFuture<'_, Result<crate::protocol::BidResponseV1>> {
        Box::pin(GenerationCoordinator::bid(self, request))
    }
    fn worker_loads(&self) -> BoxFuture<'_, Result<Vec<WorkerLoad>>> {
        Box::pin(GenerationCoordinator::worker_loads(self))
    }
    fn generate(
        &self,
        context: RequestContext,
        request: GenerationRequest,
        options: GenerationOptions,
    ) -> BoxFuture<'_, Result<GenerationOutcome>> {
        Box::pin(GenerationCoordinator::generate(
            self, context, request, options,
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerMode {
    PrefillAndDecode,
    Prefill,
    Decode,
}

/// DP-rank loads summed per worker, with the one-token idle probe removed per rank.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WorkerLoad {
    pub worker_id: u64,
    pub disaggregation_mode: WorkerMode,
    pub potential_prefill_tokens: usize,
    pub potential_decode_blocks: usize,
    pub active_requests: usize,
}

fn validate_generation_request(
    request: &GenerationRequest,
    strategy: DisaggregationStrategy,
) -> Result<()> {
    if map_get_value(&request.primary_worker_request, "router_response")
        .is_some_and(|value| !value.is_nil())
    {
        bail!("router_response must not be populated before routing");
    }
    if request
        .decode_worker_request
        .as_ref()
        .and_then(|payload| map_get_value(payload, "router_response"))
        .is_some_and(|value| !value.is_nil())
    {
        bail!("router_response must not be populated before routing");
    }
    if strategy == DisaggregationStrategy::PrefillFirst
        && map_get_value(&request.primary_worker_request, "sampling_params")
            .and_then(|params| map_get_value(params, "max_tokens"))
            .and_then(Value::as_i64)
            .is_some_and(|max_tokens| max_tokens <= 1)
    {
        bail!("max_tokens must be at least 2 for this deployment");
    }
    Ok(())
}

fn admission_from_guard(
    guard: &crate::RouterRequestGuard,
    block_size: u32,
    fallback_worker_id: u64,
    aggregated: bool,
) -> GenerationAdmission {
    let (worker_id, dp_rank) = guard
        .routed_worker_info()
        .unwrap_or((fallback_worker_id, 0));
    GenerationAdmission {
        estimated_overlap_tokens: guard.estimated_overlap_tokens(block_size),
        best_overlap_blocks: guard.b10_best_overlap_blocks(),
        prefill_worker_id: worker_id,
        prefill_dp_rank: dp_rank,
        decode_worker_id: aggregated.then_some(worker_id),
        decode_dp_rank: aggregated.then_some(dp_rank),
    }
}

async fn next_visible_data(
    stream: &mut EngineStream<Annotated<Value>>,
    name: &str,
) -> Result<Value> {
    while let Some(item) = stream.next().await {
        // Python's AsyncResponseStream checked errors before exposing data to
        // the legacy coordinator. Preserve the original failure at this boundary.
        if let Some(error) = item.err() {
            return Err(error.into());
        }
        if let Some(data) = item.data {
            return Ok(data);
        }
    }
    bail!("{name} stream ended before a response was produced")
}

async fn next_visible_annotated(
    stream: &mut EngineStream<Annotated<Value>>,
) -> Option<Annotated<Value>> {
    while let Some(item) = stream.next().await {
        if item.is_error() || item.data.is_some() {
            return Some(item);
        }
    }
    None
}

fn prepare_handoff_response(response: &mut Value, machine_id: u64) -> Result<Option<Value>> {
    let output = first_output_mut(response)?;
    let finish_reason = map_get(output, "finish_reason").and_then(Value::as_str);
    let can_handoff = matches!(finish_reason, Some("length" | "not_finished"));
    let params = map_get(output, "disaggregated_params")
        .filter(|value| !value.is_nil())
        .cloned();

    if !can_handoff || params.is_none() {
        map_remove(output, "disaggregated_params");
        return Ok(None);
    }

    map_remove(output, "finish_reason");
    let mut params = params.expect("checked Some above");
    ensure_disagg_request_id(&mut params, machine_id)?;
    set_map_entry(output, "disaggregated_params", params.clone());
    Ok(Some(params))
}

fn ensure_disagg_request_id(params: &mut Value, machine_id: u64) -> Result<()> {
    let map = as_map_mut(params, "disaggregated_params")?;
    if map_get(map, "disagg_request_id").is_some_and(|value| !value.is_nil()) {
        return Ok(());
    }
    let id = map_get(map, "ctx_request_id")
        .filter(|value| !value.is_nil())
        .and_then(Value::as_u64)
        .unwrap_or_else(|| new_disagg_request_id(machine_id));
    set_map_entry(map, "disagg_request_id", Value::from(id));
    Ok(())
}

fn new_disagg_request_id(machine_id: u64) -> u64 {
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let counter_mask = (1_u64 << COUNTER_BITS) - 1;
    let machine_mask = (1_u64 << MACHINE_ID_BITS) - 1;
    let counter = DISAGG_REQUEST_ID_COUNTER.fetch_add(1, Ordering::Relaxed) & counter_mask;
    let raw = (timestamp_ms << (MACHINE_ID_BITS + COUNTER_BITS))
        | ((machine_id & machine_mask) << COUNTER_BITS)
        | counter;
    raw % (i64::MAX as u64 - MIN_GLOBAL_DISAGG_REQUEST_ID) + MIN_GLOBAL_DISAGG_REQUEST_ID
}

fn merge_response_routing_constraints(
    response: &Value,
    routing_request: &mut RouterRequestNew,
) -> Result<()> {
    let Some(value) = map_get_value(response, "added_topology_routing_constraints") else {
        return Ok(());
    };
    if value.is_nil() {
        return Ok(());
    }
    let engine: RoutingConstraints = rmpv::ext::from_value(value.clone())
        .context("invalid added_topology_routing_constraints")?;
    merge_routing_constraints(&mut routing_request.routing_constraints, engine);
    Ok(())
}

fn merge_routing_constraints(base: &mut RoutingConstraints, added: RoutingConstraints) {
    base.required_taints.extend(added.required_taints);
    for (taint, weight) in added.preferred_taints {
        if !base.required_taints.contains(&taint) {
            *base.preferred_taints.entry(taint).or_default() += weight;
        }
    }
    for taint in &base.required_taints {
        base.preferred_taints.remove(taint);
    }
}

fn first_output_mut(response: &mut Value) -> Result<&mut Vec<(Value, Value)>> {
    let response = as_map_mut(response, "worker response")?;
    let outputs = map_get_mut(response, "outputs").context("worker response is missing outputs")?;
    let Value::Array(outputs) = outputs else {
        bail!("worker response outputs must be an array");
    };
    let first = outputs
        .first_mut()
        .context("worker response outputs must not be empty")?;
    as_map_mut(first, "worker response output")
}

fn as_map_mut<'a>(value: &'a mut Value, name: &str) -> Result<&'a mut Vec<(Value, Value)>> {
    let Value::Map(map) = value else {
        bail!("{name} must be an object");
    };
    Ok(map)
}

fn map_get<'a>(map: &'a [(Value, Value)], key: &str) -> Option<&'a Value> {
    map.iter()
        .find(|(candidate, _)| candidate.as_str() == Some(key))
        .map(|(_, value)| value)
}

fn map_get_mut<'a>(map: &'a mut [(Value, Value)], key: &str) -> Option<&'a mut Value> {
    map.iter_mut()
        .find(|(candidate, _)| candidate.as_str() == Some(key))
        .map(|(_, value)| value)
}

fn map_get_value<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    value.as_map().and_then(|map| map_get(map, key))
}

fn map_remove(map: &mut Vec<(Value, Value)>, key: &str) -> Option<Value> {
    map.iter()
        .position(|(candidate, _)| candidate.as_str() == Some(key))
        .map(|index| map.remove(index).1)
}

fn set_map_field(target: &mut Value, key: &str, value: Value) -> Result<()> {
    let map = as_map_mut(target, "worker request")?;
    set_map_entry(map, key, value);
    Ok(())
}

fn set_map_entry(map: &mut Vec<(Value, Value)>, key: &str, value: Value) {
    if let Some(existing) = map_get_mut(map, key) {
        *existing = value;
    } else {
        map.push((Value::from(key), value));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jv(value: serde_json::Value) -> Value {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn handoff_removes_finish_reason_and_uses_context_id() {
        let mut response = jv(serde_json::json!({
            "outputs": [{
                "finish_reason": "not_finished",
                "disaggregated_params": {
                    "request_type": "context_only",
                    "ctx_request_id": 91
                }
            }]
        }));

        let handoff = prepare_handoff_response(&mut response, 7).unwrap().unwrap();

        assert_eq!(handoff["disagg_request_id"].as_u64(), Some(91));
        assert!(response["outputs"][0]["finish_reason"].is_nil());
        assert_eq!(
            response["outputs"][0]["disaggregated_params"]["disagg_request_id"].as_u64(),
            Some(91)
        );
    }

    #[test]
    fn terminal_response_drops_handoff() {
        let mut response = jv(serde_json::json!({
            "outputs": [{
                "finish_reason": "stop",
                "disaggregated_params": {"request_type": "context_only"}
            }]
        }));

        assert!(
            prepare_handoff_response(&mut response, 7)
                .unwrap()
                .is_none()
        );
        assert!(response["outputs"][0]["disaggregated_params"].is_nil());
    }

    #[test]
    fn topology_constraints_merge_and_required_wins() {
        let mut base = RoutingConstraints::default();
        base.required_taints.insert("rack=a".into());
        base.preferred_taints.insert("zone=x".into(), 1.0);
        let mut added = RoutingConstraints::default();
        added.required_taints.insert("zone=x".into());
        added.preferred_taints.insert("rack=b".into(), 2.0);

        merge_routing_constraints(&mut base, added);

        assert!(base.required_taints.contains("rack=a"));
        assert!(base.required_taints.contains("zone=x"));
        assert!(!base.preferred_taints.contains_key("zone=x"));
        assert_eq!(base.preferred_taints.get("rack=b"), Some(&2.0));
    }
}
