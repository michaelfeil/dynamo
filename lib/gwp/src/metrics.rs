// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metrics for GWP scheduling and upstream outcomes.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use dynamo_runtime::metrics::MetricsHierarchy;
use parking_lot::Mutex;
use prometheus::{GaugeVec, HistogramVec, IntCounterVec, IntGauge, IntGaugeVec};

use crate::config::EndpointId;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResponseOutcome {
    Success,
    Overloaded,
    ClientError,
    UpstreamError,
}

impl ResponseOutcome {
    pub fn from_status(status: u16) -> Self {
        match status {
            200 => Self::Success,
            429 | 503 | 529 => Self::Overloaded,
            400..=499 => Self::ClientError,
            _ => Self::UpstreamError,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Overloaded => "overloaded",
            Self::ClientError => "client_error",
            Self::UpstreamError => "upstream_error",
        }
    }
}

#[derive(Clone, Debug)]
pub struct GwpMetrics {
    routing_decisions: IntCounterVec,
    upstream_responses: IntCounterVec,
    denied_requests: IntCounterVec,
    rebooked_requests: IntCounterVec,
    session_identity: IntCounterVec,
    session_routing_decisions: IntCounterVec,
    model_stage_requests: IntCounterVec,
    affinity_lookups: IntCounterVec,
    affinity_fallback_routings: IntCounterVec,
    affinity_backend_errors: IntCounterVec,
    affinity_operation_duration_seconds: HistogramVec,
    tokenization_duration_seconds: HistogramVec,
    schedule_duration_seconds: HistogramVec,
    schedule_service_duration_seconds: HistogramVec,
    schedule_request_body_bytes: HistogramVec,
    lifecycle_rpc_total: IntCounterVec,
    lifecycle_rpc_duration_seconds: HistogramVec,
    lifecycle_reconciliations: IntCounterVec,
    optimistic_tokenization_total: IntCounterVec,
    optimistic_tokenization_active: IntGauge,
    request_outcomes: IntCounterVec,
    inflight_requests: IntGauge,
    scheduler_inflight_requests: GaugeVec,
    scheduler_inflight_tokens: GaugeVec,
    scheduler_live_workers: IntGaugeVec,
    topology_ambiguous_workers: IntGauge,
    scheduler_load: Arc<Mutex<HashMap<EndpointId, ClusterSchedulerLoad>>>,
    time_to_first_byte_seconds: HistogramVec,
    request_duration_seconds: HistogramVec,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ClusterSchedulerLoad {
    requests: usize,
    total_tokens: usize,
    cached_tokens: usize,
    live_workers: usize,
}

impl ClusterSchedulerLoad {
    fn normalized(self, value: usize) -> f64 {
        if self.live_workers == 0 {
            0.0
        } else {
            value as f64 / self.live_workers as f64
        }
    }
}

const ROUTE_LABELS: &[&str] = &["routed_endpoint", "model", "downstream_authority"];
const ROUTE_OUTCOME_LABELS: &[&str] = &[
    "routed_endpoint",
    "model",
    "downstream_authority",
    "outcome",
];

impl GwpMetrics {
    pub fn from_endpoint(endpoint: &dynamo_runtime::component::Endpoint) -> anyhow::Result<Self> {
        let metrics = endpoint.metrics();
        Ok(Self {
            routing_decisions: metrics.create_intcountervec(
                "gwp_routing_decisions_total",
                "GWP routing decisions made before forwarding upstream.",
                &[
                    "routed_endpoint",
                    "model",
                    "downstream_authority",
                    "sticky",
                ],
                &[],
            )?,
            upstream_responses: metrics.create_intcountervec(
                "gwp_upstream_responses_total",
                "Upstream response headers observed by GWP.",
                &[
                    "routed_endpoint",
                    "model",
                    "downstream_authority",
                    "status",
                    "outcome",
                ],
                &[],
            )?,
            denied_requests: metrics.create_intcountervec(
                "gwp_denied_requests_total",
                "Non-200 upstream responses that caused GWP to release the provisional booking.",
                &[
                    "routed_endpoint",
                    "model",
                    "downstream_authority",
                    "status",
                    "outcome",
                ],
                &[],
            )?,
            rebooked_requests: metrics.create_intcountervec(
                "gwp_rebooked_requests_total",
                "Successful requests moved from the provisional worker to the worker reported upstream.",
                ROUTE_LABELS,
                &[],
            )?,
            session_identity: metrics.create_intcountervec(
                "gwp_session_identity_total",
                "GWP scheduling attempts classified by session identity source.",
                &["source"],
                &[],
            )?,
            session_routing_decisions: metrics.create_intcountervec(
                "gwp_session_routing_decisions_total",
                "Successful GWP scheduling decisions classified by session identity source and whether affinity was used.",
                &["source", "sticky"],
                &[],
            )?,
            model_stage_requests: metrics.create_intcountervec(
                "gwp_model_stage_requests_total",
                "Scheduling attempts by resolved stage state; malformed or unrouted models use model=unknown.",
                &["model", "stage", "enabled"],
                &[],
            )?,
            affinity_lookups: metrics.create_intcountervec(
                "gwp_affinity_lookups_total",
                "Session-affinity lookups classified before routing.",
                &["backend", "outcome"],
                &[],
            )?,
            affinity_fallback_routings: metrics.create_intcountervec(
                "gwp_affinity_fallback_routings_total",
                "Fresh routing decisions after a stored session binding became unusable, classified by previous and newly selected endpoint.",
                &[
                    "backend",
                    "reason",
                    "previous_endpoint",
                    "routed_endpoint",
                    "cluster_result",
                ],
                &[],
            )?,
            affinity_backend_errors: metrics.create_intcountervec(
                "gwp_affinity_backend_errors_total",
                "Session-affinity backend operation failures.",
                &["backend", "operation"],
                &[],
            )?,
            affinity_operation_duration_seconds: metrics.create_histogramvec(
                "gwp_affinity_operation_duration_seconds",
                "Session-affinity backend operation latency, including timeout and bulkhead outcomes.",
                &["backend", "operation", "result"],
                &[],
                Some(vec![
                    0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05,
                    0.1, 0.2, 0.25,
                ]),
            )?,
            tokenization_duration_seconds: metrics.create_histogramvec(
                "gwp_tokenization_duration_seconds",
                "Time spent converting an OpenAI request into routing tokens.",
                &["model", "mode", "result"],
                &[],
                Some(vec![
                    0.00001, 0.000025, 0.00005, 0.0001, 0.00025, 0.0005, 0.001,
                    0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
                ]),
            )?,
            schedule_duration_seconds: metrics.create_histogramvec(
                "gwp_schedule_duration_seconds",
                "Internal schedule latency by routing path.",
                &["path", "result"],
                &[],
                Some(vec![
                    0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025,
                    0.05, 0.1, 0.25,
                ]),
            )?,
            schedule_service_duration_seconds: metrics.create_histogramvec(
                "gwp_schedule_service_duration_seconds",
                "Complete GWP schedule service time from control request arrival through response encoding.",
                &["adapter", "result"],
                &[],
                Some(vec![
                    0.00025, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.0075,
                    0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.2,
                    0.5, 1.0, 5.0,
                ]),
            )?,
            schedule_request_body_bytes: metrics.create_histogramvec(
                "gwp_schedule_request_body_bytes",
                "Request body size received by a GWP scheduling adapter.",
                &["adapter", "result"],
                &[],
                Some(vec![
                    1_024.0,
                    4_096.0,
                    16_384.0,
                    65_536.0,
                    262_144.0,
                    1_048_576.0,
                    4_194_304.0,
                    16_777_216.0,
                    52_428_800.0,
                ]),
            )?,
            lifecycle_rpc_total: metrics.create_intcountervec(
                "gwp_lifecycle_rpc_total",
                "Unary gRPC lifecycle notifications processed by GWP.",
                &["method", "result"],
                &[],
            )?,
            lifecycle_rpc_duration_seconds: metrics.create_histogramvec(
                "gwp_lifecycle_rpc_duration_seconds",
                "Time spent processing unary gRPC lifecycle notifications.",
                &["method", "result"],
                &[],
                Some(vec![
                    0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025,
                    0.05, 0.1, 0.25, 1.0,
                ]),
            )?,
            lifecycle_reconciliations: metrics.create_intcountervec(
                "gwp_lifecycle_reconciliations_total",
                "Out-of-order or missing lifecycle events reconciled by the per-request grace state.",
                &["outcome"],
                &[],
            )?,
            optimistic_tokenization_total: metrics.create_intcountervec(
                "gwp_optimistic_tokenization_total",
                "Affinity-hit background tokenization outcomes.",
                &["result"],
                &[],
            )?,
            optimistic_tokenization_active: metrics.create_intgauge(
                "gwp_optimistic_tokenization_active",
                "Affinity-hit tokenization jobs currently running in the bounded background pool.",
                &[],
            )?,
            request_outcomes: metrics.create_intcountervec(
                "gwp_request_outcomes_total",
                "Terminal request outcomes recorded after the final scheduler booking is freed.",
                ROUTE_OUTCOME_LABELS,
                &[],
            )?,
            inflight_requests: metrics.create_intgauge(
                "gwp_inflight_requests",
                "Requests with a live provisional or confirmed GWP booking.",
                &[],
            )?,
            scheduler_inflight_requests: metrics.create_gaugevec(
                "gwp_scheduler_inflight_requests",
                "Locally owned in-flight scheduler requests by routed endpoint. Sum total series across GWP replicas.",
                &["routed_endpoint", "aggregation"],
                &[],
            )?,
            scheduler_inflight_tokens: metrics.create_gaugevec(
                "gwp_scheduler_inflight_tokens",
                "Locally owned in-flight input tokens by routed endpoint and estimated cache status. Sum total series across GWP replicas.",
                &["routed_endpoint", "cache_status", "aggregation"],
                &[],
            )?,
            scheduler_live_workers: metrics.create_intgaugevec(
                "gwp_scheduler_live_workers",
                "Topology-provider live workers in the routed endpoint. Use max rather than sum across GWP replicas.",
                &["routed_endpoint"],
                &[],
            )?,
            topology_ambiguous_workers: metrics.create_intgauge(
                "gwp_topology_ambiguous_workers",
                "Worker IDs excluded because multiple endpoints currently advertise them. Use max rather than sum across GWP replicas.",
                &[],
            )?,
            scheduler_load: Arc::default(),
            time_to_first_byte_seconds: metrics.create_histogramvec(
                "gwp_time_to_first_byte_seconds",
                "Time from provisional booking to receiving upstream response headers.",
                ROUTE_OUTCOME_LABELS,
                &[],
                Some(vec![
                    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
                    60.0, 120.0,
                ]),
            )?,
            request_duration_seconds: metrics.create_histogramvec(
                "gwp_request_duration_seconds",
                "Time from provisional booking until the final scheduler booking is freed.",
                ROUTE_OUTCOME_LABELS,
                &[],
                Some(vec![
                    0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
                    120.0, 300.0, 600.0,
                ]),
            )?,
        })
    }

    pub fn record_routing_decision(
        &self,
        endpoint: &EndpointId,
        model: &str,
        downstream_authority: &str,
        sticky: bool,
    ) {
        self.routing_decisions
            .with_label_values(&[
                endpoint.0.as_str(),
                model,
                downstream_authority,
                if sticky { "true" } else { "false" },
            ])
            .inc();
    }

    pub fn record_upstream_response(
        &self,
        endpoint: &EndpointId,
        model: &str,
        downstream_authority: &str,
        status: u16,
    ) {
        let outcome = ResponseOutcome::from_status(status);
        self.upstream_responses
            .with_label_values(&[
                endpoint.0.as_str(),
                model,
                downstream_authority,
                &status.to_string(),
                outcome.as_str(),
            ])
            .inc();
    }

    pub fn record_denied(
        &self,
        endpoint: &EndpointId,
        model: &str,
        downstream_authority: &str,
        status: u16,
    ) {
        let outcome = ResponseOutcome::from_status(status);
        self.denied_requests
            .with_label_values(&[
                endpoint.0.as_str(),
                model,
                downstream_authority,
                &status.to_string(),
                outcome.as_str(),
            ])
            .inc();
    }

    pub fn record_rebooked(&self, endpoint: &EndpointId, model: &str, downstream_authority: &str) {
        self.rebooked_requests
            .with_label_values(&[endpoint.0.as_str(), model, downstream_authority])
            .inc();
    }

    pub fn record_session_identity(&self, source: &str) {
        self.session_identity.with_label_values(&[source]).inc();
    }

    pub fn record_session_routing_decision(&self, source: &str, sticky: bool) {
        self.session_routing_decisions
            .with_label_values(&[source, if sticky { "true" } else { "false" }])
            .inc();
    }

    pub fn record_model_stage(&self, model: &str, stage: &str, enabled: bool) {
        self.model_stage_requests
            .with_label_values(&[model, stage, if enabled { "true" } else { "false" }])
            .inc();
    }

    pub fn record_affinity_lookup(&self, backend: &str, outcome: &str) {
        self.affinity_lookups
            .with_label_values(&[backend, outcome])
            .inc();
    }

    pub fn record_affinity_fallback_routing(
        &self,
        backend: &str,
        reason: &str,
        previous_endpoint: Option<&EndpointId>,
        routed_endpoint: &EndpointId,
    ) {
        let previous = previous_endpoint
            .map(|endpoint| endpoint.0.as_str())
            .unwrap_or("unknown");
        let cluster_result = match previous_endpoint {
            Some(endpoint) if endpoint == routed_endpoint => "same",
            Some(_) => "different",
            None => "unknown",
        };
        self.affinity_fallback_routings
            .with_label_values(&[
                backend,
                reason,
                previous,
                routed_endpoint.0.as_str(),
                cluster_result,
            ])
            .inc();
    }

    pub fn record_affinity_backend_error(&self, backend: &str, operation: &str) {
        self.affinity_backend_errors
            .with_label_values(&[backend, operation])
            .inc();
    }

    pub fn observe_affinity_operation(
        &self,
        backend: &str,
        operation: &str,
        result: &str,
        elapsed: Duration,
    ) {
        self.affinity_operation_duration_seconds
            .with_label_values(&[backend, operation, result])
            .observe(elapsed.as_secs_f64());
    }

    pub fn observe_tokenization(&self, model: &str, mode: &str, result: &str, elapsed: Duration) {
        self.tokenization_duration_seconds
            .with_label_values(&[model, mode, result])
            .observe(elapsed.as_secs_f64());
    }

    pub fn observe_schedule(&self, path: &str, result: &str, elapsed: Duration) {
        self.schedule_duration_seconds
            .with_label_values(&[path, result])
            .observe(elapsed.as_secs_f64());
    }

    pub fn observe_schedule_service(
        &self,
        adapter: &str,
        result: &str,
        elapsed: Duration,
        request_body_bytes: Option<u64>,
    ) {
        self.schedule_service_duration_seconds
            .with_label_values(&[adapter, result])
            .observe(elapsed.as_secs_f64());
        if let Some(bytes) = request_body_bytes {
            self.schedule_request_body_bytes
                .with_label_values(&[adapter, result])
                .observe(bytes as f64);
        }
    }

    pub fn record_lifecycle_reconciliation(&self, outcome: &str) {
        self.lifecycle_reconciliations
            .with_label_values(&[outcome])
            .inc();
    }

    pub fn observe_lifecycle_rpc(&self, method: &str, result: &str, elapsed: Duration) {
        self.lifecycle_rpc_total
            .with_label_values(&[method, result])
            .inc();
        self.lifecycle_rpc_duration_seconds
            .with_label_values(&[method, result])
            .observe(elapsed.as_secs_f64());
    }

    pub fn record_optimistic_tokenization(&self, result: &str) {
        self.optimistic_tokenization_total
            .with_label_values(&[result])
            .inc();
    }

    pub fn optimistic_tokenization_started(&self) {
        self.optimistic_tokenization_active.inc();
        self.record_optimistic_tokenization("started");
    }

    pub fn optimistic_tokenization_finished(&self, result: &str) {
        self.optimistic_tokenization_active.dec();
        self.record_optimistic_tokenization(result);
    }

    pub fn set_inflight(&self, value: usize) {
        self.inflight_requests
            .set(i64::try_from(value).unwrap_or(i64::MAX));
    }

    pub fn scheduler_request_started(
        &self,
        endpoint: &EndpointId,
        total_tokens: usize,
        cached_tokens: usize,
    ) {
        let load = {
            let mut loads = self.scheduler_load.lock();
            let load = loads.entry(endpoint.clone()).or_default();
            load.requests = load.requests.saturating_add(1);
            load.total_tokens = load.total_tokens.saturating_add(total_tokens);
            load.cached_tokens = load
                .cached_tokens
                .saturating_add(cached_tokens.min(total_tokens));
            *load
        };
        self.publish_scheduler_load(endpoint, load);
    }

    pub fn scheduler_request_finished(
        &self,
        endpoint: &EndpointId,
        total_tokens: usize,
        cached_tokens: usize,
    ) {
        let load = {
            let mut loads = self.scheduler_load.lock();
            let load = loads.entry(endpoint.clone()).or_default();
            load.requests = load.requests.saturating_sub(1);
            load.total_tokens = load.total_tokens.saturating_sub(total_tokens);
            load.cached_tokens = load
                .cached_tokens
                .saturating_sub(cached_tokens.min(total_tokens));
            *load
        };
        self.publish_scheduler_load(endpoint, load);
    }

    pub fn replace_scheduler_live_workers(&self, live: HashMap<EndpointId, usize>) {
        let snapshots = {
            let mut loads = self.scheduler_load.lock();
            for load in loads.values_mut() {
                load.live_workers = 0;
            }
            for (endpoint, workers) in live {
                loads.entry(endpoint).or_default().live_workers = workers;
            }
            loads
                .iter()
                .map(|(id, load)| (id.clone(), *load))
                .collect::<Vec<_>>()
        };
        for (endpoint, load) in snapshots {
            self.publish_scheduler_load(&endpoint, load);
        }
    }

    pub fn set_topology_ambiguous_workers(&self, value: usize) {
        self.topology_ambiguous_workers
            .set(i64::try_from(value).unwrap_or(i64::MAX));
    }

    fn publish_scheduler_load(&self, endpoint: &EndpointId, load: ClusterSchedulerLoad) {
        let endpoint = endpoint.0.as_str();
        self.scheduler_live_workers
            .with_label_values(&[endpoint])
            .set(i64::try_from(load.live_workers).unwrap_or(i64::MAX));

        self.scheduler_inflight_requests
            .with_label_values(&[endpoint, "total"])
            .set(load.requests as f64);
        self.scheduler_inflight_requests
            .with_label_values(&[endpoint, "per_live_worker"])
            .set(load.normalized(load.requests));

        let uncached_tokens = load.total_tokens.saturating_sub(load.cached_tokens);
        for (cache_status, tokens) in [
            ("total", load.total_tokens),
            ("cached", load.cached_tokens),
            ("uncached", uncached_tokens),
        ] {
            self.scheduler_inflight_tokens
                .with_label_values(&[endpoint, cache_status, "total"])
                .set(tokens as f64);
            self.scheduler_inflight_tokens
                .with_label_values(&[endpoint, cache_status, "per_live_worker"])
                .set(load.normalized(tokens));
        }
    }

    pub fn observe_time_to_first_byte(
        &self,
        endpoint: &EndpointId,
        model: &str,
        downstream_authority: &str,
        outcome: ResponseOutcome,
        elapsed: Duration,
    ) {
        self.time_to_first_byte_seconds
            .with_label_values(&[
                endpoint.0.as_str(),
                model,
                downstream_authority,
                outcome.as_str(),
            ])
            .observe(elapsed.as_secs_f64());
    }

    pub fn record_request_outcome(
        &self,
        endpoint: &EndpointId,
        model: &str,
        downstream_authority: &str,
        outcome: ResponseOutcome,
        elapsed: Duration,
    ) {
        let label_values = &[
            endpoint.0.as_str(),
            model,
            downstream_authority,
            outcome.as_str(),
        ];
        self.request_outcomes.with_label_values(label_values).inc();
        self.request_duration_seconds
            .with_label_values(label_values)
            .observe(elapsed.as_secs_f64());
    }
}

#[cfg(test)]
mod tests {
    use super::{ClusterSchedulerLoad, ResponseOutcome};

    #[test]
    fn response_outcomes_are_bounded_and_overload_is_explicit() {
        assert_eq!(ResponseOutcome::from_status(200), ResponseOutcome::Success);
        for status in [429, 503, 529] {
            assert_eq!(
                ResponseOutcome::from_status(status),
                ResponseOutcome::Overloaded
            );
        }
        for status in [400, 404, 499] {
            assert_eq!(
                ResponseOutcome::from_status(status),
                ResponseOutcome::ClientError
            );
        }
        for status in [201, 302, 500, 502, 504, 599] {
            assert_eq!(
                ResponseOutcome::from_status(status),
                ResponseOutcome::UpstreamError
            );
        }
    }

    #[test]
    fn scheduler_load_normalizes_by_live_workers_and_handles_zero() {
        let load = ClusterSchedulerLoad {
            requests: 3,
            total_tokens: 1200,
            cached_tokens: 300,
            live_workers: 4,
        };
        assert_eq!(load.normalized(load.requests), 0.75);
        assert_eq!(load.normalized(load.total_tokens), 300.0);
        assert_eq!(
            ClusterSchedulerLoad {
                live_workers: 0,
                ..load
            }
            .normalized(load.total_tokens),
            0.0
        );
    }
}
