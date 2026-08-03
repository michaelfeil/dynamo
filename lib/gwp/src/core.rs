// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The transport-agnostic scheduling core: the three lifecycle operations from
//! the design doc ("Core lifecycle") used by the Envoy gRPC adapters.
//!
//! | lifecycle call       | Envoy hook             |
//! |----------------------|------------------------|
//! | [`GwpCore::schedule`] | ext-authz `Check`      |
//! | [`GwpCore::response_started`] | Wasm response headers |
//! | [`GwpCore::request_finished`] | Wasm `onLog`      |
//!
//! Between `schedule` and `request_finished` the core keeps an **in-flight
//! entry** per request id (session id, endpoint decision). The table is
//! replica-local — Envoy pins one request's lifecycle events to the replica
//! that scheduled it. A janitor frees entries whose
//! `RequestFinished` never arrived (Envoy crash, dropped event).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use dynamo_kv_router::protocols::WorkerWithDpRank;
use tokio::sync::{Mutex, Semaphore};

use crate::config::{
    ConfigStore, EndpointConfig, EndpointId, ModelStagePolicy, RoutingRequirements,
};
use crate::identity::EndpointTable;
use crate::metrics::ResponseOutcome;
use crate::models::ModelCatalog;
use crate::reflector::LivenessSet;
use crate::router::GwpRouter;
use crate::session::{
    AffinityBinding, AffinityStore, mint_session_id, prompt_hash_session_id, request_session_id,
};
use crate::tokens::{ApproxTokens, TokenizerRegistry};
#[cfg(test)]
use crate::tokens::{pseudo_tokens, routing_text};

/// Upper bound on how long an in-flight entry may live without a
/// `request_finished`. Generous — real LLM streams run minutes, not hours.
pub const INFLIGHT_MAX_AGE: Duration = Duration::from_secs(600);

/// How often the janitor sweeps for lost `RequestFinished` events.
const JANITOR_INTERVAL: Duration = Duration::from_secs(60);

/// Bound CPU-heavy tokenization and the amount of request-body memory retained
/// by optimistic affinity tasks. Individual tokenizer jobs use Rayon
/// internally, so this stays below the production GWP CPU limit to avoid
/// multiplying runnable CPU work across too many concurrent jobs.
const TOKENIZATION_MAX_IN_FLIGHT: usize = 4;

/// A finished request whose response-started event has not arrived is retained
/// briefly so separately multiplexed HTTP/2 control calls may arrive out of
/// order without changing the terminal outcome.
const TERMINAL_EVENT_GRACE: Duration = Duration::from_secs(5);

#[derive(Clone)]
struct RequestMetadata {
    sid: String,
    endpoint_id: EndpointId,
    model: String,
    downstream_authority: String,
    planned: WorkerWithDpRank,
    policy: ModelStagePolicy,
    routed_at: tokio::time::Instant,
}

struct Booking {
    decision: WorkerWithDpRank,
    tokens: Vec<u32>,
    cached_tokens: usize,
    accounting_rid: String,
}

#[derive(Clone, Copy)]
struct ResponseStarted {
    status: u16,
    outcome: ResponseOutcome,
    confirmed: WorkerWithDpRank,
}

/// Mutable lifecycle state. Every transition and its scheduler side effects
/// happen while the owning request's mutex is held, including across awaits.
struct RequestState {
    metadata: Option<RequestMetadata>,
    booking: Option<Booking>,
    tokenization_pending: bool,
    response: Option<ResponseStarted>,
    finished_at: Option<tokio::time::Instant>,
    terminal_outcome_recorded: bool,
    counted_inflight: bool,
}

struct RequestSlot {
    created: tokio::time::Instant,
    state: Mutex<RequestState>,
}

struct OptimisticTokenizationJob {
    rid: String,
    slot: Arc<RequestSlot>,
    model: String,
    metric_model: String,
    request_path: String,
    body: serde_json::Value,
    pseudo_stride: usize,
    permit: tokio::sync::OwnedSemaphorePermit,
}

pub struct ScheduleRequest {
    pub rid: String,
    pub session_id: Option<String>,
    /// Authoritative canonical or aliased model supplied by trusted ingress.
    /// When absent, routing falls back to the OpenAI body `model` field.
    pub routing_model_id: Option<String>,
    pub routing_requirements_header: Option<String>,
    pub request_path: String,
    pub body: serde_json::Value,
}

struct ScheduledTokenLoad {
    tokens: Vec<u32>,
    cached_tokens: usize,
}

impl ScheduledTokenLoad {
    fn new(tokens: Vec<u32>, cached_tokens: usize) -> Self {
        Self {
            cached_tokens: cached_tokens.min(tokens.len()),
            tokens,
        }
    }
}

impl RequestSlot {
    fn reserved() -> Self {
        Self {
            created: tokio::time::Instant::now(),
            state: Mutex::new(RequestState {
                metadata: None,
                booking: None,
                tokenization_pending: false,
                response: None,
                finished_at: None,
                terminal_outcome_recorded: false,
                counted_inflight: true,
            }),
        }
    }
}

impl RequestMetadata {
    fn new(
        sid: String,
        endpoint_id: EndpointId,
        model: String,
        downstream_authority: String,
        planned: WorkerWithDpRank,
        policy: ModelStagePolicy,
    ) -> Self {
        Self {
            sid,
            endpoint_id,
            model,
            downstream_authority,
            planned,
            policy,
            routed_at: tokio::time::Instant::now(),
        }
    }
}

/// What `schedule` returns to the transport shell.
#[derive(Clone, Debug)]
pub struct Scheduled {
    pub session_id: String,
    /// Canonical served model used for routing and accounting this turn.
    pub model: String,
    pub worker: WorkerWithDpRank,
    pub sticky: bool,
    /// Previously confirmed worker from a live affinity-store hit. This is
    /// absent for newly minted sessions and normal scored selections.
    pub affine_worker_id: Option<u64>,
    pub endpoint_id: EndpointId,
    pub endpoint: EndpointConfig,
}

#[derive(Debug, thiserror::Error)]
pub enum ScheduleError {
    /// An explicitly configured model tokenizer could not process the request.
    #[error("{0}")]
    Tokenization(#[from] crate::tokens::TokenizationError),
    /// The Alyx routing-requirements header is malformed — 400.
    #[error("invalid x-baseten-model-apis-routing-requirements: {0}")]
    RoutingRequirements(String),
    /// The request omitted a model or named one absent from GWP routes — 400.
    #[error("invalid model: {0}")]
    InvalidModel(String),
    /// No live endpoint could be selected (planner down / empty feed) — 503.
    #[error("no routable endpoint: {0}")]
    NoRoutableEndpoint(String),
    /// The decision maps to no configured endpoint ingress — 502.
    #[error("no ingress for routed endpoint: {0}")]
    UnknownEndpoint(String),
    /// Reusing a live request ID would make scheduler ownership ambiguous.
    #[error("request id is already active: {0}")]
    DuplicateRequestId(String),
    /// A blocking tokenization task failed independently of request validity.
    #[error("tokenization task failed: {0}")]
    Internal(String),
}

/// Parse the direct routing contract carried by
/// `x-baseten-model-apis-routing-requirements`. Missing means unconstrained.
pub fn parse_routing_requirements(raw: Option<&str>) -> Result<RoutingRequirements, ScheduleError> {
    let Some(raw) = raw.filter(|value| !value.trim().is_empty()) else {
        return Ok(RoutingRequirements::default());
    };
    let requirements: RoutingRequirements = serde_json::from_str(raw)
        .map_err(|error| ScheduleError::RoutingRequirements(error.to_string()))?;
    if requirements.0.iter().any(|(dimension, constraint)| {
        dimension.trim().is_empty()
            || constraint
                .required
                .iter()
                .chain(&constraint.preferred)
                .any(|value| value.trim().is_empty())
    }) {
        return Err(ScheduleError::RoutingRequirements(
            "dimensions and values must not be empty".into(),
        ));
    }
    Ok(requirements)
}

/// The scheduling core. One instance per router replica; cheap to clone
/// (everything inside is shared).
#[derive(Clone)]
pub struct GwpCore {
    pub router: Arc<GwpRouter>,
    pub affinity: Arc<dyn AffinityStore>,
    pub liveness: LivenessSet,
    pub table: EndpointTable,
    pub config: ConfigStore,
    pub models: ModelCatalog,
    pub tokenizers: TokenizerRegistry,
    inflight: Arc<DashMap<String, Arc<RequestSlot>>>,
    active_inflight: Arc<AtomicUsize>,
    tokenization_permits: Arc<Semaphore>,
    #[cfg(test)]
    tokenization_gate: Option<Arc<Semaphore>>,
}

/// Owns a newly reserved request until the schedule response is committed.
/// Cancelling the scheduling RPC (for example when Envoy's control call times
/// out) must not leave either the local slot or a scheduler booking behind.
struct ScheduleReservationGuard {
    core: GwpCore,
    rid: String,
    slot: Arc<RequestSlot>,
    started: std::time::Instant,
    armed: bool,
}

impl ScheduleReservationGuard {
    fn new(core: GwpCore, rid: &str, slot: Arc<RequestSlot>, started: std::time::Instant) -> Self {
        Self {
            core,
            rid: rid.to_string(),
            slot,
            started,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ScheduleReservationGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let core = self.core.clone();
        let rid = self.rid.clone();
        let slot = self.slot.clone();
        let elapsed = self.started.elapsed();
        tokio::spawn(async move {
            core.abort_reserved_request(&rid, &slot).await;
            core.router
                .metrics()
                .observe_schedule("pre_route", "cancelled", elapsed);
        });
    }
}

fn downstream_authority(endpoint: &EndpointConfig) -> String {
    let host = endpoint.ingress_url.host_str().unwrap_or("unknown");
    let host = if host.contains(':') {
        format!("[{host}]")
    } else {
        host.to_string()
    };
    endpoint
        .ingress_url
        .port_or_known_default()
        .map_or(host.clone(), |port| format!("{host}:{port}"))
}

impl GwpCore {
    pub fn new(
        router: Arc<GwpRouter>,
        affinity: Arc<dyn AffinityStore>,
        liveness: LivenessSet,
        table: EndpointTable,
        config: impl Into<ConfigStore>,
    ) -> Self {
        Self::with_models(
            router,
            affinity,
            liveness,
            table,
            config,
            ModelCatalog::new(),
        )
    }

    pub fn with_models(
        router: Arc<GwpRouter>,
        affinity: Arc<dyn AffinityStore>,
        liveness: LivenessSet,
        table: EndpointTable,
        config: impl Into<ConfigStore>,
        models: ModelCatalog,
    ) -> Self {
        let config = config.into();
        let tokenizers = TokenizerRegistry::from_config(&config.load())
            .expect("configured model tokenizer bundles must load");
        Self::with_models_and_tokenizers(
            router, affinity, liveness, table, config, models, tokenizers,
        )
    }

    pub fn with_models_and_tokenizers(
        router: Arc<GwpRouter>,
        affinity: Arc<dyn AffinityStore>,
        liveness: LivenessSet,
        table: EndpointTable,
        config: impl Into<ConfigStore>,
        models: ModelCatalog,
        tokenizers: TokenizerRegistry,
    ) -> Self {
        let config = config.into();
        models.replace_from_config(&config.load());
        Self {
            router,
            affinity,
            liveness,
            table,
            config,
            models,
            tokenizers,
            inflight: Arc::new(DashMap::new()),
            active_inflight: Arc::new(AtomicUsize::new(0)),
            tokenization_permits: Arc::new(Semaphore::new(TOKENIZATION_MAX_IN_FLIGHT)),
            #[cfg(test)]
            tokenization_gate: None,
        }
    }

    fn endpoint_for(
        &self,
        worker: WorkerWithDpRank,
    ) -> Result<(EndpointId, EndpointConfig), ScheduleError> {
        let endpoint_id = self.table.get(worker.worker_id).ok_or_else(|| {
            ScheduleError::UnknownEndpoint(format!(
                "endpoint candidate {} vanished between decision and lookup",
                worker.worker_id
            ))
        })?;
        let endpoint = self
            .config
            .load()
            .endpoints
            .get(&endpoint_id)
            .cloned()
            .ok_or_else(|| {
                ScheduleError::UnknownEndpoint(format!(
                    "endpoint {} not in GWP config",
                    endpoint_id.0
                ))
            })?;
        Ok((endpoint_id, endpoint))
    }

    /// Lifecycle step 1 (Envoy ext-authz scheduling call).
    ///
    /// Books the scheduler slot under `rid`, records the in-flight entry, and
    /// returns the placement. On `Err` nothing is left booked or in-flight.
    /// Ownership lets an affinity hit move the
    /// request into bounded background tokenization without copying a large
    /// OpenAI request body.
    pub async fn schedule(&self, request: ScheduleRequest) -> Result<Scheduled, ScheduleError> {
        let schedule_started = std::time::Instant::now();
        let slot = self.reserve_request(&request.rid)?;
        let mut reservation = ScheduleReservationGuard::new(
            self.clone(),
            &request.rid,
            slot.clone(),
            schedule_started,
        );
        let rid = request.rid.clone();
        let result = self
            .schedule_reserved(request, slot.clone(), schedule_started)
            .await;
        if result.is_err() {
            self.abort_reserved_request(&rid, &slot).await;
            self.router.metrics().observe_schedule(
                "pre_route",
                "error",
                schedule_started.elapsed(),
            );
        }
        reservation.disarm();
        result
    }

    #[cfg(test)]
    async fn schedule_for_test(
        &self,
        rid: &str,
        session_id: Option<String>,
        request_path: &str,
        body: &serde_json::Value,
    ) -> Result<Scheduled, ScheduleError> {
        self.schedule(ScheduleRequest {
            rid: rid.to_string(),
            session_id,
            routing_model_id: None,
            routing_requirements_header: None,
            request_path: request_path.to_string(),
            body: body.clone(),
        })
        .await
    }

    #[cfg(test)]
    async fn schedule_for_routing_requirements_for_test(
        &self,
        rid: &str,
        session_id: Option<String>,
        routing_requirements_header: Option<String>,
        request_path: &str,
        body: &serde_json::Value,
    ) -> Result<Scheduled, ScheduleError> {
        self.schedule(ScheduleRequest {
            rid: rid.to_string(),
            session_id,
            routing_model_id: None,
            routing_requirements_header,
            request_path: request_path.to_string(),
            body: body.clone(),
        })
        .await
    }

    fn reserve_request(&self, rid: &str) -> Result<Arc<RequestSlot>, ScheduleError> {
        match self.inflight.entry(rid.to_string()) {
            Entry::Vacant(entry) => {
                let slot = Arc::new(RequestSlot::reserved());
                entry.insert(slot.clone());
                let active = self.active_inflight.fetch_add(1, Ordering::AcqRel) + 1;
                self.router.metrics().set_inflight(active);
                Ok(slot)
            }
            Entry::Occupied(_) => Err(ScheduleError::DuplicateRequestId(rid.to_string())),
        }
    }

    async fn abort_reserved_request(&self, rid: &str, slot: &Arc<RequestSlot>) {
        let mut state = slot.state.lock().await;
        if let Some(booking) = state.booking.take() {
            self.router.free(&booking.accounting_rid).await;
            if let Some(metadata) = state.metadata.as_ref() {
                self.router.metrics().scheduler_request_finished(
                    &metadata.endpoint_id,
                    booking.tokens.len(),
                    booking.cached_tokens,
                );
            }
        }
        self.mark_inactive(&mut state);
        drop(state);
        self.remove_slot_if_same(rid, slot);
    }

    async fn schedule_reserved(
        &self,
        request: ScheduleRequest,
        slot: Arc<RequestSlot>,
        schedule_started: std::time::Instant,
    ) -> Result<Scheduled, ScheduleError> {
        let ScheduleRequest {
            rid,
            session_id,
            routing_model_id,
            routing_requirements_header,
            request_path,
            body,
        } = request;
        let rid = rid.as_str();
        let session_header_present = session_id.is_some();
        let supplied_session_id = request_session_id(session_id, &body);
        let config = self.config.load();
        let routing_requirements =
            parse_routing_requirements(routing_requirements_header.as_deref())?;
        let eligible_endpoints = config.routing_candidates(&routing_requirements);
        let requested_model = routing_model_id
            .as_deref()
            .filter(|model| !model.is_empty())
            .or_else(|| body.get("model").and_then(|model| model.as_str()))
            .ok_or_else(|| {
                ScheduleError::InvalidModel(
                    "request does not specify a model known to GWP routing".into(),
                )
            })?;
        let canonical_model = config.canonical_model(requested_model);
        if config.configured_candidates(canonical_model).is_none() {
            return Err(ScheduleError::InvalidModel(format!(
                "model {requested_model:?} is not known to GWP routing"
            )));
        }
        let model = canonical_model.to_string();
        let policy = config.model_policy(&model);
        self.router
            .metrics()
            .record_model_stage(&model, "affinity", policy.affinity);
        self.router
            .metrics()
            .record_model_stage(&model, "trie", policy.trie);
        let metric_model = canonical_model.to_string();
        let pseudo_stride = config.routing.pseudo_stride;
        let block_size = config.routing.block_size;
        let prompt_hash_fallback = config.session.prompt_hash_fallback.clone();
        let mut endpoints = self.models.endpoints_for_model(canonical_model);
        if let Some(configured) = config.configured_candidates(canonical_model) {
            endpoints.retain(|endpoint| configured.contains(endpoint));
        }
        endpoints.retain(|endpoint| eligible_endpoints.contains(endpoint));
        let allowed_worker_ids = Some(self.table.workers_in_endpoints(&endpoints));
        if allowed_worker_ids
            .as_ref()
            .is_some_and(std::collections::HashSet::is_empty)
        {
            return Err(ScheduleError::NoRoutableEndpoint(format!(
                "no live endpoint is configured for model {canonical_model}"
            )));
        }
        drop(config);

        let stable_identity = supplied_session_id.is_some();
        let mut body = Some(body);
        let mut tokenization = None;
        let (sid, identity_source) = if let Some(sid) = supplied_session_id {
            (
                sid,
                if session_header_present {
                    "header"
                } else {
                    "openai_user"
                },
            )
        } else {
            let tokens = self
                .tokenize_owned(
                    model.clone(),
                    metric_model.clone(),
                    request_path.clone(),
                    body.take().expect("request body available"),
                    pseudo_stride,
                    None,
                )
                .await?;
            let sid = prompt_hash_fallback.as_ref().and_then(|fallback| {
                prompt_hash_session_id(&model, &tokens.tokens, block_size, fallback.token_position)
            });
            tokenization = Some(tokens);
            match sid {
                Some(sid) => (sid, "prompt_hash"),
                None => (mint_session_id(), "minted"),
            }
        };
        self.router
            .metrics()
            .record_session_identity(identity_source);

        let affinity_backend = self.affinity.backend_name();
        let mut affinity_fallback = None;
        let bound = if !policy.affinity {
            None
        } else {
            match self.affinity.peek_binding(&sid).await {
                Ok(None) => {
                    self.router
                        .metrics()
                        .record_affinity_lookup(affinity_backend, "miss");
                    None
                }
                Ok(Some(binding)) if !self.liveness.is_alive(binding.worker.worker_id) => {
                    self.router
                        .metrics()
                        .record_affinity_lookup(affinity_backend, "worker_unavailable");
                    affinity_fallback = Some(("worker_unavailable", binding.endpoint_id));
                    None
                }
                Ok(Some(binding))
                    if allowed_worker_ids
                        .as_ref()
                        .is_some_and(|allowed| !allowed.contains(&binding.worker.worker_id)) =>
                {
                    self.router
                        .metrics()
                        .record_affinity_lookup(affinity_backend, "ineligible");
                    affinity_fallback = Some(("ineligible", binding.endpoint_id));
                    None
                }
                Ok(Some(binding)) => {
                    self.router
                        .metrics()
                        .record_affinity_lookup(affinity_backend, "hit");
                    Some(binding.worker)
                }
                Err(error) => {
                    self.router
                        .metrics()
                        .record_affinity_lookup(affinity_backend, "backend_error");
                    tracing::warn!(
                        %error,
                        backend = affinity_backend,
                        "affinity lookup failed; falling through to normal routing"
                    );
                    None
                }
            }
        };
        if let Some(worker) = bound {
            let (endpoint_id, endpoint) = self.endpoint_for(worker)?;
            let authority = downstream_authority(&endpoint);
            let metadata = RequestMetadata::new(
                sid.clone(),
                endpoint_id.clone(),
                model.clone(),
                authority.clone(),
                worker,
                policy,
            );

            // Only identities known before tokenization can take this path.
            // A permit is reserved before publishing the route, so optimistic
            // work can never form an unbounded body-retaining queue.
            if stable_identity
                && let Ok(permit) = self.tokenization_permits.clone().try_acquire_owned()
            {
                {
                    let mut state = slot.state.lock().await;
                    state.metadata = Some(metadata);
                    state.tokenization_pending = true;
                }
                self.router.metrics().optimistic_tokenization_started();
                self.spawn_optimistic_tokenization(OptimisticTokenizationJob {
                    rid: rid.to_string(),
                    slot,
                    model: model.clone(),
                    metric_model,
                    request_path,
                    body: body.take().expect("stable-identity body available"),
                    pseudo_stride,
                    permit,
                });
                self.router.metrics().record_routing_decision(
                    &endpoint_id,
                    &model,
                    &authority,
                    true,
                );
                self.router
                    .metrics()
                    .record_session_routing_decision(identity_source, true);
                self.router.metrics().observe_schedule(
                    "optimistic_affinity",
                    "success",
                    schedule_started.elapsed(),
                );
                return Ok(Scheduled {
                    session_id: sid,
                    model,
                    worker,
                    sticky: true,
                    affine_worker_id: Some(worker.worker_id),
                    endpoint_id,
                    endpoint,
                });
            }
            if stable_identity {
                self.router
                    .metrics()
                    .record_optimistic_tokenization("saturated_fallback");
            }

            let tokenization = match tokenization {
                Some(tokenization) => tokenization,
                None => {
                    self.tokenize_owned(
                        model.clone(),
                        metric_model,
                        request_path,
                        body.take().expect("request body available"),
                        pseudo_stride,
                        None,
                    )
                    .await?
                }
            };
            let tokens = tokenization.tokens;
            let selection = self
                .router
                .book_pinned(rid, &tokens, worker, policy)
                .await
                .map_err(|error| ScheduleError::NoRoutableEndpoint(error.to_string()))?;
            self.activate_booking(
                &slot,
                metadata,
                rid.to_string(),
                selection.worker,
                ScheduledTokenLoad::new(tokens, selection.cached_tokens),
            )
            .await;
            self.router
                .metrics()
                .record_routing_decision(&endpoint_id, &model, &authority, true);
            self.router
                .metrics()
                .record_session_routing_decision(identity_source, true);
            self.router.metrics().observe_schedule(
                "synchronous_affinity",
                "success",
                schedule_started.elapsed(),
            );
            return Ok(Scheduled {
                session_id: sid,
                model,
                worker: selection.worker,
                sticky: true,
                affine_worker_id: Some(selection.worker.worker_id),
                endpoint_id,
                endpoint,
            });
        }

        let tokenization = match tokenization {
            Some(tokenization) => tokenization,
            None => {
                self.tokenize_owned(
                    model.clone(),
                    metric_model,
                    request_path,
                    body.take().expect("request body available"),
                    pseudo_stride,
                    None,
                )
                .await?
            }
        };
        let tokens = tokenization.tokens;
        let selection = self
            .router
            .pick(rid, &tokens, allowed_worker_ids, policy)
            .await
            .map_err(|e| ScheduleError::NoRoutableEndpoint(e.to_string()))?;
        let (endpoint_id, endpoint) = match self.endpoint_for(selection.worker) {
            Ok(endpoint) => endpoint,
            Err(error) => {
                self.router.free(rid).await;
                return Err(error);
            }
        };
        let authority = downstream_authority(&endpoint);
        if let Some((reason, previous_endpoint)) = affinity_fallback.as_ref() {
            self.router.metrics().record_affinity_fallback_routing(
                affinity_backend,
                reason,
                previous_endpoint.as_ref(),
                &endpoint_id,
            );
        }
        self.activate_booking(
            &slot,
            RequestMetadata::new(
                sid.clone(),
                endpoint_id.clone(),
                model.clone(),
                authority.clone(),
                selection.worker,
                policy,
            ),
            rid.to_string(),
            selection.worker,
            ScheduledTokenLoad::new(tokens, selection.cached_tokens),
        )
        .await;
        self.router
            .metrics()
            .record_routing_decision(&endpoint_id, &model, &authority, false);
        self.router
            .metrics()
            .record_session_routing_decision(identity_source, false);
        self.router.metrics().observe_schedule(
            "scored_selection",
            "success",
            schedule_started.elapsed(),
        );
        Ok(Scheduled {
            session_id: sid,
            model,
            worker: selection.worker,
            sticky: false,
            affine_worker_id: None,
            endpoint_id,
            endpoint,
        })
    }

    async fn tokenize_owned(
        &self,
        model: String,
        metric_model: String,
        request_path: String,
        body: serde_json::Value,
        pseudo_stride: usize,
        permit: Option<tokio::sync::OwnedSemaphorePermit>,
    ) -> Result<ApproxTokens, ScheduleError> {
        let permit = match permit {
            Some(permit) => permit,
            None => self
                .tokenization_permits
                .clone()
                .acquire_owned()
                .await
                .map_err(|error| ScheduleError::Internal(error.to_string()))?,
        };
        let tokenizers = self.tokenizers.clone();
        #[cfg(test)]
        if let Some(gate) = self.tokenization_gate.as_ref() {
            gate.acquire()
                .await
                .map_err(|error| ScheduleError::Internal(error.to_string()))?
                .forget();
        }
        let tokenizer_mode = tokenizers.mode_for(&model);
        let started = std::time::Instant::now();
        let result = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            tokenizers.tokenize(&model, &request_path, &body, pseudo_stride)
        })
        .await;
        match result {
            Ok(Ok(tokenization)) => {
                self.router.metrics().observe_tokenization(
                    &metric_model,
                    tokenization.tier.mode(),
                    "success",
                    started.elapsed(),
                );
                Ok(tokenization)
            }
            Ok(Err(error)) => {
                self.router.metrics().observe_tokenization(
                    &metric_model,
                    tokenizer_mode,
                    "error",
                    started.elapsed(),
                );
                Err(error.into())
            }
            Err(error) => {
                self.router.metrics().observe_tokenization(
                    &metric_model,
                    tokenizer_mode,
                    "task_error",
                    started.elapsed(),
                );
                Err(ScheduleError::Internal(error.to_string()))
            }
        }
    }

    fn spawn_optimistic_tokenization(&self, job: OptimisticTokenizationJob) {
        let core = self.clone();
        tokio::spawn(async move {
            let result = core
                .tokenize_owned(
                    job.model,
                    job.metric_model,
                    job.request_path,
                    job.body,
                    job.pseudo_stride,
                    Some(job.permit),
                )
                .await;
            core.complete_optimistic_tokenization(&job.rid, &job.slot, result)
                .await;
        });
    }

    async fn activate_booking(
        &self,
        slot: &Arc<RequestSlot>,
        metadata: RequestMetadata,
        accounting_rid: String,
        decision: WorkerWithDpRank,
        load: ScheduledTokenLoad,
    ) {
        tracing::debug!(
            rid = %accounting_rid,
            sid = %metadata.sid,
            worker = decision.worker_id,
            "scheduled"
        );
        let endpoint_id = metadata.endpoint_id.clone();
        let total_tokens = load.tokens.len();
        let cached_tokens = load.cached_tokens;
        let mut state = slot.state.lock().await;
        state.metadata = Some(metadata);
        state.booking = Some(Booking {
            decision,
            tokens: load.tokens,
            cached_tokens,
            accounting_rid,
        });
        drop(state);
        self.router
            .metrics()
            .scheduler_request_started(&endpoint_id, total_tokens, cached_tokens);
    }

    async fn complete_optimistic_tokenization(
        &self,
        rid: &str,
        slot: &Arc<RequestSlot>,
        result: Result<ApproxTokens, ScheduleError>,
    ) {
        let mut state = slot.state.lock().await;
        state.tokenization_pending = false;
        let Some(metadata) = state.metadata.clone() else {
            tracing::error!(
                rid,
                "optimistic tokenization completed without route metadata"
            );
            self.router
                .metrics()
                .optimistic_tokenization_finished("internal_error");
            return;
        };
        let tokenization = match result {
            Ok(tokenization) => tokenization,
            Err(error) => {
                tracing::warn!(rid, %error, "optimistic tokenization failed; request remains unbooked");
                self.router
                    .metrics()
                    .optimistic_tokenization_finished("tokenization_error");
                let remove = state.finished_at.is_some() && state.terminal_outcome_recorded;
                drop(state);
                if remove {
                    self.remove_slot_if_same(rid, slot);
                }
                return;
            }
        };
        if state
            .response
            .is_some_and(|response| response.status != 200)
        {
            tracing::debug!(rid, "upstream denied request before optimistic booking");
            self.router
                .metrics()
                .optimistic_tokenization_finished("denied_before_booking");
            let remove = state.terminal_outcome_recorded;
            drop(state);
            if remove {
                self.remove_slot_if_same(rid, slot);
            }
            return;
        }
        if state.finished_at.is_some() {
            tracing::debug!(rid, "request finished before optimistic booking");
            self.router
                .metrics()
                .optimistic_tokenization_finished("finished_before_booking");
            let remove = state.terminal_outcome_recorded;
            drop(state);
            if remove {
                self.remove_slot_if_same(rid, slot);
            }
            return;
        }

        let decision = state
            .response
            .map(|response| response.confirmed)
            .unwrap_or(metadata.planned);
        let tokens = tokenization.tokens;
        let selection = match self
            .router
            .book_pinned(rid, &tokens, decision, metadata.policy)
            .await
        {
            Ok(selection) => selection,
            Err(error) => {
                tracing::warn!(rid, %error, "optimistic scheduler booking failed");
                self.router
                    .metrics()
                    .optimistic_tokenization_finished("booking_error");
                let remove = state.finished_at.is_some() && state.terminal_outcome_recorded;
                drop(state);
                if remove {
                    self.remove_slot_if_same(rid, slot);
                }
                return;
            }
        };
        let cached_tokens = selection.cached_tokens;
        state.booking = Some(Booking {
            decision: selection.worker,
            tokens,
            cached_tokens,
            accounting_rid: rid.to_string(),
        });
        self.router.metrics().scheduler_request_started(
            &metadata.endpoint_id,
            state.booking.as_ref().unwrap().tokens.len(),
            cached_tokens,
        );

        if state
            .response
            .is_some_and(|response| response.status == 200)
        {
            let booking = state.booking.as_ref().expect("booking installed");
            self.router
                .mark_prefill_completed(&booking.accounting_rid)
                .await;
            if metadata.policy.trie {
                self.router
                    .record_routing_decision(
                        &booking.accounting_rid,
                        &booking.tokens,
                        booking.decision,
                    )
                    .await;
            }
        }
        self.router
            .metrics()
            .optimistic_tokenization_finished("booked");
    }

    /// Lifecycle step 2 (Envoy `encodeHeaders` / ResponseStarted).
    ///
    /// - A non-200 response immediately releases the provisional booking and
    ///   is never re-booked.
    /// - On 200, re-book the request on the downstream worker reported by
    ///   `x-baseten-dyn-worker-id` when it differs from the provisional pick.
    /// - Mark prefill completed on the confirmed booking.
    /// - On HTTP 200, attribute the prefix and bind the session to the
    ///   confirmed worker. The etcd write is spawned off-path so it never
    ///   delays the stream.
    ///
    /// Unknown `rid` (janitor already swept it, or a replica-pinning bug) is a
    /// warn + no-op.
    pub async fn response_started(
        &self,
        rid: &str,
        actual_worker_remote: Option<u64>,
        status: u16,
    ) {
        let Some(slot) = self.inflight.get(rid).map(|entry| entry.clone()) else {
            tracing::warn!(
                rid,
                "response_started for unknown request id (swept or mis-pinned)"
            );
            return;
        };
        let mut state = slot.state.lock().await;
        let Some(metadata) = state.metadata.clone() else {
            tracing::warn!(rid, "response_started before scheduling produced metadata");
            return;
        };
        if state.response.is_some() {
            tracing::debug!(rid, "duplicate response_started ignored");
            return;
        }
        if state.finished_at.is_some() {
            self.router
                .metrics()
                .record_lifecycle_reconciliation("response_after_finish");
        }

        self.router.metrics().record_upstream_response(
            &metadata.endpoint_id,
            &metadata.model,
            &metadata.downstream_authority,
            status,
        );
        let outcome = ResponseOutcome::from_status(status);
        self.router.metrics().observe_time_to_first_byte(
            &metadata.endpoint_id,
            &metadata.model,
            &metadata.downstream_authority,
            outcome,
            metadata.routed_at.elapsed(),
        );

        let mut confirmed = metadata.planned;
        if status == 200 {
            if let Some(actual_worker_id) = actual_worker_remote {
                let actual = WorkerWithDpRank::from_worker_id(actual_worker_id);
                let valid_owner = match self.table.get(actual_worker_id) {
                    Some(owner) if owner == metadata.endpoint_id => true,
                    Some(owner) => {
                        tracing::error!(
                            rid,
                            actual_worker_id,
                            expected_endpoint = %metadata.endpoint_id.0,
                            advertised_endpoint = %owner.0,
                            "refusing cross-endpoint worker reassignment"
                        );
                        false
                    }
                    None => match self
                        .table
                        .upsert_worker(actual_worker_id, metadata.endpoint_id.clone())
                    {
                        Ok(()) => {
                            // A successful response is positive liveness evidence.
                            // The next planner snapshot remains authoritative.
                            self.liveness.insert(actual_worker_id);
                            true
                        }
                        Err(error) => {
                            tracing::error!(rid, actual_worker_id, %error);
                            false
                        }
                    },
                };
                if valid_owner {
                    confirmed = actual;
                }
            } else {
                tracing::error!(
                    rid,
                    "successful response missing required x-baseten-dyn-worker-id"
                );
            }
        }
        state.response = Some(ResponseStarted {
            status,
            outcome,
            confirmed,
        });

        if status != 200 {
            self.router.metrics().record_denied(
                &metadata.endpoint_id,
                &metadata.model,
                &metadata.downstream_authority,
                status,
            );
            if let Some(booking) = state.booking.take() {
                self.router.free(&booking.accounting_rid).await;
                self.router.metrics().scheduler_request_finished(
                    &metadata.endpoint_id,
                    booking.tokens.len(),
                    booking.cached_tokens,
                );
            }
            state
                .finished_at
                .get_or_insert_with(tokio::time::Instant::now);
            self.mark_inactive(&mut state);
            tracing::info!(
                rid,
                status,
                endpoint = %metadata.endpoint_id.0,
                "upstream denied request; released scheduler booking"
            );
        } else if let Some(mut booking) = state.booking.take() {
            let rebooked = confirmed != booking.decision;
            if rebooked {
                let confirmed_rid = format!("{rid}:confirmed:{}", confirmed.worker_id);
                self.router
                    .rebook_request(
                        &booking.accounting_rid,
                        &confirmed_rid,
                        &booking.tokens,
                        booking.cached_tokens,
                        confirmed,
                    )
                    .await;
                tracing::info!(
                    rid,
                    provisional_worker_id = booking.decision.worker_id,
                    actual_worker_id = confirmed.worker_id,
                    "re-booked request on downstream worker"
                );
                self.router.metrics().record_rebooked(
                    &metadata.endpoint_id,
                    &metadata.model,
                    &metadata.downstream_authority,
                );
                booking.decision = confirmed;
                booking.accounting_rid = confirmed_rid;
            }
            self.router
                .mark_prefill_completed(&booking.accounting_rid)
                .await;
            if metadata.policy.trie {
                self.router
                    .record_routing_decision(
                        &booking.accounting_rid,
                        &booking.tokens,
                        booking.decision,
                    )
                    .await;
            }
            if state.finished_at.is_some() {
                // RequestFinished arrived first. Preserve the lifecycle order
                // at the scheduler: mark prefill/reconcile the actual worker
                // above, then release the confirmed booking immediately.
                self.router.free(&booking.accounting_rid).await;
                self.router.metrics().scheduler_request_finished(
                    &metadata.endpoint_id,
                    booking.tokens.len(),
                    booking.cached_tokens,
                );
            } else {
                state.booking = Some(booking);
            }
        }

        self.record_terminal_outcome_locked(&mut state, false);
        let remove = state.finished_at.is_some()
            && !state.tokenization_pending
            && state.terminal_outcome_recorded;
        drop(state);

        if status == 200 && metadata.policy.affinity {
            let affinity = self.affinity.clone();
            let ttl = Duration::from_secs(self.config.load().session.ttl_secs);
            let binding = AffinityBinding::with_endpoint(confirmed, metadata.endpoint_id.clone());
            let sid = metadata.sid.clone();
            tokio::spawn(async move {
                if let Err(error) = affinity.put_binding(&sid, binding, ttl).await {
                    tracing::warn!(
                        %error,
                        backend = affinity.backend_name(),
                        "affinity put failed; binding not recorded"
                    );
                }
            });
        }
        if remove {
            self.remove_slot_if_same(rid, &slot);
        }
    }

    /// Lifecycle step 3 (Envoy Wasm `onLog` / RequestFinished): release the
    /// scheduler slot and drop the in-flight entry. Idempotent.
    pub async fn request_finished(&self, rid: &str, reason: &str) {
        let Some(slot) = self.inflight.get(rid).map(|entry| entry.clone()) else {
            tracing::debug!(rid, reason, known = false, "request finished");
            return;
        };
        let mut state = slot.state.lock().await;
        if state.finished_at.is_some() {
            tracing::debug!(rid, reason, "duplicate request_finished ignored");
            return;
        }
        state.finished_at = Some(tokio::time::Instant::now());
        let wait_for_response = state.response.is_none();
        if !wait_for_response && let Some(booking) = state.booking.take() {
            self.router.free(&booking.accounting_rid).await;
            if let Some(metadata) = state.metadata.as_ref() {
                self.router.metrics().scheduler_request_finished(
                    &metadata.endpoint_id,
                    booking.tokens.len(),
                    booking.cached_tokens,
                );
            }
        }
        self.mark_inactive(&mut state);
        self.record_terminal_outcome_locked(&mut state, false);
        let remove = !state.tokenization_pending && state.terminal_outcome_recorded;
        drop(state);
        if remove {
            self.remove_slot_if_same(rid, &slot);
        } else if wait_for_response {
            self.spawn_terminal_cleanup(rid.to_string(), slot.clone());
        }
        tracing::debug!(rid, reason, known = true, "request finished");
    }

    /// Number of requests currently between `schedule` and `request_finished`.
    pub fn inflight_len(&self) -> usize {
        self.active_inflight.load(Ordering::Acquire)
    }

    /// Release every locally owned booking. Used only after the graceful
    /// shutdown deadline, when Envoy completion callbacks can no longer be
    /// awaited. Concurrent late callbacks are harmless: `request_finished`
    /// is intentionally idempotent for unknown request IDs.
    pub async fn force_finish_all(&self, reason: &str) {
        let request_ids: Vec<String> = self
            .inflight
            .iter()
            .map(|entry| entry.key().clone())
            .collect();
        for request_id in request_ids {
            self.request_finished(&request_id, reason).await;
            self.force_terminal_cleanup(&request_id).await;
        }
    }

    fn mark_inactive(&self, state: &mut RequestState) {
        if !state.counted_inflight {
            return;
        }
        state.counted_inflight = false;
        let previous = self.active_inflight.fetch_sub(1, Ordering::AcqRel);
        let active = previous.saturating_sub(1);
        self.router.metrics().set_inflight(active);
    }

    fn record_terminal_outcome_locked(&self, state: &mut RequestState, force: bool) {
        if state.terminal_outcome_recorded {
            return;
        }
        let (Some(metadata), Some(finished_at)) = (state.metadata.as_ref(), state.finished_at)
        else {
            return;
        };
        let outcome = match state.response {
            Some(response) => response.outcome,
            None if force => ResponseOutcome::UpstreamError,
            None => return,
        };
        let elapsed = finished_at
            .checked_duration_since(metadata.routed_at)
            .unwrap_or_default();
        self.router.metrics().record_request_outcome(
            &metadata.endpoint_id,
            &metadata.model,
            &metadata.downstream_authority,
            outcome,
            elapsed,
        );
        state.terminal_outcome_recorded = true;
    }

    fn remove_slot_if_same(&self, rid: &str, slot: &Arc<RequestSlot>) {
        if let Entry::Occupied(entry) = self.inflight.entry(rid.to_string())
            && Arc::ptr_eq(entry.get(), slot)
        {
            entry.remove();
        }
    }

    fn spawn_terminal_cleanup(&self, rid: String, slot: Arc<RequestSlot>) {
        let core = self.clone();
        tokio::spawn(async move {
            tokio::time::sleep(TERMINAL_EVENT_GRACE).await;
            let mut state = slot.state.lock().await;
            if state.response.is_none() {
                core.router
                    .metrics()
                    .record_lifecycle_reconciliation("response_missing_at_grace");
            }
            if let Some(booking) = state.booking.take() {
                core.router.free(&booking.accounting_rid).await;
                if let Some(metadata) = state.metadata.as_ref() {
                    core.router.metrics().scheduler_request_finished(
                        &metadata.endpoint_id,
                        booking.tokens.len(),
                        booking.cached_tokens,
                    );
                }
            }
            core.record_terminal_outcome_locked(&mut state, true);
            let remove = !state.tokenization_pending;
            drop(state);
            if remove {
                core.remove_slot_if_same(&rid, &slot);
            }
        });
    }

    async fn force_terminal_cleanup(&self, rid: &str) {
        let Some(slot) = self.inflight.get(rid).map(|entry| entry.clone()) else {
            return;
        };
        let mut state = slot.state.lock().await;
        state
            .finished_at
            .get_or_insert_with(tokio::time::Instant::now);
        if let Some(booking) = state.booking.take() {
            self.router.free(&booking.accounting_rid).await;
            if let Some(metadata) = state.metadata.as_ref() {
                self.router.metrics().scheduler_request_finished(
                    &metadata.endpoint_id,
                    booking.tokens.len(),
                    booking.cached_tokens,
                );
            }
        }
        self.mark_inactive(&mut state);
        self.record_terminal_outcome_locked(&mut state, true);
        drop(state);
        self.remove_slot_if_same(rid, &slot);
    }

    /// Sweep in-flight entries whose `RequestFinished` never arrived. Holds
    /// only weak refs so dropping the core (tests, shutdown) ends the task.
    pub fn spawn_janitor(&self) -> tokio::task::JoinHandle<()> {
        let inflight = Arc::downgrade(&self.inflight);
        let active_inflight = Arc::downgrade(&self.active_inflight);
        let router = Arc::downgrade(&self.router);
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(JANITOR_INTERVAL);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                ticker.tick().await;
                let (Some(inflight), Some(active_inflight), Some(router)) = (
                    inflight.upgrade(),
                    active_inflight.upgrade(),
                    router.upgrade(),
                ) else {
                    return;
                };
                let now = tokio::time::Instant::now();
                let expired: Vec<String> = inflight
                    .iter()
                    .filter(|e| now.duration_since(e.value().created) > INFLIGHT_MAX_AGE)
                    .map(|e| e.key().clone())
                    .collect();
                for rid in expired {
                    let Some((_, slot)) = inflight.remove(&rid) else {
                        continue;
                    };
                    tracing::warn!(
                        rid,
                        "in-flight entry expired without RequestFinished; freeing"
                    );
                    let mut state = slot.state.lock().await;
                    state.finished_at.get_or_insert(now);
                    if let Some(booking) = state.booking.take() {
                        router.free(&booking.accounting_rid).await;
                        if let Some(metadata) = state.metadata.as_ref() {
                            router.metrics().scheduler_request_finished(
                                &metadata.endpoint_id,
                                booking.tokens.len(),
                                booking.cached_tokens,
                            );
                        }
                    }
                    if !state.terminal_outcome_recorded
                        && let Some(metadata) = state.metadata.as_ref()
                    {
                        router.metrics().record_request_outcome(
                            &metadata.endpoint_id,
                            &metadata.model,
                            &metadata.downstream_authority,
                            state
                                .response
                                .map(|response| response.outcome)
                                .unwrap_or(ResponseOutcome::UpstreamError),
                            state
                                .finished_at
                                .and_then(|finished| {
                                    finished.checked_duration_since(metadata.routed_at)
                                })
                                .unwrap_or_default(),
                        );
                        state.terminal_outcome_recorded = true;
                    }
                    if state.counted_inflight {
                        state.counted_inflight = false;
                        let previous = active_inflight.fetch_sub(1, Ordering::AcqRel);
                        router.metrics().set_inflight(previous.saturating_sub(1));
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        EndpointId, GwpConfig, ModelRoute, PromptHashFallbackConfig, RoutingConfig,
    };
    use crate::session::{AffinityStore, InMemoryAffinityStore, InstrumentedAffinityStore};
    use dynamo_llm::local_model::runtime_config::ModelRuntimeConfig;
    use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

    const ENDPOINT: &str = "core-endpoint";
    const WORKER: u64 = 10;
    const CHAT_PATH: &str = "/v1/chat/completions";

    fn chat_body() -> serde_json::Value {
        serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "core lifecycle test prompt"}],
        })
    }

    fn metric_has(metrics: &str, name: &str, labels: &[&str]) -> bool {
        metrics
            .lines()
            .any(|line| line.starts_with(name) && labels.iter().all(|label| line.contains(label)))
    }

    fn metric_value(metrics: &str, name: &str, labels: &[&str]) -> Option<f64> {
        metrics.lines().find_map(|line| {
            (line.starts_with(name) && labels.iter().all(|label| line.contains(label)))
                .then(|| line.split_whitespace().last()?.parse().ok())
                .flatten()
        })
    }

    struct FailingAffinityStore;

    struct HangingAffinityStore;

    #[async_trait::async_trait]
    impl AffinityStore for FailingAffinityStore {
        fn backend_name(&self) -> &'static str {
            "redis"
        }

        async fn peek(&self, _session_id: &str) -> anyhow::Result<Option<WorkerWithDpRank>> {
            anyhow::bail!("test peek failure")
        }

        async fn get(&self, _session_id: &str) -> anyhow::Result<Option<WorkerWithDpRank>> {
            anyhow::bail!("test get failure")
        }

        async fn put(
            &self,
            _session_id: &str,
            _worker: WorkerWithDpRank,
            _ttl: Duration,
        ) -> anyhow::Result<()> {
            anyhow::bail!("test put failure")
        }

        async fn remove(&self, _session_id: &str) -> anyhow::Result<bool> {
            anyhow::bail!("test remove failure")
        }
    }

    #[async_trait::async_trait]
    impl AffinityStore for HangingAffinityStore {
        fn backend_name(&self) -> &'static str {
            "redis"
        }

        async fn peek(&self, _session_id: &str) -> anyhow::Result<Option<WorkerWithDpRank>> {
            std::future::pending().await
        }

        async fn get(&self, _session_id: &str) -> anyhow::Result<Option<WorkerWithDpRank>> {
            std::future::pending().await
        }

        async fn put(
            &self,
            _session_id: &str,
            _worker: WorkerWithDpRank,
            _ttl: Duration,
        ) -> anyhow::Result<()> {
            std::future::pending().await
        }

        async fn remove(&self, _session_id: &str) -> anyhow::Result<bool> {
            std::future::pending().await
        }
    }

    #[test]
    fn parses_dimensioned_routing_requirements() {
        let requirements = parse_routing_requirements(Some(
            r#"{"region":{"required":["us","canada"],"preferred":["us-east"]},"cloud":{"required":["gcp"]}}"#,
        ))
        .unwrap();
        assert_eq!(
            requirements.0["region"].required,
            BTreeSet::from(["canada".into(), "us".into()])
        );
        assert_eq!(
            requirements.0["region"].preferred,
            BTreeSet::from(["us-east".into()])
        );
        assert_eq!(
            requirements.0["cloud"].required,
            BTreeSet::from(["gcp".into()])
        );
    }

    #[test]
    fn rejects_invalid_routing_requirement_shapes() {
        for raw in [
            r#"["us"]"#,
            r#"{"region":{"required":[""]}}"#,
            r#"{"":{"required":["us"]}}"#,
            r#"{"region":{"unknown":["us"]}}"#,
        ] {
            assert!(
                matches!(
                    parse_routing_requirements(Some(raw)),
                    Err(ScheduleError::RoutingRequirements(_))
                ),
                "unexpectedly accepted {raw}"
            );
        }
    }

    /// A core wired by hand, as one reflector poll would.
    async fn make_core(live: bool) -> (GwpCore, crate::router::WorkerConfigSender) {
        let config = GwpConfig {
            endpoints: BTreeMap::from([(
                EndpointId(ENDPOINT.into()),
                EndpointConfig {
                    ingress_url: url::Url::parse("https://endpoint.example.com:8443/v1").unwrap(),
                    api_key: "ck".into(),
                    planner_url: url::Url::parse("http://localhost:0/deep/health").unwrap(),
                    planner_api_key: None,
                    properties: Default::default(),
                },
            )]),
            routes: vec![ModelRoute {
                models: vec!["m".into()],
                endpoints: vec![EndpointId(ENDPOINT.into())],
            }],
            routing: RoutingConfig {
                block_size: 4,
                ..Default::default()
            },
            ..Default::default()
        };
        let (router, workers_tx) = GwpRouter::new_process_local(
            config.routing.block_size,
            config.routing.approx_indexer_ttl_secs,
        )
        .await
        .unwrap();
        let table = EndpointTable::new();
        let liveness = LivenessSet::new();
        let mut live_ids = HashSet::new();
        let mut configs = HashMap::new();
        if live {
            table
                .upsert_worker(WORKER, EndpointId(ENDPOINT.into()))
                .unwrap();
            live_ids.insert(WORKER);
            configs.insert(WORKER, ModelRuntimeConfig::default());
        }
        liveness.replace(live_ids);
        workers_tx.send(configs).unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;

        let core = GwpCore::new(
            router,
            Arc::new(InMemoryAffinityStore::new()),
            liveness,
            table,
            Arc::new(config),
        );
        (core, workers_tx)
    }

    async fn make_multi_model_core() -> (GwpCore, crate::router::WorkerConfigSender) {
        let endpoint = |host: &str| EndpointConfig {
            ingress_url: url::Url::parse(&format!("http://{host}/v1")).unwrap(),
            api_key: String::new(),
            planner_url: url::Url::parse(&format!("http://{host}/deep/health")).unwrap(),
            planner_api_key: None,
            properties: Default::default(),
        };
        let config = GwpConfig {
            endpoints: BTreeMap::from([
                (EndpointId("kimi-a".into()), endpoint("kimi-a")),
                (EndpointId("glm-a".into()), endpoint("glm-a")),
                (EndpointId("glm-b".into()), endpoint("glm-b")),
            ]),
            routes: vec![
                ModelRoute {
                    models: vec!["kimi-k2".into()],
                    endpoints: vec![EndpointId("kimi-a".into())],
                },
                ModelRoute {
                    models: vec!["glm-4.7".into()],
                    endpoints: vec![EndpointId("glm-a".into()), EndpointId("glm-b".into())],
                },
            ],
            routing: RoutingConfig {
                block_size: 4,
                ..Default::default()
            },
            ..Default::default()
        };
        let (router, workers_tx) = GwpRouter::new_process_local(4, 120).await.unwrap();
        let table = EndpointTable::new();
        let liveness = LivenessSet::new();
        let mut live = HashSet::new();
        let mut configs = HashMap::new();
        for (offset, endpoint_id) in config.endpoints.keys().enumerate() {
            let worker = 100 + offset as u64;
            table.upsert_worker(worker, endpoint_id.clone()).unwrap();
            live.insert(worker);
            configs.insert(worker, ModelRuntimeConfig::default());
        }
        liveness.replace(live);
        workers_tx.send(configs).unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;
        let core = GwpCore::new(
            router,
            Arc::new(InMemoryAffinityStore::new()),
            liveness,
            table,
            config,
        );
        (core, workers_tx)
    }

    fn add_endpoint_properties(core: &GwpCore) {
        let mut config = (*core.config.load()).clone();
        let hippa = BTreeMap::from([
            ("region".into(), BTreeSet::from(["us".into()])),
            ("compliance".into(), BTreeSet::from(["hippa".into()])),
        ]);
        config
            .endpoints
            .get_mut(&EndpointId("glm-a".into()))
            .unwrap()
            .properties = hippa;
        let standard = BTreeMap::from([
            ("region".into(), BTreeSet::from(["canada".into()])),
            ("compliance".into(), BTreeSet::from(["standard".into()])),
        ]);
        config
            .endpoints
            .get_mut(&EndpointId("glm-b".into()))
            .unwrap()
            .properties = standard;
        core.config.replace(config);
    }

    async fn wait_for_binding(core: &GwpCore, sid: &str, worker_id: u64) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while tokio::time::Instant::now() < deadline {
            if core
                .affinity
                .peek(sid)
                .await
                .unwrap()
                .is_some_and(|w| w.worker_id == worker_id)
            {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("binding for {sid} never reached worker {worker_id}");
    }

    async fn wait_for_optimistic_job(core: &GwpCore, rid: &str) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while tokio::time::Instant::now() < deadline {
            let Some(slot) = core.inflight.get(rid).map(|entry| entry.clone()) else {
                return;
            };
            if !slot.state.lock().await.tokenization_pending {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("optimistic tokenization for {rid} did not finish");
    }

    async fn configure_blocked_affinity_hit(core: &mut GwpCore, sid: &str) -> Arc<Semaphore> {
        core.affinity
            .put(
                sid,
                WorkerWithDpRank::from_worker_id(WORKER),
                Duration::from_secs(60),
            )
            .await
            .unwrap();
        let gate = Arc::new(Semaphore::new(0));
        core.tokenization_gate = Some(gate.clone());
        gate
    }

    #[tokio::test]
    async fn affinity_hit_returns_before_background_tokenization() {
        let (mut core, _tx) = make_core(true).await;
        let sid = "optimistic-latency";
        let gate = configure_blocked_affinity_hit(&mut core, sid).await;

        let scheduled = tokio::time::timeout(
            Duration::from_millis(100),
            core.schedule_for_test(
                "optimistic-latency",
                Some(sid.into()),
                CHAT_PATH,
                &chat_body(),
            ),
        )
        .await
        .expect("affinity hit must not await tokenization")
        .unwrap();
        assert!(scheduled.sticky);
        let slot = core.inflight.get("optimistic-latency").unwrap().clone();
        let state = slot.state.lock().await;
        assert!(state.tokenization_pending);
        assert!(state.booking.is_none());
        drop(state);

        gate.add_permits(1);
        wait_for_optimistic_job(&core, "optimistic-latency").await;
        core.response_started("optimistic-latency", Some(WORKER), 200)
            .await;
        core.request_finished("optimistic-latency", "complete")
            .await;
        assert_eq!(core.inflight_len(), 0);
    }

    #[tokio::test]
    async fn cancelling_schedule_releases_reserved_slot() {
        let (mut core, _tx) = make_core(true).await;
        let gate = Arc::new(Semaphore::new(0));
        core.tokenization_gate = Some(gate);
        let scheduled_core = core.clone();
        let schedule = tokio::spawn(async move {
            scheduled_core
                .schedule_for_test("cancelled-schedule", None, CHAT_PATH, &chat_body())
                .await
        });

        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        while core.inflight_len() == 0 && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(core.inflight_len(), 1, "request was not reserved");
        schedule.abort();
        let _ = schedule.await;

        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        while core.inflight_len() != 0 && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(core.inflight_len(), 0, "cancelled reservation leaked");
    }

    #[tokio::test]
    async fn saturated_optimistic_pool_falls_back_to_synchronous_booking() {
        let (core, _tx) = make_core(true).await;
        let sid = "saturated-optimistic";
        core.affinity
            .put(
                sid,
                WorkerWithDpRank::from_worker_id(WORKER),
                Duration::from_secs(60),
            )
            .await
            .unwrap();
        let permits = core
            .tokenization_permits
            .clone()
            .acquire_many_owned(TOKENIZATION_MAX_IN_FLIGHT as u32)
            .await
            .unwrap();
        let scheduled_core = core.clone();
        let schedule = tokio::spawn(async move {
            scheduled_core
                .schedule_for_test(
                    "saturated-optimistic",
                    Some(sid.into()),
                    CHAT_PATH,
                    &chat_body(),
                )
                .await
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            !schedule.is_finished(),
            "saturated fallback must complete its booking before returning"
        );
        drop(permits);
        let scheduled = schedule.await.unwrap().unwrap();
        assert!(scheduled.sticky);
        let slot = core.inflight.get("saturated-optimistic").unwrap().clone();
        let state = slot.state.lock().await;
        assert!(!state.tokenization_pending);
        assert!(state.booking.is_some());
        drop(state);
        core.response_started("saturated-optimistic", Some(WORKER), 200)
            .await;
        core.request_finished("saturated-optimistic", "complete")
            .await;

        let metrics = core.router.prometheus_metrics().unwrap();
        assert!(metric_has(
            &metrics,
            "dynamo_component_gwp_optimistic_tokenization_total",
            &["result=\"saturated_fallback\""]
        ));
    }

    #[tokio::test]
    async fn response_before_tokenization_books_reported_worker_directly() {
        let (mut core, _tx) = make_core(true).await;
        let sid = "response-before-tokenization";
        let gate = configure_blocked_affinity_hit(&mut core, sid).await;
        core.schedule_for_test(
            "response-before-tokenization",
            Some(sid.into()),
            CHAT_PATH,
            &chat_body(),
        )
        .await
        .unwrap();

        core.response_started("response-before-tokenization", Some(20), 200)
            .await;
        gate.add_permits(1);
        wait_for_optimistic_job(&core, "response-before-tokenization").await;

        let slot = core
            .inflight
            .get("response-before-tokenization")
            .unwrap()
            .clone();
        let state = slot.state.lock().await;
        let booking = state.booking.as_ref().expect("late booking installed");
        assert_eq!(booking.decision.worker_id, 20);
        assert_eq!(booking.accounting_rid, "response-before-tokenization");
        drop(state);
        let tokens = pseudo_tokens(
            &routing_text(CHAT_PATH, &chat_body()),
            core.config.load().routing.pseudo_stride,
        );
        let active = core.router.potential_loads(&tokens).await;
        assert_eq!(
            active
                .iter()
                .find(|load| load.worker_id == 20)
                .map(|load| load.active_requests),
            Some(1)
        );
        assert!(
            active
                .iter()
                .all(|load| load.worker_id != WORKER || load.active_requests == 0)
        );

        core.request_finished("response-before-tokenization", "complete")
            .await;
        assert!(
            core.router
                .potential_loads(&tokens)
                .await
                .iter()
                .all(|load| load.active_requests == 0)
        );
    }

    #[tokio::test]
    async fn denial_before_tokenization_never_books() {
        let (mut core, _tx) = make_core(true).await;
        let gate = configure_blocked_affinity_hit(&mut core, "denied-before-tokenization").await;
        core.schedule_for_test(
            "denied-before-tokenization",
            Some("denied-before-tokenization".into()),
            CHAT_PATH,
            &chat_body(),
        )
        .await
        .unwrap();
        core.response_started("denied-before-tokenization", Some(20), 429)
            .await;
        gate.add_permits(1);
        wait_for_optimistic_job(&core, "denied-before-tokenization").await;
        assert_eq!(core.inflight_len(), 0);

        let tokens = pseudo_tokens(
            &routing_text(CHAT_PATH, &chat_body()),
            core.config.load().routing.pseudo_stride,
        );
        assert!(
            core.router
                .potential_loads(&tokens)
                .await
                .iter()
                .all(|load| load.active_requests == 0)
        );
        let metrics = core.router.prometheus_metrics().unwrap();
        assert!(metric_has(
            &metrics,
            "dynamo_component_gwp_optimistic_tokenization_total",
            &["result=\"denied_before_booking\""]
        ));
    }

    #[tokio::test]
    async fn finish_before_tokenization_never_resurrects_booking() {
        let (mut core, _tx) = make_core(true).await;
        let gate = configure_blocked_affinity_hit(&mut core, "finish-before-tokenization").await;
        core.schedule_for_test(
            "finish-before-tokenization",
            Some("finish-before-tokenization".into()),
            CHAT_PATH,
            &chat_body(),
        )
        .await
        .unwrap();
        core.request_finished("finish-before-tokenization", "complete")
            .await;
        gate.add_permits(1);
        wait_for_optimistic_job(&core, "finish-before-tokenization").await;
        assert_eq!(core.inflight_len(), 0);

        let tokens = pseudo_tokens(
            &routing_text(CHAT_PATH, &chat_body()),
            core.config.load().routing.pseudo_stride,
        );
        assert!(
            core.router
                .potential_loads(&tokens)
                .await
                .iter()
                .all(|load| load.active_requests == 0)
        );
    }

    #[tokio::test]
    async fn reversed_finish_and_response_are_reconciled_once() {
        let (mut core, _tx) = make_core(true).await;
        let sid = "reversed-lifecycle";
        let gate = configure_blocked_affinity_hit(&mut core, sid).await;
        core.schedule_for_test(
            "reversed-lifecycle",
            Some(sid.into()),
            CHAT_PATH,
            &chat_body(),
        )
        .await
        .unwrap();
        core.request_finished("reversed-lifecycle", "complete")
            .await;
        core.response_started("reversed-lifecycle", Some(20), 200)
            .await;
        gate.add_permits(1);
        wait_for_optimistic_job(&core, "reversed-lifecycle").await;
        wait_for_binding(&core, sid, 20).await;

        let metrics = core.router.prometheus_metrics().unwrap();
        assert_eq!(
            metric_value(
                &metrics,
                "dynamo_component_gwp_request_outcomes_total",
                &["outcome=\"success\""]
            ),
            Some(1.0)
        );
    }

    #[tokio::test]
    async fn finish_before_response_retains_booking_until_prefill_transition() {
        let (core, _tx) = make_core(true).await;
        let rid = "finish-before-response-booked";
        let body = chat_body();
        core.schedule_for_test(rid, None, CHAT_PATH, &body)
            .await
            .unwrap();

        core.request_finished(rid, "complete").await;
        let slot = core
            .inflight
            .get(rid)
            .map(|entry| entry.clone())
            .expect("terminal grace retains the request");
        assert!(
            slot.state.lock().await.booking.is_some(),
            "request-finished must not free before mark-prefill can run"
        );

        core.response_started(rid, Some(20), 200).await;
        assert!(
            !core.inflight.contains_key(rid),
            "response-started marks prefill, frees, and removes the finished request"
        );
        let metrics = core.router.prometheus_metrics().unwrap();
        assert_eq!(
            metric_value(
                &metrics,
                "dynamo_component_gwp_lifecycle_reconciliations_total",
                &["outcome=\"response_after_finish\""]
            ),
            Some(1.0)
        );
        let tokens = pseudo_tokens(
            &routing_text(CHAT_PATH, &body),
            core.config.load().routing.pseudo_stride,
        );
        assert!(
            core.router
                .potential_loads(&tokens)
                .await
                .iter()
                .all(|load| load.active_requests == 0)
        );
    }

    #[tokio::test(start_paused = true)]
    async fn missing_response_grace_frees_retained_booking_and_counts_it() {
        let (core, _tx) = make_core(true).await;
        let rid = "missing-response-grace";
        let body = chat_body();
        core.schedule_for_test(rid, None, CHAT_PATH, &body)
            .await
            .unwrap();
        core.request_finished(rid, "complete").await;

        tokio::task::yield_now().await;
        tokio::time::advance(TERMINAL_EVENT_GRACE).await;
        tokio::task::yield_now().await;

        assert!(!core.inflight.contains_key(rid));
        let tokens = pseudo_tokens(
            &routing_text(CHAT_PATH, &body),
            core.config.load().routing.pseudo_stride,
        );
        assert!(
            core.router
                .potential_loads(&tokens)
                .await
                .iter()
                .all(|load| load.active_requests == 0)
        );
        let metrics = core.router.prometheus_metrics().unwrap();
        assert_eq!(
            metric_value(
                &metrics,
                "dynamo_component_gwp_lifecycle_reconciliations_total",
                &["outcome=\"response_missing_at_grace\""]
            ),
            Some(1.0)
        );
    }

    #[tokio::test]
    async fn duplicate_schedule_id_is_rejected_without_overwrite() {
        let (core, _tx) = make_core(true).await;
        core.schedule_for_test("duplicate-rid", None, CHAT_PATH, &chat_body())
            .await
            .unwrap();
        let error = core
            .schedule_for_test("duplicate-rid", None, CHAT_PATH, &chat_body())
            .await
            .unwrap_err();
        assert!(matches!(error, ScheduleError::DuplicateRequestId(_)));
        assert_eq!(core.inflight_len(), 1);
        core.request_finished("duplicate-rid", "complete").await;
        assert_eq!(core.inflight_len(), 0);
    }

    #[tokio::test]
    async fn concurrent_response_rebook_and_finish_leave_no_slot() {
        let (core, _tx) = make_core(true).await;
        for index in 0..16 {
            let rid = format!("rebook-finish-race-{index}");
            core.schedule_for_test(&rid, None, CHAT_PATH, &chat_body())
                .await
                .unwrap();
            tokio::join!(
                core.response_started(&rid, Some(20), 200),
                core.request_finished(&rid, "complete")
            );
        }
        assert_eq!(core.inflight_len(), 0);
        let tokens = pseudo_tokens(
            &routing_text(CHAT_PATH, &chat_body()),
            core.config.load().routing.pseudo_stride,
        );
        assert!(
            core.router
                .potential_loads(&tokens)
                .await
                .iter()
                .all(|load| load.active_requests == 0)
        );
    }

    #[tokio::test]
    async fn full_lifecycle_binds_then_sticks_then_frees() {
        let (core, _tx) = make_core(true).await;
        core.router
            .metrics()
            .replace_scheduler_live_workers(HashMap::from([(EndpointId(ENDPOINT.into()), 2)]));

        // Schedule (new session): books a slot, records in-flight.
        let scheduled = core
            .schedule_for_test("rid-1", None, CHAT_PATH, &chat_body())
            .await
            .unwrap();
        assert!(scheduled.session_id.starts_with("base10-"));
        assert!(!scheduled.sticky);
        assert_eq!(scheduled.affine_worker_id, None);
        assert_eq!(core.inflight_len(), 1);
        assert_eq!(scheduled.endpoint_id, EndpointId(ENDPOINT.into()));
        let active_metrics = core.router.prometheus_metrics().unwrap();
        let requests = "dynamo_component_gwp_scheduler_inflight_requests";
        assert_eq!(
            metric_value(
                &active_metrics,
                requests,
                &["routed_endpoint=\"core-endpoint\"", "aggregation=\"total\""]
            ),
            Some(1.0)
        );
        assert_eq!(
            metric_value(
                &active_metrics,
                requests,
                &[
                    "routed_endpoint=\"core-endpoint\"",
                    "aggregation=\"per_live_worker\""
                ]
            ),
            Some(0.5)
        );
        let tokens = "dynamo_component_gwp_scheduler_inflight_tokens";
        assert_eq!(
            metric_value(
                &active_metrics,
                tokens,
                &[
                    "routed_endpoint=\"core-endpoint\"",
                    "cache_status=\"cached\"",
                    "aggregation=\"total\""
                ]
            ),
            Some(0.0)
        );
        assert!(
            metric_value(
                &active_metrics,
                tokens,
                &[
                    "routed_endpoint=\"core-endpoint\"",
                    "cache_status=\"uncached\"",
                    "aggregation=\"total\""
                ]
            )
            .is_some_and(|value| value > 0.0)
        );

        // ResponseStarted binds the session to the selected endpoint.
        core.response_started("rid-1", Some(WORKER), 200).await;
        wait_for_binding(&core, &scheduled.session_id, scheduled.worker.worker_id).await;

        // RequestFinished: slot + in-flight entry released.
        core.request_finished("rid-1", "complete").await;
        assert_eq!(core.inflight_len(), 0);
        let freed_metrics = core.router.prometheus_metrics().unwrap();
        assert_eq!(
            metric_value(
                &freed_metrics,
                requests,
                &["routed_endpoint=\"core-endpoint\"", "aggregation=\"total\""]
            ),
            Some(0.0)
        );

        // Turn 2 with the session id: sticks to the confirmed worker.
        let second = core
            .schedule_for_test(
                "rid-2",
                Some(scheduled.session_id.clone()),
                CHAT_PATH,
                &chat_body(),
            )
            .await
            .unwrap();
        assert!(second.sticky);
        assert_eq!(second.worker.worker_id, scheduled.worker.worker_id);
        assert_eq!(second.affine_worker_id, Some(scheduled.worker.worker_id));
        core.request_finished("rid-2", "complete").await;

        let metrics = core.router.prometheus_metrics().unwrap();
        let lookup = "dynamo_component_gwp_affinity_lookups_total";
        assert!(metric_has(
            &metrics,
            lookup,
            &["backend=\"memory\"", "outcome=\"miss\""]
        ));
        assert!(metric_has(
            &metrics,
            lookup,
            &["backend=\"memory\"", "outcome=\"hit\""]
        ));
        assert_eq!(
            metric_value(
                &metrics,
                "dynamo_component_gwp_session_routing_decisions_total",
                &["source=\"minted\"", "sticky=\"false\""]
            ),
            Some(1.0)
        );
        assert_eq!(
            metric_value(
                &metrics,
                "dynamo_component_gwp_session_routing_decisions_total",
                &["source=\"header\"", "sticky=\"true\""]
            ),
            Some(1.0)
        );
    }

    #[tokio::test]
    async fn model_policy_disables_affinity_lookup_and_write() {
        let (core, _tx) = make_core(true).await;
        let mut config = (*core.config.load()).clone();
        config.model_policies.models.insert(
            "m".into(),
            crate::config::ModelStagePolicyOverride {
                affinity: Some(false),
                trie: None,
                ..Default::default()
            },
        );
        core.config.replace(config);

        let sid = "disabled-affinity";
        let first = core
            .schedule_for_test("no-affinity-1", Some(sid.into()), CHAT_PATH, &chat_body())
            .await
            .unwrap();
        assert!(!first.sticky);
        core.response_started("no-affinity-1", Some(WORKER), 200)
            .await;
        core.request_finished("no-affinity-1", "complete").await;
        tokio::task::yield_now().await;
        assert_eq!(core.affinity.peek(sid).await.unwrap(), None);

        let second = core
            .schedule_for_test("no-affinity-2", Some(sid.into()), CHAT_PATH, &chat_body())
            .await
            .unwrap();
        assert!(!second.sticky);
        core.request_finished("no-affinity-2", "complete").await;

        let metrics = core.router.prometheus_metrics().unwrap();
        assert!(metric_has(
            &metrics,
            "dynamo_component_gwp_model_stage_requests_total",
            &["model=\"m\"", "stage=\"affinity\"", "enabled=\"false\""]
        ));
    }

    #[tokio::test]
    async fn model_policy_reload_affects_new_requests_but_not_inflight_lifecycle() {
        let (core, _tx) = make_core(true).await;
        let sid = "policy-snapshot";
        let first = core
            .schedule_for_test(
                "policy-before-reload",
                Some(sid.into()),
                CHAT_PATH,
                &chat_body(),
            )
            .await
            .unwrap();

        let mut config = (*core.config.load()).clone();
        config.model_policies.models.insert(
            "m".into(),
            crate::config::ModelStagePolicyOverride {
                affinity: Some(false),
                trie: Some(false),
                ..Default::default()
            },
        );
        core.config.replace(config);

        // This request captured the old policy and must still complete its
        // affinity write even though the live ConfigMap policy changed.
        core.response_started("policy-before-reload", Some(WORKER), 200)
            .await;
        wait_for_binding(&core, sid, first.worker.worker_id).await;
        core.request_finished("policy-before-reload", "complete")
            .await;

        // New requests immediately use the replacement policy and therefore
        // ignore the binding written by the older in-flight request.
        let second = core
            .schedule_for_test(
                "policy-after-reload",
                Some(sid.into()),
                CHAT_PATH,
                &chat_body(),
            )
            .await
            .unwrap();
        assert!(!second.sticky);
        core.request_finished("policy-after-reload", "complete")
            .await;
    }

    #[test]
    fn canonical_alias_prompt_hash_is_stable_after_appending() {
        let mut config = GwpConfig::default();
        config.routing.block_size = 4;
        config.routing.pseudo_stride = 1;
        config.session.prompt_hash_fallback = Some(PromptHashFallbackConfig { token_position: 16 });
        config
            .served_alias_model_map
            .insert("m-preview".into(), "m".into());

        let mut body = serde_json::json!({
            "model": "m-preview",
            "messages": [{"role": "user", "content": "a".repeat(200)}],
        });
        let prompt_hash = |body: &serde_json::Value| {
            let requested_model = body["model"].as_str().unwrap();
            let model = config.canonical_model(requested_model);
            let tokens =
                pseudo_tokens(&routing_text(CHAT_PATH, body), config.routing.pseudo_stride);
            prompt_hash_session_id(
                model,
                &tokens,
                config.routing.block_size,
                config
                    .session
                    .prompt_hash_fallback
                    .as_ref()
                    .unwrap()
                    .token_position,
            )
            .unwrap()
        };

        let first = prompt_hash(&body);
        assert!(first.starts_with("prompt-v1-"));

        body["messages"][0]["content"] =
            serde_json::Value::String(format!("{}{}", "a".repeat(200), " appended turn"));
        body["model"] = serde_json::Value::String("m".into());
        let appended = prompt_hash(&body);
        assert_eq!(first, appended);
    }

    #[tokio::test]
    async fn affinity_backend_errors_are_counted_and_fall_through() {
        let (mut core, _tx) = make_core(true).await;
        core.affinity = Arc::new(InstrumentedAffinityStore::new(
            Arc::new(FailingAffinityStore),
            core.router.metrics().clone(),
        ));

        let scheduled = core
            .schedule_for_test(
                "rid-affinity-error",
                Some(mint_session_id()),
                CHAT_PATH,
                &chat_body(),
            )
            .await
            .unwrap();
        assert!(!scheduled.sticky, "backend errors must fall through");
        core.request_finished("rid-affinity-error", "complete")
            .await;

        assert!(
            core.affinity
                .put(
                    "failed-put",
                    WorkerWithDpRank::from_worker_id(WORKER),
                    Duration::from_secs(60),
                )
                .await
                .is_err()
        );
        assert!(core.affinity.remove("failed-remove").await.is_err());

        let metrics = core.router.prometheus_metrics().unwrap();
        assert!(metric_has(
            &metrics,
            "dynamo_component_gwp_affinity_lookups_total",
            &["backend=\"redis\"", "outcome=\"backend_error\""]
        ));
        for operation in ["peek", "put", "remove"] {
            let operation = format!("operation=\"{operation}\"");
            assert!(metric_has(
                &metrics,
                "dynamo_component_gwp_affinity_backend_errors_total",
                &["backend=\"redis\"", &operation]
            ));
        }
        assert!(metric_has(
            &metrics,
            "dynamo_component_gwp_affinity_operation_duration_seconds_count",
            &[
                "backend=\"redis\"",
                "operation=\"peek\"",
                "result=\"error\""
            ]
        ));
    }

    #[tokio::test]
    async fn hanging_affinity_is_bounded_and_falls_through() {
        let (mut core, _tx) = make_core(true).await;
        let affinity = Arc::new(InstrumentedAffinityStore::new_with_limits(
            Arc::new(HangingAffinityStore),
            core.router.metrics().clone(),
            Duration::from_millis(75),
            1,
        ));
        core.affinity = affinity.clone();

        // Hold the only bulkhead permit with a backend call that never
        // responds. A concurrent call must fail immediately rather than queue.
        let held = tokio::spawn({
            let affinity = affinity.clone();
            async move { affinity.peek("held").await }
        });
        tokio::time::sleep(Duration::from_millis(10)).await;
        let saturated = tokio::time::timeout(Duration::from_millis(50), affinity.peek("saturated"))
            .await
            .expect("a saturated bulkhead must not wait for the backend deadline")
            .unwrap_err();
        assert!(saturated.to_string().contains("already in flight"));
        assert!(
            held.await
                .unwrap()
                .unwrap_err()
                .to_string()
                .contains("timed out")
        );

        let metrics = core.router.prometheus_metrics().unwrap();
        for result in ["bulkhead_rejected", "timeout"] {
            let result = format!("result=\"{result}\"");
            assert!(metric_has(
                &metrics,
                "dynamo_component_gwp_affinity_operation_duration_seconds_count",
                &["backend=\"redis\"", "operation=\"peek\"", &result]
            ));
        }

        // Once the permit is released, the hot-path lookup waits only for its
        // configured deadline, returns Err, and core falls through to B10.
        let scheduled = tokio::time::timeout(
            Duration::from_millis(250),
            core.schedule_for_test(
                "rid-hanging-affinity",
                Some(mint_session_id()),
                CHAT_PATH,
                &chat_body(),
            ),
        )
        .await
        .expect("affinity timeout must bound request latency")
        .unwrap();
        assert!(!scheduled.sticky);
        core.request_finished("rid-hanging-affinity", "complete")
            .await;

        let metrics = core.router.prometheus_metrics().unwrap();
        assert!(metric_has(
            &metrics,
            "dynamo_component_gwp_affinity_lookups_total",
            &["backend=\"redis\"", "outcome=\"backend_error\""]
        ));
        assert!(metric_has(
            &metrics,
            "dynamo_component_gwp_routing_decisions_total",
            &["sticky=\"false\""]
        ));
    }

    #[tokio::test]
    async fn unavailable_affinity_worker_is_counted_and_falls_through() {
        let (core, _tx) = make_core(true).await;
        let sid = mint_session_id();
        core.affinity
            .put_binding(
                &sid,
                AffinityBinding::with_endpoint(
                    WorkerWithDpRank::from_worker_id(999),
                    EndpointId(ENDPOINT.into()),
                ),
                Duration::from_secs(60),
            )
            .await
            .unwrap();

        let scheduled = core
            .schedule_for_test("rid-dead-affinity", Some(sid), CHAT_PATH, &chat_body())
            .await
            .unwrap();
        assert!(!scheduled.sticky);
        assert_eq!(scheduled.worker.worker_id, WORKER);
        core.request_finished("rid-dead-affinity", "complete").await;

        let metrics = core.router.prometheus_metrics().unwrap();
        assert!(metric_has(
            &metrics,
            "dynamo_component_gwp_affinity_lookups_total",
            &["backend=\"memory\"", "outcome=\"worker_unavailable\""]
        ));
        assert!(metric_has(
            &metrics,
            "dynamo_component_gwp_affinity_fallback_routings_total",
            &[
                "reason=\"worker_unavailable\"",
                "previous_endpoint=\"core-endpoint\"",
                "routed_endpoint=\"core-endpoint\"",
                "cluster_result=\"same\""
            ]
        ));
    }

    #[tokio::test]
    async fn reported_worker_replaces_provisional_booking_and_affinity() {
        let (core, _tx) = make_core(true).await;
        let sid = mint_session_id();
        core.affinity
            .put(
                &sid,
                WorkerWithDpRank::from_worker_id(WORKER),
                Duration::from_secs(60),
            )
            .await
            .unwrap();

        let scheduled = core
            .schedule_for_test("rid-1", Some(sid.clone()), CHAT_PATH, &chat_body())
            .await
            .unwrap();
        assert!(scheduled.sticky);
        assert_eq!(scheduled.worker.worker_id, WORKER);

        // The endpoint's local router served the request on worker 20 instead
        // of GWP's provisional worker 10.
        core.response_started("rid-1", Some(20), 200).await;
        wait_for_binding(&core, &sid, 20).await;
        let tokens = pseudo_tokens(
            &routing_text(CHAT_PATH, &chat_body()),
            core.config.load().routing.pseudo_stride,
        );
        let loads = core.router.potential_loads(&tokens).await;
        let active = |worker_id| {
            loads
                .iter()
                .find(|load| load.worker_id == worker_id)
                .map(|load| load.active_requests)
                .unwrap_or(0)
        };
        assert_eq!(active(WORKER), 0, "provisional booking must be freed");
        assert_eq!(active(20), 1, "actual worker must carry the workload");

        core.request_finished("rid-1", "complete").await;
        let loads = core.router.potential_loads(&tokens).await;
        assert!(
            loads
                .iter()
                .all(|load| load.worker_id != 20 || load.active_requests == 0),
            "final cleanup must free the confirmed booking"
        );

        let second = core
            .schedule_for_test("rid-2", Some(sid), CHAT_PATH, &chat_body())
            .await
            .unwrap();
        assert!(second.sticky);
        assert_eq!(second.worker.worker_id, 20);
        core.request_finished("rid-2", "complete").await;
    }

    #[tokio::test]
    async fn non_200_releases_provisional_without_rebooking_or_affinity() {
        let (core, _tx) = make_core(true).await;
        let scheduled = core
            .schedule_for_test("rid-denied", None, CHAT_PATH, &chat_body())
            .await
            .unwrap();
        assert_eq!(core.inflight_len(), 1);

        // Even if the denied response reports a different worker, it must not
        // become a confirmed booking or session binding.
        core.response_started("rid-denied", Some(20), 429).await;
        assert_eq!(core.inflight_len(), 0);
        assert_eq!(
            core.affinity.peek(&scheduled.session_id).await.unwrap(),
            None
        );

        let tokens = pseudo_tokens(
            &routing_text(CHAT_PATH, &chat_body()),
            core.config.load().routing.pseudo_stride,
        );
        let loads = core.router.potential_loads(&tokens).await;
        assert!(
            loads.iter().all(|load| load.active_requests == 0),
            "denied request must release its provisional booking without rebooking"
        );
        let metrics = core.router.prometheus_metrics().unwrap();
        assert!(metrics.contains("dynamo_component_gwp_routing_decisions_total"));
        assert!(metrics.contains("dynamo_component_gwp_upstream_responses_total"));
        assert!(metrics.contains("dynamo_component_gwp_denied_requests_total"));
        assert!(metrics.contains("dynamo_component_gwp_affinity_lookups_total"));
        assert!(metrics.contains("dynamo_component_gwp_tokenization_duration_seconds"));
        assert!(metrics.contains("mode=\"pseudo\""));
        assert!(metrics.contains("dynamo_component_gwp_time_to_first_byte_seconds"));
        assert!(metrics.contains("dynamo_component_gwp_request_duration_seconds"));
        assert!(metrics.contains("dynamo_component_gwp_request_outcomes_total"));
        assert!(metrics.contains("dynamo_frontend_worker_active_decode_blocks"));
        assert!(metrics.contains("dynamo_frontend_worker_active_prefill_tokens"));
        assert!(metrics.contains("dynamo_frontend_worker_active_requests"));
        assert!(metrics.contains("dynamo_frontend_worker_active_prefill_requests"));
        assert!(metrics.contains("dynamo_frontend_worker_active_decode_requests"));
        assert!(metrics.contains("dynamo_frontend_router_queue_pending_requests"));
        assert!(metrics.contains("dynamo_frontend_router_queue_pending_isl_tokens"));
        assert!(metrics.contains("status=\"429\""));
        assert!(metrics.contains("outcome=\"overloaded\""));

        // Envoy still emits RequestFinished after consuming the denied body.
        // The duplicate provisional free is intentionally harmless.
        core.request_finished("rid-denied", "complete").await;
    }

    #[tokio::test]
    async fn unknown_rid_and_double_finish_are_noops() {
        let (core, _tx) = make_core(true).await;
        // Must not panic or corrupt state.
        core.response_started("ghost", Some(10), 200).await;
        core.request_finished("ghost", "reset").await;

        let scheduled = core
            .schedule_for_test("rid-1", None, CHAT_PATH, &chat_body())
            .await
            .unwrap();
        core.request_finished("rid-1", "complete").await;
        core.request_finished("rid-1", "complete").await; // idempotent
        assert_eq!(core.inflight_len(), 0);
        drop(scheduled);
    }

    #[tokio::test]
    async fn forced_shutdown_cleanup_releases_every_booking() {
        let (core, _tx) = make_core(true).await;
        core.schedule_for_test("shutdown-1", None, CHAT_PATH, &chat_body())
            .await
            .unwrap();
        core.schedule_for_test("shutdown-2", None, CHAT_PATH, &chat_body())
            .await
            .unwrap();
        assert_eq!(core.inflight_len(), 2);

        core.force_finish_all("shutdown_timeout").await;
        assert_eq!(core.inflight_len(), 0);

        let tokens = pseudo_tokens(
            &routing_text(CHAT_PATH, &chat_body()),
            core.config.load().routing.pseudo_stride,
        );
        assert!(
            core.router
                .potential_loads(&tokens)
                .await
                .iter()
                .all(|load| load.active_requests == 0)
        );

        // Late Envoy completion callbacks remain harmless.
        core.request_finished("shutdown-1", "complete").await;
        core.request_finished("shutdown-2", "complete").await;
    }

    #[tokio::test]
    async fn schedule_fails_when_no_workers() {
        let (core, _tx) = make_core(false).await;
        let result = core
            .schedule_for_test("rid-1", None, CHAT_PATH, &chat_body())
            .await;
        assert!(matches!(result, Err(ScheduleError::NoRoutableEndpoint(_))));
        assert_eq!(core.inflight_len(), 0);
    }

    #[tokio::test]
    async fn empty_routing_text_does_not_shutdown_scheduler() {
        let (core, _tx) = make_core(true).await;
        let body = serde_json::json!({"model": "m"});

        let first = core
            .schedule_for_test("empty-1", None, CHAT_PATH, &body)
            .await
            .unwrap();
        core.request_finished("empty-1", "invalid_upstream_request")
            .await;

        let second = core
            .schedule_for_test("empty-2", None, CHAT_PATH, &body)
            .await
            .unwrap();
        core.request_finished("empty-2", "invalid_upstream_request")
            .await;
        assert!(!first.session_id.is_empty());
        assert!(!second.session_id.is_empty());
    }

    #[tokio::test]
    async fn schedule_rejects_an_unknown_model_as_invalid() {
        let (core, _tx) = make_core(true).await;
        let mut body = chat_body();
        body["model"] = serde_json::Value::String("not-routable".into());
        let result = core
            .schedule_for_test("rid-unknown-model", None, CHAT_PATH, &body)
            .await;
        assert!(matches!(result, Err(ScheduleError::InvalidModel(_))));
        assert_eq!(core.inflight_len(), 0);
    }

    #[tokio::test]
    async fn model_routes_constrain_endpoints_and_cross_model_session_unsticks() {
        let (core, _tx) = make_multi_model_core().await;
        let kimi_body = serde_json::json!({
            "model": "kimi-k2",
            "messages": [{"role": "user", "content": "hello"}],
        });
        let kimi = core
            .schedule_for_test("kimi-1", None, CHAT_PATH, &kimi_body)
            .await
            .unwrap();
        assert_eq!(kimi.endpoint_id, EndpointId("kimi-a".into()));
        core.response_started("kimi-1", Some(kimi.worker.worker_id), 200)
            .await;
        wait_for_binding(&core, &kimi.session_id, kimi.worker.worker_id).await;
        core.request_finished("kimi-1", "complete").await;

        let glm_body = serde_json::json!({
            "model": "glm-4.7",
            "messages": [{"role": "user", "content": "hello"}],
        });
        let glm = core
            .schedule_for_test("glm-1", Some(kimi.session_id), CHAT_PATH, &glm_body)
            .await
            .unwrap();
        assert!(!glm.sticky, "Kimi endpoint is ineligible for GLM");
        assert!(matches!(glm.endpoint_id.0.as_str(), "glm-a" | "glm-b"));
        core.request_finished("glm-1", "complete").await;
        let metrics = core.router.prometheus_metrics().unwrap();
        assert!(metric_has(
            &metrics,
            "dynamo_component_gwp_affinity_lookups_total",
            &["backend=\"memory\"", "outcome=\"ineligible\""]
        ));
        assert!(metric_has(
            &metrics,
            "dynamo_component_gwp_affinity_fallback_routings_total",
            &[
                "reason=\"ineligible\"",
                "previous_endpoint=\"kimi-a\"",
                "cluster_result=\"different\""
            ]
        ));
    }

    #[tokio::test]
    async fn routing_requirements_are_hard_constraints_and_revalidate_affinity() {
        let (core, _tx) = make_multi_model_core().await;
        add_endpoint_properties(&core);
        let body = serde_json::json!({
            "model": "glm-4.7",
            "messages": [{"role": "user", "content": "hello"}],
        });
        let sid = mint_session_id();

        let hippa = core
            .schedule_for_routing_requirements_for_test(
                "hippa-1",
                Some(sid.clone()),
                Some(
                    r#"{"region":{"required":["us","canada"]},"compliance":{"required":["hippa"]}}"#
                        .into(),
                ),
                CHAT_PATH,
                &body,
            )
            .await
            .unwrap();
        assert_eq!(hippa.endpoint_id, EndpointId("glm-a".into()));
        core.response_started("hippa-1", Some(hippa.worker.worker_id), 200)
            .await;
        wait_for_binding(&core, &sid, hippa.worker.worker_id).await;
        core.request_finished("hippa-1", "complete").await;

        let standard = core
            .schedule_for_routing_requirements_for_test(
                "standard-1",
                Some(sid.clone()),
                Some(r#"{"compliance":{"required":["standard"]}}"#.into()),
                CHAT_PATH,
                &body,
            )
            .await
            .unwrap();
        assert!(
            !standard.sticky,
            "a binding on an endpoint missing a new requirement must unstick"
        );
        assert_eq!(standard.endpoint_id, EndpointId("glm-b".into()));
        core.request_finished("standard-1", "complete").await;

        let malformed = core
            .schedule_for_routing_requirements_for_test(
                "malformed-1",
                None,
                Some("not-json".into()),
                CHAT_PATH,
                &body,
            )
            .await;
        assert!(matches!(
            malformed,
            Err(ScheduleError::RoutingRequirements(_))
        ));

        let unavailable = core
            .schedule_for_routing_requirements_for_test(
                "unavailable-1",
                None,
                Some(r#"{"region":{"required":["not-available"]}}"#.into()),
                CHAT_PATH,
                &body,
            )
            .await;
        assert!(matches!(
            unavailable,
            Err(ScheduleError::NoRoutableEndpoint(_))
        ));

        let headerless = core
            .schedule_for_test("headerless-1", None, CHAT_PATH, &body)
            .await
            .unwrap();
        core.request_finished("headerless-1", "complete").await;
        assert!(matches!(
            headerless.endpoint_id.0.as_str(),
            "glm-a" | "glm-b"
        ));
    }
}
