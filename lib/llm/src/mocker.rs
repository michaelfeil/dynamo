// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mocker module - runtime integration for the mock scheduler.
//!
//! The core mocker logic lives in the `dynamo-mocker` crate.
//! This module provides the runtime-dependent engine wrapper.

mod metrics;

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use crate::backend::ExecutionContext;
use crate::kv_router::publisher::{KvEventPublisher, KvEventSourceConfig, WorkerMetricsPublisher};
use crate::protocols::TokenIdType;
use crate::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};
use anyhow::Result;
use dashmap::DashMap;
use dynamo_kv_router::protocols::{KvCacheEvent, StorageTier};
use dynamo_mocker::common::protocols::{
    DirectRequest, KvCacheEventSink, KvEventPublishers, MockEngineArgs, OutputSignal,
    RawKvEventSink,
};
use dynamo_mocker::common::utils::sleep_precise;
use dynamo_mocker::engine::create_engine;
use dynamo_mocker::scheduler::{MockerMetrics, SchedulerHandle};
use dynamo_mocker::services::bootstrap::{BootstrapServer, connect_to_prefill};
use dynamo_mocker::services::zmq_events::ZmqKvEventSink;
use dynamo_protocols::types::{CompletionUsage, PromptTokensDetails};
use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::metrics::MetricsHierarchy;
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::{
    component::Component,
    engine::AsyncEngineContextProvider,
    pipeline::{
        AsyncEngine, Error, ManyOut, ResponseStream, SingleIn, async_trait, network::Ingress,
    },
    traits::DistributedRuntimeProvider,
};
use futures::StreamExt;
use rand::Rng;
use tokio::sync::{Notify, OnceCell, mpsc};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use self::metrics::NativeMockerMetrics;

pub const MOCKER_COMPONENT: &str = "mocker";

pub type SchedulerMetricsCallback =
    Arc<dyn Fn(MockerMetrics) -> anyhow::Result<()> + Send + Sync + 'static>;

/// Wrapper to adapt KvEventPublisher to the KvCacheEventSink trait
struct KvEventSinkAdapter(KvEventPublisher);

impl KvCacheEventSink for KvEventSinkAdapter {
    fn publish(&self, event: KvCacheEvent) -> anyhow::Result<()> {
        self.0
            .publish(event)
            .map_err(|e| anyhow::anyhow!("Failed to send KV event: {}", e))
    }

    fn publish_with_storage_tier(
        &self,
        event: KvCacheEvent,
        storage_tier: StorageTier,
    ) -> anyhow::Result<()> {
        self.0
            .publish_with_storage_tier(event, storage_tier)
            .map_err(|e| anyhow::anyhow!("Failed to send KV event: {}", e))
    }
}

/// Cumulative usage snapshot carrying the scheduler's admission cache truth
/// in `prompt_tokens_details.cached_tokens`.
fn usage_with_cached_tokens(
    prompt_tokens: usize,
    completion_tokens: usize,
    cached_tokens: usize,
) -> CompletionUsage {
    // Saturate rather than panic on pathological token counts.
    fn to_u32(value: usize) -> u32 {
        value.try_into().unwrap_or(u32::MAX)
    }
    CompletionUsage {
        prompt_tokens: to_u32(prompt_tokens),
        completion_tokens: to_u32(completion_tokens),
        total_tokens: to_u32(prompt_tokens.saturating_add(completion_tokens)),
        prompt_tokens_details: Some(PromptTokensDetails {
            audio_tokens: None,
            cached_tokens: Some(to_u32(cached_tokens)),
        }),
        completion_tokens_details: None,
    }
}

fn generate_random_token() -> TokenIdType {
    let mut rng = rand::rng();
    rng.random_range(1000..2000)
}

/// AsyncEngine wrapper around the Scheduler that generates random character tokens
pub struct MockEngine {
    active_requests: Arc<DashMap<Uuid, mpsc::UnboundedSender<OutputSignal>>>,
    request_senders: OnceCell<Vec<mpsc::UnboundedSender<DirectRequest>>>,
    senders_ready: Notify,
    engine_args: MockEngineArgs,
    unset_dp_rank_counter: AtomicU32,
    /// Bootstrap server for prefill workers in disaggregated mode
    bootstrap_server: Arc<OnceCell<Arc<BootstrapServer>>>,
    native_metrics: Arc<NativeMockerMetrics>,
    metrics_callback: Option<SchedulerMetricsCallback>,
    /// Keep schedulers alive so their CancelGuards don't fire prematurely.
    _schedulers: OnceCell<Vec<Box<dyn SchedulerHandle>>>,
    /// Forward pass metrics publisher (kept alive for the engine lifetime).
    _fpm_publisher: OnceCell<crate::fpm_publisher::FpmDirectPublisher>,
}

struct MockSessionControlEngine;

#[async_trait]
impl AsyncEngine<SingleIn<serde_json::Value>, ManyOut<Annotated<serde_json::Value>>, Error>
    for MockSessionControlEngine
{
    async fn generate(
        &self,
        request: SingleIn<serde_json::Value>,
    ) -> Result<ManyOut<Annotated<serde_json::Value>>, Error> {
        let (body, context) = request.into_parts();
        let action = body.get("action").and_then(|value| value.as_str());
        let session_id = body.get("session_id").and_then(|value| value.as_str());

        let response = match (action, session_id) {
            (Some("open_session" | "close_session"), Some(session_id)) => {
                serde_json::json!({
                    "status": "ok",
                    "session_id": session_id,
                })
            }
            (_, None) => {
                serde_json::json!({
                    "status": "error",
                    "message": "session_id required",
                })
            }
            (other, Some(session_id)) => {
                serde_json::json!({
                    "status": "error",
                    "session_id": session_id,
                    "message": format!("unsupported action {:?}", other),
                })
            }
        };

        let stream = futures::stream::iter(vec![Annotated::from_data(response)]);
        Ok(ResponseStream::new(Box::pin(stream), context.context()))
    }
}

impl MockEngine {
    /// Create a new MockEngine with the given parameters
    pub fn new(engine_args: MockEngineArgs) -> Self {
        Self::new_with_metrics_callback(engine_args, None)
    }

    pub fn new_with_metrics_callback(
        engine_args: MockEngineArgs,
        metrics_callback: Option<SchedulerMetricsCallback>,
    ) -> Self {
        let native_metrics = NativeMockerMetrics::new(engine_args.engine_type, engine_args.dp_size)
            .expect("mocker native metrics collectors should be valid");
        Self {
            active_requests: Arc::new(DashMap::new()),
            request_senders: OnceCell::new(),
            senders_ready: Notify::new(),
            engine_args,
            unset_dp_rank_counter: AtomicU32::new(0),
            bootstrap_server: Arc::new(OnceCell::new()),
            native_metrics,
            metrics_callback,
            _schedulers: OnceCell::new(),
            _fpm_publisher: OnceCell::new(),
        }
    }

    fn resolve_dp_rank(&self, request: &PreprocessedRequest) -> u32 {
        if let Some(dp_rank) = request.routing.as_ref().and_then(|routing| routing.dp_rank) {
            return dp_rank;
        }

        self.unset_dp_rank_counter.fetch_add(1, Ordering::Relaxed) % self.engine_args.dp_size
    }

    pub async fn start(&self, component: Component, metrics_component: Component) -> Result<()> {
        // Use primary_token() instead of child_token() so the mocker continues running
        // during graceful shutdown (Phase 1/2) and only stops in Phase 3.
        // child_token() is a child of endpoint_shutdown_token which is cancelled in Phase 1.
        // primary_token() is only cancelled in Phase 3, after waiting for inflight requests.
        let cancel_token = component.drt().primary_token();
        self.native_metrics
            .register(component.get_metrics_registry())?;

        // Simulate engine startup time if configured
        if let Some(startup_time_secs) = self.engine_args.startup_time {
            tracing::info!("Simulating engine startup time: {:.2}s", startup_time_secs);
            tokio::time::sleep(Duration::from_secs_f64(startup_time_secs)).await;
            tracing::info!("Engine startup simulation completed");
        }

        Self::start_session_control_endpoint(component.clone());

        // Start bootstrap server for prefill workers in disaggregated mode
        if self.engine_args.is_prefill()
            && let Some(port) = self.engine_args.bootstrap_port
        {
            let server = BootstrapServer::start(port, cancel_token.clone()).await?;
            let _ = self.bootstrap_server.set(server);
            tracing::info!(port = port, "Bootstrap server started for prefill worker");
        }

        let kv_component = if self.engine_args.needs_kv_publisher() {
            tracing::info!(
                "Initializing KV event publisher with block_size {}, enable_local_indexer={}",
                self.engine_args.block_size,
                self.engine_args.enable_local_indexer
            );
            Some(&component)
        } else {
            None
        };

        // Create FPM publisher upfront and get per-dp-rank sink handles.
        let worker_id = metrics_component.drt().connection_id().to_string();
        let fpm_sinks = match crate::fpm_publisher::FpmDirectPublisher::new(
            metrics_component.clone(),
            worker_id,
            self.engine_args.dp_size,
        )
        .await
        {
            Ok((publisher, sinks)) => {
                let _ = self._fpm_publisher.set(publisher);
                sinks
            }
            Err(e) => {
                tracing::error!("Failed to start FPM publisher: {e}");
                (0..self.engine_args.dp_size)
                    .map(|_| dynamo_mocker::common::protocols::FpmPublisher::default())
                    .collect()
            }
        };

        let schedulers = self
            .start_schedulers(kv_component, cancel_token.clone(), fpm_sinks)
            .await;

        Self::start_metrics_publishing(
            &schedulers,
            metrics_component,
            self.native_metrics.clone(),
            self.metrics_callback.clone(),
            cancel_token.clone(),
        )
        .await?;

        let _ = self._schedulers.set(schedulers);

        Ok(())
    }

    fn start_session_control_endpoint(component: Component) {
        let ingress = match Ingress::for_engine(Arc::new(MockSessionControlEngine)) {
            Ok(ingress) => ingress,
            Err(e) => {
                tracing::error!("Failed to build mocker session_control ingress: {e}");
                return;
            }
        };

        tokio::spawn(async move {
            if let Err(e) = component
                .endpoint("session_control")
                .endpoint_builder()
                .handler(ingress)
                .graceful_shutdown(true)
                .start()
                .await
            {
                tracing::error!("Mocker session_control endpoint failed: {e}");
            }
        });
    }

    /// Send a request to the appropriate scheduler, waiting for initialization if needed.
    pub async fn direct(&self, request: DirectRequest, dp_rank: usize) {
        let sender = self.request_sender(dp_rank).await;
        let _ = sender.send(request);
    }

    async fn request_sender(&self, dp_rank: usize) -> mpsc::UnboundedSender<DirectRequest> {
        if let Some(senders) = self.request_senders.get() {
            return senders[dp_rank].clone();
        }

        // Register the waiter *before* re-checking to avoid a TOCTOU race
        // where `start_schedulers` sets + notifies between our check and subscribe.
        let notified = self.senders_ready.notified();
        if let Some(senders) = self.request_senders.get() {
            return senders[dp_rank].clone();
        }
        notified.await;

        let senders = self
            .request_senders
            .get()
            .expect("must be set after notify");
        senders[dp_rank].clone()
    }

    /// Create schedulers and spawn their background tasks for distributing token notifications.
    async fn start_schedulers(
        &self,
        component: Option<&Component>,
        cancel_token: CancellationToken,
        fpm_sinks: Vec<dynamo_mocker::common::protocols::FpmPublisher>,
    ) -> Vec<Box<dyn SchedulerHandle>> {
        let args = &self.engine_args;
        let mut schedulers = Vec::<Box<dyn SchedulerHandle>>::new();
        let mut senders = Vec::with_capacity(args.dp_size as usize);

        for (dp_rank, fpm_publisher) in (0..args.dp_size).zip(fpm_sinks) {
            let (output_tx, mut output_rx) = mpsc::unbounded_channel::<Vec<OutputSignal>>();

            let (kv_event_publishers, relay_publisher): (
                KvEventPublishers,
                Option<KvEventPublisher>,
            ) = match component {
                Some(comp) if args.zmq_kv_events_port.is_some() => {
                    let zmq_port = args.zmq_kv_events_port.unwrap() + dp_rank as u16;
                    let replay_port = args.zmq_replay_port.map(|p| p + dp_rank as u16);
                    match ZmqKvEventSink::new(
                        zmq_port,
                        replay_port,
                        dp_rank,
                        args.block_size as u32,
                    )
                    .await
                    {
                        Ok(sink) => {
                            let source_config = Some(KvEventSourceConfig::Zmq {
                                endpoint: format!("tcp://127.0.0.1:{zmq_port}"),
                                topic: String::new(),
                            });
                            match KvEventPublisher::new_with_local_indexer(
                                comp.clone(),
                                args.block_size as u32,
                                source_config,
                                args.enable_local_indexer,
                                dp_rank,
                                None,
                            ) {
                                Ok(publisher) => (
                                    KvEventPublishers::new(
                                        None,
                                        Some(Arc::new(sink) as Arc<dyn RawKvEventSink>),
                                    ),
                                    Some(publisher),
                                ),
                                Err(e) => {
                                    tracing::error!(
                                        "Failed to create KV event relay for dp_rank {dp_rank}: {e}"
                                    );
                                    (KvEventPublishers::default(), None)
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!(
                                "Failed to create ZMQ KV event sink for dp_rank {dp_rank}: {e}"
                            );
                            (KvEventPublishers::default(), None)
                        }
                    }
                }
                Some(comp) => {
                    match KvEventPublisher::new_with_local_indexer(
                        comp.clone(),
                        args.block_size as u32,
                        None,
                        args.enable_local_indexer,
                        dp_rank,
                        None,
                    ) {
                        Ok(publisher) => (
                            KvEventPublishers::new(
                                Some(Arc::new(KvEventSinkAdapter(publisher))
                                    as Arc<dyn KvCacheEventSink>),
                                None,
                            ),
                            None,
                        ),
                        Err(e) => {
                            tracing::error!(
                                "Failed to create KV event publisher for dp_rank {dp_rank}: {e}"
                            );
                            (KvEventPublishers::default(), None)
                        }
                    }
                }
                None => (KvEventPublishers::default(), None),
            };

            let scheduler = create_engine(
                args.clone(),
                dp_rank,
                Some(output_tx),
                kv_event_publishers,
                Some(cancel_token.clone()),
                fpm_publisher,
            );

            senders.push(scheduler.request_sender());
            schedulers.push(scheduler);

            let active_requests_clone = self.active_requests.clone();
            let cancel_token_cloned = cancel_token.clone();

            tokio::spawn(async move {
                // Keep the relay publisher alive for the lifetime of this task.
                // Dropping it would cancel its background ZMQ→NATS relay tasks.
                let _relay_publisher = relay_publisher;

                loop {
                    tokio::select! {
                        signal_result = output_rx.recv() => {
                            let Some(output_batch) = signal_result else {
                                break; // Channel closed
                            };

                            for signal in output_batch {
                                if let Some(request_tx) = active_requests_clone.get(&signal.uuid) {
                                    let _ = request_tx.send(signal);
                                }
                            }
                        }
                        _ = cancel_token_cloned.cancelled() => {
                            tracing::info!("Scheduler output task cancelled, clearing active requests");
                            active_requests_clone.clear();
                            break;
                        }
                    }
                }
            });
        }

        // Set the senders once and notify waiters
        self.request_senders
            .set(senders)
            .expect("Already initialized");
        self.senders_ready.notify_waiters();

        schedulers
    }

    /// Start background tasks to publish metrics on change
    async fn start_metrics_publishing(
        schedulers: &[Box<dyn SchedulerHandle>],
        component: Component,
        native_metrics: Arc<NativeMockerMetrics>,
        metrics_callback: Option<SchedulerMetricsCallback>,
        cancel_token: CancellationToken,
    ) -> Result<()> {
        let metrics_publisher = Arc::new(WorkerMetricsPublisher::new()?);

        if let Err(e) = metrics_publisher.create_endpoint(component).await {
            tracing::error!("Metrics endpoint failed: {e}");
        }

        // Match vLLM and TRT-LLM: advertise each rank at zero load before the
        // first scheduler iteration so the router can select an idle worker.
        for scheduler in schedulers {
            let metrics_rx = scheduler.metrics_receiver();
            let metrics = metrics_rx.borrow().clone();
            if let Err(e) = metrics_publisher.publish(Some(metrics.dp_rank), None, Some(0)) {
                tracing::warn!(
                    "Failed to publish initial metrics for DP rank {}: {e}",
                    metrics.dp_rank
                );
            }
        }

        for scheduler in schedulers.iter() {
            let mut metrics_rx = scheduler.metrics_receiver();
            let publisher = metrics_publisher.clone();
            let native_metrics = native_metrics.clone();
            let metrics_callback = metrics_callback.clone();
            let cancel_token = cancel_token.clone();

            tokio::spawn(async move {
                loop {
                    tokio::select! {
                        // Watch for metrics changes
                        Ok(_) = metrics_rx.changed() => {
                            // Get the latest metrics
                            let metrics = metrics_rx.borrow().clone();
                            native_metrics.update_scheduler_snapshot(&metrics);

                            // Publish metrics using flat API
                            if let Err(e) = publisher.publish(
                                Some(metrics.dp_rank),
                                None,
                                Some(metrics.active_decode_blocks),
                            ) {
                                tracing::warn!("Failed to publish metrics for DP rank {}: {e}", metrics.dp_rank);
                            } else {
                                tracing::debug!(
                                    dp_rank = metrics.dp_rank,
                                    active_decode_blocks = metrics.active_decode_blocks,
                                    total_blocks = metrics.total_blocks,
                                    gpu_cache_usage_perc = metrics.gpu_cache_usage_perc,
                                    "published mocker load metrics"
                                );
                            }

                            if let Some(callback) = metrics_callback.as_ref() {
                                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    callback(metrics.clone())
                                })) {
                                    Ok(Ok(())) => {}
                                    Ok(Err(e)) => {
                                        tracing::warn!(
                                            dp_rank = metrics.dp_rank,
                                            "Failed to publish mocker scheduler metrics: {e}"
                                        );
                                    }
                                    Err(_) => {
                                        tracing::error!(
                                            dp_rank = metrics.dp_rank,
                                            "Mocker scheduler metrics callback panicked"
                                        );
                                    }
                                }
                            }
                        }
                        _ = cancel_token.cancelled() => {
                            tracing::debug!("Metrics publishing cancelled");
                            break;
                        }
                    }
                }
            });
        }
        tracing::info!("Metrics background tasks started");
        Ok(())
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<LLMEngineOutput>, Error> for MockEngine {
    async fn generate(
        &self,
        input: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<LLMEngineOutput>, Error> {
        let (request, ctx) = input.into_parts();
        let request_start = Instant::now();

        let dp_rank = self.resolve_dp_rank(&request);

        // Validate dp_rank
        if dp_rank >= self.engine_args.dp_size {
            return Err(Error::msg(format!(
                "dp_rank {} is out of bounds for dp_size {}",
                dp_rank, self.engine_args.dp_size
            )));
        }

        let request_uuid = ctx.id().parse().unwrap_or(Uuid::new_v4());
        let is_prefill = self.engine_args.is_prefill();
        let max_output_tokens = if is_prefill {
            1
        } else {
            request
                .stop_conditions
                .max_tokens
                .ok_or_else(|| Error::msg("max_output_tokens must be specified for mocker"))?
                as usize
        };
        let native_timing = self
            .native_metrics
            .request_timing(&request.model, dp_rank, is_prefill, request_start)
            .await;

        // Bootstrap rendezvous for disaggregated serving
        // - Decode: send receiver metadata to prefill, then wait for prefill completion
        // - Prefill: wait for decode metadata before emitting output, then complete_room()
        let bootstrap_room = request.bootstrap_info.as_ref().map(|b| b.bootstrap_room);
        if let Some(bootstrap_info) = &request.bootstrap_info
            && self.engine_args.is_decode()
        {
            connect_to_prefill(
                &bootstrap_info.bootstrap_host,
                bootstrap_info.bootstrap_port,
                bootstrap_info.bootstrap_room,
            )
            .await
            .map_err(|e| Error::msg(format!("Bootstrap connection failed: {e}")))?;
        }

        // Convert PreprocessedRequest to DirectRequest for scheduler
        let prompt_tokens_count = request.token_ids.len();
        let direct_request = DirectRequest {
            tokens: request.token_ids.clone(),
            max_output_tokens,
            uuid: Some(request_uuid),
            dp_rank,
            arrival_timestamp_ms: request.request_timestamp_ms,
        };

        let (request_tx, mut request_rx) = mpsc::unbounded_channel::<OutputSignal>();
        self.active_requests.insert(request_uuid, request_tx);

        let bootstrap_server = self.bootstrap_server.clone();
        let delayed_prefill_submission = if is_prefill {
            match (bootstrap_server.get().cloned(), bootstrap_room) {
                (Some(server), Some(room_id)) => {
                    let sender = self.request_sender(dp_rank as usize).await;
                    Some((server, room_id, sender, direct_request))
                }
                _ => {
                    self.direct(direct_request, dp_rank as usize).await;
                    None
                }
            }
        } else {
            self.direct(direct_request, dp_rank as usize).await;
            None
        };

        // Create a simple channel for the stream
        let (stream_tx, stream_rx) = mpsc::unbounded_channel::<LLMEngineOutput>();

        let active_requests = self.active_requests.clone();
        let async_context = ctx.context();
        let reasoning = self.engine_args.reasoning.clone();
        let mut native_timing = native_timing;

        // Spawn a task to handle the complex async logic
        tokio::spawn(async move {
            if let Some((server, room_id, sender, direct_request)) = delayed_prefill_submission {
                tokio::select! {
                    result = server.wait_for_decode_ready(room_id) => {
                        if let Err(e) = result {
                            let _ = stream_tx.send(LLMEngineOutput::error(format!(
                                "Bootstrap wait for decode metadata failed: {e}"
                            )));
                            active_requests.remove(&request_uuid);
                            return;
                        }
                    }
                    _ = async_context.stopped() => {
                        let _ = stream_tx.send(LLMEngineOutput::cancelled());
                        active_requests.remove(&request_uuid);
                        return;
                    }
                }

                if sender.send(direct_request).is_err() {
                    let _ = stream_tx.send(LLMEngineOutput::error(
                        "Scheduler input channel closed before bootstrap prefill submission"
                            .to_string(),
                    ));
                    active_requests.remove(&request_uuid);
                    return;
                }
            }

            let mut token_count = 0;
            let mut cached_prefix_tokens: Option<usize> = None;
            let think_len = reasoning
                .as_ref()
                .map(|cfg| cfg.num_thinking_tokens(max_output_tokens))
                .unwrap_or(0);

            loop {
                tokio::select! {
                    maybe_signal = request_rx.recv() => {
                        let Some(signal) = maybe_signal else {
                            let _ = stream_tx.send(LLMEngineOutput::error("All output transmitters closed".to_string()));
                            break;
                        };

                        if let Some(cached) = signal.cached_tokens {
                            cached_prefix_tokens = Some(cached);
                        }

                        // Generate a token (with thinking boundaries if configured)
                        let token_id = if token_count == 0 && think_len > 0 {
                            reasoning.as_ref().unwrap().start_thinking_token_id
                        } else if think_len > 0 && token_count == think_len - 1 {
                            reasoning.as_ref().unwrap().end_thinking_token_id
                        } else {
                            generate_random_token()
                        };
                        token_count += 1;

                        // The first chunk carries the admission cache truth; the
                        // final chunk repeats cumulative totals (OpenAI convention).
                        let output = LLMEngineOutput {
                            token_ids: vec![token_id],
                            disaggregated_params: is_prefill.then(|| serde_json::json!("dummy")),
                            completion_usage: signal.cached_tokens.map(|cached| {
                                usage_with_cached_tokens(prompt_tokens_count, token_count, cached)
                            }),
                            ..Default::default()
                        };

                        if signal.completed && token_count < max_output_tokens {
                            let _ = stream_tx.send(LLMEngineOutput::error("Completion signal received before max tokens reached".to_string()));
                            break;
                        }

                        if signal.completed {
                            if stream_tx.send(output).is_err() {
                                tracing::error!("Output stream receiver closed.");
                                break;
                            }
                            native_timing.record_tokens(1);

                            // Prefill-to-decode handoff delay is emitted by the shared mocker core.
                            if is_prefill
                                && let Some(delay_ms) = signal.handoff_delay_ms
                            {
                                sleep_precise(Duration::from_secs_f64(delay_ms / 1000.0)).await;
                            }

                            // Prefill: after first token, mark room complete (unblocks decode)
                            if is_prefill
                                && let (Some(server), Some(room_id)) = (bootstrap_server.get(), bootstrap_room)
                            {
                                server.complete_room(room_id);
                            }

                            let mut final_output = LLMEngineOutput::length();
                            final_output.completion_usage = cached_prefix_tokens.map(|cached| {
                                usage_with_cached_tokens(prompt_tokens_count, token_count, cached)
                            });
                            if stream_tx.send(final_output).is_err() {
                                tracing::error!("Output stream receiver closed.");
                                break;
                            }
                            native_timing.record_normal_completion();
                            break;
                        }

                        if stream_tx.send(output).is_err() {
                            tracing::error!("Output stream receiver closed.");
                            break;
                        }
                        native_timing.record_tokens(1);
                    }

                    _ = async_context.stopped() => {
                        let _ = stream_tx.send(LLMEngineOutput::cancelled());
                        break;
                    }
                }
            }

            active_requests.remove(&request_uuid);
        });

        let stream = UnboundedReceiverStream::new(stream_rx);
        Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
    }
}

pub struct AnnotatedMockEngine {
    inner: Arc<MockEngine>,
}

impl AnnotatedMockEngine {
    pub fn new(
        inner: MockEngine,
        distributed_runtime: DistributedRuntime,
        endpoint_id: dynamo_runtime::protocols::EndpointId,
        metrics_component: Component,
    ) -> Self {
        let inner = Arc::new(inner);
        let inner_clone = inner.clone();

        // Start background task to wait for component service and start the engine
        let cancel_token = distributed_runtime.primary_token();
        tokio::spawn(async move {
            let component = loop {
                if cancel_token.is_cancelled() {
                    tracing::debug!("Mocker engine startup cancelled");
                    return;
                }

                let ready = distributed_runtime
                    .namespace(&endpoint_id.namespace)
                    .and_then(|ns| ns.component(&endpoint_id.component))
                    .ok();

                if let Some(comp) = ready
                    && let Ok(instances) = comp.list_instances().await
                    && !instances.is_empty()
                {
                    break comp;
                }

                tracing::debug!("Component service not available yet, retrying...");
                tokio::time::sleep(Duration::from_millis(100)).await;
            };

            tracing::debug!("Component service is now available, starting mocker engine");
            if let Err(e) = inner_clone.start(component, metrics_component).await {
                tracing::error!("Failed to start mocker engine: {e}");
            }
        });

        Self { inner }
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for AnnotatedMockEngine
{
    async fn generate(
        &self,
        input: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let stream = self.inner.generate(input).await?;
        let context = stream.context();

        // Convert stream of LLMEngineOutput to Annotated<LLMEngineOutput>
        let annotated_stream = stream.map(Annotated::from_data);

        Ok(ResponseStream::new(Box::pin(annotated_stream), context))
    }
}

/// Create a mocker engine as ExecutionContext
pub async fn make_mocker_engine(
    distributed_runtime: DistributedRuntime,
    endpoint_id: dynamo_runtime::protocols::EndpointId,
    args: MockEngineArgs,
    metrics_endpoint_id: dynamo_runtime::protocols::EndpointId,
    metrics_callback: Option<SchedulerMetricsCallback>,
) -> Result<ExecutionContext, Error> {
    // Create the mocker engine
    tracing::info!("Creating mocker engine with config: {args:?}");
    let metrics_component = distributed_runtime
        .namespace(&metrics_endpoint_id.namespace)?
        .component(&metrics_endpoint_id.component)?;
    let annotated_engine = AnnotatedMockEngine::new(
        MockEngine::new_with_metrics_callback(args, metrics_callback),
        distributed_runtime,
        endpoint_id,
        metrics_component,
    );

    Ok(Arc::new(annotated_engine))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
    use dynamo_mocker::common::protocols::WorkerType;
    use std::collections::BTreeMap;

    /// Build a prefill engine with a wired scheduler channel and return its
    /// output stream plus the per-request signal sender, so tests can drive
    /// the relay directly without a live runtime.
    async fn wired_prefill_stream(
        uuid: Uuid,
    ) -> (
        ManyOut<LLMEngineOutput>,
        mpsc::UnboundedSender<OutputSignal>,
    ) {
        let args = MockEngineArgs::builder()
            .worker_type(WorkerType::Prefill)
            .block_size(4)
            .num_gpu_blocks(64)
            .max_num_batched_tokens(Some(64))
            .speedup_ratio(1000.0)
            .build()
            .unwrap();
        let engine = MockEngine::new(args);

        // Wire the scheduler input channel so `generate` can submit the request
        // without a live runtime; the test drives the output signals directly.
        let (direct_tx, _direct_rx) = mpsc::unbounded_channel::<DirectRequest>();
        engine.request_senders.set(vec![direct_tx]).unwrap();

        let request = PreprocessedRequest::builder()
            .model("mock".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(StopConditions {
                max_tokens: Some(1),
                ..Default::default()
            })
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .build()
            .unwrap();

        let stream = engine
            .generate(SingleIn::with_id_and_metadata(
                request,
                uuid.to_string(),
                BTreeMap::new(),
            ))
            .await
            .unwrap();
        let signal_tx = engine
            .active_requests
            .get(&uuid)
            .expect("generate must register the request channel")
            .value()
            .clone();
        (stream, signal_tx)
    }

    /// The relay is the user-facing half of the feature: the scheduler's admission
    /// cache truth (`OutputSignal.cached_tokens`) must ride the first stream chunk
    /// and repeat cumulative totals on the final length chunk (OpenAI convention).
    /// A request that completes on its first signal exercises both chunks.
    #[tokio::test]
    async fn first_and_final_chunk_carry_admission_cache_truth() {
        let uuid = Uuid::from_u128(100);
        let (mut stream, signal_tx) = wired_prefill_stream(uuid).await;

        // The request's only signal: a completed prefill with 0 cached tokens.
        signal_tx
            .send(OutputSignal {
                uuid,
                completed: true,
                handoff_delay_ms: None,
                cached_tokens: Some(0),
            })
            .unwrap();

        // First chunk carries the admission cache truth.
        let first = stream.next().await.unwrap();
        assert_eq!(first.token_ids.len(), 1);
        assert_eq!(
            first.completion_usage,
            Some(CompletionUsage {
                prompt_tokens: 3,
                completion_tokens: 1,
                total_tokens: 4,
                prompt_tokens_details: Some(PromptTokensDetails {
                    audio_tokens: None,
                    cached_tokens: Some(0),
                }),
                completion_tokens_details: None,
            })
        );

        // Final chunk repeats cumulative totals.
        let mut expected_finish = LLMEngineOutput::length();
        expected_finish.completion_usage = Some(CompletionUsage {
            prompt_tokens: 3,
            completion_tokens: 1,
            total_tokens: 4,
            prompt_tokens_details: Some(PromptTokensDetails {
                audio_tokens: None,
                cached_tokens: Some(0),
            }),
            completion_tokens_details: None,
        });
        assert_eq!(stream.next().await.unwrap(), expected_finish);
        assert!(stream.next().await.is_none());
    }

    /// Multi-signal stream: the first chunk carries the admission cache truth,
    /// later chunks carry none, and the final chunk repeats cumulative totals
    /// from the stored value — not the current signal's, which is None.
    #[tokio::test]
    async fn final_chunk_repeats_cumulative_totals_across_signals() {
        let uuid = Uuid::from_u128(101);
        let (mut stream, signal_tx) = wired_prefill_stream(uuid).await;

        // First signal: admission cache truth, request still decoding.
        signal_tx
            .send(OutputSignal {
                uuid,
                completed: false,
                handoff_delay_ms: None,
                cached_tokens: Some(8),
            })
            .unwrap();
        // Terminal signal: no cache truth, request complete.
        signal_tx
            .send(OutputSignal {
                uuid,
                completed: true,
                handoff_delay_ms: None,
                cached_tokens: None,
            })
            .unwrap();

        // First chunk carries the admission cache truth.
        let first = stream.next().await.unwrap();
        assert_eq!(
            first.completion_usage,
            Some(CompletionUsage {
                prompt_tokens: 3,
                completion_tokens: 1,
                total_tokens: 4,
                prompt_tokens_details: Some(PromptTokensDetails {
                    audio_tokens: None,
                    cached_tokens: Some(8),
                }),
                completion_tokens_details: None,
            })
        );

        // Intermediate chunk: later signals carry no cache truth.
        let middle = stream.next().await.unwrap();
        assert_eq!(middle.completion_usage, None);

        // Final chunk repeats cumulative totals from the stored value.
        let mut expected_finish = LLMEngineOutput::length();
        expected_finish.completion_usage = Some(CompletionUsage {
            prompt_tokens: 3,
            completion_tokens: 2,
            total_tokens: 5,
            prompt_tokens_details: Some(PromptTokensDetails {
                audio_tokens: None,
                cached_tokens: Some(8),
            }),
            completion_tokens_details: None,
        });
        assert_eq!(stream.next().await.unwrap(), expected_finish);
        assert!(stream.next().await.is_none());
    }
}
