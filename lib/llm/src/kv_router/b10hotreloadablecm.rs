// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hot-reloadable configuration for B10 KV Router
//!
//! This module provides automatic reloading of router configuration from a YAML file
//! specified by the DYN_LLMAPI_CONFIG_PATH environment variable, typically pointing to
//! /configs/llm_api_config_router.yaml.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{
    Arc, OnceLock, RwLock,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::Duration;

use std::collections::HashMap;

const DEFAULT_CONFIG_PATH: &str = "/configs/llm_api_config_router.yaml";
const RELOAD_INTERVAL_SECS: u64 = 15; // Reload every 15 seconds
const DEFAULT_ROUTER_ACTIVE_REQUEST_DP_BLEND: f64 = 2.0 / 3.0;
const ROUTER_ACTIVE_REQUEST_DP_BLEND_MIN: f64 = 0.0001;
const ROUTER_ACTIVE_REQUEST_DP_BLEND_MAX: f64 = 0.9999;
const DEFAULT_ROUTER_RESIDENCY_EVICTION_COST: f64 = 0.0;
const DEFAULT_ROUTER_RESIDENCY_HALF_LIFE_SECS: f64 = 120.0;
const DEFAULT_ROUTER_ACTIVE_REQUEST_ISL_PENALTY_RAMP: (f64, f64) = (2048.0, 32_768.0);
static LOG_NO_CHANGES: AtomicBool = AtomicBool::new(false);
static ENGINE_METRICS_TOTAL_KV_BLOCKS_OVERRIDE: std::sync::LazyLock<Arc<AtomicU64>> =
    std::sync::LazyLock::new(|| Arc::new(AtomicU64::new(0)));

pub fn set_log_no_changes(enabled: bool) {
    LOG_NO_CHANGES.store(enabled, Ordering::Relaxed);
}

fn log_no_changes() -> bool {
    LOG_NO_CHANGES.load(Ordering::Relaxed)
}

fn is_warning_disabled() -> bool {
    static DISABLE_WARNING: OnceLock<bool> = OnceLock::new();
    *DISABLE_WARNING.get_or_init(|| {
        std::env::var("B10_CONFIGMAP_DISABLE_WARNING")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Partial override structure for B10 routing config, we can't reuse the B10RoutingConfig struct because of the default values
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct B10RoutingConfigOverride {
    router_temperature: Option<f64>,
    #[serde(alias = "router_prefill_block_weight")]
    router_overlap_score_weight: Option<f64>,
    router_decode_block_weight: Option<f64>,
    router_prefill_token_discount: Option<f64>,
    router_decode_token_discount: Option<f64>,
    router_active_request_weight: Option<f64>,
    router_active_request_dp_blend: Option<f64>,
    router_active_replicas: Option<usize>,
    router_cache_miss_weight: Option<f64>,
    router_cache_miss_min_isl: Option<usize>,
    router_residency_eviction_cost: Option<f64>,
    router_residency_half_life: Option<f64>,
    router_queue_threshold: Option<Option<f64>>,
    router_queue_threshold_decode_tokens: Option<u64>,
    router_active_request_isl_mismatch_penalty: Option<f64>,
    router_active_request_isl_penalty_ramp: Option<(f64, f64)>,
}

/// B10 Routing configuration parameters
/// subset of Pytorch B10 routing config. Keep in sync with `B10RoutingConfig` pydantic model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct B10RoutingConfig {
    #[serde(default = "default_router_temperature")]
    pub router_temperature: f64,

    #[serde(
        default = "default_router_overlap_score_weight",
        alias = "router_prefill_block_weight"
    )]
    pub router_overlap_score_weight: f64,

    /// Multiplier applied to the decode-block term of the routing logit.
    /// Mirrors `router_overlap_score_weight` (the prefill-block multiplier) so
    /// the prefill and decode contributions to the logit can be scaled
    /// independently. Default 1.0 preserves the prior behavior of adding the
    /// raw decode-block count.
    #[serde(default = "default_router_decode_block_weight")]
    pub router_decode_block_weight: f64,

    #[serde(default = "default_router_prefill_token_discount")]
    pub router_prefill_token_discount: f64,

    #[serde(default = "default_router_decode_token_discount")]
    pub router_decode_token_discount: f64,

    #[serde(default = "default_router_active_request_weight")]
    pub router_active_request_weight: f64,

    #[serde(default = "default_router_active_request_dp_blend")]
    pub router_active_request_dp_blend: f64,

    #[serde(default = "default_router_active_replicas")]
    pub router_active_replicas: usize,

    #[serde(default = "default_router_cache_miss_weight")]
    pub router_cache_miss_weight: f64,

    #[serde(default = "default_router_cache_miss_min_isl")]
    pub router_cache_miss_min_isl: usize,

    #[serde(default = "default_router_residency_eviction_cost")]
    pub router_residency_eviction_cost: f64,

    /// Half-life in seconds for residency eviction recency cost.
    #[serde(default = "default_router_residency_half_life")]
    pub router_residency_half_life: f64,

    /// Queue admission threshold fraction of max_num_batched_tokens.
    /// None means "not configured here"; use 0 to explicitly disable queueing.
    #[serde(default)]
    pub router_queue_threshold: Option<f64>,

    /// Absolute threshold for the median per-worker decode tokens inflight.
    /// When the median of (active decode blocks × block_size) across all workers
    /// exceeds this value, requests are backpressured (queued or rejected)
    /// in addition to the prefill-busy check. 0 or None disables the check.
    #[serde(default)]
    pub router_queue_threshold_decode_tokens: Option<u64>,

    #[serde(default)]
    pub router_active_request_isl_mismatch_penalty: f64,

    #[serde(default = "default_router_active_request_isl_penalty_ramp")]
    pub router_active_request_isl_penalty_ramp: (f64, f64),
}

impl B10RoutingConfig {
    fn apply_override(&mut self, overrides: &B10RoutingConfigOverride) {
        if let Some(value) = overrides.router_temperature {
            self.router_temperature = value;
        }
        if let Some(value) = overrides.router_overlap_score_weight {
            self.router_overlap_score_weight = value;
        }
        if let Some(value) = overrides.router_decode_block_weight {
            self.router_decode_block_weight = value;
        }
        if let Some(value) = overrides.router_prefill_token_discount {
            self.router_prefill_token_discount = value;
        }
        if let Some(value) = overrides.router_decode_token_discount {
            self.router_decode_token_discount = value;
        }
        if let Some(value) = overrides.router_active_request_weight {
            self.router_active_request_weight = value;
        }
        if let Some(value) = overrides.router_active_request_dp_blend {
            self.router_active_request_dp_blend = value;
        }
        if let Some(value) = overrides.router_active_replicas {
            self.router_active_replicas = value;
        }
        if let Some(value) = overrides.router_cache_miss_weight {
            self.router_cache_miss_weight = value;
        }
        if let Some(value) = overrides.router_cache_miss_min_isl {
            self.router_cache_miss_min_isl = value;
        }
        if let Some(value) = overrides.router_residency_eviction_cost {
            self.router_residency_eviction_cost = value;
        }
        if let Some(value) = overrides.router_residency_half_life {
            self.router_residency_half_life = value;
        }
        if let Some(value) = overrides.router_queue_threshold {
            self.router_queue_threshold = value;
        }
        if let Some(value) = overrides.router_queue_threshold_decode_tokens {
            self.router_queue_threshold_decode_tokens = Some(value);
        }
        if let Some(value) = overrides.router_active_request_isl_mismatch_penalty {
            self.router_active_request_isl_mismatch_penalty = value;
        }
        if let Some(value) = overrides.router_active_request_isl_penalty_ramp {
            self.router_active_request_isl_penalty_ramp = value;
        }
    }
}

impl Default for B10RoutingConfig {
    fn default() -> Self {
        Self {
            router_temperature: default_router_temperature(),
            router_overlap_score_weight: default_router_overlap_score_weight(),
            router_decode_block_weight: default_router_decode_block_weight(),
            router_prefill_token_discount: default_router_prefill_token_discount(),
            router_decode_token_discount: default_router_decode_token_discount(),
            router_active_request_weight: default_router_active_request_weight(),
            router_active_request_dp_blend: default_router_active_request_dp_blend(),
            router_active_replicas: default_router_active_replicas(),
            router_cache_miss_weight: default_router_cache_miss_weight(),
            router_cache_miss_min_isl: default_router_cache_miss_min_isl(),
            router_residency_eviction_cost: default_router_residency_eviction_cost(),
            router_residency_half_life: default_router_residency_half_life(),
            router_queue_threshold: None,
            router_queue_threshold_decode_tokens: None,
            router_active_request_isl_mismatch_penalty: 0.0,
            router_active_request_isl_penalty_ramp: default_router_active_request_isl_penalty_ramp(
            ),
        }
    }
}

fn default_router_temperature() -> f64 {
    std::env::var("KV_ROUTER_TEMPERATURE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.01)
}

fn default_router_overlap_score_weight() -> f64 {
    std::env::var("KV_ROUTER_OVERLAP_SCORE_WEIGHT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3.5)
}

fn default_router_decode_block_weight() -> f64 {
    std::env::var("B10_KV_ROUTER_DECODE_BLOCK_WEIGHT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0)
}

fn default_router_prefill_token_discount() -> f64 {
    std::env::var("B10_KV_ROUTER_PREFILL_TOKEN_DISCOUNT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.35)
}

fn default_router_decode_token_discount() -> f64 {
    std::env::var("B10_KV_ROUTER_DECODE_TOKEN_DISCOUNT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.8)
}

fn default_router_active_request_weight() -> f64 {
    std::env::var("B10_KV_ROUTER_ACTIVE_REQUEST_WEIGHT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0)
}

fn default_router_active_request_dp_blend() -> f64 {
    std::env::var("B10_KV_ROUTER_ACTIVE_REQUEST_DP_BLEND")
        .ok()
        .and_then(|s| s.parse().ok())
        .map(sanitize_router_active_request_dp_blend)
        .unwrap_or(DEFAULT_ROUTER_ACTIVE_REQUEST_DP_BLEND)
}

fn sanitize_router_active_request_dp_blend(blend: f64) -> f64 {
    if !blend.is_finite() {
        tracing::error!(
            configured_blend = ?blend,
            sanitized_blend = DEFAULT_ROUTER_ACTIVE_REQUEST_DP_BLEND,
            "router_active_request_dp_blend must be finite, using default"
        );
        return DEFAULT_ROUTER_ACTIVE_REQUEST_DP_BLEND;
    }

    let sanitized = blend.clamp(
        ROUTER_ACTIVE_REQUEST_DP_BLEND_MIN,
        ROUTER_ACTIVE_REQUEST_DP_BLEND_MAX,
    );
    if sanitized != blend {
        tracing::error!(
            configured_blend = blend,
            sanitized_blend = sanitized,
            min_blend = ROUTER_ACTIVE_REQUEST_DP_BLEND_MIN,
            max_blend = ROUTER_ACTIVE_REQUEST_DP_BLEND_MAX,
            "router_active_request_dp_blend outside bounds, clamping"
        );
    }

    sanitized
}

fn default_router_cache_miss_weight() -> f64 {
    std::env::var("B10_KV_ROUTER_CACHE_MISS_WEIGHT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.02)
}

fn default_router_cache_miss_min_isl() -> usize {
    std::env::var("B10_KV_ROUTER_CACHE_MISS_MIN_ISL")
        .ok()
        .and_then(|s| s.parse().ok())
        // a new worker coming up does not have the system prompt. If a isl is only 512 tokens, its around system prompt.
        .unwrap_or(4096)
}

fn default_router_residency_eviction_cost() -> f64 {
    std::env::var("B10_KV_ROUTER_RESIDENCY_EVICTION_COST")
        .ok()
        .and_then(|s| s.parse().ok())
        .map(sanitize_router_residency_eviction_cost)
        .unwrap_or(DEFAULT_ROUTER_RESIDENCY_EVICTION_COST)
}

fn default_router_residency_half_life() -> f64 {
    std::env::var("B10_KV_ROUTER_RESIDENCY_HALF_LIFE")
        .ok()
        .and_then(|s| s.parse().ok())
        .map(sanitize_router_residency_half_life)
        .unwrap_or(DEFAULT_ROUTER_RESIDENCY_HALF_LIFE_SECS)
}

fn default_router_active_request_isl_penalty_ramp() -> (f64, f64) {
    DEFAULT_ROUTER_ACTIVE_REQUEST_ISL_PENALTY_RAMP
}

fn sanitize_router_residency_eviction_cost(cost: f64) -> f64 {
    if cost.is_finite() && cost >= 0.0 {
        return cost;
    }

    tracing::error!(
        configured_cost = ?cost,
        sanitized_cost = DEFAULT_ROUTER_RESIDENCY_EVICTION_COST,
        "router_residency_eviction_cost must be finite and >= 0, using default"
    );
    DEFAULT_ROUTER_RESIDENCY_EVICTION_COST
}

fn sanitize_router_residency_half_life(half_life: f64) -> f64 {
    if half_life.is_finite() && half_life > 0.0 {
        return half_life;
    }

    tracing::error!(
        configured_half_life = ?half_life,
        sanitized_half_life = DEFAULT_ROUTER_RESIDENCY_HALF_LIFE_SECS,
        "router_residency_half_life must be finite and > 0, using default"
    );
    DEFAULT_ROUTER_RESIDENCY_HALF_LIFE_SECS
}

fn sanitize_engine_metrics_total_kv_blocks_override(value: Option<u64>) -> Option<u64> {
    match value {
        Some(0) => {
            tracing::error!(
                "engine_metrics_total_kv_blocks_override must be > 0 when set, ignoring override"
            );
            None
        }
        other => other,
    }
}

fn set_engine_metrics_total_kv_blocks_override(value: Option<u64>) {
    ENGINE_METRICS_TOTAL_KV_BLOCKS_OVERRIDE.store(value.unwrap_or(0), Ordering::Relaxed);
}

/// Override configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OverrideConfig {
    #[serde(default)]
    b10_routing_config: Option<B10RoutingConfigOverride>,

    #[serde(default)]
    tensor_parallel_size: Option<usize>,

    #[serde(default)]
    enable_attention_dp: Option<bool>,

    #[serde(default)]
    engine_metrics_total_kv_blocks_override: Option<u64>,
}

/// Root configuration structure for parsing YAML
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LLMConfig {
    #[serde(default)]
    b10_routing_config: B10RoutingConfig,

    #[serde(default)]
    override_args: Option<HashMap<String, OverrideConfig>>,

    // Runtime config fields
    #[serde(default)]
    tensor_parallel_size: Option<usize>,

    #[serde(default)]
    enable_attention_dp: Option<bool>,

    #[serde(default)]
    engine_metrics_total_kv_blocks_override: Option<u64>,
}

/// Runtime configuration fields from llm_api_config_router.yaml
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LLMRuntimeConfig {
    pub tensor_parallel_size: Option<usize>,
    pub enable_attention_dp: Option<bool>,
}

impl LLMRuntimeConfig {
    // Compute data_parallel_size based on the rules:
    // - If enable_attention_dp is Some(true), return tensor_parallel_size
    // if enable_attention_dp is true, in trt, will use size 1 dp ranks, so data_parallel_size = tensor_parallel_size
    // - Otherwise, return 1
    pub fn compute_data_parallel_size(&self) -> Option<usize> {
        match self.enable_attention_dp {
            Some(true) => self.tensor_parallel_size,
            _ => Some(1),
        }
    }
}

/// Convenience function to get data parallel size
/// Returns the computed value based on enable_attention_dp and tensor_parallel_size
pub fn get_data_parallel_size() -> Option<usize> {
    let runtime = &get_config().get().runtime;
    runtime.compute_data_parallel_size()
}

pub fn get_engine_metrics_total_kv_blocks_override() -> Option<u64> {
    match ENGINE_METRICS_TOTAL_KV_BLOCKS_OVERRIDE.load(Ordering::Relaxed) {
        0 => None,
        value => Some(value),
    }
}

/// Unified config containing both routing and runtime configuration
#[derive(Debug, Clone, PartialEq)]
pub struct UnifiedConfig {
    pub routing: B10RoutingConfig,
    pub router_active_replicas: usize,
    pub runtime: LLMRuntimeConfig,
}

impl Default for UnifiedConfig {
    fn default() -> Self {
        Self {
            routing: B10RoutingConfig::default(),
            router_active_replicas: default_router_active_replicas(),
            runtime: LLMRuntimeConfig::default(),
        }
    }
}

fn default_router_active_replicas() -> usize {
    1
}

/// Hot-reloadable config manager with unified config
pub struct HotReloadableConfig {
    config: Arc<RwLock<UnifiedConfig>>,
    config_path: PathBuf,
}

impl Default for HotReloadableConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl HotReloadableConfig {
    /// Create a new hot-reloadable config manager
    pub fn new() -> Self {
        let config_path_env = std::env::var("DYN_LLMAPI_CONFIG_PATH");

        let (initial_config, config_path) = if let Ok(path_str) = config_path_env {
            let config_path = PathBuf::from(path_str);
            // Only load from file if env var is set
            let config = Self::load_config(&config_path).unwrap_or_else(|e| {
                if !is_warning_disabled() {
                    tracing::warn!(
                        "Failed to load config from {:?}: {:?}, using defaults",
                        config_path,
                        e
                    );
                }
                UnifiedConfig::default()
            });
            (config, config_path)
        } else {
            // No env var set - use fast defaults and default path
            if !is_warning_disabled() {
                tracing::warn!(
                    "DYN_LLMAPI_CONFIG_PATH not set, using default UnifiedConfig values"
                );
            }
            (UnifiedConfig::default(), PathBuf::from(DEFAULT_CONFIG_PATH))
        };

        Self {
            config: Arc::new(RwLock::new(initial_config)),
            config_path,
        }
    }

    fn validate_config(path: &PathBuf) -> Option<LLMConfig> {
        if !path.exists() || !path.is_file() {
            if !is_warning_disabled() {
                tracing::warn!("Config file {:?} does not exist or is not a file", path);
            }
            return None;
        }

        let contents = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                if !is_warning_disabled() {
                    tracing::warn!("Failed to read config file {:?}: {:?}", path, e);
                }
                return None;
            }
        };
        match serde_yaml::from_str(&contents) {
            Ok(config) => Some(config),
            Err(e) => {
                if !is_warning_disabled() {
                    tracing::warn!("Failed to parse YAML config from {:?}: {:?}", path, e);
                }
                None
            }
        }
    }

    /// Load config from file
    fn load_config(path: &PathBuf) -> Result<UnifiedConfig> {
        let root_config = Self::validate_config(path);

        if root_config.is_none() {
            return Err(anyhow::anyhow!(
                "Failed to load or validate config from {:?}",
                path
            ));
        }

        let mut root_config = root_config.unwrap();

        // Apply overrides if present
        if let Ok(override_group) = std::env::var("ENGINE_ARGS_OVERRIDE_GROUP")
            && !override_group.is_empty()
            && let Some(overrides) = &root_config.override_args
            && let Some(group_config) = overrides.get(&override_group)
        {
            if log_no_changes() {
                tracing::info!("Applying override group '{}'", override_group);
            }
            if let Some(routing_override) = &group_config.b10_routing_config {
                root_config
                    .b10_routing_config
                    .apply_override(routing_override);
            }
            if let Some(tp) = group_config.tensor_parallel_size {
                root_config.tensor_parallel_size = Some(tp);
            }
            if let Some(adp) = group_config.enable_attention_dp {
                root_config.enable_attention_dp = Some(adp);
            }
            if let Some(total_kv_blocks) = group_config.engine_metrics_total_kv_blocks_override {
                root_config.engine_metrics_total_kv_blocks_override = Some(total_kv_blocks);
            }
        }

        let routing = &mut root_config.b10_routing_config;
        routing.router_active_request_dp_blend =
            sanitize_router_active_request_dp_blend(routing.router_active_request_dp_blend);
        routing.router_residency_eviction_cost =
            sanitize_router_residency_eviction_cost(routing.router_residency_eviction_cost);
        routing.router_residency_half_life =
            sanitize_router_residency_half_life(routing.router_residency_half_life);
        // Build UnifiedConfig with separate routing and runtime configs
        let engine_metrics_total_kv_blocks_override =
            sanitize_engine_metrics_total_kv_blocks_override(
                root_config.engine_metrics_total_kv_blocks_override,
            );
        set_engine_metrics_total_kv_blocks_override(engine_metrics_total_kv_blocks_override);

        let runtime_config = LLMRuntimeConfig {
            tensor_parallel_size: root_config.tensor_parallel_size,
            enable_attention_dp: root_config.enable_attention_dp,
        };

        let unified_config = UnifiedConfig {
            router_active_replicas: root_config.b10_routing_config.router_active_replicas,
            routing: root_config.b10_routing_config,
            runtime: runtime_config,
        };

        // Compute data_parallel_size for logging
        let data_parallel_size = unified_config.runtime.compute_data_parallel_size();

        dynamo_kv_router::sequences::set_token_load_discounts(
            unified_config.routing.router_prefill_token_discount,
            unified_config.routing.router_decode_token_discount,
        );

        let decode_tokens_threshold = unified_config
            .routing
            .router_queue_threshold_decode_tokens
            .unwrap_or(0);

        if decode_tokens_threshold > 0 && decode_tokens_threshold < 1000 {
            tracing::warn!(
                router_queue_threshold_decode_tokens = decode_tokens_threshold,
                "router_queue_threshold_decode_tokens is set below 1000; typical values are 50k-2000k. \
                 This may cause excessive backpressure."
            );
        }

        dynamo_kv_router::scheduling::queue::set_router_queue_threshold_decode_tokens(
            decode_tokens_threshold,
        );

        if log_no_changes() {
            tracing::info!(
                "Loaded config from {:?}: prefill_discount={}, decode_discount={}, temperature={}, active_request_dp_blend={}, residency_eviction_cost={}, residency_half_life={}, router_active_replicas={}, tensor_parallel_size={:?}, enable_attention_dp={:?}, data_parallel_size={:?}, engine_metrics_total_kv_blocks_override={:?}, router_queue_threshold_decode_tokens={:?}",
                path,
                unified_config.routing.router_prefill_token_discount,
                unified_config.routing.router_decode_token_discount,
                unified_config.routing.router_temperature,
                unified_config.routing.router_active_request_dp_blend,
                unified_config.routing.router_residency_eviction_cost,
                unified_config.routing.router_residency_half_life,
                unified_config.router_active_replicas,
                unified_config.runtime.tensor_parallel_size,
                unified_config.runtime.enable_attention_dp,
                data_parallel_size,
                engine_metrics_total_kv_blocks_override,
                unified_config.routing.router_queue_threshold_decode_tokens,
            );
        }

        Ok(unified_config)
    }

    /// Get a clone of the current config
    pub fn get(&self) -> UnifiedConfig {
        self.config.read().unwrap().clone()
    }

    /// Start background task to reload config periodically
    pub fn start_reloader(self: Arc<Self>) {
        std::thread::spawn(move || {
            tracing::info!(
                "Starting B10RouterConfig hot-reloader thread, monitoring {:?}",
                self.config_path
            );
            loop {
                match Self::load_config(&self.config_path) {
                    Ok(new_config) => {
                        if let Ok(mut config) = self.config.write() {
                            let config_changed = *config != new_config;
                            *config = new_config;

                            if config_changed || log_no_changes() {
                                tracing::info!(
                                    "B10RouterConfig hot-reload {}: {:?}",
                                    if config_changed {
                                        "(HAS CHANGED!)"
                                    } else {
                                        "(no change)"
                                    },
                                    config
                                );
                            }
                        }
                    }
                    Err(e) => {
                        if !is_warning_disabled() {
                            tracing::warn!("Failed to hot-reload b10 router config: {:?}", e);
                        }
                    }
                }
                // Sleep after each reload attempt
                std::thread::sleep(Duration::from_secs(RELOAD_INTERVAL_SECS));
            }
        });
    }
}

/// Global config instance
static CONFIG: std::sync::LazyLock<Arc<HotReloadableConfig>> = std::sync::LazyLock::new(|| {
    let config = Arc::new(HotReloadableConfig::new());
    config.clone().start_reloader();
    config
});

/// Get the global config instance
pub fn get_config() -> Arc<HotReloadableConfig> {
    CONFIG.clone()
}

/// Convenience function to get prefill token discount
pub fn get_prefill_token_discount() -> f64 {
    get_config().get().routing.router_prefill_token_discount
}

/// Convenience function to get decode token discount
pub fn get_decode_token_discount() -> f64 {
    get_config().get().routing.router_decode_token_discount
}

/// Convenience function to get router temperature
pub fn get_router_temperature() -> f64 {
    get_config().get().routing.router_temperature
}

/// Convenience function to get router overlap score weight
pub fn get_router_overlap_score_weight() -> f64 {
    get_config().get().routing.router_overlap_score_weight
}

/// Convenience function to get router decode block weight
pub fn get_router_decode_block_weight() -> f64 {
    get_config().get().routing.router_decode_block_weight
}

/// Convenience function to get router active request weight
pub fn get_active_request_weight() -> f64 {
    get_config().get().routing.router_active_request_weight
}

/// Convenience function to get router active request DP blend
pub fn get_active_request_dp_blend() -> f64 {
    get_config().get().routing.router_active_request_dp_blend
}

/// Convenience function to get router cache miss weight
pub fn get_router_cache_miss_weight() -> f64 {
    get_config().get().routing.router_cache_miss_weight
}

/// Convenience function to get router cache miss min isl
pub fn get_router_cache_miss_min_isl() -> usize {
    get_config().get().routing.router_cache_miss_min_isl
}

/// Convenience function to get router residency eviction cost weight
pub fn get_router_residency_eviction_cost() -> f64 {
    get_config().get().routing.router_residency_eviction_cost
}

/// Convenience function to get router residency half-life in seconds
pub fn get_router_residency_half_life() -> f64 {
    get_config().get().routing.router_residency_half_life
}

pub fn get_router_queue_threshold() -> Option<f64> {
    get_config().get().routing.router_queue_threshold
}

pub fn get_router_queue_threshold_decode_tokens() -> Option<u64> {
    get_config()
        .get()
        .routing
        .router_queue_threshold_decode_tokens
}

/// Convenience function to get tensor parallel size
pub fn get_tensor_parallel_size() -> Option<usize> {
    get_config().get().runtime.tensor_parallel_size
}

/// Convenience function to get enable attention dp
pub fn get_enable_attention_dp() -> Option<bool> {
    get_config().get().runtime.enable_attention_dp
}

/// Convenience function to get router active replicas.
pub fn get_router_active_replicas() -> usize {
    get_config().get().router_active_replicas
}

pub fn validate_config() -> bool {
    // gets result of validation + starts lazy reloader if not already started
    HotReloadableConfig::validate_config(&CONFIG.config_path).is_some()
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::*;

    #[test]
    fn test_default_config() {
        let config = HotReloadableConfig::default();
        let unified_config = config.get();

        assert_eq!(unified_config.routing.router_temperature, 0.01);
        assert_eq!(unified_config.routing.router_overlap_score_weight, 3.5);
        assert_eq!(unified_config.routing.router_decode_block_weight, 1.0);
        assert_eq!(unified_config.routing.router_prefill_token_discount, 0.35);
        assert_eq!(unified_config.routing.router_decode_token_discount, 0.8);
        assert_eq!(
            unified_config.routing.router_active_request_dp_blend,
            2.0 / 3.0
        );
        assert_eq!(unified_config.router_active_replicas, 1);
    }

    #[test]
    fn test_config_access() {
        let prefill_discount = get_prefill_token_discount();
        let decode_discount = get_decode_token_discount();
        let active_request_dp_blend = get_active_request_dp_blend();
        assert!(prefill_discount >= 0.0);
        assert!(decode_discount >= 0.0);
        assert!(active_request_dp_blend >= 0.0);
    }

    #[test]
    #[serial]
    fn test_override_args_applied() {
        use std::io::Write;

        // Create a temp config file
        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        let config_content = r#"
b10_routing_config:
  router_temperature: 0.15
  router_overlap_score_weight: 3.5
  router_active_request_dp_blend: 0.2
  router_queue_threshold: 0.25
  router_active_replicas: 2
tensor_parallel_size: 8
enable_attention_dp: true
engine_metrics_total_kv_blocks_override: 150000

override_args:
  test_group:
    engine_metrics_total_kv_blocks_override: 250000
    b10_routing_config:
      router_temperature: 0.99
      router_overlap_score_weight: 1.0
      router_active_request_dp_blend: 0.75
      router_queue_threshold: 0
      router_active_replicas: 3
"#;
        write!(temp_file, "{}", config_content).unwrap();
        let path = temp_file.path().to_path_buf();

        // Set env vars
        unsafe {
            std::env::set_var("ENGINE_ARGS_OVERRIDE_GROUP", "test_group");
        }

        // Load config
        let config = HotReloadableConfig::load_config(&path).unwrap();

        // Check overrides applied
        assert_eq!(config.routing.router_temperature, 0.99);
        assert_eq!(config.routing.router_overlap_score_weight, 1.0);
        assert_eq!(config.routing.router_active_request_dp_blend, 0.75);
        assert_eq!(config.routing.router_queue_threshold, Some(0.0));
        assert_eq!(config.router_active_replicas, 3);
        // Check runtime config and computed data_parallel_size
        assert_eq!(config.runtime.tensor_parallel_size, Some(8));
        assert_eq!(config.runtime.enable_attention_dp, Some(true));
        assert_eq!(get_engine_metrics_total_kv_blocks_override(), Some(250000));
        assert_eq!(config.runtime.compute_data_parallel_size(), Some(8)); // Because enable_attention_dp=true returns tensor_parallel_size

        // Cleanup
        unsafe {
            std::env::remove_var("ENGINE_ARGS_OVERRIDE_GROUP");
        }
    }

    #[test]
    #[serial]
    fn test_prefill_block_weight_alias_populates_overlap_score_weight() {
        use std::io::Write;

        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        let config_content = r#"
b10_routing_config:
  router_prefill_block_weight: 5.5
"#;
        write!(temp_file, "{}", config_content).unwrap();
        let path = temp_file.path().to_path_buf();

        unsafe {
            std::env::remove_var("ENGINE_ARGS_OVERRIDE_GROUP");
        }

        let config = HotReloadableConfig::load_config(&path).unwrap();

        // The alias populates the same field as router_overlap_score_weight.
        assert_eq!(config.routing.router_overlap_score_weight, 5.5);
    }

    #[test]
    #[serial]
    fn test_decode_block_weight_default_and_override() {
        use std::io::Write;

        // Default is 1.0 (preserves prior behavior).
        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        let config_content = r#"
b10_routing_config:
  router_temperature: 0.01
"#;
        write!(temp_file, "{}", config_content).unwrap();
        let path = temp_file.path().to_path_buf();
        unsafe {
            std::env::remove_var("ENGINE_ARGS_OVERRIDE_GROUP");
        }
        let config = HotReloadableConfig::load_config(&path).unwrap();
        assert_eq!(config.routing.router_decode_block_weight, 1.0);

        // Override group applies the decode weight.
        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        let config_content = r#"
b10_routing_config:
  router_decode_block_weight: 0.5

override_args:
  test_group:
    b10_routing_config:
      router_decode_block_weight: 2.0
"#;
        write!(temp_file, "{}", config_content).unwrap();
        let path = temp_file.path().to_path_buf();
        unsafe {
            std::env::set_var("ENGINE_ARGS_OVERRIDE_GROUP", "test_group");
        }
        let config = HotReloadableConfig::load_config(&path).unwrap();
        assert_eq!(config.routing.router_decode_block_weight, 2.0);
        unsafe {
            std::env::remove_var("ENGINE_ARGS_OVERRIDE_GROUP");
        }
    }

    #[test]
    #[serial]
    fn test_override_args_preserve_router_active_replicas_when_missing() {
        use std::io::Write;

        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        let config_content = r#"
b10_routing_config:
  router_temperature: 0.15
  router_active_replicas: 4

override_args:
  test_group:
    b10_routing_config:
      router_temperature: 0.99
"#;
        write!(temp_file, "{}", config_content).unwrap();
        let path = temp_file.path().to_path_buf();

        unsafe {
            std::env::set_var("ENGINE_ARGS_OVERRIDE_GROUP", "test_group");
        }

        let config = HotReloadableConfig::load_config(&path).unwrap();

        assert_eq!(config.routing.router_temperature, 0.99);
        assert_eq!(config.router_active_replicas, 4);

        unsafe {
            std::env::remove_var("ENGINE_ARGS_OVERRIDE_GROUP");
        }
    }

    #[test]
    #[serial]
    fn test_override_args_not_applied_when_env_missing() {
        use std::io::Write;

        // Create a temp config file
        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        let config_content = r#"
b10_routing_config:
  router_temperature: 0.15
  router_overlap_score_weight: 3.5
  router_active_request_dp_blend: 0.4
  router_queue_threshold: 0.25
  router_active_replicas: 2

override_args:
  test_group:
    b10_routing_config:
      router_temperature: 0.99
      router_overlap_score_weight: 1.0
"#;
        write!(temp_file, "{}", config_content).unwrap();
        let path = temp_file.path().to_path_buf();

        // Ensure env var is NOT set
        // Note: We need a mutex to ensure tests don't trample on each other's env vars
        // but for this specific test file, we can just ensure we clear it.
        unsafe {
            std::env::remove_var("ENGINE_ARGS_OVERRIDE_GROUP");
        }

        // Load config
        let config = HotReloadableConfig::load_config(&path).unwrap();

        // Check overrides NOT applied
        assert_eq!(config.routing.router_temperature, 0.15);
        assert_eq!(config.routing.router_overlap_score_weight, 3.5);
        assert_eq!(config.routing.router_active_request_dp_blend, 0.4);
        assert_eq!(config.routing.router_queue_threshold, Some(0.25));
        assert_eq!(config.router_active_replicas, 2);
    }

    #[test]
    fn test_active_request_dp_blend_sanitization() {
        use std::io::Write;

        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        let config_content = r#"
b10_routing_config:
  router_active_request_dp_blend: 2.0
"#;
        write!(temp_file, "{}", config_content).unwrap();
        let path = temp_file.path().to_path_buf();

        let config = HotReloadableConfig::load_config(&path).unwrap();
        assert_eq!(
            config.routing.router_active_request_dp_blend,
            ROUTER_ACTIVE_REQUEST_DP_BLEND_MAX
        );
        assert_eq!(
            sanitize_router_active_request_dp_blend(f64::NAN),
            DEFAULT_ROUTER_ACTIVE_REQUEST_DP_BLEND
        );
        assert_eq!(
            sanitize_router_active_request_dp_blend(f64::INFINITY),
            DEFAULT_ROUTER_ACTIVE_REQUEST_DP_BLEND
        );
        assert_eq!(
            sanitize_router_active_request_dp_blend(-1.0),
            ROUTER_ACTIVE_REQUEST_DP_BLEND_MIN
        );
    }

    #[test]
    #[serial]
    fn test_nested_router_active_replicas() {
        use std::io::Write;

        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        let config_content = r#"
b10_routing_config:
  router_temperature: 0.15
  router_active_replicas: 2
"#;
        write!(temp_file, "{}", config_content).unwrap();
        let path = temp_file.path().to_path_buf();

        unsafe {
            std::env::remove_var("ENGINE_ARGS_OVERRIDE_GROUP");
        }

        let config = HotReloadableConfig::load_config(&path).unwrap();

        assert_eq!(config.routing.router_temperature, 0.15);
        assert_eq!(config.router_active_replicas, 2);
    }

    #[test]
    #[serial]
    fn test_override_args_nested_router_active_replicas_applied() {
        use std::io::Write;

        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        let config_content = r#"
b10_routing_config:
  router_active_replicas: 2

override_args:
  test_group:
    b10_routing_config:
      router_active_replicas: 3
"#;
        write!(temp_file, "{}", config_content).unwrap();
        let path = temp_file.path().to_path_buf();

        unsafe {
            std::env::set_var("ENGINE_ARGS_OVERRIDE_GROUP", "test_group");
        }

        let config = HotReloadableConfig::load_config(&path).unwrap();

        assert_eq!(config.router_active_replicas, 3);

        unsafe {
            std::env::remove_var("ENGINE_ARGS_OVERRIDE_GROUP");
        }
    }

    #[test]
    #[serial]
    fn test_engine_metrics_total_kv_blocks_override_sanitization() {
        use std::io::Write;

        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        let config_content = r#"
engine_metrics_total_kv_blocks_override: 0
"#;
        write!(temp_file, "{}", config_content).unwrap();
        let path = temp_file.path().to_path_buf();

        let config = HotReloadableConfig::load_config(&path).unwrap();

        assert_eq!(config.runtime.tensor_parallel_size, None);
        assert_eq!(get_engine_metrics_total_kv_blocks_override(), None);
    }

    #[test]
    fn test_data_parallel_size_computation() {
        // Test case 1: enable_attention_dp = Some(true), tp = Some(8) -> should return Some(8)
        let config1 = LLMRuntimeConfig {
            tensor_parallel_size: Some(8),
            enable_attention_dp: Some(true),
        };
        assert_eq!(config1.compute_data_parallel_size(), Some(8));

        // Test case 2: enable_attention_dp = Some(false), tp = Some(8) -> should return Some(1)
        let config2 = LLMRuntimeConfig {
            tensor_parallel_size: Some(8),
            enable_attention_dp: Some(false),
        };
        assert_eq!(config2.compute_data_parallel_size(), Some(1));

        // Test case 3: enable_attention_dp = None, tp = Some(4) -> should return Some(1)
        let config3 = LLMRuntimeConfig {
            tensor_parallel_size: Some(4),
            enable_attention_dp: None,
        };
        assert_eq!(config3.compute_data_parallel_size(), Some(1));

        // Test case 4: enable_attention_dp = Some(true), tp = None -> should return None
        let config4 = LLMRuntimeConfig {
            tensor_parallel_size: None,
            enable_attention_dp: Some(true),
        };
        assert_eq!(config4.compute_data_parallel_size(), None);

        // Test case 5: both None -> should return Some(1) (default when enable_attention_dp is not true)
        let config5 = LLMRuntimeConfig {
            tensor_parallel_size: None,
            enable_attention_dp: None,
        };
        assert_eq!(config5.compute_data_parallel_size(), Some(1));
    }
}
