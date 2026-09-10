// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

/// Listener settings apply at startup; null port disables HTTP.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GenerationCoordinatorConfig {
    pub host: IpAddr,
    pub port: Option<u16>,
    pub remotes: Option<BTreeMap<String, String>>,
}

impl Default for GenerationCoordinatorConfig {
    fn default() -> Self {
        Self {
            host: Ipv4Addr::UNSPECIFIED.into(),
            port: None,
            remotes: None,
        }
    }
}

impl GenerationCoordinatorConfig {
    pub fn validate(&self) -> Result<()> {
        if let Some(remotes) = &self.remotes {
            anyhow::ensure!(
                remotes.len() == 1,
                "coordinator requires exactly one remote backend"
            );
            let (name, endpoint) = remotes.first_key_value().expect("length checked");
            anyhow::ensure!(
                !name.trim().is_empty(),
                "coordinator backend name cannot be empty"
            );
            let url = url::Url::parse(endpoint)?;
            anyhow::ensure!(
                matches!(url.scheme(), "http" | "https") && url.host_str().is_some(),
                "coordinator backend must be an HTTP(S) URL with a host"
            );
            anyhow::ensure!(
                url.username().is_empty() && url.password().is_none() && url.fragment().is_none(),
                "coordinator backend URL cannot contain credentials or a fragment"
            );
        }
        Ok(())
    }

    pub fn listen_address(&self) -> Option<SocketAddr> {
        self.port.map(|port| SocketAddr::new(self.host, port))
    }
}

const DEFAULT_ROUTER_ACTIVE_REQUEST_DP_BLEND: f64 = 2.0 / 3.0;
const ROUTER_ACTIVE_REQUEST_DP_BLEND_MIN: f64 = 0.0001;
const ROUTER_ACTIVE_REQUEST_DP_BLEND_MAX: f64 = 0.9999;
const DEFAULT_ROUTER_RESIDENCY_EVICTION_COST: f64 = 0.0;
const DEFAULT_ROUTER_RESIDENCY_HALF_LIFE_SECS: f64 = 120.0;
const DEFAULT_ROUTER_ACTIVE_REQUEST_ISL_PENALTY_RAMP: (f64, f64) = (2048.0, 32_768.0);
/// Floor for `router_temperature`. A temperature of 0 breaks exact ties by
/// `worker_id` (u64); clamping to this tiny floor routes ties through
/// `softmax_sample` (random) instead.
const MIN_ROUTER_TEMPERATURE: f64 = 1e-12;
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
    #[serde(alias = "router_session_affinity_discount")]
    router_session_affinity_score_multiplier: Option<f64>,
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

    /// Multiplier applied to the score when a worker is an affinity cache hit.
    /// Lower values prefer the matched worker more strongly; 1.0 disables affinity weighting.
    #[serde(
        default = "default_router_session_affinity_score_multiplier",
        alias = "router_session_affinity_discount"
    )]
    pub router_session_affinity_score_multiplier: f64,

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
        if let Some(value) = overrides.router_session_affinity_score_multiplier {
            self.router_session_affinity_score_multiplier = value;
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
            router_session_affinity_score_multiplier:
                default_router_session_affinity_score_multiplier(),
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
    0.01
}

fn default_router_overlap_score_weight() -> f64 {
    3.5
}

fn default_router_decode_block_weight() -> f64 {
    1.0
}

fn default_router_prefill_token_discount() -> f64 {
    0.35
}

fn default_router_decode_token_discount() -> f64 {
    0.8
}

fn default_router_active_request_weight() -> f64 {
    0.0
}

fn default_router_active_request_dp_blend() -> f64 {
    DEFAULT_ROUTER_ACTIVE_REQUEST_DP_BLEND
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
    0.02
}

fn default_router_cache_miss_min_isl() -> usize {
    4096
}

fn default_router_session_affinity_score_multiplier() -> f64 {
    0.5
}

fn default_router_residency_eviction_cost() -> f64 {
    DEFAULT_ROUTER_RESIDENCY_EVICTION_COST
}

fn default_router_residency_half_life() -> f64 {
    DEFAULT_ROUTER_RESIDENCY_HALF_LIFE_SECS
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

/// Clamp `router_temperature` to `MIN_ROUTER_TEMPERATURE` when it is missing,
/// zero, negative, or non-finite. This guarantees the selector never takes
/// the deterministic `temperature == 0.0` branch that breaks ties by
/// `worker_id` (u64); instead ties go through `softmax_sample`, which breaks
/// them uniformly at random.
pub fn sanitize_router_temperature(value: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        return value;
    }

    tracing::error!(
        configured_temperature = ?value,
        sanitized_temperature = MIN_ROUTER_TEMPERATURE,
        "router_temperature must be finite and > 0; clamping to {:e} to avoid \
         worker_id tie-breaking",
        MIN_ROUTER_TEMPERATURE
    );
    MIN_ROUTER_TEMPERATURE
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

/// Override configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OverrideConfig {
    #[serde(default)]
    b10_generation_coordinator_config: Option<GenerationCoordinatorConfig>,
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
    b10_generation_coordinator_config: GenerationCoordinatorConfig,
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

/// Unified config containing both routing and runtime configuration
#[derive(Debug, Clone, PartialEq)]
pub struct UnifiedConfig {
    pub generation_coordinator: GenerationCoordinatorConfig,
    pub routing: B10RoutingConfig,
    pub router_active_replicas: usize,
    pub runtime: LLMRuntimeConfig,
    pub engine_metrics_total_kv_blocks_override: Option<u64>,
}

impl Default for UnifiedConfig {
    fn default() -> Self {
        Self {
            generation_coordinator: GenerationCoordinatorConfig::default(),
            routing: B10RoutingConfig::default(),
            router_active_replicas: default_router_active_replicas(),
            runtime: LLMRuntimeConfig::default(),
            engine_metrics_total_kv_blocks_override: None,
        }
    }
}

fn default_router_active_replicas() -> usize {
    1
}

impl B10RoutingConfig {
    pub(crate) fn from_env() -> Self {
        Self {
            router_temperature: {
                sanitize_router_temperature(
                    std::env::var("KV_ROUTER_TEMPERATURE")
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.01),
                )
            },
            router_overlap_score_weight: {
                std::env::var("KV_ROUTER_OVERLAP_SCORE_WEIGHT")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(3.5)
            },
            router_decode_block_weight: {
                std::env::var("B10_KV_ROUTER_DECODE_BLOCK_WEIGHT")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1.0)
            },
            router_prefill_token_discount: {
                std::env::var("B10_KV_ROUTER_PREFILL_TOKEN_DISCOUNT")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.35)
            },
            router_decode_token_discount: {
                std::env::var("B10_KV_ROUTER_DECODE_TOKEN_DISCOUNT")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.8)
            },
            router_active_request_weight: {
                std::env::var("B10_KV_ROUTER_ACTIVE_REQUEST_WEIGHT")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0)
            },
            router_active_request_dp_blend: {
                std::env::var("B10_KV_ROUTER_ACTIVE_REQUEST_DP_BLEND")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .map(sanitize_router_active_request_dp_blend)
                    .unwrap_or(DEFAULT_ROUTER_ACTIVE_REQUEST_DP_BLEND)
            },
            router_cache_miss_weight: {
                std::env::var("B10_KV_ROUTER_CACHE_MISS_WEIGHT")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.02)
            },
            router_cache_miss_min_isl: {
                std::env::var("B10_KV_ROUTER_CACHE_MISS_MIN_ISL")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    // a new worker coming up does not have the system prompt. If a isl is only 512 tokens, its around system prompt.
                    .unwrap_or(4096)
            },
            router_session_affinity_score_multiplier: {
                std::env::var("B10_KV_ROUTER_SESSION_AFFINITY_SCORE_MULTIPLIER")
                    .or_else(|_| std::env::var("B10_KV_ROUTER_SESSION_AFFINITY_DISCOUNT"))
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.5)
            },
            router_residency_eviction_cost: {
                std::env::var("B10_KV_ROUTER_RESIDENCY_EVICTION_COST")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .map(sanitize_router_residency_eviction_cost)
                    .unwrap_or(DEFAULT_ROUTER_RESIDENCY_EVICTION_COST)
            },
            router_residency_half_life: {
                std::env::var("B10_KV_ROUTER_RESIDENCY_HALF_LIFE")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .map(sanitize_router_residency_half_life)
                    .unwrap_or(DEFAULT_ROUTER_RESIDENCY_HALF_LIFE_SECS)
            },
            ..Self::default()
        }
    }
}

impl UnifiedConfig {
    /// Parse a document using explicit defaults and override selection. No process state is changed.
    pub fn parse(
        contents: &str,
        override_group: Option<&str>,
        defaults: &B10RoutingConfig,
    ) -> Result<Self> {
        let mut document: serde_yaml::Value = serde_yaml::from_str(contents)?;
        let root = document
            .as_mapping_mut()
            .ok_or_else(|| anyhow::anyhow!("configuration must be a YAML mapping"))?;
        let routing_key = serde_yaml::Value::String("b10_routing_config".into());
        let routing = root
            .entry(routing_key)
            .or_insert_with(|| serde_yaml::Value::Mapping(Default::default()));
        if let Some(routing) = routing.as_mapping_mut() {
            let serde_yaml::Value::Mapping(mut fallback) = serde_yaml::to_value(defaults)? else {
                unreachable!()
            };
            for (canonical, alias) in [
                ("router_overlap_score_weight", "router_prefill_block_weight"),
                (
                    "router_session_affinity_score_multiplier",
                    "router_session_affinity_discount",
                ),
            ] {
                if routing.contains_key(serde_yaml::Value::String(alias.into())) {
                    fallback.remove(serde_yaml::Value::String(canonical.into()));
                }
            }
            for (key, value) in fallback {
                routing.entry(key).or_insert(value);
            }
        }
        let mut root_config: LLMConfig = serde_yaml::from_value(document)?;
        if let Some(group) = override_group
            && let Some(group_config) = root_config
                .override_args
                .as_ref()
                .and_then(|groups| groups.get(group))
        {
            if let Some(coordinator) = &group_config.b10_generation_coordinator_config {
                root_config.b10_generation_coordinator_config = coordinator.clone();
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
        routing.router_temperature = sanitize_router_temperature(routing.router_temperature);
        // Build UnifiedConfig with separate routing and runtime configs
        let engine_metrics_total_kv_blocks_override =
            sanitize_engine_metrics_total_kv_blocks_override(
                root_config.engine_metrics_total_kv_blocks_override,
            );

        let runtime_config = LLMRuntimeConfig {
            tensor_parallel_size: root_config.tensor_parallel_size,
            enable_attention_dp: root_config.enable_attention_dp,
        };

        root_config.b10_generation_coordinator_config.validate()?;
        let unified_config = UnifiedConfig {
            generation_coordinator: root_config.b10_generation_coordinator_config,
            router_active_replicas: root_config.b10_routing_config.router_active_replicas,
            routing: root_config.b10_routing_config,
            runtime: runtime_config,
            engine_metrics_total_kv_blocks_override,
        };

        Ok(unified_config)
    }

    pub(crate) fn sanitize(mut self) -> Self {
        self.router_active_replicas = self.routing.router_active_replicas;
        self.routing.router_active_request_dp_blend =
            sanitize_router_active_request_dp_blend(self.routing.router_active_request_dp_blend);
        self.routing.router_residency_eviction_cost =
            sanitize_router_residency_eviction_cost(self.routing.router_residency_eviction_cost);
        self.routing.router_residency_half_life =
            sanitize_router_residency_half_life(self.routing.router_residency_half_life);
        self.routing.router_temperature =
            sanitize_router_temperature(self.routing.router_temperature);
        self.engine_metrics_total_kv_blocks_override =
            sanitize_engine_metrics_total_kv_blocks_override(
                self.engine_metrics_total_kv_blocks_override,
            );
        self
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_default_config() {
        let unified_config = UnifiedConfig::default();

        assert_eq!(unified_config.routing.router_temperature, 0.01);
        assert_eq!(unified_config.routing.router_overlap_score_weight, 3.5);
        assert_eq!(unified_config.routing.router_decode_block_weight, 1.0);
        assert_eq!(unified_config.routing.router_prefill_token_discount, 0.35);
        assert_eq!(unified_config.routing.router_decode_token_discount, 0.8);
        assert_eq!(
            unified_config
                .routing
                .router_session_affinity_score_multiplier,
            0.5
        );
        assert_eq!(
            unified_config.routing.router_active_request_dp_blend,
            2.0 / 3.0
        );
        assert_eq!(unified_config.router_active_replicas, 1);
    }

    #[test]
    fn test_override_args_applied() {
        use std::io::Write;

        // Create a temp config file
        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        let config_content = r#"
b10_routing_config:
  router_temperature: 0.15
  router_overlap_score_weight: 3.5
  router_active_request_dp_blend: 0.2
  router_session_affinity_score_multiplier: 0.4
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
      router_session_affinity_discount: 0.25
      router_queue_threshold: 0
      router_active_replicas: 3
"#;
        write!(temp_file, "{}", config_content).unwrap();
        let path = temp_file.path().to_path_buf();

        // Set env vars

        // Load config
        let config = UnifiedConfig::parse(
            &std::fs::read_to_string(&path).unwrap(),
            Some("test_group"),
            &B10RoutingConfig::default(),
        )
        .unwrap();

        // Check overrides applied
        assert_eq!(config.routing.router_temperature, 0.99);
        assert_eq!(config.routing.router_overlap_score_weight, 1.0);
        assert_eq!(config.routing.router_active_request_dp_blend, 0.75);
        assert_eq!(
            config.routing.router_session_affinity_score_multiplier,
            0.25
        );
        assert_eq!(config.routing.router_queue_threshold, Some(0.0));
        assert_eq!(config.router_active_replicas, 3);
        // Check runtime config and computed data_parallel_size
        assert_eq!(config.runtime.tensor_parallel_size, Some(8));
        assert_eq!(config.runtime.enable_attention_dp, Some(true));
        assert_eq!(config.engine_metrics_total_kv_blocks_override, Some(250000));
        assert_eq!(config.runtime.compute_data_parallel_size(), Some(8)); // Because enable_attention_dp=true returns tensor_parallel_size

        // Cleanup
    }

    #[test]
    fn test_prefill_block_weight_alias_populates_overlap_score_weight() {
        use std::io::Write;

        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        let config_content = r#"
b10_routing_config:
  router_prefill_block_weight: 5.5
"#;
        write!(temp_file, "{}", config_content).unwrap();
        let path = temp_file.path().to_path_buf();

        let config = UnifiedConfig::parse(
            &std::fs::read_to_string(&path).unwrap(),
            None,
            &B10RoutingConfig::default(),
        )
        .unwrap();

        // The alias populates the same field as router_overlap_score_weight.
        assert_eq!(config.routing.router_overlap_score_weight, 5.5);
    }

    #[test]
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

        let config = UnifiedConfig::parse(
            &std::fs::read_to_string(&path).unwrap(),
            None,
            &B10RoutingConfig::default(),
        )
        .unwrap();
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

        let config = UnifiedConfig::parse(
            &std::fs::read_to_string(&path).unwrap(),
            Some("test_group"),
            &B10RoutingConfig::default(),
        )
        .unwrap();
        assert_eq!(config.routing.router_decode_block_weight, 2.0);
    }

    #[test]
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

        let config = UnifiedConfig::parse(
            &std::fs::read_to_string(&path).unwrap(),
            Some("test_group"),
            &B10RoutingConfig::default(),
        )
        .unwrap();

        assert_eq!(config.routing.router_temperature, 0.99);
        assert_eq!(config.router_active_replicas, 4);
    }

    #[test]
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

        // Load config
        let config = UnifiedConfig::parse(
            &std::fs::read_to_string(&path).unwrap(),
            None,
            &B10RoutingConfig::default(),
        )
        .unwrap();

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

        let config = UnifiedConfig::parse(
            &std::fs::read_to_string(&path).unwrap(),
            None,
            &B10RoutingConfig::default(),
        )
        .unwrap();
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

        let config = UnifiedConfig::parse(
            &std::fs::read_to_string(&path).unwrap(),
            None,
            &B10RoutingConfig::default(),
        )
        .unwrap();

        assert_eq!(config.routing.router_temperature, 0.15);
        assert_eq!(config.router_active_replicas, 2);
    }

    #[test]
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

        let config = UnifiedConfig::parse(
            &std::fs::read_to_string(&path).unwrap(),
            Some("test_group"),
            &B10RoutingConfig::default(),
        )
        .unwrap();

        assert_eq!(config.router_active_replicas, 3);
    }

    #[test]
    fn test_engine_metrics_total_kv_blocks_override_sanitization() {
        use std::io::Write;

        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        let config_content = r#"
engine_metrics_total_kv_blocks_override: 0
"#;
        write!(temp_file, "{}", config_content).unwrap();
        let path = temp_file.path().to_path_buf();

        let config = UnifiedConfig::parse(
            &std::fs::read_to_string(&path).unwrap(),
            None,
            &B10RoutingConfig::default(),
        )
        .unwrap();

        assert_eq!(config.runtime.tensor_parallel_size, None);
        assert_eq!(config.engine_metrics_total_kv_blocks_override, None);
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

    #[test]
    fn test_router_temperature_sanitization() {
        use std::io::Write;

        // Zero / negative / non-finite all clamp to the floor.
        assert_eq!(sanitize_router_temperature(0.0), MIN_ROUTER_TEMPERATURE);
        assert_eq!(sanitize_router_temperature(-1.0), MIN_ROUTER_TEMPERATURE);
        assert_eq!(
            sanitize_router_temperature(f64::NAN),
            MIN_ROUTER_TEMPERATURE
        );
        assert_eq!(
            sanitize_router_temperature(f64::INFINITY),
            MIN_ROUTER_TEMPERATURE
        );

        // Positive finite values pass through unchanged.
        assert_eq!(sanitize_router_temperature(0.01), 0.01);
        assert_eq!(sanitize_router_temperature(1.0), 1.0);
        // The floor itself is allowed (it is > 0 and finite).
        assert_eq!(
            sanitize_router_temperature(MIN_ROUTER_TEMPERATURE),
            MIN_ROUTER_TEMPERATURE
        );

        // A config that explicitly sets temperature to 0 is clamped on load.
        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        let config_content = r#"
b10_routing_config:
  router_temperature: 0.0
"#;
        write!(temp_file, "{}", config_content).unwrap();
        let path = temp_file.path().to_path_buf();

        let config = UnifiedConfig::parse(
            &std::fs::read_to_string(&path).unwrap(),
            None,
            &B10RoutingConfig::default(),
        )
        .unwrap();
        assert_eq!(config.routing.router_temperature, MIN_ROUTER_TEMPERATURE);
    }
}
