// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ConfigMap schema and hot-reloadable loader for GWP.
//!
//! The schema mirrors the boundary GWP can control:
//!
//! `model route -> endpoint deployment`. The current topology producer joins
//! this configuration with planner-observed workers; alternate producers can
//! populate the same normalized topology boundary.
//!
//! Endpoints are defined once and model routes reference them. Workers never
//! appear in this schema; they are provider observations used to estimate
//! endpoint load, not independently addressable egress targets.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;
use serde::{Deserialize, Serialize};

/// Stable identifier for an independently routable ingress deployment.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EndpointId(pub String);

/// One routable Dynamo graph deployment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EndpointConfig {
    /// OpenAI-compatible ingress the data plane forwards to.
    pub ingress_url: url::Url,
    /// Bearer/API key used for ingress.
    #[serde(default)]
    pub api_key: String,
    /// This deployment's planner `/deep/health` URL.
    pub planner_url: url::Url,
    /// Planner bearer/API key; falls back to `api_key` when absent.
    #[serde(default)]
    pub planner_api_key: Option<String>,
    /// Values advertised by routing dimension, for example
    /// `region: [us]` or `compliance: [hipaa]`.
    #[serde(default)]
    pub properties: BTreeMap<String, BTreeSet<String>>,
}

impl EndpointConfig {
    pub fn planner_api_key(&self) -> &str {
        self.planner_api_key.as_deref().unwrap_or(&self.api_key)
    }
}

/// Models sharing the same candidate endpoint set.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelRoute {
    pub models: Vec<String>,
    pub endpoints: Vec<EndpointId>,
}

/// Hard and future soft constraints for one Alyx routing dimension.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DimensionRequirements {
    /// At least one of these values must be advertised by an endpoint.
    #[serde(default)]
    pub required: BTreeSet<String>,
    /// Reserved for preference-aware scoring; does not affect eligibility yet.
    #[serde(default)]
    pub preferred: BTreeSet<String>,
}

/// The value of `X-Baseten-Model-Apis-Routing-Requirements`.
///
/// Dimensions AND together. Values in a dimension's `required` list OR
/// together, so `region.required: [us, canada]` accepts either region.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RoutingRequirements(pub BTreeMap<String, DimensionRequirements>);

pub(crate) fn properties_satisfy_routing_requirements(
    properties: &BTreeMap<String, BTreeSet<String>>,
    requirements: &RoutingRequirements,
) -> bool {
    requirements.0.iter().all(|(dimension, constraint)| {
        constraint.required.is_empty()
            || properties
                .get(dimension)
                .is_some_and(|values| !values.is_disjoint(&constraint.required))
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionConfig {
    #[serde(default = "default_session_ttl_secs")]
    pub ttl_secs: u64,
    /// Derive a stable affinity key from four rolling prompt-prefix hashes
    /// when neither a recognized header nor OpenAI `user` is present.
    #[serde(default)]
    pub prompt_hash_fallback: Option<PromptHashFallbackConfig>,
    /// Preferred tagged affinity backend configuration.
    #[serde(default)]
    pub backend: Option<SessionBackendConfig>,
    /// Legacy etcd configuration. Kept so existing ConfigMaps remain valid.
    /// New configurations should use `backend`.
    #[serde(default)]
    pub etcd_endpoints: Option<Vec<String>>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            ttl_secs: default_session_ttl_secs(),
            prompt_hash_fallback: None,
            backend: None,
            etcd_endpoints: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromptHashFallbackConfig {
    /// Token position at which the stable prompt prefix is fingerprinted.
    /// Only complete routing blocks at or before this position participate.
    pub token_position: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionBackendConfig {
    Etcd {
        endpoints: Vec<String>,
    },
    Redis {
        url: String,
        #[serde(default = "default_redis_key_prefix")]
        key_prefix: String,
    },
}

fn default_redis_key_prefix() -> String {
    "gwp:affinity:".into()
}

impl SessionConfig {
    /// Resolve the tagged backend, or translate the legacy etcd field.
    pub fn resolved_backend(&self) -> anyhow::Result<SessionBackendConfig> {
        match (&self.backend, &self.etcd_endpoints) {
            (Some(_), Some(_)) => {
                anyhow::bail!("session.backend and session.etcd_endpoints are mutually exclusive")
            }
            (Some(backend), None) => Ok(backend.clone()),
            (None, Some(endpoints)) => Ok(SessionBackendConfig::Etcd {
                endpoints: endpoints.clone(),
            }),
            (None, None) => anyhow::bail!("session.backend must be configured"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutingConfig {
    /// Approximate-tokenization pseudo-token stride.
    #[serde(default = "default_pseudo_stride")]
    pub pseudo_stride: usize,
    /// Block size for endpoint-level approximate prefix hashes.
    #[serde(default = "default_block_size")]
    pub block_size: u32,
    /// Hold an endpoint's last-good planner snapshot for this many seconds.
    #[serde(default = "default_planner_staleness_grace_secs")]
    pub planner_staleness_grace_secs: u64,
    /// When peer replicas already exist, remain unready for this many seconds
    /// after subscribing to their live lifecycle event streams.
    #[serde(default = "default_replica_warmup_secs")]
    pub replica_warmup_secs: u64,
    /// Maximum time to admit lifecycle completion events while draining.
    #[serde(default = "default_shutdown_grace_secs")]
    pub shutdown_grace_secs: u64,
    /// Lifetime of entries in the approximate indexer (the local prune-TTL'd
    /// radix tree populated by `record_routing_decision`), in seconds.
    /// Routing decisions older than this are pruned. Mirrors the KV router's
    /// `router_ttl_secs`. Startup-only; not hot-reloadable.
    #[serde(default = "default_approx_indexer_ttl_secs")]
    pub approx_indexer_ttl_secs: u64,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            pseudo_stride: default_pseudo_stride(),
            block_size: default_block_size(),
            planner_staleness_grace_secs: default_planner_staleness_grace_secs(),
            replica_warmup_secs: default_replica_warmup_secs(),
            shutdown_grace_secs: default_shutdown_grace_secs(),
            approx_indexer_ttl_secs: default_approx_indexer_ttl_secs(),
        }
    }
}

/// Tokenization policy for one exact OpenAI model name.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ModelTokenizationConfig {
    /// Cheap deterministic byte packing. This is also the implicit default.
    Pseudo,
    /// Load `tokenizer.json`, `chat_template.jinja`, and
    /// `tokenizer_config.json` from one portable model bundle.
    Real { directory: PathBuf },
    /// Render `chat_template.jinja` (for chat-completions) but tokenize the
    /// rendered text with the cheap pseudo byte-chunk heuristic instead of a
    /// real tokenizer. Loads `chat_template.jinja` and
    /// `tokenizer_config.json`; `tokenizer.json` is not required. This keeps
    /// the chat template's effect on prefix identity without paying the real
    /// tokenizer's startup and per-request cost.
    TemplateOnly { directory: PathBuf },
}

/// Per-model tokenization settings. Models absent from this map use pseudo
/// tokenization; opting into real tokenization is always explicit.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TokenizationConfig {
    #[serde(default)]
    pub models: BTreeMap<String, ModelTokenizationConfig>,
}

/// Fully resolved request-stage policy for one canonical model.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelStagePolicy {
    /// Read and write session affinity bindings.
    pub affinity: bool,
    /// Expose approximate cache-locality inputs to selection and record
    /// successful routes. Generic scheduler active-prefix and load tracking
    /// remain enabled when this is false.
    pub trie: bool,
    /// Optional per-model selector tuning. Unset fields inherit the
    /// hot-reloaded process-wide B10 routing configuration.
    pub load_balancing: LoadBalancingPolicy,
}

impl Default for ModelStagePolicy {
    fn default() -> Self {
        Self {
            affinity: true,
            trie: true,
            load_balancing: LoadBalancingPolicy::default(),
        }
    }
}

/// GWP-specific per-model load-source and scoring overrides.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoadBalancingPolicy {
    /// Weight for the delayed, cache-blind planner baseline.
    pub planner_weight: Option<f64>,
    /// Weight for current cache-aware local scheduler load, including the
    /// candidate request's trie-adjusted prefill cost. With both source
    /// weights at 1, this is applied as a signed delta from the refresh anchor.
    pub local_weight: Option<f64>,
    pub prefill_weight: Option<f64>,
    pub decode_weight: Option<f64>,
    pub active_request_weight: Option<f64>,
    pub cache_miss_weight: Option<f64>,
    pub temperature: Option<f64>,
}

impl LoadBalancingPolicy {
    fn merge(self, overrides: Self) -> Self {
        Self {
            planner_weight: overrides.planner_weight.or(self.planner_weight),
            local_weight: overrides.local_weight.or(self.local_weight),
            prefill_weight: overrides.prefill_weight.or(self.prefill_weight),
            decode_weight: overrides.decode_weight.or(self.decode_weight),
            active_request_weight: overrides
                .active_request_weight
                .or(self.active_request_weight),
            cache_miss_weight: overrides.cache_miss_weight.or(self.cache_miss_weight),
            temperature: overrides.temperature.or(self.temperature),
        }
    }
}

/// Partial per-model override merged on top of `model_policies.default`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelStagePolicyOverride {
    pub affinity: Option<bool>,
    pub trie: Option<bool>,
    pub load_balancing: LoadBalancingPolicy,
}

impl ModelStagePolicyOverride {
    fn apply(self, mut policy: ModelStagePolicy) -> ModelStagePolicy {
        if let Some(enabled) = self.affinity {
            policy.affinity = enabled;
        }
        if let Some(enabled) = self.trie {
            policy.trie = enabled;
        }
        policy.load_balancing = policy.load_balancing.merge(self.load_balancing);
        policy
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelPoliciesConfig {
    pub default: ModelStagePolicy,
    /// Canonical model name to a partial policy override.
    pub models: BTreeMap<String, ModelStagePolicyOverride>,
}

/// Root of the GWP ConfigMap.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GwpConfig {
    /// Named, independently routable deployment endpoints.
    pub endpoints: BTreeMap<EndpointId, EndpointConfig>,
    /// Authoritative external model-to-endpoint policy.
    #[serde(default)]
    pub routes: Vec<ModelRoute>,
    /// Public, legacy, or deployment-advertised model name -> canonical model.
    /// Aliases may remain in `routes`; route construction collapses them.
    #[serde(default)]
    pub served_alias_model_map: BTreeMap<String, String>,
    #[serde(default)]
    pub session: SessionConfig,
    #[serde(default)]
    pub routing: RoutingConfig,
    #[serde(default)]
    pub tokenization: TokenizationConfig,
    /// Hot-reloadable stage switches resolved by canonical model name.
    #[serde(default)]
    pub model_policies: ModelPoliciesConfig,
}

fn default_pseudo_stride() -> usize {
    4
}
fn default_session_ttl_secs() -> u64 {
    600
}
fn default_block_size() -> u32 {
    32
}
fn default_planner_staleness_grace_secs() -> u64 {
    30
}
fn default_replica_warmup_secs() -> u64 {
    90
}
fn default_shutdown_grace_secs() -> u64 {
    30
}
fn default_approx_indexer_ttl_secs() -> u64 {
    120
}

impl GwpConfig {
    pub fn config_path() -> PathBuf {
        std::env::var("DYN_GWP_CONFIG_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/configs/gwp.yaml"))
    }

    pub fn load() -> anyhow::Result<Self> {
        let path = Self::config_path();
        let raw = std::fs::read_to_string(&path)
            .map_err(|error| anyhow::anyhow!("reading GWP config {}: {error}", path.display()))?;
        Self::load_from_str(&raw)
    }

    pub fn load_from_str(raw: &str) -> anyhow::Result<Self> {
        let config: Self = serde_yaml::from_str(raw)?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.endpoints.is_empty(),
            "GWP config must define at least one endpoint"
        );
        anyhow::ensure!(
            self.routing.pseudo_stride > 0,
            "routing.pseudo_stride must be positive"
        );
        anyhow::ensure!(
            self.routing.block_size > 0,
            "routing.block_size must be positive"
        );
        if let Some(prompt_hash) = &self.session.prompt_hash_fallback {
            let block_size = self.routing.block_size as usize;
            anyhow::ensure!(
                prompt_hash.token_position / block_size >= 4,
                "session.prompt_hash_fallback.token_position must cover at least four routing blocks"
            );
        }
        anyhow::ensure!(
            self.routing.planner_staleness_grace_secs > 0,
            "routing.planner_staleness_grace_secs must be positive"
        );
        anyhow::ensure!(
            self.routing.shutdown_grace_secs > 0,
            "routing.shutdown_grace_secs must be positive"
        );
        anyhow::ensure!(
            self.routing.approx_indexer_ttl_secs > 0,
            "routing.approx_indexer_ttl_secs must be positive"
        );
        match self.session.resolved_backend()? {
            SessionBackendConfig::Etcd { endpoints } => anyhow::ensure!(
                !endpoints.is_empty() && endpoints.iter().all(|e| !e.trim().is_empty()),
                "session.backend etcd endpoints must contain at least one non-empty endpoint"
            ),
            SessionBackendConfig::Redis { url, key_prefix } => {
                anyhow::ensure!(!url.trim().is_empty(), "session.backend redis url is empty");
                anyhow::ensure!(
                    !key_prefix.is_empty(),
                    "session.backend redis key_prefix is empty"
                );
            }
        }
        // A planner response is the source of truth for worker ownership. If
        // two routable endpoints poll the same planner, every worker appears
        // to belong to both ingresses and the reflector cannot safely route
        // requests. Reject this before the configuration reaches the live
        // reflector (including on hot reload).
        let mut planner_urls: HashMap<&url::Url, &EndpointId> = HashMap::new();
        for (endpoint_id, endpoint) in &self.endpoints {
            anyhow::ensure!(
                endpoint.ingress_url.scheme() == "http",
                "endpoint {} ingress_url must use http for the Envoy dynamic-forward-proxy data plane",
                endpoint_id.0
            );
            anyhow::ensure!(
                endpoint.ingress_url.host_str().is_some(),
                "endpoint {} ingress_url must include a host",
                endpoint_id.0
            );
            anyhow::ensure!(
                endpoint.ingress_url.path() == "/v1",
                "endpoint {} ingress_url path must be /v1",
                endpoint_id.0
            );
            anyhow::ensure!(
                endpoint.ingress_url.query().is_none() && endpoint.ingress_url.fragment().is_none(),
                "endpoint {} ingress_url must not include a query or fragment",
                endpoint_id.0
            );
            anyhow::ensure!(
                endpoint.api_key.is_ascii()
                    && !endpoint.api_key.contains('\r')
                    && !endpoint.api_key.contains('\n'),
                "endpoint {} api_key contains invalid header characters",
                endpoint_id.0
            );
            if let Some(previous_endpoint) = planner_urls.insert(&endpoint.planner_url, endpoint_id)
            {
                anyhow::bail!(
                    "endpoints {} and {} share planner_url {}; each endpoint must use a distinct planner URL",
                    previous_endpoint.0,
                    endpoint_id.0,
                    endpoint.planner_url
                );
            }
        }
        for route in &self.routes {
            anyhow::ensure!(!route.models.is_empty(), "route models must not be empty");
            anyhow::ensure!(
                !route.endpoints.is_empty(),
                "route endpoints must not be empty"
            );
            for endpoint in &route.endpoints {
                anyhow::ensure!(
                    self.endpoints.contains_key(endpoint),
                    "route references unknown endpoint {}",
                    endpoint.0
                );
            }
        }
        for (alias, canonical) in &self.served_alias_model_map {
            anyhow::ensure!(
                !alias.trim().is_empty(),
                "served model alias must not be empty"
            );
            anyhow::ensure!(
                !canonical.trim().is_empty(),
                "canonical served model for alias {alias} must not be empty"
            );
            anyhow::ensure!(
                alias != canonical,
                "served model alias {alias} must differ from its canonical model"
            );
            anyhow::ensure!(
                !self.served_alias_model_map.contains_key(canonical),
                "served model alias chains are not allowed: {alias} -> {canonical}"
            );
        }
        let routed_models: HashSet<&str> = self
            .routes
            .iter()
            .flat_map(|route| route.models.iter())
            .map(|model| self.canonical_model(model))
            .collect();
        for (alias, canonical) in &self.served_alias_model_map {
            anyhow::ensure!(
                routed_models.contains(canonical.as_str()),
                "served model alias {alias} references unrouted canonical model {canonical}"
            );
        }
        for (endpoint_id, endpoint) in &self.endpoints {
            anyhow::ensure!(
                endpoint
                    .properties
                    .iter()
                    .all(|(dimension, values)| !dimension.trim().is_empty()
                        && !values.is_empty()
                        && values.iter().all(|value| !value.trim().is_empty())),
                "endpoint {} properties must contain non-empty dimensions and values",
                endpoint_id.0
            );
        }
        for (model, tokenization) in &self.tokenization.models {
            anyhow::ensure!(
                !model.trim().is_empty(),
                "tokenization.models keys must not be empty"
            );
            anyhow::ensure!(
                !self.served_alias_model_map.contains_key(model),
                "tokenization.models must use canonical model names; {model} is an alias"
            );
            if let ModelTokenizationConfig::Real { directory }
            | ModelTokenizationConfig::TemplateOnly { directory } = tokenization
            {
                anyhow::ensure!(
                    !directory.as_os_str().is_empty(),
                    "tokenization.models.{model}.directory must not be empty"
                );
            }
        }
        for model in self.model_policies.models.keys() {
            anyhow::ensure!(
                !model.trim().is_empty(),
                "model_policies.models keys must not be empty"
            );
            anyhow::ensure!(
                !self.served_alias_model_map.contains_key(model),
                "model_policies.models must use canonical model names; {model} is an alias"
            );
            anyhow::ensure!(
                routed_models.contains(model.as_str()),
                "model_policies.models references unrouted canonical model {model}"
            );
        }
        for (scope, policy) in std::iter::once((
            "model_policies.default".to_string(),
            self.model_policies.default.load_balancing,
        ))
        .chain(self.model_policies.models.iter().map(|(model, policy)| {
            (
                format!("model_policies.models.{model}"),
                policy.load_balancing,
            )
        })) {
            for (name, value) in [
                ("planner_weight", policy.planner_weight),
                ("local_weight", policy.local_weight),
                ("prefill_weight", policy.prefill_weight),
                ("decode_weight", policy.decode_weight),
                ("active_request_weight", policy.active_request_weight),
                ("cache_miss_weight", policy.cache_miss_weight),
                ("temperature", policy.temperature),
            ] {
                if let Some(value) = value {
                    anyhow::ensure!(
                        value.is_finite() && value >= 0.0,
                        "{scope}.load_balancing.{name} must be finite and non-negative"
                    );
                }
            }
        }
        Ok(())
    }

    /// Configured model names keyed by endpoint.
    pub fn models_by_endpoint(&self) -> HashMap<EndpointId, HashSet<String>> {
        let mut models_by_endpoint: HashMap<EndpointId, HashSet<String>> = self
            .endpoints
            .keys()
            .cloned()
            .map(|endpoint| (endpoint, HashSet::new()))
            .collect();
        for route in &self.routes {
            for endpoint in &route.endpoints {
                models_by_endpoint
                    .entry(endpoint.clone())
                    .or_default()
                    .extend(
                        route
                            .models
                            .iter()
                            .map(|model| self.canonical_model(model).to_string()),
                    );
            }
        }
        models_by_endpoint
    }

    /// Resolve a public/legacy request name to the one canonical served model
    /// used for routing, tokenization, affinity, and metrics.
    pub fn canonical_model<'a>(&'a self, requested: &'a str) -> &'a str {
        self.served_alias_model_map
            .get(requested)
            .map(String::as_str)
            .unwrap_or(requested)
    }

    /// Resolve hot-reloadable stage switches for a requested name, alias, or
    /// canonical name. Canonicalizing an already-canonical name is intentional
    /// and keeps callers from needing separate APIs.
    pub fn model_policy(&self, requested_model: &str) -> ModelStagePolicy {
        let canonical = self.canonical_model(requested_model);
        self.model_policies
            .models
            .get(canonical)
            .copied()
            .unwrap_or_default()
            .apply(self.model_policies.default)
    }

    /// Canonical model names plus configured public aliases for model
    /// discovery. Aliases are included only when their canonical model is in
    /// the supplied routable set.
    pub fn advertised_models(&self, canonical_models: &HashSet<String>) -> HashSet<String> {
        let mut advertised = canonical_models.clone();
        advertised.extend(
            self.served_alias_model_map
                .iter()
                .filter(|(_, canonical)| canonical_models.contains(canonical.as_str()))
                .map(|(alias, _)| alias.clone()),
        );
        advertised
    }

    /// Explicit candidates for `model`, or `None` when the model is not
    /// configured and must not be routed.
    pub fn configured_candidates(&self, model: &str) -> Option<HashSet<EndpointId>> {
        let model = self.canonical_model(model);
        let mut matched = false;
        let mut candidates = HashSet::new();
        for route in &self.routes {
            if route
                .models
                .iter()
                .any(|configured| self.canonical_model(configured) == model)
            {
                matched = true;
                candidates.extend(route.endpoints.iter().cloned());
            }
        }
        matched.then_some(candidates)
    }

    /// Endpoints satisfying every hard routing dimension. Within a dimension,
    /// any advertised value may satisfy any required value.
    pub fn routing_candidates(&self, requirements: &RoutingRequirements) -> HashSet<EndpointId> {
        self.endpoints
            .iter()
            .filter(|(_, endpoint)| {
                properties_satisfy_routing_requirements(&endpoint.properties, requirements)
            })
            .map(|(endpoint_id, _)| endpoint_id.clone())
            .collect()
    }
}

/// Atomically replaceable configuration shared by request and reflector paths.
#[derive(Clone)]
pub struct ConfigStore {
    inner: Arc<ArcSwap<GwpConfig>>,
}

impl ConfigStore {
    pub fn new(config: GwpConfig) -> Self {
        Self {
            inner: Arc::new(ArcSwap::from_pointee(config)),
        }
    }

    pub fn load(&self) -> Arc<GwpConfig> {
        self.inner.load_full()
    }

    pub fn replace(&self, config: GwpConfig) {
        self.inner.store(Arc::new(config));
    }

    /// Poll a ConfigMap-mounted YAML file and atomically apply valid changes.
    /// Read/parse failures retain the last-known-good configuration.
    pub fn spawn_file_reloader(
        &self,
        path: PathBuf,
        interval: Duration,
    ) -> tokio::task::JoinHandle<()> {
        let store = self.clone();
        tokio::spawn(async move {
            let mut last_contents = std::fs::read_to_string(&path).ok();
            let mut ticker = tokio::time::interval(interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                ticker.tick().await;
                let contents = match std::fs::read_to_string(&path) {
                    Ok(contents) => contents,
                    Err(error) => {
                        tracing::warn!(
                            path = %path.display(),
                            %error,
                            "GWP config reload failed; retaining last-known-good config"
                        );
                        continue;
                    }
                };
                if last_contents.as_deref() == Some(contents.as_str()) {
                    continue;
                }
                match GwpConfig::load_from_str(&contents) {
                    Ok(mut next) => {
                        let previous = store.load();
                        if next.routing.block_size != previous.routing.block_size {
                            tracing::warn!(
                                old = previous.routing.block_size,
                                new = next.routing.block_size,
                                "routing.block_size is startup-only; retaining current value"
                            );
                            next.routing.block_size = previous.routing.block_size;
                        }
                        if next.routing.replica_warmup_secs != previous.routing.replica_warmup_secs
                        {
                            tracing::warn!(
                                old = previous.routing.replica_warmup_secs,
                                new = next.routing.replica_warmup_secs,
                                "routing.replica_warmup_secs is startup-only; retaining current value"
                            );
                            next.routing.replica_warmup_secs = previous.routing.replica_warmup_secs;
                        }
                        if next.routing.shutdown_grace_secs != previous.routing.shutdown_grace_secs
                        {
                            tracing::warn!(
                                old = previous.routing.shutdown_grace_secs,
                                new = next.routing.shutdown_grace_secs,
                                "routing.shutdown_grace_secs is startup-only; retaining current value"
                            );
                            next.routing.shutdown_grace_secs = previous.routing.shutdown_grace_secs;
                        }
                        if next.routing.approx_indexer_ttl_secs
                            != previous.routing.approx_indexer_ttl_secs
                        {
                            tracing::warn!(
                                old = previous.routing.approx_indexer_ttl_secs,
                                new = next.routing.approx_indexer_ttl_secs,
                                "routing.approx_indexer_ttl_secs is startup-only; retaining current value"
                            );
                            next.routing.approx_indexer_ttl_secs =
                                previous.routing.approx_indexer_ttl_secs;
                        }
                        if next.tokenization != previous.tokenization {
                            tracing::warn!(
                                "tokenization is startup-only; retaining current model tokenizers"
                            );
                            next.tokenization = previous.tokenization.clone();
                        }
                        if next.session.resolved_backend().ok()
                            != previous.session.resolved_backend().ok()
                        {
                            tracing::warn!(
                                "session.backend is startup-only; retaining current affinity backend"
                            );
                            next.session.backend = previous.session.backend.clone();
                            next.session.etcd_endpoints = previous.session.etcd_endpoints.clone();
                        }
                        store.replace(next);
                        last_contents = Some(contents);
                        tracing::info!(path = %path.display(), "reloaded GWP config");
                    }
                    Err(error) => {
                        tracing::warn!(
                            path = %path.display(),
                            %error,
                            "invalid GWP config update; retaining last-known-good config"
                        );
                    }
                }
            }
        })
    }

    pub fn from_path(path: &Path) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|error| anyhow::anyhow!("reading GWP config {}: {error}", path.display()))?;
        Ok(Self::new(GwpConfig::load_from_str(&raw)?))
    }
}

impl From<GwpConfig> for ConfigStore {
    fn from(config: GwpConfig) -> Self {
        Self::new(config)
    }
}

impl From<Arc<GwpConfig>> for ConfigStore {
    fn from(config: Arc<GwpConfig>) -> Self {
        Self::new((*config).clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
endpoints:
  us-east:
    ingress_url: "http://us-east.example.com/v1"
    api_key: "cluster-key-1"
    planner_url: "http://planner.us-east.example.com/deep/health"
    planner_api_key: "planner-key-1"
    properties:
      region: [us]
      compliance: [hipaa]
  eu-west:
    ingress_url: "http://eu-west.example.com/v1"
    api_key: "cluster-key-2"
    planner_url: "http://planner.eu-west.example.com/deep/health"
    properties:
      region: [eu]
routes:
  - models: ["deepseek-v3", "llama-3-70b"]
    endpoints: ["us-east", "eu-west"]
served_alias_model_map:
  deepseek-v3-preview: deepseek-v3
session:
  ttl_secs: 300
  prompt_hash_fallback:
    token_position: 100000
  etcd_endpoints: ["http://127.0.0.1:2379"]
routing:
  planner_staleness_grace_secs: 45
tokenization:
  models:
    deepseek-v3:
      mode: real
      directory: /opt/dynamo-gwp/tokenizers/deepseek-v3
    llama-3-70b:
      mode: pseudo
model_policies:
  default:
    affinity: true
    trie: false
    load_balancing:
      planner_weight: 0.75
  models:
    deepseek-v3:
      trie: true
      load_balancing:
        local_weight: 1.25
    llama-3-70b:
      affinity: false
"#;

    #[test]
    fn parses_normalized_schema_and_derives_seeds() {
        let config = GwpConfig::load_from_str(SAMPLE).expect("parse");
        assert_eq!(config.endpoints.len(), 2);
        assert_eq!(
            config.endpoints[&EndpointId("us-east".into())].planner_api_key(),
            "planner-key-1"
        );
        assert_eq!(
            config.endpoints[&EndpointId("eu-west".into())].planner_api_key(),
            "cluster-key-2"
        );
        assert!(matches!(
            config.tokenization.models["deepseek-v3"],
            ModelTokenizationConfig::Real { .. }
        ));
        assert!(matches!(
            config.tokenization.models["llama-3-70b"],
            ModelTokenizationConfig::Pseudo
        ));
        assert_eq!(config.session.ttl_secs, 300);
        assert_eq!(config.canonical_model("deepseek-v3-preview"), "deepseek-v3");
        assert_eq!(config.canonical_model("llama-3-70b"), "llama-3-70b");
        assert_eq!(
            config
                .session
                .prompt_hash_fallback
                .as_ref()
                .unwrap()
                .token_position,
            100_000
        );
        assert_eq!(config.routing.pseudo_stride, 4);
        assert_eq!(config.routing.block_size, 32);
        assert_eq!(config.routing.planner_staleness_grace_secs, 45);
        assert_eq!(config.routing.replica_warmup_secs, 90);
        assert_eq!(config.routing.shutdown_grace_secs, 30);
        assert_eq!(config.routing.approx_indexer_ttl_secs, 120);
        assert_eq!(
            config.model_policy("deepseek-v3-preview"),
            ModelStagePolicy {
                affinity: true,
                trie: true,
                load_balancing: LoadBalancingPolicy {
                    planner_weight: Some(0.75),
                    local_weight: Some(1.25),
                    ..Default::default()
                },
            }
        );
        assert_eq!(
            config.model_policy("llama-3-70b"),
            ModelStagePolicy {
                affinity: false,
                trie: false,
                load_balancing: LoadBalancingPolicy {
                    planner_weight: Some(0.75),
                    ..Default::default()
                },
            }
        );
        assert_eq!(
            config
                .configured_candidates("deepseek-v3")
                .expect("configured")
                .len(),
            2
        );
        assert_eq!(
            config
                .configured_candidates("deepseek-v3-preview")
                .expect("alias configured")
                .len(),
            2
        );
        assert_eq!(
            config.advertised_models(&HashSet::from(["deepseek-v3".into()])),
            HashSet::from(["deepseek-v3".into(), "deepseek-v3-preview".into()])
        );
        assert_eq!(
            config.models_by_endpoint()[&EndpointId("us-east".into())].len(),
            2
        );
        assert_eq!(
            config.routing_candidates(&RoutingRequirements(BTreeMap::from([
                (
                    "region".into(),
                    DimensionRequirements {
                        required: BTreeSet::from(["us".into(), "canada".into()]),
                        preferred: BTreeSet::new(),
                    },
                ),
                (
                    "compliance".into(),
                    DimensionRequirements {
                        required: BTreeSet::from(["hipaa".into()]),
                        preferred: BTreeSet::new(),
                    },
                ),
            ]))),
            HashSet::from([EndpointId("us-east".into())])
        );
        assert_eq!(
            config
                .routing_candidates(&RoutingRequirements::default())
                .len(),
            2
        );
        assert!(
            config
                .routing_candidates(&RoutingRequirements(BTreeMap::from([(
                    "region".into(),
                    DimensionRequirements {
                        required: BTreeSet::from(["not-available".into()]),
                        preferred: BTreeSet::new(),
                    },
                )])))
                .is_empty(),
            "unknown required values match no endpoint"
        );
    }

    #[test]
    fn validates_route_references() {
        let invalid = r#"
endpoints:
  a:
    ingress_url: http://a/v1
    planner_url: http://a/deep/health
routes:
  - models: [m]
    endpoints: [missing]
session:
  etcd_endpoints: ["http://127.0.0.1:2379"]
"#;
        assert!(GwpConfig::load_from_str(invalid).is_err());
    }

    #[test]
    fn rejects_duplicate_planner_urls() {
        let invalid = SAMPLE.replace(
            "http://planner.eu-west.example.com/deep/health",
            "http://planner.us-east.example.com/deep/health",
        );
        let error = GwpConfig::load_from_str(&invalid).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("us-east"));
        assert!(message.contains("eu-west"));
        assert!(message.contains("share planner_url"));
    }

    #[test]
    fn rejects_empty_endpoint_map() {
        assert!(GwpConfig::load_from_str("endpoints: {}").is_err());
    }

    #[test]
    fn requires_shared_session_affinity() {
        let without_backend = SAMPLE.replace("  etcd_endpoints: [\"http://127.0.0.1:2379\"]\n", "");
        let error = GwpConfig::load_from_str(&without_backend).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("session.backend must be configured")
        );

        let empty_backend = SAMPLE.replace("[\"http://127.0.0.1:2379\"]", "[]");
        let error = GwpConfig::load_from_str(&empty_backend).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("must contain at least one non-empty endpoint")
        );
    }

    #[test]
    fn parses_tagged_redis_backend_and_default_prefix() {
        let redis = SAMPLE.replace(
            "  etcd_endpoints: [\"http://127.0.0.1:2379\"]",
            "  backend:\n    type: redis\n    url: redis://redis.example.com:6379/",
        );
        let config = GwpConfig::load_from_str(&redis).unwrap();
        assert_eq!(
            config.session.resolved_backend().unwrap(),
            SessionBackendConfig::Redis {
                url: "redis://redis.example.com:6379/".into(),
                key_prefix: "gwp:affinity:".into(),
            }
        );
    }

    #[test]
    fn parses_tagged_etcd_backend() {
        let tagged = SAMPLE.replace(
            "  etcd_endpoints: [\"http://127.0.0.1:2379\"]",
            "  backend:\n    type: etcd\n    endpoints: [\"http://127.0.0.1:2379\"]",
        );
        let config = GwpConfig::load_from_str(&tagged).unwrap();
        assert_eq!(
            config.session.resolved_backend().unwrap(),
            SessionBackendConfig::Etcd {
                endpoints: vec!["http://127.0.0.1:2379".into()],
            }
        );
    }

    #[test]
    fn rejects_ambiguous_or_invalid_session_backends() {
        let both = SAMPLE.replace(
            "session:\n",
            "session:\n  backend:\n    type: redis\n    url: redis://redis:6379/\n",
        );
        let error = GwpConfig::load_from_str(&both).unwrap_err();
        assert!(error.to_string().contains("mutually exclusive"));

        let empty_prefix = SAMPLE.replace(
            "  etcd_endpoints: [\"http://127.0.0.1:2379\"]",
            "  backend:\n    type: redis\n    url: redis://redis:6379/\n    key_prefix: \"\"",
        );
        let error = GwpConfig::load_from_str(&empty_prefix).unwrap_err();
        assert!(error.to_string().contains("redis key_prefix is empty"));
    }

    #[test]
    fn rejects_prompt_hash_cutoff_shorter_than_four_blocks() {
        let invalid = SAMPLE.replace("token_position: 100000", "token_position: 64");
        let error = GwpConfig::load_from_str(&invalid).unwrap_err();
        assert!(error.to_string().contains("at least four routing blocks"));
    }

    #[test]
    fn validates_envoy_ingress_contract() {
        let https = SAMPLE.replacen(
            "ingress_url: \"http://us-east.example.com/v1\"",
            "ingress_url: \"https://us-east.example.com/v1\"",
            1,
        );
        assert!(
            GwpConfig::load_from_str(&https)
                .unwrap_err()
                .to_string()
                .contains("must use http")
        );

        let missing_v1 = SAMPLE.replacen(
            "ingress_url: \"http://us-east.example.com/v1\"",
            "ingress_url: \"http://us-east.example.com/inference\"",
            1,
        );
        assert!(
            GwpConfig::load_from_str(&missing_v1)
                .unwrap_err()
                .to_string()
                .contains("path must be /v1")
        );

        let prefixed_v1 = SAMPLE.replacen(
            "ingress_url: \"http://us-east.example.com/v1\"",
            "ingress_url: \"http://us-east.example.com/internal/v1\"",
            1,
        );
        assert!(
            GwpConfig::load_from_str(&prefixed_v1)
                .unwrap_err()
                .to_string()
                .contains("path must be /v1")
        );

        let query = SAMPLE.replacen(
            "ingress_url: \"http://us-east.example.com/v1\"",
            "ingress_url: \"http://us-east.example.com/v1?debug=true\"",
            1,
        );
        assert!(
            GwpConfig::load_from_str(&query)
                .unwrap_err()
                .to_string()
                .contains("must not include a query or fragment")
        );

        let injected_key = SAMPLE.replacen(
            "api_key: \"cluster-key-1\"",
            "api_key: \"cluster-key-1\\r\\nx-injected: yes\"",
            1,
        );
        assert!(
            GwpConfig::load_from_str(&injected_key)
                .unwrap_err()
                .to_string()
                .contains("invalid header characters")
        );
    }

    #[test]
    fn validates_served_model_alias_groups_and_rejects_invalid_maps() {
        let unknown = SAMPLE.replace(
            "deepseek-v3-preview: deepseek-v3",
            "deepseek-v3-preview: missing-model",
        );
        let error = GwpConfig::load_from_str(&unknown).unwrap_err();
        assert!(error.to_string().contains("unrouted canonical model"));

        let routed_alias = SAMPLE.replace(
            "[\"deepseek-v3\", \"llama-3-70b\"]",
            "[\"deepseek-v3\", \"llama-3-70b\", \"deepseek-v3-preview\"]",
        );
        let routed_alias = GwpConfig::load_from_str(&routed_alias).unwrap();
        assert_eq!(
            routed_alias.models_by_endpoint()[&EndpointId("us-east".into())],
            HashSet::from(["deepseek-v3".into(), "llama-3-70b".into()])
        );

        let chained = SAMPLE.replace(
            "deepseek-v3-preview: deepseek-v3",
            "deepseek-v3-preview: old-deepseek\n  old-deepseek: deepseek-v3",
        );
        let error = GwpConfig::load_from_str(&chained).unwrap_err();
        assert!(error.to_string().contains("alias chains are not allowed"));

        let alias_tokenizer = SAMPLE.replace(
            "    deepseek-v3:\n      mode: real",
            "    deepseek-v3-preview:\n      mode: real",
        );
        let error = GwpConfig::load_from_str(&alias_tokenizer).unwrap_err();
        assert!(error.to_string().contains("must use canonical model names"));

        let alias_policy = SAMPLE.replace(
            "    deepseek-v3:\n      trie: true",
            "    deepseek-v3-preview:\n      trie: true",
        );
        let error = GwpConfig::load_from_str(&alias_policy).unwrap_err();
        assert!(error.to_string().contains("must use canonical model names"));

        let unrouted_policy = SAMPLE.replace(
            "    deepseek-v3:\n      trie: true",
            "    not-routed:\n      trie: true",
        );
        let error = GwpConfig::load_from_str(&unrouted_policy).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("references unrouted canonical model not-routed")
        );
    }

    #[test]
    fn alias_routes_merge_into_one_canonical_load_balancing_group() {
        let config = GwpConfig::load_from_str(
            r#"
endpoints:
  fast:
    ingress_url: http://fast/v1
    planner_url: http://fast/deep/health
  fast-alt:
    ingress_url: http://fast-alt/v1
    planner_url: http://fast-alt/deep/health
routes:
  - models: [composer-2-5-fast]
    endpoints: [fast]
  - models: [composer-2-5]
    endpoints: [fast-alt]
served_alias_model_map:
  composer-2-5-fast: composer-2-5
session:
  etcd_endpoints: [http://127.0.0.1:2379]
"#,
        )
        .unwrap();

        let expected = HashSet::from([EndpointId("fast".into()), EndpointId("fast-alt".into())]);
        assert_eq!(
            config.configured_candidates("composer-2-5"),
            Some(expected.clone())
        );
        assert_eq!(
            config.configured_candidates("composer-2-5-fast"),
            Some(expected)
        );
        assert_eq!(
            config.models_by_endpoint()[&EndpointId("fast".into())],
            HashSet::from(["composer-2-5".into()])
        );
    }

    #[test]
    fn rejects_empty_endpoint_property() {
        let invalid = SAMPLE.replace("region: [us]", "region: [us, \"\"]");
        let error = GwpConfig::load_from_str(&invalid).unwrap_err();
        assert!(error.to_string().contains("properties must contain"));
    }
}
