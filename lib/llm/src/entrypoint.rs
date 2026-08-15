// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The entrypoint module provides tools to build a Dynamo runner.
//! - Create an EngineConfig of the engine (potentially auto-discovered) to execute
//! - Connect it to an Input

pub mod input;
pub use input::{PreprocessedRouting, build_preprocessed_routing};

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use dynamo_kv_router::{PrefillLoadEstimator, config::KvRouterConfig, selector::WorkerSelector};
use dynamo_runtime::{discovery::ModelCardInstanceId, pipeline::RouterMode};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    backend::ExecutionContext,
    discovery::LoadThresholdConfig,
    engines::StreamingEngine,
    local_model::{LocalModel, runtime_config::ModelRuntimeConfig},
    model_card::ModelDeploymentCard,
    types::openai::chat_completions::OpenAIChatCompletionsStreamingEngine,
};

/// Callback type for chat engine factory (async)
pub type PrefillRoutedEngine = dynamo_runtime::pipeline::ServiceEngine<
    dynamo_runtime::pipeline::SingleIn<crate::protocols::common::preprocessor::PreprocessedRequest>,
    dynamo_runtime::pipeline::ManyOut<
        crate::types::Annotated<crate::protocols::common::llm_backend::LLMEngineOutput>,
    >,
>;

pub type ChatEngineFactoryCallback = Arc<
    dyn Fn(
            ModelCardInstanceId,
            ModelDeploymentCard,
            PrefillRoutedEngine,
        ) -> Pin<
            Box<dyn Future<Output = anyhow::Result<OpenAIChatCompletionsStreamingEngine>> + Send>,
        > + Send
        + Sync,
>;

pub type CustomWorkerSelector = Arc<dyn WorkerSelector<ModelRuntimeConfig> + Send + Sync + 'static>;

#[derive(Clone, Default)]
pub enum RouterSelector {
    #[default]
    Default,
    B10,
    Custom(CustomWorkerSelector),
}

impl std::fmt::Debug for RouterSelector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Default => f.write_str("Default"),
            Self::B10 => f.write_str("B10"),
            Self::Custom(_) => f.write_str("Custom"),
        }
    }
}

impl Serialize for RouterSelector {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Default => serializer.serialize_str("Default"),
            Self::B10 => serializer.serialize_str("B10"),
            Self::Custom(_) => serializer.serialize_str("Custom"),
        }
    }
}

impl<'de> Deserialize<'de> for RouterSelector {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        match String::deserialize(deserializer)?.as_str() {
            "Default" => Ok(Self::Default),
            "B10" => Ok(Self::B10),
            other => Err(serde::de::Error::custom(format!(
                "unsupported router selector '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RouterConfig {
    pub router_mode: RouterMode,
    pub kv_router_config: KvRouterConfig,
    pub router_selector: RouterSelector,
    /// Load threshold configuration for overload detection
    pub load_threshold_config: LoadThresholdConfig,
    pub enforce_disagg: bool,
    #[serde(default)]
    pub session_affinity_ttl_secs: Option<u64>,
}

impl RouterConfig {
    pub fn new(router_mode: RouterMode, kv_router_config: KvRouterConfig) -> Self {
        Self {
            router_mode,
            kv_router_config,
            router_selector: RouterSelector::Default,
            load_threshold_config: LoadThresholdConfig::default(),
            enforce_disagg: false,
            session_affinity_ttl_secs: None,
        }
    }

    pub fn with_load_threshold_config(mut self, config: LoadThresholdConfig) -> Self {
        self.load_threshold_config = config;
        self
    }

    pub fn with_enforce_disagg(mut self, enforce_disagg: bool) -> Self {
        self.enforce_disagg = enforce_disagg;
        self
    }

    pub fn with_router_selector(mut self, selector: RouterSelector) -> Self {
        self.router_selector = selector;
        self
    }

    pub fn with_session_affinity_ttl_secs(mut self, ttl_secs: u64) -> Self {
        self.session_affinity_ttl_secs = Some(ttl_secs);
        self
    }
}

#[derive(Clone)]
pub enum EngineConfig {
    /// Remote networked engines that we discover via etcd
    Dynamic {
        model: Box<LocalModel>,
        chat_engine_factory: Option<ChatEngineFactoryCallback>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    },

    /// A Text engine receives text, does it's own tokenization and prompt formatting.
    InProcessText {
        engine: Arc<dyn StreamingEngine>,
        model: Box<LocalModel>,
    },

    /// A Tokens engine receives tokens, expects to be wrapped with pre/post processors that handle tokenization.
    InProcessTokens {
        engine: ExecutionContext,
        model: Box<LocalModel>,
        is_prefill: bool,
    },
}

impl EngineConfig {
    pub fn local_model(&self) -> &LocalModel {
        use EngineConfig::*;
        match self {
            Dynamic { model, .. } => model,
            InProcessText { model, .. } => model,
            InProcessTokens { model, .. } => model,
        }
    }

    pub fn chat_engine_factory(&self) -> Option<&ChatEngineFactoryCallback> {
        match self {
            EngineConfig::Dynamic {
                chat_engine_factory,
                ..
            } => chat_engine_factory.as_ref(),
            _ => None,
        }
    }
}
