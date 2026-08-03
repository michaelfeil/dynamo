// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configured model catalog for workload-plane eligibility.
//!
//! Model names in this catalog are canonical served names from routes. Public
//! and legacy aliases are resolved by [`GwpConfig::canonical_model`] before a
//! lookup. Downstream `/v1/models` responses may contain internal model IDs
//! and are not used for routing.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use parking_lot::RwLock;

use crate::config::{EndpointId, GwpConfig};

#[derive(Clone, Default)]
pub struct ModelCatalog {
    inner: Arc<RwLock<HashMap<EndpointId, HashSet<String>>>>,
}

impl ModelCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn replace_from_config(&self, config: &GwpConfig) {
        *self.inner.write() = config.models_by_endpoint();
    }

    pub fn endpoint_serves(&self, endpoint_id: &EndpointId, model: &str) -> bool {
        self.inner
            .read()
            .get(endpoint_id)
            .is_some_and(|models| models.contains(model))
    }

    pub fn endpoints_for_model(&self, model: &str) -> HashSet<EndpointId> {
        self.inner
            .read()
            .iter()
            .filter(|(_, models)| models.contains(model))
            .map(|(endpoint, _)| endpoint.clone())
            .collect()
    }

    pub fn all_models(&self) -> Vec<String> {
        let mut models: Vec<String> = self
            .inner
            .read()
            .values()
            .flat_map(|models| models.iter().cloned())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        models.sort();
        models
    }

    pub fn models_for_endpoint(&self, endpoint_id: &EndpointId) -> HashSet<String> {
        self.inner
            .read()
            .get(endpoint_id)
            .cloned()
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{EndpointConfig, ModelRoute};
    use std::collections::BTreeMap;

    fn config(seed: &str) -> GwpConfig {
        GwpConfig {
            endpoints: BTreeMap::from([(
                EndpointId("a".into()),
                EndpointConfig {
                    ingress_url: url::Url::parse("http://a.example/v1").unwrap(),
                    api_key: String::new(),
                    planner_url: url::Url::parse("http://a.example/deep/health").unwrap(),
                    planner_api_key: None,
                    properties: Default::default(),
                },
            )]),
            routes: vec![ModelRoute {
                models: vec![seed.into()],
                endpoints: vec![EndpointId("a".into())],
            }],
            ..Default::default()
        }
    }

    #[test]
    fn catalog_reconciles_and_aggregates_config() {
        let catalog = ModelCatalog::new();
        catalog.replace_from_config(&config("m1"));
        assert_eq!(catalog.all_models(), vec!["m1"]);
        assert_eq!(catalog.endpoints_for_model("m1").len(), 1);
        assert!(catalog.endpoint_serves(&EndpointId("a".into()), "m1"));
    }

    #[test]
    fn hot_reload_replaces_configured_models() {
        let catalog = ModelCatalog::new();
        let endpoint = EndpointId("a".into());
        catalog.replace_from_config(&config("seed-v1"));
        assert!(catalog.endpoint_serves(&endpoint, "seed-v1"));

        catalog.replace_from_config(&config("seed-v2"));
        assert!(catalog.endpoint_serves(&endpoint, "seed-v2"));
        assert!(!catalog.endpoint_serves(&endpoint, "seed-v1"));
    }
}
