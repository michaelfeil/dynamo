// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! File-source identity, shared-reader resolution, and construction scopes.

use crate::reader::ReaderInner;
use crate::{B10RoutingConfig, ConfigReader, UnifiedConfig, logging};
use anyhow::{Context, Result};
use arc_swap::ArcSwapOption;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex, Weak};
use std::time::Duration;

const DEFAULT_CONFIG_PATH: &str = "/configs/llm_api_config_router.yaml";
const RELOAD_INTERVAL: Duration = Duration::from_secs(15);

/// All inputs affecting file interpretation, captured before reader registration.
#[derive(Clone, Debug)]
pub struct FileSource {
    path: PathBuf,
    override_group: Option<String>,
    defaults: B10RoutingConfig,
}

impl FileSource {
    pub fn new(path: impl Into<PathBuf>, override_group: Option<String>) -> Self {
        Self {
            path: path.into(),
            override_group,
            defaults: B10RoutingConfig::default(),
        }
    }

    /// Captures environment defaults once. Reloading does not consult the environment.
    pub fn from_env() -> Self {
        Self {
            path: std::env::var_os("DYN_LLMAPI_CONFIG_PATH")
                .map(PathBuf::from)
                .unwrap_or_else(|| DEFAULT_CONFIG_PATH.into()),
            override_group: std::env::var("ENGINE_ARGS_OVERRIDE_GROUP")
                .ok()
                .filter(|s| !s.is_empty()),
            defaults: B10RoutingConfig::from_env(),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn load(&self) -> Result<UnifiedConfig> {
        let contents = std::fs::read_to_string(&self.path)
            .with_context(|| format!("reading configuration {}", self.path.display()))?;
        UnifiedConfig::parse(&contents, self.override_group.as_deref(), &self.defaults)
            .with_context(|| format!("parsing configuration {}", self.path.display()))
    }

    fn key(&self) -> Result<SourceKey> {
        Ok(SourceKey {
            // Do not canonicalize the symlink: mounted ConfigMaps replace its target.
            path: std::path::absolute(&self.path)?,
            override_group: self.override_group.clone(),
            defaults: serde_yaml::to_string(&self.defaults)?,
        })
    }
}

#[derive(Hash, PartialEq, Eq)]
struct SourceKey {
    path: PathBuf,
    override_group: Option<String>,
    defaults: String,
}

/// A registry shares a snapshot slot for each source; it does not own readers forever.
#[derive(Default)]
pub struct ReaderRegistry {
    readers: Mutex<HashMap<SourceKey, Weak<ReaderInner>>>,
}

impl ReaderRegistry {
    fn lookup(&self, source: &FileSource) -> Result<Option<ConfigReader>> {
        let key = source.key()?;
        Ok(self
            .readers
            .lock()
            .unwrap()
            .get(&key)
            .and_then(Weak::upgrade)
            .map(ConfigReader))
    }
    /// Resolve without starting a background thread. Invalid initial files return an error.
    pub fn resolve(&self, source: FileSource) -> Result<ConfigReader> {
        self.resolve_with(source, FileSource::load)
    }

    fn resolve_or_default(&self, source: FileSource) -> Result<ConfigReader> {
        // Load the resolved path even when it came from the built-in default.
        // Constructor-time consumers must not wait for the first polling tick.
        self.resolve_with(source, |source| {
            match source.load() {
                Ok(config) => return Ok(config),
                Err(error) if !logging::warnings_disabled() => tracing::warn!(%error, "using default Baseten configuration until file reload succeeds"),
                Err(_) => {},
            }
            Ok(UnifiedConfig { routing: source.defaults.clone(), ..Default::default() })
        })
    }

    fn resolve_with(
        &self,
        mut source: FileSource,
        initial: impl FnOnce(&FileSource) -> Result<UnifiedConfig>,
    ) -> Result<ConfigReader> {
        let key = source.key()?;
        source.path = key.path.clone();
        let mut readers = self.readers.lock().unwrap();
        readers.retain(|_, reader| reader.strong_count() != 0);
        if let Some(inner) = readers.get(&key).and_then(Weak::upgrade) {
            return Ok(ConfigReader(inner));
        }
        let initial = initial(&source)?;
        let reader = ConfigReader::new(initial, Some(source));
        readers.insert(key, Arc::downgrade(&reader.0));
        Ok(reader)
    }
}

static REGISTRY: LazyLock<ReaderRegistry> = LazyLock::new(ReaderRegistry::default);
// Pin the most recently resolved environment source for legacy free getters.
// Objects retain their own reader, so switching this view cannot rebind them.
static PROCESS_READER: LazyLock<ArcSwapOption<ReaderInner>> = LazyLock::new(ArcSwapOption::empty);

fn default_reader() -> ConfigReader {
    let reader = REGISTRY
        .resolve_or_default(FileSource::from_env())
        .expect("resolving the default configuration source");
    if let Err(error) = reader.start_reloader(RELOAD_INTERVAL) {
        tracing::warn!(%error, "could not start Baseten configuration reloader");
    }
    PROCESS_READER.store(Some(reader.0.clone()));
    reader
}

thread_local! {
    static CONSTRUCTION_READERS: RefCell<Vec<ConfigReader>> = const { RefCell::new(Vec::new()) };
}

/// Resolve at object construction, then retain the returned reader across threads/tasks.
pub fn current_reader() -> ConfigReader {
    scoped_reader().unwrap_or_else(default_reader)
}

fn scoped_reader() -> Option<ConfigReader> {
    CONSTRUCTION_READERS.with(|readers| readers.borrow().last().cloned())
}

/// Resolve at construction when explicitly configured, or reuse a registered source.
/// Without a scope, explicit path, or registered source, standalone libraries use
/// their own defaults. Environment resolution never occurs on snapshot reads.
pub fn try_current_reader() -> Option<ConfigReader> {
    scoped_reader().or_else(|| {
        if std::env::var_os("DYN_LLMAPI_CONFIG_PATH").is_some() {
            Some(default_reader())
        } else {
            REGISTRY
                .lookup(&FileSource::from_env())
                .expect("resolving configuration source")
        }
    })
}

/// Legacy access to the most recently resolved environment reader. Does not inspect
/// the environment once initialized. New objects should retain current_reader().
pub fn process_reader() -> ConfigReader {
    scoped_reader()
        .or_else(|| PROCESS_READER.load_full().map(ConfigReader))
        .unwrap_or_else(default_reader)
}

/// Override reader resolution only during synchronous construction. Do not return an
/// unpolled future expecting the scope to follow it: constructed objects must retain readers.
pub fn with_reader<T>(reader: &ConfigReader, construct: impl FnOnce() -> T) -> T {
    struct Pop;
    impl Drop for Pop {
        fn drop(&mut self) {
            CONSTRUCTION_READERS.with(|readers| {
                readers.borrow_mut().pop();
            });
        }
    }
    CONSTRUCTION_READERS.with(|readers| readers.borrow_mut().push(reader.clone()));
    let _pop = Pop;
    construct()
}

/// Resolve a shared file reader explicitly, without starting its reloader.
pub fn reader_for(source: FileSource) -> Result<ConfigReader> {
    REGISTRY.resolve(source)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_resolution_loads_file_before_polling_or_falls_back() {
        // Exercise default_reader's initialization with an isolated resolved path,
        // without modifying the process-wide /configs mount or starting a poller.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.yaml");
        std::fs::write(
            &path,
            "tensor_parallel_size: 8\nenable_attention_dp: true\nb10_routing_config: {router_decode_block_weight: 7, router_active_replicas: 3}\n",
        )
        .unwrap();
        let reader = ReaderRegistry::default()
            .resolve_or_default(FileSource::new(&path, None))
            .unwrap();
        let snapshot = reader.snapshot();
        assert_eq!(snapshot.routing.router_decode_block_weight, 7.0);
        assert_eq!(snapshot.routing.router_active_replicas, 3);
        assert_eq!(snapshot.runtime.compute_data_parallel_size(), Some(8));

        std::fs::write(&path, "b10_routing_config: [").unwrap();
        let mut source = FileSource::new(&path, None);
        source.defaults.router_decode_block_weight = 2.0;
        let invalid = ReaderRegistry::default()
            .resolve_or_default(source.clone())
            .unwrap();
        assert_eq!(invalid.snapshot().routing.router_decode_block_weight, 2.0);
        std::fs::remove_file(&path).unwrap();
        let missing = ReaderRegistry::default()
            .resolve_or_default(source)
            .unwrap();
        assert_eq!(missing.snapshot().routing.router_decode_block_weight, 2.0);
    }

    #[test]
    fn captured_defaults_are_part_of_reader_identity() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.yaml");
        std::fs::write(&path, "{}").unwrap();
        let registry = ReaderRegistry::default();
        let mut source = FileSource::new(&path, None);
        let first = registry.resolve(source.clone()).unwrap();
        source.defaults.router_decode_block_weight = 7.0;
        let second = registry.resolve(source).unwrap();
        assert!(!first.shares_source(&second));
        first.reload().unwrap();
        assert_eq!(first.snapshot().routing.router_decode_block_weight, 1.0);
        assert_eq!(second.snapshot().routing.router_decode_block_weight, 7.0);
    }

    #[test]
    fn construction_scope_is_removed_after_unwind() {
        let reader = ConfigReader::in_memory(UnifiedConfig::default());
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_reader(&reader, || panic!("construction failed"));
        }));
        CONSTRUCTION_READERS.with(|readers| assert!(readers.borrow().is_empty()));
    }
}
