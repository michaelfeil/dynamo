// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Snapshot ownership, publication, and background reload lifecycle.

use crate::{FileSource, UnifiedConfig, logging};
use anyhow::{Context, Result};
use arc_swap::ArcSwap;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

pub(super) struct ReaderInner {
    snapshot: ArcSwap<UnifiedConfig>,
    source: Option<FileSource>,
    writer: Mutex<()>,
    reloader: Mutex<Option<Arc<ReloadControl>>>,
}

impl Drop for ReaderInner {
    fn drop(&mut self) {
        if let Some(control) = self.reloader.get_mut().unwrap().take() {
            control.stop();
        }
    }
}

#[derive(Default)]
struct ReloadControl {
    stopped: Mutex<bool>,
    wake: Condvar,
}

impl ReloadControl {
    fn stop(&self) {
        *self.stopped.lock().unwrap() = true;
        self.wake.notify_all();
    }

    fn wait(&self, interval: Duration) -> bool {
        let (stopped, _) = self
            .wake
            .wait_timeout_while(self.stopped.lock().unwrap(), interval, |stopped| !*stopped)
            .unwrap();
        !*stopped
    }
}

/// Stable handle to a replaceable snapshot. Cloning never resolves another source.
#[derive(Clone)]
pub struct ConfigReader(pub(super) Arc<ReaderInner>);

impl std::fmt::Debug for ConfigReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConfigReader")
            .field("source", &self.0.source.as_ref().map(FileSource::path))
            .finish_non_exhaustive()
    }
}

impl ConfigReader {
    pub(super) fn new(config: UnifiedConfig, source: Option<FileSource>) -> Self {
        Self(Arc::new(ReaderInner {
            snapshot: ArcSwap::from_pointee(config.sanitize()),
            source,
            writer: Mutex::new(()),
            reloader: Mutex::new(None),
        }))
    }

    /// An isolated, writable reader with no file or reload task.
    pub fn in_memory(config: UnifiedConfig) -> Self {
        Self::new(config, None)
    }

    pub fn snapshot(&self) -> Arc<UnifiedConfig> {
        self.0.snapshot.load_full()
    }

    pub fn shares_source(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// Publish in memory; does not write the mounted file. A later reload may supersede it.
    pub fn replace(&self, config: UnifiedConfig) {
        let _writer = self.0.writer.lock().unwrap();
        self.0.snapshot.store(Arc::new(config.sanitize()));
    }

    /// Read, validate, and publish under one writer lock. Failure retains the old snapshot.
    pub fn reload(&self) -> Result<bool> {
        self.reload_if_active(None)
    }

    fn reload_if_active(&self, control: Option<&ReloadControl>) -> Result<bool> {
        let _writer = self.0.writer.lock().unwrap();
        if control.is_some_and(|control| *control.stopped.lock().unwrap()) {
            return Ok(false);
        }
        let source = self
            .0
            .source
            .as_ref()
            .context("in-memory reader has no file source")?;
        let candidate = source.load()?;
        if *self.snapshot() == candidate {
            return Ok(false);
        }
        self.0.snapshot.store(Arc::new(candidate));
        Ok(true)
    }

    pub fn validate_file(&self) -> bool {
        self.0
            .source
            .as_ref()
            .is_some_and(|source| source.load().is_ok())
    }

    /// Idempotent across readers sharing a source. Stops when the last reader is dropped.
    pub fn start_reloader(&self, interval: Duration) -> Result<()> {
        anyhow::ensure!(!interval.is_zero(), "reload interval must be positive");
        anyhow::ensure!(
            self.0.source.is_some(),
            "in-memory reader has no file source"
        );
        let mut reloader = self.0.reloader.lock().unwrap();
        if reloader.is_some() {
            return Ok(());
        }
        let control = Arc::new(ReloadControl::default());
        let worker_control = control.clone();
        // Capture logging policy before the polling thread starts; it must not read env.
        let warnings_disabled = logging::warnings_disabled();
        let reader = Arc::downgrade(&self.0);
        std::thread::Builder::new().name("baseten-configmap".into()).spawn(move || {
            while worker_control.wait(interval) {
                let Some(inner) = reader.upgrade() else { break };
                let reader = ConfigReader(inner);
                match reader.reload_if_active(Some(&worker_control)) {
                    Ok(true) => tracing::info!("Baseten configuration reloaded"),
                    Ok(false) if logging::log_no_changes() => tracing::info!("Baseten configuration unchanged"),
                    Err(error) if !warnings_disabled => tracing::warn!(%error, "configuration reload failed; retaining previous snapshot"),
                    _ => {},
                }
            }
        })?;
        *reloader = Some(control);
        Ok(())
    }

    pub fn stop_reloader(&self) {
        let _writer = self.0.writer.lock().unwrap();
        if let Some(control) = self.0.reloader.lock().unwrap().take() {
            control.stop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ReaderRegistry;

    #[test]
    fn a_source_has_one_reloader_and_last_reader_stops_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.yaml");
        std::fs::write(&path, "{}").unwrap();
        let registry = ReaderRegistry::default();
        let first = registry.resolve(FileSource::new(&path, None)).unwrap();
        let second = registry.resolve(FileSource::new(&path, None)).unwrap();
        first.start_reloader(Duration::from_secs(3600)).unwrap();
        let control = first.0.reloader.lock().unwrap().as_ref().unwrap().clone();
        second.start_reloader(Duration::from_secs(3600)).unwrap();
        assert!(Arc::ptr_eq(
            &control,
            second.0.reloader.lock().unwrap().as_ref().unwrap()
        ));
        drop(first);
        assert!(!*control.stopped.lock().unwrap());
        drop(second);
        assert!(*control.stopped.lock().unwrap());
    }

    #[test]
    fn stopped_poll_does_not_overwrite_a_later_publication() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.yaml");
        std::fs::write(&path, "{}").unwrap();
        let reader = ReaderRegistry::default()
            .resolve(FileSource::new(&path, None))
            .unwrap();
        reader.start_reloader(Duration::from_secs(3600)).unwrap();
        let control = reader.0.reloader.lock().unwrap().as_ref().unwrap().clone();
        reader.stop_reloader();
        let mut replacement = UnifiedConfig::default();
        replacement.routing.router_decode_block_weight = 9.0;
        reader.replace(replacement);
        assert!(!reader.reload_if_active(Some(&control)).unwrap());
        assert_eq!(reader.snapshot().routing.router_decode_block_weight, 9.0);
    }
}
