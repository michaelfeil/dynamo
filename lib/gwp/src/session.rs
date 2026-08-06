// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Session affinity for GWP. See design doc "Stick/unstick" mental model,
//! Decision 2, and "Approximate router lifecycle".
//!
//! The resolver honors a binding iff the bound worker is still alive (per the
//! current [`TopologySnapshot`](crate::topology::TopologySnapshot); otherwise it
//! unsticks and falls through to the approximate router. Stickiness is a
//! **bypass for the scoring decision** (`find_best_match`), not a bypass of
//! the scheduler: a stick still tokenizes and `add_request`s the bound worker
//! candidate so in-flight load stays honest. That accounting is performed in
//! the bounded optimistic-tokenization path while Envoy proceeds with egress.
//!
//! The trait is **async**, diverging from the sync
//! `dynamo_llm::kv_router::sticky::AffinityStore`: etcd is a network round
//! trip, and a sync trait would force the async scheduling service to block.
//! Semantics otherwise mirror the llm trait.
//!
//! Affinity is an optimization, not a correctness requirement. An etcd read
//! error is therefore treated as an affinity miss and normal routing proceeds;
//! failed writes are logged and dropped. GWP never invents replica-local
//! affinity while configured for a shared backend.

use std::fmt::Write;
use std::future::Future;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use dynamo_kv_router::protocols::{
    BlockHashOptions, WorkerWithDpRank, compute_block_hash, compute_block_hash_for_seq,
    compute_seq_hash_for_block,
};

use crate::config::EndpointId;
use crate::metrics::GwpMetrics;

const AFFINITY_OPERATION_TIMEOUT: Duration = Duration::from_millis(200);
const AFFINITY_MAX_IN_FLIGHT: usize = 256;

/// Persisted session placement. `endpoint_id` was added after the original
/// worker-only format and remains optional so existing Redis/etcd values can
/// be read during a rolling upgrade.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AffinityBinding {
    pub worker: WorkerWithDpRank,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint_id: Option<EndpointId>,
}

impl AffinityBinding {
    pub fn worker_only(worker: WorkerWithDpRank) -> Self {
        Self {
            worker,
            endpoint_id: None,
        }
    }

    pub fn with_endpoint(worker: WorkerWithDpRank, endpoint_id: EndpointId) -> Self {
        Self {
            worker,
            endpoint_id: Some(endpoint_id),
        }
    }
}

/// Trait abstraction over the affinity store. Production GWP uses a shared
/// etcd or Redis backend; the test-only implementation keeps unit and CI tests
/// self-contained.
///
/// Semantics (mirroring `dynamo_llm::kv_router::sticky::AffinityStore`):
/// - `peek` reads the binding **without** refreshing TTL — used on the request
///   path for the stick/unstick check. The caller then consults the
///   [`TopologySnapshot`](crate::topology::TopologySnapshot) to decide stick vs. unstick.
/// - `put` writes the selected endpoint after the upstream responds
///   successfully. Backend errors are logged, not returned.
/// - `get` refreshes TTL; not used on the hot path, kept for admin/diag.
#[async_trait]
pub trait AffinityStore: Send + Sync {
    /// Stable bounded metric label for this backend.
    fn backend_name(&self) -> &'static str;
    /// Read the binding without refreshing TTL. This is what the resolver calls
    /// for the stick/unstick check.
    async fn peek(&self, session_id: &str) -> anyhow::Result<Option<WorkerWithDpRank>>;
    /// Read the worker plus its last routed endpoint when the backend has the
    /// newer binding format. Legacy bindings return `endpoint_id: None`.
    async fn peek_binding(&self, session_id: &str) -> anyhow::Result<Option<AffinityBinding>> {
        Ok(self
            .peek(session_id)
            .await?
            .map(AffinityBinding::worker_only))
    }
    /// Read with TTL refresh. Not used on the hot path; for diagnostics.
    async fn get(&self, session_id: &str) -> anyhow::Result<Option<WorkerWithDpRank>>;
    /// Record the last worker confirmed by the selected endpoint.
    async fn put(
        &self,
        session_id: &str,
        worker: WorkerWithDpRank,
        ttl: Duration,
    ) -> anyhow::Result<()>;
    /// Persist a worker and endpoint together. Backends that have not opted in
    /// retain worker affinity via the compatibility implementation.
    async fn put_binding(
        &self,
        session_id: &str,
        binding: AffinityBinding,
        ttl: Duration,
    ) -> anyhow::Result<()> {
        self.put(session_id, binding.worker, ttl).await
    }
    async fn remove(&self, session_id: &str) -> anyhow::Result<bool>;
}

/// Adds availability isolation and backend-operation error counters around an
/// affinity store. Every operation is capped at 200 ms, and a semaphore
/// bulkhead bounds the number of calls waiting on an unhealthy backend. Calls
/// beyond that bound fail immediately so request routing can fall through.
pub struct InstrumentedAffinityStore {
    inner: std::sync::Arc<dyn AffinityStore>,
    metrics: GwpMetrics,
    operation_timeout: Duration,
    max_in_flight: usize,
    permits: std::sync::Arc<tokio::sync::Semaphore>,
}

impl InstrumentedAffinityStore {
    pub fn new(inner: std::sync::Arc<dyn AffinityStore>, metrics: GwpMetrics) -> Self {
        Self::with_limits(
            inner,
            metrics,
            AFFINITY_OPERATION_TIMEOUT,
            AFFINITY_MAX_IN_FLIGHT,
        )
    }

    fn with_limits(
        inner: std::sync::Arc<dyn AffinityStore>,
        metrics: GwpMetrics,
        operation_timeout: Duration,
        max_in_flight: usize,
    ) -> Self {
        Self {
            inner,
            metrics,
            operation_timeout,
            max_in_flight,
            permits: std::sync::Arc::new(tokio::sync::Semaphore::new(max_in_flight)),
        }
    }

    #[cfg(test)]
    pub(crate) fn new_with_limits(
        inner: std::sync::Arc<dyn AffinityStore>,
        metrics: GwpMetrics,
        operation_timeout: Duration,
        max_in_flight: usize,
    ) -> Self {
        Self::with_limits(inner, metrics, operation_timeout, max_in_flight)
    }

    async fn execute<T>(
        &self,
        operation: &'static str,
        future: impl Future<Output = anyhow::Result<T>>,
    ) -> anyhow::Result<T> {
        let started = Instant::now();
        let backend = self.backend_name();
        let _permit = match self.permits.clone().try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => {
                self.metrics.observe_affinity_operation(
                    backend,
                    operation,
                    "bulkhead_rejected",
                    started.elapsed(),
                );
                return Err(anyhow::anyhow!(
                    "{backend} affinity {operation} skipped: {} operations already in flight",
                    self.max_in_flight
                ));
            }
        };
        match tokio::time::timeout(self.operation_timeout, future).await {
            Ok(Ok(value)) => {
                self.metrics.observe_affinity_operation(
                    backend,
                    operation,
                    "success",
                    started.elapsed(),
                );
                Ok(value)
            }
            Ok(Err(error)) => {
                self.metrics.observe_affinity_operation(
                    backend,
                    operation,
                    "error",
                    started.elapsed(),
                );
                Err(error)
            }
            Err(_) => {
                self.metrics.observe_affinity_operation(
                    backend,
                    operation,
                    "timeout",
                    started.elapsed(),
                );
                Err(anyhow::anyhow!(
                    "{backend} affinity {operation} timed out after {} ms",
                    self.operation_timeout.as_millis()
                ))
            }
        }
    }

    fn record<T>(&self, operation: &str, result: &anyhow::Result<T>) {
        if result.is_err() {
            self.metrics
                .record_affinity_backend_error(self.backend_name(), operation);
        }
    }
}

#[async_trait]
impl AffinityStore for InstrumentedAffinityStore {
    fn backend_name(&self) -> &'static str {
        self.inner.backend_name()
    }

    async fn peek(&self, session_id: &str) -> anyhow::Result<Option<WorkerWithDpRank>> {
        let result = self.execute("peek", self.inner.peek(session_id)).await;
        self.record("peek", &result);
        result
    }

    async fn peek_binding(&self, session_id: &str) -> anyhow::Result<Option<AffinityBinding>> {
        let result = self
            .execute("peek", self.inner.peek_binding(session_id))
            .await;
        self.record("peek", &result);
        result
    }

    async fn get(&self, session_id: &str) -> anyhow::Result<Option<WorkerWithDpRank>> {
        let result = self.execute("get", self.inner.get(session_id)).await;
        self.record("get", &result);
        result
    }

    async fn put(
        &self,
        session_id: &str,
        worker: WorkerWithDpRank,
        ttl: Duration,
    ) -> anyhow::Result<()> {
        let result = self
            .execute("put", self.inner.put(session_id, worker, ttl))
            .await;
        self.record("put", &result);
        result
    }

    async fn put_binding(
        &self,
        session_id: &str,
        binding: AffinityBinding,
        ttl: Duration,
    ) -> anyhow::Result<()> {
        let result = self
            .execute("put", self.inner.put_binding(session_id, binding, ttl))
            .await;
        self.record("put", &result);
        result
    }

    async fn remove(&self, session_id: &str) -> anyhow::Result<bool> {
        let result = self.execute("remove", self.inner.remove(session_id)).await;
        self.record("remove", &result);
        result
    }
}

/// Mint a new session id of the form `base10-<uuid>`.
pub fn mint_session_id() -> String {
    format!("base10-{}", uuid::Uuid::new_v4())
}

/// Resolve the affinity key for an OpenAI request. An explicit routing header
/// always wins; otherwise the OpenAI `user` field provides stable affinity
/// for clients that already set it. Missing or non-string `user` values leave
/// the request without an affinity key so the caller can mint one.
pub fn request_session_id(
    header_session_id: Option<String>,
    body: &serde_json::Value,
) -> Option<String> {
    header_session_id.or_else(|| {
        body.get("user")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
    })
}

/// Number of consecutive rolling sequence hashes encoded in a derived prompt
/// affinity key. Four u64 values keep accidental collisions negligible for a
/// routing hint while remaining compact enough for a response header.
const PROMPT_HASH_COMPONENTS: usize = 4;

/// Derive a stable, ASCII-only affinity key from the prompt prefix ending at
/// `token_position`. The four rolling sequence hashes immediately preceding
/// the cutoff are encoded, and the model name is separately hashed to prevent
/// cross-model affinity.
///
/// Returns `None` until the prompt reaches the configured position. Only full
/// routing blocks participate, matching the approximate KV indexer's hashing
/// semantics exactly.
pub fn prompt_hash_session_id(
    model: &str,
    tokens: &[u32],
    block_size: u32,
    token_position: usize,
) -> Option<String> {
    let block_size = usize::try_from(block_size).ok().filter(|size| *size > 0)?;
    if tokens.len() < token_position {
        return None;
    }
    let cutoff_blocks = token_position / block_size;
    if cutoff_blocks < PROMPT_HASH_COMPONENTS {
        return None;
    }
    let cutoff_tokens = cutoff_blocks.checked_mul(block_size)?;
    let block_hashes = compute_block_hash_for_seq(
        &tokens[..cutoff_tokens],
        u32::try_from(block_size).ok()?,
        BlockHashOptions::default(),
    );
    let sequence_hashes = compute_seq_hash_for_block(&block_hashes);
    let components = sequence_hashes.get(cutoff_blocks - PROMPT_HASH_COMPONENTS..cutoff_blocks)?;

    let model_hash = compute_block_hash(model.as_bytes()).0;
    let mut session_id = format!("prompt-v1-{model_hash:016x}");
    for hash in components {
        write!(&mut session_id, "-{hash:016x}").expect("writing to String cannot fail");
    }
    Some(session_id)
}

/// Resolve the first non-empty session identifier advertised by a known
/// OpenAI-compatible client. GWP's canonical `x-session-id` wins; the
/// remaining names are compatibility fallbacks in deterministic precedence
/// order.
pub fn client_session_id<'a>(
    mut get_header: impl FnMut(&str) -> Option<&'a str>,
) -> Option<String> {
    headers::SESSION_ID_CANDIDATES
        .iter()
        .find_map(|name| get_header(name).filter(|value| !value.is_empty()))
        .map(str::to_owned)
}

/// Session-related header names GWP consumes. See design doc Decision 6 and
/// Decision 7. The Envoy filters propagate the selected session ID to the
/// client response.
pub mod headers {
    /// The session id used this turn (header, OpenAI `user`, or minted fallback).
    pub const SESSION_ID: &str = "x-session-id";
    /// Vendor-neutral session-affinity compatibility header.
    pub const SESSION_AFFINITY: &str = "x-session-affinity";
    /// Codex native session header.
    pub const CODEX_SESSION_ID: &str = "session-id";
    /// Claude Code top-level session header.
    pub const CLAUDE_CODE_SESSION_ID: &str = "x-claude-code-session-id";
    /// OpenCode subagent parent session header.
    pub const PARENT_SESSION_ID: &str = "x-parent-session-id";
    /// Claude Code subagent identifier.
    pub const CLAUDE_CODE_AGENT_ID: &str = "x-claude-code-agent-id";
    /// Claude Code subagent parent identifier.
    pub const CLAUDE_CODE_PARENT_AGENT_ID: &str = "x-claude-code-parent-agent-id";
    /// Deployment-configured generic user identifier; lowest-priority fallback.
    pub const USER_ID: &str = "user-id";
    /// Canonical session header followed by compatibility fallbacks.
    pub const SESSION_ID_CANDIDATES: &[&str] = &[
        SESSION_ID,
        SESSION_AFFINITY,
        CODEX_SESSION_ID,
        CLAUDE_CODE_SESSION_ID,
        PARENT_SESSION_ID,
        CLAUDE_CODE_AGENT_ID,
        CLAUDE_CODE_PARENT_AGENT_ID,
        USER_ID,
    ];
    /// Dimensioned hard and soft routing constraints stamped by Alyx.
    pub const ROUTING_REQUIREMENTS: &str = "x-baseten-model-apis-routing-requirements";
    /// Stable configured endpoint ID GWP routed to this turn.
    pub const ROUTED_ENDPOINT: &str = "x-routed-endpoint";
    /// Required `WorkerId` set by the remote Dynamo frontend naming the worker
    /// that served a successful request.
    pub const SERVED_WORKER: &str = "x-baseten-dyn-worker-id";
}

// ---------------------------------------------------------------------------
// In-memory store
// ---------------------------------------------------------------------------

#[cfg(test)]
struct InMemoryEntry {
    binding: AffinityBinding,
    ttl: Duration,
    expires_at: tokio::time::Instant,
}

/// Test store: a `DashMap` with lazy expiry on read plus an
/// opportunistic sweep every [`SWEEP_EVERY_PUTS`] writes (no background task —
/// the store must be constructible outside a runtime and never leak unbounded
/// under write-heavy load).
#[cfg(test)]
#[derive(Default)]
pub struct InMemoryAffinityStore {
    map: dashmap::DashMap<String, InMemoryEntry>,
    puts_since_sweep: std::sync::atomic::AtomicU64,
}

#[cfg(test)]
const SWEEP_EVERY_PUTS: u64 = 4096;

#[cfg(test)]
impl InMemoryAffinityStore {
    pub fn new() -> Self {
        Self::default()
    }

    fn read(&self, session_id: &str, refresh: bool) -> Option<AffinityBinding> {
        let now = tokio::time::Instant::now();
        let mut entry = self.map.get_mut(session_id)?;
        if entry.expires_at <= now {
            drop(entry);
            self.map.remove(session_id);
            return None;
        }
        if refresh {
            let ttl = entry.ttl;
            entry.expires_at = now + ttl;
        }
        Some(entry.binding.clone())
    }

    fn sweep_if_due(&self) {
        use std::sync::atomic::Ordering;
        let n = self.puts_since_sweep.fetch_add(1, Ordering::Relaxed);
        if n != 0 && n.is_multiple_of(SWEEP_EVERY_PUTS) {
            let now = tokio::time::Instant::now();
            self.map.retain(|_, e| e.expires_at > now);
        }
    }
}

#[cfg(test)]
#[async_trait]
impl AffinityStore for InMemoryAffinityStore {
    fn backend_name(&self) -> &'static str {
        "memory"
    }

    async fn peek(&self, session_id: &str) -> anyhow::Result<Option<WorkerWithDpRank>> {
        Ok(self.read(session_id, false).map(|binding| binding.worker))
    }

    async fn peek_binding(&self, session_id: &str) -> anyhow::Result<Option<AffinityBinding>> {
        Ok(self.read(session_id, false))
    }

    async fn get(&self, session_id: &str) -> anyhow::Result<Option<WorkerWithDpRank>> {
        Ok(self.read(session_id, true).map(|binding| binding.worker))
    }

    async fn put(
        &self,
        session_id: &str,
        worker: WorkerWithDpRank,
        ttl: Duration,
    ) -> anyhow::Result<()> {
        self.sweep_if_due();
        self.map.insert(
            session_id.to_string(),
            InMemoryEntry {
                binding: AffinityBinding::worker_only(worker),
                ttl,
                expires_at: tokio::time::Instant::now() + ttl,
            },
        );
        Ok(())
    }

    async fn put_binding(
        &self,
        session_id: &str,
        binding: AffinityBinding,
        ttl: Duration,
    ) -> anyhow::Result<()> {
        self.sweep_if_due();
        self.map.insert(
            session_id.to_string(),
            InMemoryEntry {
                binding,
                ttl,
                expires_at: tokio::time::Instant::now() + ttl,
            },
        );
        Ok(())
    }

    async fn remove(&self, session_id: &str) -> anyhow::Result<bool> {
        Ok(self.map.remove(session_id).is_some())
    }
}

// ---------------------------------------------------------------------------
// Redis store (multi-replica backend)
// ---------------------------------------------------------------------------

#[cfg(feature = "redis")]
pub use redis_store::RedisAffinityStore;

#[cfg(feature = "redis")]
mod redis_store {
    use super::*;

    const GET_AND_REFRESH: &str = r#"
local value = redis.call("GET", KEYS[1])
if value then
  local binding = cjson.decode(value)
  redis.call("EXPIRE", KEYS[1], math.max(1, binding.ttl_secs))
end
return value
"#;

    /// TTL is persisted with the binding because the diagnostic `get` method
    /// refreshes the original lifetime. Request-path `peek` never refreshes it.
    #[derive(serde::Serialize, serde::Deserialize)]
    struct Binding {
        worker: WorkerWithDpRank,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        endpoint_id: Option<EndpointId>,
        ttl_secs: u64,
    }

    /// Redis-backed worker affinity. `ConnectionManager` is cheap to clone and
    /// reconnects after transient connection loss.
    pub struct RedisAffinityStore {
        connection: redis::aio::ConnectionManager,
        key_prefix: String,
    }

    impl RedisAffinityStore {
        pub async fn connect(url: &str, key_prefix: String) -> anyhow::Result<Self> {
            let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
            let client = redis::Client::open(url)?;
            let mut connection =
                tokio::time::timeout(AFFINITY_OPERATION_TIMEOUT, client.get_connection_manager())
                    .await
                    .map_err(|_| {
                        anyhow::anyhow!(
                            "Redis connection timed out after {} ms",
                            AFFINITY_OPERATION_TIMEOUT.as_millis()
                        )
                    })??;
            let _: String = tokio::time::timeout(
                AFFINITY_OPERATION_TIMEOUT,
                redis::cmd("PING").query_async(&mut connection),
            )
            .await
            .map_err(|_| {
                anyhow::anyhow!(
                    "Redis PING timed out after {} ms",
                    AFFINITY_OPERATION_TIMEOUT.as_millis()
                )
            })??;
            Ok(Self {
                connection,
                key_prefix,
            })
        }

        fn key(&self, session_id: &str) -> String {
            format!("{}{session_id}", self.key_prefix)
        }

        async fn read_redis(&self, session_id: &str) -> anyhow::Result<Option<Binding>> {
            let mut connection = self.connection.clone();
            let value: Option<Vec<u8>> = redis::cmd("GET")
                .arg(self.key(session_id))
                .query_async(&mut connection)
                .await?;
            value
                .map(|value| serde_json::from_slice(&value).map_err(anyhow::Error::from))
                .transpose()
        }

        async fn write_redis(
            &self,
            session_id: &str,
            binding: AffinityBinding,
            ttl: Duration,
        ) -> anyhow::Result<()> {
            let ttl_secs = ttl.as_secs().max(1);
            let value = serde_json::to_vec(&Binding {
                worker: binding.worker,
                endpoint_id: binding.endpoint_id,
                ttl_secs,
            })?;
            let mut connection = self.connection.clone();
            let _: () = redis::cmd("SET")
                .arg(self.key(session_id))
                .arg(value)
                .arg("EX")
                .arg(ttl_secs)
                .query_async(&mut connection)
                .await?;
            Ok(())
        }

        async fn read_and_refresh_redis(
            &self,
            session_id: &str,
        ) -> anyhow::Result<Option<Binding>> {
            let mut connection = self.connection.clone();
            let value: Option<Vec<u8>> = redis::cmd("EVAL")
                .arg(GET_AND_REFRESH)
                .arg(1)
                .arg(self.key(session_id))
                .query_async(&mut connection)
                .await?;
            value
                .map(|value| serde_json::from_slice(&value).map_err(anyhow::Error::from))
                .transpose()
        }
    }

    #[async_trait]
    impl AffinityStore for RedisAffinityStore {
        fn backend_name(&self) -> &'static str {
            "redis"
        }

        async fn peek(&self, session_id: &str) -> anyhow::Result<Option<WorkerWithDpRank>> {
            Ok(self
                .read_redis(session_id)
                .await?
                .map(|binding| binding.worker))
        }

        async fn peek_binding(&self, session_id: &str) -> anyhow::Result<Option<AffinityBinding>> {
            Ok(self
                .read_redis(session_id)
                .await?
                .map(|binding| AffinityBinding {
                    worker: binding.worker,
                    endpoint_id: binding.endpoint_id,
                }))
        }

        async fn get(&self, session_id: &str) -> anyhow::Result<Option<WorkerWithDpRank>> {
            Ok(self
                .read_and_refresh_redis(session_id)
                .await?
                .map(|binding| binding.worker))
        }

        async fn put(
            &self,
            session_id: &str,
            worker: WorkerWithDpRank,
            ttl: Duration,
        ) -> anyhow::Result<()> {
            self.write_redis(session_id, AffinityBinding::worker_only(worker), ttl)
                .await
        }

        async fn put_binding(
            &self,
            session_id: &str,
            binding: AffinityBinding,
            ttl: Duration,
        ) -> anyhow::Result<()> {
            self.write_redis(session_id, binding, ttl).await
        }

        async fn remove(&self, session_id: &str) -> anyhow::Result<bool> {
            let mut connection = self.connection.clone();
            let deleted: u64 = redis::cmd("DEL")
                .arg(self.key(session_id))
                .query_async(&mut connection)
                .await?;
            Ok(deleted > 0)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn legacy_worker_only_binding_remains_readable() {
            let worker = WorkerWithDpRank::from_worker_id(42);
            let legacy = serde_json::json!({
                "worker": worker,
                "ttl_secs": 60
            });
            let binding: Binding = serde_json::from_value(legacy).unwrap();
            assert_eq!(binding.worker, worker);
            assert_eq!(binding.endpoint_id, None);
        }

        /// Requires a live Redis at DYN_GWP_TEST_REDIS.
        #[tokio::test]
        #[ignore]
        async fn redis_roundtrip_and_expiry() {
            let url = std::env::var("DYN_GWP_TEST_REDIS")
                .unwrap_or_else(|_| "redis://127.0.0.1:6379/".to_string());
            let prefix = format!("gwp:test:{}:", uuid::Uuid::new_v4());
            let store = RedisAffinityStore::connect(&url, prefix).await.unwrap();
            let sid = mint_session_id();
            let worker = WorkerWithDpRank::from_worker_id(42);

            assert_eq!(store.peek(&sid).await.unwrap(), None);
            store
                .put(&sid, worker, Duration::from_secs(1))
                .await
                .unwrap();
            assert_eq!(store.peek(&sid).await.unwrap(), Some(worker));
            let last_worker = WorkerWithDpRank::from_worker_id(43);
            store
                .put(&sid, last_worker, Duration::from_secs(1))
                .await
                .unwrap();
            assert_eq!(
                store.peek(&sid).await.unwrap(),
                Some(last_worker),
                "last returned worker wins the session binding"
            );
            tokio::time::sleep(Duration::from_millis(1100)).await;
            assert_eq!(store.peek(&sid).await.unwrap(), None);

            store
                .put(&sid, worker, Duration::from_secs(5))
                .await
                .unwrap();
            assert_eq!(store.get(&sid).await.unwrap(), Some(worker));
            assert!(store.remove(&sid).await.unwrap());
            assert_eq!(store.peek(&sid).await.unwrap(), None);
        }
    }
}

// ---------------------------------------------------------------------------
// etcd store (multi-replica backend)
// ---------------------------------------------------------------------------

#[cfg(feature = "etcd")]
pub use etcd_store::EtcdAffinityStore;

#[cfg(feature = "etcd")]
mod etcd_store {
    use super::*;

    const KEY_PREFIX: &str = "gwp/affinity/";

    /// What we persist per session. TTL travels with the value so `get` can
    /// refresh with the original TTL.
    #[derive(serde::Serialize, serde::Deserialize)]
    struct Binding {
        worker: WorkerWithDpRank,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        endpoint_id: Option<EndpointId>,
        ttl_secs: u64,
    }

    /// etcd-backed affinity (design doc Decision 2). One lease per `put`, no
    /// keepalive — the binding expires unless re-`put` by a later turn, which
    /// is exactly the sliding-TTL-on-success semantic. Reads are serializable
    /// (served locally by the contacted member): a stale binding is caught by
    /// the liveness check, so linearizable reads buy nothing on this path.
    ///
    pub struct EtcdAffinityStore {
        client: etcd_client::Client,
    }

    impl EtcdAffinityStore {
        pub async fn connect(endpoints: Vec<String>) -> anyhow::Result<Self> {
            let client = etcd_client::Client::connect(endpoints, None).await?;
            Ok(Self { client })
        }

        fn key(session_id: &str) -> String {
            format!("{KEY_PREFIX}{session_id}")
        }

        async fn read_etcd(&self, session_id: &str) -> anyhow::Result<Option<Binding>> {
            let mut kv = self.client.kv_client();
            let resp = kv
                .get(
                    Self::key(session_id),
                    Some(etcd_client::GetOptions::new().with_serializable()),
                )
                .await?;
            let Some(kv) = resp.kvs().first() else {
                return Ok(None);
            };
            Ok(Some(serde_json::from_slice(kv.value())?))
        }

        async fn write_etcd(
            &self,
            session_id: &str,
            binding: AffinityBinding,
            ttl: Duration,
        ) -> anyhow::Result<()> {
            let ttl_secs = ttl.as_secs().max(1) as i64;
            let mut client = self.client.clone();
            let lease = client.lease_grant(ttl_secs, None).await?;
            let value = serde_json::to_vec(&Binding {
                worker: binding.worker,
                endpoint_id: binding.endpoint_id,
                ttl_secs: ttl_secs as u64,
            })?;
            client
                .kv_client()
                .put(
                    Self::key(session_id),
                    value,
                    Some(etcd_client::PutOptions::new().with_lease(lease.id())),
                )
                .await?;
            Ok(())
        }
    }

    #[async_trait]
    impl AffinityStore for EtcdAffinityStore {
        fn backend_name(&self) -> &'static str {
            "etcd"
        }

        async fn peek(&self, session_id: &str) -> anyhow::Result<Option<WorkerWithDpRank>> {
            Ok(self
                .read_etcd(session_id)
                .await?
                .map(|binding| binding.worker))
        }

        async fn peek_binding(&self, session_id: &str) -> anyhow::Result<Option<AffinityBinding>> {
            Ok(self
                .read_etcd(session_id)
                .await?
                .map(|binding| AffinityBinding {
                    worker: binding.worker,
                    endpoint_id: binding.endpoint_id,
                }))
        }

        async fn get(&self, session_id: &str) -> anyhow::Result<Option<WorkerWithDpRank>> {
            match self.read_etcd(session_id).await {
                Ok(Some(binding)) => {
                    // Refresh: re-put with the binding's original TTL.
                    let ttl = Duration::from_secs(binding.ttl_secs);
                    self.put_binding(
                        session_id,
                        AffinityBinding {
                            worker: binding.worker,
                            endpoint_id: binding.endpoint_id,
                        },
                        ttl,
                    )
                    .await?;
                    Ok(Some(binding.worker))
                }
                Ok(None) => Ok(None),
                Err(error) => Err(error),
            }
        }

        async fn put(
            &self,
            session_id: &str,
            worker: WorkerWithDpRank,
            ttl: Duration,
        ) -> anyhow::Result<()> {
            self.write_etcd(session_id, AffinityBinding::worker_only(worker), ttl)
                .await
        }

        async fn put_binding(
            &self,
            session_id: &str,
            binding: AffinityBinding,
            ttl: Duration,
        ) -> anyhow::Result<()> {
            self.write_etcd(session_id, binding, ttl).await
        }

        async fn remove(&self, session_id: &str) -> anyhow::Result<bool> {
            let mut kv = self.client.kv_client();
            Ok(kv.delete(Self::key(session_id), None).await?.deleted() > 0)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Requires a live etcd at DYN_GWP_TEST_ETCD (e.g. http://127.0.0.1:2379).
        #[tokio::test]
        #[ignore]
        async fn etcd_roundtrip() {
            let endpoint = std::env::var("DYN_GWP_TEST_ETCD")
                .unwrap_or_else(|_| "http://127.0.0.1:2379".to_string());
            let store = EtcdAffinityStore::connect(vec![endpoint]).await.unwrap();
            let sid = mint_session_id();
            let worker = WorkerWithDpRank::from_worker_id(42);

            assert_eq!(store.peek(&sid).await.unwrap(), None);
            store
                .put(&sid, worker, Duration::from_secs(5))
                .await
                .unwrap();
            assert_eq!(store.peek(&sid).await.unwrap(), Some(worker));
            assert!(store.remove(&sid).await.unwrap());
            assert_eq!(store.peek(&sid).await.unwrap(), None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_session_id_prefers_header_then_openai_user() {
        let body = serde_json::json!({"user": "openai-user"});
        assert_eq!(
            request_session_id(Some("header-session".into()), &body).as_deref(),
            Some("header-session")
        );
        assert_eq!(
            request_session_id(None, &body).as_deref(),
            Some("openai-user")
        );
        assert_eq!(
            request_session_id(None, &serde_json::json!({"user": 7})),
            None
        );
        assert_eq!(request_session_id(None, &serde_json::json!({})), None);
    }

    #[test]
    fn prompt_hash_session_id_is_stable_after_cutoff_and_model_scoped() {
        let prefix: Vec<u32> = (0..24).collect();
        let mut extended = prefix.clone();
        extended.extend(24..40);

        let original = prompt_hash_session_id("model-a", &prefix, 4, 20).unwrap();
        let appended = prompt_hash_session_id("model-a", &extended, 4, 20).unwrap();
        let other_model = prompt_hash_session_id("model-b", &prefix, 4, 20).unwrap();

        assert_eq!(original, appended);
        assert_ne!(original, other_model);
        assert!(original.starts_with("prompt-v1-"));
        assert_eq!(original.split('-').count(), 7);
    }

    #[test]
    fn prompt_hash_session_id_requires_cutoff_and_four_complete_blocks() {
        let tokens: Vec<u32> = (0..20).collect();
        assert_eq!(prompt_hash_session_id("model", &tokens, 4, 24), None);
        assert_eq!(prompt_hash_session_id("model", &tokens, 4, 15), None);
        assert_eq!(prompt_hash_session_id("model", &tokens, 0, 16), None);
    }

    #[test]
    fn client_session_id_uses_deterministic_fallback_precedence() {
        let values = std::collections::HashMap::from([
            (headers::CODEX_SESSION_ID, "codex"),
            (headers::CLAUDE_CODE_SESSION_ID, "claude"),
            (headers::CLAUDE_CODE_AGENT_ID, "claude-agent"),
            (headers::USER_ID, "generic-user"),
        ]);
        assert_eq!(
            client_session_id(|name| values.get(name).copied()).as_deref(),
            Some("codex")
        );

        let canonical = std::collections::HashMap::from([
            (headers::SESSION_ID, "gwp"),
            (headers::SESSION_AFFINITY, "affinity"),
            (headers::CODEX_SESSION_ID, "codex"),
        ]);
        assert_eq!(
            client_session_id(|name| canonical.get(name).copied()).as_deref(),
            Some("gwp")
        );

        let affinity = std::collections::HashMap::from([
            (headers::SESSION_AFFINITY, "affinity"),
            (headers::CODEX_SESSION_ID, "codex"),
        ]);
        assert_eq!(
            client_session_id(|name| affinity.get(name).copied()).as_deref(),
            Some("affinity")
        );

        let empty_then_parent = std::collections::HashMap::from([
            (headers::SESSION_ID, ""),
            (headers::PARENT_SESSION_ID, "parent"),
        ]);
        assert_eq!(
            client_session_id(|name| empty_then_parent.get(name).copied()).as_deref(),
            Some("parent")
        );

        let user_id_only = std::collections::HashMap::from([(headers::USER_ID, "generic-user")]);
        assert_eq!(
            client_session_id(|name| user_id_only.get(name).copied()).as_deref(),
            Some("generic-user")
        );
    }

    fn worker(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::from_worker_id(id)
    }

    #[test]
    fn served_worker_header_matches_frontend_contract() {
        assert_eq!(headers::SERVED_WORKER, "x-baseten-dyn-worker-id");
    }

    #[tokio::test]
    async fn in_memory_roundtrip() {
        let store = InMemoryAffinityStore::new();
        assert_eq!(store.peek("s1").await.unwrap(), None);

        store
            .put("s1", worker(7), Duration::from_secs(60))
            .await
            .unwrap();
        assert_eq!(store.peek("s1").await.unwrap(), Some(worker(7)));
        assert_eq!(store.get("s1").await.unwrap(), Some(worker(7)));

        // A later successful request refreshes/overwrites the worker binding.
        store
            .put("s1", worker(8), Duration::from_secs(60))
            .await
            .unwrap();
        assert_eq!(store.peek("s1").await.unwrap(), Some(worker(8)));

        let endpoint_binding =
            AffinityBinding::with_endpoint(worker(9), EndpointId("cluster-a".into()));
        store
            .put_binding("s1", endpoint_binding.clone(), Duration::from_secs(60))
            .await
            .unwrap();
        assert_eq!(
            store.peek_binding("s1").await.unwrap(),
            Some(endpoint_binding)
        );

        assert!(store.remove("s1").await.unwrap());
        assert!(!store.remove("s1").await.unwrap());
        assert_eq!(store.peek("s1").await.unwrap(), None);
    }

    #[tokio::test(start_paused = true)]
    async fn in_memory_ttl_expiry_and_peek_does_not_refresh() {
        let store = InMemoryAffinityStore::new();
        store
            .put("s1", worker(1), Duration::from_secs(10))
            .await
            .unwrap();

        // peek at t=9s must not extend the TTL...
        tokio::time::advance(Duration::from_secs(9)).await;
        assert_eq!(store.peek("s1").await.unwrap(), Some(worker(1)));
        // ...so at t=11s the binding is gone.
        tokio::time::advance(Duration::from_secs(2)).await;
        assert_eq!(store.peek("s1").await.unwrap(), None);
    }

    #[tokio::test(start_paused = true)]
    async fn in_memory_get_refreshes_ttl() {
        let store = InMemoryAffinityStore::new();
        store
            .put("s1", worker(1), Duration::from_secs(10))
            .await
            .unwrap();

        tokio::time::advance(Duration::from_secs(9)).await;
        assert_eq!(store.get("s1").await.unwrap(), Some(worker(1))); // refresh at t=9

        tokio::time::advance(Duration::from_secs(9)).await; // t=18 < 9+10
        assert_eq!(store.peek("s1").await.unwrap(), Some(worker(1)));

        tokio::time::advance(Duration::from_secs(2)).await; // t=20 > 19
        assert_eq!(store.peek("s1").await.unwrap(), None);
    }

    #[test]
    fn minted_ids_are_unique_and_prefixed() {
        let a = mint_session_id();
        let b = mint_session_id();
        assert_ne!(a, b);
        assert!(a.starts_with("base10-"));
    }
}
