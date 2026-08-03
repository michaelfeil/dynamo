// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Global Workload Plane (GWP).
//!
//! A multi-replica, session-aware reverse proxy that routes OpenAI-style
//! requests across routable endpoint deployments by reusing
//! `dynamo-kv-router`'s approximate
//! indexer and selector primitives. See `docs/design-docs/global-workload-plane-design.md`.
//!
//! Two routing tiers:
//! 1. Session tier - `x-session-id`, a recognized client-native session
//!    header, the OpenAI `user` field, or a configured long-prompt prefix
//!    fingerprint is forwarded to the endpoint selected for the previous turn
//!    (looked up via
//!    [`session::AffinityStore`], honored iff the endpoint is alive and
//!    model-eligible).
//! 2. Approximate tier - new/unstuck sessions are pseudo-tokenized via
//!    [`tokens`], scored by a [`router::GwpRouter`] (a `KvRouter` with etcd
//!    discovery and ZMQ replica sync), and fed the workers observed behind
//!    each routable endpoint by the
//!    [`reflector::Reflector`]'s 1s planner poll, and the resolved session id
//!    is returned in the response headers.
//!
//! The production control plane is served over gRPC by [`server`] (feature
//! `server`; binary `dynamo-gwp`). Multi-replica session affinity uses etcd
//! (feature `etcd`) and treats backend errors as affinity misses.

pub mod config;
pub mod control;
pub mod core;
#[cfg(feature = "server")]
pub mod grpc;
pub mod identity;
pub mod lifecycle;
pub mod metrics;
pub mod models;
pub mod reflector;
pub mod router;
#[cfg(feature = "server")]
pub mod server;
pub mod session;
pub mod tokens;
mod worker_selector;

pub use config::{
    ConfigStore, EndpointConfig, EndpointId, GwpConfig, LoadBalancingPolicy, ModelRoute,
    ModelTokenizationConfig, RoutingConfig, SessionConfig, TokenizationConfig,
};
pub use identity::EndpointTable;
pub use router::{GwpRouter, WorkerConfigSender};
