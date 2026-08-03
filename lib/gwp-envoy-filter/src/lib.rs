// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Envoy Proxy-Wasm lifecycle adapter for the GWP gRPC control API.
//!
//! Envoy's native gRPC `ext_authz` filter owns body buffering, scheduling, and
//! typed request-header mutation. This module observes the resulting upstream
//! response and stream teardown. Lifecycle notifications are handed to a
//! VM-root shared queue so neither response headers nor stream completion wait
//! for the GWP control-plane responses.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;

use prost::Message;
use proxy_wasm::hostcalls;
use proxy_wasm::traits::{Context, HttpContext, RootContext};
use proxy_wasm::types::{Action, ContextType, LogLevel, MetricType};

mod pb {
    include!(concat!(env!("OUT_DIR"), "/dynamo.gwp.v1.rs"));
}

use pb::queued_lifecycle_event::Event;
use pb::{QueuedLifecycleEvent, RequestFinishedRequest, ResponseStartedRequest};

const LIFECYCLE_CLUSTER: &str = "gwp_lifecycle_control";
const LIFECYCLE_SERVICE: &str = "dynamo.gwp.v1.Lifecycle";
const LIFECYCLE_RPC_TIMEOUT: Duration = Duration::from_secs(5);

const REQUEST_HEADERS: &[&str] = &[
    "accept",
    "authorization",
    "content-length",
    "content-type",
    "host",
    "user-agent",
    "x-baseten-model-apis-routing-requirements",
    // Forwarded only when GWP found a live affinity-store binding. This tells
    // the selected deployment which worker served the session previously.
    "x-gwp-affine-worker-id",
    "x-gwp-authority",
    // Envoy synthesizes this before the filter chain. Dynamic forward proxy
    // route reconstruction requires it after ext-authz clears the route cache.
    "x-forwarded-proto",
    "x-request-id",
];

#[derive(Default)]
struct Metrics {
    lifecycle_dispatched: Option<u32>,
    lifecycle_dispatch_failure: Option<u32>,
    lifecycle_response_failure: Option<u32>,
}

impl Metrics {
    fn define() -> Self {
        Self {
            lifecycle_dispatched: metric(MetricType::Counter, "gwp_wasm.lifecycle_dispatched"),
            lifecycle_dispatch_failure: metric(
                MetricType::Counter,
                "gwp_wasm.lifecycle_dispatch_failure",
            ),
            lifecycle_response_failure: metric(
                MetricType::Counter,
                "gwp_wasm.lifecycle_response_failure",
            ),
        }
    }

    fn increment(&self, id: Option<u32>) {
        if let Some(id) = id {
            let _ = hostcalls::increment_metric(id, 1);
        }
    }
}

fn metric(kind: MetricType, name: &str) -> Option<u32> {
    match hostcalls::define_metric(kind, name) {
        Ok(id) => Some(id),
        Err(error) => {
            log::error!("failed to define {name}: {error:?}");
            None
        }
    }
}

struct GwpRoot {
    queue_id: Option<u32>,
    metrics: Rc<RefCell<Metrics>>,
}

impl GwpRoot {
    fn new() -> Self {
        Self {
            queue_id: None,
            metrics: Rc::new(RefCell::new(Metrics::default())),
        }
    }

    fn dispatch_lifecycle(&self, event: QueuedLifecycleEvent) {
        let Some((method, payload)) = event.event.map(|event| match event {
            Event::ResponseStarted(request) => ("ResponseStarted", request.encode_to_vec()),
            Event::RequestFinished(request) => ("RequestFinished", request.encode_to_vec()),
        }) else {
            let metrics = self.metrics.borrow();
            metrics.increment(metrics.lifecycle_dispatch_failure);
            return;
        };

        match self.dispatch_grpc_call(
            LIFECYCLE_CLUSTER,
            LIFECYCLE_SERVICE,
            method,
            vec![],
            Some(&payload),
            LIFECYCLE_RPC_TIMEOUT,
        ) {
            Ok(_) => {
                let metrics = self.metrics.borrow();
                metrics.increment(metrics.lifecycle_dispatched);
            }
            Err(error) => {
                log::debug!("lifecycle dispatch failed: {error:?}");
                let metrics = self.metrics.borrow();
                metrics.increment(metrics.lifecycle_dispatch_failure);
            }
        }
    }
}

impl Context for GwpRoot {
    fn on_grpc_call_response(&mut self, _token_id: u32, status_code: u32, _response_size: usize) {
        if status_code != 0 {
            log::debug!("lifecycle gRPC response failed with status {status_code}");
            let metrics = self.metrics.borrow();
            metrics.increment(metrics.lifecycle_response_failure);
        }
    }
}

impl RootContext for GwpRoot {
    fn on_vm_start(&mut self, _configuration_size: usize) -> bool {
        *self.metrics.borrow_mut() = Metrics::define();
        self.queue_id = Some(self.register_shared_queue("gwp_lifecycle"));
        true
    }

    fn on_queue_ready(&mut self, queue_id: u32) {
        if self.queue_id != Some(queue_id) {
            return;
        }
        loop {
            match self.dequeue_shared_queue(queue_id) {
                Ok(Some(bytes)) => match QueuedLifecycleEvent::decode(bytes.as_slice()) {
                    Ok(event) => self.dispatch_lifecycle(event),
                    Err(error) => log::error!("invalid lifecycle queue event: {error}"),
                },
                Ok(None) => break,
                Err(error) => {
                    log::debug!("lifecycle queue read failed: {error:?}");
                    break;
                }
            }
        }
    }

    fn get_type(&self) -> Option<ContextType> {
        Some(ContextType::HttpContext)
    }

    fn create_http_context(&self, _context_id: u32) -> Option<Box<dyn HttpContext>> {
        Some(Box::new(LifecycleHttp::new(
            self.queue_id,
            Rc::clone(&self.metrics),
        )))
    }
}

struct LifecycleHttp {
    lifecycle_queue_id: Option<u32>,
    metrics: Rc<RefCell<Metrics>>,
    request_id: Option<String>,
    endpoint_id: String,
    session_id: String,
    scheduled: bool,
    response_started_enqueued: bool,
    finish_enqueued: bool,
}

impl LifecycleHttp {
    fn new(lifecycle_queue_id: Option<u32>, metrics: Rc<RefCell<Metrics>>) -> Self {
        Self {
            lifecycle_queue_id,
            metrics,
            request_id: None,
            endpoint_id: String::new(),
            session_id: String::new(),
            scheduled: false,
            response_started_enqueued: false,
            finish_enqueued: false,
        }
    }

    fn enqueue(&self, event: QueuedLifecycleEvent) {
        let Some(queue_id) = self.lifecycle_queue_id else {
            let metrics = self.metrics.borrow();
            metrics.increment(metrics.lifecycle_dispatch_failure);
            return;
        };
        let payload = event.encode_to_vec();
        if let Err(error) = self.enqueue_shared_queue(queue_id, Some(&payload)) {
            log::debug!("lifecycle queue write failed: {error:?}");
            let metrics = self.metrics.borrow();
            metrics.increment(metrics.lifecycle_dispatch_failure);
        }
    }

    fn upsert_response_header(&self, name: &str, value: &str) {
        if self.get_http_response_header(name).is_some() {
            self.set_http_response_header(name, Some(value));
        } else {
            self.add_http_response_header(name, value);
        }
    }

    fn set_access_log_field(&self, key: &str, value: &str) {
        let path = format!("dynamo.gwp.{key}");
        self.set_property(vec![&path], Some(value.as_bytes()));
    }
}

impl Context for LifecycleHttp {}

impl HttpContext for LifecycleHttp {
    fn on_http_request_headers(&mut self, _num_headers: usize, _end_of_stream: bool) -> Action {
        let Some(request_id) = self.get_http_request_header("x-gwp-internal-request-id") else {
            return Action::Continue;
        };
        self.endpoint_id = self
            .get_http_request_header("x-gwp-endpoint-id")
            .unwrap_or_default();
        self.session_id = self
            .get_http_request_header("x-gwp-session-id")
            .unwrap_or_default();
        let model = self
            .get_http_request_header("x-gwp-model")
            .unwrap_or_default();
        let sticky = self
            .get_http_request_header("x-gwp-sticky")
            .unwrap_or_default();
        self.request_id = Some(request_id);
        self.scheduled = true;
        for (name, _) in self.get_http_request_headers() {
            let lower = name.to_ascii_lowercase();
            if !name.starts_with(':') && !REQUEST_HEADERS.contains(&lower.as_str()) {
                self.remove_http_request_header(&name);
            }
        }
        self.set_access_log_field("scheduled", "true");
        self.set_access_log_field("endpoint_id", &self.endpoint_id);
        self.set_access_log_field("model", &model);
        self.set_access_log_field("sticky", &sticky);
        Action::Continue
    }

    fn on_http_response_headers(&mut self, _num_headers: usize, _end_of_stream: bool) -> Action {
        if !self.scheduled || self.response_started_enqueued {
            return Action::Continue;
        }
        self.response_started_enqueued = true;
        let worker_id = self.get_http_response_header("x-baseten-dyn-worker-id");
        let status = self
            .get_http_response_header(":status")
            .unwrap_or_else(|| "200".to_string());
        if !self.session_id.is_empty() {
            self.upsert_response_header("x-session-id", &self.session_id);
        }
        if !self.endpoint_id.is_empty() {
            self.upsert_response_header("x-routed-endpoint", &self.endpoint_id);
        }
        if let Some(request_id) = self.request_id.clone() {
            let actual_worker_id = match worker_id {
                Some(value) => match value.parse() {
                    Ok(worker_id) => Some(worker_id),
                    Err(_) => {
                        log::debug!("invalid x-baseten-dyn-worker-id response header");
                        let metrics = self.metrics.borrow();
                        metrics.increment(metrics.lifecycle_dispatch_failure);
                        return Action::Continue;
                    }
                },
                None => None,
            };
            self.enqueue(QueuedLifecycleEvent {
                event: Some(Event::ResponseStarted(ResponseStartedRequest {
                    request_id,
                    actual_worker_id,
                    upstream_status: status.parse().unwrap_or(200),
                })),
            });
        }
        Action::Continue
    }

    fn on_log(&mut self) {
        if self.finish_enqueued || !self.scheduled {
            return;
        }
        let Some(request_id) = self.request_id.clone() else {
            return;
        };
        self.finish_enqueued = true;
        self.enqueue(QueuedLifecycleEvent {
            event: Some(Event::RequestFinished(RequestFinishedRequest {
                request_id,
                finish_reason: Some("complete".to_string()),
            })),
        });
    }
}

proxy_wasm::main! {{
    proxy_wasm::set_log_level(LogLevel::Info);
    proxy_wasm::set_root_context(|_| -> Box<dyn RootContext> { Box::new(GwpRoot::new()) });
}}
