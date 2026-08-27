// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::logging::{self, DistributedTraceContext};
use dynamo_runtime::pipeline::AsyncEngineContext;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Language-neutral request state needed by the B10 admission lifecycle.
#[derive(Clone)]
pub struct RequestContext {
    inner: Arc<dyn AsyncEngineContext>,
    trace_context: Option<DistributedTraceContext>,
    metadata: BTreeMap<String, String>,
}

impl RequestContext {
    pub fn new(
        inner: Arc<dyn AsyncEngineContext>,
        trace_context: Option<DistributedTraceContext>,
        metadata: BTreeMap<String, String>,
    ) -> Self {
        Self {
            inner,
            trace_context,
            metadata,
        }
    }

    pub fn id(&self) -> &str {
        self.inner.id()
    }

    pub fn inner(&self) -> Arc<dyn AsyncEngineContext> {
        Arc::clone(&self.inner)
    }

    pub fn trace_context(&self) -> Option<&DistributedTraceContext> {
        self.trace_context.as_ref()
    }

    pub fn metadata_snapshot(&self) -> BTreeMap<String, String> {
        self.metadata.clone()
    }

    pub(crate) fn direct_span(&self, operation: &str, instance_id: u64) -> tracing::Span {
        logging::make_client_request_span(
            operation,
            self.id(),
            self.trace_context(),
            Some(&instance_id.to_string()),
        )
    }
}
