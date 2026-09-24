// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::protocols::common::extensions::SESSION_AFFINITY_CONTEXT_KEY;
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

    /// Log the resolved session affinity ID and the configured metadata values.
    /// Joins the frontend request summary on `b10_request_id`.
    pub fn log_request_metadata(&self, request_metadata_keys: &[String]) {
        let Some((session_id, request_metadata)) =
            selected_request_metadata(&self.metadata, request_metadata_keys)
        else {
            return;
        };
        // The JSONL layer parses a JSON string field into an object, so Loki's `| json`
        // flattens it to `request_metadata_<key>`. String IDs use Display so that a
        // numeric-looking value stays a string.
        let request_metadata = (!request_metadata.is_empty()).then(|| {
            serde_json::to_string(&request_metadata).expect("string map serializes to JSON")
        });
        let b10_request_id = b10_request_id(self.id());
        tracing::info!(
            request_id = %self.id(),
            b10_request_id = b10_request_id.map(tracing::field::display),
            session_id = session_id.map(tracing::field::display),
            request_metadata = request_metadata.as_deref(),
            "Request metadata"
        );
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

/// Session affinity ID and the non-empty values of the configured metadata keys, or
/// `None` when neither is present. Key selection only controls extra metadata.
fn selected_request_metadata<'a>(
    metadata: &'a BTreeMap<String, String>,
    keys: &'a [String],
) -> Option<(Option<&'a str>, BTreeMap<&'a str, &'a str>)> {
    let nonempty = |key: &str| {
        metadata
            .get(key)
            .map(String::as_str)
            .filter(|v| !v.is_empty())
    };
    let selected = keys
        .iter()
        .filter_map(|key| Some((key.as_str(), nonempty(key)?)))
        .collect::<BTreeMap<_, _>>();
    let session_id = nonempty(SESSION_AFFINITY_CONTEXT_KEY);
    (session_id.is_some() || !selected.is_empty()).then_some((session_id, selected))
}

/// The request ID segment of a Baseten `{org}--{request}--{model version}[--extras]` context ID.
fn b10_request_id(context_id: &str) -> Option<&str> {
    let mut parts = context_id.split("--");
    let request_id = parts.nth(1)?;
    parts.next()?;
    Some(request_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_session_affinity_and_configured_nonempty_keys() {
        let mut metadata: BTreeMap<String, String> = [
            (SESSION_AFFINITY_CONTEXT_KEY, "session-1"),
            ("x-example-agent-id", "agent-1"),
            ("x-example-workload-id", ""),
            ("x-example-unselected", "nope"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        let keys = [
            "x-example-agent-id",
            "x-example-workload-id",
            "x-example-missing",
        ]
        .map(String::from);
        assert_eq!(
            selected_request_metadata(&metadata, &keys),
            Some((
                Some("session-1"),
                BTreeMap::from([("x-example-agent-id", "agent-1")])
            ))
        );

        for selected_keys in [&keys[2..], &[]] {
            assert_eq!(
                selected_request_metadata(&metadata, selected_keys),
                Some((Some("session-1"), BTreeMap::new()))
            );
        }
        metadata.remove(SESSION_AFFINITY_CONTEXT_KEY);
        assert_eq!(
            selected_request_metadata(&metadata, &keys),
            Some((None, BTreeMap::from([("x-example-agent-id", "agent-1")])))
        );
        assert_eq!(selected_request_metadata(&metadata, &[]), None);
        assert_eq!(selected_request_metadata(&BTreeMap::new(), &keys), None);
    }

    #[test]
    fn parses_b10_request_id_from_context_id() {
        assert_eq!(b10_request_id("org-a--req-1--ver"), Some("req-1"));
        assert_eq!(b10_request_id("org-a--req-1--ver--ray:user"), Some("req-1"));
        assert_eq!(b10_request_id("org-a--req-1"), None);
        assert_eq!(b10_request_id("plain-uuid"), None);
    }
}
