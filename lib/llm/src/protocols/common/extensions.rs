// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::HeaderMap;
use std::sync::Arc;

use crate::http::service::baseten::baseten_preferred_session_affinity_from_headers;
use dynamo_runtime::pipeline::Context;

pub const SESSION_AFFINITY_CONTEXT_KEY: &str = "dynamo.llm.session_affinity";
pub const HEADER_DYNAMO_SESSION_ID: &str = "x-dynamo-session-id";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionAffinityId(String);

impl SessionAffinityId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

pub fn session_affinity_from_headers(headers: &HeaderMap) -> Option<SessionAffinityId> {
    baseten_preferred_session_affinity_from_headers(headers).map(SessionAffinityId::new)
}

/// Attach affinity to both forms of request context state. The typed registry
/// is used inside a process; metadata survives request-plane hops, including
/// the Python standalone-router client.
pub(crate) fn insert_session_affinity<T: Send + Sync + 'static>(
    context: &mut Context<T>,
    session_id: impl Into<String>,
) {
    let session_id = session_id.into();
    context.insert_metadata(SESSION_AFFINITY_CONTEXT_KEY, session_id.clone());
    context.insert(
        SESSION_AFFINITY_CONTEXT_KEY,
        SessionAffinityId::new(session_id),
    );
}

/// Read affinity from the typed registry, falling back to propagated context
/// metadata after a request-plane hop.
pub(crate) fn session_affinity_from_context<T: Send + Sync + 'static>(
    context: &Context<T>,
) -> Result<Option<Arc<SessionAffinityId>>, String> {
    if let Some(session_id) =
        context.get_optional::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)?
    {
        return Ok(Some(session_id));
    }
    Ok(context
        .metadata()
        .get(SESSION_AFFINITY_CONTEXT_KEY)
        .filter(|value| !value.is_empty())
        .cloned()
        .map(SessionAffinityId::new)
        .map(Arc::new))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requires_nonempty_explicit_header() {
        let mut headers = HeaderMap::new();
        assert!(session_affinity_from_headers(&headers).is_none());
        headers.insert(HEADER_DYNAMO_SESSION_ID, "session-123".parse().unwrap());
        assert_eq!(
            session_affinity_from_headers(&headers).unwrap().as_str(),
            "session-123"
        );
    }

    #[test]
    fn affinity_survives_context_metadata_only_hop() {
        let mut source = Context::new(());
        insert_session_affinity(&mut source, "session-123");

        let forwarded =
            Context::with_id_and_metadata((), "request-2".to_string(), source.metadata().clone());
        assert_eq!(
            session_affinity_from_context(&forwarded)
                .unwrap()
                .unwrap()
                .as_str(),
            "session-123"
        );
    }
}
