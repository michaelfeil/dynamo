// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared Baseten HTTP and request-context contracts.

use axum::http::HeaderMap;
use axum::response::Response;
use serde::Deserialize;
use std::{
    collections::{BTreeMap, HashMap},
    sync::{LazyLock, Mutex},
};

pub const X_BASETEN_DYN_WORKER_ID_HEADER: &str = "x-baseten-dyn-worker-id";
pub const X_BASETEN_DYN_PREFILL_WORKER_ID_HEADER: &str = "x-baseten-dyn-prefill-worker-id";
pub const X_BASETEN_DYN_PREFILL_DP_RANK_HEADER: &str = "x-baseten-dyn-prefill-dp-rank";
pub const X_BASETEN_DYN_DECODE_DP_RANK_HEADER: &str = "x-baseten-dyn-decode-dp-rank";
const BASETEN_PREFERRED_SESSION_AFFINITY_HEADERS: &[&str] = &[
    "x-baseten-session-id",
    "x-baseten-session",
    "x-dynamo-session-id",
    "x-session-id",
    "x-session-affinity",
    "session-id",
    "x-claude-code-session-id",
    "x-parent-session-id",
    "x-claude-code-agent-id",
    "x-claude-code-parent-agent-id",
];

pub(crate) fn baseten_preferred_session_affinity_from_headers(headers: &HeaderMap) -> Option<&str> {
    BASETEN_PREFERRED_SESSION_AFFINITY_HEADERS
        .iter()
        .find_map(|name| nonempty_header(headers, name))
}

pub(crate) fn baseten_session_affinity_from_request(
    headers: &HeaderMap,
    body_session_id: Option<&str>,
) -> Option<String> {
    baseten_preferred_session_affinity_from_headers(headers)
        .map(str::to_owned)
        .or_else(|| {
            body_session_id
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerResponseMetadata {
    pub prefill_worker_id: Option<u64>,
    pub prefill_dp_rank: Option<u32>,
    pub decode_worker_id: Option<u64>,
    pub decode_dp_rank: Option<u32>,
}

impl WorkerResponseMetadata {
    pub fn from_metadata(metadata: &BTreeMap<String, String>) -> Option<Self> {
        let parse_u64 = |key: RoutingMetadataKey| {
            metadata
                .get(key.as_str())
                .and_then(|value| value.parse::<u64>().ok())
        };
        let parse_u32 = |key: RoutingMetadataKey| {
            metadata
                .get(key.as_str())
                .and_then(|value| value.parse::<u32>().ok())
        };
        let worker = Self {
            prefill_worker_id: parse_u64(RoutingMetadataKey::PrefillWorkerId),
            prefill_dp_rank: parse_u32(RoutingMetadataKey::PrefillDpRank),
            decode_worker_id: parse_u64(RoutingMetadataKey::DecodeWorkerId),
            decode_dp_rank: parse_u32(RoutingMetadataKey::DecodeDpRank),
        };
        (worker.prefill_worker_id.is_some() || worker.decode_worker_id.is_some()).then_some(worker)
    }
}

// The Python HTTP engine has already awaited its first item when this is
// published. This narrow side channel lets the handler set headers without
// polling the response stream again or changing the runtime context trait.
static WORKER_RESPONSE_METADATA: LazyLock<Mutex<HashMap<String, WorkerResponseMetadata>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub fn publish_worker_response_metadata(context_id: String, metadata: WorkerResponseMetadata) {
    WORKER_RESPONSE_METADATA
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(context_id.clone(), metadata);
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        take_worker_response_metadata(&context_id);
    });
}

pub fn take_worker_response_metadata(context_id: &str) -> Option<WorkerResponseMetadata> {
    WORKER_RESPONSE_METADATA
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .remove(context_id)
}

pub fn apply_worker_response_headers(headers: &mut HeaderMap, worker: WorkerResponseMetadata) {
    insert_numeric_header(
        headers,
        X_BASETEN_DYN_WORKER_ID_HEADER,
        worker.decode_worker_id.or(worker.prefill_worker_id),
    );
    insert_numeric_header(
        headers,
        X_BASETEN_DYN_PREFILL_WORKER_ID_HEADER,
        worker.prefill_worker_id,
    );
    insert_numeric_header(
        headers,
        X_BASETEN_DYN_PREFILL_DP_RANK_HEADER,
        worker.prefill_dp_rank,
    );
    insert_numeric_header(
        headers,
        X_BASETEN_DYN_DECODE_DP_RANK_HEADER,
        worker.decode_dp_rank,
    );
}

pub fn attach_worker_response_headers(
    mut response: Response,
    worker: Option<WorkerResponseMetadata>,
) -> Response {
    if let Some(worker) = worker {
        apply_worker_response_headers(response.headers_mut(), worker);
    }
    response
}

fn insert_numeric_header<T: std::fmt::Display>(
    headers: &mut HeaderMap,
    header_name: &'static str,
    value: Option<T>,
) {
    let Some(value) = value else {
        return;
    };
    headers.insert(
        header_name,
        value
            .to_string()
            .parse()
            .expect("numeric routing metadata is a valid HTTP header value"),
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingMetadataKey {
    PrefillWorkerId,
    PrefillDpRank,
    DecodeWorkerId,
    DecodeDpRank,
}

impl RoutingMetadataKey {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PrefillWorkerId => "dynamo.routing.prefill_worker_id",
            Self::PrefillDpRank => "dynamo.routing.prefill_dp_rank",
            Self::DecodeWorkerId => "dynamo.routing.decode_worker_id",
            Self::DecodeDpRank => "dynamo.routing.decode_dp_rank",
        }
    }
}

/// JSON body of the `X-Baseten-Customer-Request-Context` header set by SEG.
/// Field order here is also the order they are appended to the extras segment.
#[derive(Debug, Default, Deserialize)]
struct CustomerRequestContext {
    #[serde(default)]
    cf_ray: String,
    #[serde(default)]
    user_id: String,
}

/// Build the context ID: `{org_namespace}--{b10_request_id}--{model_version_id}[--{extras}]`.
///
/// - `org_namespace`: `X-Baseten-Org-Namespace`, fallback `X-Baseten-Billing-Org-Id` (else "none").
/// - `b10_request_id`: `X-Baseten-Request-Id`, truncated at the first `:` (SEG may
///   append a `:cf-ray:user-id` suffix); a UUID if absent.
/// - `model_version_id`: `X-Baseten-Model-APIs-Version-Id`, fallback
///   `X-Baseten-Model-Version-ID` (else "none").
/// - `extras` (optional): non-empty `cf_ray`/`user_id` from
///   `X-Baseten-Customer-Request-Context` joined with `:`, appended as a 4th
///   segment only when present. Opaque/pass-through; sanitized so it has no `--`.
///
/// Consumers split on `--` into 3 or 4 parts (the 4th, `extras`, is optional).
/// org/request/model_version are assumed `--`-free.
pub(crate) fn get_or_create_context_id(headers: &HeaderMap) -> String {
    // Prefer the org-namespace header; fall back to the legacy billing-org header
    // (beefeater sets that on the direct BIS route).
    let org_namespace = nonempty_header(headers, "X-Baseten-Org-Namespace")
        .or_else(|| nonempty_header(headers, "X-Baseten-Billing-Org-Id"))
        .unwrap_or("none");

    // Prefer the model-apis header; fall back to the legacy model-version header.
    let model_version_id = nonempty_header(headers, "X-Baseten-Model-APIs-Version-Id")
        .or_else(|| nonempty_header(headers, "X-Baseten-Model-Version-ID"))
        .unwrap_or("none");

    let raw_request_id = nonempty_header(headers, "X-Baseten-Request-Id")
        .map(|s| s.to_owned())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    // Keep only the head before any `:` suffix; extras come from the JSON header.
    let b10_request_id = match raw_request_id.find(':') {
        Some(pos) => &raw_request_id[..pos],
        None => raw_request_id.as_str(),
    };

    let extras = parse_customer_request_context(headers);
    // Untrusted input: collapse any `--` so a value can't forge the segment
    // separator. `:` is allowed — extras is the opaque trailing segment.
    let extras_joined: Vec<String> = [extras.cf_ray.as_str(), extras.user_id.as_str()]
        .into_iter()
        .filter(|s| !s.is_empty())
        .map(collapse_double_dash)
        .collect();

    let mut context_id = format!("{org_namespace}--{b10_request_id}--{model_version_id}");
    if !extras_joined.is_empty() {
        context_id.push_str("--");
        context_id.push_str(&extras_joined.join(":"));
    }
    context_id
}

/// Header value as `&str`, treating a present-but-empty value as absent so the
/// fallback chains (and the UUID default) fire on empty headers too.
fn nonempty_header<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
}

/// Collapse any run of 2+ `-` into a single `-` so an untrusted value can't forge
/// the `--` segment separator. Other characters (including `:`) pass through.
fn collapse_double_dash(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_dash = false;
    for c in s.chars() {
        let is_dash = c == '-';
        if !(is_dash && prev_dash) {
            out.push(c);
        }
        prev_dash = is_dash;
    }
    out
}

fn parse_customer_request_context(headers: &HeaderMap) -> CustomerRequestContext {
    let Some(raw) = nonempty_header(headers, "X-Baseten-Customer-Request-Context") else {
        return CustomerRequestContext::default();
    };
    serde_json::from_str(raw).unwrap_or_else(|err| {
        tracing::warn!(
            error = %err,
            "failed to parse X-Baseten-Customer-Request-Context as JSON; dropping extras"
        );
        CustomerRequestContext::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::extensions::session_affinity_from_headers;

    fn headers_with(pairs: &[(&str, &str)]) -> HeaderMap {
        use axum::http::HeaderName;
        let mut headers = HeaderMap::new();
        for (k, v) in pairs {
            let name: HeaderName = k.parse().unwrap();
            headers.insert(name, v.parse().unwrap());
        }
        headers
    }

    #[test]
    fn full_id_with_all_parts() {
        // Extras emitted in fixed cf_ray:user_id order regardless of JSON key order.
        let headers = headers_with(&[
            ("X-Baseten-Org-Namespace", "my-org"),
            ("X-Baseten-Request-Id", "abc123"),
            ("X-Baseten-Model-APIs-Version-Id", "mv-789"),
            (
                "X-Baseten-Customer-Request-Context",
                r#"{"user_id":"chatcmpl-abc","cf_ray":"ray-1"}"#,
            ),
        ]);
        assert_eq!(
            get_or_create_context_id(&headers),
            "my-org--abc123--mv-789--ray-1:chatcmpl-abc"
        );
    }

    #[test]
    fn session_header_aliases_and_precedence() {
        let pairs = BASETEN_PREFERRED_SESSION_AFFINITY_HEADERS
            .iter()
            .map(|name| (*name, *name))
            .collect::<Vec<_>>();
        let mut headers = headers_with(&pairs);
        for expected in BASETEN_PREFERRED_SESSION_AFFINITY_HEADERS {
            assert_eq!(
                baseten_preferred_session_affinity_from_headers(&headers),
                Some(*expected)
            );
            headers.remove(*expected);
        }
        assert!(session_affinity_from_headers(&headers).is_none());
    }

    #[test]
    fn session_body_fallback_is_normalized_and_headers_win() {
        assert_eq!(
            baseten_session_affinity_from_request(&HeaderMap::new(), Some(" body-session ")),
            Some("body-session".to_string())
        );
        assert_eq!(
            baseten_session_affinity_from_request(&HeaderMap::new(), Some("  ")),
            None
        );

        let headers = headers_with(&[("x-baseten-session-id", "header-session")]);
        assert_eq!(
            baseten_session_affinity_from_request(&headers, Some("body-session")),
            Some("header-session".to_string())
        );
    }

    #[test]
    fn defaults_to_none_and_generates_uuid() {
        let only_id = headers_with(&[("X-Baseten-Request-Id", "abc123")]);
        assert_eq!(get_or_create_context_id(&only_id), "none--abc123--none");

        // No request id either: org="none", UUID middle, model_version="none".
        let generated = get_or_create_context_id(&HeaderMap::new());
        assert!(generated.starts_with("none--"), "got: {generated}");
        assert!(generated.ends_with("--none"), "got: {generated}");
        assert!(generated.len() > "none----none".len());
    }

    #[test]
    fn request_id_truncated_at_first_colon() {
        // SEG may append a `:cf-ray:user-id` suffix; only the head is kept.
        let headers = headers_with(&[("X-Baseten-Request-Id", "abc123:legacy-ray:legacy-user")]);
        assert_eq!(get_or_create_context_id(&headers), "none--abc123--none");
    }

    #[test]
    fn empty_or_absent_org_falls_back_to_billing_org() {
        // A present-but-empty X-Baseten-Org-Namespace is treated as absent and
        // falls back to the billing-org header (beefeater's direct BIS route).
        let headers = headers_with(&[
            ("X-Baseten-Org-Namespace", ""),
            ("X-Baseten-Request-Id", "abc123"),
            ("X-Baseten-Billing-Org-Id", "billing-org"),
        ]);
        assert_eq!(
            get_or_create_context_id(&headers),
            "billing-org--abc123--none"
        );
    }

    #[test]
    fn model_version_prefers_apis_over_legacy() {
        let legacy_only = headers_with(&[
            ("X-Baseten-Request-Id", "abc123"),
            ("X-Baseten-Model-Version-ID", "legacy-mv"),
        ]);
        assert_eq!(
            get_or_create_context_id(&legacy_only),
            "none--abc123--legacy-mv"
        );

        let both = headers_with(&[
            ("X-Baseten-Request-Id", "abc123"),
            ("X-Baseten-Model-APIs-Version-Id", "mv-new"),
            ("X-Baseten-Model-Version-ID", "legacy-mv"),
        ]);
        assert_eq!(get_or_create_context_id(&both), "none--abc123--mv-new");
    }

    #[test]
    fn partial_or_malformed_extras() {
        // Only cf_ray present → user_id omitted; extras is the optional 4th segment.
        let partial = headers_with(&[
            ("X-Baseten-Request-Id", "abc123"),
            (
                "X-Baseten-Customer-Request-Context",
                r#"{"cf_ray":"ray-1"}"#,
            ),
        ]);
        assert_eq!(
            get_or_create_context_id(&partial),
            "none--abc123--none--ray-1"
        );

        // Malformed JSON → extras dropped, no panic, 3-part id.
        let malformed = headers_with(&[
            ("X-Baseten-Request-Id", "abc123"),
            ("X-Baseten-Customer-Request-Context", "not-json"),
        ]);
        assert_eq!(get_or_create_context_id(&malformed), "none--abc123--none");
    }

    #[test]
    fn double_dash_in_extras_collapsed() {
        // Only `--` is collapsed (can't forge a segment break); `:` passes through.
        let headers = headers_with(&[
            ("X-Baseten-Request-Id", "abc123"),
            (
                "X-Baseten-Customer-Request-Context",
                r#"{"cf_ray":"ray--e:vil","user_id":"u--v"}"#,
            ),
        ]);
        assert_eq!(
            get_or_create_context_id(&headers),
            "none--abc123--none--ray-e:vil:u-v"
        );
    }
}
