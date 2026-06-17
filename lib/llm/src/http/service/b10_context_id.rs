// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Helpers for building the Baseten context ID from inbound HTTP headers.

use axum::http::HeaderMap;
use serde::Deserialize;

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
/// - `org_namespace`: `X-Baseten-Org-Namespace` (else "none").
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
pub(super) fn get_or_create_context_id(headers: &HeaderMap) -> String {
    let org_namespace = headers
        .get("X-Baseten-Org-Namespace")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("none");

    // Prefer the model-apis header; fall back to the legacy model-version header.
    let model_version_id = headers
        .get("X-Baseten-Model-APIs-Version-Id")
        .or_else(|| headers.get("X-Baseten-Model-Version-ID"))
        .and_then(|h| h.to_str().ok())
        .unwrap_or("none");

    let raw_request_id = headers
        .get("X-Baseten-Request-Id")
        .and_then(|h| h.to_str().ok())
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
    let Some(raw) = headers
        .get("X-Baseten-Customer-Request-Context")
        .and_then(|h| h.to_str().ok())
    else {
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
