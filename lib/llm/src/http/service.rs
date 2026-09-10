// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP Service for Dynamo LLM
//!
//! The primary purpose of this crate is to service the dynamo-llm protocols via OpenAI compatible HTTP endpoints. This component
//! is meant to be a gateway/ingress into the Dynamo LLM Distributed Runtime.
//!
//! In order to create a common pattern, the HttpService forwards the incoming OAI Chat Request or OAI Completion Request to the
//! to a model-specific engines.  The engines can be attached and detached dynamically using the [`ModelManager`].
//!
//! Note: All requests, whether the client requests `stream=true` or `stream=false`, are propagated downstream as `stream=true`.
//! This enables use to handle only 1 pattern of request-response in the downstream services. Non-streaming user requests are
//! aggregated by the HttpService and returned as a single response.
//!
//! TODO(): Add support for model-specific metadata and status. Status will allow us to return a 503 when the model is supposed
//! to be ready, but there is a problem with the model.
//!
//! The [`service_v2::HttpService`] can be further extended to host any [`axum::Router`] using the [`service_v2::HttpServiceConfigBuilder`].

mod anthropic;
pub mod baseten;
pub mod metadata;
mod openai;

pub mod b10_rate_limiter;
pub mod busy_threshold;
pub mod disconnect;
pub mod error;
pub mod health;
pub mod health_b10;
pub mod metrics;
pub mod openapi_docs;
pub mod realtime;
pub mod service_v2;
pub mod waypoints;

pub use axum;
pub use metrics::Metrics;

/// The client's request body: the raw bytes plus the `Value` parsed from them.
///
/// The Messages and Responses handlers need both. The canonicalizer consumes the `Value`
/// verbatim, and the typed view is parsed off the same bytes so deserialization errors can
/// report a byte offset into what the client actually sent.
///
/// Replaces `Json<serde_json::Value>` in those two handlers. Content-type and syntax-error
/// rejections keep the status and message `Json` produced, so nothing else about the edge
/// changes.
pub(crate) struct RawJson {
    pub bytes: axum::body::Bytes,
    pub value: serde_json::Value,
}

/// Same rule `Json` applies: parse the whole header as a MIME type and accept
/// `application/json` or any `application/*+json` suffix type. Parsing the whole value matters —
/// a malformed parameter section (`application/json; charset`) is not a valid MIME type and must
/// be rejected, so this cannot just read the part before the first `;`.
fn is_json_content_type(headers: &axum::http::HeaderMap) -> bool {
    let Some(content_type) = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
    else {
        return false;
    };
    let Ok(mime) = content_type.parse::<mime::Mime>() else {
        return false;
    };
    mime.type_() == mime::APPLICATION
        && (mime.subtype() == mime::JSON || mime.suffix() == Some(mime::JSON))
}

impl<S: Send + Sync> axum::extract::FromRequest<S> for RawJson {
    type Rejection = axum::response::Response;

    async fn from_request(req: axum::extract::Request, state: &S) -> Result<Self, Self::Rejection> {
        use axum::response::IntoResponse;
        if !is_json_content_type(req.headers()) {
            return Err((
                axum::http::StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "Expected request with `Content-Type: application/json`",
            )
                .into_response());
        }
        // Honors the router's DefaultBodyLimit.
        let bytes = axum::body::Bytes::from_request(req, state)
            .await
            .map_err(IntoResponse::into_response)?;
        let value = serde_json::from_slice(&bytes).map_err(|err| {
            (
                axum::http::StatusCode::BAD_REQUEST,
                format!("Failed to parse the request body as JSON: {err}"),
            )
                .into_response()
        })?;
        Ok(Self { bytes, value })
    }
}

/// Parse the typed request view from the client's raw body, keeping the field path and byte
/// offset in the error message.
///
/// `T::deserialize(&value)` over an already-parsed `Value` reports only the innermost failure —
/// "unknown variant `tool`" with no hint that it came from `messages[6].role`, and no position,
/// because a `Value` has no source offsets. Deserializing from the bytes through
/// `serde_path_to_error` restores both, matching what the `Json<T>` extractor reported. It also
/// restores `Json<T>`'s rejection of a repeated known member ("duplicate field `model`"), which
/// a parse through `Value` cannot see because `Value` keeps only the last one.
///
/// Takes the buffer by value: bodies run to the 45 MB limit, the `Value` beside it is already a
/// second copy, and a handler holds its locals across every later await. Consuming it here drops
/// it at the parse instead.
pub(crate) fn deserialize_body<T: serde::de::DeserializeOwned>(
    body: axum::body::Bytes,
) -> Result<T, String> {
    let mut deserializer = serde_json::Deserializer::from_slice(&body);
    serde_path_to_error::deserialize(&mut deserializer).map_err(|err| {
        let path = err.path().to_string();
        // serde_path_to_error renders the root path as "."; nothing to prefix then.
        if path.is_empty() || path == "." {
            err.into_inner().to_string()
        } else {
            format!("{path}: {}", err.into_inner())
        }
    })
}

/// Documentation for a route
#[derive(Debug, Clone)]
pub struct RouteDoc {
    method: axum::http::Method,
    path: String,
}

impl std::fmt::Display for RouteDoc {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} {}", self.method, self.path)
    }
}

impl RouteDoc {
    pub fn new<T: Into<String>>(method: axum::http::Method, path: T) -> Self {
        RouteDoc {
            method,
            path: path.into(),
        }
    }
}

#[cfg(test)]
mod deserialize_body_tests {
    use super::*;

    #[derive(serde::Deserialize, Debug)]
    #[allow(dead_code)]
    struct Msg {
        role: Role,
    }

    #[derive(serde::Deserialize, Debug)]
    #[serde(rename_all = "lowercase")]
    enum Role {
        User,
        Assistant,
    }

    #[derive(serde::Deserialize, Debug)]
    #[allow(dead_code)]
    struct Req {
        max_tokens: u32,
        messages: Vec<Msg>,
    }

    /// The message names the field that failed and where it is in the client's body. Parsing the
    /// typed view off an already-parsed `Value` loses both.
    #[test]
    fn error_carries_the_field_path_and_position() {
        let body = br#"{"max_tokens": 8, "messages": [{"role": "user"}, {"role": "tool"}]}"#;
        let err = deserialize_body::<Req>(body.as_slice().into()).unwrap_err();
        assert!(err.starts_with("messages[1].role: "), "{err}");
        assert!(err.contains("unknown variant `tool`"), "{err}");
        assert!(err.contains("at line 1 column "), "{err}");
    }

    /// A failure at the root has no path to prefix, so the message is just the serde error.
    #[test]
    fn root_error_is_not_prefixed() {
        let err = deserialize_body::<Req>(br#"{"messages": []}"#.as_slice().into()).unwrap_err();
        assert!(err.starts_with("missing field `max_tokens`"), "{err}");
    }

    /// A repeated known member is rejected, as `Json<T>` rejected it. A parse routed through
    /// `Value` cannot see the duplicate, because `Value` keeps only the last one.
    #[test]
    fn duplicate_member_is_rejected() {
        let body = br#"{"max_tokens": 8, "max_tokens": 9, "messages": []}"#;
        let err = deserialize_body::<Req>(body.as_slice().into()).unwrap_err();
        assert!(err.starts_with("duplicate field `max_tokens`"), "{err}");
    }

    #[test]
    fn accepts_json_content_types_and_rejects_others() {
        let header = |value: &str| {
            let mut headers = axum::http::HeaderMap::new();
            headers.insert(
                axum::http::header::CONTENT_TYPE,
                axum::http::HeaderValue::from_str(value).unwrap(),
            );
            headers
        };
        for value in [
            "application/json",
            "application/json; charset=utf-8",
            "Application/JSON",
            "application/vnd.api+json",
        ] {
            assert!(is_json_content_type(&header(value)), "{value}");
        }
        for value in [
            "text/plain",
            "application/xml",
            "application/jsonish",
            // Not a valid MIME type: a parameter with no value.
            "application/json; charset",
        ] {
            assert!(!is_json_content_type(&header(value)), "{value}");
        }
        assert!(!is_json_content_type(&axum::http::HeaderMap::new()));
    }
}
