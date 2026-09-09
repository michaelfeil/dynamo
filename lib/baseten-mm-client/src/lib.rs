// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../README.md")]

mod types;
pub use types::{
    Detail, EmbeddingPayload, EncodeRequest, EncoderRejection, Media, MultiModalResponse,
    ReplayRequest,
};

use anyhow::{Context, Result, bail};
use baseten_performance_client_core::http::HttpMethod;
use baseten_performance_client_core::{PerformanceClientCore, RequestProcessingPreference};
use std::collections::HashMap;

impl MultiModalClient {
    pub async fn replay(&self, requests: Vec<ReplayRequest>) -> Result<Vec<MultiModalResponse>> {
        let inputs = requests.iter().map(|item| item.request.clone()).collect();
        let responses = self.encode(inputs).await?;
        if responses.len() != requests.len() {
            bail!("Encoder replay returned an unexpected response count");
        }
        for (expected, response) in requests.iter().zip(&responses) {
            if expected.mm_hash != response.mm_hash || expected.length != response.length {
                bail!("Encoder replay metadata differs from the routed request");
            }
        }
        Ok(responses)
    }
    pub async fn encode(&self, requests: Vec<EncodeRequest>) -> Result<Vec<MultiModalResponse>> {
        let requests = requests
            .into_iter()
            .map(serde_json::to_value)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let batch = self.call_batch(requests).await?;
        batch
            .responses
            .iter()
            .map(MultiModalResponse::from_envelope)
            .collect()
    }
}

pub fn http_error_status(error: &anyhow::Error) -> Option<u16> {
    match error.downcast_ref::<baseten_performance_client_core::errors::ClientError>() {
        Some(baseten_performance_client_core::errors::ClientError::Http { status, .. }) => {
            Some(*status)
        }
        _ => None,
    }
}

pub struct HttpEncoderConfig {
    pub url: String,
    pub api_key: String,
    pub cache_urls: Vec<String>,
    pub proxy: Option<String>,
    pub request_timeout_s: f64,
    pub max_retries: u32,
    pub max_concurrent_requests: usize,
}

impl HttpEncoderConfig {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            api_key: String::new(),
            cache_urls: Vec::new(),
            proxy: None,
            request_timeout_s: 300.0,
            max_retries: 1,
            max_concurrent_requests: 64,
        }
    }
}

#[derive(Clone)]
pub struct MultiModalClient {
    client: PerformanceClientCore,
    request_path: String,
    preference: RequestProcessingPreference,
    method: HttpMethod,
}

#[derive(Debug, Default)]
pub struct EncoderBatch {
    pub responses: Vec<rmpv::Value>,
    pub individual_request_times: Vec<f64>,
    pub total_time: f64,
    pub num_cached: usize,
    pub num_shared: usize,
}

impl MultiModalClient {
    pub fn new(config: HttpEncoderConfig) -> Result<Self> {
        let proxy = match config.proxy {
            Some(proxy) => Some(proxy),
            None => bdn_proxy(
                std::env::var("BDN_PROXY").ok().as_deref(),
                std::env::var("DYNAMO_DISABLE_BDN_PROXY").ok().as_deref(),
            )?,
        };
        let mut headers = HashMap::new();
        if !config.cache_urls.is_empty() {
            let mut hops = config.cache_urls[1..].to_vec();
            hops.push(config.url.clone());
            headers.insert(
                "X-Baseten-Customer-Transparent-Route".into(),
                hops.join(";"),
            );
            if !config.api_key.is_empty() {
                headers.insert(
                    "X-Baseten-Customer-Transparent-Auth".into(),
                    config.api_key.clone(),
                );
            }
        }
        let method = if proxy.is_some() {
            headers.insert("Clamshack-Cache-Put".into(), "1".into());
            HttpMethod::PUT
        } else {
            HttpMethod::POST
        };
        let endpoint = url::Url::parse(config.cache_urls.first().unwrap_or(&config.url))
            .context("Invalid encoder endpoint URL")?;
        let request_path =
            endpoint[url::Position::BeforePath..url::Position::AfterQuery].to_owned();
        let client = PerformanceClientCore::new(
            endpoint[..url::Position::BeforePath].to_owned(),
            Some(config.api_key),
            1,
            None,
            proxy,
            None,
        )?;
        Ok(Self {
            client,
            request_path,
            method,
            preference: RequestProcessingPreference {
                max_concurrent_requests: Some(config.max_concurrent_requests),
                timeout_s: Some(config.request_timeout_s),
                max_retries: Some(config.max_retries),
                extra_headers: Some(headers),
                ..Default::default()
            },
        })
    }

    pub async fn call_batch(&self, requests: Vec<serde_json::Value>) -> Result<EncoderBatch> {
        if requests.is_empty() {
            return Ok(EncoderBatch::default());
        }
        let (results, elapsed) = self
            .client
            .process_batch_post_requests(
                self.request_path.clone(),
                requests,
                &self.preference,
                self.method,
            )
            .await?;
        let mut batch = EncoderBatch {
            responses: Vec::with_capacity(results.len()),
            individual_request_times: Vec::with_capacity(results.len()),
            total_time: elapsed.as_secs_f64(),
            num_cached: 0,
            num_shared: 0,
        };
        for (response, headers, duration) in results {
            match cache_status(&headers).as_str() {
                "HIT" => batch.num_cached += 1,
                "SHARED" => batch.num_shared += 1,
                _ => {}
            }
            batch.individual_request_times.push(duration.as_secs_f64());
            batch.responses.push(response);
        }
        let times = &batch.individual_request_times;
        tracing::info!(
            total_time = batch.total_time,
            mean_time = times.iter().sum::<f64>() / times.len().max(1) as f64,
            min_time = times.iter().copied().reduce(f64::min).unwrap_or_default(),
            num_requests = times.len(),
            num_cached = batch.num_cached,
            num_shared = batch.num_shared,
            "Encoder HTTP request metrics"
        );
        Ok(batch)
    }
}

fn bdn_proxy(proxy: Option<&str>, disabled: Option<&str>) -> Result<Option<String>> {
    if disabled.is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true")) {
        return Ok(None);
    }
    let Some(proxy) = proxy.filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    let mut parsed = url::Url::parse(proxy).context("Invalid BDN_PROXY URL")?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        bail!("BDN_PROXY must be an HTTP(S) URL");
    }
    if parsed.username().is_empty() && parsed.password().is_none() {
        parsed
            .set_username("bdn")
            .map_err(|_| anyhow::anyhow!("Invalid BDN_PROXY authority"))?;
        parsed
            .set_password(Some("dynamo-image-cache"))
            .map_err(|_| anyhow::anyhow!("Invalid BDN_PROXY authority"))?;
    }
    Ok(Some(parsed.into()))
}

fn cache_status(headers: &HashMap<String, String>) -> String {
    [
        "clamshack-cache-status",
        "x-cache-status",
        "x-baseten-customer-cache-status",
    ]
    .iter()
    .find_map(|key| headers.get(*key).filter(|value| !value.is_empty()))
    .map(|value| value.to_uppercase())
    .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bdn_proxy_configuration() -> Result<()> {
        assert_eq!(bdn_proxy(None, None)?, None);
        assert_eq!(bdn_proxy(Some(""), None)?, None);
        for disabled in ["1", "true", "TRUE"] {
            assert_eq!(bdn_proxy(Some("not a URL"), Some(disabled))?, None);
        }
        assert_eq!(
            bdn_proxy(Some("http://localhost:8080"), Some("0"))?.as_deref(),
            Some("http://bdn:dynamo-image-cache@localhost:8080/")
        );
        assert_eq!(
            bdn_proxy(Some("http://alice:secret@[::1]:8080"), None)?.as_deref(),
            Some("http://alice:secret@[::1]:8080/")
        );
        assert!(bdn_proxy(Some("not a URL"), None).is_err());
        assert!(bdn_proxy(Some("file:///tmp/proxy"), None).is_err());
        Ok(())
    }

    fn json_to_transport(value: &serde_json::Value) -> anyhow::Result<rmpv::Value> {
        Ok(rmpv::decode::read_value(
            &mut rmp_serde::to_vec_named(value)?.as_slice(),
        )?)
    }

    #[tokio::test]
    async fn http_proxy_preserves_binary_payload_and_cancels_on_drop() -> Result<()> {
        use std::{sync::Arc, time::Duration};
        use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
        tokio::time::timeout(Duration::from_secs(10), async {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
            let proxy = format!("http://{}", listener.local_addr()?);
            let started = Arc::new(tokio::sync::Notify::new());
            let notified = started.clone();
            let server = tokio::spawn(async move {
                for index in 0..2 {
                    let (stream, _) = listener.accept().await?;
                    let mut stream = BufReader::new(stream);
                    let mut line = String::new();
                    stream.read_line(&mut line).await?;
                    assert_eq!(line, "PUT http://encoder/encode HTTP/1.1\r\n");
                    let mut headers = HashMap::new();
                    loop {
                        line.clear();
                        stream.read_line(&mut line).await?;
                        if line == "\r\n" { break; }
                        let (key, value) = line.trim_end().split_once(": ").unwrap();
                        headers.insert(key.to_lowercase(), value.to_owned());
                    }
                    assert_eq!(headers["clamshack-cache-put"], "1");
                    assert_eq!(headers["proxy-authorization"], "Basic YmRuOmR5bmFtby1pbWFnZS1jYWNoZQ==");
                    let mut body = vec![0; headers["content-length"].parse::<usize>()?];
                    stream.read_exact(&mut body).await?;
                    if index == 1 {
                        notified.notify_one();
                        assert_eq!(stream.read(&mut [0]).await?, 0);
                        continue;
                    }
                    let payload = rmpv::Value::Map(vec![
                        ("success".into(), true.into()),
                        ("mm_response".into(), rmpv::Value::Map(vec![
                            ("mm_hash".into(), "binary".into()), ("length".into(), 3.into()),
                            ("mm_kwargs".into(), rmpv::Value::Binary(vec![0, 1, 2])),
                        ])),
                    ]);
                    let mut bytes = Vec::new();
                    rmpv::encode::write_value(&mut bytes, &payload)?;
                    stream.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/msgpack\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", bytes.len()).as_bytes()).await?;
                    stream.write_all(&bytes).await?;
                    stream.flush().await?;
                }
                Ok::<_, anyhow::Error>(())
            });
            let mut config = HttpEncoderConfig::new("http://encoder/encode");
            config.api_key = "secret".into();
            config.proxy = bdn_proxy(Some(&proxy), None)?;
            config.max_retries = 0;
            let client = MultiModalClient::new(config)?;
            let request = EncodeRequest::new(Media::Image { image_url: [("url".into(), "https://media/image".into())].into() }, "served");
            let responses = client.encode(vec![request.clone()]).await?;
            assert_eq!(responses[0].mm_kwargs, EmbeddingPayload::Bytes(vec![0, 1, 2]));
            let pending = tokio::spawn(async move { client.encode(vec![request]).await });
            started.notified().await;
            pending.abort();
            assert!(pending.await.unwrap_err().is_cancelled());
            server.await??;
            Ok::<_, anyhow::Error>(())
        }).await?
    }

    #[tokio::test]
    async fn http_batch_preserves_routing_order_cache_metrics_and_errors() {
        use axum::{
            Json, Router,
            http::{HeaderMap, StatusCode},
            routing::post,
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!(
            "http://{}/generate?version=1",
            listener.local_addr().unwrap()
        );
        let app = Router::new().route("/generate", post(|uri: axum::http::Uri, headers: HeaderMap, Json(request): Json<serde_json::Value>| async move {
            assert_eq!(uri.query(), Some("version=1"));
            assert_eq!(headers["authorization"], "Bearer secret");
            assert_eq!(headers["x-baseten-customer-transparent-route"], "http://cache2;http://encoder");
            assert_eq!(headers["x-baseten-customer-transparent-auth"], "secret");
            assert!(!headers.contains_key("clamshack-cache-put"));
            if request["model"] == "bad" {
                return (StatusCode::BAD_REQUEST, [("clamshack-cache-status", ""), ("x-cache-status", "MISS"), ("x-baseten-customer-cache-status", "")], Json(serde_json::json!({"error": "bad model"})));
            }
            let index = request["index"].as_u64().unwrap_or(0);
            if index == 0 {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
            (StatusCode::OK, [("clamshack-cache-status", if index == 0 { "hit" } else { "" }), ("x-cache-status", if index == 0 { "MISS" } else { "SHARED" }), ("x-baseten-customer-cache-status", "HIT")], Json(serde_json::json!({
                "success": true,
                "mm_response": {"mm_hash": index.to_string(), "mm_kwargs": "AAEC", "length": 3}
            })))
        }));
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = MultiModalClient::new(HttpEncoderConfig {
            url: "http://encoder".into(),
            api_key: "secret".into(),
            cache_urls: vec![url, "http://cache2".into()],
            proxy: None,
            request_timeout_s: 5.0,
            max_retries: 0,
            max_concurrent_requests: 2,
        })
        .unwrap();
        let batch = client
            .call_batch(vec![
                serde_json::json!({"index": 0}),
                serde_json::json!({"index": 1}),
            ])
            .await
            .unwrap();
        assert_eq!(
            batch.responses[0]["mm_response"]["mm_hash"].as_str(),
            Some("0")
        );
        assert_eq!(
            batch.responses[1]["mm_response"]["mm_hash"].as_str(),
            Some("1")
        );
        assert_eq!((batch.num_cached, batch.num_shared), (1, 1));
        let replay = ReplayRequest {
            request: EncodeRequest::new(
                Media::Image {
                    image_url: [("url".into(), "https://media/item".into())].into(),
                },
                "served",
            ),
            mm_hash: "0".into(),
            length: 3,
        };
        let replay: ReplayRequest =
            serde_json::from_slice(&serde_json::to_vec(&replay).unwrap()).unwrap();
        assert_eq!(
            client.replay(vec![replay.clone()]).await.unwrap()[0].length,
            3
        );
        let mut changed = replay;
        changed.length = 4;
        assert!(
            client
                .replay(vec![changed])
                .await
                .unwrap_err()
                .to_string()
                .contains("metadata differs")
        );
        assert_eq!(batch.individual_request_times.len(), 2);
        assert!(
            batch
                .individual_request_times
                .iter()
                .all(|time| *time > 0.0)
        );
        let error = client
            .call_batch(vec![serde_json::json!({"model": "bad"})])
            .await
            .err()
            .unwrap();
        assert_eq!(http_error_status(&error), Some(400));
        server.abort();
        let _ = server.await;
    }

    #[test]
    fn transport_preserves_opaque_payload_and_error_envelopes() {
        for media in ["image_url", "video_url", "audio_url"] {
            let input = serde_json::json!({"type": media, media: {"url": "https://media/item"}, "model": "served"});
            let request: EncodeRequest = serde_json::from_value(input.clone()).unwrap();
            let output = serde_json::to_value(request).unwrap();
            assert_eq!(output["detail"], "high");
            assert_eq!(output["allow_bytes_without_b64"], true);
            assert_eq!(output[media], input[media]);
        }
        let json = serde_json::json!({
            "success": true,
            "mm_response": {"mm_hash": "hash", "mm_kwargs": "AAEC", "length": 3}
        });
        let decoded = json_to_transport(&json).unwrap();
        assert_eq!(decoded["mm_response"]["mm_kwargs"].as_str(), Some("AAEC"));
        assert_eq!(decoded["mm_response"]["length"].as_u64(), Some(3));
        let response = MultiModalResponse::from_envelope(&decoded).unwrap();
        assert_eq!(response.mm_kwargs.bytes().unwrap().as_ref(), &[0, 1, 2]);
        assert_eq!(response.length, 3);
        for length in [
            serde_json::json!(true),
            serde_json::json!(0),
            serde_json::json!(-1),
            serde_json::Value::Null,
        ] {
            let mut malformed = json.clone();
            malformed["mm_response"]["length"] = length;
            let malformed = json_to_transport(&malformed).unwrap();
            assert!(MultiModalResponse::from_envelope(&malformed).is_err());
        }

        let error = serde_json::json!({
            "success": false, "response_code": 413, "error_message": "image too large"
        });
        let decoded = json_to_transport(&error).unwrap();
        assert_eq!(decoded["response_code"].as_u64(), Some(413));
        let error = MultiModalResponse::from_envelope(&decoded).unwrap_err();
        assert_eq!(error.downcast_ref::<EncoderRejection>().unwrap().code, 413);
    }
}
