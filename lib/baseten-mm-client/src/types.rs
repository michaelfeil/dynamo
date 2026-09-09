// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result, bail};
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::{borrow::Cow, collections::BTreeMap};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Media {
    #[serde(rename = "image_url")]
    Image { image_url: BTreeMap<String, String> },
    #[serde(rename = "video_url")]
    Video { video_url: BTreeMap<String, String> },
    #[serde(rename = "audio_url")]
    Audio { audio_url: BTreeMap<String, String> },
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Detail {
    Low,
    #[default]
    High,
    Original,
    Auto,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncodeRequest {
    #[serde(flatten)]
    pub media: Media,
    pub model: String,
    #[serde(default)]
    pub detail: Detail,
    #[serde(default = "allow_bytes_default")]
    pub allow_bytes_without_b64: bool,
}

fn allow_bytes_default() -> bool {
    true
}

impl EncodeRequest {
    pub fn new(media: Media, model: impl Into<String>) -> Self {
        Self {
            media,
            model: model.into(),
            detail: Detail::High,
            allow_bytes_without_b64: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum EmbeddingPayload {
    Bytes(Vec<u8>),
    Base64(String),
}

impl EmbeddingPayload {
    pub fn bytes(&self) -> Result<Cow<'_, [u8]>> {
        match self {
            Self::Bytes(bytes) => Ok(Cow::Borrowed(bytes)),
            Self::Base64(text) => Ok(Cow::Owned(
                base64::engine::general_purpose::STANDARD
                    .decode(text)
                    .context("Field 'mm_kwargs' is not valid base64 data")?,
            )),
        }
    }

    pub fn unpack(&self) -> Result<rmpv::Value> {
        let bytes = self.bytes()?;
        let mut remaining = bytes.as_ref();
        let payload = rmpv::decode::read_value(&mut remaining)?;
        if !remaining.is_empty() {
            bail!("Trailing data after mm_kwargs MessagePack payload");
        }
        if !payload.is_map() {
            bail!("Expected msgpack-decoded mm_kwargs to be a mapping");
        }
        Ok(payload)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MultiModalResponse {
    pub mm_hash: String,
    pub mm_kwargs: EmbeddingPayload,
    pub length: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayRequest {
    pub request: EncodeRequest,
    pub mm_hash: String,
    pub length: u64,
}

#[derive(Debug)]
pub struct EncoderRejection {
    pub code: u16,
    pub message: String,
}

impl std::fmt::Display for EncoderRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Encoder call failed ({}): {}", self.code, self.message)
    }
}
impl std::error::Error for EncoderRejection {}

impl MultiModalResponse {
    pub fn from_envelope(envelope: &rmpv::Value) -> Result<Self> {
        if envelope["success"].as_bool() != Some(true) {
            return Err(EncoderRejection {
                code: envelope["response_code"]
                    .as_u64()
                    .unwrap_or(500)
                    .try_into()?,
                message: envelope["error_message"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .unwrap_or("encoder request failed")
                    .to_owned(),
            }
            .into());
        }
        let payload = &envelope["mm_response"];
        if !payload.is_map() {
            bail!("Encoder returned success but no payload");
        }
        let mm_hash = payload["mm_hash"]
            .as_str()
            .context("Expected str mm_hash")?
            .to_owned();
        let length = payload["length"]
            .as_u64()
            .filter(|n| *n > 0)
            .context("Expected integer length >= 1")?;
        let mm_kwargs = match &payload["mm_kwargs"] {
            rmpv::Value::Binary(bytes) => EmbeddingPayload::Bytes(bytes.clone()),
            rmpv::Value::String(text) => EmbeddingPayload::Base64(
                text.as_str().context("Invalid UTF-8 mm_kwargs")?.to_owned(),
            ),
            _ => bail!("Field 'mm_kwargs' must be base64 string or bytes"),
        };
        Ok(Self {
            mm_hash,
            mm_kwargs,
            length,
        })
    }
}
