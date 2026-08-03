// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-model tokenization for approximate prefix routing.
//!
//! Pseudo tokenization is the default. A model can explicitly opt into a
//! startup-loaded `tokenizer.json` plus chat template, which keeps request-path
//! tokenization local and ensures every request for that model uses one hash
//! space.
//!
//! Tier 3 (pseudo) packs each `stride`-byte chunk of the plain-rendered
//! conversation into one `u32` (little-endian, zero-padded tail). This is a
//! deliberate upgrade over the design doc's original "every Nth byte" sampling:
//! the token count is the same (~bytes/stride, which for stride 4 approximates
//! a real tokenizer's output length on English text), but no bytes are
//! discarded — sampling leaves 3 of 4 bytes invisible, which manufactures
//! false prefix overlap between genuinely different prompts.
//!
//! Prefix consistency: rendering is append-only (each turn appends
//! `role:content\n`), and chunking is deterministic from offset 0, so two
//! conversations sharing a message prefix share their leading tokens. The only
//! seam is the final partial chunk of the shared prefix (one token, i.e. at
//! most one block of overlap credit lost at the append point).

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Context;
use basetenkenizer::{ChatTemplateOptions, ChatTemplateRenderer, Tokenizer};
use serde_json::{Map, Value};

use crate::config::{GwpConfig, ModelTokenizationConfig};

/// Which implementation produced a token vector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tier {
    /// Full chat template plus real tokenizer.
    ExactTemplate,
    /// Real tokenizer on a completions prompt (no chat template).
    TokenizerNoTemplate,
    /// Byte-chunk heuristic (each `stride`-byte chunk -> u32).
    Pseudo,
}

impl Tier {
    pub fn mode(self) -> &'static str {
        match self {
            Self::ExactTemplate | Self::TokenizerNoTemplate => "real",
            Self::Pseudo => "pseudo",
        }
    }
}

/// Approximate tokenization result.
#[derive(Clone, Debug)]
pub struct ApproxTokens {
    pub tokens: Vec<u32>,
    pub tier: Tier,
}

struct ModelTokenizer {
    tokenizer: Tokenizer,
    renderer: ChatTemplateRenderer,
    add_generation_prompt: bool,
    special_tokens: Map<String, Value>,
}

/// Immutable startup registry shared by all clones of a GWP core.
#[derive(Clone, Default)]
pub struct TokenizerRegistry {
    models: Arc<HashMap<String, ModelTokenizer>>,
}

#[derive(Debug, thiserror::Error)]
#[error("tokenization failed for model {model}: {detail}")]
pub struct TokenizationError {
    model: String,
    detail: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RequestShape {
    ChatCompletions,
    Completions,
    Unknown,
}

impl RequestShape {
    fn from_path(path: &str) -> Self {
        match path.split_once('?').map_or(path, |(path, _)| path) {
            "/v1/chat/completions" => Self::ChatCompletions,
            "/v1/completions" => Self::Completions,
            _ => Self::Unknown,
        }
    }

    fn uses_messages(self, body: &Value) -> bool {
        match self {
            Self::ChatCompletions => true,
            Self::Completions => false,
            Self::Unknown => body.get("messages").is_some(),
        }
    }
}

impl TokenizationError {
    fn new(model: &str, error: impl std::fmt::Display) -> Self {
        Self {
            model: model.to_string(),
            detail: error.to_string(),
        }
    }
}

impl TokenizerRegistry {
    /// Stable low-cardinality metric label for the configured implementation.
    pub fn mode_for(&self, model: &str) -> &'static str {
        if self.models.contains_key(model) {
            "real"
        } else {
            "pseudo"
        }
    }

    /// Load every explicitly configured real tokenizer. A broken opt-in is a
    /// startup error; silently degrading would mix incompatible prefix hashes.
    pub fn from_config(config: &GwpConfig) -> anyhow::Result<Self> {
        let mut models = HashMap::new();
        for (model, policy) in &config.tokenization.models {
            let ModelTokenizationConfig::Real { directory } = policy else {
                continue;
            };
            let tokenizer_json = directory.join("tokenizer.json");
            let chat_template = directory.join("chat_template.jinja");
            let tokenizer_config = directory.join("tokenizer_config.json");
            let tokenizer = Tokenizer::from_file(&tokenizer_json).with_context(|| {
                format!(
                    "loading tokenizer.json for model {model} from {}",
                    tokenizer_json.display()
                )
            })?;
            let template = std::fs::read_to_string(&chat_template).with_context(|| {
                format!(
                    "reading chat template for model {model} from {}",
                    chat_template.display()
                )
            })?;
            let renderer = ChatTemplateRenderer::new(&template).with_context(|| {
                format!(
                    "compiling chat template for model {model} from {}",
                    chat_template.display()
                )
            })?;
            let special_tokens = load_special_tokens(&tokenizer_config).with_context(|| {
                format!(
                    "loading tokenizer config for model {model} from {}",
                    tokenizer_config.display()
                )
            })?;
            models.insert(
                model.clone(),
                ModelTokenizer {
                    tokenizer,
                    renderer,
                    add_generation_prompt: true,
                    special_tokens,
                },
            );
        }
        Ok(Self {
            models: Arc::new(models),
        })
    }

    /// Tokenize an OpenAI request according to the exact model-name policy.
    /// Missing models and explicit `pseudo` policies take the pseudo path.
    pub fn tokenize(
        &self,
        model: &str,
        request_path: &str,
        body: &Value,
        pseudo_stride: usize,
    ) -> Result<ApproxTokens, TokenizationError> {
        let Some(real) = self.models.get(model) else {
            return Ok(ApproxTokens {
                tokens: pseudo_tokens(&routing_text(request_path, body), pseudo_stride),
                tier: Tier::Pseudo,
            });
        };

        let shape = RequestShape::from_path(request_path);
        let (tokens, tier) = if shape.uses_messages(body) {
            let messages = body.get("messages").cloned().unwrap_or(Value::Null);
            let options = real.options(body);
            let rendered = real
                .renderer
                .render(messages, options)
                .map_err(|error| TokenizationError::new(model, error))?;
            (
                real.tokenizer
                    .encode(&rendered)
                    .map_err(|error| TokenizationError::new(model, error))?,
                Tier::ExactTemplate,
            )
        } else {
            let prompt = body
                .get("prompt")
                .and_then(Value::as_str)
                .unwrap_or_default();
            (
                real.tokenizer
                    .encode(prompt)
                    .map_err(|error| TokenizationError::new(model, error))?,
                Tier::TokenizerNoTemplate,
            )
        };

        Ok(ApproxTokens {
            tokens: nonempty(tokens),
            tier,
        })
    }
}

fn load_special_tokens(path: &std::path::Path) -> anyhow::Result<Map<String, Value>> {
    let config: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let object = config
        .as_object()
        .context("tokenizer_config.json must contain a JSON object")?;
    Ok(object
        .iter()
        .filter(|(key, _)| key.ends_with("_token"))
        .filter_map(|(key, value)| {
            if value.is_string() {
                Some((key.clone(), value.clone()))
            } else {
                value
                    .get("content")
                    .filter(|content| content.is_string())
                    .cloned()
                    .map(|content| (key.clone(), content))
            }
        })
        .collect())
}

impl ModelTokenizer {
    fn options(&self, body: &Value) -> ChatTemplateOptions {
        let mut extra_context = Map::new();
        if let Some(object) = body.as_object() {
            for (key, value) in object {
                if self.renderer.undeclared_variables().contains(key)
                    && !matches!(
                        key.as_str(),
                        "messages"
                            | "tools"
                            | "documents"
                            | "add_generation_prompt"
                            | "continue_final_message"
                    )
                {
                    extra_context.insert(key.clone(), value.clone());
                }
            }
        }
        ChatTemplateOptions {
            add_generation_prompt: self.add_generation_prompt,
            tools: body.get("tools").cloned(),
            documents: body.get("documents").cloned(),
            special_tokens: self.special_tokens.clone(),
            extra_context,
            ..Default::default()
        }
    }
}

fn nonempty(tokens: Vec<u32>) -> Vec<u32> {
    if tokens.is_empty() {
        vec![u32::MAX]
    } else {
        tokens
    }
}

/// Render an OpenAI `messages` array to plain text, append-only per turn:
/// `role:content\n`. Handles both string content and content-part arrays
/// (text parts only; non-text parts are skipped — images etc. do not
/// contribute to prefix identity at the GWP tier).
pub fn render_plain(messages: &serde_json::Value) -> String {
    let mut out = String::new();
    let Some(turns) = messages.as_array() else {
        return out;
    };
    for turn in turns {
        let role = turn.get("role").and_then(|v| v.as_str()).unwrap_or("");
        out.push_str(role);
        out.push(':');
        match turn.get("content") {
            Some(serde_json::Value::String(text)) => out.push_str(text),
            Some(serde_json::Value::Array(parts)) => {
                for part in parts {
                    if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                        out.push_str(text);
                    }
                }
            }
            _ => {}
        }
        out.push('\n');
    }
    out
}

/// Text used for pseudo routing-prefix identity. Recognized OpenAI paths select
/// their body shape explicitly. Unknown paths retain best-effort field
/// detection so adding another compatible inference endpoint does not make it
/// unroutable.
pub fn routing_text(request_path: &str, body: &Value) -> String {
    if RequestShape::from_path(request_path).uses_messages(body) {
        return body.get("messages").map(render_plain).unwrap_or_default();
    }
    body.get("prompt")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// Tier-3 pseudo-tokenization: pack each `stride`-byte chunk into one `u32`
/// (little-endian; the trailing partial chunk is zero-padded). Empty input
/// produces one `u32::MAX` sentinel because the scheduler requires a non-zero
/// input sequence length. The sentinel cannot collide with valid UTF-8 input.
/// `stride` is clamped to `1..=4` — a u32 holds at most 4 bytes, and the
/// default of 4 makes the token count approximate a real tokenizer's
/// (~4 bytes/token).
///
/// This is the cheap (µs) path used for sticky-hit `add_request` accounting
/// and as the fall-through tier for unknown models.
pub fn pseudo_tokens(text: &str, stride: usize) -> Vec<u32> {
    let stride = stride.clamp(1, 4);
    let tokens: Vec<u32> = text
        .as_bytes()
        .chunks(stride)
        .map(|chunk| {
            let mut packed = [0u8; 4];
            packed[..chunk.len()].copy_from_slice(chunk);
            u32::from_le_bytes(packed)
        })
        .collect();
    nonempty(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn deterministic() {
        let request =
            json!({"model": "any-model", "messages": [{"role": "user", "content": "hello world"}]});
        let registry = TokenizerRegistry::default();
        let a = registry
            .tokenize("any-model", "/v1/chat/completions", &request, 4)
            .unwrap();
        let b = registry
            .tokenize("any-model", "/v1/chat/completions", &request, 4)
            .unwrap();
        assert_eq!(a.tokens, b.tokens);
        assert_eq!(a.tier, Tier::Pseudo);
        assert!(!a.tokens.is_empty());
    }

    #[test]
    fn empty_input_uses_non_colliding_scheduler_sentinel() {
        assert_eq!(pseudo_tokens("", 4), vec![u32::MAX]);
        assert_eq!(
            TokenizerRegistry::default()
                .tokenize("any-model", "/v1/chat/completions", &json!({}), 4)
                .unwrap()
                .tokens,
            vec![u32::MAX]
        );
    }

    #[test]
    fn shared_message_prefix_shares_leading_tokens() {
        // Turn 2 extends turn 1: the shared prefix must produce identical
        // leading tokens (all but the seam chunk at the append point).
        let turn1 = json!([{"role": "user", "content": "tell me about rust"}]);
        let turn2 = json!([
            {"role": "user", "content": "tell me about rust"},
            {"role": "assistant", "content": "Rust is a systems language."},
        ]);
        let t1 = pseudo_tokens(&render_plain(&turn1), 4);
        let t2 = pseudo_tokens(&render_plain(&turn2), 4);
        assert!(t2.len() > t1.len());
        // All complete chunks of the shared prefix are identical.
        let shared_complete = t1.len() - 1; // last chunk of t1 may be partial
        assert_eq!(t1[..shared_complete], t2[..shared_complete]);
    }

    #[test]
    fn different_content_diverges() {
        let a = pseudo_tokens(
            &render_plain(&json!([{"role": "user", "content": "AAAA"}])),
            4,
        );
        let b = pseudo_tokens(
            &render_plain(&json!([{"role": "user", "content": "AAAB"}])),
            4,
        );
        assert_ne!(a, b, "lossless packing must distinguish these");
    }

    #[test]
    fn nth_byte_sampling_would_have_collided() {
        // The failure mode that motivated chunk-packing over sampling: two
        // texts differing only in a non-sampled byte. With stride-4 sampling
        // (bytes 0, 4, 8, ...) these collide; with packing they must not.
        let a = pseudo_tokens("abcdefgh", 4);
        let b = pseudo_tokens("abXdefgh", 4);
        assert_ne!(a, b);
    }

    #[test]
    fn content_parts_render() {
        let msgs = json!([{
            "role": "user",
            "content": [
                {"type": "text", "text": "part one "},
                {"type": "image_url", "image_url": {"url": "ignored"}},
                {"type": "text", "text": "part two"},
            ],
        }]);
        assert_eq!(render_plain(&msgs), "user:part one part two\n");
    }

    #[test]
    fn known_paths_select_body_shape_and_unknown_paths_fall_back() {
        let body = json!({
            "messages": [{"role": "user", "content": "chat"}],
            "prompt": "completion",
        });
        assert_eq!(
            routing_text("/v1/chat/completions?debug=true", &body),
            "user:chat\n"
        );
        assert_eq!(routing_text("/v1/completions", &body), "completion");
        assert_eq!(routing_text("/v1/future-inference", &body), "user:chat\n");
    }

    #[test]
    fn stride_is_clamped() {
        // stride 0 must not panic (clamped to 1); stride > 4 clamps to 4.
        assert_eq!(pseudo_tokens("abcd", 0).len(), 4);
        assert_eq!(pseudo_tokens("abcd", 99).len(), 1);
    }

    #[test]
    fn partial_tail_is_padded() {
        // 5 bytes at stride 4 -> 2 tokens, tail zero-padded, no panic.
        let toks = pseudo_tokens("abcde", 4);
        assert_eq!(toks.len(), 2);
        assert_eq!(toks[1], u32::from_le_bytes([b'e', 0, 0, 0]));
    }

    #[test]
    fn configured_glm_bundle_uses_real_chat_and_completion_tokenization() {
        let mut config = GwpConfig::default();
        config.tokenization.models.insert(
            "glm-5.2".to_string(),
            ModelTokenizationConfig::Real {
                directory: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("vendored_tokenizers/glm5.2"),
            },
        );
        let registry = TokenizerRegistry::from_config(&config).expect("load GLM tokenizer bundle");

        let chat = registry
            .tokenize(
                "glm-5.2",
                "/v1/chat/completions",
                &json!({
                    "model": "glm-5.2",
                    "messages": [{"role": "user", "content": "hello"}],
                    "enable_thinking": false
                }),
                4,
            )
            .expect("tokenize chat");
        assert_eq!(chat.tier, Tier::ExactTemplate);
        assert!(chat.tokens.len() > 2);

        let completion = registry
            .tokenize(
                "glm-5.2",
                "/v1/completions",
                &json!({"model": "glm-5.2", "prompt": "hello"}),
                4,
            )
            .expect("tokenize completion");
        assert_eq!(completion.tier, Tier::TokenizerNoTemplate);
        assert!(!completion.tokens.is_empty());

        let unknown = registry
            .tokenize(
                "newly-discovered",
                "/v1/completions",
                &json!({"model": "newly-discovered", "prompt": "hello"}),
                4,
            )
            .expect("pseudo fallback");
        assert_eq!(unknown.tier, Tier::Pseudo);
    }
}
