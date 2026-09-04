// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Anthropic Messages API types.
//!
//! Pure protocol types for the `/v1/messages` endpoint -- request, response,
//! streaming events, error shapes, and count-tokens types.

use serde::{Deserialize, Serialize};

/// Anthropic-style cache control hint for prefix pinning with TTL.
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub control_type: CacheControlType,
    /// TTL as seconds (integer) or shorthand ("5m" = 300s, "1h" = 3600s). Clamped to [300, 3600].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum CacheControlType {
    #[default]
    Ephemeral,
    #[serde(other)]
    Unknown,
}

const MIN_TTL_SECONDS: u64 = 300;
const MAX_TTL_SECONDS: u64 = 3600;

impl CacheControl {
    /// Parse TTL string to seconds, clamped to [300, 3600].
    ///
    /// Accepts integer seconds ("120", "600") or shorthand ("5m", "1h").
    /// Values below 300 are clamped to 300; values above 3600 are clamped to 3600.
    /// Unrecognized strings default to 300s.
    pub fn ttl_seconds(&self) -> u64 {
        let raw = match self.ttl.as_deref() {
            None => return MIN_TTL_SECONDS,
            Some("5m") => 300,
            Some("1h") => 3600,
            Some(other) => match other.parse::<u64>() {
                Ok(secs) => secs,
                Err(_) => {
                    tracing::warn!("Unrecognized TTL '{}', defaulting to 300s", other);
                    return MIN_TTL_SECONDS;
                }
            },
        };
        raw.clamp(MIN_TTL_SECONDS, MAX_TTL_SECONDS)
    }
}
/// Parsed system prompt content. This is a LOSSY view of the wire `system`
/// field: a block array is collapsed to one string (blocks joined with `\n`)
/// and only the last `cache_control` marker is kept — per-block boundaries,
/// per-block `cache_control`, and any other block attributes are gone. It
/// serves the typed request's own consumers (template rendering); the shared
/// canonicalizer reads the client's original JSON instead, so it never sees
/// this collapse.
///
/// Serializes back in Anthropic's own wire shapes — a plain string, or a
/// single text block when a `cache_control` has to be carried — never as
/// this struct's own `{"text": ...}` object, which no Anthropic consumer
/// would read as system text. That output is well-formed Anthropic, not a
/// byte-for-byte round trip of a multi-block input.
#[derive(Debug, Clone, Deserialize)]
pub struct SystemContent {
    /// The concatenated text from all system blocks (or the plain string).
    pub text: String,
    /// Cache control from the last system block that had one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl Serialize for SystemContent {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let Some(cache_control) = &self.cache_control else {
            return serializer.serialize_str(&self.text);
        };
        #[derive(Serialize)]
        struct SystemBlockRef<'a> {
            #[serde(rename = "type")]
            block_type: &'static str,
            text: &'a str,
            cache_control: &'a CacheControl,
        }
        let block = SystemBlockRef {
            block_type: "text",
            text: &self.text,
            cache_control,
        };
        let mut seq = serializer.serialize_seq(Some(1))?;
        seq.serialize_element(&block)?;
        seq.end()
    }
}

/// Deserialize `system` from either a plain string or an array of text blocks.
/// The Anthropic API accepts both `"system": "text"` and
/// `"system": [{"type": "text", "text": "...", "cache_control": {...}}]`.
fn deserialize_system_prompt<'de, D>(deserializer: D) -> Result<Option<SystemContent>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum SystemPrompt {
        Text(String),
        Blocks(Vec<SystemBlock>),
    }

    #[derive(Deserialize)]
    struct SystemBlock {
        text: String,
        #[serde(default)]
        cache_control: Option<CacheControl>,
    }

    let maybe: Option<SystemPrompt> = Option::deserialize(deserializer)?;
    Ok(maybe.map(|sp| match sp {
        SystemPrompt::Text(s) => SystemContent {
            text: s,
            cache_control: None,
        },
        SystemPrompt::Blocks(blocks) => {
            let cache_control = blocks.iter().rev().find_map(|b| b.cache_control.clone());
            let text = blocks
                .into_iter()
                .map(|b| b.text)
                .collect::<Vec<_>>()
                .join("\n");
            SystemContent {
                text,
                cache_control,
            }
        }
    }))
}
/// Top-level request body for `POST /v1/messages`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicCreateMessageRequest {
    /// The model to use (e.g. "claude-sonnet-4-20250514").
    pub model: String,

    /// The maximum number of tokens to generate.
    pub max_tokens: u32,

    /// The conversation messages.
    pub messages: Vec<AnthropicMessage>,

    /// Optional system prompt (string or array of `{"type":"text","text":"..."}` blocks).
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_system_prompt"
    )]
    pub system: Option<SystemContent>,

    /// Sampling temperature (0.0 - 1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Nucleus sampling parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Top-K sampling parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    /// Custom stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,

    /// Whether to stream the response.
    #[serde(default)]
    pub stream: bool,

    /// Optional metadata (e.g. user_id).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,

    /// Tools the model may call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,

    /// How the model should choose which tool to call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<AnthropicToolChoice>,

    /// Top-level cache control for automatic prompt prefix caching.
    /// When present, the system caches all content up to the last cacheable block.
    /// Matches the Anthropic Messages API automatic caching mode.
    /// See: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#automatic-caching
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,

    /// Extended thinking configuration. When enabled, the model produces
    /// `thinking` content blocks containing its internal reasoning before
    /// the final response. The `budget_tokens` field controls how many tokens
    /// the model may use for thinking (must be >= 1024 and < max_tokens).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,

    /// Service tier selection: `"auto"` or `"standard_only"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,

    /// Container identifier for stateful sandbox sessions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<String>,

    /// Output configuration: effort level and optional JSON schema format.
    /// `effort` can be `"low"`, `"medium"`, `"high"`, or `"max"`.
    /// `format` specifies structured JSON output constraints.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_config: Option<serde_json::Value>,

    /// Verbatim passthrough of request fields not modeled above (ordered
    /// under `preserve_order`; see `CreateChatCompletionRequest.unmodeled`).
    #[serde(flatten)]
    pub unmodeled: serde_json::Map<String, serde_json::Value>,
}

/// Extended thinking configuration for the request.
///
/// When `type` is `"enabled"`, the model will produce `thinking` content blocks
/// with its internal reasoning. `budget_tokens` controls the maximum tokens
/// available for thinking (minimum 1024, must be less than `max_tokens`).
/// When `type` is `"disabled"`, no thinking blocks are produced.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingConfig {
    /// Either `"enabled"` or `"disabled"`.
    #[serde(rename = "type")]
    pub thinking_type: String,
    /// Maximum tokens for internal reasoning. Only relevant when type is "enabled".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
}

/// A single message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: AnthropicRole,
    #[serde(flatten)]
    pub content: AnthropicMessageContent,
}

/// The role of a message sender.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicRole {
    User,
    Assistant,
    /// Compatibility for clients that place system instructions in `messages[]`
    /// instead of the top-level `system` field.
    System,
}

/// Message content -- either a plain string or an array of content blocks.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum AnthropicMessageContent {
    /// Plain text content.
    Text { content: String },
    /// Array of structured content blocks.
    Blocks { content: Vec<AnthropicContentBlock> },
}

/// Hand-written so a malformed block reports the field that is actually
/// wrong. `#[serde(untagged)]` discards every inner error and yields only
/// "data did not match any variant of untagged enum AnthropicMessageContent",
/// which is useless in a 400 body.
impl<'de> Deserialize<'de> for AnthropicMessageContent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct ContentField {
            content: serde_json::Value,
        }

        let content = ContentField::deserialize(deserializer)?.content;
        match content {
            serde_json::Value::String(text) => Ok(Self::Text { content: text }),
            serde_json::Value::Array(_) => Ok(Self::Blocks {
                content: serde_json::from_value(content).map_err(serde::de::Error::custom)?,
            }),
            other => Err(serde::de::Error::custom(format!(
                "message `content` must be a string or an array of content blocks, got {}",
                match other {
                    serde_json::Value::Null => "null",
                    serde_json::Value::Bool(_) => "a boolean",
                    serde_json::Value::Number(_) => "a number",
                    serde_json::Value::Object(_) => "an object",
                    serde_json::Value::String(_) | serde_json::Value::Array(_) => unreachable!(),
                }
            ))),
        }
    }
}

/// A single content block within a message.
///
/// Uses a custom deserializer so that unknown block types (e.g. `citations`,
/// `server_tool_use`, `redacted_thinking`) are captured as `Other(Value)` instead
/// of causing a hard deserialization failure. This is important because Claude
/// Code may send block types that we don't yet handle.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum AnthropicContentBlock {
    /// Text content block. May optionally include `citations` -- references to
    /// source documents that support the text content. Citations are generated
    /// by the model when document/PDF content is provided and citation mode is enabled.
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        citations: Option<Vec<serde_json::Value>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// Image content block.
    #[serde(rename = "image")]
    Image { source: AnthropicImageSource },
    /// Tool use request from assistant.
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// Tool result from user.
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<ToolResultContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// Thinking content block from assistant (extended thinking / reasoning).
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        signature: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// Redacted thinking block from assistant. Contains encrypted reasoning data
    /// that is opaque to the client but must be passed back verbatim in multi-turn
    /// conversations so the model can maintain its chain of thought.
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
    /// Server-initiated tool use block. Represents a tool call that the API
    /// executes server-side (e.g., web search). The client receives the result
    /// via a corresponding `web_search_tool_result` or similar block.
    #[serde(rename = "server_tool_use")]
    ServerToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: serde_json::Value,
    },
    /// Result from a server-initiated tool (e.g., web search results).
    /// Contains structured content returned by the server-side tool execution.
    #[serde(rename = "web_search_tool_result")]
    WebSearchToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: serde_json::Value,
    },
    /// Catch-all for unrecognized block types. Preserves the full JSON value
    /// so that new Anthropic features don't break the endpoint and can be
    /// round-tripped or inspected.
    #[serde(untagged)]
    Other(serde_json::Value),
}

/// Content of a `tool_result` block -- either a plain string or an array of
/// content blocks (the Anthropic API accepts both).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<ToolResultContentBlock>),
}

impl ToolResultContent {
    /// Extract the text content, concatenating array blocks if needed.
    pub fn into_text(self) -> String {
        match self {
            ToolResultContent::Text(s) => s,
            ToolResultContent::Blocks(blocks) => blocks
                .into_iter()
                .filter_map(|b| match b {
                    ToolResultContentBlock::Text { text } => Some(text),
                    ToolResultContentBlock::Document(doc) => doc.text(),
                    ToolResultContentBlock::SearchResult(result) => result.text(),
                    ToolResultContentBlock::Image { .. } | ToolResultContentBlock::Other(_) => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

/// A content block within a `tool_result.content` array.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultContentBlock {
    Text {
        text: String,
    },
    /// Image block inside a tool result. Agent clients (e.g. Claude Code's
    /// Read/screenshot tools) deliver images to the model via `tool_result`
    /// content arrays, so these must be preserved through conversion.
    Image {
        source: AnthropicImageSource,
    },
    /// Document block inside a tool result (citations / RAG patterns).
    Document(DocumentBlock),
    /// Search-result block inside a tool result.
    SearchResult(SearchResultBlock),
    /// Catch-all for other non-text blocks in tool results.
    #[serde(untagged)]
    Other(serde_json::Value),
}

/// A `document` block: a titled document with a typed source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentBlock {
    pub source: DocumentSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub citations: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl DocumentBlock {
    /// Text representation of the document, if its source carries one.
    pub fn text(&self) -> Option<String> {
        self.source.text()
    }
}

/// The source payload of a `document` block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DocumentSource {
    /// Plain-text document.
    Text {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        media_type: Option<String>,
        data: String,
    },
    /// Document composed of nested content blocks.
    Content {
        content: Vec<ToolResultContentBlock>,
    },
    /// Base64-encoded binary document (e.g. PDF). No text representation.
    Base64 { media_type: String, data: String },
    /// Document referenced by URL. No text representation.
    Url { url: String },
}

impl DocumentSource {
    /// Text representation of the source, if it carries one.
    pub fn text(&self) -> Option<String> {
        match self {
            DocumentSource::Text { data, .. } => Some(data.clone()),
            DocumentSource::Content { content } => concat_text_blocks(content),
            DocumentSource::Base64 { .. } | DocumentSource::Url { .. } => None,
        }
    }
}

/// A `search_result` block: a search hit with text content and citation
/// metadata. Fields Anthropic requires are still optional here so that a
/// partial block degrades to "no text" rather than failing the request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResultBlock {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default)]
    pub content: Vec<ToolResultContentBlock>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub citations: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl SearchResultBlock {
    /// Text representation of the search result's content, if any.
    pub fn text(&self) -> Option<String> {
        concat_text_blocks(&self.content)
    }
}

/// Concatenate the text blocks in a nested content array.
fn concat_text_blocks(blocks: &[ToolResultContentBlock]) -> Option<String> {
    let texts: Vec<&str> = blocks
        .iter()
        .filter_map(|block| match block {
            ToolResultContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    if texts.is_empty() {
        None
    } else {
        Some(texts.concat())
    }
}

impl<'de> Deserialize<'de> for ToolResultContentBlock {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        match value.get("type").and_then(|value| value.as_str()) {
            Some("text") => {
                let text = value
                    .get("text")
                    .and_then(|value| value.as_str())
                    .ok_or_else(|| serde::de::Error::missing_field("text"))?;
                Ok(Self::Text {
                    text: text.to_string(),
                })
            }
            Some("image") => {
                let source = value
                    .get("source")
                    .cloned()
                    .ok_or_else(|| serde::de::Error::missing_field("source"))
                    .and_then(|value| {
                        serde_json::from_value(value).map_err(serde::de::Error::custom)
                    })?;
                Ok(Self::Image { source })
            }
            // Typed blocks with a graceful fallback: a document/search_result
            // whose shape we don't understand degrades to Other (skipped by
            // conversions) rather than failing the whole request.
            Some("document") => Ok(
                match serde_json::from_value::<DocumentBlock>(value.clone()) {
                    Ok(doc) => Self::Document(doc),
                    Err(_) => Self::Other(value),
                },
            ),
            Some("search_result") => Ok(
                match serde_json::from_value::<SearchResultBlock>(value.clone()) {
                    Ok(result) => Self::SearchResult(result),
                    Err(_) => Self::Other(value),
                },
            ),
            None => match value.get("text").and_then(|value| value.as_str()) {
                Some(text) => Ok(Self::Text {
                    text: text.to_string(),
                }),
                None => Ok(Self::Other(value)),
            },
            _ => Ok(Self::Other(value)),
        }
    }
}

/// Custom deserializer for `AnthropicContentBlock` that handles unknown types
/// gracefully. Since serde's `#[serde(other)]` is not supported on internally
/// tagged enums, we deserialize as `Value` first and dispatch manually.
impl<'de> Deserialize<'de> for AnthropicContentBlock {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let block_type = value
            .get("type")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string();

        match block_type.as_str() {
            "text" => {
                let text = value
                    .get("text")
                    .and_then(|t| t.as_str())
                    .ok_or_else(|| serde::de::Error::missing_field("text"))?
                    .to_string();
                let citations: Option<Vec<serde_json::Value>> = value
                    .get("citations")
                    .cloned()
                    .and_then(|v| serde_json::from_value(v).ok());
                let cache_control: Option<CacheControl> = value
                    .get("cache_control")
                    .cloned()
                    .and_then(|v| serde_json::from_value(v).ok());
                Ok(AnthropicContentBlock::Text {
                    text,
                    citations,
                    cache_control,
                })
            }
            "image" => {
                let source: AnthropicImageSource =
                    serde_json::from_value(value.get("source").cloned().unwrap_or_default())
                        .map_err(serde::de::Error::custom)?;
                Ok(AnthropicContentBlock::Image { source })
            }
            "tool_use" => {
                let id = value
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| serde::de::Error::missing_field("id"))?
                    .to_string();
                let name = value
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| serde::de::Error::missing_field("name"))?
                    .to_string();
                let input = value.get("input").cloned().unwrap_or(serde_json::json!({}));
                let cache_control: Option<CacheControl> = value
                    .get("cache_control")
                    .cloned()
                    .and_then(|v| serde_json::from_value(v).ok());
                Ok(AnthropicContentBlock::ToolUse {
                    id,
                    name,
                    input,
                    cache_control,
                })
            }
            "tool_result" => {
                let tool_use_id = value
                    .get("tool_use_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| serde::de::Error::missing_field("tool_use_id"))?
                    .to_string();
                let content: Option<ToolResultContent> = value
                    .get("content")
                    .cloned()
                    .and_then(|v| serde_json::from_value(v).ok());
                let is_error = value.get("is_error").and_then(|v| v.as_bool());
                let cache_control: Option<CacheControl> = value
                    .get("cache_control")
                    .cloned()
                    .and_then(|v| serde_json::from_value(v).ok());
                Ok(AnthropicContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                    cache_control,
                })
            }
            "thinking" => {
                let thinking = value
                    .get("thinking")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| serde::de::Error::missing_field("thinking"))?
                    .to_string();
                let signature = value
                    .get("signature")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| serde::de::Error::missing_field("signature"))?
                    .to_string();
                let cache_control: Option<CacheControl> = value
                    .get("cache_control")
                    .cloned()
                    .and_then(|v| serde_json::from_value(v).ok());
                Ok(AnthropicContentBlock::Thinking {
                    thinking,
                    signature,
                    cache_control,
                })
            }
            "redacted_thinking" => {
                let data = value
                    .get("data")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| serde::de::Error::missing_field("data"))?
                    .to_string();
                Ok(AnthropicContentBlock::RedactedThinking { data })
            }
            "server_tool_use" => {
                let id = value
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| serde::de::Error::missing_field("id"))?
                    .to_string();
                let name = value
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| serde::de::Error::missing_field("name"))?
                    .to_string();
                let input = value.get("input").cloned().unwrap_or(serde_json::json!({}));
                Ok(AnthropicContentBlock::ServerToolUse { id, name, input })
            }
            "web_search_tool_result" => {
                let tool_use_id = value
                    .get("tool_use_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| serde::de::Error::missing_field("tool_use_id"))?
                    .to_string();
                let content = value
                    .get("content")
                    .cloned()
                    .unwrap_or(serde_json::json!([]));
                Ok(AnthropicContentBlock::WebSearchToolResult {
                    tool_use_id,
                    content,
                })
            }
            other => {
                tracing::debug!(
                    "Unrecognized Anthropic content block type '{}', preserving as Other",
                    other
                );
                Ok(AnthropicContentBlock::Other(value))
            }
        }
    }
}

/// Image source for image content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

/// A tool definition.
///
/// Client tools (custom) require `name` + `input_schema`. Server tools
/// (web_search, bash, text_editor, code_execution, etc.) are discriminated
/// by their `type` field (e.g. `"web_search_20260209"`) and may not have
/// `input_schema`. We keep all fields optional beyond `name` so both
/// kinds deserialize successfully and pass through to the backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicTool {
    /// Tool name (required for client tools, present on server tools too).
    pub name: String,
    /// Tool type discriminator. Client tools use `"custom"` (or omit).
    /// Server tools use versioned types like `"web_search_20260209"`.
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub tool_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for the tool input. Required for client tools, absent on
    /// server tools (which define their own input shape server-side).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<serde_json::Value>,
    /// Cache control breakpoint on this tool definition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
    /// Anthropic `defer_loading`: the tool's definition may be loaded lazily.
    /// Modeled so it survives conversion instead of being silently dropped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub defer_loading: Option<bool>,
}

/// Tool choice specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicToolChoice {
    /// Named tool: `{type: "tool", name: "..."}`
    /// Must be listed before Simple so serde tries the stricter shape first.
    Named(AnthropicToolChoiceNamed),
    /// Simple mode: "auto", "any", or "none".
    Simple(AnthropicToolChoiceSimple),
}

/// Simple tool choice modes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicToolChoiceSimple {
    #[serde(rename = "type")]
    pub choice_type: AnthropicToolChoiceMode,
    /// When true, the model will call tools one at a time instead of
    /// potentially issuing multiple tool calls in a single response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disable_parallel_tool_use: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicToolChoiceMode {
    Auto,
    Any,
    None,
    Tool,
}

/// Named tool choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicToolChoiceNamed {
    #[serde(rename = "type")]
    pub choice_type: AnthropicToolChoiceMode,
    pub name: String,
    /// When true, the model will call tools one at a time instead of
    /// potentially issuing multiple tool calls in a single response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disable_parallel_tool_use: Option<bool>,
}
/// Response body for `POST /v1/messages` (non-streaming).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessageResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub object_type: String,
    pub role: String,
    pub content: Vec<AnthropicResponseContentBlock>,
    pub model: String,
    pub stop_reason: Option<AnthropicStopReason>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

/// A content block in the response.
///
/// The Anthropic API returns up to 12 different block types. We model the
/// common ones explicitly and catch the rest as `Other` so the proxy can
/// forward them without losing data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicResponseContentBlock {
    #[serde(rename = "thinking")]
    Thinking { thinking: String, signature: String },
    /// Anthropic's own API never returns a `tool_result` — the client sends
    /// those. A server-side tool loop does return them, as the other half of
    /// each `tool_use` it resolved, so the response block set has to include
    /// one.
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        /// Omitted rather than `false`, matching how a client sends a
        /// successful result.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        citations: Option<Vec<serde_json::Value>>,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
    #[serde(rename = "server_tool_use")]
    ServerToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: serde_json::Value,
    },
    #[serde(rename = "web_search_tool_result")]
    WebSearchToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: serde_json::Value,
    },
    /// Catch-all for new/uncommon block types (web_fetch_tool_result,
    /// code_execution_tool_result, container_upload, etc.) so the proxy
    /// can serialize them back without data loss.
    #[serde(untagged)]
    Other(serde_json::Value),
}

/// Token usage information.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// Number of input tokens used to create a new cache entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    /// Number of input tokens read from the prompt cache (prefix cache hits).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
}

/// Reason the model stopped generating.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AnthropicStopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
    /// The model paused to yield control in an agentic loop, intending to
    /// continue in a subsequent turn. Used with extended thinking / tool use.
    PauseTurn,
    /// The model refused to generate content (safety refusal).
    Refusal,
}
/// SSE event types for the Anthropic streaming API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicStreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: AnthropicMessageResponse },

    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: u32,
        content_block: AnthropicResponseContentBlock,
    },

    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: u32, delta: AnthropicDelta },

    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: u32 },

    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: AnthropicMessageDeltaBody,
        usage: AnthropicUsage,
    },

    #[serde(rename = "message_stop")]
    MessageStop {},

    #[serde(rename = "ping")]
    Ping {},

    #[serde(rename = "error")]
    Error { error: AnthropicErrorBody },
}

/// Delta content in a streaming content_block_delta event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicDelta {
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
    /// Incremental signature for a thinking block (sent at the end).
    #[serde(rename = "signature_delta")]
    SignatureDelta { signature: String },
    /// Incremental citation attached to a text block.
    #[serde(rename = "citations_delta")]
    CitationsDelta { citation: serde_json::Value },
}

/// The delta body in a message_delta event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessageDeltaBody {
    pub stop_reason: Option<AnthropicStopReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
}
/// Anthropic API error response wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicErrorResponse {
    #[serde(rename = "type")]
    pub object_type: String,
    pub error: AnthropicErrorBody,
}

/// Error body within an error response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicErrorBody {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

impl AnthropicErrorResponse {
    /// Create an `invalid_request_error` response.
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            object_type: "error".to_string(),
            error: AnthropicErrorBody {
                error_type: "invalid_request_error".to_string(),
                message: message.into(),
            },
        }
    }

    /// Create an `api_error` (internal server error) response.
    pub fn api_error(message: impl Into<String>) -> Self {
        Self {
            object_type: "error".to_string(),
            error: AnthropicErrorBody {
                error_type: "api_error".to_string(),
                message: message.into(),
            },
        }
    }

    /// Create a `not_found_error` response.
    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            object_type: "error".to_string(),
            error: AnthropicErrorBody {
                error_type: "not_found_error".to_string(),
                message: message.into(),
            },
        }
    }
}
/// Request body for `POST /v1/messages/count_tokens`.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicCountTokensRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_system_prompt"
    )]
    pub system: Option<SystemContent>,
    #[serde(default)]
    pub tools: Option<Vec<AnthropicTool>>,
}

/// Response body for `POST /v1/messages/count_tokens`.
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicCountTokensResponse {
    pub input_tokens: u32,
}

impl AnthropicCountTokensRequest {
    /// Estimate input token count using a `len/3` heuristic.
    pub fn estimate_tokens(&self) -> u32 {
        let mut total_len: usize = 0;

        if let Some(system) = &self.system {
            total_len += system.text.len();
        }

        for msg in &self.messages {
            // Count role
            total_len += match msg.role {
                AnthropicRole::User => 4,
                AnthropicRole::Assistant => 9,
                AnthropicRole::System => 6,
            };
            // Count content
            match &msg.content {
                AnthropicMessageContent::Text { content } => total_len += content.len(),
                AnthropicMessageContent::Blocks { content } => {
                    for block in content {
                        total_len += estimate_block_len(block);
                    }
                }
            }
        }

        if let Some(tools) = &self.tools {
            for tool in tools {
                total_len += tool.name.len();
                if let Some(desc) = &tool.description {
                    total_len += desc.len();
                }
                if let Some(schema) = &tool.input_schema {
                    total_len += schema.to_string().len();
                }
            }
        }

        let tokens = total_len / 3;
        if tokens == 0 && total_len > 0 {
            1
        } else {
            tokens as u32
        }
    }
}

fn estimate_block_len(block: &AnthropicContentBlock) -> usize {
    match block {
        AnthropicContentBlock::Text { text, .. } => text.len(),
        AnthropicContentBlock::ToolUse { name, input, .. } => name.len() + input.to_string().len(),
        AnthropicContentBlock::ToolResult { content, .. } => content
            .as_ref()
            .map(|c| match c {
                ToolResultContent::Text(s) => s.len(),
                ToolResultContent::Blocks(blocks) => blocks
                    .iter()
                    .map(|b| match b {
                        ToolResultContentBlock::Text { text } => text.len(),
                        ToolResultContentBlock::Image { .. } => 256, // rough estimate for image metadata
                        ToolResultContentBlock::Document(doc) => {
                            doc.text().map(|t| t.len()).unwrap_or(256)
                        }
                        ToolResultContentBlock::SearchResult(result) => {
                            result.text().map(|t| t.len()).unwrap_or(0)
                        }
                        ToolResultContentBlock::Other(v) => v.to_string().len(),
                    })
                    .sum(),
            })
            .unwrap_or(0),
        AnthropicContentBlock::Thinking { thinking, .. } => thinking.len(),
        AnthropicContentBlock::RedactedThinking { data, .. } => data.len(),
        AnthropicContentBlock::ServerToolUse { name, input, .. } => {
            name.len() + input.to_string().len()
        }
        AnthropicContentBlock::WebSearchToolResult { content, .. } => content.to_string().len(),
        AnthropicContentBlock::Image { .. } => 256, // rough estimate for image metadata
        AnthropicContentBlock::Other(v) => v.to_string().len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `system` re-serializes in Anthropic's own wire shapes (string, or a text block carrying
    /// the cache_control), never as the parsed struct's `{"text": ...}` object — a consumer
    /// reading the re-serialized request must see the same system prompt the client sent.
    #[test]
    fn system_content_reserializes_in_wire_shape() {
        let req: AnthropicCreateMessageRequest = serde_json::from_value(serde_json::json!({
            "model": "m", "max_tokens": 1, "messages": [],
            "system": "You are helpful."
        }))
        .unwrap();
        let out = serde_json::to_value(&req).unwrap();
        assert_eq!(out["system"], "You are helpful.");

        let req: AnthropicCreateMessageRequest = serde_json::from_value(serde_json::json!({
            "model": "m", "max_tokens": 1, "messages": [],
            "system": [
                {"type": "text", "text": "A"},
                {"type": "text", "text": "B", "cache_control": {"type": "ephemeral"}}
            ]
        }))
        .unwrap();
        let out = serde_json::to_value(&req).unwrap();
        assert_eq!(
            out["system"],
            serde_json::json!([{"type": "text", "text": "A\nB", "cache_control": {"type": "ephemeral"}}])
        );
        // And it reads back as the same parsed content.
        let again: AnthropicCreateMessageRequest = serde_json::from_value(out).unwrap();
        let system = again.system.unwrap();
        assert_eq!(system.text, "A\nB");
        assert_eq!(
            serde_json::to_value(system.cache_control.unwrap()).unwrap()["type"],
            "ephemeral"
        );
    }

    #[test]
    fn message_content_errors_name_the_actual_problem() {
        let err = serde_json::from_value::<AnthropicMessage>(serde_json::json!({
            "role": "user",
            "content": 42
        }))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("must be a string or an array of content blocks"),
            "{err}"
        );
        assert!(err.contains("a number"), "{err}");

        // Strict shapes still parse.
        let text: AnthropicMessage = serde_json::from_value(serde_json::json!({
            "role": "user",
            "content": "hello"
        }))
        .unwrap();
        assert!(matches!(text.content, AnthropicMessageContent::Text { .. }));
        let blocks: AnthropicMessage = serde_json::from_value(serde_json::json!({
            "role": "user",
            "content": [{"type": "text", "text": "hello"}]
        }))
        .unwrap();
        assert!(matches!(
            blocks.content,
            AnthropicMessageContent::Blocks { .. }
        ));
    }

    #[test]
    fn response_tool_result_block_round_trips() {
        let success = serde_json::json!({
            "type": "tool_result",
            "tool_use_id": "srvtoolu_1",
            "content": "result body"
        });
        let block: AnthropicResponseContentBlock = serde_json::from_value(success.clone()).unwrap();
        assert!(matches!(
            block,
            AnthropicResponseContentBlock::ToolResult { ref tool_use_id, is_error: None, .. }
                if tool_use_id == "srvtoolu_1"
        ));
        // is_error omitted on success, matching how a client sends one.
        assert_eq!(serde_json::to_value(block).unwrap(), success);
    }

    /// Unknown top-level request fields survive a parse/serialize round trip in the client's
    /// order (the passthrough contract the shared crate builds on), while every modeled field
    /// still lands in its typed home.
    #[test]
    fn unmodeled_request_fields_round_trip_in_client_order() {
        let input = serde_json::json!({
            "model": "m",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "context_management": {"edits": []},
            "zeta_first": 1,
            "alpha_second": 2
        });
        let request: AnthropicCreateMessageRequest = serde_json::from_value(input.clone()).unwrap();
        assert_eq!(request.model, "m");
        assert_eq!(
            request.unmodeled.keys().collect::<Vec<_>>(),
            ["context_management", "zeta_first", "alpha_second"]
        );
        let output = serde_json::to_value(&request).unwrap();
        assert_eq!(output["context_management"], input["context_management"]);
        assert_eq!(output["zeta_first"], 1);
        assert_eq!(output["alpha_second"], 2);
        assert_eq!(
            serde_json::to_string(&request)
                .unwrap()
                .matches("\"model\"")
                .count(),
            1
        );
    }

    /// `defer_loading` is optional on the wire and omitted when absent; a present value parses
    /// and re-serializes.
    #[test]
    fn tool_defer_loading_is_optional_and_round_trips() {
        let without: AnthropicTool = serde_json::from_value(serde_json::json!({
            "name": "t", "input_schema": {"type": "object"}
        }))
        .unwrap();
        assert_eq!(without.defer_loading, None);
        assert!(
            serde_json::to_value(&without)
                .unwrap()
                .get("defer_loading")
                .is_none()
        );
        let with: AnthropicTool = serde_json::from_value(serde_json::json!({
            "name": "t", "input_schema": {"type": "object"}, "defer_loading": true
        }))
        .unwrap();
        assert_eq!(with.defer_loading, Some(true));
        assert_eq!(serde_json::to_value(&with).unwrap()["defer_loading"], true);
    }

    #[test]
    fn tool_result_blocks_parse_typed_and_round_trip() {
        let input = serde_json::json!([
            {"type": "text", "text": "Screenshot captured"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "aGVsbG8="
                }
            },
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "aGVsbG8="
                }
            },
            {
                "type": "search_result",
                "source": "https://example.com",
                "title": "Example",
                "content": [{"type": "text", "text": "hit"}]
            },
            {"type": "tool_reference", "tool_name": "mcp__slack__read_thread"}
        ]);
        let content: ToolResultContent = serde_json::from_value(input.clone()).unwrap();

        let ToolResultContent::Blocks(blocks) = &content else {
            panic!("expected content blocks");
        };
        assert!(matches!(blocks[1], ToolResultContentBlock::Image { .. }));
        assert!(matches!(
            &blocks[2],
            ToolResultContentBlock::Document(DocumentBlock {
                source: DocumentSource::Base64 { .. },
                ..
            })
        ));
        assert!(matches!(blocks[3], ToolResultContentBlock::SearchResult(_)));
        assert!(matches!(blocks[4], ToolResultContentBlock::Other(_)));
        assert_eq!(serde_json::to_value(content).unwrap(), input);

        let legacy: ToolResultContentBlock =
            serde_json::from_value(serde_json::json!({"text": "legacy"})).unwrap();
        assert!(matches!(legacy, ToolResultContentBlock::Text { .. }));

        // A document whose source shape is unknown degrades to Other and
        // round-trips byte-identically.
        let odd = serde_json::json!({"type": "document", "source": {"type": "mystery"}});
        let block: ToolResultContentBlock = serde_json::from_value(odd.clone()).unwrap();
        assert!(matches!(block, ToolResultContentBlock::Other(_)));
        assert_eq!(serde_json::to_value(block).unwrap(), odd);
    }
}
