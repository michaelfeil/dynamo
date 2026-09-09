// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Re-exports upstream async-openai chat types and defines inference-serving
// extensions on top. Types prefixed with `Dynamo` or entirely absent from the
// upstream spec are documented with the rationale for the extension.

use std::pin::Pin;

use derive_builder::Builder;
use futures::Stream;
use serde::{Deserialize, Serialize};
use url::Url;
use uuid::Uuid;

use crate::error::OpenAIError;

// ---------------------------------------------------------------------------
// Re-exports from upstream async-openai (unchanged types)
// ---------------------------------------------------------------------------
// These types are structurally identical to the upstream definitions.
// Consumers should use them via `dynamo_protocols::types::*` as before.

// Keep one-entry-per-line; rustfmt would otherwise repack this block into
// Mixed layout whenever an entry is added/removed, producing large noisy
// diffs (see the inline `// Builder types` comment that previously pinned
// this layout).
#[rustfmt::skip]
pub use async_openai::types::chat::{
    ChatCompletionAudio,
    ChatCompletionAudioFormat,
    ChatCompletionAudioVoice,
    ChatCompletionFunctionCall,
    ChatCompletionFunctions,
    ChatCompletionFunctionsArgs,
    ChatCompletionRequestAssistantMessageAudio,
    ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestAssistantMessageContentPart,
    ChatCompletionRequestDeveloperMessage,
    ChatCompletionRequestDeveloperMessageArgs,
    ChatCompletionRequestDeveloperMessageContent,
    ChatCompletionRequestDeveloperMessageContentPart,
    ChatCompletionRequestFunctionMessage,
    ChatCompletionRequestFunctionMessageArgs,
    ChatCompletionRequestMessageContentPartAudio,
    ChatCompletionRequestMessageContentPartRefusal,
    ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestSystemMessageContent,
    ChatCompletionRequestSystemMessageContentPart,
    ChatCompletionResponseMessageAudio,
    Choice,
    CompletionFinishReason,
    CompletionTokensDetails,
    CompletionUsage,
    FunctionObject,
    FunctionObjectArgs,
    InputAudio,
    InputAudioFormat,
    Logprobs,
    PredictionContent,
    PredictionContentContent,
    Prompt,
    PromptTokensDetails,
    ReasoningEffort,
    ResponseFormat,
    ResponseFormatJsonSchema,
    Role,
    ServiceTier,
    TopLogprobs,
    WebSearchContextSize,
    WebSearchLocation,
    WebSearchOptions,
    WebSearchUserLocation,
    WebSearchUserLocationType,
};

// ---------------------------------------------------------------------------
// Dynamo-owned override: ChatCompletionRequestToolMessageContent
// ---------------------------------------------------------------------------
// Upstream `async-openai` 0.34 restricts tool-message content parts to `Text`
// only (the OpenAPI spec says "For tool messages, only type `text` is
// supported"). Some OpenAI-compatible clients send multimodal tool-observation
// payloads, such as `image_url` parts inside tool-message `content` arrays.
// Upstream rejects those shapes at deserialization before Dynamo can decide how
// to handle them.
//
// Reuse Dynamo's existing request content-part enum rather than maintaining a
// second near-identical tool-message enum. This lets the request enter Dynamo's
// typed protocol world with the image/audio payload preserved; processor support
// can be added separately.
//
// When async-openai upstream supports multimodal tool content, delete this owned
// content type and re-add upstream tool-message types to the `pub use` block.

pub type ChatCompletionRequestToolMessageContentPart = ChatCompletionRequestUserMessageContentPart;

/// Dynamo-owned `ChatCompletionRequestToolMessageContent` referencing the
/// existing multimodal-aware request content part enum. Shape mirrors upstream
/// except that array parts are not restricted to text only.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum ChatCompletionRequestToolMessageContent {
    /// The text contents of the tool message.
    Text(String),
    /// An array of content parts with a defined type. Dynamo extends this to use
    /// the same content-part set accepted by user messages.
    Array(Vec<ChatCompletionRequestToolMessageContentPart>),
}

impl Default for ChatCompletionRequestToolMessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

/// Dynamo-owned `ChatCompletionRequestToolMessage` struct that references the
/// owned `ChatCompletionRequestToolMessageContent` so serde uses the
/// multimodal-aware content type. Shape mirrors upstream exactly.
#[derive(Debug, Deserialize, Serialize, Default, Clone, Builder, PartialEq)]
#[builder(name = "ChatCompletionRequestToolMessageArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct ChatCompletionRequestToolMessage {
    pub content: ChatCompletionRequestToolMessageContent,
    pub tool_call_id: String,
}

/// Dynamo-owned `ChatCompletionRequestSystemMessage`.
///
/// Extends upstream with:
/// - `content` is OPTIONAL: Kimi K3's official protocol sends system messages
///   that carry only a `tools` list and no `content`; upstream rejects those
///   at deserialization ("missing field `content`") before the worker can
///   render them.
/// - `tools`: message-level (dynamic) tool declarations, forwarded opaquely
///   to the worker (Kimi K3's `encoding_k3` renders them as a dynamic
///   tool-declare block). Workers/templates that don't read the key ignore
///   it harmlessly.
#[derive(Debug, Deserialize, Serialize, Default, Clone, Builder, PartialEq)]
#[builder(name = "ChatCompletionRequestSystemMessageArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct ChatCompletionRequestSystemMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<ChatCompletionRequestSystemMessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Message-level (dynamic) tool declarations, forwarded opaquely.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct ChatChoiceLogprobs {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<ChatCompletionTokenLogprob>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<Vec<ChatCompletionTokenLogprob>>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct ChatCompletionTokenLogprob {
    pub token: String,
    pub logprob: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_id: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<TopLogprobs>,
}

/// OpenAI stop configuration, with Dynamo's token-id stop extension.
///
/// The standard OpenAI shape accepts a string or string array. Dynamo also
/// accepts an integer array, e.g. `"stop": [576]`, to express token-id stop
/// conditions for tokenized in/out workflows. Strings like `"token_id:576"`
/// remain ordinary string stops; the `token_id:<id>` format is only an output
/// display format for logprobs.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum Stop {
    String(String),
    StringArray(Vec<String>),
    TokenIdArray(Vec<u32>),
}

impl Stop {
    pub fn strings(&self) -> Option<Vec<String>> {
        match self {
            Stop::String(s) => Some(vec![s.clone()]),
            Stop::StringArray(arr) => Some(arr.clone()),
            Stop::TokenIdArray(_) => None,
        }
    }

    pub fn token_ids(&self) -> Option<Vec<u32>> {
        match self {
            Stop::TokenIdArray(arr) => Some(arr.clone()),
            Stop::String(_) | Stop::StringArray(_) => None,
        }
    }
}

impl From<String> for Stop {
    fn from(value: String) -> Self {
        Stop::String(value)
    }
}

impl From<&str> for Stop {
    fn from(value: &str) -> Self {
        Stop::String(value.to_string())
    }
}

impl From<Vec<String>> for Stop {
    fn from(value: Vec<String>) -> Self {
        Stop::StringArray(value)
    }
}

impl From<Vec<u32>> for Stop {
    fn from(value: Vec<u32>) -> Self {
        Stop::TokenIdArray(value)
    }
}

impl From<async_openai::types::chat::StopConfiguration> for Stop {
    fn from(value: async_openai::types::chat::StopConfiguration) -> Self {
        match value {
            async_openai::types::chat::StopConfiguration::String(value) => Stop::String(value),
            async_openai::types::chat::StopConfiguration::StringArray(value) => {
                Stop::StringArray(value)
            }
        }
    }
}

// Upstream renamed FinishReason (streaming) -- re-export
pub use async_openai::types::chat::FinishReason;

// Upstream uses FunctionType where we used ChatCompletionToolType.
// Re-export both names for compatibility.
pub use async_openai::types::chat::FunctionType;

// ---------------------------------------------------------------------------
// Flexible `arguments` deserialisation helpers
// ---------------------------------------------------------------------------
// Some agent frameworks (e.g. LangChain, custom harnesses) send tool-call
// arguments as a pre-parsed JSON object instead of the canonical JSON
// string.  The helpers below normalise both representations to a `String` so
// downstream code never needs to branch on the wire format.

fn deserialize_arguments<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    let value = serde_json::Value::deserialize(deserializer)?;
    match value {
        serde_json::Value::String(s) => Ok(s),
        v @ serde_json::Value::Object(_) => {
            // serde_json::to_string on a Value is infallible
            Ok(serde_json::to_string(&v).unwrap())
        }
        other => Err(D::Error::custom(format!(
            "expected string or object for `arguments`, got {other}"
        ))),
    }
}

fn deserialize_arguments_opt<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    match value {
        None => Ok(None),
        Some(serde_json::Value::String(s)) => Ok(Some(s)),
        Some(v @ serde_json::Value::Object(_)) => serde_json::to_string(&v)
            .map(Some)
            .map_err(|e| D::Error::custom(e.to_string())),
        Some(other) => Err(D::Error::custom(format!(
            "expected string or object for `arguments`, got {other}"
        ))),
    }
}

/// The canonical reasoning-effort spellings. `none` turns thinking off; the
/// other six are levels, weakest to strongest.
///
/// One vocabulary with the serve side, whose copy is `VALID_REASONING_EFFORTS`
/// in the baseten repo's
/// `mp/baseten_dynamo/cache_aware_routing_trtllm/src/common/reasoning_effort.py`.
/// Which of these a given model accepts is that policy's business, not this
/// crate's.
pub const B10_REASONING_EFFORT_LEVELS: [&str; 7] =
    ["none", "minimal", "low", "medium", "high", "xhigh", "max"];

/// A reasoning effort as the client sent it.
///
/// Only the serve side holds the per-model policy that says which levels a
/// model distinguishes, so this type carries the request's value instead of
/// judging it: a canonical level becomes its variant, and anything else — an
/// unknown word, a boolean, a number — rides through in [`Self::Other`] byte
/// for byte, to be snapped onto the model's levels or refused there with a
/// message naming them.
///
/// Matching is exact, so `"Max"` is an `Other` rather than `Max`: case folding
/// is a client-spelling rule, and those belong with the policy that owns the
/// rest of them.
///
/// Deliberately no `Default`: picking a level is a policy decision, not a
/// parsing one.
#[derive(Clone, Debug, PartialEq)]
pub enum B10ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
    /// A value that is not one of [`B10_REASONING_EFFORT_LEVELS`], preserved
    /// exactly as received.
    Other(serde_json::Value),
}

impl B10ReasoningEffort {
    /// The canonical spelling, or `None` for a value that is not a level.
    pub fn as_level(&self) -> Option<&'static str> {
        match self {
            Self::None => Some("none"),
            Self::Minimal => Some("minimal"),
            Self::Low => Some("low"),
            Self::Medium => Some("medium"),
            Self::High => Some("high"),
            Self::Xhigh => Some("xhigh"),
            Self::Max => Some("max"),
            Self::Other(_) => None,
        }
    }

    /// The effort a client sent, canonical or not. Infallible by construction:
    /// judging a spelling takes the per-model policy this crate does not have.
    pub fn from_client_value(value: serde_json::Value) -> Self {
        let level = value.as_str().and_then(Self::from_level);
        level.unwrap_or(Self::Other(value))
    }

    /// The level a canonical spelling names, or `None` for anything else.
    fn from_level(s: &str) -> Option<Self> {
        match s {
            "none" => Some(Self::None),
            "minimal" => Some(Self::Minimal),
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            "xhigh" => Some(Self::Xhigh),
            "max" => Some(Self::Max),
            _ => None,
        }
    }

    /// The nearest value the async-openai `ReasoningEffort` enum can spell.
    ///
    /// For the `/v1/responses` echo only -- the request keeps whatever the
    /// client sent, since [`Serialize`] writes `Other` back verbatim. Two
    /// values narrow here because the response type cannot hold them: `max`
    /// echoes as `xhigh`, the strongest that enum has, and a non-level echoes
    /// nothing, so the field is omitted rather than reporting an effort the
    /// client did not ask for.
    pub fn to_async_openai(&self) -> Option<ReasoningEffort> {
        match self {
            Self::None => Some(ReasoningEffort::None),
            Self::Minimal => Some(ReasoningEffort::Minimal),
            Self::Low => Some(ReasoningEffort::Low),
            Self::Medium => Some(ReasoningEffort::Medium),
            Self::High => Some(ReasoningEffort::High),
            Self::Xhigh | Self::Max => Some(ReasoningEffort::Xhigh),
            Self::Other(_) => None,
        }
    }
}

impl Serialize for B10ReasoningEffort {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Other(value) => value.serialize(serializer),
            Self::None => serializer.serialize_str("none"),
            Self::Minimal => serializer.serialize_str("minimal"),
            Self::Low => serializer.serialize_str("low"),
            Self::Medium => serializer.serialize_str("medium"),
            Self::High => serializer.serialize_str("high"),
            Self::Xhigh => serializer.serialize_str("xhigh"),
            Self::Max => serializer.serialize_str("max"),
        }
    }
}

impl<'de> Deserialize<'de> for B10ReasoningEffort {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self::from_client_value(serde_json::Value::deserialize(
            deserializer,
        )?))
    }
}

// ---------------------------------------------------------------------------
// FunctionCall / FunctionCallStream — local definitions with flexible deser
// ---------------------------------------------------------------------------
// Upstream `async-openai` only accepts a JSON string for `arguments`.
// We define these locally so we can attach `#[serde(deserialize_with)]` and
// accept both string and object representations on the wire.

/// The name and arguments of a function that should be called.
///
/// Accepts `arguments` as either a JSON string (`"{\"key\":\"value\"}"`) or a
/// JSON object (`{"key": "value"}`); both are normalised to a JSON string
/// on deserialisation so callers always see the canonical form.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Default)]
pub struct FunctionCall {
    pub name: String,
    #[serde(deserialize_with = "deserialize_arguments")]
    pub arguments: String,
}

/// Streaming variant of [`FunctionCall`] where both fields are optional.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Default)]
pub struct FunctionCallStream {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(
        default,
        deserialize_with = "deserialize_arguments_opt",
        skip_serializing_if = "Option::is_none"
    )]
    pub arguments: Option<String>,
}

/// Streaming tool-call chunk.
///
/// Defined locally (instead of re-exporting from upstream) because its
/// `function` field references our local [`FunctionCallStream`] with the
/// flexible `arguments` deserialiser.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Default)]
pub struct ChatCompletionMessageToolCallChunk {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#type: Option<FunctionType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallStream>,
}

// ---------------------------------------------------------------------------
// Types with structural differences from upstream (kept locally)
// ---------------------------------------------------------------------------

/// Image detail level. Kept locally because upstream uses different field types in ImageUrl.
#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    #[default]
    Auto,
    Low,
    High,
    Original,
}

/// Image content part -- uses our extended `ImageUrl` with `url::Url` and `uuid`.
#[derive(Debug, Serialize, Deserialize, Clone, Builder, PartialEq)]
#[builder(name = "ChatCompletionRequestMessageContentPartImageArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option))]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct ChatCompletionRequestMessageContentPartImage {
    pub image_url: ImageUrl,
}

/// Image URL with `url::Url` type and optional UUID.
///
/// Differs from upstream: uses `url::Url` instead of `String`, adds `uuid` field
/// for tracking multimodal assets through the pipeline.
#[derive(Debug, Serialize, Deserialize, Clone, Builder, PartialEq)]
#[builder(name = "ImageUrlArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option))]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct ImageUrl {
    pub url: Url,
    pub detail: Option<ImageDetail>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uuid: Option<Uuid>,
}

#[derive(Clone, Serialize, Default, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ChatCompletionToolType {
    #[default]
    Function,
}

#[derive(Clone, Serialize, Default, Debug, Deserialize, PartialEq)]
pub struct FunctionName {
    pub name: String,
}

#[derive(Clone, Serialize, Default, Debug, Deserialize, PartialEq)]
pub struct ChatCompletionNamedToolChoice {
    pub r#type: ChatCompletionToolType,
    pub function: FunctionName,
}

fn default_function_type() -> FunctionType {
    FunctionType::Function
}

/// Tool call kept locally to preserve `type: "function"` in unary request/response payloads.
///
/// Differs from upstream: `type` is serialized by default and also defaults to
/// `function` when omitted during deserialization, preserving compatibility with
/// both Dynamo's historical wire format and upstream spec-compliant inputs.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ChatCompletionMessageToolCall {
    pub id: String,
    #[serde(default = "default_function_type")]
    pub r#type: FunctionType,
    pub function: FunctionCall,
}

/// Tool choice enum kept locally because upstream changed variant names.
#[derive(Clone, Serialize, Default, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ChatCompletionToolChoiceOption {
    #[default]
    None,
    Auto,
    Required,
    #[serde(untagged)]
    Named(ChatCompletionNamedToolChoice),
}

#[derive(Clone, Serialize, Default, Debug, Builder, Deserialize, PartialEq)]
#[builder(name = "ChatCompletionToolArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct ChatCompletionTool {
    #[builder(default = "ChatCompletionToolType::Function")]
    pub r#type: ChatCompletionToolType,
    pub function: FunctionObject,
}

// ---------------------------------------------------------------------------
// Inference-serving extensions (not in upstream)
// ---------------------------------------------------------------------------

/// Matched stop condition from the backend.
///
/// Inference backends (vLLM, SGLang) report which stop condition triggered:
/// - `String`: a matched user-provided stop sequence
/// - `Int`: a matched stop token ID
/// - `IntArray`: matched stop token IDs reported as a sequence
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum StopReason {
    String(String),
    Int(i64),
    IntArray(Vec<i64>),
}

/// Reasoning content from a previous assistant turn.
///
/// Deserializes from either:
/// - A plain string: `"reasoning_content": "thinking..."` -> `Text("thinking...")`
/// - An array of strings: `"reasoning_content": ["seg1", "seg2"]` -> `Segments(["seg1", "seg2"])`
///
/// The `Segments` variant preserves interleaved reasoning order needed for KV cache-correct
/// context reconstruction. `segments[i]` is the reasoning that preceded `tool_calls[i]`;
/// `segments[tool_calls.len()]` is any trailing reasoning after the last tool call.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(untagged)]
pub enum ReasoningContent {
    /// Flat string -- single reasoning block or legacy backward-compat form.
    Text(String),
    /// Interleaved segments. segments[i] precedes tool_calls[i];
    /// segments[N] is trailing reasoning after the last tool call.
    Segments(Vec<String>),
}

impl ReasoningContent {
    /// Join all segments (or return text as-is) into a single flat string.
    pub fn to_flat_string(&self) -> String {
        match self {
            ReasoningContent::Text(s) => s.clone(),
            ReasoningContent::Segments(segs) => segs
                .iter()
                .filter(|s| !s.is_empty())
                .cloned()
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }

    /// Returns the segments if this is the `Segments` variant, `None` for `Text`.
    pub fn segments(&self) -> Option<&[String]> {
        match self {
            ReasoningContent::Segments(segs) => Some(segs),
            ReasoningContent::Text(_) => None,
        }
    }
}

// -- Multimodal content types for responses (not in upstream) --

/// Response content part for text in assistant messages
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ChatCompletionResponseContentPartText {
    pub text: String,
}

/// Response content part for image URLs in assistant messages
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ChatCompletionResponseContentPartImageUrl {
    pub image_url: ImageUrlResponse,
}

/// Response content part for video URLs in assistant messages
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ChatCompletionResponseContentPartVideoUrl {
    pub video_url: VideoUrlResponse,
}

/// Response content part for audio URLs in assistant messages
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ChatCompletionResponseContentPartAudioUrl {
    pub audio_url: AudioUrlResponse,
}

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ImageUrlResponse {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct VideoUrlResponse {
    pub url: String,
}

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct AudioUrlResponse {
    pub url: String,
}

/// Content parts for assistant responses supporting multiple modalities
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatCompletionResponseContentPart {
    Text(ChatCompletionResponseContentPartText),
    ImageUrl(ChatCompletionResponseContentPartImageUrl),
    VideoUrl(ChatCompletionResponseContentPartVideoUrl),
    AudioUrl(ChatCompletionResponseContentPartAudioUrl),
}

/// Assistant message content -- can be a simple string or multimodal content parts.
///
/// Upstream uses `Option<String>` for the content field. We extend this to
/// support multimodal responses (text + images + video + audio) from backends
/// like vLLM that can return non-text content.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ChatCompletionMessageContent {
    /// Simple text content (backward compatible)
    Text(String),
    /// Array of content parts (for multimodal responses)
    Parts(Vec<ChatCompletionResponseContentPart>),
}

// -- Multimodal input types (video/audio URL support, not in upstream) --

#[derive(Debug, Serialize, Deserialize, Clone, Builder, PartialEq)]
#[builder(name = "VideoUrlArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option))]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct VideoUrl {
    pub url: Url,
    pub detail: Option<ImageDetail>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uuid: Option<Uuid>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Builder, PartialEq)]
#[builder(name = "ChatCompletionRequestMessageContentPartVideoArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option))]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct ChatCompletionRequestMessageContentPartVideo {
    pub video_url: VideoUrl,
}

#[derive(Debug, Serialize, Deserialize, Clone, Builder, PartialEq)]
#[builder(name = "AudioUrlArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option))]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct AudioUrl {
    pub url: Url,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uuid: Option<Uuid>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Builder, PartialEq)]
#[builder(name = "ChatCompletionRequestMessageContentPartAudioUrlArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option))]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct ChatCompletionRequestMessageContentPartAudioUrl {
    pub audio_url: AudioUrl,
}

// -- Extended request/response types --

/// User message content -- references our extended content part enum.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum ChatCompletionRequestUserMessageContent {
    Text(String),
    Array(Vec<ChatCompletionRequestUserMessageContentPart>),
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, Builder, PartialEq)]
#[builder(name = "ChatCompletionRequestUserMessageArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct ChatCompletionRequestUserMessage {
    pub content: ChatCompletionRequestUserMessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Default for ChatCompletionRequestUserMessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl From<&str> for ChatCompletionRequestUserMessageContent {
    fn from(value: &str) -> Self {
        Self::Text(value.into())
    }
}

impl From<String> for ChatCompletionRequestUserMessageContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<Vec<ChatCompletionRequestUserMessageContentPart>>
    for ChatCompletionRequestUserMessageContent
{
    fn from(value: Vec<ChatCompletionRequestUserMessageContentPart>) -> Self {
        Self::Array(value)
    }
}

/// User message content part with video and audio URL support.
///
/// Extends upstream `ChatCompletionRequestUserMessageContentPart` with:
/// - `VideoUrl`: video input for multimodal models
/// - `AudioUrl`: audio URL input (distinct from base64 InputAudio)
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ChatCompletionRequestUserMessageContentPart {
    Text(ChatCompletionRequestMessageContentPartText),
    ImageUrl(ChatCompletionRequestMessageContentPartImage),
    VideoUrl(ChatCompletionRequestMessageContentPartVideo),
    AudioUrl(ChatCompletionRequestMessageContentPartAudioUrl),
    InputAudio(ChatCompletionRequestMessageContentPartAudio),
}

/// Assistant message with reasoning content support.
///
/// Extends upstream `ChatCompletionRequestAssistantMessage` with:
/// - `reasoning_content`: interleaved reasoning segments for KV cache correctness
///   (DeepSeek-R1, QwQ models)
#[derive(Debug, Serialize, Deserialize, Default, Clone, Builder, PartialEq)]
#[builder(name = "ChatCompletionRequestAssistantMessageArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct ChatCompletionRequestAssistantMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<ChatCompletionRequestAssistantMessageContent>,
    /// Reasoning content from a previous assistant turn.
    // b10: a customer still wants to send either "reasoning" or "reasoning_content"; accept both.
    #[serde(alias = "reasoning", skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<ReasoningContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<ChatCompletionRequestAssistantMessageAudio>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatCompletionMessageToolCall>>,
    /// Kimi-style assistant prefill marker (official Moonshot protocol):
    /// when true on the final assistant message, generation continues inside
    /// that message's open channel. Forwarded opaquely to the worker.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partial: Option<bool>,
    #[deprecated]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
}

/// Chat completion request message enum.
///
/// Redefined to use our extended `ChatCompletionRequestAssistantMessage`
/// (with reasoning_content) and `ChatCompletionRequestUserMessage`
/// (which references our extended content parts with video/audio).
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "role")]
#[serde(rename_all = "lowercase")]
pub enum ChatCompletionRequestMessage {
    Developer(ChatCompletionRequestDeveloperMessage),
    System(ChatCompletionRequestSystemMessage),
    User(ChatCompletionRequestUserMessage),
    Assistant(ChatCompletionRequestAssistantMessage),
    Tool(ChatCompletionRequestToolMessage),
    Function(ChatCompletionRequestFunctionMessage),
}

/// Response tier enum for responses (distinct from request `ServiceTier`).
///
/// Not in upstream -- backends report which tier actually served the request.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ServiceTierResponse {
    Scale,
    Default,
    Flex,
    Priority,
}

/// Chat completion response message with multimodal content and reasoning.
///
/// Extends upstream `ChatCompletionResponseMessage` with:
/// - `content`: `Option<ChatCompletionMessageContent>` (multimodal) instead of `Option<String>`
/// - `reasoning_content`: model reasoning output (DeepSeek-R1, QwQ)
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct ChatCompletionResponseMessage {
    /// Always serialized (as `null` when None) so clients can rely on the
    /// `content` key being present alongside `reasoning_content` or
    /// `tool_calls`. Matches the upstream OpenAI API shape (DGH-651).
    pub content: Option<ChatCompletionMessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatCompletionMessageToolCall>>,
    pub role: Role,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[deprecated]
    pub function_call: Option<FunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<ChatCompletionResponseMessageAudio>,
    /// Reasoning content produced by the model (DeepSeek-R1, QwQ).
    /// Skipped when `None`: a non-thinking model's response must not carry a
    /// `reasoning_content` key at all — OpenAI has no such field, so emitting
    /// `null` invents one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

/// Stream options with per-chunk usage reporting.
///
/// Extends upstream `ChatCompletionStreamOptions` with:
/// - `continuous_usage_stats`: emit usage in every chunk, not just the final one
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
pub struct ChatCompletionStreamOptions {
    pub include_usage: bool,
    /// When true, usage statistics are included in every streaming chunk.
    /// Backends like vLLM/SGLang support this for real-time token counting.
    #[serde(default)]
    pub continuous_usage_stats: bool,
}

/// Chat completion request with multimodal processor support.
///
/// Extends upstream `CreateChatCompletionRequest` with:
/// - `mm_processor_kwargs`: multimodal processor configuration (vLLM-specific)
/// - Uses our extended `ChatCompletionRequestMessage` (with reasoning, video/audio)
/// - Uses our extended `ChatCompletionStreamOptions` (with continuous_usage_stats)
#[derive(Clone, Serialize, Default, Debug, Builder, Deserialize, PartialEq)]
#[builder(name = "CreateChatCompletionRequestArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct CreateChatCompletionRequest {
    pub messages: Vec<ChatCompletionRequestMessage>,
    pub model: String,
    /// Multimodal processor configuration (vLLM-specific)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mm_processor_kwargs: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<B10ReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<std::collections::HashMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
    #[deprecated]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<async_openai::types::chat::ResponseModalities>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction: Option<PredictionContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<ChatCompletionAudio>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Stop>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<ChatCompletionStreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ChatCompletionTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ChatCompletionToolChoiceOption>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[deprecated]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<ChatCompletionFunctionCall>,
    #[deprecated]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub functions: Option<Vec<ChatCompletionFunctions>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search_options: Option<WebSearchOptions>,
}

/// Chat choice with extended response message.
///
/// Uses our `ChatCompletionResponseMessage` (multimodal content + reasoning).
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatCompletionResponseMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatChoiceLogprobs>,
}

/// Non-streaming chat completion response.
#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct CreateChatCompletionResponse {
    pub id: String,
    pub choices: Vec<ChatChoice>,
    pub created: u32,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTierResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    pub object: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<CompletionUsage>,
}

pub type ChatCompletionResponseStream =
    Pin<Box<dyn Stream<Item = Result<CreateChatCompletionStreamResponse, OpenAIError>> + Send>>;

/// Streaming delta with reasoning content.
///
/// Extends upstream `ChatCompletionStreamResponseDelta` with:
/// - `content`: `Option<ChatCompletionMessageContent>` (multimodal) instead of `Option<String>`
/// - `reasoning_content`: streaming reasoning tokens (DeepSeek-R1, QwQ)
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct ChatCompletionStreamResponseDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<ChatCompletionMessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<ChatCompletionStreamResponseDeltaFunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatCompletionMessageToolCallChunk>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    /// Streaming reasoning content (DeepSeek-R1, QwQ models).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct ChatCompletionStreamResponseDeltaFunctionCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(
        default,
        deserialize_with = "deserialize_arguments_opt",
        skip_serializing_if = "Option::is_none"
    )]
    pub arguments: Option<String>,
}

/// Streaming chat choice.
#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct ChatChoiceStream {
    pub index: u32,
    pub delta: ChatCompletionStreamResponseDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatChoiceLogprobs>,
}

/// Streaming chat completion response with extended choices.
#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct CreateChatCompletionStreamResponse {
    pub id: String,
    pub choices: Vec<ChatChoiceStream>,
    pub created: u32,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTierResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    pub object: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<CompletionUsage>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_message_omits_absent_reasoning_content() {
        #[allow(deprecated)]
        let message = ChatCompletionResponseMessage {
            content: Some(ChatCompletionMessageContent::Text("hi".to_string())),
            refusal: None,
            tool_calls: None,
            role: Role::Assistant,
            function_call: None,
            audio: None,
            reasoning_content: None,
        };
        let value = serde_json::to_value(message).unwrap();
        assert!(value.get("reasoning_content").is_none());
    }

    #[test]
    fn stop_accepts_token_id_array() {
        let stop: Stop = serde_json::from_value(serde_json::json!([32, 34])).unwrap();

        assert_eq!(stop, Stop::TokenIdArray(vec![32, 34]));
    }

    #[test]
    fn stop_accepts_string_and_string_array() {
        let stop: Stop = serde_json::from_value(serde_json::json!(" The")).unwrap();

        assert_eq!(stop, Stop::String(" The".to_string()));

        let stop: Stop = serde_json::from_value(serde_json::json!(["A", "B"])).unwrap();

        assert_eq!(
            stop,
            Stop::StringArray(vec!["A".to_string(), "B".to_string()])
        );
    }

    #[test]
    fn stop_token_id_display_string_remains_string_stop() {
        let stop: Stop = serde_json::from_value(serde_json::json!("token_id:576")).unwrap();

        assert_eq!(stop, Stop::String("token_id:576".to_string()));

        let stop: Stop = serde_json::from_value(serde_json::json!(["token_id:576"])).unwrap();

        assert_eq!(stop, Stop::StringArray(vec!["token_id:576".to_string()]));
    }

    #[test]
    fn stop_rejects_single_token_id() {
        let result = serde_json::from_value::<Stop>(serde_json::json!(576));

        assert!(result.is_err());
    }

    #[test]
    fn stop_converts_from_upstream_stop_configuration() {
        let upstream =
            async_openai::types::chat::StopConfiguration::StringArray(vec!["END".to_string()]);

        assert_eq!(
            Stop::from(upstream),
            Stop::StringArray(vec!["END".to_string()])
        );
    }

    #[test]
    fn image_url_detail_accepts_original() {
        // "original" is a Baseten docs extension; rejecting it fails the whole
        // untagged ChatCompletionRequestUserMessageContent match with an opaque 400.
        let content: ChatCompletionRequestUserMessageContent =
            serde_json::from_value(serde_json::json!([
                {"type": "text", "text": "Reply OK."},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png", "detail": "original"}}
            ]))
            .unwrap();

        let ChatCompletionRequestUserMessageContent::Array(parts) = content else {
            panic!("expected content part array");
        };
        let ChatCompletionRequestUserMessageContentPart::ImageUrl(image) = &parts[1] else {
            panic!("expected image_url part");
        };
        assert_eq!(image.image_url.detail, Some(ImageDetail::Original));
    }

    #[test]
    fn image_detail_original_serializes_lowercase() {
        assert_eq!(
            serde_json::to_value(ImageDetail::Original).unwrap(),
            serde_json::json!("original")
        );
    }

    #[test]
    fn tool_call_defaults_type_on_deserialize() {
        let tool_call: ChatCompletionMessageToolCall = serde_json::from_value(serde_json::json!({
            "id": "call_123",
            "function": {
                "name": "get_weather",
                "arguments": "{\"location\":\"SF\"}"
            }
        }))
        .unwrap();

        assert_eq!(tool_call.r#type, FunctionType::Function);
    }

    #[test]
    fn tool_call_serializes_type_for_wire_compat() {
        let tool_call = ChatCompletionMessageToolCall {
            id: "call_123".into(),
            r#type: FunctionType::Function,
            function: FunctionCall {
                name: "get_weather".into(),
                arguments: "{\"location\":\"SF\"}".into(),
            },
        };

        let json = serde_json::to_value(tool_call).unwrap();
        assert_eq!(json["type"], "function");
    }

    // -- dict-format arguments tests --

    #[test]
    fn function_call_accepts_string_arguments() {
        let fc: FunctionCall = serde_json::from_value(serde_json::json!({
            "name": "get_weather",
            "arguments": "{\"location\":\"SF\"}"
        }))
        .unwrap();
        assert_eq!(fc.arguments, "{\"location\":\"SF\"}");
    }

    #[test]
    fn function_call_accepts_dict_arguments() {
        let fc: FunctionCall = serde_json::from_value(serde_json::json!({
            "name": "get_weather",
            "arguments": {"location": "SF"}
        }))
        .unwrap();
        assert_eq!(fc.arguments, "{\"location\":\"SF\"}");
    }

    #[test]
    fn function_call_rejects_integer_arguments() {
        let result = serde_json::from_value::<FunctionCall>(serde_json::json!({
            "name": "f",
            "arguments": 42
        }));
        assert!(result.is_err());
    }

    #[test]
    fn function_call_rejects_boolean_arguments() {
        let result = serde_json::from_value::<FunctionCall>(serde_json::json!({
            "name": "f",
            "arguments": true
        }));
        assert!(result.is_err());
    }

    #[test]
    fn function_call_rejects_null_arguments() {
        let result = serde_json::from_value::<FunctionCall>(serde_json::json!({
            "name": "f",
            "arguments": null
        }));
        assert!(result.is_err());
    }

    #[test]
    fn function_call_rejects_array_arguments() {
        let result = serde_json::from_value::<FunctionCall>(serde_json::json!({
            "name": "f",
            "arguments": [1, 2, 3]
        }));
        assert!(result.is_err());
    }

    #[test]
    fn function_call_stream_null_arguments_produces_none() {
        let fcs: FunctionCallStream = serde_json::from_value(serde_json::json!({
            "name": "f",
            "arguments": null
        }))
        .unwrap();
        assert_eq!(fcs.arguments, None);
    }

    #[test]
    fn function_call_stream_rejects_integer_arguments() {
        let result = serde_json::from_value::<FunctionCallStream>(serde_json::json!({
            "name": "f",
            "arguments": 42
        }));
        assert!(result.is_err());
    }

    #[test]
    fn function_call_stream_rejects_boolean_arguments() {
        let result = serde_json::from_value::<FunctionCallStream>(serde_json::json!({
            "name": "f",
            "arguments": true
        }));
        assert!(result.is_err());
    }

    #[test]
    fn function_call_stream_accepts_dict_arguments() {
        let fcs: FunctionCallStream = serde_json::from_value(serde_json::json!({
            "name": "get_weather",
            "arguments": {"location": "SF"}
        }))
        .unwrap();
        assert_eq!(fcs.arguments.as_deref(), Some("{\"location\":\"SF\"}"));
    }

    #[test]
    fn function_call_stream_accepts_null_arguments() {
        let fcs: FunctionCallStream = serde_json::from_value(serde_json::json!({
            "name": "get_weather"
        }))
        .unwrap();
        assert_eq!(fcs.arguments, None);
    }

    #[test]
    fn tool_call_with_dict_arguments_roundtrip() {
        let tc: ChatCompletionMessageToolCall = serde_json::from_value(serde_json::json!({
            "id": "call_abc",
            "type": "function",
            "function": {
                "name": "search",
                "arguments": {"query": "hello", "limit": 10}
            }
        }))
        .unwrap();
        // Compare as parsed JSON values since key order is non-deterministic
        let parsed: serde_json::Value = serde_json::from_str(&tc.function.arguments).unwrap();
        assert_eq!(parsed, serde_json::json!({"query": "hello", "limit": 10}));
        // Re-serialisation produces a string, not an object
        let json = serde_json::to_value(&tc).unwrap();
        assert!(json["function"]["arguments"].is_string());
    }

    #[test]
    fn stream_delta_function_call_accepts_dict_arguments() {
        let delta: ChatCompletionStreamResponseDeltaFunctionCall =
            serde_json::from_value(serde_json::json!({
                "name": "get_weather",
                "arguments": {"location": "SF"}
            }))
            .unwrap();
        assert_eq!(delta.arguments.as_deref(), Some("{\"location\":\"SF\"}"));
    }

    #[test]
    fn chat_stream_response_omits_null_fields_recursively() {
        let response = CreateChatCompletionStreamResponse {
            id: "chatcmpl-123".to_string(),
            choices: vec![ChatChoiceStream {
                index: 0,
                delta: ChatCompletionStreamResponseDelta {
                    content: Some(ChatCompletionMessageContent::Text("<think>".to_string())),
                    function_call: None,
                    tool_calls: Some(vec![ChatCompletionMessageToolCallChunk {
                        index: 0,
                        id: None,
                        r#type: None,
                        function: Some(FunctionCallStream {
                            name: None,
                            arguments: Some("{\"query\":\"weather\"}".to_string()),
                        }),
                    }]),
                    role: Some(Role::Assistant),
                    refusal: None,
                    reasoning_content: None,
                },
                finish_reason: None,
                logprobs: None,
            }],
            created: 1,
            model: "test-model".to_string(),
            service_tier: None,
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
        };

        let value = serde_json::to_value(response).expect("serialize response");
        let choice = &value["choices"][0];
        let delta = &choice["delta"];
        let tool_call = &delta["tool_calls"][0];
        let function = &tool_call["function"];

        assert_eq!(delta["content"], "<think>");
        assert_eq!(delta["role"], "assistant");
        assert_eq!(function["arguments"], "{\"query\":\"weather\"}");
        assert!(value.get("service_tier").is_none());
        assert!(value.get("system_fingerprint").is_none());
        assert!(value.get("usage").is_none());
        assert!(choice.get("finish_reason").is_none());
        assert!(choice.get("logprobs").is_none());
        assert!(delta.get("function_call").is_none());
        assert!(delta.get("refusal").is_none());
        assert!(delta.get("reasoning_content").is_none());
        assert!(tool_call.get("id").is_none());
        assert!(tool_call.get("type").is_none());
        assert!(function.get("name").is_none());
    }

    #[test]
    fn chat_logprobs_include_token_id_when_present() {
        let logprobs = ChatChoiceLogprobs {
            content: Some(vec![ChatCompletionTokenLogprob {
                token: " streaming".to_string(),
                logprob: -1.2885475,
                token_id: Some(27098),
                bytes: Some(vec![32, 115, 116, 114, 101, 97, 109, 105, 110, 103]),
                top_logprobs: vec![],
            }]),
            refusal: None,
        };

        let value = serde_json::to_value(logprobs).expect("serialize logprobs");
        assert_eq!(value["content"][0]["token_id"], 27098);
    }

    #[test]
    fn chat_stream_response_keeps_present_tool_call_fields() {
        let response = CreateChatCompletionStreamResponse {
            id: "chatcmpl-123".to_string(),
            choices: vec![ChatChoiceStream {
                index: 0,
                delta: ChatCompletionStreamResponseDelta {
                    content: None,
                    function_call: None,
                    tool_calls: Some(vec![ChatCompletionMessageToolCallChunk {
                        index: 0,
                        id: Some("call_123".to_string()),
                        r#type: Some(FunctionType::Function),
                        function: Some(FunctionCallStream {
                            name: Some("search".to_string()),
                            arguments: Some("{\"query\":\"weather\"}".to_string()),
                        }),
                    }]),
                    role: None,
                    refusal: None,
                    reasoning_content: None,
                },
                finish_reason: None,
                logprobs: None,
            }],
            created: 1,
            model: "test-model".to_string(),
            service_tier: None,
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
        };

        let value = serde_json::to_value(response).expect("serialize response");
        let tool_call = &value["choices"][0]["delta"]["tool_calls"][0];

        assert_eq!(tool_call["id"], "call_123");
        assert_eq!(tool_call["type"], "function");
        assert_eq!(tool_call["function"]["name"], "search");
        assert_eq!(
            tool_call["function"]["arguments"],
            "{\"query\":\"weather\"}"
        );
    }

    // --- tool message multimodal content ---

    #[test]
    fn tool_message_image_content_deserializes_into_request_message() {
        let json = serde_json::json!([
            {"role": "user", "content": "Take a screenshot."},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "screenshot", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": [
                {"type": "text", "text": "Captured frame:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
            ]}
        ]);
        let msgs: Vec<ChatCompletionRequestMessage> = serde_json::from_value(json).unwrap();
        assert_eq!(msgs.len(), 3);
        let ChatCompletionRequestMessage::Tool(tool) = &msgs[2] else {
            panic!("expected tool message");
        };
        let ChatCompletionRequestToolMessageContent::Array(parts) = &tool.content else {
            panic!("expected tool message content array");
        };
        assert!(matches!(
            parts[1],
            ChatCompletionRequestUserMessageContentPart::ImageUrl(_)
        ));
    }

    #[test]
    fn every_canonical_level_round_trips_to_its_variant() {
        for level in B10_REASONING_EFFORT_LEVELS {
            let parsed: B10ReasoningEffort =
                serde_json::from_value(serde_json::json!(level)).unwrap();
            assert_eq!(parsed.as_level(), Some(level), "level {level}");
            assert_eq!(
                serde_json::to_value(&parsed).unwrap(),
                serde_json::json!(level),
                "level {level} re-serializes"
            );
        }
    }

    #[test]
    fn off_ladder_values_are_carried_verbatim() {
        // Values with no variant on the upstream enum: spellings clients send,
        // and the non-string shapes a client can put in the field.
        for raw in [
            serde_json::json!("ultra"),
            serde_json::json!("persistent"),
            serde_json::json!("turbo"),
            serde_json::json!("Max"),
            serde_json::json!(""),
            serde_json::json!(true),
            serde_json::json!(4096),
        ] {
            let parsed: B10ReasoningEffort = serde_json::from_value(raw.clone()).unwrap();
            assert_eq!(parsed, B10ReasoningEffort::Other(raw.clone()), "{raw}");
            assert_eq!(parsed.as_level(), None, "{raw}");
            assert_eq!(
                serde_json::to_value(&parsed).unwrap(),
                raw,
                "{raw} re-serializes byte for byte"
            );
        }
    }

    #[test]
    fn only_max_folds_when_echoed_through_the_upstream_enum() {
        for level in B10_REASONING_EFFORT_LEVELS {
            let parsed: B10ReasoningEffort =
                serde_json::from_value(serde_json::json!(level)).unwrap();
            let echoed = parsed.to_async_openai().expect("a level always echoes");
            let expected = if level == "max" { "xhigh" } else { level };
            assert_eq!(
                serde_json::to_value(echoed).unwrap(),
                serde_json::json!(expected),
                "level {level}"
            );
        }
    }

    #[test]
    fn reasoning_effort_is_absent_rather_than_defaulted_when_unset() {
        // The upstream enum defaults to `medium`; a request that named no
        // effort must stay distinguishable from one that asked for `medium`,
        // or the policy's own default can never apply.
        let req: CreateChatCompletionRequest =
            serde_json::from_value(serde_json::json!({"messages": [], "model": "m"})).unwrap();
        assert_eq!(req.reasoning_effort, None);
        let body = serde_json::to_value(&req).unwrap();
        assert!(!body.as_object().unwrap().contains_key("reasoning_effort"));
    }
}
