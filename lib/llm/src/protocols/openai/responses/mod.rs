// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod stream_converter;

use std::collections::HashMap;

use dynamo_protocols::types::responses::{
    AssistantRole, FunctionToolCall, IncludeEnum, IncompleteDetails, InputTokenDetails,
    Instructions, NamespaceToolParamTool, OutputItem, OutputMessage, OutputMessageContent,
    OutputStatus, OutputTextContent, OutputTokenDetails, PromptCacheRetention, Reasoning,
    ReasoningItem, Response, ResponseTextParam, ResponseUsage, ServiceTier, Status, SummaryPart,
    SummaryTextContent, TextResponseFormatConfiguration, Tool, ToolChoiceOptions, ToolChoiceParam,
    Truncation,
};
use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;
use validator::Validate;

use super::baseten_ext::BasetenExt;
use super::chat_completions::{NvCreateChatCompletionRequest, NvCreateChatCompletionResponse};
use super::nvext::{NvExt, NvExtProvider};
use super::{OpenAISamplingOptionsProvider, OpenAIStopConditionsProvider};

/// Request body for `POST /v1/responses`. Uses a plain
/// `#[derive(Deserialize)]` — the relaxed input shapes are handled by
/// Dynamo-owning the input chain in `dynamo_protocols::types::responses`
/// (see that crate's `CLAUDE.md`), not by a custom pre-parse JSON patcher.
/// An earlier iteration of this type carried a hand-written `impl Deserialize`
/// that walked `serde_json::Value` to inject synthetic defaults for missing
/// `id` / `status` / `annotations`; that was replaced by typed ownership for
/// correctness and to avoid the double-deserialize cost.
#[derive(ToSchema, Serialize, Deserialize, Validate, Debug, Clone, Default)]
pub struct NvCreateResponse {
    /// Flattened CreateResponse fields (model, input, temperature, etc.).
    ///
    /// `CreateResponse` and its `input` chain (`InputParam`, `InputItem`,
    /// `Item`, `MessageItem`, `InputOutputMessage`, `InputOutputMessageContent`,
    /// `InputOutputTextContent`) are Dynamo-owned in `dynamo-protocols`. They
    /// mirror upstream async-openai but accept the relaxed shapes real clients
    /// emit (optional `id` / `status` / `content` on assistant messages,
    /// optional `annotations` on `output_text` parts). See
    /// `dynamo_protocols::types::responses` for the full rationale.
    #[serde(flatten)]
    #[schema(value_type = Object)]
    pub inner: dynamo_protocols::types::responses::CreateResponse,

    /// Baseten-specific extensions (cache_control, dynamic_temperature,
    /// thinking, chat_template_args, etc.) forwarded at root level.
    #[serde(flatten, default, skip_serializing_if = "BasetenExt::is_empty")]
    pub baseten_ext: BasetenExt,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<NvExt>,
}

#[derive(ToSchema, Deserialize, Validate, Debug, Clone)]
pub struct NvResponse {
    /// Flattened Response fields (includes upstream + extended spec fields).
    #[serde(flatten)]
    #[schema(value_type = Object)]
    pub inner: dynamo_protocols::types::responses::Response,

    /// NVIDIA extension field for response metadata (worker IDs, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<serde_json::Value>,

    /// OpenResponses spec requires these as non-null scalars on every response,
    /// but async-openai's `Response` doesn't model them. Populated from the
    /// originating request. Surfaced during serialization (see `Serialize`
    /// impl below); not persisted as top-level fields on the inner struct.
    #[serde(default)]
    pub presence_penalty: f32,
    #[serde(default)]
    pub frequency_penalty: f32,
    #[serde(default)]
    pub store: bool,
}

/// Patch an already-serialized `Response` JSON object to match the
/// OpenResponses spec. Applied both to one-shot `NvResponse` serialization
/// and to every `Response` embedded inside a streaming event payload.
///
/// Reconciles two spec gaps between upstream async-openai's `Response` and
/// the OpenResponses spec:
///
///  1. Fields the spec requires as `T | null` that upstream marks
///     `Option<T>` with `skip_serializing_if = Option::is_none`. These are
///     silently dropped when None; the spec wants them present as null.
///  2. Fields the spec requires (`presence_penalty`, `frequency_penalty`,
///     `store`) that are absent from upstream `Response` entirely.
///
/// Rather than fork the upstream output chain (which would cascade into
/// `OutputItem`, streaming events, and a long tail of sub-types, per
/// `lib/protocols/CLAUDE.md`), we patch the serialized JSON. Adds a
/// single `serde_json::to_value` round-trip per response, which is
/// negligible next to tokenization/inference cost.
pub(crate) fn patch_response_for_spec(
    obj: &mut serde_json::Map<String, serde_json::Value>,
    presence_penalty: f32,
    frequency_penalty: f32,
    store: bool,
) {
    for key in dynamo_protocols::types::responses::SPEC_NULLABLE_REQUIRED_RESPONSE_FIELDS {
        obj.entry(*key).or_insert(serde_json::Value::Null);
    }

    obj.insert(
        "presence_penalty".into(),
        serde_json::json!(presence_penalty),
    );
    obj.insert(
        "frequency_penalty".into(),
        serde_json::json!(frequency_penalty),
    );
    obj.insert("store".into(), serde_json::json!(store));

    // openai-python >= 2.53 types `usage.input_tokens_details.cache_write_tokens`
    // as required. The engine does not report cache writes, so emit 0 rather
    // than omitting the field and failing typed clients.
    if let Some(serde_json::Value::Object(usage)) = obj.get_mut("usage")
        && let Some(serde_json::Value::Object(details)) = usage.get_mut("input_tokens_details")
    {
        details
            .entry("cache_write_tokens")
            .or_insert(serde_json::json!(0));
    }
}

impl Serialize for NvResponse {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut value = serde_json::to_value(&self.inner).map_err(serde::ser::Error::custom)?;
        let serde_json::Value::Object(obj) = &mut value else {
            return value.serialize(serializer);
        };

        patch_response_for_spec(
            obj,
            self.presence_penalty,
            self.frequency_penalty,
            self.store,
        );

        if let Some(nvext) = &self.nvext {
            obj.insert("nvext".into(), nvext.clone());
        }

        value.serialize(serializer)
    }
}

/// Implements `NvExtProvider` for `NvCreateResponse`,
/// providing access to NVIDIA-specific extensions.
impl NvExtProvider for NvCreateResponse {
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    fn raw_prompt(&self) -> Option<String> {
        None
    }
}

/// Implements `AnnotationsProvider` for `NvCreateResponse`,
/// enabling retrieval and management of request annotations.
impl AnnotationsProvider for NvCreateResponse {
    fn annotations(&self) -> Option<Vec<String>> {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.clone())
    }

    fn has_annotation(&self, annotation: &str) -> bool {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.as_ref())
            .map(|annotations| annotations.contains(&annotation.to_string()))
            .unwrap_or(false)
    }
}

impl OpenAISamplingOptionsProvider for NvCreateResponse {
    fn get_temperature(&self) -> Option<f32> {
        self.inner.temperature
    }

    fn get_top_p(&self) -> Option<f32> {
        self.inner.top_p
    }

    fn get_frequency_penalty(&self) -> Option<f32> {
        None
    }

    fn get_presence_penalty(&self) -> Option<f32> {
        None
    }

    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    fn get_seed(&self) -> Option<i64> {
        None
    }

    fn get_n(&self) -> Option<u8> {
        None
    }

    fn get_best_of(&self) -> Option<u8> {
        None
    }
}

impl OpenAIStopConditionsProvider for NvCreateResponse {
    #[allow(deprecated)]
    fn get_max_tokens(&self) -> Option<u32> {
        self.inner.max_output_tokens
    }

    fn get_min_tokens(&self) -> Option<u32> {
        None
    }

    fn get_stop(&self) -> Option<Vec<String>> {
        None
    }

    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
}

// ---------------------------------------------------------------------------
// Responses API -> Chat Completions conversion
// ---------------------------------------------------------------------------

/// The flat name a namespaced tool is declared under across the chat bridge.
///
/// Namespaces exist so tool names can overlap between groups; a bare-name
/// flattening would collide. The worker/model sees `{namespace}__{name}` and
/// `resolve_tool_identity` maps emitted calls back to the wire pair.
fn namespaced_chat_tool_name(namespace: &str, name: &str) -> String {
    format!("{namespace}__{name}")
}

/// Map a model-emitted (chat-bridge) tool name back to the wire identity —
/// `(name, namespace)` — using the request's declared tools. Codex dispatches
/// tool calls on the exact (name, namespace) pair with no fallback, so
/// emitted function_call items must echo the declaring group's namespace and
/// the member's original name. Exact-match lookup against the declarations;
/// no string parsing, so member names containing `__` stay unambiguous.
pub(super) fn resolve_tool_identity(
    tools: Option<&[Tool]>,
    chat_name: &str,
) -> (String, Option<String>) {
    for tool in tools.unwrap_or_default() {
        if let Tool::Namespace(ns) = tool {
            for member in &ns.tools {
                let member_name = match member {
                    NamespaceToolParamTool::Function(f) => &f.name,
                    NamespaceToolParamTool::Custom(c) => &c.name,
                };
                if chat_name == namespaced_chat_tool_name(&ns.name, member_name) {
                    return (member_name.clone(), Some(ns.name.clone()));
                }
            }
        }
    }
    (chat_name.to_string(), None)
}

/// Canonicalize a `/v1/responses` request body — the JSON exactly as the
/// client sent it — into the engine-bound Chat Completions request.
///
/// Goes through the shared api-translation crate (tool-bank's ingress);
/// server-tool-shaped tools are dropped with a warning inside the crate
/// (standard-dynamo behavior). The handler threads the parsed body here
/// rather than re-serializing its typed `NvCreateResponse`, so what the
/// canonicalizer sees is what the client wrote.
pub fn responses_body_to_chat_request(
    body: serde_json::Value,
) -> anyhow::Result<NvCreateChatCompletionRequest> {
    canonicalize_responses_body(body).map(|canonical| canonical.request)
}

/// b10: [`responses_body_to_chat_request`] plus the typed loss record, timed
/// and logged as the `canonicalize` stage.
pub fn canonicalize_responses_body(
    body: serde_json::Value,
) -> anyhow::Result<crate::protocols::unified::Canonicalized> {
    let started = std::time::Instant::now();
    let result = canonicalize_responses_body_inner(body);
    crate::protocols::unified::b10_log_canonicalize_stage("responses", started, &result);
    result
}

fn canonicalize_responses_body_inner(
    body: serde_json::Value,
) -> anyhow::Result<crate::protocols::unified::Canonicalized> {
    // Respect the caller's stream preference; default to true so the
    // non-streaming path aggregates internally.
    let stream = body
        .get("stream")
        .and_then(serde_json::Value::as_bool)
        .or(Some(true));
    let adapted = b10_dynamo_api_translation::request::adapt_request_json(
        body,
        b10_dynamo_api_translation::ClientProtocol::Responses,
        &http::HeaderMap::new(),
        &mut b10_dynamo_api_translation::hooks::DropServerTools,
    )
    .map_err(anyhow::Error::new)?;
    let losses = adapted.request.losses;
    let lowered = adapted.request.request;
    // Wire-edge re-parse: extension keys distribute into their typed homes
    // exactly as if the deployment had received the bytes. Residual keys —
    // fields neither the wire types nor the extension surface model — land
    // in the wrapper's `unsupported_fields` catch-all (warned during
    // validation, never serialized), so they cannot ride the engine-bound
    // body: strict engine-side parsers (the python harness's extra=forbid
    // pydantic models) 400 on them.
    let mut nv: NvCreateChatCompletionRequest =
        serde_json::from_value(serde_json::to_value(&lowered)?)?;
    nv.inner.stream = stream;
    Ok(crate::protocols::unified::Canonicalized {
        request: nv,
        losses,
    })
}

/// Typed-struct entry for callers that no longer hold the client's bytes:
/// re-serializes the wrapper (extension fields reassemble at the root where
/// the crate expects them) and canonicalizes that. The HTTP handler uses
/// [`responses_body_to_chat_request`] on the original body instead.
impl TryFrom<NvCreateResponse> for NvCreateChatCompletionRequest {
    type Error = anyhow::Error;

    fn try_from(resp: NvCreateResponse) -> Result<Self, Self::Error> {
        responses_body_to_chat_request(serde_json::to_value(&resp)?)
    }
}

/// Parse `<tool_call>` blocks from model text output.
/// Returns a list of (name, arguments_json) tuples.
/// Returns an empty vec immediately if no `<tool_call>` tag is present.
fn parse_tool_call_text(text: &str) -> Vec<(String, String)> {
    if !text.contains("<tool_call>") {
        return Vec::new();
    }
    let mut results = Vec::new();
    let mut search_start = 0;
    while let Some(start) = text[search_start..].find("<tool_call>") {
        let abs_start = search_start + start + "<tool_call>".len();
        if let Some(end) = text[abs_start..].find("</tool_call>") {
            let block = text[abs_start..abs_start + end].trim();
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(block) {
                let name = parsed
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let arguments = if let Some(args) = parsed.get("arguments") {
                    if args.is_string() {
                        args.as_str().unwrap_or("{}").to_string()
                    } else {
                        serde_json::to_string(args).unwrap_or_else(|_| "{}".to_string())
                    }
                } else {
                    "{}".to_string()
                };
                if !name.is_empty() {
                    results.push((name, arguments));
                }
            }
            search_start = abs_start + end + "</tool_call>".len();
        } else {
            break;
        }
    }
    results
}

/// Strip `<tool_call>...</tool_call>` blocks and any `<think>...</think>` blocks from text.
/// Returns the original string (no allocation) if no tags are present.
fn strip_tool_call_text(text: &str) -> std::borrow::Cow<'_, str> {
    let has_tool = text.contains("<tool_call>");
    let has_think = text.contains("<think>");
    if !has_tool && !has_think {
        return std::borrow::Cow::Borrowed(text);
    }

    fn strip_tag(input: &mut String, open: &str, close: &str) {
        while let Some(start) = input.find(open) {
            if let Some(end_offset) = input[start..].find(close) {
                input.replace_range(start..start + end_offset + close.len(), "");
            } else {
                input.truncate(start);
                break;
            }
        }
    }

    let mut result = text.to_string();
    if has_tool {
        strip_tag(&mut result, "<tool_call>", "</tool_call>");
    }
    if has_think {
        strip_tag(&mut result, "<think>", "</think>");
    }
    std::borrow::Cow::Owned(result)
}

// ---------------------------------------------------------------------------
// Chat Completions -> Responses API response conversion
// ---------------------------------------------------------------------------

/// Request parameters to echo back in Response objects.
/// Extracted from the incoming CreateResponse request so that
/// response objects reflect actual request values.
#[derive(Clone, Debug, Default)]
pub struct ResponseParams {
    pub model: Option<String>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_output_tokens: Option<u32>,
    pub parallel_tool_calls: Option<bool>,
    pub store: Option<bool>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoiceParam>,
    pub instructions: Option<String>,
    pub reasoning: Option<Reasoning>,
    pub text: Option<ResponseTextParam>,
    pub service_tier: Option<ServiceTier>,
    pub include: Option<Vec<IncludeEnum>>,
    pub truncation: Option<Truncation>,
    /// OpenResponses spec requires these fields on the response body. Upstream
    /// `CreateResponse` doesn't model them on the request yet, so for now they
    /// pass through as `None`; the response serializer defaults to 0.0 (the
    /// effective sglang default). Wired through `ResponseParams` anyway so
    /// that when upstream relaxes or we shadow `CreateResponse`, threading a
    /// real value becomes a one-line change at the request-extraction site.
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    /// Pass-through metadata fields. Codex and other clients send these as
    /// hints for OpenAI's caching/moderation backends; Dynamo doesn't act on
    /// them, but the spec includes them on the response body so we echo back
    /// what the caller sent rather than silently dropping. Echoing makes
    /// receipt observable to the client without needing a real backend.
    pub prompt_cache_key: Option<String>,
    pub prompt_cache_retention: Option<PromptCacheRetention>,
    pub safety_identifier: Option<String>,
    /// Echoed verbatim: the spec puts the request's `metadata` on the response
    /// body, so a caller can confirm what it tagged the request with.
    pub metadata: Option<HashMap<String, String>>,
    /// Echoed as sent; the spec default (0) applies only when the request
    /// omitted it.
    pub top_logprobs: Option<u8>,
}

impl ResponseParams {
    fn reasoning_summary_requested(&self) -> bool {
        self.reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.summary)
            .is_some()
    }
}

/// Normalize tools so that `FunctionTool.strict` is always set.
/// The upstream type uses `skip_serializing_if = "Option::is_none"` on `strict`,
/// so `None` causes the field to be omitted during JSON serialization.
/// Schema validators (Zod, etc.) expect `strict` to always be present.
/// OpenAI defaults `strict` to `true`.
pub(super) fn normalize_tools(tools: Vec<Tool>) -> Vec<Tool> {
    tools
        .into_iter()
        .map(|tool| match tool {
            Tool::Function(mut ft) => {
                if ft.strict.is_none() {
                    ft.strict = Some(true);
                }
                Tool::Function(ft)
            }
            other => other,
        })
        .collect()
}

/// Build an assistant text message output item.
fn make_text_message(id: String, text: String) -> OutputItem {
    OutputItem::Message(OutputMessage {
        id,
        role: AssistantRole::Assistant,
        status: OutputStatus::Completed,
        phase: None,
        content: vec![OutputMessageContent::OutputText(OutputTextContent {
            text,
            annotations: vec![],
            logprobs: Some(vec![]),
        })],
    })
}

/// Build a function call output item with generated IDs.
fn make_function_call(name: String, arguments: String, namespace: Option<String>) -> OutputItem {
    OutputItem::FunctionCall(FunctionToolCall {
        arguments,
        call_id: format!("call_{}", Uuid::new_v4().simple()),
        namespace,
        name,
        id: Some(format!("fc_{}", Uuid::new_v4().simple())),
        status: Some(OutputStatus::Completed),
    })
}

/// Convert a ChatCompletion response into a Responses API response object,
/// echoing back the actual request parameters from `params`.
pub fn chat_completion_to_response(
    nv_resp: NvCreateChatCompletionResponse,
    params: &ResponseParams,
    api_context: Option<&crate::protocols::unified::ResponsesContext>,
) -> Result<NvResponse, anyhow::Error> {
    let nvext = nv_resp.nvext.clone();
    let chat_resp = nv_resp.inner;
    let message_id = format!("msg_{}", Uuid::new_v4().simple());
    let response_id = format!("resp_{}", Uuid::new_v4().simple());

    let choice = chat_resp.choices.into_iter().next();
    let mut output = Vec::new();
    let mut output_limit_reached = false;

    if let Some(choice) = choice {
        output_limit_reached =
            choice.finish_reason == Some(dynamo_protocols::types::FinishReason::Length);

        // Handle structured tool calls
        if let Some(tool_calls) = choice.message.tool_calls {
            for tc in &tool_calls {
                let (name, namespace) =
                    resolve_tool_identity(params.tools.as_deref(), &tc.function.name);
                output.push(OutputItem::FunctionCall(FunctionToolCall {
                    arguments: tc.function.arguments.clone(),
                    call_id: tc.id.clone(),
                    namespace,
                    name,
                    id: Some(format!("fc_{}", Uuid::new_v4().simple())),
                    status: Some(OutputStatus::Completed),
                }));
            }
        }

        // Map reasoning_content to a Reasoning output item
        if let Some(reasoning_text) = choice.message.reasoning_content
            && !reasoning_text.is_empty()
            && params.reasoning_summary_requested()
        {
            output.push(OutputItem::Reasoning(ReasoningItem {
                id: Some(format!("rs_{}", Uuid::new_v4().simple())),
                summary: vec![SummaryPart::SummaryText(SummaryTextContent {
                    text: reasoning_text,
                })],
                content: None,
                encrypted_content: None,
                status: Some(OutputStatus::Completed),
            }));
        }

        // Handle text content -- also parse <tool_call> blocks from models
        // that emit tool calls as text (e.g. Qwen3)
        let content_text = match choice.message.content {
            Some(dynamo_protocols::types::ChatCompletionMessageContent::Text(text)) => Some(text),
            Some(dynamo_protocols::types::ChatCompletionMessageContent::Parts(_)) => {
                tracing::warn!(
                    "Multimodal content in responses API not yet supported, using placeholder"
                );
                Some("[multimodal content]".to_string())
            }
            None => None,
        };
        if let Some(content_text) = content_text
            && !content_text.is_empty()
        {
            let parsed_calls = parse_tool_call_text(&content_text);
            if !parsed_calls.is_empty() {
                for (name, arguments) in parsed_calls {
                    let (name, namespace) = resolve_tool_identity(params.tools.as_deref(), &name);
                    output.push(make_function_call(name, arguments, namespace));
                }
                let remaining = strip_tool_call_text(&content_text);
                if !remaining.trim().is_empty() {
                    output.push(make_text_message(
                        message_id.clone(),
                        remaining.into_owned(),
                    ));
                }
            } else {
                output.push(make_text_message(message_id.clone(), content_text));
            }
        }

        if output.is_empty() {
            output.push(make_text_message(message_id, String::new()));
        }
    } else {
        tracing::warn!("No choices in chat completion response, using empty content");
        output.push(make_text_message(message_id, String::new()));
    }

    // Apply `include` filtering: strip logprobs from output text unless
    // the caller explicitly requested them via `message.output_text.logprobs`.
    let keep_logprobs = params
        .include
        .as_ref()
        .is_some_and(|inc| inc.contains(&IncludeEnum::MessageOutputTextLogprobs));
    for item in &mut output {
        if let OutputItem::Message(msg) = item {
            for content in &mut msg.content {
                if let OutputMessageContent::OutputText(text) = content
                    && (!keep_logprobs || text.logprobs.is_none())
                {
                    text.logprobs = Some(Vec::new());
                }
            }
        }
    }

    let created_at = chat_resp.created as u64;
    let status = if output_limit_reached {
        Status::Incomplete
    } else {
        Status::Completed
    };
    if output_limit_reached {
        let reasoning_completed = output
            .iter()
            .any(|item| matches!(item, OutputItem::Message(_) | OutputItem::FunctionCall(_)));
        for item in &mut output {
            match item {
                OutputItem::Message(message) => message.status = OutputStatus::Incomplete,
                OutputItem::FunctionCall(call) => call.status = Some(OutputStatus::Incomplete),
                OutputItem::Reasoning(reasoning) if !reasoning_completed => {
                    reasoning.status = Some(OutputStatus::Incomplete)
                }
                _ => {}
            }
        }
    }
    let response = Response {
        id: response_id,
        object: "response".to_string(),
        created_at,
        completed_at: (!output_limit_reached).then_some(created_at),
        model: if chat_resp.model == "unknown" {
            params.model.clone().unwrap_or(chat_resp.model)
        } else {
            chat_resp.model
        },
        status,
        output,
        // Spec-required defaults (OpenResponses requires these as non-null)
        background: Some(false),
        metadata: Some(params.metadata.clone().unwrap_or_default()),
        parallel_tool_calls: params.parallel_tool_calls.or(Some(true)),
        temperature: params.temperature.or(Some(1.0)),
        text: Some(params.text.clone().unwrap_or(ResponseTextParam {
            format: TextResponseFormatConfiguration::Text,
            verbosity: None,
        })),
        tool_choice: params
            .tool_choice
            .clone()
            .or(Some(ToolChoiceParam::Mode(ToolChoiceOptions::Auto))),
        tools: Some(
            params
                .tools
                .clone()
                .map(normalize_tools)
                .unwrap_or_default(),
        ),
        top_p: params.top_p.or(Some(1.0)),
        truncation: Some(params.truncation.unwrap_or(Truncation::Disabled)),
        // Nullable but required to be present (null is valid)
        billing: None,
        conversation: None,
        error: None,
        incomplete_details: output_limit_reached.then(|| IncompleteDetails {
            reason: "max_output_tokens".to_string(),
        }),
        instructions: params.instructions.clone().map(Instructions::Text),
        max_output_tokens: params.max_output_tokens,
        previous_response_id: api_context.and_then(|ctx| ctx.previous_response_id.clone()),
        prompt: None,
        prompt_cache_key: params.prompt_cache_key.clone(),
        prompt_cache_retention: params.prompt_cache_retention,
        reasoning: params.reasoning.clone(),
        safety_identifier: params.safety_identifier.clone(),
        service_tier: Some(params.service_tier.unwrap_or(ServiceTier::Auto)),
        top_logprobs: Some(params.top_logprobs.unwrap_or(0)),
        usage: chat_resp.usage.map(|u| ResponseUsage {
            input_tokens: u.prompt_tokens,
            input_tokens_details: InputTokenDetails {
                cached_tokens: u
                    .prompt_tokens_details
                    .map(|d| d.cached_tokens.unwrap_or(0))
                    .unwrap_or(0),
            },
            output_tokens: u.completion_tokens,
            output_tokens_details: OutputTokenDetails {
                reasoning_tokens: u
                    .completion_tokens_details
                    .map(|d| d.reasoning_tokens.unwrap_or(0))
                    .unwrap_or(0),
            },
            total_tokens: u.total_tokens,
        }),
    };

    Ok(NvResponse {
        inner: response,
        nvext,
        presence_penalty: params.presence_penalty.unwrap_or(0.0),
        frequency_penalty: params.frequency_penalty.unwrap_or(0.0),
        store: params.store.unwrap_or(false),
    })
}

#[cfg(test)]
mod b10_tests;

#[cfg(test)]
mod tests {
    use dynamo_protocols::types::responses::{
        CreateResponse, EasyInputContent, EasyInputMessage, FunctionCallOutput,
        FunctionCallOutputItemParam, FunctionTool, FunctionToolCall, InputContent,
        InputImageContent, InputItem, InputMessage, InputOutputMessage, InputOutputMessageContent,
        InputOutputTextContent, InputParam, InputRole, InputTextContent, Item, MessageItem,
        Role as ResponseRole, Tool,
    };
    use dynamo_protocols::types::{
        ChatCompletionMessageToolCall, ChatCompletionRequestAssistantMessageContent,
        ChatCompletionRequestSystemMessageContent, ChatCompletionToolChoiceOption, FunctionType,
        ReasoningContent,
    };
    use dynamo_protocols::types::{
        ChatCompletionRequestMessage, ChatCompletionRequestUserMessageContent,
    };

    use super::*;
    use crate::types::openai::chat_completions::NvCreateChatCompletionResponse;
    use dynamo_protocols::types::responses::{FunctionToolParam, NamespaceToolParam};

    /// Regression: `NvCreateResponse` flattens the wire `CreateResponse`
    /// alongside the flattened `BasetenExt`. A flattened catch-all on the wire
    /// type would swallow the extension keys and `nvext` too, so
    /// re-serializing the wrapper emitted them twice.
    #[test]
    fn wrapper_reserializes_each_key_once() {
        let request: NvCreateResponse = serde_json::from_value(serde_json::json!({
            "model": "m",
            "input": "hi",
            "nvext": {"ignore_eos": true},
            "priority": {"level": 1},
            "chat_template_kwargs": {"enable_thinking": true},
        }))
        .unwrap();
        assert!(request.nvext.is_some());
        assert!(request.baseten_ext.priority.is_some());
        let text = serde_json::to_string(&request).unwrap();
        for key in ["\"model\"", "\"input\"", "\"nvext\"", "\"priority\""] {
            assert_eq!(text.matches(key).count(), 1, "{key} duplicated in {text}");
        }
    }

    /// tool_choice with no surviving tools (codex sends this on auxiliary
    /// turns like auto-compaction; hosted tool kinds are also filtered) must
    /// be dropped — forwarding it trips the worker's "When using
    /// tool_choice, tools must be set" rejection mid-session.
    #[test]
    fn tool_choice_without_surviving_tools_is_dropped() {
        let mut resp = make_response_with_input("hello");
        resp.inner.tool_choice = Some(ToolChoiceParam::Mode(ToolChoiceOptions::Auto));
        resp.inner.tools = None;
        let chat_req: NvCreateChatCompletionRequest = resp.try_into().unwrap();
        assert!(chat_req.inner.tools.is_none());
        assert!(chat_req.inner.tool_choice.is_none());

        // With a surviving function tool, tool_choice is forwarded.
        let mut resp = make_response_with_input("hello");
        resp.inner.tool_choice = Some(ToolChoiceParam::Mode(ToolChoiceOptions::Auto));
        resp.inner.tools = Some(vec![Tool::Function(FunctionTool {
            name: "get_weather".into(),
            parameters: None,
            strict: None,
            description: None,
            defer_loading: None,
        })]);
        let chat_req: NvCreateChatCompletionRequest = resp.try_into().unwrap();
        assert!(chat_req.inner.tools.is_some());
        assert!(matches!(
            chat_req.inner.tool_choice,
            Some(ChatCompletionToolChoiceOption::Auto)
        ));
    }

    #[test]
    fn namespace_tool_group_flattens_and_maps() {
        let tools = vec![
            Tool::Function(FunctionTool {
                name: "shell".into(),
                parameters: None,
                strict: None,
                description: None,
                defer_loading: None,
            }),
            Tool::Namespace(NamespaceToolParam {
                name: "mcp__codex_apps__gmail".into(),
                description: "Gmail tools".into(),
                tools: vec![NamespaceToolParamTool::Function(FunctionToolParam {
                    name: "get_recent_emails".into(),
                    description: Some("List recent emails".into()),
                    ..Default::default()
                })],
            }),
        ];

        // The worker sees a flat function list; namespace members are mangled
        // so names can overlap between groups.
        let names = tool_names_via_conversion(tools.clone());
        assert_eq!(
            names,
            vec!["shell", "mcp__codex_apps__gmail__get_recent_emails"]
        );

        // Emitted calls map back to the wire (name, namespace) pair.
        assert_eq!(
            resolve_tool_identity(Some(&tools), "mcp__codex_apps__gmail__get_recent_emails"),
            (
                "get_recent_emails".to_string(),
                Some("mcp__codex_apps__gmail".to_string())
            )
        );
        assert_eq!(
            resolve_tool_identity(Some(&tools), "shell"),
            ("shell".to_string(), None)
        );
        assert_eq!(
            resolve_tool_identity(None, "anything"),
            ("anything".to_string(), None)
        );
    }

    /// The point of namespaces: the same tool name in two groups must stay
    /// two distinct, individually-callable tools.
    #[test]
    fn overlapping_tool_names_across_namespaces_stay_distinct() {
        let member = |name: &str| {
            NamespaceToolParamTool::Function(FunctionToolParam {
                name: name.into(),
                ..Default::default()
            })
        };
        let tools = vec![
            Tool::Namespace(NamespaceToolParam {
                name: "gmail".into(),
                description: "Gmail".into(),
                tools: vec![member("search")],
            }),
            Tool::Namespace(NamespaceToolParam {
                name: "drive".into(),
                description: "Drive".into(),
                tools: vec![member("search")],
            }),
        ];

        let names = tool_names_via_conversion(tools.clone());
        assert_eq!(names, vec!["gmail__search", "drive__search"]);

        assert_eq!(
            resolve_tool_identity(Some(&tools), "gmail__search"),
            ("search".to_string(), Some("gmail".to_string()))
        );
        assert_eq!(
            resolve_tool_identity(Some(&tools), "drive__search"),
            ("search".to_string(), Some("drive".to_string()))
        );
    }

    /// Flattened tool names as the worker sees them, via the full
    /// conversion path (the shared crate owns the flattening now).
    fn tool_names_via_conversion(tools: Vec<Tool>) -> Vec<String> {
        let resp = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Text("hi".into()),
                model: Some("test-model".into()),
                tools: Some(tools),
                ..Default::default()
            },
            baseten_ext: Default::default(),
            nvext: None,
        };
        let chat: NvCreateChatCompletionRequest = resp.try_into().unwrap();
        chat.inner
            .tools
            .unwrap_or_default()
            .into_iter()
            .map(|t| t.function.name)
            .collect()
    }

    fn make_response_with_input(text: &str) -> NvCreateResponse {
        NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Text(text.into()),
                model: Some("test-model".into()),
                max_output_tokens: Some(1024),
                temperature: Some(0.5),
                top_p: Some(0.9),
                top_logprobs: Some(15),
                ..Default::default()
            },
            nvext: Some(NvExt {
                annotations: Some(vec!["debug".into(), "trace".into()]),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn test_annotations_trait_behavior() {
        let req = make_response_with_input("hello");
        assert_eq!(
            req.annotations(),
            Some(vec!["debug".to_string(), "trace".to_string()])
        );
        assert!(req.has_annotation("debug"));
        assert!(req.has_annotation("trace"));
        assert!(!req.has_annotation("missing"));
    }

    #[test]
    fn test_openai_sampling_trait_behavior() {
        let req = make_response_with_input("hello");
        assert_eq!(req.get_temperature(), Some(0.5));
        assert_eq!(req.get_top_p(), Some(0.9));
        assert_eq!(req.get_frequency_penalty(), None);
        assert_eq!(req.get_presence_penalty(), None);
    }

    #[test]
    fn test_openai_stop_conditions_trait_behavior() {
        let req = make_response_with_input("hello");
        assert_eq!(req.get_max_tokens(), Some(1024));
        assert_eq!(req.get_min_tokens(), None);
        assert_eq!(req.get_stop(), None);
    }

    #[test]
    fn test_into_nvcreate_chat_completion_request() {
        let nv_req: NvCreateChatCompletionRequest =
            make_response_with_input("hi there").try_into().unwrap();

        assert_eq!(nv_req.inner.model, "test-model");
        assert_eq!(nv_req.inner.temperature, Some(0.5));
        assert_eq!(nv_req.inner.top_p, Some(0.9));
        assert_eq!(nv_req.inner.max_completion_tokens, Some(1024));
        assert_eq!(nv_req.inner.top_logprobs, Some(15));
        assert_eq!(nv_req.inner.stream, Some(true));

        let messages = &nv_req.inner.messages;
        assert_eq!(messages.len(), 1);
        match &messages[0] {
            ChatCompletionRequestMessage::User(user_msg) => match &user_msg.content {
                ChatCompletionRequestUserMessageContent::Text(t) => {
                    assert_eq!(t, "hi there");
                }
                _ => panic!("unexpected user content type"),
            },
            _ => panic!("expected user message"),
        }
    }

    #[test]
    fn test_into_chat_completion_preserves_omitted_max_output_tokens() {
        let mut response_req = make_response_with_input("hi there");
        response_req.inner.max_output_tokens = None;

        let nv_req: NvCreateChatCompletionRequest = response_req.try_into().unwrap();

        assert_eq!(nv_req.inner.max_completion_tokens, None);
    }

    #[test]
    fn test_store_mapped_to_chat_completion_request() {
        let mut req = make_response_with_input("audit me");
        req.inner.store = Some(true);

        let nv_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert_eq!(nv_req.inner.store, Some(true));
    }

    #[test]
    fn test_instructions_prepended_as_system_message() {
        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Text("hello".into()),
                model: Some("test-model".into()),
                instructions: Some("You are a helpful assistant.".into()),
                ..Default::default()
            },
            nvext: None,
            ..Default::default()
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let messages = &chat_req.inner.messages;
        assert_eq!(messages.len(), 2);

        match &messages[0] {
            ChatCompletionRequestMessage::System(sys) => match &sys.content {
                Some(ChatCompletionRequestSystemMessageContent::Text(t)) => {
                    assert_eq!(t, "You are a helpful assistant.");
                }
                _ => panic!("expected text content"),
            },
            _ => panic!("expected system message first"),
        }
    }

    #[test]
    fn test_input_items_multi_turn() {
        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![
                    InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
                        content: vec![InputContent::InputText(InputTextContent {
                            text: "Be concise.".into(),
                        })],
                        role: InputRole::System,
                        status: None,
                    }))),
                    InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
                        content: vec![InputContent::InputText(InputTextContent {
                            text: "What is 2+2?".into(),
                        })],
                        role: InputRole::User,
                        status: None,
                    }))),
                    InputItem::Item(Item::Message(MessageItem::Output(InputOutputMessage {
                        id: Some("msg_1".into()),
                        role: AssistantRole::Assistant,
                        status: Some(OutputStatus::Completed),
                        phase: None,
                        content: vec![InputOutputMessageContent::OutputText(
                            InputOutputTextContent {
                                text: "4".into(),
                                annotations: vec![],
                                logprobs: None,
                            },
                        )],
                    }))),
                    InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
                        content: vec![InputContent::InputText(InputTextContent {
                            text: "And 3+3?".into(),
                        })],
                        role: InputRole::User,
                        status: None,
                    }))),
                ]),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
            ..Default::default()
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let messages = &chat_req.inner.messages;
        assert_eq!(messages.len(), 4);
        assert!(matches!(
            messages[0],
            ChatCompletionRequestMessage::System(_)
        ));
        assert!(matches!(messages[1], ChatCompletionRequestMessage::User(_)));
        assert!(matches!(
            messages[2],
            ChatCompletionRequestMessage::Assistant(_)
        ));
        assert!(matches!(messages[3], ChatCompletionRequestMessage::User(_)));
    }

    #[test]
    fn test_input_items_with_image() {
        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![InputItem::Item(Item::Message(MessageItem::Input(
                    InputMessage {
                        content: vec![
                            InputContent::InputText(InputTextContent {
                                text: "What is in this image?".into(),
                            }),
                            InputContent::InputImage(InputImageContent {
                                detail: Default::default(), // ImageDetail::Auto
                                file_id: None,
                                image_url: Some("https://example.com/cat.jpg".into()),
                            }),
                        ],
                        role: InputRole::User,
                        status: None,
                    },
                )))]),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
            ..Default::default()
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let messages = &chat_req.inner.messages;
        assert_eq!(messages.len(), 1);
        match &messages[0] {
            ChatCompletionRequestMessage::User(u) => match &u.content {
                ChatCompletionRequestUserMessageContent::Array(parts) => {
                    assert_eq!(parts.len(), 2);
                }
                _ => panic!("expected array content"),
            },
            _ => panic!("expected user message"),
        }
    }

    /// EasyMessage path (no `type: "message"`) with user role + multimodal
    /// content must preserve image parts all the way to the chat request.
    /// Regression for issue #9468 review feedback: previously the EasyMessage
    /// handler text-flattened the ContentList before dispatching on role, so
    /// `input_image` parts were silently dropped on a no-type user payload.
    #[test]
    fn test_easy_message_user_multimodal_preserves_images() {
        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![InputItem::EasyMessage(EasyInputMessage {
                    role: ResponseRole::User,
                    content: EasyInputContent::ContentList(vec![
                        InputContent::InputText(InputTextContent {
                            text: "What is in this image?".into(),
                        }),
                        InputContent::InputImage(InputImageContent {
                            detail: Default::default(),
                            file_id: None,
                            image_url: Some("https://example.com/cat.jpg".into()),
                        }),
                    ]),
                    ..Default::default()
                })]),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
            ..Default::default()
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let messages = &chat_req.inner.messages;
        assert_eq!(messages.len(), 1);
        match &messages[0] {
            ChatCompletionRequestMessage::User(u) => match &u.content {
                ChatCompletionRequestUserMessageContent::Array(parts) => {
                    assert_eq!(parts.len(), 2, "expected text + image parts to survive");
                    let has_text = parts.iter().any(|p| {
                        matches!(
                            p,
                            dynamo_protocols::types::ChatCompletionRequestUserMessageContentPart::Text(_)
                        )
                    });
                    let has_image = parts.iter().any(|p| {
                        matches!(
                            p,
                            dynamo_protocols::types::ChatCompletionRequestUserMessageContentPart::ImageUrl(_)
                        )
                    });
                    assert!(has_text, "text part missing");
                    assert!(has_image, "image part dropped — regression of #9468 review");
                }
                ChatCompletionRequestUserMessageContent::Text(t) => panic!(
                    "expected Array content with image preserved, got Text({t:?}) — images were dropped",
                ),
            },
            _ => panic!("expected user message"),
        }
    }

    /// EasyMessage path text-only user payload still produces a plain-text
    /// content (single-text-part short-circuit in
    /// `convert_input_content_to_user_content`).
    #[test]
    fn test_easy_message_user_text_only_stays_text() {
        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![InputItem::EasyMessage(EasyInputMessage {
                    role: ResponseRole::User,
                    content: EasyInputContent::Text("hello".into()),
                    ..Default::default()
                })]),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
            ..Default::default()
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        match &chat_req.inner.messages[0] {
            ChatCompletionRequestMessage::User(u) => match &u.content {
                ChatCompletionRequestUserMessageContent::Text(t) => assert_eq!(t, "hello"),
                other => panic!("expected Text user content, got {other:?}"),
            },
            other => panic!("expected user message, got {other:?}"),
        }
    }

    /// EasyMessage with role=system carrying a `ContentList` text part still
    /// produces a `System` chat message — chat-completions has no multimodal
    /// system slot, so collapsing to text is the right thing.
    #[test]
    fn test_easy_message_system_contentlist_collapses_to_text() {
        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![
                    InputItem::EasyMessage(EasyInputMessage {
                        role: ResponseRole::System,
                        content: EasyInputContent::ContentList(vec![InputContent::InputText(
                            InputTextContent {
                                text: "You are helpful.".into(),
                            },
                        )]),
                        ..Default::default()
                    }),
                    InputItem::EasyMessage(EasyInputMessage {
                        role: ResponseRole::User,
                        content: EasyInputContent::Text("hi".into()),
                        ..Default::default()
                    }),
                ]),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
            ..Default::default()
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert!(matches!(
            chat_req.inner.messages[0],
            ChatCompletionRequestMessage::System(_)
        ));
    }

    /// EasyMessage prior-assistant turn with a `ContentList` (text part) still
    /// coalesces into the pending assistant accumulator and emits an
    /// assistant chat message, preserving the turn boundary.
    #[test]
    fn test_easy_message_assistant_contentlist_collapses_to_text() {
        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![
                    InputItem::EasyMessage(EasyInputMessage {
                        role: ResponseRole::Assistant,
                        content: EasyInputContent::ContentList(vec![InputContent::InputText(
                            InputTextContent { text: "ok".into() },
                        )]),
                        ..Default::default()
                    }),
                    InputItem::EasyMessage(EasyInputMessage {
                        role: ResponseRole::User,
                        content: EasyInputContent::Text("next".into()),
                        ..Default::default()
                    }),
                ]),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
            ..Default::default()
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let msgs = &chat_req.inner.messages;
        assert!(matches!(
            msgs[0],
            ChatCompletionRequestMessage::Assistant(_)
        ));
        assert!(matches!(msgs[1], ChatCompletionRequestMessage::User(_)));
    }

    #[test]
    fn test_function_call_input_items() {
        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![
                    InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
                        content: vec![InputContent::InputText(InputTextContent {
                            text: "What's the weather?".into(),
                        })],
                        role: InputRole::User,
                        status: None,
                    }))),
                    InputItem::Item(Item::FunctionCall(FunctionToolCall {
                        arguments: r#"{"location":"SF"}"#.into(),
                        call_id: "call_123".into(),
                        namespace: None,
                        name: "get_weather".into(),
                        id: None,
                        status: None,
                    })),
                    InputItem::Item(Item::FunctionCallOutput(FunctionCallOutputItemParam {
                        call_id: "call_123".into(),
                        output: FunctionCallOutput::Text(r#"{"temp":"72F"}"#.into()),
                        id: None,
                        status: None,
                    })),
                ]),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
            ..Default::default()
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let messages = &chat_req.inner.messages;
        assert_eq!(messages.len(), 3);
        assert!(matches!(messages[0], ChatCompletionRequestMessage::User(_)));
        assert!(matches!(
            messages[1],
            ChatCompletionRequestMessage::Assistant(_)
        ));
        assert!(matches!(messages[2], ChatCompletionRequestMessage::Tool(_)));
    }

    #[test]
    fn test_function_call_with_interstitial_assistant_message_is_coalesced() {
        // Regression: prior turn was `function_call` + assistant text + `function_call_output`.
        // The converter must emit a SINGLE assistant chat message carrying both `content`
        // and `tool_calls`, otherwise chat templates that require a tool message to
        // immediately follow its assistant tool_call (e.g. MiniMax) will reject the input.
        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![
                    InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
                        content: vec![InputContent::InputText(InputTextContent {
                            text: "What's the weather?".into(),
                        })],
                        role: InputRole::User,
                        status: None,
                    }))),
                    InputItem::Item(Item::FunctionCall(FunctionToolCall {
                        arguments: r#"{"location":"SF"}"#.into(),
                        call_id: "call_123".into(),
                        namespace: None,
                        name: "get_weather".into(),
                        id: None,
                        status: None,
                    })),
                    InputItem::Item(Item::Message(MessageItem::Output(InputOutputMessage {
                        id: Some("msg_interstitial".into()),
                        role: AssistantRole::Assistant,
                        status: Some(OutputStatus::Completed),
                        phase: None,
                        content: vec![InputOutputMessageContent::OutputText(
                            InputOutputTextContent {
                                text: "\n\n".into(),
                                annotations: vec![],
                                logprobs: None,
                            },
                        )],
                    }))),
                    InputItem::Item(Item::FunctionCallOutput(FunctionCallOutputItemParam {
                        call_id: "call_123".into(),
                        output: FunctionCallOutput::Text(r#"{"temp":"72F"}"#.into()),
                        id: None,
                        status: None,
                    })),
                ]),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
            ..Default::default()
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let messages = &chat_req.inner.messages;
        assert_eq!(
            messages.len(),
            3,
            "expected coalesced [user, assistant, tool]"
        );
        assert!(matches!(messages[0], ChatCompletionRequestMessage::User(_)));
        match &messages[1] {
            ChatCompletionRequestMessage::Assistant(a) => {
                let tool_calls = a.tool_calls.as_ref().expect("tool_calls must be present");
                assert_eq!(tool_calls.len(), 1);
                assert_eq!(tool_calls[0].id, "call_123");
                assert_eq!(tool_calls[0].function.name, "get_weather");
                match a
                    .content
                    .as_ref()
                    .expect("content must carry interstitial text")
                {
                    ChatCompletionRequestAssistantMessageContent::Text(t) => {
                        assert_eq!(t, "\n\n");
                    }
                    _ => panic!("expected text content"),
                }
            }
            _ => panic!("expected a single merged assistant message at index 1"),
        }
        assert!(matches!(messages[2], ChatCompletionRequestMessage::Tool(_)));
    }

    #[test]
    fn test_easy_message_assistant_coalesced_with_adjacent_function_call() {
        // The same coalescing rule applies to EasyInputMessage shape (string content,
        // role=assistant, no `type:"message"` discriminator).
        use dynamo_protocols::types::responses::{
            EasyInputContent, EasyInputMessage, Role as ResponseRole,
        };

        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![
                    InputItem::EasyMessage(EasyInputMessage {
                        role: ResponseRole::User,
                        content: EasyInputContent::Text("x".into()),
                        ..Default::default()
                    }),
                    InputItem::Item(Item::FunctionCall(FunctionToolCall {
                        arguments: "{}".into(),
                        call_id: "c".into(),
                        namespace: None,
                        name: "f".into(),
                        id: None,
                        status: None,
                    })),
                    InputItem::EasyMessage(EasyInputMessage {
                        role: ResponseRole::Assistant,
                        content: EasyInputContent::Text("".into()),
                        ..Default::default()
                    }),
                    InputItem::Item(Item::FunctionCallOutput(FunctionCallOutputItemParam {
                        call_id: "c".into(),
                        output: FunctionCallOutput::Text("x".into()),
                        id: None,
                        status: None,
                    })),
                ]),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
            ..Default::default()
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let messages = &chat_req.inner.messages;
        assert_eq!(messages.len(), 3);
        match &messages[1] {
            ChatCompletionRequestMessage::Assistant(a) => {
                assert!(a.tool_calls.is_some());
                assert_eq!(a.tool_calls.as_ref().unwrap().len(), 1);
            }
            _ => panic!("expected merged assistant message"),
        }
        assert!(matches!(messages[2], ChatCompletionRequestMessage::Tool(_)));
    }

    #[test]
    fn test_standalone_assistant_message_with_empty_content_preserves_turn() {
        // A prior assistant turn that produced no text (empty content or
        // refusal-only parts the converter strips) must still emit an assistant
        // message. Otherwise adjacent user turns get silently merged, which
        // breaks strict-alternation chat templates and distorts the context
        // the model sees.
        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![
                    InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
                        content: vec![InputContent::InputText(InputTextContent {
                            text: "first question".into(),
                        })],
                        role: InputRole::User,
                        status: None,
                    }))),
                    InputItem::Item(Item::Message(MessageItem::Output(InputOutputMessage {
                        id: None,
                        role: AssistantRole::Assistant,
                        status: None,
                        phase: None,
                        content: vec![InputOutputMessageContent::OutputText(
                            InputOutputTextContent {
                                text: "".into(),
                                annotations: vec![],
                                logprobs: None,
                            },
                        )],
                    }))),
                    InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
                        content: vec![InputContent::InputText(InputTextContent {
                            text: "second question".into(),
                        })],
                        role: InputRole::User,
                        status: None,
                    }))),
                ]),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
            ..Default::default()
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let messages = &chat_req.inner.messages;
        assert_eq!(
            messages.len(),
            3,
            "empty assistant turn must not be silently dropped"
        );
        assert!(matches!(messages[0], ChatCompletionRequestMessage::User(_)));
        match &messages[1] {
            ChatCompletionRequestMessage::Assistant(a) => {
                assert!(a.tool_calls.is_none());
                match a.content.as_ref().expect("empty turn still emits content") {
                    ChatCompletionRequestAssistantMessageContent::Text(t) => {
                        assert_eq!(t, "");
                    }
                    _ => panic!("expected text content"),
                }
            }
            _ => panic!("expected assistant turn boundary preserved"),
        }
        assert!(matches!(messages[2], ChatCompletionRequestMessage::User(_)));
    }

    #[test]
    fn test_easy_assistant_message_with_empty_content_preserves_turn() {
        // Same turn-boundary preservation applies to EasyInputMessage shape.
        use dynamo_protocols::types::responses::{
            EasyInputContent, EasyInputMessage, Role as ResponseRole,
        };

        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![
                    InputItem::EasyMessage(EasyInputMessage {
                        role: ResponseRole::User,
                        content: EasyInputContent::Text("first".into()),
                        ..Default::default()
                    }),
                    InputItem::EasyMessage(EasyInputMessage {
                        role: ResponseRole::Assistant,
                        content: EasyInputContent::Text("".into()),
                        ..Default::default()
                    }),
                    InputItem::EasyMessage(EasyInputMessage {
                        role: ResponseRole::User,
                        content: EasyInputContent::Text("second".into()),
                        ..Default::default()
                    }),
                ]),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
            ..Default::default()
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let messages = &chat_req.inner.messages;
        assert_eq!(messages.len(), 3);
        assert!(matches!(
            messages[1],
            ChatCompletionRequestMessage::Assistant(_)
        ));
    }

    #[test]
    fn test_pure_function_call_turn_emits_null_content() {
        // Chat Completions spec allows `content: null` on assistant messages
        // that carry only `tool_calls`. Some Jinja templates gate on
        // `{% if message.content is not none %}`; we must not emit
        // `content: ""` for pure-tool-call turns. Turn-boundary cases (empty
        // OutputMessage with no tool_calls) still emit `Some(Text(""))`.
        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![
                    InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
                        content: vec![InputContent::InputText(InputTextContent {
                            text: "hi".into(),
                        })],
                        role: InputRole::User,
                        status: None,
                    }))),
                    InputItem::Item(Item::FunctionCall(FunctionToolCall {
                        arguments: "{}".into(),
                        call_id: "c".into(),
                        namespace: None,
                        name: "f".into(),
                        id: None,
                        status: None,
                    })),
                    InputItem::Item(Item::FunctionCallOutput(FunctionCallOutputItemParam {
                        call_id: "c".into(),
                        output: FunctionCallOutput::Text("ok".into()),
                        id: None,
                        status: None,
                    })),
                ]),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
            ..Default::default()
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let messages = &chat_req.inner.messages;
        assert_eq!(messages.len(), 3);
        match &messages[1] {
            ChatCompletionRequestMessage::Assistant(a) => {
                assert!(
                    a.content.is_none(),
                    "pure tool-call turn must have content: null, got {:?}",
                    a.content
                );
                assert!(a.tool_calls.is_some());
            }
            _ => panic!("expected assistant message"),
        }
    }

    #[test]
    fn test_reasoning_item_routed_into_reasoning_content() {
        // Regression: Codex / Agents SDK round-trip Item::Reasoning mid-turn.
        // The converter must route the reasoning summary into the coalesced
        // assistant message's `reasoning_content`, not silently drop it.
        use dynamo_protocols::types::responses::{ReasoningItem, SummaryPart, SummaryTextContent};

        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![
                    InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
                        content: vec![InputContent::InputText(InputTextContent {
                            text: "solve".into(),
                        })],
                        role: InputRole::User,
                        status: None,
                    }))),
                    InputItem::Item(Item::Reasoning(ReasoningItem {
                        id: Some("rs_1".into()),
                        summary: vec![SummaryPart::SummaryText(SummaryTextContent {
                            text: "thinking step 1".into(),
                        })],
                        content: None,
                        encrypted_content: None,
                        status: None,
                    })),
                    InputItem::Item(Item::FunctionCall(FunctionToolCall {
                        arguments: "{}".into(),
                        call_id: "c".into(),
                        namespace: None,
                        name: "f".into(),
                        id: None,
                        status: None,
                    })),
                    InputItem::Item(Item::FunctionCallOutput(FunctionCallOutputItemParam {
                        call_id: "c".into(),
                        output: FunctionCallOutput::Text("ok".into()),
                        id: None,
                        status: None,
                    })),
                ]),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
            ..Default::default()
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let messages = &chat_req.inner.messages;
        assert_eq!(messages.len(), 3);
        match &messages[1] {
            ChatCompletionRequestMessage::Assistant(a) => {
                match a
                    .reasoning_content
                    .as_ref()
                    .expect("reasoning must be preserved")
                {
                    // The replayed function_call closes the reasoning segment, so the shape is
                    // Segments: segment 0 precedes call 0, one empty trailing segment follows it.
                    ReasoningContent::Segments(segments) => {
                        assert_eq!(
                            segments,
                            &vec!["thinking step 1".to_string(), String::new()]
                        );
                    }
                    ReasoningContent::Text(t) => assert_eq!(t, "thinking step 1"),
                }
                assert!(a.tool_calls.is_some());
            }
            _ => panic!("expected assistant message with reasoning + tool_calls"),
        }
    }

    #[test]
    fn test_unsupported_item_variant_flushes_pending() {
        // Sequence: function_call → (an unsupported tool-output variant) →
        // function_call → function_call_output. Without a flush on the
        // catch-all, the two FunctionCalls would coalesce into a single
        // assistant `tool_calls` list despite being different semantic turns.
        use dynamo_protocols::types::responses::{
            ComputerCallOutputItemParam, ComputerScreenshotImage, ComputerScreenshotImageType,
        };

        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![
                    InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
                        content: vec![InputContent::InputText(InputTextContent {
                            text: "go".into(),
                        })],
                        role: InputRole::User,
                        status: None,
                    }))),
                    InputItem::Item(Item::FunctionCall(FunctionToolCall {
                        arguments: "{}".into(),
                        call_id: "c1".into(),
                        namespace: None,
                        name: "f".into(),
                        id: None,
                        status: None,
                    })),
                    InputItem::Item(Item::ComputerCallOutput(ComputerCallOutputItemParam {
                        call_id: "cc1".into(),
                        output: ComputerScreenshotImage {
                            r#type: ComputerScreenshotImageType::ComputerScreenshot,
                            image_url: None,
                            file_id: None,
                        },
                        acknowledged_safety_checks: None,
                        id: None,
                        status: None,
                    })),
                    InputItem::Item(Item::FunctionCall(FunctionToolCall {
                        arguments: "{}".into(),
                        call_id: "c2".into(),
                        namespace: None,
                        name: "f".into(),
                        id: None,
                        status: None,
                    })),
                    InputItem::Item(Item::FunctionCallOutput(FunctionCallOutputItemParam {
                        call_id: "c2".into(),
                        output: FunctionCallOutput::Text("ok".into()),
                        id: None,
                        status: None,
                    })),
                ]),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
            ..Default::default()
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let messages = &chat_req.inner.messages;
        // Expected: User, Assistant(tc=[c1]), Assistant(tc=[c2]), Tool(c2)
        // Without the catch-all flush, we'd get Assistant(tc=[c1,c2]) instead.
        assert!(messages.len() >= 4, "catch-all must flush pending");
        let tc_msgs: Vec<_> = messages
            .iter()
            .filter_map(|m| match m {
                ChatCompletionRequestMessage::Assistant(a) => a.tool_calls.as_ref(),
                _ => None,
            })
            .collect();
        assert_eq!(
            tc_msgs.len(),
            2,
            "two tool-call turns must not coalesce across unsupported variant"
        );
        assert_eq!(tc_msgs[0].len(), 1);
        assert_eq!(tc_msgs[0][0].id, "c1");
        assert_eq!(tc_msgs[1].len(), 1);
        assert_eq!(tc_msgs[1][0].id, "c2");
    }

    #[test]
    fn test_function_call_then_output_text_then_output_merges_to_one_turn() {
        // Canonical MiniMax repro (the Codex/Agents-SDK sequence that first
        // broke): user → function_call → assistant text → function_call_output.
        // Must yield 3 chat messages: user, assistant(content + tool_calls),
        // tool. Any other shape breaks the chat template's tool-call pairing.
        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![
                    InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
                        content: vec![InputContent::InputText(InputTextContent {
                            text: "call say".into(),
                        })],
                        role: InputRole::User,
                        status: None,
                    }))),
                    InputItem::Item(Item::FunctionCall(FunctionToolCall {
                        arguments: r#"{"x":"hi"}"#.into(),
                        call_id: "c".into(),
                        namespace: None,
                        name: "say".into(),
                        id: None,
                        status: None,
                    })),
                    InputItem::Item(Item::Message(MessageItem::Output(InputOutputMessage {
                        id: None,
                        role: AssistantRole::Assistant,
                        status: None,
                        phase: None,
                        content: vec![InputOutputMessageContent::OutputText(
                            InputOutputTextContent {
                                text: "\n\n\n".into(),
                                annotations: vec![],
                                logprobs: None,
                            },
                        )],
                    }))),
                    InputItem::Item(Item::FunctionCallOutput(FunctionCallOutputItemParam {
                        call_id: "c".into(),
                        output: FunctionCallOutput::Text("hi".into()),
                        id: None,
                        status: None,
                    })),
                ]),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
            ..Default::default()
        };
        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let messages = &chat_req.inner.messages;
        assert_eq!(messages.len(), 3);
        assert!(matches!(messages[0], ChatCompletionRequestMessage::User(_)));
        match &messages[1] {
            ChatCompletionRequestMessage::Assistant(a) => {
                let tool_calls = a.tool_calls.as_ref().expect("tool_calls present");
                assert_eq!(tool_calls.len(), 1);
                assert_eq!(tool_calls[0].id, "c");
                match a.content.as_ref().expect("text content present") {
                    ChatCompletionRequestAssistantMessageContent::Text(t) => {
                        assert_eq!(t, "\n\n\n");
                    }
                    _ => panic!("expected text content"),
                }
            }
            _ => panic!("expected merged assistant message"),
        }
        assert!(matches!(messages[2], ChatCompletionRequestMessage::Tool(_)));
    }

    #[test]
    fn test_output_text_then_function_call_then_output_merges_to_one_turn() {
        // Reverse ordering: assistant text before the function_call. The
        // coalescer's accumulator is order-agnostic — both orderings must
        // produce the same merged assistant message.
        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![
                    InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
                        content: vec![InputContent::InputText(InputTextContent {
                            text: "call say".into(),
                        })],
                        role: InputRole::User,
                        status: None,
                    }))),
                    InputItem::Item(Item::Message(MessageItem::Output(InputOutputMessage {
                        id: None,
                        role: AssistantRole::Assistant,
                        status: None,
                        phase: None,
                        content: vec![InputOutputMessageContent::OutputText(
                            InputOutputTextContent {
                                text: "let me call it".into(),
                                annotations: vec![],
                                logprobs: None,
                            },
                        )],
                    }))),
                    InputItem::Item(Item::FunctionCall(FunctionToolCall {
                        arguments: r#"{"x":"hi"}"#.into(),
                        call_id: "c".into(),
                        namespace: None,
                        name: "say".into(),
                        id: None,
                        status: None,
                    })),
                    InputItem::Item(Item::FunctionCallOutput(FunctionCallOutputItemParam {
                        call_id: "c".into(),
                        output: FunctionCallOutput::Text("hi".into()),
                        id: None,
                        status: None,
                    })),
                ]),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
            ..Default::default()
        };
        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let messages = &chat_req.inner.messages;
        assert_eq!(messages.len(), 3);
        match &messages[1] {
            ChatCompletionRequestMessage::Assistant(a) => {
                assert_eq!(a.tool_calls.as_ref().expect("tool_calls present").len(), 1);
                match a.content.as_ref().expect("content present") {
                    ChatCompletionRequestAssistantMessageContent::Text(t) => {
                        assert_eq!(t, "let me call it");
                    }
                    _ => panic!("expected text content"),
                }
            }
            _ => panic!("expected merged assistant message"),
        }
    }

    #[test]
    fn test_multiple_function_calls_merge_into_single_assistant_message() {
        // Parallel tool calls (`parallel_tool_calls: true`) produce multiple
        // adjacent Item::FunctionCall items. They must coalesce into a single
        // assistant message carrying all tool_calls.
        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![
                    InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
                        content: vec![InputContent::InputText(InputTextContent {
                            text: "do two things".into(),
                        })],
                        role: InputRole::User,
                        status: None,
                    }))),
                    InputItem::Item(Item::FunctionCall(FunctionToolCall {
                        arguments: "{}".into(),
                        call_id: "c1".into(),
                        namespace: None,
                        name: "f".into(),
                        id: None,
                        status: None,
                    })),
                    InputItem::Item(Item::FunctionCall(FunctionToolCall {
                        arguments: "{}".into(),
                        call_id: "c2".into(),
                        namespace: None,
                        name: "g".into(),
                        id: None,
                        status: None,
                    })),
                    InputItem::Item(Item::FunctionCallOutput(FunctionCallOutputItemParam {
                        call_id: "c1".into(),
                        output: FunctionCallOutput::Text("r1".into()),
                        id: None,
                        status: None,
                    })),
                    InputItem::Item(Item::FunctionCallOutput(FunctionCallOutputItemParam {
                        call_id: "c2".into(),
                        output: FunctionCallOutput::Text("r2".into()),
                        id: None,
                        status: None,
                    })),
                ]),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
            ..Default::default()
        };
        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let messages = &chat_req.inner.messages;
        // user, assistant(tc=[c1, c2]), tool(c1), tool(c2)
        assert_eq!(messages.len(), 4);
        match &messages[1] {
            ChatCompletionRequestMessage::Assistant(a) => {
                let tool_calls = a.tool_calls.as_ref().expect("tool_calls present");
                assert_eq!(tool_calls.len(), 2, "parallel tool_calls must coalesce");
                assert_eq!(tool_calls[0].id, "c1");
                assert_eq!(tool_calls[1].id, "c2");
                assert!(a.content.is_none(), "pure-tool-call turn has null content");
            }
            _ => panic!("expected single merged assistant message"),
        }
        assert!(matches!(messages[2], ChatCompletionRequestMessage::Tool(_)));
        assert!(matches!(messages[3], ChatCompletionRequestMessage::Tool(_)));
    }

    #[test]
    fn test_refusal_content_folded_into_assistant_text() {
        // Refusal parts in a prior assistant turn must survive to the next
        // turn. We fold refusal text into the assistant's `content` so
        // templates render it identically to normal content; otherwise the
        // model loses visibility into what it previously refused.
        use dynamo_protocols::types::responses::RefusalContent;
        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![
                    InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
                        content: vec![InputContent::InputText(InputTextContent {
                            text: "try again".into(),
                        })],
                        role: InputRole::User,
                        status: None,
                    }))),
                    InputItem::Item(Item::Message(MessageItem::Output(InputOutputMessage {
                        id: None,
                        role: AssistantRole::Assistant,
                        status: None,
                        phase: None,
                        content: vec![InputOutputMessageContent::Refusal(RefusalContent {
                            refusal: "I cannot help with that.".into(),
                        })],
                    }))),
                    InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
                        content: vec![InputContent::InputText(InputTextContent {
                            text: "ok different question".into(),
                        })],
                        role: InputRole::User,
                        status: None,
                    }))),
                ]),
                model: Some("test-model".into()),
                ..Default::default()
            },
            nvext: None,
            ..Default::default()
        };
        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let messages = &chat_req.inner.messages;
        assert_eq!(messages.len(), 3);
        match &messages[1] {
            ChatCompletionRequestMessage::Assistant(a) => {
                match a.content.as_ref().expect("refusal folded into content") {
                    ChatCompletionRequestAssistantMessageContent::Text(t) => {
                        assert_eq!(t, "I cannot help with that.");
                    }
                    _ => panic!("expected text content"),
                }
            }
            _ => panic!("expected assistant message carrying folded refusal"),
        }
    }

    #[test]
    fn test_tools_conversion() {
        let req = NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Text("hello".into()),
                model: Some("test-model".into()),
                tools: Some(vec![Tool::Function(FunctionTool {
                    name: "get_weather".into(),
                    parameters: Some(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    })),
                    strict: Some(true),
                    description: Some("Get weather info".into()),
                    defer_loading: None,
                })]),
                ..Default::default()
            },
            nvext: None,
            ..Default::default()
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert!(chat_req.inner.tools.is_some());
        let tools = chat_req.inner.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "get_weather");
    }

    #[allow(deprecated)]
    #[test]
    fn test_into_nvresponse_from_chat_response() {
        let now = 1_726_000_000;
        let chat_resp = NvCreateChatCompletionResponse {
            inner: dynamo_protocols::types::CreateChatCompletionResponse {
                id: "chatcmpl-xyz".into(),
                choices: vec![dynamo_protocols::types::ChatChoice {
                    index: 0,
                    message: dynamo_protocols::types::ChatCompletionResponseMessage {
                        content: Some(dynamo_protocols::types::ChatCompletionMessageContent::Text(
                            "This is a reply".to_string(),
                        )),
                        refusal: None,
                        tool_calls: None,
                        role: dynamo_protocols::types::Role::Assistant,
                        function_call: None,
                        audio: None,
                        reasoning_content: None,
                    },
                    finish_reason: None,
                    logprobs: None,
                }],
                created: now,
                model: "llama-3.1-8b-instruct".into(),
                service_tier: None,
                system_fingerprint: None,
                object: "chat.completion".to_string(),
                usage: None,
            },
            nvext: None,
        };

        let wrapped =
            chat_completion_to_response(chat_resp, &ResponseParams::default(), None).unwrap();

        assert_eq!(wrapped.inner.model, "llama-3.1-8b-instruct");
        assert_eq!(wrapped.inner.status, Status::Completed);
        assert_eq!(wrapped.inner.object, "response");
        assert!(wrapped.inner.id.starts_with("resp_"));

        let msg = match &wrapped.inner.output[0] {
            OutputItem::Message(m) => m,
            _ => panic!("Expected Message variant"),
        };
        assert_eq!(msg.role, AssistantRole::Assistant);

        match &msg.content[0] {
            OutputMessageContent::OutputText(txt) => {
                assert_eq!(txt.text, "This is a reply");
            }
            _ => panic!("Expected OutputText content"),
        }
    }

    #[allow(deprecated)]
    #[test]
    fn test_response_with_tool_calls() {
        let now = 1_726_000_000;
        let chat_resp = NvCreateChatCompletionResponse {
            inner: dynamo_protocols::types::CreateChatCompletionResponse {
                id: "chatcmpl-xyz".into(),
                choices: vec![dynamo_protocols::types::ChatChoice {
                    index: 0,
                    message: dynamo_protocols::types::ChatCompletionResponseMessage {
                        content: None,
                        refusal: None,
                        tool_calls: Some(vec![ChatCompletionMessageToolCall {
                            id: "call_abc".into(),
                            r#type: FunctionType::Function,
                            function: dynamo_protocols::types::FunctionCall {
                                name: "get_weather".into(),
                                arguments: r#"{"location":"SF"}"#.into(),
                            },
                        }]),
                        role: dynamo_protocols::types::Role::Assistant,
                        function_call: None,
                        audio: None,
                        reasoning_content: None,
                    },
                    finish_reason: None,
                    logprobs: None,
                }],
                created: now,
                model: "test-model".into(),
                service_tier: None,
                system_fingerprint: None,
                object: "chat.completion".to_string(),
                usage: None,
            },
            nvext: None,
        };

        let wrapped =
            chat_completion_to_response(chat_resp, &ResponseParams::default(), None).unwrap();
        assert_eq!(wrapped.inner.output.len(), 1);
        match &wrapped.inner.output[0] {
            OutputItem::FunctionCall(fc) => {
                assert_eq!(fc.call_id, "call_abc");
                assert_eq!(fc.name, "get_weather");
            }
            _ => panic!("Expected FunctionCall output"),
        }
    }

    #[test]
    fn test_top_logprobs_validated_not_silently_clamped() {
        // Deliberate change with the shared-crate switchover: an
        // out-of-range top_logprobs is refused (OpenAI's hosted API 400s
        // at 21), not silently clamped to 20 as the old converter did.
        let with_top = |n: u8| {
            let mut resp = make_response_with_input("hi");
            resp.inner.top_logprobs = Some(n);
            NvCreateChatCompletionRequest::try_from(resp)
        };
        assert_eq!(with_top(5).unwrap().inner.top_logprobs, Some(5));
        assert!(with_top(21).is_err());
        assert!(with_top(255).is_err());
        let mut without = make_response_with_input("hi");
        without.inner.top_logprobs = None;
        assert_eq!(
            NvCreateChatCompletionRequest::try_from(without)
                .unwrap()
                .inner
                .top_logprobs,
            None
        );
    }

    #[test]
    fn test_parse_tool_call_text() {
        // Standard Qwen3 format
        let text = r#"<think>
Let me check the weather.
</think>

<tool_call>
{"name": "get_weather", "arguments": {"location": "San Francisco"}}
</tool_call>"#;
        let calls = parse_tool_call_text(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].1).unwrap();
        assert_eq!(args["location"], "San Francisco");
    }

    #[test]
    fn test_parse_tool_call_text_multiple() {
        let text = r#"<tool_call>
{"name": "func_a", "arguments": {"x": 1}}
</tool_call>
<tool_call>
{"name": "func_b", "arguments": {"y": 2}}
</tool_call>"#;
        let calls = parse_tool_call_text(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].0, "func_a");
        assert_eq!(calls[1].0, "func_b");
    }

    #[test]
    fn test_parse_tool_call_text_no_calls() {
        let text = "Just a regular message with no tool calls.";
        let calls = parse_tool_call_text(text);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_strip_tool_call_text() {
        let text = r#"<think>
thinking
</think>

<tool_call>
{"name": "f", "arguments": {}}
</tool_call>"#;
        let stripped = strip_tool_call_text(text);
        assert!(!stripped.contains("<tool_call>"));
        assert!(!stripped.contains("<think>"));
    }

    // ── PR1: reasoning / text.format / service_tier pass-through tests ──

    #[test]
    fn test_reasoning_effort_mapped_to_chat_completion() {
        use dynamo_protocols::types::B10ReasoningEffort;
        use dynamo_protocols::types::responses::B10ReasoningParam;

        // `max` also covers the round trip: this conversion re-serializes the
        // Responses body and re-parses it as Chat Completions, so an effort the
        // upstream enum cannot spell has to survive both halves.
        for effort in [B10ReasoningEffort::Medium, B10ReasoningEffort::Max] {
            let mut req = make_response_with_input("think hard");
            req.inner.reasoning = Some(B10ReasoningParam {
                effort: Some(effort.clone()),
                ..Default::default()
            });

            let chat: NvCreateChatCompletionRequest = req.try_into().unwrap();
            assert_eq!(chat.inner.reasoning_effort, Some(effort));
        }
    }

    #[test]
    fn test_reasoning_none_leaves_chat_field_none() {
        let req = make_response_with_input("no reasoning");
        let chat: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert_eq!(chat.inner.reasoning_effort, None);
    }

    #[test]
    fn test_text_format_json_object_mapped() {
        use dynamo_protocols::types::ResponseFormat;
        use dynamo_protocols::types::responses::{
            ResponseTextParam, TextResponseFormatConfiguration,
        };

        let mut req = make_response_with_input("give json");
        req.inner.text = Some(ResponseTextParam {
            format: TextResponseFormatConfiguration::JsonObject,
            verbosity: None,
        });

        let chat: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert_eq!(chat.inner.response_format, Some(ResponseFormat::JsonObject));
    }

    #[test]
    fn test_text_format_json_schema_mapped() {
        use dynamo_protocols::types::responses::{
            ResponseTextParam, TextResponseFormatConfiguration,
        };
        use dynamo_protocols::types::{ResponseFormat, ResponseFormatJsonSchema};

        let schema = ResponseFormatJsonSchema {
            name: "city".into(),
            description: None,
            schema: serde_json::json!({"type": "object"}),
            strict: Some(true),
        };
        let mut req = make_response_with_input("structured");
        req.inner.text = Some(ResponseTextParam {
            format: TextResponseFormatConfiguration::JsonSchema(schema.clone()),
            verbosity: None,
        });

        let chat: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert_eq!(
            chat.inner.response_format,
            Some(ResponseFormat::JsonSchema {
                json_schema: schema
            })
        );
    }

    #[test]
    fn test_text_format_plain_text_leaves_response_format_none() {
        use dynamo_protocols::types::responses::{
            ResponseTextParam, TextResponseFormatConfiguration,
        };

        let mut req = make_response_with_input("plain");
        req.inner.text = Some(ResponseTextParam {
            format: TextResponseFormatConfiguration::Text,
            verbosity: None,
        });

        let chat: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert_eq!(chat.inner.response_format, None);
    }

    #[test]
    fn test_service_tier_mapped_to_chat_completion() {
        use dynamo_protocols::types::ServiceTier as ChatServiceTier;
        use dynamo_protocols::types::responses::ServiceTier as RespServiceTier;

        let mut req = make_response_with_input("priority");
        req.inner.service_tier = Some(RespServiceTier::Priority);

        let chat: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert_eq!(chat.inner.service_tier, Some(ChatServiceTier::Priority));
    }

    #[test]
    fn test_parallel_tool_calls_mapped_to_chat_completion() {
        let mut req = make_response_with_input("parallel tools off");
        req.inner.parallel_tool_calls = Some(false);

        let chat: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert_eq!(chat.inner.parallel_tool_calls, Some(false));
    }

    #[test]
    fn test_response_echoes_reasoning() {
        use dynamo_protocols::types::ReasoningEffort;
        use dynamo_protocols::types::responses::Reasoning;

        let params = ResponseParams {
            reasoning: Some(Reasoning {
                effort: Some(ReasoningEffort::High),
                ..Default::default()
            }),
            ..Default::default()
        };

        let chat_resp = NvCreateChatCompletionResponse {
            inner: dynamo_protocols::types::CreateChatCompletionResponse {
                choices: vec![],
                created: 0,
                id: "test".into(),
                model: "m".into(),
                service_tier: None,
                system_fingerprint: None,
                object: "chat.completion".into(),
                usage: None,
            },
            nvext: None,
        };

        let resp = chat_completion_to_response(chat_resp, &params, None).unwrap();
        let reasoning = resp.inner.reasoning.unwrap();
        assert_eq!(reasoning.effort, Some(ReasoningEffort::High));
    }

    #[test]
    fn test_response_echoes_text_format() {
        use dynamo_protocols::types::responses::{
            ResponseTextParam, TextResponseFormatConfiguration,
        };

        let params = ResponseParams {
            text: Some(ResponseTextParam {
                format: TextResponseFormatConfiguration::JsonObject,
                verbosity: None,
            }),
            ..Default::default()
        };

        let chat_resp = NvCreateChatCompletionResponse {
            inner: dynamo_protocols::types::CreateChatCompletionResponse {
                choices: vec![],
                created: 0,
                id: "test".into(),
                model: "m".into(),
                service_tier: None,
                system_fingerprint: None,
                object: "chat.completion".into(),
                usage: None,
            },
            nvext: None,
        };

        let resp = chat_completion_to_response(chat_resp, &params, None).unwrap();
        let text = resp.inner.text.unwrap();
        assert_eq!(text.format, TextResponseFormatConfiguration::JsonObject);
    }

    #[test]
    fn test_response_echoes_service_tier() {
        use dynamo_protocols::types::responses::ServiceTier;

        let params = ResponseParams {
            service_tier: Some(ServiceTier::Flex),
            ..Default::default()
        };

        let chat_resp = NvCreateChatCompletionResponse {
            inner: dynamo_protocols::types::CreateChatCompletionResponse {
                choices: vec![],
                created: 0,
                id: "test".into(),
                model: "m".into(),
                service_tier: None,
                system_fingerprint: None,
                object: "chat.completion".into(),
                usage: None,
            },
            nvext: None,
        };

        let resp = chat_completion_to_response(chat_resp, &params, None).unwrap();
        assert_eq!(resp.inner.service_tier, Some(ServiceTier::Flex));
    }

    #[test]
    fn test_response_echoes_parallel_tool_calls() {
        let params = ResponseParams {
            parallel_tool_calls: Some(false),
            ..Default::default()
        };

        let chat_resp = NvCreateChatCompletionResponse {
            inner: dynamo_protocols::types::CreateChatCompletionResponse {
                choices: vec![],
                created: 0,
                id: "test".into(),
                model: "m".into(),
                service_tier: None,
                system_fingerprint: None,
                object: "chat.completion".into(),
                usage: None,
            },
            nvext: None,
        };

        let resp = chat_completion_to_response(chat_resp, &params, None).unwrap();
        assert_eq!(resp.inner.parallel_tool_calls, Some(false));
    }

    #[test]
    fn test_bare_assistant_output_message_deserializes_via_owned_types() {
        // Regression: upstream async-openai's OutputMessage required `id` and
        // `status`. Dynamo-owned types make them optional so real-world client
        // shapes (no id/status, no annotations) round-trip successfully.
        let json = serde_json::json!({
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello!"}],
            "type": "message"
        });

        let item: InputItem =
            serde_json::from_value(json).expect("relaxed deserialize should succeed");
        match item {
            InputItem::Item(Item::Message(MessageItem::Output(msg))) => {
                assert_eq!(msg.role, AssistantRole::Assistant);
                assert!(msg.id.is_none());
                assert!(msg.status.is_none());
            }
            other => panic!("Expected Item::Message(Output), got {:?}", other),
        }
    }

    #[test]
    fn test_nvcreate_response_accepts_bare_assistant_messages() {
        // End-to-end: a real Codex-style payload with an interstitial assistant
        // text item (no id/status/annotations) deserializes into NvCreateResponse
        // via the standard derive on our Dynamo-owned CreateResponse chain.
        let body = serde_json::json!({
            "model": "m",
            "input": [
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "hi"}
                ]},
                {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"},
                {"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "\n\n\n"}
                ]},
                {"type": "function_call_output", "call_id": "c", "output": "x"}
            ]
        });

        let req: NvCreateResponse =
            serde_json::from_value(body).expect("relaxed deserialize should succeed");
        let items = match &req.inner.input {
            InputParam::Items(items) => items,
            _ => panic!("expected Items input"),
        };
        assert_eq!(items.len(), 4);
        match &items[2] {
            InputItem::Item(Item::Message(MessageItem::Output(out))) => {
                assert_eq!(out.role, AssistantRole::Assistant);
            }
            other => panic!("expected MessageItem::Output, got {:?}", other),
        }
    }

    #[test]
    fn test_output_message_with_id_and_status_still_works() {
        use dynamo_protocols::types::responses::{InputItem, Item, MessageItem, OutputStatus};

        let json = serde_json::json!({
            "role": "assistant",
            "id": "msg_abc123",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Hello!", "annotations": []}],
            "type": "message"
        });

        let item: InputItem = serde_json::from_value(json).unwrap();
        match item {
            InputItem::Item(Item::Message(MessageItem::Output(msg))) => {
                assert_eq!(msg.id.as_deref(), Some("msg_abc123"));
                assert_eq!(msg.status, Some(OutputStatus::Completed));
            }
            other => panic!("Expected Item::Message(Output), got {:?}", other),
        }
    }

    // ── PR2: include filtering + truncation echo-back tests ──

    fn make_chat_resp_with_text(text: &str) -> NvCreateChatCompletionResponse {
        use dynamo_protocols::types::{
            ChatChoice, ChatCompletionMessageContent, ChatCompletionResponseMessage, FinishReason,
        };
        NvCreateChatCompletionResponse {
            inner: dynamo_protocols::types::CreateChatCompletionResponse {
                choices: vec![ChatChoice {
                    index: 0,
                    #[allow(deprecated)]
                    message: ChatCompletionResponseMessage {
                        content: Some(ChatCompletionMessageContent::Text(text.into())),
                        role: dynamo_protocols::types::Role::Assistant,
                        tool_calls: None,
                        refusal: None,
                        reasoning_content: None,
                        function_call: None,
                        audio: None,
                    },
                    finish_reason: Some(FinishReason::Stop),
                    logprobs: None,
                }],
                created: 0,
                id: "test".into(),
                model: "m".into(),
                service_tier: None,
                system_fingerprint: None,
                object: "chat.completion".into(),
                usage: None,
            },
            nvext: None,
        }
    }

    fn make_chat_resp_with_reasoning(reasoning: &str) -> NvCreateChatCompletionResponse {
        let mut response = make_chat_resp_with_text("answer");
        response.inner.choices[0].message.reasoning_content = Some(reasoning.into());
        response
    }

    fn make_chat_resp_with_tool_call(
        finish_reason: dynamo_protocols::types::FinishReason,
        arguments: &str,
    ) -> NvCreateChatCompletionResponse {
        let mut response = make_chat_resp_with_text("");
        let choice = &mut response.inner.choices[0];
        choice.finish_reason = Some(finish_reason);
        choice.message.content = None;
        choice.message.tool_calls = Some(vec![ChatCompletionMessageToolCall {
            id: "call_abc".into(),
            r#type: FunctionType::Function,
            function: dynamo_protocols::types::FunctionCall {
                name: "get_weather".into(),
                arguments: arguments.into(),
            },
        }]);
        response
    }

    #[test]
    fn b10_reasoning_summary_requires_explicit_request() {
        use dynamo_protocols::types::responses::{Reasoning, ReasoningSummary};

        let unrequested = chat_completion_to_response(
            make_chat_resp_with_reasoning("private reasoning"),
            &ResponseParams::default(),
            None,
        )
        .unwrap();
        assert!(
            unrequested
                .inner
                .output
                .iter()
                .all(|item| !matches!(item, OutputItem::Reasoning(_)))
        );

        let params = ResponseParams {
            reasoning: Some(Reasoning {
                effort: None,
                summary: Some(ReasoningSummary::Auto),
            }),
            ..Default::default()
        };
        let requested =
            chat_completion_to_response(make_chat_resp_with_reasoning("summary"), &params, None)
                .unwrap();
        let reasoning = requested
            .inner
            .output
            .iter()
            .find_map(|item| match item {
                OutputItem::Reasoning(reasoning) => Some(reasoning),
                _ => None,
            })
            .expect("requested reasoning summary output");
        assert_eq!(
            reasoning.summary,
            vec![SummaryPart::SummaryText(SummaryTextContent {
                text: "summary".into(),
            })]
        );
    }

    #[test]
    fn test_include_logprobs_empty_by_default() {
        // OpenResponses schema requires `logprobs` to be an array. When the
        // caller did not request them via `include`, emit an empty array
        // rather than null.
        let chat_resp = make_chat_resp_with_text("hello");
        let params = ResponseParams::default();
        let resp = chat_completion_to_response(chat_resp, &params, None).unwrap();

        for item in &resp.inner.output {
            if let OutputItem::Message(msg) = item {
                for content in &msg.content {
                    if let OutputMessageContent::OutputText(t) = content {
                        assert_eq!(
                            t.logprobs.as_deref(),
                            Some(&[][..]),
                            "logprobs should be an empty array by default"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_include_logprobs_kept_when_requested() {
        use dynamo_protocols::types::responses::IncludeEnum;

        let chat_resp = make_chat_resp_with_text("hello");
        let params = ResponseParams {
            include: Some(vec![IncludeEnum::MessageOutputTextLogprobs]),
            ..Default::default()
        };
        let resp = chat_completion_to_response(chat_resp, &params, None).unwrap();

        let mut found_text = false;
        for item in &resp.inner.output {
            if let OutputItem::Message(msg) = item {
                for content in &msg.content {
                    if let OutputMessageContent::OutputText(t) = content {
                        found_text = true;
                        assert!(
                            t.logprobs.is_some(),
                            "logprobs should be preserved when included"
                        );
                    }
                }
            }
        }
        assert!(found_text, "Expected text output");
    }

    #[test]
    fn test_truncation_auto_echoed_back() {
        use dynamo_protocols::types::responses::Truncation;

        let chat_resp = make_chat_resp_with_text("hello");
        let params = ResponseParams {
            truncation: Some(Truncation::Auto),
            ..Default::default()
        };
        let resp = chat_completion_to_response(chat_resp, &params, None).unwrap();
        assert_eq!(resp.inner.truncation, Some(Truncation::Auto));
    }

    #[test]
    fn test_truncation_defaults_to_disabled() {
        let chat_resp = make_chat_resp_with_text("hello");
        let params = ResponseParams::default();
        let resp = chat_completion_to_response(chat_resp, &params, None).unwrap();
        assert_eq!(resp.inner.truncation, Some(Truncation::Disabled));
    }

    #[test]
    fn b10_length_finish_reason_returns_incomplete_response() {
        let mut chat_resp = make_chat_resp_with_text("partial");
        chat_resp.inner.choices[0].finish_reason =
            Some(dynamo_protocols::types::FinishReason::Length);

        let resp =
            chat_completion_to_response(chat_resp, &ResponseParams::default(), None).unwrap();

        assert_eq!(resp.inner.status, Status::Incomplete);
        assert_eq!(resp.inner.completed_at, None);
        assert_eq!(
            resp.inner
                .incomplete_details
                .as_ref()
                .map(|details| details.reason.as_str()),
            Some("max_output_tokens")
        );
        let OutputItem::Message(message) = &resp.inner.output[0] else {
            panic!("expected message output");
        };
        assert_eq!(message.status, OutputStatus::Incomplete);
    }

    #[test]
    fn b10_tool_calls_finish_reason_returns_completed_response() {
        let chat_resp = make_chat_resp_with_tool_call(
            dynamo_protocols::types::FinishReason::ToolCalls,
            r#"{"location":"SF"}"#,
        );

        let response =
            chat_completion_to_response(chat_resp, &ResponseParams::default(), None).unwrap();

        assert_eq!(response.inner.status, Status::Completed);
        assert!(response.inner.incomplete_details.is_none());
        let OutputItem::FunctionCall(call) = &response.inner.output[0] else {
            panic!("expected function call output");
        };
        assert_eq!(call.status, Some(OutputStatus::Completed));
    }

    #[test]
    fn b10_unmodified_length_with_tool_call_returns_incomplete_response() {
        let chat_resp = make_chat_resp_with_tool_call(
            dynamo_protocols::types::FinishReason::Length,
            r#"{"location":"SF"#,
        );

        let response =
            chat_completion_to_response(chat_resp, &ResponseParams::default(), None).unwrap();

        assert_eq!(response.inner.status, Status::Incomplete);
        assert_eq!(
            response
                .inner
                .incomplete_details
                .as_ref()
                .map(|details| details.reason.as_str()),
            Some("max_output_tokens")
        );
        let OutputItem::FunctionCall(call) = &response.inner.output[0] else {
            panic!("expected function call output");
        };
        assert_eq!(call.status, Some(OutputStatus::Incomplete));
    }

    #[test]
    fn b10_length_finish_reason_preserves_completed_reasoning_status() {
        use dynamo_protocols::types::responses::{Reasoning, ReasoningSummary};

        let mut chat_resp = make_chat_resp_with_reasoning("complete reasoning");
        chat_resp.inner.choices[0].finish_reason =
            Some(dynamo_protocols::types::FinishReason::Length);
        let params = ResponseParams {
            reasoning: Some(Reasoning {
                effort: None,
                summary: Some(ReasoningSummary::Auto),
            }),
            ..Default::default()
        };

        let response = chat_completion_to_response(chat_resp, &params, None)
            .unwrap()
            .inner;
        assert_eq!(response.status, Status::Incomplete);
        let OutputItem::Reasoning(reasoning) = &response.output[0] else {
            panic!("expected reasoning output");
        };
        assert_eq!(reasoning.status, Some(OutputStatus::Completed));
        let OutputItem::Message(message) = &response.output[1] else {
            panic!("expected message output");
        };
        assert_eq!(message.status, OutputStatus::Incomplete);
    }

    #[test]
    fn b10_length_finish_reason_marks_terminal_reasoning_incomplete() {
        use dynamo_protocols::types::responses::{Reasoning, ReasoningSummary};

        let mut chat_resp = make_chat_resp_with_reasoning("partial reasoning");
        chat_resp.inner.choices[0].message.content = None;
        chat_resp.inner.choices[0].finish_reason =
            Some(dynamo_protocols::types::FinishReason::Length);
        let params = ResponseParams {
            reasoning: Some(Reasoning {
                effort: None,
                summary: Some(ReasoningSummary::Auto),
            }),
            ..Default::default()
        };

        let response = chat_completion_to_response(chat_resp, &params, None)
            .unwrap()
            .inner;
        assert_eq!(response.status, Status::Incomplete);
        let OutputItem::Reasoning(reasoning) = &response.output[0] else {
            panic!("expected reasoning output");
        };
        assert_eq!(reasoning.status, Some(OutputStatus::Incomplete));
    }

    /// Pass-through metadata fields the OpenResponses spec includes on the
    /// response body. Codex sends `prompt_cache_key` on every request; we
    /// echo it back so the caller can confirm receipt without enforcing any
    /// caching semantics. Same pattern for `prompt_cache_retention` and
    /// `safety_identifier`.
    #[test]
    fn test_response_echoes_passthrough_metadata() {
        let chat_resp = make_chat_resp_with_text("hello");
        let params = ResponseParams {
            prompt_cache_key: Some("cache-key-codex-1".into()),
            prompt_cache_retention: Some(PromptCacheRetention::InMemory),
            safety_identifier: Some("user-abc".into()),
            ..Default::default()
        };
        let resp = chat_completion_to_response(chat_resp, &params, None).unwrap();
        assert_eq!(
            resp.inner.prompt_cache_key.as_deref(),
            Some("cache-key-codex-1")
        );
        assert_eq!(
            resp.inner.prompt_cache_retention,
            Some(PromptCacheRetention::InMemory)
        );
        assert_eq!(resp.inner.safety_identifier.as_deref(), Some("user-abc"));
    }

    /// Every echoed parameter reports what the client sent, not a spec default
    /// or a hard-coded zero: `metadata`, `top_logprobs`, and the two sampling
    /// penalties (which live outside the typed `CreateResponse` and are
    /// projected from the raw body by the handler). The streaming twin lives
    /// in `stream_converter::tests`.
    #[test]
    fn test_response_echoes_metadata_top_logprobs_and_penalties() {
        let params = ResponseParams {
            metadata: Some(HashMap::from([("job".to_string(), "x".to_string())])),
            top_logprobs: Some(5),
            presence_penalty: Some(0.75),
            frequency_penalty: Some(0.25),
            service_tier: Some(ServiceTier::Auto),
            ..Default::default()
        };
        let resp =
            chat_completion_to_response(make_chat_resp_with_text("hello"), &params, None).unwrap();
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["metadata"], serde_json::json!({"job": "x"}));
        assert_eq!(json["top_logprobs"], 5);
        assert_eq!(json["presence_penalty"], 0.75);
        assert_eq!(json["frequency_penalty"], 0.25);
        assert_eq!(json["service_tier"], "auto");
    }

    /// Validate the JSON wire shape of NvResponse matches the OpenResponses
    /// spec: required scalars always present, nullable-required fields
    /// emitted as `null` when None.
    #[test]
    fn test_response_wire_format_shape() {
        let chat_resp = make_chat_resp_with_text("hello");
        let params = ResponseParams::default();
        let resp = chat_completion_to_response(chat_resp, &params, None).unwrap();
        let json = serde_json::to_value(&resp).unwrap();

        // Required scalars the spec mandates on every response. Upstream
        // async-openai's Response struct doesn't model these; NvResponse's
        // custom serializer injects them.
        assert_eq!(json["frequency_penalty"], 0.0);
        assert_eq!(json["presence_penalty"], 0.0);
        assert_eq!(json["store"], false);

        // Other required fields with expected values
        assert_eq!(json["object"], "response");
        assert_eq!(json["status"], "completed");
        assert_eq!(json["metadata"], serde_json::json!({}));
        assert!(json["output"].is_array());
        assert!(json["output"][0].get("id").is_some());
        assert!(json["output"][0].get("status").is_some());

        // Nullable-required fields must be present as null (not missing).
        for key in [
            "error",
            "incomplete_details",
            "billing",
            "conversation",
            "safety_identifier",
            "max_tool_calls",
            "instructions",
            "previous_response_id",
            "prompt_cache_key",
            "reasoning",
        ] {
            assert_eq!(
                json.get(key),
                Some(&serde_json::Value::Null),
                "expected {key} to be present as null"
            );
        }

        // nvext should be omitted when None
        assert!(json.get("nvext").is_none());
    }
}
