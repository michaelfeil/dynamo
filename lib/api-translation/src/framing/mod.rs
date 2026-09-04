//! Per-client-protocol rendering of TB's output. The loop and the egress speak one protocol-neutral
//! vocabulary ([`SemanticChunk`], [`CompletedIteration`], [`BufferedResponse`]); each protocol's
//! module turns that into its own bytes.
//!
//! [`ClientProtocol::envelope`] is the single place a protocol becomes an implementation, so adding
//! one is a new module plus one arm there. (Ingress adaptation dispatches separately, in
//! [`super::request`] — it is the other edge, with its own per-protocol signature.)
//!
//! Both traits are entirely synchronous: the async edge is the frame channel, owned by
//! [`super::sse_emitter::SseEmitter`], so `dyn` dispatch here costs no boxed future per chunk.

mod cc;
mod messages;
mod responses;

pub use responses::ResponsesParams;

use std::collections::HashSet;

use dynamo_protocols::error::{ApiError, WrappedError};
use dynamo_protocols::types::CompletionUsage;

use crate::baseten_response_extension::{
    IterationScope, ServerToolCallOutcome, ServerToolCallRecord,
};
use crate::coding_adapter::CodingAdapter;
use crate::model::{BackendError, ServerToolCall, Termination, ToolCall, ToolInvocation};
use crate::wire::{sse_frame, to_json_string};
use crate::{CcMessage, ClientProtocol, SemanticChunk};

/// OpenAI's error shape (`{"error": {...}}`), identical on CC and Responses — shared rather than
/// duplicated per envelope.
fn openai_error_json(error_code: Option<&str>, message: &str) -> String {
    to_json_string(&WrappedError {
        error: ApiError {
            message: message.to_string(),
            r#type: Some("api_error".to_string()),
            param: None,
            code: error_code.map(str::to_owned),
        },
    })
}

fn openai_error_sse_frame(error_code: Option<&str>, message: &str) -> String {
    sse_frame(None, &openai_error_json(error_code, message))
}

/// One iteration's `baseten` scope before a protocol shapes it, so [`StreamFraming`] needs one
/// staging method rather than one per [`IterationScope`] field.
pub struct StagedIteration<'a> {
    pub index: u32,
    pub usage: Option<&'a CompletionUsage>,
    pub debug_msg: Option<&'a str>,
}

impl StagedIteration<'_> {
    fn scope<Usage: serde::Serialize>(
        &self,
        convert_usage: impl Fn(&CompletionUsage) -> Usage,
    ) -> IterationScope<Usage> {
        IterationScope {
            usage: self.usage.map(convert_usage),
            debug_msg: self.debug_msg.map(str::to_owned).into_iter().collect(),
            ..IterationScope::at(self.index)
        }
    }
}

/// Holds the in-flight iteration's scope out of the flushable extension until its index advances
/// or the stream finishes. One iteration's parts arrive in different frames (a steering
/// `debug_msg` before the model call, `usage` after it), and flushing them as they come hands a
/// streamed client two `iterations[]` entries at one index where the buffered body merges them.
pub(crate) struct OpenIterationScope<Usage: serde::Serialize>(Option<IterationScope<Usage>>);

impl<Usage: serde::Serialize> Default for OpenIterationScope<Usage> {
    fn default() -> Self {
        Self(None)
    }
}

impl<Usage: serde::Serialize> OpenIterationScope<Usage> {
    /// Merges into the open scope on a matching index; a new index closes the open scope and
    /// returns it, complete and ready to flush.
    pub(crate) fn stage(&mut self, scope: IterationScope<Usage>) -> Option<IterationScope<Usage>> {
        match &mut self.0 {
            Some(open) if open.index == scope.index => {
                debug_assert!(
                    open.usage.is_none() || scope.usage.is_none(),
                    "two usage stagings for iteration {}",
                    scope.index
                );
                if open.usage.is_none() {
                    open.usage = scope.usage;
                }
                open.server_tool_calls.extend(scope.server_tool_calls);
                open.continuation_messages
                    .extend(scope.continuation_messages);
                open.debug_msg.extend(scope.debug_msg);
                None
            }
            _ => {
                if let Some(open) = &self.0 {
                    debug_assert!(
                        open.index < scope.index,
                        "iteration scopes staged out of order: {} after {}",
                        scope.index,
                        open.index
                    );
                }
                self.0.replace(scope)
            }
        }
    }

    pub(crate) fn close(&mut self) -> Option<IterationScope<Usage>> {
        self.0.take()
    }
}

/// Rendering state for one streamed client response: every method returns the frames to send.
pub trait StreamFraming: Send {
    fn on_chunk(&mut self, chunk: &SemanticChunk) -> Vec<String>;
    /// One iteration's end: its finished server-tool calls and the messages a client appends to
    /// continue from it. Messages renders the `tool_result` blocks (their `tool_use` already
    /// streamed) and needs no continuation history; CC has no native shape for either.
    fn emit_completed_iteration(&mut self, iteration: &CompletedIteration<'_>) -> Vec<String>;
    /// One server tool the moment it is dispatched, so a CC client can show the arguments while the
    /// call runs. Messages already streamed the same call as a native `tool_use` block.
    fn emit_dispatched_server_tool(
        &mut self,
        iteration: u32,
        server_call: &ServerToolCall,
    ) -> Vec<String>;
    /// Client-executed tool calls on the terminal iteration. CC emits them natively; Messages
    /// already streamed them as `tool_use` blocks.
    fn emit_client_tool_calls(&mut self, calls: &[ToolCall]) -> Vec<String>;
    /// Where an iteration scope lands: a frame of its own (CC) or the next frame that preserves
    /// extras (Messages, Responses).
    fn stage_iteration(&mut self, staged: StagedIteration<'_>) -> Vec<String>;

    fn emit_iteration_usage(&mut self, iteration: u32, usage: &CompletionUsage) -> Vec<String> {
        self.stage_iteration(StagedIteration {
            index: iteration,
            usage: Some(usage),
            debug_msg: None,
        })
    }

    fn emit_iteration_debug_msg(&mut self, iteration: u32, message: &str) -> Vec<String> {
        self.stage_iteration(StagedIteration {
            index: iteration,
            usage: None,
            debug_msg: Some(message),
        })
    }
    /// Close the response with terminal stop, cumulative usage, and the request-level server-tool
    /// transcript (every completed call, in dispatch order across iterations).
    fn finish(
        &mut self,
        termination: Termination,
        usage: &CompletionUsage,
        server_tool_calls: &[ServerToolCallOutcome],
    ) -> Vec<String>;
    /// The terminal error frame, mid-stream. On the live framing (`&mut self`), not the stateless
    /// envelope: Responses' `sequence_number` must continue from the frames already sent.
    fn error_sse_frame(&mut self, error_code: Option<&str>, message: &str) -> String;
    /// Terminal backend failure: the stream is over and this error is why. CC and Messages render
    /// their single protocol error frame; Responses overrides with the spec failure envelope
    /// (`response.failed` carrying the partial output) and the truncation rescue.
    fn finish_with_backend_error(&mut self, error: &BackendError) -> Vec<String> {
        vec![self.error_sse_frame(error.error_code.as_deref(), &error.message)]
    }
}

/// One client protocol's response shapes: a streaming renderer and the buffered body. Holds the
/// request's [`CodingAdapter`], if any, so neither the loop nor the egress has to carry it.
pub trait ProtocolEnvelope: Send + Sync {
    fn stream_framing(self: Box<Self>, model: String) -> Box<dyn StreamFraming>;
    /// The whole non-streaming response body, already encoded.
    fn buffered_body(&self, response: &BufferedResponse<'_>) -> String;
}

impl ClientProtocol {
    /// `responses_params` are the Responses request's echo-back params
    /// ([`AdaptedIngress::responses_params`](crate::request::AdaptedIngress)); the other
    /// protocols ignore them, pass `Default::default()`.
    pub fn envelope(
        self,
        coding_adapter: Option<Box<dyn CodingAdapter>>,
        responses_params: ResponsesParams,
    ) -> Box<dyn ProtocolEnvelope> {
        match self {
            Self::ChatCompletions => Box::new(cc::CcEnvelope),
            Self::Messages => Box::new(messages::MessagesEnvelope::new(coding_adapter)),
            Self::Responses => Box::new(responses::ResponsesEnvelope::new(
                coding_adapter,
                responses_params,
            )),
        }
    }

    /// The protocol's JSON error object: OpenAI `{"error": {...}}` / Anthropic error body. A
    /// function of the protocol alone, so the pre-flight path reaches it with no request in flight.
    /// Messages has no code slot: Anthropic's `error.type` is a closed vocabulary, not ours to
    /// extend.
    pub fn error_json(self, error_code: Option<&str>, message: &str) -> String {
        match self {
            Self::ChatCompletions | Self::Responses => openai_error_json(error_code, message),
            Self::Messages => messages::error_json(message),
        }
    }
}

/// One iteration that has finished, tool calls included. Both edges take the same borrow, so they
/// cannot disagree about which iteration a result belongs to.
pub struct CompletedIteration<'a> {
    pub index: u32,
    /// Empty when the model called no server tool in this iteration.
    pub invocations: &'a [ToolInvocation],
    /// This iteration's slice of the loop transcript: its assistant message plus the `tool` results
    /// answering it.
    pub continuation_messages: &'a [CcMessage],
}

impl CompletedIteration<'_> {
    pub(crate) fn server_tool_call_outcomes(&self) -> Vec<ServerToolCallOutcome> {
        self.invocations
            .iter()
            .map(
                |ToolInvocation {
                     server_call,
                     output,
                 }| { ServerToolCallOutcome::completed(server_call, output) },
            )
            .collect()
    }

    pub(crate) fn server_tool_calls(&self) -> Vec<ServerToolCallRecord> {
        self.invocations
            .iter()
            .map(
                |ToolInvocation {
                     server_call,
                     output,
                 }| { ServerToolCallRecord::completed(server_call, output) },
            )
            .collect()
    }
}

/// A finished non-streaming response, protocol-neutral. Each [`ProtocolEnvelope`] renders it into its
/// own body shape.
pub struct BufferedResponse<'a> {
    pub model: &'a str,
    /// Everything the loop appended, in order: one assistant message per iteration, each followed by
    /// the `tool` results answering it. The single source for the body's content — a second ordered
    /// accumulation could disagree with it about what the model saw.
    pub transcript: &'a [CcMessage],
    /// The calls the client must execute, which the transcript declares but does not answer.
    pub client_tool_calls: &'a [ToolCall],
    /// Recorded in canonical CC form, as everything between the two edges is; each protocol maps it
    /// into its own usage shape when rendering.
    pub iterations: &'a [IterationScope<CompletionUsage>],
    /// The request-level server-tool transcript, for the `request` scope.
    pub request_server_tool_calls: &'a [ServerToolCallOutcome],
    pub termination: Termination,
    pub usage: &'a CompletionUsage,
}

impl BufferedResponse<'_> {
    /// The ids of the server-tool calls that failed. The CC transcript's `tool` messages carry no
    /// error flag, so buffered renderers read it from the iteration records.
    fn failed_server_tool_call_ids(&self) -> HashSet<&str> {
        self.iterations
            .iter()
            .flat_map(|iteration| &iteration.server_tool_calls)
            .filter(|record| record.is_error == Some(true))
            .map(|record| record.id.as_str())
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::baseten_response_extension::ServerToolCallRecord;
    use crate::coding_adapter::CodingAdapter;
    use crate::history::MessageHistoryAccumulator;
    use crate::model::{ServerToolCallStatus, ToolOutput};
    use crate::test_utils::FakeResponsesSearchAdapter;
    use dynamo_protocols::types::FinishReason;
    use serde_json::Value;
    use serde_json::json;

    fn usage() -> CompletionUsage {
        serde_json::from_value(
            json!({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}),
        )
        .unwrap()
    }

    fn dispatched_server_call() -> crate::model::ServerToolCall {
        crate::test_utils::server_tool_call(server_call())
    }

    fn server_call() -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: "baseten__parallel__ws".into(),
            args: json!({"q": "x"}),
            raw_args: r#"{"q":"x"}"#.into(),
        }
    }

    /// The transcript two iterations produce: one that reasons, answers, and calls a server tool, and
    /// one that answers. Built through the accumulator, so the fixture cannot drift from what the loop
    /// actually appends.
    fn two_iteration_transcript() -> Vec<CcMessage> {
        let mut accumulator = MessageHistoryAccumulator::new(vec![]);
        accumulator.push(&SemanticChunk::ThinkingDelta("let me search".into()));
        accumulator.push(&SemanticChunk::TextDelta("searching now".into()));
        accumulator.push(&SemanticChunk::ToolCall(server_call()));
        accumulator.commit_assistant_message();
        accumulator.append_tool_results(&[ToolInvocation {
            server_call: dispatched_server_call(),
            output: ToolOutput {
                content: json!("RESULT"),
                status: ServerToolCallStatus::Succeeded,
                billable: true,
                sku: None,
            },
        }]);
        accumulator.push(&SemanticChunk::TextDelta("final answer".into()));
        accumulator.commit_assistant_message();
        accumulator.transcript().to_vec()
    }

    /// One iteration whose server tool failed, transcript and record agreeing on the outcome —
    /// the shape [`crate::react_loop::dispatch`]'s Failed arm actually produces.
    fn failed_tool_iteration() -> (Vec<CcMessage>, Vec<IterationScope<CompletionUsage>>) {
        let mut accumulator = MessageHistoryAccumulator::new(vec![]);
        accumulator.push(&SemanticChunk::ToolCall(server_call()));
        accumulator.commit_assistant_message();
        accumulator.append_tool_results(&[ToolInvocation {
            server_call: dispatched_server_call(),
            output: ToolOutput {
                content: json!("tool execution failed: connect timeout"),
                status: ServerToolCallStatus::Failed,
                billable: true,
                sku: None,
            },
        }]);
        accumulator.push(&SemanticChunk::TextDelta("could not search".into()));
        accumulator.commit_assistant_message();
        let iterations = vec![IterationScope {
            debug_msg: Vec::new(),
            index: 0,
            usage: Some(usage()),
            server_tool_calls: vec![ServerToolCallRecord::completed(
                &dispatched_server_call(),
                &ToolOutput {
                    content: json!("tool execution failed: connect timeout"),
                    status: ServerToolCallStatus::Failed,
                    billable: true,
                    sku: None,
                },
            )],
            continuation_messages: Vec::new(),
        }];
        (accumulator.transcript().to_vec(), iterations)
    }

    /// A failed server tool renders as failed in the buffered body: the CC transcript's `tool`
    /// messages carry no error flag, so the renderer reads it from the iteration records.
    #[test]
    fn buffered_failed_server_tool_renders_as_failed() {
        let (transcript, iterations) = failed_tool_iteration();
        let responses_body = buffered_full(
            ClientProtocol::Responses,
            &transcript,
            &[],
            &iterations,
            Termination::Model(FinishReason::Stop),
        );
        let mcp_call = &responses_body["output"][0];
        assert_eq!(mcp_call["type"], "mcp_call");
        assert_eq!(mcp_call["status"], "failed");
        assert_eq!(mcp_call["error"], "tool execution failed: connect timeout");

        let (transcript, iterations) = failed_tool_iteration();
        let messages_body = buffered_full(
            ClientProtocol::Messages,
            &transcript,
            &[],
            &iterations,
            Termination::Model(FinishReason::Stop),
        );
        let tool_result = &messages_body["content"][1];
        assert_eq!(tool_result["type"], "tool_result");
        assert_eq!(tool_result["is_error"], true);
    }

    #[test]
    fn buffered_failed_web_search_renders_as_failed() {
        let (transcript, iterations) = failed_tool_iteration();
        let adapter: Box<dyn CodingAdapter> = Box::new(FakeResponsesSearchAdapter {
            tool_name: "baseten__parallel__ws",
        });
        let body = buffered_full_with_adapter(
            ClientProtocol::Responses,
            &transcript,
            &[],
            &iterations,
            Termination::Model(FinishReason::Stop),
            Some(adapter),
        );
        assert_eq!(body["output"][0]["type"], "web_search_call");
        assert_eq!(body["output"][0]["status"], "failed");
    }

    fn frame_json(frame: &str) -> Value {
        let data_line = frame
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .unwrap();
        serde_json::from_str(data_line).unwrap()
    }

    /// Where a staged iteration scope lands, per protocol: a frame of its own for CC, the next
    /// extras-preserving frame for Messages, the terminal envelope's `response.baseten` for
    /// Responses (nothing else follows in this sequence).
    #[test]
    fn staged_iteration_scope_lands_on_each_protocols_carrier() {
        let usage = usage();

        let mut cc = ClientProtocol::ChatCompletions
            .envelope(None, Default::default())
            .stream_framing("m".to_string());
        let frames = cc.emit_iteration_debug_msg(0, "appended steering message");
        assert_eq!(frames.len(), 1, "CC stages on a frame of its own");
        let iteration = &frame_json(&frames[0])["baseten"]["iterations"][0];
        assert_eq!(iteration["index"], 0);
        assert_eq!(iteration["debug_msg"], json!(["appended steering message"]));

        let mut messages = ClientProtocol::Messages
            .envelope(None, Default::default())
            .stream_framing("m".to_string());
        let opening_frames = messages.on_chunk(&SemanticChunk::TextDelta("hi".into()));
        assert!(!opening_frames.is_empty());
        assert!(
            messages
                .emit_iteration_debug_msg(0, "appended steering message")
                .is_empty(),
            "Messages stages silently, riding the next preserving frame"
        );
        let finish_frames = messages.finish(Termination::Model(FinishReason::Stop), &usage, &[]);
        let carrier = finish_frames
            .iter()
            .map(|frame| frame_json(frame))
            .find(|body| body["baseten"]["iterations"][0]["index"] == json!(0))
            .expect("a later preserving frame carries the staged scope");
        assert_eq!(
            carrier["baseten"]["iterations"][0]["debug_msg"],
            json!(["appended steering message"])
        );

        let mut responses = ClientProtocol::Responses
            .envelope(None, Default::default())
            .stream_framing("m".to_string());
        let opening_frames = responses.on_chunk(&SemanticChunk::TextDelta("hi".into()));
        assert!(!opening_frames.is_empty());
        assert!(
            responses
                .emit_iteration_debug_msg(0, "appended steering message")
                .is_empty(),
            "Responses stages silently, riding the next preserving frame"
        );
        let finish_frames = responses.finish(Termination::Model(FinishReason::Stop), &usage, &[]);
        let carrier = finish_frames
            .iter()
            .map(|frame| frame_json(frame))
            .find_map(|body| {
                [&body["baseten"], &body["response"]["baseten"]]
                    .into_iter()
                    .find(|baseten| baseten["iterations"][0]["index"] == json!(0))
                    .cloned()
            })
            .expect("a later preserving frame carries the staged scope");
        assert_eq!(
            carrier["iterations"][0]["debug_msg"],
            json!(["appended steering message"])
        );
    }

    /// The terminal `request` scope carries the whole server-tool transcript — `sku` and
    /// `is_error` included — on every protocol, streaming and buffered.
    #[test]
    fn request_scope_carries_the_server_tool_transcript_on_every_protocol() {
        let usage = usage();
        let mut records = vec![ServerToolCallOutcome::completed(
            &dispatched_server_call(),
            &ToolOutput {
                content: json!("RESULT"),
                status: ServerToolCallStatus::Succeeded,
                billable: true,
                sku: Some("search-pro".into()),
            },
        )];
        records.push(ServerToolCallOutcome::completed(
            &dispatched_server_call(),
            &ToolOutput {
                content: json!("not executed: budget spent"),
                status: ServerToolCallStatus::Refused,
                billable: false,
                sku: None,
            },
        ));
        let request_scope_of = |baseten_carriers: [&Value; 2]| {
            baseten_carriers
                .into_iter()
                .find(|baseten| !baseten["request"]["server_tool_calls"].is_null())
                .map(|baseten| baseten["request"].clone())
        };

        for protocol in [
            ClientProtocol::ChatCompletions,
            ClientProtocol::Messages,
            ClientProtocol::Responses,
        ] {
            let mut framing = protocol
                .envelope(None, Default::default())
                .stream_framing("m".to_string());
            framing.on_chunk(&SemanticChunk::TextDelta("hi".into()));
            let finish_frames =
                framing.finish(Termination::Model(FinishReason::Stop), &usage, &records);
            let request_scope = finish_frames
                .iter()
                .map(|frame| frame_json(frame))
                .find_map(|body| request_scope_of([&body["baseten"], &body["response"]["baseten"]]))
                .unwrap_or_else(|| {
                    panic!("{protocol:?}: no terminal frame carries the request scope")
                });
            assert_eq!(request_scope["server_tool_calls"][0]["id"], "c1");
            assert_eq!(request_scope["server_tool_calls"][0]["sku"], "search-pro");
            assert_eq!(request_scope["server_tool_calls"][0]["status"], "succeeded");
            assert_eq!(request_scope["server_tool_calls"][0]["billable"], true);
            assert_eq!(request_scope["server_tool_calls"][1]["status"], "refused");
            assert_eq!(request_scope["server_tool_calls"][1]["billable"], false);
            assert!(request_scope["server_tool_calls"][1]["sku"].is_null());

            let body =
                protocol
                    .envelope(None, Default::default())
                    .buffered_body(&BufferedResponse {
                        model: "m",
                        transcript: &two_iteration_transcript(),
                        client_tool_calls: &[],
                        iterations: &[],
                        request_server_tool_calls: &records,
                        termination: Termination::Model(FinishReason::Stop),
                        usage: &usage,
                    });
            let body: Value = serde_json::from_str(&body).unwrap();
            let request_scope = request_scope_of([&body["baseten"], &body["response"]["baseten"]])
                .unwrap_or_else(|| panic!("{protocol:?}: buffered body misses the request scope"));
            assert_eq!(request_scope["server_tool_calls"][0]["sku"], "search-pro");
        }
    }

    /// The stagers' `!self.started` guard, claimed unreachable given the loop's cadence: a scope
    /// staged before any frame is dropped (with a warning), not deferred — the terminal frames
    /// carry no iteration entry for it.
    #[test]
    fn iteration_scope_staged_before_start_is_dropped_not_deferred() {
        let usage = usage();
        for protocol in [ClientProtocol::Messages, ClientProtocol::Responses] {
            let mut framing = protocol
                .envelope(None, Default::default())
                .stream_framing("m".to_string());
            assert!(framing.emit_iteration_debug_msg(0, "too early").is_empty());
            let finish_frames = framing.finish(Termination::Model(FinishReason::Stop), &usage, &[]);
            for frame in &finish_frames {
                let body = frame_json(frame);
                for scope_root in [&body["baseten"], &body["response"]["baseten"]] {
                    assert!(
                        scope_root["iterations"]
                            .as_array()
                            .is_none_or(|iterations| {
                                iterations.iter().all(|i| i["debug_msg"].is_null())
                            }),
                        "{protocol:?}: dropped scope resurfaced: {body}"
                    );
                }
            }
        }
    }

    /// CC's error frame is the bare OpenAI `{"error": {...}}` object; Messages' is Anthropic's
    /// named `event: error`. (Responses' typed error event is pinned in its own module's tests.)
    #[test]
    fn cc_and_messages_error_frames_carry_their_protocol_shapes() {
        let cc_frame = ClientProtocol::ChatCompletions
            .envelope(None, Default::default())
            .stream_framing("m".to_string())
            .error_sse_frame(Some("upstream_error"), "upstream unavailable");
        let cc_body: Value =
            serde_json::from_str(cc_frame.strip_prefix("data: ").unwrap().trim_end()).unwrap();
        assert_eq!(cc_body["error"]["message"], "upstream unavailable");
        assert_eq!(cc_body["error"]["type"], "api_error");
        assert_eq!(cc_body["error"]["code"], "upstream_error");

        let messages_frame = ClientProtocol::Messages
            .envelope(None, Default::default())
            .stream_framing("m".to_string())
            .error_sse_frame(Some("upstream_error"), "upstream unavailable");
        let (event_line, data_line) = messages_frame.trim_end().split_once('\n').unwrap();
        assert_eq!(event_line, "event: error");
        let messages_body: Value =
            serde_json::from_str(data_line.strip_prefix("data: ").unwrap()).unwrap();
        assert_eq!(messages_body["type"], "error");
        assert_eq!(messages_body["error"]["message"], "upstream unavailable");
    }

    fn buffered(protocol: ClientProtocol, transcript: &[CcMessage]) -> Value {
        buffered_with(protocol, transcript, Termination::Model(FinishReason::Stop))
    }

    fn buffered_with(
        protocol: ClientProtocol,
        transcript: &[CcMessage],
        termination: Termination,
    ) -> Value {
        buffered_full(protocol, transcript, &[], &[], termination)
    }

    fn buffered_full(
        protocol: ClientProtocol,
        transcript: &[CcMessage],
        client_tool_calls: &[ToolCall],
        iterations: &[IterationScope<CompletionUsage>],
        termination: Termination,
    ) -> Value {
        buffered_full_with_adapter(
            protocol,
            transcript,
            client_tool_calls,
            iterations,
            termination,
            None,
        )
    }

    fn buffered_full_with_adapter(
        protocol: ClientProtocol,
        transcript: &[CcMessage],
        client_tool_calls: &[ToolCall],
        iterations: &[IterationScope<CompletionUsage>],
        termination: Termination,
        coding_adapter: Option<Box<dyn CodingAdapter>>,
    ) -> Value {
        let usage = usage();
        let body = protocol
            .envelope(coding_adapter, Default::default())
            .buffered_body(&BufferedResponse {
                model: "m",
                transcript,
                client_tool_calls,
                iterations,
                request_server_tool_calls: &[],
                termination,
                usage: &usage,
            });
        serde_json::from_str(&body).unwrap()
    }

    fn one_iteration() -> Vec<IterationScope<CompletionUsage>> {
        vec![IterationScope {
            debug_msg: Vec::new(),
            index: 0,
            usage: Some(usage()),
            server_tool_calls: vec![ServerToolCallRecord::completed(
                &dispatched_server_call(),
                &ToolOutput {
                    content: json!("RESULT"),
                    status: ServerToolCallStatus::Succeeded,
                    billable: true,
                    sku: Some("search.pro".to_string()),
                },
            )],
            continuation_messages: two_iteration_transcript(),
        }]
    }

    /// The scopes are one set of types, so the only fields that may differ between the protocols are
    /// the CC-only tool activity and each protocol's own usage shape. Anything else diverging is a
    /// rendering path that drifted.
    #[test]
    fn both_protocols_render_the_same_extension_apart_from_the_cc_tool_activity() {
        let iterations = one_iteration();
        let extension = |protocol| {
            buffered_full(
                protocol,
                &[],
                &[],
                &iterations,
                Termination::ReactCapExhausted,
            )["baseten"]
                .clone()
        };
        let mut cc = extension(ClientProtocol::ChatCompletions);
        let mut messages = extension(ClientProtocol::Messages);

        assert!(
            !cc["iterations"][0]["continuation_messages"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        assert!(
            messages["iterations"][0]
                .get("continuation_messages")
                .is_none()
        );
        assert!(messages["iterations"][0].get("server_tool_calls").is_none());
        for divergent in ["continuation_messages", "server_tool_calls", "usage"] {
            for extension in [&mut cc, &mut messages] {
                extension["iterations"][0]
                    .as_object_mut()
                    .unwrap()
                    .remove(divergent);
            }
        }
        assert_eq!(cc, messages);
    }

    /// The buffered body is a CC client's only channel for the loop's transcript: the transcript itself
    /// renders into one concatenated `content`, so without this the calls it paid for leave no trace.
    #[test]
    fn buffered_cc_carries_the_transcript_per_iteration() {
        let body = buffered_full(
            ClientProtocol::ChatCompletions,
            &two_iteration_transcript(),
            &[],
            &one_iteration(),
            Termination::Model(FinishReason::Stop),
        );
        let iteration = &body["baseten"]["iterations"][0];
        assert_eq!(iteration["index"], 0);
        assert_eq!(iteration["server_tool_calls"][0]["id"], "c1");
        assert_eq!(iteration["server_tool_calls"][0]["is_error"], false);
        assert_eq!(iteration["server_tool_calls"][0]["sku"], "search.pro");
        assert_eq!(iteration["continuation_messages"][1]["tool_call_id"], "c1");
        assert_eq!(iteration["usage"]["completion_tokens"], 2);
    }

    /// A provider that reports no sku must leave the field off the wire, not send an empty string.
    #[test]
    fn an_unreported_sku_is_absent_from_the_frame() {
        let record = ServerToolCallRecord::completed(
            &dispatched_server_call(),
            &ToolOutput {
                content: json!("RESULT"),
                status: ServerToolCallStatus::Succeeded,
                billable: true,
                sku: None,
            },
        );
        let serialized = serde_json::to_value(&record).expect("record serializes");
        assert!(serialized.get("sku").is_none(), "{serialized}");
    }

    /// The transcript's own order carries into the Messages body: which text came before which tool
    /// call, and each `tool_use` before the `tool_result` that answers it. A client echoing this back
    /// replays exactly what the model saw.
    #[test]
    fn buffered_messages_renders_the_transcript_in_order() {
        let body = buffered(ClientProtocol::Messages, &two_iteration_transcript());
        let blocks = body["content"].as_array().unwrap();
        let kinds: Vec<&str> = blocks.iter().map(|b| b["type"].as_str().unwrap()).collect();
        assert_eq!(
            kinds,
            ["thinking", "text", "tool_use", "tool_result", "text"]
        );
        assert_eq!(blocks[1]["text"], "searching now");
        assert_eq!(blocks[2]["id"], "c1");
        assert_eq!(blocks[2]["input"]["q"], "x");
        assert_eq!(blocks[3]["tool_use_id"], "c1");
        assert_eq!(blocks[3]["content"], "RESULT");
        assert!(
            blocks[3].get("is_error").is_none(),
            "success omits is_error"
        );
        assert_eq!(blocks[4]["text"], "final answer");
        assert_eq!(body["stop_reason"], "end_turn");
        assert_eq!(body["usage"]["output_tokens"], 2);
    }

    /// CC has one `content` field for the whole request, so every iteration's text concatenates there
    /// and the transcript is the only place their boundaries survive.
    #[test]
    fn buffered_cc_concatenates_text_across_iterations() {
        let body = buffered(ClientProtocol::ChatCompletions, &two_iteration_transcript());
        let message = &body["choices"][0]["message"];
        assert_eq!(message["content"], "searching nowfinal answer");
        assert_eq!(message["reasoning_content"], "let me search");
        assert!(
            message.get("tool_calls").is_none(),
            "the server tool is not the client's to run"
        );
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn buffered_responses_reasoning_carries_the_reasoning_text_discriminator() {
        let body = buffered(ClientProtocol::Responses, &two_iteration_transcript());
        let reasoning = body["output"]
            .as_array()
            .expect("output items")
            .iter()
            .find(|item| item["type"] == "reasoning")
            .expect("a reasoning output item");
        assert_eq!(reasoning["content"][0]["type"], "reasoning_text");
        assert_eq!(reasoning["content"][0]["text"], "let me search");
    }

    /// Buffered bodies carry the termination too, not just the streaming frames, in both protocols.
    #[test]
    fn buffered_react_cap_maps_to_in_enum_stop_with_termination_reason() {
        let messages = buffered_with(
            ClientProtocol::Messages,
            &[],
            Termination::ReactCapExhausted,
        );
        assert_eq!(messages["stop_reason"], "pause_turn");
        assert_eq!(
            messages["baseten"]["request"]["termination_reason"],
            "max_react_iterations_reached"
        );

        let cc = buffered_with(
            ClientProtocol::ChatCompletions,
            &[],
            Termination::ReactCapExhausted,
        );
        assert_eq!(cc["choices"][0]["finish_reason"], "length");
        assert_eq!(
            cc["baseten"]["request"]["termination_reason"],
            "max_react_iterations_reached"
        );
    }

    /// A client tool call is the one tool shape CC carries natively. It is also in the transcript, so
    /// the client answers it after appending — and must not append `choices` on top.
    #[test]
    fn buffered_cc_emits_client_tool_calls_with_null_content() {
        let call = ToolCall {
            id: "c9".into(),
            name: "get_weather".into(),
            args: json!({"city": "SF"}),
            raw_args: r#"{"city":"SF"}"#.into(),
        };
        let body = buffered_full(
            ClientProtocol::ChatCompletions,
            &[],
            std::slice::from_ref(&call),
            &[],
            Termination::Model(FinishReason::ToolCalls),
        );
        let message = &body["choices"][0]["message"];
        assert_eq!(message["content"], Value::Null);
        assert_eq!(message["tool_calls"][0]["id"], "c9");
        assert_eq!(message["tool_calls"][0]["type"], "function");
        // Verbatim model bytes, not a reserialize.
        assert_eq!(
            message["tool_calls"][0]["function"]["arguments"],
            r#"{"city":"SF"}"#
        );
    }
}
