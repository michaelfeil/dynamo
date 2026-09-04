//! Client egress: the ReAct loop fans each [`SemanticChunk`] here (no tee). `Streaming` renders
//! per-chunk SSE via [`SseEmitter`]; `Buffered` renders one non-streaming body at `finish`
//! (`stream:false` / async) from the loop transcript. The model call is always streamed regardless.
//! Uniform methods so the loop drives both without branching.

use dynamo_protocols::types::CompletionUsage;

use crate::baseten_response_extension::{
    IterationScope, ServerToolCallOutcome, ServerToolCallRecord,
};
use crate::framing::{BufferedResponse, CompletedIteration, ProtocolEnvelope};
use crate::model::{BackendError, ServerToolCall, Termination, ToolCall, TranslationError};
use crate::sse_emitter::SseEmitter;
use crate::{CcMessage, SemanticChunk, SseFrameTx};

pub enum ClientEgress {
    Streaming(SseEmitter),
    Buffered(BufferedEgress),
}

impl ClientEgress {
    /// Buffered ignores chunks: its body renders from the transcript, which the history accumulator
    /// folds from the same chunks.
    pub async fn push(&mut self, chunk: &SemanticChunk) -> Result<(), TranslationError> {
        match self {
            Self::Streaming(emitter) => emitter.push(chunk).await,
            Self::Buffered(_) => Ok(()),
        }
    }

    pub async fn emit_completed_iteration(
        &mut self,
        iteration: &CompletedIteration<'_>,
    ) -> Result<(), TranslationError> {
        match self {
            Self::Streaming(emitter) => emitter.emit_completed_iteration(iteration).await,
            Self::Buffered(buffered) => {
                buffered.record_completed_iteration(iteration);
                Ok(())
            }
        }
    }

    pub async fn emit_client_tool_calls(
        &mut self,
        calls: &[ToolCall],
    ) -> Result<(), TranslationError> {
        match self {
            Self::Streaming(emitter) => emitter.emit_client_tool_calls(calls).await,
            Self::Buffered(buffered) => {
                buffered.client_tool_calls.extend_from_slice(calls);
                Ok(())
            }
        }
    }

    /// One server tool as it is dispatched: CC shows the arguments while the call runs, Messages
    /// already streamed the native `tool_use` block.
    pub async fn emit_dispatched_server_tool(
        &mut self,
        iteration: u32,
        server_call: &ServerToolCall,
    ) -> Result<(), TranslationError> {
        match self {
            Self::Streaming(emitter) => {
                emitter
                    .emit_dispatched_server_tool(iteration, server_call)
                    .await
            }
            Self::Buffered(buffered) => {
                buffered
                    .iteration_scope(iteration)
                    .server_tool_calls
                    .push(ServerToolCallRecord::dispatched(server_call));
                Ok(())
            }
        }
    }

    pub async fn emit_iteration_usage(
        &mut self,
        iteration: u32,
        usage: &CompletionUsage,
    ) -> Result<(), TranslationError> {
        match self {
            Self::Streaming(emitter) => emitter.emit_iteration_usage(iteration, usage).await,
            Self::Buffered(buffered) => {
                buffered.iteration_scope(iteration).usage = Some(usage.clone());
                Ok(())
            }
        }
    }

    pub async fn emit_iteration_debug_msg(
        &mut self,
        iteration: u32,
        message: &str,
    ) -> Result<(), TranslationError> {
        match self {
            Self::Streaming(emitter) => emitter.emit_iteration_debug_msg(iteration, message).await,
            Self::Buffered(buffered) => {
                buffered
                    .iteration_scope(iteration)
                    .debug_msg
                    .push(message.to_owned());
                Ok(())
            }
        }
    }

    /// `transcript` is the loop's own messages, which only the buffered body needs: a streaming client
    /// already received them frame by frame.
    pub async fn finish(
        &mut self,
        termination: Termination,
        usage: &CompletionUsage,
        transcript: &[CcMessage],
    ) -> Result<(), TranslationError> {
        match self {
            Self::Streaming(emitter) => emitter.finish(termination, usage).await,
            Self::Buffered(buffered) => buffered.finish(termination, usage, transcript).await,
        }
    }

    pub fn is_disconnected(&self) -> bool {
        match self {
            Self::Streaming(emitter) => emitter.is_disconnected(),
            Self::Buffered(buffered) => buffered.tx.is_closed(),
        }
    }

    /// Terminal backend failure, streaming only (a buffered request reports errors as an HTTP
    /// error body): the protocol's failure envelope.
    pub async fn finish_with_backend_error(
        &mut self,
        error: &BackendError,
    ) -> Result<(), TranslationError> {
        match self {
            Self::Streaming(emitter) => emitter.finish_with_backend_error(error).await,
            Self::Buffered(_) => Ok(()),
        }
    }

    /// The terminal error frame, streaming only: a buffered request reports errors as an HTTP error
    /// body, not in-band.
    pub fn streaming_error_sse_frame(
        &mut self,
        error_code: Option<&str>,
        message: &str,
    ) -> Option<String> {
        match self {
            Self::Streaming(emitter) => Some(emitter.error_sse_frame(error_code, message)),
            Self::Buffered(_) => None,
        }
    }
}

/// Records what the transcript does not carry, then renders one non-streaming body at `finish`. The
/// content itself is derived from the loop transcript, so nothing here can disagree with it about
/// what the model saw.
pub struct BufferedEgress {
    tx: SseFrameTx,
    envelope: Box<dyn ProtocolEnvelope>,
    model: String,
    /// The transcript declares these but does not answer them: they are the client's to execute.
    client_tool_calls: Vec<ToolCall>,
    /// Ordered by iteration index, so the body carries the loop's own sequence. A CC client has no
    /// other way to see it: the transcript renders into one concatenated `content`.
    iterations: Vec<IterationScope<CompletionUsage>>,
    /// The request-level transcript for the terminal `request` scope, accumulated the same way the
    /// streaming emitter does so the two paths cannot drift.
    request_server_tool_calls: Vec<ServerToolCallOutcome>,
}

impl BufferedEgress {
    pub fn new(tx: SseFrameTx, envelope: Box<dyn ProtocolEnvelope>, model: String) -> Self {
        Self {
            tx,
            envelope,
            model,
            client_tool_calls: Vec::new(),
            iterations: Vec::new(),
            request_server_tool_calls: Vec::new(),
        }
    }

    /// The loop lends the invocations to every consumer, so the buffered body needs its own copy to
    /// render at `finish`, long after the borrow ends.
    fn record_completed_iteration(&mut self, iteration: &CompletedIteration<'_>) {
        self.request_server_tool_calls
            .extend(iteration.server_tool_call_outcomes());
        let scope = self.iteration_scope(iteration.index);
        // Replaces the dispatch-time records, same calls plus their `is_error`.
        scope.server_tool_calls = iteration.server_tool_calls();
        scope.continuation_messages = iteration.continuation_messages.to_vec();
    }

    /// This iteration's scope, appended on first mention: usage and tool results arrive on separate
    /// calls, and both belong to the same entry.
    fn iteration_scope(&mut self, index: u32) -> &mut IterationScope<CompletionUsage> {
        let position = self
            .iterations
            .iter()
            .position(|scope| scope.index == index)
            .unwrap_or_else(|| {
                self.iterations.push(IterationScope::at(index));
                self.iterations.len() - 1
            });
        &mut self.iterations[position]
    }

    async fn finish(
        &mut self,
        termination: Termination,
        usage: &CompletionUsage,
        transcript: &[CcMessage],
    ) -> Result<(), TranslationError> {
        let body = self.envelope.buffered_body(&BufferedResponse {
            model: &self.model,
            transcript,
            client_tool_calls: &self.client_tool_calls,
            iterations: &self.iterations,
            request_server_tool_calls: &self.request_server_tool_calls,
            termination,
            usage,
        });
        self.tx
            .send(body)
            .await
            .map_err(|_| TranslationError::ClientDisconnected)
    }
}
