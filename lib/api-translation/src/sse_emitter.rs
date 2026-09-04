//! The streaming client edge: hands each rendered frame to the egress channel. All protocol
//! rendering lives behind [`StreamFraming`] (see [`super::framing`]); this is the only piece that
//! awaits, and it holds no protocol knowledge at all.
//!
//! The single boundary source ([`super::sse_parser::SseParser`]) already drew every semantic
//! boundary, so nothing here re-derives one.

use dynamo_protocols::types::CompletionUsage;

use crate::baseten_response_extension::ServerToolCallOutcome;
use crate::framing::{CompletedIteration, StreamFraming};
use crate::model::{BackendError, ServerToolCall, Termination, ToolCall, TranslationError};
use crate::{SemanticChunk, SseFrameTx};

pub struct SseEmitter {
    tx: SseFrameTx,
    framing: Box<dyn StreamFraming>,
    /// The request-level transcript, folded into the terminal frame's `request` scope. Streaming
    /// renders and forgets each frame, so this is the only record.
    request_server_tool_calls: Vec<ServerToolCallOutcome>,
}

impl SseEmitter {
    pub fn new(tx: SseFrameTx, framing: Box<dyn StreamFraming>) -> Self {
        Self {
            tx,
            framing,
            request_server_tool_calls: Vec::new(),
        }
    }

    pub async fn push(&mut self, chunk: &SemanticChunk) -> Result<(), TranslationError> {
        Self::send_all(&self.tx, self.framing.on_chunk(chunk)).await
    }

    pub async fn emit_completed_iteration(
        &mut self,
        iteration: &CompletedIteration<'_>,
    ) -> Result<(), TranslationError> {
        self.request_server_tool_calls
            .extend(iteration.server_tool_call_outcomes());
        Self::send_all(&self.tx, self.framing.emit_completed_iteration(iteration)).await
    }

    pub async fn emit_dispatched_server_tool(
        &mut self,
        iteration: u32,
        server_call: &ServerToolCall,
    ) -> Result<(), TranslationError> {
        Self::send_all(
            &self.tx,
            self.framing
                .emit_dispatched_server_tool(iteration, server_call),
        )
        .await
    }

    pub async fn emit_client_tool_calls(
        &mut self,
        calls: &[ToolCall],
    ) -> Result<(), TranslationError> {
        Self::send_all(&self.tx, self.framing.emit_client_tool_calls(calls)).await
    }

    pub async fn emit_iteration_usage(
        &mut self,
        iteration: u32,
        usage: &CompletionUsage,
    ) -> Result<(), TranslationError> {
        Self::send_all(
            &self.tx,
            self.framing.emit_iteration_usage(iteration, usage),
        )
        .await
    }

    pub async fn emit_iteration_debug_msg(
        &mut self,
        iteration: u32,
        message: &str,
    ) -> Result<(), TranslationError> {
        Self::send_all(
            &self.tx,
            self.framing.emit_iteration_debug_msg(iteration, message),
        )
        .await
    }

    pub async fn finish(
        &mut self,
        termination: Termination,
        usage: &CompletionUsage,
    ) -> Result<(), TranslationError> {
        Self::send_all(
            &self.tx,
            self.framing
                .finish(termination, usage, &self.request_server_tool_calls),
        )
        .await
    }

    /// Terminal backend failure: the protocol's failure envelope (see
    /// [`StreamFraming::finish_with_backend_error`]). Ends the response; nothing follows.
    pub async fn finish_with_backend_error(
        &mut self,
        error: &BackendError,
    ) -> Result<(), TranslationError> {
        Self::send_all(&self.tx, self.framing.finish_with_backend_error(error)).await
    }

    pub fn is_disconnected(&self) -> bool {
        self.tx.is_closed()
    }

    /// Returned, not sent: the caller bounds the send with its own timeout ([`super::SseFrameTx`]
    /// reserve), which `send_all` does not.
    pub fn error_sse_frame(&mut self, error_code: Option<&str>, message: &str) -> String {
        self.framing.error_sse_frame(error_code, message)
    }

    /// Takes the channel rather than `&self`: a `&SseEmitter` held across the await would require the
    /// boxed framing to be `Sync`, which nothing here needs.
    async fn send_all(tx: &SseFrameTx, frames: Vec<String>) -> Result<(), TranslationError> {
        for frame in frames {
            tx.send(frame)
                .await
                .map_err(|_| TranslationError::ClientDisconnected)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "sse_emitter_test.rs"]
mod tests;
