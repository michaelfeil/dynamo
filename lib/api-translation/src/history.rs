//! The single owner of the ReAct loop's message history: the client's messages plus every model
//! call's assistant message and the server-tool `tool` results, as canonical [`CcMessage`]s. It owns
//! the whole dialogue because each iteration's predict body replays the full prefix.
//! Single-owner by design (the loop holds it, lends it by borrow); never shared.
//!
//! Ported from `DeltaAggregator` (per-turn fold: concat text/thinking, collect tool calls) in
//! basetenlabs/dynamo @ 68dec805 (Apache-2.0):
//! https://github.com/basetenlabs/dynamo/blob/68dec805e22e66851dc0bf2a8faba7ce0665b00d/lib/llm/src/protocols/openai/chat_completions/aggregator.rs
//! SPDX-License-Identifier: Apache-2.0.

use dynamo_protocols::types::{
    ChatCompletionMessageToolCall, ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage,
    ChatCompletionRequestToolMessage, ChatCompletionRequestToolMessageContent, FunctionCall,
    FunctionType, ReasoningContent,
};

use crate::model::{ToolCall, ToolInvocation};
use crate::{CcMessage, SemanticChunk};

/// Owns the full CC dialogue and folds the current model turn as its semantic chunks arrive.
pub struct MessageHistoryAccumulator {
    messages: Vec<CcMessage>,
    assistant_message: AssistantMessageBuffer,
    /// Where the client's own messages end and the loop's begin. Fixed at construction.
    client_prefix_end: usize,
    /// Where the current iteration's messages start, moved by [`Self::commit_assistant_message`].
    iteration_start: usize,
}

/// One CC assistant message under construction. Shared with the request edge, which rebuilds the
/// same messages from an echoed Messages history, so both directions produce byte-identical output
/// by construction — what the KV-cache prefix contract needs.
#[derive(Default)]
pub(crate) struct AssistantMessageBuffer {
    pub text: String,
    pub thinking: String,
    pub tool_calls: Vec<ToolCall>,
}

impl AssistantMessageBuffer {
    fn is_empty(&self) -> bool {
        self.text.is_empty() && self.thinking.is_empty() && self.tool_calls.is_empty()
    }

    /// Append as one assistant message and reset. An empty buffer appends nothing: it would alter
    /// the replayed prefix.
    pub(crate) fn flush_into(&mut self, messages: &mut Vec<CcMessage>) {
        if let Some(assistant) = std::mem::take(self).into_assistant_message() {
            messages.push(ChatCompletionRequestMessage::Assistant(assistant));
        }
    }

    /// `None` when nothing accumulated.
    pub(crate) fn into_assistant_message(self) -> Option<ChatCompletionRequestAssistantMessage> {
        if self.is_empty() {
            return None;
        }
        let tool_calls = (!self.tool_calls.is_empty())
            .then(|| self.tool_calls.into_iter().map(Into::into).collect());
        #[allow(deprecated)]
        Some(ChatCompletionRequestAssistantMessage {
            content: (!self.text.is_empty()).then_some(
                ChatCompletionRequestAssistantMessageContent::Text(self.text),
            ),
            // Replay fidelity for the KV-cache prefix: carry the model's reasoning verbatim so the
            // re-render matches; the chat template (not us) decides model-specific strip policy.
            reasoning_content: (!self.thinking.is_empty())
                .then_some(ReasoningContent::Text(self.thinking)),
            refusal: None,
            name: None,
            audio: None,
            tool_calls,
            partial: None,
            function_call: None,
        })
    }
}

/// A CC `role: tool` message. One builder for both edges: the loop appending its own tool results and
/// the request edge rebuilding them from an echoed Anthropic `tool_result` block.
pub(crate) fn tool_result_message(
    tool_call_id: String,
    content: ChatCompletionRequestToolMessageContent,
) -> CcMessage {
    ChatCompletionRequestMessage::Tool(ChatCompletionRequestToolMessage {
        content,
        tool_call_id,
    })
}

impl From<ToolCall> for ChatCompletionMessageToolCall {
    fn from(call: ToolCall) -> Self {
        Self {
            id: call.id,
            r#type: FunctionType::Function,
            function: FunctionCall {
                name: call.name,
                // Model's verbatim bytes, not a reserialize — byte-exact cache prefix.
                arguments: call.raw_args,
            },
        }
    }
}

impl MessageHistoryAccumulator {
    /// Seed with the client's messages (from `adapt_request`); loop turns append after them.
    pub fn new(initial: Vec<CcMessage>) -> Self {
        Self {
            client_prefix_end: initial.len(),
            iteration_start: initial.len(),
            messages: initial,
            assistant_message: AssistantMessageBuffer::default(),
        }
    }

    /// Fold one semantic chunk in. Text/thinking concat; a complete tool call is collected.
    /// Usage/Stop carry no history (usage accounting and loop control are the loop's).
    pub fn push(&mut self, chunk: &SemanticChunk) {
        match chunk {
            SemanticChunk::TextDelta(t) => self.assistant_message.text.push_str(t),
            SemanticChunk::ThinkingDelta(t) => self.assistant_message.thinking.push_str(t),
            SemanticChunk::ToolCall(tc) => self.assistant_message.tool_calls.push(tc.clone()),
            SemanticChunk::Usage(_) | SemanticChunk::Stop { .. } => {}
        }
    }

    /// Finalize the accumulated model output into one assistant [`CcMessage`] and append it. This is
    /// the iteration boundary in the history: [`Self::iteration_messages`] reads from here on.
    pub fn commit_assistant_message(&mut self) {
        self.iteration_start = self.messages.len();
        self.assistant_message.flush_into(&mut self.messages);
    }

    /// Append the server-tool results (as CC `role:tool` messages) after their assistant message, in
    /// call order (KV-cache prefix fidelity). `tool_call_id` pairs each to the model's tool id.
    pub fn append_tool_results(&mut self, invocations: &[ToolInvocation]) {
        for invocation in invocations {
            self.messages.push(tool_result_message(
                invocation.server_call.call.id.clone(),
                ChatCompletionRequestToolMessageContent::Text(invocation.output.text()),
            ));
        }
    }

    /// The full history, for the next iteration's predict body ([`build_next_request`]).
    pub fn messages(&self) -> &[CcMessage] {
        &self.messages
    }

    /// The current iteration's messages: its assistant message and the tool results appended after
    /// it, which is what a CC client appends to continue from that iteration. Read it after
    /// [`Self::append_tool_results`], or the call/result pair is still half-open.
    pub fn iteration_messages(&self) -> &[CcMessage] {
        &self.messages[self.iteration_start..]
    }

    /// Everything the loop appended, without the client's own messages: the whole transcript, which
    /// the buffered body renders and which concatenating every iteration's messages reproduces.
    pub fn transcript(&self) -> &[CcMessage] {
        &self.messages[self.client_prefix_end..]
    }
}

#[cfg(test)]
#[path = "history_test.rs"]
mod tests;
