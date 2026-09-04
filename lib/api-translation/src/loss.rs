//! What adaptation did not carry through. Every place ingress drops, skips, degrades, or folds a
//! part of the client's request records a [`Loss`] instead of (or as well as) logging it, so a
//! consumer can answer "what did this request lose becoming Chat Completions?" from data: the
//! record rides [`crate::request::AdaptedRequest::losses`], each one is handed to
//! [`crate::hooks::IngressHooks::on_loss`], and the ingress stage line carries the per-kind count.
//!
//! `kind` is a closed enum whose label is a `&'static str`, so it is safe as a metric label.
//! `field` is a JSON path within the request (bounded, client-controlled key names only) and
//! `detail` is a bounded human sentence; neither carries message content.

use std::fmt;

/// Loss categories. Adding a variant is additive for consumers that count by kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LossKind {
    /// A server-tool-shaped `tools[]` entry was dropped: nothing on this endpoint executes it.
    ServerToolDropped,
    /// A `tool_choice` demanding a tool that was dropped or is unsupported degraded to `auto`.
    ToolChoiceDegraded,
    /// A request field with no Chat Completions meaning was dropped.
    RequestFieldDropped,
    /// A `baseten` extension member this crate does not model was dropped.
    ExtensionFieldDropped,
    /// A message content block with no Chat Completions translation was skipped.
    ContentBlockSkipped,
    /// An oversized unknown `tool_result` block was replaced by a placeholder.
    ToolResultBlockOmitted,
    /// A replayed Responses input item with no translation was skipped.
    InputItemSkipped,
    /// A replayed Responses `compaction` item was ignored.
    ReplayItemIgnored,
    /// A replayed `agent_message` had no plaintext content and was dropped.
    AgentMessageDropped,
    /// A Responses `include` this stack cannot produce was ignored.
    IncludeIgnored,
    /// A duplicate tool declaration was dropped (first declaration kept).
    DuplicateToolDropped,
}

impl LossKind {
    /// Every kind, for consumers that pre-register one metric series per label.
    pub const ALL: [LossKind; 11] = [
        Self::ServerToolDropped,
        Self::ToolChoiceDegraded,
        Self::RequestFieldDropped,
        Self::ExtensionFieldDropped,
        Self::ContentBlockSkipped,
        Self::ToolResultBlockOmitted,
        Self::InputItemSkipped,
        Self::ReplayItemIgnored,
        Self::AgentMessageDropped,
        Self::IncludeIgnored,
        Self::DuplicateToolDropped,
    ];

    /// snake_case label, `&'static` so it can be a metric label.
    pub fn as_label(self) -> &'static str {
        match self {
            Self::ServerToolDropped => "server_tool_dropped",
            Self::ToolChoiceDegraded => "tool_choice_degraded",
            Self::RequestFieldDropped => "request_field_dropped",
            Self::ExtensionFieldDropped => "extension_field_dropped",
            Self::ContentBlockSkipped => "content_block_skipped",
            Self::ToolResultBlockOmitted => "tool_result_block_omitted",
            Self::InputItemSkipped => "input_item_skipped",
            Self::ReplayItemIgnored => "replay_item_ignored",
            Self::AgentMessageDropped => "agent_message_dropped",
            Self::IncludeIgnored => "include_ignored",
            Self::DuplicateToolDropped => "duplicate_tool_dropped",
        }
    }
}

impl fmt::Display for LossKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_label())
    }
}

/// One thing adaptation did not carry through.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Loss {
    pub kind: LossKind,
    /// JSON path of the affected member within the client's request (`tools[2]`, `cache_control`,
    /// `messages[3].content[1]`, `input[4]`). Bounded; key names only.
    pub field: String,
    /// What happened to it, in one bounded sentence. Never message content.
    pub detail: String,
}

/// Ceiling on `field`/`detail`: both may embed client-supplied key names.
const LOSS_TEXT_LIMIT: usize = 160;

/// The losses recorded while adapting one request.
#[derive(Debug, Default)]
pub struct Losses {
    entries: Vec<Loss>,
}

impl Losses {
    /// Record one loss. Also logs it at WARN under `event_name = "ingress.loss"`, so a plain
    /// deployment with no counter still has a trace.
    pub(crate) fn record(
        &mut self,
        kind: LossKind,
        field: impl Into<String>,
        detail: impl Into<String>,
    ) {
        let field = crate::util::truncate(&field.into(), LOSS_TEXT_LIMIT);
        let detail = crate::util::truncate(&detail.into(), LOSS_TEXT_LIMIT);
        tracing::warn!(
            event_name = "ingress.loss",
            kind = kind.as_label(),
            field = %field,
            "{detail}"
        );
        self.entries.push(Loss {
            kind,
            field,
            detail,
        });
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Loss> {
        self.entries.iter()
    }

    pub fn into_vec(self) -> Vec<Loss> {
        self.entries
    }

    /// `kind=n` pairs in [`LossKind::ALL`] order, kinds with no occurrences omitted; the shape the
    /// stage line carries.
    pub fn summary(&self) -> String {
        LossKind::ALL
            .iter()
            .filter_map(|kind| {
                let n = self
                    .entries
                    .iter()
                    .filter(|loss| loss.kind == *kind)
                    .count();
                (n > 0).then(|| format!("{}={n}", kind.as_label()))
            })
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Per-kind occurrence counts over a recorded list, for consumers that hold the `Vec<Loss>`.
pub fn count_by_kind(losses: &[Loss]) -> Vec<(LossKind, usize)> {
    LossKind::ALL
        .iter()
        .filter_map(|kind| {
            let n = losses.iter().filter(|loss| loss.kind == *kind).count();
            (n > 0).then_some((*kind, n))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_lists_kinds_in_declaration_order_and_skips_zeroes() {
        let mut losses = Losses::default();
        losses.record(LossKind::RequestFieldDropped, "cache_control", "dropped");
        losses.record(LossKind::ServerToolDropped, "tools[0]", "dropped");
        losses.record(LossKind::RequestFieldDropped, "service_tier", "dropped");
        assert_eq!(
            losses.summary(),
            "server_tool_dropped=1,request_field_dropped=2"
        );
        assert_eq!(
            count_by_kind(&losses.into_vec()),
            vec![
                (LossKind::ServerToolDropped, 1),
                (LossKind::RequestFieldDropped, 2)
            ]
        );
    }

    #[test]
    fn labels_are_unique_snake_case_and_cover_every_kind() {
        let labels: std::collections::HashSet<_> =
            LossKind::ALL.iter().map(|kind| kind.as_label()).collect();
        assert_eq!(labels.len(), LossKind::ALL.len());
        for label in labels {
            assert!(
                label.chars().all(|c| c.is_ascii_lowercase() || c == '_'),
                "{label}"
            );
        }
    }

    #[test]
    fn record_bounds_client_supplied_text() {
        let mut losses = Losses::default();
        losses.record(
            LossKind::RequestFieldDropped,
            "k".repeat(500),
            "d".repeat(500),
        );
        let loss = &losses.into_vec()[0];
        assert!(loss.field.starts_with(&"k".repeat(LOSS_TEXT_LIMIT)));
        assert!(loss.field.contains("truncated"), "{}", loss.field);
        assert!(loss.detail.contains("truncated"), "{}", loss.detail);
    }
}
