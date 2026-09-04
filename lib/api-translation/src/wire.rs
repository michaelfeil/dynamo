//! Minimal SSE/JSON wire helpers the framings share. Copied from tool-bank `src/wire.rs`
//! (Apache-2.0), minus the inbound SSE byte decoder and HTTP-client machinery, which stay with the
//! service that owns the sockets.

/// One SSE frame. `event` names the event (Messages, Responses); `None` emits a bare
/// `data:` frame (ChatCompletions).
pub(crate) fn sse_frame(event: Option<&str>, data: &str) -> String {
    let mut frame = String::new();
    if let Some(name) = event {
        frame.push_str("event: ");
        frame.push_str(name);
        frame.push('\n');
    }
    // A raw newline in `data` would start a new SSE field; per spec each line is its own `data:`.
    for line in data.split('\n') {
        frame.push_str("data: ");
        frame.push_str(line);
        frame.push('\n');
    }
    frame.push('\n');
    frame
}

/// Legal raw in a JSON string, but a client that decodes SSE to text before splitting lines
/// (httpx's `iter_lines`, `requests` with `decode_unicode`) tears the frame on these mid-JSON.
/// Tool results carry scraped web text, where they do occur.
const LINE_BREAKING_IN_TEXT: [char; 3] = ['\u{85}', '\u{2028}', '\u{2029}'];

/// Encode a wire type as a JSON value; same cannot-fail contract as [`to_json_string`].
pub(crate) fn to_json_value<T: serde::Serialize>(value: &T) -> serde_json::Value {
    serde_json::to_value(value).expect("wire type serializes")
}

/// Encode a wire type. Every one is a derived `Serialize` over owned or borrowed data, so it cannot
/// fail. Escaping happens here because this is the only encoder both edges use.
pub(crate) fn to_json_string<T: serde::Serialize>(value: &T) -> String {
    let json = serde_json::to_string(value).expect("wire type serializes");
    if !json.contains(LINE_BREAKING_IN_TEXT) {
        return json;
    }
    // One buffer, not one allocation per character: a tool result is tens of KB and this runs per frame.
    let mut escaped = String::with_capacity(json.len() + 16);
    for character in json.chars() {
        match character {
            '\u{85}' => escaped.push_str("\\u0085"),
            '\u{2028}' => escaped.push_str("\\u2028"),
            '\u{2029}' => escaped.push_str("\\u2029"),
            other => escaped.push(other),
        }
    }
    escaped
}

/// Process-unique suffix for a synthetic response id (monotonic; resets per process). Each protocol
/// builds its own id format around this in [`crate::framing`].
pub(crate) fn next_id_seq() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    SEQ.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cc_frame_is_bare_data() {
        assert_eq!(sse_frame(None, "{\"x\":1}"), "data: {\"x\":1}\n\n");
    }

    #[test]
    fn messages_frame_names_event() {
        assert_eq!(
            sse_frame(Some("message_stop"), "{}"),
            "event: message_stop\ndata: {}\n\n"
        );
    }

    /// A client that decodes to text before splitting lines breaks on these; escaping keeps the JSON
    /// semantically identical and the frame on one line.
    #[test]
    fn text_line_breaks_are_escaped_not_emitted_raw() {
        let framed = to_json_string(&serde_json::json!({"content": "a\u{2028}b\u{85}c\u{2029}d"}));
        assert_eq!(framed, r#"{"content":"a\u2028b\u0085c\u2029d"}"#);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&framed).unwrap()["content"],
            "a\u{2028}b\u{85}c\u{2029}d",
            "escaping must not change the decoded value"
        );
    }

    #[test]
    fn multiline_data_prefixes_each_line() {
        assert_eq!(sse_frame(None, "a\nb"), "data: a\ndata: b\n\n");
    }
}
