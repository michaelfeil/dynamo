//! Cross-module helpers with no single owner. Copied from tool-bank `src/util.rs` (Apache-2.0),
//! trimmed to what the translation layer calls.

/// Seconds since the Unix epoch, for the `created` field of authored responses.
pub(crate) fn unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub(crate) fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let head: String = s.chars().take(max_chars).collect();
    format!("{head}…[truncated, {} total chars]", s.chars().count())
}
