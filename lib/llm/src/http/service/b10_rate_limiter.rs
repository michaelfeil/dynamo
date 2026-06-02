use axum::http::HeaderMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const STALE_LIMIT_S: u64 = 30;
const INF_LIMIT: f64 = 1e12_f64;
const HEADER_NAME_RATE_LIMIT_TOKEN: &str = "X-Baseten-Token-Rate-Limit-Percentage";
const HEADER_NAME_RATE_LIMIT_REQUEST: &str = "X-Baseten-Request-Rate-Limit-Percentage";

static RATE_LIMIT_LEVEL: AtomicF64 = AtomicF64::new(INF_LIMIT);
static LAST_UPDATED_EPOCH_S: AtomicU64 = AtomicU64::new(0);

pub struct AtomicF64 {
    storage: AtomicU64,
}

impl AtomicF64 {
    pub const fn new(value: f64) -> Self {
        Self {
            storage: AtomicU64::new(value.to_bits()),
        }
    }

    pub fn store(&self, value: f64, ordering: Ordering) {
        self.storage.store(value.to_bits(), ordering)
    }

    pub fn load(&self, ordering: Ordering) -> f64 {
        f64::from_bits(self.storage.load(ordering))
    }
}

fn epoch_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn get_current_rate_limit_level() -> f64 {
    let last_updated = LAST_UPDATED_EPOCH_S.load(Ordering::Relaxed);
    let loaded_limit = RATE_LIMIT_LEVEL.load(Ordering::Relaxed);
    if last_updated == 0 || epoch_seconds().saturating_sub(last_updated) > STALE_LIMIT_S {
        if loaded_limit < INF_LIMIT {
            tracing::warn!(
                "Rate limit level is stale (last updated {} seconds ago), returning default limit",
                epoch_seconds().saturating_sub(last_updated)
            );
        }
        INF_LIMIT
    } else {
        loaded_limit
    }
}

pub fn set_current_rate_limit_level(new_limit: f64) {
    let limit = new_limit.clamp(0.0, INF_LIMIT);
    if limit != new_limit {
        tracing::error!(
            "Rate limit level {} is out of bounds, setting to {}",
            new_limit,
            limit
        );
    }
    RATE_LIMIT_LEVEL.store(limit, Ordering::Relaxed);
    LAST_UPDATED_EPOCH_S.store(epoch_seconds(), Ordering::Relaxed);
}

pub fn check_rate_limit(headers: &HeaderMap) -> Option<(String, f64)> {
    let request_limit: f64 = headers
        .get(HEADER_NAME_RATE_LIMIT_REQUEST)
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let token_limit: f64 = headers
        .get(HEADER_NAME_RATE_LIMIT_TOKEN)
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let request_level = request_limit.max(token_limit);

    let current_level = get_current_rate_limit_level();
    if request_level > current_level {
        Some((
            format!(
                "Rate limit exceeded: usage fraction {} percent exceeds current level {} percent",
                request_level * 100.0,
                current_level * 100.0
            ),
            request_level,
        ))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn header_with_usage(value: f64) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            HEADER_NAME_RATE_LIMIT_REQUEST,
            value.to_string().parse().unwrap(),
        );
        headers
    }

    #[test]
    #[serial]
    fn test_rate_limit_blocks_when_over_limit() {
        set_current_rate_limit_level(0.0);
        let headers = header_with_usage(0.1);
        assert!(check_rate_limit(&headers).is_some());
    }

    #[test]
    #[serial]
    fn test_rate_limit_allows_when_under_limit() {
        set_current_rate_limit_level(1.5);
        let headers = header_with_usage(1.0);
        assert!(check_rate_limit(&headers).is_none());
    }

    #[test]
    #[serial]
    fn test_rate_limit_stale_allows_requests() {
        set_current_rate_limit_level(0.0);
        let stale_epoch = epoch_seconds().saturating_sub(STALE_LIMIT_S + 1);
        LAST_UPDATED_EPOCH_S.store(stale_epoch, Ordering::Relaxed);
        let headers = header_with_usage(100.0);
        assert!(check_rate_limit(&headers).is_none());
    }
}
