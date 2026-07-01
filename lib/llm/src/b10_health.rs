use std::sync::{LazyLock, RwLock};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

const HEALTH_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Default)]
struct HealthState {
    last_health_update: Option<(Instant, bool, Duration)>,
    unrecoverable_down: bool,
    runtime_cancel_token: Option<CancellationToken>,
}

static HEALTH_STATE: LazyLock<RwLock<HealthState>> =
    LazyLock::new(|| RwLock::new(HealthState::default()));

/// Set the readiness state with an optional per-update timeout.
///
/// When `healthy` is true, the process remains healthy until `timeout` elapses
/// unless another call refreshes the state. Each healthy update replaces the
/// previous lease; it does not take the max of the old and new timeouts. `None`
/// preserves the default timeout.
pub fn set_health(healthy: bool, reason: &str, timeout: Option<Duration>) {
    let mut state = HEALTH_STATE.write().unwrap();
    let previous_state = state
        .last_health_update
        .as_ref()
        .map(|(_, was_healthy, _)| *was_healthy);
    let state_changed = previous_state != Some(healthy);

    if state_changed {
        if healthy {
            tracing::info!("Health state changed to HEALTHY: {}", reason);
        } else {
            tracing::warn!(
                unified_logs = true,
                "Health state changed to UNHEALTHY: {}",
                reason
            );
        }
    }

    state.last_health_update = Some((Instant::now(), healthy, timeout.unwrap_or(HEALTH_TIMEOUT)));
}

pub fn set_poisoned() {
    let mut state = HEALTH_STATE.write().unwrap();
    if !state.unrecoverable_down {
        tracing::warn!("Health state changed to POISONED");
    }
    state.unrecoverable_down = true;
}

pub fn register_runtime_cancel_token(token: CancellationToken) {
    HEALTH_STATE.write().unwrap().runtime_cancel_token = Some(token);
}

pub fn is_healthy() -> bool {
    let state = HEALTH_STATE.read().unwrap();
    if state.unrecoverable_down {
        return false;
    }

    let runtime_cancel_token = state.runtime_cancel_token.clone();
    let last_health_update = state.last_health_update;
    drop(state);

    if runtime_cancel_token.is_some_and(|token| token.is_cancelled()) {
        return false;
    }

    match last_health_update {
        Some((last_update, was_healthy, timeout)) => {
            was_healthy && last_update.elapsed() <= timeout
        }
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::thread;

    fn cleanup_health_state() {
        *HEALTH_STATE.write().unwrap() = HealthState::default();
    }

    #[test]
    #[serial]
    fn test_initial_state_unhealthy() {
        cleanup_health_state();
        assert!(!is_healthy());
    }

    #[test]
    #[serial]
    fn test_health_operations() {
        cleanup_health_state();
        set_health(true, "test: setting healthy", None);
        assert!(is_healthy());
        set_health(false, "test: setting unhealthy", None);
        assert!(!is_healthy());
    }

    #[test]
    #[serial]
    fn test_health_timeout() {
        cleanup_health_state();
        set_health(true, "test: initial healthy state", None);
        assert!(is_healthy());

        {
            let mut state = HEALTH_STATE.write().unwrap();
            state.last_health_update = Some((
                Instant::now() - Duration::from_secs(65),
                true,
                HEALTH_TIMEOUT,
            ));
        }

        assert!(!is_healthy());
    }

    #[test]
    #[serial]
    fn test_health_refresh() {
        cleanup_health_state();
        set_health(true, "test: initial healthy state", None);
        assert!(is_healthy());

        thread::sleep(Duration::from_millis(100));
        set_health(true, "test: refresh healthy state", None);
        assert!(is_healthy());
    }

    #[test]
    #[serial]
    fn test_set_poisoned_makes_unhealthy() {
        cleanup_health_state();
        set_health(true, "test: initial healthy state", None);
        assert!(is_healthy());

        set_poisoned();
        assert!(!is_healthy());
    }

    #[test]
    #[serial]
    fn test_poisoned_state_cannot_be_recovered() {
        cleanup_health_state();
        set_poisoned();
        assert!(!is_healthy());

        set_health(true, "test: attempt recovery after poison", None);
        assert!(!is_healthy());
    }

    #[test]
    #[serial]
    fn test_registered_runtime_cancel_token_makes_health_unhealthy() {
        cleanup_health_state();
        let token = CancellationToken::new();
        register_runtime_cancel_token(token.clone());

        set_health(true, "test: initial healthy state", None);
        assert!(is_healthy());

        token.cancel();
        assert!(!is_healthy());
    }

    #[test]
    #[serial]
    fn test_registered_runtime_cancel_token_can_be_replaced() {
        cleanup_health_state();
        let old_token = CancellationToken::new();
        register_runtime_cancel_token(old_token.clone());
        set_health(true, "test: initial healthy state", None);
        assert!(is_healthy());

        old_token.cancel();
        assert!(!is_healthy());

        let new_token = CancellationToken::new();
        register_runtime_cancel_token(new_token.clone());
        assert!(is_healthy());
    }
}
