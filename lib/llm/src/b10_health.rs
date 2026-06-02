use std::sync::Mutex;
use std::time::{Duration, Instant};

const HEALTH_TIMEOUT: Duration = Duration::from_secs(60);

static LAST_HEALTH_UPDATE: Mutex<Option<(Instant, bool)>> = Mutex::new(None);
static UNRECOVERABLE_DOWN: Mutex<bool> = Mutex::new(false);

pub fn set_health(healthy: bool, reason: &str) {
    let mut guard = LAST_HEALTH_UPDATE.lock().unwrap();
    let previous_state = guard.map(|(_, was_healthy)| was_healthy);
    let state_changed = previous_state != Some(healthy);

    if state_changed {
        if healthy {
            tracing::info!("Health state changed to HEALTHY: {}", reason);
        } else {
            tracing::warn!("Health state changed to UNHEALTHY: {}", reason);
        }
    }

    *guard = Some((Instant::now(), healthy));
}

pub fn set_poisoned() {
    let mut unrecoverable_guard = UNRECOVERABLE_DOWN.lock().unwrap();
    if !(*unrecoverable_guard) {
        tracing::warn!("Health state changed to POISONED");
    }
    *unrecoverable_guard = true;
}

pub fn is_healthy() -> bool {
    let unrecoverable_guard = UNRECOVERABLE_DOWN.lock().unwrap();
    if *unrecoverable_guard {
        return false;
    }
    drop(unrecoverable_guard);

    let guard = LAST_HEALTH_UPDATE.lock().unwrap();
    match *guard {
        Some((last_update, was_healthy)) => was_healthy && last_update.elapsed() <= HEALTH_TIMEOUT,
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::thread;

    fn cleanup_health_state() {
        let mut last_update_guard = LAST_HEALTH_UPDATE.lock().unwrap();
        *last_update_guard = None;
        drop(last_update_guard);

        let mut unrecoverable_guard = UNRECOVERABLE_DOWN.lock().unwrap();
        *unrecoverable_guard = false;
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
        set_health(true, "test: setting healthy");
        assert!(is_healthy());
        set_health(false, "test: setting unhealthy");
        assert!(!is_healthy());
    }

    #[test]
    #[serial]
    fn test_health_timeout() {
        cleanup_health_state();
        set_health(true, "test: initial healthy state");
        assert!(is_healthy());

        {
            let mut last_update_guard = LAST_HEALTH_UPDATE.lock().unwrap();
            *last_update_guard = Some((Instant::now() - Duration::from_secs(65), true));
        }

        assert!(!is_healthy());
    }

    #[test]
    #[serial]
    fn test_health_refresh() {
        cleanup_health_state();
        set_health(true, "test: initial healthy state");
        assert!(is_healthy());

        thread::sleep(Duration::from_millis(100));
        set_health(true, "test: refresh healthy state");
        assert!(is_healthy());
    }

    #[test]
    #[serial]
    fn test_set_poisoned_makes_unhealthy() {
        cleanup_health_state();
        set_health(true, "test: initial healthy state");
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

        set_health(true, "test: attempt recovery after poison");
        assert!(!is_healthy());
    }
}
