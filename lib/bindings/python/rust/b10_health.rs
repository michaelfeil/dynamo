use dynamo_llm::b10_health;
use pyo3::prelude::*;
use std::time::Duration;

use crate::DistributedRuntime;

const MAX_HEALTH_TIMEOUT_SECS: f64 = 600.0;

#[pyfunction]
#[pyo3(signature = (healthy, reason = "", timeout_secs = None))]
pub fn set_health(healthy: bool, reason: &str, timeout_secs: Option<f64>) -> PyResult<()> {
    let timeout = match timeout_secs {
        Some(secs) if secs.is_finite() && (0.0..=MAX_HEALTH_TIMEOUT_SECS).contains(&secs) => {
            Some(Duration::from_secs_f64(secs))
        }
        Some(_) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "timeout_secs must be a finite float between 0 and 600",
            ));
        }
        None => None,
    };
    b10_health::set_health(healthy, reason, timeout);
    Ok(())
}

#[pyfunction]
#[pyo3(text_signature = "(runtime)")]
pub fn register_runtime(runtime: &DistributedRuntime) -> PyResult<()> {
    b10_health::register_runtime_cancel_token(runtime.inner().primary_token());
    Ok(())
}

#[pyfunction]
#[pyo3(text_signature = "()")]
pub fn set_poisoned() -> PyResult<()> {
    b10_health::set_poisoned();
    tracing::warn!("System marked as poisoned via Python binding");
    Ok(())
}

#[pyfunction]
#[pyo3(text_signature = "()")]
pub fn is_healthy() -> PyResult<bool> {
    Ok(b10_health::is_healthy())
}
