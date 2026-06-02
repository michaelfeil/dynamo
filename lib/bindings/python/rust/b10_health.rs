use dynamo_llm::b10_health;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (healthy, reason = ""))]
pub fn set_health(healthy: bool, reason: &str) -> PyResult<()> {
    b10_health::set_health(healthy, reason);
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
