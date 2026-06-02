use dynamo_llm::http::service::b10_rate_limiter;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (level))]
pub fn set_rate_limit_level(level: f64) -> PyResult<()> {
    b10_rate_limiter::set_current_rate_limit_level(level);
    Ok(())
}
