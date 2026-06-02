use super::RouteDoc;
use crate::b10_health::is_healthy;
use axum::{Router, http::StatusCode, response::IntoResponse, routing::get};

async fn health_file_check() -> impl IntoResponse {
    if !is_healthy() {
        tracing::warn!("Health check failed: system marked unhealthy");
        return StatusCode::SERVICE_UNAVAILABLE;
    }
    tracing::info!(
        skip_unified_model_logs = true,
        "Health check: system marked healthy"
    );
    StatusCode::OK
}

pub fn add_health_file_router(_state: Option<()>) -> (Vec<RouteDoc>, Router) {
    let path = "/health_file".to_string();
    let doc = RouteDoc::new(axum::http::Method::GET, &path);
    let router = Router::new().route(&path, get(health_file_check));
    (vec![doc], router)
}
