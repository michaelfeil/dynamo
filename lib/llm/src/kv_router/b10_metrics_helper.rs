//! Baseten-specific metric labeling helpers.

use std::collections::HashMap;

/// Baseten deployment-identity const labels, matching the labels the Python
/// serving metrics attach in-process. Empty/unset env vars are omitted.
///
/// Any dynamo-exported metric that should be usable in-product (Baseten
/// dashboards/charts) MUST carry these labels — product queries filter on
/// exported_namespace/model_version_id, and the central remote-write keep-list
/// assumes them. Apply this helper to new metrics before exposing them.
pub fn bis_const_labels() -> HashMap<String, String> {
    let mut labels = HashMap::new();
    for (env_var, label) in [
        ("BIS_EXPORTED_NAMESPACE", "exported_namespace"),
        ("BIS_MODEL_VERSION_ID", "model_version_id"),
    ] {
        if let Ok(value) = std::env::var(env_var)
            && !value.is_empty()
        {
            labels.insert(label.to_string(), value);
        }
    }
    labels
}
