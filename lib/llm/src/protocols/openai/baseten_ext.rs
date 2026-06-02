use std::collections::HashMap;

use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validator::Validate;

#[derive(ToSchema, Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct Thinking {
    #[serde(rename = "type")]
    pub thinking_type: ThinkingType,
}

#[derive(ToSchema, Clone, Serialize, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingType {
    Enabled,
    Disabled,
}

#[derive(ToSchema, Clone, Copy, Serialize, Debug, Deserialize, PartialEq)]
#[serde(try_from = "f32", into = "f32")]
pub struct DynamicTemperature(f32);

impl DynamicTemperature {
    pub fn get(self) -> f32 {
        self.0
    }
}

impl TryFrom<f32> for DynamicTemperature {
    type Error = String;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        if (0.0..=2.0).contains(&value) {
            Ok(Self(value))
        } else {
            Err("dynamic_temperature values must be between 0.0 and 2.0".into())
        }
    }
}

impl From<DynamicTemperature> for f32 {
    fn from(value: DynamicTemperature) -> Self {
        value.0
    }
}

#[derive(ToSchema, Clone, Serialize, Debug, Deserialize, PartialEq, Eq, Hash)]
#[serde(try_from = "String", into = "String")]
pub struct DynamicTemperatureSuffix(String);

impl DynamicTemperatureSuffix {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for DynamicTemperatureSuffix {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.is_empty() {
            Err("dynamic_temperature keys must be non-empty".into())
        } else {
            Ok(Self(value))
        }
    }
}

impl TryFrom<&str> for DynamicTemperatureSuffix {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::try_from(value.to_string())
    }
}

impl From<DynamicTemperatureSuffix> for String {
    fn from(value: DynamicTemperatureSuffix) -> Self {
        value.0
    }
}

pub type DynamicTemperatureMap = HashMap<DynamicTemperatureSuffix, DynamicTemperature>;

#[derive(ToSchema, Serialize, Deserialize, Builder, Validate, Debug, Clone, Default)]
pub struct BasetenExt {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub b10_cache_control: Option<serde_json::Value>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub baseten: Option<serde_json::Value>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub dynamic_temperature: Option<DynamicTemperatureMap>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub thinking: Option<Thinking>,
}

impl BasetenExt {
    pub fn builder() -> BasetenExtBuilder {
        BasetenExtBuilder::default()
    }

    pub fn is_empty(&self) -> bool {
        self.b10_cache_control.is_none()
            && self.baseten.is_none()
            && self.dynamic_temperature.is_none()
            && self.thinking.is_none()
    }

    pub fn validate_request(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

pub fn validate_request_fields(baseten_ext: &BasetenExt) -> anyhow::Result<()> {
    baseten_ext.validate_request()
}

pub trait BasetenExtProvider {
    fn baseten_ext(&self) -> Option<&BasetenExt>;

    fn get_b10_cache_control(&self) -> Option<serde_json::Value> {
        self.baseten_ext()
            .and_then(|ext| ext.b10_cache_control.clone())
    }

    fn get_baseten(&self) -> Option<serde_json::Value> {
        self.baseten_ext().and_then(|ext| ext.baseten.clone())
    }

    fn get_dynamic_temperature(&self) -> Option<&DynamicTemperatureMap> {
        self.baseten_ext()
            .and_then(|ext| ext.dynamic_temperature.as_ref())
    }

    fn get_thinking(&self) -> Option<&Thinking> {
        self.baseten_ext().and_then(|ext| ext.thinking.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn suffix(value: &str) -> DynamicTemperatureSuffix {
        value.try_into().unwrap()
    }

    fn temperature(value: f32) -> DynamicTemperature {
        value.try_into().unwrap()
    }

    #[test]
    fn test_baseten_ext_builder_default() {
        let baseten_ext = BasetenExt::builder().build().unwrap();
        assert_eq!(baseten_ext.b10_cache_control, None);
        assert_eq!(baseten_ext.baseten, None);
        assert_eq!(baseten_ext.dynamic_temperature, None);
        assert_eq!(baseten_ext.thinking, None);
    }

    #[test]
    fn test_baseten_ext_builder_with_values() {
        let baseten_ext = BasetenExt::builder()
            .b10_cache_control(serde_json::json!("no-cache"))
            .baseten(serde_json::json!({"key": "value"}))
            .dynamic_temperature(HashMap::from([(suffix("</think>"), temperature(0.7))]))
            .build()
            .unwrap();

        assert_eq!(
            baseten_ext.b10_cache_control,
            Some(serde_json::json!("no-cache"))
        );
        assert_eq!(
            baseten_ext.baseten,
            Some(serde_json::json!({"key": "value"}))
        );
        assert_eq!(
            baseten_ext.dynamic_temperature,
            Some(HashMap::from([(suffix("</think>"), temperature(0.7))]))
        );
    }

    #[test]
    fn test_dynamic_temperature_deserialization_rejects_invalid_value() {
        let json = r#"{"dynamic_temperature":{"</think>":2.1}}"#;
        let err = serde_json::from_str::<BasetenExt>(json).unwrap_err();
        assert!(
            err.to_string()
                .contains("dynamic_temperature values must be between 0.0 and 2.0")
        );
    }

    #[test]
    fn test_dynamic_temperature_deserialization_rejects_empty_suffix() {
        let json = r#"{"dynamic_temperature":{"":0.7}}"#;
        let err = serde_json::from_str::<BasetenExt>(json).unwrap_err();
        assert!(
            err.to_string()
                .contains("dynamic_temperature keys must be non-empty")
        );
    }
}
