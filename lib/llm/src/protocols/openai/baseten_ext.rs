// Baseten-specific extensions for OpenAI API requests, including cache-control and dynamic temperature overrides.
// Codex/claude: Please merge file - if possible, as is, in future forks e.g. dyn1.3/1.4 etc.
// Do not depend on other dynamo crates to avoid circular dependencies.

use std::collections::HashMap;

use derive_builder::Builder;
use serde::{Deserialize, Deserializer, Serialize};
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

/// Cache-control priority tier for Baseten-managed KV behavior.
#[derive(
    ToSchema, Clone, Copy, Serialize, Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord, Default,
)]
#[serde(rename_all = "lowercase")]
pub enum CacheControlTier {
    Basic,
    #[default]
    Standard,
    Premium,
}

/// Decode cache-control TTL values supported by Baseten.
#[derive(ToSchema, Clone, Copy, Serialize, Debug, Deserialize, PartialEq, Eq, Default)]
pub enum CacheControlTtl {
    #[serde(rename = "5m")]
    #[default]
    FiveMinutes,
    #[serde(rename = "1m")]
    OneMinute,
}

/// Half-open token range `[start, end)` with an optional open-ended tail.
#[derive(ToSchema, Clone, Serialize, Debug, Deserialize, PartialEq, Eq)]
pub struct CacheControlRange {
    pub start: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end: Option<u64>,
    #[serde(default)]
    pub tier: CacheControlTier,
}

/// Decode-phase cache-control policy.
#[derive(ToSchema, Clone, Serialize, Debug, Deserialize, PartialEq, Eq)]
pub struct DecodeCacheControl {
    #[serde(default)]
    pub tier: CacheControlTier,
    #[serde(default)]
    pub decode_ttl: CacheControlTtl,
}

const CACHE_CONTROL_EXAMPLE: &str = r#"{"cache_control":[{"start":0,"end":1024,"tier":"standard"},{"start":1024,"end":null,"tier":"basic"}]}"#;

fn merge_adjacent_cache_control_ranges(ranges: Vec<CacheControlRange>) -> Vec<CacheControlRange> {
    ranges
        .into_iter()
        .fold(Vec::<CacheControlRange>::new(), |mut merged, range| {
            if let Some(prev) = merged.last_mut()
                && prev.tier == range.tier
                && prev.end == Some(range.start)
            {
                prev.end = range.end;
                return merged;
            }

            merged.push(range);
            merged
        })
}

fn deserialize_cache_control_ranges<'de, D>(
    deserializer: D,
) -> Result<Option<Vec<CacheControlRange>>, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Option::<Vec<CacheControlRange>>::deserialize(deserializer)?
        .map(merge_adjacent_cache_control_ranges))
}

fn validate_cache_control_ranges(ranges: &[CacheControlRange]) -> anyhow::Result<()> {
    if ranges.is_empty() {
        anyhow::bail!(
            "cache_control must contain at least one range. Minimal valid example: {CACHE_CONTROL_EXAMPLE}"
        );
    }

    if ranges[0].start != 0 {
        anyhow::bail!(
            "cache_control ranges must start at 0. Minimal valid example: {CACHE_CONTROL_EXAMPLE}"
        );
    }

    let mut prev_end = None;
    for (idx, range) in ranges.iter().enumerate() {
        if let Some(end) = range.end
            && end <= range.start
        {
            anyhow::bail!("cache_control range requires end > start");
        }

        let is_last = idx == ranges.len() - 1;
        match (is_last, range.end) {
            (true, Some(_)) => anyhow::bail!(
                "the last cache_control range must be open-ended with end=null. Minimal valid example: {CACHE_CONTROL_EXAMPLE}"
            ),
            (false, None) => anyhow::bail!(
                "only the last cache_control range may be open-ended. Minimal valid example: {CACHE_CONTROL_EXAMPLE}"
            ),
            _ => {}
        }

        if let Some(expected_start) = prev_end
            && range.start != expected_start
        {
            anyhow::bail!(
                "cache_control ranges must be adjacent. Minimal valid example: {CACHE_CONTROL_EXAMPLE}"
            );
        }

        if idx > 0 && range.tier > ranges[idx - 1].tier {
            anyhow::bail!(
                "cache_control tiers must be ordered from premium to basic after merging adjacent ranges with the same tier"
            );
        }

        prev_end = range.end;
    }

    Ok(())
}

#[derive(ToSchema, Serialize, Deserialize, Builder, Validate, Debug, Clone, Default)]
pub struct BasetenExt {
    /// Baseten prompt cache-control ranges.
    #[serde(
        default,
        deserialize_with = "deserialize_cache_control_ranges",
        skip_serializing_if = "Option::is_none"
    )]
    #[builder(default, setter(strip_option))]
    pub cache_control: Option<Vec<CacheControlRange>>,

    /// Baseten decode-phase cache-control policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub decode_cache_control: Option<DecodeCacheControl>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub baseten: Option<serde_json::Value>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub dynamic_temperature: Option<DynamicTemperatureMap>,

    /// Baseten request priority payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub priority: Option<serde_json::Value>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub thinking: Option<Thinking>,

    /// OpenAI-style reasoning block forwarded opaquely to the worker.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub reasoning: Option<serde_json::Value>,
}

impl BasetenExt {
    pub fn builder() -> BasetenExtBuilder {
        BasetenExtBuilder::default()
    }

    pub fn is_empty(&self) -> bool {
        self.cache_control.is_none()
            && self.decode_cache_control.is_none()
            && self.baseten.is_none()
            && self.dynamic_temperature.is_none()
            && self.priority.is_none()
            && self.thinking.is_none()
            && self.reasoning.is_none()
    }

    pub fn validate_request(&self) -> anyhow::Result<()> {
        if let Some(ranges) = self.cache_control.as_deref() {
            validate_cache_control_ranges(ranges)?;
        }

        if let Some(decode_cache_control) = &self.decode_cache_control {
            let last_tier = self
                .cache_control
                .as_deref()
                .and_then(|ranges| ranges.last())
                .map(|range| range.tier)
                .unwrap_or_default();

            if decode_cache_control.tier > last_tier {
                anyhow::bail!(
                    "decode_cache_control tier must be no higher than the last cache_control tier. Minimal valid example: {{\"cache_control\":[{{\"start\":0,\"end\":1024,\"tier\":\"standard\"}},{{\"start\":1024,\"end\":null,\"tier\":\"basic\"}}],\"decode_cache_control\":{{\"tier\":\"basic\"}}}}"
                );
            }
        }

        Ok(())
    }
}

pub fn validate_request_fields(baseten_ext: &BasetenExt) -> anyhow::Result<()> {
    baseten_ext.validate_request()
}

pub trait BasetenExtProvider {
    fn baseten_ext(&self) -> Option<&BasetenExt>;

    fn get_cache_control(&self) -> Option<&[CacheControlRange]> {
        self.baseten_ext()
            .and_then(|ext| ext.cache_control.as_deref())
    }

    fn get_decode_cache_control(&self) -> Option<&DecodeCacheControl> {
        self.baseten_ext()
            .and_then(|ext| ext.decode_cache_control.as_ref())
    }

    fn get_baseten(&self) -> Option<serde_json::Value> {
        self.baseten_ext().and_then(|ext| ext.baseten.clone())
    }

    fn get_dynamic_temperature(&self) -> Option<&DynamicTemperatureMap> {
        self.baseten_ext()
            .and_then(|ext| ext.dynamic_temperature.as_ref())
    }

    fn get_priority(&self) -> Option<&serde_json::Value> {
        self.baseten_ext().and_then(|ext| ext.priority.as_ref())
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
        assert_eq!(baseten_ext.cache_control, None);
        assert_eq!(baseten_ext.decode_cache_control, None);
        assert_eq!(baseten_ext.baseten, None);
        assert_eq!(baseten_ext.dynamic_temperature, None);
        assert_eq!(baseten_ext.priority, None);
        assert_eq!(baseten_ext.thinking, None);
        assert_eq!(baseten_ext.reasoning, None);
    }

    #[test]
    fn test_reasoning_block_round_trips_through_serde() {
        let json = r#"{"reasoning":{"enabled":false,"effort":"medium","max_tokens":1024}}"#;
        let parsed: BasetenExt = serde_json::from_str(json).unwrap();
        assert_eq!(
            parsed.reasoning,
            Some(serde_json::json!({
                "enabled": false,
                "effort": "medium",
                "max_tokens": 1024,
            }))
        );

        let reserialized = serde_json::to_string(&parsed).unwrap();
        let reparsed: BasetenExt = serde_json::from_str(&reserialized).unwrap();
        assert_eq!(reparsed.reasoning, parsed.reasoning);
    }

    #[test]
    fn test_baseten_ext_builder_with_values() {
        let cache_control = vec![
            CacheControlRange {
                start: 0,
                end: Some(1024),
                tier: CacheControlTier::Standard,
            },
            CacheControlRange {
                start: 1024,
                end: None,
                tier: CacheControlTier::Basic,
            },
        ];
        let baseten_ext = BasetenExt::builder()
            .cache_control(cache_control.clone())
            .baseten(serde_json::json!({"key": "value"}))
            .dynamic_temperature(HashMap::from([(suffix("</think>"), temperature(0.7))]))
            .priority(serde_json::json!({"level": "high"}))
            .build()
            .unwrap();

        assert_eq!(baseten_ext.cache_control, Some(cache_control));
        assert_eq!(
            baseten_ext.baseten,
            Some(serde_json::json!({"key": "value"}))
        );
        assert_eq!(
            baseten_ext.dynamic_temperature,
            Some(HashMap::from([(suffix("</think>"), temperature(0.7))]))
        );
        assert_eq!(
            baseten_ext.priority,
            Some(serde_json::json!({"level": "high"}))
        );
    }

    #[test]
    fn test_baseten_ext_serialization() {
        let cache_control = vec![
            CacheControlRange {
                start: 0,
                end: Some(1024),
                tier: CacheControlTier::Standard,
            },
            CacheControlRange {
                start: 1024,
                end: None,
                tier: CacheControlTier::Basic,
            },
        ];
        let baseten_ext = BasetenExt {
            cache_control: Some(cache_control),
            baseten: Some(serde_json::json!({"key": "value"})),
            dynamic_temperature: Some(HashMap::from([(suffix("</think>"), temperature(0.7))])),
            priority: Some(serde_json::json!({"level": "high"})),
            ..Default::default()
        };

        let json = serde_json::to_string(&baseten_ext).unwrap();
        let parsed: BasetenExt = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.cache_control, baseten_ext.cache_control);
        assert_eq!(parsed.baseten, baseten_ext.baseten);
        assert_eq!(parsed.dynamic_temperature, baseten_ext.dynamic_temperature);
        assert_eq!(parsed.priority, baseten_ext.priority);
    }

    #[test]
    fn test_cache_control_validates_adjacent_open_ended_ranges() {
        let baseten_ext: BasetenExt = serde_json::from_str(
            r#"{
                "cache_control": [
                    {"start": 0, "end": 1024, "tier": "standard"},
                    {"start": 1024, "end": null, "tier": "basic"}
                ]
            }"#,
        )
        .unwrap();

        baseten_ext.validate_request().unwrap();
    }

    #[test]
    fn test_cache_control_merges_adjacent_ranges_with_same_tier() {
        let baseten_ext: BasetenExt = serde_json::from_str(
            r#"{
                "cache_control": [
                    {"start": 0, "end": 512, "tier": "premium"},
                    {"start": 512, "end": 1024, "tier": "premium"},
                    {"start": 1024, "end": 2048, "tier": "standard"},
                    {"start": 2048, "end": null, "tier": "standard"}
                ]
            }"#,
        )
        .unwrap();

        assert_eq!(
            baseten_ext.cache_control,
            Some(vec![
                CacheControlRange {
                    start: 0,
                    end: Some(1024),
                    tier: CacheControlTier::Premium,
                },
                CacheControlRange {
                    start: 1024,
                    end: None,
                    tier: CacheControlTier::Standard,
                },
            ])
        );
        baseten_ext.validate_request().unwrap();
    }

    #[test]
    fn test_cache_control_rejects_non_list() {
        let err = serde_json::from_str::<BasetenExt>(
            r#"{"cache_control": {"start": 0, "end": null, "tier": "standard"}}"#,
        )
        .unwrap_err();

        assert!(err.to_string().contains("invalid type"));
    }

    #[test]
    fn test_cache_control_rejects_non_object_range() {
        let err = serde_json::from_str::<BasetenExt>(r#"{"cache_control": [0]}"#).unwrap_err();

        assert!(err.to_string().contains("invalid type"));
    }

    #[test]
    fn test_cache_control_rejects_non_null_final_end() {
        let baseten_ext: BasetenExt = serde_json::from_str(
            r#"{
                "cache_control": [
                    {"start": 0, "end": 1024, "tier": "standard"}
                ]
            }"#,
        )
        .unwrap();

        let err = baseten_ext.validate_request().unwrap_err();
        assert!(
            err.to_string()
                .contains("last cache_control range must be open-ended")
        );
    }

    #[test]
    fn test_cache_control_rejects_non_adjacent_ranges() {
        let baseten_ext: BasetenExt = serde_json::from_str(
            r#"{
                "cache_control": [
                    {"start": 0, "end": 1024, "tier": "standard"},
                    {"start": 2048, "end": null, "tier": "basic"}
                ]
            }"#,
        )
        .unwrap();

        let err = baseten_ext.validate_request().unwrap_err();
        assert!(err.to_string().contains("ranges must be adjacent"));
    }

    #[test]
    fn test_cache_control_rejects_tiers_that_increase_after_merge() {
        let baseten_ext: BasetenExt = serde_json::from_str(
            r#"{
                "cache_control": [
                    {"start": 0, "end": 1, "tier": "standard"},
                    {"start": 1, "end": 2, "tier": "premium"},
                    {"start": 2, "end": null, "tier": "standard"}
                ]
            }"#,
        )
        .unwrap();

        let err = baseten_ext.validate_request().unwrap_err();
        assert!(err.to_string().contains("premium to basic"));
    }

    #[test]
    fn test_decode_cache_control_allows_same_or_lower_tier_than_last_cache_control_range() {
        let baseten_ext: BasetenExt = serde_json::from_str(
            r#"{
                "cache_control": [
                    {"start": 0, "end": null, "tier": "standard"}
                ],
                "decode_cache_control": {"tier": "basic"}
            }"#,
        )
        .unwrap();

        baseten_ext.validate_request().unwrap();
    }

    #[test]
    fn test_decode_cache_control_rejects_higher_tier_than_last_cache_control_range() {
        let baseten_ext: BasetenExt = serde_json::from_str(
            r#"{
                "cache_control": [
                    {"start": 0, "end": null, "tier": "basic"}
                ],
                "decode_cache_control": {"tier": "standard"}
            }"#,
        )
        .unwrap();

        let err = baseten_ext.validate_request().unwrap_err();
        assert!(
            err.to_string()
                .contains("decode_cache_control tier must be no higher")
        );
    }

    #[test]
    fn test_dynamic_temperature_newtype_validation_valid() {
        assert_eq!(DynamicTemperature::try_from(0.0).unwrap().get(), 0.0);
        assert_eq!(DynamicTemperature::try_from(1.0).unwrap().get(), 1.0);
        assert_eq!(DynamicTemperature::try_from(2.0).unwrap().get(), 2.0);
        assert_eq!(
            DynamicTemperatureSuffix::try_from("</think>")
                .unwrap()
                .as_str(),
            "</think>"
        );
    }

    #[test]
    fn test_dynamic_temperature_newtype_validation_invalid() {
        assert!(DynamicTemperature::try_from(-0.1).is_err());
        assert!(DynamicTemperature::try_from(2.1).is_err());
        assert!(DynamicTemperatureSuffix::try_from(String::new()).is_err());
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
