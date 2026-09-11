// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bidirectional conversion between the coordinator protocol and native requests.
//! HTTP transports do not interpret worker payloads.

use super::*;
use crate::protocol;
use crate::{
    DeniedGenerationRequest, DeniedRequest, DisaggregationStrategy, GenerationAdmission,
    GenerationRequest, RequestContext, RouterRequestNew,
};
use anyhow::{Context, Result, anyhow, bail};
use dynamo_kv_router::protocols::{BlockExtraInfo, BlockMmObjectInfo, RoutingConstraints};
use dynamo_runtime::logging::DistributedTraceContext;
use rmpv::Value;
use std::collections::{HashMap, HashSet};

pub(crate) fn encode_request(
    context: &RequestContext,
    request: GenerationRequest,
) -> Result<NewRequestV1> {
    let GenerationRequest {
        routing_request,
        primary_worker_request,
        decode_worker_request: _,
    } = request;
    let Value::Map(mut worker) = primary_worker_request else {
        bail!("worker_args must be a map");
    };
    let session_id = optional_worker_string(&worker, "user")?;
    let cache_salt = optional_worker_string(&worker, "cache_salt")?;
    let mm_args = worker
        .iter()
        .position(|(key, _)| key.as_str() == Some("mm_args"))
        .map(|index| worker.remove(index).1)
        .filter(|value| !value.is_nil());
    let mm_payloads = mm_args
        .map(|value| {
            let args = rmpv::ext::from_value(value).context("invalid mm_args")?;
            mm_payloads_from_args(args)
        })
        .transpose()?;
    worker.retain(|(key, _)| key.as_str() != Some("tokens"));
    let mm_routing_args = routing_request
        .block_mm_infos
        .as_ref()
        .map(|blocks| mm_routing_args_from_blocks(blocks));
    let mut required_taints = routing_request
        .routing_constraints
        .required_taints
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    required_taints.sort();
    let mut preferred_taints = routing_request
        .routing_constraints
        .preferred_taints
        .iter()
        .map(|(taint, weight)| WeightedTaintV1 {
            taint: taint.clone(),
            weight: *weight,
        })
        .collect::<Vec<_>>();
    preferred_taints.sort_by(|left, right| left.taint.cmp(&right.taint));
    let mut allowed_worker_ids = routing_request
        .allowed_worker_ids
        .unwrap_or_default()
        .into_iter()
        .collect::<Vec<_>>();
    allowed_worker_ids.sort_unstable();
    let trace_context = context.trace_context().map(|trace| TraceContextV1 {
        trace_id: trace.trace_id.clone(),
        span_id: trace.span_id.clone(),
        parent_id: trace.parent_id.clone(),
        tracestate: trace.tracestate.clone(),
        x_request_id: trace.x_request_id.clone(),
        request_id: trace.request_id.clone(),
    });
    let request = NewRequestV1 {
        request_id: context.id().to_string(),
        trace_context,
        metadata: context.metadata_snapshot(),
        tokens: routing_request.tokens,
        mm_routing_args,
        routing: Some(RoutingV1 {
            session_id,
            cache_salt,
            do_not_queue: routing_request.do_not_queue,
            priority_jump: routing_request.priority_jump,
            priority_load_shed_percent: u32::from(routing_request.priority_load_shed_percent),
            allowed_worker_ids,
            constraints: Some(RoutingConstraintsV1 {
                required_taints,
                preferred_taints,
            }),
        }),
        mm_payloads,
        worker_msgpack: rmp_serde::to_vec_named(&Value::Map(worker))?,
    };
    protocol::validate_request(&request)?;
    Ok(request)
}

#[cfg(test)]
pub(crate) fn map_value<'a>(map: &'a Value, key: &str) -> Option<&'a Value> {
    let Value::Map(entries) = map else {
        return None;
    };
    entries
        .iter()
        .find_map(|(candidate, value)| (candidate.as_str() == Some(key)).then_some(value))
}

fn optional_worker_string(worker: &[(Value, Value)], key: &str) -> Result<Option<String>> {
    match worker
        .iter()
        .find(|(candidate, _)| candidate.as_str() == Some(key))
        .map(|(_, value)| value)
    {
        None | Some(Value::Nil) => Ok(None),
        Some(value) => Ok(Some(
            value
                .as_str()
                .with_context(|| format!("worker_args.{key} must be a string"))?
                .to_owned(),
        )),
    }
}

#[derive(serde::Deserialize)]
struct MmArgs {
    mm_embeds_encoded: Option<Vec<Value>>,
    mm_kwargs: Option<Vec<Value>>,
    media_token_id: Option<i64>,
    mm_hashes: Option<Vec<String>>,
    mm_positions: Option<Vec<u64>>,
    mm_lengths: Option<Vec<u64>>,
    mm_native_inputs: Option<Vec<NativeMmInput>>,
}

#[derive(serde::Deserialize)]
struct NativeMmInput {
    #[serde(rename = "type")]
    kind: String,
    url: String,
}

fn mm_routing_args_from_blocks(
    blocks: &[Option<dynamo_kv_router::protocols::BlockExtraInfo>],
) -> MmRoutingArgsV1 {
    MmRoutingArgsV1 {
        blocks: blocks
            .iter()
            .map(|block| MmRoutingBlockV1 {
                present: block.is_some(),
                objects: block
                    .as_ref()
                    .map(|block| {
                        block
                            .mm_objects
                            .iter()
                            .map(|object| MmRoutingObjectV1 {
                                mm_hash: object.mm_hash,
                                offsets: object
                                    .offsets
                                    .iter()
                                    .map(|&(start, end)| TokenRangeV1 {
                                        start: start as u64,
                                        end: end as u64,
                                    })
                                    .collect(),
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
            })
            .collect(),
    }
}

fn mm_payloads_from_args(args: MmArgs) -> Result<MmPayloadsV1> {
    let embeddings = args
        .mm_embeds_encoded
        .unwrap_or_default()
        .into_iter()
        .map(|value| match value {
            Value::Binary(value) => Ok(value),
            _ => bail!("worker_args.mm_args.mm_embeds_encoded must contain bytes"),
        })
        .collect::<Result<Vec<_>>>()?;
    let kwargs = args
        .mm_kwargs
        .unwrap_or_default()
        .into_iter()
        .map(|value| {
            let value = match value {
                Value::Binary(value) => mm_kwarg_v1::Value::Bytes(value),
                Value::String(value) => mm_kwarg_v1::Value::String(
                    value
                        .into_str()
                        .context("worker_args.mm_args.mm_kwargs contains invalid UTF-8")?,
                ),
                _ => bail!("worker_args.mm_args.mm_kwargs must contain bytes or strings"),
            };
            Ok(MmKwargV1 { value: Some(value) })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(MmPayloadsV1 {
        embeddings,
        kwargs,
        media_token_id: args.media_token_id.unwrap_or(-1),
        hashes: args.mm_hashes.unwrap_or_default(),
        positions: args.mm_positions.unwrap_or_default(),
        lengths: args.mm_lengths.unwrap_or_default(),
        native_inputs: args
            .mm_native_inputs
            .unwrap_or_default()
            .into_iter()
            .map(|input| NativeMmInputV1 {
                kind: input.kind,
                url: input.url,
            })
            .collect(),
    })
}

pub(crate) fn decode_new_request(
    request_id: String,
    request: NewRequestV1,
    strategy: DisaggregationStrategy,
) -> Result<GenerationRequest> {
    let routing = request.routing.context("new_request.routing is required")?;
    let payload: Value =
        rmp_serde::from_slice(&request.worker_msgpack).context("invalid worker_msgpack")?;
    let Value::Map(mut worker) = payload else {
        bail!("worker_msgpack must contain a map");
    };
    if worker
        .iter()
        .any(|(key, _)| matches!(key.as_str(), Some("tokens" | "mm_args")))
    {
        bail!("worker_msgpack must exclude tokens and mm_args");
    }
    worker.retain(|(key, _)| key.as_str() != Some("id"));
    insert(&mut worker, "id", Value::from(request_id));
    insert(
        &mut worker,
        "tokens",
        Value::Map(vec![(Value::from("tokens"), uint_array(&request.tokens))]),
    );
    // Decode needs the same generation parameters, but not the potentially
    // large multimodal input blobs consumed during prefill.
    let decode_worker_request =
        (strategy == DisaggregationStrategy::PrefillFirst).then(|| Value::Map(worker.clone()));
    if let Some(mm_payloads) = request.mm_payloads {
        insert(&mut worker, "mm_args", mm_payloads_to_value(mm_payloads)?);
    }
    let primary_worker_request = Value::Map(worker);
    let routing_constraints = routing.constraints.unwrap_or_default();
    let block_mm_infos = request
        .mm_routing_args
        .map(mm_routing_args_from_wire)
        .transpose()?;
    Ok(GenerationRequest {
        routing_request: RouterRequestNew {
            tokens: request.tokens,
            block_mm_infos,
            routing_constraints: RoutingConstraints {
                required_taints: routing_constraints.required_taints.into_iter().collect(),
                preferred_taints: routing_constraints
                    .preferred_taints
                    .into_iter()
                    .map(|taint| (taint.taint, taint.weight))
                    .collect::<HashMap<_, _>>(),
            },
            allowed_worker_ids: (!routing.allowed_worker_ids.is_empty()).then(|| {
                routing
                    .allowed_worker_ids
                    .into_iter()
                    .collect::<HashSet<_>>()
            }),
            priority_jump: routing.priority_jump,
            priority_load_shed_percent: u8::try_from(routing.priority_load_shed_percent)
                .context("routing.priority_load_shed_percent exceeds uint8")?,
            do_not_queue: routing.do_not_queue,
        },
        primary_worker_request,
        decode_worker_request,
    })
}

pub(crate) fn trace_context_from_wire(trace: protocol::TraceContextV1) -> DistributedTraceContext {
    DistributedTraceContext::new(
        trace.trace_id,
        trace.span_id,
        trace.parent_id,
        trace.tracestate,
        trace.x_request_id,
        trace.request_id,
    )
}

pub(crate) fn mm_routing_args_from_wire(
    args: MmRoutingArgsV1,
) -> Result<Vec<Option<BlockExtraInfo>>> {
    args.blocks
        .into_iter()
        .map(|block| {
            if !block.present {
                return Ok(None);
            }
            Ok(Some(BlockExtraInfo {
                mm_objects: block
                    .objects
                    .into_iter()
                    .map(|object| {
                        Ok(BlockMmObjectInfo {
                            mm_hash: object.mm_hash,
                            offsets: object
                                .offsets
                                .into_iter()
                                .map(|range| {
                                    Ok((usize::try_from(range.start)?, usize::try_from(range.end)?))
                                })
                                .collect::<Result<Vec<_>>>()?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            }))
        })
        .collect()
}

fn mm_payloads_to_value(payloads: MmPayloadsV1) -> Result<Value> {
    let kwargs = payloads
        .kwargs
        .into_iter()
        .map(
            |kwarg| match kwarg.value.context("multimodal kwarg value is required")? {
                mm_kwarg_v1::Value::Bytes(value) => Ok(Value::Binary(value)),
                mm_kwarg_v1::Value::String(value) => Ok(Value::from(value)),
            },
        )
        .collect::<Result<Vec<_>>>()?;
    let native_inputs = payloads
        .native_inputs
        .into_iter()
        .map(|input| {
            Value::Map(vec![
                (Value::from("type"), Value::from(input.kind)),
                (Value::from("url"), Value::from(input.url)),
            ])
        })
        .collect::<Vec<_>>();
    Ok(Value::Map(vec![
        (
            Value::from("mm_embeds_encoded"),
            Value::Array(payloads.embeddings.into_iter().map(Value::Binary).collect()),
        ),
        (Value::from("mm_kwargs"), Value::Array(kwargs)),
        (
            Value::from("media_token_id"),
            Value::from(payloads.media_token_id),
        ),
        (
            Value::from("mm_hashes"),
            Value::Array(payloads.hashes.into_iter().map(Value::from).collect()),
        ),
        (Value::from("mm_positions"), uint_array(&payloads.positions)),
        (Value::from("mm_lengths"), uint_array(&payloads.lengths)),
        (Value::from("mm_native_inputs"), Value::Array(native_inputs)),
    ]))
}

fn uint_array<T>(values: &[T]) -> Value
where
    T: Copy + Into<u64>,
{
    Value::Array(
        values
            .iter()
            .map(|value| Value::from((*value).into()))
            .collect(),
    )
}

fn insert(map: &mut Vec<(Value, Value)>, key: &'static str, value: Value) {
    map.push((Value::from(key), value));
}

impl From<crate::GenerationAdmission> for AdmissionV1 {
    fn from(value: crate::GenerationAdmission) -> Self {
        Self {
            estimated_overlap_tokens: value.estimated_overlap_tokens,
            best_overlap_blocks: value.best_overlap_blocks,
            prefill_worker_id: value.prefill_worker_id,
            prefill_dp_rank: value.prefill_dp_rank,
            decode_worker_id: value.decode_worker_id,
            decode_dp_rank: value.decode_dp_rank,
        }
    }
}

pub(crate) fn denial_frame(value: DeniedGenerationRequest) -> GenerationResponseFrameV1 {
    GenerationResponseFrameV1 {
        frame: Some(generation_response_frame_v1::Frame::Denied(
            DeniedGenerationV1 {
                denied: Some(denied_request_to_wire(value.denied)),
                admission: value.admission.map(Into::into),
            },
        )),
    }
}

fn denied_request_to_wire(value: DeniedRequest) -> DeniedRequestV1 {
    let mut output = DeniedRequestV1::default();
    match value {
        DeniedRequest::RouterBackpressure {
            reason,
            queued_isl_tokens,
            max_queued_isl_tokens,
        } => {
            output.kind = DeniedKindV1::RouterBackpressure.into();
            output.reason = Some(reason);
            output.queued_isl_tokens = queued_isl_tokens as u64;
            output.max_queued_isl_tokens = max_queued_isl_tokens.map(|value| value as u64);
        }
        DeniedRequest::RequiredComponentsDown { name } => {
            output.kind = DeniedKindV1::RequiredComponentsDown.into();
            output.name = Some(name);
        }
        DeniedRequest::NextRouterUnreachable { error } => {
            output.kind = DeniedKindV1::NextRouterUnreachable.into();
            output.error = Some(error);
        }
        DeniedRequest::ProtocolError { received } => {
            output.kind = DeniedKindV1::ProtocolError.into();
            output.received = Some(received);
        }
        DeniedRequest::Cancelled() => output.kind = DeniedKindV1::Cancelled.into(),
        DeniedRequest::FirstWorkerEventFailed { error } => {
            output.kind = DeniedKindV1::FirstWorkerEventFailed.into();
            output.error = Some(error);
        }
    }
    output
}

impl From<AdmissionV1> for GenerationAdmission {
    fn from(value: AdmissionV1) -> Self {
        Self {
            estimated_overlap_tokens: value.estimated_overlap_tokens,
            best_overlap_blocks: value.best_overlap_blocks,
            prefill_worker_id: value.prefill_worker_id,
            prefill_dp_rank: value.prefill_dp_rank,
            decode_worker_id: value.decode_worker_id,
            decode_dp_rank: value.decode_dp_rank,
        }
    }
}

impl TryFrom<DeniedGenerationV1> for DeniedGenerationRequest {
    type Error = anyhow::Error;

    fn try_from(value: DeniedGenerationV1) -> Result<Self> {
        let denied = value
            .denied
            .context("generation denial did not contain a reason")?
            .try_into()?;
        Ok(Self {
            denied,
            admission: value.admission.map(Into::into),
        })
    }
}

impl TryFrom<DeniedRequestV1> for DeniedRequest {
    type Error = anyhow::Error;

    fn try_from(value: DeniedRequestV1) -> Result<Self> {
        let kind = DeniedKindV1::try_from(value.kind)
            .map_err(|_| anyhow!("unknown generation denial kind {}", value.kind))?;
        let number = |value: u64, field: &str| {
            usize::try_from(value).with_context(|| format!("{field} exceeds platform usize"))
        };
        Ok(match kind {
            DeniedKindV1::RouterBackpressure => Self::RouterBackpressure {
                reason: value.reason.context("router denial is missing reason")?,
                queued_isl_tokens: number(value.queued_isl_tokens, "queued_isl_tokens")?,
                max_queued_isl_tokens: value
                    .max_queued_isl_tokens
                    .map(|value| number(value, "max_queued_isl_tokens"))
                    .transpose()?,
            },
            DeniedKindV1::RequiredComponentsDown => Self::RequiredComponentsDown {
                name: value.name.context("component denial is missing name")?,
            },
            DeniedKindV1::NextRouterUnreachable => Self::NextRouterUnreachable {
                error: value.error.context("unreachable denial is missing error")?,
            },
            DeniedKindV1::ProtocolError => Self::ProtocolError {
                received: value
                    .received
                    .context("protocol denial is missing response")?,
            },
            DeniedKindV1::Cancelled => Self::Cancelled(),
            DeniedKindV1::FirstWorkerEventFailed => Self::FirstWorkerEventFailed {
                error: value.error.context("worker denial is missing error")?,
            },
            DeniedKindV1::Unspecified => bail!("generation denial kind is unspecified"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::pipeline::context::Controller;
    use std::sync::Arc;

    #[test]
    fn request_round_trip_preserves_worker_field_shapes() {
        let context = RequestContext::new(
            Arc::new(Controller::new("token-shape".to_string())),
            None,
            Default::default(),
        );
        let worker = Value::Map(vec![
            ("model".into(), "test-model".into()),
            ("sampling_params".into(), Value::Map(vec![])),
            ("streaming".into(), false.into()),
            ("user".into(), "session-a".into()),
            ("cache_salt".into(), "tenant-a".into()),
            ("prompt_logprobs".into(), Value::Array(vec![(-0.5).into()])),
            (
                "future_worker_option".into(),
                Value::Binary(vec![0, 255, 3]),
            ),
            (
                "tokens".into(),
                Value::Map(vec![(
                    "tokens".into(),
                    Value::Binary(
                        [1_u32, 2, 151_643]
                            .into_iter()
                            .flat_map(u32::to_le_bytes)
                            .collect(),
                    ),
                )]),
            ),
            (
                "mocker_config".into(),
                Value::Map(vec![("speedup_ratio".into(), 2.0.into())]),
            ),
            (
                "chat_template_kwargs".into(),
                Value::Map(vec![("thinking".into(), true.into())]),
            ),
            (
                "b10_request_performance".into(),
                Value::Map(vec![("priority_engine".into(), 7.into())]),
            ),
        ]);
        let original = worker.clone();
        let wire = encode_request(
            &context,
            GenerationRequest {
                routing_request: RouterRequestNew {
                    tokens: vec![1, 2, 151_643],
                    ..Default::default()
                },
                primary_worker_request: worker,
                decode_worker_request: None,
            },
        )
        .unwrap();
        let payload: Value = rmp_serde::from_slice(&wire.worker_msgpack).unwrap();
        assert!(map_value(&payload, "tokens").is_none());
        assert!(map_value(&payload, "mm_args").is_none());
        assert_eq!(
            wire.routing.as_ref().unwrap().session_id.as_deref(),
            Some("session-a")
        );
        assert_eq!(
            wire.routing.as_ref().unwrap().cache_salt.as_deref(),
            Some("tenant-a")
        );
        use prost::Message;
        let wire = NewRequestV1::decode(wire.encode_to_vec().as_slice()).unwrap();

        let decoded = decode_new_request(
            "token-shape".to_string(),
            wire,
            DisaggregationStrategy::PrefillFirst,
        )
        .unwrap();
        let expected = uint_array(&[1_u32, 2, 151_643]);
        assert_eq!(decoded.primary_worker_request["tokens"]["tokens"], expected);
        let decode = decoded.decode_worker_request.unwrap();
        assert_eq!(decode["tokens"]["tokens"], expected);
        for payload in [&decoded.primary_worker_request, &decode] {
            for (key, value) in original.as_map().unwrap() {
                if key.as_str() != Some("tokens") {
                    assert_eq!(&payload[key.as_str().unwrap()], value, "field {key}");
                }
            }
        }
    }

    #[test]
    fn multimodal_conversion_moves_large_buffers_in_both_directions() {
        // Embeddings and kwargs are alternative worker input formats.
        for field in ["mm_embeds_encoded", "mm_kwargs"] {
            let bytes = vec![7; 10 * 1024 * 1024];
            let allocation = bytes.as_ptr();
            let worker = Value::Map(vec![
                ("model".into(), "test-model".into()),
                ("sampling_params".into(), Value::Map(vec![])),
                (
                    "mm_args".into(),
                    Value::Map(vec![(
                        field.into(),
                        Value::Array(vec![Value::Binary(bytes)]),
                    )]),
                ),
            ]);
            let context = RequestContext::new(
                Arc::new(Controller::new("large-mm".to_string())),
                None,
                Default::default(),
            );
            let wire = encode_request(
                &context,
                GenerationRequest {
                    routing_request: RouterRequestNew {
                        tokens: vec![1],
                        ..Default::default()
                    },
                    primary_worker_request: worker,
                    decode_worker_request: None,
                },
            )
            .unwrap();
            let general: Value = rmp_serde::from_slice(&wire.worker_msgpack).unwrap();
            assert!(map_value(&general, "tokens").is_none());
            assert!(map_value(&general, "mm_args").is_none());
            let mm = wire.mm_payloads.as_ref().unwrap();
            let encoded = if field == "mm_embeds_encoded" {
                &mm.embeddings[0]
            } else {
                let Some(mm_kwarg_v1::Value::Bytes(bytes)) = &mm.kwargs[0].value else {
                    panic!("kwargs must retain binary encoding")
                };
                bytes
            };
            assert_eq!(encoded.as_ptr(), allocation, "encoder copied {field}");

            let decoded = decode_new_request(
                "large-mm".to_string(),
                wire,
                DisaggregationStrategy::PrefillFirst,
            )
            .unwrap();
            let Value::Binary(bytes) = &decoded.primary_worker_request["mm_args"][field][0] else {
                panic!("worker input must retain binary encoding")
            };
            assert_eq!(bytes.as_ptr(), allocation, "decoder copied {field}");
            assert_eq!(bytes.len(), 10 * 1024 * 1024);
            assert!(decoded.decode_worker_request.unwrap()["mm_args"].is_nil());
        }
    }
}
