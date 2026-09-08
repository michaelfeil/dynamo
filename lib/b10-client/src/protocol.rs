// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Version-one protobuf messages for remote generation coordination.
//!
//! HTTP request bodies contain one encoded [`NewRequestV1`]. Response
//! bodies contain [`GenerationResponseFrameV1`] messages, each prefixed by a
//! four-byte big-endian payload length. The HTTP path selects protocol version one.

use prost::Message;

pub(crate) mod codec;

mod wire {
    include!(concat!(env!("OUT_DIR"), "/dynamo.b10.generation.v1.rs"));
}

pub use wire::denied_request::Kind as DeniedKindV1;
pub use wire::{
    Admission as AdmissionV1, DeniedGeneration as DeniedGenerationV1,
    DeniedRequest as DeniedRequestV1, GenerationResponseFrame as GenerationResponseFrameV1,
    MmKwarg as MmKwargV1, MmPayloads as MmPayloadsV1, MmRoutingArgs as MmRoutingArgsV1,
    MmRoutingBlock as MmRoutingBlockV1, MmRoutingObject as MmRoutingObjectV1,
    NativeMmInput as NativeMmInputV1, NewRequest as NewRequestV1, ProtocolError as ProtocolErrorV1,
    Routing as RoutingV1, RoutingConstraints as RoutingConstraintsV1, TokenRange as TokenRangeV1,
    TraceContext as TraceContextV1, WeightedTaint as WeightedTaintV1,
    generation_response_frame as generation_response_frame_v1, mm_kwarg as mm_kwarg_v1,
};

pub const REQUEST_CONTENT_TYPE: &str = "application/x-protobuf";
pub const RESPONSE_CONTENT_TYPE: &str = "application/x-protobuf-stream";
pub const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

pub fn validate_request(request: &NewRequestV1) -> anyhow::Result<()> {
    use anyhow::{bail, ensure};

    ensure!(!request.request_id.is_empty(), "request_id is required");
    for key in request.metadata.keys() {
        ensure!(!key.is_empty(), "metadata keys cannot be empty");
    }
    if let Some(trace) = &request.trace_context {
        ensure!(
            !trace.trace_id.is_empty(),
            "trace_context.trace_id is required"
        );
        ensure!(
            !trace.span_id.is_empty(),
            "trace_context.span_id is required"
        );
    }
    ensure!(!request.model.is_empty(), "new_request.model is required");
    ensure!(!request.tokens.is_empty(), "new_request.tokens is required");
    ensure!(request.routing.is_some(), "new_request.routing is required");
    let routing = request.routing.as_ref().expect("checked above");
    ensure!(
        routing.priority_load_shed_percent <= 100,
        "new_request.routing.priority_load_shed_percent must be at most 100"
    );
    if routing.session_id.as_deref() == Some("") {
        bail!("new_request.routing.session_id cannot be empty");
    }
    ensure!(
        !request.sampling_msgpack.is_empty(),
        "new_request.sampling_msgpack is required"
    );
    if request.lora.as_deref() == Some("") {
        bail!("new_request.lora cannot be empty");
    }
    for block in request.mm_routing_args.iter().flat_map(|args| &args.blocks) {
        ensure!(
            block.present || block.objects.is_empty(),
            "new_request.mm_routing_args absent blocks cannot contain objects"
        );
        for range in block.objects.iter().flat_map(|object| &object.offsets) {
            ensure!(
                range.start <= range.end,
                "new_request.mm_routing_args offsets must be ordered"
            );
        }
    }
    if let Some(payloads) = &request.mm_payloads
        && !(payloads.hashes.len() == payloads.positions.len()
            && payloads.hashes.len() == payloads.lengths.len())
    {
        bail!("new_request.mm_payloads hashes, positions, and lengths must have equal lengths");
    }
    if let Some(payloads) = &request.mm_payloads {
        ensure!(
            payloads.kwargs.iter().all(|kwarg| kwarg.value.is_some()),
            "new_request.mm_payloads kwargs require values"
        );
        ensure!(
            payloads
                .native_inputs
                .iter()
                .all(|input| !input.kind.is_empty() && !input.url.is_empty()),
            "new_request.mm_payloads native inputs require kind and url"
        );
    }
    Ok(())
}

pub fn encode_response_frame(frame: &GenerationResponseFrameV1) -> Vec<u8> {
    let payload = frame.encode_to_vec();
    let mut framed = Vec::with_capacity(4 + payload.len());
    framed.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    framed.extend_from_slice(&payload);
    framed
}
