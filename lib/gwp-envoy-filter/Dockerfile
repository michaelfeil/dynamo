# syntax=docker/dockerfile:1
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG RUST_IMAGE=rust:1.93.1-bookworm
ARG ENVOY_IMAGE=envoyproxy/envoy:v1.39.0

FROM ${RUST_IMAGE} AS builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends libprotobuf-dev protobuf-compiler \
    && rm -rf /var/lib/apt/lists/* \
    && rustup target add wasm32-wasip1
WORKDIR /workspace/lib/gwp-envoy-filter
COPY lib/gwp-envoy-filter/Cargo.toml lib/gwp-envoy-filter/Cargo.lock ./
COPY lib/gwp-envoy-filter/build.rs ./build.rs
COPY lib/gwp-envoy-filter/src ./src
COPY lib/gwp/proto /workspace/lib/gwp/proto
RUN cargo build --locked --release --target wasm32-wasip1

FROM ${ENVOY_IMAGE}
COPY --from=builder \
  /workspace/lib/gwp-envoy-filter/target/wasm32-wasip1/release/dynamo_gwp_envoy_filter.wasm \
  /usr/local/lib/dynamo_gwp_envoy_filter.wasm
