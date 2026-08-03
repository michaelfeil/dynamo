# syntax=docker/dockerfile:1
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG RUST_IMAGE=rust:1.93.1-bookworm
ARG RUNTIME_IMAGE=debian:bookworm-slim

FROM ${RUST_IMAGE} AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        clang \
        cmake \
        libprotobuf-dev \
        libssl-dev \
        pkg-config \
        protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .
RUN cargo build --release -p dynamo-gwp --features server,etcd,redis --bin dynamo-gwp

FROM ${RUNTIME_IMAGE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libssl3 libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 65532 --create-home gwp

COPY --from=builder /workspace/target/release/dynamo-gwp /usr/local/bin/dynamo-gwp
COPY --from=builder /workspace/lib/gwp/vendored_tokenizers /opt/dynamo-gwp/tokenizers

ENV DYN_SYSTEM_PORT=9090
USER 65532
EXPOSE 8091 9090
ENTRYPOINT ["/usr/local/bin/dynamo-gwp"]
