// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() -> std::io::Result<()> {
    println!("cargo:rerun-if-changed=proto/generation_coordinator.proto");
    prost_build::Config::new()
        .btree_map([".dynamo.b10.generation.v1.NewRequest.metadata"])
        .compile_protos(&["proto/generation_coordinator.proto"], &["proto"])
}
