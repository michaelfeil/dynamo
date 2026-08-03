// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto = "../gwp/proto/gwp.proto";
    println!("cargo:rerun-if-changed={proto}");
    prost_build::Config::new().compile_protos(&[proto], &["../gwp/proto"])?;
    Ok(())
}
