// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use baseten_configmap::current_reader;

#[test]
fn environment_path_is_resolved_at_construction() {
    const CHILD: &str = "BASETEN_CONFIGMAP_ENV_TEST_CHILD";
    const SECOND_PATH: &str = "BASETEN_CONFIGMAP_ENV_TEST_SECOND_PATH";
    if std::env::var_os(CHILD).is_none() {
        let dir = tempfile::tempdir().unwrap();
        let first_path = dir.path().join("first.yaml");
        let second_path = dir.path().join("second.yaml");
        std::fs::write(
            &first_path,
            "b10_routing_config: {router_decode_block_weight: 2}",
        )
        .unwrap();
        std::fs::write(
            &second_path,
            "b10_routing_config: {router_decode_block_weight: 3}",
        )
        .unwrap();
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "environment_path_is_resolved_at_construction",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("DYN_LLMAPI_CONFIG_PATH", &first_path)
            .env(SECOND_PATH, &second_path)
            .env_remove("ENGINE_ARGS_OVERRIDE_GROUP")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    // Only this test runs in the subprocess. Reader construction captures all
    // environment inputs before starting a polling thread.
    let first_path = std::env::var_os("DYN_LLMAPI_CONFIG_PATH").unwrap();
    let second_path = std::env::var_os(SECOND_PATH).unwrap();
    let first = current_reader();
    assert_eq!(first.snapshot().routing.router_decode_block_weight, 2.0);
    // SAFETY: isolated test process; the polling threads do not read env.
    unsafe {
        std::env::set_var("DYN_LLMAPI_CONFIG_PATH", &second_path);
    }
    let second = current_reader();
    assert_eq!(second.snapshot().routing.router_decode_block_weight, 3.0);
    assert!(!first.shares_source(&second));
    // SAFETY: as above. Restoring env must not rebind either existing reader.
    unsafe {
        std::env::set_var("DYN_LLMAPI_CONFIG_PATH", &first_path);
    }
    assert!(first.shares_source(&current_reader()));
    std::fs::write(
        &first_path,
        "b10_routing_config: {router_decode_block_weight: 4}",
    )
    .unwrap();
    first.reload().unwrap();
    assert_eq!(first.snapshot().routing.router_decode_block_weight, 4.0);
    assert_eq!(second.snapshot().routing.router_decode_block_weight, 3.0);
    first.stop_reloader();
    second.stop_reloader();
    std::thread::spawn(move || {
        assert_eq!(second.snapshot().routing.router_decode_block_weight, 3.0);
    })
    .join()
    .unwrap();
}
