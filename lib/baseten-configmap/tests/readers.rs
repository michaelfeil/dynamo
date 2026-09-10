// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use baseten_configmap::{
    B10RoutingConfig, ConfigReader, FileSource, ReaderRegistry, UnifiedConfig, current_reader,
    with_reader,
};
use std::time::Duration;

fn config(weight: f64) -> UnifiedConfig {
    UnifiedConfig {
        routing: B10RoutingConfig {
            router_decode_block_weight: weight,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn coordinator_listener_defaults_disable_and_reload() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.yaml");
    std::fs::write(&path, "{}").unwrap();
    let reader = ReaderRegistry::default()
        .resolve(FileSource::new(&path, None))
        .unwrap();
    let initial = reader.snapshot();
    assert_eq!(initial.generation_coordinator.listen_address(), None);

    std::fs::write(
        &path,
        "b10_generation_coordinator_config: {host: '::1', port: 9000}",
    )
    .unwrap();
    reader.reload().unwrap();
    assert_eq!(
        reader.snapshot().generation_coordinator.listen_address(),
        Some("[::1]:9000".parse().unwrap())
    );
    assert_eq!(initial.generation_coordinator.port, None);

    for invalid in [
        "{port: -1}",
        "{port: 65536}",
        "{host: invalid}",
        "{ports: 9000}",
        "{remotes: {}}",
        "{remotes: {a: 'http://a', b: 'http://a/'}}",
        "{remotes: {a: 'http://a', b: 'file:///tmp/server'}}",
        "{affinity: {ttl_secs: 60}}",
        "{remotes: {a: 'http://a'}, affinity: {ttl_secs: 0}}",
        "{remotes: {a: 'http://a'}, affinity: {namespace: routing, component: frontend, ttl_secs: 60}}",
        "{remotes: {'': 'http://a'}}",
        "{remotes: {default: 'file:///tmp/server'}}",
        "{remotes: {default: 'http://user:secret@server'}}",
    ] {
        std::fs::write(
            &path,
            format!("b10_generation_coordinator_config: {invalid}"),
        )
        .unwrap();
        assert!(reader.reload().is_err());
        assert_eq!(reader.snapshot().generation_coordinator.port, Some(9000));
    }
    std::fs::write(&path, "b10_generation_coordinator_config: {port: null}").unwrap();
    reader.reload().unwrap();
    assert_eq!(
        reader.snapshot().generation_coordinator.listen_address(),
        None
    );

    std::fs::write(&path, "b10_generation_coordinator_config: {port: 9000}\noverride_args:\n  frontend:\n    b10_generation_coordinator_config: {host: '127.0.0.1', port: 0}\n").unwrap();
    let frontend = ReaderRegistry::default()
        .resolve(FileSource::new(&path, Some("frontend".into())))
        .unwrap();
    assert_eq!(
        frontend.snapshot().generation_coordinator.listen_address(),
        Some("127.0.0.1:0".parse().unwrap())
    );
    std::fs::write(&path, "b10_generation_coordinator_config: {port: null, remotes: {default: 'http://coordinator:8080/v1/coordinate'}}").unwrap();
    reader.reload().unwrap();
    assert_eq!(
        reader.snapshot().generation_coordinator.listen_address(),
        None
    );
    assert_eq!(
        reader
            .snapshot()
            .generation_coordinator
            .remotes
            .as_ref()
            .unwrap()["default"],
        "http://coordinator:8080/v1/coordinate"
    );
    std::fs::write(&path, "b10_generation_coordinator_config: {remotes: {default: 'http://a', canary: 'http://b'}, affinity: {ttl_secs: 60}}").unwrap();
    reader.reload().unwrap();
    let snapshot = reader.snapshot();
    assert_eq!(
        snapshot
            .generation_coordinator
            .remotes
            .as_ref()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        snapshot
            .generation_coordinator
            .affinity
            .as_ref()
            .unwrap()
            .ttl_secs,
        60
    );
}

#[test]
fn shared_source_reloads_existing_readers_without_mutating_old_snapshots() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.yaml");
    std::fs::write(&path, "b10_routing_config: {router_decode_block_weight: 2}").unwrap();
    let registry = ReaderRegistry::default();
    let source = FileSource::new(&path, None);
    let first = registry.resolve(source.clone()).unwrap();
    let second = registry.resolve(source).unwrap();
    assert!(first.shares_source(&second));
    let old = first.snapshot();
    std::fs::write(&path, "b10_routing_config: {router_decode_block_weight: 3}").unwrap();
    assert!(second.reload().unwrap());
    assert_eq!(first.snapshot().routing.router_decode_block_weight, 3.0);
    assert_eq!(old.routing.router_decode_block_weight, 2.0);
    assert!(!first.reload().unwrap());

    std::fs::write(&path, "b10_routing_config: [").unwrap();
    assert!(first.reload().is_err());
    assert_eq!(second.snapshot().routing.router_decode_block_weight, 3.0);
    std::fs::remove_file(&path).unwrap();
    assert!(first.reload().is_err());
    assert_eq!(second.snapshot().routing.router_decode_block_weight, 3.0);
}

#[test]
fn override_groups_and_registries_are_independent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.yaml");
    std::fs::write(
        &path,
        "override_args:\n  prefill:\n    b10_routing_config: {router_decode_block_weight: 4}\n",
    )
    .unwrap();
    let registry = ReaderRegistry::default();
    let base = registry.resolve(FileSource::new(&path, None)).unwrap();
    let prefill = registry
        .resolve(FileSource::new(&path, Some("prefill".into())))
        .unwrap();
    let isolated = ReaderRegistry::default()
        .resolve(FileSource::new(&path, None))
        .unwrap();
    assert!(!base.shares_source(&prefill));
    assert!(!base.shares_source(&isolated));
    base.replace(config(9.0));
    assert_eq!(prefill.snapshot().routing.router_decode_block_weight, 4.0);
    assert_eq!(isolated.snapshot().routing.router_decode_block_weight, 1.0);
}

#[test]
fn scoped_construction_binds_readers_across_threads_and_nested_scopes() {
    let first = ConfigReader::in_memory(config(2.0));
    let second = ConfigReader::in_memory(config(3.0));
    let (old_component, new_component) = with_reader(&first, || {
        let old_component = current_reader();
        let new_component = with_reader(&second, current_reader);
        assert!(current_reader().shares_source(&first));
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_reader(&second, || panic!("construction failed"));
        }));
        assert!(current_reader().shares_source(&first));
        (old_component, new_component)
    });
    first.replace(config(5.0));
    std::thread::spawn(move || {
        assert_eq!(
            old_component.snapshot().routing.router_decode_block_weight,
            5.0
        );
        assert_eq!(
            new_component.snapshot().routing.router_decode_block_weight,
            3.0
        );
    })
    .join()
    .unwrap();
}

#[test]
fn concurrent_readers_see_whole_replacements() {
    let reader = ConfigReader::in_memory(config(1.0));
    let worker = reader.clone();
    let task = std::thread::spawn(move || {
        for n in 1..1000 {
            let mut next = config(n as f64);
            next.routing.router_active_replicas = n;
            worker.replace(next);
        }
    });
    for _ in 0..1000 {
        let snapshot = reader.snapshot();
        assert_eq!(
            snapshot.routing.router_decode_block_weight,
            snapshot.routing.router_active_replicas as f64
        );
    }
    task.join().unwrap();
}

#[cfg(unix)]
#[test]
fn reload_follows_replaced_mount_symlink() {
    use std::os::unix::fs::symlink;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.yaml");
    std::fs::write(
        dir.path().join("first"),
        "b10_routing_config: {router_decode_block_weight: 2}",
    )
    .unwrap();
    std::fs::write(
        dir.path().join("second"),
        "b10_routing_config: {router_decode_block_weight: 3}",
    )
    .unwrap();
    symlink("first", &path).unwrap();
    let reader = ReaderRegistry::default()
        .resolve(FileSource::new(&path, None))
        .unwrap();
    symlink("second", dir.path().join("next")).unwrap();
    std::fs::rename(dir.path().join("next"), &path).unwrap();
    reader.reload().unwrap();
    assert_eq!(reader.snapshot().routing.router_decode_block_weight, 3.0);
}

#[test]
fn background_reload_can_be_stopped_and_restarted() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.yaml");
    std::fs::write(&path, "{}").unwrap();
    let reader = ReaderRegistry::default()
        .resolve(FileSource::new(&path, None))
        .unwrap();
    reader.start_reloader(Duration::from_millis(5)).unwrap();
    std::fs::write(&path, "b10_routing_config: {router_decode_block_weight: 2}").unwrap();
    let wait_for_weight = |weight| {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while reader.snapshot().routing.router_decode_block_weight != weight {
            assert!(
                std::time::Instant::now() < deadline,
                "background reload timed out"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    };
    wait_for_weight(2.0);
    reader.stop_reloader();
    reader.replace(config(9.0));
    assert_eq!(reader.snapshot().routing.router_decode_block_weight, 9.0);
    reader.start_reloader(Duration::from_millis(5)).unwrap();
    wait_for_weight(2.0);
    reader.stop_reloader();
}
