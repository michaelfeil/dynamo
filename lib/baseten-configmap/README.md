# Baseten ConfigMap readers

Shared immutable snapshots of the Rust-consumed fields in the mounted Baseten
YAML configuration. This crate has no dependencies on other Dynamo crates.
Parsing and publishing never mutate scheduler state.

## Crate layout and tests

`src/lib.rs` exposes the public API. `src/config.rs` owns the schema and parser,
`src/reader.rs` owns snapshots and reload lifecycle, `src/registry.rs` owns file
sources and reader resolution, and `src/logging.rs` owns logging controls.
Implementation-specific unit tests stay beside their code. Public-API integration
tests live in `tests/readers.rs` and `tests/environment.rs`; the latter isolates
environment mutation in a subprocess. Run both with `cargo test -p baseten-configmap`.

## Usage

Resolve a reader at component construction and retain it:

```rust
use baseten_configmap::{ConfigReader, current_reader};

struct Component {
    config: ConfigReader,
}

impl Component {
    fn new() -> Self {
        Self { config: current_reader() }
    }

    fn evaluate(&self) {
        let snapshot = self.config.snapshot();
        let _threshold = snapshot.routing.router_queue_threshold_decode_tokens;
        // Keep this snapshot for the evaluation instead of loading each field separately.
    }
}
```

`current_reader()` uses a construction-scoped reader when present; otherwise it
resolves `DYN_LLMAPI_CONFIG_PATH`, `ENGINE_ARGS_OVERRIDE_GROUP`, and the existing
routing environment defaults at construction. Changing the environment affects
new resolutions, not existing readers. Snapshots and reloads never reread these
inputs. Initial resolution synchronously loads the configured file, including
`/configs/llm_api_config_router.yaml` when no path is set, before returning the
reader. Missing/invalid files fall back to the captured defaults. Environment-resolved
readers then start one 15-second reloader per source. `try_current_reader()`
also resolves an explicitly configured path, but allows standalone libraries to
retain their own defaults when no path, scope, or matching reader is registered.

This supports constructing Python objects inside a patched environment:

```python
with patch.dict(os.environ, {"DYN_LLMAPI_CONFIG_PATH": test_config_path}):
    component = Component(...)  # Rust constructor captures its reader here.
# component still follows test_config_path after the patch is restored.
```

Environment patches are process-wide: serialize such tests or run them in
subprocesses. For parallel Rust tests, use the construction scope below.

Use `reader_for(FileSource::new(path, group))` for an explicit source, or create
a `ReaderRegistry` for independent ownership. Equal paths, override groups, and
captured defaults within a registry share a snapshot slot. Explicit resolution
returns initial load errors and does not start a thread. Call
`reader.start_reloader(interval)` to enable polling, or `reader.reload()` manually.
Polling is idempotent per slot and stops on `stop_reloader()` or when the last
reader is dropped. Dropping one reader does not stop other readers sharing it.
The most recently environment-resolved reader is pinned for process-wide runtime
metadata access through `process_reader()`; this accessor does not recheck env.
Unlike components with retained readers, that metadata view is process-wide.

File paths remain paths to the mounted file, not canonicalized symlink targets.
Reloading follows Kubernetes mount replacement. An invalid reload retains the
previous snapshot; existing snapshot references are immutable and stay valid.
Writers are serialized across loading and publication. `replace(config)` publishes
an in-memory snapshot; it does not persist YAML, and later file reloads can replace
it. Use an in-memory reader when the file should not remain authoritative.

## Isolated construction in tests

```rust
use baseten_configmap::{ConfigReader, UnifiedConfig, with_reader};

let mut settings = UnifiedConfig::default();
settings.routing.router_queue_threshold_decode_tokens = Some(128);
let reader = ConfigReader::in_memory(settings.clone());
// let component = with_reader(&reader, Component::new);
settings.routing.router_queue_threshold_decode_tokens = Some(256);
reader.replace(settings);
// component's next snapshot observes 256, even on a different thread.
```

`with_reader` affects only synchronous constructor resolution on the current
thread. Constructed objects retain the reader and may move across threads or
Tokio tasks. The scope is nested and unwind-safe. It does not propagate into an
unpolled async constructor or spawned task; resolve/capture the reader before
that boundary and scope the synchronous construction where needed.

The extracted schema retains Rust's defaults, aliases, field-level routing
overrides, and sanitization. It does not merge arbitrary engine/frontend fields or
change Python's separate override semantics. Related values are coherent within
one snapshot; separate routing phases may intentionally take newer snapshots.

## Generation coordinator listener

`GenerationCoordinator` reads the Rust snapshot at construction; `start()` initializes
the selected backend and optional listener. Python does not parse this section:

```yaml
b10_generation_coordinator_config:
  host: 0.0.0.0
  port: null  # HTTP disabled; set 8080 to listen, or 0 for an ephemeral port
  remotes: null  # local orchestration
```

These are the defaults. `start()` takes no arguments; configuration is Rust-owned.
Direct generation remains available with HTTP disabled. Listener settings
are fixed at construction; an already-running listener requires a process restart
to change address. An override group's coordinator section replaces the whole
section. The native listener serves `/health` and `/v1/coordinate` and drains on
runtime shutdown. It has no HTTP authentication; expose it only on a trusted network.

`remotes` independently selects the backend for both Python and HTTP requests:

```yaml
b10_generation_coordinator_config:
  port: null  # client-only frontend; use 8080 to also expose an HTTP relay
  remotes:
    default: http://coordinator-frontend:8080/v1/coordinate
```

Exactly one named HTTP(S) backend is supported. Null/omitted remotes use local
orchestration. Local versus remote mode is fixed when the coordinator is constructed;
changing modes requires restart. Remote-only startup does not connect local
router/worker clients. In remote mode, endpoint updates apply to new
requests after the reader reloads (15-second polling); existing streams keep their
backend and cancellation behavior. Unchanged backends reuse their HTTP connection
pool. Local coordinators ignore remote endpoint updates. Removing remotes from a
remote coordinator rejects new requests rather than switching to local orchestration.
Invalid reloads retain the previous snapshot. Do not point a relay to itself
or create cycles between relays.
