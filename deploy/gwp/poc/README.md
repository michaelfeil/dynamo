# Composer live-cluster PoC

This overlay runs the GWP sidecar and Envoy adapter against two Composer
deployments. The general configuration and operations guide is
[`docs/components/gwp`](../../../docs/components/gwp/README.md).

| component | address |
| --- | --- |
| GWP gRPC control | `127.0.0.1:8091` |
| GWP metrics | `127.0.0.1:9090` |
| Envoy ingress | `127.0.0.1:8080` |
| Envoy admin | `127.0.0.1:9901` |

`gwp.yaml` contains the live planner and ingress URLs. Model routes are
authoritative; the PoC does not use `/v1/models` for discovery.

## Run locally

Start a local etcd server on `127.0.0.1:2379`, then build and start GWP from
the repository root:

```console
cargo build -p dynamo-gwp --features server,etcd,redis --bin dynamo-gwp

DYN_GWP_CONFIG_PATH=deploy/gwp/poc/gwp.yaml \
DYN_LLMAPI_CONFIG_PATH=deploy/gwp/poc/b10-router.yaml \
DYN_GWP_GRPC_PORT=8091 \
DYN_SYSTEM_PORT=9090 \
ETCD_ENDPOINTS=http://127.0.0.1:2379 \
  target/debug/dynamo-gwp
```

Build and start the Envoy lifecycle adapter in another terminal:

```console
./lib/gwp-envoy-filter/build.sh
sudo install -D -m 0644 \
  lib/gwp-envoy-filter/target/wasm32-wasip1/release/dynamo_gwp_envoy_filter.wasm \
  /usr/local/lib/dynamo_gwp_envoy_filter.wasm
envoy --mode validate -c deploy/gwp/poc/envoy.yaml
envoy -c deploy/gwp/poc/envoy.yaml --concurrency 4
```

The corresponding container images are built from the repository root:

```console
docker build -f lib/gwp/Dockerfile -t dynamo-gwp:dev .
docker build -f lib/gwp-envoy-filter/Dockerfile -t dynamo-gwp-envoy:dev .
```

## Smoke test

Chat completions:

```console
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-session-id: gwp-chat-smoke' \
  -d '{"model":"composer-2-5","messages":[{"role":"user","content":"Reply with OK"}],"max_tokens":16}'
```

Text completions:

```console
curl -fsS http://127.0.0.1:8080/v1/completions \
  -H 'content-type: application/json' \
  -H 'x-session-id: gwp-completions-smoke' \
  -d '{"model":"composer-2-5","prompt":"Reply with OK","max_tokens":16}'
```

Successful responses include `x-routed-endpoint`, `x-session-id`, and the
downstream `x-baseten-dyn-worker-id`. Inspect GWP and Envoy metrics with:

```console
curl -fsS http://127.0.0.1:9090/metrics
curl -fsS 'http://127.0.0.1:9901/stats/prometheus?filter=gwp'
```

Edit `gwp.yaml` to exercise endpoint, route, or per-model load-balancing hot
reload. Process-wide scoring defaults in `b10-router.yaml` are reloaded
independently. Kubernetes manifests and
scraping details live in [`deploy/gwp/kubernetes`](../kubernetes/README.md).
