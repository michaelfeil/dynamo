# dynamo-gwp

Global Workload Plane routes OpenAI-compatible requests across Dynamo ingress
deployments. Envoy owns client streams; GWP provides gRPC ext-authz scheduling,
lifecycle accounting, health, planner reflection, and shared session affinity.

This crate is an experimental proof of concept. Start with:

- [Setup and operations](../../docs/components/gwp/README.md)
- [Design](../../docs/design-docs/global-workload-plane-design.md)
- [Kubernetes deployment](../../deploy/gwp/kubernetes/README.md)
- [Live Composer PoC](../../deploy/gwp/poc/README.md)

## Code map

- `config`, `models` — endpoint routes, aliases, policies, and hot reload.
- `reflector`, `identity` — planner polling and worker ownership.
- `session`, `tokens` — affinity backends and real/pseudo tokenization.
- `router`, `core` — selection, booking, reconciliation, and lifecycle state.
- `grpc`, `server` — Envoy scheduling/lifecycle APIs and process wiring.
- `metrics`, `lifecycle` — observability, readiness, drain, and cleanup.

## Build

```console
cargo build -p dynamo-gwp --features server,etcd,redis --bin dynamo-gwp
cargo test -p dynamo-gwp --features server,etcd,redis
```

Features are opt-in: `server` enables the Tonic entrypoint, while `etcd` and
`redis` enable the corresponding shared affinity stores. The sidecar image
enables all three:

```console
docker build -f lib/gwp/Dockerfile -t dynamo-gwp:dev .
```

Pseudo tokenization is the default. A real tokenizer bundle is a directory
containing `tokenizer.json`, `chat_template.jinja`, and
`tokenizer_config.json`. A `template_only` policy loads only
`chat_template.jinja` and `tokenizer_config.json` — it renders the chat
template for chat-completions but pseudo-tokenizes the rendered text, keeping
the template's effect on prefix identity without the real tokenizer's startup
and per-request cost. Tokenizer selection and affinity backend selection are
startup configuration because they define stable routing identities.

```yaml
tokenization:
  models:
    glm-5.2:
      mode: template_only
      directory: /workspace/lib/gwp/vendored_tokenizers/glm5.2
```
