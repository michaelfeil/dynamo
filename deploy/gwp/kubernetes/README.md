# Global Workload Plane on Kubernetes

This base runs three replicas. Each pod contains:

- Envoy on port 8080, owning client connections and streaming.
- One GWP process on port 8091, serving Envoy's gRPC ext-authz scheduling
  checks, asynchronous lifecycle RPCs, and standard gRPC health.
- GWP metrics on port 9090.
- Read-only Envoy Prometheus metrics on port 9902.

The Service exposes Envoy on port 80, GWP metrics on port 9090, and the
read-only Envoy metrics bridge on port 9902. It deliberately does not expose
the GWP gRPC port or full Envoy admin API. Envoy returns 404 for `/v1/models`;
configured routes are authoritative and model discovery does not use that
endpoint.

## Configure

Edit `configmap.yaml` before deployment:

1. Set `data.etcd-endpoints` to the etcd cluster used for GWP replica
   discovery.
2. Configure `session.backend` as shared etcd or Redis affinity. It is
   independent of discovery and may use a different service.
3. Set endpoint ingress and planner URLs.
4. Declare every external model alias under `routes`.
5. Add dimensioned endpoint `properties` (for example `region: [us]`) when
   Alyx routing-requirement filtering is needed.
6. Tune GWP worker scoring defaults in `data.b10-router.yaml`, or override
   planner/local source weights and scoring terms per canonical model under
   `model_policies.models.<model>.load_balancing` in `gwp.yaml`. Both files are
   hot-reloaded.

The ConfigMap name is stable so mounted `gwp.yaml` and `b10-router.yaml`
updates are hot-reloaded. Changing endpoint and route references in one update
keeps the configuration valid.

The Kustomization intentionally uses replaceable image names. Point
`dynamo-gwp` at the image built from `lib/gwp/Dockerfile`, and
`dynamo-gwp-envoy` at the Envoy v1.39.0 image containing the Rust lifecycle
Wasm module:

```console
docker build -f lib/gwp/Dockerfile -t REGISTRY/dynamo-gwp:TAG .
docker push REGISTRY/dynamo-gwp:TAG
docker build -f lib/gwp-envoy-filter/Dockerfile \
  -t REGISTRY/dynamo-gwp-envoy:TAG .
docker push REGISTRY/dynamo-gwp-envoy:TAG

cd deploy/gwp
kustomize edit set image dynamo-gwp=REGISTRY/dynamo-gwp:TAG
kustomize edit set image dynamo-gwp-envoy=REGISTRY/dynamo-gwp-envoy:TAG
```

Render and inspect the resources before handing them to the cluster:

```console
kustomize build deploy/gwp > /tmp/dynamo-gwp.yaml
```

The base targets the separate `gwp` namespace. Apply it through the cluster's normal
GitOps or manifest deployment path.

## Networking and rollout

All replicas must reach:

- etcd on its client port;
- every configured ingress and planner URL;
- when `DYN_EVENT_PLANE=zmq`, every other GWP pod on dynamically allocated TCP
  ports used by direct publisher/subscriber synchronization;
- when `DYN_EVENT_PLANE=nats`, the configured `NATS_SERVER` on its client port.

If a NetworkPolicy is present, allow pod-to-pod TCP between pods labeled
`app.kubernetes.io/name=dynamo-gwp`. The Service is not used for ZMQ because
each publisher advertises its own pod address through etcd.

The checked-in ConfigMap defaults to direct ZMQ. Set its `event-plane` value to
`nats` to move replica lifecycle events to NATS Core pub-sub; etcd remains the
membership/discovery plane in either mode. The request plane remains TCP.
Each canonical routable model owns an independent scheduler and event channel;
aliases resolve to the canonical model before selecting that channel.

The rolling update keeps one replica available. Readiness waits for an initial
topology with at least one routable worker and for replica warm-up; an individual
model without workers fails at scheduling time without removing the whole GWP
replica from service. The 45-second pod termination grace exceeds GWP's
30-second scheduler drain window.

## Optional development datastores

For a self-contained development or proof-of-concept deployment, the
[`gwp-datastores`](../../helm/charts/gwp-datastores/README.md) Helm chart
installs one shared Redis, etcd, and NATS instance. These are separate cluster
services rather than per-GWP sidecars. NATS is available for consumers that are
explicitly configured to use it; installing the chart does not replace GWP's
ZMQ event plane. The chart is intentionally single-replica and ephemeral; use
managed, highly available services in production.

## Metrics scraping

GWP exposes Prometheus metrics on port 9090 at `/metrics`. Envoy keeps its full
admin API loopback-only on port 9901 and exposes a read-only bridge containing
only `GET /stats/prometheus` on port 9902. The `ServiceMonitor`
(`kubernetes/servicemonitor.yaml`) scrapes both endpoints for every GWP pod.

The pod `prometheus.io/scrape` annotations remain a vanilla-Prometheus fallback
for GWP's port 9090. Prometheus installations without ServiceMonitor support
must add an equivalent second scrape for the `envoy-metrics` service port and
`/stats/prometheus`; Kubernetes scrape annotations cannot describe two ports.
The ServiceMonitor needs the `monitoring.coreos.com/v1` CRD; drop it from
`kustomization.yaml` on clusters without it.
