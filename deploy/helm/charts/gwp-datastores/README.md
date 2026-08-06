<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GWP datastores

This optional Helm chart installs one shared etcd, Redis, and NATS instance for
Global Workload Plane development and proof-of-concept deployments:

```console
helm install gwp-stores deploy/helm/charts/gwp-datastores --namespace dynamo
```

Configure GWP with:

```text
ETCD_ENDPOINTS=http://gwp-stores-gwp-datastores-etcd:2379
```

```yaml
session:
  backend:
    type: redis
    url: redis://gwp-stores-gwp-datastores-redis:6379/
```

NATS Core pub-sub is available at:

```text
NATS_SERVER=nats://gwp-stores-gwp-datastores-nats:4222
```

Installing NATS does not change GWP's event-plane transport automatically;
consumers must be configured to use this server. Redis, etcd, and NATS can be
disabled independently with `redis.enabled=false`, `etcd.enabled=false`, and
`nats.enabled=false`.

The chart intentionally uses single replicas and ephemeral storage. Use
externally managed, persistent, highly available services for production.
