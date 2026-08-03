<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GWP datastores

This optional Helm chart installs one shared etcd and one shared Redis instance
for Global Workload Plane development and proof-of-concept deployments:

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

Redis and etcd can be disabled independently with `redis.enabled=false` and
`etcd.enabled=false`.

The chart intentionally uses single replicas and ephemeral storage. Use
externally managed, persistent, highly available services for production.
