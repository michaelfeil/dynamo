<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.s
-->

# 🔨 [Experimental] Using `dynamo build` to containerize inference graphs

This guide explains how to use the `dynamo build` command to containerize Dynamo inference graphs (pipelines) for deployment.

## Table of Contents

- [What is dynamo build?](#what-is-dynamo-build)
- [Building a containerized inference graph](#building-a-containerized-inference-graph)
- [Configuration options](#configuration-options)
- [Guided Example](#guided-example)
- [Advanced topics](#advanced-topics)

## What is dynamo build?

`dynamo build` is a command-line tool that helps containerize inference graphs created with Dynamo SDK. Simply run `dynamo build --containerize` to build a stand-alone Docker container that encapsulates your entire inference graph. This image can then be shared and run standalone. 

**Note:** This is currently an experimental feature and has only been tested on the examples available in the `examples/` directory. You may have to make some modifications, in particular if your inference graph introduces custom dependencies. 

## Building a containerized inference graph

The basic workflow for using `dynamo build` involves:

1. Defining your inference graph and testing locally with `dynamo serve`
2. Specifying a base image for your inference graph. More on this below.
3. Running `dynamo build` to build a containerized inference graph

### Basic usage

```bash
dynamo build <graph_definition> --containerize
```

## Guided Example for containerizing Hello World pipeline

This section will walk through a complete example of building a containerized inference graph. In this example, we simply containerize the Hello World pipeline available at `examples/hello_world`

### 1. Define your graph and check that it works with `dynamo serve`

```bash
cd examples/hello_world
dynamo serve hello_world:Frontend
```

### 2. Build a base image

We intend to support 2 base images which will be available as buildable targets in the top-level Earthfile. You may then use one of these images as the base image to build your inference graph.

1. Leaner image without CUDA and vLLM making it suitable for CPU only deployments. This is what we will use for the Hello World example. Available as `dynamo-base-docker` in the top-level Earthfile.
2. Base image with CUDA and vLLM making it suitable for GPU deployments. **Note:** While this is not yet available in the top-level Earthfile, you may use `dynamo:latest-vllm` image created from running `./container/build.sh` as a valid base image for this purpose.

```bash
export CI_REGISTRY_IMAGE=my-registry
export CI_COMMIT_SHA=hello-world

earthly +dynamo-base-docker --CI_REGISTRY_IMAGE=$CI_REGISTRY_IMAGE --CI_COMMIT_SHA=$CI_COMMIT_SHA
# Image should succesfully be built and tagged as my-registry/dynamo-base-docker:hello-world
```

### 3. Containerize your graph with `dynamo build`

```bash
export DYNAMO_IMAGE=my-registry/dynamo-base-docker:hello-world
dynamo build hello_world:Frontend --containerize

# Output will contain tag for the newly created image
# e.g frontend-hello-world:latest
```

### 4. Run your container

As a prerequisite, ensure you have NATS and etcd running by running the docker compose in the deploy directory. You can find it [here](../../deploy/docker-compose.yml).

```bash
docker compose up -d
```

Starting your container with host networking and required environment variables:
```bash
# Host networking is required for NATS and etcd to be accessible from the container
docker run --network host \
  --entrypoint sh \
  -w /src \
  frontend-hello-world:latest \
  -c "uv run dynamo serve hello_world:Frontend"
```
