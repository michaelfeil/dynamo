# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os

# Import shared memory monitoring
import sys

import uvloop

from dynamo.runtime import DistributedRuntime, dynamo_worker

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared_memory_monitor import create_monitor, setup_background_monitor

# Create monitor if profiling is enabled
monitor = create_monitor("BACKEND")

uvloop.install()


class RequestHandler:
    def __init__(self):
        print("Initialized backend request handler with memory monitoring")
        if monitor:
            monitor.log_memory("Initial:")

    async def generate(self, request):
        if monitor:
            monitor.increment_request()

        max_tokens = request.get("max_tokens", 10)
        for i in range(max_tokens):
            await asyncio.sleep(1e-5)  # yield / switch contextW
            yield f"chunk{i}"


@dynamo_worker(register_shutdown=True)
async def worker(runtime: DistributedRuntime):
    component = runtime.namespace("openai/pipeline").component("worker")
    await component.create_service()

    endpoint = component.endpoint("generate")

    # Setup background memory monitoring
    monitor_task = setup_background_monitor(monitor)

    try:
        await endpoint.serve_endpoint(RequestHandler().generate)
    finally:
        if monitor:
            print("\nShutdown - Final memory state:")
            monitor.log_memory("Final:")
        if monitor_task:
            monitor_task.cancel()
        raise


asyncio.run(worker())
