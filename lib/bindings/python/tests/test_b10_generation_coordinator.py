# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from urllib.error import URLError
from urllib.request import urlopen

import pytest

from dynamo._core import GenerationCoordinator
from dynamo.runtime import DistributedRuntime

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.mark.asyncio
@pytest.mark.parametrize("client_count", [0, 1, 2])
async def test_coordinator_accepts_clients_and_endpoint_strings(
    temp_file_store, client_count
):
    runtime = DistributedRuntime(asyncio.get_running_loop(), "file", "tcp")
    try:
        clients = ["test.worker.generate", "test.router.generate"]
        for index in range(client_count):
            clients[index] = await runtime.endpoint(clients[index]).client()
        coordinator = GenerationCoordinator(
            primary_worker_client=clients[0],
            primary_router_client=clients[1],
            model_name="test",
            kv_block_size=16,
            runtime=runtime,
        )
        url = await asyncio.wait_for(coordinator.serve(host="127.0.0.1", port=0), 10)

        def health():
            with urlopen(
                url.replace("/v1/coordinate", "/health"), timeout=5
            ) as response:
                return response.status

        assert await asyncio.to_thread(health) == 200
        # The runtime, not the Python handle, owns the listener lifetime.
        del coordinator
        assert await asyncio.to_thread(health) == 200
        runtime.shutdown()
        async with asyncio.timeout(10):
            while True:
                try:
                    await asyncio.to_thread(health)
                except URLError:
                    break
                await asyncio.sleep(0.01)
    finally:
        runtime.shutdown()


@pytest.mark.parametrize("kwargs", [{}, {"runtime": None}])
def test_coordinator_requires_runtime(kwargs):
    with pytest.raises(TypeError, match="runtime"):
        GenerationCoordinator(
            primary_worker_client="test.worker.generate",
            primary_router_client="test.router.generate",
            model_name="test",
            kv_block_size=16,
            disagg_request_id_machine_id=1,
            **kwargs,
        )
