# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
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
    coordinator = None
    try:
        clients = ["test.worker.generate", "test.router.generate"]
        for index in range(client_count):
            clients[index] = await runtime.endpoint(clients[index]).client()
        coordinator = GenerationCoordinator(
            primary_worker_client=clients[0],
            primary_router_client=clients[1],
            model_name="test",
            kv_block_size=16,
            **(
                {"runtime": runtime}
                if client_count < 2
                else {"disagg_request_id_machine_id": 1}
            ),
        )
        await asyncio.wait_for(
            asyncio.gather(coordinator.start(), coordinator.start()), 10
        )
        url = await coordinator.serve(host="127.0.0.1", port=0)

        def health():
            with urlopen(
                url.replace("/v1/coordinate", "/health"), timeout=5
            ) as response:
                return response.status

        assert await asyncio.to_thread(health) == 200
    finally:
        if coordinator is not None:
            await coordinator.shutdown()
        runtime.shutdown()


def test_coordinator_endpoint_strings_require_runtime():
    with pytest.raises(ValueError, match="runtime is required for endpoint strings"):
        GenerationCoordinator(
            primary_worker_client="test.worker.generate",
            primary_router_client="test.router.generate",
            model_name="test",
            kv_block_size=16,
            disagg_request_id_machine_id=1,
        )
