# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import dynamo.runtime as runtime_mod
from dynamo.runtime import register_endpoint_for_shutdown

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


def test_b10_shutdown_handler_without_endpoint_shutdowns_immediately():
    runtime_mod.B10_SHUTDOWN_INITIATED.clear()
    runtime_mod._ENDPOINTS_TO_SHUTDOWN["early"].clear()
    runtime_mod._ENDPOINTS_TO_SHUTDOWN["default"].clear()
    runtime = MagicMock()

    runtime_mod._b10_shutdown_handler(runtime)

    runtime.shutdown.assert_called_once_with()


def test_register_endpoint_for_shutdown_dedupes_by_identity():
    runtime_mod._ENDPOINTS_TO_SHUTDOWN["early"].clear()
    runtime_mod._ENDPOINTS_TO_SHUTDOWN["default"].clear()
    endpoint = MagicMock()

    register_endpoint_for_shutdown(endpoint)
    register_endpoint_for_shutdown(endpoint)

    assert runtime_mod._ENDPOINTS_TO_SHUTDOWN["default"] == [endpoint]


def test_b10_shutdown_handler_unregisters_endpoint_before_runtime_shutdown(monkeypatch):
    async def run_test():
        runtime_mod.B10_SHUTDOWN_INITIATED.clear()
        runtime_mod._ENDPOINTS_TO_SHUTDOWN["early"].clear()
        runtime_mod._ENDPOINTS_TO_SHUTDOWN["default"].clear()
        monkeypatch.setattr(runtime_mod, "ENDPOINT_SHUTDOWN_DRAIN_SECS", 0)

        events = []
        runtime = MagicMock()
        runtime.shutdown.side_effect = lambda: events.append("runtime_shutdown")
        endpoint = MagicMock()
        endpoint.unregister_endpoint_instance = AsyncMock(
            side_effect=lambda: events.append("unregister_endpoint")
        )

        async def unregister_model(endpoint_arg):
            assert endpoint_arg is endpoint
            events.append("unregister_model")

        monkeypatch.setattr(runtime_mod, "unregister_model", unregister_model)
        register_endpoint_for_shutdown(endpoint)

        runtime_mod._b10_shutdown_handler(runtime)
        await asyncio.wait_for(runtime_mod._SHUTDOWN_TASK, timeout=1)

        assert events == [
            "unregister_model",
            "unregister_endpoint",
            "runtime_shutdown",
        ]

    asyncio.run(run_test())


@pytest.mark.asyncio
async def test_b10_shutdown_started_resolves_when_shutdown_starts(temp_file_store):
    runtime = runtime_mod.DistributedRuntime(asyncio.get_running_loop(), "file", "tcp")
    started = asyncio.ensure_future(runtime.b10_shutdown_started())
    await asyncio.sleep(0.1)
    assert not started.done()

    runtime.shutdown()

    await asyncio.wait_for(started, timeout=5)
