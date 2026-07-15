# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib

import pytest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
]


async def _generate(request, context):
    assert dict(context.metadata.items()) == request.get("expected_metadata", {})
    if request["kind"] == "error":
        raise ValueError("direct adapter test error")
    if request["kind"] == "malformed":
        yield object()
        return
    if request["kind"] == "explicit-none":
        yield {
            "_dynamo_annotated": True,
            "data": {"nested": {"value": 42}},
            "id": None,
            "event": None,
            "comment": None,
            "error": None,
        }
        return
    if request["kind"] == "reused-mutable":
        shared = {"sequence": 0}
        for sequence in range(64):
            shared["sequence"] = sequence
            yield shared
        return

    yield request["payload"]
    yield {
        "_dynamo_annotated": True,
        "data": {"annotated": request["payload"]},
        "id": "chunk-2",
        "event": "delta",
        "comment": ["direct", "python"],
    }


@pytest.fixture
async def request_plane_client(runtime):
    endpoint = runtime.endpoint("direct-python-msgpack.backend.generate")
    health_payload = {
        "kind": "normal",
        "payload": {"health": True},
        "expected_metadata": {},
    }
    server_task = asyncio.ensure_future(
        endpoint.serve_endpoint(_generate, health_check_payload=health_payload)
    )
    client = await endpoint.client()
    try:
        await client.wait_for_instances()
        yield client
    finally:
        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server_task


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("request_plane", ["tcp", "nats"], indirect=True)
async def test_client_instances_snapshot(request_plane, request_plane_client):
    """Client.instances() exposes the endpoint's instances with transport
    details on each request plane: "tcp" carries a "host:port/..." address,
    "nats" carries a "nats_tcp" subject address."""
    expected_kind = "nats_tcp" if request_plane == "nats" else "tcp"

    instances = request_plane_client.instances()
    assert {i.instance_id for i in instances} == set(
        request_plane_client.instance_ids()
    )
    for instance in instances:
        assert instance.namespace == "direct-python-msgpack"
        assert instance.component == "backend"
        assert instance.endpoint == "generate"
        assert instance.device_type in (None, "cpu", "cuda")
        assert instance.transport.kind == expected_kind
        assert instance.transport.address  # populated on both planes
