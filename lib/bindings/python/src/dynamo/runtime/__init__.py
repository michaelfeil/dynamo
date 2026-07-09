# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import signal
import threading
import warnings
from functools import wraps
from typing import Any, AsyncGenerator, Callable, Literal, Optional, Type, Union

from pydantic import BaseModel, ValidationError

# List all the classes in the _core module for re-export
# import * causes "unable to detect undefined names"
from dynamo._core import Client as Client
from dynamo._core import Context as Context
from dynamo._core import DistributedRuntime as DistributedRuntime
from dynamo._core import Endpoint as Endpoint
from dynamo._core import unregister_model as unregister_model

logger = logging.getLogger(__name__)
B10_SHUTDOWN_INITIATED = threading.Event()
ENDPOINT_PHASE_SHUTDOWN_DRAIN_SECS = 1
ENDPOINT_SHUTDOWN_DRAIN_SECS = 5
_SHUTDOWN_TASK: Optional[asyncio.Task] = None
ShutdownPhase = Literal["early", "default"]
_ENDPOINTS_TO_SHUTDOWN: dict[ShutdownPhase, list[Endpoint]] = {
    "early": [],
    "default": [],
}


def _b10_shutdown_handler(runtime: DistributedRuntime):
    logger.info(
        "Shutdown signal received, initiating graceful shutdown...",
        extra={"unified_model_logs": True},
    )
    if B10_SHUTDOWN_INITIATED.is_set():
        return

    B10_SHUTDOWN_INITIATED.set()

    if not any(_ENDPOINTS_TO_SHUTDOWN.values()):
        runtime.shutdown()
        return

    global _SHUTDOWN_TASK
    loop = asyncio.get_running_loop()
    _SHUTDOWN_TASK = loop.create_task(_shutdown_registered_endpoints(runtime))


async def _shutdown_registered_endpoints(runtime: DistributedRuntime):
    try:
        early_endpoints = list(_ENDPOINTS_TO_SHUTDOWN["early"])
        default_endpoints = list(_ENDPOINTS_TO_SHUTDOWN["default"])

        await _shutdown_endpoint_group(early_endpoints)
        if early_endpoints and default_endpoints:
            await asyncio.sleep(ENDPOINT_PHASE_SHUTDOWN_DRAIN_SECS)
        await _shutdown_endpoint_group(default_endpoints)

        await asyncio.sleep(ENDPOINT_SHUTDOWN_DRAIN_SECS)
    except Exception:
        logger.exception("Failed during endpoint shutdown")
    finally:
        try:
            runtime.shutdown()
        except Exception:
            logger.exception("Failed to shutdown runtime")


async def _shutdown_endpoint_group(endpoints: list[Endpoint]):
    for endpoint in endpoints:
        try:
            await unregister_model(endpoint)
        except Exception:
            logger.exception("Failed to unregister model during shutdown")

        try:
            await endpoint.unregister_endpoint_instance()
        except Exception:
            logger.exception("Failed to unregister endpoint during shutdown")


def register_endpoint_for_shutdown(
    endpoint: Endpoint, phase: ShutdownPhase = "default"
):
    registry = _ENDPOINTS_TO_SHUTDOWN[phase]
    if not any(registered is endpoint for registered in registry):
        registry.append(endpoint)


def unregister_endpoint_for_shutdown(endpoint: Endpoint):
    for registry in _ENDPOINTS_TO_SHUTDOWN.values():
        registry[:] = [
            registered for registered in registry if registered is not endpoint
        ]


def b10_register_shutdown_signals(runtime: DistributedRuntime):
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: _b10_shutdown_handler(runtime))


def dynamo_worker(enable_nats: Optional[bool] = None, register_shutdown: bool = False):
    """
    Decorator that creates a DistributedRuntime and passes it to the worker function.

    Args:
        enable_nats: Deprecated. NATS enablement is now determined automatically
            from the event-plane configuration. This parameter is accepted for
            backwards compatibility but will be removed in a future release.
        register_shutdown: Whether to register signal handlers for graceful shutdown.
    """
    if enable_nats is not None:
        warnings.warn(
            "The 'enable_nats' parameter is deprecated and will be removed in a "
            "future release. NATS enablement is now determined automatically from "
            "the event-plane configuration.",
            DeprecationWarning,
            stacklevel=2,
        )

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            loop = asyncio.get_running_loop()
            request_plane = os.environ.get("DYN_REQUEST_PLANE", "tcp")
            discovery_backend = os.environ.get("DYN_DISCOVERY_BACKEND", "etcd")
            runtime = DistributedRuntime(loop, discovery_backend, request_plane)

            if register_shutdown:
                b10_register_shutdown_signals(runtime)

            await func(runtime, *args, **kwargs)

        return wrapper

    return decorator


def dynamo_endpoint(
    request_model: Union[Type[BaseModel], Type[Any]], response_model: Type[BaseModel]
) -> Callable:
    def decorator(
        func: Callable[..., AsyncGenerator[Any, None]],
    ) -> Callable[..., AsyncGenerator[Any, None]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> AsyncGenerator[Any, None]:
            # Validate the request
            try:
                args_list = list(args)
                if len(args) in [1, 2] and issubclass(request_model, BaseModel):
                    if isinstance(args[-1], str):
                        args_list[-1] = request_model.parse_raw(args[-1])
                    elif isinstance(args[-1], dict):
                        args_list[-1] = request_model.parse_obj(args[-1])
                    else:
                        raise ValueError(f"Invalid request: {args[-1]}")
            except ValidationError as e:
                raise ValueError(f"Invalid request: {e}")

            # Wrap the async generator
            async for item in func(*args_list, **kwargs):
                # Validate the response
                # TODO: Validate the response
                try:
                    yield item
                except ValidationError as e:
                    raise ValueError(f"Invalid response: {e}")

        return wrapper

    return decorator
