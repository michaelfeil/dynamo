# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from enum import Enum

from common.logging import check_required_workers

from dynamo.sdk import dynamo_context


class RequestType(Enum):
    CHAT = "chat"
    COMPLETION = "completion"


class BaseProcessor(ABC):
    """
    Base class for all processors.
    """

    worker_class = None
    router_class = None

    def __init__(self):
        self.min_workers = 1
        self.runtime = dynamo_context["runtime"]
        self.worker_client = None
        self.router_client = None

    @abstractmethod
    async def async_init(self, router_mode: str):
        """Initialize async components including clients"""

        # Setup worker client
        worker_ns, worker_name = self.worker_class.dynamo_address()
        self.worker_client = (
            await self.runtime.namespace(worker_ns)
            .component(worker_name)
            .endpoint("generate")
            .client()
        )

        # Setup router client if needed
        if router_mode == "kv":
            router_ns, router_name = self.router_class.dynamo_address()
            self.router_client = (
                await self.runtime.namespace(router_ns)
                .component(router_name)
                .endpoint("generate")
                .client()
            )

        # Wait for workers to be ready
        await check_required_workers(self.worker_client, self.min_workers)

    @abstractmethod
    def generate_chat(self) -> str:
        """Get the router mode from configuration"""
        pass
