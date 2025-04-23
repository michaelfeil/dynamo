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

import logging
import subprocess
from abc import ABC
from pathlib import Path

from pydantic import BaseModel

from dynamo import sdk
from dynamo.sdk import async_on_shutdown
from dynamo.sdk.lib.config import ServiceConfig

logger = logging.getLogger(__name__)


class ApiServerConfig(BaseModel):
    """Configuration for the API server including model and HTTP server settings"""

    served_model_name: str
    endpoint: str
    port: int = 8080


# TODO: incorporate completions endpoint as well
class BaseApiServer(ABC):
    """Base class for LLM API servers that handles common HTTP server functionality"""

    def __init__(self, config_name="ApiServer"):
        """Initialize API server with HTTP server and model configuration"""
        self.process = None
        config = ServiceConfig.get_instance()
        self.config = ApiServerConfig(**config.get(config_name, {}))
        self.setup_model()
        self.start_http_server()

    def get_http_binary_path(self):
        """Find the HTTP binary path in SDK or fallback to 'http' command"""
        sdk_path = Path(sdk.__file__)
        binary_path = sdk_path.parent / "cli/bin/http"
        return str(binary_path) if binary_path.exists() else "http"

    def setup_model(self):
        """Configure the model for HTTP service using llmctl"""
        # Remove existing model if present
        subprocess.run(
            [
                "llmctl",
                "http",
                "remove",
                "chat-models",
                self.config.served_model_name,
            ],
            check=False,
        )

        # Add model configuration
        subprocess.run(
            [
                "llmctl",
                "http",
                "add",
                "chat-models",
                self.config.served_model_name,
                self.config.endpoint,
            ],
            check=False,
        )

    def start_http_server(self):
        """Start the HTTP server on the configured port"""
        logger.info("Starting HTTP server")
        http_binary = self.get_http_binary_path()

        self.process = subprocess.Popen(
            [http_binary, "-p", str(self.config.port)],
            stdout=None,
            stderr=None,
        )

    @async_on_shutdown
    async def cleanup(self):
        """Clean up resources before shutdown"""
        subprocess.run(
            [
                "llmctl",
                "http",
                "remove",
                "chat-models",
                self.config.served_model_name,
            ],
            check=False,
        )
