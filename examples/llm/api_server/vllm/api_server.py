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

from api_server.base_api_server import BaseApiServer
from processor.vllm import vLLMProcessor
from worker.vllm import vLLMWorker

from dynamo.sdk import depends, service
from dynamo.sdk.lib.image import DYNAMO_IMAGE


# todo this should be called ApiServer
@service(
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
    image=DYNAMO_IMAGE,
)
class vLLMApiServer(BaseApiServer):
    worker = depends(vLLMWorker)
    processor = depends(vLLMProcessor)

    def __init__(self):
        super().__init__(config_name="vLLMApiServer")
