#  SPDX-FileCopyrightText: Copyright (c) 2020 Atalaya Tech. Inc
#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  Modifications Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES

from __future__ import annotations

import json
import logging
import typing as t
from http import HTTPStatus
from pathlib import Path

import typer
from bentoml._internal.cloud.base import Spinner
from bentoml._internal.cloud.deployment import Deployment, DeploymentConfigParameters
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.utils import add_experimental_docstring
from bentoml.exceptions import BentoMLException as DynamoException
from rich.console import Console
from simple_di import Provide, inject

from dynamo.sdk.lib.logging import configure_server_logging

from .utils import resolve_service_config

configure_server_logging()

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from bentoml._internal.cloud import BentoCloudClient as DynamoCloudClient

    TupleStrAny = tuple[str, ...]
else:
    TupleStrAny = tuple


def raise_deployment_config_error(err: DynamoException, action: str) -> t.NoReturn:
    if err.error_code == HTTPStatus.UNAUTHORIZED:
        raise DynamoException(
            f"{err}\n* Dynamo Cloud API token is required for authorization. Run `dynamo cloud login` command to login"
        ) from None
    raise DynamoException(
        f"Failed to {action} deployment due to invalid configuration: {err}"
    ) from None


def deploy(
    dynamo_pipeline: str = typer.Argument(..., help="The path to the Dynamo pipeline to deploy"),
    name: str = typer.Option(
        None, "--name", "-n", help="Deployment name"
    ),
    config_file: Path = typer.Option(
        None, "--config-file", "-f", help="Configuration file path"
    ),
    # TODO: Reinstate wait option once its implementation is ready
    # wait: bool = typer.Option(
    #     False, help="Wait for deployment to be ready", show_default=True
    # ),
    timeout: int = typer.Option(
        3600, help="Timeout for deployment to be ready in seconds", show_default=True
    ),
    ctx: typer.Context = typer.Context,
) -> None:
    """Create a deployment on Dynamo Cloud (shorthand for 'deployment create').

    Deploy a Dynamo pipeline to Dynamo Cloud using parameters or a config file.
    """
    config_file_io = config_file.open() if config_file else None
    create_deployment(
        dynamo_pipeline=dynamo_pipeline,
        name=name,
        config_file=config_file_io,
        wait=False,  # TODO: Reinstate wait option parameter once implementation is ready
        timeout=timeout,
        args=ctx.args,
    )


app = typer.Typer(
    name="deployment", 
    help="Deploy Dynamo applications to Kubernetes cluster",
    add_completion=True, 
    no_args_is_help=True
)


@app.command()
def create(
    dynamo_pipeline: str = typer.Argument(..., help="The path to the Dynamo pipeline to deploy"),
    name: str = typer.Option(
        None, "--name", "-n", help="Deployment name"
    ),
    config_file: Path = typer.Option(
        None, "--config-file", "-f", help="Configuration file path"
    ),
    # TODO: Reinstate wait option once its implementation is ready
    # wait: bool = typer.Option(
    #     False, help="Wait for deployment to be ready", show_default=True
    # ),
    timeout: int = typer.Option(
        3600, help="Timeout for deployment to be ready in seconds", show_default=True
    ),
    ctx: typer.Context = typer.Context,
) -> None:
    """Create a deployment on Dynamo Cloud.

    Deploy a Dynamo pipeline to Dynamo Cloud using parameters or a config file.
    Use the 'deploy' command as a shorthand for this command.
    """
    config_file_io = config_file.open() if config_file else None
    create_deployment(
        dynamo_pipeline=dynamo_pipeline,
        name=name,
        config_file=config_file_io,
        wait=False,  # TODO: Reinstate wait option parameter once implementation is ready
        timeout=timeout,
        args=ctx.args,
    )


@inject
def create_deployment(
    dynamo_pipeline: str | None = None,
    name: str | None = None,
    config_file: str | t.TextIO | None = None,
    wait: bool = False,  # Default to False until implementation is ready
    timeout: int = 3600,
    dev: bool = False,
    args: list[str] | None = None,
    _cloud_client: "DynamoCloudClient" = Provide[BentoMLContainer.bentocloud_client],
) -> Deployment:
    # Load config from file and serialize to env
    service_configs = resolve_service_config(config_file=config_file, args=args)
    print(f"service_configs: {service_configs}")
    env_dicts = []
    if service_configs:
        config_json = json.dumps(service_configs)
        logger.info(f"Deployment service configuration: {config_json}")
        env_dicts.append({"name": "DYN_DEPLOYMENT_CONFIG", "value": config_json})

    config_params = DeploymentConfigParameters(
        name=name,
        bento=dynamo_pipeline,  # API still expects 'bento' parameter
        envs=env_dicts,
        secrets=None,
        cli=True,
        dev=dev,
    )

    try:
        config_params.verify()
    except DynamoException as e:
        raise_deployment_config_error(e, "create")

    console = Console(highlight=False)
    with Spinner(console=console) as spinner:
        spinner.update("Creating deployment on Dynamo Cloud")
        deployment = _cloud_client.deployment.create(
            deployment_config_params=config_params
        )
        spinner.log(
            f':white_check_mark: Created deployment "{deployment.name}" in cluster "{deployment.cluster}"'
        )
        spinner.log(f":laptop_computer: View Dashboard: {deployment.admin_console}")
        
        # TODO: Reinstate and implement wait functionality in the future
        if wait:
            # This code is currently inactive since wait defaults to False
            spinner.update(
                "[bold blue]Waiting for deployment to be ready...[/]",
            )
            retcode = deployment.wait_until_ready(timeout=timeout, spinner=spinner)
            if retcode != 0:
                raise SystemExit(retcode)
        
        return deployment
