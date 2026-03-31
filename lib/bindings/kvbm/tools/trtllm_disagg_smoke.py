#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace
import traceback
from typing import Any


class _FakeTensor:
    def __init__(self, shape: tuple[int, ...], strides: tuple[int, ...], ptr: int) -> None:
        self.shape = shape
        self._strides = strides
        self._ptr = ptr

    def stride(self, dim: int) -> int:
        return self._strides[dim]

    def element_size(self) -> int:
        return 2

    def data_ptr(self) -> int:
        return self._ptr


def _state_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name
    return str(value)


def _make_dist(mapping: Any) -> Any:
    class _Dist:
        def __init__(self, mapping: Any) -> None:
            self.rank = int(mapping.rank)
            self.world_size = int(mapping.world_size)
            self.tp_size = int(mapping.tp_size)
            self.pp_size = int(mapping.pp_size)

        def broadcast(self, value: Any, root: int) -> Any:
            del root
            return value if value is not None else "ctx-endpoint"

        def allgather(self, value: Any) -> list[Any]:
            return [value for _ in range(self.world_size)]

        def pp_allgather(self, value: Any) -> list[Any]:
            return [value for _ in range(self.pp_size)]

        def tp_allgather(self, value: Any) -> list[Any]:
            return [list(value) for _ in range(self.tp_size)]

    return _Dist(mapping)


def _build_generation_request(*, context_request: Any, module: Any) -> Any:
    params = module.DisaggregatedParams(
        request_type="generation_only",
        ctx_request_id=context_request.context_phase_params.req_id,
        ctx_dp_rank=context_request.context_phase_params.ctx_dp_rank,
        ctx_info_endpoint=context_request.context_phase_params.disagg_info_endpoint,
    )
    return SimpleNamespace(
        request_id=context_request.request_id,
        py_request_id=context_request.py_request_id,
        py_disaggregated_params=params,
        prompt_len=context_request.prompt_len,
        state=None,
    )


def _manager_kwargs(*, use_fake_transfer_worker: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "tokens_per_block": 4,
        "dtype": "float16",
        "head_dim": 16,
        "pp_layers": [4, 5],
        "total_num_kv_heads_per_layer": [8, 8, 8, 8, 6, 6],
        "max_seq_len": 128,
        "num_blocks": 12,
        "primary_pool": _FakeTensor(
            shape=(12, 2, 2, 4, 6, 16),
            strides=(1536, 768, 384, 96, 16, 1),
            ptr=12288,
        ),
        "world_size": 4,
        "tp_size": 2,
        "pp_size": 2,
    }
    if use_fake_transfer_worker:
        kwargs.update(device_id=2, tp_rank=1, pp_rank=1)
    else:
        # Keep the bounded native probe on a leader rank so it can expose a
        # real rank-info server endpoint instead of a fake placeholder.
        kwargs.update(device_id=0, tp_rank=0, pp_rank=0)
    return kwargs


def _run_smoke_once(*, use_fake_transfer_worker: bool = True) -> dict[str, Any]:
    import tensorrt_llm.disaggregated_params as disaggregated_params_mod
    from tensorrt_llm._torch.disaggregation.base.transfer import SessionState, SessionStatus
    from tensorrt_llm._torch.disaggregation.resource.kv_extractor import (
        build_page_table_from_manager,
    )
    import tensorrt_llm._torch.disaggregation.native.py_cache_transceiver as transceiver_mod

    from kvbm.trtllm_integration import KvbmKVCacheManager

    class _FakeSession:
        def __init__(self, req: Any) -> None:
            self.unique_rid = req.request_id
            self.state = SessionState(status=SessionStatus.READY, finished_tasks=[])
            self.sent_slices: list[Any] = []
            self.received_slices: list[Any] = []

        def send(self, slice_: Any) -> int:
            self.sent_slices.append(slice_)
            self.state = SessionState(status=SessionStatus.TRANSFERRED, finished_tasks=[11])
            return 11

        def receive(self, slice_: Any) -> int:
            self.received_slices.append(slice_)
            self.state = SessionState(status=SessionStatus.TRANSFERRED, finished_tasks=[12])
            return 12

        def wait_complete(
            self,
            task_id: int,
            wait_aux: bool = False,
            timeout_ms: int | None = None,
        ) -> bool:
            del task_id, wait_aux, timeout_ms
            return True

        def close(self) -> None:
            return None

    class _FakeTransferWorker:
        instances: list[Any] = []

        def __init__(
            self,
            *,
            kv_cache_manager: Any,
            mapping: Any,
            device_id: int,
            instance_name: str,
            aux_buffer: Any,
        ) -> None:
            self.kv_cache_manager = kv_cache_manager
            self.mapping = mapping
            self.device_id = device_id
            self.instance_name = instance_name
            self.aux_buffer = aux_buffer
            self.page_table = build_page_table_from_manager(kv_cache_manager)
            self._rank_info = SimpleNamespace(page_table=self.page_table)
            self._rank_info_server = SimpleNamespace(endpoint="ctx-endpoint")
            self._sender = SimpleNamespace(endpoint=f"sender-{device_id}")
            self.rank_info_calls: list[dict[str, Any]] = []
            self.tx_sessions: list[_FakeSession] = []
            self.rx_sessions: list[_FakeSession] = []
            self.cleared: list[int] = []
            self.shutdown_calls = 0
            type(self).instances.append(self)

        def populate_instance_and_rank_info(
            self, *, endpoints: Any, layer_num_per_pp: Any
        ) -> None:
            self.rank_info_calls.append(
                {
                    "endpoints": list(endpoints),
                    "layer_num_per_pp": list(layer_num_per_pp),
                }
            )

        def create_tx_session(self, req: Any) -> _FakeSession:
            session = _FakeSession(req)
            self.tx_sessions.append(session)
            return session

        def create_rx_session(self, req: Any) -> _FakeSession:
            session = _FakeSession(req)
            self.rx_sessions.append(session)
            return session

        def clear_session(self, session: _FakeSession) -> None:
            self.cleared.append(session.unique_rid)

        def has_all_peer_req_infos_for_send(self, rid: int) -> bool:
            del rid
            return True

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    original_transfer_worker = transceiver_mod.TransferWorker
    original_current_device = transceiver_mod.torch.cuda.current_device

    transfer_worker_type = _FakeTransferWorker
    if not use_fake_transfer_worker:
        class _RecordingTransferWorker(original_transfer_worker):
            instances: list[Any] = []

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                type(self).instances.append(self)

        transfer_worker_type = _RecordingTransferWorker

    transceiver_mod.TransferWorker = transfer_worker_type
    transceiver_mod.torch.cuda.current_device = lambda: 0
    try:
        manager = KvbmKVCacheManager(**_manager_kwargs(use_fake_transfer_worker=use_fake_transfer_worker))
        manager.add_dummy_requests([901], token_nums=[12])
        page_table = build_page_table_from_manager(manager)
        dist = _make_dist(manager.mapping)

        try:
            transceiver = transceiver_mod.PyNativeCacheTransceiver(
                manager.mapping,
                dist,
                manager,
                None,
                SimpleNamespace(
                    kv_transfer_timeout_ms=7000,
                    kv_transfer_sender_future_timeout_ms=3000,
                ),
            )
        except Exception as exc:
            result: dict[str, Any] = {
                "mode": "fake-transfer-worker" if use_fake_transfer_worker else "real-transfer-worker",
                "status": "error",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc().splitlines(),
                "page_table": {
                    "tokens_per_block": page_table.tokens_per_block,
                    "pool_slots": page_table.pool_groups[0].pools[0].num_slots,
                    "slot_bytes": page_table.pool_groups[0].pools[0].slot_bytes,
                    "buffer_entries": page_table.layer_groups[0]
                    .pool_views[0]
                    .buffer_entries.tolist(),
                },
                "distributed": {
                    "dist_rank": dist.rank,
                    "mapping_rank": manager.mapping.rank,
                },
            }
            instances = getattr(transfer_worker_type, "instances", [])
            if instances:
                worker = instances[-1]
                rank_info_server = getattr(worker, "_rank_info_server", None)
                sender = getattr(worker, "_sender", None)
                result["transfer_worker"] = {
                    "type": type(worker).__name__,
                    "rank_info_server_is_none": rank_info_server is None,
                    "rank_info_server_endpoint": getattr(rank_info_server, "endpoint", None),
                    "sender_endpoint": getattr(sender, "endpoint", None),
                }
            return result
        worker = transfer_worker_type.instances[-1]
        request = SimpleNamespace(
            request_id=901,
            py_request_id=901,
            py_disaggregated_params=None,
            prompt_len=12,
            state=None,
        )
        transceiver.respond_and_send_async(request)
        completed_ctx, failed_ctx = transceiver.check_context_transfer_status(
            1, mark_complete=True
        )
        context_state = _state_name(request.state)
        tx_sessions = getattr(worker, "tx_sessions", [])
        rank_info_calls = getattr(
            worker,
            "rank_info_calls",
            [
                {
                    "endpoints": dist.allgather(getattr(worker._sender, "endpoint", None)),
                    "layer_num_per_pp": dist.pp_allgather(len(manager.pp_layers)),
                }
            ],
        )

        if not use_fake_transfer_worker:
            unique_endpoints = sorted(set(rank_info_calls[0]["endpoints"]))
            if len(unique_endpoints) == 1 and manager.world_size > 1:
                transceiver.shutdown()
                return {
                    "mode": "real-transfer-worker",
                    "status": "blocked",
                    "reason": (
                        "single-process smoke cannot emulate distinct remote TRT-LLM peers; "
                        "all gathered sender endpoints collapse to the same live worker"
                    ),
                    "blocked_phase": "generation_transfer",
                    "native_observation": (
                        "installed TRT-LLM NIXL transport reaches real rank-info server bring-up "
                        "and context send startup, but generation receive would self-connect to "
                        "the local agent in this one-process harness"
                    ),
                    "page_table": {
                        "tokens_per_block": page_table.tokens_per_block,
                        "pool_slots": page_table.pool_groups[0].pools[0].num_slots,
                        "slot_bytes": page_table.pool_groups[0].pools[0].slot_bytes,
                        "buffer_entries": page_table.layer_groups[0]
                        .pool_views[0]
                        .buffer_entries.tolist(),
                    },
                    "context": {
                        "completed": completed_ctx,
                        "failed": failed_ctx,
                        "state": context_state,
                        "ctx_dp_rank": request.context_phase_params.ctx_dp_rank,
                        "ctx_info_endpoint": request.context_phase_params.disagg_info_endpoint,
                        "ctx_request_id": request.context_phase_params.req_id,
                        "sent_block_ids": (
                            tx_sessions[0].sent_slices[0].block_ids_per_layer_groups
                            if tx_sessions
                            else None
                        ),
                    },
                    "distributed": {
                        "dist_rank": dist.rank,
                        "mapping_rank": manager.mapping.rank,
                    },
                    "rank_info_calls": rank_info_calls,
                    "transfer_worker": {
                        "type": type(worker).__name__,
                        "sender_endpoint": getattr(worker._sender, "endpoint", None),
                        "rank_info_server_endpoint": getattr(
                            getattr(worker, "_rank_info_server", None), "endpoint", None
                        ),
                    },
                    "shutdown_calls": getattr(worker, "shutdown_calls", None),
                }

        generation_request = _build_generation_request(
            context_request=request,
            module=disaggregated_params_mod,
        )
        transceiver.request_and_receive_async(generation_request)
        transceiver.check_gen_transfer_status(1)
        generation_state = _state_name(generation_request.state)
        rx_sessions = getattr(worker, "rx_sessions", [])

        waiting_request = SimpleNamespace(
            request_id=902,
            py_request_id=902,
            py_disaggregated_params=None,
            prompt_len=4,
            state=None,
        )
        transceiver.prepare_context_requests([waiting_request])

        result = {
            "mode": "fake-transfer-worker" if use_fake_transfer_worker else "real-transfer-worker",
            "status": "ok",
            "page_table": {
                "tokens_per_block": page_table.tokens_per_block,
                "pool_slots": page_table.pool_groups[0].pools[0].num_slots,
                "slot_bytes": page_table.pool_groups[0].pools[0].slot_bytes,
                "buffer_entries": page_table.layer_groups[0]
                .pool_views[0]
                .buffer_entries.tolist(),
            },
            "context": {
                "completed": completed_ctx,
                "failed": failed_ctx,
                "state": context_state,
                "ctx_dp_rank": request.context_phase_params.ctx_dp_rank,
                "ctx_info_endpoint": request.context_phase_params.disagg_info_endpoint,
                "ctx_request_id": request.context_phase_params.req_id,
                "sent_block_ids": tx_sessions[0].sent_slices[0].block_ids_per_layer_groups,
            },
            "generation": {
                "state": generation_state,
                "complete": transceiver.check_gen_transfer_complete(),
                "ctx_request_id": generation_request.py_disaggregated_params.ctx_request_id,
                "ctx_info_endpoint": generation_request.py_disaggregated_params.ctx_info_endpoint,
                "received_block_ids": rx_sessions[0].received_slices[0].block_ids_per_layer_groups,
            },
            "waiting_request_state": _state_name(waiting_request.state),
            "disagg_params": transceiver.get_disaggregated_params(),
            "distributed": {
                "dist_rank": dist.rank,
                "mapping_rank": manager.mapping.rank,
            },
            "rank_info_calls": rank_info_calls,
        }
        transceiver.shutdown()
        result["shutdown_calls"] = worker.shutdown_calls
        return result
    finally:
        transceiver_mod.TransferWorker = original_transfer_worker
        transceiver_mod.torch.cuda.current_device = original_current_device


def _extract_json(stdout: str) -> dict[str, Any] | None:
    for index in range(len(stdout) - 1, -1, -1):
        if stdout[index] != "{":
            continue
        try:
            return json.loads(stdout[index:])
        except json.JSONDecodeError:
            continue
    return None


def _run_real_transfer_worker_subprocess() -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--real-transfer-worker-internal",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    parsed = _extract_json(completed.stdout)
    if parsed is not None:
        return parsed

    signal_name = None
    if completed.returncode < 0:
        try:
            signal_name = signal.Signals(-completed.returncode).name
        except ValueError:
            signal_name = f"SIG{-completed.returncode}"

    return {
        "mode": "real-transfer-worker",
        "status": "error",
        "phase": "native-transfer-worker-startup",
        "reason": "native TRT-LLM worker exited before emitting structured JSON",
        "subprocess_returncode": completed.returncode,
        "signal": signal_name,
        "stdout_tail": completed.stdout.splitlines()[-40:],
        "stderr_tail": completed.stderr.splitlines()[-40:],
    }


def run_smoke(*, use_fake_transfer_worker: bool = True) -> dict[str, Any]:
    if use_fake_transfer_worker:
        return _run_smoke_once(use_fake_transfer_worker=True)
    return _run_real_transfer_worker_subprocess()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-transfer-worker",
        action="store_true",
        help="Use the live TRT-LLM TransferWorker instead of the fake repo-local worker.",
    )
    parser.add_argument(
        "--real-transfer-worker-internal",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit non-zero when the smoke result reports status=error.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.real_transfer_worker_internal:
        result = _run_smoke_once(use_fake_transfer_worker=False)
    else:
        result = run_smoke(use_fake_transfer_worker=not args.real_transfer_worker)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.fail_on_error and result.get("status") == "error":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
