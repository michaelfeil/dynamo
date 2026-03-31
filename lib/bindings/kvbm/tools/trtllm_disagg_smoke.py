#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace
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


def run_smoke() -> dict[str, Any]:
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

    class _Dist:
        rank = 0
        tp_size = 2

        def broadcast(self, value: Any, root: int) -> Any:
            del root
            return value if value is not None else "broadcast-value"

        def allgather(self, value: Any) -> list[Any]:
            return [value, value]

        def pp_allgather(self, value: Any) -> list[Any]:
            return [value, value]

        def tp_allgather(self, value: Any) -> list[Any]:
            return [list(value), list(value)]

    original_transfer_worker = transceiver_mod.TransferWorker
    original_current_device = transceiver_mod.torch.cuda.current_device
    transceiver_mod.TransferWorker = _FakeTransferWorker
    transceiver_mod.torch.cuda.current_device = lambda: 0
    try:
        manager = KvbmKVCacheManager(
            tokens_per_block=4,
            dtype="float16",
            head_dim=16,
            pp_layers=[4, 5],
            total_num_kv_heads_per_layer=[8, 8, 8, 8, 6, 6],
            max_seq_len=128,
            num_blocks=12,
            primary_pool=_FakeTensor(
                shape=(12, 2, 2, 4, 6, 16),
                strides=(1536, 768, 384, 96, 16, 1),
                ptr=12288,
            ),
            device_id=2,
            world_size=4,
            tp_size=2,
            tp_rank=1,
            pp_size=2,
            pp_rank=1,
        )
        manager.add_dummy_requests([901], token_nums=[12])
        page_table = build_page_table_from_manager(manager)

        transceiver = transceiver_mod.PyNativeCacheTransceiver(
            manager.mapping,
            _Dist(),
            manager,
            None,
            SimpleNamespace(
                kv_transfer_timeout_ms=7000,
                kv_transfer_sender_future_timeout_ms=3000,
            ),
        )
        worker = _FakeTransferWorker.instances[-1]
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

        transceiver.request_and_receive_async(request)
        transceiver.check_gen_transfer_status(1)
        generation_state = _state_name(request.state)

        waiting_request = SimpleNamespace(
            request_id=902,
            py_request_id=902,
            py_disaggregated_params=None,
            prompt_len=4,
            state=None,
        )
        transceiver.prepare_context_requests([waiting_request])

        result = {
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
                "sent_block_ids": worker.tx_sessions[0]
                .sent_slices[0]
                .block_ids_per_layer_groups,
            },
            "generation": {
                "state": generation_state,
                "complete": transceiver.check_gen_transfer_complete(),
                "received_block_ids": worker.rx_sessions[0]
                .received_slices[0]
                .block_ids_per_layer_groups,
            },
            "waiting_request_state": _state_name(waiting_request.state),
            "disagg_params": transceiver.get_disaggregated_params(),
            "rank_info_calls": worker.rank_info_calls,
        }
        transceiver.shutdown()
        result["shutdown_calls"] = worker.shutdown_calls
        return result
    finally:
        transceiver_mod.TransferWorker = original_transfer_worker
        transceiver_mod.torch.cuda.current_device = original_current_device


def main() -> int:
    print(json.dumps(run_smoke(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
