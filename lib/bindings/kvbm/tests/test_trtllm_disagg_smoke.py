# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest


PYTHON_ROOT = Path(__file__).resolve().parents[1] / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


from lib.bindings.kvbm.tools import trtllm_disagg_smoke


class TrtllmDisaggSmokeHelpersTest(unittest.TestCase):
    def test_make_dist_tracks_mapping_identity(self) -> None:
        mapping = types.SimpleNamespace(rank=3, world_size=4, tp_size=2, pp_size=2)

        dist = trtllm_disagg_smoke._make_dist(mapping)

        self.assertEqual(dist.rank, 3)
        self.assertEqual(dist.allgather("sender"), ["sender"] * 4)
        self.assertEqual(dist.pp_allgather(2), [2, 2])
        self.assertEqual(dist.tp_allgather([0, 1]), [[0, 1], [0, 1]])
        self.assertEqual(dist.broadcast(None, 0), "ctx-endpoint")

    def test_generation_request_uses_context_phase_params(self) -> None:
        class _DisaggregatedParams:
            def __init__(self, **kwargs) -> None:
                self.__dict__.update(kwargs)

        context_request = types.SimpleNamespace(
            request_id=901,
            py_request_id=901,
            prompt_len=12,
            context_phase_params=types.SimpleNamespace(
                req_id=777,
                ctx_dp_rank=5,
                disagg_info_endpoint="tcp://ctx-info:1234",
            ),
        )

        generation_request = trtllm_disagg_smoke._build_generation_request(
            context_request=context_request,
            module=types.SimpleNamespace(DisaggregatedParams=_DisaggregatedParams),
        )

        self.assertEqual(generation_request.request_id, 901)
        self.assertEqual(generation_request.prompt_len, 12)
        self.assertEqual(generation_request.py_disaggregated_params.request_type, "generation_only")
        self.assertEqual(generation_request.py_disaggregated_params.ctx_request_id, 777)
        self.assertEqual(generation_request.py_disaggregated_params.ctx_dp_rank, 5)
        self.assertEqual(
            generation_request.py_disaggregated_params.ctx_info_endpoint,
            "tcp://ctx-info:1234",
        )

    def test_extract_json_ignores_prefix_output(self) -> None:
        parsed = trtllm_disagg_smoke._extract_json(
            "warning line\n{\n  \"status\": \"ok\",\n  \"mode\": \"fake-transfer-worker\"\n}\n"
        )

        self.assertEqual(parsed, {"status": "ok", "mode": "fake-transfer-worker"})


if __name__ == "__main__":
    unittest.main()
