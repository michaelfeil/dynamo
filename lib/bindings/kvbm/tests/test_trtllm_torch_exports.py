# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
import sys
import unittest


PYTHON_ROOT = Path(__file__).resolve().parents[1] / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


class TrtllmTorchExportSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
            from kvbm import _core
            from kvbm.trtllm_integration import KvbmKVCacheManager
        except ImportError as exc:
            raise unittest.SkipTest(f"real kvbm/torch export smoke unavailable: {exc}") from exc

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is unavailable for the real torch export smoke")

        create_primary_pool = getattr(_core._trtllm_integration, "create_primary_pool", None)
        if not callable(create_primary_pool):
            raise unittest.SkipTest("real KVBM primary-pool exports are unavailable")

        cls.torch = torch
        cls.create_primary_pool = create_primary_pool
        cls.KvbmKVCacheManager = KvbmKVCacheManager
        cls.device_id = 0

    def test_raw_primary_pool_export_accepts_modern_torch_dlpack_call(self) -> None:
        primary_pool = self.create_primary_pool(
            num_blocks=2,
            num_layers=1,
            kv_factor=2,
            page_size=4,
            inner_dim=128,
            dtype="float16",
            device_id=self.device_id,
        )

        tensor = self.torch.utils.dlpack.from_dlpack(primary_pool)

        self.assertEqual(tuple(tensor.shape), (2, 1, 2, 4, 128))
        self.assertEqual(tuple(tensor.stride()), (1024, 1024, 512, 128, 1))
        self.assertEqual(tensor.device, self.torch.device("cuda", self.device_id))
        self.assertEqual(tensor.dtype, self.torch.float16)
        self.assertEqual(tuple(primary_pool.shape), (2, 1, 2, 4, 128))
        self.assertEqual(
            tuple(primary_pool.stride(dim) for dim in range(len(primary_pool.shape))),
            (1024, 1024, 512, 128, 1),
        )
        self.assertEqual(primary_pool.element_size(), 2)
        self.assertGreater(primary_pool.data_ptr(), 0)

    def test_standard_rank_local_exports_round_trip_through_torch(self) -> None:
        manager = self.KvbmKVCacheManager(
            tokens_per_block=4,
            dtype="float16",
            head_dim=16,
            pp_layers=[4, 5],
            total_num_kv_heads_per_layer=[8, 8, 8, 8, 6, 6],
            max_seq_len=128,
            num_blocks=12,
            device_id=self.device_id,
            world_size=4,
            tp_size=2,
            tp_rank=1,
            pp_size=2,
            pp_rank=1,
        )
        self.addCleanup(manager.shutdown)

        primary = manager.get_unique_primary_pool()
        layer_nhd = manager.get_buffers(4, "NHD")
        layer_hnd = manager.get_buffers(4, "HND")

        self.assertIsInstance(primary, self.torch.Tensor)
        self.assertEqual(primary.device, self.torch.device("cuda", self.device_id))
        self.assertEqual(primary.dtype, self.torch.float16)
        self.assertEqual(tuple(primary.shape), (12, 2, 2, 4, 3, 16))
        self.assertEqual(tuple(primary.stride()), (768, 384, 192, 48, 16, 1))

        self.assertEqual(tuple(layer_nhd.shape), (12, 2, 4, 3, 16))
        self.assertEqual(tuple(layer_nhd.stride()), (768, 192, 48, 16, 1))
        self.assertEqual(tuple(layer_hnd.shape), (12, 2, 3, 4, 16))
        self.assertEqual(tuple(layer_hnd.stride()), (768, 192, 16, 48, 1))

        primary.zero_()
        primary[0, 0, 0, 0, 0, 0] = 11
        primary[0, 0, 1, 0, 0, 0] = 17

        self.assertEqual(layer_nhd[0, 0, 0, 0, 0].item(), 11)
        self.assertEqual(layer_nhd[0, 1, 0, 0, 0].item(), 17)
        self.assertEqual(layer_hnd[0, 0, 0, 0, 0].item(), 11)
        self.assertEqual(layer_hnd[0, 1, 0, 0, 0].item(), 17)

    def test_mla_rank_local_exports_round_trip_through_torch(self) -> None:
        manager = self.KvbmKVCacheManager(
            tokens_per_block=4,
            dtype="float16",
            head_dim=576,
            pp_layers=[0, 1],
            total_num_kv_heads_per_layer=[1, 1, 1, 1],
            max_seq_len=64,
            num_blocks=10,
            device_id=self.device_id,
            world_size=4,
            tp_size=4,
            tp_rank=2,
            pp_size=1,
            pp_rank=0,
            cache_mode="mla",
        )
        self.addCleanup(manager.shutdown)

        primary = manager.get_unique_primary_pool()
        layer_nhd = manager.get_buffers(1, "NHD")
        layer_hnd = manager.get_buffers(1, "HND")

        self.assertEqual(primary.device, self.torch.device("cuda", self.device_id))
        self.assertEqual(primary.dtype, self.torch.float16)
        self.assertEqual(tuple(primary.shape), (10, 2, 1, 4, 1, 576))
        self.assertEqual(tuple(primary.stride()), (4608, 2304, 2304, 576, 576, 1))

        self.assertEqual(tuple(layer_nhd.shape), (10, 1, 4, 1, 576))
        self.assertEqual(tuple(layer_nhd.stride()), (4608, 2304, 576, 576, 1))
        self.assertEqual(tuple(layer_hnd.shape), (10, 1, 1, 4, 576))
        self.assertEqual(tuple(layer_hnd.stride()), (4608, 2304, 576, 576, 1))

        primary.zero_()
        primary[0, 1, 0, 0, 0, 0] = 29

        self.assertEqual(layer_nhd[0, 0, 0, 0, 0].item(), 29)
        self.assertEqual(layer_hnd[0, 0, 0, 0, 0].item(), 29)


if __name__ == "__main__":
    unittest.main()
