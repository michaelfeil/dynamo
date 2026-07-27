# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import json
import threading

import pytest

from dynamo.llm import RadixTree

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


@pytest.mark.timeout(5)  # Expected: ~1s, timeout set to 5x for safety
def test_radix_tree_binding():
    """Test RadixTree binding directly with store event and find matches"""
    import json

    # Create RadixTree instance
    radix_tree = RadixTree()

    # Create a store event with parent_hash=None, block_hash=0
    # Following the KvCacheEvent format from the Rust protocols
    store_event = {
        "event_id": 1,
        "data": {
            "stored": {
                "parent_hash": None,
                "blocks": [
                    {
                        "block_hash": 0,
                        "tokens_hash": 0,  # Using 0 for both hashes to match tokens [0]
                    }
                ],
            }
        },
    }

    # Convert to JSON bytes
    event_bytes = json.dumps(store_event).encode("utf-8")

    # Apply the event to worker_id 0
    worker_id = 0
    radix_tree.apply_event(worker_id, event_bytes)

    # Find matches for tokens [0]
    # The sequence parameter expects token hashes, so we use [0] to match tokens_hash=0
    overlap_scores = radix_tree.find_matches([0])

    # Verify the results
    # Note: scores is now Dict[(worker_id, dp_rank), score]
    assert overlap_scores.scores is not None
    assert (
        len(overlap_scores.scores) == 1
    ), f"Expected 1 worker in scores, got {len(overlap_scores.scores)}"
    worker_key = (worker_id, 0)  # (worker_id, dp_rank)
    assert (
        worker_key in overlap_scores.scores
    ), f"Worker {worker_key} not found in scores"
    assert (
        overlap_scores.scores[worker_key] == 1
    ), f"Expected score 1 for worker {worker_key}, got {overlap_scores.scores[worker_key]}"

    blocks = radix_tree.dump_tree_as_events()
    assert len(blocks) == 1, f"Expected 1 block event, got {len(blocks)}"
    json.loads(blocks[0])  # check valid json

    # cleanup
    radix_tree.remove_worker(worker_id)
    blocks_empty = radix_tree.dump_tree_as_events()
    assert (
        len(blocks_empty) == 0
    ), f"Expected 0 block events after removal, got {len(blocks_empty)}"

    print(
        f"✓ RadixTree test passed: worker {worker_key} has score {overlap_scores.scores[worker_key]}"
    )


@pytest.mark.timeout(5)  # Expected: ~1s, timeout set to 5x for safety
@pytest.mark.parametrize("num_threads", [2, 3, 5, 128])
@pytest.mark.parametrize("prepopulate_worker_ids", [True, False])
@pytest.mark.parametrize("expiration_duration_secs", [None])
@pytest.mark.parametrize("is_threaded", [True, False])
def test_radix_tree_thread_safety(
    num_threads,
    prepopulate_worker_ids,
    expiration_duration_secs,
    is_threaded,
):
    """Test RadixTree thread safety by applying events from multiple threads."""
    radix_tree = RadixTree(expiration_duration_secs=expiration_duration_secs)
    threads = []
    done_counter = 0
    exception_counter = 0

    def worker(worker_id, prepopulate_worker_ids: bool = False):
        try:
            nonlocal done_counter
            worker_id = worker_id
            hash = worker_id
            if prepopulate_worker_ids:
                hash = (
                    2**32 - worker_id
                )  # use different hash for prepopulate_worker_ids
            assert 0 <= hash < 2**64  # needs to be valid u64
            store_event = {
                "event_id": worker_id,
                "data": {
                    "stored": {
                        "parent_hash": None,
                        "blocks": [
                            {
                                "block_hash": hash,
                                "tokens_hash": hash,
                            }
                        ],
                    }
                },
            }
            event_bytes = json.dumps(store_event).encode("utf-8")
            radix_tree.apply_event(worker_id, event_bytes)
            if not prepopulate_worker_ids:
                done_counter += 1
        except Exception as e:
            print(f"Exception in worker {worker_id}: {e}")
            nonlocal exception_counter
            exception_counter += 1

    if prepopulate_worker_ids:
        for i in range(num_threads):
            worker(i, prepopulate_worker_ids=True)
        assert (
            exception_counter == 0
        ), f"Warmup: expected 0 exceptions, got {exception_counter}"

    for i in range(num_threads):
        if is_threaded:
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        else:
            worker(i)
    if is_threaded:
        timeout = 10  # seconds
        for t in threads:
            t.join(timeout)
            assert not t.is_alive(), "Thread timed out"
    assert exception_counter == 0, f"Expected 0 exceptions, got {exception_counter}"
    assert (
        done_counter == num_threads
    ), f"Expected {num_threads} done, got {done_counter}"

    for i in range(num_threads):
        overlap_scores = radix_tree.find_matches([i])
        assert overlap_scores.scores is not None
        worker_key = (i, 0)
        assert (
            worker_key in overlap_scores.scores
        ), f"Worker {worker_key} not found in scores"
        assert (
            overlap_scores.scores[worker_key] == 1
        ), f"Expected score 1 for worker {worker_key}, got {overlap_scores.scores[worker_key]}"
    # get all blocks
    blocks = radix_tree.dump_tree_as_events()
    expected_blocks = num_threads + (prepopulate_worker_ids * num_threads)
    assert (
        len(blocks) == expected_blocks
    ), f"Expected {expected_blocks} block events, got {len(blocks)}"
    # remove single worker
    radix_tree.remove_worker(0)
    expected_blocks_after_removal = expected_blocks - (
        2 if prepopulate_worker_ids else 1
    )
    blocks_after_removal = radix_tree.dump_tree_as_events()
    assert (
        len(blocks_after_removal) == expected_blocks_after_removal
    ), f"Expected {expected_blocks_after_removal} block events after removal, got {len(blocks_after_removal)}"


# ----- token-id inputs: list[int] vs numpy array -----
#
# The routing entry points accept a NumPy uint32/int64 array in addition to a
# Python sequence of ints, so callers holding tokens in a uint32 buffer do not
# have to materialize one Python int per token to cross the binding. Both
# forms must produce identical results.


@pytest.mark.timeout(5)
@pytest.mark.parametrize("dtype", ["uint32", "int64"])
def test_compute_block_hash_for_seq_accepts_numpy(dtype):
    """compute_block_hash_for_seq: numpy array matches the list[int] result."""
    import numpy as np

    from dynamo.llm import compute_block_hash_for_seq

    tokens = list(range(64))

    from_list = compute_block_hash_for_seq(tokens, 16)
    from_array = compute_block_hash_for_seq(np.array(tokens, dtype=dtype), 16)

    assert from_list == from_array
    assert len(from_list) == 4


@pytest.mark.timeout(5)
@pytest.mark.parametrize("dtype", ["uint32", "int64"])
def test_compute_block_hash_for_seq_accepts_strided_numpy(dtype):
    """A non-contiguous NumPy view matches the list[int] result."""
    import numpy as np

    from dynamo.llm import compute_block_hash_for_seq

    tokens = np.arange(128, dtype=dtype)[::2]

    assert compute_block_hash_for_seq(tokens, 16) == compute_block_hash_for_seq(
        tokens.tolist(), 16
    )


@pytest.mark.timeout(5)
def test_compute_block_hash_for_seq_rejects_bad_token_input():
    """A non-sequence token input raises ValueError, not a panic."""
    from dynamo.llm import compute_block_hash_for_seq

    with pytest.raises(ValueError):
        compute_block_hash_for_seq(object(), 16)


@pytest.mark.timeout(5)
@pytest.mark.parametrize("dtype", ["uint32", "int64"])
def test_router_request_new_accepts_numpy_tokens(dtype):
    """PyRouterRequestNew: numpy tokens land as the same list[int]."""
    import numpy as np

    from dynamo._core import PyRouterRequestNew

    tokens = [1, 2, 3, 4_000_000_000 if dtype == "uint32" else 4]

    from_list = PyRouterRequestNew(tokens=tokens)
    from_array = PyRouterRequestNew(tokens=np.array(tokens, dtype=dtype))

    assert from_array.tokens == from_list.tokens == tokens

    # The setter accepts the same inputs and always reads back a list[int].
    from_list.tokens = np.array(tokens, dtype=dtype)
    assert from_list.tokens == tokens


@pytest.mark.timeout(5)
def test_router_request_new_rejects_out_of_range_int64_tokens():
    """int64 token ids outside u32 are rejected rather than silently wrapped."""
    import numpy as np

    from dynamo._core import PyRouterRequestNew

    with pytest.raises(ValueError):
        PyRouterRequestNew(tokens=np.array([-1, 2, 3], dtype="int64"))
