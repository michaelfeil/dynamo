# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import pickle

import pytest

from dynamo._core import MediaEmbedding

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def test_b10_media_embedding_exposes_length_bytes_and_repr():
    embedding = MediaEmbedding(b"packed")

    assert len(embedding) == 6
    assert embedding.to_bytes() == b"packed"
    assert repr(embedding) == "MediaEmbedding(6 bytes)"


def test_b10_media_embedding_cannot_be_pickled_or_deep_copied():
    embedding = MediaEmbedding(b"packed")

    with pytest.raises(TypeError):
        pickle.dumps(embedding)
    with pytest.raises(TypeError):
        copy.deepcopy(embedding)
