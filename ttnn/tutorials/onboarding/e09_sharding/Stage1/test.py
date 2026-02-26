# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for e09 Stage1: Interleaved to Sharded.

Run: ./run.sh "Stage1 and solution"
"""

from pathlib import Path

import pytest
import torch
import ttnn
from loguru import logger
from ttnn.tutorials.onboarding.common.utils import load_module, pcc

LESSON_DIR = Path(__file__).parent


@pytest.mark.parametrize("module_name", ["exercise", "solution"])
@pytest.mark.parametrize(
    "shard_strategy",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
def test_e09_stage1_interleaved_to_sharded(module_name, shard_strategy):
    """Test interleaved-to-sharded produces identical data for all shard strategies."""
    module = load_module(module_name, LESSON_DIR)
    reference = load_module("reference", LESSON_DIR)

    device = ttnn.open_device(device_id=0)

    M, N = 512, 256
    input_tensor = torch.randn(M, N, dtype=torch.float32)

    result = module.interleaved_to_sharded(device, input_tensor, shard_strategy)
    expected = reference.interleaved_to_sharded(input_tensor)

    ttnn.close_device(device)

    logger.info(f"Result shape: {result.shape}, Expected shape: {expected.shape}")

    result_pcc = pcc(result, expected)
    logger.info(f"PCC: {result_pcc:.6f}")
    assert result_pcc >= 0.99, f"PCC {result_pcc:.4f} < 0.99"
