# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for e04_matmul_add.

Run: ./run.sh "e04 and solution"
"""

from pathlib import Path

import pytest
import torch
import ttnn
from loguru import logger
from ttnn.tutorials.onboarding.common.utils import load_module, pcc

LESSON_DIR = Path(__file__).parent


@pytest.mark.parametrize("module_name", ["exercise", "solution"])
def test_e04_matmul_add(module_name):
    """Test matmul_add implementation against PyTorch reference."""
    module = load_module(module_name, LESSON_DIR)
    reference = load_module("reference", LESSON_DIR)

    # device = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))
    device = ttnn.open_device(device_id=0)

    # A: (M, K), B: (K, N), C: (M, N)
    M, K, N = 64, 64, 64
    a = torch.zeros(M, K, dtype=torch.float32)
    b = torch.zeros(K, N, dtype=torch.float32)
    c = torch.zeros(M, N, dtype=torch.float32)

    a[0][0] = 1.0  # Set one element to 1 to have a non-zero result
    b[0][0] = 2.0  # Set one element to 2 to have a non-zero result

    b[0][32] = 5.0  # Set one element to 3 to have a non-zero result in the second tile
    # c[0][32] = 1.0  # Set one element to

    a[0][32] = 3.0  # Set one element to 3 to have a non-zero result in the second tile

    a[32][0] = 4.0  # Set one element to 4 to have a non-zero result in the second tile

    a[32][32] = 6.0  # Set one element to 6 to have a non-zero result in the second tile
    b[32][32] = 7.0  # Set one element to 7 to have a non-zero result in the second tile
    b[32][0] = 8.0  # Set one element to 8 to have a non-zero result in the second tile

    result = module.matmul_add(device, a, b, c)
    expected = reference.matmul_add(a, b, c)

    ttnn.close_device(device)

    for i in range(64):
        for j in range(64):
            if result[i][j] != expected[i][j]:
                logger.info(f"Mismatch at element c[{i}][{j}]: {c[i][j]:.4f}")
                logger.info(f"result[{i}][{j}] = {result[i][j]:.4f}, expected[{i}][{j}] = {expected[i][j]:.4f}")

    logger.info(f"expected[32][0]:{expected[32][0]:.4f}, result[32][0]: {result[32][0]:.4f}")
    logger.info(f"expected[0][0]:{expected[0][0]:.4f}, result[0][0]: {result[0][0]:.4f}")
    logger.info(f"expected[32][32]:{expected[32][32]:.4f}, result[32][32]: {result[32][32]:.4f}")
    logger.info(f"expected[0][32]:{expected[0][32]:.4f}, result[0][32]: {result[0][32]:.4f}")

    # logger.info(f'Result:\n{result}\nExpected:\n{expected}')

    # Matmul has more numerical error due to accumulation
    result_pcc = pcc(result, expected)
    assert result_pcc >= 0.98, f"PCC {result_pcc:.4f} < 0.98"
