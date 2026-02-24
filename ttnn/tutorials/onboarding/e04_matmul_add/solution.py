# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Solution: Call the custom matmul_add operation from Python.

This calls the operation defined in solution_cpp/.
Build: cmake --build build -- onboarding
"""

import torch
import ttnn
from _e04_solution import s04_matmul_add
from loguru import logger


def matmul_add(device, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    tt_a = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_c = ttnn.from_torch(c, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    logger.info(f"tt_a shape: {tt_a.shape}, layout: {tt_a.layout}, memory config: {tt_a.memory_config()}")
    logger.info(f"tt_b shape: {tt_b.shape}, layout: {tt_b.layout}, memory config: {tt_b.memory_config()}")
    logger.info(f"tt_c shape: {tt_c.shape}, layout: {tt_c.layout}, memory config: {tt_c.memory_config()}")

    # Call the custom operation (registered via nanobind)
    tt_result = s04_matmul_add(tt_a, tt_b, tt_c)
    logger.info(
        f"tt_result shape: {tt_result.shape}, layout: {tt_result.layout}, memory config: {tt_result.memory_config()}"
    )
    # return tt_result
    return ttnn.to_torch(tt_result)
