# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Solution: Call the interleaved_to_sharded operation from Python.

This calls the operation defined in solution_cpp/.
Build: cmake --build build -- onboarding
"""

import torch
import ttnn
from _e09_solution import s09s1_interleaved_to_sharded
from loguru import logger


def interleaved_to_sharded(device, input: torch.Tensor, shard_strategy=None) -> torch.Tensor:
    if shard_strategy is None:
        shard_strategy = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    # Send input to device as DRAM INTERLEAVED (default)
    tt_input = ttnn.from_torch(input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    logger.info(
        f"Input: shape={tt_input.shape}, layout={tt_input.layout}, " f"memory_config={tt_input.memory_config()}"
    )

    # Run our custom interleaved-to-sharded operation
    tt_output = s09s1_interleaved_to_sharded(tt_input, shard_strategy.value)

    logger.info(
        f"Output: shape={tt_output.shape}, layout={tt_output.layout}, " f"memory_config={tt_output.memory_config()}"
    )

    return ttnn.to_torch(tt_output)
