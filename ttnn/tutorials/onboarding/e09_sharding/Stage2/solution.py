# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Solution: Sharded elementwise add from Python.

Shards both inputs via Stage1's interleaved_to_sharded, then calls the
Stage2 sharded_add C++ operation.

Build: cmake --build build -- onboarding
"""

import torch
import ttnn
from _e09_solution import s09s1_interleaved_to_sharded, s09s2_sharded_add
from loguru import logger


def sharded_add(device, a: torch.Tensor, b: torch.Tensor, shard_strategy=None) -> torch.Tensor:
    if shard_strategy is None:
        shard_strategy = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    # Convert torch tensors to TTNN tensors on device (DRAM INTERLEAVED)
    tt_a = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Shard both inputs via Stage1's operation
    tt_a_sharded = s09s1_interleaved_to_sharded(tt_a, shard_strategy.value)
    tt_b_sharded = s09s1_interleaved_to_sharded(tt_b, shard_strategy.value)

    logger.info(f"input_a sharded: shape={tt_a_sharded.shape}, memory_config={tt_a_sharded.memory_config()}")
    logger.info(f"input_b sharded: shape={tt_b_sharded.shape}, memory_config={tt_b_sharded.memory_config()}")

    # Sharded elementwise add
    tt_output = s09s2_sharded_add(tt_a_sharded, tt_b_sharded)

    logger.info(f"output: shape={tt_output.shape}, memory_config={tt_output.memory_config()}")

    return ttnn.to_torch(tt_output)
