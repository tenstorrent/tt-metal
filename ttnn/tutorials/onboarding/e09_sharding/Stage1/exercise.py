# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Exercise: Call the interleaved_to_sharded operation from Python.

This calls the operation defined in exercise_cpp/.
Build: cmake --build build -- onboarding
"""

import torch
import ttnn
from _e09s1_exercise import s09s1_interleaved_to_sharded
from loguru import logger


def interleaved_to_sharded(device, input: torch.Tensor, shard_strategy=None) -> torch.Tensor:
    if shard_strategy is None:
        shard_strategy = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    # TODO: Implement the Python wrapper
    #
    # 1. Convert the torch tensor to a TTNN tensor on the device:
    #    ttnn.from_torch(input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    #
    # 2. Call the custom operation with shard_strategy:
    #    s09s1_interleaved_to_sharded(tt_input, shard_strategy.value)
    #
    # 3. Convert the result back to a torch tensor:
    #    ttnn.to_torch(tt_output)
    raise NotImplementedError("TODO: implement interleaved_to_sharded")
