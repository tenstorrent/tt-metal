# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Exercise: Sharded elementwise add from Python.

Build: cmake --build build -- onboarding
"""

import torch
import ttnn
from _e09_exercise import s09s1_interleaved_to_sharded, s09s2_sharded_add
from loguru import logger


def sharded_add(device, a: torch.Tensor, b: torch.Tensor, shard_strategy=None) -> torch.Tensor:
    if shard_strategy is None:
        shard_strategy = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    # TODO: Implement the Python wrapper
    #
    # 1. Convert both torch tensors to TTNN tensors on the device:
    #    ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    #
    # 2. Shard both inputs using Stage1's operation:
    #    s09s1_interleaved_to_sharded(tt_tensor, shard_strategy.value)
    #
    # 3. Call the sharded add operation:
    #    s09s2_sharded_add(tt_a_sharded, tt_b_sharded)
    #
    # 4. Convert the result back to a torch tensor:
    #    ttnn.to_torch(tt_output)
    raise NotImplementedError("TODO: implement sharded_add")
