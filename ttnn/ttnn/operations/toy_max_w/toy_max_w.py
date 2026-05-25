# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Toy max op — per-row maximum over the W dimension using the streaming-reduce
helpers.

The reduction is chunked into blocks of BLOCK_SIZE tiles each so the W
dimension can be arbitrarily wide (e.g. 32x64000) without exceeding L1.

Constraints (intentionally narrow for a toy):
- Input is on-device, TILE_LAYOUT, bfloat16.
- Reduction is over the last dimension (W) only.
- Single-core, single-tile-per-row output. Output shape is the input shape
  with the last dim collapsed to 32; the max value lives in column 0.
"""

import ttnn
from .toy_max_w_program_descriptor import create_program_descriptor


def toy_max_w(
    input_tensor: ttnn.Tensor,
    *,
    block_size: int | None = None,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Per-row maximum over the W dimension via streaming accumulate_reduce<MAX>.

    Args:
        input_tensor: TILE_LAYOUT bfloat16 tensor.
        block_size: Optional override for the streaming block size (in tiles).
            Must divide Wt = ceil(W/32). Defaults to the largest divisor <= 8.
        memory_config: Output memory config (default: DRAM interleaved).
    """
    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Padded W positions in the last W-tile are handled by the partial scaler:
    # for MAX, the helper fills padded positions with -inf so they never win
    # the max. Caller need not pre-fill the implicit padding.

    input_shape = list(input_tensor.shape)
    output_shape = input_shape[:-1] + [32]

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(input_tensor, output_tensor, block_size=block_size)
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
