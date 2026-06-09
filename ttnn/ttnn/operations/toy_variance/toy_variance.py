# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Toy variance op — computes per-row population variance over the W dimension
using the streaming-reduce helpers.

The reduction is chunked into blocks of BLOCK_SIZE tiles each so the W
dimension can be arbitrarily wide (e.g. 32x64000) without exceeding L1.

Constraints (intentionally narrow for a toy):
- Input is on-device, TILE_LAYOUT, bfloat16.
- H and W are tile-aligned (multiples of 32).
- Reduction is over the last dimension (W) only.
- Single-core, single-tile-per-row output. Output shape is the input shape
  with the last dim collapsed to 32; the variance value lives in column 0.
"""

import ttnn
from .toy_variance_program_descriptor import create_program_descriptor


def toy_variance(
    input_tensor: ttnn.Tensor,
    *,
    std_dev: bool = False,
    block_size: int | None = None,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Per-row population variance (or standard deviation) over the W dimension.

    Args:
        input_tensor: TILE_LAYOUT bfloat16 tensor with tile-aligned H and W.
        std_dev: If True, return std deviation = sqrt(variance). The sqrt is
            applied as the last-block post-op of the streaming reduce — no
            extra pass over the data. Default False (returns variance).
        block_size: Optional override for the streaming block size (in tiles).
            Must divide Wt = W/32. Defaults to 8 (or the largest divisor <= 8).
        memory_config: Output memory config (default: DRAM interleaved).
    """
    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # NOTE on implicit tile padding: padded W positions flow through sub<COL>
    # + square_in_place *before* the partial scaler zeros them in the reduce.
    # The partial-scaler tile multiplies the last W-tile of the last block
    # by zero at padded positions, so any FINITE garbage there ends up as
    # (garbage - mean)^2 * 0 = 0 in the accumulator. Caller is responsible
    # for ensuring padded values are finite — if you have inf/nan garbage,
    # call ttnn.fill_implicit_tile_padding(input, 0.0) first to avoid
    # inf * 0 = nan propagating into the result.

    input_shape = list(input_tensor.shape)
    output_shape = input_shape[:-1] + [32]

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(input_tensor, output_tensor, block_size=block_size, std_dev=std_dev)
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
