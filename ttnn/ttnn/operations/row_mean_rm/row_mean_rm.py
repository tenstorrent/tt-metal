# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
row_mean_rm - Compute mean across the last dimension (W) of a row-major tensor.

Computes:
    mean[b,c,h] = sum(input[b,c,h,:]) / W

Output shape is (..., H, 32) — a single tile column. The mean value for each
row of 32 physical rows lives in column 0 of the output tile.

Inputs:
    input_tensor: RM interleaved bfloat16, rank >= 2, last 2 dims tile-aligned

Output:
    (..., H, 32) RM interleaved bfloat16
"""

import ttnn
from .row_mean_rm_program_descriptor import create_program_descriptor


def row_mean_rm(
    input_tensor: ttnn.Tensor,
    *,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    _validate_inputs(input_tensor)

    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output is single tile wide: (..., H, 32)
    rank = len(input_tensor.shape)
    output_shape = [input_tensor.shape[i] for i in range(rank - 1)] + [32]

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(input_tensor, output_tensor)

    # Output MUST be last in the list
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)


def _validate_inputs(input_tensor: ttnn.Tensor) -> None:
    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("row_mean_rm: Input must be row-major")
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError("row_mean_rm: Input must be bfloat16")
    if len(input_tensor.shape) < 2:
        raise ValueError("row_mean_rm: Input must be at least 2D")

    rank = len(input_tensor.shape)
    W = input_tensor.shape[rank - 1]
    H = input_tensor.shape[rank - 2]
    if W % 32 != 0:
        raise ValueError(f"row_mean_rm: W={W} must be tile-aligned (multiple of 32)")
    if H % 32 != 0:
        raise ValueError(f"row_mean_rm: H={H} must be tile-aligned (multiple of 32)")
