# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Toy reduce with partial scaler — demonstrates tile-padding avoidance for
non-tile-aligned dimensions using the dual-scaler pattern.

Supports both REDUCE_ROW (dim=-1, W) and REDUCE_COL (dim=-2, H) via a single
set of kernels parameterized by compile-time defines.
"""

import ttnn
from .toy_reduce_partial_program_descriptor import create_program_descriptor


def toy_reduce_partial(
    input_tensor: ttnn.Tensor,
    dim: int,
    *,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Sum-reduce over dim with partial-scaler tile-padding avoidance.

    Args:
        input_tensor: Input tensor on device (TILE_LAYOUT, bfloat16).
        dim: Reduction dimension. -1 or 3 for W (REDUCE_ROW), -2 or 2 for H (REDUCE_COL).
        memory_config: Output memory config (default: DRAM interleaved).
    """
    ndim = len(input_tensor.shape)
    if dim < 0:
        dim += ndim
    if dim not in (ndim - 1, ndim - 2):
        raise ValueError(f"toy_reduce_partial: dim must be -1 (W) or -2 (H), got {dim - ndim}")

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    input_shape = list(input_tensor.shape)
    if dim == ndim - 1:  # REDUCE_ROW: collapse W to 1 tile
        output_shape = input_shape[:-1] + [32]
    else:  # REDUCE_COL: collapse H to 1 tile
        output_shape = input_shape[:-2] + [32, input_shape[-1]]

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        device,
        output_memory_config,
    )

    reduce_row = dim == ndim - 1
    program_descriptor = create_program_descriptor(input_tensor, output_tensor, reduce_row=reduce_row)
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
