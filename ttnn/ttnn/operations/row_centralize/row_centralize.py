# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Row Centralize - Main Entry Point

Row-wise standardization (LayerNorm without affine parameters):
    mu      = mean(x, dim=-1, keepdim=True)
    c       = x - mu
    var     = mean(c^2, dim=-1, keepdim=True)
    s       = rsqrt(var + epsilon)
    y       = c * s

Input:  RM interleaved bfloat16, at least 2D, dims divisible by 32
Output: RM interleaved bfloat16, same shape

Usage:
    from ttnn.operations.row_centralize import row_centralize
    output = row_centralize(input_tensor, epsilon=1e-5)
"""

import ttnn
from .row_centralize_program_descriptor import create_program_descriptor

TILE_SIZE = 32


def row_centralize(
    input_tensor: ttnn.Tensor,
    *,
    epsilon: float = 1e-5,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Row-wise standardization of a tensor.

    For each row r:
        mu_r    = mean(x[r, :])
        c_r     = x[r, :] - mu_r
        var_r   = mean(c_r^2)
        s_r     = rsqrt(var_r + epsilon)
        y[r, :] = c_r * s_r

    Args:
        input_tensor: Input tensor. Must be:
            - At least 2D
            - bfloat16 dtype
            - ROW_MAJOR layout
            - INTERLEAVED memory layout
            - On device
            - Last dimension divisible by 32
            - Second-to-last dimension divisible by 32
        epsilon: Small constant for numerical stability (default 1e-5)
        memory_config: Memory configuration for output (default: DRAM interleaved)

    Returns:
        Output tensor with same shape, dtype=bfloat16, layout=ROW_MAJOR, INTERLEAVED.
    """
    _validate_input(input_tensor)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output shape is identical to input shape
    shape = input_tensor.shape
    output_shape = [shape[i] for i in range(len(shape))]

    # Allocate output tensor on device (POSITIONAL args required)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(input_tensor, output_tensor, epsilon)

    # Output tensor MUST be last in the list
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)


def _validate_input(input_tensor: ttnn.Tensor) -> None:
    """Validate input tensor meets all requirements from the spec."""
    shape = input_tensor.shape
    rank = len(shape)

    if rank < 2:
        raise ValueError(f"row_centralize: Input tensor must have rank >= 2, got rank {rank}")

    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"row_centralize: Input tensor must be in ROW_MAJOR layout, got {input_tensor.layout}")

    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError(f"row_centralize: Input tensor dtype must be bfloat16, got {input_tensor.dtype}")

    if not input_tensor.is_allocated():
        raise ValueError("row_centralize: Input tensor must be on device (not allocated)")

    memory_config = input_tensor.memory_config()
    if memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError("row_centralize: Input tensor must have interleaved memory layout")

    last_dim = shape[rank - 1]
    if last_dim % TILE_SIZE != 0:
        raise ValueError(f"row_centralize: Input tensor last dimension must be divisible by 32, got {last_dim}")

    second_to_last_dim = shape[rank - 2]
    if second_to_last_dim % TILE_SIZE != 0:
        raise ValueError(
            f"row_centralize: Input tensor second-to-last dimension must be divisible by 32, "
            f"got {second_to_last_dim}"
        )
