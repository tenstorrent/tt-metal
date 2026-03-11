# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Softmax Operation - Main Entry Point

Computes numerically-stable softmax along a specified dimension:
    output[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))

Usage:
    from ttnn.operations.softmax import softmax
    output = softmax(input_tensor, dim=-1)
"""

import ttnn
from .softmax_program_descriptor import create_program_descriptor


def softmax(
    input_tensor: ttnn.Tensor,
    dim: int = -1,
    numeric_stable: bool = True,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Softmax operation entry point.

    Args:
        input_tensor: Input tensor (must be on device, bfloat16, TILE_LAYOUT, rank >= 2)
        dim: Dimension along which softmax is computed (-1 for width, -2 for height)
        numeric_stable: If True, subtract max before exp for numerical stability
        memory_config: Memory configuration for output tensor (default: DRAM interleaved)

    Returns:
        Output tensor with softmax applied along the specified dimension
    """
    _validate_input(input_tensor, dim)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output shape is the same as input shape
    output_shape = list(input_tensor.shape)

    # Allocate output tensor on device (positional args required)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    # Create program descriptor with dim and numeric_stable parameters
    program_descriptor = create_program_descriptor(input_tensor, output_tensor, dim=dim, numeric_stable=numeric_stable)

    # Output tensor MUST be last in the list
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)


def _validate_input(input_tensor: ttnn.Tensor, dim: int) -> None:
    """Validate input tensor meets requirements for softmax."""
    shape = input_tensor.shape
    rank = len(shape)

    # Must be at least rank 2
    if rank < 2:
        raise ValueError(f"softmax: input must have at least 2 dimensions, got rank {rank}")

    # Must be 4D (N, C, H, W)
    if rank != 4:
        raise ValueError(f"softmax: input must be 4D (N, C, H, W), got rank {rank}")

    # Must be bfloat16
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError(f"softmax: input must be bfloat16, got {input_tensor.dtype}")

    # Must be TILE_LAYOUT
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"softmax: input must be TILE_LAYOUT, got {input_tensor.layout}")

    # dim must be -1 or -2
    if dim not in (-1, -2):
        raise ValueError(f"softmax: dim must be -1 or -2, got {dim}")

    # H and W must be divisible by 32 (tile size)
    H = shape[2]
    W = shape[3]
    if H % 32 != 0:
        raise ValueError(f"softmax: H must be divisible by 32, got H={H}")
    if W % 32 != 0:
        raise ValueError(f"softmax: W must be divisible by 32, got W={W}")
