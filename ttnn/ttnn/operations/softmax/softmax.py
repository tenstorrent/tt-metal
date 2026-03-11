# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Softmax - Main Entry Point

Implements softmax(x, dim) = exp(x - max(x, dim)) / sum(exp(x - max(x, dim)), dim)

Supports dim=-1 (along width) and dim=-2 (along height).

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
        input_tensor: Input tensor (bfloat16, TILE_LAYOUT, 4D on device)
        dim: Dimension along which softmax is computed (-1 or -2)
        numeric_stable: Whether to subtract max before exp for numerical stability
        memory_config: Memory configuration for output tensor (default: DRAM interleaved)

    Returns:
        Output tensor with same shape, dtype, layout as input
    """
    _validate_input(input_tensor, dim)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output shape is identical to input shape
    output_shape = list(input_tensor.shape)

    # Allocate output tensor on device (POSITIONAL args required)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(input_tensor, output_tensor, dim=dim, numeric_stable=numeric_stable)

    # Output tensor MUST be last in the list
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)


def _validate_input(input_tensor: ttnn.Tensor, dim: int) -> None:
    """Validate input tensor meets softmax requirements."""
    shape = input_tensor.shape

    # Check rank
    if len(shape) < 2:
        raise ValueError("softmax requires at least 2D input")

    # Check dtype
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError("softmax requires bfloat16 input")

    # Check layout
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("softmax requires TILE_LAYOUT")

    # Check dim
    if dim not in (-1, -2):
        raise ValueError("softmax only supports dim=-1 or dim=-2")

    # Check tile alignment (H and W must be divisible by 32)
    H = shape[-2]
    W = shape[-1]
    if H % 32 != 0:
        raise ValueError(f"H ({H}) must be tile-aligned (divisible by 32)")
    if W % 32 != 0:
        raise ValueError(f"W ({W}) must be tile-aligned (divisible by 32)")
