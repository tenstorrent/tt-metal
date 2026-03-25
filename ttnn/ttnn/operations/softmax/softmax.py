# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Softmax - Main Entry Point

Computes softmax along a specified dimension:
  stable:   output[...,i,...] = exp(x[...,i,...] - max(x)) / sum(exp(x[...,j,...] - max(x)))
  unstable: output[...,i,...] = exp(x[...,i,...]) / sum(exp(x[...,j,...]))

Supports dim=-1 (width) and dim=-2 (height).
"""

import ttnn
from .softmax_program_descriptor import create_program_descriptor


def softmax(
    input_tensor: ttnn.Tensor,
    dim: int = -1,
    *,
    numeric_stable: bool = True,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Compute softmax along the specified dimension.

    Args:
        input_tensor: Input tensor (bfloat16, TILE_LAYOUT, 4D, on device)
        dim: Dimension along which softmax is computed (-1 for W, -2 for H)
        numeric_stable: If True, subtract max before exp for numerical stability
        memory_config: Memory configuration for output tensor (default: DRAM interleaved)

    Returns:
        Output tensor with softmax applied (same shape, dtype, layout as input)

    Raises:
        ValueError: If input tensor has wrong dtype, layout, rank, or invalid dim
    """
    _validate_input(input_tensor, dim)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output shape is same as input
    output_shape = list(input_tensor.shape)

    # Allocate output tensor on device (POSITIONAL args required)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    # Create program descriptor with dim and numeric_stable parameters
    program_descriptor = create_program_descriptor(input_tensor, output_tensor, dim=dim, numeric_stable=numeric_stable)

    # Execute - output tensor MUST be last in list
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)


def _validate_input(input_tensor: ttnn.Tensor, dim: int) -> None:
    """Validate input tensor meets requirements for softmax."""
    # Check dtype
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError(f"softmax: input dtype must be bfloat16, got {input_tensor.dtype}")

    # Check layout
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"softmax: input layout must be TILE_LAYOUT, got {input_tensor.layout}")

    # Check rank
    rank = len(input_tensor.shape)
    if rank < 2:
        raise ValueError(f"softmax: input must have at least 2 dimensions, got {rank}")

    # Check dim
    if dim not in (-1, -2):
        raise ValueError(f"softmax: dim must be -1 or -2, got {dim}")
