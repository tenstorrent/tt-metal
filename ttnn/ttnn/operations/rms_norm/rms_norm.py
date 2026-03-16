# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
RMS Norm - Main Entry Point

Implements RMSNorm(x) = x / sqrt(mean(x^2, dim=-1, keepdim=True) + epsilon) * gamma

This file provides the user-facing function that:
1. Validates input/gamma tensors
2. Computes output tensor shape
3. Allocates output tensor on device
4. Creates the program descriptor
5. Launches via ttnn.generic_op
"""

from typing import Optional

import ttnn

from .rms_norm_program_descriptor import create_program_descriptor


def rms_norm(
    input_tensor: ttnn.Tensor,
    *,
    gamma: Optional[ttnn.Tensor] = None,
    epsilon: float = 1e-6,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    RMS Norm operation.

    Normalizes each row (last dimension) by its root mean square,
    then optionally scales by gamma.

    RMSNorm(x) = x / sqrt(mean(x^2, dim=-1, keepdim=True) + epsilon) * gamma

    Args:
        input_tensor: Input tensor (must be on device, rank >= 2).
                      Supports ROW_MAJOR_LAYOUT and TILE_LAYOUT.
                      Supported dtypes: bfloat16, float32.
        gamma: Optional per-element scale tensor with shape (1,1,1,W) in
               ROW_MAJOR_LAYOUT, where W matches input's last dimension.
        epsilon: Numerical stability constant (default: 1e-6).
        memory_config: Memory configuration for output tensor (default: DRAM interleaved).

    Returns:
        Output tensor with same shape, dtype, and layout as input.

    Raises:
        ValueError: If input tensor has rank < 2.
        ValueError: If gamma's last dimension doesn't match input's last dimension.
        ValueError: If input is TILE_LAYOUT and H or W is not divisible by 32.
    """
    _validate_input(input_tensor, gamma)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output shape: reduced (stage 2 - square_reduce: tile-aligned reduced width)
    output_shape = list(input_tensor.shape)[:-1] + [32]

    # Output layout matches input layout (RM in -> RM out, TILE in -> TILE out)
    output_layout = input_tensor.layout

    # CRITICAL: allocate_tensor_on_device requires POSITIONAL args
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        output_layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        gamma=gamma,
        epsilon=epsilon,
    )

    # Build the tensor list for generic_op.
    # Output tensor MUST be last in the list.
    io_tensors = [input_tensor]
    if gamma is not None:
        io_tensors.append(gamma)
    io_tensors.append(output_tensor)

    return ttnn.generic_op(io_tensors, program_descriptor)


def _validate_input(input_tensor: ttnn.Tensor, gamma: Optional[ttnn.Tensor] = None) -> None:
    """Validate input tensor meets requirements."""
    input_shape = input_tensor.shape
    rank = len(input_shape)

    if rank < 2:
        raise ValueError(f"rms_norm: input must have at least 2 dimensions, got rank {rank}")

    W = input_shape[-1]
    H = input_shape[-2]

    # TILE_LAYOUT requires H and W divisible by 32
    if input_tensor.layout == ttnn.TILE_LAYOUT:
        if H % 32 != 0:
            raise ValueError(f"rms_norm: TILE_LAYOUT requires H divisible by 32, got H={H}")
        if W % 32 != 0:
            raise ValueError(f"rms_norm: TILE_LAYOUT requires W divisible by 32, got W={W}")

    # Validate gamma if provided
    if gamma is not None:
        gamma_W = gamma.shape[-1]
        if gamma_W != W:
            raise ValueError(
                f"rms_norm: gamma's last dimension ({gamma_W}) must match " f"input's last dimension ({W})"
            )
