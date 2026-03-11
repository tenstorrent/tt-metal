# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Main Entry Point

Performs layer normalization on row-major interleaved bfloat16 tensors.
The reader tilizes in-kernel, compute operates on tiles, writer untilizes in-kernel.

Usage:
    from ttnn.operations.layer_norm_rm import layer_norm_rm
    output = layer_norm_rm(input_tensor)
    output = layer_norm_rm(input_tensor, gamma, beta, epsilon=1e-5)
"""

import ttnn
from .layer_norm_rm_program_descriptor import create_program_descriptor


def layer_norm_rm(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    *,
    epsilon: float = 1e-5,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Layer normalization on row-major interleaved bfloat16 tensors.

    Normalizes each row (last dimension) independently:
        mean = sum(x_i) / W
        var = sum((x_i - mean)^2) / W
        x_norm = (x_i - mean) / sqrt(var + eps)
        output = x_norm * gamma + beta   (if gamma/beta provided)

    Args:
        input_tensor: Input tensor on device (RM, interleaved, bfloat16, tile-aligned)
        gamma: Optional per-element scale, shape (1,1,1,W), RM bfloat16
        beta: Optional per-element shift, shape (1,1,1,W), RM bfloat16
        epsilon: Stability constant for variance (default 1e-5)
        memory_config: Memory configuration for output (default: DRAM interleaved)

    Returns:
        Output tensor with same shape, dtype, layout as input.
    """
    _validate_input(input_tensor, gamma, beta)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output shape same as input
    output_shape = list(input_tensor.shape)

    # Allocate output tensor on device (POSITIONAL args required)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    # Build the io_tensors list: all inputs first, output LAST
    io_tensors = [input_tensor]
    if gamma is not None:
        io_tensors.append(gamma)
    if beta is not None:
        io_tensors.append(beta)
    io_tensors.append(output_tensor)

    # Create program descriptor
    program_descriptor = create_program_descriptor(input_tensor, output_tensor, gamma=gamma, beta=beta, epsilon=epsilon)

    # Execute - output tensor is last in io_tensors
    return ttnn.generic_op(io_tensors, program_descriptor)


def _validate_input(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
) -> None:
    """Validate input tensors meet requirements."""
    # dtype check
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError(f"layer_norm_rm: input must be bfloat16, got {input_tensor.dtype}")

    # layout check
    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(
            f"layer_norm_rm: input must be ROW_MAJOR_LAYOUT (tilize happens in-kernel), got {input_tensor.layout}"
        )

    # shape check - H and W must be multiples of 32
    shape = input_tensor.shape
    if len(shape) < 2:
        raise ValueError("layer_norm_rm: input must have at least 2 dimensions")
    H = shape[-2]
    W = shape[-1]
    if H % 32 != 0:
        raise ValueError(f"layer_norm_rm: height ({H}) must be a multiple of 32 for tile alignment")
    if W % 32 != 0:
        raise ValueError(f"layer_norm_rm: width ({W}) must be a multiple of 32 for tile alignment")

    # gamma validation
    if gamma is not None:
        if gamma.dtype != ttnn.bfloat16:
            raise ValueError(f"layer_norm_rm: gamma must be bfloat16, got {gamma.dtype}")
        gamma_W = gamma.shape[-1]
        if gamma_W != W:
            raise ValueError(f"layer_norm_rm: gamma width ({gamma_W}) must match input width ({W})")

    # beta validation
    if beta is not None:
        if beta.dtype != ttnn.bfloat16:
            raise ValueError(f"layer_norm_rm: beta must be bfloat16, got {beta.dtype}")
        beta_W = beta.shape[-1]
        if beta_W != W:
            raise ValueError(f"layer_norm_rm: beta width ({beta_W}) must match input width ({W})")
