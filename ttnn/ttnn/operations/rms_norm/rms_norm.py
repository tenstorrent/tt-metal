# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
RMS Norm - Main Entry Point

RMSNorm(x) = x / sqrt(mean(x^2, dim=-1, keepdim=True) + epsilon) * gamma

This file provides the user-facing function that:
1. Validates input tensor and gamma
2. Computes output tensor shape
3. Allocates output tensor on device
4. Creates the program descriptor
5. Launches via ttnn.generic_op

Usage:
    from ttnn.operations.rms_norm import rms_norm
    output = rms_norm(input_tensor, gamma=gamma_tensor, epsilon=1e-6)
"""

import ttnn
from .rms_norm_program_descriptor import create_program_descriptor


def rms_norm(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    epsilon: float = 1e-6,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    RMS normalization along the last dimension.

    Args:
        input_tensor: Input tensor (must be on device, rank >= 2, bfloat16 or float32)
        gamma: Optional scale parameter tensor, shape (1,1,1,W), ROW_MAJOR_LAYOUT
        epsilon: Stability constant (default: 1e-6)
        memory_config: Memory configuration for output tensor (default: DRAM interleaved)

    Returns:
        Output tensor with same shape, dtype, and layout as input
    """
    _validate_input(input_tensor, gamma)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output shape is same as input
    output_shape = list(input_tensor.shape)

    # Output layout matches input layout
    output_layout = input_tensor.layout

    # Allocate output tensor on device (positional args required)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        output_layout,
        device,
        output_memory_config,
    )

    # Create program descriptor
    program_descriptor = create_program_descriptor(input_tensor, output_tensor, gamma=gamma, epsilon=epsilon)

    # Execute - output tensor MUST be last in the list
    io_tensors = [input_tensor]
    if gamma is not None:
        io_tensors.append(gamma)
    io_tensors.append(output_tensor)

    return ttnn.generic_op(io_tensors, program_descriptor)


def _validate_input(input_tensor: ttnn.Tensor, gamma: ttnn.Tensor = None) -> None:
    """Validate input tensor and gamma meet requirements."""
    if len(input_tensor.shape) < 2:
        raise RuntimeError("rms_norm: input must have at least 2 dimensions")

    if gamma is not None:
        input_w = input_tensor.shape[-1]
        gamma_w = gamma.shape[-1]
        if gamma_w != input_w:
            raise RuntimeError(
                f"rms_norm: gamma last dimension ({gamma_w}) must match " f"input last dimension ({input_w})"
            )
