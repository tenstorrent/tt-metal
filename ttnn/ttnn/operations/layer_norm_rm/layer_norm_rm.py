# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Layer Norm RM - Main Entry Point

Per-row layer normalization on row-major interleaved tensors.

    mean = sum(x) / W
    centered = x - mean
    var = sum(centered^2) / W
    output = centered * rsqrt(var + eps) * gamma + beta

Usage:
    from ttnn.operations.layer_norm_rm import layer_norm_rm
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
) -> ttnn.Tensor:
    """
    Row-major layer normalization.

    Args:
        input_tensor: Input tensor (bfloat16, ROW_MAJOR_LAYOUT, interleaved DRAM).
                      Shape must be at least 2D with last two dims divisible by 32.
        gamma: Optional scale parameter, shape (1,1,1,W) bfloat16 RM.
        beta: Optional bias parameter, shape (1,1,1,W) bfloat16 RM.
        epsilon: Numerical stability constant (default 1e-5).

    Returns:
        Output tensor with same shape, dtype, layout, and memory config as input.
    """
    _validate_input(input_tensor, gamma, beta)

    device = input_tensor.device()
    output_memory_config = ttnn.DRAM_MEMORY_CONFIG

    # Output shape same as input
    output_shape = list(input_tensor.shape)

    # Allocate output tensor on device (POSITIONAL args only)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    # Create program descriptor
    program_descriptor = create_program_descriptor(input_tensor, output_tensor, gamma=gamma, beta=beta, epsilon=epsilon)

    # Build io_tensors list: inputs first, output LAST
    io_tensors = [input_tensor]
    if gamma is not None:
        io_tensors.append(gamma)
    if beta is not None:
        io_tensors.append(beta)
    io_tensors.append(output_tensor)

    return ttnn.generic_op(io_tensors, program_descriptor)


def _validate_input(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
) -> None:
    """Validate input tensors meet requirements."""
    shape = input_tensor.shape

    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError("layer_norm_rm: Input must be bfloat16")

    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("layer_norm_rm: Input must be row-major")

    if len(shape) < 2:
        raise ValueError("layer_norm_rm: Input must be at least 2D")

    W = shape[-1]
    H = shape[-2]

    if W % 32 != 0:
        raise ValueError(f"layer_norm_rm: Width must be tile-aligned (div 32), got {W}")

    if H % 32 != 0:
        raise ValueError(f"layer_norm_rm: Height must be tile-aligned (div 32), got {H}")

    if gamma is not None:
        gamma_W = gamma.shape[-1]
        if gamma_W != W:
            raise ValueError(f"layer_norm_rm: gamma width ({gamma_W}) must match input width ({W})")

    if beta is not None:
        beta_W = beta.shape[-1]
        if beta_W != W:
            raise ValueError(f"layer_norm_rm: beta width ({beta_W}) must match input width ({W})")
