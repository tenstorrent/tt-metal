# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Main Entry Point

Row-wise layer normalization on ROW_MAJOR interleaved tensors.
Normalizes across the last dimension (W).

    output = (x - mean) / sqrt(var + epsilon) * gamma + beta

Usage:
    from ttnn.operations.layer_norm_rm import layer_norm_rm
    output = layer_norm_rm(input_tensor)
    output = layer_norm_rm(input_tensor, gamma, beta, epsilon=1e-6)
"""

import ttnn
from .layer_norm_rm_program_descriptor import create_program_descriptor


def layer_norm_rm(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    epsilon: float = 1e-5,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Row-wise layer normalization on ROW_MAJOR interleaved tensors.

    Args:
        input_tensor: Input tensor (bfloat16, ROW_MAJOR, on device, H and W multiples of 32)
        gamma: Optional per-element scale tensor of shape (1,1,1,W) (bfloat16, ROW_MAJOR)
        beta: Optional per-element bias tensor of shape (1,1,1,W) (bfloat16, ROW_MAJOR)
        epsilon: Stability constant added to variance (default 1e-5)
        memory_config: Memory configuration for output tensor (default: DRAM interleaved)

    Returns:
        Output tensor with same shape, dtype, and layout as input
    """
    _validate_input(input_tensor, gamma, beta)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output shape: same as input (centered output is full width)
    output_shape = list(input_tensor.shape)

    # Allocate output tensor on device (positional args)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(input_tensor, output_tensor, gamma=gamma, beta=beta, epsilon=epsilon)

    # Build IO tensor list: all inputs first, output MUST be last
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
    """Validate input tensor and optional gamma/beta meet requirements."""
    # Dtype check
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError("layer_norm_rm: Input must be bfloat16")

    # Layout check - must be ROW_MAJOR
    if input_tensor.layout == ttnn.TILE_LAYOUT:
        raise ValueError("layer_norm_rm: Input must be row-major (ROW_MAJOR_LAYOUT), not TILE_LAYOUT")

    # Rank check
    if len(input_tensor.shape) < 2:
        raise ValueError("layer_norm_rm: Input must be at least 2D")

    # Tile alignment check (last 2 dims must be multiples of 32)
    shape = input_tensor.shape
    H = shape[-2]
    W = shape[-1]
    if H % 32 != 0:
        raise ValueError(f"layer_norm_rm: H dimension ({H}) must be a multiple of 32")
    if W % 32 != 0:
        raise ValueError(f"layer_norm_rm: W dimension ({W}) must be a multiple of 32")

    # Gamma width check
    if gamma is not None:
        gamma_W = gamma.shape[-1]
        if gamma_W != W:
            raise ValueError(f"layer_norm_rm: gamma width ({gamma_W}) must match input width ({W})")

    # Beta width check
    if beta is not None:
        beta_W = beta.shape[-1]
        if beta_W != W:
            raise ValueError(f"layer_norm_rm: beta width ({beta_W}) must match input width ({W})")
