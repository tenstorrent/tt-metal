# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Row-Major Layer Normalization Entry Point

Implements: output = gamma * (x - mean) / sqrt(var + eps) + beta
row-wise, using a hybrid RM <-> tilized compute approach via generic_op.

Usage:
    from ttnn.operations.layer_norm_rm import layer_norm_rm
    output = layer_norm_rm(input_tensor, gamma_tensor, beta_tensor, epsilon=1e-5)
"""

import ttnn
from .layer_norm_rm_program_descriptor import create_program_descriptor


def layer_norm_rm(
    input_tensor: ttnn.Tensor,
    gamma_tensor: ttnn.Tensor = None,
    beta_tensor: ttnn.Tensor = None,
    *,
    epsilon: float = 1e-5,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Row-major layer normalization.

    Computes: output = gamma * (x - mean) / sqrt(var + eps) + beta
    for each row of the input tensor.

    Args:
        input_tensor: Input tensor, ROW_MAJOR layout, bfloat16, interleaved DRAM.
                      Must be >= 2D with last 2 dims tile-aligned (multiples of 32).
        gamma_tensor: Optional scale parameter, shape (1, 1, 1, W), ROW_MAJOR, bfloat16.
                      If None, gamma defaults to ones.
        beta_tensor:  Optional bias parameter, shape (1, 1, 1, W), ROW_MAJOR, bfloat16.
                      If None, beta defaults to zeros.
        epsilon:      Numerical stability constant added to variance (default: 1e-5).
        memory_config: Memory configuration for the output tensor (default: DRAM interleaved).

    Returns:
        Output tensor with same shape as input, ROW_MAJOR layout, bfloat16.
    """
    _validate_input(input_tensor, gamma_tensor, beta_tensor)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output shape matches input shape exactly
    output_shape = [input_tensor.shape[i] for i in range(len(input_tensor.shape))]

    # CRITICAL: allocate_tensor_on_device requires POSITIONAL args, not keyword args
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(
        input_tensor,
        gamma_tensor,
        beta_tensor,
        output_tensor,
        epsilon=epsilon,
    )

    # NOTE: Output tensor MUST be last in the list
    tensors = [input_tensor]
    if gamma_tensor is not None:
        tensors.append(gamma_tensor)
    if beta_tensor is not None:
        tensors.append(beta_tensor)
    tensors.append(output_tensor)

    return ttnn.generic_op(tensors, program_descriptor)


def _validate_input(
    input_tensor: ttnn.Tensor,
    gamma_tensor: ttnn.Tensor,
    beta_tensor: ttnn.Tensor,
) -> None:
    """Validate input tensors meet requirements."""
    if len(input_tensor.shape) < 2:
        raise ValueError("layer_norm_rm: input must have at least 2 dimensions")

    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("layer_norm_rm: input must be row-major layout")

    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError("layer_norm_rm: currently only bfloat16 is supported")

    shape = input_tensor.shape
    W = shape[-1]
    H = shape[-2]

    if W % 32 != 0:
        raise ValueError(f"layer_norm_rm: width (last dim) must be a multiple of 32, got {W}")

    if H % 32 != 0:
        raise ValueError(f"layer_norm_rm: height (second-last dim) must be a multiple of 32, got {H}")

    if gamma_tensor is not None:
        if gamma_tensor.dtype != ttnn.bfloat16:
            raise ValueError("layer_norm_rm: gamma must be bfloat16")
        if gamma_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError("layer_norm_rm: gamma must be row-major layout")
        if gamma_tensor.shape[-1] != W:
            raise ValueError(f"layer_norm_rm: gamma width {gamma_tensor.shape[-1]} must match input width {W}")

    if beta_tensor is not None:
        if beta_tensor.dtype != ttnn.bfloat16:
            raise ValueError("layer_norm_rm: beta must be bfloat16")
        if beta_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError("layer_norm_rm: beta must be row-major layout")
        if beta_tensor.shape[-1] != W:
            raise ValueError(f"layer_norm_rm: beta width {beta_tensor.shape[-1]} must match input width {W}")
