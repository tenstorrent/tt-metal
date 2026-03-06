# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Main Entry Point

Layer normalization along the last dimension (W) of row-major interleaved
bfloat16 tensors, with optional affine parameters gamma and beta.

Mathematical definition:
    mean[b,h]    = (1/W) * sum(input[b,h,w] for w in 0..W-1)
    centered[b,h,w] = input[b,h,w] - mean[b,h]
    var[b,h]     = (1/W) * sum(centered[b,h,w]^2 for w in 0..W-1)
    inv_std[b,h] = rsqrt(var[b,h] + epsilon)
    output[b,h,w] = gamma[w] * centered[b,h,w] * inv_std[b,h] + beta[w]

Usage:
    from ttnn.operations.layer_norm_rm import layer_norm_rm
    output = layer_norm_rm(input_tensor)
    output = layer_norm_rm(input_tensor, gamma=gamma_tensor, beta=beta_tensor, epsilon=1e-5)
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
    Layer normalization along the last dimension for row-major interleaved tensors.

    Args:
        input_tensor: Input tensor. Must be:
            - ROW_MAJOR layout
            - bfloat16 dtype
            - At least 2D
            - Last 2 dims must be multiples of 32 (tile-aligned)
            - Interleaved DRAM memory
        gamma: Optional affine scale parameter, shape (1, 1, 1, W), bfloat16, ROW_MAJOR.
               If None, treated as all-ones.
        beta:  Optional affine bias parameter, shape (1, 1, 1, W), bfloat16, ROW_MAJOR.
               If None, treated as all-zeros.
        epsilon: Numerical stability constant added to variance. Default: 1e-5.
        memory_config: Memory configuration for output tensor. Default: DRAM interleaved.

    Returns:
        Output tensor with same shape, dtype, and layout as input.
    """
    _validate_inputs(input_tensor, gamma, beta)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output shape same as input
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
        output_tensor,
        gamma=gamma,
        beta=beta,
        epsilon=epsilon,
    )

    # NOTE: Output tensor MUST be last in the list
    tensors = [input_tensor]
    if gamma is not None:
        tensors.append(gamma)
    if beta is not None:
        tensors.append(beta)
    tensors.append(output_tensor)

    return ttnn.generic_op(tensors, program_descriptor)


def _validate_inputs(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor,
    beta: ttnn.Tensor,
) -> None:
    """Validate input tensors meet requirements."""
    # Minimum 2 dimensions
    if len(input_tensor.shape) < 2:
        raise ValueError(f"layer_norm_rm: input must have at least 2 dimensions, got {len(input_tensor.shape)}")

    # Layout must be ROW_MAJOR
    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"layer_norm_rm: input must be ROW_MAJOR layout, got {input_tensor.layout}")

    # Dtype must be bfloat16
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError(f"layer_norm_rm: input must be bfloat16 dtype, got {input_tensor.dtype}")

    # Last two dims must be multiples of 32 (tile-aligned)
    shape = input_tensor.shape
    ndim = len(shape)
    H = shape[ndim - 2]
    W = shape[ndim - 1]
    if H % 32 != 0:
        raise ValueError(f"layer_norm_rm: input height (second-to-last dim) must be a multiple of 32, got {H}")
    if W % 32 != 0:
        raise ValueError(f"layer_norm_rm: input width (last dim) must be a multiple of 32, got {W}")

    # Validate gamma shape if provided
    if gamma is not None:
        if gamma.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError(f"layer_norm_rm: gamma must be ROW_MAJOR layout, got {gamma.layout}")
        if gamma.dtype != ttnn.bfloat16:
            raise ValueError(f"layer_norm_rm: gamma must be bfloat16 dtype, got {gamma.dtype}")
        gamma_w = gamma.shape[-1]
        if gamma_w != W:
            raise ValueError(f"layer_norm_rm: gamma last dim ({gamma_w}) must match input last dim ({W})")

    # Validate beta shape if provided
    if beta is not None:
        if beta.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError(f"layer_norm_rm: beta must be ROW_MAJOR layout, got {beta.layout}")
        if beta.dtype != ttnn.bfloat16:
            raise ValueError(f"layer_norm_rm: beta must be bfloat16 dtype, got {beta.dtype}")
        beta_w = beta.shape[-1]
        if beta_w != W:
            raise ValueError(f"layer_norm_rm: beta last dim ({beta_w}) must match input last dim ({W})")
