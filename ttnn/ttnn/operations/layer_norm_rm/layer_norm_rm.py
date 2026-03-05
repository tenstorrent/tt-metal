# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Layer Normalization over Row-Major tensors

Computes layer normalization over the last dimension:
    mean[b,h] = sum(x[b,h,:]) / W
    centered[b,h,w] = x[b,h,w] - mean[b,h]
    var[b,h] = sum(centered[b,h,:]^2) / W
    inv_std[b,h] = rsqrt(var[b,h] + epsilon)
    x_norm[b,h,w] = centered[b,h,w] * inv_std[b,h]
    output[b,h,w] = gamma[w] * x_norm[b,h,w] + beta[w]

Usage:
    from ttnn.operations.layer_norm_rm import layer_norm_rm
    output = layer_norm_rm(input_tensor)
    output = layer_norm_rm(input_tensor, gamma=gamma_tt, beta=beta_tt, epsilon=1e-6)
"""

import ttnn
from .layer_norm_rm_program_descriptor import create_program_descriptor


def layer_norm_rm(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    *,
    epsilon: float = 1e-6,
) -> ttnn.Tensor:
    """
    Layer normalization over the last dimension for row-major tensors.

    Args:
        input_tensor: Input tensor. Must be bfloat16, ROW_MAJOR_LAYOUT, interleaved DRAM.
                      Rank >= 2. Last 2 dims must be multiples of 32.
        gamma:        Optional scale tensor. bfloat16, RM, shape (1,1,1,W).
        beta:         Optional bias tensor. bfloat16, RM, shape (1,1,1,W).
        epsilon:      Numerical stability constant for rsqrt. Default 1e-6.

    Returns:
        Output tensor with same shape, dtype=bfloat16, ROW_MAJOR_LAYOUT, interleaved DRAM.
    """
    _validate_inputs(input_tensor, gamma, beta)

    device = input_tensor.device()

    # Output: same shape as input, bfloat16, RM, DRAM interleaved
    output_shape = [input_tensor.shape[i] for i in range(len(input_tensor.shape))]
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    program_descriptor = create_program_descriptor(input_tensor, output_tensor, gamma, beta, epsilon)

    # Build tensor list; output MUST be last
    io_tensors = [input_tensor]
    if gamma is not None:
        io_tensors.append(gamma)
    if beta is not None:
        io_tensors.append(beta)
    io_tensors.append(output_tensor)

    return ttnn.generic_op(io_tensors, program_descriptor)


def _validate_inputs(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor,
    beta: ttnn.Tensor,
) -> None:
    """Validate all inputs meet layer_norm_rm requirements."""
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError(f"layer_norm_rm: Input must be bfloat16, got {input_tensor.dtype}")

    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"layer_norm_rm: Input must be row-major layout, got {input_tensor.layout}")

    if len(input_tensor.shape) < 2:
        raise ValueError(f"layer_norm_rm: Need at least 2 dimensions, got rank {len(input_tensor.shape)}")

    rank = len(input_tensor.shape)
    H = input_tensor.shape[rank - 2]
    W = input_tensor.shape[rank - 1]

    if H % 32 != 0:
        raise ValueError(f"layer_norm_rm: H (second-to-last dim) must be a multiple of 32, got H={H}")

    if W % 32 != 0:
        raise ValueError(f"layer_norm_rm: W (last dim) must be a multiple of 32, got W={W}")

    if gamma is not None:
        if gamma.dtype != ttnn.bfloat16:
            raise ValueError(f"layer_norm_rm: gamma must be bfloat16, got {gamma.dtype}")
        if gamma.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError(f"layer_norm_rm: gamma must be row-major layout, got {gamma.layout}")

    if beta is not None:
        if beta.dtype != ttnn.bfloat16:
            raise ValueError(f"layer_norm_rm: beta must be bfloat16, got {beta.dtype}")
        if beta.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError(f"layer_norm_rm: beta must be row-major layout, got {beta.layout}")
