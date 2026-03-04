# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Layer normalization on row-major interleaved tensors.

Computes:
    mean[b,c,h] = sum(input[b,c,h,:]) / W
    centered[b,c,h,w] = input[b,c,h,w] - mean[b,c,h]
    var[b,c,h] = sum(centered[b,c,h,:]^2) / W
    inv_std[b,c,h] = rsqrt(var[b,c,h] + epsilon)
    output[b,c,h,w] = gamma[w] * centered[b,c,h,w] * inv_std[b,c,h] + beta[w]

Inputs:
    input_tensor: RM interleaved bfloat16, rank >= 2, last 2 dims tile-aligned
    gamma: RM interleaved bfloat16 (1, 1, 1, W)
    beta:  RM interleaved bfloat16 (1, 1, 1, W)
    epsilon: float scalar (default 1e-5)

Output:
    Same shape as input, RM interleaved bfloat16
"""

import ttnn
from .layer_norm_rm_program_descriptor import create_program_descriptor


def layer_norm_rm(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor,
    beta: ttnn.Tensor,
    epsilon: float = 1e-5,
    *,
    memory_config: ttnn.MemoryConfig = None,
    bisect_phase: int = 99,
) -> ttnn.Tensor:
    """
    Layer normalization on row-major interleaved tensors.

    Args:
        input_tensor: Input tensor (ROW_MAJOR, INTERLEAVED, bfloat16, rank>=2,
                      last two dims multiples of 32)
        gamma: Affine scale parameter, shape (1,1,1,W)
        beta:  Affine shift parameter, shape (1,1,1,W)
        epsilon: Numerical stability constant (default 1e-5)
        memory_config: Output memory config (default DRAM_MEMORY_CONFIG)

    Returns:
        Output tensor with same shape as input
    """
    _validate_inputs(input_tensor, gamma, beta)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output shape is identical to input shape
    output_shape = [input_tensor.shape[i] for i in range(len(input_tensor.shape))]

    # Allocate output tensor on device (positional args required)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(
        input_tensor, gamma, beta, output_tensor, epsilon, bisect_phase=bisect_phase
    )

    # Output MUST be last in the list
    return ttnn.generic_op([input_tensor, gamma, beta, output_tensor], program_descriptor)


def _validate_inputs(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor,
    beta: ttnn.Tensor,
) -> None:
    """Validate all input tensors meet requirements."""
    # Layout checks
    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("layer_norm_rm: Input must be row-major")
    if gamma.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("layer_norm_rm: Gamma must be row-major")
    if beta.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("layer_norm_rm: Beta must be row-major")

    # Dtype checks
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError("layer_norm_rm: Input must be bfloat16")
    if gamma.dtype != ttnn.bfloat16:
        raise ValueError("layer_norm_rm: Gamma must be bfloat16")
    if beta.dtype != ttnn.bfloat16:
        raise ValueError("layer_norm_rm: Beta must be bfloat16")

    # Rank check
    if len(input_tensor.shape) < 2:
        raise ValueError("layer_norm_rm: Input must be at least 2D")

    # Tile-alignment checks on last two dims
    rank = len(input_tensor.shape)
    W = input_tensor.shape[rank - 1]
    H = input_tensor.shape[rank - 2]
    if W % 32 != 0:
        raise ValueError(f"layer_norm_rm: W={W} must be tile-aligned (multiple of 32)")
    if H % 32 != 0:
        raise ValueError(f"layer_norm_rm: H={H} must be tile-aligned (multiple of 32)")
