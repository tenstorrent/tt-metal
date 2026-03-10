# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm — Row-Major Layer Normalization Entry Point

Computes per-row layer normalization on bfloat16 row-major tensors:
    output[b,h,w] = gamma[w] * (x[b,h,w] - mean[b,h]) / sqrt(var[b,h] + eps) + beta[w]

Gamma and beta are optional. Input must be bfloat16, ROW_MAJOR, INTERLEAVED,
with last two dimensions divisible by 32.

Usage:
    from ttnn.operations.layer_norm_rm import layer_norm_rm
    out = layer_norm_rm(x)
    out = layer_norm_rm(x, gamma, beta, epsilon=1e-5)
"""

import ttnn
from .layer_norm_rm_program_descriptor import create_program_descriptor


def layer_norm_rm(
    input: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    *,
    epsilon: float = 1e-5,
) -> ttnn.Tensor:
    """
    Apply layer normalization on a row-major bfloat16 tensor.

    Args:
        input:   Input tensor. Must be bfloat16, ROW_MAJOR, INTERLEAVED.
                 At least 2D; last two dims must be divisible by 32.
        gamma:   Optional per-feature scale. Shape (1,1,1,W), bfloat16, ROW_MAJOR.
        beta:    Optional per-feature shift. Shape (1,1,1,W), bfloat16, ROW_MAJOR.
        epsilon: Numerical stability constant (keyword-only). Default 1e-5.

    Returns:
        Output tensor with same shape, bfloat16, ROW_MAJOR, INTERLEAVED.
    """
    _validate_inputs(input, gamma, beta)

    device = input.device()
    output_shape = [input.shape[i] for i in range(len(input.shape))]

    # Allocate output: same shape/dtype/layout/memory as input
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input.dtype,
        input.layout,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    program_descriptor = create_program_descriptor(input, output, gamma, beta, epsilon)

    # Output MUST be last in the list
    tensors = [input]
    if gamma is not None:
        tensors.append(gamma)
    if beta is not None:
        tensors.append(beta)
    tensors.append(output)

    return ttnn.generic_op(tensors, program_descriptor)


def _validate_inputs(input: ttnn.Tensor, gamma, beta) -> None:
    """Validate tensor requirements."""
    # Dtype check
    if input.dtype != ttnn.bfloat16:
        raise ValueError(f"layer_norm_rm: input dtype must be bfloat16, got {input.dtype}")

    # Layout check — must NOT be tile layout
    if input.layout == ttnn.TILE_LAYOUT:
        raise ValueError("layer_norm_rm: input must be ROW_MAJOR layout, got TILE_LAYOUT")

    # Rank check
    shape = input.shape
    if len(shape) < 2:
        raise ValueError(f"layer_norm_rm: input must have at least 2 dimensions, got {len(shape)}")

    # Alignment check: last two dims divisible by 32
    H = shape[-2]
    W = shape[-1]
    if H % 32 != 0:
        raise ValueError(f"layer_norm_rm: input height (shape[-2]={H}) must be divisible by 32")
    if W % 32 != 0:
        raise ValueError(f"layer_norm_rm: input width (shape[-1]={W}) must be divisible by 32")

    # Gamma/beta width validation
    if gamma is not None:
        gW = gamma.shape[-1]
        if gW != W:
            raise ValueError(f"layer_norm_rm: gamma width {gW} does not match input width {W}")

    if beta is not None:
        bW = beta.shape[-1]
        if bW != W:
            raise ValueError(f"layer_norm_rm: beta width {bW} does not match input width {W}")
