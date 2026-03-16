# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Entry Point

Row-wise layer normalization on ROW_MAJOR tensors.
  mean[i]   = (1/W) * sum(input[i, :])
  var[i]    = (1/W) * sum((input[i, :] - mean[i])^2)
  output[i, j] = (input[i, j] - mean[i]) / sqrt(var[i] + eps)
  if gamma:  output[i, j] *= gamma[j]
  if beta:   output[i, j] += beta[j]

Usage:
    from ttnn.operations.layer_norm_rm import layer_norm_rm
    output = layer_norm_rm(input_tensor)
    output = layer_norm_rm(input_tensor, gamma=gamma, beta=beta, epsilon=1e-5)
"""

import ttnn
from .layer_norm_rm_program_descriptor import build_program_descriptor


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
                      Shape: (..., H, W) where H % 32 == 0 and W % 32 == 0.
        gamma: Optional per-element scale tensor, shape (1,1,1,W), RM, bfloat16.
        beta:  Optional per-element shift tensor, shape (1,1,1,W), RM, bfloat16.
        epsilon: Small constant for numerical stability (default 1e-5).

    Returns:
        Output tensor with same shape, dtype, and layout as input.
    """
    _validate_inputs(input_tensor, gamma, beta)

    device = input_tensor.device()

    # Output: same shape, bfloat16, ROW_MAJOR, interleaved DRAM
    output_shape = list(input_tensor.shape)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    program_descriptor = build_program_descriptor(input_tensor, output_tensor, gamma=gamma, beta=beta, epsilon=epsilon)

    # Build io_tensors list: all inputs first, output last.
    io_tensors = [input_tensor]
    if gamma is not None:
        io_tensors.append(gamma)
    if beta is not None:
        io_tensors.append(beta)
    io_tensors.append(output_tensor)  # output MUST be last

    return ttnn.generic_op(io_tensors, program_descriptor)


def _validate_inputs(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
) -> None:
    """Validate input tensors meet requirements."""
    # dtype check
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError("layer_norm_rm: Input must be bfloat16")

    # layout check
    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("layer_norm_rm: Input must be ROW_MAJOR_LAYOUT")

    shape = input_tensor.shape
    ndim = len(shape)
    if ndim < 2:
        raise ValueError("layer_norm_rm: Input must have at least 2 dimensions")

    W = shape[ndim - 1]
    H = shape[ndim - 2]

    if W % 32 != 0:
        raise ValueError("layer_norm_rm: Width must be tile-aligned (multiple of 32)")
    if H % 32 != 0:
        raise ValueError("layer_norm_rm: Height must be tile-aligned (multiple of 32)")

    # Gamma validation
    if gamma is not None:
        if gamma.dtype != ttnn.bfloat16:
            raise ValueError("layer_norm_rm: Gamma must be bfloat16")
        if gamma.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError("layer_norm_rm: Gamma must be ROW_MAJOR_LAYOUT")
        gamma_W = gamma.shape[len(gamma.shape) - 1]
        if gamma_W != W:
            raise RuntimeError(f"layer_norm_rm: Gamma width ({gamma_W}) must match input width ({W})")

    # Beta validation
    if beta is not None:
        if beta.dtype != ttnn.bfloat16:
            raise ValueError("layer_norm_rm: Beta must be bfloat16")
        if beta.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError("layer_norm_rm: Beta must be ROW_MAJOR_LAYOUT")
        beta_W = beta.shape[len(beta.shape) - 1]
        if beta_W != W:
            raise RuntimeError(f"layer_norm_rm: Beta width ({beta_W}) must match input width ({W})")
