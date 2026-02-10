# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Layer Norm RM - Main Entry Point

This operation performs layer normalization on row-major tensors:
1. Computes row-wise mean and variance
2. Standardizes to zero mean and unit variance
3. Applies learned affine transformation (gamma scale, beta shift)

The operation is single-core and requires W and H to be multiples of 32.
"""

import ttnn
from .layer_norm_rm_program_descriptor import create_program_descriptor


def layer_norm_rm(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor,
    beta: ttnn.Tensor,
    *,
    epsilon: float = 1e-5,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Layer normalization for row-major tensors.

    Args:
        input_tensor: Input tensor (must be on device, ROW_MAJOR layout)
        gamma: Scale parameter tensor (shape [..., 1, W], ROW_MAJOR layout)
        beta: Shift parameter tensor (shape [..., 1, W], ROW_MAJOR layout)
        epsilon: Numerical stability constant (default: 1e-5)
        memory_config: Memory configuration for output tensor (default: DRAM interleaved)

    Returns:
        Output tensor (same shape/dtype/layout as input)
    """
    # Validate inputs
    _validate_inputs(input_tensor, gamma, beta, epsilon)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output shape same as input
    output_shape = list(input_tensor.shape)

    # CRITICAL: allocate_tensor_on_device requires POSITIONAL args, not keyword args
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(input_tensor, gamma, beta, output_tensor, epsilon)

    # NOTE: Output tensor MUST be last in the list
    return ttnn.generic_op([input_tensor, gamma, beta, output_tensor], program_descriptor)


def _validate_inputs(input_tensor: ttnn.Tensor, gamma: ttnn.Tensor, beta: ttnn.Tensor, epsilon: float) -> None:
    """Validate input tensors meet requirements."""
    # Check input is on device
    if not hasattr(input_tensor, "device") or input_tensor.device() is None:
        raise ValueError("layer_norm_rm: input must be on device")

    # Check rank
    if len(input_tensor.shape) < 2:
        raise ValueError("layer_norm_rm: input must have rank >= 2")

    # Check layout
    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("layer_norm_rm: input must be in ROW_MAJOR layout")

    # Check dtype
    if input_tensor.dtype not in [ttnn.bfloat16, ttnn.float32]:
        raise ValueError("layer_norm_rm: unsupported dtype, must be bfloat16 or float32")

    # Check dimensions are multiples of 32
    H = input_tensor.shape[-2]
    W = input_tensor.shape[-1]
    if W % 32 != 0:
        raise ValueError("layer_norm_rm: last dimension must be a multiple of 32")
    if H % 32 != 0:
        raise ValueError("layer_norm_rm: second-to-last dimension must be a multiple of 32")

    # Check gamma shape
    if gamma.shape[-1] != W:
        raise ValueError("layer_norm_rm: gamma last dim must match input last dim")
    if gamma.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("layer_norm_rm: gamma must be in ROW_MAJOR layout")
    if gamma.dtype != input_tensor.dtype:
        raise ValueError("layer_norm_rm: gamma dtype must match input dtype")

    # Check beta shape
    if beta.shape[-1] != W:
        raise ValueError("layer_norm_rm: beta last dim must match input last dim")
    if beta.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("layer_norm_rm: beta must be in ROW_MAJOR layout")
    if beta.dtype != input_tensor.dtype:
        raise ValueError("layer_norm_rm: beta dtype must match input dtype")

    # Check epsilon
    if epsilon <= 0:
        raise ValueError("layer_norm_rm: epsilon must be positive")
