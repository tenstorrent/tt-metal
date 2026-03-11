# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Main Entry Point

Performs layer normalization on row-major interleaved tensors.
Pattern: RM sticks -> tilize in reader -> compute (mean, center, var, inv_sqrt, normalize) -> untilize in writer -> RM sticks

Usage:
    from ttnn.operations.layer_norm_rm import layer_norm_rm
    output = layer_norm_rm(input_tensor)
    output = layer_norm_rm(input_tensor, gamma)
    output = layer_norm_rm(input_tensor, gamma, beta)
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
    Layer normalization on row-major interleaved tensors.

    Args:
        input_tensor: Input tensor (ROW_MAJOR, bfloat16, interleaved DRAM, last 2 dims divisible by 32)
        gamma: Optional per-feature scale tensor, shape (1,1,1,W) ROW_MAJOR bfloat16
        beta: Optional per-feature shift tensor, shape (1,1,1,W) ROW_MAJOR bfloat16
        epsilon: Numerical stability constant (default 1e-5)
        memory_config: Output memory config (default: DRAM interleaved)

    Returns:
        Output tensor with same shape, dtype, layout as input
    """
    _validate_inputs(input_tensor, gamma, beta)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output shape is identical to input
    output_shape = list(input_tensor.shape)

    # Allocate output tensor on device (POSITIONAL args required)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    # Create program descriptor
    program_descriptor = create_program_descriptor(input_tensor, output_tensor, gamma=gamma, beta=beta, epsilon=epsilon)

    # Build io_tensors list -- output MUST be last
    io_tensors = [input_tensor]
    if gamma is not None:
        io_tensors.append(gamma)
    if beta is not None:
        io_tensors.append(beta)
    io_tensors.append(output_tensor)

    return ttnn.generic_op(io_tensors, program_descriptor)


def _validate_inputs(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
) -> None:
    """Validate input tensors meet requirements."""
    # Dtype check
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError(f"layer_norm_rm: input must be bfloat16, got {input_tensor.dtype}")

    # Layout check
    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"layer_norm_rm: input must be ROW_MAJOR_LAYOUT, got {input_tensor.layout}")

    # Rank check
    if len(input_tensor.shape) < 2:
        raise ValueError("layer_norm_rm: input must have at least 2 dimensions")

    # Alignment check (last 2 dims divisible by 32)
    shape = input_tensor.shape
    W = shape[-1]
    H = shape[-2] if len(shape) >= 2 else 1
    if W % 32 != 0:
        raise ValueError(f"layer_norm_rm: last dimension must be divisible by 32, got W={W}")
    if H % 32 != 0:
        raise ValueError(f"layer_norm_rm: second-to-last dimension must be divisible by 32, got H={H}")

    # Gamma validation
    if gamma is not None:
        if gamma.dtype != ttnn.bfloat16:
            raise ValueError(f"layer_norm_rm: gamma must be bfloat16, got {gamma.dtype}")
        if gamma.shape[-1] != W:
            raise ValueError(f"layer_norm_rm: gamma width ({gamma.shape[-1]}) must match input width ({W})")

    # Beta validation
    if beta is not None:
        if beta.dtype != ttnn.bfloat16:
            raise ValueError(f"layer_norm_rm: beta must be bfloat16, got {beta.dtype}")
        if beta.shape[-1] != W:
            raise ValueError(f"layer_norm_rm: beta width ({beta.shape[-1]}) must match input width ({W})")
