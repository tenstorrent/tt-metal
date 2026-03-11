# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Layer Norm RM - Main Entry Point

Performs layer normalization over the last dimension of a ROW_MAJOR interleaved
tensor. Tilize/untilize happen in-kernel.

Usage:
    from ttnn.operations.layer_norm_rm import layer_norm_rm
    output = layer_norm_rm(input_tensor, gamma=gamma, beta=beta, epsilon=1e-5)
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
    Layer normalization over the last dimension (W) of a row-major tensor.

    y[b,c,h,w] = (x[b,c,h,w] - mean[b,c,h]) / sqrt(var[b,c,h] + epsilon)
    if gamma: y *= gamma[w]
    if beta:  y += beta[w]

    Args:
        input_tensor: Input tensor (bfloat16, ROW_MAJOR, interleaved DRAM, on device).
        gamma: Optional per-element scale tensor, shape (1,1,1,W), bfloat16 RM.
        beta: Optional per-element shift tensor, shape (1,1,1,W), bfloat16 RM.
        epsilon: Variance stabilizer (default 1e-5).
        memory_config: Output memory config (default: DRAM interleaved).

    Returns:
        Normalized output tensor (same shape, dtype, layout as input).
    """
    # --- Validation ---
    _validate_input(input_tensor)

    W = input_tensor.shape[-1]

    if gamma is not None:
        if gamma.shape[-1] != W:
            raise ValueError(f"layer_norm_rm: gamma width ({gamma.shape[-1]}) must match input width ({W})")

    if beta is not None:
        if beta.shape[-1] != W:
            raise ValueError(f"layer_norm_rm: beta width ({beta.shape[-1]}) must match input width ({W})")

    # --- Output allocation ---
    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    output_shape = list(input_tensor.shape)

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    # --- Program descriptor ---
    program_descriptor = create_program_descriptor(input_tensor, output_tensor, gamma=gamma, beta=beta, epsilon=epsilon)

    # --- Execute: output MUST be last in list ---
    io_tensors = [input_tensor]
    if gamma is not None:
        io_tensors.append(gamma)
    if beta is not None:
        io_tensors.append(beta)
    io_tensors.append(output_tensor)

    return ttnn.generic_op(io_tensors, program_descriptor)


def _validate_input(input_tensor: ttnn.Tensor) -> None:
    """Validate input tensor meets layer_norm_rm requirements."""
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError(f"layer_norm_rm only supports bfloat16, got {input_tensor.dtype}")

    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"layer_norm_rm requires ROW_MAJOR layout, got {input_tensor.layout}")

    ndim = len(input_tensor.shape)
    if ndim < 2:
        raise ValueError("layer_norm_rm: tensor must be at least 2D")

    H = input_tensor.shape[-2]
    W = input_tensor.shape[-1]
    if H % 32 != 0:
        raise ValueError(f"layer_norm_rm: H ({H}) must be a multiple of 32")
    if W % 32 != 0:
        raise ValueError(f"layer_norm_rm: W ({W}) must be a multiple of 32")
