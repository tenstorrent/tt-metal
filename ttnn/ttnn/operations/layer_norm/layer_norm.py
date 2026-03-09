# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Layer Norm - Main Entry Point

Computes layer normalization over the W dimension:
    mu[h]      = (1/W) * sum(x[h, w])
    var[h]     = (1/W) * sum((x[h, w] - mu[h])^2)
    y[h, w]    = (x[h, w] - mu[h]) / sqrt(var[h] + eps) * gamma[w] + beta[w]

Usage:
    from ttnn.operations.layer_norm import layer_norm
    output = layer_norm(input_tensor)
    output = layer_norm(input_tensor, gamma, beta, eps=1e-5)
"""

import ttnn
from .layer_norm_program_descriptor import create_program_descriptor


def layer_norm(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    *,
    eps: float = 1e-5,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Layer normalization over the W (last) dimension.

    Args:
        input_tensor: Input tensor, shape [1, 1, H, W], BFP16.
                      Accepts TILE_LAYOUT or ROW_MAJOR_LAYOUT (auto-converted).
        gamma: Optional weight tensor, shape [1, 1, 1, W], BFP16, TILE_LAYOUT.
        beta: Optional bias tensor, shape [1, 1, 1, W], BFP16, TILE_LAYOUT.
        eps: Numerical stability constant (default 1e-5).
        memory_config: Memory configuration for output tensor (default: DRAM interleaved).

    Returns:
        Output tensor, same shape as input, BFP16, TILE_LAYOUT, DRAM interleaved.
    """
    # Convert to TILE layout if needed (stage tests supply ROW_MAJOR)
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)

    _validate_input(input_tensor, gamma, beta)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output shape is same as input
    output_shape = [input_tensor.shape[i] for i in range(len(input_tensor.shape))]

    # CRITICAL: allocate_tensor_on_device requires POSITIONAL args, not keyword args
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(input_tensor, output_tensor, gamma, beta, eps)

    # Build IO tensor list: inputs first, output last
    io_tensors = [input_tensor]
    if gamma is not None:
        io_tensors.append(gamma)
    if beta is not None:
        io_tensors.append(beta)
    io_tensors.append(output_tensor)  # Output MUST be last

    return ttnn.generic_op(io_tensors, program_descriptor)


def _validate_input(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor,
    beta: ttnn.Tensor,
) -> None:
    """Validate input tensor and optional gamma/beta meet requirements."""
    shape = input_tensor.shape

    if len(shape) != 4:
        raise ValueError(f"layer_norm: input must have 4 dimensions [1, 1, H, W], got shape {list(shape)}")

    if shape[0] != 1 or shape[1] != 1:
        raise ValueError(f"layer_norm: batch dimensions must be 1, got shape {list(shape)}")

    H = shape[2]
    W = shape[3]

    if H % 32 != 0:
        raise ValueError(f"layer_norm: H={H} must be a multiple of 32")

    if W % 32 != 0:
        raise ValueError(f"layer_norm: W={W} must be a multiple of 32")

    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError(f"layer_norm: input dtype must be bfloat16, got {input_tensor.dtype}")

    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"layer_norm: input layout must be TILE_LAYOUT, got {input_tensor.layout}")

    if gamma is not None:
        g_shape = gamma.shape
        if len(g_shape) != 4 or g_shape[0] != 1 or g_shape[1] != 1 or g_shape[2] != 1 or g_shape[3] != W:
            raise ValueError(f"layer_norm: gamma shape must be [1, 1, 1, {W}], got {list(g_shape)}")
        if gamma.layout != ttnn.TILE_LAYOUT:
            raise ValueError("layer_norm: gamma must be TILE_LAYOUT")

    if beta is not None:
        b_shape = beta.shape
        if len(b_shape) != 4 or b_shape[0] != 1 or b_shape[1] != 1 or b_shape[2] != 1 or b_shape[3] != W:
            raise ValueError(f"layer_norm: beta shape must be [1, 1, 1, {W}], got {list(b_shape)}")
        if beta.layout != ttnn.TILE_LAYOUT:
            raise ValueError("layer_norm: beta must be TILE_LAYOUT")
