# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Layer Norm - Main Entry Point

Implements layer normalization over the last dimension W of a tiled tensor.
Supports optional affine transform via gamma (scale) and beta (bias).

Mathematical definition:
    E[x]       = (1/W) * sum(x[h, 0..W-1])        for each row h
    Var[x]     = (1/W) * sum((x[h,w] - E[x])^2)   for each row h
    output[h,w] = (x[h,w] - E[x]) / sqrt(Var[x] + eps) * gamma[w] + beta[w]

Usage:
    from ttnn.operations.layer_norm import layer_norm
    output = layer_norm(input_tensor, epsilon=1e-5, weight=gamma, bias=beta)
"""

import ttnn
from .layer_norm_program_descriptor import create_program_descriptor


def layer_norm(
    input_tensor: ttnn.Tensor,
    weight: ttnn.Tensor = None,  # gamma
    bias: ttnn.Tensor = None,  # beta
    *,
    epsilon: float = 1e-5,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Layer normalization over the last dimension W of the input tensor.

    Args:
        input_tensor: Input tensor on device. Must be TILE_LAYOUT, bfloat16,
                      4D [N, C, H, W], with W divisible by 32.
        weight:       Optional gamma (scale) tensor of shape [1, 1, 1, W] or
                      [1, 1, 32, W] in tiles (bfloat16, TILE_LAYOUT).
        bias:         Optional beta (bias) tensor of same shape constraint as weight.
        epsilon:      Numerical stability constant (default 1e-5).
        memory_config: Memory configuration for output tensor (default: DRAM interleaved).

    Returns:
        Normalized output tensor with same shape as input.
    """
    _validate_inputs(input_tensor, weight, bias)

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

    program_descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        weight=weight,
        bias=bias,
        epsilon=epsilon,
    )

    # Build io_tensors list; output MUST be last
    io_tensors = [input_tensor]
    if weight is not None:
        io_tensors.append(weight)
    if bias is not None:
        io_tensors.append(bias)
    io_tensors.append(output_tensor)

    return ttnn.generic_op(io_tensors, program_descriptor)


def _validate_inputs(
    input_tensor: ttnn.Tensor,
    weight: ttnn.Tensor,
    bias: ttnn.Tensor,
) -> None:
    """Validate input and optional affine tensors."""
    if len(input_tensor.shape) < 2:
        raise ValueError("layer_norm: input must have at least 2 dimensions")

    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("layer_norm: input must be in TILE_LAYOUT")

    W = input_tensor.shape[-1]
    if W % 32 != 0:
        raise ValueError(f"layer_norm: W={W} must be a multiple of 32 (tile-aligned)")

    if len(input_tensor.shape) < 2:
        raise ValueError("layer_norm: input must have at least 2 dimensions")

    if weight is not None:
        if weight.layout != ttnn.TILE_LAYOUT:
            raise ValueError("layer_norm: weight (gamma) must be in TILE_LAYOUT")

    if bias is not None:
        if bias.layout != ttnn.TILE_LAYOUT:
            raise ValueError("layer_norm: bias (beta) must be in TILE_LAYOUT")
