# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
LayerNorm - Main Entry Point

Implements: y = gamma * (x - E[x]) / sqrt(Var[x] + eps) + beta

Normalizes each row (last dimension) independently.

Usage:
    from ttnn.operations.layer_norm import layer_norm
    output = layer_norm(input_tensor, gamma=gamma_tt, beta=beta_tt, eps=1e-5)
"""

import ttnn
from .layer_norm_program_descriptor import create_program_descriptor


def layer_norm(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    eps: float = 1e-5,
    *,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    LayerNorm operation: y = gamma * (x - E[x]) / sqrt(Var[x] + eps) + beta

    Args:
        input_tensor: Input tensor. Must be BFLOAT16, TILE_LAYOUT, on device.
                      Shape [1,1,N,W] or [N,W] where W is a multiple of 32.
        gamma: Optional per-element scale tensor, shape [1,1,1,W] or [1,W].
               Must be BFLOAT16, TILE_LAYOUT, on device.
        beta: Optional per-element bias tensor, shape [1,1,1,W] or [1,W].
              Must be BFLOAT16, TILE_LAYOUT, on device.
        eps: Variance stability constant (default: 1e-5).
        memory_config: Memory config for output tensor (default: DRAM interleaved).

    Returns:
        Normalized output tensor with same shape, dtype, and layout as input.
    """
    _validate_input(input_tensor, gamma, beta)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output has same shape as input
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
        eps=eps,
    )

    # Build io_tensors list: [input, (gamma,) (beta,) output]
    # Output MUST be last
    io_tensors = [input_tensor]
    if gamma is not None:
        io_tensors.append(gamma)
    if beta is not None:
        io_tensors.append(beta)
    io_tensors.append(output_tensor)

    return ttnn.generic_op(io_tensors, program_descriptor)


def _validate_input(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor,
    beta: ttnn.Tensor,
) -> None:
    """Validate input tensors meet requirements."""
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"layer_norm: input must be TILE_LAYOUT, got {input_tensor.layout}")
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError(f"layer_norm: input must be BFLOAT16, got {input_tensor.dtype}")
    if len(input_tensor.shape) < 2:
        raise ValueError(f"layer_norm: input must have at least 2 dimensions, got {len(input_tensor.shape)}")

    # Width must be multiple of 32
    W = input_tensor.shape[-1]
    if W % 32 != 0:
        raise ValueError(f"layer_norm: width (last dim) must be a multiple of 32, got {W}")

    if gamma is not None:
        if gamma.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"layer_norm: gamma must be TILE_LAYOUT, got {gamma.layout}")
        if gamma.dtype != ttnn.bfloat16:
            raise ValueError(f"layer_norm: gamma must be BFLOAT16, got {gamma.dtype}")

    if beta is not None:
        if beta.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"layer_norm: beta must be TILE_LAYOUT, got {beta.layout}")
        if beta.dtype != ttnn.bfloat16:
            raise ValueError(f"layer_norm: beta must be BFLOAT16, got {beta.dtype}")
