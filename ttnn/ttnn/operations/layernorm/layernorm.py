# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
LayerNorm - Main Entry Point

Implements: y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta

Input: 2D [H, W] ROW_MAJOR bfloat16 DRAM interleaved
       W and H must be multiples of 32.
Optional gamma (weight): 1D [W] bfloat16 ROW_MAJOR
Optional beta (bias): 1D [W] bfloat16 ROW_MAJOR
Output: Same shape as input, ROW_MAJOR bfloat16

Usage:
    from ttnn.operations.layernorm import layernorm
    output = layernorm(input_tensor, gamma=gamma_tensor, beta=beta_tensor, eps=1e-5)
"""

import ttnn
from .layernorm_program_descriptor import create_program_descriptor


def layernorm(
    input_tensor: ttnn.Tensor,
    *,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    eps: float = 1e-5,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    LayerNorm operation entry point.

    Args:
        input_tensor: Input tensor 2D [H, W], ROW_MAJOR, bfloat16, on device
        gamma: Optional scale parameter 1D [W], bfloat16, on device
        beta: Optional shift parameter 1D [W], bfloat16, on device
        eps: Variance stabilizer (default 1e-5)
        memory_config: Memory configuration for output tensor (default: DRAM interleaved)

    Returns:
        Output tensor with same shape as input, ROW_MAJOR bfloat16
    """
    _validate_input(input_tensor, gamma, beta)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output shape same as input
    output_shape = list(input_tensor.shape)

    # Allocate output tensor - POSITIONAL args required
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(input_tensor, output_tensor, gamma=gamma, beta=beta, eps=eps)

    # Build io_tensors list: all inputs first, output MUST be last
    io_tensors = [input_tensor]
    if gamma is not None:
        io_tensors.append(gamma)
    if beta is not None:
        io_tensors.append(beta)
    io_tensors.append(output_tensor)

    return ttnn.generic_op(io_tensors, program_descriptor)


def _validate_input(input_tensor: ttnn.Tensor, gamma, beta) -> None:
    """Validate input tensors meet requirements."""
    shape = input_tensor.shape
    if len(shape) != 2:
        raise ValueError(f"layernorm: input must be 2D [H, W], got rank {len(shape)}")

    H = shape[0]
    W = shape[1]

    if W % 32 != 0:
        raise ValueError(f"layernorm: width must be a multiple of 32, got {W}")
    if H % 32 != 0:
        raise ValueError(f"layernorm: height must be a multiple of 32, got {H}")

    if gamma is not None:
        gamma_shape = gamma.shape
        if len(gamma_shape) != 1 or gamma_shape[0] != W:
            raise ValueError(f"layernorm: gamma must be 1D with size W={W}, got shape {list(gamma_shape)}")

    if beta is not None:
        beta_shape = beta.shape
        if len(beta_shape) != 1 or beta_shape[0] != W:
            raise ValueError(f"layernorm: beta must be 1D with size W={W}, got shape {list(beta_shape)}")
