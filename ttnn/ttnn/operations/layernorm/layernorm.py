# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
LayerNorm - Main Entry Point

Implements: y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta

Input: ND tensor (any rank), ROW_MAJOR bfloat16 DRAM interleaved
       Last dim (W) must be multiple of 32.
       Product of all other dims (H) must be multiple of 32.
Optional gamma (weight): 1D or 2D [1, W] bfloat16 ROW_MAJOR
Optional beta (bias): 1D or 2D [1, W] bfloat16 ROW_MAJOR
Output: Same shape as input, ROW_MAJOR bfloat16

Usage:
    from ttnn.operations.layernorm import layernorm
    output = layernorm(input_tensor, gamma=gamma_tensor, beta=beta_tensor, eps=1e-5)
"""

import math
import ttnn
from .layernorm_program_descriptor import create_program_descriptor


def layernorm(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    *,
    eps: float = 1e-5,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    LayerNorm operation entry point.

    Args:
        input_tensor: Input tensor (any rank), ROW_MAJOR, bfloat16, on device.
                      W (last dim) must be multiple of 32.
                      Product of all other dims must be multiple of 32.
        gamma: Optional scale parameter, bfloat16, on device
        beta: Optional shift parameter, bfloat16, on device
        eps: Variance stabilizer (default 1e-5)
        memory_config: Memory configuration for output tensor (default: DRAM interleaved)

    Returns:
        Output tensor with same shape as input, ROW_MAJOR bfloat16
    """
    original_shape = list(input_tensor.shape)
    W = original_shape[-1]
    H = 1
    for d in original_shape[:-1]:
        H *= d

    _validate_input(H, W, gamma, beta)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Reshape to 2D [H, W] for the kernel
    input_2d = ttnn.reshape(input_tensor, [H, W])

    # Allocate output tensor as 2D
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([H, W]),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(input_2d, output_tensor, gamma=gamma, beta=beta, eps=eps)

    # Build io_tensors list: all inputs first, output MUST be last
    io_tensors = [input_2d]
    if gamma is not None:
        io_tensors.append(gamma)
    if beta is not None:
        io_tensors.append(beta)
    io_tensors.append(output_tensor)

    result = ttnn.generic_op(io_tensors, program_descriptor)

    # Reshape back to original shape
    result = ttnn.reshape(result, original_shape)
    return result


def _validate_input(H: int, W: int, gamma, beta) -> None:
    """Validate input dimensions meet requirements."""
    if W % 32 != 0:
        raise ValueError(f"layernorm: width must be a multiple of 32, got {W}")
    if H % 32 != 0:
        raise ValueError(f"layernorm: height must be a multiple of 32, got {H}")
