# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Entry Point

Layer normalization on row-major interleaved tensors.

Mathematical definition:
    mean[b,h] = (1/W) * sum_w(input[b,h,w])
    centered[b,h,w] = input[b,h,w] - mean[b,h]
    var[b,h] = (1/W) * sum_w(centered[b,h,w]^2)
    output[b,h,w] = centered[b,h,w] / sqrt(var[b,h] + eps)
    if gamma: output *= gamma[w]
    if beta:  output += beta[w]

Call patterns:
    layer_norm_rm(input)
    layer_norm_rm(input, gamma)
    layer_norm_rm(input, gamma, beta)
    layer_norm_rm(input, gamma, beta, epsilon=1e-5)
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
        input_tensor: Input tensor. Must be bfloat16, ROW_MAJOR layout, interleaved.
                      Rank >= 2, width and height must be multiples of 32.
        gamma: Optional scale parameter. Shape (1,1,1,W), bfloat16, ROW_MAJOR.
        beta: Optional shift parameter. Shape (1,1,1,W), bfloat16, ROW_MAJOR.
        epsilon: Numerical stability constant (default 1e-5).
        memory_config: Output memory configuration (default: DRAM interleaved).

    Returns:
        Output tensor, same shape as input, bfloat16, ROW_MAJOR, interleaved.
    """
    # Validate inputs
    _validate_input(input_tensor)
    if gamma is not None:
        _validate_gamma_beta(gamma, input_tensor, "gamma")
    if beta is not None:
        _validate_gamma_beta(beta, input_tensor, "beta")

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output shape is same as input
    output_shape = list(input_tensor.shape)

    # Allocate output tensor on device (POSITIONAL args, not keyword args)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    # Create program descriptor
    program_descriptor = create_program_descriptor(input_tensor, output_tensor, gamma=gamma, beta=beta, epsilon=epsilon)

    # Build IO tensor list: inputs first, output MUST be last
    io_tensors = [input_tensor]
    if gamma is not None:
        io_tensors.append(gamma)
    if beta is not None:
        io_tensors.append(beta)
    io_tensors.append(output_tensor)

    return ttnn.generic_op(io_tensors, program_descriptor)


def _validate_input(input_tensor: ttnn.Tensor) -> None:
    """Validate input tensor meets requirements."""
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError("layer_norm_rm: Input must be bfloat16")

    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("layer_norm_rm: Input must be row-major")

    shape = input_tensor.shape
    rank = len(shape)
    if rank < 2:
        raise ValueError("layer_norm_rm: Input rank must be >= 2")

    W = shape[-1]
    H = shape[-2]
    if W % 32 != 0:
        raise ValueError(f"layer_norm_rm: Width must be tile-aligned (multiple of 32), got {W}")
    if H % 32 != 0:
        raise ValueError(f"layer_norm_rm: Height must be tile-aligned (multiple of 32), got {H}")


def _validate_gamma_beta(param_tensor: ttnn.Tensor, input_tensor: ttnn.Tensor, name: str) -> None:
    """Validate gamma or beta tensor."""
    if param_tensor.dtype != ttnn.bfloat16:
        raise ValueError(f"layer_norm_rm: {name} must be bfloat16")

    if param_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"layer_norm_rm: {name} must be row-major")

    input_W = input_tensor.shape[-1]
    param_W = param_tensor.shape[-1]
    if param_W != input_W:
        raise ValueError(f"layer_norm_rm: {name} width ({param_W}) must match input width ({input_W})")
