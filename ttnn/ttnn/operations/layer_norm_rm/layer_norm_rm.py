# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Layer Norm RM - Entry Point

Row-major layer normalization over the last dimension (W).
All tensors are ROW_MAJOR_LAYOUT; tilize/untilize happens in-kernel.

Mathematical definition:
    mean[b,h] = sum(x[b,h,:]) / W
    centered[b,h,w] = x[b,h,w] - mean[b,h]
    var[b,h] = sum(centered[b,h,:]^2) / W
    output[b,h,w] = centered[b,h,w] * rsqrt(var[b,h] + epsilon)
    if gamma: output *= gamma[w]
    if beta:  output += beta[w]
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
    Row-major layer normalization over the last dimension.

    Args:
        input_tensor: Input tensor (bfloat16, ROW_MAJOR_LAYOUT, interleaved on device).
        gamma: Optional scale tensor of shape (1,1,1,W), bfloat16, ROW_MAJOR_LAYOUT.
        beta: Optional shift tensor of shape (1,1,1,W), bfloat16, ROW_MAJOR_LAYOUT.
        epsilon: Numerical stability constant (default 1e-5).
        memory_config: Output memory config (default: DRAM interleaved).

    Returns:
        Output tensor (same shape as input, bfloat16, ROW_MAJOR_LAYOUT).
    """
    _validate_input(input_tensor, gamma, beta)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output shape: reduced for variance stage (tile-aligned W=32)
    input_shape = list(input_tensor.shape)
    output_shape = input_shape[:-1] + [32]

    # Allocate output tensor on device (positional args, ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    # Create the program descriptor
    program_descriptor = create_program_descriptor(input_tensor, output_tensor, gamma=gamma, beta=beta, epsilon=epsilon)

    # Build tensor list: inputs first, output MUST be last
    io_tensors = [input_tensor]
    if gamma is not None:
        io_tensors.append(gamma)
    if beta is not None:
        io_tensors.append(beta)
    io_tensors.append(output_tensor)

    return ttnn.generic_op(io_tensors, program_descriptor)


def _validate_input(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
) -> None:
    """Validate input tensors meet requirements."""
    # dtype must be bfloat16
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError(f"layer_norm_rm: Input must be bfloat16, got {input_tensor.dtype}")

    # layout must be ROW_MAJOR
    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"layer_norm_rm: Input must be row-major, got {input_tensor.layout}")

    # rank must be >= 2
    shape = input_tensor.shape
    rank = len(shape)
    if rank < 2:
        raise ValueError(f"layer_norm_rm: Input must be at least 2D, got {rank}D")

    W = shape[-1]
    H = shape[-2]

    # W and H must be aligned to 32
    if W % 32 != 0:
        raise ValueError(f"layer_norm_rm: Width must be aligned to 32, got W={W}")
    if H % 32 != 0:
        raise ValueError(f"layer_norm_rm: Height must be aligned to 32, got H={H}")

    # Validate gamma shape width matches
    if gamma is not None:
        gamma_W = gamma.shape[-1]
        if gamma_W != W:
            raise ValueError(f"layer_norm_rm: gamma width ({gamma_W}) must match input width ({W})")

    # Validate beta shape width matches
    if beta is not None:
        beta_W = beta.shape[-1]
        if beta_W != W:
            raise ValueError(f"layer_norm_rm: beta width ({beta_W}) must match input width ({W})")
