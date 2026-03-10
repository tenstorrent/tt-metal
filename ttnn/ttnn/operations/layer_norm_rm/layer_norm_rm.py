# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Main Entry Point

Row-major layer normalization over the last dimension (W).

  mean[b,h] = sum(x[b,h,:]) / W
  centered[b,h,w] = x[b,h,w] - mean[b,h]
  var[b,h] = sum(centered[b,h,:]^2) / W
  output[b,h,w] = centered[b,h,w] * rsqrt(var[b,h] + epsilon)
  If gamma: output *= gamma[w]
  If beta:  output += beta[w]

Usage:
    from ttnn.operations.layer_norm_rm import layer_norm_rm
    output = layer_norm_rm(input_tensor)
    output = layer_norm_rm(input_tensor, gamma, beta, epsilon=1e-5)
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
        input_tensor: Input tensor (bfloat16, ROW_MAJOR, interleaved, on device).
                      Last two dims must be multiples of 32.
        gamma: Optional per-element scale tensor, shape (1,1,1,W), RM bfloat16.
        beta: Optional per-element shift tensor, shape (1,1,1,W), RM bfloat16.
        epsilon: Variance stabilizer (default 1e-5, must be > 0).
        memory_config: Memory config for output (default: DRAM interleaved).

    Returns:
        Output tensor, same shape/dtype/layout as input.
    """
    _validate_inputs(input_tensor, gamma, beta, epsilon)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output has same shape, dtype, layout as input
    output_shape = list(input_tensor.shape)

    # Allocate output tensor on device (POSITIONAL args required)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    # Build the program descriptor
    program_descriptor = create_program_descriptor(input_tensor, output_tensor, gamma=gamma, beta=beta, epsilon=epsilon)

    # Build IO tensor list: inputs first, output LAST
    io_tensors = [input_tensor]
    if gamma is not None:
        io_tensors.append(gamma)
    if beta is not None:
        io_tensors.append(beta)
    io_tensors.append(output_tensor)

    return ttnn.generic_op(io_tensors, program_descriptor)


def _validate_inputs(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor,
    beta: ttnn.Tensor,
    epsilon: float,
) -> None:
    """Validate all inputs meet the operation requirements."""

    # Dtype check
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError("layer_norm_rm: Input must be bfloat16")

    # Layout check
    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("layer_norm_rm: Input must be row-major")

    # Rank check
    shape = input_tensor.shape
    rank = len(shape)
    if rank < 2:
        raise ValueError("layer_norm_rm: Input must be at least 2D")

    # Alignment checks
    H = shape[rank - 2]
    W = shape[rank - 1]
    if H % 32 != 0:
        raise ValueError("layer_norm_rm: Height must be tile-aligned (multiple of 32)")
    if W % 32 != 0:
        raise ValueError("layer_norm_rm: Width must be tile-aligned (multiple of 32)")

    # Epsilon check
    if epsilon <= 0:
        raise ValueError("layer_norm_rm: epsilon must be > 0")

    # Gamma validation
    if gamma is not None:
        gamma_W = gamma.shape[len(gamma.shape) - 1]
        if gamma_W != W:
            raise ValueError(f"layer_norm_rm: gamma width ({gamma_W}) must match input width ({W})")

    # Beta validation
    if beta is not None:
        beta_W = beta.shape[len(beta.shape) - 1]
        if beta_W != W:
            raise ValueError(f"layer_norm_rm: beta width ({beta_W}) must match input width ({W})")
