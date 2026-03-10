# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Entry Point

Per-row layer normalization on row-major interleaved tensors.

  mean[b,h]       = sum(x[b,h,:]) / W
  centered[b,h,w] = x[b,h,w] - mean[b,h]
  var[b,h]        = sum(centered[b,h,:]^2) / W
  output[b,h,w]   = centered[b,h,w] * rsqrt(var[b,h] + eps) [* gamma[w] + beta[w]]

Usage:
    from ttnn.operations.layer_norm_rm import layer_norm_rm
    output = layer_norm_rm(input_tensor)
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
    Row-major layer normalization.

    Args:
        input_tensor: Input tensor, RM interleaved bfloat16, shape (N, C, H, W)
                      where H and W are multiples of 32.
        gamma: Optional per-element scale, RM bfloat16, shape (1, 1, 1, W).
        beta: Optional per-element bias, RM bfloat16, shape (1, 1, 1, W).
        epsilon: Numerical stability constant (default 1e-5).
        memory_config: Output memory config (default DRAM interleaved).

    Returns:
        Output tensor, same shape/layout/dtype as input.
    """
    _validate_inputs(input_tensor, gamma, beta)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output shape identical to input
    output_shape = list(input_tensor.shape)

    # Allocate output on device -- POSITIONAL args required
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    # Build program descriptor
    program_descriptor = create_program_descriptor(
        input_tensor=input_tensor,
        output_tensor=output_tensor,
        gamma=gamma,
        beta=beta,
        epsilon=epsilon,
    )

    # Assemble io_tensors list -- output MUST be last
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
    """Validate input tensor and optional gamma/beta."""
    shape = input_tensor.shape
    if len(shape) != 4:
        raise ValueError(f"layer_norm_rm: input must be 4D, got {len(shape)}D")

    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError(f"layer_norm_rm: input must be bfloat16, got {input_tensor.dtype}")

    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"layer_norm_rm: input must be ROW_MAJOR_LAYOUT, got {input_tensor.layout}")

    H = shape[2]
    W = shape[3]
    if H % 32 != 0:
        raise ValueError(f"layer_norm_rm: H must be a multiple of 32, got {H}")
    if W % 32 != 0:
        raise ValueError(f"layer_norm_rm: W must be a multiple of 32, got {W}")

    if gamma is not None:
        gw = gamma.shape[-1]
        if gw != W:
            raise ValueError(f"layer_norm_rm: gamma width {gw} != input width {W}")

    if beta is not None:
        bw = beta.shape[-1]
        if bw != W:
            raise ValueError(f"layer_norm_rm: beta width {bw} != input width {W}")
