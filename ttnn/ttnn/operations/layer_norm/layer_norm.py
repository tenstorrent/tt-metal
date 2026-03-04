# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
LayerNorm - Main Entry Point

Implements y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
over the last dimension (W) of a 2D row-major input tensor.

Input requirements:
  - x: 2D (N, W), ROW_MAJOR_LAYOUT, INTERLEAVED, bfloat16 or float32
  - N must be a multiple of 32 (tile height)
  - W must be a multiple of 32 (tile width)
  - weight: optional 1D (W,) tensor for gamma scaling
  - bias:   optional 1D (W,) tensor for beta shift

Output:
  - Same shape as x, ROW_MAJOR_LAYOUT, INTERLEAVED, same dtype as x

Usage:
    from ttnn.operations.layer_norm import layer_norm
    output = layer_norm(input_tensor)
    output = layer_norm(input_tensor, weight=gamma, bias=beta, eps=1e-5)
"""

import ttnn
from .layer_norm_program_descriptor import create_program_descriptor


def layer_norm(
    input_tensor: ttnn.Tensor,
    *,
    weight: ttnn.Tensor = None,
    bias: ttnn.Tensor = None,
    eps: float = 1e-5,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    LayerNorm operation entry point.

    Args:
        input_tensor: 2D input tensor (N, W), ROW_MAJOR_LAYOUT, INTERLEAVED,
                      dtype bfloat16 or float32. N and W must be multiples of 32.
        weight:       Optional 1D gamma tensor of shape (W,). ROW_MAJOR_LAYOUT.
        bias:         Optional 1D beta tensor of shape (W,). ROW_MAJOR_LAYOUT.
        eps:          Epsilon for numerical stability (default: 1e-5).
        memory_config: Memory configuration for output (default: DRAM_MEMORY_CONFIG).

    Returns:
        Output tensor of shape (N, W), ROW_MAJOR_LAYOUT, same dtype as input.
    """
    _validate_inputs(input_tensor, weight, bias)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output shape is identical to input shape
    N = input_tensor.shape[0]
    W = input_tensor.shape[1]
    output_shape = [N, W]

    # Allocate output tensor on device.
    # CRITICAL: allocate_tensor_on_device requires POSITIONAL args only.
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
        eps=eps,
    )

    # Build the tensor list. Output MUST be last.
    tensors = [input_tensor]
    if weight is not None:
        tensors.append(weight)
    if bias is not None:
        tensors.append(bias)
    tensors.append(output_tensor)

    return ttnn.generic_op(tensors, program_descriptor)


def _validate_inputs(
    input_tensor: ttnn.Tensor,
    weight: ttnn.Tensor,
    bias: ttnn.Tensor,
) -> None:
    """Validate input tensor meets layer_norm requirements."""
    if len(input_tensor.shape) != 2:
        raise ValueError(f"layer_norm: input must be 2D (N, W), got shape {list(input_tensor.shape)}")

    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"layer_norm: input must be ROW_MAJOR_LAYOUT, got {input_tensor.layout}")

    if input_tensor.dtype not in (ttnn.bfloat16, ttnn.float32):
        raise ValueError(f"layer_norm: input dtype must be bfloat16 or float32, got {input_tensor.dtype}")

    N = input_tensor.shape[0]
    W = input_tensor.shape[1]

    if N % 32 != 0:
        raise ValueError(f"layer_norm: N ({N}) must be a multiple of 32 (tile height)")

    if W % 32 != 0:
        raise ValueError(f"layer_norm: W ({W}) must be a multiple of 32 (tile width)")

    if weight is not None and weight.shape[0] != W:
        raise ValueError(f"layer_norm: weight shape {list(weight.shape)} must match W={W}")

    if bias is not None and bias.shape[0] != W:
        raise ValueError(f"layer_norm: bias shape {list(bias.shape)} must match W={W}")
