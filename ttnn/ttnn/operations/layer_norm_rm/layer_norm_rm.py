# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Layer Norm RM - Main Entry Point

Row-major layer normalization: normalizes each row (last dimension) independently.
All tensors are ROW_MAJOR_LAYOUT. The reader tilizes in-kernel, compute processes
in tile space, and the writer untilizes back to row-major.

Usage:
    from ttnn.operations.layer_norm_rm import layer_norm_rm
    output = layer_norm_rm(input_tensor, gamma, beta, epsilon=1e-5)
"""

import ttnn
from .layer_norm_rm_program_descriptor import create_program_descriptor


def layer_norm_rm(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    *,
    epsilon: float = 1e-5,
) -> ttnn.Tensor:
    """
    Row-major layer normalization.

    Normalizes each row (last dimension W) of the input tensor independently:
        mean = (1/W) * sum(input)
        var  = (1/W) * sum((input - mean)^2)
        output = gamma * (input - mean) / sqrt(var + epsilon) + beta

    Args:
        input_tensor: Input tensor, bfloat16, ROW_MAJOR_LAYOUT, interleaved.
                      Shape must have H and W divisible by 32.
        gamma: Optional scale tensor, shape (1,1,1,W), bfloat16, ROW_MAJOR_LAYOUT.
        beta:  Optional shift tensor, shape (1,1,1,W), bfloat16, ROW_MAJOR_LAYOUT.
        epsilon: Numerical stability constant (default 1e-5).

    Returns:
        Output tensor, same shape as input, bfloat16, ROW_MAJOR_LAYOUT, interleaved.
    """
    # Validate inputs
    _validate_input(input_tensor)
    _validate_optional_param(gamma, input_tensor, "gamma")
    _validate_optional_param(beta, input_tensor, "beta")

    device = input_tensor.device()

    # Output shape is same as input
    output_shape = list(input_tensor.shape)

    # Output is ROW_MAJOR_LAYOUT in DRAM (same as input)
    # CRITICAL: allocate_tensor_on_device requires POSITIONAL args
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    # Create program descriptor
    program_descriptor = create_program_descriptor(input_tensor, output_tensor, gamma=gamma, beta=beta, epsilon=epsilon)

    # Build IO tensor list: all inputs first, output LAST
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
        raise ValueError(f"layer_norm_rm: input dtype must be bfloat16, got {input_tensor.dtype}")

    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"layer_norm_rm: input must be ROW_MAJOR_LAYOUT, got {input_tensor.layout}")

    shape = input_tensor.shape
    if len(shape) < 2:
        raise ValueError("layer_norm_rm: input must have at least 2 dimensions")

    H = shape[-2]
    W = shape[-1]
    if H % 32 != 0:
        raise ValueError(f"layer_norm_rm: input height must be divisible by 32, got H={H}")
    if W % 32 != 0:
        raise ValueError(f"layer_norm_rm: input width must be divisible by 32, got W={W}")


def _validate_optional_param(param: ttnn.Tensor, input_tensor: ttnn.Tensor, name: str) -> None:
    """Validate optional gamma or beta tensor."""
    if param is None:
        return

    if param.dtype != ttnn.bfloat16:
        raise ValueError(f"layer_norm_rm: {name} dtype must be bfloat16, got {param.dtype}")

    if param.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"layer_norm_rm: {name} must be ROW_MAJOR_LAYOUT, got {param.layout}")

    input_W = input_tensor.shape[-1]
    param_W = param.shape[-1]
    if param_W != input_W:
        raise ValueError(f"layer_norm_rm: {name} width ({param_W}) must match input width ({input_W})")
