# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
rms_norm — main entry point.

Per the operation design (op_design.md):
  - Layout: ROW_MAJOR_LAYOUT or TILE_LAYOUT
  - Dtype:  bfloat16 or float32
  - Gamma:  Optional, shape (1, 1, 1, W), ROW_MAJOR_LAYOUT
  - Epsilon: float (default 1e-6)

Mathematical definition:
    output[..., i, j] = input[..., i, j] / sqrt(mean(input[..., i, :]^2) + epsilon) * gamma[j]
"""

from typing import Optional

import ttnn

from .rms_norm_program_descriptor import create_program_descriptor


_SUPPORTED_INPUT_DTYPES = (ttnn.bfloat16, ttnn.float32)


def rms_norm(
    input_tensor: ttnn.Tensor,
    *,
    gamma: Optional[ttnn.Tensor] = None,
    epsilon: float = 1e-6,
) -> ttnn.Tensor:
    """RMSNorm along the last dimension.

    Args:
        input_tensor: Input tensor on device (rank >= 2). Dtype must be bfloat16 or float32.
                     Layout may be ROW_MAJOR_LAYOUT or TILE_LAYOUT (TILE requires H,W divisible by 32).
        gamma: Optional learnable scale, shape (1, 1, 1, W), ROW_MAJOR_LAYOUT.
        epsilon: Numerical stabilizer added inside the sqrt.

    Returns:
        Output tensor with same shape, dtype, and layout as input_tensor.
    """
    _validate_input(input_tensor, gamma)

    device = input_tensor.device()

    # Output: same shape, dtype, layout, memory_config as input.
    output_shape = list(input_tensor.shape)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    program_descriptor = create_program_descriptor(input_tensor, gamma, output_tensor, epsilon=epsilon)

    io_tensors = [input_tensor]
    if gamma is not None:
        io_tensors.append(gamma)
    io_tensors.append(output_tensor)

    return ttnn.generic_op(io_tensors, program_descriptor)


def _validate_input(input_tensor: ttnn.Tensor, gamma: Optional[ttnn.Tensor]) -> None:
    """Python-side validation per op_design.md."""
    shape = list(input_tensor.shape)
    if len(shape) < 2:
        raise ValueError("rms_norm: input must have at least 2 dimensions")

    if input_tensor.dtype not in _SUPPORTED_INPUT_DTYPES:
        raise RuntimeError("rms_norm: only bfloat16 and float32 inputs are supported")

    if input_tensor.layout == ttnn.TILE_LAYOUT:
        H, W = shape[-2], shape[-1]
        if H % 32 != 0 or W % 32 != 0:
            raise RuntimeError("rms_norm: TILE_LAYOUT input requires H and W divisible by 32")

    if gamma is not None:
        gamma_shape = list(gamma.shape)
        if gamma_shape[-1] != shape[-1]:
            raise ValueError(
                f"rms_norm: gamma last dim must match input last dim "
                f"(got gamma {gamma_shape[-1]} vs input {shape[-1]})"
            )
