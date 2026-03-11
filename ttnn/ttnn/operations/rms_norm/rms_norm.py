# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
RMS Norm - Main Entry Point

RMS (Root Mean Square) normalization along the last dimension:
    output = input / sqrt(mean(input^2, dim=-1) + epsilon) [* gamma]

Usage:
    from ttnn.operations.rms_norm import rms_norm
    output = rms_norm(input_tensor, gamma=gamma_tensor, epsilon=1e-6)
"""

from typing import Optional

import ttnn
from .rms_norm_program_descriptor import create_program_descriptor


def rms_norm(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    epsilon: float = 1e-6,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    RMS normalization along the last dimension.

    Args:
        input_tensor: Input tensor (must be on device, rank >= 2)
        gamma: Optional per-channel scale tensor, shape (1,1,1,W) in ROW_MAJOR_LAYOUT
        epsilon: Numerical stability constant (default: 1e-6)
        memory_config: Memory configuration for output tensor (default: DRAM interleaved)

    Returns:
        Output tensor with same shape and layout as input
    """
    _validate_input(input_tensor, gamma)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output shape matches input shape
    output_shape = list(input_tensor.shape)

    # CRITICAL: allocate_tensor_on_device requires POSITIONAL args
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(input_tensor, output_tensor, gamma, epsilon)

    # Build IO tensor list: inputs first, output MUST be last
    if gamma is not None:
        return ttnn.generic_op([input_tensor, gamma, output_tensor], program_descriptor)
    else:
        return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)


def _validate_input(input_tensor: ttnn.Tensor, gamma: ttnn.Tensor = None) -> None:
    """Validate input tensor meets requirements."""
    if len(input_tensor.shape) < 2:
        raise ValueError("rms_norm: input must have at least 2 dimensions")

    if gamma is not None:
        # Gamma last dimension must match input last dimension
        if gamma.shape[-1] != input_tensor.shape[-1]:
            raise ValueError(
                f"rms_norm: gamma last dimension ({gamma.shape[-1]}) "
                f"must match input last dimension ({input_tensor.shape[-1]})"
            )
