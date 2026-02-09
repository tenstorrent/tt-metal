# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Row Standardize - Main Entry Point

Performs per-row standardization: output = (x - mean_row) / sqrt(var_row + epsilon)

This operation:
1. Validates input tensor (ROW_MAJOR layout, bfloat16 or float32, H and W multiples of 32)
2. Allocates output tensor on device
3. Creates program descriptor for tilize->standardize->untilize pipeline
4. Executes via ttnn.generic_op
"""

import ttnn
from .row_standardize_program_descriptor import create_program_descriptor


def row_standardize(
    input_tensor: ttnn.Tensor,
    *,
    epsilon: float = 1e-5,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Row standardize operation entry point.

    Args:
        input_tensor: Input tensor (must be on device, ROW_MAJOR layout, bfloat16 or float32)
                     Shape: (..., H, W) where H and W are multiples of 32
        epsilon: Small constant for numerical stability in rsqrt (default: 1e-5)
        memory_config: Memory configuration for output tensor (default: DRAM interleaved)

    Returns:
        Output tensor with same shape, dtype, and layout as input

    Raises:
        ValueError: If input tensor doesn't meet requirements
    """
    # Validate input
    _validate_input(input_tensor)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output shape same as input
    output_shape = [input_tensor.shape[i] for i in range(len(input_tensor.shape))]

    # CRITICAL: allocate_tensor_on_device requires POSITIONAL args, not keyword args
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(input_tensor, output_tensor, epsilon)

    # NOTE: Output tensor MUST be last in the list
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)


def _validate_input(input_tensor: ttnn.Tensor) -> None:
    """Validate input tensor meets requirements."""
    shape = input_tensor.shape

    # Rank check
    if len(shape) < 2:
        raise ValueError("row_standardize: Input tensor must have rank >= 2")

    # Layout check
    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("row_standardize: Input tensor must be in ROW_MAJOR layout")

    # Dtype check
    if input_tensor.dtype not in [ttnn.bfloat16, ttnn.float32]:
        raise ValueError("row_standardize: Input tensor must be bfloat16 or float32")

    # Dimension alignment checks
    H = shape[-2]
    W = shape[-1]

    if W % 32 != 0:
        raise ValueError(f"row_standardize: Last dimension (W={W}) must be a multiple of 32 (tile width)")

    if H % 32 != 0:
        raise ValueError(f"row_standardize: Second-to-last dimension (H={H}) must be a multiple of 32 (tile height)")

    # Device check (implicit - accessing device() will fail if not on device)
    # This is sufficient for "must be on device" check
