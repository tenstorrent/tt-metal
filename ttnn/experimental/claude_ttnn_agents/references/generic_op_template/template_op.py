# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Template Op - Main Entry Point

This file provides the user-facing function that:
1. Validates input tensor
2. Computes output tensor shape
3. Allocates output tensor on device
4. Creates the program descriptor
5. Launches via ttnn.generic_op

Usage:
    from ttnn.operations.<op_name> import <op_name>
    output = <op_name>(input_tensor)
"""

import ttnn
from .template_op_program_descriptor import create_program_descriptor


def template_op(
    input_tensor: ttnn.Tensor,
    *,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Template operation entry point.

    Args:
        input_tensor: Input tensor (must be on device)
        memory_config: Memory configuration for output tensor (default: DRAM interleaved)

    Returns:
        Output tensor
    """
    # Validate input
    _validate_input(input_tensor)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output shape same as input (modify as needed)
    output_shape = list(input_tensor.shape)

    # CRITICAL: allocate_tensor_on_device requires POSITIONAL args, not keyword args
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(input_tensor, output_tensor)

    # NOTE: Output tensor MUST be last in the list
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)


def _validate_input(input_tensor: ttnn.Tensor) -> None:
    """Validate input tensor meets requirements."""
    if len(input_tensor.shape) < 2:
        raise ValueError("template_op: input must have at least 2 dimensions")

    # Add more validation as needed:
    # - layout checks (TILE_LAYOUT vs ROW_MAJOR_LAYOUT)
    # - dtype checks (bfloat16, float32, etc.)
    # - alignment checks (divisible by 32)
