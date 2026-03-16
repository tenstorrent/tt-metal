# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
matmul_sc - Single-Core Tiled Matrix Multiplication

Entry point for C = A x B where:
  - A: [M, K] rank-2 bfloat16 TILE_LAYOUT interleaved
  - B: [K, N] rank-2 bfloat16 TILE_LAYOUT interleaved
  - C: [M, N] rank-2 bfloat16 TILE_LAYOUT interleaved

All dimensions must be multiples of 32.
"""

import ttnn
from .matmul_sc_program_descriptor import create_program_descriptor


def matmul_sc(
    input_a: ttnn.Tensor,
    input_b: ttnn.Tensor,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Single-core tiled matrix multiplication: C = A x B.

    Args:
        input_a: Matrix A, shape [M, K], rank-2 bfloat16 TILE_LAYOUT (on device)
        input_b: Matrix B, shape [K, N], rank-2 bfloat16 TILE_LAYOUT (on device)
        memory_config: Output memory config (default: DRAM interleaved)

    Returns:
        Output tensor C, shape [M, N]
    """
    _validate_inputs(input_a, input_b)

    device = input_a.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output shape: [M, N]
    M = input_a.shape[0]
    N = input_b.shape[1]
    output_shape = [M, N]

    # CRITICAL: allocate_tensor_on_device requires POSITIONAL args
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_a.dtype,
        input_a.layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(input_a, input_b, output_tensor)

    # NOTE: Output tensor MUST be last in the list
    return ttnn.generic_op([input_a, input_b, output_tensor], program_descriptor)


def _validate_inputs(input_a: ttnn.Tensor, input_b: ttnn.Tensor) -> None:
    """Validate input tensors meet requirements."""
    if len(input_a.shape) != 2:
        raise ValueError("matmul_sc: inputs must be rank-2")
    if len(input_b.shape) != 2:
        raise ValueError("matmul_sc: inputs must be rank-2")

    if input_a.dtype != ttnn.bfloat16:
        raise ValueError("matmul_sc: inputs must be bfloat16")
    if input_b.dtype != ttnn.bfloat16:
        raise ValueError("matmul_sc: inputs must be bfloat16")

    if input_a.layout != ttnn.TILE_LAYOUT:
        raise ValueError("matmul_sc: inputs must be tiled")
    if input_b.layout != ttnn.TILE_LAYOUT:
        raise ValueError("matmul_sc: inputs must be tiled")

    # Inner dimensions must match: A.shape[1] == B.shape[0]
    if input_a.shape[1] != input_b.shape[0]:
        raise ValueError(
            f"matmul_sc: inner dimensions must match, got A.shape[1]={input_a.shape[1]} "
            f"vs B.shape[0]={input_b.shape[0]}"
        )

    # All dims must be divisible by 32
    for dim in [input_a.shape[0], input_a.shape[1], input_b.shape[1]]:
        if dim % 32 != 0:
            raise ValueError("matmul_sc: dimensions must be multiples of 32")
