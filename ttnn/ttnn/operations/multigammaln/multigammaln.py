# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
multigammaln — entry point.

Validates the input, allocates the output tensor, builds the ProgramDescriptor,
and launches via ttnn.generic_op.

Order is hard-coded at p = 4 — there is NO `p` argument exposed at the API.
Math: lgamma(a) + lgamma(a - 0.5) + lgamma(a - 1.0) + lgamma(a - 1.5) + 3*log(pi).
"""

import ttnn

from .multigammaln_program_descriptor import create_program_descriptor


def multigammaln(
    input_tensor: ttnn.Tensor,
    *,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Compute multivariate log-gamma at p = 4, elementwise.

    Args:
        input_tensor: float32 tensor, TILE_LAYOUT, rank-4 (N, C, H, W) with H and W
            both divisible by 32. Must be on device.
        memory_config: output memory config (default: DRAM interleaved).

    Returns:
        A tensor of the same shape, dtype, and layout as the input.
    """
    _validate_input(input_tensor)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    output_shape = list(input_tensor.shape)

    # CRITICAL: allocate_tensor_on_device requires POSITIONAL args.
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(input_tensor, output_tensor)

    # Output tensor MUST be last in the I/O list.
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)


def _validate_input(input_tensor: ttnn.Tensor) -> None:
    """Phase-0 validation (matches op_design.md)."""
    if input_tensor.dtype != ttnn.float32:
        raise ValueError(f"multigammaln: only float32 is supported in Phase 0, got {input_tensor.dtype}")

    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"multigammaln: only TILE_LAYOUT is supported in Phase 0, got {input_tensor.layout}")

    shape = list(input_tensor.shape)
    if len(shape) != 4:
        raise ValueError(f"multigammaln: input must be rank-4 (N, C, H, W), got shape {shape}")

    if shape[-1] % 32 != 0 or shape[-2] % 32 != 0:
        raise ValueError(f"multigammaln: input H and W must be divisible by 32, got shape {shape}")
