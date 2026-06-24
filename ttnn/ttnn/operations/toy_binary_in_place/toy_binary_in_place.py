# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Toy binary add — demonstrates both add_in_place and regular add compute helpers.

Supports in-place mode (modifies A's CB) and normal mode (writes to separate output CB).
Both modes support all broadcast dimensions: NONE, ROW, COL, SCALAR.
"""

import ttnn
from .toy_binary_in_place_program_descriptor import create_program_descriptor


def toy_binary_in_place(
    input_a: ttnn.Tensor,
    input_b: ttnn.Tensor,
    *,
    broadcast_mode: str = "none",
    in_place: bool = True,
    op: str = "add",
    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Binary operation with optional in-place mode.

    Args:
        input_a: Input tensor A on device (TILE_LAYOUT).
        input_b: Input tensor B on device (TILE_LAYOUT). Ignored for op="square".
        broadcast_mode: "none", "row", "col", or "scalar".
        in_place: If True, use in-place helper. If False, use regular helper.
        op: "add", "sub", "mul", or "square".
        math_fidelity: Math fidelity for MUL/SQUARE (default: HiFi4). Ignored for ADD/SUB.
        memory_config: Output memory config (default: DRAM interleaved).
    """
    device = input_a.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_a.shape)),
        input_a.dtype,
        ttnn.TILE_LAYOUT,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(
        input_a,
        input_b,
        output_tensor,
        broadcast_mode=broadcast_mode,
        in_place=in_place,
        op=op,
        math_fidelity=math_fidelity,
    )
    return ttnn.generic_op([input_a, input_b, output_tensor], program_descriptor)
