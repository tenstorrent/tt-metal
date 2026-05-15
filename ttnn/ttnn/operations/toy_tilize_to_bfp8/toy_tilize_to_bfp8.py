# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Toy Tilize bf16 -> bfp8 Op

ROW_MAJOR bfloat16 input -> TILE_LAYOUT bfloat8_b output via
compute_kernel_lib::tilize helper (pack-side dtype conversion).
"""

import ttnn
from .toy_tilize_to_bfp8_program_descriptor import create_program_descriptor


def toy_tilize_to_bfp8(
    input_tensor: ttnn.Tensor,
    *,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT, "Input must be ROW_MAJOR"
    assert input_tensor.dtype == ttnn.bfloat16, "Input must be bfloat16"

    h, w = input_tensor.shape[-2], input_tensor.shape[-1]
    assert h % 32 == 0 and w % 32 == 0, "Shape must be tile-aligned (multiple of 32)"

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(input_tensor, output_tensor)
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
