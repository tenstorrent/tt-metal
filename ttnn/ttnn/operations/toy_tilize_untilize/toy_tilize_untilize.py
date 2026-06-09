# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Toy Tilize-Untilize Op

Identity operation: reads ROW_MAJOR sticks, tilizes in compute, untilizes back,
writes ROW_MAJOR sticks. Exercises the tilize/untilize dataflow + compute helpers.
"""

import ttnn
from .toy_tilize_untilize_program_descriptor import create_program_descriptor


def toy_tilize_untilize(
    input_tensor: ttnn.Tensor,
    *,
    use_row_granularity: bool = False,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT, "Input must be ROW_MAJOR"

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(input_tensor, output_tensor, use_row_granularity=use_row_granularity)
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
