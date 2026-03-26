# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from .row_sum_program_descriptor import create_program_descriptor


def row_sum(
    input_tensor: ttnn.Tensor,
    *,
    output_layout: ttnn.Layout = None,
    fp32_dest_acc_en: bool = False,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Sum each row of a 2D tensor.

    Input shape [H, W] -> Output shape [H, 32] (only column 0 is valid).

    Args:
        input_tensor: 2D input tensor on device (bfloat16 or float32)
        output_layout: Output layout (TILE_LAYOUT or ROW_MAJOR_LAYOUT).
                       Defaults to matching input layout.
        fp32_dest_acc_en: Enable FP32 destination accumulation in the reduce.
        memory_config: Output memory config (default: DRAM interleaved)

    Returns:
        Output tensor of shape [H, 32]
    """
    if len(input_tensor.shape) != 2:
        raise ValueError(f"row_sum: input must be 2D, got {len(input_tensor.shape)}D")

    H, W = input_tensor.shape

    if output_layout is None:
        output_layout = input_tensor.layout

    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    output_shape = [H, 32]

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        output_layout,
        input_tensor.device(),
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(input_tensor, output_tensor, fp32_dest_acc_en)

    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
