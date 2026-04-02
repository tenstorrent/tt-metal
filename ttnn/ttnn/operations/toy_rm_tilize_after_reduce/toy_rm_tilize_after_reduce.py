# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
toy_rm_tilize_after_reduce — capture tilized tiles after a REDUCE_ROW call.

The reduce result is intentionally discarded. The returned tensor is the
tilized input snapshot written out after reduce has already run.
"""

import ttnn

from .toy_rm_tilize_after_reduce_program_descriptor import create_program_descriptor


def toy_rm_tilize_after_reduce(
    input_tensor: ttnn.Tensor,
    *,
    memory_config: ttnn.MemoryConfig = None,
    compute_kernel_config: dict = None,
    post_tilize_nops: int = 0,
    insert_tensix_sync: bool = False,
) -> ttnn.Tensor:
    _validate_input(input_tensor)

    if post_tilize_nops < 0:
        raise ValueError(f"toy_rm_tilize_after_reduce: post_tilize_nops must be >= 0, got {post_tilize_nops}")

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        compute_kernel_config=compute_kernel_config,
        post_tilize_nops=post_tilize_nops,
        insert_tensix_sync=insert_tensix_sync,
    )

    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)


def _validate_input(input_tensor: ttnn.Tensor) -> None:
    if len(input_tensor.shape) < 2:
        raise ValueError("toy_rm_tilize_after_reduce: input must have at least 2 dimensions")

    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"toy_rm_tilize_after_reduce: input must be ROW_MAJOR_LAYOUT, got {input_tensor.layout}")

    if input_tensor.dtype not in (ttnn.bfloat16, ttnn.float32):
        raise ValueError(f"toy_rm_tilize_after_reduce: input must be bfloat16 or float32, got {input_tensor.dtype}")

    if input_tensor.shape[-1] % 32 != 0:
        raise ValueError(f"toy_rm_tilize_after_reduce: width must be a multiple of 32, got {input_tensor.shape[-1]}")

    if input_tensor.shape[-2] % 32 != 0:
        raise ValueError(f"toy_rm_tilize_after_reduce: height must be a multiple of 32, got {input_tensor.shape[-2]}")
