# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Toy Tilize-Untilize - Program Descriptor

Flow: Reader (sticks→CB0) → Compute (tilize CB0→CB24, untilize CB24→CB16) → Writer (CB16→sticks)
"""

from pathlib import Path
import math
import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    use_row_granularity: bool = False,
) -> ttnn.ProgramDescriptor:
    # --- Tensor metadata ---
    row_bytes = input_tensor.buffer_page_size()
    total_num_rows = input_tensor.buffer_num_pages()
    elem_size = input_tensor.element_size()
    tile_size = ttnn.tile_size(input_tensor.dtype)
    tile_h = 32
    tile_w = 32
    tile_row_bytes = tile_w * elem_size
    padded_row_bytes = math.ceil(row_bytes / tile_row_bytes) * tile_row_bytes
    width_tiles = padded_row_bytes // tile_row_bytes
    num_blocks = math.ceil(total_num_rows / tile_h)

    # --- Core grid (single core) ---
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # --- Circular buffers ---
    cb_rm_in = 0  # row-major input  (reader → tilize)
    cb_tilized = 24  # tiled intermediate (tilize → untilize)
    cb_rm_out = 16  # row-major output  (untilize → writer)

    double_buffer = 2
    # Total CB size is the same for both modes (width_tiles * tile_size == tile_h * padded_row_bytes)
    cb_total_size = double_buffer * width_tiles * tile_size

    # Input CB page_size depends on granularity
    if use_row_granularity:
        input_cb_page_size = padded_row_bytes
    else:
        input_cb_page_size = tile_size

    cb_in_desc = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=cb_rm_in,
                data_format=input_tensor.dtype,
                page_size=input_cb_page_size,
            )
        ],
    )
    cb_mid_desc = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=cb_tilized,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )
    cb_out_desc = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=cb_rm_out,
                data_format=output_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # --- Reader kernel ---
    granularity_flag = 1 if use_row_granularity else 0
    reader_ct_args = [row_bytes, granularity_flag]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),
        total_num_rows,
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Writer kernel (always TILE granularity) ---
    writer_ct_args = [row_bytes]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),
        total_num_rows,
    ]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # --- Compute kernel ---
    fp32_dest = input_tensor.dtype == ttnn.float32
    compute_ct_args = [width_tiles, num_blocks, granularity_flag, total_num_rows]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32_dest),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[cb_in_desc, cb_mid_desc, cb_out_desc],
    )
