# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Toy Tilize bf16 -> bfp8 - Program Descriptor

Flow: Reader (bf16 sticks -> cb_in) -> Compute (tilize cb_in bf16 -> cb_out bfp8)
      -> Writer (cb_out bfp8 tiles -> DRAM)
"""

from pathlib import Path
import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
) -> ttnn.ProgramDescriptor:
    row_bytes = input_tensor.buffer_page_size()
    total_num_rows = input_tensor.buffer_num_pages()

    in_tile_size = ttnn.tile_size(input_tensor.dtype)
    out_tile_size = ttnn.tile_size(output_tensor.dtype)

    tile_h = 32
    tile_w = 32
    elem_size = input_tensor.element_size()
    width_tiles = row_bytes // (tile_w * elem_size)
    num_blocks = total_num_rows // tile_h
    total_out_tiles = width_tiles * num_blocks

    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    cb_in = 0
    cb_out = 16

    double_buffer = 2
    cb_in_total = double_buffer * width_tiles * in_tile_size
    cb_out_total = double_buffer * width_tiles * out_tile_size

    cb_in_desc = ttnn.CBDescriptor(
        total_size=cb_in_total,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=cb_in,
                data_format=input_tensor.dtype,
                page_size=in_tile_size,
            )
        ],
    )
    cb_out_desc = ttnn.CBDescriptor(
        total_size=cb_out_total,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=cb_out,
                data_format=output_tensor.dtype,
                page_size=out_tile_size,
            )
        ],
    )

    reader_ct_args = [row_bytes]
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

    writer_ct_args = [total_out_tiles]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),
        0,  # start_id
    ]
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    compute_ct_args = [width_tiles, num_blocks]
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[cb_in_desc, cb_out_desc],
    )
