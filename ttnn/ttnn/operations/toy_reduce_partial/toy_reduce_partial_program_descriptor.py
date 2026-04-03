# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Program descriptor for toy_reduce_partial.

A single set of kernels handles both REDUCE_ROW and REDUCE_COL via a
compile-time arg (REDUCE_ROW_MODE). The partial dimension (partial_w or
partial_h) is also passed as a compile-time arg so the reader and compute
kernels can statically select between single and dual scaler tiles.
"""

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    reduce_row: bool,
) -> ttnn.ProgramDescriptor:
    input_shape = list(input_tensor.shape)
    origin_W = input_shape[-1]
    origin_H = input_shape[-2]

    Wt = (origin_W + TILE_DIM - 1) // TILE_DIM
    Ht = (origin_H + TILE_DIM - 1) // TILE_DIM
    NC = 1
    for d in input_shape[:-2]:
        NC *= d

    # Partial amount depends on which dimension is being reduced
    if reduce_row:
        partial = origin_W % TILE_DIM
    else:
        partial = origin_H % TILE_DIM
    has_partial = 1 if partial > 0 else 0

    input_page_size = input_tensor.buffer_page_size()
    input_num_pages = input_tensor.buffer_num_pages()
    output_page_size = output_tensor.buffer_page_size()
    output_num_pages = output_tensor.buffer_num_pages()

    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # --- Circular Buffers ---
    CB_IN = 0
    CB_SCALER = 2
    CB_OUT = 16

    num_scaler_tiles = 2 if has_partial else 1
    scaler_tile_size = ttnn.tile_size(ttnn.bfloat16)

    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * input_page_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_IN, data_format=input_tensor.dtype, page_size=input_page_size)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=num_scaler_tiles * scaler_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_SCALER, data_format=ttnn.bfloat16, page_size=scaler_tile_size)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=2 * output_page_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUT, data_format=output_tensor.dtype, page_size=output_page_size
                )
            ],
        ),
    ]

    # --- Reader ---
    scaler_float_bits = struct.unpack("I", struct.pack("f", 1.0))[0]

    reduce_row_mode = 1 if reduce_row else 0

    reader_ct_args = [
        input_num_pages,
        scaler_float_bits,
        has_partial,
        partial if has_partial else TILE_DIM,
        reduce_row_mode,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [input_tensor.buffer_address(), 0]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Writer ---
    writer_ct_args = [output_num_pages]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [output_tensor.buffer_address(), 0]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # --- Compute ---
    compute_ct_args = [Ht, Wt, NC, has_partial, reduce_row_mode]

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
        cbs=cbs,
    )
