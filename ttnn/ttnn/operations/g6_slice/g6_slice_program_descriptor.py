# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor of g6_slice: reader (NCRISC, NoC0) -> cb_slice_pages -> writer (BRISC, NoC1).

Pure data movement: no compute kernel. Shapes, block knobs and regime flags are compile-time; begins and
buffer addresses are common runtime args and each core's output-page range is a per-core runtime arg, so
one compiled program serves every begin of a shape class (the generic-op hash covers only RT arg counts).
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

CB_SLICE_PAGES = 0
CB_RIGHT_COL_TILE = 1

READER_KERNEL_INDEX = 0
WRITER_KERNEL_INDEX = 1
# Slot of the buffer address in each kernel's common runtime args; the plan cache rewrites only these.
ADDRESS_ARG_SLOT = 0


def _cb(buffer_index, num_pages, plan, dtype, core_ranges):
    return ttnn.CBDescriptor(
        total_size=num_pages * plan.page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=buffer_index, data_format=dtype, page_size=plan.page_bytes)
        ],
    )


def create_program_descriptor(input_tensor, output_tensor, plan):
    cores = plan.all_cores
    cbs = [_cb(CB_SLICE_PAGES, plan.cb_depth * plan.block_pages, plan, input_tensor.dtype, cores)]
    if plan.w_shift:
        # Reader-private scratch: one right-column tile per page of the block in flight.
        cbs.append(_cb(CB_RIGHT_COL_TILE, plan.block_pages, plan, input_tensor.dtype, cores))

    b4 = plan.b4
    reader_ct = [
        CB_SLICE_PAGES,
        CB_RIGHT_COL_TILE,
        plan.page_bytes,
        plan.elem_bytes,
        plan.block_pages,
        int(plan.row_runs),
        int(plan.w_shift),
        plan.tensor_in_ht_tiles,
        plan.tensor_in_wt_tiles,
        plan.tensor_out_ht_tiles,
        plan.tensor_out_wt_tiles,
        plan.N4[1],
        plan.M4[1],
        plan.rows_last_tile_row,
        plan.cols_last_tile_col,
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_common = [
        input_tensor.buffer_address(),
        b4[0],
        b4[1],
        b4[2] // 32,
        b4[2] % 32,
        b4[3] // 32,
        b4[3] % 32,
    ]

    writer_ct = [
        CB_SLICE_PAGES,
        plan.page_bytes,
        plan.block_pages,
        int(plan.row_runs),
        plan.tensor_out_ht_tiles,
        plan.tensor_out_wt_tiles,
        plan.rows_last_tile_row,
    ]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_common = [output_tensor.buffer_address(), b4[2] % 32]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    for core, start_page, pages, (o0, o1, th, tw) in plan.assignment:
        reader_rt[core.x][core.y] = [pages, o0, o1, th, tw]
        writer_rt[core.x][core.y] = [start_page, pages, th, tw]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_slice_reader.cpp"),
        core_ranges=cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        common_runtime_args=reader_common,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_slice_writer.cpp"),
        core_ranges=cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        common_runtime_args=writer_common,
        config=ttnn.WriterConfigDescriptor(),
    )
    kernels = [None, None]
    kernels[READER_KERNEL_INDEX] = reader
    kernels[WRITER_KERNEL_INDEX] = writer
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def rebind_addresses(program_descriptor, input_tensor, output_tensor):
    """Point a cached descriptor at this call's buffers; nothing else changes between hits."""
    kernels = program_descriptor.kernels
    kernels[READER_KERNEL_INDEX].common_runtime_args[ADDRESS_ARG_SLOT] = input_tensor.buffer_address()
    kernels[WRITER_KERNEL_INDEX].common_runtime_args[ADDRESS_ARG_SLOT] = output_tensor.buffer_address()
