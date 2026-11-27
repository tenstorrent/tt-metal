# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for g6_embedding: CBs, kernels and their arguments, all derived from a Plan.

Buffer addresses are common runtime args (reader: idx, weight; writer: out) so a cached
descriptor is reused across calls by patching only those slots; no per-core arg depends on
an address.
"""

from __future__ import annotations

import os
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

CB_INDEX_VALUES = 0
CB_GATHERED_ROWS = 1
CB_OUTPUT_TILES = 2

# The zone build (BB-21): adds a DeviceZoneScopedN per named block operation. Off in the plain build.
ZONES_ENV = "G6_EMBEDDING_ZONES"


def _assign_blocks(grid, total_blocks: int):
    """Contiguous block-id runs in row-wise core order: (all_cores, [(core, first_block, num_blocks)])."""
    _, all_cores, group_1, group_2, blocks_1, blocks_2 = ttnn.split_work_to_cores(grid, total_blocks, row_wise=True)
    assignment = []
    first_block = 0
    for group, blocks_per_core in ((group_1, blocks_1), (group_2, blocks_2)):
        if blocks_per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, first_block, blocks_per_core))
            first_block += blocks_per_core
    return all_cores, assignment


def create_program_descriptor(input_tensor: ttnn.Tensor, weight: ttnn.Tensor, output: ttnn.Tensor, plan):
    """Returns (descriptor, reader kernel index, writer kernel index)."""
    grid = ttnn.CoreCoord(plan.grid_x, plan.grid_y)
    all_cores, assignment = _assign_blocks(grid, plan.total_blocks)

    tile_bytes = output.buffer_page_size()
    block_pages = plan.block_pages

    cbs = [
        ttnn.CBDescriptor(
            total_size=plan.index_depth * plan.index_slot_bytes,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INDEX_VALUES, data_format=ttnn.uint32, page_size=plan.index_slot_bytes
                )
            ],
        ),
        # UInt16 on both tilize CBs: the standard (integer) tilize path moves the bf16 bits
        # untouched; a Float16_b CB would select fast tilize, which rewrites NaNs and -0.0.
        ttnn.CBDescriptor(
            total_size=plan.rows_depth * block_pages * tile_bytes,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_GATHERED_ROWS, data_format=ttnn.uint16, page_size=tile_bytes)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=plan.out_depth * block_pages * tile_bytes,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_OUTPUT_TILES, data_format=ttnn.uint16, page_size=tile_bytes)
            ],
        ),
    ]

    defines = [("G6_EMBEDDING_ZONES", "1")] if os.environ.get(ZONES_ENV) == "1" else []

    reader_ct = [
        CB_INDEX_VALUES,
        CB_GATHERED_ROWS,
        plan.block_col_tiles,
        plan.block_tile_rows,
        plan.index_depth,
        plan.col_blocks_per_row,
        plan.tile_rows_per_image,
        plan.token_extent,
        plan.tensor_tile_rows,
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(weight).get_compile_time_args())

    writer_ct = [
        CB_OUTPUT_TILES,
        plan.block_col_tiles,
        plan.block_tile_rows,
        plan.col_blocks_per_row,
        plan.tensor_col_tiles,
        plan.tensor_tile_rows,
    ]
    writer_ct.extend(ttnn.TensorAccessorArgs(output).get_compile_time_args())

    compute_ct = [CB_GATHERED_ROWS, CB_OUTPUT_TILES, plan.block_col_tiles, plan.block_tile_rows]

    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for core, first_block, num_blocks in assignment:
        reader_rt[core.x][core.y] = [first_block, num_blocks]
        writer_rt[core.x][core.y] = [first_block, num_blocks]
        compute_rt[core.x][core.y] = [num_blocks]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_embedding_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        defines=defines,
        runtime_args=reader_rt,
        common_runtime_args=[input_tensor.buffer_address(), weight.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_embedding_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        defines=defines,
        runtime_args=writer_rt,
        common_runtime_args=[output.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_embedding_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct,
        defines=defines,
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False, dst_full_sync_en=False),
    )

    descriptor = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    return descriptor, 0, 1
