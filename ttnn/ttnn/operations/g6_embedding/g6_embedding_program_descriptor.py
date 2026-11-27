# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for g6_embedding: CBs, kernels and their arguments, derived from a Plan.

Buffer addresses travel as common runtime arguments, so a cached descriptor serves every call of
its key: bind() rewrites three words and nothing else is rebuilt (BB-07).
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_HEIGHT = 32
TILE_WIDTH = 32
TILE_BYTES = TILE_HEIGHT * TILE_WIDTH * 2  # one 16-bit tile
INDEX_SLOT_BYTES = TILE_HEIGHT * 4  # the uint32 indices of one tile-row

CB_INDEX_VALUES = 0  # reader-private: uint32 indices of the tile-rows being gathered
CB_GATHERED_ROWS = 1  # row-major table-row segments (bf16 bits as UInt16), reader -> compute
CB_OUTPUT_TILES = 2  # tilized output tiles (UInt16), compute -> writer

# Positions of the buffer addresses inside each kernel's common runtime args.
READER_INDEX_ADDR = 0
READER_WEIGHT_ADDR = 1
WRITER_OUTPUT_ADDR = 0

KERNEL_READER = 0
KERNEL_WRITER = 1
KERNEL_COMPUTE = 2


class ProgramArtifacts:
    def __init__(self, descriptor):
        self.descriptor = descriptor
        kernels = descriptor.kernels
        self._reader_common = kernels[KERNEL_READER].common_runtime_args
        self._writer_common = kernels[KERNEL_WRITER].common_runtime_args

    def bind(self, input_tensor, weight, output):
        self._reader_common[READER_INDEX_ADDR] = input_tensor.buffer_address()
        self._reader_common[READER_WEIGHT_ADDR] = weight.buffer_address()
        self._writer_common[WRITER_OUTPUT_ADDR] = output.buffer_address()


def _core_assignment(plan, grid):
    """(all_cores, [(core, first_block, num_blocks)]) over contiguous block-id ranges, row-wise."""
    _, all_cores, group_1, group_2, blocks_1, blocks_2 = ttnn.split_work_to_cores(
        grid, plan.total_blocks, row_wise=True
    )
    assignment = []
    first_block = 0
    for group, blocks in ((group_1, blocks_1), (group_2, blocks_2)):
        if blocks == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, first_block, blocks))
            first_block += blocks
    assert first_block == plan.total_blocks, f"split covers {first_block} of {plan.total_blocks} blocks"
    assert len(assignment) == plan.num_cores, f"split uses {len(assignment)} cores, plan says {plan.num_cores}"
    return all_cores, assignment


def _cb(index, data_format, page_size, num_pages, core_ranges):
    return ttnn.CBDescriptor(
        total_size=page_size * num_pages,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def create_program_artifacts(plan, input_tensor, weight, output) -> ProgramArtifacts:
    grid = input_tensor.device().compute_with_storage_grid_size()
    all_cores, assignment = _core_assignment(plan, grid)

    block_pages = plan.block_tile_rows * plan.block_col_tiles
    cbs = [
        _cb(
            CB_INDEX_VALUES,
            ttnn.uint32,
            INDEX_SLOT_BYTES,
            plan.index_depth * plan.block_tile_rows,
            all_cores,
        ),
        # UInt16 on both tilize CBs: the standard (non-fast) tilize moves the 16-bit patterns as
        # integers, so NaN payloads, -0.0 and subnormals survive (op_design.md, Mechanism caps).
        _cb(CB_GATHERED_ROWS, ttnn.uint16, TILE_BYTES, plan.rows_depth * block_pages, all_cores),
        _cb(CB_OUTPUT_TILES, ttnn.uint16, TILE_BYTES, plan.out_depth * block_pages, all_cores),
    ]

    reader_ct = [
        CB_INDEX_VALUES,
        CB_GATHERED_ROWS,
        plan.block_col_tiles,
        plan.block_tile_rows,
        plan.index_depth,
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(weight).get_compile_time_args())
    reader_common = [0] * 2 + [
        plan.tensor_token_extent,
        plan.tile_rows_per_image,
        plan.tensor_tile_rows,
        plan.col_blocks_per_row,
    ]

    writer_ct = [CB_OUTPUT_TILES, plan.block_col_tiles, plan.block_tile_rows]
    writer_ct.extend(ttnn.TensorAccessorArgs(output).get_compile_time_args())
    writer_common = [0] + [plan.tensor_col_tiles, plan.tensor_tile_rows, plan.col_blocks_per_row]

    compute_ct = [CB_GATHERED_ROWS, CB_OUTPUT_TILES, plan.block_col_tiles, plan.block_tile_rows]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    for core, first_block, num_blocks in assignment:
        reader_rt[core.x][core.y] = [first_block, num_blocks]
        writer_rt[core.x][core.y] = [first_block, num_blocks]
        compute_rt[core.x][core.y] = [num_blocks]

    kernels = [None, None, None]
    kernels[KERNEL_READER] = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_embedding_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        common_runtime_args=reader_common,
        config=ttnn.ReaderConfigDescriptor(),
    )
    kernels[KERNEL_WRITER] = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_embedding_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        common_runtime_args=writer_common,
        config=ttnn.WriterConfigDescriptor(),
    )
    kernels[KERNEL_COMPUTE] = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_embedding_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False, dst_full_sync_en=False),
    )

    descriptor = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    return ProgramArtifacts(descriptor)
