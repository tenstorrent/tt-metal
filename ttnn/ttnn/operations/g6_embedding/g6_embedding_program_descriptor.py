# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_embedding blocking plan and ProgramDescriptor.

A block is BLOCK_TILE_ROWS tile-rows x block_col_tiles tile-columns of the result. The reader
gathers the block's table-row segments (addressed by index values it reads from DRAM itself) into
a row-major staging CB, compute tilizes them with both CBs declared UInt16 (bit passthrough), and
the writer stores whole tiles. Every capacity, trip count and arg below derives from the module
knobs and from `select_plan`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# --- Block knobs (single source of truth) ----------------------------------------------------
BLOCK_TILE_ROWS = 1  # tile-rows per block (tile_row axis extent)
INDEX_DEPTH = 2  # index slot-groups: the current block's indices + the prefetched next block's
ROWS_DEPTH = 2  # blocks resident in cb_gathered_rows
OUT_DEPTH = 2  # blocks resident in cb_output_tiles
L1_BUDGET_BYTES = 1_347_776  # BB-05: per-core L1 left to the op

TILE_HEIGHT = 32
TILE_WIDTH = 32
INDEX_ELEMENT_BYTES = 4  # uint32
STAGING_DTYPE = ttnn.uint16  # bf16 bits moved as integers: the tilize path never touches a float
STAGING_TILE_BYTES = ttnn.tile_size(STAGING_DTYPE)
INDEX_SLOT_BYTES = TILE_HEIGHT * INDEX_ELEMENT_BYTES  # indices of one tile-row

CB_INDEX_VALUES = 0
CB_GATHERED_ROWS = 1
CB_OUTPUT_TILES = 2

assert INDEX_DEPTH >= 2, "the next block's indices are prefetched while the current slot is in use"


@dataclass(frozen=True)
class Plan:
    regime: str
    col_blocks_per_row: int
    block_col_tiles: int
    block_tile_rows: int
    tensor_images: int
    tensor_token_extent: int
    tile_rows_per_image: int
    tensor_tile_rows: int
    tensor_col_tiles: int
    row_blocks: int
    total_blocks: int
    num_cores: int
    l1_bytes_per_core: int


def _divisors(n: int) -> list[int]:
    return [q for q in range(1, n + 1) if n % q == 0]


def _l1_bytes_per_core(block_col_tiles: int) -> int:
    index_bytes = INDEX_DEPTH * BLOCK_TILE_ROWS * INDEX_SLOT_BYTES
    tile_bytes = (ROWS_DEPTH + OUT_DEPTH) * BLOCK_TILE_ROWS * block_col_tiles * STAGING_TILE_BYTES
    return index_bytes + tile_bytes


def col_tile_cap() -> int:
    index_bytes = INDEX_DEPTH * BLOCK_TILE_ROWS * INDEX_SLOT_BYTES
    return (L1_BUDGET_BYTES - index_bytes) // ((ROWS_DEPTH + OUT_DEPTH) * BLOCK_TILE_ROWS * STAGING_TILE_BYTES)


def plan_for(index_shape, weight_shape, grid_x: int, grid_y: int) -> Plan:
    """The blocking plan of a call (op_design.md, Regimes): pure function of shapes and grid."""
    index_shape = tuple(int(e) for e in index_shape)
    tensor_images = index_shape[0] if len(index_shape) == 2 else 1
    tensor_token_extent = index_shape[-1]
    tile_rows_per_image = -(-tensor_token_extent // TILE_HEIGHT)
    tensor_tile_rows = tensor_images * tile_rows_per_image
    tensor_col_tiles = int(weight_shape[-1]) // TILE_WIDTH
    num_grid_cores = grid_x * grid_y

    row_blocks = -(-tensor_tile_rows // BLOCK_TILE_ROWS)
    divisors = _divisors(tensor_col_tiles)
    cap = col_tile_cap()
    q_l1 = min(q for q in divisors if tensor_col_tiles // q <= cap)
    q_fill = max(q for q in divisors if q <= max(1, num_grid_cores // row_blocks))
    col_blocks_per_row = max(q_l1, q_fill)
    block_col_tiles = tensor_col_tiles // col_blocks_per_row
    total_blocks = row_blocks * col_blocks_per_row
    return Plan(
        regime="whole_row" if col_blocks_per_row == 1 else "column_blocked",
        col_blocks_per_row=col_blocks_per_row,
        block_col_tiles=block_col_tiles,
        block_tile_rows=BLOCK_TILE_ROWS,
        tensor_images=tensor_images,
        tensor_token_extent=tensor_token_extent,
        tile_rows_per_image=tile_rows_per_image,
        tensor_tile_rows=tensor_tile_rows,
        tensor_col_tiles=tensor_col_tiles,
        row_blocks=row_blocks,
        total_blocks=total_blocks,
        num_cores=min(total_blocks, num_grid_cores),
        l1_bytes_per_core=_l1_bytes_per_core(block_col_tiles),
    )


def _split_blocks(grid, total_blocks: int):
    """(all_cores, [(core, first_block, num_blocks)]): contiguous block ranges in row-wise core order."""
    _, all_cores, group_1, group_2, blocks_1, blocks_2 = ttnn.split_work_to_cores(grid, total_blocks, row_wise=True)
    assignment = []
    first_block = 0
    for group, per_core in ((group_1, blocks_1), (group_2, blocks_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, first_block, per_core))
            first_block += per_core
    assert first_block == total_blocks
    return all_cores, assignment


class CachedProgram:
    """Program artifacts of one cache key. Holds no buffer address: `bind` writes the three
    address slots (common runtime args) of each call and returns the descriptor to dispatch."""

    def __init__(self, kernels, cbs):
        self._reader, self._writer, self._compute = kernels
        self._cbs = cbs

    def bind(self, input_tensor, weight, output) -> ttnn.ProgramDescriptor:
        self._reader.common_runtime_args = [input_tensor.buffer_address(), weight.buffer_address()]
        self._writer.common_runtime_args = [output.buffer_address()]
        return ttnn.ProgramDescriptor(
            kernels=[self._reader, self._writer, self._compute],
            semaphores=[],
            cbs=self._cbs,
        )


def build_program(input_tensor, weight, output, plan: Plan) -> CachedProgram:
    grid = input_tensor.device().compute_with_storage_grid_size()
    all_cores, assignment = _split_blocks(grid, plan.total_blocks)
    assert len(assignment) == plan.num_cores

    block_pages = plan.block_tile_rows * plan.block_col_tiles
    row_segment_bytes = plan.block_col_tiles * TILE_WIDTH * weight.element_size()

    cbs = [
        ttnn.CBDescriptor(
            total_size=INDEX_DEPTH * BLOCK_TILE_ROWS * INDEX_SLOT_BYTES,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INDEX_VALUES, data_format=ttnn.uint32, page_size=INDEX_SLOT_BYTES
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=ROWS_DEPTH * block_pages * STAGING_TILE_BYTES,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_GATHERED_ROWS, data_format=STAGING_DTYPE, page_size=STAGING_TILE_BYTES
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=OUT_DEPTH * block_pages * STAGING_TILE_BYTES,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUTPUT_TILES, data_format=STAGING_DTYPE, page_size=STAGING_TILE_BYTES
                )
            ],
        ),
    ]

    reader_ct_args = [
        CB_INDEX_VALUES,
        CB_GATHERED_ROWS,
        plan.block_col_tiles,
        plan.block_tile_rows,
        plan.col_blocks_per_row,
        INDEX_DEPTH,
        plan.tile_rows_per_image,
        plan.tensor_token_extent,
        plan.tensor_tile_rows,
        row_segment_bytes,
        INDEX_SLOT_BYTES,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(weight).get_compile_time_args())

    writer_ct_args = [
        CB_OUTPUT_TILES,
        plan.block_col_tiles,
        plan.block_tile_rows,
        plan.col_blocks_per_row,
        plan.tensor_tile_rows,
        plan.tensor_col_tiles,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output).get_compile_time_args())

    compute_ct_args = [CB_GATHERED_ROWS, CB_OUTPUT_TILES, plan.block_col_tiles, plan.block_tile_rows]

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()
    for core, first_block, num_blocks in assignment:
        reader_rt_args[core.x][core.y] = [first_block, num_blocks]
        writer_rt_args[core.x][core.y] = [first_block, num_blocks]
        compute_rt_args[core.x][core.y] = [num_blocks]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_embedding_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        common_runtime_args=[0, 0],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_embedding_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        common_runtime_args=[0],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_embedding_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False, dst_full_sync_en=False),
    )
    return CachedProgram((reader, writer, compute), cbs)
