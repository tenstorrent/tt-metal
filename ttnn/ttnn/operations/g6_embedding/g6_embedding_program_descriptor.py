# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_embedding: plan selection and ProgramDescriptor.

Units are (token tile row, column piece) pairs numbered piece-major; each core runs a contiguous unit
range in blocks of block_height_tiles units. select_plan() is the single source of every block knob;
the descriptor derives CB sizes, loop counts and kernel args from the plan alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_HEIGHT = 32
L1_BOUND_BYTES = 1347776  # BB-05: L1 the model leaves to the op on every core
TILE_BYTES = ttnn.tile_size(ttnn.uint16)  # staged rows and tiles are UInt16-labelled bf16 bits
INDEX_ROW_BYTES = TILE_HEIGHT * 4  # one unit's 32 uint32 index values
ROWS_DEPTH_BLOCKS = 2  # cb_row_staging: gather block k+1 while block k is tilized
TILES_DEPTH_ROWS = 2  # cb_out_tiles: pack tile row r+1 while tile row r is written
INDEX_SLOTS = 1  # cb_index_slots: block k's values are loaded before block k+1's land
UNIT_OVERHEAD_TILES = 1  # per-unit fixed cost in tile-equivalents, for the piece_count cost model
STAGGER_WRITES = True  # rotate each core's first written tile column

REGIME = "gather_tilize"

CB_ROW_STAGING = 0
CB_INDEX_SLOTS = 1
CB_OUT_TILES = 2


@dataclass(frozen=True)
class CoreAssignment:
    core: tuple  # (x, y)
    unit_start: int
    core_units: int
    num_blocks: int
    core_linear_idx: int


@dataclass(frozen=True)
class G6EmbeddingPlan:
    regime: str
    tokens_per_image: int
    tiles_per_image: int
    tensor_tile_rows: int
    tensor_d_tiles: int
    piece_count: int
    block_width_tiles: int
    block_height_tiles: int
    num_units: int
    num_cores: int
    core_ranges: ttnn.CoreRangeSet
    units_per_core: tuple  # (group 1, group 2)
    assignments: tuple  # CoreAssignment per used core, in unit order
    l1_bytes_per_core: int

    @property
    def roles(self):
        """Every used core runs the same three kernels."""
        return {a.core: "gather" for a in self.assignments}

    @property
    def last_tile_valid_rows(self):
        return self.tokens_per_image - TILE_HEIGHT * (self.tiles_per_image - 1)


def _ceil_div(a, b):
    return -(-a // b)


def _grid_coord(grid_size):
    if isinstance(grid_size, ttnn.CoreCoord):
        return grid_size
    x, y = grid_size
    return ttnn.CoreCoord(x, y)


def _divisors(n):
    return [p for p in range(1, n + 1) if n % p == 0]


def l1_bytes_per_core(block_width_tiles, block_height_tiles):
    return (
        TILE_BYTES * block_width_tiles * (ROWS_DEPTH_BLOCKS * block_height_tiles + TILES_DEPTH_ROWS)
        + INDEX_SLOTS * INDEX_ROW_BYTES * block_height_tiles
    )


def select_plan(index_shape, weight_shape, grid_size) -> G6EmbeddingPlan:
    """Host-only, deterministic plan for one call shape on one grid (BB-08)."""
    index_shape = tuple(int(e) for e in index_shape)
    weight_shape = tuple(int(e) for e in weight_shape)
    grid = _grid_coord(grid_size)

    tokens_per_image = index_shape[-1]
    batch = index_shape[0] if len(index_shape) == 2 else 1
    tensor_d_tiles = weight_shape[-1] // TILE_HEIGHT
    tiles_per_image = _ceil_div(tokens_per_image, TILE_HEIGHT)
    tensor_tile_rows = batch * tiles_per_image
    grid_cores = grid.x * grid.y

    # Column pieces only where they shorten the busiest core's path; ties go to fewer pieces.
    bw_cap = (L1_BOUND_BYTES - INDEX_SLOTS * INDEX_ROW_BYTES) // ((ROWS_DEPTH_BLOCKS + TILES_DEPTH_ROWS) * TILE_BYTES)
    candidates = [p for p in _divisors(tensor_d_tiles) if tensor_d_tiles // p <= bw_cap]

    def cost(pieces):
        return _ceil_div(tensor_tile_rows * pieces, grid_cores) * (tensor_d_tiles // pieces + UNIT_OVERHEAD_TILES)

    piece_count = min(candidates, key=lambda p: (cost(p), p))
    block_width_tiles = tensor_d_tiles // piece_count
    num_units = tensor_tile_rows * piece_count

    num_cores, all_cores, group_1, group_2, units_1, units_2 = ttnn.split_work_to_cores(grid, num_units, row_wise=True)
    core_units_max = max(units_1, units_2)

    block_height_l1_cap = (L1_BOUND_BYTES - TILES_DEPTH_ROWS * block_width_tiles * TILE_BYTES) // (
        ROWS_DEPTH_BLOCKS * block_width_tiles * TILE_BYTES + INDEX_SLOTS * INDEX_ROW_BYTES
    )
    assert block_height_l1_cap >= 1  # bw <= bw_cap is the block_height_tiles = 1 fit
    num_blocks_max = _ceil_div(core_units_max, block_height_l1_cap)
    block_height_tiles = _ceil_div(core_units_max, num_blocks_max)

    assignments = []
    unit_start = 0
    for group, units in ((group_1, units_1), (group_2, units_2)):
        if units == 0:
            continue
        for core in ttnn.corerange_to_cores(group, row_wise=True):
            assignments.append(
                CoreAssignment(
                    core=(core.x, core.y),
                    unit_start=unit_start,
                    core_units=units,
                    num_blocks=_ceil_div(units, block_height_tiles),
                    core_linear_idx=len(assignments),
                )
            )
            unit_start += units
    assert unit_start == num_units and len(assignments) == num_cores

    return G6EmbeddingPlan(
        regime=REGIME,
        tokens_per_image=tokens_per_image,
        tiles_per_image=tiles_per_image,
        tensor_tile_rows=tensor_tile_rows,
        tensor_d_tiles=tensor_d_tiles,
        piece_count=piece_count,
        block_width_tiles=block_width_tiles,
        block_height_tiles=block_height_tiles,
        num_units=num_units,
        num_cores=num_cores,
        core_ranges=all_cores,
        units_per_core=(units_1, units_2),
        assignments=tuple(assignments),
        l1_bytes_per_core=l1_bytes_per_core(block_width_tiles, block_height_tiles),
    )


def _cb(index, num_pages, page_size, data_format, core_ranges):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size),
        ],
    )


class G6EmbeddingProgram:
    """A built ProgramDescriptor for one plan; bind() re-points it at a call's three buffers."""

    def __init__(self, plan: G6EmbeddingPlan, input_tensor, weight, output_tensor):
        self.plan = plan
        cores = plan.core_ranges
        bw, bh = plan.block_width_tiles, plan.block_height_tiles

        self.cbs = [
            _cb(CB_ROW_STAGING, ROWS_DEPTH_BLOCKS * bh * bw, TILE_BYTES, ttnn.uint16, cores),
            _cb(CB_INDEX_SLOTS, INDEX_SLOTS * bh, INDEX_ROW_BYTES, ttnn.uint32, cores),
            _cb(CB_OUT_TILES, TILES_DEPTH_ROWS * bw, TILE_BYTES, ttnn.uint16, cores),
        ]

        reader_ct = [
            CB_ROW_STAGING,
            CB_INDEX_SLOTS,
            bw,
            bh,
            plan.tensor_tile_rows,
            plan.tiles_per_image,
            plan.last_tile_valid_rows,
            TILE_BYTES,
            INDEX_ROW_BYTES,
        ]
        reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
        reader_ct.extend(ttnn.TensorAccessorArgs(weight).get_compile_time_args())

        writer_ct = [
            CB_OUT_TILES,
            bw,
            bh,
            plan.tensor_tile_rows,
            plan.tensor_d_tiles,
            TILE_BYTES,
            int(STAGGER_WRITES),
        ]
        writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

        compute_ct = [CB_ROW_STAGING, CB_OUT_TILES, bw, bh]

        reader_rt = ttnn.RuntimeArgs()
        writer_rt = ttnn.RuntimeArgs()
        compute_rt = ttnn.RuntimeArgs()
        for a in plan.assignments:
            x, y = a.core
            reader_rt[x][y] = [a.unit_start, a.core_units, a.num_blocks]
            writer_rt[x][y] = [a.unit_start, a.core_units, a.num_blocks, a.core_linear_idx]
            compute_rt[x][y] = [a.num_blocks]

        self.reader = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "g6_embedding_reader.cpp"),
            core_ranges=cores,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            common_runtime_args=[input_tensor.buffer_address(), weight.buffer_address()],
            config=ttnn.ReaderConfigDescriptor(),
        )
        self.writer = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "g6_embedding_writer.cpp"),
            core_ranges=cores,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            common_runtime_args=[output_tensor.buffer_address()],
            config=ttnn.WriterConfigDescriptor(),
        )
        # 16-bit DEST and the default unpack-to-dest mode keep the UInt16 tilize path bit-opaque.
        self.compute = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "g6_embedding_compute.cpp"),
            core_ranges=cores,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False, dst_full_sync_en=False),
        )
        self.descriptor = ttnn.ProgramDescriptor(
            kernels=[self.reader, self.writer, self.compute],
            semaphores=[],
            cbs=self.cbs,
        )

    def bind(self, input_tensor, weight, output_tensor) -> ttnn.ProgramDescriptor:
        """Only the buffer addresses differ between calls of one cache entry; they are common runtime
        args, which the device program cache does not hash."""
        self.reader.common_runtime_args = [input_tensor.buffer_address(), weight.buffer_address()]
        self.writer.common_runtime_args = [output_tensor.buffer_address()]
        # ProgramDescriptor.kernels holds copies of the descriptors: hand it the updated ones.
        self.descriptor.kernels = [self.reader, self.writer, self.compute]
        return self.descriptor
