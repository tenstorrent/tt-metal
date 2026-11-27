# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_embedding program descriptor: regime R1 (gather -> UInt16 tilize -> tile store).

Work unit = (token tile row, width block): 32 token positions x `block_width_tiles` tile
columns. Units are split across the whole grid; every core runs fill -> gather -> tilize ->
store on its own contiguous unit range. Index values never reach the host: only the
buffer addresses change per call, and they travel as common runtime args.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---- Blocking-model knobs (op_design.md, Blocking Model); each is defined once here. ----
MAX_BLOCK_WIDTH_TILES = 80  # L1 cap on block_width_tiles (4 slots x 80 tiles x 2 KiB)
UNIT_OVERHEAD_TILES = 2  # per-unit fixed cost in tile-equivalents (perf lamp LP1)
BLOCK_TILE_ROWS = 1  # units per block (perf lamp LP7)
GATHER_DEPTH_BLOCKS = 2  # cb_gathered_rows depth in blocks (perf lamp LP6)
OUTPUT_DEPTH_BLOCKS = 2  # cb_output_tiles depth in blocks (perf lamp LP6)
INDEX_WINDOW_MAX_TILE_ROWS = 64  # cb_index_window cap; the reader refills past it

L1_BOUND_BYTES = 1_347_776  # BB-05: L1 the model leaves to the op on every core

TILE_HEIGHT = 32
TILE_WIDTH = 32
INDEX_ELEMENT_BYTES = 4  # uint32 index
INDEX_TILE_ROW_BYTES = TILE_HEIGHT * INDEX_ELEMENT_BYTES  # one tile row's 32 indices

# CB slots (arbitrary distinct indices).
CB_INDEX_WINDOW = 0
CB_GATHERED_ROWS = 1
CB_OUTPUT_TILES = 2

REGIME = "R1"
ROLE = "gather+tilize+store"


def select_block_width(tile_rows: int, width_tiles: int, num_cores: int) -> int:
    """Widest divisor of `width_tiles` (<= cap) minimising the busiest core's load."""
    candidates = [bw for bw in range(1, width_tiles + 1) if width_tiles % bw == 0 and bw <= MAX_BLOCK_WIDTH_TILES]

    def cost(bw):
        return math.ceil(tile_rows * (width_tiles // bw) / num_cores) * (bw + UNIT_OVERHEAD_TILES)

    return min(candidates, key=lambda bw: (cost(bw), -bw))


@dataclass(frozen=True)
class Plan:
    tokens_per_image: int  # S_img
    images: int  # B_img
    tile_rows_per_image: int
    tile_rows: int  # T
    width_tiles: int  # Dt
    element_bytes: int
    tile_bytes: int
    block_width_tiles: int
    width_blocks_per_row: int
    block_tile_rows: int
    total_units: int
    num_cores: int
    all_cores: object  # ttnn.CoreRangeSet
    core_units: tuple  # ((CoreCoord, first_unit, num_units), ...)
    units_per_core_group_1: int
    units_per_core_group_2: int
    index_window_tile_rows: int
    gathered_rows_pages: int
    output_tiles_pages: int
    footprint_bytes: int


def build_plan(index_shape, weight_shape, element_bytes, grid) -> Plan:
    tokens_per_image = int(index_shape[-1])
    images = int(index_shape[0]) if len(index_shape) == 2 else 1
    width = int(weight_shape[-1])

    tile_rows_per_image = math.ceil(tokens_per_image / TILE_HEIGHT)
    tile_rows = images * tile_rows_per_image
    width_tiles = width // TILE_WIDTH
    tile_bytes = TILE_HEIGHT * TILE_WIDTH * element_bytes

    num_grid_cores = grid.x * grid.y
    block_width_tiles = select_block_width(tile_rows, width_tiles, num_grid_cores)
    width_blocks_per_row = width_tiles // block_width_tiles
    total_units = tile_rows * width_blocks_per_row

    # A block is a rectangle only when one unit spans a whole tile row; it must also fit the
    # width cap and stay inside one index window.
    block_tile_rows = BLOCK_TILE_ROWS if width_blocks_per_row == 1 else 1
    block_tile_rows = max(
        1, min(block_tile_rows, MAX_BLOCK_WIDTH_TILES // block_width_tiles, INDEX_WINDOW_MAX_TILE_ROWS)
    )

    (num_cores, all_cores, core_group_1, core_group_2, units_g1, units_g2) = ttnn.split_work_to_cores(
        grid, total_units, row_wise=True
    )

    core_units = []
    next_unit = 0
    max_tile_row_span = 0
    for group, units_per_core in ((core_group_1, units_g1), (core_group_2, units_g2)):
        if units_per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            core_units.append((core, next_unit, units_per_core))
            first_tile_row = next_unit // width_blocks_per_row
            last_tile_row = (next_unit + units_per_core - 1) // width_blocks_per_row
            max_tile_row_span = max(max_tile_row_span, last_tile_row - first_tile_row + 1)
            next_unit += units_per_core
    assert next_unit == total_units, f"g6_embedding: unit split covers {next_unit} of {total_units} units"

    index_window_tile_rows = min(INDEX_WINDOW_MAX_TILE_ROWS, max_tile_row_span)
    block_pages = block_tile_rows * block_width_tiles
    gathered_rows_pages = GATHER_DEPTH_BLOCKS * block_pages
    output_tiles_pages = OUTPUT_DEPTH_BLOCKS * block_pages
    footprint_bytes = INDEX_TILE_ROW_BYTES * index_window_tile_rows + tile_bytes * (
        gathered_rows_pages + output_tiles_pages
    )

    return Plan(
        tokens_per_image=tokens_per_image,
        images=images,
        tile_rows_per_image=tile_rows_per_image,
        tile_rows=tile_rows,
        width_tiles=width_tiles,
        element_bytes=element_bytes,
        tile_bytes=tile_bytes,
        block_width_tiles=block_width_tiles,
        width_blocks_per_row=width_blocks_per_row,
        block_tile_rows=block_tile_rows,
        total_units=total_units,
        num_cores=num_cores,
        all_cores=all_cores,
        core_units=tuple(core_units),
        units_per_core_group_1=units_g1,
        units_per_core_group_2=units_g2,
        index_window_tile_rows=index_window_tile_rows,
        gathered_rows_pages=gathered_rows_pages,
        output_tiles_pages=output_tiles_pages,
        footprint_bytes=footprint_bytes,
    )


def plan_report(plan: Plan) -> dict:
    """BB-08 plan report and BB-21 role dump."""
    return {
        "regime": REGIME,
        "block_width_tiles": plan.block_width_tiles,
        "width_blocks_per_row": plan.width_blocks_per_row,
        "block_tile_rows": plan.block_tile_rows,
        "total_units": plan.total_units,
        "num_cores": plan.num_cores,
        "units_per_core": {"group_1": plan.units_per_core_group_1, "group_2": plan.units_per_core_group_2},
        "cb_pages": {
            "cb_index_window": plan.index_window_tile_rows,
            "cb_gathered_rows": plan.gathered_rows_pages,
            "cb_output_tiles": plan.output_tiles_pages,
        },
        "footprint_bytes": plan.footprint_bytes,
        "l1_bound_bytes": L1_BOUND_BYTES,
        "roles": [
            {"core": (core.x, core.y), "role": ROLE, "first_unit": first, "num_units": count}
            for core, first, count in plan.core_units
        ],
    }


def program_hash(cache_key) -> int:
    digest = hashlib.blake2b(repr(("g6_embedding", REGIME, cache_key)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _accessor_ct_args(tensor):
    args = ttnn.TensorAccessorArgs(tensor)
    # Interleaved accessors are fully compile-time; the kernels index common RT args from 0.
    assert not list(args.get_common_runtime_args()), "g6_embedding: accessor needs runtime args"
    return list(args.get_compile_time_args())


def create_program_descriptor(plan: Plan, input_tensor, weight, output_tensor, cache_key) -> ttnn.ProgramDescriptor:
    """Build the descriptor once per cache key; `set_buffer_addresses` refreshes it per call."""
    all_cores = plan.all_cores

    cbs = [
        ttnn.CBDescriptor(
            total_size=INDEX_TILE_ROW_BYTES * plan.index_window_tile_rows,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INDEX_WINDOW, data_format=ttnn.uint32, page_size=INDEX_TILE_ROW_BYTES
                )
            ],
        ),
        # UInt16 on both tilize CBs: the integer unpack/datacopy/pack path moves every bfloat16
        # bit pattern unchanged (the float path flushes subnormals and rewrites NaN).
        ttnn.CBDescriptor(
            total_size=plan.tile_bytes * plan.gathered_rows_pages,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_GATHERED_ROWS, data_format=ttnn.uint16, page_size=plan.tile_bytes
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=plan.tile_bytes * plan.output_tiles_pages,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUTPUT_TILES, data_format=ttnn.uint16, page_size=plan.tile_bytes
                )
            ],
        ),
    ]

    reader_ct_args = [
        CB_INDEX_WINDOW,
        CB_GATHERED_ROWS,
        plan.block_width_tiles,
        plan.width_blocks_per_row,
        plan.block_tile_rows,
        plan.tile_rows_per_image,
        plan.tokens_per_image,
        plan.index_window_tile_rows,
        plan.tile_bytes,
        input_tensor.buffer_aligned_page_size(),
        weight.buffer_aligned_page_size(),
    ]
    reader_ct_args.extend(_accessor_ct_args(input_tensor))
    reader_ct_args.extend(_accessor_ct_args(weight))

    writer_ct_args = [
        CB_OUTPUT_TILES,
        plan.block_width_tiles,
        plan.width_blocks_per_row,
        plan.block_tile_rows,
        plan.width_tiles,
        plan.tile_bytes,
    ]
    writer_ct_args.extend(_accessor_ct_args(output_tensor))

    compute_ct_args = [CB_GATHERED_ROWS, CB_OUTPUT_TILES, plan.block_width_tiles]

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()
    for core, first_unit, num_units in plan.core_units:
        reader_rt_args[core.x][core.y] = [first_unit, num_units]
        writer_rt_args[core.x][core.y] = [first_unit, num_units]
        compute_rt_args[core.x][core.y] = [num_units]

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
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False),
    )

    descriptor = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    descriptor.custom_program_hash = program_hash(cache_key)
    return descriptor


READER_KERNEL = 0
WRITER_KERNEL = 1


def set_buffer_addresses(descriptor: ttnn.ProgramDescriptor, input_tensor, weight, output_tensor) -> None:
    descriptor.kernels[READER_KERNEL].common_runtime_args = [input_tensor.buffer_address(), weight.buffer_address()]
    descriptor.kernels[WRITER_KERNEL].common_runtime_args = [output_tensor.buffer_address()]
