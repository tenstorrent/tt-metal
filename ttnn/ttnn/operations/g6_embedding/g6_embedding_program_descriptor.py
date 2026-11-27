# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Plan and ProgramDescriptor for g6_embedding, regime R1 (gather -> UInt16 tilize -> store).

Work unit = (token tile row, width block): 32 positions x block_width_tiles tile columns.
Units are numbered tile-row-major (width blocks inner) and split in contiguous ranges over
the whole grid. Every block knob is a module constant or the output of select_block_width;
CB capacities, trip counts and kernel args all derive from them.
"""

from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_HEIGHT = 32
TILE_WIDTH = 32
INDEX_ELEMENT_BYTES = 4  # uint32 index values

# Block knobs (op_design.md, Blocking Model).
BLOCK_TILE_ROWS = 1  # units per block along the token tile-row axis
GATHER_DEPTH_BLOCKS = 2  # cb_gathered_rows depth, in blocks
OUTPUT_DEPTH_BLOCKS = 2  # cb_output_tiles depth, in blocks
INDEX_WINDOW_MAX_TILE_ROWS = 64  # cap on tile rows of indices resident in cb_index_window
MAX_BLOCK_WIDTH_TILES = 80  # L1 cap on block_width_tiles
UNIT_OVERHEAD_TILES = 2  # per-unit fixed cost in tile-equivalents, used by select_block_width

# BB-05: L1 bytes per core left to the op while the model holds its own tensors.
L1_BUDGET_BYTES = 1_347_776

CB_INDEX_WINDOW = 0
CB_GATHERED_ROWS = 1
CB_OUTPUT_TILES = 2

# UInt16 keeps tilize on the integer unpack/datacopy/pack path: bfloat16 bits pass through
# unchanged (a Float16_b CB would take the float route, which flushes subnormals and rewrites NaN).
GATHER_FORMAT = ttnn.uint16
INDEX_FORMAT = ttnn.uint32

READER_KERNEL = 0
WRITER_KERNEL = 1
COMPUTE_KERNEL = 2

REGIME = "R1"
CORE_ROLE = "gather+tilize+store"


def select_block_width(tensor_tile_rows, tensor_width_tiles, num_grid_cores):
    """Width block (in tiles) minimizing the busiest core's load; ties go to the wider block."""
    candidates = [
        bw for bw in range(1, tensor_width_tiles + 1) if tensor_width_tiles % bw == 0 and bw <= MAX_BLOCK_WIDTH_TILES
    ]

    def busiest_core_cost(bw):
        units = tensor_tile_rows * (tensor_width_tiles // bw)
        return -(-units // num_grid_cores) * (bw + UNIT_OVERHEAD_TILES)

    return min(candidates, key=lambda bw: (busiest_core_cost(bw), -bw))


@dataclass(frozen=True)
class CoreAssignment:
    core: ttnn.CoreCoord
    first_unit: int
    num_units: int


@dataclass(frozen=True)
class Plan:
    images: int
    tokens_per_image: int
    tile_rows_per_image: int
    tensor_tile_rows: int
    tensor_width_tiles: int
    block_width_tiles: int
    width_blocks_per_row: int
    block_tile_rows: int
    total_units: int
    num_cores: int
    all_cores: ttnn.CoreRangeSet
    assignments: tuple
    units_per_core_group_1: int
    units_per_core_group_2: int
    index_window_tile_rows: int
    tile_bytes: int
    index_slot_bytes: int

    @property
    def gathered_rows_pages(self):
        return GATHER_DEPTH_BLOCKS * self.block_tile_rows * self.block_width_tiles

    @property
    def output_tiles_pages(self):
        return OUTPUT_DEPTH_BLOCKS * self.block_tile_rows * self.block_width_tiles

    @property
    def footprint_bytes(self):
        return self.index_slot_bytes * self.index_window_tile_rows + self.tile_bytes * (
            self.gathered_rows_pages + self.output_tiles_pages
        )


@functools.lru_cache(maxsize=256)
def make_plan(index_shape, row_length, grid_x, grid_y):
    """The blocking plan of one call shape on a grid_x x grid_y worker grid."""
    tokens_per_image = index_shape[-1]
    images = index_shape[0] if len(index_shape) == 2 else 1
    tile_rows_per_image = -(-tokens_per_image // TILE_HEIGHT)
    tensor_tile_rows = images * tile_rows_per_image
    tensor_width_tiles = row_length // TILE_WIDTH
    grid = ttnn.CoreCoord(grid_x, grid_y)

    block_width_tiles = select_block_width(tensor_tile_rows, tensor_width_tiles, grid_x * grid_y)
    width_blocks_per_row = tensor_width_tiles // block_width_tiles
    # A multi-unit block must stay a rectangle (whole tile rows) and within the width cap.
    if width_blocks_per_row == 1:
        block_tile_rows = max(1, min(BLOCK_TILE_ROWS, MAX_BLOCK_WIDTH_TILES // block_width_tiles))
    else:
        block_tile_rows = 1
    total_units = tensor_tile_rows * width_blocks_per_row

    num_cores, all_cores, core_group_1, _, units_g1, units_g2 = ttnn.split_work_to_cores(
        grid, total_units, row_wise=True
    )
    group_1_cores = core_group_1.num_cores()
    assignments = []
    max_tile_row_span = 0
    next_unit = 0
    for i, core in enumerate(ttnn.grid_to_cores(num_cores, grid_x, grid_y, True)):
        units = units_g1 if i < group_1_cores else units_g2
        assignments.append(CoreAssignment(core, next_unit, units))
        span = (next_unit + units - 1) // width_blocks_per_row - next_unit // width_blocks_per_row + 1
        max_tile_row_span = max(max_tile_row_span, span)
        next_unit += units
    assert next_unit == total_units

    return Plan(
        images=images,
        tokens_per_image=tokens_per_image,
        tile_rows_per_image=tile_rows_per_image,
        tensor_tile_rows=tensor_tile_rows,
        tensor_width_tiles=tensor_width_tiles,
        block_width_tiles=block_width_tiles,
        width_blocks_per_row=width_blocks_per_row,
        block_tile_rows=block_tile_rows,
        total_units=total_units,
        num_cores=num_cores,
        all_cores=all_cores,
        assignments=tuple(assignments),
        units_per_core_group_1=units_g1,
        units_per_core_group_2=units_g2,
        index_window_tile_rows=min(INDEX_WINDOW_MAX_TILE_ROWS, max_tile_row_span),
        tile_bytes=ttnn.tile_size(GATHER_FORMAT),
        index_slot_bytes=TILE_HEIGHT * INDEX_ELEMENT_BYTES,
    )


def plan_for(input_tensor, weight):
    grid = input_tensor.device().compute_with_storage_grid_size()
    return make_plan(tuple(input_tensor.shape), int(weight.shape[-1]), grid.x, grid.y)


def describe_plan(input_tensor, weight, memory_config=None):
    """BB-08 / BB-21: the plan a call selects and the role and unit range of every used core."""
    plan = plan_for(input_tensor, weight)
    return {
        "regime": REGIME,
        "block_width_tiles": plan.block_width_tiles,
        "width_blocks_per_row": plan.width_blocks_per_row,
        "block_tile_rows": plan.block_tile_rows,
        "total_units": plan.total_units,
        "num_cores": plan.num_cores,
        "units_per_core": (plan.units_per_core_group_1, plan.units_per_core_group_2),
        "index_window_tile_rows": plan.index_window_tile_rows,
        "cb_pages": {
            "cb_index_window": plan.index_window_tile_rows,
            "cb_gathered_rows": plan.gathered_rows_pages,
            "cb_output_tiles": plan.output_tiles_pages,
        },
        "footprint_bytes": plan.footprint_bytes,
        "cores": [
            {
                "core": (a.core.x, a.core.y),
                "role": CORE_ROLE,
                "units": (a.first_unit, a.first_unit + a.num_units),
            }
            for a in plan.assignments
        ],
    }


def _cb(index, data_format, page_size, num_pages, core_ranges):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def _build_program_descriptor(plan, input_tensor, weight, output_tensor, program_hash):
    cores = plan.all_cores
    cbs = [
        _cb(CB_INDEX_WINDOW, INDEX_FORMAT, plan.index_slot_bytes, plan.index_window_tile_rows, cores),
        _cb(CB_GATHERED_ROWS, GATHER_FORMAT, plan.tile_bytes, plan.gathered_rows_pages, cores),
        _cb(CB_OUTPUT_TILES, GATHER_FORMAT, plan.tile_bytes, plan.output_tiles_pages, cores),
    ]

    row_slice_bytes = plan.block_width_tiles * TILE_WIDTH * weight.element_size()
    reader_ct_args = [
        CB_INDEX_WINDOW,
        CB_GATHERED_ROWS,
        plan.block_width_tiles,
        plan.width_blocks_per_row,
        plan.block_tile_rows,
        plan.tile_rows_per_image,
        plan.tokens_per_image,
        plan.index_window_tile_rows,
        row_slice_bytes,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(weight).get_compile_time_args())

    writer_ct_args = [
        CB_OUTPUT_TILES,
        plan.block_width_tiles,
        plan.width_blocks_per_row,
        plan.block_tile_rows,
        plan.tensor_width_tiles,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    compute_ct_args = [CB_GATHERED_ROWS, CB_OUTPUT_TILES, plan.block_width_tiles]

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()
    for a in plan.assignments:
        reader_rt_args[a.core.x][a.core.y] = [a.first_unit, a.num_units]
        writer_rt_args[a.core.x][a.core.y] = [a.first_unit, a.num_units]
        compute_rt_args[a.core.x][a.core.y] = [a.num_units]

    kernels = [None, None, None]
    kernels[READER_KERNEL] = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_embedding_reader.cpp"),
        core_ranges=cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        common_runtime_args=[input_tensor.buffer_address(), weight.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    kernels[WRITER_KERNEL] = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_embedding_writer.cpp"),
        core_ranges=cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        common_runtime_args=[output_tensor.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    kernels[COMPUTE_KERNEL] = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_embedding_compute.cpp"),
        core_ranges=cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(
            fp32_dest_acc_en=False,
            math_approx_mode=False,
            dst_full_sync_en=False,
        ),
    )

    descriptor = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    descriptor.custom_program_hash = program_hash
    return descriptor


# BB-07: one descriptor per call signature; a hit only rewrites the three buffer addresses.
_DESCRIPTOR_CACHE = {}


def _cache_key(input_tensor, weight, output_tensor, grid):
    out_mc = output_tensor.memory_config()
    return (
        "g6_embedding",
        tuple(input_tensor.shape),
        repr(input_tensor.layout),
        tuple(weight.shape),
        repr(weight.dtype),
        repr(out_mc.buffer_type),
        repr(out_mc.memory_layout),
        (grid.x, grid.y),
        input_tensor.device().id(),
    )


def get_program_descriptor(input_tensor, weight, output_tensor):
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    key = _cache_key(input_tensor, weight, output_tensor, grid)
    descriptor = _DESCRIPTOR_CACHE.get(key)
    if descriptor is None:
        program_hash = int.from_bytes(hashlib.blake2b(repr(key).encode(), digest_size=8).digest(), "little")
        plan = make_plan(tuple(input_tensor.shape), int(weight.shape[-1]), grid.x, grid.y)
        descriptor = _build_program_descriptor(plan, input_tensor, weight, output_tensor, program_hash)
        _DESCRIPTOR_CACHE[key] = descriptor
        return descriptor

    kernels = descriptor.kernels
    reader_common = kernels[READER_KERNEL].common_runtime_args
    reader_common[0] = input_tensor.buffer_address()
    reader_common[1] = weight.buffer_address()
    kernels[WRITER_KERNEL].common_runtime_args[0] = output_tensor.buffer_address()
    return descriptor
