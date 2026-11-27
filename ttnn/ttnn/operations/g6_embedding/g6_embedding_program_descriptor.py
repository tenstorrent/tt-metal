# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_embedding host plan and ProgramDescriptor.

The plan (regime, block knobs, work split) depends only on shapes and the grid. The descriptor built
from it is cached per call signature; per call only the three buffer addresses change, and they are
common runtime args set on the cached kernel descriptors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
INDEX_CHUNK_BYTES = TILE * 4  # one tile row of uint32 indices
ROW_TILE_BYTES = ttnn.tile_size(ttnn.uint16)  # 32 rows x 32 bf16 values carried as UInt16

# Per-core L1 the model leaves to the op (BB-05).
L1_BUDGET = 1_347_776
# Widest tilize block whose unpacker row stride has been exercised on device.
MAX_COL_BLOCK_TILES = 64
MIN_BLOCKS_PER_CORE = 4
INDEX_DEPTH = 2
ROWS_DEPTH = 2
TILES_DEPTH = 2

CB_INDEX = 0
CB_ROWS = 1
CB_TILES = 2

REGIME_TOKEN_SPLIT = "token_split"
REGIME_TOKEN_COLUMN_SPLIT = "token_column_split"


def footprint(block_tile_rows, col_block_tiles):
    """Bytes of L1 the op's CBs hold on one core."""
    index_bytes = INDEX_CHUNK_BYTES * INDEX_DEPTH * block_tile_rows
    rows_bytes = ROW_TILE_BYTES * ROWS_DEPTH * block_tile_rows * col_block_tiles
    tiles_bytes = ROW_TILE_BYTES * TILES_DEPTH * col_block_tiles
    return index_bytes + rows_bytes + tiles_bytes


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


@dataclass(frozen=True)
class Plan:
    regime: str
    image_count: int  # B
    image_tokens: int  # S (N for a rank-1 index)
    image_tile_rows: int  # St
    tensor_tile_rows: int  # T
    tensor_col_tiles: int  # Dt
    table_rows: int  # V
    grid_cores: int  # C
    col_block_tiles: int
    col_pieces: int  # P
    total_units: int  # U
    block_tile_rows: int
    num_cores: int
    all_cores: object
    assignment: tuple  # ((core, unit_begin, num_units), ...)
    l1_bytes_per_core: int


def make_plan(index_shape, weight_shape, grid_size):
    """Regime, block knobs and work split for one call signature (op_design.md, Blocking Model)."""
    image_count = index_shape[0] if len(index_shape) == 2 else 1
    image_tokens = index_shape[-1]
    table_rows = weight_shape[-2]
    tensor_col_tiles = weight_shape[-1] // TILE
    image_tile_rows = math.ceil(image_tokens / TILE)
    tensor_tile_rows = image_count * image_tile_rows
    grid_cores = grid_size.x * grid_size.y

    col_divisors = _divisors(tensor_col_tiles)
    capped = [d for d in col_divisors if d <= MAX_COL_BLOCK_TILES and footprint(1, d) <= L1_BUDGET]
    if not capped:
        raise ValueError(
            f"g6_embedding: no column block of D={weight_shape[-1]} fits the per-core L1 bound of {L1_BUDGET} bytes"
        )
    col_cap = max(capped)
    if tensor_tile_rows >= grid_cores:
        col_block_tiles = col_cap
    else:
        filling = [d for d in col_divisors if d <= col_cap and tensor_tile_rows * (tensor_col_tiles // d) >= grid_cores]
        col_block_tiles = max(filling) if filling else 1
    col_pieces = tensor_col_tiles // col_block_tiles
    regime = REGIME_TOKEN_SPLIT if col_pieces == 1 else REGIME_TOKEN_COLUMN_SPLIT
    total_units = tensor_tile_rows * col_pieces

    num_cores, all_cores, core_group_1, core_group_2, units_g1, units_g2 = ttnn.split_work_to_cores(
        grid_size, total_units, row_wise=True
    )
    assignment = []
    unit_begin = 0
    for group, units in ((core_group_1, units_g1), (core_group_2, units_g2)):
        if units == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, unit_begin, units))
            unit_begin += units
    assert unit_begin == total_units, (unit_begin, total_units)

    core_units_max = max(units_g1, units_g2)
    block_l1 = (L1_BUDGET - ROW_TILE_BYTES * col_block_tiles * TILES_DEPTH) // (
        ROW_TILE_BYTES * col_block_tiles * ROWS_DEPTH + INDEX_CHUNK_BYTES * INDEX_DEPTH
    )
    block_tile_rows = max(1, min(block_l1, math.ceil(core_units_max / MIN_BLOCKS_PER_CORE)))
    l1_bytes = footprint(block_tile_rows, col_block_tiles)
    if l1_bytes > L1_BUDGET:
        raise ValueError(
            f"g6_embedding: plan needs {l1_bytes} bytes of L1 per core, above the bound of {L1_BUDGET} bytes"
        )

    return Plan(
        regime=regime,
        image_count=image_count,
        image_tokens=image_tokens,
        image_tile_rows=image_tile_rows,
        tensor_tile_rows=tensor_tile_rows,
        tensor_col_tiles=tensor_col_tiles,
        table_rows=table_rows,
        grid_cores=grid_cores,
        col_block_tiles=col_block_tiles,
        col_pieces=col_pieces,
        total_units=total_units,
        block_tile_rows=block_tile_rows,
        num_cores=num_cores,
        all_cores=all_cores,
        assignment=tuple(assignment),
        l1_bytes_per_core=l1_bytes,
    )


def plan_report(plan):
    """The plan a call selected (BB-08); core_ranges doubles as the role dump: every core runs all three roles."""
    return {
        "regime": plan.regime,
        "col_block_tiles": plan.col_block_tiles,
        "col_pieces": plan.col_pieces,
        "block_tile_rows": plan.block_tile_rows,
        "tensor_tile_rows": plan.tensor_tile_rows,
        "total_units": plan.total_units,
        "num_cores": plan.num_cores,
        "core_ranges": [((r.start.x, r.start.y), (r.end.x, r.end.y)) for r in plan.all_cores.ranges()],
        "l1_bytes_per_core": plan.l1_bytes_per_core,
    }


def _cb(index, num_pages, page_bytes, data_format, core_ranges):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_bytes)],
    )


@dataclass
class CachedProgram:
    plan: Plan
    cbs: list
    reader: object
    writer: object
    compute: object


def build_cached_program(plan, input_tensor, weight, output_tensor):
    """CBs and kernel descriptors for a plan; buffer addresses are filled per call by program_descriptor()."""
    cores = plan.all_cores
    cbs = [
        _cb(CB_INDEX, INDEX_DEPTH * plan.block_tile_rows, INDEX_CHUNK_BYTES, ttnn.uint32, cores),
        _cb(CB_ROWS, ROWS_DEPTH * plan.block_tile_rows * plan.col_block_tiles, ROW_TILE_BYTES, ttnn.uint16, cores),
        _cb(CB_TILES, TILES_DEPTH * plan.col_block_tiles, ROW_TILE_BYTES, ttnn.uint16, cores),
    ]

    reader_ct_args = [
        CB_INDEX,
        CB_ROWS,
        plan.block_tile_rows,
        plan.col_block_tiles,
        INDEX_DEPTH,
        plan.tensor_tile_rows,
        plan.image_tile_rows,
        plan.image_tokens,
        plan.table_rows,
        input_tensor.buffer_aligned_page_size(),
        weight.buffer_aligned_page_size(),
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(weight).get_compile_time_args())

    writer_ct_args = [
        CB_TILES,
        plan.col_block_tiles,
        plan.tensor_tile_rows,
        plan.tensor_col_tiles,
        output_tensor.buffer_aligned_page_size(),
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    compute_ct_args = [plan.col_block_tiles, CB_ROWS, CB_TILES]

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()
    for core, unit_begin, num_units in plan.assignment:
        reader_rt_args[core.x][core.y] = [unit_begin, num_units]
        writer_rt_args[core.x][core.y] = [unit_begin, num_units]
        compute_rt_args[core.x][core.y] = [num_units]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_embedding_reader.cpp"),
        core_ranges=cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        common_runtime_args=[0, 0],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_embedding_writer.cpp"),
        core_ranges=cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        common_runtime_args=[0],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_embedding_compute.cpp"),
        core_ranges=cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False, dst_full_sync_en=False),
    )
    return CachedProgram(plan=plan, cbs=cbs, reader=reader, writer=writer, compute=compute)


def program_descriptor(cached, input_tensor, weight, output_tensor):
    cached.reader.common_runtime_args = [input_tensor.buffer_address(), weight.buffer_address()]
    cached.writer.common_runtime_args = [output_tensor.buffer_address()]
    return ttnn.ProgramDescriptor(
        kernels=[cached.reader, cached.writer, cached.compute],
        semaphores=[],
        cbs=cached.cbs,
    )
