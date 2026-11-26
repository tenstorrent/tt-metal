# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_embedding program descriptor: block plan, CBs, kernels, per-signature cache.

Data path per core (op_design.md, Blocking Model):
    reader  (NCRISC, NoC0): index chunks -> cb_index (private ring); one row-slice read per valid
                            token, addressed by the index value -> cb_rows (UInt16 row-major)
    compute (TRISCs)      : UInt16 tilize cb_rows -> cb_tiles (bit permutation, no float path)
    writer  (BRISC, NoC1) : whole 2048-B tiles cb_tiles -> output DRAM

Work unit = one output tile row (32 tokens of one image) x one column piece of col_block_tiles
tile columns, numbered column-piece-major (unit = piece * T + tile_row).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_DIM = 32
TILE_BYTES = 32 * 32 * 2  # one 16-bit tile; equals ttnn.tile_size(ttnn.uint16) and the bf16 output page
INDEX_CHUNK_BYTES = TILE_DIM * 4  # the uint32 indices of one tile row

# The per-core L1 the model leaves to the op while it holds its own tensors (BB-05).
L1_BUDGET = 1347776

# Widest tilize block whose unpacker row stride the probes cover with margin (production max is 40).
MAX_COL_BLOCK_TILES = 64

# Buffer-depth knobs (Blocking Model, buffer-depth table).
INDEX_DEPTH = 2  # cb_index slots: prefetch of block k+1's indices under block k's row reads
ROWS_DEPTH = 2  # cb_rows blocks: reader gathers block k+1 while compute tilizes block k
TILES_DEPTH = 2  # cb_tiles units: compute tilizes unit j+1 while the writer drains unit j

# block_tile_rows = ceil(units per core / MIN_BLOCKS_PER_CORE), so the first gather (pipeline fill)
# is a quarter of a core's work, not all of it.
MIN_BLOCKS_PER_CORE = 4

CB_INDEX = 0
CB_ROWS = 1
CB_TILES = 2


def footprint(block_tile_rows: int, col_block_tiles: int) -> int:
    """Bytes of L1 the three CBs take on one core."""
    return (
        TILE_BYTES * col_block_tiles * (ROWS_DEPTH * block_tile_rows + TILES_DEPTH)
        + INDEX_CHUNK_BYTES * INDEX_DEPTH * block_tile_rows
    )


@dataclass(frozen=True)
class Plan:
    regime: str
    batch: int  # B (1 for a rank-1 index)
    tokens_per_image: int  # S (N for a rank-1 index)
    vocab_rows: int  # V
    tensor_col_tiles: int  # Dt
    image_tile_rows: int  # St
    tensor_tile_rows: int  # T = B * St
    col_block_tiles: int
    col_pieces: int  # P = Dt / col_block_tiles
    total_units: int  # U = T * P
    block_tile_rows: int
    num_cores: int
    core_ranges: object  # ttnn.CoreRangeSet
    core_assignment: tuple  # ((CoreCoord, unit_begin, num_units), ...) in unit order
    l1_bytes_per_core: int


def _divisors(n: int):
    return [d for d in range(1, n + 1) if n % d == 0]


def _split(grid_x: int, grid_y: int, total_units: int):
    grid = ttnn.CoreCoord(grid_x, grid_y)
    (num_cores, all_cores, group_1, group_2, units_g1, units_g2) = ttnn.split_work_to_cores(
        grid, total_units, row_wise=True
    )
    assignment = []
    begin = 0
    for group, per_core in ((group_1, units_g1), (group_2, units_g2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, begin, per_core))
            begin += per_core
    assert begin == total_units, (begin, total_units)
    return num_cores, all_cores, tuple(assignment), max(units_g1, units_g2)


@lru_cache(maxsize=256)
def compute_plan(index_shape: tuple, weight_shape: tuple, grid_x: int, grid_y: int) -> Plan:
    """The block plan for one call signature (shapes and grid only; never index values)."""
    if len(index_shape) == 1:
        batch, tokens = 1, index_shape[0]
    else:
        batch, tokens = index_shape
    vocab, width = weight_shape[-2], weight_shape[-1]

    tensor_col_tiles = width // TILE_DIM
    image_tile_rows = math.ceil(tokens / TILE_DIM)
    tensor_tile_rows = batch * image_tile_rows
    grid_cores = grid_x * grid_y

    fitting = [d for d in _divisors(tensor_col_tiles) if d <= MAX_COL_BLOCK_TILES and footprint(1, d) <= L1_BUDGET]
    if not fitting:
        raise ValueError(
            f"g6_embedding: no column block of D={width} fits the per-core L1 bound of {L1_BUDGET} bytes (BB-05)"
        )
    ct_cap = max(fitting)
    if tensor_tile_rows >= grid_cores:
        col_block_tiles = ct_cap
    else:
        filling = [d for d in fitting if d <= ct_cap and tensor_tile_rows * (tensor_col_tiles // d) >= grid_cores]
        col_block_tiles = max(filling) if filling else 1

    col_pieces = tensor_col_tiles // col_block_tiles
    total_units = tensor_tile_rows * col_pieces
    regime = "token_split" if col_pieces == 1 else "token_column_split"

    num_cores, core_ranges, assignment, core_units_max = _split(grid_x, grid_y, total_units)

    bt_l1 = (L1_BUDGET - TILE_BYTES * col_block_tiles * TILES_DEPTH) // (
        TILE_BYTES * col_block_tiles * ROWS_DEPTH + INDEX_CHUNK_BYTES * INDEX_DEPTH
    )
    block_tile_rows = max(1, min(bt_l1, math.ceil(core_units_max / MIN_BLOCKS_PER_CORE)))

    l1_bytes = footprint(block_tile_rows, col_block_tiles)
    if l1_bytes > L1_BUDGET:
        raise ValueError(
            f"g6_embedding: plan needs {l1_bytes} bytes of L1 per core, above the bound of {L1_BUDGET} bytes (BB-05)"
        )

    return Plan(
        regime=regime,
        batch=batch,
        tokens_per_image=tokens,
        vocab_rows=vocab,
        tensor_col_tiles=tensor_col_tiles,
        image_tile_rows=image_tile_rows,
        tensor_tile_rows=tensor_tile_rows,
        col_block_tiles=col_block_tiles,
        col_pieces=col_pieces,
        total_units=total_units,
        block_tile_rows=block_tile_rows,
        num_cores=num_cores,
        core_ranges=core_ranges,
        core_assignment=assignment,
        l1_bytes_per_core=l1_bytes,
    )


def plan_report(plan: Plan) -> dict:
    return {
        "regime": plan.regime,
        "col_block_tiles": plan.col_block_tiles,
        "col_pieces": plan.col_pieces,
        "block_tile_rows": plan.block_tile_rows,
        "tensor_tile_rows": plan.tensor_tile_rows,
        "total_units": plan.total_units,
        "num_cores": plan.num_cores,
        "core_ranges": [((r.start.x, r.start.y), (r.end.x, r.end.y)) for r in plan.core_ranges.ranges()],
        "l1_bytes_per_core": plan.l1_bytes_per_core,
        # Every used core runs the same three roles (BB-21 role dump).
        "roles": {"reader": "NCRISC", "compute": "TRISC0-2", "writer": "BRISC"},
    }


@dataclass
class _CachedProgram:
    plan: Plan
    reader: object  # ttnn.KernelDescriptor
    writer: object
    compute: object
    cbs: list


def _cb(index: int, data_format, page_size: int, num_pages: int, core_ranges):
    return ttnn.CBDescriptor(
        total_size=page_size * num_pages,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def _build(plan: Plan, input_tensor, weight, output_tensor) -> _CachedProgram:
    cores = plan.core_ranges
    ct = plan.col_block_tiles
    bt = plan.block_tile_rows

    cbs = [
        _cb(CB_INDEX, ttnn.uint32, INDEX_CHUNK_BYTES, INDEX_DEPTH * bt, cores),
        # UInt16 carries the bf16 bits: a Float16_b-typed tilize rewrites -0.0, subnormals and NaNs.
        _cb(CB_ROWS, ttnn.uint16, TILE_BYTES, ROWS_DEPTH * bt * ct, cores),
        _cb(CB_TILES, ttnn.uint16, TILE_BYTES, TILES_DEPTH * ct, cores),
    ]

    reader_ct = [
        CB_INDEX,
        CB_ROWS,
        ct,
        bt,
        INDEX_DEPTH,
        plan.tensor_tile_rows,
        plan.image_tile_rows,
        plan.tokens_per_image,
        plan.vocab_rows,
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(weight).get_compile_time_args())

    writer_ct = [CB_TILES, ct, plan.tensor_tile_rows, plan.tensor_col_tiles]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    compute_ct = [CB_ROWS, CB_TILES, ct]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    for core, unit_begin, num_units in plan.core_assignment:
        reader_rt[core.x][core.y] = [unit_begin, num_units]
        writer_rt[core.x][core.y] = [unit_begin, num_units]
        compute_rt[core.x][core.y] = [num_units]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_embedding_reader.cpp"),
        core_ranges=cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        common_runtime_args=[0, 0],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_embedding_writer.cpp"),
        core_ranges=cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        common_runtime_args=[0],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_embedding_compute.cpp"),
        core_ranges=cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        # 16-bit DEST, no UnpackToDest: the probed bit-exact UInt16 tilize configuration.
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False, dst_full_sync_en=False),
    )
    return _CachedProgram(plan=plan, reader=reader, writer=writer, compute=compute, cbs=cbs)


_PROGRAM_CACHE: dict = {}


def create_program_descriptor(cache_key, plan: Plan, input_tensor, weight, output_tensor) -> ttnn.ProgramDescriptor:
    """Descriptor for this call; built once per signature, only the buffer addresses change per call."""
    cached = _PROGRAM_CACHE.get(cache_key)
    if cached is None:
        cached = _build(plan, input_tensor, weight, output_tensor)
        _PROGRAM_CACHE[cache_key] = cached
    cached.reader.common_runtime_args = [input_tensor.buffer_address(), weight.buffer_address()]
    cached.writer.common_runtime_args = [output_tensor.buffer_address()]
    return ttnn.ProgramDescriptor(
        kernels=[cached.reader, cached.writer, cached.compute],
        semaphores=[],
        cbs=cached.cbs,
    )
