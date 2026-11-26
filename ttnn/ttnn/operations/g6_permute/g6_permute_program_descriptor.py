# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_permute regime R1 `rows_tile`: plan selection, cached ProgramDescriptor, plan report.

dims (0, 2, 1, 3) on TILE bfloat16: out[n, b, a, :] = x[n, a, b, :]. A block is a sub-rectangle of one closed
cell (n, at, bt, wt): block_a rows of a inside one output tile row, block_b rows of b inside one input tile row,
block_wt tile columns. Reader (NCRISC/NoC0) loads a block's input tiles; writer (BRISC/NoC1) regroups rows into
staging tiles (block_a > 1) or writes them straight from the input slot (block_a == 1), then stores to DRAM.
No compute kernel.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

REGIME_ROWS_TILE = "rows_tile"

# Block and depth knobs: each is defined here once; CB pages, loop bounds and the block grid derive from them.
BLOCK_N = 1  # one image per block (implicit in the block grid)
BLOCK_WT = 1
MAX_DEPTH_IN = 2
MAX_DEPTH_STAGING = 2
TILE_ROW_SPLITS = (32, 16, 8, 4, 2, 1)  # legal block_a / block_b: divisors of the tile height

CB_INPUT_TILES = 0
CB_OUTPUT_STAGING = 1

_EXPECTED_TILE_SHAPE = [32, 32]
_EXPECTED_FACE_SHAPE = [16, 16]


def _div_up(x, y):
    return -(-x // y)


def count_blocks(N, A, B, Wt, block_a, block_b, block_wt=BLOCK_WT):
    return N * _div_up(A, block_a) * _div_up(B, block_b) * _div_up(Wt, block_wt)


def select_plan(N, A, B, Wt, max_cores):
    """Occupancy walk: closed cells, then halve block_a to 1, then halve block_b.

    Each step is taken iff its block count fits max_cores; the walk stops at the first step that does not.
    num_blocks never decreases along the walk, so the result is the finest split that puts at most one block on
    a core, or the closed cells when even those outnumber the cores.
    """
    tile_h = TILE_ROW_SPLITS[0]
    walk = [(a, tile_h) for a in TILE_ROW_SPLITS] + [(1, b) for b in TILE_ROW_SPLITS[1:]]
    chosen = walk[0]
    for step in walk[1:]:
        if count_blocks(N, A, B, Wt, *step) > max_cores:
            break
        chosen = step
    return chosen


@dataclass(frozen=True)
class Plan:
    regime: str
    N: int
    A: int
    B: int
    At: int
    Bt: int
    Wt: int
    block_a: int
    block_b: int
    block_wt: int
    num_a_groups: int
    num_b_groups: int
    num_wt_groups: int
    num_blocks: int
    num_cores: int
    depth_in: int
    depth_staging: int
    all_cores: object  # ttnn.CoreRangeSet
    assignment: tuple  # ((ttnn.CoreCoord, start_block, num_blocks), ...)
    tile_bytes: int
    tile_height: int
    face_height: int
    face_cols: int
    face_bytes: int
    face_row_bytes: int

    @property
    def regroup(self):
        return self.block_a > 1


def _cache_key(input_tensor, dims):
    grid = input_tensor.device().compute_with_storage_grid_size()
    memory_config = input_tensor.memory_config()
    return (
        tuple(int(d) for d in input_tensor.shape),
        tuple(int(d) for d in input_tensor.padded_shape),
        tuple(dims),
        str(input_tensor.dtype),
        str(input_tensor.layout),
        str(memory_config.buffer_type),
        str(memory_config.memory_layout),
        (int(grid.x), int(grid.y)),
    )


def _assign_blocks(grid, num_blocks):
    _, all_cores, core_group_1, core_group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(
        grid, num_blocks, row_wise=True
    )
    assignment = []
    start = 0
    for group, per_core in ((core_group_1, per_core_1), (core_group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, start, per_core))
            start += per_core
    assert start == num_blocks, f"work split covered {start} of {num_blocks} blocks"
    return all_cores, tuple(assignment)


def _build_plan(input_tensor):
    tile = input_tensor.tile
    tile_shape = [int(v) for v in tile.tile_shape]
    face_shape = [int(v) for v in tile.face_shape]
    assert tile_shape == _EXPECTED_TILE_SHAPE, f"g6_permute rows_tile needs 32x32 tiles, got {tile_shape}"
    assert face_shape == _EXPECTED_FACE_SHAPE, f"g6_permute rows_tile needs 16x16 faces, got {face_shape}"
    tile_h, tile_w = tile_shape
    face_h, face_w = face_shape
    elem_bytes = input_tensor.element_size()
    tile_bytes = input_tensor.buffer_page_size()
    assert tile_bytes == tile_h * tile_w * elem_bytes

    N, A, B, W = (int(d) for d in input_tensor.shape)
    At, Bt, Wt = _div_up(A, tile_h), _div_up(B, tile_h), _div_up(W, tile_w)

    grid = input_tensor.device().compute_with_storage_grid_size()
    max_cores = int(grid.x) * int(grid.y)
    block_a, block_b = select_plan(N, A, B, Wt, max_cores)
    block_wt = BLOCK_WT
    num_blocks = count_blocks(N, A, B, Wt, block_a, block_b, block_wt)
    all_cores, assignment = _assign_blocks(grid, num_blocks)
    max_blocks_per_core = max(count for _, _, count in assignment)

    depth_in = min(MAX_DEPTH_IN, max_blocks_per_core)
    depth_staging = min(MAX_DEPTH_STAGING, max_blocks_per_core) if block_a > 1 else 0

    return Plan(
        regime=REGIME_ROWS_TILE,
        N=N,
        A=A,
        B=B,
        At=At,
        Bt=Bt,
        Wt=Wt,
        block_a=block_a,
        block_b=block_b,
        block_wt=block_wt,
        num_a_groups=_div_up(A, block_a),
        num_b_groups=_div_up(B, block_b),
        num_wt_groups=_div_up(Wt, block_wt),
        num_blocks=num_blocks,
        num_cores=len(assignment),
        depth_in=depth_in,
        depth_staging=depth_staging,
        all_cores=all_cores,
        assignment=assignment,
        tile_bytes=tile_bytes,
        tile_height=tile_h,
        face_height=face_h,
        face_cols=tile_w // face_w,
        face_bytes=face_h * face_w * elem_bytes,
        face_row_bytes=face_w * elem_bytes,
    )


_PLANS = {}
_PROGRAMS = {}


def get_plan(input_tensor, dims):
    key = _cache_key(input_tensor, dims)
    plan = _PLANS.get(key)
    if plan is None:
        plan = _build_plan(input_tensor)
        _PLANS[key] = plan
    return plan


def describe_plan_dict(plan):
    return {
        "regime": plan.regime,
        "block_a": plan.block_a,
        "block_b": plan.block_b,
        "block_wt": plan.block_wt,
        "num_blocks": plan.num_blocks,
        "num_cores": plan.num_cores,
        "depth_in": plan.depth_in,
        "depth_staging": plan.depth_staging,
        "regroup": plan.regroup,
        "cores": [
            {
                "core": (int(core.x), int(core.y)),
                "role": f"{plan.regime} reader+writer",
                "start_block": start,
                "num_blocks": count,
            }
            for core, start, count in plan.assignment
        ],
    }


def _plan_ct_args(plan):
    # Order is the contract with kernels/g6_permute_common.hpp.
    return [
        plan.A,
        plan.B,
        plan.At,
        plan.Bt,
        plan.Wt,
        plan.block_a,
        plan.block_b,
        plan.block_wt,
        plan.num_a_groups,
        plan.num_b_groups,
        plan.num_wt_groups,
        plan.tile_bytes,
        plan.tile_height,
        plan.face_height,
        plan.face_cols,
        plan.face_bytes,
        plan.face_row_bytes,
        plan.depth_in,
        plan.depth_staging,
        CB_INPUT_TILES,
        CB_OUTPUT_STAGING,
    ]


def _program_hash(key):
    digest = hashlib.blake2b(repr(("g6_permute", REGIME_ROWS_TILE) + key).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


@dataclass
class _CachedProgram:
    reader: object  # ttnn.KernelDescriptor
    writer: object  # ttnn.KernelDescriptor
    cbs: list
    program_hash: int


def _build_program(plan, input_tensor, output_tensor, program_hash):
    def cb(index, num_pages):
        return ttnn.CBDescriptor(
            total_size=num_pages * plan.tile_bytes,
            core_ranges=plan.all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=input_tensor.dtype, page_size=plan.tile_bytes)
            ],
        )

    cbs = [cb(CB_INPUT_TILES, plan.depth_in * plan.block_a * plan.block_wt)]
    if plan.regroup:
        cbs.append(cb(CB_OUTPUT_STAGING, plan.depth_staging * plan.block_b * plan.block_wt))

    plan_args = _plan_ct_args(plan)
    reader_ct_args = plan_args + list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct_args = plan_args + list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    per_core_args = ttnn.RuntimeArgs()
    for core, start, count in plan.assignment:
        per_core_args[core.x][core.y] = [start, count]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_permute_reader.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=per_core_args,
        common_runtime_args=[input_tensor.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_permute_writer.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=per_core_args,
        common_runtime_args=[output_tensor.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    return _CachedProgram(reader=reader, writer=writer, cbs=cbs, program_hash=program_hash)


def program_descriptor(input_tensor, output_tensor, dims):
    """The cached rows_tile program with this call's buffer addresses patched in."""
    key = _cache_key(input_tensor, dims)
    cached = _PROGRAMS.get(key)
    if cached is None:
        cached = _build_program(get_plan(input_tensor, dims), input_tensor, output_tensor, _program_hash(key))
        _PROGRAMS[key] = cached
    else:
        cached.reader.common_runtime_args = [input_tensor.buffer_address()]
        cached.writer.common_runtime_args = [output_tensor.buffer_address()]
    descriptor = ttnn.ProgramDescriptor(kernels=[cached.reader, cached.writer], semaphores=[], cbs=cached.cbs)
    descriptor.custom_program_hash = cached.program_hash
    return descriptor
