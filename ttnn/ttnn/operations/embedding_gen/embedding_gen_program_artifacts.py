# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for embedding_gen: gather-direct (op_design.md, Blocking Model).

Two data-movement kernels on every cluster that gets work, and no compute kernel:

    reader  index slice (DRAM) -> scratch_token_indices; selected table pieces (DRAM) -> gathered_block entry,
            each piece landing at its final position in the output layout (row slab or tile face row)
    writer  gathered_block entry -> output pages (whole tiles, or row slabs)

The work unit is a block = (token group, column chunk): `block_tokens` tokens of one batch x `block_cols`
32-element column units. `plan_blocks` is the single place every knob is decided; every buffer size, loop bound
and runtime argument below is derived from its return value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# --- Knobs (op_design.md, Blocking Model). Each is defined once; everything else derives from them. ---------
# Thread plan (R, C, W): reader harts, compute threads, writer harts per cluster. There is no compute kernel.
THREAD_PLAN = (1, 0, 1)
RING_DEPTH = 2  # gathered_block entries per producer thread: assemble block k+1 while block k drains
INDEX_SLOTS = 2  # index slices per reader thread: the next block's slice lands during this block's gather
BLOCK_TOKENS_MAX = 32  # cap on tokens per block; sizes one index slot

# --- Mechanism caps and part facts (target page; Python has no query for them). -----------------------------
DFB_MAX_BYTES = 1_048_560  # one ring, at most 1 MiB - 16 B
FREE_DM_HARTS = 6  # of 8 per cluster
L1_RESERVE_BYTES = 64 * 1024  # headroom kept back from the unreserved L1 budget
TILE_HEIGHT = 32
TILE_WIDTH = 32
FACE_HEIGHT = 16
FACE_WIDTH = 16
INDEX_BYTES = 4  # uint32

assert TILE_HEIGHT <= BLOCK_TOKENS_MAX, "a tile-layout block is one tile-row of the output"


def validate_thread_plan(threads):
    r, c, w = threads
    if not (r >= 1 and w >= 1 and c == 0 and r + w <= FREE_DM_HARTS and (r % w == 0 or w % r == 0)):
        raise ValueError(f"thread plan (R, C, W) = {threads}: need C = 0, R + W <= {FREE_DM_HARTS}, R | W or W | R")


@dataclass(frozen=True)
class BlockPlan:
    num_cores: int
    block_tokens: int
    groups_per_batch: int
    n_token_groups: int
    num_col_chunks: int
    block_cols: int
    num_units: int
    ring_entries: int
    entry_bytes: int
    scratch_bytes: int


def plan_blocks(B, S, Dt, esize, out_tiled, grid, l1_budget, dfb_max_bytes=DFB_MAX_BYTES, threads=THREAD_PLAN):
    """The blocking of one call (op_design.md, `plan_blocks` steps 1-6)."""
    R, _, W = threads
    num_cores = grid.x * grid.y
    # 1 occupancy: a tile-layout block is one output tile-row; a row-major one spreads the tokens over the grid.
    block_tokens = TILE_HEIGHT if out_tiled else min(max(math.ceil(B * S / num_cores), 1), BLOCK_TOKENS_MAX)
    # 2 groups never cross a batch
    groups_per_batch = math.ceil(S / block_tokens)
    n_token_groups = B * groups_per_batch
    # 3 fill the cores the tokens leave idle with column chunks
    num_col_chunks = min(max(num_cores // n_token_groups, 1), Dt)
    # 4 ring cap
    ring_entries = RING_DEPTH * max(R, W)
    scratch_bytes = R * INDEX_SLOTS * BLOCK_TOKENS_MAX * INDEX_BYTES
    col_unit_bytes = TILE_WIDTH * esize
    cols_cap = min(dfb_max_bytes, l1_budget - scratch_bytes) // (ring_entries * block_tokens * col_unit_bytes)
    if cols_cap < 1:
        raise ValueError(f"embedding_gen: no column block fits a ring of {dfb_max_bytes} B")
    num_col_chunks = max(num_col_chunks, math.ceil(Dt / cols_cap))
    # 5 balance: only the last chunk is ragged
    block_cols = math.ceil(Dt / num_col_chunks)
    num_col_chunks = math.ceil(Dt / block_cols)
    # 6 deal
    num_units = n_token_groups * num_col_chunks
    entry_bytes = block_tokens * block_cols * col_unit_bytes
    return BlockPlan(num_cores, block_tokens, groups_per_batch, n_token_groups, num_col_chunks, block_cols,
                     num_units, ring_entries, entry_bytes, scratch_bytes)


def split_units(num_units, grid):
    """(core set, {core: units}, {core: first unit}) over the clusters that get work, row-wise."""
    _, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(grid, num_units, row_wise=True)
    counts, starts, start = {}, {}, 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            counts[core], starts[core] = per_core, start
            start += per_core
    assert start == num_units
    return all_cores, counts, starts


def _dm_config(make, explicit_rings):
    hw = make()
    hw.config_2xx = E.DataMovementHardwareConfig.DataMovement2XXConfig(disable_dfb_implicit_sync_for=explicit_rings)
    return hw


def element_size(dtype):
    return {ttnn.bfloat16: 2, ttnn.float32: 4}[dtype]


def create_program_artifacts(indices, table, out, *, dfb_max_bytes=DFB_MAX_BYTES, threads=THREAD_PLAN, grid=None):
    """(ProgramSpec, ProgramRunArgs, tensor indices) of one lookup. `dfb_max_bytes`, `threads` and `grid` are
    the knobs a test may turn; the op passes the defaults."""
    validate_thread_plan(threads)
    R, _, W = threads
    indices_shape = [int(d) for d in indices.shape]
    table_shape = [int(d) for d in table.shape]
    B = 1 if len(indices_shape) == 1 else indices_shape[0]
    S = indices_shape[-1]
    V, D = table_shape[-2], table_shape[-1]
    Dt = math.ceil(D / TILE_WIDTH)
    esize = element_size(table.dtype)
    table_tiled = table.layout == ttnn.TILE_LAYOUT
    out_tiled = out.layout == ttnn.TILE_LAYOUT
    grid = grid or indices.device().compute_with_storage_grid_size()
    l1_budget = ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES

    plan = plan_blocks(B, S, Dt, esize, out_tiled, grid, l1_budget, dfb_max_bytes, threads)
    core_set, counts, starts = split_units(plan.num_units, grid)

    ring = E.DataflowBufferSpec(unique_id="gathered_block", entry_size=plan.entry_bytes, num_entries=plan.ring_entries)
    scratch = E.ScratchpadSpec(unique_id="token_indices", size_per_node=plan.scratch_bytes)

    ct_args = {
        "reader_threads": R,
        "writer_threads": W,
        "table_tiled": int(table_tiled),
        "out_tiled": int(out_tiled),
        "index_slots": INDEX_SLOTS,
        "block_tokens_max": BLOCK_TOKENS_MAX,
        "index_bytes": INDEX_BYTES,
        "tile_height": TILE_HEIGHT,
        "face_height": FACE_HEIGHT,
        "faces_per_row": TILE_WIDTH // FACE_WIDTH,  # 16-element halves of one tile row
        "col_unit_bytes": TILE_WIDTH * esize,  # one 32-element column unit of a row
        "face_row_bytes": FACE_WIDTH * esize,  # one row of a 16x16 face
        "face_bytes": FACE_HEIGHT * FACE_WIDTH * esize,
        "tile_bytes": TILE_HEIGHT * TILE_WIDTH * esize,
    }
    common = {
        "S": S,
        "V": V,
        "Dt": Dt,
        "block_tokens": plan.block_tokens,
        "block_cols": plan.block_cols,
        "num_col_chunks": plan.num_col_chunks,
        "groups_per_batch": plan.groups_per_batch,
    }
    schema = E.KernelSpec.RuntimeArgSchema(
        runtime_arg_names=["start_unit", "num_units"], common_runtime_arg_names=list(common)
    )
    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "embedding_gen_reader.cpp"),
        hw_config=_dm_config(E.create_reader_datamovement_config, ["gathered_block"]),
        num_threads=R,
        dfb_bindings=[E.producer_of("gathered_block", "gathered_block")],
        scratchpad_bindings=[E.ScratchpadBinding(scratchpad_spec_name="token_indices", accessor_name="token_indices")],
        tensor_bindings=[
            E.TensorBinding(tensor_parameter_name="indices", accessor_name="indices"),
            E.TensorBinding(tensor_parameter_name="table", accessor_name="table"),
        ],
        compile_time_args=ct_args,
        runtime_arg_schema=schema,
    )
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "embedding_gen_writer.cpp"),
        hw_config=_dm_config(E.create_writer_datamovement_config, ["gathered_block"]),
        num_threads=W,
        dfb_bindings=[E.consumer_of("gathered_block", "gathered_block")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="out", accessor_name="out")],
        compile_time_args=ct_args,
        runtime_arg_schema=schema,
    )
    spec = E.ProgramSpec(
        name="embedding_gen",
        kernels=[reader, writer],
        dataflow_buffers=[ring],
        scratchpads=[scratch],
        tensor_parameters=[
            E.TensorParameter(unique_id=name, spec=t.spec)
            for name, t in (("indices", indices), ("table", table), ("out", out))
        ],
        work_units=[E.WorkUnitSpec(name="main", kernels=["reader", "writer"], target_nodes=core_set)],
    )
    per_core = {"start_unit": starts, "num_units": counts}
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel=k, runtime_arg_values=per_core, common_runtime_arg_values=common)
            for k in ("reader", "writer")
        ]
    )
    return spec, run_args, {"indices": 0, "table": 1, "out": 2}

