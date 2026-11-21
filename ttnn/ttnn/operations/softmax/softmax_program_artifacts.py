# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for softmax along dim=-2 (regime `column_resident`, op_design.md).

Unit of work: a tile-column u = nc * Wt + wt (its Ht tiles). Columns are dealt to NEOs one per round: in round r
of a cluster, NEO t computes column col_start + r*C + t when that column exists, else NEO 0's column of the
round (recomputed and stored again with identical bytes). The host deals whole rounds of C columns to the
clusters, so at most C - 1 columns are recomputed in the whole op.

A NEO keeps its columns resident in `cb_input_tiles` for three block operations (column max, exp-sum, normalize),
`block_cols` columns (rounds) per block. Every knob below is one host constant or one value derived by
select_blocking(); the kernels take them as compile-time arguments.
"""

from __future__ import annotations

import math
from pathlib import Path

import ttnn
from ttnn.operations._op_contract import SupportRefusal

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# Thread plan (R readers, C compute NEOs, W writers) per cluster: compute-bound, every input tile read once.
THREAD_PLAN = (2, 4, 2)
FREE_DM_HARTS = 6  # target page, cluster model: DM0 serves ring interrupts, DM1 is the remapper

# Ring cap on the compute side (target page, cluster model). No run-time query exists.
DFB_RING_MAX_BYTES = 1_048_560
L1_RESERVE_BYTES = 0

TILE_HW = 32
TILE_F32_BYTES = 4096  # a Float32 stat tile (m, 1/s)

OUT_DEPTH = 2  # output chunks in flight per NEO: the writer drains one while compute packs the next
OUT_CHUNK_MAX_TILES = 8  # the largest DEST section (fp32 DEST, full sync)
IN_DEPTH_MULTI = 2  # input blocks in flight per NEO when a NEO's share needs several blocks
BLOCK_COLS_CAP = None  # test override: cap on block_cols (None = unbounded)

NEG_BIG_BF16 = 0xFF7F  # lowest finite bfloat16: the additive mask value of a padding row


def validate_thread_plan(threads):
    r, c, w = threads
    if not (c in (1, 2, 4) and r >= 1 and w >= 1 and r + w <= FREE_DM_HARTS and c % r == 0 and c % w == 0):
        raise ValueError(f"thread plan (R, C, W) = {threads}: need C in (1, 2, 4), R + W <= {FREE_DM_HARTS}, R | C, W | C")


def _geometry(x):
    padded = list(x.padded_shape)
    logical = list(x.shape)
    Ht = padded[-2] // TILE_HW
    Wt = padded[-1] // TILE_HW
    NC = math.prod(padded[:-2])
    h_valid_last = logical[-2] - (Ht - 1) * TILE_HW
    return Ht, Wt, NC, h_valid_last


def _budget():
    return ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES


def _footprint(bc, depth, Ht, tile_in, C):
    """L1 bytes per cluster of every ring (l1_ledger.md, footprint per cluster)."""
    return C * (
        depth * bc * Ht * tile_in  # cb_input_tiles
        + 2 * bc * TILE_F32_BYTES  # cb_col_max + cb_col_recip
        + OUT_DEPTH * min(bc * Ht, OUT_CHUNK_MAX_TILES) * tile_in  # cb_output_tiles
        + tile_in  # cb_h_mask
    )


def _fits(bc, depth, Ht, tile_in, C, budget):
    return (
        depth * bc * Ht * tile_in * C <= DFB_RING_MAX_BYTES
        and bc * TILE_F32_BYTES * C <= DFB_RING_MAX_BYTES
        and OUT_DEPTH * min(bc * Ht, OUT_CHUNK_MAX_TILES) * tile_in * C <= DFB_RING_MAX_BYTES
        and _footprint(bc, depth, Ht, tile_in, C) <= budget
    )


def select_regime(x, threads=None):
    """`column_resident` iff a whole column fits one ring with one block of one column; else refuse."""
    C = (threads or THREAD_PLAN)[1]
    Ht, _, _, _ = _geometry(x)
    tile_in = x.buffer_page_size()
    if not _fits(1, 1, Ht, tile_in, C, _budget()):
        raise SupportRefusal(
            f"softmax: a column of {Ht} tiles does not stay resident on one cluster "
            "(column_streaming is a later refinement)"
        )
    return "column_resident"


def select_blocking(max_rounds, Ht, tile_in, C):
    """(block_cols, in_depth): the whole per-NEO share as one block when it fits, else the largest block."""
    budget = _budget()
    cap = max_rounds if BLOCK_COLS_CAP is None else min(max_rounds, BLOCK_COLS_CAP)
    if cap == max_rounds and _fits(max_rounds, 1, Ht, tile_in, C, budget):
        return max_rounds, 1
    for depth in (IN_DEPTH_MULTI, 1):
        for bc in range(cap, 0, -1):
            if _fits(bc, depth, Ht, tile_in, C, budget):
                return bc, depth
    raise SupportRefusal("softmax: no block of one column fits (column_streaming is a later refinement)")


def split_columns(num_cols, grid_size, C):
    """(core set, {core: rounds}, {core: real columns}, {core: first column}): whole rounds of C to clusters."""
    total_rounds = -(-num_cols // C)
    _, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(
        grid_size, total_rounds, row_wise=True
    )
    rounds, cols, starts, start = {}, {}, {}, 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            rounds[core] = per_core
            cols[core] = min(per_core * C, num_cols - start)
            starts[core] = start
            start += cols[core]
    return all_cores, rounds, cols, starts


def _dm_config(make, explicit_rings):
    hw = make()
    if explicit_rings:
        hw.config_2xx = E.DataMovementHardwareConfig.DataMovement2XXConfig(disable_dfb_implicit_sync_for=explicit_rings)
    return hw


def create_program_artifacts(x, out, compute_config, *, threads=None):
    threads = threads or THREAD_PLAN  # read at call time: the one place the plan is stated
    validate_thread_plan(threads)
    R, C, W = threads
    Ht, Wt, NC, h_valid_last = _geometry(x)
    tile_in = x.buffer_page_size()
    num_cols = NC * Wt
    has_mask = int(h_valid_last < TILE_HW)

    core_set, rounds, cols, starts = split_columns(num_cols, x.device().compute_with_storage_grid_size(), C)
    max_rounds = max(rounds.values())
    block_cols, in_depth = select_blocking(max_rounds, Ht, tile_in, C)
    out_chunk_tiles = min(block_cols * Ht, OUT_CHUNK_MAX_TILES)

    # STRIDED rings are sliced per NEO: every capacity is entries per NEO x C.
    def ring(name, entries_per_neo, entry_size, fmt):
        return E.DataflowBufferSpec(
            unique_id=name, entry_size=entry_size, num_entries=entries_per_neo * C, data_format_metadata=fmt
        )

    dfbs = [
        ring("cb_input_tiles", in_depth * block_cols * Ht, tile_in, x.dtype),
        ring("cb_h_mask", 1, tile_in, x.dtype),
        ring("cb_col_max", block_cols, TILE_F32_BYTES, ttnn.float32),
        ring("cb_col_recip", block_cols, TILE_F32_BYTES, ttnn.float32),
        ring("cb_output_tiles", OUT_DEPTH * out_chunk_tiles, tile_in, out.dtype),
    ]
    tensor_parameters = [E.TensorParameter(unique_id=n, spec=t.spec) for n, t in (("input", x), ("output", out))]

    plan = {"reader_threads": R, "compute_threads": C, "writer_threads": W}
    geometry = {"Ht": Ht, "Wt": Wt}
    dm_schema = E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["rounds", "num_cols", "col_start"])

    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "softmax_reader.cpp"),
        hw_config=_dm_config(E.create_reader_datamovement_config, ["cb_h_mask"]),
        num_threads=R,
        dfb_bindings=[
            E.producer_of("cb_input_tiles", "cb_input_tiles"),
            E.producer_of("cb_h_mask", "cb_h_mask"),
        ],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="input", accessor_name="input")],
        compile_time_args={
            **plan,
            **geometry,
            "has_mask": has_mask,
            "h_valid_last": h_valid_last,
            "neg_big": NEG_BIG_BF16,
        },
        runtime_arg_schema=dm_schema,
    )
    approx = bool(compute_config.math_approx_mode)
    compute = E.KernelSpec(
        unique_id="compute",
        source=str(KERNEL_DIR / "softmax_compute.cpp"),
        hw_config=E.ComputeHardwareConfig(
            fpu_math_fidelity=compute_config.math_fidelity,  # always explicit
            sfpu_precision_mode=E.Precision.Approximate if approx else E.Precision.Precise,
            enable_32_bit_dest=bool(compute_config.fp32_dest_acc_en),
            # The Float32 stat rings are reloaded through SrcA (default unpack), not with UnpackToDest.
            unpack_modes={name: E.UnpackMode.UnpackToSrc for name in ("cb_col_max", "cb_col_recip")},
        ),
        num_threads=C,
        dfb_bindings=[
            E.consumer_of("cb_input_tiles", "cb_input_tiles"),
            E.consumer_of("cb_h_mask", "cb_h_mask"),
            E.producer_of("cb_col_max", "cb_col_max"),
            E.consumer_of("cb_col_max", "cb_col_max"),
            E.producer_of("cb_col_recip", "cb_col_recip"),
            E.consumer_of("cb_col_recip", "cb_col_recip"),
            E.producer_of("cb_output_tiles", "cb_output_tiles"),
        ],
        compile_time_args={
            **plan,
            "Ht": Ht,
            "block_cols": block_cols,
            "has_mask": has_mask,
            "exp_approx": int(approx),
        },
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["rounds"]),
    )
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "softmax_writer.cpp"),
        hw_config=_dm_config(E.create_writer_datamovement_config, []),
        num_threads=W,
        dfb_bindings=[E.consumer_of("cb_output_tiles", "cb_output_tiles")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="output", accessor_name="output")],
        compile_time_args={**plan, **geometry},
        runtime_arg_schema=dm_schema,
    )
    spec = E.ProgramSpec(
        name="softmax",
        kernels=[reader, compute, writer],
        dataflow_buffers=dfbs,
        tensor_parameters=tensor_parameters,
        work_units=[E.WorkUnitSpec(name="main", kernels=["reader", "compute", "writer"], target_nodes=core_set)],
    )
    dm_args = {"rounds": rounds, "num_cols": cols, "col_start": starts}
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel="reader", runtime_arg_values=dm_args),
            E.KernelRunArgs(kernel="compute", runtime_arg_values={"rounds": rounds}),
            E.KernelRunArgs(kernel="writer", runtime_arg_values=dm_args),
        ]
    )
    return spec, run_args, {"input": 0, "output": 1}
