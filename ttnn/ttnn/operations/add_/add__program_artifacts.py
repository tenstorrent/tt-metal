# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for add_: regime `sharded_resident` (op_design.md, Blocking Model).

Both operands are height-sharded in L1 with one shard spec. Every ring is BORROWED from a caller's shard
(`dfb_spec_from_sharded_tensor(..., borrowed_from=...)`), so the op allocates no L1 and moves nothing over
the NoC or to DRAM:

    cb_a_shard       borrowed from a   reader (credit only) -> compute (unpack srcA)
    cb_b_shard       borrowed from b   reader (credit only) -> compute (unpack srcB)
    cb_result_shard  borrowed from a   compute (pack, over the tile just unpacked) -> writer (ack only)

Blocking: core k of the caller's shard map owns tile rows [k * shard_row_tiles, ...) of the tensor
(`block_row_tiles` = the whole resident shard), every column (`block_col_tiles` = tensor_col_tiles), and
deals its tiles round-robin to `compute_threads` NEOs on STRIDED rings; NEO t takes its whole share in one
helper call with IterationShape block size `neo_block_tiles`. Cores whose shard holds no tile get no
kernels (rule R8).
"""

from __future__ import annotations

import math
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental
TILE = 32

# Thread plan (R readers, C NEOs, W writers) per cluster: op_design.md, Work Distribution. The op is
# compute-bound in this regime (zero bytes move), so C = 4 splits the tile walk over all NEOs; the data
# movement harts only post credits, so one of each is enough. The one place the plan is stated.
THREAD_PLAN = (1, 4, 1)

# Ring names (DFB unique ids and kernel accessor names) and tensor-parameter names.
DFB_A = "cb_a_shard"
DFB_B = "cb_b_shard"
DFB_RESULT = "cb_result_shard"
TP_A = "a"
TP_B = "b"


def tensor_tiles(shape):
    """(tensor_row_tiles, tensor_col_tiles): every leading slab is tile-padded independently."""
    shape = [1] * (2 - len(shape)) + [int(d) for d in shape]
    return math.prod(shape[:-2]) * -(-shape[-2] // TILE), -(-shape[-1] // TILE)


def shard_cores(shard_spec):
    """The cores of a shard grid in shard order: core k holds the k-th shard."""
    return list(ttnn.corerange_to_cores(shard_spec.grid, None, shard_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR))


def core_tile_counts(shape, shard_spec):
    """{core: tiles of the tensor its shard holds} over the shard grid, in shard order (rule R8: a grid may
    hold more rows than the tensor, so the last shards are partly filled or empty)."""
    row_tiles, col_tiles = tensor_tiles(shape)
    shard_row_tiles = shard_spec.shape[0] // TILE
    return {
        core: max(0, min(row_tiles - k * shard_row_tiles, shard_row_tiles)) * col_tiles
        for k, core in enumerate(shard_cores(shard_spec))
    }


def clamp_thread_plan(plan, shard_tiles, max_core_tiles):
    """The plan's NEO count clamped so that it divides the ring (a STRIDED ring's capacity is per NEO and the
    tile -> NEO deal must match the shard layout) and is no larger than the busiest core's tile count; the
    harts then clamped to divisors of it (R | C, W | C)."""
    r, c, w = plan
    while c > 1 and (shard_tiles % c != 0 or c > max_core_tiles):
        c //= 2
    return math.gcd(r, c), c, math.gcd(w, c)


def _explicit(make, rings):
    # The data-movement kernels keep the credits of their rings themselves (reserve / push, wait / pop):
    # those rings are opted out of implicit sync. They issue no NoC transfer, so no tag is ever set.
    hw = make()
    hw.config_2xx = E.DataMovementHardwareConfig.DataMovement2XXConfig(disable_dfb_implicit_sync_for=rings)
    return hw


def create_program_artifacts(a, b, *, pack_relu, compute_kernel_config, thread_plan=THREAD_PLAN, tag_threads=False):
    """(spec, run_args, tensor_args) for `a <- act(a + b)` on height-sharded L1 operands.

    `tag_threads` is a test probe: odd NEOs subtract instead of add, so a test sees which NEO computed which
    tile.
    """
    shard_spec = a.memory_config().shard_spec
    counts = core_tile_counts(tuple(a.shape), shard_spec)
    engaged = [core for core, n in counts.items() if n > 0]
    shard_tiles = (shard_spec.shape[0] // TILE) * (shard_spec.shape[1] // TILE)
    R, C, W = clamp_thread_plan(thread_plan, shard_tiles, max(counts.values()))
    # The per-NEO ring capacity: each NEO's whole share fits one helper call (block size of the tile walk).
    neo_block_tiles = shard_tiles // C

    dfbs = [
        E.dfb_spec_from_sharded_tensor(DFB_A, a, borrowed_from=TP_A),
        E.dfb_spec_from_sharded_tensor(DFB_B, b, borrowed_from=TP_B),
        E.dfb_spec_from_sharded_tensor(DFB_RESULT, a, borrowed_from=TP_A),
    ]
    tensor_parameters = [E.TensorParameter(unique_id=TP_A, spec=a.spec), E.TensorParameter(unique_id=TP_B, spec=b.spec)]
    plan = {"reader_threads": R, "compute_threads": C, "writer_threads": W}
    schema = E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["core_tiles"])

    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "add__reader.cpp"),
        hw_config=_explicit(E.create_reader_datamovement_config, [DFB_A, DFB_B]),
        num_threads=R,
        dfb_bindings=[E.producer_of(DFB_A, DFB_A), E.producer_of(DFB_B, DFB_B)],
        compile_time_args=plan,
        runtime_arg_schema=schema,
    )
    cfg = compute_kernel_config
    compute = E.KernelSpec(
        unique_id="compute",
        source=str(KERNEL_DIR / "add__compute.cpp"),
        # Fidelity is always set explicitly.
        hw_config=E.ComputeHardwareConfig(
            fpu_math_fidelity=cfg.math_fidelity,
            enable_32_bit_dest=bool(cfg.fp32_dest_acc_en),
            sfpu_precision_mode=E.Precision.Approximate if cfg.math_approx_mode else E.Precision.Precise,
        ),
        num_threads=C,
        dfb_bindings=[E.consumer_of(DFB_A, DFB_A), E.consumer_of(DFB_B, DFB_B), E.producer_of(DFB_RESULT, DFB_RESULT)],
        compile_time_args={
            **plan,
            "neo_block_tiles": neo_block_tiles,
            "pack_relu": int(bool(pack_relu)),
            "tag_threads": int(bool(tag_threads)),
        },
        runtime_arg_schema=schema,
    )
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "add__writer.cpp"),
        hw_config=_explicit(E.create_writer_datamovement_config, [DFB_RESULT]),
        num_threads=W,
        dfb_bindings=[E.consumer_of(DFB_RESULT, DFB_RESULT)],
        compile_time_args=plan,
        runtime_arg_schema=schema,
    )
    nodes = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in engaged])
    spec = E.ProgramSpec(
        name="add_",
        kernels=[reader, compute, writer],
        dataflow_buffers=dfbs,
        tensor_parameters=tensor_parameters,
        work_units=[E.WorkUnitSpec(name="main", kernels=["reader", "compute", "writer"], target_nodes=nodes)],
    )
    core_args = {"core_tiles": {core: counts[core] for core in engaged}}
    run_args = E.ProgramRunArgs(
        kernel_run_args=[E.KernelRunArgs(kernel=k, runtime_arg_values=core_args) for k in ("reader", "compute", "writer")]
    )
    # a last: generic_op returns the last tensor, the in-place output.
    return spec, run_args, {TP_B: 0, TP_A: 1}
