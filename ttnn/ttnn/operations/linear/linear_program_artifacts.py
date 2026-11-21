# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for linear (regime R-2D-stream of op_design.md).

Index space, in tiles: rows `m_t` (every image's rows, tile-padded), columns `n_t`, contraction `k_t`.

  block   [block_h x block_w] output tiles, owned by one NEO, accumulated in DEST over the whole of K,
          which streams through in chunks of block_k tiles. With a bias, the NEO first copies the bias row
          (broadcast down the 32 rows) into DEST, and the matmul accumulates onto it.
  round   C blocks with the same rows and C consecutive column blocks, one per NEO of a cluster. Its
          activation chunk reaches the cluster once (an ALL ring: every NEO sees every entry); each NEO
          takes its own weight block from a BLOCKED ring. In the last round of a row block a NEO without a
          real column block recomputes NEO 0's, and the writer stores the same bytes again.
  split   whole rounds are dealt to clusters (split_work_to_cores), so padding happens once per row block.

Thread plan (one host constant, THREAD_PLAN): one activation-reader hart, one weight-reader hart (a matmul
operand ring must have one producer thread), C NEOs and W
writer harts. Every count reaches the kernels as a compile-time argument.

Every knob is defined once here: the block extents come from `select_blocking`, the ring depths from the
*_DEPTH constants, the thread counts from THREAD_PLAN, and every ring capacity and loop count derives from
them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

TILE = 32


@dataclass(frozen=True)
class ThreadPlan:
    act_readers: int
    weight_readers: int
    compute: int
    writers: int

    @property
    def readers(self) -> int:
        return self.act_readers + self.weight_readers


# The cluster model of the target page (.claude/references/targets/quasar.md); Python has no query for them.
FREE_DM_HARTS = 6  # of 8: DM0 serves the ring interrupts, DM1 the remapper
NEOS = 4
# The largest ring on the compute side (target page, Cluster model). No run-time query exists.
DFB_RING_CAP_BYTES = 1_048_560
# The largest block of a BLOCKED binding with a Tensix endpoint (DataflowBuffer.md, C2).
BLOCKED_BLOCK_CAP = 1023
# L1 kept back from the unreserved region for the runtime's own use.
L1_RESERVE_BYTES = 65536

# The plan of op_design.md, Work Distribution: (R, C, W) = (1 + 1, 4, 2).
THREAD_PLAN = ThreadPlan(act_readers=1, weight_readers=1, compute=4, writers=2)

# Ring depths, in units of each ring's live set (op_design.md, Buffer-depth knobs).
ACT_DEPTH = 2  # K chunks of the activation in flight
WEIGHT_DEPTH = 2  # K chunks of the weight in flight, per NEO
BIAS_DEPTH = 2  # rounds of bias in flight, per NEO
OUT_DEPTH = 2  # packed blocks in flight, per NEO


def validate_thread_plan(plan: ThreadPlan):
    c, w = plan.compute, plan.writers
    if not (
        c in (1, 2, 4)
        and plan.act_readers == 1
        and plan.weight_readers == 1
        and w >= 1
        and plan.readers + w <= FREE_DM_HARTS
        and c % w == 0
    ):
        raise ValueError(
            f"linear: thread plan {plan}: need C in (1, 2, 4), one reader hart per matmul operand, "
            f"R + W <= {FREE_DM_HARTS} and W | C"
        )


def _divisors(n: int):
    return [d for d in range(1, n + 1) if n % d == 0]


@dataclass(frozen=True)
class BlockPlan:
    m_t: int  # output tile-rows (all images)
    n_t: int  # output tile-columns
    k_t: int  # contraction tiles
    block_h: int
    block_w: int
    block_k: int
    threads: ThreadPlan
    has_bias: bool
    tile_bytes: int

    @property
    def num_row_blocks(self) -> int:
        return self.m_t // self.block_h

    @property
    def col_blocks(self) -> int:
        return self.n_t // self.block_w

    @property
    def rounds_per_row_block(self) -> int:
        return -(-self.col_blocks // self.threads.compute)

    @property
    def num_rounds(self) -> int:
        return self.num_row_blocks * self.rounds_per_row_block

    # Ring capacities, in entries (tiles). A ring with a BLOCKED or STRIDED compute side counts per NEO x C.
    @property
    def act_entries(self) -> int:
        return ACT_DEPTH * self.block_h * self.block_k  # ALL: one copy serves every NEO

    @property
    def weight_entries(self) -> int:
        return WEIGHT_DEPTH * self.threads.compute * self.block_k * self.block_w

    @property
    def bias_entries(self) -> int:
        return BIAS_DEPTH * self.threads.compute * self.block_w

    @property
    def out_entries(self) -> int:
        return OUT_DEPTH * self.threads.compute * self.block_h * self.block_w

    def footprint_bytes(self) -> int:
        bias = self.bias_entries if self.has_bias else 0
        return self.tile_bytes * (self.act_entries + self.weight_entries + bias + self.out_entries)


def select_blocking(m_t, n_t, k_t, fp32_dest_acc_en, threads, num_cores, budget, tile_bytes, has_bias):
    """op_design.md, `select_blocking`: occupancy first, then the coarsest block, then the taller one; then
    the coarsest K chunk that fits the ring caps and the L1 budget."""
    C = threads.compute
    dest_tiles = 4 if fp32_dest_acc_en else 8  # half-sync DEST (double_buffer_dest=True)
    best_key, best = None, None
    for bh in _divisors(m_t):
        for bw in _divisors(n_t):
            if bh * bw > dest_tiles:
                continue
            col_blocks = n_t // bw
            num_rounds = (m_t // bh) * -(-col_blocks // C)
            key = (min(num_rounds, num_cores) * min(col_blocks, C), bh * bw, bh)
            if best_key is None or key > best_key:
                best_key, best = key, (bh, bw)
    bh, bw = best
    for d in sorted(_divisors(k_t), reverse=True):
        plan = BlockPlan(m_t, n_t, k_t, bh, bw, d, threads, has_bias, tile_bytes)
        if (
            plan.act_entries * tile_bytes <= DFB_RING_CAP_BYTES
            and plan.weight_entries * tile_bytes <= DFB_RING_CAP_BYTES
            and d * bw <= BLOCKED_BLOCK_CAP
            and plan.footprint_bytes() <= budget
        ):
            return plan
    raise NotImplementedError(
        f"linear: no K chunk fits L1 for blocks of {bh}x{bw} tiles "
        f"(footprint at block_k=1: {BlockPlan(m_t, n_t, k_t, bh, bw, 1, threads, has_bias, tile_bytes).footprint_bytes()} B, "
        f"budget {budget} B)"
    )


def plan_blocking(a, b, compute_config, device, threads=None, has_bias=False):
    """The BlockPlan of one call on this device."""
    threads = threads or THREAD_PLAN
    validate_thread_plan(threads)
    a_pad, b_pad = tuple(a.padded_shape), tuple(b.padded_shape)
    images = math.prod(a_pad[:-2]) if len(a_pad) > 2 else 1
    m_t = images * (a_pad[-2] // TILE)
    k_t = a_pad[-1] // TILE
    n_t = b_pad[-1] // TILE
    grid = device.compute_with_storage_grid_size()
    budget = ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES
    return select_blocking(
        m_t, n_t, k_t, bool(compute_config.fp32_dest_acc_en), threads, grid.x * grid.y, budget,
        a.buffer_page_size(), has_bias,
    )


def split_rounds(num_rounds, grid_size):
    """(core set, {core: rounds}, {core: first round}): whole rounds dealt to the clusters, row-major."""
    _, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(
        grid_size, num_rounds, row_wise=True
    )
    counts, starts, start = {}, {}, 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            counts[core], starts[core] = per_core, start
            start += per_core
    return all_cores, counts, starts


def create_program_artifacts(a, b, bias, out, compute_config, plan: BlockPlan):
    has_bias = bias is not None
    if plan.has_bias != has_bias:
        raise ValueError("linear: the block plan was sized for another bias setting")
    threads = plan.threads
    C, W = threads.compute, threads.writers
    device = a.device()
    core_set, counts, starts = split_rounds(plan.num_rounds, device.compute_with_storage_grid_size())

    def ring(name, entries, dtype):
        return E.DataflowBufferSpec(
            unique_id=name, entry_size=plan.tile_bytes, num_entries=entries, data_format_metadata=dtype
        )

    # Every ring a data-movement kernel fills or drains uses implicit sync (the default on this part): one
    # tagged transfer per entry, no reserve/push or wait/pop on the data-movement side.
    dfbs = [
        ring("cb_act_tiles", plan.act_entries, a.dtype),
        ring("cb_weight_tiles", plan.weight_entries, b.dtype),
        ring("cb_output_tiles", plan.out_entries, out.dtype),
    ]
    if has_bias:
        dfbs.append(ring("cb_bias_tiles", plan.bias_entries, bias.dtype))

    named = [("a", a), ("b", b)] + ([("bias", bias)] if has_bias else []) + [("out", out)]
    tensor_parameters = [E.TensorParameter(unique_id=n, spec=t.spec) for n, t in named]
    tensor_indices = {n: i for i, (n, _) in enumerate(named)}

    blocks = {"block_h": plan.block_h, "block_w": plan.block_w, "block_k": plan.block_k}
    threads_ct = {"compute_threads": C, "writer_threads": W}
    columns = {
        "n_tiles": plan.n_t,
        "col_blocks": plan.col_blocks,
        "rounds_per_row_block": plan.rounds_per_row_block,
    }
    defines = {"LINEAR_HAS_BIAS": "1"} if has_bias else {}
    k_schema = ["num_rounds", "round_start", "k_tile_start", "k_tiles"]

    reader_act = E.KernelSpec(
        unique_id="reader_act",
        source=str(KERNEL_DIR / "linear_reader.cpp"),
        hw_config=E.create_reader_datamovement_config(),
        num_threads=threads.act_readers,
        dfb_bindings=[E.producer_of("cb_act_tiles", "cb_act_tiles")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="a", accessor_name="a")],
        compile_time_args={**blocks, "a_k_tiles": plan.k_t, "rounds_per_row_block": plan.rounds_per_row_block},
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=k_schema),
    )
    weight_bindings = [E.producer_of("cb_weight_tiles", "cb_weight_tiles")]
    weight_tensors = [E.TensorBinding(tensor_parameter_name="b", accessor_name="b")]
    if has_bias:
        weight_bindings.append(E.producer_of("cb_bias_tiles", "cb_bias_tiles"))
        weight_tensors.append(E.TensorBinding(tensor_parameter_name="bias", accessor_name="bias"))
    reader_weight = E.KernelSpec(
        unique_id="reader_weight",
        source=str(KERNEL_DIR / "linear_weight_reader.cpp"),
        hw_config=E.create_reader_datamovement_config(),
        num_threads=threads.weight_readers,
        compiler_options=E.KernelSpec.CompilerOptions(defines=defines),
        dfb_bindings=weight_bindings,
        tensor_bindings=weight_tensors,
        compile_time_args={**blocks, **columns, "compute_threads": C},
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=k_schema),
    )
    compute_bindings = [
        E.all_consumer_of("cb_act_tiles", "cb_act_tiles"),
        E.blocked_consumer_of("cb_weight_tiles", "cb_weight_tiles", plan.block_k * plan.block_w),
        E.producer_of("cb_output_tiles", "cb_output_tiles"),
    ]
    if has_bias:
        compute_bindings.append(E.blocked_consumer_of("cb_bias_tiles", "cb_bias_tiles", plan.block_w))
    compute = E.KernelSpec(
        unique_id="compute",
        source=str(KERNEL_DIR / "linear_compute.cpp"),
        hw_config=E.ComputeHardwareConfig(
            fpu_math_fidelity=compute_config.math_fidelity,  # always explicit
            sfpu_precision_mode=E.Precision.Approximate if compute_config.math_approx_mode else E.Precision.Precise,
            enable_32_bit_dest=bool(compute_config.fp32_dest_acc_en),
            double_buffer_dest=True,
        ),
        num_threads=C,
        compiler_options=E.KernelSpec.CompilerOptions(defines=defines),
        dfb_bindings=compute_bindings,
        compile_time_args={**blocks, **threads_ct},
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_rounds", "k_tiles"]),
    )
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "linear_writer.cpp"),
        hw_config=E.create_writer_datamovement_config(),
        num_threads=W,
        dfb_bindings=[E.consumer_of("cb_output_tiles", "cb_output_tiles")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="out", accessor_name="out")],
        compile_time_args={**blocks, **threads_ct, **columns},
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_rounds", "round_start"]),
    )
    spec = E.ProgramSpec(
        name="linear",
        kernels=[reader_act, reader_weight, compute, writer],
        dataflow_buffers=dfbs,
        tensor_parameters=tensor_parameters,
        work_units=[
            E.WorkUnitSpec(
                name="main", kernels=["reader_act", "reader_weight", "compute", "writer"], target_nodes=core_set
            )
        ],
    )
    # The K range is a runtime argument so that a split-K regime can hand a cluster a sub-range.
    k_tile_start = {core: 0 for core in counts}
    k_tiles = {core: plan.k_t for core in counts}
    reader_args = {"num_rounds": counts, "round_start": starts, "k_tile_start": k_tile_start, "k_tiles": k_tiles}
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel="reader_act", runtime_arg_values=reader_args),
            E.KernelRunArgs(kernel="reader_weight", runtime_arg_values=reader_args),
            E.KernelRunArgs(kernel="compute", runtime_arg_values={"num_rounds": counts, "k_tiles": k_tiles}),
            E.KernelRunArgs(kernel="writer", runtime_arg_values={"num_rounds": counts, "round_start": starts}),
        ]
    )
    return spec, run_args, tensor_indices
