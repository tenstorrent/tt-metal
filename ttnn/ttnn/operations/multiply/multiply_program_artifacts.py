# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts (`ttnn.ProgramSpec`) for multiply: one program, reader / compute / writer on
every cluster that gets work, in one of two regimes chosen on the host (op_design.md, Blocking Model):

    flat_stream        broadcast "none": a, b and out share page ids; the output's tiles are split into
                       contiguous runs over the clusters and streamed (every tensor crosses once).
    row_bcast_strips   broadcast "over_h": b has one tile row per image. Units are column strips
                       (n, g, c); the clusters get contiguous unit runs; a block is `rows x cols` tiles of
                       one image and row group, and b's `cols` tiles are read once per block and held while
                       compute walks the block's rows (b crosses G times, a and out once).

Every block knob is defined once here: THREAD_PLAN, STREAM_DEPTH, B_WINDOW_DEPTH, and the per-call
`block_cols` and `row_groups` derivations. Ring capacities, compile-time and runtime args derive from them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# Thread plan (R readers, C NEOs, W writers) per cluster. Bound by data movement (one FPU op per tile against
# three tile transfers), so one thread of each kind sits on the DRAM floor (target page, Choosing the plan;
# toy_thread_plan measured (1,1,1) = (2,2,2) = (4,4,2) within 1.6%). flat_stream runs under any legal plan;
# row_bcast_strips deals whole blocks and requires (1, 1, 1).
THREAD_PLAN = (1, 1, 1)

# Cluster model of the target page (.claude/references/targets/quasar.md); Python has no query for these.
NEOS = 4
FREE_DM_HARTS = 6
DFB_RING_MAX_BYTES = 1_048_560

# Ring depths, in entries per NEO.
STREAM_DEPTH = 4  # a, b (flat) and out rings: two implicit-sync txn-id halves of 2
B_WINDOW_DEPTH = 2  # b windows in the row_bcast ring: current + prefetched; >= 2 is a liveness requirement

REGIMES = {"flat_stream": 0, "row_bcast_strips": 1}

# Extra L1 the op leaves free beyond its rings (a refinement knob; 0 in Phase 0).
L1_RESERVE_BYTES = 0


def validate_thread_plan(threads):
    r, c, w = threads
    if not (1 <= c <= NEOS and r >= 1 and w >= 1 and r + w <= FREE_DM_HARTS and c % r == 0 and c % w == 0):
        raise ValueError(
            f"thread plan (R, C, W) = {threads}: need C <= {NEOS}, R + W <= {FREE_DM_HARTS}, R | C, W | C"
        )


def pad4(shape):
    shape = [int(d) for d in shape]
    if len(shape) > 4:
        raise NotImplementedError(f"multiply: rank above 4 is not supported (shape {shape})")
    return [1] * (4 - len(shape)) + shape


@dataclass
class Geometry:
    """The output's tile grid [N_out, Ht, Wt] and b's image strides over the two leading output dims."""

    o1: int
    n_out: int
    ht: int
    wt: int
    sb0: int
    sb1: int


def geometry(a_shape, b_shape, out_shape):
    o0, o1, h, w = pad4(out_shape)
    b0, b1, _, _ = pad4(b_shape)
    # b's image index for output image n = d0 * o1 + d1; an operand of size 1 along a dim does not step.
    sb1 = 0 if b1 == 1 else 1
    sb0 = 0 if b0 == 1 else b1
    return Geometry(o1=o1, n_out=o0 * o1, ht=math.ceil(h / 32), wt=math.ceil(w / 32), sb0=sb0, sb1=sb1)


def select_regime(broadcast: str) -> str:
    """The regime-selection function of the design (exact)."""
    if broadcast == "none":
        return "flat_stream"
    if broadcast == "over_h":
        return "row_bcast_strips"
    raise NotImplementedError(f"multiply: no regime for broadcast={broadcast!r}")


def split_units(num_units, grid_size):
    """(core set, {core: units}, {core: first unit}) over the clusters that get work, row-major."""
    _, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(
        grid_size, num_units, row_wise=True
    )
    counts, starts, start = {}, {}, 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            counts[core], starts[core] = per_core, start
            start += per_core
    return all_cores, counts, starts


def row_groups(num_cores, geo: Geometry) -> int:
    """G: split H across cores only when the (n, c) strips under-fill the grid (each extra group re-reads b)."""
    return max(1, min(math.ceil(num_cores / (geo.n_out * geo.wt)), geo.ht))


def block_cols_cap(a_tile_bytes, b_tile_bytes, out_tile_bytes, C) -> int:
    """Largest b window (in b tiles) whose ring fits one DFB ring and, beside the a and out streams, the L1
    budget. Each ring is sized with its own page size (a, b and out may differ in dtype)."""
    budget = ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES
    stream_bytes = STREAM_DEPTH * C * (a_tile_bytes + out_tile_bytes)
    return max(1, min(DFB_RING_MAX_BYTES, budget - stream_bytes) // (B_WINDOW_DEPTH * C * b_tile_bytes))


def compute_hardware_config(cfg):
    """The caller's compute config, passed through whole (GUIDANCE: `config=compute_kernel_config`); validate()
    gates only fp32_dest_acc_en, and math_fidelity / math_approx_mode are taken as given (R4)."""
    return E.ComputeHardwareConfig(
        fpu_math_fidelity=cfg.math_fidelity,
        sfpu_precision_mode=E.Precision.Approximate if cfg.math_approx_mode else E.Precision.Precise,
        enable_32_bit_dest=bool(cfg.fp32_dest_acc_en),
        double_buffer_dest=not bool(cfg.dst_full_sync_en),
    )


@dataclass
class Plan:
    """Everything the host decided for one call; tests read it to see which regime and split ran."""

    regime: str
    threads: tuple
    geo: Geometry
    G: int
    num_units: int
    block_cols: int
    core_set: object
    counts: dict
    starts: dict


def make_plan(a, b, out, broadcast: str, threads=THREAD_PLAN) -> Plan:
    validate_thread_plan(threads)
    R, C, W = threads
    regime = select_regime(broadcast)
    geo = geometry(list(a.shape), list(b.shape), list(out.shape))
    grid = a.device().compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    if regime == "flat_stream":
        G = 1
        num_units = geo.n_out * geo.ht * geo.wt
        if num_units != out.buffer_num_pages():
            raise RuntimeError(f"multiply: tile count {num_units} != output pages {out.buffer_num_pages()}")
        core_set, counts, starts = split_units(num_units, grid)
        block_cols = 1  # not a block axis of flat_stream (the run is one block)
    else:
        if threads != (1, 1, 1):
            raise ValueError(f"row_bcast_strips deals whole blocks and needs the plan (1, 1, 1), got {threads}")
        G = row_groups(num_cores, geo)
        num_units = geo.n_out * G * geo.wt
        core_set, counts, starts = split_units(num_units, grid)
        max_units_per_core = max(counts.values())
        cap = block_cols_cap(a.buffer_page_size(), b.buffer_page_size(), out.buffer_page_size(), C)
        block_cols = min(geo.wt, max_units_per_core, cap)
    return Plan(regime, threads, geo, G, num_units, block_cols, core_set, counts, starts)


def _dm_config(make, explicit_rings):
    hw = make()
    if explicit_rings:
        hw.config_2xx = E.DataMovementHardwareConfig.DataMovement2XXConfig(disable_dfb_implicit_sync_for=explicit_rings)
    return hw


def create_program_artifacts(a, b, out, *, broadcast: str, compute_kernel_config, threads=THREAD_PLAN):
    plan = make_plan(a, b, out, broadcast, threads)
    R, C, W = plan.threads
    row = plan.regime == "row_bcast_strips"

    b_entries = (B_WINDOW_DEPTH * plan.block_cols if row else STREAM_DEPTH) * C
    dfbs = [
        E.DataflowBufferSpec(
            unique_id="in_a", entry_size=a.buffer_page_size(), num_entries=STREAM_DEPTH * C,
            data_format_metadata=a.dtype,
        ),
        E.DataflowBufferSpec(
            unique_id="in_b", entry_size=b.buffer_page_size(), num_entries=b_entries, data_format_metadata=b.dtype
        ),
        E.DataflowBufferSpec(
            unique_id="out", entry_size=out.buffer_page_size(), num_entries=STREAM_DEPTH * C,
            data_format_metadata=out.dtype,
        ),
    ]
    tensor_parameters = [E.TensorParameter(unique_id=n, spec=t.spec) for n, t in (("a", a), ("b", b), ("out", out))]
    ct = {
        "regime": REGIMES[plan.regime],
        "reader_threads": R,
        "compute_threads": C,
        "writer_threads": W,
        "block_cols": plan.block_cols,
    }
    common_names = ["G", "Ht", "Wt", "o1", "sb0", "sb1"]
    common_values = {
        "G": plan.G, "Ht": plan.geo.ht, "Wt": plan.geo.wt, "o1": plan.geo.o1, "sb0": plan.geo.sb0, "sb1": plan.geo.sb1,
    }
    schema = E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["start", "count"], common_runtime_arg_names=common_names)

    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "multiply_reader.cpp"),
        # row_bcast_strips: the b window is assembled from several reads, so the reader keeps its credits.
        hw_config=_dm_config(E.create_reader_datamovement_config, ["in_b"] if row else []),
        num_threads=R,
        dfb_bindings=[E.producer_of("in_a", "in_a"), E.producer_of("in_b", "in_b")],
        tensor_bindings=[
            E.TensorBinding(tensor_parameter_name="a", accessor_name="a"),
            E.TensorBinding(tensor_parameter_name="b", accessor_name="b"),
        ],
        compile_time_args=ct,
        runtime_arg_schema=schema,
    )
    compute = E.KernelSpec(
        unique_id="compute",
        source=str(KERNEL_DIR / "multiply_compute.cpp"),
        hw_config=compute_hardware_config(compute_kernel_config),
        num_threads=C,
        dfb_bindings=[E.consumer_of("in_a", "in_a"), E.consumer_of("in_b", "in_b"), E.producer_of("out", "out")],
        compile_time_args=ct,
        runtime_arg_schema=schema,
    )
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "multiply_writer.cpp"),
        hw_config=_dm_config(E.create_writer_datamovement_config, []),
        num_threads=W,
        dfb_bindings=[E.consumer_of("out", "out")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="out", accessor_name="out")],
        compile_time_args=ct,
        runtime_arg_schema=schema,
    )
    spec = E.ProgramSpec(
        name=f"multiply_{plan.regime}",
        kernels=[reader, compute, writer],
        dataflow_buffers=dfbs,
        tensor_parameters=tensor_parameters,
        work_units=[E.WorkUnitSpec(name="main", kernels=["reader", "compute", "writer"], target_nodes=plan.core_set)],
    )
    rt = {"start": plan.starts, "count": plan.counts}
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel=k, runtime_arg_values=rt, common_runtime_arg_values=common_values)
            for k in ("reader", "compute", "writer")
        ]
    )
    return spec, run_args, {"a": 0, "b": 1, "out": 2}, plan
