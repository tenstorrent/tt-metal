# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for plus_one: an in-place integer increment on the data-movement harts.

Regime A of op_design.md (`segment_split_in_place`). The tensor's elements are cut into units (row segments:
`seg_elems` elements of one page, a whole row when there are rows enough), the units are dealt contiguously to
the engaged clusters by `split_work_to_cores`, and inside a cluster round-robin to the `WORKER_THREADS` threads
of one data-movement kernel. Each thread is its own reader, incrementer and writer: it reads a block of its
units into its own line-aligned region of one scratchpad, adds one to every 32-bit word with the hart's
integer ALU, and writes the same bytes back where they came from. No DataflowBuffer, no compute kernel.

Every knob of the Blocking Model is one named constant (or one derived quantity) here; the kernel takes the
thread count as a compile-time argument and the geometry as common runtime arguments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# --- Blocking-model knobs (op_design.md, Blocking Model) -------------------------------------------------------
# The thread plan: one data-movement kernel of WORKER_THREADS threads per cluster, no NEO (C = 0).
WORKER_THREADS = 4
# Grid-synchronization floor: below this many bytes another worker does not shorten the critical path.
MIN_WORKER_BYTES = 1024
# Blocks resident per thread region (the threads of a cluster supply the overlap at depth 1).
SCRATCH_DEPTH = 1
# L1 held back from the scratchpad's budget; the budget query already excludes firmware regions.
L1_RESERVE_BYTES = 0
# DM L2 cache line (L2_CACHE_LINE_SIZE, tt_metal/hw/inc/internal/tt-2xx/quasar/overlay/overlay_addresses.h:44).
# Python has no query for it. Thread regions are line-aligned so one thread's invalidate never drops another's
# dirty words.
CACHE_LINE_BYTES = 64
# Bytes per element: the op handles int32 and uint32 alike, as raw 32-bit words.
ELEM_BYTES = 4

KERNEL_NAME = "worker"
TENSOR_PARAM = "t"
SCRATCH_NAME = "scratch_units"


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _align_up(a: int, b: int) -> int:
    return _ceil_div(a, b) * b


@dataclass
class SplitPlan:
    """The host split of one call: every derived quantity of the Blocking Model, from the knobs above."""

    rows: int
    row_elems: int
    seg_elems: int
    segs_per_row: int
    num_units: int
    num_workers: int
    num_cores: int
    worker_threads: int
    slot_stride_bytes: int
    block_units: int
    region_stride_bytes: int
    scratch_bytes: int
    core_set: object = None
    # {core: (unit_start, unit_count)}
    assignment: dict = field(default_factory=dict)

    def unit_extent(self, unit: int):
        """(row, element offset, element count) of a unit; the last segment of a row may be short."""
        row, seg = divmod(unit, self.segs_per_row)
        off = seg * self.seg_elems
        return row, off, min(self.seg_elems, self.row_elems - off)

    def owner_thread(self, unit: int) -> int:
        """The thread (of its cluster's worker kernel) that increments `unit`."""
        for start, count in self.assignment.values():
            if start <= unit < start + count:
                return (unit - start) % self.worker_threads
        raise ValueError(f"unit {unit} is assigned to no cluster")


def split_plan(
    shape,
    *,
    buffer_alignment: int,
    l1_alignment: int,
    grid_size,
    l1_budget: int,
    worker_threads: int = WORKER_THREADS,
    min_worker_bytes: int = MIN_WORKER_BYTES,
    scratch_depth: int = SCRATCH_DEPTH,
) -> SplitPlan:
    """Units, cores and block sizes for a row-major tensor of `shape` (op_design.md, derived quantities)."""
    row_elems = int(shape[-1])
    rows = math.prod(int(d) for d in shape[:-1]) if len(shape) > 1 else 1
    grid_cores = grid_size.x * grid_size.y

    # Per-thread scratch slot: what one thread may hold at once without the scratchpad outgrowing L1.
    max_block_bytes = (
        (l1_budget - CACHE_LINE_BYTES) // (worker_threads * scratch_depth) // CACHE_LINE_BYTES * CACHE_LINE_BYTES
    )
    seg_granule = max(1, buffer_alignment // ELEM_BYTES)
    max_seg_elems = max(seg_granule, max_block_bytes // ELEM_BYTES // seg_granule * seg_granule)

    total_bytes = rows * row_elems * ELEM_BYTES
    workers_wanted = min(max(_ceil_div(total_bytes, max(1, min_worker_bytes)), 1), grid_cores * worker_threads)

    if rows >= workers_wanted:
        seg_elems = row_elems
    else:
        seg_elems = _align_up(_ceil_div(row_elems, _ceil_div(workers_wanted, rows)), seg_granule)
    seg_elems = max(1, min(seg_elems, row_elems, max_seg_elems))
    segs_per_row = _ceil_div(row_elems, seg_elems)
    num_units = rows * segs_per_row
    num_workers = min(workers_wanted, num_units)
    num_cores = min(_ceil_div(num_workers, worker_threads), grid_cores)

    slot_stride_bytes = _align_up(seg_elems * ELEM_BYTES, l1_alignment)
    max_units_per_core = _ceil_div(num_units, num_cores)
    block_units = max(1, min(_ceil_div(max_units_per_core, worker_threads), max_block_bytes // slot_stride_bytes))
    region_stride_bytes = _align_up(scratch_depth * block_units * slot_stride_bytes, CACHE_LINE_BYTES)
    # + one line of slack: the scratchpad's base is only L1-aligned, the kernel line-aligns the first region.
    scratch_bytes = worker_threads * region_stride_bytes + CACHE_LINE_BYTES

    return SplitPlan(
        rows=rows,
        row_elems=row_elems,
        seg_elems=seg_elems,
        segs_per_row=segs_per_row,
        num_units=num_units,
        num_workers=num_workers,
        num_cores=num_cores,
        worker_threads=worker_threads,
        slot_stride_bytes=slot_stride_bytes,
        block_units=block_units,
        region_stride_bytes=region_stride_bytes,
        scratch_bytes=scratch_bytes,
    )


def assign_cores(plan: SplitPlan, grid_size) -> SplitPlan:
    """Engage `plan.num_cores` clusters row-major and deal the units to them in contiguous ranges."""
    engaged = ttnn.num_cores_to_corerangeset(plan.num_cores, grid_size, True)
    _, core_set, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(
        engaged, plan.num_units, row_wise=True
    )
    assignment, start = {}, 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment[core] = (start, per_core)
            start += per_core
    assert start == plan.num_units, (start, plan.num_units)
    plan.core_set = core_set
    plan.assignment = assignment
    return plan


def plan_for(tensor: ttnn.Tensor, **knobs) -> SplitPlan:
    """The split of one call on `tensor`, every predicate from a device query."""
    device = tensor.device()
    grid = device.compute_with_storage_grid_size()
    is_dram = tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
    plan = split_plan(
        tuple(tensor.shape),
        buffer_alignment=ttnn.get_dram_alignment() if is_dram else ttnn.get_l1_alignment(),
        l1_alignment=ttnn.get_l1_alignment(),
        grid_size=grid,
        l1_budget=ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES,
        **knobs,
    )
    return assign_cores(plan, grid)


def create_program_artifacts(
    t: ttnn.Tensor,
    *,
    skip_negative_entries: bool = False,
    worker_threads: int = WORKER_THREADS,
    min_worker_bytes: int = MIN_WORKER_BYTES,
    scratch_depth: int = SCRATCH_DEPTH,
    probe_tag_threads: bool = False,
):
    """(spec, run_args, tensor_indices) for `ttnn.generic_op([t], ...)`: `t` is read and written in place.

    `worker_threads`, `min_worker_bytes`, `scratch_depth` are the Blocking-Model knobs (default: the constants).
    `probe_tag_threads` is a test probe: thread `i` adds `1 + i` instead of 1, so a test sees which thread
    incremented which element.
    """
    plan = plan_for(t, worker_threads=worker_threads, min_worker_bytes=min_worker_bytes, scratch_depth=scratch_depth)
    assert plan.rows == t.buffer_num_pages(), (plan.rows, t.buffer_num_pages())

    scratchpad = E.ScratchpadSpec(unique_id=SCRATCH_NAME, size_per_node=plan.scratch_bytes)
    # The geometry every thread shares; the schema's names are this dict's keys (one list, not two).
    common_values = {
        "row_elems": plan.row_elems,
        "seg_elems": plan.seg_elems,
        "segs_per_row": plan.segs_per_row,
        "slot_stride_bytes": plan.slot_stride_bytes,
        "block_units": plan.block_units,
        "region_stride_bytes": plan.region_stride_bytes,
        "scratch_depth": scratch_depth,
    }
    worker = E.KernelSpec(
        unique_id=KERNEL_NAME,
        source=str(KERNEL_DIR / "plus_one_worker.cpp"),
        hw_config=E.create_reader_datamovement_config(),
        num_threads=worker_threads,
        scratchpad_bindings=[E.ScratchpadBinding(scratchpad_spec_name=SCRATCH_NAME, accessor_name=SCRATCH_NAME)],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name=TENSOR_PARAM, accessor_name=TENSOR_PARAM)],
        compile_time_args={
            "worker_threads": worker_threads,
            "skip_negative_entries": int(bool(skip_negative_entries)),
            "tag_threads": int(bool(probe_tag_threads)),
            "cache_line_bytes": CACHE_LINE_BYTES,
        },
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(
            runtime_arg_names=["unit_start", "unit_count"], common_runtime_arg_names=list(common_values)
        ),
    )
    spec = E.ProgramSpec(
        name="plus_one",
        kernels=[worker],
        scratchpads=[scratchpad],
        tensor_parameters=[E.TensorParameter(unique_id=TENSOR_PARAM, spec=t.spec)],
        work_units=[E.WorkUnitSpec(name="main", kernels=[KERNEL_NAME], target_nodes=plan.core_set)],
    )
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(
                kernel=KERNEL_NAME,
                runtime_arg_values={
                    "unit_start": {core: s for core, (s, _) in plan.assignment.items()},
                    "unit_count": {core: n for core, (_, n) in plan.assignment.items()},
                },
                common_runtime_arg_values=common_values,
            )
        ]
    )
    # io_tensors is the single element [t]: it is the input and the output.
    return spec, run_args, {TENSOR_PARAM: 0}
