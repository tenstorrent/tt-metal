# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_indexed_fill program: the plan, its ProgramDescriptor and the descriptor cache.

`derive_plan` is the single source of every knob (work split, block extent, staging depth, row-chunk
size, ids window, regime); every CB size, compile-time argument and runtime argument is derived from
the plan it returns. Nothing here reads a value of batch_ids: the kernel reads the ids buffer on the
device in every run, and the program carries only shapes, the plan and buffer addresses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
DM_KERNEL = KERNEL_DIR / "g6_indexed_fill_dm.cpp"

# Per-core L1 the model leaves to the op (BB-05).
L1_BOUND_BYTES = 1347776

# Knob caps. Read at call time (never copied), so an override reaches the next plan and cache key.
PLAN_CAPS = {
    "max_work_cores": None,  # None: the full compute grid
    "staging_budget_bytes": 1 << 20,  # staging_depth * block_units * row_chunk_bytes
    "max_staging_depth": 2,  # staging_depth when a core has more than one block
    "min_row_chunk_bytes": 2048,  # granularity floor of the row (w) split
    "max_row_chunk_bytes": 65536,  # one NoC transfer per row chunk
    "ids_chunk_cap_entries": 16384,  # ids window; K above it selects ids_streamed
}

CB_BATCH_IDS = 0  # the ids window (aligned superset of the scanned entries)
CB_STAGED_PAGES = 1  # staged row chunks, staging_depth slot sets of block_units slots
CB_BLOCK_DEST = 2  # one destination page (or SKIP) per unit of the current block

# Runtime-arg slots patched on every call (buffer addresses); the rest are the core's unit range.
RT_IDS_ADDR, RT_X_ADDR, RT_S_ADDR, RT_Y_ADDR = 0, 1, 2, 3


def _align_up(value, alignment):
    return -(-value // alignment) * alignment


def _align_down(value, alignment):
    return value - value % alignment


def _ceil_div(a, b):
    return -(-a // b)


@dataclass(frozen=True)
class Role:
    core: tuple  # (x, y) logical coordinate
    unit_begin: int
    unit_count: int
    x_units: int
    value_units: int


@dataclass(frozen=True)
class Plan:
    B: int
    R: int
    K: int
    P: int
    Q: int
    page_bytes: int
    dram_alignment: int
    row_chunk_bytes: int
    row_chunks_per_page: int
    units: int
    num_cores: int
    all_cores: object  # ttnn.CoreRangeSet
    block_units: int
    staging_depth: int
    max_blocks_per_core: int
    ids_chunk_entries: int
    ids_chunks: int
    regime: str
    ids_window_bytes: int
    dest_bytes: int
    l1_bytes_per_core: int
    roles: tuple

    def describe(self):
        return {
            "regime": self.regime,
            "num_cores": self.num_cores,
            "core_ranges": len(self.all_cores.ranges()),
            "units": self.units,
            "block_units": self.block_units,
            "staging_depth": self.staging_depth,
            "max_blocks_per_core": self.max_blocks_per_core,
            "row_chunk_bytes": self.row_chunk_bytes,
            "row_chunks_per_page": self.row_chunks_per_page,
            "ids_chunk_entries": self.ids_chunk_entries,
            "ids_chunks": self.ids_chunks,
            "l1_bytes_per_core": self.l1_bytes_per_core,
            "roles": [
                {
                    "core": role.core,
                    "unit_begin": role.unit_begin,
                    "unit_count": role.unit_count,
                    "x_units": role.x_units,
                    "value_units": role.value_units,
                }
                for role in self.roles
            ],
        }


def derive_plan(ids_shape, input_shape, value_shape, grid):
    """The plan for one call: a pure function of the three shapes, the compute grid and PLAN_CAPS."""
    caps = PLAN_CAPS
    A = ttnn.get_dram_alignment()
    B, d1, d2, W = (int(v) for v in input_shape)
    K = int(ids_shape[0])
    R = d1 * d2
    page_bytes = 4 * W
    P, Q = B * R, K * R
    pages = P + Q

    grid_cores = grid.x * grid.y
    cores_cap = grid_cores if caps["max_work_cores"] is None else min(grid_cores, caps["max_work_cores"])

    # Split rows only when whole pages leave cores idle, and never below the granularity floor.
    n_split = max(1, min(cores_cap // pages, page_bytes // caps["min_row_chunk_bytes"]))
    max_chunk = max(A, _align_down(caps["max_row_chunk_bytes"], A))
    row_chunk_bytes = min(_align_up(_ceil_div(page_bytes, n_split), A), max_chunk)
    row_chunks_per_page = _ceil_div(page_bytes, row_chunk_bytes)
    units = pages * row_chunks_per_page
    if units >= 2**32:
        raise ValueError(f"g6_indexed_fill: {units} work units overflow the kernel's 32-bit unit index")

    core_set = ttnn.num_cores_to_corerangeset(cores_cap, grid, True)
    num_cores, all_cores, group_1, group_2, n1, n2 = ttnn.split_work_to_cores(core_set, units, True)

    # The coarsest block that fits the staging budget: the whole core assignment when it does.
    if n1 * row_chunk_bytes <= caps["staging_budget_bytes"]:
        staging_depth, block_units = 1, n1
    else:
        staging_depth = caps["max_staging_depth"]
        block_units = max(1, caps["staging_budget_bytes"] // (staging_depth * row_chunk_bytes))
    max_blocks_per_core = _ceil_div(n1, block_units)

    ids_chunk_entries = min(K, caps["ids_chunk_cap_entries"])
    ids_chunks = _ceil_div(K, ids_chunk_entries)
    regime = "ids_resident" if ids_chunks == 1 else "ids_streamed"
    # A window of entries [j0, j1) is read as [align_down(4*j0), align_up(4*j1)); when the window
    # stride is not a multiple of A that superset can span one more granule.
    ids_window_bytes = _align_up(4 * ids_chunk_entries, A) + (
        A if ids_chunks > 1 and (4 * ids_chunk_entries) % A else 0
    )
    dest_bytes = _align_up(4 * block_units, A)
    l1_bytes_per_core = ids_window_bytes + staging_depth * block_units * row_chunk_bytes + dest_bytes
    if l1_bytes_per_core > L1_BOUND_BYTES:
        raise ValueError(f"g6_indexed_fill: plan needs {l1_bytes_per_core} B of L1 per core (bound {L1_BOUND_BYTES})")

    x_units_end = P * row_chunks_per_page
    roles = []
    unit_begin = 0
    for group, per_core in ((group_1, n1), (group_2, n2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            unit_end = unit_begin + per_core
            x_units = max(0, min(unit_end, x_units_end) - unit_begin)
            roles.append(Role((core.x, core.y), unit_begin, per_core, x_units, per_core - x_units))
            unit_begin = unit_end
    assert unit_begin == units and len(roles) == num_cores

    return Plan(
        B=B,
        R=R,
        K=K,
        P=P,
        Q=Q,
        page_bytes=page_bytes,
        dram_alignment=A,
        row_chunk_bytes=row_chunk_bytes,
        row_chunks_per_page=row_chunks_per_page,
        units=units,
        num_cores=num_cores,
        all_cores=all_cores,
        block_units=block_units,
        staging_depth=staging_depth,
        max_blocks_per_core=max_blocks_per_core,
        ids_chunk_entries=ids_chunk_entries,
        ids_chunks=ids_chunks,
        regime=regime,
        ids_window_bytes=ids_window_bytes,
        dest_bytes=dest_bytes,
        l1_bytes_per_core=l1_bytes_per_core,
        roles=tuple(roles),
    )


def _cb(index, page_size, total_size, core_ranges):
    return ttnn.CBDescriptor(
        total_size=total_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.uint32, page_size=page_size)],
    )


def _build_program(plan, batch_ids, input_tensor, value_tensor, output_tensor):
    """(kernel descriptor, CB descriptors) of the plan, runtime addresses those of the given tensors."""
    cores = plan.all_cores
    cbs = [
        _cb(CB_BATCH_IDS, plan.ids_window_bytes, plan.ids_window_bytes, cores),
        _cb(
            CB_STAGED_PAGES,
            plan.row_chunk_bytes,
            plan.staging_depth * plan.block_units * plan.row_chunk_bytes,
            cores,
        ),
        _cb(CB_BLOCK_DEST, plan.dest_bytes, plan.dest_bytes, cores),
    ]

    ct_args = [
        CB_BATCH_IDS,
        CB_STAGED_PAGES,
        CB_BLOCK_DEST,
        plan.B,
        plan.R,
        plan.K,
        plan.P,
        plan.page_bytes,
        plan.row_chunk_bytes,
        plan.row_chunks_per_page,
        plan.block_units,
        plan.staging_depth,
        plan.ids_chunk_entries,
        plan.ids_chunks,
        1 if plan.regime == "ids_resident" else 0,
        plan.dram_alignment,
    ]
    for tensor in (batch_ids, input_tensor, value_tensor, output_tensor):
        ct_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())

    addresses = _addresses(batch_ids, input_tensor, value_tensor, output_tensor)
    rt_args = ttnn.RuntimeArgs()
    for role in plan.roles:
        x, y = role.core
        rt_args[x][y] = addresses + [role.unit_begin, role.unit_count]

    kernel = ttnn.KernelDescriptor(
        kernel_source=str(DM_KERNEL),
        core_ranges=cores,
        compile_time_args=ct_args,
        runtime_args=rt_args,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
    )
    return kernel, cbs


def _addresses(batch_ids, input_tensor, value_tensor, output_tensor):
    return [
        batch_ids.buffer_address(),
        input_tensor.buffer_address(),
        value_tensor.buffer_address(),
        output_tensor.buffer_address(),
    ]


def _cache_key(device, batch_ids, input_tensor, value_tensor):
    tensors = (batch_ids, input_tensor, value_tensor)
    grid = device.compute_with_storage_grid_size()
    return (
        device.id(),
        (grid.x, grid.y),
        tuple(tuple(int(v) for v in t.shape) for t in tensors),
        tuple(t.dtype for t in tensors),
        tuple(t.layout for t in tensors),
        tuple(str(t.memory_config()) for t in tensors),
        tuple(sorted(PLAN_CAPS.items())),
    )


class _Program:
    """A built program of one cache key: the plan, its kernel and CBs, and the cores to patch."""

    def __init__(self, plan, kernel, cbs):
        self.plan = plan
        self.kernel = kernel
        self.cbs = cbs

    def descriptor_for(self, batch_ids, input_tensor, value_tensor, output_tensor):
        addresses = _addresses(batch_ids, input_tensor, value_tensor, output_tensor)
        runtime_args = self.kernel.runtime_args
        for role in self.plan.roles:
            x, y = role.core
            args = runtime_args[x][y]
            args[RT_IDS_ADDR] = addresses[0]
            args[RT_X_ADDR] = addresses[1]
            args[RT_S_ADDR] = addresses[2]
            args[RT_Y_ADDR] = addresses[3]
        return ttnn.ProgramDescriptor(kernels=[self.kernel], semaphores=[], cbs=self.cbs)


_PROGRAM_CACHE = {}


def plan_for(batch_ids, input_tensor, value_tensor):
    device = input_tensor.device()
    return derive_plan(
        tuple(batch_ids.shape),
        tuple(input_tensor.shape),
        tuple(value_tensor.shape),
        device.compute_with_storage_grid_size(),
    )


def program_descriptor(batch_ids, input_tensor, value_tensor, output_tensor):
    """The ProgramDescriptor of this call. A repeat key reuses the built program and patches only the
    four buffer addresses in every core's runtime args."""
    key = _cache_key(input_tensor.device(), batch_ids, input_tensor, value_tensor)
    program = _PROGRAM_CACHE.get(key)
    if program is None:
        plan = plan_for(batch_ids, input_tensor, value_tensor)
        kernel, cbs = _build_program(plan, batch_ids, input_tensor, value_tensor, output_tensor)
        program = _Program(plan, kernel, cbs)
        _PROGRAM_CACHE[key] = program
    return program.descriptor_for(batch_ids, input_tensor, value_tensor, output_tensor)
