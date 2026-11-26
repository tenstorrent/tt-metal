# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_untilize program descriptor: the row_gather plan and its one data-movement kernel.

Every block knob is a module constant read once by make_plan(); the CB size, the kernel's loop
extents and the core range all derive from the Plan, never from a whole-tensor dimension.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
UNTILIZE_DM_KERNEL = KERNEL_DIR / "g6_untilize_dm.cpp"

# ---------------------------------------------------------------------------
# Knobs (op_design.md, Blocking Model). Each is defined once here.
# ---------------------------------------------------------------------------

# Tile columns per block; None = the whole width, so one block row is one whole output page.
BLOCK_TILE_COLS_REQ = None
# Depth of the staging-slot ring; the plan clamps it to the blocks a core actually runs.
SCRATCH_DEPTH = 2
# Cores the plan may use; None = the whole compute-with-storage grid.
CORE_BUDGET_REQ = None
# Logical origin of the core rectangle.
CORE_ORIGIN = (0, 0)
# RISC and NoC that run untilize_dm.
KERNEL_PROCESSOR = ttnn.DataMovementProcessor.RISCV_0
KERNEL_NOC = ttnn.NOC.NOC_0
# Per-core L1 the model leaves to the op (BB-05).
L1_BUDGET_BYTES = 1347776

# ---------------------------------------------------------------------------
# Storage geometry of a 32x32 tile with four 16x16 faces.
# ---------------------------------------------------------------------------

TILE_HEIGHT = 32
TILE_WIDTH = 32
FACE_HEIGHT = 16
FACE_WIDTH = 16

CB_ROW_STAGING = 0

REGIME = "row_gather"


@dataclass(frozen=True)
class Plan:
    regime: str
    # geometry
    rows_per_image: int
    tile_rows_per_image: int
    tensor_tile_cols: int
    tensor_rows: int
    element_bytes: int
    out_row_bytes: int
    # knobs
    block_rows: int
    block_tile_cols: int
    scratch_slots: int
    # work split
    col_blocks: int
    row_blocks: int
    total_blocks: int
    num_cores: int
    cores_x: int
    cores_y: int
    origin_x: int
    origin_y: int
    min_blocks_per_core: int
    max_blocks_per_core: int

    @property
    def tile_row_bytes(self) -> int:
        return TILE_WIDTH * self.element_bytes

    @property
    def block_row_bytes(self) -> int:
        return self.block_tile_cols * self.tile_row_bytes

    @property
    def slot_bytes(self) -> int:
        return self.block_rows * self.block_row_bytes

    @property
    def staging_bytes(self) -> int:
        return self.slot_bytes * self.scratch_slots

    def core_ranges(self) -> ttnn.CoreRangeSet:
        """One rectangle; when no divisor of num_cores fits the grid, full rows plus one partial row."""
        x0, y0 = self.origin_x, self.origin_y
        full_rows = self.num_cores // self.cores_x
        partial = self.num_cores - full_rows * self.cores_x
        ranges = []
        if full_rows:
            ranges.append(
                ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x0 + self.cores_x - 1, y0 + full_rows - 1))
            )
        if partial:
            ranges.append(
                ttnn.CoreRange(ttnn.CoreCoord(x0, y0 + full_rows), ttnn.CoreCoord(x0 + partial - 1, y0 + full_rows))
            )
        return ttnn.CoreRangeSet(ranges)


def _div_up(a: int, b: int) -> int:
    return -(-a // b)


def _core_rectangle(num_cores: int, grid_x: int, grid_y: int) -> tuple[int, int]:
    for d in range(min(num_cores, grid_x), 0, -1):
        if num_cores % d == 0 and num_cores // d <= grid_y:
            return d, num_cores // d
    return grid_x, _div_up(num_cores, grid_x)


def make_plan(shape, element_bytes: int, grid_x: int, grid_y: int) -> Plan:
    d0, d1, h, w = (int(v) for v in shape)
    images = d0 * d1
    tile_rows_per_image = _div_up(h, TILE_HEIGHT)
    tensor_tile_cols = _div_up(w, TILE_WIDTH)
    tensor_rows = images * h
    tile_row_bytes = TILE_WIDTH * element_bytes

    origin_x, origin_y = CORE_ORIGIN
    avail_x, avail_y = grid_x - origin_x, grid_y - origin_y
    core_budget = avail_x * avail_y if CORE_BUDGET_REQ is None else min(CORE_BUDGET_REQ, avail_x * avail_y)

    # 1) tile columns per block, capped so SCRATCH_DEPTH one-row slots fit the L1 bound
    tile_cols_fit_cap = max(1, L1_BUDGET_BYTES // (tile_row_bytes * SCRATCH_DEPTH))
    block_tile_cols = min(BLOCK_TILE_COLS_REQ or tensor_tile_cols, tensor_tile_cols, tile_cols_fit_cap)
    col_blocks = _div_up(tensor_tile_cols, block_tile_cols)
    block_row_bytes = block_tile_cols * tile_row_bytes

    # 2) occupancy: rows per block so that the blocks spread over the core budget
    rows_fit_cap = max(1, L1_BUDGET_BYTES // (block_row_bytes * SCRATCH_DEPTH))
    block_rows = min(max(_div_up(tensor_rows * col_blocks, core_budget), 1), rows_fit_cap)
    row_blocks = _div_up(tensor_rows, block_rows)
    total_blocks = row_blocks * col_blocks

    # 3) cores, 4) slot-ring depth
    num_cores = min(total_blocks, core_budget)
    min_blocks = total_blocks // num_cores
    max_blocks = _div_up(total_blocks, num_cores)
    scratch_slots = min(SCRATCH_DEPTH, max_blocks)
    cores_x, cores_y = _core_rectangle(num_cores, avail_x, avail_y)

    plan = Plan(
        regime=REGIME,
        rows_per_image=h,
        tile_rows_per_image=tile_rows_per_image,
        tensor_tile_cols=tensor_tile_cols,
        tensor_rows=tensor_rows,
        element_bytes=element_bytes,
        out_row_bytes=w * element_bytes,
        block_rows=block_rows,
        block_tile_cols=block_tile_cols,
        scratch_slots=scratch_slots,
        col_blocks=col_blocks,
        row_blocks=row_blocks,
        total_blocks=total_blocks,
        num_cores=num_cores,
        cores_x=cores_x,
        cores_y=cores_y,
        origin_x=origin_x,
        origin_y=origin_y,
        min_blocks_per_core=min_blocks,
        max_blocks_per_core=max_blocks,
    )
    assert plan.staging_bytes <= L1_BUDGET_BYTES, f"g6_untilize: staging {plan.staging_bytes} B exceeds the L1 bound"
    return plan


def describe(plan: Plan) -> dict:
    """Plan report (BB-08 / BB-21): every core in the range has the row_gather role."""
    core_ranges = plan.core_ranges()
    return {
        "regime": plan.regime,
        "core_ranges": [(r.start.x, r.start.y, r.end.x, r.end.y) for r in core_ranges.ranges()],
        "num_cores": plan.num_cores,
        "block_rows": plan.block_rows,
        "block_tile_cols": plan.block_tile_cols,
        "scratch_slots": plan.scratch_slots,
        "blocks_per_core": {"min": plan.min_blocks_per_core, "max": plan.max_blocks_per_core},
        "roles": {"row_gather": [(r.start.x, r.start.y, r.end.x, r.end.y) for r in core_ranges.ranges()]},
        "staging_bytes_per_core": plan.staging_bytes,
        "plan": asdict(plan),
    }


class CachedProgram:
    """A built descriptor; a call only rewrites the two address words (BB-07)."""

    def __init__(self, descriptor: ttnn.ProgramDescriptor, kernel: ttnn.KernelDescriptor, plan: Plan):
        self.descriptor = descriptor
        self.kernel = kernel
        self.plan = plan

    def bind(self, input_address: int, output_address: int) -> ttnn.ProgramDescriptor:
        self.kernel.common_runtime_args = [input_address, output_address]
        self.descriptor.kernels = [self.kernel]
        return self.descriptor


def create_program(input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor, plan: Plan) -> CachedProgram:
    core_ranges = plan.core_ranges()

    cb_row_staging = ttnn.CBDescriptor(
        total_size=plan.staging_bytes,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_ROW_STAGING,
                data_format=input_tensor.dtype,
                page_size=plan.block_row_bytes,
            )
        ],
    )

    ct_args = [
        input_tensor.buffer_aligned_page_size(),
        output_tensor.buffer_aligned_page_size(),
        plan.out_row_bytes,
        plan.rows_per_image,
        plan.tile_rows_per_image,
        plan.tensor_tile_cols,
        plan.tensor_rows,
        plan.block_rows,
        plan.block_tile_cols,
        plan.col_blocks,
        plan.total_blocks,
        plan.num_cores,
        plan.cores_x,
        plan.origin_x,
        plan.origin_y,
        plan.scratch_slots,
        TILE_HEIGHT,
        FACE_HEIGHT,
        FACE_WIDTH * plan.element_bytes,
        FACE_HEIGHT * FACE_WIDTH * plan.element_bytes,
        plan.tile_row_bytes,
        CB_ROW_STAGING,
    ]
    ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    kernel = ttnn.KernelDescriptor(
        kernel_source=str(UNTILIZE_DM_KERNEL),
        core_ranges=core_ranges,
        compile_time_args=ct_args,
        runtime_args=ttnn.RuntimeArgs(),
        common_runtime_args=[input_tensor.buffer_address(), output_tensor.buffer_address()],
        config=ttnn.DataMovementConfigDescriptor(processor=KERNEL_PROCESSOR, noc=KERNEL_NOC),
    )

    descriptor = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb_row_staging])
    # Hashes every input that shapes the program (source, CT args, cores, CB, RISC/NoC), so the
    # per-call descriptor walk can be skipped; addresses are runtime args and stay out of it.
    program_identity = (
        "g6_untilize",
        str(UNTILIZE_DM_KERNEL),
        tuple(ct_args),
        describe(plan)["core_ranges"],
        plan.staging_bytes,
        plan.block_row_bytes,
        str(input_tensor.dtype),
        str(KERNEL_PROCESSOR),
        str(KERNEL_NOC),
    )
    digest = hashlib.blake2b(repr(program_identity).encode(), digest_size=8).digest()
    descriptor.custom_program_hash = int.from_bytes(digest, "little")
    return CachedProgram(descriptor, kernel, plan)
