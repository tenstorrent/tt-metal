# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_untilize program descriptor: the row_gather plan and its one-kernel ProgramDescriptor.

Blocking knobs (op_design.md, Blocking Model) are the module constants below; every CB size,
loop extent and core count is derived from them in `make_plan`, once per cached plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
DM_KERNEL = KERNEL_DIR / "g6_untilize_dm.cpp"

# ---- Tunable knobs ---------------------------------------------------------------------------
# Tile columns per block; None = the whole width (one block row = one whole output page).
BLOCK_TILE_COLS_REQ = None
# Cores the split may use; None = the whole worker grid.
CORE_BUDGET = None
# Slot-ring depth cap; the plan uses min(SCRATCH_DEPTH, blocks per core).
SCRATCH_DEPTH = 2
# Logical origin of the core rectangle.
CORE_ORIGIN = (0, 0)
# RISC and NoC of the single data-movement kernel.
DM_PROCESSOR = ttnn.DataMovementProcessor.RISCV_0
DM_NOC = ttnn.NOC.NOC_0

# L1 the model leaves to the op on every core (BB-05).
L1_BUDGET_BYTES = 1347776

TILE_DIM = 32
CB_ROW_STAGING = 0


def knob_values():
    """Everything a knob change alters in the program: part of the program-cache key."""
    return (BLOCK_TILE_COLS_REQ, CORE_BUDGET, SCRATCH_DEPTH, CORE_ORIGIN, DM_PROCESSOR, DM_NOC, L1_BUDGET_BYTES)


@dataclass(frozen=True)
class Plan:
    regime: str
    # geometry
    tensor_rows: int
    rows_per_image: int
    tile_rows_per_image: int
    tensor_tile_cols: int
    page_bytes: int
    element_bytes: int
    # blocking
    block_rows: int
    block_tile_cols: int
    block_row_bytes: int
    col_blocks: int
    row_blocks: int
    total_blocks: int
    scratch_slots: int
    # work split
    num_cores: int
    cores_x: int
    core_origin: tuple
    core_ranges: tuple  # ((x0, y0, x1, y1), ...) inclusive

    @property
    def staging_bytes(self) -> int:
        return self.block_rows * self.block_row_bytes * self.scratch_slots

    @property
    def blocks_per_core(self) -> tuple:
        q, rem = divmod(self.total_blocks, self.num_cores)
        return (q, q + (1 if rem else 0))


def _div_up(a: int, b: int) -> int:
    return -(-a // b)


def _core_rectangle(num_cores: int, avail_x: int, avail_y: int, origin: tuple) -> tuple:
    """One rectangle when a divisor of num_cores fits the grid, else full rows plus one partial row.
    Rows are filled x-first, matching the kernel's index (y - y0) * cores_x + (x - x0)."""
    x0, y0 = origin
    for cores_x in range(min(num_cores, avail_x), 0, -1):
        if num_cores % cores_x == 0 and num_cores // cores_x <= avail_y:
            cores_y = num_cores // cores_x
            return cores_x, ((x0, y0, x0 + cores_x - 1, y0 + cores_y - 1),)
    cores_x = avail_x
    full_rows, partial = divmod(num_cores, cores_x)
    ranges = [(x0, y0, x0 + cores_x - 1, y0 + full_rows - 1)]
    if partial:
        ranges.append((x0, y0 + full_rows, x0 + partial - 1, y0 + full_rows))
    return cores_x, tuple(ranges)


def make_plan(shape, element_bytes: int, grid_x: int, grid_y: int) -> Plan:
    d0, d1, H, W = (int(v) for v in shape)
    images = d0 * d1
    tile_rows_per_image = _div_up(H, TILE_DIM)
    tensor_tile_cols = _div_up(W, TILE_DIM)
    tensor_rows = images * H
    page_bytes = W * element_bytes
    tile_row_bytes = TILE_DIM * element_bytes

    origin_x, origin_y = CORE_ORIGIN
    avail_x, avail_y = grid_x - origin_x, grid_y - origin_y
    if avail_x <= 0 or avail_y <= 0:
        raise ValueError(f"g6_untilize: core origin {CORE_ORIGIN} lies outside the {grid_x}x{grid_y} grid")
    core_budget = avail_x * avail_y if CORE_BUDGET is None else min(CORE_BUDGET, avail_x * avail_y)

    # Knob order (op_design.md, Work Distribution): tile columns, then rows by occupancy.
    # The fit caps assume the deepest ring, so the staging fits BB-05 at any scratch_slots.
    tile_cols_fit_cap = max(1, L1_BUDGET_BYTES // (tile_row_bytes * SCRATCH_DEPTH))
    requested_cols = tensor_tile_cols if BLOCK_TILE_COLS_REQ is None else BLOCK_TILE_COLS_REQ
    block_tile_cols = max(1, min(requested_cols, tensor_tile_cols, tile_cols_fit_cap))
    block_row_bytes = block_tile_cols * tile_row_bytes
    col_blocks = _div_up(tensor_tile_cols, block_tile_cols)

    rows_fit_cap = max(1, L1_BUDGET_BYTES // (block_row_bytes * SCRATCH_DEPTH))
    block_rows = min(max(_div_up(tensor_rows * col_blocks, core_budget), 1), rows_fit_cap)
    row_blocks = _div_up(tensor_rows, block_rows)
    total_blocks = row_blocks * col_blocks

    num_cores = min(total_blocks, core_budget)
    scratch_slots = min(SCRATCH_DEPTH, _div_up(total_blocks, num_cores))
    cores_x, core_ranges = _core_rectangle(num_cores, avail_x, avail_y, CORE_ORIGIN)

    plan = Plan(
        regime="row_gather",
        tensor_rows=tensor_rows,
        rows_per_image=H,
        tile_rows_per_image=tile_rows_per_image,
        tensor_tile_cols=tensor_tile_cols,
        page_bytes=page_bytes,
        element_bytes=element_bytes,
        block_rows=block_rows,
        block_tile_cols=block_tile_cols,
        block_row_bytes=block_row_bytes,
        col_blocks=col_blocks,
        row_blocks=row_blocks,
        total_blocks=total_blocks,
        scratch_slots=scratch_slots,
        num_cores=num_cores,
        cores_x=cores_x,
        core_origin=CORE_ORIGIN,
        core_ranges=core_ranges,
    )
    assert plan.staging_bytes <= L1_BUDGET_BYTES, plan
    return plan


def core_range_set(plan: Plan) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for x0, y0, x1, y1 in plan.core_ranges]
    )


def create_program_parts(input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor, plan: Plan):
    """The kernel and CB descriptors of the plan. Buffer addresses are common runtime args, set per call."""
    cores = core_range_set(plan)

    cb_row_staging = ttnn.CBDescriptor(
        total_size=plan.staging_bytes,
        core_ranges=cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_ROW_STAGING,
                data_format=input_tensor.dtype,
                page_size=plan.block_row_bytes,
            )
        ],
    )

    ct_args = [
        plan.tensor_rows,
        plan.rows_per_image,
        plan.tile_rows_per_image,
        plan.tensor_tile_cols,
        plan.page_bytes,
        plan.element_bytes,
        plan.block_rows,
        plan.block_tile_cols,
        plan.block_row_bytes,
        plan.col_blocks,
        plan.total_blocks,
        plan.num_cores,
        plan.cores_x,
        plan.core_origin[0],
        plan.core_origin[1],
        plan.scratch_slots,
        CB_ROW_STAGING,
    ]
    ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    untilize_dm = ttnn.KernelDescriptor(
        kernel_source=str(DM_KERNEL),
        core_ranges=cores,
        compile_time_args=ct_args,
        config=ttnn.DataMovementConfigDescriptor(processor=DM_PROCESSOR, noc=DM_NOC),
    )
    untilize_dm.common_runtime_args = [input_tensor.buffer_address(), output_tensor.buffer_address()]
    return untilize_dm, cb_row_staging
