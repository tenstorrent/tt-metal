# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_untilize program descriptor: plan() (the single source of every block knob) and the one-kernel program.

Regime R1 `gather_rows`: one data-movement kernel on BRISC/NOC_0 over one rectangle of cores. Each core owns a
balanced contiguous run of output rows, gathers their 64 B face-row pieces from DRAM into row-major L1, and writes
each row as one page. No compute kernel, no semaphores, one CB.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
UNTILIZE_ROWS_KERNEL = str(KERNEL_DIR / "g6_untilize_untilize_rows.cpp")

REGIME = "gather_rows"

# Per-core L1 the model leaves to the op (BB-05).
L1_BUDGET = 1347776

TILE_H = 32
TILE_W = 32
FACE_W = 16

CB_ROW_STAGING = 0


@dataclass(frozen=True)
class Plan:
    x0: int
    y0: int
    cx: int
    cy: int
    total_rows: int
    H: int
    Ht: int
    W: int
    Wt: int
    elem: int
    block_rows: int
    block_tile_cols: int
    staging_depth: int
    rotate_tile_order: bool

    @property
    def num_cores(self) -> int:
        return self.cx * self.cy

    @property
    def staged_row_bytes(self) -> int:
        return self.block_tile_cols * TILE_W * self.elem

    @property
    def staging_bytes(self) -> int:
        return self.staging_depth * self.block_rows * self.staged_row_bytes

    def rows_of_core(self, i: int) -> tuple[int, int]:
        q, rem = divmod(self.total_rows, self.num_cores)
        return i * q + min(i, rem), q + (1 if i < rem else 0)

    def core_range(self) -> ttnn.CoreRange:
        return ttnn.CoreRange(
            ttnn.CoreCoord(self.x0, self.y0), ttnn.CoreCoord(self.x0 + self.cx - 1, self.y0 + self.cy - 1)
        )


def _rectangle(num_cores_target: int, grid_x: int, grid_y: int) -> tuple[int, int]:
    """Smallest cx*cy >= target inside the grid; ties go to the wider rectangle (row-wise placement)."""
    best = None
    for cy in range(1, grid_y + 1):
        cx = min(grid_x, -(-num_cores_target // cy))
        if cx * cy < num_cores_target:
            continue
        key = (cx * cy, -cx)
        if best is None or key < best[0]:
            best = (key, (cx, cy))
    return best[1]


def plan(
    shape,
    elem: int,
    grid,
    *,
    max_cores: int | None = None,
    block_rows: int | None = None,
    block_tile_cols: int | None = None,
    staging_depth: int | None = None,
    rotate_tile_order: bool = True,
) -> Plan:
    """Pure host function of (shape, elem, grid). Keyword overrides exist to turn knobs for perf lamps; the entry
    point passes none, so the defaults are the design's: as many cores as rows (up to the grid), each core's whole
    run as one block clamped by L1, one staging slot unless a core owns more than one block."""
    *lead, H, W = (int(d) for d in shape)
    B = 1
    for d in lead:
        B *= d
    total_rows = B * H
    Ht = -(-H // TILE_H)
    Wt = -(-W // TILE_W)
    grid_x, grid_y = int(grid.x), int(grid.y)

    face_row_bytes = FACE_W * elem
    dram_align = ttnn.get_dram_alignment()
    if face_row_bytes % dram_align != 0:
        raise ValueError(f"g6_untilize: face row of {face_row_bytes} B breaks the {dram_align} B DRAM read alignment")

    target = max(1, min(total_rows, grid_x * grid_y))
    if max_cores is not None:
        target = max(1, min(target, max_cores))
    cx, cy = _rectangle(target, grid_x, grid_y)
    num_cores = cx * cy
    rows_max = max(1, -(-total_rows // num_cores))

    tile_col_bytes = TILE_W * elem
    cols = Wt if block_tile_cols is None else max(1, min(block_tile_cols, Wt))
    if block_rows is None:
        if rows_max * cols * tile_col_bytes <= L1_BUDGET:
            rows = rows_max
        else:
            depth = 2 if staging_depth is None else staging_depth
            rows = min(rows_max, L1_BUDGET // (depth * cols * tile_col_bytes))
            if rows == 0:
                rows = 1
                cols = max(1, min(cols, L1_BUDGET // (depth * tile_col_bytes)))
    else:
        rows = max(1, min(block_rows, rows_max))

    blocks_per_core_max = -(-rows_max // rows) * -(-Wt // cols)
    depth = staging_depth if staging_depth is not None else (1 if blocks_per_core_max == 1 else 2)
    depth = max(1, min(depth, blocks_per_core_max))

    p = Plan(
        x0=0,
        y0=0,
        cx=cx,
        cy=cy,
        total_rows=total_rows,
        H=H,
        Ht=Ht,
        W=W,
        Wt=Wt,
        elem=elem,
        block_rows=rows,
        block_tile_cols=cols,
        staging_depth=depth,
        rotate_tile_order=rotate_tile_order,
    )
    if p.staging_bytes > L1_BUDGET:
        raise ValueError(f"g6_untilize: staging needs {p.staging_bytes} B per core, over the {L1_BUDGET} B L1 bound")
    return p


def create_program_descriptor(
    p: Plan, input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor
) -> tuple[ttnn.ProgramDescriptor, ttnn.KernelDescriptor]:
    """Returns the descriptor and its one kernel descriptor (the object whose common RT args a cache hit patches)."""
    cores = ttnn.CoreRangeSet([p.core_range()])

    cb_row_staging = ttnn.CBDescriptor(
        total_size=p.staging_bytes,
        core_ranges=cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_ROW_STAGING,
                data_format=input_tensor.dtype,
                page_size=p.staged_row_bytes,
            )
        ],
    )

    ct_args = [
        p.x0,
        p.y0,
        p.cx,
        p.num_cores,
        p.total_rows,
        p.H,
        p.Ht,
        p.W,
        p.Wt,
        p.elem,
        p.block_rows,
        p.block_tile_cols,
        p.staging_depth,
        1 if p.rotate_tile_order else 0,
        input_tensor.buffer_aligned_page_size(),
        output_tensor.buffer_aligned_page_size(),
        CB_ROW_STAGING,
    ]
    ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    kernel = ttnn.KernelDescriptor(
        kernel_source=UNTILIZE_ROWS_KERNEL,
        core_ranges=cores,
        compile_time_args=ct_args,
        common_runtime_args=[input_tensor.buffer_address(), output_tensor.buffer_address()],
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0,
            noc=ttnn.NOC.NOC_0,
        ),
    )

    descriptor = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb_row_staging])
    return descriptor, kernel
