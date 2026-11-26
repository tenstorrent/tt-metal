# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_reshard program descriptor — PULL regime.

Every core of out's trimmed rectangle runs `ISSUERS_PER_CORE` instances of one gather kernel
(issuer 0 on NCRISC/NOC0, issuer 1 on BRISC/NOC1). Each issuer reads its row range of the core's
out block from the in0 cores holding it, straight into the out shard. No CB, no compute kernel,
no semaphore, no per-core runtime args: per-core identity comes from the core's logical coords.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
GATHER_KERNEL = str(KERNEL_DIR / "g6_reshard_gather.cpp")

TILE = 32

# Block knobs (op_design.md, Blocking Model). ISSUERS_PER_CORE is 1 or 2 (one kernel per
# data-movement RISC). BLOCK_ROWS_TILES / BLOCK_COLS_TILES of None mean the coarsest block:
# the issuer's whole row share and the out shard's whole width.
ISSUERS_PER_CORE = 2
BLOCK_ROWS_TILES = None
BLOCK_COLS_TILES = None

# (processor, noc) of issuer k.
_ISSUER_CONFIGS = (
    (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
    (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
)


def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass(frozen=True)
class Placement:
    """A BLOCK_SHARDED placement in tile units: origin core, shard shape, orientation."""

    x0: int
    y0: int
    x1: int
    y1: int
    shard_rows_tiles: int
    shard_cols_tiles: int
    col_major: bool

    @staticmethod
    def from_memory_config(memory_config: ttnn.MemoryConfig) -> "Placement":
        spec = memory_config.shard_spec
        box = spec.grid.bounding_box()
        return Placement(
            x0=box.start.x,
            y0=box.start.y,
            x1=box.end.x,
            y1=box.end.y,
            shard_rows_tiles=int(spec.shape[0]) // TILE,
            shard_cols_tiles=int(spec.shape[1]) // TILE,
            col_major=spec.orientation == ttnn.ShardOrientation.COL_MAJOR,
        )

    def num_block_rows(self, tensor_rows_tiles: int) -> int:
        return _div_up(tensor_rows_tiles, self.shard_rows_tiles)

    def num_block_cols(self, tensor_cols_tiles: int) -> int:
        return _div_up(tensor_cols_tiles, self.shard_cols_tiles)

    def block_core(self, r: int, c: int) -> tuple[int, int]:
        """Logical core holding block (r, c)."""
        return (self.x0 + r, self.y0 + c) if self.col_major else (self.x0 + c, self.y0 + r)

    def trimmed_extent(self, num_block_rows: int, num_block_cols: int) -> tuple[int, int]:
        """(x extent, y extent) in cores of the rectangle that holds blocks."""
        return (num_block_rows, num_block_cols) if self.col_major else (num_block_cols, num_block_rows)


def tensor_tiles(padded_shape) -> tuple[int, int]:
    """(tile rows, tile cols) of the padded 2-D view: all dims but the last folded into H."""
    rows = 1
    for d in list(padded_shape)[:-1]:
        rows *= int(d)
    return rows // TILE, int(padded_shape[-1]) // TILE


@dataclass(frozen=True)
class Plan:
    tensor_rows_tiles: int
    tensor_cols_tiles: int
    page_bytes: int
    p_in: Placement
    p_out: Placement
    issuers_per_core: int
    issuer_rows_tiles: int
    block_rows_tiles: int
    block_cols_tiles: int

    @property
    def in_num_block_rows(self) -> int:
        return self.p_in.num_block_rows(self.tensor_rows_tiles)

    @property
    def in_num_block_cols(self) -> int:
        return self.p_in.num_block_cols(self.tensor_cols_tiles)

    @property
    def out_num_block_rows(self) -> int:
        return self.p_out.num_block_rows(self.tensor_rows_tiles)

    @property
    def out_num_block_cols(self) -> int:
        return self.p_out.num_block_cols(self.tensor_cols_tiles)

    def out_core_range(self) -> ttnn.CoreRange:
        ext_x, ext_y = self.p_out.trimmed_extent(self.out_num_block_rows, self.out_num_block_cols)
        return ttnn.CoreRange(
            ttnn.CoreCoord(self.p_out.x0, self.p_out.y0),
            ttnn.CoreCoord(self.p_out.x0 + ext_x - 1, self.p_out.y0 + ext_y - 1),
        )


def make_plan(padded_shape, page_bytes: int, p_in: Placement, p_out: Placement) -> Plan:
    assert 1 <= ISSUERS_PER_CORE <= len(_ISSUER_CONFIGS), "one issuer kernel per data-movement RISC"
    rows, cols = tensor_tiles(padded_shape)
    issuer_rows_tiles = _div_up(p_out.shard_rows_tiles, ISSUERS_PER_CORE)
    return Plan(
        tensor_rows_tiles=rows,
        tensor_cols_tiles=cols,
        page_bytes=page_bytes,
        p_in=p_in,
        p_out=p_out,
        issuers_per_core=ISSUERS_PER_CORE,
        issuer_rows_tiles=issuer_rows_tiles,
        block_rows_tiles=BLOCK_ROWS_TILES or issuer_rows_tiles,
        block_cols_tiles=BLOCK_COLS_TILES or p_out.shard_cols_tiles,
    )


def _in_noc_tables(device, plan: Plan) -> tuple[list[int], list[int]]:
    """Virtual NoC coordinate of every in0 block row and block column.

    A block row maps to one core y (ROW_MAJOR) or x (COL_MAJOR), a block column to the other, so
    the kernel needs one table per axis instead of one entry per core. That only holds if the
    logical→virtual map is separable; it is asserted for every core holding an in0 block.
    """
    p = plan.p_in
    nbr, nbc = plan.in_num_block_rows, plan.in_num_block_cols

    def virt(core):
        v = device.worker_core_from_logical_core(ttnn.CoreCoord(*core))
        return v.x, v.y

    # Block row r varies y for ROW_MAJOR (x for COL_MAJOR); block column c the other coordinate.
    row_axis = 0 if p.col_major else 1
    col_axis = 1 - row_axis
    row_noc = [virt(p.block_core(r, 0))[row_axis] for r in range(nbr)]
    col_noc = [virt(p.block_core(0, c))[col_axis] for c in range(nbc)]

    for r in range(nbr):
        for c in range(nbc):
            v = virt(p.block_core(r, c))
            want = [0, 0]
            want[row_axis], want[col_axis] = row_noc[r], col_noc[c]
            if v != tuple(want):
                raise RuntimeError(
                    f"g6_reshard: logical->virtual NoC map is not separable at block ({r}, {c}) "
                    f"core {p.block_core(r, c)}: {v} != {tuple(want)}"
                )
    return row_noc, col_noc


def create_program_descriptor(device, plan: Plan, in_addr: int, out_addr: int) -> ttnn.ProgramDescriptor:
    row_noc, col_noc = _in_noc_tables(device, plan)
    common_rt_args = [in_addr, out_addr, *row_noc, *col_noc]

    p_in, p_out = plan.p_in, plan.p_out
    shared_ct_args = [
        plan.issuers_per_core,
        plan.block_rows_tiles,
        plan.block_cols_tiles,
        plan.page_bytes,
        plan.tensor_rows_tiles,
        plan.tensor_cols_tiles,
        p_in.shard_rows_tiles,
        p_in.shard_cols_tiles,
        int(p_in.col_major),
        plan.in_num_block_rows,
        plan.in_num_block_cols,
        p_out.shard_rows_tiles,
        p_out.shard_cols_tiles,
        int(p_out.col_major),
        p_out.x0,
        p_out.y0,
    ]

    core_ranges = ttnn.CoreRangeSet([plan.out_core_range()])
    kernels = []
    for issuer in range(plan.issuers_per_core):
        processor, noc = _ISSUER_CONFIGS[issuer]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=GATHER_KERNEL,
                core_ranges=core_ranges,
                compile_time_args=[issuer, *shared_ct_args],
                common_runtime_args=list(common_rt_args),
                config=ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc),
            )
        )
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[])


def patch_addresses(descriptor: ttnn.ProgramDescriptor, in_addr: int, out_addr: int) -> None:
    """Rewrite the only per-call values (common RT args 0, 1) of a cached descriptor."""
    kernels = descriptor.kernels
    for kernel in kernels:
        kernel.common_runtime_args[0] = in_addr
        kernel.common_runtime_args[1] = out_addr
    descriptor.kernels = kernels


def describe_plan(plan: Plan) -> dict:
    """Plan / role dump (BB-08, BB-21). Pure host arithmetic."""
    out_range = plan.out_core_range()
    targets = {}
    for r in range(plan.out_num_block_rows):
        core_rows = min(plan.p_out.shard_rows_tiles, plan.tensor_rows_tiles - r * plan.p_out.shard_rows_tiles)
        for c in range(plan.out_num_block_cols):
            core_cols = min(plan.p_out.shard_cols_tiles, plan.tensor_cols_tiles - c * plan.p_out.shard_cols_tiles)
            issuers = []
            for k in range(plan.issuers_per_core):
                lo = min(k * plan.issuer_rows_tiles, core_rows)
                hi = min(lo + plan.issuer_rows_tiles, core_rows)
                row_lo, row_hi = r * plan.p_out.shard_rows_tiles + lo, r * plan.p_out.shard_rows_tiles + hi
                col_lo, col_hi = c * plan.p_out.shard_cols_tiles, c * plan.p_out.shard_cols_tiles + core_cols
                issuers.append(
                    {
                        "issuer": k,
                        "noc": k,
                        "rows": (row_lo, row_hi),
                        "runs": _count_runs(plan, row_lo, row_hi, col_lo, col_hi),
                    }
                )
            targets[plan.p_out.block_core(r, c)] = {"block": (r, c), "issuers": issuers}

    out_cores = set(targets)
    source_only = sorted(
        plan.p_in.block_core(r, c)
        for r in range(plan.in_num_block_rows)
        for c in range(plan.in_num_block_cols)
        if plan.p_in.block_core(r, c) not in out_cores
    )
    return {
        "regime": "PULL",
        "issuers_per_core": plan.issuers_per_core,
        "block_rows_tiles": plan.block_rows_tiles,
        "block_cols_tiles": plan.block_cols_tiles,
        "kernel_core_ranges": [((out_range.start.x, out_range.start.y), (out_range.end.x, out_range.end.y))],
        "targets": targets,
        "source_only": source_only,
    }


def _count_runs(plan: Plan, row_lo: int, row_hi: int, col_lo: int, col_hi: int) -> int:
    """Run count of an issuer range, mirroring the kernel's piece walk (block knobs included)."""
    runs = 0
    for b_row in range(row_lo, row_hi, plan.block_rows_tiles):
        b_row_hi = min(b_row + plan.block_rows_tiles, row_hi)
        for b_col in range(col_lo, col_hi, plan.block_cols_tiles):
            b_col_hi = min(b_col + plan.block_cols_tiles, col_hi)
            sr, sc = plan.p_in.shard_rows_tiles, plan.p_in.shard_cols_tiles
            for br in range(b_row // sr, _div_up(b_row_hi, sr)):
                piece_rows = min(b_row_hi, (br + 1) * sr) - max(b_row, br * sr)
                for bc in range(b_col // sc, _div_up(b_col_hi, sc)):
                    piece_cols = min(b_col_hi, (bc + 1) * sc) - max(b_col, bc * sc)
                    one_run = piece_rows == 1 or (piece_cols == sc and piece_cols == plan.p_out.shard_cols_tiles)
                    runs += 1 if one_run else piece_rows
    return runs
