# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_reshard program descriptor: the PULL regime.

Every core holding an out block runs one gather kernel per issuer RISC; each issuer reads its
rows of the block from the in0 source shards straight into the out shard. No CBs, no compute
kernel, no semaphores, no per-core runtime args: every per-core value derives in-kernel from the
placement compile-time args and the core's logical coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
GATHER_KERNEL = KERNEL_DIR / "g6_reshard_gather.cpp"

# --- Blocking knobs (single sources; see op_design.md, Blocking Model) -------------------------
# Data-movement RISCs per target core that issue reads: issuer 0 on RISCV_1 / NOC0, issuer 1 on
# RISCV_0 / NOC1.
ISSUERS_PER_CORE = 2
# Tile rows of one issuer block; None = the issuer's whole share of its out shard (one block).
BLOCK_ROWS_TILES = None
# Tile columns of one issuer block; None = the whole out shard width (one block).
BLOCK_COLS_TILES = None

_ISSUER_CONFIGS = (
    (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
    (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
)

# Common runtime args: [in_addr, out_addr, in_block_row_noc..., in_block_col_noc...].
IN_ADDR_ARG = 0
OUT_ADDR_ARG = 1


def _div_up(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class Placement:
    """A BLOCK_SHARDED placement in tiles: grid origin, blocks per axis, shard shape, orientation."""

    x0: int
    y0: int
    shard_rows_tiles: int
    shard_cols_tiles: int
    col_major: bool
    num_block_rows: int
    num_block_cols: int

    def core_of(self, block_row: int, block_col: int) -> tuple[int, int]:
        """Logical core holding block (block_row, block_col)."""
        if self.col_major:
            return self.x0 + block_row, self.y0 + block_col
        return self.x0 + block_col, self.y0 + block_row

    def block_of(self, x: int, y: int) -> tuple[int, int]:
        if self.col_major:
            return x - self.x0, y - self.y0
        return y - self.y0, x - self.x0

    def trimmed_range(self) -> ttnn.CoreRange:
        """The cores that hold a block: the grid trimmed to the block counts."""
        nx, ny = (
            (self.num_block_rows, self.num_block_cols) if self.col_major else (self.num_block_cols, self.num_block_rows)
        )
        return ttnn.CoreRange(ttnn.CoreCoord(self.x0, self.y0), ttnn.CoreCoord(self.x0 + nx - 1, self.y0 + ny - 1))


def placement_from_shard_spec(shard_spec, tensor_rows_tiles: int, tensor_cols_tiles: int, tile_hw) -> Placement:
    box = shard_spec.grid.bounding_box()
    tile_h, tile_w = tile_hw
    shard_h, shard_w = int(shard_spec.shape[0]), int(shard_spec.shape[1])
    shard_rows_tiles, shard_cols_tiles = shard_h // tile_h, shard_w // tile_w
    return Placement(
        x0=box.start.x,
        y0=box.start.y,
        shard_rows_tiles=shard_rows_tiles,
        shard_cols_tiles=shard_cols_tiles,
        col_major=shard_spec.orientation == ttnn.ShardOrientation.COL_MAJOR,
        num_block_rows=_div_up(tensor_rows_tiles, shard_rows_tiles),
        num_block_cols=_div_up(tensor_cols_tiles, shard_cols_tiles),
    )


def tensor_tiles(padded_shape, tile_hw) -> tuple[int, int]:
    """(tile rows, tile columns) of the padded 2-D view: all dims but the last fold into rows."""
    tile_h, tile_w = tile_hw
    outer = 1
    for d in padded_shape[:-2]:
        outer *= int(d)
    return outer * int(padded_shape[-2]) // tile_h, int(padded_shape[-1]) // tile_w


@dataclass(frozen=True)
class Plan:
    tensor_rows_tiles: int
    tensor_cols_tiles: int
    page_bytes: int
    p_in: Placement
    p_out: Placement
    issuers_per_core: int
    block_rows_tiles: int
    block_cols_tiles: int

    @property
    def issuer_rows_share(self) -> int:
        return _div_up(self.p_out.shard_rows_tiles, self.issuers_per_core)


def make_plan(
    tensor_rows_tiles: int, tensor_cols_tiles: int, page_bytes: int, p_in: Placement, p_out: Placement
) -> Plan:
    if not 1 <= ISSUERS_PER_CORE <= len(_ISSUER_CONFIGS):
        raise RuntimeError(
            f"g6_reshard: ISSUERS_PER_CORE={ISSUERS_PER_CORE}; a core has {len(_ISSUER_CONFIGS)} DM RISCs"
        )
    issuer_rows_share = _div_up(p_out.shard_rows_tiles, ISSUERS_PER_CORE)
    return Plan(
        tensor_rows_tiles=tensor_rows_tiles,
        tensor_cols_tiles=tensor_cols_tiles,
        page_bytes=page_bytes,
        p_in=p_in,
        p_out=p_out,
        issuers_per_core=ISSUERS_PER_CORE,
        block_rows_tiles=BLOCK_ROWS_TILES or issuer_rows_share,
        block_cols_tiles=BLOCK_COLS_TILES or p_out.shard_cols_tiles,
    )


# --- Host mirror of the kernel's piece walk (plan dump only; never on the dispatch path) --------


def issuer_rows(plan: Plan, r_out: int, issuer: int) -> tuple[int, int]:
    """Tensor tile rows [lo, hi) issuer `issuer` of out block row r_out owns."""
    block_row0 = r_out * plan.p_out.shard_rows_tiles
    core_rows = min(plan.p_out.shard_rows_tiles, plan.tensor_rows_tiles - block_row0)
    lo = min(issuer * plan.issuer_rows_share, core_rows)
    hi = min(lo + plan.issuer_rows_share, core_rows)
    return block_row0 + lo, block_row0 + hi


def core_cols(plan: Plan, c_out: int) -> tuple[int, int]:
    block_col0 = c_out * plan.p_out.shard_cols_tiles
    return block_col0, block_col0 + min(plan.p_out.shard_cols_tiles, plan.tensor_cols_tiles - block_col0)


def issuer_runs(plan: Plan, r_out: int, c_out: int, issuer: int):
    """(source block (br, bc), run tiles) of every NoC read the issuer issues, in issue order."""
    p_in, p_out = plan.p_in, plan.p_out
    row_lo0, row_hi0 = issuer_rows(plan, r_out, issuer)
    col_lo0, col_hi0 = core_cols(plan, c_out)
    runs = []
    for row_lo in range(row_lo0, row_hi0, plan.block_rows_tiles):
        row_hi = min(row_lo + plan.block_rows_tiles, row_hi0)
        for col_lo in range(col_lo0, col_hi0, plan.block_cols_tiles):
            col_hi = min(col_lo + plan.block_cols_tiles, col_hi0)
            br = row_lo // p_in.shard_rows_tiles
            while br * p_in.shard_rows_tiles < row_hi:
                piece_rows = min(row_hi, (br + 1) * p_in.shard_rows_tiles) - max(row_lo, br * p_in.shard_rows_tiles)
                bc = col_lo // p_in.shard_cols_tiles
                while bc * p_in.shard_cols_tiles < col_hi:
                    piece_cols = min(col_hi, (bc + 1) * p_in.shard_cols_tiles) - max(col_lo, bc * p_in.shard_cols_tiles)
                    one_run = piece_rows == 1 or (
                        p_in.shard_cols_tiles == p_out.shard_cols_tiles and piece_cols == p_in.shard_cols_tiles
                    )
                    if one_run:
                        runs.append(((br, bc), piece_rows * piece_cols))
                    else:
                        runs.extend(((br, bc), piece_cols) for _ in range(piece_rows))
                    bc += 1
                br += 1
    return runs


# --- Descriptor ---------------------------------------------------------------------------------


def _noc_tables(device, p_in: Placement) -> tuple[list[int], list[int]]:
    """Virtual NoC coordinate of each in0 block row and block column.

    The kernel builds a source core's NoC address from one entry of each table, which holds only
    if virtual x depends on logical x alone and virtual y on logical y alone; asserted per core.
    """

    def virtual(block_row, block_col):
        x, y = p_in.core_of(block_row, block_col)
        v = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        return v.x, v.y

    row_noc, col_noc = [], []
    for br in range(p_in.num_block_rows):
        vx, vy = virtual(br, 0)
        row_noc.append(vx if p_in.col_major else vy)
    for bc in range(p_in.num_block_cols):
        vx, vy = virtual(0, bc)
        col_noc.append(vy if p_in.col_major else vx)
    for br in range(p_in.num_block_rows):
        for bc in range(p_in.num_block_cols):
            want = (row_noc[br], col_noc[bc]) if p_in.col_major else (col_noc[bc], row_noc[br])
            if virtual(br, bc) != want:
                raise RuntimeError(
                    f"g6_reshard: virtual NoC coordinates of in0 core {p_in.core_of(br, bc)} are not separable "
                    f"({virtual(br, bc)} vs {want})"
                )
    return row_noc, col_noc


def _compile_time_args(plan: Plan, issuer: int) -> list[int]:
    p_in, p_out = plan.p_in, plan.p_out
    return [
        plan.issuers_per_core,
        issuer,
        plan.block_rows_tiles,
        plan.block_cols_tiles,
        plan.page_bytes,
        plan.tensor_rows_tiles,
        plan.tensor_cols_tiles,
        p_in.shard_rows_tiles,
        p_in.shard_cols_tiles,
        int(p_in.col_major),
        p_in.num_block_rows,
        p_in.num_block_cols,
        p_out.shard_rows_tiles,
        p_out.shard_cols_tiles,
        int(p_out.col_major),
        p_out.x0,
        p_out.y0,
    ]


def create_program_descriptor(plan: Plan, device, in_addr: int, out_addr: int) -> ttnn.ProgramDescriptor:
    row_noc, col_noc = _noc_tables(device, plan.p_in)
    common_args = [in_addr, out_addr, *row_noc, *col_noc]
    cores = ttnn.CoreRangeSet([plan.p_out.trimmed_range()])
    kernels = []
    for issuer in range(plan.issuers_per_core):
        processor, noc = _ISSUER_CONFIGS[issuer]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(GATHER_KERNEL),
                core_ranges=cores,
                compile_time_args=_compile_time_args(plan, issuer),
                common_runtime_args=common_args,
                config=ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc),
            )
        )
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[])


def set_buffer_addresses(descriptor: ttnn.ProgramDescriptor, in_addr: int, out_addr: int) -> None:
    """Patch the only per-call values of a cached descriptor (in place)."""
    for kernel in descriptor.kernels:
        kernel.common_runtime_args[IN_ADDR_ARG] = in_addr
        kernel.common_runtime_args[OUT_ADDR_ARG] = out_addr
