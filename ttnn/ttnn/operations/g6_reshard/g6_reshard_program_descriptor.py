# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_reshard program descriptor: the PULL regime.

Every core that holds an out block pulls that block's tiles straight from the L1 of the cores holding
them, each run landing at its final slot of the out shard. One dataflow kernel source runs as
NUM_ISSUERS instances per out-block core, each pulling one row band of the block. No CB, no compute
kernel, no semaphore.

All tile geometry is taken from the padded 2-D view: rows = prod(padded_shape[:-1]) / 32,
cols = padded_shape[-1] / 32. Every placement value is a compile-time arg, so two placements never
share a compiled program; the per-core block coordinates are runtime args and the two buffer
addresses are common runtime args, rewritten on every call.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
PULL_KERNEL = KERNEL_DIR / "g6_reshard_reader.cpp"

TILE_DIM = 32

# Row bands per out block = pull kernels per out-block core (1 or 2: one per data-movement RISC).
NUM_ISSUERS = 2

# issuer_idx -> (RISC, NoC). Issuer 0 takes the reader slot, issuer 1 the writer slot, so the two bands
# travel on different NoCs.
ISSUER_PLACEMENT = (
    (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
    (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
)

REGIME_PULL = "PULL"

# Common RT arg slots, patched on every call.
_COMMON_ARG_IN0_ADDR = 0
_COMMON_ARG_OUT_ADDR = 1


def div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass(frozen=True)
class BlockPlacement:
    """A BLOCK_SHARDED placement of the 2-D view: block (r, c) sits on core (x0 + c, y0 + r) for
    ROW_MAJOR and on (x0 + r, y0 + c) for COL_MAJOR."""

    x0: int
    y0: int
    x1: int
    y1: int
    shard_row_tiles: int
    shard_col_tiles: int
    row_major: bool

    @property
    def grid_width(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def grid_height(self) -> int:
        return self.y1 - self.y0 + 1

    def block_counts(self, tensor_row_tiles: int, tensor_col_tiles: int) -> tuple[int, int]:
        return div_up(tensor_row_tiles, self.shard_row_tiles), div_up(tensor_col_tiles, self.shard_col_tiles)

    def core_of(self, block_row: int, block_col: int) -> tuple[int, int]:
        if self.row_major:
            return self.x0 + block_col, self.y0 + block_row
        return self.x0 + block_row, self.y0 + block_col

    def used_extent(self, block_rows: int, block_cols: int) -> tuple[int, int]:
        """(x span, y span) of the cores that hold a block."""
        return (block_cols, block_rows) if self.row_major else (block_rows, block_cols)


def placement_from_shard_spec(shard_spec) -> BlockPlacement:
    """The caller has checked that the grid is one rectangle and the shard dims are multiples of 32."""
    box = shard_spec.grid.bounding_box()
    return BlockPlacement(
        x0=box.start.x,
        y0=box.start.y,
        x1=box.end.x,
        y1=box.end.y,
        shard_row_tiles=int(shard_spec.shape[0]) // TILE_DIM,
        shard_col_tiles=int(shard_spec.shape[1]) // TILE_DIM,
        row_major=shard_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR,
    )


def tile_grid(padded_shape) -> tuple[int, int, int]:
    """(rows, cols, images) of the padded 2-D view in tiles; images = prod(padded_shape[:-2])."""
    dims = [int(d) for d in padded_shape]
    return prod(dims[:-1]) // TILE_DIM, dims[-1] // TILE_DIM, prod(dims[:-2])


@dataclass(frozen=True)
class ReshardPlan:
    tensor_row_tiles: int
    tensor_col_tiles: int
    tile_stride_bytes: int
    in_placement: BlockPlacement
    out_placement: BlockPlacement
    num_issuers: int = NUM_ISSUERS
    regime: str = REGIME_PULL

    @property
    def in_blocks(self) -> tuple[int, int]:
        return self.in_placement.block_counts(self.tensor_row_tiles, self.tensor_col_tiles)

    @property
    def out_blocks(self) -> tuple[int, int]:
        return self.out_placement.block_counts(self.tensor_row_tiles, self.tensor_col_tiles)

    def out_block_cores(self) -> list[tuple[tuple[int, int], int, int]]:
        """((x, y), block_row, block_col) for every out block."""
        out_block_rows, out_block_cols = self.out_blocks
        return [
            (self.out_placement.core_of(block_row, block_col), block_row, block_col)
            for block_row in range(out_block_rows)
            for block_col in range(out_block_cols)
        ]

    def pull_core_range(self) -> ttnn.CoreRangeSet:
        """The out-block cores: one rectangle from out's grid start, trimmed to the block counts."""
        p = self.out_placement
        x_span, y_span = p.used_extent(*self.out_blocks)
        return ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(p.x0, p.y0), ttnn.CoreCoord(p.x0 + x_span - 1, p.y0 + y_span - 1))}
        )

    def issuer_role(self) -> str:
        return "pull:" + "+".join(f"issuer{i}/{ISSUER_PLACEMENT[i][1].name}" for i in range(self.num_issuers))

    def core_roles(self) -> dict[tuple[int, int], str]:
        """Role of every core of out's grid (BB-21): pullers, and grid cores beyond the block count idle."""
        p = self.out_placement
        roles = {(x, y): "idle" for x in range(p.x0, p.x1 + 1) for y in range(p.y0, p.y1 + 1)}
        for core, _, _ in self.out_block_cores():
            roles[core] = self.issuer_role()
        return roles

    def report(self) -> dict:
        return {"regime": self.regime, "num_issuers": self.num_issuers, "cores": self.core_roles()}


def create_program_descriptor(
    input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor, plan: ReshardPlan
) -> ttnn.ProgramDescriptor:
    assert 1 <= plan.num_issuers <= len(ISSUER_PLACEMENT), plan.num_issuers
    in_block_rows, in_block_cols = plan.in_blocks
    core_range = plan.pull_core_range()

    geometry_ct_args = [
        plan.num_issuers,
        plan.tensor_row_tiles,
        plan.tensor_col_tiles,
        plan.in_placement.shard_row_tiles,
        plan.in_placement.shard_col_tiles,
        in_block_cols,
        plan.out_placement.shard_row_tiles,
        plan.out_placement.shard_col_tiles,
        plan.tile_stride_bytes,
    ]
    in0_accessor_ct_args = list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    common_rt_args = [0, 0]
    common_rt_args[_COMMON_ARG_IN0_ADDR] = input_tensor.buffer_address()
    common_rt_args[_COMMON_ARG_OUT_ADDR] = output_tensor.buffer_address()

    block_rt_args = ttnn.RuntimeArgs()
    for (x, y), block_row, block_col in plan.out_block_cores():
        block_rt_args[x][y] = [block_row, block_col]

    kernels = []
    for issuer_idx in range(plan.num_issuers):
        processor, noc = ISSUER_PLACEMENT[issuer_idx]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(PULL_KERNEL),
                core_ranges=core_range,
                compile_time_args=[issuer_idx] + geometry_ct_args + in0_accessor_ct_args,
                runtime_args=block_rt_args,
                common_runtime_args=list(common_rt_args),
                config=ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc),
            )
        )

    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[])


def patch_addresses(descriptor: ttnn.ProgramDescriptor, input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor) -> None:
    """Rewrite the only per-call values: the in0 and out base addresses of every pull kernel."""
    in0_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    for kernel in descriptor.kernels:
        kernel.common_runtime_args[_COMMON_ARG_IN0_ADDR] = in0_addr
        kernel.common_runtime_args[_COMMON_ARG_OUT_ADDR] = out_addr
