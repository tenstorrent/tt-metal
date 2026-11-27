# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_reshard program descriptor: PULL reshard, block = out shard, on-device piece walk.

Every out-holding core runs ISSUERS_PER_CORE instances of one dataflow kernel. Issuer k reads
its contiguous share of the block's tile rows from in0's shards straight into the out shard.
There is no CB, no compute kernel and no semaphore.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
ISSUER_KERNEL_SOURCE = KERNEL_DIR / "g6_reshard_reader.cpp"

REGIME = "PULL"

# Block knob: number of data-movement RISC-Vs that split each out block's tile rows.
ISSUERS_PER_CORE = 2
# (processor, NoC) of issuer k; one entry per data-movement RISC-V, so ISSUERS_PER_CORE <= len().
ISSUER_CONFIGS = (
    (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
    (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
)

# Per-core runtime-arg layout of the issuer kernel.
_RT_BLOCK_ROW_BEGIN, _RT_BLOCK_COL_BEGIN, _RT_BLOCK_COL_END, _RT_ROW_BEGIN, _RT_ROW_END = range(5)
# Common runtime-arg layout: the only per-call values.
_CRT_IN_ADDR, _CRT_OUT_ADDR = range(2)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class BlockPlacement:
    """A BLOCK_SHARDED placement of the padded 2-D tile view."""

    x0: int
    y0: int
    x1: int
    y1: int
    block_height_tiles: int
    block_width_tiles: int
    row_major: bool

    @classmethod
    def of(cls, memory_config: ttnn.MemoryConfig, tile_shape) -> "BlockPlacement":
        spec = memory_config.shard_spec
        box = spec.grid.bounding_box()
        tile_h, tile_w = tile_shape
        return cls(
            x0=box.start.x,
            y0=box.start.y,
            x1=box.end.x,
            y1=box.end.y,
            block_height_tiles=int(spec.shape[0]) // tile_h,
            block_width_tiles=int(spec.shape[1]) // tile_w,
            row_major=spec.orientation == ttnn.ShardOrientation.ROW_MAJOR,
        )

    def core_of(self, block_row: int, block_col: int) -> tuple[int, int]:
        if self.row_major:
            return self.x0 + block_col, self.y0 + block_row
        return self.x0 + block_row, self.y0 + block_col

    def key(self) -> tuple:
        return (
            self.x0,
            self.y0,
            self.x1,
            self.y1,
            self.block_height_tiles,
            self.block_width_tiles,
            self.row_major,
        )


@dataclass(frozen=True)
class OutBlock:
    block_row: int
    block_col: int
    core: tuple[int, int]
    row_begin: int
    row_end: int
    col_begin: int
    col_end: int
    issuer_rows: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ReshardPlan:
    tensor_height_tiles: int
    tensor_width_tiles: int
    src: BlockPlacement
    dst: BlockPlacement
    issuers_per_core: int

    @property
    def num_block_rows(self) -> int:
        return _ceil_div(self.tensor_height_tiles, self.dst.block_height_tiles)

    @property
    def num_block_cols(self) -> int:
        return _ceil_div(self.tensor_width_tiles, self.dst.block_width_tiles)

    def kernel_rectangle(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Out's block holders: the trimmed grid the allocator uses for BLOCK_SHARDED."""
        last = self.dst.core_of(self.num_block_rows - 1, self.num_block_cols - 1)
        return (self.dst.x0, self.dst.y0), last

    def split_rows(self, row_begin: int, row_end: int) -> tuple[tuple[int, int], ...]:
        """Contiguous row ranges per issuer; the first (rows % n) issuers take one extra row."""
        rows = row_end - row_begin
        n = self.issuers_per_core
        ranges = []
        begin = row_begin
        for k in range(n):
            size = rows // n + (1 if k < rows % n else 0)
            ranges.append((begin, begin + size))
            begin += size
        return tuple(ranges)

    def blocks(self):
        bh = self.dst.block_height_tiles
        bw = self.dst.block_width_tiles
        for r in range(self.num_block_rows):
            row_begin = r * bh
            row_end = min(row_begin + bh, self.tensor_height_tiles)
            for c in range(self.num_block_cols):
                col_begin = c * bw
                col_end = min(col_begin + bw, self.tensor_width_tiles)
                yield OutBlock(
                    block_row=r,
                    block_col=c,
                    core=self.dst.core_of(r, c),
                    row_begin=row_begin,
                    row_end=row_end,
                    col_begin=col_begin,
                    col_end=col_end,
                    issuer_rows=self.split_rows(row_begin, row_end),
                )

    def piece_count(self, row_begin: int, row_end: int, col_begin: int, col_end: int) -> int:
        if row_begin >= row_end:
            return 0
        ih = self.src.block_height_tiles
        iw = self.src.block_width_tiles
        in_rows = (row_end - 1) // ih - row_begin // ih + 1
        in_cols = (col_end - 1) // iw - col_begin // iw + 1
        return in_rows * in_cols

    def key(self) -> tuple:
        return (
            self.tensor_height_tiles,
            self.tensor_width_tiles,
            self.src.key(),
            self.dst.key(),
            self.issuers_per_core,
        )


def build_plan(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ReshardPlan:
    tile_h, tile_w = input_tensor.tile.tile_shape
    padded = list(input_tensor.padded_shape)
    height = 1
    for d in padded[:-1]:
        height *= int(d)
    if not 1 <= ISSUERS_PER_CORE <= len(ISSUER_CONFIGS):
        raise RuntimeError(f"g6_reshard: ISSUERS_PER_CORE={ISSUERS_PER_CORE} outside [1, {len(ISSUER_CONFIGS)}]")
    return ReshardPlan(
        tensor_height_tiles=height // tile_h,
        tensor_width_tiles=int(padded[-1]) // tile_w,
        src=BlockPlacement.of(input_tensor.memory_config(), (tile_h, tile_w)),
        dst=BlockPlacement.of(memory_config, (tile_h, tile_w)),
        issuers_per_core=ISSUERS_PER_CORE,
    )


def role_dump(plan: ReshardPlan) -> list[dict]:
    """Per core of the kernel range: its out block, and each issuer's rows and piece count."""
    roles = []
    for block in plan.blocks():
        issuers = []
        for k, (row_begin, row_end) in enumerate(block.issuer_rows):
            processor, noc = ISSUER_CONFIGS[k]
            issuers.append(
                {
                    "issuer": k,
                    "processor": str(processor),
                    "noc": str(noc),
                    "rows": (row_begin, row_end),
                    "pieces": plan.piece_count(row_begin, row_end, block.col_begin, block.col_end),
                }
            )
        roles.append(
            {
                "core": block.core,
                "out_block": (block.block_row, block.block_col),
                "cols": (block.col_begin, block.col_end),
                "issuers": issuers,
            }
        )
    return roles


def plan_report(plan: ReshardPlan) -> dict:
    return {
        "regime": REGIME,
        "issuers_per_core": plan.issuers_per_core,
        "kernel_rectangle": plan.kernel_rectangle(),
        "num_out_blocks": (plan.num_block_rows, plan.num_block_cols),
        "out_block_tiles": (plan.dst.block_height_tiles, plan.dst.block_width_tiles),
        "in_block_tiles": (plan.src.block_height_tiles, plan.src.block_width_tiles),
    }


class CachedProgram:
    """A built issuer program. Only the two buffer addresses change between calls."""

    def __init__(self, kernels: list, program_hash: int, plan: ReshardPlan):
        self.kernels = kernels
        self.program_hash = program_hash
        self.plan = plan

    def bind(self, in_addr: int, out_addr: int) -> ttnn.ProgramDescriptor:
        common_args = [0] * 2
        common_args[_CRT_IN_ADDR] = in_addr
        common_args[_CRT_OUT_ADDR] = out_addr
        for kernel in self.kernels:
            kernel.common_runtime_args = common_args
        descriptor = ttnn.ProgramDescriptor(kernels=self.kernels, semaphores=[], cbs=[])
        descriptor.custom_program_hash = self.program_hash
        return descriptor


def program_hash(cache_key: tuple) -> int:
    digest = hashlib.blake2b(repr(("g6_reshard", str(ISSUER_KERNEL_SOURCE), cache_key)).encode(), digest_size=8)
    return int.from_bytes(digest.digest(), "little")


def create_program(
    input_tensor: ttnn.Tensor, plan: ReshardPlan, aligned_page_bytes: int, cache_key: tuple
) -> CachedProgram:
    (kx0, ky0), (kx1, ky1) = plan.kernel_rectangle()
    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(kx0, ky0), ttnn.CoreCoord(kx1, ky1))])

    in_accessor_args = ttnn.TensorAccessorArgs(input_tensor)
    if len(in_accessor_args.get_common_runtime_args()) != 0:
        raise RuntimeError("g6_reshard: expected in0's TensorAccessorArgs to be fully compile-time")
    compile_time_args = [
        plan.src.block_height_tiles,
        plan.src.block_width_tiles,
        plan.dst.block_width_tiles,
        plan.tensor_width_tiles,
        aligned_page_bytes,
    ]
    compile_time_args.extend(in_accessor_args.get_compile_time_args())

    per_issuer_args = [ttnn.RuntimeArgs() for _ in range(plan.issuers_per_core)]
    for block in plan.blocks():
        x, y = block.core
        for k, (row_begin, row_end) in enumerate(block.issuer_rows):
            args = [0] * 5
            args[_RT_BLOCK_ROW_BEGIN] = block.row_begin
            args[_RT_BLOCK_COL_BEGIN] = block.col_begin
            args[_RT_BLOCK_COL_END] = block.col_end
            args[_RT_ROW_BEGIN] = row_begin
            args[_RT_ROW_END] = row_end
            per_issuer_args[k][x][y] = args

    kernels = []
    for k in range(plan.issuers_per_core):
        processor, noc = ISSUER_CONFIGS[k]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(ISSUER_KERNEL_SOURCE),
                core_ranges=core_ranges,
                compile_time_args=compile_time_args,
                runtime_args=per_issuer_args[k],
                common_runtime_args=[0, 0],
                config=ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc),
            )
        )
    return CachedProgram(kernels, program_hash(cache_key), plan)
