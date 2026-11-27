# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_reshard program: PULL reshard of a BLOCK_SHARDED L1 tensor.

Every core holding a block of out runs the issuer kernel up to ISSUERS_PER_CORE times
(one per data-movement RISC-V / NoC). Each instance reads, straight into the out shard,
the pieces of in0's blocks that overlap its contiguous range of the block's tile rows.
No CB, no compute kernel, no semaphore: the source shards are resident and immutable.

The program depends only on the placements, so it is built once per placement pair and
cached; a hit patches the two buffer addresses (common runtime args) and nothing else.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
ISSUER_KERNEL = str(KERNEL_DIR / "g6_reshard_reader.cpp")

TILE = 32

# Issuers per out-holding core: the block's tile rows are split into this many
# contiguous ranges, one per data-movement RISC-V. Slot i uses _ISSUER_CONFIGS[i].
ISSUERS_PER_CORE = 2


def _issuer_configs():
    return [
        ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
        ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
    ]


MAX_ISSUERS_PER_CORE = 2

# Common runtime arg slots (identical on every core: both tensors are lockstep-allocated).
CRTA_IN_ADDR = 0
CRTA_OUT_ADDR = 1


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class Placement:
    """BLOCK_SHARDED placement of the padded 2-D view, in tiles."""

    grid: tuple  # ((x0, y0, x1, y1), ...) inclusive core rectangles
    block_height_tiles: int
    block_width_tiles: int
    row_major: bool

    @staticmethod
    def of(memory_config: ttnn.MemoryConfig) -> "Placement":
        spec = memory_config.shard_spec
        grid = tuple(sorted((r.start.x, r.start.y, r.end.x, r.end.y) for r in spec.grid.ranges()))
        return Placement(
            grid=grid,
            block_height_tiles=spec.shape[0] // TILE,
            block_width_tiles=spec.shape[1] // TILE,
            row_major=spec.orientation == ttnn.ShardOrientation.ROW_MAJOR,
        )

    def num_blocks(self, tensor_height_tiles: int, tensor_width_tiles: int) -> tuple[int, int]:
        return (
            _ceil_div(tensor_height_tiles, self.block_height_tiles),
            _ceil_div(tensor_width_tiles, self.block_width_tiles),
        )

    def holder(self, block_row: int, block_col: int) -> tuple[int, int]:
        x0, y0 = self.grid[0][0], self.grid[0][1]
        if self.row_major:
            return (x0 + block_col, y0 + block_row)
        return (x0 + block_row, y0 + block_col)


def tensor_tiles(input_tensor: ttnn.Tensor) -> tuple[int, int]:
    """(tile rows, tile columns) of the padded 2-D view: every image is tile-padded on its own."""
    padded = list(input_tensor.padded_shape)
    height = 1
    for d in padded[:-1]:
        height *= d
    return height // TILE, padded[-1] // TILE


def _split_rows(row_begin: int, row_end: int, parts: int) -> list[tuple[int, int]]:
    """Contiguous ranges, the larger ones first (issuer 0 gets ceil(rows / parts))."""
    rows = row_end - row_begin
    base, extra = divmod(rows, parts)
    ranges, start = [], row_begin
    for i in range(parts):
        end = start + base + (1 if i < extra else 0)
        ranges.append((start, end))
        start = end
    return ranges


def _piece_count(rows: tuple[int, int], cols: tuple[int, int], in_placement: Placement) -> int:
    if rows[0] >= rows[1] or cols[0] >= cols[1]:
        return 0
    ih, iw = in_placement.block_height_tiles, in_placement.block_width_tiles
    return ((rows[1] - 1) // ih - rows[0] // ih + 1) * ((cols[1] - 1) // iw - cols[0] // iw + 1)


@dataclass
class Plan:
    """A built program for one placement pair, plus its BB-08 report and BB-21 role dump."""

    key: tuple
    program_descriptor: ttnn.ProgramDescriptor
    kernels: list
    report: dict
    roles: list = field(default_factory=list)

    def bind(self, input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor) -> ttnn.ProgramDescriptor:
        in_addr, out_addr = input_tensor.buffer_address(), output_tensor.buffer_address()
        for kernel in self.kernels:
            kernel.common_runtime_args[CRTA_IN_ADDR] = in_addr
            kernel.common_runtime_args[CRTA_OUT_ADDR] = out_addr
        # program_descriptor.kernels holds copies: hand the patched kernels back.
        self.program_descriptor.kernels = self.kernels
        return self.program_descriptor


_PLAN_CACHE: dict = {}
_LAST_PLAN: list = [None]


def plan_key(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> tuple:
    return (
        "g6_reshard",
        tuple(input_tensor.padded_shape),
        str(input_tensor.dtype),
        str(input_tensor.layout),
        Placement.of(input_tensor.memory_config()),
        Placement.of(memory_config),
        input_tensor.device().id(),
        ISSUERS_PER_CORE,
    )


def get_plan(input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> Plan:
    key = plan_key(input_tensor, memory_config)
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = _build_plan(key, input_tensor, output_tensor, memory_config)
        _PLAN_CACHE[key] = plan
    _LAST_PLAN[0] = plan
    return plan


def last_plan_report() -> dict | None:
    """BB-08: the plan the last call selected."""
    return None if _LAST_PLAN[0] is None else dict(_LAST_PLAN[0].report)


def last_plan_roles() -> list | None:
    """BB-21: per core of the kernel range, its out block, each issuer's rows/cols and piece count."""
    return None if _LAST_PLAN[0] is None else list(_LAST_PLAN[0].roles)


def _build_plan(key, input_tensor, output_tensor, memory_config) -> Plan:
    if not 1 <= ISSUERS_PER_CORE <= MAX_ISSUERS_PER_CORE:
        raise ValueError(f"g6_reshard: ISSUERS_PER_CORE={ISSUERS_PER_CORE} not in [1, {MAX_ISSUERS_PER_CORE}]")

    in_placement = Placement.of(input_tensor.memory_config())
    out_placement = Placement.of(memory_config)
    tensor_height_tiles, tensor_width_tiles = tensor_tiles(input_tensor)
    num_block_rows, num_block_cols = out_placement.num_blocks(tensor_height_tiles, tensor_width_tiles)
    bh, bw = out_placement.block_height_tiles, out_placement.block_width_tiles

    aligned_page_bytes = input_tensor.buffer_aligned_page_size()
    if output_tensor.buffer_aligned_page_size() != aligned_page_bytes:
        raise RuntimeError(
            f"g6_reshard: aligned page size differs (in {aligned_page_bytes}, "
            f"out {output_tensor.buffer_aligned_page_size()})"
        )

    # Kernel range: exactly the block-holder rectangle of out's placement.
    first = out_placement.holder(0, 0)
    last = out_placement.holder(num_block_rows - 1, num_block_cols - 1)
    core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*first), ttnn.CoreCoord(*last))})

    issuer_rt_args = [ttnn.RuntimeArgs() for _ in range(ISSUERS_PER_CORE)]
    roles = []
    for block_row in range(num_block_rows):
        row_begin = block_row * bh
        row_end = min(row_begin + bh, tensor_height_tiles)
        issuer_rows = _split_rows(row_begin, row_end, ISSUERS_PER_CORE)
        for block_col in range(num_block_cols):
            col_begin = block_col * bw
            col_end = min(col_begin + bw, tensor_width_tiles)
            x, y = out_placement.holder(block_row, block_col)
            issuers = []
            for i, (r0, r1) in enumerate(issuer_rows):
                issuer_rt_args[i][x][y] = [r0, r1, row_begin, col_begin, col_end]
                issuers.append(
                    {
                        "rows": (r0, r1),
                        "cols": (col_begin, col_end),
                        "pieces": _piece_count((r0, r1), (col_begin, col_end), in_placement),
                    }
                )
            roles.append({"core": (x, y), "block": (block_row, block_col), "issuers": issuers})

    ct_args = [
        in_placement.block_height_tiles,
        in_placement.block_width_tiles,
        bw,
        tensor_width_tiles,
        aligned_page_bytes,
    ]
    ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    configs = _issuer_configs()
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=ISSUER_KERNEL,
            core_ranges=core_range,
            compile_time_args=ct_args,
            runtime_args=issuer_rt_args[i],
            common_runtime_args=[input_tensor.buffer_address(), output_tensor.buffer_address()],
            config=configs[i],
        )
        for i in range(ISSUERS_PER_CORE)
    ]

    program_descriptor = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[])
    # generic_op's cache key: the whole placement pair, so I0/I3 and P/O1 never share a program.
    digest = hashlib.blake2b(repr(key).encode(), digest_size=8).digest()
    program_descriptor.custom_program_hash = int.from_bytes(digest, "little")

    report = {
        "regime": "PULL",
        "issuers_per_core": ISSUERS_PER_CORE,
        "kernel_core_range": (first, last),
        "num_blocks": (num_block_rows, num_block_cols),
        "block_tiles": (bh, bw),
        "in_block_tiles": (in_placement.block_height_tiles, in_placement.block_width_tiles),
        "tensor_tiles": (tensor_height_tiles, tensor_width_tiles),
    }
    return Plan(key=key, program_descriptor=program_descriptor, kernels=kernels, report=report, roles=roles)
