# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_reshard program descriptor: PULL reshard, BLOCK_SHARDED L1 -> BLOCK_SHARDED L1.

Every core that holds a block of out runs `issuers_per_core` instances of one dataflow kernel.
Each instance owns a contiguous range of the block's tile rows, enumerates on device the pieces of
in0's blocks overlapping it, and reads them straight into the out shard. No CB, no compute kernel,
no semaphore.

The built descriptor is cached per placement pair; a cache hit only rewrites the two buffer
addresses (common runtime args) and is dispatched as is.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
ISSUER_KERNEL = KERNEL_DIR / "g6_reshard_reader.cpp"

REGIME = "PULL"

# Issuer i of a core runs on ISSUER_PLACEMENTS[i] (processor, NoC). ISSUERS_PER_CORE selects how
# many of them run; the row split of every block derives from it.
ISSUER_PLACEMENTS = (
    (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
    (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
)
ISSUERS_PER_CORE = 2

# L1 per core the model leaves to the op (BB-05); out's shard is the op's only allocation.
L1_FREE_BYTES = 1347776

# Common runtime-arg slots, shared with the kernel.
_COMMON_ARG_IN_ADDR = 0
_COMMON_ARG_OUT_ADDR = 1


@dataclass(frozen=True)
class Placement:
    """A BLOCK_SHARDED placement of the padded 2-D view, in tiles."""

    x0: int
    y0: int
    x1: int
    y1: int
    block_height_tiles: int
    block_width_tiles: int
    row_major: bool

    def holder(self, block_row: int, block_col: int) -> tuple[int, int]:
        if self.row_major:
            return (self.x0 + block_col, self.y0 + block_row)
        return (self.x0 + block_row, self.y0 + block_col)


def placement_of(memory_config: ttnn.MemoryConfig, tile_shape) -> Placement:
    spec = memory_config.shard_spec
    (rng,) = spec.grid.ranges()
    tile_h, tile_w = tile_shape
    return Placement(
        x0=rng.start.x,
        y0=rng.start.y,
        x1=rng.end.x,
        y1=rng.end.y,
        block_height_tiles=spec.shape[0] // tile_h,
        block_width_tiles=spec.shape[1] // tile_w,
        row_major=spec.orientation == ttnn.ShardOrientation.ROW_MAJOR,
    )


def tensor_tiles(input_tensor: ttnn.Tensor) -> tuple[int, int]:
    """(tensor_height_tiles, tensor_width_tiles) of the padded 2-D view: each image tile-padded on its own."""
    padded = list(input_tensor.padded_shape)
    tile_h, tile_w = input_tensor.tile.tile_shape
    height = 1
    for d in padded[:-1]:
        height *= d
    return height // tile_h, padded[-1] // tile_w


def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


def block_grid(extent_tiles: int, block_tiles: int) -> int:
    return _div_up(extent_tiles, block_tiles)


def split_rows(row_begin: int, row_end: int, issuers: int) -> list[tuple[int, int]]:
    """Contiguous row ranges, earlier issuers taking the remainder (two issuers: ceil, floor)."""
    rows = row_end - row_begin
    base, extra = divmod(rows, issuers)
    ranges = []
    begin = row_begin
    for i in range(issuers):
        end = begin + base + (1 if i < extra else 0)
        ranges.append((begin, end))
        begin = end
    return ranges


def count_pieces(row_begin, row_end, col_begin, col_end, in_block_height_tiles, in_block_width_tiles) -> int:
    """Pieces the kernel walks for one issuer sub-block (same rule as the device walk)."""
    if row_begin >= row_end:
        return 0
    rows = (row_end - 1) // in_block_height_tiles - row_begin // in_block_height_tiles + 1
    cols = (col_end - 1) // in_block_width_tiles - col_begin // in_block_width_tiles + 1
    return rows * cols


@dataclass
class Plan:
    key: tuple
    descriptor: ttnn.ProgramDescriptor
    kernels: list
    core_range: tuple
    roles: list = field(default_factory=list)

    def report(self) -> dict:
        return {
            "regime": REGIME,
            "issuers_per_core": ISSUERS_PER_CORE,
            "core_range": self.core_range,
            "num_cores": len(self.roles),
            "issuer_placements": [
                (str(processor), str(noc)) for processor, noc in ISSUER_PLACEMENTS[:ISSUERS_PER_CORE]
            ],
        }


def plan_key(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> tuple:
    tile_shape = tuple(input_tensor.tile.tile_shape)
    in_place = placement_of(input_tensor.memory_config(), tile_shape)
    out_place = placement_of(memory_config, tile_shape)
    return (
        "g6_reshard",
        REGIME,
        ISSUERS_PER_CORE,
        tuple(input_tensor.padded_shape),
        tile_shape,
        str(input_tensor.dtype),
        str(input_tensor.layout),
        in_place,
        out_place,
        input_tensor.device().id(),
    )


def _program_hash(key: tuple) -> int:
    return int.from_bytes(hashlib.blake2b(repr(key).encode(), digest_size=8).digest(), "little")


def build_plan(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig, key: tuple) -> Plan:
    tile_shape = tuple(input_tensor.tile.tile_shape)
    in_place = placement_of(input_tensor.memory_config(), tile_shape)
    out_place = placement_of(memory_config, tile_shape)
    tensor_height_tiles, tensor_width_tiles = tensor_tiles(input_tensor)
    aligned_page_bytes = input_tensor.buffer_aligned_page_size()

    num_block_rows = block_grid(tensor_height_tiles, out_place.block_height_tiles)
    num_block_cols = block_grid(tensor_width_tiles, out_place.block_width_tiles)

    first = out_place.holder(0, 0)
    last = out_place.holder(num_block_rows - 1, num_block_cols - 1)
    core_range = (first, last)
    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(first[0], first[1]), ttnn.CoreCoord(last[0], last[1]))]
    )

    ct_args = [
        in_place.block_height_tiles,
        in_place.block_width_tiles,
        out_place.block_width_tiles,
        tensor_width_tiles,
        aligned_page_bytes,
    ]
    ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    rt_args = [ttnn.RuntimeArgs() for _ in range(ISSUERS_PER_CORE)]
    roles = []
    for block_row in range(num_block_rows):
        row_begin = block_row * out_place.block_height_tiles
        row_end = min(row_begin + out_place.block_height_tiles, tensor_height_tiles)
        for block_col in range(num_block_cols):
            col_begin = block_col * out_place.block_width_tiles
            col_end = min(col_begin + out_place.block_width_tiles, tensor_width_tiles)
            x, y = out_place.holder(block_row, block_col)
            issuers = []
            for issuer, (issuer_begin, issuer_end) in enumerate(split_rows(row_begin, row_end, ISSUERS_PER_CORE)):
                rt_args[issuer][x][y] = [issuer_begin, issuer_end, row_begin, col_begin, col_end]
                issuers.append(
                    {
                        "rows": (issuer_begin, issuer_end),
                        "pieces": count_pieces(
                            issuer_begin,
                            issuer_end,
                            col_begin,
                            col_end,
                            in_place.block_height_tiles,
                            in_place.block_width_tiles,
                        ),
                    }
                )
            roles.append(
                {"core": (x, y), "block": (block_row, block_col), "cols": (col_begin, col_end), "issuers": issuers}
            )

    kernels = []
    for issuer in range(ISSUERS_PER_CORE):
        processor, noc = ISSUER_PLACEMENTS[issuer]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(ISSUER_KERNEL),
                core_ranges=core_grid,
                compile_time_args=ct_args,
                runtime_args=rt_args[issuer],
                common_runtime_args=[0, 0],
                config=ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc),
            )
        )

    descriptor = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[])
    descriptor.custom_program_hash = _program_hash(key)
    return Plan(key=key, descriptor=descriptor, kernels=kernels, core_range=core_range, roles=roles)


_PLAN_CACHE: dict[tuple, Plan] = {}
_LAST_PLAN: list = [None]


def get_plan(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> Plan:
    key = plan_key(input_tensor, memory_config)
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = build_plan(input_tensor, memory_config, key)
        _PLAN_CACHE[key] = plan
    return plan


def create_program_descriptor(
    input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig
) -> ttnn.ProgramDescriptor:
    """The cached descriptor for this placement pair, with this call's buffer addresses."""
    plan = get_plan(input_tensor, memory_config)
    _LAST_PLAN[0] = plan
    # Both buffers are lockstep-allocated sharded L1 buffers: one address serves every core.
    addresses = [0, 0]
    addresses[_COMMON_ARG_IN_ADDR] = input_tensor.buffer_address()
    addresses[_COMMON_ARG_OUT_ADDR] = output_tensor.buffer_address()
    for kernel in plan.kernels:
        kernel.common_runtime_args = addresses
    # The descriptor holds copies of the kernel descriptors: hand the patched ones back.
    plan.descriptor.kernels = plan.kernels
    return plan.descriptor


def last_plan() -> dict | None:
    """The plan the last call selected (BB-08)."""
    plan = _LAST_PLAN[0]
    return None if plan is None else plan.report()


def role_dump(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> list:
    """Per core of the kernel range: its out block, column range, and each issuer's rows and piece count (BB-21)."""
    return get_plan(input_tensor, memory_config).roles
