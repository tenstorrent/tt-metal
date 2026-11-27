# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Program descriptor of g6_to_memory_config.

Regime selection, work split, kernels and the descriptor cache. Every page is moved by a core that
holds it in its own L1 (owner regimes shard_pull / shard_push / bank_pull / bank_push, one mover per
data-movement RISC); only DRAM -> DRAM goes through a staging CB on a full-grid split (staged).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
MOVER_KERNEL = KERNEL_DIR / "g6_to_memory_config_mover.cpp"
STAGED_READER_KERNEL = KERNEL_DIR / "g6_to_memory_config_reader.cpp"
STAGED_WRITER_KERNEL = KERNEL_DIR / "g6_to_memory_config_writer.cpp"

# Blocking knobs (op_design.md, Blocking Model). Every CB size, loop bound and split derives from these.
MOVERS_PER_CORE = 2
STAGED_BLOCK_PAGES = 16
STAGED_BUFFER_DEPTH = 2

# L1 per core the model leaves to the op (BB-05).
L1_BUDGET_BYTES = 1347776

# Mover m runs on its own data-movement RISC and NoC, so the two halves of a core's pages issue in parallel.
_MOVER_CONFIGS = (
    (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
    (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
)
assert 1 <= MOVERS_PER_CORE <= len(_MOVER_CONFIGS)

CB_STAGED_PAGES = 0

SHARD_PULL = "shard_pull"
SHARD_PUSH = "shard_push"
BANK_PULL = "bank_pull"
BANK_PUSH = "bank_push"
STAGED = "staged"
PULL_REGIMES = (SHARD_PULL, BANK_PULL)
BANK_REGIMES = (BANK_PULL, BANK_PUSH)
SHARD_REGIMES = (SHARD_PULL, SHARD_PUSH)

_BUFFER_NAME = {ttnn.BufferType.DRAM: "dram", ttnn.BufferType.L1: "l1"}
_SCHEME_NAME = {
    ttnn.TensorMemoryLayout.INTERLEAVED: "interleaved",
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: "height_sharded",
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: "width_sharded",
    ttnn.TensorMemoryLayout.BLOCK_SHARDED: "block_sharded",
}


# ---------------------------------------------------------------------------
# Placements and tile geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Placement:
    buffer: str  # "dram" | "l1"
    scheme: str  # "interleaved" | "height_sharded" | "width_sharded" | "block_sharded"
    ranges: tuple = ()  # ((start_x, start_y, end_x, end_y), ...) in the CoreRangeSet's own order
    shard_shape: tuple = ()  # (height, width) in elements
    orientation: Optional[object] = None

    @property
    def is_sharded(self):
        return self.scheme != "interleaved"

    @property
    def is_l1_sharded(self):
        return self.buffer == "l1" and self.is_sharded

    @property
    def is_l1_interleaved(self):
        return self.buffer == "l1" and not self.is_sharded

    def key(self):
        return (self.buffer, self.scheme, self.ranges, self.shard_shape, str(self.orientation))


def placement_of(memory_config):
    buffer = _BUFFER_NAME.get(memory_config.buffer_type)
    scheme = _SCHEME_NAME.get(memory_config.memory_layout)
    if buffer is None:
        raise ValueError(f"g6_to_memory_config: buffer type {memory_config.buffer_type} is not DRAM or L1")
    if scheme is None:
        raise ValueError(f"g6_to_memory_config: memory layout {memory_config.memory_layout} is not supported")
    if scheme == "interleaved":
        return Placement(buffer, scheme)
    spec = memory_config.shard_spec
    if spec is None:
        raise ValueError(f"g6_to_memory_config: {scheme} memory config carries no ShardSpec")
    ranges = tuple((r.start.x, r.start.y, r.end.x, r.end.y) for r in spec.grid.ranges())
    return Placement(buffer, scheme, ranges, tuple(int(d) for d in spec.shape), spec.orientation)


@dataclass(frozen=True)
class TileGeometry:
    """2D page view every placement shares: leading dims fold into tile rows, page p = r * tile_cols + c."""

    tile_h: int
    tile_w: int
    tile_rows: int
    tile_cols: int

    @property
    def pages(self):
        return self.tile_rows * self.tile_cols


def tile_geometry(tensor):
    padded = [int(d) for d in tensor.padded_shape]
    tile_h, tile_w = (int(d) for d in tensor.tile.tile_shape)
    return TileGeometry(tile_h, tile_w, math.prod(padded[:-2]) * (padded[-2] // tile_h), padded[-1] // tile_w)


@dataclass(frozen=True)
class ShardGrid:
    shard_tile_rows: int
    shard_tile_cols: int
    shards_down: int
    shards_across: int

    @property
    def num_shards(self):
        return self.shards_down * self.shards_across


def shard_grid(placement, geom):
    shard_rows = placement.shard_shape[0] // geom.tile_h
    shard_cols = placement.shard_shape[1] // geom.tile_w
    return ShardGrid(
        shard_rows, shard_cols, math.ceil(geom.tile_rows / shard_rows), math.ceil(geom.tile_cols / shard_cols)
    )


def _range_cores(placement):
    """Cores of the shard grid, range by range in the order the CoreRangeSet holds them, row-wise in each."""
    return [(x, y) for sx, sy, ex, ey in placement.ranges for y in range(sy, ey + 1) for x in range(sx, ex + 1)]


def check_shard_spec(placement, geom, what):
    """Refuse (ValueError) a shard spec whose shards do not cover the tile grid one shard per core."""
    if placement.orientation != ttnn.ShardOrientation.ROW_MAJOR:
        raise ValueError(f"g6_to_memory_config: {what} shard spec must be ROW_MAJOR, got {placement.orientation}")
    height, width = placement.shard_shape
    if height <= 0 or width <= 0 or height % geom.tile_h or width % geom.tile_w:
        raise ValueError(
            f"g6_to_memory_config: {what} shard shape {placement.shard_shape} is not a multiple of the "
            f"{geom.tile_h}x{geom.tile_w} tile"
        )
    grid = shard_grid(placement, geom)
    if placement.scheme == "block_sharded":
        if len(placement.ranges) != 1:
            raise ValueError(f"g6_to_memory_config: {what} block-sharded grid must be one core range")
        sx, sy, ex, ey = placement.ranges[0]
        if grid.shards_across > ex - sx + 1 or grid.shards_down > ey - sy + 1:
            raise ValueError(
                f"g6_to_memory_config: {what} block shard grid {grid.shards_down}x{grid.shards_across} does not "
                f"fit the core range {placement.ranges[0]}"
            )
    elif grid.num_shards > len(_range_cores(placement)):
        raise ValueError(
            f"g6_to_memory_config: {what} needs {grid.num_shards} shards but its grid has "
            f"{len(_range_cores(placement))} cores"
        )


@dataclass(frozen=True)
class ShardOwner:
    core: tuple  # (x, y) logical
    r0: int
    c0: int
    rows: int
    cols: int

    @property
    def pages(self):
        return self.rows * self.cols


def shard_owners(placement, geom):
    """Shard s (row-major over the shard grid) -> core, as the buffer distribution spec places it:
    ROUND_ROBIN_1D (height/width) takes the CoreRangeSet's cores in order, GRID_2D (block) the
    row-major cores of its single range trimmed to the shard grid. Cores without a shard are omitted."""
    grid = shard_grid(placement, geom)
    if placement.scheme == "block_sharded":
        sx, sy = placement.ranges[0][:2]
        cores = [(sx + j, sy + i) for i in range(grid.shards_down) for j in range(grid.shards_across)]
    else:
        cores = _range_cores(placement)[: grid.num_shards]
    owners = []
    for s, core in enumerate(cores):
        i, j = divmod(s, grid.shards_across)
        r0, c0 = i * grid.shard_tile_rows, j * grid.shard_tile_cols
        owners.append(
            ShardOwner(
                core,
                r0,
                c0,
                min(grid.shard_tile_rows, geom.tile_rows - r0),
                min(grid.shard_tile_cols, geom.tile_cols - c0),
            )
        )
    return owners


def mover_slice(num_pages, mover):
    """Contiguous slice [begin, end) of a core's page sequence taken by one mover."""
    half = math.ceil(num_pages / MOVERS_PER_CORE)
    begin = min(num_pages, mover * half)
    return begin, min(num_pages, begin + half)


def coalesce(cores):
    """Cover a core set with few maximal rectangles (launch messages go out once per range, BB-17)."""
    remaining = set(cores)
    ranges = []
    for x, y in sorted(remaining, key=lambda c: (c[1], c[0])):
        if (x, y) not in remaining:
            continue
        x_end = x
        while (x_end + 1, y) in remaining:
            x_end += 1
        y_end = y
        while all((xx, y_end + 1) in remaining for xx in range(x, x_end + 1)):
            y_end += 1
        for yy in range(y, y_end + 1):
            for xx in range(x, x_end + 1):
                remaining.discard((xx, yy))
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x_end, y_end)))
    return ttnn.CoreRangeSet(ranges)


# ---------------------------------------------------------------------------
# Regime selection and work split
# ---------------------------------------------------------------------------


def select_regime(src, dst, geom):
    if dst.is_l1_sharded and src.is_l1_sharded:
        # Equal bytes either way; the side with more shards keeps more cores busy, ties pull.
        if shard_grid(dst, geom).num_shards >= shard_grid(src, geom).num_shards:
            return SHARD_PULL
        return SHARD_PUSH
    if dst.is_l1_sharded:
        return SHARD_PULL
    if src.is_l1_sharded:
        return SHARD_PUSH
    if dst.is_l1_interleaved:
        return BANK_PULL
    if src.is_l1_interleaved:
        return BANK_PUSH
    return STAGED


class Plan(NamedTuple):
    regime: str
    owner_side: Optional[str]  # "output" (pull), "input" (push), None (staged)
    num_cores: int
    movers_per_core: int


@dataclass
class Work:
    regime: str
    geom: TileGeometry
    src: Placement
    dst: Placement
    page_size: int
    owners: list  # shard_*: ShardOwner per owner core
    grid_cores: list  # bank_*: every worker core
    staged: list  # staged: (core, start_page, num_pages)
    staged_block_pages: int
    num_l1_banks: int
    core_ranges: object

    @property
    def owner_placement(self):
        return self.dst if self.regime in PULL_REGIMES else self.src

    def plan(self):
        if self.regime in SHARD_REGIMES:
            num_cores = len(self.owners)
        elif self.regime in BANK_REGIMES:
            num_cores = len(self.grid_cores)
        else:
            num_cores = len(self.staged)
        owner_side = None if self.regime == STAGED else ("output" if self.regime in PULL_REGIMES else "input")
        movers = 1 if self.regime == STAGED else MOVERS_PER_CORE
        return Plan(self.regime, owner_side, num_cores, movers)


_L1_BANKS = {}


def _device_key(device):
    return device.id() if hasattr(device, "id") else id(device)


def l1_num_banks(device):
    key = _device_key(device)
    if key not in _L1_BANKS:
        _L1_BANKS[key] = int(ttnn._ttnn.reports.get_device_info(device).l1_num_banks)
    return _L1_BANKS[key]


def plan_work(input_tensor, src, dst):
    device = input_tensor.device()
    geom = tile_geometry(input_tensor)
    regime = select_regime(src, dst, geom)
    grid = device.compute_with_storage_grid_size()
    num_banks = l1_num_banks(device)
    work = Work(
        regime=regime,
        geom=geom,
        src=src,
        dst=dst,
        page_size=int(input_tensor.buffer_aligned_page_size()),
        owners=[],
        grid_cores=[],
        staged=[],
        staged_block_pages=0,
        num_l1_banks=num_banks,
        core_ranges=None,
    )
    if regime in SHARD_REGIMES:
        work.owners = shard_owners(work.owner_placement, geom)
        work.core_ranges = coalesce([o.core for o in work.owners])
    elif regime in BANK_REGIMES:
        # Every worker core owns one L1 bank; which one is discovered on device, so all of them launch.
        if num_banks != grid.x * grid.y:
            raise RuntimeError(
                f"g6_to_memory_config: {num_banks} L1 banks on a {grid.x}x{grid.y} worker grid; the bank "
                "regimes need one bank per worker core"
            )
        work.grid_cores = [(x, y) for y in range(grid.y) for x in range(grid.x)]
        work.core_ranges = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
        )
    else:
        _, all_cores, group_1, group_2, pages_1, pages_2 = ttnn.split_work_to_cores(grid, geom.pages, row_wise=True)
        start = 0
        for group, count in ((group_1, pages_1), (group_2, pages_2)):
            for core in ttnn.corerange_to_cores(group, row_wise=True):
                work.staged.append(((core.x, core.y), start, count))
                start += count
        assert start == geom.pages, (start, geom.pages)
        work.staged_block_pages = min(STAGED_BLOCK_PAGES, max(pages_1, pages_2))
        work.core_ranges = all_cores
    return work


def op_l1_bytes_per_core(work):
    """Output L1 on the busiest core plus the op's own buffers (l1_ledger.md, BB-05)."""
    total = 0
    if work.dst.is_l1_sharded:
        grid = shard_grid(work.dst, work.geom)
        total += grid.shard_tile_rows * grid.shard_tile_cols * work.page_size
    elif work.dst.is_l1_interleaved:
        total += math.ceil(work.geom.pages / work.num_l1_banks) * work.page_size
    if work.regime == STAGED:
        total += STAGED_BUFFER_DEPTH * work.staged_block_pages * work.page_size
    return total


def check_l1_budget(work):
    used = op_l1_bytes_per_core(work)
    if used > L1_BUDGET_BYTES:
        raise ValueError(
            f"g6_to_memory_config: needs {used} B of L1 per core (output + op buffers), over the "
            f"{L1_BUDGET_BYTES} B the model leaves to the op"
        )


def select_plan(input_tensor, memory_config):
    """The plan a call with these arguments runs (BB-08)."""
    return plan_work(input_tensor, placement_of(input_tensor.memory_config()), placement_of(memory_config)).plan()


def dump_core_roles(input_tensor, memory_config):
    """[(core_xy, role, pages)] of every launched kernel (BB-21). Bank movers learn their bank on device,
    so their page count is reported as None."""
    work = plan_work(input_tensor, placement_of(input_tensor.memory_config()), placement_of(memory_config))
    roles = []
    if work.regime in SHARD_REGIMES:
        for owner in work.owners:
            for mover in range(MOVERS_PER_CORE):
                begin, end = mover_slice(owner.pages, mover)
                roles.append((owner.core, f"{work.regime}/mover{mover}", end - begin))
    elif work.regime in BANK_REGIMES:
        for core in work.grid_cores:
            for mover in range(MOVERS_PER_CORE):
                roles.append((core, f"{work.regime}/mover{mover}", None))
    else:
        for core, _, count in work.staged:
            roles.append((core, "staged/reader", count))
            roles.append((core, "staged/writer", count))
    return roles


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------


def _accessor_args(tensor):
    return list(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())


def _owner_descriptor(work, input_tensor, output_tensor, common_args):
    is_pull = work.regime in PULL_REGIMES
    owner_is_bank = work.regime in BANK_REGIMES
    shard_tile_cols = 0 if owner_is_bank else shard_grid(work.owner_placement, work.geom).shard_tile_cols
    accessors = _accessor_args(input_tensor) + _accessor_args(output_tensor)
    kernels = []
    for mover in range(MOVERS_PER_CORE):
        processor, noc = _MOVER_CONFIGS[mover]
        compile_args = [
            int(is_pull),
            int(owner_is_bank),
            mover,
            MOVERS_PER_CORE,
            work.page_size,
            work.geom.pages,
            work.geom.tile_cols,
            shard_tile_cols,
        ] + accessors
        runtime_args = ttnn.RuntimeArgs()
        for owner in work.owners:
            begin, end = mover_slice(owner.pages, mover)
            runtime_args[owner.core[0]][owner.core[1]] = [owner.r0, owner.c0, owner.cols, begin, end]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(MOVER_KERNEL),
                core_ranges=work.core_ranges,
                compile_time_args=compile_args,
                runtime_args=runtime_args,
                common_runtime_args=list(common_args),
                config=ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc),
            )
        )
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[])


def _staged_descriptor(work, input_tensor, output_tensor, common_args):
    block = work.staged_block_pages
    cb = ttnn.CBDescriptor(
        total_size=STAGED_BUFFER_DEPTH * block * work.page_size,
        core_ranges=work.core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_STAGED_PAGES, data_format=input_tensor.dtype, page_size=work.page_size
            )
        ],
    )
    runtime_args = ttnn.RuntimeArgs()
    for core, start, count in work.staged:
        runtime_args[core[0]][core[1]] = [start, count]
    head = [CB_STAGED_PAGES, block, work.page_size]
    reader = ttnn.KernelDescriptor(
        kernel_source=str(STAGED_READER_KERNEL),
        core_ranges=work.core_ranges,
        compile_time_args=head + _accessor_args(input_tensor),
        runtime_args=runtime_args,
        common_runtime_args=list(common_args),
        config=ttnn.DataMovementConfigDescriptor(processor=_MOVER_CONFIGS[0][0], noc=_MOVER_CONFIGS[0][1]),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(STAGED_WRITER_KERNEL),
        core_ranges=work.core_ranges,
        compile_time_args=head + _accessor_args(output_tensor),
        runtime_args=runtime_args,
        common_runtime_args=list(common_args),
        config=ttnn.DataMovementConfigDescriptor(processor=_MOVER_CONFIGS[1][0], noc=_MOVER_CONFIGS[1][1]),
    )
    return ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=[cb])


def create_program_descriptor(work, input_tensor, output_tensor):
    if int(output_tensor.buffer_aligned_page_size()) != work.page_size:
        raise RuntimeError("g6_to_memory_config: input and output page sizes differ")
    common_args = [input_tensor.buffer_address(), output_tensor.buffer_address()]
    if work.regime == STAGED:
        return _staged_descriptor(work, input_tensor, output_tensor, common_args)
    return _owner_descriptor(work, input_tensor, output_tensor, common_args)


# ---------------------------------------------------------------------------
# Dispatch with a descriptor cache (BB-07)
# ---------------------------------------------------------------------------

_DESCRIPTORS = {}


def _cache_key(input_tensor, src, dst):
    return (
        _device_key(input_tensor.device()),
        tuple(int(d) for d in input_tensor.shape),
        tuple(int(d) for d in input_tensor.padded_shape),
        str(input_tensor.dtype),
        src.key(),
        dst.key(),
        MOVERS_PER_CORE,
        STAGED_BLOCK_PAGES,
        STAGED_BUFFER_DEPTH,
    )


def _program_hash(key):
    digest = hashlib.sha256(("g6_to_memory_config" + repr(key)).encode()).digest()
    return int.from_bytes(digest[:8], "little") >> 1


def dispatch(input_tensor, memory_config):
    """Allocate the output in memory_config and run the one program that moves every page into it.
    A repeat configuration reuses its descriptor; only the two buffer addresses are rewritten."""
    src = placement_of(input_tensor.memory_config())
    dst = placement_of(memory_config)
    key = _cache_key(input_tensor, src, dst)
    descriptor = _DESCRIPTORS.get(key)
    work = None
    if descriptor is None:
        work = plan_work(input_tensor, src, dst)
        check_l1_budget(work)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        memory_config,
    )
    if descriptor is None:
        descriptor = create_program_descriptor(work, input_tensor, output_tensor)
        descriptor.custom_program_hash = _program_hash(key)
        _DESCRIPTORS[key] = descriptor
    else:
        src_addr, dst_addr = input_tensor.buffer_address(), output_tensor.buffer_address()
        for kernel in descriptor.kernels:
            kernel.common_runtime_args[0] = src_addr
            kernel.common_runtime_args[1] = dst_addr
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
