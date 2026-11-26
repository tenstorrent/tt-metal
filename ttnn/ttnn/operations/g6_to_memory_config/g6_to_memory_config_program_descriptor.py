# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_to_memory_config: plan selection and ProgramDescriptor.

Every page is moved by a core that holds it in its own L1: the target's owner pulls it, or the
source's owner pushes it (1 NoC crossing, no CB). Only DRAM -> DRAM needs a staging buffer
(`staged`), because a NoC transaction cannot go DRAM -> DRAM.

Regimes: shard_pull, shard_push, bank_pull, bank_push, staged (op_design.md, Blocking Model).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
MOVER_KERNEL = str(KERNEL_DIR / "g6_to_memory_config_mover.cpp")
STAGED_READER_KERNEL = str(KERNEL_DIR / "g6_to_memory_config_reader.cpp")
STAGED_WRITER_KERNEL = str(KERNEL_DIR / "g6_to_memory_config_writer.cpp")

# Block-model knobs (op_design.md -> Blocking Model). Each is defined once here; everything that
# depends on it (kernel count, RT slices, CB size, cache key) derives from it.
MOVERS_PER_CORE = 2  # owner regimes: data-movement RISCs sharing one core's page sequence (1 or 2)
STAGED_BLOCK_PAGES = 16  # staged: pages per CB handshake / read barrier
STAGED_BUFFER_DEPTH = 2  # staged: blocks resident in cb_staged_pages

L1_BUDGET_BYTES = 1347776  # BB-05: L1 per core the model leaves to the op

CB_STAGED_PAGES = 0

# mover m runs on (processor, NoC); m = 0 is NCRISC/NoC0, m = 1 is BRISC/NoC1.
_MOVER_PLACEMENT = (
    (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
    (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
)
assert 1 <= MOVERS_PER_CORE <= len(_MOVER_PLACEMENT)

REGIMES = ("shard_pull", "shard_push", "bank_pull", "bank_push", "staged")

_BUFFER_NAME = {ttnn.BufferType.DRAM: "dram", ttnn.BufferType.L1: "l1"}
_SCHEME_NAME = {
    ttnn.TensorMemoryLayout.INTERLEAVED: "interleaved",
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: "height_sharded",
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: "width_sharded",
    ttnn.TensorMemoryLayout.BLOCK_SHARDED: "block_sharded",
}


class Plan(NamedTuple):
    """The plan a call selects (BB-08). owner_side: "target" (pull), "source" (push), "none" (staged)."""

    regime: str
    owner_side: str
    num_cores: int
    movers_per_core: int


@dataclass(frozen=True)
class Geometry:
    """2D page view every placement uses: leading dims fold into tile rows, page p = r * cols + c."""

    tile_h: int
    tile_w: int
    tensor_tile_rows: int
    tensor_tile_cols: int
    page_size: int

    @property
    def tensor_pages(self) -> int:
        return self.tensor_tile_rows * self.tensor_tile_cols


@dataclass(frozen=True)
class Placement:
    buffer: str  # "dram" | "l1"
    scheme: str  # "interleaved" | "height_sharded" | "width_sharded" | "block_sharded"
    ranges: tuple = ()  # shard grid ranges (sx, sy, ex, ey), in CoreRangeSet order
    shard_shape: tuple = ()  # shard shape in elements (h, w)
    orientation: Optional[str] = None

    @property
    def is_sharded(self) -> bool:
        return self.scheme != "interleaved"

    @property
    def is_l1_sharded(self) -> bool:
        return self.buffer == "l1" and self.is_sharded

    @property
    def is_l1_interleaved(self) -> bool:
        return self.buffer == "l1" and not self.is_sharded


@dataclass(frozen=True)
class OwnerShard:
    """One owner core and the rectangle of tile pages its shard holds (clipped at the tensor edge)."""

    core: tuple  # (x, y) logical
    r0: int
    c0: int
    core_rows: int
    core_cols: int


def geometry_of(input_tensor) -> Geometry:
    shape = list(input_tensor.shape)
    tile_h, tile_w = (int(v) for v in input_tensor.tile.tile_shape)
    leading = math.prod(shape[:-2]) if len(shape) > 2 else 1
    return Geometry(
        tile_h=tile_h,
        tile_w=tile_w,
        tensor_tile_rows=leading * math.ceil(shape[-2] / tile_h),
        tensor_tile_cols=math.ceil(shape[-1] / tile_w),
        page_size=int(input_tensor.buffer_aligned_page_size()),
    )


def placement_of(memory_config) -> Placement:
    buffer = _BUFFER_NAME.get(memory_config.buffer_type)
    scheme = _SCHEME_NAME.get(memory_config.memory_layout)
    if buffer is None or scheme is None:
        raise ValueError(
            f"g6_to_memory_config: unsupported placement {memory_config.buffer_type}/{memory_config.memory_layout}"
        )
    if scheme == "interleaved":
        return Placement(buffer=buffer, scheme=scheme)
    spec = memory_config.shard_spec
    if spec is None:
        raise ValueError(f"g6_to_memory_config: {scheme} memory_config has no shard spec")
    ranges = tuple((r.start.x, r.start.y, r.end.x, r.end.y) for r in spec.grid.ranges())
    orientation = "row_major" if spec.orientation == ttnn.ShardOrientation.ROW_MAJOR else "col_major"
    return Placement(
        buffer=buffer,
        scheme=scheme,
        ranges=ranges,
        shard_shape=tuple(int(v) for v in spec.shape),
        orientation=orientation,
    )


def shard_tiles(placement: Placement, geom: Geometry) -> tuple:
    """Shard extent in tile pages (rows, cols); shard shapes are validated to be tile multiples."""
    return placement.shard_shape[0] // geom.tile_h, placement.shard_shape[1] // geom.tile_w


def shard_grid(placement: Placement, geom: Geometry) -> tuple:
    """(shards along tile rows, shards along tile cols)."""
    shard_rows, shard_cols = shard_tiles(placement, geom)
    return math.ceil(geom.tensor_tile_rows / shard_rows), math.ceil(geom.tensor_tile_cols / shard_cols)


def num_shards(placement: Placement, geom: Geometry) -> int:
    rows, cols = shard_grid(placement, geom)
    return rows * cols


def _num_grid_cores(placement: Placement) -> int:
    return sum((ex - sx + 1) * (ey - sy + 1) for sx, sy, ex, ey in placement.ranges)


def shard_layout_error(placement: Placement, geom: Geometry) -> Optional[str]:
    """Why the shard spec cannot hold the tensor as one shard per core, or None when it can."""
    shard_h, shard_w = placement.shard_shape
    if shard_h == 0 or shard_w == 0 or shard_h % geom.tile_h or shard_w % geom.tile_w:
        return f"shard shape {list(placement.shard_shape)} is not a multiple of the {geom.tile_h}x{geom.tile_w} tile"
    if placement.orientation != "row_major":
        return "shard orientation must be ROW_MAJOR"
    if placement.scheme == "block_sharded":
        if len(placement.ranges) != 1:
            return "a BLOCK_SHARDED grid must be one core range"
        sx, sy, ex, ey = placement.ranges[0]
        rows, cols = shard_grid(placement, geom)
        if cols > ex - sx + 1 or rows > ey - sy + 1:
            return f"{rows}x{cols} block shards do not fit the {ey - sy + 1}x{ex - sx + 1} core range"
        return None
    if num_shards(placement, geom) > _num_grid_cores(placement):
        return f"{num_shards(placement, geom)} shards need as many cores; the grid has {_num_grid_cores(placement)}"
    return None


def shard_owner_cores(placement: Placement, geom: Geometry) -> list:
    """Shard s (row-major over the shard grid) -> logical core, as BufferDistributionSpec places it.

    HEIGHT/WIDTH (ROUND_ROBIN_1D): the grid's cores row-wise, range by range in CoreRangeSet order;
    one shard per core, so shard s sits on cores[s]. BLOCK (GRID_2D): row-major cores of the single
    range trimmed to the shard grid.
    """
    rows, cols = shard_grid(placement, geom)
    if placement.scheme == "block_sharded":
        sx, sy, _, _ = placement.ranges[0]
        return [(sx + j, sy + i) for i in range(rows) for j in range(cols)]
    cores = []
    for sx, sy, ex, ey in placement.ranges:
        cores.extend((x, y) for y in range(sy, ey + 1) for x in range(sx, ex + 1))
    return cores[: rows * cols]


def owner_shards(placement: Placement, geom: Geometry) -> list:
    shard_rows, shard_cols = shard_tiles(placement, geom)
    _, grid_cols = shard_grid(placement, geom)
    shards = []
    for s, core in enumerate(shard_owner_cores(placement, geom)):
        r0 = (s // grid_cols) * shard_rows
        c0 = (s % grid_cols) * shard_cols
        shards.append(
            OwnerShard(
                core=core,
                r0=r0,
                c0=c0,
                core_rows=min(shard_rows, geom.tensor_tile_rows - r0),
                core_cols=min(shard_cols, geom.tensor_tile_cols - c0),
            )
        )
    return shards


def mover_slices(num_pages: int, movers_per_core: int = MOVERS_PER_CORE) -> list:
    """Contiguous slices [begin, end) of one core's page sequence, one per mover."""
    half = math.ceil(num_pages / movers_per_core)
    return [(min(num_pages, m * half), min(num_pages, (m + 1) * half)) for m in range(movers_per_core)]


def select_regime(src: Placement, dst: Placement, geom: Geometry) -> str:
    """Exact regime-selection function of the design; depends on the placements only."""
    if dst.is_l1_sharded and src.is_l1_sharded:
        return "shard_pull" if num_shards(dst, geom) >= num_shards(src, geom) else "shard_push"
    if dst.is_l1_sharded:
        return "shard_pull"
    if src.is_l1_sharded:
        return "shard_push"
    if dst.is_l1_interleaved:
        return "bank_pull"
    if src.is_l1_interleaved:
        return "bank_push"
    return "staged"


# ---------------------------------------------------------------------------
# Per-call plan (everything that shapes the program; cached by placement key)
# ---------------------------------------------------------------------------


@dataclass
class _Build:
    plan: Plan
    geom: Geometry
    src: Placement
    dst: Placement
    owners: list  # OwnerShard list (shard_*)
    grid_size: tuple  # compute_with_storage_grid (x, y)
    staged_split: Optional[tuple] = None  # split_work_to_cores result (staged)
    staged_block_pages: int = 0
    descriptor: Optional[ttnn.ProgramDescriptor] = None


_BUILD_CACHE = {}
_BANK_CHECKED_DEVICES = set()


def _cache_key(input_tensor, memory_config) -> tuple:
    device = input_tensor.device()
    return (
        device.id(),
        tuple(int(v) for v in input_tensor.shape),
        input_tensor.dtype,
        tuple(int(v) for v in input_tensor.tile.tile_shape),
        placement_of(input_tensor.memory_config()),
        placement_of(memory_config),
        MOVERS_PER_CORE,
        STAGED_BLOCK_PAGES,
        STAGED_BUFFER_DEPTH,
    )


def _check_one_l1_bank_per_core(device, grid_size) -> None:
    """bank_* launches on the full worker grid and each core discovers its own L1 bank."""
    if device.id() in _BANK_CHECKED_DEVICES:
        return
    l1_banks = ttnn._ttnn.reports.get_device_info(device).l1_num_banks
    if l1_banks != grid_size[0] * grid_size[1]:
        raise RuntimeError(
            f"g6_to_memory_config: bank regimes need one L1 bank per worker core; "
            f"{l1_banks} banks on a {grid_size[0]}x{grid_size[1]} grid"
        )
    _BANK_CHECKED_DEVICES.add(device.id())


def _build(input_tensor, memory_config) -> _Build:
    key = _cache_key(input_tensor, memory_config)
    cached = _BUILD_CACHE.get(key)
    if cached is not None:
        return cached

    geom = geometry_of(input_tensor)
    src = placement_of(input_tensor.memory_config())
    dst = placement_of(memory_config)
    regime = select_regime(src, dst, geom)
    grid = input_tensor.device().compute_with_storage_grid_size()
    grid_size = (grid.x, grid.y)

    owners = []
    staged_split = None
    staged_block_pages = 0
    if regime in ("shard_pull", "shard_push"):
        owners = owner_shards(dst if regime == "shard_pull" else src, geom)
        plan = Plan(regime, "target" if regime == "shard_pull" else "source", len(owners), MOVERS_PER_CORE)
    elif regime in ("bank_pull", "bank_push"):
        _check_one_l1_bank_per_core(input_tensor.device(), grid_size)
        plan = Plan(
            regime, "target" if regime == "bank_pull" else "source", grid_size[0] * grid_size[1], MOVERS_PER_CORE
        )
    else:
        staged_split = ttnn.split_work_to_cores(grid, geom.tensor_pages, row_wise=True)
        staged_block_pages = min(STAGED_BLOCK_PAGES, staged_split[4])
        plan = Plan(regime, "none", staged_split[0], 1)

    build = _Build(
        plan=plan,
        geom=geom,
        src=src,
        dst=dst,
        owners=owners,
        grid_size=grid_size,
        staged_split=staged_split,
        staged_block_pages=staged_block_pages,
    )
    _BUILD_CACHE[key] = build
    return build


def select_plan(input_tensor, memory_config) -> Plan:
    return _build(input_tensor, memory_config).plan


def op_l1_bytes_per_core(input_tensor, memory_config) -> int:
    """BB-05 footprint (l1_ledger.md): the output's own L1 on a core plus the op's CBs."""
    build = _build(input_tensor, memory_config)
    geom, dst = build.geom, build.dst
    total = 0
    if dst.is_l1_sharded:
        shard_rows, shard_cols = shard_tiles(dst, geom)
        total += shard_rows * shard_cols * geom.page_size
    elif dst.is_l1_interleaved:
        total += math.ceil(geom.tensor_pages / (build.grid_size[0] * build.grid_size[1])) * geom.page_size
    if build.plan.regime == "staged":
        total += STAGED_BUFFER_DEPTH * build.staged_block_pages * geom.page_size
    return total


def dump_core_roles(input_tensor, memory_config) -> list:
    """BB-21: [(core_xy, role, pages)] for every launched core and role.

    bank_* pages are None: a core's L1 bank, and so its page count, is found on device.
    """
    build = _build(input_tensor, memory_config)
    regime = build.plan.regime
    roles = []
    if regime in ("shard_pull", "shard_push"):
        for owner in build.owners:
            for m, (begin, end) in enumerate(mover_slices(owner.core_rows * owner.core_cols)):
                roles.append((owner.core, f"{regime}/mover{m}", end - begin))
    elif regime in ("bank_pull", "bank_push"):
        for y in range(build.grid_size[1]):
            for x in range(build.grid_size[0]):
                roles.extend(((x, y), f"{regime}/mover{m}", None) for m in range(MOVERS_PER_CORE))
    else:
        for core, _, num_pages in _staged_assignment(build):
            roles.append(((core.x, core.y), "staged/reader", num_pages))
            roles.append(((core.x, core.y), "staged/writer", num_pages))
    return roles


# ---------------------------------------------------------------------------
# ProgramDescriptor
# ---------------------------------------------------------------------------


def _core_range_set(cores) -> ttnn.CoreRangeSet:
    """Cores coalesced into maximal rectangles (BB-17)."""
    singles = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])
    return ttnn.CoreRangeSet([]).merge(singles)


def _staged_assignment(build: _Build) -> list:
    """[(core, start_page, num_pages)]: contiguous page ranges in split_work_to_cores order."""
    _, _, group_1, group_2, per_core_1, per_core_2 = build.staged_split
    assignment = []
    start = 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, row_wise=True):
            assignment.append((core, start, per_core))
            start += per_core
    assert start == build.geom.tensor_pages
    return assignment


def _dm_config(processor, noc):
    return ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc)


def _mover_kernels(build: _Build, input_tensor, output_tensor) -> list:
    regime = build.plan.regime
    is_pull = regime in ("shard_pull", "bank_pull")
    is_bank = regime in ("bank_pull", "bank_push")
    geom = build.geom
    owner_placement = build.dst if is_pull else build.src
    owner_shard_tile_cols = 0 if is_bank else shard_tiles(owner_placement, geom)[1]

    accessor_args = list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    accessor_args += list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    if is_bank:
        core_ranges = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(build.grid_size[0] - 1, build.grid_size[1] - 1))]
        )
    else:
        core_ranges = _core_range_set([owner.core for owner in build.owners])

    kernels = []
    for m in range(MOVERS_PER_CORE):
        ct_args = [
            m,
            MOVERS_PER_CORE,
            int(is_pull),
            int(is_bank),
            owner_shard_tile_cols,
            geom.tensor_tile_cols,
            geom.tensor_pages,
        ] + accessor_args
        rt_args = ttnn.RuntimeArgs()
        if not is_bank:
            for owner in build.owners:
                begin, end = mover_slices(owner.core_rows * owner.core_cols)[m]
                rt_args[owner.core[0]][owner.core[1]] = [owner.r0, owner.c0, owner.core_cols, begin, end]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=MOVER_KERNEL,
                core_ranges=core_ranges,
                compile_time_args=ct_args,
                runtime_args=rt_args,
                common_runtime_args=[0, 0],
                config=_dm_config(*_MOVER_PLACEMENT[m]),
            )
        )
    return kernels, []


def _staged_kernels(build: _Build, input_tensor, output_tensor):
    all_cores = build.staged_split[1]
    block = build.staged_block_pages
    reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for core, start, num_pages in _staged_assignment(build):
        reader_rt[core.x][core.y] = [start, num_pages]
        writer_rt[core.x][core.y] = [start, num_pages]

    reader = ttnn.KernelDescriptor(
        kernel_source=STAGED_READER_KERNEL,
        core_ranges=all_cores,
        compile_time_args=[CB_STAGED_PAGES, block]
        + list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()),
        runtime_args=reader_rt,
        common_runtime_args=[0, 0],
        config=_dm_config(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=STAGED_WRITER_KERNEL,
        core_ranges=all_cores,
        compile_time_args=[CB_STAGED_PAGES, block]
        + list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()),
        runtime_args=writer_rt,
        common_runtime_args=[0, 0],
        config=_dm_config(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
    )
    page_size = build.geom.page_size
    cb_staged_pages = ttnn.CBDescriptor(
        total_size=STAGED_BUFFER_DEPTH * block * page_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_STAGED_PAGES, data_format=input_tensor.dtype, page_size=page_size)
        ],
    )
    return [reader, writer], [cb_staged_pages]


def _program_hash(key: tuple) -> int:
    digest = hashlib.blake2b(repr(("g6_to_memory_config", key)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def create_program_descriptor(input_tensor, output_tensor, memory_config) -> ttnn.ProgramDescriptor:
    """The cached descriptor for this configuration, with this call's buffer addresses patched in.

    Buffer addresses are the only values that change between calls with the same configuration
    (BB-07); they ride in each kernel's common RT args [src_base_addr, dst_base_addr].
    """
    build = _build(input_tensor, memory_config)
    if build.descriptor is None:
        if build.plan.regime == "staged":
            kernels, cbs = _staged_kernels(build, input_tensor, output_tensor)
        else:
            kernels, cbs = _mover_kernels(build, input_tensor, output_tensor)
        descriptor = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        descriptor.custom_program_hash = _program_hash(_cache_key(input_tensor, memory_config))
        build.descriptor = descriptor

    addresses = [input_tensor.buffer_address(), output_tensor.buffer_address()]
    for kernel in build.descriptor.kernels:
        kernel.common_runtime_args = addresses
    return build.descriptor
