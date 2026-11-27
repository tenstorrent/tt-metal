# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Plan selection and ProgramDescriptor for g6_to_memory_config.

Regimes (op_design.md, Blocking Model):
  pull   (R1) every core holding target pages reads them from the source into its own target slots
  push   (R2) every core holding source pages writes them to their target slots
  bounce (R3) neither side is held by a core: grid workers stage target-order chunks in an L1 CB

A plan is a pure function of the placement pair and the device; it and the descriptor built from it are
cached, and a repeat call only patches the two buffer base addresses (common runtime args).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
GATHER_KERNEL = str(KERNEL_DIR / "g6_to_memory_config_reader.cpp")
SCATTER_KERNEL = str(KERNEL_DIR / "g6_to_memory_config_writer.cpp")

# Knobs (single source; every dependent quantity is derived from these).
MODEL_L1_BUDGET_BYTES = 1347776  # L1 the model leaves to the op per core (BB-05)
MAX_TRANSFER_BYTES = 1 << 20  # noc_async_read/write size limit; caps one segment
BOUNCE_DEPTH = 2  # staging slots per bounce worker
OWNER_BLOCK_PAGES: Optional[int] = None  # pull/push positions per block; None = the whole owned set

CB_STAGING_PAGES = 0

LOCAL_OWNED_SHARD = 0
LOCAL_OWNED_BANKS = 1
LOCAL_STAGING = 2

_BUFFER_NAME = {ttnn.BufferType.DRAM: "dram", ttnn.BufferType.L1: "l1"}
_SCHEME_NAME = {
    ttnn.TensorMemoryLayout.INTERLEAVED: "interleaved",
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: "height_sharded",
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: "width_sharded",
    ttnn.TensorMemoryLayout.BLOCK_SHARDED: "block_sharded",
}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def tile_grid(shape):
    """(tensor_rows_tiles, tensor_cols_tiles). Each image is tile-padded on its own."""
    shape = list(shape)
    rows = math.prod(shape[:-2]) * math.ceil(shape[-2] / 32)
    cols = math.ceil(shape[-1] / 32)
    return rows, cols


def buffer_name(memory_config):
    return _BUFFER_NAME[memory_config.buffer_type]


def scheme_name(memory_config):
    return _SCHEME_NAME[memory_config.memory_layout]


def _ranges(grid):
    return tuple((r.start.x, r.start.y, r.end.x, r.end.y) for r in grid.ranges())


def memory_config_key(memory_config):
    """Every field of a placement that changes the program."""
    key = (buffer_name(memory_config), scheme_name(memory_config))
    spec = memory_config.shard_spec
    if spec is not None:
        key += (_ranges(spec.grid), tuple(spec.shape), str(spec.orientation))
    return key


@dataclass(frozen=True)
class Side:
    """One placement of the tensor's tile grid."""

    buffer: str
    scheme: str
    shard_rows: int = 0  # tiles
    shard_cols: int = 0
    num_shard_rows: int = 0
    num_shard_cols: int = 0
    shard_cores: tuple = ()  # core (x, y) of shard id k, in the buffer's own order
    grid_cores: tuple = ()  # every core of the shard grid

    @property
    def sharded(self):
        return self.scheme != "interleaved"

    @property
    def num_shards(self):
        return self.num_shard_rows * self.num_shard_cols


def describe_side(memory_config, rows, cols):
    buffer, scheme = buffer_name(memory_config), scheme_name(memory_config)
    if scheme == "interleaved":
        return Side(buffer, scheme)
    spec = memory_config.shard_spec
    shard_rows, shard_cols = spec.shape[0] // 32, spec.shape[1] // 32
    num_shard_rows, num_shard_cols = math.ceil(rows / shard_rows), math.ceil(cols / shard_cols)
    grid_cores = tuple((c.x, c.y) for c in ttnn.corerange_to_cores(spec.grid, None, True))
    if scheme == "block_sharded":
        # GRID_2D: the first range trimmed to the shard grid, row-major.
        start = spec.grid.ranges()[0].start
        shard_cores = tuple((start.x + j, start.y + i) for i in range(num_shard_rows) for j in range(num_shard_cols))
    else:
        # ROUND_ROBIN_1D over the grid's cores, range by range, row-major.
        shard_cores = grid_cores[: num_shard_rows * num_shard_cols]
    return Side(buffer, scheme, shard_rows, shard_cols, num_shard_rows, num_shard_cols, shard_cores, grid_cores)


def shard_extent(side, shard_id, rows, cols):
    """(row0, col0, real rows, real cols) of a shard; the last shard row/column may be ragged."""
    si, sj = divmod(shard_id, side.num_shard_cols)
    row0, col0 = si * side.shard_rows, sj * side.shard_cols
    return row0, col0, min(side.shard_rows, rows - row0), min(side.shard_cols, cols - col0)


def l1_bank_owners(device):
    grid = device.compute_with_storage_grid_size()
    return ttnn._ttnn.reports.get_device_info(device).l1_num_banks == grid.x * grid.y


def holders(side, tensor_pages, num_l1_banks, bank_owners):
    if side.buffer == "dram":
        return 0
    if not side.sharded:
        return min(tensor_pages, num_l1_banks) if bank_owners else 0
    return side.num_shards


def select_regime(h_in, h_out):
    if h_in == 0 and h_out == 0:
        return "bounce"
    return "pull" if h_out >= h_in else "push"


def merge_rectangles(cores):
    """Few rectangular CoreRanges covering exactly `cores` (BB-17): row runs, extended downward."""
    remaining = set(cores)
    ranges = []
    for y in sorted({c[1] for c in remaining}):
        row = sorted(x for x, cy in remaining if cy == y)
        runs, start = [], None
        for i, x in enumerate(row):
            if start is None:
                start = x
            if i + 1 == len(row) or row[i + 1] != x + 1:
                runs.append((start, x))
                start = None
        for x0, x1 in runs:
            y1 = y
            while all((x, y1 + 1) in remaining for x in range(x0, x1 + 1)):
                y1 += 1
            for yy in range(y, y1 + 1):
                for x in range(x0, x1 + 1):
                    remaining.discard((x, yy))
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(x0, y), ttnn.CoreCoord(x1, y1)))
    return ttnn.CoreRangeSet(ranges)


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class G6Plan:
    regime: str
    holders_in: int
    holders_out: int
    l1_bank_owners: bool
    kernel_cores: ttnn.CoreRangeSet
    roles: dict  # (x, y) -> "pull" | "push" | "bounce_reader+writer" | "idle"
    bounce_block_pages: Optional[int]
    bounce_depth: Optional[int]
    # Kernel parameters (not part of the BB-08 report, derived once with it).
    tensor_rows: int
    tensor_cols: int
    tensor_pages: int
    aligned_page_size: int
    local_kind: int
    walk_side: Optional[Side]
    max_segment_pages: int
    block_pages: int
    core_args: dict  # (x, y) -> [walk_start, walk_offset, num_positions]

    def staging_bytes_per_core(self):
        if self.regime != "bounce":
            return 0
        return self.bounce_depth * self.bounce_block_pages * self.aligned_page_size


def _owner_plan(regime, side, rows, cols, tensor_pages, grid):
    """R1/R2 mover set: the owned side's shard grid, or every core of the compute grid for L1 banks."""
    core_args = {}
    roles = {}
    if side.sharded:
        for core in side.grid_cores:
            core_args[core] = [0, 0, 0]
            roles[core] = "idle"
        max_owned = 0
        for shard_id, core in enumerate(side.shard_cores):
            _, _, nr, nc = shard_extent(side, shard_id, rows, cols)
            core_args[core] = [shard_id, 0, nr * nc]
            roles[core] = regime
            max_owned = max(max_owned, nr * nc)
        kernel_cores = merge_rectangles(side.grid_cores)
        local_kind = LOCAL_OWNED_SHARD
    else:
        kernel_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        for y in range(grid.y):
            for x in range(grid.x):
                core_args[(x, y)] = [0, 0, 0]
                roles[(x, y)] = regime
        max_owned = math.ceil(tensor_pages / (grid.x * grid.y))
        local_kind = LOCAL_OWNED_BANKS
    block_pages = OWNER_BLOCK_PAGES if OWNER_BLOCK_PAGES is not None else max(1, max_owned)
    return kernel_cores, roles, core_args, local_kind, block_pages


def _target_order_starts(out_side, rows, cols, chunk_starts):
    """(walk_start, walk_offset) of each target-order position in `chunk_starts`."""
    if not out_side.sharded:
        return [(start, 0) for start in chunk_starts]
    starts, shard_id, shard_first = [], 0, 0
    for position in chunk_starts:
        while True:
            _, _, nr, nc = shard_extent(out_side, shard_id, rows, cols)
            if position < shard_first + nr * nc:
                break
            shard_first += nr * nc
            shard_id += 1
        starts.append((shard_id, position - shard_first))
    return starts


def _bounce_plan(out_side, rows, cols, tensor_pages, grid, page_size):
    num_cores, all_cores, group_1, group_2, units_1, units_2 = ttnn.split_work_to_cores(grid, tensor_pages, True)
    cores = [(c.x, c.y) for c in ttnn.corerange_to_cores(group_1, None, True)]
    counts = [units_1] * len(cores)
    if units_2 > 0:
        cores_2 = [(c.x, c.y) for c in ttnn.corerange_to_cores(group_2, None, True)]
        cores += cores_2
        counts += [units_2] * len(cores_2)
    chunk_starts = [sum(counts[:i]) for i in range(len(counts))]
    starts = _target_order_starts(out_side, rows, cols, chunk_starts)
    core_args = {core: [start, offset, count] for core, (start, offset), count in zip(cores, starts, counts)}
    max_chunk_pages = max(counts)
    bounce_block_pages = min(
        math.ceil(max_chunk_pages / BOUNCE_DEPTH),
        MODEL_L1_BUDGET_BYTES // (BOUNCE_DEPTH * page_size),
    )
    roles = {core: "bounce_reader+writer" for core in cores}
    return all_cores, roles, core_args, bounce_block_pages


_PLAN_CACHE = {}
_DESCRIPTOR_CACHE = {}


def plan_key(input_tensor, memory_config):
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    return (
        device.id(),
        (grid.x, grid.y),
        tuple(input_tensor.shape),
        tuple(input_tensor.padded_shape),
        str(input_tensor.dtype),
        memory_config_key(input_tensor.memory_config()),
        memory_config_key(memory_config),
    )


def select_plan(input_tensor, memory_config) -> G6Plan:
    """The plan a call with these placements runs (BB-08); deterministic per placement pair and device."""
    key = plan_key(input_tensor, memory_config)
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = _build_plan(input_tensor, memory_config)
        _PLAN_CACHE[key] = plan
    return plan


def _build_plan(input_tensor, memory_config):
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    num_l1_banks = ttnn._ttnn.reports.get_device_info(device).l1_num_banks
    bank_owners = num_l1_banks == grid.x * grid.y
    rows, cols = tile_grid(input_tensor.shape)
    tensor_pages = rows * cols
    assert tensor_pages < 2**32, "page ids are 32-bit"
    page_size = input_tensor.buffer_aligned_page_size()

    in_side = describe_side(input_tensor.memory_config(), rows, cols)
    out_side = describe_side(memory_config, rows, cols)
    for side in (in_side, out_side):
        if side.sharded:
            assert side.num_shards <= len(side.shard_cores), "one shard per core"
    h_in = holders(in_side, tensor_pages, num_l1_banks, bank_owners)
    h_out = holders(out_side, tensor_pages, num_l1_banks, bank_owners)
    regime = select_regime(h_in, h_out)
    max_segment_pages = max(1, MAX_TRANSFER_BYTES // page_size)

    bounce_block_pages = bounce_depth = None
    if regime == "bounce":
        kernel_cores, roles, core_args, bounce_block_pages = _bounce_plan(
            out_side, rows, cols, tensor_pages, grid, page_size
        )
        bounce_depth = BOUNCE_DEPTH
        local_kind, walk_side, block_pages = LOCAL_STAGING, out_side, bounce_block_pages
    else:
        walk_side = out_side if regime == "pull" else in_side
        kernel_cores, roles, core_args, local_kind, block_pages = _owner_plan(
            regime, walk_side, rows, cols, tensor_pages, grid
        )

    return G6Plan(
        regime=regime,
        holders_in=h_in,
        holders_out=h_out,
        l1_bank_owners=bank_owners,
        kernel_cores=kernel_cores,
        roles=roles,
        bounce_block_pages=bounce_block_pages,
        bounce_depth=bounce_depth,
        tensor_rows=rows,
        tensor_cols=cols,
        tensor_pages=tensor_pages,
        aligned_page_size=page_size,
        local_kind=local_kind,
        walk_side=walk_side,
        max_segment_pages=max_segment_pages,
        block_pages=block_pages,
        core_args=core_args,
    )


def output_l1_bytes_per_core(input_tensor, memory_config, plan):
    """The output's own L1 on its busiest core (BB-05 counts it). A sharded slot is the full shard."""
    if buffer_name(memory_config) == "dram":
        return 0
    if scheme_name(memory_config) == "interleaved":
        num_l1_banks = ttnn._ttnn.reports.get_device_info(input_tensor.device()).l1_num_banks
        return math.ceil(plan.tensor_pages / num_l1_banks) * plan.aligned_page_size
    spec = memory_config.shard_spec
    return (spec.shape[0] // 32) * (spec.shape[1] // 32) * plan.aligned_page_size


# ---------------------------------------------------------------------------
# Descriptor
# ---------------------------------------------------------------------------


def _kernel(source, plan, config, input_tensor, output_tensor):
    side = plan.walk_side
    sharded_walk = side is not None and side.sharded
    ct_args = [
        plan.local_kind,
        1 if sharded_walk else 0,
        plan.tensor_pages,
        plan.tensor_rows,
        plan.tensor_cols,
        side.shard_rows if sharded_walk else 0,
        side.shard_cols if sharded_walk else 0,
        side.num_shard_cols if sharded_walk else 0,
        plan.max_segment_pages,
        plan.block_pages,
        CB_STAGING_PAGES,
    ]
    ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    rt_args = ttnn.RuntimeArgs()
    for (x, y), args in plan.core_args.items():
        rt_args[x][y] = list(args)
    return ttnn.KernelDescriptor(
        kernel_source=source,
        core_ranges=plan.kernel_cores,
        compile_time_args=ct_args,
        runtime_args=rt_args,
        common_runtime_args=[input_tensor.buffer_address(), output_tensor.buffer_address()],
        config=config,
    )


def _build_descriptor_parts(plan, input_tensor, output_tensor):
    kernels, cbs = [], []
    if plan.regime in ("pull", "bounce"):
        kernels.append(_kernel(GATHER_KERNEL, plan, ttnn.ReaderConfigDescriptor(), input_tensor, output_tensor))
    if plan.regime in ("push", "bounce"):
        kernels.append(_kernel(SCATTER_KERNEL, plan, ttnn.WriterConfigDescriptor(), input_tensor, output_tensor))
    if plan.regime == "bounce":
        cbs.append(
            ttnn.CBDescriptor(
                total_size=plan.staging_bytes_per_core(),
                core_ranges=plan.kernel_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_STAGING_PAGES,
                        data_format=input_tensor.dtype,
                        page_size=plan.aligned_page_size,
                    )
                ],
            )
        )
    return kernels, cbs


def create_program_descriptor(input_tensor, output_tensor, memory_config, plan) -> ttnn.ProgramDescriptor:
    key = plan_key(input_tensor, memory_config)
    cached = _DESCRIPTOR_CACHE.get(key)
    if cached is None:
        kernels, cbs = _build_descriptor_parts(plan, input_tensor, output_tensor)
        cached = (kernels, cbs, hash(key) & 0xFFFFFFFFFFFFFFFF)
        _DESCRIPTOR_CACHE[key] = cached
    kernels, cbs, program_hash = cached
    addresses = [input_tensor.buffer_address(), output_tensor.buffer_address()]
    for kernel in kernels:
        kernel.common_runtime_args = addresses
    descriptor = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    descriptor.custom_program_hash = program_hash
    return descriptor
