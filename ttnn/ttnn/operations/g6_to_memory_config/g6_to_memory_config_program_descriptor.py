# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Plan and ProgramDescriptor of g6_to_memory_config (op_design.md, Blocking Model).

Every page has one mover, chosen to own one side of it:
  pull_shard / pull_bank  the core holding the page's target slot reads it into place (reader kernel);
  push_shard / push_bank  the core holding the page's resident source writes it out (writer kernel);
  bounce                  neither side is any core's L1 (DRAM -> DRAM): grid workers read a block into
                          cb_bounce_pages and write it out (reader + writer kernels).
A plan depends only on placements, shapes and the device, so plan_of() and the op share it.
"""

from __future__ import annotations

import math
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
PULL_KERNEL = KERNEL_DIR / "g6_to_memory_config_reader.cpp"
PUSH_KERNEL = KERNEL_DIR / "g6_to_memory_config_writer.cpp"

# Block knobs. Every CT/RT value, loop trip count and the CB capacity derive from these.
BOUNCE_DEPTH = 2  # bounce blocks in flight: the reader fills block k+1 while the writer drains block k
MOVERS_PER_OWNER = 1  # data-movement RISCs sharing one owner's block (1 or 2)
L1_OP_BUDGET_BYTES = 1347776  # per-core L1 the model leaves the op (BB-05)

CB_BOUNCE_PAGES = 0
WALK_SHARD, WALK_BANK, WALK_BOUNCE = 0, 1, 2  # g6_to_memory_config_common.hpp
NO_SHARD = 0xFFFFFFFF

_WALK = {
    "pull_shard": WALK_SHARD,
    "push_shard": WALK_SHARD,
    "pull_bank": WALK_BANK,
    "push_bank": WALK_BANK,
    "bounce": WALK_BOUNCE,
}

_bank_cover_by_device = {}


def tile_grid(tensor):
    """(tensor_rows_tiles, tensor_cols_tiles, tile_h, tile_w) of the flattened padded tile grid."""
    tile_h, tile_w = tensor.tile.tile_shape
    padded = list(tensor.padded_shape)
    rows_tiles = math.prod(padded[:-2]) * math.ceil(padded[-2] / tile_h)
    cols_tiles = math.ceil(padded[-1] / tile_w)
    return rows_tiles, cols_tiles, tile_h, tile_w


def shard_grid(memory_config, rows_tiles, cols_tiles, tile_h, tile_w):
    """(shard_h_tiles, shard_w_tiles, n_shard_rows, n_shard_cols) of a sharded placement."""
    shard_h, shard_w = memory_config.shard_spec.shape
    shard_h_tiles, shard_w_tiles = shard_h // tile_h, shard_w // tile_w
    return (
        shard_h_tiles,
        shard_w_tiles,
        math.ceil(rows_tiles / shard_h_tiles),
        math.ceil(cols_tiles / shard_w_tiles),
    )


def grid_cores_row_wise(core_range_set):
    """Cores of a CoreRangeSet, range by range in the order written, row-wise inside each range."""
    cores = []
    for core_range in core_range_set.ranges():
        for y in range(core_range.start.y, core_range.end.y + 1):
            for x in range(core_range.start.x, core_range.end.x + 1):
                cores.append((x, y))
    return cores


def owner_cores(memory_config, n_shard_rows, n_shard_cols):
    """Core holding shard k, at index k, exactly as the buffer places the shards.

    1D (height / width): the grid linearized row-wise, range by range, first num_shards cores.
    2D (block): the grid trimmed to the shard grid from its start, row-wise
    (BufferDistributionSpec::compute_core_list).
    """
    grid = memory_config.shard_spec.grid
    if memory_config.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        start = grid.ranges()[0].start
        return [(start.x + c, start.y + r) for r in range(n_shard_rows) for c in range(n_shard_cols)]
    return grid_cores_row_wise(grid)[: n_shard_rows * n_shard_cols]


def bank_cores_cover_grid(device):
    """True when every L1 bank sits on a compute core of the launch grid, one bank per core."""
    device_id = device.id()
    if device_id not in _bank_cover_by_device:
        grid = device.compute_with_storage_grid_size()
        info = ttnn._ttnn.reports.get_device_info(device)
        _bank_cover_by_device[device_id] = (info.l1_num_banks == grid.x * grid.y, info.l1_num_banks)
    return _bank_cover_by_device[device_id]


def _bounding_box(cores):
    xs = [x for x, _ in cores]
    ys = [y for _, y in cores]
    return (min(xs), min(ys), max(xs), max(ys))


def build_plan(input_tensor, memory_config):
    """The plan of one call: regime, movers, per-core work and block sizes. No allocation."""
    device = input_tensor.device()
    in_config = input_tensor.memory_config()
    rows_tiles, cols_tiles, tile_h, tile_w = tile_grid(input_tensor)
    num_pages = rows_tiles * cols_tiles
    page_size = input_tensor.buffer_aligned_page_size()
    grid = device.compute_with_storage_grid_size()
    cover, l1_num_banks = bank_cores_cover_grid(device)

    src_l1 = in_config.buffer_type == ttnn.BufferType.L1
    dst_l1 = memory_config.buffer_type == ttnn.BufferType.L1
    src_sh = in_config.is_sharded()
    dst_sh = memory_config.is_sharded()

    src_owners = []
    if src_l1 and src_sh:
        _, _, rows, cols = shard_grid(in_config, rows_tiles, cols_tiles, tile_h, tile_w)
        src_owners = owner_cores(in_config, rows, cols)
    dst_owners = []
    dst_share_pages = 0
    if dst_l1 and dst_sh:
        shard_h_tiles, shard_w_tiles, rows, cols = shard_grid(memory_config, rows_tiles, cols_tiles, tile_h, tile_w)
        dst_owners = owner_cores(memory_config, rows, cols)
        dst_share_pages = shard_h_tiles * shard_w_tiles
    elif dst_l1:
        dst_share_pages = math.ceil(num_pages / l1_num_banks)

    # Selection function of op_design.md (Regimes): the side with more owner cores moves, ties pull.
    if dst_l1 and dst_sh:
        regime = "push_shard" if (src_owners and len(src_owners) > len(dst_owners)) else "pull_shard"
    elif dst_l1:
        regime = "push_shard" if src_owners else "pull_bank"
    elif src_l1:
        regime = "push_shard" if src_owners else "push_bank"
    else:
        regime = "bounce"
    fallback = regime in ("pull_bank", "push_bank") and not cover
    if fallback:
        regime = "bounce"

    full_grid = [(x, y) for y in range(grid.y) for x in range(grid.x)]
    roles = {}
    core_args = {}
    bounce_block_pages = 0
    owners = []
    if regime in ("pull_shard", "push_shard"):
        owners = dst_owners if regime == "pull_shard" else src_owners
        movers = list(owners)
        kernel_cores = _bounding_box(owners)
        shard_of = {core: k for k, core in enumerate(owners)}
        sx, sy, ex, ey = kernel_cores
        direction = regime.split("_")[0]
        for y in range(sy, ey + 1):
            for x in range(sx, ex + 1):
                k = shard_of.get((x, y))
                core_args[(x, y)] = [NO_SHARD if k is None else k]
                roles[(x, y)] = "idle" if k is None else f"{direction}_owner {k}"
    elif regime in ("pull_bank", "push_bank"):
        movers = full_grid
        kernel_cores = (0, 0, grid.x - 1, grid.y - 1)
        roles = {core: "bank_walker" for core in full_grid}
    else:
        movers = full_grid
        kernel_cores = (0, 0, grid.x - 1, grid.y - 1)
        _, _, group_1, group_2, pages_1, pages_2 = ttnn.split_work_to_cores(grid, num_pages, row_wise=True)
        start = 0
        for group, core_pages in ((group_1, pages_1), (group_2, pages_2)):
            for core in grid_cores_row_wise(group):
                core_args[core] = [start, core_pages]
                roles[core] = f"bounce [{start}, {start + core_pages})"
                start += core_pages
        assert start == num_pages, f"bounce split covers {start} of {num_pages} pages"
        for core in full_grid:
            if core not in core_args:
                core_args[core] = [0, 0]
                roles[core] = "idle"
        core_pages_max = pages_1
        bounce_block_pages = max(
            1,
            min(math.ceil(core_pages_max / BOUNCE_DEPTH), L1_OP_BUDGET_BYTES // (BOUNCE_DEPTH * page_size)),
        )

    cb_pages = BOUNCE_DEPTH * bounce_block_pages if regime == "bounce" else 0
    return {
        "regime": regime,
        "fallback_from_bank": fallback,
        "walk": _WALK[regime],
        "movers": movers,
        "owner_cores": owners,
        "kernel_cores": kernel_cores,
        "roles": roles,
        "core_args": core_args,
        "num_pages": num_pages,
        "tensor_rows_tiles": rows_tiles,
        "tensor_cols_tiles": cols_tiles,
        "page_size": page_size,
        "movers_per_owner": MOVERS_PER_OWNER if regime != "bounce" else 1,
        "bounce_depth": BOUNCE_DEPTH,
        "bounce_block_pages": bounce_block_pages,
        "cb_bounce_pages": cb_pages,
        "l1_op_bytes": dst_share_pages * page_size + cb_pages * page_size,
    }


def _mover_configs(regime, movers_per_owner):
    """(processor/NoC config, kernel source, mover_index) per kernel; mover 0 takes the default placement.

    bounce is one mover split into a reader and a writer, so both halves are mover 0 of 1.
    """
    pull = ttnn.ReaderConfigDescriptor()
    pull_second = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1)
    push = ttnn.WriterConfigDescriptor()
    push_second = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0)
    if regime == "bounce":
        return [(pull, PULL_KERNEL, 0), (push, PUSH_KERNEL, 0)]
    if regime.startswith("pull"):
        return [(pull, PULL_KERNEL, 0), (pull_second, PULL_KERNEL, 1)][:movers_per_owner]
    return [(push, PUSH_KERNEL, 0), (push_second, PUSH_KERNEL, 1)][:movers_per_owner]


def create_program_descriptor(plan, input_tensor, output_tensor, program_hash):
    """One ProgramDescriptor for the plan; base addresses are common RT args patched per call."""
    assert output_tensor.buffer_aligned_page_size() == plan["page_size"], "input and output page sizes differ"
    assert list(output_tensor.padded_shape) == list(input_tensor.padded_shape), "input and output tile grids differ"

    sx, sy, ex, ey = plan["kernel_cores"]
    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(sx, sy), ttnn.CoreCoord(ex, ey))])
    accessor_args = list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()) + list(
        ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
    )
    common_args = [input_tensor.buffer_address(), output_tensor.buffer_address()]

    kernels = []
    for config, source, mover_index in _mover_configs(plan["regime"], plan["movers_per_owner"]):
        ct_args = [
            plan["walk"],
            plan["page_size"],
            plan["num_pages"],
            plan["movers_per_owner"],
            mover_index,
            CB_BOUNCE_PAGES,
            plan["bounce_block_pages"],
        ] + accessor_args
        runtime_args = ttnn.RuntimeArgs()
        for (x, y), args in plan["core_args"].items():
            runtime_args[x][y] = args
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(source),
                core_ranges=core_ranges,
                compile_time_args=ct_args,
                runtime_args=runtime_args,
                common_runtime_args=common_args,
                config=config,
            )
        )

    cbs = []
    if plan["regime"] == "bounce":
        cbs.append(
            ttnn.CBDescriptor(
                total_size=plan["cb_bounce_pages"] * plan["page_size"],
                core_ranges=core_ranges,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_BOUNCE_PAGES,
                        data_format=input_tensor.dtype,
                        page_size=plan["page_size"],
                    )
                ],
            )
        )

    descriptor = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    descriptor.custom_program_hash = program_hash
    return descriptor


def patch_addresses(descriptor, src_address, dst_address):
    """Point a cached descriptor at this call's buffers (common RT args 0 and 1 of every kernel)."""
    for kernel in descriptor.kernels:
        kernel.common_runtime_args[0] = src_address
        kernel.common_runtime_args[1] = dst_address
