# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for g6_sharded_to_interleaved (owner-writes).

Every shard-holding core is one block: its whole resident shard, written tile by tile from the
shard slots to the interleaved output pages. The block's local tile range is split across
NUM_LANES data-movement RISCs. There is no compute kernel and no staging buffer; the only CBs
are zero-copy bindings of the input shard, one index per lane so each CB has one consumer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
STORE_KERNEL = str(KERNEL_DIR / "g6_sharded_to_interleaved_writer.cpp")

TILE = 32

# Issue lanes per core. Lane j runs on _LANE_PLACEMENT[j] and reads the shard through
# CB index _CB_SHARD_LANE[j]. The two lanes must sit on different NoCs: each RISC's write barrier
# counts the transactions of its own NoC.
NUM_LANES = 2
_LANE_PLACEMENT = (
    (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),  # BRISC
    (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),  # NCRISC
)
_CB_SHARD_LANE = (0, 1)  # cb_shard_lane0, cb_shard_lane1


@dataclass(frozen=True)
class TileGeometry:
    tensor_rows_tiles: int
    tensor_cols_tiles: int
    shard_rows_tiles: int
    shard_cols_tiles: int
    num_shards: int


@dataclass(frozen=True)
class CoreAssignment:
    """Role of one core of the input grid in a call (the plan report)."""

    core: tuple[int, int]
    shard_index: int | None  # None: grid core past the last shard, no block
    a0: int = 0
    b0: int = 0
    block_extent_rows: int = 0
    block_extent_cols: int = 0
    lane_ranges: tuple[tuple[int, int], ...] = ()  # (lane_start, lane_tiles) per lane

    @property
    def num_blocks(self) -> int:
        return 0 if self.shard_index is None else 1


def tile_geometry(input_tensor: ttnn.Tensor) -> TileGeometry:
    padded = list(input_tensor.padded_shape)
    tensor_rows_tiles = math.prod(padded[:-1]) // TILE
    tensor_cols_tiles = padded[-1] // TILE
    shard_h, shard_w = input_tensor.memory_config().shard_spec.shape
    shard_rows_tiles = shard_h // TILE
    shard_cols_tiles = shard_w // TILE
    num_shards = -(-tensor_rows_tiles // shard_rows_tiles) * -(-tensor_cols_tiles // shard_cols_tiles)
    return TileGeometry(tensor_rows_tiles, tensor_cols_tiles, shard_rows_tiles, shard_cols_tiles, num_shards)


def _lane_ranges(n_tiles: int, num_lanes: int) -> tuple[tuple[int, int], ...]:
    per_lane = -(-n_tiles // num_lanes)
    ranges = []
    for lane in range(num_lanes):
        start = min(lane * per_lane, n_tiles)
        ranges.append((start, min(per_lane, n_tiles - start)))
    return tuple(ranges)


def compute_plan(input_tensor: ttnn.Tensor, num_lanes: int = NUM_LANES) -> tuple[TileGeometry, list[CoreAssignment]]:
    """Per-core assignment of a call: shard k goes to the k-th core of the input's own grid walk."""
    spec = input_tensor.memory_config().shard_spec
    geometry = tile_geometry(input_tensor)
    row_wise = spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
    cores = ttnn.corerange_to_cores(spec.grid, None, row_wise)
    n_cols_shards = -(-geometry.tensor_cols_tiles // geometry.shard_cols_tiles)

    plan = []
    for k, core in enumerate(cores):
        xy = (core.x, core.y)
        if k >= geometry.num_shards:
            plan.append(CoreAssignment(core=xy, shard_index=None))
            continue
        a0 = (k // n_cols_shards) * geometry.shard_rows_tiles
        b0 = (k % n_cols_shards) * geometry.shard_cols_tiles
        rows = min(geometry.shard_rows_tiles, geometry.tensor_rows_tiles - a0)
        cols = min(geometry.shard_cols_tiles, geometry.tensor_cols_tiles - b0)
        plan.append(
            CoreAssignment(
                core=xy,
                shard_index=k,
                a0=a0,
                b0=b0,
                block_extent_rows=rows,
                block_extent_cols=cols,
                lane_ranges=_lane_ranges(rows * cols, num_lanes),
            )
        )
    return geometry, plan


def create_program_descriptor(
    input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor, num_lanes: int = NUM_LANES
) -> ttnn.ProgramDescriptor:
    if not 1 <= num_lanes <= len(_LANE_PLACEMENT):
        raise ValueError(f"g6_sharded_to_interleaved: num_lanes={num_lanes} outside [1, {len(_LANE_PLACEMENT)}]")

    geometry, plan = compute_plan(input_tensor, num_lanes)
    grid = input_tensor.memory_config().shard_spec.grid
    slot_bytes = input_tensor.buffer_aligned_page_size()
    page_bytes = output_tensor.buffer_page_size()
    out_accessor_args = list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    cbs = [ttnn.cb_descriptor_from_sharded_tensor(_CB_SHARD_LANE[lane], input_tensor) for lane in range(num_lanes)]

    kernels = []
    for lane in range(num_lanes):
        compile_time_args = [
            _CB_SHARD_LANE[lane],
            geometry.shard_cols_tiles,
            geometry.tensor_cols_tiles,
            slot_bytes,
            page_bytes,
            *out_accessor_args,
        ]
        runtime_args = ttnn.RuntimeArgs()
        for assignment in plan:
            args = [assignment.num_blocks]
            if assignment.num_blocks:
                lane_start, lane_tiles = assignment.lane_ranges[lane]
                args += [assignment.a0, assignment.b0, assignment.block_extent_cols, lane_start, lane_tiles]
            runtime_args[assignment.core[0]][assignment.core[1]] = args
        processor, noc = _LANE_PLACEMENT[lane]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=STORE_KERNEL,
                core_ranges=grid,
                compile_time_args=compile_time_args,
                runtime_args=runtime_args,
                common_runtime_args=[output_tensor.buffer_address()],
                config=ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc),
            )
        )

    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


class _CachedProgram:
    """A built descriptor whose per-call values (shard binding, output address) are patched in place."""

    def __init__(self, descriptor: ttnn.ProgramDescriptor):
        self.descriptor = descriptor
        self._kernels = list(descriptor.kernels)
        self._cbs = list(descriptor.cbs)

    def bind(self, input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor) -> ttnn.ProgramDescriptor:
        shard_binding = ttnn.cb_descriptor_from_sharded_tensor(_CB_SHARD_LANE[0], input_tensor)
        for cb in self._cbs:
            cb.set_buffer_from_cb(shard_binding)
        out_addr = [output_tensor.buffer_address()]
        for kernel in self._kernels:
            kernel.common_runtime_args = out_addr
        return self.descriptor


_PROGRAM_CACHE: dict[tuple, _CachedProgram] = {}


def program_cache_key(input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor, num_lanes: int) -> tuple:
    """Everything that changes the descriptor; the descriptor holds no device-specific value."""
    in_mc = input_tensor.memory_config()
    spec = in_mc.shard_spec
    ranges = tuple((r.start.x, r.start.y, r.end.x, r.end.y) for r in spec.grid.ranges())
    return (
        tuple(input_tensor.padded_shape),
        input_tensor.dtype,
        in_mc.memory_layout,
        tuple(spec.shape),
        ranges,
        spec.orientation,
        output_tensor.memory_config().buffer_type,
        num_lanes,
    )


def get_program_descriptor(
    input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor, num_lanes: int = NUM_LANES
) -> ttnn.ProgramDescriptor:
    key = program_cache_key(input_tensor, output_tensor, num_lanes)
    cached = _PROGRAM_CACHE.get(key)
    if cached is None:
        cached = _CachedProgram(create_program_descriptor(input_tensor, output_tensor, num_lanes))
        _PROGRAM_CACHE[key] = cached
    return cached.bind(input_tensor, output_tensor)
