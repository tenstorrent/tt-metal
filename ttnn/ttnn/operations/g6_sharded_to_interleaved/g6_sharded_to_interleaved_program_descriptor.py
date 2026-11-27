# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Program descriptor for g6_sharded_to_interleaved (regime `owner_writes_resident`).

Every core of the input's shard grid writes its own resident shard (the block: the whole shard, one
block per core) to the interleaved output pages. The core's data-movement RISCs split the block into
`NUM_WRITER_LANES` contiguous lane sub-blocks; each lane reads its tiles in place through its own CB
backed on the sharded input buffer (zero-copy) and issues one NoC write per tile. No compute kernel.

The shard index is derived on device from the core's logical coordinates against a compile-time copy of
the shard grid, so the only runtime value is the output address (one common runtime arg per lane).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
WRITER_KERNEL = str(KERNEL_DIR / "g6_sharded_to_interleaved_writer.cpp")

TILE_HW = 32

# Lane knob (design: `num_writer_lanes`, L in {1, 2}). Sets the number of writer kernels and shard CBs.
NUM_WRITER_LANES = 2

# Lane j -> (CB index, RISC, NoC). Lane 0 on BRISC / NoC 1, lane 1 on NCRISC / NoC 0.
_LANE_PLACEMENT = (
    (0, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
    (1, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
)
assert 1 <= NUM_WRITER_LANES <= len(_LANE_PLACEMENT)

_STRATEGIES = {
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: "height_sharded",
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: "width_sharded",
}


@dataclass(frozen=True)
class ShardGeometry:
    """Every host-derived parameter of the page map, from the input tensor alone."""

    in_scheme: str
    tensor_tile_rows: int  # R: tile rows of the 2-D view (leading dims folded into rows)
    tensor_tile_cols: int  # C
    shard_tiles_h: int
    shard_tiles_w: int
    shard_row_step: int
    shard_col_step: int
    num_shards: int
    orientation_row_major: bool
    core_ranges: tuple  # ((x0, y0, x1, y1), ...) in the shard grid's stored order
    page_bytes: int

    def shard_extent(self, k):
        """(row0, col0, rows, cols) of shard k; the last shard may be ragged."""
        row0 = k * self.shard_row_step
        col0 = k * self.shard_col_step
        rows = min(self.shard_tiles_h, self.tensor_tile_rows - row0)
        cols = min(self.shard_tiles_w, self.tensor_tile_cols - col0)
        return row0, col0, rows, cols

    @staticmethod
    def lane_range(block_tiles, lane, num_lanes=NUM_WRITER_LANES):
        per_lane = -(-block_tiles // num_lanes)
        begin = min(lane * per_lane, block_tiles)
        return begin, min(begin + per_lane, block_tiles)


def shard_geometry(input_tensor) -> ShardGeometry:
    """Derive the page-map parameters. Assumes validate() passed (whole-tile shard, height/width)."""
    padded = tuple(int(d) for d in input_tensor.padded_shape)
    spec = input_tensor.memory_config().shard_spec
    shard_h, shard_w = (int(d) for d in spec.shape)
    in_scheme = _STRATEGIES[input_tensor.memory_config().memory_layout]

    tensor_tile_rows = math.prod(padded[:-1]) // TILE_HW
    tensor_tile_cols = padded[-1] // TILE_HW
    shard_tiles_h = shard_h // TILE_HW
    shard_tiles_w = shard_w // TILE_HW
    if in_scheme == "height_sharded":
        shard_row_step, shard_col_step = shard_tiles_h, 0
        num_shards = -(-tensor_tile_rows // shard_tiles_h)
    else:
        shard_row_step, shard_col_step = 0, shard_tiles_w
        num_shards = -(-tensor_tile_cols // shard_tiles_w)

    core_ranges = tuple((r.start.x, r.start.y, r.end.x, r.end.y) for r in spec.grid.ranges())
    return ShardGeometry(
        in_scheme=in_scheme,
        tensor_tile_rows=tensor_tile_rows,
        tensor_tile_cols=tensor_tile_cols,
        shard_tiles_h=shard_tiles_h,
        shard_tiles_w=shard_tiles_w,
        shard_row_step=shard_row_step,
        shard_col_step=shard_col_step,
        num_shards=num_shards,
        orientation_row_major=spec.orientation == ttnn.ShardOrientation.ROW_MAJOR,
        core_ranges=core_ranges,
        page_bytes=int(input_tensor.buffer_aligned_page_size()),
    )


def program_cache_key(input_tensor, memory_config) -> tuple:
    """Everything a CT arg, core range or CB size of the program derives from (BB-07)."""
    mc = input_tensor.memory_config()
    spec = mc.shard_spec
    return (
        "g6_sharded_to_interleaved/owner_writes_resident",
        tuple(int(d) for d in input_tensor.shape),
        tuple(int(d) for d in input_tensor.padded_shape),
        str(input_tensor.dtype),
        str(input_tensor.layout),
        str(mc.memory_layout),
        tuple(int(d) for d in spec.shape),
        tuple((r.start.x, r.start.y, r.end.x, r.end.y) for r in spec.grid.ranges()),
        str(spec.orientation),
        str(memory_config.buffer_type),
        NUM_WRITER_LANES,
    )


def _custom_hash(key) -> int:
    return int.from_bytes(hashlib.blake2b(repr(key).encode(), digest_size=8).digest(), "little")


def _shard_cb(cb_index, input_tensor):
    return ttnn.cb_descriptor_from_sharded_tensor(cb_index, input_tensor)


def create_program_descriptor(input_tensor, output_tensor, cache_key) -> ttnn.ProgramDescriptor:
    geo = shard_geometry(input_tensor)
    if int(output_tensor.buffer_aligned_page_size()) != geo.page_bytes:
        raise ValueError(
            "g6_sharded_to_interleaved: input shard slot stride "
            f"{geo.page_bytes} B != output page size {output_tensor.buffer_aligned_page_size()} B"
        )
    shard_grid = input_tensor.memory_config().shard_spec.grid

    range_table = [v for rng in geo.core_ranges for v in rng]
    output_accessor_ct = ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()

    kernels = []
    cbs = []
    for lane in range(NUM_WRITER_LANES):
        cb_index, processor, noc = _LANE_PLACEMENT[lane]
        cbs.append(_shard_cb(cb_index, input_tensor))
        ct_args = [
            cb_index,
            lane,
            NUM_WRITER_LANES,
            geo.tensor_tile_rows,
            geo.tensor_tile_cols,
            geo.shard_tiles_h,
            geo.shard_tiles_w,
            geo.shard_row_step,
            geo.shard_col_step,
            geo.num_shards,
            int(geo.orientation_row_major),
            len(geo.core_ranges),
            *range_table,
            *output_accessor_ct,
        ]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=WRITER_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=shard_grid,
                compile_time_args=ct_args,
                common_runtime_args=[output_tensor.buffer_address()],
                config=ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc),
            )
        )

    descriptor = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    descriptor.custom_program_hash = _custom_hash(cache_key)
    return descriptor


def refresh_program_descriptor(descriptor, input_tensor, output_tensor) -> None:
    """Point the cached program at this call's tensors: shard CBs -> input buffer, common RT -> output.

    Element access on the descriptor's kernel / CB vectors returns references, so this updates in place.
    """
    output_addr = output_tensor.buffer_address()
    for kernel in descriptor.kernels:
        kernel.common_runtime_args[0] = output_addr
    for lane, cb in enumerate(descriptor.cbs):
        cb.set_buffer_from_cb(_shard_cb(_LANE_PLACEMENT[lane][0], input_tensor))


def role_dump(input_tensor) -> list:
    """Per-core role (BB-21): core -> shard k, each lane's local tile range, first / last output page."""
    geo = shard_geometry(input_tensor)
    grid = input_tensor.memory_config().shard_spec.grid
    cores = ttnn.corerange_to_cores(grid, None, row_wise=geo.orientation_row_major)
    roles = []
    for k, core in enumerate(cores):
        if k >= geo.num_shards:
            roles.append({"core": (core.x, core.y), "shard": None, "lanes": []})
            continue
        row0, col0, rows, cols = geo.shard_extent(k)
        lanes = []
        for lane in range(NUM_WRITER_LANES):
            begin, end = geo.lane_range(rows * cols, lane)
            pages = [(row0 + t // cols) * geo.tensor_tile_cols + col0 + t % cols for t in (begin, end - 1)]
            lanes.append({"tiles": (begin, end), "pages": tuple(pages) if end > begin else None})
        roles.append({"core": (core.x, core.y), "shard": k, "lanes": lanes})
    return roles
