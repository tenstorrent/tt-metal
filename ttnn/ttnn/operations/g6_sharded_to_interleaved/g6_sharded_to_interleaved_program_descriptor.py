# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Program descriptor for g6_sharded_to_interleaved: the `owner_writes` regime.

Every shard core writes its own resident tiles straight to their interleaved output pages (one NoC
write per tile, zero reads). The plan and the descriptor are memoized per program-shaping key; a call
only patches the output address (common RT arg) and rebinds `cb_input_shard` to its own input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
WRITER_KERNEL = KERNEL_DIR / "g6_sharded_to_interleaved_writer.cpp"

TILE = 32
CB_INPUT_SHARD = 0

# Blocking-model knobs (op_design.md -> Blocking Model). Each is defined once here; every dependent
# quantity (lane rectangles, block extents, kernel instances, RT args) is derived from them.
#


@dataclass(frozen=True)
class Knobs:
    # Data-movement RISCs per shard core issuing the block's writes. 1 = BRISC on NoC1; 2 adds an
    # instance of the same kernel on NCRISC / NoC0, the block cut along its longer tile axis.
    num_write_lanes: int = 1
    # Block extents along tile_row / tile_col; None = the lane's whole rectangle (one block per lane).
    block_tile_rows: int | None = None
    block_tile_cols: int | None = None


DEFAULT_KNOBS = Knobs()

# (processor, NoC) of write lane i.
_LANE_CONFIGS = (
    (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
    (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
)

NOC_MAX_BURST_SIZE = 16384
MODEL_L1_BUDGET = 1347776


@dataclass(frozen=True)
class Block:
    first_tile_row: int
    first_tile_col: int
    tile_rows: int
    tile_cols: int


@dataclass(frozen=True)
class Plan:
    tensor_tile_rows: int
    tensor_tile_cols: int
    shard_tile_rows: int
    shard_tile_cols: int
    num_shards: int
    cores: tuple  # ttnn.CoreCoord per shard slot k, in the buffer's own shard order
    blocks: tuple  # Block per core, None for a grid core without a shard
    page_size: int
    knobs: Knobs


def _tile_grid(input_tensor):
    padded = tuple(input_tensor.padded_shape)
    return math.prod(padded[:-1]) // TILE, padded[-1] // TILE


def _split(extent, parts):
    """Near-even split of `extent` into `parts` contiguous pieces, larger pieces first."""
    base, rem = divmod(extent, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def _lane_rects(block, num_lanes):
    """Cuts a block into `num_lanes` disjoint rectangles along its longer tile axis.

    Returns (row_offset, col_offset, rows, cols) per lane, offsets local to the block.
    """
    rects = []
    offset = 0
    if block.tile_rows >= block.tile_cols:
        for rows in _split(block.tile_rows, num_lanes):
            rects.append((offset, 0, rows, block.tile_cols))
            offset += rows
    else:
        for cols in _split(block.tile_cols, num_lanes):
            rects.append((0, offset, block.tile_rows, cols))
            offset += cols
    return rects


def build_plan(input_tensor, memory_config, knobs=DEFAULT_KNOBS):
    spec = input_tensor.memory_config().shard_spec
    tensor_tile_rows, tensor_tile_cols = _tile_grid(input_tensor)
    shard_tile_rows, shard_tile_cols = spec.shape[0] // TILE, spec.shape[1] // TILE
    layout = input_tensor.memory_config().memory_layout

    if layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED and shard_tile_cols != tensor_tile_cols:
        raise RuntimeError(
            f"g6_sharded_to_interleaved: height-sharded shard is {shard_tile_cols} tile cols wide, "
            f"tensor is {tensor_tile_cols}"
        )
    if layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED and shard_tile_rows != tensor_tile_rows:
        raise RuntimeError(
            f"g6_sharded_to_interleaved: width-sharded shard is {shard_tile_rows} tile rows high, "
            f"tensor is {tensor_tile_rows}"
        )
    if not 1 <= knobs.num_write_lanes <= len(_LANE_CONFIGS):
        raise RuntimeError(f"g6_sharded_to_interleaved: num_write_lanes={knobs.num_write_lanes} not in [1, 2]")
    for cap in (knobs.block_tile_rows, knobs.block_tile_cols):
        if cap is not None and cap < 1:
            raise RuntimeError(f"g6_sharded_to_interleaved: block extent {cap} < 1")

    # The buffer maps shard k to cores[k] with exactly this walk: ranges in order, each along rows
    # (ROW_MAJOR) or columns (COL_MAJOR). Rebuilding or sorting the ranges would permute the map.
    row_wise = spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
    cores = tuple(ttnn.corerange_to_cores(spec.grid, None, row_wise))

    shards_along_rows = math.ceil(tensor_tile_rows / shard_tile_rows)
    shards_along_cols = math.ceil(tensor_tile_cols / shard_tile_cols)
    num_shards = shards_along_rows * shards_along_cols
    if num_shards > len(cores):
        raise RuntimeError(f"g6_sharded_to_interleaved: {num_shards} shards on {len(cores)} cores")

    blocks = []
    for k in range(len(cores)):
        if k >= num_shards:
            blocks.append(None)
            continue
        first_row = (k // shards_along_cols) * shard_tile_rows
        first_col = (k % shards_along_cols) * shard_tile_cols
        blocks.append(
            Block(
                first_row,
                first_col,
                min(shard_tile_rows, tensor_tile_rows - first_row),
                min(shard_tile_cols, tensor_tile_cols - first_col),
            )
        )

    covered = sum(b.tile_rows * b.tile_cols for b in blocks if b is not None)
    if covered != tensor_tile_rows * tensor_tile_cols:
        raise RuntimeError(
            f"g6_sharded_to_interleaved: blocks cover {covered} of {tensor_tile_rows * tensor_tile_cols} pages"
        )

    page_size = ttnn.tile_size(input_tensor.dtype)
    if input_tensor.buffer_aligned_page_size() != page_size:
        raise RuntimeError(
            f"g6_sharded_to_interleaved: shard slot stride {input_tensor.buffer_aligned_page_size()} B "
            f"!= tile size {page_size} B"
        )
    if page_size > NOC_MAX_BURST_SIZE:
        raise RuntimeError(f"g6_sharded_to_interleaved: page {page_size} B exceeds one NoC packet")

    return Plan(
        tensor_tile_rows=tensor_tile_rows,
        tensor_tile_cols=tensor_tile_cols,
        shard_tile_rows=shard_tile_rows,
        shard_tile_cols=shard_tile_cols,
        num_shards=num_shards,
        cores=cores,
        blocks=tuple(blocks),
        page_size=page_size,
        knobs=knobs,
    )


def _check_l1_budget(plan, device, memory_config):
    if memory_config.buffer_type != ttnn.BufferType.L1:
        return
    n_l1_banks = ttnn.get_memory_view(device, ttnn.BufferType.L1).num_banks
    pages = plan.tensor_tile_rows * plan.tensor_tile_cols
    per_bank = math.ceil(pages / n_l1_banks) * plan.page_size
    if per_bank > MODEL_L1_BUDGET:
        raise RuntimeError(
            f"g6_sharded_to_interleaved: L1 output needs {per_bank} B per bank, model budget is {MODEL_L1_BUDGET} B"
        )


def _block_extent(cap, extent):
    return extent if cap is None else min(cap, extent)


def create_program_descriptor(input_tensor, output_tensor, plan):
    """ProgramDescriptor for `plan`. Addresses in it belong to this call's tensors; see `CachedProgram.bind`."""
    if output_tensor.buffer_page_size() != plan.page_size:
        raise RuntimeError(
            f"g6_sharded_to_interleaved: output page {output_tensor.buffer_page_size()} B != input page {plan.page_size} B"
        )

    grid = input_tensor.memory_config().shard_spec.grid
    cb_input_shard = ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT_SHARD, input_tensor)

    ct_args = [CB_INPUT_SHARD, plan.page_size]
    ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    common_rt_args = [output_tensor.buffer_address(), plan.shard_tile_cols, plan.tensor_tile_cols]

    knobs = plan.knobs
    lane_rt_args = [ttnn.RuntimeArgs() for _ in range(knobs.num_write_lanes)]
    for core, block in zip(plan.cores, plan.blocks):
        rects = _lane_rects(block, knobs.num_write_lanes) if block is not None else []
        for lane in range(knobs.num_write_lanes):
            args = [0, 0, 0, 0, 0, 0]
            if lane < len(rects) and rects[lane][2] * rects[lane][3] > 0:
                row_off, col_off, rows, cols = rects[lane]
                args = [
                    row_off * plan.shard_tile_cols + col_off,
                    (block.first_tile_row + row_off) * plan.tensor_tile_cols + block.first_tile_col + col_off,
                    rows,
                    cols,
                    _block_extent(knobs.block_tile_rows, rows),
                    _block_extent(knobs.block_tile_cols, cols),
                ]
            lane_rt_args[lane][core.x][core.y] = args

    kernels = []
    for lane in range(knobs.num_write_lanes):
        processor, noc = _LANE_CONFIGS[lane]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(WRITER_KERNEL),
                core_ranges=grid,
                compile_time_args=ct_args,
                runtime_args=lane_rt_args[lane],
                common_runtime_args=common_rt_args,
                config=ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc),
            )
        )

    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb_input_shard])


class CachedProgram:
    """A memoized descriptor plus live references into it for the per-call patch."""

    def __init__(self, plan, descriptor):
        self.plan = plan
        self.descriptor = descriptor
        # Element references into the C++ vectors: mutating them mutates the descriptor in place.
        self._kernels = list(descriptor.kernels)
        self._cb_input_shard = descriptor.cbs[0]

    def bind(self, input_tensor, output_tensor):
        """Points the descriptor at this call's tensors. The program cache hashes neither address."""
        output_addr = output_tensor.buffer_address()
        for kernel in self._kernels:
            kernel.common_runtime_args[0] = output_addr
        self._cb_input_shard.set_buffer_from_cb(ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT_SHARD, input_tensor))
        return self.descriptor


_PROGRAMS = {}


def program_key(input_tensor, memory_config, knobs=DEFAULT_KNOBS):
    """Everything that shapes the plan or the descriptor. Orientation and range order change only
    per-core RT values (not the C++ program hash), so they must be in this key."""
    mc = input_tensor.memory_config()
    spec = mc.shard_spec
    ranges = tuple((r.start.x, r.start.y, r.end.x, r.end.y) for r in spec.grid.ranges())
    return (
        id(input_tensor.device()),
        tuple(input_tensor.padded_shape),
        input_tensor.dtype,
        mc.memory_layout,
        tuple(spec.shape),
        ranges,
        spec.orientation,
        memory_config.buffer_type,
        knobs,
    )


def get_program(input_tensor, output_tensor, memory_config, knobs=DEFAULT_KNOBS):
    key = program_key(input_tensor, memory_config, knobs)
    cached = _PROGRAMS.get(key)
    if cached is None:
        plan = build_plan(input_tensor, memory_config, knobs)
        _check_l1_budget(plan, input_tensor.device(), memory_config)
        cached = CachedProgram(plan, create_program_descriptor(input_tensor, output_tensor, plan))
        _PROGRAMS[key] = cached
    return cached


def cached_plan(input_tensor, memory_config, knobs=DEFAULT_KNOBS):
    cached = _PROGRAMS.get(program_key(input_tensor, memory_config, knobs))
    return None if cached is None else cached.plan
