# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Plan and ProgramDescriptor for g6_sharded_to_interleaved, regime `owner_writes`.

Every core of the input's shard grid stores the tiles of its own resident shard straight to their
interleaved output pages: one NoC write per tile, no NoC read, no op-allocated L1. The shard is
read in place through `cb_input_shard`, a CB bound to the input's own buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
WRITER_KERNEL = str(KERNEL_DIR / "g6_sharded_to_interleaved_writer.cpp")

TILE = 32

# Data-movement RISCs per shard core that issue the block's writes. 1: the writer on BRISC / NoC1.
# 2: the block is cut in two along its longer tile axis, the second half issued by a second
# instance of the same kernel on NCRISC / NoC0.
NUM_WRITE_LANES = 1

# Lane l runs on _LANE_PLACEMENT[l]: (processor, noc).
_LANE_PLACEMENT = (
    (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
    (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
)

CB_INPUT_SHARD = 0

# L1 the model leaves to the op on every core (BB-05).
L1_BUDGET_BYTES = 1347776
# One NoC packet on Blackhole: NOC_MAX_BURST_SIZE. A page must fit the one-packet write path.
NOC_MAX_BURST_BYTES = 16384

# Common RT args of the writer: [output base address, tensor_tile_cols, shard_tile_cols].
# The address is the only per-call value and is patched on every call.
_COMMON_RT_OUTPUT_ADDR = 0


@dataclass(frozen=True)
class Block:
    """A rectangle of the padded 2-D tile grid, in tiles."""

    first_row: int
    first_col: int
    rows: int
    cols: int


@dataclass(frozen=True)
class Plan:
    tensor_tile_rows: int
    tensor_tile_cols: int
    shard_tile_rows: int
    shard_tile_cols: int
    page_size: int
    num_write_lanes: int
    # Shard k is owned by cores[k]; cores past the last shard hold no block.
    cores: tuple
    # blocks[i]: the block of cores[i] (its clipped shard), or None.
    blocks: tuple
    # lane_blocks[l][i]: the rectangle lane l of cores[i] stores, or None.
    lane_blocks: tuple


def _split_for_lanes(block, num_write_lanes):
    """Cut a block into num_write_lanes rectangles along its longer tile axis, near-equal."""
    along_rows = block.rows >= block.cols
    extent = block.rows if along_rows else block.cols
    pieces = []
    start = 0
    for lane in range(num_write_lanes):
        length = extent // num_write_lanes + (1 if lane < extent % num_write_lanes else 0)
        if length == 0:
            pieces.append(None)
        elif along_rows:
            pieces.append(Block(block.first_row + start, block.first_col, length, block.cols))
        else:
            pieces.append(Block(block.first_row, block.first_col + start, block.rows, length))
        start += length
    return pieces


def build_plan(input_tensor, memory_config, num_write_lanes=NUM_WRITE_LANES):
    """Ownership plan: shard k -> cores[k], its block = the shard rectangle clipped to the tensor."""
    if num_write_lanes not in range(1, len(_LANE_PLACEMENT) + 1):
        raise RuntimeError(
            f"g6_sharded_to_interleaved: num_write_lanes={num_write_lanes} not in 1..{len(_LANE_PLACEMENT)}"
        )

    padded_shape = tuple(input_tensor.padded_shape)
    rows = 1
    for dim in padded_shape[:-1]:
        rows *= dim
    tensor_tile_rows = rows // TILE
    tensor_tile_cols = padded_shape[-1] // TILE

    mc = input_tensor.memory_config()
    spec = mc.shard_spec
    shard_tile_rows = spec.shape[0] // TILE
    shard_tile_cols = spec.shape[1] // TILE

    page_size = ttnn.tile_size(input_tensor.dtype)
    if input_tensor.buffer_aligned_page_size() != page_size:
        raise RuntimeError(
            f"g6_sharded_to_interleaved: input page {input_tensor.buffer_aligned_page_size()} B != tile size {page_size} B"
        )
    if page_size > NOC_MAX_BURST_BYTES:
        raise RuntimeError(f"g6_sharded_to_interleaved: page {page_size} B exceeds one NoC packet")

    if mc.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED and shard_tile_cols != tensor_tile_cols:
        raise RuntimeError(
            f"g6_sharded_to_interleaved: height shard spans {shard_tile_cols} tile cols, tensor has {tensor_tile_cols}"
        )
    if mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED and shard_tile_rows != tensor_tile_rows:
        raise RuntimeError(
            f"g6_sharded_to_interleaved: width shard spans {shard_tile_rows} tile rows, tensor has {tensor_tile_rows}"
        )

    # The buffer maps its own shards with this walk: ranges in their stored order, each walked
    # along rows (row-major orientation) or columns.
    cores = tuple(
        (c.x, c.y)
        for c in ttnn.corerange_to_cores(
            spec.grid, None, row_wise=(spec.orientation == ttnn.ShardOrientation.ROW_MAJOR)
        )
    )

    shards_along_rows = -(-tensor_tile_rows // shard_tile_rows)
    shards_along_cols = -(-tensor_tile_cols // shard_tile_cols)
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

    covered = sum(b.rows * b.cols for b in blocks if b is not None)
    if covered != tensor_tile_rows * tensor_tile_cols:
        raise RuntimeError(
            f"g6_sharded_to_interleaved: blocks cover {covered} tiles of {tensor_tile_rows * tensor_tile_cols}"
        )

    if memory_config.buffer_type == ttnn.BufferType.L1:
        device = input_tensor.device()
        n_l1_banks = ttnn.get_memory_view(device, ttnn.BufferType.L1).num_banks
        per_bank = -(-(tensor_tile_rows * tensor_tile_cols) // n_l1_banks) * page_size
        if per_bank > L1_BUDGET_BYTES:
            raise RuntimeError(
                f"g6_sharded_to_interleaved: L1 output needs {per_bank} B per bank, over the {L1_BUDGET_BYTES} B budget"
            )

    lane_blocks = [[None] * len(cores) for _ in range(num_write_lanes)]
    for i, block in enumerate(blocks):
        if block is None:
            continue
        for lane, piece in enumerate(_split_for_lanes(block, num_write_lanes)):
            lane_blocks[lane][i] = piece

    return Plan(
        tensor_tile_rows=tensor_tile_rows,
        tensor_tile_cols=tensor_tile_cols,
        shard_tile_rows=shard_tile_rows,
        shard_tile_cols=shard_tile_cols,
        page_size=page_size,
        num_write_lanes=num_write_lanes,
        cores=cores,
        blocks=tuple(blocks),
        lane_blocks=tuple(tuple(lb) for lb in lane_blocks),
    )


def _lane_runtime_args(plan, lane):
    """Per-core RT args of one lane: [num_blocks, first_slot, first_page, block_tile_rows, block_tile_cols]."""
    rt_args = ttnn.RuntimeArgs()
    for i, (x, y) in enumerate(plan.cores):
        piece = plan.lane_blocks[lane][i]
        if piece is None:
            rt_args[x][y] = [0, 0, 0, 0, 0]
            continue
        shard = plan.blocks[i]
        first_slot = (piece.first_row - shard.first_row) * plan.shard_tile_cols + (piece.first_col - shard.first_col)
        first_page = piece.first_row * plan.tensor_tile_cols + piece.first_col
        rt_args[x][y] = [1, first_slot, first_page, piece.rows, piece.cols]
    return rt_args


def create_program_descriptor(input_tensor, output_tensor, plan):
    if output_tensor.buffer_aligned_page_size() != plan.page_size:
        raise RuntimeError(
            f"g6_sharded_to_interleaved: output page {output_tensor.buffer_aligned_page_size()} B != {plan.page_size} B"
        )

    cb_input_shard = ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT_SHARD, input_tensor)
    grid = input_tensor.memory_config().shard_spec.grid

    ct_args = [CB_INPUT_SHARD, plan.page_size]
    ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    common_rt_args = [output_tensor.buffer_address(), plan.tensor_tile_cols, plan.shard_tile_cols]

    kernels = []
    for lane in range(plan.num_write_lanes):
        processor, noc = _LANE_PLACEMENT[lane]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=WRITER_KERNEL,
                core_ranges=grid,
                compile_time_args=ct_args,
                runtime_args=_lane_runtime_args(plan, lane),
                common_runtime_args=common_rt_args,
                config=ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc),
            )
        )

    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb_input_shard])


def patch_program_descriptor(descriptor, input_tensor, output_tensor):
    """Point a memoized descriptor at this call's input shard buffer and output address.

    The program-cache hash covers neither, and a cache hit re-applies both from the descriptor
    passed in, so every per-call value must be refreshed here before dispatch.
    """
    descriptor.cbs[0].set_buffer_from_cb(ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT_SHARD, input_tensor))
    output_addr = output_tensor.buffer_address()
    for kernel in descriptor.kernels:
        kernel.common_runtime_args[_COMMON_RT_OUTPUT_ADDR] = output_addr
