# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_reshard: move every tile of a BLOCK_SHARDED L1 tensor, as stored, into a new
BLOCK_SHARDED L1 tensor on the placement memory_config names. One device program."""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_reshard_program_descriptor import Placement, get_plan, tensor_tiles


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — no shape-derived axis: placements are arguments, not axes.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = {}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat8_b],
    "layout": [ttnn.TILE_LAYOUT],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(input_tensor, memory_config=None):
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((input_tensor.shape,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_reshard: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_reshard: unsupported combination (refinement candidate): {exc}")


# The L1 per core the model leaves to the op; out's shard is charged against it (BB-05).
L1_FREE_BYTES = 1347776


def _check_block_sharded_l1(memory_config, what):
    if memory_config.memory_layout != ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        raise ValueError(f"g6_reshard: {what} must be BLOCK_SHARDED, got {memory_config.memory_layout}")
    if memory_config.buffer_type != ttnn.BufferType.L1:
        raise ValueError(f"g6_reshard: {what} must be in L1, got {memory_config.buffer_type}")
    if memory_config.shard_spec is None:
        raise ValueError(f"g6_reshard: {what} has no shard spec")


def _check_inputs(input_tensor, memory_config):
    if not isinstance(memory_config, ttnn.MemoryConfig):
        raise ValueError(f"g6_reshard: memory_config must be a ttnn.MemoryConfig, got {type(memory_config)}")
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("g6_reshard: input_tensor must be on device")
    _check_block_sharded_l1(input_tensor.memory_config(), "input_tensor")
    _check_block_sharded_l1(memory_config, "memory_config")

    shape = list(input_tensor.shape)
    if len(shape) != 4:
        raise ValueError(f"g6_reshard: input_tensor must be rank 4, got shape {shape}")
    height = shape[0] * shape[1] * shape[2]
    if height % 32 or shape[3] % 32:
        raise ValueError(f"g6_reshard: 2-D view ({height}, {shape[3]}) must have H and W multiples of 32")

    spec = memory_config.shard_spec
    shard_h, shard_w = spec.shape[0], spec.shape[1]
    if shard_h <= 0 or shard_w <= 0 or shard_h % 32 or shard_w % 32:
        raise ValueError(f"g6_reshard: memory_config shard shape ({shard_h}, {shard_w}) must be multiples of 32")

    ranges = spec.grid.ranges()
    if len(ranges) != 1:
        raise ValueError(f"g6_reshard: memory_config grid must be one core rectangle, got {len(ranges)} ranges")
    grid = ranges[0]
    size = input_tensor.device().compute_with_storage_grid_size()
    if grid.end.x >= size.x or grid.end.y >= size.y:
        raise ValueError(
            f"g6_reshard: memory_config grid ({grid.start.x},{grid.start.y})-({grid.end.x},{grid.end.y}) "
            f"outside the {size.x}x{size.y} worker grid"
        )

    out_placement = Placement.of(memory_config)
    num_block_rows, num_block_cols = out_placement.num_blocks(*tensor_tiles(input_tensor))
    grid_cols = grid.end.x - grid.start.x + 1
    grid_rows = grid.end.y - grid.start.y + 1
    need_cols, need_rows = (
        (num_block_cols, num_block_rows) if out_placement.row_major else (num_block_rows, num_block_cols)
    )
    if grid_cols < need_cols or grid_rows < need_rows:
        raise ValueError(
            f"g6_reshard: memory_config grid {grid_cols}x{grid_rows} cores too small for "
            f"{num_block_rows}x{num_block_cols} blocks ({spec.orientation})"
        )

    out_shard_bytes = (
        out_placement.block_height_tiles * out_placement.block_width_tiles * input_tensor.buffer_aligned_page_size()
    )
    if out_shard_bytes > L1_FREE_BYTES:
        raise RuntimeError(
            f"g6_reshard: out shard of {out_shard_bytes} B exceeds the {L1_FREE_BYTES} B of L1 left to the op"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_reshard(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    validate(input_tensor, memory_config)
    _check_inputs(input_tensor, memory_config)

    output_tensor = ttnn.allocate_tensor_on_device(
        input_tensor.shape,
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        memory_config,
    )
    plan = get_plan(input_tensor, output_tensor, memory_config)
    program_descriptor = plan.bind(input_tensor, output_tensor)
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
