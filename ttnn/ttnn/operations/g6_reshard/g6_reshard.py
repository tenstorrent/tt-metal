# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_reshard: move every tile of a BLOCK_SHARDED L1 tensor, as stored, into a new tensor
BLOCK_SHARDED in L1 on the placement `memory_config` names. One device program per call."""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_reshard_program_descriptor import (
    L1_FREE_BYTES,
    block_grid,
    create_program_descriptor,
    last_plan,
    role_dump,
    tensor_tiles,
)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — no shape-derived axis; placements are arguments, not axes.
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


# ---------------------------------------------------------------------------
# Input validation (legality, not support)
# ---------------------------------------------------------------------------


def _is_block_sharded_l1(memory_config) -> bool:
    return (
        memory_config.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
        and memory_config.buffer_type == ttnn.BufferType.L1
        and memory_config.shard_spec is not None
    )


def _check_inputs(input_tensor, memory_config):
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("g6_reshard: input_tensor must be on a device")
    if not _is_block_sharded_l1(input_tensor.memory_config()):
        raise ValueError("g6_reshard: input_tensor must be BLOCK_SHARDED in L1 with a shard spec")
    if not isinstance(memory_config, ttnn.MemoryConfig) or not _is_block_sharded_l1(memory_config):
        raise ValueError("g6_reshard: memory_config must be BLOCK_SHARDED in L1 with a shard spec")

    shape = list(input_tensor.shape)
    if len(shape) != 4:
        raise ValueError(f"g6_reshard: input_tensor must have rank 4, got shape {shape}")
    height = shape[0] * shape[1] * shape[2]
    if height % 32 != 0 or shape[3] % 32 != 0:
        raise ValueError(f"g6_reshard: 2-D view ({height}, {shape[3]}) must have H and W multiples of 32")

    spec = memory_config.shard_spec
    shard_h, shard_w = spec.shape
    tile_h, tile_w = input_tensor.tile.tile_shape
    if shard_h <= 0 or shard_w <= 0 or shard_h % 32 or shard_w % 32 or shard_h % tile_h or shard_w % tile_w:
        raise ValueError(f"g6_reshard: memory_config shard shape {[shard_h, shard_w]} must be multiples of 32")

    ranges = spec.grid.ranges()
    if len(ranges) != 1:
        raise ValueError("g6_reshard: memory_config core grid must be one rectangle")
    (rng,) = ranges
    grid_cols = rng.end.x - rng.start.x + 1
    grid_rows = rng.end.y - rng.start.y + 1

    worker = input_tensor.device().compute_with_storage_grid_size()
    if rng.end.x >= worker.x or rng.end.y >= worker.y:
        raise ValueError(
            f"g6_reshard: memory_config grid ({rng.start.x},{rng.start.y})-({rng.end.x},{rng.end.y}) "
            f"outside the {worker.x}x{worker.y} worker grid"
        )

    tensor_height_tiles, tensor_width_tiles = tensor_tiles(input_tensor)
    num_block_rows = block_grid(tensor_height_tiles, shard_h // tile_h)
    num_block_cols = block_grid(tensor_width_tiles, shard_w // tile_w)
    if spec.orientation == ttnn.ShardOrientation.ROW_MAJOR:
        need_cols, need_rows = num_block_cols, num_block_rows
    else:
        need_cols, need_rows = num_block_rows, num_block_cols
    if grid_cols < need_cols or grid_rows < need_rows:
        raise ValueError(
            f"g6_reshard: memory_config grid {grid_cols}x{grid_rows} cannot hold "
            f"{num_block_rows}x{num_block_cols} blocks in {spec.orientation}"
        )

    out_shard_bytes = (shard_h // tile_h) * (shard_w // tile_w) * input_tensor.buffer_aligned_page_size()
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
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        memory_config,
    )
    if output_tensor.buffer_aligned_page_size() != input_tensor.buffer_aligned_page_size():
        raise RuntimeError("g6_reshard: input and output tile slots differ in aligned size")

    program_descriptor = create_program_descriptor(input_tensor, output_tensor, memory_config)
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)


__all__ = [
    "INPUT_TAGGERS",
    "SUPPORTED",
    "EXCLUSIONS",
    "validate",
    "g6_reshard",
    "last_plan",
    "role_dump",
]
