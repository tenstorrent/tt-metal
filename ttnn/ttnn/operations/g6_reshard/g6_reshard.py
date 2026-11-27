# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_reshard: move every tile of a BLOCK_SHARDED L1 tensor, as stored, into a
new tensor BLOCK_SHARDED in L1 on the placement `memory_config` names.

One `ttnn.generic_op` dispatch per call (PULL: each out-block holder reads its
pieces of in0 straight into its own out shard). See op_design.md.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_reshard_program_descriptor import (
    L1_FREE_BYTES,
    describe_plan,
    get_program_descriptor,
    placement_of,
)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — placements are arguments, not axes: no shape-derived axis.
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


def validate(input_tensor, memory_config):
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
# Input validation (after validate(), before any device work)
# ---------------------------------------------------------------------------


def _check_block_sharded_l1(memory_config, what):
    if memory_config.memory_layout != ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        raise ValueError(f"g6_reshard: {what} must be BLOCK_SHARDED, got {memory_config.memory_layout}")
    if memory_config.buffer_type != ttnn.BufferType.L1:
        raise ValueError(f"g6_reshard: {what} must live in L1, got {memory_config.buffer_type}")
    if memory_config.shard_spec is None:
        raise ValueError(f"g6_reshard: {what} has no shard spec")


def _check_inputs(input_tensor, memory_config):
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("g6_reshard: input_tensor must be on a device")
    _check_block_sharded_l1(input_tensor.memory_config(), "input_tensor")
    _check_block_sharded_l1(memory_config, "memory_config")

    shape = list(input_tensor.shape)
    if len(shape) != 4:
        raise ValueError(f"g6_reshard: input_tensor must be rank 4, got shape {shape}")
    height = shape[0] * shape[1] * shape[2]
    width = shape[3]
    if height % 32 != 0 or width % 32 != 0:
        raise ValueError(f"g6_reshard: the 2-D view ({height}, {width}) must have H and W multiples of 32")

    shard_h, shard_w = (int(d) for d in memory_config.shard_spec.shape)
    if shard_h <= 0 or shard_w <= 0 or shard_h % 32 != 0 or shard_w % 32 != 0:
        raise ValueError(f"g6_reshard: memory_config shard shape ({shard_h}, {shard_w}) must be multiples of 32")

    grid = memory_config.shard_spec.grid
    box = grid.bounding_box()
    box_cores = (box.end.x - box.start.x + 1) * (box.end.y - box.start.y + 1)
    if grid.num_cores() != box_cores:
        raise ValueError(f"g6_reshard: memory_config core grid {grid} is not one rectangle")

    worker = input_tensor.device().compute_with_storage_grid_size()
    if box.end.x >= worker.x or box.end.y >= worker.y:
        raise ValueError(
            f"g6_reshard: memory_config core grid {grid} lies outside the {worker.x}x{worker.y} worker grid"
        )

    out = placement_of(input_tensor, memory_config)
    grid_cols = box.end.x - box.start.x + 1
    grid_rows = box.end.y - box.start.y + 1
    if out.orientation == ttnn.ShardOrientation.ROW_MAJOR:
        need_cols, need_rows = out.num_block_cols, out.num_block_rows
    else:
        need_cols, need_rows = out.num_block_rows, out.num_block_cols
    if grid_cols < need_cols or grid_rows < need_rows:
        raise ValueError(
            f"g6_reshard: memory_config grid {grid_cols}x{grid_rows} (cols x rows) cannot hold "
            f"{out.num_block_rows}x{out.num_block_cols} blocks in {out.orientation}"
        )

    if input_tensor.buffer_aligned_page_size() * out.block_height_tiles * out.block_width_tiles > L1_FREE_BYTES:
        raise RuntimeError(
            f"g6_reshard: an out shard of {out.block_height_tiles}x{out.block_width_tiles} tiles exceeds "
            f"the {L1_FREE_BYTES} B of L1 the model leaves free"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_last_plan = None


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
    if output_tensor.buffer_aligned_page_size() != input_tensor.buffer_aligned_page_size():
        raise RuntimeError("g6_reshard: input and output tile slots differ in size")

    global _last_plan
    program_descriptor, _last_plan = get_program_descriptor(input_tensor, output_tensor, memory_config)
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)


def last_plan():
    """Plan of the last call (BB-08): regime and each issuer's rectangle per core."""
    return _last_plan


def dump_roles(input_tensor, memory_config):
    """Per-core role of the kernel range (BB-21): out block, issuer row ranges, piece counts."""
    return describe_plan(input_tensor, memory_config)["roles"]
