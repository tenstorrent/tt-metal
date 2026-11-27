# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_reshard: move every tile of a BLOCK_SHARDED L1 tensor, as stored, into a new tensor
BLOCK_SHARDED in L1 on the placement `memory_config` names, in one device program."""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_reshard_program_descriptor import build_plan, create_program, plan_report, role_dump

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

PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
}

# L1 per core the model leaves free to the op; out's shard is the op's only allocation.
L1_FREE_BYTES = 1347776


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
# Host validation of the placements (legal-input checks, not support refusals)
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
        raise ValueError(f"g6_reshard: input_tensor must be rank 4, got shape {shape}")
    tile_h, tile_w = input_tensor.tile.tile_shape
    height = shape[0] * shape[1] * shape[2]
    if height % tile_h != 0 or shape[3] % tile_w != 0:
        raise ValueError(f"g6_reshard: 2-D view ({height}, {shape[3]}) must be a multiple of ({tile_h}, {tile_w})")

    spec = memory_config.shard_spec
    shard_h, shard_w = int(spec.shape[0]), int(spec.shape[1])
    if shard_h <= 0 or shard_w <= 0 or shard_h % tile_h != 0 or shard_w % tile_w != 0:
        raise ValueError(f"g6_reshard: memory_config shard {[shard_h, shard_w]} must be multiples of the tile")
    if len(spec.grid.ranges()) != 1:
        raise ValueError("g6_reshard: memory_config core grid must be one rectangle")

    grid = spec.grid.ranges()[0]
    worker_grid = input_tensor.device().compute_with_storage_grid_size()
    if grid.end.x >= worker_grid.x or grid.end.y >= worker_grid.y:
        raise ValueError(
            f"g6_reshard: memory_config grid ends at ({grid.end.x}, {grid.end.y}), "
            f"outside the {worker_grid.x}x{worker_grid.y} worker grid"
        )


def _check_plan(plan, memory_config, aligned_page_bytes):
    grid = memory_config.shard_spec.grid.ranges()[0]
    grid_cols = grid.end.x - grid.start.x + 1
    grid_rows = grid.end.y - grid.start.y + 1
    need_cols, need_rows = plan.num_block_cols, plan.num_block_rows
    if not plan.dst.row_major:
        need_cols, need_rows = need_rows, need_cols
    if grid_cols < need_cols or grid_rows < need_rows:
        raise ValueError(
            f"g6_reshard: memory_config grid {grid_cols}x{grid_rows} cannot hold "
            f"{plan.num_block_rows}x{plan.num_block_cols} blocks in its orientation"
        )
    out_shard_bytes = plan.dst.block_height_tiles * plan.dst.block_width_tiles * aligned_page_bytes
    if out_shard_bytes > L1_FREE_BYTES:
        raise RuntimeError(
            f"g6_reshard: out shard needs {out_shard_bytes} B of L1 per core, above the {L1_FREE_BYTES} B left free"
        )


# ---------------------------------------------------------------------------
# Host plan cache (BB-07): one built program per placement pair; a hit rewrites the addresses only.
# ---------------------------------------------------------------------------

_PROGRAM_CACHE = {}
_LAST_PLAN = None


def _placement_key(memory_config):
    spec = memory_config.shard_spec
    ranges = tuple(sorted((r.start.x, r.start.y, r.end.x, r.end.y) for r in spec.grid.ranges()))
    return ranges, tuple(int(d) for d in spec.shape), str(spec.orientation)


def _cache_key(input_tensor, memory_config):
    return (
        tuple(int(d) for d in input_tensor.padded_shape),
        str(input_tensor.dtype),
        str(input_tensor.layout),
        _placement_key(input_tensor.memory_config()),
        _placement_key(memory_config),
        input_tensor.device().id(),
    )


def g6_reshard_last_plan():
    """The plan the last call dispatched (BB-08)."""
    return _LAST_PLAN


def g6_reshard_role_dump(input_tensor, memory_config):
    """Per core of the kernel range: its out block and each issuer's rows and piece count (BB-21)."""
    return role_dump(build_plan(input_tensor, memory_config))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_reshard(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    validate(input_tensor, memory_config)
    _check_inputs(input_tensor, memory_config)

    aligned_page_bytes = input_tensor.buffer_aligned_page_size()
    key = _cache_key(input_tensor, memory_config)
    cached = _PROGRAM_CACHE.get(key)
    if cached is None:
        plan = build_plan(input_tensor, memory_config)
        _check_plan(plan, memory_config, aligned_page_bytes)
        cached = create_program(input_tensor, plan, aligned_page_bytes, (key, plan.key()))
        _PROGRAM_CACHE[key] = cached

    device = input_tensor.device()
    output_tensor = ttnn.allocate_tensor_on_device(
        input_tensor.shape,
        input_tensor.dtype,
        input_tensor.layout,
        device,
        memory_config,
    )
    if output_tensor.buffer_aligned_page_size() != aligned_page_bytes:
        raise RuntimeError(
            f"g6_reshard: out page stride {output_tensor.buffer_aligned_page_size()} B "
            f"differs from input_tensor's {aligned_page_bytes} B"
        )

    global _LAST_PLAN
    _LAST_PLAN = plan_report(cached.plan)

    program_descriptor = cached.bind(input_tensor.buffer_address(), output_tensor.buffer_address())
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
