# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_sharded_to_interleaved: a height- or width-sharded L1 TILE tensor copied, bit for bit, into a new
interleaved tensor in DRAM or L1 (as `memory_config` names), in one device program.

Each shard core writes its own resident tiles straight to their output pages (`owner_writes`): every
tile crosses the NoC once and nothing is read over the NoC. See op_design.md.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_sharded_to_interleaved_program_descriptor import DEFAULT_KNOBS, build_plan, cached_plan, get_program

TILE = 32


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — `inputs` is (scenario,), scenario as built in validate() / feature_spec.py
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


def tag_in_scheme(inputs, axes):
    return inputs[0]["in_scheme"]


def tag_buffer(inputs, axes):
    return inputs[0]["buffer"]


INPUT_TAGGERS = {
    "dtype": tag_dtype,
    "layout": tag_layout,
    "in_scheme": tag_in_scheme,
    "buffer": tag_buffer,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "in_scheme": ["width_sharded"],
    "buffer": ["dram"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------

_IN_SCHEME = {
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: "width_sharded",
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: "height_sharded",
}
_BUFFER = {
    ttnn.BufferType.DRAM: "dram",
    ttnn.BufferType.L1: "l1",
}
_ORIENTATION = {
    ttnn.ShardOrientation.ROW_MAJOR: "row_major",
    ttnn.ShardOrientation.COL_MAJOR: "column_major",
}


def _scenario(input_tensor, memory_config):
    spec = input_tensor.memory_config().shard_spec
    return {
        "shape": tuple(input_tensor.shape),
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "in_scheme": _IN_SCHEME[input_tensor.memory_config().memory_layout],
        "shard_shape": tuple(spec.shape),
        "grid": tuple(((r.start.x, r.start.y), (r.end.x, r.end.y)) for r in spec.grid.ranges()),
        "orientation": _ORIENTATION[spec.orientation],
        "buffer": _BUFFER[memory_config.buffer_type],
    }


def _validate_structure(input_tensor, memory_config):
    # Structural errors are ValueError, never a support refusal, so they precede the SUPPORTED gate.
    if not isinstance(input_tensor, ttnn.Tensor) or not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("g6_sharded_to_interleaved: input_tensor must be a ttnn.Tensor stored on a device")
    rank = len(input_tensor.shape)
    if rank not in (2, 4):
        raise ValueError(f"g6_sharded_to_interleaved: input rank must be 2 or 4, got {rank}")
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"g6_sharded_to_interleaved: input must be TILE_LAYOUT, got {input_tensor.layout}")
    if tuple(input_tensor.tile.tile_shape) != (TILE, TILE):
        raise ValueError(f"g6_sharded_to_interleaved: input tiles must be 32x32, got {input_tensor.tile.tile_shape}")
    in_mc = input_tensor.memory_config()
    if not in_mc.is_sharded() or in_mc.buffer_type != ttnn.BufferType.L1:
        raise ValueError(f"g6_sharded_to_interleaved: input must be sharded in L1, got {in_mc}")
    if in_mc.memory_layout not in _IN_SCHEME:
        raise ValueError(
            f"g6_sharded_to_interleaved: input must be HEIGHT_SHARDED or WIDTH_SHARDED, got {in_mc.memory_layout}"
        )
    shard_shape = tuple(in_mc.shard_spec.shape)
    if shard_shape[0] % TILE or shard_shape[1] % TILE:
        raise ValueError(f"g6_sharded_to_interleaved: shard shape {shard_shape} is not a whole number of tiles")
    if not isinstance(memory_config, ttnn.MemoryConfig):
        raise ValueError("g6_sharded_to_interleaved: memory_config must be a ttnn.MemoryConfig")
    if memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED or memory_config.buffer_type not in _BUFFER:
        raise ValueError(
            f"g6_sharded_to_interleaved: memory_config must be INTERLEAVED in DRAM or L1, got {memory_config}"
        )


def validate(input_tensor, memory_config):
    _validate_structure(input_tensor, memory_config)

    scenario = _scenario(input_tensor, memory_config)
    axes = {}
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_sharded_to_interleaved: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for excluded in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in excluded.items()):
            raise ExcludedCell(f"g6_sharded_to_interleaved: unsupported combination (refinement candidate): {excluded}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _dispatch(input_tensor, memory_config, knobs=DEFAULT_KNOBS):
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        memory_config,
    )
    program = get_program(input_tensor, output_tensor, memory_config, knobs)
    return ttnn.generic_op([input_tensor, output_tensor], program.bind(input_tensor, output_tensor))


def g6_sharded_to_interleaved(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    validate(input_tensor, memory_config)
    return _dispatch(input_tensor, memory_config)


def describe_plan(input_tensor, memory_config, knobs=DEFAULT_KNOBS) -> dict:
    """The plan a call with these arguments runs (BB-08, BB-21): regime, lanes, per-core role and block."""
    plan = cached_plan(input_tensor, memory_config, knobs) or build_plan(input_tensor, memory_config, knobs)
    cores = []
    for core, block in zip(plan.cores, plan.blocks):
        entry = {"core": (core.x, core.y), "role": "idle" if block is None else "writer", "block": None}
        if block is not None:
            entry["block"] = (block.first_tile_row, block.first_tile_col, block.tile_rows, block.tile_cols)
        cores.append(entry)
    return {
        "plan": "owner_writes",
        "num_write_lanes": plan.knobs.num_write_lanes,
        "block_tile_rows": plan.knobs.block_tile_rows,
        "block_tile_cols": plan.knobs.block_tile_cols,
        "tensor_tiles": (plan.tensor_tile_rows, plan.tensor_tile_cols),
        "shard_tiles": (plan.shard_tile_rows, plan.shard_tile_cols),
        "num_shards": plan.num_shards,
        "page_size": plan.page_size,
        "cores": cores,
    }
