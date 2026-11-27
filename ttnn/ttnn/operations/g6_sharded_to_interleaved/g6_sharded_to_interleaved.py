# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_sharded_to_interleaved: copy a height- or width-sharded L1 TILE tensor into a newly allocated
interleaved TILE tensor (DRAM or L1, as memory_config names), bit for bit, as one device program.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_sharded_to_interleaved_program_descriptor import check_l1_bound, create_program_descriptor, get_plan


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — each reads its axis off the scenario dict inputs[0]
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

_TILE = 32

_IN_SCHEME = {
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: "width_sharded",
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: "height_sharded",
}
_ORIENTATION = {
    ttnn.ShardOrientation.ROW_MAJOR: "row_major",
    ttnn.ShardOrientation.COL_MAJOR: "column_major",
}
_BUFFER = {
    ttnn.BufferType.DRAM: "dram",
    ttnn.BufferType.L1: "l1",
}


def _check_structure(input_tensor, memory_config):
    """Inputs the op can never take: ValueError, never a support refusal."""
    if not isinstance(input_tensor, ttnn.Tensor):
        raise ValueError(f"g6_sharded_to_interleaved: input_tensor must be a ttnn.Tensor, got {type(input_tensor)}")
    if input_tensor.storage_type() != ttnn.StorageType.DEVICE:
        raise ValueError("g6_sharded_to_interleaved: input_tensor must be on a device")
    if len(input_tensor.shape) not in (2, 4):
        raise ValueError(f"g6_sharded_to_interleaved: input rank must be 2 or 4, got {len(input_tensor.shape)}")
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"g6_sharded_to_interleaved: input layout must be TILE, got {input_tensor.layout}")
    if tuple(input_tensor.tile.tile_shape) != (_TILE, _TILE):
        raise ValueError(f"g6_sharded_to_interleaved: tile shape must be 32x32, got {input_tensor.tile.tile_shape}")
    if not input_tensor.is_sharded():
        raise ValueError("g6_sharded_to_interleaved: input_tensor must be sharded")
    in_mc = input_tensor.memory_config()
    if in_mc.buffer_type != ttnn.BufferType.L1:
        raise ValueError(f"g6_sharded_to_interleaved: input must be in L1, got {in_mc.buffer_type}")
    if in_mc.memory_layout not in _IN_SCHEME:
        raise ValueError(
            f"g6_sharded_to_interleaved: input must be HEIGHT_SHARDED or WIDTH_SHARDED, got {in_mc.memory_layout}"
        )
    shard_shape = tuple(in_mc.shard_spec.shape)
    if shard_shape[0] % _TILE or shard_shape[1] % _TILE:
        raise ValueError(f"g6_sharded_to_interleaved: shard shape {shard_shape} is not a whole number of tiles")
    if not isinstance(memory_config, ttnn.MemoryConfig):
        raise ValueError(f"g6_sharded_to_interleaved: memory_config must be a ttnn.MemoryConfig, got {memory_config}")
    if memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError(
            f"g6_sharded_to_interleaved: memory_config must be INTERLEAVED, got {memory_config.memory_layout}"
        )
    if memory_config.buffer_type not in _BUFFER:
        raise ValueError(
            f"g6_sharded_to_interleaved: memory_config buffer type must be DRAM or L1, got {memory_config.buffer_type}"
        )


def _scenario(input_tensor, memory_config):
    in_mc = input_tensor.memory_config()
    spec = in_mc.shard_spec
    return {
        "shape": tuple(input_tensor.shape),
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "in_scheme": _IN_SCHEME[in_mc.memory_layout],
        "shard_shape": tuple(spec.shape),
        "grid": tuple(((r.start.x, r.start.y), (r.end.x, r.end.y)) for r in spec.grid.ranges()),
        "orientation": _ORIENTATION[spec.orientation],
        "buffer": _BUFFER[memory_config.buffer_type],
    }


def validate(input_tensor, memory_config):
    _check_structure(input_tensor, memory_config)

    inputs = (_scenario(input_tensor, memory_config),)
    axes = {}
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger(inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_sharded_to_interleaved: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_sharded_to_interleaved: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_sharded_to_interleaved(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    validate(input_tensor, memory_config)
    check_l1_bound(get_plan(input_tensor, memory_config))

    # New storage on every call (and on every device of a mesh); the input is never aliased.
    output_tensor = ttnn.allocate_tensor_on_device(
        input_tensor.shape,
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        memory_config,
    )
    program_descriptor = create_program_descriptor(input_tensor, output_tensor, memory_config)
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)


def describe_plan(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> dict:
    """The plan a call with these arguments selects: regime and, per core, its role and blocks."""
    validate(input_tensor, memory_config)
    plan = get_plan(input_tensor, memory_config)
    cores = []
    for core_plan in plan.cores:
        blocks = [
            {"riscv": riscv, "block": (b.origin_row, b.origin_col, b.rows, b.cols)}
            for riscv, riscv_blocks in enumerate(core_plan.blocks_per_riscv)
            for b in riscv_blocks
        ]
        cores.append(
            {
                "core": core_plan.core,
                "role": "writer" if blocks else "idle",
                "shard": core_plan.shard_index,
                "blocks": blocks,
            }
        )
    return {
        "regime": plan.regime,
        "num_writer_riscs": plan.num_writer_riscs,
        "tensor_tiles": (plan.tensor_tile_rows, plan.tensor_tile_cols),
        "shard_tiles": (plan.shard_tile_rows, plan.shard_tile_cols),
        "num_shards": plan.num_shards,
        "cores": cores,
    }
