# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_interleaved_to_sharded: move a DRAM-interleaved bfloat16 TILE tensor, bit for bit, into the
L1 height or width shards that memory_config names, in one device program (owner-fetch).

Registry model: INPUT_TAGGERS, SUPPORTED, EXCLUSIONS and validate() are declared here; INVALID
lives in eval/golden_tests/g6_interleaved_to_sharded/feature_spec.py.
"""

from __future__ import annotations

import copy

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from . import g6_interleaved_to_sharded_program_descriptor as _descriptor


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — each reads inputs[0], the scenario dict of one call.
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


def tag_rank(inputs, axes):
    return int(len(inputs[0]["shape"]))


def tag_alignment(inputs, axes):
    shape = inputs[0]["shape"]
    H, W = shape[-2], shape[-1]
    if H % 32 == 0 and W % 32 == 0:
        return "tile_aligned"
    if W % 32 != 0:
        return "w_non_aligned"
    return "h_non_aligned"


def tag_out_scheme(inputs, axes):
    return inputs[0]["out_memory"]["scheme"]


INPUT_TAGGERS = {
    "dtype": tag_dtype,
    "layout": tag_layout,
    "rank": tag_rank,
    "alignment": tag_alignment,
    "out_scheme": tag_out_scheme,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "rank": [2],
    "alignment": ["h_non_aligned"],
    "out_scheme": ["width_sharded"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------

_SCHEME_NAME = {
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: "height_sharded",
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: "width_sharded",
}
_ORIENTATION_NAME = {
    ttnn.ShardOrientation.ROW_MAJOR: "row_major",
    ttnn.ShardOrientation.COL_MAJOR: "col_major",
}


def _refuse(message):
    raise ValueError(f"g6_interleaved_to_sharded: {message}")


def _check_malformed(input_tensor, memory_config):
    """Malformed requests (ValueError), checked before any support decision."""
    if not isinstance(input_tensor, ttnn.Tensor):
        _refuse(f"input_tensor must be a ttnn.Tensor, got {type(input_tensor).__name__}")
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        _refuse("input_tensor is not on a device")
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        _refuse(f"input_tensor must be in TILE_LAYOUT, got {input_tensor.layout}")
    if tuple(input_tensor.tile.tile_shape) != (32, 32) or tuple(input_tensor.tile.face_shape) != (16, 16):
        _refuse(f"input tile must be 32x32 with 16x16 faces, got {input_tensor.tile.tile_shape}")
    in_mc = input_tensor.memory_config()
    if in_mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        _refuse(f"input_tensor must be interleaved, got {in_mc.memory_layout}")
    if in_mc.buffer_type != ttnn.BufferType.DRAM:
        _refuse(f"input_tensor must be in DRAM, got {in_mc.buffer_type}")
    if input_tensor.dtype != ttnn.bfloat16:
        _refuse(f"dtype must be bfloat16, got {input_tensor.dtype}")
    rank = len(input_tensor.shape)
    if rank not in (2, 4):
        _refuse(f"rank must be 2 or 4, got {rank}")

    if not isinstance(memory_config, ttnn.MemoryConfig):
        _refuse(f"memory_config must be a ttnn.MemoryConfig, got {type(memory_config).__name__}")
    if memory_config.buffer_type != ttnn.BufferType.L1:
        _refuse(f"memory_config must name L1, got {memory_config.buffer_type}")
    if memory_config.memory_layout not in _SCHEME_NAME:
        _refuse(f"memory_config must be HEIGHT_SHARDED or WIDTH_SHARDED, got {memory_config.memory_layout}")
    shard_spec = memory_config.shard_spec
    if shard_spec is None:
        _refuse("memory_config has no shard_spec")
    shard_h, shard_w = shard_spec.shape
    tile_h, tile_w = input_tensor.tile.tile_shape
    if shard_h <= 0 or shard_w <= 0 or shard_h % tile_h or shard_w % tile_w:
        _refuse(f"shard shape {list(shard_spec.shape)} is not a positive multiple of the {tile_h}x{tile_w} tile")

    geom = _descriptor.compute_geometry(input_tensor, memory_config)
    # A shorter/narrower shard leaves tiles without an owner; a taller/wider one cannot be
    # allocated at all (TensorSpec requires equality on the unsharded dim).
    if geom.scheme == _descriptor.WIDTH and geom.shard_tile_rows != geom.tensor_tile_rows:
        _refuse(
            f"width shard of {geom.shard_tile_rows} tile rows does not match the tensor's "
            f"{geom.tensor_tile_rows} tile rows"
        )
    if geom.scheme == _descriptor.HEIGHT and geom.shard_tile_cols != geom.tensor_tile_cols:
        _refuse(
            f"height shard of {geom.shard_tile_cols} tile columns does not match the tensor's "
            f"{geom.tensor_tile_cols} tile columns"
        )
    if geom.num_shards > len(geom.cores):
        _refuse(f"{geom.num_shards} shards need more cores than the grid's {len(geom.cores)}")
    shard_bytes = geom.shard_tile_rows * geom.shard_tile_cols * geom.tile_bytes
    if shard_bytes > _descriptor.L1_BUDGET_BYTES:
        _refuse(f"shard of {shard_bytes} B exceeds the per-core L1 budget of {_descriptor.L1_BUDGET_BYTES} B")


def _scenario(input_tensor, memory_config):
    """The live call as the scenario dict the taggers read (one tagging source)."""
    shard_spec = memory_config.shard_spec
    box = shard_spec.grid.bounding_box()
    return {
        "shape": list(input_tensor.shape),
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "in_memory": {"buffer": "dram", "scheme": "interleaved"},
        "out_memory": {
            "buffer": "l1",
            "scheme": _SCHEME_NAME[memory_config.memory_layout],
            "grid": [[box.start.x, box.start.y], [box.end.x, box.end.y]],
            "shard_shape": list(shard_spec.shape),
            "orientation": _ORIENTATION_NAME[shard_spec.orientation],
        },
    }


def validate(input_tensor, memory_config):
    _check_malformed(input_tensor, memory_config)

    inputs = (_scenario(input_tensor, memory_config),)
    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_interleaved_to_sharded: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for cell in EXCLUSIONS:
        if all(axes.get(key) == value for key, value in cell.items()):
            raise ExcludedCell(f"g6_interleaved_to_sharded: unsupported combination (refinement candidate): {cell}")


# ---------------------------------------------------------------------------
# Plan report and public entry point
# ---------------------------------------------------------------------------


def describe_plan(input_tensor, memory_config) -> dict:
    """The plan a call with these arguments runs (BB-08, BB-21), from the same cached planning step."""
    validate(input_tensor, memory_config)
    # A copy: the cached plan must not change under a caller that edits the report.
    return copy.deepcopy(_descriptor.get_program(input_tensor, memory_config).plan)


def g6_interleaved_to_sharded(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    validate(input_tensor, memory_config)

    cached = _descriptor.get_program(input_tensor, memory_config)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        memory_config,
    )
    program = _descriptor.create_program_descriptor(cached, input_tensor, output_tensor)
    return ttnn.generic_op([input_tensor, output_tensor], program)
