# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_sharded_to_interleaved: copy a height- or width-sharded L1 TILE tensor, bit for bit, into a
newly allocated interleaved TILE tensor in the buffer type `memory_config` names.

One device program: each shard-holding core writes the tiles of its own resident shard straight
to their interleaved output pages (one NoC crossing per tile). Design: op_design.md.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_sharded_to_interleaved_program_descriptor import TILE, get_program_descriptor


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — each reads its axis off the scenario dict
#    {"shape", "dtype", "layout", "in_scheme", "shard_shape", "grid", "orientation", "buffer"}
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


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------

_IN_SCHEME = {
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: "height_sharded",
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: "width_sharded",
}
_ORIENTATION = {
    ttnn.ShardOrientation.ROW_MAJOR: "row_major",
    ttnn.ShardOrientation.COL_MAJOR: "column_major",
}
_BUFFER = {
    ttnn.BufferType.DRAM: "dram",
    ttnn.BufferType.L1: "l1",
}

# L1 per core the model leaves to the op (BB-05); the output's own L1 pages come out of it.
L1_BUDGET_BYTES_PER_CORE = 1347776

_L1_BANKS_BY_DEVICE: dict[int, int] = {}


def _num_l1_banks(device) -> int:
    key = id(device)
    if key not in _L1_BANKS_BY_DEVICE:
        _L1_BANKS_BY_DEVICE[key] = ttnn.get_memory_view(device, ttnn.BufferType.L1).num_banks
    return _L1_BANKS_BY_DEVICE[key]


def _check_structure(input_tensor, memory_config):
    name = "g6_sharded_to_interleaved"
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError(f"{name}: input_tensor must be on a device")
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"{name}: input_tensor must be in TILE_LAYOUT, got {input_tensor.layout}")
    rank = len(input_tensor.shape)
    if rank not in (2, 4):
        raise ValueError(f"{name}: input rank must be 2 or 4, got {rank}")
    in_mc = input_tensor.memory_config()
    if not in_mc.is_sharded() or in_mc.shard_spec is None:
        raise ValueError(f"{name}: input_tensor must be sharded, got {in_mc.memory_layout}")
    if in_mc.buffer_type != ttnn.BufferType.L1:
        raise ValueError(f"{name}: input_tensor must be sharded in L1, got {in_mc.buffer_type}")
    if in_mc.memory_layout not in _IN_SCHEME:
        raise ValueError(f"{name}: input must be HEIGHT_SHARDED or WIDTH_SHARDED, got {in_mc.memory_layout}")
    shard_h, shard_w = in_mc.shard_spec.shape
    if shard_h % TILE or shard_w % TILE:
        raise ValueError(f"{name}: shard shape ({shard_h}, {shard_w}) is not a whole number of {TILE}x{TILE} tiles")
    if memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError(f"{name}: memory_config must be INTERLEAVED, got {memory_config.memory_layout}")


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
        "buffer": _BUFFER.get(memory_config.buffer_type, str(memory_config.buffer_type)),
    }


def validate(input_tensor, memory_config):
    _check_structure(input_tensor, memory_config)

    inputs = (_scenario(input_tensor, memory_config),)
    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_sharded_to_interleaved: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_sharded_to_interleaved: unsupported combination (refinement candidate): {exc}")

    if memory_config.buffer_type == ttnn.BufferType.L1:
        num_banks = _num_l1_banks(input_tensor.device())
        page_bytes = ttnn.round_up(input_tensor.buffer_page_size(), ttnn.get_l1_alignment())
        per_bank = -(-input_tensor.buffer_num_pages() // num_banks) * page_bytes
        if per_bank > L1_BUDGET_BYTES_PER_CORE:
            raise RuntimeError(
                f"g6_sharded_to_interleaved: L1 output needs {per_bank} B per bank, "
                f"above the {L1_BUDGET_BYTES_PER_CORE} B the model leaves per core"
            )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_sharded_to_interleaved(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    validate(input_tensor, memory_config)
    output_tensor = ttnn.allocate_tensor_on_device(
        input_tensor.shape,
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        memory_config,
    )
    return _launch(input_tensor, output_tensor)


def _launch(input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor) -> ttnn.Tensor:
    """Dispatch the op's one program into a preallocated interleaved output."""
    return ttnn.generic_op([input_tensor, output_tensor], get_program_descriptor(input_tensor, output_tensor))
