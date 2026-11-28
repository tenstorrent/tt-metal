# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_sharded_to_interleaved — move a height/width-sharded L1 TILE tensor, bit for bit,
into a newly allocated interleaved TILE tensor in the buffer type ``memory_config`` names.

One device program per call (regime ``owner_writes``): every shard core writes its own
resident tiles straight to their interleaved output pages; each tile crosses the NoC once.
"""

from __future__ import annotations

import hashlib

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_sharded_to_interleaved_program_descriptor import (
    DEFAULT_KNOBS,
    TILE,
    Knobs,
    bind_call,
    build_program_descriptor,
    core_roles,
)

_OP = "g6_sharded_to_interleaved"


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — inputs is (scenario,)
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
# 2. SUPPORTED (Phase 0)
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "in_scheme": ["width_sharded"],
    "buffer": ["dram"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS — none
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------

_SCHEME_NAME = {
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: "width_sharded",
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: "height_sharded",
}
_BUFFER_NAME = {ttnn.BufferType.DRAM: "dram", ttnn.BufferType.L1: "l1"}
_ORIENT_NAME = {ttnn.ShardOrientation.ROW_MAJOR: "row_major", ttnn.ShardOrientation.COL_MAJOR: "column_major"}


def _grid_tuple(grid):
    return tuple(((r.start.x, r.start.y), (r.end.x, r.end.y)) for r in grid.ranges())


def _structural_checks(input_tensor, memory_config):
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError(f"{_OP}: input_tensor must be on device")
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"{_OP}: input_tensor must be TILE_LAYOUT, got {input_tensor.layout}")
    in_mc = input_tensor.memory_config()
    if in_mc.memory_layout not in _SCHEME_NAME:
        raise ValueError(f"{_OP}: input_tensor must be HEIGHT_SHARDED or WIDTH_SHARDED, got {in_mc.memory_layout}")
    if in_mc.buffer_type != ttnn.BufferType.L1:
        raise ValueError(f"{_OP}: input_tensor must be sharded in L1, got {in_mc.buffer_type}")
    spec = in_mc.shard_spec
    if spec is None:
        raise ValueError(f"{_OP}: input_tensor has no shard spec")
    if spec.shape[0] % TILE != 0 or spec.shape[1] % TILE != 0:
        raise ValueError(f"{_OP}: shard shape {tuple(spec.shape)} is not a whole number of {TILE}x{TILE} tiles")
    if len(input_tensor.shape) not in (2, 4):
        raise ValueError(f"{_OP}: input rank must be 2 or 4, got {len(input_tensor.shape)}")
    if memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError(f"{_OP}: memory_config must be INTERLEAVED, got {memory_config.memory_layout}")
    if memory_config.buffer_type not in _BUFFER_NAME:
        raise ValueError(f"{_OP}: output buffer type must be DRAM or L1, got {memory_config.buffer_type}")


def _scenario(input_tensor, memory_config):
    in_mc = input_tensor.memory_config()
    spec = in_mc.shard_spec
    return {
        "shape": tuple(input_tensor.shape),
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "in_scheme": _SCHEME_NAME[in_mc.memory_layout],
        "shard_shape": tuple(spec.shape),
        "grid": _grid_tuple(spec.grid),
        "orientation": _ORIENT_NAME[spec.orientation],
        "buffer": _BUFFER_NAME[memory_config.buffer_type],
    }


def validate(input_tensor, memory_config):
    _structural_checks(input_tensor, memory_config)
    inputs = (_scenario(input_tensor, memory_config),)
    axes = {}
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger(inputs, axes)
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"{_OP}: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"{_OP}: unsupported combination (refinement candidate): {exc}")
    return inputs[0]


# ---------------------------------------------------------------------------
# Program cache (BB-07): build once per key, patch per call.
# ---------------------------------------------------------------------------

_BUILD_CACHE = {}
_LAST_PLAN = None


def _cache_key(scenario, input_tensor, knobs):
    return (
        scenario["shape"],
        tuple(input_tensor.padded_shape),
        str(scenario["dtype"]),
        scenario["in_scheme"],
        scenario["shard_shape"],
        scenario["grid"],
        scenario["orientation"],
        scenario["buffer"],
        knobs,
    )


def _program_hash(key):
    digest = hashlib.sha256(repr((_OP,) + key).encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def last_plan():
    """BB-08: the plan the most recent call selected."""
    return _LAST_PLAN


def roles(input_tensor):
    """BB-21: core -> role map for a given input."""
    return core_roles(input_tensor)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_sharded_to_interleaved(
    input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig, *, _knobs: Knobs = DEFAULT_KNOBS
) -> ttnn.Tensor:
    scenario = validate(input_tensor, memory_config)
    global _LAST_PLAN

    output_tensor = ttnn.allocate_tensor_on_device(
        input_tensor.shape,
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        memory_config,
    )

    key = _cache_key(scenario, input_tensor, _knobs)
    entry = _BUILD_CACHE.get(key)
    if entry is None:
        kernels, plan = build_program_descriptor(input_tensor, output_tensor, _knobs)
        entry = (kernels, plan, _program_hash(key))
        _BUILD_CACHE[key] = entry
    kernels, plan, program_hash = entry
    _LAST_PLAN = plan

    desc = bind_call(kernels, input_tensor, output_tensor, program_hash)
    return ttnn.generic_op([input_tensor, output_tensor], desc)
