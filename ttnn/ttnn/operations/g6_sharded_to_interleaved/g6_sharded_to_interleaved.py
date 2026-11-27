# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_sharded_to_interleaved: a height- or width-sharded L1 TILE tensor -> a new interleaved tensor.

out[i] = input_tensor[i] bit for bit; shape, dtype and TILE kept; placement from `memory_config`
(interleaved, DRAM or L1). One generic_op dispatch per call.
"""

from __future__ import annotations

import math

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_sharded_to_interleaved_program_descriptor import (
    TILE_HW,
    create_program_descriptor,
    program_cache_key,
    refresh_program_descriptor,
)

_OP = "g6_sharded_to_interleaved"

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


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — `inputs` is (scenario,), the scenario dict of feature_spec.py
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


def _check_structure(input_tensor, memory_config):
    """Structural refusals: ValueError, never a support refusal."""
    if not isinstance(input_tensor, ttnn.Tensor):
        raise ValueError(f"{_OP}: input_tensor must be a ttnn.Tensor, got {type(input_tensor)}")
    if input_tensor.storage_type() != ttnn.StorageType.DEVICE:
        raise ValueError(f"{_OP}: input_tensor must be on a device")
    rank = len(input_tensor.shape)
    if rank not in (2, 4):
        raise ValueError(f"{_OP}: input rank must be 2 or 4, got {rank}")
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"{_OP}: input_tensor must be in TILE_LAYOUT, got {input_tensor.layout}")
    if tuple(input_tensor.tile.tile_shape) != (TILE_HW, TILE_HW):
        raise ValueError(f"{_OP}: input tile must be {TILE_HW}x{TILE_HW}, got {input_tensor.tile.tile_shape}")

    in_mc = input_tensor.memory_config()
    if in_mc.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        raise ValueError(f"{_OP}: block-sharded input is not a valid input (height or width sharding only)")
    if in_mc.memory_layout not in _IN_SCHEME or not input_tensor.is_sharded():
        raise ValueError(f"{_OP}: input_tensor must be height- or width-sharded, got {in_mc.memory_layout}")
    if in_mc.buffer_type != ttnn.BufferType.L1:
        raise ValueError(f"{_OP}: input_tensor must be sharded in L1, got {in_mc.buffer_type}")
    spec = in_mc.shard_spec
    if spec is None:
        raise ValueError(f"{_OP}: sharded input_tensor has no shard spec")
    shard_h, shard_w = (int(d) for d in spec.shape)
    if shard_h <= 0 or shard_w <= 0 or shard_h % TILE_HW or shard_w % TILE_HW:
        raise ValueError(f"{_OP}: shard shape {(shard_h, shard_w)} is not a whole number of tiles")

    padded = tuple(int(d) for d in input_tensor.padded_shape)
    padded_rows, padded_w = math.prod(padded[:-1]), padded[-1]
    if in_mc.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        if shard_w != padded_w:
            raise ValueError(f"{_OP}: height shard width {shard_w} != padded width {padded_w}")
        num_shards = -(-padded_rows // shard_h)
    else:
        if shard_h != padded_rows:
            raise ValueError(f"{_OP}: width shard height {shard_h} != padded height {padded_rows}")
        num_shards = -(-padded_w // shard_w)
    if num_shards > spec.grid.num_cores():
        raise ValueError(f"{_OP}: {num_shards} shards exceed the {spec.grid.num_cores()} cores of the shard grid")

    if not isinstance(memory_config, ttnn.MemoryConfig):
        raise ValueError(f"{_OP}: memory_config must be a ttnn.MemoryConfig, got {type(memory_config)}")
    if memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError(f"{_OP}: memory_config must be interleaved, got {memory_config.memory_layout}")


def _scenario(input_tensor, memory_config):
    in_mc = input_tensor.memory_config()
    spec = in_mc.shard_spec
    return {
        "shape": tuple(int(d) for d in input_tensor.shape),
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "in_scheme": _IN_SCHEME[in_mc.memory_layout],
        "shard_shape": tuple(int(d) for d in spec.shape),
        "grid": tuple(((r.start.x, r.start.y), (r.end.x, r.end.y)) for r in spec.grid.ranges()),
        "orientation": _ORIENTATION.get(spec.orientation, str(spec.orientation)),
        "buffer": _BUFFER.get(memory_config.buffer_type, str(memory_config.buffer_type)),
    }


def validate(input_tensor, memory_config):
    _check_structure(input_tensor, memory_config)

    scenario = _scenario(input_tensor, memory_config)
    axes = {key: scenario[key] for key in ("dtype", "layout", "in_scheme", "buffer")}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"{_OP}: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"{_OP}: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# Program descriptors by program_cache_key (BB-07): a repeat call rebuilds nothing, it only re-points the
# cached descriptor at this call's input buffer and output address.
_PROGRAM_CACHE = {}


def g6_sharded_to_interleaved(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    validate(input_tensor, memory_config)

    # The call's first and only device allocation, so a freed output-sized range is reused.
    output_tensor = ttnn.allocate_tensor_on_device(
        input_tensor.shape,
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        memory_config,
    )

    key = program_cache_key(input_tensor, memory_config)
    descriptor = _PROGRAM_CACHE.get(key)
    if descriptor is None:
        descriptor = create_program_descriptor(input_tensor, output_tensor, key)
        _PROGRAM_CACHE[key] = descriptor
    else:
        refresh_program_descriptor(descriptor, input_tensor, output_tensor)

    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
