# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_interleaved_to_sharded: move a DRAM-interleaved TILE tensor, values unchanged, into the L1
height or width shards that memory_config names. One program: each shard-owning core reads its own
pages from DRAM straight into its shard."""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from . import g6_interleaved_to_sharded_program_descriptor as _descriptor
from .g6_interleaved_to_sharded_program_descriptor import (
    L1_BUDGET_BYTES,
    create_program_descriptor,
    describe_plan,
    plan,
    set_addresses,
    tile_grid,
)

__all__ = [
    "g6_interleaved_to_sharded",
    "describe_plan",
    "validate",
    "INPUT_TAGGERS",
    "SUPPORTED",
    "EXCLUSIONS",
]


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — each reads the scenario dict inputs[0]
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


def tag_rank(inputs, axes):
    return int(len(inputs[0]["shape"]))


def tag_alignment(inputs, axes):
    shape = inputs[0]["shape"]
    height, width = shape[-2], shape[-1]
    if height % 32 == 0 and width % 32 == 0:
        return "tile_aligned"
    if width % 32 != 0:
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

_SCHEME_NAMES = {
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: "height_sharded",
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: "width_sharded",
}
_ORIENTATION_NAMES = {
    ttnn.ShardOrientation.ROW_MAJOR: "row_major",
    ttnn.ShardOrientation.COL_MAJOR: "col_major",
}


def _check_request(input_tensor, memory_config):
    """Malformed requests: ValueError, before the support check."""
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("g6_interleaved_to_sharded: input_tensor must be on a device")
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"g6_interleaved_to_sharded: input layout must be TILE, got {input_tensor.layout}")
    in_config = input_tensor.memory_config()
    if in_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError(f"g6_interleaved_to_sharded: input must be interleaved, got {in_config.memory_layout}")
    if in_config.buffer_type != ttnn.BufferType.DRAM:
        raise ValueError(f"g6_interleaved_to_sharded: input must be in DRAM, got {in_config.buffer_type}")
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError(f"g6_interleaved_to_sharded: dtype must be bfloat16, got {input_tensor.dtype}")
    rank = len(input_tensor.shape)
    if rank not in (2, 4):
        raise ValueError(f"g6_interleaved_to_sharded: rank must be 2 or 4, got {rank}")

    if memory_config.buffer_type != ttnn.BufferType.L1:
        raise ValueError(f"g6_interleaved_to_sharded: target must be L1, got {memory_config.buffer_type}")
    if memory_config.memory_layout not in _SCHEME_NAMES:
        raise ValueError(
            f"g6_interleaved_to_sharded: target must be HEIGHT_SHARDED or WIDTH_SHARDED, "
            f"got {memory_config.memory_layout}"
        )
    shard_spec = memory_config.shard_spec
    if shard_spec is None:
        raise ValueError("g6_interleaved_to_sharded: target memory_config has no shard_spec")

    tile_h, tile_w = input_tensor.tile.tile_shape
    shard_h, shard_w = shard_spec.shape
    if shard_h % tile_h or shard_w % tile_w or shard_h == 0 or shard_w == 0:
        raise ValueError(
            f"g6_interleaved_to_sharded: shard shape {list(shard_spec.shape)} is not a multiple of the "
            f"tile {tile_h}x{tile_w}"
        )
    images, rows_per_image, cols, _ = tile_grid(input_tensor)
    rows = images * rows_per_image
    if memory_config.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        if shard_w < cols * tile_w:
            raise ValueError(
                f"g6_interleaved_to_sharded: height shard width {shard_w} is narrower than the tensor's "
                f"padded width {cols * tile_w}"
            )
        num_shards = -(-rows // (shard_h // tile_h))
    else:
        if shard_h < rows * tile_h:
            raise ValueError(
                f"g6_interleaved_to_sharded: width shard height {shard_h} is shorter than the tensor's "
                f"folded padded height {rows * tile_h}"
            )
        num_shards = -(-cols // (shard_w // tile_w))
    if num_shards > shard_spec.grid.num_cores():
        raise ValueError(
            f"g6_interleaved_to_sharded: {num_shards} shards do not fit the {shard_spec.grid.num_cores()} "
            f"cores of the shard grid"
        )
    shard_bytes = (shard_h // tile_h) * (shard_w // tile_w) * input_tensor.buffer_page_size()
    if shard_bytes > L1_BUDGET_BYTES:
        raise ValueError(
            f"g6_interleaved_to_sharded: the {shard_bytes} B output shard exceeds the {L1_BUDGET_BYTES} B "
            f"of L1 per core"
        )


def _scenario(input_tensor, memory_config):
    """The call as the golden scenario dict, so tests and runtime share one tagging path."""
    in_config = input_tensor.memory_config()
    shard_spec = memory_config.shard_spec
    (start, end) = (shard_spec.grid.bounding_box().start, shard_spec.grid.bounding_box().end)
    return {
        "shape": [int(d) for d in input_tensor.shape],
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "in_memory": {
            "buffer": "dram" if in_config.buffer_type == ttnn.BufferType.DRAM else "l1",
            "scheme": "interleaved",
        },
        "out_memory": {
            "buffer": "l1",
            "scheme": _SCHEME_NAMES[memory_config.memory_layout],
            "grid": [[start.x, start.y], [end.x, end.y]],
            "shard_shape": list(shard_spec.shape),
            "orientation": _ORIENTATION_NAMES[shard_spec.orientation],
        },
    }


def validate(input_tensor, memory_config):
    _check_request(input_tensor, memory_config)

    inputs = (_scenario(input_tensor, memory_config),)
    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_interleaved_to_sharded: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_interleaved_to_sharded: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Program cache (BB-07): everything that shapes the program is in the key; a hit patches only the
# two buffer addresses.
# ---------------------------------------------------------------------------

_DESCRIPTOR_CACHE = {}


def _cache_key(input_tensor, memory_config):
    in_config = input_tensor.memory_config()
    shard_spec = memory_config.shard_spec
    grid = tuple((r.start.x, r.start.y, r.end.x, r.end.y) for r in shard_spec.grid.ranges())
    return (
        "g6_interleaved_to_sharded",
        input_tensor.device().id(),
        tuple(int(d) for d in input_tensor.shape),
        tuple(int(d) for d in input_tensor.padded_shape),
        str(input_tensor.dtype),
        tuple(input_tensor.tile.tile_shape),
        str(in_config.buffer_type),
        str(in_config.memory_layout),
        str(memory_config.memory_layout),
        str(memory_config.buffer_type),
        grid,
        tuple(shard_spec.shape),
        str(shard_spec.orientation),
        _descriptor.PARTIAL_TILE_TRANSFER,
        _descriptor.READS_IN_FLIGHT,
        _descriptor.READER_PROCESSOR,
        _descriptor.READER_SPLIT,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_interleaved_to_sharded(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    validate(input_tensor, memory_config)

    output_tensor = ttnn.allocate_tensor_on_device(
        input_tensor.shape,
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        memory_config,
    )

    key = _cache_key(input_tensor, memory_config)
    descriptor = _DESCRIPTOR_CACHE.get(key)
    if descriptor is None:
        descriptor = create_program_descriptor(plan(input_tensor, memory_config), input_tensor, output_tensor)
        _DESCRIPTOR_CACHE[key] = descriptor
    else:
        set_addresses(descriptor, input_tensor, output_tensor)

    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
