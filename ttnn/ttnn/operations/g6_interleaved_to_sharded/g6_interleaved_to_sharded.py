# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_interleaved_to_sharded: move a DRAM-interleaved bfloat16 TILE tensor, bit for bit,
into the L1 height or width shards that memory_config names, in one program.

Registry model (eval/op_template.py): INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate().
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from . import g6_interleaved_to_sharded_program_descriptor as program_descriptor

OP_NAME = "g6_interleaved_to_sharded"

_SCHEME_NAME = {
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: "height_sharded",
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: "width_sharded",
}


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — each projects the scenario dict (inputs = (scenario,))
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


def _malformed(message):
    return ValueError(f"{OP_NAME}: {message}")


def _check_well_formed(input_tensor, memory_config):
    """Malformed requests raise ValueError, before and independently of the support check."""
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise _malformed("input_tensor must be on a device")
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise _malformed(f"input_tensor must be in TILE_LAYOUT, got {input_tensor.layout}")
    input_memory = input_tensor.memory_config()
    if (
        input_memory.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
        or input_memory.buffer_type != ttnn.BufferType.DRAM
    ):
        raise _malformed(
            f"input_tensor must be DRAM interleaved, got {input_memory.buffer_type} {input_memory.memory_layout}"
        )
    if input_tensor.dtype != ttnn.bfloat16:
        raise _malformed(f"input_tensor must be bfloat16, got {input_tensor.dtype}")
    rank = len(input_tensor.shape)
    if rank not in (2, 4):
        raise _malformed(f"input_tensor must have rank 2 or 4, got {rank}")

    if (
        memory_config.buffer_type != ttnn.BufferType.L1
        or memory_config.memory_layout not in _SCHEME_NAME
        or memory_config.shard_spec is None
    ):
        raise _malformed(
            "memory_config must be L1 HEIGHT_SHARDED or WIDTH_SHARDED with a shard spec, got "
            f"{memory_config.buffer_type} {memory_config.memory_layout}"
        )

    tile_h, tile_w, num_images, tiles_h_per_image, tensor_cols_tiles = program_descriptor.tile_grid(input_tensor)
    tensor_rows_tiles = num_images * tiles_h_per_image
    shard_spec = memory_config.shard_spec
    shard_h, shard_w = (int(d) for d in shard_spec.shape)
    if shard_h % tile_h != 0 or shard_w % tile_w != 0:
        raise _malformed(f"shard shape ({shard_h}, {shard_w}) must be a multiple of the tile ({tile_h}, {tile_w})")
    height_sharded = memory_config.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    if height_sharded and shard_w != tile_w * tensor_cols_tiles:
        raise _malformed(
            f"a height shard must span the padded width {tile_w * tensor_cols_tiles}, got shard width {shard_w}"
        )
    if not height_sharded and shard_h != tile_h * tensor_rows_tiles:
        raise _malformed(
            f"a width shard must span the padded height {tile_h * tensor_rows_tiles}, got shard height {shard_h}"
        )
    num_shards = program_descriptor.num_shards_of(
        height_sharded, tensor_rows_tiles, tensor_cols_tiles, shard_h // tile_h, shard_w // tile_w
    )
    num_cores = shard_spec.grid.num_cores()
    if num_shards > num_cores:
        raise _malformed(f"{num_shards} shards do not fit on the {num_cores} cores of the shard grid")
    if memory_config.experimental_is_per_core_allocation():
        raise _malformed("per-core L1 allocation is not accepted: every shard must share one address")


def _scenario_of(input_tensor, memory_config):
    """The registry scenario dict of a real call, so validate() runs the same taggers as the harness."""
    return {
        "shape": [int(d) for d in input_tensor.shape],
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "out_memory": {"scheme": _SCHEME_NAME[memory_config.memory_layout]},
    }


def validate(input_tensor, memory_config):
    _check_well_formed(input_tensor, memory_config)

    inputs = (_scenario_of(input_tensor, memory_config),)
    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"{OP_NAME}: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"{OP_NAME}: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def describe_plan(input_tensor, memory_config) -> dict:
    """The plan a call with these arguments selects (BB-08), in shard-grid core order."""
    validate(input_tensor, memory_config)
    return program_descriptor.describe(program_descriptor.derive_plan(input_tensor, memory_config))


def g6_interleaved_to_sharded(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    validate(input_tensor, memory_config)

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        memory_config,
    )
    descriptor = program_descriptor.get_program_descriptor(input_tensor, output_tensor, memory_config)
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
