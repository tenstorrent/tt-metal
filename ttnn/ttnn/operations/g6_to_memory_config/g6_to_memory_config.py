# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_to_memory_config: return a new TILE tensor with the input's values placed as `memory_config` names.

One device program per call. Regimes (op_design.md, Blocking Model / Regimes):
  PULL  — target L1-sharded: each target-shard owner reads its pages into its own shard.
  PUSH  — source L1-sharded: each source-shard owner writes its pages from its own shard.
  STAGE — both ends banks: a full-grid set of movers stages page ranges through an L1 ring.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_to_memory_config_program_descriptor import create_program_descriptor, select_regime  # noqa: F401

TILE = 32

_BUFFER_NAME = {ttnn.BufferType.DRAM: "dram", ttnn.BufferType.L1: "l1"}
_SCHEME_NAME = {
    ttnn.TensorMemoryLayout.INTERLEAVED: "interleaved",
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: "height_sharded",
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: "width_sharded",
    ttnn.TensorMemoryLayout.BLOCK_SHARDED: "block_sharded",
}


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — inputs is (scenario,), scenario a dict with "shape", "dtype",
#    "layout", "in_memory", "out_memory" (feature_spec.py / axes.py)
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


def tag_rank(inputs, axes):
    return int(len(inputs[0]["shape"]))


def tag_alignment(inputs, axes):
    shape = inputs[0]["shape"]
    if shape[-1] % TILE == 0 and shape[-2] % TILE == 0:
        return "tile_aligned"
    return "h_non_aligned"


def tag_in_scheme(inputs, axes):
    return inputs[0]["in_memory"]["scheme"]


def tag_in_buffer(inputs, axes):
    return inputs[0]["in_memory"]["buffer"]


def tag_out_scheme(inputs, axes):
    return inputs[0]["out_memory"]["scheme"]


def tag_out_buffer(inputs, axes):
    return inputs[0]["out_memory"]["buffer"]


INPUT_TAGGERS = {
    "dtype": tag_dtype,
    "layout": tag_layout,
    "rank": tag_rank,
    "alignment": tag_alignment,
    "in_scheme": tag_in_scheme,
    "in_buffer": tag_in_buffer,
    "out_scheme": tag_out_scheme,
    "out_buffer": tag_out_buffer,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

_SCHEMES = ["interleaved", "height_sharded", "width_sharded", "block_sharded"]

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "rank": [2, 3, 4],
    "alignment": ["tile_aligned", "h_non_aligned"],
    "in_scheme": list(_SCHEMES),
    "in_buffer": ["dram", "l1"],
    "out_scheme": list(_SCHEMES),
    "out_buffer": ["dram", "l1"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    # DRAM width-sharded targets (bf4 weights): STAGE with a sharded DRAM output accessor,
    # lifted once that page path is tested (op_design.md, Registry).
    {"out_scheme": "width_sharded", "out_buffer": "dram"},
]


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "bounded_cb": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _memory_entry(memory_config):
    return {
        "buffer": _BUFFER_NAME.get(memory_config.buffer_type, str(memory_config.buffer_type)),
        "scheme": _SCHEME_NAME.get(memory_config.memory_layout, str(memory_config.memory_layout)),
    }


def _tile_grid(shape):
    leading = 1
    for d in shape[:-2]:
        leading *= d
    return leading * ((shape[-2] + TILE - 1) // TILE), (shape[-1] + TILE - 1) // TILE


def _check_shard_spec(what, memory_config, tile_rows, tile_cols):
    if not memory_config.is_sharded():
        return
    spec = memory_config.shard_spec
    if spec is None:
        raise ValueError(f"g6_to_memory_config: {what} is sharded but has no shard spec")
    if spec.orientation != ttnn.ShardOrientation.ROW_MAJOR:
        raise ValueError(f"g6_to_memory_config: {what} shard orientation must be ROW_MAJOR, got {spec.orientation}")
    sh, sw = spec.shape
    if sh % TILE or sw % TILE or sh <= 0 or sw <= 0:
        raise ValueError(f"g6_to_memory_config: {what} shard shape {tuple(spec.shape)} must be multiples of {TILE}")
    n_rows = (tile_rows + sh // TILE - 1) // (sh // TILE)
    n_cols = (tile_cols + sw // TILE - 1) // (sw // TILE)
    layout = memory_config.memory_layout
    if layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        bbox = spec.grid.bounding_box()
        bw = bbox.end.x - bbox.start.x + 1
        bh = bbox.end.y - bbox.start.y + 1
        if bh < n_rows or bw < n_cols:
            raise ValueError(
                f"g6_to_memory_config: {what} block shard grid {bw}x{bh} cannot hold {n_cols}x{n_rows} shards"
            )
    else:
        if layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED and n_cols != 1:
            raise ValueError(f"g6_to_memory_config: {what} height shard width must cover the tensor width")
        if layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED and n_rows != 1:
            raise ValueError(f"g6_to_memory_config: {what} width shard height must cover the tensor height")
        if spec.grid.num_cores() < n_rows * n_cols:
            raise ValueError(
                f"g6_to_memory_config: {what} shard grid has {spec.grid.num_cores()} cores, "
                f"needs {n_rows * n_cols} shards"
            )


def validate(input_tensor, memory_config):
    # Host validation (ValueError) first.
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("g6_to_memory_config: input tensor must be on device")
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"g6_to_memory_config: input must be TILE layout, got {input_tensor.layout}")
    shape = list(input_tensor.shape)
    if not 2 <= len(shape) <= 4:
        raise ValueError(f"g6_to_memory_config: rank must be 2..4, got {len(shape)}")
    if memory_config is None:
        raise ValueError("g6_to_memory_config: memory_config is required")
    tile_rows, tile_cols = _tile_grid(shape)
    _check_shard_spec("input", input_tensor.memory_config(), tile_rows, tile_cols)
    _check_shard_spec("memory_config", memory_config, tile_rows, tile_cols)

    scenario = {
        "shape": shape,
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "in_memory": _memory_entry(input_tensor.memory_config()),
        "out_memory": _memory_entry(memory_config),
    }
    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_to_memory_config: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_to_memory_config: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_to_memory_config(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    validate(input_tensor, memory_config)
    output_tensor = ttnn.allocate_tensor_on_device(
        input_tensor.shape,
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        memory_config,
    )
    program_descriptor = create_program_descriptor(input_tensor, output_tensor)
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
