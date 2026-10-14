# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""tilize — re-lay a ROW_MAJOR tensor into TILE layout in one device dispatch.

Registry-model op file: INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate() and the
public entry point. The taggers read the golden SCENARIO DICT (see
eval/golden_tests/tilize/feature_spec.py, "Expected INPUT_TAGGERS"); validate()
rebuilds the same scenario dict from the live call and runs the very same
taggers on it, so the runtime gate and the test-side decoration cannot drift.

Phase 0 builds the `interleaved_row_split` regime of op_design.md.
"""

from __future__ import annotations

import math

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .tilize_program_descriptor import create_program_descriptor

_INTERLEAVED = ttnn.TensorMemoryLayout.INTERLEAVED
_SHARDED_LAYOUTS = (
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
)

# Dimensionless aspect ratio at which one tile-grid axis dominates the other.
# Must equal feature_spec.DOMINANT (the op may not import the eval harness).
DOMINANT = 16

_VALID_TILE_HEIGHTS = (32, 16, 8, 4, 2, 1)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — each reads inputs[0], the whole scenario dict.
# ---------------------------------------------------------------------------


def _scenario(inputs):
    return inputs[0]


def _side_is_sharded(side):
    return side.get("kind") == "sharded"


def tag_low_l1(inputs, axes):
    return bool(_scenario(inputs).get("low_l1", False))


def tag_shard_api(inputs, axes):
    s = _scenario(inputs)
    sides = [s["in"], s["out"]]
    if not any(_side_is_sharded(side) for side in sides):
        return "none"
    if any(_side_is_sharded(side) and side.get("scheme") is None for side in sides):
        return "nd"
    return "legacy_2d"


def tag_out_scheme(inputs, axes):
    out = _scenario(inputs)["out"]
    if not _side_is_sharded(out):
        return _INTERLEAVED
    return ttnn.TensorMemoryLayout.ND_SHARDED if out.get("scheme") is None else out["scheme"]


def _buffer_word(buffer_type):
    return "dram" if buffer_type == ttnn.BufferType.DRAM else "l1"


def tag_buffer(inputs, axes):
    s = _scenario(inputs)
    return f"{_buffer_word(s['in']['buffer'])}_to_{_buffer_word(s['out']['buffer'])}"


def tag_rank(inputs, axes):
    return int(len(_scenario(inputs)["input_shape"]))


def tag_orientation(inputs, axes):
    s = _scenario(inputs)
    if _side_is_sharded(s["out"]):
        return s["out"]["orientation"]
    if _side_is_sharded(s["in"]):
        return s["in"]["orientation"]
    return "none"


def tag_pad_mode(inputs, axes):
    return _scenario(inputs).get("pad_mode", "none") or "none"


def tag_pad_value(inputs, axes):
    s = _scenario(inputs)
    if (s.get("pad_mode", "none") or "none") == "none":
        return "none"
    v = s.get("pad_value")
    if v is None or v == 0:
        return "zero"
    return "positive" if v > 0 else "negative"


def tag_tile_height(inputs, axes):
    return int(_scenario(inputs).get("tile_height", 32) or 32)


def tag_in_tile_height(inputs, axes):
    v = _scenario(inputs).get("in_tile_height")
    return "none" if v is None or v == "none" else int(v)


def _output_tile_height(scenario):
    return int(scenario.get("tile_height", 32) or 32)


def _tile_grid_dims(shape, tile_h):
    """(R, C) output tile grid: leading dims fold into R with a per-image ceil."""
    leading = 1
    for d in shape[:-2]:
        leading *= int(d)
    rows = leading * math.ceil(int(shape[-2]) / int(tile_h))
    cols = math.ceil(int(shape[-1]) / 32)
    return rows, cols


def tag_tile_grid(inputs, axes):
    s = _scenario(inputs)
    shape = list(s["input_shape"])
    if len(shape) < 2:
        return "single_tile"
    rows, cols = _tile_grid_dims(shape, _output_tile_height(s))
    if rows == 1 and cols == 1:
        return "single_tile"
    if cols >= DOMINANT * rows:
        return "short_wide"
    if rows >= DOMINANT * cols:
        return "tall_narrow"
    if rows * cols < DOMINANT * DOMINANT:
        return "small"
    return "square_large"


def tag_alignment(inputs, axes):
    s = _scenario(inputs)
    shape = list(s["input_shape"])
    if len(shape) < 2:
        return "hw_non_aligned"
    h_ok = shape[-2] % _output_tile_height(s) == 0
    w_ok = shape[-1] % 32 == 0
    if h_ok and w_ok:
        return "tile_aligned"
    if h_ok:
        return "w_non_aligned"
    if w_ok:
        return "h_non_aligned"
    return "hw_non_aligned"


INPUT_TAGGERS = {
    "low_l1": tag_low_l1,
    "shard_api": tag_shard_api,
    "out_scheme": tag_out_scheme,
    "buffer": tag_buffer,
    "rank": tag_rank,
    "orientation": tag_orientation,
    "pad_mode": tag_pad_mode,
    "pad_value": tag_pad_value,
    "tile_height": tag_tile_height,
    "in_tile_height": tag_in_tile_height,
    "tile_grid": tag_tile_grid,
    "alignment": tag_alignment,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED — Phase 0: the interleaved_row_split corner.
# ---------------------------------------------------------------------------
#
# tile_grid: the row split places min(R, grid) cores. tall_narrow (R >= 16C) puts
# the work on the row axis; single_tile / small have fewer tiles than the grid
# has cores. short_wide / square_large need the column cut (interleaved_2d_split)
# and are refused until it lands.

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "output_dtype": [ttnn.bfloat16],
    "low_l1": [False],
    "shard_api": ["none"],
    "out_scheme": [_INTERLEAVED],
    "buffer": ["dram_to_dram"],
    "rank": [4],
    "orientation": ["none"],
    "pad_mode": ["none"],
    "pad_value": ["none"],
    "tile_height": [32],
    "in_tile_height": ["none"],
    "tile_grid": ["single_tile", "small", "tall_narrow"],
    "alignment": ["tile_aligned"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "bounded_cb": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# Live call -> scenario dict (same keys the golden scenarios carry)
# ---------------------------------------------------------------------------


def _side_dict(mem_config):
    nd_spec = getattr(mem_config, "nd_shard_spec", None)
    if nd_spec is not None:
        return {"kind": "sharded", "buffer": mem_config.buffer_type, "orientation": nd_spec.orientation, "scheme": None}
    if mem_config.memory_layout in _SHARDED_LAYOUTS:
        spec = mem_config.shard_spec
        return {
            "kind": "sharded",
            "buffer": mem_config.buffer_type,
            "orientation": spec.orientation if spec is not None else "none",
            "scheme": mem_config.memory_layout,
        }
    return {"kind": "interleaved", "buffer": mem_config.buffer_type}


def _tile_height_of(tile):
    shape = getattr(tile, "tile_shape", None)
    if shape is None:
        shape = getattr(tile, "shape", None)
    return int(shape[0]), int(shape[1])


def _scenario_from_call(input_tensor, memory_config, *, low_l1, output_padded_shape, pad_value, tile):
    in_cfg = input_tensor.memory_config()
    out_cfg = memory_config if memory_config is not None else in_cfg
    in_tile_height = None
    if input_tensor.layout == ttnn.TILE_LAYOUT:
        in_tile_height = _tile_height_of(input_tensor.tile)[0]
    if tile is not None:
        tile_height = _tile_height_of(tile)[0]
    else:
        tile_height = in_tile_height or 32
    if output_padded_shape is not None:
        pad_mode = "explicit"
    elif pad_value is not None:
        pad_mode = "auto"
    else:
        pad_mode = "none"
    return {
        "input_shape": list(input_tensor.shape),
        "in": _side_dict(in_cfg),
        "out": _side_dict(out_cfg),
        "low_l1": bool(low_l1),
        "pad_mode": pad_mode,
        "pad_value": pad_value,
        "tile_height": tile_height,
        "in_tile_height": in_tile_height if in_tile_height is not None else "none",
    }


def _resolve_output_dtype(input_tensor, dtype):
    if dtype is not None:
        return dtype
    if input_tensor.dtype == ttnn.fp8_e4m3:
        return ttnn.float32
    return input_tensor.dtype


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _validate_well_formed(input_tensor, *, output_padded_shape, pad_value, tile):
    """Malformed calls -> ValueError (distinct from SUPPORTED refusals)."""
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("tilize: input tensor must be on device")
    if input_tensor.layout not in (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT):
        raise ValueError(f"tilize: input layout must be ROW_MAJOR or TILE, got {input_tensor.layout}")
    if input_tensor.layout == ttnn.TILE_LAYOUT and tile is None:
        raise ValueError("tilize: a TILE input requires tile= (nothing to re-tile to)")
    if tile is not None:
        th, tw = _tile_height_of(tile)
        if tw != 32 or th not in _VALID_TILE_HEIGHTS:
            raise ValueError(f"tilize: tile must be [h, 32] with h in {_VALID_TILE_HEIGHTS}, got [{th}, {tw}]")
        tile_h = th
    else:
        tile_h = 32
    shape = list(input_tensor.shape)
    padding_requested = output_padded_shape is not None or pad_value is not None
    if not padding_requested:
        if len(shape) < 2:
            raise ValueError(f"tilize: rank-{len(shape)} input requires a padding argument")
        if shape[-1] % 32 != 0 or shape[-2] % tile_h != 0:
            raise ValueError(
                f"tilize: input shape {shape} not aligned to tile ({tile_h}, 32) and no padding argument given"
            )
    if output_padded_shape is not None:
        target = list(output_padded_shape)
        if len(target) < len(shape) or any(t < s for t, s in zip(target[len(target) - len(shape) :], shape)):
            raise ValueError(f"tilize: output_padded_shape {target} is smaller than the input shape {shape}")


def validate(
    input_tensor,
    memory_config=None,
    *,
    dtype=None,
    low_l1=False,
    output_padded_shape=None,
    pad_value=None,
    tile=None,
):
    _validate_well_formed(input_tensor, output_padded_shape=output_padded_shape, pad_value=pad_value, tile=tile)

    scenario = _scenario_from_call(
        input_tensor,
        memory_config,
        low_l1=low_l1,
        output_padded_shape=output_padded_shape,
        pad_value=pad_value,
        tile=tile,
    )
    axes = {
        "dtype": input_tensor.dtype,
        "output_dtype": _resolve_output_dtype(input_tensor, dtype),
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"tilize: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"tilize: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def tilize(
    input_tensor: ttnn.Tensor,
    memory_config: ttnn.MemoryConfig | None = None,
    *,
    dtype: ttnn.DataType | None = None,
    low_l1: bool = False,
    output_padded_shape=None,
    pad_value=None,
    tile: ttnn.Tile | None = None,
) -> ttnn.Tensor:
    validate(
        input_tensor,
        memory_config,
        dtype=dtype,
        low_l1=low_l1,
        output_padded_shape=output_padded_shape,
        pad_value=pad_value,
        tile=tile,
    )

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else input_tensor.memory_config()
    output_dtype = _resolve_output_dtype(input_tensor, dtype)

    # Phase 0: no padding, so the output's logical == padded == input shape.
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        output_dtype,
        ttnn.TILE_LAYOUT,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(input_tensor, output_tensor, low_l1=low_l1)
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
