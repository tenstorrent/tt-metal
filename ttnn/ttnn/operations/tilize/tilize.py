# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""tilize — re-lay a Layout::ROW_MAJOR tensor into Layout::TILE in one generic_op dispatch.

Registry model (eval/op_template.py): INPUT_TAGGERS, SUPPORTED, EXCLUSIONS,
validate(), then the public entry point `tilize`.

The twelve INPUT_TAGGERS follow the `Expected INPUT_TAGGERS` block of
eval/golden_tests/tilize/feature_spec.py: each reads inputs[0] as a SCENARIO
DICT (keys: input_shape, shard_api, in, out, and optionally low_l1 / pad_mode /
output_padded_shape / pad_value / tile_height / in_tile_height). validate()
builds the same scenario dict from the live call (`_scenario_from_call`) and
runs the very same taggers over it, so the runtime gate and the golden
harness's xfail decisions share one set of rules.

Phase 0 regime: `row_split_interleaved` (op_design.md -> Blocking Model).
"""

from __future__ import annotations

import math

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .tilize_program_descriptor import create_program_descriptor


def _dominant():
    """Dimensionless ratio at which one tile-grid axis "dominates" the other.

    Imported (lazily, so importing the op never triggers the feature spec's arch
    probe) from the golden feature spec rather than restated (op_design.md). The
    fallback only exists for installs that ship ttnn without the eval tree.
    """
    global _DOMINANT
    if _DOMINANT is None:
        try:
            from eval.golden_tests.tilize.feature_spec import DOMINANT
        except Exception:  # noqa: BLE001 — eval tree absent outside the repo checkout
            DOMINANT = 16
        _DOMINANT = DOMINANT
    return _DOMINANT


_DOMINANT = None

TILE_WIDTH = 32  # a tile's width is always 32 elements
LEGAL_TILE_HEIGHTS = (32, 16, 8, 4, 2, 1)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — each reads inputs[0], the whole scenario dict
# ---------------------------------------------------------------------------


def _is_sharded(spec):
    return spec["kind"] == "sharded"


def _is_nd(spec):
    return _is_sharded(spec) and spec.get("scheme") is None


def tag_low_l1(inputs, axes):
    return bool(inputs[0].get("low_l1", False))


def tag_shard_api(inputs, axes):
    s = inputs[0]
    if not _is_sharded(s["in"]) and not _is_sharded(s["out"]):
        return "none"
    if _is_nd(s["in"]) or _is_nd(s["out"]):
        return "nd"
    return "legacy_2d"


def tag_out_scheme(inputs, axes):
    out = inputs[0]["out"]
    if not _is_sharded(out):
        return "interleaved"
    if _is_nd(out):
        return "nd"
    return out["scheme"]


def _buffer_word(buffer_type):
    return "dram" if buffer_type == ttnn.BufferType.DRAM else "l1"


def tag_buffer(inputs, axes):
    s = inputs[0]
    return f"{_buffer_word(s['in']['buffer'])}_to_{_buffer_word(s['out']['buffer'])}"


def tag_rank(inputs, axes):
    return int(len(inputs[0]["input_shape"]))


def tag_orientation(inputs, axes):
    s = inputs[0]
    if _is_sharded(s["out"]):
        return s["out"]["orientation"]
    if _is_sharded(s["in"]):
        return s["in"]["orientation"]
    return "none"


def tag_pad_mode(inputs, axes):
    return inputs[0].get("pad_mode", "none")


def tag_pad_value(inputs, axes):
    s = inputs[0]
    if s.get("pad_mode", "none") == "none":
        return "none"
    v = s.get("pad_value", 0)
    if v is None or v == 0:
        return "zero"
    return "positive" if v > 0 else "negative"


def tag_tile_height(inputs, axes):
    return int(inputs[0].get("tile_height", 32))


def tag_in_tile_height(inputs, axes):
    v = inputs[0].get("in_tile_height", "none")
    return "none" if v is None else v


def _tile_grid_dims(shape, tile_height):
    """(R, C): output tile-rows (leading dims folded, per-image ceil) and tile-columns."""
    leading = 1
    for d in shape[:-2]:
        leading *= int(d)
    rows = leading * math.ceil(int(shape[-2]) / int(tile_height))
    cols = math.ceil(int(shape[-1]) / TILE_WIDTH)
    return rows, cols


def tag_tile_grid(inputs, axes):
    shape = list(inputs[0]["input_shape"])
    if len(shape) < 2:
        return "single_tile"
    tile_height = axes.get("tile_height", inputs[0].get("tile_height", 32))
    rows, cols = _tile_grid_dims(shape, tile_height)
    dominant = _dominant()
    if rows == 1 and cols == 1:
        return "single_tile"
    if cols >= dominant * rows:
        return "short_wide"
    if rows >= dominant * cols:
        return "tall_narrow"
    if rows * cols < dominant * dominant:
        return "small"
    return "square_large"


def tag_alignment(inputs, axes):
    shape = list(inputs[0]["input_shape"])
    if len(shape) < 2:
        return "hw_non_aligned"
    tile_height = axes.get("tile_height", inputs[0].get("tile_height", 32))
    h_ok = shape[-2] % int(tile_height) == 0
    w_ok = shape[-1] % TILE_WIDTH == 0
    if h_ok and w_ok:
        return "tile_aligned"
    if h_ok:
        return "w_non_aligned"
    if w_ok:
        return "h_non_aligned"
    return "hw_non_aligned"


# Order matters: tile_grid and alignment read the resolved tile_height.
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
# 2. SUPPORTED — Phase 0 rectangle (op_design.md -> "Phase 0 SUPPORTED rectangle")
# ---------------------------------------------------------------------------
#
# tile_grid: the row split (`split_work_to_cores(grid, R, row_wise=True)`)
# reaches min(R, N) Tensix cores. On single_tile / small that is all the work
# there is; on tall_narrow (R >= 16*C) it fills the grid whenever R >= N and is
# within one tile-row of the 2-D optimum otherwise. It does NOT reach the grid
# on short_wide (R is 1-2) nor balance square_large, so both are refused until
# the grid_2d_split refinement.

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "output_dtype": [ttnn.bfloat16],
    "low_l1": [False],
    "shard_api": ["none"],
    "out_scheme": ["interleaved"],
    "buffer": ["dram_to_dram"],
    "rank": [4],
    "orientation": ["none"],
    "pad_mode": ["none"],
    "pad_value": ["none"],
    "alignment": ["tile_aligned"],
    "tile_height": [32],
    "in_tile_height": ["none"],
    "tile_grid": ["single_tile", "small", "tall_narrow"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 3b. PROPERTIES
# ---------------------------------------------------------------------------

PROPERTIES = {
    # split_work_to_cores over device.compute_with_storage_grid_size(): min(R, N) Tensix cores.
    "multi_core": {"value": True, "source": "declared"},
    # Every CB is block_width * per_col_tile_bytes <= CB_BUDGET_BYTES[low_l1] (l1_ledger.md).
    "bounded_cb": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------

_SHARDED_LAYOUTS = (
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
)


def _side_spec(mem_config):
    """Scenario-dict side spec ({kind, buffer, orientation, scheme}) from a live MemoryConfig.

    Mirrors eval/golden_tests/tilize/axes.py:_spec_of so the two agree.
    """
    nd_spec = getattr(mem_config, "nd_shard_spec", None)
    if nd_spec is not None:
        return {"kind": "sharded", "buffer": mem_config.buffer_type, "orientation": nd_spec.orientation, "scheme": None}
    if mem_config.memory_layout in _SHARDED_LAYOUTS:
        shard_spec = mem_config.shard_spec
        return {
            "kind": "sharded",
            "buffer": mem_config.buffer_type,
            "orientation": shard_spec.orientation if shard_spec is not None else "none",
            "scheme": mem_config.memory_layout,
        }
    return {"kind": "interleaved", "buffer": mem_config.buffer_type}


def _tile_height_of(tile):
    if tile is None:
        return None
    shape = getattr(tile, "tile_shape", None) or getattr(tile, "shape", None)
    return int(shape[0]) if shape is not None else None


def _input_tile_height(input_tensor):
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        return None
    return _tile_height_of(getattr(input_tensor, "tile", None)) or 32


def _resolve_output_dtype(input_tensor, dtype):
    if dtype is not None:
        return dtype
    if input_tensor.dtype == ttnn.fp8_e4m3:
        return ttnn.float32
    return input_tensor.dtype


def _scenario_from_call(input_tensor, memory_config, *, low_l1, output_padded_shape, pad_value, tile):
    in_cfg = input_tensor.memory_config()
    out_cfg = memory_config if memory_config is not None else in_cfg
    scenario = {
        "input_shape": list(input_tensor.shape),
        "in": _side_spec(in_cfg),
        "out": _side_spec(out_cfg),
        "low_l1": bool(low_l1),
        "tile_height": _tile_height_of(tile) or _input_tile_height(input_tensor) or 32,
    }
    if output_padded_shape is not None:
        scenario["pad_mode"] = "explicit"
        scenario["output_padded_shape"] = list(output_padded_shape)
        scenario["pad_value"] = 0 if pad_value is None else pad_value
    elif pad_value is not None:
        scenario["pad_mode"] = "auto"
        scenario["pad_value"] = pad_value
    in_tile_h = _input_tile_height(input_tensor)
    if in_tile_h is not None:
        scenario["in_tile_height"] = in_tile_h
    return scenario


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
    """Registry gate: SUPPORTED per-axis, then EXCLUSIONS. Raises NotImplementedError subclasses."""
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
    inputs = (scenario,)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"tilize: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"tilize: unsupported combination (refinement candidate): {exc}")
    return axes


# ---------------------------------------------------------------------------
# Malformed-call checks (op_design.md -> validation order, rules 1-6)
# ---------------------------------------------------------------------------


def _check_well_formed(input_tensor, *, output_padded_shape, pad_value, tile):
    # 1. on device
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("tilize: input tensor must be on device")
    # 2. layout
    if input_tensor.layout not in (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT):
        raise ValueError(f"tilize: input layout {input_tensor.layout} is neither ROW_MAJOR nor TILE")
    # 3. a TILE input needs a target geometry
    if input_tensor.layout == ttnn.TILE_LAYOUT and tile is None:
        raise ValueError("tilize: a TILE_LAYOUT input needs tile= (there is nothing to re-tile to)")
    # 4. tile geometry
    if tile is not None:
        th, tw = (int(d) for d in tile.tile_shape)
        if tw != TILE_WIDTH or th not in LEGAL_TILE_HEIGHTS:
            raise ValueError(
                f"tilize: tile {[th, tw]} must be [h, 32] with h a power-of-two fraction of 32 {LEGAL_TILE_HEIGHTS}"
            )
    tile_h = _tile_height_of(tile) or _input_tile_height(input_tensor) or 32
    shape = list(input_tensor.shape)
    has_pad = output_padded_shape is not None or pad_value is not None
    # 5. alignment without a padding argument
    if not has_pad:
        if len(shape) < 2 or shape[-2] % tile_h != 0 or shape[-1] % TILE_WIDTH != 0:
            raise ValueError(
                f"tilize: shape {shape} is not a multiple of ({tile_h}, {TILE_WIDTH}) and no padding argument was given"
            )
    # 6. explicit padded shape
    if output_padded_shape is not None:
        padded = [int(d) for d in output_padded_shape]
        expanded = [1] * max(0, len(padded) - len(shape)) + shape
        if len(padded) < len(shape) or any(p < s for p, s in zip(padded, expanded)):
            raise ValueError(f"tilize: output_padded_shape {padded} is smaller than the input shape {shape}")
        if len(padded) < 2 or padded[-2] % tile_h != 0 or padded[-1] % TILE_WIDTH != 0:
            raise ValueError(
                f"tilize: output_padded_shape {padded} last two dims must be multiples of ({tile_h}, {TILE_WIDTH})"
            )


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
    """Re-lay `input_tensor` (Layout::ROW_MAJOR) into Layout::TILE; values and logical shape unchanged."""
    _check_well_formed(input_tensor, output_padded_shape=output_padded_shape, pad_value=pad_value, tile=tile)
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
    out_mem_config = memory_config if memory_config is not None else input_tensor.memory_config()
    out_dtype = _resolve_output_dtype(input_tensor, dtype)
    tile_h = _tile_height_of(tile) or 32

    # CRITICAL: allocate_tensor_on_device takes positional args only.
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        out_dtype,
        ttnn.TILE_LAYOUT,
        device,
        out_mem_config,
    )

    program_descriptor = create_program_descriptor(input_tensor, output_tensor, tile_h=tile_h, low_l1=low_l1)
    # Output tensor MUST be last in the list.
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
