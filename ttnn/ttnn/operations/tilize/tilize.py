# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""tilize — re-lay a ROW_MAJOR tensor into TILE layout in one device dispatch.

Registry-model op file: INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate() and the
public entry point. The taggers read the golden SCENARIO DICT (see
eval/golden_tests/tilize/feature_spec.py, "Expected INPUT_TAGGERS"); validate()
rebuilds the same scenario dict from the live call and runs the very same
taggers on it, so the runtime gate and the test-side decoration cannot drift.

Dtypes (Refinement 5): every legal dtype x output_dtype pair; cast at pack time (output CB format =
output dtype); 32-bit inputs tilize bit-exact through a 32-bit DEST.

Built regimes (op_design.md): `interleaved_row_split`, `interleaved_2d_split` (the host picks the
per-core rectangle), in-kernel pad fill (reader) for `pad_mode` auto / explicit, and the sharded
regimes `sharded_same_spec_in_place`, `sharded_input_pinned`, `sharded_output_pinned`,
`sharded_full_grid` (resident shards consumed / produced in place via shard-backed CBs), and
`retile_tile_input` (a TILE input at another tile height, de-faced by the reader).
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
_MAX_PADDED_RANK = 6


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
# tile_grid: the row split places min(R, grid) cores; when R < grid the host switches to the
# 2-D rectangle split (interleaved_2d_split), which reaches the grid on short_wide / square_large.
# Padding (pad_mode / pad_value / alignment / rank 0-1): filled in-kernel by the reader.

# low_l1 / buffer / rank / tile_height: knob-turns the Phase 0 kernels already realize
# (low_l1 caps block_width_tiles at LOW_L1_BLOCK_WIDTH_TILES; TensorAccessor is buffer-type
# agnostic; leading dims fold into tile-rows; tiny output tiles = CB tile dims + the helper's
# tilize_block fallback). Verified bit-exact by the verifier and claimed here.
SUPPORTED = {
    "dtype": [ttnn.bfloat16, ttnn.float32, ttnn.fp8_e4m3, ttnn.uint32, ttnn.int32, ttnn.uint16, ttnn.uint8],
    "output_dtype": [
        ttnn.bfloat16,
        ttnn.float32,
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
        ttnn.uint32,
        ttnn.int32,
        ttnn.uint16,
        ttnn.uint8,
    ],
    "low_l1": [False, True],
    "shard_api": ["none", "legacy_2d", "nd"],
    "out_scheme": [
        _INTERLEAVED,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.TensorMemoryLayout.ND_SHARDED,
    ],
    "buffer": ["dram_to_dram", "dram_to_l1", "l1_to_l1", "l1_to_dram"],
    "rank": [0, 1, 2, 3, 4, 5, 6],
    "orientation": ["none", ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
    "pad_mode": ["none", "auto", "explicit"],
    "pad_value": ["none", "zero", "positive", "negative"],
    "tile_height": [32, 16, 8, 4, 2, 1],
    "in_tile_height": ["none", 32, 16, 8, 4, 2, 1],
    "tile_grid": ["single_tile", "small", "tall_narrow", "short_wide", "square_large"],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned", "hw_non_aligned"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

# retile_tile_input reads whole input tiles and de-faces them; it has no pad-fill stage (a TILE
# input already carries its own tile padding, which becomes the output's H tail).
EXCLUSIONS = [{"in_tile_height": h, "pad_mode": mode} for h in (32, 16, 8, 4, 2, 1) for mode in ("auto", "explicit")]
# Block-float outputs share one exponent per 16-element face row; tiny tiles (< 32 rows) have no
# block-float layout in ttnn (the reference refuses them too).
EXCLUSIONS += [
    {"output_dtype": d, "tile_height": h} for d in (ttnn.bfloat8_b, ttnn.bfloat4_b) for h in (16, 8, 4, 2, 1)
]
# A rank-0 scalar into a block-float output is one lossy value: its PCC is undefined (0), so the
# value can only be checked exactly, which a 7-/3-bit block mantissa cannot meet.
EXCLUSIONS += [{"output_dtype": d, "rank": 0} for d in (ttnn.bfloat8_b, ttnn.bfloat4_b)]
# A negative fill is outside the uint16 / uint8 value domain: the bit pattern lands (two's
# complement, bit_cast), but it reads back as 2^w - |v|, never as the negative value itself.
EXCLUSIONS += [{"dtype": d, "pad_value": "negative"} for d in (ttnn.uint16, ttnn.uint8)]


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "bounded_cb": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# Live call -> scenario dict (same keys the golden scenarios carry)
# ---------------------------------------------------------------------------


def _side_dict(mem_config):
    # A live tensor's memory_config() carries BOTH shard_spec and nd_shard_spec for a legacy
    # 2D shard, and an ND spec expressible in 2D is canonicalized to HEIGHT/WIDTH/BLOCK — so
    # legacy vs nd is not recoverable at runtime. Prefer the legacy reading whenever the
    # layout is a 2D scheme; only a genuinely ND-only layout tags as nd. (Sharding support must
    # therefore land legacy_2d and nd together.)
    if mem_config.memory_layout in _SHARDED_LAYOUTS and mem_config.shard_spec is not None:
        return {
            "kind": "sharded",
            "buffer": mem_config.buffer_type,
            "orientation": mem_config.shard_spec.orientation,
            "scheme": mem_config.memory_layout,
        }
    nd_spec = getattr(mem_config, "nd_shard_spec", None)
    if nd_spec is not None:
        return {"kind": "sharded", "buffer": mem_config.buffer_type, "orientation": nd_spec.orientation, "scheme": None}
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
    # The alignment rule applies to ROW_MAJOR sources only: a TILE input already carries its
    # own tile padding, so an H that is not a multiple of the OUTPUT tile height is a retile
    # with an implicit H tail (tagged by alignment / in_tile_height), not a malformed call.
    if not padding_requested and input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT:
        if len(shape) < 2:
            raise ValueError(f"tilize: rank-{len(shape)} input requires a padding argument")
        if shape[-1] % 32 != 0 or shape[-2] % tile_h != 0:
            raise ValueError(
                f"tilize: input shape {shape} not aligned to tile ({tile_h}, 32) and no padding argument given"
            )
    if output_padded_shape is not None:
        target = [int(d) for d in output_padded_shape]
        if len(target) < len(shape) or any(t < s for t, s in zip(target[len(target) - len(shape) :], shape)):
            raise ValueError(f"tilize: output_padded_shape {target} is smaller than the input shape {shape}")
        if len(target) < 2 or target[-1] % 32 != 0 or target[-2] % tile_h != 0:
            raise ValueError(f"tilize: output_padded_shape {target} must end in multiples of ({tile_h}, 32)")
        if len(target) > _MAX_PADDED_RANK:
            raise ValueError(f"tilize: output_padded_shape rank {len(target)} > {_MAX_PADDED_RANK}")


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
    out_tile = tile if tile is not None else ttnn.Tile([32, 32])
    tile_h = _tile_height_of(out_tile)[0]

    logical_shape = [int(d) for d in input_tensor.shape]
    padded_shape = _resolve_padded_shape(logical_shape, output_padded_shape, pad_value, tile_h)

    # The output buffer is laid out at the padded shape; a metadata-only view then restores the
    # input's logical shape (no second dispatch).
    output_spec = _output_spec(
        padded_shape if padded_shape is not None else logical_shape, output_dtype, output_memory_config, out_tile
    )
    output_tensor = ttnn.allocate_tensor_on_device(output_spec, device)

    fill_word = _fill_word(input_tensor.dtype, pad_value) if padded_shape is not None else 0
    program_descriptor = create_program_descriptor(
        input_tensor, output_tensor, low_l1=low_l1, padded_shape=padded_shape, fill_word=fill_word
    )
    output_tensor = ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
    if padded_shape is not None and padded_shape != logical_shape:
        output_tensor = ttnn.experimental.view(output_tensor, ttnn.Shape(logical_shape), ttnn.Shape(padded_shape))
    return output_tensor


def _output_spec(shape, dtype, mem_config, tile):
    """TILE-layout spec at `shape` placed per `mem_config` (interleaved, legacy 2-D shard or ND shard)."""
    shape = ttnn.Shape(shape)
    if mem_config.memory_layout in _SHARDED_LAYOUTS and mem_config.shard_spec is not None:
        return ttnn.TensorSpec(
            shape,
            dtype,
            ttnn.TILE_LAYOUT,
            mem_config.memory_layout,
            mem_config.shard_spec,
            buffer_type=mem_config.buffer_type,
            tile=tile,
        )
    nd_spec = getattr(mem_config, "nd_shard_spec", None)
    if mem_config.memory_layout != _INTERLEAVED and nd_spec is not None:
        return ttnn.TensorSpec(shape, dtype, ttnn.TILE_LAYOUT, nd_spec, buffer_type=mem_config.buffer_type, tile=tile)
    return ttnn.TensorSpec(shape, dtype, ttnn.TILE_LAYOUT, buffer_type=mem_config.buffer_type, tile=tile)


def _round_up(x, unit):
    return ((x + unit - 1) // unit) * unit


def _resolve_padded_shape(shape, output_padded_shape, pad_value, tile_h):
    """The output's padded shape, or None when no padding argument was given.

    explicit: `output_padded_shape` as given. auto (`pad_value` only): last two dims rounded to
    (tile_h, 32); rank 0 / 1 synthesize the tile dims ([tile_h, 32] / [tile_h, round(W, 32)]).
    """
    if output_padded_shape is not None:
        return [int(d) for d in output_padded_shape]
    if pad_value is None:
        return None
    if len(shape) == 0:
        return [tile_h, 32]
    if len(shape) == 1:
        return [tile_h, _round_up(shape[0], 32)]
    return list(shape[:-2]) + [_round_up(shape[-2], tile_h), _round_up(shape[-1], 32)]


def _fill_word(in_dtype, pad_value):
    """The pad value encoded in the INPUT format, replicated to one 32-bit word.

    Floats convert exactly as torch does; integers bit_cast (a negative int fill is its
    two's-complement bit pattern, never truncated through a float).
    """
    import torch  # local: ttnn forbids a global torch import

    v = 0 if pad_value is None else pad_value
    if in_dtype == ttnn.bfloat16:
        # Same python-float -> bf16 conversion F.pad applies to its `value`.
        bf16 = int(torch.tensor(float(v), dtype=torch.bfloat16).view(torch.int16).item()) & 0xFFFF
        return bf16 | (bf16 << 16)
    if in_dtype == ttnn.float32:
        return int(torch.tensor(float(v), dtype=torch.float32).view(torch.int32).item()) & 0xFFFFFFFF
    if in_dtype == ttnn.fp8_e4m3:
        b = int(torch.tensor(float(v), dtype=torch.float8_e4m3fn).view(torch.uint8).item()) & 0xFF
        return b * 0x01010101
    if in_dtype in (ttnn.uint32, ttnn.int32):
        return int(v) & 0xFFFFFFFF
    if in_dtype == ttnn.uint16:
        h = int(v) & 0xFFFF
        return h | (h << 16)
    if in_dtype == ttnn.uint8:
        b = int(v) & 0xFF
        return b * 0x01010101
    raise ValueError(f"tilize: no pad-fill encoding for input dtype {in_dtype}")
