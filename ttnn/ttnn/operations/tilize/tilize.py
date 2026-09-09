# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""tilize — ROW_MAJOR -> TILE layout re-lay, one native dispatch.

Registry-model op file: the four declarations (INPUT_TAGGERS, SUPPORTED,
EXCLUSIONS, validate) plus the public entry point.

INVALID is deliberately NOT declared here — it is a test-harness concept living
in `eval/golden_tests/tilize/feature_spec.py`.

The twelve INPUT_TAGGERS project off the golden SCENARIO DICT (`inputs[0]` is
the whole dict, not a shape tuple) as specified in feature_spec.py's
`Expected INPUT_TAGGERS` block; that is a deliberate deviation from
eval/op_template.py shared with the other pure-layout ops. Because scenario-dict
taggers cannot be replayed against a live call, validate() re-derives the same
axis values from the real tensors and kwargs — the same rules
`eval/golden_tests/tilize/axes.py:classify_call` implements.
"""

from __future__ import annotations

import math

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .tilize_program_descriptor import LEGAL_TILE_HEIGHTS, TILE_WIDTH, create_program_descriptor

# Dimensionless ratio at which one tile-grid axis "dominates" the other. NOT a
# core count: the classification of a shape must be identical on every arch,
# while HOW MANY pieces to cut is read from the device at runtime.
DOMINANT = 16


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS  — each reads inputs[0], the whole scenario dict
# ---------------------------------------------------------------------------

_SHARDED_LAYOUTS = (
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
)


def _side_is_sharded(side):
    return side.get("kind") == "sharded"


def _side_is_nd(side):
    """An nd (NdShardSpec) side is a sharded side whose `scheme` is None."""
    return _side_is_sharded(side) and side.get("scheme") is None


def _buffer_word(buffer_type):
    return "dram" if buffer_type == ttnn.BufferType.DRAM else "l1"


def _out_tile_height(scenario):
    """The OUTPUT tile height the scenario asks for (32 unless `tile=` moves it)."""
    return int(scenario.get("tile_height", 32))


def tag_low_l1(inputs, axes):
    return bool(inputs[0].get("low_l1", False))


def tag_shard_api(inputs, axes):
    scenario = inputs[0]
    side_in, side_out = scenario["in"], scenario["out"]
    if not _side_is_sharded(side_in) and not _side_is_sharded(side_out):
        return "none"
    if _side_is_nd(side_in) or _side_is_nd(side_out):
        return "nd"
    return "legacy_2d"


def tag_out_scheme(inputs, axes):
    side_out = inputs[0]["out"]
    if not _side_is_sharded(side_out):
        return "interleaved"
    return "nd" if _side_is_nd(side_out) else side_out["scheme"]


def tag_buffer(inputs, axes):
    scenario = inputs[0]
    return f"{_buffer_word(scenario['in']['buffer'])}_to_{_buffer_word(scenario['out']['buffer'])}"


def tag_rank(inputs, axes):
    return int(len(inputs[0]["input_shape"]))


def tag_orientation(inputs, axes):
    """ "none" when both sides are interleaved — there is no shard to orient.
    Otherwise the sharded side's orientation (output side wins when both are)."""
    scenario = inputs[0]
    side_in, side_out = scenario["in"], scenario["out"]
    if _side_is_sharded(side_out):
        return side_out["orientation"]
    if _side_is_sharded(side_in):
        return side_in["orientation"]
    return "none"


def tag_pad_mode(inputs, axes):
    return inputs[0].get("pad_mode", "none")


def tag_pad_value(inputs, axes):
    """The SIGN bucket of the fill, not the number."""
    scenario = inputs[0]
    if scenario.get("pad_mode", "none") == "none":
        return "none"
    value = scenario.get("pad_value")
    if value is None or value == 0:
        return "zero"
    return "positive" if value > 0 else "negative"


def tag_tile_height(inputs, axes):
    return _out_tile_height(inputs[0])


def tag_in_tile_height(inputs, axes):
    """ "none" IS row-major: a ROW_MAJOR input has no tile geometry of its own."""
    return inputs[0].get("in_tile_height", "none")


def tag_alignment(inputs, axes):
    """Four-way alignment of the last two dims. H is measured against the
    OUTPUT TILE HEIGHT (a tiny-tile call redefines "aligned" on that axis);
    W against the literal 32, a tile's width always being 32. Rank < 2 is
    hw_non_aligned — both tile dims are synthesized by the pad."""
    scenario = inputs[0]
    shape = scenario["input_shape"]
    if len(shape) < 2:
        return "hw_non_aligned"
    tile_h = _out_tile_height(scenario)
    h_ok = int(shape[-2]) % tile_h == 0
    w_ok = int(shape[-1]) % TILE_WIDTH == 0
    if h_ok and w_ok:
        return "tile_aligned"
    if h_ok:
        return "w_non_aligned"
    if w_ok:
        return "h_non_aligned"
    return "hw_non_aligned"


def tag_tile_grid(inputs, axes):
    """Which axis of the OUTPUT tile grid carries the parallelism.

    Holds NO grid size: DOMINANT is a dimensionless ratio between R and C, so
    the classification is identical on Wormhole, Blackhole and Quasar.
    """
    scenario = inputs[0]
    shape = scenario["input_shape"]
    if len(shape) < 2:
        return "single_tile"
    tile_h = _out_tile_height(scenario)
    leading = 1
    for d in shape[:-2]:
        leading *= int(d)
    rows = leading * math.ceil(int(shape[-2]) / tile_h)
    cols = math.ceil(int(shape[-1]) / TILE_WIDTH)
    return _classify_tile_grid(rows, cols)


def _classify_tile_grid(rows, cols):
    if rows == 1 and cols == 1:
        return "single_tile"
    if cols >= DOMINANT * rows:
        return "short_wide"
    if rows >= DOMINANT * cols:
        return "tall_narrow"
    if rows * cols < DOMINANT * DOMINANT:
        return "small"
    return "square_large"


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
    "alignment": tag_alignment,
    "tile_grid": tag_tile_grid,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED — Phase 0
# ---------------------------------------------------------------------------
#
# `tile_grid` is declared COMPLETE, and that is a claim about the work split,
# not about correctness (every value of this axis is trivially correct under any
# scheme). The claim: the block grid is `num_row_groups x num_w_chunks` and BOTH
# indices go into `split_work_to_cores`, so the distribution reaches the full
# device grid on every geometry — including `short_wide`, where R == 1 and a
# row-only split would collapse onto one core. See op_design.md's worked table
# and tilize_program_descriptor.derive_plan.
#
# The five absent-argument sentinels (shard_api / orientation / pad_mode /
# pad_value / in_tile_height) carry "none", which is ALWAYS legal — four of the
# Phase 0 values below ARE sentinels.
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
    "tile_grid": ["single_tile", "small", "tall_narrow", "short_wide", "square_large"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS — cells inside cartesian(SUPPORTED) refused for now
# ---------------------------------------------------------------------------
EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 3b. PROPERTIES — non-axis capabilities
# ---------------------------------------------------------------------------
PROPERTIES = {
    # The block grid is 2-D and both indices are core-assignment indices; the
    # core count comes from device.compute_with_storage_grid_size() at runtime.
    "multi_core": {"value": True, "source": "declared"},
    # Per-core L1 = INPUT_DEPTH_ROWS*W*tb_in + OUTPUT_DEPTH_BATCHES*wrpb*W*tb_out
    # with W <= W_FIT, a constant of (dtype, tile_h, device). No tensor
    # dimension appears — see l1_ledger.md.
    "bounded_cb": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# Malformed-request checks (ValueError / RuntimeError) — NOT support refusals
# ---------------------------------------------------------------------------


def _tile_shape(tile):
    shape = getattr(tile, "tile_shape", None) or getattr(tile, "shape", None)
    return (int(shape[0]), int(shape[1]))


def _check_request(input_tensor, *, output_padded_shape, tile):
    """Malformed REQUESTS. These run ahead of the per-axis support loop so a
    malformed call never reaches the support rectangle."""
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("tilize: input_tensor must be on device")

    if input_tensor.layout not in (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT):
        raise ValueError(f"tilize: input layout must be ROW_MAJOR_LAYOUT or TILE_LAYOUT, got {input_tensor.layout}")

    if input_tensor.layout == ttnn.TILE_LAYOUT and tile is None:
        raise ValueError(
            "tilize: a TILE_LAYOUT input needs an explicit `tile=` to re-tile to — there is nothing to re-tile to otherwise"
        )

    if tile is not None:
        tile_h, tile_w = _tile_shape(tile)
        if tile_w != TILE_WIDTH:
            raise ValueError(f"tilize: `tile` width must be {TILE_WIDTH}, got {tile_w}")
        if tile_h not in LEGAL_TILE_HEIGHTS:
            raise ValueError(f"tilize: `tile` height must be a power-of-two fraction of 32, got {tile_h}")

    shape = list(input_tensor.shape)
    if output_padded_shape is not None:
        target = [int(d) for d in list(output_padded_shape)]
        if len(target) != len(shape):
            raise ValueError(f"tilize: output_padded_shape rank {len(target)} != input rank {len(shape)}")
        for i, (t, s) in enumerate(zip(target, shape)):
            if t < s:
                raise ValueError(f"tilize: output_padded_shape[{i}]={t} is smaller than the input's {s}")


def _check_alignment_request(axes, *, pad_requested):
    """Padding is opt-in: with no padding argument, a non-tile-aligned input is
    refused rather than silently padded. A malformed REQUEST, not a support
    refusal — hence ValueError and not UnsupportedAxisValue."""
    if not pad_requested and axes["alignment"] != "tile_aligned":
        raise ValueError(
            "tilize: input's last two dims are not tile-aligned "
            f"({axes['alignment']}) and no padding argument was given; "
            "pass pad_value= or output_padded_shape= to opt into padding"
        )


# ---------------------------------------------------------------------------
# Runtime axis derivation (the live-call analogue of INPUT_TAGGERS)
# ---------------------------------------------------------------------------


def _spec_of(mem_config):
    """(is_sharded, is_nd, out_scheme, orientation) for one side of the call."""
    nd_spec = getattr(mem_config, "nd_shard_spec", None)
    if nd_spec is not None:
        return True, True, "nd", nd_spec.orientation
    if mem_config.memory_layout in _SHARDED_LAYOUTS:
        shard_spec = mem_config.shard_spec
        return (
            True,
            False,
            mem_config.memory_layout,
            shard_spec.orientation if shard_spec is not None else "none",
        )
    return False, False, "interleaved", "none"


def _in_tile_height(input_tensor):
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        return None
    tile = getattr(input_tensor, "tile", None)
    return _tile_shape(tile)[0] if tile is not None else 32


def _axes_from_call(input_tensor, memory_config, dtype, low_l1, output_padded_shape, pad_value, tile):
    in_cfg = input_tensor.memory_config()
    out_cfg = memory_config if memory_config is not None else in_cfg
    in_sharded, in_nd, _in_scheme, in_orientation = _spec_of(in_cfg)
    out_sharded, out_nd, out_scheme, out_orientation = _spec_of(out_cfg)

    if not in_sharded and not out_sharded:
        shard_api = "none"
    elif in_nd or out_nd:
        shard_api = "nd"
    else:
        shard_api = "legacy_2d"

    if output_padded_shape is not None:
        pad_mode = "explicit"
    elif pad_value is not None:
        pad_mode = "auto"
    else:
        pad_mode = "none"

    if pad_mode == "none":
        pad_bucket = "none"
    elif pad_value is None or pad_value == 0:
        pad_bucket = "zero"
    else:
        pad_bucket = "positive" if pad_value > 0 else "negative"

    # The OUTPUT tile height: `tile=` when given, else the op keeps the input's
    # geometry (its own tile for a TILE input, 32 for ROW_MAJOR).
    out_tile_h = (_tile_shape(tile)[0] if tile is not None else None) or _in_tile_height(input_tensor) or 32

    shape = list(input_tensor.shape)
    scenario = {
        "input_shape": shape,
        "in": {"kind": "interleaved" if not in_sharded else "sharded"},
        "out": {"kind": "interleaved" if not out_sharded else "sharded"},
        "tile_height": out_tile_h,
    }

    return {
        "dtype": input_tensor.dtype,
        "output_dtype": dtype if dtype is not None else input_tensor.dtype,
        "low_l1": bool(low_l1),
        "shard_api": shard_api,
        "out_scheme": out_scheme,
        "buffer": f"{_buffer_word(in_cfg.buffer_type)}_to_{_buffer_word(out_cfg.buffer_type)}",
        "rank": len(shape),
        "orientation": (out_orientation if out_sharded else in_orientation if in_sharded else "none"),
        "pad_mode": pad_mode,
        "pad_value": pad_bucket,
        # tag_alignment / tag_tile_grid reuse the tagger bodies verbatim so the
        # runtime rule and the declared rule cannot drift.
        "alignment": tag_alignment((scenario,), None),
        "tile_height": out_tile_h,
        "in_tile_height": _in_tile_height(input_tensor) or "none",
        "tile_grid": tag_tile_grid((scenario,), None),
    }


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


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
    """Malformed-request checks first (ValueError / RuntimeError), then the
    registry support gate (UnsupportedAxisValue / ExcludedCell)."""
    _check_request(input_tensor, output_padded_shape=output_padded_shape, tile=tile)

    axes = _axes_from_call(input_tensor, memory_config, dtype, low_l1, output_padded_shape, pad_value, tile)

    _check_alignment_request(axes, pad_requested=(axes["pad_mode"] != "none"))

    # 1. SUPPORTED — per-axis
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"tilize: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    # 2. EXCLUSIONS — cell-level inside SUPPORTED
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"tilize: unsupported combination (refinement candidate): {exc}")

    return axes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def tilize(
    input_tensor: ttnn.Tensor,
    memory_config: "ttnn.MemoryConfig | None" = None,
    *,
    dtype: "ttnn.DataType | None" = None,
    low_l1: bool = False,
    output_padded_shape=None,
    pad_value=None,
    tile: "ttnn.Tile | None" = None,
) -> ttnn.Tensor:
    """Re-lay `input_tensor` from ROW_MAJOR into TILE layout on device.

    Values and logical positions are preserved; only the addresses move. The
    output carries the input's logical shape unchanged, the input's dtype unless
    `dtype=` is given, the input's placement unless `memory_config=` is given,
    and the tile geometry `tile=` names (32x32 otherwise).
    """
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
    out_memory_config = memory_config if memory_config is not None else input_tensor.memory_config()
    out_dtype = dtype if dtype is not None else input_tensor.dtype
    out_tile = tile if tile is not None else ttnn.Tile([32, TILE_WIDTH])

    # The output's LOGICAL shape is the input's. A padded call would grow only
    # the padded shape, which the TensorSpec derives from (logical shape, tile).
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(
            ttnn.Shape(list(input_tensor.shape)),
            out_dtype,
            ttnn.TILE_LAYOUT,
            out_memory_config.memory_layout,
            out_memory_config.shard_spec,
            out_memory_config.buffer_type,
            out_tile,
        ),
        device,
    )

    program_descriptor = create_program_descriptor(input_tensor, output_tensor, low_l1=low_l1)
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
