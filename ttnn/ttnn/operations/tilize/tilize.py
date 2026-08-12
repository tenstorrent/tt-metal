# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""tilize — re-lay a ROW_MAJOR tensor into TILE layout (registry model).

No arithmetic: `output[tile_index(i)] = input[i]` is a bijection on byte
positions. See `op_design.md` for the binding design; this file carries the four
registry declarations (INPUT_TAGGERS / SUPPORTED / EXCLUSIONS / validate) plus
the public entry point.

Axis names are exactly the ones `eval/golden_tests/tilize/feature_spec.py`
TARGET uses, and every tagger reads the SAME scenario-dict shape the golden
suite passes — `validate()` synthesizes that scenario from the live call
(`_scenario_from_call`) so the op and the registry can never disagree about
which cell a call lands in.

INVALID is deliberately NOT declared here (it lives in feature_spec.py; the
harness skips those cells before the op runs).
"""

from __future__ import annotations

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .tilize_program_descriptor import (
    DEFAULT_TILE_HEIGHT,
    LEGACY_SHARD_SCHEMES,
    TILE_WIDTH,
    auto_padded_shape,
    create_program_descriptor,
    output_tensor_spec,
    pad_plan,
    plan_placement,
    tile_geometry,
)

# ---------------------------------------------------------------------------
# scenario helpers — the one place a MemoryConfig is projected onto a spec dict
# ---------------------------------------------------------------------------

_LEGACY_SCHEMES = LEGACY_SHARD_SCHEMES  # one source of truth (program descriptor)


def _spec_from_memory_config(memory_config):
    """The scenario `in`/`out` spec dict for a live MemoryConfig."""
    if not memory_config.is_sharded():
        return {"kind": "interleaved", "buffer": memory_config.buffer_type}

    layout = memory_config.memory_layout
    if layout in _LEGACY_SCHEMES:
        shard_spec = memory_config.shard_spec
        return {
            "kind": "sharded",
            "buffer": memory_config.buffer_type,
            "grid": shard_spec.grid,
            "shard_shape": tuple(shard_spec.shape),
            "orientation": shard_spec.orientation,
            "scheme": layout,
        }
    nd_spec = memory_config.nd_shard_spec
    return {
        "kind": "sharded",
        "buffer": memory_config.buffer_type,
        "grid": nd_spec.grid,
        "shard_shape": tuple(nd_spec.shard_shape),
        "orientation": nd_spec.orientation,
        "scheme": None,  # nd
    }


def _shard_api(in_spec, out_spec):
    specs = [s for s in (in_spec, out_spec) if s["kind"] == "sharded"]
    if not specs:
        return "none"
    return "nd" if any(s["scheme"] is None for s in specs) else "legacy_2d"


def _buffer_name(spec):
    return "dram" if spec["buffer"] == ttnn.BufferType.DRAM else "l1"


def _scenario_from_call(
    input_tensor,
    out_memory_config,
    *,
    use_multicore,
    use_double_buffer,
    pad_mode,
    output_padded_shape,
    pad_value,
    tile_height,
):
    """Build the golden-suite scenario dict for a live call.

    Single chokepoint: the taggers below are then applied to exactly the same
    shape of input the golden harness feeds them.
    """
    in_spec = _spec_from_memory_config(input_tensor.memory_config())
    out_spec = _spec_from_memory_config(out_memory_config)
    scenario = {
        "input_shape": list(input_tensor.shape),
        "use_multicore": bool(use_multicore),
        "use_double_buffer": bool(use_double_buffer),
        "shard_api": _shard_api(in_spec, out_spec),
        "in": in_spec,
        "out": out_spec,
        "pad_mode": pad_mode,
        "tile_height": tile_height,
        "in_layout": input_tensor.layout,
    }
    if pad_mode != "none":
        scenario["pad_value"] = pad_value
    if output_padded_shape is not None:
        scenario["output_padded_shape"] = list(output_padded_shape)
    if input_tensor.layout == ttnn.TILE_LAYOUT:
        scenario["in_tile_height"] = int(list(input_tensor.tile.tile_shape)[0])
    return scenario


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS  (inputs[0] is the scenario dict)
# ---------------------------------------------------------------------------


def tag_use_multicore(inputs, axes):
    return bool(inputs[0]["use_multicore"])


def tag_shard_api(inputs, axes):
    return inputs[0]["shard_api"]


def tag_out_scheme(inputs, axes):
    out = inputs[0]["out"]
    if out["kind"] == "interleaved":
        return "interleaved"
    return "nd" if out["scheme"] is None else out["scheme"]


def tag_buffer(inputs, axes):
    s = inputs[0]
    return f"{_buffer_name(s['in'])}_to_{_buffer_name(s['out'])}"


def tag_rank(inputs, axes):
    return int(len(inputs[0]["input_shape"]))


def tag_double_buffer(inputs, axes):
    return bool(inputs[0].get("use_double_buffer", True))


def tag_pad_mode(inputs, axes):
    return inputs[0].get("pad_mode", "none")


def tag_pad_value(inputs, axes):
    s = inputs[0]
    if s.get("pad_mode", "none") == "none":
        return "none"
    value = s.get("pad_value")
    if value is None:
        return "none"
    if value == 0:
        return "zero"
    return "positive" if value > 0 else "negative"


def tag_alignment(inputs, axes):
    """H measured against the scenario's TILE HEIGHT (not a hardcoded 32) — a
    tiny-tile call redefines what "aligned" means on the H axis."""
    s = inputs[0]
    shape = s["input_shape"]
    if len(shape) < 2:
        return "hw_non_aligned"  # both tile dims are synthesized by the pad
    tile_h = int(s.get("tile_height", DEFAULT_TILE_HEIGHT))
    h_aligned = shape[-2] % tile_h == 0
    w_aligned = shape[-1] % TILE_WIDTH == 0
    if h_aligned and w_aligned:
        return "tile_aligned"
    if h_aligned:
        return "w_non_aligned"
    if w_aligned:
        return "h_non_aligned"
    return "hw_non_aligned"


def tag_orientation(inputs, axes):
    s = inputs[0]
    if s["in"]["kind"] == "sharded":
        return s["in"]["orientation"]
    if s["out"]["kind"] == "sharded":
        return s["out"]["orientation"]
    return "none"


def tag_tile_height(inputs, axes):
    return int(inputs[0].get("tile_height", DEFAULT_TILE_HEIGHT))


def tag_in_layout(inputs, axes):
    return inputs[0].get("in_layout", ttnn.ROW_MAJOR_LAYOUT)


def tag_in_tile_height(inputs, axes):
    """The "none" sentinel exactly when the input is ROW_MAJOR (a row-major
    tensor has no tile geometry of its own)."""
    s = inputs[0]
    if s.get("in_layout", ttnn.ROW_MAJOR_LAYOUT) != ttnn.TILE_LAYOUT:
        return "none"
    return int(s.get("in_tile_height", DEFAULT_TILE_HEIGHT))


INPUT_TAGGERS = {
    "use_multicore": tag_use_multicore,
    "shard_api": tag_shard_api,
    "out_scheme": tag_out_scheme,
    "buffer": tag_buffer,
    "rank": tag_rank,
    "double_buffer": tag_double_buffer,
    "pad_mode": tag_pad_mode,
    "pad_value": tag_pad_value,
    "alignment": tag_alignment,
    "orientation": tag_orientation,
    "tile_height": tag_tile_height,
    "in_layout": tag_in_layout,
    "in_tile_height": tag_in_tile_height,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED — Phase 0 rectangle (narrow, but one entry per TARGET axis)
# ---------------------------------------------------------------------------
#
# Phase 0 = A0: interleaved DRAM->DRAM, single-core, bf16, rank 4, no padding,
# 32x32 tiles, ROW_MAJOR in. Every knob the design names is already a live
# parameter in the program descriptor (grid_cores, CB_DEPTH, WT_BLOCK,
# NEEDS_CAST, tile_h) — the refinements flip SUPPORTED entries, they do not add
# kernel code paths for these axes.
#
# Refinement 1 (A1 + A5 + A6) — the interleaved path at full generality. Four
# axes flip, none of which needs a kernel-source change:
#   use_multicore  += True   the 2-D `b = wchunk*nt_h + r` split IS the only code
#                            path; use_multicore=False is its grid_cores=1 value.
#   rank           += 2,3,5  `nimg = prod(shape[:-2])` is rank-agnostic.
#   buffer         += the three L1 directions — a TensorAccessor buffer-type
#                            difference, already baked as a CT arg.
#   double_buffer  += False  CB_DEPTH is already `2 if use_double_buffer and
#                            depth2_fits_l1 else 1`.

#
# Refinement 5 (P1 + P2 + P4 + P5) — the padded path. Four axes flip, behind ONE
# CT-selected reader body (`pad_enabled`), never four features:
#   pad_mode   += auto, explicit  auto tile-rounds the last two dims; explicit
#                            honours any tile-multiple target, including one
#                            BEYOND the tile-rounded shape (whole pad tiles).
#   pad_value  += zero/positive/negative — the SIGN buckets exist to catch a
#                            fill written once instead of replicated across the
#                            32-bit store word (`pad_fill_word`).
#   alignment  += w/h/hw_non_aligned — the pad is what makes them legal at all;
#                            without a pad argument they still RAISE.
#   rank       += 0          the scalar -> one tile case, reachable only with a
#                            pad (its tile dims are synthesized by the target).
#
# Refinement 7 (A4 + A5b + P3) — the numeric surface. Two axes flip, and the
# mechanism behind them is `numeric_policy()` in the program descriptor, which
# turns the (in_dtype, out_dtype) pair into ONE ComputeConfigDescriptor plus ONE
# kernel template argument:
#   dtype        += float32, uint32/uint16/int32, uint8. The cast is already the
#                   CT `needs_cast` flag -> UnpackAndPackReconfigure; what R7 adds
#                   is the two formats that are not merely "another width":
#                     fp32 -> fp32 needs Fp32Mode::Lossless + fp32_dest_acc_en +
#                       UnpackToDestFp32 (Fast truncates fp32 -> tf32: measured
#                       max diff 1.6e-2 on an op whose contract is a BIJECTION).
#                     uint8 needs a 32-bit DEST (fp32_dest_acc_en); with the
#                       16-bit DEST the packer reads the int8 payload as a float
#                       denormal and the whole tile comes back ZERO (measured).
#   output_dtype += float32, bfloat8_b, uint32/uint16/int32, uint8. bfloat8_b is
#                   output-only, and only an fp32 INPUT pays for the precise
#                   packer (the fast packer clears the golden PCC gate from bf16
#                   at ~1.4x less cost).
# P3 (the padded path per dtype) needed no new code: `pad_fill_word` already
# packs in the INPUT format and replicates across the 32-bit store word (4x for
# uint8, 2x for bf16/uint16, 1x for fp32/uint32/int32), with the signed->unsigned
# bit_cast for a negative integer fill.
#
# Refinement 2 (A3 + A3b + A3d + A5c) — sharded placement. Three axes flip, and
# the mechanism behind them is `plan_placement()` in the program descriptor:
#   shard_api   += legacy_2d, nd  both APIs project onto one ShardSpec view.
#   out_scheme  += HEIGHT/WIDTH/BLOCK/nd — the scheme picks the shard's shape,
#                            and the shard IS the per-core block either way.
#   orientation += ROW_MAJOR, COL_MAJOR — a non-issue on the zero-copy path
#                            (each core tilizes the block in its own L1, so it
#                            never needs to know which shard that is).

SUPPORTED = {
    # Refinement 7 (A4 + A5b + P3). `bfloat8_b` is OUTPUT-ONLY — block-float has
    # no row-major form, and the tilize helper ASSERTs a non-block-float input.
    # uint16 / int32 are beyond the golden TARGET (which collapses the wider
    # integers onto uint32) but are the op's declared dtype family
    # (op_design.md §3) and are exercised by `test_regression.py`, so they are
    # declared rather than silently refused.
    "dtype": [ttnn.bfloat16, ttnn.float32, ttnn.uint32, ttnn.uint16, ttnn.int32, ttnn.uint8],
    "output_dtype": [
        ttnn.bfloat16,
        ttnn.float32,
        ttnn.bfloat8_b,
        ttnn.uint32,
        ttnn.uint16,
        ttnn.int32,
        ttnn.uint8,
    ],
    "use_multicore": [False, True],
    "double_buffer": [False, True],
    "shard_api": ["none", "legacy_2d", "nd"],
    "out_scheme": [
        "interleaved",
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "nd",
    ],
    "buffer": ["dram_to_dram", "dram_to_l1", "l1_to_l1", "l1_to_dram"],
    # Rank 0 is padding-only (a scalar has no tile dims of its own), so it
    # arrives with the pad axes below rather than with Refinement 1's rank flip.
    "rank": [0, 2, 3, 4, 5],
    "pad_mode": ["none", "auto", "explicit"],
    "pad_value": ["none", "zero", "positive", "negative"],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned", "hw_non_aligned"],
    "orientation": ["none", ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
    # Refinement 8 (T1 + T2) — tile geometry. `tile_height` is the OUTPUT tile's
    # height; every legal sub-32 value of `Tile({h, 32})` is accepted, and
    # nothing in the op keys on 32 (see `tag_alignment` / `DEFAULT_TILE_HEIGHT`).
    "tile_height": [DEFAULT_TILE_HEIGHT, 16, 8, 4, 2, 1],
    # T2 — the RETILE path: an already-tiled input re-laid at a different tile
    # height. `"none"` is the ROW_MAJOR sentinel and is always legal.
    "in_layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
    "in_tile_height": ["none", DEFAULT_TILE_HEIGHT, 16, 8, 4, 2, 1],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS — cells inside cartesian(SUPPORTED) refused *for now*
# ---------------------------------------------------------------------------
#
# A sharded input is inherently multi-core (its cores are fixed by the shard
# spec), so single-core + sharded is refused rather than unsupported forever.

EXCLUSIONS = [
    {"use_multicore": False, "shard_api": "legacy_2d"},
    {"use_multicore": False, "shard_api": "nd"},
    # Refinement 8 (T1) — BLOCK-FLOAT OUTPUT below a 16-row FACE. A bfloat8_b
    # tile carries a shared-exponent section whose size the packer programs as
    # `exp_section_size = partial_face ? 1 : num_faces`
    # (`cpack_common.h`), while the section a sub-16 tile actually needs is
    # `face_r_dim * num_faces` bytes (`Tile::get_tile_size`) — the register
    # cannot express it once `face_r_dim < 16`. `tile.cpp`'s
    # TILE_FACE_HW_CHOICES says the same thing in words: 8x32 and below are
    # "not supported yet on llk, just for host loopback".
    #
    # MEASURED, per height, bf16 -> bfloat8_b on [1,1,128,256] (max |diff| vs
    # the torch source; a correct bfp8 round-trip is ~0.03 on this data):
    #   tile_height 32 -> 0.037   16 -> 0.037   (both fine, face_r_dim == 16)
    #   tile_height  8 -> 7.15     4 -> 6.55     2 -> 6.46     1 -> 6.63
    # Not a packer-mode question: `bfp8_precise` 0/1 and an fp32 vs bf16 input
    # all give the identical wrong number. Every NON-block-float dtype is
    # bit-exact at every one of these heights, so the gap is block-float's
    # exponent section and nothing else.
    #
    # `tile_height: 16` is deliberately NOT here — it keeps `face_r_dim == 16`
    # (`face_shape` is `{min(h,16), 16}`) and measures correct.
    {"output_dtype": ttnn.bfloat8_b, "tile_height": 8},
    {"output_dtype": ttnn.bfloat8_b, "tile_height": 4},
    {"output_dtype": ttnn.bfloat8_b, "tile_height": 2},
    {"output_dtype": ttnn.bfloat8_b, "tile_height": 1},
]


# ---------------------------------------------------------------------------
# 3b. PROPERTIES
# ---------------------------------------------------------------------------

PROPERTIES = {
    # The 2-D (tile-row x tile-column) split is wired and parameterized by
    # `grid_cores`, and Refinement 1 (A1) flipped use_multicore=True into
    # SUPPORTED. The core count is ASSERTED, not inferred:
    # test_tilize_debug.py::test_multicore_fills_the_grid_on_wide_short pins
    # len(cores) == min(total_blocks, grid_cores) in every regime.
    "multi_core": {"value": True, "source": "verified"},
    # per-core CB L1 = CB_DEPTH * WT_BLOCK * (in_tile + out_tile), and
    # WT_BLOCK = min(Wt, WT_BLOCK_MAX) -> independent of H, W, Wt, rank, batch.
    # On a RESIDENT (zero-copy) side the CB *is* the caller's shard, so it costs
    # no L1 beyond the tensor the caller already allocated; A3d's clamp keeps the
    # streamed side of a crossover bounded too (a wide shard downgrades to the
    # byte-target block width rather than growing the CB with W).
    "bounded_cb": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _derive_pad_mode(output_padded_shape, pad_value):
    if output_padded_shape is None and pad_value is None:
        return "none"
    if output_padded_shape is None:
        return "auto"
    return "explicit"


def _check_structural(input_tensor, scenario, output_padded_shape, pad_value):
    """The two structural refusals (ValueError, message mentions `pad`) plus
    argument-coherence checks. These are contract violations, NOT support gaps,
    so they are raised BEFORE the SUPPORTED/EXCLUSIONS gate and do not use the
    registry refusal types."""
    pad_mode = scenario["pad_mode"]
    shape = scenario["input_shape"]

    if output_padded_shape is not None and pad_value is None:
        raise ValueError(
            "tilize: output_padded_shape given without pad_value — padding needs a "
            "fill value. Pass pad_value=<fill> to pad, or drop output_padded_shape."
        )

    # Retile and padding are mutually exclusive: a TILE input's last two dims
    # are tile multiples by construction, so there is nothing to pad.
    if input_tensor.layout == ttnn.TILE_LAYOUT and pad_mode != "none":
        raise ValueError(
            "tilize: a TILE_LAYOUT input (retile path) cannot be padded — its last "
            "two dims are tile multiples by construction. Drop pad_value / "
            "output_padded_shape, or pass a ROW_MAJOR input."
        )

    # Padding is never implicit — a ROW_MAJOR-input contract only. A TILE input
    # (the retile path) carries its OWN tile geometry and is tile-aligned in it
    # by construction; measuring it against the REQUESTED output tile height
    # would turn a legal retile (e.g. H=16 at in_tile_height=16, retiled to 32)
    # into a bogus "you must ask for padding" ValueError, hiding the honest
    # refusal (`in_layout` is not in SUPPORTED yet). feature_spec INVALID rule 4
    # says the same thing: a TILE input can be neither padded nor non-aligned.
    if pad_mode == "none" and input_tensor.layout != ttnn.TILE_LAYOUT:
        tile_h = scenario["tile_height"]
        if len(shape) < 2:
            raise ValueError(
                f"tilize: rank-{len(shape)} input has no tile dims of its own; it can "
                "only be tilized as a pad target. Pass pad_value= (and optionally "
                "output_padded_shape=) to pad it up to a tile."
            )
        if shape[-2] % tile_h or shape[-1] % TILE_WIDTH:
            raise ValueError(
                f"tilize: input last two dims {tuple(shape[-2:])} are not multiples of "
                f"({tile_h}, {TILE_WIDTH}) and no padding was requested. Padding is "
                "never implicit — pass pad_value= to pad up to the next tile multiple, "
                "or output_padded_shape= with pad_value= for an explicit pad target."
            )

    if pad_mode == "explicit":
        target = list(output_padded_shape)
        tile_h = scenario["tile_height"]
        padded_in = list(shape) if len(shape) >= 2 else [1] * (len(target) - len(shape)) + list(shape)
        if len(target) < 2:
            raise ValueError(f"tilize: pad target {target} must have rank >= 2")
        if len(target) != len(padded_in):
            raise ValueError(f"tilize: pad target rank {len(target)} does not match input rank {len(shape)}")
        if any(t < i for t, i in zip(target, padded_in)):
            raise ValueError(f"tilize: pad target {target} must be >= the input shape {list(shape)} in every dim")
        if target[-2] % tile_h or target[-1] % TILE_WIDTH:
            raise ValueError(
                f"tilize: pad target last two dims {tuple(target[-2:])} must be multiples "
                f"of ({tile_h}, {TILE_WIDTH})"
            )


def validate(
    input_tensor,
    memory_config=None,
    *,
    dtype=None,
    use_multicore=True,
    use_double_buffer=True,
    output_padded_shape=None,
    pad_value=None,
    tile=None,
):
    """Runtime gate. Structural refusals first (ValueError), then the registry
    contract: SUPPORTED per-axis, then EXCLUSIONS cell-level."""
    out_memory_config = memory_config if memory_config is not None else input_tensor.memory_config()
    out_dtype = dtype if dtype is not None else input_tensor.dtype
    tile_shape = list(tile.tile_shape) if tile is not None else [DEFAULT_TILE_HEIGHT, TILE_WIDTH]
    if tile_shape[1] != TILE_WIDTH:
        raise ValueError(f"tilize: tile width must be {TILE_WIDTH}, got {tile_shape[1]}")
    tile_height = int(tile_shape[0])

    pad_mode = _derive_pad_mode(output_padded_shape, pad_value)
    scenario = _scenario_from_call(
        input_tensor,
        out_memory_config,
        use_multicore=use_multicore,
        use_double_buffer=use_double_buffer,
        pad_mode=pad_mode,
        output_padded_shape=output_padded_shape,
        pad_value=pad_value,
        tile_height=tile_height,
    )

    _check_structural(input_tensor, scenario, output_padded_shape, pad_value)

    axes = {"dtype": input_tensor.dtype, "output_dtype": out_dtype}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

    # 1. SUPPORTED — per-axis
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"tilize: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    # 2. EXCLUSIONS — cell-level inside SUPPORTED
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"tilize: unsupported combination (refinement candidate): {exc}")

    # 3. Placement — the ONE sharded property no registry axis can carry, because
    #    it is a relation between the two specs and the shape rather than a value
    #    of either. `plan_placement` is the same function the descriptor builds
    #    from, so a refusal here can never disagree with what the op can do.
    _check_retile(input_tensor, tile_height)
    _check_placement(
        input_tensor,
        out_memory_config,
        tile_height,
        pad_plan(
            list(input_tensor.shape),
            tile_height,
            output_padded_shape,
            pad_value,
            input_tensor.dtype,
            input_tensor.element_size(),
        ),
    )

    return scenario, axes


def _check_retile(input_tensor, tile_height):
    """R8 (T2): the two geometry facts the retile reader needs and cannot pad for.

    The reader projects output row `g` onto source tile-row `g // in_tile_h`,
    row `g % in_tile_h` — exact only when H is a whole number of BOTH tile
    heights. It has no pad body (a TILE input's last two dims are tile multiples
    by construction, which is why retile + pad is refused outright), so a
    mismatch would read the input's own tile-padding rows as if they were data.
    A typed support refusal, not a ValueError: `feature_spec.py` INVALID rule 4
    already declares a non-aligned TILE input structurally impossible, so
    nothing the golden suite runs reaches this.
    """
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        return
    shape = list(input_tensor.shape)
    in_tile_h = int(list(input_tensor.tile.tile_shape)[0])
    h = shape[-2] if len(shape) >= 2 else 1
    w = shape[-1] if len(shape) >= 1 else 1
    if h % in_tile_h or h % tile_height or w % TILE_WIDTH:
        raise UnsupportedAxisValue(
            f"tilize: a retile of {tuple(shape[-2:])} from a {in_tile_h}x{TILE_WIDTH} tile to a "
            f"{tile_height}x{TILE_WIDTH} one needs H a multiple of both tile heights and W a "
            f"multiple of {TILE_WIDTH}; a TILE input carries no pad path to make up the difference"
        )


def _check_placement(input_tensor, out_memory_config, tile_height, pad=None):
    """Refuse a shard geometry neither placement mechanism can address.

    Every sharded cell in the golden TARGET is either same-spec (zero-copy on
    both sides), a crossover (zero-copy on the sharded side), or cross-spec with
    a full-row-width input shard (streamed) — so nothing in SUPPORTED lands here.
    What does: a ROW_MAJOR input whose shard is NARROWER than a row and is not
    L1-resident (e.g. a DRAM width-shard), where the streamed reader's stick
    indexing would silently read the wrong bytes.
    """
    elem_size = input_tensor.element_size()
    in_shape = list(input_tensor.shape)
    pad_enabled = bool(pad is not None and pad["enabled"])
    shape = list(pad["padded_shape"]) if pad_enabled else in_shape
    nt_h, Wt, _, _ = tile_geometry(shape, tile_height)
    in_tile_bytes = tile_height * TILE_WIDTH * elem_size
    plan = plan_placement(
        shape=shape,
        tile_height=tile_height,
        in_memory_config=input_tensor.memory_config(),
        out_memory_config=out_memory_config,
        Wt=Wt,
        nt_h=nt_h,
        in_tile_bytes=in_tile_bytes,
        out_tile_bytes=in_tile_bytes,  # only the RESIDENT/STREAMED choice matters here
        in_shape=in_shape,
        pad_enabled=pad_enabled,
        retile=input_tensor.layout == ttnn.TILE_LAYOUT,
    )
    if plan["error"] is not None:
        raise UnsupportedAxisValue(plan["error"])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def tilize(
    input_tensor: ttnn.Tensor,
    memory_config: "ttnn.MemoryConfig | None" = None,
    *,
    dtype: "ttnn.DataType | None" = None,
    use_multicore: bool = True,
    use_double_buffer: bool = True,
    output_padded_shape=None,
    pad_value=None,
    tile: "ttnn.Tile | None" = None,
) -> ttnn.Tensor:
    """ROW_MAJOR -> TILE layout conversion (see module docstring / op_design.md)."""
    validate(
        input_tensor,
        memory_config,
        dtype=dtype,
        use_multicore=use_multicore,
        use_double_buffer=use_double_buffer,
        output_padded_shape=output_padded_shape,
        pad_value=pad_value,
        tile=tile,
    )
    return _dispatch(
        input_tensor,
        memory_config,
        dtype=dtype,
        use_multicore=use_multicore,
        use_double_buffer=use_double_buffer,
        output_padded_shape=output_padded_shape,
        pad_value=pad_value,
        tile=tile,
    )


def _dispatch(
    input_tensor,
    memory_config=None,
    *,
    dtype=None,
    use_multicore=True,
    use_double_buffer=True,
    output_padded_shape=None,
    pad_value=None,
    tile=None,
    levers=None,
):
    """Allocate the output and launch the generic op.

    Separated from `tilize()` so (a) the multi-core / double-buffer values of the
    distribution parameters can be exercised (bench + refinement tests) while
    Phase 0's SUPPORTED rectangle still only *accepts* their trivial values, and
    (b) the perf bench can force a lever off (`levers=dict(<knob>=0)`) without
    the public entry point ever exposing an ablation switch. `levers=None` is the
    production path.
    """
    device = input_tensor.device()
    out_memory_config = memory_config if memory_config is not None else input_tensor.memory_config()
    out_dtype = dtype if dtype is not None else input_tensor.dtype
    tile_height = int(list(tile.tile_shape)[0]) if tile is not None else DEFAULT_TILE_HEIGHT

    pad = pad_plan(
        list(input_tensor.shape),
        tile_height,
        output_padded_shape,
        pad_value,
        input_tensor.dtype,
        input_tensor.element_size(),
    )

    # The op's contract (op_design.md §8.3, risk 7): the LOGICAL shape stays the
    # input's; only the PADDED shape becomes the pad target. Two ways to say
    # that, and the first covers every cell whose target is the natural tile
    # rounding of the logical shape — a TILE tensor allocated at the logical
    # shape ALREADY has that padded shape, so no view is needed and the sharded
    # padded topologies never go near a reshape. The second is for a target that
    # exceeds the tile rounding (`50 -> 128`, the whole-pad-tile case): allocate
    # the buffer the target needs, then take a zero-cost view that keeps the
    # logical shape (`ttnn.reshape(t, logical, padded)`, same buffer address).
    logical_shape = pad["logical_shape"] if pad is not None else list(input_tensor.shape)
    padded_shape = pad["padded_shape"] if pad is not None else None
    needs_view = padded_shape is not None and auto_padded_shape(logical_shape, tile_height) != padded_shape

    # R8 (T1): the tile geometry has to reach the OUTPUT BUFFER, not only the
    # CBs — `allocate_tensor_on_device(shape, ...)` hardcodes the default 32x32
    # tile, so a tiny-tile call would allocate 32-row pages under a kernel
    # writing `tile_height`-row ones. `output_tensor_spec` is the one place that
    # is stated, for every tile height including 32.
    output_tensor = ttnn.allocate_tensor_on_device(
        output_tensor_spec(
            ttnn.Shape(padded_shape if needs_view else logical_shape),
            out_dtype,
            out_memory_config,
            tile_height,
        ),
        device,
    )

    program_descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        use_multicore=use_multicore,
        use_double_buffer=use_double_buffer,
        tile_height=tile_height,
        pad=pad,
        levers=levers,
    )
    result = ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
    if needs_view:
        result = ttnn.reshape(result, ttnn.Shape(logical_shape), ttnn.Shape(padded_shape))
    return result
