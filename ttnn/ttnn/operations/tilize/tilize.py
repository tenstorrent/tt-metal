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
    create_program_descriptor,
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
# Refinement 2 (A3 + A3b + A3d + A5c) — sharded placement. Three axes flip, and
# the mechanism behind them is `plan_placement()` in the program descriptor:
#   shard_api   += legacy_2d, nd  both APIs project onto one ShardSpec view.
#   out_scheme  += HEIGHT/WIDTH/BLOCK/nd — the scheme picks the shard's shape,
#                            and the shard IS the per-core block either way.
#   orientation += ROW_MAJOR, COL_MAJOR — a non-issue on the zero-copy path
#                            (each core tilizes the block in its own L1, so it
#                            never needs to know which shard that is).

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "output_dtype": [ttnn.bfloat16],
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
    "rank": [2, 3, 4, 5],
    "pad_mode": ["none"],
    "pad_value": ["none"],
    "alignment": ["tile_aligned"],
    "orientation": ["none", ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
    "tile_height": [DEFAULT_TILE_HEIGHT],
    "in_layout": [ttnn.ROW_MAJOR_LAYOUT],
    "in_tile_height": ["none"],
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
    _check_placement(input_tensor, out_memory_config, tile_height)

    return scenario, axes


def _check_placement(input_tensor, out_memory_config, tile_height):
    """Refuse a shard geometry neither placement mechanism can address.

    Every sharded cell in the golden TARGET is either same-spec (zero-copy on
    both sides), a crossover (zero-copy on the sharded side), or cross-spec with
    a full-row-width input shard (streamed) — so nothing in SUPPORTED lands here.
    What does: a ROW_MAJOR input whose shard is NARROWER than a row and is not
    L1-resident (e.g. a DRAM width-shard), where the streamed reader's stick
    indexing would silently read the wrong bytes.
    """
    elem_size = input_tensor.element_size()
    shape = list(input_tensor.shape)
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
        tile=tile,
    )


def _dispatch(
    input_tensor,
    memory_config=None,
    *,
    dtype=None,
    use_multicore=True,
    use_double_buffer=True,
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

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        out_dtype,
        ttnn.TILE_LAYOUT,
        device,
        out_memory_config,
    )

    program_descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        use_multicore=use_multicore,
        use_double_buffer=use_double_buffer,
        tile_height=tile_height,
        levers=levers,
    )
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
