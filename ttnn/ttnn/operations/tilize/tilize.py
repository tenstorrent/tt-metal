# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""``ttnn.operations.tilize.tilize`` — ROW_MAJOR -> TILE layout conversion.

Registry-model op (see ``eval/op_template.py`` and ``eval/REGISTRY_MODEL.md``):
four declarations (INPUT_TAGGERS / SUPPORTED / EXCLUSIONS / validate) plus the
public entry point, which calls ``validate()`` on its first line.

The blocking model, dataflow topology and work split are pinned by
``ttnn/ttnn/operations/tilize/op_design.md``; every derived quantity has a
single source in ``derive_blocking()`` (in the program-descriptor module).

INVALID is deliberately NOT declared here — it lives in
``eval/golden_tests/tilize/feature_spec.py``.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .tilize_program_descriptor import (
    DEFAULT_TILE_WIDTH,
    create_program_descriptor,
)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------
#
# Each tagger maps (inputs_tuple, axes_dict) -> categorical value. For this op
# `inputs[0]` is a *scenario dict* (the golden suite's per-case conversion
# description, see feature_spec.py). validate() synthesizes the same dict from
# the live tensors + kwargs and runs these very functions, so the runtime gate
# and the test harness project the axes identically — one source of truth.


def _buffer_name(spec):
    return "dram" if spec["buffer"] == ttnn.BufferType.DRAM else "l1"


def tag_use_multicore(inputs, axes):
    return bool(inputs[0]["use_multicore"])


def tag_shard_api(inputs, axes):
    return inputs[0]["shard_api"]


def tag_out_scheme(inputs, axes):
    out = inputs[0]["out"]
    if out["kind"] == "interleaved":
        return "interleaved"
    return out["scheme"] if out["scheme"] is not None else "nd"


def tag_buffer(inputs, axes):
    scenario = inputs[0]
    return f"{_buffer_name(scenario['in'])}_to_{_buffer_name(scenario['out'])}"


def tag_rank(inputs, axes):
    return int(len(inputs[0]["input_shape"]))


def tag_double_buffer(inputs, axes):
    return bool(inputs[0].get("use_double_buffer", True))


def tag_pad_mode(inputs, axes):
    return inputs[0].get("pad_mode", "none")


def tag_pad_value(inputs, axes):
    if inputs[0].get("pad_mode", "none") == "none":
        return "none"
    value = inputs[0].get("pad_value")
    if value is None:
        return "none"
    if value == 0:
        return "zero"
    return "positive" if value > 0 else "negative"


def tag_alignment(inputs, axes):
    """Last two dims vs the tile geometry the call asks for.

    H is measured against the OUTPUT tile height (a tiny tile redefines what
    "aligned" means on that axis), W always against 32.
    """
    scenario = inputs[0]
    shape = list(scenario["input_shape"])
    if len(shape) < 2:
        return "hw_non_aligned"
    tile_h = scenario.get("tile_height", 32)
    h_aligned = shape[-2] % tile_h == 0
    w_aligned = shape[-1] % DEFAULT_TILE_WIDTH == 0
    if h_aligned and w_aligned:
        return "tile_aligned"
    if h_aligned:
        return "w_non_aligned"
    if w_aligned:
        return "h_non_aligned"
    return "hw_non_aligned"


def tag_orientation(inputs, axes):
    scenario = inputs[0]
    in_spec, out_spec = scenario["in"], scenario["out"]
    if in_spec["kind"] == "interleaved" and out_spec["kind"] == "interleaved":
        return "none"
    if in_spec["kind"] == "sharded":
        return in_spec["orientation"]
    return out_spec["orientation"]


def tag_tile_height(inputs, axes):
    return inputs[0].get("tile_height", 32)


def tag_in_layout(inputs, axes):
    return inputs[0].get("in_layout", ttnn.ROW_MAJOR_LAYOUT)


def tag_in_tile_height(inputs, axes):
    return inputs[0].get("in_tile_height", "none")


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
# 2. SUPPORTED
# ---------------------------------------------------------------------------
#
# Phase 0: the whole INTERLEAVED surface (DRAM/L1, both directions), single- and
# multi-core, both CB depths, bf16/fp32 in, bf16/fp32/bf8b out, rank 2..5, and
# the three pad modes.
# Refinement 1: the SHARDED placement surface — both sharding APIs, all three
# legacy schemes plus nd, both orientations.
# Refinement 4: the INTEGER dtype family (uint32 / uint16 / int32 / uint8) plus
# rank 0. tilize is a byte permutation, so an integer datum is just a width —
# the LLK picks its own 8-bit tilize path off the CB format (there is no host
# knob), and every byte quantity in the descriptor already derives from
# `element_size()`. Tiny tiles and retile are not built yet and are refused
# (xfail) by the axis gate.

SUPPORTED = {
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
    "rank": [0, 2, 3, 4, 5],
    "orientation": ["none", ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
    "tile_height": [32],
    "in_layout": [ttnn.ROW_MAJOR_LAYOUT],
    "in_tile_height": ["none"],
    "pad_mode": ["none", "auto", "explicit"],
    "pad_value": ["none", "zero", "positive", "negative"],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned", "hw_non_aligned"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    # --- Refinement 4 -------------------------------------------------------
    # Both Phase-0 padding EXCLUSIONS are GONE.
    #
    # (1) `bfloat8_b` output x pad_mode {auto, explicit}: nothing was actually
    #     wrong. The fill is materialized into the INPUT CB (a plain float format)
    #     and the packer builds the block-float shared exponent over whatever it
    #     is handed, pad included — the 16x16 face structure never sees a
    #     half-written exponent because a pad position is an ordinary datum by
    #     the time it reaches the pack stage.
    #
    # (2) `bfloat16 -> float32` x pad_value {positive, negative}: fixed rather
    #     than refused. The reader still fills in the INPUT element format (that
    #     is a hard contract — packing the fill in output_dtype is garbage the
    #     moment a cast is requested), so a fill that is inexact in bf16 used to
    #     land bf16-rounded in an fp32 output. The writer now re-stamps the pad
    #     region of each output tile with a SECOND fill word packed in the OUTPUT
    #     format, after the cast (`pad_word_out` / R4_OUT_FILL). See
    #     `tilize_writer.cpp` and `_pack_pad_word`.
]

# --- Refinement 1 (sharded placement) ---------------------------------------
# A sharded tensor is inherently MULTI-CORE: the shards pin both the core set and
# the per-core work, so there is no single-core realization of a sharded call.
for _api in ("legacy_2d", "nd"):
    EXCLUSIONS.append({"use_multicore": False, "shard_api": _api})
del _api
# Refinement 2 lifted `pad_mode in {auto, explicit}` x sharded: eligibility for the
# zero-copy CB is now decided PER SIDE, so a padded call streams its (filled) input
# CB while still packing straight into a resident destination shard.


# ---------------------------------------------------------------------------
# 3b. PROPERTIES
# ---------------------------------------------------------------------------

PROPERTIES = {
    "multi_core": {"value": True, "source": "verified"},
    "bounded_cb": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# Canonicalization (runs before the support check so validate() sees one form)
# ---------------------------------------------------------------------------


class _Plan:
    """Canonical, fully-derived description of one tilize call."""

    __slots__ = (
        "tile_h",
        "tile_w",
        "in_shape",
        "in_padded",
        "read_shape",
        "read_padded",
        "target",
        "pad_mode",
        "pad_value",
        "out_dtype",
        "out_memory_config",
        "use_multicore",
        "use_double_buffer",
    )

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def has_pad_region(self):
        """The pad region is non-empty — the R_PAD reader regime.

        Keyed on the pad region actually existing, NOT on the pad_mode string:
        `pad_value=` on an already-aligned input must take the (byte-identical)
        aligned reader.
        """
        return list(self.target) != list(self.read_padded)


def _round_up(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


def _expand_rank(shape, rank):
    """Left-expand `shape` with 1s up to `rank` (a no-op once it is long enough).

    A rank < 2 input (the rank-0 scalar) has NO tile dims of its own — the pad
    target synthesizes them (`[]` -> `[32, 32]`). Every geometry consumer (the
    reader's `h_in` / `w_in_bytes` / image count, the source page geometry, the
    shard plan) needs a 2-D-or-deeper view, so the promotion lives HERE, once,
    and the kernels never see a degenerate rank.
    """
    shape = list(shape)
    return [1] * (rank - len(shape)) + shape if len(shape) < rank else shape


def _canonicalize(
    input_tensor,
    memory_config,
    dtype,
    use_multicore,
    use_double_buffer,
    output_padded_shape,
    pad_value,
    tile,
):
    tile = tile if tile is not None else ttnn.Tile([32, DEFAULT_TILE_WIDTH])
    tile_h, tile_w = (int(d) for d in tile.tile_shape)

    in_shape = list(input_tensor.shape)
    in_padded = list(input_tensor.padded_shape)

    if output_padded_shape is None and pad_value is None:
        pad_mode = "none"
        target = list(in_padded)
    elif output_padded_shape is None:
        pad_mode = "auto"
        # A rank < 2 input has no tile dims to round: they are SYNTHESIZED (a
        # scalar pads to one tile), which is the same promotion _expand_rank does.
        target = _expand_rank(in_shape, 2)
        target[-2] = _round_up(target[-2], tile_h)
        target[-1] = _round_up(target[-1], tile_w)
    else:
        pad_mode = "explicit"
        target = [int(d) for d in output_padded_shape]
        if pad_value is None:
            raise ValueError("tilize: output_padded_shape requires pad_value — a pad target with no fill is undefined")

    if len(target) < 2:
        raise ValueError(f"tilize: output padded shape {target} must have rank >= 2")
    if target[-2] % tile_h or target[-1] % tile_w:
        raise ValueError(
            f"tilize: output padded shape {target} last two dims must be multiples of the tile ({tile_h}, {tile_w})"
        )

    if pad_mode == "explicit":
        # A rank < 2 input is the one legal rank mismatch: it carries no tile dims,
        # so the target supplies them (`[] -> [32, 32]`). The comparison below then
        # runs against the promoted view, so `[]` trivially fits any target.
        if len(target) != len(in_shape) and len(in_shape) >= 2:
            raise ValueError(f"tilize: output_padded_shape {target} must have the same rank as the input {in_shape}")
        for i, (got, want) in enumerate(zip(target, _expand_rank(in_shape, len(target)))):
            if got < want:
                raise ValueError(
                    f"tilize: output_padded_shape {target} must be >= the input shape {in_shape} in every dim "
                    f"(dim {i}: {got} < {want})"
                )

    if pad_mode == "none":
        # Padding is NEVER implicit. The message must mention "pad".
        if len(in_shape) >= 2 and (in_shape[-2] % tile_h or in_shape[-1] % tile_w):
            raise ValueError(
                f"tilize: input shape {in_shape} is not tile-aligned ({tile_h}, {tile_w}) and no padding was "
                f"requested — pass pad_value= (and optionally output_padded_shape=) to pad explicitly"
            )

    return _Plan(
        tile_h=tile_h,
        tile_w=tile_w,
        in_shape=in_shape,
        in_padded=in_padded,
        # The geometry view every kernel-facing derivation reads (rank >= 2).
        read_shape=_expand_rank(in_shape, len(target)),
        read_padded=_expand_rank(in_padded, len(target)),
        target=target,
        pad_mode=pad_mode,
        pad_value=pad_value,
        out_dtype=dtype if dtype is not None else input_tensor.dtype,
        out_memory_config=memory_config if memory_config is not None else input_tensor.memory_config(),
        use_multicore=bool(use_multicore),
        use_double_buffer=bool(use_double_buffer),
    )


def _placement_spec(memory_config):
    """Project a MemoryConfig onto the scenario-dict placement spec the taggers read."""
    buffer_type = memory_config.buffer_type
    if not memory_config.is_sharded():
        return {"kind": "interleaved", "buffer": buffer_type}, "none"

    nd_spec = getattr(memory_config, "nd_shard_spec", None)
    if nd_spec is not None:
        return (
            {"kind": "sharded", "buffer": buffer_type, "scheme": None, "orientation": nd_spec.orientation},
            "nd",
        )
    shard_spec = memory_config.shard_spec
    return (
        {
            "kind": "sharded",
            "buffer": buffer_type,
            "scheme": memory_config.memory_layout,
            "orientation": shard_spec.orientation,
        },
        "legacy_2d",
    )


def _scenario_from_call(
    input_tensor, out_memory_config, *, tile_h, pad_mode, pad_value, use_multicore, use_double_buffer
):
    """The scenario dict the golden suite would have written for this call.

    Built from the RAW call arguments, not from a canonical plan: the support
    gate has to run *before* the shape-legality checks, so that an out-of-
    rectangle cell (e.g. a rank the op does not build yet) is refused as an
    unsupported axis value rather than by a shape ValueError raised earlier.
    """
    in_spec, in_api = _placement_spec(input_tensor.memory_config())
    out_spec, out_api = _placement_spec(out_memory_config)
    shard_api = "none" if (in_api == "none" and out_api == "none") else (in_api if in_api != "none" else out_api)
    return {
        "input_shape": list(input_tensor.shape),
        "use_multicore": bool(use_multicore),
        "use_double_buffer": bool(use_double_buffer),
        "shard_api": shard_api,
        "in": in_spec,
        "out": out_spec,
        "pad_mode": pad_mode,
        "pad_value": pad_value,
        "tile_height": tile_h,
        "in_layout": input_tensor.layout,
        "in_tile_height": "none"
        if input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
        else int(input_tensor.tile.tile_shape[0]),
    }


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


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
    """Runtime support gate. Returns the canonical plan for the entry point.

    Order matters: the SUPPORTED / EXCLUSIONS gate runs on axis values projected
    straight from the raw call, BEFORE the shape-legality checks in
    `_canonicalize()`. Otherwise a cell outside the rectangle (say a rank the op
    does not build yet) would be refused by a shape ValueError instead of the
    typed support refusal the registry contract promises.
    """
    # --- axis values, cheaply (no shape legality decided here) ---
    tile_h = int((tile if tile is not None else ttnn.Tile([32, DEFAULT_TILE_WIDTH])).tile_shape[0])
    if output_padded_shape is None and pad_value is None:
        pad_mode = "none"
    elif output_padded_shape is None:
        pad_mode = "auto"
    else:
        pad_mode = "explicit"

    scenario = _scenario_from_call(
        input_tensor,
        memory_config if memory_config is not None else input_tensor.memory_config(),
        tile_h=tile_h,
        pad_mode=pad_mode,
        pad_value=pad_value,
        use_multicore=use_multicore,
        use_double_buffer=use_double_buffer,
    )
    axes = {
        "dtype": input_tensor.dtype,
        "output_dtype": dtype if dtype is not None else input_tensor.dtype,
    }
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

    # 3. Inside the rectangle — now the call's own legality (shapes, pad target).
    return _canonicalize(
        input_tensor,
        memory_config,
        dtype,
        use_multicore,
        use_double_buffer,
        output_padded_shape,
        pad_value,
        tile,
    )


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
    """Convert a ROW_MAJOR tensor to TILE layout (optionally cast / padded)."""
    plan = validate(
        input_tensor,
        memory_config,
        dtype=dtype,
        use_multicore=use_multicore,
        use_double_buffer=use_double_buffer,
        output_padded_shape=output_padded_shape,
        pad_value=pad_value,
        tile=tile,
    )

    device = input_tensor.device()

    # The device buffer is allocated at the PADDED target so the kernels write
    # whole tiles; the logical view is restored below (zero-copy reshape) so the
    # output keeps the input's logical shape, as the contract demands.
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(plan.target),
        plan.out_dtype,
        ttnn.TILE_LAYOUT,
        device,
        plan.out_memory_config,
    )

    program_descriptor = create_program_descriptor(input_tensor, output_tensor, plan)
    output_tensor = ttnn.generic_op([input_tensor, output_tensor], program_descriptor)

    # Restore the input's LOGICAL shape (only the padded shape grows). A rank < 2
    # input is the exception: a logical shape and a padded shape must share a rank,
    # and a scalar has no tile dims, so its padded view IS its shape — there is no
    # unpadded logical view to restore (the golden oracle skips that check for the
    # same reason).
    if len(plan.in_shape) >= 2 and list(plan.target) != list(plan.in_shape):
        output_tensor = ttnn.reshape(output_tensor, ttnn.Shape(plan.in_shape), ttnn.Shape(plan.target))
    return output_tensor


__all__ = ["tilize", "validate", "INPUT_TAGGERS", "SUPPORTED", "EXCLUSIONS", "PROPERTIES"]
