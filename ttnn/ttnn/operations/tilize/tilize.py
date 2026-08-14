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
# the three pad modes. Sharded placement, tiny tiles, retile and the integer
# dtype family are not built yet and are refused (xfail) by the axis gate.

SUPPORTED = {
    "dtype": [ttnn.bfloat16, ttnn.float32],
    "output_dtype": [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b],
    "use_multicore": [False, True],
    "double_buffer": [False, True],
    "shard_api": ["none"],
    "out_scheme": ["interleaved"],
    "buffer": ["dram_to_dram", "dram_to_l1", "l1_to_l1", "l1_to_dram"],
    "rank": [2, 3, 4, 5],
    "orientation": ["none"],
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
    # bfloat8_b is a block-float format: its shared exponent is defined over the
    # tile's 16x16 face structure, and the pad fill is materialized in the INPUT
    # element format before the pack. Padding into bf8b is therefore not wired
    # up yet (a future refinement could enable it).
    {"output_dtype": ttnn.bfloat8_b, "pad_mode": "auto"},
    {"output_dtype": ttnn.bfloat8_b, "pad_mode": "explicit"},
]


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
        return list(self.target) != list(self.in_padded)


def _round_up(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


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
        if len(in_shape) < 2:
            raise ValueError(f"tilize: rank {len(in_shape)} input requires an explicit output_padded_shape")
        target = list(in_shape)
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
        if len(target) != len(in_shape):
            raise ValueError(f"tilize: output_padded_shape {target} must have the same rank as the input {in_shape}")
        for i, (got, want) in enumerate(zip(target, in_shape)):
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


def _scenario_from_call(input_tensor, plan):
    """The scenario dict the golden suite would have written for this call."""
    in_spec, in_api = _placement_spec(input_tensor.memory_config())
    out_spec, out_api = _placement_spec(plan.out_memory_config)
    shard_api = "none" if (in_api == "none" and out_api == "none") else (in_api if in_api != "none" else out_api)
    return {
        "input_shape": plan.in_shape,
        "use_multicore": plan.use_multicore,
        "use_double_buffer": plan.use_double_buffer,
        "shard_api": shard_api,
        "in": in_spec,
        "out": out_spec,
        "pad_mode": plan.pad_mode,
        "pad_value": plan.pad_value,
        "tile_height": plan.tile_h,
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
    """Runtime support gate. Returns the canonical plan for the entry point."""
    plan = _canonicalize(
        input_tensor,
        memory_config,
        dtype,
        use_multicore,
        use_double_buffer,
        output_padded_shape,
        pad_value,
        tile,
    )

    scenario = _scenario_from_call(input_tensor, plan)
    axes = {"dtype": input_tensor.dtype, "output_dtype": plan.out_dtype}
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

    return plan


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

    if list(plan.target) != list(plan.in_shape):
        output_tensor = ttnn.reshape(output_tensor, ttnn.Shape(plan.in_shape), ttnn.Shape(plan.target))
    return output_tensor


__all__ = ["tilize", "validate", "INPUT_TAGGERS", "SUPPORTED", "EXCLUSIONS", "PROPERTIES"]
