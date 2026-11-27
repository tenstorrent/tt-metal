# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_expand: broadcast x (d_0, d_1, 1, W) along dim 2 into a new DRAM-interleaved TILE tensor.

out[n, c, h, w] = x[n, c, 0, w], bit-identical. One device program per call: the
`column_group_broadcast` regime (op_design.md) — each column group's sender reads its x pages once,
expands row 0 of each tile into all 32 rows in L1, and multicasts the expanded tiles along its grid
row; every core of the rectangle writes its own out tile rows of those columns.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_expand_program_descriptor import (
    DEFAULT_KNOBS,
    TILE_FACE_SHAPE,
    TILE_SHAPE,
    PlanKnobs,
    bind_tensor_addresses,
    get_program_artifacts,
    plan_report,
    select_plan,
)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def _output_shape(shape, sizes):
    return [int(d) if int(s) == -1 else int(s) for d, s in zip(shape, sizes)]


def tag_alignment(inputs, axes):
    """Tile alignment of the OUTPUT shape (sizes with every -1 replaced by x's size)."""
    scenario = inputs[0]
    out_shape = _output_shape(scenario["shape"], scenario["sizes"])
    if out_shape[-1] % 32:
        return "w_non_aligned"
    if out_shape[-2] % 32:
        return "h_non_aligned"
    return "tile_aligned"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned", "w_non_aligned"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _check_call_contract(input_tensor, sizes):
    """Input requirements of the contract; violations are caller errors (ValueError)."""
    shape = list(input_tensor.shape)
    if len(shape) != 4:
        raise ValueError(f"g6_expand: input must have rank 4, got rank {len(shape)}")
    if not isinstance(sizes, (list, tuple)):
        raise ValueError(f"g6_expand: sizes must be a list or tuple, got {type(sizes).__name__}")
    sizes = [int(s) for s in sizes]
    if len(sizes) != len(shape):
        raise ValueError(f"g6_expand: sizes {sizes} must have the rank of the input ({len(shape)})")
    for s in sizes:
        if s < -1 or s == 0:
            raise ValueError(f"g6_expand: sizes {sizes} holds {s}; every size must be -1 or >= 1")
    for k, (d, s) in enumerate(zip(shape, sizes)):
        if d != 1 and s not in (-1, d):
            raise ValueError(f"g6_expand: sizes[{k}]={s} differs from the input's non-unit size {d}")
        if k != 2 and d == 1 and s not in (-1, 1):
            raise ValueError(f"g6_expand: sizes[{k}]={s} expands dim {k}; only dim 2 may be expanded")
    if shape[2] != 1:
        raise ValueError(f"g6_expand: the input must hold one row per image (dim 2 == 1), got {shape[2]}")
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("g6_expand: input must be on a device")
    memory_config = input_tensor.memory_config()
    if (
        memory_config.buffer_type != ttnn.BufferType.DRAM
        or memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    ):
        raise ValueError(f"g6_expand: input must be DRAM interleaved, got {memory_config}")
    if input_tensor.layout == ttnn.TILE_LAYOUT:
        tile = input_tensor.tile
        if list(tile.tile_shape) != list(TILE_SHAPE) or list(tile.face_shape) != list(TILE_FACE_SHAPE):
            raise ValueError(
                f"g6_expand: tile {list(tile.tile_shape)} / face {list(tile.face_shape)} is not "
                f"{list(TILE_SHAPE)} / {list(TILE_FACE_SHAPE)}"
            )
    return sizes


def validate(input_tensor, sizes):
    sizes = _check_call_contract(input_tensor, sizes)

    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
    }
    scenario = {"shape": list(input_tensor.shape), "sizes": list(sizes)}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_expand: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_expand: unsupported combination (refinement candidate): {exc}")

    return sizes


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def g6_expand(input_tensor: ttnn.Tensor, sizes) -> ttnn.Tensor:
    """Return a new tensor of shape `sizes` (-1 keeps x's size) with x's row repeated along dim 2."""
    return _g6_expand(input_tensor, sizes, DEFAULT_KNOBS)


def _g6_expand(input_tensor: ttnn.Tensor, sizes, knobs: PlanKnobs) -> ttnn.Tensor:
    sizes = validate(input_tensor, sizes)
    out_shape = _output_shape(input_tensor.shape, sizes)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        input_tensor.memory_config(),
    )
    _, descriptor, address_slots = get_program_artifacts(input_tensor, output_tensor, out_shape, knobs)
    bind_tensor_addresses(descriptor, address_slots, input_tensor, output_tensor)
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)


def g6_expand_plan(input_tensor: ttnn.Tensor, sizes, knobs: PlanKnobs = DEFAULT_KNOBS) -> dict:
    """The plan `g6_expand(input_tensor, sizes)` runs: regime, block knobs, core count and roles."""
    sizes = validate(input_tensor, sizes)
    return plan_report(select_plan(input_tensor, _output_shape(input_tensor.shape, sizes), knobs))
