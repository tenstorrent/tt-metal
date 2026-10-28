# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""clamp_square_practice: y = clamp(x, min_val, max_val) ** 2.

One ttnn.generic_op dispatch. Each core streams a contiguous page range through a
fused Copy -> Clamp -> Square -> Pack eltwise_chain (regime `interleaved_stream`,
see op_design.md).
"""

from __future__ import annotations

import numbers

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .clamp_square_practice_program_descriptor import create_program_descriptor


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_rank(inputs, axes):
    """Rank >= 2 is required (the H / W tile axes)."""
    return "rank_ge_2" if len(inputs[0]) >= 2 else "rank_lt_2"


def tag_alignment(inputs, axes):
    """Last two logical dims tile-aligned (multiples of 32)."""
    shape = inputs[0]
    if len(shape) < 2:
        return "non_tile_aligned"
    if shape[-1] % 32 == 0 and shape[-2] % 32 == 0:
        return "tile_aligned"
    return "non_tile_aligned"


INPUT_TAGGERS = {
    "rank": tag_rank,
    "alignment": tag_alignment,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "memory_layout": [ttnn.TensorMemoryLayout.INTERLEAVED],
    "buffer_type": [ttnn.BufferType.DRAM],
    "rank": ["rank_ge_2"],
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
# 4. validate()
# ---------------------------------------------------------------------------


def _check_bound(name, value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"clamp_square_practice: {name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"clamp_square_practice: {name} must be finite, got {value}")
    return value


def validate(input_tensor, min_val, max_val):
    mem = input_tensor.memory_config()
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "memory_layout": mem.memory_layout,
        "buffer_type": mem.buffer_type,
    }
    shape = tuple(input_tensor.shape)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"clamp_square_practice: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"clamp_square_practice: unsupported combination (refinement candidate): {exc}")

    lo = _check_bound("min_val", min_val)
    hi = _check_bound("max_val", max_val)
    if lo > hi:
        raise ValueError(f"clamp_square_practice: min_val ({lo}) must be <= max_val ({hi})")
    return lo, hi


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def clamp_square_practice(input_tensor: ttnn.Tensor, min_val: float, max_val: float) -> ttnn.Tensor:
    """Compute clamp(input_tensor, min_val, max_val) ** 2 in one device dispatch."""
    lo, hi = validate(input_tensor, min_val, max_val)

    device = input_tensor.device()
    # Positional args only.
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    program_descriptor = create_program_descriptor(input_tensor, output_tensor, lo, hi)
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
