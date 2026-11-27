# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_pad: constant high-end padding of the trailing dims, one device program per call.

out[j] = x[j] where j_i < d_i for every i, else c (value converted to the dtype as torch fills).

Two regimes, chosen from the shapes alone (op_design.md, Regime selection):
- extension_band: TILE, only dim -2 grows and stays inside the last tile row of every slice.
  The output is a metadata-only view of the input's buffer; the kernel writes c into the new
  rows of the last tile row in place.
- new_storage_tiles: every other TILE call. Each input page is read once, edge tiles are masked
  in L1, and every output page is written once.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_pad_program_descriptor import (
    G6PadPlan,
    build_plan,
    fill_bits,
    get_program_descriptor,
    high_padding,
)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------
#
# Every tagger reads the scenario dict s = inputs[0] (keys shape, padding, value, dtype,
# layout, memory_config). The harness tags before validate() runs, including on the
# argument-refusal scenarios, so the taggers must be total over malformed padding.

TILE = 32


def tag_alignment(inputs, axes):
    shape = inputs[0]["shape"]
    if shape[-1] % TILE != 0:
        return "w_non_aligned"
    if shape[-2] % TILE != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    return len(inputs[0]["shape"])


def _tagged_high(pair):
    try:
        high = pair[1]
    except (TypeError, IndexError, KeyError):
        return 0
    return high if isinstance(high, int) and not isinstance(high, bool) else 0


def tag_padded_dims(inputs, axes):
    s = inputs[0]
    rank = len(s["shape"])
    pairs = list(s["padding"]) if isinstance(s["padding"], (list, tuple)) else []
    aligned = pairs[-min(len(pairs), rank) :] if pairs else []
    highs = [0] * (rank - len(aligned)) + [_tagged_high(pair) for pair in aligned]
    h_padded = rank >= 2 and highs[-2] > 0
    w_padded = rank >= 1 and highs[-1] > 0
    if h_padded and w_padded:
        return "hw"
    if h_padded:
        return "h"
    if w_padded:
        return "w"
    return "leading"


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


def tag_memory_config(inputs, axes):
    return inputs[0]["memory_config"]


INPUT_TAGGERS = {
    "dtype": tag_dtype,
    "layout": tag_layout,
    "memory_config": tag_memory_config,
    "alignment": tag_alignment,
    "rank": tag_rank,
    "padded_dims": tag_padded_dims,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "memory_config": ["dram"],
    "alignment": ["w_non_aligned", "h_non_aligned"],
    "rank": [3, 4],
    "padded_dims": ["h", "hw"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------

_INT_RANGES = {
    ttnn.int32: (-(2**31), 2**31 - 1),
    ttnn.uint32: (0, 2**32 - 1),
}


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _check_padding(padding, rank):
    if not isinstance(padding, (list, tuple)) or not 1 <= len(padding) <= rank:
        raise ValueError(f"g6_pad: padding must be a list/tuple of 1..{rank} (low, high) pairs, got {padding!r}")
    for pair in padding:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or not all(_is_int(v) for v in pair):
            raise ValueError(f"g6_pad: every padding pair must be 2 ints (low, high), got {pair!r}")
        low, high = pair
        if low != 0:
            raise ValueError(f"g6_pad: only low = 0 is supported, got pair {pair!r}")
        if high < 0:
            raise ValueError(f"g6_pad: high must be >= 0, got pair {pair!r}")


def _check_value(value, dtype):
    if not (_is_int(value) or isinstance(value, float)):
        raise ValueError(f"g6_pad: value must be an int or a float, got {value!r}")
    if dtype in _INT_RANGES:
        lo, hi = _INT_RANGES[dtype]
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            raise ValueError(f"g6_pad: value {value!r} has no {dtype} representation")
        if not lo <= int(value) <= hi:
            raise ValueError(f"g6_pad: value {value!r} outside the range of {dtype}")


def _memory_config_tag(input_tensor):
    return "l1" if input_tensor.memory_config().buffer_type == ttnn.BufferType.L1 else "dram"


def validate(input_tensor, padding, value=0.0):
    shape = list(input_tensor.shape)
    _check_padding(padding, len(shape))
    _check_value(value, input_tensor.dtype)
    # A SupportRefusal is a NotImplementedError; the contract refuses sharded inputs with ValueError,
    # so this check precedes the support gate.
    if input_tensor.memory_config().is_sharded():
        raise ValueError("g6_pad: sharded input tensors are not supported (interleaved DRAM or L1 only)")

    scenario = {
        "shape": shape,
        "padding": [list(pair) for pair in padding],
        "value": value,
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "memory_config": _memory_config_tag(input_tensor),
    }
    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_pad: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_pad: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def g6_pad_plan(input_tensor: ttnn.Tensor, padding, value=0.0) -> G6PadPlan:
    """The plan a g6_pad call with these arguments selects (BB-08). Launches nothing."""
    validate(input_tensor, padding, value)
    return build_plan(
        input_tensor, high_padding(len(input_tensor.shape), padding), fill_bits(value, input_tensor.dtype)
    )


def g6_pad(input_tensor: ttnn.Tensor, padding, value=0.0) -> ttnn.Tensor:
    validate(input_tensor, padding, value)

    highs = high_padding(len(input_tensor.shape), padding)
    c_bits = fill_bits(value, input_tensor.dtype)
    plan = build_plan(input_tensor, highs, c_bits)
    out_shape = ttnn.Shape(plan.out_shape)

    if plan.regime == "extension_band":
        # Metadata-only view of the input's buffer: same address and padded shape, no program.
        output_tensor = ttnn.reshape(input_tensor, out_shape, input_tensor.padded_shape)
    else:
        output_tensor = ttnn.allocate_tensor_on_device(
            out_shape,
            input_tensor.dtype,
            input_tensor.layout,
            input_tensor.device(),
            input_tensor.memory_config(),
        )

    program_descriptor = get_program_descriptor(plan, input_tensor, output_tensor)
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
