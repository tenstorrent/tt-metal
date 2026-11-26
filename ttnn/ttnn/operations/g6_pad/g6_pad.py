# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_pad: constant padding at the high end of the trailing dims, bit-identical, one device program.

Two regimes, selected from the shapes alone (op_design.md, Blocking Model):

- E (tile_padding_extension): only dim -2 grows inside the tiles x already occupies. One
  write-only kernel writes c into the new rows of the last tile row of every slice of x, in
  place; the output is a metadata-only view of x's buffer at the output shape.
- N (new_storage): every output tile written once into a new tensor; input-backed tiles are
  read once and assembled in L1 (boundary tiles masked with c), fill-only tiles are written
  from a per-core fill page and never read.
"""

from __future__ import annotations

import math

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_pad_program_descriptor import (
    REGIME_EXTENSION,
    REGIME_NEW_STORAGE,
    create_program_descriptor,
    set_call_args,
)

TILE = 32


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — total: the harness tags malformed refusal scenarios too.
# ---------------------------------------------------------------------------


def _scenario(inputs):
    return inputs[0]


def _trailing_highs(s):
    """High padding per dim, the pairs aligned to the trailing dims; never raises."""
    shape = list(s["shape"])
    try:
        hs = [pair[1] for pair in s["padding"]]
    except Exception:  # noqa: BLE001 — taggers must be total
        hs = []
    return [0] * max(0, len(shape) - len(hs)) + hs


def tag_alignment(inputs, axes):
    shape = _scenario(inputs)["shape"]
    if shape[-1] % TILE != 0:
        return "w_non_aligned"
    if len(shape) >= 2 and shape[-2] % TILE != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    return len(_scenario(inputs)["shape"])


def tag_padded_dims(inputs, axes):
    highs = _trailing_highs(_scenario(inputs))
    try:
        hh = highs[-2] if len(highs) >= 2 else 0
        hw = highs[-1] if highs else 0
        grow_h, grow_w = hh > 0, hw > 0
    except Exception:  # noqa: BLE001 — taggers must be total
        return "leading"
    if grow_h and grow_w:
        return "hw"
    if grow_h:
        return "h"
    if grow_w:
        return "w"
    return "leading"


def tag_dtype(inputs, axes):
    return _scenario(inputs)["dtype"]


def tag_layout(inputs, axes):
    return _scenario(inputs)["layout"]


def tag_memory_config(inputs, axes):
    return _scenario(inputs)["memory_config"]


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "rank": tag_rank,
    "padded_dims": tag_padded_dims,
    "dtype": tag_dtype,
    "layout": tag_layout,
    "memory_config": tag_memory_config,
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


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _check_arguments(input_tensor, padding, value):
    """Argument form (ValueError), before any tagging: a refusal here is not a support statement."""
    if not isinstance(input_tensor, ttnn.Tensor):
        raise ValueError(f"g6_pad: input_tensor must be a ttnn.Tensor, got {type(input_tensor).__name__}")
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("g6_pad: input_tensor must be on device")
    if input_tensor.memory_config().is_sharded():
        raise ValueError("g6_pad: sharded input_tensor is not supported (interleaved only)")

    rank = len(input_tensor.shape)
    if not isinstance(padding, (list, tuple)) or not 1 <= len(padding) <= rank:
        raise ValueError(f"g6_pad: padding must be a list/tuple of 1..{rank} (low, high) pairs, got {padding!r}")
    for pair in padding:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or not all(_is_int(p) for p in pair):
            raise ValueError(f"g6_pad: every padding pair must be 2 ints (low, high), got {pair!r}")
        if pair[0] != 0:
            raise ValueError(f"g6_pad: only low = 0 padding is supported, got {pair!r}")
        if pair[1] < 0:
            raise ValueError(f"g6_pad: high padding must be >= 0, got {pair!r}")

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"g6_pad: value must be an int or a float, got {type(value).__name__}")
    if input_tensor.dtype in _INT_RANGES:
        if not math.isfinite(value):
            raise ValueError(f"g6_pad: value {value!r} is not finite for {input_tensor.dtype}")
        lo, hi = _INT_RANGES[input_tensor.dtype]
        if not lo <= math.trunc(value) <= hi:
            raise ValueError(f"g6_pad: value {value!r} outside the range of {input_tensor.dtype}")


def _memory_config_tag(tensor):
    return "l1" if tensor.memory_config().buffer_type == ttnn.BufferType.L1 else "dram"


def validate(input_tensor, padding, value=0.0):
    _check_arguments(input_tensor, padding, value)

    scenario = {
        "shape": list(input_tensor.shape),
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
# Host plan
# ---------------------------------------------------------------------------


def _highs(shape, padding):
    return [0] * (len(shape) - len(padding)) + [int(pair[1]) for pair in padding]


def _is_extension(shape, highs, layout):
    if layout != ttnn.TILE_LAYOUT:
        return False
    rank = len(shape)
    only_h = highs[-2] > 0 and all(h == 0 for i, h in enumerate(highs) if i != rank - 2)
    return only_h and -(-shape[-2] // TILE) == -(-(shape[-2] + highs[-2]) // TILE)


def _fill_bits(dtype, value):
    """c as the bit pattern torch stores when it fills a tensor of x's dtype with `value`."""
    import torch

    if dtype == ttnn.bfloat16:
        return int(torch.full((1,), value, dtype=torch.bfloat16).view(torch.int16).item()) & 0xFFFF
    if dtype == ttnn.float32:
        return int(torch.full((1,), value, dtype=torch.float32).view(torch.int32).item()) & 0xFFFFFFFF
    return math.trunc(value) & 0xFFFFFFFF


# (regime, shape, highs, dtype, layout, buffer_type, grid) -> CachedProgram (BB-07).
_PROGRAM_CACHE = {}


def g6_pad(input_tensor: ttnn.Tensor, padding, value=0.0) -> ttnn.Tensor:
    validate(input_tensor, padding, value)

    shape = list(input_tensor.shape)
    highs = _highs(shape, padding)
    out_shape = [d + h for d, h in zip(shape, highs)]
    regime = REGIME_EXTENSION if _is_extension(shape, highs, input_tensor.layout) else REGIME_NEW_STORAGE

    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    memory_config = input_tensor.memory_config()

    output_tensor = None
    if regime == REGIME_NEW_STORAGE:
        output_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape(out_shape), input_tensor.dtype, input_tensor.layout, device, memory_config
        )

    key = (
        regime,
        tuple(shape),
        tuple(highs),
        input_tensor.dtype,
        input_tensor.layout,
        memory_config.buffer_type,
        (grid.x, grid.y),
    )
    cached = _PROGRAM_CACHE.get(key)
    if cached is None:
        cached = create_program_descriptor(regime, input_tensor, output_tensor, highs, grid)
        _PROGRAM_CACHE[key] = cached

    fill_bits = _fill_bits(input_tensor.dtype, value)
    if regime == REGIME_EXTENSION:
        program = set_call_args(cached, input_tensor.buffer_address(), None, fill_bits)
        g6_pad.last_plan = cached.plan
        ttnn.generic_op([input_tensor], program)
        # Metadata-only view of x's buffer at the output shape: launches no device program.
        return ttnn.reshape(input_tensor, ttnn.Shape(out_shape), input_tensor.padded_shape)

    program = set_call_args(cached, input_tensor.buffer_address(), output_tensor.buffer_address(), fill_bits)
    g6_pad.last_plan = cached.plan
    return ttnn.generic_op([input_tensor, output_tensor], program)


g6_pad.last_plan = None
