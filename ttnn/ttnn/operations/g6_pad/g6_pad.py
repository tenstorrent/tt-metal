# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_pad: constant padding at the high end of the trailing dims, bit-identical, one device
program per call.

A tile_padding_extension call (TILE, only dim -2 grows inside the tile rows x already occupies)
writes c into the new rows of the last tile row of each slice of x in place and returns a
metadata-only view of x's buffer at the output shape. Every other call writes a new tensor in one
pass: copy and boundary tiles are read once and assembled in L1, fill-only tiles are never read.
"""

from __future__ import annotations

import math

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from . import g6_pad_program_descriptor as descriptors

TILE = 32


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS (total: they also see malformed refusal scenarios)
# ---------------------------------------------------------------------------


def _number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _trailing_highs(s):
    """High padding of dims -2 and -1 after aligning the pairs to the trailing dims."""
    highs = []
    for pair in s["padding"]:
        high = pair[1] if isinstance(pair, (list, tuple)) and len(pair) >= 2 else 0
        highs.append(high if _number(high) else 0)
    highs = [0, 0] + highs
    return highs[-2], highs[-1]


def tag_alignment(inputs, axes):
    s = inputs[0]
    if s["shape"][-1] % TILE != 0:
        return "w_non_aligned"
    if s["shape"][-2] % TILE != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    return len(inputs[0]["shape"])


def tag_padded_dims(inputs, axes):
    high_h, high_w = _trailing_highs(inputs[0])
    if high_h > 0 and high_w > 0:
        return "hw"
    if high_h > 0:
        return "h"
    if high_w > 0:
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


def _check_arguments(input_tensor, padding, value):
    """Argument form (ValueError), before any tagging."""
    if not isinstance(input_tensor, ttnn.Tensor):
        raise ValueError(f"g6_pad: input_tensor must be a ttnn.Tensor, got {type(input_tensor).__name__}")
    if input_tensor.storage_type() != ttnn.StorageType.DEVICE:
        raise ValueError("g6_pad: input_tensor must be on device")
    if input_tensor.memory_config().is_sharded():
        raise ValueError("g6_pad: sharded input_tensor is not accepted (interleaved DRAM or L1 only)")

    rank = len(input_tensor.shape)
    if not isinstance(padding, (list, tuple)) or not 1 <= len(padding) <= rank:
        raise ValueError(f"g6_pad: padding must be a list or tuple of 1 to {rank} (low, high) pairs, got {padding!r}")
    for pair in padding:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"g6_pad: every padding entry must be a (low, high) pair, got {pair!r}")
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in pair):
            raise ValueError(f"g6_pad: padding pairs must hold ints, got {pair!r}")
        if pair[0] != 0 or pair[1] < 0:
            raise ValueError(f"g6_pad: every pair must be (0, high) with high >= 0, got {tuple(pair)}")

    if not _number(value):
        raise ValueError(f"g6_pad: value must be an int or a float, got {type(value).__name__}")
    if input_tensor.dtype in _INT_RANGES:
        if not math.isfinite(value):
            raise ValueError(f"g6_pad: value {value!r} is not finite for {input_tensor.dtype}")
        lo, hi = _INT_RANGES[input_tensor.dtype]
        if not lo <= math.trunc(value) <= hi:
            raise ValueError(f"g6_pad: value {value!r} outside the range of {input_tensor.dtype}")


def _scenario(input_tensor, padding, value):
    buffer_type = input_tensor.memory_config().buffer_type
    return {
        "shape": list(input_tensor.shape),
        "padding": [list(pair) for pair in padding],
        "value": value,
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "memory_config": "l1" if buffer_type == ttnn.BufferType.L1 else "dram",
    }


def validate(input_tensor, padding, value=0.0):
    _check_arguments(input_tensor, padding, value)

    axes = {}
    scenario = _scenario(input_tensor, padding, value)
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_pad: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_pad: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Host-side planning
# ---------------------------------------------------------------------------


def _fill_bits(dtype, value):
    """c as the bit pattern of x's dtype, converted the way torch fills a tensor of that dtype."""
    import torch

    if dtype == ttnn.bfloat16:
        return int(torch.full((1,), value, dtype=torch.bfloat16).view(torch.int16).item()) & 0xFFFF
    if dtype == ttnn.float32:
        return int(torch.full((1,), value, dtype=torch.float32).view(torch.int32).item()) & 0xFFFFFFFF
    if dtype in _INT_RANGES:
        return math.trunc(value) & 0xFFFFFFFF
    raise ValueError(f"g6_pad: no fill conversion for {dtype}")


def _is_extension(layout, in_shape, highs):
    rank = len(in_shape)
    if layout != ttnn.TILE_LAYOUT or highs[-2] <= 0:
        return False
    if any(h != 0 for i, h in enumerate(highs) if i != rank - 2):
        return False
    return -(-in_shape[-2] // TILE) == -(-(in_shape[-2] + highs[-2]) // TILE)


# Built descriptors keyed by everything that changes the program (BB-07). A hit only rewrites the
# common runtime args (buffer addresses, fill bits).
_DESCRIPTOR_CACHE = {}
_last_plan = {}


def last_plan():
    """The plan the most recent call selected (BB-08, BB-21)."""
    return dict(_last_plan)


def _descriptor_for(regime, input_tensor, output_tensor, in_shape, highs, out_shape):
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    key = (
        regime,
        tuple(in_shape),
        tuple(highs),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.memory_config().buffer_type,
        (grid.x, grid.y),
        descriptors.knobs_key(),
    )
    entry = _DESCRIPTOR_CACHE.get(key)
    if entry is None:
        geom = descriptors.make_geometry(
            in_shape, out_shape, input_tensor.element_size(), ttnn.tile_size(input_tensor.dtype)
        )
        if regime == "E":
            entry = descriptors.build_extension_descriptor(geom, input_tensor, grid)
        else:
            entry = descriptors.build_new_storage_descriptor(geom, input_tensor, output_tensor, grid)
        _DESCRIPTOR_CACHE[key] = entry
    return entry


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_pad(input_tensor: ttnn.Tensor, padding, value: float = 0.0) -> ttnn.Tensor:
    validate(input_tensor, padding, value)

    in_shape = list(input_tensor.shape)
    rank = len(in_shape)
    highs = [0] * (rank - len(padding)) + [int(pair[1]) for pair in padding]
    out_shape = [d + h for d, h in zip(in_shape, highs)]
    fill_bits = _fill_bits(input_tensor.dtype, value)

    if _is_extension(input_tensor.layout, in_shape, highs):
        descriptor, plan, roles = _descriptor_for("E", input_tensor, None, in_shape, highs, out_shape)
        descriptor.kernels[roles["extension_writer"]].common_runtime_args = [input_tensor.buffer_address(), fill_bits]
        _last_plan.clear()
        _last_plan.update(plan)
        ttnn.generic_op([input_tensor], descriptor)
        # Same buffer, same padded shape, new logical shape: a view that launches no program.
        return ttnn.reshape(input_tensor, ttnn.Shape(out_shape), input_tensor.padded_shape)

    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise NotImplementedError("g6_pad: ROW_MAJOR new_storage (regime R) is not built")

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        input_tensor.memory_config(),
    )
    descriptor, plan, roles = _descriptor_for("N", input_tensor, output_tensor, in_shape, highs, out_shape)
    descriptor.kernels[roles["new_storage_reader"]].common_runtime_args = [input_tensor.buffer_address(), fill_bits]
    descriptor.kernels[roles["new_storage_writer"]].common_runtime_args = [output_tensor.buffer_address(), fill_bits]
    _last_plan.clear()
    _last_plan.update(plan)
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
