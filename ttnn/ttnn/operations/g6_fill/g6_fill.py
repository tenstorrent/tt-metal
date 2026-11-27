# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_fill: write cast_T(fill_value) into every logical position of output_tensor.

    out[i] = cast_T(v)    for every logical position i of input_tensor (rank 1, [N])

No value of input_tensor is read; it supplies shape, dtype, layout and
placement. The result is written into output_tensor's own buffer by one device
program and output_tensor is returned; output_tensor may be input_tensor.
"""

from __future__ import annotations

import math
import struct

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from . import g6_fill_program_descriptor as pd

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — inputs is (scenario,) with keys shape, dtype, layout, fill_value
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


def tag_alignment(inputs, axes):
    # [N] reads as the tile plane [1, N]; H = 1 is never tile-aligned, so "tile_aligned" cannot occur.
    n = inputs[0]["shape"][-1]
    return "w_non_aligned" if n % ttnn.TILE_SIZE else "h_non_aligned"


INPUT_TAGGERS = {
    "dtype": tag_dtype,
    "layout": tag_layout,
    "alignment": tag_alignment,
}

# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.float32],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["w_non_aligned", "h_non_aligned"],
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
# Value conversion e_bits = cast_T(v)
# ---------------------------------------------------------------------------

_CONTRACT_DTYPES = (ttnn.float32, ttnn.int32)
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1
FLT_MAX = 3.4028234663852886e38
# FLT_MAX is an integer; ints are range-checked against it exactly, never through float().
_FLT_MAX_INT = int(FLT_MAX)
_F32_MANTISSA_BITS = 24


def _check_value(dtype, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"g6_fill: fill_value must be an int or a float, got {type(value).__name__}")
    if dtype == ttnn.int32:
        if isinstance(value, float):
            raise ValueError("g6_fill: a float fill_value cannot be cast exactly to an int32 tensor")
        if not INT32_MIN <= value <= INT32_MAX:
            raise ValueError(f"g6_fill: fill_value {value} is outside the int32 range")
    elif isinstance(value, int):
        if abs(value) > _FLT_MAX_INT:
            raise ValueError(f"g6_fill: fill_value {value} exceeds the float32 range")
    elif math.isfinite(value) and abs(value) > FLT_MAX:
        raise ValueError(f"g6_fill: fill_value {value!r} exceeds the float32 range")


def _f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _int_to_f32_bits(value):
    # int -> double -> float rounds twice (2**60 + 2**36 + 1 lands one ulp low); round once, exactly.
    magnitude = abs(value)
    excess = magnitude.bit_length() - _F32_MANTISSA_BITS
    if excess > 0:
        q, r = divmod(magnitude, 1 << excess)
        half = 1 << (excess - 1)
        if r > half or (r == half and q & 1):
            q += 1
        magnitude = q << excess
    bits = _f32_bits(float(magnitude))
    return bits | 0x80000000 if value < 0 else bits


def element_bits(dtype, value):
    """The 32-bit pattern of cast_T(value); value must already have passed _check_value."""
    if dtype == ttnn.int32:
        return value & 0xFFFFFFFF
    if isinstance(value, int):
        return _int_to_f32_bits(value)
    return _f32_bits(value)


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _is_dram_interleaved(tensor):
    mc = tensor.memory_config()
    return mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED and mc.buffer_type == ttnn.BufferType.DRAM


def _check_tensors(input_tensor, output_tensor):
    if output_tensor is None:
        raise ValueError("g6_fill: output_tensor is required")
    for name, t in (("input_tensor", input_tensor), ("output_tensor", output_tensor)):
        if not isinstance(t, ttnn.Tensor):
            raise ValueError(f"g6_fill: {name} must be a ttnn.Tensor, got {type(t).__name__}")
        if not ttnn.is_tensor_storage_on_device(t):
            raise ValueError(f"g6_fill: {name} must be on the device")
    if len(input_tensor.shape) != 1:
        raise ValueError(f"g6_fill: input_tensor must have rank 1, got shape {tuple(input_tensor.shape)}")
    if input_tensor.dtype not in _CONTRACT_DTYPES:
        raise ValueError(f"g6_fill: dtype {input_tensor.dtype} is not float32 or int32")
    for name, t in (("input_tensor", input_tensor), ("output_tensor", output_tensor)):
        if not _is_dram_interleaved(t):
            raise ValueError(f"g6_fill: {name} must be DRAM interleaved, got {t.memory_config()}")
    if tuple(output_tensor.shape) != tuple(input_tensor.shape):
        raise ValueError(
            f"g6_fill: output_tensor shape {tuple(output_tensor.shape)} != input shape {tuple(input_tensor.shape)}"
        )
    if output_tensor.dtype != input_tensor.dtype:
        raise ValueError(f"g6_fill: output_tensor dtype {output_tensor.dtype} != input dtype {input_tensor.dtype}")
    if output_tensor.layout != input_tensor.layout:
        raise ValueError(f"g6_fill: output_tensor layout {output_tensor.layout} != input layout {input_tensor.layout}")
    if output_tensor.device() != input_tensor.device():
        raise ValueError("g6_fill: output_tensor is on another device than input_tensor")


def _axes(input_tensor, fill_value):
    inputs = (
        {
            "shape": tuple(input_tensor.shape),
            "dtype": input_tensor.dtype,
            "layout": input_tensor.layout,
            "fill_value": fill_value,
        },
    )
    axes = {}
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger(inputs, axes)
    return axes


def validate(input_tensor, fill_value, *, output_tensor):
    # Every refusal is raised before anything is dispatched, so a refused call leaves output_tensor as it was.
    _check_tensors(input_tensor, output_tensor)
    _check_value(input_tensor.dtype, fill_value)

    axes = _axes(input_tensor, fill_value)
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_fill: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_fill: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Plan report and descriptor cache
# ---------------------------------------------------------------------------

# Keyed by everything that changes the program; the fill value and buffer address are common RT args.
_DESCRIPTOR_CACHE = {}


def _cache_key(tensor):
    grid = tensor.device().compute_with_storage_grid_size()
    return (
        tuple(tensor.shape),
        tensor.dtype,
        tensor.layout,
        grid.x,
        grid.y,
        pd.MAX_ACTIVE_CORES,
        pd.MAX_RECT_W,
        pd.BLOCK_W_TILES,
        pd.FILL_ROW_DEPTH,
    )


def g6_fill_plan(input_tensor):
    """The plan a call on input_tensor selects: regime, active rectangle and per-core segment ranges."""
    plan = pd.plan_for(input_tensor)
    return {
        "regime": plan.regime,
        "rect": (plan.rect_w, plan.rect_h),
        "per_core": dict(plan.per_core),
        "role": {core: "writer" for core in plan.per_core},
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_fill(input_tensor, fill_value, *, output_tensor):
    validate(input_tensor, fill_value, output_tensor=output_tensor)

    e_bits = element_bits(input_tensor.dtype, fill_value)
    key = _cache_key(output_tensor)
    entry = _DESCRIPTOR_CACHE.get(key)
    if entry is None:
        plan = pd.plan_for(output_tensor)
        descriptor, writer = pd.create_program_descriptor(plan, output_tensor, e_bits)
        entry = _DESCRIPTOR_CACHE[key] = (plan, descriptor, writer)
    else:
        plan, descriptor, writer = entry
        # descriptor.kernels holds copies of the KernelDescriptor: re-assign instead of mutating in place.
        writer.common_runtime_args = pd.common_runtime_args(plan, output_tensor, e_bits)
        descriptor.kernels = [writer]

    ttnn.generic_op([output_tensor], descriptor)
    return output_tensor
