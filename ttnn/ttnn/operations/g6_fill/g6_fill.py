# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_fill: out[i] = cast_T(fill_value) for every logical position of output_tensor.

One device program per call (ttnn.generic_op), a single writer kernel. The
input is never read; it supplies shape, dtype, layout and placement only.
"""

from __future__ import annotations

import math
import struct

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from . import g6_fill_program_descriptor as _pd


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------
#
# `inputs` is (scenario,), a scenario being {"shape", "dtype", "layout", "fill_value"}.


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


def tag_alignment(inputs, axes):
    # [N] is the tile plane [1, N]: H = 1 is never tile-aligned, so "tile_aligned" cannot occur.
    return "w_non_aligned" if inputs[0]["shape"][-1] % 32 else "h_non_aligned"


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


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------

FLT_MAX = 3.4028234663852886e38
_FLT_MAX_INT = int(FLT_MAX)
_INT32_MIN, _INT32_MAX = -(2**31), 2**31 - 1
_ADMITTED_DTYPES = (ttnn.float32, ttnn.int32)


def _check_tensor(tensor, what):
    if not isinstance(tensor, ttnn.Tensor):
        raise ValueError(f"g6_fill: {what} must be a ttnn.Tensor, got {type(tensor).__name__}")
    if not ttnn.is_tensor_storage_on_device(tensor):
        raise ValueError(f"g6_fill: {what} must be on the device")
    mc = tensor.memory_config()
    if mc.buffer_type != ttnn.BufferType.DRAM or mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError(f"g6_fill: {what} must be DRAM interleaved, got {mc.buffer_type}, {mc.memory_layout}")


def _check_fill_value(dtype, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"g6_fill: fill_value must be a Python int or float, got {type(value).__name__}")
    if dtype == ttnn.int32:
        if isinstance(value, float):
            raise ValueError("g6_fill: a float fill_value cannot fill an int32 tensor")
        if not _INT32_MIN <= value <= _INT32_MAX:
            raise ValueError(f"g6_fill: fill_value {value} is outside the int32 range")
        return
    if isinstance(value, int):
        if abs(value) > _FLT_MAX_INT:
            raise ValueError(f"g6_fill: fill_value {value} exceeds the float32 range")
    elif math.isfinite(value) and abs(value) > FLT_MAX:
        raise ValueError(f"g6_fill: fill_value {value!r} exceeds the float32 range")


def _scenario(input_tensor, fill_value):
    return (
        {
            "shape": tuple(input_tensor.shape),
            "dtype": input_tensor.dtype,
            "layout": input_tensor.layout,
            "fill_value": fill_value,
        },
    )


def validate(input_tensor, fill_value, output_tensor):
    # Contract violations (ValueError) come before the support gate: a support
    # refusal reads as "not yet supported", which these inputs never become.
    _check_tensor(input_tensor, "input_tensor")
    if len(input_tensor.shape) != 1 or int(input_tensor.shape[0]) < 1:
        raise ValueError(f"g6_fill: input_tensor must have shape [N] with N >= 1, got {tuple(input_tensor.shape)}")
    if input_tensor.dtype not in _ADMITTED_DTYPES:
        raise ValueError(f"g6_fill: input_tensor dtype must be float32 or int32, got {input_tensor.dtype}")

    _check_fill_value(input_tensor.dtype, fill_value)

    if output_tensor is None:
        raise ValueError("g6_fill: output_tensor is required")
    _check_tensor(output_tensor, "output_tensor")
    if output_tensor.device().id() != input_tensor.device().id():
        raise ValueError("g6_fill: output_tensor must be on the device of input_tensor")
    if (tuple(output_tensor.shape), output_tensor.dtype, output_tensor.layout) != (
        tuple(input_tensor.shape),
        input_tensor.dtype,
        input_tensor.layout,
    ):
        raise ValueError(
            f"g6_fill: output_tensor {tuple(output_tensor.shape)} {output_tensor.dtype} {output_tensor.layout} "
            f"differs from input_tensor {tuple(input_tensor.shape)} {input_tensor.dtype} {input_tensor.layout}"
        )
    if output_tensor.memory_config() != input_tensor.memory_config():
        raise ValueError("g6_fill: output_tensor memory config differs from input_tensor")

    inputs = _scenario(input_tensor, fill_value)
    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_fill: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_fill: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Element conversion e = cast_T(v), as its 32-bit pattern
# ---------------------------------------------------------------------------


def _int_to_float32_exact(value: int) -> float:
    """The float32 nearest to an int (ties to even), rounded once from the exact integer.

    float(value) would round to float64 first and then to float32: a double
    rounding for |value| > 2**53.
    """
    magnitude = abs(value)
    if magnitude <= 2**53:
        return float(value)
    shift = magnitude.bit_length() - 24
    q, r = divmod(magnitude, 1 << shift)
    half = 1 << (shift - 1)
    if r > half or (r == half and q & 1):
        q += 1
        if q == 1 << 24:
            q >>= 1
            shift += 1
    result = float(q << shift)
    return -result if value < 0 else result


def fill_bits(dtype, value) -> int:
    if dtype == ttnn.int32:
        return value & 0xFFFFFFFF
    as_float = _int_to_float32_exact(value) if isinstance(value, int) else value
    return struct.unpack("<I", struct.pack("<f", as_float))[0]


# ---------------------------------------------------------------------------
# Plan report and descriptor cache
# ---------------------------------------------------------------------------

_DESCRIPTOR_CACHE = {}


def g6_fill_plan(input_tensor: ttnn.Tensor) -> dict:
    """The plan a call on this tensor dispatches (regime, cores, per-core tiles, roles). No dispatch."""
    return _pd.plan_fill(input_tensor).report()


def _cached_descriptor(output_tensor):
    key = _pd.descriptor_cache_key(output_tensor)
    entry = _DESCRIPTOR_CACHE.get(key)
    if entry is None:
        entry = _pd.create_program_descriptor(output_tensor, _pd.plan_fill(output_tensor))
        _DESCRIPTOR_CACHE[key] = entry
    return entry


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_fill(input_tensor: ttnn.Tensor, fill_value, *, output_tensor: ttnn.Tensor) -> ttnn.Tensor:
    validate(input_tensor, fill_value, output_tensor)

    descriptor, writer = _cached_descriptor(output_tensor)
    common_args = [0, 0]
    common_args[_pd.COMMON_ARG_DST_ADDR] = output_tensor.buffer_address()
    common_args[_pd.COMMON_ARG_FILL_BITS] = fill_bits(output_tensor.dtype, fill_value)
    writer.common_runtime_args = common_args
    # ProgramDescriptor.kernels is a by-value vector: patching an element read
    # back from it edits a copy, so the patched kernel is assigned back whole.
    descriptor.kernels = [writer]

    ttnn.generic_op([output_tensor], descriptor)
    return output_tensor
