# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_fill: out[i] = cast_T(fill_value) for every logical position of output_tensor.

One device program per call, writer-only. The input tensor supplies shape, dtype,
layout and placement and is never read. The 32-bit element reaches the cached
program as a common runtime argument, so a new fill_value never rebuilds it.
"""

from __future__ import annotations

import math
import struct

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from . import g6_fill_program_descriptor as _pd

FLT_MAX = 3.4028234663852886e38
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1
_DTYPES = (ttnn.float32, ttnn.int32)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — `inputs` is (scenario,) with keys shape, dtype, layout, fill_value
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


def tag_alignment(inputs, axes):
    # [N] is the tile plane [1, N]: H = 1 is never aligned, so "tile_aligned" cannot occur.
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


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _on_device(tensor):
    return isinstance(tensor, ttnn.Tensor) and tensor.storage_type() == ttnn.StorageType.DEVICE


def _is_dram_interleaved(memory_config):
    return (
        memory_config.buffer_type == ttnn.BufferType.DRAM
        and memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    )


def _check_input(input_tensor):
    if not _on_device(input_tensor):
        raise ValueError("g6_fill: input_tensor must be a ttnn.Tensor on the device")
    shape = tuple(input_tensor.shape)
    if len(shape) != 1:
        raise ValueError(f"g6_fill: input_tensor must have rank 1, got shape {shape}")
    if shape[0] < 1:
        raise ValueError(f"g6_fill: input_tensor must have N >= 1, got shape {shape}")
    if input_tensor.dtype not in _DTYPES:
        raise ValueError(f"g6_fill: input_tensor dtype must be float32 or int32, got {input_tensor.dtype}")
    if not _is_dram_interleaved(input_tensor.memory_config()):
        raise ValueError(f"g6_fill: input_tensor must be DRAM interleaved, got {input_tensor.memory_config()}")


def _check_fill_value(dtype, fill_value):
    if isinstance(fill_value, bool) or not isinstance(fill_value, (int, float)):
        raise ValueError(f"g6_fill: fill_value must be an int or a float, got {type(fill_value).__name__}")
    if dtype == ttnn.int32:
        if isinstance(fill_value, float):
            raise ValueError("g6_fill: a float fill_value is not admitted for an int32 tensor")
        if not INT32_MIN <= fill_value <= INT32_MAX:
            raise ValueError(f"g6_fill: fill_value {fill_value} is outside the int32 range")
    elif isinstance(fill_value, int):
        # Compared exactly: float(fill_value) could round an over-range int down to FLT_MAX.
        if abs(fill_value) > int(FLT_MAX):
            raise ValueError(f"g6_fill: fill_value {fill_value} exceeds the float32 range")
    elif math.isfinite(fill_value) and abs(fill_value) > FLT_MAX:
        raise ValueError(f"g6_fill: fill_value {fill_value!r} exceeds the float32 range")


def _check_output(input_tensor, output_tensor):
    if output_tensor is None:
        raise ValueError("g6_fill: output_tensor is required")
    if not _on_device(output_tensor):
        raise ValueError("g6_fill: output_tensor must be a ttnn.Tensor on the device")
    if output_tensor.device().id() != input_tensor.device().id():
        raise ValueError("g6_fill: output_tensor must be on the device of input_tensor")
    got = (tuple(output_tensor.shape), output_tensor.dtype, output_tensor.layout)
    want = (tuple(input_tensor.shape), input_tensor.dtype, input_tensor.layout)
    if got != want:
        raise ValueError(f"g6_fill: output_tensor is (shape, dtype, layout) {got}, input_tensor is {want}")
    if output_tensor.memory_config() != input_tensor.memory_config():
        raise ValueError(
            f"g6_fill: output_tensor memory config {output_tensor.memory_config()} "
            f"differs from input_tensor's {input_tensor.memory_config()}"
        )


def _scenario(input_tensor, fill_value):
    return (
        {
            "shape": tuple(int(d) for d in input_tensor.shape),
            "dtype": input_tensor.dtype,
            "layout": input_tensor.layout,
            "fill_value": fill_value,
        },
    )


def validate(input_tensor, fill_value, *, output_tensor):
    # Contract violations (ValueError) come before the support gate: a SupportRefusal
    # would be read as "not yet supported" instead of "invalid".
    _check_input(input_tensor)
    _check_fill_value(input_tensor.dtype, fill_value)
    _check_output(input_tensor, output_tensor)

    inputs = _scenario(input_tensor, fill_value)
    axes = {"dtype": input_tensor.dtype, "layout": input_tensor.layout}
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


def _int_to_float_once(v):
    """Python float holding cast_float32(v) exactly, rounded once from the exact integer.

    float(v) rounds to 53 bits and struct.pack then to 24, which double-rounds above 2**53.
    """
    a = abs(v)
    if a <= 2**53:
        return float(v)
    shift = a.bit_length() - 24
    q, r = divmod(a, 1 << shift)
    half = 1 << (shift - 1)
    if r > half or (r == half and q & 1):
        q += 1
        if q == 1 << 24:
            q >>= 1
            shift += 1
    f = float(q << shift)
    return -f if v < 0 else f


def fill_bits_of(dtype, fill_value):
    if dtype == ttnn.int32:
        return fill_value & 0xFFFFFFFF
    v = _int_to_float_once(fill_value) if isinstance(fill_value, int) else fill_value
    return struct.unpack("<I", struct.pack("<f", v))[0]


# ---------------------------------------------------------------------------
# Descriptor cache (BB-07): build once per key, patch the common runtime args per call
# ---------------------------------------------------------------------------

_DESCRIPTOR_CACHE = {}


def _cache_key(tensor):
    memory_config = tensor.memory_config()
    return (
        tensor.device().id(),
        tuple(int(d) for d in tensor.shape),
        tensor.dtype,
        tensor.layout,
        memory_config.buffer_type,
        memory_config.memory_layout,
        _pd.MAX_FILL_CORES,
        _pd.BLOCK_W_TILES,
        _pd.FILL_ROW_DEPTH,
    )


def _cached_descriptor(output_tensor):
    key = _cache_key(output_tensor)
    entry = _DESCRIPTOR_CACHE.get(key)
    if entry is None:
        entry = _pd.create_program_descriptor(output_tensor)
        _DESCRIPTOR_CACHE[key] = entry
    return entry


def g6_fill_plan(input_tensor):
    """The plan a call on input_tensor selects (regime, cores, per-core tiles, roles). No dispatch."""
    return _pd.plan_report(input_tensor)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_fill(input_tensor, fill_value, *, output_tensor):
    validate(input_tensor, fill_value, output_tensor=output_tensor)

    desc, writer = _cached_descriptor(output_tensor)
    common_args = [0] * _pd.NUM_COMMON_RT_ARGS
    common_args[_pd.CRTA_DST_ADDR] = output_tensor.buffer_address()
    common_args[_pd.CRTA_FILL_BITS] = fill_bits_of(output_tensor.dtype, fill_value)
    writer.common_runtime_args = common_args
    # desc.kernels is a by-value property: the patched writer must be assigned back.
    desc.kernels = [writer]

    # output_tensor is the only io tensor: the input is never an operand, so the
    # in-place form needs nothing of its own.
    ttnn.generic_op([output_tensor], desc)
    return output_tensor
