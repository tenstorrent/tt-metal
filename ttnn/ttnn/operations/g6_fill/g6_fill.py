# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_fill: out[i] = cast_T(fill_value) for every logical position of output_tensor.

One device program per call: a single writer kernel per core writes the fill element
into the row-0 face segments of its tiles. input_tensor supplies shape, dtype, layout
and placement only; none of its values is read.
"""

from __future__ import annotations

import math
import struct

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from . import g6_fill_program_descriptor as _pd


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — inputs is (scenario,), scenario has shape / dtype / layout / fill_value
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


def tag_alignment(inputs, axes):
    # [N] read as the tile plane [1, N]: H = 1 is never tile-aligned, so "tile_aligned" never occurs.
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
# Contract constants and element conversion
# ---------------------------------------------------------------------------

FLT_MAX = 3.4028234663852886e38
_FLT_MAX_INT = int(FLT_MAX)
_INT32_MIN, _INT32_MAX = -(2**31), 2**31 - 1
_EXACT_INT_IN_DOUBLE = 2**53
_FLOAT32_SIGNIFICAND_BITS = 24

_CONTRACT_DTYPES = (ttnn.float32, ttnn.int32)


def _float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _int_to_float32_exact(value: int) -> float:
    """value rounded once to 24 significant bits (ties to even), returned as an exact double.

    float(value) for |value| > 2**53 already rounds to 53 bits, and packing that to float32
    rounds a second time.
    """
    magnitude = abs(value)
    shift = magnitude.bit_length() - _FLOAT32_SIGNIFICAND_BITS
    if shift <= 0:
        return float(value)
    q, r = divmod(magnitude, 1 << shift)
    half = 1 << (shift - 1)
    if r > half or (r == half and q & 1):
        q += 1
        if q == 1 << _FLOAT32_SIGNIFICAND_BITS:
            q >>= 1
            shift += 1
    rounded = float(q << shift)
    return -rounded if value < 0 else rounded


def element_bits(dtype, fill_value) -> int:
    """The 32-bit pattern of e = cast_T(fill_value). Assumes validate() admitted the pair."""
    if dtype == ttnn.int32:
        return fill_value & 0xFFFFFFFF
    if isinstance(fill_value, int) and abs(fill_value) > _EXACT_INT_IN_DOUBLE:
        return _float_bits(_int_to_float32_exact(fill_value))
    return _float_bits(float(fill_value))


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _is_device_tensor(tensor) -> bool:
    return isinstance(tensor, ttnn.Tensor) and ttnn.is_tensor_storage_on_device(tensor)


def _is_dram_interleaved(tensor) -> bool:
    memory_config = tensor.memory_config()
    return (
        memory_config.buffer_type == ttnn.BufferType.DRAM
        and memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    )


def _check_input_tensor(input_tensor):
    if not isinstance(input_tensor, ttnn.Tensor):
        raise ValueError(f"g6_fill: input_tensor must be a ttnn.Tensor, got {type(input_tensor).__name__}")
    if not _is_device_tensor(input_tensor):
        raise ValueError("g6_fill: input_tensor must be on the device")
    shape = tuple(input_tensor.shape)
    if len(shape) != 1:
        raise ValueError(f"g6_fill: input_tensor must have rank 1, got shape {shape}")
    if shape[0] < 1:
        raise ValueError(f"g6_fill: input_tensor must have N >= 1, got shape {shape}")
    if input_tensor.dtype not in _CONTRACT_DTYPES:
        raise ValueError(f"g6_fill: input_tensor dtype must be float32 or int32, got {input_tensor.dtype}")
    if not _is_dram_interleaved(input_tensor):
        raise ValueError(f"g6_fill: input_tensor must be DRAM interleaved, got {input_tensor.memory_config()}")


def _check_fill_value(dtype, fill_value):
    if isinstance(fill_value, bool) or not isinstance(fill_value, (int, float)):
        raise ValueError(f"g6_fill: fill_value must be a Python int or float, got {type(fill_value).__name__}")
    if dtype == ttnn.int32:
        if isinstance(fill_value, float):
            raise ValueError("g6_fill: a float fill_value cannot fill an int32 tensor")
        if not _INT32_MIN <= fill_value <= _INT32_MAX:
            raise ValueError(f"g6_fill: fill_value {fill_value} is outside the int32 range")
        return
    if isinstance(fill_value, int):
        if abs(fill_value) > _FLT_MAX_INT:
            raise ValueError(f"g6_fill: fill_value {fill_value} exceeds the float32 range")
    elif math.isfinite(fill_value) and abs(fill_value) > FLT_MAX:
        raise ValueError(f"g6_fill: fill_value {fill_value!r} exceeds the float32 range")


def _check_output_tensor(input_tensor, output_tensor):
    if output_tensor is None:
        raise ValueError("g6_fill: output_tensor is required")
    if not isinstance(output_tensor, ttnn.Tensor):
        raise ValueError(f"g6_fill: output_tensor must be a ttnn.Tensor, got {type(output_tensor).__name__}")
    if output_tensor is input_tensor:
        return
    if not _is_device_tensor(output_tensor):
        raise ValueError("g6_fill: output_tensor must be on the device")
    if output_tensor.device().id() != input_tensor.device().id():
        raise ValueError("g6_fill: output_tensor must be on the device of input_tensor")
    if tuple(output_tensor.shape) != tuple(input_tensor.shape):
        raise ValueError(
            f"g6_fill: output_tensor shape {tuple(output_tensor.shape)} differs from {tuple(input_tensor.shape)}"
        )
    if output_tensor.dtype != input_tensor.dtype:
        raise ValueError(f"g6_fill: output_tensor dtype {output_tensor.dtype} differs from {input_tensor.dtype}")
    if output_tensor.layout != input_tensor.layout:
        raise ValueError(f"g6_fill: output_tensor layout {output_tensor.layout} differs from {input_tensor.layout}")
    if output_tensor.memory_config() != input_tensor.memory_config():
        raise ValueError(
            f"g6_fill: output_tensor memory config {output_tensor.memory_config()} differs from "
            f"{input_tensor.memory_config()}"
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


def validate(input_tensor, fill_value, output_tensor):
    # Contract violations (ValueError) come before the support gate, so an invalid call is never
    # mistaken for a not-yet-supported one.
    _check_input_tensor(input_tensor)
    _check_fill_value(input_tensor.dtype, fill_value)
    _check_output_tensor(input_tensor, output_tensor)

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
# Plan report and descriptor cache
# ---------------------------------------------------------------------------

_DESCRIPTOR_CACHE = {}


def _plan(tensor):
    max_fill_cores, block_w_tiles, fill_row_depth = _pd.knob_values()
    return _pd.compute_plan(
        tensor, max_fill_cores=max_fill_cores, block_w_tiles=block_w_tiles, fill_row_depth=fill_row_depth
    )


def g6_fill_plan(input_tensor) -> dict:
    """The plan a g6_fill call on this tensor dispatches (no dispatch)."""
    plan = _plan(input_tensor)
    report = {k: v for k, v in plan.items() if k not in ("all_cores", "assignment")}
    return report


def _cache_key(tensor):
    return (
        tensor.device().id(),
        tuple(int(d) for d in tensor.shape),
        tensor.dtype,
        tensor.layout,
        str(tensor.memory_config()),
        _pd.knob_values(),
    )


def _cached_program(output_tensor):
    key = _cache_key(output_tensor)
    entry = _DESCRIPTOR_CACHE.get(key)
    if entry is None:
        entry = _pd.create_program_descriptor(output_tensor, _plan(output_tensor))
        _DESCRIPTOR_CACHE[key] = entry
    return entry


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_fill(input_tensor: ttnn.Tensor, fill_value, *, output_tensor: ttnn.Tensor) -> ttnn.Tensor:
    validate(input_tensor, fill_value, output_tensor)

    descriptor, writer = _cached_program(output_tensor)
    common_args = [0] * _pd.NUM_COMMON_ARGS
    common_args[_pd.COMMON_ARG_DST_ADDR] = output_tensor.buffer_address()
    common_args[_pd.COMMON_ARG_FILL_BITS] = element_bits(output_tensor.dtype, fill_value)
    writer.common_runtime_args = common_args
    descriptor.kernels = [writer]

    # input_tensor is not an operand: it is never read, and the in-place form needs no aliasing.
    ttnn.generic_op([output_tensor], descriptor)
    return output_tensor
