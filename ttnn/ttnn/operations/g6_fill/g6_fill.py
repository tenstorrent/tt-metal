# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_fill: out[i] = cast_T(fill_value) for every logical position of output_tensor.

One device program per call (a single writer kernel). No value of input_tensor is read; it
supplies shape, dtype, layout and placement only. The result is written into output_tensor's
own buffer, which may be input_tensor itself.
"""

from __future__ import annotations

import math
import struct

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from . import g6_fill_program_descriptor as _pd

FLT_MAX = 3.4028234663852886e38
_FLT_MAX_INT = int(FLT_MAX)
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1
_FLOAT32_MANTISSA_BITS = 24


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS. `inputs` is (scenario,), a dict with "shape", "dtype", "layout",
# "fill_value".
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


def tag_alignment(inputs, axes):
    # [N] is the tile plane [1, N]: H = 1 is never tile-aligned, so only W decides.
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
# Contract checks (ValueError). They run before the support gate so a contract violation is
# never reported as a support refusal.
# ---------------------------------------------------------------------------

_DTYPES = (ttnn.float32, ttnn.int32)


def _on_device(tensor) -> bool:
    return ttnn.is_tensor_storage_on_device(tensor)


def _check_input_tensor(input_tensor):
    if not isinstance(input_tensor, ttnn.Tensor):
        raise ValueError(f"g6_fill: input_tensor must be a ttnn.Tensor, got {type(input_tensor).__name__}")
    if not _on_device(input_tensor):
        raise ValueError("g6_fill: input_tensor must be on the device")
    shape = tuple(input_tensor.shape)
    if len(shape) != 1:
        raise ValueError(f"g6_fill: input_tensor must have rank 1, got shape {shape}")
    if shape[0] < 1:
        raise ValueError(f"g6_fill: input_tensor must hold at least one element, got shape {shape}")
    if input_tensor.dtype not in _DTYPES:
        raise ValueError(f"g6_fill: dtype {input_tensor.dtype} is not float32 or int32")
    memory_config = input_tensor.memory_config()
    if (
        memory_config.buffer_type != ttnn.BufferType.DRAM
        or memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    ):
        raise ValueError(f"g6_fill: input_tensor must be DRAM interleaved, got {memory_config}")


def _check_fill_value(dtype, fill_value):
    if isinstance(fill_value, bool) or not isinstance(fill_value, (int, float)):
        raise ValueError(f"g6_fill: fill_value must be a Python int or float, got {type(fill_value).__name__}")
    if dtype == ttnn.int32:
        if isinstance(fill_value, float):
            raise ValueError("g6_fill: a float fill_value cannot fill an int32 tensor")
        if not INT32_MIN <= fill_value <= INT32_MAX:
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
    if not _on_device(output_tensor):
        raise ValueError("g6_fill: output_tensor must be on the device")
    if output_tensor.device().id() != input_tensor.device().id():
        raise ValueError("g6_fill: output_tensor must be on the device of input_tensor")
    if tuple(output_tensor.shape) != tuple(input_tensor.shape):
        raise ValueError(
            f"g6_fill: output shape {tuple(output_tensor.shape)} != input shape {tuple(input_tensor.shape)}"
        )
    if output_tensor.dtype != input_tensor.dtype:
        raise ValueError(f"g6_fill: output dtype {output_tensor.dtype} != input dtype {input_tensor.dtype}")
    if output_tensor.layout != input_tensor.layout:
        raise ValueError(f"g6_fill: output layout {output_tensor.layout} != input layout {input_tensor.layout}")
    if output_tensor.memory_config() != input_tensor.memory_config():
        raise ValueError(
            f"g6_fill: output memory config {output_tensor.memory_config()} != input {input_tensor.memory_config()}"
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


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(input_tensor, fill_value, output_tensor):
    _check_input_tensor(input_tensor)
    _check_fill_value(input_tensor.dtype, fill_value)
    _check_output_tensor(input_tensor, output_tensor)

    axes = {"dtype": input_tensor.dtype, "layout": input_tensor.layout}
    inputs = _scenario(input_tensor, fill_value)
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger(inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_fill: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for excluded in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in excluded.items()):
            raise ExcludedCell(f"g6_fill: unsupported combination (refinement candidate): {excluded}")


# ---------------------------------------------------------------------------
# Element conversion: e = cast_T(fill_value) as a 32-bit pattern
# ---------------------------------------------------------------------------


def _float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _int_to_float32_exact(value: int) -> float:
    """Round an int to float32 once, ties to even, as a float that packs without a second rounding.

    float(value) rounds to 53 bits first, which double-rounds ints wider than 2**53.
    """
    magnitude = abs(value)
    if magnitude <= 2**53:
        return float(value)
    shift = magnitude.bit_length() - _FLOAT32_MANTISSA_BITS
    q, r = divmod(magnitude, 1 << shift)
    half = 1 << (shift - 1)
    if r > half or (r == half and q & 1):
        q += 1
    if q == 1 << _FLOAT32_MANTISSA_BITS:
        q >>= 1
        shift += 1
    rounded = float(q << shift)
    return -rounded if value < 0 else rounded


def fill_bits(dtype, fill_value) -> int:
    """The 32-bit pattern of e; fill_value must already have passed validate()."""
    if dtype == ttnn.int32:
        return fill_value & 0xFFFFFFFF
    if isinstance(fill_value, int):
        return _float_bits(_int_to_float32_exact(fill_value))
    return _float_bits(fill_value)


# ---------------------------------------------------------------------------
# Plan report and descriptor cache
# ---------------------------------------------------------------------------


def _plan(tensor):
    return _pd.make_plan(tensor, max_fill_cores=_pd.MAX_FILL_CORES, block_w_tiles=_pd.BLOCK_W_TILES)


def g6_fill_plan(input_tensor) -> dict:
    """The plan a call on input_tensor dispatches, with the role of every core. No dispatch."""
    plan = _plan(input_tensor)
    assignment = _pd.core_assignment(plan)
    return {
        "regime": plan.regime,
        "num_cores": plan.num_cores,
        "core_ranges": [((r.start.x, r.start.y), (r.end.x, r.end.y)) for r in plan.all_cores.ranges()],
        "tiles_per_core": {(core.x, core.y): n for core, _, n in assignment},
        "block_w_tiles": plan.block_w_tiles,
        "segment_bytes": plan.segment_bytes,
        "segments_per_tile": plan.segments_per_tile,
        "roles": {(core.x, core.y): "fill_writer" for core, _, _ in assignment},
    }


_DESCRIPTOR_CACHE = {}


def _cache_key(tensor):
    return (
        tensor.device().id(),
        tuple(tensor.shape),
        tensor.dtype,
        tensor.layout,
        str(tensor.memory_config()),
        _pd.MAX_FILL_CORES,
        _pd.BLOCK_W_TILES,
    )


def _cached_descriptor(output_tensor):
    key = _cache_key(output_tensor)
    entry = _DESCRIPTOR_CACHE.get(key)
    if entry is None:
        entry = _pd.create_program_descriptor(output_tensor, _plan(output_tensor))
        _DESCRIPTOR_CACHE[key] = entry
    return entry


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_fill(input_tensor, fill_value, *, output_tensor):
    validate(input_tensor, fill_value, output_tensor)
    descriptor, writer = _cached_descriptor(output_tensor)
    _pd.set_call_args(descriptor, writer, output_tensor.buffer_address(), fill_bits(output_tensor.dtype, fill_value))
    ttnn.generic_op([output_tensor], descriptor)
    return output_tensor
