# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_fill: out[i] = cast_T(fill_value) for every logical position of output_tensor.

One device program per call (a writer kernel per core); no value of input_tensor is read.
Registry model: INPUT_TAGGERS, SUPPORTED, EXCLUSIONS and validate() below; see eval/op_template.py.
"""

from __future__ import annotations

import math
import struct

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from . import g6_fill_program_descriptor as _program

FLT_MAX = 3.4028234663852886e38
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1
FLOAT32_SIGNIFICAND_BITS = 24
TILE_W = 32

_ADMITTED_DTYPES = (ttnn.float32, ttnn.int32)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: inputs is (scenario,) with keys shape, dtype, layout, fill_value
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


def tag_alignment(inputs, axes):
    # [N] is the tile plane [1, N]: H = 1 is never tile-aligned, so "tile_aligned" cannot occur.
    return "w_non_aligned" if inputs[0]["shape"][-1] % TILE_W else "h_non_aligned"


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
# Contract checks and the element conversion
# ---------------------------------------------------------------------------


def _is_device_tensor(tensor) -> bool:
    return isinstance(tensor, ttnn.Tensor) and ttnn.is_tensor_storage_on_device(tensor)


def _is_dram_interleaved(tensor) -> bool:
    memory_config = tensor.memory_config()
    return (
        memory_config.buffer_type == ttnn.BufferType.DRAM
        and memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    )


def _check_input(input_tensor) -> None:
    if not _is_device_tensor(input_tensor):
        raise ValueError("g6_fill: input_tensor must be a ttnn.Tensor on the device")
    if len(input_tensor.shape) != 1:
        raise ValueError(f"g6_fill: input_tensor must have rank 1, got shape {tuple(input_tensor.shape)}")
    if int(input_tensor.shape[0]) < 1:
        raise ValueError("g6_fill: input_tensor must have at least one element")
    if input_tensor.dtype not in _ADMITTED_DTYPES:
        raise ValueError(f"g6_fill: input_tensor dtype must be float32 or int32, got {input_tensor.dtype}")
    if not _is_dram_interleaved(input_tensor):
        raise ValueError(f"g6_fill: input_tensor must be DRAM interleaved, got {input_tensor.memory_config()}")


def _check_fill_value(dtype, fill_value) -> None:
    if isinstance(fill_value, bool) or not isinstance(fill_value, (int, float)):
        raise ValueError(f"g6_fill: fill_value must be an int or a float, got {type(fill_value).__name__}")
    if dtype == ttnn.int32:
        if isinstance(fill_value, float):
            raise ValueError("g6_fill: a float fill_value cannot fill an int32 tensor")
        if not INT32_MIN <= fill_value <= INT32_MAX:
            raise ValueError(f"g6_fill: fill_value {fill_value} is outside the int32 range")
    elif isinstance(fill_value, int):
        # Compared as integers: float(fill_value) would round before the comparison.
        if abs(fill_value) > int(FLT_MAX):
            raise ValueError(f"g6_fill: fill_value {fill_value} is outside the float32 range")
    elif math.isfinite(fill_value) and abs(fill_value) > FLT_MAX:
        raise ValueError(f"g6_fill: fill_value {fill_value!r} is outside the float32 range")


def _check_output(input_tensor, output_tensor) -> None:
    if output_tensor is None:
        raise ValueError("g6_fill: output_tensor is required")
    if not _is_device_tensor(output_tensor):
        raise ValueError("g6_fill: output_tensor must be a ttnn.Tensor on the device")
    if output_tensor.device().id() != input_tensor.device().id():
        raise ValueError("g6_fill: output_tensor must be on the device of input_tensor")
    mismatches = [
        name
        for name, got, want in (
            ("shape", tuple(output_tensor.shape), tuple(input_tensor.shape)),
            ("dtype", output_tensor.dtype, input_tensor.dtype),
            ("layout", output_tensor.layout, input_tensor.layout),
            ("memory_config", output_tensor.memory_config(), input_tensor.memory_config()),
        )
        if got != want
    ]
    if mismatches:
        raise ValueError(f"g6_fill: output_tensor differs from input_tensor in {', '.join(mismatches)}")


def _round_int_to_float32(value: int) -> float:
    """value rounded once to float32 (nearest, ties to even), returned as an exact Python float.

    float(value) rounds to float64 first, and a second rounding to float32 can then land on the
    wrong neighbour (2**60 + 2**36 + 1 -> 2**60 instead of 2**60 + 2**37).
    """
    magnitude = abs(value)
    shift = magnitude.bit_length() - FLOAT32_SIGNIFICAND_BITS
    if shift > 0:
        significand, remainder = divmod(magnitude, 1 << shift)
        half = 1 << (shift - 1)
        if remainder > half or (remainder == half and significand & 1):
            significand += 1
        magnitude = significand << shift
    return math.copysign(float(magnitude), value)


def fill_bits_of(dtype, fill_value) -> int:
    """The 32-bit pattern of e = cast_T(fill_value); fill_value already passed _check_fill_value."""
    if dtype == ttnn.int32:
        return fill_value & 0xFFFFFFFF
    if isinstance(fill_value, int):
        fill_value = _round_int_to_float32(fill_value)
    # struct rounds the double once to float32 and keeps -0.0, inf, NaN and subnormals.
    return struct.unpack("<I", struct.pack("<f", fill_value))[0]


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _scenario(input_tensor, fill_value):
    return (
        {
            "shape": tuple(int(d) for d in input_tensor.shape),
            "dtype": input_tensor.dtype,
            "layout": input_tensor.layout,
            "fill_value": fill_value,
        },
    )


def validate(input_tensor, fill_value, output_tensor) -> None:
    # Contract violations (ValueError) come before the support gate: a support refusal is a
    # NotImplementedError and would read as "not yet supported" instead of "invalid".
    _check_input(input_tensor)
    _check_fill_value(input_tensor.dtype, fill_value)
    _check_output(input_tensor, output_tensor)

    inputs = _scenario(input_tensor, fill_value)
    axes = {"dtype": input_tensor.dtype, "layout": input_tensor.layout}
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger(inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_fill: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for excluded in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in excluded.items()):
            raise ExcludedCell(f"g6_fill: unsupported combination (refinement candidate): {excluded}")


# ---------------------------------------------------------------------------
# Plan report (BB-08, BB-21)
# ---------------------------------------------------------------------------


def g6_fill_plan(input_tensor) -> dict:
    """The plan a call on input_tensor dispatches, without dispatching it."""
    plan = _program.build_plan(
        input_tensor, max_fill_cores=_program.MAX_FILL_CORES, block_w_tiles=_program.BLOCK_W_TILES
    )
    cores = [(core.x, core.y) for core, *_ in plan["assignment"]]
    return {
        "regime": plan["regime"],
        "num_cores": plan["num_cores"],
        "core_ranges": [((r.start.x, r.start.y), (r.end.x, r.end.y)) for r in plan["all_cores"].ranges()],
        "start_tile": {(core.x, core.y): start for core, start, _, _ in plan["assignment"]},
        "tiles_per_core": {(core.x, core.y): count for core, _, count, _ in plan["assignment"]},
        "block_w_tiles": {(core.x, core.y): block for core, _, _, block in plan["assignment"]},
        "segment_bytes": plan["segment_bytes"],
        "segments_per_tile": plan["segments_per_tile"],
        "roles": {core: "fill_writer" for core in cores},
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_fill(input_tensor, fill_value, *, output_tensor):
    validate(input_tensor, fill_value, output_tensor)
    _program.dispatch(output_tensor, fill_bits_of(input_tensor.dtype, fill_value))
    return output_tensor
