# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_zeros: a new DRAM-interleaved TILE tensor whose every element is +0.0.

One device program per call: each worker core zeroes a small L1 source once and writes its share
of one DRAM bank as wide single-packet NoC writes. Nothing is read from DRAM.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_zeros_program_descriptor import get_program, plan_for


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------
# `inputs` is (scenario,), scenario = {"shape": tuple, "dtype": ..., "layout": ...}: the op has no
# input tensor, so both axes are projected off the call's own arguments.


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


INPUT_TAGGERS = {
    "dtype": tag_dtype,
    "layout": tag_layout,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.float32],
    "layout": [ttnn.TILE_LAYOUT],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------

_RANK = 4
_TILE_DIM = 32


def _check_structure(shape, layout, device, memory_config):
    if not isinstance(shape, (list, tuple)) or len(shape) != _RANK:
        raise ValueError(f"g6_zeros: shape must be a list or tuple of {_RANK} ints, got {shape!r}")
    for d in shape:
        if isinstance(d, bool) or not isinstance(d, int):
            raise ValueError(f"g6_zeros: shape must hold ints, got {shape!r}")
        if d < 1:
            raise ValueError(f"g6_zeros: every size must be >= 1, got {shape!r}")
    if shape[-2] % _TILE_DIM != 0 or shape[-1] % _TILE_DIM != 0:
        raise ValueError(f"g6_zeros: the last two sizes must be multiples of {_TILE_DIM}, got {shape!r}")
    if layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"g6_zeros: layout must be TILE_LAYOUT, got {layout!r}")
    if not isinstance(memory_config, ttnn.MemoryConfig) or not (
        memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
        and memory_config.buffer_type == ttnn.BufferType.DRAM
    ):
        raise ValueError(f"g6_zeros: memory_config must be DRAM interleaved, got {memory_config!r}")
    if not isinstance(device, ttnn.MeshDevice):
        raise ValueError(f"g6_zeros: device must be an open device or mesh device, got {device!r}")


def validate(shape, *, dtype, layout, device, memory_config):
    _check_structure(shape, layout, device, memory_config)

    scenario = {"shape": tuple(shape), "dtype": dtype, "layout": layout}
    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_zeros: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_zeros: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_zeros(shape, *, dtype, layout, device, memory_config):
    validate(shape, dtype=dtype, layout=layout, device=device, memory_config=memory_config)

    output = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dtype, layout, device, memory_config)
    program = get_program(output, device)
    return ttnn.generic_op([output], program)


def describe_plan(shape, *, dtype, layout, device, memory_config):
    """The plan a call with these arguments selects (regime, knobs, per-core roles); dispatches nothing."""
    validate(shape, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
    output = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dtype, layout, device, memory_config)
    try:
        return plan_for(output, device)
    finally:
        ttnn.deallocate(output)
