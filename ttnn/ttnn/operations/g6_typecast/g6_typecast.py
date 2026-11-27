# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_typecast: convert every value of one interleaved tensor to `dtype` in one device program,
keeping shape, layout and memory placement (registry model, see eval/op_template.py)."""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_typecast_program_descriptor import CORE_ROLES, build_program, make_plan


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — every axis is projected from the scenario dict (inputs = (scenario,))
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_output_dtype(inputs, axes):
    return inputs[0]["output_dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


def tag_buffer(inputs, axes):
    return inputs[0]["buffer"]


def tag_alignment(inputs, axes):
    shape = inputs[0]["input_shape"]
    W = shape[-1]
    H = shape[-2] if len(shape) >= 2 else 1
    if W % 32 != 0:
        return "w_non_aligned"
    if H % 32 != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    return len(inputs[0]["input_shape"])


INPUT_TAGGERS = {
    "dtype": tag_dtype,
    "output_dtype": tag_output_dtype,
    "layout": tag_layout,
    "buffer": tag_buffer,
    "alignment": tag_alignment,
    "rank": tag_rank,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED — Phase 0: bfloat16 TILE -> float32 / bfloat8_b / int32, DRAM or L1, ranks 3-4
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "output_dtype": [ttnn.float32, ttnn.bfloat8_b, ttnn.int32],
    "layout": [ttnn.TILE_LAYOUT],
    "buffer": ["dram", "l1"],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank": [3, 4],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS — stays empty: every TARGET cell is a production case
# ---------------------------------------------------------------------------

EXCLUSIONS = []


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _scenario(input_tensor, dtype):
    buffer_type = input_tensor.memory_config().buffer_type
    return {
        "input_shape": tuple(input_tensor.shape),
        "dtype": input_tensor.dtype,
        "output_dtype": dtype,
        "layout": input_tensor.layout,
        "buffer": "l1" if buffer_type == ttnn.BufferType.L1 else "dram",
    }


def validate(input_tensor, dtype, *, sub_core_grids=None):
    # Contract errors first; they are not support refusals.
    if not isinstance(dtype, ttnn.DataType):
        raise ValueError(f"g6_typecast: dtype must be a ttnn.DataType, got {type(dtype).__name__}")
    if sub_core_grids is not None:
        raise ValueError("g6_typecast: sub_core_grids must be None")
    rank = len(input_tensor.shape)
    if not 1 <= rank <= 4:
        raise ValueError(f"g6_typecast: rank must be 1 to 4, got rank {rank}")
    if input_tensor.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError("g6_typecast: input_tensor must be interleaved (DRAM or L1)")

    scenario = _scenario(input_tensor, dtype)
    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_typecast: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_typecast: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Plan reporting and program cache
# ---------------------------------------------------------------------------


def describe_plan(input_tensor, dtype):
    """The plan a call selects, without dispatching (BB-08, BB-21)."""
    plan = make_plan(input_tensor, dtype)
    role = CORE_ROLES[plan.regime]
    return {
        "regime": plan.regime,
        "num_cores": plan.num_cores,
        "block_tiles": plan.block_tiles,
        "core_roles": {(core.x, core.y): role for core, _, _ in plan.assignment},
    }


# Everything that changes the program is in the key; buffer addresses are patched per call (BB-07).
_PROGRAM_CACHE = {}


def _cache_key(input_tensor, dtype):
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    memory_config = input_tensor.memory_config()
    return (
        device.id(),
        tuple(input_tensor.shape),
        tuple(input_tensor.padded_shape),
        tuple(input_tensor.tile.tile_shape),
        input_tensor.dtype,
        dtype,
        input_tensor.layout,
        memory_config.memory_layout,
        memory_config.buffer_type,
        (grid.x, grid.y),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_typecast(input_tensor: ttnn.Tensor, dtype: ttnn.DataType, *, sub_core_grids=None) -> ttnn.Tensor:
    validate(input_tensor, dtype, sub_core_grids=sub_core_grids)

    output_tensor = ttnn.allocate_tensor_on_device(
        input_tensor.shape,
        dtype,
        input_tensor.layout,
        input_tensor.device(),
        input_tensor.memory_config(),
    )

    key = _cache_key(input_tensor, dtype)
    program = _PROGRAM_CACHE.get(key)
    if program is None:
        program = build_program(make_plan(input_tensor, dtype), input_tensor, output_tensor)
        _PROGRAM_CACHE[key] = program

    return ttnn.generic_op([input_tensor, output_tensor], program.descriptor(input_tensor, output_tensor))
