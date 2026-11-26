# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_typecast — convert every value of one interleaved tensor to `dtype` in one device program.

Registry model (eval/op_template.py): INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate().
INVALID lives in eval/golden_tests/g6_typecast/feature_spec.py, not here.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_typecast_program_descriptor import (
    create_program_descriptor,
    patch_addresses,
    plan_work,
    select_regime,
)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — `inputs` is (scenario,), the dict validate() builds.
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
    w = shape[-1]
    h = shape[-2] if len(shape) >= 2 else 1
    if w % 32 != 0:
        return "w_non_aligned"
    if h % 32 != 0:
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
# 2. SUPPORTED — Phase 0: bfloat16 TILE -> float32 / bfloat8_b / int32.
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
# 3. EXCLUSIONS — every TARGET cell is a production case; stays empty.
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _buffer_name(tensor):
    return "l1" if tensor.memory_config().buffer_type == ttnn.BufferType.L1 else "dram"


def _scenario(input_tensor, dtype):
    return {
        "input_shape": tuple(input_tensor.shape),
        "dtype": input_tensor.dtype,
        "output_dtype": dtype,
        "layout": input_tensor.layout,
        "buffer": _buffer_name(input_tensor),
    }


def validate(input_tensor, dtype, *, sub_core_grids=None):
    # Contract errors first; each message names the offending argument.
    if not isinstance(dtype, ttnn.DataType):
        raise ValueError(f"g6_typecast: dtype must be a ttnn.DataType, got {type(dtype).__name__} {dtype!r}")
    if sub_core_grids is not None:
        raise ValueError("g6_typecast: sub_core_grids must be None")
    rank = len(input_tensor.shape)
    if not 1 <= rank <= 4:
        raise ValueError(f"g6_typecast: input rank must be 1 to 4, got rank {rank}")

    scenario = _scenario(input_tensor, dtype)
    axes = {}
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_typecast: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_typecast: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Plan reporting and program cache
# ---------------------------------------------------------------------------

_CORE_ROLE = {"tile_convert": "reader+compute+writer"}


def describe_plan(input_tensor, dtype):
    """The plan a call selects, without dispatching."""
    regime = select_regime(input_tensor.layout, input_tensor.dtype, dtype)
    plan = plan_work(input_tensor.device(), input_tensor.buffer_num_pages())
    return {
        "regime": regime,
        "num_cores": plan["num_cores"],
        "block_tiles": plan["block_tiles"],
        "core_roles": {(core.x, core.y): _CORE_ROLE[regime] for core, _, _ in plan["assignment"]},
    }


# Built descriptors keyed by everything that changes the program; a hit patches only the two
# buffer addresses (common RT args), which the device-side program hash excludes.
_DESCRIPTOR_CACHE = {}


def _cache_key(input_tensor, dtype, device):
    grid = device.compute_with_storage_grid_size()
    return (
        device.id(),
        tuple(input_tensor.shape),
        tuple(input_tensor.padded_shape),
        input_tensor.dtype,
        dtype,
        input_tensor.layout,
        input_tensor.memory_config(),
        (grid.x, grid.y),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_typecast(input_tensor, dtype, *, sub_core_grids=None):
    validate(input_tensor, dtype, sub_core_grids=sub_core_grids)

    device = input_tensor.device()
    output_tensor = ttnn.allocate_tensor_on_device(
        input_tensor.shape, dtype, input_tensor.layout, device, input_tensor.memory_config()
    )

    key = _cache_key(input_tensor, dtype, device)
    descriptor = _DESCRIPTOR_CACHE.get(key)
    if descriptor is None:
        select_regime(input_tensor.layout, input_tensor.dtype, dtype)
        plan = plan_work(device, input_tensor.buffer_num_pages())
        descriptor = create_program_descriptor(input_tensor, output_tensor, plan)
        _DESCRIPTOR_CACHE[key] = descriptor
    else:
        patch_addresses(descriptor, input_tensor, output_tensor)

    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
