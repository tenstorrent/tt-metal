# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_typecast: convert every value of one tensor to `dtype` in one device program.

The output keeps the input's shape, layout and interleaved memory config. Registry model:
INPUT_TAGGERS / SUPPORTED / EXCLUSIONS / validate() below drive both the runtime gate and the
golden harness.
"""

from __future__ import annotations

import hashlib

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_typecast_program_descriptor import Knobs, _plan, create_program_descriptor, patch_addresses


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — every axis is projected from the scenario dict
#    {"input_shape", "dtype", "output_dtype", "layout", "buffer"}; inputs = (scenario,).
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
    if W % 32:
        return "w_non_aligned"
    if H % 32:
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
# 2. SUPPORTED
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
# 3. EXCLUSIONS — every TARGET cell is a production case, so this stays empty.
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _scenario(input_tensor, dtype):
    is_l1 = input_tensor.memory_config().buffer_type == ttnn.BufferType.L1
    return {
        "input_shape": tuple(input_tensor.shape),
        "dtype": input_tensor.dtype,
        "output_dtype": dtype,
        "layout": input_tensor.layout,
        "buffer": "l1" if is_l1 else "dram",
    }


def validate(input_tensor, dtype, *, sub_core_grids=None):
    if not isinstance(dtype, ttnn.DataType):
        raise ValueError(f"g6_typecast: dtype must be a ttnn.DataType, got {type(dtype).__name__}")
    if sub_core_grids is not None:
        raise ValueError("g6_typecast: sub_core_grids must be None")
    rank = len(input_tensor.shape)
    if not 1 <= rank <= 4:
        raise ValueError(f"g6_typecast: input rank must be 1..4, got {rank}")
    if input_tensor.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError("g6_typecast: input must be interleaved")

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
# Program cache (BB-07): one Plan + ProgramDescriptor per key; a hit only rewrites the
# input/output buffer addresses.
# ---------------------------------------------------------------------------

_CACHE = {}


def _cache_key(input_tensor, dtype, knobs):
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    return (
        tuple(input_tensor.shape),
        tuple(input_tensor.padded_shape),
        input_tensor.dtype,
        dtype,
        input_tensor.layout,
        input_tensor.memory_config().buffer_type,
        device.id(),
        (grid.x, grid.y),
        knobs,
    )


def _program_hash(key):
    digest = hashlib.blake2b(repr(("g6_typecast",) + key).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _lookup(input_tensor, output_tensor, dtype, knobs):
    key = _cache_key(input_tensor, dtype, knobs)
    entry = _CACHE.get(key)
    if entry is None:
        plan = _plan(input_tensor, dtype, knobs)
        descriptor = create_program_descriptor(input_tensor, output_tensor, plan)
        descriptor.custom_program_hash = _program_hash(key)
        entry = (plan, descriptor)
        _CACHE[key] = entry
    else:
        plan, descriptor = entry
        patch_addresses(descriptor, plan, input_tensor, output_tensor)
    return entry


def describe_plan(input_tensor, dtype, *, knobs: Knobs = Knobs()):
    """Report the plan a call with these arguments selects (BB-08, BB-21)."""
    validate(input_tensor, dtype)
    plan = _plan(input_tensor, dtype, knobs)
    return {
        "regime": plan.regime,
        "num_cores": plan.num_cores,
        "core_ranges": str(plan.all_cores),
        "block_tiles": plan.block_tiles,
        "pipeline_depth": plan.pipeline_depth,
        "page_stride": plan.page_stride,
        "max_cores": plan.max_cores,
        "tensor_tiles": plan.tensor_tiles,
        "cb_input_tiles_bytes": plan.cb_input_bytes,
        "cb_output_tiles_bytes": plan.cb_output_bytes,
        "fp32_dest_acc_en": plan.fp32_dest_acc_en,
        "core_roles": {(w.x, w.y): ("reader", "compute", "writer") for w in plan.cores},
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _dispatch(input_tensor, dtype, knobs):
    output_tensor = ttnn.allocate_tensor_on_device(
        input_tensor.shape,
        dtype,
        input_tensor.layout,
        input_tensor.device(),
        input_tensor.memory_config(),
    )
    _, descriptor = _lookup(input_tensor, output_tensor, dtype, knobs)
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)


def g6_typecast_with_knobs(input_tensor, dtype, *, knobs: Knobs):
    """Perf-sweep entry: the same op with the blocking knobs overridden."""
    validate(input_tensor, dtype)
    return _dispatch(input_tensor, dtype, knobs)


def g6_typecast(input_tensor, dtype, *, sub_core_grids=None):
    validate(input_tensor, dtype, sub_core_grids=sub_core_grids)
    return _dispatch(input_tensor, dtype, Knobs())
