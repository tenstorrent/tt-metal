# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_to_layout: TILE <-> ROW_MAJOR layout conversion in one device program.

Phase 0 builds regime R1 (`untilize_interleaved`): a bf16 TILE tensor, DRAM or L1 interleaved,
to a bit-identical bf16 ROW_MAJOR interleaved tensor.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_to_layout_program_descriptor import (
    Plan,
    compute_plan,
    create_program_descriptor,
    patch_buffer_addresses,
)

OP_NAME = "g6_to_layout"


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — every axis is projected from the scenario dict `(scenario,)`
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


def tag_output_dtype(inputs, axes):
    scenario = inputs[0]
    if scenario["out_dtype"] is not None:
        return scenario["out_dtype"]
    if scenario["dtype"] == ttnn.bfloat8_b and scenario["target_layout"] == ttnn.ROW_MAJOR_LAYOUT:
        return ttnn.bfloat16
    return scenario["dtype"]


def tag_memory_layout(inputs, axes):
    return inputs[0]["memory_layout"]


def tag_alignment(inputs, axes):
    shape = inputs[0]["shape"]
    height, width = shape[-2], shape[-1]
    if width % 32:
        return "w_non_aligned"
    if height % 32:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    return len(inputs[0]["shape"])


INPUT_TAGGERS = {
    "dtype": tag_dtype,
    "layout": tag_layout,
    "output_dtype": tag_output_dtype,
    "memory_layout": tag_memory_layout,
    "alignment": tag_alignment,
    "rank": tag_rank,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "output_dtype": [ttnn.bfloat16],
    "memory_layout": [ttnn.TensorMemoryLayout.INTERLEAVED],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank": [2, 3, 4],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _coord(core_range, which):
    coord = getattr(core_range, f"{which}_coord", None) or getattr(core_range, which)
    return (coord.x, coord.y)


def _shard_spec_of(memory_config):
    spec = memory_config.shard_spec
    if spec is None:
        return None
    return {
        "grid": [(_coord(r, "start"), _coord(r, "end")) for r in spec.grid.ranges()],
        "shard_shape": tuple(spec.shape),
        "orientation": spec.orientation,
    }


def _scenario(input_tensor, layout, dtype, memory_config):
    input_memory = input_tensor.memory_config()
    return {
        "shape": tuple(input_tensor.shape),
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "memory_layout": input_memory.memory_layout,
        "buffer": input_memory.buffer_type,
        "shard_spec": _shard_spec_of(input_memory),
        "target_layout": layout,
        "out_dtype": dtype,
        "memory_config": memory_config,
    }


def validate(input_tensor, layout, *, dtype=None, memory_config=None):
    """Raise ValueError for malformed calls, then the registry refusals; returns the axes dict."""
    if layout not in (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT):
        raise ValueError(f"{OP_NAME}: layout must be ROW_MAJOR_LAYOUT or TILE_LAYOUT, got {layout}")
    if layout == input_tensor.layout:
        raise ValueError(f"{OP_NAME}: layout {layout} equals the input layout; nothing to convert")
    rank = len(input_tensor.shape)
    if not 2 <= rank <= 4:
        raise ValueError(f"{OP_NAME}: input rank must be 2 to 4, got {rank}")
    if dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"{OP_NAME}: bfloat8_b has no ROW_MAJOR form")

    scenario = _scenario(input_tensor, layout, dtype, memory_config)
    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"{OP_NAME}: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"{OP_NAME}: unsupported combination (refinement candidate): {exc}")

    if memory_config is not None and memory_config.is_sharded():
        raise ValueError(f"{OP_NAME}: memory_config must be interleaved, got {memory_config.memory_layout}")
    if memory_config is None and input_tensor.memory_config().is_sharded():
        raise ValueError(f"{OP_NAME}: a sharded input needs an interleaved memory_config for the output")
    return axes


# ---------------------------------------------------------------------------
# Plan + descriptor cache (BB-07 / BB-08)
# ---------------------------------------------------------------------------

_PLAN_CACHE = {}
_DESCRIPTOR_CACHE = {}


def _output_dtype(input_tensor, layout, dtype):
    if dtype is not None:
        return dtype
    if input_tensor.dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        return ttnn.bfloat16
    return input_tensor.dtype


def _output_memory_config(input_tensor, memory_config):
    if memory_config is not None:
        return memory_config
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, input_tensor.memory_config().buffer_type)


def _num_banks(device, buffer_type):
    return ttnn.get_memory_view(device, buffer_type).num_banks


def _cache_key(input_tensor, layout, dtype, memory_config):
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    output_memory = _output_memory_config(input_tensor, memory_config)
    return (
        tuple(input_tensor.shape),
        input_tensor.dtype,
        input_tensor.layout,
        repr(input_tensor.memory_config()),
        _output_dtype(input_tensor, layout, dtype),
        layout,
        repr(output_memory),
        (grid.x, grid.y),
    )


def select_plan(input_tensor, layout, *, dtype=None, memory_config=None) -> Plan:
    """The plan a call with these arguments runs (memoized; a pure function of the cache key)."""
    key = _cache_key(input_tensor, layout, dtype, memory_config)
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        device = input_tensor.device()
        grid = device.compute_with_storage_grid_size()
        output_memory = _output_memory_config(input_tensor, memory_config)
        plan = compute_plan(
            tuple(input_tensor.shape),
            input_tensor.dtype,
            _output_dtype(input_tensor, layout, dtype),
            output_memory,
            grid.x,
            grid.y,
            ttnn.get_memory_view(device, ttnn.BufferType.L1).num_banks,
            ttnn.get_l1_alignment(),
            _num_banks(device, input_tensor.memory_config().buffer_type),
            _num_banks(device, output_memory.buffer_type),
        )
        _PLAN_CACHE[key] = plan
    return plan


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_to_layout(
    input_tensor: ttnn.Tensor,
    layout: ttnn.Layout,
    *,
    dtype: ttnn.DataType = None,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    validate(input_tensor, layout, dtype=dtype, memory_config=memory_config)

    key = _cache_key(input_tensor, layout, dtype, memory_config)
    plan = select_plan(input_tensor, layout, dtype=dtype, memory_config=memory_config)

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        plan.output_dtype,
        layout,
        input_tensor.device(),
        plan.output_memory_config,
    )

    descriptor = _DESCRIPTOR_CACHE.get(key)
    if descriptor is None:
        descriptor = create_program_descriptor(plan, input_tensor, output_tensor)
        _DESCRIPTOR_CACHE[key] = descriptor
    else:
        patch_buffer_addresses(descriptor, input_tensor, output_tensor)

    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
