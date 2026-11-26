# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_to_layout: TILE <-> ROW_MAJOR layout conversion as ONE device program.

Phase 0 builds regime R1 `interleaved_untilize_2d` (op_design.md, Blocking Model):
bfloat16 TILE -> ROW_MAJOR, interleaved DRAM or L1, every alignment, rank 2-4.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_to_layout_program_descriptor import create_program_descriptor, plan_blocking


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — every axis is derived from the scenario dict `(scenario,)`
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


def tag_output_dtype(inputs, axes):
    s = inputs[0]
    if s.get("out_dtype") is not None:
        return s["out_dtype"]
    if s["dtype"] == ttnn.bfloat8_b and s["target_layout"] == ttnn.ROW_MAJOR_LAYOUT:
        return ttnn.bfloat16
    return s["dtype"]


def tag_memory_layout(inputs, axes):
    return inputs[0]["memory_layout"]


def tag_alignment(inputs, axes):
    shape = inputs[0]["shape"]
    H, W = shape[-2], shape[-1]
    if W % 32:
        return "w_non_aligned"
    if H % 32:
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
# 2. SUPPORTED (Phase 0: R1 only)
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
# 3. EXCLUSIONS (Phase 0: none)
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _scenario(input_tensor, layout, dtype, memory_config):
    mem = input_tensor.memory_config()
    shard_spec = None
    if mem.shard_spec is not None:
        spec = mem.shard_spec
        shard_spec = {"shard_shape": tuple(spec.shape), "orientation": spec.orientation}
    return {
        "shape": tuple(input_tensor.shape),
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "memory_layout": mem.memory_layout,
        "buffer": mem.buffer_type,
        "shard_spec": shard_spec,
        "target_layout": layout,
        "out_dtype": dtype,
        "memory_config": memory_config,
    }


def validate(input_tensor, layout, *, dtype=None, memory_config=None):
    # 1. layout must be the other one
    if layout not in (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT) or layout == input_tensor.layout:
        raise ValueError(f"g6_to_layout: target layout {layout} must differ from the input's {input_tensor.layout}")
    # 2. rank
    rank = len(input_tensor.shape)
    if rank not in (2, 3, 4):
        raise ValueError(f"g6_to_layout: rank must be 2, 3 or 4, got {rank}")
    # 3. bfloat8_b row-major is not a format
    if dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("g6_to_layout: bfloat8_b requires TILE_LAYOUT")

    scenario = _scenario(input_tensor, layout, dtype, memory_config)
    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

    # 4. SUPPORTED per axis
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_to_layout: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    # 5. EXCLUSIONS
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_to_layout: unsupported combination (refinement candidate): {exc}")
    # 6. sharded output memory config
    if memory_config is not None:
        if memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            raise ValueError("g6_to_layout: output memory_config must be interleaved")
    elif input_tensor.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError("g6_to_layout: a sharded input needs an explicit interleaved memory_config")
    return axes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _output_dtype(input_tensor, layout, dtype):
    if dtype is not None:
        return dtype
    if input_tensor.dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        return ttnn.bfloat16
    return input_tensor.dtype


def _output_memory_config(input_tensor, memory_config):
    if memory_config is not None:
        return memory_config
    buffer_type = input_tensor.memory_config().buffer_type
    return ttnn.L1_MEMORY_CONFIG if buffer_type == ttnn.BufferType.L1 else ttnn.DRAM_MEMORY_CONFIG


def plan_for(input_tensor, layout, *, dtype=None, memory_config=None):
    """Return the blocking plan (regime, block extents, chunks, units, core ranges) of a call."""
    validate(input_tensor, layout, dtype=dtype, memory_config=memory_config)
    out_mem = _output_memory_config(input_tensor, memory_config)
    return plan_blocking(input_tensor.device(), tuple(input_tensor.shape), out_mem.buffer_type)


def g6_to_layout(input_tensor, layout, *, dtype=None, memory_config=None):
    validate(input_tensor, layout, dtype=dtype, memory_config=memory_config)

    out_dtype = _output_dtype(input_tensor, layout, dtype)
    out_mem = _output_memory_config(input_tensor, memory_config)
    device = input_tensor.device()

    # CRITICAL: allocate_tensor_on_device requires POSITIONAL args
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        out_dtype,
        layout,
        device,
        out_mem,
    )
    program_descriptor = create_program_descriptor(input_tensor, output_tensor)
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
