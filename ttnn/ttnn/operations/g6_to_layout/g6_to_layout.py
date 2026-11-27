# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_to_layout: return input_tensor's logical values in the other layout, in one device program.

Phase 0: TILE -> ROW_MAJOR, bfloat16, interleaved DRAM/L1 in and out (regimes staged_reorder
and face_direct, see g6_to_layout_program_descriptor.py).
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_to_layout_program_descriptor import G6ToLayoutProgram, compute_plan


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
    if shape[-1] % 32 != 0:
        return "w_non_aligned"
    if shape[-2] % 32 != 0:
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


def _shard_spec_of(memory_config):
    spec = memory_config.shard_spec
    if spec is None:
        return None
    return {
        "grid": [((r.start.x, r.start.y), (r.end.x, r.end.y)) for r in spec.grid.ranges()],
        "shard_shape": tuple(spec.shape),
        "orientation": spec.orientation,
    }


def _scenario(input_tensor, layout, dtype, memory_config):
    tensor_memory = input_tensor.memory_config()
    return {
        "shape": tuple(input_tensor.shape),
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "memory_layout": tensor_memory.memory_layout,
        "buffer": tensor_memory.buffer_type,
        "shard_spec": _shard_spec_of(tensor_memory),
        "target_layout": layout,
        "out_dtype": dtype,
        "memory_config": memory_config,
    }


def _axes_of(scenario):
    axes = {}
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger((scenario,), axes)
    return axes


def validate(input_tensor, layout, *, dtype=None, memory_config=None):
    if layout not in (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT):
        raise ValueError(f"g6_to_layout: target layout {layout} is neither ROW_MAJOR_LAYOUT nor TILE_LAYOUT")
    if layout == input_tensor.layout:
        raise ValueError(f"g6_to_layout: target layout {layout} equals the input's layout")
    rank = len(input_tensor.shape)
    if rank not in (2, 3, 4):
        raise ValueError(f"g6_to_layout: input rank {rank} is outside 2 to 4")
    if dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("g6_to_layout: dtype bfloat8_b has no ROW_MAJOR_LAYOUT form")

    axes = _axes_of(_scenario(input_tensor, layout, dtype, memory_config))
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_to_layout: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exclusion in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exclusion.items()):
            raise ExcludedCell(f"g6_to_layout: unsupported combination (refinement candidate): {exclusion}")

    if memory_config is not None and memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError("g6_to_layout: memory_config names a sharded placement; the output is interleaved")
    if memory_config is None and input_tensor.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError("g6_to_layout: a sharded input needs an interleaved memory_config for the output")


# ---------------------------------------------------------------------------
# Plan + program cache (BB-07, BB-08)
# ---------------------------------------------------------------------------

_PLAN_CACHE = {}
_PROGRAM_CACHE = {}


def _output_dtype(input_tensor, layout, dtype):
    return tag_output_dtype(({"out_dtype": dtype, "dtype": input_tensor.dtype, "target_layout": layout},), {})


def _output_memory_config(input_tensor, memory_config):
    return memory_config if memory_config is not None else input_tensor.memory_config()


def _grid_of(device):
    grid = device.compute_with_storage_grid_size()
    return (int(grid.x), int(grid.y))


def _cache_key(input_tensor, layout, out_dtype, out_memory_config, grid):
    in_memory = input_tensor.memory_config()
    return (
        tuple(input_tensor.shape),
        input_tensor.dtype,
        input_tensor.layout,
        in_memory.memory_layout,
        in_memory.buffer_type,
        layout,
        out_dtype,
        out_memory_config.memory_layout,
        out_memory_config.buffer_type,
        grid,
    )


def _plan(input_tensor, layout, out_dtype, out_memory_config):
    device = input_tensor.device()
    grid = _grid_of(device)
    key = _cache_key(input_tensor, layout, out_dtype, out_memory_config, grid)
    cached = _PLAN_CACHE.get(key)
    if cached is None:
        output_in_l1 = out_memory_config.buffer_type == ttnn.BufferType.L1
        cached = compute_plan(
            tuple(input_tensor.shape),
            grid,
            output_in_l1=output_in_l1,
            num_l1_banks=ttnn.get_memory_view(device, ttnn.BufferType.L1).num_banks if output_in_l1 else 1,
            out_alignment=ttnn.get_l1_alignment() if output_in_l1 else ttnn.get_dram_alignment(),
        )
        _PLAN_CACHE[key] = cached
    return key, cached


def select_plan(input_tensor, layout, *, dtype=None, memory_config=None) -> dict:
    """The plan a call selects (BB-08), with every core's roles (BB-21); dispatches nothing."""
    validate(input_tensor, layout, dtype=dtype, memory_config=memory_config)
    out_dtype = _output_dtype(input_tensor, layout, dtype)
    out_memory_config = _output_memory_config(input_tensor, memory_config)
    _, selected = _plan(input_tensor, layout, out_dtype, out_memory_config)
    return selected.report()


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

    out_dtype = _output_dtype(input_tensor, layout, dtype)
    out_memory_config = _output_memory_config(input_tensor, memory_config)
    key, selected = _plan(input_tensor, layout, out_dtype, out_memory_config)

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        out_dtype,
        layout,
        input_tensor.device(),
        out_memory_config,
    )

    program = _PROGRAM_CACHE.get(key)
    if program is None:
        program = G6ToLayoutProgram(selected, input_tensor, output_tensor)
        _PROGRAM_CACHE[key] = program
    return ttnn.generic_op([input_tensor, output_tensor], program.descriptor_for(input_tensor, output_tensor))
