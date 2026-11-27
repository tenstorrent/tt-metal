# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_embedding: out[i, :] = W[idx[i], :], bit for bit, as a TILE tensor idx.shape + [D].

One device program per call: an index-driven row gather (reader), a UInt16 tilize (compute) and
whole-tile writes (writer), spread over the full grid by tile rows and, when the tile rows alone
cannot fill the grid, by column pieces as well.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_embedding_program_descriptor import (
    L1_BUDGET,
    TILE,
    build_cached_program,
    make_plan,
    plan_report,
    program_descriptor,
)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------
#
# inputs = (scenario,), scenario keys: index_shape, weight_shape, index_layout, memory_config.

_INDEX_LAYOUTS = {"ROW_MAJOR": ttnn.ROW_MAJOR_LAYOUT, "TILE": ttnn.TILE_LAYOUT}


def tag_alignment(inputs, axes):
    """Token extent (S of [B, S], N of [N]) a multiple of 32, or a partial last tile row per image."""
    return "tile_aligned" if inputs[0]["index_shape"][-1] % TILE == 0 else "h_non_aligned"


def tag_layout(inputs, axes):
    """Layout of the index tensor."""
    return _INDEX_LAYOUTS[inputs[0]["index_layout"]]


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "layout": tag_layout,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.ROW_MAJOR_LAYOUT],
    "alignment": ["tile_aligned", "h_non_aligned"],
    "output_layout": [ttnn.TILE_LAYOUT],
    "runtime_rows": [False],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------

_LAYOUT_NAMES = {ttnn.ROW_MAJOR_LAYOUT: "ROW_MAJOR", ttnn.TILE_LAYOUT: "TILE"}


def _is_dram_interleaved(memory_config):
    return (
        memory_config.buffer_type == ttnn.BufferType.DRAM
        and memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    )


def _memory_config_name(memory_config):
    if memory_config is None:
        return None
    if _is_dram_interleaved(memory_config):
        return "dram_interleaved"
    return f"{memory_config.buffer_type}/{memory_config.memory_layout}"


def _scenario(input_tensor, weight, memory_config):
    return {
        "index_shape": tuple(input_tensor.shape),
        "weight_shape": tuple(weight.shape),
        "index_layout": _LAYOUT_NAMES.get(input_tensor.layout, str(input_tensor.layout)),
        "memory_config": _memory_config_name(memory_config),
    }


def _check_requirements(input_tensor, weight, memory_config):
    index_rank = len(input_tensor.shape)
    if input_tensor.dtype != ttnn.uint32:
        raise ValueError(f"g6_embedding: input_tensor must be ttnn.uint32, got {input_tensor.dtype}")
    if index_rank not in (1, 2):
        raise ValueError(f"g6_embedding: input_tensor must have rank 1 or 2, got rank {index_rank}")
    weight_shape = tuple(weight.shape)
    if len(weight_shape) not in (2, 3):
        raise ValueError(f"g6_embedding: weight must be [V, D] or [1, V, D], got {weight_shape}")
    if len(weight_shape) == 3 and weight_shape[0] != 1:
        raise ValueError(f"g6_embedding: a rank-3 weight must have a leading extent of 1, got {weight_shape}")
    if weight.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"g6_embedding: weight must be ROW_MAJOR, got {weight.layout}")
    if weight_shape[-1] % TILE != 0:
        raise ValueError(f"g6_embedding: D = {weight_shape[-1]} must be a multiple of {TILE}")
    for name, tensor in (("input_tensor", input_tensor), ("weight", weight)):
        if tensor.storage_type() != ttnn.StorageType.DEVICE:
            raise ValueError(f"g6_embedding: {name} must be on the device (DRAM interleaved)")
        if not _is_dram_interleaved(tensor.memory_config()):
            raise ValueError(f"g6_embedding: {name} must be DRAM interleaved, got {tensor.memory_config()}")
    if memory_config is not None and not _is_dram_interleaved(memory_config):
        raise ValueError(f"g6_embedding: memory_config must be None or DRAM interleaved, got {memory_config}")


def validate(input_tensor, weight, *, layout=ttnn.TILE_LAYOUT, memory_config=None, runtime_rows=False):
    axes = {
        "dtype": weight.dtype,
        "output_layout": layout,
        "runtime_rows": bool(runtime_rows),
    }
    inputs = (_scenario(input_tensor, weight, memory_config),)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_embedding: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_embedding: unsupported combination (refinement candidate): {exc}")

    _check_requirements(input_tensor, weight, memory_config)


# ---------------------------------------------------------------------------
# Program cache (BB-07): one plan + descriptor per call signature, addresses rewritten per call.
# ---------------------------------------------------------------------------

_PROGRAM_CACHE = {}


def _device_key(device):
    try:
        return device.id()
    except (AttributeError, TypeError):
        return id(device)


def _cache_key(input_tensor, weight, layout, memory_config):
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    return (
        tuple(input_tensor.shape),
        input_tensor.layout,
        input_tensor.dtype,
        tuple(weight.shape),
        weight.dtype,
        layout,
        _memory_config_name(memory_config) or "dram_interleaved",
        _device_key(device),
        (grid.x, grid.y),
    )


def _plan_for(input_tensor, weight):
    grid = input_tensor.device().compute_with_storage_grid_size()
    return make_plan(tuple(input_tensor.shape), tuple(weight.shape), grid)


def describe_plan(input_tensor, weight, *, memory_config=None):
    """The plan g6_embedding selects for these inputs: regime, block knobs, cores, L1 per core."""
    key = _cache_key(input_tensor, weight, ttnn.TILE_LAYOUT, memory_config)
    cached = _PROGRAM_CACHE.get(key)
    plan = cached.plan if cached is not None else _plan_for(input_tensor, weight)
    return plan_report(plan)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_embedding(
    input_tensor: ttnn.Tensor,
    weight: ttnn.Tensor,
    *,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    memory_config: ttnn.MemoryConfig = None,
    runtime_rows: bool = False,
) -> ttnn.Tensor:
    validate(input_tensor, weight, layout=layout, memory_config=memory_config, runtime_rows=runtime_rows)

    device = input_tensor.device()
    output_shape = list(input_tensor.shape) + [weight.shape[-1]]
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        weight.dtype,
        ttnn.TILE_LAYOUT,
        device,
        memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
    )

    key = _cache_key(input_tensor, weight, layout, memory_config)
    cached = _PROGRAM_CACHE.get(key)
    if cached is None:
        plan = _plan_for(input_tensor, weight)
        cached = build_cached_program(plan, input_tensor, weight, output_tensor)
        _PROGRAM_CACHE[key] = cached

    descriptor = program_descriptor(cached, input_tensor, weight, output_tensor)
    return ttnn.generic_op([input_tensor, weight, output_tensor], descriptor)


__all__ = [
    "EXCLUSIONS",
    "INPUT_TAGGERS",
    "L1_BUDGET",
    "SUPPORTED",
    "describe_plan",
    "g6_embedding",
    "validate",
]
