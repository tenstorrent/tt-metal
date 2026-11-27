# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_embedding: out[i, :] = W[idx[i], :], returned as a TILE tensor of shape idx.shape + [D].

One device program per call. The table rows are chosen from the index values on the device; no
index value reaches the host, the program arguments or the cache key.
"""

from __future__ import annotations

import os

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_embedding_program_descriptor import (
    L1_BUDGET_BYTES,
    TILE_DIM,
    Plan,
    core_roles,
    create_program_artifacts,
    select_plan,
)

OP_NAME = "g6_embedding"

_LAYOUT_OF_NAME = {"ROW_MAJOR": ttnn.ROW_MAJOR_LAYOUT, "TILE": ttnn.TILE_LAYOUT}
_NAME_OF_LAYOUT = {layout: name for name, layout in _LAYOUT_OF_NAME.items()}


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — over the (scenario,) tuple of feature_spec.py
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    return "tile_aligned" if inputs[0]["index_shape"][-1] % TILE_DIM == 0 else "h_non_aligned"


def tag_layout(inputs, axes):
    return _LAYOUT_OF_NAME[inputs[0]["index_layout"]]


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


def _is_dram_interleaved(memory_config) -> bool:
    return (
        memory_config.buffer_type == ttnn.BufferType.DRAM
        and memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    )


def _scenario(input_tensor, weight, memory_config) -> dict:
    if memory_config is None:
        placement = None
    elif _is_dram_interleaved(memory_config):
        placement = "dram_interleaved"
    else:
        placement = str(memory_config)
    return {
        "index_shape": tuple(input_tensor.shape),
        "weight_shape": tuple(weight.shape),
        "index_layout": _NAME_OF_LAYOUT.get(input_tensor.layout, str(input_tensor.layout)),
        "memory_config": placement,
    }


def _require_dram_interleaved_on_device(tensor, name):
    if not ttnn.is_tensor_storage_on_device(tensor):
        raise ValueError(f"{OP_NAME}: {name} must be on the device")
    if not _is_dram_interleaved(tensor.memory_config()):
        raise ValueError(f"{OP_NAME}: {name} must be DRAM interleaved, got {tensor.memory_config()}")


def validate(input_tensor, weight, *, layout=ttnn.TILE_LAYOUT, memory_config=None, runtime_rows=False) -> Plan:
    inputs = (_scenario(input_tensor, weight, memory_config),)
    axes = {
        "dtype": weight.dtype,
        "output_layout": layout,
        "runtime_rows": bool(runtime_rows),
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"{OP_NAME}: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for excluded in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in excluded.items()):
            raise ExcludedCell(f"{OP_NAME}: unsupported combination (refinement candidate): {excluded}")

    index_shape = tuple(input_tensor.shape)
    weight_shape = tuple(weight.shape)
    if len(index_shape) not in (1, 2):
        raise ValueError(f"{OP_NAME}: input_tensor must have rank 1 or 2, got shape {index_shape}")
    if input_tensor.dtype != ttnn.uint32:
        raise ValueError(f"{OP_NAME}: input_tensor must be uint32, got {input_tensor.dtype}")
    if len(weight_shape) not in (2, 3):
        raise ValueError(f"{OP_NAME}: weight must have rank 2 or 3, got shape {weight_shape}")
    if len(weight_shape) == 3 and weight_shape[0] != 1:
        raise ValueError(f"{OP_NAME}: a rank-3 weight must have leading extent 1, got shape {weight_shape}")
    if weight.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"{OP_NAME}: weight must be ROW_MAJOR, got {weight.layout}")
    if weight_shape[-1] % TILE_DIM != 0:
        raise ValueError(f"{OP_NAME}: weight row length D={weight_shape[-1]} must be a multiple of {TILE_DIM}")
    _require_dram_interleaved_on_device(input_tensor, "input_tensor")
    _require_dram_interleaved_on_device(weight, "weight")
    if memory_config is not None and not _is_dram_interleaved(memory_config):
        raise ValueError(f"{OP_NAME}: memory_config must be DRAM interleaved, got {memory_config}")

    plan = select_plan(input_tensor, weight)
    if plan.l1_bytes_per_core > L1_BUDGET_BYTES:
        raise ValueError(
            f"{OP_NAME}: the plan needs {plan.l1_bytes_per_core} B of L1 per core, above the bound of "
            f"{L1_BUDGET_BYTES} B per core"
        )
    return plan


# ---------------------------------------------------------------------------
# Program cache (BB-07): the artifacts hold no buffer address and no index value.
# ---------------------------------------------------------------------------

_PROGRAM_ARTIFACTS = {}


def _zones_enabled() -> bool:
    """Zone build (BB-21): G6_EMBEDDING_ZONES=1 adds a profiler zone per block operation."""
    return os.environ.get("G6_EMBEDDING_ZONES", "0") not in ("", "0")


def _cache_key(input_tensor, weight, grid, zones):
    return (
        tuple(input_tensor.shape),
        input_tensor.layout,
        input_tensor.dtype,
        tuple(weight.shape),
        weight.layout,
        weight.dtype,
        grid.x,
        grid.y,
        zones,
    )


def dump_core_roles(input_tensor: ttnn.Tensor, weight: ttnn.Tensor) -> dict:
    """Roles and block run of every core the call's program runs on (BB-21)."""
    grid = input_tensor.device().compute_with_storage_grid_size()
    return core_roles(select_plan(input_tensor, weight), grid)


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
    plan = validate(input_tensor, weight, layout=layout, memory_config=memory_config, runtime_rows=runtime_rows)

    device = input_tensor.device()
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape) + [weight.shape[-1]]),
        weight.dtype,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    grid = device.compute_with_storage_grid_size()
    zones = _zones_enabled()
    key = _cache_key(input_tensor, weight, grid, zones)
    artifacts = _PROGRAM_ARTIFACTS.get(key)
    if artifacts is None:
        artifacts = create_program_artifacts(input_tensor, weight, output_tensor, plan, zones=zones)
        _PROGRAM_ARTIFACTS[key] = artifacts
    spec, run_args, tensor_indices = artifacts

    # generic_op returns io_tensors[-1]: the output goes last.
    return ttnn.generic_op([input_tensor, weight, output_tensor], spec, run_args, tensor_indices)
