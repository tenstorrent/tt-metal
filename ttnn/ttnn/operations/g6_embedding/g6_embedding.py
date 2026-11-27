# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_embedding: out[i, :] = W[idx[i], :], gathered on the device, returned as TILE pages.

One device program per call: the reader gathers the table rows the indices name into a row-major
staging buffer, compute tilizes them as UInt16 (bit passthrough), the writer stores the tiles.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_embedding_program_descriptor import L1_BUDGET_BYTES, Plan, create_program_artifacts, select_plan

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — inputs is (scenario,), scenario keys as in feature_spec.py
# ---------------------------------------------------------------------------

_INDEX_LAYOUTS = {"ROW_MAJOR": ttnn.ROW_MAJOR_LAYOUT, "TILE": ttnn.TILE_LAYOUT}


def tag_layout(inputs, axes):
    """Layout of the index tensor."""
    return _INDEX_LAYOUTS[inputs[0]["index_layout"]]


def tag_alignment(inputs, axes):
    """Token extent (last index dim) a multiple of 32, or a partial last tile-row per image."""
    return "tile_aligned" if inputs[0]["index_shape"][-1] % 32 == 0 else "h_non_aligned"


INPUT_TAGGERS = {
    "layout": tag_layout,
    "alignment": tag_alignment,
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


def _is_dram_interleaved(memory_config) -> bool:
    return (
        memory_config.buffer_type == ttnn.BufferType.DRAM
        and memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    )


def _check_structure(input_tensor, weight, memory_config):
    for name, tensor in (("input_tensor", input_tensor), ("weight", weight)):
        if not ttnn.is_tensor_storage_on_device(tensor):
            raise ValueError(f"g6_embedding: {name} must be on the device")
        if not _is_dram_interleaved(tensor.memory_config()):
            raise ValueError(f"g6_embedding: {name} must be DRAM interleaved, got {tensor.memory_config()}")

    index_shape = tuple(input_tensor.shape)
    if len(index_shape) not in (1, 2):
        raise ValueError(f"g6_embedding: input_tensor must have rank 1 or 2, got shape {index_shape}")
    if input_tensor.dtype != ttnn.uint32:
        raise ValueError(f"g6_embedding: input_tensor must be ttnn.uint32, got {input_tensor.dtype}")

    weight_shape = tuple(weight.shape)
    if len(weight_shape) not in (2, 3):
        raise ValueError(f"g6_embedding: weight must have rank 2 or 3, got shape {weight_shape}")
    if len(weight_shape) == 3 and weight_shape[0] != 1:
        raise ValueError(f"g6_embedding: a rank-3 weight must be [1, V, D], got shape {weight_shape}")
    if weight.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"g6_embedding: weight must be ROW_MAJOR, got {weight.layout}")
    if weight_shape[-1] % 32 != 0:
        raise ValueError(f"g6_embedding: weight D={weight_shape[-1]} must be a multiple of 32")

    if memory_config is not None and not _is_dram_interleaved(memory_config):
        raise ValueError(f"g6_embedding: memory_config must be None or DRAM interleaved, got {memory_config}")


def _scenario(input_tensor, weight, memory_config):
    return (
        {
            "index_shape": tuple(input_tensor.shape),
            "weight_shape": tuple(weight.shape),
            "index_layout": _LAYOUT_NAMES.get(input_tensor.layout, str(input_tensor.layout)),
            "memory_config": None if memory_config is None else "dram_interleaved",
        },
    )


def validate(input_tensor, weight, *, layout=ttnn.TILE_LAYOUT, memory_config=None, runtime_rows=False):
    _check_structure(input_tensor, weight, memory_config)

    axes = {
        "dtype": weight.dtype,
        "output_layout": layout,
        "runtime_rows": bool(runtime_rows),
    }
    inputs = _scenario(input_tensor, weight, memory_config)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_embedding: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for excluded in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in excluded.items()):
            raise ExcludedCell(f"g6_embedding: unsupported combination (refinement candidate): {excluded}")

    plan = select_plan(input_tensor, weight)
    if plan.l1_bytes_per_core > L1_BUDGET_BYTES:
        raise ValueError(
            f"g6_embedding: plan needs {plan.l1_bytes_per_core} bytes of L1 per core, above the bound of "
            f"{L1_BUDGET_BYTES} bytes"
        )
    return plan


# ---------------------------------------------------------------------------
# Program cache (BB-07): artifacts hold no buffer address, so a hit rebuilds nothing.
# ---------------------------------------------------------------------------

_ARTIFACTS = {}


def _cache_key(input_tensor, weight, plan: Plan):
    return (
        tuple(input_tensor.shape),
        input_tensor.layout,
        input_tensor.dtype,
        tuple(weight.shape),
        weight.dtype,
        plan.grid_x,
        plan.grid_y,
    )


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
    output_shape = list(input_tensor.shape) + [int(weight.shape[-1])]
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        weight.dtype,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    key = _cache_key(input_tensor, weight, plan)
    artifacts = _ARTIFACTS.get(key)
    if artifacts is None:
        artifacts = create_program_artifacts(plan, input_tensor, weight, output_tensor)
        _ARTIFACTS[key] = artifacts
    spec, run_args, tensor_indices = artifacts

    return ttnn.generic_op([input_tensor, weight, output_tensor], spec, run_args, tensor_indices)
