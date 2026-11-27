# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_embedding: out[i, :] = W[idx[i], :] as a TILE tensor of shape idx.shape + [D], bit for bit.

One device program per call. The index values are read on the device only: no value reaches the
host, the program arguments or the cache key, so a replayed trace picks the rows of its own indices.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_embedding_program_descriptor import (
    L1_BUDGET_BYTES,
    TILE_WIDTH,
    create_program_artifacts,
    select_plan,
)

_LAYOUT_BY_NAME = {"ROW_MAJOR": ttnn.ROW_MAJOR_LAYOUT, "TILE": ttnn.TILE_LAYOUT}


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — read the (scenario,) tuple feature_spec.py documents
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    token_extent = inputs[0]["index_shape"][-1]
    return "tile_aligned" if token_extent % 32 == 0 else "h_non_aligned"


def tag_layout(inputs, axes):
    return _LAYOUT_BY_NAME[inputs[0]["index_layout"]]


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


def _is_dram_interleaved(memory_config):
    return (
        memory_config.buffer_type == ttnn.BufferType.DRAM
        and memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    )


def _check_tensors(input_tensor, weight, memory_config):
    """Structural refusals (ValueError): outside the op's contract, not refinement candidates."""
    for name, tensor in (("input_tensor", input_tensor), ("weight", weight)):
        if not ttnn.is_tensor_storage_on_device(tensor):
            raise ValueError(f"g6_embedding: {name} must be on the device")
        if not _is_dram_interleaved(tensor.memory_config()):
            raise ValueError(f"g6_embedding: {name} must be DRAM interleaved, got {tensor.memory_config()}")
    if len(input_tensor.shape) not in (1, 2):
        raise ValueError(f"g6_embedding: input_tensor must have rank 1 or 2, got shape {tuple(input_tensor.shape)}")
    if input_tensor.dtype != ttnn.uint32:
        raise ValueError(f"g6_embedding: input_tensor must be uint32, got {input_tensor.dtype}")
    weight_shape = tuple(weight.shape)
    if len(weight_shape) not in (2, 3) or (len(weight_shape) == 3 and weight_shape[0] != 1):
        raise ValueError(f"g6_embedding: weight must be [V, D] or [1, V, D], got shape {weight_shape}")
    if weight.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"g6_embedding: weight must be ROW_MAJOR, got {weight.layout}")
    if weight_shape[-1] % TILE_WIDTH != 0:
        raise ValueError(f"g6_embedding: weight row length D={weight_shape[-1]} must be a multiple of {TILE_WIDTH}")
    if memory_config is not None and not _is_dram_interleaved(memory_config):
        raise ValueError(f"g6_embedding: memory_config must be DRAM interleaved, got {memory_config}")


def validate(input_tensor, weight, *, layout=ttnn.TILE_LAYOUT, memory_config=None, runtime_rows=False):
    _check_tensors(input_tensor, weight, memory_config)

    scenario = {
        "index_shape": tuple(input_tensor.shape),
        "weight_shape": tuple(weight.shape),
        "index_layout": "TILE" if input_tensor.layout == ttnn.TILE_LAYOUT else "ROW_MAJOR",
        "memory_config": None if memory_config is None else "dram_interleaved",
    }
    axes = {
        "dtype": weight.dtype,
        "output_layout": layout,
        "runtime_rows": bool(runtime_rows),
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_embedding: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for excluded in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in excluded.items()):
            raise ExcludedCell(f"g6_embedding: unsupported combination (refinement candidate): {excluded}")

    plan = select_plan(input_tensor, weight)
    if plan.l1_bytes_per_core > L1_BUDGET_BYTES:
        raise ValueError(
            f"g6_embedding: plan needs {plan.l1_bytes_per_core} B of L1 per core, above the "
            f"{L1_BUDGET_BYTES} B bound"
        )
    return plan


def plan_of(input_tensor, weight):
    """The plan a call with these tensors selects (BB-08)."""
    return select_plan(input_tensor, weight)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# (index shape, index layout, weight shape, weight dtype, grid) -> (spec, run_args, tensor_indices).
# validate() pins every tensor to DRAM interleaved and the index dtype to uint32, so these fields
# determine the program completely; no buffer address or index value is part of an entry.
_PROGRAM_CACHE = {}


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

    key = (
        tuple(input_tensor.shape),
        input_tensor.layout,
        tuple(weight.shape),
        weight.dtype,
        plan.grid,
    )
    artifacts = _PROGRAM_CACHE.get(key)
    if artifacts is None:
        artifacts = create_program_artifacts(plan, input_tensor, weight, output_tensor)
        _PROGRAM_CACHE[key] = artifacts
    spec, run_args, tensor_indices = artifacts
    return ttnn.generic_op([input_tensor, weight, output_tensor], spec, run_args, tensor_indices)
