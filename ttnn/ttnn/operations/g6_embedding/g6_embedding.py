# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_embedding: out[i, d] = W[idx[i], d], gathered on the device, returned as a bf16 TILE tensor.

One device program per call. The index values never reach the host: the reader picks the table rows
from them on the device, and only the named rows are read (no pass over the table).
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_embedding_program_descriptor import (
    L1_BUDGET,
    TILE_DIM,
    compute_plan,
    create_program_descriptor,
    plan_report,
)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — inputs is (scenario,), the dict feature_spec.py describes
# ---------------------------------------------------------------------------

_LAYOUT_BY_NAME = {"ROW_MAJOR": ttnn.ROW_MAJOR_LAYOUT, "TILE": ttnn.TILE_LAYOUT}
_NAME_BY_LAYOUT = {ttnn.ROW_MAJOR_LAYOUT: "ROW_MAJOR", ttnn.TILE_LAYOUT: "TILE"}


def tag_alignment(inputs, axes):
    """Token extent (last index dim) a multiple of 32, or a partial last tile row per image."""
    return "tile_aligned" if inputs[0]["index_shape"][-1] % TILE_DIM == 0 else "h_non_aligned"


def tag_layout(inputs, axes):
    """Layout of the index tensor."""
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


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "bounded_cb": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _is_dram_interleaved(memory_config) -> bool:
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
        "index_layout": _NAME_BY_LAYOUT.get(input_tensor.layout, str(input_tensor.layout)),
        "memory_config": _memory_config_name(memory_config),
    }


def _check_on_device_dram(name, tensor):
    if not ttnn.is_tensor_storage_on_device(tensor):
        raise ValueError(f"g6_embedding: {name} must be on the device (DRAM interleaved)")
    if not _is_dram_interleaved(tensor.memory_config()):
        raise ValueError(f"g6_embedding: {name} must be DRAM interleaved, got {tensor.memory_config()}")


def validate(input_tensor, weight, *, layout=ttnn.TILE_LAYOUT, memory_config=None, runtime_rows=False):
    if len(input_tensor.shape) not in (1, 2):
        raise ValueError(f"g6_embedding: input_tensor must have rank 1 or 2, got shape {tuple(input_tensor.shape)}")

    scenario = _scenario(input_tensor, weight, memory_config)
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

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_embedding: unsupported combination (refinement candidate): {exc}")

    if input_tensor.dtype != ttnn.uint32:
        raise ValueError(f"g6_embedding: input_tensor must be ttnn.uint32, got {input_tensor.dtype}")
    if any(int(e) == 0 for e in input_tensor.shape):
        raise ValueError(f"g6_embedding: input_tensor has an empty extent {tuple(input_tensor.shape)}")

    weight_shape = tuple(weight.shape)
    if len(weight_shape) not in (2, 3):
        raise ValueError(f"g6_embedding: weight must have rank 2 or 3, got shape {weight_shape}")
    if len(weight_shape) == 3 and weight_shape[0] != 1:
        raise ValueError(f"g6_embedding: a rank-3 weight must have a leading extent of 1, got {weight_shape}")
    if weight.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"g6_embedding: weight must be ROW_MAJOR, got {weight.layout}")
    if weight_shape[-1] % TILE_DIM != 0:
        raise ValueError(f"g6_embedding: D={weight_shape[-1]} must be a multiple of {TILE_DIM}")

    _check_on_device_dram("input_tensor", input_tensor)
    _check_on_device_dram("weight", weight)
    if memory_config is not None and not _is_dram_interleaved(memory_config):
        raise ValueError(f"g6_embedding: memory_config must be None or DRAM interleaved, got {memory_config}")

    # Raises a ValueError naming the BB-05 bound when the plan does not fit.
    _plan_for(input_tensor, weight)


# ---------------------------------------------------------------------------
# Plan and entry point
# ---------------------------------------------------------------------------


def _grid(device):
    grid = device.compute_with_storage_grid_size()
    return grid.x, grid.y


def _plan_for(input_tensor, weight):
    grid_x, grid_y = _grid(input_tensor.device())
    return compute_plan(tuple(input_tensor.shape), tuple(weight.shape), grid_x, grid_y)


def _device_key(device):
    try:
        return device.id()
    except (AttributeError, TypeError):
        return id(device)


def describe_plan(input_tensor, weight, *, memory_config=None) -> dict:
    """The plan a call with these inputs selects (regime, blocks, cores, L1 per core) — BB-08."""
    if memory_config is not None and not _is_dram_interleaved(memory_config):
        raise ValueError(f"g6_embedding: memory_config must be None or DRAM interleaved, got {memory_config}")
    report = plan_report(_plan_for(input_tensor, weight))
    report["l1_budget"] = L1_BUDGET
    return report


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
    plan = _plan_for(input_tensor, weight)
    index_shape = tuple(input_tensor.shape)
    weight_shape = tuple(weight.shape)
    width = weight_shape[-1]

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(index_shape) + [width]),
        weight.dtype,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    cache_key = (
        index_shape,
        input_tensor.layout,
        weight_shape,
        weight.dtype,
        layout,
        "dram_interleaved",
        _device_key(device),
        _grid(device),
    )
    program_descriptor = create_program_descriptor(cache_key, plan, input_tensor, weight, output_tensor)
    return ttnn.generic_op([input_tensor, weight, output_tensor], program_descriptor)
