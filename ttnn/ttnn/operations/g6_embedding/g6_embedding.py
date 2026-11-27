# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_embedding: out[i, :] = W[idx[i], :] as a TILE tensor of shape idx.shape + [D].

One device program per call. The index values are read on the device only; the table is
read only at the rows the indices name; every value is copied bit for bit.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_embedding_program_descriptor import (
    L1_BOUND_BYTES,
    TILE_WIDTH,
    build_plan,
    create_program_descriptor,
    plan_report,
    set_buffer_addresses,
)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — inputs is (scenario,), see feature_spec.py
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    return "tile_aligned" if inputs[0]["index_shape"][-1] % 32 == 0 else "h_non_aligned"


_INDEX_LAYOUTS = {"ROW_MAJOR": ttnn.ROW_MAJOR_LAYOUT, "TILE": ttnn.TILE_LAYOUT}


def tag_layout(inputs, axes):
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
# Host-side plan and descriptor cache (BB-07, BB-08)
# ---------------------------------------------------------------------------

_LAYOUT_NAMES = {ttnn.ROW_MAJOR_LAYOUT: "ROW_MAJOR", ttnn.TILE_LAYOUT: "TILE"}

_PLANS = {}
_DESCRIPTORS = {}


def _is_dram_interleaved(memory_config) -> bool:
    return (
        memory_config.buffer_type == ttnn.BufferType.DRAM
        and memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    )


def _device_id(device):
    try:
        return device.id()
    except Exception:  # noqa: BLE001 — a mesh handle may not expose a single id
        return id(device)


def _cache_key(input_tensor, weight, device):
    grid = device.compute_with_storage_grid_size()
    return (
        tuple(input_tensor.shape),
        _LAYOUT_NAMES.get(input_tensor.layout, str(input_tensor.layout)),
        tuple(weight.shape),
        str(weight.dtype),
        "dram_interleaved",  # the only output placement of the contract
        (grid.x, grid.y),
        _device_id(device),
    )


def _plan(input_tensor, weight, device, cache_key):
    plan = _PLANS.get(cache_key)
    if plan is None:
        plan = build_plan(
            tuple(input_tensor.shape),
            tuple(weight.shape),
            weight.element_size(),
            device.compute_with_storage_grid_size(),
        )
        _PLANS[cache_key] = plan
    return plan


def describe_plan(input_tensor, weight, memory_config=None) -> dict:
    """The plan a call with these inputs selects (BB-08) and each core's role (BB-21)."""
    device = input_tensor.device()
    return plan_report(_plan(input_tensor, weight, device, _cache_key(input_tensor, weight, device)))


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _check_inputs(input_tensor, weight, memory_config):
    """Input requirements of the contract: ValueError."""
    for name, tensor in (("input_tensor", input_tensor), ("weight", weight)):
        if not isinstance(tensor, ttnn.Tensor):
            raise ValueError(f"g6_embedding: {name} must be a ttnn.Tensor")
        if not ttnn.is_tensor_storage_on_device(tensor):
            raise ValueError(f"g6_embedding: {name} must be on the device")
        if not _is_dram_interleaved(tensor.memory_config()):
            raise ValueError(f"g6_embedding: {name} must be DRAM interleaved")

    index_rank = len(input_tensor.shape)
    if index_rank not in (1, 2):
        raise ValueError(f"g6_embedding: input_tensor must have rank 1 or 2, got {index_rank}")
    if input_tensor.dtype != ttnn.uint32:
        raise ValueError(f"g6_embedding: input_tensor must be uint32, got {input_tensor.dtype}")

    weight_shape = tuple(weight.shape)
    if len(weight_shape) not in (2, 3):
        raise ValueError(f"g6_embedding: weight must have rank 2 or 3, got {len(weight_shape)}")
    if len(weight_shape) == 3 and weight_shape[0] != 1:
        raise ValueError(f"g6_embedding: a rank-3 weight must be [1, V, D], got {list(weight_shape)}")
    if weight.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"g6_embedding: weight must be ROW_MAJOR, got {weight.layout}")
    if weight_shape[-1] % TILE_WIDTH != 0:
        raise ValueError(f"g6_embedding: D = {weight_shape[-1]} must be a multiple of {TILE_WIDTH}")

    if memory_config is not None and not _is_dram_interleaved(memory_config):
        raise ValueError("g6_embedding: memory_config must be DRAM interleaved")


def validate(
    input_tensor,
    weight,
    *,
    layout=ttnn.TILE_LAYOUT,
    memory_config=None,
    runtime_rows=False,
):
    _check_inputs(input_tensor, weight, memory_config)

    device = input_tensor.device()
    plan = _plan(input_tensor, weight, device, _cache_key(input_tensor, weight, device))
    if plan.footprint_bytes > L1_BOUND_BYTES:
        raise ValueError(
            f"g6_embedding: per-core L1 use {plan.footprint_bytes} B exceeds the bound of {L1_BOUND_BYTES} B (BB-05)"
        )

    scenario = {
        "index_shape": tuple(input_tensor.shape),
        "weight_shape": tuple(weight.shape),
        "index_layout": _LAYOUT_NAMES.get(input_tensor.layout, str(input_tensor.layout)),
        "memory_config": "dram_interleaved" if memory_config is not None else None,
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

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_embedding: unsupported combination (refinement candidate): {exc}")

    return plan


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
        memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
    )

    cache_key = _cache_key(input_tensor, weight, device)
    descriptor = _DESCRIPTORS.get(cache_key)
    if descriptor is None:
        descriptor = create_program_descriptor(plan, input_tensor, weight, output_tensor, cache_key)
        _DESCRIPTORS[cache_key] = descriptor
    set_buffer_addresses(descriptor, input_tensor, weight, output_tensor)

    return ttnn.generic_op([input_tensor, weight, output_tensor], descriptor)
