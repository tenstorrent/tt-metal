# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_embedding: out[p, d] = W[idx[p], d], gathered on device and returned as a TILE tensor.

One generic_op dispatch per call. The index values are read only by the device; the table is
touched only at the rows they name; every value is a bit-identical copy of a table value.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_embedding_program_descriptor import L1_BUDGET_BYTES, get_program_descriptor, plan_for


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — inputs is (scenario,), scenario keys per feature_spec.py
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


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "bounded_cb": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------

_LAYOUT_NAMES = {ttnn.ROW_MAJOR_LAYOUT: "ROW_MAJOR", ttnn.TILE_LAYOUT: "TILE"}


def _is_dram_interleaved(memory_config):
    return (
        memory_config.buffer_type == ttnn.BufferType.DRAM
        and memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    )


def _check_device_dram_interleaved(tensor, name):
    if tensor.storage_type() != ttnn.StorageType.DEVICE:
        raise ValueError(f"g6_embedding: {name} must be on the device")
    if not _is_dram_interleaved(tensor.memory_config()):
        raise ValueError(f"g6_embedding: {name} must be DRAM interleaved, got {tensor.memory_config()}")


def _check_input_requirements(input_tensor, weight, memory_config):
    index_rank = len(input_tensor.shape)
    if index_rank not in (1, 2):
        raise ValueError(f"g6_embedding: input_tensor must have rank 1 or 2, got {index_rank}")
    if input_tensor.dtype != ttnn.uint32:
        raise ValueError(f"g6_embedding: input_tensor must be uint32, got {input_tensor.dtype}")

    weight_shape = tuple(weight.shape)
    if len(weight_shape) not in (2, 3):
        raise ValueError(f"g6_embedding: weight must be [V, D] or [1, V, D], got {weight_shape}")
    if len(weight_shape) == 3 and weight_shape[0] != 1:
        raise ValueError(f"g6_embedding: a rank-3 weight must have a leading extent of 1, got {weight_shape}")
    if weight.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"g6_embedding: weight must be ROW_MAJOR, got {weight.layout}")
    if weight_shape[-1] % 32 != 0:
        raise ValueError(f"g6_embedding: row length D={weight_shape[-1]} must be a multiple of 32")

    _check_device_dram_interleaved(input_tensor, "input_tensor")
    _check_device_dram_interleaved(weight, "weight")
    if memory_config is not None and not _is_dram_interleaved(memory_config):
        raise ValueError(f"g6_embedding: memory_config must be DRAM interleaved, got {memory_config}")

    footprint = plan_for(input_tensor, weight).footprint_bytes
    if footprint > L1_BUDGET_BYTES:
        raise ValueError(
            f"g6_embedding: per-core L1 footprint {footprint} B exceeds the BB-05 bound of {L1_BUDGET_BYTES} B"
        )


def _scenario(input_tensor, weight, memory_config):
    return {
        "index_shape": tuple(input_tensor.shape),
        "weight_shape": tuple(weight.shape),
        "index_layout": _LAYOUT_NAMES.get(input_tensor.layout, str(input_tensor.layout)),
        "memory_config": "dram_interleaved" if memory_config is not None else None,
    }


def validate(input_tensor, weight, *, layout=ttnn.TILE_LAYOUT, memory_config=None, runtime_rows=False):
    _check_input_requirements(input_tensor, weight, memory_config)

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

    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    output_shape = list(input_tensor.shape) + [int(weight.shape[-1])]
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        weight.dtype,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        output_memory_config,
    )
    descriptor = get_program_descriptor(input_tensor, weight, output_tensor)
    return ttnn.generic_op([input_tensor, weight, output_tensor], descriptor)
