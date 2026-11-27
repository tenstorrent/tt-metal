# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_embedding: out[i, :] = W[idx[i], :] as a TILE tensor of shape idx.shape + [D].

Index values are read on the device by the reader kernel and never reach the host; the result
is a bit-identical copy of the named table rows (the tilize runs on UInt16 bits).
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_embedding_program_descriptor import L1_BUDGET_BYTES, Plan, build_program, plan_for

_LAYOUT_OF_NAME = {"ROW_MAJOR": ttnn.ROW_MAJOR_LAYOUT, "TILE": ttnn.TILE_LAYOUT}
_NAME_OF_LAYOUT = {ttnn.ROW_MAJOR_LAYOUT: "ROW_MAJOR", ttnn.TILE_LAYOUT: "TILE"}


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — read the scenario dict of feature_spec.py: inputs == (scenario,)
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """Token extent (last index dim) a multiple of 32, or a partial last tile-row per image."""
    return "tile_aligned" if inputs[0]["index_shape"][-1] % 32 == 0 else "h_non_aligned"


def tag_layout(inputs, axes):
    """Layout of the INDEX tensor."""
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


def _check_device_dram_interleaved(tensor, name):
    if not ttnn.is_tensor_storage_on_device(tensor):
        raise ValueError(f"g6_embedding: {name} must be on the device")
    if not _is_dram_interleaved(tensor.memory_config()):
        raise ValueError(f"g6_embedding: {name} must be DRAM interleaved, got {tensor.memory_config()}")


def _check_structure(input_tensor, weight, memory_config):
    index_rank = len(input_tensor.shape)
    if index_rank not in (1, 2):
        raise ValueError(f"g6_embedding: input_tensor must have rank 1 or 2, got shape {tuple(input_tensor.shape)}")
    if input_tensor.dtype != ttnn.uint32:
        raise ValueError(f"g6_embedding: input_tensor must be uint32, got {input_tensor.dtype}")
    weight_shape = tuple(weight.shape)
    if len(weight_shape) not in (2, 3):
        raise ValueError(f"g6_embedding: weight must have rank 2 or 3, got shape {weight_shape}")
    if len(weight_shape) == 3 and weight_shape[0] != 1:
        raise ValueError(f"g6_embedding: a rank-3 weight must have leading extent 1, got shape {weight_shape}")
    if weight.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"g6_embedding: weight must be ROW_MAJOR, got {weight.layout}")
    if weight_shape[-1] % 32 != 0:
        raise ValueError(f"g6_embedding: weight row length D={weight_shape[-1]} must be a multiple of 32")
    _check_device_dram_interleaved(input_tensor, "input_tensor")
    _check_device_dram_interleaved(weight, "weight")
    if memory_config is not None and not _is_dram_interleaved(memory_config):
        raise ValueError(f"g6_embedding: memory_config must be DRAM interleaved, got {memory_config}")


def _scenario(input_tensor, weight, memory_config):
    return {
        "index_shape": tuple(input_tensor.shape),
        "weight_shape": tuple(weight.shape),
        "index_layout": _NAME_OF_LAYOUT.get(input_tensor.layout, str(input_tensor.layout)),
        "memory_config": None if memory_config is None else "dram_interleaved",
    }


def validate(input_tensor, weight, *, layout=ttnn.TILE_LAYOUT, memory_config=None, runtime_rows=False):
    _check_structure(input_tensor, weight, memory_config)

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

    plan = select_plan(input_tensor, weight)
    if plan.l1_bytes_per_core > L1_BUDGET_BYTES:
        raise ValueError(
            f"g6_embedding: plan needs {plan.l1_bytes_per_core} B of L1 per core, above the "
            f"{L1_BUDGET_BYTES} B bound (BB-05)"
        )


# ---------------------------------------------------------------------------
# Plan report (BB-08) and program cache (BB-07)
# ---------------------------------------------------------------------------

_PLAN_CACHE: dict = {}
_PROGRAM_CACHE: dict = {}


def _cache_key(input_tensor, weight):
    grid = input_tensor.device().compute_with_storage_grid_size()
    return (
        tuple(input_tensor.shape),
        input_tensor.layout,
        tuple(weight.shape),
        weight.dtype,
        grid.x,
        grid.y,
    )


def select_plan(input_tensor, weight) -> Plan:
    """The blocking plan this call runs: regime, block extents, block and core counts, L1 per core."""
    key = _cache_key(input_tensor, weight)
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = plan_for(tuple(input_tensor.shape), tuple(weight.shape), key[4], key[5])
        _PLAN_CACHE[key] = plan
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
    validate(input_tensor, weight, layout=layout, memory_config=memory_config, runtime_rows=runtime_rows)

    device = input_tensor.device()
    output_shape = list(input_tensor.shape) + [weight.shape[-1]]
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        weight.dtype,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    key = _cache_key(input_tensor, weight)
    program = _PROGRAM_CACHE.get(key)
    if program is None:
        program = build_program(input_tensor, weight, output, select_plan(input_tensor, weight))
        _PROGRAM_CACHE[key] = program
    return ttnn.generic_op([input_tensor, weight, output], program.bind(input_tensor, weight, output))
