# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_embedding: out[i, :] = W[idx[i], :] as a TILE tensor of shape idx.shape + [D].

One device program per call: the reader takes the table rows named by index values it reads on the
device, compute tilizes them bit-opaquely, the writer stores whole tile pages.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_embedding_program_descriptor import L1_BOUND_BYTES, TILE_HEIGHT, G6EmbeddingProgram, select_plan

_LAYOUT_NAME = {ttnn.ROW_MAJOR_LAYOUT: "ROW_MAJOR", ttnn.TILE_LAYOUT: "TILE"}


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — inputs is (scenario,), keys as in feature_spec.py
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    tokens = inputs[0]["index_shape"][-1]
    return "tile_aligned" if tokens % TILE_HEIGHT == 0 else "h_non_aligned"


def tag_layout(inputs, axes):
    return {"ROW_MAJOR": ttnn.ROW_MAJOR_LAYOUT, "TILE": ttnn.TILE_LAYOUT}[inputs[0]["index_layout"]]


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
        "index_layout": _LAYOUT_NAME.get(input_tensor.layout, str(input_tensor.layout)),
        "memory_config": _memory_config_name(memory_config),
    }


def validate(input_tensor, weight, *, layout=ttnn.TILE_LAYOUT, memory_config=None, runtime_rows=False):
    axes = {
        "dtype": weight.dtype,
        "output_layout": layout,
        "runtime_rows": bool(runtime_rows),
    }
    inputs = (_scenario(input_tensor, weight, memory_config),)
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger(inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_embedding: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_embedding: unsupported combination (refinement candidate): {exc}")


def _check_inputs(input_tensor, weight, memory_config):
    """Input requirements outside the axis contract; no index value is read."""
    if input_tensor.dtype != ttnn.uint32:
        raise ValueError(f"g6_embedding: input_tensor must be ttnn.uint32, got {input_tensor.dtype}")
    if len(input_tensor.shape) not in (1, 2):
        raise ValueError(f"g6_embedding: input_tensor must have rank 1 or 2, got rank {len(input_tensor.shape)}")
    weight_shape = tuple(weight.shape)
    if len(weight_shape) not in (2, 3):
        raise ValueError(f"g6_embedding: weight must have rank 2 or 3, got rank {len(weight_shape)}")
    if len(weight_shape) == 3 and weight_shape[0] != 1:
        raise ValueError(f"g6_embedding: a rank-3 weight must be [1, V, D], got {weight_shape}")
    if weight.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"g6_embedding: weight must be ROW_MAJOR, got {weight.layout}")
    if weight_shape[-1] % TILE_HEIGHT != 0:
        raise ValueError(f"g6_embedding: D={weight_shape[-1]} must be a multiple of 32")
    for name, tensor in (("input_tensor", input_tensor), ("weight", weight)):
        if not ttnn.is_tensor_storage_on_device(tensor):
            raise ValueError(f"g6_embedding: {name} must be on the device")
        if not _is_dram_interleaved(tensor.memory_config()):
            raise ValueError(f"g6_embedding: {name} must be DRAM interleaved, got {tensor.memory_config()}")
    if memory_config is not None and not _is_dram_interleaved(memory_config):
        raise ValueError(f"g6_embedding: memory_config must be DRAM interleaved, got {memory_config}")


# ---------------------------------------------------------------------------
# Program cache (BB-07): one built descriptor per call signature; a hit only re-points buffers.
# ---------------------------------------------------------------------------

_PROGRAMS = {}


def _placement(memory_config):
    return (str(memory_config.buffer_type), str(memory_config.memory_layout))


def _cache_key(input_tensor, weight, output_memory_config, layout, runtime_rows, device, grid):
    return (
        tuple(input_tensor.shape),
        str(input_tensor.layout),
        str(input_tensor.dtype),
        tuple(weight.shape),
        str(weight.dtype),
        str(weight.layout),
        _placement(input_tensor.memory_config()),
        _placement(weight.memory_config()),
        _placement(output_memory_config),
        str(layout),
        bool(runtime_rows),
        device.id(),
        (grid.x, grid.y),
    )


def _plan_for(input_tensor, weight, grid):
    plan = select_plan(tuple(input_tensor.shape), tuple(weight.shape), grid)
    if plan.l1_bytes_per_core > L1_BOUND_BYTES:
        raise ValueError(
            f"g6_embedding: plan needs {plan.l1_bytes_per_core} B of L1 per core, above the "
            f"{L1_BOUND_BYTES} B bound"
        )
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
    _check_inputs(input_tensor, weight, memory_config)

    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    key = _cache_key(input_tensor, weight, output_memory_config, layout, runtime_rows, device, grid)
    program = _PROGRAMS.get(key)
    plan = program.plan if program is not None else _plan_for(input_tensor, weight, grid)

    output_shape = list(input_tensor.shape) + [plan.tensor_d_tiles * TILE_HEIGHT]
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        weight.dtype,
        ttnn.TILE_LAYOUT,
        device,
        output_memory_config,
    )

    if program is None:
        program = G6EmbeddingProgram(plan, input_tensor, weight, output_tensor)
        _PROGRAMS[key] = program
    descriptor = program.bind(input_tensor, weight, output_tensor)
    return ttnn.generic_op([input_tensor, weight, output_tensor], descriptor)


def plan_of(input_tensor, weight):
    """The plan a call with these tensors selects (BB-08)."""
    return select_plan(
        tuple(input_tensor.shape), tuple(weight.shape), input_tensor.device().compute_with_storage_grid_size()
    )
