# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_embedding: out[i, d] = W[idx[i], d] in one device program per call.

The reader reads each tile-row's index values on the device and gathers the named table-row
segments into a row-major staging CB; compute tilizes them with both CBs declared UInt16 (bit
passthrough, every bf16 pattern survives); the writer stores whole tiles. Blocking and regimes:
op_design.md -> Blocking Model.
"""

from __future__ import annotations

import functools

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_embedding_program_descriptor import (
    Knobs,
    Plan,
    bind_program_descriptor,
    compute_plan,
    core_roles,
    create_program_artifacts,
)

# Block knobs (op_design.md -> Blocking Model). Every CB capacity, loop trip count and the
# column-block L1 clamp derive from these.
BLOCK_TILE_ROWS = 1
INDEX_DEPTH = 2
ROWS_DEPTH = 2
OUT_DEPTH = 2
WRITER_BANK_ROTATION = True
READ_NOC_SPLIT_MAX_SEGMENT_BYTES = 256
READ_NOC_SPLIT_MIN_CORES = 16
L1_BUDGET_BYTES = 1_347_776
# Zone build (BB-21): set True before a profiled call to compile a named zone around every block
# operation and blocking wait. Part of the program cache key; False leaves the kernels zone-free.
DEVICE_ZONES = False

KNOBS = Knobs(
    block_tile_rows=BLOCK_TILE_ROWS,
    index_depth=INDEX_DEPTH,
    rows_depth=ROWS_DEPTH,
    out_depth=OUT_DEPTH,
    l1_budget_bytes=L1_BUDGET_BYTES,
    writer_bank_rotation=WRITER_BANK_ROTATION,
    read_noc_split_max_segment_bytes=READ_NOC_SPLIT_MAX_SEGMENT_BYTES,
    read_noc_split_min_cores=READ_NOC_SPLIT_MIN_CORES,
)

_LAYOUT_BY_NAME = {"ROW_MAJOR": ttnn.ROW_MAJOR_LAYOUT, "TILE": ttnn.TILE_LAYOUT}
_NAME_BY_LAYOUT = {layout: name for name, layout in _LAYOUT_BY_NAME.items()}


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — inputs is the 1-tuple (scenario,) that feature_spec.py documents
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    return "tile_aligned" if inputs[0]["index_shape"][-1] % 32 == 0 else "h_non_aligned"


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


def _is_dram_interleaved(memory_config) -> bool:
    return (
        memory_config.buffer_type == ttnn.BufferType.DRAM
        and memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    )


def _check_device_tensor(tensor, name):
    if not ttnn.is_tensor_storage_on_device(tensor):
        raise ValueError(f"g6_embedding: {name} must be on the device")
    if not _is_dram_interleaved(tensor.memory_config()):
        raise ValueError(f"g6_embedding: {name} must be DRAM interleaved, got {tensor.memory_config()}")


def _scenario(input_tensor, weight, memory_config):
    return (
        {
            "index_shape": tuple(input_tensor.shape),
            "weight_shape": tuple(weight.shape),
            "index_layout": _NAME_BY_LAYOUT.get(input_tensor.layout, str(input_tensor.layout)),
            "memory_config": None if memory_config is None else "dram_interleaved",
        },
    )


def validate(input_tensor, weight, *, layout=ttnn.TILE_LAYOUT, memory_config=None, runtime_rows=False):
    index_rank = len(input_tensor.shape)
    if index_rank not in (1, 2):
        raise ValueError(f"g6_embedding: input_tensor must have rank 1 or 2, got rank {index_rank}")
    if input_tensor.dtype != ttnn.uint32:
        raise ValueError(f"g6_embedding: input_tensor must be uint32, got {input_tensor.dtype}")
    weight_shape = tuple(weight.shape)
    if len(weight_shape) not in (2, 3):
        raise ValueError(f"g6_embedding: weight must have rank 2 or 3, got shape {weight_shape}")
    if len(weight_shape) == 3 and weight_shape[0] != 1:
        raise ValueError(f"g6_embedding: a rank-3 weight must be [1, V, D], got shape {weight_shape}")
    if weight.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"g6_embedding: weight must be ROW_MAJOR, got {weight.layout}")
    if weight_shape[-1] % 32 != 0:
        raise ValueError(f"g6_embedding: weight row length D={weight_shape[-1]} must be a multiple of 32")
    _check_device_tensor(input_tensor, "input_tensor")
    _check_device_tensor(weight, "weight")
    if memory_config is not None and not _is_dram_interleaved(memory_config):
        raise ValueError(f"g6_embedding: memory_config must be DRAM interleaved, got {memory_config}")

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

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_embedding: unsupported combination (refinement candidate): {exc}")

    plan = select_plan(input_tensor, weight)
    if plan.l1_bytes_per_core > L1_BUDGET_BYTES:
        raise ValueError(
            f"g6_embedding: plan needs {plan.l1_bytes_per_core} B of L1 per core, "
            f"above the per-core bound of {L1_BUDGET_BYTES} B"
        )


# ---------------------------------------------------------------------------
# Plan report (BB-08) and program cache (BB-07)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _plan_for(index_shape, weight_shape, num_grid_cores) -> Plan:
    return compute_plan(index_shape, weight_shape, num_grid_cores, KNOBS)


def select_plan(input_tensor, weight) -> Plan:
    grid = input_tensor.device().compute_with_storage_grid_size()
    return _plan_for(tuple(input_tensor.shape), tuple(weight.shape), grid.x * grid.y)


def dump_core_roles(input_tensor, weight):
    """Role of every core of the call's program: [(core, first_block, num_blocks, roles)]."""
    grid = input_tensor.device().compute_with_storage_grid_size()
    return core_roles(select_plan(input_tensor, weight), grid)


# Keyed by everything that changes the program; the values hold no buffer address.
_PROGRAM_CACHE = {}


def _program_key(input_tensor, weight, grid):
    return (
        tuple(input_tensor.shape),
        input_tensor.layout,
        tuple(weight.shape),
        weight.dtype,
        grid.x,
        grid.y,
        DEVICE_ZONES,
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
    validate(input_tensor, weight, layout=layout, memory_config=memory_config, runtime_rows=runtime_rows)

    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    output_shape = list(input_tensor.shape) + [weight.shape[-1]]
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        weight.dtype,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    key = _program_key(input_tensor, weight, grid)
    artifacts = _PROGRAM_CACHE.get(key)
    if artifacts is None:
        plan = select_plan(input_tensor, weight)
        artifacts = create_program_artifacts(input_tensor, weight, output, plan, KNOBS, grid, DEVICE_ZONES)
        _PROGRAM_CACHE[key] = artifacts

    program_descriptor = bind_program_descriptor(artifacts, input_tensor, weight, output)
    return ttnn.generic_op([input_tensor, weight, output], program_descriptor)
