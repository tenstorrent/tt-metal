# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_embedding: out[i, :] = W[idx[i], :] as a TILE tensor of shape idx.shape + [D].

One device program per call. The reader gathers table-row segments named by the index values
(read on the device, never on the host) into a row-major staging CB, compute tilizes them with
both CBs declared UInt16 so every bfloat16 bit pattern passes through unchanged, and the writer
stores whole tiles. Blocking model and regimes: op_design.md.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_embedding_program_descriptor import (
    INDEX_SLOT_BYTES,
    TILE_BYTES,
    TILE_HEIGHT,
    TILE_WIDTH,
    create_program_artifacts,
)

# ---------------------------------------------------------------------------
# Blocking knobs (single source of truth; every capacity and the plan derive from these)
# ---------------------------------------------------------------------------

BLOCK_TILE_ROWS = 1  # tile-rows per block along `tile_row` (op_design.md, Axes: why 1)
INDEX_DEPTH = 2  # index slots (blocks of tile-rows) in cb_index_values
ROWS_DEPTH = 2  # blocks in cb_gathered_rows
OUT_DEPTH = 2  # blocks in cb_output_tiles
L1_BUDGET_BYTES = 1_347_776  # BB-05: L1 left to the op on every core


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------

_LAYOUT_BY_NAME = {"ROW_MAJOR": ttnn.ROW_MAJOR_LAYOUT, "TILE": ttnn.TILE_LAYOUT}
_NAME_BY_LAYOUT = {layout: name for name, layout in _LAYOUT_BY_NAME.items()}


def tag_alignment(inputs, axes):
    """Token extent (last index dim) a multiple of 32, or a partial last tile-row per image."""
    return "tile_aligned" if inputs[0]["index_shape"][-1] % TILE_HEIGHT == 0 else "h_non_aligned"


def tag_layout(inputs, axes):
    """Layout of the index tensor."""
    name = inputs[0]["index_layout"]
    return _LAYOUT_BY_NAME.get(name, name)


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
# Plan (BB-08: reportable; op_design.md, Regimes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    regime: str
    tensor_images: int
    tensor_token_extent: int
    tile_rows_per_image: int
    tensor_tile_rows: int
    tensor_col_tiles: int
    block_tile_rows: int
    row_blocks: int
    col_blocks_per_row: int
    block_col_tiles: int
    total_blocks: int
    num_cores: int
    index_depth: int
    rows_depth: int
    out_depth: int
    l1_bytes_per_core: int


def _divisors(n):
    return [q for q in range(1, n + 1) if n % q == 0]


def _l1_bytes(block_col_tiles):
    index_bytes = INDEX_DEPTH * BLOCK_TILE_ROWS * INDEX_SLOT_BYTES
    tile_bytes = (ROWS_DEPTH + OUT_DEPTH) * BLOCK_TILE_ROWS * block_col_tiles * TILE_BYTES
    return index_bytes + tile_bytes


@functools.lru_cache(maxsize=256)
def _plan(index_shape, weight_shape, num_grid_cores):
    images = index_shape[0] if len(index_shape) == 2 else 1
    tokens = index_shape[-1]
    tile_rows_per_image = -(-tokens // TILE_HEIGHT)
    tensor_tile_rows = images * tile_rows_per_image
    tensor_col_tiles = weight_shape[-1] // TILE_WIDTH
    row_blocks = -(-tensor_tile_rows // BLOCK_TILE_ROWS)

    col_tile_cap = (L1_BUDGET_BYTES - INDEX_DEPTH * BLOCK_TILE_ROWS * INDEX_SLOT_BYTES) // (
        (ROWS_DEPTH + OUT_DEPTH) * BLOCK_TILE_ROWS * TILE_BYTES
    )
    divisors = _divisors(tensor_col_tiles)
    l1_fitting = [q for q in divisors if tensor_col_tiles // q <= col_tile_cap]
    # Unreachable at the module depths (col_tile_cap >= 1); validate() turns it into the BB-05 refusal.
    q_l1 = min(l1_fitting) if l1_fitting else tensor_col_tiles
    fill_cap = max(1, num_grid_cores // row_blocks)
    q_fill = max(q for q in divisors if q <= fill_cap)
    col_blocks_per_row = max(q_l1, q_fill)
    block_col_tiles = tensor_col_tiles // col_blocks_per_row
    total_blocks = row_blocks * col_blocks_per_row

    return Plan(
        regime="whole_row" if col_blocks_per_row == 1 else "column_blocked",
        tensor_images=images,
        tensor_token_extent=tokens,
        tile_rows_per_image=tile_rows_per_image,
        tensor_tile_rows=tensor_tile_rows,
        tensor_col_tiles=tensor_col_tiles,
        block_tile_rows=BLOCK_TILE_ROWS,
        row_blocks=row_blocks,
        col_blocks_per_row=col_blocks_per_row,
        block_col_tiles=block_col_tiles,
        total_blocks=total_blocks,
        num_cores=min(total_blocks, num_grid_cores),
        index_depth=INDEX_DEPTH,
        rows_depth=ROWS_DEPTH,
        out_depth=OUT_DEPTH,
        l1_bytes_per_core=_l1_bytes(block_col_tiles),
    )


def _grid(device):
    grid = device.compute_with_storage_grid_size()
    return grid.x, grid.y


def select_plan(input_tensor, weight):
    """The plan a call with these tensors runs (regime, block extents, cores, L1 per core)."""
    grid_x, grid_y = _grid(input_tensor.device())
    return _plan(tuple(input_tensor.shape), tuple(weight.shape), grid_x * grid_y)


def core_roles(input_tensor, weight):
    """BB-21 role dump: every core in the program runs the reader, compute and writer kernels."""
    plan = select_plan(input_tensor, weight)
    grid = input_tensor.device().compute_with_storage_grid_size()
    _, _, group_1, group_2, blocks_1, blocks_2 = ttnn.split_work_to_cores(grid, plan.total_blocks, row_wise=True)
    roles = {}
    for group, blocks in ((group_1, blocks_1), (group_2, blocks_2)):
        if blocks == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            roles[(core.x, core.y)] = {"roles": ("reader", "compute", "writer"), "blocks": blocks}
    return roles


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _is_dram_interleaved(memory_config):
    return (
        memory_config.buffer_type == ttnn.BufferType.DRAM
        and memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    )


def _check_device_tensor(name, tensor):
    if not ttnn.is_tensor_storage_on_device(tensor):
        raise ValueError(f"g6_embedding: {name} must be on the device")
    if not _is_dram_interleaved(tensor.memory_config()):
        raise ValueError(f"g6_embedding: {name} must be DRAM interleaved, got {tensor.memory_config()}")


def _scenario(input_tensor, weight, memory_config):
    return {
        "index_shape": tuple(input_tensor.shape),
        "weight_shape": tuple(weight.shape),
        "index_layout": _NAME_BY_LAYOUT.get(input_tensor.layout, str(input_tensor.layout)),
        "memory_config": None if memory_config is None else "dram_interleaved",
    }


def validate(input_tensor, weight, *, layout=ttnn.TILE_LAYOUT, memory_config=None, runtime_rows=False):
    # Structural refusals that the axis taggers depend on.
    if len(input_tensor.shape) not in (1, 2):
        raise ValueError(f"g6_embedding: input_tensor must have rank 1 or 2, got shape {tuple(input_tensor.shape)}")
    if input_tensor.dtype != ttnn.uint32:
        raise ValueError(f"g6_embedding: input_tensor must be uint32, got {input_tensor.dtype}")
    weight_shape = tuple(weight.shape)
    if len(weight_shape) not in (2, 3):
        raise ValueError(f"g6_embedding: weight must have rank 2 or 3, got shape {weight_shape}")
    if len(weight_shape) == 3 and weight_shape[0] != 1:
        raise ValueError(f"g6_embedding: a rank-3 weight must be [1, V, D], got shape {weight_shape}")
    if weight.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"g6_embedding: weight must be ROW_MAJOR, got {weight.layout}")
    _check_device_tensor("input_tensor", input_tensor)
    _check_device_tensor("weight", weight)

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

    if weight_shape[-1] % TILE_WIDTH != 0:
        raise ValueError(f"g6_embedding: weight row length D={weight_shape[-1]} must be a multiple of {TILE_WIDTH}")
    if memory_config is not None and not _is_dram_interleaved(memory_config):
        raise ValueError(f"g6_embedding: memory_config must be DRAM interleaved, got {memory_config}")
    plan = select_plan(input_tensor, weight)
    if plan.l1_bytes_per_core > L1_BUDGET_BYTES:
        raise ValueError(
            f"g6_embedding: plan needs {plan.l1_bytes_per_core} B of L1 per core, above the "
            f"{L1_BUDGET_BYTES} B bound (BB-05)"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# BB-07: program artifacts per program-shaping key; a hit only re-binds the three buffer addresses.
_PROGRAM_CACHE = {}


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
    plan = select_plan(input_tensor, weight)
    output_memory_config = ttnn.DRAM_MEMORY_CONFIG if memory_config is None else memory_config
    output_shape = list(input_tensor.shape) + [weight.shape[-1]]
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        weight.dtype,
        ttnn.TILE_LAYOUT,
        device,
        output_memory_config,
    )

    grid_x, grid_y = _grid(device)
    key = (
        tuple(input_tensor.shape),
        input_tensor.layout,
        tuple(weight.shape),
        weight.dtype,
        grid_x,
        grid_y,
    )
    artifacts = _PROGRAM_CACHE.get(key)
    if artifacts is None:
        artifacts = create_program_artifacts(plan, input_tensor, weight, output)
        _PROGRAM_CACHE[key] = artifacts
    artifacts.bind(input_tensor, weight, output)

    return ttnn.generic_op([input_tensor, weight, output], artifacts.descriptor)
