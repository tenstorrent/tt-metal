# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_embedding: out[i, :] = W[idx[i], :], gathered on the device and written as TILE pages.

One device program per call: the reader gathers the table-row segments named by the index values
into a row-major staging CB, compute tilizes them with UInt16-declared CBs (bit passthrough), the
writer stores whole tiles. Blocking, regimes and buffer depths: op_design.md (Blocking Model).
"""

from __future__ import annotations

from dataclasses import dataclass

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_embedding_program_descriptor import (
    INDEX_BYTES,
    TILE_DIM,
    VALUE_BYTES,
    create_program_descriptor,
    patch_buffer_addresses,
)

# ---------------------------------------------------------------------------
# Blocking knobs (single source of truth; every capacity, trip count and the col_tile cap derive
# from these and from select_plan)
# ---------------------------------------------------------------------------

# Tile-rows per block along the `tile_row` axis. 1: the compute block is the core's whole run (one
# tilize helper call), so a coarser value buys no init amortization (op_design.md, Axes).
BLOCK_TILE_ROWS = 1
# Index slots (each one block's tile-rows of indices): the block being gathered + the prefetch.
INDEX_DEPTH = 2
# Blocks of cb_gathered_rows: reader fills block k+1 while compute tilizes block k.
ROWS_DEPTH = 2
# Blocks of cb_output_tiles: compute packs block k+1 while the writer drains block k.
OUT_DEPTH = 2
# BB-05: L1 the model leaves to the op on every core.
L1_BUDGET_BYTES = 1_347_776

TILE_BYTES = TILE_DIM * TILE_DIM * VALUE_BYTES
INDEX_ROW_BYTES = TILE_DIM * INDEX_BYTES  # indices of one tile-row


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """Token extent (S of [B, S], N of [N]) a multiple of 32, or a partial last tile-row."""
    index_shape = inputs[0]["index_shape"]
    return "tile_aligned" if index_shape[-1] % TILE_DIM == 0 else "h_non_aligned"


_LAYOUT_BY_NAME = {"ROW_MAJOR": ttnn.ROW_MAJOR_LAYOUT, "TILE": ttnn.TILE_LAYOUT}


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


# ---------------------------------------------------------------------------
# Plan (BB-08: select_plan reports the plan a call uses)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    regime: str
    col_blocks_per_row: int
    block_col_tiles: int
    tensor_tile_rows: int
    tensor_col_tiles: int
    total_blocks: int
    num_cores: int
    l1_bytes_per_core: int
    tensor_images: int
    tensor_token_extent: int
    tile_rows_per_image: int
    block_tile_rows: int
    index_depth: int
    rows_depth: int
    out_depth: int
    grid_x: int
    grid_y: int


def _divisors(n):
    return [q for q in range(1, n + 1) if n % q == 0]


def _col_tile_cap():
    """Largest block_col_tiles whose CBs fit the BB-05 budget at the current depths."""
    index_bytes = INDEX_DEPTH * BLOCK_TILE_ROWS * INDEX_ROW_BYTES
    bytes_per_col_tile = (ROWS_DEPTH + OUT_DEPTH) * BLOCK_TILE_ROWS * TILE_BYTES
    return (L1_BUDGET_BYTES - index_bytes) // bytes_per_col_tile


def _l1_bytes_per_core(block_col_tiles):
    return (
        INDEX_DEPTH * BLOCK_TILE_ROWS * INDEX_ROW_BYTES
        + (ROWS_DEPTH + OUT_DEPTH) * BLOCK_TILE_ROWS * block_col_tiles * TILE_BYTES
    )


def _plan_from_shapes(index_shape, weight_shape, grid_x, grid_y):
    if len(index_shape) == 2:
        tensor_images, tensor_token_extent = index_shape
    else:
        tensor_images, tensor_token_extent = 1, index_shape[0]
    tile_rows_per_image = -(-tensor_token_extent // TILE_DIM)
    tensor_tile_rows = tensor_images * tile_rows_per_image
    tensor_col_tiles = weight_shape[-1] // TILE_DIM
    tile_row_groups = -(-tensor_tile_rows // BLOCK_TILE_ROWS)
    num_grid_cores = grid_x * grid_y

    col_tile_cap = _col_tile_cap()
    divisors = _divisors(tensor_col_tiles)
    fitting = [q for q in divisors if tensor_col_tiles // q <= col_tile_cap]
    q_l1 = min(fitting) if fitting else tensor_col_tiles
    q_fill = max(q for q in divisors if q <= max(1, num_grid_cores // tile_row_groups))
    col_blocks_per_row = max(q_l1, q_fill)
    block_col_tiles = tensor_col_tiles // col_blocks_per_row
    total_blocks = tile_row_groups * col_blocks_per_row

    return Plan(
        regime="whole_row" if col_blocks_per_row == 1 else "column_blocked",
        col_blocks_per_row=col_blocks_per_row,
        block_col_tiles=block_col_tiles,
        tensor_tile_rows=tensor_tile_rows,
        tensor_col_tiles=tensor_col_tiles,
        total_blocks=total_blocks,
        num_cores=min(total_blocks, num_grid_cores),
        l1_bytes_per_core=_l1_bytes_per_core(block_col_tiles),
        tensor_images=tensor_images,
        tensor_token_extent=tensor_token_extent,
        tile_rows_per_image=tile_rows_per_image,
        block_tile_rows=BLOCK_TILE_ROWS,
        index_depth=INDEX_DEPTH,
        rows_depth=ROWS_DEPTH,
        out_depth=OUT_DEPTH,
        grid_x=grid_x,
        grid_y=grid_y,
    )


def select_plan(input_tensor: ttnn.Tensor, weight: ttnn.Tensor) -> Plan:
    """The blocking plan a call with these tensors uses (regime, block extents, cores, L1)."""
    grid = input_tensor.device().compute_with_storage_grid_size()
    return _plan_from_shapes(tuple(input_tensor.shape), tuple(weight.shape), grid.x, grid.y)


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _is_dram_interleaved(memory_config):
    return (
        memory_config.buffer_type == ttnn.BufferType.DRAM
        and memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    )


def _check_tensor_placement(tensor, name):
    if not ttnn.is_tensor_storage_on_device(tensor):
        raise ValueError(f"g6_embedding: {name} must be on the device")
    if not _is_dram_interleaved(tensor.memory_config()):
        raise ValueError(f"g6_embedding: {name} must be DRAM interleaved, got {tensor.memory_config()}")


def _check_inputs(input_tensor, weight, memory_config):
    if input_tensor.dtype != ttnn.uint32:
        raise ValueError(f"g6_embedding: input_tensor must be uint32, got {input_tensor.dtype}")
    if len(input_tensor.shape) not in (1, 2):
        raise ValueError(f"g6_embedding: input_tensor must have rank 1 or 2, got shape {tuple(input_tensor.shape)}")
    weight_shape = tuple(weight.shape)
    if len(weight_shape) not in (2, 3):
        raise ValueError(f"g6_embedding: weight must have rank 2 or 3, got shape {weight_shape}")
    if len(weight_shape) == 3 and weight_shape[0] != 1:
        raise ValueError(f"g6_embedding: a rank-3 weight must have leading extent 1, got shape {weight_shape}")
    if weight.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"g6_embedding: weight must be ROW_MAJOR, got {weight.layout}")
    if weight_shape[-1] % TILE_DIM != 0:
        raise ValueError(f"g6_embedding: weight row length D={weight_shape[-1]} must be a multiple of {TILE_DIM}")
    _check_tensor_placement(input_tensor, "input_tensor")
    _check_tensor_placement(weight, "weight")
    if memory_config is not None and not _is_dram_interleaved(memory_config):
        raise ValueError(f"g6_embedding: memory_config must be DRAM interleaved, got {memory_config}")


def _scenario(input_tensor, weight, memory_config):
    return {
        "index_shape": tuple(input_tensor.shape),
        "weight_shape": tuple(weight.shape),
        "index_layout": "TILE" if input_tensor.layout == ttnn.TILE_LAYOUT else "ROW_MAJOR",
        "memory_config": None if memory_config is None else "dram_interleaved",
    }


def validate(input_tensor, weight, *, layout=ttnn.TILE_LAYOUT, memory_config=None, runtime_rows=False):
    _check_inputs(input_tensor, weight, memory_config)

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
            raise UnsupportedAxisValue(f"g6_embedding: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_embedding: unsupported combination (refinement candidate): {exc}")

    plan = select_plan(input_tensor, weight)
    if plan.l1_bytes_per_core > L1_BUDGET_BYTES:
        raise ValueError(
            f"g6_embedding: plan needs {plan.l1_bytes_per_core} B of L1 per core, above the "
            f"{L1_BUDGET_BYTES} B per-core bound (BB-05)"
        )
    return plan


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# BB-07: one ProgramDescriptor per key, built on the first call; later calls only patch the buffer
# addresses (the descriptor holds no other per-call state, and index values never enter it).
_PROGRAM_CACHE = {}


def _program_key(input_tensor, weight, plan):
    return (
        tuple(input_tensor.shape),
        input_tensor.layout,
        tuple(weight.shape),
        weight.dtype,
        plan.grid_x,
        plan.grid_y,
    )


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
    output_shape = list(input_tensor.shape) + [weight.shape[-1]]
    # CRITICAL: allocate_tensor_on_device takes positional args only.
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        weight.dtype,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    key = _program_key(input_tensor, weight, plan)
    program_descriptor = _PROGRAM_CACHE.get(key)
    if program_descriptor is None:
        program_descriptor = create_program_descriptor(plan, input_tensor, weight, output_tensor)
        _PROGRAM_CACHE[key] = program_descriptor
    patch_buffer_addresses(program_descriptor, input_tensor, weight, output_tensor)

    return ttnn.generic_op([input_tensor, weight, output_tensor], program_descriptor)
