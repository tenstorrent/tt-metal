# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_embedding: out[i, d] = W[idx[i], d], one device program per call.

The reader gathers the table rows named by the index values (read on the device) into a
row-major staging CB, compute tilizes them with both CBs declared UInt16 so every bf16 bit
pattern passes through unchanged, and the writer stores whole tiles. Blocking and regimes:
op_design.md -> Blocking Model.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_embedding_program_descriptor import create_program_descriptor

# Block knobs (op_design.md -> Blocking Model). Every CB capacity, loop trip count and the
# column-block clamp derive from these.
BLOCK_TILE_ROWS = 1
INDEX_DEPTH = 2
ROWS_DEPTH = 2
OUT_DEPTH = 2
L1_BUDGET_BYTES = 1_347_776

TILE_HEIGHT = 32
TILE_WIDTH = 32
# bf16 bits moved as UInt16; one 32x32 tile of them.
TILE_BYTES = TILE_HEIGHT * TILE_WIDTH * 2
INDEX_BYTES = 4


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — inputs is the 1-tuple (scenario,) documented in feature_spec.py
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    token_extent = inputs[0]["index_shape"][-1]
    return "tile_aligned" if token_extent % TILE_HEIGHT == 0 else "h_non_aligned"


_LAYOUT_BY_NAME = {"ROW_MAJOR": ttnn.ROW_MAJOR_LAYOUT, "TILE": ttnn.TILE_LAYOUT}
_NAME_BY_LAYOUT = {layout: name for name, layout in _LAYOUT_BY_NAME.items()}


def tag_layout(inputs, axes):
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
# Host plan (BB-08: the selected plan is reportable through select_plan)
# ---------------------------------------------------------------------------


class Plan(NamedTuple):
    regime: str
    col_blocks_per_row: int
    block_col_tiles: int
    tensor_tile_rows: int
    tensor_col_tiles: int
    total_blocks: int
    num_cores: int
    l1_bytes_per_core: int
    tensor_images: int
    token_extent: int
    tile_rows_per_image: int
    row_blocks: int
    block_tile_rows: int
    index_depth: int
    rows_depth: int
    out_depth: int
    grid_x: int
    grid_y: int

    @property
    def index_slot_bytes(self) -> int:
        return self.block_tile_rows * TILE_HEIGHT * INDEX_BYTES

    @property
    def block_pages(self) -> int:
        return self.block_tile_rows * self.block_col_tiles


def _divisors(n: int):
    return [q for q in range(1, n + 1) if n % q == 0]


def _l1_bytes(block_col_tiles: int) -> int:
    return (
        INDEX_DEPTH * BLOCK_TILE_ROWS * TILE_HEIGHT * INDEX_BYTES
        + (ROWS_DEPTH + OUT_DEPTH) * BLOCK_TILE_ROWS * block_col_tiles * TILE_BYTES
    )


@functools.lru_cache(maxsize=256)
def _plan_for(index_shape: tuple, weight_shape: tuple, grid_x: int, grid_y: int) -> Plan:
    tensor_images = index_shape[0] if len(index_shape) == 2 else 1
    token_extent = index_shape[-1]
    tile_rows_per_image = -(-token_extent // TILE_HEIGHT)
    tensor_tile_rows = tensor_images * tile_rows_per_image
    tensor_col_tiles = weight_shape[-1] // TILE_WIDTH
    row_blocks = -(-tensor_tile_rows // BLOCK_TILE_ROWS)
    num_grid_cores = grid_x * grid_y

    col_tile_cap = (L1_BUDGET_BYTES - INDEX_DEPTH * BLOCK_TILE_ROWS * TILE_HEIGHT * INDEX_BYTES) // (
        (ROWS_DEPTH + OUT_DEPTH) * BLOCK_TILE_ROWS * TILE_BYTES
    )
    divisors = _divisors(tensor_col_tiles)
    # The tilize block width is a template parameter, so column blocks must divide the row exactly.
    fitting = [q for q in divisors if tensor_col_tiles // q <= col_tile_cap]
    q_l1 = min(fitting) if fitting else tensor_col_tiles
    q_fill = max(q for q in divisors if q <= max(1, num_grid_cores // row_blocks))
    col_blocks_per_row = max(q_l1, q_fill)
    block_col_tiles = tensor_col_tiles // col_blocks_per_row
    total_blocks = row_blocks * col_blocks_per_row

    return Plan(
        regime="whole_row" if col_blocks_per_row == 1 else "column_blocked",
        col_blocks_per_row=col_blocks_per_row,
        block_col_tiles=block_col_tiles,
        tensor_tile_rows=tensor_tile_rows,
        tensor_col_tiles=tensor_col_tiles,
        total_blocks=total_blocks,
        num_cores=min(total_blocks, num_grid_cores),
        l1_bytes_per_core=_l1_bytes(block_col_tiles),
        tensor_images=tensor_images,
        token_extent=token_extent,
        tile_rows_per_image=tile_rows_per_image,
        row_blocks=row_blocks,
        block_tile_rows=BLOCK_TILE_ROWS,
        index_depth=INDEX_DEPTH,
        rows_depth=ROWS_DEPTH,
        out_depth=OUT_DEPTH,
        grid_x=grid_x,
        grid_y=grid_y,
    )


def select_plan(input_tensor: ttnn.Tensor, weight: ttnn.Tensor) -> Plan:
    grid = input_tensor.device().compute_with_storage_grid_size()
    return _plan_for(tuple(input_tensor.shape), tuple(weight.shape), grid.x, grid.y)


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


def _check_structure(input_tensor, weight, memory_config) -> None:
    for name, tensor in (("input_tensor", input_tensor), ("weight", weight)):
        if not ttnn.is_tensor_storage_on_device(tensor):
            raise ValueError(f"g6_embedding: {name} must be on the device")
        if not _is_dram_interleaved(tensor.memory_config()):
            raise ValueError(f"g6_embedding: {name} must be DRAM interleaved, got {tensor.memory_config()}")

    index_rank = len(input_tensor.shape)
    if index_rank not in (1, 2):
        raise ValueError(f"g6_embedding: input_tensor must have rank 1 or 2, got shape {tuple(input_tensor.shape)}")
    if input_tensor.dtype != ttnn.uint32:
        raise ValueError(f"g6_embedding: input_tensor must be ttnn.uint32, got {input_tensor.dtype}")

    weight_shape = tuple(weight.shape)
    if len(weight_shape) not in (2, 3):
        raise ValueError(f"g6_embedding: weight must have rank 2 or 3, got shape {weight_shape}")
    if len(weight_shape) == 3 and weight_shape[0] != 1:
        raise ValueError(f"g6_embedding: a rank-3 weight must have leading extent 1, got shape {weight_shape}")
    if weight.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"g6_embedding: weight must be ROW_MAJOR, got {weight.layout}")
    if weight_shape[-1] % TILE_WIDTH != 0:
        raise ValueError(f"g6_embedding: weight row length D={weight_shape[-1]} must be a multiple of {TILE_WIDTH}")

    if memory_config is not None and not _is_dram_interleaved(memory_config):
        raise ValueError(f"g6_embedding: memory_config must be DRAM interleaved, got {memory_config}")


def validate(input_tensor, weight, *, layout=ttnn.TILE_LAYOUT, memory_config=None, runtime_rows=False):
    _check_structure(input_tensor, weight, memory_config)

    scenario = {
        "index_shape": tuple(input_tensor.shape),
        "weight_shape": tuple(weight.shape),
        "index_layout": _NAME_BY_LAYOUT.get(input_tensor.layout, str(input_tensor.layout)),
        "memory_config": _memory_config_name(memory_config),
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

    plan = select_plan(input_tensor, weight)
    if plan.l1_bytes_per_core > L1_BUDGET_BYTES:
        raise ValueError(
            f"g6_embedding: plan needs {plan.l1_bytes_per_core} B of L1 per core, above the "
            f"{L1_BUDGET_BYTES} B per-core bound (BB-05)"
        )


# ---------------------------------------------------------------------------
# Program cache (BB-07): one descriptor per key, buffer addresses patched per call
# ---------------------------------------------------------------------------


class _CachedProgram(NamedTuple):
    descriptor: ttnn.ProgramDescriptor
    reader: ttnn.KernelDescriptor
    writer: ttnn.KernelDescriptor


_PROGRAM_CACHE: dict = {}


def _program_for(input_tensor, weight, output, plan: Plan) -> _CachedProgram:
    key = (
        tuple(input_tensor.shape),
        input_tensor.layout,
        tuple(weight.shape),
        weight.dtype,
        plan.grid_x,
        plan.grid_y,
    )
    cached = _PROGRAM_CACHE.get(key)
    if cached is None:
        descriptor, reader_index, writer_index = create_program_descriptor(input_tensor, weight, output, plan)
        # References into the descriptor's own kernels: assigning their common runtime args
        # updates the descriptor in place.
        kernels = descriptor.kernels
        cached = _CachedProgram(descriptor, kernels[reader_index], kernels[writer_index])
        _PROGRAM_CACHE[key] = cached
    return cached


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

    plan = select_plan(input_tensor, weight)
    output_shape = list(input_tensor.shape) + [weight.shape[-1]]
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        weight.dtype,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
    )

    program = _program_for(input_tensor, weight, output, plan)
    program.reader.common_runtime_args = [input_tensor.buffer_address(), weight.buffer_address()]
    program.writer.common_runtime_args = [output.buffer_address()]
    return ttnn.generic_op([input_tensor, weight, output], program.descriptor)
