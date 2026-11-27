# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_embedding host plan (select_plan) and the Metal 2.0 ProgramSpec it builds.

Work unit: a block = (tile-row group G, column block c), block id = G * col_blocks_per_row + c,
spread over the grid as contiguous id runs. Per core the reader gathers the table-row segments its
blocks' indices name into gathered_rows, compute tilizes them on the UInt16 path into output_tiles,
and the writer stores whole tile pages.

The spec holds no buffer address: tensors are bound per call through TensorParameters, so one spec
serves every call of a shape key.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_HEIGHT = 32
TILE_WIDTH = 32

# Block-size and buffer-depth knobs (op_design.md, Blocking Model). Every CB capacity, the index
# scratch size and the L1 clamp derive from these.
BLOCK_TILE_ROWS = 1  # tile-rows per block
INDEX_DEPTH = 2  # blocks of index slots: the block being gathered + the prefetched next one
ROWS_DEPTH = 2  # blocks resident in gathered_rows
OUT_DEPTH = 2  # blocks resident in output_tiles

L1_BUDGET_BYTES = 1_347_776  # BB-05: L1 left to the op per core

# The bf16 bits travel as UInt16 so the tilize takes the integer path (fast tilize rewrites NaN
# payloads and the sign of zero).
STAGING_DTYPE = ttnn.uint16
ELEMENT_BYTES = 2
TILE_BYTES = TILE_HEIGHT * TILE_WIDTH * ELEMENT_BYTES
COL_TILE_SEGMENT_BYTES = TILE_WIDTH * ELEMENT_BYTES  # one tile-column of one table row
INDEX_SLOT_BYTES = TILE_HEIGHT * 4  # the uint32 indices of one tile-row

DFB_GATHERED_ROWS = "gathered_rows"
DFB_OUTPUT_TILES = "output_tiles"
SCRATCH_INDEX_VALUES = "index_values"
TP_INDEX = "index"
TP_WEIGHT = "weight"
TP_OUTPUT = "output"
K_READER = "reader"
K_COMPUTE = "compute"
K_WRITER = "writer"


@dataclass(frozen=True)
class G6EmbeddingPlan:
    regime: str
    col_blocks_per_row: int
    block_col_tiles: int
    block_tile_rows: int
    tensor_images: int
    token_extent: int
    tile_rows_per_image: int
    tensor_tile_rows: int
    tensor_col_tiles: int
    total_blocks: int
    num_cores: int
    l1_bytes_per_core: int
    grid: tuple = field(default=(0, 0))


def _divisors(n):
    return [q for q in range(1, n + 1) if n % q == 0]


def col_tile_cap(block_tile_rows=BLOCK_TILE_ROWS):
    """Widest column block whose CBs and index scratch fit L1_BUDGET_BYTES."""
    fixed = INDEX_DEPTH * block_tile_rows * INDEX_SLOT_BYTES
    per_col_tile = (ROWS_DEPTH + OUT_DEPTH) * block_tile_rows * TILE_BYTES
    return (L1_BUDGET_BYTES - fixed) // per_col_tile


def l1_bytes_per_core(block_col_tiles, block_tile_rows=BLOCK_TILE_ROWS):
    return (
        INDEX_DEPTH * block_tile_rows * INDEX_SLOT_BYTES
        + (ROWS_DEPTH + OUT_DEPTH) * block_tile_rows * block_col_tiles * TILE_BYTES
    )


def plan_for_shapes(index_shape, weight_shape, grid_x, grid_y):
    index_shape = tuple(int(d) for d in index_shape)
    weight_shape = tuple(int(d) for d in weight_shape)
    tensor_images = index_shape[0] if len(index_shape) == 2 else 1
    token_extent = index_shape[-1]
    tile_rows_per_image = math.ceil(token_extent / TILE_HEIGHT)
    tensor_tile_rows = tensor_images * tile_rows_per_image
    tensor_col_tiles = weight_shape[-1] // TILE_WIDTH
    num_grid_cores = grid_x * grid_y

    row_groups = math.ceil(tensor_tile_rows / BLOCK_TILE_ROWS)
    cap = col_tile_cap()
    divisors = _divisors(tensor_col_tiles)
    fitting = [q for q in divisors if tensor_col_tiles // q <= cap]
    # Unreachable fallback (cap >= 1 for every depth setting that leaves room for one tile); it keeps
    # the plan defined so validate() can report the bound instead of failing here.
    q_l1 = min(fitting) if fitting else tensor_col_tiles
    q_fill = max(q for q in divisors if q <= max(1, num_grid_cores // row_groups))
    col_blocks_per_row = max(q_l1, q_fill)
    block_col_tiles = tensor_col_tiles // col_blocks_per_row
    total_blocks = row_groups * col_blocks_per_row

    return G6EmbeddingPlan(
        regime="whole_row" if col_blocks_per_row == 1 else "column_blocked",
        col_blocks_per_row=col_blocks_per_row,
        block_col_tiles=block_col_tiles,
        block_tile_rows=BLOCK_TILE_ROWS,
        tensor_images=tensor_images,
        token_extent=token_extent,
        tile_rows_per_image=tile_rows_per_image,
        tensor_tile_rows=tensor_tile_rows,
        tensor_col_tiles=tensor_col_tiles,
        total_blocks=total_blocks,
        num_cores=min(total_blocks, num_grid_cores),
        l1_bytes_per_core=l1_bytes_per_core(block_col_tiles),
        grid=(grid_x, grid_y),
    )


def select_plan(input_tensor, weight):
    """The plan a call with these tensors runs (BB-08)."""
    grid = input_tensor.device().compute_with_storage_grid_size()
    return plan_for_shapes(tuple(input_tensor.shape), tuple(weight.shape), grid.x, grid.y)


def _assign_blocks(plan):
    """(all_cores, [(core, first_block, num_blocks)]) in row-wise core order."""
    grid = ttnn.CoreCoord(plan.grid[0], plan.grid[1])
    _, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(
        grid, plan.total_blocks, row_wise=True
    )
    assignment = []
    first = 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, first, per_core))
            first += per_core
    assert first == plan.total_blocks, f"assigned {first} of {plan.total_blocks} blocks"
    return all_cores, assignment


def _tile_rows_of_blocks(plan, first_block, num_blocks):
    """Tile-rows the compute kernel tilizes for a core: a ragged last row group holds fewer."""
    total = 0
    for block in range(first_block, first_block + num_blocks):
        first_row = (block // plan.col_blocks_per_row) * plan.block_tile_rows
        total += min(plan.block_tile_rows, plan.tensor_tile_rows - first_row)
    return total


def create_program_artifacts(plan, input_tensor, weight, output_tensor):
    """(spec, run_args, tensor_indices) for generic_op; io_tensors = [input_tensor, weight, output]."""
    all_cores, assignment = _assign_blocks(plan)

    gathered_rows_entries = ROWS_DEPTH * plan.block_tile_rows * plan.block_col_tiles
    output_tiles_entries = OUT_DEPTH * plan.block_tile_rows * plan.block_col_tiles
    dataflow_buffers = [
        ttnn.experimental.DataflowBufferSpec(
            unique_id=DFB_GATHERED_ROWS,
            entry_size=TILE_BYTES,
            num_entries=gathered_rows_entries,
            data_format_metadata=STAGING_DTYPE,
        ),
        ttnn.experimental.DataflowBufferSpec(
            unique_id=DFB_OUTPUT_TILES,
            entry_size=TILE_BYTES,
            num_entries=output_tiles_entries,
            data_format_metadata=STAGING_DTYPE,
        ),
    ]
    scratchpads = [
        ttnn.experimental.ScratchpadSpec(
            unique_id=SCRATCH_INDEX_VALUES,
            size_per_node=INDEX_DEPTH * plan.block_tile_rows * INDEX_SLOT_BYTES,
        )
    ]

    block_args = {
        "block_col_tiles": plan.block_col_tiles,
        "col_blocks_per_row": plan.col_blocks_per_row,
        "block_tile_rows": plan.block_tile_rows,
        "tensor_tile_rows": plan.tensor_tile_rows,
    }
    reader = ttnn.experimental.KernelSpec(
        unique_id=K_READER,
        source=str(KERNEL_DIR / "g6_embedding_reader.cpp"),
        hw_config=ttnn.experimental.create_reader_datamovement_config(),
        dfb_bindings=[ttnn.experimental.producer_of(DFB_GATHERED_ROWS, DFB_GATHERED_ROWS)],
        scratchpad_bindings=[
            ttnn.experimental.ScratchpadBinding(
                scratchpad_spec_name=SCRATCH_INDEX_VALUES, accessor_name=SCRATCH_INDEX_VALUES
            )
        ],
        tensor_bindings=[
            ttnn.experimental.TensorBinding(tensor_parameter_name=TP_INDEX, accessor_name=TP_INDEX),
            ttnn.experimental.TensorBinding(tensor_parameter_name=TP_WEIGHT, accessor_name=TP_WEIGHT),
        ],
        compile_time_args={
            **block_args,
            "tile_rows_per_image": plan.tile_rows_per_image,
            "token_extent": plan.token_extent,
            "index_depth": INDEX_DEPTH,
        },
        runtime_arg_schema=ttnn.experimental.KernelSpec.RuntimeArgSchema(
            runtime_arg_names=["first_block", "num_blocks"]
        ),
    )
    compute = ttnn.experimental.KernelSpec(
        unique_id=K_COMPUTE,
        source=str(KERNEL_DIR / "g6_embedding_compute.cpp"),
        # 16-bit DEST: the UInt16 tilize moves the bf16 bits untouched.
        hw_config=ttnn.experimental.ComputeHardwareConfig(enable_32_bit_dest=False, double_buffer_dest=True),
        dfb_bindings=[
            ttnn.experimental.consumer_of(DFB_GATHERED_ROWS, DFB_GATHERED_ROWS),
            ttnn.experimental.producer_of(DFB_OUTPUT_TILES, DFB_OUTPUT_TILES),
        ],
        compile_time_args={"block_col_tiles": plan.block_col_tiles},
        runtime_arg_schema=ttnn.experimental.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_tile_rows"]),
    )
    writer = ttnn.experimental.KernelSpec(
        unique_id=K_WRITER,
        source=str(KERNEL_DIR / "g6_embedding_writer.cpp"),
        hw_config=ttnn.experimental.create_writer_datamovement_config(),
        dfb_bindings=[ttnn.experimental.consumer_of(DFB_OUTPUT_TILES, DFB_OUTPUT_TILES)],
        tensor_bindings=[ttnn.experimental.TensorBinding(tensor_parameter_name=TP_OUTPUT, accessor_name=TP_OUTPUT)],
        compile_time_args={**block_args, "tensor_col_tiles": plan.tensor_col_tiles},
        runtime_arg_schema=ttnn.experimental.KernelSpec.RuntimeArgSchema(
            runtime_arg_names=["first_block", "num_blocks"]
        ),
    )

    spec = ttnn.experimental.ProgramSpec(
        name="g6_embedding",
        kernels=[reader, compute, writer],
        dataflow_buffers=dataflow_buffers,
        scratchpads=scratchpads,
        tensor_parameters=[
            ttnn.experimental.TensorParameter(unique_id=TP_INDEX, spec=input_tensor.spec),
            ttnn.experimental.TensorParameter(unique_id=TP_WEIGHT, spec=weight.spec),
            ttnn.experimental.TensorParameter(unique_id=TP_OUTPUT, spec=output_tensor.spec),
        ],
        work_units=[
            ttnn.experimental.WorkUnitSpec(name="main", kernels=[K_READER, K_COMPUTE, K_WRITER], target_nodes=all_cores)
        ],
    )

    first_block = {core: first for core, first, _ in assignment}
    num_blocks = {core: count for core, _, count in assignment}
    num_tile_rows = {core: _tile_rows_of_blocks(plan, first, count) for core, first, count in assignment}
    dm_args = {"first_block": first_block, "num_blocks": num_blocks}
    run_args = ttnn.experimental.ProgramRunArgs(
        kernel_run_args=[
            ttnn.experimental.KernelRunArgs(kernel=K_READER, runtime_arg_values=dm_args),
            ttnn.experimental.KernelRunArgs(kernel=K_COMPUTE, runtime_arg_values={"num_tile_rows": num_tile_rows}),
            ttnn.experimental.KernelRunArgs(kernel=K_WRITER, runtime_arg_values=dm_args),
        ]
    )
    tensor_indices = {TP_INDEX: 0, TP_WEIGHT: 1, TP_OUTPUT: 2}
    return spec, run_args, tensor_indices
