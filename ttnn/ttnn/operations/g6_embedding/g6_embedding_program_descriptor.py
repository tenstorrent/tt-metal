# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Blocking plan and Metal 2.0 ProgramSpec artifacts for g6_embedding.

The block knobs below are the single source of every CB capacity, scratchpad size, loop trip
count and core assignment; see op_design.md (Blocking Model) and l1_ledger.md.

The artifacts carry no buffer address: tensors are bound per call through TensorParameter /
TensorBinding, so one cached (spec, run_args, tensor_indices) triple serves every call of a shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---- Block knobs (op_design.md, Blocking Model) -------------------------------------------------
# Tile-rows per block. 1: the compute block is the core's whole run (one tilize call), so a coarser
# CB quantum buys no init amortization and only delays the first push.
BLOCK_TILE_ROWS = 1
# Index slots, each holding one block's indices: the block being gathered + the prefetched next.
INDEX_DEPTH = 2
# Blocks in flight in the gathered-rows CB (gather k+1 overlaps tilize k).
ROWS_DEPTH = 2
# Blocks in flight in the output CB (tilize k+1 overlaps the write of k).
OUT_DEPTH = 2
# L1 the model leaves to the op on every core (BB-05).
L1_BUDGET_BYTES = 1_347_776

# ---- Format facts ------------------------------------------------------------------------------
TILE_DIM = 32
# One 32x32 tile of 16-bit values: the bf16 output tile and the UInt16 staging tile alike.
TILE_BYTES = TILE_DIM * TILE_DIM * 2
INDEX_BYTES = 4
# Indices of one tile-row (32 uint32).
TILE_ROW_INDEX_BYTES = TILE_DIM * INDEX_BYTES

# ---- Names shared with the kernels (dfb::, scratch::, tensor::, args::) -------------------------
DFB_GATHERED_ROWS = "gathered_rows"
DFB_OUTPUT_TILES = "output_tiles"
SCRATCH_INDEX_VALUES = "index_values"
TP_INDEX = "index"
TP_TABLE = "table"
TP_OUTPUT = "output"
K_READER = "reader"
K_COMPUTE = "compute"
K_WRITER = "writer"


@dataclass(frozen=True)
class Plan:
    """The blocking a call runs with (BB-08). Every field derives from the shapes and the grid."""

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
    l1_bytes_per_core: int
    grid_x: int
    grid_y: int


def _divisors(n: int) -> list[int]:
    return [q for q in range(1, n + 1) if n % q == 0]


def l1_bytes_per_core(block_col_tiles: int) -> int:
    index_bytes = INDEX_DEPTH * BLOCK_TILE_ROWS * TILE_ROW_INDEX_BYTES
    staged_bytes = (ROWS_DEPTH + OUT_DEPTH) * BLOCK_TILE_ROWS * block_col_tiles * TILE_BYTES
    return index_bytes + staged_bytes


def col_tile_cap() -> int:
    index_bytes = INDEX_DEPTH * BLOCK_TILE_ROWS * TILE_ROW_INDEX_BYTES
    return (L1_BUDGET_BYTES - index_bytes) // ((ROWS_DEPTH + OUT_DEPTH) * BLOCK_TILE_ROWS * TILE_BYTES)


def plan_for_shapes(index_shape, weight_shape, grid_x: int, grid_y: int) -> Plan:
    index_shape = tuple(int(d) for d in index_shape)
    images = index_shape[0] if len(index_shape) == 2 else 1
    tokens = index_shape[-1]
    tile_rows_per_image = -(-tokens // TILE_DIM)
    tensor_tile_rows = images * tile_rows_per_image
    tensor_col_tiles = int(weight_shape[-1]) // TILE_DIM
    row_blocks = -(-tensor_tile_rows // BLOCK_TILE_ROWS)
    num_grid_cores = grid_x * grid_y

    divisors = _divisors(tensor_col_tiles)
    cap = col_tile_cap()
    fitting = [q for q in divisors if tensor_col_tiles // q <= cap]
    if not fitting:
        raise ValueError(
            f"g6_embedding: no column block of weight D={weight_shape[-1]} fits the per-core L1 bound "
            f"of {L1_BUDGET_BYTES} bytes"
        )
    # The coarsest column split that fits L1, raised to the finest one that still gives every core at
    # most one block when tile-rows alone cannot fill the grid.
    q_l1 = min(fitting)
    q_fill = max(q for q in divisors if q <= max(1, num_grid_cores // row_blocks))
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
        l1_bytes_per_core=l1_bytes_per_core(block_col_tiles),
        grid_x=grid_x,
        grid_y=grid_y,
    )


def select_plan(input_tensor: ttnn.Tensor, weight: ttnn.Tensor) -> Plan:
    """The plan g6_embedding(input_tensor, weight) runs with on this device."""
    grid = input_tensor.device().compute_with_storage_grid_size()
    return plan_for_shapes(tuple(input_tensor.shape), tuple(weight.shape), grid.x, grid.y)


def _tile_rows_in_row_block(plan: Plan, row_block: int) -> int:
    return min(plan.block_tile_rows, plan.tensor_tile_rows - row_block * plan.block_tile_rows)


def _assign_blocks(plan: Plan):
    """(core_set, [(core, first_block, num_blocks, num_tile_rows)]): contiguous block-id runs in
    row-wise core order."""
    grid = ttnn.CoreCoord(plan.grid_x, plan.grid_y)
    _, core_set, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(
        grid, plan.total_blocks, row_wise=True
    )
    assignment = []
    first_block = 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            tile_rows = sum(
                _tile_rows_in_row_block(plan, block // plan.col_blocks_per_row)
                for block in range(first_block, first_block + per_core)
            )
            assignment.append((core, first_block, per_core, tile_rows))
            first_block += per_core
    if first_block != plan.total_blocks:
        raise RuntimeError(f"g6_embedding: assigned {first_block} of {plan.total_blocks} blocks")
    return core_set, assignment


def create_program_artifacts(plan: Plan, input_tensor: ttnn.Tensor, weight: ttnn.Tensor, output_tensor: ttnn.Tensor):
    """(spec, run_args, tensor_indices) for io_tensors = [input_tensor, weight, output_tensor]."""
    core_set, assignment = _assign_blocks(plan)
    block_pages = plan.block_tile_rows * plan.block_col_tiles

    dataflow_buffers = [
        # UInt16, not Float16_b: the tilize helper then takes the integer path (no fast tilize through
        # float registers), so NaN payloads, -0.0 and subnormals pass bit for bit.
        ttnn.experimental.DataflowBufferSpec(
            unique_id=DFB_GATHERED_ROWS,
            entry_size=TILE_BYTES,
            num_entries=ROWS_DEPTH * block_pages,
            data_format_metadata=ttnn.uint16,
        ),
        ttnn.experimental.DataflowBufferSpec(
            unique_id=DFB_OUTPUT_TILES,
            entry_size=TILE_BYTES,
            num_entries=OUT_DEPTH * block_pages,
            data_format_metadata=ttnn.uint16,
        ),
    ]
    scratchpads = [
        ttnn.experimental.ScratchpadSpec(
            unique_id=SCRATCH_INDEX_VALUES,
            size_per_node=INDEX_DEPTH * plan.block_tile_rows * TILE_ROW_INDEX_BYTES,
            data_format_metadata=ttnn.uint32,
        )
    ]
    tensor_parameters = [
        ttnn.experimental.TensorParameter(unique_id=TP_INDEX, spec=input_tensor.spec),
        ttnn.experimental.TensorParameter(unique_id=TP_TABLE, spec=weight.spec),
        ttnn.experimental.TensorParameter(unique_id=TP_OUTPUT, spec=output_tensor.spec),
    ]

    block_ct_args = {"block_col_tiles": plan.block_col_tiles, "block_tile_rows": plan.block_tile_rows}
    block_run_args = ["first_block", "num_blocks"]

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
            ttnn.experimental.TensorBinding(tensor_parameter_name=TP_TABLE, accessor_name=TP_TABLE),
        ],
        compile_time_args={**block_ct_args, "index_depth": INDEX_DEPTH},
        runtime_arg_schema=ttnn.experimental.KernelSpec.RuntimeArgSchema(
            runtime_arg_names=block_run_args,
            common_runtime_arg_names=[
                "col_blocks_per_row",
                "tile_rows_per_image",
                "tensor_tile_rows",
                "token_extent",
            ],
        ),
    )

    compute = ttnn.experimental.KernelSpec(
        unique_id=K_COMPUTE,
        source=str(KERNEL_DIR / "g6_embedding_compute.cpp"),
        # 16-bit DEST and half-sync (the defaults): the stock UInt16 tilize configuration.
        hw_config=ttnn.experimental.ComputeHardwareConfig(enable_32_bit_dest=False),
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
        compile_time_args=block_ct_args,
        runtime_arg_schema=ttnn.experimental.KernelSpec.RuntimeArgSchema(
            runtime_arg_names=block_run_args,
            common_runtime_arg_names=["col_blocks_per_row", "tensor_tile_rows", "tensor_col_tiles"],
        ),
    )

    spec = ttnn.experimental.ProgramSpec(
        name="g6_embedding",
        kernels=[reader, compute, writer],
        dataflow_buffers=dataflow_buffers,
        scratchpads=scratchpads,
        tensor_parameters=tensor_parameters,
        work_units=[
            ttnn.experimental.WorkUnitSpec(name="main", kernels=[K_READER, K_COMPUTE, K_WRITER], target_nodes=core_set)
        ],
    )

    first_block_by_core = {core: first for core, first, _, _ in assignment}
    num_blocks_by_core = {core: count for core, _, count, _ in assignment}
    num_tile_rows_by_core = {core: tile_rows for core, _, _, tile_rows in assignment}
    block_values = {"first_block": first_block_by_core, "num_blocks": num_blocks_by_core}

    run_args = ttnn.experimental.ProgramRunArgs(
        kernel_run_args=[
            ttnn.experimental.KernelRunArgs(
                kernel=K_READER,
                runtime_arg_values=block_values,
                common_runtime_arg_values={
                    "col_blocks_per_row": plan.col_blocks_per_row,
                    "tile_rows_per_image": plan.tile_rows_per_image,
                    "tensor_tile_rows": plan.tensor_tile_rows,
                    "token_extent": plan.tensor_token_extent,
                },
            ),
            ttnn.experimental.KernelRunArgs(
                kernel=K_COMPUTE, runtime_arg_values={"num_tile_rows": num_tile_rows_by_core}
            ),
            ttnn.experimental.KernelRunArgs(
                kernel=K_WRITER,
                runtime_arg_values=block_values,
                common_runtime_arg_values={
                    "col_blocks_per_row": plan.col_blocks_per_row,
                    "tensor_tile_rows": plan.tensor_tile_rows,
                    "tensor_col_tiles": plan.tensor_col_tiles,
                },
            ),
        ]
    )

    tensor_indices = {TP_INDEX: 0, TP_TABLE: 1, TP_OUTPUT: 2}
    return spec, run_args, tensor_indices
