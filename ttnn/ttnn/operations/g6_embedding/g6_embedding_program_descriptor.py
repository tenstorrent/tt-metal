# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Plan selection and the Metal 2.0 ProgramSpec of g6_embedding.

Work unit: a block = (row block rb, column block c), block id = rb * col_blocks_per_row + c. A row
block is BLOCK_TILE_ROWS tile-rows of the result; a column block is block_col_tiles tile-columns.
Tile-row g of the result is tile-row (g mod tile_rows_per_image) of image g // tile_rows_per_image,
so a tile-row never holds rows of two images.

The cached artifacts hold no buffer address: tensors are bound per call by generic_op through the
TensorParameters, so the same spec serves every call of a key, traced replays included (BB-07).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_DIM = 32
INDEX_BYTES = 4  # uint32 index
# The staging and output buffers carry the bf16 bits as UInt16 so the tilize stays an integer
# passthrough (no fast float tilize: NaN payloads, -0.0 and subnormals survive bit for bit).
STAGING_DTYPE = ttnn.uint16
TILE_BYTES = TILE_DIM * TILE_DIM * 2

# Blocking knobs (op_design.md, Blocking Model). Everything below derives from these.
BLOCK_TILE_ROWS = 1
INDEX_DEPTH = 2
ROWS_DEPTH = 2
OUT_DEPTH = 2
L1_BUDGET_BYTES = 1_347_776

INDEX_SLOT_BYTES = BLOCK_TILE_ROWS * TILE_DIM * INDEX_BYTES

assert INDEX_DEPTH >= 2, "the reader prefetches the next row block's indices into a second slot"

DFB_INDEX_VALUES = "index_values"
DFB_GATHERED_ROWS = "gathered_rows"
DFB_OUTPUT_TILES = "output_tiles"

K_READER = "reader"
K_COMPUTE = "compute"
K_WRITER = "writer"

TP_INDEX = "index"
TP_WEIGHT = "weight"
TP_OUT = "out"


@dataclass(frozen=True)
class Plan:
    regime: str
    tensor_images: int
    tensor_token_extent: int
    tile_rows_per_image: int
    tensor_tile_rows: int
    tensor_col_tiles: int
    row_blocks: int
    col_blocks_per_row: int
    block_col_tiles: int
    total_blocks: int
    num_cores: int
    l1_bytes_per_core: int


def _divisors(n: int):
    return [q for q in range(1, n + 1) if n % q == 0]


def col_tile_cap() -> int:
    """Largest block_col_tiles whose buffers fit the per-core L1 bound."""
    per_col_tile = (ROWS_DEPTH + OUT_DEPTH) * BLOCK_TILE_ROWS * TILE_BYTES
    return (L1_BUDGET_BYTES - INDEX_DEPTH * INDEX_SLOT_BYTES) // per_col_tile


def l1_bytes_per_core(block_col_tiles: int) -> int:
    return INDEX_DEPTH * INDEX_SLOT_BYTES + (ROWS_DEPTH + OUT_DEPTH) * BLOCK_TILE_ROWS * block_col_tiles * TILE_BYTES


def plan_for_shapes(index_shape, weight_shape, num_grid_cores: int) -> Plan:
    index_shape = tuple(index_shape)
    tensor_images = index_shape[0] if len(index_shape) == 2 else 1
    token_extent = index_shape[-1]
    tile_rows_per_image = math.ceil(token_extent / TILE_DIM)
    tensor_tile_rows = tensor_images * tile_rows_per_image
    tensor_col_tiles = weight_shape[-1] // TILE_DIM
    row_blocks = math.ceil(tensor_tile_rows / BLOCK_TILE_ROWS)

    divisors = _divisors(tensor_col_tiles)
    cap = col_tile_cap()
    # Coarsest column block that fits L1, then the coarsest that still fills the grid in one wave.
    q_l1 = min(q for q in divisors if tensor_col_tiles // q <= cap)
    q_fill = max(q for q in divisors if q <= max(1, num_grid_cores // row_blocks))
    col_blocks_per_row = max(q_l1, q_fill)
    block_col_tiles = tensor_col_tiles // col_blocks_per_row
    total_blocks = row_blocks * col_blocks_per_row

    return Plan(
        regime="whole_row" if col_blocks_per_row == 1 else "column_blocked",
        tensor_images=tensor_images,
        tensor_token_extent=token_extent,
        tile_rows_per_image=tile_rows_per_image,
        tensor_tile_rows=tensor_tile_rows,
        tensor_col_tiles=tensor_col_tiles,
        row_blocks=row_blocks,
        col_blocks_per_row=col_blocks_per_row,
        block_col_tiles=block_col_tiles,
        total_blocks=total_blocks,
        num_cores=min(total_blocks, num_grid_cores),
        l1_bytes_per_core=l1_bytes_per_core(block_col_tiles),
    )


def select_plan(input_tensor: ttnn.Tensor, weight: ttnn.Tensor) -> Plan:
    """The plan a call with these tensors runs (BB-08)."""
    grid = input_tensor.device().compute_with_storage_grid_size()
    return plan_for_shapes(tuple(input_tensor.shape), tuple(weight.shape), grid.x * grid.y)


def _core_assignment(plan: Plan, grid):
    """(core set, [(core, first_block, num_blocks)]): contiguous block runs in row-wise core order."""
    _, all_cores, group_1, group_2, blocks_1, blocks_2 = ttnn.split_work_to_cores(
        grid, plan.total_blocks, row_wise=True
    )
    assignment = []
    first = 0
    for group, per_core in ((group_1, blocks_1), (group_2, blocks_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, first, per_core))
            first += per_core
    assert first == plan.total_blocks
    return all_cores, assignment


def core_roles(plan: Plan, grid) -> dict:
    """{(x, y): {"roles": [...], "first_block": b, "num_blocks": n}} for every core the program runs on."""
    _, assignment = _core_assignment(plan, grid)
    return {
        (core.x, core.y): {"roles": [K_READER, K_COMPUTE, K_WRITER], "first_block": first, "num_blocks": count}
        for core, first, count in assignment
    }


def create_program_artifacts(input_tensor, weight, output_tensor, plan: Plan, *, zones: bool = False):
    grid = input_tensor.device().compute_with_storage_grid_size()
    core_set, assignment = _core_assignment(plan, grid)

    row_segment_bytes = plan.block_col_tiles * TILE_DIM * weight.element_size()
    block_tiles = BLOCK_TILE_ROWS * plan.block_col_tiles

    dfbs = [
        # Reader-private index slots (a self-loop buffer): one slot per row block, INDEX_DEPTH slots so
        # the next row block's indices land while the current block's rows are gathered.
        ttnn.experimental.DataflowBufferSpec(
            unique_id=DFB_INDEX_VALUES,
            entry_size=INDEX_SLOT_BYTES,
            num_entries=INDEX_DEPTH,
            data_format_metadata=ttnn.uint32,
        ),
        ttnn.experimental.DataflowBufferSpec(
            unique_id=DFB_GATHERED_ROWS,
            entry_size=TILE_BYTES,
            num_entries=ROWS_DEPTH * block_tiles,
            data_format_metadata=STAGING_DTYPE,
        ),
        ttnn.experimental.DataflowBufferSpec(
            unique_id=DFB_OUTPUT_TILES,
            entry_size=TILE_BYTES,
            num_entries=OUT_DEPTH * block_tiles,
            data_format_metadata=STAGING_DTYPE,
        ),
    ]

    compiler_options = ttnn.experimental.KernelSpec.CompilerOptions(
        defines={"G6_EMBEDDING_ZONES": "1"} if zones else {}
    )
    block_args = {"block_col_tiles": plan.block_col_tiles, "block_tile_rows": BLOCK_TILE_ROWS}
    walk_args = {
        **block_args,
        "col_blocks_per_row": plan.col_blocks_per_row,
        "tensor_tile_rows": plan.tensor_tile_rows,
        "tensor_col_tiles": plan.tensor_col_tiles,
    }
    run_arg_schema = ttnn.experimental.KernelSpec.RuntimeArgSchema(runtime_arg_names=["first_block", "num_blocks"])

    reader = ttnn.experimental.KernelSpec(
        unique_id=K_READER,
        source=str(KERNEL_DIR / "g6_embedding_reader.cpp"),
        hw_config=ttnn.experimental.create_reader_datamovement_config(),
        compiler_options=compiler_options,
        dfb_bindings=[
            ttnn.experimental.producer_of(DFB_INDEX_VALUES, DFB_INDEX_VALUES),
            ttnn.experimental.consumer_of(DFB_INDEX_VALUES, DFB_INDEX_VALUES),
            ttnn.experimental.producer_of(DFB_GATHERED_ROWS, DFB_GATHERED_ROWS),
        ],
        tensor_bindings=[
            ttnn.experimental.TensorBinding(tensor_parameter_name=TP_INDEX, accessor_name=TP_INDEX),
            ttnn.experimental.TensorBinding(tensor_parameter_name=TP_WEIGHT, accessor_name=TP_WEIGHT),
        ],
        compile_time_args={
            **walk_args,
            "tile_rows_per_image": plan.tile_rows_per_image,
            "token_extent": plan.tensor_token_extent,
            "row_segment_bytes": row_segment_bytes,
            "index_slot_bytes": INDEX_SLOT_BYTES,
        },
        runtime_arg_schema=run_arg_schema,
    )

    compute = ttnn.experimental.KernelSpec(
        unique_id=K_COMPUTE,
        source=str(KERNEL_DIR / "g6_embedding_compute.cpp"),
        hw_config=ttnn.experimental.ComputeHardwareConfig(enable_32_bit_dest=False),
        compiler_options=compiler_options,
        dfb_bindings=[
            ttnn.experimental.consumer_of(DFB_GATHERED_ROWS, DFB_GATHERED_ROWS),
            ttnn.experimental.producer_of(DFB_OUTPUT_TILES, DFB_OUTPUT_TILES),
        ],
        compile_time_args=block_args,
        runtime_arg_schema=ttnn.experimental.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_blocks"]),
    )

    writer = ttnn.experimental.KernelSpec(
        unique_id=K_WRITER,
        source=str(KERNEL_DIR / "g6_embedding_writer.cpp"),
        hw_config=ttnn.experimental.create_writer_datamovement_config(),
        compiler_options=compiler_options,
        dfb_bindings=[ttnn.experimental.consumer_of(DFB_OUTPUT_TILES, DFB_OUTPUT_TILES)],
        tensor_bindings=[ttnn.experimental.TensorBinding(tensor_parameter_name=TP_OUT, accessor_name=TP_OUT)],
        compile_time_args=walk_args,
        runtime_arg_schema=run_arg_schema,
    )

    spec = ttnn.experimental.ProgramSpec(
        name="g6_embedding",
        kernels=[reader, compute, writer],
        dataflow_buffers=dfbs,
        tensor_parameters=[
            ttnn.experimental.TensorParameter(unique_id=TP_INDEX, spec=input_tensor.spec),
            ttnn.experimental.TensorParameter(unique_id=TP_WEIGHT, spec=weight.spec),
            ttnn.experimental.TensorParameter(unique_id=TP_OUT, spec=output_tensor.spec),
        ],
        work_units=[
            ttnn.experimental.WorkUnitSpec(name="main", kernels=[K_READER, K_COMPUTE, K_WRITER], target_nodes=core_set)
        ],
    )

    first_by_core = {core: first for core, first, _ in assignment}
    count_by_core = {core: count for core, _, count in assignment}
    dm_args = {"first_block": first_by_core, "num_blocks": count_by_core}
    run_args = ttnn.experimental.ProgramRunArgs(
        kernel_run_args=[
            ttnn.experimental.KernelRunArgs(kernel=K_READER, runtime_arg_values=dm_args),
            ttnn.experimental.KernelRunArgs(kernel=K_COMPUTE, runtime_arg_values={"num_blocks": count_by_core}),
            ttnn.experimental.KernelRunArgs(kernel=K_WRITER, runtime_arg_values=dm_args),
        ]
    )

    # io_tensors of the call: [input_tensor, weight, output_tensor].
    tensor_indices = {TP_INDEX: 0, TP_WEIGHT: 1, TP_OUT: 2}
    return spec, run_args, tensor_indices
