# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Plan selection and program artifacts for g6_embedding (op_design.md -> Blocking Model).

Work unit = block = (block-row gb, column block c), block id = gb * col_blocks_per_row + c, where a
block-row is BLOCK_TILE_ROWS consecutive output tile-rows and a column block is block_col_tiles
consecutive tile-columns. Blocks are spread over the grid as contiguous id runs.

Nothing here reads a tensor value: every quantity derives from shapes, the grid and the block knobs.
Buffer addresses are not part of the artifacts; the entry point patches them into the common
runtime args on every call.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_HEIGHT = 32
TILE_WIDTH = 32
# One 32x32 tile of 16-bit values: bf16 bits moved through UInt16 CBs.
TILE_BYTES = TILE_HEIGHT * TILE_WIDTH * 2
# Index bytes of one tile-row: 32 uint32 positions.
INDEX_ROW_BYTES = TILE_HEIGHT * 4

CB_INDEX_VALUES = 0
CB_GATHERED_ROWS = 1
CB_OUTPUT_TILES = 2


class Knobs(NamedTuple):
    block_tile_rows: int
    index_depth: int
    rows_depth: int
    out_depth: int
    l1_budget_bytes: int
    # store_block starts each core's tile-row at tile (core ordinal mod block_col_tiles), so cores
    # whose blocks start on the same DRAM bank do not walk the banks in lockstep.
    writer_bank_rotation: bool
    # Narrow row segments read by many cores at once are bound by the request rate of each NoC's DRAM
    # endpoints, not by bytes, so odd rows go to the second NoC for its request capacity. Wide segments
    # stay on the reader's NoC (NoC1 is left to the writes), and so do plans with few cores or one row
    # per block, where too few requests are in flight for the split to pay for dynamic NoC mode.
    read_noc_split_max_segment_bytes: int
    read_noc_split_min_cores: int


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
    block_rows: int
    block_tile_rows: int
    read_noc_split: bool


def _divisors(n: int) -> list[int]:
    return [q for q in range(1, n + 1) if n % q == 0]


def l1_bytes_per_core(knobs: Knobs, block_col_tiles: int) -> int:
    index_bytes = knobs.index_depth * knobs.block_tile_rows * INDEX_ROW_BYTES
    tile_bytes = (knobs.rows_depth + knobs.out_depth) * knobs.block_tile_rows * block_col_tiles * TILE_BYTES
    return index_bytes + tile_bytes


def col_tile_cap(knobs: Knobs) -> int:
    free = knobs.l1_budget_bytes - knobs.index_depth * knobs.block_tile_rows * INDEX_ROW_BYTES
    per_col_tile = (knobs.rows_depth + knobs.out_depth) * knobs.block_tile_rows * TILE_BYTES
    return free // per_col_tile


def compute_plan(index_shape: tuple, weight_shape: tuple, num_grid_cores: int, knobs: Knobs) -> Plan:
    """The exact selection function of op_design.md -> Regimes, over block-rows of BLOCK_TILE_ROWS."""
    tensor_images = index_shape[0] if len(index_shape) == 2 else 1
    token_extent = index_shape[-1]
    tile_rows_per_image = -(-token_extent // TILE_HEIGHT)
    tensor_tile_rows = tensor_images * tile_rows_per_image
    tensor_col_tiles = weight_shape[-1] // TILE_WIDTH
    block_rows = -(-tensor_tile_rows // knobs.block_tile_rows)

    cap = col_tile_cap(knobs)
    if cap < 1:
        raise ValueError(
            f"g6_embedding: one tile-column block needs {l1_bytes_per_core(knobs, 1)} B of L1 per core, "
            f"above the per-core bound of {knobs.l1_budget_bytes} B"
        )
    divisors = _divisors(tensor_col_tiles)
    q_l1 = min(q for q in divisors if tensor_col_tiles // q <= cap)
    q_fill = max(q for q in divisors if q <= max(1, num_grid_cores // block_rows))
    col_blocks_per_row = max(q_l1, q_fill)
    block_col_tiles = tensor_col_tiles // col_blocks_per_row
    total_blocks = block_rows * col_blocks_per_row
    segment_bytes = block_col_tiles * TILE_WIDTH * 2
    return Plan(
        regime="whole_row" if col_blocks_per_row == 1 else "column_blocked",
        col_blocks_per_row=col_blocks_per_row,
        block_col_tiles=block_col_tiles,
        tensor_tile_rows=tensor_tile_rows,
        tensor_col_tiles=tensor_col_tiles,
        total_blocks=total_blocks,
        num_cores=min(total_blocks, num_grid_cores),
        l1_bytes_per_core=l1_bytes_per_core(knobs, block_col_tiles),
        tensor_images=tensor_images,
        token_extent=token_extent,
        tile_rows_per_image=tile_rows_per_image,
        block_rows=block_rows,
        block_tile_rows=knobs.block_tile_rows,
        read_noc_split=(
            segment_bytes <= knobs.read_noc_split_max_segment_bytes
            and min(total_blocks, num_grid_cores) >= knobs.read_noc_split_min_cores
            and token_extent > 1
        ),
    )


def _tile_rows_in_run(plan: Plan, first_block: int, num_blocks: int) -> int:
    """Tile-rows the compute kernel tilizes for blocks [first_block, first_block + num_blocks).

    Only the last block-row can be short (tensor_tile_rows not a multiple of BLOCK_TILE_ROWS).
    """
    b = plan.block_tile_rows
    last_row_tile_rows = plan.tensor_tile_rows - (plan.block_rows - 1) * b
    last_row_first_block = (plan.block_rows - 1) * plan.col_blocks_per_row
    blocks_in_last_row = max(0, first_block + num_blocks - max(first_block, last_row_first_block))
    return num_blocks * b - blocks_in_last_row * (b - last_row_tile_rows)


def _assign_blocks(plan: Plan, grid):
    """(all_cores, [(core, first_block, num_blocks)]): contiguous block runs in row-wise core order."""
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
    assert first == plan.total_blocks, f"split covers {first} of {plan.total_blocks} blocks"
    return all_cores, assignment


class ProgramArtifacts(NamedTuple):
    plan: Plan
    reader: ttnn.KernelDescriptor
    writer: ttnn.KernelDescriptor
    compute: ttnn.KernelDescriptor
    cbs: list


def core_roles(plan: Plan, grid):
    """[(core, first_block, num_blocks, roles)]: every core of the program runs all three roles."""
    _, assignment = _assign_blocks(plan, grid)
    return [(core, first, count, ("reader", "compute", "writer")) for core, first, count in assignment]


def create_program_artifacts(
    input_tensor, weight, output, plan: Plan, knobs: Knobs, grid, device_zones: bool = False
) -> ProgramArtifacts:
    """Kernels and CBs of one plan. The address slots of the common runtime args are placeholders.

    device_zones builds the profiling zones into the kernels (the zone build); off, the kernels carry
    no zone code at all.
    """
    all_cores, assignment = _assign_blocks(plan, grid)
    defines = [("G6_DEVICE_ZONES", "1")] if device_zones else []
    b = knobs.block_tile_rows
    bct = plan.block_col_tiles

    def cb(index, data_format, page_size, num_pages):
        return ttnn.CBDescriptor(
            total_size=page_size * num_pages,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)
            ],
        )

    cbs = [
        cb(CB_INDEX_VALUES, ttnn.uint32, INDEX_ROW_BYTES, knobs.index_depth * b),
        # UInt16 on both tilize CBs keeps the helper off the float fast-tilize path, which rewrites
        # NaN payloads and the sign of zero; the bf16 bit patterns pass through as integers.
        cb(CB_GATHERED_ROWS, ttnn.uint16, TILE_BYTES, knobs.rows_depth * b * bct),
        cb(CB_OUTPUT_TILES, ttnn.uint16, TILE_BYTES, knobs.out_depth * b * bct),
    ]

    reader_ct = [CB_INDEX_VALUES, CB_GATHERED_ROWS, bct, b, knobs.index_depth, int(plan.read_noc_split)]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(weight).get_compile_time_args())
    writer_ct = [CB_OUTPUT_TILES, bct, b, TILE_BYTES]
    writer_ct.extend(ttnn.TensorAccessorArgs(output).get_compile_time_args())
    compute_ct = [CB_GATHERED_ROWS, CB_OUTPUT_TILES, bct]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    for ordinal, (core, first_block, num_blocks) in enumerate(assignment):
        tile_rotation = ordinal % bct if knobs.writer_bank_rotation else 0
        reader_rt[core.x][core.y] = [first_block, num_blocks]
        writer_rt[core.x][core.y] = [first_block, num_blocks, tile_rotation]
        compute_rt[core.x][core.y] = [_tile_rows_in_run(plan, first_block, num_blocks)]

    # Common runtime args: slots 0..1 (reader) and 0 (writer) are buffer addresses, patched per call.
    reader_common = [
        0,
        0,
        input_tensor.buffer_page_size(),
        weight.buffer_page_size(),
        plan.token_extent,
        plan.tile_rows_per_image,
        plan.tensor_tile_rows,
        plan.col_blocks_per_row,
    ]
    writer_common = [0, plan.tensor_col_tiles, plan.tensor_tile_rows, plan.col_blocks_per_row]

    # Reads on both NoCs need dynamic NoC mode on both data-movement kernels: a read barrier then
    # counts the responses of both RISCs on that NoC.
    if plan.read_noc_split:
        reader_config = ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC
        )
        writer_config = ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC
        )
    else:
        reader_config = ttnn.ReaderConfigDescriptor()
        writer_config = ttnn.WriterConfigDescriptor()

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_embedding_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        common_runtime_args=reader_common,
        config=reader_config,
        defines=defines,
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_embedding_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        common_runtime_args=writer_common,
        config=writer_config,
        defines=defines,
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_embedding_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False, dst_full_sync_en=False),
        defines=defines,
    )
    return ProgramArtifacts(plan=plan, reader=reader, writer=writer, compute=compute, cbs=cbs)


def bind_program_descriptor(artifacts: ProgramArtifacts, input_tensor, weight, output) -> ttnn.ProgramDescriptor:
    """Patch this call's three buffer addresses into the cached kernels; nothing else is rebuilt."""
    reader_common = list(artifacts.reader.common_runtime_args)
    reader_common[0] = input_tensor.buffer_address()
    reader_common[1] = weight.buffer_address()
    artifacts.reader.common_runtime_args = reader_common
    writer_common = list(artifacts.writer.common_runtime_args)
    writer_common[0] = output.buffer_address()
    artifacts.writer.common_runtime_args = writer_common
    return ttnn.ProgramDescriptor(
        kernels=[artifacts.reader, artifacts.writer, artifacts.compute],
        semaphores=[],
        cbs=artifacts.cbs,
    )
