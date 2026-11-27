# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_embedding ProgramDescriptor: three CBs, reader / compute / writer over one core set.

Every extent comes from the Plan built by select_plan; nothing here restates a block size. Buffer
addresses live only in common runtime args so a cached descriptor is reused by patching them.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

CB_INDEX_VALUES = 0  # reader-private: index values of the block being gathered + the prefetched one
CB_GATHERED_ROWS = 1  # row-major table-row segments, UInt16 (bf16 bits)
CB_OUTPUT_TILES = 2  # tilized output tiles, UInt16 (bf16 bits)

READER_KERNEL = 0
WRITER_KERNEL = 1
COMPUTE_KERNEL = 2

# Common runtime-arg slots holding buffer addresses.
READER_INDEX_ADDR_SLOT = 0
READER_WEIGHT_ADDR_SLOT = 1
WRITER_OUTPUT_ADDR_SLOT = 0

TILE_DIM = 32
INDEX_BYTES = 4
VALUE_BYTES = 2


def _core_assignment(plan):
    """(all_cores, [(core, first_block, num_blocks), ...]) as contiguous block ranges, row-wise."""
    grid = ttnn.CoreCoord(plan.grid_x, plan.grid_y)
    (
        _,
        all_cores,
        core_group_1,
        core_group_2,
        blocks_per_core_1,
        blocks_per_core_2,
    ) = ttnn.split_work_to_cores(grid, plan.total_blocks, row_wise=True)

    assignment = []
    first_block = 0
    for group, blocks_per_core in ((core_group_1, blocks_per_core_1), (core_group_2, blocks_per_core_2)):
        if blocks_per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, first_block, blocks_per_core))
            first_block += blocks_per_core
    assert first_block == plan.total_blocks, f"assigned {first_block} of {plan.total_blocks} blocks"
    return all_cores, assignment


def _cb(buffer_index, data_format, page_size, num_pages, core_ranges):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=buffer_index, data_format=data_format, page_size=page_size)
        ],
    )


def create_program_descriptor(plan, input_tensor, weight, output_tensor) -> ttnn.ProgramDescriptor:
    all_cores, assignment = _core_assignment(plan)

    tile_bytes = TILE_DIM * TILE_DIM * VALUE_BYTES
    index_slot_bytes = plan.block_tile_rows * TILE_DIM * INDEX_BYTES
    block_pages = plan.block_tile_rows * plan.block_col_tiles

    cbs = [
        _cb(CB_INDEX_VALUES, ttnn.uint32, index_slot_bytes, plan.index_depth, all_cores),
        # UInt16, not bfloat16: a Float16_b CB selects the float tilize path, which rewrites NaN
        # payloads and the sign of zero. As UInt16 the bf16 bits pass through 16-bit DEST untouched.
        _cb(CB_GATHERED_ROWS, ttnn.uint16, tile_bytes, plan.rows_depth * block_pages, all_cores),
        _cb(CB_OUTPUT_TILES, ttnn.uint16, tile_bytes, plan.out_depth * block_pages, all_cores),
    ]

    reader_ct_args = [
        CB_INDEX_VALUES,
        CB_GATHERED_ROWS,
        plan.block_col_tiles,
        plan.block_tile_rows,
        plan.index_depth,
        plan.col_blocks_per_row,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(weight).get_compile_time_args())
    reader_common_args = [
        input_tensor.buffer_address(),
        weight.buffer_address(),
        plan.tensor_token_extent,
        plan.tile_rows_per_image,
        plan.tensor_tile_rows,
    ]

    writer_ct_args = [
        CB_OUTPUT_TILES,
        plan.block_col_tiles,
        plan.block_tile_rows,
        plan.col_blocks_per_row,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_common_args = [
        output_tensor.buffer_address(),
        plan.tensor_tile_rows,
        plan.tensor_col_tiles,
    ]

    compute_ct_args = [
        CB_GATHERED_ROWS,
        CB_OUTPUT_TILES,
        plan.block_col_tiles,
        plan.block_tile_rows,
    ]

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()
    for core, first_block, num_blocks in assignment:
        reader_rt_args[core.x][core.y] = [first_block, num_blocks]
        writer_rt_args[core.x][core.y] = [first_block, num_blocks]
        compute_rt_args[core.x][core.y] = [num_blocks]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_embedding_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        common_runtime_args=reader_common_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_embedding_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        common_runtime_args=writer_common_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    compute_config = ttnn.ComputeConfigDescriptor()
    compute_config.fp32_dest_acc_en = False  # 16-bit DEST: the UInt16 passthrough relies on it
    compute_config.dst_full_sync_en = False
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_embedding_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=compute_config,
    )

    kernels = [None, None, None]
    kernels[READER_KERNEL] = reader_kernel
    kernels[WRITER_KERNEL] = writer_kernel
    kernels[COMPUTE_KERNEL] = compute_kernel
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def patch_buffer_addresses(program_descriptor, input_tensor, weight, output_tensor):
    """Point a cached descriptor at this call's tensors (the only per-call state it carries)."""
    kernels = program_descriptor.kernels
    reader_common_args = kernels[READER_KERNEL].common_runtime_args
    reader_common_args[READER_INDEX_ADDR_SLOT] = input_tensor.buffer_address()
    reader_common_args[READER_WEIGHT_ADDR_SLOT] = weight.buffer_address()
    kernels[WRITER_KERNEL].common_runtime_args[WRITER_OUTPUT_ADDR_SLOT] = output_tensor.buffer_address()


def core_roles(plan):
    """BB-21 role dump: {(x, y): (roles, first_block, num_blocks)}; every core runs all three roles."""
    _, assignment = _core_assignment(plan)
    roles = ("reader", "compute", "writer")
    return {(core.x, core.y): (roles, first_block, num_blocks) for core, first_block, num_blocks in assignment}
