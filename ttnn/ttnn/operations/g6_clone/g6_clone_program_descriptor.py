# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_clone program descriptor: regime flat_page_range.

The flat page axis (every page-level tensor axis folded) is split into contiguous
page ranges over the compute grid. Each core stages blocks of `block_pages` whole
aligned pages in cb_page_staging: the reader fills a block behind one read
barrier, the writer drains it behind one flush.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# Per-core L1 the model leaves to the op (BB-05).
L1_BUDGET_BYTES = 1347776
# Upper bound on staged blocks per core; the realized depth is capped by the block count.
CB_DEPTH_KNOB = 2

CB_PAGE_STAGING = 0

READER_KERNEL_INDEX = 0
WRITER_KERNEL_INDEX = 1

REGIME = "flat_page_range"


def plan_key(input_tensor):
    """Everything the descriptor depends on except the two buffer addresses."""
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    memory_config = input_tensor.memory_config()
    return (
        device.id(),
        grid.x,
        grid.y,
        input_tensor.dtype,
        input_tensor.layout,
        memory_config.buffer_type,
        memory_config.memory_layout,
        input_tensor.buffer_num_pages(),
        input_tensor.buffer_aligned_page_size(),
    )


def g6_clone_plan(input_tensor):
    """The blocking plan a call of g6_clone on `input_tensor` runs (BB-08, BB-21)."""
    grid = input_tensor.device().compute_with_storage_grid_size()
    tensor_pages = input_tensor.buffer_num_pages()
    aligned_page_bytes = input_tensor.buffer_aligned_page_size()

    block_pages_l1_cap = L1_BUDGET_BYTES // (CB_DEPTH_KNOB * aligned_page_bytes)
    if block_pages_l1_cap < 1:
        raise RuntimeError(
            f"g6_clone: a page of {aligned_page_bytes} B does not fit {CB_DEPTH_KNOB} staging slots "
            f"in the {L1_BUDGET_BYTES} B per-core L1 budget"
        )

    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        core_pages_g1,
        core_pages_g2,
    ) = ttnn.split_work_to_cores(grid, tensor_pages, row_wise=True)

    # Group 1 holds the larger per-core count, so it bounds every core's assignment.
    core_pages_max = core_pages_g1
    block_pages = min(core_pages_max, block_pages_l1_cap)
    num_blocks_max = -(-core_pages_max // block_pages)
    cb_depth_blocks = min(CB_DEPTH_KNOB, num_blocks_max)

    roles = []
    start_page = 0
    for group, core_pages in ((core_group_1, core_pages_g1), (core_group_2, core_pages_g2)):
        if core_pages == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            roles.append((core, "reader+writer", start_page, core_pages))
            start_page += core_pages
    if start_page != tensor_pages:
        raise RuntimeError(f"g6_clone: work split covers {start_page} of {tensor_pages} pages")

    return {
        "regime": REGIME,
        "grid": (grid.x, grid.y),
        "num_cores": num_cores,
        "core_ranges": all_cores,
        "tensor_pages": tensor_pages,
        "aligned_page_bytes": aligned_page_bytes,
        "core_pages_g1": core_pages_g1,
        "core_pages_g2": core_pages_g2,
        "block_pages": block_pages,
        "cb_depth_blocks": cb_depth_blocks,
        "num_blocks_max": num_blocks_max,
        "roles": roles,
    }


def create_program_descriptor(input_tensor, output_tensor, plan):
    page_bytes = plan["aligned_page_bytes"]
    block_pages = plan["block_pages"]
    all_cores = plan["core_ranges"]

    # The payload is raw bytes; data_format only labels the descriptor, no unpacker reads it.
    cb_page_staging = ttnn.CBDescriptor(
        total_size=plan["cb_depth_blocks"] * block_pages * page_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_PAGE_STAGING,
                data_format=input_tensor.dtype,
                page_size=page_bytes,
            )
        ],
    )

    reader_ct_args = [CB_PAGE_STAGING, block_pages]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct_args = [CB_PAGE_STAGING, block_pages]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    for core, _, start_page, core_pages in plan["roles"]:
        reader_rt_args[core.x][core.y] = [start_page, core_pages]
        writer_rt_args[core.x][core.y] = [start_page, core_pages]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_clone_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        common_runtime_args=[input_tensor.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_clone_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        common_runtime_args=[output_tensor.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )

    kernels = [None, None]
    kernels[READER_KERNEL_INDEX] = reader_kernel
    kernels[WRITER_KERNEL_INDEX] = writer_kernel
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb_page_staging])


def set_buffer_addresses(program_descriptor, input_address, output_address):
    """Patch the per-call buffer addresses (common RT arg 0 of reader and writer)."""
    kernels = program_descriptor.kernels
    kernels[READER_KERNEL_INDEX].common_runtime_args = [input_address]
    kernels[WRITER_KERNEL_INDEX].common_runtime_args = [output_address]
