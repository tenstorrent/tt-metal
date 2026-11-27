# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_clone program descriptor: regime `flat_page_range`.

Reader (NCRISC, NoC0) -> cb_page_staging -> writer (BRISC, NoC1); no compute kernel, so the
bytes never pass through an unpacker, DEST or a packer. The flat page axis is split into
contiguous page ranges over the compute grid; each core walks its range in blocks of
`block_pages` pages.

Every block knob is defined once here and every dependent quantity is derived from it:
`block_pages` and `cb_depth_blocks` set the CB capacity, the push/pop quantum (CT arg) and the
per-core block count the kernels derive from `num_pages_this_core`.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

REGIME = "flat_page_range"

# Buffer-depth knob of cb_page_staging: slots of `block_pages` pages when a core has more than
# one block, so the reader fills block k+1 while the writer drains block k.
CB_DEPTH_KNOB = 2

# L1 the model environment leaves to the op on every core (BB-05).
L1_BUDGET_BYTES = 1347776

CB_PAGE_STAGING = 0

# Order of the kernels in the descriptor; patch_addresses indexes by it.
READER_KERNEL_IDX = 0
WRITER_KERNEL_IDX = 1


def compute_plan(device, tensor_pages: int, aligned_page_bytes: int) -> dict:
    """Work split and block knobs for one (grid, page count, page size) triple."""
    grid = device.compute_with_storage_grid_size()

    block_pages_l1_cap = L1_BUDGET_BYTES // (CB_DEPTH_KNOB * aligned_page_bytes)
    if block_pages_l1_cap < 1:
        raise RuntimeError(
            f"g6_clone: a page of {aligned_page_bytes} B does not fit {CB_DEPTH_KNOB} staging slots "
            f"in the {L1_BUDGET_BYTES} B per-core L1 budget"
        )

    num_cores, all_cores, core_group_1, core_group_2, core_pages_g1, core_pages_g2 = ttnn.split_work_to_cores(
        grid, tensor_pages, row_wise=True
    )
    core_pages_max = core_pages_g1

    block_pages = min(core_pages_max, block_pages_l1_cap)
    num_blocks_max = -(-core_pages_max // block_pages)
    cb_depth_blocks = min(CB_DEPTH_KNOB, num_blocks_max)

    # split_work_to_cores(row_wise=True) hands group 1 the first cores in row-major order and
    # group 2 the rest; start_page is the running page count in that same order.
    num_cores_g1 = core_group_1.num_cores()
    roles = []
    start_page = 0
    for idx, core in enumerate(ttnn.grid_to_cores(num_cores, grid.x, grid.y, True)):
        num_pages = core_pages_g1 if idx < num_cores_g1 else core_pages_g2
        roles.append((core, "reader+writer", start_page, num_pages))
        start_page += num_pages
    if start_page != tensor_pages:
        raise RuntimeError(f"g6_clone: work split covers {start_page} of {tensor_pages} pages")

    return {
        "regime": REGIME,
        "grid": (grid.x, grid.y),
        "num_cores": num_cores,
        "core_ranges": all_cores,
        "core_pages_g1": core_pages_g1,
        "core_pages_g2": core_pages_g2,
        "block_pages": block_pages,
        "cb_depth_blocks": cb_depth_blocks,
        "num_blocks_max": num_blocks_max,
        "roles": roles,
        "tensor_pages": tensor_pages,
        "aligned_page_bytes": aligned_page_bytes,
    }


def create_program_descriptor(
    input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor, plan: dict
) -> ttnn.ProgramDescriptor:
    """Descriptor for `plan`. Buffer addresses sit in common RT args, slot 0 of each kernel,
    so a cached descriptor is reused by patching two values (see patch_addresses)."""
    all_cores = plan["core_ranges"]
    block_pages = plan["block_pages"]
    aligned_page_bytes = plan["aligned_page_bytes"]

    cb_page_staging = ttnn.CBDescriptor(
        total_size=plan["cb_depth_blocks"] * block_pages * aligned_page_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_PAGE_STAGING,
                data_format=input_tensor.dtype,
                page_size=aligned_page_bytes,
            )
        ],
    )

    input_accessor_args = ttnn.TensorAccessorArgs(input_tensor)
    output_accessor_args = ttnn.TensorAccessorArgs(output_tensor)
    # Common RT slot 0 holds the buffer address; an interleaved accessor adds no common RT args.
    if len(input_accessor_args.get_common_runtime_args()) or len(output_accessor_args.get_common_runtime_args()):
        raise RuntimeError("g6_clone: interleaved TensorAccessorArgs unexpectedly carry common runtime args")

    reader_ct_args = [CB_PAGE_STAGING, block_pages]
    reader_ct_args.extend(input_accessor_args.get_compile_time_args())
    writer_ct_args = [CB_PAGE_STAGING, block_pages]
    writer_ct_args.extend(output_accessor_args.get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    for core, _, start_page, num_pages in plan["roles"]:
        reader_rt_args[core.x][core.y] = [start_page, num_pages]
        writer_rt_args[core.x][core.y] = [start_page, num_pages]

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

    # List order is READER_KERNEL_IDX, WRITER_KERNEL_IDX.
    return ttnn.ProgramDescriptor(kernels=[reader_kernel, writer_kernel], semaphores=[], cbs=[cb_page_staging])


def patch_addresses(descriptor: ttnn.ProgramDescriptor, input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor) -> None:
    """Write this call's buffer addresses into a cached descriptor, in place."""
    kernels = descriptor.kernels
    kernels[READER_KERNEL_IDX].common_runtime_args[0] = input_tensor.buffer_address()
    kernels[WRITER_KERNEL_IDX].common_runtime_args[0] = output_tensor.buffer_address()
