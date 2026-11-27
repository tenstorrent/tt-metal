# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_slice ProgramDescriptor: one reader (NCRISC/NoC0) and one writer (BRISC/NoC1) over the used
cores, joined by cb_slice_pages; no compute kernel.

Compile-time args carry shapes, block knobs and regime flags; begins and buffer addresses are
common runtime args and the per-core page range is a per-core runtime arg, so one compiled program
serves every begin of a shape class.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

CB_SLICE_PAGES = 0
CB_RIGHT_COL_TILE = 1


@dataclass
class CachedProgram:
    descriptor: ttnn.ProgramDescriptor
    reader_kernel: ttnn.KernelDescriptor
    writer_kernel: ttnn.KernelDescriptor
    reader_begins: list
    writer_begins: list


def _common_args(address, begins):
    return [address, *begins]


def set_buffer_addresses(cached: CachedProgram, input_tensor, output_tensor):
    """Point a cached program at new x / out buffers. ProgramDescriptor.kernels holds copies, so the
    kernels are re-assigned after their address slot is rewritten."""
    cached.reader_kernel.common_runtime_args = _common_args(input_tensor.buffer_address(), cached.reader_begins)
    cached.writer_kernel.common_runtime_args = _common_args(output_tensor.buffer_address(), cached.writer_begins)
    cached.descriptor.kernels = [cached.reader_kernel, cached.writer_kernel]


def create_program_descriptor(input_tensor, output_tensor, plan) -> CachedProgram:
    all_cores = plan["all_cores"]
    page_bytes = plan["page_bytes"]
    block_pages = plan["block_pages"]
    w_shift = plan["regime"] == "col_realign"
    row_runs = plan["regime"] != "page_copy"

    cbs = [
        ttnn.CBDescriptor(
            total_size=plan["cb_depth"] * block_pages * page_bytes,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_SLICE_PAGES, data_format=input_tensor.dtype, page_size=page_bytes
                )
            ],
        )
    ]
    if w_shift:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=block_pages * page_bytes,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_RIGHT_COL_TILE, data_format=input_tensor.dtype, page_size=page_bytes
                    )
                ],
            )
        )

    shape4, out4 = plan["shape4"], plan["out4"]
    reader_ct_args = [
        CB_SLICE_PAGES,
        CB_RIGHT_COL_TILE,
        page_bytes,
        block_pages,
        int(row_runs),
        int(w_shift),
        plan["tensor_in_ht_tiles"],
        plan["tensor_in_wt_tiles"],
        plan["tensor_out_ht_tiles"],
        plan["tensor_out_wt_tiles"],
        shape4[1],
        out4[1],
        plan["rows_last_tile_row"],
        plan["cols_last_tile_col"],
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    writer_ct_args = [
        CB_SLICE_PAGES,
        page_bytes,
        block_pages,
        int(row_runs),
        plan["tensor_out_ht_tiles"],
        plan["tensor_out_wt_tiles"],
        plan["rows_last_tile_row"],
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    begins4 = plan["begins4"]
    reader_begins = [
        begins4[0],
        begins4[1],
        plan["first_input_tile_row"],
        plan["row_offset_in_tile"],
        plan["first_input_tile_col"],
        plan["col_offset_in_tile"],
    ]
    writer_begins = [plan["row_offset_in_tile"]]

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    for a in plan["assignment"]:
        core = a["core"]
        o0, o1, th, tw = a["coord"]
        reader_rt_args[core.x][core.y] = [a["pages"], o0, o1, th, tw]
        writer_rt_args[core.x][core.y] = [a["start_page"], a["pages"], th, tw]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_slice_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        common_runtime_args=_common_args(input_tensor.buffer_address(), reader_begins),
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_slice_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        common_runtime_args=_common_args(output_tensor.buffer_address(), writer_begins),
        config=ttnn.WriterConfigDescriptor(),
    )

    descriptor = ttnn.ProgramDescriptor(kernels=[reader_kernel, writer_kernel], semaphores=[], cbs=cbs)
    return CachedProgram(descriptor, reader_kernel, writer_kernel, reader_begins, writer_begins)
