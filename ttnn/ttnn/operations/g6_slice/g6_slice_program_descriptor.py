# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_slice program descriptor: block plan (single source of every knob), CBs, kernels, args."""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32

# Regime ids, shared with the kernels through a CT arg (g6_slice_common.hpp mirrors them).
REGIME_PAGE_COPY = 0
REGIME_ROW_SHIFT = 1
REGIME_ELEMENT_SHIFT = 2

# BB-05: L1 the model leaves to the op on every core; every op buffer on a core counts against it.
L1_OP_BUDGET = 1347776
MAX_BUFFER_DEPTH = 2
# Op L1 that does not scale with block_pages (begin stick, L1-resident out share): none in Phase 0.
FIXED_BYTES = 0

CB_STAGED_TILES = 0
CB_ASSEMBLED_TILES = 1

# Common runtime-arg slots (identical layout for reader and writer; g6_slice_common.hpp reads them).
COMMON_ARG_FIELDS = (
    "tensor_addr",
    "M_D1",
    "out_h_tiles",
    "out_w_tiles",
    "N_D1",
    "in_h_tiles",
    "in_w_tiles",
    "b_D0",
    "b_D1",
    "hb",
    "wb",
    "row_offset",
    "col_offset",
    "M_H",
    "M_W",
)


def _div_up(a, b):
    return -(-a // b)


def plan_blocks(device, geometry, page_bytes):
    """Work split + block knobs of op_design.md "Blocking Model" (the only place they are computed)."""
    regime = geometry["regime"]
    below = int(geometry["needs_below_any"])
    right = int(geometry["needs_right_any"])
    if regime == REGIME_PAGE_COPY:
        sources_per_output, assembled_per_output = 1, 0
    elif regime == REGIME_ROW_SHIFT:
        sources_per_output, assembled_per_output = 1 + below, 0
    else:
        sources_per_output, assembled_per_output = (1 + right) * (1 + below), 1 + below

    grid = device.compute_with_storage_grid_size()
    num_cores, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(
        grid, geometry["total_out_pages"], row_wise=True
    )

    pages_per_core = per_core_1
    per_page_bytes = MAX_BUFFER_DEPTH * sources_per_output * page_bytes + assembled_per_output * page_bytes
    block_pages_cap = (L1_OP_BUDGET - FIXED_BYTES) // per_page_bytes
    num_blocks_max = _div_up(pages_per_core, block_pages_cap)
    block_pages = _div_up(pages_per_core, num_blocks_max)
    buffer_depth = 1 if num_blocks_max == 1 else MAX_BUFFER_DEPTH

    assignment = []
    start = 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, start, per_core))
            start += per_core
    assert start == geometry["total_out_pages"]

    return {
        "num_cores": num_cores,
        "all_cores": all_cores,
        "assignment": assignment,
        "pages_per_core": pages_per_core,
        "block_pages": block_pages,
        "buffer_depth": buffer_depth,
        "sources_per_output": sources_per_output,
        "assembled_per_output": assembled_per_output,
        "page_bytes": page_bytes,
    }


def _cb(index, num_pages, page_bytes, dtype, cores):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_bytes,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes)],
    )


def create_program_descriptor(input_tensor, output_tensor, geometry, blocks):
    cores = blocks["all_cores"]
    page_bytes = blocks["page_bytes"]
    block_pages = blocks["block_pages"]
    sources_per_output = blocks["sources_per_output"]
    assembled_per_output = blocks["assembled_per_output"]

    cbs = [
        _cb(
            CB_STAGED_TILES,
            blocks["buffer_depth"] * sources_per_output * block_pages,
            page_bytes,
            input_tensor.dtype,
            cores,
        )
    ]
    if assembled_per_output:
        cbs.append(_cb(CB_ASSEMBLED_TILES, assembled_per_output * block_pages, page_bytes, output_tensor.dtype, cores))

    # CT: everything that changes kernel text or CB layout. Begins, extents, ranges are RT.
    shared_ct_args = [
        CB_STAGED_TILES,
        CB_ASSEMBLED_TILES,
        block_pages,
        geometry["regime"],
        sources_per_output,
        int(geometry["needs_right_any"]),
        int(geometry["needs_below_any"]),
        page_bytes,
        input_tensor.element_size(),
    ]
    reader_ct_args = shared_ct_args + list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct_args = shared_ct_args + list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    geometry_args = [geometry[f] for f in COMMON_ARG_FIELDS[1:]]
    reader_common_args = [input_tensor.buffer_address()] + geometry_args
    writer_common_args = [output_tensor.buffer_address()] + geometry_args

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    for core, start_page, num_pages in blocks["assignment"]:
        reader_rt_args[core.x][core.y] = [start_page, num_pages]
        writer_rt_args[core.x][core.y] = [start_page, num_pages]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_slice_reader.cpp"),
        core_ranges=cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        common_runtime_args=reader_common_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_slice_writer.cpp"),
        core_ranges=cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        common_runtime_args=writer_common_args,
        config=ttnn.WriterConfigDescriptor(),
    )
    # No compute kernel: the op moves bytes only (op_design.md, Classification).
    return ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=cbs)
