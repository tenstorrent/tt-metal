# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for g6_concat: a reader (NCRISC/NOC0) and a writer (BRISC/NOC1), no compute.

Work split: contiguous ranges of the row-major output page index, balanced to +-1 page by
ttnn.split_work_to_cores(row_wise=True). Each core walks its range in blocks of `block_pages`
output pages through one double-buffered CB (op_design.md, Blocking Model).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---- Block knobs (single source of truth; every CB size and kernel arg derives from these) ----
BLOCK_BYTES_CAP = 128 * 1024  # L1 bytes of one block of output pages (L1 coexistence with model tensors)
OUTPUT_PAGES_DEPTH = 2  # blocks resident in cb_output_pages: reader fills i+1 while writer drains i
TRIM_ROWS_MAX = 16  # a page with v <= TRIM_ROWS_MAX valid rows moves only those rows
TRIM_ROWS_MAX_BLOCK_FLOAT = 0  # bfloat8_b keeps its exponents in a tile header: whole pages only
MAX_CORES = None  # None: the whole compute-with-storage grid

# ---- Mechanism caps ----
K_MAX = 64  # two common runtime-arg words per input under the 341-word per-kernel cap
L1_OP_BUDGET = 1347776  # per-core L1 the model leaves to the op (BB-05)

TILE_ROWS = 32
FACE_ROWS = 16

CB_OUTPUT_PAGES = 0
CB_GATHER_STAGING = 1

REGIME_PAGE_COPY = 0
REGIME_ROW_GATHER = 1
REGIME_NAMES = {REGIME_PAGE_COPY: "page_copy", REGIME_ROW_GATHER: "row_gather"}

# Common runtime-arg layout shared by both kernels' headers (kernels mirror these indices).
READER_CARG_INPUTS = 5  # [K, block_pages, out_Wt, out_Ht, v_last, (addr_k, extent_k) * K]
WRITER_CARG_OUT_ADDR = 0  # [out_addr, block_pages, out_Wt, out_Ht, v_last]


def _ct(x):
    return (x + TILE_ROWS - 1) // TILE_ROWS


def select_regime(shapes, d, layout):
    """row_gather iff a TILE h join puts the start of some input (other than the first) inside a tile."""
    r = len(shapes[0])
    if layout == ttnn.TILE_LAYOUT and d == r - 2:
        offset = 0
        for s in shapes[:-1]:
            offset += s[d]
            if offset % TILE_ROWS != 0:
                return "row_gather"
    return "page_copy"


def _page_copy_map(shapes, d):
    """(O, [C_k]): output page o*C + c comes from input k page o*C_k + (c - Coff_k)."""
    r = len(shapes[0])
    s0 = shapes[0]
    if d <= r - 3:
        outer = prod(s0[:d])
        inner = prod(s0[d + 1 : r - 2]) * _ct(s0[r - 2]) * _ct(s0[r - 1])
        return outer, [s[d] * inner for s in shapes]
    if d == r - 2:
        return prod(s0[: r - 2]), [_ct(s[d]) * _ct(s0[r - 1]) for s in shapes]
    return prod(s0[: r - 2]) * _ct(s0[r - 2]), [_ct(s[d]) for s in shapes]


def _trim_rows_max(dtype):
    return TRIM_ROWS_MAX_BLOCK_FLOAT if dtype == ttnn.bfloat8_b else TRIM_ROWS_MAX


def _row_bytes(page_bytes):
    # One 16-element face row; a tile is 64 face rows (4 faces x 16 rows).
    return page_bytes // (4 * FACE_ROWS)


@dataclass
class ProgramEntry:
    descriptor: ttnn.ProgramDescriptor
    reader: ttnn.KernelDescriptor
    writer: ttnn.KernelDescriptor
    reader_common: list
    writer_common: list
    kernel_input_indices: list  # position in `tensors` of each input the kernels read
    plan: dict = field(default_factory=dict)


def _core_grid(device):
    grid = device.compute_with_storage_grid_size()
    if MAX_CORES is not None and MAX_CORES < grid.x * grid.y:
        return ttnn.num_cores_to_corerangeset(MAX_CORES, grid, True)
    return grid


def build_program(tensors, output, d):
    shapes = [list(t.shape) for t in tensors]
    r = len(shapes[0])
    dtype = tensors[0].dtype
    layout = tensors[0].layout
    device = tensors[0].device()

    # Inputs with no extent along d contribute no pages; the kernels never see them.
    kernel_input_indices = [i for i, s in enumerate(shapes) if s[d] > 0]
    kernel_shapes = [shapes[i] for i in kernel_input_indices]
    kernel_tensors = [tensors[i] for i in kernel_input_indices]
    if not kernel_tensors:
        raise ValueError("g6_concat: every input is empty along dim; the output has no elements")
    num_inputs = len(kernel_tensors)

    out_shape = list(output.shape)
    page_bytes = output.buffer_page_size()
    if any(t.buffer_page_size() != page_bytes for t in kernel_tensors):
        raise RuntimeError("g6_concat: input and output page sizes differ")
    out_imgs = prod(out_shape[: r - 2])
    out_Ht = _ct(out_shape[r - 2])
    out_Wt = _ct(out_shape[r - 1])
    total_pages = output.buffer_num_pages()
    assert total_pages == out_imgs * out_Ht * out_Wt, (total_pages, out_imgs, out_Ht, out_Wt)
    last_valid_rows = out_shape[r - 2] - TILE_ROWS * (out_Ht - 1)
    row_bytes = _row_bytes(page_bytes)
    trim_rows_max = _trim_rows_max(dtype)

    regime_name = select_regime(kernel_shapes, d, layout)
    regime = REGIME_PAGE_COPY if regime_name == "page_copy" else REGIME_ROW_GATHER

    if regime == REGIME_PAGE_COPY:
        num_slices, pages_per_slice = _page_copy_map(kernel_shapes, d)
        pages_per_slice_total = sum(pages_per_slice)
        assert num_slices * pages_per_slice_total == total_pages, (num_slices, pages_per_slice, total_pages)
        input_extents = pages_per_slice
        # Every input page keeps its tile-row index inside the output page, so a page's valid rows
        # are the output page's: only the last input of an h join may end inside a tile.
        assert d != r - 2 or all(s[d] % TILE_ROWS == 0 for s in kernel_shapes[:-1])
    else:
        input_extents = [s[d] for s in kernel_shapes]

    # ---- Work split: contiguous output-page ranges over the grid ----
    grid = _core_grid(device)
    num_cores, all_cores, core_group_1, core_group_2, pages_core_1, pages_core_2 = ttnn.split_work_to_cores(
        grid, total_pages, True
    )
    block_pages_cap = max(1, BLOCK_BYTES_CAP // page_bytes)
    block_pages = min(pages_core_1, block_pages_cap)

    # ---- Circular buffers ----
    dram_read_alignment = ttnn.get_dram_alignment()
    has_staging = regime == REGIME_ROW_GATHER and row_bytes % dram_read_alignment != 0

    output_cb_bytes = OUTPUT_PAGES_DEPTH * block_pages * page_bytes
    cbs = [
        ttnn.CBDescriptor(
            total_size=output_cb_bytes,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_OUTPUT_PAGES, data_format=dtype, page_size=page_bytes)
            ],
        )
    ]
    # One mirror slot per page of a block, +1 page for the <= alignment-sized shift past the last slot.
    staging_cb_bytes = (block_pages + 1) * page_bytes if has_staging else 0
    if has_staging:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=staging_cb_bytes,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=CB_GATHER_STAGING, data_format=dtype, page_size=page_bytes)
                ],
            )
        )
    l1_total_bytes = output_cb_bytes + staging_cb_bytes

    # ---- Kernel arguments ----
    reader_ct = [
        regime,
        CB_OUTPUT_PAGES,
        CB_GATHER_STAGING,
        page_bytes,
        row_bytes,
        trim_rows_max,
        int(has_staging),
        dram_read_alignment,
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(kernel_tensors[0]).get_compile_time_args())
    writer_ct = [CB_OUTPUT_PAGES, page_bytes, row_bytes, trim_rows_max]
    writer_ct.extend(ttnn.TensorAccessorArgs(output).get_compile_time_args())

    reader_common = [num_inputs, block_pages, out_Wt, out_Ht, last_valid_rows]
    assert len(reader_common) == READER_CARG_INPUTS
    for t, extent in zip(kernel_tensors, input_extents):
        reader_common.extend([t.buffer_address(), extent])
    writer_common = [output.buffer_address(), block_pages, out_Wt, out_Ht, last_valid_rows]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    roles = []
    start = 0
    for group, pages_this_core in ((core_group_1, pages_core_1), (core_group_2, pages_core_2)):
        if pages_this_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            tile_row = (start // out_Wt) % out_Ht
            col = start % out_Wt
            if regime == REGIME_PAGE_COPY:
                slice_idx, page_in_slice = divmod(start, pages_per_slice_total)
                input_idx = 0
                while page_in_slice >= pages_per_slice[input_idx]:
                    page_in_slice -= pages_per_slice[input_idx]
                    input_idx += 1
                cursor = [slice_idx, input_idx, page_in_slice]
            else:
                cursor = [start // (out_Ht * out_Wt)]
            reader_rt[core.x][core.y] = [pages_this_core, tile_row, col, *cursor]
            writer_rt[core.x][core.y] = [pages_this_core, start, tile_row, col]
            roles.append(
                {
                    "core": (core.x, core.y),
                    "role": f"{regime_name} reader+writer",
                    "first_page": start,
                    "num_pages": pages_this_core,
                }
            )
            start += pages_this_core
    assert start == total_pages, (start, total_pages)

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_concat_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        common_runtime_args=reader_common,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_concat_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        common_runtime_args=writer_common,
        config=ttnn.WriterConfigDescriptor(),
    )
    descriptor = ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=cbs)

    plan = {
        "regime": regime_name,
        "num_cores": num_cores,
        "core_ranges": str(all_cores),
        "block_pages": block_pages,
        "depth": OUTPUT_PAGES_DEPTH,
        "total_pages": total_pages,
        "cb_output_pages_bytes": output_cb_bytes,
        "cb_gather_staging_bytes": staging_cb_bytes,
        "l1_total_bytes": l1_total_bytes,
        "roles": roles,
    }
    return ProgramEntry(
        descriptor=descriptor,
        reader=reader,
        writer=writer,
        reader_common=reader_common,
        writer_common=writer_common,
        kernel_input_indices=kernel_input_indices,
        plan=plan,
    )


def refresh_addresses(entry, tensors, output):
    """Rewrite only the buffer-address common runtime args of a cached program."""
    for slot, i in enumerate(entry.kernel_input_indices):
        entry.reader_common[READER_CARG_INPUTS + 2 * slot] = tensors[i].buffer_address()
    entry.writer_common[WRITER_CARG_OUT_ADDR] = output.buffer_address()
    entry.reader.common_runtime_args = entry.reader_common
    entry.writer.common_runtime_args = entry.writer_common
    entry.descriptor.kernels = [entry.reader, entry.writer]
