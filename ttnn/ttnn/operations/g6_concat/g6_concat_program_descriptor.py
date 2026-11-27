# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_concat program descriptor: plan (regime, work split, block size, L1) and ProgramDescriptor.

Blocking model (op_design.md): the work is split by output page into contiguous ranges over
min(P_out, grid) cores; each core walks its range in blocks of `block_pages` pages through one
double-buffered reader->writer CB. Two readers fill a block: `page_copy` (each output page is one
input page) and `row_gather` (h joins with unaligned offsets, each output page assembled from
row-range reads). One writer serves both.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# --- Blocking knobs (single source of truth) -------------------------------------------------
# Byte cap of one block; block_pages = min(pages of the most-loaded core, cap // page_bytes).
BLOCK_BYTES_CAP = 128 * 1024
# Blocks resident in cb_output_pages: the reader fills block i+1 while the writer stores block i.
OUTPUT_PAGES_DEPTH = 2
# Optional cap on the number of cores; None uses the whole compute grid.
MAX_CORES = None
# Valid-row threshold under which a partially valid tile row moves only rows [0, v).
# bfloat8_b pages move whole: its exponents sit in a header, not next to the rows.
TRIM_ROWS_MAX = {
    ttnn.bfloat16: 16,
    ttnn.float32: 16,
    ttnn.bfloat8_b: 0,
}

# --- Mechanism caps --------------------------------------------------------------------------
# Two common runtime words per input; 341 words per core are shared by unique and common args.
K_MAX = 64
# L1 the model leaves to the op on every core (BB-05).
L1_OP_BUDGET = 1347776

TILE_ROWS = 32
FACE_ROWS = 16

CB_OUTPUT_PAGES = 0
CB_GATHER_STAGING = 1

# Common runtime-arg layout (kernels read the same indices).
READER_HEADER_WORDS = 5  # block_pages, K, out_Wt, out_Ht, v_last | N
WRITER_ADDR_INDEX = 0


def canonical_dim(dim, rank):
    return dim + rank if dim < 0 else dim


def tile_count(x):
    return (x + TILE_ROWS - 1) // TILE_ROWS


def _prod(values):
    out = 1
    for v in values:
        out *= v
    return out


def select_regime(shapes, d, layout):
    """`row_gather` iff a TILE h join has a join offset o_1..o_{K-1} off a tile boundary."""
    r = len(shapes[0])
    if layout == ttnn.TILE_LAYOUT and d == r - 2:
        offset = 0
        for s in shapes[:-1]:
            offset += s[d]
            if offset % TILE_ROWS != 0:
                return "row_gather"
    return "page_copy"


@dataclass
class Plan:
    regime: str
    d: int
    shapes: list
    dtype: object
    page_bytes: int
    face_row_bytes: int
    face_bytes: int
    trim_rows_max: int
    out_shape: list
    out_imgs: int
    out_Ht: int
    out_Wt: int
    v_last: int
    num_out_pages: int
    num_cores: int
    all_cores: object
    assignment: list  # (core, start_page, num_pages)
    block_pages: int
    depth: int
    needs_staging: bool
    input_l1_mask: int
    output_in_l1: bool
    cb_output_pages_bytes: int
    cb_gather_staging_bytes: int
    output_l1_bytes_per_bank: int
    l1_total_bytes: int
    # page_copy address map: O outer slices, C_k pages of input k per slice.
    outer_slices: int = 0
    pages_per_slice: list = field(default_factory=list)

    def report(self):
        """Plan report (BB-08/BB-21): what a call selected, and each core's role."""
        role = f"{self.regime} reader+writer"
        return {
            "regime": self.regime,
            "num_cores": self.num_cores,
            "core_ranges": str(self.all_cores),
            "block_pages": self.block_pages,
            "depth": self.depth,
            "cb_output_pages_bytes": self.cb_output_pages_bytes,
            "cb_gather_staging_bytes": self.cb_gather_staging_bytes,
            "l1_total_bytes": self.l1_total_bytes,
            "cores": [
                {"core": (core.x, core.y), "role": role, "pages": (start, start + count)}
                for core, start, count in self.assignment
            ],
        }


def _work_split(device, num_out_pages):
    grid = device.compute_with_storage_grid_size()
    if MAX_CORES is not None and MAX_CORES < grid.x * grid.y:
        cores = ttnn.num_cores_to_corerangeset(MAX_CORES, grid, True)
        split = ttnn.split_work_to_cores(cores, num_out_pages, row_wise=True)
    else:
        split = ttnn.split_work_to_cores(grid, num_out_pages, row_wise=True)
    num_cores, all_cores, core_group_1, core_group_2, n1, n2 = split
    assignment = []
    start = 0
    for group, per_core in ((core_group_1, n1), (core_group_2, n2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, start, per_core))
            start += per_core
    assert start == num_out_pages, f"g6_concat: work split covers {start} of {num_out_pages} pages"
    return num_cores, all_cores, assignment, n1


def make_plan(tensors, d, output_memory_config):
    """Host plan of one call. Pure host math; dispatches nothing."""
    device = tensors[0].device()
    shapes = [list(t.shape) for t in tensors]
    r = len(shapes[0])
    dtype = tensors[0].dtype
    layout = tensors[0].layout
    page_bytes = tensors[0].buffer_page_size()
    # Bytes of one 16-element face row. bfloat8_b has no addressable face rows (shared exponents in
    # a header); it never trims and never gathers, so 0 marks it unused.
    element_bytes = {ttnn.bfloat16: 2, ttnn.float32: 4}.get(dtype)
    face_row_bytes = FACE_ROWS * element_bytes if element_bytes is not None else 0
    face_bytes = FACE_ROWS * face_row_bytes
    trim_rows_max = TRIM_ROWS_MAX[dtype]

    regime = select_regime(shapes, d, layout)

    out_shape = list(shapes[0])
    out_shape[d] = sum(s[d] for s in shapes)
    out_imgs = _prod(out_shape[: r - 2])
    out_Ht = tile_count(out_shape[r - 2])
    out_Wt = tile_count(out_shape[r - 1])
    v_last = out_shape[r - 2] - TILE_ROWS * (out_Ht - 1)
    num_out_pages = out_imgs * out_Ht * out_Wt

    outer_slices = 0
    pages_per_slice = []
    if regime == "page_copy":
        H, W = shapes[0][r - 2], shapes[0][r - 1]
        if d <= r - 3:
            outer_slices = _prod(shapes[0][:d])
            pages_per_slice = [s[d] * _prod(s[d + 1 : r - 2]) * tile_count(H) * tile_count(W) for s in shapes]
        elif d == r - 2:
            outer_slices = _prod(shapes[0][: r - 2])
            pages_per_slice = [tile_count(s[d]) * tile_count(W) for s in shapes]
            # Only the last input may end mid tile: its padding rows become the output's.
            assert all(s[d] % TILE_ROWS == 0 for s in shapes[:-1])
        else:
            outer_slices = _prod(shapes[0][: r - 2]) * tile_count(H)
            pages_per_slice = [tile_count(s[d]) for s in shapes]
        assert outer_slices * sum(pages_per_slice) == num_out_pages
    else:
        assert dtype != ttnn.bfloat8_b, "row_gather moves face rows; bfloat8_b exponents need row_gather_bfp8"

    num_cores, all_cores, assignment, n1 = _work_split(device, num_out_pages)
    block_pages_cap = max(1, BLOCK_BYTES_CAP // page_bytes)
    block_pages = min(n1, block_pages_cap)

    input_l1_mask = 0
    for k, t in enumerate(tensors):
        if t.memory_config().buffer_type == ttnn.BufferType.L1:
            input_l1_mask |= 1 << k
    any_dram_input = input_l1_mask != (1 << len(tensors)) - 1
    dram_read_alignment = ttnn.get_dram_alignment()
    needs_staging = regime == "row_gather" and any_dram_input and face_row_bytes % dram_read_alignment != 0

    output_in_l1 = output_memory_config.buffer_type == ttnn.BufferType.L1
    cb_output_pages_bytes = OUTPUT_PAGES_DEPTH * block_pages * page_bytes
    cb_gather_staging_bytes = (block_pages + 1) * page_bytes if needs_staging else 0
    output_l1_bytes_per_bank = 0
    if output_in_l1:
        # Interleaved L1 places one bank on each worker core.
        grid = device.compute_with_storage_grid_size()
        num_l1_banks = grid.x * grid.y
        output_l1_bytes_per_bank = -(-num_out_pages // num_l1_banks) * page_bytes
    l1_total_bytes = cb_output_pages_bytes + cb_gather_staging_bytes + output_l1_bytes_per_bank
    if l1_total_bytes > L1_OP_BUDGET:
        raise RuntimeError(
            f"g6_concat: plan needs {l1_total_bytes} B of L1 per core, over the {L1_OP_BUDGET} B the model leaves"
        )

    return Plan(
        regime=regime,
        d=d,
        shapes=shapes,
        dtype=dtype,
        page_bytes=page_bytes,
        face_row_bytes=face_row_bytes,
        face_bytes=face_bytes,
        trim_rows_max=trim_rows_max,
        out_shape=out_shape,
        out_imgs=out_imgs,
        out_Ht=out_Ht,
        out_Wt=out_Wt,
        v_last=v_last,
        num_out_pages=num_out_pages,
        num_cores=num_cores,
        all_cores=all_cores,
        assignment=assignment,
        block_pages=block_pages,
        depth=OUTPUT_PAGES_DEPTH,
        needs_staging=needs_staging,
        input_l1_mask=input_l1_mask,
        output_in_l1=output_in_l1,
        cb_output_pages_bytes=cb_output_pages_bytes,
        cb_gather_staging_bytes=cb_gather_staging_bytes,
        output_l1_bytes_per_bank=output_l1_bytes_per_bank,
        l1_total_bytes=l1_total_bytes,
        outer_slices=outer_slices,
        pages_per_slice=pages_per_slice,
    )


def _page_copy_start_cursor(plan, start_page):
    """(o, k, c') of an output page under the page_copy map."""
    C = sum(plan.pages_per_slice)
    o, c = divmod(start_page, C)
    for k, C_k in enumerate(plan.pages_per_slice):
        if c < C_k:
            return o, k, c
        c -= C_k
    raise AssertionError("unreachable: c < C")


def reader_common_args(plan, tensors):
    if plan.regime == "page_copy":
        header = [plan.block_pages, len(tensors), plan.out_Wt, plan.out_Ht, plan.v_last]
        per_input = plan.pages_per_slice
    else:
        r = len(plan.out_shape)
        header = [plan.block_pages, len(tensors), plan.out_Wt, plan.out_Ht, plan.out_shape[r - 2]]
        per_input = [s[plan.d] for s in plan.shapes]
    assert len(header) == READER_HEADER_WORDS
    args = list(header)
    for t, value in zip(tensors, per_input):
        args.extend([t.buffer_address(), value])
    return args


def reader_address_indices(num_inputs):
    return [READER_HEADER_WORDS + 2 * k for k in range(num_inputs)]


def writer_common_args(plan, output_tensor):
    args = [output_tensor.buffer_address(), plan.block_pages, plan.out_Wt, plan.out_Ht, plan.v_last]
    assert args[WRITER_ADDR_INDEX] == output_tensor.buffer_address()
    return args


def _input_accessor_ct_args(tensors, plan):
    # One DRAM and one L1 accessor-args set; a kind with no input borrows the other kind's args,
    # and the kernel never instantiates reads through it (the CT mask has no such bit).
    dram_src = next((t for k, t in enumerate(tensors) if not (plan.input_l1_mask >> k) & 1), tensors[0])
    l1_src = next((t for k, t in enumerate(tensors) if (plan.input_l1_mask >> k) & 1), tensors[0])
    args = list(ttnn.TensorAccessorArgs(dram_src).get_compile_time_args())
    args.extend(ttnn.TensorAccessorArgs(l1_src).get_compile_time_args())
    return args


def program_hash(key):
    digest = hashlib.blake2b(repr(("g6_concat", key)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def create_program_descriptor(plan, tensors, output_tensor, key):
    assert output_tensor.buffer_num_pages() == plan.num_out_pages
    assert output_tensor.buffer_page_size() == plan.page_bytes

    cbs = [
        ttnn.CBDescriptor(
            total_size=plan.cb_output_pages_bytes,
            core_ranges=plan.all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUTPUT_PAGES, data_format=output_tensor.dtype, page_size=plan.page_bytes
                )
            ],
        )
    ]
    if plan.needs_staging:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=plan.cb_gather_staging_bytes,
                core_ranges=plan.all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_GATHER_STAGING, data_format=output_tensor.dtype, page_size=plan.page_bytes
                    )
                ],
            )
        )

    mask_lo = plan.input_l1_mask & 0xFFFFFFFF
    mask_hi = (plan.input_l1_mask >> 32) & 0xFFFFFFFF
    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    if plan.regime == "page_copy":
        reader_source = KERNEL_DIR / "g6_concat_page_reader.cpp"
        reader_ct = [
            CB_OUTPUT_PAGES,
            plan.page_bytes,
            plan.trim_rows_max,
            plan.face_row_bytes,
            plan.face_bytes,
            mask_lo,
            mask_hi,
        ]
        for core, start, count in plan.assignment:
            o, k, c_in = _page_copy_start_cursor(plan, start)
            reader_rt[core.x][core.y] = [start, count, o, k, c_in]
    else:
        reader_source = KERNEL_DIR / "g6_concat_gather_reader.cpp"
        reader_ct = [
            CB_OUTPUT_PAGES,
            CB_GATHER_STAGING,
            plan.page_bytes,
            plan.face_row_bytes,
            plan.face_bytes,
            int(plan.needs_staging),
            mask_lo,
            mask_hi,
        ]
        for core, start, count in plan.assignment:
            reader_rt[core.x][core.y] = [start, count]
    reader_ct.extend(_input_accessor_ct_args(tensors, plan))
    for core, start, count in plan.assignment:
        writer_rt[core.x][core.y] = [start, count]

    writer_ct = [CB_OUTPUT_PAGES, plan.page_bytes, plan.trim_rows_max, plan.face_row_bytes, plan.face_bytes]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    reader = ttnn.KernelDescriptor(
        kernel_source=str(reader_source),
        core_ranges=plan.all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        common_runtime_args=reader_common_args(plan, tensors),
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_concat_writer.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        common_runtime_args=writer_common_args(plan, output_tensor),
        config=ttnn.WriterConfigDescriptor(),
    )
    descriptor = ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=cbs)
    descriptor.custom_program_hash = program_hash(key)
    return descriptor, reader, writer
