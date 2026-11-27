# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Plan and ProgramDescriptor of g6_concat (op_design.md: Blocking Model, Work Distribution).

The output is split by page: core i of the row-major core order owns a contiguous output page range
and walks it in blocks of block_pages pages. A reader fills cb_output_pages (page_copy: one input
page per output page; row_gather: row-range reads assembled in place) and the shared writer stores
the block. No compute kernel: values never pass through DEST.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
PAGE_READER_SOURCE = KERNEL_DIR / "g6_concat_page_reader.cpp"
GATHER_READER_SOURCE = KERNEL_DIR / "g6_concat_gather_reader.cpp"
WRITER_SOURCE = KERNEL_DIR / "g6_concat_writer.cpp"

# --- Blocking knobs (single source of truth; everything below derives from them) -----------------
# L1 cap of one block. block_pages = min(pages of the most-loaded core, BLOCK_BYTES_CAP // page_bytes).
BLOCK_BYTES_CAP = 128 * 1024
# Blocks held by cb_output_pages: the reader fills block i+1 while the writer stores block i.
OUTPUT_PAGES_DEPTH = 2
# A tile row with v valid rows, v <= TRIM_ROWS_MAX, moves only rows [0, v). bfloat8_b keeps its
# exponents in a page header, so its pages always move whole.
TRIM_ROWS_MAX = {ttnn.bfloat16: 16, ttnn.float32: 16, ttnn.bfloat8_b: 0}
# Cores used: None = the whole compute grid; an int caps the row-major core count.
MAX_CORES = None

# --- Mechanism caps --------------------------------------------------------------------------------
# Two common runtime words per input share the 341-word per-kernel runtime-arg space.
K_MAX = 64
# L1 the model leaves to the op on every core (BB-05).
L1_OP_BUDGET = 1347776

TILE_DIM = 32
FACE_DIM = 16

CB_OUTPUT_PAGES = 0
CB_GATHER_STAGING = 1

# Common runtime arg layouts; each mirrors the CRTA_* constants of its kernel.
PAGE_READER_COMMON_FIRST_INPUT = 5  # block_pages, out_wt, out_ht, last_row_valid_rows, K, then (base, C_k)*
GATHER_READER_COMMON_FIRST_INPUT = 5  # block_pages, out_wt, out_ht, out_rows, K, then (base, n_k)*
WRITER_COMMON_OUTPUT_BASE = 0  # output base, block_pages, out_wt, out_ht, last_row_valid_rows
COMMON_WORDS_PER_INPUT = 2


def ct(x):
    """Tile count of a dim (per image)."""
    return (x + TILE_DIM - 1) // TILE_DIM


def canonical_dim(dim, rank):
    return dim + rank if dim < 0 else dim


def select_regime(shapes, d, layout):
    r = len(shapes[0])
    if layout == ttnn.TILE_LAYOUT and d == r - 2:
        offset = 0
        for shape in shapes[:-1]:
            offset += shape[d]
            if offset % TILE_DIM != 0:
                return "row_gather"
    return "page_copy"


def page_copy_address_map(shapes, d):
    """(O, [C_k]): output page o*C + Coff_k + c' is input k's page o*C_k + c'."""
    r = len(shapes[0])
    s0 = shapes[0]
    if d <= r - 3:
        outer = math.prod(s0[:d])
        per_slice = [s[d] * math.prod(s0[d + 1 : r - 2]) * ct(s0[r - 2]) * ct(s0[r - 1]) for s in shapes]
    elif d == r - 2:
        outer = math.prod(s0[: r - 2])
        per_slice = [ct(s[d]) * ct(s0[r - 1]) for s in shapes]
    else:
        outer = math.prod(s0[: r - 2]) * ct(s0[r - 2])
        per_slice = [ct(s[d]) for s in shapes]
    return outer, per_slice


@dataclass
class CoreWork:
    core: ttnn.CoreCoord
    start_page: int
    num_pages: int


@dataclass
class G6ConcatPlan:
    regime: str
    shapes: list
    d: int
    out_shape: list
    dtype: object
    page_bytes: int
    face_row_bytes: int
    trim_rows_max: int
    out_wt: int
    out_ht: int
    out_rows: int
    last_row_valid_rows: int
    num_out_pages: int
    num_cores: int
    all_cores: object
    core_work: list
    block_pages: int
    depth: int
    has_staging: bool
    cb_output_bytes: int
    cb_staging_bytes: int
    l1_total: int
    pages_per_slice: list  # page_copy only: C_k
    num_slices: int  # page_copy only: O

    def report(self):
        role = f"{self.regime} reader+writer"
        return {
            "regime": self.regime,
            "num_cores": self.num_cores,
            "core_ranges": str(self.all_cores),
            "block_pages": self.block_pages,
            "depth": self.depth,
            "cb_output_pages_bytes": self.cb_output_bytes,
            "cb_gather_staging_bytes": self.cb_staging_bytes,
            "l1_total": self.l1_total,
            "cores": [
                {
                    "core": (w.core.x, w.core.y),
                    "role": role,
                    "pages": (w.start_page, w.start_page + w.num_pages),
                    "num_blocks": math.ceil(w.num_pages / self.block_pages),
                }
                for w in self.core_work
            ],
        }


def _core_assignment(device, num_out_pages):
    grid = device.compute_with_storage_grid_size()
    grid_cores = grid.x * grid.y
    if MAX_CORES is None or MAX_CORES >= grid_cores:
        split = ttnn.split_work_to_cores(grid, num_out_pages, row_wise=True)
    else:
        allowed = ttnn.num_cores_to_corerangeset(MAX_CORES, grid, row_wise=True)
        split = ttnn.split_work_to_cores(allowed, num_out_pages, row_wise=True)
    num_cores, all_cores, group_1, group_2, pages_g1, pages_g2 = split
    cores = [(c, pages_g1) for c in ttnn.corerange_to_cores(group_1, row_wise=True)]
    if pages_g2 > 0:
        cores += [(c, pages_g2) for c in ttnn.corerange_to_cores(group_2, row_wise=True)]
    assert len(cores) == num_cores
    assert [(c.y, c.x) for c, _ in cores] == sorted((c.y, c.x) for c, _ in cores), "cores must be row-major"
    core_work = []
    start = 0
    for core, pages in cores:
        core_work.append(CoreWork(core, start, pages))
        start += pages
    assert start == num_out_pages
    return num_cores, all_cores, core_work, pages_g1


def build_plan(tensors, d, output_memory_config, device):
    shapes = [list(t.shape) for t in tensors]
    r = len(shapes[0])
    dtype = tensors[0].dtype
    layout = tensors[0].layout
    regime = select_regime(shapes, d, layout)

    out_shape = list(shapes[0])
    out_shape[d] = sum(s[d] for s in shapes)
    out_rows = out_shape[r - 2]
    out_ht = ct(out_rows)
    out_wt = ct(out_shape[r - 1])
    num_out_pages = math.prod(out_shape[: r - 2]) * out_ht * out_wt
    last_row_valid_rows = out_rows - TILE_DIM * (out_ht - 1)

    page_bytes = tensors[0].buffer_page_size()
    face_row_bytes = FACE_DIM * (page_bytes // (TILE_DIM * TILE_DIM))
    trim_rows_max = TRIM_ROWS_MAX[dtype]

    num_slices, pages_per_slice = 0, []
    if regime == "page_copy":
        num_slices, pages_per_slice = page_copy_address_map(shapes, d)
        assert num_slices * sum(pages_per_slice) == num_out_pages
        if d == r - 2:
            # v of every input page equals v of its output page: only the last input may be partial.
            assert all(s[d] % TILE_DIM == 0 for s in shapes[:-1])
    else:
        assert dtype in (ttnn.bfloat16, ttnn.float32), "row_gather moves face rows; bfloat8_b needs row_gather_bfp8"

    num_cores, all_cores, core_work, pages_most_loaded_core = _core_assignment(device, num_out_pages)

    block_pages_cap = max(1, BLOCK_BYTES_CAP // page_bytes)
    block_pages = min(pages_most_loaded_core, block_pages_cap)

    any_dram_input = any(t.memory_config().buffer_type == ttnn.BufferType.DRAM for t in tensors)
    has_staging = regime == "row_gather" and any_dram_input and face_row_bytes % ttnn.get_dram_alignment() != 0

    cb_output_bytes = OUTPUT_PAGES_DEPTH * block_pages * page_bytes
    # One mirror slot per page of the block, plus one page of slack for the shift past the last slot.
    cb_staging_bytes = (block_pages + 1) * page_bytes if has_staging else 0
    # An L1 output adds its per-bank bytes to this total; validate() admits DRAM outputs only.
    assert output_memory_config.buffer_type == ttnn.BufferType.DRAM
    l1_total = cb_output_bytes + cb_staging_bytes
    if l1_total > L1_OP_BUDGET:
        raise RuntimeError(f"g6_concat: {l1_total} B of L1 per core exceeds the op budget of {L1_OP_BUDGET} B")

    return G6ConcatPlan(
        regime=regime,
        shapes=shapes,
        d=d,
        out_shape=out_shape,
        dtype=dtype,
        page_bytes=page_bytes,
        face_row_bytes=face_row_bytes,
        trim_rows_max=trim_rows_max,
        out_wt=out_wt,
        out_ht=out_ht,
        out_rows=out_rows,
        last_row_valid_rows=last_row_valid_rows,
        num_out_pages=num_out_pages,
        num_cores=num_cores,
        all_cores=all_cores,
        core_work=core_work,
        block_pages=block_pages,
        depth=OUTPUT_PAGES_DEPTH,
        has_staging=has_staging,
        cb_output_bytes=cb_output_bytes,
        cb_staging_bytes=cb_staging_bytes,
        l1_total=l1_total,
        pages_per_slice=pages_per_slice,
        num_slices=num_slices,
    )


def _page_copy_start_cursor(plan, start_page):
    total_per_slice = sum(plan.pages_per_slice)
    slice_idx, page_in_slice = divmod(start_page, total_per_slice)
    for input_idx, pages in enumerate(plan.pages_per_slice):
        if page_in_slice < pages:
            return slice_idx, input_idx, page_in_slice
        page_in_slice -= pages
    raise AssertionError("start page outside the output")


def _reader_common_args(plan, tensors):
    if plan.regime == "page_copy":
        args = [plan.block_pages, plan.out_wt, plan.out_ht, plan.last_row_valid_rows, len(tensors)]
        assert len(args) == PAGE_READER_COMMON_FIRST_INPUT
        for t, pages in zip(tensors, plan.pages_per_slice):
            args += [t.buffer_address(), pages]
    else:
        args = [plan.block_pages, plan.out_wt, plan.out_ht, plan.out_rows, len(tensors)]
        assert len(args) == GATHER_READER_COMMON_FIRST_INPUT
        for t, shape in zip(tensors, plan.shapes):
            args += [t.buffer_address(), shape[plan.d]]
    return args


def _writer_common_args(plan, output):
    args = [output.buffer_address(), plan.block_pages, plan.out_wt, plan.out_ht, plan.last_row_valid_rows]
    assert args[WRITER_COMMON_OUTPUT_BASE] == output.buffer_address()
    return args


def reader_address_slots(plan, num_inputs):
    first = PAGE_READER_COMMON_FIRST_INPUT if plan.regime == "page_copy" else GATHER_READER_COMMON_FIRST_INPUT
    return [first + k * COMMON_WORDS_PER_INPUT for k in range(num_inputs)]


@dataclass
class G6ConcatProgram:
    plan: G6ConcatPlan
    descriptor: ttnn.ProgramDescriptor
    reader: ttnn.KernelDescriptor
    writer: ttnn.KernelDescriptor
    reader_common_args: list
    writer_common_args: list

    def bind(self, tensors, output):
        """Rewrite the buffer addresses (the only call-dependent values) into the common runtime args."""
        for slot, t in zip(reader_address_slots(self.plan, len(tensors)), tensors):
            self.reader_common_args[slot] = t.buffer_address()
        self.writer_common_args[WRITER_COMMON_OUTPUT_BASE] = output.buffer_address()
        self.reader.common_runtime_args = self.reader_common_args
        self.writer.common_runtime_args = self.writer_common_args
        self.descriptor.kernels = [self.reader, self.writer]
        return self.descriptor


def build_program(plan, tensors, output, program_hash):
    assert output.buffer_num_pages() == plan.num_out_pages
    assert output.buffer_page_size() == plan.page_bytes
    input_accessor_ct = ttnn.TensorAccessorArgs(tensors[0]).get_compile_time_args()
    for t in tensors[1:]:
        assert (
            ttnn.TensorAccessorArgs(t).get_compile_time_args() == input_accessor_ct
        ), "inputs share one accessor; a per-input placement mask is needed for mixed buffers"

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    for work in plan.core_work:
        x, y = work.core.x, work.core.y
        if plan.regime == "page_copy":
            slice_idx, input_idx, page_in_slice = _page_copy_start_cursor(plan, work.start_page)
            reader_rt[x][y] = [work.start_page, work.num_pages, slice_idx, input_idx, page_in_slice]
        else:
            reader_rt[x][y] = [work.start_page, work.num_pages]
        writer_rt[x][y] = [work.start_page, work.num_pages]

    if plan.regime == "page_copy":
        reader_source = PAGE_READER_SOURCE
        reader_ct = [CB_OUTPUT_PAGES, plan.page_bytes, plan.face_row_bytes, plan.trim_rows_max]
    else:
        reader_source = GATHER_READER_SOURCE
        reader_ct = [CB_OUTPUT_PAGES, CB_GATHER_STAGING, int(plan.has_staging), plan.page_bytes, plan.face_row_bytes]
    reader_ct += input_accessor_ct

    writer_ct = [CB_OUTPUT_PAGES, plan.page_bytes, plan.face_row_bytes, plan.trim_rows_max]
    writer_ct += ttnn.TensorAccessorArgs(output).get_compile_time_args()

    reader_common_args = _reader_common_args(plan, tensors)
    writer_common_args = _writer_common_args(plan, output)

    reader = ttnn.KernelDescriptor(
        kernel_source=str(reader_source),
        core_ranges=plan.all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        common_runtime_args=reader_common_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(WRITER_SOURCE),
        core_ranges=plan.all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        common_runtime_args=writer_common_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    cbs = [
        ttnn.CBDescriptor(
            total_size=plan.cb_output_bytes,
            core_ranges=plan.all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_OUTPUT_PAGES, data_format=plan.dtype, page_size=plan.page_bytes)
            ],
        )
    ]
    if plan.has_staging:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=plan.cb_staging_bytes,
                core_ranges=plan.all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_GATHER_STAGING, data_format=plan.dtype, page_size=plan.page_bytes
                    )
                ],
            )
        )

    descriptor = ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=cbs)
    descriptor.custom_program_hash = program_hash
    return G6ConcatProgram(plan, descriptor, reader, writer, reader_common_args, writer_common_args)
