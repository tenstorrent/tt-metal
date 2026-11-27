# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_slice: blocking plan and ProgramDescriptor.

One program: a reader (NCRISC, NoC0) fetches the input bytes of `block_pages` consecutive output pages
into cb_slice_pages, a writer (BRISC, NoC1) stores them to `out`. No compute kernel: the op is a bit
copy, every byte moves by NoC (or by RISC-V column copies for a last-dim begin inside a tile).

Work split: the output pages, in (lead0, lead1, tile row, tile column) order, are cut into contiguous
per-core ranges over the full grid (split_work_to_cores, row-wise). Shapes and regime flags are
compile-time; begins, buffer addresses and per-core ranges are runtime, so every begin of a shape
class shares one compiled program.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32

CB_SLICE_PAGES = 0
CB_RIGHT_COL_TILE = 1

# CBs are allocated at DRAM alignment; BB-05 charges each CB this much slack.
CB_ALLOC_ALIGN_BYTES = 64

# Four 16x16 faces of 16 rows each.
FACE_ROWS_PER_TILE = 64

# Dtypes whose tile page is plain face rows (no shared-exponent section), so a row band is a byte run.
FACE_ROW_DTYPES = (ttnn.bfloat16, ttnn.float32, ttnn.int32, ttnn.uint32)

REGIMES = ("page_copy", "row_runs", "col_realign")


def _div_up(a, b):
    return -(-a // b)


def select_regime(begins4, out_shape4):
    """col_realign for a last-dim begin inside a tile, page_copy when every output tile row is a whole
    input tile row, row_runs otherwise. Depends only on begins and output shape, never on the grid."""
    row_off = begins4[2] % TILE
    col_off = begins4[3] % TILE
    if col_off != 0:
        return "col_realign"
    if row_off == 0 and out_shape4[2] % TILE == 0:
        return "page_copy"
    return "row_runs"


@dataclass(frozen=True)
class SlicePlan:
    regime: str
    in_shape4: tuple
    begins4: tuple
    out_shape4: tuple
    in_ht: int
    in_wt: int
    out_ht: int
    out_wt: int
    rows_last_tile_row: int
    cols_last_tile_col: int
    total_out_pages: int
    page_bytes: int
    face_row_bytes: int
    num_cores: int
    all_cores: object  # ttnn.CoreRangeSet
    assignment: tuple  # ((core_x, core_y, start_page, pages), ...) in output page order
    core_pages_max: int
    block_pages: int
    cb_depth: int
    num_blocks_max: int
    cb_bytes: dict  # {cb name: bytes per core}
    l1_bytes_per_core: int

    def report(self):
        return {
            "regime": self.regime,
            "num_cores": self.num_cores,
            "block_pages": self.block_pages,
            "cb_depth": self.cb_depth,
            "num_blocks_max": self.num_blocks_max,
            "l1_bytes_per_core": self.l1_bytes_per_core,
            "core_roles": {(cx, cy): "reader+writer" for cx, cy, _, _ in self.assignment},
            "total_out_pages": self.total_out_pages,
            "core_pages_max": self.core_pages_max,
            "cb_bytes": dict(self.cb_bytes),
        }


def _out_l1_bytes_per_core(input_tensor):
    # `out` takes x's placement; only DRAM interleaved is supported, which holds nothing in L1.
    if input_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM:
        return 0
    raise NotImplementedError("g6_slice: L1 placement of out is not supported")


def _decompose(page, out_shape4, out_ht, out_wt):
    tw = page % out_wt
    rest = page // out_wt
    th = rest % out_ht
    rest //= out_ht
    return rest // out_shape4[1], rest % out_shape4[1], th, tw


def make_plan(input_tensor, begins4, out_shape4, *, block_pages_cap, cb_depth, max_cores, l1_budget_bytes):
    """Host blocking decision for one call signature (BB-08: reported by describe_plan)."""
    regime = select_regime(begins4, out_shape4)
    padded = list(input_tensor.padded_shape)
    in_shape4 = tuple([1] * (4 - len(padded)) + [int(d) for d in input_tensor.shape])
    in_ht = int(padded[-2]) // TILE
    in_wt = int(padded[-1]) // TILE
    out_ht = _div_up(out_shape4[2], TILE)
    out_wt = _div_up(out_shape4[3], TILE)
    total_out_pages = out_shape4[0] * out_shape4[1] * out_ht * out_wt

    page_bytes = input_tensor.buffer_page_size()
    face_row_bytes = page_bytes // FACE_ROWS_PER_TILE if input_tensor.dtype in FACE_ROW_DTYPES else 0

    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    if max_cores is None:
        work_grid = grid
    else:
        work_grid = ttnn.num_cores_to_corerangeset(min(max_cores, grid.x * grid.y), grid, True)
    num_cores, all_cores, group_1, group_2, pages_1, pages_2 = ttnn.split_work_to_cores(
        work_grid, total_out_pages, row_wise=True
    )

    num_cbs = 2 if regime == "col_realign" else 1
    slots_per_page = cb_depth + (1 if regime == "col_realign" else 0)
    l1_budget = min(l1_budget_bytes, ttnn.device.get_max_worker_l1_unreserved_size())
    fixed_bytes = _out_l1_bytes_per_core(input_tensor) + CB_ALLOC_ALIGN_BYTES * num_cbs
    l1_cap = (l1_budget - fixed_bytes) // (slots_per_page * page_bytes)
    cap = l1_cap if block_pages_cap is None else min(block_pages_cap, l1_cap)
    if cap < 1:
        raise RuntimeError(
            f"g6_slice: one {page_bytes}-B page per CB slot ({slots_per_page} slots) does not fit the "
            f"{l1_budget}-B L1 budget (BB-05)"
        )
    block_pages = min(pages_1, cap)
    num_blocks_max = _div_up(pages_1, block_pages)

    assignment = []
    start = 0
    for group, pages in ((group_1, pages_1), (group_2, pages_2)):
        if pages == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core.x, core.y, start, pages))
            start += pages
    assert start == total_out_pages, (start, total_out_pages)

    cb_bytes = {"cb_slice_pages": cb_depth * block_pages * page_bytes}
    if regime == "col_realign":
        cb_bytes["cb_right_col_tile"] = block_pages * page_bytes
    l1_bytes_per_core = _out_l1_bytes_per_core(input_tensor) + sum(cb_bytes.values())

    return SlicePlan(
        regime=regime,
        in_shape4=in_shape4,
        begins4=tuple(begins4),
        out_shape4=tuple(out_shape4),
        in_ht=in_ht,
        in_wt=in_wt,
        out_ht=out_ht,
        out_wt=out_wt,
        rows_last_tile_row=out_shape4[2] - TILE * (out_ht - 1),
        cols_last_tile_col=out_shape4[3] - TILE * (out_wt - 1),
        total_out_pages=total_out_pages,
        page_bytes=page_bytes,
        face_row_bytes=face_row_bytes,
        num_cores=num_cores,
        all_cores=all_cores,
        assignment=tuple(assignment),
        core_pages_max=pages_1,
        block_pages=block_pages,
        cb_depth=cb_depth,
        num_blocks_max=num_blocks_max,
        cb_bytes=cb_bytes,
        l1_bytes_per_core=l1_bytes_per_core,
    )


def _cb(index, total_size, plan, dtype):
    return ttnn.CBDescriptor(
        total_size=total_size,
        core_ranges=plan.all_cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=plan.page_bytes)],
    )


# Common-runtime-arg slot holding the buffer address, per kernel (reader: x, writer: out).
READER_KERNEL, WRITER_KERNEL = 0, 1
ADDRESS_SLOT = 0


def create_program_descriptor(input_tensor, output_tensor, plan: SlicePlan) -> ttnn.ProgramDescriptor:
    assert output_tensor.buffer_num_pages() == plan.total_out_pages, (
        output_tensor.buffer_num_pages(),
        plan.total_out_pages,
    )
    b4, M4, N4 = plan.begins4, plan.out_shape4, plan.in_shape4
    row_runs = int(plan.regime != "page_copy")
    w_shift = int(plan.regime == "col_realign")

    cbs = [_cb(CB_SLICE_PAGES, plan.cb_bytes["cb_slice_pages"], plan, input_tensor.dtype)]
    if w_shift:
        cbs.append(_cb(CB_RIGHT_COL_TILE, plan.cb_bytes["cb_right_col_tile"], plan, input_tensor.dtype))

    reader_ct = [
        CB_SLICE_PAGES,
        CB_RIGHT_COL_TILE,
        plan.page_bytes,
        plan.face_row_bytes,
        plan.block_pages,
        row_runs,
        w_shift,
        N4[1],
        M4[1],
        plan.in_ht,
        plan.in_wt,
        plan.out_ht,
        plan.out_wt,
        plan.rows_last_tile_row,
        plan.cols_last_tile_col,
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_common = [
        input_tensor.buffer_address(),
        b4[0],
        b4[1],
        b4[2] // TILE,
        b4[2] % TILE,
        b4[3] // TILE,
        b4[3] % TILE,
    ]

    writer_ct = [
        CB_SLICE_PAGES,
        plan.page_bytes,
        plan.face_row_bytes,
        plan.block_pages,
        row_runs,
        plan.out_ht,
        plan.out_wt,
        plan.rows_last_tile_row,
    ]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_common = [output_tensor.buffer_address(), b4[2] % TILE]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    for cx, cy, start, pages in plan.assignment:
        o0, o1, th, tw = _decompose(start, M4, plan.out_ht, plan.out_wt)
        reader_rt[cx][cy] = [pages, o0, o1, th, tw]
        writer_rt[cx][cy] = [start, pages, th, tw]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_slice_reader.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        common_runtime_args=reader_common,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_slice_writer.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        common_runtime_args=writer_common,
        config=ttnn.WriterConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=cbs)


def patch_buffer_addresses(descriptor: ttnn.ProgramDescriptor, input_tensor, output_tensor):
    """Cache hit: the only values that change between calls of one signature are the buffer addresses."""
    kernels = descriptor.kernels
    kernels[READER_KERNEL].common_runtime_args[ADDRESS_SLOT] = input_tensor.buffer_address()
    kernels[WRITER_KERNEL].common_runtime_args[ADDRESS_SLOT] = output_tensor.buffer_address()
