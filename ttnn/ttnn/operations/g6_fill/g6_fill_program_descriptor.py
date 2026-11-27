# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_fill program descriptor: the work plan, the single writer kernel and its one scratch CB.

The program is one writer kernel per active core. It reads no tensor; the fill
element and the output address arrive as common runtime args, so one cached
descriptor serves every fill value and every output buffer of the same shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

CB_FILL_ROW = 0

# --- Blocking knobs (single source; every dependent quantity derives from these) ---
# Cap on the active-core count; None means the whole compute grid.
MAX_ACTIVE_CORES = None
# Cap on the active rectangle's width (its aspect); None means the grid width, i.e. the fewest rows.
MAX_RECT_W = None
# Tiles per store_block; None means a core's whole assignment (one block per core).
BLOCK_W_TILES = None
# Pages of cb_fill_row. The page is written once and only read afterwards, so depth buys nothing.
FILL_ROW_DEPTH = 1

# Common runtime-arg slots of the writer kernel.
CRTA_OUT_ADDR = 0
CRTA_E_BITS = 1
CRTA_BLOCK_SEGMENTS = 2


@dataclass(frozen=True)
class Plan:
    regime: str
    segment_bytes: int
    segments_per_page: int
    segment_stride_bytes: int
    elem_bytes: int
    page_bytes: int
    tensor_w_tiles: int
    tensor_segments: int
    rect_w: int
    rect_h: int
    block_segments: int
    # {(x, y): (seg_start, seg_count)} over the active rectangle, row-major.
    per_core: dict

    @property
    def core_ranges(self):
        return ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.rect_w - 1, self.rect_h - 1))]
        )


def _ceil_div(a, b):
    return -(-a // b)


def make_plan(
    n, tile, elem_bytes, page_bytes, grid_x, grid_y, *, max_active_cores=None, max_rect_w=None, block_w_tiles=None
):
    """Spread the tile columns of the [1, n] tile plane over one rectangle anchored at core (0, 0).

    Only row 0 of each tile is logical: it is written as face-row segments, the
    longest contiguous logical runs in a tile.
    """
    tile_w = tile.tile_shape[1]
    face_h, face_w = tile.face_shape
    segments_per_page = tile_w // face_w
    segment_bytes = face_w * elem_bytes
    segment_stride_bytes = face_h * face_w * elem_bytes

    tensor_w_tiles = _ceil_div(n, tile_w)
    # A tail tile with at most face_w logical positions has only its face-0 segment.
    tensor_segments = _ceil_div(n, face_w)

    row_w = grid_x if max_rect_w is None else max(1, min(max_rect_w, grid_x))
    grid_cores = row_w * grid_y
    cap = grid_cores if max_active_cores is None else max(1, min(max_active_cores, grid_cores))
    target = min(tensor_w_tiles, cap)
    rect_h = _ceil_div(target, row_w)
    rect_w = target // rect_h
    num_cores = rect_w * rect_h

    q, r = divmod(tensor_w_tiles, num_cores)
    per_core = {}
    for y in range(rect_h):
        for x in range(rect_w):
            i = y * rect_w + x
            core_w_tiles = q + (1 if i < r else 0)
            start_tile = i * q + min(i, r)
            seg_start = start_tile * segments_per_page
            seg_end = min((start_tile + core_w_tiles) * segments_per_page, tensor_segments)
            per_core[(x, y)] = (seg_start, max(0, seg_end - seg_start))

    max_core_w_tiles = q + (1 if r else 0)
    block_tiles = max_core_w_tiles if block_w_tiles is None else max(1, min(block_w_tiles, max_core_w_tiles))

    return Plan(
        regime="segment_write",
        segment_bytes=segment_bytes,
        segments_per_page=segments_per_page,
        segment_stride_bytes=segment_stride_bytes,
        elem_bytes=elem_bytes,
        page_bytes=page_bytes,
        tensor_w_tiles=tensor_w_tiles,
        tensor_segments=tensor_segments,
        rect_w=rect_w,
        rect_h=rect_h,
        block_segments=block_tiles * segments_per_page,
        per_core=per_core,
    )


def plan_for(tensor):
    grid = tensor.device().compute_with_storage_grid_size()
    return make_plan(
        tensor.shape[-1],
        tensor.tile,
        tensor.element_size(),
        tensor.buffer_aligned_page_size(),
        grid.x,
        grid.y,
        max_active_cores=MAX_ACTIVE_CORES,
        max_rect_w=MAX_RECT_W,
        block_w_tiles=BLOCK_W_TILES,
    )


def common_runtime_args(plan, output_tensor, e_bits):
    args = [0] * 3
    args[CRTA_OUT_ADDR] = output_tensor.buffer_address()
    args[CRTA_E_BITS] = e_bits
    args[CRTA_BLOCK_SEGMENTS] = plan.block_segments
    return args


def create_program_descriptor(plan, output_tensor, e_bits):
    """Returns (ProgramDescriptor, writer KernelDescriptor); the caller re-applies common RT args per call."""
    core_ranges = plan.core_ranges

    cb_fill_row = ttnn.CBDescriptor(
        total_size=FILL_ROW_DEPTH * plan.segment_bytes,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_FILL_ROW,
                data_format=output_tensor.dtype,
                page_size=plan.segment_bytes,
            )
        ],
    )

    writer_ct_args = [
        CB_FILL_ROW,
        plan.segment_bytes,
        plan.segments_per_page,
        plan.segment_stride_bytes,
        plan.elem_bytes,
        plan.page_bytes,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    for (x, y), (seg_start, seg_count) in plan.per_core.items():
        writer_rt_args[x][y] = [seg_start, seg_count]

    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_fill_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        common_runtime_args=common_runtime_args(plan, output_tensor, e_bits),
        config=ttnn.WriterConfigDescriptor(),
    )

    descriptor = ttnn.ProgramDescriptor(kernels=[writer], semaphores=[], cbs=[cb_fill_row])
    return descriptor, writer
