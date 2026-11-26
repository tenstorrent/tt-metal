# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_concat — plan selection and ProgramDescriptor.

Blocking Model (op_design.md): one linear output-page axis, split output-stationary into contiguous
ranges over the full compute grid (split_work_to_cores, row-wise); each core's range is processed
in blocks of `block_pages` consecutive output pages, double-buffered through `cb_out_tiles`
(reader NCRISC/NoC0 -> writer BRISC/NoC1). Two regimes:

  page_relocation — every output page is one whole input page (outer / w / tile-aligned h).
  row_gather      — "h" concat with a misaligned input offset: each output tile is assembled from
                    row pieces (aligned DRAM windows -> cb_row_stage -> L1->L1 placement).

Knobs (each defined ONCE here; every CB size / loop bound derives from them):
  BLOCK_PAGES_MAX       -> block_pages = min(core_pages_max, BLOCK_PAGES_MAX)  (CT arg)
  OUT_BUFFER_DEPTH      -> cb_out_tiles capacity = OUT_BUFFER_DEPTH * block_pages
  STAGE_PAGES_PER_TILE  -> cb_row_stage page = STAGE_PAGES_PER_TILE * tile_bytes (mechanism cap)
  grid                  -> device.compute_with_storage_grid_size()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---- Knobs (single source of truth) ----
BLOCK_PAGES_MAX = 32
OUT_BUFFER_DEPTH = 2
STAGE_PAGES_PER_TILE = 2
K_MAX = 128

TILE_H = 32
TILE_W = 32
FACE_H = 16
FACE_W = 16

CB_OUT_TILES = 0
CB_ROW_STAGE = 1

REGIME_PAGE_RELOCATION = "page_relocation"
REGIME_ROW_GATHER = "row_gather"


@dataclass
class Plan:
    regime: str
    tensor_slices: int  # P
    per_input: list  # run_k pages (page_relocation) or n_k rows (row_gather)
    wt: int
    out_h_tiles: int  # row_gather: ceil(N/32) per slice
    total_rows: int  # row_gather: N
    tensor_out_pages: int
    block_pages: int
    num_cores: int
    core_ranges: object = field(repr=False)
    cores: list = field(repr=False)  # [(CoreCoord, start_page, core_pages)]

    def core_roles(self):
        """{(x, y): role} for every active core (BB-21): every core is a reader+writer peer."""
        return {(c.x, c.y): "reader+writer" for c, _, _ in self.cores}


def _tile_grid(shape):
    """Per-image tile grid: leading dims + [Ht, Wt] (alignment-aware ceil)."""
    return list(shape[:-2]) + [math.ceil(shape[-2] / TILE_H), math.ceil(shape[-1] / TILE_W)]


def select_plan(shapes, d, dtype, layout, grid):
    """Pure host function: regime, block size and core assignment for a call."""
    rank = len(shapes[0])
    K = len(shapes)
    wt = math.ceil(shapes[0][-1] / TILE_W)
    offsets = [sum(s[d] for s in shapes[:k]) for k in range(K)]
    total = sum(s[d] for s in shapes)

    is_h = d == rank - 2
    if is_h and any(o % TILE_H for o in offsets[1:]):
        regime = REGIME_ROW_GATHER
        P = math.prod(shapes[0][: rank - 2])
        per_input = [s[d] for s in shapes]  # rows
        out_h_tiles = math.ceil(total / TILE_H)
        tensor_out_pages = P * out_h_tiles * wt
    else:
        regime = REGIME_PAGE_RELOCATION
        grids = [_tile_grid(s) for s in shapes]
        P = math.prod(grids[0][:d])
        per_input = [math.prod(g[d:]) for g in grids]  # run_k
        out_h_tiles = 0
        tensor_out_pages = P * sum(per_input)

    num_cores, all_cores, g1, g2, pages_g1, pages_g2 = ttnn.split_work_to_cores(grid, tensor_out_pages, True)
    n_g1 = g1.num_cores()
    core_list = ttnn.grid_to_cores(num_cores, grid.x, grid.y, True)
    cores = []
    start = 0
    for i, c in enumerate(core_list):
        pages = pages_g1 if i < n_g1 else pages_g2
        cores.append((c, start, pages))
        start += pages
    assert start == tensor_out_pages, (start, tensor_out_pages)

    block_pages = max(1, min(max(pages_g1, pages_g2), BLOCK_PAGES_MAX))
    return Plan(
        regime=regime,
        tensor_slices=P,
        per_input=per_input,
        wt=wt,
        out_h_tiles=out_h_tiles,
        total_rows=total,
        tensor_out_pages=tensor_out_pages,
        block_pages=block_pages,
        num_cores=num_cores,
        core_ranges=all_cores,
        cores=cores,
    )


# Memoized programs (BB-07): key -> (reader KernelDescriptor, writer KernelDescriptor, cbs, plan)
_PROGRAM_CACHE = {}


def _cache_key(tensors, d, output, grid):
    return (
        tuple(tuple(t.shape) for t in tensors),
        d,
        tensors[0].dtype,
        tensors[0].layout,
        tuple(t.memory_config().buffer_type for t in tensors),
        output.memory_config().buffer_type,
        (grid.x, grid.y),
    )


def _build(tensors, d, output, grid):
    shapes = [list(t.shape) for t in tensors]
    dtype = tensors[0].dtype
    plan = select_plan(shapes, d, dtype, tensors[0].layout, grid)

    tile_bytes = output.buffer_page_size()
    assert all(t.buffer_page_size() == tile_bytes for t in tensors)
    assert output.buffer_num_pages() == plan.tensor_out_pages, (output.buffer_num_pages(), plan.tensor_out_pages)
    dram_align = ttnn.get_dram_alignment()
    elem_bytes = tensors[0].element_size()
    face_row_bytes = FACE_W * elem_bytes
    face_bytes = FACE_H * face_row_bytes

    all_cores = plan.core_ranges
    block_pages = plan.block_pages

    cbs = [
        ttnn.CBDescriptor(
            total_size=OUT_BUFFER_DEPTH * block_pages * tile_bytes,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_OUT_TILES, data_format=dtype, page_size=tile_bytes)
            ],
        )
    ]
    is_gather = plan.regime == REGIME_ROW_GATHER
    if is_gather:
        # Mechanism cap: a staged window is at most 2x its live bytes iff 2 * face_row_bytes >= A.
        assert STAGE_PAGES_PER_TILE * face_row_bytes >= dram_align, (face_row_bytes, dram_align)
        stage_page = STAGE_PAGES_PER_TILE * tile_bytes
        cbs.append(
            ttnn.CBDescriptor(
                total_size=block_pages * stage_page,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=CB_ROW_STAGE, data_format=dtype, page_size=stage_page)
                ],
            )
        )

    K = len(tensors)
    # ---- Reader ----
    # CT: [regime, num_inputs, P, Wt, out_h_tiles, total_rows, block_pages, tile_bytes, A,
    #      face_bytes, face_row_bytes, cb_out, cb_stage, per_input[K]..., TensorAccessorArgs(input)]
    reader_ct = [
        1 if is_gather else 0,
        K,
        plan.tensor_slices,
        plan.wt,
        plan.out_h_tiles,
        plan.total_rows,
        block_pages,
        tile_bytes,
        dram_align,
        face_bytes,
        face_row_bytes,
        CB_OUT_TILES,
        CB_ROW_STAGE,
    ]
    reader_ct.extend(plan.per_input)
    reader_ct.extend(ttnn.TensorAccessorArgs(tensors[0]).get_compile_time_args())

    writer_ct = [block_pages, tile_bytes, CB_OUT_TILES]
    writer_ct.extend(ttnn.TensorAccessorArgs(output).get_compile_time_args())

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    for c, start, pages in plan.cores:
        reader_rt[c.x][c.y] = [start, pages]
        writer_rt[c.x][c.y] = [start, pages]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_concat_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        common_runtime_args=[t.buffer_address() for t in tensors],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_concat_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        common_runtime_args=[output.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    return reader, writer, cbs, plan


def build_program(tensors, d, output):
    """Return (ProgramDescriptor, plan). A cache hit only patches the K + 1 address common RT args."""
    grid = tensors[0].device().compute_with_storage_grid_size()
    key = _cache_key(tensors, d, output, grid)
    entry = _PROGRAM_CACHE.get(key)
    if entry is None:
        entry = _build(tensors, d, output, grid)
        _PROGRAM_CACHE[key] = entry
    else:
        reader, writer, _, _ = entry
        reader.common_runtime_args = [t.buffer_address() for t in tensors]
        writer.common_runtime_args = [output.buffer_address()]
    reader, writer, cbs, plan = entry
    program = ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=cbs)
    return program, plan
