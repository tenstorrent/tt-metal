# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_concat program: host plan, program cache and ProgramDescriptor.

One program, two data-movement kernels per core (reader on NCRISC / NoC0, writer on
BRISC / NoC1) joined by `cb_moved_pages`. No compute kernel: nothing is computed.

Work unit: one output page. The P output pages are split into contiguous ranges over
the grid; a block is `block_pages` consecutive pages of a core's range (row_gather:
also cut at output tile-row ends). Shape quantities and addresses travel as common
runtime args, so one binary serves a shape key; per-core args are the page range only.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# --- Blocking knobs (op_design.md, Blocking Model). Each is defined once here. ---
BUFFER_DEPTH = 2
MIN_PAGES_PER_CORE = 1
# Per-core L1 the model leaves to the op (BB-05).
L1_OP_BUDGET = 1347776

TILE_DIM = 32
FACE_DIM = 16

CB_MOVED_PAGES = 0
CB_PARITY_STAGE = 1

# Common runtime-arg table: fixed header, then (address, count) per input.
READER_HEADER_WORDS = 5
WORDS_PER_INPUT = 2
PER_CORE_RT_WORDS = 2
# Portable floor for unique + common runtime args per kernel (kernel_types.hpp max_runtime_args).
MAX_RUNTIME_ARGS = 341
K_MAX = (MAX_RUNTIME_ARGS - READER_HEADER_WORDS - PER_CORE_RT_WORDS) // WORDS_PER_INPUT

REGIME_PAGE_COPY = "page_copy"
REGIME_ROW_GATHER = "row_gather"


def _prod(values):
    out = 1
    for v in values:
        out *= int(v)
    return out


def _buffer_name(buffer_type):
    return "l1" if buffer_type == ttnn.BufferType.L1 else "dram"


def output_shape(shapes, d):
    out = list(shapes[0])
    out[d] = sum(int(s[d]) for s in shapes)
    return out


def select_regime(shapes, d, layout):
    """row_gather iff some input offset o_k (k >= 1) along h is not tile-aligned."""
    rank = len(shapes[0])
    if layout == ttnn.TILE_LAYOUT and d == rank - 2:
        if any(int(s[d]) % TILE_DIM != 0 for s in shapes[:-1]):
            return REGIME_ROW_GATHER
    return REGIME_PAGE_COPY


def _geometry(shapes, d, regime):
    """Output-page geometry and the per-input count of the common RT table."""
    rank = len(shapes[0])
    out = output_shape(shapes, d)
    h_out, w_out = out[-2], out[-1]
    out_tile_cols = math.ceil(w_out / TILE_DIM)
    out_tile_rows_per_image = math.ceil(h_out / TILE_DIM)
    out_rows_tail = h_out % TILE_DIM
    out_images = _prod(out[: rank - 2])
    num_pages = out_images * out_tile_rows_per_image * out_tile_cols

    if regime == REGIME_ROW_GATHER:
        # Kernel derives row offsets and source tile rows from the per-input row count.
        per_input_count = [int(s[d]) for s in shapes]
        pm_cols = out_tile_cols
    else:
        ht = math.ceil(int(shapes[0][-2]) / TILE_DIM)
        wt = math.ceil(int(shapes[0][-1]) / TILE_DIM)
        if d == rank - 1:
            pm_rows = out_images * ht
            per_input_count = [math.ceil(int(s[-1]) / TILE_DIM) for s in shapes]
        elif d == rank - 2:
            pm_rows = out_images
            per_input_count = [math.ceil(int(s[-2]) / TILE_DIM) * wt for s in shapes]
        else:
            pm_rows = _prod(shapes[0][:d])
            inner = _prod(shapes[0][d + 1 : rank - 2])
            per_input_count = [int(s[d]) * inner * ht * wt for s in shapes]
        pm_cols = sum(per_input_count)
        assert pm_rows * pm_cols == num_pages, (pm_rows, pm_cols, num_pages)

    return dict(
        out_shape=out,
        num_pages=num_pages,
        pm_cols=pm_cols,
        out_tile_cols=out_tile_cols,
        out_tile_rows_per_image=out_tile_rows_per_image,
        out_rows_tail=out_rows_tail,
        per_input_count=per_input_count,
    )


def _block_pages(regime, core_out_pages_max, page_bytes, has_stage, dram_read_align, out_tile_cols):
    """Coarsest block that keeps `BUFFER_DEPTH` meaningful and fits L1_OP_BUDGET (closed form)."""
    by_depth = math.ceil(core_out_pages_max / BUFFER_DEPTH)
    stage = 1 if has_stage else 0
    cap_l1 = (L1_OP_BUDGET - stage * dram_read_align) // ((BUFFER_DEPTH + stage) * page_bytes)
    block = min(by_depth, cap_l1)
    if regime == REGIME_ROW_GATHER:
        block = min(block, out_tile_cols)
    return max(1, block)


def _work_split(grid, num_pages):
    """Contiguous out_page ranges over min(grid, ceil(P / MIN_PAGES_PER_CORE)) cores, row-wise."""
    grid_cores = grid.x * grid.y
    target_cores = min(grid_cores, math.ceil(num_pages / MIN_PAGES_PER_CORE))
    candidate = ttnn.num_cores_to_corerangeset(target_cores, grid, True)
    num_cores, all_cores, group_1, group_2, pages_1, pages_2 = ttnn.split_work_to_cores(candidate, num_pages, True)
    ranges = []
    start = 0
    for group, pages in ((group_1, pages_1), (group_2, pages_2)):
        if pages == 0 or group.num_cores() == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            ranges.append((core, start, pages))
            start += pages
    assert start == num_pages and len(ranges) == num_cores, (start, num_pages, len(ranges), num_cores)
    return num_cores, all_cores, ranges, pages_1


def _key(tensors, d, out_buffer_type, grid):
    first = tensors[0]
    return (
        (grid.x, grid.y),
        tuple(tuple(int(v) for v in t.shape) for t in tensors),
        d,
        first.dtype,
        first.layout,
        tuple(_buffer_name(t.memory_config().buffer_type) for t in tensors),
        _buffer_name(out_buffer_type),
        tuple(ttnn.TensorAccessorArgs(first).get_compile_time_args()),
    )


def _build(tensors, d, grid, out_buffer_type):
    first = tensors[0]
    shapes = [list(t.shape) for t in tensors]
    regime = select_regime(shapes, d, first.layout)
    geo = _geometry(shapes, d, regime)

    page_bytes = first.buffer_page_size()
    face_row_bytes = FACE_DIM * first.element_size()
    dram_read_align = ttnn.get_dram_alignment()
    # A face row narrower than the DRAM-read congruence puts odd row shifts in another class.
    has_stage = regime == REGIME_ROW_GATHER and face_row_bytes % dram_read_align != 0

    num_cores, all_cores, ranges, core_out_pages_max = _work_split(grid, geo["num_pages"])
    block_pages = _block_pages(regime, core_out_pages_max, page_bytes, has_stage, dram_read_align, geo["out_tile_cols"])

    per_core_rt = ttnn.RuntimeArgs()
    for core, start, pages in ranges:
        per_core_rt[core.x][core.y] = [start, pages]

    cbs = [
        ttnn.CBDescriptor(
            total_size=BUFFER_DEPTH * block_pages * page_bytes,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_MOVED_PAGES, data_format=first.dtype, page_size=page_bytes)
            ],
        )
    ]
    if has_stage:
        # One destination-image tile per block column, plus slack to shift the base into the odd class.
        stage_bytes = block_pages * page_bytes + dram_read_align
        cbs.append(
            ttnn.CBDescriptor(
                total_size=stage_bytes,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_PARITY_STAGE, data_format=first.dtype, page_size=dram_read_align
                    )
                ],
            )
        )

    is_row_gather = 1 if regime == REGIME_ROW_GATHER else 0
    reader_ct = [
        CB_MOVED_PAGES,
        CB_PARITY_STAGE,
        block_pages,
        page_bytes,
        face_row_bytes,
        dram_read_align,
        is_row_gather,
        1 if has_stage else 0,
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(first).get_compile_time_args())

    # Header of the reader's common table; addresses are filled per call.
    reader_common_header = [
        len(tensors),
        geo["pm_cols"],
        geo["out_tile_cols"],
        geo["out_tile_rows_per_image"],
        geo["out_rows_tail"],
    ]
    assert len(reader_common_header) == READER_HEADER_WORDS
    writer_common_shape = [geo["out_tile_cols"], geo["out_tile_rows_per_image"], geo["out_rows_tail"]]

    plan = {
        "regime": regime,
        "num_cores": num_cores,
        "block_pages": block_pages,
        "buffer_depth": BUFFER_DEPTH,
        "output_pages": geo["num_pages"],
        "core_out_pages_max": core_out_pages_max,
        "parity_stage": has_stage,
        "l1_bytes_per_core": sum(cb.total_size for cb in cbs),
        "roles": {(core.x, core.y): "mover" for core, _, _ in ranges},
    }
    key_digest = hashlib.blake2b(repr(("g6_concat", _key(tensors, d, out_buffer_type, grid))).encode(), digest_size=8)
    return dict(
        plan=plan,
        out_shape=geo["out_shape"],
        all_cores=all_cores,
        per_core_rt=per_core_rt,
        cbs=cbs,
        reader_ct=reader_ct,
        reader_common_header=reader_common_header,
        per_input_count=geo["per_input_count"],
        writer_ct_head=[CB_MOVED_PAGES, block_pages, page_bytes, face_row_bytes, is_row_gather],
        writer_common_shape=writer_common_shape,
        program_hash=int.from_bytes(key_digest.digest(), "little"),
    )


_PROGRAMS: dict = {}


def built_program(tensors, d, out_memory_config):
    """The cached plan and descriptor parts of a shape key (BB-07); built on the first call."""
    grid = tensors[0].device().compute_with_storage_grid_size()
    key = _key(tensors, d, out_memory_config.buffer_type, grid)
    built = _PROGRAMS.get(key)
    if built is None:
        built = _build(tensors, d, grid, out_memory_config.buffer_type)
        _PROGRAMS[key] = built
    return built


def create_program_descriptor(built, tensors, output_tensor) -> ttnn.ProgramDescriptor:
    """Thin per-call wrapper: cached CBs / CT args / per-core args, fresh common args (addresses)."""
    reader_common = list(built["reader_common_header"])
    for t, count in zip(tensors, built["per_input_count"]):
        reader_common.extend([t.buffer_address(), count])
    writer_common = [output_tensor.buffer_address(), *built["writer_common_shape"]]
    writer_ct = list(built["writer_ct_head"]) + list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_concat_reader.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=built["all_cores"],
        compile_time_args=built["reader_ct"],
        runtime_args=built["per_core_rt"],
        common_runtime_args=reader_common,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_concat_writer.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=built["all_cores"],
        compile_time_args=writer_ct,
        runtime_args=built["per_core_rt"],
        common_runtime_args=writer_common,
        config=ttnn.WriterConfigDescriptor(),
    )
    program = ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=built["cbs"])
    program.custom_program_hash = built["program_hash"]
    return program
