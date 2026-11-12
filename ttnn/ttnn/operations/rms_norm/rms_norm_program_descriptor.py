# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm ProgramDescriptor — regime R1 (row-resident, rows split across the full grid).

Every block knob of op_design.md's Blocking Model is computed exactly once here and passed down
as a compile-time / runtime arg; CB page counts and kernel trip counts derive from those knobs:

    block_rows        (r)  tile-rows per block                 -> CT (reader/compute/writer)
    w_chunk_tiles     (c)  tile-columns per normalize chunk     -> CT
    num_w_chunks      (K)  chunks per row; row_stride = K*c     -> CT
    core_w_tiles      (Wc) W slice per core (R1: whole row)     -> CT
    w_tile_start           first W tile of the core's slice     -> RT (R1: 0)
    stream_depth           depth of the streaming CB windows    -> host constant
    grid                   device.compute_with_storage_grid_size()
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32

# CB slots (semantic names; the index is just the buffer slot). Passed to every kernel as CT args
# so the kernels never restate them.
CB_X_RM = 0  # RM input sticks staged for tilize (RM only)
CB_X = 1  # resident x block (tiles)
CB_REDUCE_MASK = 2  # 0/1 mask for the partial last W tile (partial W only)
CB_SUMSQ = 3  # per-row sum of x^2 (L1-accumulated) [+ squared tail tile]
CB_RSTD = 4  # per-row 1/sqrt(mean + eps), col-0 valid
CB_GAMMA = 5  # gamma chunk, row 0 of every tile valid (gamma only)
CB_NORMED = 6  # x * rstd chunk (gamma or RM)
CB_OUT = 7  # output tiles (TILE only)
CB_OUT_RM = 8  # untilized output (RM only)
CB_IDS = [CB_X_RM, CB_X, CB_REDUCE_MASK, CB_SUMSQ, CB_RSTD, CB_GAMMA, CB_NORMED, CB_OUT, CB_OUT_RM]

# Depth (in chunks) of every streaming window: cb_x_rm, cb_gamma, cb_out, cb_out_rm.
STREAM_DEPTH = 2
# Granularity floor of a W chunk (whole tiles, >= 8 per NoC batch; catalog `double_buffer`).
W_CHUNK_FLOOR_TILES = 8
# Slack kept free of CBs in the per-core L1 budget.
L1_SAFETY_MARGIN = 32 * 1024


def _f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _prod(xs):
    p = 1
    for x in xs:
        p *= int(x)
    return p


class _Geometry:
    """Host-side constants derived once from the tensors (single source each)."""

    def __init__(self, x, gamma, out):
        shape = [int(d) for d in x.shape]
        self.is_rm = x.layout == ttnn.ROW_MAJOR_LAYOUT
        self.has_gamma = gamma is not None
        self.W = shape[-1]
        self.tensor_w_tiles = math.ceil(self.W / TILE)
        self.w_partial_elems = self.W % TILE
        if self.is_rm:
            self.num_valid_sticks = _prod(shape[:-1])
            self.tensor_rows_tiles = math.ceil(self.num_valid_sticks / TILE)
        else:
            self.num_valid_sticks = 0
            self.tensor_rows_tiles = _prod(shape[:-2]) * math.ceil(shape[-2] / TILE)
        self.s_x = ttnn.tile_size(x.dtype)
        self.s_o = ttnn.tile_size(out.dtype)
        self.s_g = ttnn.tile_size(gamma.dtype) if self.has_gamma else 0
        self.s_f = ttnn.tile_size(ttnn.float32)
        self.x_elem = x.element_size()
        self.out_elem = out.element_size()
        self.g_elem = gamma.element_size() if self.has_gamma else 0


def _n_sq(g: _Geometry, core_w_tiles: int) -> int:
    """Sumsq tiles per row: one L1-accumulated tile over the whole tiles + the squared partial tail.

    Derived from the `core_w_tiles` knob (not the tensor width) — the compute kernel derives the same
    quantity from the same CT args, so the CB size and the kernel's push count cannot drift apart.
    """
    p = 1 if g.w_partial_elems else 0
    return (1 if core_w_tiles - p > 0 else 0) + p


def _footprint(g: _Geometry, core_w_tiles: int, r: int, K: int) -> int:
    """F(r, K) of l1_ledger.md (sum over the CBs this layout/gamma variant allocates)."""
    c = math.ceil(core_w_tiles / K)
    row_stride = K * c
    p = 1 if g.w_partial_elems else 0
    f = r * row_stride * g.s_x  # cb_x
    if g.is_rm:
        f += STREAM_DEPTH * c * g.s_x  # cb_x_rm
    f += p * g.s_f  # cb_reduce_mask (fp32, matches cb_sumsq's srcB format)
    f += r * g.s_f  # cb_rstd
    f += r * _n_sq(g, core_w_tiles) * g.s_f  # cb_sumsq
    if g.has_gamma or g.is_rm:
        f += r * c * g.s_f  # cb_normed
    if g.has_gamma:
        f += STREAM_DEPTH * c * g.s_g  # cb_gamma
    f += STREAM_DEPTH * c * g.s_o  # cb_out | cb_out_rm
    return f


def _select_blocks(g: _Geometry, core_w_tiles: int, rows_cap: int, budget: int):
    """Block-selection function of op_design.md: maximize block_rows, then minimize num_w_chunks."""
    c_floor = min(core_w_tiles, W_CHUNK_FLOOR_TILES)
    k_floor = math.ceil(core_w_tiles / c_floor)
    if _footprint(g, core_w_tiles, 1, k_floor) > budget:
        raise NotImplementedError("rms_norm: row too wide for single-core residency; needs W-split regime R3")
    r = 1
    for cand in range(rows_cap, 0, -1):
        if _footprint(g, core_w_tiles, cand, k_floor) <= budget:
            r = cand
            break
    K = k_floor
    for cand in range(1, k_floor + 1):
        if _footprint(g, core_w_tiles, r, cand) <= budget:
            K = cand
            break
    c = math.ceil(core_w_tiles / K)
    K = math.ceil(core_w_tiles / c)  # drop chunks that would be all padding
    return r, c, K


def _cb(index, num_pages, page_size, data_format, core_ranges):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def create_program_descriptor(x, gamma, out, epsilon, compute_config) -> ttnn.ProgramDescriptor:
    g = _Geometry(x, gamma, out)
    device = x.device()

    # ---------------- work distribution (rows over the full grid) ----------------
    grid = device.compute_with_storage_grid_size()
    (num_cores, all_cores, core_group_1, core_group_2, rows_per_core_1, rows_per_core_2) = ttnn.split_work_to_cores(
        grid, g.tensor_rows_tiles, row_wise=True
    )
    rows_cap = max(rows_per_core_1, rows_per_core_2)

    # ---------------- block selection ----------------
    core_w_tiles = g.tensor_w_tiles  # R1: each core owns the whole row
    w_tile_start = 0
    # Per-core L1 left for CBs: the allocator's largest free contiguous L1 region (already net of
    # every L1-resident tensor, including the freshly allocated output), minus a safety margin.
    l1_view = ttnn.device.get_memory_view(device, ttnn.BufferType.L1)
    budget = l1_view.largest_contiguous_bytes_free_per_bank - L1_SAFETY_MARGIN
    block_rows, w_chunk_tiles, num_w_chunks = _select_blocks(g, core_w_tiles, rows_cap, budget)
    row_stride_tiles = num_w_chunks * w_chunk_tiles

    # ---------------- circular buffers ----------------
    fp32 = ttnn.float32
    cbs = [
        _cb(CB_X, block_rows * row_stride_tiles, g.s_x, x.dtype, all_cores),
        _cb(CB_SUMSQ, block_rows * _n_sq(g, core_w_tiles), g.s_f, fp32, all_cores),
        _cb(CB_RSTD, block_rows, g.s_f, fp32, all_cores),
    ]
    if g.is_rm:
        cbs.append(_cb(CB_X_RM, STREAM_DEPTH * w_chunk_tiles, g.s_x, x.dtype, all_cores))
        cbs.append(_cb(CB_OUT_RM, STREAM_DEPTH * w_chunk_tiles, g.s_o, out.dtype, all_cores))
    else:
        cbs.append(_cb(CB_OUT, STREAM_DEPTH * w_chunk_tiles, g.s_o, out.dtype, all_cores))
    if g.w_partial_elems:
        cbs.append(_cb(CB_REDUCE_MASK, 1, g.s_f, fp32, all_cores))
    if g.has_gamma:
        cbs.append(_cb(CB_GAMMA, STREAM_DEPTH * w_chunk_tiles, g.s_g, gamma.dtype, all_cores))
    if g.has_gamma or g.is_rm:
        cbs.append(_cb(CB_NORMED, block_rows * w_chunk_tiles, g.s_f, fp32, all_cores))

    # ---------------- per-core runtime args ----------------
    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    x_addr = x.buffer_address()
    gamma_addr = gamma.buffer_address() if g.has_gamma else 0
    out_addr = out.buffer_address()
    start = 0
    for group, per_core in ((core_group_1, rows_per_core_1), (core_group_2, rows_per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            reader_rt[core.x][core.y] = [x_addr, gamma_addr, start, per_core, w_tile_start]
            writer_rt[core.x][core.y] = [out_addr, start, per_core, w_tile_start]
            compute_rt[core.x][core.y] = [per_core]
            start += per_core
    assert start == g.tensor_rows_tiles

    # ---------------- kernels ----------------
    blocking_ct = [block_rows, w_chunk_tiles, num_w_chunks, core_w_tiles]

    reader_ct = (
        CB_IDS
        + blocking_ct
        + [
            g.tensor_w_tiles,
            g.w_partial_elems,
            int(g.has_gamma),
            int(g.is_rm),
            g.W,
            g.num_valid_sticks,
            g.x_elem,
            g.g_elem,
        ]
    )
    reader_ct.extend(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    reader_ct.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if g.has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    writer_ct = CB_IDS + blocking_ct + [g.tensor_w_tiles, int(g.is_rm), g.W, g.num_valid_sticks, g.out_elem]
    writer_ct.extend(ttnn.TensorAccessorArgs(out).get_compile_time_args())

    compute_ct = (
        CB_IDS
        + blocking_ct
        + [
            g.w_partial_elems,
            int(g.has_gamma),
            int(g.is_rm),
            _f32_bits(1.0 / g.W),
            _f32_bits(epsilon),
        ]
    )

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=compute_config,
    )

    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
