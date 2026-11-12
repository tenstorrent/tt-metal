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

# x / out placement (dataflow CT arg):
PLACEMENT_INTERLEAVED = 0  # TensorAccessor reads/writes (DRAM or L1 interleaved)
PLACEMENT_RESIDENT = 1  # HEIGHT_SHARDED, CBs backed by the shards (zero-copy)
PLACEMENT_LOCAL_STICKS = 2  # HEIGHT_SHARDED RM whose shard is not tilize-consumable in place

# Depth (in chunks) of every streaming window: cb_x_rm, cb_gamma, cb_out, cb_out_rm.
STREAM_DEPTH = 2
# Depth (in blocks) of cb_x when a core owns > 1 tile-row (op_design "x_block_depth" perf lamp): 2 lets
# the reader prefetch block b+1 while block b is normalized + written. Live knob, parked at 1 (the
# byte-identical pre-R3 selection): measured on the interleaved prefill shapes (2-3 tile-rows/core),
# depth 2 with block_rows = 1 is 10-12% SLOWER (8192x1024: 89.3 -> 98.2 us) — the serial read-all /
# write-all phases use DRAM better than mixed read+write traffic at this geometry.
X_BLOCK_DEPTH = 1
# Target blocks per core when pipelining (caps block_rows at ceil(rows_cap / this)); more blocks = more
# read/write overlap, fewer = fewer per-block compute passes. 1 = whole per-core assignment per block.
BLOCKS_PER_CORE = 1
# Granularity floor of a W chunk (whole tiles, >= 8 per NoC batch; catalog `double_buffer`).
W_CHUNK_FLOOR_TILES = 8
# Minimum gamma row-0 read (bytes): the DRAM read alignment (Blackhole 64 B; WH 32 B).
GAMMA_ROW_READ_ALIGN = 64
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

    def __init__(self, x, gamma, out, fp32_dest_acc_en=True):
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
        # Compute-intermediate page format follows the DEST width (numeric-formats §4): fp32 DEST ->
        # Float32 pages; 16-bit DEST -> bf16 pages (DEST already rounded to 16 bits, a wider page stores
        # nothing extra). Single source for cb_sumsq / cb_rstd / cb_normed / cb_reduce_mask.
        self.interm_fmt = ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16
        self.s_f = ttnn.tile_size(self.interm_fmt)
        # cb_sumsq is the exception: it is a PACKER L1 accumulator — the Σx² adds happen in L1 at the page
        # format, not in DEST — so Float32 pages carry real extra bits even with 16-bit DEST (bf16 pages
        # bias rstd by ~2% at W=7168). Only r*n_sq (<= 2 per row) tiles; cb_reduce_mask must match it
        # (the masked fold unpacks the mask as srcB under cb_sumsq's format).
        self.sumsq_fmt = ttnn.float32
        self.s_sq = ttnn.tile_size(self.sumsq_fmt)
        self.gamma_tiled = self.has_gamma and gamma.layout == ttnn.TILE_LAYOUT
        # TILE gamma: only row 0 (the broadcast row) is consumed — the reader fetches the head of face 0
        # and face 1 (16 elements each, rounded up to the DRAM read alignment) instead of the whole page.
        # Block-float gamma (shared exponents ahead of the data) keeps whole-page reads (0).
        self.g_row0_bytes, self.g_face1_offset = 0, 0
        if self.gamma_tiled and gamma.dtype in (ttnn.bfloat16, ttnn.float32):
            g_el = 2 if gamma.dtype == ttnn.bfloat16 else 4
            self.g_row0_bytes = max(16 * g_el, GAMMA_ROW_READ_ALIGN)
            self.g_face1_offset = 16 * 16 * g_el
        # Element sizes only address ROW_MAJOR sticks; block-float (bfloat8_b, TILE-only) has none.
        self.x_elem = x.element_size() if self.is_rm else 0
        self.out_elem = out.element_size() if self.is_rm else 0
        self.g_elem = gamma.element_size() if self.has_gamma and not self.gamma_tiled else 0

        # ---- placement (regime R1 interleaved | R6 HEIGHT_SHARDED in place) ----
        self.is_sharded = x.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        self.stick_page_bytes = x.buffer_aligned_page_size() if self.is_rm else 0
        if self.is_sharded:
            spec = x.memory_config().shard_spec
            self.shard_h = int(spec.shape[0])  # rows (TILE: multiple of 32; RM: sticks)
            shard_w = int(spec.shape[1])
            # TILE: the shard spans the padded row; RM: the (page-granule padded) row.
            if (shard_w != self.tensor_w_tiles * TILE) if not self.is_rm else (shard_w < self.W):
                raise NotImplementedError(f"rms_norm: HEIGHT_SHARDED shard width {shard_w} must span the whole row")
            self.shard_grid = spec.grid
            self.shard_row_wise = spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
            # RM shard is tilize-consumable in place only when its stick stride is exactly Wc*32 elements
            # and every tile-row of the shard allocation is whole (32 sticks); otherwise the reader
            # restrides the resident sticks into cb_x_rm (local L1 copy, no DRAM).
            self.rm_in_place = self.is_rm and self.w_partial_elems == 0 and self.shard_h % TILE == 0
        else:
            self.shard_h = 0
            self.rm_in_place = False
        # cb_x (TILE) / cb_x_rm + cb_out_rm (RM in place) / cb_out (TILE) are the resident shards.
        self.x_resident = self.is_sharded and (not self.is_rm or self.rm_in_place)


def _n_sq(g: _Geometry, core_w_tiles: int) -> int:
    """Sumsq tiles per row: one L1-accumulated tile over the whole tiles + the squared partial tail.

    Derived from the `core_w_tiles` knob (not the tensor width) — the compute kernel derives the same
    quantity from the same CT args, so the CB size and the kernel's push count cannot drift apart.
    """
    p = 1 if g.w_partial_elems else 0
    return (1 if core_w_tiles - p > 0 else 0) + p


def _footprint(g: _Geometry, core_w_tiles: int, r: int, K: int, x_depth: int = 1) -> int:
    """F(r, K) of l1_ledger.md (sum over the CBs this layout/gamma variant allocates)."""
    c = math.ceil(core_w_tiles / K)
    row_stride = K * c
    p = 1 if g.w_partial_elems else 0
    f = 0
    if g.is_rm or not g.x_resident:
        f += x_depth * r * row_stride * g.s_x  # cb_x (TILE sharded: the resident input shard, not allocated)
    if g.is_rm and not g.x_resident:
        f += STREAM_DEPTH * c * g.s_x  # cb_x_rm (RM in place: the resident input shard)
    f += p * g.s_sq  # cb_reduce_mask (matches cb_sumsq's srcB format)
    f += r * g.s_f  # cb_rstd
    f += r * _n_sq(g, core_w_tiles) * g.s_sq  # cb_sumsq
    if g.has_gamma or g.is_rm:
        f += r * c * g.s_f  # cb_normed
    if g.has_gamma:
        f += STREAM_DEPTH * c * g.s_g  # cb_gamma
    if not g.x_resident:
        f += STREAM_DEPTH * c * g.s_o  # cb_out | cb_out_rm (resident output shard when x_resident)
    return f


def _select_blocks(g: _Geometry, core_w_tiles: int, rows_cap: int, budget: int):
    """Block-selection function of op_design.md -> (block_rows, w_chunk_tiles, num_w_chunks, x_block_depth).

    Streaming x (interleaved / LOCAL_STICKS): pipeline first. With > 1 tile-row per core, cb_x holds
    X_BLOCK_DEPTH blocks and block_rows is capped at ceil(rows_cap / BLOCKS_PER_CORE) so the reader
    fetches block b+1 while block b is normalized and written (read/write overlap on the DRAM NoCs).
    Then maximize block_rows within that cap, then minimize num_w_chunks. Falls back to depth 1 / the
    full rows_cap (the pre-pipelining selection) when the deeper cb_x does not fit.
    """
    if g.x_resident:
        r, c, K = _select_resident_blocks(g, core_w_tiles, rows_cap, budget)
        return r, c, K, 1
    c_floor = min(core_w_tiles, W_CHUNK_FLOOR_TILES)
    k_floor = math.ceil(core_w_tiles / c_floor)
    if _footprint(g, core_w_tiles, 1, k_floor) > budget:
        raise NotImplementedError("rms_norm: row too wide for single-core residency; needs W-split regime R3")
    depth, r_cap = 1, rows_cap
    if rows_cap > 1 and X_BLOCK_DEPTH > 1 and _footprint(g, core_w_tiles, 1, k_floor, X_BLOCK_DEPTH) <= budget:
        depth, r_cap = X_BLOCK_DEPTH, max(1, math.ceil(rows_cap / BLOCKS_PER_CORE))
    r = 1
    for cand in range(r_cap, 0, -1):
        if _footprint(g, core_w_tiles, cand, k_floor, depth) <= budget:
            r = cand
            break
    K = k_floor
    for cand in range(1, k_floor + 1):
        if _footprint(g, core_w_tiles, r, cand, depth) <= budget:
            K = cand
            break
    c = math.ceil(core_w_tiles / K)
    K = math.ceil(core_w_tiles / c)  # drop chunks that would be all padding
    return r, c, K, depth


def _select_resident_blocks(g: _Geometry, core_w_tiles: int, rows_cap: int, budget: int):
    """R6: x / out are the resident shards (row stride Wc, output produced in place in row-major order).

    Preferred: K = 1 (whole row per chunk) and the largest block_rows that fits — the whole shard when
    it fits. When even one whole-row chunk does not fit (wide rows: cb_normed is r*c intermediate tiles), fall
    back to block_rows = 1 with W chunks: c must divide Wc so the chunk grid lands exactly on the shard's
    row stride, and r = 1 keeps the chunk-major pack order equal to the shard's row-major order.
    """
    for r in range(rows_cap, 0, -1):
        if _footprint(g, core_w_tiles, r, 1) <= budget:
            return r, core_w_tiles, 1
    if not g.is_rm:  # RM resident tilizes whole rows in place; the caller falls back to LOCAL_STICKS
        for c in range(core_w_tiles - 1, 0, -1):
            if core_w_tiles % c == 0 and _footprint(g, core_w_tiles, 1, core_w_tiles // c) <= budget:
                return 1, c, core_w_tiles // c
    raise NotImplementedError("rms_norm: row too wide for single-core residency; needs W-split regime R3")


def _cb_budget(device) -> int:
    """Per-core L1 bytes available to this program's CBs, minus a safety margin.

    The static CB region grows up from the allocator base and must end below the LOWEST allocated
    L1 buffer (tensors allocate top-down; the freshly allocated output is already among them).
    The largest free block is the wrong bound: a free hole above a low buffer is unusable by CBs.
    block_table addresses are relative to the allocator base.
    """
    l1_view = ttnn.device.get_memory_view(device, ttnn.BufferType.L1)
    lowest = l1_view.total_bytes_per_bank
    for blk in l1_view.block_table:
        if blk["allocated"] == "yes":
            lowest = min(lowest, int(blk["address"]))
    return lowest - L1_SAFETY_MARGIN


def _cb(index, num_pages, page_size, data_format, core_ranges):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def create_program_descriptor(x, gamma, out, epsilon, compute_config) -> ttnn.ProgramDescriptor:
    g = _Geometry(x, gamma, out, bool(compute_config.fp32_dest_acc_en))
    device = x.device()

    # ---------------- work distribution ----------------
    # Per core: (core, start_tile_row, num_tile_rows, stick_base, stick_end).
    #   interleaved (R1): tile-rows split over the full grid (split_work_to_cores).
    #   HEIGHT_SHARDED (R6): the shard grid IS the split — core i owns shard i (ragged last shard).
    # RM tile-rows are core-local: tile-row t covers sticks [stick_base + 32t, min(+32, stick_end)).
    work = []
    if g.is_sharded:
        all_cores = g.shard_grid
        for i, core in enumerate(ttnn.corerange_to_cores(all_cores, None, g.shard_row_wise)):
            if g.is_rm:
                stick_base = min(i * g.shard_h, g.num_valid_sticks)
                stick_end = min(stick_base + g.shard_h, g.num_valid_sticks)
                n = math.ceil((stick_end - stick_base) / TILE)
                start = 0
            else:
                tr = g.shard_h // TILE
                start = min(i * tr, g.tensor_rows_tiles)
                n = min(tr, g.tensor_rows_tiles - start)
                stick_base = stick_end = 0
            work.append((core, start, n, stick_base, stick_end))
        assert sum(w[2] for w in work) >= 1
    else:
        grid = device.compute_with_storage_grid_size()
        (_, all_cores, core_group_1, core_group_2, rows_per_core_1, rows_per_core_2) = ttnn.split_work_to_cores(
            grid, g.tensor_rows_tiles, row_wise=True
        )
        start = 0
        for group, per_core in ((core_group_1, rows_per_core_1), (core_group_2, rows_per_core_2)):
            if per_core == 0:
                continue
            for core in ttnn.corerange_to_cores(group, None, True):
                work.append((core, start, per_core, start * TILE, g.num_valid_sticks))
                start += per_core
        assert start == g.tensor_rows_tiles
    rows_cap = max(1, max(w[2] for w in work))

    # ---------------- block selection ----------------
    core_w_tiles = g.tensor_w_tiles  # R1: each core owns the whole row
    w_tile_start = 0
    budget = _cb_budget(device)
    if g.rm_in_place and _footprint(g, core_w_tiles, 1, 1) > budget:
        # Whole-row in-place tilize does not fit: restride the resident sticks instead (W chunks allowed).
        g.rm_in_place = g.x_resident = False
    block_rows, w_chunk_tiles, num_w_chunks, x_block_depth = _select_blocks(g, core_w_tiles, rows_cap, budget)
    row_stride_tiles = num_w_chunks * w_chunk_tiles

    # ---------------- circular buffers ----------------
    interm = g.interm_fmt  # compute-intermediate format (DEST width)

    def _resident(index, tensor, page_size):
        # Zero-copy CB over the tensor's local shard; tile-sized pages so CB credit counts tiles.
        desc = ttnn.cb_descriptor_from_sharded_tensor(index, tensor)
        desc.format_descriptors = [
            ttnn.CBFormatDescriptor(buffer_index=index, data_format=tensor.dtype, page_size=page_size)
        ]
        return desc

    cbs = [
        _cb(CB_SUMSQ, block_rows * _n_sq(g, core_w_tiles), g.s_sq, g.sumsq_fmt, all_cores),
        _cb(CB_RSTD, block_rows, g.s_f, interm, all_cores),
    ]
    if g.is_rm or not g.x_resident:
        cbs.append(_cb(CB_X, x_block_depth * block_rows * row_stride_tiles, g.s_x, x.dtype, all_cores))
    if g.is_rm:
        if g.x_resident:
            cbs.append(_resident(CB_X_RM, x, g.s_x))
            cbs.append(_resident(CB_OUT_RM, out, g.s_o))
        else:
            cbs.append(_cb(CB_X_RM, STREAM_DEPTH * w_chunk_tiles, g.s_x, x.dtype, all_cores))
            cbs.append(_cb(CB_OUT_RM, STREAM_DEPTH * w_chunk_tiles, g.s_o, out.dtype, all_cores))
    elif g.x_resident:
        cbs.append(_resident(CB_X, x, g.s_x))
        cbs.append(_resident(CB_OUT, out, g.s_o))
    else:
        cbs.append(_cb(CB_OUT, STREAM_DEPTH * w_chunk_tiles, g.s_o, out.dtype, all_cores))
    if g.w_partial_elems:
        cbs.append(_cb(CB_REDUCE_MASK, 1, g.s_sq, g.sumsq_fmt, all_cores))
    if g.has_gamma:
        cbs.append(_cb(CB_GAMMA, STREAM_DEPTH * w_chunk_tiles, g.s_g, gamma.dtype, all_cores))
    if g.has_gamma or g.is_rm:
        cbs.append(_cb(CB_NORMED, block_rows * w_chunk_tiles, g.s_f, interm, all_cores))

    # ---------------- per-core runtime args ----------------
    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    x_addr = x.buffer_address()
    gamma_addr = gamma.buffer_address() if g.has_gamma else 0
    out_addr = out.buffer_address()
    for core, start, n, stick_base, stick_end in work:
        reader_rt[core.x][core.y] = [x_addr, gamma_addr, start, n, w_tile_start, stick_base, stick_end]
        writer_rt[core.x][core.y] = [out_addr, start, n, w_tile_start, stick_base, stick_end]
        compute_rt[core.x][core.y] = [n]

    # x / out placement as seen by the dataflow kernels (single source for reader and writer).
    if not g.is_sharded:
        placement = PLACEMENT_INTERLEAVED
    elif g.x_resident:
        placement = PLACEMENT_RESIDENT
    else:
        placement = PLACEMENT_LOCAL_STICKS

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
            placement,
            int(g.gamma_tiled),
            g.stick_page_bytes,
            g.g_row0_bytes,
            g.g_face1_offset,
        ]
    )
    reader_ct.extend(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    reader_ct.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if g.has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    writer_ct = (
        CB_IDS
        + blocking_ct
        + [g.tensor_w_tiles, int(g.is_rm), g.W, g.num_valid_sticks, g.out_elem, placement, g.stick_page_bytes]
    )
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
