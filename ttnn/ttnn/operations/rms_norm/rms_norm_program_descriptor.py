# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm ProgramDescriptor — regimes R1 (row-resident, rows split across the full grid), R6
(HEIGHT_SHARDED / single-slice sharded, local reduction) and R3 (W split across a group of G cores,
partial sums of squares gathered to the group root, rstd multicast back).

Every block knob of op_design.md's Blocking Model is computed exactly once here and passed down
as a compile-time / runtime arg; CB page counts and kernel trip counts derive from those knobs:

    block_rows        (r)  tile-rows per block                 -> CT (reader/compute/writer)
    w_chunk_tiles     (c)  tile-columns per normalize chunk     -> CT
    num_w_chunks      (K)  chunks per row; row_stride = K*c     -> CT
    core_w_tiles      (Wc) W slice per core (R1: whole row)     -> CT (one kernel group per distinct slice)
    w_tile_start           first W tile of the core's slice     -> RT (R1: 0)
    w_elem_start/end       the core's slice in elements          -> RT (R1: [0, W))
    group_size        (G)  cores sharing one row-group (R3)      -> CT
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
# R3 (W-split) only:
CB_PARTIAL = 9  # this core's per-row Σx² over its W slice (col-0 valid), handed to the writer
CB_GATHER = 10  # root: the G partials of each row, gathered by the writers (tile (row, slot) at row*G + slot)
CB_RSTD_ROOT = 11  # combine owner: its rows' combined rstd, handed to the writer (relay to the root / multicast)
CB_REDUCE_SCALER = 13  # R3 partial on the FPU reduce datapath: the SUM scaler tile (1.0)
CB_RSTD_GATHER = 12  # root: every owner's rstd rows (row i at tile i), the multicast source (multi-owner blocks)

# Semaphores (R3): the partial-gather counter on every combine owner, the rstd-relay counter on the root; the
# rstd multicast channel allocates from SEM_MCAST_BASE.
SEM_GATHER = 0
SEM_RSTD_GATHER = 1
SEM_MCAST_BASE = 2
MCAST_PREFIX = "rstd_mcast"

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
# R3 on interleaved input (logical W-split): each row-group is a gx x gy core rectangle (see _r3_group_rect),
# row-groups stack down the grid's y axis.
# Minimum W tiles per core's slice: bounds G = Wt / this (more cores = shorter DRAM phases, but a longer
# gather incast + combine). Measured (32-row decode, Blackhole 12x10): 5 is best over {2, 4, 5, 6, 8, 12}:
# 32x7168 G=44 6.8 us (G=110 8.8, G=18 7.1), 32x1024 G=6 4.1 us.
R3_MIN_SLICE_TILES = 5
# Cap on G (cores per row-group = the root's combine fan-in). 0 = no cap beyond R3_MIN_SLICE_TILES and the grid.
R3_MAX_GROUP_CORES = 0
# Regime predicate: interleaved input takes R3 iff R1 cannot hold one whole row on a core, or the W-split puts
# >= R3_OCCUPANCY_RATIO x R1's cores to work (R1 runs min(Rt, grid) cores). Pinned from measurement: R3 wins
# at 1.2x (3200x2048: 73.5 -> 70.7 us; 3200x7168: 252 -> 226 us) and above (640x7168: 61 -> 47 us), loses at
# 0.64x (1504x544: 11.6 vs 19.3 us). Prefill (Rt >= grid cores) never flips: R3 cannot exceed the grid.
R3_OCCUPANCY_RATIO = 1.2
# W chunk cap for R3 slices (0 = whole slice per chunk). Live knob, parked: finer chunks pipeline read/compute/
# write but measured slower on decode (32x7168: c=8 8.17 us, c=4 8.34, c=2 8.87) — per-chunk overhead outweighs
# the overlap at 4-8 tile slices.
R3_STREAM_CHUNK_TILES = 0
# Stall shadow (R3, TILE + gamma): x·γ while the combine is in flight; only ·rstd remains after rstd arrives.
R3_GAMMA_SHADOW = True
# Resident multi-row shards (BLOCK grids) take the shadow only from this many tile-rows per block: below it the
# gamma read (re-read by every core of a grid column) outlasts sumsq + combine and the shadow would expose it.
R3_SHADOW_RESIDENT_MIN_ROWS = 4
# R3 combine distribution (row-wise reduce-scatter, catalog `tensix_all_reduce`): the rows of a block are split
# over the row-group's G cores in runs of combine_rows = ceil(block_rows / G); core o combines rows
# [o*q, (o+1)*q) (G partials each + the rstd post-op) and relays its rstd rows to the root, which multicasts the
# whole block. False = the root combines every row (q = block_rows; one combine owner).
R3_SCATTER_COMBINE = True
# R3 partial_block datapath: the per-core Σx² tile of each row is ONE tile, where the FPU matmul reduce
# (ReduceTile, scaler 1.0) beats accumulate + SFPU collapse (catalog `reduce_block` / `row_reduce_accumulate`:
# AccumulateViaAdd only wins from ~4 tiles per output). Slices with a partial tail keep AccumulateViaAdd (masked).
R3_PARTIAL_REDUCE_TILE = True
# ... interleaved into the sumsq loop, one reduce per combine owner's run of rows, so the partial gather overlaps
# the rest of sumsq instead of trailing it.
R3_PARTIAL_INTERLEAVE = True
# FORCE_W_SPLIT is a test hook (regime-pinned tests flip it to exercise R3 on shapes R1 can hold).
FORCE_W_SPLIT = False


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

        # ---- placement (R1 interleaved | R6 local shard | R3 W-split) ----
        mem = x.memory_config()
        self.memory_layout = mem.memory_layout
        self.is_sharded = self.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
        self.stick_page_bytes = x.buffer_aligned_page_size() if self.is_rm else 0
        self.shard_h = self.shard_w = 0
        self.shard_cols = 1  # W slices of the shard grid (> 1 -> R3 cross-core combine)
        self.rm_in_place = False
        if self.is_sharded:
            spec = mem.shard_spec
            self.shard_h = int(spec.shape[0])  # rows (TILE: multiple of 32; RM: sticks)
            self.shard_w = int(spec.shape[1])  # elements (TILE: multiple of 32; RM: L1-aligned granule)
            self.shard_grid = spec.grid
            self.shard_row_wise = spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
            self.shard_cols = math.ceil(self.W / self.shard_w)
            if self.shard_cols == 1:
                # R6: the shard spans the whole row. TILE: the padded row; RM: the (granule padded) row.
                if (self.shard_w != self.tensor_w_tiles * TILE) if not self.is_rm else (self.shard_w < self.W):
                    raise NotImplementedError(f"rms_norm: shard width {self.shard_w} must span the whole row")
                # RM shard is tilize-consumable in place only when its stick stride is exactly Wc*32 elements
                # and every tile-row of the shard allocation is whole (32 sticks); otherwise the reader
                # restrides the resident sticks into cb_x_rm (local L1 copy, no DRAM).
                self.rm_in_place = self.is_rm and self.w_partial_elems == 0 and self.shard_h % TILE == 0
            elif self.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
                raise NotImplementedError("rms_norm: HEIGHT_SHARDED shard must span the whole row")
            elif self.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED and not self.shard_row_wise:
                raise NotImplementedError("rms_norm: BLOCK_SHARDED requires ROW_MAJOR shard orientation")
        # TILE shards are consumed in place (cb_x / cb_out backed by the shards). RM shards are only when
        # rm_in_place; W-split RM shards (narrow, granule-wide sticks) are restrided into cb_x_rm.
        self.x_resident = self.is_sharded and (not self.is_rm or self.rm_in_place)
        # R3 sets these (W-split): G cores per row-group.
        self.w_split = False
        self.group_size = 1
        # R3 stall shadow (TILE + gamma): x·γ is computed into a block-wide cb_normed while the combine is in
        # flight. Set by create_program_descriptor once the regime is known (falls back if it does not fit).
        self.gamma_shadow = False
        # Does any core's slice end in a partial tile (-> masked tail sumsq tile, cb_reduce_mask)? R1/R6: the
        # tensor's last tile; R3: set by the planner from the per-core slices.
        self.sq_partial = self.w_partial_elems != 0


def _n_sq(g: _Geometry, core_w_tiles: int) -> int:
    """Sumsq tiles per row: one L1-accumulated tile over the whole tiles + the squared partial tail.

    Derived from the `core_w_tiles` knob (not the tensor width) — the compute kernel derives the same
    quantity from the same CT args, so the CB size and the kernel's push count cannot drift apart.
    """
    p = 1 if g.sq_partial else 0
    return (1 if core_w_tiles - p > 0 else 0) + p


def _combine_rows(g: _Geometry, r: int) -> int:
    """Rows per combine owner (q) for a block of r tile-rows — single source for CB sizes and kernel CT args."""
    if not g.w_split:
        return 0
    return math.ceil(r / g.group_size) if R3_SCATTER_COMBINE else r


def _footprint(g: _Geometry, core_w_tiles: int, r: int, K: int, x_depth: int = 1) -> int:
    """F(r, K) of l1_ledger.md (sum over the CBs this layout/gamma variant allocates)."""
    c = math.ceil(core_w_tiles / K)
    row_stride = K * c
    p = 1 if g.sq_partial else 0
    f = 0
    if g.is_rm or not g.x_resident:
        f += x_depth * r * row_stride * g.s_x  # cb_x (TILE sharded: the resident input shard, not allocated)
    if g.is_rm and not g.x_resident:
        f += STREAM_DEPTH * c * g.s_x  # cb_x_rm (RM in place: the resident input shard)
    f += p * g.s_sq  # cb_reduce_mask (matches cb_sumsq's srcB format)
    f += r * g.s_f  # cb_rstd
    f += r * _n_sq(g, core_w_tiles) * g.s_sq  # cb_sumsq
    if g.has_gamma or g.is_rm:
        f += r * (row_stride if g.gamma_shadow else c) * g.s_f  # cb_normed (block-wide under the stall shadow)
    if g.has_gamma:
        f += STREAM_DEPTH * c * g.s_g  # cb_gamma
    if not g.x_resident:
        f += STREAM_DEPTH * c * g.s_o  # cb_out | cb_out_rm (resident output shard when x_resident)
    if g.w_split:
        f += r * g.s_f  # cb_partial
        q = _combine_rows(g, r)
        f += g.group_size * q * g.s_f  # cb_gather (allocated on every core so its address is uniform)
        f += q * g.s_f  # cb_rstd_root
        if q < r:
            f += r * g.s_f  # cb_rstd_gather (root's multicast source; uniform address)
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
    if g.w_split and R3_STREAM_CHUNK_TILES > 0:
        K = max(K, math.ceil(core_w_tiles / R3_STREAM_CHUNK_TILES))
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


class _CoreWork:
    """One core's assignment: a run of tile-rows x a W slice, and its place in its R3 row-group."""

    def __init__(self, core, start, n, stick_base, stick_end, w_elem_start, w_elem_end, group=0, slot=0, root=None):
        self.core = core
        self.start = start  # TILE: first global tile-row
        self.n = n  # tile-rows
        self.stick_base = stick_base  # RM: first stick of tile-row 0
        self.stick_end = stick_end  # RM: one past the core's last valid stick
        self.w_elem_start = w_elem_start  # the core's W slice [w_elem_start, w_elem_end) in elements
        self.w_elem_end = w_elem_end
        self.w_tile_start = w_elem_start // TILE
        self.core_w_tiles = math.ceil((w_elem_end - w_elem_start) / TILE)
        self.w_partial = (w_elem_end - w_elem_start) % TILE  # valid elems of the slice's last tile (0 = full)
        self.group = group  # R3 row-group id
        self.slot = slot  # R3 index within the row-group (0 = root)
        self.root = root if root is not None else core  # R3 logical root core of the row-group


def _cores_to_range_set(cores):
    """CoreRangeSet over `cores`, merged into per-y runs of consecutive x."""
    by_y = {}
    for c in cores:
        by_y.setdefault(c.y, []).append(c.x)
    ranges = []
    for y, xs in by_y.items():
        xs = sorted(set(xs))
        run_start = prev = xs[0]
        for x in xs[1:] + [None]:
            if x is not None and x == prev + 1:
                prev = x
                continue
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(run_start, y), ttnn.CoreCoord(prev, y)))
            if x is not None:
                run_start = prev = x
    return ttnn.CoreRangeSet(set(ranges))


def _plan_sharded(g: _Geometry):
    """Shard grid -> per-core work (R6 when the shard spans the row; R3 when W is split across shard columns).

    Shard (i, j) = (row slab, W slice). HEIGHT: (idx, 0); WIDTH: (0, idx); BLOCK (ROW_MAJOR): (y - y0, x - x0).
    Row-group = shard row i (all shard columns of that slab); root = its j = 0 core.
    """
    ML = ttnn.TensorMemoryLayout
    cores = list(ttnn.corerange_to_cores(g.shard_grid, None, g.shard_row_wise))
    origin = g.shard_grid.bounding_box().start
    ij = []
    for idx, core in enumerate(cores):
        if g.memory_layout == ML.HEIGHT_SHARDED:
            ij.append((idx, 0))
        elif g.memory_layout == ML.WIDTH_SHARDED:
            ij.append((0, idx))
        else:
            ij.append((core.y - origin.y, core.x - origin.x))
    n_cols = 1 + max(j for _, j in ij)
    if n_cols != g.shard_cols:
        raise NotImplementedError(
            f"rms_norm: shard grid has {n_cols} W slices but W={g.W} needs {g.shard_cols} of width {g.shard_w}"
        )
    roots = {i: core for core, (i, j) in zip(cores, ij) if j == 0}
    work = []
    for core, (i, j) in zip(cores, ij):
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
        e0 = j * g.shard_w
        e1 = min(e0 + g.shard_w, g.W)
        work.append(_CoreWork(core, start, n, stick_base, stick_end, e0, e1, group=i, slot=j, root=roots[i]))
    if g.shard_cols > 1:
        if g.gamma_tiled and g.shard_w % TILE != 0:
            raise NotImplementedError("rms_norm: TILE gamma needs tile-aligned W slices")
        g.w_split = True
        g.group_size = g.shard_cols
        g.sq_partial = any(w.w_partial for w in work)
    return work, g.shard_grid


def _plan_r1(g: _Geometry, grid):
    """R1: tile-rows split over the full grid (split_work_to_cores), whole row per core."""
    (_, all_cores, core_group_1, core_group_2, rows_per_core_1, rows_per_core_2) = ttnn.split_work_to_cores(
        grid, g.tensor_rows_tiles, row_wise=True
    )
    work = []
    start = 0
    for group, per_core in ((core_group_1, rows_per_core_1), (core_group_2, rows_per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            work.append(_CoreWork(core, start, per_core, start * TILE, g.num_valid_sticks, 0, g.W))
            start += per_core
    assert start == g.tensor_rows_tiles
    return work, all_cores


def _r1_fits(g: _Geometry, budget: int) -> bool:
    """R1 predicate: one whole row (at the finest W chunking) fits one core's CB budget."""
    c_floor = min(g.tensor_w_tiles, W_CHUNK_FLOOR_TILES)
    return _footprint(g, g.tensor_w_tiles, 1, math.ceil(g.tensor_w_tiles / c_floor)) <= budget


def _r3_group_rect(g: _Geometry, grid, n_groups: int):
    """R3 interleaved W-group shape (gx, gy): the core rectangle one row-group spans.

    Row-groups stack down the grid's y axis, so a group may take up to grid.y // n_groups grid rows. The
    group size G = gx*gy is the largest product within the caps — every core keeps >= R3_MIN_SLICE_TILES
    W tiles and the root's combine fan-in stays <= R3_MAX_GROUP_CORES — preferring fewer grid rows on ties
    (a 1-row group multicasts along a single NoC row).
    """
    gy_max = max(1, grid.y // n_groups)
    g_cap = max(1, g.tensor_w_tiles // R3_MIN_SLICE_TILES)
    if R3_MAX_GROUP_CORES > 0:
        g_cap = min(g_cap, R3_MAX_GROUP_CORES)
    best = (1, 1)
    for gy in range(1, gy_max + 1):
        gx = min(grid.x, g_cap // gy)
        if gx >= 1 and gx * gy > best[0] * best[1]:
            best = (gx, gy)
    return best


def _plan_r3_interleaved(g: _Geometry, grid):
    """R3 logical W-split on an interleaved tensor. Row-group i is a gx x gy core rectangle at grid rows
    [i*gy, (i+1)*gy); its G = gx*gy cores (row-major slot order = the rstd mcast group order) each read a
    contiguous tile-aligned W slice from DRAM (an even split: the first Wt % G slices get one extra tile).
    Tile-rows are split evenly over min(grid.y, Rt) row-groups."""
    num_groups = min(grid.y, g.tensor_rows_tiles)
    gx, gy = _r3_group_rect(g, grid, num_groups)
    G = gx * gy
    w_base, w_rem = divmod(g.tensor_w_tiles, G)
    base, rem = divmod(g.tensor_rows_tiles, num_groups)
    work, cores = [], []
    start = 0
    for i in range(num_groups):
        n = base + (1 if i < rem else 0)
        root = ttnn.CoreCoord(0, i * gy)
        for j in range(G):
            core = ttnn.CoreCoord(j % gx, i * gy + j // gx)
            ts = j * w_base + min(j, w_rem)
            nt = w_base + (1 if j < w_rem else 0)
            e0 = ts * TILE
            e1 = min(g.W, (ts + nt) * TILE)
            work.append(_CoreWork(core, start, n, start * TILE, g.num_valid_sticks, e0, e1, group=i, slot=j, root=root))
            cores.append(core)
        start += n
    assert start == g.tensor_rows_tiles
    g.w_split = G > 1
    g.group_size = G
    g.sq_partial = any(w.w_partial for w in work)
    return work, _cores_to_range_set(cores)


def _take_r3_interleaved(g: _Geometry, grid, budget: int) -> bool:
    """Interleaved regime predicate: R3 when R1 cannot hold one whole row on a core, or when the W-split
    occupies at least R3_OCCUPANCY_RATIO x the cores R1 would (the decode / few-row profile: R1 runs Rt cores)."""
    if FORCE_W_SPLIT or not _r1_fits(g, budget):
        return True
    r1_cores = min(g.tensor_rows_tiles, grid.x * grid.y)
    n_groups = min(grid.y, g.tensor_rows_tiles)
    gx, gy = _r3_group_rect(g, grid, n_groups)
    return gx * gy > 1 and n_groups * gx * gy >= R3_OCCUPANCY_RATIO * r1_cores


def create_program_descriptor(x, gamma, out, epsilon, compute_config) -> ttnn.ProgramDescriptor:
    g = _Geometry(x, gamma, out, bool(compute_config.fp32_dest_acc_en))
    device = x.device()
    budget = _cb_budget(device)

    # ---------------- regime + work distribution ----------------
    #   interleaved: R1 when one whole row fits a core (pinned predicate _r1_fits), else R3 (logical W-split).
    #   sharded: the shard grid IS the split — R6 when the shard spans the row, R3 when W is split.
    if g.is_sharded:
        work, all_cores = _plan_sharded(g)
    else:
        grid = device.compute_with_storage_grid_size()
        if _take_r3_interleaved(g, grid, budget):
            work, all_cores = _plan_r3_interleaved(g, grid)
        else:
            work, all_cores = _plan_r1(g, grid)
    rows_cap = max(1, max(w.n for w in work))

    # ---------------- block selection ----------------
    # W extent the CBs are sized for: the widest slice; a resident shard's row stride is its allocated width.
    core_w_tiles = max(w.core_w_tiles for w in work)
    if g.x_resident:
        core_w_tiles = g.shard_w // TILE
    if g.rm_in_place and _footprint(g, core_w_tiles, 1, 1) > budget:
        # Whole-row in-place tilize does not fit: restride the resident sticks instead (W chunks allowed).
        g.rm_in_place = g.x_resident = False
    g.gamma_shadow = R3_GAMMA_SHADOW and g.w_split and g.has_gamma and not g.is_rm
    try:
        block_rows, w_chunk_tiles, num_w_chunks, x_block_depth = _select_blocks(g, core_w_tiles, rows_cap, budget)
    except NotImplementedError:
        if not g.gamma_shadow:
            raise
        g.gamma_shadow = False  # the block-wide cb_normed does not fit: normalize then scale after rstd
        block_rows, w_chunk_tiles, num_w_chunks, x_block_depth = _select_blocks(g, core_w_tiles, rows_cap, budget)
    if g.gamma_shadow and g.x_resident and 1 < block_rows < R3_SHADOW_RESIDENT_MIN_ROWS:
        # Resident multi-row shards (BLOCK grids): gamma is the only DRAM traffic and every core of a grid column
        # re-reads the same slice — measured 8 us on 640x7168 [64,1024]x(7,10), longer than sumsq + combine. The
        # shadow would pull that read onto the critical path; scaling after rstd keeps it hidden. Dropping the
        # shadow only shrinks cb_normed, so the selected blocking still fits.
        g.gamma_shadow = False
    row_stride_tiles = num_w_chunks * w_chunk_tiles
    combine_rows = _combine_rows(g, block_rows)  # q: rows per combine owner (R3)
    partial_reduce_tile = R3_PARTIAL_REDUCE_TILE and g.w_split and not g.sq_partial

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
    if g.sq_partial:
        cbs.append(_cb(CB_REDUCE_MASK, 1, g.s_sq, g.sumsq_fmt, all_cores))
    if partial_reduce_tile:
        cbs.append(_cb(CB_REDUCE_SCALER, 1, ttnn.tile_size(ttnn.bfloat16), ttnn.bfloat16, all_cores))
    if g.has_gamma:
        cbs.append(_cb(CB_GAMMA, STREAM_DEPTH * w_chunk_tiles, g.s_g, gamma.dtype, all_cores))
    if g.has_gamma or g.is_rm:
        normed_tiles = row_stride_tiles if g.gamma_shadow else w_chunk_tiles
        cbs.append(_cb(CB_NORMED, block_rows * normed_tiles, g.s_f, interm, all_cores))
    if g.w_split:
        # Partials leave DEST once (no L1 accumulation), so their pages follow the DEST width like cb_rstd: a
        # Float32 page under 16-bit DEST stores nothing extra and doubles cb_gather (G tiles per row).
        cbs.append(_cb(CB_PARTIAL, block_rows, g.s_f, interm, all_cores))
        # Every core allocates cb_gather (sized G*block_rows so each block starts at the CB base) so its L1
        # address — which contributors write into on the root — is the same on every core.
        # Each combine owner receives G partials for each of its <= combine_rows rows; the writer always pushes
        # the full G*combine_rows quantum so every block starts at the CB base (contributors address it remotely).
        cbs.append(_cb(CB_GATHER, g.group_size * combine_rows, g.s_f, interm, all_cores))
        cbs.append(_cb(CB_RSTD_ROOT, combine_rows, g.s_f, interm, all_cores))
        if combine_rows < block_rows:
            cbs.append(_cb(CB_RSTD_GATHER, block_rows, g.s_f, interm, all_cores))

    # x / out placement as seen by the dataflow kernels (single source for reader and writer).
    if not g.is_sharded:
        placement = PLACEMENT_INTERLEAVED
    elif g.x_resident:
        placement = PLACEMENT_RESIDENT
    else:
        placement = PLACEMENT_LOCAL_STICKS

    # ---------------- kernels: one reader/writer/compute triple per distinct W slice ----------------
    # core_w_tiles and the slice's partial-tile width stay compile-time: a ragged last slice (or the RM shard
    # granule) gets its own kernel group.
    x_addr = x.buffer_address()
    gamma_addr = gamma.buffer_address() if g.has_gamma else 0
    out_addr = out.buffer_address()
    # R3: NoC coords of each row-group's cores by slot (combine owners are addressed by slot).
    group_noc = {}
    for w in sorted(work, key=lambda w: w.slot):
        v = device.worker_core_from_logical_core(w.core)
        group_noc.setdefault(w.group, []).append(v.x | (v.y << 16))
    kernel_groups = {}
    for w in work:
        kernel_groups.setdefault((w.core_w_tiles, w.w_partial), []).append(w)

    readers, writers, computes = [], [], []
    for (group_w_tiles, group_partial), members in kernel_groups.items():
        cores = _cores_to_range_set([w.core for w in members])
        reader_rt = ttnn.RuntimeArgs()
        writer_rt = ttnn.RuntimeArgs()
        compute_rt = ttnn.RuntimeArgs()
        for w in members:
            c = w.core
            root_v = device.worker_core_from_logical_core(w.root)
            is_root = int(w.slot == 0)
            reader_rt[c.x][c.y] = [
                x_addr, gamma_addr, w.start, w.n, w.w_tile_start, w.stick_base, w.stick_end,
                w.w_elem_start, w.w_elem_end,
            ]  # fmt: skip
            writer_rt[c.x][c.y] = [
                out_addr, w.start, w.n, w.w_tile_start, w.stick_base, w.stick_end, w.w_elem_start, w.w_elem_end,
                is_root, w.slot, root_v.x, root_v.y,
            ] + (group_noc[w.group] if g.w_split else [])  # fmt: skip
            compute_rt[c.x][c.y] = [w.n, w.slot]

        blocking_ct = [block_rows, w_chunk_tiles, num_w_chunks, group_w_tiles]
        reader_ct = (
            CB_IDS
            + blocking_ct
            + [
                g.tensor_w_tiles,
                group_partial,
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
                int(partial_reduce_tile),
                CB_REDUCE_SCALER,
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
            + [int(g.w_split), g.group_size, CB_PARTIAL, CB_GATHER, CB_RSTD_ROOT, SEM_GATHER]
            + [combine_rows, CB_RSTD_GATHER, SEM_RSTD_GATHER]
        )
        writer_ct.extend(ttnn.TensorAccessorArgs(out).get_compile_time_args())
        compute_ct = (
            CB_IDS
            + blocking_ct
            + [group_partial, int(g.has_gamma), int(g.is_rm), _f32_bits(1.0 / g.W), _f32_bits(epsilon)]
            + [int(g.w_split), g.group_size, CB_PARTIAL, CB_GATHER, CB_RSTD_ROOT, int(g.gamma_shadow)]
            + [combine_rows, int(partial_reduce_tile), CB_REDUCE_SCALER]
            + [int(partial_reduce_tile and R3_PARTIAL_INTERLEAVE)]
        )

        readers.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
                core_ranges=cores,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            )
        )
        writers.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
                core_ranges=cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            )
        )
        computes.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
                core_ranges=cores,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=compute_config,
            )
        )

    semaphores = []
    if g.w_split:
        semaphores.append(ttnn.SemaphoreDescriptor(id=SEM_GATHER, core_ranges=all_cores, initial_value=0))
        semaphores.append(ttnn.SemaphoreDescriptor(id=SEM_RSTD_GATHER, core_ranges=all_cores, initial_value=0))
    descriptor = ttnn.ProgramDescriptor(kernels=[], semaphores=semaphores, cbs=cbs)
    if g.w_split:
        # rstd broadcast: one multicast group per row-group (consecutive runs of G cores in shard / row-major
        # order), sender = the group's slot-0 core (the root). The writer runs on NOC_1.
        order = (
            ttnn.McastCoreOrder.RowMajor if (not g.is_sharded or g.shard_row_wise) else ttnn.McastCoreOrder.ColumnMajor
        )
        mcast = ttnn.Mcast(
            device,
            ttnn.McastConfig(noc=ttnn.NOC.NOC_1),
            all_cores,
            g.group_size,
            ttnn.McastFixedSenderConfig(sender_index=0),
            order,
        )
        mcast.attach(descriptor, MCAST_PREFIX, writers, SEM_MCAST_BASE)
    else:
        for wk in writers:
            ttnn.attach_absent_mcast(wk, MCAST_PREFIX)
    descriptor.kernels = readers + writers + computes
    return descriptor
