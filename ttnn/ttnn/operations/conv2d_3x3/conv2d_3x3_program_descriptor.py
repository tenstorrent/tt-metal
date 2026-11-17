# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_3x3 program descriptor: blocking (single source of every block knob), CBs, kernels, args.

Regime R0 (interleaved-slab): output pixels M are split over the grid in contiguous tile ranges;
each core runs `num_m_blocks_this_core` M-blocks of `block_m_tiles` tile-rows x all C_out, walking
K in `num_k_blocks` blocks of `block_k_tiles` that never straddle a kernel row (kh).

Regime R3 (TILE input) keeps that blocking. The reader streams the M-block's footprint tile-rows
into `cb_input_tiles` and compute untilizes them into `cb_halo_slab` (bf16 RM, one tile-row = 32
pixels), from which the reader gathers exactly as on the ROW_MAJOR path.

Regime R2 (HEIGHT_SHARDED input, either layout) keeps the same pipeline on the input shard grid: work
index i runs on shard core i and owns output shard i. The reader fills the slab with page runs from
any owner core, `cb_output_tiles` is backed on the output shard, and the writer stores nothing.

Regime R4 (BLOCK_SHARDED input, more than one shard column) splits K: core (r, c) owns output rows of
grid row r and channel slice c of the input. It runs the same pipeline over its slice against the
column's weight K-slice and hands its fp32 partials [bm, all C_out] to the writer, which scatters N-slice
y to column y of the same grid row (row-local reduce-scatter, two semaphores per core). Column y sums the
k_cols contributions, adds bias, applies ReLU and packs its output shard [bm, out_n_tiles] in place. A
single shard column is R2 on that column.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
TILE_BYTES_BF16 = 2 * TILE * TILE
TILE_BYTES_FP32 = 4 * TILE * TILE
BF16_BYTES = 2

# fp32 DEST, half sync: 4 tiles per DEST half. Bounds the matmul subblock, never a block.
DEST_TILES_FP32_HALF_SYNC = 4

# L1 left free for L1-resident tensors other code keeps alive while this program runs.
L1_TENSOR_HEADROOM_BYTES = 128 * 1024
# Sharded calls: CBs must clear this call's input and output shards plus one more such pair, the one a
# chained model (and a cache-hit re-dispatch) keeps alive. L1 buffers grow down from the top, CBs up from
# the base, and every enqueue re-checks that the two do not meet.
RESIDENT_SHARD_SETS = 2

# CB indices (semantic names; slots are arbitrary but fixed).
CB_HALO_SLAB = 0
CB_IM2COL_RM = 1
CB_IM2COL_TILES = 2
CB_WEIGHTS = 3
CB_BIAS = 4
CB_MATMUL_PARTIALS = 5
CB_OUTPUT_TILES = 6
CB_INPUT_TILES = 7
# R4 only: every row peer's fp32 N-slice for this core's output columns, and their sum before the bias.
CB_COMBINE_SLOTS = 8
CB_COMBINED = 9

# R4 reduce-scatter semaphores; the weight multicast channel allocates its own after them.
SEM_COMBINE_CREDIT = 0
SEM_COMBINE_ARRIVAL = 1
SEM_COMBINE_SEND_GO = 2
SEM_COMBINE_SEND_DONE = 3

# R4: share the partial scatter between the writer (NoC1, own and lower columns) and the reader (NoC0, higher
# columns). Every core sends and receives k_cols slices per round; on one NoC interface that traffic bounds
# the round. False keeps the whole scatter on the writer.
COMBINE_SPLIT_NOCS = True

# Tap-major im2col: when a K-block is whole taps of whole tiles, stride is 1 and no lane is padding, the
# reader lays each tap of a tile-row out as its own [32 pixels, C_in_p] block, which is one contiguous run of
# slab pixels per output image row, and compute tilizes it c_in_tiles wide. False keeps one im2col row per
# pixel (one gather read per pixel per K-block).
TAP_MAJOR_IM2COL = True

# Face-direct im2col: when one output pixel's step through the slab (stride * C_in_p channels) is exactly one
# tile-face row (16 elements), the 16 im2col rows of a 16x16 face are one contiguous run of slab pixels. The
# reader then writes the bf16 in0 tiles itself, one read per face run, and compute skips the tilize. False keeps
# the row gather + tilize.
FACE_DIRECT_IM2COL = True
FACE_DIM = 16
# On face-direct shapes, take the whole K walk as one K-block (op_design lamp P1): one matmul init, one pack of
# the fp32 block and no L1-accumulation passes per M-block. False keeps one K-block per kernel row.
FACE_DIRECT_WHOLE_K_BLOCK = True

# HEIGHT-sharded ROW_MAJOR slab fill split by NoC: the reader (NoC0) reads the runs of its own and lower input
# shards, the writer (NoC1) those of higher shards, so no read response wraps around the ring. False keeps the
# whole fill on the reader.
SLAB_SPLIT_NOCS = True
# Face-direct in0 streaming: with one K-block per M-block, the reader pushes the in0 block in row groups of this
# many tile-rows (rounded down to whole matmul subblocks; the last group may be shorter) and compute runs one matmul
# call per group, so the matmul overlaps the gather. A value >= block_m_tiles pushes and multiplies the whole block
# at once.
IN0_STREAM_TILE_ROWS = 2
# In0 streaming with a bias: row groups per bias-add call. Each call pays its own init and reconfig, and only the
# last call trails the gather; a value >= the group count adds the bias once after the last matmul.
BIAS_STREAM_GROUPS = 5
# Face-direct gather split between the two data-movement RISCs: the writer gathers every other row group on NoC1
# while the reader gathers the rest, since the gather is bound by read issue. Applies with the split slab fill (the
# writer then holds the slab plan) and in0 streaming. False keeps the whole gather on the reader.
GATHER_SPLIT_RISCS = True

# Buffer-depth knobs (Blocking Model depth table).
IM2COL_RM_DEPTH_ROWS = 2
WEIGHT_DEPTH = 2
OUT_DEPTH_ROW_GROUPS = 2
# cb_input_tiles depth in untilize batches: the reader reads batch b+1 while compute untilizes b.
INPUT_TILE_DEPTH_BATCHES = 2

# Slab-fill start offset per core index (op_design lamp P9): consecutive pixels rotate over DRAM banks,
# so offsetting each core's first read spreads the cores over the banks. 0 disables the stagger.
SLAB_READ_STAGGER_PX_PER_CORE = 0

# Weight multicast (regime R1): one sender reads each weight K-block from DRAM once and multicasts it
# into every participant's cb_weights slot. Participants are the cores that run the most M-blocks
# (all but a ragged last core), and the channel is built only when at least this many of them exist.
# A large value disables it (every core streams its own weights, regime R0).
WEIGHT_MCAST_MIN_CORES = 2
WEIGHT_MCAST_PREFIX = "weight_mcast"
# Hardware multicast rectangles one channel may carry (dataflow_kernel_lib::MAX_MCAST_RECTANGLES).
MAX_MCAST_RECTANGLES = 3


def l1_cb_budget_bytes(resident_shard_bytes=0):
    """CB bytes per core. `resident_shard_bytes` is one call's input plus output shard on a core."""
    return (
        ttnn.get_max_worker_l1_unreserved_size() - L1_TENSOR_HEADROOM_BYTES - RESIDENT_SHARD_SETS * resident_shard_bytes
    )


def _div_up(a, b):
    return -(-a // b)


@dataclass(frozen=True)
class Conv3x3Geometry:
    input_height: int
    input_width: int
    c_in_p: int
    c_out: int
    padding: int
    stride: int
    dilation: int
    output_height: int
    output_width: int
    c_in: int
    tile_input: bool
    in0_tile_bytes: int
    weight_tile_bytes: int
    out_tile_bytes: int
    has_bias: bool
    # cb_output_tiles is backed on the output shard: its pages are not CB-region bytes, and every M-block
    # of a core must land inside the shard (block_m_tiles divides core_m_tiles).
    sharded_output: bool = False
    # Core columns K's channel component is split over (BLOCK_SHARDED: one input shard column each).
    k_cols: int = 1

    @property
    def k_split(self):
        return self.k_cols > 1

    @property
    def out_n_tiles(self):
        """Output tile columns per core: all of C_out, or one column's N-slice under a K split."""
        return _div_up(self.tensor_n_tiles, self.k_cols)

    @property
    def num_out_px(self):
        return self.output_height * self.output_width

    @property
    def tensor_m_tiles(self):
        return _div_up(self.num_out_px, TILE)

    @property
    def tensor_n_tiles(self):
        return _div_up(self.c_out, TILE)

    @property
    def num_in_px(self):
        return self.input_height * self.input_width

    @property
    def c_in_tiles(self):
        return self.c_in_p // TILE

    @property
    def tensor_kkh_tiles(self):
        return _div_up(3 * self.c_in_p, TILE)

    @property
    def tensor_k_tiles(self):
        return 3 * self.tensor_kkh_tiles

    @property
    def face_taps(self):
        """Taps of one kernel row in one face row (face-direct im2col)."""
        return max(1, FACE_DIM // self.c_in_p)

    @property
    def face_direct_capable(self):
        """A face's rows step through the slab by exactly one face row, the face's taps are adjacent pixels,
        the activation is bf16 like the slab, and no lane of a pixel is padding."""
        return (
            FACE_DIRECT_IM2COL
            and not self.tile_input
            and self.stride * self.c_in_p == FACE_DIM
            and (self.face_taps == 1 or self.dilation == 1)
            and self.c_in == self.k_cols * self.c_in_p
        )

    @property
    def band_last_tap(self):
        """Last kernel-row tap the gather reads. A face-direct face that holds the third tap also reads the
        next pixel into its K-padding lanes (zero weight rows), so the band reaches one tap further."""
        if self.face_direct_capable:
            return _div_up(3, self.face_taps) * self.face_taps - 1
        return 2


@dataclass(frozen=True)
class Conv3x3Blocking:
    grid_x: int
    core_m_tiles: int
    num_m_groups: int
    block_m_tiles: int
    block_k_tiles: int
    block_n_tiles: int
    num_k_blocks: int
    max_m_blocks_per_core: int
    band_len_px: int
    band_stride_px: int
    slab_tile_rows: int
    untilize_batch_rows: int
    out_subblock_h: int
    out_subblock_w: int
    combine_subblock_h: int
    combine_subblock_w: int
    in0_stream_rows: int
    im2col_rm_depth_rows: int
    weight_depth: int
    out_depth_row_groups: int
    footprint_bytes: int

    @property
    def slab_px(self):
        if self.slab_tile_rows:
            return self.slab_tile_rows * TILE
        return 2 * self.band_stride_px + self.band_len_px


def _largest_divisor_at_most(n, cap):
    for d in range(min(n, cap), 0, -1):
        if n % d == 0:
            return d
    return 1


def _out_subblock(block_m_tiles, block_n_tiles):
    """Width first (it is what pays), then the tallest height that still fits DEST."""
    sb_w = _largest_divisor_at_most(block_n_tiles, DEST_TILES_FP32_HALF_SYNC)
    sb_h = _largest_divisor_at_most(block_m_tiles, DEST_TILES_FP32_HALF_SYNC // sb_w)
    return sb_h, sb_w


def m_split(g: Conv3x3Geometry, num_cores):
    """(core_m_tiles, num_m_groups): contiguous output tile-row ranges over `num_cores` cores. On a sharded
    input this is also the output shard: roundup32(ceil(M / n)) / 32 == ceil(ceil(M / 32) / n)."""
    core_m_tiles = _div_up(g.tensor_m_tiles, num_cores)
    return core_m_tiles, _div_up(g.tensor_m_tiles, core_m_tiles)


def _core_m_ranges(tensor_m_tiles, core_m_tiles):
    return [(start, min(core_m_tiles, tensor_m_tiles - start)) for start in range(0, tensor_m_tiles, core_m_tiles)]


def _block_band_len(g: Conv3x3Geometry, o_first, o_last):
    """Pixels one kernel row of the block touches: from tap (kh, 0) of o_first to tap (kh, band_last_tap) of
    o_last."""
    ho0, wo0 = divmod(o_first, g.output_width)
    ho1, wo1 = divmod(o_last, g.output_width)
    return (ho1 - ho0) * g.stride * g.input_width + (wo1 - wo0) * g.stride + g.band_last_tap * g.dilation + 1


def _block_first_tap(g: Conv3x3Geometry, o_first):
    """Flat input index of tap (0, 0) of output pixel o_first (negative inside the top/left padding)."""
    ho0, wo0 = divmod(o_first, g.output_width)
    return (ho0 * g.stride - g.padding) * g.input_width + wo0 * g.stride - g.padding


def _tile_slab_rows(g: Conv3x3Geometry, f, band_len):
    """Tile-rows the reader's TILE slab plan places for one M-block (mirror of the reader kernel).
    Band kh is its in-image pixels rounded out to whole 32-pixel input tile-rows; a band that starts
    inside the previous band's tile-rows shares them, otherwise it is placed after them."""
    next_free = 0
    prev = None  # (t_lo, t_hi, slab_row) of the last non-empty band
    for kh in range(3):
        v_lo = f + kh * g.dilation * g.input_width
        lo, hi = max(v_lo, 0), min(v_lo + band_len, g.num_in_px)
        if lo >= hi:
            continue
        t_lo, t_hi = lo // TILE, _div_up(hi, TILE)
        slab_row = prev[2] + (t_lo - prev[0]) if prev is not None and t_lo < prev[1] else next_free
        next_free = slab_row + (t_hi - t_lo)
        prev = (t_lo, t_hi, slab_row)
    return next_free


@functools.lru_cache(maxsize=1024)
def _block_extents(g: Conv3x3Geometry, core_m_tiles, block_m_tiles):
    """(max band_len_px, max TILE slab tile-rows) over every M-block of every core."""
    worst_band, worst_slab_rows = 0, 0
    for start, actual in _core_m_ranges(g.tensor_m_tiles, core_m_tiles):
        for block_start in range(0, actual, block_m_tiles):
            tiles = min(block_m_tiles, actual - block_start)
            o_first = (start + block_start) * TILE
            o_last = min((start + block_start + tiles) * TILE, g.num_out_px) - 1
            band_len = _block_band_len(g, o_first, o_last)
            worst_band = max(worst_band, band_len)
            if g.tile_input:
                worst_slab_rows = max(worst_slab_rows, _tile_slab_rows(g, _block_first_tap(g, o_first), band_len))
    return worst_band, worst_slab_rows


def _default_untilize_batch_rows(g: Conv3x3Geometry, band_len, raw_slab_rows):
    """One untilize batch = the tile-rows one band can touch, so the reader fences about once per band."""
    if not g.tile_input:
        return 1
    return max(1, min(raw_slab_rows, _div_up(band_len, TILE) + 1))


def face_direct_im2col(g: Conv3x3Geometry, bk):
    """Face-direct gather: needs whole kernel rows per K-block, so every K-block has one column layout. Under a
    K split the matmul's L1-accumulating packs to the caller-owned partials are ordered only by the tilize's own
    pack between K-blocks, so without a tilize the K walk must be one K-block."""
    return g.face_direct_capable and bk % g.tensor_kkh_tiles == 0 and (not g.k_split or bk == g.tensor_k_tiles)


def _in0_stream_rows(g: Conv3x3Geometry, bm, bk, sb_h):
    """Tile-rows per in0 push and matmul call: whole matmul subblocks, at most IN0_STREAM_TILE_ROWS; the last group
    of a block may be shorter. Only face-direct with one K-block per M-block streams (the reader produces in0, and
    a K-block walk would need every row group of every K-block); otherwise the whole block."""
    if not (face_direct_im2col(g, bk) and bk == g.tensor_k_tiles and not g.k_split):
        return bm
    return min(bm, max(IN0_STREAM_TILE_ROWS // sb_h, 1) * sb_h)


def _footprint(g: Conv3x3Geometry, core_m_tiles, bm, bk, bn, ub_cap=None):
    num_k_blocks = g.tensor_k_tiles // bk
    max_m_blocks = _div_up(core_m_tiles, bm)
    weight_depth = 1 if num_k_blocks * max_m_blocks == 1 else WEIGHT_DEPTH
    band_len, raw_slab_rows = _block_extents(g, core_m_tiles, bm)
    band_stride = min(g.dilation * g.input_width, band_len)
    sb_h, sb_w = _out_subblock(bm, bn)
    ns = g.out_n_tiles
    combine_sb_h, combine_sb_w = _out_subblock(bm, ns)
    # R4: every row peer's slice of this core's output columns, plus their sum before the bias add.
    combine_bytes = (g.k_cols * bm * ns + (bm * ns if g.has_bias else 0)) * TILE_BYTES_FP32 if g.k_split else 0
    if g.tile_input:
        ub = _default_untilize_batch_rows(g, band_len, raw_slab_rows)
        if ub_cap is not None:
            ub = min(ub, ub_cap)
        # The slab is a whole number of batches: every reader push and compute untilize has one quantum.
        slab_tile_rows = _div_up(max(raw_slab_rows, 1), ub) * ub
        slab_bytes = slab_tile_rows * g.c_in_tiles * TILE_BYTES_BF16
        input_tile_bytes = INPUT_TILE_DEPTH_BATCHES * ub * g.c_in_tiles * g.in0_tile_bytes
    else:
        ub, slab_tile_rows = 1, 0
        slab_bytes = (2 * band_stride + band_len) * g.c_in_p * BF16_BYTES
        input_tile_bytes = 0
    total = (
        slab_bytes
        + input_tile_bytes
        + (0 if face_direct_im2col(g, bk) else IM2COL_RM_DEPTH_ROWS * bk * TILE_BYTES_BF16)
        + bm * bk * g.in0_tile_bytes
        + weight_depth * bk * bn * g.weight_tile_bytes
        + (ns * g.weight_tile_bytes if g.has_bias else 0)
        + bm * bn * TILE_BYTES_FP32
        + combine_bytes
        + (0 if g.sharded_output else OUT_DEPTH_ROW_GROUPS * sb_h * bn * g.out_tile_bytes)
    )
    return dict(
        num_k_blocks=num_k_blocks,
        max_m_blocks_per_core=max_m_blocks,
        weight_depth=weight_depth,
        band_len_px=band_len,
        band_stride_px=band_stride,
        slab_tile_rows=slab_tile_rows,
        untilize_batch_rows=ub,
        out_subblock_h=sb_h,
        out_subblock_w=sb_w,
        combine_subblock_h=combine_sb_h,
        combine_subblock_w=combine_sb_w,
        in0_stream_rows=_in0_stream_rows(g, bm, bk, sb_h),
        footprint_bytes=total,
    )


@functools.lru_cache(maxsize=256)
def select_blocking(g: Conv3x3Geometry, grid, l1_budget) -> Conv3x3Blocking:
    """The one place block extents are chosen. Order of the L1 solve (op_design Mechanism caps #1):
    shrink bk to divisors of the kh block down to one pixel's channels (costs no bytes), then shrink bm
    (re-streams weights and halo) and take the coarsest bk that fits it, then bk below the tap floor. bn
    is never shrunk. On a TILE input the untilize batch shrinks before any of them.

    `grid` is the (x, y) extent the work is split over; a HEIGHT-sharded input passes (shard cores, 1),
    and the descriptor maps work index i to shard core i. A BLOCK-sharded input passes (grid rows, 1):
    the M split runs down the rows, and every core of a row shares that row's M range."""
    grid_x, grid_y = grid
    core_m_tiles, num_m_groups = m_split(g, grid_x * grid_y)
    bn = g.tensor_n_tiles
    kkh = g.tensor_kkh_tiles

    divisors = [d for d in range(kkh, 0, -1) if kkh % d == 0]
    tap_floor = g.c_in_p // TILE if g.c_in_p % TILE == 0 else kkh
    bk_above_floor = [d for d in divisors if d >= tap_floor]
    if g.face_direct_capable and FACE_DIRECT_WHOLE_K_BLOCK:
        bk_above_floor = [g.tensor_k_tiles] + bk_above_floor
    bk_below_floor = [d for d in divisors if d < tap_floor]
    if g.sharded_output:
        bm_candidates = [d for d in range(core_m_tiles, 0, -1) if core_m_tiles % d == 0]
    else:
        bm_candidates = sorted({_div_up(core_m_tiles, n) for n in range(1, core_m_tiles + 1)}, reverse=True)

    # A smaller bm frees L1, so each bm retries the coarsest bk again: at a fixed bm, bk costs no bytes.
    search = [(bm, bk) for bm in bm_candidates for bk in bk_above_floor]
    search += [(bm_candidates[-1], bk) for bk in bk_below_floor]

    for bm, bk in search:
        # The untilize batch shrinks first: a smaller batch costs one more read fence per tile-row,
        # a smaller bk costs a whole K-block (tilize, matmul init, L1-acc pass).
        band_len, raw_slab_rows = _block_extents(g, core_m_tiles, bm)
        ub_default = _default_untilize_batch_rows(g, band_len, raw_slab_rows)
        fit = None
        for ub_cap in range(ub_default, 0, -1):
            candidate = _footprint(g, core_m_tiles, bm, bk, bn, ub_cap)
            if candidate["footprint_bytes"] <= l1_budget:
                fit = candidate
                break
        if fit is not None:
            return Conv3x3Blocking(
                grid_x=grid_x,
                core_m_tiles=core_m_tiles,
                num_m_groups=num_m_groups,
                block_m_tiles=bm,
                block_k_tiles=bk,
                block_n_tiles=bn,
                im2col_rm_depth_rows=IM2COL_RM_DEPTH_ROWS,
                out_depth_row_groups=OUT_DEPTH_ROW_GROUPS,
                **fit,
            )
    raise ValueError(
        f"conv2d_3x3: no blocking of {g.input_height}x{g.input_width}x{g.c_in_p}->{g.c_out} "
        f"fits the {l1_budget} B L1 CB budget"
    )


# ---------------------------------------------------------------------------
# Descriptor
# ---------------------------------------------------------------------------


def shard_cores(tensor):
    """A sharded tensor's cores in shard order (core i holds shard i), as the buffer assigns them."""
    spec = tensor.memory_config().shard_spec
    return ttnn.corerange_to_cores(spec.grid, None, spec.orientation == ttnn.ShardOrientation.ROW_MAJOR)


def block_grid_shape(shard_spec):
    """(rows, cols) of a BLOCK shard grid. Shard (h, w) sits on shard core h * cols + w: a ROW_MAJOR
    orientation runs the height shards down the grid's y axis, COL_MAJOR down its x axis."""
    size = shard_spec.grid.bounding_box().grid_size()
    if shard_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR:
        return size.y, size.x
    return size.x, size.y


def tap_major_im2col(g: Conv3x3Geometry, b: Conv3x3Blocking):
    return (
        TAP_MAJOR_IM2COL
        and g.stride == 1
        and g.c_in_p % TILE == 0
        # Every channel slice is all data: no lane needs excluding from a whole-pixel copy.
        and g.c_in == g.k_cols * g.c_in_p
        and b.block_k_tiles % g.c_in_tiles == 0
    )


def _slice_channels(g: Conv3x3Geometry, col):
    """Real input channels of channel slice `col` (the last slice may be short)."""
    return min(g.c_in_p, g.c_in - col * g.c_in_p)


def _work_cores(input_tensor, num_cores, grid_x):
    """Core of each work index. Interleaved: row-major over the live grid. Sharded: input shard order, so
    work index i (= output shard i) runs where the output buffer places shard i."""
    if input_tensor.memory_config().is_sharded():
        return shard_cores(input_tensor)[:num_cores]
    return [ttnn.CoreCoord(i % grid_x, i // grid_x) for i in range(num_cores)]


def _core_range_set(cores):
    """Compact CoreRangeSet over `cores`: x-runs per row, then identical runs on consecutive rows merged."""
    rows = {}
    for c in cores:
        rows.setdefault(c.y, []).append(c.x)
    runs = []  # [y0, y1, x0, x1]
    for y in sorted(rows):
        xs = sorted(rows[y])
        start = prev = xs[0]
        row_runs = []
        for x in xs[1:]:
            if x != prev + 1:
                row_runs.append((start, prev))
                start = x
            prev = x
        row_runs.append((start, prev))
        for x0, x1 in row_runs:
            match = next((r for r in runs if r[1] == y - 1 and r[2] == x0 and r[3] == x1), None)
            if match is not None:
                match[1] = y
            else:
                runs.append([y, y, x0, x1])
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for y0, y1, x0, x1 in runs]
    )


def _weight_mcast_cores(core_list, m_blocks_per_core):
    """Cores sharing one weight stream: those that run the most M-blocks, so every participant takes
    the same rounds and its cb_weights write pointer matches the sender's each round. The sender is
    work core 0. None when the channel would be too small or needs more rectangles than a sender has."""
    most = max(m_blocks_per_core)
    cores = [c for c, n in zip(core_list, m_blocks_per_core) if n == most]
    if len(cores) < max(WEIGHT_MCAST_MIN_CORES, 2):
        return None
    receivers = _core_range_set(cores)
    if len(receivers.ranges()) > MAX_MCAST_RECTANGLES:
        return None
    return receivers


def _column_weight_mcast(input_tensor, work, m_blocks_per_index):
    """K split: one weight stream per channel slice, shared by that slice's grid column. Participants are
    the grid rows that run the most M-blocks (same rounds, as in _weight_mcast_cores); each column is one
    group, its first participating row the sender. (receivers, group size, receiver order), or None when a
    column would have too few participants or the participants are not one rectangle."""
    most = max(m_blocks_per_index[i] for _, i, _ in work)
    rows = {i for _, i, _ in work if m_blocks_per_index[i] == most}
    if len(rows) < max(WEIGHT_MCAST_MIN_CORES, 2):
        return None
    receivers = _core_range_set([core for core, i, _ in work if i in rows])
    if len(receivers.ranges()) != 1:
        return None
    # A ROW_MAJOR grid keeps a channel slice on one x (shard (h, w) on core (x0 + w, y0 + h)); COL_MAJOR on one y.
    by_column = input_tensor.memory_config().shard_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
    order = ttnn.McastCoreOrder.ColumnMajor if by_column else ttnn.McastCoreOrder.RowMajor
    return receivers, len(rows), order


def _cb(index, data_format, page_size, num_pages, cores):
    return ttnn.CBDescriptor(
        total_size=page_size * num_pages,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def _work_items(input_tensor, g: Conv3x3Geometry, b: Conv3x3Blocking):
    """(core, m_work_index, channel_slice) per working core. M work index i owns output tile-rows
    [i * core_m_tiles, ...); under a K split all k_cols cores of grid row i share it."""
    if not g.k_split:
        return [(core, i, 0) for i, core in enumerate(_work_cores(input_tensor, b.num_m_groups, b.grid_x))]
    grid_cores = shard_cores(input_tensor)
    return [(grid_cores[row * g.k_cols + col], row, col) for row in range(b.num_m_groups) for col in range(g.k_cols)]


def create_program_descriptor(
    input_tensor,
    weight_tensor,
    bias_tensor,
    output_tensor,
    *,
    geometry: Conv3x3Geometry,
    blocking: Conv3x3Blocking,
    compute_kernel_config,
    apply_relu: bool,
) -> ttnn.ProgramDescriptor:
    g, b = geometry, blocking
    has_bias = bias_tensor is not None
    pixel_bytes = g.c_in_p * BF16_BYTES
    input_page_bytes = g.in0_tile_bytes if g.tile_input else pixel_bytes
    if input_tensor.buffer_aligned_page_size() != input_page_bytes:
        raise ValueError(
            f"conv2d_3x3: input page stride {input_tensor.buffer_aligned_page_size()} B != expected "
            f"{input_page_bytes} B"
        )

    work = _work_items(input_tensor, g, b)
    core_list = [core for core, _, _ in work]
    cores = _core_range_set(core_list)
    bm, bk, bn = b.block_m_tiles, b.block_k_tiles, b.block_n_tiles
    ns = g.out_n_tiles

    if g.tile_input:
        # Untilized footprint tile-rows: one bf16 tile page holds 32 slab pixels of 32 channels.
        slab_cb = _cb(CB_HALO_SLAB, ttnn.bfloat16, TILE_BYTES_BF16, b.slab_tile_rows * g.c_in_tiles, cores)
    else:
        slab_cb = _cb(CB_HALO_SLAB, ttnn.bfloat16, pixel_bytes, b.slab_px, cores)
    face_direct = face_direct_im2col(g, bk)
    cbs = [slab_cb]
    if not face_direct:
        cbs.append(_cb(CB_IM2COL_RM, ttnn.bfloat16, TILE_BYTES_BF16, b.im2col_rm_depth_rows * bk, cores))
    cbs += [
        # Face-direct: the reader writes these tiles; it relies on every K-block starting at the CB base.
        _cb(CB_IM2COL_TILES, input_tensor.dtype, g.in0_tile_bytes, bm * bk, cores),
        _cb(CB_WEIGHTS, weight_tensor.dtype, g.weight_tile_bytes, b.weight_depth * bk * bn, cores),
        _cb(CB_MATMUL_PARTIALS, ttnn.float32, TILE_BYTES_FP32, bm * bn, cores),
    ]
    if g.sharded_output:
        # Packs land in the output shard; its capacity is the core's whole M range, and the M-blocks
        # (bm divides core_m_tiles) push exactly that many tiles, so compute never waits on a consumer.
        output_cb = ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT_TILES, output_tensor, core_ranges=cores)
        if output_cb.total_size != b.core_m_tiles * ns * g.out_tile_bytes:
            raise ValueError(
                f"conv2d_3x3: output shard holds {output_cb.total_size} B, expected "
                f"{b.core_m_tiles} x {ns} tiles of {g.out_tile_bytes} B"
            )
        cbs.append(output_cb)
    else:
        cbs.append(
            _cb(
                CB_OUTPUT_TILES,
                output_tensor.dtype,
                g.out_tile_bytes,
                b.out_depth_row_groups * b.out_subblock_h * bn,
                cores,
            )
        )
    if has_bias:
        cbs.append(_cb(CB_BIAS, bias_tensor.dtype, g.weight_tile_bytes, ns, cores))
    if g.tile_input:
        input_tile_pages = INPUT_TILE_DEPTH_BATCHES * b.untilize_batch_rows * g.c_in_tiles
        cbs.append(_cb(CB_INPUT_TILES, input_tensor.dtype, g.in0_tile_bytes, input_tile_pages, cores))
    if g.k_split:
        cbs.append(_cb(CB_COMBINE_SLOTS, ttnn.float32, TILE_BYTES_FP32, g.k_cols * bm * ns, cores))
        if has_bias:
            cbs.append(_cb(CB_COMBINED, ttnn.float32, TILE_BYTES_FP32, bm * ns, cores))

    # Channel-slice reads (R4): the core's slice of input row v is page v * row_page_stride + col offset,
    # and one owner shard holds input_shard_rows consecutive rows of it contiguously.
    slab_split = SLAB_SPLIT_NOCS and g.sharded_output and not g.tile_input and not g.k_split
    if g.k_split:
        in_spec = input_tensor.memory_config().shard_spec
        page_cols = TILE if g.tile_input else g.c_in_p
        row_page_stride = _div_up(int(input_tensor.padded_shape[-1]), page_cols)
        input_shard_rows = int(in_spec.shape[0]) // (TILE if g.tile_input else 1)
    elif slab_split:
        row_page_stride, input_shard_rows = 0, int(input_tensor.memory_config().shard_spec.shape[0])
    else:
        row_page_stride, input_shard_rows = 0, 0
    pages_per_slice_row = g.c_in_tiles if g.tile_input else 1
    combine_split_nocs = g.k_split and COMBINE_SPLIT_NOCS
    tap_major = tap_major_im2col(g, b)
    # Width of one tilize block of im2col rows: one tap (tap-major) or the whole K-block.
    tilize_width = g.c_in_tiles if tap_major else bk

    stream_rows = b.in0_stream_rows
    gather_split = GATHER_SPLIT_RISCS and face_direct and slab_split and _div_up(bm, stream_rows) >= 2
    # Semaphore ids: the R4 combine's four, then the split slab fill's two and the split gather's two; the weight
    # channel attaches after.
    sem_ids = (SEM_COMBINE_CREDIT, SEM_COMBINE_ARRIVAL, SEM_COMBINE_SEND_GO, SEM_COMBINE_SEND_DONE) if g.k_split else ()
    slab_sem_ids = (len(sem_ids), len(sem_ids) + 1) if slab_split else (0, 0)
    sem_ids += slab_sem_ids if slab_split else ()
    gather_sem_ids = (len(sem_ids), len(sem_ids) + 1) if gather_split else (0, 0)
    sem_ids += gather_sem_ids if gather_split else ()

    def reader_ct_args(slice_channels):
        args = [
            CB_HALO_SLAB,
            CB_IM2COL_RM,
            CB_INPUT_TILES,
            g.input_height,
            g.input_width,
            g.output_width,
            g.num_out_px,
            slice_channels,
            g.c_in_p,
            g.padding,
            g.stride,
            g.dilation,
            bm,
            bk,
            g.tensor_kkh_tiles,
            b.num_k_blocks,
            b.band_stride_px,
            int(g.tile_input),
            input_page_bytes,
            b.slab_tile_rows,
            b.untilize_batch_rows,
            b.im2col_rm_depth_rows,
            int(g.k_split),
            input_shard_rows,
            row_page_stride,
            int(combine_split_nocs),
            g.k_cols,
            bn,
            ns,
            g.tensor_n_tiles,
            CB_MATMUL_PARTIALS,
            CB_COMBINE_SLOTS,
            SEM_COMBINE_SEND_GO,
            SEM_COMBINE_SEND_DONE,
            int(tap_major),
            int(face_direct),
            g.band_last_tap,
            CB_IM2COL_TILES,
            int(slab_split),
            *slab_sem_ids,
            int(gather_split),
            stream_rows,
            *gather_sem_ids,
        ]
        args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
        return args

    compute_ct_args = [
        CB_IM2COL_RM,
        CB_IM2COL_TILES,
        CB_WEIGHTS,
        CB_BIAS,
        CB_MATMUL_PARTIALS,
        CB_OUTPUT_TILES,
        bm,
        bk,
        bn,
        b.num_k_blocks,
        b.out_subblock_h,
        b.out_subblock_w,
        int(has_bias),
        int(apply_relu),
        CB_INPUT_TILES,
        CB_HALO_SLAB,
        int(g.tile_input),
        g.c_in_tiles,
        b.slab_tile_rows,
        int(g.k_split),
        g.k_cols,
        ns,
        CB_COMBINE_SLOTS,
        CB_COMBINED,
        b.combine_subblock_h,
        b.combine_subblock_w,
        tilize_width,
        int(face_direct),
        b.in0_stream_rows,
        max(1, min(BIAS_STREAM_GROUPS, _div_up(bm, b.in0_stream_rows))),
    ]

    writer_ct_args = [
        CB_WEIGHTS,
        CB_BIAS,
        CB_OUTPUT_TILES,
        bm,
        bk,
        bn,
        b.num_k_blocks,
        b.out_subblock_h,
        g.tensor_n_tiles,
        int(has_bias),
        int(g.sharded_output),
        ns,
        int(g.k_split),
        g.k_cols,
        CB_MATMUL_PARTIALS,
        CB_COMBINE_SLOTS,
        SEM_COMBINE_CREDIT,
        SEM_COMBINE_ARRIVAL,
        int(combine_split_nocs),
        SEM_COMBINE_SEND_GO,
        SEM_COMBINE_SEND_DONE,
        int(slab_split),
        CB_HALO_SLAB,
        g.input_height,
        g.input_width,
        g.output_width,
        g.num_out_px,
        g.padding,
        g.stride,
        g.dilation,
        b.band_stride_px,
        g.band_last_tap,
        input_page_bytes,
        input_shard_rows,
        *slab_sem_ids,
        int(gather_split),
        stream_rows,
        *gather_sem_ids,
        CB_IM2COL_TILES,
        g.c_in_p,
        g.tensor_kkh_tiles,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(weight_tensor).get_compile_time_args())
    writer_ct_args.extend(
        ttnn.TensorAccessorArgs(bias_tensor).get_compile_time_args()
        if has_bias
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    writer_ct_args.extend(
        ttnn.TensorAccessorArgs().get_compile_time_args()
        if g.sharded_output
        else ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
    )
    writer_ct_args.extend(
        ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
        if slab_split
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    # One reader kernel per distinct slice width: the gather's lane handling is compile-time.
    reader_groups = {}  # slice channels -> (cores, RuntimeArgs)
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    input_addr = input_tensor.buffer_address()
    weight_addr = weight_tensor.buffer_address()
    bias_addr = bias_tensor.buffer_address() if has_bias else 0
    output_addr = output_tensor.buffer_address()
    device = input_tensor.device()
    core_m_ranges = _core_m_ranges(g.tensor_m_tiles, b.core_m_tiles)
    m_blocks_per_index = [_div_up(actual_m_tiles, bm) for _, actual_m_tiles in core_m_ranges]
    row_peer_coords = {}
    if g.k_split:
        for core, row, _ in work:
            virtual = device.worker_core_from_logical_core(core)
            row_peer_coords.setdefault(row, []).extend([virtual.x, virtual.y])
    for core, m_index, col in work:
        m_tile_start, actual_m_tiles = core_m_ranges[m_index]
        num_m_blocks_this_core = m_blocks_per_index[m_index]
        # The core's output columns: all of C_out, or N-slice `col` under the K split (possibly short or
        # empty past the last C_out tile).
        out_n_tile_start = col * ns
        out_valid_tiles = max(0, min(ns, g.tensor_n_tiles - out_n_tile_start))
        reader_cores, reader_rt = reader_groups.setdefault(_slice_channels(g, col), ([], ttnn.RuntimeArgs()))
        reader_cores.append(core)
        reader_rt[core.x][core.y] = [
            input_addr,
            m_tile_start,
            actual_m_tiles,
            num_m_blocks_this_core,
            m_index * SLAB_READ_STAGGER_PX_PER_CORE,
            col * pages_per_slice_row,
            col,
            m_index,
        ] + row_peer_coords.get(m_index, [])
        compute_rt[core.x][core.y] = [num_m_blocks_this_core, int(out_valid_tiles > 0)]
        writer_rt[core.x][core.y] = [
            weight_addr,
            bias_addr,
            output_addr,
            m_tile_start,
            actual_m_tiles,
            num_m_blocks_this_core,
            0,  # first N tile of the matmul block: the core multiplies against all of C_out
            col * g.tensor_k_tiles,  # the column's weight K-slice (rows ordered (col, kh, kw, c))
            out_n_tile_start,
            out_valid_tiles,
            col,
            input_addr,
            m_index,
        ] + row_peer_coords.get(m_index, [])

    readers = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "conv2d_3x3_reader.cpp"),
            core_ranges=_core_range_set(reader_cores),
            compile_time_args=reader_ct_args(slice_channels),
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        for slice_channels, (reader_cores, reader_rt) in reader_groups.items()
    ]
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "conv2d_3x3_writer.cpp"),
        core_ranges=cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "conv2d_3x3_compute.cpp"),
        core_ranges=cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=compute_kernel_config.math_fidelity,
            math_approx_mode=compute_kernel_config.math_approx_mode,
            fp32_dest_acc_en=True,
        ),
    )
    semaphores = [ttnn.SemaphoreDescriptor(id=sem_id, core_ranges=cores, initial_value=0) for sem_id in sem_ids]
    descriptor = ttnn.ProgramDescriptor(kernels=[], semaphores=semaphores, cbs=cbs)
    # The writer's NoC (WriterConfigDescriptor).
    weight_mcast_config = ttnn.McastConfig(noc=ttnn.NOC.NOC_1)
    weight_mcast = None
    if g.k_split:
        # Each column multiplies against its own weight K-slice: one stream per column.
        column_channel = _column_weight_mcast(input_tensor, work, m_blocks_per_index)
        if column_channel is not None:
            receivers, group_size, order = column_channel
            weight_mcast = ttnn.Mcast(
                device, weight_mcast_config, receivers, group_size, ttnn.McastFixedSenderConfig(), order
            )
    else:
        mcast_receivers = _weight_mcast_cores(core_list, [m_blocks_per_index[i] for _, i, _ in work])
        if mcast_receivers is not None:
            weight_mcast = ttnn.Mcast(
                device,
                weight_mcast_config,
                mcast_receivers,
                mcast_receivers.num_cores(),
                ttnn.McastExplicitFixedSenderConfig([core_list[0]]),
            )
    if weight_mcast is None:
        ttnn.attach_absent_mcast(writer, WEIGHT_MCAST_PREFIX)
    else:
        weight_mcast.attach(descriptor, WEIGHT_MCAST_PREFIX, [writer], len(semaphores))
    descriptor.kernels = readers + [writer, compute]
    return descriptor
