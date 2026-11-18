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
import math
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
# R4 only: every row peer's fp32 N-slice for this core's output columns, and (K-blocked walk) their sum before the
# bias. On the N-slice-outer ring, slot k_cols - 1 - j holds step j's partial and slot 0 the row's sum.
CB_COMBINE_SLOTS = 8
CB_COMBINED = 9
# N-slice-outer ring: one 16 B page per packed row group, compute (packer) -> reader (see conv2d_3x3_nslice_order.hpp).
CB_RING_TOKEN = 10
RING_TOKEN_BYTES = 16
# Tap-major only: 32 zero pixels, so an out-of-image tap run is one read instead of one per 512 B MEM_ZEROS chunk.
CB_ZERO_SRC = 11
# Center-tap direct in0 (see CENTER_TAP_DIRECT): the kw = 1 K-blocks' input tiles, copied by the reader.
CB_CENTER_TILES = 12
# (N-slice center taps reuse it: [3, bm, c_in_tiles], see NSLICE_CENTER_TAPS.)
# Tap-direct N-slice prelude (see _tap_direct): one 16 B entry of tap source addresses per cb_im2col_rm unit slot.
CB_SRC_TABLE = 13

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

# Strided-window im2col: at stride 2 (unit dilation, whole-tile pixels) im2col row i of a tile-row is the 3 consecutive
# input pixels from 2 * wo_i - padding, so the rows of one output-row run are overlapping windows of one input run, two
# pixels apart. The reader lays a kernel row of a tile-row out as that run (one read per run instead of one per output
# pixel) and compute tilizes it in place at a 2-pixel row stride; on the N-slice-outer walk compute tilizes in-image runs
# straight from the slab. False keeps the per-pixel row gather.
STRIDED_WINDOW_IM2COL = True

# Center-tap direct in0 (sharded TILE input, stride 1, padding == dilation, W % 32 == 0): the kw = 1 tap of kernel row
# kh reads input pixel o + (kh*d - p)*W for output pixel o, a whole-tile-row shift, so its in0 tiles ARE input tile-rows
# (same format as cb_im2col_tiles) or zero tile-rows above / below the image. The reader copies them from the own or a
# neighbour shard and compute skips their untilize -> gather -> tilize round trip (3 of the 9 taps):
#   one tap per K-block: the kw = 1 K-blocks come from cb_center_tiles (matmul_block's In0SourceFn);
#   one kernel row per K-block (CENTER_TAP_SLOTS, bm >= 2): the reader writes each tile-row's kw = 1 tiles straight into
#     their cb_im2col_tiles slots, and compute's tilize reserves and pushes those slots unpacked. cb_im2col_tiles holds
#     two K-blocks, so the reader writes K-block k's center tiles while compute still multiplies K-block k - 1
#     (reserving K-block k's first im2col row already implies the matmul of K-block k - 2 is done).
# Both take bm * bk more in0 tiles, and only from the L1 left at the solved blocking, so they never change bm, bk, the
# untilize batch or the im2col ring. False keeps the tilize for every K-block.
CENTER_TAP_DIRECT = True
CENTER_TAP_SLOTS = True

# N-slice center taps (the K split's N-slice-outer walk with the tap-direct prelude, TILE input, stride 1,
# padding == dilation, W % 32 == 0, one kernel row per unit): the kw = 1 tap of kernel row kh for output tile-row t is
# input tile-row t + (kh*d - p) * W / 32 of the core's channel slice (or a zero tile-row outside the image), so compute
# tilizes only taps 0 and 2 (in0 is [bm, 6 * c_in_tiles]) and the ring walk reads the center taps from cb_center_tiles
# ([3, bm, c_in_tiles], kernel-row major). The reader copies them, right after its fill, from the whole-slab input ring
# (cb_input_tiles in cb_im2col_tiles' L1 holds every in-image center tile-row, a short slice's padding tiles zeroed): a
# local copy, landed before compute's tilize overwrites the ring. L1-neutral (in0 shrinks by what the center CB takes).
# Measured on the tap-direct prelude, where the tilize paced it (LoFi, median of 4): aspp.convs.3 37435 -> 36328 ns,
# aspp.convs.1 67310 -> 66083 ns; res5.x flat; res4.x 16530 -> 16697 ns (inside the noise band: its prelude is paced by
# the reader's per-group gather work, which this does not shorten). Only where the input ring holds the whole slab (one
# M-block per core); elsewhere every tap is tilized. False tilizes every tap.
NSLICE_CENTER_TAPS = True

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
# HEIGHT-sharded TILE input: the fill is split by NoC like the ROW_MAJOR one. The reader (NoC0) reads the tile-rows of
# its own and lower input shards into each cb_input_tiles batch, the writer (NoC1) those of higher shards into the batch
# the reader reserved and handed over; the writer serves hand-overs wherever it would otherwise wait (weight multicast
# receive, weight reserve, bias read, row-gather serving). Only on cores whose writer is a pure weight-multicast
# receiver: the sender and the own-stream cores spend their writer in blocking weight DRAM reads, which delays their
# fill and with it every round (measured slower), so they keep the whole fill on the reader.
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

# N-slice-outer K split (R4, the default when its resident in0 fits L1): the core tilizes its whole K of in0 once,
# then multiplies one owner N-slice at a time over the whole K (one DEST accumulation) and the row reduce-scatters the
# fp32 partials along a ring (conv2d_3x3_nslice_order.hpp): each step's slice is packed with L1 accumulation onto the
# partial the previous column forwarded, one NoC hop per step, so the owner's sum is complete when its own slice lands
# (no combine sum, no all-to-all). Weight CB depth in slices (two rounds ahead, plus the sender's prefetch).
NSLICE_WEIGHT_DEPTH = 4
# Shared-L1 N-slice walk, whenever every core runs one M-block: cb_combine_slots shares cb_halo_slab's L1, dead once
# the core's gather landed; the credit to the previous column follows it.

# Split row gather (the per-pixel row gather with more than one tile-row per K-block): the writer gathers the units the
# reader hands it (at most ROW_QUEUE outstanding) on NoC1, so two RISCs issue the reads. cb_im2col_rm then holds up to
# ROW_SPLIT_DEPTH_ROWS units, so both RISCs and the next K-block's units stay in flight during a matmul.
ROW_QUEUE = 2
ROW_SPLIT_DEPTH_ROWS = 8

# Buffer-depth knobs (Blocking Model depth table). On the K-blocked walk with the non-split gather, cb_im2col_rm then
# grows past IM2COL_RM_DEPTH_ROWS toward one K-block of tile-rows, as far as the L1 left at the solved blocking allows
# (_im2col_ring_target).
IM2COL_RM_DEPTH_ROWS = 2
# Tap-direct N-slice prelude: units (tile-row x K-block) the reader reserves, fences and pushes at once; the ring aims for
# two groups from the L1 left after the solve (_im2col_ring_target). The reader's per-unit work, not its reads, paced
# the prelude tilize, so one trid wait and one push per group instead of per unit.
TAP_DIRECT_GROUP = 3
WEIGHT_DEPTH = 2
OUT_DEPTH_ROW_GROUPS = 2
# cb_input_tiles depth in untilize batches: the reader reads batch b+1 while compute untilizes b. Where every core runs
# one M-block the ring instead holds the whole slab in cb_im2col_tiles' L1 (see create_program_descriptor).
INPUT_TILE_DEPTH_BATCHES = 2

# Windowed TILE fill (compacted slab): cb_input_tiles batches the reader keeps in flight, each under its own
# transaction id, and the split fill's writer likewise. One batch at a time was one read round trip per batch: on
# aspp.convs.1 @ LoFi (2-tile-row batches) the fill ran at ~22 GB/s and the untilize waited on it.
FILL_WINDOW_BATCHES = 4
FILL_WRITER_WINDOW_BATCHES = 4
# The split TILE fill under a K split runs on a column receiver only when one of its M-blocks reads at least this many
# bytes from higher-index shards. The split pays by taking those reads off NoC0's wrap-around; a small hand-over is
# pure latency (res4 layers, ~3 KB per core: +2.0% / +4.3% median of 3 with the split; res5.0, ~11 KB: flat;
# res5.1 / res5.2, ~20 KB, and aspp, 36-72 KB: faster).
SPLIT_FILL_MIN_BYTES = 8192

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
    # The K split runs N-slice-outer (see NSLICE_WEIGHT_DEPTH).
    nslice: bool = False
    # Shared-L1 N-slice walk (see NSLICE_WEIGHT_DEPTH): byte size of the slab + combine-slots region (0: not shared).
    shared_slab_bytes: int = 0
    # Center-tap direct mode in effect (CENTER_NONE / CENTER_K_BLOCK / CENTER_SLOTS), and the K-blocks cb_im2col_tiles
    # holds (2 under CENTER_SLOTS on the K-blocked walk).
    center_mode: int = 0
    center_slot_blocks: int = 1
    # Ring slab (see ring_capable): slab_tile_rows is then the ring capacity.
    ring: bool = False

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


def _ring_subblock(block_m_tiles, out_n_tiles):
    """N-slice ring walk subblock (rows, cols), also the ring's row group. Four tiles long along the slice's longer side:
    the BH matmul unpacker reads rt + rt * ct tiles per k-step (ct + ct * rt when rt > ct), so 4 x 1 or 1 x 4 costs
    1.25 unpacks per tile-MAC against 1.5 for 1 x 2 or 2 x 2. A tall subblock may leave a shorter last row group (bm 7:
    4 + 3); a wide one must divide out_n, so it falls back to _out_subblock."""
    tall = min(block_m_tiles, DEST_TILES_FP32_HALF_SYNC)
    if tall >= min(out_n_tiles, DEST_TILES_FP32_HALF_SYNC):
        return tall, 1
    return _out_subblock(block_m_tiles, out_n_tiles)


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


def _higher_shard_rows(g: Conv3x3Geometry, core_m_tiles, block_m_tiles, m_index, shard_rows):
    """The most slab tile-rows one M-block of work index m_index reads from a higher-index input shard (the split
    fill's writer share; mirror of the kernels' tile_slab_plan segments)."""
    start, actual = _core_m_ranges(g.tensor_m_tiles, core_m_tiles)[m_index]
    most = 0
    for block_start in range(0, actual, block_m_tiles):
        tiles = min(block_m_tiles, actual - block_start)
        o_first = (start + block_start) * TILE
        o_last = min((start + block_start + tiles) * TILE, g.num_out_px) - 1
        band_len = _block_band_len(g, o_first, o_last)
        f = _block_first_tap(g, o_first)
        prev_hi = None
        count = 0
        for kh in range(3):
            v_lo = f + kh * g.dilation * g.input_width
            lo, hi = max(v_lo, 0), min(v_lo + band_len, g.num_in_px)
            if lo >= hi:
                continue
            t_lo, t_hi = lo // TILE, _div_up(hi, TILE)
            first_new = prev_hi if prev_hi is not None and t_lo < prev_hi else t_lo
            count += sum(1 for r in range(first_new, t_hi) if r // shard_rows > m_index)
            prev_hi = t_hi
        most = max(most, count)
    return most


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


def short_slice_lanes_zeroed(g: Conv3x3Geometry):
    """A short last channel slice (K split, C_in not a multiple of the slice width) whose padded lanes the reader
    keeps at zero in the slab: zeroed once, never read into. TILE needs the slice's real channels to be whole tiles;
    lanes inside a real tile are the tensor's own tile padding, which no whole-tile zeroing reaches."""
    if not g.k_split:
        return False
    short = g.c_in - (g.k_cols - 1) * g.c_in_p
    return 0 < short < g.c_in_p and (not g.tile_input or short % TILE == 0)


def tap_major_capable(g: Conv3x3Geometry):
    """Tap-major im2col apart from the K-block: stride 1, whole-tile taps, and every slab lane a whole-pixel copy
    may carry is data or zero."""
    return (
        TAP_MAJOR_IM2COL
        and g.stride == 1
        and g.c_in_p % TILE == 0
        and (g.c_in == g.k_cols * g.c_in_p or short_slice_lanes_zeroed(g))
    )


def _tap_major(g: Conv3x3Geometry, bk):
    return tap_major_capable(g) and bk % g.c_in_tiles == 0


def strided_window_capable(g: Conv3x3Geometry):
    """Strided-window im2col apart from the K-block: stride 2, unit dilation, whole-tile pixels whose every lane is data
    or zero (a whole-pixel copy carries them). Two output-row runs of one tile-row (W_out not a multiple of 32) share
    one pixel slot, which must be zero for both: the next row's left padding and the previous row's last right tap."""
    if not STRIDED_WINDOW_IM2COL or g.stride != 2 or g.dilation != 1 or g.c_in_p % TILE != 0:
        return False
    if not (g.c_in == g.k_cols * g.c_in_p or short_slice_lanes_zeroed(g)):
        return False
    if g.output_width % TILE != 0:
        return g.padding >= 1 and 2 * (g.output_width - 1) + 2 - g.padding >= g.input_width
    return True


def _strided_window(g: Conv3x3Geometry, bk):
    """One K-block per kernel row: a unit (tile-row x K-block) is one kernel row's window run."""
    return strided_window_capable(g) and bk == g.tensor_kkh_tiles and not face_direct_im2col(g, bk)


CENTER_NONE, CENTER_K_BLOCK, CENTER_SLOTS = 0, 1, 2


def _tap_direct(g: Conv3x3Geometry, bm, bk, nslice):
    """Tap-direct N-slice prelude: compute tilizes each tap block of a unit from an L1 address the reader publishes (a
    slab run when the tile-row's pixels are one output-row run inside the image, the zero block for a kernel row
    outside it, else the unit's slot, which the reader gathers as before). Saves the reader's copy of most tap blocks,
    which is what paces the N-slice prelude. Needs the tap-major gather."""
    return (
        nslice
        and _tap_major(g, bk)
        and not face_direct_im2col(g, bk)
        and not _row_split(g, bm, bk)
        and not _strided_window(g, bk)
    )


def _src_table_bytes(ring_units):
    return _div_up(ring_units * 16, 64) * 64


def nslice_center_capable(g: Conv3x3Geometry, b):
    """N-slice center taps apart from the input ring (see NSLICE_CENTER_TAPS): the kw = 1 tap is a whole-tile-row
    vertical shift of the core's channel slice, on the tap-direct N-slice prelude with one kernel row per unit."""
    return (
        NSLICE_CENTER_TAPS
        and g.k_split
        and b.nslice
        and g.tile_input
        and g.stride == 1
        and g.padding == g.dilation
        and g.input_width % TILE == 0
        and b.block_k_tiles == 3 * g.c_in_tiles
        and _tap_direct(g, b.block_m_tiles, b.block_k_tiles, b.nslice)
    )


def center_tap_mode(g: Conv3x3Geometry, bm, bk):
    """How the kw = 1 taps skip the tilize (see CENTER_TAP_DIRECT): they are whole input tile-rows (a vertical-only
    shift by whole tile-rows).
      CENTER_K_BLOCK  one tap per K-block: K-block k with k % 3 == 1 comes from cb_center_tiles.
      CENTER_SLOTS    one kernel row per K-block (bm >= 2): the reader writes each tile-row's center tiles into their
                      slots of a second in0 K-block, and they land before the reader pushes the K-block's last im2col row.
    Not under a K split. Its K-blocked walk's L1-accumulating packs are ordered only by the tilize between K-blocks. On
    the N-slice-outer walk, copying from the owner shards into in0 slots measured slower (round 3: res5.1 +1.5%,
    aspp.convs.1 +2.3%, LoFi median of 3); that walk copies its center taps from the input ring into their own CB
    instead (NSLICE_CENTER_TAPS)."""
    if g.k_split:
        return CENTER_NONE
    # Sharded input only: the center tiles are then L1 copies from the own or a neighbour shard. On an interleaved input
    # they are a second DRAM read of tile-rows the fill already streams (measured 7-16% slower).
    if not (
        CENTER_TAP_DIRECT
        and g.tile_input
        and g.sharded_output
        and g.stride == 1
        and g.padding == g.dilation
        and g.input_width % TILE == 0
        and _tap_major(g, bk)
    ):
        return CENTER_NONE
    if bk == g.c_in_tiles:
        return CENTER_K_BLOCK
    if CENTER_TAP_SLOTS and bk == 3 * g.c_in_tiles and bm >= 2:
        return CENTER_SLOTS
    return CENTER_NONE


def _row_split(g: Conv3x3Geometry, bm, bk):
    """The split row gather: the per-pixel row gather (neither face-direct, tap-major nor strided-window) with more than
    one tile-row per K-block; with one there is nothing to hand over (the reader takes each K-block's first unit)."""
    return bm > 1 and not face_direct_im2col(g, bk) and not _tap_major(g, bk) and not _strided_window(g, bk)


def _im2col_ring_target(g: Conv3x3Geometry, bm, bk, nslice):
    """Tile-rows cb_im2col_rm aims for: one K-block on the K-blocked walk with the non-split gather, so the reader
    gathers K-block k+1 while compute multiplies K-block k and the tilize never waits on a reserve-blocked gather. The
    split row gather has its own ring, face-direct has no cb_im2col_rm. The N-slice-outer walk tilizes all of K before
    its first matmul, so there is nothing for the gather to run under, except on tap-direct: two TAP_DIRECT_GROUP
    groups, so compute tilizes one group while the reader prepares the next."""
    if nslice and _tap_direct(g, bm, bk, nslice):
        return max(IM2COL_RM_DEPTH_ROWS, 2 * TAP_DIRECT_GROUP)
    if nslice or face_direct_im2col(g, bk) or _row_split(g, bm, bk):
        return IM2COL_RM_DEPTH_ROWS
    return max(IM2COL_RM_DEPTH_ROWS, bm)


def _lcm(a, b):
    return a * b // math.gcd(a, b)


def _shared_region_bytes(a_bytes, a_page, b_bytes, b_page):
    """One L1 region holding either of two CBs: whole pages of both."""
    unit = _lcm(a_page, b_page)
    return _div_up(max(a_bytes, b_bytes), unit) * unit


def _footprint(
    g: Conv3x3Geometry, core_m_tiles, bm, bk, bn, ub_cap=None, nslice=False, im2col_depth=None, shared=False
):
    num_k_blocks = g.tensor_k_tiles // bk
    im2col_rm_depth_rows = (
        max(IM2COL_RM_DEPTH_ROWS, min(ROW_SPLIT_DEPTH_ROWS, bm * num_k_blocks))
        if _row_split(g, bm, bk)
        else (IM2COL_RM_DEPTH_ROWS if im2col_depth is None else im2col_depth)
    )
    max_m_blocks = _div_up(core_m_tiles, bm)
    num_slices = _div_up(g.tensor_n_tiles, g.out_n_tiles)
    weight_rounds = (num_slices if nslice else num_k_blocks) * max_m_blocks
    weight_depth = 1 if weight_rounds == 1 else (NSLICE_WEIGHT_DEPTH if nslice else WEIGHT_DEPTH)
    band_len, raw_slab_rows = _block_extents(g, core_m_tiles, bm)
    band_stride = min(g.dilation * g.input_width, band_len)
    sb_h, sb_w = _out_subblock(bm, bn)
    ns = g.out_n_tiles
    combine_sb_h, combine_sb_w = _out_subblock(bm, ns)
    # R4: every row peer's slice of this core's output columns, plus (K-blocked walk) their sum before the bias add.
    # The N-slice ring leaves the sum in slot 0 and adds the bias from there.
    combine_bytes = (
        (g.k_cols * bm * ns + (bm * ns if g.has_bias and not nslice else 0)) * TILE_BYTES_FP32 if g.k_split else 0
    )
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
    zero_src_bytes = TILE * g.c_in_p * BF16_BYTES if _tap_major(g, bk) else 0
    shared_slab_bytes = 0
    if shared:
        slots_bytes = g.k_cols * bm * ns * TILE_BYTES_FP32
        slab_page = TILE_BYTES_BF16 if g.tile_input else g.c_in_p * BF16_BYTES
        shared_slab_bytes = _shared_region_bytes(slab_bytes, slab_page, slots_bytes, TILE_BYTES_FP32)
        combine_bytes -= slots_bytes
        slab_bytes = shared_slab_bytes
    total = (
        slab_bytes
        + input_tile_bytes
        + (0 if face_direct_im2col(g, bk) else im2col_rm_depth_rows * bk * TILE_BYTES_BF16)
        + zero_src_bytes
        + bm * (g.tensor_k_tiles if nslice else bk) * g.in0_tile_bytes
        + weight_depth * (g.tensor_k_tiles * ns if nslice else bk * bn) * g.weight_tile_bytes
        + (ns * g.weight_tile_bytes if g.has_bias else 0)
        + (RING_TOKEN_BYTES + TILE_BYTES_FP32 if nslice else bm * bn * TILE_BYTES_FP32)
        + combine_bytes
        + (0 if g.sharded_output else OUT_DEPTH_ROW_GROUPS * sb_h * bn * g.out_tile_bytes)
        + (_src_table_bytes(im2col_rm_depth_rows) if _tap_direct(g, bm, bk, nslice) else 0)
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
        nslice=nslice,
        im2col_rm_depth_rows=im2col_rm_depth_rows,
        shared_slab_bytes=shared_slab_bytes,
    )


def _ring_window(g: Conv3x3Geometry, o_first, o_last):
    """In-image input tile-rows [lo, hi) the three bands of output pixels [o_first, o_last] touch (mirror of the
    reader's ring_window), or None when every band is outside the image."""
    band_len = _block_band_len(g, o_first, o_last)
    f = _block_first_tap(g, o_first)
    lo_hi = None
    for kh in range(3):
        v_lo = f + kh * g.dilation * g.input_width
        lo, hi = max(v_lo, 0), min(v_lo + band_len, g.num_in_px)
        if lo >= hi:
            continue
        t_lo, t_hi = lo // TILE, _div_up(hi, TILE)
        lo_hi = (t_lo, t_hi) if lo_hi is None else (min(lo_hi[0], t_lo), max(lo_hi[1], t_hi))
    return lo_hi


@functools.lru_cache(maxsize=1024)
def ring_plan(g: Conv3x3Geometry, core_m_tiles, block_m_tiles):
    """Per core, per M-block: (lo, hi, new_rows) of the ring slab (mirror of the reader / writer). lo is the lowest
    window row of this and every later block of the core (the rows the ring must still hold: a later block's window may
    start lower, e.g. at the image top once band 0 enters the image), hi the highest window end so far (a block at the
    image bottom may end lower), and new_rows = hi - max(previous hi, lo) the rows the block untilizes. None when a
    block touches no in-image row."""
    plans = []
    for start, actual in _core_m_ranges(g.tensor_m_tiles, core_m_tiles):
        windows = []
        for block_start in range(0, actual, block_m_tiles):
            tiles = min(block_m_tiles, actual - block_start)
            o_first = (start + block_start) * TILE
            o_last = min((start + block_start + tiles) * TILE, g.num_out_px) - 1
            w = _ring_window(g, o_first, o_last)
            if w is None:
                return None
            windows.append(w)
        blocks = []
        prev_hi = None
        for m, (_, hi) in enumerate(windows):
            lo = min(w[0] for w in windows[m:])
            hi = hi if prev_hi is None else max(prev_hi, hi)
            push_lo = lo if prev_hi is None else max(prev_hi, lo)
            blocks.append((lo, hi, hi - push_lo))
            prev_hi = hi
        plans.append(tuple(blocks))
    return tuple(plans)


def ring_capable(g: Conv3x3Geometry, bm, bk):
    """Ring slab: cb_halo_slab is a FIFO ring of consecutive input tile-rows across M-blocks. M-block m+1 untilizes only
    the rows past M-block m's window, the reader pops only the rows below M-block m+1's window, and the gather wraps at
    the ring end. Capacity = the largest M-block window (in-image tile-rows from its first band's first row to its last
    band's last row), taken only from the L1 left at the solved blocking; otherwise one compacted slab per M-block.
    Needs TILE input, the tap-major gather (the only gather with ring-wrapping reads here) and no K split."""
    return (
        g.tile_input
        and not g.k_split
        and _tap_major(g, bk)
        and not face_direct_im2col(g, bk)
        # The narrow (1-tile) untilize packs whole sections at one address: they would cross the ring end.
        and g.c_in_tiles > 1
    )


def ring_rows(g: Conv3x3Geometry, core_m_tiles, bm):
    """Ring capacity: the largest M-block window."""
    return max(hi - lo for core in ring_plan(g, core_m_tiles, bm) for lo, hi, _ in core)


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
    if tap_major_capable(g):
        # Tap-major needs whole taps per K-block. Above the tap floor the only tap-splitting divisor of the kernel
        # row (3 taps) is 1.5 taps; the row gather it would fall back to costs far more than the extra K-blocks.
        bk_above_floor = [d for d in bk_above_floor if d % g.c_in_tiles == 0]
    if g.sharded_output:
        bm_candidates = [d for d in range(core_m_tiles, 0, -1) if core_m_tiles % d == 0]
    else:
        bm_candidates = sorted({_div_up(core_m_tiles, n) for n in range(1, core_m_tiles + 1)}, reverse=True)

    # A smaller bm frees L1, so each bm retries the coarsest bk again: at a fixed bm, bk costs no bytes.
    search = [(bm, bk) for bm in bm_candidates for bk in bk_above_floor]
    search += [(bm_candidates[-1], bk) for bk in bk_below_floor]

    def solve(candidates, nslice, shared=False):
        for bm, bk in candidates:
            # The untilize batch shrinks first: a smaller batch costs one more read fence per tile-row,
            # a smaller bk costs a whole K-block (tilize, matmul init, L1-acc pass).
            band_len, raw_slab_rows = _block_extents(g, core_m_tiles, bm)
            ub_default = _default_untilize_batch_rows(g, band_len, raw_slab_rows)
            for ub_cap in range(ub_default, 0, -1):
                fit = _footprint(g, core_m_tiles, bm, bk, bn, ub_cap, nslice, shared=shared)
                if fit["footprint_bytes"] <= l1_budget:
                    # The deeper im2col ring only takes L1 left at this (bm, bk, ub): it never costs a coarser block.
                    for depth in range(_im2col_ring_target(g, bm, bk, nslice), IM2COL_RM_DEPTH_ROWS, -1):
                        deep = _footprint(
                            g, core_m_tiles, bm, bk, bn, ub_cap, nslice, im2col_depth=depth, shared=shared
                        )
                        if deep["footprint_bytes"] <= l1_budget:
                            fit = deep
                            break
                    # Center-tap direct: bm * bk more in0 tiles, from the L1 left after everything else.
                    mode = center_tap_mode(g, bm, bk)
                    extra = bm * bk * g.in0_tile_bytes
                    if mode != CENTER_NONE and fit["footprint_bytes"] + extra <= l1_budget:
                        fit = dict(
                            fit,
                            center_mode=mode,
                            center_slot_blocks=2 if mode == CENTER_SLOTS else 1,
                            footprint_bytes=fit["footprint_bytes"] + extra,
                        )
                    # Ring slab: the ring replaces the compacted slab, only from the L1 left after everything else
                    # (counting it in the solve shrinks blocks for it: measured +34% on a DRAM C_in 512 shape).
                    if not nslice and ring_capable(g, bm, bk) and ring_plan(g, core_m_tiles, bm) is not None:
                        ring = ring_rows(g, core_m_tiles, bm)
                        delta = (ring - fit["slab_tile_rows"]) * g.c_in_tiles * TILE_BYTES_BF16
                        if fit["footprint_bytes"] + delta <= l1_budget:
                            fit = dict(
                                fit, ring=True, slab_tile_rows=ring, footprint_bytes=fit["footprint_bytes"] + delta
                            )
                    return Conv3x3Blocking(
                        grid_x=grid_x,
                        core_m_tiles=core_m_tiles,
                        num_m_groups=num_m_groups,
                        block_m_tiles=bm,
                        block_k_tiles=bk,
                        block_n_tiles=bn,
                        out_depth_row_groups=OUT_DEPTH_ROW_GROUPS,
                        **fit,
                    )
        return None

    blocking = solve(search, nslice=False)
    if blocking is not None and g.k_split:
        # N-slice-outer at the K-blocked solve's bm, unless its resident [bm, K] in0 does not fit. There bk only sets
        # the gather / tilize granularity and costs almost no L1, so the coarsest that fits wins.
        # The slab and the combine slots share L1 wherever every core runs one M-block. With more, the slab is refilled
        # for the next block while the row's peers may still write this block's slots.
        shared = blocking.max_m_blocks_per_core == 1
        nslice = solve(
            [(blocking.block_m_tiles, bk) for bk in bk_above_floor + bk_below_floor], nslice=True, shared=shared
        )
        blocking = nslice or blocking
    if blocking is not None:
        return blocking
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
    return _tap_major(g, b.block_k_tiles)


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
    group, whose sender is staggered down a diagonal (McastSenderPlacement.Staggered: group g's sender is its
    participant g % group size). (receivers, group size, receiver order), or None when a column would have too
    few participants or the participants are not one rectangle."""
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


def _shared_cb(total_bytes, first, second, cores):
    """One CB region under two buffer indices, each (index, data_format, page_size)."""
    return ttnn.CBDescriptor(
        total_size=total_bytes,
        core_ranges=cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=i, data_format=f, page_size=p) for i, f, p in (first, second)
        ],
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
    if b.shared_slab_bytes:
        # One region, two CB indices: the slab (dead once the core's gather landed) and the row's combine slots
        # (written by peers only after this core's credit, which follows the gather).
        slab_cb = _shared_cb(
            b.shared_slab_bytes,
            (CB_HALO_SLAB, ttnn.bfloat16, slab_cb.format_descriptors[0].page_size),
            (CB_COMBINE_SLOTS, ttnn.float32, TILE_BYTES_FP32),
            cores,
        )
    face_direct = face_direct_im2col(g, bk)

    # Whole-slab input ring: where every core runs one M-block, cb_input_tiles shares cb_im2col_tiles' L1 (compute
    # tilizes in0 only after its untilize consumed every input batch, and no reader write lands in in0 before then: no
    # center-tap slots) and holds the whole slab, so the reader never waits on a reserve and the split fill hands the
    # writer every batch up front. Costs no L1 (it frees the separate ring).
    def whole_slab_ring(in0_k_tiles):
        return (
            g.tile_input
            and not b.ring
            and b.max_m_blocks_per_core == 1
            and b.center_mode == CENTER_NONE
            and not face_direct
            and b.slab_tile_rows * g.c_in_tiles <= bm * (in0_k_tiles if b.nslice else bk * b.center_slot_blocks)
        )

    # N-slice center taps: in0 holds taps 0 and 2 of every kernel row, cb_center_tiles the center taps, copied from the
    # whole-slab input ring (which must then fit the smaller in0).
    nslice_center = nslice_center_capable(g, b) and whole_slab_ring(g.tensor_k_tiles - 3 * g.c_in_tiles)
    in0_k_tiles = g.tensor_k_tiles - (3 * g.c_in_tiles if nslice_center else 0)
    cbs = [slab_cb]
    if not face_direct:
        cbs.append(_cb(CB_IM2COL_RM, ttnn.bfloat16, TILE_BYTES_BF16, b.im2col_rm_depth_rows * bk, cores))
    cbs += [
        # Face-direct: the reader writes these tiles; it relies on every K-block starting at the CB base.
        _cb(
            CB_IM2COL_TILES,
            input_tensor.dtype,
            g.in0_tile_bytes,
            bm * (in0_k_tiles if b.nslice else bk * b.center_slot_blocks),
            cores,
        ),
        _cb(
            CB_WEIGHTS,
            weight_tensor.dtype,
            g.weight_tile_bytes,
            b.weight_depth * (g.tensor_k_tiles * ns if b.nslice else bk * bn),
            cores,
        ),
        # Unused by the N-slice ring: one page there keeps the face-direct startup's output format defined.
        _cb(CB_MATMUL_PARTIALS, ttnn.float32, TILE_BYTES_FP32, 1 if b.nslice else bm * bn, cores),
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
    input_ring_batches = INPUT_TILE_DEPTH_BATCHES
    alias_ring = whole_slab_ring(in0_k_tiles)
    if alias_ring:
        input_ring_batches = b.slab_tile_rows // b.untilize_batch_rows
    if alias_ring:
        for i, cb in enumerate(cbs):
            if cb.format_descriptors[0].buffer_index == CB_IM2COL_TILES:
                cbs[i] = _shared_cb(
                    cb.total_size,
                    (CB_IM2COL_TILES, input_tensor.dtype, g.in0_tile_bytes),
                    (CB_INPUT_TILES, input_tensor.dtype, g.in0_tile_bytes),
                    cores,
                )
    elif g.tile_input:
        input_tile_pages = input_ring_batches * b.untilize_batch_rows * g.c_in_tiles
        cbs.append(_cb(CB_INPUT_TILES, input_tensor.dtype, g.in0_tile_bytes, input_tile_pages, cores))
    if b.nslice:
        cbs.append(_cb(CB_RING_TOKEN, ttnn.float32, RING_TOKEN_BYTES, 1, cores))
    if tap_major_im2col(g, b):
        cbs.append(_cb(CB_ZERO_SRC, ttnn.bfloat16, pixel_bytes, TILE, cores))
    tap_direct = _tap_direct(g, bm, bk, b.nslice)
    if tap_direct:
        cbs.append(_cb(CB_SRC_TABLE, ttnn.bfloat16, 64, _src_table_bytes(b.im2col_rm_depth_rows) // 64, cores))
    center_mode = b.center_mode
    if center_mode == CENTER_K_BLOCK:
        cbs.append(_cb(CB_CENTER_TILES, input_tensor.dtype, g.in0_tile_bytes, bm * bk, cores))
    elif nslice_center:
        cbs.append(_cb(CB_CENTER_TILES, input_tensor.dtype, g.in0_tile_bytes, 3 * bm * g.c_in_tiles, cores))
    if g.k_split:
        if not b.shared_slab_bytes:
            cbs.append(_cb(CB_COMBINE_SLOTS, ttnn.float32, TILE_BYTES_FP32, g.k_cols * bm * ns, cores))
        if has_bias and not b.nslice:
            cbs.append(_cb(CB_COMBINED, ttnn.float32, TILE_BYTES_FP32, bm * ns, cores))

    # Channel-slice reads (R4): the core's slice of input row v is page v * row_page_stride + col offset,
    # and one owner shard holds input_shard_rows consecutive rows of it contiguously.
    slab_split = SLAB_SPLIT_NOCS and g.sharded_output and not g.tile_input and not g.k_split
    # Split TILE fill. Under a K split (N-slice-outer walk) only with the whole slab in the ring, where the reader hands
    # the writer every batch up front; a per-batch hand-over on the 2-batch ring measured slower than no split there
    # (aspp.convs.1 @ LoFi 75.8 vs 71.0 us).
    tile_fill_split = g.sharded_output and g.tile_input and (not g.k_split or (b.nslice and alias_ring))
    if g.k_split:
        in_spec = input_tensor.memory_config().shard_spec
        page_cols = TILE if g.tile_input else g.c_in_p
        row_page_stride = _div_up(int(input_tensor.padded_shape[-1]), page_cols)
        input_shard_rows = int(in_spec.shape[0]) // (TILE if g.tile_input else 1)
    elif slab_split:
        row_page_stride, input_shard_rows = 0, int(input_tensor.memory_config().shard_spec.shape[0])
    elif tile_fill_split:
        row_page_stride, input_shard_rows = 0, int(input_tensor.memory_config().shard_spec.shape[0]) // TILE
    else:
        row_page_stride, input_shard_rows = 0, 0
    core_m_ranges = _core_m_ranges(g.tensor_m_tiles, b.core_m_tiles)
    m_blocks_per_index = [_div_up(actual_m_tiles, bm) for _, actual_m_tiles in core_m_ranges]
    # The weight multicast's cores (the channel below is built from the same set). The split TILE fill runs on its pure
    # receivers: not the sender (work core 0), nor the own-stream cores outside the channel.
    mcast_receivers = None if g.k_split else _weight_mcast_cores(core_list, [m_blocks_per_index[i] for _, i, _ in work])
    split_cores = set()
    if tile_fill_split and mcast_receivers is not None:
        split_cores = {(c.x, c.y) for c in core_list[1:] if mcast_receivers.contains(c)}
    # K split: each column multiplies against its own weight K-slice, one stream per column (built once: the split
    # fill picks its receivers from it).
    column_channel = _column_weight_mcast(input_tensor, work, m_blocks_per_index) if g.k_split else None
    if tile_fill_split and g.k_split and column_channel is not None:
        # The column channels' pure receivers (the staggered sender of column g is participant row g % rows: its writer
        # blocks in DRAM reads and multicast sends: measured +9% with it split) that read enough from higher shards.
        most = max(m_blocks_per_index[i] for _, i, _ in work)
        part_rows = sorted({i for _, i, _ in work if m_blocks_per_index[i] == most})
        hi_bytes = {
            row: _higher_shard_rows(g, b.core_m_tiles, bm, row, input_shard_rows) * g.c_in_tiles * g.in0_tile_bytes
            for row in part_rows
        }
        for core, row, col in work:
            if row in part_rows and row != part_rows[col % len(part_rows)] and hi_bytes[row] >= SPLIT_FILL_MIN_BYTES:
                split_cores.add((core.x, core.y))

    # Under a K split, the split kernel only where some core splits (no idle semaphores or accessor args otherwise).
    tile_fill_split = tile_fill_split and (not g.k_split or bool(split_cores))
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
    fill_split = slab_split or tile_fill_split
    slab_sem_ids = (len(sem_ids), len(sem_ids) + 1) if fill_split else (0, 0)
    sem_ids += slab_sem_ids if fill_split else ()
    gather_sem_ids = (len(sem_ids), len(sem_ids) + 1) if gather_split else (0, 0)
    sem_ids += gather_sem_ids if gather_split else ()
    # Split row gather: row_go, row_done, row_limit, row_assigned, then ROW_QUEUE mailbox words.
    row_split = _row_split(g, bm, bk)
    strided_window = _strided_window(g, bk)
    # Slab-direct window units (N-slice-outer walk): compute tilizes in-image runs from the slab in place.
    window_direct = strided_window and b.nslice
    row_sem_ids = tuple(range(len(sem_ids), len(sem_ids) + 5)) if row_split else (0, 0, 0, 0, 0)
    sem_ids += tuple(range(len(sem_ids), len(sem_ids) + 4 + ROW_QUEUE)) if row_split else ()
    k_padded = g.tensor_kkh_tiles * TILE > 3 * g.c_in_p

    def zero_im2col_once(slice_channels):
        """The reader zeroes cb_im2col_rm once (mirror of its zero_im2col_once; the tap-major gather copies whole slab
        pixels, lanes included, so its im2col needs no zeroing)."""
        return (
            not face_direct and not tap_major and bk == g.tensor_kkh_tiles and (slice_channels < g.c_in_p or k_padded)
        )

    def reader_ct_args(slice_channels, split_core=None):
        split_core = tile_fill_split if split_core is None else split_core
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
            # Short-slice lane zeroing: slab pixels (ROW_MAJOR; TILE reads a row's padding tiles from the zero block).
            0 if g.tile_input else b.slab_px,
            CB_ZERO_SRC,
            int(b.nslice),
            CB_RING_TOKEN,
            SEM_COMBINE_CREDIT,
            SEM_COMBINE_ARRIVAL,
            int(row_split),
            ROW_QUEUE,
            *row_sem_ids,
            # Shared-L1 N-slice walk: slab_dead reuses send_go's id, which the N-slice walk never uses.
            int(b.shared_slab_bytes > 0),
            SEM_COMBINE_SEND_GO,
            int(split_core),
            int(strided_window),
            int(window_direct),
            center_mode,
            CB_CENTER_TILES,
            int(b.ring),
            int(tap_direct),
            CB_SRC_TABLE,
            # N-slice ring: tile-rows per forwarded row group, the compute's slice subblock height.
            _ring_subblock(bm, ns)[0],
            # Tap-direct group: at most half the ring, so compute always has the other half.
            max(1, min(TAP_DIRECT_GROUP, b.im2col_rm_depth_rows // 2)) if tap_direct else 1,
            input_ring_batches,
            # A window past the ring would reserve more than the CB holds.
            min(FILL_WINDOW_BATCHES, input_ring_batches) if g.tile_input and not b.ring else 1,
            int(nslice_center),
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
        int(b.nslice),
        CB_RING_TOKEN,
        *_ring_subblock(bm, ns),
        b.untilize_batch_rows,
        int(strided_window),
        center_mode,
        CB_CENTER_TILES,
        int(b.ring),
        int(tap_direct),
        CB_SRC_TABLE,
        b.im2col_rm_depth_rows,
        int(nslice_center),
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
        int(b.nslice),
        b.weight_depth,
        int(row_split),
        ROW_QUEUE,
        *row_sem_ids,
        CB_IM2COL_RM,
        b.im2col_rm_depth_rows,
        g.c_in_p,  # this core's slice channels, set per writer kernel
        int(g.tile_input),
        0,  # zero_im2col_once, set per writer kernel
        int(b.shared_slab_bytes > 0),
        SEM_COMBINE_SEND_GO,  # slab_dead
    ]
    writer_slice_arg = len(writer_ct_args) - 5
    writer_split_arg = len(writer_ct_args)
    writer_ct_args += [
        int(tile_fill_split),
        CB_INPUT_TILES,
        b.slab_tile_rows,
        b.untilize_batch_rows,
        input_ring_batches,
        int(b.ring),
        int(g.k_split),
        row_page_stride,
        0,  # fill_zero_lanes, set per writer kernel
        CB_ZERO_SRC,
        FILL_WRITER_WINDOW_BATCHES if not b.ring else 1,
    ]
    writer_zero_lanes_arg = len(writer_ct_args) - 3

    def writer_ct_args_for(slice_channels, split_core=None):
        args = list(writer_ct_args)
        args[writer_slice_arg] = slice_channels
        args[writer_slice_arg + 2] = int(zero_im2col_once(slice_channels))
        args[writer_zero_lanes_arg] = int(tap_major and slice_channels < g.c_in_p)
        if split_core is not None:
            args[writer_split_arg] = int(split_core)
        return args

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
        if fill_split
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    # One reader kernel per distinct slice width: the gather's lane handling is compile-time.
    reader_groups = {}  # slice channels -> (cores, RuntimeArgs)
    writer_groups = {}  # slice channels -> (cores, RuntimeArgs): the split row gather's lane handling is compile-time
    compute_rt = ttnn.RuntimeArgs()
    input_addr = input_tensor.buffer_address()
    weight_addr = weight_tensor.buffer_address()
    bias_addr = bias_tensor.buffer_address() if has_bias else 0
    output_addr = output_tensor.buffer_address()
    device = input_tensor.device()
    row_peer_coords = {}
    if g.k_split:
        for core, row, _ in work:
            virtual = device.worker_core_from_logical_core(core)
            row_peer_coords.setdefault(row, []).extend([virtual.x, virtual.y])

    def split_of(core):
        return (core.x, core.y) in split_cores

    for core, m_index, col in work:
        m_tile_start, actual_m_tiles = core_m_ranges[m_index]
        num_m_blocks_this_core = m_blocks_per_index[m_index]
        # The core's output columns: all of C_out, or N-slice `col` under the K split (possibly short or
        # empty past the last C_out tile).
        out_n_tile_start = col * ns
        out_valid_tiles = max(0, min(ns, g.tensor_n_tiles - out_n_tile_start))
        # The split is a runtime flag: a per-core kernel split fragments the core ranges (a staggered sender per
        # column), and the launch of many-range kernels measured up to ~10 us of start skew.
        reader_cores, reader_rt = reader_groups.setdefault(
            (_slice_channels(g, col), bool(split_cores)), ([], ttnn.RuntimeArgs())
        )
        reader_cores.append(core)
        reader_rt[core.x][core.y] = (
            [
                input_addr,
                m_tile_start,
                actual_m_tiles,
                num_m_blocks_this_core,
                m_index * SLAB_READ_STAGGER_PX_PER_CORE,
                col * pages_per_slice_row,
                col,
                m_index,
            ]
            + row_peer_coords.get(m_index, [])
            + [int(split_of(core))]
        )
        compute_args = [num_m_blocks_this_core, int(out_valid_tiles > 0), col]
        if b.ring:
            # Per M-block: tile-rows to untilize, and the unread tail rows of its last cb_input_tiles batch.
            ub = b.untilize_batch_rows
            for _, _, new_rows in ring_plan(g, b.core_m_tiles, bm)[m_index]:
                compute_args += [new_rows, -new_rows % ub]
        compute_rt[core.x][core.y] = compute_args
        writer_cores, writer_rt = writer_groups.setdefault(
            (_slice_channels(g, col), bool(split_cores)), ([], ttnn.RuntimeArgs())
        )
        writer_cores.append(core)
        writer_rt[core.x][core.y] = (
            [
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
            ]
            + row_peer_coords.get(m_index, [])
            + [int(split_of(core))]
        )

    readers = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "conv2d_3x3_reader.cpp"),
            core_ranges=_core_range_set(reader_cores),
            compile_time_args=reader_ct_args(*slice_channels),
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        for slice_channels, (reader_cores, reader_rt) in reader_groups.items()
    ]
    writers = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "conv2d_3x3_writer.cpp"),
            core_ranges=_core_range_set(writer_cores),
            compile_time_args=writer_ct_args_for(*slice_channels),
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        )
        for slice_channels, (writer_cores, writer_rt) in writer_groups.items()
    ]
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
        if column_channel is not None:
            receivers, group_size, order = column_channel
            # Staggered senders: with every column's sender on one grid row, all the columns' DRAM read responses
            # share that row's NoC links, and the weight chain is bound there (weights-only chain on aspp.convs.1
            # @ LoFi: 65.5 us with the senders on row 0, 38.3 us staggered; whole op 72.1 -> 71.0 us, aspp.convs.2
            # 74.4 -> 69.8 us).
            sender_config = ttnn.McastFixedSenderConfig(sender_index=0, placement=ttnn.McastSenderPlacement.Staggered)
            weight_mcast = ttnn.Mcast(device, weight_mcast_config, receivers, group_size, sender_config, order)
    else:
        if mcast_receivers is not None:
            weight_mcast = ttnn.Mcast(
                device,
                weight_mcast_config,
                mcast_receivers,
                mcast_receivers.num_cores(),
                ttnn.McastExplicitFixedSenderConfig([core_list[0]]),
            )
    if weight_mcast is None:
        for writer in writers:
            ttnn.attach_absent_mcast(writer, WEIGHT_MCAST_PREFIX)
    else:
        weight_mcast.attach(descriptor, WEIGHT_MCAST_PREFIX, writers, len(semaphores))
    descriptor.kernels = readers + writers + [compute]
    return descriptor
