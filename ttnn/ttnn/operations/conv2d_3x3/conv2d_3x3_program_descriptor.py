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
single shard column is R2 on that column. The M split runs down the fewest grid rows, from the shard grid's first
row to the device edge, that give the fewest output tile-rows per row (block_compute_rows); where those rows are not
the output's shard rows, column y packs its finished block into a local CB and the writer NoC-writes each tile to
the output shard that holds it (the output keeps the input's shard grid, shard shape and orientation). On the N-slice
walk those rows may own different tile-row counts (balanced_block_ranges: balanced by each row's prelude and in-image
kernel-row walk), the CT block height being the largest.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, replace
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
# Four pages: the packer runs up to four groups ahead of the reader's forwards, and the reader keeps the arrival count
# it writes behind each forward in one of the first four pages (one per transaction id in rotation).
CB_RING_TOKEN = 10
RING_TOKEN_BYTES = 16
RING_TOKEN_PAGES = 4
# Tap-major only: 32 zero pixels, so an out-of-image tap run is one read instead of one per 512 B MEM_ZEROS chunk.
CB_ZERO_SRC = 11
# Center-tap direct in0 (see CENTER_TAP_DIRECT): the kw = 1 K-blocks' input tiles, copied by the reader.
CB_CENTER_TILES = 12
# (N-slice center taps reuse it: [3, bm, c_in_tiles], see NSLICE_CENTER_TAPS.)
# Tap-direct (see _tap_direct): N-slice prelude, one 16 B entry of tap source addresses per cb_im2col_rm unit slot;
# K-blocked walk, a FIFO of one 16 B entry per unit (_tap_direct_fifo_units).
CB_SRC_TABLE = 13

# kw-fold halo exchange tokens (conv2d_3x3_kw_fold.hpp): compute pack -> writer (a boundary send is ready), writer ->
# compute unpack (a neighbour's rows landed).
CB_KWF_HALO_OUT = 14
CB_KWF_HALO_IN = 15
# Bank-run fill: the ROW_MAJOR DRAM-interleaved slab fill reads each chunk of consecutive pixels as one run per DRAM
# bank into a staging ring (BRF_NSTG chunks of up to BRF_CHUNK_BYTES), then de-interleaves it into the slab with local
# NoC copies. One read per pixel page is DRAM-transaction-bound (~600 M reads/s across the chip); bank runs are
# bandwidth-bound.
BRF_NSTG = 4
BRF_CHUNK_BYTES = 16384
BRF_MIN_READ_BYTES = 2048  # smallest per-bank read worth a staging chunk (else fewer, larger chunks)


def _brf_staging(g, slack_bytes):
    """(chunks, chunk pixels) of the staging ring that fit in slack_bytes, or None (the per-pixel fill). A chunk is
    whole bank rows (a multiple of 8 pixels), at most BRF_CHUNK_BYTES; the most chunks whose chunk is >= 8 KB (a bank
    read of >= 1 KB), else 2 chunks of what fits."""
    pixel_bytes = g.c_in_p * 2
    for nstg in range(BRF_NSTG, 1, -1):
        chunk_px = min(BRF_CHUNK_BYTES, slack_bytes // nstg) // pixel_bytes // 8 * 8
        if chunk_px * pixel_bytes >= 8 * BRF_MIN_READ_BYTES or (nstg == 2 and chunk_px >= 8):
            return nstg, chunk_px
    return None


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

# Padded slab (kernels/conv2d_3x3_padded_slab.hpp), on the N-slice center-tap prelude (stride 1, p == d, W % 32 == 0,
# one M-block per core): compute untilizes each image row of the slab into its own slot of W + d pixels behind d zero
# pixels (the reader zeroes the gaps with its center copy), so every tap block of every tile-row is one contiguous slab
# run or the zero block. Compute then tilizes the prelude from sources it computes itself (padded_tilize): the reader
# publishes no table, gathers no edge tap and runs no per-group handshake; it only fills, copies the center taps and
# waits for a one-page "slab dead" token. The unit ring shrinks to one unit (format only). The slab grows by d / W plus
# the partial rows at the band ends, taken only where it fits the leftover L1 plus the freed ring. The fill's unread
# tail rows are no longer untilized. Measured (Blackhole p150b, LoFi, bit-identical): res4.x 15901 -> 13431 ns (median
# of 3; the prelude tilize 4.5 -> 2.4 us, now at its 1-wide payload rate), res5.x -3% to -5%, aspp.convs.x flat.
PADDED_SLAB = True

# Tap shift (kernels/conv2d_3x3_tap_shift.hpp), on the padded-slab prelude with d < 32: compute builds the kw = 0 / 2
# taps on the FPU straight from the input ring's TILE slab, kernel-row major, each group of taps waiting only for the
# slab rows it reads: no untilize, no row-major slab, no gap zeroing. Each input tile is unpacked once per (kernel row,
# channel tile, pair of output tile-rows) into a pixel-contiguous DEST row buffer, and the packer reads every tap as a
# shifted DEST window (~34-42 ns per tap, against ~60 for the per-row FPU shift of Perf 10 and ~23-42 for the fast /
# 1-wide tilize). The input ring leaves in0's L1 for the slab / combine-slots region (dead until the slab-dead token,
# as the slab was), so the prelude no longer waits for the reader's center push and runs behind the fill. Taken where
# the region, rounded to a multiple of its three page sizes, fits the L1 the padded slab left. Measured (Blackhole
# p150b, LoFi, bit-identical, unzoned, medians of alternating pairs vs the Perf 10 per-row shift): see changelog Perf 11
# (tap_shift_rate). It replaced the Perf 10 carve-out (halo tile-rows per side < 3 x slice tiles: aspp.convs.1,
# res5.0), which now gains too. False keeps the padded-slab untilize / tilize prelude.
TAP_SHIFT = True

# Stride-2 taps from DEST (kernels/conv2d_3x3_s2_taps.hpp, on the strided-window prelude: stride 2, p 1, d 1, TILE input,
# W % 64 == 0): compute builds every tap tile straight from the input's TILE slab, one DEST section per output tile:
# 4-row SrcB -> DEST moves into a pixel-contiguous buffer (kw = 1: the packer decimates it with interface mask 0101) and
# one-row moves of the odd pixels into an O plane (kw = 2 and, one row earlier, kw = 0: 4-row windows), in input-row
# order behind the fill. No untilize, no RM slab, no window gather, no strided tilize: the reader only fills. The input
# ring holds the whole slab in its own L1 (not in0's) and in0 holds every K-block (the K-blocked walk's tilize hook does
# nothing). Taken where those CBs fit the L1 the RM slab and the unit ring give back. Measured (Blackhole p150b, unzoned,
# 2 alternating pairs, bit-identical): res3.0 @ LoFi 17.45 -> 13.41 us, res4.0 @ LoFi 15.02 -> 12.53 us; changelog
# Perf 12 (stride2_dest_taps). False keeps the untilize / window gather / strided tilize prelude.
S2_TAPS = True
# S2_TAPS fill batches: at most this many input tile-rows per cb_input_tiles push (a divisor of the untilize batch), so
# the tap build, which waits per run for the slab rows it reads, starts as the first rows land.
S2_FILL_BATCH_ROWS = 4

# Work-balanced BLOCK M split (balanced_block_ranges). On the N-slice ring walk a grid row's time is its prelude (fill-
# paced untilize of its slab rows, gather-paced tilize of its tile-rows) plus its walk (in-image kernel rows of each ring
# row group, after the kernel-row skip), so with dilation the rows whose output rows keep all three kernel rows in the
# image are critical while the edge rows idle. balanced_m_ranges splits M into contiguous per-row ranges of different
# tile-row counts that minimize the costliest row of a cost model; the CT block height is the largest range, the kernels
# size the prelude, walk and ring forwarding by each row's valid tile-rows at runtime, and the walk's group matmuls take
# the group height as a template constant. Measured (Blackhole p150b, LoFi, unchanged precision: bit-identical):
# aspp.convs.2 59.4 -> 51.1 us, aspp.convs.1 58.6 -> 53.6, aspp.convs.3 27.1 -> 26.1, res5.1 36.6 -> 34.9, res5.2 36.6 ->
# 35.6, res5.0 flat; HiFi4 BLOCK shapes -3% to -13%.
# Row cost model, cycles (fitted on per-row zones of aspp.convs.1/2/3 and res5.2 @ LoFi): untilize per slab tile (fill-
# paced), tilize per unit (tile-row x K-block) and per in0 tile (gather-paced), walk per k-tile of a ring row group of r
# rows (an r x 1 subblock costs nearly as much for 1 row as for 4: the in1 unpack and the call cost are per group), and
# per ring row group and step (token, pack, re-init). The uniform split's group heights are runtime (aspp.convs.2 per
# grid row over 8 steps, 3 kernel rows: 4 + 1 rows 38.2 us; 2 kernel rows: 4 + 4 rows 30.0 us, 4 + 3 rows 28.0 us), the
# balanced split's template constants (res5.2 4 + 1 rows, aspp.convs.1 4 + 2 rows; a 3-row group measured no faster).
BALANCE_UNTILIZE_TILE = 115
BALANCE_TILIZE_UNIT = 150
BALANCE_TILIZE_TILE = 30
BALANCE_WALK_GROUP_KTILE = {1: 49, 2: 54, 3: 61, 4: 70}
BALANCE_WALK_GROUP_KTILE_CT = {1: 46, 2: 48, 3: 61, 4: 70}
BALANCE_WALK_GROUP = 180
# Among block heights whose costliest row is within this fraction of the best, the smallest wins (less L1, and the
# model's prelude term is coarsest on the longest rows).
BALANCE_CAP_TIE = 0.01
# Largest range tried above the uniform split's (each step costs the CT block one more tile-row of L1 on every core).
BALANCE_MAX_EXTRA_ROWS = 2
# The balanced split is taken only when the model's costliest row drops by at least this fraction.
BALANCE_MIN_GAIN = 0.02

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
# Slab-direct face im2col (kernels/conv2d_3x3_slab_direct.hpp): on face-direct whole-K shapes whose output rows are
# whole tile-rows, compute unpacks every in0 tile straight from a padded slab (one zero gap of `padding` pixels in front
# of every input row) as two contiguous 1 KB runs, so neither data-movement RISC gathers anything. False keeps the
# face-direct gather.
SLAB_DIRECT = True
# Slab-direct with a bias: add it as one more matmul K-step (ones block x the bias tile split into two parts of at most
# 5 significant bits, exact under LoFi), so the matmul packs the final values and no bias pass runs. False keeps the
# fp32 partials + bias-add pass.
SD_FUSED_BIAS = True
# Slab-direct streamed fill (kernels/conv2d_3x3_slab_stream.hpp): each data-movement RISC reads its part of the slab
# in chunks (the reader SD_STREAM_CHUNK_PX pixels, at most SD_STREAM_NTRID_R in flight under transaction ids 3..15; the
# writer SD_STREAM_CHUNK_PX_W pixels, at most SD_STREAM_NTRID under ids 9..15), the reader pushes the slab CB as the
# landed prefix grows, and compute multiplies each subblock row once the slab pages it reads are in. 0 keeps one push
# per block. stem.conv1, unzoned: 512-px chunks with 6 ids each 10376 ns, 1024 px with 13 / 7 ids 10066-10316 ns, 2048
# px 10307 ns (a small chunk window throttles the reader's fill).
SD_STREAM_CHUNK_PX = 1024
SD_STREAM_NTRID_R = 13
SD_STREAM_CHUNK_PX_W = 1024
SD_STREAM_NTRID = 7
# Stream only blocks of at least this many subblock rows: with fewer, the first matmul call already needs (nearly) the
# whole slab, so there is no fill to hide and the chunked fill costs time (bm 1, 32x34x16 -> 64: +6%; 64x64x8 -> 32:
# +28%; bm 2 at 2-row subblocks, 64x64x16 -> 64: +3.5%; bm 3 at 1-row subblocks, 128x256x8 -> 64: +0.3%).
SD_STREAM_MIN_SUBBLOCK_ROWS = 3
# Slab-direct round 0 over NoC0: with a fixed-sender weight multicast (no weight servers), the sender core's reader
# multicasts the first weight round on NoC0 (kernels/conv2d_3x3_reader.cpp, SD_R0_NOC0), which the forward fill of the
# critical cores (the writers' NoC1 reads of higher shards) does not use; its writer reads the round from DRAM first
# thing at kernel entry. stem.conv1 @ LoFi, unzoned, 3 alternating pairs: 10.27 -> 9.83 us (the NoC1 multicast slowed
# the last grid row's forward fill by ~0.4 us; a diagnostic with no multicast at all: 9.80 us). No receiver ack (its slot
# is fresh at kernel start): 64x128x8 -> 64 s2 3.34 -> 2.98 us, stem.conv1 flat.


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
# Tap-direct on the K-blocked walk: the reader reserves, fences and pushes this many unit entries at a time.
TAP_DIRECT_FIFO_GROUP = 4
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
# Under a K split (BLOCK) the slab fill is NoC0-column-link bound and lands nearly all at once, so the untilize, which
# waits per batch, started only once the whole slab was in (aspp.convs.2: three 8-tile-row batches landing within
# 0.4 us of each other at ~6 us). Batches of at most KSPLIT_UNTILIZE_BATCH_ROWS tile-rows with up to
# KSPLIT_FILL_WINDOW_BATCHES of them in flight keep the whole slab in flight and let the untilize start on the first
# batches while the rest lands. 4 rows: a whole DEST section of the 1-wide untilize (3 measured +1.7% on res4.x);
# aspp.convs.2 50.4 -> 49.7 us, aspp.convs.1 53.3 -> 52.3 us (zoned, median of 3), flat on res4 / res5 / HiFi4 BLOCK.
# Fills without a K split keep the default batches and window: there the same values measured slower (unzoned, 2 runs
# each, non-overlapping: DRAM interleaved H64 W128 C_in 128 and H128 W256 C_in 64 TILE bf8_b +1.9% / +2.1%, HEIGHT
# stem.conv2 / res3.0 @ LoFi +1.4% / +1.7%).
KSPLIT_UNTILIZE_BATCH_ROWS = 4
KSPLIT_FILL_WINDOW_BATCHES = 8
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
# Weight servers (HEIGHT-sharded K-blocked walk): when the device has a spare core per rectangle of the weight channel,
# each rectangle's rounds come from one spare core outside the work grid (its reader reads DRAM on NoC0, its writer
# multicasts on NoC1) instead of from work core 0, which read on NoC1 (the unfavourable read NoC, ~23 GB/s) and then
# multicast every round to each rectangle in turn. Each round is read and multicast in up to WEIGHT_SERVER_CHUNKS
# chunks. Measured (LoFi, bf8_b): res3.x 18.2 -> 15.7 us, sem dec res3 fc0 59.9 -> 45.9, fc1 49.5 -> 36.3, inst dec
# res3 fc0 35.3 -> 29.0, at the weights-stubbed bound. Not on interleaved input: the server's NoC0 DRAM reads meet
# every core's NoC0 DRAM fill there (32x64x128 ROW_MAJOR +4.9%, 64x128x128 +1.5% / +2.3%).
# The server writer multicasts every landed chunk in one linked chain (one path set-up per chain, not per chunk), the
# server multicasts the bias too (no receiver reads it from DRAM), and a receiver acks a slot's reuse as soon as compute
# freed it. Measured together (LoFi, bf8_b, perf focus, bit-identical): res3.x 14.95 -> 12.12 (medians of 3), res2
# 13.7 -> 12.2, center_head.1 11.7 -> 11.0, inst res2 fc1 -2.5%, the other server-fed layers flat. 3 chunks per round:
# on res3.x 6 measured 12.6-12.9 (a chunk per path set-up where the reads pace the stream), 4 12.9, 2 12.3, 3 12.1.
WEIGHT_SERVERS = True
WEIGHT_SERVER_CHUNKS = 3
# Dual-source weight servers (kw-fold): where the spare cores allow, every rectangle gets a second server, its partner,
# which reads and multicasts half of every chunk (whole slot rows) on NoC0 while the first server sends the other half
# on NoC1: one multicast source per rectangle capped the round at ~34 GB/s beside the fill, and round 0 gated compute's
# first kernel rows (conv2d_3x3_weight_server_writer.cpp). Both servers issue every chunk of a round ahead
# (conv2d_3x3_weight_server_reader.cpp). Placement (_weight_servers): partners first, each the spare core with the
# fewest NoC0 hops (+x then +y, torus) to its rectangle's low corner whose path crosses no receiver rectangle, then the
# first servers the same way on NoC1 to the high corner. Nearest-core placement routes a partner's NoC0 multicast the
# long way round through another rectangle (res3.x: the 9-core row rectangle's partner through the 77-core one, 11.9
# us), and a rectangle left on one server becomes the critical one. Measured (Blackhole p150b, LoFi bf8_b,
# unzoned, bit-identical): res3.x 9.82 -> 8.94 us (medians of 3), res2 -5.1%, inst dec res3 fc0 -2.0%, sem dec res3 fc0
# -1.7%, sem dec res3 fc1 -1.1%, the other kw-fold layers flat. Not off kw-fold: there the first matmul waits for the
# tilize, round 0 is hidden, and the partner's NoC0 multicast only slows the reader's NoC0 fill (res3.0, stride 2:
# fill 3.62 -> 3.93 us, +2.0% to +2.7% over 3 options x 3 runs).
# Round 0 gates compute's first subblock. Each partner's writer reads the partner's part of round 0 itself, on NoC0 at
# kernel start, while its reader reads the later rounds on NoC1, and the first server takes WEIGHT_SERVER_R0_SHARE of
# round 0's slot rows (rounded, the extra rows to the earliest chunks; later rounds keep the halves)
# (conv2d_3x3_weight_server_writer.cpp). The partner is round 0's late half otherwise: its kernel starts ~0.3-0.45 us
# after the first server's (the last kernel group's launch message lands last), and on the 103-core layers its NoC1
# responses run west along row 11 through the 4-core rectangle, under that rectangle's own NoC1 multicast (inst res2
# fc1, zoned: first server done ~2.2 us, partner ~2.9-3.5, round 0 complete ~3.0-3.5 -> 2.54 us). The NoC0 read and
# the share go together: on the 86-core layers either alone slowed the fill (res3.x NoC0 read +2.9%, share +0.5% to
# +1.4%), both flat (-0.7%). Measured (unzoned, LoFi bf8_b, bit-identical): inst res2 fc1 -1.6%, inst dec res2 fc0
# -2.0% (pooled medians), the other kw-fold layers flat.
WEIGHT_SERVER_PARTNERS = True
WEIGHT_SERVER_R0_SHARE = (7, 12)
# Blackhole NoC grid (x, y), the torus of the partner placement's hop metric.
_NOC_GRID = (17, 12)
# Hardware multicast rectangles one channel may carry (dataflow_kernel_lib::MAX_MCAST_RECTANGLES).
MAX_MCAST_RECTANGLES = 3
# Spare-column M-groups (K split on the N-slice walk, one M-block per core): the device columns past the shard grid
# (ROW_MAJOR; the rows below it on COL_MAJOR) host more M-groups, one per spare column, laid out transposed: channel
# slice k of the group sits on the spare column's row k, and its ring runs down the column (one hop per step, as on a
# grid row). Taken where more groups lower the output tile-rows per group (block_spare_groups): res4.x and res5.x run 13
# groups of 5 tile-rows on 104 cores instead of 10 of 7 on 80. A member fills its slab from its channel slice's shards
# like any extended row, and the owners store remotely. The spare groups take the cheapest M ranges (spare_m_indices).
# Their weights come from the column weight streams without another DRAM read: the column participant after the sender
# forwards each round, once it landed and was pushed, to the spare members of its slice (one row of the spare columns:
# a second channel, WEIGHT_MCAST_SPARE_PREFIX, with the forwarder as its explicit sender).
# Measured (Blackhole p150b, unzoned, medians of 3, bit-identical), with the N-slice async bias and the wide short
# group (compute RING_WIDE_SHORT_GROUP): res4.x @ LoFi 12098 -> 11607 ns, res5.0 33368 -> 28524, res5.1 32741 -> 28131,
# res5.2 32436 -> 26806; HiFi4 res4 30491 -> 22859, res5.0 103157 -> 79981. Options measured and not taken (zoned):
# the column sender multicasting each round twice (+0.3 us per res4.x round on the stream every core waits for: res4.x
# +9%); one DRAM stream per spare row (same pages as the column sender's, same time: res5.1 / res5.2 +6% / +7%); the
# spare members' fill split over both NoCs (flat); the last M ranges instead of the cheapest (res5.2 +1.4 us).
SPARE_MGROUPS = True
# N-slice round 0 (writer NSLICE_ROUND0): every weight slot is fresh at kernel start, so round 0 of each column stream
# goes out without a handshake from one round-0 reader per column, its read issued at walk setup; the column sender
# relays it to its slice's spare members from its own row (the forwarder starts at round 1). On Blackhole, with one
# ND shard per round of at least NSLICE_ROUND0_MODEL_MIN_BYTES and one grid row per column, the reader is the column
# participant whose DRAM read path is least shared (_round0_reader_rows); elsewhere it is the column sender.
# Measured (Blackhole p150b, LoFi, unzoned, medians of 3 in two sessions, bit-identical; the async bias read after
# round 0 and fresh slots read without the slot-reuse flush are part of it): res5.0 -1.9% / -2.6% (27402 -> 26876 ns),
# res4.x -1.7% / -1.3%, res5.1 and res5.2 within +-1.2%: bounded by the spare members' tap-shift prelude (fill-bound).
# With the fill stubbed (round 0 then gates every core): res5.0 -7.1%, res5.1 -7.2%, res5.2 -7.1%, res4.x -4.0%; the
# reader rows are worth -2.4 / -7.3 points of that on res5.1 / res5.2 (readers on the sender rows: -4.8% / +0.2%).
NSLICE_ROUND0_MODEL = True
NSLICE_ROUND0_MODEL_MIN_BYTES = 16384
# N-slice spread column stream (writer NSLICE_SPREAD): where every weight round of a core's stream fits its own
# cb_weights slot (depth = the core's rounds, from the L1 left after the solve) and one flag semaphore per round fits,
# no round needs a handshake: every slot is fresh at kernel start, as round 0's already is. Round r's reader member
# reads it from DRAM into slot r and multicasts it down its column, then sets flag semaphore r there (same path, after
# the data); the column sender relays every round to its slice's spare members as it lands on it. Readers: round 0 the
# NSLICE_ROUND0 reader, round 1 the column sender (the handshake stream's start-up reads), round r >= 2 the column's
# other rows in turn. A reader issues round r >= NSLICE_SPREAD_LOOKAHEAD once round r - NSLICE_SPREAD_LOOKAHEAD landed
# on it, and round 1 goes down the column only once round 0 landed on its sender (nothing shares the column's links
# with round 0, which gates every walk). The handshake stream's round was a serial chain: a receiver acks round r + 1
# only once round r landed, the sender multicasts once every ack is in (res4.x: ~0.85 us a round against ~0.75 us of
# step compute; the spare members, one forward later, ended the op). Measured: changelog Perf 14 (nslice_stream_spread).
NSLICE_SPREAD = True
NSLICE_SPREAD_LOOKAHEAD = 2
# Blackhole NoC1 DRAM endpoint row per bank (soc descriptor dram_views worker_endpoint[1]; the relocated set once
# CMFW moved its MRISC port), keyed by whether the endpoints are relocated. Banks 0-3 sit on DRAM column x 0, 4-7 on
# x 9; worker rows are physical y 2..11.
_ROUND0_NOC1_EP_Y = {
    False: {0: 1, 1: 10, 2: 4, 3: 7, 4: 1, 5: 10, 6: 4, 7: 7},
    True: {0: 11, 1: 3, 2: 8, 3: 6, 4: 0, 5: 2, 6: 9, 7: 5},
}
# Semaphore ids per program (the device's limit: ids 0..15).
MAX_SEMAPHORES = 16
WEIGHT_MCAST_SPARE_PREFIX = "weight_mcast2"
# Host-planned BLOCK fill (K split on the N-slice walk, one M-block per core, the whole slab in the input ring): the
# host passes each data-movement RISC its slab reads, per untilize batch, as (owner core, offset in the owner's shard,
# slab row, rows) pieces (_host_fill_args). The reader issues them at its fill start and plans the device-side slab
# state (center taps, band pages) under the reads; the writer of a split core issues its part at kernel start (the
# ring is empty then) and counts each batch in slab_fwd_done as it lands, with no hand-over. A short last slice reads
# its real tiles only and zeroes the padding tiles, as before. On the device the writer's split fill cost ~0.4 us of
# planning before its first read and ~0.27 us of RISC time per 4-row batch (identical on every core: control, not
# NoC). Spare-column members split their fill too (SPARE_SPLIT_FILL): all 8 members of a group take their slab from
# the same owner rows, so on NoC0 every response funnels down the spare column (res4.0: idx 11 reads owner rows 8 / 9
# and wraps down column 14; fills ended 4.1-4.8 us vs ~1.6-2.5 on grid cores, and the backed-up reads slowed the grid
# cores of columns 2-4 to ~4.5 us); the writer reads the slab's upper half on NoC1, the column's other links. Measured
# (Blackhole p150b, unzoned, bit-identical; changelog Perf 13, block_fill_start): res4.0 @ LoFi 12.55 -> 11.27 us
# (-10%, five sessions), HiFi4 res4.0 -6%, res5.1 / res5.2 -1% to -3%, the rest of the BLOCK set flat. Options: the
# host plan alone (spare reads still on NoC0) +1% on res4.0; the spare split with the device-planned writer -2% to -5%;
# the writer reading the whole spare slab -9%. The writer stops serving the fill in its waits once the fill is counted
# (serving an empty fill in every poll iteration measured +1.6% on res4.x).
BLOCK_HOST_FILL = True
SPARE_SPLIT_FILL = True
# Writer batches in flight under the host-planned fill (transaction ids 8..15).
HOST_FILL_WRITER_WINDOW = 8
# Host-planned words per core at most (a core's runtime args stay under the device's 341).
HOST_FILL_MAX_WORDS = 256
# Balanced, kernel-row-gated host fill (BLOCK_HOST_FILL on Blackhole; _balance_fill_nocs, _host_fill_args). A grid-row
# core reads its slab from its column's owner shards, and every response of a same-column read runs along that column:
# on NoC0 (+y) the higher-index shards' rows come the long way round, on NoC1 (-y) the lower ones. The direction split
# (higher shards on the writer, not on the column sender) left aspp.convs.1's middle NoC0 column links with 268 KB
# (325 KB in the columns whose sender row read its higher shards on NoC0) against <= 83 KB on NoC1 (a link moves ~62
# B/cycle). BLOCK_FILL_BALANCE: per column, the host moves single rows of any (core, owner) piece between the NoC0
# reader and the NoC1 writer, either way round the column ring, while that lowers a smooth max (sum of eighth powers)
# of the column's link, owner-injection and reader-ejection loads, with spare-column members' reads from the column's
# owners as fixed load: busiest link 268 -> ~183 KB (an LP optimum: ~182). A core keeps no writer share under
# SPLIT_FILL_MIN_BYTES. BLOCK_FILL_GATE: the tap build reads the slab kernel row by kernel row, and with every batch in
# flight at once the first band (the farthest rows) landed with the rest (aspp.convs.1 critical core: first band 5.8
# us, fill end 6.8); each RISC now issues the first band's batches alone and the rest once they landed. Both only where
# the kernel-row bands are disjoint (_bands_disjoint): with overlapping bands (small d, stride 2) the slab is one short
# run, and the moved rows and the gate only add latency (32x64x128 on 4 slices, d 1: +7.3%; stride 2 on 4 slices:
# +6% / +7%). The reader counts the writer's landed batches itself from NoC1's transaction-id counters (the writer
# publishes how many it issued), so a writer with weight duties never has to serve the fill in its waits.
# Measured (Blackhole p150b, unzoned, bit-identical; changelog Perf 14, block_fill_balance): aspp.convs.1 @ LoFi
# 47.78 -> 45.64 us (medians of 5 pairs over 3 sessions; one session's 3 alternating pairs 47.18 -> 45.64),
# aspp.convs.2 -5.0%, aspp.convs.3 -4.5%, 2 slices d 8 -2.9%, HiFi4 aspp.convs.1 / .2 / .3 -1.3% / -1.1% / -1.5%, the
# rest flat. Options measured on aspp.convs.1: the balance alone (all batches at once) -2.5%, the gate alone flat, the
# direction split with the sender rows split too -1.8%; the reader's zero block issued behind the fill, or its first
# band's batches one at a time, slower.
BLOCK_FILL_BALANCE = True
BLOCK_FILL_GATE = True
# Column ring rows per NoC on Blackhole (physical y 0..11; worker rows 2..11).
_BH_RING_ROWS = 12


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
    # int p (all four sides), or per-side (top, bottom, left, right) when the sides differ. Coordinates move only with
    # pad_top (rows) and pad_left (columns); bottom / right only set the output extent. The fast paths gated on
    # `padding == dilation` (or == 1) take the int form only.
    padding: int | tuple
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
    # K split on more grid rows than the output has shard rows (block_compute_rows): cb_output_tiles is a local CB of
    # one [bm, out_n] block, which the writer stores to the output shards.
    remote_output: bool = False
    # K split with a work-balanced M split (balanced_m_ranges): grid row i owns output tile-rows
    # [m_ranges[i][0], m_ranges[i][0] + m_ranges[i][1]); empty: the uniform split (_core_m_ranges).
    m_ranges: tuple = ()
    # Spare-column M-groups (SPARE_MGROUPS): M-groups on grid rows, the others transposed in the spare columns
    # (spare_m_indices, block_spare_core). 0: every group is a grid row.
    block_rows: int = 0

    @property
    def pad_sides(self):
        """(top, bottom, left, right)."""
        return self.padding if isinstance(self.padding, tuple) else (self.padding,) * 4

    @property
    def pad_top(self):
        return self.pad_sides[0]

    @property
    def pad_left(self):
        return self.pad_sides[2]

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


def ring_owner_cols(g: Conv3x3Geometry, b: Conv3x3Blocking):
    """N-slice ring: the columns that sum and store an N-slice (in N-slice order), or None for columns 0 ..
    num_owning - 1. Where C_out has fewer N-slices than the row has columns and the output goes through the remote
    store, the non-owning columns are spread round the ring, one per gap, the last column among them (it carries the
    short channel slice when C_in does not divide): kernels/conv2d_3x3_nslice_order.hpp (RING_OWNER_MASK). A local
    output shard pins each N-slice to the column that holds it."""
    if not (g.k_split and b.nslice and g.remote_output):
        return None
    num_owning = _div_up(g.tensor_n_tiles, g.out_n_tiles)
    idle_n = g.k_cols - num_owning
    if idle_n <= 0:
        return None
    idle = {g.k_cols - 1 - (i * g.k_cols) // idle_n for i in range(idle_n)}
    return [c for c in range(g.k_cols) if c not in idle]


def _nslice_kh_masks(g: Conv3x3Geometry, bm, sb_h, m_tile_start, actual_m_tiles, num_m_blocks):
    """N-slice kernel-row skip (compute runtime args 3..): per M-block, ceil(groups / 10) words of 3-bit masks, ring row
    group gi at bits 3 * (gi % 10) of word gi // 10. Bit kh: kernel row kh is in the image for some output pixel of the
    group's valid rows (input row oy * stride + kh * d - p in [0, H)); the walk skips the others, whose in0 K tiles are
    all zeros. The groups are rows [gi * sb_h, ...) of the M-block for every owner (ring_group_row0 only reorders them).
    The walk runs a contiguous range: a group with no valid row takes kernel row 1, one with rows 0 and 2 but not 1 all
    three (both only add zeros), so every group runs a matmul.
    Measured (Blackhole p150b, LoFi, median of 3): aspp.convs.3 (d = 18, a third of every core's tile-MACs) 33729 ->
    27297 ns. Where the middle grid rows keep all three kernel rows (aspp.convs.1/2, res5.x, res4) the edge rows' walks
    shrink but the op time is flat."""
    groups = _div_up(bm, sb_h)
    words_per_block = _div_up(groups, 10)
    args = []
    for m in range(num_m_blocks):
        t0 = m_tile_start + m * bm
        valid = max(0, min(bm, actual_m_tiles - m * bm))
        o_end = min((t0 + valid) * TILE, g.num_out_px)
        words = [0] * words_per_block
        for gi in range(groups):
            mask = 0
            for r in range(gi * sb_h, min((gi + 1) * sb_h, bm)):
                o_a = (t0 + r) * TILE
                o_b = min(o_a + TILE, o_end)
                if o_a >= o_b:
                    continue
                for oy in range(o_a // g.output_width, (o_b - 1) // g.output_width + 1):
                    for kh in range(3):
                        if 0 <= oy * g.stride + kh * g.dilation - g.pad_top < g.input_height:
                            mask |= 1 << kh
            mask = mask or 0b010
            mask = 0b111 if mask == 0b101 else mask
            words[gi // 10] |= mask << (3 * (gi % 10))
        args += words
    return args


def m_split(g: Conv3x3Geometry, num_cores):
    """(core_m_tiles, num_m_groups): contiguous output tile-row ranges over `num_cores` cores. On a sharded
    input this is also the output shard: roundup32(ceil(M / n)) / 32 == ceil(ceil(M / 32) / n). With a balanced split
    (g.m_ranges): the largest range and the range count."""
    if g.m_ranges:
        return max(n for _, n in g.m_ranges), len(g.m_ranges)
    core_m_tiles = _div_up(g.tensor_m_tiles, num_cores)
    return core_m_tiles, _div_up(g.tensor_m_tiles, core_m_tiles)


def _core_m_ranges(tensor_m_tiles, core_m_tiles):
    return [(start, min(core_m_tiles, tensor_m_tiles - start)) for start in range(0, tensor_m_tiles, core_m_tiles)]


def core_m_ranges(g: Conv3x3Geometry, core_m_tiles):
    """(first output tile-row, tile-rows) per M work index: the balanced split's ranges, else the uniform split."""
    if g.m_ranges:
        return list(g.m_ranges)
    return _core_m_ranges(g.tensor_m_tiles, core_m_tiles)


def _block_band_len(g: Conv3x3Geometry, o_first, o_last):
    """Pixels one kernel row of the block touches: from tap (kh, 0) of o_first to tap (kh, band_last_tap) of
    o_last."""
    ho0, wo0 = divmod(o_first, g.output_width)
    ho1, wo1 = divmod(o_last, g.output_width)
    return (ho1 - ho0) * g.stride * g.input_width + (wo1 - wo0) * g.stride + g.band_last_tap * g.dilation + 1


def _block_first_tap(g: Conv3x3Geometry, o_first):
    """Flat input index of tap (0, 0) of output pixel o_first (negative inside the top/left padding)."""
    ho0, wo0 = divmod(o_first, g.output_width)
    return (ho0 * g.stride - g.pad_top) * g.input_width + wo0 * g.stride - g.pad_left


def _higher_shard_rows(g: Conv3x3Geometry, core_m_tiles, block_m_tiles, m_index, shard_rows):
    """The most slab tile-rows one M-block of work index m_index reads from a higher-index input shard (the split
    fill's writer share; mirror of the kernels' tile_slab_plan segments)."""
    start, actual = core_m_ranges(g, core_m_tiles)[m_index]
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


def _padded_slab_plan(g: Conv3x3Geometry, f, band_len):
    """(slots, rows) of one M-block's padded slab: mirror of conv2d_3x3::padded_slab::plan (image rows of the fill's
    tile-rows ranked in fill order, a segment starting in the row the previous one ended in sharing its slot)."""
    tpr = max(1, g.input_width // TILE)
    prev_t_hi, last_row, last_slot, rows = None, -1, -1, 0
    for kh in range(3):
        v_lo = f + kh * g.dilation * g.input_width
        lo, hi = max(v_lo, 0), min(v_lo + band_len, g.num_in_px)
        if lo >= hi:
            continue
        t_lo, t_hi = lo // TILE, _div_up(hi, TILE)
        first_new = prev_t_hi if prev_t_hi is not None and t_lo < prev_t_hi else t_lo
        if t_hi > first_new:
            r0, r1 = first_new // tpr, (t_hi - 1) // tpr
            s0 = last_slot if last_row >= 0 and r0 == last_row else last_slot + 1
            rows += t_hi - first_new
            last_row, last_slot = r1, s0 + (r1 - r0)
        prev_t_hi = t_hi
    return last_slot + 1, rows


def padded_slab_capable(g: Conv3x3Geometry):
    """Geometry side of PADDED_SLAB: every tap block of a tile-row is one run of a slab whose rows sit d pixels apart
    (stride 1, p == d, whole-tile image rows)."""
    return g.tile_input and g.stride == 1 and g.padding == g.dilation and g.input_width % TILE == 0


def padded_slab_extent(g: Conv3x3Geometry, core_m_tiles, block_m_tiles):
    """(slots_max, slab pixels) over every M-block of every core: slots_max + 1 gaps of d pixels around slots_max rows
    of W pixels (the fill's unread tail rows are not untilized)."""
    slots_max = 0
    for start, actual in core_m_ranges(g, core_m_tiles):
        for block_start in range(0, actual, block_m_tiles):
            tiles = min(block_m_tiles, actual - block_start)
            o_first = (start + block_start) * TILE
            o_last = min((start + block_start + tiles) * TILE, g.num_out_px) - 1
            slots, _ = _padded_slab_plan(g, _block_first_tap(g, o_first), _block_band_len(g, o_first, o_last))
            slots_max = max(slots_max, slots)
    return slots_max, slots_max * (g.input_width + g.dilation) + g.dilation


@functools.lru_cache(maxsize=1024)
def _block_extents(g: Conv3x3Geometry, core_m_tiles, block_m_tiles):
    """(max band_len_px, max TILE slab tile-rows) over every M-block of every core."""
    worst_band, worst_slab_rows = 0, 0
    for start, actual in core_m_ranges(g, core_m_tiles):
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
    """One untilize batch = the tile-rows one band can touch, so the reader fences about once per band; under a K
    split at most KSPLIT_UNTILIZE_BATCH_ROWS (see there)."""
    if not g.tile_input:
        return 1
    ub = max(1, min(raw_slab_rows, _div_up(band_len, TILE) + 1))
    return ub if not g.k_split else min(ub, KSPLIT_UNTILIZE_BATCH_ROWS)


def face_direct_im2col(g: Conv3x3Geometry, bk):
    """Face-direct gather: needs whole kernel rows per K-block, so every K-block has one column layout. Under a
    K split the matmul's L1-accumulating packs to the caller-owned partials are ordered only by the tilize's own
    pack between K-blocks, so without a tilize the K walk must be one K-block."""
    return g.face_direct_capable and bk % g.tensor_kkh_tiles == 0 and (not g.k_split or bk == g.tensor_k_tiles)


def slab_direct_capable(g: Conv3x3Geometry, bk):
    """Slab-direct (see SLAB_DIRECT): face-direct with one K-block (no K split), every tile-row inside one output
    row."""
    return (
        SLAB_DIRECT
        and face_direct_im2col(g, bk)
        and bk == g.tensor_k_tiles
        and not g.k_split
        and g.output_width % TILE == 0
    )


def slab_direct_band_last_tap(g: Conv3x3Geometry):
    """The fill also lands the pixel the K-padding run of the last K-tile reads (C_in 16: tap 3)."""
    return 3


def slab_direct_layout(g: Conv3x3Geometry, core_m_tiles, block_m_tiles):
    """(gap, pitch, zone, slots_max, slab pixels): slot r = image row ih_lo + r behind `gap` zero pixels, `pitch`
    pixels per slot, and one more gap after the last slot. The gap covers both the left padding of its row and what
    the last im2col runs of the previous row read past that row's end (its right padding, and the K-padding run's
    pixels against zero weight rows), so no gap pixel is an image pixel: zone == gap."""
    over = g.stride * (g.output_width - 1) - g.pad_left + 3 * g.dilation - (g.input_width - 1)
    gap = max(g.pad_left, over, 0)
    pitch = g.input_width + gap
    zone = gap
    slots_max = 0
    for start, actual in core_m_ranges(g, core_m_tiles):
        for block_start in range(0, actual, block_m_tiles):
            tiles = min(block_m_tiles, actual - block_start)
            o_first = (start + block_start) * TILE
            o_last = min((start + block_start + tiles) * TILE, g.num_out_px) - 1
            ho0, ho1 = o_first // g.output_width, o_last // g.output_width
            slots_max = max(slots_max, g.stride * (ho1 - ho0) + 2 * g.dilation + 1)
    return gap, pitch, zone, slots_max, slots_max * pitch + zone


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
        return g.pad_left >= 1 and 2 * (g.output_width - 1) + 2 - g.pad_left >= g.input_width
    return True


def s2_taps_capable(g: Conv3x3Geometry):
    """Geometry side of S2_TAPS: the stride-2 window's taps of an output tile are the even / odd pixels of two whole
    input tiles of one image row (W % 64 == 0) plus the previous tile's last pixel (padding 1, dilation 1)."""
    return (
        S2_TAPS
        and strided_window_capable(g)
        and g.tile_input
        and g.padding == 1
        and g.input_width % (2 * TILE) == 0
        and g.output_width * 2 == g.input_width
    )


def _strided_window(g: Conv3x3Geometry, bk):
    """One K-block per kernel row: a unit (tile-row x K-block) is one kernel row's window run."""
    return strided_window_capable(g) and bk == g.tensor_kkh_tiles and not face_direct_im2col(g, bk)


CENTER_NONE, CENTER_K_BLOCK, CENTER_SLOTS = 0, 1, 2


def _tap_direct(g: Conv3x3Geometry, bm, bk, nslice):
    """Tap-direct: compute tilizes each tap block of a unit from an L1 address the reader publishes (a slab run when the
    tile-row's pixels are one output-row run inside the image, the zero block for a kernel row outside it, else a block
    the reader gathers as before). Saves the reader's copy of most tap blocks, which paced the N-slice prelude and the
    K-blocked walk's tilize (HEIGHT stem / res2 / center-head layers). Needs the tap-major gather. N-slice prelude: one
    entry per cb_im2col_rm slot; K-blocked walk: an entry FIFO (_tap_direct_fifo_units)."""
    return (
        _tap_major(g, bk) and not face_direct_im2col(g, bk) and not _row_split(g, bm, bk) and not _strided_window(g, bk)
    )


def _src_table_bytes(ring_units):
    return _div_up(ring_units * 16, 64) * 64


def _tap_direct_fifo_units(g: Conv3x3Geometry, bm, bk, nslice):
    """K-blocked tap-direct: entries of the unit FIFO, one K-block (so the reader runs a whole K-block ahead of the tilize
    and, under the center-tap slots, no further). cb_im2col_rm then holds only gathered tap blocks. 0 on the N-slice
    walk or without tap-direct."""
    return bm if not nslice and _tap_direct(g, bm, bk, nslice) else 0


def _tap_direct_windows(g: Conv3x3Geometry, bm, bk, nslice, center_mode):
    """K-blocked tap-direct window units: with d = 1, three taps per unit (one kernel row per K-block) and no center
    slots, a unit whose taps are adjacent slab runs is one strided window tilized c_in_tiles-strided, 3 * c_in_tiles
    wide (stem.conv2 @ LoFi 52.5 -> 47.6 us). One-tile taps keep the per-tile unpack tilize."""
    return bool(
        _tap_direct_fifo_units(g, bm, bk, nslice)
        and g.dilation == 1
        and bk == 3 * g.c_in_tiles
        and g.c_in_tiles > 1
        and center_mode != CENTER_SLOTS
    )


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
        + (RING_TOKEN_PAGES * RING_TOKEN_BYTES + TILE_BYTES_FP32 if nslice else bm * bn * TILE_BYTES_FP32)
        + combine_bytes
        + (
            bm * ns * g.out_tile_bytes
            if g.remote_output
            else (0 if g.sharded_output else OUT_DEPTH_ROW_GROUPS * sb_h * bn * g.out_tile_bytes)
        )
        + (
            (_tap_direct_fifo_units(g, bm, bk, nslice) * 16 or _src_table_bytes(im2col_rm_depth_rows))
            if _tap_direct(g, bm, bk, nslice)
            else 0
        )
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
    for start, actual in core_m_ranges(g, core_m_tiles):
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


def _group_ktile_cycles(rows, sb_w, ct_rows):
    """Walk cycles per k-tile of one ring row group of `rows` rows (BALANCE_WALK_GROUP_KTILE: r x 1 subblocks; other
    shapes scale with the unpacked tiles, rt + rt * ct per k-step, ct + ct * rt when rt > ct)."""
    if sb_w == 1 and rows in BALANCE_WALK_GROUP_KTILE:
        return (BALANCE_WALK_GROUP_KTILE_CT if ct_rows else BALANCE_WALK_GROUP_KTILE)[rows]
    unpack = sb_w + sb_w * rows if rows > sb_w else rows + rows * sb_w
    return BALANCE_WALK_GROUP_KTILE[4] * unpack / 5


def _balance_cost_table(g: Conv3x3Geometry, cap, sb_h, sb_w, ct_rows):
    """cost[a][n - 1] (cycles, n = 1..cap): the row cost model (BALANCE_*) of output tile-rows [a, a + n) on the N-slice
    ring walk: untilize of the range's slab rows, gather + tilize of its tile-rows, and its walk over ring row groups of
    sb_h valid rows, each multiplying its contiguous in-image kernel-row range (as _nslice_kh_masks)."""
    M = g.tensor_m_tiles
    tile_mask = []
    for t in range(M):
        o_a, o_b = t * TILE, min(t * TILE + TILE, g.num_out_px)
        m = 0
        for oy in range(o_a // g.output_width, (o_b - 1) // g.output_width + 1):
            for kh in range(3):
                if 0 <= oy * g.stride + kh * g.dilation - g.pad_top < g.input_height:
                    m |= 1 << kh
        tile_mask.append(m)
    ns = g.out_n_tiles
    steps = _div_up(g.tensor_n_tiles, ns)
    kkh = g.tensor_kkh_tiles
    in0_per_row = g.tensor_k_tiles
    if g.stride == 1 and g.padding == g.dilation and g.input_width % TILE == 0:
        in0_per_row -= 3 * g.c_in_tiles  # N-slice center taps: copied, not tilized
    tilize_row = 3 * BALANCE_TILIZE_UNIT + BALANCE_TILIZE_TILE * in0_per_row

    def group_cost(rows, mask):
        mask = mask or 0b010
        mask = 0b111 if mask == 0b101 else mask
        return bin(mask).count("1") * kkh * (ns // sb_w) * _group_ktile_cycles(rows, sb_w, ct_rows)

    table = []
    for a in range(M):
        row = []
        done = 0  # cost of the complete groups
        cur_rows, cur_mask = 0, 0
        for n in range(1, min(cap, M - a) + 1):
            cur_rows += 1
            cur_mask |= tile_mask[a + n - 1]
            groups = _div_up(n, sb_h)
            walk = steps * (done + group_cost(cur_rows, cur_mask) + groups * BALANCE_WALK_GROUP)
            if cur_rows == sb_h:
                done += group_cost(cur_rows, cur_mask)
                cur_rows, cur_mask = 0, 0
            o_first = a * TILE
            o_last = min((a + n) * TILE, g.num_out_px) - 1
            slab = (
                _tile_slab_rows(g, _block_first_tap(g, o_first), _block_band_len(g, o_first, o_last))
                if g.tile_input
                else 0
            )
            row.append(BALANCE_UNTILIZE_TILE * g.c_in_tiles * slab + tilize_row * n + walk)
        table.append(row)
    return table


@functools.lru_cache(maxsize=256)
def balanced_m_ranges(g: Conv3x3Geometry, rows, cap, ct_rows):
    """(cost, ranges): the contiguous split of the output tile-rows over at most `rows` grid rows, each at most `cap`
    tile-rows, that minimizes the costliest row of _balance_cost_table (exact DP); ties keep fewer tile-rows on the
    costliest rows. None when no split fits."""
    M = g.tensor_m_tiles
    sb_h, sb_w = _ring_subblock(cap, g.out_n_tiles)
    table = _balance_cost_table(g, cap, sb_h, sb_w, ct_rows)
    inf = float("inf")
    # best[r][a]: the least costliest row over splits of [a, M) into at most r rows.
    best = [[inf] * (M + 1) for _ in range(rows + 1)]
    take = [[0] * (M + 1) for _ in range(rows + 1)]
    for r in range(rows + 1):
        best[r][M] = 0.0
    for r in range(1, rows + 1):
        for a in range(M - 1, -1, -1):
            costs = table[a]
            for n in range(1, len(costs) + 1):
                v = max(costs[n - 1], best[r - 1][a + n])
                if v < best[r][a]:
                    best[r][a], take[r][a] = v, n
    if best[rows][0] == inf:
        return None
    ranges, a, r = [], 0, rows
    while a < M:
        n = take[r][a]
        ranges.append((a, n))
        a, r = a + n, r - 1
    return best[rows][0], tuple(ranges)


@functools.lru_cache(maxsize=256)
def uniform_range_cost(g: Conv3x3Geometry, core_m_tiles, ct_rows):
    """The row cost model's costliest row of the uniform split."""
    sb_h, sb_w = _ring_subblock(core_m_tiles, g.out_n_tiles)
    table = _balance_cost_table(g, core_m_tiles, sb_h, sb_w, ct_rows)
    return max(table[a][n - 1] for a, n in _core_m_ranges(g.tensor_m_tiles, core_m_tiles))


@functools.lru_cache(maxsize=256)
def balanced_block_ranges(g: Conv3x3Geometry, rows, shard_ranges, l1_budget):
    """K split on the N-slice walk with one M-block per core: (ranges, remote_output) of the work-balanced M split over
    at most `rows` grid rows (balanced_m_ranges), at the uniform split's block height or up to BALANCE_MAX_EXTRA_ROWS
    above it where that beats the smaller heights by more than BALANCE_CAP_TIE, and whose N-slice blocking fits L1 with
    one M-block per core; remote_output unless the ranges are the output's shard rows. None where the uniform split
    stays: the model's costliest row drops by less than BALANCE_MIN_GAIN, or no split fits."""
    uniform_core_m, _ = m_split(g, rows)
    try:
        base = select_blocking(g, (rows, 1), l1_budget)
    except ValueError:
        return None
    if not (base.nslice and base.max_m_blocks_per_core == 1 and base.block_m_tiles == uniform_core_m):
        return None
    # The uniform split keeps runtime group heights; the balanced one runs them as template constants.
    base_cost = uniform_range_cost(g, uniform_core_m, False)
    found = []
    for cap in range(uniform_core_m, uniform_core_m + BALANCE_MAX_EXTRA_ROWS + 1):
        split = balanced_m_ranges(g, rows, cap, _ring_subblock(cap, g.out_n_tiles) == (4, 1))
        if split is not None and split[0] <= base_cost * (1 - BALANCE_MIN_GAIN):
            found.append(split)
    # Cheapest first; a larger block height only where it beats the smaller ones by more than BALANCE_CAP_TIE.
    found.sort(key=lambda split: split[0])
    if found:
        best = found[0][0]
        found.sort(key=lambda split: (split[0] > best * (1 + BALANCE_CAP_TIE), max(n for _, n in split[1]), split[0]))
    for cost, ranges in found:
        remote = tuple(ranges) != tuple(shard_ranges)
        g2 = replace(g, m_ranges=tuple(ranges), remote_output=remote)
        bm = max(n for _, n in ranges)
        try:
            b = select_blocking(g2, (len(ranges), 1), l1_budget)
        except ValueError:
            continue
        if b.nslice and b.max_m_blocks_per_core == 1 and b.block_m_tiles == bm:
            return tuple(ranges), remote
    return None


def block_compute_rows(shard_spec, device_grid, tensor_m_tiles):
    """Grid rows the K split's M split runs down: the fewest rows, from the shard grid's first row to the device edge
    along its M axis (y on a ROW_MAJOR grid, x on COL_MAJOR), that reach the fewest output tile-rows per row; the shard
    rows unless more rows lower that. Every row of the K split is k_cols cores, one per channel slice; rows past the
    shard grid read their whole slab from the column's owner shards, like any halo. The walk, untilize and tilize are
    M-proportional (aspp.convs @ LoFi: 64 tile-rows on 10 rows of 7 instead of 8 rows of 8, -7 to -9%)."""
    rows, _ = block_grid_shape(shard_spec)
    start = shard_spec.grid.bounding_box().start
    if shard_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR:
        max_rows = device_grid.y - start.y
    else:
        max_rows = device_grid.x - start.x
    best = rows
    for r in range(rows + 1, max_rows + 1):
        if _div_up(tensor_m_tiles, r) < _div_up(tensor_m_tiles, best):
            best = r
    return best


def block_core(shard_spec, row, col):
    """Core of BLOCK grid position (row, col), rows past the shard grid included: a ROW_MAJOR grid runs the rows down y
    (shard (h, w) on (x0 + w, y0 + h)), COL_MAJOR down x."""
    start = shard_spec.grid.bounding_box().start
    if shard_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR:
        return ttnn.CoreCoord(start.x + col, start.y + row)
    return ttnn.CoreCoord(start.x + row, start.y + col)


@functools.lru_cache(maxsize=64)
def _round0_reader_rows(banks, sender_rows, num_rows, spare, relocated):
    """Grid row of each column's round-0 reader (NSLICE_ROUND0_MODEL), from a Blackhole NoC1 path model: a DRAM read
    response climbs north (y decreasing, mod 12) along its bank's DRAM column from the bank's NoC1 endpoint to the
    reader's row, then runs west along that row. Readers on distinct rows share only the climbs; a read slows with the
    number of reads sharing its busiest climb link (res5.0, 39 KB rounds: 0.8 us alone, 1.1 us shared by two, 1.5-2.4
    us by five to eight). Minimizes, over distinct rows: the worst column's sharing (plus one for a reader off the
    sender's row where spare members wait for the sender's relay), then the sum, the climb hops, the readers moved."""
    ep_y = _ROUND0_NOC1_EP_Y[relocated]
    n = len(banks)
    rows = range(num_rows)

    def climb(bank, row):
        y, side, links = ep_y[bank], bank // 4, []
        while y != row + 2:
            links.append((side, y))
            y = (y - 1) % 12
        return tuple(links)

    climbs = {(k, r): climb(banks[k], r) for k in range(n) for r in rows}

    def score(assign):
        use = {}
        for k, r in assign.items():
            for link in climbs[(k, r)]:
                use[link] = use.get(link, 0) + 1
        per = [
            max((use[link] for link in climbs[(k, r)]), default=1) + (1 if spare and r != sender_rows[k] else 0)
            for k, r in assign.items()
        ]
        return (
            max(per),
            sum(per),
            sum(len(climbs[(k, r)]) for k, r in assign.items()),
            sum(1 for k, r in assign.items() if r != sender_rows[k]),
        )

    best = [None, None]
    order = sorted(range(n), key=lambda k: min(len(climbs[(k, r)]) for r in rows))

    def search(i, assign, used):
        if best[0] is not None and assign and score(assign)[0] > best[0][0]:
            return
        if i == n:
            s = score(assign)
            if best[0] is None or s < best[0]:
                best[0], best[1] = s, dict(assign)
            return
        k = order[i]
        for r in sorted(rows, key=lambda r: (len(climbs[(k, r)]), r != sender_rows[k])):
            if r not in used:
                assign[k] = r
                used.add(r)
                search(i + 1, assign, used)
                used.discard(r)
                del assign[k]

    search(0, {}, set())
    return tuple(best[1][k] for k in range(n))


def _round0_args(input_tensor, weight_tensor, g, b, work, m_blocks_per_index, spare_groups, column_channel, spare):
    """NSLICE_ROUND0 writer args per core (x, y): [round-0 reader, column rectangle (NoC1-ordered), column dests,
    spare rectangle, spare dests] (zeros where absent): the reader's column fields, the column sender's spare ones."""
    device = input_tensor.device()
    _, rows_n, _ = column_channel
    most = max(m_blocks_per_index[i] for _, i, _ in work)
    grid_rows = sorted({i for _, i, _ in work if i not in spare_groups and m_blocks_per_index[i] == most})
    spare_rows = {i for _, i, _ in work if i in spare_groups and m_blocks_per_index[i] == most}
    k_cols = max(c for _, _, c in work) + 1

    def rect(cores):
        lo = device.worker_core_from_logical_core(ttnn.CoreCoord(min(c.x for c in cores), min(c.y for c in cores)))
        hi = device.worker_core_from_logical_core(ttnn.CoreCoord(max(c.x for c in cores), max(c.y for c in cores)))
        return [hi.x, hi.y, lo.x, lo.y]  # the writer's NoC1 order

    # The staggered column sender of column c is participant c % rows (_column_weight_mcast).
    reader_rows = [c % rows_n for c in range(k_cols)]
    wspec = weight_tensor.memory_config().nd_shard_spec
    round_bytes = b.block_k_tiles * b.num_k_blocks * g.out_n_tiles * g.weight_tile_bytes
    shard_rows = {i: core.y for core, i, c in work if c == 0}
    if (
        NSLICE_ROUND0_MODEL
        and device.arch() == ttnn._ttnn.device.Arch.BLACKHOLE
        and device.dram_grid_size().x == 8
        and input_tensor.memory_config().shard_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
        and wspec is not None
        and list(wspec.shard_shape)[-2:] == [b.block_k_tiles * b.num_k_blocks * TILE, g.out_n_tiles * TILE]
        and round_bytes >= NSLICE_ROUND0_MODEL_MIN_BYTES
        and k_cols <= rows_n
        and all(shard_rows[grid_rows[r]] == r for r in range(rows_n))
    ):
        # Bank 0's NoC0 endpoint as a virtual coordinate (y = 12 + 3 x row-in-column + endpoint index): its endpoint
        # index tells the relocated endpoint set from the original one.
        relocated = {14: False, 12: True}.get(device.dram_core_from_logical_core(ttnn.CoreCoord(0, 0)).y)
        if relocated is not None:
            owners = ring_owner_cols(g, b)
            num_owning = _div_up(g.tensor_n_tiles, g.out_n_tiles)
            banks = []
            for c in range(k_cols):
                # The column's first owner slice (conv2d_3x3_nslice_order.hpp), in ND shard (c, slice): bank round-robin.
                for j in range(k_cols):
                    o = (c + 2 * k_cols - 1 - j) % k_cols
                    if (o in owners) if owners is not None else (o < num_owning):
                        banks.append((c * num_owning + (owners.index(o) if owners is not None else o)) % 8)
                        break
            if len(banks) == k_cols:
                reader_rows = list(
                    _round0_reader_rows(tuple(banks), tuple(reader_rows), rows_n, spare is not None, relocated)
                )
    args = {}
    for c in range(k_cols):
        column = [core for core, i, cc in work if i in grid_rows and cc == c]
        reader = next(core for core, i, cc in work if i == grid_rows[reader_rows[c]] and cc == c)
        sender = next(core for core, i, cc in work if i == grid_rows[c % rows_n] and cc == c)
        args.setdefault((reader.x, reader.y), [0] * 11)[0:6] = [1] + rect(column) + [len(column) - 1]
        members = [core for core, i, cc in work if i in spare_rows and cc == c]
        if spare is not None and members:
            args.setdefault((sender.x, sender.y), [0] * 11)[6:11] = rect(members) + [len(members)]
    return args


def _spread_args(input_tensor, work, m_blocks_per_index, spare_groups, column_channel, spare, round0_args, rounds):
    """NSLICE_SPREAD writer args per core (x, y), in the NSLICE_ROUND0 args' place: [rounds this core reads (bit mask),
    the column's multicast rectangle (NoC1-ordered), column dests, its slice's spare members' rectangle, spare dests,
    rounds this core relays to them (bit mask)] (zeros where absent). Round 0's reader is the NSLICE_ROUND0 reader,
    round 1's the column sender, round r >= 2's the column's other rows in turn; the column sender relays every
    round."""
    device = input_tensor.device()
    _, rows_n, _ = column_channel
    most = max(m_blocks_per_index[i] for _, i, _ in work)
    grid_rows = sorted({i for _, i, _ in work if i not in spare_groups and m_blocks_per_index[i] == most})
    spare_rows = {i for _, i, _ in work if i in spare_groups and m_blocks_per_index[i] == most}
    k_cols = max(c for _, _, c in work) + 1

    def rect(cores):
        lo = device.worker_core_from_logical_core(ttnn.CoreCoord(min(c.x for c in cores), min(c.y for c in cores)))
        hi = device.worker_core_from_logical_core(ttnn.CoreCoord(max(c.x for c in cores), max(c.y for c in cores)))
        return [hi.x, hi.y, lo.x, lo.y]  # the writer's NoC1 order

    args = {}
    for c in range(k_cols):
        column = [next(core for core, i, cc in work if i == row and cc == c) for row in grid_rows]
        r0 = next(p for p, core in enumerate(column) if round0_args.get((core.x, core.y), [0])[0] == 1)
        # The staggered column sender of column c is participant c % rows (_column_weight_mcast).
        sender = c % rows_n
        others = [p for p in ((r0 + k) % rows_n for k in range(1, rows_n)) if p != sender] or [sender]
        readers = [r0, sender] + [others[(r - 2) % len(others)] for r in range(2, rounds)]
        for p, core in enumerate(column):
            mask = sum(1 << r for r in range(rounds) if readers[r] == p)
            if mask:
                args.setdefault((core.x, core.y), [0] * 12)[0:6] = [mask] + rect(column) + [len(column) - 1]
        members = [core for core, i, cc in work if i in spare_rows and cc == c]
        if spare is not None and members:
            relay = column[sender]
            args.setdefault((relay.x, relay.y), [0] * 12)[6:12] = rect(members) + [len(members), (1 << rounds) - 1]
    return args


def _block_device_extents(shard_spec, device_grid):
    """(rows, columns) of the device from the shard grid's first core, along its M axis and its channel-slice axis."""
    start = shard_spec.grid.bounding_box().start
    if shard_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR:
        return device_grid.y - start.y, device_grid.x - start.x
    return device_grid.x - start.x, device_grid.y - start.y


def block_spare_groups(shard_spec, device_grid, k_cols, rows, tensor_m_tiles):
    """Spare-column M-groups (SPARE_MGROUPS) on top of `rows` grid-row groups: the fewest that reach the fewest output
    tile-rows per group. One group per spare column (its k_cols members on the column's first k_cols rows), so the
    members of channel slice k across the spare groups are one row of the spare columns (one multicast rectangle)."""
    max_rows, max_cols = _block_device_extents(shard_spec, device_grid)
    if not SPARE_MGROUPS or k_cols < 2 or max_rows < k_cols:
        return 0
    best = 0
    for e in range(1, max_cols - k_cols + 1):
        if _div_up(tensor_m_tiles, rows + e) < _div_up(tensor_m_tiles, rows + best):
            best = e
    return best


@functools.lru_cache(maxsize=256)
def spare_m_indices(g: Conv3x3Geometry, core_m_tiles, num_groups):
    """M work indices placed on the spare columns (g.block_rows grid-row groups, the rest spare): the cheapest ranges by
    the row cost model (_balance_cost_table). The spare groups' slab fill comes from farther (all 8 members' reads share
    one column's links) and their weight rounds land one forward later, so they take the least work: res5.2 29.25 ->
    27.83 us, res5.0 / res5.1 -0.7 / -0.9 us (zoned, one run each) over the last ranges."""
    if not g.block_rows or num_groups <= g.block_rows:
        return ()
    spare = num_groups - g.block_rows
    ranges = core_m_ranges(g, core_m_tiles)[:num_groups]
    cap = max(n for _, n in ranges)
    sb_h, sb_w = _ring_subblock(cap, g.out_n_tiles)
    table = _balance_cost_table(g, cap, sb_h, sb_w, True)
    cost = [table[a][n - 1] for a, n in ranges]
    cheapest = sorted(range(len(ranges)), key=lambda i: (cost[i], -i))[:spare]
    return tuple(sorted(cheapest))


def block_spare_core(shard_spec, k_cols, extra, col):
    """Core of channel slice `col` of spare-column group `extra`: spare column k_cols + extra of the shard grid's
    first core, row `col` (ROW_MAJOR; transposed on COL_MAJOR)."""
    start = shard_spec.grid.bounding_box().start
    if shard_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR:
        return ttnn.CoreCoord(start.x + k_cols + extra, start.y + col)
    return ttnn.CoreCoord(start.x + col, start.y + k_cols + extra)


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


def _weight_shard_tiles(weight_tensor, slot_rows_whole):
    """The server reader's page runs: (shard rows, shard columns) in tiles of ND-sharded weights
    (conv2d_3x3.weight_memory_config), or (0, 0), one read per tile, when the weights are DRAM interleaved or a slot row
    is not a whole prepared tile-row (K-blocked rounds of fewer than all C_out columns)."""
    spec = weight_tensor.memory_config().nd_shard_spec
    if spec is None or not slot_rows_whole:
        return (0, 0)
    shape = list(spec.shard_shape)
    return (shape[-2] // TILE, shape[-1] // TILE)


def _weight_servers(input_tensor, core_list, receivers, partners):
    """One spare core per rectangle of the weight channel (closest to the rectangle's NoC1 multicast start, its high
    corner), as [(rectangle, server core, partner core or None)], or None when the device grid has too few spare cores.
    A spare core holds neither work nor an input shard: the fill reads every input shard, and a server's traffic on a
    shard's core slowed those reads (stem.conv1, input on 110 cores and work on 108: 22.7 -> 24.1 us with servers on the
    two shard cores). With partners (WEIGHT_SERVER_PARTNERS) every rectangle also gets a partner server (multicasting
    on NoC0) while spare cores remain, largest rectangle first; both roles are then placed by NoC hops (see
    WEIGHT_SERVER_PARTNERS)."""
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    used = {(c.x, c.y) for c in core_list} | {(c.x, c.y) for c in shard_cores(input_tensor)}
    spare = [(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in used]
    rects = list(receivers.ranges())
    if len(spare) < len(rects):
        return None
    if not partners or len(spare) <= len(rects):
        out = []
        for rect in rects:
            hx, hy = rect.end.x, rect.end.y
            best = min(spare, key=lambda c: (abs(c[0] - hx) + abs(c[1] - hy), c[1], c[0]))
            spare.remove(best)
            out.append((rect, ttnn.CoreCoord(*best), None))
        return out

    def virtual(c):
        v = device.worker_core_from_logical_core(ttnn.CoreCoord(*c))
        return v.x, v.y

    W, H = _NOC_GRID
    receiving = {
        virtual((x, y))
        for rect in rects
        for x in range(rect.start.x, rect.end.x + 1)
        for y in range(rect.start.y, rect.end.y + 1)
    }

    def path_cost(src, dst, noc):
        """NoC hops from src to dst (NoC0: +x then +y; NoC1: -y then -x; torus), +100 per receiver core passed."""
        (sx, sy), (tx, ty) = src, dst
        if noc == 0:
            dx, dy = (tx - sx) % W, (ty - sy) % H
            nodes = [((sx + i) % W, sy) for i in range(1, dx + 1)] + [(tx, (sy + j) % H) for j in range(1, dy + 1)]
        else:
            dx, dy = (sx - tx) % W, (sy - ty) % H
            nodes = [(sx, (sy - j) % H) for j in range(1, dy + 1)] + [((sx - i) % W, ty) for i in range(1, dx + 1)]
        return dx + dy + 100 * sum(1 for n in nodes[:-1] if n in receiving)

    size = lambda r: r.grid_size().x * r.grid_size().y
    order = sorted(range(len(rects)), key=lambda i: -size(rects[i]))
    partner_of = {}
    for i in order:
        if len(spare) <= len(rects):
            break
        low = virtual((rects[i].start.x, rects[i].start.y))
        best = min(spare, key=lambda c: (path_cost(virtual(c), low, 0), c[1], c[0]))
        spare.remove(best)
        partner_of[i] = ttnn.CoreCoord(*best)
    server_of = {}
    for i in order:
        high = virtual((rects[i].end.x, rects[i].end.y))
        best = min(spare, key=lambda c: (path_cost(virtual(c), high, 1), c[1], c[0]))
        spare.remove(best)
        server_of[i] = ttnn.CoreCoord(*best)
    return [(rects[i], server_of[i], partner_of.get(i)) for i in range(len(rects))]


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


def _spare_weight_mcast(input_tensor, work, m_blocks_per_index, spare, column_channel):
    """Spare-column groups (SPARE_MGROUPS): the second multicast of each column weight stream. Group k is the spare
    members of channel slice k that run the column's rounds (the most M-blocks), one row of the spare columns
    (ROW_MAJOR; a column on COL_MAJOR); its explicit sender is the column participant after column k's staggered sender
    (participant (k + 1) % rows of the column channel, _column_weight_mcast), a pure receiver that forwards each round
    once it landed. (receivers, group size, receiver order, senders), or None."""
    if column_channel is None or not spare:
        return None
    _, rows, _ = column_channel
    most = max(m_blocks_per_index[i] for _, i, _ in work)
    groups = sorted({i for _, i, _ in work if i in spare and m_blocks_per_index[i] == most})
    if not groups:
        return None
    k_cols = max(col for _, _, col in work) + 1
    by_column = input_tensor.memory_config().shard_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
    members = [core for core, i, _ in work if i in groups]
    receivers = _core_range_set(members)
    if len(receivers.ranges()) != 1:
        return None
    # Slice k's members share a y (ROW_MAJOR): RowMajor order makes each group one row, slices in order.
    order = ttnn.McastCoreOrder.RowMajor if by_column else ttnn.McastCoreOrder.ColumnMajor
    grid_rows = sorted({i for _, i, _ in work if i not in spare and m_blocks_per_index[i] == most})
    shift = 1 if rows > 1 else 0
    senders = []
    for col in range(k_cols):
        row = grid_rows[(col + shift) % rows]
        senders.append(next(core for core, i, c in work if i == row and c == col))
    return receivers, len(groups), order, senders


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
    spec = input_tensor.memory_config().shard_spec
    spare = spare_m_indices(g, b.core_m_tiles, b.num_m_groups)
    grid_row = {i: r for r, i in enumerate(i for i in range(b.num_m_groups) if i not in spare)}
    spare_col = {i: e for e, i in enumerate(spare)}

    def core(i, col):
        if i in grid_row:
            return block_core(spec, grid_row[i], col)
        return block_spare_core(spec, g.k_cols, spare_col[i], col)

    return [(core(i, col), i, col) for i in range(b.num_m_groups) for col in range(g.k_cols)]


# kw-fold + output shift (kernels/conv2d_3x3_kw_fold.hpp): on HEIGHT-sharded TILE input at stride 1, padding ==
# dilation, W % 32 == 0 and whole-tile C_in, the kernel rows are whole input tile-rows apart, so [Q_0 | Q_1 | Q_2] is
# one matmul on the input's own tiles (K' = 3 C_in, N' = 3 C_out) and the kw shift moves to the output (fp32 packer
# L1 accumulation at a +-d row offset). No untilize, gather or tilize. Measured against the K-blocked tap-direct walk
# (Blackhole p150b, LoFi bf8_b, perf focus): stem.conv2 47.9 -> 36.4, stem.conv3 69.8 -> 51.7, center_head.1 22.1 ->
# 12.1, inst res2.fuse_conv.0 44.4 -> 33.3, inst res2.fuse_conv.1 34.7 -> 28.1, inst res3.fuse_conv.0 29.0 -> 25.6,
# res2 16.7 -> 13.7 us. Where it does not pay (_kw_fold_pays) the K-blocked walk runs.
KW_FOLD = True
# kw-fold's final pass adds the bias as srcB rows 0 and 1 (row-broadcast): the fp32 bias row [b_0 .. b_31] of one output
# column tile.
KWF_BIAS_ROWS_BYTES = TILE * 4
# The final pass reaches a tile's right half-plane in at most two srcA Z steps of the unpacker's 16-bit x_dim (fp32
# datums): an ACC plane of at most 8191 rows (1 MB, more than a core's L1 holds with the rest of kw-fold's CBs).
KWF_MAX_PLANE_BYTES = 8191 * TILE * 4
IM2COL_NS_PER_TILE = 25
# With more than one M-block per core either form recomputes the two halo tile-rows' products, about 6 * Ct * N extra
# tile-MACs per M-block, at the user's fidelity, to save the im2col (tilized taps plus the untilized slab, ~25 ns per
# tile). The weights stream in column rounds, so each Q tile-row is packed once and the first matmul starts when the
# first round lands; what decides is the halo against the im2col, of which the tap-direct walk hides part (its prelude
# untilize and first tilize) under the first weight round both walks wait for. Measured (Blackhole p150b, kw-fold vs the
# tap-direct walk, halo / im2col in brackets): wins up to 0.52 (HiFi4 inst res2 fc1 0.52: 86.8 -> 83.9 us; LoFi inst dec
# res3 fc0 0.37: 29.0 -> 25.6, res2 0.07: 16.7 -> 13.7; LoFi res3.x 0.37 flat), losses from 0.57 (LoFi sem dec res3 fc0
# 0.57: +3.4%, fc1 0.73: +6.1%; HiFi4 res3 1.47: +5.7%, inst dec res3 fc0 1.47: +9.6%). With one M-block per core the
# neighbour cores exchange the boundary rows instead (kw_fold_halo_exchange): no halo cost, so kw-fold is taken wherever
# it fits. That flips sem dec res3 fc0 / fc1 onto kw-fold (LoFi 44.91 -> 40.97 / 36.22 -> 33.01 us; HiFi4 -2.8% / -2.5%)
# and the HiFi4 res3-scale layers (res3 -12.4%, inst dec res3 fc0 -8.2%).
KW_FOLD_MAX_HALO_SHARE = 0.55
LOFI_NS_PER_TILE_MAC = 16 / 1.35
FIDELITY_PHASES = {
    ttnn.MathFidelity.LoFi: 1,
    ttnn.MathFidelity.HiFi2: 2,
    ttnn.MathFidelity.HiFi3: 3,
    ttnn.MathFidelity.HiFi4: 4,
}


def kw_fold_mode(g, b, has_bias, l1_budget, math_fidelity):
    return kw_fold_plan(g, b, has_bias, l1_budget, math_fidelity) is not None


def kw_fold_plan(g, b, has_bias, l1_budget, math_fidelity):
    """kw-fold's blocking, or None where it does not apply, fit or pay. Its bm is the largest divisor of the core's
    tile-rows whose kw-fold CBs fit, whatever bm the walk `b` solved to: fewer M-blocks recompute fewer halo rows and
    stream the weights fewer times (sem res2.fuse_conv.1: bm 5 x 2 blocks 96.1 us, bm 10 89.7 us). The gate weighs
    that blocking's halo against the im2col of the walk `b`."""
    if not (KW_FOLD and has_bias and g.tile_input and g.sharded_output and not g.k_split):
        return None
    if g.stride != 1 or g.padding != g.dilation or not 1 <= g.dilation <= 32:
        return None
    if g.input_width % TILE or g.c_in % TILE or g.c_in_p != g.c_in:
        return None
    for bm in (d for d in range(b.core_m_tiles, 0, -1) if b.core_m_tiles % d == 0):
        if _kw_fold_plane_bytes(g, bm) > KWF_MAX_PLANE_BYTES:
            continue
        kb = _kw_fold_blocking(g, b, bm)
        if _kw_fold_cb_bytes(g, kb) <= l1_budget:
            # K-blocked rounds: a third weight slot where it fits at this bm (never at the cost of a smaller bm).
            deep = replace(kb, weight_depth=KW_FOLD_KBLOCK_WEIGHT_DEPTH)
            if kb.weight_depth < deep.weight_depth and _kw_fold_cb_bytes(g, deep) <= l1_budget:
                kb = deep
            return kb if _kw_fold_pays(g, b, kb, math_fidelity) else None
    return None


def _kw_fold_pays(g, b, kb, math_fidelity):
    """Whether kw-fold at blocking `kb` beats the K-blocked walk `b` would run (see KW_FOLD_MAX_HALO_SHARE)."""
    ct, n, bm = g.c_in_tiles, g.tensor_n_tiles, b.block_m_tiles
    taps = 6 if b.center_mode != CENTER_NONE else 9
    im2col_ns = b.max_m_blocks_per_core * (taps * ct * bm + b.slab_tile_rows * ct) * IM2COL_NS_PER_TILE
    halo_ns = kb.max_m_blocks_per_core * 6 * ct * n * FIDELITY_PHASES[math_fidelity] * LOFI_NS_PER_TILE_MAC
    if kw_fold_halo_exchange(kb):
        # The neighbours exchange the boundary rows: no halo tile-row is recomputed.
        halo_ns = 0
    return halo_ns <= KW_FOLD_MAX_HALO_SHARE * im2col_ns


def kw_fold_halo_exchange(kb):
    """kw-fold's halo exchange (kernels/conv2d_3x3_kw_fold.hpp): a core boundary inside an image row takes the boundary
    rows' missing kw partial from the neighbour core instead of recomputing a halo tile-row. Only where every core runs
    one M-block: with more, a core's first block would wait on its upper neighbour's last block (a chain)."""
    return kb.max_m_blocks_per_core == 1


def _kwf_xchg_flags(g, m_tile_start, actual_m_tiles):
    """(top, bottom): the core's boundary with its upper / lower neighbour lies inside an image row."""
    tpr = g.input_width // TILE
    t_end = m_tile_start + actual_m_tiles
    return int(m_tile_start % tpr != 0), int(t_end % tpr != 0 and t_end < g.tensor_m_tiles)


def _kwf_rowbase(g, t0, i):
    """Mirror of kw_fold.hpp rowbase: the ACC row of tile-row index i (tile-row t0 - 1 + i)."""
    tpr = g.input_width // TILE
    img0 = (t0 - 1 + tpr) // tpr
    img = (t0 - 1 + i + tpr) // tpr
    return g.dilation + TILE * i + g.dilation * (img - img0)


def _kwf_xchg_writer_args(g, work, m_index, ranges, device):
    """The writer's halo exchange args: [top, bottom, upper x, y, lower x, y, fv, lve, upper's lve, lower's fv] (ACC
    rows: fv the core's first valid row, lve one past its last)."""

    def fv_lve(i):
        t, v = ranges[i]
        return _kwf_rowbase(g, t, 1), _kwf_rowbase(g, t, v) + TILE

    top, bot = _kwf_xchg_flags(g, *ranges[m_index])
    fv, lve = fv_lve(m_index)
    up = lo = (0, 0)
    up_lve = lo_fv = 0
    if top:
        v = device.worker_core_from_logical_core(work[m_index - 1][0])
        up, up_lve = (v.x, v.y), fv_lve(m_index - 1)[1]
    if bot:
        v = device.worker_core_from_logical_core(work[m_index + 1][0])
        lo, lo_fv = (v.x, v.y), fv_lve(m_index + 1)[0]
    return [top, bot, *up, *lo, fv, lve, up_lve, lo_fv]


# kw-fold slab fill, planned on the host (kernels: the reader's and writer's kw-fold fill runtime args). The fill was
# control-bound: the on-device plan (kwf_slab_plan on both DM RISCs), a TensorAccessor lookup per page run (~0.1-0.2 us
# each) and a hand-over per 4-row batch cost the res2 reader ~2.2 us of a ~10.5 us call with every read stubbed. The host
# knows every M-block's live slab rows and every owner shard's core, so each RISC gets its reads as (core, offset, slab
# row, rows) pieces. On a split core the writer (NoC1) reads the higher shards' rows plus the last own-shard rows that
# balance the two RISCs (center_head.1's 122 KB slab is bandwidth-bound: 9.51 -> 9.10 us with the balance). Each part is
# pushed in groups of at most KWF_FILL_GROUP_ROWS slab tile-rows, the reader's first: compute waits per subblock for the
# rows it reads, so a large slab (stem.conv2: 72 rows) still starts compute on its first groups; 4 or 16 rows measured
# no better. Measured with round 0 streamed (below; LoFi bf8_b, unzoned, medians of 3 alternating): res2 10.23 -> 9.77,
# center_head.1 9.60 -> 9.10, res3.x 10.26 -> 9.87, inst dec res3 fc0 -6.2%, sem dec res3 fc0 -3.6%, stem.conv2 -1.4%.
KWF_FILL_GROUP_ROWS = 8
KWF_FILL_MAX_GROUPS = 16  # kernels: per-block group array


def _kwf_live_segments(g, t0, valid, xchg):
    """[(first input tile-row, rows, slab row)] of one M-block, in slab order: tile_slab_plan's placement (compute's
    kw_fold.hpp slab_bases mirrors it), but each band's segment holds only the input tile-rows the walk reads. The walk
    covers tile-rows [t0 - ht, t_end + hb), the halo tile-row on a side only where it recomputes it (a block boundary
    inside an image row, without the halo exchange), and reads row t + (kh - 1) R of each, so up to six rounded-out band
    rows are dead: never filled, never read. Bands that overlap share slab rows; a segment contiguous in the input and
    the slab with the previous one extends it."""
    tpr = g.input_width // TILE
    r = g.dilation * tpr
    m_in = g.num_in_px // TILE
    o_first = t0 * TILE
    o_last = min((t0 + valid) * TILE, g.num_out_px) - 1
    band_len = _block_band_len(g, o_first, o_last)
    f = _block_first_tap(g, o_first)
    t_end = t0 + valid
    live_lo = t0 - (1 if not xchg and t0 % tpr != 0 else 0)
    live_hi = t_end + (1 if not xchg and t_end % tpr != 0 and t_end < m_in else 0)
    segs = []
    next_free, read_hi, prev = 0, 0, None
    for kh in range(3):
        v_lo = f + kh * g.dilation * g.input_width
        lo, hi = max(v_lo, 0), min(v_lo + band_len, g.num_in_px)
        if lo >= hi:
            continue
        t_lo, t_hi = lo // TILE, _div_up(hi, TILE)
        row = prev[2] + (t_lo - prev[0]) if prev is not None and t_lo < prev[1] else next_free
        a = max(live_lo + (kh - 1) * r, t_lo, read_hi)
        b = min(live_hi + (kh - 1) * r, t_hi)
        if a < b:
            slab = row + (a - t_lo)
            if segs and segs[-1][0] + segs[-1][1] == a and segs[-1][2] + segs[-1][1] == slab:
                segs[-1][1] += b - a
            else:
                segs.append([a, b - a, slab])
            read_hi = b
        next_free = row + (t_hi - t_lo)
        prev = (t_lo, t_hi, row)
    return segs


def _kwf_fill_args(g, t0, valid, xchg, slab_rows, shard_rows, my_shard, split, shard_xy, row_bytes):
    """One M-block's fill: (reader args, writer args). On a split core the writer reads a suffix of the slab: the
    higher shards' rows (NoC1 is their short way back) and the last own-shard rows that balance the two RISCs' rows.
    Each part is pushed in groups of at most KWF_FILL_GROUP_ROWS slab rows, the reader's first.
    reader: [groups, writer groups] + per group [slab rows pushed] + (a reader group) [pieces] + its pieces;
    writer: [groups] + per group [pieces] + its pieces. A piece is [x << 16 | y of the owner shard's core, byte offset in
    the shard, slab row << 16 | tile-rows] (reader: slab row from the group's first row; writer: from the slab base)."""
    pieces = []  # (slab row, rows, shard, row in shard), slab order
    for a, n, s in _kwf_live_segments(g, t0, valid, xchg):
        row = a
        while row < a + n:
            sh = row // shard_rows
            end = min(a + n, (sh + 1) * shard_rows)
            pieces.append((s + row - a, end - row, sh, row - sh * shard_rows))
            row = end
    mine = [p for p in pieces if not (split and p[2] > my_shard)]
    theirs = [p for p in pieces if split and p[2] > my_shard]
    if split and theirs:
        # The last own-shard rows (a local copy on either RISC; in slab order right before the higher shards' rows, so
        # the writer's part stays a suffix) move to the writer until the two parts' rows balance.
        move = (sum(p[1] for p in mine) - sum(p[1] for p in theirs)) // 2
        while move > 0 and mine and mine[-1][2] == my_shard:
            p = mine.pop()
            k = min(move, p[1])
            if k < p[1]:
                mine.append((p[0], p[1] - k, p[2], p[3]))
            theirs.insert(0, (p[0] + p[1] - k, k, p[2], p[3] + p[1] - k))
            move -= k
    w_first = min((p[0] for p in theirs), default=slab_rows)
    assert all(p[0] + p[1] <= w_first for p in mine), "the writer's rows are a slab suffix"

    def enc(p, lo):
        x, y = shard_xy[p[2]]
        return [(x << 16) | y, p[3] * row_bytes, ((p[0] - lo) << 16) | p[1]]

    def groups(part, lo, hi, base):
        out = []  # (slab rows pushed, encoded pieces)
        while lo < hi:
            g_hi = min(hi, lo + KWF_FILL_GROUP_ROWS)
            enc_g = []
            for p in part:
                a, b = max(p[0], lo), min(p[0] + p[1], g_hi)
                if a < b:
                    enc_g += enc((a, b - a, p[2], p[3] + a - p[0]), lo if base is None else base)
            out.append((g_hi - lo, enc_g))
            lo = g_hi
        return out

    rd = groups(mine, 0, w_first, None)
    wr = groups(theirs, w_first, slab_rows, 0) if theirs else []
    reader = [len(rd) + len(wr), len(wr)]
    for rows, enc_g in rd:
        reader += [rows, len(enc_g) // 3] + enc_g
    reader += [rows for rows, _ in wr]
    writer = [len(wr)]
    for _, enc_g in wr:
        writer += [len(enc_g) // 3] + enc_g
    assert reader[0] <= KWF_FILL_MAX_GROUPS
    return reader, writer


def _kw_fold_cb_bytes(g, kb):
    """L1 of the kw-fold CBs (the descriptor's kwf CB list)."""
    slabs = 2 if kb.max_m_blocks_per_core > 1 else 1
    total = slabs * kb.slab_tile_rows * g.c_in_tiles * g.in0_tile_bytes
    total += kb.weight_depth * kb.block_k_tiles * kb.block_n_tiles * g.weight_tile_bytes
    total += _kw_fold_acc_bytes(g, kb.block_m_tiles)
    total += g.tensor_n_tiles * (g.weight_tile_bytes + KWF_BIAS_ROWS_BYTES)
    total += TILE * g.c_in_p * BF16_BYTES  # cb_zero_src (tap-major reader start)
    return total


# K-blocked kw-fold (the column rounds double-buffered, re-streamed per M-block) vs whole-K (all three resident across
# M-blocks): K-blocked from KW_FOLD_KBLOCK_MIN_WEIGHT_TILES weight tiles, for L1. Both pack each Q tile-row once and
# start at round 0; whole-K on the K-blocked LoFi cases measured flat (res3.x 14.94 -> 14.48, inst res2 fc1 28.09 ->
# 28.14 us, medians of 3).
KW_FOLD_KBLOCK_MIN_WEIGHT_TILES = 128
# kw-fold's slab rows are a whole number of these (the writer locates a block's slab in the ring in these units).
KW_FOLD_FILL_BATCH_ROWS = 4
# K-blocked kw-fold's weight slots where L1 permits at the chosen bm (2 otherwise). With two, a weight server sends round
# r + 2 only once every receiver of its rectangle freed round r's slot, then multicasts it: on res3.x (3 rounds, ~2 us
# of matmul each) round 2 landed ~0.5 us after compute reached it. Measured with the ND-sharded weights' faster server
# reads (LoFi bf8_b, unzoned, medians of 3): res3.x 10.88 -> 10.42 us, inst res2 fc1 -0.4%, the rest flat (their
# matmul per round outlasts the stream: the extra slot only fills earlier).
KW_FOLD_KBLOCK_WEIGHT_DEPTH = 3


def kw_fold_kblocked(g):
    return 9 * g.c_in_tiles * g.tensor_n_tiles >= KW_FOLD_KBLOCK_MIN_WEIGHT_TILES


# kw-fold subblocks over 1-wide ACC planes (conv2d_3x3_kw_fold.hpp): 1 x sbw where 3 or 4 output column tiles divide N,
# else 4 x 1 column runs. Measured (Blackhole p150b, LoFi bf8_b): N = 2 as 4 x 1 instead of 2 x 2, stem.conv2 34.3 ->
# 32.2 us (2 x 2 runs two unpack contexts per matmul call); N = 4 as 4 x 1 instead of 1 x 4 (four srcB unpacks per
# 4 tile-MACs instead of one), +2% to +22%.
def _kw_fold_group(n_tiles):
    """Output column tiles per kw-fold pass (one n-group): the ACC planes and a K-blocked weight sub-round's width."""
    return _largest_divisor_at_most(n_tiles, DEST_TILES_FP32_HALF_SYNC)


def _kw_fold_subblock(n_tiles):
    gw = _kw_fold_group(n_tiles)
    return (gw, 1) if gw >= 3 else (1, DEST_TILES_FP32_HALF_SYNC)


# Fused side rounds (conv2d_3x3_kw_fold.hpp): at a 2-tile n-group (4 x 1 subblocks, each in0 tile feeding one
# tile-MAC per K step), Q_0 and Q_2 of a tile-row run as one 1 x 4 subblock that unpacks each in0 tile once for both
# columns' weights; Q_1 keeps its 4 x 1 round. Weight rounds 1 and 2 then hold the pair interleaved (the writer, the
# weight server and compute share that order). Measured (Blackhole p150b, LoFi bf8_b, unzoned, bit-identical):
# stem.conv2 31.13 -> 28.21, res2 11.28 -> 10.55 us; forced K-blocked N = 2 HiFi4 -4.4%, N = 10 K-blocked -4.2%. A
# 1-tile n-group keeps the three rounds: there the pair is a 1 x 2 subblock (two tile-MACs per matmul call instead of
# four) and center_head.1 went 10.68 -> 12.14 us (zoned).
def kw_fold_fused_gw(n_tiles):
    """The fused side rounds' n-group width (2), or 0 where kw-fold keeps its three column rounds."""
    return 2 if _kw_fold_group(n_tiles) == 2 else 0


def _kw_fold_blocking(g, b, bm):
    """kw-fold at bm tile-rows per M-block (a divisor of core_m_tiles): the walk's blocking re-solved at bm (its
    per-block extents), then kw-fold's weight rounds and slab."""
    ct, n = g.c_in_tiles, g.tensor_n_tiles
    if bm != b.block_m_tiles:
        b = replace(b, block_m_tiles=bm, **_footprint(g, b.core_m_tiles, bm, b.block_k_tiles, b.block_n_tiles))
    r = g.dilation * g.input_width // TILE
    ub = KW_FOLD_FILL_BATCH_ROWS
    slab_rows = _div_up(bm + 2 + 2 * r, ub) * ub
    # One weight round: K-blocked, one column kw x one n-group of gw output column tiles ([3 C_in, gw] tiles,
    # streamed double-buffered); whole-K, one column kw x all of C_out (the three rounds resident).
    rw = _kw_fold_group(n) if kw_fold_kblocked(g) else n
    return replace(
        b,
        block_k_tiles=ct,
        block_n_tiles=3 * rw,
        num_k_blocks=3 * (n // rw),
        # K-blocked: double-buffered sub-rounds; with the fused side rounds three, so an n-group's rounds 1 and 2 (the
        # fused pair, waited for together) always sit in consecutive slots (2 slots wrap them every other n-group).
        weight_depth=2 if kw_fold_kblocked(g) and not kw_fold_fused_gw(n) else 3,
        center_mode=CENTER_NONE,
        center_slot_blocks=1,
        ring=False,
        slab_tile_rows=slab_rows,
        untilize_batch_rows=ub,
        in0_stream_rows=bm,
    )


def _kw_fold_plane_bytes(g, bm):
    """One ACC plane, one output column tile wide: [rows, 32] fp32 as two [rows, 16] half-planes (face-split, see
    conv2d_3x3_kw_fold.hpp); rows bound every core's rowbase(bm + 1) + 32 + d."""
    tpr = g.input_width // TILE
    rows = TILE * (bm + 3) + g.dilation * ((bm + 2) // tpr + 3)
    return rows * TILE * 4


def _kw_fold_acc_bytes(g, bm):
    """ACC: the planes of one n-group (the n-groups take turns)."""
    return _div_up(_kw_fold_group(g.tensor_n_tiles) * _kw_fold_plane_bytes(g, bm), TILE_BYTES_FP32) * TILE_BYTES_FP32


def _tile_slab_segments(g: Conv3x3Geometry, f, band_len):
    """The reader's TILE slab plan (mirror of conv2d_3x3::tile_slab_plan): [(first input tile-row, rows)] in slab
    order; slab row r of the plan is row r of the concatenated segments."""
    segs = []
    next_free = 0
    prev = None  # (t_lo, t_hi, slab_row) of the last non-empty band
    for kh in range(3):
        v_lo = f + kh * g.dilation * g.input_width
        lo, hi = max(v_lo, 0), min(v_lo + band_len, g.num_in_px)
        if lo >= hi:
            continue
        t_lo, t_hi = lo // TILE, _div_up(hi, TILE)
        shares = prev is not None and t_lo < prev[1]
        row = prev[2] + (t_lo - prev[0]) if shares else next_free
        first_new = prev[1] if shares else t_lo
        if t_hi > first_new:
            segs.append((first_new, t_hi - first_new))
        next_free = row + (t_hi - t_lo)
        prev = (t_lo, t_hi, row)
    return segs


def _host_fill_segments(g: Conv3x3Geometry, m_tile_start, actual_m_tiles):
    o_first = m_tile_start * TILE
    o_last = min((m_tile_start + actual_m_tiles) * TILE, g.num_out_px) - 1
    return _tile_slab_segments(g, _block_first_tap(g, o_first), _block_band_len(g, o_first, o_last))


def _spare_split_row(g: Conv3x3Geometry, m_tile_start, actual_m_tiles):
    """A spare member's split fill (SPARE_SPLIT_FILL): the input tile-row from which the writer reads (half the slab's
    rows on each NoC)."""
    rows = [
        r for first, count in _host_fill_segments(g, m_tile_start, actual_m_tiles) for r in range(first, first + count)
    ]
    return rows[len(rows) // 2]


def _host_fill_args(
    g: Conv3x3Geometry,
    m_tile_start,
    actual_m_tiles,
    ub,
    num_batches,
    shard_rows,
    owners,
    row_bytes,
    writer_row,
    writer_rows=None,
    gate_rows=0,
):
    """Host-planned fill of a one-M-block core (BLOCK_HOST_FILL). Input tile-rows in writer_rows, else at or past
    writer_row, are the writer's (None: no split). Reader args: per batch of ub slab rows, (writer flag << 31 | gate
    << 24 | reader pieces) and 3 words per piece; writer args: (gate << 16 | batches with writer pieces), then per such
    batch (pieces) and 3 words per piece. A piece is (owner x << 16 | y, byte offset in the owner's shard, slab row <<
    16 | rows); pieces stop at owner-shard edges, at batch ends and where the RISC changes. The gate (BLOCK_FILL_GATE):
    the batches holding the first gate_rows slab rows, which each RISC issues alone (the reader's count in batch 0's
    word, the writer's among its own batches)."""
    segs = _host_fill_segments(g, m_tile_start, actual_m_tiles)
    # The reader pushes num_batches batches (its slab rows); those past the plan's rows are pushed unread.
    assert sum(n for _, n in segs) <= num_batches * ub, (segs, num_batches, ub)
    batches = [([], []) for _ in range(num_batches)]
    slab_row = 0
    for first, count in segs:
        row = first
        while row < first + count:
            b = slab_row // ub
            end = min(first + count, (row // shard_rows + 1) * shard_rows, row + (b + 1) * ub - slab_row)
            if writer_rows is not None:
                on_writer = row in writer_rows
                end = next((r for r in range(row + 1, end) if (r in writer_rows) != on_writer), end)
            else:
                if writer_row is not None and row < writer_row < end:
                    end = writer_row
                on_writer = writer_row is not None and row >= writer_row
            shard = row // shard_rows
            piece = [owners[shard], (row - shard * shard_rows) * row_bytes, (slab_row << 16) | (end - row)]
            batches[b][1 if on_writer else 0].append(piece)
            slab_row += end - row
            row = end
    gate = min(_div_up(gate_rows, ub), num_batches)
    reader, writer = [], []
    writer_batches = 0
    writer_gate = 0
    for bi, (rp, wp) in enumerate(batches):
        reader.append((int(bool(wp)) << 31) | ((gate << 24) if bi == 0 else 0) | len(rp))
        for p in rp:
            reader += p
        if wp:
            writer_batches += 1
            writer_gate += int(bi < gate)
            writer.append(len(wp))
            for p in wp:
                writer += p
    return reader, [(writer_gate << 16) | writer_batches] + writer


def _first_band_rows(g: Conv3x3Geometry, m_tile_start, actual_m_tiles):
    """Slab rows of the first kernel row whose band has rows: the slab prefix the prelude reads first (BLOCK_FILL_GATE)."""
    o_first = m_tile_start * TILE
    o_last = min((m_tile_start + actual_m_tiles) * TILE, g.num_out_px) - 1
    f = _block_first_tap(g, o_first)
    band_len = _block_band_len(g, o_first, o_last)
    total = sum(n for _, n in _tile_slab_segments(g, f, band_len))
    for kh in range(3):
        lo = max(f + kh * g.dilation * g.input_width, 0)
        hi = min(f + kh * g.dilation * g.input_width + band_len, g.num_in_px)
        if lo < hi:
            return min(_div_up(hi, TILE) - lo // TILE, total)
    return 0


def _bands_disjoint(g: Conv3x3Geometry, m_tile_start, actual_m_tiles):
    """The first kernel row's band is a slab run of its own: the next band starts past it (a gap of input tile-rows)."""
    segs = _host_fill_segments(g, m_tile_start, actual_m_tiles)
    return (
        g.stride == 1
        and len(segs) >= 2
        and segs[0][1] == _first_band_rows(g, m_tile_start, actual_m_tiles)
        and segs[1][0] > segs[0][0] + segs[0][1]
    )


def _ring_links(src, dst, noc, ring):
    """Column links (by their source row) a response from row src to row dst crosses: NoC0 runs +y, NoC1 -y."""
    hops = (dst - src) % ring if noc == 0 else (src - dst) % ring
    step = 1 if noc == 0 else -1
    return [(src + step * i) % ring for i in range(hops)]


def _balance_fill_nocs(column, ring, background=None):
    """BLOCK_FILL_BALANCE: one column's grid-row fills, column = [(reader y, [(owner y, rows)])], pieces in slab order;
    background: fixed rows per ("inj" | "link", noc, y) of reads the column carries for cores outside it (spare-column
    members). Returns per reader the rows of each piece the writer reads on NoC1 (the first rows of the piece). Greedy
    from the direction split (owners below on NoC1): move the one row of a piece to the other NoC that most lowers the
    sum of the eighth powers of the column's loads in rows (a smooth max over every column link per NoC, every owner's
    injection and every reader's ejection per NoC), until no move does. aspp.convs.1 / .2 / .3: busiest resource within
    1% / 1.5% / 7% of an LP optimum, ~10-20 ms per column."""
    pieces = [(ri, oy, n) for ri, (ry, lst) in enumerate(column) for oy, n in lst]
    ry_of = [ry for ry, _ in column]
    keys = {}

    def kid(k):
        return keys.setdefault(k, len(keys))

    res = []  # per piece and NoC: the resources one row of it loads
    for ri, oy, _ in pieces:
        res.append(
            [
                [kid(("inj", noc, oy)), kid(("ej", noc, ri))]
                + [kid(("link", noc, y)) for y in _ring_links(oy, ry_of[ri], noc, ring)]
                for noc in (0, 1)
            ]
        )
    for k in background or {}:
        kid(k)
    x1 = [n if oy > ry_of[ri] else 0 for ri, oy, n in pieces]
    load = [0] * len(keys)
    for k, v in (background or {}).items():
        load[keys[k]] += v
    for i, (_, _, n) in enumerate(pieces):
        for k in res[i][0]:
            load[k] += n - x1[i]
        for k in res[i][1]:
            load[k] += x1[i]
    while True:
        best, best_i, best_d = 0, -1, 0
        for i, (_, _, n) in enumerate(pieces):
            for d, frm, to in ((1, 0, 1), (-1, 1, 0)):
                if not 0 <= x1[i] + d <= n:
                    continue
                delta = 0
                for k in res[i][frm]:
                    v = load[k]
                    delta += (v - 1) ** 8 - v**8
                for k in res[i][to]:
                    v = load[k]
                    delta += (v + 1) ** 8 - v**8
                if delta < best:
                    best, best_i, best_d = delta, i, d
        if best_i < 0:
            break
        frm, to = (0, 1) if best_d == 1 else (1, 0)
        for k in res[best_i][frm]:
            load[k] -= 1
        for k in res[best_i][to]:
            load[k] += 1
        x1[best_i] += best_d
    out = [[] for _ in column]
    for i, (ri, _, _) in enumerate(pieces):
        out[ri].append(x1[i])
    return out


def _host_fill_fits(g: Conv3x3Geometry, b: Conv3x3Blocking, shard_rows):
    """Every M work index's host-planned fill within HOST_FILL_MAX_WORDS per RISC: the plan without a split, at the
    reader's largest batch count, plus a split's at most one more piece per core and one count per writer batch."""
    ub = b.untilize_batch_rows
    num_batches = b.slab_tile_rows // ub
    for start, actual in core_m_ranges(g, b.core_m_tiles)[: b.num_m_groups]:
        owners = [0] * (_div_up(_div_up(g.num_in_px, TILE), shard_rows) + 1)
        reader, _ = _host_fill_args(g, start, actual, ub, num_batches, shard_rows, owners, 0, None)
        if len(reader) + 3 + num_batches > HOST_FILL_MAX_WORDS:
            return False
    return True


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
    resident = (
        input_tensor.memory_config().shard_spec.shape[0] // TILE * g.c_in_tiles * g.in0_tile_bytes
        + b.core_m_tiles * g.out_n_tiles * g.out_tile_bytes
        if g.sharded_output and g.tile_input
        else 0
    )
    kwf_args = (g, b, has_bias, l1_cb_budget_bytes(resident), compute_kernel_config.math_fidelity)
    kwf = kw_fold_mode(*kwf_args)
    # The reader keeps the original K-blocking in its CT args (it only fills the slab: its K-block asserts stay true).
    reader_bk, reader_nkb = b.block_k_tiles, b.num_k_blocks
    if kwf:
        b = kw_fold_plan(*kwf_args)
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
    slab_direct = slab_direct_capable(g, bk) and not kwf
    if slab_direct:
        sd_gap, sd_pitch, sd_zone, sd_slots, sd_slab_px = slab_direct_layout(g, b.core_m_tiles, bm)
        slab_cb = _cb(CB_HALO_SLAB, ttnn.bfloat16, pixel_bytes, sd_slab_px, cores)

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
    # Padded slab (see PADDED_SLAB): cb_im2col_rm keeps one unit (compute's tilize only takes its format) and
    # cb_src_table one token page; the slab grows by the gaps and the partial rows at the band ends. Taken where that
    # fits the L1 left at the solved blocking plus what the unit ring and the source table give back.
    padded_slab = False
    pad_slots_max = 0
    if PADDED_SLAB and nslice_center and not kwf and padded_slab_capable(g) and b.max_m_blocks_per_core == 1:
        pad_slots_max, pad_px = padded_slab_extent(g, b.core_m_tiles, bm)
        pad_bytes = _div_up(pad_px * pixel_bytes, TILE_BYTES_BF16) * TILE_BYTES_BF16
        old_bytes = slab_cb.total_size
        new_bytes = max(old_bytes, pad_bytes)
        given_back = (b.im2col_rm_depth_rows - 1) * bk * TILE_BYTES_BF16 + _src_table_bytes(b.im2col_rm_depth_rows)
        leftover = l1_cb_budget_bytes(resident) - b.footprint_bytes + given_back - RING_TOKEN_BYTES
        if new_bytes - old_bytes <= leftover:
            padded_slab = True
            if b.shared_slab_bytes:
                slab_cb = _shared_cb(
                    _div_up(new_bytes, TILE_BYTES_FP32) * TILE_BYTES_FP32,
                    (CB_HALO_SLAB, ttnn.bfloat16, TILE_BYTES_BF16),
                    (CB_COMBINE_SLOTS, ttnn.float32, TILE_BYTES_FP32),
                    cores,
                )
            else:
                slab_cb = _cb(CB_HALO_SLAB, ttnn.bfloat16, TILE_BYTES_BF16, new_bytes // TILE_BYTES_BF16, cores)
    # Tap shift (see TAP_SHIFT): the input ring shares the slab / combine-slots region instead of in0. The region's size
    # must be a multiple of all three page sizes.
    tap_shift = False
    if TAP_SHIFT and padded_slab and b.shared_slab_bytes and g.dilation < 32:
        ring_bytes = b.slab_tile_rows * g.c_in_tiles * g.in0_tile_bytes
        unit = math.lcm(TILE_BYTES_BF16, TILE_BYTES_FP32, g.in0_tile_bytes)
        region = _div_up(max(slab_cb.total_size, ring_bytes), unit) * unit
        if region - slab_cb.total_size <= leftover - (new_bytes - old_bytes):
            tap_shift = True
            slab_cb = ttnn.CBDescriptor(
                total_size=region,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=i, data_format=f, page_size=p)
                    for i, f, p in (
                        (CB_HALO_SLAB, ttnn.bfloat16, TILE_BYTES_BF16),
                        (CB_COMBINE_SLOTS, ttnn.float32, TILE_BYTES_FP32),
                        (CB_INPUT_TILES, input_tensor.dtype, g.in0_tile_bytes),
                    )
                ],
            )
    # Stride-2 taps from DEST (see S2_TAPS): the RM slab and the unit ring shrink to one page / unit (the shared
    # slab / combine-slots region of the N-slice walk keeps its size), the input ring holds the whole slab in its own
    # L1, and the K-blocked walk's in0 holds every K-block.
    s2_taps = False
    s2_in0_tiles = 0
    if (
        s2_taps_capable(g)
        and _strided_window(g, bk)
        and not kwf
        and not face_direct
        and not slab_direct
        and not b.ring
        and b.center_mode == CENTER_NONE
    ):
        in0_old = bm * (in0_k_tiles if b.nslice else bk * b.center_slot_blocks)
        s2_in0_tiles = bm * (in0_k_tiles if b.nslice else bk * b.num_k_blocks)
        input_old = (
            0
            if whole_slab_ring(in0_k_tiles)
            else INPUT_TILE_DEPTH_BATCHES * b.untilize_batch_rows * g.c_in_tiles * g.in0_tile_bytes
        )
        old = (
            (0 if b.shared_slab_bytes else slab_cb.total_size)
            + b.im2col_rm_depth_rows * bk * TILE_BYTES_BF16
            + in0_old * g.in0_tile_bytes
            + input_old
        )
        new = (
            (0 if b.shared_slab_bytes else TILE_BYTES_BF16)
            + bk * TILE_BYTES_BF16
            + s2_in0_tiles * g.in0_tile_bytes
            + b.slab_tile_rows * g.c_in_tiles * g.in0_tile_bytes
        )
        s2_taps = new - old <= l1_cb_budget_bytes(resident) - b.footprint_bytes
        if s2_taps and not b.shared_slab_bytes:
            slab_cb = _cb(CB_HALO_SLAB, ttnn.bfloat16, TILE_BYTES_BF16, 1, cores)
        if s2_taps:
            ub0 = b.untilize_batch_rows
            ub_s2 = max(d for d in range(1, max(1, S2_FILL_BATCH_ROWS) + 1) if ub0 % d == 0)
            if ub_s2 != ub0:
                b = replace(b, untilize_batch_rows=ub_s2)
    cbs = [slab_cb]
    if not face_direct:
        im2col_rm_pages = (1 if padded_slab or s2_taps else b.im2col_rm_depth_rows) * bk
        cbs.append(_cb(CB_IM2COL_RM, ttnn.bfloat16, TILE_BYTES_BF16, im2col_rm_pages, cores))
    cbs += [
        # Face-direct: the reader writes these tiles; it relies on every K-block starting at the CB base.
        _cb(
            CB_IM2COL_TILES,
            input_tensor.dtype,
            g.in0_tile_bytes,
            (
                2
                if slab_direct
                else (s2_in0_tiles if s2_taps else bm * (in0_k_tiles if b.nslice else bk * b.center_slot_blocks))
            ),
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
    if g.remote_output:
        # One [bm, out_n] block (tile-row major), drained by the writer to the output shards every M-block.
        cbs.append(_cb(CB_OUTPUT_TILES, output_tensor.dtype, g.out_tile_bytes, bm * ns, cores))
    elif g.sharded_output:
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
    alias_ring = whole_slab_ring(in0_k_tiles) and not s2_taps
    if alias_ring or s2_taps:
        input_ring_batches = b.slab_tile_rows // b.untilize_batch_rows
    if alias_ring and not tap_shift:
        for i, cb in enumerate(cbs):
            if cb.format_descriptors[0].buffer_index == CB_IM2COL_TILES:
                cbs[i] = _shared_cb(
                    cb.total_size,
                    (CB_IM2COL_TILES, input_tensor.dtype, g.in0_tile_bytes),
                    (CB_INPUT_TILES, input_tensor.dtype, g.in0_tile_bytes),
                    cores,
                )
    elif g.tile_input and not tap_shift:
        input_tile_pages = input_ring_batches * b.untilize_batch_rows * g.c_in_tiles
        cbs.append(_cb(CB_INPUT_TILES, input_tensor.dtype, g.in0_tile_bytes, input_tile_pages, cores))
    if b.nslice:
        cbs.append(_cb(CB_RING_TOKEN, ttnn.float32, RING_TOKEN_BYTES, RING_TOKEN_PAGES, cores))
    if tap_major_im2col(g, b):
        cbs.append(_cb(CB_ZERO_SRC, ttnn.bfloat16, pixel_bytes, TILE, cores))
    # kw-fold has no gather (its reader only fills the slab) and owns CB_SRC_TABLE as its ACC-ready token.
    tap_direct = not kwf and _tap_direct(g, bm, bk, b.nslice)
    tap_direct_fifo_units = 0 if kwf else _tap_direct_fifo_units(g, bm, bk, b.nslice)
    tap_direct_windows = not kwf and _tap_direct_windows(g, bm, bk, b.nslice, b.center_mode)
    if tap_direct_fifo_units:
        cbs.append(_cb(CB_SRC_TABLE, ttnn.bfloat16, 16, tap_direct_fifo_units, cores))
    elif tap_direct and padded_slab:
        # Padded slab: a one-page "slab dead" token, compute (packer) -> reader.
        cbs.append(_cb(CB_SRC_TABLE, ttnn.float32, RING_TOKEN_BYTES, 1, cores))
    elif tap_direct:
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

    if kwf:
        # kw-fold CBs: the whole slab (input tiles), the weights (whole-K: the three [3 C_in, C_out] column rounds;
        # K-blocked: two [3 C_in, gw] sub-round slots), the fp32 accumulator plane, the bias (the writer's read), its
        # fp32 rows (the reader's decode; final-pass srcB) and the output shard.
        alias_ring = False
        input_ring_batches = (2 if b.max_m_blocks_per_core > 1 else 1) * (b.slab_tile_rows // b.untilize_batch_rows)
        cbs = [
            # Two slabs with more than one M-block per core: the next block's fill lands during this block's walk.
            _cb(
                CB_INPUT_TILES,
                input_tensor.dtype,
                g.in0_tile_bytes,
                (2 if b.max_m_blocks_per_core > 1 else 1) * b.slab_tile_rows * g.c_in_tiles,
                cores,
            ),
            _cb(CB_WEIGHTS, weight_tensor.dtype, g.weight_tile_bytes, b.weight_depth * bk * bn, cores),
            _cb(CB_MATMUL_PARTIALS, ttnn.float32, TILE_BYTES_FP32, _kw_fold_acc_bytes(g, bm) // TILE_BYTES_FP32, cores),
            output_cb,
            _cb(CB_BIAS, bias_tensor.dtype, g.weight_tile_bytes, ns, cores),
            _cb(CB_COMBINED, ttnn.float32, KWF_BIAS_ROWS_BYTES, ns, cores),
            # ACC-ready token, compute pack -> compute unpack, once per M-block.
            _cb(CB_SRC_TABLE, ttnn.float32, RING_TOKEN_BYTES, 1, cores),
            # Halo exchange tokens (one per send / arrival, at most 3 each per n-group). Compute pops an n-group's
            # arrivals at once after its final pass: a page per arrival of the kernel keeps that pop off the ring end.
            _cb(CB_KWF_HALO_OUT, ttnn.float32, RING_TOKEN_BYTES, 3, cores),
            _cb(
                CB_KWF_HALO_IN,
                ttnn.float32,
                RING_TOKEN_BYTES,
                3 * (g.tensor_n_tiles // _kw_fold_group(g.tensor_n_tiles)),
                cores,
            ),
        ]
        if tap_major_im2col(g, b):
            cbs.append(_cb(CB_ZERO_SRC, ttnn.bfloat16, pixel_bytes, TILE, cores))
    # Channel-slice reads (R4): the core's slice of input row v is page v * row_page_stride + col offset,
    # and one owner shard holds input_shard_rows consecutive rows of it contiguously.
    slab_split = SLAB_SPLIT_NOCS and g.sharded_output and not g.tile_input and not g.k_split
    # Split TILE fill. Under a K split (N-slice-outer walk) only with the whole slab in the ring, where the reader hands
    # the writer every batch up front; a per-batch hand-over on the 2-batch ring measured slower than no split there
    # (aspp.convs.1 @ LoFi 75.8 vs 71.0 us).
    # Host-planned BLOCK fill (BLOCK_HOST_FILL): one M-block per core, the whole slab in the input ring (the stride-2
    # prelude's own ring included), every core's plan within the runtime-arg budget.
    host_fill = (
        BLOCK_HOST_FILL
        and g.k_split
        and g.tile_input
        and not kwf
        and b.nslice
        and not b.ring
        and b.max_m_blocks_per_core == 1
        and (alias_ring or s2_taps)
        and input_tensor.memory_config().is_sharded()
        and _host_fill_fits(g, b, int(input_tensor.memory_config().shard_spec.shape[0]) // TILE)
    )
    tile_fill_split = g.sharded_output and g.tile_input and (not g.k_split or (b.nslice and (alias_ring or host_fill)))
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
    m_ranges = core_m_ranges(g, b.core_m_tiles)
    m_blocks_per_index = [_div_up(actual_m_tiles, bm) for _, actual_m_tiles in m_ranges]
    # The weight multicast's cores (the channel below is built from the same set). The split TILE fill runs on its pure
    # receivers: not the sender (work core 0), nor the own-stream cores outside the channel.
    mcast_receivers = None if g.k_split else _weight_mcast_cores(core_list, [m_blocks_per_index[i] for _, i, _ in work])
    weight_servers = None
    if WEIGHT_SERVERS and g.sharded_output and not g.k_split and mcast_receivers is not None:
        # Partners under kw-fold, where a chunk splits into two parts (see WEIGHT_SERVER_PARTNERS).
        server_chunk_tiles = bk * bn // max(d for d in range(1, WEIGHT_SERVER_CHUNKS + 1) if (bk * bn) % d == 0)
        weight_servers = _weight_servers(
            input_tensor, core_list, mcast_receivers, WEIGHT_SERVER_PARTNERS and bool(kwf) and server_chunk_tiles >= 2
        )
    split_cores = set()
    if tile_fill_split and mcast_receivers is not None:
        # With weight servers every participant is a pure receiver.
        first = 0 if weight_servers is not None else 1
        split_cores = {(c.x, c.y) for c in core_list[first:] if mcast_receivers.contains(c)}
    # K split: each column multiplies against its own weight K-slice, one stream per column (built once: the split
    # fill picks its receivers from it).
    spare_groups = spare_m_indices(g, b.core_m_tiles, b.num_m_groups) if g.k_split else ()
    # The column channels take the grid-row groups; the spare-column groups get their rounds forwarded (spare_channel).
    grid_work = [w for w in work if w[1] not in spare_groups]
    column_channel = _column_weight_mcast(input_tensor, grid_work, m_blocks_per_index) if g.k_split else None
    # Spare-column groups: the column streams' second multicast (see SPARE_MGROUPS).
    spare_channel = (
        _spare_weight_mcast(input_tensor, work, m_blocks_per_index, spare_groups, column_channel)
        if spare_groups
        else None
    )
    round0_args = (
        _round0_args(
            input_tensor, weight_tensor, g, b, work, m_blocks_per_index, spare_groups, column_channel, spare_channel
        )
        if g.k_split and b.nslice and not kwf and column_channel is not None
        else {}
    )
    if tile_fill_split and g.k_split and column_channel is not None:
        # The column channels' pure receivers (the staggered sender of column g is participant row g % rows: its writer
        # blocks in DRAM reads and multicast sends: measured +9% with it split) that read enough from higher shards.
        most = max(m_blocks_per_index[i] for _, i, _ in work)
        part_rows = sorted({i for _, i, _ in work if m_blocks_per_index[i] == most and i not in spare_groups})
        hi_bytes = {
            row: _higher_shard_rows(g, b.core_m_tiles, bm, row, input_shard_rows) * g.c_in_tiles * g.in0_tile_bytes
            for row in part_rows
        }
        for core, row, col in work:
            if row in part_rows and row != part_rows[col % len(part_rows)] and hi_bytes[row] >= SPLIT_FILL_MIN_BYTES:
                split_cores.add((core.x, core.y))

    # Spare-column members: the writer reads the upper half of the slab rows on NoC1 (see SPARE_SPLIT_FILL).
    split_row_of = {}
    if host_fill and SPARE_SPLIT_FILL and tile_fill_split and spare_groups:
        for core, row, _ in work:
            if row in spare_groups:
                split_row_of[(core.x, core.y)] = _spare_split_row(g, *m_ranges[row])
                split_cores.add((core.x, core.y))
    tile_fill_split = tile_fill_split and (not g.k_split or bool(split_cores))
    pages_per_slice_row = g.c_in_tiles if g.tile_input else 1
    combine_split_nocs = g.k_split and COMBINE_SPLIT_NOCS
    tap_major = tap_major_im2col(g, b)
    # Width of one tilize block of im2col rows: one tap (tap-major) or the whole K-block.
    tilize_width = g.c_in_tiles if tap_major else bk

    stream_rows = b.in0_stream_rows
    gather_split = (
        GATHER_SPLIT_RISCS and face_direct and not slab_direct and slab_split and _div_up(bm, stream_rows) >= 2
    )
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
    # Weight servers: the server's ack counter, one data-ready flag per cb_weights slot, then the bias flag (the server
    # multicasts the bias wherever the receivers take it from cb_bias: not under the split slab fill).
    server_sem_ids = (len(sem_ids), len(sem_ids) + 1) if weight_servers is not None else (0, 0)
    # Slab-direct multiplies the bias as soon as the slab is in, so a server multicasts it there too (no core reads it).
    server_bias = weight_servers is not None and has_bias and (not slab_split or slab_direct)
    # Slab-direct fused bias without servers: every weight round carries the bias tiles after its K-block (the round's
    # DRAM reader reads and rewrites them once), so no core reads the bias pages by itself.
    sd_bias_in_weights = slab_direct and SD_FUSED_BIAS and has_bias and weight_servers is None
    if sd_bias_in_weights:
        for i, cb in enumerate(cbs):
            if cb.format_descriptors[0].buffer_index == CB_WEIGHTS:
                cbs[i] = _cb(
                    CB_WEIGHTS, weight_tensor.dtype, g.weight_tile_bytes, b.weight_depth * (bk + 1) * bn, cores
                )
    # kw-fold under weight servers streams round 0 (KWF_ROUND0_STREAM; conv2d_3x3_kw_fold.hpp): one early flag per server
    # chunk but the last, after the bias flag. res3.x's 52 KB round 0 lands over ~0.8 us of multicast; compute now
    # multiplies the first subblocks' kernel rows as each lands (res3.x 10.16 -> 9.69 us over the host fill alone,
    # unzoned, 2 alternating pairs). One-burst rounds (res2, center_head.1) land together: nothing to stream, no cost.
    server_chunks = max(d for d in range(1, WEIGHT_SERVER_CHUNKS + 1) if (bk * bn) % d == 0)
    round0_stream = kwf and weight_servers is not None and server_chunks > 1
    sem_ids += (
        tuple(range(len(sem_ids), len(sem_ids) + 2 + b.weight_depth + (server_chunks - 1 if round0_stream else 0)))
        if weight_servers is not None
        else ()
    )
    # kw-fold halo exchange: rows-landed counters from the upper and from the lower neighbour.
    kwf_xchg = kwf and kw_fold_halo_exchange(b)
    kwf_xchg_sem_ids = (len(sem_ids), len(sem_ids) + 1) if kwf_xchg else (0, 0)
    sem_ids += kwf_xchg_sem_ids if kwf_xchg else ()
    # Weight servers with more rounds than cb_weights slots: one server ack counter per slot (slot 0 keeps the server's
    # ack semaphore, the others from SERVER_SLOT_ACKS on; conv2d_3x3_writer.cpp server_reuse_ack). With one total
    # counter, a receiver that runs ahead (the ragged short core) acks later slots before a slower receiver acked an
    # earlier one, and the server could overwrite that slot under the slower core's matmul.
    server_rounds = (
        b.num_k_blocks
        if kwf and not kw_fold_kblocked(g)
        else max(m_blocks_per_index[i] for _, i, _ in work) * b.num_k_blocks
    )
    slot_ack_defines = []
    if (
        weight_servers is not None
        and 1 < b.weight_depth < server_rounds
        and len(sem_ids) + b.weight_depth - 1 <= MAX_SEMAPHORES
    ):
        slot_ack_defines = [("SERVER_SLOT_ACKS", str(len(sem_ids)))]
        sem_ids += tuple(range(len(sem_ids), len(sem_ids) + b.weight_depth - 1))
    # Slab-direct round 0 over NoC0 (SD_R0_NOC0): every participant's ready flag, the sender's local go / source-read
    # state (no ack count: round 0's slot is fresh at kernel start).
    sd_r0_noc0 = slab_direct and weight_servers is None and mcast_receivers is not None
    if sd_r0_noc0:
        assert len(sem_ids) + 2 <= MAX_SEMAPHORES, sem_ids
    sd_r0_sems = tuple(range(len(sem_ids), len(sem_ids) + 2)) if sd_r0_noc0 else ()
    sem_ids += sd_r0_sems
    # N-slice spread column stream (see NSLICE_SPREAD): every round of a core's stream in its own slot, one flag
    # semaphore per round, no channel. Not where cores run different M-block counts (the cores outside the column
    # channel stream their own rounds), a stream of one round (nothing to spread), or the rounds or their flags do not
    # fit (aspp.convs.1 / .2: 8 x 59 KB rounds).
    spread_rounds = 0
    spread_flag_sem0 = 0
    spread_banks = 0
    if NSLICE_SPREAD and round0_args:
        most = max(m_blocks_per_index[i] for _, i, _ in work)
        rounds = _div_up(g.tensor_n_tiles, ns) * most
        slice_bytes = g.tensor_k_tiles * ns * g.weight_tile_bytes
        extra = max(rounds - b.weight_depth, 0) * slice_bytes
        if (
            all(m_blocks_per_index[i] == most for _, i, _ in work)
            and 2 <= rounds <= 32
            and len(sem_ids) + rounds <= MAX_SEMAPHORES
            and sum(cb.total_size for cb in cbs) + extra <= l1_cb_budget_bytes(resident)
        ):
            spread_rounds = rounds
            spread_flag_sem0 = len(sem_ids)
            sem_ids += tuple(range(len(sem_ids), len(sem_ids) + rounds))
            round0_args = _spread_args(
                input_tensor, work, m_blocks_per_index, spare_groups, column_channel, spare_channel, round0_args, rounds
            )
            if rounds > b.weight_depth:
                b = replace(b, weight_depth=rounds)
                for i, cb in enumerate(cbs):
                    if cb.format_descriptors[0].buffer_index == CB_WEIGHTS:
                        cbs[i] = _cb(
                            CB_WEIGHTS,
                            weight_tensor.dtype,
                            g.weight_tile_bytes,
                            rounds * slice_bytes // g.weight_tile_bytes,
                            cores,
                        )
            # A round is ND shard (channel slice, N-slice) of conv2d_3x3._block_weight_memory_config, round-robin over
            # the DRAM banks: the writer computes its bank and offset itself (the TensorAccessor's page-to-shard
            # arithmetic costs ~0.15 us of RISC time per read). Any other placement keeps the accessor.
            wspec = weight_tensor.memory_config().nd_shard_spec
            if (
                wspec is not None
                and list(wspec.shard_shape)[-2:] == [b.block_k_tiles * b.num_k_blocks * TILE, ns * TILE]
                and g.tensor_n_tiles % ns == 0
                and wspec.shard_distribution_strategy == ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D
                and wspec.grid.num_cores() == input_tensor.device().dram_grid_size().x
                and weight_tensor.buffer_aligned_page_size() == g.weight_tile_bytes
            ):
                spread_banks = wspec.grid.num_cores()
    k_padded = g.tensor_kkh_tiles * TILE > 3 * g.c_in_p
    # Work-balanced M split (g.m_ranges): the N-slice prelude, walk and forwarding sized by each M-block's valid
    # tile-rows (the split row gather keeps its CT unit sequence), the fill / untilize by the block's own slab rows where
    # every core runs one M-block (the split fill's writer counts batches per block at the CT slab), and the walk's group
    # matmuls specialized per group height. Not on the uniform split, where they only cost (measured, LoFi: runtime
    # sizing alone aspp.convs.3 +4.4 / +5.2 / +6.2%, its ragged and one-kernel-row rows finishing early, +1.1% where no
    # row is ragged; template group heights alone -1% to +2%).
    rt_valid = bool(g.m_ranges) and g.k_split and b.nslice and not kwf
    rt_prelude = rt_valid and not row_split
    # Spare-column groups: their uniform split's short last group (bm 5: 4 + 1) runs at its template height too, and a
    # 1-row group of a 2..4-tile slice as one 1 x out_n subblock (compute RING_WIDE_SHORT_GROUP: 1 + out_n unpacks per
    # k-step, not 2 out_n). res5.0 / res5.1 on 13 groups: 33.7 -> 29.3 / 33.7 -> 29.1 us zoned; runtime heights or 1 x 1
    # short groups leave the 5-row walk as slow as HEAD's 6-row one (4 + 2).
    ct_group_rows = (rt_valid or (g.block_rows > 0 and b.nslice and not kwf)) and _ring_subblock(bm, ns) == (4, 1)

    def rt_slab_rows(m_tile_start, actual_m_tiles, num_m_blocks):
        if not (g.tile_input and b.max_m_blocks_per_core == 1):
            return [b.slab_tile_rows] * num_m_blocks
        o_first = m_tile_start * TILE
        o_last = min((m_tile_start + actual_m_tiles) * TILE, g.num_out_px) - 1
        rows = _tile_slab_rows(g, _block_first_tap(g, o_first), _block_band_len(g, o_first, o_last))
        ub = b.untilize_batch_rows
        rows = _div_up(max(rows, 1), ub) * ub
        assert rows <= b.slab_tile_rows, (rows, b.slab_tile_rows)
        return [rows]

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
            g.pad_top,
            g.stride,
            g.dilation,
            bm,
            reader_bk,
            g.tensor_kkh_tiles,
            reader_nkb,
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
            slab_direct_band_last_tap(g) if slab_direct else g.band_last_tap,
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
            # Tap-direct group: N-slice prelude, at most half the ring, so compute always has the other half; K-blocked
            # walk, entries of the unit FIFO.
            (
                max(1, min(TAP_DIRECT_FIFO_GROUP, tap_direct_fifo_units))
                if tap_direct_fifo_units
                else max(1, min(TAP_DIRECT_GROUP, b.im2col_rm_depth_rows // 2))
            )
            if tap_direct
            else 1,
            input_ring_batches,
            # A window past the ring would reserve more than the CB holds.
            (
                min(FILL_WINDOW_BATCHES if not g.k_split else KSPLIT_FILL_WINDOW_BATCHES, input_ring_batches)
                if g.tile_input and not b.ring
                else 1
            ),
            tap_direct_fifo_units,
            int(tap_direct_windows),
            int(nslice_center),
            int(rt_valid),
            g.pad_left,
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
        int(tap_direct_fifo_units > 0),
        int(tap_direct_windows),
        int(nslice_center),
        int(rt_valid),
        int(rt_prelude),
        int(ct_group_rows),
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
        g.pad_top,
        g.stride,
        g.dilation,
        b.band_stride_px,
        slab_direct_band_last_tap(g) if slab_direct else g.band_last_tap,
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
        HOST_FILL_WRITER_WINDOW if host_fill else (FILL_WRITER_WINDOW_BATCHES if not b.ring else 1),
        int(weight_servers is not None),
        *server_sem_ids,
        int(g.remote_output),
    ]
    writer_zero_lanes_arg = len(writer_ct_args) - 7
    writer_ct_args.append(g.pad_left)

    def writer_ct_args_for(slice_channels, split_core=None):
        args = list(writer_ct_args)
        args[writer_slice_arg] = slice_channels
        args[writer_slice_arg + 2] = int(zero_im2col_once(slice_channels) and not s2_taps)
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
        if g.sharded_output and not g.remote_output
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

    # Host-planned fill: each column's owner cores (one per input shard row), and per core the reader / writer words.
    owner_xy = {}
    host_fill_words = {}
    if host_fill:
        in_spec = input_tensor.memory_config().shard_spec
        shard_grid_rows, _ = block_grid_shape(in_spec)
        for col in range(g.k_cols):
            owner_xy[col] = []
            for shard in range(shard_grid_rows):
                v = device.worker_core_from_logical_core(block_core(in_spec, shard, col))
                owner_xy[col].append((v.x << 16) | v.y)
        owner_row_bytes = (int(in_spec.shape[1]) // TILE) * input_page_bytes
        # BLOCK_FILL_BALANCE: per column, the grid-row cores' writer rows.
        writer_rows_of = {}
        if (
            BLOCK_FILL_BALANCE
            and tile_fill_split
            and g.stride == 1
            and device.arch() == ttnn._ttnn.device.Arch.BLACKHOLE
        ):
            for col in range(g.k_cols):
                members = [(core, m_index) for core, m_index, c in work if c == col and m_index not in spare_groups]
                # The column ring model: every member and owner on one NoC column (ROW_MAJOR shards).
                if (
                    len(
                        {device.worker_core_from_logical_core(core).x for core, _ in members}
                        | {xy >> 16 for xy in owner_xy[col]}
                    )
                    != 1
                ):
                    continue
                # Spare-column members of this slice read from this column's owners: their NoC1 half climbs the
                # column (Y first) before it turns along the member's row; both halves load the owners' injection.
                background = {}
                for core, m_index, c in work:
                    if c != col or m_index not in spare_groups:
                        continue
                    sy = device.worker_core_from_logical_core(core).y
                    split = split_row_of.get((core.x, core.y))
                    for first, count in _host_fill_segments(g, *m_ranges[m_index]):
                        for r in range(first, first + count):
                            oy = owner_xy[col][r // input_shard_rows] & 0xFFFF
                            noc = 1 if split is not None and r >= split else 0
                            background[("inj", noc, oy)] = background.get(("inj", noc, oy), 0) + 1
                            for y in _ring_links(oy, sy, 1, _BH_RING_ROWS) if noc == 1 else ():
                                background[("link", 1, y)] = background.get(("link", 1, y), 0) + 1
                # Only where the kernel-row bands are disjoint (_bands_disjoint): with overlapping bands (small d) the
                # slab is one short run and the moved rows only add writer hand-overs (32x64x128 on 4 slices, d 1:
                # +7.3%; stride 2 on 4 slices: +6% / +7%).
                if not any(_bands_disjoint(g, *m_ranges[m_index]) for _, m_index in members):
                    continue
                column = []
                rows_of = []
                for core, m_index in members:
                    pieces, rows = [], []
                    for first, count in _host_fill_segments(g, *m_ranges[m_index]):
                        for r in range(first, first + count):
                            shard = r // input_shard_rows
                            if pieces and pieces[-1][1] == shard and rows[-1] == r - 1:
                                pieces[-1][2] += 1
                            else:
                                pieces.append([owner_xy[col][shard] & 0xFFFF, shard, 1])
                            rows.append(r)
                    column.append((device.worker_core_from_logical_core(core).y, [(oy, n) for oy, _, n in pieces]))
                    rows_of.append((core, pieces, rows))
                for (core, pieces, rows), xs in zip(rows_of, _balance_fill_nocs(column, _BH_RING_ROWS, background)):
                    w, at = set(), 0
                    for (_, _, n), k in zip(pieces, xs):
                        w.update(rows[at : at + k])
                        at += n
                    if len(w) * owner_row_bytes < SPLIT_FILL_MIN_BYTES:
                        # A small writer share is pure latency (as SPLIT_FILL_MIN_BYTES): 32x64x128 on 4 slices
                        # (~1 KB rows) measured +6% / +7% with it, stride 2 on 4 slices +7%.
                        w = set()
                    writer_rows_of[(core.x, core.y)] = w
                    if w:
                        split_cores.add((core.x, core.y))
                    else:
                        split_cores.discard((core.x, core.y))
        for core, m_index, col in work:
            writer_rows = writer_rows_of.get((core.x, core.y)) or None
            if (core.x, core.y) in split_row_of:
                writer_row = split_row_of[(core.x, core.y)]
            elif split_of(core):
                writer_row = (m_index + 1) * input_shard_rows
            else:
                writer_row = None
            fill_args = (
                g,
                *m_ranges[m_index],
                b.untilize_batch_rows,
                (rt_slab_rows(*m_ranges[m_index], 1)[0] if rt_valid else b.slab_tile_rows) // b.untilize_batch_rows,
                input_shard_rows,
                owner_xy[col],
                owner_row_bytes,
            )
            # The gate only where the first band is a run of its own (_bands_disjoint): else it would hold back only the
            # next bands' short tails (32x64x128 on 4 slices, d 1: +2.8% with it).
            gate_rows = (
                _first_band_rows(g, *m_ranges[m_index])
                if BLOCK_FILL_GATE and _bands_disjoint(g, *m_ranges[m_index])
                else 0
            )
            words = _host_fill_args(*fill_args, writer_row, writer_rows, gate_rows)
            if writer_rows and max(len(w) for w in words) > HOST_FILL_MAX_WORDS:
                # The balanced plan cuts more pieces: past the runtime-arg budget, the direction split (one cut).
                words = _host_fill_args(*fill_args, (m_index + 1) * input_shard_rows, None, gate_rows)
            host_fill_words[(core.x, core.y)] = words
            assert max(len(w) for w in words) <= HOST_FILL_MAX_WORDS

    def server_rt_args(core):
        """Weight servers: [complete-flag value (1, or 0x0101 with a partner), server x, server y] on a receiver, else
        zeros."""
        for rect, server, partner in weight_servers or []:
            if rect.contains(core):
                virtual = device.worker_core_from_logical_core(server)
                return [0x0101 if partner is not None else 1, virtual.x, virtual.y]
        return [0, 0, 0]

    ring_owners = None if kwf else ring_owner_cols(g, b)
    kwf_shard_xy = []
    if kwf:
        assert input_tensor.memory_config().is_sharded() and tile_fill_split and input_shard_rows > 0
        for c in shard_cores(input_tensor):
            v = device.worker_core_from_logical_core(c)
            kwf_shard_xy.append((v.x, v.y))
    # SD_R0_NOC0: the sender's reader args (the reader's SD_R0_ARG; 0 on every other core): role, the round's bytes, the
    # rectangles, and per rectangle its NoC0 multicast start (low corner), end (high corner) and destination count (the
    # sender excluded).
    sd_r0_arg = None
    sd_r0_args = []
    if sd_r0_noc0:
        sender = core_list[0]
        rects = list(mcast_receivers.ranges())
        inside = [r.start.x <= sender.x <= r.end.x and r.start.y <= sender.y <= r.end.y for r in rects]
        sd_r0_args = [1, (bk + int(sd_bias_in_weights)) * bn * g.weight_tile_bytes, len(rects)]
        for rect, ins in zip(rects, inside):
            lo = device.worker_core_from_logical_core(rect.start)
            hi = device.worker_core_from_logical_core(rect.end)
            sd_r0_args += [lo.x, lo.y, hi.x, hi.y, rect.grid_size().x * rect.grid_size().y - int(ins)]
    for core, m_index, col in work:
        m_tile_start, actual_m_tiles = m_ranges[m_index]
        num_m_blocks_this_core = m_blocks_per_index[m_index]
        # The core's output columns: all of C_out, or N-slice `col` under the K split (possibly short or
        # empty past the last C_out tile); with spread ring owners, the owner's rank among them, or none.
        out_slice = col if ring_owners is None else (ring_owners.index(col) if col in ring_owners else g.k_cols)
        out_n_tile_start = out_slice * ns
        out_valid_tiles = max(0, min(ns, g.tensor_n_tiles - out_n_tile_start))
        # The split is a runtime flag: a per-core kernel split fragments the core ranges (a staggered sender per
        # column), and the launch of many-range kernels measured up to ~10 us of start skew.
        kwf_reader_fill, kwf_writer_fill = [], []
        if kwf:
            for m in range(num_m_blocks_this_core):
                rd, wr = _kwf_fill_args(
                    g,
                    m_tile_start + m * bm,
                    min(bm, actual_m_tiles - m * bm),
                    kwf_xchg,
                    b.slab_tile_rows,
                    input_shard_rows,
                    m_index,
                    split_of(core),
                    kwf_shard_xy,
                    g.c_in_tiles * g.in0_tile_bytes,
                )
                kwf_reader_fill += rd
                kwf_writer_fill += wr
        reader_cores, reader_rt = reader_groups.setdefault(
            (_slice_channels(g, col), bool(split_cores)), ([], ttnn.RuntimeArgs())
        )
        reader_cores.append(core)
        # rt_valid: each M-block's slab rows (reader args after the split flag, compute args after the masks).
        slab_rows_per_block = rt_slab_rows(m_tile_start, actual_m_tiles, num_m_blocks_this_core) if rt_valid else []
        reader_args = (
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
            + slab_rows_per_block
            + kwf_reader_fill
            + (host_fill_words[(core.x, core.y)][0] if host_fill else [])
        )
        if sd_r0_noc0:
            assert sd_r0_arg in (None, len(reader_args))
            sd_r0_arg = len(reader_args)
            reader_args += sd_r0_args if (core.x, core.y) == (core_list[0].x, core_list[0].y) else [0]
        reader_rt[core.x][core.y] = reader_args
        compute_args = [num_m_blocks_this_core, int(out_valid_tiles > 0), col]
        if g.k_split and b.nslice and not kwf:
            # The N-slice ring walk's kernel-row masks (args 3..: no ring slab under a K split).
            assert not b.ring
            compute_args += _nslice_kh_masks(
                g, bm, _ring_subblock(bm, ns)[0], m_tile_start, actual_m_tiles, num_m_blocks_this_core
            )
            if rt_valid:
                for m in range(num_m_blocks_this_core):
                    compute_args += [min(bm, actual_m_tiles - m * bm), slab_rows_per_block[m]]
        if slab_direct or s2_taps:
            compute_args += [m_tile_start, actual_m_tiles]
        if kwf or padded_slab:
            compute_args += [m_tile_start, actual_m_tiles]
            compute_args += _kwf_xchg_flags(g, m_tile_start, actual_m_tiles) if kwf_xchg else [0, 0]
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
            + server_rt_args(core)
            + (_kwf_xchg_writer_args(g, work, m_index, m_ranges, device) if kwf_xchg else [])
            + kwf_writer_fill
            + (round0_args.get((core.x, core.y), [0] * (12 if spread_rounds else 11)) if round0_args else [])
            + (host_fill_words[(core.x, core.y)][1] if host_fill else [])
        )

    # kw-fold kernels: the mode flag and its compile-time shape (defines keep the regular kernels' CT args intact).
    _kwf_dm_defines, _kwf_compute_defines = [], []
    if kwf:
        _sbw, _sbh = _kw_fold_subblock(g.tensor_n_tiles)
        _tpr = g.input_width // TILE
        _kwf_dm_defines = [
            ("CONV2D_3X3_KW_FOLD", "1"),
            ("KWF_KBLOCKED", str(int(kw_fold_kblocked(g)))),
            ("KWF_CB_BIAS_ROWS", str(CB_COMBINED)),
            ("KWF_CB_BIAS", str(CB_BIAS)),
            ("KWF_N", str(g.tensor_n_tiles)),
            ("KWF_BIAS_BFP8", str(int(bias_tensor.dtype == ttnn.bfloat8_b))),
            ("KWF_FUSED_GW", str(kw_fold_fused_gw(g.tensor_n_tiles))),
            ("KWF_FILL_MAX_GROUPS", str(KWF_FILL_MAX_GROUPS)),
        ]
        if kwf_xchg:
            _kwf_dm_defines += [
                ("KWF_XCHG", "1"),
                ("KWF_CB_HALO_OUT", str(CB_KWF_HALO_OUT)),
                ("KWF_CB_HALO_IN", str(CB_KWF_HALO_IN)),
                ("KWF_CB_ACC", str(CB_MATMUL_PARTIALS)),
                ("KWF_D", str(g.dilation)),
                # ACC: one n-group's planes, one output column tile wide each (conv2d_3x3_kw_fold.hpp).
                # A half-plane row (the face-split ACC: each plane is two [rows, 16] fp32 half-planes).
                ("KWF_PITCH", str(16 * 4)),
                ("KWF_PLANE_BYTES", str(_kw_fold_plane_bytes(g, bm))),
                ("KWF_PLANES", str(_kw_fold_group(g.tensor_n_tiles))),
                ("KWF_NG", str(g.tensor_n_tiles // _kw_fold_group(g.tensor_n_tiles))),
                ("KWF_SEM_FROM_UPPER", str(kwf_xchg_sem_ids[0])),
                ("KWF_SEM_FROM_LOWER", str(kwf_xchg_sem_ids[1])),
            ]
        _kwf_compute_defines = [
            ("CONV2D_3X3_KW_FOLD", "1"),
            ("KWF_D", str(g.dilation)),
            ("KWF_TPR", str(_tpr)),
            ("KWF_SBW", str(_sbw)),
            ("KWF_SBH", str(_sbh)),
            ("KWF_GW", str(_kw_fold_group(g.tensor_n_tiles))),
            ("KWF_M_TILES", str(g.tensor_m_tiles)),
            ("KWF_KBLOCKED", str(int(kw_fold_kblocked(g)))),
            ("KWF_PLANE_BYTES", str(_kw_fold_plane_bytes(g, bm))),
            # Out-of-line subblock (see conv2d_3x3_kw_fold.hpp): whole-K with multi-row subblocks.
            ("KWF_NOINLINE", str(int(not kw_fold_kblocked(g) and _sbh > 1))),
            ("KWF_CB_HALO_OUT", str(CB_KWF_HALO_OUT)),
            ("KWF_CB_HALO_IN", str(CB_KWF_HALO_IN)),
            # The halo exchange's code only where it runs (it costs RISC time on small subblocks even when idle).
            ("KWF_XCHG", str(int(kwf_xchg))),
            ("KWF_FUSED", str(int(kw_fold_fused_gw(g.tensor_n_tiles) != 0))),
        ]
    _pad_dm_defines, _pad_compute_defines = [], []
    if padded_slab:
        _pad_dm_defines = [("CONV2D_3X3_PADDED_SLAB", "1"), ("PAD_SLAB_SLOTS_MAX", str(pad_slots_max))]
        if tap_shift:
            _pad_dm_defines.append(("CONV2D_3X3_TAP_SHIFT", "1"))
        _pad_compute_defines = [
            ("CONV2D_3X3_PADDED_SLAB", "1"),
            ("PAD_SLAB_IN_H", str(g.input_height)),
            ("PAD_SLAB_IN_W", str(g.input_width)),
            ("PAD_SLAB_OUT_W", str(g.output_width)),
            ("PAD_SLAB_PAD", str(g.padding)),
            ("PAD_SLAB_DIL", str(g.dilation)),
            ("PAD_SLAB_NUM_IN_PX", str(g.num_in_px)),
            ("PAD_SLAB_NUM_OUT_PX", str(g.num_out_px)),
            ("PAD_SLAB_SLOTS_MAX", str(pad_slots_max)),
            ("PAD_SLAB_CB_ZERO", str(CB_ZERO_SRC)),
            *([("CONV2D_3X3_TAP_SHIFT", "1")] if tap_shift else []),
        ]
    _sd_dm_defines, _sd_compute_defines = [], []
    if slab_direct:
        _sd_dm_defines = [
            ("CONV2D_3X3_SLAB_DIRECT", "1"),
            ("SD_GAP", str(sd_gap)),
            ("SD_PITCH", str(sd_pitch)),
            ("SD_ZONE", str(sd_zone)),
            ("SD_SLOTS", str(sd_slots)),
            ("SD_SLAB_PX", str(sd_slab_px)),
            ("SD_HALF_BYTES", str(g.face_taps * g.dilation * pixel_bytes)),
        ]
        if sd_bias_in_weights:
            _sd_dm_defines += [("SD_BIAS_IN_WEIGHTS", "1")]
        if sd_r0_noc0:
            _sd_dm_defines += [
                ("SD_R0_NOC0", "1"),
                ("SD_R0_READY_SEM", str(sd_r0_sems[0])),
                ("SD_R0_LOCAL_SEM", str(sd_r0_sems[1])),
                ("SD_R0_ARG", str(sd_r0_arg)),
                ("SD_R0_CB_WEIGHTS", str(CB_WEIGHTS)),
            ]
        if SD_STREAM_CHUNK_PX > 0 and bm // b.out_subblock_h >= SD_STREAM_MIN_SUBBLOCK_ROWS:
            _sd_dm_defines += [
                ("SD_STREAM", "1"),
                ("SD_STREAM_CHUNK_PX", str(SD_STREAM_CHUNK_PX)),
                ("SD_STREAM_NTRID", str(SD_STREAM_NTRID)),
                ("SD_STREAM_CHUNK_PX_W", str(SD_STREAM_CHUNK_PX_W)),
                ("SD_STREAM_NTRID_R", str(SD_STREAM_NTRID_R)),
            ]
        if SD_FUSED_BIAS and has_bias:
            _sd_dm_defines += [("SD_FUSED_BIAS", "1")]
            if compute_kernel_config.math_fidelity == ttnn.MathFidelity.LoFi:
                # LoFi multiplies 5 significant bits of in1: the writer splits the bias into two exact parts. HiFi2 and
                # up multiply 9: the prepared tile (row 0 the bias, its zero padding elsewhere) is used as is.
                _sd_dm_defines += [
                    ("SD_BIAS_SPLIT", "1"),
                    ("SD_BIAS_BF16", str(int(bias_tensor.dtype == ttnn.bfloat16))),
                ]
        _sd_compute_defines = _sd_dm_defines + [
            ("SD_IN_H", str(g.input_height)),
            ("SD_IN_W", str(g.input_width)),
            ("SD_OUT_W", str(g.output_width)),
            ("SD_STRIDE", str(g.stride)),
            ("SD_PAD_TOP", str(g.pad_top)),
            ("SD_PAD_LEFT", str(g.pad_left)),
            ("SD_DIL", str(g.dilation)),
            ("SD_PIXEL_BYTES", str(pixel_bytes)),
            ("SD_KKH", str(g.tensor_kkh_tiles)),
        ]
    # Spread N-slice ring owners (ring_owner_cols): the owning columns' mask, for the ring order of all three kernels.
    _ring_owner_defines = (
        [("RING_OWNER_MASK", f"{sum(1 << c for c in ring_owners)}u")] if ring_owners is not None else []
    )
    _wide_short_defines = [("RING_WIDE_SHORT_GROUP", "1")] if ct_group_rows else []
    # N-slice ring: the packer waits for its incoming partials on the arrival count itself (see the reader's
    # nslice_outer forwarding). Only a semaphore id: harmless where the walk is not compiled.
    _ring_arrival_defines = [("RING_ARRIVAL_SEM_ID", str(SEM_COMBINE_ARRIVAL))]
    _s2_dm_defines = [("CONV2D_3X3_S2_TAPS", "1")] if s2_taps else []
    _s2_compute_defines = (
        [
            ("CONV2D_3X3_S2_TAPS", "1"),
            ("S2_IN_H", str(g.input_height)),
            ("S2_IN_W", str(g.input_width)),
            ("S2_OUT_W", str(g.output_width)),
            ("S2_NUM_IN_PX", str(g.num_in_px)),
            ("S2_NUM_OUT_PX", str(g.num_out_px)),
        ]
        if s2_taps
        else []
    )
    # N-slice walk: the writer reads the bias without a wait once the first weight round is issued and pushes it when it
    # lands, so the column weight stream starts at kernel start (with the spare groups that stream paces the walk:
    # res4.x 12022 -> 11607 ns; without them flat to -1.7%).
    _async_bias_defines = [("NSLICE_ASYNC_BIAS", "1")] if g.k_split and b.nslice and has_bias and not kwf else []
    _round0_defines = [("KWF_ROUND0_STREAM", str(server_chunks))] if round0_stream else []
    # N-slice walk with one weight round per ND shard (conv2d_3x3._block_weight_memory_config): the writer reads a
    # round as one run. Weights placed any other way (DRAM interleaved) keep the one-tile reads.
    _wspec = weight_tensor.memory_config().nd_shard_spec
    _page_run_defines = (
        [("NSLICE_WEIGHT_PAGE_RUNS", "1")]
        if g.k_split
        and b.nslice
        and not kwf
        and _wspec is not None
        and list(_wspec.shard_shape)[-2:] == [b.block_k_tiles * b.num_k_blocks * TILE, g.out_n_tiles * TILE]
        else []
    )
    # Bank-run fill wherever a ROW_MAJOR input sits DRAM-interleaved with one pixel per page (C_in a multiple of 32: no
    # padded lanes) on the K-blocked tap-direct walk. Its staging ring aliases that walk's gathered-block FIFO
    # (cb_im2col_rm), which is dead during the fill: the reader refills the slab only once compute popped every unit of
    # the previous M-block, and it gathers again only after the fill. No new L1, no blocking change. Elsewhere the
    # per-pixel fill: TILE and sharded inputs have no pixel pages to coalesce, L1-interleaved pages stride by the L1
    # bank count, and the other walks keep cb_im2col_rm live during the fill.
    brf_stage = _brf_staging(g, b.im2col_rm_depth_rows * bk * TILE_BYTES_BF16)
    brf_fill = (
        brf_stage is not None
        and not g.tile_input
        and input_tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
        and input_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
        and g.c_in == g.c_in_p
        and input_tensor.buffer_aligned_page_size() == pixel_bytes
        and tap_direct_fifo_units > 0
    )
    _host_fill_defines = (
        [("CONV2D_3X3_HOST_FILL", "1"), ("HOST_FILL_WRITER_WINDOW", str(HOST_FILL_WRITER_WINDOW))] if host_fill else []
    )
    _brf_defines = (
        [
            ("BANK_RUN_FILL", "1"),
            ("BRF_NSTG", str(brf_stage[0])),
            ("BRF_CHUNK_PX", str(brf_stage[1])),
        ]
        if brf_fill
        else []
    )
    readers = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "conv2d_3x3_reader.cpp"),
            core_ranges=_core_range_set(reader_cores),
            compile_time_args=reader_ct_args(*slice_channels),
            runtime_args=reader_rt,
            defines=_kwf_dm_defines
            + _pad_dm_defines
            + _ring_owner_defines
            + _sd_dm_defines
            + _s2_dm_defines
            + _host_fill_defines
            + _brf_defines,
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
            defines=_kwf_dm_defines
            + _ring_owner_defines
            + _sd_dm_defines
            + _async_bias_defines
            + _page_run_defines
            + _round0_defines
            + slot_ack_defines
            + _host_fill_defines
            + ([("NSLICE_ROUND0", "1")] if round0_args else [])
            + (
                [
                    ("NSLICE_SPREAD", "1"),
                    ("NSLICE_SPREAD_ROUNDS", str(spread_rounds)),
                    ("NSLICE_SPREAD_FLAG_SEM0", str(spread_flag_sem0)),
                    ("NSLICE_SPREAD_LOOKAHEAD", str(NSLICE_SPREAD_LOOKAHEAD)),
                    ("NSLICE_SPREAD_BANKS", str(spread_banks)),
                ]
                if spread_rounds
                else []
            ),
            config=ttnn.WriterConfigDescriptor(),
        )
        for slice_channels, (writer_cores, writer_rt) in writer_groups.items()
    ]
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "conv2d_3x3_compute.cpp"),
        core_ranges=cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt,
        defines=_kwf_compute_defines
        + _pad_compute_defines
        + _ring_owner_defines
        + _sd_compute_defines
        + _wide_short_defines
        + _s2_compute_defines
        + _ring_arrival_defines,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=compute_kernel_config.math_fidelity,
            math_approx_mode=compute_kernel_config.math_approx_mode,
            fp32_dest_acc_en=True,
        ),
    )
    server_kernels = []
    sem_cores = cores
    if weight_servers is not None:
        # The servers hold cb_weights (same index and size, so the same L1 address as the receivers' slots) and the
        # program's semaphores; their kernels stream every participant's rounds.
        servers = [server for _, server, _ in weight_servers]
        partners = [partner for _, _, partner in weight_servers if partner is not None]
        sem_cores = _core_range_set(core_list + servers + partners)
        for i, cb in enumerate(cbs):
            if cb.format_descriptors[0].buffer_index == CB_WEIGHTS:
                pages = cb.total_size // g.weight_tile_bytes
                cbs[i] = _cb(CB_WEIGHTS, weight_tensor.dtype, g.weight_tile_bytes, pages, sem_cores)
            elif server_bias and cb.format_descriptors[0].buffer_index == CB_BIAS:
                # HEIGHT: every receiver's cb_bias holds all of C_out, and the server multicasts it there.
                assert ns == g.tensor_n_tiles
                cbs[i] = _cb(CB_BIAS, bias_tensor.dtype, g.weight_tile_bytes, ns, sem_cores)
        # Whole-K kw-fold keeps its weights resident across M-blocks: one stream.
        num_rounds = server_rounds
        num_chunks = max(d for d in range(1, WEIGHT_SERVER_CHUNKS + 1) if (bk * bn) % d == 0)
        server_reader_rt = ttnn.RuntimeArgs()
        server_writer_rt = ttnn.RuntimeArgs()
        partner_reader_rt = ttnn.RuntimeArgs()
        partner_writer_rt = ttnn.RuntimeArgs()
        chunk_tiles = bk * bn // num_chunks
        # A partner takes the second half of every chunk in whole slot rows (kw-fold: rw tiles) where a chunk holds two
        # or more, so its page runs stay whole.
        slot_row = max(1, bn // 3) if kwf else bn
        chunk_rows = chunk_tiles // slot_row if chunk_tiles % slot_row == 0 else 0
        split = (chunk_rows // 2) * slot_row if chunk_rows >= 2 else chunk_tiles // 2
        # Round 0's split with a partner: the first server's slot tiles of every chunk, one byte per chunk (see
        # WEIGHT_SERVER_R0_SHARE); 0 (one server): every round takes the args' part.
        r0_share = [split] * num_chunks
        if chunk_rows >= 2:
            num, den = WEIGHT_SERVER_R0_SHARE
            rows = (2 * num * num_chunks * chunk_rows + den) // (2 * den)
            r0_share = [
                (rows // num_chunks + (1 if c < rows % num_chunks else 0)) * slot_row for c in range(num_chunks)
            ]
        for rect, server, partner in weight_servers:
            lo = device.worker_core_from_logical_core(rect.start)
            hi = device.worker_core_from_logical_core(rect.end)
            n_receivers = rect.grid_size().x * rect.grid_size().y
            end = split if partner is not None else chunk_tiles
            ahead = num_chunks if partner is not None else 1
            r0_split = 0
            if partner is not None:
                assert num_chunks <= 4 and chunk_tiles <= 0xFF and all(0 < t <= chunk_tiles for t in r0_share)
                r0_split = sum(t << (8 * c) for c, t in enumerate(r0_share))
            server_reader_rt[server.x][server.y] = [weight_addr, num_rounds, 0, end, ahead, r0_split]
            # NoC1 multicast: from the rectangle's high corner to its low corner.
            fwd = 0
            if partner is not None:
                pv = device.worker_core_from_logical_core(partner)
                fwd = (pv.x | (pv.y << 8)) + 1
            server_writer_rt[server.x][server.y] = [
                num_rounds,
                n_receivers,
                hi.x,
                hi.y,
                lo.x,
                lo.y,
                bias_addr if server_bias else 0,
                0,
                end,
                1 if partner is not None else 0,
                fwd,
                r0_split,
            ]
            if partner is not None:
                partner_reader_rt[partner.x][partner.y] = [
                    weight_addr,
                    num_rounds,
                    split,
                    chunk_tiles,
                    ahead,
                    r0_split,
                ]
                # NoC0 multicast: from the low corner to the high corner.
                partner_writer_rt[partner.x][partner.y] = [
                    num_rounds,
                    n_receivers,
                    lo.x,
                    lo.y,
                    hi.x,
                    hi.y,
                    0,
                    split,
                    chunk_tiles,
                    2,
                    0,
                    r0_split,
                    weight_addr,
                ]
        server_cores = _core_range_set(servers)
        # The page-run parameters (conv2d_3x3_server_pages.hpp) and the weight accessor: the server reader's, and the
        # partner writer's for its part of round 0.
        server_pages_ct = [
            bk,
            bn,
            b.num_k_blocks,
            (1 + kw_fold_fused_gw(g.tensor_n_tiles)) if kwf else 0,
            g.tensor_n_tiles,
            *_weight_shard_tiles(weight_tensor, kwf or bn == g.tensor_n_tiles),
        ] + list(ttnn.TensorAccessorArgs(weight_tensor).get_compile_time_args())
        server_reader_ct = [CB_WEIGHTS, *server_pages_ct[:3], num_chunks, *server_pages_ct[3:]]
        server_kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "conv2d_3x3_weight_server_reader.cpp"),
                core_ranges=server_cores,
                compile_time_args=server_reader_ct,
                runtime_args=server_reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "conv2d_3x3_weight_server_writer.cpp"),
                core_ranges=server_cores,
                compile_time_args=[
                    CB_WEIGHTS,
                    bk * bn,
                    *server_sem_ids,
                    b.weight_depth,
                    num_chunks,
                    int(server_bias),
                    CB_BIAS,
                    g.tensor_n_tiles,
                ]
                + (
                    ttnn.TensorAccessorArgs(bias_tensor).get_compile_time_args()
                    if server_bias
                    else ttnn.TensorAccessorArgs().get_compile_time_args()
                ),
                runtime_args=server_writer_rt,
                defines=_round0_defines + slot_ack_defines,
                config=ttnn.WriterConfigDescriptor(),
            ),
        ]
        if partners:
            # The partners run the same two kernels with their NoCs swapped (reads on NoC1, multicast on NoC0) and no
            # bias; their ack semaphore counts the first server's forwarded clearances. A partner's writer reads its
            # part of round 0 itself, on NoC0 (SERVER_R0_SELF_READ; see WEIGHT_SERVER_R0_SHARE).
            partner_cores = _core_range_set(partners)
            server_kernels += [
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "conv2d_3x3_weight_server_reader.cpp"),
                    core_ranges=partner_cores,
                    compile_time_args=server_reader_ct,
                    runtime_args=partner_reader_rt,
                    defines=[("SERVER_R0_SELF_READ", "1")],
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1
                    ),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "conv2d_3x3_weight_server_writer.cpp"),
                    core_ranges=partner_cores,
                    compile_time_args=[CB_WEIGHTS, bk * bn, *server_sem_ids, b.weight_depth, num_chunks, 0, CB_BIAS, 0]
                    + list(ttnn.TensorAccessorArgs().get_compile_time_args())
                    + server_pages_ct,
                    runtime_args=partner_writer_rt,
                    defines=_round0_defines + [("SERVER_FWD_ACKS", "1"), ("SERVER_R0_SELF_READ", "1")],
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0
                    ),
                ),
            ]
    semaphores = [ttnn.SemaphoreDescriptor(id=sem_id, core_ranges=sem_cores, initial_value=0) for sem_id in sem_ids]
    descriptor = ttnn.ProgramDescriptor(kernels=[], semaphores=semaphores, cbs=cbs)
    # The writer's NoC (WriterConfigDescriptor).
    weight_mcast_config = ttnn.McastConfig(noc=ttnn.NOC.NOC_1)
    weight_mcast = None
    spare_mcast = None
    if g.k_split:
        if column_channel is not None:
            receivers, group_size, order = column_channel
            # Staggered senders: with every column's sender on one grid row, all the columns' DRAM read responses
            # share that row's NoC links, and the weight chain is bound there (weights-only chain on aspp.convs.1
            # @ LoFi: 65.5 us with the senders on row 0, 38.3 us staggered; whole op 72.1 -> 71.0 us, aspp.convs.2
            # 74.4 -> 69.8 us).
            sender_config = ttnn.McastFixedSenderConfig(sender_index=0, placement=ttnn.McastSenderPlacement.Staggered)
            weight_mcast = ttnn.Mcast(device, weight_mcast_config, receivers, group_size, sender_config, order)
        if spare_channel is not None:
            receivers, group_size, order, senders = spare_channel
            spare_mcast = ttnn.Mcast(
                device,
                weight_mcast_config,
                receivers,
                group_size,
                ttnn.McastExplicitFixedSenderConfig(senders),
                order,
            )
    else:
        if mcast_receivers is not None and weight_servers is None:
            weight_mcast = ttnn.Mcast(
                device,
                weight_mcast_config,
                mcast_receivers,
                mcast_receivers.num_cores(),
                ttnn.McastExplicitFixedSenderConfig([core_list[0]]),
            )
    if spread_rounds:
        # The spread stream's rounds go out raw, flagged on their own semaphores: no channel.
        weight_mcast = spare_mcast = None
    if weight_mcast is None:
        for writer in writers:
            ttnn.attach_absent_mcast(writer, WEIGHT_MCAST_PREFIX)
    else:
        weight_mcast.attach(descriptor, WEIGHT_MCAST_PREFIX, writers, len(semaphores))
    if spare_mcast is None:
        for writer in writers:
            ttnn.attach_absent_mcast(writer, WEIGHT_MCAST_SPARE_PREFIX)
    else:
        spare_mcast.attach(descriptor, WEIGHT_MCAST_SPARE_PREFIX, writers, weight_mcast.next_semaphore_id())
    descriptor.kernels = readers + writers + [compute] + server_kernels
    return descriptor
