# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm ProgramDescriptor: geometry, select_blocking(), CBs, kernels, per-core args.

Blocking (op_design.md -> Blocking Model):
  * row axis (tile-rows, leading dims folded): split over row-groups across the whole Tensix grid;
    each group's assignment is processed in blocks of `block_row_tiles` tile-rows.
  * W axis: split over `w_split_cores` (k) consecutive Tensix cores (one grid row while k <= grid_x, else
    row-major across grid rows); every member keeps its W share resident (cb_x) across pass 1 and pass 2,
    and the k members combine their per-row sum-of-squares partials over the NoC (k > 1 only): symmetric
    all-gather for k <= 8, per-row owner gather + inv_rms return beyond (Refinement 5).
Every knob below (k, block_row_tiles, x_block_depth, rm_chunk_w_tiles, out_unit_tiles) is computed
once here and handed to the kernels as a CT/RT arg; the kernels derive all trip counts from them.

HEIGHT / WIDTH / BLOCK_SHARDED (Refinements 2, 4): the shard spec fixes the work split — one group per shard row
(BLOCK / HEIGHT) or a single group (WIDTH), k = shard columns (HEIGHT: k = 1, no combine), members and their NoC coordinates from the shard
grid (orientation-aware, groups may span grid rows). Block = the resident shard (sub-blocked along rows
only under L1 pressure). TILE: cb_x / cb_out are backed on the input / output shards (zero-copy). RM:
the resident sticks are copied locally into the tilize staging CB (never gathered across Tensix cores).
Combine topology: k <= K_SYM_MAX_* symmetric unicast all-gather; beyond, a per-row owner gathers the
k partials, combines + finalizes, and returns inv_rms to every member (L1 per member ~ k * ceil(b/k)).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, replace
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32

# CB indices — must match kernels/rms_norm_common.hpp.
CB_X_RM = 0  # RM: raw stick pieces (tile-sized pages) -> tilize
CB_X = 1  # resident x block (ping)
CB_GAMMA = 2  # gamma slice (faces 0/1 pages, row-broadcast by compute), program lifetime
CB_SUMSQ = 3  # per tile-row DEST-accumulated sum of x^2 (pass-1 output)
CB_PARTIAL = 4  # k > 1: this member's per-row partial (column 0)
CB_GATHERED = 5  # k > 1: k partials per row, all-gathered over the NoC
CB_INV_RMS = 6  # per tile-row rsqrt(mean + eps) (column 0)
CB_OUT = 7  # normalized tiles (TILE: -> writer; RM: -> untilize)
CB_OUT_RM = 8  # RM: untilized stick pieces -> writer
CB_X_ALT = 9  # resident x block (pong) — only when x_block_depth == 2
CB_TAIL_MASK = 11  # sharded TILE W tail: 0/1 row mask of the valid columns of the last W tile
CB_INV_OWNED = 12  # owner combine (k > K_SYM_MAX_*): inv_rms of the tile-rows this member owns
CB_ONES = 13  # 1.0 reduce scaler tile of the FPU row collapse (reduce_tile ROW)

SEM_PARTIALS_ARRIVED = 0
SEM_INV_ARRIVED = 1  # owner combine: inv_rms tiles returned by the row owners
SEM_GAMMA_MCAST_FIRST = 2  # gamma multicast channel (ttnn.Mcast allocates its semaphores from this id)
# Combine pipeline: odd-sub-block partials / inv_rms semaphores. Only on sharded k > 1 owner combine, where neither
# ttnn.Mcast user (gamma: sharded k = 1 only; inv_rms: off on the pipelined path) allocates ids >= 2 — asserted below.
SEM_PARTIALS_PAR = 2
SEM_INV_PAR = 3
WRITER_NOC_XY_ARG_BASE = 13  # writer RT index of the first peer (x, y) pair — passed to the kernel as a CT arg

# Knob constants (design Phase 0 values).
# Interleaved W split cap: the writer carries one (x, y) NoC pair per member as runtime args, so k is bounded by
# the per-kernel runtime-arg limit (tt-metalium kernel_types.hpp max_runtime_args = 341). Beyond K_SYM_MAX_INTERLEAVED
# the owner combine runs; beyond grid_x a group spans grid rows (members placed row-major over the grid).
MAX_RUNTIME_ARGS = 341
K_CAP_MAX = (MAX_RUNTIME_ARGS - WRITER_NOC_XY_ARG_BASE) // 2
# Combine topology: symmetric all-gather up to K_SYM_MAX_* members, per-row owner gather + inv_rms return beyond.
# Interleaved (W split k <= 8): catalog tensix_all_reduce (k <= 8 unicast all-gather). Sharded: the owner
# topology for every k > 1 — measured BH p150b (bf16, HiFi2, 16-bit DEST): BLOCK (7168,1024) [736,128] 8x10
# 70.8 -> 32.3 us, (8192,1024) [1024,128] 8x8 96.7 -> 42.7 us (each member combines + finalizes rows/k
# tile-rows instead of all of them); 1- and 2-tile-row shards flat (<= 3%).
K_SYM_MAX_INTERLEAVED = 8
K_SYM_MAX_SHARDED = 1
# Interleaved: the Tensix cores sharing a W slice get gamma by one multicast (one DRAM read per slice) instead
# of each reading it (the writer loads it, off the reader's x stream). BH p150b, perf config: the per-core load
# was a 131 us serial prefix of (8192,7168) (130 Tensix cores reading the same 224 gamma tiles, 2 x 64 B reads
# each); measured 771 -> 630 us (8192,7168), 90.8 -> 76.3 us (7168,1024), (640,6144) k=6 51.9 -> 45.7 us.
GAMMA_MCAST = True
# Sharded k = 1 (HEIGHT, or a single shard column): every Tensix core holds the whole gamma, so the same one-group
# multicast applies (one DRAM read of gamma per launch instead of one per Tensix core). WIDTH / BLOCK k > 1 keep the
# per-core slice load (a shard grid column need not hold a single W slice).
GAMMA_MCAST_SHARDED_K1 = True
# Owner combine, single-owner blocks (one tile-row: member 0 owns it): return inv_rms to the k members with ONE
# multicast (kernel_lib Mcast, fixed sender = member 0, no handshake) instead of k unicasts + k semaphore increments.
# BH p150b perf cfg: WIDTH [32,256] 7x4 k = 28 8.9 -> 7.6 us, [32,160] 8x4 k = 32 8.2 -> 6.6 us, (32,32768) k = 49
# 1.05x. Blocks with >= 2 owners keep the unicast return (multicast measured 1.7-2.1x slower there: owners collide).
INV_MCAST = True
MAX_MCAST_RECTANGLES = 3  # kernel_lib mcast_protocol.hpp: receiver rectangles per multicast group
# k > 1, per-core gamma load (no multicast): the writer issues the gamma DRAM reads at kernel start but finishes the
# load (barrier [+ RM face-1 copy]) only after block 0's partials are sent / its owner duties are done, so the cross-core
# combine no longer waits on it. Pays when pass 1 is short (few tile-rows per block): BH p150b, perf config,
# blocks of 1-2 tile-rows — (1,1,32,32768) k=49 24.3 -> 20.8 us, (160,11008) k=25 29.4 -> 24.8 us, (352,4096)
# 25.9 -> 24.2 us, WIDTH (32,7168) 9.3 -> 8.3 us. With long blocks the early load is already hidden behind pass 1
# and the late finish only delays pass 2: BLOCK (7168,1024) [736,128] (23 tile-rows) 29.9 -> 30.8 us. Gated on
# block_row_tiles <= GAMMA_AFTER_SEND_MAX_ROWS.
GAMMA_AFTER_SEND = True
GAMMA_AFTER_SEND_MAX_ROWS = 8
# Combine signal fence: flush (writes left this Tensix core) instead of a write barrier before each semaphore
# increment of the partial exchange / inv_rms return (catalog tensix_all_reduce, design perf lamp "Signal fence").
SIGNAL_AFTER_FLUSH = True
PARTIAL_FACES_ONLY = True  # ship faces 0/2 of a partial / inv_rms tile (column 0 lives there), not the whole tile
# Combine pipeline (sharded TILE, owner combine, resident gamma): cut the resident shard's crt tile-rows into
# n sub-blocks of s = ceil(crt / n) tile-rows and run compute as front(t+1), combine(t), apply(t-1) against a
# two-round exchange window in the writer (parity semaphores), so the cross-core exchange / owner combine / inv_rms
# return of a sub-block overlaps pass 1 + collapse and pass 2 of its neighbours. Row owners rotate per sub-block
# (row r of t owned by (t * s + r) % k). Bit-exact with the one-block path.
# Sub-block count n (the most, from the top, that passes the gates; _pipe_subblock_rows), w = shard width in tiles:
#   n = COMBINE_PIPELINE_SUBBLOCKS (3) — 2 and 4..8 measured slower on w >= 4 (BLOCK [736,128] / [1024,128] /
#       [736,192], WIDTH (1024,1000) [1024,128]);
#   w == 1: n = max(3, ceil(crt / max(COMBINE_PIPELINE_NARROW_MAX_ROWS, k))) — one-tile-wide shards have too little
#       work per tile-row to hide the exchange behind 3 rounds (WIDTH (2048,256) [2048,32] k=8: n3 33.8 -> n8 31.4 us),
#       but never fewer than k tile-rows per sub-block (WIDTH (2048,1024) [2048,32] k=32: n8 70.1 vs n3 47.6 us — only
#       s of the k members own a row per round).
# Gates (measured regressions otherwise):
#   s >= COMBINE_PIPELINE_MIN_ROWS and s * (w + 1) >= COMBINE_PIPELINE_MIN_WORK — a round's fixed exchange latency
#       must hide behind its sub-block ((640,6144) [64,704] s=1: 19.5 -> 20.8 us; BLOCK (2048,256) w=1 s=2: 1.07x,
#       s=3: 0.96x, s=4: 0.90x);
#   k <= COMBINE_PIPELINE_K_PER_WORK * s * (w + 1) — each owner's k-way inv_rms return per round must hide behind one
#       sub-block of work ((256,2048) WIDTH auto k=64, w=1, s=3: 26.0 -> 30.8 us).
# GAMMA_AFTER_SEND applies per sub-block (s <= GAMMA_AFTER_SEND_MAX_ROWS: finished after sub-block 0's inv_rms return):
# RM gamma (one DRAM page read by every Tensix core) BLOCK (2048,1024) auto 12.6 -> 10.3 us vs finishing it at kernel
# start; TILE gamma within ~2 %.
# The single-owner inv_rms multicast (INV_MCAST) is off on the pipelined path: the rotated owner of a one-tile-row
# sub-block need not be member 0, the multicast's fixed sender.
COMBINE_PIPELINE = True
COMBINE_PIPELINE_SUBBLOCKS = 3
COMBINE_PIPELINE_NARROW_MAX_ROWS = 8  # w == 1: s <= max(this, k)
COMBINE_PIPELINE_MIN_WORK = 8  # s * (w + 1) floor (x-tile equivalents of pass 1 + pass 2 per sub-block)
COMBINE_PIPELINE_MIN_ROWS = 2
COMBINE_PIPELINE_K_PER_WORK = 2
OUT_DEPTH = 2  # cb_out / cb_out_rm buffer depth
# Streamed gamma (fallback when the resident gamma slice does not fit L1 — e.g. HEIGHT k = 1 with a wide W
# beside the two resident shards): gamma flows through a GAMMA_STREAM_DEPTH-chunk ring of <= GAMMA_STREAM_CHUNK_CAP
# tiles, re-loaded per (block, chunk). The largest chunk that fits is chosen.
GAMMA_STREAM_CHUNK_CAP = 16
GAMMA_STREAM_DEPTH = 2
GAMMA_STREAM_FORCE_CHUNK = 0  # > 0: stream gamma in chunks of this many tiles even when resident fits (test hook)
RM_IN_DEPTH = 2  # cb_x_rm buffer depth
OUT_UNIT_CAP = 16  # TILE writer batch (tiles per wait/flush/pop)
RM_CHUNK_CAP = 32  # balanced RM streaming chunk cap (tiles)
RM_PIECE_FLOOR_BYTES = 512  # occupancy split must keep RM stick pieces >= this
OCC_TILES_PER_MEMBER = 4  # occupancy split keeps >= 4 W tiles per member
COMBINE_TILE_COST = 3  # k > 1: per-tile-row collapse + exchange + combine cost, in x-tile equivalents
OWNER_MEMBER_COST = 0.4  # owner combine: per gathered partial on the owner (add + return write), x-tile equivalents
# Whole assignment fits: split it into this many depth-2 blocks (1 = one block per core). Measured 2 on BH p150b
# (perf config): (8192,7168) 630 -> 673 us, (7168,1024) 76 -> 81 us — the k = 1 prefill shapes are DRAM-write-
# bound (compute stubbed: same time; write-only 438 vs read-only 313 us at 8192x7168), so 1 stays the default.
OVERLAP_BLOCKS = 1
# Perf-probe ablation defines (never set by the op): e.g. ("RMS_ABLATE_EMPTY", "RMS_ABLATE_DM",
# "CKL_ELTWISE_CHAIN_SKIP_COMPUTE"). Stubs a stage's payload, keeps its synchronization (/perf-measure).
PERF_ABLATE = ()
# NoC pairing (Perf 1, dram_noc_balance): interleaved inputs only. Under the usual pairing (reader NoC0 / writer NoC1)
# the NoC arbitration starves part of the grid (BH p150b, (8192,7168): rd_read 15..545 us for equal work per Tensix
# core). The Tensix cores in the NOC_SWAP_COLUMNS logical worker columns nearest DRAM bank 0's column (asked of the
# device: get_optimal_dram_bank_to_logical_worker_assignment) run the reversed pairing (reader NoC1 / writer NoC0),
# which moves part of the read stream off NoC0 (catalog noc_placement `swap_columns`). Every multicast sender keeps
# the usual pairing (a sender's writer must issue on its channel's NoC — kernel_lib Mcast static_asserts it — and a
# gamma read + multicast on the read-heavy NoC0 measured 10-22 us to issue): the inv_rms senders (member 0 of each
# owner group) and the k > 1 gamma column senders stay unswapped; k = 1 sends gamma from the first unswapped Tensix
# core. Bit-identical output. BH p150b, see perf_experiments/dram_noc_balance/port. 0 = usual pairing everywhere.
NOC_SWAP_COLUMNS = 3


# HEIGHT_SHARDED ROW_MAJOR "flat" regime (Perf 2, perf_experiments/height_rm_flat; kernels rms_norm_flat_*.cpp): a
# stick of W datums is contiguous in the shard, so it is normalized as ceil(W / 1024) flat tiles (1024 consecutive
# datums each) — sum x^2 = DEST-accumulated x_j * x_j + one FPU REDUCE_SCALAR + finalize; out_j = x_j * inv (bcast
# scalar) * gamma_j, gamma_j = flat tile j of the RM gamma page (resident: W * elem bytes, one DRAM read + multicast).
# cb_x / cb_out are backed on the shards (page = gcd(stick pitch, tile bytes)); a partial last flat tile per stick
# (W % 1024 != 0) is staged through a zero-filled tail tile. No tilize / untilize / 32x stick padding / gamma stream.
# BH p150b: (1,1,128,4096) fp32 1 stick / core 93.3 -> 3.3 us, (1,224,11008) 2 sticks 121.6 -> 6.9 us,
# (3104,4064) 24 sticks 43.2 -> 29.5 us. Gate (cost model, bf16 / fp32 DEST off / HiFi4 calibration; fp32 and fp32
# DEST only make the tilized path slower, so the model is conservative there):
#   flat    ~ FLAT_FIXED_NS + S * (FLAT_STICK_NS + FLAT_TILE_NS * ceil(W / 1024) + FLAT_TAIL_NS * [W % 1024 != 0])
#   tilized ~ ceil(S / 32) * (TILIZED_ROW_NS + TILIZED_WTILE_NS * ceil(W / 32))      (S = sticks per Tensix core)
# Crossover: W = 256: S ~ 7; 512: ~13; 1024: ~29; 2048: ~37; >= 4096: >= 64. Also used when the tilized plan does not
# fit L1 and the flat one does (fp32 [32, 4096] shards).
RM_FLAT = True
RM_FLAT_FORCE = False  # test hook: every expressible + L1-fitting HEIGHT RM input takes the flat path
FLAT_FIXED_NS = 1200
FLAT_STICK_NS = 150  # per stick: REDUCE_SCALAR + one-vector finalize + pack
FLAT_TILE_NS = 230  # per flat tile: x*x (pass 1) + x*inv*gamma (pass 2)
FLAT_TAIL_NS = 80  # per stick with a partial last flat tile (tail staging copy + second CB)
TILIZED_ROW_NS = 2300  # per tile-row of the tilized path
TILIZED_WTILE_NS = 310  # per W tile of a tile-row
FLAT_MAX_CB_PAGES = 32768  # zero-copy cb_x / cb_out pages per Tensix core (16-bit CB page counters)


def _f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass
class Geometry:
    W: int
    H: int
    num_images: int
    image_row_tiles: int
    tensor_row_tiles: int
    tensor_w_tiles: int
    w_valid_in_last_tile: int


def derive_geometry(shape) -> Geometry:
    shape = [int(d) for d in shape]
    W, H = shape[-1], shape[-2]
    num_images = 1
    for d in shape[:-2]:
        num_images *= d
    image_row_tiles = _div_up(H, TILE)
    tensor_w_tiles = _div_up(W, TILE)
    return Geometry(
        W=W,
        H=H,
        num_images=num_images,
        image_row_tiles=image_row_tiles,
        tensor_row_tiles=num_images * image_row_tiles,
        tensor_w_tiles=tensor_w_tiles,
        w_valid_in_last_tile=W - TILE * (tensor_w_tiles - 1),
    )


def acc_dtype(fp32_dest_acc_en: bool):
    """Format of every DEST-written intermediate CB (sumsq / partial / gathered / inv_rms).

    Follows the DEST width: with 16-bit DEST the value is already rounded before pack_tile(), so a Float32
    page would carry no extra bits while doubling packer/unpacker/NoC bytes and L1.
    """
    return ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16


def _gamma_page_bytes(gamma_dtype) -> int:
    """cb_gamma page bytes: the bfp exponent section (if any) + faces 0/1 (half the datum area) of a gamma tile."""
    elem = 1 if gamma_dtype == ttnn.bfloat8_b else ttnn.tile_size(gamma_dtype) // (TILE * TILE)
    exp = ttnn.tile_size(gamma_dtype) - TILE * TILE * elem
    return exp + (TILE * TILE * elem) // 2


class _Sizes:
    def __init__(self, x_dtype, gamma_dtype, is_rm, has_gamma, fp32_dest_acc_en):
        self.T_in = ttnn.tile_size(x_dtype)
        self.T_out = self.T_in
        # cb_gamma page: [bfp exponent section +] faces 0/1 of a gamma tile — compute reads gamma ROW-broadcast
        # (row 0 of faces 0/1 only), so faces 2/3 are never stored.
        self.T_g = _gamma_page_bytes(gamma_dtype) if has_gamma else 0
        self.T_acc = ttnn.tile_size(acc_dtype(fp32_dest_acc_en))
        self.is_rm = is_rm
        self.has_gamma = has_gamma


def _normalize_k(tw: int, k: int):
    s = _div_up(tw, k)
    return _div_up(tw, s), s


def _out_unit(slice_w_tiles: int) -> int:
    """TILE writer batch (tiles per wait/write/flush/pop) — also the cb_out depth unit."""
    return min(slice_w_tiles, OUT_UNIT_CAP)


def _rm_chunk(slice_w_tiles: int, cap: int = RM_CHUNK_CAP):
    chunks = _div_up(slice_w_tiles, cap)
    c = _div_up(slice_w_tiles, chunks)
    return c, chunks * c


@dataclass(frozen=True)
class PlanOpts:
    """Regime switches that change the CB inventory (one source of truth for footprint() and the CB list)."""

    shard_backed: bool = False  # TILE sharded: cb_x / cb_out alias the shards (tensor buffers, not CB budget)
    owner_combine: bool = False  # k > K_SYM_MAX_*: per-row owner gather + return instead of the all-gather
    gather_depth: int = 2  # symmetric all-gather: blocks of cb_gathered (1 when every core runs one block)
    mask_tail: bool = False  # sharded TILE, W % 32 != 0: tail tile's x^2 masked inside pass 1 (0/1 row mask)
    gamma_chunk: int = 0  # 0: gamma slice resident (program lifetime); G > 0: streamed in G-tile chunks (ring)
    rm_chunk_cap: int = RM_CHUNK_CAP  # RM: cap of the streaming chunk C (lowered only when the plan does not fit)
    pipe_nb: int = 0  # combine pipeline: sub-blocks per Tensix core (0: off); b is then the sub-block (s tile-rows)


def cb_plan(sz: _Sizes, k: int, slice_w: int, b: int, d: int, opts: PlanOpts = PlanOpts()) -> dict:
    """Single source of truth for every CB's capacity: {cb_index: (pages, page_bytes)}.

    Both footprint() (the L1-fit solve in select_blocking) and create_program_descriptor() (the CB
    declarations) read this dict, so a knob change lands in exactly one place (l1_ledger.md rows).
    """
    if sz.is_rm:
        c, pitch = _rm_chunk(slice_w, opts.rm_chunk_cap)
    else:
        c, pitch = 1, slice_w
    plan = {
        CB_SUMSQ: (b, sz.T_acc),
        # combine pipeline: a slot is rewritten only after its apply (lag 1 -> 3 sub-blocks in flight)
        CB_INV_RMS: (min(3, opts.pipe_nb) * b if opts.pipe_nb else b, sz.T_acc),
        CB_ONES: (1, ttnn.tile_size(ttnn.bfloat16)),
    }
    if not opts.shard_backed:
        plan[CB_X] = (b * pitch, sz.T_in)  # resident x block
        if d == 2:
            plan[CB_X_ALT] = (b * pitch, sz.T_in)  # second half of the depth-2 x budget
    if sz.has_gamma:
        plan[CB_GAMMA] = (GAMMA_STREAM_DEPTH * opts.gamma_chunk if opts.gamma_chunk else pitch, sz.T_g)
    if k > 1:
        pipe = 2 if opts.pipe_nb else 1  # combine pipeline: front(t+1) precedes send(t); two-round exchange window
        plan[CB_PARTIAL] = (pipe * b, sz.T_acc)
        if opts.owner_combine:
            owned = _div_up(b, k)  # tile-rows a member owns per block (row r -> member (r + ofs) % k)
            plan[CB_GATHERED] = (pipe * owned * k, sz.T_acc)  # k partials per owned row (uniform -> lockstep slots)
            plan[CB_INV_OWNED] = (owned, sz.T_acc)
        else:
            plan[CB_GATHERED] = (opts.gather_depth * k * b, sz.T_acc)  # lockstep slot-reuse mechanism cap
    if opts.mask_tail:
        plan[CB_TAIL_MASK] = (1, sz.T_acc)
    if sz.is_rm:
        plan[CB_X_RM] = (RM_IN_DEPTH * c, sz.T_in)
        plan[CB_OUT] = (b * c, sz.T_out)  # spans one chunk column of the block (untilize input)
        plan[CB_OUT_RM] = (OUT_DEPTH * c, sz.T_out)
    elif not opts.shard_backed:
        plan[CB_OUT] = (OUT_DEPTH * _out_unit(slice_w), sz.T_out)
    return plan


def footprint(sz: _Sizes, tw: int, k: int, b: int, d: int, opts: PlanOpts = PlanOpts()) -> int:
    """Per-Tensix-core CB bytes — the l1_ledger.md expression, summed from cb_plan()."""
    k, s = _normalize_k(tw, k)
    return _plan_bytes(cb_plan(sz, k, s, b, d, opts))


def _plan_bytes(plan: dict) -> int:
    return sum(pages * page_bytes for pages, page_bytes in plan.values())


@dataclass
class Member:
    """One Tensix core's share of the work: a run of tile-rows x a W slice."""

    core: object  # logical ttnn.CoreCoord
    m: int  # member index inside its group (W-slice order)
    row_start: int  # first global tile-row (interleaved addressing; sharded: informational)
    core_rows: int  # tile-rows on this core
    w_tile_start: int  # first global W tile
    core_w: int  # W tiles this core reduces (valid, not padding)
    col0: int  # first global column (datums)
    valid_cols: int  # valid columns of this slice (datums)
    valid_sticks: int = 0  # RM sharded: valid sticks (rows) in the resident shard


@dataclass
class Blocking:
    k: int  # w_split_cores (members per group)
    slice_w_tiles: int  # W tiles of every non-last member
    rm_chunk_w_tiles: int  # C (RM streaming quantum along W; TILE: unused, = 1)
    x_row_pitch: int  # tile pitch of one tile-row inside cb_x (uniform across cores)
    groups: list  # per group: list of k Member
    block_row_tiles: int
    x_block_depth: int
    out_unit_tiles: int
    opts: PlanOpts
    footprint: int


def _pitch(sz: _Sizes, slice_w: int, rm_chunk_cap: int = RM_CHUNK_CAP):
    """(rm_chunk_w_tiles, x_row_pitch) for a slice of `slice_w` tiles."""
    return _rm_chunk(slice_w, rm_chunk_cap) if sz.is_rm else (1, slice_w)


def _solve_rows(
    sz: _Sizes,
    k: int,
    slice_w: int,
    crt_max: int,
    budget: int,
    opts: PlanOpts,
    allow_depth2: bool,
    overlap_blocks: int = 1,
):
    """Largest block_row_tiles (and x_block_depth) whose CB plan fits `budget`.

    When the whole assignment fits: overlap_blocks > 1 (and depth 2 allowed) splits it into ceil(crt_max /
    overlap_blocks)-row blocks at depth 2, so the read of block b+1 overlaps compute + write of block b
    (design perf lamp "Overlap (one-block assignment)"); overlap_blocks == 1 keeps one block per core.
    Otherwise the largest sub-block at depth 2 (TILE interleaved) / 1. The symmetric all-gather needs a
    2-block cb_gathered only when a core runs > 1 block.
    """

    def fits(b, d):
        single = b >= crt_max
        o = replace(opts, gather_depth=1 if single else 2)
        return _plan_bytes(cb_plan(sz, k, slice_w, b, d, o)) <= budget, o

    if allow_depth2 and overlap_blocks > 1 and crt_max > 1:
        b_ov = _div_up(crt_max, min(overlap_blocks, crt_max))
        ok, o = fits(b_ov, 2)
        if ok:
            return b_ov, 2, o
    ok, o = fits(crt_max, 1)
    if ok:
        return crt_max, 1, o
    d = 2 if allow_depth2 else 1
    for cand in range(crt_max, 0, -1):
        ok, o = fits(cand, d)
        if ok:
            return cand, (1 if cand >= crt_max else d), o
    ok, o = fits(1, 1)
    if not ok:
        raise RuntimeError(f"rms_norm: one tile-row block does not fit L1 ({budget} B per core)")
    return 1, 1, o


def _sharded_plan_candidates(sz: _Sizes, slice_w: int, opts: PlanOpts):
    """Sharded CB regimes in preference order; the first whose one-tile-row plan fits L1 is used.

    Default: resident gamma slice (loaded / multicast once) and the RM chunk at RM_CHUNK_CAP. Wide W beside two
    resident shards may not fit that (HEIGHT k = 1 holds the whole row), so fall back to
      TILE: gamma streamed in the largest G <= GAMMA_STREAM_CHUNK_CAP chunks (per block, 2-chunk ring);
      RM  : per halved chunk cap C (staging / cb_out / untilize buffers scale with C): resident gamma, then gamma
            streamed with G = C (pass 2 already walks C-tile windows). The tilized x row stays resident.
    GAMMA_STREAM_FORCE_CHUNK (test hook) skips the resident-gamma candidates (TILE: also caps G).
    """
    force = bool(GAMMA_STREAM_FORCE_CHUNK) and sz.has_gamma
    if not sz.is_rm:
        if not force:
            yield opts
        if sz.has_gamma:
            for g in range(min(GAMMA_STREAM_FORCE_CHUNK or GAMMA_STREAM_CHUNK_CAP, slice_w), 0, -1):
                yield replace(opts, gamma_chunk=g)
        return
    cap = RM_CHUNK_CAP
    while True:
        o = replace(opts, rm_chunk_cap=cap)
        if not force:
            yield o
        if sz.has_gamma:
            yield replace(o, gamma_chunk=_rm_chunk(slice_w, cap)[0])
        if cap == 1:
            return
        cap //= 2


def _solve_rows_sharded(sz: _Sizes, k: int, slice_w: int, crt_max: int, budget: int, opts: PlanOpts):
    """_solve_rows over _sharded_plan_candidates(): the first regime in which one tile-row block fits."""
    for cand in _sharded_plan_candidates(sz, slice_w, opts):
        try:
            return _solve_rows(sz, k, slice_w, crt_max, budget, cand, allow_depth2=False)
        except RuntimeError:
            continue
    raise RuntimeError(
        f"rms_norm: one tile-row block does not fit L1 ({budget} B per core), even with streamed gamma / RM chunk 1"
    )


def _max_groups(k: int, grid_x: int, grid_y: int) -> int:
    """Row-groups a W split k fits on the grid: k <= grid_x keeps each group inside one grid row (grid_x // k per
    row — a grid column then holds a single W slice, which the gamma multicast relies on); k > grid_x packs the
    groups row-major over the whole grid (a group spans grid rows)."""
    return (grid_x // k) * grid_y if k <= grid_x else (grid_x * grid_y) // k


def _occ_cost(trt: int, tw: int, k: int, grid_x: int, grid_y: int) -> float:
    """Per-core critical-path work of a W split k, in x-tile equivalents.

    Rows of the busiest group x (its slice width + the per-tile-row cost of the k > 1 collapse / exchange /
    combine). The owner combine (k > K_SYM_MAX_INTERLEAVED) adds, on the busiest owner, ceil(rows / k) owned rows
    x k gathered partials (one combine add + one inv_rms return write each): with few rows it grows with k.
    """
    groups = min(_max_groups(k, grid_x, grid_y), trt)
    rows = _div_up(trt, groups)
    cost = rows * (_div_up(tw, k) + (COMBINE_TILE_COST if k > 1 else 0))
    if k > K_SYM_MAX_INTERLEAVED:
        cost += _div_up(rows, k) * k * OWNER_MEMBER_COST
    return cost


def _choose_k(
    geo: Geometry, sz: _Sizes, grid_x: int, grid_y: int, k_fit: int, k_cap: int, elem_bytes: int, budget: int
) -> int:
    """w_split_cores: the k >= k_fit that minimizes the busiest Tensix core's work (catalog width_split).

    A split is a candidate only while every member keeps >= OCC_TILES_PER_MEMBER W tiles (RM: stick pieces
    >= RM_PIECE_FLOOR_BYTES). Ties keep the smaller k (fewer combine members).
    """
    tw, trt = geo.tensor_w_tiles, geo.tensor_row_tiles
    best_k, best = k_fit, _occ_cost(trt, tw, _normalize_k(tw, k_fit)[0], grid_x, grid_y)
    if tw < 2 * OCC_TILES_PER_MEMBER:
        return best_k
    for cand in range(k_fit + 1, k_cap + 1):
        kk, s = _normalize_k(tw, cand)
        if s < OCC_TILES_PER_MEMBER:
            break
        if sz.is_rm and s * TILE * elem_bytes < RM_PIECE_FLOOR_BYTES:
            break
        if _max_groups(kk, grid_x, grid_y) < 1:
            break
        if footprint(sz, tw, kk, 1, 1, PlanOpts(owner_combine=kk > K_SYM_MAX_INTERLEAVED)) > budget:
            continue  # the owner's k gathered partials outgrow L1
        c = _occ_cost(trt, tw, kk, grid_x, grid_y)
        if c < best:
            best_k, best = kk, c
    return best_k


def select_blocking(geo: Geometry, sz: _Sizes, grid_x: int, grid_y: int, budget: int, elem_bytes: int) -> Blocking:
    """INTERLEAVED: row-groups over the whole grid x a W split over k consecutive Tensix cores (one grid row while
    k <= grid_x, else row-major across grid rows)."""
    k_cap = min(K_CAP_MAX, grid_x * grid_y)
    tw, trt = geo.tensor_w_tiles, geo.tensor_row_tiles

    k_fit = None
    for k in range(1, k_cap + 1):
        if footprint(sz, tw, k, 1, 1) <= budget:
            k_fit = k
            break
    if k_fit is None:
        raise RuntimeError(
            f"rms_norm: W={geo.W} too wide — even w_split_cores={k_cap} does not fit L1 ({budget} B per core)"
        )

    k, slice_w = _normalize_k(tw, _choose_k(geo, sz, grid_x, grid_y, k_fit, k_cap, elem_bytes, budget))
    c, pitch = _pitch(sz, slice_w)

    num_groups = min(_max_groups(k, grid_x, grid_y), trt)
    base, rem = divmod(trt, num_groups)

    def core_of(g, m):
        if k <= grid_x:
            gpr = grid_x // k
            return ttnn.CoreCoord((g % gpr) * k + m, g // gpr)
        idx = g * k + m
        return ttnn.CoreCoord(idx % grid_x, idx // grid_x)

    groups, start = [], 0
    for g in range(num_groups):
        n = base + (1 if g < rem else 0)
        members = []
        for m in range(k):
            w_start = m * slice_w
            core_w = min(slice_w, tw - w_start)
            col0 = w_start * TILE
            members.append(Member(core_of(g, m), m, start, n, w_start, core_w, col0, min(core_w * TILE, geo.W - col0)))
        groups.append(members)
        start += n
    crt_max = base + (1 if rem else 0)

    b, d, opts = _solve_rows(
        sz,
        k,
        slice_w,
        crt_max,
        budget,
        PlanOpts(owner_combine=k > K_SYM_MAX_INTERLEAVED),
        allow_depth2=not sz.is_rm,
        overlap_blocks=1 if sz.is_rm else OVERLAP_BLOCKS,
    )
    return Blocking(
        k=k,
        slice_w_tiles=slice_w,
        rm_chunk_w_tiles=c,
        x_row_pitch=pitch,
        groups=groups,
        block_row_tiles=b,
        x_block_depth=d,
        out_unit_tiles=_out_unit(slice_w),
        opts=opts,
        footprint=_plan_bytes(cb_plan(sz, k, slice_w, b, d, opts)),
    )


def _pipe_subblock_rows(crt: int, k: int, w: int) -> int:
    """Combine pipeline sub-block s (tile-rows) for a crt-tile-row shard of w tiles in a k-member group; 0: off.

    The most sub-blocks n in [2, n_max] whose s = ceil(crt / n) passes the gates (see COMBINE_PIPELINE): s >= MIN_ROWS,
    s * (w + 1) >= MIN_WORK, k <= K_PER_WORK * s * (w + 1). n_max = SUBBLOCKS, raised on one-tile-wide shards (w == 1)
    to ceil(crt / max(NARROW_MAX_ROWS, k)).
    """
    nb = COMBINE_PIPELINE_SUBBLOCKS
    if w == 1:
        nb = max(nb, _div_up(crt, max(COMBINE_PIPELINE_NARROW_MAX_ROWS, k)))
    for n in range(nb, 1, -1):
        s = _div_up(crt, n)
        work = s * (w + 1)
        if (
            _div_up(crt, s) > 1
            and s >= COMBINE_PIPELINE_MIN_ROWS
            and work >= COMBINE_PIPELINE_MIN_WORK
            and k <= COMBINE_PIPELINE_K_PER_WORK * work
        ):
            return s
    return 0


def _ordered_shard_cores(grid, row_major: bool):
    """Cores of a shard grid in shard-assignment order (row-major or column-major)."""
    cores = set()
    for r in grid.ranges():
        for y in range(r.start.y, r.end.y + 1):
            for x in range(r.start.x, r.end.x + 1):
                cores.add((x, y))
    return sorted(cores, key=(lambda c: (c[1], c[0])) if row_major else (lambda c: (c[0], c[1])))


def select_sharded_blocking(geo: Geometry, sz: _Sizes, input_tensor, budget: int) -> Blocking:
    """HEIGHT / WIDTH / BLOCK_SHARDED: the shard spec IS the work split; block = the resident shard.

    Groups = shard rows (WIDTH: one), k = shard columns (HEIGHT: one — local reduction, no combine). Member -> (slice, core) from the shard grid in its
    orientation: WIDTH walks the grid's cores in order (a group may span grid rows); BLOCK puts shard
    (row g, column i) at (x0 + i, y0 + g) for ROW_MAJOR, (x0 + g, y0 + i) for COL_MAJOR. Ragged last shard
    row / column (Split.PAD): the members own only their valid tile-rows / W tiles.
    """
    mc = input_tensor.memory_config()
    ss = mc.shard_spec
    sh, sw = int(ss.shape[0]), int(ss.shape[1])
    row_major = ss.orientation == ttnn.ShardOrientation.ROW_MAJOR
    if sz.is_rm:
        rows_ext, unit_h = geo.num_images * geo.H, sh  # sticks
        w_ext, unit_w = geo.W, sw  # datums
        slice_w = _div_up(sw, TILE)
    else:
        if sh % TILE or sw % TILE:
            raise RuntimeError(f"rms_norm: TILE shard {sh}x{sw} is not a whole number of tiles")
        rows_ext, unit_h = geo.tensor_row_tiles, sh // TILE  # tile-rows
        w_ext, unit_w = geo.tensor_w_tiles, sw // TILE  # tiles
        slice_w = unit_w
    n_h, n_w = _div_up(rows_ext, unit_h), _div_up(w_ext, unit_w)

    if mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        if n_h != 1:
            raise RuntimeError(f"rms_norm: WIDTH shard height {sh} does not cover the tensor")
        order = _ordered_shard_cores(ss.grid, row_major)
        if len(order) < n_w:
            raise RuntimeError(f"rms_norm: {n_w} W shards but only {len(order)} shard-grid cores")

        def core_of(g, i):
            return order[i]

    elif mc.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        # k = 1: every Tensix core holds whole rows (the reduction is local, no combine). Shard row g lives on the
        # g-th core of the grid in its orientation (a multi-range grid, e.g. full rows + a partial last row).
        if n_w != 1:
            raise RuntimeError(f"rms_norm: HEIGHT shard width {sw} does not cover the tensor")
        order = _ordered_shard_cores(ss.grid, row_major)
        if len(order) < n_h:
            raise RuntimeError(f"rms_norm: {n_h} row shards but only {len(order)} shard-grid cores")

        def core_of(g, i):
            return order[g]

    else:
        bb = ss.grid.bounding_box()
        x0, y0 = bb.start.x, bb.start.y

        def core_of(g, i):
            return (x0 + i, y0 + g) if row_major else (x0 + g, y0 + i)

    k = n_w
    groups = []
    for g in range(n_h):
        members = []
        for i in range(k):
            x, y = core_of(g, i)
            if sz.is_rm:
                stick0 = g * sh
                sticks = min(sh, rows_ext - stick0)
                col0 = i * sw
                valid_cols = min(sw, geo.W - col0)
                core_w = _div_up(valid_cols, TILE)
                mem = Member(
                    ttnn.CoreCoord(x, y), i, 0, _div_up(sticks, TILE), col0 // TILE, core_w, col0, valid_cols, sticks
                )
            else:
                row0 = g * unit_h
                w0 = i * unit_w
                core_w = min(unit_w, geo.tensor_w_tiles - w0)
                col0 = w0 * TILE
                mem = Member(
                    ttnn.CoreCoord(x, y),
                    i,
                    row0,
                    min(unit_h, rows_ext - row0),
                    w0,
                    core_w,
                    col0,
                    min(core_w * TILE, geo.W - col0),
                )
            members.append(mem)
        groups.append(members)
    crt_max = max(grp[0].core_rows for grp in groups)

    base_opts = PlanOpts(
        shard_backed=not sz.is_rm,
        owner_combine=k > K_SYM_MAX_SHARDED,
        mask_tail=(not sz.is_rm) and geo.W % TILE != 0,
    )
    b, d, opts = _solve_rows_sharded(sz, k, slice_w, crt_max, budget, base_opts)
    # Combine pipeline: sub-block the resident shard (see COMBINE_PIPELINE).
    s_rows = _pipe_subblock_rows(crt_max, k, slice_w)
    pipe_ok = opts.shard_backed and opts.owner_combine and not opts.gamma_chunk and b >= crt_max
    if COMBINE_PIPELINE and pipe_ok and s_rows:
        nb = _div_up(crt_max, s_rows)
        o2 = replace(opts, pipe_nb=nb)
        if nb > 1 and _plan_bytes(cb_plan(sz, k, slice_w, s_rows, 1, o2)) <= budget:
            b, d, opts = s_rows, 1, o2
    c, pitch = _pitch(sz, slice_w, opts.rm_chunk_cap)
    return Blocking(
        k=k,
        slice_w_tiles=slice_w,
        rm_chunk_w_tiles=c,
        x_row_pitch=pitch,
        groups=groups,
        block_row_tiles=b,
        x_block_depth=d,
        out_unit_tiles=_out_unit(slice_w),
        opts=opts,
        footprint=_plan_bytes(cb_plan(sz, k, slice_w, b, d, opts)),
    )


def _gamma_mcast_groups(blk: Blocking, grid_x: int):
    """(receiver_group_size, core order) of the interleaved gamma multicast, or None (per-core gamma load).

    ttnn.Mcast cuts the ordered active cores into equal consecutive groups, and a group must share one W slice.
    k == 1: every Tensix core holds the same slice -> one group of all of them. k > 1: member m of row-group g sits
    in grid column (g % gpr) * k + m, so a grid column holds a single slice; when the row-groups fill whole grid
    rows every column has the same count -> one group per column (column-major order). Otherwise (a partial last
    grid row of row-groups) the columns differ in size and each Tensix core loads its own slice.
    """
    n = len(blk.groups)
    if blk.k == 1:
        return n, ttnn.McastCoreOrder.RowMajor
    if blk.k > grid_x:
        return None  # groups span grid rows: a slice's Tensix cores are not one grid column
    gpr = grid_x // blk.k
    if n % gpr == 0:
        return n // gpr, ttnn.McastCoreOrder.ColumnMajor
    return None


def _inv_mcast_order(blk: Blocking, all_cores, row_major: bool):
    """McastCoreOrder for the single-owner inv_rms multicast, or None (unicast return everywhere).

    Only the owner topology has single-owner blocks, and only blocks of one tile-row use them. ttnn.Mcast cuts the
    ordered active Tensix cores into equal consecutive groups with the fixed sender first: that must reproduce every
    W-split group with member 0 (the owner of tile-row 0) first, and each group must decompose into at most
    MAX_MCAST_RECTANGLES member-only rectangles. Otherwise (custom multi-range shard grids) keep the unicast return.
    """
    if not (INV_MCAST and blk.k > 1 and blk.opts.owner_combine):
        return None
    if blk.opts.pipe_nb:
        return None  # combine pipeline: rotated owners — a one-tile-row sub-block's owner need not be member 0
    b = blk.block_row_tiles
    if not any(b == 1 or (m.core_rows - 1) % b == 0 for grp in blk.groups for m in grp):
        return None  # no block of exactly one tile-row on any Tensix core
    order = ttnn.McastCoreOrder.RowMajor if row_major else ttnn.McastCoreOrder.ColumnMajor
    key = (lambda c: (c[1], c[0])) if row_major else (lambda c: (c[0], c[1]))
    ordered = sorted(((c.x, c.y) for c in all_cores), key=key)
    k = blk.k
    by_first = {}
    for i in range(0, len(ordered), k):
        by_first[ordered[i]] = set(ordered[i : i + k])
    for grp in blk.groups:
        first = (grp[0].core.x, grp[0].core.y)
        if by_first.get(first) != {(m.core.x, m.core.y) for m in grp}:
            return None
        merged = ttnn.CoreRangeSet([ttnn.CoreRange(m.core, m.core) for m in grp]).merge(ttnn.CoreRangeSet([]))
        if len(merged.ranges()) > MAX_MCAST_RECTANGLES:
            return None
    return order


def _noc_swapped_columns(device) -> set:
    """Logical worker columns whose Tensix cores run the reversed NoC pairing (NOC_SWAP_COLUMNS nearest DRAM bank 0)."""
    if not NOC_SWAP_COLUMNS:
        return set()
    grid = device.compute_with_storage_grid_size()
    nearest = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.NOC_0)
    anchor = nearest[0].x
    return set(sorted(range(grid.x), key=lambda col: (abs(col - anchor), col))[:NOC_SWAP_COLUMNS])


def _ordered(cores, order):
    key = (lambda c: (c.y, c.x)) if order == ttnn.McastCoreOrder.RowMajor else (lambda c: (c.x, c.y))
    return sorted(cores, key=key)


def _noc_swapped_cores(device, blk, all_cores, gamma_groups, inv_order) -> set:
    """(x, y) of the active Tensix cores that run the reversed NoC pairing (interleaved only; see NOC_SWAP_COLUMNS)."""
    cols = _noc_swapped_columns(device)
    swapped = {(c.x, c.y) for c in all_cores if c.x in cols}
    if inv_order is not None:  # inv_rms multicast senders: member 0 of each owner group
        swapped -= {(grp[0].core.x, grp[0].core.y) for grp in blk.groups}
    if gamma_groups is not None and gamma_groups[1] == ttnn.McastCoreOrder.ColumnMajor:  # k > 1 gamma senders
        size, order = gamma_groups
        ordered = _ordered(all_cores, order)
        swapped -= {(ordered[i].x, ordered[i].y) for i in range(0, len(ordered), size)}
    return swapped


def _dm_config(processor, noc):
    return ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc, noc_mode=ttnn.NOC_MODE.DM_DEDICATED_NOC)


def is_sharded(tensor) -> bool:
    """Any shard scheme (HEIGHT / WIDTH / BLOCK): the shard spec fixes the work split, consumed in place."""
    return tensor.memory_config().is_sharded()


def cb_l1_budget(device) -> int:
    """Bytes of L1 per Tensix core the program's CBs may occupy.

    CBs are placed bottom-up from the allocator's L1 base; L1 buffers (interleaved-L1 / sharded
    tensors, incl. this op's own output) are allocated top-down. The budget is therefore the free
    region between the allocator base and the lowest allocated L1 buffer. NOT
    `ttnn.get_max_worker_l1_unreserved_size()`: that figure starts at the kernel-config ring buffer,
    below the allocator base, and overshoots by its size (~69 KB on BH) — shapes whose footprint fell
    in that gap threw "circular buffers grow to N B which is beyond max L1 size".
    """
    view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    allocated = [int(b["address"]) for b in view.block_table if b["allocated"] == "yes"]
    return min(allocated) if allocated else int(view.total_bytes_per_bank)


def _datum_bytes(tensor) -> int:
    """Bytes per datum in L1 (bfloat8_b: one mantissa byte — Tensor.element_size() throws for block-float)."""
    return 1 if tensor.dtype == ttnn.bfloat8_b else tensor.element_size()


def _bfp_exp_section_bytes(dtype, elem_bytes: int) -> int:
    """Bytes of shared-exponent section ahead of the datums in one tile of `dtype` (0 for plain formats).

    A bfloat8_b tile is [one exponent byte per 16-datum face row (64 B)][one mantissa byte per datum]; the
    datum area is TILE*TILE*elem_bytes, so the section is whatever the tile size holds beyond it.
    """
    return ttnn.tile_size(dtype) - TILE * TILE * elem_bytes


def _cb(index, pages, page_size, dtype, core_ranges):
    return ttnn.CBDescriptor(
        total_size=pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
    )


def _shard_backed_cb(index, tensor, core_ranges):
    """CB aliasing a TILE tensor's resident shard (zero-copy: the shard IS the CB's L1)."""
    cb = ttnn.cb_descriptor_from_sharded_tensor(index, tensor)
    cb.core_ranges = core_ranges
    return cb


def _create_tilized_program_descriptor(input_tensor, gamma, output_tensor, *, epsilon, compute_kernel_config):
    device = input_tensor.device()
    geo = derive_geometry(input_tensor.shape)
    is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    sharded = is_sharded(input_tensor)
    has_gamma = gamma is not None
    x_dtype = input_tensor.dtype
    g_dtype = gamma.dtype if has_gamma else x_dtype
    fp32_dest = bool(compute_kernel_config.fp32_dest_acc_en)
    t_acc = acc_dtype(fp32_dest)
    sz = _Sizes(x_dtype, g_dtype, is_rm, has_gamma, fp32_dest)
    elem_bytes = _datum_bytes(input_tensor)
    g_elem_bytes = _datum_bytes(gamma) if has_gamma else elem_bytes
    gamma_is_tile = has_gamma and gamma.layout == ttnn.TILE_LAYOUT

    budget = cb_l1_budget(device)
    grid = device.compute_with_storage_grid_size()
    if sharded:
        blk = select_sharded_blocking(geo, sz, input_tensor, budget)
    else:
        blk = select_blocking(geo, sz, grid.x, grid.y, budget, elem_bytes)
    k, C, pitch = blk.k, blk.rm_chunk_w_tiles, blk.x_row_pitch
    b, d = blk.block_row_tiles, blk.x_block_depth
    opts = blk.opts
    combine_lookahead = 1 if (k > 1 and d == 2 and not opts.owner_combine) else 0

    # ---- active Tensix cores: every member of every group (identical CB list on all of them: peers address
    #      cb_gathered_sumsq / cb_inv_rms by their own write pointer) ----
    all_cores = [mem.core for grp in blk.groups for mem in grp]
    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in all_cores])

    # ---- circular buffers ----
    cb_dtype = {
        CB_X: x_dtype,
        CB_X_ALT: x_dtype,
        CB_GAMMA: g_dtype,
        CB_SUMSQ: t_acc,
        CB_PARTIAL: t_acc,
        CB_GATHERED: t_acc,
        CB_INV_RMS: t_acc,
        CB_X_RM: x_dtype,
        CB_OUT: x_dtype,
        CB_OUT_RM: x_dtype,
        CB_ONES: ttnn.bfloat16,
        CB_TAIL_MASK: t_acc,
        CB_INV_OWNED: t_acc,
    }
    plan = cb_plan(sz, k, blk.slice_w_tiles, b, d, opts)
    cbs = [_cb(idx, pages, page_bytes, cb_dtype[idx], core_ranges) for idx, (pages, page_bytes) in plan.items()]
    if opts.shard_backed:
        cbs.append(_shard_backed_cb(CB_X, input_tensor, core_ranges))
        cbs.append(_shard_backed_cb(CB_OUT, output_tensor, core_ranges))
        out_cb_pages = 0  # TILE sharded writer drains per block; no ring clamp
    else:
        out_cb_pages = plan[CB_OUT][0]

    defines = []
    if is_rm:
        defines.append(("RMS_RM", "1"))
    if has_gamma:
        defines.append(("RMS_HAS_GAMMA", "1"))
    if sharded:
        defines.append(("RMS_SHARDED", "1"))
    if opts.mask_tail:
        defines.append(("RMS_MASK_TAIL", "1"))
    defines.extend((name, "1") for name in PERF_ABLATE)
    gamma_groups = None
    if opts.gamma_chunk:
        defines.append(("RMS_GAMMA_CHUNK", str(opts.gamma_chunk)))  # streamed gamma: per-core chunk ring, no mcast
        defines.append(("RMS_GAMMA_STREAM_DEPTH", str(GAMMA_STREAM_DEPTH)))
    elif has_gamma and (GAMMA_MCAST_SHARDED_K1 and k == 1 if sharded else GAMMA_MCAST):
        gamma_groups = _gamma_mcast_groups(blk, grid.x)
    gamma_mcast = gamma_groups is not None
    if gamma_mcast:
        defines.append(("RMS_GAMMA_MCAST", "1"))
    inv_order = _inv_mcast_order(
        blk,
        [mem.core for grp in blk.groups for mem in grp],
        not sharded or input_tensor.memory_config().shard_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR,
    )
    if inv_order is not None:
        defines.append(("RMS_INV_MCAST", "1"))

    x_page = input_tensor.buffer_aligned_page_size()
    out_page = output_tensor.buffer_aligned_page_size()
    g_page = gamma.buffer_aligned_page_size() if has_gamma else 0

    # Sharded: the local shard is consumed in place, never through an accessor -> empty accessor args.
    x_acc_args = ttnn.TensorAccessorArgs() if sharded else ttnn.TensorAccessorArgs(input_tensor)
    out_acc_args = ttnn.TensorAccessorArgs() if sharded else ttnn.TensorAccessorArgs(output_tensor)

    reader_ct = [geo.tensor_w_tiles, geo.W, geo.H, geo.image_row_tiles, pitch, C, x_page, elem_bytes, d]
    reader_ct.extend(x_acc_args.get_compile_time_args())

    writer_ct = [
        geo.tensor_w_tiles,
        geo.W,
        geo.H,
        geo.image_row_tiles,
        C,
        out_page,
        elem_bytes,
        k,
        blk.out_unit_tiles,
        SEM_PARTIALS_ARRIVED,
        out_cb_pages,
        WRITER_NOC_XY_ARG_BASE,
        1 if opts.owner_combine else 0,
        1 if PARTIAL_FACES_ONLY else 0,
        SEM_INV_ARRIVED,
        pitch,
        g_page,
        g_elem_bytes,
        1 if gamma_is_tile else 0,
        _bfp_exp_section_bytes(g_dtype, g_elem_bytes) if has_gamma else 0,
        1 if (GAMMA_AFTER_SEND and b <= GAMMA_AFTER_SEND_MAX_ROWS) else 0,
        1 if SIGNAL_AFTER_FLUSH else 0,
        sz.T_g,
        1 if opts.pipe_nb else 0,
        SEM_PARTIALS_PAR,
        SEM_INV_PAR,
    ]
    writer_ct.extend(out_acc_args.get_compile_time_args())
    writer_ct.extend(
        (ttnn.TensorAccessorArgs(gamma) if has_gamma else ttnn.TensorAccessorArgs()).get_compile_time_args()
    )

    compute_ct = [k, pitch, C, d, combine_lookahead, 1 if opts.owner_combine else 0, 1 if opts.pipe_nb else 0]

    inv_w_bits = _f32_bits(1.0 / geo.W)
    eps_bits = _f32_bits(epsilon)
    x_addr = input_tensor.buffer_address()
    g_addr = gamma.buffer_address() if has_gamma else 0
    out_addr = output_tensor.buffer_address()

    reader_rt, writer_rt, compute_rt = {}, {}, ttnn.RuntimeArgs()
    tw = geo.tensor_w_tiles
    for members in blk.groups:
        noc_xy = []
        for mem in members:
            v = device.worker_core_from_logical_core(mem.core)
            noc_xy.extend([v.x, v.y])
        for mem in members:
            c = mem.core
            core_rows = mem.core_rows
            num_blocks = _div_up(core_rows, b)
            last_rows = core_rows - (num_blocks - 1) * b
            chunks_this = _div_up(mem.core_w, C)
            owns_tail = 1 if (opts.mask_tail and mem.w_tile_start + mem.core_w == tw) else 0
            # TILE sharded: every tile-row pushes the whole shard row (pitch) into the shard-backed cb_out.
            norm_w = pitch if opts.shard_backed else mem.core_w
            reader_rt[(c.x, c.y)] = [
                x_addr,
                mem.row_start,
                num_blocks,
                b,
                last_rows,
                mem.w_tile_start,
                mem.core_w,
                chunks_this,
                mem.valid_cols,
                mem.valid_sticks,
                owns_tail,
            ]
            compute_rt[c.x][c.y] = [
                num_blocks,
                b,
                last_rows,
                mem.core_w,
                chunks_this,
                inv_w_bits,
                eps_bits,
                mem.m,
                norm_w,
                owns_tail,
            ]
            writer_base = [
                out_addr,
                mem.row_start,
                num_blocks,
                b,
                last_rows,
                mem.w_tile_start,
                mem.core_w,
                chunks_this,
                mem.m,
                mem.valid_cols,
                mem.valid_sticks,
                g_addr,
                mem.col0,
            ]
            assert len(writer_base) == WRITER_NOC_XY_ARG_BASE
            writer_rt[(c.x, c.y)] = writer_base + noc_xy

    # DM kernel groups: the usual pairing (reader NoC0 / writer NoC1) and, on the swapped Tensix cores, the reversed one.
    swapped = set() if sharded else _noc_swapped_cores(device, blk, all_cores, gamma_groups, inv_order)
    dm_groups = []  # (cores, reader config, writer config)
    for cores, rcfg, wcfg in (
        (
            [c for c in all_cores if (c.x, c.y) not in swapped],
            ttnn.ReaderConfigDescriptor(),
            ttnn.WriterConfigDescriptor(),
        ),
        (
            [c for c in all_cores if (c.x, c.y) in swapped],
            _dm_config(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
            _dm_config(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
        ),
    ):
        if cores:
            dm_groups.append((cores, rcfg, wcfg))

    def _rt(cores, table):
        rt = ttnn.RuntimeArgs()
        for c in cores:
            rt[c.x][c.y] = table[(c.x, c.y)]
        return rt

    readers, writers = [], []
    for cores, rcfg, wcfg in dm_groups:
        crs = core_ranges if len(dm_groups) == 1 else ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        readers.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
                core_ranges=crs,
                compile_time_args=reader_ct,
                defines=defines,
                runtime_args=_rt(cores, reader_rt),
                config=rcfg,
            )
        )
        writers.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
                core_ranges=crs,
                compile_time_args=writer_ct,
                defines=defines,
                runtime_args=_rt(cores, writer_rt),
                config=wcfg,
            )
        )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=core_ranges,
        compile_time_args=compute_ct,
        defines=defines,
        runtime_args=compute_rt,
        config=compute_kernel_config,
    )
    semaphores = []
    if k > 1:
        semaphores.append(ttnn.SemaphoreDescriptor(id=SEM_PARTIALS_ARRIVED, core_ranges=core_ranges, initial_value=0))
        if opts.owner_combine:
            semaphores.append(ttnn.SemaphoreDescriptor(id=SEM_INV_ARRIVED, core_ranges=core_ranges, initial_value=0))
        if opts.pipe_nb:
            # ids 2 / 3 are the first ttnn.Mcast ids: neither multicast may be attached on the pipelined path.
            assert not gamma_mcast and inv_order is None
            for sem in (SEM_PARTIALS_PAR, SEM_INV_PAR):
                semaphores.append(ttnn.SemaphoreDescriptor(id=sem, core_ranges=core_ranges, initial_value=0))
    descriptor = ttnn.ProgramDescriptor(kernels=[], semaphores=semaphores, cbs=cbs)
    next_sem = SEM_GAMMA_MCAST_FIRST
    if gamma_mcast:
        # Sender = each group's first Tensix core (reads the slice); one-shot, no handshake. k = 1 (one group of every
        # Tensix core) with swapped cores: the first Tensix core on the usual pairing (all swapped: core 0 on NoC0).
        group_size, order = gamma_groups
        sender, noc = 0, ttnn.NOC.NOC_1  # the usual writer NoC (WriterConfigDescriptor)
        if swapped and order == ttnn.McastCoreOrder.RowMajor:
            plain = [i for i, c in enumerate(_ordered(all_cores, order)) if (c.x, c.y) not in swapped]
            sender, noc = (plain[0], ttnn.NOC.NOC_1) if plain else (0, ttnn.NOC.NOC_0)
        mc = ttnn.Mcast(
            device,
            ttnn.McastConfig(noc=noc, handshake=False),
            core_ranges,
            group_size,
            ttnn.McastFixedSenderConfig(sender_index=sender),
            order,
        )
        mc.attach(descriptor, "gamma_mcast", writers, next_sem)
        next_sem = mc.next_semaphore_id()
    if inv_order is not None:
        # Single-owner blocks: member 0 (the group's first Tensix core in `inv_order`, always on the usual pairing)
        # multicasts inv_rms to its group.
        inv_mc = ttnn.Mcast(
            device,
            ttnn.McastConfig(noc=ttnn.NOC.NOC_1, handshake=False),  # member 0's writer NoC
            core_ranges,
            k,
            ttnn.McastFixedSenderConfig(),
            inv_order,
        )
        inv_mc.attach(descriptor, "inv_mcast", writers, next_sem)
    descriptor.kernels = readers + writers + [compute]
    return descriptor


# ---------------------------------------------------------------------------------------------------------------------
# HEIGHT_SHARDED ROW_MAJOR flat regime (see RM_FLAT)
# ---------------------------------------------------------------------------------------------------------------------


def flat_expressible(input_tensor, gamma) -> bool:
    """HEIGHT_SHARDED ROW_MAJOR fp32 / bf16 input whose shard holds whole sticks, gamma absent or ROW_MAJOR.

    A TILE gamma would need a row-0 gather into flat order (not built; RM input + TILE gamma under HEIGHT sharding is
    golden-INVALID) — it keeps the tilized path.
    """
    if not RM_FLAT or input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        return False
    mc = input_tensor.memory_config()
    if mc.memory_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        return False
    if input_tensor.dtype not in (ttnn.float32, ttnn.bfloat16):
        return False
    if gamma is not None and (
        gamma.layout != ttnn.ROW_MAJOR_LAYOUT or gamma.dtype not in (ttnn.float32, ttnn.bfloat16)
    ):
        return False
    return int(mc.shard_spec.shape[1]) >= int(input_tensor.shape[-1])


def flat_plan(input_tensor, gamma, fp32_dest: bool) -> dict:
    """Geometry + CB plan of the flat path: {cb: (pages, page_bytes, dtype)}; shard-backed CBs carry pages = None.

    cb_x / cb_out alias the input / output shards with page = gcd(stick pitch, tile bytes), so flat tile j of stick s
    is page s * stick_pages + j * tile_pages. W % 1024 != 0: cb_x_rm / cb_out_rm hold one zero-padded tail tile per
    stick. cb_ones (the REDUCE_SCALAR 1.0 scaler) follows the accumulator dtype: REDUCE_SCALAR moves its 32 first-stage
    partial sums DEST -> srcB (MOVD2B), so a bf16 scaler would truncate them to bf16 (fp32 DEST, no gamma: rms error vs
    torch 0.0027 -> 0.0015 bf16, 0.0014 -> 0.00056 fp32; same device time).
    """
    W = int(input_tensor.shape[-1])
    x_dtype = input_tensor.dtype
    elem = _datum_bytes(input_tensor)
    T = ttnn.tile_size(x_dtype)
    sh = int(input_tensor.memory_config().shard_spec.shape[0])
    pitch = input_tensor.buffer_aligned_page_size()  # stick pitch inside the shard
    nt = _div_up(W, 1024)
    n_body = W // 1024
    tail_bytes = (W - 1024 * n_body) * elem
    page = math.gcd(pitch, T)
    t_acc = acc_dtype(fp32_dest)
    cbs = {CB_X: (None, page, x_dtype), CB_OUT: (None, page, x_dtype)}
    if tail_bytes:
        cbs[CB_X_RM] = (sh, T, x_dtype)
        cbs[CB_OUT_RM] = (sh, T, x_dtype)
    cbs[CB_SUMSQ] = (sh, ttnn.tile_size(t_acc), t_acc)
    cbs[CB_INV_RMS] = (sh, ttnn.tile_size(t_acc), t_acc)
    cbs[CB_ONES] = (1, ttnn.tile_size(t_acc), t_acc)
    if gamma is not None:
        cbs[CB_GAMMA] = (nt, ttnn.tile_size(gamma.dtype), gamma.dtype)
    return dict(
        W=W,
        elem=elem,
        T=T,
        sh=sh,
        pitch=pitch,
        nt=nt,
        n_body=n_body,
        has_tail_cb=tail_bytes != 0,
        tail_bytes=tail_bytes,
        page=page,
        stick_pages=pitch // page,
        tile_pages=T // page,
        cbs=cbs,
        footprint=sum(n * b for (n, b, _) in cbs.values() if n is not None),
    )


def flat_fits(plan, budget: int) -> bool:
    return plan["footprint"] <= budget and plan["sh"] * plan["stick_pages"] < FLAT_MAX_CB_PAGES


def flat_preferred(plan) -> bool:
    """Cost model (see RM_FLAT): the flat path is estimated no slower than the tilized one."""
    S, W, nt = plan["sh"], plan["W"], plan["nt"]
    flat = FLAT_FIXED_NS + S * (FLAT_STICK_NS + FLAT_TILE_NS * nt + FLAT_TAIL_NS * (1 if W % 1024 else 0))
    tilized = _div_up(S, TILE) * (TILIZED_ROW_NS + TILIZED_WTILE_NS * _div_up(W, TILE))
    return flat <= tilized


def create_program_descriptor(input_tensor, gamma, output_tensor, *, epsilon, compute_kernel_config):
    """Flat HEIGHT RM regime when expressible + fits L1 + (cost model prefers it); else the tilized program. If the
    tilized plan does not fit L1 but the flat one does, the flat path runs."""
    kw = dict(epsilon=epsilon, compute_kernel_config=compute_kernel_config)
    flat_ok = False
    if flat_expressible(input_tensor, gamma):
        plan = flat_plan(input_tensor, gamma, bool(compute_kernel_config.fp32_dest_acc_en))
        flat_ok = flat_fits(plan, cb_l1_budget(input_tensor.device()))
        if flat_ok and (RM_FLAT_FORCE or flat_preferred(plan)):
            return create_flat_program_descriptor(input_tensor, gamma, output_tensor, **kw)
    try:
        return _create_tilized_program_descriptor(input_tensor, gamma, output_tensor, **kw)
    except RuntimeError:
        if not flat_ok:
            raise
        return create_flat_program_descriptor(input_tensor, gamma, output_tensor, **kw)


def _backed_cb_paged(index, tensor, dtype, page, core_ranges):
    """CB aliasing a sharded tensor's resident shard (zero-copy) with an explicit page size."""
    cb = ttnn.cb_descriptor_from_sharded_tensor(index, tensor)
    cb.core_ranges = core_ranges
    cb.format_descriptors = [ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)]
    return cb


def create_flat_program_descriptor(input_tensor, gamma, output_tensor, *, epsilon, compute_kernel_config):
    device = input_tensor.device()
    shape = [int(d) for d in input_tensor.shape]
    W = shape[-1]
    rows = math.prod(shape[:-1])
    has_gamma = gamma is not None
    plan = flat_plan(input_tensor, gamma, bool(compute_kernel_config.fp32_dest_acc_en))
    pitch, nt, T = plan["pitch"], plan["nt"], plan["T"]
    assert output_tensor.buffer_aligned_page_size() == pitch
    n_body, has_tail_cb, tail_bytes = plan["n_body"], plan["has_tail_cb"], plan["tail_bytes"]
    stick_pages, tile_pages = plan["stick_pages"], plan["tile_pages"]

    # Active Tensix cores: shard g (sh sticks, the last one ragged) on the g-th core of the shard grid in its orientation.
    ss = input_tensor.memory_config().shard_spec
    sh = plan["sh"]
    order = _ordered_shard_cores(ss.grid, ss.orientation == ttnn.ShardOrientation.ROW_MAJOR)
    n_shards = _div_up(rows, sh)
    assert len(order) >= n_shards
    cores = [ttnn.CoreCoord(x, y) for (x, y) in order[:n_shards]]
    sticks_of = [min(sh, rows - g * sh) for g in range(n_shards)]
    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])

    cbs = []
    for idx, (pages, page_bytes, dtype) in plan["cbs"].items():
        if pages is None:
            cbs.append(
                _backed_cb_paged(idx, input_tensor if idx == CB_X else output_tensor, dtype, page_bytes, core_ranges)
            )
        else:
            cbs.append(_cb(idx, pages, page_bytes, dtype, core_ranges))
    g_page = gamma.buffer_aligned_page_size() if has_gamma else 0  # one read of the whole (aligned) gamma page
    assert not has_gamma or g_page <= nt * ttnn.tile_size(gamma.dtype)

    defines = []
    if has_gamma:
        defines.append(("RMS_HAS_GAMMA", "1"))
    defines.extend((name, "1") for name in PERF_ABLATE)
    gamma_mcast = has_gamma and GAMMA_MCAST_SHARDED_K1
    if gamma_mcast:
        defines.append(("RMS_GAMMA_MCAST", "1"))

    stick_bytes = W * plan["elem"]
    reader_ct = [n_body, int(has_tail_cb), tail_bytes, pitch, T, stick_pages, stick_bytes]
    compute_ct = [n_body, int(has_tail_cb), stick_pages, tile_pages]
    writer_ct = [n_body, int(has_tail_cb), tail_bytes, pitch, T, stick_pages, stick_bytes, g_page, nt, g_page]
    writer_ct.extend(
        (ttnn.TensorAccessorArgs(gamma) if has_gamma else ttnn.TensorAccessorArgs()).get_compile_time_args()
    )

    x_addr, out_addr = input_tensor.buffer_address(), output_tensor.buffer_address()
    g_addr = gamma.buffer_address() if has_gamma else 0
    inv_w_bits, eps_bits = _f32_bits(1.0 / W), _f32_bits(epsilon)
    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, n in zip(cores, sticks_of):
        reader_rt[c.x][c.y] = [x_addr, n]
        compute_rt[c.x][c.y] = [n, inv_w_bits, eps_bits]
        writer_rt[c.x][c.y] = [out_addr, n, g_addr]

    def _k(name, ct, rt, cfg):
        return ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / name),
            core_ranges=core_ranges,
            compile_time_args=ct,
            defines=defines,
            runtime_args=rt,
            config=cfg,
        )

    reader = _k("rms_norm_flat_reader.cpp", reader_ct, reader_rt, ttnn.ReaderConfigDescriptor())
    writer = _k("rms_norm_flat_writer.cpp", writer_ct, writer_rt, ttnn.WriterConfigDescriptor())
    compute = _k("rms_norm_flat_compute.cpp", compute_ct, compute_rt, compute_kernel_config)
    descriptor = ttnn.ProgramDescriptor(kernels=[], semaphores=[], cbs=cbs)
    if gamma_mcast:
        # One group of every active Tensix core; sender = core 0 (reads the gamma page once) on the writer NoC; one-shot.
        mc = ttnn.Mcast(
            device,
            ttnn.McastConfig(noc=ttnn.NOC.NOC_1, handshake=False),
            core_ranges,
            len(cores),
            ttnn.McastFixedSenderConfig(sender_index=0),
            ttnn.McastCoreOrder.RowMajor,
        )
        mc.attach(descriptor, "gamma_mcast", [writer], SEM_GAMMA_MCAST_FIRST)
    descriptor.kernels = [reader, writer, compute]
    return descriptor
