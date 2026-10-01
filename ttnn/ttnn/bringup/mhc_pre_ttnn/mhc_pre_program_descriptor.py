# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""mhc_pre — ProgramDescriptor (regime R1 `group_ksplit_resident`, see op_design.md Blocking Model).

Token tile-rows are split over *groups* (rectangles of `group_w x group_h` cores); inside a group the
n*C reduction axis is split by stream-column slice over the group's *ranks*. Each rank keeps its X block
resident from the projection to the y-mix (X crosses DRAM once), sends its (mix, sum x^2) partial pair to
the group root (push gather + monotonic semaphore), the root folds the partials in rank order and
multicasts the combined pair back (mcast_pipe), every rank computes the coefficients itself and mixes its
own y columns, and the Sinkhorn + post/comb output of each token row is owned round-robin by one rank.

Every block knob is derived here ONCE and handed to the kernels as a CT/RT arg:
    group geometry  -> group_w, group_h, group_cores, num_groups (core-assignment knobs)
    block_token_tiles, x_block_depth (L1 selection function)
    y_chunk_tiles, y_depth           (writer window knobs)
"""

import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
F32_TILE_BYTES = 4096
BF16_TILE_BYTES = 2048

# ---- CB indices (semantic names; the slot numbers are arbitrary) ----
CB_X_RESIDENT = 0
CB_WEIGHT = 1
CB_BIAS_COEF = 2
CB_REDUCE_SCALER = 3
CB_SQ_ACC = 4
CB_PARTIAL = 5
CB_GATHERED = 6
CB_COMBINED = 7
CB_COEF_IN = 8
CB_COEF_KEEP = 9  # owned rows: the coefficient-major tile after the Sinkhorn (reloaded for the pre tiles)
CB_COMB_COEF = 11
CB_PRE_COLS = 12
CB_Y_OUT = 13
CB_WEIGHT_SPLIT = 15  # aliases CB_WEIGHT's allocation (fp32 W only): bf16 pages [W_hi(k), W_lo(k)] per k
CB_X_FP32 = 16  # aliases CB_X_RESIDENT's allocation (fp32 X only): the same tiles, read UnpackToDestFp32
CB_X_PIECES = 17  # fp32 X only: bf16 pieces [x0, x1_hi, x1_mid] of one K chunk window (streams K)
CB_MIX_RUN = 18  # fp32 X only: running fp32 mix partial between K chunk windows (exact reload)
CB_MAX_LANES = 19  # fp32 X only: lane-wise max|v| tile (x row-tile / W slice) -> reduce MAX
CB_MAX_SCALAR = 20  # fp32 X only: reduce<MAX, REDUCE_SCALAR> result (element (0, 0))
CB_GRID = 21  # fp32 X only: per token row-tile grid rounding constants (+ the W one, once)
CB_MAX_SCALER = 22  # fp32 X only: reduce scaler for <MAX, REDUCE_SCALAR> (1.0)
CB_W_OWN_READY = 23  # W column all-gather token (no payload): reader -> compute "own W share landed"
CB_W_OWN_SPLIT = 24  # W column all-gather token (no payload): compute -> reader "own W share split"
CB_W_SHARE_LANDED = 25  # token (no payload): reader -> writer "this core's W column share landed" (W_SHARE_ON_READER)
TOKEN_PAGE_BYTES = 32
NUM_CB_SLOTS = 64
UNPACK_TO_DEST_FP32_CBS = (CB_BIAS_COEF, CB_SQ_ACC, CB_GATHERED, CB_COEF_IN, CB_COEF_KEEP)


def w_pieces(w_dtype):
    """bf16 pieces of W the projection accumulates in one DEST window (op_design.md, W hi/lo split).

    An fp32 W is split exactly into W_hi = bf16-truncate(W) and W_lo = W - W_hi (the FPU would otherwise
    read it as ~tf32); a bf16 W is already exact. Single source for the descriptor and the compute kernel.
    """
    return 2 if w_dtype == ttnn.float32 else 1


def x_pieces(x_dtype):
    """bf16 pieces of X the projection consumes (Refinement 2, fp32 streams).

    The FPU reads fp32 as ~tf32 (truncating) AND rounds each in-tile 32-long dot product to ~11 bits below
    its largest product, so exact bf16 pieces alone do not reach fp32. An fp32 X is split per K chunk into
    x0 = x rounded to a per-row-tile power-of-two grid (X_GRID_BITS bits: its products with the likewise
    gridded W0 sum exactly in-tile), x1_hi = bf16-truncate(x - x0), x1_mid = bf16-RNE(the rest). A bf16 X
    keeps the unsplit path. Single source for the descriptor and the compute kernel.
    """
    return 3 if x_dtype == ttnn.float32 else 1


# ---- semaphores ----
SEM_GATHER = 0  # monotonic partial-arrival counter on the group root
SEM_MCAST_READY = 1  # mcast_pipe data-ready flag
SEM_MCAST_CONSUMED = 2  # mcast_pipe consumer-ready (pre-handshake) counter
SEM_W_READY = 3  # W column broadcast: data-ready Counter (one event per W chunk)

# ---- host constants (tunable knobs, single source) ----
GROUP_CORES_CAP = 32  # flat-root gather cap (cb_gathered grows with group_cores)
X_BLOCK_DEPTH_DEFAULT = 2  # prefetch depth of cb_x_resident (stall shadow of the combine round trip)
# Blocks of S held in cb_coef_in (Perf 1, cross-block pipeline): the root multicasts S(b+1) while every rank still
# holds S(b), with no consumer-ready handshake. Must be 2 (the writer's write-once landing argument).
COEF_IN_BLOCKS = 2
# Each X block is read as one NoC burst but published to the compute in this many K-ordered chunks (one NoC
# transaction id each), so the projection runs under the rest of the burst instead of after it (Refinement 4).
# 1 = one publish per block (the pre-Refinement-4 behaviour); at most 14 (NoC transaction ids; 15 is the W column share's).
X_STREAM_CHUNKS = 4
# Chunks of one X block in flight at once (Refinement 4). With every chunk issued up front the banks serve all
# cores' requests interleaved and even chunk 0 lands only at the end of the burst; a bounded look-ahead keeps
# this core's chunks roughly in order. >= X_STREAM_CHUNKS = all issued up front.
X_STREAM_INFLIGHT = 2
Y_DEPTH = 2  # cb_y_out windows in flight
Y_CHUNK_TILES_CAP = 8  # 4-8 writes in flight per barrier saturate (catalog: double_buffer)
# W is pushed into cb_weight in chunks of this many tiles (one read barrier each), so the compute kernel's
# fp32 W hi/lo split of chunk j runs under the DRAM read of chunk j+1 instead of after the whole W slice.
W_CHUNK_TILES = 8
# Math fidelity of the X @ W_lo products (fp32 W split). W_lo <= 2^-8 |W|, so LoFi's truncation lands far
# below the FPU accumulation floor; set to the caller's fidelity to disable (byte-identical to one fidelity).
W_LO_FIDELITY = ttnn.MathFidelity.LoFi
# fp32 X only: K tiles per x-piece chunk window. The pieces are recomputed per chunk from the resident fp32
# X block (never a second resident copy); larger = fewer running-partial reloads, more L1.
X_CHUNK_K_TILES = 8
X_PIECE_DEPTH = 2  # chunk windows in flight in cb_x_pieces
# Grid bits of the leading pieces: |x0 / g_x| <= 2^X_GRID_BITS, |W0 / g_w| <= 2^W_GRID_BITS. Their products
# must stay inside the FPU's exact in-tile window (probe: exact up to 2^11), so X + W <= 10.
X_GRID_BITS = 4
W_GRID_BITS = 6
# Piece products kept: x_q @ W_p for q + p <= this (x order [x0, x1_hi, x1_mid], W order [W0, W1]); the
# dropped x1_mid @ W1 is ~2^-(X+W+9) of the mix.
PRODUCT_ORDER_MAX = 2
# Products with q + p >= this (x1_mid @ W0, x1_hi @ W1: <= ~2^-(X_GRID_BITS+2) of the mix) run at
# X_LO_FIDELITY; > PRODUCT_ORDER_MAX disables. Measured (640x7168 fp32 / T64 large-logit z rms): HiFi3 is
# lossless (3.7e-5, same as HiFi4), HiFi2 4.2e-5, LoFi 8.5e-5 (too lossy); the perf difference is within
# noise (the matmul is not the binding stage), so HiFi3 is a live knob with no measured win yet.
PRODUCT_LO_ORDER = 2
X_LO_FIDELITY = ttnn.MathFidelity.HiFi3
DEST_TILES_FP32 = 4  # DEST_AUTO_LIMIT at fp32_dest_acc_en, half sync: bound on the projection sub-block height
L1_SAFETY_MARGIN = 96 * 1024  # headroom below the allocator's unreserved L1 (kernel config ring incl. the kernel
# binaries, stack). Measured Refinement 4: the CB base sits ~70.7 KB above the unreserved base (1x1x2048x20480
# bf16 overflowed at 64 KB once the compute binary grew).
# Upper bound on block_token_tiles (the selection function takes min(this, core share, L1 fit)).
# Measured on BH p150 (fp32, device kernel ns, bt=coarsest-fit -> bt=1): 640x7168 384->383 us,
# 640x1792 133->128, 1280x4096 420->381, 4096x1792 591->521 (bt 2/4/7 in between). The whole K slice
# is still one block; finer token blocks let the X read of block b+1 overlap block b's combine round
# trip, y-mix and y stores (design perf lamp L1). Raise it to trade that overlap for fewer per-block
# fixed costs.
BLOCK_TOKEN_TILES_CAP = 1
# Group-width selection (core-assignment knob, perf lamp L2; Refinement 4, re-derived in Perf 2 for the cross-block
# pipeline). When token tile-rows fill the grid (Mt >= grid_y, one-row groups) and X is bf16, every group_w in
# min(grid_x, Ct)..1 that fits L1 is a candidate and the plan with the lowest `_block_schedule_cost` wins, after
# (1) any plan that can prefetch X (depth >= 2, or a single block): at depth 1 the X read of block b+1 waits for
# block b's whole tail. (The Refinement 4 rule "fewest blocks, widest among those" ignored the per-core X load and
# the depth: it picked group_w 2-3 with 30-60 % of the cores idle or a depth-1 fallback -- 1024x1792 152 us --
# and missed that with the pipeline an extra block costs only its exposed round trip -- 640x7168 145 us at
# group_w 5 / depth 1 vs 128 us at group_w 11 / 2 blocks.) False = always group_w = min(grid_x, Ct).
# Measured (BH p150 11x10, bf16 X, device us, median of 3, Refinement 4 rule -> cost model):
#   fp32 W: 640x7168 145.4 -> 128.2, 640x1792 43.9 -> 43.6 and 1280x4096 138.7 -> 138.6 (same plan),
#     1024x1792 152.1 -> 68.1, 2048x4096 315.8 -> 241.6, 4096x6144 814.3 -> 703.7; over T in {512..4096} x
#     C in {1792..7168} (42 cells) 32 faster by 2.5-55 %, 1280x5120 176.2 -> 175.4, 9 the same plan.
#   bf16 W: 1024x4096 219.3 -> 140.4, 2048x1792 203.1 -> 130.1, 2048x6144 421.0 -> 353.4; one loss, 4096x4096
#     455.9 -> 471.0 (Refinement 4 picked group_w 2 at depth 1, which the prefetch rule excludes); rest within +-1.5 %.
NARROW_GROUPS = True
# `_block_schedule_cost` constants, in units of one X tile read by one core under full-grid DRAM contention (~0.55 us
# on BH p150), calibrated on the sweep above (flat for RT_TILES_BASE 30-34 at TAIL_FRAC 0.35-0.45) and cross-checked
# with the stage zones (640x1792, group_w 5 vs 11):
RT_TILES_BASE = 32  # one block's round trip after its last X tile (partial -> gather -> mcast -> coef/pre, and the
# owner's Sinkhorn): ~8 us gather / mcast / coefficients + ~7 us Sinkhorn
RT_TILES_PER_RANK = 0.75  # the root's rank-ordered fold grows with group_cores (zones: ~0.45 us per partial)
RT_TILES_UNSTREAMED_PROJ = 4  # bf16 W: the projection is one whole-block window, not streamed under the X read
TAIL_FRAC = 0.4  # a block's own tail (coefficients + y-mix + y write) per K tile, relative to its X read
SINKHORN_TILES = 12  # one-row groups (owner_fixed): OWNER_C_DISCOUNT takes the owner's Sinkhorn off the critical path


def _block_schedule_cost(f, n, streamed_proj):
    """Critical-path estimate of one group_h = 1 plan (`fit` result), in X-tile-read units (see NARROW_GROUPS).

    stream        blocks * kmax          the rank's X blocks cross the NoC back to back (depth >= 2 prefetch)
    last block    H + kmax / n           its round trip and y-mix / y write (y is 1/n of X) are always exposed
    step b < B-1  pipelined (pipe_at):   max(0, H - TAIL_FRAC * kmax)        S(b+1)'s round trip under tail(b)
                  serial:                max(0, H - (1 - TAIL_FRAC) * kmax)  S(b)'s round trip under the reader's
                                                                             prefetch of X(b+1), minus tail(b)
    H = RT_TILES_BASE + RT_TILES_PER_RANK * group_cores (+ RT_TILES_UNSTREAMED_PROJ); one-row groups drop
    SINKHORN_TILES from the last block. pipe_at mirrors the kernels' block schedule (compute / writer).
    """
    B, d, k, G = f["blocks"], f["depth"], f["kmax"], f["group_cores"]
    H = RT_TILES_BASE + RT_TILES_PER_RANK * G + (0 if streamed_proj else RT_TILES_UNSTREAMED_PROJ)
    cost = B * k + H + k / n - (SINKHORN_TILES if max(f["core_token_tiles"]) <= 1 else 0)
    for b in range(B - 1):
        pipelined = b + 1 < B and (d >= 3 or b + d >= B)
        cost += max(0.0, H - TAIL_FRAC * k) if pipelined else max(0.0, H - (1.0 - TAIL_FRAC) * k)
    return cost


# Regime R2 `W column broadcast` (op_design.md Regimes): W does not vary along the token-group split, so the
# cores of one physical column (= one rank of every group row) need the same W slice. Column all-gather
# (Refinement 4): each row reads only its 1/rows share of the slice from DRAM and multicasts it down the
# column (rotating-sender Mcast1D(PerColumn), Counter signal, write-once landing => no handshake); with an
# fp32 W and bf16 X its compute first splits that share into [W_hi, W_lo] in place, so the one-time W split
# is also spread 1/rows per core. Applies when groups are one core-row tall (group_h == 1) and at least two
# full group rows are active; otherwise every core reads (and splits) W itself (R1). False disables.
W_BCAST = True
# NoC placement (noc_placement): the reader's X stream rides READER_NOC; the writer (W fill, both multicasts,
# partial / y / post / comb stores) rides the other one.
READER_NOC = ttnn.NOC.NOC_0
# Groups whose first logical core-row is < the flip row count swap the two NoCs (reader on the other NoC, writer on
# READER_NOC). With every reader on NoC0 the X burst starves the top core rows (their DRAM responses share the most
# south links: measured 26 us vs 8 us for the bottom row at 640x1792; 1280x4096: top rows' X lands at ~120 us vs
# ~95 us at the bottom), and those rows set the wall. Flip row count = round(READER_NOC_FLIP_FRACTION * grid_y)
# (Refinement 5; measured on BH 11x10 with the W share on the reader, bf16 / fp32 X, device us, flip 0 -> 4 rows:
# 1280x4096 174.3 -> 148.5 / 393.8 -> 380.9, 640x1792 46.6 -> 44.1 / 130.7 -> 122.4, 640x7168 174.4 -> 145.1 /
# 457.6 -> 411.8, 2048x5120 323.4 -> 314.3 / 722.0 -> 717.5; 2 and 3 rows in between, 5 rows starves NoC1).
# READER_NOC_FLIP_ROWS (int) overrides the derived count; 0 = no row flipped.
READER_NOC_FLIP_FRACTION = 0.4
READER_NOC_FLIP_ROWS = None
# W column share on the reader (Refinement 5): with the W column all-gather (W_ROLE_SPREAD) the reader, not the
# writer, reads this core's W share from DRAM -- issued ahead of its X burst (own NoC transaction id), so it rides
# the NoC this row's X reads use (the uncongested one for that row, whatever READER_NOC_FLIP_ROWS says), and hands
# it to the writer (token CB_W_SHARE_LANDED) for the split / multicast. On the writer's NoC a share read queued
# behind the other rows' X burst (measured 56-60 us at 1280x4096 on flipped rows, vs 7-13 us), and every
# projection waits for the whole column's all-gather. False = the writer reads its share (the Refinement 4 path).
W_SHARE_ON_READER = True
# ... and land it (barrier) before the first X read is issued: all rows' shares then cross the NoCs / banks with no
# X traffic (the all-gather waits for the slowest row), at the cost of starting X a few us later.
W_SHARE_BEFORE_X = False
# Path gate of the two placement levers above (derived flip rows, W share on the reader). Measured wins (flip 0 +
# writer W -> defaults, device us): bf16 X / fp32 W 1280x4096 176.6 -> 148.4, 2048x5120 345.5 -> 315.4, 1x7168
# 58.0 -> 46.5; fp32 X / fp32 W 640x1792 128.4 -> 121.2; fp32 X / bf16 W 640x7168 392.7 -> 351.6, 1280x4096
# 387.0 -> 371.3. bf16 X / bf16 W (unstreamed matmul_block projection) loses: 1280x4096 183.2 -> 188.3, 2048x5120
# 354.3 -> 377.9 -- that path keeps the Refinement 4 placement (no flip, W share on the writer).
PLACEMENT_LEVERS_BF16_X_BF16_W = False
# Without the W column broadcast (R1: every core reads its whole W slice on the writer -- decode, group_h > 1) only
# the flip applies. 1x7168: fp32 W wins (bf16 X 58.7 -> 46.1, fp32 X 130.7 -> 119.8 us), bf16 W loses (fp32 X 80.1
# -> 93.2, bf16 X 40.8 -> 40.3) -- a bf16 W there keeps the Refinement 4 placement.
PLACEMENT_LEVERS_BF16_W_R1 = False


def _other_noc(noc):
    return ttnn.NOC.NOC_1 if noc == ttnn.NOC.NOC_0 else ttnn.NOC.NOC_0


def _placement_levers(x_dtype, w_dtype, w_bcast):
    """Whether the derived flip rows + the reader-side W share apply (measured path gate, see above)."""
    if w_dtype != ttnn.bfloat16:
        return True
    if not w_bcast:  # per-core DRAM W (R1: decode / group_h > 1): the flip only wins with an fp32 W
        return PLACEMENT_LEVERS_BF16_W_R1
    return PLACEMENT_LEVERS_BF16_X_BF16_W or x_dtype != ttnn.bfloat16


def _flip_rows(grid_y, levers):
    if READER_NOC_FLIP_ROWS is not None:
        return READER_NOC_FLIP_ROWS
    return int(round(READER_NOC_FLIP_FRACTION * grid_y)) if levers else 0


def _reader_noc_of(group_y0, flip_rows):
    return _other_noc(READER_NOC) if group_y0 < flip_rows else READER_NOC


# Stream-column tiles moved off rank 0 when rank 0 owns every Sinkhorn row (one token tile-row per group): the
# owner's tail is its projection / sum x^2 / y-mix + the Sinkhorn, every other rank's only the former
# (Refinement 4). Sized ~ Sinkhorn time / per-C-tile tail time x (G-1)/G (~6.5 us / ~0.7 us x 4/5 ~ 7), both
# independent of C. Measured (BH, bf16 X, device ns, with the fused owned block): 640x1792 46.8 (even) -> 44.6
# us, 640x7168 157.1 -> 151.9 us. Note: the stream stride C/32 is a multiple of the DRAM bank count for these
# shapes, so a rank's reads walk banks (c_start + c) mod banks; discounts that make two ranks' c_start collide
# mod banks measure slower (640x1792: 4 -> 48.7, 8 -> 45.4 us). 0 = even split.
OWNER_C_DISCOUNT = 7
W_ROLE_DRAM, W_ROLE_SPREAD = 0, 1  # mirrors the writer's W_ROLE_* constants
W_MCAST_PLACEHOLDER_CT = [0, SEM_W_READY, 0xFFFFFFFF, 0, 0x2, 0]  # inactive McastArgs wire (Counter)


# Measurement hook (perf tournaments): extra preprocessor defines for ALL three kernels, read from the environment
# as MHC_PRE_KERNEL_DEFINES="A=1;B". Unset = no defines (the production program). KERNEL_PERF_ZONES turns on the
# permanent per-stage MaybeDeviceZoneScope zones (with TT_METAL_DEVICE_PROFILER=1); MHC_ABLATE_* stub a stage's
# payload for ablation runs (outputs are then garbage).
def _kernel_defines():
    defines = []
    for item in os.environ.get("MHC_PRE_KERNEL_DEFINES", "").split(
        ";"
    ):  # diagnostic: perf/measurement knob, same result
        if item.strip():
            name, _, value = item.partition("=")
            defines.append((name.strip(), value.strip() or "1"))
    return defines


def _dm_config(processor, noc):
    cfg = ttnn.DataMovementConfigDescriptor()
    cfg.processor = processor
    cfg.noc = noc
    return cfg


def _f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _split(total, parts):
    """ceil/floor split: the first `total % parts` parts get the ceil."""
    base, rem = divmod(total, parts)
    sizes = [base + (1 if p < rem else 0) for p in range(parts)]
    starts = [0] * parts
    for p in range(1, parts):
        starts[p] = starts[p - 1] + sizes[p - 1]
    return sizes, starts


def _c_split(Ct, group_cores, owner_fixed):
    """Per-rank stream-column split. owner_fixed (every group has <= 1 token tile-row, so rank 0 owns every
    Sinkhorn row): rank 0 gets OWNER_C_DISCOUNT tiles fewer than the even share (>= 1 kept), the other ranks
    split the rest evenly, so the owner's y-mix / projection shrink by what its Sinkhorn adds."""
    even = Ct // group_cores
    d = min(OWNER_C_DISCOUNT, even - 1) if owner_fixed and group_cores > 1 else 0
    if d <= 0:
        return _split(Ct, group_cores)
    rest, rest_starts = _split(Ct - (even - d), group_cores - 1)
    sizes = [even - d] + rest
    return sizes, [0] + [even - d + st for st in rest_starts]


@dataclass
class Plan:
    n: int
    Mt: int
    Ct: int
    Kt: int
    grid_x: int
    grid_y: int
    group_w: int
    group_h: int
    group_cores: int
    groups_x: int
    groups_y: int
    num_groups: int
    core_c_tiles: list
    c_start: list
    core_token_tiles: list  # per group
    t_start: list  # per group
    core_k_tiles_max: int
    block_token_tiles: int
    x_block_depth: int
    y_chunk_tiles: int
    y_depth: int


def _cb_table(*, bt, depth, kmax, G, y_chunk, n, x_tile, w_tile, y_tile, x_dtype, w_dtype, y_dtype):
    """THE per-core CB inventory: [(cb_index, num_pages, page_bytes, data_format, aliases)].

    `aliases` = ((cb_index, page_bytes, data_format), ...) extra CB indices sharing the SAME allocation
    (they cost no L1). cb_weight_split aliases cb_weight: the compute kernel rewrites each fp32 W tile k
    in place into the bf16 pair [W_hi(k), W_lo(k)] (same 4096 B), so the split costs zero extra L1.

    Single source of truth for every CB size: the L1 selection function (`_l1_bytes`) and the
    ProgramDescriptor (`create_program_descriptor`) both read this table, so a knob turn lands in one
    place. Mirrors l1_ledger.md row by row.
    """
    f32, fT = ttnn.float32, F32_TILE_BYTES
    pieces = w_pieces(w_dtype)
    w_alias = ((CB_WEIGHT_SPLIT, w_tile // pieces, ttnn.bfloat16),) if pieces > 1 else ()
    xp = x_pieces(x_dtype)
    x_alias = ((CB_X_FP32, x_tile, x_dtype),) if xp > 1 else ()
    sb_rows = min(bt, DEST_TILES_FP32)  # projection sub-block rows (<= DEST); concave in bt -> the affine
    # L1 solve over-estimates it (conservative)
    table = [
        (CB_X_RESIDENT, depth * bt * kmax, x_tile, x_dtype, x_alias),
        (CB_WEIGHT, kmax, w_tile, w_dtype, w_alias),
        (CB_BIAS_COEF, 1, fT, f32),
        (CB_REDUCE_SCALER, 1, BF16_TILE_BYTES, ttnn.bfloat16),
        # fp32 X: one exact sum x^2 tile per row; bf16 X: one partial per (row, K chunk) (streamed with the
        # fp32-W projection; a bf16 W keeps one whole-row window, using the first page)
        (CB_SQ_ACC, bt if xp > 1 else bt * X_STREAM_CHUNKS, fT, f32),
        (CB_PARTIAL, 2 * bt, fT, f32),
        (CB_GATHERED, G * 2 * bt, fT, f32),
        (CB_COMBINED, 2 * bt, fT, f32),
        # landed S blocks [mix x bt | sum(x^2) x bt] (group multicast): 2 blocks, S(b+1) lands while S(b) is held
        # (Perf 1 cross-block pipeline; the handshake-free multicast relies on the second slot, see the writer)
        (CB_COEF_IN, COEF_IN_BLOCKS * 2 * bt, fT, f32),
        (CB_COMB_COEF, 2 * bt, fT, f32),  # owned rows: [post, comb] row-major output tiles
        (CB_COEF_KEEP, 1, fT, f32),  # owned row: coefficient-major tile, packed then reloaded in place
        (CB_PRE_COLS, n * bt, fT, f32),
        (CB_Y_OUT, Y_DEPTH * y_chunk, y_tile, y_dtype),
        (CB_W_OWN_READY, 1, TOKEN_PAGE_BYTES, ttnn.bfloat16),
        (CB_W_OWN_SPLIT, 1, TOKEN_PAGE_BYTES, ttnn.bfloat16),
        (CB_W_SHARE_LANDED, 1, TOKEN_PAGE_BYTES, ttnn.bfloat16),
    ]
    if xp == 1:
        table += [(CB_MIX_RUN, 1, fT, f32)]  # running fp32 mix between the streamed K chunk windows
    if xp > 1:
        table += [
            (CB_X_PIECES, X_PIECE_DEPTH * xp * X_CHUNK_K_TILES * sb_rows, BF16_TILE_BYTES, ttnn.bfloat16),
            (CB_MIX_RUN, sb_rows, fT, f32),
            (CB_MAX_LANES, 1, fT, f32),
            (CB_MAX_SCALAR, 1, fT, f32),
            (CB_GRID, max(bt, 1), fT, f32),
            (CB_MAX_SCALER, 1, BF16_TILE_BYTES, ttnn.bfloat16),
        ]
    return [e if len(e) == 5 else e + ((),) for e in table]


def _l1_bytes(**table_kwargs):
    """Per-core L1 footprint (identical on every core; see l1_ledger.md 'Total per-core footprint')."""
    return sum(pages * page_bytes for _, pages, page_bytes, _, _ in _cb_table(**table_kwargs))


def make_plan(device, x_tensor, w_tensor, n):
    padded = list(x_tensor.padded_shape)
    lead_tiles = 1
    for d in padded[:-2]:
        lead_tiles *= d
    Mt = lead_tiles * (padded[-2] // TILE)  # per-image tile padding is already in the padded shape
    C = x_tensor.shape[-1] // n
    Ct = C // TILE
    Kt = n * Ct

    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = grid.x, grid.y

    x_tile = x_tensor.buffer_page_size()
    w_tile = w_tensor.buffer_page_size()
    y_tile = x_tile
    dtypes = dict(x_dtype=x_tensor.dtype, w_dtype=w_tensor.dtype, y_dtype=x_tensor.dtype)
    budget = ttnn.get_max_worker_l1_unreserved_size() - L1_SAFETY_MARGIN

    def fit(group_w, group_h):
        """(bt, depth, geometry) of a group_w x group_h group shape, or None if no block fits L1."""
        group_cores = group_w * group_h
        groups_x, groups_y = grid_x // group_w, grid_y // group_h
        core_token_tiles, t_start = _split(Mt, groups_x * groups_y)
        ctt_max = max(core_token_tiles)
        c_tiles, c_starts = _c_split(Ct, group_cores, owner_fixed=ctt_max <= 1)
        kmax = n * max(c_tiles)
        y_chunk = min(max(c_tiles), Y_CHUNK_TILES_CAP)

        def l1_at(bt_, depth_):
            return _l1_bytes(
                bt=bt_,
                depth=depth_,
                kmax=kmax,
                G=group_cores,
                y_chunk=y_chunk,
                n=n,
                x_tile=x_tile,
                w_tile=w_tile,
                y_tile=y_tile,
                **dtypes,
            )

        bt_cap = min(ctt_max, BLOCK_TOKEN_TILES_CAP)
        for depth in (X_BLOCK_DEPTH_DEFAULT, 1):  # depth-1 fallback only if the default does not fit
            fixed = l1_at(0, depth)  # the footprint is affine in bt
            per_bt = l1_at(1, depth) - fixed
            bt = min(bt_cap, (budget - fixed) // per_bt) if budget > fixed else 0
            if bt >= 1:
                return dict(
                    group_w=group_w,
                    group_h=group_h,
                    group_cores=group_cores,
                    kmax=kmax,
                    y_chunk=y_chunk,
                    groups_x=groups_x,
                    groups_y=groups_y,
                    core_token_tiles=core_token_tiles,
                    t_start=t_start,
                    c_tiles=c_tiles,
                    c_starts=c_starts,
                    bt=bt,
                    depth=depth,
                    blocks=math.ceil(ctt_max / bt),
                )
        return None

    chosen = None
    if Mt >= grid_y and NARROW_GROUPS and x_pieces(x_tensor.dtype) == 1:
        # Group width = the core-assignment knob (design perf lamp L2): the plan that can prefetch X, then the
        # lowest block-schedule cost (see NARROW_GROUPS), widest on a tie (min keeps the first), subject to the
        # L1 fit. For bf16 X the one-time W prelude is spread over the column (W column all-gather) or absent,
        # but the fp32-X W grid split needs the whole slice's max, so it grows with the slice: measured a loss
        # there (1280x4096 415 -> 506 us, 4096x1792 634 -> 712 us), and fp32 X keeps the full-row width.
        fits = [f for f in (fit(group_w, 1) for group_w in range(min(grid_x, Ct), 0, -1)) if f is not None]
        if fits:
            streamed = w_pieces(w_tensor.dtype) > 1  # bf16 X + fp32 W: the projection streams under the X read
            chosen = min(
                fits,
                key=lambda f: (f["depth"] < 2 and f["blocks"] > 1, _block_schedule_cost(f, n, streamed)),
            )
    if chosen is None:
        group_w = min(grid_x, Ct)
        if Mt >= grid_y:
            group_h = 1
        else:
            group_h = max(1, min(grid_y // Mt, Ct // group_w, GROUP_CORES_CAP // group_w))
        while True:
            chosen = fit(group_w, group_h)
            if chosen is not None:
                break
            # Still does not fit: grow the group (smaller per-rank K slice).
            if group_h * 2 <= grid_y and group_w * group_h * 2 <= GROUP_CORES_CAP and group_w * group_h * 2 <= Ct:
                group_h *= 2
                continue
            raise RuntimeError(
                f"mhc_pre: no blocking fits L1 (C={C}, Mt={Mt}, grid={grid_x}x{grid_y}, budget={budget} B)"
            )
    group_w, group_h, group_cores = chosen["group_w"], chosen["group_h"], chosen["group_cores"]
    groups_x, groups_y = chosen["groups_x"], chosen["groups_y"]
    num_groups = groups_x * groups_y
    core_token_tiles, t_start = chosen["core_token_tiles"], chosen["t_start"]
    kmax, y_chunk, bt, depth = chosen["kmax"], chosen["y_chunk"], chosen["bt"], chosen["depth"]

    core_c_tiles, c_start = chosen["c_tiles"], chosen["c_starts"]
    return Plan(
        n=n,
        Mt=Mt,
        Ct=Ct,
        Kt=Kt,
        grid_x=grid_x,
        grid_y=grid_y,
        group_w=group_w,
        group_h=group_h,
        group_cores=group_cores,
        groups_x=groups_x,
        groups_y=groups_y,
        num_groups=num_groups,
        core_c_tiles=core_c_tiles,
        c_start=c_start,
        core_token_tiles=core_token_tiles,
        t_start=t_start,
        core_k_tiles_max=kmax,
        block_token_tiles=bt,
        x_block_depth=depth,
        y_chunk_tiles=y_chunk,
        y_depth=Y_DEPTH,
    )


def _cb(index, core_ranges, num_pages, page_bytes, dtype, aliases=()):
    fmts = [(index, page_bytes, dtype)] + list(aliases)
    return ttnn.CBDescriptor(
        total_size=num_pages * page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=d, page_size=pb) for i, pb, d in fmts],
    )


def create_program_descriptor(
    x_tensor, w_tensor, b_tensor, y_tensor, post_tensor, comb_tensor, *, n, scale, sinkhorn_iters, eps, norm_eps, cfg
):
    device = x_tensor.device()
    plan = make_plan(device, x_tensor, w_tensor, n)
    G = plan.group_cores
    bt = plan.block_token_tiles
    kmax = plan.core_k_tiles_max

    # ---- active groups (a group with 0 token rows is not launched) ----
    groups = []
    for g in range(plan.num_groups):
        if plan.core_token_tiles[g] == 0:
            continue
        gx0 = (g % plan.groups_x) * plan.group_w
        gy0 = (g // plan.groups_x) * plan.group_h
        groups.append((g, gx0, gy0))
    ranges = [
        ttnn.CoreRange(ttnn.CoreCoord(gx0, gy0), ttnn.CoreCoord(gx0 + plan.group_w - 1, gy0 + plan.group_h - 1))
        for _, gx0, gy0 in groups
    ]
    all_cores = ttnn.CoreRangeSet(ranges)

    x_tile = x_tensor.buffer_page_size()
    w_tile = w_tensor.buffer_page_size()
    y_tile = y_tensor.buffer_page_size()

    # ---- CBs: identical descriptors on every launched core (uniform L1 addresses for remote writes) ----
    cbs = [
        _cb(index, all_cores, pages, page_bytes, fmt, aliases)
        for index, pages, page_bytes, fmt, aliases in _cb_table(
            bt=bt,
            depth=plan.x_block_depth,
            kmax=kmax,
            G=G,
            y_chunk=plan.y_chunk_tiles,
            n=n,
            x_tile=x_tile,
            w_tile=w_tile,
            y_tile=y_tile,
            x_dtype=x_tensor.dtype,
            w_dtype=w_tensor.dtype,
            y_dtype=y_tensor.dtype,
        )
    ]

    semaphores = [
        ttnn.SemaphoreDescriptor(id=SEM_GATHER, core_ranges=all_cores, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_READY, core_ranges=all_cores, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_CONSUMED, core_ranges=all_cores, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_W_READY, core_ranges=all_cores, initial_value=0),
    ]

    # ---- W column broadcast (R2): one Mcast1D(PerColumn) over the active rectangle, sender = row 0 ----
    # Kernel sets: the groups sharing one (reader NoC, writer NoC) placement get their own reader / writer
    # descriptors (the NoC is a kernel-config property; the multicast wires depend on it too).
    active_rows = len(groups) // plan.groups_x
    w_bcast = W_BCAST and plan.group_h == 1 and len(groups) % plan.groups_x == 0 and active_rows >= 2
    levers = _placement_levers(x_tensor.dtype, w_tensor.dtype, w_bcast)
    flip_rows = _flip_rows(plan.grid_y, levers)
    reader_nocs = sorted({_reader_noc_of(gy0, flip_rows) for _, _, gy0 in groups}, key=lambda c: c.value)

    w_mcast = {}  # reader NoC of the kernel set -> Mcast1D (same rectangle and semaphore; senders may ride either NoC)
    if w_bcast:
        w_rect = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(plan.groups_x * plan.group_w - 1, active_rows - 1))]
        )
        for rnoc in reader_nocs:
            w_cfg = ttnn.McastConfig(
                noc=_other_noc(rnoc),
                handshake=False,
                data_ready=ttnn.McastDataReady.Counter,
                rotating_sender=True,
                sem_ids=[SEM_W_READY],
            )
            w_mcast[rnoc] = ttnn.Mcast1D(device, w_rect, ttnn.Mcast1DShape.PerColumn, 0, w_cfg)

    # fp32 W hi/lo split done per column share (bf16 X only: the fp32-X grid split needs the whole slice's max).
    w_presplit = bool(w_mcast) and w_pieces(w_tensor.dtype) > 1 and x_pieces(x_tensor.dtype) == 1

    # ---- group combine mcast (one Mcast2D per group; identical CT wire across groups) ----
    helpers = {}
    mcast_ct = {}  # reader NoC of the kernel set -> the group-mcast CT wire (identical within a set)
    if G > 1:
        for g, gx0, gy0 in groups:
            rnoc = _reader_noc_of(gy0, flip_rows)
            # No consumer-ready handshake (Perf 1): cb_coef_in holds 2 blocks and the root multicasts S(b+1) only after
            # gathering every P(b+1), so the landing slot is free and the Flag was reset (see the writer). A Counter
            # data-ready signal hangs here (its non-posted multicast atomic also waits for the looped-back root's ack).
            mcast_cfg = ttnn.McastConfig(noc=_other_noc(rnoc), handshake=False, sem_ids=[SEM_MCAST_READY])
            rect = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(
                        ttnn.CoreCoord(gx0, gy0), ttnn.CoreCoord(gx0 + plan.group_w - 1, gy0 + plan.group_h - 1)
                    )
                ]
            )
            helpers[g] = ttnn.Mcast2D(device, rect, ttnn.CoreCoord(gx0, gy0), mcast_cfg)
            ct = list(helpers[g].compile_time_args())
            assert mcast_ct.get(rnoc, ct) == ct, "mcast CT wire must be identical across the groups of a set"
            mcast_ct[rnoc] = ct
    else:
        # group_cores == 1: no receivers, the pipe is never used. Placeholder wire with real sem ids.
        mcast_ct = {rnoc: [0, SEM_MCAST_READY, SEM_MCAST_CONSUMED, 0, 1, 0] for rnoc in reader_nocs}

    # ---- kernel CT args ----
    reader_ct = [
        CB_X_RESIDENT,
        CB_REDUCE_SCALER,
        n,
        bt,
        kmax,
        plan.Ct,
        CB_MAX_SCALER,
        int(x_pieces(x_tensor.dtype) > 1),
        X_STREAM_CHUNKS,
        X_STREAM_INFLIGHT,
        CB_WEIGHT,
        CB_W_SHARE_LANDED,
        int(W_SHARE_BEFORE_X),
    ]
    assert len(reader_ct) == 13  # TensorAccessorArgs base in the reader
    reader_ct += ttnn.TensorAccessorArgs(x_tensor).get_compile_time_args()
    reader_ct += ttnn.TensorAccessorArgs(w_tensor).get_compile_time_args()

    compute_ct = [
        CB_X_RESIDENT,
        CB_WEIGHT,
        CB_BIAS_COEF,
        CB_REDUCE_SCALER,
        CB_SQ_ACC,
        CB_PARTIAL,
        CB_GATHERED,
        CB_COMBINED,
        CB_COEF_IN,
        CB_COEF_KEEP,
        0,  # CT 10 unused (formerly a writer-scattered coefficient CB)
        CB_COMB_COEF,
        CB_PRE_COLS,
        CB_Y_OUT,
        n,
        bt,
        kmax,
        G,
        CB_WEIGHT_SPLIT,
        w_pieces(w_tensor.dtype),
        W_CHUNK_TILES,
        W_LO_FIDELITY.value,
        cfg.math_fidelity.value,
        CB_X_FP32,
        CB_X_PIECES,
        CB_MIX_RUN,
        x_pieces(x_tensor.dtype),
        X_CHUNK_K_TILES,
        min(bt, DEST_TILES_FP32),
        CB_MAX_LANES,
        CB_MAX_SCALAR,
        CB_GRID,
        CB_MAX_SCALER,
        X_GRID_BITS,
        W_GRID_BITS,
        PRODUCT_ORDER_MAX,
        PRODUCT_LO_ORDER,
        X_LO_FIDELITY.value,
        CB_W_OWN_READY,
        CB_W_OWN_SPLIT,
        int(w_presplit),
        X_STREAM_CHUNKS,
        plan.x_block_depth,  # block schedule (cross-block pipeline)
    ]

    writer_ct = [
        CB_PARTIAL,
        CB_GATHERED,
        CB_COMBINED,
        CB_COEF_IN,
        CB_COMB_COEF,
        CB_Y_OUT,
        n,
        bt,
        plan.Ct,
        G,
        plan.y_chunk_tiles,
        plan.y_depth * plan.y_chunk_tiles,
        SEM_GATHER,
        n * (n + 2),
        # The writer also produces the resident constants: the coefficient-major bias and the W slice (NoC1).
        CB_BIAS_COEF,
        CB_WEIGHT,
        W_CHUNK_TILES,
        CB_W_OWN_READY,
        CB_W_OWN_SPLIT,
        int(w_presplit),
        CB_W_SHARE_LANDED,
        plan.x_block_depth,  # block schedule (cross-block pipeline; mirrors the compute)
    ]
    assert len(writer_ct) == 22  # MCAST_CT_BASE in the writer
    writer_tail_ct = []
    writer_tail_ct += ttnn.TensorAccessorArgs(y_tensor).get_compile_time_args()
    writer_tail_ct += ttnn.TensorAccessorArgs(post_tensor).get_compile_time_args()
    writer_tail_ct += ttnn.TensorAccessorArgs(comb_tensor).get_compile_time_args()
    writer_tail_ct += ttnn.TensorAccessorArgs(b_tensor).get_compile_time_args()
    writer_tail_ct += ttnn.TensorAccessorArgs(w_tensor).get_compile_time_args()

    def writer_ct_of(rnoc):
        w_ct = list(w_mcast[rnoc].compile_time_args()) if w_mcast else W_MCAST_PLACEHOLDER_CT
        return writer_ct + mcast_ct[rnoc] + w_ct + writer_tail_ct

    # ---- per-core RT args ----
    a_pre, a_post, a_res = scale
    C = plan.Ct * TILE
    scalar_bits = [
        _f32_bits(a_pre),
        _f32_bits(a_post),
        _f32_bits(a_res),
        _f32_bits(eps),
        _f32_bits(norm_eps),
        _f32_bits(1.0 / (n * C)),
        int(sinkhorn_iters),
    ]
    reader_rt = {rnoc: ttnn.RuntimeArgs() for rnoc in reader_nocs}
    writer_rt = {rnoc: ttnn.RuntimeArgs() for rnoc in reader_nocs}
    compute_rt = ttnn.RuntimeArgs()
    for g, gx0, gy0 in groups:
        rnoc = _reader_noc_of(gy0, flip_rows)
        ctt = plan.core_token_tiles[g]
        ts = plan.t_start[g]
        num_blocks = math.ceil(ctt / bt)
        root_virtual = device.worker_core_from_logical_core(ttnn.CoreCoord(gx0, gy0))
        for dy in range(plan.group_h):
            for dx in range(plan.group_w):
                x, y = gx0 + dx, gy0 + dy
                rank = dy * plan.group_w + dx
                cc = plan.core_c_tiles[rank]
                cs = plan.c_start[rank]
                own = [0, 0]
                share_on_reader = 0
                if not w_mcast:
                    w_rt, w_mc_rt = [W_ROLE_DRAM, 0, 0, 0, 0], [0, 0, 0, 0]
                else:
                    # Column share: row y of the column reads / splits / multicasts W tiles [own0, own1).
                    sizes, starts = _split(n * cc, active_rows)
                    own = [starts[y], starts[y] + sizes[y]]
                    events = sum(1 for sz in sizes if sz > 0) - (1 if sizes[y] > 0 else 0)
                    share_on_reader = int(W_SHARE_ON_READER and levers)
                    w_rt = [W_ROLE_SPREAD] + own + [events, share_on_reader]
                    w_mc_rt = list(w_mcast[rnoc].runtime_args(ttnn.CoreCoord(x, y)))
                reader_rt[rnoc][x][y] = [x_tensor.buffer_address(), ts, ctt, cs, cc, num_blocks] + (
                    [w_tensor.buffer_address(), share_on_reader] + own
                )
                mcast_rt = list(helpers[g].runtime_args(ttnn.CoreCoord(x, y))) if G > 1 else [0, 0, 0, 0]
                writer_rt[rnoc][x][y] = (
                    [
                        y_tensor.buffer_address(),
                        post_tensor.buffer_address(),
                        comb_tensor.buffer_address(),
                        ts,
                        ctt,
                        cs,
                        cc,
                        num_blocks,
                        rank,
                        root_virtual.x,
                        root_virtual.y,
                        b_tensor.buffer_address(),
                        w_tensor.buffer_address(),
                    ]
                    + w_rt
                    + mcast_rt
                    + w_mc_rt
                )
                compute_rt[x][y] = [num_blocks, ctt, cc, rank] + scalar_bits + own

    kernel_defines = _kernel_defines()
    dm_kernels = []
    for rnoc in reader_nocs:
        set_cores = ttnn.CoreRangeSet(
            [r for (_, _, gy0), r in zip(groups, ranges) if _reader_noc_of(gy0, flip_rows) == rnoc]
        )
        dm_kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "mhc_pre_reader.cpp"),
                core_ranges=set_cores,
                compile_time_args=reader_ct,
                runtime_args=reader_rt[rnoc],
                defines=kernel_defines,
                config=_dm_config(ttnn.DataMovementProcessor.RISCV_1, rnoc),
            )
        )
        dm_kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "mhc_pre_writer.cpp"),
                core_ranges=set_cores,
                compile_time_args=writer_ct_of(rnoc),
                runtime_args=writer_rt[rnoc],
                defines=kernel_defines,
                config=_dm_config(ttnn.DataMovementProcessor.RISCV_0, _other_noc(rnoc)),
            )
        )
    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=cfg.math_fidelity,
        fp32_dest_acc_en=True,
        math_approx_mode=cfg.math_approx_mode,
    )
    modes = [ttnn.UnpackToDestMode.Default] * NUM_CB_SLOTS
    # The fp32 W CB is only read by the split's copy_tile (the matmul reads cb_weight_split); a bf16 W
    # feeds the FPU matmul directly and must stay Default.
    fp32_cbs = UNPACK_TO_DEST_FP32_CBS + ((CB_WEIGHT,) if w_pieces(w_tensor.dtype) > 1 else ())
    # fp32 X: the FPU y-mix keeps reading CB_X_RESIDENT; the exact split / sum x^2 read the
    # alias CB_X_FP32 straight into DEST. CB_MIX_RUN is reloaded exactly between K chunks.
    fp32_cbs += (CB_X_FP32, CB_GRID, CB_MAX_SCALAR) if x_pieces(x_tensor.dtype) > 1 else ()
    fp32_cbs += (CB_MIX_RUN,)  # reloaded exactly between K chunk windows (both X paths)
    for idx in fp32_cbs:
        modes[idx] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_cfg.unpack_to_dest_mode = modes
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "mhc_pre_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        defines=kernel_defines,
        config=compute_cfg,
    )
    return ttnn.ProgramDescriptor(kernels=dm_kernels + [compute], semaphores=semaphores, cbs=cbs), plan
