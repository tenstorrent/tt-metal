# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for rms_norm.

Two regimes (selected by the host heuristic):
  - Regime A: row-parallel. Each core owns a disjoint set of tile-rows and holds a
    full tile-row (Wt tiles) resident; zero cross-core communication.
  - Regime B: wide-W cross-core W-split. Each core owns a Wt/K shard of a row-group,
    computes its local partial sum-of-squares, all-gathers the K partials over an
    mcast rectangle, sums them, then normalizes its own shard.

The A-vs-B decision is by L1 fit: Regime A whenever a full row (input + gamma if
present) fits the resident budget; otherwise Regime B with K chosen to split W into
rectangular bands that saturate the grid.
"""

from pathlib import Path
import math
import struct

import ttnn


KERNEL_DIR = Path(__file__).parent / "kernels"

# CB indices (semantic names live at the kernel boundary).
CB_INPUT_RESIDENT = 0
CB_GAMMA = 1
CB_SCALER = 8
CB_PARTIALS_GATHERED = 9
CB_OUTPUT = 16
CB_SQUARED = 24
CB_PARTIAL_SUMSQ = 25
CB_RECIP_RMS = 26
CB_NORMALIZED = 27
# Regime B only: single-push handoff of the fully-accumulated local Sum(x^2) from
# compute (PASS-1) to the mcast reader. Decoupled from CB_PARTIAL_SUMSQ so the reader's
# cb_wait_front observes only the final value (Refinement 1 correctness fix).
CB_LOCAL_SUMSQ = 28
# Refinement 3 (TILE input + ROW_MAJOR gamma): the reader stages row-major gamma
# sticks here and compute tilizes them once into CB_GAMMA. Only allocated when
# gamma is supplied ROW_MAJOR with a TILE input.
CB_GAMMA_RM = 3

# Resident budget in tiles (≈1.12 MB for bf16), per op_design.md P3.
# SUPERSEDED by the byte-aware L1_RESIDENT_BUDGET_BYTES (Refinement 6): the OOM floor
# and _select_k now bound the resident footprint in *bytes* (dtype-correct, and
# including R5's enlarged cb_squared) rather than a bf16-only tile count. Kept only as
# documentation of the original P3 budget; no longer consulted by any decision.
RESIDENT_BUDGET_TILES = 560

# --- Refinement 6: A-vs-B crossover (measured, not assumed) ---------------------
# The old heuristic chose Regime B "whenever a rectangular partition adds cores"
# (Ht_total*K > regime_a_cores) with NO cost model for the mcast all-gather. But
# the all-gather has a large fixed cost (semaphores + mcast rounds + the K-partial
# combine): for a row that FITS one core's L1 and is only moderately wide, paying
# that overhead to gain cores is *slower* than staying in single/few-core Regime A.
#
# Two-tier decision (encoded in create_program_descriptor):
#   1. OOM floor (hard): when a full row does NOT fit one core's resident L1 budget,
#      the W-split is mandatory — Regime B is the only option (correctness, never a
#      perf tradeoff).
#   2. Perf crossover: when the row DOES fit L1, only go B if the W-split adds cores
#      AND Wt >= REGIME_B_MIN_WT. Below the crossover, Regime A's single/few-core
#      run beats B's mcast setup cost.
#
# The crossover is measured against the FINAL, K-tuned Regime B (see _select_k —
# the proxy-min-K policy makes B ~3-6x faster than the old maximize-K version). With
# that fast B, the crossover is LOW and the same for both layouts: B wins from Wt>=16.
# Measured (A-forced vs B-forced device-kernel-ns, single-tile-row (1,1,32,W), 8x8
# Wormhole grid, see tests/.../test_rms_norm_perf.py):
#   TILE: Wt=8 A=9.8 B=11.3us (A wins, 15%); Wt=16 A=15.7 B=12.0 (B); Wt=64 A=52 B=17
#         (B 3x); Wt=128 A=100 B=23 (B 4x). Regime A scales ~linearly with Wt (one
#         core does the whole row); K-tuned B grows slowly (~11->23us over Wt 8->128).
#   ROW_MAJOR: Wt=8 A=13.7 B=14.8 (A); Wt=16 A=22.6 B=15.2 (B); Wt=64 A=75 B=20 (B).
# So A only wins for a single tile-wide row (Wt=8, W=256); everything wider that fits
# a rectangular W-split is faster in (K-tuned) B. Threshold 16 keeps Wt=8 in A and
# routes Wt>=16 to B. (Earlier intermediate measurements gave 160/96, but that was
# BEFORE the _select_k K-tuning — those numbers are stale; the K-tuning is what makes
# the low crossover correct.)
REGIME_B_MIN_WT_TILE = 16
REGIME_B_MIN_WT_RM = 16

# Precision ceiling on the crossover (NOT a perf number — a correctness floor):
# Regime A computes Σx² as a SINGLE reduce over the whole resident shard. With
# fp32_dest_acc_en=False that reduce accumulates in a bf16 DEST, so a wide shard
# (many tiles summed in bf16) loses precision — measured: a Wt=128 (W=4096) bf16
# row + gamma with fp32_dest_acc_en=False overshoots the 0.052 relative-Frobenius
# band in Regime A, while Regime B (narrow Wt_s shards, each reduced separately and
# plain-summed) stays inside it. Regime B is therefore the precision-safe choice for
# wide rows when fp32 accumulation is OFF. When fp32_dest_acc_en=True the reduce
# accumulates in fp32 and stays precise at any width, so the perf crossover governs.
# Empirically the bf16-accumulation cliff sits between Wt=64 (passes in A) and
# Wt=128 (fails in A): 96 forces the precision-risky rows (Wt>=96) into B.
# NOTE: with the K-tuned perf crossover now at 16 (< 96), this ceiling is currently
# subsumed — every Wt>=16 row already routes to B on perf grounds, so the precision
# cliff at Wt~128 is never reached in A. It is retained as a defensive floor: if the
# perf crossover is ever raised above 96, this keeps fp32_acc=False wide rows in B.
REGIME_B_PRECISION_CEILING_NO_FP32ACC = 96


def _b_min_wt(layout_is_rm: bool, fp32_acc: bool) -> int:
    """Smallest Wt at which the heuristic routes a row-fitting shape to Regime B.

    The threshold is the min of the (measured) perf crossover and the precision
    ceiling: when fp32 accumulation is off, Regime A's wide bf16 reduce forces B
    earlier than perf alone would.
    """
    perf = REGIME_B_MIN_WT_RM if layout_is_rm else REGIME_B_MIN_WT_TILE
    if not fp32_acc:
        return min(perf, REGIME_B_PRECISION_CEILING_NO_FP32ACC)
    return perf


# Measurement-only hook (perf tests). None -> use the heuristic. "A"/"B" -> force
# that regime for an apples-to-apples A-vs-B crossover measurement on one shape.
# NEVER set in production; the public entry point does not touch it. When forcing a
# regime that is infeasible for the shape (e.g. "A" on an OOM row, or "B" with no
# rectangular partition) the descriptor falls back to the heuristic choice.
_FORCE_REGIME = None
# Measurement-only hook (perf tests). None -> _select_k picks the largest valid K.
# An int -> force that K (if it qualifies) to sweep the W-split factor. Never set in
# production.
_FORCE_K = None
# Measurement-only hook (perf tests, Refinement 7). None -> _regime_a_block_height
# picks bh from the heuristic. An int -> force that BLOCK_HEIGHT (caller's
# responsibility to pick a value that divides the per-core row count). Used to
# measure the bh=1 baseline vs the row-blocked path on the same shape. Never set in
# production.
_FORCE_BH = None

# --- Refinement 9 (Part A): cross-core all-reduce transport selector ---------------
# The Regime-B all-reduce of the K partial Σx² has two implementations in the unified
# reader, gated by the `transport_mode` compile-time arg:
#   0 (TRANSPORT_MCAST_ALLGATHER) — baseline: K-round rotating-sender mcast all-gather
#       (every one of the K cores mcasts its partial to the K-1 others). O(K) serialized
#       mcast rounds; the dominant non-reduce term in Regime B (R6/R8).
#   1 (TRANSPORT_ROOT_RELAY) — root-relay gather-then-broadcast: rank 0 unicast-reads all
#       K-1 peer partials (one parallel read phase gated by a "produced" counter), then a
#       SINGLE mcast of the assembled K-tile block to the group. O(1) transport phases.
#       Compute is identical (every core still combines the K gathered partials).
# Production default is decided by _select_transport. _FORCE_TRANSPORT (perf tests only)
# overrides it for the transport bake-off; NEVER set in production.
TRANSPORT_MCAST_ALLGATHER = 0
TRANSPORT_ROOT_RELAY = 1
_FORCE_TRANSPORT = None
# Semaphore id for the mode-1 "produced" counter (peers -> root). Distinct from the
# DATA_READY (0) / CONSUMED (1) pair used by the broadcast mcast pipe.
PRODUCED_SEM = 2


def _select_transport(K):
    """Choose the Regime-B all-reduce transport.

    Refinement 9 (Part A) bake-off result: the root-relay gather-then-broadcast (mode 1)
    beats the baseline rotating-sender mcast all-gather (mode 0) on EVERY measured wide-W
    Regime-B shape — 1.10x (Wt=128, K=16) to 1.48x (Wt=1024, K=32) — because it collapses
    the baseline's O(K) serialized mcast rounds to O(1) transport phases (one parallel
    gather + one mcast). The win grows with K, so it is the production default for all K.
    _FORCE_TRANSPORT (perf tests only) overrides for the bake-off measurement itself."""
    if _FORCE_TRANSPORT is not None:
        return _FORCE_TRANSPORT
    return TRANSPORT_ROOT_RELAY


# ---- ROW_MAJOR (tilize-wrapped) regime CB indices ----
# Refinement 3: ROW_MAJOR input/output handled natively via a tilize-wrapped,
# row-parallel path. The math is identical to Regime A but reads/writes
# row-major sticks. Distinct from the TILE-regime CB map above.
CB_RM_IN = 0  # row-major input sticks (reader -> compute tilize)
CB_RM_GAMMA = 1  # row-major gamma sticks (reader -> compute tilize)
CB_RM_GAMMA_TILED = 2  # tilized gamma chunk (compute internal)
CB_RM_INPUT_RESIDENT = 24  # tilized resident input block (compute internal)
CB_RM_OUT = 16  # row-major output sticks (compute untilize -> writer)
CB_RM_SQUARED = 25
CB_RM_PARTIAL_SUMSQ = 26
CB_RM_RECIP_RMS = 27
CB_RM_NORMALIZED = 28
CB_RM_OUT_TILED = 29
# Refinement 4: ROW_MAJOR routed through the cross-core mcast all-gather (Regime B).
# The RM legs reuse the same SenderPipe/ReceiverPipe machinery as TILE Regime B;
# these two CBs are the RM-map analogues of CB_PARTIALS_GATHERED / CB_LOCAL_SUMSQ
# (distinct indices so they don't collide with the RM intermediates above).
CB_RM_PARTIALS_GATHERED = 9
CB_RM_LOCAL_SUMSQ = 10


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _resolve_compute_config(compute_kernel_config):
    if compute_kernel_config is not None:
        return compute_kernel_config
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


def _dest_limit(cfg) -> int:
    fp32 = bool(getattr(cfg, "fp32_dest_acc_en", True))
    full_sync = bool(getattr(cfg, "dst_full_sync_en", False))
    if full_sync:
        return 8 if fp32 else 16
    return 4 if fp32 else 8


def _intermediate_dtype(input_dtype, fp32_acc):
    """CB format for the accumulator/phase-boundary intermediates (Σx², the
    reduce scaler, recip-rms, the pass-2 normalized block, and the Regime-B
    gathered partials).

    Numeric-formats rule (skill §4): when the running Σx² crosses the CB,
    promote it to Float32 if fp32_dest_acc_en so the fp32 dest accumulation is
    not truncated at the pack boundary. A block-float (bf8b) input must never
    keep a bf8b accumulator — promote it. bf16 input keeps its bf16
    intermediates (byte-identical to Phase 0 / Refinement 1, no regression):
    bf16 ⊂ TF32 so the dest math is already exact and the baseline passes.
    """
    if input_dtype == ttnn.bfloat16:
        return ttnn.bfloat16
    # float32 or bfloat8_b input.
    return ttnn.float32 if fp32_acc else ttnn.bfloat16


def _tile_bytes(dtype) -> int:
    """Bytes for a standard 32x32 tile of `dtype` (Float32=4096, bf16=2048,
    bf8b=1088). Used for per-CB sizing now that input / intermediate / gamma /
    output CBs can each carry a different format."""
    return ttnn.tile_size(dtype)


# --- Refinement 6: byte-aware Regime-A resident footprint (the OOM floor) -------
# The OOM floor decides whether a full row can stay resident on ONE core (Regime A)
# or must be W-split across cores (Regime B). The original check counted only
# `input + gamma` tiles against RESIDENT_BUDGET_TILES (560, a bf16-tuned *tile
# count*). Two things make that unsound:
#   1. dtype: a tile is 2048 B (bf16) / 4096 B (fp32) / 1088 B (bf8b) — a tile
#      count cannot bound bytes across dtypes (fp32 doubles the footprint).
#   2. Refinement 5 grew `cb_squared` from one reduce_block to the *whole* shard
#      (Wt tiles), so it is now a third resident CB the count omitted.
# Pre-R6 this was masked because wide rows always took Regime B (it "added cores");
# Refinement 6's crossover deliberately keeps moderate rows in Regime A, which
# surfaced an fp32+gamma OOM (input+squared+gamma all resident in 4096 B tiles).
#
# Real Regime-A peak resident = cb_input_resident (Wt, input dtype)
#                             + cb_squared        (Wt, intermediate dtype)
#                             + cb_gamma          (Wt, gamma dtype, if present).
# The remaining CBs (output, recip, partial, normalized, scaler, RM streaming
# double-buffers) are small constants (≈ tens of KB), covered by the headroom
# between this budget and the 1.5 MB L1.
L1_RESIDENT_BUDGET_BYTES = 1_340_000


def _regime_a_resident_bytes(Wt_resident, input_dtype, inter_dtype, gamma_dtype, has_gamma) -> int:
    b = Wt_resident * _tile_bytes(input_dtype) + Wt_resident * _tile_bytes(inter_dtype)
    if has_gamma:
        b += Wt_resident * _tile_bytes(gamma_dtype)
    return b


def _row_fits_l1(Wt_resident, input_dtype, fp32_acc, gamma, has_gamma) -> bool:
    """True iff a full row (input + squared + gamma) stays resident on one core."""
    inter = _intermediate_dtype(input_dtype, fp32_acc)
    gamma_dt = gamma.dtype if has_gamma else input_dtype
    return _regime_a_resident_bytes(Wt_resident, input_dtype, inter, gamma_dt, has_gamma) <= L1_RESIDENT_BUDGET_BYTES


# --- Refinement 7: row-blocking (BLOCK_HEIGHT > 1) -----------------------------
# The "forgotten knob" from op_design.md:229-232 / Refinement 6: process `bh`
# tile-rows per work-unit in Regime A so the per-row PASS-1 reduce / FINALIZE
# helper init amortizes over a taller block (R6 measured a ~10% ceiling on this
# kernel — the per-row cost is data-movement/compute bound, not init bound, so
# row-blocking only recovers the init overhead).
#
# Hard design constraint (op_design.md:229-232): BLOCK_HEIGHT may grow ONLY AFTER
# every core already has work — it must NOT reduce active cores. So it applies only
# to the grid-saturated many-row case (Ht_total > total_cores, each core owns
# multiple rows); we group a core's owned rows into blocks of `bh`.
#
# Blast-radius bound (Refinement 7 scope): TILE Regime A no-gamma only. Every other
# path (gamma, ROW_MAJOR, Regime B) keeps bh==1 and is byte-identical to pre-R7.
#
# DISABLED BY DEFAULT (_ENABLE_ROW_BLOCKING = False). The knob is fully implemented and
# correct (golden 1683/1683 with it forced on), but measurement (R6 + R7) shows it is
# net-NEGATIVE on this kernel: rms_norm is memory-bound (~10us per tile-row of DRAM
# read+write at W=256), and row-blocking only amortizes per-row COMPUTE init — which is
# not the bottleneck. Measured device time with bh>1 is 0.82-0.98x of bh=1 (i.e. SLOWER)
# across (1,1,4096..16384,256) and (1,1,8192,512); double-buffering cb_input_resident did
# not recover it. Enabling it would regress production, so it stays off until a kernel
# change shifts the balance (see the follow-up refinement). Tests force it on via
# `_ENABLE_ROW_BLOCKING = True` (or `_FORCE_BH`) to exercise the path.
_ENABLE_ROW_BLOCKING = False


#
# `bh` must divide each core's row count with no remainder (the kernel has no
# remainder loop), so we enable bh>1 only when Ht_total % total_cores == 0 (every
# core gets exactly Ht_total//total_cores rows) and pick bh as a divisor of that
# per-core count. It is further bounded by:
#   - DEST capacity (_dest_limit, halved when fp32_dest_acc_en): the FINALIZE chain
#     and any DEST batching stay within the register file.
#   - L1: cb_input_resident and cb_squared both grow to bh*Wt tiles.
def _regime_a_block_height(Ht_total, num_cores, has_gamma, gamma_is_rm, Wt, input_dtype, fp32_acc, cfg):
    """Row-blocking factor for TILE Regime A no-gamma. Returns 1 (no blocking) for
    every other case or when the grid is not yet saturated."""
    if has_gamma or gamma_is_rm:
        return 1
    if _FORCE_BH is not None:  # measurement-only override (perf tests)
        return _FORCE_BH
    if not _ENABLE_ROW_BLOCKING:  # net-negative on this memory-bound kernel; off by default
        return 1
    if Ht_total <= num_cores:  # not many-row: each core owns <= 1 row group
        return 1
    if Ht_total % num_cores != 0:  # uneven split -> keep bh=1 (no remainder handling)
        return 1
    rows_per_core = Ht_total // num_cores
    dest = _dest_limit(cfg)
    inter = _intermediate_dtype(input_dtype, fp32_acc)
    in_b = _tile_bytes(input_dtype)
    inter_b = _tile_bytes(inter)
    bh = 1
    # Largest divisor of rows_per_core that fits both the DEST cap and the L1 budget.
    for cand in range(2, min(rows_per_core, dest) + 1):
        if rows_per_core % cand != 0:
            continue
        # Resident footprint when row-blocked: cb_input_resident (bh*Wt) + cb_squared (bh*Wt).
        resident = cand * Wt * in_b + cand * Wt * inter_b
        if resident <= L1_RESIDENT_BUDGET_BYTES:
            bh = cand
    return bh


# --- Refinement 8: row-blocking / coalesced-mcast for Regime B — NOT IMPLEMENTED ---
# R8 asked whether grouping `bh` tile-row-groups onto one K-core band and issuing ONE
# coalesced mcast of `bh` partials (vs `bh` separate K-round gathers) could amortize the
# all-gather fixed cost, while keeping active-core count unchanged. It cannot — closed as
# net-negative (mirrors R7's Regime-A row-blocking gate). Measured + structurally proven:
#   1. Keeping core count constant while grouping `bh` row-groups forces K up by factor
#      `bh` (fewer, wider bands). The all-gather cost grows with K, so this is strictly
#      net-negative. Measured on (1,1,32,8192) Wt=256 via _FORCE_K: K=16=34.0us,
#      K=32=50.5us, K=64=110.0us. bh>1 buys exactly this K increase; the per-core reduce
#      work it would save is already FLAT (bh*(Wt/K) = Wt/K_old, invariant).
#   2. There is NO serialized per-gather fixed cost to amortize: row-groups already run
#      in PARALLEL on disjoint K-core rectangles. Measured: 1 row-group K=16 = 34.0us vs
#      2 row-groups K=16 = 37.7us (ratio 1.11, not ~2x) — the per-gather handshake is
#      already hidden by spatial parallelism.
#   3. The only theoretical Regime-B win is a COVERAGE extension (oversubscribed grids:
#      shapes with num_row_groups*K_min > total_cores that today fall back to slow
#      Regime A, e.g. (512,8192)=234us). That is a *different* mechanism than coalesced
#      mcast, and the coalescing on top of it saves only O(bh) mcast setups (invisible
#      vs DRAM-bound compute). It also applies to NO shape in the golden/LOOSE suite —
#      every wide-W golden shape is 1-2 tile-rows (num_row_groups <= 2), so each already
#      gets an optimal one-row-group-per-core Regime B partition with nothing to block.
# Net: Regime B keeps bh==1 (compute_rt=[1]); the mcast all-gather is UNCHANGED. R9 (the
# combine/transport refinement) therefore inherits the plain per-row-group all-gather —
# there is no coalesced `bh*K` payload to preserve.
# See tests/.../test_rms_norm_perf.py::test_rms_norm_regime_b_rowblocking_exhausted and
# changelog.md (Refinement 8) for the full evidence.


def _even_split(total, num_cores):
    """Return a list of (start, count) contiguous chunks summing to `total`."""
    base = total // num_cores
    rem = total % num_cores
    out = []
    start = 0
    for c in range(num_cores):
        count = base + (1 if c < rem else 0)
        out.append((start, count))
        start += count
    return out


def create_program_descriptor(input_tensor, output_tensor, gamma, epsilon, compute_kernel_config):
    has_gamma = gamma is not None

    cfg = _resolve_compute_config(compute_kernel_config)

    # Refinement 4: ROW_MAJOR input is tilize-wrapped (math stays on tiles) and now
    # routes through the SAME A/B heuristic as TILE — a wide-W RM row that does not
    # fit one core's L1 (or where a W-split adds cores) goes through the cross-core
    # mcast all-gather (Regime B RM), exactly like TILE Regime B. Otherwise it stays
    # row-parallel (Regime A RM).
    if input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT:
        device = input_tensor.device()
        grid = device.compute_with_storage_grid_size()
        total_cores = grid.x * grid.y
        shape = list(input_tensor.shape)
        W = int(shape[-1])
        total_sticks = 1
        for d in shape[:-1]:
            total_sticks *= int(d)
        Wt = (W + 31) // 32
        num_blocks_total = (total_sticks + 31) // 32
        reduce_block = min(Wt, _dest_limit(cfg))
        num_chunks = (Wt + reduce_block - 1) // reduce_block
        Wt_padded = num_chunks * reduce_block
        fp32_acc = bool(getattr(cfg, "fp32_dest_acc_en", True))
        # Byte-aware OOM floor (Refinement 6): RM Regime A holds the whole tilized
        # row (Wt_padded) resident, so its footprint is identical in shape to TILE.
        row_fits = _row_fits_l1(Wt_padded, input_tensor.dtype, fp32_acc, gamma, has_gamma)

        _gamma_dt = gamma.dtype if has_gamma else input_tensor.dtype
        K = _select_k(Wt, num_blocks_total, grid, total_cores, has_gamma, input_tensor.dtype, fp32_acc, _gamma_dt)
        regime_a_cores = min(num_blocks_total, total_cores)
        adds_cores = K is not None and num_blocks_total * K > regime_a_cores

        def _rm_a():
            return _regime_rm_descriptor(input_tensor, output_tensor, gamma, has_gamma, cfg, epsilon)

        def _rm_b():
            return _regime_rm_b_descriptor(input_tensor, output_tensor, gamma, has_gamma, cfg, epsilon, grid)

        # Refinement 6: same force hook + two-tier (OOM floor / Wt crossover) policy
        # as the TILE path. Wt here is the full-row tile count (RM Regime A holds the
        # whole tilized row resident, so the OOM floor is identical).
        if _FORCE_REGIME == "A" and row_fits:
            return _rm_a()
        if _FORCE_REGIME == "B" and adds_cores:
            return _rm_b()
        if num_blocks_total >= total_cores and row_fits:
            return _rm_a()
        if not row_fits:
            if adds_cores:
                return _rm_b()
            return _rm_a()  # bounded-streaming fallback (matches prior behavior)
        if adds_cores and Wt >= _b_min_wt(layout_is_rm=True, fp32_acc=fp32_acc):
            return _rm_b()
        return _rm_a()

    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    total_cores = grid.x * grid.y

    padded = input_tensor.padded_shape
    Wt = padded[-1] // 32
    total_tiles = input_tensor.buffer_num_pages()
    Ht_total = total_tiles // Wt

    W = int(input_tensor.shape[-1])
    inv_W_bits = _f32_bits(1.0 / W)
    eps_bits = _f32_bits(epsilon)

    # Byte-aware OOM floor (Refinement 6): counts input + squared + gamma in their
    # actual tile-byte formats, so fp32 (4096 B/tile) and the R5-enlarged cb_squared
    # are no longer undercounted (was a bf16-only tile count of input + gamma).
    fp32_acc = bool(getattr(cfg, "fp32_dest_acc_en", True))
    row_fits = _row_fits_l1(Wt, input_tensor.dtype, fp32_acc, gamma, has_gamma)

    def _make_a():
        return _regime_a_descriptor(
            input_tensor, output_tensor, gamma, has_gamma, cfg, Wt, Ht_total, total_cores, inv_W_bits, eps_bits
        )

    def _make_b():
        return _regime_b_descriptor(
            input_tensor, output_tensor, gamma, has_gamma, cfg, Wt, Ht_total, grid, total_cores, inv_W_bits, eps_bits
        )

    _gamma_dt = gamma.dtype if has_gamma else input_tensor.dtype
    K = _select_k(Wt, Ht_total, grid, total_cores, has_gamma, input_tensor.dtype, fp32_acc, _gamma_dt)
    regime_a_cores = min(Ht_total, total_cores)
    adds_cores = K is not None and Ht_total * K > regime_a_cores

    # Refinement 6: measurement-only force hook (perf tests). Falls back to the
    # heuristic when the forced regime is infeasible for this shape.
    if _FORCE_REGIME == "A" and row_fits:
        return _make_a()
    if _FORCE_REGIME == "B" and adds_cores:
        return _make_b()

    # Regime A whenever it already saturates the grid and a full row fits L1 — no
    # cross-core split can add cores, so never pay the mcast cost.
    if Ht_total >= total_cores and row_fits:
        return _make_a()

    # Refinement 6 two-tier decision:
    #   - OOM floor (hard): a row that does NOT fit L1 *must* W-split (Regime B).
    #   - Perf crossover: a row that fits L1 goes B only if the split adds cores AND
    #     Wt >= REGIME_B_MIN_WT (below the crossover, single/few-core A beats B's
    #     mcast setup cost). Replaces the old "B whenever it adds any cores."
    if not row_fits:
        if adds_cores:
            return _make_b()
        raise NotImplementedError(
            f"rms_norm: row (Wt={Wt}, gamma={has_gamma}) exceeds L1 budget and no rectangular "
            f"Regime B partition exists for row_groups={Ht_total}, grid=({grid.x},{grid.y})."
        )
    if adds_cores and Wt >= _b_min_wt(layout_is_rm=False, fp32_acc=fp32_acc):
        return _make_b()
    return _make_a()


def _regime_a_descriptor(
    input_tensor,
    output_tensor,
    gamma,
    has_gamma,
    cfg,
    Wt,
    Ht_total,
    total_cores,
    inv_W_bits,
    eps_bits,
):
    num_cores = min(Ht_total, total_cores)
    reduce_block = min(Wt, _dest_limit(cfg))

    core_ranges = ttnn.num_cores_to_corerangeset(num_cores, ttnn.CoreCoord(8, 8), row_wise=True)
    cores = ttnn.corerange_to_cores(core_ranges, num_cores, row_wise=True)
    splits = _even_split(Ht_total, num_cores)

    # ---------- circular buffers (per-CB format) ----------
    # Input / output CBs follow the tensor dtype; accumulator intermediates
    # (Σx², scaler, recip-rms, normalized block) follow _intermediate_dtype;
    # gamma follows its own dtype (mixed precision). Each format gets its own
    # tile byte size.
    def cb(index, dtype, num_pages):
        pb = _tile_bytes(dtype)
        return ttnn.CBDescriptor(
            total_size=num_pages * pb,
            core_ranges=core_ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=pb)],
        )

    dt = input_tensor.dtype
    fp32_acc = bool(getattr(cfg, "fp32_dest_acc_en", True))
    inter = _intermediate_dtype(dt, fp32_acc)
    gamma_dt = gamma.dtype if has_gamma else dt
    # Refinement 3: TILE input may pair with ROW_MAJOR gamma. When so, the reader
    # stages gamma sticks in CB_GAMMA_RM and compute tilizes them into CB_GAMMA,
    # which must be padded to a whole number of reduce_block chunks.
    gamma_is_rm = 1 if (has_gamma and gamma.layout == ttnn.ROW_MAJOR_LAYOUT) else 0
    num_chunks = (Wt + reduce_block - 1) // reduce_block
    W = int(input_tensor.shape[-1])
    # gamma_elem is only consumed by the reader's ROW_MAJOR-gamma path, where gamma
    # is guaranteed non-bf8b ({bf8b, ROW_MAJOR} is INVALID). Never call
    # element_size() on a block format (bf8b) — it raises "datum invalid".
    gamma_elem = gamma.element_size() if gamma_is_rm else 0
    cb_gamma_pages = (num_chunks * reduce_block) if gamma_is_rm else Wt

    # Refinement 7: row-blocking factor (TILE Regime A no-gamma, grid-saturated). 1 in
    # every other case -> byte-identical to pre-R7. When bh>1 the compute processes bh
    # rows per block, so the resident input + squared CBs hold bh*Wt tiles and the
    # per-row partial-Σx² / recip CBs hold bh tiles (one per row).
    bh = _regime_a_block_height(Ht_total, num_cores, has_gamma, gamma_is_rm, Wt, dt, fp32_acc, cfg)
    sumsq_pages = max(2, bh)
    # Double-buffer the resident input when row-blocked so the reader prefetches the
    # next bh-row block during compute (PASS-1 holds the whole block) [static-analyzer F3].
    input_resident_pages = (2 * bh * Wt) if bh > 1 else Wt
    cbs = [
        cb(CB_INPUT_RESIDENT, dt, input_resident_pages),
        cb(CB_SCALER, inter, 1),
        cb(CB_OUTPUT, dt, 2),
        # Refinement 5: PASS-1 is now a single square + single reduce over the whole
        # resident shard, so cb_squared holds the full shard (Wt tiles) rather than one
        # reduce_block chunk. Wt is host-bounded by the same A/B heuristic that bounds
        # cb_input_resident (≤32 tiles for every routed shape), so this stays well within L1.
        # Refinement 7: bh*Wt when row-blocked (one squared block per bh-row work-unit).
        cb(CB_SQUARED, inter, bh * Wt),
        cb(CB_PARTIAL_SUMSQ, inter, sumsq_pages),
        cb(CB_RECIP_RMS, inter, sumsq_pages),
    ]
    if has_gamma:
        cbs.append(cb(CB_GAMMA, gamma_dt, cb_gamma_pages))
        # cb_normalized is the pass-2 Col->Row streaming intermediate, sized to one
        # REDUCE_BLOCK (constant) — NOT Wt — so the resident L1 footprint does not scale
        # with the row width (compute streams pass-2 per block).
        cbs.append(cb(CB_NORMALIZED, inter, reduce_block))
        if gamma_is_rm:
            cbs.append(cb(CB_GAMMA_RM, gamma_dt, 2 * reduce_block))

    # ---------- reader (unified; layout_is_rm=0, num_partials=1) ----------
    reader_ct = [
        CB_INPUT_RESIDENT,
        CB_GAMMA,
        CB_SCALER,
        CB_PARTIAL_SUMSQ,  # cb_local_sumsq (unused; num_partials==1)
        CB_PARTIALS_GATHERED,  # (unused; num_partials==1)
        Wt,
        Wt,  # Wt_gamma_resident (== Wt for TILE; no synthetic gamma padding)
        int(has_gamma),
        1,  # num_partials = 1 (Regime A: no gather)
        0,  # data_ready_sem_id (unused)
        0,  # consumed_sem_id (unused)
        gamma_is_rm,
        CB_GAMMA_RM,
        reduce_block,
        num_chunks,
        W,
        0,  # in_elem (TILE: unused)
        gamma_elem,
        0,  # layout_is_rm = 0 (TILE)
        CB_INPUT_RESIDENT,  # cb_rm_in (unused)
        TRANSPORT_MCAST_ALLGATHER,  # transport_mode (unused; num_partials==1)
        0,  # produced_sem_id (unused; num_partials==1)
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    gamma_addr = gamma.buffer_address() if has_gamma else 0

    for core, (start, count) in zip(cores, splits):
        # reader: input_addr, gamma_addr, input_page_base, gamma_page_base, start_unit,
        #         num_units, total_sticks
        reader_rt[core.x][core.y] = [
            input_tensor.buffer_address(),
            gamma_addr,
            start * Wt,  # input_page_base of first owned row
            0,  # gamma_page_base (full gamma, no shard)
            start,  # start_unit (unused for TILE)
            count,  # num_units = owned rows
            0,  # total_sticks (RM only)
        ]
        writer_rt[core.x][core.y] = [output_tensor.buffer_address(), start * Wt, count * Wt, 0]
        compute_rt[core.x][core.y] = [count]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---------- writer (unified; layout_is_rm=0) ----------
    writer_ct = [CB_OUTPUT, 0, 0, 0, 0, 0, 0]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ---------- compute ----------
    compute_ct = [
        CB_INPUT_RESIDENT,
        CB_GAMMA,
        CB_SCALER,
        CB_PARTIALS_GATHERED,
        CB_OUTPUT,
        CB_SQUARED,
        CB_PARTIAL_SUMSQ,
        CB_RECIP_RMS,
        CB_NORMALIZED,
        Wt,
        reduce_block,
        int(has_gamma),
        inv_W_bits,
        eps_bits,
        1,  # num_partials = 1
        CB_PARTIAL_SUMSQ,  # cb_local_sumsq (unused in Regime A; num_partials==1 elides it)
        gamma_is_rm,
        CB_GAMMA_RM,
        0,  # layout_is_rm = 0 (TILE input)
        CB_INPUT_RESIDENT,  # cb_rm_in (unused for TILE)
        CB_OUTPUT,  # cb_rm_out (unused for TILE)
        bh,  # Refinement 7: BLOCK_HEIGHT (rows per work-unit; 1 = no row-blocking)
    ]
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=core_ranges,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=cfg,
    )

    program = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
    io = [input_tensor, gamma, output_tensor] if has_gamma else [input_tensor, output_tensor]
    return program, io


def _select_k(Wt, num_row_groups, grid, total_cores, has_gamma, input_dtype, fp32_acc, gamma_dtype):
    """W-split factor K forming full-width rectangular bands: K divides Wt, K is a
    multiple of grid.x (full-width bands), num_row_groups*K fits the grid, and the
    per-core shard (Wt/K) fits the L1 resident byte budget. Returns None if none
    qualifies.

    Refinement 6 (K tuning, measured — this is NOT 'maximize K'): the all-gather cost
    grows ~O(K) (each of the K cores mcasts its partial and the combine sums K tiles),
    while the per-core reduce shrinks ~O(Wt/K). Total device time ≈ Wt/K + c·K, so
    "maximize K" is the WORST choice — measured on (1,1,32,8192) Wt=256, Regime B is
    K=16→33.9us vs K=64→109.8us (3.2x), and (1,1,32,16384) Wt=512 is K=16→48us vs
    K=64→116us. We therefore pick the K that MINIMIZES the cost proxy (Wt//K + K)
    among the qualifying candidates (tie-break: smaller K = less gather). This lands
    K≈16 for Wt=256/512 (matching the measured optima) and adapts up (~√Wt) for wider
    rows. The OOM floor still constrains the minimum K from below (a shard must fit
    L1), and num_row_groups*K<=grid bounds it from above. _FORCE_K (perf tests only)
    overrides the choice to sweep K."""
    gx = grid.x
    inter = _intermediate_dtype(input_dtype, fp32_acc)

    def _qualifies(K):
        if K % gx != 0 or Wt % K != 0 or num_row_groups * K > total_cores:
            return False
        Wt_s = Wt // K
        return _regime_a_resident_bytes(Wt_s, input_dtype, inter, gamma_dtype, has_gamma) <= L1_RESIDENT_BUDGET_BYTES

    if _FORCE_K is not None:
        return _FORCE_K if _qualifies(_FORCE_K) else None

    best = None
    best_cost = None
    for K in range(gx, total_cores + 1, gx):
        if not _qualifies(K):
            continue
        cost = (Wt // K) + K  # ≈ per-core reduce work + all-gather cost
        # Strictly-less keeps the SMALLEST K on ties (loop ascends), favouring less
        # mcast/gather overhead at equal proxy cost.
        if best_cost is None or cost < best_cost:
            best, best_cost = K, cost
    return best


def _regime_b_descriptor(
    input_tensor, output_tensor, gamma, has_gamma, cfg, Wt, Ht_total, grid, total_cores, inv_W_bits, eps_bits
):
    num_row_groups = Ht_total  # Phase 0: bh = 1 tile-row per group
    gx = grid.x
    _fp32_acc = bool(getattr(cfg, "fp32_dest_acc_en", True))
    _gamma_dt = gamma.dtype if has_gamma else input_tensor.dtype
    K = _select_k(Wt, num_row_groups, grid, total_cores, has_gamma, input_tensor.dtype, _fp32_acc, _gamma_dt)
    if K is None:
        raise NotImplementedError(
            f"rms_norm: no rectangular Regime B partition for Wt={Wt}, "
            f"row_groups={num_row_groups}, grid=({grid.x},{grid.y})"
        )
    transport_mode = _select_transport(K)  # Refinement 9 (Part A)

    gh = K // gx  # band height (rows) per group
    Wt_s = Wt // K
    reduce_block = min(Wt_s, _dest_limit(cfg))
    dt = input_tensor.dtype
    fp32_acc = bool(getattr(cfg, "fp32_dest_acc_en", True))
    inter = _intermediate_dtype(dt, fp32_acc)
    gamma_dt = gamma.dtype if has_gamma else dt
    device = input_tensor.device()
    # Refinement 3: TILE input + ROW_MAJOR gamma. Reader stages gamma sticks in
    # CB_GAMMA_RM; compute tilizes them into CB_GAMMA (padded to whole chunks).
    gamma_is_rm = 1 if (has_gamma and gamma.layout == ttnn.ROW_MAJOR_LAYOUT) else 0
    num_chunks = (Wt_s + reduce_block - 1) // reduce_block
    W = int(input_tensor.shape[-1])
    # See Regime A note: never call element_size() on bf8b. gamma_elem is only used
    # by the reader's ROW_MAJOR-gamma path (gamma guaranteed non-bf8b there).
    gamma_elem = gamma.element_size() if gamma_is_rm else 0
    cb_gamma_pages = (num_chunks * reduce_block) if gamma_is_rm else Wt_s

    DATA_READY = 0
    CONSUMED = 1

    # All cores used (num_row_groups bands of K cores each).
    used_cores = num_row_groups * K
    core_ranges = ttnn.num_cores_to_corerangeset(used_cores, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)

    # Per-CB format: input/output follow the tensor dtype; the mcast-gathered
    # partials (cb_local_sumsq → cb_partials_gathered) and the accumulator
    # intermediates follow _intermediate_dtype, so the cross-core all-gather
    # transfers a matched tile-byte count (both endpoints share the format).
    def cb(index, dtype, num_pages):
        pb = _tile_bytes(dtype)
        return ttnn.CBDescriptor(
            total_size=num_pages * pb,
            core_ranges=core_ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=pb)],
        )

    cbs = [
        cb(CB_INPUT_RESIDENT, dt, Wt_s),
        cb(CB_SCALER, inter, 1),
        cb(CB_OUTPUT, dt, 2),
        # Refinement 5: single square + single reduce over the whole resident shard;
        # cb_squared holds the full per-core shard (Wt_s tiles), not one reduce_block.
        # Wt_s is bounded by _select_k (the W-split keeps it small), so L1 is unaffected.
        cb(CB_SQUARED, inter, Wt_s),
        cb(CB_PARTIAL_SUMSQ, inter, 2),
        cb(CB_RECIP_RMS, inter, 2),
        cb(CB_PARTIALS_GATHERED, inter, K),
        cb(CB_LOCAL_SUMSQ, inter, 2),
    ]
    if has_gamma:
        cbs.append(cb(CB_GAMMA, gamma_dt, cb_gamma_pages))
        # Constant-sized pass-2 streaming intermediate (one REDUCE_BLOCK), not Wt_s.
        cbs.append(cb(CB_NORMALIZED, inter, reduce_block))
        if gamma_is_rm:
            cbs.append(cb(CB_GAMMA_RM, gamma_dt, 2 * reduce_block))

    # Semaphores on the full union of used cores (disjoint groups reuse the same IDs).
    semaphores = [
        ttnn.SemaphoreDescriptor(id=DATA_READY, core_ranges=core_ranges, initial_value=0),
        ttnn.SemaphoreDescriptor(id=CONSUMED, core_ranges=core_ranges, initial_value=0),
        # Refinement 9 (Part A): peers->root "produced" counter for the mode-1 root-relay
        # gather (host-init 0). Unused when transport_mode==0 but always allocated so the
        # union of group cores carries it; cheap (one L1 word).
        ttnn.SemaphoreDescriptor(id=PRODUCED_SEM, core_ranges=core_ranges, initial_value=0),
    ]

    def vcoord(lx, ly):
        v = device.worker_core_from_logical_core(ttnn.CoreCoord(lx, ly))
        return v.x, v.y

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    gamma_addr = gamma.buffer_address() if has_gamma else 0

    for g in range(num_row_groups):
        # Group rectangle (logical): full grid width, rows [g*gh, g*gh+gh)
        rx0, ry0 = 0, g * gh
        rx1, ry1 = gx - 1, g * gh + gh - 1
        vrx0, vry0 = vcoord(rx0, ry0)
        vrx1, vry1 = vcoord(rx1, ry1)
        # Sender virtual coords for each rank j within this group.
        sender_coords = []
        for j in range(K):
            jx, jy = j % gx, g * gh + j // gx
            vx, vy = vcoord(jx, jy)
            sender_coords.extend([vx, vy])

        for r in range(K):
            lx, ly = r % gx, g * gh + r // gx
            input_page_base = g * Wt + r * Wt_s
            gamma_page_base = r * Wt_s

            # unified reader RT: input_addr, gamma_addr, input_page_base, gamma_page_base,
            # start_unit(0), num_units(1), total_sticks(0), my_rank, rect(4), sender_coords
            reader_rt[lx][ly] = [
                input_tensor.buffer_address(),
                gamma_addr,
                input_page_base,
                gamma_page_base,
                0,  # start_unit (unused for TILE)
                1,  # num_units = 1 row-group per core
                0,  # total_sticks (RM only)
                r,  # my_rank
                vrx0,
                vry0,
                vrx1,
                vry1,
            ] + sender_coords
            writer_rt[lx][ly] = [output_tensor.buffer_address(), input_page_base, Wt_s, 0]
            compute_rt[lx][ly] = [1]  # one tile-row group per core

    # ---------- reader (unified; layout_is_rm=0, num_partials=K -> mcast all-gather) ----------
    reader_ct = [
        CB_INPUT_RESIDENT,
        CB_GAMMA,
        CB_SCALER,
        CB_LOCAL_SUMSQ,
        CB_PARTIALS_GATHERED,
        Wt_s,
        Wt_s,  # Wt_gamma_resident (== Wt_s for TILE)
        int(has_gamma),
        K,  # num_partials
        DATA_READY,
        CONSUMED,
        gamma_is_rm,
        CB_GAMMA_RM,
        reduce_block,
        num_chunks,
        W,
        0,  # in_elem (TILE: unused)
        gamma_elem,
        0,  # layout_is_rm = 0 (TILE)
        CB_INPUT_RESIDENT,  # cb_rm_in (unused)
        transport_mode,  # Refinement 9 (Part A): all-reduce transport selector
        PRODUCED_SEM,  # produced_sem_id (mode 1 gather counter)
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---------- writer (unified; layout_is_rm=0) ----------
    writer_ct = [CB_OUTPUT, 0, 0, 0, 0, 0, 0]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ---------- compute ----------
    compute_ct = [
        CB_INPUT_RESIDENT,
        CB_GAMMA,
        CB_SCALER,
        CB_PARTIALS_GATHERED,
        CB_OUTPUT,
        CB_SQUARED,
        CB_PARTIAL_SUMSQ,
        CB_RECIP_RMS,
        CB_NORMALIZED,
        Wt_s,
        reduce_block,
        int(has_gamma),
        inv_W_bits,
        eps_bits,
        K,
        CB_LOCAL_SUMSQ,
        gamma_is_rm,
        CB_GAMMA_RM,
        0,  # layout_is_rm = 0 (TILE input)
        CB_INPUT_RESIDENT,  # cb_rm_in (unused for TILE)
        CB_OUTPUT,  # cb_rm_out (unused for TILE)
        1,  # Refinement 7: BLOCK_HEIGHT = 1 (Regime B is not row-blocked)
    ]
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=core_ranges,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=cfg,
    )

    program = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=semaphores,
        cbs=cbs,
    )
    io = [input_tensor, gamma, output_tensor] if has_gamma else [input_tensor, output_tensor]
    return program, io


def _regime_rm_descriptor(input_tensor, output_tensor, gamma, has_gamma, cfg, epsilon):
    """ROW_MAJOR (tilize-wrapped) row-parallel descriptor.

    Each stick (one (b,c,h) row of W elements) is RMS-normalized independently.
    Sticks are processed in 32-stick tile-blocks; the W axis is chunked by
    `reduce_block` tiles so the per-core L1 footprint is bounded regardless of W.
    Non-tile-aligned W (zero-padded columns) and H (partial last block) are
    handled natively in the dataflow kernels — no host-side pad/slice/to_layout.
    """
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    total_cores = grid.x * grid.y

    shape = list(input_tensor.shape)
    W = int(shape[-1])
    total_sticks = 1
    for d in shape[:-1]:
        total_sticks *= int(d)

    Wt = (W + 31) // 32
    num_blocks_total = (total_sticks + 31) // 32  # 32 sticks per tile-block

    reduce_block = min(Wt, _dest_limit(cfg))
    num_chunks = (Wt + reduce_block - 1) // reduce_block
    Wt_padded = num_chunks * reduce_block

    inv_W_bits = _f32_bits(1.0 / W)
    eps_bits = _f32_bits(epsilon)

    dt = input_tensor.dtype
    fp32_acc = bool(getattr(cfg, "fp32_dest_acc_en", True))
    inter = _intermediate_dtype(dt, fp32_acc)
    gamma_dt = gamma.dtype if has_gamma else dt
    in_elem = input_tensor.element_size()  # RM input is never bf8b ({bf8b, RM} INVALID)
    out_elem = output_tensor.element_size()
    gamma_is_tile = 1 if (has_gamma and gamma.layout == ttnn.TILE_LAYOUT) else 0
    gamma_is_rm = 1 if (has_gamma and not gamma_is_tile) else 0
    # gamma_elem only feeds the reader's ROW_MAJOR-gamma path (gamma non-bf8b there;
    # bf8b gamma is always TILE). Avoid element_size() on a bf8b TILE gamma.
    gamma_elem = gamma.element_size() if (has_gamma and not gamma_is_tile) else in_elem

    num_cores = min(num_blocks_total, total_cores)
    core_ranges = ttnn.num_cores_to_corerangeset(num_cores, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    cores = ttnn.corerange_to_cores(core_ranges, num_cores, row_wise=True)
    splits = _even_split(num_blocks_total, num_cores)

    db = 2  # double-buffer for streamed chunk CBs

    def cb(index, dtype, num_pages):
        pb = _tile_bytes(dtype)
        return ttnn.CBDescriptor(
            total_size=num_pages * pb,
            core_ranges=core_ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=pb)],
        )

    cbs = [
        cb(CB_RM_IN, dt, db * reduce_block),
        cb(CB_SCALER, inter, 1),
        cb(CB_RM_OUT, dt, db * reduce_block),
        cb(CB_RM_INPUT_RESIDENT, dt, Wt_padded),
        # Refinement 5: single square + single reduce over the whole tilized resident
        # shard; cb_squared holds the full padded shard (Wt_padded tiles), not one chunk.
        cb(CB_RM_SQUARED, inter, Wt_padded),
        cb(CB_RM_PARTIAL_SUMSQ, inter, 2),
        cb(CB_RM_RECIP_RMS, inter, 2),
        cb(CB_RM_OUT_TILED, dt, db * reduce_block),
    ]
    if has_gamma:
        # CB_RM_GAMMA: ROW_MAJOR gamma stick staging (double-buffered chunks; only
        # fed when gamma is ROW_MAJOR). CB_RM_GAMMA_TILED: resident tiled gamma
        # (Wt_padded tiles), the unified compute's resident-gamma model — read once,
        # held across all blocks, indexed by per-chunk TileOffset in PASS-2.
        cbs.append(cb(CB_RM_GAMMA, gamma_dt, db * reduce_block))
        cbs.append(cb(CB_RM_GAMMA_TILED, gamma_dt, Wt_padded))
        cbs.append(cb(CB_RM_NORMALIZED, inter, reduce_block))

    # ---------- reader (unified; layout_is_rm=1, num_partials=1) ----------
    reader_ct = [
        CB_RM_INPUT_RESIDENT,  # cb_input_resident (unused by RM reader)
        CB_RM_GAMMA_TILED,  # cb_gamma (resident tiled gamma)
        CB_SCALER,
        CB_RM_PARTIAL_SUMSQ,  # cb_local_sumsq (unused; num_partials==1)
        CB_PARTIALS_GATHERED,  # (unused; num_partials==1)
        Wt,  # real tiles along W (ceil(W/32))
        Wt_padded,  # Wt_gamma_resident (padded so each pass-2 chunk reads a full block)
        int(has_gamma),
        1,  # num_partials = 1 (row-parallel RM)
        0,  # data_ready_sem_id (unused)
        0,  # consumed_sem_id (unused)
        gamma_is_rm,
        CB_RM_GAMMA,  # cb_gamma_rm (stick staging)
        reduce_block,
        num_chunks,
        W,
        in_elem,
        gamma_elem,
        1,  # layout_is_rm = 1 (ROW_MAJOR input)
        CB_RM_IN,  # cb_rm_in
        TRANSPORT_MCAST_ALLGATHER,  # transport_mode (unused; RM Regime A num_partials==1)
        0,  # produced_sem_id (unused; num_partials==1)
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    gamma_addr = gamma.buffer_address() if has_gamma else 0

    for core, (start, count) in zip(cores, splits):
        # reader: input_addr, gamma_addr, input_page_base(0), gamma_page_base(0),
        #         start_unit=start_block, num_units=count blocks, total_sticks
        reader_rt[core.x][core.y] = [
            input_tensor.buffer_address(),
            gamma_addr,
            0,  # input_page_base (RM uses start_unit)
            0,  # gamma_page_base (full gamma)
            start,  # start_unit = start_block
            count,  # num_units = owned blocks
            total_sticks,
        ]
        writer_rt[core.x][core.y] = [output_tensor.buffer_address(), start, count, total_sticks, 0]
        compute_rt[core.x][core.y] = [count]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---------- writer (unified; layout_is_rm=1) ----------
    writer_ct = [CB_RM_OUT, 1, Wt, reduce_block, num_chunks, W, out_elem]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ---------- compute (unified kernel; layout_is_rm = 1) ----------
    compute_ct = [
        CB_RM_INPUT_RESIDENT,  # cb_input_resident (tilize dest, resident)
        CB_RM_GAMMA_TILED,  # cb_gamma (resident tiled gamma)
        CB_SCALER,
        CB_PARTIALS_GATHERED,  # cb_partials_gathered (unused; num_partials==1)
        CB_RM_OUT_TILED,  # cb_pass2_out (untilize source)
        CB_RM_SQUARED,
        CB_RM_PARTIAL_SUMSQ,
        CB_RM_RECIP_RMS,
        CB_RM_NORMALIZED,
        Wt_padded,  # Wt (padded shard width -> every chunk is a full reduce_block)
        reduce_block,
        int(has_gamma),
        inv_W_bits,
        eps_bits,
        1,  # num_partials = 1 (row-parallel RM)
        CB_RM_PARTIAL_SUMSQ,  # cb_local_sumsq (unused; num_partials==1)
        gamma_is_rm,  # ROW_MAJOR gamma -> tilize once into cb_gamma
        CB_RM_GAMMA,  # cb_gamma_rm (stick staging)
        1,  # layout_is_rm = 1 (ROW_MAJOR input)
        CB_RM_IN,  # cb_rm_in (input stick source -> tilize)
        CB_RM_OUT,  # cb_rm_out (untilize dest)
        1,  # Refinement 7: BLOCK_HEIGHT = 1 (ROW_MAJOR is not row-blocked)
    ]
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=core_ranges,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=cfg,
    )

    program = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
    io = [input_tensor, gamma, output_tensor] if has_gamma else [input_tensor, output_tensor]
    return program, io


def _regime_rm_b_descriptor(input_tensor, output_tensor, gamma, has_gamma, cfg, epsilon, grid):
    """ROW_MAJOR routed through the cross-core mcast all-gather (Regime B RM).

    Structurally identical to _regime_b_descriptor (TILE Regime B): each core owns
    one 32-stick block-group x one W-column shard. The only difference is the
    data-access boundary — input/output are row-major sticks (the unified kernels'
    layout_is_rm=1 path tilizes the resident shard and untilizes the result). Each
    core reads its shard columns [shard_col0, shard_col0 + Wt_s*32) as sticks,
    computes the local partial Sum(x^2) over those columns, all-gathers the K
    shard partials, sums them to the GLOBAL Sum(x^2) (inv_W = 1/W over the full
    row), normalizes its shard, untilizes, and writes its shard columns back.
    """
    device = input_tensor.device()
    total_cores = grid.x * grid.y
    gx = grid.x

    shape = list(input_tensor.shape)
    W = int(shape[-1])
    total_sticks = 1
    for d in shape[:-1]:
        total_sticks *= int(d)
    Wt = (W + 31) // 32
    num_block_groups = (total_sticks + 31) // 32  # 32-stick blocks (analogue of Ht_total)

    _fp32_acc = bool(getattr(cfg, "fp32_dest_acc_en", True))
    _gamma_dt = gamma.dtype if has_gamma else input_tensor.dtype
    K = _select_k(Wt, num_block_groups, grid, total_cores, has_gamma, input_tensor.dtype, _fp32_acc, _gamma_dt)
    if K is None:
        raise NotImplementedError(
            f"rms_norm (RM): no rectangular Regime B partition for Wt={Wt}, "
            f"block_groups={num_block_groups}, grid=({grid.x},{grid.y})"
        )
    transport_mode = _select_transport(K)  # Refinement 9 (Part A)

    gh = K // gx  # band height (rows of cores) per block-group
    Wt_s = Wt // K
    reduce_block = min(Wt_s, _dest_limit(cfg))
    num_chunks = (Wt_s + reduce_block - 1) // reduce_block
    Wt_padded = num_chunks * reduce_block

    inv_W_bits = _f32_bits(1.0 / W)
    eps_bits = _f32_bits(epsilon)

    dt = input_tensor.dtype
    fp32_acc = bool(getattr(cfg, "fp32_dest_acc_en", True))
    inter = _intermediate_dtype(dt, fp32_acc)
    gamma_dt = gamma.dtype if has_gamma else dt
    in_elem = input_tensor.element_size()  # RM input is never bf8b ({bf8b, RM} INVALID)
    out_elem = output_tensor.element_size()
    gamma_is_tile = 1 if (has_gamma and gamma.layout == ttnn.TILE_LAYOUT) else 0
    gamma_is_rm = 1 if (has_gamma and not gamma_is_tile) else 0
    gamma_elem = gamma.element_size() if (has_gamma and not gamma_is_tile) else in_elem

    DATA_READY = 0
    CONSUMED = 1

    used_cores = num_block_groups * K
    core_ranges = ttnn.num_cores_to_corerangeset(used_cores, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)

    db = 2

    def cb(index, dtype, num_pages):
        pb = _tile_bytes(dtype)
        return ttnn.CBDescriptor(
            total_size=num_pages * pb,
            core_ranges=core_ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=pb)],
        )

    cbs = [
        cb(CB_RM_IN, dt, db * reduce_block),
        cb(CB_SCALER, inter, 1),
        cb(CB_RM_OUT, dt, db * reduce_block),
        cb(CB_RM_INPUT_RESIDENT, dt, Wt_padded),
        # Refinement 5: single square + single reduce over the whole tilized resident
        # shard; cb_squared holds the full padded shard (Wt_padded tiles), not one chunk.
        cb(CB_RM_SQUARED, inter, Wt_padded),
        cb(CB_RM_PARTIAL_SUMSQ, inter, 2),
        cb(CB_RM_RECIP_RMS, inter, 2),
        cb(CB_RM_OUT_TILED, dt, db * reduce_block),
        cb(CB_RM_PARTIALS_GATHERED, inter, K),
        cb(CB_RM_LOCAL_SUMSQ, inter, 2),
    ]
    if has_gamma:
        cbs.append(cb(CB_RM_GAMMA, gamma_dt, db * reduce_block))
        cbs.append(cb(CB_RM_GAMMA_TILED, gamma_dt, Wt_padded))
        cbs.append(cb(CB_RM_NORMALIZED, inter, reduce_block))

    semaphores = [
        ttnn.SemaphoreDescriptor(id=DATA_READY, core_ranges=core_ranges, initial_value=0),
        ttnn.SemaphoreDescriptor(id=CONSUMED, core_ranges=core_ranges, initial_value=0),
        # Refinement 9 (Part A): peers->root "produced" counter for the mode-1 root-relay
        # gather (host-init 0). Unused when transport_mode==0 but always allocated so the
        # union of group cores carries it; cheap (one L1 word).
        ttnn.SemaphoreDescriptor(id=PRODUCED_SEM, core_ranges=core_ranges, initial_value=0),
    ]

    def vcoord(lx, ly):
        v = device.worker_core_from_logical_core(ttnn.CoreCoord(lx, ly))
        return v.x, v.y

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    gamma_addr = gamma.buffer_address() if has_gamma else 0

    for g in range(num_block_groups):
        rx0, ry0 = 0, g * gh
        rx1, ry1 = gx - 1, g * gh + gh - 1
        vrx0, vry0 = vcoord(rx0, ry0)
        vrx1, vry1 = vcoord(rx1, ry1)
        sender_coords = []
        for j in range(K):
            jx, jy = j % gx, g * gh + j // gx
            vx, vy = vcoord(jx, jy)
            sender_coords.extend([vx, vy])

        for r in range(K):
            lx, ly = r % gx, g * gh + r // gx
            shard_col0 = r * Wt_s * 32  # this shard's first W-column
            gamma_page_base = r * Wt_s  # gamma shard (tile index / *32 for RM gamma)

            # unified reader RT: input_addr, gamma_addr, input_page_base(=shard_col0),
            # gamma_page_base, start_unit(=block-group g), num_units(1), total_sticks,
            # my_rank, rect(4), sender_coords
            reader_rt[lx][ly] = [
                input_tensor.buffer_address(),
                gamma_addr,
                shard_col0,
                gamma_page_base,
                g,  # start_unit = block-group index
                1,  # num_units = 1 block-group per core
                total_sticks,
                r,  # my_rank
                vrx0,
                vry0,
                vrx1,
                vry1,
            ] + sender_coords
            writer_rt[lx][ly] = [output_tensor.buffer_address(), g, 1, total_sticks, shard_col0]
            compute_rt[lx][ly] = [1]

    # ---------- reader (unified; layout_is_rm=1, num_partials=K) ----------
    reader_ct = [
        CB_RM_INPUT_RESIDENT,  # cb_input_resident (unused by RM reader)
        CB_RM_GAMMA_TILED,  # cb_gamma (resident tiled gamma)
        CB_SCALER,
        CB_RM_LOCAL_SUMSQ,
        CB_RM_PARTIALS_GATHERED,
        Wt_s,  # real shard tiles
        Wt_padded,  # Wt_gamma_resident
        int(has_gamma),
        K,  # num_partials
        DATA_READY,
        CONSUMED,
        gamma_is_rm,
        CB_RM_GAMMA,  # cb_gamma_rm (stick staging)
        reduce_block,
        num_chunks,
        W,
        in_elem,
        gamma_elem,
        1,  # layout_is_rm = 1
        CB_RM_IN,  # cb_rm_in
        transport_mode,  # Refinement 9 (Part A): all-reduce transport selector
        PRODUCED_SEM,  # produced_sem_id (mode 1 gather counter)
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---------- writer (unified; layout_is_rm=1) ----------
    writer_ct = [CB_RM_OUT, 1, Wt_s, reduce_block, num_chunks, W, out_elem]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ---------- compute (unified; layout_is_rm=1, num_partials=K) ----------
    compute_ct = [
        CB_RM_INPUT_RESIDENT,
        CB_RM_GAMMA_TILED,
        CB_SCALER,
        CB_RM_PARTIALS_GATHERED,
        CB_RM_OUT_TILED,  # cb_pass2_out (untilize source)
        CB_RM_SQUARED,
        CB_RM_PARTIAL_SUMSQ,
        CB_RM_RECIP_RMS,
        CB_RM_NORMALIZED,
        Wt_padded,  # Wt (padded shard width)
        reduce_block,
        int(has_gamma),
        inv_W_bits,  # global 1/W
        eps_bits,
        K,  # num_partials
        CB_RM_LOCAL_SUMSQ,
        gamma_is_rm,
        CB_RM_GAMMA,
        1,  # layout_is_rm = 1
        CB_RM_IN,
        CB_RM_OUT,
        1,  # Refinement 7: BLOCK_HEIGHT = 1 (Regime B RM is not row-blocked)
    ]
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=core_ranges,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=cfg,
    )

    program = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=semaphores,
        cbs=cbs,
    )
    io = [input_tensor, gamma, output_tensor] if has_gamma else [input_tensor, output_tensor]
    return program, io
