# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for rms_norm.

Row-parallel, bounded two-pass streaming reduce over W (op_design.md §1, §5-§9):

  Pass 1: x -> square -> accumulate_reduce(SUM, 1/W)  -> cb_rstd = mean(x^2)
          cb_rstd -> (+eps, rsqrt)                    -> cb_rstd = 1/RMS  (held)
  Pass 2: x -> mul<Col>(x, rstd)                       -> cb_norm
          cb_norm -> mul<Row>(norm, gamma) / copy      -> cb_out

Work distribution: the R = NC*ceil(H/32) independent tile-rows are split across
the whole compute grid via `split_work_to_cores(R, grid, row_wise=True)`; each
core owns a contiguous run [row_start, row_start+num_rows) and loops its rows,
each row a 2-pass streaming reduce over NUM_BLOCKS blocks of BLOCK_SIZE tiles.

Every block knob is a live parameter (single source of truth):
  BLOCK_SIZE = pick_block_size(Wt)   -> CT arg (reader, compute)
  NUM_BLOCKS = Wt // BLOCK_SIZE       -> derived
  DEPTH      = 2                      -> CB depth (num_pages = DEPTH*BLOCK_SIZE)
No CB is sized by an op dimension (Wt/W/H/R); cb_rstd is 1 tile per row.

RM regime uses the tilize/untilize dataflow helpers
(dataflow_kernel_lib::read_sticks_for_tilize / write_sticks_after_untilize),
which handle non-tile-aligned W (row_bytes) and H (partial last block) and the
per-core start-row offset natively — no host-side pad/slice.
"""

import os
import struct
from pathlib import Path

import ttnn

# ---- Perf 1 fast-path master switch (debug/A-B knob; default ON) ----
# The Perf-1 tournament compute levers (PASS1_FUSED, PASS2_RECONFIG_SKIP) are predicate-guarded
# fast paths. This env knob force-disables BOTH so a same-session baseline-vs-graduated A/B can
# be measured (RMS_PERF1_FASTPATH=0 -> the byte-identical R6f/R6g compute path everywhere). It
# does NOT change which cells are supported; it is a measurement/debug aid only. Default: ON.
_PERF1_FASTPATH = os.environ.get("RMS_PERF1_FASTPATH", "1") != "0"

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32
DEPTH = 2  # per-streaming-CB double-buffer depth (op_design.md §1)

# ---- Resident single-read fast-path knobs (op_design.md §1 lamp 1; Refinement 3) ----
# RESIDENT_X_DEPTH is the MAX resident input-CB depth (whole tile-rows) the host will
# use: at depth d the reader prefetches d-1 tile-rows ahead while compute reads the
# current one from L1. The host picks the largest depth in [1, RESIDENT_X_DEPTH] that
# fits L1_RESIDENT_BUDGET, so wide rows single-buffer and narrow rows double-buffer.
# L1_RESIDENT_BUDGET is the per-core CB budget the resident footprint must fit under —
# the explicit predicate that keeps the resident path from overflowing L1 (Blackhole
# L1 ~1.5 MB; a conservative 1.1 MB CB budget). Because the intermediates stay
# per-block, only cb_x_in + cb_gamma scale with Wt, so every prefill width fits. Both
# are live tunables with a single source of truth — no CB literal derives independently.
RESIDENT_X_DEPTH = 2
L1_RESIDENT_BUDGET = 1_100_000

# ---- Cross-core round-batching knob (op_requirements.md R6a lever 1) ----
# The cross-core stat round (gather -> master fold -> broadcast) is fully synchronous and
# costs a FLAT ~3150 ns per round regardless of per-core work (R6 ablation), so it dominates
# any shard with many tile-rows per group (BLOCK 8x8: HT_LOCAL=32 -> 32 rounds -> 5.76x above
# achievable). STAT_BATCH_ROWS is the max number of tile-rows whose partials one round exchanges;
# rounds drop from HT_LOCAL to ceil(HT_LOCAL / C). Bounded per-program by XCORE_STAT_L1_BUDGET
# (cb_gather scales K*C fp32 tiles) — the sanctioned "relax the one-round-per-tile-row invariant
# under an explicit L1 gate" exception (same class as R3's resident dual-path). C=1 is the
# trivial default (byte-identical to R4); only the pure tiled resident-shard cross-core path
# batches (RM / logical-out-to-DRAM keep C=1, their per-tile-row output drain unchanged).
STAT_BATCH_ROWS = 8  # cap on C (round-batching factor); L1-gated per program by XCORE_STAT_L1_BUDGET
XCORE_STAT_L1_BUDGET = 1_400_000  # per-core L1 arena the tiled-sharded xcore CBs must fit under

# ---- Cross-core stat-tile compaction (op_requirements.md R6b lever 1) ----
# The gathered/broadcast cross-core stat is a REDUCE_ROW result whose only consumed data is
# COLUMN 0 (the master fold is element-wise; pass-2 reads it via BroadcastDim::Col). In an
# fp32 tile column 0 lives entirely in faces 0 (rows 0-15) + 2 (rows 16-31), i.e. byte ranges
# [0,1024) and [2048,3072). Moving only those bytes during the GATHER (the dominant ~86% of
# the round, K partials converging on the master) leaves the untransferred faces stale — the
# master fold sums-then-ignores them, so the output is numerically BYTE-IDENTICAL. `_gather_runs`
# turns STAT_COMPACT_MODE into up to two contiguous (offset,len) runs applied per stat tile
# (the slot stride stays a full fp32 tile). MODE 0 is byte-identical to R6a (the trivial default,
# a live knob); the tiled sharded path uses the col-0-faces compaction. Ablation (BLOCK 8x8,
# blackhole, median of 8 fresh trials): mode 0 = 119036 ns, mode 1 (3 KB, 1 txn) = 116219 ns,
# mode 2 (2 KB, 2 txn) = 108028 ns -> the core-to-core L1 gather is BANDWIDTH-dominated (unlike
# the DRAM `tile_reorder` regime), so skipping the unused middle face wins despite the extra
# transaction. Only the GATHER (~86% of the round) is compacted; the broadcast leg (~14%, one
# mcast of C contiguous tiles) stays full to avoid per-tile mcast-splitting overhead.
#   0 = full 4 KB (1 run)            — byte-identical R6a default
#   1 = faces 0-2 contiguous 3 KB    — 1 run, 25% fewer bytes
#   2 = faces 0 & 2 only 2 KB        — 2 runs, 50% fewer bytes (skips the unused middle face) [WINNER]
STAT_COMPACT_MODE = 2
_STAT_TILE_BYTES = 4096  # fp32 32x32 tile (4 faces x 1024 B); face = 1024 B (asserted == tile_fp32)
_STAT_FACE_BYTES = _STAT_TILE_BYTES // 4
# Round/compute pipelining (R6b lever 2). The COMPLEMENTARY step to the gather compaction: flat
# on its own (mode0 pipe-on 119036 == baseline 119027) but wins 1.05x once the compacted gather
# shrinks the round (BLOCK 8x8 113810 pipe-off -> 107995 pipe-on). Shipped ON; gated to the
# multi-round tiled path (single-round WIDTH groups degenerate byte-identically). Live knob.
XCORE_PIPELINE_LOOKAHEAD = True

# ---- Cross-core two-stage (hierarchical) gather (op_requirements.md R6c lever 1) ----
# The flat gather fans K-1 workers into ONE master (fan-in ~68 ns/core, master-serialized fold on
# the round's critical path — R6b's characterized residual). For a group that is a clean 2D
# rectangle (nx x ny, nx>1 and ny>1) the gather is done in TWO stages over the grid rectangle:
#   stage 1 : each grid row's nx cores -> the row's x0 "row-leader" (fan-in nx-1, in parallel across
#             the ny rows); each row-leader folds its nx partials -> a row-partial (no finalize).
#   stage 2 : the ny row-leaders -> the root (= the (x0,y0) master; fan-in ny-1); the root folds the
#             ny row-partials + (+eps, rsqrt) -> 1/RMS, then broadcasts (the existing segmented mcast).
# Fan-in drops from K-1 to (nx-1)+(ny-1) and the fold is distributed off the single master — the
# `tensix_all_reduce` grid-two-stage pattern (examples/master.md: grid two-stage wins when the grid
# is busy or the payload is tiny; on a 1-D group it collapses to the flat root reduce). Engaged ONLY
# on the pure tiled sharded cross-core path when EVERY group is such a rectangle with the master at
# its low corner AND Ht_local==1 (single round, C=1 — the 2D WIDTH perf geometries 8x4/7x4). 1-D
# groups (WIDTH n x 1, BLOCK per-row) and ragged/logical/RM keep the flat root gather byte-identical.
# Live knob (single source of truth); False falls back to the R6b flat gather everywhere.
XCORE_TWO_STAGE_GATHER = True
# The extra stage-2 handshake+fold (row-leader write -> SEM_GATHER2 -> root wait -> root fold) is a
# ~fixed ~900 ns cost on the round's critical path, so two-stage only pays off once the fan-in it
# saves — (NX-1)*(NY-1) transfers at the measured ~68 ns/transfer, i.e. break-even ~13 transfers —
# exceeds it. Measured (blackhole, gap-free 7-wide K-sweep, per_w_t=1): saving 6 (7x2) REGRESSES
# 2.5%, saving 12 (7x3) is flat, saving 18 (7x4) wins 1.05x. So engage only when
# (NX-1)*(NY-1) >= 13 — the 2D WIDTH perf geometries (8x4 saving 21, 7x4 saving 18) qualify; small-ny
# 2D groups fall back to the byte-identical flat gather (no regression).
XCORE_TWO_STAGE_MIN_SAVING = 13

# ---- Cross-core two-phase (tile-index) distributed fold (op_requirements.md R6e) ----
# The dominant BLOCK-8x8 round residual (in-op ablation: 29.7 us = 28% of the op, fully on the
# critical path) is the group MASTER's SERIAL K-partial fold arithmetic — the K-1 add_tiles +
# rsqrt for all HT_LOCAL tile-rows done on ONE core while everyone waits (the gather DATA MOVEMENT
# is only 1.6 us / 1.5%). Two-stage (R6c) distributes the fold but is topologically a no-op on
# BLOCK's 1-D groups; two-phase distributes by TILE-INDEX so it works on a 1-D line: every core
# pushes each batched tile-row's partial to that row's FOLDER (owner = row % NUM_FOLDERS), each of
# the NUM_FOLDERS = min(C, K) folders gathers+folds a disjoint set of the C rows, scatters its
# finalized 1/RMS to the root, and the root assembles all C and mcasts back. This is the R6d
# on-device bake-off winner (two_phase_reduce_mcast 1.40x vs Pattern A at the exact BLOCK 1-D
# topology). Engaged ONLY on the pure tiled sharded cross-core path with C>1 multi-round batching
# (Ht_local % C == 0 -> uniform per-round ownership) and every group mcast-able (n_seg>0). C=1 /
# WIDTH single-round / RM / logical / two-stage keep the byte-identical flat master fold. Live knob.
XCORE_TWO_PHASE_FOLD = True

# ---- Pass-2 batching (op_requirements.md R6g lever 1) ----
# R6f's ablation pins the dominant BLOCK-8x8 residual to the per-core COMPUTE floor; R6f lever 3
# already batched pass 1's square into one chain, but pass 2 (x·rstd + ·gamma) was still issued one
# one-tile chain per W-tile. Batch pass 2's x·rstd (and the tile-aligned ·gamma) over the whole
# per_w_t-tile W-slice in ONE eltwise_chain per tile-row so the fixed per-call chain init/reconfig
# amortizes over the block — exactly the R6f lever-3 pattern, applied to pass 2, and identical to
# the interleaved resident pass 2 (rms_norm_compute.cpp block_shape). This deepens cb_norm 2 ->
# 2*per_w_t (the batch buffer) — the SAME size cb_xsq has carried unconditionally since R4 (the
# pass-1 square block buffer), so it is proven-safe on every cross-core path; it is folded into the
# XCORE_STAT_L1_BUDGET C gate (single source of truth) and gated to the non-RM cross-core path (RM
# keeps its own per-tile pass-2 loop + cb_norm=2). Live knob; False keeps the R4/R6f per-tile pass 2
# (byte-identical), cb_norm back at depth 2.
XCORE_PASS2_BATCH = True


def _gather_runs(mode: int):
    """(off0, len0, off1, len1) contiguous byte runs the gather moves per stat tile."""
    if mode == 1:
        return (0, 3 * _STAT_FACE_BYTES, 0, 0)  # faces 0,1,2 contiguous (col-0 in faces 0,2)
    if mode == 2:
        return (0, _STAT_FACE_BYTES, 2 * _STAT_FACE_BYTES, _STAT_FACE_BYTES)  # faces 0 and 2 only
    return (0, _STAT_TILE_BYTES, 0, 0)  # full tile (byte-identical)


def _ceildiv(a: int, b: int) -> int:
    return -(-a // b)


def _pick_block_size(Wt: int) -> int:
    """Largest divisor of Wt that is <= 8 (the double_buffer sweet spot; not 1).

    Phase-1 value of the BLOCK_SIZE knob (op_design.md §1). Kept a function so a
    later refinement can raise the cap / change the policy in one place.
    """
    for candidate in range(min(8, Wt), 0, -1):
        if Wt % candidate == 0:
            return candidate
    return 1


def _f32_bits(value: float) -> int:
    return struct.unpack("I", struct.pack("f", float(value)))[0]


def _elem_size(tensor: ttnn.Tensor) -> int:
    """Bytes per element — used ONLY for RM stick-byte math (`cols * elem`).

    Block-float dtypes (bfloat8_b) have no fixed per-element size (16 values
    share one exponent), so `element_size()` raises for them. bf8b is TILE-only
    (bf8b+RM is INVALID), so the RM regime — the only consumer of this value —
    never runs for it; return 0 as an unused placeholder instead of raising.
    Page-size math (below) uses `buffer_aligned_page_size()`, which is correct
    for block-float (returns the tile page), so it is left as-is.
    """
    try:
        return tensor.element_size()
    except (ValueError, RuntimeError):
        return 0


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    gamma: "ttnn.Tensor | None" = None,
    epsilon: float = 1e-6,
    compute_kernel_config: "ttnn.ComputeConfigDescriptor | None" = None,
) -> ttnn.ProgramDescriptor:
    device = input_tensor.device()

    # Cross-core W-split scheme (op_design.md §1 lamp 2, §5): WIDTH/BLOCK-sharded
    # inputs pre-place the hidden dim across cores, so the RMS reduce is cross-core.
    # Dispatched to a separate builder + kernel set; the interleaved row-parallel
    # path below is untouched (Refinement 4).
    _ml = input_tensor.memory_config().memory_layout
    if _ml in (ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TensorMemoryLayout.BLOCK_SHARDED):
        return _create_sharded_xcore_descriptor(
            input_tensor,
            output_tensor,
            gamma=gamma,
            epsilon=epsilon,
            compute_kernel_config=compute_kernel_config,
        )

    # HEIGHT_SHARDED local per-core reduction (op_design.md §1 lamp 3, Refinement 5):
    # rows are split across cores, each core keeps FULL-W rows, so the RMS reduce stays
    # LOCAL per core — the exact row-parallel scheme, just with the row-shard resident in
    # L1. A knob-turn: reuse the interleaved row-parallel reader/compute (the R3 resident
    # indexed two-pass) with cb_x_in/cb_out backed ZERO-COPY on the sharded buffers (no
    # NoC read/write), core assignment pinned by the shard grid. Not the cross-core
    # WIDTH/BLOCK scheme (orthogonal mechanism).
    if _ml == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        return _create_height_sharded_descriptor(
            input_tensor,
            output_tensor,
            gamma=gamma,
            epsilon=epsilon,
            compute_kernel_config=compute_kernel_config,
        )

    shape = list(input_tensor.shape)

    origin_W = int(shape[-1])
    origin_H = int(shape[-2])
    NC = 1
    for d in shape[:-2]:
        NC *= int(d)

    # Alignment-aware tile geometry (per-image ceil; op_design.md §6).
    Ht_img = (origin_H + TILE_DIM - 1) // TILE_DIM
    Wt = (origin_W + TILE_DIM - 1) // TILE_DIM
    R = NC * Ht_img
    partial_w = origin_W % TILE_DIM
    has_partial_w = partial_w != 0

    BLOCK_SIZE = _pick_block_size(Wt)
    NUM_BLOCKS = Wt // BLOCK_SIZE

    is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    has_gamma = gamma is not None
    # gamma layout is an independent knob from the input layout (a valid TARGET
    # cell is RM input + TILE gamma). RM gamma -> reader reads sticks + compute
    # tilizes; TILE gamma -> reader reads tiles straight into cb_gamma and the
    # compute-side gamma tilize is skipped (op_design.md §5 knob-turn).
    gamma_is_rm = has_gamma and gamma.layout == ttnn.ROW_MAJOR_LAYOUT

    # ---- Logical wide-interleaved / decode W-split (op_design.md §1 lamp 2, R4a) ----
    # When the row-parallel split under-fills the grid (R < num_cores) AND W has more
    # tiles than tile-rows (Wt > R, i.e. wide/few-row: decode rows=32 -> R=1, or wide
    # W=16384/32768/12288), split W across cores via the cross-core combine instead of
    # running few-core. TILE input only — the xcore compute is tile-based; RM input
    # keeps the streaming row-parallel path. Prefill (R=256 >= num_cores) already fills
    # the grid, so it stays on the row-parallel/resident path below.
    _grid = device.compute_with_storage_grid_size()
    _num_cores = _grid.x * _grid.y
    if (not is_rm) and (R < _num_cores) and (Wt > R) and (Wt >= 2):
        return _create_logical_xcore_descriptor(
            input_tensor,
            output_tensor,
            gamma=gamma,
            epsilon=epsilon,
            compute_kernel_config=compute_kernel_config,
        )

    inv_N_bits = _f32_bits(1.0 / float(origin_W))  # scaler = 1/origin_W (true element count)
    eps_bits = _f32_bits(epsilon)

    in_dtype = input_tensor.dtype
    out_dtype = output_tensor.dtype
    in_elem = _elem_size(input_tensor)  # RM-only; 0 placeholder for block-float
    out_elem = _elem_size(output_tensor)
    in_page = input_tensor.buffer_aligned_page_size()
    out_page = output_tensor.buffer_aligned_page_size()

    tile_in = ttnn.tile_size(in_dtype)
    tile_out = ttnn.tile_size(out_dtype)
    tile_bf16 = ttnn.tile_size(ttnn.bfloat16)
    tile_fp32 = ttnn.tile_size(ttnn.float32)

    if has_gamma:
        gamma_dtype = gamma.dtype
        gamma_elem = _elem_size(gamma)  # RM-only; 0 placeholder for block-float
        gamma_page = gamma.buffer_aligned_page_size()
        tile_gamma = ttnn.tile_size(gamma_dtype)
    else:
        gamma_dtype = in_dtype
        gamma_elem = in_elem
        gamma_page = in_page
        tile_gamma = tile_in

    # ---- Resident single-read fast-path predicate (op_design.md §1 lamp 1) ----
    # When the tile-row's x (Wt tiles) + resident gamma fit one core's L1, load x
    # ONCE and read it from L1 in BOTH passes — eliminating the 2nd DRAM read of x —
    # and hold gamma resident (read once per core instead of re-read per tile-row).
    # The compute reads each block at the absolute front offset b*BLOCK_SIZE, so the
    # intermediates (cb_xsq / cb_norm) stay 2*BLOCK_SIZE (NOT sized by Wt) and every
    # prefill width fits. Only cb_x_in and cb_gamma are sized by Wt — the design's
    # sanctioned dual-path exception, gated by the explicit L1 budget. The streaming
    # two-pass path (below) is the fallback. TILE input only (+ TILE / no gamma —
    # RM gamma keeps its tilize path); the perf config is TILE input + TILE gamma.
    is_tile_input = (not is_rm) and (not gamma_is_rm)

    def _resident_l1(x_depth):
        # cb_x_in (x_depth whole rows) + cb_gamma (1 row) + small streaming CBs.
        return (
            x_depth * Wt * tile_in
            + (Wt * tile_gamma if has_gamma else 0)
            + 2 * BLOCK_SIZE * tile_in  # cb_xsq  (per-block)
            + 2 * BLOCK_SIZE * tile_in  # cb_norm (per-block)
            + DEPTH * BLOCK_SIZE * tile_out  # cb_out  (per-block, streamed)
            + max(DEPTH, 2) * tile_fp32  # cb_rstd
            + (2 if has_partial_w else 1) * tile_bf16  # cb_scaler
        )

    # Pick the largest input-CB depth that fits the budget (RESIDENT_X_DEPTH..1) so
    # the reader prefetches the next tile-row where L1 allows, and single-buffers the
    # widest rows that only fit at depth 1. Single source of truth for the depth.
    resident_x_depth = 0
    for d in range(RESIDENT_X_DEPTH, 0, -1):
        if _resident_l1(d) <= L1_RESIDENT_BUDGET:
            resident_x_depth = d
            break
    use_resident = is_tile_input and (resident_x_depth > 0)

    # CB page counts branch on the path (single source of truth for each CB). Only
    # cb_x_in and cb_gamma are resident-sized; the intermediates stay per-block.
    x_in_pages = resident_x_depth * Wt if use_resident else DEPTH * BLOCK_SIZE
    xsq_pages = 2 * BLOCK_SIZE
    norm_pages = 2 * BLOCK_SIZE
    gamma_cb_pages = Wt if use_resident else DEPTH * BLOCK_SIZE
    out_pages = DEPTH * BLOCK_SIZE

    # ---- Work distribution: split R tile-rows across the whole grid ----
    grid_size = device.compute_with_storage_grid_size()
    (
        _num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        rows_per_core_g1,
        rows_per_core_g2,
    ) = ttnn.split_work_to_cores(grid_size, R, row_wise=True)

    assignment = []  # (core, row_start, num_rows)
    start = 0
    for group, per_core in ((core_group_1, rows_per_core_g1), (core_group_2, rows_per_core_g2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, start, per_core))
            start += per_core

    # ---- Circular buffers (op_design.md §7); no CB sized by an op dimension ----
    CB_X_STICKS = 0
    CB_X_IN = 1
    CB_SCALER = 2
    CB_GAMMA = 3
    CB_GAMMA_STICKS = 4
    CB_OUT = 16
    CB_OUT_STICKS = 17
    CB_XSQ = 24
    CB_RSTD = 25
    CB_NORM = 26

    cbs = []

    def add_cb(idx, page_size, num_pages, fmt):
        cbs.append(
            ttnn.CBDescriptor(
                total_size=num_pages * page_size,
                core_ranges=all_cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page_size)],
            )
        )

    # input tiles. streaming: DEPTH*BLOCK_SIZE (both passes re-read from DRAM).
    # resident: resident_x_depth*Wt — whole tile-row(s) held so pass 2 reads L1, not DRAM.
    add_cb(CB_X_IN, tile_in, x_in_pages, in_dtype)
    # reduce scaler: 1/W (+ partial tile), bf16, wait-not-pop across all rows
    add_cb(CB_SCALER, tile_bf16, 2 if has_partial_w else 1, ttnn.bfloat16)
    # pass-1 intermediate x^2 (square -> reduce), per-block (2*BLOCK_SIZE) both paths
    add_cb(CB_XSQ, tile_in, xsq_pages, in_dtype)
    # 1/RMS (1 tile/row), fp32 accumulate, held across pass 2
    add_cb(CB_RSTD, tile_fp32, max(DEPTH, 2), ttnn.float32)
    # pass-2 intermediate x*rstd (mul<Col> -> mul<Row>), per-block (2*BLOCK_SIZE) both paths
    add_cb(CB_NORM, tile_in, norm_pages, in_dtype)
    # output tiles (TILE: -> writer; RM: -> untilize), per-block streamed (DEPTH*BLOCK_SIZE)
    add_cb(CB_OUT, tile_out, out_pages, out_dtype)

    if is_rm:
        # RM x: raw sticks packed for compute-side tilize (tile-paged, TILE granularity)
        add_cb(CB_X_STICKS, tile_in, DEPTH * BLOCK_SIZE, in_dtype)
        # RM out: untilized row-major (tile-paged output of compute untilize)
        add_cb(CB_OUT_STICKS, tile_out, DEPTH * BLOCK_SIZE, out_dtype)

    if has_gamma:
        # cb_gamma holds tiles in both regimes: TILE gamma -> reader is the
        # producer (direct tile read); RM gamma -> compute-tilize is the producer.
        # Single producer per compiled program (CT-arg dispatch), same pattern as
        # cb_x_in for the input layout.
        add_cb(CB_GAMMA, tile_gamma, gamma_cb_pages, gamma_dtype)
        # cb_gamma_sticks is the RM-gamma-only tilize input; not needed for TILE gamma.
        if gamma_is_rm:
            add_cb(CB_GAMMA_STICKS, tile_gamma, DEPTH * BLOCK_SIZE, gamma_dtype)

    # ---- Reader kernel ----
    reader_ct_args = [
        Ht_img,
        Wt,
        BLOCK_SIZE,
        NUM_BLOCKS,
        origin_H,
        origin_W,
        inv_N_bits,
        1 if has_partial_w else 0,
        partial_w if has_partial_w else TILE_DIM,
        1 if is_rm else 0,
        1 if has_gamma else 0,
        in_elem,
        gamma_elem,
        in_page,
        gamma_page,
        1 if gamma_is_rm else 0,
        1 if use_resident else 0,
        0,  # X_RESIDENT: interleaved reads x from DRAM (Refinement 5 sets this on HEIGHT)
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    reader_rt_args = ttnn.RuntimeArgs()
    in_addr = input_tensor.buffer_address()
    gamma_addr = gamma.buffer_address() if has_gamma else 0
    for core, row_start, num_rows in assignment:
        reader_rt_args[core.x][core.y] = [in_addr, gamma_addr, row_start, num_rows]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---- Writer kernel ----
    writer_ct_args = [
        Ht_img,
        Wt,
        BLOCK_SIZE,
        NUM_BLOCKS,
        origin_H,
        origin_W,
        1 if is_rm else 0,
        out_elem,
        out_page,
        0,  # X_RESIDENT: interleaved writer drains cb_out to DRAM (RM+HEIGHT sets this on R5a)
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    out_addr = output_tensor.buffer_address()
    for core, row_start, num_rows in assignment:
        writer_rt_args[core.x][core.y] = [out_addr, row_start, num_rows]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ---- Compute kernel ----
    compute_ct_args = [
        BLOCK_SIZE,
        NUM_BLOCKS,
        1 if is_rm else 0,
        1 if has_gamma else 0,
        1 if has_partial_w else 0,
        eps_bits,
        1 if gamma_is_rm else 0,
        1 if use_resident else 0,
        0,  # X_RESIDENT: interleaved (Refinement 5 sets this on HEIGHT)
    ]

    compute_rt_args = ttnn.RuntimeArgs()
    for core, row_start, num_rows in assignment:
        compute_rt_args[core.x][core.y] = [num_rows]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=compute_kernel_config if compute_kernel_config is not None else ttnn.ComputeConfigDescriptor(),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )


# ---------------------------------------------------------------------------
# HEIGHT_SHARDED local per-core reduction builder (op_design.md §1 lamp 3, R5)
# ---------------------------------------------------------------------------
# Rows are split across cores; each core keeps FULL-W rows, so the RMS reduce
# stays LOCAL per core — the row-parallel scheme with the row-shard resident in
# each core's L1. A knob-turn on the interleaved row-parallel path: it REUSES the
# interleaved reader + compute (the R3 resident indexed two-pass) unchanged except
# for two compile-time flags:
#   * cb_x_in is backed ZERO-COPY on the resident input shard (compute self-arms it
#     via X_RESIDENT; the reader skips the x read — no NoC read of the own shard);
#   * cb_out is backed ZERO-COPY on the resident output shard (sized to the whole
#     shard; the compute's Streaming pack fills it in place, so NO writer is needed).
# Core assignment + per-core tile-row count are pinned by the shard grid (each core's
# resident shard = its per-core block). Not the cross-core WIDTH/BLOCK scheme.
def _create_height_sharded_descriptor(
    input_tensor,
    output_tensor,
    *,
    gamma=None,
    epsilon=1e-6,
    compute_kernel_config=None,
):
    device = input_tensor.device()
    shape = list(input_tensor.shape)
    origin_W = int(shape[-1])
    origin_H = int(shape[-2])
    NC = 1
    for d in shape[:-2]:
        NC *= int(d)

    Ht_img = _ceildiv(origin_H, TILE_DIM)
    Wt = _ceildiv(origin_W, TILE_DIM)
    partial_w = origin_W % TILE_DIM
    has_partial_w = partial_w != 0

    # BLOCK_SIZE / NUM_BLOCKS: the same live block knob as the interleaved path
    # (single source of truth). The per-core block is the WHOLE resident shard;
    # within a tile-row it is streamed in NUM_BLOCKS blocks of BLOCK_SIZE tiles.
    BLOCK_SIZE = _pick_block_size(Wt)
    NUM_BLOCKS = Wt // BLOCK_SIZE

    has_gamma = gamma is not None
    gamma_is_rm = has_gamma and gamma.layout == ttnn.ROW_MAJOR_LAYOUT

    inv_N_bits = _f32_bits(1.0 / float(origin_W))
    eps_bits = _f32_bits(epsilon)

    in_dtype = input_tensor.dtype
    out_dtype = output_tensor.dtype
    in_page = input_tensor.buffer_aligned_page_size()
    out_page = output_tensor.buffer_aligned_page_size()
    # RM input (Refinement 5a): the resident shard is row-major sticks. element bytes drive
    # the loopback repack/write byte math; the shard stick stride is the buffer's aligned
    # page size (in_page / out_page). Not used on the TILE path.
    is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    in_elem = _elem_size(input_tensor) if is_rm else 0
    out_elem = _elem_size(output_tensor) if is_rm else 0

    tile_in = ttnn.tile_size(in_dtype)
    tile_out = ttnn.tile_size(out_dtype)
    tile_bf16 = ttnn.tile_size(ttnn.bfloat16)
    tile_fp32 = ttnn.tile_size(ttnn.float32)

    if has_gamma:
        gamma_dtype = gamma.dtype
        gamma_page = gamma.buffer_aligned_page_size()
        tile_gamma = ttnn.tile_size(gamma_dtype)
        gamma_elem = _elem_size(gamma)
    else:
        gamma_dtype = in_dtype
        gamma_page = in_page
        tile_gamma = tile_in
        gamma_elem = 0

    # ---- Shard geometry: each core holds a resident FULL-W row-shard ----
    # The shard's CoreRangeSet pins the core assignment (row-major shard order).
    #   TILE input: shard extents are tile multiples -> sh/32 = per-core tile-rows,
    #     sw/32 = Wt (full W); cb_x_in/cb_out zero-copy on the resident tile shards.
    #   RM input (Refinement 5a): shard extents are RM-granule multiples -> sh = per-core
    #     rows (granule 1, generally NOT a multiple of 32), sw = W padded to the RM granule
    #     (8 bf16 / 4 fp32). The collapsed NC*H row sequence is split per_h rows/core in
    #     row-major order (the last core may hold fewer valid rows -> per-core
    #     valid_rows_total). Each core reduces its rows LOCALLY (full-W rows, no cross-core
    #     combine); the reader loopback-repacks the resident RM sticks into tile-padded
    #     cb_x_sticks, compute tilizes into an allocated cb_x_in and untilizes cb_out back
    #     into cb_out_sticks, and a writer loopback-writes the valid columns into the
    #     resident RM output shard. phase=0 (full W); only the W%32 mask applies.
    mem = input_tensor.memory_config()
    ss = mem.shard_spec
    sh, sw = int(ss.shape[0]), int(ss.shape[1])
    grid = ss.grid
    cores = ttnn.corerange_to_cores(grid, None, True)  # row-major shard order

    if is_rm:
        per_h = sh  # rows this core holds (RM granule 1; generally not a mult of 32)
        Ht_local = _ceildiv(per_h, TILE_DIM)  # tile-rows covering the shard (last H-partial)
        total_rows = NC * origin_H  # collapsed row sequence split per_h rows/core

        def _rm_valid_rows(i):
            row0 = i * per_h
            return max(0, min(per_h, total_rows - row0))

    else:
        per_h_tiles = sh // TILE_DIM  # tile-rows this core holds (its resident block)
        Ht_local = per_h_tiles
        assert sw // TILE_DIM == Wt, f"HEIGHT shard width {sw} inconsistent with Wt*32={Wt * TILE_DIM}"

    # ---- Circular buffers (op_design.md §7) ----
    CB_X_STICKS = 0
    CB_X_IN = 1
    CB_SCALER = 2
    CB_GAMMA = 3
    CB_GAMMA_STICKS = 4
    CB_SHARD_IN = 8
    CB_SHARD_OUT = 9
    CB_OUT = 16
    CB_OUT_STICKS = 17
    CB_XSQ = 24
    CB_RSTD = 25
    CB_NORM = 26

    cbs = []

    def add_cb(idx, page_size, num_pages, fmt):
        cbs.append(
            ttnn.CBDescriptor(
                total_size=num_pages * page_size,
                core_ranges=grid,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page_size)],
            )
        )

    if is_rm:
        # RM input (Refinement 5a): cb_shard_in/out zero-copy alias the resident RM
        # row-shards (reader/writer loopback endpoints, no remote re-fetch). cb_x_sticks is
        # tile-padded stick staging (reader loopback -> compute tilize); cb_x_in is the
        # tilize output; cb_out feeds untilize -> cb_out_sticks (writer loopback into the RM
        # output shard). All per-block (DEPTH*BLOCK_SIZE) — the streaming 2-pass re-tilize —
        # so NO CB is sized by Wt and every W/dtype fits L1 (a resident whole-row cb_x_in
        # would OOM for wide fp32; that single-loopback fast-path is the R6 perf lever).
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_SHARD_IN, input_tensor))
        add_cb(CB_X_STICKS, tile_in, DEPTH * BLOCK_SIZE, in_dtype)
        add_cb(CB_X_IN, tile_in, DEPTH * BLOCK_SIZE, in_dtype)
        add_cb(CB_OUT, tile_out, DEPTH * BLOCK_SIZE, out_dtype)
        add_cb(CB_OUT_STICKS, tile_out, DEPTH * BLOCK_SIZE, out_dtype)
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_SHARD_OUT, output_tensor))
    else:
        # zero-copy: consumed/produced in place on the resident tile row-shard (no NoC).
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_X_IN, input_tensor))
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, output_tensor))

    # 1/W reduce scaler (+partial tile), bf16, wait-not-pop across all rows/passes.
    add_cb(CB_SCALER, tile_bf16, 2 if has_partial_w else 1, ttnn.bfloat16)
    # pass-1 x^2 (square -> reduce) and pass-2 x*rstd (mul<Col> -> mul<Row>): per-block.
    add_cb(CB_XSQ, tile_in, 2 * BLOCK_SIZE, in_dtype)
    add_cb(CB_RSTD, tile_fp32, max(DEPTH, 2), ttnn.float32)
    add_cb(CB_NORM, tile_in, 2 * BLOCK_SIZE, in_dtype)
    if has_gamma:
        # gamma STREAMED per block (DEPTH*BLOCK_SIZE): a full-W resident gamma (Wt tiles)
        # would blow L1 on top of the resident input+output shards for wide W. cb_gamma
        # stays small (never sized by Wt), so HEIGHT fits any W (op_design.md §7).
        add_cb(CB_GAMMA, tile_gamma, DEPTH * BLOCK_SIZE, gamma_dtype)
        if gamma_is_rm:
            # RM gamma: reader streams sticks per block into cb_gamma_sticks; compute
            # tilizes them into cb_gamma (mirror of the interleaved RM-gamma knob-turn).
            add_cb(CB_GAMMA_STICKS, tile_gamma, DEPTH * BLOCK_SIZE, gamma_dtype)

    # ---- Reader: scaler prep + resident gamma; x resident (TILE zero-copy) or
    #      loopback-repacked (RM). num_rows = Ht_local tile-rows; RM also passes the
    #      core's true valid_rows_total (last core is short) at rt arg 4. ----
    reader_ct_args = [
        Ht_img,
        Wt,
        BLOCK_SIZE,
        NUM_BLOCKS,
        origin_H,
        origin_W,
        inv_N_bits,
        1 if has_partial_w else 0,
        partial_w if has_partial_w else TILE_DIM,
        1 if is_rm else 0,  # is_rm: RM input loopback-repacks the resident shard (R5a)
        1 if has_gamma else 0,
        in_elem,  # RM loopback byte math (0 for TILE)
        gamma_elem,
        in_page,  # shard stick stride (RM); TensorAccessor page (TILE)
        gamma_page,
        1 if gamma_is_rm else 0,
        0 if is_rm else 1,  # use_resident: TILE = R3 resident indexed two-pass; RM = streaming
        1,  # X_RESIDENT: x resident in L1 (TILE zero-copy shard; RM loopback-repack, no DRAM)
    ]
    # Input accessor is declared-but-unused on the resident path (x never read from DRAM);
    # gamma accessor addresses the interleaved DRAM gamma.
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    reader_rt_args = ttnn.RuntimeArgs()
    in_addr = input_tensor.buffer_address()
    gamma_addr = gamma.buffer_address() if has_gamma else 0
    for i, core in enumerate(cores):
        # row_start unused (x resident); num_rows = tile-rows in the core's shard.
        # RM: arg4 = valid_rows_total (the core's true row count; last core may be short).
        vrt = _rm_valid_rows(i) if is_rm else 0
        reader_rt_args[core.x][core.y] = [in_addr, gamma_addr, 0, Ht_local, vrt]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---- Compute: R3 resident indexed two-pass; TILE self-arms cb_x_in, RM tilizes
    #      cb_x_sticks -> cb_x_in per tile-row and untilizes cb_out -> cb_out_sticks. ----
    compute_ct_args = [
        BLOCK_SIZE,
        NUM_BLOCKS,
        1 if is_rm else 0,  # is_rm: tilize resident sticks + untilize output (R5a)
        1 if has_gamma else 0,
        1 if has_partial_w else 0,
        eps_bits,
        1 if gamma_is_rm else 0,
        0 if is_rm else 1,  # use_resident: TILE = resident indexed two-pass; RM = streaming
        1,  # X_RESIDENT: TILE cb_x_in/cb_out zero-copy (unused for RM streaming compute path)
    ]
    compute_rt_args = ttnn.RuntimeArgs()
    for core in cores:
        compute_rt_args[core.x][core.y] = [Ht_local]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=grid,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=compute_kernel_config if compute_kernel_config is not None else ttnn.ComputeConfigDescriptor(),
    )

    kernels = [reader_kernel, compute_kernel]

    # ---- Writer: RM only (Refinement 5a). TILE has none — compute packs the zero-copy
    #      cb_out in place. RM loopback-writes the untilized cb_out_sticks valid columns
    #      into the resident RM output shard (cb_shard_out alias). ----
    if is_rm:
        writer_ct_args = [
            Ht_img,
            Wt,
            BLOCK_SIZE,
            NUM_BLOCKS,
            origin_H,
            origin_W,
            1,  # is_rm
            out_elem,  # RM loopback byte math + ELEM
            out_page,  # shard stick stride (SHARD_STICK_BYTES)
            1,  # X_RESIDENT: loopback-write to the resident RM output shard
        ]
        # Output accessor declared-but-unused (loopback write, no DRAM).
        writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
        writer_rt_args = ttnn.RuntimeArgs()
        out_addr = output_tensor.buffer_address()
        for i, core in enumerate(cores):
            # arg3 = valid_rows_total (the core's true row count).
            writer_rt_args[core.x][core.y] = [out_addr, 0, Ht_local, _rm_valid_rows(i)]
        writer_kernel = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
            core_ranges=grid,
            compile_time_args=writer_ct_args,
            runtime_args=writer_rt_args,
            config=ttnn.WriterConfigDescriptor(),
        )
        kernels.append(writer_kernel)

    return ttnn.ProgramDescriptor(
        kernels=kernels,
        semaphores=[],
        cbs=cbs,
    )


# ---------------------------------------------------------------------------
# Cross-core W-split builder (op_design.md §1 lamp 2, §5) — WIDTH/BLOCK sharded
# ---------------------------------------------------------------------------
# The hidden dim W is pre-placed across a group of K cores. Each core reduces its
# local W-slice to a partial Σx²·(1/W); one cross-core round per tile-row (gather
# to the group master -> master folds K partials + finalize (+eps, rsqrt) -> the
# 1/RMS is broadcast back) precedes the per-core normalize. Slices are consumed
# LOCALLY via zero-copy sharded CBs (never re-read through a TensorAccessor). The
# reduction group is:
#   WIDTH_SHARDED : ALL cores (every core owns a W-slice of the SAME full-height
#                   rows) -> one group, master = shard core 0.
#   BLOCK_SHARDED : the cores of one grid ROW (same tile-rows, disjoint W-slices)
#                   -> one group per grid row, master = the row's x=0 core.
# All-unicast transport (topology-agnostic; NoC-mcast/two-stage is the R6 lever).

# Semaphore ids (monotone counters; reused across disjoint BLOCK groups — a sem id
# resolves to a per-(core,id) L1 cell, so disjoint groups never interfere, §5).
_SEM_GATHER = 0
_SEM_BCAST = 1
_SEM_DONE = 2
_SEM_GATHER2 = 3  # R6c two-stage: row-leader -> root stage-2 gather counter (monotone)


def _create_sharded_xcore_descriptor(
    input_tensor,
    output_tensor,
    *,
    gamma=None,
    epsilon=1e-6,
    compute_kernel_config=None,
):
    device = input_tensor.device()
    shape = list(input_tensor.shape)
    origin_W = int(shape[-1])
    origin_H = int(shape[-2])
    NC = 1
    for d in shape[:-2]:
        NC *= int(d)

    Ht_img = (origin_H + TILE_DIM - 1) // TILE_DIM
    Wt = (origin_W + TILE_DIM - 1) // TILE_DIM
    partial_w = origin_W % TILE_DIM
    has_partial_w = partial_w != 0

    has_gamma = gamma is not None
    # gamma layout is an independent knob (Refinement 2 / 4a): RM gamma -> reader
    # reads the W-slice as row-major sticks + compute tilizes them into cb_gamma;
    # TILE gamma -> reader reads whole tiles straight into cb_gamma. Mirrors the
    # interleaved RM-gamma knob-turn on the cross-core path.
    gamma_is_rm = has_gamma and gamma.layout == ttnn.ROW_MAJOR_LAYOUT
    in_dtype = input_tensor.dtype
    out_dtype = output_tensor.dtype

    mem = input_tensor.memory_config()
    ml = mem.memory_layout
    ss = mem.shard_spec
    sh, sw = int(ss.shape[0]), int(ss.shape[1])
    grid = ss.grid
    cores = ttnn.corerange_to_cores(grid, None, True)  # row-major shard order
    is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    def _v(core):
        vc = device.worker_core_from_logical_core(ttnn.CoreCoord(core.x, core.y))
        return int(vc.x), int(vc.y)

    # ---- Per-core group assignment (single source of truth for the topology) ----
    # entry: (core, slice_index, master_core, is_partial_holder, w_tile_start, vwt)
    entries = []
    rm_percore = None  # RM-input only: {(x,y): (valid_cols, valid_rows_total, reduce_partial_w)}
    if is_rm:
        # Refinement 4b — RM-input sharded: the resident W-slice is `sw` elements wide,
        # an arbitrary multiple of the RM granule (8/4 el), NOT a whole number of tiles.
        # The slice starts at element column w_offset, which is generally sub-tile. We
        # PHASE-ALIGN it to the global 32-tile grid: g0 = w_offset//32 is the first
        # global tile, phase = w_offset%32 is the leading offset inside tile 0. The
        # reader loopback-writes x into cb_x_sticks at column `phase` (leading [0,phase)
        # columns stay 0 -> contribute 0 to the SUM reduce), and the gamma W-slice is
        # read at the tile-ALIGNED column (g0+wt)*32 (so the DRAM read is aligned — a
        # sub-tile column offset faults). valid_end = phase+valid_cols; the reduce spans
        # ceil(valid_end/32) tiles with the partial scaler masking valid_end%32. Ht_local
        # ceil's the (possibly H-non-aligned) shard height; valid_rows_total clamps to
        # the true collapsed row count so H tensor-padding is never written back.
        Ht_local = _ceildiv(sh, TILE_DIM)  # tile-rows (last may be H-partial)
        NCH = NC * origin_H  # true collapsed row count
        rm_percore = {}
        per_w_t = 1  # uniform tilize width = max over cores of ceil((phase+sw)/32)

        def _rm_core(c, slice_index, master, w_offset, valid_rows_total):
            nonlocal per_w_t
            g0 = w_offset // TILE_DIM
            phase = w_offset % TILE_DIM
            valid_cols = max(0, min(sw, origin_W - w_offset))
            valid_end = phase + valid_cols
            vwt_reduce = _ceildiv(valid_end, TILE_DIM)
            rpw = valid_end % TILE_DIM
            per_w_t = max(per_w_t, _ceildiv(phase + sw, TILE_DIM))
            entries.append((c, slice_index, master, rpw != 0, g0, vwt_reduce))
            rm_percore[(int(c.x), int(c.y))] = (valid_cols, valid_rows_total, rpw, phase)

        if ml == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
            K = len(cores)
            master = cores[0]
            for i, c in enumerate(cores):
                _rm_core(c, i, master, i * sw, sh)  # WIDTH: full-height rows
        else:  # BLOCK_SHARDED — one group per grid row (W split within a row)
            bb = grid.bounding_box()
            x0, y0 = int(bb.start.x), int(bb.start.y)
            x1 = int(bb.end.x)
            nx = x1 - x0 + 1
            K = nx
            for c in cores:
                xrel = int(c.x) - x0
                yrel = int(c.y) - y0
                master = ttnn.CoreCoord(x0, int(c.y))
                _rm_core(c, xrel, master, xrel * sw, max(0, min(sh, NCH - yrel * sh)))
    else:
        per_h_t = sh // TILE_DIM  # tile-rows this core holds
        per_w_t = sw // TILE_DIM  # W-tiles this core holds
        Ht_local = per_h_t
        if ml == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
            # One group; every core owns a W-slice of the full-height rows.
            K = len(cores)
            master = cores[0]
            for i, c in enumerate(cores):
                w_tile_start = i * per_w_t
                vwt = max(0, min(per_w_t, Wt - w_tile_start))
                entries.append((c, i, master, i == len(cores) - 1, w_tile_start, vwt))
        else:  # BLOCK_SHARDED — one group per grid row
            bb = grid.bounding_box()
            x0, y0 = int(bb.start.x), int(bb.start.y)
            x1 = int(bb.end.x)
            nx = x1 - x0 + 1
            K = nx
            for c in cores:
                xrel = int(c.x) - x0
                master = ttnn.CoreCoord(x0, int(c.y))
                w_tile_start = xrel * per_w_t
                vwt = max(0, min(per_w_t, Wt - w_tile_start))
                entries.append((c, xrel, master, xrel == nx - 1, w_tile_start, vwt))

    # workers per master (for the master's broadcast list), keyed by (mx, my)
    workers_of = {}
    for c, slice_index, master, _iph, _wts, _vwt in entries:
        key = (int(master.x), int(master.y))
        if slice_index != 0:
            workers_of.setdefault(key, []).append(c)

    inv_N_bits = _f32_bits(1.0 / float(origin_W))
    eps_bits = _f32_bits(epsilon)

    tile_in = ttnn.tile_size(in_dtype)
    tile_out = ttnn.tile_size(out_dtype)
    tile_bf16 = ttnn.tile_size(ttnn.bfloat16)
    tile_fp32 = ttnn.tile_size(ttnn.float32)

    if has_gamma:
        gamma_dtype = gamma.dtype
        gamma_page = gamma.buffer_aligned_page_size()
        tile_gamma = ttnn.tile_size(gamma_dtype)
        gamma_elem = _elem_size(gamma)  # RM stick-byte math only (col * elem)
    else:
        gamma_dtype = in_dtype
        gamma_page = input_tensor.buffer_aligned_page_size()
        tile_gamma = tile_in
        gamma_elem = 0

    # Sharded: x/out consumed/produced locally via zero-copy sharded CBs. RM input
    # (Refinement 4b) tilizes/untilizes the resident RM shard around the same combine.
    return _assemble_xcore_kernels(
        input_tensor,
        output_tensor,
        gamma,
        entries=entries,
        workers_of=workers_of,
        K=K,
        per_w_t=per_w_t,
        Ht_local=Ht_local,
        Wt=Wt,
        origin_W=origin_W,
        has_gamma=has_gamma,
        gamma_is_rm=gamma_is_rm,
        has_partial_w=has_partial_w,
        partial_w=partial_w,
        inv_N_bits=inv_N_bits,
        eps_bits=eps_bits,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        gamma_dtype=gamma_dtype,
        tile_gamma=tile_gamma,
        gamma_page=gamma_page,
        gamma_elem=gamma_elem,
        all_cores=grid,
        device=device,
        compute_kernel_config=compute_kernel_config,
        x_zero_copy=True,
        out_to_dram=False,
        is_rm=is_rm,
        rm_percore=rm_percore,
    )


# ---------------------------------------------------------------------------
# Logical wide-interleaved / decode W-split (op_design.md §1 lamp 2; Refinement 4a)
# ---------------------------------------------------------------------------
# Same cross-core combine as the physical WIDTH shard, but the input is an
# INTERLEAVED tensor whose W is split LOGICALLY across K cores (no physical shard):
# each core reads its W/K slice from DRAM (per-core tile-column offset) and writes its
# output slice back to DRAM. Used when the row-parallel split under-fills the grid
# (wide/few-row shapes: W=16384/32768/12288; decode rows=32 -> R=1 tile-row). One
# group of K cores handles all R tile-rows (the WIDTH topology), so it reuses the
# xcore reader/compute/writer via the X_FROM_DRAM / X_ZERO_COPY / OUT_TO_DRAM flags.


def _pick_wsplit_cores(Wt, num_cores):
    """Number of W-slices to split Wt tiles across (bounded by cores and Wt).

    per_w_t = ceil(Wt / num_cores) (finest split the grid allows); the actual core
    count is ceil(Wt / per_w_t) so no core is all-padding. Single source of truth for
    the logical split geometry."""
    per_w_t = -(-Wt // num_cores)  # ceildiv
    k_actual = -(-Wt // per_w_t)
    return per_w_t, k_actual


def _create_logical_xcore_descriptor(
    input_tensor,
    output_tensor,
    *,
    gamma=None,
    epsilon=1e-6,
    compute_kernel_config=None,
):
    device = input_tensor.device()
    shape = list(input_tensor.shape)
    origin_W = int(shape[-1])
    origin_H = int(shape[-2])
    NC = 1
    for d in shape[:-2]:
        NC *= int(d)

    Ht_img = (origin_H + TILE_DIM - 1) // TILE_DIM
    Wt = (origin_W + TILE_DIM - 1) // TILE_DIM
    R = NC * Ht_img
    partial_w = origin_W % TILE_DIM
    has_partial_w = partial_w != 0

    has_gamma = gamma is not None
    gamma_is_rm = has_gamma and gamma.layout == ttnn.ROW_MAJOR_LAYOUT
    in_dtype = input_tensor.dtype
    out_dtype = output_tensor.dtype

    grid_size = device.compute_with_storage_grid_size()
    num_cores = grid_size.x * grid_size.y

    # One group of K cores splits W; each core handles all R tile-rows (HT_LOCAL=R).
    per_w_t, K = _pick_wsplit_cores(Wt, num_cores)
    cores = ttnn.corerange_to_cores(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))}),
        None,
        True,
    )[:K]
    all_cores = ttnn.num_cores_to_corerangeset(K, grid_size, True)

    # entry: (core, slice_index, master_core, is_partial_holder, w_tile_start, vwt)
    master = cores[0]
    entries = []
    for i, c in enumerate(cores):
        w_tile_start = i * per_w_t
        vwt = max(0, min(per_w_t, Wt - w_tile_start))
        entries.append((c, i, master, has_partial_w and (i == K - 1), w_tile_start, vwt))

    workers_of = {}
    for c, slice_index, m, _iph, _wts, _vwt in entries:
        if slice_index != 0:
            workers_of.setdefault((int(m.x), int(m.y)), []).append(c)

    inv_N_bits = _f32_bits(1.0 / float(origin_W))
    eps_bits = _f32_bits(epsilon)

    if has_gamma:
        gamma_dtype = gamma.dtype
        gamma_page = gamma.buffer_aligned_page_size()
        tile_gamma = ttnn.tile_size(gamma_dtype)
        gamma_elem = _elem_size(gamma)
    else:
        gamma_dtype = in_dtype
        gamma_page = input_tensor.buffer_aligned_page_size()
        tile_gamma = ttnn.tile_size(in_dtype)
        gamma_elem = 0

    return _assemble_xcore_kernels(
        input_tensor,
        output_tensor,
        gamma,
        entries=entries,
        workers_of=workers_of,
        K=K,
        per_w_t=per_w_t,
        Ht_local=R,
        Wt=Wt,
        origin_W=origin_W,
        has_gamma=has_gamma,
        gamma_is_rm=gamma_is_rm,
        has_partial_w=has_partial_w,
        partial_w=partial_w,
        inv_N_bits=inv_N_bits,
        eps_bits=eps_bits,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        gamma_dtype=gamma_dtype,
        tile_gamma=tile_gamma,
        gamma_page=gamma_page,
        gamma_elem=gamma_elem,
        all_cores=all_cores,
        device=device,
        compute_kernel_config=compute_kernel_config,
        x_zero_copy=False,
        out_to_dram=True,
    )


def _two_stage_topology(masters, workers_of, Ht_local, device):
    """R6c hierarchical (two-stage) gather topology for 2D reduction groups.

    Returns ``(engaged, nx, ny, percore)``. ``engaged`` is True only when EVERY reduction
    group is a clean 2D rectangle (nx>1 and ny>1, nx*ny == group size) whose master sits at
    the group's low (x0,y0) corner, and ``Ht_local==1`` (single round, C=1 — the 2D WIDTH
    perf geometries). All groups must share the same (nx,ny) so the program-level CT flag can
    express them uniformly. ``percore[(x,y)] = (role, xrel, yrel, rl_vx, rl_vy)`` where role
    2=root, 1=row-leader (x0 column, y!=y0), 0=worker; (rl_vx,rl_vy) are the VIRTUAL coords of
    the core's row-leader (for a row-leader/root, its own coords — the self-loopback source).
    1-D groups (WIDTH n x 1; BLOCK per-row) collapse to the flat root reduce, so this returns
    False for them and the caller keeps the byte-identical flat gather.
    """

    def _v(core):
        vc = device.worker_core_from_logical_core(ttnn.CoreCoord(core.x, core.y))
        return int(vc.x), int(vc.y)

    if Ht_local != 1:
        return False, 1, 1, {}
    nx_g = ny_g = None
    percore = {}
    for key, master in masters.items():
        members = [master] + workers_of.get(key, [])
        xs = [int(m.x) for m in members]
        ys = [int(m.y) for m in members]
        x0, y0 = min(xs), min(ys)
        nx = max(xs) - x0 + 1
        ny = max(ys) - y0 + 1
        if not (nx > 1 and ny > 1 and nx * ny == len(members)):
            return False, 1, 1, {}  # 1-D group or ragged rectangle -> flat gather
        if (nx - 1) * (ny - 1) < XCORE_TWO_STAGE_MIN_SAVING:
            return False, 1, 1, {}  # fan-in saving too small to pay for the extra stage -> flat
        if (int(master.x), int(master.y)) != (x0, y0):
            return False, 1, 1, {}  # root must be the low corner (row-leader of row y0)
        if nx_g is None:
            nx_g, ny_g = nx, ny
        elif (nx, ny) != (nx_g, ny_g):
            return False, 1, 1, {}  # non-uniform groups -> a single CT flag cannot express both
        for m in members:
            xrel = int(m.x) - x0
            yrel = int(m.y) - y0
            role = 2 if (xrel == 0 and yrel == 0) else (1 if xrel == 0 else 0)
            rl = ttnn.CoreCoord(x0, int(m.y))  # this core's row-leader (x0 column, same grid row)
            rlvx, rlvy = _v(rl)
            percore[(int(m.x), int(m.y))] = (role, xrel, yrel, rlvx, rlvy)
    return True, nx_g, ny_g, percore


def _mcast_segments(members_v, master_v):
    """Segmented NoC-mcast plan for one reduction group's broadcast (R6 + R6a lever 2).

    ``members_v`` is the list of (vx, vy) VIRTUAL NoC coords of the group (incl. master);
    ``master_v`` is the mcast sender's (vx, vy). Returns ``(n_seg, segs)`` where ``segs`` is
    a list of ``(xlo, ylo, xhi, yhi, ndests)`` VIRTUAL rectangles:

      * ``n_seg == 1`` — one gap-free rectangle (bounding-box area == group size; the R6 case);
      * ``n_seg == 2`` — two contiguous virtual-x runs that straddle the Blackhole DRAM columns
        (virtual x=8,9 have no worker cores), each a FULL rectangle over the group's y-range
        (the R6a gap-aware case that unblocks the 8-wide WIDTH/BLOCK groups);
      * ``n_seg == 0`` — ragged (logical decode's multi-row-major set; a WIDTH auto-shard group
        wrapping a partial grid row; or any y-gap) — the caller keeps the all-unicast fallback.

    The mcast sender (master) is auto-excluded from the segment it sits in, so that segment's
    ``ndests`` is (members-1); a segment without the master carries its full member count. This
    finds contiguous virtual-x runs and validates each is a full rectangle, so a naive mcast
    to the [xlo..xhi] bounding box (which would fault on the DRAM columns) is never issued.
    """
    if len(members_v) <= 1:
        return 0, []
    vylo = min(v[1] for v in members_v)
    vyhi = max(v[1] for v in members_v)
    xs = sorted({v[0] for v in members_v})
    runs = []
    start = prev = xs[0]
    for x in xs[1:]:
        if x == prev + 1:
            prev = x
        else:
            runs.append((start, prev))
            start = prev = x
    runs.append((start, prev))
    if len(runs) > 2:
        return 0, []  # too fragmented for a 1/2-segment mcast
    segs = []
    covered = 0
    for rxlo, rxhi in runs:
        run_members = [v for v in members_v if rxlo <= v[0] <= rxhi]
        # each run must be a FULL rectangle over the group's whole virtual y-range
        if (rxhi - rxlo + 1) * (vyhi - vylo + 1) != len(run_members):
            return 0, []
        covered += len(run_members)
        master_in = rxlo <= master_v[0] <= rxhi and vylo <= master_v[1] <= vyhi
        ndests = len(run_members) - (1 if master_in else 0)
        segs.append((rxlo, vylo, rxhi, vyhi, ndests))
    if covered != len(members_v):
        return 0, []
    return len(segs), segs


# ---------------------------------------------------------------------------
# Shared cross-core kernel assembly (sharded + logical W-split, one source of truth)
# ---------------------------------------------------------------------------
# The topology (entries/K/per_w_t) and CB placement (zero-copy vs allocated) are
# computed by the two builders above; this assembles the identical xcore
# reader/compute/writer kernels + semaphores from those pieces. x_zero_copy selects
# whether cb_x_in is a zero-copy sharded CB (compute self-arms) or reader-fed from
# DRAM; out_to_dram selects whether cb_out is a zero-copy sharded CB (compute's pack
# finalizes in place) or writer-drained to DRAM.


def _assemble_xcore_kernels(
    input_tensor,
    output_tensor,
    gamma,
    *,
    entries,
    workers_of,
    K,
    per_w_t,
    Ht_local,
    Wt,
    origin_W,
    has_gamma,
    gamma_is_rm,
    has_partial_w,
    partial_w,
    inv_N_bits,
    eps_bits,
    in_dtype,
    out_dtype,
    gamma_dtype,
    tile_gamma,
    gamma_page,
    gamma_elem,
    all_cores,
    device,
    compute_kernel_config,
    x_zero_copy,
    out_to_dram,
    is_rm=False,
    rm_percore=None,
):
    def _v(core):
        vc = device.worker_core_from_logical_core(ttnn.CoreCoord(core.x, core.y))
        return int(vc.x), int(vc.y)

    tile_in = ttnn.tile_size(in_dtype)
    tile_out = ttnn.tile_size(out_dtype)
    tile_bf16 = ttnn.tile_size(ttnn.bfloat16)
    tile_fp32 = ttnn.tile_size(ttnn.float32)
    in_page = input_tensor.buffer_aligned_page_size()
    out_page = output_tensor.buffer_aligned_page_size()
    # RM-input (Refinement 4b): element bytes drive the loopback repack byte math; the
    # resident RM shard stick stride is the buffer's aligned page size. Not used on the
    # tiled path (bf8b is TILE-only, so _elem_size never raises here).
    in_elem = _elem_size(input_tensor) if is_rm else 0

    CB_X_STICKS = 0
    CB_X_IN = 1
    CB_SCALER = 2
    CB_GAMMA = 3
    CB_GAMMA_STICKS = 4
    CB_GATHER = 5
    CB_STAT_HANDOFF = 6
    CB_STAT_GLOBAL = 7
    CB_SHARD_IN = 8
    CB_SHARD_OUT = 9
    CB_ROWPARTIAL = 10  # R6c two-stage: row-leader stage-1 fold output (compute -> writer)
    CB_GATHER2 = 11  # R6c two-stage: root stage-2 fan-in (ny row-partials, fixed base)
    CB_OUT = 16
    CB_OUT_STICKS = 17
    CB_XSQ = 24
    CB_STAT_LOCAL = 25
    CB_NORM = 26

    shard_tiles = Ht_local * per_w_t

    # ---- Pass-2 batching + cb_norm depth (op_requirements.md R6g lever 1) ----
    # Batch pass-2's x·rstd (+ tile-aligned ·gamma) over the per_w_t-tile W-slice in one chain per
    # tile-row (compute PASS2_BATCH), which turns cb_norm into the block batch buffer (depth 2 ->
    # 2*per_w_t — the SAME depth cb_xsq has carried since R4). Gated to the non-RM cross-core path;
    # RM keeps its own per-tile pass-2 loop and cb_norm=2. `norm_depth` is the single source of
    # truth for the cb_norm allocation AND the C-gate fixed_cb below.
    pass2_batch = XCORE_PASS2_BATCH and (not is_rm)
    norm_depth = (2 * per_w_t) if pass2_batch else 2

    # ---- Round-batching factor C (op_requirements.md R6a lever 1) ----
    # Batch C tile-rows' partials into one cross-core round so the flat ~3150 ns round
    # latency amortizes over C rows (rounds = ceil(Ht_local / C)). Only the pure tiled
    # resident-shard cross-core path batches: RM (is_rm) and logical (out_to_dram) keep
    # C=1 (their per-tile-row output drain stays per-tile-row), and single-tile-row groups
    # (Ht_local==1, e.g. WIDTH shards) gain nothing. C is bounded by an explicit L1 gate —
    # cb_gather scales K*C fp32 tiles — so the master never OOMs (the sanctioned relaxation
    # of the R4 "cb_gather stays K" invariant). Single source of truth: every stat-CB depth
    # and the compute/writer round loops derive from this one C.
    batch_rows = 1
    if x_zero_copy and (not is_rm) and (not out_to_dram) and Ht_local > 1:
        shard_l1 = shard_tiles * (tile_in + tile_out)  # zero-copy resident in+out shards
        fixed_cb = (2 * per_w_t) * tile_in  # cb_xsq
        fixed_cb += norm_depth * tile_in  # cb_norm (R6g: 2*per_w_t when pass2_batch, else 2)
        fixed_cb += (2 if has_partial_w else 1) * tile_bf16  # cb_scaler
        if has_gamma:
            fixed_cb += per_w_t * tile_gamma  # cb_gamma
        avail = XCORE_STAT_L1_BUDGET - shard_l1 - fixed_cb
        # stat CBs scale as C*(K+5) fp32 tiles: cb_gather K*C, cb_stat_local 2*C,
        # cb_stat_handoff 2*C, cb_stat_global C.
        per_C = (K + 5) * tile_fp32
        c_max_l1 = max(1, avail // per_C)
        batch_rows = max(1, min(Ht_local, STAT_BATCH_ROWS, c_max_l1))
    C = batch_rows

    # ---- Perf 1: pass-1 fused square-DEST-accumulate reduce + pass-2 reconfig-skip ----
    # Two compute-side per-call-overhead levers on the cross-core compute floor (Perf-1
    # tournament, measured isolated + end-to-end on Blackhole; see changelog "## Perf 1").
    # Both are predicate-guarded fast paths WIDENED to their measured winning domain — every
    # case outside the predicate keeps the byte-identical R6f/R6g path (no regression):
    #   * PASS1_FUSED (1.48× on pass 1 @per_w_t=4): fuse the square INTO an FPU DEST-accumulate
    #     (the vwt x-tiles collapse into ONE summed x² tile in DEST, packed once to cb_xsq),
    #     then a SINGLE REDUCE_ROW over that 1 tile — vs the streaming square(vwt)+matmul-
    #     reduce(vwt). Winning domain: the tiled path (not RM), NO partial W (full scaler), and
    #     2 <= per_w_t <= 4. Lower bound 2: per_w_t=1 is measured FLAT (nothing to collapse).
    #     Upper bound 4: bounds the bf16-DEST accumulation error above the 0.9995 soft-PCC gate
    #     under fp32_dest_acc_en=False (PCC 0.99984 @vwt4; ~0.9992 @vwt8 -> excluded).
    #   * PASS2_RECONFIG_SKIP (byte-identical; wins on MULTI-ROW pass-2): drop the per-chain
    #     srcA/pack data-format reconfigs in pass 2 that are provably constant (input/output
    #     dtype), establishing them once at pass-2 entry. Winning domain: the non-RM batched
    #     path with gamma present, no partial W (so the pad-copy path is not interleaved), AND
    #     C > 1 (multi-tile-row rounds — the entry reconfig amortizes over C·2 chains/call).
    #     At C=1 (single-row WIDTH/decode) there is nothing to amortize and the entry reconfig
    #     is a net ~1-2% LOSS (measured), so those stay on the byte-identical per-chain path.
    pass1_fused = _PERF1_FASTPATH and (not is_rm) and (not has_partial_w) and (2 <= per_w_t <= 4)
    pass2_reconfig_skip = _PERF1_FASTPATH and pass2_batch and has_gamma and (not has_partial_w) and (C > 1)

    # ---- R6c two-stage (hierarchical) gather topology ----
    # `masters` maps each group's master (mx,my) -> master core; derived once here (single
    # source of truth), reused by the two-stage topology and the mcast-segment plan below.
    masters = {}
    for _c, _si, _master, _iph, _wts, _vwt in entries:
        masters.setdefault((int(_master.x), int(_master.y)), _master)
    two_stage = False
    nx_ts = ny_ts = 1
    ts_percore = {}
    if XCORE_TWO_STAGE_GATHER and x_zero_copy and (not is_rm) and (not out_to_dram):
        two_stage, nx_ts, ny_ts, ts_percore = _two_stage_topology(masters, workers_of, Ht_local, device)

    # ---- Per-group segmented NoC-mcast plan (R6 + R6a lever 2) — computed once, single source ----
    # Reused by the two-phase gate below (needs every group mcast-able) and the writer RT builder.
    group_seg = {}
    for _key, _master in masters.items():
        _members = [_master] + workers_of.get(_key, [])
        _members_v = [_v(m) for m in _members]
        group_seg[_key] = _mcast_segments(_members_v, _v(_master))

    # ---- R6e two-phase (tile-index) distributed fold topology ----
    # NUM_FOLDERS = min(C, K) folder cores (slice_index 0..NUM_FOLDERS-1) each fold a disjoint set
    # of the C batched tile-rows (owner(cc) = cc % NUM_FOLDERS). owned_max(f) = number of rows in
    # [0,C) with cc%NUM_FOLDERS==f = ceil((C-f)/NUM_FOLDERS) (constant per round since Ht_local%C==0).
    # folder_coords[key] = virtual coords of folders 0..NUM_FOLDERS-1 (every core needs them to push).
    two_phase = False
    num_folders = 1
    folder_coords = {}  # group key -> [(vx,vy)] * num_folders
    tp_owned = {}  # (x,y) -> (is_folder, owned_max)
    if (
        XCORE_TWO_PHASE_FOLD
        and x_zero_copy
        and (not is_rm)
        and (not out_to_dram)
        and (not two_stage)
        and C > 1
        and K > 1
        and (Ht_local % C == 0)
        and all(group_seg[key][0] > 0 for key in masters)  # every group mcast-able
    ):
        # NUM_FOLDERS must DIVIDE C so every folder owns EXACTLY C/num_folders tile-rows
        # and its cb_gather advances by owned*K == depth each round, WRAPPING back to the
        # fixed cross-core-write base. min(C,K) is only safe when it divides C: when it does
        # NOT (e.g. K=7 < C=8), folder 0 owns an extra row, so its cb_gather advances 2*K per
        # round (wraps) while folders 1..K-1 advance K per round (do NOT wrap) — the per-folder
        # write pointers desynchronize after round 0, the fixed-base gather-push lands partials
        # in stale slots, the fold sums garbage, and rsqrt(negative) -> NaN (the R6e gate
        # violation on BLOCK 7-wide). Take the LARGEST divisor of C that is <= min(C,K) so
        # ownership is uniform; two-phase engages only when that divisor is >= 2 (num_folders==1
        # degenerates to the single-master fold the byte-identical flat path already handles).
        cap = min(C, K)
        num_folders = max((d for d in range(cap, 0, -1) if C % d == 0), default=1)
        if num_folders >= 2:
            # slice_index -> core, per group; folders are slice 0..num_folders-1
            two_phase = True
            for key, master in masters.items():
                by_si = {}
                for c, si, m, _iph, _wts, _vwt in entries:
                    if (int(m.x), int(m.y)) == key:
                        by_si[si] = c
                if any(f not in by_si for f in range(num_folders)):
                    two_phase = False  # a folder slice is missing -> bail (keep flat)
                    break
                folder_coords[key] = [_v(by_si[f]) for f in range(num_folders)]
            if two_phase:
                for c, si, m, _iph, _wts, _vwt in entries:
                    is_f = si < num_folders
                    owned = (C - si + num_folders - 1) // num_folders if is_f else 0
                    tp_owned[(int(c.x), int(c.y))] = (1 if is_f else 0, owned)
    max_owned = ((C - 0 + num_folders - 1) // num_folders) if two_phase else 0  # folder 0 owns the most

    cbs = []

    def add_cb(idx, page_size, num_pages, fmt):
        cbs.append(
            ttnn.CBDescriptor(
                total_size=num_pages * page_size,
                core_ranges=all_cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page_size)],
            )
        )

    if is_rm:
        # RM-input sharded (Refinement 4b): the resident W-slice is row-major sticks.
        # cb_shard_in/out zero-copy alias the resident shards (reader/writer loopback
        # endpoints, no NoC re-fetch of remote data). cb_x_sticks / cb_out_sticks are
        # tile-padded stick staging; cb_x_in / cb_out are the (allocated) tile CBs the
        # compute tilize/untilize produce/consume — per_w_t = ceil(sw/32) padded tiles.
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_SHARD_IN, input_tensor))
        add_cb(CB_X_STICKS, tile_in, 2 * per_w_t, in_dtype)
        add_cb(CB_X_IN, tile_in, 2 * per_w_t, in_dtype)
        add_cb(CB_OUT, tile_out, 2 * per_w_t, out_dtype)
        add_cb(CB_OUT_STICKS, tile_out, 2 * per_w_t, out_dtype)
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_SHARD_OUT, output_tensor))
    else:
        if x_zero_copy:
            # zero-copy sharded input W-slice (consumed locally, no NoC read)
            cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_X_IN, input_tensor))
        else:
            # logical split: reader reads this core's W/K slice from DRAM into cb_x_in.
            add_cb(CB_X_IN, tile_in, shard_tiles, in_dtype)
        if not out_to_dram:
            # zero-copy sharded output W-slice (compute's pack finalizes it in place)
            cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, output_tensor))
        else:
            # logical split: compute streams per-tile-row output; writer drains to DRAM.
            add_cb(CB_OUT, tile_out, 2 * per_w_t, out_dtype)

    # RM always emits full(tile0)+partial-or-full(tile1); tiled path is 2 iff has_partial_w.
    add_cb(CB_SCALER, tile_bf16, 2 if (has_partial_w or is_rm) else 1, ttnn.bfloat16)
    # pass-1 squares the whole vwt-tile block before the single block-reduce, so
    # cb_xsq must hold a full W-slice block (2*per_w_t double-buffers it).
    add_cb(CB_XSQ, tile_in, 2 * per_w_t, in_dtype)
    # Stat CBs scale with the round-batch factor C (R6a). cb_gather / cb_stat_global are
    # cross-core-written at a FIXED base, so their depth must be EXACTLY the per-round count
    # (K*C / C) — full rounds wrap the fifo back to base so every core's get_write_ptr matches.
    # cb_stat_local / cb_stat_handoff are local (compute<->writer), double-buffered at 2*C.
    add_cb(CB_STAT_LOCAL, tile_fp32, 2 * C, ttnn.float32)
    if two_stage:
        # R6c: two-stage gather (C=1, single round). cb_gather is the STAGE-1 fan-in (nx
        # partials per row-leader, fixed base -> depth == the per-round count so the fifo
        # wraps to base on every core); cb_gather2 is the STAGE-2 fan-in on the root (ny
        # row-partials, fixed base); cb_rowpartial is the row-leader fold output (compute ->
        # writer handoff, double-buffered). All allocated on every core (uniform CB layout);
        # only the role uses each. Stat compaction (col-0 faces) composes through both legs.
        add_cb(CB_GATHER, tile_fp32, nx_ts, ttnn.float32)  # stage-1 fan-in (nx partials)
        add_cb(CB_ROWPARTIAL, tile_fp32, 2, ttnn.float32)  # row-leader stage-1 fold output
        add_cb(CB_GATHER2, tile_fp32, ny_ts, ttnn.float32)  # stage-2 fan-in (ny row-partials)
    elif two_phase:
        # R6e: each folder's fan-in is only its owned rows × K partials (fixed base, drained/round),
        # and cb_rowpartial is the folder-local finalized-rstd buffer (compute fold -> writer scatter).
        add_cb(CB_GATHER, tile_fp32, max_owned * K, ttnn.float32)  # folder fan-in (max_owned rows × K)
        add_cb(CB_ROWPARTIAL, tile_fp32, 2 * max_owned, ttnn.float32)  # folder-local finalized rstds
    else:
        add_cb(CB_GATHER, tile_fp32, K * C, ttnn.float32)  # fixed-base fan-in (K partials/round × C rows)
    add_cb(CB_STAT_HANDOFF, tile_fp32, 2 * C, ttnn.float32)
    add_cb(CB_STAT_GLOBAL, tile_fp32, C, ttnn.float32)  # depth C -> fixed base for the batched mcast-back
    # cb_norm is the pass-2 x·rstd intermediate. R6g lever 1 batches pass 2 over the per_w_t-tile
    # W-slice, making cb_norm the block batch buffer (depth 2*per_w_t, same as cb_xsq); depth 2 on
    # the non-batched (RM / knob-off) path. Single source: norm_depth.
    add_cb(CB_NORM, tile_in, norm_depth, in_dtype)
    if has_gamma:
        # cb_gamma holds tiles in both regimes; single producer per compiled
        # program (reader for TILE gamma, compute-tilize for RM gamma).
        add_cb(CB_GAMMA, tile_gamma, per_w_t, gamma_dtype)
        if gamma_is_rm:
            # RM-gamma-only tilize input: the reader pushes the W-slice as sticks
            # (one tile-wide page per read); compute tilizes vwt tiles into cb_gamma.
            add_cb(CB_GAMMA_STICKS, tile_gamma, per_w_t, gamma_dtype)

    # ---- Reader (scaler + gamma W-slice + [logical] x-from-DRAM) ----
    reader_ct = [
        1 if has_gamma else 0,
        inv_N_bits,
        1 if has_partial_w else 0,
        partial_w if has_partial_w else TILE_DIM,
        gamma_page,
        1 if gamma_is_rm else 0,
        gamma_elem,
        origin_W,
        0 if x_zero_copy else 1,  # X_FROM_DRAM
        Wt,
        Ht_local,
        per_w_t,
        in_page,
        1 if is_rm else 0,  # IS_RM
        in_elem,  # ELEM (RM loopback byte math)
        in_page,  # SHARD_STICK_BYTES (resident RM input shard stick stride)
    ]
    # gamma accessor at idx 16; input accessor chained after it (logical path only).
    reader_ct.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    reader_ct.extend(
        ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
        if not x_zero_copy
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    reader_rt = ttnn.RuntimeArgs()
    gamma_addr = gamma.buffer_address() if has_gamma else 0
    in_addr = input_tensor.buffer_address() if not x_zero_copy else 0
    for c, _si, _m, _iph, w_tile_start, vwt in entries:
        vc, vrt, rpw, phase = rm_percore[(int(c.x), int(c.y))] if is_rm else (0, 0, 0, 0)
        reader_rt[c.x][c.y] = [gamma_addr, w_tile_start, vwt, in_addr, vc, vrt, rpw, phase]
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_xcore_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---- Stat-tile compaction (R6b lever 1) ----
    # Move only the col-0-bearing faces of each fp32 stat tile during the gather (numerically
    # byte-identical). Applied on the pure tiled sharded cross-core path (WIDTH + BLOCK physical
    # shards, where the K-partial gather is the dominant round cost); RM / logical paths keep the
    # full-tile transfer (mode 0, byte-identical to R6a). Single source of truth: `_gather_runs`.
    stat_compact_mode = STAT_COMPACT_MODE if (x_zero_copy and (not is_rm) and (not out_to_dram)) else 0
    assert (
        _STAT_TILE_BYTES == tile_fp32
    ), f"stat compaction face offsets assume an fp32 stat tile ({_STAT_TILE_BYTES} B); got {tile_fp32}"
    g_off0, g_len0, g_off1, g_len1 = _gather_runs(stat_compact_mode)

    # ---- Round/compute pipelining (R6b lever 2 → R6f lever 2) ----
    # Issue batch r+1's pass 1 one round ahead so the local reduce overlaps the writer's
    # synchronous cross-core round. R6b measured it FLAT while the master's serial fold sat ON the
    # round's critical path (the round dwarfed the compute, so overlap bought nothing). R6f lever 2:
    # R6e distributed that fold off the single master (two-phase, folders own disjoint tile-rows),
    # so the round is now short enough that overlapping batch r+1's pass 1 under round r WINS on the
    # multi-round BLOCK path — so pipelining is enabled on the two-phase path too (its own loop
    # honors PIPELINE_LOOKAHEAD; the writer is serial across rounds and unchanged). Only meaningful
    # on the multi-round tiled path (num_rounds>1); single-round groups (WIDTH, two-stage C=1)
    # degenerate byte-identically. Two-stage (R6c, Ht_local==1) stays off (single round anyway).
    pipeline_lookahead = (
        XCORE_PIPELINE_LOOKAHEAD and (x_zero_copy and (not is_rm) and (not out_to_dram)) and (not two_stage)
    )

    # ---- Writer (cross-core transport + [logical] out-to-DRAM) ----
    writer_ct = [
        _SEM_GATHER,
        _SEM_BCAST,
        _SEM_DONE,
        Ht_local,
        K,
        tile_fp32,
        1 if out_to_dram else 0,  # OUT_TO_DRAM
        Wt,
        per_w_t,
        out_page,
        1 if is_rm else 0,  # IS_RM
        in_elem,  # ELEM (RM loopback byte math)
        out_page,  # SHARD_STICK_BYTES (resident RM output shard stick stride)
        C,  # C_ROWS (round-batching factor; idx 13)
        g_off0,  # gather run0 offset (idx 14)
        g_len0,  # gather run0 length (idx 15)
        g_off1,  # gather run1 offset (idx 16)
        g_len1,  # gather run1 length (idx 17)
        1 if two_stage else 0,  # TWO_STAGE (R6c; idx 18)
        nx_ts,  # NX — cores per grid row in the group (stage-1 fan-in; idx 19)
        ny_ts,  # NY — grid rows in the group (stage-2 fan-in; idx 20)
        _SEM_GATHER2,  # stage-2 gather semaphore id (idx 21)
        1 if two_phase else 0,  # TWO_PHASE_FOLD (R6e; idx 22)
        num_folders,  # NUM_FOLDERS (R6e; idx 23) — accessor chained after at <24>
    ]
    writer_ct.extend(
        ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
        if out_to_dram
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    # ---- Refinement 6 + 6a collective-topology lever: per-group NoC-mcast plan ----
    # `group_seg` (the per-group segmented-mcast plan) was derived once above (single source of
    # truth for the two-phase gate, the two-stage topology, and this writer RT). See its comment.
    writer_rt = ttnn.RuntimeArgs()
    out_addr = output_tensor.buffer_address() if out_to_dram else 0
    for c, slice_index, master, _iph, w_tile_start, vwt in entries:
        mvx, mvy = _v(master)
        is_master = 1 if slice_index == 0 else 0
        vc, vrt, _rpw, phase = rm_percore[(int(c.x), int(c.y))] if is_rm else (0, 0, 0, 0)
        n_seg, segs = group_seg[(int(master.x), int(master.y))]
        # fixed fields 0-9; num_workers at 10; n_mcast_seg at 11; seg0 12-16; seg1 17-21;
        # R6c two-stage per-core fields 22-26 (role, xrel, yrel, rl_vx, rl_vy); then either the
        # R6e two-phase tail (is_folder 27, owned_max 28, folder coords 29+ — every core) OR the
        # master's unicast-fallback worker coords from 27 (WORKER_COORDS_BASE). Two-phase and the
        # unicast fallback are mutually exclusive CT-gated paths, so 27+ is reused safely.
        row = [out_addr, w_tile_start, vwt, is_master, slice_index, mvx, mvy, vc, vrt, phase]
        if is_master:
            wl = workers_of.get((int(master.x), int(master.y)), [])
            row.append(len(wl))  # 10 num_workers
            row.append(n_seg)  # 11 n_mcast_seg (0 -> unicast fallback)
            for si in range(2):
                if si < n_seg:
                    row.extend(list(segs[si]))  # (xlo, ylo, xhi, yhi, ndests)
                else:
                    row.extend([0, 0, 0, 0, 0])
        else:
            row.append(0)  # 10 num_workers
            row.append(0)  # 11 n_mcast_seg (workers never broadcast)
            row.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 12-21 seg0/seg1 (unused)
        # 22-26: two-stage role + slots + row-leader coords (0s when two_stage is off)
        role, xrel, yrel, rl_vx, rl_vy = ts_percore.get((int(c.x), int(c.y)), (0, 0, 0, 0, 0))
        row.extend([role, xrel, yrel, rl_vx, rl_vy])
        if two_phase:
            # 27 is_folder, 28 owned_max, 29+ folder coords (every core pushes to all folders)
            is_f, owned = tp_owned[(int(c.x), int(c.y))]
            row.append(is_f)
            row.append(owned)
            for fvx, fvy in folder_coords[(int(master.x), int(master.y))]:
                row.extend([fvx, fvy])
        elif is_master:
            for w in workers_of.get((int(master.x), int(master.y)), []):
                wvx, wvy = _v(w)
                row.extend([wvx, wvy])  # 27+ worker coords (unicast fallback broadcast)
        writer_rt[c.x][c.y] = row
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_xcore_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ---- Compute (local reduce + master combine + normalize) ----
    compute_ct = [
        per_w_t,
        Ht_local,
        K,
        1 if has_gamma else 0,
        1 if has_partial_w else 0,
        eps_bits,
        1 if gamma_is_rm else 0,
        1 if x_zero_copy else 0,  # X_ZERO_COPY (self-arm cb_x_in)
        1 if is_rm else 0,  # IS_RM (tilize resident RM shard, untilize output)
        C,  # C_ROWS (round-batching factor; idx 9)
        1 if pipeline_lookahead else 0,  # PIPELINE_LOOKAHEAD (R6b lever 2; idx 10)
        1 if two_stage else 0,  # TWO_STAGE (R6c; idx 11)
        nx_ts,  # NX — stage-1 fold count (idx 12)
        ny_ts,  # NY — stage-2 fold count (idx 13)
        1 if two_phase else 0,  # TWO_PHASE_FOLD (R6e; idx 14)
        num_folders,  # NUM_FOLDERS (R6e; idx 15)
        1 if pass2_batch else 0,  # PASS2_BATCH (R6g lever 1; idx 16)
        1 if pass1_fused else 0,  # PASS1_FUSED (Perf 1; idx 17)
        1 if pass2_reconfig_skip else 0,  # PASS2_RECONFIG_SKIP (Perf 1; idx 18)
    ]
    compute_rt = ttnn.RuntimeArgs()
    for c, slice_index, _m, is_partial_holder, _wts, vwt in entries:
        role = ts_percore.get((int(c.x), int(c.y)), (0, 0, 0, 0, 0))[0]
        is_f, owned = tp_owned.get((int(c.x), int(c.y)), (0, 0))
        compute_rt[c.x][c.y] = [
            vwt,
            1 if is_partial_holder else 0,
            1 if slice_index == 0 else 0,  # is_master (== root on the two-stage/two-phase path)
            1 if role == 1 else 0,  # is_row_leader (R6c; role 1 = x0 column, y != y0)
            is_f,  # is_folder (R6e; idx 4)
            owned,  # owned_max (R6e; idx 5)
        ]
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_xcore_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=compute_kernel_config if compute_kernel_config is not None else ttnn.ComputeConfigDescriptor(),
    )

    semaphores = [
        ttnn.SemaphoreDescriptor(id=_SEM_GATHER, core_ranges=all_cores, initial_value=0),
        ttnn.SemaphoreDescriptor(id=_SEM_BCAST, core_ranges=all_cores, initial_value=0),
        ttnn.SemaphoreDescriptor(id=_SEM_DONE, core_ranges=all_cores, initial_value=0),
        ttnn.SemaphoreDescriptor(id=_SEM_GATHER2, core_ranges=all_cores, initial_value=0),  # R6c stage-2
    ]

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=semaphores,
        cbs=cbs,
    )
