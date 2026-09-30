# Changelog: mhc_pre

## Phase 0 — Core Implementation
- **Date**: 2026-09-30
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier).
  - Regime R1 `group_ksplit_resident`: tokens go to groups and nC is split over group ranks; the X block
    stays resident from the projection to the y-mix (X read once).
  - Group combine: push gather to the root, an fp32 SFPU rank-ordered fold, and an `mcast_pipe` broadcast.
  - Custom fp32 SFPU coefficient and Sinkhorn block ops; FPU y-mix; the Sinkhorn is owned round-robin.
  - Implementer perf pass (BH p150, fp32): 640×7168 went from 415 to 382 µs; the composite baseline is 2396 µs.
- **SUPPORTED at Phase 0**: dtype=[float32], layout=[TILE], weight_dtype=[float32, bfloat16],
  fp32_dest_acc_en=[True], alignment=[tile_aligned, h_non_aligned]
- **Accuracy achieved** (fp32 X, 4 shapes, `test_mhc_pre_precision_baseline.py`):
  - y: PCC ≥ 0.9999998, max_abs ≤ 9.5e-3, rel_rms ≈ 8.6e-4 (ratio median 0.99932: FPU truncation of fp32 X)
  - post: PCC ≥ 0.9999998, max_abs ≤ 9.4e-4, rel_rms ≤ 5.3e-4
  - comb: PCC ≥ 0.9999999, max_abs ≤ 4.0e-4, rel_rms ≤ 4.5e-4
  - The Sinkhorn in isolation is fp32-exact (1.9e-7 vs fp64). Repeat calls are bitwise-identical.
- **Golden suite at Phase 0**: 106 supported_pass, 98 xfail_expected (all `dtype = bfloat16`), 1
  supported_fail. The failure is `test_large_sinkhorn_logits[T64_nC4096]`, a precision failure tracked in
  Refinement 2. Drift is 0 (`verifier_report.json`).
- **Issues encountered / verifier fixes**:
  - DRY: CB page counts were restated in `_l1_bytes`. Both the L1 fit and the ProgramDescriptor now read
    `_cb_table` (behaviour unchanged).
  - `weight_dtype += bfloat16` on golden evidence (no kernel change).
  - `dtype = bfloat16` runs, but 11 bf16-X × fp32-W cells and the bf16 depth chain miss the 5e-4 coefficient
    gate. Root cause, measured: the FPU keeps ~9 bits of a tf32 W, and its in-tile matmul accumulation is not
    fp32-exact. It is not claimed; it moves to Refinement 1 (W hi/lo split).
- **Tests added**: `test_mhc_pre_precision_baseline.py` (verifier), plus probes 002–007. The acceptance
  suite (`test_mhc_pre.py`), `test_mhc_pre_blocking.py` and `test_mhc_pre_perf.py` were already in place.

## Refinement 1 — bf16 residual streams (lands the perf-focus contract)
- **Date**: 2026-09-30
- **What was done**:
  - `SUPPORTED["dtype"]` gains `bfloat16`.
  - For an fp32 W, the compute kernel runs `w_split_block` once, before block 0:
    - Each resident fp32 W tile is loaded via `copy_tile`, with `cb_weight` tagged UnpackToDestFp32.
    - The SFPU computes `W_hi = W & 0xFFFF0000` and `W_lo = W − W_hi`.
    - Both are packed as bf16, in place, into `cb_weight_split`, a second buffer index aliasing
      `cb_weight`'s allocation. There is 0 extra L1, and both are declared in `_cb_table`.
  - `project_block_pieces<2>` (a thin block op over `matmul_tiles`; `matmul_block` cannot take two in1
    operands into one DEST window) accumulates X@W_hi (HiFi4) and X@W_lo (`W_LO_FIDELITY` = LoFi) in the
    same DEST window. The X block is still retained for Σx² and the y-mix.
  - A bf16 W keeps the unchanged `matmul_block` helper path (`w_pieces == 1`).
  - Perf levers: the reader pushes W in `W_CHUNK_TILES = 8` chunks, so the split overlaps the W DRAM read.
    The LoFi lo pass saves 3 of the 4 fidelity phases on half of the doubled matmul.
  - Knobs `W_CHUNK_TILES`, `W_LO_FIDELITY` and `w_pieces()` are single-source in the descriptor.
- **Reused**: the op file gate, `_cb_table`, the reader W load, the `matmul_block` path (bf16 W), and every
  other phase.
- **Added**: the `cb_weight_split` alias, `w_split_block`, and `project_block_pieces`.
- **Accuracy achieved** (bf16 X × fp32 W, `test_mhc_pre_bf16_stream_fp32_weight_precision`):
  - post rel-RMS 2.4–3.1e-4 and comb 2.5–2.6e-4 on 17×512, 1000×7168 and 256×6144, against the 5e-4 gate.
    Before this refinement they were 5.0–5.5e-4.
  - The fp32-X baseline shapes are unchanged or better (post/comb ≤ 4.0e-4 rel-RMS, PCC ≥ 0.99999993).
- **Golden test progress**: 205/206 (was 106 pass + 98 xfail + 1 fail).
  - All 98 bf16 cells pass, plus `test_comb_depth_chain[bf16]`.
  - The only failure is `test_large_sinkhorn_logits[T64_nC4096]` (Refinement 2, unchanged at worst row
    0.0674).
- **Perf** (BH, device kernel ns, bf16 X / fp32 W, before → after):

  | Shape (T×C) | Before | After |
  |---|---|---|
  | 640×7168 | 267.3 µs | 269.6 µs |
  | 640×1792 | 103.6 µs | 105.1 µs |
  | 1280×4096 | 261.9 µs | 264.6 µs |
  | 4096×1792 | 371.6 µs | 373.4 µs |

  fp32 X, after: 640×7168 378 µs (was 382–383), 4096×1792 523 µs (was 521).

  The first split measured 295 µs. Ablation showed ~20 µs from the SFPU split sitting serially after the W
  read, and ~6 µs from the doubled HiFi4 matmul. The chunked W push and the LoFi lo pass removed both.
- **Issues encountered**: `MATH_FIDELITY` exists only on TRISC_MATH. The lo-pass layout decision must match
  on every TRISC, so the main fidelity is passed as a CT arg.
- **Tests added**:
  - `test_mhc_pre_precision_baseline.py::test_mhc_pre_bf16_stream_fp32_weight_precision` (3 shapes).
  - `test_mhc_pre_perf.py` is parametrized over X dtype.

## Refinement 2 — fp32-stream projection precision (large Sinkhorn logits)
- Date: 2026-09-30
- What was done:
  - **Diagnosis (probes 008–015)**: exact bf16 x/W pieces alone were not enough. With them the worst row moved
    only 0.0674 → 0.0673; the limit is 0.06727. The FPU's in-tile (32-long) matmul dot product rounds its sum
    to ~11 bits below the largest product (round-to-nearest, unbiased). That holds even for exact 3-bit × 3-bit
    operands, and it gives a ~2.9e-4 rms post-logit noise floor (the "(b)" floor of Refinement 1). DEST
    accumulation across matmul calls is ~fp32. Sums of products that all lie on one power-of-two grid are exact
    in-tile, for products up to 2^11 (probes 014/015).
  - **Lever — exact-grid (Ozaki-style) split, fp32 X only, compile-time gated** (the bf16-X path compiles to what
    Refinement 1 shipped):
    - W is split once: its global max|W| sets a grid, and W → [W0 = W rounded to a 2^-6 grid, W − W0]. This
      happens in place in `cb_weight_split`, replacing the hi/lo split on this path.
    - Per token row-tile, one SFPU pass gives the exact lane-wise Σx². It feeds `cb_sq_acc` (so r is now
      fp32-exact) and, through `reduce<MAX, REDUCE_SCALAR>` and a scalar broadcast, a bound of |x| ≤ sqrt(max).
      That bound sets the row-tile's grid.
    - Per K chunk (`X_CHUNK_K_TILES = 8`, streamed, never a second resident copy), x → [x0 on a 2^-4 grid,
      x1_hi, x1_mid] as bf16. One DEST window reloads the fp32 running mix exactly and accumulates the 5
      products with q + p ≤ 2. x0·W0 is exact; the remainder products are ≥ 2^-4 smaller, and so is their
      in-tile noise.
  - Knobs, single-sourced in the descriptor: `X_GRID_BITS = 4`, `W_GRID_BITS = 6`, `PRODUCT_ORDER_MAX = 2`,
    `PRODUCT_LO_ORDER = 2` / `X_LO_FIDELITY = HiFi3`, `X_CHUNK_K_TILES = 8`, `X_PIECE_DEPTH = 2`. The config was
    chosen with a CPU model of the FPU rounding, calibrated to the device.
  - **Reused**: `_cb_table`, the `cb_weight_split` alias, the reader W/X loads, the SUM-reduce, the combine,
    coefficients, y-mix and Sinkhorn, and every bf16 path.
  - **Added**:
    - CBs: the `cb_x_fp32` alias, `cb_x_pieces`, `cb_mix_run`, `cb_max_lanes`, `cb_max_scalar`, `cb_grid` and
      `cb_max_scaler` (the reader prepares it as <MAX, REDUCE_SCALAR>).
    - Block ops: `stats_pass`, `grid_block`, `w_grid_split_block`, `x_stats_block`, `x_split_window` and
      `project_block_split`.
- Accuracy achieved (fp32 X, fp32 W, HiFi4):
  - `test_large_sinkhorn_logits[T64_nC4096]`: worst row 0.0672204, against reference 0.0672202 and limit
    0.0672702 (before: 0.0674). Post-logit z rms error vs fp64 is 3.45e-5 (before: 2.9e-4).
  - The columns stay exact (max|colsum − 1| ≈ 1e-6).
  - Every fp32 golden cell passes its PCC / RMS gate.
- Golden test progress: in the slices run, 98/98 fp32-X `test_golden` cells, 10/10 `test_regression.py`
  (including both depth chains) and 22/22 bf16 representatives passed. The bf16 path is compile-time identical.
  The full suite is expected at 206/206 (was 205/206).
- Perf (BH, device kernel ns; fp32 streams are not the perf focus):

  | Shape (T×C) | fp32 X before | fp32 X after |
  |---|---|---|
  | 640×7168 | 378 µs | 567 µs |
  | 640×1792 | ~128 µs | 176 µs |
  | 1280×4096 | ~381 µs | 515 µs |
  | 4096×1792 | 523 µs | 660 µs |

  bf16 640×7168 is 270.2 µs (unchanged: 269.6).
  - The first version measured 652–671 µs.
  - Ablation (640×7168): x split SFPU ~108 µs, stats SFPU ~42, the 3 extra products ~38, the W grid split ~47,
    and ~80 of window/copy structure.
  - Levers kept:
    - Slim split, with the pack converting the tiny tail piece: −50 µs.
    - Σx²-derived bound instead of a per-element max compare: stats SFPU becomes negligible.
    - Uniform grid constant loaded once per tile.
  - Levers measured and not kept:
    - A per-W-chunk grid, so the W split pipelines with the W read: 613 vs 585 µs, because the extra
      reduce/broadcast/init phases cost more than the read they hide.
    - Chunk size / depth sweeps (4–28 / 1–2): flat within noise.
    - LoFi on the order-2 products: too lossy (z rms 8.5e-5).
  - The op is now compute-bound on the fp32 path. The remaining cost is the per-tile split (SFPU ~60 µs, plus
    copies and packs), the one-time W split (~47 µs), and the 5 HiFi products.
- Issues encountered: a 2-piece x split (a wider x0) cannot meet the target with a global W grid (model: ≥ 9e-5
  rms), so 3 x pieces are required.
- Tests added: `test_mhc_pre_precision_baseline.py::test_mhc_pre_fp32_stream_exact_projection` (2 shapes, post-logit
  rms < 1e-4). Probes 008–022 (FPU rounding characterization).

## Refinement 3 — Speed up the perf-focus profile T=640, C=7168, bf16 streams (W column broadcast)
- Date: 2026-09-30
- What was done: built design regime R2 (W column broadcast). Rank r of the first group row reads its W slice from
  DRAM once and multicasts it chunk by chunk (`W_CHUNK_TILES`, so compute's W split still pipelines) down its
  physical column to rank r of every other group: `Mcast1D(PerColumn)` + `mcast_pipe` Sender/ReceiverPipe on the
  reader's NoC0, Counter data-ready, no handshake (the `cb_weight` landing is write-once). The sender publishes
  chunk j to its own compute before the mcast and has chunk j+1's DRAM read in flight during it. Receivers issue
  their X block 0 read before the W receive (verifier note: overlap the mcast with the first X block). Path gate:
  `W_BCAST` knob and `group_h == 1` and ≥ 2 full active group rows; every other shape keeps the R1 per-core W read.
  Bias is still read per core.
  - Reused: `cb_weight` and its chunked push contract, the compute kernel (unchanged), the combine, `_cb_table`.
  - Added: reader W roles (DRAM / sender / receiver), `SEM_W_READY`, the Mcast1D wire (CT + RT) in the descriptor.
- Perf (BH p150, device kernel ns):

  | Shape (T×C) | bf16 X before | bf16 X after | fp32 X before | fp32 X after |
  |---|---|---|---|---|
  | 640×7168 | 272.0 µs | 191.6 µs | 567 µs | 504 µs |
  | 640×1792 | 105.8 µs | 88.4 µs | 176 µs | 152 µs |
  | 1280×4096 | 263.9 µs | 197.3 µs | 515 µs | 422 µs |
  | 4096×1792 | 372.5 µs | 351.3 µs | 660 µs | 651 µs |

  X0-before-W prefetch on vs off (bf16): 191.6 vs 193.9, 88.4 vs 91.2, 197.3 vs 214.5, 351.3 vs 355.7 µs (kept).
  Now: the reader ends at ~122 µs on 640×7168 bf16 (≈ the 121 µs DRAM target); compute/writer run to ~190 µs. The
  remaining gap is the per-block compute + combine tail after the last X block (2 blocks per core), i.e.
  Refinement 5's knobs / perf lamp L3.
- Accuracy achieved: unchanged math (bit-identical W landing). Unit suite 27/27 (acceptance, blocking, precision
  baselines incl. bf16-X/fp32-W post/comb < 5e-4 rel-RMS and fp32 exact-projection gates).
- Golden test progress: slice 26/26 (640×28672 ×4 dtype combos, 1000×7168, 1×28672 decode, 1280×7168, large
  Sinkhorn logits, identical streams, both depth chains); full suite expected 206/206 as before.
- Issues encountered: one Tracy capture-tool crash / timeout on a profile run (capture-side infra; the re-run was clean).
- Tests added: none (the existing perf test covers the path; `--dev` run of 640×7168 clean under the watcher).
