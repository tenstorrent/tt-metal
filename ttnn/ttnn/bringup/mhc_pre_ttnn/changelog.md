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

## Refinement 3b — Speed up the perf-focus profile T=640, C=7168, bf16 streams (W column broadcast) (debug: fix gate violations)
- Date: 2026-09-30
- What was done: fixed an intermittent precision failure in the R2 W column broadcast. The sender published
  each W chunk to its own compute (`cb_push_back`) before `sender.send()`. With fp32 W, the compute's hi/lo split
  rewrites `cb_weight` in place (aliased `cb_weight_split`), so it could overwrite the chunk while the multicast
  was still reading that source L1. Receivers then got partially split W. The reader now pushes after `send()`
  returns, since that is when the pipe's source guard allows reuse.
  - Reused: the whole R2 path.
  - Changed: the order of 2 lines in the reader's sender loop.
- Accuracy achieved: bf16-X/fp32-W post/comb rel-RMS 2.4–3.1e-4 (gate 5e-4) on [17×512, 1000×28672, 256×24576].
  Stress, 40 seeds at 256×24576: 1 failure before the fix (comb 1.36e-2), 0 after.
- Golden test progress: 206/206 (full suite).
- Perf (BH device-ns):

  | Shape | bf16 | fp32 X |
  |---|---|---|
  | 640×7168 | 191.9 µs | 543.5 µs |
  | 640×1792 | 86.5 µs | 157.3 µs |
  | 1280×4096 | 199.2 µs | 447.6 µs |
  | 4096×1792 | 352.8 µs | 633.5 µs |

  fp32-X 640×7168 lost about 40 µs against the racy version: the sender's in-place split now waits for each
  chunk's mcast. It is still below the pre-R3 567 µs.
- Issues encountered: None beyond the race.
- Tests added: none. The stress probe was saved under `ttnn/ttnn/bringup/mhc_pre_ttnn/tests/unit/probes/`.

## Refinement 4 — Speed up the perf-focus profile T=640, C=1792, bf16 streams
- Date: 2026-09-30
- What was done (perf; nothing added to SUPPORTED). All levers were measured on device. BH device-ns was read in
  process (`ttnn.ReadDeviceProfiler`, `test_mhc_pre_perf_inproc.py`), because the Tracy capture tool crashed on
  every `--profile` run this session. Per-stage `DeviceZoneScopedN` zones were used for the analysis and then
  removed.
  - **W fill on the writer (NoC1), column all-gather with per-share fp32-W split** (earlier R4 step): each row reads
    1/rows of the W slice, its compute splits that share in place, and it multicasts it down the column. The bias is
    also loaded by the writer.
  - **Narrow groups** (perf lamp L2, knob `NARROW_GROUPS`, bf16 X): group_w = the widest width with the fewest
    blocks per group. At 640×1792 that is 20 groups of 5 with one block each (was 10 × 11, 2 blocks). This
    replaced the tree combine; the verifier note asked for the two to be compared, and narrow groups won.
  - **In-DEST coefficient layout transforms** (earlier R4 step): transpose + SFPU subvector transpose. They remove
    the writer's S scatter and the post/comb staging (`cb_coef_out`, `cb_logits_coef`, `cb_out_stage` deleted).
  - **Bounded X read look-ahead**: `X_STREAM_CHUNKS = 4` chunks per block, `X_STREAM_INFLIGHT = 2` outstanding, one
    NoC trid per chunk, each chunk published as it lands. Issuing everything up front let the banks interleave all
    cores' requests, so even chunk 0 landed at the end of the burst.
  - **Streamed projection + Σx²** (`project_sumsq_streamed`, bf16 X / fp32 W): per K chunk, the projection window
    reloads the fp32 running mix exactly (`cb_mix_run`), and a Σx² window packs one partial into `cb_sq_acc`. Then
    one REDUCE_ROW per row runs over the partials. `cb_partial` is now [mix rows | sumsq rows], and the writer
    sends it in one transfer when the block is full.
  - **Fused owned block**: owned rows run coefficients + Sinkhorn + post/comb in one DEST window and park the
    coefficient tile in `cb_coef_keep`. The pre tiles reload it instead of re-running the S gather +
    coefficients. The post/comb store now goes out before the y-mix.
  - **Owner C discount** (`OWNER_C_DISCOUNT = 7`, `_c_split`): when every group has ≤ 1 token tile-row, rank 0
    owns every Sinkhorn row. It gets 7 fewer stream columns and the other ranks absorb them. The fit
    (`core_k_tiles_max`) reads the same split.
  - **Per-row NoC flip** (`READER_NOC_FLIP_ROWS`, default 0, plus the `READER_NOC` knob): groups in the top rows swap
    reader/writer NoCs. This needed per-NoC kernel sets and mcast wires. Measured null (see below), so it is parked
    at 0.
  - `L1_SAFETY_MARGIN` 64 → 96 KB. The CB base sits ~70.7 KB above the unreserved base because the kernel
    binaries grew, and 1×1×2048×20480 bf16 overflowed L1 by 3.2 KB.
  - Reused: the whole R3 path, `project_block_pieces`' matmul realization, the fp32-X exact-reload pattern,
    `sumsq_row` + the reduce helper, and the group mcast / gather. Added: one compute block op, one 1-tile CB,
    host knobs and the per-NoC kernel sets.
- Perf (BH p150, device kernel ns, bf16 X / fp32 X, fp32 W):

  | Shape | R3 bf16 | R4 bf16 | R3 fp32 X | R4 fp32 X |
  |---|---|---|---|---|
  | 640×1792 (focus) | 86.5 µs | **44.3–45.6 µs** | 157.3 µs | 128–130 µs |
  | 640×7168 | 191.9 µs | 151–155 µs | 543.5 µs | 405–411 µs |
  | 1280×4096 | 199.2 µs | 175–178 µs | 447.6 µs | 384–389 µs |
  | 4096×1792 | 352.8 µs | 246 µs | 633.5 µs | 557–563 µs |

  Knob A/Bs at 640×1792 bf16:
  - Owner discount: 0 → 46.6–46.8, 6 → 45.5, 7 → 44.3–44.9, 8 → 45.4 µs. A discount that makes two ranks' `c_start`
    collide mod the 8 DRAM banks is slower (2 → 50.2, 4 → 48.7). The stream stride is 56 tiles ≡ 0 mod 8, so a
    rank's reads walk banks (c_start + c) mod 8.
  - X chunks: 1 → 47.0–47.4 vs 4 → 44.7–45.4 µs.
  - Reader NoC: NoC1 for all readers → 61 µs. Flipping the top 2–5 rows → 48.2–49.7 µs: the flipped rows get fast,
    rows 4–6 become the starved ones, and the X burst still ends at ~25 µs. The flipped writers' W share reads also
    queue behind the X burst, which delays the column all-gather.
- Bottleneck now (zone analysis): the X read is aggregate NoC/DRAM-bound, 9.2 MB by ~24–26 µs (~370 GB/s). On NoC0
  the bottom row finishes at 8 µs and the top rows at 26 µs. The critical group (top row) then pays a ~19 µs serial
  tail: proj + Σx² on the late data ~5 µs, gather + fold + mcast ~3.5 µs, owner coefficients + Sinkhorn ~8 µs
  (Sinkhorn alone: 20 → 1 iteration saves 3.8 µs), y-mix + post/comb stores ~1.5 µs.
- Accuracy achieved: unchanged gates. bf16-X/fp32-W post/comb rel-RMS 2.4–3.1e-4 (gate 5e-4) and y 1.6–1.7e-3 on
  [17×512, 1000×28672, 256×24576]. fp32 X post/comb rel-RMS 2.3–6.1e-5 on [64×4096, 100×4096, 640×7168, 640×28672].
- Golden test progress: two slices together cover all 206 golden + regression cells, all passing. The first
  slice was 127 passed plus 3 loose 20480-wide cells failing on the L1 margin; after the fix all 96 loose cells
  pass, and the complementary slice was 86/86.
- Issues encountered: the Tracy capture tool crashed on every `--profile` run (infra), so measurements used the
  in-process profiler. The L1 margin underestimate is fixed above.
- Tests added: `ttnn/ttnn/bringup/mhc_pre_ttnn/tests/unit/test_mhc_pre_perf_inproc.py`, an in-process device-ns
  probe over `SHAPES` × dtypes × the knob sets in `test_mhc_pre_perf_sweep.py`. It is skipped unless
  `TT_METAL_DEVICE_PROFILER=1`.

## Refinement 5 — Speed up the perf-focus profile T=1280, C=4096, bf16 streams (block × depth co-tune)
- Date: 2026-09-30
- What was done (perf; nothing added to SUPPORTED). All device-ns below are BH p150 in-process
  `ttnn.ReadDeviceProfiler` medians of 3–5 calls (`test_mhc_pre_perf_inproc.py`, new `MHC_PRE_PERF_REPEAT` /
  `_XSHAPES` / `_WDTYPE`). Temporary `DeviceZoneScopedN` zones were used for the analysis, then removed.
  - **Diagnosis (1280×4096 bf16 X / fp32 W, 175.8 µs).** 20 groups of 5 cores, 2 token blocks per core. Per-row
    zones: with every reader on NoC0, the top core rows are starved. Row y=3's block-0 X read takes 116 µs, while
    the bottom rows finish everything by ~100 µs. The starved cores then run their whole compute after the data
    lands, a ~52 µs tail (projection + Σx² alone is 15 µs per block with the data present).
  - **Block × depth co-tune (the heading's knobs): measured null or negative, defaults kept.**
    - `BLOCK_TOKEN_TILES_CAP` 2 → 163.8 µs; 2 with depth 1 → 163.9; 4 → 186.7. A bt > 1 block at C=4096 only fits
      with fewer or wider groups, or with depth 1, so it loses the block-to-block overlap (lamp L1).
    - `X_BLOCK_DEPTH_DEFAULT` 3 → 148.8 (vs 148.1); 1 → 174.3.
    - `Y_CHUNK_TILES_CAP` 4 / 16 → 150.1 / 147.8; `Y_DEPTH` 1 / 3 → 151.4 / 148.4; 16 × 3 → 146.4.
    - X stream chunks 2 / 6 / 8 → 151.4 / 146.9 / 148.0; in-flight 1 / 3 / all → 147.7 / 150.4 / 153.6;
      `OWNER_C_DISCOUNT` 0 / 4 → 149.6 / 148.9.
    - All of these are within the ±2 µs noise band or worse, so the bf16 L1 headroom stays unspent. The
      Σx² `cb_sq_acc` handshake question from the verifier notes does not arise (bt stays 1).
  - **NoC placement (`noc_placement`, the lever that won).** Two coupled changes:
    1. `READER_NOC_FLIP_ROWS` is now derived: `round(READER_NOC_FLIP_FRACTION = 0.4 × grid_y)` top rows swap
       reader and writer NoCs. The explicit int override is kept, and 0 = the Refinement 4 placement.
    2. **The W column share moved to the reader** (`W_SHARE_ON_READER`). The reader DRAM-reads it into the
       writer-owned `cb_weight` at the writer's slots, issued ahead of the X burst on its own NoC with transaction
       id 15 (`X_STREAM_CHUNKS` is now capped at 14). It hands the share to the writer through the new token CB
       `cb_w_share_landed` before the first X chunk is waited for. The writer still splits, multicasts and publishes.
    - Why both: flipping alone put the flipped rows' writer W-share reads on the congested NoC. They took
      56–60 µs, and every projection waits for the column all-gather, which ends at its slowest row (flip4 alone
      with the writer W: 157 µs). The share on the reader alone (no flip) starves NoC0's top rows' shares instead
      (640×7168 153 → 174 µs).
    - Ordering the share before X on the writer's NoC (a reader wait on a "W issued" token) measured flat.
      Landing it before any X read (`W_SHARE_BEFORE_X`, parked False) was 159.7 vs 148.
  - **Path gate** (`_placement_levers`). Both levers are off for the pairs where they lose, which keep the
    Refinement 4 placement:
    - bf16 X / bf16 W (unstreamed `matmul_block` projection): 1280×4096 183.2 → 188.3, 2048×5120 354.3 → 377.9 µs.
    - bf16 W without the column broadcast (decode / `group_h > 1`, per-core DRAM W): 1×7168 fp32 X 80.1 → 93.2 µs.
    - Knobs: `PLACEMENT_LEVERS_BF16_X_BF16_W`, `PLACEMENT_LEVERS_BF16_W_R1`, both False.
  - **Latent bug fixed.** On the fp32-X / bf16-W path, nothing waited for the resident `cb_weight` before the split
    projection's matmul read it. The fp32-W paths wait inside their W split, and the bf16-X path waits inside
    `matmul_block`. It worked only because W happened to land before the first X block. Both placement levers
    shift W timing, so the race showed up: golden 202/206, with 4 fp32-X/bf16-W precision failures. Fix: wait for
    `cb_weight` at block start on that path, before the X stats pass, which is where the fp32-W path waits. A wait
    placed later, right before the projection, still failed deterministically at 640×1792 with no flip: whole
    block-0 token rows of the bottom groups came out wrong. I did not pin down the thread-level mechanism for that.
  - Reused: the X stream / transaction-id machinery, the W all-gather (split, multicast, publish unchanged), and the
    per-NoC kernel sets. Added: one token CB, reader W-share issue/landing, derived flip rows, the path gate, and one
    compute wait.
- Perf (device µs, R4 placement → Refinement 5 defaults, same build):

  | Shape | bf16 X / fp32 W | fp32 X / fp32 W | fp32 X / bf16 W | bf16 X / bf16 W |
  |---|---|---|---|---|
  | 1280×4096 (focus) | **175.8 → 148.2** | 387.8 → 378.9 | 393.4 → 349.0 | 183.7 → 182.9 (gated) |
  | 640×1792 | 45.0 → 43.4 | 128.4 → 121.9 | 123.4 → 115.5 | 44.0 → 43.8 (gated) |
  | 640×7168 | 152.6 → 146.4 | 408.3 → 411.2 (+0.7 %, 10 samples, noise band) | 395.4 → 353.1 | 148.5 → 148.2 (gated) |
  | 4096×1792 | 244.9 → 239.2 | 559.6 → 552.5 | 529.5 → 521.9 | 261.0 → 261.9 (gated) |
  | 2048×5120 | 344.2 → 314.7 | 719.2 → 714.7 | 713.7 → 672.1 | 353.2 → 352.8 (gated) |
  | 1×7168 decode | 58.7 → 46.1 | 130.7 → 119.8 | 80.1 → 80.1 (gated) | 40.5 → 40.8 (gated) |
  | 64×4096 | 49.7 → 42.7 | 96.2 → 92.1 | 64.5 → 63.4 | 37.9 → 37.8 (gated) |
  | 32×128 | 19.4 → 18.9 | 24.5 → 24.0 | 20.0 → 19.9 | 19.2 → 19.0 (gated) |

- Bottleneck now (1280×4096 bf16, flip4 zones): the W column all-gather ends at ~50–58 µs. It waits for its slowest
  row's share, which is logical row 4, the top NoC0 row, and is still starved. About 70–85 µs of per-core compute
  follows it: 2 × (projection + Σx² 15.5 µs, coefficients 5–17 µs including the group wait and the owner's
  Sinkhorn, y-mix 7 µs). The middle rows' X lands last, at ~100–105 µs. Landing W first (~26 µs) instead leaves
  those rows X-bound with a ~45 µs tail. That is the same wall.
- Accuracy achieved: gates unchanged. Golden 206/206. Unit directory 91 passed / 1 skipped (includes the new cases).
  Alternating-seed stress (stale L1 cannot mask a race): 0 bad on 1280×16384, 640×7168, 2048×20480,
  256×24576, 64×16384 and 640×28672 across all four dtype pairs × {default, noflip, W on the writer}.
- Golden test progress: 206/206.
- Issues encountered: the latent fp32-X/bf16-W W wait (above). Run-to-run noise at 1280×4096 is ±2–4 µs across
  processes, so every A/B used 3–5 call medians.
- Tests added: `test_mhc_pre_blocking.py::test_mhc_pre_noc_placement_knobs` covers the non-default placement
  branches (noflip, W share on the writer, both) × {fp32 X / bf16 W, bf16 X / fp32 W}, with two alternating
  seeds. Without the fix it fails. `test_mhc_pre_perf_sweep.py`: new knob sets (bt / depth / y window / placement).
  Its unselected default is now a 5-setting smoke set; `MHC_PRE_PERF_KNOBS=all` runs every setting.

## Perf 1 — perf tournament round 1 (measured breakdown, 2 experiments, 2 graduated)
- Date: 2026-09-30. Perf only: nothing was added to SUPPORTED, and the precision contract is unchanged.
- **Focus config** (feature_spec `_PERF_FOCUS`, the `attention:` LOOSE_CASES): bf16 X TILE, fp32 W, fp32_dest_acc_en=True,
  with T×C = 640×7168, 640×1792 and 1280×4096. All three are in SUPPORTED.
- **How it was measured:** BH p150, 100 cores, in-process `DEVICE KERNEL DURATION` (`ttnn.ReadDeviceProfiler`, via
  `test_mhc_pre_perf_inproc.py`). Numbers are medians of 3–10 calls. There is no trial loop: device kernel time has no
  warm-up transient.

### Instrumentation (permanent)
- `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp` was missing from this tree, so it was restored (the same header
  other ops' Perf 1 rounds added). `MaybeDeviceZoneScope` is opt-in: it records only when both `PROFILE_KERNEL` and
  the `KERNEL_PERF_ZONES` define are set, and costs nothing otherwise.
- Stage zones, one question per zone (the wait is split from the work):
  - Reader: `r_w_issue`, `r_w_land`, `r_x_reserve`, `r_x_issue`, `r_x_barrier`.
  - Writer: `w_w_share_wait`, `w_w_split_wait`, `w_w_send`, `w_w_recv`, `w_w_dram`, `w_bias`, `w_partial_wait`,
    `w_partial_send`, `w_gather_wait`, `w_combined_wait`, `w_s_send`, `w_s_recv`, `w_coef_reserve`, `w_y_wait`,
    `w_y_write`, `w_pc_wait`, `w_pc_write`.
  - Compute: `c_w_split`, `c_w_publish`, `c_x_wait`, `c_proj`, `c_sumsq`, `c_sq_reduce`, `c_gather_wait`,
    `c_combine`, `c_coef_wait`, `c_owned`, `c_pre`, `c_ymix`.
- Marker budget: at most 78 of 250 markers are used, and the last zone ends at 100% of each RISC's KERNEL span.
- Hooks:
  - `MHC_PRE_KERNEL_DEFINES="A;B=1"` passes defines to all 3 kernels. Unset, the program is byte-identical to
    production (measured: no change).
  - Ablation switches `MHC_ABLATE_{PROJ,COEF,YMIX,XREAD,YWRITE}` stub a stage's payload and keep its sync
    scaffolding.
  - Tools: `probes/perf_run.sh` (card-pinned, per-card profiler dir) and `probes/zone_report.py` (per-RISC
    per-zone p50 / max, critical-core timeline, marker-cap and coverage checks).

### Measured breakdown (before)
Baseline, bf16 X / fp32 W: 640×7168 145.6 µs, 640×1792 43.8, 1280×4096 147.2. DRAM targets are 118.8 / 29.7 / 131.8.

Cumulative ablation (µs; columns 640×1792 / 640×7168 / 1280×4096):

| arm | 640×1792 | 640×7168 | 1280×4096 |
|---|---|---|---|
| full | 43.6 | 146.0 | 147.9 |
| − projection + Σx² | 40.9 | 133.5 | 137.6 |
| − also coefficients / Sinkhorn | 39.8 | 130.5 | 136.2 |
| − also y-mix (all compute) | 38.1 | 126.9 | 133.2 |
| − also y write | 32.6 | 109.3 | 111.1 |
| − also X read (everything) | 21.7 | 42.1 | 39.7 |
| − X read and y write only (compute on) | 32.0 | 77.1 | 85.2 |

The stages are balanced: neither the compute-only arm nor the DM-only arm reaches the full time. Even with every
payload stubbed, 40–42 µs remain at the big shapes (half the op at 640×1792). That floor is the W column all-gather
prelude plus the group combine scaffolding.

Zone findings, ranked by measured headroom:
1. **1280×4096: block-to-block serialization.**
   - The X read is DRAM-bound: ~106–115 µs, ≈ 380 GB/s.
   - Block 0 waits 28.6 µs for its group's S while X(1) has already landed and sits unprocessed.
   - proj + Σx²(1) is ~15 µs of FPU work, and it then runs serially after y-mix(0).
2. **640×7168: W all-gather gating.**
   - On the critical core, X has landed by 62 µs, but the W column all-gather ends at ~90 µs, so the projection
     cannot stream under the X burst.
   - `w_w_share_wait` reaches 83 µs, because the reader hands the landed share over only after it has *issued* 2
     X chunks (`r_x_issue` up to 85 µs).
3. **640×1792: tail after the X read.**
   - The writer prelude (W gather + `w_bias` 8.7–13.8 µs) runs before the partial send.
   - The tail is then the root's gather wait, the fold (2 µs), the owner's coefficients + Sinkhorn (6.7 µs) and the
     y-mix.

### Portfolio (cap: 2 experiments)
- **Selected:**
  - E1 `prelude_off_critical_path`: hand the reader's W share over as soon as it lands (a poll of trid 15 in the X
    issue loop), plus a writer twin that takes the bias load off the pre-partial path.
  - E2 `cross_block_pipeline`: project block b+1 before coefficients / y-mix of block b.
- **Floated, not tested:**
  - Owner Sinkhorn after its y-mix, to hide it under the y writes.
  - The Sinkhorn SFPU in scaling-vector form (r, c updates instead of renormalizing the full matrix).
  - A group all-gather with a local fold instead of the root fold + multicast.
  - 2 blocks per core at 640×*, so the y write overlaps the X read.

### Verdicts
Every variant of both experiments is **bitwise identical** to the baseline output.

- **E1 `prelude_off_critical_path`: WIN for option f; the rest is recorded as options.**
  - **f (graduated): a fast bias fill.** The coefficient-major tile is 16 contiguous 64-word blocks
    (`slot_index(2j+e, l) = 64j + 2l + e`), so the fill becomes 1024 straight word stores (1.07 µs) instead of a NoC
    zero fill plus 768 scattered stores (5.5 µs BRISC).
    - Focus shapes: flat (145.3 → 145.4 µs, 43.4 → 43.6, and the 1280×4096 gain is E2's).
    - Small shapes (before → after, µs): 32×128 bf16/bf16 19.0 → 14.5, 32×128 bf16/fp32 18.9 → 17.2, 64×4096
      bf16/bf16 38.1 → 35.5.
    - Domain: everywhere, no exceptions.
  - **a (poll) alone: REGRESSION.** At 1280×4096 it goes 147.4–149.2 → 155.9–160.9 µs. The W exchange now finishes at
    the X peak, and the bias read then queues behind the X burst.
  - **a + early bias reads (option `graduation_ab1f.patch`): not graduated.**
    - Wins: 640×7168 bf16 144.9 → 141.5 µs (−3.4), fp32 −7, 2048×5120 fp32 −8.5.
    - Measured regressions: fp32X/fp32W 640×1792 122.6 → 129.3, fp32X/bf16W 640×1792 115.9 → 119.2, fp32X/fp32W
      4096×1792 548 → 553–556, bf16X/fp32W 64×4096 43.0 → 44.1.
    - Cause of the regressions: every core's t = 0 bias read hits the one DRAM bank that holds the bias tile. That
      delays the W share reads from that bank.
    - The split between the two groups of cells is not structural, so a carve-out would have been an allow-list of
      the measured shapes. The patch is kept in the experiment dir for round 2.
  - **Premise correction:** with the poll, the share itself lands late (50–72 µs on the NoC0 rows just below the
    flip at 640×7168). The late token cost only ~10 µs of the ~28 µs W-gating.
- **E2 `cross_block_pipeline`: WIN.** Focus 1280×4096 goes 147.7 → 136.8 µs in the subagent's bench.
  - Menu (µs):

    | option | µs |
    |---|---|
    | next projection early only | 146.0 |
    | split around the tail | 147.8–150.0 |
    | S(b+1) multicast ahead, with handshake | 142.8 |
    | no handshake | 143.5 |
    | root: coef(b) before fold(b+1), with handshake | 142.1 |
    | **no handshake + root coef(b) before fold(b+1): graduated** | **136.8** |

  - The full pipeline at depth 2 (every step) regresses the DRAM-bound multi-block cells: 1280×4096 fp32/fp32
    378 → 400, 2048×5120 fp32/fp32 711 → 756.
  - Depth 3 + full pipeline wins on small-slice fp32-X cells (4096×1792 541.8 → 523.8 fp32 W, 536.7 → 491.3 bf16 W)
    but regresses 1280×4096 / 2048×5120 fp32 (377 → 394, 711 → 779). Both were recorded as options, not graduated.

### What graduated (one path each, the replaced code deleted)
- **Cross-block pipeline** (compute + writer + descriptor).
  - `pipe_at(b) = b+1 < num_blocks && (x_block_depth ≥ 3 || b + x_block_depth ≥ num_blocks)`. This is the ONE
    schedule. At depth 2 it pipelines the last step.
  - Depth-1 plans cannot hold X(b+1) next to X(b), so they stay serial through the same predicate. This is an
    `inexpressible` exception, not a guard, and in practice only bf16X/bf16W multi-block cells take depth 1.
  - `cb_coef_in` is now 2 blocks (+8 KB at bt = 1, `COEF_IN_BLOCKS`).
  - The group S multicast is handshake-free (Flag data-ready). A Counter hangs in the send's atomic barrier on the
    looped-back root copy.
  - fp32 X: `cb_grid` is popped right after the projection.
- **Fast bias fill** (writer `load_bias`), everywhere.
- **Carve-outs:** none. The only measured regression is fp32X/fp32W 4096×1792: 552.0 → 557.4 µs (+1.0%, 10-call
  median, bimodal 540 / 557 modes in both), and the subagent measured 541.7 → 553.8 (+2.2%).
  - It was not carved out. It is ~1% on a non-focus fp32 cell, and the same shape with bf16 W and every other
    multi-block cell win.
  - A dual schedule would double the pipeline's sync invariants.
  - Revisit it in round 2 (the depth-3 option wins that cell).

### Whole-op before → after (same session, µs medians of 5; HEAD via `git stash`, then the graduated tree)

| shape | bf16X/fp32W | fp32X/fp32W | fp32X/bf16W | bf16X/bf16W |
|---|---|---|---|---|
| 640×7168 (focus) | 145.3 → 145.4 | 414.4 → 406.1 | 356.1 → 346.5 | 148.3 → 148.3 |
| 640×1792 (focus) | 43.4 → 43.6 | 122.1 → 118.7 | 116.5 → 112.2 | 44.6 → 44.0 |
| **1280×4096 (focus)** | **147.5 → 138.4** | 380.5 → 370.5 | 360.6 → 335.3 (noisy) | 183.2 → 183.1 |
| 4096×1792 | 239.7 → 229.4 | 552.0 → 557.4 (10 calls) | 523.1 → 531.3 (bimodal 514 / 532 in both) | 260.7 → 261.1 |
| 2048×5120 | 315.0 → 302.9 | 716.8 → 704.5 | 655.5 → 663.3 (noisy: 638–680 before, 635–665 after) | 352.0 → 353.3 |
| 1×7168 (decode, R1 per-core W) | 46.3 → 46.5 | 119.6 → 114.7 | 80.0 → 80.6 | 40.7 → 40.0 |
| 64×4096 | 43.0 → 42.9 | 91.9 → 88.6 | 64.6 → 64.0 | 38.1 → 35.5 |
| 32×128 | 18.9 → 17.2 | 23.9 → 23.0 | 19.9 → 19.8 | 19.0 → 14.5 |

- A re-check of the focus shapes after the stash pop gave 640×7168 145.5, 640×1792 43.8 and 1280×4096 138.3 µs.
- Summary: 2 experiments measured; 2 graduated (E2 plus E1 option f); E1's poll alone is a measured regression, and
  the poll + early bias reads were not graduated (measured regressions).
- Focus 1280×4096 is **−9.1 µs (−6.2%)**, and 640×7168 and 640×1792 are flat. Only 1280×4096 moved, because the
  pipeline needs ≥ 2 blocks per core, and the 640×* focus shapes are single-block with their bottleneck elsewhere.
- After the change, 1280×4096 is X-read-bound on its slowest core (X(1) lands at ~102 µs; target 131.8 µs).
- **Guard-set result:** there is no material regression. The worst cell is +1.0% (4096×1792 fp32X/fp32W, recorded
  above).
- Correctness:
  - Golden `eval/golden_tests/mhc_pre/`: **206/206**.
  - Unit `ttnn/ttnn/bringup/mhc_pre_ttnn/tests/unit/`: 91 passed, 1 skipped.
  - Subagent stress: 11 shapes × 4 dtype pairs × alternating seeds, bitwise identical to base, with ragged and
    ring-wrap shapes of 3–13 blocks.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| matmul_block | capability | `in0` is always read from the CB front with no tile-index base / offset, so it cannot project the next block, which sits *behind* the resident X(b) in the same CB (wrap-aware page offset). The pipelined proj(b+1) on the bf16 X / bf16 W path therefore uses the existing raw `project_block_pieces<1>` (bitwise identical DEST accumulation). The front-block projection keeps the helper. | n/a (inexpressible) | 104500 (1000×1792 bf16X/bf16W, vs 104200 serial helper schedule; flat) | mhc_pre_compute.cpp:1189 |

### Artifacts
`ttnn/ttnn/bringup/mhc_pre_ttnn/perf_experiments/{prelude_off_critical_path,cross_block_pipeline}/` (README.md,
graduation patches, bench tests, zone dumps). `perf_experiments/` has no `__init__.py`, so `import ttnn`'s package
walker does not execute the benches. They import as a namespace package.

## Perf 2 — perf tournament round 2 (measured breakdown, 2 experiments, 2 graduated)
- Date: 2026-09-30. Perf only: nothing was added to SUPPORTED, and the precision contract (fp32_dest_acc_en, fidelity,
  approx mode, dtypes) is unchanged.
- **Focus config** (feature_spec `_PERF_FOCUS`, the `attention:` LOOSE_CASES): bf16 X TILE, fp32 W,
  fp32_dest_acc_en=True, with T×C = 640×7168, 640×1792 and 1280×4096. All three are in SUPPORTED.
- **How it was measured:** BH p150, 11×10 grid, 100 active cores. In-process `DEVICE KERNEL DURATION` via
  `probes/perf_run.sh`, with one fresh run per setting. Medians of 3–7 calls are used only to beat the ±1–2 µs
  noise. Zones come from the permanent `MaybeDeviceZoneScope` set (`KERNEL_PERF_ZONES`).
  - Instrumentation is unchanged. Neither graduation added a stage. At most 78 of 250 markers are used, and the
    last zone ends at 100% of each RISC's KERNEL span on the new plans.

### Measured breakdown (before)
Baseline (µs): 640×7168 146.4, 640×1792 43.4, 1280×4096 136.9. DRAM targets are 118.8 / 29.7 / 131.8.

Cumulative ablation (µs; `MHC_ABLATE_*`):

| arm | 640×1792 | 640×7168 | 1280×4096 |
|---|---|---|---|
| full | 43.4 | 146.4 | 136.9 |
| − projection + Σx² | 41.1 | 130.0 | 139.7 |
| − also coefficients / Sinkhorn | 39.8 | 128.4 | 138.3 |
| − also y-mix (all compute) | 38.1 | 130.0 | 134.8 |
| − also y write | 32.7 | 109.3 | 111.1 |
| − also X read (everything) | 18.2 | 37.4 | 31.3 |
| − X read and y write only (compute on) | 31.1 | 75.0 | 75.7 |

Zone findings, ranked by measured headroom (per-core end-time tables from the zone CSV):
1. **640×7168: one block per core at group_w = 5, depth 1.**
   - The X read is at the DRAM roofline: X lands by ~98 µs on the slowest rows, ≈ 375–450 GB/s for X + 2× W.
     Roofline-gated: no ideas were spent on the X read.
   - The W column all-gather publishes at 79–91 µs on the left group column, so the ~25 µs projection runs after
     X instead of under it.
   - The whole-slice y-mix + y write (~20 µs) is exposed after the combine. With a single block nothing overlaps.
   - Diagnostic knob run (not an idea): `fullrow` (w = 11, 2 blocks + the Perf 1 pipeline) measured 126.1 µs. But
     it loses at 640×1792 (54.1 vs 43.5) and at 1280×4096 (148.3 vs 137.6). The Refinement 4 width rule predates
     the pipeline.
2. **Owner Sinkhorn on the critical path.** `c_owned` is 6.7 µs (SFPU).
   - At 640×1792 the group roots own the only token row, and they end last (43.8 vs 38–41 µs).
   - At 640×7168 under `fullrow`, the owner of block 1 ends ~6 µs after the rest.
3. **640×1792: the tail after the X read (~25 µs) is ~18 µs.**
   - Projection tail 3, round trip 4, coefficients 2.7, Sinkhorn 6.7, y-mix + y write ~4.
   - The all-stubbed scaffolding floor is 18 µs. It is dominated by the W all-gather prelude.

### Portfolio (cap: 2 experiments)
- **Selected:**
  - E1 `pipeline_aware_group_width`: re-derive the bf16-X width selection as a measured cost model, not an
    allow-list.
  - E2 `sinkhorn_sfpu_fast`: a faster SFPU Sinkhorn at identical fp32 precision (ILP, LREG residency, fewer DEST
    round trips, `SFPLOADMACRO`, scaling-vector form).
- **Floated, not tested:**
  - W share priority / per-share K-ordered projection streaming. It conflicts with E1 at 640×7168.
  - The owner's Sinkhorn after its y-mix.
  - A tree / all-gather combine.
  - Round 1's leftover `graduation_ab1f.patch` (early bias).

### Verdicts
- **E1 `pipeline_aware_group_width`: WIN (host-only).**
  - **Candidates:** every group_w ≤ min(grid_x, Ct) that fits L1. A plan at depth < 2 with more than one block goes
    last. Ties go to the widest group.
  - **Cost:** `_block_schedule_cost`, in X-tile-read units ≈ 0.55 µs:
    - `blocks·kmax` for the X stream,
    - `+ H + kmax/n` for the exposed last round trip + y-mix / y write,
    - `+ Σ middle steps max(0, H − 0.4·kmax)` if `pipe_at`, else `0.6·kmax`,
    - where `H = 32 + 0.75·group_cores` (+4 for bf16 W's unstreamed projection), and −12 when groups have ≤ 1
      token row (the owner discount already hides the Sinkhorn).
    - The constants come from a decision fit over all group widths × 48 LOOSE cells. The fp32-W picks are stable
      across base 30–34 × tail fraction 0.35–0.45.
  - **Focus (µs):** 640×7168 145.4 → 128.2 (11/2/2 instead of 5/1/1). 640×1792 and 1280×4096 keep the same plan.
  - **Across the 42 fp32-W cells with T ≥ 512:**
    - 32 are 2.5–55% faster. Examples: 1024×1792 152.1 → 68.1, 2048×4096 315.8 → 241.6, 4096×7168 969.0 → 858.5.
    - 9 keep the same plan, and 1280×5120 is flat (+1.6%, within noise).
    - T = 256 never reaches the rule (Mt < grid_y).
  - **bf16 W:** mostly faster (1024×4096 219.3 → 140.4, 2048×1792 203.1 → 130.1).
    - One `measured-regression`: 4096×4096 bf16 W 455.9 → 471.0 (+3.3%, interleaved 3×3). The old w = 2 depth-1
      plan happens to win there.
  - **Domain:** bf16 X, both W dtypes. fp32 X is out of scope: it is full-row by design, with a measured loss
    recorded in make_plan.
  - **Precision:** only the K partition (summation order) changes. Golden passed 206/206 in the subagent's run.
  - Artifacts: `perf_experiments/pipeline_aware_group_width/` (data/final_tables.md holds all 96 cells).
- **E2 `sinkhorn_sfpu_fast`: WIN.**
  - Option menu (one tile, 20 iterations, µs incl. ~0.11 copy):

    | option | technique | µs | precision |
    |---|---|---|---|
    | v0 | current kernel | 4.97 | — |
    | v1 | constants in L12/L13 | 4.70 | bitwise |
    | v2 | + 4 interleaved sum/recip chains | 3.66 | bitwise |
    | v3 | + cross-direction sums in each scaling pass | 3.53 | bitwise |
    | v4 / v5 | + unrolled softmax / + SFPSWAP max | 3.49 | bitwise |
    | **v6 (graduated)** | + hand-scheduled `SFPLOADMACRO` passes (raw TTI) | **2.73** | bitwise vs the plain formula with fused Newton MADs |
    | v7 | scaling-vector form | 3.61 | not bitwise: ≤ 2.4e-7 (vs fp64 0.85–2.0e-7). Slower than v6, not taken |

  - **The build flips `2 − x·y`.** In the previous binary, sfpi compiled the Newton residual `2 − x·y` either as a
    fused MAD or as MUL + ADDI, depending on unrelated code; `KERNEL_PERF_ZONES` alone flips it. So comb differs
    from the previous build by ≤ 2.4e-7 (≤ 12 ulp).
    - v6 pins the fused form. An in-op lane-by-lane check against v0/v4/v5 ran on the real logits: 0 mismatches.
    - Against fp64 on logits dumped from the op, max abs is 1.19–2.01e-7 (base 1.20–1.96e-7), and mean abs is
      lower in all 6 cells.
  - `c_owned` goes 6.7 → 4.5 µs.
  - **Whole op, µs:**

    | cell | before → after |
    |---|---|
    | 640×1792 bf16/fp32 W | 43.9 → 42.9 |
    | 640×1792 fp32/fp32 | 118.7 → 113.7 |
    | 64×4096 | 42.9 → 40.8 |
    | 32×128 | 17.3 → 14.8 |
    | 640×1792 bf16/bf16 | 44.2 → 41.1 |

    - 640×7168 and 1280×4096 are flat: the owner is not binding there at the old geometry.
  - `OWNER_C_DISCOUNT`: re-measured at 3–7 with the faster Sinkhorn, and 7 is kept. 640×7168 is best at 7
    (143.98 vs 145.9+). 640×1792 is best at 6 by 0.8 µs, and that was not taken as a shape-specific value.
  - **Domain:** everywhere the Sinkhorn runs, with no exceptions. n = 4 is static-asserted, and it is the op's only
    n.
  - Artifacts: `perf_experiments/sinkhorn_sfpu_fast/` (generator + rule checker `bench/gen_sinkhorn_lm.py`).

### What graduated (one path each, the replaced code deleted)
- **E1:** the width rule in `make_plan`. `_block_schedule_cost` replaces the "fewest blocks" key, with no dual
  path, and `NARROW_GROUPS` keeps its meaning (False = full row).
  - **Carve-outs:** none. The only measured regression is 4096×4096 bf16 W (+3.3%, not a focus cell). A structural
    exception would have to drop the depth-1 exclusion for bf16 W, and that costs that path's larger wins
    (−16 to −36%). A cell-keyed guard would be an allow-list. It is recorded here.
- **E2:** `row_norm` / `col_norm` / the plain iteration loop are replaced by the scheduled passes inside
  `sinkhorn()`. The softmax is kept as sfpi, restructured. There is no fallback.

### Whole-op before → after (same card, HEAD via `git stash` of the two op files, medians of 3; µs)

| shape | bf16X/fp32W | fp32X/fp32W | fp32X/bf16W | bf16X/bf16W |
|---|---|---|---|---|
| **640×7168 (focus)** | **145.4 → 123.8** | 407.0 → 404.1 | 347.4 → 342.8 | 148.0 → 144.3 |
| **640×1792 (focus)** | **43.8 → 43.0** | 119.6 → 112.9 | 112.0 → 107.0 | 43.9 → 41.6 |
| **1280×4096 (focus)** | **137.7 → 137.8** | 368.9 → 366.6 | 334.3 → 346.8* | 182.9 → 161.5 |
| 4096×1792 | 227.3 → 228.5 | 560.5 → 517.3 | 516.1 → 505.5 | 261.6 → 227.2 |
| 2048×5120 | 305.1 → 293.5 | 709.3 → 697.0 | 636.2 → 671.7* | 354.6 → 299.6 |
| 2048×4096 | 320.0 → 252.9 | 572.8 → 565.9 | 525.1 → 512.3 | 288.0 → 256.2 |
| 1024×2560 | 205.3 → 89.6 | 238.2 → 235.3 | 224.2 → 220.4 | 145.0 → 94.8 |
| 1×7168 (decode) | 45.9 → 44.6 | 114.4 → 115.0 | 79.7 → 77.3 | 39.8 → 37.4 |
| 64×4096 | 43.0 → 40.8 | 88.7 → 89.0 | 64.3 → 61.5 | 35.8 → 33.5 |
| 32×128 | 17.3 → 14.8 | 23.1 → 20.6 | 19.8 → 17.3 | 14.5 → 12.0 |

- Cells marked * were re-measured interleaved as 2 × (base, new) × 7 calls:
  - 1280×4096 fp32X/bf16W: 351.7 / 344.3 → 351.4 / 344.8, **flat**.
  - 2048×5120 fp32X/bf16W: 647.0 / 664.1 → 677.6 / 673.2, about +2.8% on the medians. The ranges are 636–686
    before and 647–697 after, so they overlap heavily. This cell was already recorded as noisy in Perf 1.
  - fp32 X does not take E1, and on this path E2 only makes a measured-faster Sinkhorn. It was not carved out;
    revisit if it reproduces.
- Re-check of the final tree on 640×7168 (5 calls): median 123.7 µs; with zones on, one call: 123.3 µs.
- The new critical core at 640×7168 is the 11-rank group root: the block-1 gather wait (~11 µs) + fold (4.6 µs ×
  2 blocks). That is the round-3 target (tree / all-gather combine).
- **Summary:** 2 experiments measured, 2 graduated, 0 null.
  - Focus 640×7168 is **−21.6 µs (−14.9%)**, 145.4 → 123.8. The DRAM target is 118.8.
  - 640×1792 is −0.8 µs, and 1280×4096 is flat.
- **Guard set:** no material regression on any focus cell. The worst is 2048×5120 fp32X/bf16W, about +2.8% and
  within noise (above), plus E1's recorded 4096×4096 bf16W +3.3%.
- Correctness:
  - Golden `eval/golden_tests/mhc_pre/`: **206/206** on the combined tree.
  - Unit `test_mhc_pre.py` + `test_mhc_pre_blocking.py`: 24/24.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| no helper covers this (SFPU elementwise / sfpi family) | capability | No sfpi construct or kernel_lib helper can issue `SFPLOADMACRO`, which does a DEST load + a templated MAD (SFPMUL by a pinned LREG) + a delayed store-back in ONE issued instruction. There is also no way to pin LREGs outside the sfpi allocator (L0–L3 macro temps / accumulator, L4–L7 multipliers, L12/L13 = 2.0 / eps), and no API for the macro-config backdoor (`SFPCONFIG` InstructionTemplate / Misc). The sub-units have no interlocks, so the schedule must be static and rule-checked (generated by `perf_experiments/sinkhorn_sfpu_fast/bench/gen_sinkhorn_lm.py`). The best plain-sfpi form (v5) is 3.49 µs. | 3490 (best sfpi, v5) / 4970 (previous kernel) | 2730 | mhc_pre_compute.cpp:594 (`sinkhorn`), passes above it |
