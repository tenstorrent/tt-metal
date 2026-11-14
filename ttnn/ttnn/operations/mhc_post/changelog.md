# Changelog: mhc_post

## Phase 0 — Core Implementation
- **Date**: 2026-11-13
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier).
  Regime R1 plain-2D: one `generic_op` dispatch; split chooser over token-tile rows × C-column tiles on the
  max_grid rectangle (up to 110 cores); reader expands post / comb columns into column-0 fp32 pages; compute
  mixes a block of `block_cols` units per coefficient broadcast with fp32 SFPU MADs (unpack-to-dest, never the
  FPU); writer streams X' with one flush per block.
- **SUPPORTED at Phase 0**: dtype=[float32], sublayer_dtype=[float32], layout=[TILE], fp32_dest_acc_en=[True],
  mode=[plain], coef_source=[separate], norm=[off], feed=[off], alignment=[tile_aligned, h_non_aligned],
  feed_split=[block_aligned, not_block_aligned]
- **Accuracy achieved**: PCC = 1 − 1.3e-15, max_abs_err ≤ 1.0e-6, rel_rms_err = 5.3e-8, ULP p99 = 12, got/true
  ratio median 1.0 (measured on 4 shapes via test_mhc_post_precision_baseline.py)
- **Device time (plain fp32, profiler)**: T160 C5120 117.5 µs (49 % of DRAM peak); T640 C7168 516 µs (vs the
  81-op composite's 3 781 µs); T1024 C1792 214 µs.
- **Golden suite at Phase 0**: 92 / 92 supported cells passing; 625 xfail_expected, 2578 INVALID skipped,
  0 supported_fail / xpass_drift / xfail_wrong_mode (per `verifier_report.json`)
- **Issues encountered** (verifier):
  - DRY fix: compute kernel's DEST slot base / DEST-fit assert now come from the host's `MIX_FIXED_SLOTS`
    (CT arg) instead of a literal 2; reader's coefficient word count derived from the page size.
  - L1 ledger: `cb_coef_raw` row corrected to the 2 pages Phase 0 allocates.
  - Tried a single-block ragged column pass (templated remainder): overflowed the kernel-config buffer;
    reverted, filed as a Refinement 4 lever.
  - Eval harness defect found: `perf_measure.py` uploads inputs inside trace capture (TT_FATAL; one hang).
    Escalated in verification_report.md — blocks the perf refinements until fixed by the eval owner.
- **Tests added**: test_mhc_post.py (planner), test_mhc_post_perf.py (implementer),
  test_mhc_post_precision_baseline.py (verifier)

## Refinement 1 — bf16 streams + packed coefficients + the fused gather (stats and collapse modes)
- Date: 2026-11-14
- What was done:
  - SUPPORTED: dtype / sublayer_dtype += bfloat16 (all four pairs), coef_source += packed, mode += stats, collapse.
    bf16 and packed needed only the SUPPORTED change: CB formats follow the tensors, and the packer's fp32→bf16 is
    round-to-nearest-even, which the X' bf16 bias gate confirms (the design-risk check). Caller buffers stats_output /
    hc_output / y_output and the hc scalars as runtime args (bit-cast fp32, re-sent every call).
  - Regime R2 fused-gather (one generic_op dispatch):
    - project_block = ONE `matmul_block` per token-tile row whose K-blocks are the row's column passes; its
      PreKBlockFn runs mix_block (+ collapse_block), so each pass's bf16 X' is multiplied as soon as it is packed and
      never re-read from DRAM.
    - W is resident per core (or streamed, behind the R2-W-streamed predicate).
    - Σ mix² is an SFPU MAD into a reload accumulator, zeroed per row (packer L1-acc avoided: fp32 DEST).
      finalize_row uses SFPU `sfpu_reduce` REDUCE_ROW as a custom chain element; the writer merges column 0 into
      partial column 24.
    - Combine: fixed 2-level tree with fan-in COMBINE_FANIN = 11, rotating leaders and roots, slot-indexed NoC
      writes plus semaphores (deterministic fold order).
      - Leader slot reuse is credit-controlled by a per-member "ready" bitmask semaphore (bit t%32).
      - The level-1 fold is **lagged one row**, so leaders never wait on the chunk's slowest member.
      - Root work (level-2 fold, stats write, hc) is **deferred past the root's own rows**; each rooted row has its
        own slot region, so no core's row loop ever waits on a root.
    - hc_step_row: a custom SFPU chain element in lane layout (value v = vector v, token = lane; the writer permutes
      in and out with batched loads). Its fp32 recip / rsqrt use Newton steps without programmable constants;
      exp is `_sfpu_exp_fp32_accurate_`. It runs as runtime loops over the DEST pointer (code size).
    - collapse_block: y = Σ P_i·X'bf16_i, as fp32 SFPU MADs with one bf16 rounding.
  - Split chooser, W path: the fused pass is compute-bound per core, so its per-core term is
    rows·(ROW_NS_W + cols·UNIT_NS_W), fitted on device, restricted to single-pass splits
    (cols ≤ MAX_BLOCK_COLS_W = 4). It still sees only (T, C, dtypes, has_W, |L|), so stats bits are identical
    across stats and collapse. It picks R=2 at the perf-focus shapes. Measured forced-R sweep at T160 / T1280:

    | R | T160 stats / collapse | T1280 stats / collapse |
    |---|---|---|
    | 1 | 118 / 161 µs | 709 / 841 µs |
    | 2 | 110 / 150 µs | 490 / 594 µs |
    | 3 | 128 / 170 µs | 596 / 713 µs |
    | 5 | 127 / 169 µs | 569 / 680 µs |

  - Kernel-config buffer (70 656 B): the collapse program hit 79 KB, then 74 KB at block_cols = 5. Fixes:
    - the per-row zero fill replaces a "seed" variant of every mix chain;
    - the hc step is looped instead of unrolled;
    - in the W path a ragged pass runs the full-block chains on zero-filled filler units (reader zeroes F / X / W
      filler from MEM_ZEROS; the writer drops filler X' / y), which removes all one-unit instantiations;
    - MAX_BLOCK_COLS_W = 4.

    TRISC1 went from 43.7 KB to 28.7 KB; collapse binaries total ≈ 44 KB.
  - Reader, shared with plain: coefficient landing double-buffered (row t+1 prefetched behind row t's first block
    barrier), non-volatile column expansion. Plain fp32 measured flat: 117.4 / 512 / 211 µs vs 117.5 / 516 / 214.
- Device time (BH p300c, 110 cores, bf16, packed, HiFi2, profiler `DEVICE KERNEL DURATION`):

  | Shape | plain | stats | collapse |
  |---|---|---|---|
  | T160 C5120 | 77 µs | 110 µs (bring-up op 103 µs) | 134 µs (bring-up op 126.5 µs) |
  | T1280 C5120 | 379 µs | 480 µs | 577 µs |

  Progression during this refinement, T160 collapse: 274 → 173 (deferred root) → 161 (lagged leader fold) → 150
  (chooser) → 134 µs (fast permutation + hoisted hc scalars).
- Remaining headroom (measured, not acted on):
  - The fused pass is compute-bound per core (≈ 6.8 µs per unit: the fp32 SFPU mix chain plus collapse); see
    Refinement 4's mix-chain lamp.
  - The T160 collapse tail (last row: lagged fold → root fold → permute → hc → inverse permute) is ≈ 25 µs.
- Accuracy achieved:
  - stats projection rel L2 ≈ 4e-3 at HiFi2 (gate 1e-2), Σ X'² well inside 1e-6.
  - hc passes HC_OWN_REL 1e-5 and HC_E2E.
  - y passes PCC / RMS (0.99995, 4e-3) and the bias gate.
  - X' is bit-identical across plain / stats / collapse; stats are bit-identical between stats and collapse;
    every output is deterministic.
- Golden test progress:
  - test_golden: 140 / 140 stats + collapse cells; 364 / 364 plain cells (all dtype pairs × coef sources); 70 xfail
    (norm=on, Refinement 2).
  - test_regression: 25 passed, 8 skipped (norm, Refinement 2), 2 failed — `test_projection_hifi4` (see Issues).
- Issues encountered:
  - **`test_projection_hifi4` cannot pass with any FPU projection on this board.** The gate (1e-5) is documented in
    helpers.py as "from the operand-bit model, not yet calibrated on device". Measured:
    - single products are bit-exact (one-hot rows);
    - summing two exact products is inexact 23 % of the time;
    - stock `ttnn.matmul` (bf16 × fp32 or bf16, HiFi4, fp32 acc) has the same ~3.1e-4 floor at K = 32 … 2560.

    The op's HiFi4 projection measures 3.1e-4, exactly that floor. The deterministic evidence is
    test_mhc_post_debug.py. Reported for the eval owner to recalibrate; an exact SFPU projection would cost
    ~10× the HiFi2 path.
  - Program size overflow of the kernel-config buffer (above).
  - `dst_reg +=` immediates are limited to ±3 vectors per INCRWC (split by `advance<K>()`).
  - The tracy `--profile` wrapper breaks `-k` expressions containing spaces / brackets; perf cases are therefore
    parameter-free functions in test_mhc_post_perf.py.
- Tests added:
  - tests/ttnn/unit_tests/operations/mhc_post/test_mhc_post_debug.py (HiFi4 projection precision diagnosis; kept
    per policy).
  - test_mhc_post_perf.py: `test_mhc_post_perf_ds41` (plain / stats / collapse × T160 / T1280),
    `test_mhc_post_perf_hc_iters`, `test_mhc_post_perf_collapse_t160`.

## Refinement 2 — norm + feed (y delivery: rstd return and the fabric all-to-all)
- Date: 2026-11-14
- What was done:
  - SUPPORTED: norm += on, feed += on. EXCLUSIONS += {feed: on, alignment: h_non_aligned} (structural gap, Refinement 2b).
  - **R2-norm** (reused: the combine tree, the hc block op, the partial-merge writer, the deferred root):
    - collapse_block packs y_b into `cb_y_hold`; Σ y_b² of the bf16 y_b (reloaded, exact fp32) accumulates in
      `cb_ysq_acc` next to Σ X'²; finalize_row row-sums it into a second `cb_sumsq_col` page that the writer merges
      into partial column 25, so it rides the same fixed-order 2-level combine.
    - The root's hc step also computes rstd = rsqrt(Σy²·(1/C) + norm_eps) in lane vector 25 (3-Newton fp32 rsqrt);
      the root writer copies it to token order and unicasts 128 B into slot (t − t0) of every member's
      `cb_rstd_land` + SEM_RSTD_COUNT; the member's reader expands it to a column-0 `cb_rstd_col` page.
    - norm_block = y = (y_b · rstd) · w as one block chain per pass (rstd Col broadcast once, w Row broadcast
      per unit, fp32 SFPU, one bf16 rounding), run after the root work: Refinement 1 deferred the roots past
      their own rows, so rstd of every row returns only at the end of the call and the core holds all its rows'
      y_b (≤ 384 KiB on the golden shapes, `Y_HOLD_BUDGET` 512 KiB).
    - X', hc and stats (except column 25) are bit-identical with and without norm_weight.
  - **R4 feed-DRAM-staged** (one MeshProgramDescriptor dispatch: every chip runs the worker program plus its own
    port kernels):
    - Worker set: whenever the fabric is active on a multi-chip mesh, 2·MAX_FEED_LINKS = 4 port cores on logical
      row 0 (centred, the fabric_ring_exchange placement) leave the worker list, feed or not, so a feed call's X',
      stats and hc are the bits of the same call without the feed. The kernel core sets stay merged rectangles
      (general run-merge, ≤ 4 rectangles).
    - Workers write their own column block of y straight into the local feed output D at rows [p·T, ...) and the
      G−1 remote blocks to y_output (staging); after each row (behind a write barrier) they count it in their own
      SEM_FEED_ROWS. The port reader polls every worker's count per wave (per-worker minimum: a shared counter
      let fast workers' row i+1 stand in for a slow worker's row i — measured as stale tiles in D before the fix).
    - TRANSPORT from the examples: port per (direction, link) (fabric_all_gather's reader + sender split); direct
      2-D sends addressed by destination node, the far chip reached through the fabric (fabric_ring_exchange
      `direct`); chunks = bank-contiguous runs of the destination's D pages, ≤ payload // 2048 = 7 tiles per packet
      (14 KiB payload, AG report: the payload is the dominant transport knob); 4 packets in flight, 112 KiB ring
      (AG baseline: 2 per batch is 16-18 % slower, 8 gains ≤ 2 %); one fused write + arrival increment on the last
      packet of each stream (AG: increments stall the receiving router; every 16-32 chunks was 1-2 % faster at
      size); ready fence from every destination before the first packet, both caller semaphores re-armed to 0
      (repeated-call gate passes). Far block is not split per direction (direct 2-D routing fixes its first hop).
    - Port slots are assigned per chip from its distinct first-hop directions: on a QuietBox 2 the 1×4 mesh sits on
      a physical 2×2, so the directions mix E/W with N/S (a fixed E/W→slot map hung the fabric).
- Accuracy achieved: norm passes every gate (≤ 1 bf16 step vs float64 of the norm on the op's own y_b, < 1e-3 of the
  elements off the rounding, Σy² max rel ≤ 1e-5, end-to-end rel L2) on T 64–2048, C 128–7168; the feed output is
  bit-exact (the all-to-all of the no-feed call's y) on every chip at 1 and 2 links, FABRIC_2D_TORUS_XY and FABRIC_2D.
- Device time (BH p300c, single chip, profiler, bf16 packed HiFi2, collapse + norm vs collapse):
  T160 C5120 154 µs vs 134 µs (+20 µs; bring-up op 131.3 µs); T1280 C5120 645 µs vs 561 µs. T1280 ablation of the
  norm's +84 µs: Σy² accumulation ≈ 15 µs, norm_block compute ≈ 32 µs, the rest (≈ 37 µs) the y DRAM writes and the
  rstd round trip, which all sit in the tail because the roots are deferred. Blocking norm_block over the pass
  (one rstd broadcast per pass) moved it < 1 %: the tail, not the chain, binds. Lever (not taken here: it
  restructures Refinement 1's deferred-root schedule and is the same change Refinement 3 needs for per-row feed
  overlap): lag the root work a fixed number of rows instead of deferring it, so rstd returns within
  `norm_rows_in_flight` rows and norm_block / the y writes / the feed publication run inside the row loop.
  The feed call's per-chip time is not measured: `perf_measure.py` still uploads inputs inside the trace capture
  (verification_report.md), which once hung the device.
- Golden test progress: norm (single device) 66/66 collapse cells; feed (1×4 mesh) 100/100 tile-aligned cells +
  4/4 loose feed cases, 28 xfail (h_non_aligned feed, EXCLUSIONS); test_feed.py gates 16/16 (matches-no-feed at
  1/2 links × 5 shapes, raw-y feed, FABRIC_2D, repeated calls, refusals); test_regression norm gates 4/4; prior
  slices unchanged (test_regression 44/46: the 2 known `test_projection_hifi4` calibration failures of Refinement 1).
- Issues encountered:
  - The h_non_aligned feed: chip q's rows start inside a D tile shared with chip q−1 (T = 1: every chip shares one
    tile row), which needs sub-tile row placement → Refinement 2b.
  - Under `--dev` (watcher) the feed program's worker kernels exceed the 70 656 B kernel-config buffer (71 168 B at
    T160); non-dev builds fit. Debug the feed path with DEVICE_PRINT on the port kernels (a separate kernel group,
    not counted against the workers').
- Tests added: tests/ttnn/unit_tests/operations/mhc_post/test_mhc_post_perf.py `test_mhc_post_perf_{norm,nonorm}_t{160,1280}`
  (parameter-free collapse ± norm probes for `--profile`); probes/probe_006..011 (feed debugging, kept).
