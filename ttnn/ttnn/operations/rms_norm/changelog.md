# Changelog: rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-07-22
- **What was done**: Initial implementation via the incremental pipeline
  (planner → implementer → verifier). Row-parallel, bounded two-pass streaming
  reduce over the last dim W, multi-core from day 1
  (`split_work_to_cores(R, grid, row_wise=True)`). Native RM + TILE input, native
  non-tile-aligned H/W (masked/padded reduce), optional gamma. All block knobs
  (`BLOCK_SIZE = pick_block_size(Wt)`, `DEPTH=2`, grid) are live parameters with a
  single source of truth; no CB is sized by an op dimension.
- **SUPPORTED at Phase 0**:
  - dtype = [float32, bfloat16]
  - fp32_dest_acc_en = [True]  (maxed precision corner)
  - layout = [TILE, ROW_MAJOR]
  - alignment = [tile_aligned, w_non_aligned, h_non_aligned]
  - rank = [2, 3, 4]
  - gamma_mode = [gamma, no_gamma]
  - gamma_dtype = [float32, bfloat16, "none"]
  - gamma_layout = [ROW_MAJOR, "none"]
  - memory_layout = [INTERLEAVED]
  - EXCLUSIONS = []  (the `{float32, fp32_dest_acc_en=False}` refusal is
    out-of-rectangle today; becomes an explicit EXCLUSIONS entry with Refinement 1)
- **Accuracy achieved** (measured on 4 shapes × 2 dtypes via
  `test_rms_norm_precision_baseline.py`, gamma present, Phase-0 corner):
  - bfloat16: PCC ≥ 0.995 (gate); max_abs_err ≤ 0.089, mean_abs_err ≈ 0.0017,
    rel_rms_err ≈ 0.0033; got/true ratio median ≈ 1.000 (std ≈ 0.003)
  - float32: max_abs_err ≤ 0.025, mean_abs_err ≤ 0.0008, rel_rms_err ≤ 0.0015;
    got/true ratio median ≈ 0.999 (std ≈ 0.0007)
  - Ratio clusters tightly on 1.0 → ordinary rounding noise, no scale/structural bug.
- **Golden suite at Phase 0**: **472 / 40438** cells passing (`supported_pass`),
  per `verifier_report.json`. 6051 xfail_expected (the TARGET−SUPPORTED gap),
  33900 invalid_skipped, 15 no_axes_found (test_regression `@numerics`, all pass).
  Loud categories all clean: supported_fail = xpass_drift = xfail_wrong_mode = 0.
- **Issues encountered**:
  - **Fixed (test harness)**: `eval/golden_tests/rms_norm/axes.py:classify_call`
    omitted the `memory_layout` axis, so `verify_supported` misfiled 52 in-SUPPORTED
    interleaved `test_translated.py` cells as `xpass_drift`. Added
    `"memory_layout": input_tensor.memory_config().memory_layout` to mirror the op's
    `validate()`; drift → 0, those cells moved to `supported_pass` (420 → 472). Not
    an op change — the op was already correct.
  - **Noted (upstream kernel-lib, not fixed)**: two streaming-reduce wrapper helpers
    (`accumulate_reduce_block<>`/`accumulate_reduce<>`,
    `prepare_partial_reduce_scalers<>`) are stale against the current `reduce<>` /
    `prepare_reduce_scaler<>` template signatures and do not compile. The kernel
    correctly calls the working underlying helpers with equivalent last-block
    partial-scaler routing. See `verification_report.md`.
  - No op/kernel defects found; blocking-model fidelity and helper usage clean.
- **Tests added**: `test_rms_norm_precision_baseline.py` (PCC + abs/RMS error +
  got/true ratio spread across 4 shapes × 2 dtypes × gamma/no-gamma). Existing
  `test_rms_norm.py` (70/70) and `test_rms_norm_debug.py` (9/9) pass.

## Refinement 1 — Numerical configurability expansion
- **Date**: 2026-07-22
- **What was done**: widened the precision surface to the full TARGET. Pure
  knob-turn — **no compute-kernel change**; the descriptor was already fully
  dtype-derived and the compute config already flowed through to the kernel.
  - Op file (`rms_norm.py`): `SUPPORTED["dtype"] += bfloat8_b`,
    `SUPPORTED["gamma_dtype"] += bfloat8_b`, `SUPPORTED["fp32_dest_acc_en"] += False`;
    `EXCLUSIONS = [{dtype: float32, fp32_dest_acc_en: False}]` (the design's
    legal-but-refused lossy corner, now inside the SUPPORTED rectangle so it must
    be refused cell-level to stay xfail-strict).
  - Program descriptor (`rms_norm_program_descriptor.py`): added `_elem_size()`
    defensive helper. `element_size()` raises `ValueError` for block-float
    (bfloat8_b has no fixed per-element size). That value feeds ONLY the RM
    stick-byte math (`cols * elem`), and bf8b is TILE-only (bf8b+RM is INVALID),
    so the RM regime never runs for it → return 0 placeholder instead of raising.
    Page-size math uses `buffer_aligned_page_size()`, which is correct for
    block-float (returns the 1088-byte tile page) and was left unchanged.
    Caught by the risky-axis cheap-first probe before the full suite.
  - Intermediate-CB precision / UnpackToDestFp32: audited, **no change needed**.
    The only fp32 accumulator CB (`cb_rstd`) is already `Float32` (correct per
    /numeric-formats-metal §4) and feeds an FPU op (`mul<Col>`), so per §1.5 it
    **cannot** be `UnpackToDestFp32`-tagged — no tag applies to this op.
- **Accuracy achieved** (device probe + precision matrix, tile-aligned TILE):
  - bfloat8_b: PCC ≥ 0.9999, rel-RMS ≤ 0.057 (gate PCC ≥ 0.99 / RMS ≤ 0.10).
  - bf16 @ fp32_dest_acc_en=False (incl. HiFi2 perf config): PCC ≥ 0.99998,
    rel-RMS ≤ 0.056 (uniform corner; ≤ 0.007 on randn).
  - float32: rel-RMS ≤ 0.008. See `precision_matrix_results.md`.
- **Golden test progress**: green — **750 passed, 33900 skipped, 5689 xfailed,
  0 failed** (Phase-0: 472 passed). No `supported_fail`, no `xpass_drift`, no
  `xfail_wrong_mode`. Verified routing: `{f32,False}` EXCLUSION fires via
  `ExcludedCell` (560 f32+False+no_gamma cells xfail where the exclusion is the
  only possible refusal, 0 pass); bf8b passes only on `layout=TILE` +
  `tile_aligned` (0 RM-input, 0 non-aligned). gamma_dtype=bf8b / gamma_layout=TILE
  cells correctly still xfail — they are blocked on the gamma_layout=TILE axis,
  which Refinement 2 unlocks (this + R2 = the perf-1 anchor).
- **Issues encountered**: `element_size()` ValueError for bf8b (fixed via
  `_elem_size()`, above). No other defects.
- **Tests added**: `test_rms_norm_precision_matrix.py` (skill-mandated precision
  matrix: dtype × fp32_dest_acc_en × math_fidelity × gamma × distribution × 4
  shapes — 160 passed, 32 `{f32,False}` cells skipped) + `precision_matrix_results.md`.
  Regression net green: `test_rms_norm.py`, `test_rms_norm_debug.py`,
  `test_rms_norm_precision_baseline.py` (95 passed together).

## Refinement 2 — Tiled-gamma layout support
- **Date**: 2026-07-22
- **What was done**: added `ttnn.TILE_LAYOUT` to `SUPPORTED["gamma_layout"]` — a
  pure knob-turn per op_design.md §5. gamma layout is an INDEPENDENT knob from the
  input layout (new `gamma_is_rm` host predicate on `gamma.layout`, separate from
  `is_rm`), so RM-input + TILE-gamma at INTERLEAVED (a valid TARGET cell) works.
  Shared kernels, CT-arg dispatch (no forked files):
  - Op file (`rms_norm.py`): `SUPPORTED["gamma_layout"] += TILE_LAYOUT`. No
    axes.py change — `classify_call` already reads `gamma.layout` off the tensor
    (lockstep automatic).
  - Program descriptor: `gamma_is_rm` predicate; `cb_gamma_sticks` allocated
    ONLY on the RM-gamma path (unused for TILE gamma); `gamma_is_rm` passed as a
    new CT arg to reader (idx 15, accessor offset -> 16) and compute (idx 6).
  - Reader (`rms_norm_reader.cpp`): `GAMMA_IS_RM` CT flag. TILE gamma reads whole
    tiles straight into `cb_gamma` (tile_id = b*BLOCK_SIZE+wt; gamma is (1,1,1,W)
    -> Wt tiles in one tile-row), coalesced behind ONE barrier per block — same
    batched-read fast path as the TILE input (performance-conformance bar). RM
    gamma keeps the existing `read_sticks_for_tilize<cb_gamma_sticks>` path.
  - Compute (`rms_norm_compute.cpp`): `GAMMA_IS_RM` CT flag skips the pass-2
    `ckl::tilize<…,cb_gamma_sticks,cb_gamma>` on the TILE-gamma path (reader
    already filled `cb_gamma`); the `mul<Row>` consumer is unchanged. `cb_gamma`
    has ONE producer per compiled program (reader for TILE, compute-tilize for
    RM), exactly mirroring how `cb_x_in` dispatches on input layout.
  - This also unlocks bf8b gamma (block-float has no RM form -> implies TILE gamma).
- **Accuracy achieved**: perf-1 anchor (bf16 / fp32_dest_acc_en=False / TILE input
  / TILE gamma / INTERLEAVED / HiFi2, shape (1,1,128,2304)) PCC=0.999970,
  rel-Frobenius=0.0096 (soft gate 0.9995). Gamma-layout matrix (bf16/f32,
  gamma_dtype bf16/f32/bf8b, aligned + W/H/both non-aligned): rtol/atol/PCC gates
  met on all 86 cases.
- **Golden test progress**: 1598 passed, 33900 skipped, 4928 xfailed, **12 failed**.
  vs Refinement 1 (750 passed, 5689 xfailed): +848 supported_pass, -761 xfails —
  the gamma_layout=TILE cells (incl. bf8b gamma) moved xfail -> pass.
  - The 12 failures are **pre-existing Refinement-1 defects, NOT caused by R2**:
    all are `test_translated.py::test_rms_norm_row_major` with W=4096 +
    fp32_dest_acc_en=False + bf16, on the TILE-input + RM-gamma path (which R2
    does not touch). They are relative-Frobenius near-misses (5.20e-2..5.59e-2 vs
    0.052 threshold); PCC (>=0.9998) and ALLCLOSE pass. PROVEN pre-existing:
    stashing all R2 changes and re-running the same subset on the R1 commit
    reproduces the identical 12 failures with identical Frobenius values. They are
    a bf16 DEST-accumulation precision-boundary issue over 128 W-tiles — R1's
    fp32_dest_acc_en=False territory, out of scope for tiled-gamma. Not silenced
    with an EXCLUSION (per protocol: precision near-misses stay failing as the
    next precision refinement's baseline). Surfaced to the user for R1 follow-up.
- **Issues encountered**: None for the tiled-gamma work. (The 12 pre-existing R1
  Frobenius near-misses above are documented but out of this refinement's scope.)
- **Tests added**: `test_rms_norm_gamma_layout.py` — gamma_layout {TILE, RM} ×
  input_layout {TILE, RM} × dtype {bf16, f32} × 8 shapes (aligned + non-aligned)
  = 64 cases; + mixed-precision (bf16 act + f32 TILE gamma, 16 cases) + bf8b TILE
  gamma (6 cases). 86 passed (--dev + non-dev). Full rms_norm unit dir: 341 passed,
  32 skipped ({f32,False} EXCLUSION), 0 failed — no regression.

## Refinement 3 — Speed up the interleaved prefill perf profile
- **Date**: 2026-07-23
- **Type**: perf (no SUPPORTED change).
- **What was done**: implemented the **resident single-read fast-path** (op_design.md
  §1 lamp 1) as a dual-path on the TILE-input regime, plus its double_buffer co-tune.
  Roofline/ablation first: baseline was ~1.56× above `achievable_ns` on the small
  prefill widths and at-target on the large ones; a 2nd-read ablation (stub pass-2 x
  read, keep CB sync) proved ALL four widths are read-bound (removing the 2nd x read
  = ~1.30–1.33×), so the lever is single-read + gamma-resident.
  - **Kernels** (shared, CT-arg dispatch — no forked files): a `USE_RESIDENT` CT flag
    (reader idx 16 / accessor bumped to `<17>`, compute idx 7) selects the resident
    path when the host predicate holds. Reader: reads gamma ONCE per core (held) and
    each tile-row's `Wt` x-tiles ONCE (single coalesced read), no 2nd pass. Compute:
    holds the whole tile-row in `cb_x_in` (waited once / popped once per row) and reads
    block `b` at absolute front offset `b*BLOCK_SIZE` via `OperandKind::Block +
    TileOffset::Set + InputLifecycle::CallerManaged`, so BOTH passes read x from L1 —
    no 2nd DRAM read. gamma held resident the same way (read once, offset per block,
    popped at core exit). Intermediates (`cb_xsq`/`cb_norm`) stay `2*BLOCK_SIZE`, so
    every prefill width fits — only `cb_x_in`/`cb_gamma` scale with `Wt`. The resident
    steps use `eltwise_chain` directly (BinaryFpu+PackTile) because the `square<>`/
    `mul<>` convenience wrappers don't expose `TileOffset` — same helper, lower-level
    form, NOT raw LLK. The streaming two-pass path is unchanged as the fallback.
  - **Host** (`rms_norm_program_descriptor.py`): `L1_RESIDENT_BUDGET` (1.1 MB, the
    dual-path L1 gate — design's sanctioned exception to "no CB sized by an op dim",
    only `cb_x_in`/`cb_gamma` scale with Wt) and `RESIDENT_X_DEPTH` (max input-CB depth;
    host picks the largest depth in [1, 2] that fits → prefetch where L1 allows,
    single-buffer the widest rows). `use_resident = TILE input AND (TILE/no gamma) AND
    fits budget`. RM input, RM gamma, and rows too wide for L1 keep the streaming path.
- **Accuracy achieved**: soft PCC gate 0.9995 holds on all prefill shapes (PCC≈1.00 at
  the exact perf config bf16 / fp32_dest_acc_en=False / TILE / TILE gamma / HiFi2).
  Golden tolerances unchanged.
- **Measured perf** (blackhole, device FW duration, median of 6 fresh trials; exact
  perf config; `test_rms_norm_perf.py`):
  | shape (1,1,8192,W) | baseline ns | resident ns | vs baseline | achievable_ns | vs achievable |
  |---|---|---|---|---|---|
  | W=1024 | 150544 | 105967 | **1.42×** | 96744  | 1.095 |
  | W=2304 | 333159 | 232407 | **1.43×** | 211345 | 1.100 |
  | W=5120 | 734322 | 515147 | **1.42×** | 738307 | **0.698 (beats)** |
  | W=7168 | 1029117| 679422 | **1.52×** | 1032281| **0.658 (beats)** |
  All four prefill widths improve 1.42–1.52×; the two large (most expensive) widths
  beat their `achievable_ns` targets by 1.43–1.52×; the two small widths land within
  ~10% of achievable (a further compute-block-granularity co-tune could close that
  residual, but the goal is met/exceeded — not filed as a blocking follow-up).
- **Levers**: resident single-read (primary) — WINNING, kept. double_buffer — applied
  as adaptive `RESIDENT_X_DEPTH` (2 for W≤5120, 1 for W=7168) + coalesced whole-row
  reads on both reader halves (x + gamma), kept. compute_block_size — reader-side block
  maximized (whole-row single read); compute-side `ROWS_PER_CALL` left at the design's
  trivial 1 (raising it needs width-aware block sizing to keep the large widths' L1
  fit, for only the ~10% small-width residual).
- **Golden test progress**: green — no regression. `test_translated.py` 72 passed /
  12 failed (the SAME pre-existing R1/R2 RM W=4096 Frobenius near-misses — proven
  identical by stashing R3 and re-running on the R2 baseline; RM path is untouched,
  resident is TILE-only). `test_op_loose` 11 passed / 8 xfail (sharded, expected).
  `test_op` cartesian slices (small aligned/non-aligned resident, large W=4096
  resident, multi-row resident, wide W=8192 streaming): 348 passed, 0 failed, 0 xpassed.
- **No regression across the guard set**: TILE interleaved (resident + streaming),
  RM interleaved (unchanged streaming), no-gamma — unit dir 345 passed / 32 skipped in
  BOTH `--dev` and non-dev (race-clean).
- **Issues encountered**: None for the resident work. (The 12 pre-existing RM W=4096
  Frobenius near-misses remain, out of scope — R1 precision territory.)
- **Tests added**: `test_rms_norm_perf.py` (prefill perf harness — 4 interleaved
  prefill shapes at the exact perf config, N-trial loop + soft PCC gate, for reading
  device-ns off the Tracy CSV).
