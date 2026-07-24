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

## Refinement 4 — Cross-core W-split: WIDTH/BLOCK sharding + logical wide-interleaved split
- Date: 2026-07-23
- Type: scheme-change (partial — `[~]`).
- What was done: implemented the **native cross-core W-split** (op_design.md §1
  lamp 2, §5) for WIDTH/BLOCK-sharded inputs. Added `WIDTH_SHARDED` +
  `BLOCK_SHARDED` to `SUPPORTED["memory_layout"]`.
  - **Reuse**: existing `reduce` / `square` / `eltwise_chain` compute helpers
    (indexed resident access, same lower-level form as the R3 resident path);
    zero-copy sharded CB placement via `ttnn.cb_descriptor_from_sharded_tensor`;
    the interleaved row-parallel path is untouched (host branches on
    `memory_layout` → separate builder + kernel set, so zero regression risk).
  - **Added — 3 xcore kernels** (`rms_norm_xcore_{reader,writer,compute}.cpp`):
    * reader — prepares the 1/W reduce scaler (+partial tile when W non-aligned)
      and reads this core's gamma W-slice (TILE gamma) once into `cb_gamma` (held);
      x is NOT read (resident sharded `cb_x_in`).
    * compute — arms the resident sharded W-slice, pass-1 block-reduce over the
      slice → per-tile-row partial `Σx²·(1/W)`; the group MASTER folds the K
      gathered partials (raw-LLK add-tiles + `+eps, rsqrt` — the sanctioned
      tensix_all_reduce fold pattern, no helper reduces across N CB tiles; with
      the required `reconfig_data_format`/`pack_reconfig_data_format`) → `1/RMS`
      into a SEPARATE `cb_stat_handoff` (never `cb_stat_global`, §7 two-consumer
      trap); pass-2 `x·rstd·gamma` → zero-copy sharded `cb_out`.
    * writer — the cross-core transport: Pattern-A **all-unicast** gather →
      master fold → broadcast (topology-agnostic; the WIDTH auto-shard group is a
      ragged 64-core set on the 11-wide grid, so mcast can't address it — NoC-mcast
      / two-stage topology is the R6 lever). One fully-synchronous round per
      tile-row with 3 MONOTONE counter semaphores (SEM_GATHER / SEM_BCAST /
      SEM_DONE — no reset → no clobber race), fixed-base cross-core CBs, so
      `cb_gather` stays K tiles and no CB grows with the tile-row count.
  - **Host** (`_create_sharded_xcore_descriptor`): derives the group topology from
    the shard spec — WIDTH = all cores (one group, master = shard core 0), BLOCK =
    one group per grid row (master = row's x=0 core); per-core `vwt` (valid W-tiles)
    + `is_partial_holder` for non-aligned W; virtual coords for the NoC transport.
  - **Deferred to R4a** (EXCLUSIONS, xfail-strict): RM-input-sharded and
    RM-gamma-sharded (need a tilize/untilize on the cross-core kernels), and the
    logical wide-interleaved / decode W-split (parallelism prep for R6).
- Accuracy achieved: PCC ≥ 0.999996 on WIDTH (per_w_t=1/2, R=1/2, ragged auto-64,
  non-aligned W=50) and BLOCK (256×512, multi-group) — device probes; rtol/atol
  via golden TOLERANCES (bf16 PCC≥0.995). Fixed a per_w_t>1 pass-1 bug (block-reduce
  the whole vwt-tile slice in ONE `reduce` call, not per-tile accumulate).
- Golden test progress: `test_op_loose` **18 passed / 1 xfail** (HEIGHT_SHARDED = R5)
  — all WIDTH/BLOCK `_SHARDED` + `_perf_case` geometries pass. `test_op` sharded
  slice (`-k "WIDTH_SHARDED or BLOCK_SHARDED"`): **1160 passed, 0 failed, 1840
  xfailed** (the deferred RM EXCLUSIONS), 17160 INVALID skipped. No supported_fail,
  no xpass drift. Unit dir: 345 passed / 32 skipped (no regression).
  Full `test_op` cartesian: **2660 passed, 3760 xfailed, 33900 skipped, 0 failed**
  (up from R2's 1598 passed → +1062 sharded cells; golden green, no supported_fail).
- Issues encountered: (1) raw-LLK master fold tripped an unpacker-A src-format
  LLK_ASSERT (hang) — fixed with `reconfig_data_format`/`pack_reconfig_data_format`.
  (2) `noc_semaphore_inc` atomics unflushed at writer exit tripped the
  "atomics flushed" assert — fixed with a final `noc_async_atomic_barrier()`.
  (3) per_w_t>1 gave PCC 0.965/maxdiff 4 — fixed by the single block-reduce (above).
- Tests added: probes in `tests/ttnn/unit_tests/operations/rms_norm/probes/`
  (controlled + auto shard geometries, per_w_t/R sweeps, BLOCK multi-group).

## Refinement 4a — Cross-core W-split: RM tilize path + logical interleaved W-split
- Date: 2026-07-23
- Type: scheme-completion (partial — `[~]`).
- What was done: completed two of R4's three deferred corners, reusing the R4 xcore
  cross-core combine (reader/compute/writer kernels + topology) via CT-arg flags —
  no forked kernel files. Extracted a shared `_assemble_xcore_kernels()` so the
  physical-shard and logical-interleaved builders single-source the kernel arg
  layout + CB set.
  - **Part 1 — RM gamma + sharded (TILE input)**: `GAMMA_IS_RM` flag on the xcore
    reader + compute. Reader reads this core's gamma W-slice as row-major sticks (one
    tile-column per `read_sticks_for_tilize`, `cb_gamma_sticks`); compute
    `tilize<1, cb_gamma_sticks, cb_gamma>(vwt)` once before the pass-2 `·gamma`, held
    resident. gamma stays interleaved DRAM (only the input is sharded). Mirror of the
    interleaved RM-gamma knob-turn. Removed the two `{gamma_layout: ROW_MAJOR,
    memory_layout: *_SHARDED}` EXCLUSIONS.
  - **Part 3 — logical wide-interleaved / decode W-split**: new
    `_create_logical_xcore_descriptor` (one group of `K = min(Wt, num_cores)` cores
    splits W; each core handles all `R` tile-rows, `HT_LOCAL = R`) + a host trigger in
    the interleaved builder (`TILE input AND INTERLEAVED AND R < num_cores AND Wt > R`).
    Kernel flags: `X_FROM_DRAM` (reader reads its W/K slice tiles from interleaved DRAM
    into `cb_x_in` via `TensorAccessor`, `tile_id = t*Wt + w_tile_start + w`),
    `X_ZERO_COPY=0` (compute waits on the reader's push instead of self-arming the
    zero-copy shard), `OUT_TO_DRAM` (writer drains `cb_out` to DRAM per tile-row after
    the stat round). `cb_x_in`/`cb_out` allocated (not zero-copy) on the logical path.
    Wide (`W=16384/32768/12288`) + decode (`rows=32`) shapes now fill `K>1` cores
    instead of 1–2. Prefill (`R=256 ≥ num_cores`) is untouched — stays on the R3
    resident single-read path.
  - **Deferred to Refinement 4b — RM input + sharded** (structural, characterized at
    depth): RM WIDTH/BLOCK `auto_shard_config` uses a width granule of 8 (bf16) / 4
    (fp32), so a core's resident W-slice is `8·k` elements wide — NEVER a multiple of
    32 for any golden W (W=64→8el, W=1024→16el, W=4096→40el, W=8192→80el). Core
    boundaries straddle 32-wide tile-column boundaries, so the tile-based cross-core
    reduce (whole `per_w_t` tiles + one partial-holder) cannot consume the shard. Needs
    a per-core arbitrary-width tilize into `ceil(w/32)` padded tiles + a per-core
    partial scaler (EVERY core zeros its last tile's `[w%32, 32)`) + untilize back —
    filed as R4b with the exact levers. The two `{layout: ROW_MAJOR, memory_layout:
    *_SHARDED}` EXCLUSIONS stay xfail-strict (not silenced; they are R4b's baseline).
- Accuracy achieved: PCC ≥ 0.999996 on all probed geometries — RM-gamma sharded
  (WIDTH per_w_t=1/2, R=2, auto-64, BLOCK 256×512, non-aligned W=50); logical W-split
  decode (1024/2304), wide (16384/32768, PCC≈1.0), R=2 (12288, PCC≈1.0), no-gamma,
  RM-gamma, non-aligned W=4100. Golden `TOLERANCES` (bf16 PCC≥0.995) met throughout.
- Golden test progress: full `test_golden.py` — **3258 passed, 33900 skipped, 3181
  xfailed, 0 failed, 0 xpassed** (was 2660 passed / 3760 xfailed at R4) → +598
  RM-gamma sharded cells moved xfail→pass. Sharded slice 1740 passed / 1260 xfailed
  (was 1160/1840). Interleaved slice 1500 passed / 0 failed (non-dev race check).
  `test_op_loose` 18 passed / 1 xfail (HEIGHT=R5). The remaining 1260 sharded xfails
  are the RM-input EXCLUSIONS (R4b).
- Issues encountered: None for the two shipped parts. (RM-input sharded is a
  characterized structural gap, deferred to R4b — see above.)
- Tests added: probes 017 (RM-gamma sharded) and 018 (logical W-split decode/wide/
  R2/non-aligned/gamma variants) in `tests/ttnn/unit_tests/operations/rms_norm/probes/`.
  Unit dir unchanged: 345 passed / 32 skipped (no regression).

## Refinement 4b — RM-input sharded: per-core arbitrary-width tilize sub-scheme
- Date: 2026-07-23
- Type: scheme-completion (full — `[x]`).
- What was done: completed R4a's last deferred corner — **RM input + WIDTH/BLOCK
  sharded** — where a core's resident RM W-slice is `sw` elements wide (a multiple of
  the RM granule 8/4, generally sub-tile), so core boundaries straddle 32-wide tile
  columns. Landed as an `IS_RM` CT flag on the three R4 xcore kernels (no forked files),
  reusing the cross-core combine + transport unchanged.
  - **Reuse**: the R4 xcore cross-core combine (all-unicast gather → master fold
    (+eps, rsqrt) → broadcast `1/RMS`) + its 3-semaphore transport + topology
    derivation; the interleaved RM tilize/untilize dataflow pattern; per-core partial
    scaler (`prepare_reduce_scaler(..., partial_w)` + `ReducePartialScaler::last_tile_at(1)`).
  - **Added — phase-aligned sub-scheme (all in the shared `_assemble_xcore_kernels`
    + `IS_RM` kernel branches)**:
    * Host geometry: per core, `g0 = w_offset//32` (first global tile), `phase =
      w_offset%32` (leading offset), `valid_cols = clamp(W - w_offset, 0, sw)`,
      `valid_end = phase + valid_cols`, `vwt = ceil(valid_end/32)` (reduce tiles),
      `reduce_partial_w = valid_end%32`; the uniform tilize width `per_w_t =
      max_cores ceil((phase+sw)/32)`. `Ht_local = ceil(sh/32)` (RM shards H-non-align);
      per-core `valid_rows_total` clamps H tensor-padding out of the writeback (BLOCK
      splits H too). WIDTH = one all-core group; BLOCK = one group per grid row.
    * Reader (`IS_RM`): zeros `cb_x_sticks` once, then **loopback-repacks** the resident
      RM shard (`cb_shard_in` = `cb_descriptor_from_sharded_tensor` alias; local NoC
      loopback via `my_x/my_y[noc_index]`, no remote re-fetch) into tile-padded
      `cb_x_sticks` at column `phase`, reading only `valid_cols` per stick (shard/tensor
      padding never enters the reduce). gamma read at the tile-ALIGNED column
      `(g0+wt)*32` (a sub-tile DRAM byte offset faults — the first bug found). Emits
      full(tile0)+partial(tile1) scaler tiles.
    * Compute (`IS_RM`): per tile-row `tilize<per_w_t>(cb_x_sticks→cb_x_in)`; pass-1
      squares/reduces `vwt` tiles (leading `[0,phase)`=0 → 0 contribution, trailing
      masked by the per-core partial scaler); same master fold; pass-2 `x·rstd·gamma`
      (gamma on the `vwt` valid tiles, copy elsewhere) → `cb_out`; `untilize<per_w_t>`
      → `cb_out_sticks`.
    * Writer (`IS_RM`): after each tile-row's stat round, loopback-writes the valid
      columns `[phase, phase+valid_cols)` of `cb_out_sticks` into the resident RM
      output shard (`cb_shard_out` zero-copy alias).
  - **EXCLUSIONS**: both `{layout: ROW_MAJOR, memory_layout: WIDTH/BLOCK_SHARDED}`
    removed. The `{float32, fp32_dest_acc_en=False}` cell stays refused by its own
    EXCLUSION (still xfail-strict, including on this path).
- Accuracy achieved: PCC ≥ 0.99997, rtol/atol via golden `TOLERANCES` (bf16 PCC≥0.995,
  f32 PCC≥0.999) on 13 device probes — WIDTH + BLOCK, gamma/no-gamma, W-non-aligned
  (50), H-non-aligned (17), both, wide `per_w_t=2` (4096), HT_LOCAL=32 (2x4x128x512),
  3D/2D, and fp32. maxdiff ≤ 0.10 (wide bf16).
- Golden test progress: `test_op` sharded slice across ~20 shapes (all alignment/rank/
  wide/BLOCK buckets): **0 failed, 0 xpassed**; input-`ROW_MAJOR` WIDTH/BLOCK cells moved
  xfail→pass (the only remaining xfails are `{float32, fp32_dest_acc_en=False}`, the
  standing EXCLUSION). `test_op_loose` 18 passed / 1 xfail (HEIGHT_SHARDED = R5). Unit
  dir 345 passed / 32 skipped (no regression). Interleaved + TILE-sharded spot checks
  unregressed (0 failed).
- Issues encountered: (1) gamma DRAM read at a sub-tile element column offset
  (`w_offset*elem`, not 32-aligned) tripped an "invalid address alignment in NOC
  transaction" — fixed by the phase-alignment (gamma read at tile-aligned `(g0+wt)*32`,
  x's leading columns zeroed). No other defects.
- Tests added: probes 019 (shard-geometry survey) + 020–024 (RM WIDTH/BLOCK correctness
  sweeps) in `tests/ttnn/unit_tests/operations/rms_norm/probes/`.

## Refinement 5 — HEIGHT_SHARDED (local per-core reduction)
- Date: 2026-07-23
- Type: knob-turn (partial — `[~]`).
- What was done: added `ttnn.TensorMemoryLayout.HEIGHT_SHARDED` to
  `SUPPORTED["memory_layout"]` for **TILE input** (TILE gamma, RM gamma, no gamma) via a
  native zero-copy resident-shard path (op_design.md §1 lamp 3). HEIGHT sharding splits
  rows across cores, each core keeps FULL-W rows → the RMS reduce stays LOCAL per core, so
  this is a knob-turn on the interleaved row-parallel path (NOT the cross-core WIDTH/BLOCK
  scheme).
  - **Reuse**: `rms_norm_reader.cpp` + `rms_norm_compute.cpp` (the R3 resident indexed
    two-pass) unchanged except two `if constexpr (X_RESIDENT)` branches — the interleaved
    perf path is byte-identical at `X_RESIDENT=0`. No forked kernel files.
  - **Added — host** `_create_height_sharded_descriptor`: core assignment pinned by the
    shard grid (`cores = corerange_to_cores(shard_spec.grid)`, `num_rows = sh//32` per core =
    the whole resident shard as the per-core block); `cb_x_in`/`cb_out` backed ZERO-COPY via
    `ttnn.cb_descriptor_from_sharded_tensor` (no NoC read/write — the local shard is consumed
    in place, never re-read through a TensorAccessor); **no writer kernel** (compute packs the
    zero-copy `cb_out` in place; the whole-shard-sized CB fills exactly with no drain).
  - **Added — reader** `X_RESIDENT`: skips the x read (x resident); streams gamma per block
    per row (TILE tiles into `cb_gamma`, or RM sticks into `cb_gamma_sticks`).
  - **Added — compute** `X_RESIDENT`: self-arms the resident `cb_x_in` (whole shard pushed
    once), per-row `cb_wait_front/pop(Wt)` walks it so the block-offset indexing is identical
    to R3; gamma STREAMED per block (small `cb_gamma`, never sized by Wt) so HEIGHT fits any W
    — a full-W resident gamma (Wt tiles) would clash with the resident input+output shards in
    L1 for wide W (found via probe on W=8192); RM gamma tilizes per block before the `·gamma`.
  - **Deferred to R5a** (characterized at depth): RM INPUT + HEIGHT — the resident shard is
    full-W row-major sticks, needing a tilize-on-resident-shard (loopback repack → tilize) and
    untilize-back (loopback write, a new writer) — a local-reduction analog of the R4b RM
    sub-scheme. The one `{layout: ROW_MAJOR, memory_layout: HEIGHT_SHARDED}` EXCLUSION stays
    xfail-strict (not silenced). RM+HEIGHT+TILE-gamma is INVALID.
- Accuracy achieved: PCC ≥ 0.99967 (rtol/atol via golden `TOLERANCES`: bf16 PCC≥0.995,
  f32 PCC≥0.999, bf8b PCC≥0.99) on 20 probe geometries + 22 regression cases — tile-aligned,
  W-/H-/both-non-aligned, `per_h>1` (R>grid, 86 cores), wide W=8192 (single-core L1 pressure),
  fp32, bf8b, 2D/3D/4D, TILE/RM/no gamma incl. mixed precision (bf16 act + f32 gamma).
- Golden test progress: HEIGHT cartesian slice **852 passed / 630 xfailed / 0 failed /
  0 xpassed** (630 xfails = the deferred RM-input EXCLUSION). Loose `test_op_loose`
  **19 passed / 0 xfail** (the HEIGHT `(1,1,256,512)` loose case moved xfail→pass; was 18/1
  at R4b). Interleaved + WIDTH/BLOCK cartesian spot + loose unregressed (0 failed / 0 xpass).
  Unit dir **381 passed / 32 skipped** (`--dev` + non-dev, race-clean).
- Issues encountered: wide-W HEIGHT (e.g. `(1,1,32,8192)`, single core) tripped an L1
  static-CB-vs-shard clash when gamma was held resident (Wt=256 tiles ≈ 512KB on top of the
  resident input+output shards) — fixed by streaming gamma per block (small `cb_gamma`,
  op_design.md §7 "no CB sized by an op dimension"). No other defects.
- Tests added: `test_rms_norm_height_sharded.py` (22 cases: TILE/RM/no gamma × aligned +
  non-aligned + per_h>1 + wide + fp32 + bf8b + 2D/3D/4D + mixed precision). Probes 025-028.

## Refinement 5a — RM-input HEIGHT_SHARDED (tilize-on-resident-shard)
- Date: 2026-07-23
- Type: scheme-completion (full — `[x]`).
- What was done: completed R5's deferred corner — **RM input + HEIGHT_SHARDED** — where a
  core's resident row-shard is full-W ROW-MAJOR sticks (not tiles). Each core keeps FULL-W
  rows so the RMS reduce stays LOCAL per core (no cross-core combine, phase=0, only the
  standard W%32 mask). Removed the one `{layout: ROW_MAJOR, memory_layout: HEIGHT_SHARDED}`
  EXCLUSION.
  - **Reuse**: extended the interleaved `_create_height_sharded_descriptor` and the shared
    `rms_norm_{reader,compute,writer}.cpp` via CT-arg flags (no forked kernel files). The
    compute REUSES the existing streaming RM path UNCHANGED — the RM boundary is entirely
    in the reader (loopback source) and writer (loopback sink).
  - **Reader** (`IS_RM && X_RESIDENT`): loopback-repacks the resident RM row-shard sticks
    (`cb_shard_in` = zero-copy `cb_descriptor_from_sharded_tensor` alias → local NoC loopback
    via `my_x/my_y`, no DRAM/remote read — native local consumption) into tile-padded
    `cb_x_sticks`, reading only `origin_W` valid columns per stick (phase=0; W%32 pad stays 0
    from the up-front zeroing). Mirrors the interleaved streaming RM reader order exactly
    (2 passes; gamma interleaved per pass-2 block). gamma = RM sticks or none (RM+TILE-gamma
    INVALID).
  - **Compute**: streaming RM path unchanged (`use_resident=0`, `is_rm=1`): tilize
    `cb_x_sticks`→`cb_x_in` per block per pass, two-pass reduce/normalize, untilize
    `cb_out`→`cb_out_sticks` per block. Every CB stays per-block (`DEPTH*BLOCK_SIZE`, never
    Wt), so it fits any W/dtype.
  - **Writer** (NEW loopback branch, `IS_RM && X_RESIDENT`): loopback-writes the valid
    columns of `cb_out_sticks` into the resident RM output shard (`cb_shard_out` alias).
    Per-core `valid_rows_total` handles H non-alignment / the short last core; pad rows/cols
    (tensor padding) are not written.
  - **Host geometry**: RM HEIGHT shard = `sh` per-core rows (RM granule 1, generally not a
    mult of 32) × `sw` = W padded to 8/4. `Ht_local = ceil(sh/32)`; the collapsed NC*H row
    sequence is split `sh` rows/core row-major, so per-core `valid_rows_total =
    clamp(NC*H - i*sh, 0, sh)`. `SHARD_STICK_BYTES = buffer_aligned_page_size = sw*elem`.
  - **OOM fix (why streaming, not the note's resident single-tilize)**: a whole-tile-row
    resident `cb_x_in` (Wt fp32 tiles = 1 MB at W=8192) + intermediates + shards clashes L1
    (golden CB-clash on fp32 W=8192 — a *feasible* cell, since RM shards are small/per-row,
    unlike TILE HEIGHT which the harness SKIPs via oom_guard). The streaming re-tilize
    (2× LOCAL loopback, no DRAM) fits every cell; the single-tilize fast-path is folded into
    the R6 sharded-perf pass.
- Accuracy achieved: PCC ≥ 0.99996 (rtol/atol via golden `TOLERANCES`: bf16 PCC≥0.995,
  f32 PCC≥0.999) on 13 device probes + 20 regression cases — tile-aligned + W-/H-/both-non-
  aligned, last-core-short (per_h < shard), wide W=8192 (incl. the prior-OOM fp32 W=8192),
  fp32, 2D/3D/4D, RM gamma / no gamma / mixed precision. maxdiff ≤ 0.14 (bf16).
- Golden test progress: HEIGHT_SHARDED cartesian slice **1168 passed / 0 failed / 0 xpassed
  / 315 xfailed** (RM-input HEIGHT cells moved xfail→pass; the remaining 315 xfails are the
  standing `{float32, fp32_dest_acc_en=False}` EXCLUSION, which applies across all layouts).
  HEIGHT+ROW_MAJOR slice 599 passed / 0 failed. `test_op_loose` 19 passed / 0 failed.
  Interleaved + WIDTH/BLOCK spot 216 passed / 0 failed (writer CT-arg addition
  X_RESIDENT=0 unregressed). Unit dir 401 passed / 32 skipped (`--dev` + non-dev,
  race-clean).
- Issues encountered: initial resident single-tilize approach (per the verifier note) OOM'd
  on fp32 W=8192 RM HEIGHT (resident cb_x_in = 1 MB); pivoted to the streaming re-tilize
  path (reuses the streaming RM compute unchanged, fits any W/dtype). No other defects.
- Tests added: `test_rms_norm_rm_height_sharded.py` (20 cases: RM gamma / no gamma ×
  aligned + W-/H-/both-non-aligned + last-core-short + wide + fp32 + 2D/3D/4D + mixed
  precision). Probes 030-034.

## Refinement 6 — Speed up the decode + sharded perf profiles
- Date: 2026-07-23
- Type: perf (partial — `[~]`); no SUPPORTED change.
- What was done: landed ONE collective-topology lever (NoC-mcast broadcast) on the shared
  cross-core (`_assemble_xcore_kernels`) transport; the DECODE half of the goal was found
  ALREADY MET by R4a's logical W-split (measurement, no change needed). Roofline/ablation-first
  per the perf methodology.
  - **Measured baseline (blackhole_p150b, 110-core grid, median of 8 fresh trials, exact perf
    configs; `test_rms_norm_perf_r6.py`)**:
    * Decode interleaved `(1,1,32,W)` W∈{1024,2304,5120,7168}: 8.7/14.8/16.8/16.9µs vs
      achievable 9.1/17.0/75.8/104.3µs → **0.95×/0.87×/0.22×/0.16× — already BEATS** (R4a's
      logical W-split). No R6 change applied to decode (its groups are ragged → all-unicast).
    * WIDTH sharded 8×1/9×1/8×4/7×4: 1.41×/1.71×/1.95×/1.78× above achievable.
    * BLOCK sharded 8×8 (HT_LOCAL=32): **5.76×** above achievable — the worst offender.
  - **Lever — mcast broadcast (`rms_norm_xcore_writer.cpp` + `_assemble_xcore_kernels`)**: the
    group master now broadcasts the finalized `1/RMS` with ONE `noc_async_write_multicast` +
    ONE `noc_semaphore_set_multicast` (monotone: set = t+1) instead of K-1 serial unicast
    writes + K-1 sem-incs — a K-independent broadcast (the R4 writer comment's flagged R6
    lever). Host detects a GAP-FREE virtual rectangle per group (`group_rect`: bounding-box
    area == group size) and sets a `use_mcast` RT flag + rect corners; ragged groups keep the
    byte-identical all-unicast fallback. ONLY the broadcast leg's transport mechanism changed —
    the 3-monotone-counter protocol (SEM_GATHER/BCAST/DONE) + CB back-pressure are UNCHANGED,
    so the proven synchronization is intact. Writer RT layout: added `use_mcast`(11) +
    rect corners(12-15); `WORKER_COORDS_BASE` 11→16.
  - **Result**: wins **1.15×** on the one gap-free perf geometry (WIDTH 7×4 / K=28:
    11.20→9.73µs). The other 4 sharded targets did not move — characterized at depth:
    1. **Blackhole DRAM-column gap**: virtual-coord map skips x=8,9 (logical x=0..10 → virtual
       [1..7,10..13]), so any group spanning logical x=0..7 (8×1/9×1/8×4/8×8 targets) is not a
       gap-free virtual rectangle → strict rect check correctly keeps them on unicast (a naive
       mcast to the [1..10] box would fault on the DRAM columns). Only 7×4 (logical x=0..6)
       qualifies.
    2. **The broadcast is only ~14% of a round.** Ablation (`test_rms_norm_r6_ablation.py`,
       fixed per_w_t=1, gap-free 7-wide, mcast engaged): the per-tile-row synchronous
       gather→fold→broadcast round costs a FLAT **~3150 ns**, fully serialized × HT_LOCAL
       (K7·HT4→13.7µs / HT16→51.0µs / HT32→100.8µs, perfectly linear) — this dominates BLOCK's
       5.76×. Residual **~92 ns/core** gather fan-in K-cost remains (K7→4.3µs, K28→6.2µs).
- Accuracy achieved: soft PCC gate 0.9995 holds on all decode + sharded perf shapes
  (PCC ≥ 0.99998). Golden `TOLERANCES` unchanged.
- Golden test progress: green — `test_op_loose` **19/19**; `test_op` WIDTH/BLOCK cartesian
  slice **78 passed / 0 failed / 0 xpassed / 18 xfailed** (the standing `{f32,acc=False}`
  EXCLUSION). No supported_fail, no xpass drift.
- No regression across the guard set: unit dir 165 correctness (TILE/RM interleaved, gamma
  layouts, debug) + RM/HEIGHT/RM-HEIGHT sharded pass; R3 prefill perf 4/4. The mcast lever is
  byte-identical on every ragged/gap group, so nothing regressed. `--dev` clean (watcher, no
  hang, no assert).
- Levers: mcast broadcast — CORRECT, KEPT (wins 1.15× on gap-free 7×4; neutral/byte-identical
  elsewhere; can never regress via the strict `group_rect` gate). NOT reverted — it is the
  named collective-topology lever, correct, and safely parked. The dominant sharded bottleneck
  (per-tile-row synchronous round ~3150 ns × HT_LOCAL) needs a round-granularity restructure →
  filed as R6a (batch C tile-rows/round + gap-aware mcast + two-stage gather, in priority order).
- Issues encountered: initial hypothesis (K-scaling == broadcast) was confounded by per_w_t
  differences across the perf cases; the controlled ablation (fixed per_w_t, gap-free grids)
  corrected it — the per-tile-row synchronous round is the real cost, and the Blackhole
  DRAM-column gap limits where a raw mcast can engage.
- Tests added: `test_rms_norm_perf_r6.py` (decode interleaved + WIDTH/BLOCK sharded perf
  harness at the exact perf configs, N-trial loop + soft PCC gate) and
  `test_rms_norm_r6_ablation.py` (K-scaling + HT_LOCAL-scaling ablation on gap-free 7-wide
  grids — documents the per-round-sync bottleneck). Probes 035-037 (grid/topology survey).

## Refinement 6a — Sharded cross-core: batch the per-tile-row round + gap-aware mcast
- Date: 2026-07-23
- Type: perf (partial — `[~]`); no SUPPORTED change.
- What was done: landed the two headline R6a levers on the shared cross-core
  (`_assemble_xcore_kernels`) reader/compute/writer transport — no forked kernel files;
  both correct, race-clean (`--dev`), and non-regressing. Reused the R4/R6 xcore combine +
  topology; only the round granularity and the broadcast segmentation changed.
  - **Lever 1 — batch C tile-rows' stats per cross-core round (the headline; BLOCK).** One
    round now exchanges C tile-rows' partials: compute produces C local partials
    (`cb_stat_local` depth 2*C), the writer gathers `K*C` (fixed-base fan-in, layout
    slot = cc*K + slice_index), the master folds C rstds (`cb_stat_handoff` depth 2*C),
    broadcasts C (`cb_stat_global` depth C, exact for the fixed-base wrap), then pass-2
    covers the C tile-rows (rstd for row cc read from `cb_stat_global` tile cc via the Col
    operand's `TileOffset::Set` base). Sync rounds drop from `HT_LOCAL` to `ceil(HT_LOCAL/C)`;
    the 3 monotone counter semaphores now count per ROUND. C is a host tunable
    (`STAT_BATCH_ROWS=8`) bounded per program by an explicit L1 gate (`XCORE_STAT_L1_BUDGET`;
    `cb_gather` scales `K*C` fp32 tiles) — the sanctioned relaxation of R4's "cb_gather stays
    K" invariant, same exception class as R3's resident dual-path. **C=1 is byte-identical to
    R4**, and the host sets C>1 ONLY on the pure tiled resident-shard cross-core path — RM
    (`is_rm`) and logical (`out_to_dram`) keep C=1 (their per-tile-row output drain is
    unchanged), and single-tile-row groups (`Ht_local==1`, e.g. WIDTH shards) get C=1.
  - **Lever 2 — gap-aware mcast (unblocks the 8-wide WIDTH/BLOCK broadcast).** The 1/RMS
    broadcast now mcasts in up to TWO contiguous virtual-x runs (`_mcast_segments`): a
    gap-free group is one rectangle (R6), and a group straddling the Blackhole DRAM columns
    (virtual x=8,9) splits into `[xlo..7]` + `[10..xhi]`, each a full rectangle the master
    mcasts to separately (the sender is auto-excluded from the segment it sits in, so its
    `ndests` is members-1). Truly ragged groups (logical decode's multi-row-major set; a
    WIDTH auto-shard wrapping a partial row; any y-gap) get 0 segments and keep the
    byte-identical all-unicast fallback. Writer RT layout: `n_mcast_seg`(11) + seg0(12-16) +
    seg1(17-21); `WORKER_COORDS_BASE` 16->22. Compute/writer gained a `C_ROWS` CT arg
    (compute idx 9; writer idx 13, accessor bumped to `<14>`).
- Accuracy achieved: soft PCC gate 0.9995 holds on all sharded perf + batched-round
  correctness cases (PCC >= 0.99998; `test_rms_norm_perf_r6a.py` 14/14 incl. short-last-round
  edges HT=3/5/13 with C not dividing HT_LOCAL). Golden `TOLERANCES` unchanged.
- Measured perf (blackhole_p150b, 110-core grid, median of 8 fresh trials, exact perf config
  bf16 / fp32_dest_acc_en=False / TILE / TILE gamma / HiFi2):
  | case | grid/K | R6 baseline | R6a | vs achievable | speedup |
  |---|---|---|---|---|---|
  | BLOCK 8x8 (HT_LOCAL=32) | K=8 | 147729 | 119030 | 5.76x -> 4.64x | **1.24x** (lever 1 + small lever-2 mcast) |
  | WIDTH 8x4 | K=32 | 10204 | 8920 | 1.94x -> 1.69x | **1.14x** (lever 2 mcast; A/B: 8884 mcast vs 10173 unicast) |
  | WIDTH 7x4 | K=28 | 9870 | 9850 | 1.80x | flat (already mcast in R6; A/B confirms 9764 mcast vs 11173 unicast) |
  | WIDTH 8x1 | K=8 | 5781 | 5836 | 1.42x | flat (HT=1 -> lever 1 no-op; K=8 broadcast cheap either way) |
  | WIDTH 9x1 | K=9 | 7889 | 7883 | 1.71x | flat (HT=1; K=9) |
  - Lever-1 C-sweep on BLOCK 8x8 (mcast off): C=1 147849 / C=2 134810 / C=4 127731 / C=8
    125027 ns — diminishing returns (per-round cost scales with C: `round(C) ≈ 1029 + C*2121`
    ns, so batching amortizes only the fixed ~1us/round, and the ~2.1us/tile-row stat
    data-movement floor remains). Lever-2 A/B (mcast on vs forced unicast) confirms mcast
    wins on the large-K broadcasts (W8x4 1.15x, W7x4 1.14x, BLK8x8 1.05x) and is flat on the
    small-K HT=1 cases (broadcast to 7-8 workers is cheap).
- Levers: lever 1 (round-batching) — WINNING on BLOCK, kept (C=8, L1-gated). lever 2
  (gap-aware mcast) — WINNING on the large-K WIDTH broadcasts, kept (can never regress via the
  strict per-segment rectangle check). lever 3 (two-stage gather) — NOT implemented (deferred,
  see below).
- Deferred to R6b (characterized at depth): the dominant BLOCK-8x8 residual is NOT the
  broadcast (lever 2) nor the round count (lever 1 plateaus ~4.6x) — it is the **per-tile-row
  stat data-movement** (~2.1us/tile-row: K bloated 4KB fp32 stat tiles gathered per tile-row,
  each carrying only a 32-value column) PLUS the compute floor (~47us, unpipelined). The two
  next levers: (a) round/compute PIPELINING (overlap batch b+1's pass-1 with batch b's
  synchronous round — the round waits are where the master idles); (b) stat-tile COMPACTION
  (transfer only the meaningful reduce column, not the full 4KB tile — a ~32x gather/broadcast
  bandwidth cut). Lever 3 (two-stage gather) remains relevant only for the large-K 2D WIDTH
  gather fan-in (W8x4 K=32) and is the smallest — filed with the exact levers in R6b.
- Golden test progress: green — no regression. `test_op_loose` 19/19. `test_op` cartesian
  slices: BLOCK bfloat16 815 passed / 0 failed / 0 xpassed / 105 xfailed; WIDTH bfloat16 815
  passed / 0 failed; WIDTH+BLOCK float32 1210 passed / 0 failed / 630 xfailed (the standing
  `{f32,acc=False}` EXCLUSION). No supported_fail, no xpass drift.
- No regression across the guard set: unit dir **431 passed / 32 skipped** (`--dev` + non-dev,
  race-clean); RM/HEIGHT/RM-HEIGHT sharded (56 passed); RM WIDTH/BLOCK sharded (14 passed);
  interleaved TILE/RM + decode logical W-split (test_rms_norm 70/70, decode perf 4/4). The
  C=1 default keeps every non-batched sharded path byte-identical; the segmented mcast is
  byte-identical on ragged/gap-free groups.
- Issues encountered: None (both levers correct on first device run; the modest BLOCK win is
  the per-tile-row-stat-transport floor, characterized above and filed as R6b — not a defect).
- Tests added: `test_rms_norm_perf_r6a.py` (batched-round correctness across HT_LOCAL incl.
  short-last-round edges + gap-aware-mcast targets, gamma/no-gamma, soft PCC gate).

## Refinement 6b — Sharded cross-core: stat-tile compaction + round/compute pipelining (+ two-stage gather)
- Date: 2026-07-23
- Type: perf (partial — `[~]`); no SUPPORTED change.
- What was done: landed the two headline R6b levers on the shared cross-core
  (`_assemble_xcore_kernels`) reader/compute/writer transport — no forked kernel files; both
  numerically byte-identical, `--dev`-clean, and non-regressing. Reused the R4/R6/R6a xcore combine
  + segmented-mcast transport unchanged; only the gather transfer size and the compute issue order
  changed. Roofline/ablation-first per the perf methodology.
  - **Lever 1 — stat-tile compaction (the winner).** The cross-core stat is a REDUCE_ROW result whose
    ONLY consumed data is COLUMN 0 (the master fold is element-wise; pass-2 reads it via
    `BroadcastDim::Col`). In an fp32 tile column 0 lives entirely in faces 0 (rows 0-15) + 2
    (rows 16-31) — byte ranges [0,1024) and [2048,3072). The GATHER (K partials converging on the
    master, ~86% of the round per the R6a ablation) now moves ONLY those faces via `G_OFF0/G_LEN0/
    G_OFF1/G_LEN1` writer CT args (`_gather_runs(STAT_COMPACT_MODE)` on the host, single source of
    truth); the untransferred faces leave stale L1 that the fold sums-then-ignores, so the output is
    **numerically byte-identical** (PCC 1.001005 == baseline). The literal "32× column-only" transfer
    is infeasible (col 0 is strided across faces at 64 B stride; the NoC rewards contiguous runs, and
    every in-tree precedent — `tensix_all_reduce`, `combine_welford` — moves whole tiles), so the
    real compaction is the col-0 FACES. Gated to the pure tiled sharded cross-core path (WIDTH/BLOCK
    physical shards); RM / logical / decode keep the full transfer (mode 0, byte-identical to R6a).
    The broadcast leg (~14% of the round, one mcast of C contiguous tiles) stays full — per-tile
    mcast-splitting would cost more than the byte saving (R6a ablation: broadcast is off the critical
    path).
  - **Lever 2 — round/compute pipelining (the COMPLEMENTARY step to lever 1).** Compute issues batch
    r+1's pass-1 one round AHEAD (via a `PIPELINE_LOOKAHEAD` CT flag + a do_pass1/do_fold/do_pass2
    lambda refactor) so the local reduce overlaps the writer's synchronous cross-core round.
    `cb_stat_local` is already 2*C deep (two rounds in flight) and the writer/semaphore protocol +
    fixed-base `cb_gather`/`cb_stat_global` addressing are UNCHANGED — only the compute ISSUE order
    changes (both loop forms are numerically identical). Shipped ON on the multi-round tiled path;
    single-round WIDTH groups (num_rounds==1) degenerate byte-identically to the R6a sequential order.
  - **Levers COMPOSE (the key finding).** Pipelining is FLAT on its own (ablation mode0: pipe-on
    119036 == baseline 119027 — with the full 4 KB gather the round dwarfs the compute so overlap
    buys nothing), but WINS once lever 1 shrinks the gather (BLOCK 8x8 113810 pipe-off → 107995
    pipe-on, 1.05x on top of compaction). This is the "a batched reader only pays off once the writer
    batches too" pattern — the empirical measurement caught it after an initial (wrong) "pipelining is
    flat" read from the mode-0 ablation.
- Accuracy achieved: soft PCC gate 0.9995 holds on all decode + sharded perf shapes (PCC ≥ 0.99998;
  BLOCK 8x8 PCC 1.001005 == baseline, confirming byte-identical). Golden `TOLERANCES` unchanged.
- Measured perf (blackhole_p150b, 110-core grid, median of 8 fresh trials, exact perf config
  bf16 / fp32_dest_acc_en=False / TILE / TILE gamma / HiFi2; `test_rms_norm_perf_r6.py`):
  | case | grid/K | R6a | R6b | vs achievable | speedup |
  |---|---|---|---|---|---|
  | BLOCK 8x8 (HT_LOCAL=32) | K=8 | 119027 | 107995 | 4.64x → 4.21x | **1.10x** (compaction 1.046x + pipeline 1.054x) |
  | WIDTH 8x4 | K=32 | 8933 | 8132 | 1.70x → 1.54x | **1.10x** (compaction; pipeline no-op at HT=1) |
  | WIDTH 7x4 | K=28 | 9850 | 9182 | 1.68x | **1.07x** (compaction, gap-free) |
  | WIDTH 8x1 | K=8 | 5836 | 5619 | 1.37x | **1.04x** (compaction) |
  | WIDTH 9x1 | K=9 | 7883 | 7691 | 1.67x | **1.02x** (compaction) |
  | decode 32x1024 (logical) | — | 8700 | 8691 | 0.95x | flat (gated OFF, byte-identical) |
  - Mode ablation (BLOCK 8x8, gather transfer size): full 4 KB = 119036 / faces 0-2 3 KB 1-txn = 116219
    / faces 0&2 2 KB 2-txn = 108028 — the core-to-core L1 gather is BANDWIDTH-dominated (unlike the
    DRAM `tile_reorder` regime), so mode 2 (skip the unused middle face) wins despite the extra
    transaction. Shipped mode 2.
- Levers: lever 1 (stat compaction) — WINNING, kept (mode 2, tiled path). lever 2 (round/compute
  pipelining) — WINNING as lever 1's complementary step, kept ON (multi-round tiled path). lever 3
  (two-stage gather) — NOT implemented (deferred to R6c; it is the remaining dominant cost — the
  master-serialized fold+gather — and the complementary step that would let pipelining also hide the
  fold).
- Golden test progress: green — no regression. `test_op_loose` 19/19. `test_op` cartesian slice
  (1x1x2048x256, multi-round BLOCK) 165 passed / 0 failed / 0 xpassed / 39 xfailed (the standing
  `{f32,acc=False}` EXCLUSION). No supported_fail, no xpass drift.
- No regression across the guard set: unit dir **431 passed / 32 skipped** (`--dev` + non-dev,
  race-clean); R6a batched-round correctness **14/14** (`--dev` + non-dev); decode logical W-split
  byte-identical (gated off). Both levers gate to the tiled cross-core path, so every RM / logical /
  interleaved / HEIGHT cell is byte-identical.
- Issues encountered: initial mode-0 ablation read pipelining as "flat" (it is, in isolation); the
  correct picture — pipelining is lever 1's complementary step and wins once the gather is compacted —
  emerged only from measuring the SHIPPED combination (mode 2 + pipeline). The literal 32× column
  compaction was found infeasible (strided col-0 across tile faces; NoC rewards contiguous transfers)
  and replaced with the byte-identical col-0-FACES compaction. No defects.
- Tests added: none new (the R6a `test_rms_norm_perf_r6a.py` batched-round correctness + `test_rms_norm_perf_r6.py`
  perf harness cover the changed path); probes 038-042 (prior R6b attempt's grid/geometry surveys) preserved.

## Refinement 6c — Sharded cross-core: two-stage (hierarchical) gather
- Date: 2026-07-23
- Type: perf (partial — `[~]`); no SUPPORTED change.
- What was done: landed the R6c lever (two-stage/hierarchical gather) on the shared cross-core
  (`_assemble_xcore_kernels`) reader/compute/writer transport — no forked kernel files; correct
  (`--dev` + non-dev clean), gated to never regress. Reused the R4/R6/R6a/R6b xcore combine +
  segmented-mcast broadcast + stat-compaction unchanged; only the GATHER+FOLD topology changed for
  clean 2D rectangle groups. Roofline/ablation-first per the perf methodology.
  - **Lever — two-stage (hierarchical) gather.** For a 2D reduction rectangle (NX×NY, both > 1):
    stage 1 gathers each grid row's NX partials to its x0 row-leader (fan-in NX-1, parallel across
    the NY rows) which folds them to a row-partial (no finalize); stage 2 gathers the NY row-partials
    to the root (fan-in NY-1) which folds + finalizes (+eps, rsqrt) -> 1/RMS. Fan-in drops K-1 ->
    (NX-1)+(NY-1) and the fold is distributed off the single master (the `tensix_all_reduce`
    grid-two-stage pattern; master.md: grid two-stage wins when the grid is busy or the payload is
    tiny; on a 1-D group it collapses to the flat root reduce). New `SEM_GATHER2` stage-2 counter
    (id 3) + two small fp32 CBs `cb_rowpartial`(10)/`cb_gather2`(11); single round (C=1) so no
    round-to-round back-pressure. Compaction (col-0 faces) composes through both gather legs
    (numerically byte-identical to the flat fold); the broadcast leg (root -> all members) reuses
    the R6/R6a segmented mcast unchanged.
  - **Host gate (single source of truth: `XCORE_TWO_STAGE_GATHER=True`, `XCORE_TWO_STAGE_MIN_SAVING=13`).**
    Engages ONLY on the pure tiled WIDTH-sharded cross-core path when every group is a clean rectangle
    (NX·NY==K) with the master at the low corner, `Ht_local==1` (single round), AND fan-in saving
    `(NX-1)·(NY-1) >= 13`. Everything else keeps the byte-identical flat gather: 1-D groups (WIDTH n×1;
    BLOCK per-row groups — 1-D, where two-stage collapses to flat), ragged/logical/RM, and small-saving
    2D groups (the ny=2 regression, see below). C-batching + pipelining are off on this path (single
    round makes them no-ops).
- Accuracy achieved: soft PCC gate 0.9995 holds on all engaged + fallback shapes; PCC ≥ 0.99998 on the
  R6c correctness suite (aligned, NON-aligned-W partial-holder through BOTH stages, no-gamma, and the
  flat-fallback 1-D / small-saving cases). Golden `TOLERANCES` unchanged.
- Measured perf (blackhole_p150b, 110-core grid, median of 8 fresh trials, exact perf config
  bf16 / fp32_dest_acc_en=False / TILE / TILE gamma / HiFi2; `test_rms_norm_perf_r6.py`):
  | case | grid/K | R6b flat | R6c two-stage | vs achievable | speedup |
  |---|---|---|---|---|---|
  | WIDTH 8×4 (2D, HT=1) | K=32 | 8115 | 7615 | 1.54x → 1.45x | **1.066×** (WIN) |
  | WIDTH 7×4 (2D, HT=1) | K=28 | 9163 | 9136 | 1.67x | flat (per_w_t=8: compute dominates) |
  | WIDTH 8×1 (1-D) | K=8 | 5621 | 5619 | 1.37x | byte-identical (flat) |
  | WIDTH 9×1 (1-D) | K=9 | 7708 | 7749 | 1.68x | byte-identical (flat) |
  | BLOCK 8×8 (1-D rows) | K=8 | 107957 | 107990 | 4.21x | byte-identical (flat) |
  - Ablation K-sweep (gap-free 7-wide, per_w_t=1, `test_rms_norm_r6_ablation.py`) isolates the
    mechanism at constant per-core work: flat fan-in ~68 ns/core (K7 4096 → K28 5534). Two-stage
    flattens it — K28 5534 → 5253 (**1.05×**) — but REGRESSES small ny: K14 (ny=2) 4764 → 4884
    (−2.5%) because the extra stage-2 handshake+fold (~900 ns ≈ 13 transfers) exceeds a saving of only
    6 transfers; K21 (ny=3, saving 12) is flat; K28 (ny=4, saving 18) wins. Hence the `>= 13` gate.
- Levers: two-stage gather — CORRECT, KEPT, WINNING on the 2D WIDTH large-K target (8×4, 1.066×);
  gated so it can NEVER regress (1-D / small-saving fall back to byte-identical flat). Not reverted.
- Why `[~]` (not `[x]`): the lever wins on WIDTH 8×4 but is topologically a no-op on the dominant
  BLOCK 8×8 residual (4.21×) — BLOCK's groups are one grid ROW each (1-D), so two-stage collapses to
  the flat root reduce and cannot touch the master-serialized fold that R6a/R6b/R6c characterized as
  BLOCK's bottleneck. Removing that on a 1-D group needs an allgather (Pattern B), a different lever →
  filed as R6d. WIDTH 8×4 also still has headroom (1.45x). Real work landed + characterized at depth.
- Golden test progress: green — no regression. `test_op_loose` 19/19 (incl. the WIDTH 8×4/7×4 two-stage
  perf geometries + BLOCK 8×8). `test_op` cartesian WIDTH_SHARDED bf16 slice 665 passed / 0 failed /
  0 xpassed / 105 xfailed; BLOCK_SHARDED bf16 slice 665 passed / 0 failed. No supported_fail, no xpass drift.
- No regression across the guard set: unit dir **431 passed / 32 skipped** (`--dev` + non-dev, race-clean);
  R6c correctness 7/7 (`--dev` + non-dev); ablation 7/7 (`--dev`). The gate keeps every 1-D / ragged /
  logical / RM / interleaved / HEIGHT cell byte-identical (flat gather).
- Issues encountered: (1) the ny=2 regression (characterized above) — resolved by the fan-in-saving gate,
  not a defect. (2) test-authoring only: an early R6c correctness test passed the shard MemoryConfig as
  `compute_kernel_config` and used over-provisioned WIDTH shards (grid wider than the width, W ≠ K·sw);
  fixed to `W_padded == K·sw` with a partial last tile for the non-aligned cases. No kernel defect.
- Tests added: `test_rms_norm_perf_r6c.py` (two-stage correctness across 2D WIDTH geometries: aligned,
  NON-aligned-W partial-holder through both stages, per_w_t>1, no-gamma; plus 1-D + small-saving
  flat-fallback cases; soft PCC gate).

## Refinement 6d — Sharded cross-core: allgather (Pattern B) for the 1-D master-bottleneck residual
- Date: 2026-07-23
- Type: perf (partial — `[~]`); no SUPPORTED change, no kernel change.
- What was done: evaluated R6d's named lever — the **allgather (Pattern B)** for the 1-D
  BLOCK/WIDTH master-bottleneck residual — AT DEPTH on device (try-cheap-first, per the perf
  methodology) before committing a ~600-line restructure of the shared xcore transport. The
  authoritative measurement is the in-tree `tensix_all_reduce` collective bake-off run at the
  EXACT BLOCK 8x8 per-group topology (1x8 line, K=8, num_tiles=8 ~ the C=8 batched stat
  tiles/round), on THIS blackhole_p150b box, isolated AND grid-filling. It compares the current
  Pattern A (`reduce_root_mcast`), the R6d lever (Pattern B `mcast_all_gather` + `unicast_all_gather`),
  and the sibling `two_phase_reduce_mcast`.
- Measured baseline (blackhole_p150b, median of 8 fresh trials; `test_rms_norm_perf_r6.py`,
  UNCHANGED — R6d ships no kernel change): BLOCK 8x8 **108162 ns (4.22x)** — the dominant target;
  WIDTH 8x1 5628 (1.37x), 9x1 7759 (1.68x), 8x4 7613 (1.45x), 7x4 9148 (1.67x).
- Measured collective bake-off (median of 5 trials x 10 in-kernel collectives, num_tiles=8;
  `AR_GROUP_SHAPE=1,8 AR_NUM_TILES=8 AR_KERNEL_ITERS=10 ... test_tensix_all_reduce_device_perf`):
  | topology | 1x8 grid-filling (8 grp, 64 cores) | 1x8 isolated (1 grp) | vs Pattern A |
  |---|---|---|---|
  | `reduce_root_mcast` (Pattern A, CURRENT) | 3189.6 ns | 3182.2 ns | 1.00x |
  | `mcast_all_gather` (Pattern B — R6d LEVER) | 6134.5 ns | 6135.5 ns | **0.52x (~2x SLOWER)** |
  | `unicast_all_gather` (Pattern B all-to-all) | 11140.3 ns | — | 0.29x |
  | `two_phase_reduce_mcast` | 2291.3 ns | 2272.7 ns | **1.39-1.40x (WINNER)** |
- FINDING (characterized at depth): on a **1-D line the allgather is decisively inferior to the
  current Pattern A, isolated OR grid-filling (0.52x, regime-independent)**. The penalty is
  INTRINSIC and unfixable by any complementary step: the rotating mcast allgather is K serial
  mcast sub-rounds, and the all-to-all form multiplies the per-core receive traffic by K (every
  core receives all K partials instead of only the master). Eliminating the broadcast leg (~14%
  of a round, R6b) does not pay for doubling the gather. So — unlike a "batched reader needs its
  writer twin" incomplete lever — the allgather is a **topological dead-end for 1-D groups**;
  engaging it would REGRESS BLOCK 8x8 by ~11% (0.52x collective x 4 rounds). It is therefore NOT
  shipped (parked; shipping a measured-2x-slower scheme as gated dead code on the shared xcore
  kernels would only add regression surface for zero benefit).
- The measured winner is **`two_phase_reduce_mcast`** (tile-index workers gather+fold disjoint
  tile-rows -> root assembles + mcasts): reduced communication volume WITHOUT the allgather's
  traffic multiplication. It engages only when C>1 (BLOCK's batched tile-rows); WIDTH n x 1 are
  single-round (C=1) so it degenerates to root. Filed as **Refinement 6e** with the exact lever
  + numbers. NOTE (honest ceiling): the pure collective is only PART of BLOCK 8x8's in-context
  round (R6b: round ~68us dominated by per-tile-row stat data-movement + the ~47us compute
  floor), so even the 1.40x collective win projects to a modest op-level gain — R6e must confirm
  the in-op yield by ablation, and the deeper residual (per-tile-row stat movement + compute
  floor) may need a separate lever.
- Accuracy achieved: all three collective topologies are NUMERICALLY correct at the BLOCK 8x8
  1-D topology (the R6d verdict is perf, not correctness). Shipped op (Pattern A) unregressed on
  BLOCK 8x8: PCC=1.001005 (== the R6b/R6c byte-identical baseline). Golden `TOLERANCES` unchanged.
- Golden test progress: green — no change (no kernel/op change). `test_op_loose` **19/19**
  (incl. BLOCK 8x8 + WIDTH 8x4/7x4). No supported_fail, no xpass drift.
- No regression: unit dir unchanged (no kernel/op edit); `test_rms_norm_perf_r6.py` sharded 5/5;
  golden loose 19/19; the new ablation 7/7 (`--run-all`).
- Levers: allgather (Pattern B) — measured decisively inferior on the target 1-D topology
  (0.52x, regime-independent, intrinsic) -> NOT shipped/parked (a topological dead-end, not a
  correct-but-incomplete lever). Real next lever `two_phase_reduce_mcast` (1.40x collective) filed
  as R6e.
- Issues encountered: None (the finding is the deliverable — R6d's named lever is disproven on
  device, and the queue is redirected to the measured-winning sibling).
- Tests added: `test_rms_norm_r6d_ablation.py` (reproduces the collective-topology bake-off at the
  exact BLOCK 8x8 1-D topology via the `tensix_all_reduce` example API — correctness of Pattern A /
  Pattern B allgather / two_phase at isolated + grid-filling; + a BLOCK 8x8 Pattern-A regression
  guard; measured ns table + repro command in the docstring). 7/7 pass.

## Refinement 6e — Sharded cross-core: two-phase (tile-index) reduce-mcast for the 1-D master bottleneck
- Date: 2026-07-23
- Type: perf (partial — `[~]`); no SUPPORTED change.
- What was done: landed the R6d-named winner — **two-phase (tile-index) distributed fold** — on the
  shared cross-core (`_assemble_xcore_kernels`) reader/compute/writer transport via a `TWO_PHASE_FOLD`
  CT flag + `NUM_FOLDERS` (no forked kernel files); correct (`--dev` + non-dev clean), gated so every
  non-BLOCK-multi-round cell is byte-identical. Try-cheap-first per the verifier's "ablate the in-op
  yield before committing" directive.
  - **In-op ablation FIRST (blackhole_p150b, BLOCK 8×8, temporary kernel stubs, perf-only — revised the
    honest-ceiling):** baseline **108.0 µs**; stub gather DATA MOVEMENT → 106.4 µs (gather bytes only
    **1.6 µs / 1.5 %** — the honest-ceiling's "per-tile-row stat movement" is NOT the bottleneck); stub
    the master FOLD ARITHMETIC keeping the round → 78.3 µs (**master serial fold = 29.7 µs / 28 %, fully
    on the critical path**); stub the whole round (compute floor) → 67.1 µs. The dominant round cost is
    the master's serial K-1 add_tiles + rsqrt for all HT_LOCAL tile-rows on ONE core — exactly what
    two-phase distributes by tile-index (works on BLOCK's 1-D groups where R6c two-stage was a no-op).
    So the collective is ON the critical path (first "Done when" branch applies, not the R6f escape).
  - **Lever — two-phase distributed fold.** Every core pushes each batched tile-row's partial to that
    row's FOLDER (owner = row % `NUM_FOLDERS`, `NUM_FOLDERS = min(C, K)`); each folder gathers its owned
    rows' K partials, compute folds them (+eps, rsqrt) → `cb_rowpartial` (reused; two-stage off on this
    path), the folder scatters the finalized 1/RMS to the root's `cb_stat_global`, and the root assembles
    all C and mcasts them back (reusing the R6/R6a segmented mcast). Fully synchronous per round; both
    `cb_stat_global` back-pressures are FREE (the round barrier + the gather including every core, whose
    round-r push is gated behind its own pass-2 r-1 pop). SEM reuse only (SEM_GATHER cores→folder waits
    `(r+1)*K`, SEM_GATHER2 folders→root waits `(r+1)*NUM_FOLDERS`, SEM_BCAST root→members set `r+1`); no
    new semaphore, no new kernel file. `cb_gather` shrinks to `max_owned*K` (vs flat `K*C`). do_pass1 /
    do_pass2 are byte-identical to the flat path — only the fold is distributed off the master.
  - **Host gate (single source, `XCORE_TWO_PHASE_FOLD=True`):** engages ONLY on the pure tiled sharded
    cross-core path with C>1 multi-round batching (`Ht_local % C == 0` → uniform per-round ownership,
    monotone semaphore counts stay clean) AND every group mcast-able (`n_seg>0`). C=1 / WIDTH single-round
    / RM / logical / two-stage keep the byte-identical flat master fold. Two-phase and R6b pipelining are
    mutually exclusive (two-phase has its own fully-synchronous loop).
- Accuracy achieved: soft PCC gate 0.9995 holds; BLOCK 8×8 PCC **1.001005 == baseline** (byte-identical
  output — the fold is the same associative sum, just distributed). `test_rms_norm_r6e.py` 4/4 (`--dev` +
  non-dev): (2048,256)/(1024,512) grid 4×4 gamma+no-gamma (owned_max=2 multi-owned-row folds) and the
  (8192,1024) grid 8×8 perf-target topology (owned_max=1). Golden `TOLERANCES` unchanged.
- Measured perf (blackhole_p150b, 110-core grid, median of 8 fresh trials, exact perf config
  bf16 / fp32_dest_acc_en=False / TILE / TILE gamma / HiFi2; `test_rms_norm_perf_r6.py`):
  | case | grid/K | R6d/prev | R6e | vs achievable | speedup |
  |---|---|---|---|---|---|
  | BLOCK 8x8 (HT_LOCAL=32, C=8) | K=8 | 108023 | 86606 | 4.21x → 3.38x | **1.247x** (fold distributed across 8 folders) |
  | WIDTH 8x4 | K=32 | 7616 | 7616 | 1.45x | byte-identical (single-round C=1 → two-phase gated off; R6c two-stage still engages) |
  | WIDTH 7x4 | K=28 | 9160 | 9160 | 1.67x | byte-identical |
  | WIDTH 8x1 | K=8 | 5613 | 5613 | 1.37x | byte-identical |
  | WIDTH 9x1 | K=9 | 7776 | 7776 | 1.68x | byte-identical |
- Levers: two-phase distributed fold — CORRECT, KEPT, WINNING on the BLOCK multi-round target (1.247×);
  gated so single-round / non-BLOCK stay byte-identical (can never regress). Not reverted.
- Why `[~]` (not `[x]`): the named lever fully landed and won, but BLOCK 8×8 is still 3.38× above
  achievable — the ablation pins the residual to the **per-core compute floor (67.1 µs = 62 % of the op)**,
  a different lever family (compute: pass-2 mul fusion, compute/round overlap now that the fold is
  distributed, coarser DST batching). Filed as R6f with the exact next levers.
- Golden test progress: green — no regression. `test_op_loose` 19/19 (incl. BLOCK 8×8 two-phase geometry).
  `test_op` BLOCK bf16 tile-aligned slice 500 passed / 0 failed / 0 xpassed / 60 xfailed (standing
  exclusions). No supported_fail, no xpass drift.
- No regression across the guard set: core + RM/HEIGHT/RM-HEIGHT/WIDTH/BLOCK sharded + gamma-layout unit
  221 passed (`--dev` + non-dev); R6a batched-round + R6c two-stage correctness 21; precision baseline +
  R6d ablation 23. WIDTH perf byte-identical; two-phase gated to BLOCK multi-round only.
- Issues encountered: None — the semaphore choreography (gather→folder, scatter→root, assemble+mcast) was
  correct on the first device run; both `cb_stat_global` back-pressures proved free under the fully-sync
  round barrier (no extra semaphore needed). The ablation-first discipline (measure the in-op yield before
  the ~250-line restructure) confirmed the lever was on the critical path and projected the 1.247× win.
- Tests added: `test_rms_norm_r6e.py` (two-phase correctness across multi-round BLOCK geometries: owned_max
  1 and 2, gamma/no-gamma, the 8×8 perf topology; soft PCC gate). The in-op ablation was run via temporary
  `ABLATE_STUB_*` kernel macros (reverted after measurement; numbers recorded above + in breadcrumbs).

## Refinement 6f (debug) — Fix R6e gate violation: two-phase non-uniform-ownership nan
- Date: 2026-07-23
- What was done: fixed the hard completion-gate violation from Refinement 6e
  (`test_rms_norm_r6_ablation.py::test_ablate[K28_HT16_512x224]` → `AssertionError: nan`).
  - **Root cause**: the R6e two-phase (tile-index) distributed fold writes each folder's stat
    partials into `cb_gather` using `get_write_ptr(cb_gather)` as a REMOTE-BASE PROXY. That proxy
    is valid only when every core's write pointer wraps back to the same base each round (the
    fixed-base convention documented in the xcore writer header). Each folder advanced the fifo by
    `owned*K`, but `NUM_FOLDERS = min(C, K)` need not divide `C`. For `K28_HT16` (BLOCK 7×4:
    per-group K=7, C=8, folders=7) folder 0 owned 2 rows and folders 1-6 owned 1, so their pushes
    (14 vs 7) did not wrap the depth-14 `cb_gather` fifo to base uniformly → the gather-push landed
    at wrong/out-of-bounds slots → corrupted Σx² → `rsqrt` of garbage → nan. R6e passed earlier only
    because every tested topology (8×8 K=8/C=8; 4×4 K=4/C=8) has UNIFORM ownership.
  - **Fix (host-only, `rms_norm_program_descriptor.py`)**: constrain `NUM_FOLDERS` to the LARGEST
    divisor of `C` that is `<= min(C, K)`, and engage two-phase only when that divisor is `>= 2`.
    This guarantees uniform ownership — every folder owns exactly `C/NUM_FOLDERS` rows — so each
    folder advances `cb_gather` by `owned*K == depth` per round, the fifo always wraps to base, and
    the fixed-base `get_write_ptr` proxy is valid. For `K28_HT16` this selects `NUM_FOLDERS=4`
    (uniform, owned=2) instead of the broken 7. On the divisible cases the choice is identical to the
    old `min(C,K)`: the 8×8 perf target keeps `NUM_FOLDERS=8`, so the two-phase 1.247× BLOCK win is
    unchanged; every WIDTH single-round group (C=1) is unaffected (two-phase already gated off).
  - Reused: the R4→R6e xcore kernels + transport are UNCHANGED — the fix is a single host-side
    `num_folders` selection. No kernel edits, no forked files.
- Accuracy achieved: PCC ≥ 0.99999 on the previously-nan K=7 geometries (`K28_HT16` 512×224,
  `K28_HT32` 1024×224 BLOCK 7×4); PCC 1.001005 (byte-identical) on the divisible 8×8 target.
- Golden test progress: full `test_golden.py` ran to completion against the fixed tree — **5056 passed
  / 33918 skipped / 1365 xfailed / 0 failed / 0 xpassed** (238.14s, no hang). Golden
  regression+translated 99 passed.
- Issues encountered: None beyond the diagnosed root cause; the divisor fix passed on the first run
  (`--dev` + non-dev).
- Tests added: none new — `test_rms_norm_r6_ablation.py` already exercises the exact K∤C class
  (`K28_HT16` C=8, `K28_HT32` C=8, BLOCK 7-wide) the gate caught; both now pass (7/7, `--dev`).
  Full guard: unit dir 452 passed / 32 skipped, `test_rms_norm_r6e.py` 4/4.

## Refinement 6f — Sharded cross-core: shrink the per-core compute floor (pass-2 fusion + compute/round overlap)
- Date: 2026-07-23
- What was done: perf refinement on the sharded cross-core compute floor. Two of the three named
  levers shipped on the shared R4→R6e xcore kernels via CT flags (no forked files), both numerically
  byte-identical (PCC 1.001005 == baseline) and correct (`--dev` + non-dev clean):
  * **Lever 2 — overlap the distributed round under compute (the complementary step to R6e).** R6e
    shipped the two-phase loop fully synchronous because R6b measured pipelining flat while the
    master's serial fold sat ON the round's critical path. R6e distributed that fold off the master
    (folders own disjoint tile-rows), so re-enabling the R6b `PIPELINE_LOOKAHEAD` lookahead on the
    two-phase loop now WINS — the compute two-phase branch honors the flag (issue batch r+1's pass 1
    before batch r's fold/pass 2), and the host `pipeline_lookahead` gate drops its `and (not two_phase)`
    clause. `cb_stat_local` is already 2*C deep and the writer is serial across rounds (unchanged), so
    the reorder is byte-identical. BLOCK 8×8 86619 → 80755 ns (1.073×); single-round WIDTH degenerates
    byte-identically.
  * **Lever 3 (partial) — coarser DST batching on pass-1's square.** `do_pass1` now squares the whole
    vwt-tile block in ONE `eltwise_chain` (Block-walk from the tile-row's resident base — the R3
    resident-square pattern) instead of vwt one-tile chains, amortizing the per-call init/reconfig.
    `cb_xsq` is already 2*per_w_t so no CB resize; byte-identical. BLOCK 8×8 80755 → 71669 ns (1.127×
    on top of lever 2); helps the single-round WIDTH groups lever 2 could not (8×1 1.04×, 9×1 1.08×,
    8×4 1.04×, 7×4 1.07×).
  * **Lever 1 (pass-2 mul fusion, the named "biggest") — characterized as a helper-surface dead-end
    for the gamma target, NOT shipped.** The literal fusion is not expressible: a second `BinaryFpu`
    cannot consume the running DEST, and the only DEST-consuming binary element (`DestReuseBinary` →
    `binary_dest_reuse_tiles`) has no broadcast — while `·gamma` needs `BroadcastDim::Row`. Reaching a
    ROW-bcast dest-reuse needs net-new custom LLK (forks off the mandated helper surface), and the
    in-tree `examples/compute_fusion` MEASURED the FPU-consumes-DEST form at 0.82× on Wormhole (the
    pack-to-L1 round-trip is 1.22× faster) — so even raw-LLK / gamma-pre-expansion is measured-inferior
    on this FPU-consumer shape. Filed R6g to try-cheap-first on Blackhole before committing.
- Accuracy achieved: PCC=1.001005 (byte-identical to the R6e baseline) on BLOCK 8×8; soft PCC ≥ 0.9995
  on all sharded perf shapes; golden `test_op_loose` 19/19; golden `test_op` BLOCK/WIDTH `1x1x2048x256`
  multi-round slice 39 passed / 0 failed / 0 xpassed / 9 xfailed (standing `{f32,acc=False}` EXCLUSION).
- Measured speedups (blackhole_p150b, median of 8 fresh trials, exact perf config): BLOCK 8×8
  86619 → 71669 ns (**1.209×**, 3.38× → 2.80× above achievable); WIDTH 8×1 5640→5404 (1.04×),
  9×1 7725→7164 (1.08×), 8×4 7590→7320 (1.04×), 7×4 9153→8571 (1.07×).
- Golden test progress: no cell change (perf refinement, no SUPPORTED change); no regression —
  loose 19/19, unit dir 449 passed / 32 skipped (`--dev` + non-dev), `test_rms_norm_r6e.py` 4/4 +
  `test_rms_norm_r6_ablation.py` 7/7 (`--dev` + non-dev).
- Issues encountered: Lever 1 (the named "biggest") is blocked at the kernel-lib helper surface (no
  broadcast dest-reuse) and measured-inferior even via raw LLK (`compute_fusion` 0.82× on Wormhole);
  characterized at depth and deferred to R6g. Pass-2 x·rstd batching deferred to R6g because it needs a
  `cb_norm` resize that broadens L1 pressure across every cross-core program (wide logical/decode/RM
  per_w_t) — belongs behind a per-path L1 gate, not this focused diff.
- Tests added: none new — reused `test_rms_norm_perf_r6.py` (device-ns), `test_rms_norm_r6e.py`
  (two-phase correctness), `test_rms_norm_r6_ablation.py` (K∤C gate). Filed follow-up Refinement 6g
  (pass-2 batching + gamma-fusion residual).

## Refinement 6g — Sharded cross-core: pass-2 batching + gamma-fusion residual on the compute floor
- Date: 2026-07-23
- Type: perf (full — `[x]`); no SUPPORTED change.
- What was done: closed more of the R6f BLOCK-8×8 compute-floor residual with the two named
  pass-2 levers, both on the shared R4→R6f xcore kernels via a CT flag (no forked files). Lever 1
  (pass-2 batching) shipped and WINS; lever 2 (gamma fusion) measured a Blackhole dead-end and
  recorded (not shipped) — the R6g "Done when" OR is satisfied on both counts. Roofline/ablation
  lineage from R6e/R6f pinned the residual to the per-core compute floor; R6f lever 3 batched
  pass-1's square, this batches pass 2.
  - **Lever 1 — batch pass-2's x·rstd (+ tile-aligned ·gamma) per tile-row (WINNER).** `do_pass2`
    in `rms_norm_xcore_compute.cpp` now issues x·rstd as ONE `eltwise_chain` over the tile-row's
    `per_w_t` W-tiles (Block-walk x from the resident base `t*per_w_t`; Col-broadcast rstd tile
    `cc` — Ht=1 so the Col index stays `cc` across the walk) instead of `per_w_t` one-tile chains,
    and batches ·gamma over the `vwt` valid tiles in one chain (front-walked `cb_norm` × Row gamma),
    leaving only the trailing pad tail (w ≥ vwt) to per-tile copy. Mirrors the proven interleaved
    resident pass 2 (`rms_norm_compute.cpp` block_shape) and amortizes the fixed per-call chain
    init/reconfig over the block — the R6f lever-3 square pattern applied to pass 2. Numerically
    byte-identical: same x·rstd per tile, same ·gamma per valid tile, same copy per pad tile.
    - `cb_norm` deepened 2 → `2*per_w_t` (the batch buffer) — the SAME depth `cb_xsq` has carried
      unconditionally since R4 (the pass-1 square block buffer), so it is proven-safe on every
      cross-core path (RM/logical/decode/physical); folded into the `XCORE_STAT_L1_BUDGET` C gate
      via a single-source `norm_depth`. `XCORE_PASS2_BATCH` module knob + `PASS2_BATCH` compute CT
      arg (idx 16) gated to the non-RM xcore path; the RM path keeps its own per-tile pass-2 loop +
      `cb_norm=2`; `PASS2_BATCH=0` degenerates byte-identically to R4/R6f.
  - **Lever 2 — pass-2 mul fusion via gamma pre-expansion (measured Blackhole dead-end, NOT
    shipped).** The only expressible fusion of `(x·rstd)·gamma` is `BinaryFpu` x·rstd →
    `DestReuseBinary<cb_gamma_full, Mul, DEST_TO_SRCA>` (a second FPU op consuming DEST). Ran the
    in-tree `examples/compute_fusion --scenario fpu_sfpu` (which A/Bs exactly this dstreuse-vs-
    unfused combine) on Blackhole p150b: dstreuse is **0.94–1.00× vs unfused** across tiles 4/16/64
    (0.94–0.95× at 64) — never beats the pack-to-L1 round-trip (the FPU wants operands in source
    registers; DEST→srca costs more than the pack+unpack it replaces). Confirms the Wormhole 0.82×
    on Blackhole; the lever would also ADD a per-core gamma `[1,W]→[32,W]` pre-expansion + double
    gamma's L1. Hardware dead-end (like R6d's allgather) — unshipped, measurement recorded.
- Measured perf (blackhole_p150b, median of 8 fresh trials, exact perf config bf16 /
  fp32_dest_acc_en=False / TILE / TILE gamma / HiFi2; `test_rms_norm_perf_r6.py`):
  | shape | R6f ns | R6g ns | speedup | achievable | vs achievable |
  |---|---|---|---|---|---|
  | BLOCK 8×8 (1,1,8192,1024) | 71653 | **54060** | **1.325×** | 25640 | 2.79× → **2.11×** |
  | WIDTH 8×1 (1,1,32,1024)  | 5422  | 4924  | 1.10×  | 4110 | 1.32× → 1.20× |
  | WIDTH 9×1 (1,1,32,2304)  | 7212  | 5841  | 1.235× | 4617 | 1.56× → 1.27× |
  | WIDTH 8×4 (1,1,32,5120)  | 7341  | 6626  | 1.108× | 5267 | 1.39× → 1.26× |
  | WIDTH 7×4 (1,1,32,7168)  | 8631  | 7341  | 1.176× | 5481 | 1.57× → 1.34× |
- Accuracy achieved: soft PCC gate 0.9995 holds on all sharded + decode perf shapes (PCC ≥ 0.9995);
  golden `TOLERANCES` unchanged; lever 1 is numerically byte-identical (same ops, batched issue).
- Golden test progress: green — `test_op_loose` 19/19; `test_op` WIDTH/BLOCK cartesian slice
  **2370 passed / 0 failed / 0 xpassed / 630 xfailed** (the standing `{f32,acc=False}` EXCLUSION).
  No supported_fail, no xpass drift.
- No regression across the guard set: unit dir **449 passed / 32 skipped** (`--dev` + non-dev);
  decode interleaved logical W-split (also batched) correct; interleaved/HEIGHT/RM cells
  byte-identical (`rms_norm_compute.cpp` untouched — lever 1 only touches the xcore `do_pass2`).
- Levers: lever 1 (pass-2 batching) — WINNING, kept (`XCORE_PASS2_BATCH=True`, live knob).
  lever 2 (gamma fusion) — measured Blackhole dead-end (compute_fusion dstreuse 0.94–1.00×),
  NOT shipped, measurement recorded (not parked as dead code).
- Issues encountered: None. Residual (BLOCK 8×8 still 2.11× above achievable) is the cross-core
  round + remaining compute floor — a new lever family outside R6g's named pass-2 scope.
- Tests added: none new — reused `test_rms_norm_perf_r6.py` (device-ns + soft PCC) and the
  in-tree `examples/compute_fusion` bake-off (lever-2 Blackhole measurement).

## Perf 1 — Sharded cross-core compute floor: fused pass-1 reduce + pass-2 reconfig-skip
- Date: 2026-07-24
- Type: perf tournament (no SUPPORTED change; `verify_supported` categories unchanged).
- Focus shape (perf-flagged loose case, its EXACT config): **BLOCK_SHARDED (1,1,8192,1024),
  8×8 grid, bf16 input + bf16 TILE gamma, HiFi2, fp32_dest_acc_en=False** — achievable_ns=25640;
  it was the worst offender (2.10× above achievable), the mandatory primary target.

### Measured breakdown (Step 1 — instrumented, ablated, roofline-gated)
- Added PERMANENT `MaybeDeviceZoneScope` instrumentation to the three xcore kernels
  (`rms_norm_xcore_{compute,reader,writer}.cpp`): zones `xc_pass1`, `xc_fold`, `xc_pass2`,
  `xc_wr_round`, `xc_rd_gamma`, `xc_rd_x`, `xc_rm_tilize`, `xc_rm_pass2`. Free when the profiler is
  off; never remove. (The interleaved/HEIGHT path uses `rms_norm_compute.cpp` — instrumented by a
  prior round's zones where present; untouched here.)
- Per-stage device-ns on the focus shape (blackhole_p150b, median of trials 2–8, `--profile`):
  per-core geometry HT_LOCAL=32 tile-rows, PER_W_T=4 W-tiles, K=8, C=8, 4 rounds
  (TWO_PHASE_FOLD + PIPELINE_LOOKAHEAD + PASS2_BATCH). Whole-op = **53822 ns**; per critical core:
  | stage | ns | % | note |
  |---|---|---|---|
  | pass2 (x·rstd + ·gamma) | ~34,000 | 63% | DOMINANT — 2 mul-chains/tile-row × 32 rows |
  | pass1 (square + reduce) | ~18,000 | 34% | square(4)+matmul-reduce × 32 rows |
  | fold | ~1–5,000 | ~3% | distributed (two-phase); tiny |
  | writer round (transport) | 48,584 (wall) | — | HIDDEN under compute (< compute path); mostly idle-waiting on compute |
- **Verdict: COMPUTE-BOUND.** Input+output are zero-copy sharded → NO DRAM; the roofline is the
  compute/per-call-overhead floor, not data-movement. The cross-core round is not on the critical
  path (the writer's 48.6µs is dominated by waiting on compute — R6e measured the real gather at
  ~1.6µs), so cross-core/mcast/allgather levers (already exhausted R6–R6e) are NOT floated. Ranked
  targets: pass2 (63%) then pass1 (34%).

### Portfolio floated (4 ideas, deliberate overlap; sized to the T2 compute headroom)
1. `pass2_batch_rows` — batch pass-2 chains across the C=8 tile-rows/round (compute_block_size).
2. `pass1_reduce_restructure` — replace square+matmul-reduce with accumulate+finalize (reduce_block/
   row_reduce_accumulate).
3. `pass1_batch_rows` — batch pass-1 square+reduce across C rows (compute_block_size). Overlaps #2.
4. `pass2_fuse_and_reconfig` — re-check x·rstd·gamma fusion (R6g dead-end) + skip redundant pass-2
   data-format reconfig (compute_block_size 2nd lever). Overlaps #1.
Each fanned out to a `blocking-perf-part-optimizer` building an isolated single-core micro-bench in
`ttnn/ttnn/operations/rms_norm/perf_experiments/<slug>/`.

### Per-idea verdicts (isolated single-core Blackhole bench; precision contract held, PCC ~0.9999)
| idea | best variant | isolated speedup | L1 | raw-LLK | disposition |
|---|---|---|---|---|---|
| pass1_reduce_restructure | **fused_fpu** (square→FPU DEST-accumulate→1 summed tile→reduce-1-tile) | **1.51× on pass 1** (@vwt4) | none | no (pure helper) | **GRADUATED** |
| pass2_fuse_and_reconfig | **reconfig_skip** (skip constant srcA/pack reconfig in pass 2) | **1.44× on pass 2** (byte-identical) | none | no | **GRADUATED** |
| pass2_batch_rows | batch_both (2 chains/round) | 1.41× on pass 2 | +48 KB (cb_norm) | no | superseded by reconfig_skip (composes to 1.47×; +2% for +48 KB) → **deferred to Perf 2** |
| pass1_batch_rows | batch square + blocked reduce | 1.39× on pass 1 | +cb_xsq | no | superseded by fused_fpu (1.51×) |
| pass2_fuse (DEST-reuse fusion) | — | — | — | — | **NULL / dead-end** — re-confirmed: `(x·rstd)·gamma` DEST-reuse is not expressible with a broadcast (no BroadcastDim on DestReuseBinary) and R6g measured the non-bcast form 0.94–1.00× (never beats pack-to-L1). Not shipped. |

### Graduated (predicate-guarded fast paths, widened to their measured winning domain)
Both live in `rms_norm_xcore_compute.cpp` (the shared cross-core compute), routed via two new CT
flags from a single host source of truth in `_assemble_xcore_kernels`. Everything outside the
predicate keeps the byte-identical R6f/R6g path (no regression by construction). Both are pure
`kernel_lib` helpers (no raw-LLK bypass, so the verifier's helper-usage pass is clean).
- **PASS1_FUSED (compute idx 17).** `BinaryFpu<Mul, DestAccumulation::Enabled>` collapses the vwt
  x-tiles into ONE summed x² tile in DEST (packed once to `cb_xsq`), then a SINGLE REDUCE_ROW over
  that 1 tile yields the per-row Σx²·(1/W) — vs the streaming square(vwt)+matmul-reduce(vwt). Kills
  the vwt-wide `cb_xsq` round-trip and runs the reduce datapath once. Predicate:
  `not is_rm and not has_partial_w and 2 <= per_w_t <= 4`. Lower bound 2 (per_w_t=1 is measured
  flat); upper bound 4 (bounds the bf16-DEST accumulation error above the 0.9995 soft-PCC gate under
  fp32_dest_acc_en=False — PCC 0.99984 @vwt4, ~0.9992 @vwt8 → excluded; fp32-DEST paths are exact).
- **PASS2_RECONFIG_SKIP (compute idx 18).** Establish pass-2's constant srcA/pack formats ONCE at
  pass-2 entry (`reconfig_data_format(cb_x_in)` + `pack_reconfig_data_format(cb_out)`), then use
  `BinaryDataFormatReconfig::SrcB` (srcB genuinely alternates fp32-rstd↔gamma) + `PackTileReconfig::
  None` on every chain — dropping ~2 wasted reconfigs/chain. Numerically byte-identical. Predicate:
  `not is_rm and has_gamma and not has_partial_w and C > 1` (multi-tile-row rounds; the entry
  reconfig amortizes over C·2 chains/call). At C=1 (single-row WIDTH/decode) there is nothing to
  amortize and the entry reconfig is a net ~1–2% loss (measured), so those stay byte-identical.
- A debug master switch `RMS_PERF1_FASTPATH` (env; default ON) force-disables both for A/B
  re-measurement; it does not change any supported cell.

### Whole-op result (blackhole_p150b, `--profile`, exact perf config)
- **Focus BLOCK_SHARDED (1,1,8192,1024) 8×8: 53822 → 43190 ns = 1.25× whole-op; 2.10× → 1.68× above
  achievable.** Per-stage after: pass1 18→12.5µs, pass2 34→28.5µs (reconfig-skip; still the
  dominant residual), writer round 48.6→38.6µs (tracks compute down — confirms it was hidden).
- Guard-set NO-REGRESSION proven by **same-session A/B** (`RMS_PERF1_FASTPATH` 0 vs 1, median of
  trials 2–8), which controls for the run-to-run drift that confounds cross-session compares:
  | case | baseline OFF | graduated ON | speedup | verdict |
  |---|---|---|---|---|
  | BLOCK 8×8 (focus) | 54738 | 43147 | **1.269×** | WIN |
  | WIDTH 8×1 (per_w_t=4) | 4954 | 4887 | **1.014×** | WIN (pass1_fused; C=1 so no pass-2 skip) |
  | WIDTH 9×1 / 8×4 / 7×4 | — | — | 0.995–1.000× | byte-identical (per_w_t>4 and C=1 → flags off) |
  | decode 32×{1024,2304,5120,7168} | — | — | 0.996–1.006× | byte-identical (out_to_dram → C=1; per_w_t=1) |

### Golden green (correctness — a faster wrong answer is a regression)
- `test_golden.py::test_op_loose` 19/19 (all BLOCK/WIDTH perf geometries + sharded loose).
- Cartesian slices, 0 failed / 0 xpassed: BLOCK_SHARDED bf16 tile_aligned 500 passed / 60 xfail;
  WIDTH_SHARDED bf16 tile_aligned 500 passed / 60 xfail; BLOCK_SHARDED float32 tile_aligned 380
  passed / 180 xfail (xfails = the standing `{f32, fp32_dest_acc_en=False}` EXCLUSION).
- Cross-core correctness `test_rms_norm_r6e.py` + `test_rms_norm_perf_r6a.py` 18 passed
  (PCC ≥ 0.99998; focus-shape perf-test PCC 1.001004 == the R6g byte-identical baseline for the
  reconfig-skip half, +accuracy for the fused-reduce half).
- Interleaved / HEIGHT / RM-HEIGHT use `rms_norm_compute.cpp` (a DIFFERENT kernel, untouched) —
  unaffected; RM-sharded and non-aligned-W and single-round WIDTH/decode xcore cells keep both
  flags off → byte-identical.

### Summary
All 4 ideas measured; **2 graduated** (pass1 fused-reduce 1.51× isolated → whole-op driver; pass2
reconfig-skip 1.44× isolated), 2 superseded (batch-rows on both stages, the pass-2 one deferred to
Perf 2), 1 null (DEST-reuse fusion, re-confirmed dead-end). **Op faster on the focus BLOCK 8×8 by
1.269× (2.10×→1.68× above achievable), + WIDTH 8×1 by 1.014×, no regression anywhere else** (proven
by same-session A/B + golden green). pass2 remains the dominant stage (28.5µs) → the deferred pass-2
row-batch is the top Perf-2 candidate.
- Tests added: 4 isolated micro-benches under `perf_experiments/<slug>/` (durable artifacts,
  committed). Reused `test_rms_norm_perf_r6.py` for whole-op device-ns + soft-PCC.
