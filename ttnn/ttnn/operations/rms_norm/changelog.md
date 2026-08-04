# Changelog: rms_norm

## Phase 0 — Core Implementation

- **Date**: 2026-08-04
- **What was done**: Initial implementation via the incremental pipeline
  (planner → implementer → verifier). Row-parallel, multi-core, coarse-blocked scheme with
  a dual-path fits-in-L1 predicate (RESIDENT / STREAM), native TILE **and** ROW_MAJOR
  layouts, native non-tile-aligned H and/or W, optional gamma at an independent
  dtype/layout. Reader (NCRISC/NoC0) + compute + writer (BRISC/NoC1); every compute phase is
  a `compute_kernel_lib` helper; every block / depth / grid knob is a parameter in
  `rms_norm_program_descriptor.py` solved from `L1_SAFETY_FRACTION` and
  `ttnn.get_max_worker_l1_unreserved_size()`.
- **SUPPORTED at Phase 0**: dtype=[float32, bfloat16], fp32_dest_acc_en=[True],
  layout=[TILE, ROW_MAJOR], alignment=[tile_aligned, w_non_aligned, h_non_aligned],
  rank=[2, 3, 4], gamma_mode=[gamma, no_gamma], gamma_dtype=[float32, bfloat16, "none"],
  gamma_layout=[TILE, ROW_MAJOR, "none"], memory_layout=[INTERLEAVED].
  EXCLUSIONS=[{float32, fp32_dest_acc_en=False}].
- **Accuracy achieved** (4 shapes × 2 dtypes × 2 layouts via
  `test_rms_norm_precision_baseline.py`, HiFi4 + fp32_dest_acc_en=True):
  bfloat16 — PCC=0.999997, max_abs_err=0.0452, mean_abs_err=0.00128, rel_rms_err=0.0024,
  ≤2 ULP (bf16 grid);
  float32 — PCC=0.9999997, max_abs_err=0.0246, mean_abs_err=0.00082, rel_rms_err=0.0015.
  Uniform across shape, regime (RESIDENT/STREAM), layout and alignment. got/true ratio
  median 0.9996 (bf16) / 0.9988 (fp32) with a spread wider than the offset ⇒ precision
  noise, **not** a uniform scale error (the fp32 residue pins at exactly 1 − 2⁻¹⁰, an
  SFPU/FPU datapath effect — see `verification_report.md`).
- **Golden suite at Phase 0**: **737 / 737** supported cells passing; 6172 xfail_expected,
  33900 invalid_skipped, 2 infeasible_skipped, 15 non-registry regression tests passing.
  `supported_fail = 0`, `xpass_drift = 0`, `xfail_wrong_mode = 0` (per `verifier_report.json`).
  Runner line: `PASSED=752 FAILED=0 ERRORS=0 SKIPPED=33902 HANGS=0 TOTAL=40828`.
- **Issues encountered**: no drift fixes were needed — SUPPORTED was already honest.
  Six code-review fixes applied by the verifier, all non-behavioural (golden summary
  identical before/after): deduplicated the block-scoped-CB multiplier into
  `_cb_block_mult()` (was written twice, in the RESIDENT predicate and the STREAM solve);
  deduplicated the scaler CB page count into `scaler_pages`; removed the dead `x_resident`
  variable; made the dead `GRID_W` knob live via an explicit `NotImplementedError` guard;
  removed a bare `except Exception` in `_cores_in()`; added the writer's missing
  CT-arg-offset assert. Known deviations left in place and documented: **D3** (the fp32
  reduce runs `ReduceFp32Mode::Fast` because `accumulate_reduce_block<>` does not expose the
  slot) and the prime-`Wt` STREAM chunk-granularity cliff — both carried into
  `op_requirements.md`. `feature_spec.py`'s INVALID list has three mis-categorised
  author-scoped entries and two missing gamma-bf8b entries; relayed in the report, not
  edited.
- **Tests added**: `test_rms_norm.py` (acceptance, immutable — 205 cases),
  `test_rms_norm_perf.py` (10 device-perf probes),
  `test_rms_norm_precision_baseline.py` (16 cases, new this pass — PCC / abs / rel-RMS /
  ULP / got-true ratio spread with a uniform-scale assertion). Unit suite total: 231 passed.

## Refinement 1 — Numerical configurability expansion (dtype + DEST/fidelity surface)

- **Date**: 2026-08-04
- **What was done** (partial — `[~]`):
  - **Reused**: every CB already declared `data_format` = the dtype of the tensor it carries
    and `page_size` = `ttnn.tile_size()` of that same dtype, so `bfloat8_b` needed **no new
    format machinery and no compute-kernel change** — the numeric-formats pass condition held
    (all phases are `compute_kernel_lib` helpers, no hard-coded formats or sizes).
    `cb_row_stat` was already `float32` and stays so in **both** `fp32_dest_acc_en` modes
    (Refinement 1 lever 1: the cross-chunk `Accumulate::at` reload stays lossless even when
    DEST is 16-bit). The compute config is still passed through unmodified, so `False`
    required only widening `SUPPORTED`.
  - **Added**: `bfloat8_b` to `SUPPORTED["dtype"]` and `SUPPORTED["gamma_dtype"]`; `False` to
    `SUPPORTED["fp32_dest_acc_en"]` (`{float32, False}` EXCLUSION intact and now load-bearing).
    `_stick_elem_bytes()` — `Tensor.element_size()` raises `"datum for bfp2, bfp4, bfp8 is
    invalid"` on a block-float dtype; the number is consumed only by the ROW_MAJOR stick byte
    math, which block-float can never reach (no sticks, only exponent blocks), so it reports 0
    there behind an asserted `layout == TILE` (descriptor **D5**).
  - **`unpack_to_dest_mode` left entirely Default, deliberately**: the only fp32 CB is
    `cb_row_stat`, and although its reduce reload (`AccumulateReloadMode::CopySeedPairs`) and
    the `transform_in_place` finalize are both `copy_tile`-into-DEST and would be compatible,
    **pass B consumes it as operand B of an FPU broadcast multiply** — an `UnpackToDestFp32`
    CB may never be an FPU operand, so tagging it would corrupt silently. Tagging
    `cb_input_sticks` is separately forbidden by `tilize`'s `Fp32Mode::Fast` static_assert
    (design R16). Recorded in the descriptor as part of **D5** so it is not "re-discovered".
  - **Fixed a latent pre-existing bug (descriptor D6)** surfaced by the cells this refinement
    unlocked: `cb_row_stat` is now `CB_ROW_STAT_DEPTH (= 2) * BLOCK_ROWS` pages, derived
    through both L1 solves from one constant. `transform_in_place` *rotates* its CB (pop 1 /
    push 1), so with a ring of exactly `BLOCK_ROWS` a **partial** final row-block left the
    finalized stats straddling the ring wrap and pass B's `OperandKind::Col` bulk-indexed read
    ran off the end — the 2nd..last row of every partial block was garbage. Reproduced
    identically at `fp32_dest_acc_en=True`, so **not** an axis bug: every Phase-0 golden cell
    had `Rt ≤ 64 <` the 110-core grid ⇒ `BLOCK_ROWS == 1` ⇒ no partial block could exist.
    This cleared 11 catastrophic cells (PCC 0.55–0.99 → 0.9999+).
  - **Zero new EXCLUSIONS.** `op_design.md` §9.2 predicted `{gamma_dtype: bfloat8_b,
    alignment: *_non_aligned}` would fail; it is clean (PCC 0.99997 / rel RMS 0.008). A bf8b
    gamma's tile padding is **zero**, and a zero never raises a block's shared exponent, so
    the real weights sharing that block are untouched. Pinned by a new test rather than
    excluded, with the reasoning recorded next to `EXCLUSIONS`.
- **Accuracy achieved** (`test_rms_norm_precision_baseline.py`, 269 unit cases):
  `fp32_dest_acc_en=True` — bfloat16 PCC 0.999997 / rel RMS 0.0024, float32 PCC 0.9999997 /
  rel RMS 0.0015 (**Phase 0 baseline held exactly**), bfloat8_b PCC 0.99987 / rel RMS 0.017.
  `fp32_dest_acc_en=False` — bfloat16 PCC 0.99999 / rel RMS 0.0024–0.006 for `W ≤ 4096`,
  bfloat8_b PCC 0.99986 / rel RMS 0.018, mixed bf8b-gamma × bf16/fp32-activation PCC 0.99997.
  Wide `W` at `False` degrades: rel RMS 0.042 @ 5120, 0.050 @ 7168, 0.127 @ 11008 (PCC stays
  0.99993+) — see *Issues*.
- **Golden test progress**: **1660 / 1670** live cells passing (Phase 0: 737 / 737), 5256
  xfail_expected, 33902 invalid_skipped, `xpass_drift = 0`, zero hangs, **no regression** on
  any previously-passing cell. Newly live and green: 240 `bfloat8_b`-activation cells, 250
  `bfloat8_b`-gamma cells, 420 `bfloat16 × fp32_dest_acc_en=False` cartesian cells, plus the
  resilience / pad-poison loose sweeps at `False`.
- **Issues encountered**: 10 cells still fail, all `severity=precision`, all
  `bfloat16 × fp32_dest_acc_en=False × W ≥ 5120`. PCC is 0.99993–0.99996 — **above** the perf
  cases' soft `pcc_threshold = 0.9995` — so what misses is the `rms ≤ 0.04` component of
  `TOLERANCES[bfloat16]` (0.041–0.127). Diagnosed to the datapath rather than guessed:
  DEVICE_PRINT on `cb_row_stat` shows the **reduce output** wrong (7904 vs a true 7033,
  +12.4 %) while the finalize is exact, and the error is **bit-invariant** across
  `NUM_W_CHUNKS` 4 → 112, across `REDUCE_BULK` ∈ {1,0} and across all four `math_fidelity`
  values. So the cross-chunk `Accumulate::at` reload (the verifier's lever 2) is a **measured
  null**, and the real source is the FPU matmul reduce's *within-tile* 32-column sum
  accumulating all-positive addends into a 16-bit DEST — unreachable by any chunking knob.
  Per protocol these precision-near-miss cells are left **failing, not excluded**; they are
  the baseline for the new `Refinement 1b`, which names
  `ReduceAlgorithm::AccumulateViaAdd` as the exact next lever (and notes that
  `accumulate_reduce_block<>` does not yet forward `reduce<>`'s `algorithm` slot — the same
  class of gap as **D3**).
- **Tests added**: `test_rms_norm_precision_baseline.py` extended in place (no second file)
  from 16 to 96 + 40 cases — matrix 1 is now `shape × dtype × layout × fp32_dest_acc_en` with
  `bfloat8_b` added, matrix 2 is the new `test_rms_norm_precision_mixed_gamma_dtype`
  (independent activation-dtype × gamma-dtype, incl. the §9.2 bf8b-gamma × non-aligned corner
  that was predicted to fail). Both share one `_measure()` body so a further axis cannot fork
  the metrics, and `_skip_unsupported()` mirrors `EXCLUSIONS` + `feature_spec.INVALID` so the
  file never asserts on a cell the op may refuse. Unit suite: **269 passed, 30 skipped**.
