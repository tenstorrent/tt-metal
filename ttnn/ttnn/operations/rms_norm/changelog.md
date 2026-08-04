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
