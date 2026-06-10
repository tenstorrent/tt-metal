# Changelog: groupnorm_sc_N_1_HW_C

## Phase 0 — Core Implementation
- **Date**: 2026-06-10
- **What was done**: Initial implementation via incremental pipeline (planner → implementer →
  verifier). Single-core GroupNorm over (N, 1, HW, C): per-(n,g) three streaming passes
  (mean → centered variance → normalize + optional affine) using kernel-lib helpers throughout.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE, ROW_MAJOR],
  alignment=[tile_aligned], groups_alignment=[aligned],
  affine=[gamma_beta, gamma_only, no_affine], affine_dtype=[bfloat16],
  affine_layout=[ROW_MAJOR, TILE]
- **Accuracy achieved**: PCC ≥ 0.999992, max_abs_err ≤ 0.080, rel_rms_err ≤ 0.0036
  (measured on 4 shapes via test_groupnorm_sc_N_1_HW_C_precision_baseline.py, bf16 gamma_beta)
- **Golden suite at Phase 0**: 300 / 7236 cells passing (3385 xfail_expected, 3551
  invalid_skipped; supported_fail = xpass_drift = xfail_wrong_mode = 0, per `verifier_report.json`)
- **Issues encountered**: verifier fixes — (1) reader/writer per-tile NoC barriers batched per
  Wg row chunk; (2) affine_layout=TILE under-claim promoted into SUPPORTED on probe evidence
  (PCC ≥ 0.99999, +120 golden cells). Known boundary: G=1 + C ≥ 2048 with gamma_beta exceeds
  L1 (no golden cell affected; see verification_report.md).
- **Tests added**: test_groupnorm_sc_N_1_HW_C.py (planner),
  test_groupnorm_sc_N_1_HW_C_precision_baseline.py,
  test_groupnorm_sc_N_1_HW_C_extended.py (verifier)

## Refinement 1 — Numerical configurability + multi-core distribution
- **Date**: 2026-06-10
- **What was done**:
  - `SUPPORTED["dtype"]` and `SUPPORTED["affine_dtype"]` extended with `float32`, `bfloat8_b`
    (incl. all mixed-precision combinations). Zero compute-kernel changes for dtype work —
    input/output/affine CB formats were already dtype-derived; helpers carry data-format reconfig.
  - Intermediate stat CBs (mean/var/centered/xhat/scaled): `Float32` when `fp32_dest_acc_en`
    or fp32 input, else `bfloat16` (incl. bf8b input — block-float intermediates lose precision
    for no L1 win). No `UnpackToDestFp32` tags: every intermediate feeds FPU helpers
    (sub/mul/square/reduce), which the tag forbids.
  - `compute_kernel_config` exposed on the entry point (WormholeComputeKernelConfig-style:
    math_fidelity, fp32_dest_acc_en, math_approx_mode, dst_full_sync_en). Defaults match the
    Phase-0 hard-coded behavior for bf16/bf8b; fp32 input defaults `fp32_dest_acc_en=True`
    (dtype-driven; fp32 had no prior behavior — measured rel_rms 0.0112→0.0075 on
    (1,1,1024,256) G=8 + bf8b gamma; flipped the only 4 failing golden cells).
  - Multi-core: N·G (n,g) groups split over the full grid via `split_work_to_cores`
    (one (n,g) group per work unit); per-core `[start_group, num_groups_here]` runtime args
    in all three kernels; gamma/beta read once per core; per-core CB footprint unchanged.
- **Accuracy achieved**: precision matrix 384/384 (8 shapes × 3 dtypes × 4 fidelities ×
  2 acc × 2 distributions): HiFi4 PCC ≥ 0.99986 all dtypes; worst overall PCC 0.9977 (LoFi).
  See tests/.../precision_matrix_results.md.
- **Golden test progress**: 1650 / 7236 supported_pass (Phase 0: 300), 2035 xfail_expected,
  3551 invalid_skipped, 0 supported_fail / 0 xpass_drift / 0 xfail_wrong_mode. EXCLUSIONS
  remains empty.
- **Issues encountered**: FLOAT32 + gamma_only + bf8b gamma initially failed rms by ~5%
  (0.0104–0.0106 vs 0.01) on default config. Probes showed bf8b-gamma quantization alone is
  only 0.0055–0.0066; the gap was TF32 dest rounding of the fp32 stats — fixed by the
  dtype-driven fp32_dest_acc_en default (no EXCLUSIONS needed). LoFi rel-RMS ≈ 0.021 vs
  0.02 HiFi band — expected fidelity tradeoff, asserted via PCC only.
- **Tests added**: test_groupnorm_sc_N_1_HW_C_refinement1.py (26 cases: new dtypes, mixed
  affine, compute config sweep, multicore split regimes incl. uneven 100-group + distinct-group
  routing), test_groupnorm_sc_N_1_HW_C_precision_matrix.py (384 cases) +
  precision_matrix_results.md, probes/probe_004.py.
