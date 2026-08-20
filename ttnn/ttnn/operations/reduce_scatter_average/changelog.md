# Changelog: reduce_scatter_average

## Phase 0 — Core Implementation
- **Date**: 2026-08-20
- **What was done**: Initial implementation via incremental pipeline (planner → implementer →
  verifier). Self-contained Python CCL op (`ttnn.generic_op` + `ttnn.MeshProgramDescriptor`,
  ONE dispatch per invocation) with 5 newly-authored kernels: single-program fused line
  store-and-forward gather + arrival-ordered incremental N-way reduce + 1/N broadcast-scalar
  scale, per-device-distinct sliced output. Verified on REAL 4-chip Blackhole hardware
  (`bh_quietbox_1x4_hw`: mesh (1,4), FABRIC_1D) via
  `scripts/run_multidevice_sim_pytest.py --runtime hardware --op reduce_scatter_average`.
- **SUPPORTED at Phase 0**: dtype=[bfloat16, float32], layout=[TILE], topology=[Linear], dim=[3]
  (negative alias -1 canonicalized); EXCLUSIONS=[], INPUT_TAGGERS={}.
- **Accuracy achieved**: bf16 worst-device PCC ≥ 0.9999952, max_abs_err ≤ 0.0156 (1–2 output-ULP
  at scale — pure bf16 quantization), rel_rms_err ≈ 0.0036; float32 worst-device PCC ≈ 0.9999998,
  max_abs_err ≤ 0.004, rel_rms_err ≈ 1.0e-3 (FPU srcA/srcB truncation) — measured on 4 shapes ×
  2 dtypes at N=4 via `test_reduce_scatter_average_precision_baseline.py` (8/8 pass).
- **Golden suite at Phase 0**: 6 / 6 in-SUPPORTED cells passing (3 INPUTS × {bf16, f32} × TILE ×
  Linear × dim=3); 18 xfail_expected (dim=2 and Ring — the refinement queue); all loud categories
  0, per `generated/reduce_scatter_average_verify/verifier_report.json`.
- **Issues encountered**: two pre-existing golden-harness defects in
  `eval/golden_tests/reduce_scatter_average/helpers.py` (both pre-flagged by `op_design.md`),
  fixed by the verifier: (1) NameError — the driver called the undefined `reduce_scatter(...)`;
  (2) SUM oracle contradicting the op's MEAN semantics (`.sum(dim=0)` → `.mean(dim=0)`). No op
  code or kernel defects found; no drift (xpass_drift = supported_fail = xfail_wrong_mode = 0).
- **Tests added**: test_reduce_scatter_average.py + test_reduce_scatter_average_debug.py
  (implementer), test_reduce_scatter_average_extended.py +
  test_reduce_scatter_average_precision_baseline.py (verifier).
