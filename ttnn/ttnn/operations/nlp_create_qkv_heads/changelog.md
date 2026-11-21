# Changelog: nlp_create_qkv_heads

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Regime
  `tile_relocate`: Metal 2.0 `ProgramSpec`, reader → STRIDED implicit-sync ring → writer, no compute kernel,
  folded input-page split over the full 8×4 Quasar grid, thread plan (1, –, 1).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], memory_config=[DRAM], sharding=[interleaved],
  transpose_k_heads=[False], output_memory=[DRAM], alignment=[tile_aligned], batch=[single] (= PHASE0)
- **Accuracy achieved**: bit-exact — PCC=1.0, max_abs_err=0, mean_abs_err=0, rel_rms_err=0, 0 ULP (measured on
  4 shapes × q/k/v via test_nlp_create_qkv_heads_precision_baseline.py)
- **Golden suite at Phase 0**: 9 / 9 supported cells passing; 15 / 439 tests passed (incl. 6 rejections),
  423 xfail_expected, 0 supported_fail / xpass_drift / xfail_wrong_mode (per `verifier_report.json`)
- **Issues encountered**: no drift. Verifier code-review fix: `output_shapes` now delegates to the
  specification's `head_shapes` (single definition of the result shapes).
- **Tests added**: test_nlp_create_qkv_heads.py (generated acceptance), test_nlp_create_qkv_heads_plan.py
  (entries, split, knobs, device sessions), test_nlp_create_qkv_heads_precision_baseline.py
- **Perf (structural, ttsim — no clock)**: 1 DRAM read + 1 DRAM write per page, 1 NoC transaction per page,
  0 cross-core bytes, `min(num_pages, 32)` clusters engaged, 16 KiB L1 per cluster. No timing measured.
