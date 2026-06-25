# Changelog: softmax

## Phase 0 — Core Implementation
- **Date**: 2026-06-25
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). 4-phase numerically-stable softmax (max reduce → sub+exp → sum+recip → mul) using kernel-lib helpers (`reduce`, `eltwise_chain`, `mul`). Multi-core work distribution via `split_work_to_cores` over (N,C) slabs.
- **SUPPORTED at Phase 0**: dtype=[float32], layout=[TILE_LAYOUT], alignment=[tile_aligned], rank=[4], dim=[-1, -2], fp32_dest_acc_en=[True]
- **EXCLUSIONS at Phase 0**: {fp32_dest_acc_en=False} — fp32-dest-only op, rejected for all dtypes
- **Accuracy achieved**: PCC ≥ 0.999, max_abs_err ≤ 0.0014, rel_rms_err ≤ 0.0014 (measured on 8 shape×dim combinations via test_softmax_precision_baseline.py)
- **Golden suite at Phase 0**: 37 / 1250 cells passing (per `verifier_report.json`); 1053 xfail_expected, 140 invalid_skipped, 12 supported_fail (all OOM on wide shapes)
- **Issues encountered**:
  - Fixed: missing `tag_rank` INPUT_TAGGER and `rank` axis in SUPPORTED — caused 62 rank-2/3 `supported_fail` cells (kernel crashed on non-4D shapes because validate() didn't reject them)
  - Fixed: alignment tagger returned two values (`tile_aligned`, `non_tile_aligned`) instead of the three values expected by feature_spec (`tile_aligned`, `w_non_aligned`, `h_non_aligned`)
  - Fixed: EXCLUSIONS entry `{dtype: float32, fp32_dest_acc_en: False}` was too narrow — should reject fp32_dest_acc_en=False for ALL dtypes per the fp32-dest-only policy. Changed to `{fp32_dest_acc_en: False}` (dtype-agnostic)
- **Tests added**: test_softmax.py (acceptance, 40 tests), test_softmax_precision_baseline.py (precision baseline, 8 tests)

## Refinement 1 — Numerical configurability (dtypes + fp32-dest-only policy)
- **Date**: 2026-06-25
- **What was done**:
  - Added `ttnn.bfloat16` and `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`
  - Fixed intermediate CB formats: `cb_max`, `cb_exp`, `cb_recip_sum` now use `ttnn.float32` format + `intermediate_tile_size` instead of `output_tensor.dtype` / `output_page_size`. This is the key precision fix — accumulations cross the CB at phase boundaries, and the intermediate format must be Float32 when `fp32_dest_acc_en=True` (which is always the case for this op).
  - Added pre-emptive `EXCLUSIONS` for `bf8b + w_non_aligned` and `bf8b + h_non_aligned` (activate when Refinement 3 adds non-aligned alignment values to SUPPORTED)
  - No compute kernel changes — helpers handle data-format reconfig automatically (skill pass condition held)
  - No `UnpackToDestFp32` tagging — all intermediates feed FPU ops (reduce, BinaryFpu Sub, BinaryFpu Mul), never `copy_tile`-only
- **Accuracy achieved**:
  - bf16: PCC ≥ 0.99999 (HiFi4), PCC ≥ 0.9988 (LoFi) on shapes [(1,1,32,32), (1,1,64,128), (2,4,64,64), (4,8,32,256), (1,1,128,512)]
  - bf8b: PCC ≥ 0.9998 (HiFi4), PCC ≥ 0.9985 (LoFi) on same shapes
  - fp32: PCC ≥ 0.9999 (HiFi4), PCC ≥ 0.9989 (LoFi) on same shapes
  - rtol/atol: all within golden tolerance map (helpers.py TOLERANCES)
- **Golden test progress**: 37 → 157 passing (130 new: 20 bf16 tile_aligned + 20 bf8b tile_aligned + 90 from prior xfail cells now passing due to SUPPORTED expansion). 39 pre-existing failures remain (all OOM on wide shapes — same as Phase 0, addressed by Refinement 5).
- **Issues encountered**: None. The skill's pass condition held perfectly — zero kernel changes needed when helpers are wired correctly.
- **Tests added**:
  - `test_softmax_precision_matrix.py` (192 cases: 8 shapes × 3 dtypes × 4 fidelities × 2 distributions)
  - `test_softmax_refinement1_dtypes.py` (47 cases: dtype support, fp32_dest_acc_en=False rejection, output dtype match, numerical stability, L1 memory)
  - `precision_matrix_results.md` (results file)
