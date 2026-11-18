# Changelog: rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-11-18
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). One `ttnn.generic_op` dispatch per call with two built regimes chosen by `select_blocking()`:
  - **LOCAL**: tile-rows split across the full grid, whole row resident per core.
  - **W_SPLIT**: lines of `w_split` cores share an R-range; Σx² partials are gathered to the line root, and rinv is multicast back via `ttnn.Mcast`.

  TILE and ROW_MAJOR are handled natively (tilize/untilize in the kernel). Non-aligned H/W are handled in the kernel (W-pad lanes zeroed in L1, logical `1/W` divisor).
- **SUPPORTED at Phase 0**:
  - dtype=[float32, bfloat16]; fp32_dest_acc_en=[True]; layout=[TILE, ROW_MAJOR]
  - alignment=[tile_aligned, w_non_aligned, h_non_aligned]; rank=[2, 3, 4]
  - gamma_mode=[gamma, no_gamma]; gamma_dtype=[float32, bfloat16, "none"]; gamma_layout=[TILE, ROW_MAJOR, "none"]
  - memory_layout=[INTERLEAVED]
  - EXCLUSIONS: {float32, fp32_dest_acc_en=False} (permanent)
- **Accuracy achieved** (`test_rms_norm_precision_baseline.py`; 4 shapes × {bf16, fp32} × {TILE, RM}, HiFi4, fp32 DEST):
  - bf16: PCC ≥ 0.999995, max_abs_err ≤ 0.088, rel_rms_err ≤ 0.0036, max 2 ULP
  - fp32: PCC ≥ 0.9999997, max_abs_err ≤ 0.028, rel_rms_err ≤ 0.0017
- **Golden suite at Phase 0** (per `verifier_report.json`):
  - 659 supported_pass, 5566 xfail_expected, 33900 invalid_skipped, 2 infeasible_skipped
  - supported_fail = xpass_drift = xfail_wrong_mode = 0
  - The pytest runner itself reported 672 passed, 0 failed, 0 hangs (`golden_results.txt`).
- **Issues encountered**: no drift fixes were needed. The verifier's code-review change: removed two unused compile-time args (reader `t_f`, writer `OUT_RM_DEPTH`) and renumbered the CT-arg indices to match.
- **Tests added**: test_rms_norm.py (acceptance, 110 cases; planner), test_rms_norm_perf.py (implementer), test_rms_norm_precision_baseline.py (verifier, 16 cases)

## Refinement 1 — Numerical configurability: fp32_dest_acc_en=False + bfloat8_b (input and γ)
- Date: 2026-11-18
- What was done:
  - **SUPPORTED** now includes `fp32_dest_acc_en` = [True, False], `dtype` += bfloat8_b and `gamma_dtype` += bfloat8_b. EXCLUSIONS keeps `{float32, False}` (permanent).
  - **Intermediate format** (ledger audit 3): `create_program_descriptor` makes one decision, `int_fmt = float32 if fp32_dest_acc_en else bfloat16`, with `t_int = tile_size(int_fmt)`. That single value feeds:
    - the six intermediate CBs: `cb_sumsq_raw`, `cb_partial`, `cb_gathered`, `cb_rinv_col`, `cb_rinv_bcast`, `cb_rinv`;
    - `_per_row_bytes()` (L1 budget, via `OpGeometry.t_int`, which replaces `t_f`), so `block_rows` grows when the tiles halve;
    - the writer's gather / multicast byte count. That CT arg was renamed from `f32_tile_bytes` to `int_tile_bytes`.
  - **Compute kernel unchanged.** It uses the same fused helper chains, and the helpers do the format reconfig.
  - **bf8b TILE γ**: the writer's `load_gamma` has a new block-float branch (CT arg 19, `gamma_is_block_float`). It reads the valid γ tiles whole, with one barrier, and zero-fills the nominal tail tiles. The face-row read stays for bf16/fp32 γ.
  - **bf8b input** uses the existing TILE path (`t_in` = 1088 via `tile_size`).
    - Host guard: bf8b requires TILE and an aligned W (ValueError otherwise; those cells are INVALID in feature_spec).
    - Reader `static_assert`: block-float is detected as `tile_bytes != 1024·es` and must never reach the RM / pad-lane paths.
    - `_element_bytes()` replaces `Tensor.element_size()`, which raises for bf8b.
  - The default compute config was left unchanged (HiFi4, fp32_dest_acc_en=True), following numeric-formats §9 "defaults must match prior behaviour". The golden harness always passes an explicit config.
  - Reused: the TILE reader/writer paths, all compute helpers, `select_blocking`. Added: one format variable, one γ read branch, guards.
- Accuracy achieved:
  - bf16 with fp32_dest_acc_en=False: PCC ≥ 0.995 on LOCAL / W_SPLIT / non-aligned shapes, TILE and RM, γ TILE and RM, poisoned padding.
  - Perf-profile config (bf16 / HiFi2 / acc off / TILE γ): PCC ≥ 0.9995 on (1,1,32,1024), (1,1,32,7168), (1,1,640,6144), (1,1,7168,1024).
  - bf8b input: PCC ≥ 0.99 versus the dequantized reference, with fp32 accumulation on and off.
  - bf8b γ: PCC ≥ 0.995 (bf16 x) / 0.999 (fp32 x).
- Golden test progress (targeted slices only; the harness full run is the record):
  - Interleaved loose cases with fp32_dest_acc_en=False: 100/100 pass, including all 12 interleaved `_perf_case` cases at PCC 0.9995 and the interleaved resilience and pad-poison cases.
  - Cartesian interleaved slice (bf8b or acc-off, 6 shapes): 129 passed, 66 xfailed (all `{float32, False}` EXCLUSIONS), 0 failed.
- Perf note (device ns, HiFi2, TILE bf16 γ, fp32-acc → bf16-acc):
  - (1,1,32,7168) W_SPLIT: 9210 → 8484 (−8%; gather and mcast bytes halved).
  - (1,1,640,6144): 48970 → 49639.
  - (1,1,8192,7168): 550390 → 552227.
  - (1,1,7168,1024): 77199 → 78061.
  - The three prefill shapes are DRAM-bound and flat within about 1.5%.
- Issues encountered: `Tensor.element_size()` raises for bfloat8_b. Fixed with the `_element_bytes()` helper. No other issues.
- Tests added: `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_numeric.py` (80 cases: acc-off matrix, bf8b input, bf8b γ, acc-off poisoned padding, perf-profile config at 0.9995, fp32+acc-off refusal).
