# Changelog: rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Quasar, one Metal 2.0 `ttnn.ProgramSpec` dispatch, regime `row_resident` (tile-rows over the full grid and the 4 NEOs of each cluster, row resident across the reduce and normalise passes), thread plan `(4, 4, 2)` with per-call `C_eff ∈ {1, 2, 4}`.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True], layout=[TILE], memory_config=[DRAM, L1], sharding=[interleaved], gamma_dtype=[bfloat16], gamma_layout=[ROW_MAJOR], memory_argument=[None], alignment=[tile_aligned, h_non_aligned], rank=[4]
- **Accuracy achieved**: PCC=0.9999986, max_abs_err=0.033, rel_rms_err=0.0017, got/true ratio median 1.000 (p5..p95 0.997..1.003) (measured on 6 shapes via test_rms_norm_precision_baseline.py); ULP p99 = 1 in every golden cell
- **Golden suite at Phase 0**: 42 / 633 passed (40 supported_pass incl. 6 numeric regression cases, 2 rejections), 590 xfail_expected, 0 supported_fail / xpass_drift / xfail_wrong_mode (per `verifier_report.json`)
- **Issues encountered**: no drift. Verifier fixes: the gamma ring's entry geometry routed from one host format constant to the reader as CT args (was bf16 literals in the kernel), with a compile-time check of the weight page against the entry format; `TILE` defined once; `l1_ledger.md` `row_stat` / `row_sum` capacities corrected to `rows_per_block · C_eff`.
- **Tests added**: test_rms_norm.py (harness acceptance), test_rms_norm_plan.py (thread plan, regimes, sessions, knobs), test_rms_norm_precision_baseline.py
