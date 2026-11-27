# Changelog: g6_expand

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier):
  regime R1 — column-major `split_work_to_cores` over output tiles, per-core row-0 fetch of x into the
  broadcast tile slot, in-L1 replicate (RISC-V rows 0..7 + local NoC doubling), whole-tile writes split
  between writer (NoC1) and reader (NoC0). One `generic_op` dispatch, single CB, no compute kernel.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], alignment=[tile_aligned, w_non_aligned]
  (= TARGET).
- **Accuracy achieved**: bit-exact — PCC=1.0, max_abs_err=0, mean_abs_err=0, rel_rms_err=0, bit diffs 0
  (measured on W ∈ {185, 196, 256, 512} via test_g6_expand_precision_baseline.py); special values and
  poisoned padding bit-exact in golden + acceptance.
- **Device time at Phase 0** (BH, DEVICE KERNEL DURATION): W=512 ≈ 2.38–2.51 µs (110 cores; goal
  ≤ 1.54 µs); W=256 ≈ 1.70 µs; W=196/207 ≈ 1.52 µs; W=185 ≈ 1.49 µs.
- **Golden suite at Phase 0**: 20 / 20 passing; verifier 14 supported_pass, 0 supported_fail /
  xpass_drift / xfail_wrong_mode (per `verifier_report.json`).
- **Issues encountered**: verifier fix — reader `cb_reserve_back` wrapped in its own profiler zone
  `G6E_RESERVE` (BB-21). No drift.
- **Tests added**: test_g6_expand.py (acceptance), test_g6_expand_precision_baseline.py,
  test_g6_expand_perf_bench.py (device-ns bench for the perf refinements).
