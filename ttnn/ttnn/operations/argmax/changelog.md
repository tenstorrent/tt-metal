# Changelog: argmax

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). One Metal 2.0 `ProgramSpec` dispatch on Quasar: one DM kernel `argmax_scan` with 6 threads per cluster on every cluster of the grid (thread plan (6, 0, 0); no compute kernel, since Quasar's NEOs have no index-carrying max). Two regimes: ROWS (rows dealt over all scanners) and SEGMENTED (long rows cut into segments, one 16-B record per segment gathered L1→L1 to a finalizer that merges in column order).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[ROW_MAJOR], memory_config=[DRAM], sharding=[interleaved], dim=[-1], keepdim=[False], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[4]
- **Accuracy achieved**: exact. PCC = 1.0, max_abs_err = 0, rel_rms = 0, 100% exact indices, value error 0 bf16 ULP (measured on 4 shapes covering both regimes via test_argmax_precision_baseline.py; graded `exact` by the specification)
- **Golden suite at Phase 0**: 15 / 15 supported cells pass (12 cells + 3 numeric regression cases), 308 xfail_expected, 0 supported_fail / xpass_drift / xfail_wrong_mode, 2/2 rejections, queue_gaps 0 (`pipeline_logs/golden_phase0/verifier_report.json`)
- **Issues encountered**: no drift. Verifier fixes: the DM cache-line size, computed in three places (cap/slot spacing, kernel CT arg, scratchpad pads), now comes from one source (`line_bytes()` → `Blocks.line_bytes`); `l1_ledger.md` brought in line with the allocation (cap 259,584 B, `sp_partials` 3,456 B, worked footprints).
- **Structural counts at Phase 0** (the baseline for the perf refinements; ttsim is untimed): `(1,1,32,4096)` 128/192 scanners, at most 1024 elements per scanner; `(128,8192)` (rank 2, after R2) 128/192, 8192 per scanner; every PHASE0 scanner walks 1 block at `in_depth = 1`; `scan_piece` does 0.5 loads per bf16 element.
- **Tests added**: test_argmax.py (harness), test_argmax_plan.py (thread plan, sessions, sub-grids, streaming), test_argmax_precision_baseline.py
