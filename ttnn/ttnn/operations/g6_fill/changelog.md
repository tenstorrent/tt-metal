# Changelog: g6_fill

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Regime R1: one writer kernel writes the two 64-B row-0 segments of every tile, with the `w_tiles` split row-wise over `min(ceil(N/32), grid)` cores. No reader and no compute kernel.
- **SUPPORTED at Phase 0**: dtype=[float32], layout=[TILE], alignment=[w_non_aligned, h_non_aligned]
- **Accuracy achieved**: bit-exact. PCC=1.0, max_abs_err=0, rms_err=0, max ULP=0, got/true ratio exactly 1.0, measured on 4 shapes x 3 values via `test_g6_fill_precision_baseline.py`.
- **Golden suite at Phase 0**: 15 / 16 supported cells pass in the graded run (`verifier_report.json`), plus 3 xfail_expected (int32 ROW_MAJOR). The one failure is the trace case `loose6`: the graded L1 capture disables the program cache, so no program can be captured into a trace. With `EVAL_CAPTURE_L1=0`, 16 / 16 pass. Escalated in `verification_report.md` (H1).
- **Issues encountered** (verifier fixes):
  - Knob single source: `Knobs` / `current_knobs()` feeds both the plan and the cache key. `FILL_ROW_DEPTH` had been missing from the cache key.
  - The common-args `dst_addr` index is now a host-owned CT arg instead of a kernel re-derivation.
  - `FILL_ROW_SOURCE` default changed to `"cb"`. With the faster `"common_args"` source the graded L1 capture sees a zero peak and fails every cell (0/19 verified). The cost is about 100 ns per call.
  - The writer's argument reads moved inside the `G6FILL-ISSUE` zone (BB-21).
  - `l1_ledger.md` brought current.
- **Tests added**: `test_g6_fill.py`, `test_g6_fill_knobs.py`, `test_g6_fill_cores_lamp.py` (implementer); `test_g6_fill_precision_baseline.py` (verifier)
