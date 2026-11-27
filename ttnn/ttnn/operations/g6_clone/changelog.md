# Changelog: g6_clone

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier).
  - Regime `flat_page_range`: the flat page axis is split into contiguous ranges over the full grid (110 cores, `row_wise=True`).
  - The reader (NCRISC, NoC0) moves each core's whole range into one CB behind one barrier. The writer (BRISC, NoC1) drains it behind one flush, then one final write barrier.
  - There is no compute kernel. The program descriptor is cached, with the buffer addresses patched per call.
- **SUPPORTED at Phase 0**: dtype=[float32], layout=[TILE]. There are no shape-derived or op-specific axes.
- **Accuracy achieved**: bit-exact. PCC=1.0, max_abs_err=0, rel_rms_err=0, max ULP=0, got/true ratio 1.0, measured on 4 shapes via `test_g6_clone_precision_baseline.py`: (1,1), (1,12,128,128), (8,12,128,128), (16,12,128,128). The golden suite also grades bit equality on uniform random bit patterns (NaN payloads, ±inf, subnormals).
- **Golden suite at Phase 0**: 7/7 SUPPORTED cells pass, 28 xfail_expected, 0 supported_fail, 0 xpass_drift, 0 xfail_wrong_mode. All 3 non-registry regression cases pass (per `golden_results/verifier_report.json`).
- **Perf at Phase 0** (device kernel ns, median of 6): p1 (16,12,128,128) f32 63182 (stock 68320, roofline 54610), p2 32492, (1,12,128,128) 4700.
- **Issues encountered**: the verifier made three code-review fixes and no drift fixes.
  - Exposed `BLOCKS_PER_CORE_KNOB`: the block count per core had been collapsed to 1.
  - Routed the entry point and `g6_clone_plan` through one `_plan()`.
  - Added a multi-block ring test for the depth-2 path, which no INPUTS shape exercises.
  - `eval.verify_supported` needs `.tt_chip_scope` sourced (the MPIX stub `LD_PRELOAD`) to import ttnn outside `run_safe_pytest.sh`.
- **Tests added**: `test_g6_clone.py` (incl. `test_g6_clone_multi_block_ring`), `test_g6_clone_precision_baseline.py`; perf/lamp probes `probes/perf_baseline_probe.py`, `probes/grid_cap_probe.py`, `probes/block_knob_probe.py`, `probes/probe_006.py` (TARGET-widened cell sweep).
