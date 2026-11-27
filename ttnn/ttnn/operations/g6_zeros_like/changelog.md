# Changelog: g6_zeros_like

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: initial implementation via the incremental pipeline (planner → implementer → verifier). The op is one writer kernel per participating core (BRISC/NoC1). It zero-fills one L1 page from the firmware zero region and writes it to every output page of its contiguous range. The page split is folded over the full grid. The input is never bound to the program.
- **SUPPORTED at Phase 0**: dtype = [float32]. Layout TILE, rank 4, tile-aligned last two dims and DRAM interleaved are structural `ValueError` checks, not axes. Both call forms (new output, `optional_tensor`) are supported.
- **Accuracy achieved**: bit-exact. PCC = 1.0, max_abs_err = 0, mean_abs_err = 0, abs RMS err = 0, max ULP = 0, 0 / 196608 elements ≠ +0.0. Relative RMS and the got/true ratio are undefined against an all-zero reference. Measured on the float32 production case in both call forms × 3 input draws (randn, random 32-bit patterns, NaN/inf/-0.0/subnormal) via `test_g6_zeros_like_precision_baseline.py`.
- **Golden suite at Phase 0**: 4 / 4 supported cells pass, 2 `xfail_expected` (bfloat8_b), 8 / 8 regression tests pass. `supported_fail = xpass_drift = xfail_wrong_mode = 0` (`verifier_report.json`).
- **Perf at Phase 0** (profiler device kernel time, median of 12): float32 (1,12,128,128) 3603 ns on 110 cores, against a goal of ≤ 2310 ns and stock 4920 ns. bfloat8_b (5,1,32,256), via direct dispatch, 819 ns on 40 cores.
- **Issues encountered** (verifier fixes):
  - The RT-arg reads and output-accessor setup ran after the zero-fill barrier and outside any profiler zone. Moved into the fill's shadow inside the `g6zl-zero-page` zone: kernel start → first write 253 → 243 ns; end-to-end unchanged within noise.
  - The `ZERO_SCRATCH_PAGES` depth knob did not reach the kernel, which always filled one page. It is now single-sourced through CT arg `zero_scratch_bytes`, which sizes both the CB and the fill.
  - `op_design.md` and `l1_ledger.md` were updated to match.
- **Tests added**: `test_g6_zeros_like_precision_baseline.py`, `test_g6_zeros_like_trace.py` (BB-16 trace capture and replay), and `test_profile_core_knobs_bf8` in `test_g6_zeros_like_perf.py`. Existing: `test_g6_zeros_like.py` (acceptance, 21 tests) and the `test_g6_zeros_like_perf.py` profiling driver.

## Refinement 1 — bfloat8_b
- Date: 2026-11-27
- What was done: added `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`. No kernel or program-descriptor change: the page size already comes from `buffer_aligned_page_size()` (1088 B for bf8b), the CB format follows the output dtype, and all-zero bf8b bytes (shared exponents and mantissas) decode to +0. `test_profile` and `test_profile_core_knobs_bf8` now call the public entry point instead of dispatching the descriptor directly. No `compute_kernel_config`, tagger, EXCLUSIONS entry or placement retune, per the verifier notes.
- Accuracy achieved: bit-exact on bfloat8_b (5,1,32,256) new_output, randn input over a freed ones-filled block: PCC = 1.0, max_abs_err = 0, 0 of 40960 elements ≠ 0. Relative RMS is undefined against an all-zero reference. float32 unchanged (bit-exact).
- Golden test progress: 6 / 6 axis cells (the 2 bf8b cells, `test_op[5x1x32x256-new_output-dtype=BFLOAT8_B]` and its loose twin, moved from `xfail_expected` to pass). Ran the slice of both bf8b cells plus the float32 new_output pair. The float32 into_optional pair was not re-run; its path is unchanged.
- Perf-guard baseline (profiler device kernel time): bfloat8_b (5,1,32,256) new_output through the public entry point, 813.5 ns median of 16 samples (min 789, max 841), 40 cores. Matches Phase 0's 819 ns via direct dispatch.
- Issues encountered: None.
- Tests added: none. The existing `test_bfloat8_b` and `test_precision_bfloat8_b` now take their supported branch.
