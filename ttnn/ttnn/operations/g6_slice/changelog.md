# Changelog: g6_slice

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). One `generic_op` program per call: reader (NCRISC/NoC0) → `cb_slice_pages` → writer (BRISC/NoC1), no compute kernel; three host-selected regimes (`page_copy`, `row_runs`, `col_realign`); contiguous output-page ranges over the full grid; begins as runtime args so one compiled program serves every begin.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], memory_config=[dram_interleaved], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4], begins_source=[host_list], step_mode=[unit_step]; EXCLUSIONS=[]
- **Accuracy achieved**: PCC=1.0, max_abs_err=0, rms_err=0, max ULP=0, got/true ratio ≡ 1 (bit copy) — measured on 5 shapes covering all three regimes via `test_g6_slice_precision_baseline.py`
- **Golden suite at Phase 0**: 881 supported_pass / 449 xfail_expected / 0 supported_fail / 0 xpass_drift / 0 xfail_wrong_mode, 1349 collected (per `verifier_report.json`; run with `EVAL_CAPTURE_L1=0`, see `verification_report.md` Harness findings)
- **Perf at Phase 0** (Blackhole, 110 cores, 1350 MHz, same session): flagged case 703 2669.6 µs (1.078 × the 2475.8 µs roofline; 1.20 × faster than stock `ttnn.slice` in a same-session pair); 587 (single row) 6.9 µs, ~3 × slower than stock
- **Issues encountered** (verifier fixes):
  - Tile-derived quantities were derived twice (plan and descriptor, with a second `tile = 32` literal); now one source in `_plan()`.
  - `cb_slice_pages` allocated a second slot that is never live when each core holds a single block; depth is now `min(CB_DEPTH, num_blocks_max)`.
  - The kernels' run-list build (~710 cycles each) ran outside every profiler zone (BB-21); added `G6_SETUP`.
  - A host `input_tensor` failed deep in planning; now a structural `ValueError`.
  - The default golden runner (`EVAL_CAPTURE_L1=1`) disables the program cache, so the `Qwen703_trace` loose case fails for any implementation and leaves the device in trace capture for the rest of the run — reported as a harness issue, not an op defect.
- **Tests added**: `test_g6_slice_precision_baseline.py`; in `test_g6_slice.py`: `col_realign` with several blocks per core, single-block CB depth, host-tensor refusal (acceptance tests `test_g6_slice.py`, `test_g6_slice_debug.py`, `test_g6_slice_perf_knobs.py` from the implementer)
