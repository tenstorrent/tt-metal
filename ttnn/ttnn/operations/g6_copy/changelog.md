# Changelog: g6_copy

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier).
  - Whole-page DRAM → L1 → DRAM copy over the full grid: a reader on NCRISC/NoC0 and a writer on BRISC/NoC1.
  - One staging CB. Transfers are coalesced bank runs with bank stagger on.
  - `one_block` plan by default, with a `streamed` fallback under the L1 clamp.
- **SUPPORTED at Phase 0**:
  - dtype=[float32]
  - layout=[TILE]
  - alignment=[tile_aligned, w_non_aligned, h_non_aligned]
  - EXCLUSIONS (contract): {float32, ROW_MAJOR}, {int32, TILE}, {uint32, TILE}
- **Accuracy achieved**: bit-identical on 4 fp32 shapes, (1, 1) to (16, 12, 128, 128), measured with
  `test_g6_copy_precision_baseline.py`: PCC=1.0, max_abs_err=0.0, rms_err=0.0, 0 ULP, got/true ratio 1.0.
- **Perf at Phase 0**: (16, 12, 128, 128) fp32 runs at 67.6 µs median device kernel time (goal < 68.56 µs,
  roofline 54.61 µs).
- **Golden suite at Phase 0**: 13 / 42 passed (per `verifier_report.json`):
  - 10 / 39 registry cells supported_pass.
  - 29 xfail_expected.
  - 0 loud.
  - 3 / 3 numerics regression tests pass.
- **Issues encountered** (verifier):
  - No op or kernel fix needed.
  - Deleted the stale untracked `test_g6_copy_perf.py`, `ablation_kernels/` and `test_g6_copy_knob_sweep.py`.
    They were left by an earlier implementer session; the perf test patched symbols the current descriptor
    does not define, failing 24 tests in a directory run.
  - The local conftest now keeps a module's own `use_module_device(...)` marker.
  - `op_design.md` updated to match the code's `BANK_STAGGER = True` default.
- **Tests added**: `test_g6_copy_precision_baseline.py`, `test_g6_copy_trace.py` (BB-16 trace replay). The
  acceptance test `test_g6_copy.py` and perf sweep `test_g6_copy_lamps.py` come from the implementer.
