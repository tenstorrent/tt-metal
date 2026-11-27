# Changelog: g6_pad

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier). Two regimes, one device program per call:
  - `extension_band`: an in-place band write over both DM RISCs, with a metadata-only view as the output.
  - `new_storage_tiles`: an NCRISC reader plus a BRISC writer that assemble each output page once in L1.
- **SUPPORTED at Phase 0** (after the verifier drift fix):
  - dtype = [bfloat16, float32, int32, uint32], layout = [TILE], memory_config = [dram, l1];
  - alignment = [tile_aligned, w_non_aligned, h_non_aligned], rank = [2, 3, 4], padded_dims = [leading, h, w, hw].
- **Accuracy achieved**: PCC = 1.0, max_abs_err = 0, rms_err = 0, max ULP = 0. This is bit-exact, measured on 4 shapes (bf16 extension ×2, bf16 new_storage with a −2^127 fill, fp32 p1) via `test_g6_pad_precision_baseline.py`.
- **Golden suite at Phase 0**:
  - 78 / 85 tagged cells pass, with 7 `xfail_expected` (all ROW_MAJOR) and 0 loud categories, with `EVAL_CAPTURE_L1=0`.
  - The default runner shows 74 pass plus 4 `supported_fail`. Those 4 are a harness L1-profiling artifact on host-side refusals (see verification_report.md).
- **Perf at Phase 0** (BH, device ns): p1 28,327 (goal < 29,910), p2 11,894, p3 5,050, p4 10,189.
- **Issues encountered / fixed by the verifier**:
  - Three stale test files from an earlier attempt broke collection of the test directory. Two were rewritten against the current API and the redundant profile test was removed.
  - The L1 cap hardcoded the buffer depth `2`; it now uses the `input_depth_blocks` knob (DRY).
  - SUPPORTED drift was fixed (`xpass_drift` on the sharded-refusal cell). Probe evidence showed 17/17 gated TILE cells passing, plus int32 TILE.
  - `l1_ledger.md` was brought up to date (the 32-bit cap is 164, the cap expression depends on the depth knob, and the page formats cover all dtypes).
- **Tests added**:
  - `test_g6_pad.py` (acceptance, by the planner);
  - `test_g6_pad_extra_cases.py` and `test_g6_pad_knobs.py` (rewritten; knob test covers the multi-block depth-2 path);
  - `test_g6_pad_dtypes.py` (fp32 / int32 / uint32 / rank 2 / L1);
  - `test_g6_pad_precision_baseline.py`;
  - `test_g6_pad_perf_ab.py` (p1 added).
