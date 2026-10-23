# Changelog: rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-10-23
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier).
  - One `ttnn.generic_op` dispatch per call.
  - Regime R1 (`row_local`): tile-rows split across up to the full 120-core grid.
  - Regime R2 (`w_split_root_combine`): W split across row-major W-groups, combined by a unicast gather to the
    group root, a root `reduce<AccumulateViaAdd, ReduceWithinTile::Skip>` with a fused mean / +ε / rsqrt, and a
    `ttnn.Mcast` multicast back to the group.
  - Native TILE and ROW_MAJOR layouts (in-kernel tilize / untilize).
  - Non-aligned W via a partial reduce scaler on the last tile, plus zero-fill of the RM tail.
  - Single-sourced blocking plan (`_select_blocking` → `BlockingPlan`).
- **SUPPORTED at Phase 0**:
  - dtype = [float32, bfloat16]; fp32_dest_acc_en = [True]; layout = [TILE, ROW_MAJOR]
  - alignment = [tile_aligned, w_non_aligned, h_non_aligned]; rank = [2, 3, 4]
  - gamma_mode = [gamma, no_gamma]; gamma_dtype = [float32, bfloat16, "none"]; gamma_layout = [TILE, ROW_MAJOR, "none"]
  - memory_layout = [INTERLEAVED]
  - EXCLUSIONS = [{float32, fp32_dest_acc_en=False}] (mandated)
- **Accuracy achieved** (`test_rms_norm_precision_baseline.py`, 4 shapes × 2 dtypes × 2 layouts, HiFi4, fp32 DEST):
  - bf16: PCC = 0.9999985, max_abs_err ≤ 0.042, rel_rms_err ≤ 0.0020, within 1 bf16 ULP (p50 = 0).
  - fp32: PCC = 0.9999997, max_abs_err ≤ 0.031, rel_rms_err ≤ 0.0019, ≈ 1.5e4 fp32 ULP p50, −0.13 % median bias.
    This is TF32-class: the FPU operand registers truncate fp32 operands. It is not a scale bug (confirmed by CPU
    emulation).
- **Golden suite at Phase 0**: 737 `supported_pass` / 0 `supported_fail` / 0 `xpass_drift` / 0 `xfail_wrong_mode`;
  6180 `xfail_expected`, 33900 `invalid_skipped` (per `verifier_report.json`).
- **Issues encountered** (verifier fixes):
  - `INPUT_TAGGERS` registered `tag_gamma_dtype` / `tag_gamma_layout`, so the golden cartesian never iterated
    gamma formats. All 1680 cartesian gamma cells were skipped as INVALID. Fixed by registering only the
    shape-derived taggers; `supported_pass` went from 177 to 737, and every newly exercised gamma cell passes.
  - The L1 solve (`_footprint`) and the allocation (`_build_cbs`) restated every CB size, so the x-depth knob was
    read from two different sources. Fixed with one `_cb_specs()` definition used by both.
  - `epsilon` was not validated (design contract: finite, ≥ 0). It now raises `ValueError`.
- **Tests added**: `test_rms_norm.py` (acceptance, planner), `test_rms_norm_debug.py` (implementer),
  `test_rms_norm_precision_baseline.py` (verifier).
