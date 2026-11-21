# Changelog: layer_norm

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier) for
  Quasar on the ttsim simulator: one Metal 2.0 `ttnn.ProgramSpec`, one `ttnn.generic_op` dispatch per call.
  - Regime `row_resident`: a whole tile-row resident on one NEO; two-pass centred variance; DEST-accumulated row
    sums plus one-tile reduces.
  - Thread plan `(4, 4, 2)`, clamped per call.
  - Rows spread over all 32 clusters, then over their NEOs.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True, False], layout=[TILE], memory_config=[DRAM],
  sharding=[interleaved], affine_dtype=[bfloat16], affine_layout=[TILE], alignment=[tile_aligned], rank=[4]
  (equals the specification's PHASE0)
- **Accuracy achieved** (measured on 4 shapes × 2 accumulators via `test_layer_norm_precision_baseline.py`):
  - fp32 DEST: PCC ≥ 0.999998, max_abs_err ≤ 0.040, rel_rms_err ≤ 0.0019 (ULP p99 ≤ 4).
  - 16-bit DEST: PCC ≥ 0.999993, max_abs_err ≤ 0.107, rel_rms_err ≤ 0.0043 (ULP p99 ≤ 20, at W = 4096).
  - got/true ratio median 1.000 ± 0.001 in every case (no scale bug).
- **Golden suite at Phase 0**: 26 supported_pass, 0 supported_fail, 522 xfail_expected, 0 xpass_drift, 0 xfail_wrong_mode;
  rejections X1/X2 pass; queue_gaps 0 (per `verifier_report.json`)
- **Issues encountered** (verifier fixes, no drift):
  - Folded the `rstd` ring into `row_partial` as its third FIFO use: their lifetimes are disjoint, and the only
    justification given was "debug path". This takes the op from 12 rings to 11 and saves bh·t_acc per NEO.
  - Removed the duplicated `Pages.t_sc` (it was identical to `t_acc`) and fixed the ledger's misstated scaler page.
  - Moved the row-dealing rule (`rounds`, `row_of`) into one header, `kernels/layer_norm_rows.hpp`, shared by
    reader, compute and writer.
  - Brought `l1_ledger.md` up to date with the code.
  - Probes found that `w_non_aligned` would violate rule R6 once enabled (the padded columns enter the variance:
    PCC 0.955 on uniform data at W = 17). It is correctly refused today and is Refinement 1's core work.
- **Tests added**: `test_layer_norm.py` (harness-generated acceptance), `test_layer_norm_interface.py`,
  `test_layer_norm_plan.py` (threads ran, plan selection, session ordering), `test_layer_norm_precision_baseline.py`
