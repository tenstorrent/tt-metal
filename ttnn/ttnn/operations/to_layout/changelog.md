# Changelog: to_layout

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Three
  regimes, each one `ttnn.ProgramSpec` and exactly one `ttnn.generic_op` dispatch per call: `tilize_stream`
  (RM → TILE), `untilize_stream` (TILE → RM), `identity_copy` (same layout, DM-only). 2-D independent split
  (tile-row × `bw`-tile column block) over the full grid, thread plan (1, 1, 1). Target Quasar on ttsim.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE, ROW_MAJOR], target_layout=[TILE, ROW_MAJOR],
  memory_config=[DRAM, L1], sharding=[interleaved], alignment=[tile_aligned, h_non_aligned], rank=[4]
- **Accuracy achieved**: bit-exact. PCC=1.000, max_abs_err=0.0, rel_rms_err=0.0, max ULP=0, got/true ratio = 1.0
  (measured on 4 shapes × 4 regimes via test_to_layout_precision_baseline.py)
- **Golden suite at Phase 0**: 120 / 120 supported cells passing; 1656 xfail_expected, 296 invalid_skipped,
  0 supported_fail / xpass_drift / xfail_wrong_mode (per `verifier_report.json`)
- **Issues encountered**: no drift and no supported-cell failure. The verifier added the R6 sharded-input refusal test the
  spec requires the op to carry. It also found a latent stick-transfer overrun for W % 32 != 0 (outside SUPPORTED,
  refused today) and made its fix a requirement of Refinement 4.
- **Tests added**: test_to_layout.py (harness), test_to_layout_plan.py (thread plan, session, worked table, split, sharded
  refusal), test_to_layout_precision_baseline.py
