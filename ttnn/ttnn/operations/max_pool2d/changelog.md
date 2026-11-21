# Changelog: max_pool2d

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier), for Quasar on the ttsim simulator. It is a Metal 2.0 `ProgramSpec`, dispatched once per call. Input rows are staged once per block into a scratchpad, or used in place from an owned width shard. Window taps are gathered by core-local copies into stream tiles and max-folded in DEST. Rows are written straight into the L1-sharded result (R9). Thread plan (4, 4, 2); output shards × whole-round position sub-ranges across the 8×4 grid.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True], layout=[TILE, ROW_MAJOR], memory_config=[DRAM, L1], sharding=[interleaved, height, width], batch=[single], channel_alignment=[tile_aligned], window=[overlapping], pad=[padded], dilate=[none]
- **Accuracy achieved**: exact on every shape (PCC=1.0, max_abs_err=0.0, rel_rms=0.0, 0 ULP), measured on 4 shapes via test_max_pool2d_precision_baseline.py
- **Golden suite at Phase 0**: 72 / 72 non-skipped, non-xfail tests passing (68 supported_pass, 896 xfail_expected, 480 invalid_skipped, 0 supported_fail / xpass_drift / xfail_wrong_mode), per `verifier_report.json`
- **Issues encountered**:
  - Verifier fix, rule R9: the result's shard orientation followed the input's. It is now always row-major, and an owner pools in place only when it also holds the matching input shard. Shown by a probe with col-major 2×4-grid inputs; covered by the new `test_col_major_input_gives_row_major_result`.
  - Removed the unused host constant `FACE`.
  - Corrected two `l1_ledger.md` statements: the IN_PLACE-only staging size and the `window_taps` page-format reason.
- **Tests added**: test_max_pool2d.py (generated acceptance test), test_max_pool2d_plan.py (thread plan, one-session, R10, regimes, plus the verifier's orientation test), test_max_pool2d_precision_baseline.py
