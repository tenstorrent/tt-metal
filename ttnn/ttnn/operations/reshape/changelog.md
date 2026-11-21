# Changelog: reshape

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Two Metal 2.0 `ProgramSpec` regimes, one `ttnn.generic_op` per call. `page_identity` copies pages one-to-one (row-major with W kept; tile layout when the tiles keep their rows). `tile_row_regroup` moves face-row runs between tiles when H changes in tile layout. Full grid (32 clusters on quasar), thread plan (1, 1, 1), no compute kernel.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE, ROW_MAJOR], memory_config=[DRAM, L1], sharding=[interleaved], changes=[leading, h], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[1, 3, 4]
- **Accuracy achieved**: bit-exact. PCC=1.000, max_abs_err=0, rms_err=0 (relative), ULP p99=0, got/true ratio 1.0 everywhere, measured on 4 shapes covering both regimes and both placements via `test_reshape_precision_baseline.py`.
- **Golden suite at Phase 0**: 124 / 644 passed (121 `supported_pass` incl. the loose case, 3 rejection tests); 520 `xfail_expected`; 0 `supported_fail` / `xpass_drift` / `xfail_wrong_mode` (per `verifier_report.json`).
- **Issues encountered** (verifier):
  - No drift and no failing cells.
  - DRY fix: the relayout kernel's tile geometry literals (`TILE_ROWS = 32`, `FACE_ROWS = 16`) now come from the host constants as compile-time arguments.
  - Added the R8 sharded-input refusal test, which the spec says the op carries itself.
- **Tests added**: `test_reshape.py` (generated acceptance), `test_reshape_regimes.py` (regime-pinned, forced blocks, engaged clusters, thread plan, device-session tests; verifier added `test_sharded_input_refused_by_name`), `test_reshape_precision_baseline.py`.
- **Structural baseline for the perf queue** (ttsim, no clock):
  - Regime A issues 1 NoC transaction per page per side (1,600 + 1,600 on `(1,1,1600,144)` RM).
  - Regime B runs 1 block per core on every regroup input.
  - Regime B reads 1.00–1.75× the minimum number of input tiles (28 / 16 on `(2,1,100,47)→(2,4,25,47)`).
  - Regime B does 1 output-ring reserve/push per output tile.
  - The regime B relayout copies every output byte on the CPU.
