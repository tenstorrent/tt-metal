# Changelog: tilize

## Phase 0 — Core Implementation
- **Date**: 2026-09-23
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier).
  - The `row_split_interleaved` regime is one `ttnn.generic_op` dispatch:
    - a custom stick reader (NCRISC / NoC0);
    - one `compute_kernel_lib::tilize` helper call per kernel (fast tilize);
    - a custom TILE-page writer (BRISC / NoC1);
    - work split over the runtime grid with `split_work_to_cores(row_wise=True)`.
  - Implementer perf pass: a per-core DRAM-bank traversal rotation (a win). A transaction-id read-ahead and a split reader were measured flat or slower and parked as live knobs.
  - Verifier: exposed the `tile_row` streaming window as a knob (`QUANTUM_MIN_TILES`, below).
- **SUPPORTED at Phase 0**:
  - `dtype=[bfloat16]`, `output_dtype=[bfloat16]`, `low_l1=[False]`
  - `shard_api=["none"]`, `out_scheme=["interleaved"]`, `buffer=["dram_to_dram"]`, `orientation=["none"]`
  - `rank=[4]`, `pad_mode=["none"]`, `pad_value=["none"]`, `alignment=[tile_aligned]`
  - `tile_height=[32]`, `in_tile_height=["none"]`
  - `tile_grid=[single_tile, small, tall_narrow]`
- **Accuracy achieved**: PCC = 1.0, max_abs_err = 0, mean_abs_err = 0, rms_err = 0, max ULP = 0, 0 bit mismatches. Measured on 5 cases (4 shapes, including one wide-exponent bf16 input) by `test_tilize_precision_baseline.py`.
- **Golden suite at Phase 0**: 30 supported_pass, 1746 xfail_expected, 2310 invalid_skipped, 0 supported_fail / xpass_drift / xfail_wrong_mode, out of 4390 rows (per `verifier_report.json`). `test_regression.py` has 10 tracked failures (fp32 / uint16 / int32 dtypes, not yet in SUPPORTED); Refinement 7 closes them.
- **Perf at Phase 0** (WH B0, 64 Tensix cores, median device-kernel ns): perf-focus [1,1,16384,64] ≈ 25.4–26.1 µs, against the LOOSE_CASES reference of 25998 ns.
- **Issues encountered (verifier fixes)**:
  1. The `tile_row` streaming window was collapsed to one tile-row per CB handshake, read barrier and write flush. It is now the `QUANTUM_MIN_TILES` → `rows_per_quantum` knob, which gives multi-tile-row quanta only where a tile-row is under 8 tiles, with at least `DEPTH_IN` quanta per core. Measured −3 % on [1,1,16384,64] and −5 to −8 % on [1,1,16384,32]; other paths unchanged.
  2. `validate()` mis-tagged `shard_api` at runtime, because an allocated legacy-sharded tensor also reports an `nd_shard_spec`. It now reads `created_with_nd_shard_spec`.
  3. DRY: removed the duplicated `TILE_WIDTH` and tile-grid `(R, C)` code from `tilize.py`, the derived `in_tile_bytes` CT arg, and the duplicated per-column CB byte formula.
  - The ledger and op_design.md block schedule were updated for the new window.
  - Harness issues reported but not edited: `axes.py:_spec_of` has the same ND mis-tag, and the `low_l1` A/B capture records the first leg. Missing INVALID entry: fp8_e4m3 × retile.
- **Tests added**:
  - `test_tilize.py` (acceptance, planner)
  - `test_tilize_knobs.py` (knob matrix; extended with quantum configs and a ragged multi-row shape)
  - `test_tilize_perf_shapes.py`
  - `test_tilize_precision_baseline.py`
  - `test_tilize_registry.py` (runtime shard tagging)
  - `test_tilize_perf_knob_sweep.py` (device-ns A/B harness, for `--profile`)
