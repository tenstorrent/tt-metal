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

## Refinement 1 — Sharded and L1 placement (legacy 2-D, ND, crossovers, DRAM-sharded) + rank widening
- **Date**: 2026-09-23
- **What was done**:
  - Built the `sharded_resident` and `sharded_accessor` regimes on the existing reader / compute / writer (no new kernel file, no second descriptor branch).
  - **Core assignment and residency.** `tilize_program_descriptor._core_assignment` sets the core assignment from the first of:
    - a `resident_ok` L1-sharded output;
    - a `resident_ok` L1-sharded input;
    - the interleaved row split.

    Each side is then resident iff its shard rectangles (`_shard_rects`, all three schemes, both orientations, ragged final shards, ND specs with a 2-D equivalent) equal the assignment. A resident input backs `cb_input_sticks` on the shard, and the reader only publishes pages. A resident output backs `cb_output_tiles`: compute packs into it and the writer issues no NoC write. With the same spec on both sides the op moves zero NoC bytes.
  - **Streamed sides** (interleaved DRAM/L1, DRAM-sharded, cross-spec remote shards, ND without a 2-D equivalent) go through `TensorAccessor`. WIDTH / BLOCK / ND Layout::ROW_MAJOR inputs have shard-width pages, so the stick-segment reads split at page boundaries (new `page_bytes` / `pages_per_stick` CT args; the one-page-per-stick fast path is unchanged).
  - **Rotation.** A resident side pins the tile-row walk to shard order, so the per-core rotation is now two RT args, `row_rotation` and `stick_rotation`. The writer rotates its tile order inside a tile-row by `stick_rotation` (the write twin of the reader's stick rotation): HEIGHT-sharded in → DRAM [1,1,2048,512] went from 18746 to 16502 ns, and the interleaved paths are within noise.
  - **SUPPORTED** gains:
    - `shard_api` legacy_2d / nd;
    - `out_scheme` HEIGHT / WIDTH / BLOCK / nd;
    - `orientation` ROW_MAJOR / COL_MAJOR;
    - `buffer` dram_to_l1 / l1_to_l1 / l1_to_dram;
    - `rank` 2 / 3 / 5 / 6;
    - `tile_grid` short_wide / square_large, with EXCLUSIONS refusing both wherever the row split would be the core assignment (Refinement 5 lifts those).
  - Reused: Walker, StickProducer, store_rows, the tilize helper call, CB slots 0 / 1, `balanced_width` / `rows_per_quantum`. Added: shard-rectangle mapping, CB backing via `cb_descriptor_from_sharded_tensor`, per-page segment reads, `input_resident` / `output_resident` CT modes, the write-order rotation.
- **Accuracy achieved**: bit-exact (`torch.equal`, PCC = 1.0, atol = rtol = 0) at bf16 on every sharded / L1 / rank shape tested. That covers 26 unit cases (HEIGHT / WIDTH / BLOCK × ROW / COL, ragged shards, cross-spec, DRAM-sharded, ND 3-D, ranks 2–6) plus the golden and translated cells below.
- **Golden test progress**:
  - `test_golden.py`: 32 passed / 758 xfailed / 2310 skipped / 0 failed / 0 XPASS; 24 passes are new (Phase 0: 8 in this file).
  - Translated sharded / ND / L1 slice: 191 passed / 1 failed (below).
  - Sharded LOOSE_CASES, WH device-kernel ns (Tensix cores):

    | Case | Measured | Reference |
    |---|---|---|
    | HEIGHT in → DRAM | 16502 (64) | 16852 |
    | DRAM → HEIGHT out | 12269 (64) | 12142 |
    | Same spec | 1914 (64) | 1891 |
    | BLOCK COL_MAJOR | 1971 (16) | 1832 |

  - Perf focus [1,1,16384,64]: 25369 ns on 64 Tensix cores (unchanged).
- **Issues encountered**:
  - `eval/golden_tests/tilize/test_translated.py::test_tilize_program_cache_addr_change[sharded_width_l1]` asserts the first call builds exactly one program. In module order, `test_tilize_col_major_orientation`'s WIDTH case (COL_MAJOR on a 1×4 row grid, which places bytes identically to ROW_MAJOR) has already built the byte-identical program, so the first call is a correct hit (entries = 0). The test passes when run alone. I did not add an artificial CT arg to force a miss.
  - The translated square_large ND cases ([23,96,160]) and the fp32 / padded variants stay refused until Refinements 4, 5 and 7.
  - `test_tilize_registry.py` refusal assertions for sharded inputs became tag assertions, since sharding is now supported.
- **Tests added**: `tests/ttnn/unit_tests/operations/tilize/test_tilize_sharded.py` (19 regime cases with residency assertions, a same-spec zero-traffic assignment check, rank 2/3/5/6 + L1 interleaved, L1 crossovers).
