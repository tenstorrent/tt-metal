# Changelog: g6_untilize

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier).
  Regime `row_gather`: the reader (NCRISC/NoC0) and writer (BRISC/NoC1) gather 64-B face-row pieces of the
  TILE input straight into row-major positions of one CB, and the writer stores whole ROW_MAJOR pages with
  `write_sticks_after_untilize`. One `generic_op` dispatch on one 8×4 rectangle, no compute kernel.
  Implementer-measured device kernel duration on `(1,1,32,128)`: 2389 → 1910 ns (bank stagger) → 1601 ns
  (gather split across both RISCs and NoCs); incumbent 2.17 µs.
- **SUPPORTED at Phase 0**: dtype=[uint32], layout=[TILE]; no shape-derived or op-specific axes (TARGET
  reached).
- **Accuracy achieved**: bit-exact. 0 mismatches, max ULP 0, PCC = 1.0, max_abs_err = 0, rms_err = 0 on
  4 input distributions of the production shape (full range, odd values above 2^24, sign bit set,
  index-coded) via `test_g6_untilize_precision_baseline.py`. BB-22 limits tests to the production shape.
- **Golden suite at Phase 0**: 6 / 6 passing. Verifier CLI: 2 supported_pass, 0 loud categories
  (`verifier_results/verifier_report.json`).
- **Issues encountered** (fixed by the verifier):
  - `derive_block_tile_cols` bound `ROW_BLOCK_DEPTH` / `L1_OP_BUDGET_BYTES` as default arguments, so a
    tuned knob never reached the block-size derivation. Both are now passed at call time.
  - The descriptor cache key lacked `L1_OP_BUDGET_BYTES` (BB-07). `_knob_key()` is now the single list of
    program-shaping knobs.
  - `validate()` refuses non-32×32 input tiles. Kernel `static_assert`s pin the host tile counts to the CB
    tile and use `NOC_DRAM_READ_ALIGNMENT_BYTES`.
  - The reader's reserve wait has its own profiler zone. `l1_ledger.md` updated to match the code.
- **Tests added**: `test_g6_untilize.py` (acceptance, 11 cases), `test_g6_untilize_precision_baseline.py`
  (4 distributions).
- **Refinement queue**: empty. TARGET − SUPPORTED = ∅, and the only cell (4 tiles) is below the
  perf-measurement threshold of `perf_rules.txt` (see `op_requirements.md`).
