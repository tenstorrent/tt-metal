# Changelog: g6_to_layout

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier). It is a dataflow-only TILE → ROW_MAJOR conversion with two regimes:
  - R1 `staged_reorder`: live face-row prefixes are read in trid-pipelined batches, reordered by 32-B self-aimed NoC copies into a row block, and written as one write per live row segment.
  - R2 `face_direct`: for one tile column with `W ≤ 16`, faces 0/2 are read and each row is written as `2·W` bytes straight from its face row.
  - Work is split over the full grid by a flattened row-major tile range (`split_work_to_cores`, `row_wise=True`).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], output_dtype=[bfloat16], memory_layout=[INTERLEAVED] (DRAM or L1), alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4]
- **Accuracy achieved**: bit-exact. PCC=1.0, max_abs_err=0.0, mean_abs_err=0.0, rel_rms_err=0.0, max ULP=0, got/true ratio 1.0 (p5–p95 1.0–1.0). All injected specials (-0.0, subnormals, ±inf, NaN payload) are preserved. Measured on 4 shapes (c309, c1993, c286, c2020) via `test_g6_to_layout_precision_baseline.py`.
- **Golden suite at Phase 0**: 58 / 89 registry cells passing (49/73 INPUTS + 9/16 LOOSE_CASES); 31 `xfail_expected`, all queued in Refinements 1–2; 0 supported_fail / xpass_drift / xfail_wrong_mode; 2/2 regression tests pass (`verifier_report.json` in `generated/g6_verify/`).
- **Perf at Phase 0** (device kernel, 110 cores): 308 (perf focus) 3331–3432 µs; 304 3008 µs; 1978 1762 µs; 286 785–794 µs. 308 is bound by its 4.19 M 6-B DRAM writes: writes alone 2167 µs, reads alone 517 µs.
- **Issues encountered**: the implementer's work for this plan was uncommitted in the working tree; it is committed with the verifier pass. Verifier fixes:
  - the R2 slot size is now defined once in the shared kernel header (it was duplicated in the R2 reader and writer);
  - the `pipeline_blocks_per_core = 16` departure from the coarsest-fit block was re-measured and recorded in `l1_ledger.md` (5–6% faster than the L1-cap block on R1).
- **Tests added**: `test_g6_to_layout.py` (acceptance, 31), `test_g6_to_layout_perf.py` (production perf shapes), `test_g6_to_layout_precision_baseline.py` (8).
