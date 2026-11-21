# Changelog: slice_write

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Regime `shard_owner_writes` (details below).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[ROW_MAJOR], memory_config=[L1], sharding=[height], step=[ones], written_region=[outer], alignment=[tile_aligned, h_non_aligned], rank=[4]
- **Accuracy achieved**: bit-exact. PCC=1.000, max_abs_err=0.0, rms_err=0.0, got/true ratio 1.0 everywhere (measured on 4 shapes via test_slice_write_precision_baseline.py)
- **Golden suite at Phase 0**: 14 / 14 supported cells passing (17/244 incl. 3 rejection tests; 167 xfail_expected, 60 invalid_skipped, 0 loud) per `verifier_report.json`
- **Issues encountered (verifier fixes)**:
  - The W writer-hart knob raised `ValueError` when W did not divide the shard rows. It now falls back to W = 1, as `op_design.md` specifies.
  - The common run-time arg names are now derived from the one value dict (DRY).
  - The ledger symbol table was updated to match.
  - No drift.
- **Tests added**: test_slice_write.py (harness), test_slice_write_plan.py (implementer, plus the verifier's W-fallback test), test_slice_write_precision_baseline.py

## Phase 0 — implementer notes
Built as designed: L1 height-sharded bfloat16 row-major source, whole `[W, C]` planes (`outer`),
destination DRAM interleaved, written in place and returned as the caller's object.

- Ring `dfb_src_shard` borrowed onto the resident shard (zero-copy, 0 B allocated); shard `i` →
  `corerange_to_cores(grid, None, row_wise = orientation == ROW_MAJOR)[i]`; cores without valid rows
  are left out of the work unit.
- Reader publishes the entries (no NoC transfer); writer walks the page map incrementally, one
  `async_write<TXN_ID>` (trid 0) per row, one barrier per `rows_per_write_batch = min(32, entry_rows)`,
  hands trid 0 back at exit.
- Thread plan `THREAD_PLAN = (1, 0, 1)` (one host constant). No compute kernel (spec: computes nothing;
  design C = 0), so there is no `slice_write_compute.cpp`.

Deviation (advisory part of the design): the design wrote `src_shard_entries = 1` and a writer loop
striding rows by `writer_threads`. On a STRIDED ring a writer hart only sees its own entries, so with one
entry a second hart would wait forever. The ring therefore has `src_shard_entries = W` entries of
`shard_rows / W` rows, hart `t` storing entry `t`: identical at the Phase 0 plan (W = 1), and the
W-writer lamp is a one-constant turn (tested at W = 1, 2, 4 in `test_slice_write_plan.py`).

Structural counts (ttsim: no timing claims): DRAM reads 0; DRAM writes = block bytes once; one write per
row; barriers per core = ceil(rows_this_core / 32); 1 ring handshake per core. Traced call 1 with the
golden-suite shard config: 213 rows / 7 barriers per core on 32 cores; with the caller's single shard:
6785 rows / 213 barriers on 1 core (the `spread_beyond_shard_grid` regime, deferred).

Tests: acceptance 4/4; golden suite 17 passed, 167 xfail (out of PHASE0), 60 skipped (INVALID), 0 failed;
`test_slice_write_plan.py` 9/9 (plan off the spec, traced single shard, ragged last shard, grid wider than
the data, COL_MAJOR grid, W = 1/2/4, tilize → slice_write ×2 → to_layout tile and back in one session).
