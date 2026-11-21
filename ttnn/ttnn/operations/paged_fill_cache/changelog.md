# Changelog: paged_fill_cache

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). One `ttnn.generic_op` dispatch of a Metal 2.0 `ProgramSpec`: reader + writer DM kernels, no compute kernel; tile-granular split over the full grid (32 clusters); page table read on the device; in place (returns `cache_tensor`).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], memory_config=[DRAM], sharding=[interleaved], alignment=[tile_aligned] (= PHASE0)
- **Accuracy achieved**: PCC=1.0, max_abs_err=0.0, rms_err=0.0 (bit-exact; got/true ratio exactly 1.0; measured on 4 shapes via test_paged_fill_cache_precision_baseline.py)
- **Golden suite at Phase 0**: 15 / 15 supported cells passing, 36 xfail_expected, 11 / 11 rejections, 0 loud (per `verifier_report.json`); queue coverage 0 gaps
- **Issues encountered** (verifier): no drift. DRY fixes applied — page-table ring entry size derived once (`page_table_entry_bytes`), the row-copy count passed to both kernels as CT arg `page_table_row_copies` instead of a literal 2, `TILE` defined once, unused `SKIP` removed; `l1_ledger.md` traffic line corrected (the second page-table read per cluster is removable locally — folded into Refinement 2).
- **Tests added**: test_paged_fill_cache.py (harness), test_paged_fill_cache_plan.py (implementer), test_paged_fill_cache_precision_baseline.py (verifier)

### Implementer notes

Built as designed (op_design.md, regime R-filter); no departure from the Blocking Model or the
L1 ledger (rings `input_tiles` = TILE_RING_ENTRIES x tile_bytes, implicit sync; `page_table_row`
= 2 x align_up(row bytes, DRAM alignment), explicit sync).

- SUPPORTED = PHASE0 (bfloat16, TILE, DRAM, interleaved, tile_aligned). `validate_contract
  --phase0` passes.
- Thread plan (1, 0, 1), host constant `THREAD_PLAN`; no compute kernel.
  `kernels/paged_fill_cache_compute.cpp` is a placeholder the program does not load (the task's
  file list names one; the design has no compute stage).
- Results (ttsim, Quasar): acceptance 4/4; golden suite 26 passed / 36 xfail (float32 and
  L1-input cells, refused) / 0 failed; `test_paged_fill_cache_plan.py` 6/6 (plan read off the
  program, mid-row cluster ranges with a skipped block, uneven split, 8192-wide row over the grid,
  all blocks skipped, one device session with `to_layout` before and after).

#### Finding: thread plan (2, 0, 2)

The kernels stride their walks by the plan's CT counts, but (2, 0, 2) on the STRIDED DM->DM rings
is not admitted (`tests/.../probes/probe_001.py`). The host
admits only the plans in `VERIFIED_THREAD_PLANS` = {(1, 0, 1)}. The design's thread-plan perf lamp
must first establish how a DM->DM STRIDED ring deals entries between multiple reader and writer
threads.

#### Structural counts (simulator: no timing claimed)

Per call, bfloat16: DRAM reads = 2048 x mapped tiles + 2 x engaged clusters x row bytes; DRAM
writes = 2048 x mapped tiles; cache never read; `-1` rows neither read nor written; clusters
engaged = min(32, input tiles).
