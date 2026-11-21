# Changelog: paged_update_cache

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Regime `direct_row_write`: one Metal 2.0 `ProgramSpec` dispatch (`ttnn.generic_op`) of one data-movement kernel (`row_writer`, thread plan (0, 0, 1)) on each core holding a user's input shard. Index reads (positions page + page-table row) under one trid-0 read barrier; 2·heads·dtiles face-row writes from the resident shard into the cache pages at byte offsets under one write barrier; trid hand-back. The cache is the caller's tensor, written in place and returned. Target: Quasar on ttsim.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], memory_config=[L1], sharding=[height], users=[1], alignment=[h_non_aligned]
- **Accuracy achieved**: exact. PCC=1.0, max_abs_err=0, rel_rms=0, 0 ULP (measured on 4 shapes via test_paged_update_cache_precision_baseline.py, whole cache and written region)
- **Golden suite at Phase 0**: 18 / 18 supported cells pass, 19 xfail_expected, 11 / 11 rejections, 0 loud (per `verifier_report.json`)
- **Issues encountered**: verifier fixes: (1) the tile/face geometry the kernel hard-coded now comes from the host constants as CT args (DRY); (2) the `l1_ledger.md` lifetime of `index_scratch` was corrected (live through the writes). There was no drift.
- **Tests added**: test_paged_update_cache.py (generated acceptance), test_paged_update_cache_plan.py (session order, off-origin shard, scratch-cap refusal, W = 2/3 thread plan, several users), test_paged_update_cache_precision_baseline.py
