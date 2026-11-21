# Changelog: embedding_gen

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier) for Quasar (Metal 2.0 `ProgramSpec`, one `generic_op` dispatch). Scheme: gather-direct, multi-core. A reader places each selected table piece straight into the output layout in a ring entry, and a writer stores the entry. No compute kernel; thread plan (1, 0, 1).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE, ROW_MAJOR], memory_config=[DRAM], sharding=[interleaved], output_layout=[TILE, ROW_MAJOR], output_memory=[None, DRAM], embeddings_type=[generic, padded, binary], alignment=[tile_aligned, h_non_aligned], rank=[2, 4], table_rank=[4] (= PHASE0)
- **Accuracy achieved**: PCC=1.0, max_abs_err=0.0, rel_rms_err=0.0, max ULP=0 — bit-exact (measured on 4 shapes covering all four table→output layout pairs via test_embedding_gen_precision_baseline.py)
- **Golden suite at Phase 0**: 96 / 96 supported cells passing + 5 / 5 rejections (per `verifier_report.json`); 800 xfail_expected, 448 skipped (bfloat8_b), 0 loud, 0 queue gaps
- **Issues encountered** (verifier code-review fixes, no drift):
  - Block geometry was defined twice (reader function + writer inline). It is now one header, `kernels/embedding_gen_common.hpp`, shared by both kernels.
  - The face-halves count was a literal `2` in the reader. It is now the host-derived CT arg `faces_per_row = TILE_WIDTH // FACE_WIDTH`.
  - The reader's next-index-slice prefetch moved ahead of `reserve_back`, so it is hidden under the ring wait as well as the gather.
- **Tests added**: test_embedding_gen.py (harness acceptance), test_embedding_gen_plan.py (thread plan, sub-modes, R2 guard, one device session), test_embedding_gen_precision_baseline.py
