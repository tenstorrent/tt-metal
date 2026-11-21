# Changelog: expand

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Regime R-H
  (tile layout, rows repeated): a Metal 2.0 ProgramSpec with a reader and a writer only (THREAD_PLAN = (1, 0, 1),
  no compute kernel), one explicit-sync DM→DM ring `cb_row_bcast_tiles`, source-major contiguous runs from
  `split_work_to_cores(..., row_wise=True)` over the whole queried grid, one `ttnn.generic_op` per call.
  - Reader: per block, one reserve, one whole-page DRAM read per source, one read barrier, the in-place row-0
    replication (4 doubling steps in faces 0/1 + faces 0-1 → 2-3, each batched over the block, 5 barriers), one push.
  - Writer: per block, one wait, one DRAM write per output tile from the resident entry, one write barrier (plus a
    drain every 32,768 writes so the 16-bit per-trid counter cannot overflow), one pop.
  - Knobs: `block_src_tiles` (default all of the busiest cluster's sources, clamped by the ring cap),
    `buffer_depth` (2, clamped to the blocks a cluster runs), `max_cores` (whole grid).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], memory_config=[DRAM], sharding=[interleaved],
  changes=[h], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2]
- **Accuracy achieved**: PCC=1.0, max_abs_err=0.0, rel_rms_err=0.0, ULP=0 — bit-exact on 4 shapes (measured via
  test_expand_precision_baseline.py), special values included.
- **Golden suite at Phase 0**: 9 / 9 supported cells passing (8 PHASE0 + the loose case), 248 xfail_expected,
  X1–X3 pass; 0 supported_fail / xpass_drift / xfail_wrong_mode (per `verifier_report.json`).
- **Structural counts** (quasar, traced call `(1,2560)→(1330,-1)`): 32 clusters, 105 output tiles each, input page
  reads 96 for 80 sources (bound `num_sources + cores − 1`), output written once, 1 block per cluster (3 sources),
  ring 3 entries (6,144 B). Timing not measured (ttsim has no clock).
- **Issues encountered** (verifier fixes):
  - Ring depth clamped to `min(buffer_depth, max_blocks_per_core)`: with one block per cluster the second ring half
    was never live (L1 ledger audit, capacity vs live set). Traced call 12,288 B → 6,144 B; ledger updated.
  - `validate()` names every axis outside SUPPORTED in one refusal, so a sharded L1 input is refused on `sharding`
    by name (R7), not only on `memory_config`.
- **Tests added**: test_expand.py (generated acceptance), test_expand_plan.py (knobs, structural counts, one-session
  tests, refusals; verifier added R6 placement, R7 sharded refusal, ring-depth count),
  test_expand_precision_baseline.py.
