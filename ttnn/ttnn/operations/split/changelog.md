# Changelog: split

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier), for
  Quasar on ttsim. One `ttnn.generic_op` dispatch (Metal 2.0 `ProgramSpec`, data-movement kernels only, thread
  plan (1, 0, 1)) writes every chunk. Two regimes: `page_remap` (one input page per output page, implicit sync)
  and `tile_row_shift` (TILE cut inside a tile: scratchpad window + self-aimed row-run assembly, explicit producer).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE, ROW_MAJOR], memory_config=[DRAM, L1],
  sharding=[interleaved], dim=[-2], chunks=[2], last_chunk=[full], alignment=[tile_aligned, w_non_aligned,
  h_non_aligned], rank=[3] (= spec PHASE0)
- **Accuracy achieved**: exact. PCC=1.0, max_abs_err=0.0, rel_rms_err=0.0, max ULP=0, got/true ratio 1.0 everywhere
  (measured on 4 shapes × 2 layouts × 2 placements via test_split_precision_baseline.py)
- **Golden suite at Phase 0**: 36 / 36 in-phase cells passing (36 supported_pass, 228 xfail_expected,
  0 supported_fail / xpass_drift / xfail_wrong_mode; 3 / 3 rejection tests), per `verifier_report.json`
- **Issues encountered** (verifier fixes):
  - `derive_geometry` sent TILE `dim = -1` at a tile multiple (a `page_remap` case) to the `dim <= -3` branch:
    wrong geometry, latent behind `validate()`. Fixed with the design's `dim = -1` page view.
  - `tile_row_shift` clamped blocks to one tile column, so every Phase 0 `tile_row_shift` shape read one input
    tile per DRAM barrier. Blocks now span columns; window `2 × BLOCK_H_TILES` slots indexed by read sequence.
    DRAM read barriers on `(1,4,8400) s2` TILE: 270 → 71; on `(1,34,4096) s17`: 128 → 32 (structural counts;
    the time gain is unconfirmed on ttsim).
  - The writer's chunk cap duplicated the host's `MAX_CHUNKS`: now passed as the define `SPLIT_MAX_CHUNKS`.
  - `tile_row_shift` now refuses non-32×32 tiles on the host (`ASSEMBLY_TILE`). `split()` no longer re-runs the
    REJECTIONS checks.
  - `l1_ledger.md`: the window row, the RM float32 page bound (32,768 B) and the totals were corrected;
    `op_design.md` carries a verifier amendment withdrawing "1 column per block".
- **Tests added**: test_split_precision_baseline.py (16 cases); in test_split_plan.py, a window-wrap case
  `(2,40,8192) s20` and `test_sharded_input_is_refused_by_name` (R6). (test_split.py and the other plan tests
  came from the harness and the implementer.)
