# Changelog: concat

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Metal 2.0 `ttnn.ProgramSpec`, reader + writer only, one ring, two regimes (`page_copy`, `run_assembly`), full-grid split of the flat output-tile axis.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], memory_config=[DRAM, L1], sharding=[interleaved], dim=[-1, -2], count=[2, 3], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[3, 4] (= PHASE0)
- **Accuracy achieved**: PCC=1.000, max_abs_err=0, rms_err=0, ULP=0 — bit-exact (measured on 6 shapes covering both regimes via test_concat_precision_baseline.py)
- **Golden suite at Phase 0**: 56 / 56 in-SUPPORTED cells passing; 1024 xfail_expected, 180 skipped (INVALID + bf8b unbuildable on quasar); 0 supported_fail / xpass_drift / xfail_wrong_mode (per `verifier_report.json`)
- **Issues encountered** (verifier fixes): added the missing R6 test (a height-sharded L1 input is refused by name, `test_sharded_refused_by_name`); output layout now follows the inputs (`first.layout`, spec OUTPUT) instead of a hard-coded TILE; `L1_RESERVE_BYTES` row added to the ledger's symbol table. No drift fixes.
- **Tests added**: test_concat.py (generated acceptance), test_concat_regimes.py (implementer; + R6 test by the verifier), test_concat_precision_baseline.py (verifier)
- **Structural counts at Phase 0** (simulator; no timing claim): NoC reads per output tile ≈1 in `page_copy`; 1.7–7.3 in row-granular `run_assembly`; 56–101 (18–28 B each) in element-granular `dim=-1` `run_assembly` — 100.7 on `(4,8,32,47)+(…,100)+(…,50)` (Refinement 3's baseline). Reader harts on `run_assembly`: 1 (Refinement 5's baseline).

### Implementer notes (Phase 0)


Built from op_design.md: Metal 2.0 `ttnn.ProgramSpec`, reader + writer only (thread plan (R, C, W) = (1, 0, 1),
host constant `THREAD_PLAN`), one ring `cb_output_tiles`, flat `out_tile` axis split over the whole grid in contiguous
ranges. Regimes `page_copy` (implicit-sync whole-page reads) and `run_assembly` (explicit sync, sub-page reads tagged
trid 0 straight into ring entries, one barrier per block), selected by `select_regime` from the shapes alone.

Knobs, each defined once in `concat_program_artifacts.py`: `BLOCK_TILES = 8` (CT `block_tiles`, clamped by the ring
cap, the L1 budget and the largest core share), `OUTPUT_DEPTH = 2` (ring = `OUTPUT_DEPTH · block_tiles` entries),
`THREAD_PLAN`. `concat_with_block_tiles` turns the block knob for tests.

Structural counts (simulator: no timing claim): 1 read + 1 write per output tile in `page_copy`; in `run_assembly`
1 reserve / 1 barrier / 1 push per block of `block_tiles` tiles, writes 1 per tile, reads merged to whole faces or
whole tiles wherever source and destination are contiguous (an aligned tile is one 2,048 B read).

Deviations from the design (advisory parts):
- `run_assembly` skips cross-axis tile padding instead of reading it into output padding (fewer bytes and transactions).
- `page_copy` finds a tile's piece by a scan over the ≤ 5 pieces per tile rather than an incremental pointer.

Tests: acceptance 7/7; `test_concat_regimes.py` 67/67; golden suite 60 passed, 0 failed, 0 hangs.
