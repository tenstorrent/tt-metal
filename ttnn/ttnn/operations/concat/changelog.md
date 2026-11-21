# concat changelog

## Phase 0 (ttnn-implementer)

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
