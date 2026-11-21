# permute changelog

## Phase 0 (2026-11-21): the first phase, built from op_design.md

Status: acceptance test 13/13; golden suite 136/136 of the PHASE0 cells, the loose case and the
rejections (0 failed, 0 hangs; the remaining 508 cells are outside SUPPORTED and xfail);
`tests/ttnn/unit_tests/operations/permute/test_permute_plan.py` 52/52 (ttsim, Quasar).

What was built, against the design:

- **All three regimes, as designed** (`select_regime` pinned as written): `page_gather`
  (reader → `cb_pages` → writer, implicit sync both ends), `tile_transpose` (reader →
  `cb_input_tiles` → NEO `transpose_block` → `cb_output_tiles` → writer) and `staged_gather` (one DM
  kernel, 6 worker harts, per-thread scratchpad regions, tagged trid-0 transfers, one read barrier +
  one L2 invalidate per box, flushed output slots, one writes-flushed wait per `out_slots` pages).
  Host program: `permute_program_artifacts.py` (`ttnn.ProgramSpec`); one `ttnn.generic_op` per call.
- **Thread plans** from one constant, `THREAD_PLANS`: page_gather (1,0,1), tile_transpose (1,1,1),
  staged_gather `GATHER_THREADS` = 6 (clamped by L1 fit, never below 1). The kernels size their loops
  from the plan's compile-time counts, so a missing thread shows as missing output
  (`test_every_gather_thread_ran`: 192 boxes over 32 clusters × 6 harts, sentinel-filled output).
- **Knobs live**: `PAGE_RING_DEPTH`, `TRANSPOSE_BLOCK_TILES`, `TRANSPOSE_RING_BLOCKS`, `OUT_SLOTS`,
  `GATHER_THREADS`, the plans and `choose_gather_block()`'s extents are all builder arguments, and
  `test_permute_plan.py` runs each at other values (plans (2,0,2), (3,0,3), (2,2,2), (1,4,1),
  (4,4,2); block 1/3/8; forced ragged boxes; 1–3 gather threads; 1–2 slots), all exact.

Deviations (advisory items, recorded):

- `page_ring_depth` is counted per thread of the ring's wider endpoint (`num_entries = depth ×
  max(R, W)`): a STRIDED ring needs `num_entries % max(P, C) == 0` (dataflow_buffer.cpp:1527). At the
  Phase 0 plan this is the design's 8 entries.
- Implicit-sync entry size = the tensor's own aligned page slot (`align_up(page_bytes, buffer
  alignment)`: 64 B in DRAM, 16 B in L1), not `align_up(page_bytes, l1_alignment)`. Same bound as the
  design's (never larger than the page's slot), and source/destination stay buffer-aligned.
- `choose_gather_block()` step 3 (occupancy) is one pass over the two chunkable axes, which gives the
  design's worked boxes; a repeated pass reached 352 boxes for `[1,4,16,8400]`, halving the boxes again
  without shortening the longest thread's work.
- RM sub-page reads and writes round their size up to the L1 alignment (16 B); a rounded transfer stays
  inside the page's own aligned slot, because a chunk boundary is buffer-aligned and the last chunk
  ends at the row end.
- Raw compute API for the transpose (no helper can express it; reason at the head of
  `kernels/permute_compute.cpp`, as the design's "Helpers considered and rejected" says).

Structural counts (ttsim has no clock: no time figure here is a performance result; gains are
unconfirmed until a timed backend measures them). Traced YOLOv8 calls:

| Call | Regime | Box / pages | Work units | Clusters | Threads |
|------|--------|-------------|------------|----------|---------|
| RM `[1,16,20,20]` `(0,2,3,1)` | staged | (1,16,1,20) | 20 boxes | 20 | 6 |
| RM `[1,4,16,8400]` `(0,1,3,2)` | staged | (1,1,16,192) | 176 boxes | 32 | 6 |
| RM `[1,4,16,400]` `(0,1,3,2)` | staged | (1,1,16,32) | 52 boxes | 32 | 6 |
| RM `[1,4,8400,1]` `(0,3,1,2)` | staged | (1,1,192,1) | 176 boxes | 32 | 6 |
| TILE L1 `[1,4,16,400]` `(0,1,3,2)` | tile_transpose | 52 tiles | ≤ 2 per cluster | 32 | (1,1,1) |
| TILE L1 `[1,4,400,1]` `(0,3,1,2)` | staged | (1,4,32,1) | 13 boxes | 13 | 6 |
| TILE L1 `[1,400,144]` `(0,2,1)` | tile_transpose | 65 tiles | ≤ 3 per cluster | 32 | (1,1,1) |

DRAM crossings: 1 input / 1 output in every regime (ledger's data-movement budget, unchanged).

Device sessions: every regime after every other in one session, and `ttnn.to_layout` after each
regime followed by permute again, all exact (`test_regimes_share_a_device_session`,
`test_a_tt_metal_program_follows`).
