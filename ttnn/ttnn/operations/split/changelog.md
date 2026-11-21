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

## Refinement 1 — Cut along W: the `tile_col_shift` and `rm_substick` regimes
- Date: 2026-11-21
- What was done: `-1` added to `SUPPORTED["dim"]`; both deferred regimes built in-kernel, in the one dispatch.
  Reused: the `tile_row_shift` reader skeleton (scratchpad window indexed by read sequence, `BLOCK_H_TILES` output
  pages per DRAM barrier across line boundaries, explicit-producer ring, every transfer trid 0, sticky-tag
  hand-back), the writer, the host split, the ring and the window. Added: `TileRowShiftWalk` generalized into
  `split::ShiftWalk` over the `[outer, cut_pages, inner_pages]` page view (tile_row_shift = (images, h_tiles,
  w_tiles); tile_col_shift = (images·h_tiles, w_tiles, 1); rm_substick = (sticks, 1, 1) with the cut in elements
  inside the page); per-regime assembly: tile_col_shift by RISC-V copies of ≤ 4 column runs per row (shift 0: one
  whole-tile NoC copy), rm_substick by one self-aimed NoC copy when the segment starts 16-B aligned, else RISC-V
  copies; a sized explicit-sync writer mode (`SPLIT_SIZED_WRITES`) for a ROW_MAJOR last chunk whose aligned page
  is shorter than the others (an implicit-sync write has no size argument). Both element sizes (bf16, fp32), any
  rank, 1–4 chunks, a smaller last chunk. `DEFERRED_REGIMES` and the "not built yet" error are gone.
- Accuracy achieved: exact (bit-equal to `torch.split`): PCC = 1.0, relative RMS = 0 on every `dim = -1` INPUTS
  entry (12 entries × TILE/RM × DRAM/L1 × bf16/fp32 = 96 cases).
- Golden test progress: targeted slice (every `dim = -1` cell + the `tile_row_shift` cells of `1x4x400` and
  `1x34x4096`): 16 passed, 96 xfail_expected, 0 failed, 0 XPASS. In-SUPPORTED golden cells: 36 → 44 (+4 each
  for `1x32x128 s64 dim -1` and `1x32x8192 s4096 dim 2`, bf16, both layouts, DRAM and L1; both reach
  `page_remap` in TILE and `rm_substick` in ROW_MAJOR). The rest of the `dim = -1` cells await Refinement 2's
  rank / chunk / dtype / last-chunk axes; the design test runs all of them now.
- Issues encountered: none blocking. Perf finding (not a follow-up): rm_substick reads whole sticks into the window
  and copies each segment into its entry with a barrier per entry (as tile_row_shift does today); when every chunk
  boundary is DRAM-aligned the window could be skipped (segment reads straight into the ring), and Refinement 3's
  batched assembly applies to all three shift regimes. The time gain of either is unconfirmed on ttsim.
- Tests added: `test_split_cut_w.py` (96 bit-exact cases below `validate()` + a regime-coverage check);
  `test_split_plan.py`: the regime-naming test now names the built regimes, and the session test runs a `dim = -1`
  call in each new regime (tile_col_shift, rm_substick aligned, rm_substick with sized writes) followed by a
  tt-metal `to_layout`.

## Refinement 2 — Claim the rest of TARGET: float32, ranks 2 and 4, 1/3/4 chunks, a smaller last chunk, dims -3 and -4
- Date: 2026-11-21
- What was done: SUPPORTED widened to the specification's TARGET on every axis: `dtype` += float32, `rank` += 2, 4,
  `chunks` += 1, 3, 4, `last_chunk` += smaller, `dim` += -3, -4 (with -1 from Refinement 1). EXCLUSIONS stays
  empty. No kernel or program change. Reused: every regime as built (the kernels take the element size, page sizes
  and the general `[outer, cut, inner]` page view as runtime args; the ring's `data_format_metadata` is the input's
  dtype, so a float32 value never crosses a bfloat16 buffer, R1; the scratchpad window is byte-sized from the
  input page). Added: nothing in the op beyond the SUPPORTED lists. /numeric-formats-metal: only its dtype part
  applies (movement op, no compute kernel, no compute config: none added); its pre-flight grep found no hard-coded
  data format or element size in the kernels. A fifth chunk is still refused by validate() (TARGET's `chunks` ends
  at 4, the writer's `MAX_CHUNKS`).
- Accuracy achieved: exact (bit-equal to torch.split): PCC = 1.0, relative RMS = 0, max ULP = 0 on
  (1,4,400) s2, (1,4,8400) s2, (1,34,4096) s17, (4,128,512) s64 × TILE/RM × DRAM/L1 × bf16/fp32 (precision
  baseline, 32 cases), and on every golden cell run.
- Golden test progress: targeted slice of 72 cells, all passing: every INPUTS entry in float32 (TILE in DRAM and
  ROW_MAJOR in L1, 68 cells, covering every regime, rank, chunk count, dim and a smaller last chunk) plus bf16
  TILE-in-L1 representatives of dim -3, dim -4, rank 2 with a smaller last chunk and the traced (1,4,400) call.
  With SUPPORTED = TARGET, every one of the 272 cells is in SUPPORTED (0 xfail_expected by construction); the
  harness's full run is the confirmation of record.
- Issues encountered: None.
- Tests added: test_split_precision_baseline.py parametrized over bf16 and fp32 (ULP distance measured in the
  dtype's own bits); test_split_plan.py: the chunk-count cases (1, 3, 4, smaller last) now go through the public
  `split()`, the session test's dim -1 calls too, and `test_fifth_chunk_is_refused` pins the 4-chunk cap.
  Unit directory: 191 passed.
