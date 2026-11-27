# Changelog: g6_pad

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier). Two regimes, one device program per call:
  - `extension_band`: an in-place band write over both DM RISCs, with a metadata-only view as the output.
  - `new_storage_tiles`: an NCRISC reader plus a BRISC writer that assemble each output page once in L1.
- **SUPPORTED at Phase 0** (after the verifier drift fix):
  - dtype = [bfloat16, float32, int32, uint32], layout = [TILE], memory_config = [dram, l1];
  - alignment = [tile_aligned, w_non_aligned, h_non_aligned], rank = [2, 3, 4], padded_dims = [leading, h, w, hw].
- **Accuracy achieved**: PCC = 1.0, max_abs_err = 0, rms_err = 0, max ULP = 0. This is bit-exact, measured on 4 shapes (bf16 extension ×2, bf16 new_storage with a −2^127 fill, fp32 p1) via `test_g6_pad_precision_baseline.py`.
- **Golden suite at Phase 0**:
  - 78 / 85 tagged cells pass, with 7 `xfail_expected` (all ROW_MAJOR) and 0 loud categories, with `EVAL_CAPTURE_L1=0`.
  - The default runner shows 74 pass plus 4 `supported_fail`. Those 4 are a harness L1-profiling artifact on host-side refusals (see verification_report.md).
- **Perf at Phase 0** (BH, device ns): p1 28,327 (goal < 29,910), p2 11,894, p3 5,050, p4 10,189.
- **Issues encountered / fixed by the verifier**:
  - Three stale test files from an earlier attempt broke collection of the test directory. Two were rewritten against the current API and the redundant profile test was removed.
  - The L1 cap hardcoded the buffer depth `2`; it now uses the `input_depth_blocks` knob (DRY).
  - SUPPORTED drift was fixed (`xpass_drift` on the sharded-refusal cell). Probe evidence showed 17/17 gated TILE cells passing, plus int32 TILE.
  - `l1_ledger.md` was brought up to date (the 32-bit cap is 164, the cap expression depends on the depth knob, and the page formats cover all dtypes).
- **Tests added**:
  - `test_g6_pad.py` (acceptance, by the planner);
  - `test_g6_pad_extra_cases.py` and `test_g6_pad_knobs.py` (rewritten; knob test covers the multi-block depth-2 path);
  - `test_g6_pad_dtypes.py` (fp32 / int32 / uint32 / rank 2 / L1);
  - `test_g6_pad_precision_baseline.py`;
  - `test_g6_pad_perf_ab.py` (p1 added).

## Refinement 1 — ROW_MAJOR layout (`new_storage_rows` regime)
- Date: 2026-11-27
- What was done: built the design's `new_storage_rows` regime natively, with no host-side layout conversion and one device program per call. `SUPPORTED["layout"]` now holds `[TILE, ROW_MAJOR]`.
  - Reused: `select_regime`, `make_plan`, the descriptor cache, `build_new_storage_descriptor`, the NCRISC reader / BRISC writer kernels, `OutTileWalker`, the `cb_input_tiles` nominal-push block discipline and `cb_fill_tile`.
  - Added: a `rows_mode` CT flag in the shared reader/writer. A unit is (output row, row chunk): the walker's tile row is a row and its tile column is a 64 B-aligned row chunk, and `source_row()` / `out_row()` were added to the walker.
  - Added knobs `row_chunk_bytes` (0 = auto) and `min_row_chunk_bytes` (1024). In auto mode each row is split into enough chunks that rows × chunks covers the grid, with the chunk L1-capped. This puts `[5,8192]` on 112 units and `[5,3,2560]` on 120 units, both spread over 110 cores.
  - Reader: reads only the input part of each source-bearing chunk into the start of its slot. Fill-only rows and chunks are never read.
  - Writer: fill-only units are written from the fill page before the input wait. The row tail `[d_w, D_w)` is filled in place: RISC stores up to the next 16 B boundary, then one local NoC copy from the fill page. Every output page is written once.
  - The fill page larger than 4 KB is RISC-seeded and grown by local NoC doubling. Tile pages are ≤ 4 KB, so the tiles path is unchanged.
- Accuracy achieved: bit-exact on all cells (exact gate, PCC = 1.0, RMS = 0). Shapes: the 6 golden RM cases, plus `[3,7,45]` bf16 hw, `[2,3,33,1000]` fp32 multi-chunk + tail, `[4,600]` fp32 L1, `[1,1,2,4097]` i32, `[200,64]` bf16, and a 9-way chunk/block knob sweep.
- Golden test progress: all 7 former `xfail_expected` ROW_MAJOR cells pass, including the loose `special_values` cell on 262. In the targeted slice (RM + TILE representatives), 17/17 pass. The unit dir passes 93/93.
- Perf (BH, device ns, 110 cores): `[5,8192]` fp32 RM 2,856; `[5,3,2560]` bf16 RM 1,590; `[1,1,196,196]` bf16 RM 1,485.
- Issues encountered: none in the op. The Tracy wrapper splits a multi-word `-k` expression, so use single-token filters with `--profile`.
- Tests added: `test_g6_pad_row_major.py` (11 regime-pinned cases plus a knob sweep). `test_g6_pad_perf_ab.py` gained 3 ROW_MAJOR guard cases (`-k rm`).
