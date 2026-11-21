# Changelog: to_layout

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Three
  regimes, each one `ttnn.ProgramSpec` and exactly one `ttnn.generic_op` dispatch per call: `tilize_stream`
  (RM → TILE), `untilize_stream` (TILE → RM), `identity_copy` (same layout, DM-only). 2-D independent split
  (tile-row × `bw`-tile column block) over the full grid, thread plan (1, 1, 1). Target Quasar on ttsim.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE, ROW_MAJOR], target_layout=[TILE, ROW_MAJOR],
  memory_config=[DRAM, L1], sharding=[interleaved], alignment=[tile_aligned, h_non_aligned], rank=[4]
- **Accuracy achieved**: bit-exact. PCC=1.000, max_abs_err=0.0, rel_rms_err=0.0, max ULP=0, got/true ratio = 1.0
  (measured on 4 shapes × 4 regimes via test_to_layout_precision_baseline.py)
- **Golden suite at Phase 0**: 120 / 120 supported cells passing; 1656 xfail_expected, 296 invalid_skipped,
  0 supported_fail / xpass_drift / xfail_wrong_mode (per `verifier_report.json`)
- **Issues encountered**: no drift and no supported-cell failure. The verifier added the R6 sharded-input refusal test the
  spec requires the op to carry. It also found a latent stick-transfer overrun for W % 32 != 0 (outside SUPPORTED,
  refused today) and made its fix a requirement of Refinement 4.
- **Tests added**: test_to_layout.py (harness), test_to_layout_plan.py (thread plan, session, worked table, split, sharded
  refusal), test_to_layout_precision_baseline.py

## Refinement 0 — Rank 3 (Llama's tilize calls)
- Date: 2026-11-21
- What was done: added `3` to `SUPPORTED["rank"]` (now `[3, 4]`). No kernel or host-program change: the probe the
  operator asked for confirmed `block_geometry` already folds every leading dim into
  `num_images = prod(shape[:-2])` and the identity copy counts `buffer_num_pages()`, so a rank-3 `[B, S, H]` tensor
  runs the rank-4 path unchanged — same block width, same full-grid 2-D split, same block-batched reader/writer
  (verified: `block_geometry` of `[1, 32, 2048]` equals that of `[1, 1, 32, 2048]`). Reused: every regime and
  kernel as-is. Added: the SUPPORTED value, one unit test.
- Accuracy achieved: bit-exact (PCC=1.0, relative RMS=0.0) on [1,32,64], [1,32,2048] (Llama's two tilize calls),
  [1,32,128], [1,17,128], [4,128,512], [2,512,1024], [1,32,4096], [1,32,8192], in all four layout pairs, DRAM and L1.
- Golden test progress: 64/64 rank-3 bfloat16 cells inside SUPPORTED passing (slice run), plus two rank-4
  representatives; the full suite was not re-run (the harness does it). Unit dir 65/65 before the new test, 26/26 for
  test_to_layout_plan.py after.
- Issues encountered: None. Rank-3 `w_non_aligned` shapes ([1,32,50], [4,128,47]) stay refused by the alignment
  axis; they and rank 2 stay with Refinement 4.
- Tests added: `test_llama_rank3_tilize` in test_to_layout_plan.py (Llama's two calls exact, and rank-3 geometry ==
  rank-4 geometry).
