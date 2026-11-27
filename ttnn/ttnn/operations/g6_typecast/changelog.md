# Changelog: g6_typecast

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Regime
  R1 `tile_stream`: interleaved TILE in → eltwise-chain typecast → interleaved TILE out, contiguous tile ranges
  over the full compute grid, one CB handshake + one reader barrier per block.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], output_dtype=[float32, bfloat8_b, int32], layout=[TILE],
  buffer=[dram, l1], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[3, 4]
- **Accuracy achieved**: float32 / int32 destinations bit-exact (PCC=1.0, max_abs_err=0, 0 ULP); bfloat8_b
  PCC=0.999975, max_abs_err=0.03125, rel_rms_err=0.0076, ≤ 1 quantization step vs host rendering (measured on
  6 production shapes via test_g6_typecast_precision_baseline.py)
- **Golden suite at Phase 0**: 42 supported_pass / 27 xfail_expected / 0 supported_fail / 0 xpass_drift /
  0 xfail_wrong_mode (44/71 incl. 2 untagged regression tests; per `verifier_report.json`)
- **Issues encountered**: verifier fix — program-cache key now includes the block knobs (`BUFFER_DEPTH`,
  `BLOCK_TILES_CAP`, `L1_OP_BUDGET_BYTES`, `L1_SLACK_BYTES`) so a turned knob never replays a stale program (BB-07).
- **Tests added**: test_g6_typecast.py, test_g6_typecast_blocking.py, test_g6_typecast_precision_baseline.py
