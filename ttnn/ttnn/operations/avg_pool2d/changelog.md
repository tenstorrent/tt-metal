# Changelog: avg_pool2d

## Phase 0 — Core Implementation
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Regime `col_split_stream`: channel tile columns spread over the full grid, one `reduce<SUM, REDUCE_COL>` per block with `1/(H*W)` in the scaler, R1 padding rows zeroed by the reader, output interleaved in the input's buffer type. Metal 2.0 `ProgramSpec`, one `ttnn.generic_op` per call, thread plan (1, 1, 1).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True, False], layout=[TILE], memory_config=[DRAM, L1], sharding=[interleaved, width], output_dtype=[bfloat16], output_layout=[TILE], window=[global], batch=[single], channel_alignment=[tile_aligned] (= the specification's PHASE0)
- **Accuracy achieved**: with a 32-bit DEST at HiFi4, PCC ≥ 0.999998, max_abs_err ≤ 3.6e-03, relative RMS ≤ 0.0036, ULP p99 ≤ 1. With a 16-bit DEST at HiFi4, PCC ≥ 0.999976 and relative RMS ≤ 0.0070. At LoFi, approximation on, 16-bit DEST on uniform inputs, PCC ≥ 0.9956. Measured on 4 shapes × 3 configs by `test_avg_pool2d_precision_baseline.py`. Known bias: `1/(H*W)` is the nearest bfloat16 in the scaler (median got/true ratio 1.0035 at H*W = 15, 0.9988 at 49).
- **Golden suite at Phase 0**: 53 / 53 supported cells pass, plus the loose case, 4 numeric regression cases and 3 rejections. 496 xfail_expected, 272 invalid_skipped; supported_fail / xpass_drift / xfail_wrong_mode = 0 (`verifier_report.json`).
- **Issues encountered**: the reader zeroes the padding rows (explicit-sync input ring). Verifier fixes:
  - The scaler was truncated to bfloat16 by `prepare_reduce_scaler`, giving a downward scale error of up to 2⁻⁸ (×0.9961 at H*W = 15). The host now passes `1/(H*W)` pre-rounded to the nearest bfloat16 (`_scaler_bits`), with no kernel change.
  - Documentation: the depth-knob comment, and an *As built* note in `op_design.md` for the explicit-sync reader.
  - No SUPPORTED change.
- **Tests added**: test_avg_pool2d.py (generated acceptance), test_avg_pool2d_design.py (thread plan, chunks/blocks, R1 poisoned padding, one device session, model call with a device config, refusals), test_avg_pool2d_precision_baseline.py
