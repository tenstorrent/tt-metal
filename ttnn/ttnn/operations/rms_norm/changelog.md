# Changelog: rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-11-17
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Verifier added the `memory_config` kwarg to the entry point (golden harness passes it on sharded cells).
- **SUPPORTED at Phase 0**: dtype=[float32, bfloat16], fp32_dest_acc_en=[True], layout=[TILE, ROW_MAJOR], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2,3,4], gamma_mode=[gamma, no_gamma], gamma_dtype=[float32, bfloat16, none], gamma_layout=[ROW_MAJOR, none], memory_layout=[INTERLEAVED]
- **Accuracy achieved**: PCC≥0.9999984 (bf16), ≥0.9999997 (fp32); max_abs_err 0.0155–0.039 (bf16), rel RMS 7.5e-4–1.9e-3 (measured on 4 shapes × 2 dtypes via test_rms_norm_precision_baseline.py)
- **Golden suite at Phase 0**: 436 / 436 supported cells passing (`verifier_report.json`); 6481 xfail_expected; 0 supported_fail / xpass_drift / xfail_wrong_mode
- **Issues encountered**: 4786 sharded cells failed with TypeError for the missing `memory_config` kwarg — fixed.
- **Tests added**: test_rms_norm.py, test_rms_norm_regimes.py, test_rms_norm_precision_baseline.py

## Refinement 1 — Precision surface + TILE gamma (perf-flagged-profile contract, part 1)
- Date: 2026-11-17
- What was done: SUPPORTED gained fp32_dest_acc_en=False, dtype bfloat8_b, gamma_dtype bfloat8_b, gamma_layout TILE. EXCLUSIONS: fp32 input + fp32_dest=False; bf8b + RM / non-aligned; bf8b gamma + RM gamma. Intermediates (cb_sq/cb_xn/cb_stat) are bf16 when DEST is 16-bit, fp32 otherwise. TILE gamma: reader reads gamma tiles directly (row 0 broadcast row), no scatter, no gamma stick scratch CB; multi-core split and segment batching unchanged. 1/W is folded into the reduce scaler when DEST is 16-bit.
- Accuracy achieved: perf-profile loose cases (interleaved) 107/107 pass; PCC >= 0.9995.
- Golden test progress: targeted slices green (loose INTERLEAVED 107/107; sampled bf8b / fp32_dest=False shapes 56 passed, rest xfail/skip as expected).
- Issues encountered: with a 16-bit DEST, sequential reduce_tile accumulation of sum-of-squares is biased high once the accumulator dwarfs the per-tile increment (all-ones W=5120 gave 0.859 instead of 1.0; independent of G_CAP, CB dtype, fidelity). Fix: for Wt > BF16_DEST_MAX_WT (64, W > 2048) the descriptor promotes the compute config to fp32_dest_acc_en=True (intermediates fp32). Finding / next lever: a bf16-only multi-level reduction tree (reduce ~4 tiles, add partials pairwise) would restore true 16-bit DEST on wide rows; perf refinements should measure the cost of the fp32 promotion.
- Tests added: tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_bf16dest_debug_probe.py (debug probe script), probes/.
