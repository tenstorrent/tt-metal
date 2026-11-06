# Changelog: mhc_post

## Phase 0 — Core Implementation
- **Date**: 2026-11-06
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier).
  - Regime: plain-2d.
  - Mix: SFPU-only fp32 `eltwise_chain` in DEST.
  - Work split: 2-D `(R, Q)` over token-tile rows × C tiles across the full grid.
  - Coefficients: column offsets are read in-kernel.
- **SUPPORTED at Phase 0**: dtype=[float32], sublayer_dtype=[float32], layout=[TILE], fp32_dest_acc_en=[True],
  mode=[plain], coef_source=[separate], alignment=[tile_aligned, h_non_aligned]
- **Accuracy achieved**: PCC ≈ 1.0 (≥ 0.99999), max_abs_err ≤ 1.07e-6, rel_rms_err = 5.3e-8, ULP p99 ≤ 14,
  signed bias ≤ 1e-9. Measured on 4 shapes via `test_mhc_post_precision_baseline.py`.
- **Golden suite at Phase 0**: 88 / 88 supported cells passing; 379 xfail_expected; 348 invalid_skipped; 0
  supported_fail / xpass_drift / xfail_wrong_mode (per `verifier_report.json`).
- **Perf at Phase 0** (device-kernel ns, Blackhole, plain fp32):

  | Shape | Time | Notes |
  |---|---|---|
  | T160 C5120 | 125.9 µs | 100 cores |
  | T640 C7168 | 550 µs | 110 cores; the 81-op composite took 3.78 ms |
  | T4096 C1792 | 820 µs | |
  | T1 C7168 | 49 µs | |

- **Issues encountered**: one verifier fix. `validate()` now refuses sharded inputs with `ValueError`; before
  that, they would have been read through the interleaved accessor. The verifier also measured the
  `MIN_PIPELINE_BLOCKS` departure from the coarsest-block default and found it justified (see
  `verification_report.md`).
- **Tests added**: `test_mhc_post.py` (acceptance), `test_mhc_post_debug.py`, `test_mhc_post_perf.py`,
  `test_mhc_post_precision_baseline.py`.
