# Changelog: mhc_post

## Phase 0 — Core Implementation
- **Date**: 2026-11-13
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier).
  Regime R1 plain-2D: one `generic_op` dispatch; split chooser over token-tile rows × C-column tiles on the
  max_grid rectangle (up to 110 cores); reader expands post / comb columns into column-0 fp32 pages; compute
  mixes a block of `block_cols` units per coefficient broadcast with fp32 SFPU MADs (unpack-to-dest, never the
  FPU); writer streams X' with one flush per block.
- **SUPPORTED at Phase 0**: dtype=[float32], sublayer_dtype=[float32], layout=[TILE], fp32_dest_acc_en=[True],
  mode=[plain], coef_source=[separate], norm=[off], feed=[off], alignment=[tile_aligned, h_non_aligned],
  feed_split=[block_aligned, not_block_aligned]
- **Accuracy achieved**: PCC = 1 − 1.3e-15, max_abs_err ≤ 1.0e-6, rel_rms_err = 5.3e-8, ULP p99 = 12, got/true
  ratio median 1.0 (measured on 4 shapes via test_mhc_post_precision_baseline.py)
- **Device time (plain fp32, profiler)**: T160 C5120 117.5 µs (49 % of DRAM peak); T640 C7168 516 µs (vs the
  81-op composite's 3 781 µs); T1024 C1792 214 µs.
- **Golden suite at Phase 0**: 92 / 92 supported cells passing; 625 xfail_expected, 2578 INVALID skipped,
  0 supported_fail / xpass_drift / xfail_wrong_mode (per `verifier_report.json`)
- **Issues encountered** (verifier):
  - DRY fix: compute kernel's DEST slot base / DEST-fit assert now come from the host's `MIX_FIXED_SLOTS`
    (CT arg) instead of a literal 2; reader's coefficient word count derived from the page size.
  - L1 ledger: `cb_coef_raw` row corrected to the 2 pages Phase 0 allocates.
  - Tried a single-block ragged column pass (templated remainder): overflowed the kernel-config buffer;
    reverted, filed as a Refinement 4 lever.
  - Eval harness defect found: `perf_measure.py` uploads inputs inside trace capture (TT_FATAL; one hang).
    Escalated in verification_report.md — blocks the perf refinements until fixed by the eval owner.
- **Tests added**: test_mhc_post.py (planner), test_mhc_post_perf.py (implementer),
  test_mhc_post_precision_baseline.py (verifier)
