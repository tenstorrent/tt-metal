# Changelog: mhc_pre

## Phase 0 — Core Implementation
- **Date**: 2026-09-30
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier).
  - Regime R1 `group_ksplit_resident`: tokens go to groups and nC is split over group ranks; the X block
    stays resident from the projection to the y-mix (X read once).
  - Group combine: push gather to the root, an fp32 SFPU rank-ordered fold, and an `mcast_pipe` broadcast.
  - Custom fp32 SFPU coefficient and Sinkhorn block ops; FPU y-mix; the Sinkhorn is owned round-robin.
  - Implementer perf pass (BH p150, fp32): 640×7168 went from 415 to 382 µs; the composite baseline is 2396 µs.
- **SUPPORTED at Phase 0**: dtype=[float32], layout=[TILE], weight_dtype=[float32, bfloat16],
  fp32_dest_acc_en=[True], alignment=[tile_aligned, h_non_aligned]
- **Accuracy achieved** (fp32 X, 4 shapes, `test_mhc_pre_precision_baseline.py`):
  - y: PCC ≥ 0.9999998, max_abs ≤ 9.5e-3, rel_rms ≈ 8.6e-4 (ratio median 0.99932: FPU truncation of fp32 X)
  - post: PCC ≥ 0.9999998, max_abs ≤ 9.4e-4, rel_rms ≤ 5.3e-4
  - comb: PCC ≥ 0.9999999, max_abs ≤ 4.0e-4, rel_rms ≤ 4.5e-4
  - The Sinkhorn in isolation is fp32-exact (1.9e-7 vs fp64). Repeat calls are bitwise-identical.
- **Golden suite at Phase 0**: 106 supported_pass, 98 xfail_expected (all `dtype = bfloat16`), 1
  supported_fail. The failure is `test_large_sinkhorn_logits[T64_nC4096]`, a precision failure tracked in
  Refinement 2. Drift is 0 (`verifier_report.json`).
- **Issues encountered / verifier fixes**:
  - DRY: CB page counts were restated in `_l1_bytes`. Both the L1 fit and the ProgramDescriptor now read
    `_cb_table` (behaviour unchanged).
  - `weight_dtype += bfloat16` on golden evidence (no kernel change).
  - `dtype = bfloat16` runs, but 11 bf16-X × fp32-W cells and the bf16 depth chain miss the 5e-4 coefficient
    gate. Root cause, measured: the FPU keeps ~9 bits of a tf32 W, and its in-tile matmul accumulation is not
    fp32-exact. It is not claimed; it moves to Refinement 1 (W hi/lo split).
- **Tests added**: `test_mhc_pre_precision_baseline.py` (verifier), plus probes 002–007. The acceptance
  suite (`test_mhc_pre.py`), `test_mhc_pre_blocking.py` and `test_mhc_pre_perf.py` were already in place.
