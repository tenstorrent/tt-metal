# Changelog: all_gather_rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-11-07
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier). One
  fused `ttnn.generic_op` (one MeshProgramDescriptor) per call: ports (one per direction × link) relay raw input
  tiles over Fabric 2D straight into the L1 landing slots of the member cores (balanced ring / line, ready fence,
  one fused write + increment per item). Members (round-robin row groups of k slice-nested cores) compute Σx²,
  combine partials in-chip (symmetric k ≤ 8 / owner k > 8, ported from the base op `rms_norm`) and normalize as rows
  land. Regime N composes dense output tiles for W_local % 32 ≠ 0.
- **SUPPORTED at Phase 0**: dtype=[float32, bfloat16, bfloat8_b], fp32_dest_acc_en=[True, False], layout=[TILE],
  alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4], gamma_mode=[gamma, no_gamma],
  gamma_dtype=[float32, bfloat16, none], gamma_layout=[TILE, ROW_MAJOR, none],
  memory_layout=[INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED], cluster_axis=[0, 1], topology=[Linear, Ring],
  num_links=[1, 2]. EXCLUSIONS: {float32, fp32_dest_acc_en=False}, {bfloat8_b, w_non_aligned}.
- **Accuracy achieved** (4 shapes × 4 precision corners, `test_all_gather_rms_norm_precision_baseline.py`):
  bf16 + fp32 DEST PCC ≥ 0.999997, max_abs ≤ 0.086, rel_rms ≤ 0.0038; bf16 + fp16 DEST PCC ≥ 0.999992,
  rel_rms ≤ 0.0062; fp32 PCC = 1.000000, max_abs ≤ 0.025, rel_rms ≤ 0.0012; bf8b PCC ≥ 0.99995, rel_rms ≤ 0.0099.
- **Golden suite at Phase 0**: 10,503 / 10,503 supported cells passing; 3,000 xfail_expected (the two EXCLUSIONS),
  180 infeasible_skipped, 0 supported_fail / xpass_drift / xfail_wrong_mode (`verifier_report.json`, run with
  `EVAL_CAPTURE_L1=0`, see verification_report.md).
- **Perf at Phase 0** (eval harness, fastest chip, QB2 1×4 ring, 2 links): 640×384 [FOCUS] 31.36 µs (target 10.1);
  640×512 41.51 µs (target 13.5); 640×1792 99.89 µs (target 47.3). Focus breakdown: TRANSPORT-only 26.22,
  LOCAL-only 14.73, empty 0.75 µs.
- **Issues encountered** (verifier fixes):
  - Correctness: race in the in-chip partial combine. A single cumulative partials semaphore let a fast peer's
    block-(b+1) increment complete a slow peer's block-b count, so a stale partial produced a wrong per-row
    inv_rms (intermittent `inf` / precision misses on the 2048-row golden cells after a program switch). Fixed with
    one semaphore per block parity (symmetric and owner combine). Reproducer: 3/20 bad → 0/40.
  - Per-stage device zones gated behind `AGRMS_PROFILE_ZONES=1`: they overflowed the perf harness's profiler buffers
    and inflated the headline number.
  - `GAMMA_READ_ALIGN` single-sourced (writer CT arg). `dst_full_sync_en` added to the program hash key. Ledger and
    design docs updated.
- **Tests added**: `test_all_gather_rms_norm_precision_baseline.py` (acceptance `test_all_gather_rms_norm.py` and
  `test_all_gather_rms_norm_debug.py` came from the planner / implementer).
