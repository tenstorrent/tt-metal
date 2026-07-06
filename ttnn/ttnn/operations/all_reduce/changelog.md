# Changelog: all_reduce

## Phase 0 — Core Implementation
- **Date**: 2026-07-06
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer →
  verifier). Self-contained Python CCL op **with a compute stage**: `ttnn.generic_op` +
  `ttnn.MeshProgramDescriptor` over a 1-D MeshDevice line, with newly-authored fabric-dataflow and
  compute (TRISC) kernels. Gather-then-reduce algorithm — Phase A line store-and-forward fabric
  gather into an op-internal `gather_buffer`, Phase B local element-wise N-way tile SUM. Fabric egress
  goes through the `ccl_helpers_dataflow.hpp` safety-by-construction typestate helper; the receive
  ingress, counting wait, and cache-reuse semaphore re-arm are op-owned. Does not wrap/import/dispatch
  any existing all_reduce / reduce_scatter / all_gather op.
- **SUPPORTED at Phase 0**: dtype=[bfloat16, float32], layout=[TILE], topology=[Linear];
  INPUT_TAGGERS={}, EXCLUSIONS=[]. Equals `feature_spec.py` TARGET exactly (full-TARGET at Phase 0).
- **Accuracy achieved**: bfloat16 PCC ≥ 0.99998 (max_abs ≤ 0.1875, mean_abs ≈ 0.017, rel_RMS ≈ 0.0087);
  float32 PCC ≥ 0.9999998 (max_abs ≤ 0.0151, mean_abs ≈ 0.0015, rel_RMS ≈ 0.0007). Measured on 4 shapes
  × 2 dtypes over the (1, 8) line mesh (N=8 summands) via test_all_reduce_precision_baseline.py.
- **Golden suite at Phase 0**: 6 / 6 cells passing (3 INPUTS × {bf16, f32} × TILE × Linear) — per
  `generated/all_reduce_verify/verifier_report.json`: supported_pass=6, supported_fail=0,
  xpass_drift=0, xfail_wrong_mode=0, xfail_expected=0, invalid_skipped=0.
- **Verification runner**: multi-device craq-sim, `scripts/run_multidevice_sim_pytest.py --op
  all_reduce` (topology `wh_t3k_allmmio_all_reduce`, mesh (1,8), FABRIC_1D). Aggregate exit 0 on
  acceptance (10 passed), golden (6 passed), and precision baseline (8 passed) runs.
- **Issues encountered**: No correctness defects. Two code-review cleanups applied by the verifier:
  (1) marked the unused `ring_size` compile-time-arg local `[[maybe_unused]]` in both Phase-A kernels
  (it is part of the uniform CT superset but genuinely unread — every index derives from `my_chip_id`
  + `num_targets_{fwd,bwd}`); (2) moved `_num_line_devices` above its first use in
  `all_reduce_program_descriptor.py`. Both are behavior-preserving; kernels recompiled fresh under the
  sim and all cells still pass. No drift — no SUPPORTED edits were needed.
- **Tests added**: test_all_reduce.py (acceptance: shape/dtype sweep + program-cache re-arm +
  output_tensor path — pre-existing), test_all_reduce_precision_baseline.py (PCC + max/mean abs error
  + relative RMS error across 4 shapes × 2 dtypes — added by the verifier).
- **Refinement queue**: EMPTY — SUPPORTED == TARGET on every axis; no `(axis, missing_value)` gap to
  close. Beyond-TARGET directions recorded in verification_report.md §Recommendations.
