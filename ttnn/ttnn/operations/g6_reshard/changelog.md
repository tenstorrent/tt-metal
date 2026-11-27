# Changelog: g6_reshard

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). PULL regime: each out-block holder pulls the pieces of in0 overlapping its block straight into its out shard. Two issuers per core, piece walk on device, maximal runs, one read barrier per issuer, no CB, no compute kernel, one `ttnn.generic_op` dispatch. The NoC per issuer comes from a route-model (`balanced`) policy.
- **SUPPORTED at Phase 0**: dtype=[bfloat8_b], layout=[TILE_LAYOUT], no shape-derived axes, every legal BLOCK_SHARDED L1 placement pair. `TARGET − SUPPORTED = ∅`.
- **Accuracy achieved**: bit-exact. PCC=1.000000, max_abs_err=0, mean_abs_err=0, relative RMS=0, max ULP=0, 0 bit mismatches, got/true ratio 1.0 everywhere. Measured on 4 placement pairs (P, I3, I5, O1) of the production shape via `test_g6_reshard_precision_baseline.py`.
- **Golden suite at Phase 0**: 9 / 9 registry cells passing (12 / 12 tests incl. 3 data-distribution regression tests). Verifier CLI: 0 supported_fail, 0 xpass_drift, 0 xfail_wrong_mode (`eval_results/g6_reshard_verifier/verifier_report.json`).
- **Perf at Phase 0 (P, Blackhole)**: 1637 ns Tracy `DEVICE KERNEL DURATION` (warm median); 1980–2041 ns golden `device_kernel_ns`; stock 1990 ns.
- **Issues encountered**:
  - Verifier fixes. `_build` now takes the issuer count from the plan, so the knob is read once per build.
  - Adopted the working-tree reader change that moves arg loads inside the `RESHARD-ISSUE` zone (BB-21).
  - Removed a broken untracked scratch test (`probes/test_floor_scratch.py`) whose `/tmp` kernel source no longer exists.
  - No drift fixes were needed.
- **Tests added**: `test_g6_reshard.py`, `test_g6_reshard_precision_baseline.py`, `test_g6_reshard_device_perf.py`, `test_g6_reshard_noc_policy_perf.py`
