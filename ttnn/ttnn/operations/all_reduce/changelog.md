# Changelog: all_reduce

## Phase 0 — Core Implementation
- **Date**: 2026-08-25
- **What was done**: Initial implementation via incremental pipeline
  (planner → implementer → verifier). Self-contained Python compute-CCL op:
  ONE `ttnn.generic_op` + `ttnn.MeshProgramDescriptor` dispatch per invocation —
  line store-and-forward gather of whole shards fused with an arrival-ordered
  incremental N-way SUM on a dedicated reduce core (5 newly-authored kernels,
  7 kernel descriptors per device program). Verified end-to-end on REAL 4-chip
  Blackhole hardware (`bh_quietbox_1x4_hw`: mesh (1,4), FABRIC_1D) via
  `python_env/bin/python3 scripts/run_multidevice_sim_pytest.py --op all_reduce`.
- **SUPPORTED at Phase 0**: dtype=[bfloat16, float32], layout=[TILE],
  topology=[Linear]. INPUT_TAGGERS={}, EXCLUSIONS=[]. This covers the ENTIRE
  feature-spec TARGET — the refinement queue is empty by gap accounting
  (see op_requirements.md).
- **Accuracy achieved** (worst-device, N=4, from
  test_all_reduce_precision_baseline.py, 8 cells on hardware):
  bf16 PCC ≥ 0.9999955, max_abs ≤ 1 ULP at output scale, rel_rms ≈ 0.0035
  (bf16 mantissa budget for an N=4 sum); f32 PCC ≥ 0.9999994, rel_rms ≈ 6.3e-4
  (FPU srcA/srcB ~10-bit truncation — hardware datapath property, matches the
  reduce_scatter reference).
- **Golden suite at Phase 0**: 6 / 6 cartesian cells passing (per
  `generated/all_reduce_verify/verifier_report.json`: supported_pass=6,
  supported_fail=0, xpass_drift=0, xfail_wrong_mode=0, xfail_expected=0,
  invalid_skipped=0) + 5 translated passes + 1 lenient-xfail beyond-TARGET Ring
  cell. Full hardware tally: acceptance 10, deterministic debug 4, extended 5,
  precision 8, golden 11 (+1 xfail) — all green, aggregate exit 0.
- **Issues encountered**: No op or kernel defects — zero code changes from
  verification. One verification-environment footgun found and neutralized: the
  login shell's bare `python3` resolves `ttnn` to a stale sibling clone
  (`/localdev/wransom/tt-metal-eval`) shipping an older two-dispatch all_reduce;
  early runs silently exercised it (caught by the missing accumulator-budget
  ValueError, confirmed by `probes/probe_budget_gate.py`). All graded runs were
  redone with this repo's `python_env/bin/python3`; the interpreter pin is now
  documented in op_requirements.md / verification_report.md. The implementer's
  documented rank-2 widening (vs. the design's rank-4 pin) was reviewed and
  kept — required by the immutable translated suite, zero kernel cost,
  hardware-verified at ranks 2/3/4.
- **Tests added**: test_all_reduce.py + test_all_reduce_debug.py +
  conftest.py (implementer); test_all_reduce_precision_baseline.py +
  test_all_reduce_extended.py + probes/probe_budget_gate.py (verifier).
