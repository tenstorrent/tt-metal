# Changelog: reduce_scatter

## Phase 0 — Core Implementation
- **Date**: 2026-08-25
- **What was done**: Initial implementation via incremental pipeline (planner → implementer →
  verifier). Self-contained Python compute-CCL op on `ttnn.generic_op` + `MeshProgramDescriptor`,
  ONE dispatch per invocation: line store-and-forward gather of whole shards fused, in the same
  program, with an arrival-ordered incremental N-way SUM on a dedicated reduce core (compute
  overlaps fabric arrival via per-block double-inc counting semaphores). Five newly-authored
  kernels; derivative of the adopted `reduce_scatter_average` minus its 1/N epilogue (no scaler CB;
  final move is a degenerate-copy `sum_blocks`). No wrapping of any existing CCL op.
- **SUPPORTED at Phase 0**: dtype=[bfloat16, float32], layout=[TILE], topology=[Linear], dim=[3]
  (negative aliases canonicalized; INPUT_TAGGERS={}, EXCLUSIONS=[]). Structural bounds: rank 4,
  interleaved DRAM/L1, tile-aligned H/W, `shape[dim] % (N·32) == 0`, slice S ≤ 256 tiles, `(1, N)`
  line mesh N ≥ 2.
- **Accuracy achieved** (worst device, N=4 Blackhole line, fp32-accumulated oracle; 4 shapes via
  `test_reduce_scatter_precision_baseline.py`): bf16 PCC=0.9999954, max_abs_err=0.0625 (= 3 bf16 ULP
  = N−1 accumulator pack roundings), rel_rms=0.0035; fp32 PCC=0.9999999, max_abs_err=0.0085,
  rel_rms=0.00064.
- **Golden suite at Phase 0**: 6 / 24 registry cells passing, 18 typed xfails
  (`topology=Ring` ×12, `dim=2` ×12, overlapping on 6), 0 loud categories (per
  `generated/reduce_scatter_verify/verifier_report.json`). Translated suite: 4 passed + 1 Ring
  refinement xfail.
- **Issues encountered**: None — code review found no correctness defects and no drift; no
  auto-fixes to SUPPORTED needed. Advisories only (fused write+inc packet saving,
  `id(mesh_device)` semaphore-cache key, relay seed page pipelining) — recorded in
  `verification_report.md`.
- **Tests added**: test_reduce_scatter.py (acceptance, 15 — planner-authored),
  test_reduce_scatter_precision_baseline.py (8), test_reduce_scatter_extended.py (5 —
  L1-interleaved input, S=256 budget boundary both sides, fp32 output_tensor path, loud-rejection
  edges). Pre-existing: test_ring_fabric_probe.py (4 — Ring wrap-link fabric precondition,
  re-confirmed green for Refinement 1). All on real silicon (`bh_quietbox_1x4_hw`, mesh (1,4),
  FABRIC_1D) via `scripts/run_multidevice_sim_pytest.py --op reduce_scatter`.
