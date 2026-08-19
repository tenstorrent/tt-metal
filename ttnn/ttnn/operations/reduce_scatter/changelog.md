# Changelog: reduce_scatter

## Phase 0 — Core Implementation
- **Date**: 2026-08-19
- **What was done**: Initial implementation via incremental pipeline (planner → implementer →
  verifier). Self-contained Python CCL op with a compute stage: `ttnn.generic_op` +
  `ttnn.MeshProgramDescriptor`, GATHER-THEN-REDUCE-LOCAL-SLICE across two ordered dispatches
  (Phase A: fabric line store-and-forward gather via `FabricStreamSender`; Phase B: local N-way
  `sum_blocks` with the scatter folded into `SliceRowWalker` source addressing). Five newly
  authored single-purpose kernels; no wrapping of any existing CCL op.
- **SUPPORTED at Phase 0**: dtype=[bfloat16, float32], layout=[TILE], topology=[Linear];
  op-level gate dim=3 (canonicalized -1 alias); EXCLUSIONS=[], INPUT_TAGGERS={}. Equals the golden
  TARGET on every axis — refinement queue is empty.
- **Accuracy achieved** (worst device over the (1,4) Blackhole line, N=4 summands, measured on
  4 shapes × 2 dtypes via `test_reduce_scatter_precision_baseline.py`):
  bf16 PCC ≥ 0.9999963, max_abs ≤ 0.0625, rel_rms ≈ 0.0027 (1–3 output-ULP at tensor scale);
  float32 PCC = 1.0000000, max_abs ≤ 0.0051, rel_rms ≈ 0.00044 (FPU-add TF32-class operand
  quantization — expected hardware budget).
- **Golden suite at Phase 0**: 6 / 6 registry cells passing (+ 4 translated passes + 1 deliberate
  Ring lenient-xfail), per `generated/reduce_scatter_verify/verifier_report.json` — all loud
  categories 0. Verified on REAL (1,4) Blackhole hardware (`bh_quietbox_1x4_hw`, FABRIC_1D) via
  `scripts/run_multidevice_sim_pytest.py --op reduce_scatter`; final combined unit-test run:
  24 passed, aggregate exit 0.
- **Issues encountered**: None requiring code changes. Verifier confirmed the implementer's
  documented deviation from `op_design.md`'s registry snippet (keeping `dim` out of `SUPPORTED`)
  is load-bearing and correct against the golden harness (a `SUPPORTED` axis absent from TARGET
  xfail-strikes every cell). Advisories only (serialized Phase-A self-copy, `id()`-keyed semaphore
  cache shared with all_reduce, gather-level Phase-A traffic) — recorded in
  `verification_report.md`.
- **Tests added**: `test_reduce_scatter.py` (implementer, acceptance, 13 tests),
  `test_reduce_scatter_extended.py` (verifier: L1-interleaved end-to-end, multi-core mid-row
  slice walk, typed refusals + output-spec mismatch),
  `test_reduce_scatter_precision_baseline.py` (verifier: PCC / abs / rel-RMS / ULP-at-scale over
  4 shapes × 2 dtypes).
