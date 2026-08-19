# Changelog: reduce_scatter

## Phase 0 — Core Implementation
- **Date**: 2026-08-19
- **What was done**: Initial implementation via incremental pipeline (planner → implementer →
  verifier). Self-contained Python CCL op with a compute stage on `ttnn.generic_op` +
  `ttnn.MeshProgramDescriptor`: Phase A line store-and-forward fabric gather
  (`FabricStreamSender` egress, op-owned counting waits + semaphore re-arm) into an op-internal
  `gather_buffer`, Phase B local N-way slice-tile sum (`SliceRowWalker` addressing +
  `compute_kernel_lib::sum_blocks`) to the per-device-distinct `[dim]/N` output. Two ordered
  dispatches on one command queue (queue order is the phase barrier).
- **SUPPORTED at Phase 0**: dtype=[bfloat16, float32], layout=[TILE], topology=[Linear],
  dim=[3, 2] (dim=2 promoted by the verifier — see below).
- **Accuracy achieved** (worst-device, N=4, dim=3, 4 shapes × 2 dtypes via
  `test_reduce_scatter_precision_baseline.py` on real (1,4) Blackhole hardware):
  bf16 PCC ≥ 0.9999963, max_abs_err = 0.03125 (one bf16 ULP at the sum's magnitude),
  rel_rms_err ≤ 0.00274; float32 PCC ≈ 1.0000000, max_abs_err ≤ 0.0052, rel_rms_err ≤ 0.00045.
- **Golden suite at Phase 0**: **12 / 12 in-SUPPORTED cells passing**, 12 xfail_expected (all
  `topology=Ring` — the single remaining TARGET gap, filed as Refinement 1), all loud categories 0
  (per `generated/reduce_scatter_verify/verifier_report.json`).
- **Issues encountered** (verifier fixes, all hardware-re-verified):
  - Drift fix-in-place: `SUPPORTED["dim"]` promoted `[3]` → `[3, 2]` — the dim-2 machinery (host
    slice rows, kernel static_assert, walker math) was already implemented; only the membership
    list gated it. Verified by 6 newly-passing golden cells + 6 extended tests.
  - `validate()` reordered: axis gate (typed `UnsupportedAxisValue`/`ExcludedCell`) now precedes
    the axis-value-dependent structural checks (tile alignment, slice divisibility, output spec),
    so out-of-SUPPORTED axis values always get the typed refusal regardless of shape.
  - Signature typing: `output_tensor: ttnn.Tensor | None = None`.
- **Tests added**: `test_reduce_scatter.py` (immutable acceptance, planner-authored),
  `test_reduce_scatter_extended.py` (verifier: dim=2 × {bf16, f32} × multibatch, dim=-2 alias,
  L1-interleaved in/out, dim=2 program-cache hit),
  `test_reduce_scatter_precision_baseline.py` (verifier: PCC + max/mean abs + relative RMS).
- **Verified on**: real 4-chip Blackhole hardware (`bh_quietbox_1x4_hw`, mesh (1,4), FABRIC_1D) via
  `scripts/run_multidevice_sim_pytest.py --op reduce_scatter`; aggregate exit 0 on acceptance,
  extended, precision, and golden (+ translated) suites.
