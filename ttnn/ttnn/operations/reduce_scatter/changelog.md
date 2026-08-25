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

## Refinement 1 — Ring topology
- **Date**: 2026-08-25
- **What was done**: Added `ttnn.Topology.Ring` to `SUPPORTED["topology"]`. Host-side only, as
  op_design.md predicted (the kernels' block indices were already ring-modular, T3 — zero kernel
  edits): `_block_flow` in the program descriptor grew a Ring branch with uniform short-way depths
  (fwd sends/arrivals = N//2, own + N//2−1 relays; bwd = (N−1)//2; even-N N/2-distance tie pinned
  to FORWARD only) plus per-device asserts `fwd+bwd sends == N−1` and `fwd+bwd arrivals == N−1`
  (the kernel-side `fwd_arrivals + bwd_arrivals + 1 == ring_size` static_assert then holds by
  construction); neighbours are modular (`(i±1) % N`) so devices N−1/0 wire the wrap link via the
  existing `_wire_direction` → `ccl_dm_route(.., Ring)` (1-hop wrap route, fixed in `32186aa74e`,
  precondition confirmed by test_ring_fabric_probe.py). Behaviour selected by the `topology` kwarg
  alone under the SAME FABRIC_1D config; the `num_sends == 0` idle path kept for Linear (and the
  degenerate N=2 ring bwd). `_wire_direction`'s `route.num_hops == 1` assert held on the wrap pair.
- **Accuracy achieved**: identical error budget to the Linear Phase-0 baseline (arithmetic
  unchanged — only who-relays-what moved): bf16 worst-device PCC=0.9999954, max_abs_err=0.0625
  (3 ULP), rel_rms=0.0044; fp32 PCC=0.9999999, max_abs_err=0.0076, rel_rms=0.0008 — on shapes
  [(1,1,64,256), (2,1,64,256)], N=4 Blackhole ring, fp32-accumulated oracle
  (test_ring_precision_metrics).
- **Golden test progress**: 12/24 registry cells passing (was 6/24), 12 typed xfails (all
  `dim=2` — the 6 Ring×dim=2 cells stay refused via the `dim` axis until Refinement 2).
  `eval.verify_supported` clean: supported_pass=12, xfail_expected=12, xpass_drift=0,
  supported_fail=0, xfail_wrong_mode=0 (generated/reduce_scatter_verify_r1/verifier_report.json).
  Translated `test_ring_reduce_scatter_refinement_axis` flipped to PASS with no edit. Full
  non-regression sweep (unit + golden dirs): 73 passed, 12 xfailed on `bh_quietbox_1x4_hw`.
- **Issues encountered**: None — first silicon run of the full ring schedule passed (20/20),
  including the program-cache-hit re-arm across the wrap link.
- **Tests added**: test_reduce_scatter_ring.py (24 — host-only depth-table invariants for
  N∈2..9 incl. send→arrival handshake + disjoint tie-broken source coverage, Linear-table
  non-regression, ring functional grid bf16/fp32 × {S=1, multi-tile, multi-batch, S=9 g=1},
  3-iteration cache-hit re-arm over the wrap link, output_tensor path, Linear↔Ring switch in one
  mesh session, ring precision metrics).
