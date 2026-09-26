# Changelog: reduce_scatter_average

## Phase 0 — Core Implementation
- **Date**: 2026-08-20
- **What was done**: Initial implementation via incremental pipeline (planner → implementer →
  verifier). Self-contained Python CCL op (`ttnn.generic_op` + `ttnn.MeshProgramDescriptor`,
  ONE dispatch per invocation) with 5 newly-authored kernels: single-program fused line
  store-and-forward gather + arrival-ordered incremental N-way reduce + 1/N broadcast-scalar
  scale, per-device-distinct sliced output. Verified on REAL 4-chip Blackhole hardware
  (`bh_quietbox_1x4_hw`: mesh (1,4), FABRIC_1D) via
  `scripts/run_multidevice_sim_pytest.py --runtime hardware --op reduce_scatter_average`.
- **SUPPORTED at Phase 0**: dtype=[bfloat16, float32], layout=[TILE], topology=[Linear], dim=[3]
  (negative alias -1 canonicalized); EXCLUSIONS=[], INPUT_TAGGERS={}.
- **Accuracy achieved**: bf16 worst-device PCC ≥ 0.9999952, max_abs_err ≤ 0.0156 (1–2 output-ULP
  at scale — pure bf16 quantization), rel_rms_err ≈ 0.0036; float32 worst-device PCC ≈ 0.9999998,
  max_abs_err ≤ 0.004, rel_rms_err ≈ 1.0e-3 (FPU srcA/srcB truncation) — measured on 4 shapes ×
  2 dtypes at N=4 via `test_reduce_scatter_average_precision_baseline.py` (8/8 pass).
- **Golden suite at Phase 0**: 6 / 6 in-SUPPORTED cells passing (3 INPUTS × {bf16, f32} × TILE ×
  Linear × dim=3); 18 xfail_expected (dim=2 and Ring — the refinement queue); all loud categories
  0, per `generated/reduce_scatter_average_verify/verifier_report.json`.
- **Issues encountered**: two pre-existing golden-harness defects in
  `eval/golden_tests/reduce_scatter_average/helpers.py` (both pre-flagged by `op_design.md`),
  fixed by the verifier: (1) NameError — the driver called the undefined `reduce_scatter(...)`;
  (2) SUM oracle contradicting the op's MEAN semantics (`.sum(dim=0)` → `.mean(dim=0)`). No op
  code or kernel defects found; no drift (xpass_drift = supported_fail = xfail_wrong_mode = 0).
- **Tests added**: test_reduce_scatter_average.py + test_reduce_scatter_average_debug.py
  (implementer), test_reduce_scatter_average_extended.py +
  test_reduce_scatter_average_precision_baseline.py (verifier).

## Refinement 1 — dim=2 scatter
- **Date**: 2026-08-20
- **What was done**: Added `2` to `SUPPORTED["dim"]` (with the `-2` alias via the existing
  canonicalization). The change is confined to the reduce READER + host CT args, exactly per the
  verifier sketch: the dim=2 slice is a contiguous tile-ROW block per (batch, channel), walked as
  B*C dense runs of `slice_Ht*Wt` tiles — `SliceRowWalker` degenerates (`walk_slice_Wt = Wt`),
  base from `sched::slice_tile_offset(dim, my_chip_id, 0, slice_Ht, walk_slice_Wt)`, and a
  `bump_base(Ht*Wt)` + `reset_offsets(0,0)` hop between channel runs tracked PER TILE inside the
  granule loop (the run boundary need not align with the g-granule; the CB protocol is
  untouched). dim=3 compiles to the identical Phase-0 walk (single run of S tiles, stride 0 —
  boundary fire after the last tile is a behavioral no-op). Host derives `slice_Ht = Ht/N` and
  threads it through the reduce-reader CT args (now
  `[..., Wt, slice_Wt, slice_Ht, P, dim, ...]`, accessors at offset 15). The Phase-0
  `static_assert(dim == 3)` was replaced by `dim == 3 || dim == 2` (R9 `is_supported_scatter_dim`
  kept). Relay layer, compute, and writer untouched — the dim=2 walk order equals the output's
  own row-major tile order (verified: output tile grid = B*C blocks of slice_Ht x Wt in exactly
  the walk order), so the dense dim-agnostic writer contract holds. `validate()` needed no change
  (its divisibility check already generalizes over `canonical_dim`). `test_typed_refusals`'
  dim=2/-2 cases updated: they now assert the axis gate PASSES (downstream structural ValueError
  on an unsplittable H — proof of gate ordering) plus a full dim=2 success run.
- **Accuracy achieved**: same class as Phase 0. bf16 PCC >= 0.99 and fp32 PCC >= 0.999 asserted
  on 5 shapes x 2 dtypes at N=4 (`assert_with_pcc`); deterministic identical-shard row-index test
  measured max abs err 0.5 at N=4 bf16 (partial sums round through the bf16 accumulator CB —
  the documented Phase-0 1-2 output-ULP behavior, not a walk defect); fp32 multibatch
  deterministic test within atol=16 on values up to ~767 (FPU srcA/srcB truncation class).
- **Golden test progress**: 12 / 12 in-SUPPORTED cells passing (was 6 / 6), 12 xfail_expected
  (all Ring — the 6 former dim=2 x Ring cells now refuse on topology alone, ready for
  Refinement 2), all loud categories 0. Full golden dir: 16 passed, 13 xfailed (incl. translated
  suite). Unit dir on `bh_quietbox_1x4_hw`: 41 passed, 11 skipped (the (1,8)-pinned acceptance
  file self-skips on a 4-chip box).
- **Issues encountered**: one test-oracle miscalibration during bring-up (first draft of the
  deterministic row-index test claimed bf16 exactness; the running partial sums round through
  the bf16 accumulator CB, e.g. 3*93 = 279 -> 280, measured max diff 0.5 — tolerance re-derived
  as rounding-bound << 32 = slice-shift signature). No kernel or host defects.
- **Tests added**: test_reduce_scatter_average_dim2.py (15 hardware cases: 5 shapes x 2 dtypes
  correctness incl. the multibatch (2,1,256,256) cursor trap and the granule-straddles-channel-
  boundary shape (2,1,256,32), dim=-2 alias, dim=2 program-cache hit, two deterministic
  hand-calculable cases, dim=3 non-regression spot check); test_reduce_scatter_average_extended.py
  test_typed_refusals updated for dim=2/-2 success semantics.

## Refinement 2 — Ring topology
- **Date**: 2026-08-20
- **What was done**: Added `ttnn.Topology.Ring` to `SUPPORTED["topology"]` via the design's
  primary sketch (ring-aware gather, op_design.md refinement-candidate 1). The kernels were
  already ring-modular (T3) and needed only comment updates — the entire functional diff is
  host-side in the program descriptor: `_linear_flow` generalized to `_block_flow(i, N,
  topology)`, which for Ring returns the short-way split (fwd sends = arrivals = N//2, bwd =
  (N-1)//2 — every block lands EXACTLY once per device; source sets {i-1..i-N//2} and
  {i+1..i+(N-1)//2} are disjoint mod N) plus wrap neighbours ((i±1) % N), wired through
  `ccl_dm_route(.., Ring)` (1-hop wrap routes; the existing `assert num_hops == 1` kept).
  Fabric contract probed FIRST per the verifier caution: the committed 3-tier
  `reduce_scatter/test_ring_fabric_probe.py` (route math, control-plane connection, real 1-hop
  wrap-link data both directions) passed 4/4 on `bh_quietbox_1x4_hw` under FABRIC_1D before any
  implementation. The exactly-once cover + relay-prefix/index agreement was also verified by a
  host-only flow simulation for N ∈ {2,3,4,5,8} before hardware runs. Ring runs under the SAME
  `fabric_config = FABRIC_1D` device_params as Linear — behavior selected by the kwarg alone.
  `test_typed_refusals` updated: Ring now runs to completion (checked against the oracle); a
  dim=1 refusal keeps the typed-refusal machinery itself exercised.
- **Accuracy achieved**: same class as Phase 0 (the topology changes only block TRANSPORT, not
  the arithmetic). bf16 PCC >= 0.99 and fp32 PCC >= 0.999 asserted on 7 shapes x 2 dtypes x
  dim {3, 2} at N=4 (`assert_with_pcc`); deterministic exactly-once test (shard c = 2^c, mean
  (2^N-1)/N exact in bf16) reproduced EXACTLY (max abs err < 1e-6 across all devices — any
  duplicated/missed block would shift by the unique 2^c/N signature); Ring output matches
  Linear on identical input at PCC 0.999 (arrival-order rounding only).
- **Golden test progress**: 24 / 24 in-SUPPORTED cells passing (was 12 / 12 + 12 xfail_expected),
  `xfail_expected = 0`, all loud categories 0 (regenerated verifier report:
  `generated/reduce_scatter_average_verify_r2/verifier_report.json` — supported_pass = 24). Full
  golden dir: 29 passed, 0 xfailed (the translated Ring refinement cell flipped to a hard pass
  with NO test edit, as designed). Unit dir on `bh_quietbox_1x4_hw`: 61 passed, 11 skipped (the
  (1,8)-pinned acceptance file self-skips on a 4-chip box).
- **Issues encountered**: None. All 20 ring tests passed on hardware first try — the fabric
  probe + flow simulation caught the risk surface before any device run (the ccl_dm_route Ring
  wrap-branch bug that would have bitten was already fixed in-tree, caught by reduce_scatter's
  earlier Refinement-1 probe on this same box lineage).
- **Tests added**: test_reduce_scatter_average_ring.py (20 hardware cases: 4 dim=3 shapes x
  2 dtypes, 3 dim=2 x Ring shapes x 2 dtypes incl. the multibatch cursor trap (2,1,256,256) and
  the granule-straddles-channel-run shape (2,1,256,32), dim=-1/-2 aliases under Ring, the
  exactly-once deterministic cover, Ring program-cache hit (R1 re-arm doubly load-bearing on a
  ring), fp32 output_tensor path, Ring-vs-Linear consistency);
  test_reduce_scatter_average_extended.py test_typed_refusals updated for Ring success + dim=1
  refusal.
