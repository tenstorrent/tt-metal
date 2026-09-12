# Operation Requirements: reduce_scatter_average

## Definition
- **Formula**: `output_i[...] = ((Σ_{c=0..N-1} shard_c) / N)[slice i along dim]` for device `i` on
  the 1-D line (N = number of devices); `output.shape[dim] = input.shape[dim] / N`. The 1/N
  scaling is part of the op — the caller passes nothing but the tensor. The output is PER-DEVICE
  DISTINCT.
- **PyTorch Reference**:
  ```python
  def reduce_scatter_average_ref(shards: list[torch.Tensor], dim: int) -> list[torch.Tensor]:
      # shards: N tensors of identical shape (one per device).
      # Accumulate the MEAN in fp32 then cast, so the reference isn't limited by bf16 rounding.
      acc = torch.stack([s.to(torch.float32) for s in shards], dim=0).mean(dim=0)
      acc = acc.to(shards[0].dtype)
      return list(torch.chunk(acc, len(shards), dim=dim))   # device i gets chunk i
  ```
- **Import Path**: `from ttnn.operations.reduce_scatter_average import reduce_scatter_average`
- **Function Signature**:
  ```python
  reduce_scatter_average(
      input_tensor: ttnn.Tensor,                       # sharded across a MeshDevice (1, N) line; each device holds one SAME-shape shard
      dim: int = 3,                                    # scatter dimension (Phase-0: 3; negative alias -1 canonicalized)
      topology: ttnn.Topology = ttnn.Topology.Linear,  # Linear (Ring is Refinement 2)
      output_tensor: ttnn.Tensor | None = None,        # optional pre-allocated output (shape = shard with [dim]/N)
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N`. Partial-tick follow-ups append a lowercase letter to the parent (`Refinement 1b`, …), ordered immediately after their parent. The runner's parser matches exactly `Refinement \d+[a-z]?`.
> **CCL verification contract**: every refinement is verified via
> `scripts/run_multidevice_sim_pytest.py --runtime hardware --op reduce_scatter_average -- <tests>`
> (NEVER `run_safe_pytest.sh` for this op). The active topology is `bh_quietbox_1x4_hw`: real
> Blackhole hardware, mesh `(1, 4)`, `fabric_config = FABRIC_1D`. Tests read
> `CCL_HW_MESH_SHAPE` (default `1,4` in the mesh-adaptive suites) — never hardcode a different
> mesh shape (a mismatch hangs fabric init: "Fabric Router Sync: Timeout").

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16, float32]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED topology**: [Linear]
- **SUPPORTED dim**: [3] (positive convention; -1 alias canonicalized before the membership test)
- **SUPPORTED shape-derived axes**: none — `INPUT_TAGGERS = {}` (every golden INPUT is
  tile-aligned by construction)
- **EXCLUSIONS**: [] (empty)
- **Cores**: 3 fixed roles per device — `(0,0)` forward relay, `(0,1)` backward relay, `(0,2)`
  reduce (reader/compute/writer). Single fabric link.
- **Algorithm**: SINGLE-dispatch fused line store-and-forward gather + ARRIVAL-ORDERED incremental
  reduce + 1/N broadcast-scalar scale. Whole shards land in an op-internal `gather_buffer`
  (block c at pages `[c*P, (c+1)*P)`; own block never written); the reduce core consumes
  contributions the moment their counting-semaphore double-inc lands (own slice first, straight
  from the input), so the accumulate of contribution k overlaps the fabric flight of k+1. Compute:
  `sum_blocks` seed → `BlockAccumulate::run(g)` per arrival → raw `mul_tiles_bcast_scalar` × 1/N.
- **Compute config**: HiFi4 + `fp32_dest_acc_en=True` (fixes DEST_AUTO_LIMIT = 4; `g ≤ 4` by
  construction, `g` divides S so no tail chunk exists).
- **Cross-device sync**: TWO op-internal GlobalSemaphores (`sem_fwd`/`sem_bwd`), created once per
  mesh_device, parked on the MeshProgramDescriptor; TWO fabric incs per block (receiving relay
  core + receiving reduce core) in-order behind the block's pages on the same connection;
  `noc_semaphore_set(sem, 0)` re-arm on every consumer for program-cache reuse (R1).
- **Golden baseline**: **6 / 6 in-SUPPORTED cells passing** (3 INPUTS × {bf16, f32} × TILE ×
  Linear × dim=3), per `generated/reduce_scatter_average_verify/verifier_report.json` —
  `supported_pass = 6`, `xfail_expected = 18`, all loud categories `0`.
- **Accuracy**: bf16 worst-device PCC ≥ 0.999995 (rel-RMS ≈ 0.0036, max err 1–2 output-ULP at
  scale = pure bf16 quantization), float32 PCC ≈ 0.9999998 (rel-RMS ≈ 1.0e-3, FPU srcA/srcB
  truncation — hardware datapath property); measured on 4 shapes × 2 dtypes at N=4; see
  `verification_report.md`.
- **Tests**: `test_reduce_scatter_average.py` (immutable acceptance, (1,8)-pinned — self-skips on
  4-chip boxes), `test_reduce_scatter_average_debug.py` (mesh-adaptive hardware mirror, 13
  passed), `test_reduce_scatter_average_extended.py` (dim=-1 alias, L1-interleaved, fp32
  output_tensor, typed refusals — 5 passed), `test_reduce_scatter_average_precision_baseline.py`
  (8 passed).

---

## Refinement queue

`TARGET − SUPPORTED` gap accounting (every `(axis, missing_value)` pair, from
`by_category.xfail_expected` of the verifier report):

| Axis | TARGET | SUPPORTED | Gap |
|---|---|---|---|
| dtype | [bfloat16, float32] | [bfloat16, float32] | ∅ |
| layout | [TILE] | [TILE] | ∅ |
| topology | [Linear, Ring] | [Linear] | **Ring → Refinement 2** (12 cells) |
| dim | [3, 2] | [3] | **2 → Refinement 1** (12 cells) |

With `INVALID = []` and `EXCLUSIONS = []`, the 18 `xfail_expected` cells decompose as
6 `dim=2×Linear` (Refinement 1 alone) + 6 `dim=3×Ring` (Refinement 2 alone) + 6 `dim=2×Ring`
(needs both). The two refinements close the entire remaining TARGET.

### [x] Refinement 1 — dim=2 scatter

**Goal**: add `2` to `SUPPORTED["dim"]`, moving the **6** golden `dim=2 × Linear` cells
(3 INPUTS × {bf16, f32}) from `xfail_expected` to passing (and flipping the 6 `dim=2 × Ring`
cells' refusal axis to topology-only, ready for Refinement 2). In-kernel change: the dim=2 slice
is a contiguous tile-ROW block per (batch, channel) — per-(B,C) loop in the reduce reader with
`SliceRowWalker` degenerating to dense runs (`slice_Wt = Wt`), base from
`sched::slice_tile_offset(dim=2, my_chip_id, C, slice_Ht, Wt)` + per-(B,C) `bump_base(Ht·Wt)`;
remove the Phase-0 `static_assert(dim == 3)` (keep `is_supported_scatter_dim`, R9-guarded); host
derives the dim-aware slice quantities (`slice_Ht = Ht/N`, `S = P/N` unchanged) and threads them
through the reduce-reader CT args.

**Verifier notes**: no skill in the inventory covers CCL slice addressing — work from
`ccl_helpers_schedule.hpp` §S4/S5 (`slice_tile_offset`, `SliceRowWalker::set_base` /
`bump_base` / `reset_offsets`) and `op_design.md` refinement-candidate 2's sketch. Constraints
that must survive: (a) the walk order must stay IDENTICAL for every contribution (R11 — it is
what keeps `add_tiles` positionally aligned across passes) AND must remain the output tensor's
own row-major tile order (the reduce writer writes dense pages 0..S-1 and is dim-agnostic —
verify the dim=2 walk preserves that before touching the writer); (b) `validate()`'s scatter
divisibility check already generalizes over `canonical_dim` — no host validation change needed;
(c) the per-(B,C) `bump_base` boundary need not align with the g-granule boundary — the walker
emits ids one at a time, so restructure the id loop, not the CB protocol; the multibatch golden
INPUT `(2,1,256,256)` is the trap for a cursor hoisted out of the batch loop. The relay layer is
dim-agnostic (it moves whole shards) — do not touch it. Order first: local, no fabric-contract
risk, and Refinement 2's verification sweep then covers both dims in one pass.

**Done when**: the 6 `dim=2 × Linear` golden cells pass on `bh_quietbox_1x4_hw`
(`supported_pass = 12`, `xfail_expected = 12`, all loud categories 0); a dim=2 (and `-2` alias)
extended test plus a dim=2 program-cache-hit test pass on hardware; acceptance + debug + extended
+ precision suites stay green; `test_typed_refusals`' dim=2/-2 cases are updated to expect
success (they assert the refusal today).

### [x] Refinement 2 — Ring topology

**Goal**: add `ttnn.Topology.Ring` to `SUPPORTED["topology"]`, moving the remaining **12** golden
`xfail_expected` cells (3 INPUTS × {bf16, f32} × dim {3, 2} × Ring) to passing. Two admissible
shapes, per `op_design.md` refinement-candidates 1 and 4:

1. **Ring-aware gather (smaller diff, design's primary sketch)** — keep the fused
   gather+arrival-ordered-reduce program; on a ring each direction relays only its short-way half
   of the blocks (fwd depth `N/2`, bwd `(N-1)//2` — the reference reduce_scatter's Ring table),
   with `ccl_dm_route(mesh, src, dst, Topology.Ring)` owning the short-way + wrap-link routing.
   The kernels' block indices are ALREADY ring-modular (`(i ∓ k) % N`, T3) — the changes are the
   host send/arrival depth tables, the wrap-link neighbour wiring for the line-end devices, and
   the split guaranteeing every block lands EXACTLY once per device (no double delivery on the
   two directions). The reduce reader's per-direction source sequences stay `(i ∓ (1+a)) % N`.
2. **True ring reduce-scatter** — the N−1-step receive+reduce+forward partial-sum ring
   (`LineSliceCursor`/`LineChannelWalk`/`SyncCadence` machine from `ccl_helpers_schedule.hpp`).
   Bandwidth-right (deletes the N× gather traffic AND the N·P-page `gather_buffer`) but rewrites
   all five kernels; the design explicitly deferred it. Choose it only if the extra scope is
   affordable — extend the host gtest schedule sweeps
   (`tests/ttnn/unit_tests/gtests/ccl/test_ccl_helpers_schedule.cpp`) BEFORE any new schedule
   variant.

**Verifier notes**: no current skill covers cross-device ring data dependencies (explicitly
outside `/interleaved-parallel`'s scope) — work from the schedule/accumulate helper banners and
the silicon-verified ring kernels (`reduce_scatter_minimal_async`,
`all_reduce_async/.../ring_reduction.cpp`) as read-only references. Queue-level cautions:
(a) **fabric contract first** — `bh_quietbox_1x4_hw` pins `fabric_config = FABRIC_1D`; before
writing kernels, probe whether `ccl_dm_route(..., Topology.Ring)` resolves wrap-link routes on
the quietbox under FABRIC_1D (a 1-hop probe per direction suffices). If the box/fabric-config
cannot route the wrap link, coordinate a topology-matrix update BEFORE implementation — do not
burn the refinement discovering an infra gap. (b) The golden Ring cells run under the SAME
`device_params = {fabric_config: FABRIC_1D}` as Linear; the op must select behavior from the
`topology` kwarg alone. (c) Keep the Linear path untouched and green — the 12 Linear cells
(post-Refinement-1) plus acceptance/debug/extended/precision suites are the non-regression gate.
(d) The double-inc arrival contract (T4) and the R1 re-arm are doubly load-bearing on a ring —
run the program-cache tests early. (e) Depends on Refinement 1 for the 6 `dim=2 × Ring` cells; if
Refinement 1 somehow did not land, ship Ring for dim=3 as `[~]` with
`EXCLUSIONS = [{"topology": Ring, "dim": 2}]` and file `Refinement 2b` for the excluded cells —
never by silently keeping Ring out of SUPPORTED. (f) The translated
`test_ring_reduce_scatter_average_refinement_axis` cell flips from lenient-xfail to a hard
pass/fail the moment Ring enters SUPPORTED — no test edit needed.

**Done when**: all 12 remaining `xfail_expected` cells pass on `bh_quietbox_1x4_hw`; the verifier
CLI reports `supported_pass = 24`, `xfail_expected = 0`, all loud categories 0; the translated
Ring cell passes with no edit; `test_typed_refusals`' Ring case is updated to expect success;
acceptance + debug + extended + precision suites stay green.

---

**No further refinements.** Everything else observed in verification is either a performance /
resource observation with no failing cell (slice-only gather, `gather_buffer` DRAM footprint,
S > 256 L1 accumulator cliff, multi-core reduce, packet coalescing, multi-link — recorded in
`verification_report.md` §Recommendations, several subsumed by Refinement 2's option 2) or a
beyond-TARGET direction (dim ∈ {1, 0}, ROW_MAJOR, bfloat8_b, sharded memory, 2-D mesh, non-power-
of-2 N) that first requires `/golden-tests` to expand `feature_spec.py`'s TARGET.
