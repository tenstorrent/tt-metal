# Operation Requirements: reduce_scatter

## Definition
- **Formula**: `output_i[...] = (Σ_{c=0..N-1} shard_c[...])[slice i along dim]` for device `i` on the
  1-D line (N = number of devices); `output.shape[dim] = input.shape[dim] / N`. The output is
  PER-DEVICE DISTINCT (unlike all_reduce's identical-everywhere sum).
- **PyTorch Reference**:
  ```python
  def reduce_scatter_ref(shards: list[torch.Tensor], dim: int) -> list[torch.Tensor]:
      # shards: N tensors of identical shape (one per device).
      # Accumulate in fp32 then cast so the reference isn't limited by bf16 rounding.
      acc = torch.stack([s.to(torch.float32) for s in shards], dim=0).sum(dim=0)
      acc = acc.to(shards[0].dtype)
      return list(torch.chunk(acc, len(shards), dim=dim))   # device i gets chunk i
  ```
- **Import Path**: `from ttnn.operations.reduce_scatter import reduce_scatter`
- **Function Signature**:
  ```python
  reduce_scatter(
      input_tensor: ttnn.Tensor,                       # sharded across a MeshDevice (1, N) line; each device holds one SAME-shape shard
      dim: int = 3,                                    # scatter dimension: 3 or 2 (negative aliases -1/-2 canonicalized)
      topology: ttnn.Topology = ttnn.Topology.Linear,  # Linear (Ring is Refinement 1)
      output_tensor: ttnn.Tensor | None = None,        # optional pre-allocated output (shape = shard with [dim]/N)
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N`. Partial-tick follow-ups append a lowercase letter to the parent (`Refinement 1b`, …), ordered immediately after their parent. The runner's parser matches exactly `Refinement \d+[a-z]?`.
> **CCL verification contract**: every refinement is verified via
> `scripts/run_multidevice_sim_pytest.py --op reduce_scatter -- <tests>` (NEVER `run_safe_pytest.sh`
> for this op). The active topology is `bh_quietbox_1x4_hw`: real Blackhole hardware, mesh `(1, 4)`,
> `fabric_config = FABRIC_1D`. Tests read `MULTIDEV_SIM_MESH_SHAPE` / `CCL_HW_MESH_SHAPE` — never
> hardcode a different mesh shape (a mismatch hangs fabric init: "Fabric Router Sync: Timeout").

### [x] Phase 0 — Core Implementation (+ verifier promotion of dim=2)

- **SUPPORTED dtype**: [bfloat16, float32]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED topology**: [Linear]
- **SUPPORTED dim**: [3, 2] — implemented as [3] by the implementer; dim=2 promoted by the verifier
  on hardware evidence (the host slice rows, the kernel `is_supported_scatter_dim` static_assert,
  and the SliceRowWalker math were already dim-general; only the membership list gated it)
- **SUPPORTED shape-derived axes**: none — `INPUT_TAGGERS = {}` (every golden INPUT is valid for
  every TARGET dim)
- **EXCLUSIONS**: [] (empty)
- **Cores**: Phase A — 2 worker cores per device (forward `(0,0)` + backward `(0,1)`), single fabric
  link. Phase B — `split_work_to_cores(compute_grid, S)` over the compute grid (S = output pages).
- **Algorithm**: gather-then-reduce-local-slice — Phase A line store-and-forward fabric gather of
  FULL shards into an op-internal `gather_buffer` (block c at pages `[c*P, (c+1)*P)`); Phase B
  local N-way tile sum over only THIS device's slice positions (`SliceRowWalker` addressing +
  `compute_kernel_lib::sum_blocks(cb, cb_out, N, 1, pop_input=true)`), dense output writes.
- **Compute config**: HiFi4 + `fp32_dest_acc_en=True` (covers the bf16 sum-of-N budget and float32).
- **Cross-device sync**: ONE op-internal `GlobalSemaphore`, created once per mesh_device, parked on
  the Phase-A descriptor, counting inc-after-block in-order on the fabric connection,
  `noc_semaphore_set(sem, 0)` receiver re-arm for program-cache reuse (§R1).
- **Golden baseline**: **12 / 12 in-SUPPORTED cells passing** (3 INPUTS × {bf16, f32} × TILE ×
  Linear × dim {3, 2}), per `generated/reduce_scatter_verify/verifier_report.json` —
  `supported_pass = 12`, `xfail_expected = 12` (all Ring), all loud categories `0`.
- **Accuracy**: bf16 worst-device PCC ≥ 0.999996 (rel-RMS ≈ 0.0027 at N=4), float32 PCC ≈ 1.0
  (rel-RMS ≈ 4.4e-4) — measured on 4 shapes × 2 dtypes; see `verification_report.md`.
- **Tests**: `test_reduce_scatter.py` (immutable acceptance, 12 passed + 1 designed skip),
  `test_reduce_scatter_extended.py` (dim=2, -2 alias, L1-interleaved, dim=2 cache hit),
  `test_reduce_scatter_precision_baseline.py` (8 cases).

---

## Refinement queue

`TARGET − SUPPORTED` gap accounting (every `(axis, missing_value)` pair):

| Axis | TARGET | SUPPORTED | Gap |
|---|---|---|---|
| dtype | [bfloat16, float32] | [bfloat16, float32] | ∅ |
| layout | [TILE] | [TILE] | ∅ |
| topology | [Linear, Ring] | [Linear] | **Ring → Refinement 1** |
| dim | [3, 2] | [3, 2] | ∅ (closed by the verifier's dim=2 promotion) |

With `INVALID = []` and `EXCLUSIONS = []`, the 12 `xfail_expected` cells are exactly the Ring
column — one refinement closes the entire remaining TARGET.

### [ ] Refinement 1 — Ring topology

**Goal**: add `ttnn.Topology.Ring` to `SUPPORTED["topology"]`, moving all **12** golden
`xfail_expected` cells (3 INPUTS × {bf16, f32} × dim {3, 2} × Ring) to passing. Two admissible
implementation shapes, both blessed by the op prompt and `op_design.md` §R12:

1. **True ring reduce-scatter (recommended)** — the classic N−1-step receive+reduce+forward, driven
   by ONE `ttnn::ccl::schedule` object constructed identically in all three kernels
   (`RingRsSchedule` / `ring_rs_step_flags` / `SyncCadence` from
   `ttnn/operations/ccl/shared_with_host/ccl_helpers_schedule.hpp`), with
   `compute_kernel_lib::BlockAccumulate::arm(...)` once + `acc.run(n)` per chunk on TRISC and the
   reader-wait/writer-inc pairing through the schedule's `SyncCadence`. Model kernels: the
   silicon-verified `reduce_scatter_minimal_async` ring triple (read-only correctness reference);
   `all_reduce_async/.../ring_reduction.cpp` for the compute model. This shape also deletes the
   full-shard gather traffic (~N× fabric bandwidth) and the N·P-page `gather_buffer` — it subsumes
   the perf recommendations in `verification_report.md`.
2. **Ring-aware gather (smaller diff)** — keep gather-then-reduce; on a ring each direction relays
   only its short-way half of the blocks (`ccl_dm_route(mesh, src, dst, Topology.Ring)` owns the
   short-way + wrap-link routing). Host block-flow tables and the kernels' relay loops change from
   the line's `i∓k` walk to ring-modular indices split so every block lands EXACTLY once per device
   (no double delivery on the two directions).

**Verifier notes**: no current skill covers cross-device ring data dependencies (explicitly outside
`/interleaved-parallel`'s scope) — work from the schedule header's banner (§S: RingRsSchedule,
LineChannelWalk, SyncCadence), the accumulate header's banner (DST-zero invariant, `arm` granularity
vs `run` count — conflating them is a CB-wait deadlock, `run_seeded` for the 3-input terminal step),
and the model kernels above. Queue-level cautions: (a) **fabric contract first** — the verification
topology `bh_quietbox_1x4_hw` pins `fabric_config = FABRIC_1D`; before writing kernels, probe
whether `ccl_dm_route(..., Topology.Ring)` resolves wrap-link routes on the quietbox under FABRIC_1D
(a 1-hop probe per direction is enough). If the box/fabric-config cannot route the wrap link,
coordinate a topology-matrix update (e.g. a FABRIC_1D_RING entry) BEFORE implementation — do not
burn the refinement discovering an infra gap. (b) The golden Ring cells run under the SAME
`device_params = {fabric_config: FABRIC_1D}` as Linear; the op must select behavior from the
`topology` kwarg alone. (c) Keep the Linear path untouched and green — the 12 Linear cells plus the
acceptance/extended/precision suites are the non-regression gate. (d) The per-step semaphore re-arm
is doubly load-bearing for a ring (one semaphore signal per step per launch) — the program-cache
tests (`test_program_cache_hit`, `test_reduce_scatter_dim2_program_cache`) are the trap that catches
a missing re-arm; run them early. (e) A `[~]` partial tick is acceptable ONLY as: Ring shipped for
`dim=3` with `EXCLUSIONS = [{"topology": Ring, "dim": 2}]` and a `Refinement 1b` filed for the
dim=2 cells — never by silently keeping Ring out of SUPPORTED.

**Done when**: all 12 currently-xfail_expected Ring cells pass on `bh_quietbox_1x4_hw`; the verifier
CLI reports `supported_pass = 24`, `xfail_expected = 0`, all loud categories 0; the translated
`test_ring_reduce_scatter_refinement_axis` cell flips from xfail to pass with no edit; acceptance +
extended + precision suites stay green.

---

**No further refinements.** Everything else observed in verification is either a performance
observation with no failing cell (slice-only gather, packet coalescing, `sum_blocks` granularity,
multi-link, self-copy serialization — recorded in `verification_report.md` §Recommendations, and
largely subsumed by Refinement 1's option 1) or a beyond-TARGET direction (dim=1, ROW_MAJOR,
bfloat8_b, sharded memory, 2-D mesh) that first requires `/golden-tests` to expand
`feature_spec.py`'s TARGET.
