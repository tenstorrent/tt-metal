# Operation Requirements: point_to_point

## Definition
- **Formula**: `output[receiver_coord] = input[sender_coord]`; `output[d] = input[d]`
  for every other device `d`. Pure cross-chip byte movement — element values
  unchanged end to end (no arithmetic; the oracle is identity).
- **PyTorch Reference** (single-pair identity):
  ```python
  def point_to_point_ref(in_shards, sender_idx, receiver_idx):
      out = [s.clone() for s in in_shards]
      out[receiver_idx] = in_shards[sender_idx].clone()
      return out
  ```
- **Import Path**: `from ttnn.operations.point_to_point import point_to_point`
- **Function Signature**:
  ```python
  def point_to_point(
      input_tensor: ttnn.Tensor,
      sender_coord: ttnn.MeshCoordinate,
      receiver_coord: ttnn.MeshCoordinate,
      *,
      topology: ttnn.Topology = ttnn.Topology.Linear,
      output_tensor: ttnn.Tensor | None = None,
      intermediate_tensor: ttnn.Tensor | None = None,
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred, `[ ]` only when nothing usable was produced.
> **Refinement ID naming (the runner parses `Refinement \d+[a-z]?`)**: primary refinements are `Refinement N`; partial follow-ups append a lowercase letter (`Refinement 1b`).

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: `[bfloat16, float32, bfloat8_b, uint16, int32, uint32]`
- **SUPPORTED layout**: `[TILE, ROW_MAJOR]`
- **SUPPORTED topology**: `[Linear, Ring]`
- **SUPPORTED shape-derived axes**: `alignment ∈ {tile_aligned, non_tile_aligned}`
  (added during verification — see changelog)
- **Memory**: interleaved (DRAM or L1); sharded rejected by `validate()`
- **Cores**: single core `(0,0)` per endpoint (sender + receiver programs)
- **Coordination**: one cached op-internal `GlobalSemaphore`; fabric atomic-inc
  handshake (ready → done) with cache-reuse reset ordering
- **Golden baseline (analytical)**: 432 cells = **396 expected-supported + 36
  INVALID-skipped** (bf8b×ROW_MAJOR); **0 xfail_expected**. On-device-observed
  counts are **not available** — the golden/acceptance suites could not run (no
  `test_golden.py` scaffold; sim fabric bring-up fails on this single-device host).
  See `verification_report.md` → "On-device verification (BLOCKED)".

---

## Refinement queue — EMPTY (no axis-expansion work outstanding)

After the Phase-0 registry-conformance fix (adding the `alignment` axis + tagger),
**`SUPPORTED` covers `TARGET` on every axis**:

| Axis | TARGET | SUPPORTED | TARGET − SUPPORTED |
|------|--------|-----------|--------------------|
| dtype | bf16, f32, bf8b, uint16, int32, uint32 | (all 6) | ∅ |
| layout | TILE, ROW_MAJOR | TILE, ROW_MAJOR | ∅ |
| topology | Linear, Ring | Linear, Ring | ∅ |
| alignment | tile_aligned, non_tile_aligned | (both) | ∅ |

The only structurally-removed region — `{bf8b, ROW_MAJOR}` — is covered by the
`INVALID` entry in `feature_spec.py`, so it is correctly *skipped*, not queued.

Every `(axis, missing_value)` pair from `TARGET − SUPPORTED` is therefore the empty
set: there are no refinements to file. This is the expected end state for a
**pure-data-movement op whose kernels are dtype/layout/alignment-agnostic byte
movers** — full TARGET coverage is reachable in the Phase-0 build, with no
per-axis kernel branching to add later.

Per the agent contract, items that are **not** axis-expansions and **not** moves
out of a tracked failure category do **not** belong in this queue. The two
remaining concerns are recorded where they belong:

1. **Verification debt — DISCHARGED (2026-07-02).** The Phase-0 `SUPPORTED` claims
   were previously verified by code review + analytical gap analysis only. They are
   now **observed** green: the full acceptance suite
   (`test_point_to_point.py`, 60 items) passes **60/60** on the `bh_8xP150_p2p` sim
   (mesh `(2,4)`, FABRIC_1D + FABRIC_1D_RING) via
   `scripts/run_multidevice_sim_pytest.py`. The prior fabric-bring-up "hang" was a
   transient/stale-sim condition, not an op defect. Highest-risk paths (bf8b/f32/bf16
   framing, Ring routing, program-cache reuse / semaphore-reset footgun, output_tensor
   and non-participating paths) all confirmed. See `changelog.md` → 2026-07-02 entry.
   *(Not re-run here: uint16/int32/uint32 — SUPPORTED but not in the acceptance
   suite's DTYPE_LAYOUTS; the segmented packet regime `page_segments > 1` — not reached
   by the acceptance suite's shard shapes; and the golden scaffold — an upstream
   `/golden-tests` task. These remain analytically supported.)*

2. **Out-of-TARGET enhancements (need a TARGET expansion first, via `/golden-tests`).**
   - *Sharded memory config* — `validate()` rejects sharded input; TARGET has no
     `memory_config` axis, so this is not a refinement candidate today.
   - *Multi-link / multi-core fan-out* — single-core per endpoint by design; a
     performance change with no SUPPORTED axis or failing cell to point at.
   Both are noted in `verification_report.md` → Recommendations #3 as performance/
   scope items, deliberately kept out of this queue.

> If a future `/golden-tests` pass widens `TARGET` (e.g. adds a `memory_config`
> axis for sharded I/O), the new `(axis, value)` pairs become the first real
> refinements here, named `Refinement 1`, `Refinement 2`, … at that time.
