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
- **Memory**: interleaved (DRAM or L1); sharded rejected by `validate()`
- **Cores**: single core `(0,0)` per endpoint (sender + receiver programs)
- **Coordination**: one cached op-internal `GlobalSemaphore`; fabric atomic-inc
  handshake (ready → done) with cache-reuse reset ordering
- **Compute config**: n/a (no compute stage — pure data movement)
- **Golden baseline (OBSERVED, this pass):** verified on the graded `bh_8xP150_p2p`
  sim topology (mesh (2,4), FABRIC_1D). 30-cell sample: **28 supported_pass +
  2 invalid_skipped** (bf8b×ROW_MAJOR); **0 supported_fail / 0 xpass_drift /
  0 xfail_wrong_mode / 0 xfail_expected**. Sample spans every axis value. Full
  cartesian = 432 (396 supported + 36 INVALID). See `verifier_report.json`.

---

## Refinement queue — EMPTY (no axis-expansion work outstanding)

**`SUPPORTED` covers `TARGET` on every axis** — verified against
`eval/golden_tests/point_to_point/feature_spec.py` and confirmed by the observed
golden run (`by_category.xfail_expected == 0`, i.e. there are no xfail-decorated
cells anywhere in the cartesian, because every non-INVALID cell is supported):

| Axis | TARGET | SUPPORTED | TARGET − SUPPORTED | Disposition |
|------|--------|-----------|--------------------|-------------|
| dtype | bf16, f32, bf8b, uint16, int32, uint32 | (all 6) | ∅ | — |
| layout | TILE, ROW_MAJOR | TILE, ROW_MAJOR | ∅ | — |
| topology | Linear, Ring | Linear, Ring | ∅ | — |
| alignment | tile_aligned, non_tile_aligned | (both) | ∅ | — |

The only structurally-removed region — `{bf8b, ROW_MAJOR}` — is covered by the
`INVALID` entry in `feature_spec.py` (observed skipped in the golden run), so it is
correctly *skipped*, not queued.

**Every `(axis, missing_value)` pair from `TARGET − SUPPORTED` is the empty set** →
there are no refinements to file. This is the expected end state for a
**pure-data-movement op whose kernels are dtype/layout/alignment/topology-agnostic
byte movers** — full TARGET coverage is reachable in the Phase-0 build with no
per-axis kernel branching to add later. (The op reaches every dtype because it
copies physical pages verbatim; both layouts because it never tilizes/untilizes;
both alignments because it copies padded tiles / RM rows as-is; both topologies
because `ccl_dm_route` owns the routing.)

Per the agent contract, items that are **not** axis-expansions and **not** moves out
of a tracked failure category (`OOM` / `numerical-precision` / `numerical-bug` /
`hang`) do **not** belong in this queue. There are no such failures — every observed
cell passes. The two forward-looking scope items are recorded where they belong:

1. **Sharded memory config** — `validate()` rejects sharded input; TARGET has no
   `memory_config` axis, so this is not a refinement candidate today. Needs a
   `/golden-tests` TARGET expansion first. (`verification_report.md` → Recommendations #2.)
2. **Multi-link / multi-core fan-out** — single-core per endpoint by design; a
   throughput change with no SUPPORTED axis or failing cell, and *not*
   embarrassingly-parallel (the fabric route + handshake are shared). Performance,
   not capability. (`verification_report.md` → Recommendations #3.)

> If a future `/golden-tests` pass widens `TARGET` (e.g. adds a `memory_config`
> axis for sharded I/O), the new `(axis, value)` pairs become the first real
> refinements here, named `Refinement 1`, `Refinement 2`, … at that time. Sharded
> I/O would bundle with the memory-config refinement (a different reader/writer),
> not stand alone — see `/memory-layouts` for the in-kernel data-access pattern.
