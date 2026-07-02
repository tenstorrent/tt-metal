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
- **Cores**: single core `(0,0)` per endpoint (SEND + RECEIVE programs), single fabric link
- **Coordination**: one cached op-internal `GlobalSemaphore`; fabric atomic-inc
  handshake (ready → done) with cache-reuse reset ordering
- **Golden baseline (OBSERVED, verifier CLI on `bh_8xP150_p2p` sim)**: merged
  30-cell run spanning **every** axis value (dtype×6, layout×2, topology×2,
  alignment×2) → **28 supported_pass + 2 invalid_skipped (bf8b×ROW_MAJOR),
  0 supported_fail / 0 xpass_drift / 0 xfail_wrong_mode / 0 xfail_expected**.
  Precision baseline 8/8 (PCC = 1.0, zero error). Acceptance suite 60/60
  (corroborating). See `verification_report.md` and
  `eval/results/point_to_point/verifier_report.json`.

---

## Refinement queue — EMPTY (no axis-expansion or failure-category work outstanding)

`SUPPORTED` covers `TARGET` on every axis, verified against the observed golden run
(all three loud categories 0, `xfail_expected` empty):

| Axis | TARGET | SUPPORTED | TARGET − SUPPORTED | Status (observed) |
|------|--------|-----------|--------------------|-------------------|
| dtype | bf16, f32, bf8b, uint16, int32, uint32 | (all 6) | ∅ | all 6 supported_pass |
| layout | TILE, ROW_MAJOR | (both) | ∅ | both supported_pass |
| topology | Linear, Ring | (both) | ∅ | both supported_pass |
| alignment | tile_aligned, non_tile_aligned | (both) | ∅ | both supported_pass |

Per the task's mandate — *"list every `(axis, missing_value)` pair from
`TARGET − SUPPORTED`; each MUST appear in a refinement, be covered by INVALID, or
have a documented reason for omission"* — the enumeration is:

- **`TARGET − SUPPORTED` = ∅ on every axis** → there are zero `(axis, missing_value)`
  pairs, hence zero axis-expansion refinements. Confirmed by the empty
  `by_category.xfail_expected` bucket in `verifier_report.json` (a non-empty bucket
  would signal a queue gap; it is empty).
- The only structurally-removed region — `{bf8b, ROW_MAJOR}` — is the `INVALID`
  entry in `feature_spec.py`, observed correctly `invalid_skipped` (2 cells). Not
  queued.
- **`supported_fail` = 0** → no failing named cells to move out of a failure
  category (`OOM` / `numerical-precision` / `numerical-bug` / `hang`). No
  failure-category refinements.

This is the expected end state for a **pure-data-movement op whose kernels are
dtype/layout/alignment-agnostic byte movers** — full TARGET coverage is reachable in
the Phase-0 build with no per-axis kernel branching to add later. Filing a
refinement here would violate the queue's hard rule (it would neither add an
`(axis, value)` to SUPPORTED nor move a named failing cell out of a failure
category).

### Out-of-scope items recorded elsewhere (deliberately NOT queued)

These are in `verification_report.md` → Recommendations; they are **not**
refinements because they are neither axis-expansions of the current TARGET nor
moves of a tracked failing cell. They become the first real refinements *only if* a
future `/golden-tests` pass widens `TARGET`.

1. **Sharded memory config** — `validate()` rejects sharded input; TARGET has no
   `memory_config` axis. A future memory-config refinement (sharded reader/writer).
2. **Multi-core / multi-link fan-out** — single-core/single-link per endpoint by
   design; a performance change with no SUPPORTED axis or failing cell. Note: the
   two endpoints coordinate over the fabric with a real data dependency, so this is
   the "multi-core with real data dependency" class, **not** embarrassingly-parallel
   (`/interleaved-parallel` does not apply); no current skill covers the fabric
   ring/handshake topology.
3. **Segmented-packet path coverage** — the `page_segments > 1` de-coalescing path
   is implemented and reviewed but unreachable by any `feature_spec.INPUTS` shape
   (max RM page ≈ 2 KB ≪ fabric max packet). Not a refinement; a coverage note for a
   future `/golden-tests` widening.

> If a future `/golden-tests` pass widens `TARGET` (e.g. adds a `memory_config`
> axis for sharded I/O, or a very-wide-last-dim shard shape), the new `(axis, value)`
> pairs become the first real refinements here, named `Refinement 1`, `Refinement 2`,
> … at that time.
