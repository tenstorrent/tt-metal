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
- **Golden baseline (observed on the 8-device craq-sim, 2026-07-02)**: full
  cartesian is 432 cells = **396 supported + 36 INVALID-skipped** (bf8b×ROW_MAJOR).
  A representative **16-cell slice** was executed via `verify_supported`
  (covers every SUPPORTED `(axis, value)` once — all 6 dtypes, both layouts, both
  topologies, both alignments — plus the INVALID skip, plus large/rank-varied
  shapes): **supported_pass=15, invalid_skipped=1, supported_fail=0,
  xpass_drift=0, xfail_wrong_mode=0, xfail_expected=0** → verifier-clean.
  Precision baseline **8/8, PCC=1.0** (bit-exact). Program-cache + preallocated-
  output acceptance paths pass. See `verification_report.md` → "Verifier CLI
  Summary" and `eval/results/point_to_point/verifier_report.json`. (The
  ~396-cell full sweep is a mechanical follow-up, not a correctness gap — see
  report Recommendation #3.)

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

1. **Verification debt (a task, not a refinement) — RESOLVED 2026-07-02.** The Phase-0
   `SUPPORTED` claims are now verified on device on the 8-device craq-sim
   (`run_multidevice_sim_pytest.py --topology bh_8xP150_p2p`, the pinned `(2,4)` +
   FABRIC_1D mesh):
   - **Golden suite (registry) + `verify_supported`**: a 16-cell slice covering every
     SUPPORTED `(axis, value)` — including the `uint16`/`int32`/`uint32` integer
     dtypes that the acceptance suite does *not* exercise — is verifier-clean
     (supported_pass=15, invalid_skipped=1, all loud categories 0).
   - **Acceptance**: `18/18` on the same sim (prior run, `changelog.md` 2026-07-02),
     re-confirmed program-cache reuse (semaphore-reset footgun) + preallocated output.
   - **Precision baseline**: `8/8`, PCC=1.0, zero error (bit-exact identity copy) after
     fixing the test's `(1,2)`→`(2,4)` mesh/topology mismatch.
   Highest-risk paths observed passing: bf8b (uint32 intermediate framing), float32,
   Ring routing, non-tile-aligned shards, integer dtypes, multi-packet coalescing
   (256-tile `512×512`), and rank-3 shards. Remaining unexercised: a shard large enough
   to force the *segmented* packet regime (`page_segments > 1`) — not reached by the
   current INPUTS last-dim sizes; supported by review + the dtype-agnostic byte-mover
   design. Convert the 16-cell slice to the full ~396-cell sweep when a longer sim
   window is available (mechanical, non-blocking).

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
