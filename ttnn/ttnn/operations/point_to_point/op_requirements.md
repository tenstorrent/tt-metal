# Operation Requirements: point_to_point

## Definition

- **Formula** (pure data movement — no arithmetic):

  ```
  output_shard[receiver_coord][i] = input_shard[sender_coord][i]
  output_shard[c]                 = output_shard_on_entry[c]      for every c != receiver_coord
  ```

- **PyTorch Reference** (standalone):

  ```python
  def torch_point_to_point(input_shards, send_idx, recv_idx):
      """input_shards: list of per-device torch tensors, in linear mesh order."""
      expected = [s.clone() for s in input_shards]
      expected[recv_idx] = input_shards[send_idx].clone()
      return expected
  ```

- **Import Path**: `from ttnn.operations.point_to_point import point_to_point`
- **Function Signature**:

  ```python
  point_to_point(
      input_tensor: ttnn.Tensor,                       # mesh-sharded, interleaved, rank >= 2
      sender_coord: ttnn.MeshCoordinate,               # device holding the shard to send
      receiver_coord: ttnn.MeshCoordinate,             # device that receives the shard
      topology: ttnn.Topology = ttnn.Topology.Linear,  # fabric topology: Linear or Ring
      output_tensor: ttnn.Tensor | None = None,        # write into existing tensor
      intermediate_tensor: ttnn.Tensor | None = None,  # optional packet staging tensor
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16, float32, bfloat8_b, uint16, int32, uint32] — the full TARGET set
- **SUPPORTED layout**: [TILE, ROW_MAJOR] — the full TARGET set
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, non_tile_aligned} — the full TARGET set
- **SUPPORTED op-specific axes**: topology ∈ {Linear, Ring} — the full TARGET set
- **EXCLUSIONS**: `[]`
- **Cores**: single worker core `(0, 0)` on each of the two participating devices; single fabric link
  (`link_idx = 0`). Every other mesh coordinate runs no program.
- **Compute config**: none — the op is dataflow-only (no compute kernel, no `math_fidelity` /
  `fp32_dest_acc_en` surface to expose). All four CBs carry opaque bytes declared `uint32`.
- **Golden baseline**: **396 / 396 cells passing** (`supported_pass = 396`, `invalid_skipped = 36`,
  every loud category 0 — see `generated/p2p_verify/verifier_report.json`).
- **Accuracy**: bit-exact (PCC = 1.0000000, max_abs = mean_abs = rel_rms = 0.0) on every shape and
  every dtype measured.

---

## The refinement queue is EMPTY

There are **no open refinements**, and this is the honest state of the op rather than an omission.

Machine-checked against `eval/golden_tests/point_to_point/feature_spec.py`:

| axis | TARGET | SUPPORTED | TARGET − SUPPORTED |
|---|---|---|---|
| `dtype` | bfloat16, float32, bfloat8_b, uint16, int32, uint32 | *all six* | **∅** |
| `layout` | TILE, ROW_MAJOR | *both* | **∅** |
| `topology` | Linear, Ring | *both* | **∅** |
| `alignment` | tile_aligned, non_tile_aligned | *both* | **∅** |

Consequently:

- There is no `(axis, missing_value)` pair to promote — every value in every `TARGET` axis is already
  in `SUPPORTED`, so no entry of the form "add X to `SUPPORTED[axis]`" can be written.
- `verifier_report.json`'s `by_category.xfail_expected` bucket is **empty** (0 entries), which is the
  same fact from the harness's side: there is no cell outside `SUPPORTED` for the harness to xfail.
- There are **no failing cells** in any non-trivial category — `supported_fail = 0`, and no `OOM`, no
  `numerical-precision`, no `numerical-bug`, no `hang` — so no entry of the form "move these named
  failing cells from category Y to passing" can be written either.
- `EXCLUSIONS` is `[]`, so there is nothing to move *out* of EXCLUSIONS.
- The single `INVALID` entry (`bfloat8_b` + `ROW_MAJOR`, 36 skipped cells) is structurally
  impossible — a block-quantized tiled format has no row-major representation — and by the registry
  model never becomes a refinement.

Per the registry model a refinement entry is valid only if it adds a value to `SUPPORTED[axis]` or
moves named failing cells into passing. Neither is possible here, so filing anything would be filing
a comment, not a refinement.

**Everything a future pass might still want lives in `verification_report.md`, deliberately not here:**

- Coverage caveats (Ring never routes the long way on a `FABRIC_1D` topology; regime-B packet framing
  had no coverage until the extended suite added it; `memory_config` is not a `TARGET` axis).
- Beyond-`TARGET` directions (sharded I/O, multi-link / worker-mux fabric, multi-core work split,
  a real `FABRIC_1D_RING` wraparound topology). Each of those requires `/golden-tests` to expand
  `feature_spec.py`'s `TARGET` **first** — a refinement can only move `SUPPORTED` toward an existing
  `TARGET`, so none of them is fileable as a refinement today.
- Performance advisories with no failing cell (per-page NoC serialization; the `ttnn.clone` full-mesh
  copy on the default output path; per-call staging allocation).
- One escalation outside this op (the fabric worker adapter's payload-overrun `ASSERT` cannot catch
  the overrun it guards, and compiles out in Release anyway).
