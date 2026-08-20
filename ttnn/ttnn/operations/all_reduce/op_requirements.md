# Operation Requirements: all_reduce

## Definition
- **Formula**: `output[d][idx] = Σ_{c=0..N-1} input_shard[c][idx]` for **every** device `d` on the
  1-D line (N = number of devices). The identical element-wise SUM lands on every device; the output
  shard has the same shape/dtype/layout as one input shard. No reduce-dim parameter.
- **PyTorch Reference**:
  ```python
  def all_reduce_ref(shards: list[torch.Tensor]) -> torch.Tensor:
      # shards: N tensors of identical shape (one per device).
      # Accumulate in fp32 then cast so the reference isn't limited by bf16 rounding.
      acc = torch.stack([s.to(torch.float32) for s in shards], dim=0).sum(dim=0)
      return acc.to(shards[0].dtype)   # returned identically on every device
  ```
- **Import Path**: `from ttnn.operations.all_reduce import all_reduce`
- **Function Signature**:
  ```python
  all_reduce(
      input_tensor: ttnn.Tensor,                       # sharded across a MeshDevice (1, N) line; each device holds one SAME-shape shard
      *,
      topology: ttnn.Topology = ttnn.Topology.Linear,  # Linear (primary)
      output_tensor: ttnn.Tensor | None = None,        # optional pre-allocated output (spec must equal input shard spec)
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N`. Partial-tick follow-ups append a lowercase letter to the parent (`Refinement 1b`, …), ordered immediately after their parent. The runner's parser matches exactly `Refinement \d+[a-z]?`.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16, float32]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED topology**: [Linear]
- **SUPPORTED shape-derived axes**: none — `INPUT_TAGGERS = {}` (every INPUT is tile-aligned by construction; the reduction is always the full element-wise sum, no reduce-dim axis)
- **EXCLUSIONS**: [] (empty)
- **Cores**: Phase A — 2 worker cores per device (forward `(0,0)` + backward `(0,1)`), single fabric link. Phase B — `split_work_to_cores(compute_grid, P)` over the compute grid.
- **Algorithm**: gather-then-reduce — Phase A line store-and-forward fabric gather into an op-internal `gather_buffer`; Phase B local element-wise N-way tile sum (seed `DST[0]` + `binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>`).
- **Compute config**: HiFi4 + `fp32_dest_acc_en=True` (single-DST accumulation; safe for any N and for float32).
- **Cross-device sync**: ONE op-internal `GlobalSemaphore`, created once per mesh_device, parked on the Phase-A descriptor, per-`(device, core)` counting, `noc_semaphore_set(sem, 0)` re-arm for program-cache reuse.
- **Golden baseline**: **6 / 6 cells passing** (3 INPUTS × {bf16, f32} × TILE × Linear), per `generated/all_reduce_verify/verifier_report.json` — `supported_pass = 6`, all loud categories `0`.
- **Accuracy**: bf16 PCC ≥ 0.99998, float32 PCC ≥ 0.9999998 (measured on 4 shapes × 2 dtypes; see `verification_report.md`).

---

## Refinement queue — EMPTY

**There are no open refinements.** `SUPPORTED` already equals `feature_spec.py`'s `TARGET` on every
axis:

| Axis | TARGET | SUPPORTED | Gap |
|---|---|---|---|
| dtype | [bfloat16, float32] | [bfloat16, float32] | ∅ |
| layout | [TILE] | [TILE] | ∅ |
| topology | [Linear] | [Linear] | ∅ |

With `INVALID = []` and `EXCLUSIONS = []`, every generated golden cell is in-SUPPORTED and passing.
The verifier's `xfail_expected` bucket is empty, so there is no `(axis, missing_value)` pair to promote
into a refinement. This op is a deliberately-scoped CCL+compute *probe*; Phase 0 delivers its full
TARGET.

**Beyond-TARGET expansion directions** (Ring topology, ROW_MAJOR, bfloat8_b, multi-link/worker-mux,
sharded I/O) are documented in `verification_report.md` §Recommendations. They are **not** refinements:
a refinement can only move SUPPORTED toward the existing TARGET, and each of these would first require
`/golden-tests` to expand `feature_spec.py`'s TARGET. When that happens, re-run the verifier and file
the newly-opened `(axis, missing_value)` gaps as `Refinement 1`, `Refinement 2`, … per the grouping
rules (dtypes bundled, layouts bundled, algorithm-fundamental changes like Ring standalone).
