# Operation Requirements: reduce_scatter

## Definition
- **Formula**: `output_i[b, c, h, w] = Σ_{j=0..N-1} shard_j[b, c, h, i·(W/N) + w]` for device `i`
  on the 1-D line (N = number of devices, scatter `dim = 3`). Every device contributes a same-shape
  shard; the shards are summed element-wise and device i keeps only the i-th of N equal slices of
  the sum along `dim` — PER-DEVICE DISTINCT outputs (unlike all_reduce's identical-everywhere sum).
- **PyTorch Reference**:
  ```python
  def reduce_scatter_ref(shards: list[torch.Tensor], dim: int = 3) -> list[torch.Tensor]:
      # shards: N tensors of identical shape (one per device).
      # Accumulate in fp32 then cast so the reference isn't limited by bf16 rounding.
      n = len(shards)
      acc = torch.stack([s.to(torch.float32) for s in shards], dim=0).sum(dim=0)
      acc = acc.to(shards[0].dtype)
      return list(torch.chunk(acc, n, dim=dim))   # device i receives slice i
  ```
- **Import Path**: `from ttnn.operations.reduce_scatter import reduce_scatter`
- **Function Signature**:
  ```python
  reduce_scatter(
      input_tensor: ttnn.Tensor,                       # sharded across a MeshDevice (1, N) line; each device holds one SAME-shape rank-4 shard
      dim: int = 3,                                    # scatter dimension (Phase-0: 3; -1 alias canonicalized)
      topology: ttnn.Topology = ttnn.Topology.Linear,  # Linear (primary)
      output_tensor: ttnn.Tensor | None = None,        # optional pre-allocated output (spec must equal the derived slice spec)
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
- **SUPPORTED shape-derived axes**: none — `INPUT_TAGGERS = {}` (every accepted input is
  tile-aligned by construction: TILE_LAYOUT + the `W % (N·32)` structural gate)
- **SUPPORTED op-specific axes**: `dim = 3` only, held as an op-level gate (`_SUPPORTED_DIMS`),
  deliberately NOT a `SUPPORTED` axis — the golden TARGET does not sweep `dim`, and the harness
  xfail-strikes any cell missing a SUPPORTED axis from its values dict. Promote into
  `SUPPORTED["dim"]` only together with a feature_spec that sweeps the axis.
- **EXCLUSIONS**: [] (empty)
- **Cores**: Phase A — 2 worker cores per device (forward `(0,0)` + backward `(0,1)`), single
  fabric link. Phase B — `split_work_to_cores(compute_grid, P_out)` over the compute grid.
- **Algorithm**: GATHER-THEN-REDUCE-LOCAL-SLICE — Phase A line store-and-forward fabric gather of
  every full shard into an op-internal `gather_buffer`; Phase B local N-way tile sum
  (`compute_kernel_lib::sum_blocks`) with the scatter folded into the reader's
  `SliceRowWalker` source addressing.
- **Compute config**: HiFi4 + `fp32_dest_acc_en=True` (DST-chunked internally by `sum_blocks`).
- **Cross-device sync**: ONE op-internal `GlobalSemaphore`, created once per mesh_device, parked
  on the Phase-A descriptor, per-(device, core) counting, `noc_semaphore_set(sem, 0)` re-arm for
  program-cache reuse.
- **Memory**: interleaved DRAM and L1 both verified (extended test).
- **Golden baseline**: **6 / 6 registry cells passing** (3 INPUTS × {bf16, f32} × TILE × Linear),
  per `generated/reduce_scatter_verify/verifier_report.json` — `supported_pass = 6`, all loud
  categories `0`. Plus 4 translated passes + 1 deliberate Ring lenient-xfail.
- **Accuracy** (worst device over 4, real (1,4) Blackhole hardware): bf16 PCC ≥ 0.999996
  (rel-RMS ≈ 0.0027, 1–3 output-ULP at tensor scale); float32 PCC = 1.0000000 (rel-RMS ≈ 4.4e-4 —
  the FPU add path's TF32-class operand quantization; expected). See `verification_report.md`.

---

## Refinement queue — EMPTY

**There are no open refinements.** `SUPPORTED` already equals `feature_spec.py`'s `TARGET` on
every axis:

| Axis | TARGET | SUPPORTED | Gap |
|---|---|---|---|
| dtype | [bfloat16, float32] | [bfloat16, float32] | ∅ |
| layout | [TILE] | [TILE] | ∅ |
| topology | [Linear] | [Linear] | ∅ |

With `INVALID = []` and `EXCLUSIONS = []`, every generated golden cell is in-SUPPORTED and
passing (`xfail_expected = 0`, iterated per-entry, not just the summary count — see
`verification_report.md` §"Refinement-queue audit"). No failing cell exists in any category
(`OOM` / `numerical-precision` / `numerical-bug` / `hang` all empty), so the second admission
criterion is also vacuous.

Beyond-TARGET directions (Ring topology — which has a live lenient-xfail translated cell waiting —
`dim ∈ {0, 1, 2}`, traffic-optimal Phase A, multi-link fabric, sharded memory) are catalogued in
`verification_report.md` §Recommendations; each requires `/golden-tests` to expand
`feature_spec.py`'s TARGET before it can legally enter this queue.
