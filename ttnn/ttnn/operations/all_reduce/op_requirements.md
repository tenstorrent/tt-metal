# Operation Requirements: all_reduce

## Definition
- **Formula**: `output[...] = Σ_{c=0..N-1} shard_c[...]` on every device of a 1-D
  MeshDevice line (N = number of devices). Output shape/dtype/layout == a single
  input shard; the result is IDENTICAL on every device (unlike reduce_scatter's
  per-device distinct slices). SUM only — no scaling, no scatter/gather dim.
- **PyTorch Reference**:
  ```python
  def all_reduce_ref(shards: list[torch.Tensor]) -> list[torch.Tensor]:
      # shards: N tensors of identical shape (one per device).
      # Accumulate in fp32 then cast, so the reference isn't limited by bf16 rounding.
      acc = torch.stack([s.to(torch.float32) for s in shards], dim=0).sum(dim=0)
      acc = acc.to(shards[0].dtype)
      return [acc.clone() for _ in shards]   # every device gets the identical sum
  ```
- **Import Path**: `from ttnn.operations.all_reduce import all_reduce`
- **Function Signature**:
  ```python
  all_reduce(
      input_tensor: ttnn.Tensor,                       # sharded across a MeshDevice (1, N) line; each device holds one SAME-shape shard
      topology: ttnn.Topology = ttnn.Topology.Linear,  # Linear (Phase-0; Ring is beyond-TARGET)
      output_tensor: ttnn.Tensor | None = None,        # optional pre-allocated output (same spec as one input shard); same handle returned
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N`. Partial-tick follow-ups append a lowercase letter to the parent (`Refinement 1b`, …), ordered immediately after their parent. The runner's parser matches exactly `Refinement \d+[a-z]?`.
> **CCL verification contract**: every refinement is verified via
> `python_env/bin/python3 scripts/run_multidevice_sim_pytest.py --op all_reduce -- <tests>`
> (NEVER `run_safe_pytest.sh` for this op). The active topology is `bh_quietbox_1x4_hw`:
> real Blackhole hardware, mesh `(1, 4)`, `fabric_config = FABRIC_1D`. Tests read
> `CCL_HW_MESH_SHAPE` (default `1,4` in the mesh-adaptive suites) — never hardcode a
> different mesh shape (a mismatch hangs fabric init: "Fabric Router Sync: Timeout").
> **Interpreter pin (load-bearing on this box)**: invoke the runner with THIS repo's
> `python_env/bin/python3`. The login shell's bare `python3` belongs to a sibling clone
> whose stale `ttnn` package shadows this tree and silently swaps in a different
> all_reduce (see `verification_report.md` § Recommendations 4).

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16, float32]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED topology**: [Linear]
- **SUPPORTED shape-derived axes**: none — `INPUT_TAGGERS = {}` (every golden INPUT is
  tile-aligned by construction; the op has no scatter/gather dim axis)
- **EXCLUSIONS**: [] (empty)
- **Cores**: 3 fixed roles per device — `(0,0)` forward relay, `(0,1)` backward relay,
  `(0,2)` reduce (reader/compute/writer). Single fabric link, ONE `generic_op` dispatch.
- **Algorithm**: line store-and-forward gather of whole shards fused, in the same
  program, with an arrival-ordered incremental full-shard SUM. Whole shards land in an
  op-internal `gather_buffer` (block c at pages `[c*P, (c+1)*P)`; own block never
  written); the reduce core consumes contributions the moment their counting-semaphore
  double-inc lands (own shard first, straight from the input), so the accumulate of
  contribution k overlaps the fabric flight of k+1. Compute is all-helper:
  `sum_blocks(1)` seed → `BlockAccumulate::rearm` → `run(g)` per arrival →
  `sum_blocks(1)` drain to the writer.
- **Compute config**: HiFi4 + `fp32_dest_acc_en=True` (fixes DEST_AUTO_LIMIT = 4;
  `g ∈ {4,2,1}` divides P by host construction, so no tail chunk exists).
- **Cross-device sync**: TWO op-internal GlobalSemaphores (`sem_fwd`/`sem_bwd`),
  created once per mesh_device (module cache, one miss-branch barrier), parked on
  `mesh_pd.semaphores`; every consumer re-arms its own counter after its final wait
  (program-cache-hit safe — hardware-verified twice).
- **Structural gates** (loud ValueError): non-mesh / non-line / N<2, rank < 2, sharded
  input, non-tile-aligned H/W, non-16B page, `P * page_size > 512 KiB` resident
  accumulator budget, output_tensor spec mismatch.
- **Golden baseline**: 6 / 6 cartesian cells passing on real (1,4) Blackhole hardware
  (per verifier CLI: supported_pass=6, all loud categories 0); + 5 translated passes and
  1 lenient-xfail Ring cell. Precision (worst device): bf16 PCC ≥ 0.9999955 (≤ 1 ULP at
  output scale), f32 PCC ≥ 0.9999994.

## Refinement queue — EMPTY (TARGET fully covered)

`TARGET − SUPPORTED = ∅` on every axis (dtype, layout, topology), verified
mechanically: the verifier CLI's `xfail_expected` bucket is empty and every golden
cartesian cell passes on hardware (`generated/all_reduce_verify/verifier_report.json`).
There are no `(axis, missing_value)` pairs to file, nothing in `EXCLUSIONS` to
dissolve, and no failing cells in any category. Per the registry model this queue is
therefore empty **by gap accounting, not by omission** — Phase 0 is the completed
TARGET.

Widening the op (Ring topology, bfloat8_b, ROW_MAJOR, sharded memory, large-P beyond
the 512 KiB accumulator budget, 2-D-mesh `cluster_axis`, multi-link) first requires
widening `TARGET` in `eval/golden_tests/all_reduce/feature_spec.py` via
`/golden-tests`; until then those are beyond-TARGET candidates, prioritized and
sketched in `verification_report.md` § Recommendations (Ring is the cheapest — the
kernels are already ring-modular, and a live translated cell flips to pass when it
lands).
