# Operation Requirements: reduce_scatter

## Definition
- **Formula**: `output_i[...] = (Σ_{j=0..N-1} shard_j)[slice i along dim]` — element-wise SUM of all
  N devices' same-shape shards on a `(1, N)` MeshDevice line; device i keeps only the i-th of N
  equal slices along `dim`. `output.shape[dim] = input.shape[dim] / N`; per-device DISTINCT.
- **PyTorch Reference**:
  ```python
  def reduce_scatter_reference(shards: list[torch.Tensor], dim: int = 3) -> list[torch.Tensor]:
      """shards[i] is device i's input; returns [device i's expected output].
      Accumulate in fp32 so the reference is not limited by bf16 rounding."""
      summed = torch.stack(shards).to(torch.float32).sum(dim=0).to(shards[0].dtype)
      return list(torch.chunk(summed, len(shards), dim=dim))
  ```
- **Import Path**: `from ttnn.operations.reduce_scatter import reduce_scatter`
- **Function Signature**:
  ```python
  reduce_scatter(
      input_tensor: ttnn.Tensor,                       # one SAME-shape shard per device on a (1, N) line mesh; TILE, interleaved DRAM/L1
      dim: int = 3,                                    # scatter dim (positive convention; negative aliases canonicalized, -1 ≡ 3)
      topology: ttnn.Topology = ttnn.Topology.Linear,
      output_tensor: ttnn.Tensor | None = None,        # written into and the SAME handle returned when supplied
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases (acceptance 15,
> golden supported cells, precision baseline 8, extended 5) on `bh_quietbox_1x4_hw` via
> `scripts/run_multidevice_sim_pytest.py --op reduce_scatter` — NEVER `run_safe_pytest.sh` (wrong
> runner for CCL ops).
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update
> SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass,
> `[~]` when real work landed but at least one named axis value is deferred (treated as completed by
> the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are
> `Refinement N`. Follow-ups of a `[~]` partial append a lowercase letter (`Refinement 1b`), ordered
> immediately after their parent. The parser matches exactly `Refinement \d+[a-z]?`.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16, float32]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED op-specific axes**: topology ∈ {Linear}, dim ∈ {3} (negative aliases canonicalized)
- **Shape-derived axes**: none (INPUT_TAGGERS = {}; golden INPUTS tile-aligned by construction)
- **Structural bounds** (loud ValueError, not axes): rank 4, interleaved (DRAM or L1), H/W
  tile-aligned, `shape[dim] % (N·32) == 0`, per-device slice S ≤ 256 tiles (resident-accumulator
  L1 budget; boundary verified at S=256 pass / S=264 reject), mesh `(1, N)` line with N ≥ 2.
- **Cores**: 3 fixed roles per device — (0,0) fwd relay, (0,1) bwd relay, (0,2) reduce.
- **Compute config**: HiFi4 + `fp32_dest_acc_en` (hard-coded; DEST_AUTO_LIMIT = 4 = max granule).
- **Golden baseline**: 6 / 24 registry cells passing, 18 typed xfails, 0 loud categories (per
  `generated/reduce_scatter_verify/verifier_report.json`). Precision: bf16 worst-PCC 0.999995
  (max err 3 ULP = N−1 accumulator pack roundings), fp32 worst-PCC 0.9999999 (rel-RMS ≈ 6.3e-4).

### [ ] Refinement 1 — Ring topology

**Goal**: add `ttnn.Topology.Ring` to `SUPPORTED["topology"]`. Ring closes the wrap link
(device N−1 ↔ device 0) so every block travels the SHORT way round: per direction, send/arrival
depths become uniform across devices — fwd sends `N/2` blocks (own + `N/2 − 1` relays), bwd sends
`(N−1)//2`, arrivals mirror (`fwd_arrivals = N/2`, `bwd_arrivals = (N−1)//2`, total N−1 — the
kernels' `fwd_arrivals + bwd_arrivals + 1 == ring_size` static_assert still holds). The kernels'
block indices are ALREADY ring-modular (op_design.md T3: fwd send k is `(i + N − k) % N`, bwd
`(i + k) % N`; reduce-reader sources `(i ∓ (1+a)) % N`) — the work is host-side: swap the
`_block_flow` Linear table for the Ring depths, wire the wrap-link neighbours (`ccl_dm_route(...,
Ring)` now returns the 1-hop wrap route — fixed in commit `32186aa74e`), and keep behaviour
selected by the `topology` kwarg alone under the SAME `FABRIC_1D` fabric config.

**Verifier notes**: no skill in the inventory covers cross-device CCL schedule work (explicitly out
of scope for `/interleaved-parallel`) — work from op_design.md's Refinement-1 sketch. The fabric
precondition is CONFIRMED live on this box: `tests/ttnn/unit_tests/operations/reduce_scatter/
test_ring_fabric_probe.py` (4/4 on `bh_quietbox_1x4_hw` under FABRIC_1D — wrap route math, wrap
connection formation, 1-hop wrap transfers both directions). Watch three seams: (1) on a ring NO
device is a line end — both directions are active everywhere, so the `num_sends == 0` idle path
goes unused (keep it; Linear still needs it) and every relay core now waits+re-arms; (2) the host
`_wire_direction` assert `route.num_hops == 1` must keep holding on the wrap pair — if it trips,
the route helper regressed, do not hand-derive `is_forward`; (3) an N/2-distance tie (even N) must
be carried by exactly ONE direction or the reduce core double-counts — pin the convention in the
depth table (fwd carries the tie: depths above) and host-assert `fwd_sends + bwd_sends == N − 1`
and `fwd_arrivals + bwd_arrivals == N − 1` per device (for N=4: fwd 2/2, bwd 1/1; the kernel-side
arrival static_assert then holds by construction). The 6 Ring×dim=2 golden cells stay xfail
(refused via the `dim` axis) until Refinement 2 also lands.

**Done when**: the 6 `topology=Ring, dim=3` golden cells pass; the translated
`test_ring_reduce_scatter_refinement_axis` flips to pass with no edit; acceptance + program-cache
tests still pass (cache-hit re-arm now exercises the wrap link too); `eval.verify_supported` clean
(supported_pass = 12, xfail_expected = 12, loud categories 0).

### [ ] Refinement 2 — dim=2 scatter

**Goal**: add `2` to `SUPPORTED["dim"]`. For dim=2 the per-device slice is rows
`[i·slice_Ht, (i+1)·slice_Ht)` of EVERY (batch, channel) plane, so the reduce reader's walk becomes
per-plane dense row-blocks: walk width `= Wt` (full rows), base from
`sched::slice_tile_offset(2, my_chip_id, 0, slice_Ht, Wt)`, then `bump_base(Ht · Wt)` (= the
design's `slice_Ht · N · Wt`) per plane, `B·C` planes per contribution. Replace the reduce reader's
`static_assert(dim == 3)` with a CT-selected dim∈{2,3} walk (the `is_supported_scatter_dim` gate
already admits 2); host passes `slice_Ht` and the dim-dependent walk parameters as CT args. The
walk order still equals the output's row-major tile order plane-by-plane, so the dense writer and
the compute kernel are UNCHANGED. `validate()` already generalizes
(`shape[canonical_dim] % (N·32)`), and `-2 ≡ 2` canonicalization already works.

**Verifier notes**: no skill match (CCL schedule work) — the worked example is the adopted
sibling's reduce reader, `ttnn/ttnn/operations/reduce_scatter_average/kernels/
reduce_scatter_average_reduce_reader.cpp:77-86,115-145` (the dim=2 per-channel restart the design
cites). Independent of Refinement 1, but run AFTER it per queue order; whichever of the two lands
second collects the 6 `Ring×dim=2` cells. The golden INPUT `(2,1,256,256)` exercises the per-batch
walk restart on dim=2 — a cursor hoisted out of the plane loop reads the WRONG slice on every plane
after the first, silently (per-device-distinct oracle catches it). Keep the IDENTICAL walk per
contribution (R11) — positional alignment across passes is what makes `add_tiles` correct.

**Done when**: the 6 `topology=Linear, dim=2` golden cells pass (plus the 6 `Ring×dim=2` if
Refinement 1 landed — then supported_pass = 24, xfail_expected = 0); a dim=2 correctness test with
B > 1 is added to the extended suite; `test_reduce_scatter_rejects_unsupported_dim` (dim=1) still
raises the typed refusal; `eval.verify_supported` clean with 0 loud categories.
