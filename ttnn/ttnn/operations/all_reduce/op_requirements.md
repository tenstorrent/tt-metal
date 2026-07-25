# Operation Requirements: all_reduce

## Definition
- **Formula**: `output[d][i] = Σ_{k=0..N-1} input[k][i]` for every device `d` on the
  N-device line and every element index `i`. Reduction op is **SUM**; the output
  shape equals a single input shard's shape and is identical on every device.
- **PyTorch Reference**:
  ```python
  def all_reduce_ref(in_shards):
      """in_shards: list of N per-device shards, all the same shape/dtype.
      Returns the per-device outputs (the same sum, replicated N times)."""
      acc = torch.stack([s.to(torch.float32) for s in in_shards], dim=0).sum(dim=0)
      out = acc.to(in_shards[0].dtype)          # accumulate in fp32, cast once
      return [out.clone() for _ in in_shards]
  ```
- **Import Path**: `from ttnn.operations.all_reduce import all_reduce`
- **Function Signature**:
  ```python
  def all_reduce(
      input_tensor: ttnn.Tensor,                       # shard on a (1, N) MeshDevice line; every device holds a SAME-shape shard
      topology: ttnn.Topology = ttnn.Topology.Linear,  # Linear (Phase 0)
      output_tensor: ttnn.Tensor | None = None,        # optional pre-allocated output, written in place and returned
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses exactly `Refinement \d+[a-z]?`)**: primary refinements are `Refinement N`; a partial's follow-up appends a lowercase letter (`Refinement 1b`, `Refinement 1c`, …) and is ordered immediately after its parent so the queue runs it next.
> **Verification runner (CCL — read this first)**: this is a MULTI-DEVICE op. `scripts/run_safe_pytest.sh` is the WRONG runner (it forces slow dispatch on sim and has no multichip/hang awareness). Verify EVERY refinement with:
> `scripts/run_multidevice_sim_pytest.py --op all_reduce -- <pytest target> -v`
> Every test **must** open a `(1, 8)` `mesh_device` with `fabric_config=FABRIC_1D` (topology `wh_t3k_allmmio_all_reduce`). A different mesh shape hangs fabric init (`Fabric Router Sync: Timeout`) — that is a test/topology mismatch, not a sim or op defect. A green run means the cross-device multicast actually executed and PCC asserted. Budget: golden dir ≈ 3 min, unit dir ≈ 4 min; pass `--timeout` accordingly.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: `[ttnn.bfloat16, ttnn.float32]` (the full TARGET dtype set)
- **SUPPORTED layout**: `[ttnn.TILE_LAYOUT]` (the full TARGET layout set)
- **SUPPORTED topology**: `[Topology.Linear]`
- **SUPPORTED shape-derived axes**: `alignment = [tile_aligned]` (tagger-only axis; last two dims both `% 32 == 0`)
- **Memory**: interleaved DRAM **or L1** (both verified); sharded rejected by `validate()`
- **Mesh**: 1-D line view `(1, N)`, `N ≥ 2`; verified on `(1, 8)` WH `FABRIC_1D`, plus a `(1, 3)` submesh for odd N
- **Cores**: single worker core `(0, 0)` per device; one `ProgramDescriptor` per mesh coordinate
- **Algorithm**: broadcast-all (duplex line multicast, last packet a fused write+atomic-inc) then a local N-way tile fold (pairwise `add_tiles` in ONE DEST register, `ceil(N/2)` FPU ops per output tile)
- **Coordination**: one op-internal `GlobalSemaphore` as a receive counter (created once per mesh, parked on the descriptor, receiver-side re-arm)
- **Compute config**: `fp32_dest_acc_en = (dtype == float32)`; `HiFi3` for fp32 (WH HiFi4+fp32-dest-acc HW bug), `HiFi4` for bf16 (`add_tiles` pins LoFi internally, so this is future-proofing only)
- **Golden baseline**: **6 / 6 registry cells passing** — `supported_pass=6`, `xfail_expected=0`, `invalid_skipped=0`, and all five loud categories **0** (`eval/results/all_reduce/verifier_report.json`). Plus 5/5 translated cells passing and the `topology=Ring` translated cell correctly xfailing.
- **Accuracy achieved**: bf16 PCC = 0.999994 (rel-RMS 3.7e-3), fp32 PCC = 0.99999996 (rel-RMS 4.4e-4), shape-independent across 4 shapes — see `verification_report.md`.

---

## Refinement queue

**`TARGET − SUPPORTED` is EMPTY on every axis.** Phase 0 already implements the
whole declared universe (`dtype ∈ {bf16, f32}`, `layout = TILE`,
`topology = Linear`), so the usual gap table is empty and the queue is short by
construction — not by omission. The two entries below are the only changes in
scope that add a value to a `SUPPORTED` axis:

| Axis | TARGET | SUPPORTED (Phase 0) | Gap → Refinement |
|------|--------|---------------------|------------------|
| dtype | bf16, f32 | bf16, f32 | ∅ |
| layout | TILE | TILE | ∅ |
| topology | Linear | Linear | ∅ — but the translated suite carries a **`Ring`** cell that `validate()` refuses → **R1** |
| alignment | *(tagger-only, absent from TARGET)* | tile_aligned | **non_tile_aligned** → **R2** |

Everything else that could look like a gap is dispositioned in
`verification_report.md` → "Refinement queue derivation" (out-of-TARGET values:
`ROW_MAJOR_LAYOUT`, `bfloat8_b`; no-axis items: sharded memory config,
`num_links > 1`, multi-core; non-blocking limits: fp32 FPU operand precision,
single-core throughput; residual risk: the cross-call semaphore window).

### [ ] Refinement 1 — Ring topology

**Goal**: add `ttnn.Topology.Ring` to `SUPPORTED["topology"]`, so the op reduces
correctly on a ring as well as a line. Moves the one named failing cell in the
suite — `eval/golden_tests/all_reduce/test_translated.py::test_ring_all_reduce_refinement_axis`
(currently a lenient xfail: `UnsupportedAxisValue: topology=Topology.Ring not in
SUPPORTED [Topology.Linear]`) — to passing, and flips the `Ring` half of any future
TARGET-widened cell set with no edit to the test.

**Verifier notes**: land this **first** — it is the only queue entry with a named
failing cell, and R2 composes on top of whatever routing it produces (a ring route
plus a non-aligned landing buffer is strictly easier than doing both at once).
Two coupled changes, both already scoped by `op_design.md`:
(a) **host** — `line_direction_slots()` currently derives the per-direction
multicast range as "chips beyond the `i+1` neighbour = `N-1-i`, beyond `i-1` = `i`",
which is the *Linear* formula. Ring needs the alternating target counts
(`ccl_common.cpp:1786-1794`, `get_forward_backward_line_mcast_distance`'s
`static_alternate` swap); keep querying `ccl_dm_route(...).is_forward` for the
slotting (it owns the fwd/bwd sign reversal *and* the ring short-way — never assume
`i+1` is fabric-forward, design Risk 4).
(b) **kernel** — the `range_hops == 0` guard becomes **mandatory**. On Linear a
zero-target direction is always also an unconnected direction, so
`DuplexConn::has(dir)` suppresses the arm and the issue; on Ring the alternating
swap can yield `num_targets == 0` while the connection *exists*, and a zero-range
multicast header encodes an all-`NOOP` routing field that hits
`default: ASSERT(false)` in the router (`fabric_erisc_router.cpp:892-894`) — a hang,
not a wrong answer. Implement the explicit guard `broadcast_tile_writer.cpp:95-98`
uses. Also re-check the invariant the receive counter depends on: the two
directions' ranges must still cover every peer **exactly once**, so each device
still receives exactly `N-1` increments — if wraparound double-covers a peer, the
counter overshoots and the *next* call's re-arm eats the surplus.
No skill pointer: no skill in the inventory covers CCL fabric routing / topology
(the inventory is single-device compute precision, in-kernel layouts, interleaved
multi-core, L1 budget). Work from the references above.
Sim caveat: `wh_t3k_allmmio_all_reduce` is a `FABRIC_1D` **line** mesh-graph
descriptor. Confirm the sim can actually route a ring on it *before* concluding
anything from a failure; if it cannot, land + review the host/kernel ring path,
mark `[~]`, file the follow-up as `Refinement 1b`, and record the sim limitation in
`changelog.md` rather than gating `Ring` back out.

**Done when**: `SUPPORTED["topology"]` contains `Ring`;
`test_ring_all_reduce_refinement_axis` passes (it needs no edit — it stops being a
refusal the moment SUPPORTED admits `Ring`); the 6 registry cells, the 5 other
translated cells and all 21 unit-dir tests still pass on the WH sim; and the
`range_hops == 0` guard is present in the writer even if the sim never exercises it.

### [ ] Refinement 2 — non-tile-aligned shards (rank-general landing-buffer mapping)

**Goal**: add `"non_tile_aligned"` to `SUPPORTED["alignment"]`, i.e. accept shards
whose last two dims are not both multiples of 32 (e.g. `(1,1,48,96)`,
`(8,8,8,16,16)`), on every supported rank.

**Implementation skill**: /memory-layouts

**Verifier notes**: **order after R1.** Read the skill for the alignment/retrofit
workflow, but note up front that the usual in-kernel lever does **not** apply here:
this op needs **no last-tile mask and no zero-padding**. It is a pure element-wise
sum over whole tiles, the tile padding is zero-filled by `from_torch` (so it sums
to padding), and `to_torch` strips it on readback — the golden oracle never sees
it. The real blocker is the **op-internal landing buffer's page mapping**
(`_gathered_shape` in `all_reduce.py`, design Risk 13): it scales dim 0 by N, and
"slot `k` == pages `[k·P, (k+1)·P)`" only holds when each shard occupies whole
tile-rows. For rank 2 with `H % 32 != 0` it breaks outright — `(48,64)` has `P = 4`
pages per shard, but the `(384,64)` gathered tensor has 24 pages, not `N·P = 32`.
The natural fix is to stop deriving the landing buffer from the *logical* shape at
all and allocate it by **page count**: a canonical flat TILE shape with exactly
`N · P` tiles (`P = input_tensor.buffer_num_pages()`), which decouples the mapping
from rank and alignment entirely and lets the kernels keep their existing
`k*P + p` arithmetic untouched. Verify `buffer_num_pages()` on the flat buffer is
exactly `N · P` for each new shape before trusting it. Keep the `tag_alignment`
tagger as-is — the axis stays load-bearing; only its accepted value set grows.
**Golden-cell caveat (why this entry has no named failing cell)**: all three
`feature_spec.INPUTS` shapes are tile-aligned, so there are currently **no** golden
cells on the non-aligned side of the tagger; this refinement adds a `SUPPORTED`
value rather than flipping counted cells. Ask `/golden-tests` to add one
non-aligned INPUT (e.g. `((1,1,48,96),)`) so the registry can score it — until
then, gate on the acceptance-suite extension below.

**Done when**: `SUPPORTED["alignment"] == ["tile_aligned", "non_tile_aligned"]`;
`tests/ttnn/unit_tests/operations/all_reduce/test_all_reduce.py` gains
non-tile-aligned shapes covering **both** the rank-2 `H % 32 != 0` case (the one
that provably breaks today) and a rank ≥ 3 case, and they pass on the WH sim; the
existing 21 unit-dir tests and 6 registry cells still pass; and if `/golden-tests`
has added a non-aligned INPUT, its cells show as `supported_pass` with
`xpass_drift = 0`.

---

## Out-of-queue (recorded in `verification_report.md`, not refinements)

- **`layout = ROW_MAJOR_LAYOUT`** and **`dtype = bfloat8_b`** — absent from
  `feature_spec.TARGET` (TARGET pins TILE because the reduction is a tile compute).
  Needs `/golden-tests` to widen TARGET first — and doing so *requires* adding the
  canonical `{bfloat8_b, ROW_MAJOR}` INVALID entry at the same time.
- **Sharded memory config** — `validate()` rejects sharded input; there is no
  `memory_config` axis in TARGET. Interleaved DRAM *and* L1 are both supported and
  now both tested.
- **`num_links > 1` / multi-core per device** — perf only, no axis, no failing
  cell. Not the embarrassingly-parallel `/interleaved-parallel` case: the duplex
  tier is defined as one worker owning both fabric directions and `MuxConn<N>`
  cannot back it, so it needs one fabric link per core.
- **Ring reduce-scatter + all-gather algorithm** — bandwidth-optimal alternative to
  broadcast-all (`2(N-1)` fabric phases instead of 1); perf only. Distinct from R1,
  which is the `Ring` *topology value* on the existing algorithm.
- **fp32 fold precision** (rel-RMS 2⁻¹¹, the WH FPU's 19-bit SrcA/SrcB operands;
  lever = SFPU fold) and **cross-call semaphore re-arm window** (design Risk 5;
  probed by `test_all_reduce_back_to_back_no_sync`, does not reproduce; lever = a
  second parked `GlobalSemaphore` as an entry barrier). Both are analysed in the
  verification report; neither has a failing cell to move.
