# Operation Requirements: all_gather

## Definition
- **Formula**: `output[d] = concat_{c=0..N-1}( input_shard[c], axis=gather_dim )`
  for every device `d` on the N-device line. Identity gather — element values are
  unchanged end to end (pure byte movement, no arithmetic; the oracle is the
  host-side concat, PCC ≈ 1.0).
- **PyTorch Reference** (gather_dim=0, the proven primary case):
  ```python
  def all_gather_ref(in_shards, gather_dim):
      full = torch.cat(in_shards, dim=gather_dim)   # in_shards: list of N per-device shards
      return [full.clone() for _ in in_shards]      # replicated on every device
  ```
- **Import Path**: `from ttnn.operations.all_gather import all_gather`
- **Function Signature**:
  ```python
  def all_gather(
      input_tensor: ttnn.Tensor,
      gather_dim: int,
      *,
      topology: ttnn.Topology = ttnn.Topology.Linear,
      output_tensor: ttnn.Tensor | None = None,
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (the runner parses exactly `Refinement \d+[a-z]?`)**: primary refinements are `Refinement N`; partial follow-ups append a lowercase letter (`Refinement 1b`), ordered immediately after their parent.
> **Verification runner (CCL — read this)**: this is a multi-device op. Verify EVERY refinement on the deterministic WH sim, NOT `run_safe_pytest.sh`:
> `scripts/run_multidevice_sim_pytest.py --op all_gather -- <pytest target> -v`.
> The op's tests MUST open a `(1, 8)` mesh with `FABRIC_1D` (topology `wh_t3k_allmmio_all_gather`) or fabric init hangs ("Fabric Router Sync: Timeout"). A green run here means the cross-device transfer actually executed and PCC asserted.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: `[bfloat16, float32]`
- **SUPPORTED layout**: `[TILE, ROW_MAJOR]`
- **SUPPORTED topology**: `[Linear]`
- **SUPPORTED gather_dim**: `[-4]` (== `gather_dim 0`, page-contiguous concat; negative-canonicalized index axis)
- **SUPPORTED shape-derived axes**: `alignment ∈ {tile_aligned, non_tile_aligned}`
- **Memory**: interleaved (DRAM or L1); sharded rejected by `validate()`
- **Mesh**: 1-D line view `(1, N)`; verified on `(1, 8)` WH `FABRIC_1D`
- **Cores**: 2 worker cores per device — `forward (0,0)`, `backward (0,1)`;
  bidirectional store-and-forward on the line
- **Coordination**: one cached op-internal `GlobalSemaphore` (parked on the
  descriptor); fabric atomic-inc counting + reader-side cache-reuse reset.
  (Design's optional startup barrier intentionally omitted — see verification
  report → Design Conformance; the counting sem + persistent output suffice and
  this avoids the helper's shared-sem-header footgun.)
- **Golden baseline (hybrid)**: 384 golden cells = **32 supported_pass + 288
  xfail_expected + 64 INVALID-skipped** (`bf8b × ROW_MAJOR`); loud categories
  (supported_fail / xpass_drift / xfail_wrong_mode) all **0**. The SUPPORTED
  rectangle was **observed passing on the WH sim**: 22/22 acceptance + 8/8
  precision, aggregate exit 0. See `eval/results/all_gather/verifier_report.json`
  and `verification_report.md`.
- **Accuracy achieved**: PCC = 1.0, max_abs = mean_abs = rel_rms = 0 (bit-for-bit
  identity copy; measured on 4 shapes × {bf16, f32}).

---

## Refinement queue

`TARGET − SUPPORTED` is non-empty on three axes. Each missing `(axis, value)`
pair below appears in exactly one refinement; `bf8b × ROW_MAJOR` is structurally
INVALID (skipped, not queued).

| Axis | TARGET | SUPPORTED (Phase 0) | Gap → Refinement |
|------|--------|---------------------|------------------|
| dtype | bf16, f32, **bf8b** | bf16, f32 | bf8b → **R1** |
| gather_dim | -4, **-3, -2, -1** | -4 | -3,-2,-1 → **R2** |
| topology | Linear, **Ring** | Linear | Ring → **R3** |
| layout | TILE, ROW_MAJOR | (both) | ∅ |
| alignment | tile_aligned, non_tile_aligned | (both) | ∅ |

None of these map onto a current implementation-skill (the inventory covers
single-device compute precision, in-kernel layouts, interleaved multi-core, and
L1 budget — none cover CCL fabric axis expansions), so all three are
verifier-authored with full goal + done-when.

### [x] Refinement 1 — bfloat8_b dtype

**Goal**: add `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`. all_gather is pure byte
movement (it copies physical pages verbatim, never tilizes), so a `bf8b` TILE
tensor should gather exactly like `bf16`/`f32` — the page is just a block-float
tile copied whole. Move the 8 currently-xfail `bf8b × TILE × Linear ×
gather_dim=-4` cells (one per `feature_spec.INPUTS` shape) to passing. The
remaining 56 `bf8b` cells are multi-gap and clear once R2/R3 also land.

**Verifier notes**: expected near-trivial — likely just the `SUPPORTED` entry +
a sim re-run, no kernel change. **This is NOT a `/numeric-formats-metal`
refinement**: that skill is about *compute* precision (math_fidelity,
fp32_dest_acc_en, intermediate-CB formats) and all_gather has no compute stage.
The one thing to verify on the sim: the `bf8b` per-shard `buffer_page_size()`
(a 32×32 block-float tile is 1088 B = 68×16, so it satisfies the kernel's
16B-page invariant) and the `cb_relay` format (`data_format = input.dtype` already
flows `bf8b` through). Land this first — it is independent of R2/R3 and the
cheapest win. Add the precision threshold `bfloat8_b: 0.99` to the precision
baseline's `PCC` map and a `bf8b` row to the test's dtype list. If any `bf8b`
combo genuinely fails the byte-copy (it should not), put that specific combo in
`EXCLUSIONS` rather than fighting it, and note it in the changelog.

**Done when**: `SUPPORTED["dtype"]` includes `bfloat8_b`; the `bf8b × TILE ×
Linear × gather_dim=0` cells pass on the WH sim; no regression in the existing
22 acceptance + 8 precision cells.

### [ ] Refinement 2 — gather_dim != 0 (strided concat addressing)

**Goal**: add `-3, -2, -1` to `SUPPORTED["gather_dim"]`. For `gather_dim = 0`
each device's shard maps to a *contiguous* output page block
(`out_page = c·P + p`). For `gather_dim != 0` the shard maps to a *strided* page
set — the concat interleaves shards along an inner dim, so the destination page
index is `tile_id_start = position·stride` with row wrapping by the output's
width-in-tiles, exactly as the reference writer computes it
(`minimal_default_writer.cpp:247-256, 449-457`; reader read-back mirror at
`minimal_default_reader.cpp:200-252`). Implement that addressing in the reader's
self-copy / relay read-back and the writer's `write_page` page-index math, keyed
on the normalized 0..3 `gather_dim` CT arg the host already passes. Move the 240
currently-xfail `gather_dim ∈ {-3,-2,-1}` cells (Linear; the Ring subset clears
with R3) to passing.

**Verifier notes**: this is the largest kernel change — it rewrites the
output-page addressing in both the reader (read-back) and writer (fabric write)
so the contiguous-block assumption (`c·P + p`) is replaced by the strided
concat-by-`gather_dim` index. The host already canonicalizes `gather_dim` to the
4-D 0..3 slot and passes it; the kernels currently ignore it (they hard-code the
contiguous path). Sequence R2 before R3 so the ring slice-walk in R3 is validated
against an already-correct strided-addressing path (Ring × `gather_dim != 0`
combos exist in TARGET). Add a `gather_dim` parametrization (e.g. `-1`, `-2`) to
a small extension of the acceptance/precision tests. No skill pointer — CCL
concat addressing is outside the current skill inventory; work from the reference
writer/reader cited above.

**Done when**: `SUPPORTED["gather_dim"]` includes `-3, -2, -1`; the
`gather_dim ∈ {-3,-2,-1} × Linear` golden cells pass on the WH sim; gather_dim=0
and all Phase-0 cells still pass.

### [ ] Refinement 3 — Ring topology

**Goal**: add `ttnn.Topology.Ring` to `SUPPORTED["topology"]`. On a *line*, the
op uses bidirectional forward/backward workers (low-index shards flow right,
high-index flow left). On a *ring*, a single direction with wraparound suffices:
the slice-walk becomes modular — `actual_slice_chip_id = (my_chip_id ± k) mod N`
— and the per-direction route uses the ring short-way (`ccl_dm_route` already
owns the short-way + sign for `Topology.Ring`). Select the ring path off the
existing `topology` CT arg + the (currently unused, reserved) `ring_size` CT arg.
Move the 160 currently-xfail `topology = Ring` cells to passing.

**Verifier notes**: this is an algorithm-fundamental change (the slice-walk and
the worker/route structure change for a ring vs a line), so it stands alone. No
skill in the inventory covers CCL ring topology / cross-device-dependency
routing — work from the reference's slice-walk modulo math
(`minimal_default_writer.cpp` ring path) and `ccl_dm_route`'s ring short-way.
Sequence LAST: it must compose with both bf8b (R1) and strided `gather_dim` (R2),
so landing those first means Ring only adds the modular routing on top of an
otherwise-complete SUPPORTED rectangle. The host `create_mesh_program_descriptor`
must compute ring routes/`num_targets` (wraparound) and the writer/reader the
modular block ids; re-confirm the counting-sem increment/wait counts still match
under wraparound. Add a `topology=Ring` parametrization to the acceptance test
(the WH sim is a `FABRIC_1D` line — confirm the sim supports the ring routing
config before drawing conclusions; if the sim topology cannot exercise Ring,
land the kernel/host changes, mark `[~]`, and note the sim limitation in the
changelog rather than gating it out).

**Done when**: `SUPPORTED["topology"]` includes `Ring`; the `topology = Ring`
golden cells pass on the WH sim (or, if the sim cannot exercise ring routing, the
ring kernel/host path is implemented + reviewed and the limitation is documented
per the partial-tick protocol); all Linear cells still pass.

---

## Out-of-queue (recorded in `verification_report.md`, not refinements)

- **Sharded memory config** — `validate()` rejects sharded input; TARGET has no
  `memory_config` axis, so not a refinement candidate until `/golden-tests`
  widens TARGET.
- **Multi-link / multi-core fan-out per direction** — single worker core per
  direction by design; a perf change with no SUPPORTED axis or failing cell.
- **Forward-reader double input read / per-page fabric flush** — perf levers with
  no failing cell to point at.
