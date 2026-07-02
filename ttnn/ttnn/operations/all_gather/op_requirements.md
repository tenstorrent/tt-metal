# Operation Requirements: all_gather

## Definition
- **Formula**: `output[d] = concat_{c=0..N-1}( input_shard[c], axis=gather_dim )` for every
  device `d` on the 1-D line of `N` devices. Identity gather (pure data movement, no
  arithmetic) — after the op every device holds the full concatenated tensor, identical on all.
- **PyTorch Reference**:
  ```python
  def all_gather_ref(shards: list[torch.Tensor], gather_dim: int) -> torch.Tensor:
      # shards[c] is device c's per-device shard; result is replicated to every device.
      return torch.cat(shards, dim=gather_dim)
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
  Input: mesh-sharded along `gather_dim` across a `(1, N)` line mesh, interleaved DRAM/L1,
  TILE or ROW_MAJOR, bfloat16/float32. Verified on WH T3K sim, mesh `(1, 8)`, `FABRIC_1D`.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to
> update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests
> pass, `[~]` when real work landed but at least one named axis value is deferred (treated as
> completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements
> are `Refinement N`. Partial-tick follow-ups append a lowercase letter to the parent
> (`Refinement 1b`), ordered immediately after their parent. The runner's parser matches exactly
> `Refinement \d+[a-z]?`.
>
> **Verification vehicle**: all_gather is a multi-device CCL op — verify with
> `scripts/run_multidevice_sim_pytest.py --op all_gather -- <target>` (mesh `(1, 8)` +
> `FABRIC_1D`, matching the `wh_t3k_allmmio_all_gather` topology). **Never** use
> `run_safe_pytest.sh` (forces slow dispatch, no multichip awareness). The full 384-case golden
> cartesian cannot complete under the runner's 900 s cap with the function-scoped `mesh_device`
> fixture — select a curated subset per axis under test.

### [x] Phase 0 — Core Implementation

Bidirectional store-and-forward ring on a 1-D line; two worker cores/device (forward `(0,0)`
dir 0 → i+1, backward `(0,1)` dir 1 → i-1), reader=NCRISC + writer=BRISC, no compute kernel.
Fabric egress via the CCL helper (`ccl_helpers_dataflow.hpp`); two parked `GlobalSemaphore`s
(barrier + counting).

- **SUPPORTED dtype**: [bfloat16, float32]
- **SUPPORTED layout**: [TILE, ROW_MAJOR]  ← both, no gap vs TARGET
- **SUPPORTED topology**: [Linear]
- **SUPPORTED gather_dim**: [-4]  (page-contiguous concat; canonicalized to negative in validate)
- **SUPPORTED alignment**: [tile_aligned, non_tile_aligned]  ← both, no gap vs TARGET
- **Cores**: fixed CCL split — 2 workers/device, one per direction (no interleaved stamp to add)
- **Golden baseline**: 8/8 supported cells passing, bit-exact (PCC=1.0, RMS=0, max_abs=0);
  40 xfail_expected; 8 invalid_skipped; all loud categories 0 (curated 56-case subset of 384).
- **Verifier fix applied**: WAR hazard on the relay/backward-seed CB slot (NON_BLOCKING fabric
  payload send + immediate `cb_pop_front` + concurrent reader refill) — added
  `noc_async_write_barrier()` before pop, consistent with the forward-seed guard.

---

### TARGET − SUPPORTED gap (source of the queue)

| Axis | TARGET | SUPPORTED (Phase 0) | Missing → disposition |
|------|--------|---------------------|-----------------------|
| dtype | bf16, f32, **bf8b** | bf16, f32 | `bf8b` → Refinement 3 (bf8b+ROW_MAJOR is INVALID) |
| layout | TILE, ROW_MAJOR | TILE, ROW_MAJOR | — none |
| topology | Linear, **Ring** | Linear | `Ring` → Refinement 2 |
| gather_dim | **-4, -3, -2, -1** | -4 | `-3, -2, -1` → Refinement 1 |
| alignment | tile_aligned, non_tile_aligned | both | — none |

Every `(axis, missing_value)` pair is covered by a refinement below or by INVALID. No queue gap.

---

### [ ] Refinement 1 — gather_dim expansion (-3, -2, -1): dim-agnostic strided concat

**Goal**: add `-3, -2, -1` to `SUPPORTED["gather_dim"]`. For `gather_dim = 0` (the -4 primary
case) each shard maps to a **contiguous** block of output pages
(`out_page = c*pages_per_shard + p`). For inner dims the concat interleaves shards along a
non-outermost axis, so the block becomes a **strided page set** — compute it exactly as the
reference writer does (`tile_id_start = position * stride`, row-wrapping by `output_tensor_Wt`;
`minimal_default_writer.cpp:247-256, 449-457`). The self-copy, fabric `write_page` dst, and
relay read-back addressing all move from `c*pages_per_shard + p` to the strided form; the
ring/barrier/counting machinery is unchanged. Done when the formerly-xfail `gather_dim ∈
{-3,-2,-1}` golden cells pass (all dtype × layout, Linear).

**Verifier notes**: no skill in the inventory covers CCL concat-by-gather_dim addressing —
verifier-authored. Highest-value lever (extends the op's core functionality to arbitrary
concat dims) and the most design-documented refinement path (`op_design.md` "Dataflow
Strategy" → page-contiguous vs strided). Independent of Refinements 2 and 3, but do it first:
it touches the same output-addressing code that Ring and bf8b will inherit, so landing the
strided addressing before them avoids re-touching it. Watch the non-tile-aligned interaction —
strided concat on inner dims plus non-`%32` H/W is the trickiest cell; if it needs structural
work beyond the reference's strided walk, `[~]`-partial the aligned-only case and file
`Refinement 1b` for the non-aligned strided path.

---

### [ ] Refinement 2 — Ring topology

**Goal**: add `ttnn.Topology.Ring` to `SUPPORTED["topology"]`. On a Ring, one direction with
wraparound suffices (vs the Linear line's mandatory bidirectional flow): the slice-walk becomes
`actual_slice_chip_id = (my_chip_id ± k) mod ring_size` and each device forwards to a single
wraparound neighbour. Host route computation already flows through
`ttnn._ttnn.fabric.ccl_dm_route(..., topology)`; pass `Ring` through, adjust
`num_targets_{fwd,bwd}` / the mcast range for the ring closure, and switch the kernel slice-walk
to the modulo form (topology is already a CT arg per the design). Done when the formerly-xfail
`topology=Ring` golden cells pass.

**Verifier notes**: verifier-authored — the communication topology *is* the work; no skill
covers CCL ring/line dataflow. Independent of Refinements 1 and 3. Order after Refinement 1 so
the ring slice-walk is built on the finished dim-agnostic addressing (a Ring gather along an
inner dim must combine both). This is algorithm-fundamental (rewrites the ring-walk /
direction logic), so it correctly stands alone rather than bundling.

---

### [ ] Refinement 3 — bfloat8_b dtype

**Goal**: add `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` (TILE only — `bf8b + ROW_MAJOR` is
INVALID and stays skipped). The op is pure byte movement, so a bf8b tile is copied verbatim as
its block-float page; the work is confirming the reader/writer page geometry
(`buffer_page_size()` / `buffer_aligned_page_size()`) is correct for the block-float layout and
that the per-page payload frames into one fabric packet — use
`ttnn._ttnn.fabric.ccl_packet_dims(dtype, page_size, num_pages, l1_alignment)` (it owns the
bf16 `bit_floor` for on-wire framing) if the raw page size needs segmentation. Done when the
formerly-xfail `dtype=bfloat8_b` + TILE golden cells pass; `bf8b + ROW_MAJOR` remains
invalid_skipped.

**Verifier notes**: verifier-authored. **`/numeric-formats-metal` does NOT apply** — that skill
configures compute-kernel precision (`ComputeKernelConfig`, math fidelity, dest-acc,
intermediate-CB formats, `UnpackToDestFp32`); this op has *no compute kernel*, so none of that
surface exists. bf8b here is purely page/packet geometry. Smallest of the three and orthogonal
(format, not addressing/topology) — order last. Cross-cutting: if landed after Refinements 1/2,
confirm the strided-concat and ring code paths also copy bf8b pages verbatim (they will — byte
copy is dtype-agnostic once the page size is right).

**Done when**: `dtype=bfloat8_b` TILE cells pass across the supported gather_dim/topology set;
bf8b+ROW_MAJOR stays skipped (INVALID).
