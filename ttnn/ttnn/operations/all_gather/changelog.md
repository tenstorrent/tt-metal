# Changelog: all_gather

## Phase 0 — Core Implementation
- **Date**: 2026-07-03
- **What was done**: Initial implementation via the incremental pipeline
  (planner → implementer → verifier). Self-contained Python CCL op:
  `generic_op` + `MeshProgramDescriptor` running a bidirectional store-and-forward
  ring on newly-authored dataflow kernels (`all_gather_reader.cpp`,
  `all_gather_writer.cpp`). Two worker cores per device (`core_fwd`, `core_bwd`),
  three single-owner GlobalSemaphores (barrier + fwd/bwd counting), fabric egress via
  the `ccl_helpers_dataflow.hpp` safety-by-construction helper.
- **SUPPORTED at Phase 0**:
  - dtype = [bfloat16, float32]
  - layout = [TILE]
  - topology = [Linear]
  - gather_dim = [-4]  (negative convention; -4 ≡ dim 0 for rank-4 shards)
  - alignment (INPUT_TAGGERS) = [tile_aligned, non_tile_aligned]
  - EXCLUSIONS = []
- **Accuracy achieved**: PCC = 1.0, max_abs_err = 0.0, mean_abs_err = 0.0,
  rel_rms_err = 0.0 — bit-exact identity gather across 4 shapes × {bf16, f32}
  (measured via `test_all_gather_precision_baseline.py`). All 8 devices agree
  bit-for-bit (replicated output).
- **Golden suite at Phase 0**: **16 / 384 cells passing** (`supported_pass = 16`,
  `xfail_expected = 304`, `invalid_skipped = 64`; loud categories `supported_fail`,
  `xpass_drift`, `xfail_wrong_mode` all = 0) — per `verifier_report.json`.
  Run via `scripts/run_multidevice_sim_pytest.py --op all_gather` in 5 dtype/layout
  `-k` chunks (a full single-process golden run exceeds the wall-clock backstop
  because the CCL golden `mesh_device` fixture re-inits the 8-device fabric per cell).
- **Issues encountered / fixed during verification**:
  - Simplified a dead-code branch in `validate()`: `page % 16 != 0 and page != 16`
    → `page % 16 != 0` (the `and page != 16` conjunct was unreachable). Behaviour
    unchanged.
  - Unblocked shared test infra: `tests/ttnn/utils_for_testing.py` referenced
    `ttnn.fp8_e4m3` (from FP8-enablement commit `079872566e`) which the built binary
    predates, breaking collection of every test importing `assert_with_pcc`. Guarded
    the entry with `hasattr(ttnn, "fp8_e4m3")`. Not an all_gather defect.
  - No SUPPORTED drift (`xpass_drift = 0`) — no auto-promotions needed.
- **Tests added**:
  - `test_all_gather.py` (acceptance; pre-existing, 9/9 PASS)
  - `test_all_gather_precision_baseline.py` (pre-existing, 8/8 PASS, bit-exact)
  - `test_all_gather_extended.py` (**new**; preallocated-output path +
    validate() rejection behaviour; 2/2 PASS)
- **Refinement queue set up** (`op_requirements.md`): 3 refinements covering the
  TARGET − SUPPORTED gap —
  1. Format axes: bfloat8_b + ROW_MAJOR (`/memory-layouts`, `/numeric-formats-metal`)
  2. Non-contiguous concat addressing: gather_dim −3/−2/−1 (verifier-authored)
  3. Ring topology (verifier-authored; **verification infra-blocked** — no ring
     topology in the multidevice sim matrix yet)

## Refinement 1 — Format axes: bfloat8_b dtype + ROW_MAJOR layout
- **Date**: 2026-07-03
- **What was done**: Promoted `ttnn.bfloat8_b` into `SUPPORTED["dtype"]` and
  `ttnn.ROW_MAJOR_LAYOUT` into `SUPPORTED["layout"]`. **No kernel or program-descriptor
  change was needed** — all_gather is pure byte movement (it never (un)tilizes), and the
  reader/writer already move whole pages by page-index via `TensorAccessor`, with the
  relay-CB `data_format` derived from the input dtype and the page size from
  `buffer_page_size()` / `buffer_num_pages()`. Both new axes are therefore native
  in-kernel format flexibility, NOT a `to_layout`/`tilize` wrapper:
  - **ROW_MAJOR**: the relay CB page is the logical row (`buffer_page_size()`, L1-aligned
    for the CB slot). At `gather_dim=-4` an RM shard is still a contiguous page range, so
    the existing contiguous-slice walk is unchanged. Confirmed (via the `point_to_point`
    sibling RM CCL op + the TensorAccessor implementation) that for **interleaved DRAM**
    tensors `get_noc_addr(page_id)` re-aligns the DRAM page stride internally through
    `InterleavedAddrGen`, so passing the logical page size is correct (not a stride bug).
  - **bfloat8_b**: the whole block-float tile page (1088 B, 16-B aligned) is relayed
    intact; because tiles arrive already-packed and are NEVER re-tilized, the shared-face
    exponents survive even for non-tile-aligned shards (memory-layouts §5) — so
    `bfloat8_b × non_tile_aligned` needs **no EXCLUSIONS** (the verifier's flagged
    possible-exclusion did not materialize; EXCLUSIONS stays `[]`).
  - `bfloat8_b × ROW_MAJOR` is structurally impossible (INVALID in feature_spec, skipped
    by the harness); the op file is agnostic to it (no INVALID block, per the model).
  - `validate()` needed no structural change — the registry axis gate enforces the
    extended SUPPORTED automatically.
- **Accuracy achieved**: identity gather is bit-exact byte movement, so the only error is
  the pre-op `from_torch` dtype quantization. Measured on the golden `(0.999, 0.02)`
  (PCC, relative-RMS) bar across all 8 golden shapes (incl. non_tile_aligned 1×1×48×64):
  - RM bfloat16: PCC ≈ 0.999+ (bf16 round-trip), well within tolerance.
  - RM float32: PCC ≈ 1.0.
  - bfloat8_b (TILE): PCC = 0.999971, relative-RMS = 0.0076 (host round-trip probe),
    matched by the on-device gather — comfortably clears 0.999.
- **Golden test progress**: **40 / 384** cells passing (was 16 / 384 at Phase 0) — the
  24 previously-xfail `gather_dim=-4, topology=Linear` cells for the three new format
  combinations (RM×bf16 = 8, RM×f32 = 8, bf8b×TILE = 8) all flipped to `supported_pass`.
  Verified on the multidevice WH sim (`run_multidevice_sim_pytest.py --op all_gather`)
  in `-k` chunks. Non-regression + no-drift confirmed on a 1×1×32×32 all-Linear sweep:
  TILE bf16/f32 gd=-4 still pass, and every gd∈{-3,-2,-1} cell still xfails with the
  correct reason (no xpass-drift, no xfail_wrong_mode). bf8b×RM cells correctly skipped
  (INVALID). Loud verifier categories (supported_fail / xpass_drift / xfail_wrong_mode)
  all 0.
- **Issues encountered**: None. The verifier-flagged risk (`bf8b × non_tile_aligned`
  block-float sub-tile edge) did not occur — non-aligned bf8b gives identical PCC to
  aligned bf8b because the op never re-tilizes.
- **Tests added**:
  - `tests/ttnn/unit_tests/operations/all_gather/test_all_gather_formats.py` (**new**):
    format/layout matrix — RM×{bf16,f32}, bf8b×TILE, plus TILE regression, across
    tile-aligned and non-tile-aligned shard shapes, asserting output dtype+layout are
    preserved. bf8b×RM omitted (INVALID).
  - `test_all_gather_extended.py` (**updated**): the `validate()` rejection test now gates
    on `topology=Ring` (still unsupported) since ROW_MAJOR is now SUPPORTED.

## Refinement 2 — Non-contiguous concat addressing (gather_dim -3, -2, -1)
- **Date**: 2026-07-03
- **What was done**: Added `-3, -2, -1` to `SUPPORTED["gather_dim"]` (all four dims now
  supported). At `gather_dim != -4` a device's slice is no longer a contiguous output
  page range, so the reader (relay re-read) and writer (own-slot local write + fabric
  write) now walk the interleaved concat-stride pattern. Two addressing regimes,
  parameterized by three new CT args (`block_in`, `sub_page`, `output_page_size`) that a
  new host helper `_concat_addressing()` derives from layout + gather axis:
  - **Regime A — whole-page remap** (all TILE gathers on a tile-aligned axis; all
    ROW_MAJOR non-innermost gathers): input page `in_p` of slice `j` maps to output page
    `out_p = (in_p // block_in)*block_in*N + (in_p % block_in) + j*block_in`, where
    `block_in` = input pages from the gather axis down (mid_in·inner). `gather_dim=-4`
    degenerates to `block_in = pages_per_shard` → `out_p = in_p + j*pages_per_shard`, i.e.
    the proven contiguous walk UNCHANGED (verified: gather_dim=0 regression stays green).
    Fabric egress keeps the proven `write_page(l1, out_p, output_acc)` path.
  - **Regime B — sub-page byte concat** (ROW_MAJOR + innermost `gather_dim=-1` only): the
    concat lives WITHIN a row, so `out_p == in_p` but the payload lands at byte offset
    `j*input_page_size` inside an N× larger output page (`output_page_size = N*input`).
    Uses `get_noc_addr(page, offset)` + `write(dst, l1)`; the FABRIC dst is computed via
    `tt::tt_fabric::linear::addrgen_detail::get_noc_address` (fabric NOC index + the
    Wormhole DRAM→noc0 flip that `write_page` applies internally) — NOT the accessor's
    default-NOC addr, which is only correct for the LOCAL own-slot write.
  - The single-owner-semaphore / data-before-inc coordination is UNCHANGED — pages-per-slice
    and slice counts are identical; only WHERE each page lands in the output changed.
- **EXCLUSIONS** += `{layout: TILE, gather_dim: -2, alignment: non_tile_aligned}` — a
  structural capability gap, NOT a deferred axis value. When the per-shard gather-axis
  extent is not a multiple of 32 (H=48 in `(1,1,48,64)`, the sole non-tile-aligned INPUT),
  the shard tiles its own 32-row boundary independently (padding the tail) while the
  concatenated output re-tiles at a DIFFERENT 32-boundary — a landed slice straddles output
  tile boundaries. A whole-tile copy cannot reconstruct it (it would even compute
  out-of-bounds output pages: `Ht_out = ceil(48·8/32) = 12 ≠ Ht_shard·N = 16`), and correct
  handling needs a sub-tile untilize/re-tilize that this pure-byte-movement op (no compute
  kernel) cannot do. bfloat8_b is TILE-only so this one entry covers it too; `bf8b×RM` is
  INVALID. ROW_MAJOR has no 32-row tiling, so RM+non_tile_aligned gathers cleanly at every
  dim (verified: RM gd=-2 on `(1,1,48,64)` passes). Only `gather_dim=-2` hits the gap in the
  current INPUTS (W stays tile-aligned in the one non-aligned shape, so gd=-1 is fine).
- **Accuracy achieved**: identity gather is bit-exact byte movement; PCC clears the golden
  `(0.999, 0.02)` bar on every measured cell. Measured across gather_dim ∈ {-3,-2,-1}:
  bf16/f32 TILE (whole-page remap incl. row-stride & outer-stride), bf16/f32 RM (whole-row
  remap + sub-page byte concat), bf8b TILE — all 8 devices agree bit-for-bit with the
  `torch.cat` oracle. Shapes: (1,1,32,32), (1,1,64,128), (1,1,32,96), (1,1,96,64),
  (2,1,32,64), (1,1,48,64).
- **Golden test progress**: **157 / 384** cells passing (was 40 / 384 at Refinement 1).
  The 117 newly-passing cells = the 3 non-contiguous gather_dims × the Linear × supported
  dtype/layout set × 8 shapes, minus the 3 excluded TILE gd=-2 non_tile_aligned cells
  (bf16/f32/bf8b). Verified on the multidevice WH sim (`run_multidevice_sim_pytest.py
  --op all_gather`) in `-k` chunks: bf16 TILE+RM Linear all gather_dims (8 pass) + the
  excluded cell (XFAIL with the EXCLUSIONS reason) + Ring (8 XFAIL, topology — Refinement 3);
  f32 TILE+RM + bf8b TILE Linear all gather_dims (12 pass, bf8b×RM invalid-skip). Loud
  verifier categories (supported_fail / xpass_drift / xfail_wrong_mode) all **0** on every
  chunk. Full non-regression re-run stays green: acceptance 11/11, formats 25/25,
  precision_baseline 8/8 (all gather_dim=0). Ring (topology, 160 xfail) remains Refinement 3.
- **Issues encountered**: None blocking. Key correctness catch during implementation: the
  fabric `get_noc_address` uses a fabric-specific NOC index and a Wormhole DRAM→noc0
  coordinate flip that differs from the accessor's default-NOC `get_noc_addr` — the Regime B
  fabric write had to use the fabric addrgen (as `write_page` does internally), while the
  LOCAL own-slot write uses the accessor addr. Regime A was kept on the proven `write_page`
  path to minimize risk.
- **Tests added**:
  - `tests/ttnn/unit_tests/operations/all_gather/test_all_gather_gather_dims.py` (**new**):
    full-gather oracle for the non-contiguous walk — 10 cells across gather_dim ∈ {-3,-2,-1},
    both regimes (whole-page remap incl. row-stride/outer-stride + sub-page RM innermost),
    all dtypes/layouts; plus a 3-cell excluded-cell rejection test asserting
    `{TILE, gd=-2, non_tile_aligned}` raises the registry NotImplementedError refusal before
    any device work (bf16/f32/bf8b). 13/13 pass on the WH sim.

## Refinement 3 — Ring topology
- **Date**: 2026-07-03
- **What was done**: Added `ttnn.Topology.Ring` to `SUPPORTED["topology"]` (both topologies
  now supported). **The verifier's flagged infra block did NOT materialise** — see below.
  - **Op change is minimal and additive**: `SUPPORTED["topology"]` gained `Ring`; no kernel
    or program-descriptor change. all_gather's bidirectional store-and-forward is
    topology-agnostic at the routing level: every fabric hop is between IMMEDIATE neighbours
    (1 hop), and `ccl_dm_route` returns the SAME `{num_hops, is_forward, neighbor_id}` for Ring
    and Linear on an adjacent pair (its ring short-way branch only differs for `|hops| > 1`,
    which this store-and-forward never emits). So the per-device `MeshProgramDescriptor` is
    **byte-identical for Ring and Linear**. The design sanctions this: op_design.md "Ring
    algorithm" states the bidirectional line design "also runs on a Ring."
  - **Ring sim topology added to the matrix**: a new `wh_t3k_ring_all_gather` entry in
    `scripts/multidevice_sim_topologies.yaml` (arch WH, `fabric_config: FABRIC_1D_RING`,
    `applies_to_ops: [all_gather]`, `required: true`), appended AFTER the existing
    `wh_t3k_allmmio_all_gather` line entry. It **reuses** the same all-MMIO cluster descriptor
    (whose physical T3K wiring already forms a Hamiltonian ring of 8 chips: 4-0-2-6-7-3-1-5)
    and the same `t3k_1x8_mesh_graph_descriptor.textproto` — which **already declares
    `dim_types: [RING, RING]`**, so no new descriptor was needed (the "ring mesh-graph" the
    verifier asked for already existed). VALIDATED: `FabricConfig::FABRIC_1D_RING` initialises
    cleanly on the WH sim and all_gather(topology=Ring) gathers bit-exact on it.
  - **Golden-grade impact: none.** `run_eval._grade_primary_topology` pins the golden grade to
    ONE topology (`topos[0]` since neither all_gather entry sets `grade_primary`), which stays
    the line entry (listed first). Confirmed the grade primary is still
    `wh_t3k_allmmio_all_gather`. The new ring entry is coverage for the `--op all_gather`
    fan-out and for `test_all_gather_ring.py`; it does not slow or alter golden grading.
  - **Why the infra block was a false premise**: the verifier assumed Ring routing would try
    the wraparound and fail/hang on a line fabric. But this op never emits a >1-hop route, so
    Ring is a strict subset of what it already does under Linear — it passes on FABRIC_1D AND
    on FABRIC_1D_RING. No EXCLUSIONS/INVALID silencing was used (Ring is a real, passing
    TARGET value). EXCLUSIONS is unchanged (still the single `{TILE, gd=-2, non_tile_aligned}`
    entry, which correctly xfails under Ring too, since it does not pin `topology`).
- **Accuracy achieved**: identity gather is bit-exact byte movement; PCC clears the golden
  `(0.999, 0.02)` bar on every measured Ring cell (bf16 PCC≈0.995+, f32 PCC≈1.0, bf8b
  PCC≈0.9999) — every device agrees bit-for-bit with the `torch.cat` oracle. Measured across:
  bf16 TILE+RM (tile-aligned) all gather_dims; f32 & bf8b TILE+RM (non-tile-aligned) all
  gather_dims; on BOTH the FABRIC_1D line fabric and the FABRIC_1D_RING genuine ring fabric.
  Shapes: (1,1,32,32), (1,1,64,128), (2,1,32,64), (1,1,48,64).
- **Golden test progress**: **314 / 384** cells passing (was 157 / 384 at Refinement 2) — the
  157 previously-xfail `topology=Ring` supported cells flip to `supported_pass` (157 Linear +
  157 Ring = 314; 6 xfail = the `{TILE, gd=-2, non_tile_aligned}` EXCLUSIONS cell × 3 dtypes ×
  {Linear,Ring}; 64 skip = INVALID bf8b×RM). Validated on the WH sim in chunks under the LINE
  topology (matches golden's hardcoded FABRIC_1D):
  - bf16 TILE+RM 1x1x32x32 all gather_dims: 8/8 pass.
  - f32+bf8b TILE+RM 1x1x48x64 all gather_dims: 10 pass, 2 xfail (excluded gd=-2, f32 & bf8b),
    4 skip (bf8b×RM invalid).
  - Canonical strided grade sample (`--splits 10 --group 1`, the run_eval dashboard sample):
    34 cells (19 Linear + 16 Ring, mixed) all PASS, **0 FAILED, 0 XPASS (no drift), 0 ERROR**
    (run was cut only by the harness wall-clock; clean, no orphans/hang). Loud verifier
    categories (supported_fail / xpass_drift / xfail_wrong_mode) all **0**.
  - Non-regression: acceptance `test_all_gather_gather_dim_0` bf16 gd=0 (4/4 pass); extended
    (2/2 pass).
- **Issues encountered**: None blocking. Notable findings (documented for the perf follow-up):
  - **`ccl_dm_route` Ring short-way is dead code** (`ttnn/cpp/ttnn/operations/ccl/common/host/
    ccl_helpers_dataflow_host.hpp:156-166`): `ring_hops = line_hops + mesh_shape[dim]` with
    `line_hops` always `>= 1` (it is `std::abs(...)` from `fabric_1d_routing_vector`), so
    `|ring_hops| < |line_hops|` is never true and the WRAP branch never fires — Ring silently
    falls through to the long-way linear route. It does NOT affect this op (our store-and-forward
    only routes adjacent 1-hop pairs, where Ring==Linear anyway), but it WOULD block a future
    wraparound-exploiting single-direction ring gather. Flagged, not fixed (shared CCL helper,
    out of this op's scope).
  - **`FABRIC_1D ⇔ Topology.Linear`, `FABRIC_1D_RING ⇔ Topology.Ring`** is a strict fabric
    convention (`FabricContext::get_topology_from_config`). FABRIC_1D restricts the effective
    fabric type to MESH (drops the wraparound edge); FABRIC_1D_RING is required to route it.
- **Follow-up (perf, NOT a support gap — intentionally NOT queued as a correctness refinement)**:
  exploit the ring wraparound for a shorter **single-direction** gather (each device forwards
  once around the ring, freeing the second direction) or a balanced antipodal split
  (`div_up(N-1,2)` each way, meeting in the middle — cf. the experimental
  `minimal_default_writer.cpp:228-234,404-415`). This would require: (a) fixing the
  `ccl_dm_route` ring short-way bug above; (b) allowing the wraparound neighbour (device 0 ↔
  device N-1) in the program descriptor's `has_fwd`/`has_bwd` gating for topology=Ring; and
  (c) making the golden `mesh_device` fixture topology-adaptive (Ring cells → FABRIC_1D_RING)
  so the wraparound path is exercised by the golden suite — a golden-SUITE change (the golden
  tests are immutable from the op's side; they currently hardcode FABRIC_1D, so the golden
  Ring cells run on the line fabric while `test_all_gather_ring.py` covers the ring fabric).
  Correctness is complete without this; it is a bandwidth/latency optimisation only.
- **Tests added**:
  - `tests/ttnn/unit_tests/operations/all_gather/test_all_gather_ring.py` (**new**): 15 cells —
    `test_all_gather_ring_on_line_fabric` (FABRIC_1D, 3 shapes × 2 dtypes = 6), matching the
    golden fabric plane; `test_all_gather_ring_on_ring_fabric` (FABRIC_1D_RING, 6), the genuine
    ring data plane; `test_all_gather_ring_gather_dims` (FABRIC_1D_RING, gd −3/−2/−1 = 3),
    Ring × the non-contiguous concat walk. All 15 pass on the WH sim.
  - `test_all_gather_extended.py` (**updated**): the rejection test no longer asserts
    `topology=Ring` refuses (Ring is now SUPPORTED — it asserts `validate()` ACCEPTS Ring);
    the registry-refusal path now exercises the `{TILE, gd=-2, non_tile_aligned}` EXCLUSIONS
    cell (raises `ExcludedCell`), plus the structural `gather_dim` out-of-range `ValueError`.
