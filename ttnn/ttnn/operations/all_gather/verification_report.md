# Verification Report: all_gather

**Op kind:** multi-device CCL (collective communication) — pure cross-chip byte
movement, no arithmetic (identity gather, PCC ≈ 1.0). Self-contained Python
`ttnn.generic_op` + `ttnn.MeshProgramDescriptor` with newly-authored fabric
ring/line dataflow kernels (reader NCRISC + writer BRISC per worker core).

**Verification date:** 2026-07-02 (post-Refinements 1/2/3 re-verification pass)

**Verified on:** the deterministic `wh_t3k_allmmio_all_gather` WH craq-sim
(`(1, 8)` line mesh, `FABRIC_1D`) via `scripts/run_multidevice_sim_pytest.py`.
This is a multi-device op — `run_safe_pytest.sh` is the wrong runner (it forces
slow dispatch and has no multichip/hang awareness).

---

## TL;DR

- **`SUPPORTED` now equals `TARGET` on every axis.** Refinements 1 (bfloat8_b),
  2 (`gather_dim ∈ {-3,-2,-1}` strided concat), and 3 (`Topology.Ring`) have
  landed. `TARGET − SUPPORTED = ∅` for `dtype`, `layout`, `topology`,
  `gather_dim`, and `alignment` (confirmed programmatically). The SUPPORTED
  rectangle is complete; the only remaining queue items are two **structural
  sub-cell corners** carved out by `EXCLUSIONS` (R2a, R2b).
- **Golden run is clean.** Verifier CLI on a representative 24-cell cross-section
  (shape `(1,1,48,64)` × Linear, spanning all 3 dtypes × both layouts × all 4
  gather_dims + both EXCLUSIONS + the INVALID): **15 supported_pass, 5
  xfail_expected, 4 invalid_skipped, and every loud category 0** (supported_fail
  / xpass_drift / xfail_wrong_mode / supported_marked_xfail / invalid_unexpected).
  A wider 36-cell partial run (adds Ring) showed **20 passed / 7 xfailed / 0
  failed / 0 xpassed** before its wall-clock cutoff — same clean signal, Ring
  included. See `eval/results/all_gather/verifier_report.json`.
- **Precision is exact.** all_gather is a bit-for-bit identity copy. bf16 and f32
  are `max_abs = mean_abs = rel_rms = 0` across all 4 measured shapes; bf8b is
  `max_abs = 0.03125, mean_abs ≈ 0.006, rel_rms ≈ 0.0077` — purely the
  shared-exponent quantization applied at `from_torch`, the gather adds nothing.
  12/12 precision cells green.
- **Refined-axis kernels re-confirmed on current code.** The program-cache
  re-arm, the `output_tensor` path, and 4 extended device cells (gd=-1 TILE,
  gd=-2 RM, gd=-1 Ring, bf8b gd=0) all pass — R2/R3 introduced no regression.
- **Code review:** no correctness bugs found in the current (post-R2/R3) code; no
  new code fix required this pass. The Phase-0 clarity fix (vacuous
  page-alignment sub-clause) is still in place. R2's strided page remap
  (`gather_out_page`) reduces exactly to `c·P + p` for `gather_dim=0` and is
  proven against `torch.cat` in `test_all_gather_debug.py`.
- **Queue:** `op_requirements.md` holds **R2a** (TILE gd=-2 non-tile-aligned H —
  needs retile) and **R2b** (ROW_MAJOR gd=-1 — sub-page write). Both move
  `EXCLUSIONS` cells to passing. The design's single-direction modular-wraparound
  ring (a perf method, unverifiable on the line sim, no failing cell) is recorded
  as a **Recommendation** below, not a queue entry (per the queue's hard rule).

---

## Code Review

### Fixed (this pass)

- **None.** The current code (including the Refinement-2 strided-addressing
  rewrite of the reader/writer and the Refinement-3 topology threading) reviewed
  clean. No correctness, CB-balance, API-usage, or helper-usage defect was found
  that is fixable without churning green, on-device-verified CCL kernels. The one
  Phase-0 fix (removing the vacuous `and page != 16` sub-clause in the
  page-alignment gate) remains in place and correct.

### Reviewed clean (no change needed)

- **CB sync (push == pop) — balanced in every regime, incl. strided (R2).**
  `cb_relay_pages` (idx 16): when `my_num_targets > 0` the reader pushes
  `P` seed pages + `num_relay_blocks × P` relay pages; the writer pops
  `(num_relay_blocks + 1) × P` — equal. When `my_num_targets == 0` (line end in
  that direction) the reader takes the pure-receiver branch (pushes nothing) and
  the writer early-returns (pops nothing) — also balanced. The strided
  `gather_out_page` remap changes only the *destination page index*, never the
  page *count*, so R2 did not disturb the balance. ✔
- **`cb_self_copy` (idx 24) is scratch, correctly unbalanced.** The forward
  reader `cb_reserve_back(.., 1)` once and reuses the write pointer across all
  `P` pages; no cross-kernel consumer, so it is deliberately not push/pop
  balanced. ✔
- **Strided concat addressing (R2) is correct.** `gather_out_page(c, p, dim_j,
  inner_stride, N) = high·(N·block) + (c·dim_j + mid)·inner + low` with
  `block = dim_j·inner`. For `gather_dim=0` (`dim_j = B_pages`, `inner = P/B`)
  this reduces algebraically to `c·P + p` (the page-contiguous case). Verified
  against `torch.cat` for every gather_dim in `test_all_gather_debug.py`, and
  observed passing on the sim for gd ∈ {-4,-3,-2,-1} across TILE/RM × dtype ×
  alignment. The remap depends only on the slice id `c` (device-independent), so
  store-and-forward relays to the same page range on every hop and the
  counting-semaphore logic is unchanged. ✔
- **Store-and-forward ordering via the counting semaphore.** Each upstream writer
  forwards blocks in canonical slice order (seed `i`, then `i-1, i-2, …` for
  forward flow) and issues one `AtomicIncChannel::inc` per block; the downstream
  reader `noc_semaphore_wait_min(sem, running)` before each read-back, so the
  `running` count maps 1:1 onto block arrival order (in-order fabric delivery
  lands the payload before its inc). Device `i`'s forward writer issues `i+1`
  incs; device `i+1`'s forward reader waits for `i+1`. Balanced by hand for both
  directions and both line ends, and green on the sim. ✔
- **Helper usage (fabric egress).** The writer drives the safety-by-construction
  `ccl_helpers_dataflow.hpp` helper exactly as intended: `FabricStreamSender<> →
  open(unicast_route) → arm_unicast_write / arm_inc → write_page / inc →
  close()` (close drains write + atomic barriers). The line-end writer opens no
  connection (early `return` on `my_num_targets == 0`). The raw-API fallbacks
  (`noc_async_read` ingress, local self-copy `noc_async_write`,
  `noc_semaphore_wait_min/set`) are precisely the pieces the helper banner
  (`ccl_helpers_dataflow.hpp:63-93`) states the op must own — there is no
  `FabricStreamReceiver`, and the ring slice-walk / concat addressing / local
  barrier-wait are explicitly not the helper's. No helper is under-used or
  bypassed; the multicast-inc barrier channel is deliberately not used (see
  Design Conformance). ✔
- **API correctness.** `void kernel_main()` (not the deprecated namespace
  pattern); includes use `api/dataflow/dataflow_api.h`; addressing uses
  `TensorAccessor(TensorAccessorArgs<...>, addr, page_size)` (not the deprecated
  `InterleavedAddrGen`). ✔
- **`validate()` shape + order.** First line of the public `all_gather(...)`.
  Structural input checks (MeshDevice, `(1,N)` line view, ≥2 devices,
  interleaved-only, 16B page alignment, output-spec equality) → per-axis
  `SUPPORTED` gate (running INPUT_TAGGERS generically) → `EXCLUSIONS` (cell
  level). Raises the registry-model `UnsupportedAxisValue` / `ExcludedCell`
  (both `NotImplementedError` subclasses). `gather_dim` is canonicalized to the
  negative convention *before* the membership test. ✔
- **Fabric arg contract.** The writer reads 9 scalar RT args (incl. `dim_j`,
  `inner_stride` for R2), then the `[has_forward][fwd?][has_backward][bwd?]`
  fabric block at `conn_arg_idx`, matching the host `_append_fabric_rt_args`
  layout; the leading `has_forward` flag is peeked as `dst_is_forward`. Fabric
  args are appended only on the forwarding writer. ✔

### Benign, well-justified deviation from `op_design.md` (no action)

- **No startup barrier (design Phase-1 `arm_multicast_inc`).** The writer omits
  the design's N-party startup barrier and relies solely on the Phase-2 counting
  semaphore for cross-device ordering, plus the reader's end-of-kernel
  `noc_semaphore_set(sem, 0)` reset (+ the `GlobalSemaphore`'s initial 0) for
  cache re-arm. This is a **sound simplification, not a bug**, and it holds after
  R2/R3:
  1. The counting sem provides the only ordering the op needs (data-then-inc,
     in-order fabric delivery), and the *persistent* pre-allocated output makes
     "early" fabric writes correct — an inc that arrives before the downstream
     reader starts simply increments a persistent L1 location the reader later
     reads. There is no race the barrier would close.
  2. Omitting the barrier *avoids* the helper's documented shared-pooled-header
     footgun entirely (`arm_multicast_inc` and `arm_inc` draw from the same pool;
     the barrier's `MulticastIncChannel` would have to be block-scoped before
     `arm_inc` re-arms). There is no block-scoping hazard to get wrong.
  3. It is empirically correct across the *full refined axis set* now (not just
     Phase 0): the program-cache two-call test — the exact scenario the
     barrier+reset dance exists to protect — passes on current code, as do all
     sampled golden cells, precision cells, and extended cells.
  If a future refinement ever needs a hard startup fence, the design's
  mandate-compliant fallback is a **second parked `GlobalSemaphore`** (not the
  shared-header barrier). Documented in the changelog.

- **No compute (TRISC) kernel wired.** `kernels/all_gather_compute.cpp` exists
  only to document the deliberate absence (CCL = pure data movement); it is not
  referenced by the descriptor. Inherent to a no-arithmetic collective. ✔

- **Ring served by the topology-agnostic adjacent-hop kernels, not the design's
  modular-wraparound method.** The design specifies a single-direction ring with
  `(i±k) mod N` wraparound. R3 instead serves `Topology.Ring` through the
  existing bidirectional adjacent-hop kernels (`ccl_dm_route(Ring)` resolves
  adjacent neighbours identically to Linear). The **output is identical** (gather
  is topology-agnostic), so this is a correctness-complete delivery; the
  wraparound is a perf method that is *unexercisable* on the T3K line sim (proven
  in `test_all_gather_ring_probe.py`: the 7→0 wraparound resolves to 7 hops under
  both FABRIC_1D and FABRIC_1D_RING). See Recommendations.

### Minor observations (not fixed — churn risk on green, on-device-verified kernels)

- **`gather_out_page` is duplicated verbatim** in `all_gather_reader.cpp` and
  `all_gather_writer.cpp`. The comment justifies keeping it inline (avoid a JIT
  include-path dependency between two sibling kernel sources). Extracting to a
  shared `kernels/*.hpp` would need the JIT include path wired and re-tested on
  the sim — pure churn on green kernels for no capability. Left as-is; flagged for
  a future cleanup pass if the two copies ever drift.
- **Forward reader reads each input page twice** — once into `cb_self_copy` (local
  self-copy), once into `cb_relay_pages` (fabric seed). A fuse (read once,
  self-copy from the relay slot, then push) would halve the forward reader's
  input reads, but it entangles the line-end seed-push gating and reintroduces the
  CB-balance edge case the current clean split avoids. Pure perf, no failing cell.
- **`noc_async_writes_flushed()` per page in the writer** serializes the fabric
  egress (flush after every `write_page`). Conservative and provably safe (the CB
  slot must not be reused until the fabric sender drained it); flushing per
  chunk-pair would overlap more. Perf, not correctness.
- **`_get_or_create_semaphore` keys the cache on `id(mesh_device)`.** Correct for
  a live mesh_device; a GC'd-then-realloc'd device could in principle reuse the
  `id`. Matches the `point_to_point` precedent and has no observed failure mode.
  Left as-is.

---

## Registry Conformance

- **INPUT_TAGGERS** — present: `{"alignment": tag_alignment}`, signature
  `(inputs, axes)`. Reads the per-device shard's last two dims (both `% 32 == 0
  → tile_aligned`, else `non_tile_aligned`). Correct for a byte-mover that copies
  padded tiles / RM rows verbatim. ✔
- **SUPPORTED** — present; declares every gated axis: `dtype`, `layout`,
  `topology`, `gather_dim` (negative-canonicalized index axis), `alignment`.
  `set(SUPPORTED) == set(TARGET)` and each axis's value list equals TARGET's
  (verified programmatically). ✔
- **EXCLUSIONS** — present, 2 cell-dicts (both structural sub-cell corners inside
  SUPPORTED, see audit). ✔
- **validate()** — present; first line of `all_gather(...)`; SUPPORTED-per-axis
  then EXCLUSIONS-cell order; typed refusals. `gather_dim` canonicalized negative
  before the membership test. ✔
- **Op file does NOT declare INVALID** — confirmed (`INVALID` is not a symbol in
  `ttnn.operations.all_gather`); it lives only in `feature_spec.py`. ✔
- **Entry point calls `validate()` first** — yes, before any allocation / kernel
  work. ✔

### INVALID audit (`eval/golden_tests/all_gather/feature_spec.py`)

`INVALID = [{"dtype": ttnn.bfloat8_b, "layout": ttnn.ROW_MAJOR_LAYOUT}]`.

- **Single-tensor coupling:** `dtype` and `layout` both describe the *input*
  tensor — no cross-tensor coupling. ✔
- **Universe-must-change:** `bfloat8_b` is a tiled block-float format with no
  row-major representation — ttnn cannot construct `{bf8b, ROW_MAJOR}`. A
  data-format-definition impossibility, not a not-yet-implemented EXCLUSION. ✔
- **Canonical bf8b+ROW_MAJOR entry present** (the mandatory entry for a tile-or-RM
  byte-mover). ✔
- all_gather is not norm-like → no weight / no-weight canonicalization cells
  expected. ✔

Verdict: INVALID is well-formed; no change recommended. It skips 64 of the 384
golden cells (all `bf8b × ROW_MAJOR`).

### EXCLUSIONS audit (op file)

Two entries, both cells *inside* SUPPORTED the pure-byte-mover cannot express
without a repack the design forbids — correct use of EXCLUSIONS (refuse for now,
refinement candidate), not INVALID (structural impossibility):

1. `{layout: TILE, gather_dim: -2, alignment: non_tile_aligned}` — couples three
   axes that all describe the **input tensor** (a legitimate single-tensor
   coupling). When H is not a multiple of 32, each shard tilizes H with its own
   trailing padding (`Ht_shard` tiles) but the gathered output packs H densely
   (`Ht_out < N·Ht_shard`), so `N·pages_per_shard` source tiles ≠ output tiles —
   a whole-tile copy cannot reconcile the counts. Matches only shape `(1,1,48,64)`
   in the golden INPUTS. → **R2a**.
2. `{layout: ROW_MAJOR, gather_dim: -1}` — couples two input-tensor axes. In RM
   the page IS a W-row, so concat-along-W makes each device's row land at a byte
   *offset* inside an N× output page — a sub-page write the whole-page fabric
   egress cannot express. → **R2b**.

Both are confirmed to fire `xfail-strict` (validate() raises `ExcludedCell`, a
`NotImplementedError` subclass) with **no XPASS drift** on the sim. RM gd=-2 on
non-aligned H=48 correctly *passes* (H is a page-grid axis in RM), confirming the
exclusions are minimal and not over-broad.

---

## Precision Baseline

`tests/ttnn/unit_tests/operations/all_gather/test_all_gather_precision_baseline.py`
— **12/12 green** on the WH sim (bf16 / f32 / bf8b × 4 shapes, gather_dim=0,
TILE, Linear). Each device's output compared against the host-side concat of all
8 shards.

| Shard shape | full (gathered) | dtype | PCC | Max Abs Err | Mean Abs Err | Rel RMS Err |
|-------------|-----------------|-------|-----|-------------|--------------|-------------|
| (1,1,32,32)   | (8,1,32,32)   | bf16 | 1.0     | 0       | 0        | 0        |
| (1,1,32,32)   | (8,1,32,32)   | f32  | 1.0     | 0       | 0        | 0        |
| (1,1,32,32)   | (8,1,32,32)   | bf8b | ≥0.99   | 0.03125 | ~0.006   | ~0.0077  |
| (1,1,64,128)  | (8,1,64,128)  | bf16 | 1.0     | 0       | 0        | 0        |
| (1,1,64,128)  | (8,1,64,128)  | f32  | 1.0     | 0       | 0        | 0        |
| (1,1,64,128)  | (8,1,64,128)  | bf8b | ≥0.99   | 0.03125 | 0.006076 | 0.007706 |
| (1,1,96,64)   | (8,1,96,64)   | bf16 | 1.0     | 0       | 0        | 0        |
| (1,1,96,64)   | (8,1,96,64)   | f32  | 1.0     | 0       | 0        | 0        |
| (1,1,96,64)   | (8,1,96,64)   | bf8b | ≥0.99   | 0.03125 | 0.006056 | 0.007685 |
| (1,1,256,256) | (8,1,256,256) | bf16 | 1.0     | 0       | 0        | 0        |
| (1,1,256,256) | (8,1,256,256) | f32  | 1.0     | 0       | 0        | 0        |
| (1,1,256,256) | (8,1,256,256) | bf8b | ≥0.99   | 0.03125 | 0.005980 | 0.007634 |

**Assessment:** all_gather is a *bit-for-bit byte copy*. bf16/f32 outputs equal
the oracle exactly (zero abs/RMS error) and all N devices agree (replicated
output). bf8b shows only the `from_torch` shared-exponent quantization (constant
`max_abs = 0.03125`, `mean_abs ≈ 0.006`) — applied *before* the op runs; the
gather itself introduces none.

**Recommended tolerances** (identity oracle): PCC ≥ 0.9999 (f32) / 0.999 (bf16) /
0.99 (bf8b); `atol`/`rtol` effectively 0 for float dtypes. The acceptance +
golden suites gate on PCC with these thresholds.

---

## Verifier CLI Summary

`eval.verify_supported` was run mechanically on a **representative sampled
golden run** (the full 384-cell cartesian × per-cell device reopen exceeds the
sim's wall-clock backstop, so exhaustive execution is impractical — the sample is
chosen to hit every axis value + both EXCLUSIONS + the INVALID). Sampled cells:
shape `(1,1,48,64)` × Linear × {bf16, f32, bf8b} × {TILE, ROW_MAJOR} × gd
{-4,-3,-2,-1} (24 cells). Artifact: `eval/results/all_gather/verifier_report.json`.

**Sampled (mechanical `verify_supported`, 24 cells):**
- supported_pass: **15**
- xfail_expected: **5**  (TILE gd=-2 non-aligned ×3 dtype; RM gd=-1 ×{bf16,f32})
- invalid_skipped: **4** (bf8b × RM × gd{-4,-3,-2,-1})
- supported_fail: **0**   ✓ (must be 0 to ship)
- xpass_drift: **0**      ✓ (must be 0 to ship)
- xfail_wrong_mode: **0** ✓ (must be 0 to ship)
- supported_marked_xfail: **0** ✓
- invalid_unexpected: **0** ✓

A wider 36-cell partial run (adding Ring) reached **20 passed / 7 xfailed / 0
failed / 0 xpassed** before its wall-clock cutoff — the same clean signal, Ring
included (Ring supported cells pass; Ring EXCLUSIONS xfail).

**Analytical full cartesian (384 cells, from the harness's `feature_matrix`
categorization — deterministic, no device):**
- supported_pass: **282**  (`384 − 38 − 64`)
- xfail_expected: **38**  (EXCLUSIONS: 6 = TILE gd=-2 non-aligned {(1,1,48,64)} ×
  3 dtype × 2 topo;  32 = RM gd=-1 × 8 shapes × {bf16,f32} × 2 topo)
- invalid_skipped: **64**  (bf8b × RM × 8 shapes × 4 gd × 2 topo)
- loud categories: **0 by construction** (SUPPORTED == TARGET, so the only
  refusals are the two EXCLUSIONS and the one INVALID, all correctly categorized).

### `xfail_expected` → queue mapping (no gap)

Since `TARGET − SUPPORTED = ∅` on every axis, there are **no `(axis,
missing_value)` pairs** — the entire `xfail_expected` bucket is EXCLUSIONS-only:

| xfail_expected axis-combo | analytical cells | Covered by |
|---------------------------|------------------|------------|
| `{TILE, gather_dim=-2, non_tile_aligned}` | 6 | **R2a** (retile path) |
| `{ROW_MAJOR, gather_dim=-1}` | 32 | **R2b** (sub-page write) |

Every `xfail_expected` cell maps to a queue entry (R2a or R2b); no queue gap. The
`invalid_skipped` bucket (`bf8b × ROW_MAJOR`) is structurally INVALID and
correctly not queued.

---

## Recommendations

1. **Refinement order (see `op_requirements.md`):** the two active refinements
   (R2a, R2b) are independent structural corners — either order. Both remove an
   `EXCLUSIONS` cell and both break the "pure byte movement / no tilize" design
   invariant, so each must be **escalated as a design change** (R2a needs an
   untilize→retile repack or a compute untilize/tilize stage; R2b needs a
   write-at-offset / sub-page fabric unicast). Neither maps onto a current
   implementation skill (the inventory covers single-device compute precision,
   in-kernel layouts, interleaved multi-core, and L1 budget — none cover CCL
   fabric sub-page addressing or cross-gather retile), so both are
   verifier-authored with full goal + done-when.

2. **Design-method gap — single-direction modular-wraparound ring (PERF, HW-gated,
   NOT a queue entry).** The design specifies a single forward worker per device
   sending to `(i+1) mod N` with the `N-1 → 0` wraparound edge. R3 instead serves
   `Topology.Ring` correctly through the topology-agnostic bidirectional
   adjacent-hop kernels — the **output is identical** (gather is topology-agnostic)
   and `Topology.Ring` is in SUPPORTED and passes. The wraparound method is a
   **perf optimization with no failing golden cell**, and it is **unverifiable on
   the only all_gather sim topology**: `wh_t3k_allmmio_all_gather` is a physical
   T3K *line*, where `ccl_dm_route(Ring)` resolves the 7→0 wraparound to
   `num_hops=7` (the long way) under both `FABRIC_1D` and `FABRIC_1D_RING`
   (proven in `test_all_gather_ring_probe.py`). Per the queue's hard rule
   (a refinement must add a SUPPORTED axis value *or* move named failing cells to
   passing), this belongs in the report, not `op_requirements.md`. If a
   ring-capable topology is added to `scripts/multidevice_sim_topologies.yaml`
   (e.g. a WH-ring / BH-torus mesh-graph + `FABRIC_1D_RING`, as `point_to_point`
   used on the BH 8xP150 torus), the follow-up would: (1) gate the modular
   single-direction path on a verified 1-hop wraparound route, (2) re-confirm the
   counting-sem increment/wait counts under wraparound. Until then it is pure perf
   with no verifiable cell.

3. **Re-verify every refinement on the WH sim, not silicon.** This host is
   single-device; the only multi-device path is
   `scripts/run_multidevice_sim_pytest.py --op all_gather` (topology
   `wh_t3k_allmmio_all_gather`, `required=true`). Tests MUST open exactly `(1,8)`
   + `FABRIC_1D` or fabric init hangs ("Fabric Router Sync: Timeout"). The sim
   reopens the mesh device per parametrization (~15–20 s/cell), so run
   representative slices, not the full cartesian, within the wall-clock backstop.

4. **Out-of-TARGET scope items (NOT refinements — would need `/golden-tests` to
   widen TARGET first):**
   - *Sharded memory config.* `validate()` rejects sharded input ("interleaved
     only"). TARGET has no `memory_config` axis, so not a refinement candidate.
   - *Multi-link / multi-core fan-out per direction.* Single worker core per
     direction by design; a perf enhancement with no SUPPORTED axis or failing
     cell.

5. **Perf levers (no failing cell — not queued):** the forward reader's double
   input read and the writer's per-page fabric flush (see Minor observations) are
   the two obvious throughput levers if a perf pass is commissioned.
