# Verification Report: all_gather

**Op class:** multi-device CCL (collective communication) — pure cross-chip data
movement, no arithmetic. Self-contained Python `generic_op` + `MeshProgramDescriptor`
with newly-authored bidirectional store-and-forward ring dataflow kernels.

**Verification runner:** `scripts/run_multidevice_sim_pytest.py --op all_gather`
(topology `wh_t3k_allmmio_all_gather`, mesh **(1, 8)**, `FABRIC_1D`, WH sim).
`run_safe_pytest.sh` is the WRONG runner here — in sim mode it forces slow dispatch
and has no multichip/hang awareness; the fabric data plane needs fast dispatch.

---

## Code Review

Reviewed `all_gather.py`, `all_gather_program_descriptor.py`, both kernels
(`all_gather_reader.cpp`, `all_gather_writer.cpp`), and the CCL helper header
`ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp`.

### Fixed

- **`validate()` dead-code branch (all_gather.py).** The page-alignment guard read
  `if page % 16 != 0 and page != 16:`. The `and page != 16` clause is dead — 16 is a
  multiple of 16, so `page % 16 != 0` is already `False` for `page == 16`. Simplified
  to `if page % 16 != 0:`. Behaviour is identical for every input; the change only
  removes the misleading dead conjunct.

- **Test-suite import break (shared infra, `tests/ttnn/utils_for_testing.py`).** The
  checked-out file (from FP8-enablement commit `079872566e`) references
  `ttnn.fp8_e4m3` at import time, but the built `ttnn` binary in this environment
  predates FP8 (`ttnn` exposes only `bfloat8_b`). This raised
  `AttributeError: module 'ttnn' has no attribute 'fp8_e4m3'` and blocked collection
  of **every** test importing `assert_with_pcc` — including all of all_gather's.
  Guarded the entry: `if hasattr(ttnn, "fp8_e4m3"): tt_dtype_to_torch_dtype[...] = ...`.
  Forward/backward compatible; not an all_gather defect, but it had to be unblocked to
  run any test.

### Reviewed and confirmed correct (no change needed)

- **CB sync (push == wait) balances in every ring role.** Traced `cb_relay`
  producer/consumer counts for all four (core, end-device) cases:
  - fwd core, `d < N-1`: reader pushes `(d+1)·pages_per_shard` (own + `d` relays);
    writer pops `num_consume_fwd = d+1` slices. ✔
  - fwd core, `d = N-1` (no fwd neighbour): reader pushes `pages_per_shard` (own
    only); writer pops `1` slice (own local write). ✔
  - bwd core, `d > 0`: reader pushes `(N-d)·pages_per_shard`; writer pops
    `num_consume_bwd = N-d`. ✔
  - bwd core, `d = 0` (no bwd neighbour): reader pushes 0, writer pops 0. ✔
- **Helper usage matches the CCL safety-by-construction contract.** The writer uses
  `FabricStreamSender → open(unicast_route(1)) → arm_multicast_inc/multicast_inc →
  arm_unicast_write/write_page → arm_inc/inc → close()`. The barrier
  `MulticastIncChannel` is block-scoped **before** `arm_inc` (the documented
  pooled-header footgun). The op-owned raw calls — `noc_semaphore_wait_min`/`set`
  (WAITING half + cache-reuse re-arm), `noc_async_read` relay ingress, the local
  own-slot `noc_async_write` — are exactly the ones the helper banner
  (`ccl_helpers_dataflow.hpp:69-93`) says the op composes, not the helper. No raw
  `noc_async_write_multicast` + hand-rolled semaphore handshake that should be a
  `SenderPipe`/`ReceiverPipe`; the fabric egress already goes through the helper.
- **`close()` drains.** The design pseudocode says "drain() + close()"; `close()`
  drains (write + atomic barriers) then closes, so the explicit `drain()` is
  correctly omitted. The trailing `noc_async_write_barrier()` acks the local
  own-slot writes.
- **Registry / API hygiene:** `TensorAccessor` (not deprecated `InterleavedAddrGen`);
  `void kernel_main()`; includes use `api/dataflow/dataflow_api.h`; the three
  `GlobalSemaphore`s are created once per mesh (cached by `id(mesh_device)`) + one
  `synchronize_device`, and parked on `mesh_program_descriptor.semaphores`.

### Design conformance

Checked against `op_design.md` on the binding dimensions — all match:
- **Algorithm:** bidirectional store-and-forward ring, single-hop relay (no
  full-matrix materialisation); slice-walk `j = ring_index ∓ r`. ✔
- **Data pipeline topology:** two worker cores per device (`core_fwd = (0,0)`,
  `core_bwd = (0,1)`), each reader (NCRISC) + writer (BRISC), `cb_relay` between. ✔
- **Coordination:** three single-owner semaphores (barrier + fwd/bwd counting),
  data-before-inc ordering on the shared 1-hop route, cache-reuse re-arm. ✔

### Advisory (not a bug, not a refinement — no failing cell)

- **Per-tile UNICAST egress instead of ≤4-tile SCATTER coalescing.** The writer
  self-documents this as an "advisory deviation": it issues one `write_page` per tile
  with one counting `inc` per whole slice, rather than packing up to
  `num_tiles_per_packet` tiles into a `write_scatter` packet
  (`arm_scatter_write`/`write_scatter`, which the helper and design both expose).
  Output is identical (counting is per-slice regardless) and it sidesteps the
  scatter partial-packet / CB-contiguity edge cases. This is a **throughput**
  optimisation with no failing golden cell to point at, so it is a recommendation,
  not a refinement. A future perf pass can reintroduce scatter coalescing.
- **Per-tile `noc_async_read_barrier()` in the reader's `stream_tiles`** and per-tile
  `noc_async_writes_flushed()` in the writer. Correct (the CB is a small streaming
  triple-buffer), but batching the read barrier / flush across the small CB window
  would cut NoC round-trips. Throughput-only; no correctness impact.

---

## Registry Conformance

- **Confirmed present and correctly wired in `all_gather.py`:** `INPUT_TAGGERS`
  (`{"alignment": tag_alignment}`, two-arg `(inputs, axes)` signature), `SUPPORTED`
  (dtype / layout / topology / gather_dim / alignment — every axis the kernel gates
  on, plus the tagger key), `EXCLUSIONS` (`[]`), and `validate()` (checks SUPPORTED
  per-axis, then EXCLUSIONS cell-level; both raise the typed refusals
  `UnsupportedAxisValue` / `ExcludedCell` from `ttnn.operations._op_contract`).
  `validate()` is the first line of the public `all_gather()` entry point.
- **`gather_dim` canonicalization** to the negative convention happens **before** the
  SUPPORTED membership check, so the positive alias `gather_dim=0 ≡ -4` (rank-4
  shards) is accepted, matching the feature_spec TARGET convention.
- **Confirmed the op file does NOT declare `INVALID`.** `grep INVALID` across the op
  package finds it only in `op_design.md`'s "Structural impossibilities" note (a
  pointer to feature_spec.py) — never as a symbol in the code. Correct for the
  registry model.
- **No auto-fixes to SUPPORTED were needed** — `xpass_drift = 0`, so there is no
  under-claim to promote.

### INVALID audit (`eval/golden_tests/all_gather/feature_spec.py`)

`INVALID = [{"dtype": ttnn.bfloat8_b, "layout": ttnn.ROW_MAJOR_LAYOUT}]` — well-formed
against all three sanity rules:
- **Single-tensor coupling:** dtype and layout both describe the (sole) input tensor. ✔
- **Universe-must-change:** `bfloat8_b` is a tiled block-float format with no
  row-major representation — ttnn cannot construct `{bfloat8_b, ROW_MAJOR}`; the data
  format definition itself would have to change. This is INVALID, not EXCLUSIONS. ✔
- **Canonical bf8b + ROW_MAJOR entry present.** ✔
- No cross-tensor-axis coupling; all_gather has no weight axes, so the norm-like
  no-weight canonicalization rule does not apply.
- Verified against the golden run: the 64 `{bf8b, ROW_MAJOR}` cells are `skipped`
  (`invalid_skipped = 64`), never dispatched. ✔

---

## Precision Baseline

`test_all_gather_precision_baseline.py`, 4 shard shapes × {bf16, f32} on the proven
primary case (gather_dim=0, TILE, Linear, mesh (1, 8)). all_gather is pure byte
movement (identity gather), so the transfer adds **zero** error — the only
quantization is the `from_torch` cast that happens before the op runs, identically on
both the input and the oracle.

| Shard shape → full shape | dtype | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err |
|--------------------------|-------|-----|-------------|--------------|------------------|
| (1,1,32,32) → (8,1,32,32)     | bf16 | 1.0 | 0.0 | 0.0 | 0.0 |
| (1,1,32,32) → (8,1,32,32)     | f32  | 1.0 | 0.0 | 0.0 | 0.0 |
| (1,1,64,128) → (8,1,64,128)   | bf16 | 1.0 | 0.0 | 0.0 | 0.0 |
| (1,1,64,128) → (8,1,64,128)   | f32  | 1.0 | 0.0 | 0.0 | 0.0 |
| (1,1,96,64) → (8,1,96,64)     | bf16 | 1.0 | 0.0 | 0.0 | 0.0 |
| (1,1,96,64) → (8,1,96,64)     | f32  | 1.0 | 0.0 | 0.0 | 0.0 |
| (1,1,256,256) → (8,1,256,256) | bf16 | 1.0 | 0.0 | 0.0 | 0.0 |
| (1,1,256,256) → (8,1,256,256) | f32  | 1.0 | 0.0 | 0.0 | 0.0 |

**Assessment:** bit-for-bit identity gather across all shapes and both dtypes, and all
8 devices agree bit-for-bit (replicated output). No ULP drift because there is no
arithmetic — the byte copy preserves the exact `from_torch` bit pattern.

**Recommended tolerances:** PCC ≥ 0.9999 (f32) / 0.999 (bf16); atol = 0, rtol = 0.
The op should be held to bit-exactness — any nonzero error is a real bug (a mis-routed
slice, a stride error, or a lost tile), not precision noise.

---

## Testing performed

| Suite | Runner | Result |
|-------|--------|--------|
| Acceptance (`test_all_gather.py`) — 4 shapes × {bf16,f32} + program-cache | multidevice sim | **9/9 PASS** (~154 s) |
| Extended (`test_all_gather_extended.py`) — preallocated-output path + validate() rejections | multidevice sim | **2/2 PASS** |
| Precision baseline (`test_all_gather_precision_baseline.py`) — 4 shapes × {bf16,f32} | multidevice sim | **8/8 PASS**, bit-exact |
| Golden suite (`eval/golden_tests/all_gather/test_golden.py`) — 384 cells | multidevice sim (5 dtype/layout chunks) | **16 pass / 304 xfail / 64 skip** |

The golden suite was run in 5 `-k` chunks (bf8b; bf16-RM; bf16-not-RM; f32-RM;
f32-not-RM), each finishing inside the wall-clock backstop — a full single-process run
exceeds it because the CCL golden `mesh_device` fixture re-initialises the 8-device
fabric per cell (the standard CCL-golden pattern; point_to_point's golden does the
same — not an all_gather defect). The 5 junits were merged and fed to the verifier CLI.

---

## Verifier CLI Summary

`python3 -m eval.verify_supported /tmp/all_gather_verify ttnn.operations.all_gather`
(artifact copied to `verifier_report.json` in this directory):

| Category | Count |
|----------|-------|
| supported_pass | **16** |
| xfail_expected | **304** |
| invalid_skipped | **64** |
| supported_fail | **0** (must be 0 to ship ✔) |
| xpass_drift | **0** (must be 0 to ship ✔) |
| xfail_wrong_mode | **0** (must be 0 to ship ✔) |
| supported_marked_xfail | 0 |
| total | 384 |

Clean golden run — all three loud categories at 0. SUPPORTED describes reality.

### `xfail_expected` breakdown (the refinement gap = TARGET − SUPPORTED)

Iterating `by_category.xfail_expected` and counting the out-of-SUPPORTED
`(axis, missing_value)` pairs across the 304 cells:

| (axis, missing value) | # xfail cells | Disposition |
|-----------------------|---------------|-------------|
| `dtype = bfloat8_b`   | 64  | **Refinement 1** (bundled with layout) |
| `layout = ROW_MAJOR`  | 128 | **Refinement 1** |
| `gather_dim = -3`     | 80  | **Refinement 2** |
| `gather_dim = -2`     | 80  | **Refinement 2** |
| `gather_dim = -1`     | 80  | **Refinement 2** |
| `topology = Ring`     | 160 | **Refinement 3** (verification infra-blocked — see below) |

(Cells counted per offending axis; a cell with two out-of-SUPPORTED axes — e.g.
`ROW_MAJOR × Ring` — is counted under both. `bf8b × ROW_MAJOR` is INVALID/skipped, not
xfail.) Every `(axis, missing_value)` pair from TARGET − SUPPORTED maps to a refinement
in `op_requirements.md`; none is orphaned.

---

## Recommendations

- **Ordering (see `op_requirements.md`):** format axes (bf8b + ROW_MAJOR) first — the
  quick, low-risk wins that reuse the existing contiguous-slice walk at gather_dim=0;
  then the non-contiguous concat addressing (gather_dim −3/−2/−1), which is the one
  algorithmically substantial change; then Ring.
- **Ring is verification-blocked on the current sim matrix.** The only all_gather
  topology in `scripts/multidevice_sim_topologies.yaml` is `FABRIC_1D` (a line). A
  Ring cell run against a line fabric cannot exercise wraparound routing (there is no
  ethernet link between device 0 and N-1), so the Ring refinement cannot be *proven*
  until a `FABRIC_RING` (or ring mesh-graph) topology with `applies_to_ops: [all_gather]`
  is added to the matrix. The refinement is still legitimate (Ring is a real TARGET
  value, not INVALID); it just can't be ticked green on today's infra. Flagged in the
  refinement's verifier notes.
- **Throughput, not correctness (no queue entry):** reintroduce ≤4-tile SCATTER
  coalescing in the writer (the documented advisory deviation) and batch the reader's
  per-tile read barrier. Both are perf-only with no failing cell — recorded here, not
  in the refinement queue.
- **L1 / memory pressure:** none. The relay CB is a fixed 8-page streaming buffer
  (`_CB_NUM_PAGES · aligned_page_size`), independent of `pages_per_shard`, so there is
  no shape-driven L1 growth and no OOM risk across the INPUTS set. No
  `/memory-budget-metal` concern.
- **No `/interleaved-parallel` refinement.** all_gather is a CCL ring with cross-device
  data dependencies (mcast barrier, counting semaphores, store-and-forward relay); the
  two-core-per-device layout is intrinsic to the algorithm, not an embarrassingly-
  parallel work split. That skill explicitly excludes cross-core-dependency algorithms.
