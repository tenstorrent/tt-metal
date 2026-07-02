# Verification Report: all_gather

CCL multi-device op (bidirectional store-and-forward ring, `ttnn.generic_op` +
`ttnn.MeshProgramDescriptor`). Verified on the deterministic WH T3K all-MMIO sim,
mesh `(1, 8)` + `FABRIC_1D`, via `scripts/run_multidevice_sim_pytest.py --op all_gather`
(the correct runner — `run_safe_pytest.sh` forces slow dispatch and has no multichip
awareness on sim).

---

## Code Review

### Fixed — WAR hazard on the relay CB slot (correctness, real-HW only)

**What:** The writer fabric-forwards each page with `writer.write_page(src, …)` where
`src` is the `cb_relay_pages` read pointer, then `cb_pop_front`s the slot. The fabric
payload send is **`NON_BLOCKING`** — `edm_fabric_utils.hpp:send_chunk_from_address` issues
`noc_async_write(src → EDM buffer)` and, in `NON_BLOCKING` mode, does **no** flush/barrier.
So after `write_page` returns, the async read of `src` is still in flight. The moment the
writer pops the slot, the concurrent reader (NCRISC) refills it via `noc_async_read`, which
**races the still-in-flight fabric read of the same L1 region** (write-after-read hazard →
corruption on real hardware).

The **forward-seed self-copy path already guarded this** (`noc_async_write_barrier()` before
pop, commented "local self-copy done before slot reuse") — but the **backward-seed path and
both relay loops did not**. That inconsistency is the tell: the author protected the
self-copy's `src` but not the fabric write's `src` (which is *also* an async read of the same
slot). The sim executes NoC writes synchronously, so it cannot expose the race — every cell is
bit-exact (PCC=1.0) on the sim regardless.

**Fix:** Added `noc_async_write_barrier()` before every `cb_pop_front` that follows a fabric
`write_page` (seed loop made unconditional; relay loop gained the barrier). A plain
`noc_async_write_barrier()` is exactly what the helper's own `drain()` uses to flush fabric
payload writes (`ccl_helpers_dataflow.inl:162`), confirming it covers the fabric worker NoC.
Cannot introduce a hang (the barrier only waits for writes that will drain). Re-verified on
the relay-heavy `(1,1,256,256)` case (64 pages/shard, middle-device relay): still bit-exact,
no hang; `program_cache` + `output_tensor` acceptance tests still pass.

*Follow-up perf note (not a refinement — no failing cell):* the barrier is per-page, which
serializes the fabric pipeline. A later optimization could write from a persistent scratch
buffer (as `point_to_point` does) or batch the flush per block, removing the per-page stall
while keeping `src` safe. No correctness impact; noted for a future perf pass.

### Reviewed — no change needed

- **Helper usage (writer):** correct safety-by-construction progression —
  `FabricStreamSender → open(unicast_route(1)) → arm_multicast_inc` (block-scoped to the
  barrier phase) `→ arm_unicast_write + arm_inc → write_page/inc → close()`. `close()` drains,
  so the extra explicit `drain()` the design mentioned is unnecessary. Line-end workers gate
  the fabric egress behind `will_forward` and still run the local barrier wait+reset. ✓
- **Reader:** raw `noc_async_read` for the seed + relay read-back is the op-owned ingress the
  helper banner explicitly does *not* own ("there is no FabricStreamReceiver"). ✓
- **Includes / syntax:** `#include "api/dataflow/dataflow_api.h"` (not the bare form),
  `void kernel_main()`, `TensorAccessor` (not deprecated `InterleavedAddrGen`). ✓
- **CB push/pop balance:** verified per direction — forward core pushes `1 + (num_relay if
  will_forward)` and the writer pops the same; line-end workers never push relay pages their
  writer won't pop. Balanced. ✓
- **Two-semaphore design:** `barrier_sem` (multicast, each core reaches `ring_size-1`) and
  `counting_sem` (per-block flow control) are separate — correctly breaks the barrier↔counting
  race the design describes. Created once per mesh_device, one `synchronize_device`, cached,
  parked via `mpd.semaphores`. The cache-reuse resets (receiver resets `counting_sem` after
  its last wait; every writer resets `barrier_sem` after the barrier) are present and exercised
  by `test_all_gather_program_cache`. ✓
- **`page_size` vs `aligned_page_size` (minor, correct):** the kernels build the
  `TensorAccessor` with the raw `buffer_page_size()` (matching `point_to_point`) but use the
  host-passed `aligned_page_size` (= `output.buffer_aligned_page_size()`) for the transfer size
  and CB sizing, whereas p2p uses `accessor.get_aligned_page_size()`. These are provably equal
  here (input and output share dtype/layout/inner dims), so it is correct — just marginally more
  fragile than deriving it from the accessor. Left as-is (changing it risks a CB-size /
  transfer-size mismatch for zero benefit). Verified bit-exact on the non-tile-aligned
  `(1,1,48,64)` shape, where `page_size ≠ aligned_page_size`.

### Design conformance

Implementation matches `op_design.md` on all binding dimensions: algorithm (bidirectional
store-and-forward ring, no materialized score matrix — N/A here, pure DM), pipeline topology
(2 worker cores/device, reader=NCRISC + writer=BRISC), work distribution (one worker per
direction), and inter-core comms (multicast barrier + counting inc via the CCL helper).
Two documented, benign deviations: CB depth is a constant 4 pages (design said 2 × pages/packet
— deeper double-buffer, still constant-bounded L1); one counting `inc` per whole block (design
allowed `chunks_per_sync` sub-block granularity — functionally identical for all test shards).

---

## Registry Conformance

- **`INPUT_TAGGERS`** = `{"alignment": tag_alignment}`; `tag_alignment(inputs, axes)` has the
  correct 2-arg signature. ✓
- **`SUPPORTED`** = `dtype [bfloat16, float32]`, `layout [TILE, ROW_MAJOR]`,
  `topology [Linear]`, `gather_dim [-4]`, `alignment [tile_aligned, non_tile_aligned]` — covers
  every axis the op gates on. ✓
- **`EXCLUSIONS`** = `[]` (nothing refused inside the SUPPORTED rectangle). ✓
- **`validate()`** canonicalizes `gather_dim` to negative *before* the axis gate (so
  `gather_dim=0 ≡ -4` for the rank-4 shards is not rejected by literal membership), then checks
  SUPPORTED per-axis and EXCLUSIONS cell-level, raising `UnsupportedAxisValue` / `ExcludedCell`.
  Order correct; the public `all_gather()` calls `validate()` on its first line. ✓
- **Op file does NOT declare `INVALID`** — confirmed (`grep` clean). INVALID is sourced from
  `eval/golden_tests/all_gather/feature_spec.py`, as required by the registry model. ✓
- **No auto-fixes to SUPPORTED needed** — `xpass_drift = 0`, so SUPPORTED does not under-claim.

### INVALID audit (`eval/golden_tests/all_gather/feature_spec.py`)

`INVALID = [{"dtype": ttnn.bfloat8_b, "layout": ttnn.ROW_MAJOR_LAYOUT}]` — well-formed:
- **Single-tensor coupling:** both `dtype` and `layout` describe the *input* tensor. ✓
- **Universe-must-change:** bfloat8_b is a tiled block-float format with no row-major
  representation — ttnn cannot construct `{bf8b, ROW_MAJOR}`; the data-format *definition* would
  have to change. Correctly INVALID, not EXCLUSIONS. ✓
- **Canonical bf8b + ROW_MAJOR present.** ✓ Verified on-sim: the 8 bf8b+RM cells are `skipped`
  (never reach the op).
- all_gather is not norm-like (no weight axes), so no no-weight canonicalization applies. ✓

No cross-tensor-axis entries, nothing that encodes "kernel doesn't support this yet" (that
would be EXCLUSIONS). No changes recommended.

---

## Precision Baseline

all_gather is **pure byte movement** (identity gather, no arithmetic): the output is a
bit-for-bit copy of the host-side concat of all N shards. The only error possible is the dtype
quantization that already happened at `from_torch` *before* the op ran — the gather adds
nothing. Measured on-sim (metrics from the golden `metrics_plugin` + `test_all_gather_precision_baseline.py`):

| Shard shape | Dtype | Layout | Alignment | PCC | Max Abs Err | Mean Abs Err | Rel RMS Err |
|-------------|-------|--------|-----------|-----|-------------|--------------|-------------|
| (1,1,256,256) | bfloat16 | TILE | tile-aligned | 1.0 | 0.0 | 0.0 | 0.0 |
| (1,1,256,256) | bfloat16 | ROW_MAJOR | tile-aligned | 1.0 | 0.0 | 0.0 | 0.0 |
| (1,1,256,256) | float32 | TILE | tile-aligned | 1.0 | 0.0 | 0.0 | 0.0 |
| (1,1,256,256) | float32 | ROW_MAJOR | tile-aligned | 1.0 | 0.0 | 0.0 | 0.0 |
| (1,1,48,64) | bfloat16 | TILE | non-tile-aligned | ~1.0¹ | 0.0 | 0.0 | 0.0 |
| (1,1,48,64) | bfloat16 | ROW_MAJOR | non-tile-aligned | ~1.0¹ | 0.0 | 0.0 | 0.0 |
| (1,1,48,64) | float32 | TILE | non-tile-aligned | ~1.0¹ | 0.0 | 0.0 | 0.0 |
| (1,1,48,64) | float32 | ROW_MAJOR | non-tile-aligned | ~1.0¹ | 0.0 | 0.0 | 0.0 |

¹ `0.9999999999999998` — float rounding in the PCC computation itself; RMS and max-abs are
exactly 0.0 (bit-exact). All N=8 devices hold identical output.

**Assessment:** bit-exact identity gather for every supported dtype × layout × alignment.
Error is structurally 0 (byte copy). **Recommended tolerances:** PCC ≥ 0.999 (headroom over the
observed 1.0); the golden suite's `(0.999, 0.02)` band is comfortable.

---

## Verifier CLI Summary

`verifier_report.json` → `generated/all_gather_verify/verifier_report.json`.

**Coverage caveat (honest scope):** the full golden cartesian is **384 cases**. The
`mesh_device` fixture is **function-scoped**, so every non-skipped case re-inits FABRIC_1D on
the sim (~30 s each), and the multi-device runner has a 900 s per-topology wall-clock cap — a
full 384-case run *cannot* complete under that cap. I ran a **curated 56-case subset** (6
runs, artifacts merged) chosen to span every verifier category and every SUPPORTED axis value
that executes the kernel:

- **supported_pass (8):** dtype {bf16, f32} × layout {TILE, ROW_MAJOR} × alignment
  {tile-aligned (256²), non-tile-aligned (48×64)}, all at `gather_dim=-4`, Linear — the full
  executing rectangle. All 8 bit-exact (PCC=1.0, RMS=0).
- **xfail_expected (40):** `gather_dim ∈ {-3,-2,-1}` (24), `topology=Ring` (8),
  `dtype=bfloat8_b` (8) — all rejected by `validate()` with `UnsupportedAxisValue`
  (a `NotImplementedError`), recorded as XFAIL.
- **invalid_skipped (8):** bf8b + ROW_MAJOR — skipped, never reached the op.

| Category | Count |
|---|---|
| supported_pass | 8 |
| xfail_expected | 40 |
| invalid_skipped | 8 |
| **supported_fail** | **0** ✅ |
| **xpass_drift** | **0** ✅ |
| **xfail_wrong_mode** | **0** ✅ |
| supported_marked_xfail | 0 |
| no_axes_found | 0 |

**All three loud categories are 0 — the golden run is clean.** SUPPORTED honestly describes
observed behavior. The subset validates the rejection *mechanism* (`validate()` checks all axes
uniformly, so Ring/bf8b/gather_dim reject through the identical code path) and the full
executing rectangle; the un-run supported cells differ only in shape (all pure byte copy, so
covered by the alignment/dtype/layout equivalence already exercised).

---

## Recommendations

1. **Refinement ordering** (see `op_requirements.md`): the three axis gaps — `gather_dim
   {-3,-2,-1}`, `topology=Ring`, `dtype=bfloat8_b` — are mutually independent (pure byte
   movement, orthogonal axes), so ordering is by value/narrative, not hard dependency. Do the
   strided-concat `gather_dim` expansion first (biggest functional lever, most design coverage),
   Ring second, bf8b last (smallest; format-only).
2. **No skill in the current inventory covers these refinements.** `/numeric-formats-metal`
   specifically does **not** apply to bf8b here — this op has no compute kernel, no
   `ComputeKernelConfig`, no math fidelity / dest-acc / intermediate CBs. bf8b support is purely
   a page-geometry / packet-framing concern (block-float pages copied verbatim). All three are
   verifier-authored (no skill pointer). `/interleaved-parallel` also does not apply — the op is
   already multi-device with a fixed CCL work split (one worker per direction per device); there
   is no single-core→multi-core stamp to add.
3. **Perf (no failing cell — not a refinement):** (a) the per-page `noc_async_write_barrier()`
   added by the correctness fix serializes the fabric pipeline; a persistent-scratch or
   per-block-flush rewrite would recover pipelining. (b) The design's optional
   `arm_scatter_write` (≤4 pages/packet coalescing) is unused — the per-page path is simplest and
   correct; scatter is a throughput lever for later. Neither unlocks a SUPPORTED axis value.
4. **L1 is bounded and safe** — `cb_relay_pages` is a constant 4 pages regardless of shard size;
   no OOM risk on any shape, so no memory-budget refinement is warranted.
5. **Real-hardware validation** is the outstanding gap: the op is verified only on the
   deterministic sim, which cannot expose async-NoC races (the WAR hazard fixed above being the
   canonical example). When T3K hardware is available, re-run the acceptance + golden suites to
   confirm the fabric ordering holds under genuine async DMA.
