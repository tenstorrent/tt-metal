# Verification Report: point_to_point

**Op kind:** multi-device CCL (collective communication) — pure cross-chip byte
movement, no arithmetic. Self-contained Python `ttnn.generic_op` +
`ttnn.MeshProgramDescriptor` with four newly-authored fabric dataflow kernels
(two endpoint programs: SEND + RECEIVE).

**Verification date:** 2026-07-02 (verifier pass 2 — mechanized golden + precision
now *observed* on the `bh_8xP150_p2p` sim; pass 1 was code-review + analytical only).

---

## TL;DR

- **Golden suite ran through `eval.verify_supported` on the sim, and is clean.**
  Every SUPPORTED axis value is now *observed*: 28 `supported_pass` + 2
  `invalid_skipped` (bf8b×ROW_MAJOR), **all three loud categories 0**
  (`supported_fail` / `xpass_drift` / `xfail_wrong_mode`). This is the first pass
  in which the registry verifier CLI ran on real (sim-)observed results — the
  golden-test scaffold (`test_golden.py` / `helpers.py` / `conftest.py`) now
  exists (it did not at pass 1).
- **`SUPPORTED == TARGET` on every axis** (dtype, layout, topology, alignment), so
  `TARGET − SUPPORTED = ∅`. `xfail_expected` is empty *by construction* and there
  are **no axis-expansion refinements** to file. The one structurally-removed
  region (`bf8b×ROW_MAJOR`) is an `INVALID` entry, correctly skipped.
- **Precision baseline: PCC = 1.0, zero error** on all 4 shapes × {bf16, f32} —
  exactly the identity-copy expectation for a bit-for-bit byte mover.
- **Two code-review fixes landed** this pass (both below). No architectural rework
  needed; kernels/descriptor/host assembly are correct by line-by-line review
  against `ccl_helpers_dataflow.hpp` and confirmed green on the sim.

---

## Code Review

### Fixed this pass

1. **Dead boolean clause in the page-size validation** (`point_to_point.py`,
   `validate()`). Was:
   ```python
   if page % 16 != 0 and page != 16:
   ```
   The `and page != 16` is unreachable dead code: whenever `page % 16 != 0` is
   true, `page != 16` is necessarily true (16 is divisible by 16), so it never
   changes the result. Simplified to the honest predicate:
   ```python
   if page % 16 != 0:
   ```
   Behaviour-identical; confirmed by the 28-cell golden run (every supported cell
   passed through the edited `validate()`).

2. **Precision-baseline mesh-shape / topology mismatch** (`test_point_to_point_
   precision_baseline.py`). The test opened `mesh_device == (1, 2)`, which does
   **not** match any p2p sim topology (`bh_8xP150_p2p` is `(2,4)`), so it would
   **hang fabric init** ("Fabric Router Sync: Timeout") — a test/topology mismatch,
   not an op defect. Changed to `MESH_SHAPE = (2, 4)` to match the required
   topology (identical to the immutable acceptance suite). After the fix it runs
   8/8 green on the sim (see Precision Baseline below).

### Reviewed clean (no change needed)

- **Helper usage is idiomatic and matches `ccl_helpers_dataflow.hpp` exactly.**
  Fabric egress (the SENDING half) goes through the safety-by-construction helper
  chain `FabricStreamSender → open(unicast_route) → arm_unicast_write / arm_inc →
  write_page / inc → close`; the receiver's one-shot "ready" uses
  `FabricStreamSender::signal`. The raw-API fallbacks — `noc_semaphore_wait_min` /
  `noc_semaphore_set` (the WAITING half + cache-reuse re-arm), the local
  `noc_async_read` receive ingress (there is no `FabricStreamReceiver` by design),
  and `tt_memmove` page↔packet coalescing — are **precisely** the ones the helper
  banner (`ccl_helpers_dataflow.hpp:69-93`) assigns to the op. Nothing is
  under-used or should be fused. `mcast_pipe.hpp` is N/A — this is a unicast
  transfer, and the mcast path is layered into `ccl_helpers_dataflow.hpp` anyway.
- **CB sync (push == wait).** `cb_input_pages`: SEND reader pushes `num_pages`,
  SEND writer waits/pops `num_pages`. `cb_output_pages`: RECEIVE reader pushes
  `num_pages`, RECEIVE writer waits/pops `num_pages`. `cb_packet_scratch`:
  single-slot working scratch (reserve-once, addressed directly, no cross-kernel
  handoff) on both endpoints — correctly *not* a balanced producer/consumer CB. ✔
- **Coalesce / de-coalesce loops.** Traced both regimes: multi-page-per-packet
  (`pages_per_packet > 1`, `page_segments == 1`) and segmented
  (`page_segments > 1`, `pages_per_packet == 1`), incl. the short-last-packet case.
  The sender's `min(P, end-page_idx-1)` recompute (after the filling page) and the
  receiver's `min(P, end-page_idx)` recompute (before de-coalescing) are the
  correct asymmetric mirror. *(Coverage note: the segmented regime is not
  reachable by any `feature_spec.INPUTS` shape — the largest RM page is ~2 KB,
  well under the fabric max packet — so it is present-and-reviewed but not
  exercised on the sim. Not a refinement: no axis, no failing cell.)* ✔
- **Fabric arg contract.** Both fabric-using kernels carry 9 scalar RT args
  (indices 0–8); the `[has_forward][fwd?][has_backward][bwd?]` block begins at
  index 9, matching `conn_arg_idx = 9` and the host `_append_fabric_rt_args`
  layout. ✔
- **Handshake + cache-reuse reset placement.** Sender resets the shared sem
  *before* its outgoing "done" inc; receiver resets *after* its "done" wait — the
  ordering the banner requires to survive program-cache reuse. **Observed green:**
  the acceptance `program_cache` test passes on both call 0 and call 1 on the
  current (edited) tree. ✔
- **API correctness.** `void kernel_main()` (not the deprecated namespace form);
  includes use `api/dataflow/dataflow_api.h`; addressing uses `TensorAccessor`
  (not deprecated `InterleavedAddrGen`); the interleaved reader/writer take the
  page size as a *runtime* arg to override a possibly-stale compile-time value. ✔
- **`validate()` shape.** First line of the public entry point; structural checks
  (MeshDevice, 2-D view, distinct in-mesh coords on a shared row/col,
  interleaved-only, 16-byte page alignment, output/intermediate spec equality),
  then the per-axis SUPPORTED gate, then EXCLUSIONS — order correct; raises typed
  `UnsupportedAxisValue` / `ExcludedCell`. ✔

### Minor observations (noted, not fixed — churn/risk not justified)

- **`round_up` (sender writer) vs `align` (receiver reader)** for the aligned page
  size. Both compute the identical value; harmonizing is cosmetic and a needless
  churn risk on kernels that otherwise mirror the reference verbatim. Left as-is.
- **`_get_or_create_semaphore` is keyed by `id(mesh_device)`.** Correct for the
  intended long-lived-mesh usage (create-once-per-mesh, cached across cache hits),
  but `id()` could in principle alias after a mesh_device is GC'd and a new one is
  allocated at the same address. Not a practical concern (meshes are long-lived);
  a `WeakValueDictionary` keyed on the device would be marginally more robust.
- **Intermediate staging as `uint32 ROW_MAJOR`** rather than "same TensorLayout as
  input" (design text). It is addressed per-packet as raw bytes (page size
  overridden to `packet_size_bytes`), so dtype/layout are cosmetic, and `uint32`
  sidesteps `element_size` being undefined for `bfloat8_b`. This is *more* correct
  than the design text (it makes bf8b work) — an intentional, documented deviation.

---

## Registry Conformance

- **INPUT_TAGGERS** — `{"alignment": tag_alignment}`, signature `(inputs, axes)`. ✔
- **SUPPORTED** — declares every gated axis: `dtype`, `layout`, `topology`,
  `alignment`. ✔
- **EXCLUSIONS** — present, empty (`[]`). No cell inside SUPPORTED is refused. ✔
- **validate()** — first line of `point_to_point(...)`; checks SUPPORTED then
  EXCLUSIONS; raises the registry-model refusal types. ✔
- **Op file does NOT declare INVALID** — confirmed (`hasattr(module, "INVALID")`
  is `False`); INVALID lives only in `feature_spec.py`. ✔

### INVALID audit (`eval/golden_tests/point_to_point/feature_spec.py`)

`INVALID = [{"dtype": ttnn.bfloat8_b, "layout": ttnn.ROW_MAJOR_LAYOUT}]`.

- **Single-tensor coupling:** `dtype` and `layout` both describe the *input*
  tensor — no cross-tensor coupling. ✔
- **Universe-must-change:** `bfloat8_b` is a block-quantized tiled format with no
  row-major representation — ttnn cannot construct the tensor. A data-format
  impossibility, not a not-yet-implemented EXCLUSION. ✔
- **Canonical bf8b×ROW_MAJOR entry present** (the mandatory entry for a tile-or-RM
  activation op). ✔ Observed correctly `invalid_skipped` (2 cells) in the golden
  run.
- Not a norm-like op → no weight/canonicalization cells expected. ✔

Verdict: INVALID is well-formed; no change recommended.

---

## Precision Baseline

`tests/ttnn/unit_tests/operations/point_to_point/test_point_to_point_precision_baseline.py`
— **8/8 pass** on `bh_8xP150_p2p` (mesh `(2,4)`, FABRIC_1D), comparing the
receiver's output shard against the *device-resident* sender input shard (identity
oracle). Run: `scripts/run_multidevice_sim_pytest.py --topology bh_8xP150_p2p -- <this test>`.

| Shape | dtype | PCC | Max Abs Err | Mean Abs Err | Rel RMS Err |
|-------|-------|-----|-------------|--------------|-------------|
| (1,1,32,32)   | bfloat16 | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| (1,1,32,32)   | float32  | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| (1,1,64,128)  | bfloat16 | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| (1,1,64,128)  | float32  | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| (1,1,96,64)   | bfloat16 | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| (1,1,96,64)   | float32  | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| (1,1,512,512) | bfloat16 | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| (1,1,512,512) | float32  | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |

**Assessment:** point_to_point is a *bit-for-bit byte copy*. The receiver shard is
exactly equal to the device-resident sender shard — PCC = 1.0 and zero error for
every dtype (any bf8b/bf16 quantization happens at `from_torch`, before the
transfer, so the transfer itself adds no error). This is the strongest possible
precision result and matches the identity oracle exactly.

**Recommended tolerances** (identity oracle): PCC ≥ 0.9999 (f32) / 0.999 (bf16) /
0.99 (bf8b); `atol`/`rtol` effectively 0 for float dtypes. The suite uses PCC as
the gate, matching the acceptance suite's per-dtype thresholds.

---

## Verifier CLI Summary

Because this is a multi-device CCL op, the single-device `eval/eval_test_runner.sh`
is the wrong driver (it neither sets up the sim mesh nor enables the fabric — it
would hang fabric init on this single-Blackhole host). The golden suite was instead
driven through the deterministic multi-device sim runner
(`scripts/run_multidevice_sim_pytest.py --topology bh_8xP150_p2p`) with the axes
plugin (`-p eval.axes_plugin`, `PYTEST_AXES_JSON=...`) and `--junitxml`; the junit
was classified via `eval.classify_failures` and joined by `eval.verify_supported`.
The two-run coverage (32×32 full dtype×layout×topology cartesian + integer dtypes
on a non-tile-aligned 48×64 shape) was merged (dedup by nodeid) so a **single**
verifier report spans every axis value. Artifact:
`eval/results/point_to_point/verifier_report.json`.

Observed on the merged 30-cell run (covers dtype∈{bf16,f32,bf8b,uint16,int32,uint32},
layout∈{TILE,ROW_MAJOR}, topology∈{Linear,Ring}, alignment∈{tile_aligned,non_tile_aligned}):

- supported_pass: **28**
- invalid_skipped: **2**   (both bf8b×ROW_MAJOR)
- supported_fail: **0**      ✅ (ship gate)
- xpass_drift: **0**         ✅ (ship gate)
- xfail_wrong_mode: **0**    ✅ (ship gate)
- xfail_expected: **0**      (correct — `SUPPORTED == TARGET`, no cell is outside SUPPORTED except INVALID)
- total: 30

Why `xfail_expected` is 0: `TARGET − SUPPORTED = ∅` on every axis, so the golden
harness generates no out-of-SUPPORTED (xfail) cells — every non-INVALID cell is
expected to pass, and does. This is the honest end state for a pure-byte-mover
whose kernels are dtype/layout/alignment-agnostic, **not** a coverage gap.

Corroborating: the immutable acceptance suite (`test_point_to_point.py`, 60 items —
bf16/f32/bf8b × TILE/RM × 5 shapes × {Linear,Ring} + program-cache + ring-wraparound
+ output_tensor + non-participating) passed 60/60 on the same sim (commit
`fb36a371`), and the `program_cache` case was re-confirmed green on the current tree
this pass.

---

## Recommendations

1. **No axis-expansion refinements exist** — see `op_requirements.md`. Every
   `(axis, missing_value)` from `TARGET − SUPPORTED` is empty; the only
   structurally-removed cell is the `INVALID` bf8b×ROW_MAJOR. The refinement queue
   is intentionally empty (documented there against the three sanity gates).

2. **Out-of-TARGET forward-looking enhancements** (NOT refinements — they need
   `/golden-tests` to widen TARGET first, then they become the first real
   refinements):
   - **Sharded memory config.** `validate()` rejects sharded input ("interleaved
     only"). TARGET has no `memory_config` axis, so this is not a refinement today.
     When it lands it is a *memory-config* refinement (sharded reader/writer), and
     — if it carries a real cross-shard data dependency (width-sharded coordination)
     — could warrant `/interleaved-parallel`'s exclusion note (sharded data placement
     constrains the work split). Plain sharded I/O bundles with the memory-config
     refinement.
   - **Multi-core / multi-link fan-out.** Single-core `(0,0)` / single-link
     (`_LINK_IDX = 0`) per endpoint by design. Striping `total_packets` across cores
     with `ttnn.split_work_to_cores` is a *performance* enhancement with no SUPPORTED
     axis and no failing cell to point at — not a refinement. (If it were ever filed
     under a widened TARGET, note it is **not** embarrassingly-parallel in the
     `/interleaved-parallel` sense: the two endpoints coordinate over the fabric with
     a real data dependency, so it is closer to the "multi-core with real data
     dependency" standalone class — no current skill covers the fabric ring/handshake
     topology.)

3. **Coverage caveat for future passes (not a defect):** the segmented-packet
   de-coalescing path (`page_segments > 1`, page larger than a fabric packet) is
   implemented and reviewed but is **not reachable** by any `feature_spec.INPUTS`
   shape (max RM page ≈ 2 KB ≪ fabric max packet). If a future `/golden-tests` pass
   adds a very-wide-last-dim shard shape, that path would begin to be exercised.
   No action now — no axis, no failing cell.
