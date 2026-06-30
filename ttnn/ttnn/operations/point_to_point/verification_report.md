# Verification Report: point_to_point

**Op kind:** multi-device CCL (collective communication) — pure cross-chip byte
movement, no arithmetic. Built as a self-contained Python `ttnn.generic_op` +
`ttnn.MeshProgramDescriptor` with four newly-authored fabric dataflow kernels.

**Verification date:** 2026-06-24

---

## TL;DR

- **Code review:** one registry-conformance bug fixed (missing `alignment` axis +
  `tag_alignment` tagger). Kernels, descriptor, and host assembly are otherwise
  correct by line-by-line review against the helper library and the reference
  CCL kernels — CB push/pop balanced, fabric arg contract exact, handshake +
  cache-reuse reset placement correct.
- **Registry:** after the fix, `SUPPORTED` fully covers `TARGET` on every axis
  (dtype, layout, topology, alignment). The golden cartesian is 432 cells: 396
  expected-supported + 36 INVALID-skipped (bf8b×ROW_MAJOR). **TARGET − SUPPORTED
  is empty — there are no axis-expansion refinements.**
- **On-device verification is BLOCKED** by an infrastructure defect (sim fabric
  bring-up), not by the op. The `SUPPORTED` claims are therefore **plausible by
  review but unverified on hardware**. This is the headline caveat; see
  "On-device verification" below.

---

## Code Review

### Fixed

1. **Registry conformance — missing `alignment` axis + tagger** (`point_to_point.py`).
   The op shipped `INPUT_TAGGERS = {}` and a `SUPPORTED` block with no `alignment`
   key, but `eval/golden_tests/point_to_point/feature_spec.py` declares `alignment`
   as a TARGET axis and its docstring explicitly specifies
   `tag_alignment(inputs, axes) -> "tile_aligned" | "non_tile_aligned"`.
   - **Consequence of the bug:** the golden harness (`eval/feature_matrix.cartesian`)
     treats any TARGET axis *not* in `INPUT_TAGGERS` as a **finite** axis and
     iterates it. With no tagger, `alignment` would have been cartesian-multiplied
     as a free axis — every shape generating one `tile_aligned` *and* one
     `non_tile_aligned` case whose label was decoupled from the actual shape
     (spurious double-counting; the support check would also silently ignore the
     axis because `SUPPORTED` had no `alignment` key).
   - **Fix:** added `tag_alignment` (examines the shard's last two dims; both
     divisible by 32 → `tile_aligned`), `INPUT_TAGGERS = {"alignment": tag_alignment}`,
     and `SUPPORTED["alignment"] = ["tile_aligned", "non_tile_aligned"]`. The op is
     pure byte movement and never tilizes/untilizes — it copies the physical pages
     (padded tiles for TILE, last-dim rows for ROW_MAJOR) verbatim — so it is
     alignment-agnostic and genuinely supports both values. `validate()` already
     iterates `INPUT_TAGGERS` generically, so no further change was needed there.
   - Verified: `tag_alignment` returns the right value for tile-aligned (`32×32`,
     `32×64`, rank-2 `32×64`) and non-tile-aligned (`48×64`, `24×24`) shards.

### Reviewed clean (no change needed)

- **CB sync (push == wait).** `cb_input_pages`: reader pushes `num_pages`, writer
  waits/pops `num_pages`. `cb_output_pages`: receiver-reader pushes `num_pages`,
  receiver-writer waits/pops `num_pages`. `cb_packet_scratch`: scratch L1 region
  (reserve-once, no cross-kernel consumer) on both endpoints — correctly *not*
  balanced. ✔
- **Coalesce / de-coalesce loops.** Traced both regimes by hand:
  *multi-page-per-packet* (`pages_per_packet > 1`, `page_segments == 1`) and
  *segmented* (`pages_per_packet == 1`, `page_segments > 1`), including the
  short-last-packet case. The sender's `min(P, end-page_idx-1)` recompute (computed
  *after* the page that fills the packet) and the receiver's `min(P, end-page_idx)`
  recompute (computed *before* de-coalescing) are the correct asymmetric mirror of
  each other. ✔
- **Fabric arg contract.** Both fabric-using kernels carry exactly 9 scalar RT
  args (indices 0–8); the `[has_forward][fwd?][has_backward][bwd?]` block begins at
  index 9, matching `conn_arg_idx = 9` in both kernels and the host
  `_append_fabric_rt_args` layout. ✔
- **Handshake + cache-reuse reset placement.** Sender resets the local sem *before*
  its outgoing "done" inc; receiver resets *after* its "done" wait — exactly the
  ordering the helper banner (`ccl_helpers_dataflow.hpp:67-69`) requires to survive
  program-cache reuse. The waiting half is a raw `noc_semaphore_wait_min`, the
  sending half is `AtomicIncChannel::inc` over fabric — the documented split. ✔
- **Helper usage.** Fabric egress goes through the safety-by-construction helper
  (`FabricStreamSender → open → arm_unicast_write/arm_inc → write_page/inc → close`).
  The raw-API fallbacks (`tt_memmove` coalesce, `noc_async_read` ingress,
  `noc_semaphore_wait_min/set`) are exactly the ones the helper banner says the op
  must own — `op_design.md` "Helpers considered and rejected" justifies each. No
  helper is under-used or bypassed. ✔
- **API correctness.** `void kernel_main()` (not the deprecated namespace pattern);
  includes use `api/dataflow/dataflow_api.h`; addressing uses `TensorAccessor`
  (not the deprecated `InterleavedAddrGen`). ✔
- **`validate()` shape.** First line of the public entry point; structural input
  checks (MeshDevice, distinct in-mesh coords on a common row/col, interleaved-only,
  16-byte page alignment, output/intermediate spec equality), then the per-axis
  `SUPPORTED` gate, then `EXCLUSIONS` — order is correct. Raises typed
  `UnsupportedAxisValue` / `ExcludedCell`. ✔

### Benign, well-justified deviations from `op_design.md` (no action)

- **Intermediate staging tensor is `uint32 ROW_MAJOR`** rather than "same
  TensorLayout as input". It is addressed per-packet as raw bytes (page size
  overridden to `packet_size_bytes`), so its dtype/layout are cosmetic, and `uint32`
  sidesteps `element_size`/`datum_size` being undefined for `bfloat8_b`. The buffer
  holds exactly `total_packets × packet_size_bytes`, satisfying the binding
  capacity + per-packet-addressing contract. This is *more* correct than the design
  text (it makes bf8b work) and is documented in the changelog.
- **Four dataflow kernels** (two endpoint programs) instead of the generic
  reader/compute/writer triple — inherent to a CCL op with no compute stage.

### Minor cosmetic note (not fixed — risk not worth it)

- The sender writer uses `round_up(page_size, alignment)` (data-movement common
  helper) while the receiver reader uses `align(page_size, alignment)` (dataflow
  builtin). Both compute the identical aligned size; harmonizing them is cosmetic
  and a needless churn risk on kernels that otherwise mirror the reference verbatim.

---

## Registry Conformance

- **INPUT_TAGGERS** — present after fix: `{"alignment": tag_alignment}`, signature
  `(inputs, axes)`. ✔
- **SUPPORTED** — present; declares every gated axis: `dtype`, `layout`, `topology`,
  `alignment`. ✔
- **EXCLUSIONS** — present, empty (`[]`). No cells inside SUPPORTED are refused. ✔
- **validate()** — present, wired as the first line of `point_to_point(...)`; checks
  SUPPORTED then EXCLUSIONS; raises the registry-model refusal types. ✔
- **Op file does NOT declare INVALID** — confirmed; INVALID lives only in
  `feature_spec.py`. ✔

### INVALID audit (`eval/golden_tests/point_to_point/feature_spec.py`)

`INVALID = [{"dtype": ttnn.bfloat8_b, "layout": ttnn.ROW_MAJOR_LAYOUT}]`.

- **Single-tensor coupling:** `dtype` and `layout` both describe the *input* tensor
  — no cross-tensor coupling. ✔
- **Universe-must-change:** `bfloat8_b` is a block-quantized tiled format with no
  row-major representation — ttnn cannot even construct the tensor. This is a
  data-format-definition impossibility, not a not-yet-implemented EXCLUSION. ✔
- **Canonical bf8b+ROW_MAJOR entry present** (the mandatory entry for a tile-or-RM
  activation op). ✔
- Not a norm-like op → no weight/canonicalization cells expected. ✔

Verdict: INVALID is well-formed; no change recommended.

---

## On-device verification (BLOCKED — read this)

This op can only run on a `ttnn.MeshDevice` with ≥2 devices on a line and the
fabric enabled. Two independent blockers prevent mechanical verification on this
host:

1. **No golden-test scaffold.** `eval/golden_tests/point_to_point/` contains only
   `feature_spec.py`; there is no `test_golden.py`/`helpers.py`/`conftest.py`/`axes.py`
   (these are authored upstream by `/golden-tests`, not by the verifier). So
   `eval/eval_test_runner.sh` collects no tests and `eval.verify_supported` has no
   `test_results.json`/`test_axes.json` to consume.
2. **Sim fabric bring-up failure.** This host has **1 Blackhole** device, so the
   only multi-device path is the deterministic sim runner
   (`scripts/run_multidevice_sim_pytest.py --op point_to_point`, topology
   `bh_8xP150_p2p`, `required=true`, paths OK). Both attempts (default 15000 ms and
   a raised 600000 ms `TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS`) fail **identically**
   inside `open_mesh_device` → `initialize_fabric_and_dispatch_fw`:

   ```
   Fabric Router Sync: Timeout ... on Device 4 ... STARTED (4 core(s) stuck) ...
   Ethernet handshake likely failed -- the link may not be healthy.
   ```

   This fires in **pytest fixture setup, before the op is ever called**
   (`ttnn/ttnn/distributed/distributed.py:671`). 600 s of wall clock is far more
   than enough sim time, so this is a genuine handshake failure in the
   `blackhole_8xP150_torus_x` sim model — a **sim-data / infra defect, not an op
   defect**. Verdict from the runner: `MULTIDEV_SIM_RESULT[bh_8xP150_p2p]: HANG`,
   aggregate exit 2.

**Consequence:** the precision baseline and acceptance suites could not be executed,
and the verifier CLI could not be run on observed results. The `SUPPORTED` block is
verified by *code review and analytical gap analysis only*. Re-running on real
≥2-device Blackhole hardware (or a fixed sim-bh fabric model) is a **mandatory
follow-up** — tracked in `op_requirements.md` as verification debt (it is a
verification task, not an axis refinement).

---

## Precision Baseline

`tests/ttnn/unit_tests/operations/point_to_point/test_point_to_point_precision_baseline.py`
was written and **collects cleanly (8 items)**, but **could not be executed** (see
blocker above). It measures PCC, max/mean abs error, and relative RMS error over
4 shapes × {bf16, f32} × Linear, comparing the receiver's output shard against the
*device-resident* sender input shard.

| Shape | dtype | PCC | Max Abs Err | Mean Abs Err | Rel RMS Err |
|-------|-------|-----|-------------|--------------|-------------|
| (1,1,32,32)   | bf16/f32 | *expected ≈ 1.0* | *expected 0* | *expected 0* | *expected 0* |
| (1,1,64,128)  | bf16/f32 | *expected ≈ 1.0* | *expected 0* | *expected 0* | *expected 0* |
| (1,1,96,64)   | bf16/f32 | *expected ≈ 1.0* | *expected 0* | *expected 0* | *expected 0* |
| (1,1,512,512) | bf16/f32 | *expected ≈ 1.0* | *expected 0* | *expected 0* | *expected 0* |

**Assessment (analytical):** point_to_point is a *bit-for-bit byte copy*. The
receiver shard should equal the device-resident sender shard exactly — PCC = 1.0
and zero error for every dtype (any bf8b/bf16 quantization already happened at
`from_torch`, before the transfer, so the transfer adds no error). Numbers above
are *expected*, not measured — pending the on-device run.

**Recommended tolerances** (identity oracle): PCC ≥ 0.9999 (f32) / 0.999 (bf16) /
0.99 (bf8b); `atol`/`rtol` effectively 0 for float dtypes (the suite uses PCC as
the gate, matching the acceptance suite's per-dtype thresholds).

---

## Verifier CLI Summary

The verifier CLI could not run on observed results (no golden scaffold + sim
blocker). The categories below are **analytically derived** from
`cartesian(TARGET, INPUT_TAGGERS, INPUTS)` crossed with `SUPPORTED`/`EXCLUSIONS`/
`INVALID` (i.e. the *expected* categories `eval.verify_supported._expected_category`
would assign), and saved to `eval/results/point_to_point/verifier_report.json` with
a `_provenance` block making the analytical nature explicit.

- supported (expected): 396
- xfail_expected: 0   ← **no TARGET − SUPPORTED gap**
- invalid_skipped: 36  (all bf8b × ROW_MAJOR)
- supported_fail: 0      (nothing ran — carries no signal yet)
- xpass_drift: 0         (nothing ran — carries no signal yet)
- xfail_wrong_mode: 0    (nothing ran — carries no signal yet)
- total: 432

Because `SUPPORTED == TARGET` on every axis, there are **no unsupported cells**, so
`xfail_expected` is empty and the loud drift categories are 0 by construction. They
become meaningful signals only once the suite runs on real hardware.

---

## Recommendations

1. **Run the suites on real ≥2-device Blackhole hardware** (or a fixed sim-bh fabric
   model) and re-confirm the `SUPPORTED` claims. Priority order to de-risk the
   *unverified* claims, which the single-device acceptance test never exercised:
   - **bf8b / uint16 / int32 / uint32** end-to-end (the intermediate-as-uint32 path
     and `ccl_packet_dims` framing for each element size);
   - the **segmented packet regime** (`page_segments > 1`) — reached by large pages
     (e.g. `(1,1,512,512)`), distinct code path from the multi-page-per-packet one;
   - **Ring topology** routing (`ccl_dm_route` short-way) vs Linear;
   - **program-cache reuse** (second call) — the semaphore reset ordering is the
     classic "green run 1, hang run 2" footgun.
2. **Author the golden-test scaffold** (`test_golden.py` + `helpers.py` + `axes.py`
   for `eval/golden_tests/point_to_point/`) via `/golden-tests`, so the registry
   verifier CLI can run mechanically for this op like it does for the single-device
   ops. This is an upstream (golden-test design) task, not a verifier task, but it
   is the thing standing between this op and a real `verifier_report.json`.
3. **Out-of-TARGET forward-looking enhancements** (not refinements under the current
   TARGET — they would require `/golden-tests` to first add axes to TARGET):
   - **Sharded memory config.** `validate()` rejects sharded input ("interleaved
     only"). TARGET has no `memory_config` axis, so this is not a refinement; if
     desired, add a `memory_config` axis to TARGET first.
   - **Multi-link / multi-core fan-out.** The op (and the reference C++ factory) is
     single-core per endpoint. A perf enhancement with no SUPPORTED-axis or
     failing-cell to point at — performance, not capability.
   Both are performance/scope items for `verification_report.md`, deliberately kept
   out of the refinement queue.
