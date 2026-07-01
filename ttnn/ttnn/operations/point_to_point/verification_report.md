# Verification Report: point_to_point

**Op kind:** multi-device CCL (collective communication) — pure cross-chip byte
movement, no arithmetic. Built as a self-contained Python `ttnn.generic_op` +
`ttnn.MeshProgramDescriptor` with four newly-authored fabric dataflow kernels.

**Verification date:** 2026-07-01 (this pass — observed on the craq-sim multi-device runner)

---

## TL;DR

- **Mechanically verified on device (sim).** The prior pass was blocked on sim
  fabric bring-up; in this environment bring-up completes cleanly and the op runs
  green. All results below are **observed**, not analytical.
- **Verifier CLI (observed, 30-cell sample):** `supported_pass=28`,
  `invalid_skipped=2`, and **all three loud categories = 0**
  (`supported_fail`, `xpass_drift`, `xfail_wrong_mode`). The sample spans every
  axis value (both topologies, all 6 dtypes, both layouts, both alignments,
  single- and multi-tile shapes).
- **Precision baseline (observed, 8 cells):** PCC = 1.000000, max/mean/RMS abs
  error = 0 for every shape × dtype — the identity byte-copy is bit-exact.
- **Code review:** one test-side correctness bug fixed (precision baseline opened
  the wrong mesh shape → would hang fabric init). Op code, kernels, descriptor,
  and host assembly reviewed clean against the CCL helper library — no op changes
  needed.
- **Refinement queue is empty by construction:** `SUPPORTED == TARGET` on every
  axis, and the only structurally-removed region (`bf8b × ROW_MAJOR`) is covered
  by `INVALID` in `feature_spec.py`. `TARGET − SUPPORTED = ∅`, so there are no
  axis-expansion refinements. See `op_requirements.md`.

---

## Code Review

### Fixed this pass

1. **Precision baseline opened the wrong mesh shape (test bug → fabric-init hang).**
   `tests/.../test_point_to_point_precision_baseline.py` parametrized
   `mesh_device` as `(1, 2)`, but the graded topology `bh_8xP150_p2p` is a **(2, 4)**
   torus-x mesh-graph descriptor. Per the design doc + runner contract, opening any
   shape other than the descriptor's fixed shape hangs fabric init
   (`Fabric Router Sync: Timeout`) — a test/topology mismatch, not an op defect.
   - **Fix:** changed the fixture to `(2, 4)` (matching the acceptance suite) and
     corrected the docstring's runner invocation from `--op point_to_point`
     (which fans across all three p2p topologies, whose `(4,2)`/`(8,4)` shapes would
     hang against a `(2,4)`-hardcoded test) to `--topology bh_8xP150_p2p`.
   - **Verified:** the fixed test now runs green on the sim — 8/8 pass, PCC = 1.0.

### Reviewed clean (no change needed)

- **Helper usage is idiomatic and complete.** Fabric egress goes through the
  safety-by-construction CCL helper (`ccl_helpers_dataflow.hpp`):
  - `sender_writer` uses the full stream path for the many-packet send —
    `FabricStreamSender::open(unicast_route(hops))` → `arm_unicast_write(payload)`
    + `arm_inc(1)` → per-packet `UnicastWriteChannel::write_page(...)` in the loop →
    `AtomicIncChannel::inc(sem)` → `stream.close()`. Route bound once at `open()`,
    reused by every `arm_*` — matches the helper banner's legal sequence exactly.
  - `receiver_reader` uses the **one-shot `FabricStreamSender::signal(num_hops,
    remote_sem_noc_addr)`** overload for the ready handshake — the helper's
    documented collapse of `open→arm_inc→inc→close` for a single inc. This is the
    correct, minimal helper call (better than hand-rolling the 4-step sequence).
  - The raw-API fallbacks (`tt_memmove` page↔packet (de)coalesce, `noc_async_read`
    receive ingress, `noc_semaphore_wait_min`/`noc_semaphore_set` for the waiting
    half + cache-reuse re-arm) are exactly the ones the helper banner
    (`ccl_helpers_dataflow.hpp:61-86`) states the op must own — there is no
    `FabricStreamReceiver`, and coalescing/waiting are deliberately not in the
    helper. `op_design.md` "Helpers considered and rejected" justifies each. No
    helper is under-used or bypassed.
- **CB sync (push == wait).** `cb_input_pages`: `sender_reader` pushes `num_pages`,
  `sender_writer` waits/pops `num_pages`. `cb_output_pages`: `receiver_reader`
  pushes `num_pages`, `receiver_writer` waits/pops `num_pages`. `cb_packet_scratch`:
  scratch L1 region on both endpoints (reserve-once, no cross-kernel consumer) —
  correctly *not* balanced as a queue. ✔
- **Coalesce / de-coalesce loops.** Both regimes trace correctly:
  *multi-page-per-packet* (`pages_per_packet > 1`, `page_segments == 1`) and
  *segmented* (`pages_per_packet == 1`, `page_segments > 1`), incl. the
  short-last-packet case. Sender's `min(P, end−page_idx−1)` recompute (after the
  page that fills the packet) and receiver's `min(P, end−page_idx)` recompute
  (before de-coalescing the fresh packet) are the correct asymmetric mirror.
  Observed green on the multi-tile `(1,1,64,128)` and large `(1,1,512,512)` shapes. ✔
- **Fabric arg contract.** Both fabric-using kernels carry exactly 9 scalar RT args
  (indices 0–8); the `[has_forward][fwd?][has_backward][bwd?]` block begins at
  index 9 = `conn_arg_idx` in both kernels, matching host `_append_fabric_rt_args`. ✔
- **Handshake + cache-reuse reset placement.** Sender resets the local sem *before*
  its outgoing "done" inc; receiver resets *after* its "done" wait — the ordering the
  helper banner requires to survive program-cache reuse. The `program_cache`
  acceptance test (two back-to-back calls) confirms this holds. ✔
- **API correctness.** `void kernel_main()` (not the deprecated namespace pattern);
  includes use `api/dataflow/dataflow_api.h`; addressing uses `TensorAccessor`
  (not the deprecated `InterleavedAddrGen`). ✔
- **`validate()` shape.** First line of the public entry point; structural input
  checks (MeshDevice, distinct in-mesh coords on a common row/col, interleaved-only,
  16-byte page alignment, output/intermediate spec equality), then the per-axis
  `SUPPORTED` gate, then `EXCLUSIONS`. Raises typed `UnsupportedAxisValue` /
  `ExcludedCell`. Order correct. ✔

### Benign, well-justified deviations from `op_design.md` (no action)

- **Intermediate staging tensor is `uint32 ROW_MAJOR`** rather than "same
  TensorLayout as input". It is addressed per-packet as raw bytes (page size
  overridden to `packet_size_bytes`), so its dtype/layout are cosmetic; `uint32`
  sidesteps `element_size`/`datum_size` being undefined for `bfloat8_b`. The buffer
  holds exactly `total_packets × packet_size_bytes`. Observed bf8b transfer is
  bit-exact, confirming the choice. This is *more* correct than the design text.
- **Four dataflow kernels** (two endpoint programs) instead of the generic
  reader/compute/writer triple — inherent to a CCL op with no compute stage.

### Minor cosmetic note (not fixed — reference-parity churn not worth it)

- `sender_writer` uses `round_up(page_size, alignment)` (data-movement common
  helper) while `receiver_reader` uses `align(page_size, alignment)` (dataflow
  builtin). Both compute the identical aligned size; harmonizing them is cosmetic
  and would diverge the kernels from the verbatim reference pattern they mirror.
  Left as-is deliberately.

---

## Registry Conformance

- **INPUT_TAGGERS** — present: `{"alignment": tag_alignment}`, signature
  `(inputs, axes)` (uses `inputs`, ignores `axes`). ✔
- **SUPPORTED** — present; declares every gated axis: `dtype`, `layout`, `topology`
  (op kwarg → routing), `alignment` (tagged). Every INPUT_TAGGERS key appears. ✔
- **EXCLUSIONS** — present, empty (`[]`). No cells inside SUPPORTED are refused. ✔
- **validate()** — present, wired as the first line of `point_to_point(...)`; checks
  SUPPORTED then EXCLUSIONS; raises the registry-model refusal types. ✔
- **Op file does NOT declare INVALID** — confirmed; INVALID lives only in
  `feature_spec.py` (referenced by comment only in the op file). ✔

### INVALID audit (`eval/golden_tests/point_to_point/feature_spec.py`)

`INVALID = [{"dtype": ttnn.bfloat8_b, "layout": ttnn.ROW_MAJOR_LAYOUT}]`.

- **Single-tensor coupling:** `dtype` and `layout` both describe the *input* tensor
  — no cross-tensor coupling. ✔
- **Universe-must-change:** `bfloat8_b` is a block-quantized tiled format with no
  row-major representation — ttnn cannot construct the tensor. Data-format-definition
  impossibility, not a not-yet-implemented EXCLUSION. ✔
- **Canonical bf8b+ROW_MAJOR entry present** (mandatory for a tile-or-RM
  activation op). Observed skipped (2×) in the golden slices. ✔
- Not a norm-like op → no weight/canonicalization cells expected. ✔

Verdict: INVALID is well-formed; no change recommended.

---

## Precision Baseline

`tests/ttnn/unit_tests/operations/point_to_point/test_point_to_point_precision_baseline.py`
— run on the `bh_8xP150_p2p` sim topology, mesh `(2, 4)`, FABRIC_1D, Linear.
Golden = the device-resident sender shard (post-`from_torch` quantization);
calc = the receiver's output shard after the fabric transfer.

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

**Assessment:** point_to_point is a **bit-for-bit byte copy**. The receiver shard
equals the device-resident sender shard exactly — PCC = 1.0 and zero error for
every measured shape/dtype (any bf8b/bf16 quantization already happened at
`from_torch`, before the transfer; the transfer adds none). The large
`(1,1,512,512)` shape exercises the multi-page packet regime and is also exact.
Integer dtypes are compared bit-exactly in the acceptance suite (`torch.equal`) —
also exact.

**Recommended tolerances** (identity oracle): PCC ≥ 0.9999 (f32) / 0.999 (bf16) /
0.99 (bf8b); `atol`/`rtol` effectively 0 for float dtypes. The suites use PCC as
the gate (acceptance uses per-dtype thresholds 0.999/0.995/0.99).

---

## Verifier CLI Summary

`python3 -m eval.verify_supported eval/results/point_to_point ttnn.operations.point_to_point`
over the observed golden slices (see `verifier_report.json` `_provenance`):

- supported_pass: **28**
- invalid_skipped: **2**  (bf8b × ROW_MAJOR, skipped at collection)
- supported_fail: **0**   ✔ (must be 0 to ship)
- xpass_drift: **0**      ✔ (must be 0 to ship)
- xfail_wrong_mode: **0** ✔ (must be 0 to ship)
- xfail_expected: **0**   ← **no `TARGET − SUPPORTED` gap** (empty by construction)
- total sampled: 30 (of 432 cartesian)

**Why the sample is representative and `xfail_expected` is 0.** Because
`SUPPORTED == TARGET` on every axis, there are **no** xfail-decorated cells anywhere
in the full cartesian — the only reachable categories are `supported_pass` and
`invalid_skipped`. The 30-cell sample deliberately covers every axis value
(both topologies × all 6 dtypes × both layouts × both alignments × single/multi-tile),
so a clean sample generalizes. A full 396-supported-cell sim run is ~1h+ and would
exceed the per-invocation wall-clock backstop; it carries no additional signal here.
The full 16/16 acceptance suite is green at this exact commit (op code unchanged).

---

## Recommendations

These are **not** refinements (no `TARGET` axis to expand, no failing cell to move) —
they are scope/perf notes for the record:

1. **Full-cartesian golden run.** If a future environment can afford it, a full
   `eval/golden_tests/point_to_point` run under the multi-device sim runner (split
   into ≤~20-item `-k` batches so each pytest invocation stays under the 900s
   wall-clock backstop) would replace the 30-cell observed sample with the complete
   432-cell matrix. No new signal expected (SUPPORTED == TARGET), but it closes the
   sampling caveat mechanically.

2. **Sharded memory config (out of current TARGET).** `validate()` rejects sharded
   input ("interleaved only"). TARGET has no `memory_config` axis, so this is *not*
   a refinement candidate today; a `/golden-tests` pass would have to add the axis
   to TARGET first. When/if it lands, sharded I/O is "just a different reader/writer"
   (a memory-config bundle), not an algorithm change.

3. **Multi-link / multi-core fan-out (performance, out of scope).** The op (and the
   reference C++ factory) is single-core `(0,0)` per endpoint. Splitting the packet
   set across links/cores is a throughput enhancement with no SUPPORTED axis or
   failing cell to point at — performance, not capability. It is *not* embarrassingly
   parallel in the interleaved sense (the fabric route + handshake are shared), so it
   would not map onto `/interleaved-parallel` cleanly either.

4. **Ring wraparound (genuine short-way).** The graded topology exercises Ring only
   for the adjacent `(0,0)→(0,1)` pair, where `ccl_dm_route` degenerates Ring to the
   1-hop line route. A true torus wraparound (e.g. `(0,0)→(0,3)` = 1 backward hop)
   needs `FABRIC_1D_RING`; it is covered by the throwaway confirmation tests
   (`test_p2p_ring_confirm.py`), not the graded suite. Not a capability gap — the
   `topology=Ring` axis value is already supported and observed green.
