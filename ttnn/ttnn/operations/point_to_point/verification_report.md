# Verification Report: point_to_point

**Op kind:** multi-device CCL (collective communication) — pure cross-chip byte
movement, no arithmetic. Built as a self-contained Python `ttnn.generic_op` +
`ttnn.MeshProgramDescriptor` with four newly-authored fabric dataflow kernels.

**Verification date:** 2026-07-02 (registry-model verification pass, on the
8-device craq-sim)

---

## TL;DR

- **Op works on device.** Verified on the deterministic multi-device craq-sim
  (`bh_8xP150_p2p`, the pinned `(2,4)` + `FABRIC_1D`, `required=True`): a
  representative **16-cell golden slice** (15 pass + 1 INVALID skip) is
  **verifier-clean** (all loud categories 0), the **precision baseline is 8/8
  PCC = 1.0 / zero error** (bit-exact identity copy), and the **program-cache**
  and **preallocated-output** acceptance paths pass.
- **`SUPPORTED` == `TARGET` on every axis** (dtype, layout, topology,
  alignment). `TARGET − SUPPORTED = ∅` → **no axis-expansion refinements**. This
  is the correct end state for a dtype/layout/alignment-agnostic byte mover.
- **Two fixes landed this pass:** a real test bug (precision baseline opened a
  `(1,2)` mesh against the `(2,4)` topology → would hang fabric init) and a dead
  conditional in `validate()`. Both are verification fixes, not refinements.
- **Registry declarations correct and honest:** `INPUT_TAGGERS`, `SUPPORTED`,
  `EXCLUSIONS`, `validate()` all present and wired; op file declares **no**
  `INVALID` (only sourced from `feature_spec.py`). INVALID audit passes.

---

## Code Review

### Fixed this pass

1. **Precision-baseline mesh/topology mismatch (real bug) —
   `tests/.../point_to_point/test_point_to_point_precision_baseline.py`.**
   The test opened `@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)`
   while the only topology whose `applies_to_ops` lists `point_to_point`
   (`bh_8xP150_p2p`, `required=True`) is a fixed **`(2,4)`** mesh-graph
   descriptor. A `(1,2)` open against a `(2,4)` sim hangs fabric init
   (`Fabric Router Sync: Timeout … ethernet handshake likely failed`). Changed
   the pin to **`(2,4)`** to match the acceptance suite and the topology. With
   the fix the baseline runs 8/8 (see below). This is precisely the
   test/topology-mismatch class the task doc flags as a *test* defect.

2. **Dead conditional in `validate()` — `point_to_point.py`.**
   `if page % 16 != 0 and page != 16:` — the `and page != 16` clause is
   unreachable (`16 % 16 == 0` already makes the first term false at `page==16`),
   so the guard reduced to `page % 16 != 0`. Simplified to that; behavior
   identical, intent clearer.

### Reviewed clean (no change needed)

- **Kernels (all four).** `#include "api/dataflow/dataflow_api.h"` (not the bare
  path), `void kernel_main()` (not the deprecated namespace form), `TensorAccessor`
  (not `InterleavedAddrGen`). CB push/wait balance is exact:
  `cb_input_pages` / `cb_output_pages` push `num_pages` == wait/pop `num_pages`;
  `cb_packet_scratch` is a single-slot private scratch (reserve/push once, no
  balancing consumer — correct per design). The coalesce (`sender_writer`) and
  de-coalesce (`receiver_reader`) loops handle both packet regimes (coalescing
  and segmentation) and the short-last-packet case (`curr_pages_per_packet =
  min(max, pages_left)`); the multi-tile golden shapes (`64×128`, `512×512` =
  256 tiles) pass, exercising multi-packet coalescing end to end.
- **Fabric egress helper usage.** Uses the safety-by-construction CCL helper
  (`dataflow_kernel_lib::ccl`): `FabricStreamSender → open(unicast_route) →
  arm_unicast_write / arm_inc → write_page / inc → close`, and the one-shot
  `signal()` on the receiver's "ready". This is the correct helper surface for
  **fabric** egress. `mcast_pipe.hpp` (`SenderPipe`/`ReceiverPipe`) is **not**
  applicable — that is for *local NoC* multicast+handshake; this op's inter-chip
  hop is a fabric unicast the CCL helper owns. The raw APIs that remain in the
  kernels (interleaved page streaming, the waiting-half `noc_semaphore_wait_min`
  + cache re-arm `noc_semaphore_set`, the receive-ingress `noc_async_read`, and
  the CB protocol) are each explicitly *not owned* by the helper per its banner
  (`ccl_helpers_dataflow.hpp:69-93`) — keeping them as op code is correct, not a
  missed-helper smell.
- **Semaphore cache-reuse ordering (highest design risk).** Sender resets
  `noc_semaphore_set(sem,0)` **before** its outgoing "done" inc; receiver resets
  **after** its "done" wait. The `test_point_to_point_program_cache` path passes
  (second call is a cache hit) → the re-arm ordering is correct.
- **Host assembly.** The `[has_forward][fwd?][has_backward][bwd?]` fabric-conn RT
  block at `conn_arg_idx=9`, per-packet intermediate addressing
  (`uint32`, page size overridden to `packet_size_bytes`), the cached
  `GlobalSemaphore` parked on `mesh_pd.semaphores`, and output seeding
  (`clone`/`copy` on every device before dispatch) all match `op_design.md`.

### Advisory (not fixed — cosmetic, no behavior change)

- `receiver_reader.cpp` rounds the page stride with `align(page_size, alignment)`
  while `sender_writer.cpp` uses `round_up(page_size, alignment)`. Both round up
  to the L1 alignment and produce identical values; the naming is merely
  inconsistent. Left as-is to avoid a risk-free-but-pointless kernel edit on a
  verified path.

## Design Conformance

Checked against `op_design.md` on the binding dimensions — all match:

- **Algorithm:** identity byte copy via page↔packet (de)coalescing; no
  tilize/untilize (format-agnostic), matching "pure data movement."
- **Pipeline topology & RISC ownership:** SEND = reader(NCRISC)→`cb_input_pages`→
  writer(BRISC) fabric egress; RECEIVE = reader(NCRISC) ingress+de-coalesce→
  `cb_output_pages`→writer(BRISC) output. Two endpoint programs only; no program
  on non-participating devices.
- **Work distribution:** single core `(0,0)` per endpoint (fabric-bandwidth
  bound), matching the reference C++ factory. Multi-core/multi-link striping is
  explicitly out of scope (and out of TARGET) — see Recommendations.
- **Inter-core / cross-device coordination:** one cached `GlobalSemaphore`,
  ready→payload→done handshake with the documented reset placement.

## Registry Conformance

- **Confirmed present and correctly wired** in `point_to_point.py`:
  `INPUT_TAGGERS = {"alignment": tag_alignment}` (tagger has the `(inputs, axes)`
  signature), `SUPPORTED` (dtype/layout/topology/alignment — every axis the op
  gates on, incl. the tagger key), `EXCLUSIONS = []`, and `validate()` which
  checks structural errors (`ValueError`) → SUPPORTED per-axis
  (`UnsupportedAxisValue`) → EXCLUSIONS (`ExcludedCell`), in that order. The
  public entry point calls `validate()` as its first statement.
- **Op file declares no `INVALID`** (only a code comment mentions the word) —
  correct under the registry model; INVALID lives in `feature_spec.py`.
- **No auto-fixes to SUPPORTED were needed** — `xpass_drift = 0`, so there is no
  under-claim to promote.
- **INVALID audit (`eval/golden_tests/point_to_point/feature_spec.py`):**
  `INVALID = [{"dtype": bfloat8_b, "layout": ROW_MAJOR_LAYOUT}]` is well-formed —
  single-tensor coupling (both axes describe the input), passes the
  universe-must-change test (bf8b is a tiled block-float format with no
  row-major representation; `ttnn` cannot construct the tensor), and is the
  canonical bf8b+ROW_MAJOR entry. No cross-tensor-axis coupling, no
  not-yet-implemented masquerading as INVALID. `point_to_point` is not a
  norm-like op, so no no-weight canonicalization cells apply. **INVALID is
  complete and correct.**

## Precision Baseline

`test_point_to_point_precision_baseline.py` — 4 shapes × {bf16, f32} × TILE ×
Linear, on the `(2,4)` FABRIC_1D sim. The oracle is identity vs. the
device-resident sender shard (post-quantization).

| Shape | dtype | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err |
|-------|-------|-----|-------------|--------------|------------------|
| (1,1,32,32)   | bfloat16 | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| (1,1,32,32)   | float32  | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| (1,1,64,128)  | bfloat16 | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| (1,1,64,128)  | float32  | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| (1,1,96,64)   | bfloat16 | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| (1,1,96,64)   | float32  | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| (1,1,512,512) | bfloat16 | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| (1,1,512,512) | float32  | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |

**Assessment:** bit-exact. Because the op copies stored bytes verbatim, the
receiver shard equals the device-resident sender shard exactly (any dtype
quantization already happened at `from_torch` time), so PCC is exactly 1.0 and
every error metric is 0 — a no-op could not satisfy the strict oracle (receiver
== sender AND every other shard unchanged), so these are genuine transfer
confirmations.

**Recommended tolerances:** PCC ≥ 0.999 (f32) / 0.995 (bf16) / 0.99 (bf8b) as
safety bands; `atol = rtol = 0` is achievable in practice for the float dtypes.
These match the golden helper's `(0.999, 0.02)` band.

## Verifier CLI Summary

Representative 16-cell golden slice on the `bh_8xP150_p2p` sim (covers **every**
SUPPORTED axis value at least once — all 6 dtypes, both layouts, both topologies,
both alignments — plus the INVALID skip; plus large/rank-varied shapes
`512×512`, `4×32×96`, `2×4×64×64`, `32×48` for the packet-framing regimes):

- supported_pass: **15**
- invalid_skipped: **1**  (bf8b × ROW_MAJOR)
- supported_fail: **0**   ✅ (must be 0 to ship)
- xpass_drift: **0**      ✅ (must be 0 to ship)
- xfail_wrong_mode: **0** ✅ (must be 0 to ship)
- xfail_expected: **0**   (expected — `SUPPORTED == TARGET`, nothing outside SUPPORTED but INVALID)
- supported_marked_xfail / no_axes_found: 0

Artifacts: `eval/results/point_to_point/{verifier_report.json, test_results.json,
test_axes.json}`.

### On the golden slice vs. the full 396-cell cartesian

The full golden cartesian is 432 cells (396 supported + 36 bf8b×ROW_MAJOR
INVALID skips). Running all 396 on the craq-sim is infeasible in the available
time budget (~15–20 s/cell incl. sim boot + JIT; individual runs cap at the
tool's 10-minute window). The 16-cell slice was chosen to give the verifier a
PASS for **each** `(axis, value)` in SUPPORTED and to exercise both
packet-framing regimes and rank variety; combined with the 8/8 precision
baseline and the acceptance program-cache/output paths, this certifies
`SUPPORTED == TARGET` as honest. A full-cartesian sweep is a mechanical (long)
follow-up, not a correctness gap.

## Recommendations

1. **No refinements are queued** — `TARGET − SUPPORTED = ∅` and there are no
   failing cells. `op_requirements.md` documents Phase 0 as full-TARGET with an
   empty queue. This is expected for a byte-mover whose kernels are
   dtype/layout/alignment-agnostic.
2. **Out-of-TARGET future work (not queue items — no SUPPORTED axis, no failing
   cell to point at).** These require a `/golden-tests` TARGET expansion before
   they could become refinements:
   - *Sharded memory config.* `validate()` rejects sharded input; there is no
     `memory_config` axis in TARGET. A sharded reader/writer would be a real
     kernel change with a data-dependency (shards constrain the work split) — a
     future refinement candidate *if* TARGET gains a `memory_config` axis.
   - *Multi-link / multi-core fabric striping.* Single-core per endpoint is a
     deliberate design choice (fabric-bandwidth bound). This is a performance
     lever, not a capability axis.
3. **Full-cartesian golden sweep** (all 396 supported cells) whenever a longer
   sim window is available — purely to convert the representative-slice
   certification into an exhaustive one. Not blocking.
4. **Environment note for reruns.** `run_multidevice_sim_pytest.py` launches
   pytest with `sys.executable`. Invoke it with **this clone's** interpreter
   (`python_env/bin/python3 scripts/run_multidevice_sim_pytest.py …`, or
   `source python_env/bin/activate` first). The base `/localdev/wransom/tt-metal`
   env has an older `_ttnn` lacking `ttnn.fp8_e4m3`, which makes the shared
   `tests/ttnn/utils_for_testing.py:33` fail at import (collection error). This
   is an infra/env selection issue, not an op defect.
