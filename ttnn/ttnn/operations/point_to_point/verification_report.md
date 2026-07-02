# Verification Report: point_to_point

`point_to_point` is a multi-device CCL op (pure fabric byte-copy, no compute kernel):
it builds a two-entry `ttnn.MeshProgramDescriptor` (a SEND program on the sender coord,
a RECEIVE program on the receiver coord) and dispatches via `ttnn.generic_op`. It was
verified on the deterministic multi-device craq-sim runner (`bh_8xP150_p2p` topology:
8-chip Blackhole `(2, 4)` torus_x mesh, `FABRIC_1D`) — **not** `run_safe_pytest.sh`,
which forces slow dispatch in sim mode and has no multichip/hang awareness.

## Code Review

**Fixes applied**

1. **Precision-baseline mesh shape (test bug → fixed).**
   `test_point_to_point_precision_baseline.py` opened a `(1, 2)` mesh, but the only
   sim topology that lists `point_to_point` (`bh_8xP150_p2p`) is fixed to `(2, 4)`.
   A shape mismatch hangs fabric init (`Fabric Router Sync: Timeout`), so the file
   would never have run. Changed the `mesh_device` parametrize to `(2, 4)`
   (sender `(0,0)` → receiver `(0,1)` are still adjacent on row 0). After the fix the
   baseline runs 8/8 green and bit-exact.

2. **Receiver kernel L1-pointer type (cleanup).**
   `point_to_point_receiver_reader.cpp` declared the packet scratch L1 address as
   `uint64_t packet_l1_addr` even though `get_write_ptr` returns a 32-bit L1 address
   that is narrowed back to `uint32_t` on every use (the local NoC-read destination and
   the `tt_memmove` page-offset base). Changed to `uint32_t` to match the sender's
   `packet_base_addr` and remove the pointless widen/narrow. Re-verified across
   single-tile → multi-tile shapes (7/7 green) — no behavioural change.

**Registry conformance (confirmed correct, no changes needed)**

- `INPUT_TAGGERS = {"alignment": tag_alignment}` — signature `(inputs, axes)` ✓ (reads
  the shard's last two dims; both `% 32 == 0` → `tile_aligned`).
- `SUPPORTED` declares every gated axis: `dtype`, `layout`, `topology`, `alignment` ✓.
- `EXCLUSIONS = []` (empty; the SUPPORTED rectangle has no interior holes) ✓.
- `validate()` runs structural gates (MeshDevice, sender≠receiver, in-mesh bounds,
  non-sharded, 16-B page alignment, output-spec match) first, then the axis gate
  (SUPPORTED per-axis, then EXCLUSIONS), raising `UnsupportedAxisValue` / `ExcludedCell`
  (both `NotImplementedError` subclasses, so the xfail-strict gate fires correctly) ✓.
- Public `point_to_point()` calls `validate()` as its first statement ✓.
- Op file does **not** declare `INVALID` ✓ (it is sourced from `feature_spec.py`).

**Design conformance (matches `op_design.md`)**

- Algorithm: identity byte copy, no tilize/untilize, no compute thread ✓.
- Topology: two per-coord programs, single Tensix core `(0,0)` each; all other mesh
  devices run no program (their output shard is the seeded input copy) ✓.
- Communication: `FabricStreamSender` unicast egress + one op-internal cross-device
  `GlobalSemaphore`; the sender resets its cell **before** its outgoing `done` inc, the
  receiver resets **after** its wait (the cache-reuse re-arm footgun) ✓ — verified live
  by `test_point_to_point_program_cache` (a second program-cache-hit call stays correct).
- Helper usage: fabric egress uses `dataflow_kernel_lib::ccl::FabricStreamSender`
  (`.open`/`.arm_unicast_write`/`.arm_inc`/`.signal`/`.close`); page↔packet coalescing
  uses `tt_memmove`; packet framing uses the host `ccl_packet_dims` helper; addressing
  uses `TensorAccessor`. The raw `noc_semaphore_wait_min` / `noc_semaphore_set` /
  local `noc_async_read` calls are the documented **non-goals** of the fabric helper
  (banner `ccl_helpers_dataflow.hpp:69-93`), so they are correctly op-owned, not a
  missed-helper smell.
- Packet-buffer sizing is safe in both framing regimes: in the coalescing regime
  `ccl_packet_dims` returns `packet_size_bytes = aligned_page × pages_per_packet`
  exactly, so the sender's `packet_page_idx × aligned_page` write offsets never overrun
  the single reserved `packet_size_bytes` CB slot; in the segmentation regime
  `packet_page_idx` resets to 0 per segment (offset 0, size ≤ payload). Verified.

**Correctness spot-checks (all pass)**

- CB sync: `cb_send_pages` / `cb_recv_pages` push-count == wait-count (`num_pages`);
  `cb_send_packet` / `cb_recv_packet` are single-slot self-recycled scratch (reserve
  once, no cross-thread consumer — the fabric write / local read consume the L1 payload
  synchronously), so no push/wait imbalance and no hang.
- `TensorAccessor` (not the deprecated `InterleavedAddrGen`), `void kernel_main()`,
  and the `api/dataflow/dataflow_api.h` include path are all used correctly in all
  four kernels.
- `ttnn.Topology is ttnn._ttnn.operations.ccl.Topology` (confirmed) — the `topology`
  axis gate compares like enums at runtime.

**Advisory notes (no change made — see Recommendations)**

- `op_design.md` "Circular Buffers" describes `cb_send_packet` / `cb_recv_packet` as
  "waits/pops `total_packets`", but the shipped kernels use a simpler and equally
  correct single-reserve scratch (the writer/reader owns both ends). The **code is
  correct**; the design-doc line is a stale over-description, not a kernel bug.
- The reader/writer issue one `noc_async_read`/`noc_async_write` + barrier per page.
  For a single-core pure copy this is correct; batching the barriers is a perf micro-opt
  with no failing cell (see Recommendations).

## Registry Conformance

- Confirmed present and correctly wired in the op file: `INPUT_TAGGERS`, `SUPPORTED`,
  `EXCLUSIONS`, `validate()`. Op file does **not** declare `INVALID` (a feature_spec
  concept). No auto-fixes to `SUPPORTED` were required — the golden run showed zero
  `xpass_drift` (no under-claim).

- **INVALID audit** (`eval/golden_tests/point_to_point/feature_spec.py`): well-formed.
  `INVALID = [{"dtype": ttnn.bfloat8_b, "layout": ttnn.ROW_MAJOR_LAYOUT}]`.
  - Single-tensor coupling ✓ — both axes describe the input tensor.
  - Universe-must-change ✓ — bfloat8_b is a block-quantized tiled format with no
    row-major representation (a data-format-definition impossibility, not a kernel
    refinement). This is the canonical `bf8b + ROW_MAJOR` entry.
  - No cross-tensor-axis entry; not a norm-like op (no weight axes / no-weight
    canonicalization needed).
  - `sender_coord` / `receiver_coord` are correctly **not** modeled as cartesian axes
    (mesh-dependent, harness-pinned), and the 16-B page-size constraint is correctly a
    shape×dtype `validate()` gate (kept satisfiable by every INPUTS shard — last dim a
    multiple of 8), not an axis. Confirmed the actual INPUTS shards produce 16-B-aligned
    pages for bf16/f32 (and valid bf8b tiles), so no supported cell trips the gate.

## Precision Baseline

`test_point_to_point_precision_baseline.py`, `(2, 4)` mesh, `FABRIC_1D`, Linear,
sender `(0,0)` → receiver `(0,1)`. Oracle = the device-resident sender shard
(post-`from_torch` quantization). p2p is a pure byte copy, so the transfer is
**bit-exact** — every metric is identically zero.

| Shape          | dtype     | PCC       | Max Abs Err | Mean Abs Err | Rel RMS Err |
|----------------|-----------|-----------|-------------|--------------|-------------|
| (1,1,32,32)    | bfloat16  | 1.000000  | 0.000e+00   | 0.000e+00    | 0.000e+00   |
| (1,1,32,32)    | float32   | 1.000000  | 0.000e+00   | 0.000e+00    | 0.000e+00   |
| (1,1,64,128)   | bfloat16  | 1.000000  | 0.000e+00   | 0.000e+00    | 0.000e+00   |
| (1,1,64,128)   | float32   | 1.000000  | 0.000e+00   | 0.000e+00    | 0.000e+00   |
| (1,1,96,64)    | bfloat16  | 1.000000  | 0.000e+00   | 0.000e+00    | 0.000e+00   |
| (1,1,96,64)    | float32   | 1.000000  | 0.000e+00   | 0.000e+00    | 0.000e+00   |
| (1,1,512,512)  | bfloat16  | 1.000000  | 0.000e+00   | 0.000e+00    | 0.000e+00   |
| (1,1,512,512)  | float32   | 1.000000  | 0.000e+00   | 0.000e+00    | 0.000e+00   |

**Assessment**: identity transfer, bit-for-bit exact end-to-end for every shape/dtype
(as designed — the op moves stored bytes verbatim). No precision degradation to track.

**Recommended tolerances**: PCC ≥ 0.999 (bf16), ≥ 0.9999 (f32); rtol = atol = 0 is
achievable in practice, but the golden/acceptance suites keep an explicit safety band
(PCC 0.995 bf16 / 0.999 f32, rtol 0.02) to absorb any future non-identity framing.

## Extended Tests

- **Added** `test_point_to_point_multi_hop` (`test_point_to_point_extended.py`): the one
  correctness path the acceptance/golden/precision suites never exercise — a `num_hops > 1`
  unicast. Row-0 transfers `(0,0)→(0,2)` (2 hops) and `(0,0)→(0,3)` (3 hops) on the
  `(2,4)` torus_x mesh (the x direction the descriptor routes), bf16 + f32, TILE, Linear.
  4/4 green: receiver shard matches bit-for-bit, sender shard unchanged, and an
  intermediate device on the path is untouched (confirms pure unicast — no relay writes).
- Deliberately **not** expanded further: shapes (single-tile / multi-tile / non-square /
  multi-batch / non-aligned), dtypes, layouts, and both topologies are already covered by
  the acceptance + golden matrices; rank/batch/multi-core edges belong in refinements.

## Verifier CLI Summary

Golden run on the `bh_8xP150_p2p` topology. The full TARGET×INPUTS matrix is 432 cells;
because `mesh_device` is function-scoped and the 8-chip sim re-inits fabric per test
(~17 s/case amortized), a curated selection covering **every category** was run:
`1x1x32x32` (all 24 cells: 10 supported, 12 integer-xfail, 2 bf8b×RM invalid-skip) plus
non-tile-aligned `1x1x48x64` supported + xfail cells. 34 cells total.

| Category               | Count |
|------------------------|-------|
| supported_pass         | 15    |
| xfail_expected         | 16    |
| invalid_skipped        | 3     |
| **supported_fail**     | **0** |
| **xpass_drift**        | **0** |
| **xfail_wrong_mode**   | **0** |
| supported_marked_xfail | 0     |

All loud categories are 0 — SUPPORTED describes reality. (Full JSON:
`verifier_report.json`, copied here from the results dir since `generated/` is
gitignored.)

**xfail_expected breakdown** (every entry maps to the single dtype gap):
`Counter({INT32: 6, UINT32: 6, UINT16: 4})` across TILE/ROW_MAJOR × Linear/Ring ×
tile_aligned/non_tile_aligned. `TARGET − SUPPORTED` per axis:
`dtype = {uint16, int32, uint32}`; `layout = ∅`; `topology = ∅`; `alignment = ∅`.
All three missing dtype values are addressed by Refinement 1 in `op_requirements.md`.

## Recommendations

- **Refinement priority**: only one axis gap exists (integer dtypes) → a single
  refinement. Everything else in TARGET is already supported and green.
- **Multi-link / multi-core (perf, not a queue item)**: the op is single-core /
  single-link by design (correct and simplest for a pure copy). Distributing pages
  across links/cores is a throughput improvement with **no failing cell and no SUPPORTED
  axis** — it does not belong in the refinement queue. Revisit only if a perf budget is
  introduced.
- **Barrier-per-page (perf)**: `sender_reader` / `receiver_writer` barrier after each
  page. Issuing all reads/writes then a single batched barrier per CB burst would cut NoC
  round-trips, but again there is no failing cell — leave until perf is in scope.
- **op_design.md CB-sync line** slightly over-describes the packet scratch CBs
  ("waits/pops total_packets"); the shipped single-reserve scratch is correct. Harmless
  doc drift; worth a one-line correction next time the design doc is touched.
- **Test-runner reminder** for future refinements: launch
  `run_multidevice_sim_pytest.py` with **this clone's** venv (`source python_env/bin/activate`)
  — a bare `python3` picked up a different clone's stale `ttnn._ttnn` (missing `fp8_e4m3`),
  which fails collection in `tests/ttnn/utils_for_testing.py`. And target
  `--topology bh_8xP150_p2p` (not `--op point_to_point`) so the run does not fan out onto
  the optional `(4,2)`/`(8,4)` topologies, whose mesh shapes the tests do not match
  (would hang fabric init — a test/topology mismatch, not an op defect).
