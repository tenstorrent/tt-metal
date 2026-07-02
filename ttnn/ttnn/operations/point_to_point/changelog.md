# point_to_point — changelog

## 2026-06-24 — Initial implementation (Mode 1, fresh op)

Self-contained Python `generic_op` + `MeshProgramDescriptor` implementation of the
`point_to_point` fabric transfer, per `op_design.md`. The bound C++ `ttnn.point_to_point`
is NOT used.

### Files
- `point_to_point.py` — entry point, registry-model `SUPPORTED`/`EXCLUSIONS`/`validate()`,
  cached `GlobalSemaphore`, output seeding, dispatch.
- `point_to_point_program_descriptor.py` — two-program `MeshProgramDescriptor` assembly,
  packet framing (`ccl_packet_dims`), routing (`ccl_dm_route`), fabric-conn RT block.
- `kernels/point_to_point_sender_reader.cpp` (NCRISC) — input shard → `cb_input_pages`.
- `kernels/point_to_point_sender_writer.cpp` (BRISC) — coalesce → fabric `UnicastWriteChannel`
  → receiver intermediate; `AtomicIncChannel` "done".
- `kernels/point_to_point_receiver_reader.cpp` (NCRISC) — fabric "ready"/wait "done" → local
  read ingress → de-coalesce → `cb_output_pages`.
- `kernels/point_to_point_receiver_writer.cpp` (BRISC) — `cb_output_pages` → output shard.

### Design parity
Kernels are logical copies of the proven reference kernels
(`writer_send.cpp`, `reader_receive.cpp`, `*_unary_interleaved_start_id_gen.cpp`).
Host arg layouts (CT + RT, including the `[has_forward][fwd?][has_backward][bwd?]`
fabric-conn block at `conn_arg_idx=9`) mirror `send_program_factory.cpp` /
`receive_program_factory.cpp` exactly — verified line-by-line.

### Advisory deviations from op_design.md
- **Intermediate staging tensor** is a `uint32` `ROW_MAJOR` interleaved buffer of
  `[total_packets, packet_size_bytes/4]` (input's `buffer_type`) rather than "same
  TensorLayout as input". It is op-internal, addressed per-packet as raw bytes (page size
  overridden to `packet_size_bytes`), so its dtype/layout are cosmetic; `uint32` sidesteps
  `element_size`/`tt::datum_size` being undefined for `bfloat8_b` (`datum_size(Bfp8_b)`
  throws). The buffer holds exactly `total_packets * packet_size_bytes` — satisfies the
  binding "buffer ≥ total_packets * packet_size_bytes" + per-packet-addressing contract.
- **4 dataflow kernels** instead of the generic reader/compute/writer triple — a CCL op
  has two endpoint programs and no compute stage.

### Verification status
- Imports clean; fixed an eager-import circular dependency (`ttnn.operations` auto-walks
  submodules before `ttnn.Topology` is bound) via `from __future__ import annotations` +
  importing `Topology` from `ttnn._ttnn.operations.ccl`.
- Host-side logic verified on a single device for **all** acceptance-test
  (dtype × layout × shard_shape) combinations: per-device page metrics, `ccl_packet_dims`
  framing (incl. bf16 `bit_floor`), and `resolve_intermediate_spec` — packet size always a
  multiple of 4, intermediate page size == `packet_size_bytes`, intermediate page count ==
  `total_packets`, capacity ≥ required.
- `ttnn-static-analyzer`: **0 structural defects** across all four kernels — CB push/wait
  balance exact, coalesce/de-coalesce loops correct (multi-page + segmented + short-last-
  packet), semaphore handshake + cache-reuse reset placement correct, full reference parity.

### Known blocker — on-device acceptance run NOT yet executed
The acceptance test needs ≥2 mesh devices on a line with the fabric enabled.
- This host has **1 Blackhole** → cannot open a 2-device mesh directly.
- The multi-device simulator (`scripts/run_multidevice_sim_pytest.py`, topology
  `bh_8xP150_p2p`) **deadlocks during fabric bring-up**: `open_mesh_device` →
  `initialize_fabric_and_dispatch_fw` → `Fabric Router Sync: Timeout ... Device 4 ...
  4 core(s) stuck at STARTED` (ethernet handshake never completes), identical for both
  `FABRIC_1D` and `FABRIC_1D_RING`. This happens in **fixture setup, before the op runs** —
  it is a sim-data / sim-build issue (the `blackhole_8xP150_torus_x` model), not an op
  defect. Needs `TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS` raised AND a working sim-bh fabric
  bring-up (or real ≥2-device Blackhole hardware) to run.

Next step: run `tests/ttnn/unit_tests/operations/point_to_point/test_point_to_point.py` on a
working multi-device environment.

## 2026-06-24 — Phase 0 verification pass (verifier)

### What was done
Code review + registry-conformance hardening + analytical capability snapshot for the
Phase-0 implementation. On-device verification attempted but blocked (see below).

### SUPPORTED at Phase 0 (after the fix)
- **dtype** = `[bfloat16, float32, bfloat8_b, uint16, int32, uint32]`
- **layout** = `[TILE, ROW_MAJOR]`
- **topology** = `[Linear, Ring]`
- **alignment** = `[tile_aligned, non_tile_aligned]`  ← **added this pass**
- After the fix, `SUPPORTED == TARGET` on every axis → **0 axis-expansion refinements**.

### Issues encountered / fixed
- **Registry-conformance bug (fixed in `point_to_point.py`).** The op shipped
  `INPUT_TAGGERS = {}` and no `alignment` key in `SUPPORTED`, but `feature_spec.py`
  declares `alignment` as a TARGET axis and specifies a `tag_alignment(inputs, axes)`
  tagger. With an empty taggers dict the golden harness would have treated `alignment`
  as a spurious *finite* axis (double-counting each shape, label decoupled from the
  real shape). Added `tag_alignment` (last-two-dims-÷32 → `tile_aligned`),
  `INPUT_TAGGERS = {"alignment": tag_alignment}`, and
  `SUPPORTED["alignment"] = ["tile_aligned", "non_tile_aligned"]`. The op is a pure
  byte mover (no tilize/untilize), so it genuinely supports both alignments;
  `validate()` already iterates `INPUT_TAGGERS` generically, so no further change.
- **Kernels / descriptor / host assembly reviewed clean** (no changes): CB push==wait
  balance, both coalesce/de-coalesce regimes incl. short-last-packet, the
  `[has_forward][fwd?][has_backward][bwd?]` fabric arg block at `conn_arg_idx=9`
  (matches host), the handshake + cache-reuse reset ordering, and the
  helper-vs-raw-API split all verified against `ccl_helpers_dataflow.hpp` and the
  reference CCL kernels. INVALID audit (`feature_spec.py`) well-formed
  (single-tensor coupling, universe-must-change, canonical bf8b×ROW_MAJOR present).

### Accuracy achieved
- **Not measured on device** (blocked — see below). Analytical oracle: pure byte
  copy ⇒ receiver shard is bit-for-bit equal to the device-resident sender shard
  ⇒ PCC = 1.0, zero error for all dtypes. Precision baseline test written and
  collects (8 items) but could not execute.

### Golden suite at Phase 0
- **Analytical only** (no `test_golden.py` scaffold + sim blocker): 432 cells =
  396 expected-supported + 36 INVALID-skipped (bf8b×ROW_MAJOR), **0 xfail_expected**,
  loud drift categories 0 by construction. Saved to
  `eval/results/point_to_point/verifier_report.json` with a `_provenance` block
  flagging it as expected-categories, not observed results.

### On-device verification — BLOCKED (infra, not the op)
- `run_multidevice_sim_pytest.py --op point_to_point` (topology `bh_8xP150_p2p`,
  required) fails fabric router sync in `open_mesh_device` (fixture setup, before the
  op runs): *"Timeout ... Device 4 ... 4 core(s) stuck at STARTED ... ethernet
  handshake likely failed."* Reproduced with both the default 15000 ms and a raised
  **600000 ms** `TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS` — identical failure, so it is
  a genuine handshake failure in the `blackhole_8xP150_torus_x` sim model, not
  slowness and not an op defect. Verdict: `MULTIDEV_SIM_RESULT[bh_8xP150_p2p]: HANG`.
- **Mandatory follow-up:** re-run acceptance + precision + golden suites on real
  ≥2-device Blackhole hardware (or a repaired sim-bh fabric) to confirm the
  (currently review-only) SUPPORTED claims.

### Tests added
- `tests/ttnn/unit_tests/operations/point_to_point/test_point_to_point_precision_baseline.py`
  (PCC + max/mean abs + relative-RMS over 4 shapes × {bf16, f32} × Linear; multi-device
  fixtures matching the acceptance suite; ready to run under the sim/hardware runner).

### Artifacts written
- `verification_report.md`, `op_requirements.md` (empty refinement queue + verification
  debt), this changelog entry, and `eval/results/point_to_point/verifier_report.json`.

## 2026-07-02 — On-device (sim) acceptance run: 60/60 PASS — verification debt discharged

The Phase-0 `SUPPORTED` claims were previously **review-only** because the multi-device
run was blocked (this host has 1 Blackhole; the `bh_8xP150_p2p` sim hung at fabric
bring-up on the prior attempt). **That blocker is resolved.** The sim now brings up the
fabric on all 8 devices in ~3s — the earlier `Fabric Router Sync: Timeout` was a
transient / stale-sim-build condition, **not** an op defect and **not** a permanent sim
defect (consistent with commit `70eff48f` "p2p verified across all simulatable BH
topologies").

### What was done
Ran the **entire** immutable acceptance suite
(`tests/ttnn/unit_tests/operations/point_to_point/test_point_to_point.py`, 60 items) under
`scripts/run_multidevice_sim_pytest.py --topology bh_8xP150_p2p` (mesh `(2,4)`, FABRIC_1D
+ FABRIC_1D_RING, fast dispatch). Run in six dtype/layout + test-function batches to fit
the per-batch wall-clock backstop (the `mesh_device` fixture is function-scoped, so every
test re-opens the mesh + re-inits fabric ~20s). No source changes were required — the
existing implementation is correct as written.

### Result — **60 passed, 0 failed, 0 errors, 0 skipped**
| Batch | Cases | Result |
|-------|-------|--------|
| coordination (nonparticipating / output_tensor / program_cache / ring_wraparound) | 10 | PASS |
| bfloat8_b × TILE (5 shapes × {Linear,Ring}) | 10 | PASS |
| float32 × TILE | 10 | PASS |
| float32 × ROW_MAJOR | 10 | PASS |
| bfloat16 × TILE | 10 | PASS |
| bfloat16 × ROW_MAJOR | 10 | PASS |

Highest-risk paths flagged by the verifier are now **observed** green, not just reviewed:
- **bf8b / f32 / bf16 end-to-end** — the intermediate-as-`uint32` staging + `ccl_packet_dims`
  framing works for every element size (identity oracle: receiver shard == sender shard).
- **Ring topology** routing (`ccl_dm_route` short-way) — incl. the `(0,0)→(0,3)` single
  wraparound hop.
- **program-cache reuse** — the second call (cache hit) transfers correctly; the
  semaphore reset-ordering footgun (sender resets before its inc, receiver after its wait)
  survives cache reuse. Both `program-cache call 0` and `call 1` matched.
- **output_tensor write-into path** and **non-participating shards unchanged** across both
  topologies.

### Accuracy achieved
Identity oracle holds exactly as predicted: PCC at/above the per-dtype acceptance
thresholds (`f32 0.999`, `bf16 0.995`, `bf8b 0.99`) for every case — the transfer is a
bit-for-bit byte copy, adding no error over the device-resident sender shard.

### Golden suite
Not re-run here (no `test_golden.py` scaffold authored yet — an upstream `/golden-tests`
task). The analytical baseline (396 expected-supported + 36 INVALID-skipped, 0 xfail) is
now corroborated by the acceptance suite passing on every axis value it exercises.

### Issues encountered
None. No kernel, descriptor, or op-file change was needed — the acceptance suite passed
on the first successful multi-device run.

## 2026-07-02 — Verifier pass 2: mechanized golden + precision now OBSERVED (verifier CLI clean)

The Phase-0 golden claims were previously verified analytically + by the acceptance
suite only; the registry verifier CLI (`eval.verify_supported`) had never run on
observed results because the golden-test scaffold did not exist at pass 1. It now
exists (`test_golden.py` / `helpers.py` / `conftest.py`), so this pass ran the golden
suite through the verifier CLI on the `bh_8xP150_p2p` sim and produced a real
`verifier_report.json`.

### What was done
- **Golden suite → `eval.verify_supported`** on the multi-device sim (single-device
  `eval_test_runner.sh` is the wrong driver for a CCL op — it neither sets up the
  mesh nor enables the fabric). Driven via
  `scripts/run_multidevice_sim_pytest.py --topology bh_8xP150_p2p` with
  `-p eval.axes_plugin` + `PYTEST_AXES_JSON` + `--junitxml`, classified with
  `eval.classify_failures`, joined by `eval.verify_supported`. Two runs merged (dedup
  by nodeid) so one report spans every axis value:
  - 32×32 full cartesian: dtype×6 × layout×2 × topology×2 = 24 cells (22 pass + 2
    bf8b×RM invalid_skipped).
  - integer dtypes (uint16/int32/uint32) × {TILE,RM} × non-tile-aligned 48×64 ×
    Linear = adds non_tile_aligned coverage for the integer dtypes.
  - **Merged: 28 supported_pass + 2 invalid_skipped; 0 supported_fail / 0 xpass_drift
    / 0 xfail_wrong_mode / 0 xfail_expected.** Artifact:
    `eval/results/point_to_point/verifier_report.json`.
- **Precision baseline** re-pointed to the required mesh and run 8/8 green: PCC = 1.0,
  max/mean abs err = 0, rel RMS = 0 for all 4 shapes × {bf16, f32} (identity copy).
- **Acceptance `program_cache`** re-confirmed green on the edited tree (cache-reuse
  semaphore-reset footgun survives).

### Issues fixed
1. **`point_to_point.py` — dead boolean clause in `validate()` page-size check.**
   `if page % 16 != 0 and page != 16:` → `if page % 16 != 0:` (the `and page != 16`
   was unreachable; behaviour-identical). Confirmed by the 28-cell golden run passing
   through the edited `validate()`.
2. **`test_point_to_point_precision_baseline.py` — mesh/topology mismatch.** The test
   opened `mesh_device == (1, 2)`, which matches no p2p sim topology and would hang
   fabric init ("Fabric Router Sync: Timeout"). Changed to `MESH_SHAPE = (2, 4)` to
   match the required `bh_8xP150_p2p` topology (same shape as the immutable acceptance
   suite). After the fix: 8/8 pass on the sim.

### SUPPORTED (unchanged this pass) — now fully OBSERVED
- dtype = `[bfloat16, float32, bfloat8_b, uint16, int32, uint32]` — all 6 supported_pass.
- layout = `[TILE, ROW_MAJOR]` — both supported_pass.
- topology = `[Linear, Ring]` — both supported_pass.
- alignment = `[tile_aligned, non_tile_aligned]` — both supported_pass.
- `SUPPORTED == TARGET` on every axis → `TARGET − SUPPORTED = ∅` → **0 axis-expansion
  refinements**; refinement queue remains empty (documented in `op_requirements.md`
  against the three sanity gates).

### Accuracy achieved
Identity oracle exact: PCC = 1.0, zero error for every measured cell (bit-for-bit
byte copy adds no error over the device-resident sender shard).

### Coverage caveats (not defects, not refinements)
- The segmented-packet de-coalescing path (`page_segments > 1`) is implemented and
  reviewed but unreachable by any `feature_spec.INPUTS` shape (max RM page ≈ 2 KB ≪
  fabric max packet), so it is not exercised on the sim.
- Only the required `bh_8xP150_p2p` topology (mesh `(2,4)`) was run; the two optional
  p2p topologies (`bh_galaxy_4x2_p2p` `[4,2]`, `bh_bh6u_8x4_p2p` `[8,4]`) were not,
  because the golden/acceptance tests hardcode the `(2,4)` mesh (matching the required
  topology). `--op` would fan to those shapes and hang fabric init on the mismatch —
  a test/topology mismatch, not an op defect.

### Tests touched
- `test_point_to_point_precision_baseline.py` — mesh `(1,2)` → `(2,4)` (fix).

### Artifacts written
- `verification_report.md` (rewritten — observed, not analytical),
  `op_requirements.md` (updated — observed empty queue),
  `eval/results/point_to_point/verifier_report.json` (real verifier CLI output),
  this changelog entry.
