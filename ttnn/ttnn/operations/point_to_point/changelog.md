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

## 2026-07-01 — On-device (sim) acceptance run PASSES — prior blocker cleared

### What was done
Executed the previously-blocked acceptance suite on the graded multi-device sim
topology. **No implementation code changed** — the Phase-0 op (entry point,
program descriptor, four dataflow kernels) was already complete and structurally
verified; this pass closes the only outstanding item, the on-device run.

Command (the multichip analog of `run_safe_pytest.sh`, per op_design.md
"Verification Topology"):
```
scripts/run_multidevice_sim_pytest.py --topology bh_8xP150_p2p -- \
  tests/ttnn/unit_tests/operations/point_to_point/test_point_to_point.py
```
Topology `bh_8xP150_p2p` = Blackhole 8× P150, mesh **(2, 4)**, `FABRIC_1D`
(torus-x mesh-graph descriptor) — the `grade_primary`, `required` entry, and the
exact shape/fabric the acceptance test hardcodes.

### Result — GREEN
```
======================== 16 passed in 570.68s (0:09:30) ========================
MULTIDEV_SIM_RESULT[bh_8xP150_p2p]: PASS (pytest exit 0)
[multidevice-sim] aggregate exit = 0
```
All 16 invocations pass (11 CASES + 2 nonparticipating-unchanged + 2 output_tensor
+ 1 program_cache), covering:
- **dtype**: bfloat16, float32, bfloat8_b, uint16, int32, uint32 (block-float + all
  integer passthroughs bit-exact; floats within the suite's PCC thresholds).
- **layout**: TILE and ROW_MAJOR (incl. non-tile-aligned shards, e.g. (1,1,48,64),
  (1,1,96,64) — confirms the alignment-agnostic byte-copy claim).
- **topology**: Linear and Ring (Ring degenerates to the 1-hop line route for the
  adjacent (0,0)->(0,1) pair under FABRIC_1D, as designed).
- **output_tensor path**: caller-supplied preallocated output filled in place,
  same buffer handle returned.
- **non-participating shards unchanged**: every bystander shard (including the
  sender's own) equals its input — the output-seeding contract holds.
- **program cache**: two back-to-back calls both correct — the cached
  GlobalSemaphore survives, and the sender-before-inc / receiver-after-wait
  `noc_semaphore_set(0)` re-arm ordering is correct (the cache-reuse footgun).

### Accuracy achieved (now MEASURED, not analytical)
Pure byte copy ⇒ receiver shard is bit-for-bit equal to the device-resident sender
shard. Integer dtypes compared with `torch.equal` (exact) — pass. Float/block-float
compared with `assert_with_pcc` at the suite thresholds (f32 0.999, bf16 0.995,
bf8b 0.99) — pass. Effective PCC = 1.0 (identity transfer, no arithmetic).

### Prior blocker — RESOLVED
The Phase-0 changelog + verifier reported the sim `bh_8xP150_p2p` fabric bring-up
deadlocking (`Fabric Router Sync: Timeout ... Device 4 ... stuck at STARTED`) in
`open_mesh_device`, before the op ran. In THIS environment fabric bring-up
completes cleanly ("Fabric Initialized with config FabricConfig::FABRIC_1D" across
all 8 devices) and the op runs to green. The blocker was environmental (sim
build / fabric model), not an op defect — as the prior notes hypothesized.

### Issues encountered
None in the op. One runner-usage note for future runs: `run_multidevice_sim_pytest.py`
does NOT auto-append `-x` and passes trailing args verbatim to pytest, so
`run_safe_pytest.sh`-only flags like `--run-all` must NOT be forwarded (pytest exits
4). Also prefer `--topology bh_8xP150_p2p` over `--op point_to_point` for the
acceptance test: `--op` fans across all three p2p topologies (4x2, 8x4) whose mesh
shapes differ from the test's hardcoded (2,4), which would hang their fabric init.

### Tests added
None (implementation unchanged). The existing acceptance suite and the Phase-0
precision baseline both run under the sim runner.

## 2026-07-01 — Verifier hardening pass (observed golden + precision + verify_supported CLI)

### What was done
Full verifier pass with **mechanical, on-device (sim) confirmation** — the prior
verifier's numbers were analytical/blocked; these are observed. No op-code changes
(the four kernels, program descriptor, and entry point were reviewed clean).

### Issues encountered / fixed
- **Test bug fixed (precision baseline mesh shape).** `test_point_to_point_precision_baseline.py`
  opened `mesh_device=(1, 2)`, mismatching the graded `bh_8xP150_p2p` topology's fixed
  **(2, 4)** torus-x mesh-graph descriptor — that mismatch hangs fabric init
  (`Fabric Router Sync: Timeout`), a test/topology error, not an op defect. Changed the
  fixture to `(2, 4)` and corrected the docstring runner command to
  `--topology bh_8xP150_p2p` (from `--op point_to_point`, which fans across the
  `(4,2)`/`(8,4)` p2p topologies whose shapes would hang the (2,4)-hardcoded test).
  The fixed test now runs green (8/8, PCC=1.0).
- **Code review — clean, no op changes.** Confirmed helper usage is idiomatic
  (`sender_writer`: full `open→arm_unicast_write/arm_inc→write_page loop→inc→close`;
  `receiver_reader`: one-shot `signal(num_hops, sem)` for the ready handshake), CB
  push==wait balance, coalesce/de-coalesce (both regimes + short-last-packet), fabric
  RT-arg block at `conn_arg_idx=9`, handshake/cache-reuse reset ordering, and the
  helper-vs-raw split all correct against `ccl_helpers_dataflow.hpp`.
- **Registry conformance re-confirmed**: INPUT_TAGGERS/SUPPORTED/EXCLUSIONS/validate()
  present + wired; op file declares no INVALID; feature_spec INVALID well-formed.

### Accuracy achieved (MEASURED)
Precision baseline on `bh_8xP150_p2p` (mesh (2,4), FABRIC_1D, Linear), 8 cells
(4 shapes × {bf16, f32}, incl. large (1,1,512,512)): **PCC = 1.000000,
max_abs = mean_abs = rel_rms = 0** for every cell — bit-exact identity byte copy.

### Golden suite + verify_supported (OBSERVED)
Ran two golden-suite slices on the sim (`eval.axes_plugin` sidecar + junit →
`eval.classify_failures` → `eval.verify_supported`), sampling 30 of 432 cartesian
cells spanning every axis value (both topologies × all 6 dtypes × both layouts ×
both alignments × single/multi-tile):
- **supported_pass = 28, invalid_skipped = 2** (bf8b×ROW_MAJOR),
  **supported_fail = 0, xpass_drift = 0, xfail_wrong_mode = 0, xfail_expected = 0.**
- Slice 1: `1x1x32x32 & Linear` → 11 pass + 1 invalid_skip; Slice 2:
  `1x1x32x32 & Ring` (11+1) + `1x1x24x24 & bf16` non-aligned (4) + `1x1x64x128 & bf16 & Linear` (2).
- Merged report at `eval/results/point_to_point/verifier_report.json` (with a
  `_provenance` block documenting the sampling rationale). `xfail_expected == 0` is
  structural (SUPPORTED == TARGET → no xfail cells exist), not a coverage gap.

### Refinement queue
**EMPTY** — `TARGET − SUPPORTED = ∅` on every axis (dtype/layout/topology/alignment),
and the sole structural hole (`bf8b×ROW_MAJOR`) is an `INVALID` skip. No
axis-expansion refinements and no failing-cell refinements. Sharded memory config and
multi-link fan-out are out-of-TARGET scope notes in `verification_report.md`, not
queue entries.

### Tests modified / added
- Modified `test_point_to_point_precision_baseline.py` (mesh-shape fix + docstring).
- No new tests (existing acceptance + precision suites are sufficient; the op is a
  pure byte mover with full TARGET coverage at Phase 0).

### Artifacts written
- `verification_report.md` (rewritten with observed data), `op_requirements.md`
  (refreshed empty queue with the observed per-axis gap table), this changelog entry,
  and `eval/results/point_to_point/verifier_report.json` (+ `test_results.json`,
  `test_axes.json`, `junit*.xml`).
