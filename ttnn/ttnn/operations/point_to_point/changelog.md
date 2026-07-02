# point_to_point — changelog

## 2026-07-02 — Registry-model verification pass (golden suite + verify_supported + precision)

Second verification pass: executed the **registry golden suite** through the
`verify_supported` CLI and the **precision baseline** on the 8-device craq-sim
(`bh_8xP150_p2p`, pinned `(2,4)` + FABRIC_1D). The prior 2026-07-02 entry had run
only the acceptance suite; this pass adds the golden/verifier signal and the
first successful precision-baseline run, and lands two verification fixes.

### Fixes
- **`test_point_to_point_precision_baseline.py`: `(1,2)` → `(2,4)` mesh (real test
  bug).** The baseline pinned a `(1,2)` mesh against the `(2,4)` `bh_8xP150_p2p`
  topology; a `(1,2)` open against the fixed `(2,4)` mesh-graph descriptor hangs
  fabric init (`Fabric Router Sync: Timeout`). Fixed to `(2,4)` (matches the
  acceptance suite). Baseline now runs 8/8.
- **`point_to_point.py`: dead conditional in `validate()`.** `if page % 16 != 0
  and page != 16:` → `if page % 16 != 0:` (the `and page != 16` term was
  unreachable). Behavior identical; no functional change.
- **No kernel / descriptor / entry-point logic changed.**

### Results (observed on device)
- **Golden slice + `verify_supported`** — 16 cells covering every SUPPORTED
  `(axis, value)` once (all 6 dtypes incl. `uint16`/`int32`/`uint32`, both layouts,
  both topologies, both alignments) + the bf8b×ROW_MAJOR INVALID skip + large/
  rank-varied shapes (`512×512`, `4×32×96`, `2×4×64×64`, `32×48`):
  **supported_pass=15, invalid_skipped=1, supported_fail=0, xpass_drift=0,
  xfail_wrong_mode=0, xfail_expected=0** → verifier-clean.
- **Precision baseline** — **8/8**, PCC=1.000000, max/mean-abs=0, rel-RMS=0
  (bit-exact identity copy) across `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,96,64)`,
  `(1,1,512,512)` × {bf16, f32}.
- **Acceptance (targeted re-run)** — `program_cache` (cache-reuse semaphore
  re-arm) + `output_tensor` (bf16 TILE, f32 ROW_MAJOR) all pass.

### SUPPORTED vs TARGET
- `SUPPORTED == TARGET` on every axis → `TARGET − SUPPORTED = ∅` →
  **0 axis-expansion refinements** (queue remains empty). No drift: `xpass_drift=0`.

### Artifacts
- `eval/results/point_to_point/{verifier_report.json, test_results.json, test_axes.json}`
- Updated `verification_report.md` (observed counts + precision table) and
  `op_requirements.md` (Phase-0 baseline reflects observed golden counts).

### Environment note
The multi-device runner launches pytest with `sys.executable`. It must be this
clone's interpreter (`python_env/bin/python3 scripts/run_multidevice_sim_pytest.py …`
or `source python_env/bin/activate` first); the base `/localdev/wransom/tt-metal`
env has an older `_ttnn` lacking `ttnn.fp8_e4m3`, which makes the shared
`tests/ttnn/utils_for_testing.py` fail at import (collection error) — infra/env, not
the op.

## 2026-07-02 — On-device acceptance verification (Mode 1, resolves verification debt)

The Phase-0 implementation had **never run on device** — the prior host had a single
Blackhole and the multi-device sim's fabric bring-up reportedly deadlocked. This run
executed the full acceptance suite on the 8-device craq-sim, closing that gap.

### What was done
- Ran `tests/ttnn/unit_tests/operations/point_to_point/test_point_to_point.py` via
  `scripts/run_multidevice_sim_pytest.py --topology bh_8xP150_p2p` (the pinned
  `(2, 4)` mesh + `FABRIC_1D`, `required=True` for this op). Fabric now initializes
  cleanly on all 8 devices — the prior "Fabric Router Sync: Timeout" blocker is **not
  present in this environment**.
- **No op code was changed.** The committed kernels / descriptor / entry point were
  already correct by static analysis; this run only supplied the missing on-device signal.

### Result — 18 / 18 acceptance tests PASS
- `test_point_to_point` — **10/10**: every `(dtype × layout × topology)` cell:
  `{bfloat16, float32, bfloat8_b} × {TILE, ROW_MAJOR}` (bf8b TILE-only) × `{Linear, Ring}`.
- `test_point_to_point_shapes` — **5/5**: single-tile `(1,1,32,32)`, multi-tile
  `(1,1,64,128)`, non-square `(1,1,96,64)`, multi-batch `(2,1,32,64)`, and
  **non-tile-aligned** `(1,1,48,64)`.
- `test_point_to_point_output_tensor` — **2/2**: bf16 TILE and f32 ROW_MAJOR
  (preallocated-output path; same handle returned).
- `test_point_to_point_program_cache` — **1/1**: second call is a program-cache hit —
  confirms the `GlobalSemaphore` survives (created once) and the cache-reuse
  `noc_semaphore_set(sem, 0)` re-arm ordering is correct (the "green run 1, hang run 2"
  footgun the verifier flagged as highest-risk).

### Accuracy
- Identity byte copy → **PCC = 1.0 in practice** for every dtype (the tests gate at the
  per-dtype safety bands PCC ≥ 0.995 bf16 / 0.999 f32 / 0.99 bf8b, all satisfied). The
  oracle is strict (receiver shard == sender shard AND every other shard unchanged), so a
  no-op could not pass — these are genuine transfer confirmations, not vacuous passes.

### Paths de-risked (previously review-only, now observed)
- **bf8b / uint32-intermediate framing** (bf8b TILE Linear+Ring) — the
  `intermediate-as-uint32` design that sidesteps `element_size` for block-float. ✔
- **float32 + ROW_MAJOR** end-to-end (both regimes of the coalesce path). ✔
- **Ring routing** (`ccl_dm_route` short-way) vs Linear — both correct on FABRIC_1D. ✔
- **program-cache reuse** — the semaphore reset ordering. ✔
- **non-tile-aligned** `(1,1,48,64)` — confirms the pure-byte-mover claim (no tilize). ✔

### Environment note (for future runs)
`run_multidevice_sim_pytest.py` launches pytest with `sys.executable` (the python that
started it). If invoked with the **base** `/localdev/wransom/tt-metal` python (whose older
ttnn build lacks `ttnn.fp8_e4m3`), collection dies at import of the shared
`tests/ttnn/utils_for_testing.py:33`. Fix: `source python_env/bin/activate` in **this
clone** before invoking the runner, so it uses the clone's ttnn (which has `fp8_e4m3`).
This is an infra/env issue, not an op defect.

### Tests added
- None (the immutable acceptance suite is the spec). Breadcrumbs written to
  `agent_logs/ttnn-implementer_breadcrumbs.jsonl`.

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
