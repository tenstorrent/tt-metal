# Changelog: point_to_point

## Phase 0 — Core Implementation

- **Date**: 2026-08-19
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer →
  verifier). A self-contained Python CCL op on `ttnn.generic_op` + `ttnn.MeshProgramDescriptor` with
  four newly authored dataflow kernels under `kernels/` — sender reader/writer and receiver
  reader/writer. No compute kernel (the op performs no arithmetic). It does **not** re-export, import,
  call, wrap or dispatch to the bound C++ `ttnn.point_to_point`.
- **SUPPORTED at Phase 0**: dtype=[bfloat16, float32, bfloat8_b, uint16, int32, uint32],
  layout=[TILE, ROW_MAJOR], topology=[Linear, Ring], alignment=[tile_aligned, non_tile_aligned].
  EXCLUSIONS=[]. This is the **full TARGET** on every axis.
- **Accuracy achieved**: **bit-exact** — PCC=1.0000000, max_abs_err=0.0, mean_abs_err=0.0,
  rms_err=0.0, measured on 4 shapes × 3 dtypes (12 cases) via
  `test_point_to_point_precision_baseline.py`. Correct for a pure byte copy: no accumulation, no
  fidelity/dest-accumulate trade-off, so anything short of exact would be a bug.
- **Golden suite at Phase 0**: **396 / 396 cells passing** (+ 36 INVALID-skipped, 432 total) per
  `generated/p2p_verify/verifier_report.json`. All loud categories 0: supported_fail=0,
  xpass_drift=0, xfail_wrong_mode=0, supported_marked_xfail=0.
- **Verified on**: real silicon — 4-chip Blackhole QuietBox, mesh `(1, 4)`, `FabricConfig.FABRIC_1D`
  (topology `bh_quietbox_1x4_hw`), via
  `scripts/run_multidevice_sim_pytest.py --op point_to_point --runtime hardware`. Aggregate exit 0.
- **Tests added**: `test_point_to_point.py` (acceptance, 90), `test_point_to_point_debug.py`
  (regression guards, 28), `test_point_to_point_extended.py` (5),
  `test_point_to_point_precision_baseline.py` (12).

### Issues encountered

**1. NoC DRAM-read alignment (implementer, fixed).** On a `(1,1,32,48)` bfloat16 `ROW_MAJOR` shard the
NoC sanitizer reported *"tried to unicast read 96 bytes to local L1[0x01b360] from
DRAM[0x00593e80] (invalid address alignment in NOC transaction)"*. A CB's page size **is** its
per-slot address stride, and the NoC requires `(l1_addr & mask) == (dram_addr & mask)` with
Blackhole's 64 B DRAM read alignment, so the design's `round_up(page_size, l1_alignment) = 96` put
slot 1 at `base + 96` ≡ 32 (mod 64). Fixed with `_dram_slot_stride()` =
`round_up(page_size, max(dram_align, l1_align))` on the two CBs that are the local side of a DRAM
transfer. The **intra-packet** stride stays `round_up(page_size, l1_alignment)` because
`ccl_packet_dims` derives `packet_size` from it; the kernels never conflate the two. No kernel change
was needed. An advisory deviation from `op_design.md`'s CB sizing table, required for correctness.

**2. `ccl_packet_dims` sized the fabric payload with the channel-buffer size (verifier, fixed).**
`ttnn/cpp/ttnn/operations/ccl/common/host/ccl_helpers_dataflow_host.hpp` capped a packet at
`get_tt_fabric_channel_buffer_size_bytes()`, but a channel buffer is
`packet_header_size + max_payload_size` and the worker writes the payload at
`slot_base + sizeof(PACKET_HEADER_TYPE)` — so a packet sized at the channel-buffer size overruns the
slot by the header size (48 B on Blackhole: `max_payload=4352`, `header=48`, `channel_buffer=4400`).
The only in-kernel guard omits the header **and compiles out in Release**, so the overrun was silent.
Reachable from the graded cartesian: the golden cell `(1,1,56,88)` `uint16` `ROW_MAJOR` framed 25
× 176 B pages into a 4400 B packet, plus every non-`bfloat16` segmented (regime B) case.
`bfloat16` was accidentally safe because its `std::bit_floor` lands on 4096.
Fixed at the source (`get_tt_fabric_max_payload_size_bytes()`), which also fixes the bound C++
`ttnn.point_to_point` that shares the helper, and keeps this op consuming the mandated helper with no
reimplementation of the framing rule. Offending cells move to 4224 B / 4352 B; `bfloat16` and every
tile-page case are unchanged.

**3. GlobalSemaphore cache keyed by `id(mesh_device)` (verifier, fixed).** A module-level dict keyed
by `id()` outlives the device (leaking a closed device's L1 allocation for the process lifetime — the
golden run opens 396 meshes) and can hand a *new* `MeshDevice` the *previous* device's semaphore,
since CPython recycles `id()`s. The handle is now bound as an attribute on the mesh-device object, so
its lifetime is exactly the device's. Deliberate deviation from `op_design.md`'s prescribed
`_SEMAPHORE_CACHE[id(mesh_device)]`.

**4. Precision baseline opened the wrong mesh shape (verifier, fixed).** It requested `(1, 2)` on the
`(1, 4)` graded topology and ERRORed on all 8 cases with `Fabric Router Sync: Timeout ... Ethernet
handshake likely failed`. Repinned to `(1, 4)`.

**5. Refuted — the "multi-packet / multi-hop ethernet wedge".** Phase 0's implementer notes reported
that any transfer of more than one packet or over more than one hop wedged the ethernet cores so that
the next `open_mesh_device` failed in `assert_active_ethernet_cores_to_reset`, and therefore that
every acceptance case needed its own pytest process plus a board reset (a 90-process driver script).
**It does not reproduce.** Measured in single pytest processes with the stock function-scoped
`mesh_device` fixture (one mesh open/close per case): acceptance 90/90 in 61 s, golden 396/396 in
143 s, `test_hop_count_stress` (1/2/3 hops × 2 payloads × 20 reps) 120/120 — roughly 600 consecutive
fabric mesh open/close cycles with real multi-hop, multi-packet traffic and no wedge. Most likely the
wedge *was* the NoC-alignment fault of issue 1 (the sanitizer **halts** the offending core, which
wedges the ethernet), removed by the `_dram_slot_stride` fix. No special per-case process isolation is
needed.

### Other verifier changes

- `_same_row_or_column()` raised `IndexError` on a rank-1 `MeshCoordinate`; now guarded.
- Removed the dead `geom["aligned_page_size"]` entry from the program descriptor.
- `test_point_to_point_debug.py`: 267 → 28 cases. Deleted the C++-reference control (it dispatched the
  bound `ttnn.point_to_point` from the generated op's own test directory, against the generation
  mandate; its conclusion is preserved in the docstring), cut the stress repetitions from 20 → 2 and
  the fixture-cycle control from 16 → 3, retargeted the `id()`-reuse probe as the guard for issue 3,
  and **widened `test_packet_geometry_within_edm_slot` from 5 (dtype, layout) pairs to all 11** — the
  narrow sweep is why issue 2 went unnoticed, and the widened one is what caught it.
- Added `test_point_to_point_extended.py` closing the three real coverage gaps: regime-B packet
  framing (`page_segments > 1` — entirely unexecuted code before, in both the sender and the
  receiver), interleaved-**L1** memory config, and the caller-supplied `intermediate_tensor` path.
- Strengthened the precision oracle from a PCC floor to **bit-exactness** (a PCC threshold hides a
  single corrupted page on a large shard — exactly what the framing/page-stride logic can produce).
- `black` (repo config, line-length 120) on the op module and the test files this pass owns.

### Refinement queue

**Empty.** `TARGET − SUPPORTED` is ∅ on all four axes, `EXCLUSIONS` is `[]`, `xfail_expected` is 0,
and there are no failing cells in any non-trivial category — so no entry of the form "add X to
`SUPPORTED[axis]`" or "move these failing cells to passing" exists to file. Beyond-`TARGET`
directions (sharded I/O, multi-link/mux, multi-core split, a real `FABRIC_1D_RING` wraparound
topology) are recorded in `verification_report.md`; each needs `/golden-tests` to expand
`feature_spec.py`'s `TARGET` first. See `op_requirements.md`.
