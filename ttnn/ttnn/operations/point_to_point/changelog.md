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
