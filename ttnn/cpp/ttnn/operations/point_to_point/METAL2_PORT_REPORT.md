# Port Report — point_to_point

## Outcome

**PORTED** — `PointToPointOp::SendReceive` (the op's only factory) is on `MeshWorkloadSpecFactoryConcept`, with all three
program builders (send, receive, local copy) and their four kernels converted through `_metal2` forks.
Verification status: see [Verification](#verification).

## Provenance

- **Recipe docs (this port):** `c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead .id form`
  (from `git log -1 … origin/akertesz/op-porting-recipe -- docs/…/metal_2.0/`; the recipe is not in this checkout).
- **Audit docs (inherited):** `c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead .id form`
- **Ported at:** on top of `131ecf145ee`. Branch-specific overrides 1–4 from `METAL2_PORT_BRIEF.md` applied.

## Verification

Host: 4x Blackhole p300b; multi-device meshes cannot open here, so only the 1x1 path runs on hardware.

- **Build:** `./build_metal.sh -b RelWithDebInfo --build-tests -e --enable-ccache` — SUCCESS.
- **Legality checks forced:** `skip_validation = false` in all nine `bool skip_validation` sites of
  `tt_metal/impl/metal2_host_api/{program_spec/program_spec.cpp,program_run_args.cpp}`, with markers in `BuildProgramFromSpec`
  and `SetProgramRunArgs`. Both markers appear in the pytest log (32 marker lines). The scaffolding was reverted before the commit.
- **Ran, local-copy program (hardware, Watcher on, `TT_METAL_WATCHER=10`):**
  `tests/ttnn/unit_tests/operations/point_to_point/test_point_to_point_same_device.py` — **8 passed** (same 8 as on the
  pre-port HEAD), including the alias-then-distinct and distinct-then-alias cache-hit cases.
- **Ran, send and receive programs (no hardware):** a throw-away gtest (not committed) on the 2-chip mock Wormhole fabric
  fixture (`ProgramSpecFabricTestGen1`) built a `ProgramSpec` with the same kernels, DFBs, bindings, named args,
  GlobalSemaphore parameter and fabric connection as each factory, ran `MakeProgramFromSpec` (spec validation on), and
  JIT-compiled it. Both passed; `writer_send_metal2` (BRISC) and `reader_receive_metal2` (NCRISC) ELFs were produced.
  This checks spec legality and kernel compilation only — it does not run the host factories or move data.
- **Not run (CI only, need ≥2 devices):** `tests/ttnn/unit_tests/base_functionality/test_copy_point_to_point.py`,
  `tests/ttnn/unit_tests/operations/point_to_point/test_moe_p2p.py`, `tests/nightly/t3000/ccl/test_point_to_point.py`, and
  `tests/ttnn/unit_tests/operations/debug/test_generic_op.py::test_point_to_point` (V1 path over the unchanged legacy kernels).
  The send/receive factories' runtime behaviour (fabric connection RTAs, GlobalSemaphore CRTA, cache hits) is unverified until
  those run.
- **Self-audit:** no `Buffer*` / `address()` / `emplace_runtime_args` in the factories (4 files); no `TensorAccessorArgs`,
  positional args or `CircularBuffer` in the 4 forks; CB-name grep hits only the unchanged legacy originals; no `.md`
  citation in changed code; `TT_FATAL` / `TT_ASSERT` / `TT_THROW` counts unchanged over 18 `.cpp`/`.hpp` files; no
  `opt_level` (DM-only factories, legacy unset → O2 both sides); `hw_config` = reader/writer defaults, as legacy; no
  `tt_metal/` file in the port diff.

## TTNN ProgramFactory

### Concept realized
- `MeshWorkloadSpecFactoryConcept`, as the audit chose. `SendReceive::create_mesh_workload_artifacts`
  (`device/host/point_to_point_device_op.cpp:199`) replaces `create_workload_descriptor` and emits one `programs` entry per
  ported-from program, over the same single-coordinate ranges: send on `send_coord` and receive on `receive_coord`, or
  one local copy on `send_coord` when the two are equal.
- No `override_runtime_arguments` (none in the ported-from op); the adapter refreshes tensor bindings per range on a hit.
- The op-owned GlobalSemaphore is created and `Synchronize`d exactly as before, then held in
  `MeshWorkloadArtifacts::global_semaphores` (override 2). Both programs' run args reference that vector element.

### Device-op-class edits
- Pybind entry points removed: none.
- Custom `compute_program_hash`: none.
- `point_to_point_device_op.hpp`: the factory method declaration and its comment (`WorkloadDescriptor` →
  `MeshWorkloadArtifacts`), the three program-builder return types (`ProgramDescriptor` → `ProgramArtifacts`), and the
  includes (`program_descriptors.hpp`, `workload_descriptor.hpp` → `ttnn/metal_v2_artifacts.hpp`). These are the factory's
  own declarations; no `validate` / `compute_output_specs` / `create_output_tensors` / attribute change.

### Open items
- `extract_tensor_buffers_t<PointToPointOp::tensor_args_t>` (`point_to_point_device_op.hpp:147-156`) is now dead for this op:
  the Metal 2.0 adapter enumerates io tensors with `visit_object_of_type<ttnn::Tensor>` and never reads it. It is outside the
  factory, so the port leaves it; the op owner can delete it with its comment.
- Relaxation candidates: none.

## Handoff points

none — no capitulation, no boundary-rule violation, no kernel-lib gap. (The four DM self-loop DFBs are Quasar debt, listed
under Open items.)

## Successes

- **Rung 2 for lent kernels** ([Caution: Porting a shared kernel] (patterns catalog)): the
  brief's note that `test_generic_op.py::test_point_to_point` binds all four kernels through the V1 path prevented an
  in-place conversion that would have broken that test. Forks: `device/kernels/dataflow/*_metal2.cpp`.
- **Self-loop DFBs** ([Pattern: Sync-free and single-ended CBs → self-loop DFB] (patterns catalog)):
  the four one-toucher scratch CBs (packet header, packet, on both send and receive) map to PRODUCER+CONSUMER on one
  kernel with one accessor name; no kernel change beyond `CircularBuffer` → `DataflowBuffer`.
- **TensorAccessor 3rd argument** drops at all four sites with the transfer sizes kept as named RTAs, per the brief — the
  prereq commit `131ecf145ee` made this a pure syntax swap.
- **Override 4** mapped cleanly: `noc_semaphore_wait/set` on a raw pointer → `experimental::GlobalSemaphore::wait/set`
  (same `noc_semaphore_*` underneath, `semaphore_dm_impl.h:200,230`), and `FabricConnectionManager::build_from_args<…START_ONLY>`
  plus the direction pick → `WorkerToFabricEdmSender::build_from_binding` + `open_start()` at the same place, with
  `open_finish()` / `close()` at the old call sites.

## Friction

### Gaps
- **Recipe has no section for `MeshWorkloadArtifacts::global_semaphores`** (override 2). The rule that matters — run args
  must hold a reference to the *vector element*, and the vector must not grow after the reference is taken — is only in a
  comment in `mesh_device_operation_adapter.hpp` ("Moving the vector keeps its elements in place"). It is the same identity
  footgun as op-owned tensors and wants the same text in the recipe.
- **Recipe has no section for fabric connections / GlobalSemaphore bindings** (overrides 1, 3, 4). The brief carried
  everything needed. Worth adding: where `open_start()` goes when the legacy kernel used `BUILD_AND_OPEN_CONNECTION_START_ONLY`,
  and that the forward/backward flag RTAs disappear with no named replacement.
- **Comments whose subject the port deletes.** Rule 8 forbids comment deletion but the recipe has no explicit carve-out for
  comments that describe deleted plumbing. Removed here, all because their subject is gone: the "Third argument page_size …
  may be stale on program cache hits" note at the four accessor sites; "reusing the last arg for fabric setup, therefore
  index overlaps" in both fabric kernels; and in the factories the `Buffer*` / `BufferBinding` / `append_fabric_connection_rt_args`
  / "Push kernels onto desc.kernels" / "hardcoded CB index" notes.
- **Provenance command** assumes the recipe docs are in the checkout; here they came from the recipe branch ref.

### Confusion
- **Kernel fork JIT coverage.** The send and receive programs cannot run on a 1x1 mesh, and the recipe's verification step
  assumes the op's tests exercise every path. The only local check for those kernels was a throw-away mock-fabric gtest
  (not committed). A recipe note on "paths the local hardware cannot run" would help.

## Open items for downstream

- **Shared kernel touches** (all *lent*, rung 2 — created the fork, pointer comment added to each original):
  | legacy kernel | fork | remaining unmigrated consumers |
  |---|---|---|
  | `device/kernels/dataflow/reader_unary_interleaved_start_id_gen.cpp` | `…_metal2.cpp` (`tensor::src`, `dfb::in0`; RTAs `num_tiles`, `start_id`, `page_bytes`) | `tests/ttnn/unit_tests/operations/debug/test_generic_op.py::test_point_to_point` |
  | `device/kernels/dataflow/writer_unary_interleaved_start_id_gen.cpp` | `…_metal2.cpp` (`tensor::dst`, `dfb::out`; RTAs `num_tiles`, `start_id`, `page_bytes`) | same |
  | `device/kernels/dataflow/writer_send.cpp` | `writer_send_metal2.cpp` (`tensor::dst`, `dfb::sender/packet_header/packet`, `global_sem::receive_semaphore`, `fabric::connection`; CTA `alignment`) | same |
  | `device/kernels/dataflow/reader_receive.cpp` | `reader_receive_metal2.cpp` (`tensor::packet_buffer`, `dfb::packet_header/packet/receiver`, `global_sem::sender_semaphore`, `fabric::connection`; CTA `alignment`) | same |
- **`device/kernels/common.hpp`** (`connection_direction_collection`) is no longer used by this op's ported path; the legacy
  kernels and four `ccl/reduce_to_root` kernels still include it.
- **Quasar debt:** the four DM self-loop DFBs (send `packet_header`, `packet`; receive `packet_header`, `packet`) are
  rejected by the spec validator on Gen2. The Quasar uplift must restructure them.
- **RTA→CRTA candidates** (not converted; changes dispatch): every RTA of the send and receive kernels runs on a single node,
  and `page_bytes`, `dst_num_hops` / `sender_num_hops`, `page_size_bytes`, `payload_size_bytes` / `packet_size_bytes`,
  `max_pages_per_packet`, `page_segments` are the same on every node in the local-copy and fabric programs.
- **`has_fabric_connection` not checked** in the fabric forks: every node of the kernel has a target (as the legacy kernel
  always built a connection), and `build_from_binding` ASSERTs the precondition.
- **Inherited anomalies** (from the audit, unchanged by the port): ring shortcut never fires
  (`point_to_point_device_op.cpp:53,61,88`); reference members in cached attributes (`.hpp:22-23`); the receiver DFB is
  labelled UInt8 although it holds output pages (`receive_program_factory.cpp`, `inter_dataformat`); `reader_receive.cpp`
  includes `tt_fabric_api.h` twice (kept in the fork for minimal diff).
- **Test coverage:** the send / receive programs run only in CI on multi-device meshes (`test_copy_point_to_point.py`,
  `test_moe_p2p.py`, `tests/nightly/t3000/ccl/test_point_to_point.py`).
