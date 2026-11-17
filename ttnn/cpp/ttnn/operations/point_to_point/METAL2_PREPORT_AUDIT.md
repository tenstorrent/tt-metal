# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/point_to_point`

**Audited at `131ecf145ee`.** Earlier audits of this op were RED. Three prerequisite commits cleared the gates:

- `2e5f720ff76` — the four kernels use the Device 2.0 CB and NoC APIs.
- `a2c4bc6610c` — the intermediate tensor is a UINT8 ROW_MAJOR byte buffer with one packet per page, so its TensorAccessor page-size override equals its own page size in every layout.
- `131ecf145ee` — the two unary kernels take the transfer size from the `page_bytes` RTA instead of `TensorAccessor::get_aligned_page_size()`, so dropping the page-size override does not change any transfer size.

- **`PointToPointOp`** (`device/host/point_to_point_device_op.{hpp,cpp}`)
  - `SendReceive` — one factory with `create_workload_descriptor` (`point_to_point_device_op.cpp:199`). It builds one program per mesh coordinate from three program builders:
    - `send_program_factory` (`device/host/send_program_factory.cpp`) on `send_coord`
    - `receive_program_factory` (`device/host/receive_program_factory.cpp`) on `receive_coord`
    - `local_copy_program_factory` (`device/host/local_copy_program_factory.cpp`) on `send_coord` when `send_coord == receive_coord`
- Kernels (all in `device/kernels/dataflow/`, all referenced):
  - `reader_unary_interleaved_start_id_gen.cpp` — send reader, local-copy reader
  - `writer_send.cpp` — send writer (fabric sender)
  - `reader_receive.cpp` — receive reader (fabric sender for the "ready" signal, then packet unpacker)
  - `writer_unary_interleaved_start_id_gen.cpp` — receive writer, local-copy writer
- Kernel header `device/kernels/common.hpp` — included by `writer_send.cpp`, `reader_receive.cpp`, and four `ccl/reduce_to_root` kernels.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead .id form` — from `git log -1 … origin/akertesz/op-porting-recipe -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`. The recipe docs are not in this checkout; the line was taken from the recipe branch ref.

**Branch-specific overrides applied** (this branch is newer than the recipe; the launching session gave these):

1. GlobalSemaphore is supported in Metal 2.0 on this branch (commits `c423c053fd1`..`8bf9dd08871`). The Appendix A "GlobalSemaphore — UNSUPPORTED" entry is not applied.
2. Op-owned GlobalSemaphores are supported on `MeshWorkloadSpecFactoryConcept` through `MeshWorkloadArtifacts::global_semaphores` (`ttnn/api/ttnn/metal_v2_artifacts.hpp:45`; commits `dd35fbcb155`, `5084957ac33`).
3. Fabric connections are supported (commits `c933f0fc1c5`..`facc8020cd5`). They replace `tt::tt_fabric::append_fabric_connection_rt_args`.
4. A sanctioned kernel-side exception: the port may replace (a) the raw GlobalSemaphore address RTA with a `global_sem::` binding and (b) the `FabricConnectionManager::build_from_args` RTAs (incl. the forward/backward flags) with a `fabric::` binding. Other non-whitelisted kernel changes follow the recipe.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/point_to_point` |
| **Overall** | **GREEN** (TTNN readiness derived from code; sheet not read — see Questions) |
| **DOps / Factories** | `PointToPointOp` → `SendReceive` (send / receive / local-copy program builders) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes.** The only Device 1.0 sites left are the GlobalSemaphore and fabric-connection sites that override 4 replaces. |
| *Prereqs* — Cross-op escapes | Ok — `✓ clean` |
| *Feature Support* — overall | GREEN — no Appendix A entry fires after override 1 |
| *Feature Support* — Variadic-CTA | Ok — no variable-index CTA reads |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Not read — sheet not reachable.** Derived from code: no blocking signal. |
| *TTNN Readiness* — Concept (current) | `WorkloadDescriptor` (`point_to_point_device_op.hpp:55`) |
| *TTNN Readiness* — Execution Model | `MPMD` — send and receive programs have different kernels, CBs and CTAs |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No (`point_to_point_nanobind.cpp` binds only the op) |
| *TTNN Readiness* — Op-owned tensors | No. Op-owned **GlobalSemaphore**: yes (`point_to_point_device_op.cpp:232,247`) — supported by override 2 |
| *TTNN Readiness* — Target concept | `MeshWorkloadSpecFactoryConcept` (derived from code; sheet not read) |
| *Port work* — Offset base pointer | none |
| *Port work* — Tensor bindings (per binding) | Case 1 for all six bindings |
| *TTNN Readiness* — TensorParameter relaxation | Not read — sheet not reachable. Code shows no need (every `TensorSpec` is in the default hash). |
| *Port work* — TensorAccessor 3rd arg | drop (Class 2) at all 4 sites |
| *Port work* — CB endpoints | 1P+1C on 3 CBs; self-loop on 4 CBs; no multi-binding; no dead CB |

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`).

Every gate-bearing subject clears: Device 2.0 (with override 4), feature compatibility (with override 1), offset base pointers, and the TensorAccessor 3rd argument (all four sites are Class 2 drops). The TTNN factory concept gate is cleared **from the code only**: this session could not read the readiness sheet, so its `Is able to port?` cell is unconfirmed (Question 1). If the sheet says `no`, that cell overrides this verdict.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **Not read.** The readiness sheet needs the claude.ai Google Drive `download_file_content` tool, which this session does not have. Values come from the code:
  - `Concept` = `WorkloadDescriptor`: `SendReceive::create_workload_descriptor` (`point_to_point_device_op.hpp:55`, `.cpp:199`). No `create_descriptor`.
  - `Custom hash`: none. The default hash covers `attribute_values()` (`send_coord`, `receive_coord`, `topology`) and `tensor_args_t` (each `Tensor` hashes its storage and `TensorSpec`, `ttnn/api/ttnn/tensor/tensor.hpp:253`).
  - `get_dynamic_runtime_args`: none. `override_runtime_arguments`: none. `Pybind descriptor`: none.
  - `Op-owned tensors?`: no (`WorkloadDescriptor::buffers` not set). `WorkloadDescriptor::semaphores` holds one GlobalSemaphore (`.cpp:247`).
  - `Secretly SPMD Workload?`: no → `MPMD`.
  - `Smuggled pointer`: none for tensors (all `Buffer*` RTAs). The GlobalSemaphore address is a raw RTA (`send_program_factory.cpp:163`, `receive_program_factory.cpp:149`), stable for the cached workload; override 4 replaces it.
  - Factory set: one factory, `SendReceive`.
- **Device 2.0 (every kernel used):** **GREEN** (with override 4). All CB access goes through `CircularBuffer` objects and all tensor transfers through `Noc::async_read` / `async_write`. Sites left on purpose, all replaced by the port under override 4:

  | File | Line | Call | Override |
  |---|---|---|---|
  | `writer_send.cpp` | 71, 72, 75 | raw GlobalSemaphore pointer, `noc_semaphore_wait` / `noc_semaphore_set` | 4a |
  | `writer_send.cpp` | 116 | `get_noc_addr(receive_semaphore_addr)` for the fabric atomic-inc header | 4a |
  | `writer_send.cpp` | 42–43, 78–79 | `FabricConnectionManager::build_from_args`, direction pick | 4b |
  | `reader_receive.cpp` | 74, 75, 109 | raw GlobalSemaphore pointer, `noc_semaphore_wait` / `noc_semaphore_set` | 4a |
  | `reader_receive.cpp` | 52 | `get_noc_addr(sender_semaphore_addr)` for the fabric atomic-inc header | 4a |
  | `reader_receive.cpp` | 45–46, 59–60 | `FabricConnectionManager::build_from_args`, direction pick | 4b |

  No sanctioned free functions (`get_tile_size`, `get_local_cb_interface`) are in use. Packet-header pointers (`reinterpret_cast<volatile PACKET_HEADER_TYPE*>` of a CB write pointer) are fabric API usage, not a Device 2.0 holdover.
- **Feature compatibility:** GREEN — no gate fired.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no GCB type, no `.global_circular_buffer` field, no remote CB |
  | CBDescriptor `address_offset` (non-zero) | N/A | no CB sets `address_offset` |
  | GlobalSemaphore | **Supported on this branch (override 1)** — not RED | `create_global_semaphore` at `point_to_point_device_op.cpp:232`; `const GlobalSemaphore&` parameters at `point_to_point_device_op.hpp:114,119`; include at `.hpp:9`. Under the recipe as written this entry would be RED. |

  Not in Appendix A, so supported: `append_fabric_connection_rt_args` (`send_program_factory.cpp:168,173`; `receive_program_factory.cpp:154,159`) → fabric connections (override 3).
- **CB endpoints (GATE-free):** see Port-work summary. Nothing blocks.
- **Offset base pointers:** GREEN. Every address RTA is a clean base passed as `Buffer*`: send reader RTA0 input (`send_program_factory.cpp:143`); send writer RTA0 intermediate (`:179`; the `mesh_buffer().address()` at `:155` is a placeholder it replaces); receive reader RTA3 intermediate (`receive_program_factory.cpp:167`; placeholder at `:145`); receive writer RTA0 output (`:177`); local-copy reader/writer RTA0 input/output (`local_copy_program_factory.cpp:96,103`). The op is not in the offset-base-pointer triage tables.
- **TensorAccessor 3rd argument:** GREEN — every site is Class 2. The input is never sharded (`point_to_point_device_op.cpp:104`); the output and intermediate tensors use the input's memory config; so every accessor is interleaved. No kernel reads `get_aligned_page_size()` any more; every transfer size is explicit.

  | Site | Tensor | 3rd arg (host value) | Tensor page | Class | Transfer size |
  |---|---|---|---|---|---|
  | `reader_unary_interleaved_start_id_gen.cpp:26` | input (send, local copy) | `page_bytes` = `compute_page_size_bytes()` (`send_program_factory.cpp:146`, `local_copy_program_factory.cpp:99`) | same | **2** — correct magnitude, interleaved | `page_bytes` RTA (line 35) |
  | `writer_unary_interleaved_start_id_gen.cpp:63` | final output (receive, local copy) | `page_bytes` (`receive_program_factory.cpp:180`, `local_copy_program_factory.cpp:106`) | same | **2** | `page_bytes` RTA (line 71) |
  | `writer_send.cpp:59` | intermediate | `payload_size_bytes` = `packet_size_bytes` (`send_program_factory.cpp:160`) | `packet_size_bytes` (UINT8 ROW_MAJOR, last dim = `packet_size_bytes`, `point_to_point_device_op.cpp:172`) | **2** — equal | explicit (lines 95–96) |
  | `reader_receive.cpp:69` | intermediate | `packet_size_bytes` (`receive_program_factory.cpp:146`) | same as above | **2** — equal | explicit (lines 87–88) |

  The page stride does not change after the drop: today and after the port, the interleaved accessor rounds the page up to the bank alignment. The 3rd-arg triage lists `point_to_point` as Class 1; no site is Class 1, because every `TensorSpec` is in the default hash and no page size varies across cache hits. Do not set `dynamic_tensor_shape`.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding; all clean bases, all fed to a `TensorAccessor`):
  - send program: input → reader (`reader_unary…:26`) — **Case 1**; intermediate → `writer_send` (`:59`, used through `tt::tt_fabric::linear::to_noc_unicast_write`) — **Case 1**
  - receive program: intermediate → `reader_receive` (`:69`) — **Case 1**; final output → writer (`writer_unary…:63`) — **Case 1**
  - local-copy program: input → reader — **Case 1**; final output → writer — **Case 1**
- **TensorParameter relaxation:** none (code-derived; sheet not read).
- **TensorAccessor 3rd arg:** drop at all four sites listed above (Class 2). `page_bytes` stays as a named RTA in the two unary kernels, because it is still the transfer size.
- **CB endpoints** (send and receive programs run on one node, `CoreCoord{1,1}` work split; the local-copy program uses many nodes with the same census on each):

  | Program | CB | Touchers | Disposition |
  |---|---|---|---|
  | send | `c_0` sender CB (`send_program_factory.cpp:47`) | reader FIFO-produces, `writer_send` FIFO-consumes | 1P+1C (legal) |
  | send | `c_1` packet-header CB (`:62`) | `writer_send` only (reserve/push twice, uses write pointer) | **self-loop** |
  | send | `c_2` packet CB (`:77`) | `writer_send` only (reserve/push once, scratch via write pointer) | **self-loop** |
  | receive | `c_0` packet-header CB (`receive_program_factory.cpp:48`) | `reader_receive` only | **self-loop** |
  | receive | `c_1` packet CB (`:63`) | `reader_receive` only (reserve, NoC read destination, push at end) | **self-loop** |
  | receive | `c_2` receiver CB (`:75`) | `reader_receive` FIFO-produces, writer FIFO-consumes | 1P+1C (legal) |
  | local copy | `c_0` (`local_copy_program_factory.cpp:42`) | reader FIFO-produces, writer FIFO-consumes | 1P+1C (legal) |

  No hidden second writer: no kernel writes a CB it does not FIFO-produce. No dead CB.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes):** none.
- **Quasar debt (not a Gen1 blocker):** the four self-loop CBs are DM self-loops. The spec validator rejects a DM self-loop on Gen2, so the Quasar uplift must restructure them (see `dm_self_loop_dfbs.md`). This matters because the goal of this branch is to run P2P on Quasar.
- **Cross-op / shared kernels:** all four kernels are op-owned, and no other op factory binds them. But `tests/ttnn/unit_tests/operations/debug/test_generic_op.py:287,295,393,401` (`test_point_to_point`) builds all four through the V1 `ProgramDescriptor` path with positional RTAs. So the port creates `_metal2` forks of all four beside the originals (rung 2; no fork exists yet). Sunset list (not authorization to convert in place): `test_generic_op.py::test_point_to_point`.
- **`device/kernels/common.hpp`** is also included by four `ccl/reduce_to_root` kernels. The `_metal2` forks no longer need it after override 4b; do not edit or delete it.
- **RTA varargs:** none after override 4 (the fabric-connection tail read by `build_from_args` is replaced by a `fabric::` binding).
- **GlobalSemaphore, fabric connection, intermediate tensor, `extract_tensor_buffers_t`:** see the brief's Watch-for section.

## Team-only

- **Out-of-directory coupling & donor shape:** `✓ clean`.

  | Op kernel | Donor | Class | Shape |
  |---|---|---|---|
  | `writer_send.cpp`, `reader_receive.cpp` | `data_movement/common/kernels/common.hpp` (`tt_memmove(Noc, …)`, `round_up`) | cross-family | `Noc` by value, L1 addresses — ✓ |
  | `writer_send.cpp` | `ccl/common/kernels/moe_utils.hpp` → `minimal_ccl_common.hpp` (`perform_payload_send`) | cross-family | templated fabric sender + L1 address — ✓ |
  | `writer_send.cpp` | `tt_metal/fabric/hw/inc/linear/addrgen_api.h` (`to_noc_unicast_write`, 5-argument form with an explicit size) | `tt_metal/*` | `TensorAccessor` (Shape 1) — ✓ |
  | `writer_send.cpp`, `reader_receive.cpp` | `tt_metal/fabric/hw/inc/...` (fabric headers) | `tt_metal/*` | no concern |
  | `writer_send.cpp`, `reader_receive.cpp` | `../common.hpp` | own | `FabricConnectionManager&` — drops with override 4b |

  Borrowed kernel files: none (all op-owned).
- **Relaxation candidates:** none.
- **TTNN factory analysis:** target `MeshWorkloadSpecFactoryConcept`. Ported-from entry point: `SendReceive::create_workload_descriptor` (`point_to_point_device_op.cpp:199`). Per coordinate: send program on `send_coord`, receive program on `receive_coord`, or one local-copy program on `send_coord` when they are equal; no program on other coordinates. No `override_runtime_arguments`. Op-owned GlobalSemaphore (not on the local-copy path). No custom hash, no `get_dynamic_runtime_args`, no pybound `create_descriptor`.

## Misc anomalies  *(team-only, non-gating)*

- **Ring shortcut never fires.** `fabric_1d_routing_vector` returns `std::abs(hops)` (`point_to_point_device_op.cpp:53,61`). `fabric_1d_routing` then computes `ring_hops = line_hops + ((line_hops < 0 ? -1 : 1) * mesh_shape[dim])` (`.cpp:88`). `line_hops` is never negative, so `Topology::Ring` always takes the linear route.
- **Reference members in cached attributes.** `operation_attributes_t::receive_coord` and `send_coord` are `const MeshCoordinate&` (`point_to_point_device_op.hpp:22–23`); a copy of the attributes that outlives the call holds dangling references.
- **Unhashed attribute.** `_input_tensor_spec` ("put this in here to hash on tensor spec", `point_to_point_device_op.hpp:26–27`) is not in `attribute_values()` (`.hpp:30`). The spec is hashed through `tensor_args`, so this is dead, not wrong.
- **CB data formats from the intermediate dtype.** `receive_program_factory.cpp:45` takes `inter_dataformat` from the intermediate tensor, which is now UINT8. The packet CB and the receiver CB (`:69,82`) are therefore labelled UInt8, although the receiver CB holds output pages. Harmless (only DM kernels touch them).
- **Includes.** `reader_receive.cpp` includes `tt_metal/fabric/hw/inc/tt_fabric_api.h` twice (lines 10 and 13). `writer_send.cpp` uses `Noc` without including `api/dataflow/noc.h` (it arrives through `moe_utils.hpp`).

## Questions for the user

1. **Readiness sheet.** Please confirm the `point_to_point` row(s): `Is able to port?`, `TensorParameter relaxation`, `Known op issues`, `Porting Target`. This audit could not read them; the GREEN verdict assumes they agree with the code.

## Recipe notes

- **Class 2 drop is not always a no-op.** The 3rd-arg taxonomy judges only the page size's role in addressing. A kernel can also read the value back through `get_aligned_page_size()` (here it was a NoC transfer size, fixed by `131ecf145ee`). In legacy that returns the passed value; after the port it returns the bank-aligned page. The two classification questions should add: "does the kernel read `get_aligned_page_size()`, and does anything depend on that value?".
- **Skip rule.** An earlier pass of this audit ran the informational subjects while one small op-code gate was open, because the fix could not change what they describe. That saved a full pass here. Suggest: allow it when the pending fix is small and local.
- **Appendix A is stale for this branch** (GlobalSemaphore, override 1). The recipe has no guidance for op-owned GlobalSemaphores on `MeshWorkloadSpecFactoryConcept` or for fabric connections.
- **Shared-kernel census misses tests.** `port_patterns.md` counts only factory binders under `ttnn/cpp/ttnn/operations/`. Here the only other binder is a test (`test_generic_op.py`) that drives the kernels through the V1 path, and it still forces `_metal2` forks.
- **Unreachable readiness sheet.** The recipe covers a broken or silent sheet, but not one the session cannot fetch. This audit derived the primary columns from code and left the derived verdict as unconfirmed.
- **Provenance command** assumes the recipe docs are in the checkout; here the line came from the recipe branch ref.
