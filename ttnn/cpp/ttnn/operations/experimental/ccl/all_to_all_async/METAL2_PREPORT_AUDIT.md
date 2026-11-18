# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async`

**Audited at `c116d1cba2f`.** No `[Prereq]` commit was necessary: every gate clears on the code as it is. The kernels
were moved to the Device 2.0 API by `66aaf87f980` (#47547) before this audit.

- **`AllToAllAsyncDeviceOperation`** (`device/all_to_all_async_device_operation.{hpp,cpp}`)
  - `AllToAllAsyncProgram` (`device/all_to_all_async_program_factory.{hpp,cpp}`) — one factory with a per-coordinate
    `create_descriptor(attrs, tensor_args, out, optional<MeshCoordinate>)` (`all_to_all_async_program_factory.cpp:149`)
    and `override_runtime_arguments(Program&, …)` (`:607`).
- Kernels (all in `device/kernels/`, all referenced, all op-owned):
  - `interleaved_all_to_all_reader.cpp` — sender reader: reads the input shards for the other devices into the sender CB.
  - `interleaved_all_to_all_writer.cpp` — sender writer: sends each shard over the fabric (forward or backward connection)
    into the intermediate buffer of the destination device, with chunked atomic increments of the destination receiver
    core's GlobalSemaphore.
  - `interleaved_all_to_all_receiver_reader.cpp` — receiver reader (one core per source device): local copy of its own
    shard from the input, or waits on the GlobalSemaphore and reads the remote shard from the intermediate buffer.
  - `interleaved_all_to_all_receiver_writer.cpp` — receiver writer: writes the receiver CB to the output tensor.
  - Census: `grep -rn "interleaved_all_to_all_" ttnn/ tests/ models/ tt_metal/` hits only this factory.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead .id form` — from
`git log -1 … origin/akertesz/op-porting-recipe -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`. The recipe
docs are not in this checkout; the line was taken from the recipe branch ref.

**Branch-specific overrides applied** (this branch is newer than the recipe; given by the launching session):

1. GlobalSemaphore is supported. The Appendix A "GlobalSemaphore — UNSUPPORTED" entry is not applied.
2. Op-owned GlobalSemaphores go in `MeshWorkloadArtifacts::global_semaphores`. **Not applicable:** this op owns none.
3. Fabric connections replace `append_fabric_connection_rt_args`.
4. Sanctioned kernel-side exception: (a) the raw GlobalSemaphore address RTAs (sender writer RTA 2, receiver reader
   RTA 2) and their raw-pointer uses become `global_sem::` bindings; (b) the fabric-connection RTAs read by
   `FabricConnectionManager::build_from_args` (incl. the forward/backward flags) become `fabric::` bindings.
5. Caller-supplied GlobalSemaphores: **applies.** `semaphore` is in `AllToAllAsyncParams`, left out of the custom hash,
   and re-applied by the legacy override. One `GlobalSemaphoreParameter`, bound by the sender writer and the receiver
   reader, refreshed by one run-args helper on the miss and the hit.
6. A binder node with no fabric target: **applies in the kernel** (the host builds each connection only when that
   neighbor exists, `.cpp:526-537`). On every config the op accepts both neighbors exist (Ring only, ring size ≥ 3), but
   the guards stay.
7. One node that talks to more than one destination: **applies.** The sender writer core opens a forward and a backward
   connection → two `FabricConnectionSpec`s, `forward` and `backward`, both bound by the sender writer.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async` |
| **Overall** | **GREEN** (TTNN readiness derived from code; sheet not read — see Questions) |
| **DOps / Factories** | `AllToAllAsyncDeviceOperation` → `AllToAllAsyncProgram` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes.** The only Device 1.0 sites left are the GlobalSemaphore and fabric-connection sites that override 4 replaces, and one `noc_async_read` on a precomposed NoC address that the Device 2.0 migration (#47547) kept on purpose. |
| *Prereqs* — Cross-op escapes | Ok — `✓ clean` (`minimal_ccl_common.hpp` helpers take a `WorkerToFabricEdmSender&`, a templated accessor and NoC addresses) |
| *Feature Support* — overall | GREEN — no Appendix A entry fires after override 1 |
| *Feature Support* — Variadic-CTA | Ok — no variable-index CTA reads |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Not read — sheet not reachable.** Derived from code: no blocking signal. |
| *TTNN Readiness* — Concept (current) | `descriptor (per-coord)` (`all_to_all_async_program_factory.hpp:16-20`) |
| *TTNN Readiness* — Execution Model | `SPMD (+ per-device args)` — every coordinate has the same kernels, CBs and core ranges; the CTA `my_chip_id` (ring index) of all four kernels, the per-coordinate RTAs and the fabric targets depend on the coordinate |
| *TTNN Readiness* — Custom hash | **Yes** (not a gate; port leaves it intact): `all_to_all_async_device_operation.cpp:246`. It leaves out `semaphore`. |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | **Yes** (not a gate; the porter translates it): `all_to_all_async_program_factory.cpp:607` |
| *TTNN Readiness* — Pybind `create_descriptor` | No (`all_to_all_async_nanobind.cpp` binds only the op) |
| *TTNN Readiness* — Op-owned tensors | No. Op-owned GlobalSemaphores: no (caller-supplied, in `operation_attributes_t`). The two persistent buffers are caller-supplied io tensors in `tensor_args_t`. |
| *TTNN Readiness* — Target concept | `MeshWorkloadSpecFactoryConcept` with the optional `override_runtime_arguments` (derived from code; sheet not read) |
| *Port work* — Offset base pointer | none |
| *Port work* — Tensor bindings (per binding) | Case 1 for all five used bindings; one dead binding (sender writer output) |
| *TTNN Readiness* — TensorParameter relaxation | Not read — sheet not reachable. Code shows no need (the custom hash keys on every `TensorSpec` in `tensor_args`). |
| *Port work* — TensorAccessor 3rd arg | none — no accessor passes a 3rd argument |
| *Port work* — CB endpoints | 1P+1C on the sender data CB and the receiver data CB; **self-loop** (DM) on the packet-header CB; no multi-binding; no dead CB |

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`).

Every gate-bearing subject clears: Device 2.0 (with override 4), feature compatibility (with override 1), offset base
pointers, and the TensorAccessor 3rd argument (no sites). The TTNN factory concept gate is cleared **from the code only**:
this session could not read the readiness sheet (Question 1). If the sheet says `no`, that cell overrides this verdict.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **Not read.** No tool in this session can fetch the readiness sheet. Values
  from the code:
  - `Concept` = `descriptor (per-coord)`: `AllToAllAsyncProgram::create_descriptor(…, const std::optional<MeshCoordinate>&)`
    (`all_to_all_async_program_factory.hpp:16-20`, `.cpp:149`). It `TT_FATAL`s without a coordinate (`.cpp:155`).
  - `Custom hash`: yes, `AllToAllAsyncDeviceOperation::compute_program_hash` (`all_to_all_async_device_operation.cpp:246`).
    It hashes `in_dim`, `out_dim`, `num_links`, `ring_size`, `output_mem_config`, `topology`, the sub-device worker core
    range set and `tensor_args` (input, intermediate and output, with their `TensorSpec`s). It leaves out `semaphore`,
    which `override_runtime_arguments` re-applies on every hit.
  - `get_dynamic_runtime_args`: none. `override_runtime_arguments`: yes (`.cpp:607`). `Pybind descriptor`: none.
  - `Op-owned tensors?`: no (`descriptor` concept; the persistent intermediate and output buffers come from the caller).
  - `Secretly SPMD Workload?`: no — `ring_index` is CTA 0 of all four kernels (`.cpp:319, 343, 374, 394`). Why
    mesh-shaped: per-coordinate CTAs, RTAs and fabric targets.
  - `Smuggled pointer`: none for tensors (all are `Buffer*` RTAs). The GlobalSemaphore address is a raw RTA (sender writer
    `.cpp:506`, receiver reader `.cpp:570`), marked `smuggled-rta-ok` and refreshed by the override; override 4a replaces it.
  - Factory set: one factory.
- **Device 2.0 (every kernel used):** **GREEN** (with override 4). All CB access goes through `CircularBuffer` objects and
  the tensor transfers through `Noc::async_read` / `async_write`, `async_read_barrier`, `async_writes_flushed`,
  `async_write_barrier`. Sites left on purpose:

  | File | Line | Call | Disposition |
  |---|---|---|---|
  | `interleaved_all_to_all_writer.cpp` | 52, 102-103 | raw GlobalSemaphore address RTA, `safe_get_noc_addr(x, y, <sem addr>, 0)` for the atomic-inc headers | override 4a |
  | `interleaved_all_to_all_writer.cpp` | 63-66, 97-99, 125-126, 194-196 | `FabricConnectionManager::build_from_args<BUILD_AND_OPEN_CONNECTION_START_ONLY>`, `open_finish`, `get_forward/backward_connection`, `close` | override 4b |
  | `interleaved_all_to_all_receiver_reader.cpp` | 34, 91-92, 108, 130 | raw GlobalSemaphore address RTA, `noc_semaphore_wait_min` / `noc_semaphore_set` on it | override 4a |
  | `interleaved_all_to_all_receiver_reader.cpp` | 113-117 | `noc_async_read(<accessor>.get_noc_addr(first_id, 0, 0), l1, 2 pages)` | **kept by the Device 2.0 migration** (`// Device 2.0 migration: legacy primitive retained: precomposed uint64_t NoC address`, #47547). Not a holdover: the port leaves it as is. |

  No sanctioned free function is in use. Packet-header pointers taken from the header CB, `fabric_set_unicast_route`,
  `tt::tt_fabric::linear::to_noc_unicast_write` and the `minimal_ccl_common.hpp` helpers (`perform_atomic_fabric_write`,
  `perform_payload_send`) are fabric API usage, not Device 2.0 holdovers.
- **Feature compatibility:** GREEN — no gate fired.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no GCB type, no `.global_circular_buffer` field, no remote CB |
  | CBDescriptor `address_offset` (non-zero) | N/A | no CB sets `address_offset` |
  | GlobalSemaphore | **Supported on this branch (override 1)** — not RED | `const ttnn::GlobalSemaphore semaphore` in `AllToAllAsyncParams` (`all_to_all_async_device_operation_types.hpp:24`); include at `:10`; address at `all_to_all_async_program_factory.cpp:506, 570, 619`. Under the recipe as written this entry would be RED. |

  Not in Appendix A, so supported: `append_fabric_connection_rt_args` (`.cpp:529, 535`) → fabric connections (override 3),
  one per neighbor direction on the sender core (override 7).
- **CB endpoints (GATE-free):** see Port-work summary. Nothing blocks.
- **Offset base pointers:** GREEN. Every address RTA is a clean base passed as `Buffer*`: sender reader RTA0 input
  (`.cpp:478`); sender writer RTA0 intermediate, RTA1 output (`.cpp:540-541`; the `0u` at `:503-504` are placeholders the
  `Buffer*` replaces); receiver reader RTA0 intermediate, RTA1 input (`:566-567`); receiver writer RTA0 output (`:588`).
  The op is not in the offset-base-pointer triage tables. The `first_id` page arithmetic is kernel-side page indexing, not a
  host fold.
- **TensorAccessor 3rd argument:** N/A — no accessor in the op passes a 3rd argument (`reader.cpp:242`,
  `writer.cpp:93, 95`, `receiver_reader.cpp:52, 54`, `receiver_writer.cpp:177` are two-argument).

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding; clean bases, all fed to a `TensorAccessor`):
  - input → sender reader (`reader.cpp:242`, RTA0) — **Case 1**
  - intermediate → sender writer (`writer.cpp:93`, RTA0; fabric unicast / fused atomic writes by page id) — **Case 1**
  - input → receiver reader (`receiver_reader.cpp:52`, RTA1; local-shard copy) — **Case 1**
  - intermediate → receiver reader (`receiver_reader.cpp:54`, RTA0; `get_noc_addr(first_id, 0, 0)`) — **Case 1**
  - output → receiver writer (`receiver_writer.cpp:177`, RTA0) — **Case 1**
  - Dead: output → sender writer (`writer.cpp:95`, RTA1). The accessor is built and never used. It drops with no binding.
- **TensorParameter relaxation:** none (code-derived; sheet not read).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:**

  | CB | Node set | Touchers on a node | Disposition |
  |---|---|---|---|
  | `c_in0` sender data (`.cpp:75-83`) | sender core | sender reader FIFO-produces; sender writer FIFO-consumes (`wait_front` / `get_read_ptr` / `pop_front`) | 1P+1C (legal) |
  | `c_in1` packet header (`.cpp:86-94`) | sender core | sender writer only (`reserve_back` / `get_write_ptr` / `push_back` ×3, raw header writes) | **self-loop** (DM) |
  | `c_in0` receiver data (`.cpp:106-114`) | receiver cores | receiver reader FIFO-produces (`reserve_back` / `get_write_ptr` / `push_back`); receiver writer FIFO-consumes | 1P+1C (legal) |

  The two `c_in0` CBs are separate CBs on disjoint core sets (same index). They become two DFBs. No hidden second writer
  (the remote sender writes land in the intermediate DRAM tensor, not in a CB). No dead CB.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes):** none.
- **Quasar debt (not a Gen1 blocker):** the packet-header CB is a DM self-loop (`dm_self_loop_dfbs.md`).
- **Cross-op / shared kernels:** none. All four kernels are op-owned → convert in place. No `_metal2` fork.
- **RTA varargs:** none after override 4b (the fabric-connection tail read through `arg_idx` is replaced by two
  `fabric::` bindings).
- **Multi-destination fabric node (override 7):** the sender core opens a forward and a backward connection, link index =
  `link`. Two `FabricConnectionSpec`s (`forward`, `backward`), both bound by the sender writer.
- **GlobalSemaphore bound by two kernels.** The sender writer needs only its address (for the remote atomic increments to
  the receiver core of the destination device); the receiver reader waits on it and resets it. The caller's semaphore must
  span the sender core and every receiver core (the tests create it over the whole compute grid,
  `tests/nightly/t3000/ccl/test_all_to_all.py:146-160`).
- **Receiver kernels shadow a CTA with an RTA of the same name.** Both receiver kernels declare a file-scope `constexpr
  num_pages_per_packet` (CTA 2) and then a local `num_pages_per_packet` (RTA) inside `kernel_main` with the same value
  (`receiver_reader.cpp:18, 45`, `receiver_writer.cpp:149, 170`). Named args must be unique per kernel: give the two
  different arg names and keep both variables.

## Team-only

- **Out-of-directory coupling & donor shape:** `✓ clean`.

  | Op kernel | Donor | Class | Shape |
  |---|---|---|---|
  | sender writer | `ccl/common/kernels/minimal_ccl_common.hpp` — `perform_atomic_fabric_write`, `perform_payload_send` | cross-family (ccl) | templated accessor ✓, `WorkerToFabricEdmSender&` ✓, `uint64_t` NoC semaphore address ⚠ — workable on this branch: the kernel builds it from the binding as `safe_get_noc_addr(x, y, gs.address(), 0)`, as `slice_reshard_async` does |
  | sender writer | `tt_metal/fabric/hw/inc/…` (`fabric_connection_manager.hpp`, `noc_addr.h`, `tt_fabric_api.h`) | `tt_metal/*` | no concern |

  Borrowed kernel files: none. Lent kernel files: none.
- **Relaxation candidates:** none.
- **TTNN factory analysis:** target `MeshWorkloadSpecFactoryConcept` with the optional `override_runtime_arguments`.
  Ported-from entry point: `create_descriptor(…, coord)` (`all_to_all_async_program_factory.cpp:149`), one program per
  coordinate. Per coordinate: ring index, forward/backward neighbors, the CTA `my_chip_id`, `num_targets_forward/backward`,
  the sender RTAs (`out_row_start`, `out_col_start`, shard sizes, receiver core) and the fabric targets. Ported-from
  override (`.cpp:607`): writes the input address (sender reader RTA0, receiver reader RTA1), the intermediate address
  (sender writer RTA0, receiver reader RTA0), the output address (sender writer RTA1, receiver writer RTA0) and the
  semaphore address (sender writer RTA2, receiver reader RTA2), on every core, for every program. No op-owned tensors or
  semaphores.

## Misc anomalies  *(team-only, non-gating)*

- **Dead RTAs.** Sender writer `out_row_tiles`, `out_col_tiles`, `wait_output_semaphore`, `reset_global_semaphore`
  (`writer.cpp:53-54, 59-60`); sender reader `in_row_tiles`, `out_row_offset`, `out_col_offset` (`reader.cpp:232, 238-239`);
  receiver reader `in_row_tiles`, `out_row_tiles`, `out_col_tiles` (`receiver_reader.cpp:35, 43-44`); receiver writer
  `in_row_tiles`, `in_col_tiles`, `input_row_device_stride`, `input_col_device_stride`, `out_row_tiles`
  (`receiver_writer.cpp:160-163, 168`). All are read and never used.
- **Dead CTAs.** `chunk_num_tiles`, `num_chunks_per_shard` in both receiver kernels; `num_pages_per_packet` (CTA 2) in both
  receiver kernels is shadowed by the RTA of the same name; sender writer `dynamic_alternate` is always `false`
  (`TT_FATAL(!dynamic_alternate)`, `.cpp:246`).
- **Dead sender writer accessor.** `output_tensor_addrgen` (`writer.cpp:94-95`) is built and never used.
- **Dead host locals.** `drain_sync_core` (`.cpp:465-473`) and `enable_async_output` (always `false`, `.cpp:219`).
- **Hardcoded `2` in the receiver reader.** The receiver reader computes `first_id` with `2 * N_DRAM_BANKS`
  (`receiver_reader.cpp:112`); the sender writer uses `contig_pages_advanced * N_DRAM_BANKS` (`writer.cpp:147`), with
  `contig_pages_advanced = pages_per_packet` (`.cpp:289`). The two agree only when a packet holds two pages. The receiver
  reader CTA `contig_pages_advanced` (= `pages_per_packet`) is used only for the transfer size.
- **NoC 0 coordinates on the reader NoC.** The receiver reader asks the intermediate accessor for a NoC 0 address
  (`get_noc_addr(first_id, 0, 0)`, `receiver_reader.cpp:115`) and issues the read on its own NoC (the reader config).
  This is correct only where NoC 0 and NoC 1 coordinates of DRAM agree (translated coordinates).
- **`num_links` is forced to 1** (`all_to_all_async_device_operation.cpp:137-140`) although the factory loops over links.
- **2-device configs** are turned into `Linear` by the op (`all_to_all_async_device_operation.cpp:295-298`), then rejected
  by the factory (`TT_FATAL(topology == Ring)`, `.cpp:164-167`), so the 2-device ring edge case of override 7 cannot
  happen.

## Questions for the user

1. **Readiness sheet.** Please confirm the `experimental/ccl/all_to_all_async` row: `Is able to port?`,
   `TensorParameter relaxation`, `Known op issues`, `Porting Target`. This audit could not read them; the GREEN verdict
   assumes they agree with the code.

## Recipe notes

- **Caller-supplied GlobalSemaphores** (override 5): as for `slice_reshard_async`, `ttnn_factory.md` covers only op-owned
  semaphores.
- **Retained Device 1.0 primitives.** The Device 2.0 gate lists raw `noc_async_read` as broad Device 1.0. The Device 2.0
  migration of this op (#47547) kept one on purpose and marked it (`legacy primitive retained: precomposed uint64_t NoC
  address`); 40 such marks exist in `ttnn/`. The audit recipe does not say whether a marked retention counts as compliant.
  This audit counts it as compliant: the Device 2.0 track already decided it, and the conversion would change the NoC id
  used for the address (see Misc anomalies).
- **Unreachable readiness sheet.** As for the earlier ports: the primary columns were derived from code and the derived
  verdict is unconfirmed.
- **Provenance command** assumes the recipe docs are in the checkout; here the line came from the recipe branch ref.
