# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async`

- **`AllToAllAsyncDeviceOperation`** (`device/all_to_all_async_device_operation.{hpp,cpp}`, params/inputs in `device/all_to_all_async_device_operation_types.hpp`)
  - `AllToAllAsyncProgram` (`device/all_to_all_async_program_factory.{hpp,cpp}`): `create_descriptor(…, const std::optional<MeshCoordinate>&)` + `override_runtime_arguments`
- Kernels (all in `device/kernels/`, all bound only by this factory):
  - `interleaved_all_to_all_reader.cpp`: sender reader (input → CB `c_in0`)
  - `interleaved_all_to_all_writer.cpp`: sender writer (CB `c_in0` → fabric → remote intermediate buffer, plus a fabric atomic-inc on the remote receiver core's `GlobalSemaphore`)
  - `interleaved_all_to_all_receiver_reader.cpp`: receiver reader (local copy from input, or intermediate → CB `c_in0` after waiting on the `GlobalSemaphore`)
  - `interleaved_all_to_all_receiver_writer.cpp`: receiver writer (CB `c_in0` → output)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `6d8d49be280 2026-11-26 docs(metal_2.0): state that a port is performance- and functionality-neutral` *(from the `akertesz/op-porting-recipe` worktree; `main` carries no copy of these docs)*

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `AllToAllAsyncDeviceOperation` → `AllToAllAsyncProgram` |
| *Prereqs* — Device 2.0 (every kernel used) | Yes |
| *Prereqs* — Cross-op escapes | Ok (one in-family donor header, all calls ✓) |
| *Feature Support* — overall | GREEN (every Appendix A row N/A) |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — `Is able to port?` (the gate) | Yes |
| *TTNN Readiness* — Concept (current) | `descriptor (per-coord)` |
| *TTNN Readiness* — Execution Model | `MPMD` (`Secretly SPMD Workload?` = `no`, cross-checked: `ring_index` is a compile-time arg) |
| *TTNN Readiness* — Custom hash | Yes (not a gate; port leaves it intact): `device/all_to_all_async_device_operation.cpp:246` |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | Yes (not a gate; the porter translates it): `AllToAllAsyncProgram`, `device/all_to_all_async_program_factory.cpp:607` |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MeshWorkloadSpecFactoryConcept` (with the translated override) |
| *Port work* — Offset base pointer | none |
| *Port work* — Tensor bindings (per binding) | Case 1 ×3 (`input`, `intermediate`, `output`) |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | none — no accessor passes a 3rd arg |
| *Port work* — CB endpoints | legal ×2, self-loop ×1 |

## Result

**GREEN → brief issued.** Every gate clears. The op is a per-coordinate `ProgramDescriptor` CCL op with one caller-supplied `GlobalSemaphore` (excluded from the hash, re-applied in the override) and a forward/backward `FabricConnectionManager` on one link. Both map onto the Metal 2.0 constructs in `ccl_port.md` with no unsupported feature in play.

**Verification caveat (not a gate):** the op cannot run on two devices. On 2 devices the entry point demotes `Ring` to `Linear` (`device/all_to_all_async_device_operation.cpp:295-298`), and the factory `TT_FATAL`s unless the topology is `Ring` (`device/all_to_all_async_program_factory.cpp:164-167`). Even a `Ring` call would then fail the forward ≠ backward check (`:196-200`). Its functional tests are all 1×8 / ≥3-device ring tests. See *Questions for the user*.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN. The sheet's `Is able to port?` is `yes`, and the cross-check is clean:
  - `Concept` `descriptor (per-coord)`: `create_descriptor` takes `const std::optional<ttnn::MeshCoordinate>&` (`device/all_to_all_async_program_factory.hpp:16-20`). ✓
  - `Custom hash` `yes`: `compute_program_hash` at `device/all_to_all_async_device_operation.cpp:246`. ✓
  - `Runtime-args update (get_dynamic_runtime_args)` `no`: no hook in the op directory. ✓
  - `Override runtime args method?` `yes`: `device/all_to_all_async_program_factory.cpp:607`. ✓
  - `Pybind descriptor` `no`: `all_to_all_async_nanobind.cpp` binds only the op. ✓
  - `Secretly SPMD Workload?` `no`: `ring_index` is CTA 0 of every kernel (`device/all_to_all_async_program_factory.cpp:319, 343, 374, 394`). ✓
  - Factory set: one row, one factory (`AllToAllAsyncProgram`). ✓
  - Invariants: `Op-owned tensors?` `no` on a `descriptor (per-coord)` row. ✓
- **Device 2.0 (every kernel used):** GREEN. All four kernels use `Noc` / `CircularBuffer` / `TensorAccessor`. The remaining raw calls are sanctioned:
  - `interleaved_all_to_all_receiver_reader.cpp:108` `noc_semaphore_wait_min` and `:130` `noc_semaphore_set`: raw `GlobalSemaphore` primitives on the caller semaphore's address (also marked `legacy primitive retained`).
  - `interleaved_all_to_all_receiver_reader.cpp:113-117` `noc_async_read(packet_addr, …)`: marked `// Device 2.0 migration: legacy primitive retained: precomposed uint64_t NoC address`. The address comes from `intermediate_tensor_addrgen.get_noc_addr(…)`, not hand-built.
  - `interleaved_all_to_all_writer.cpp`: fabric device API only (`FabricConnectionManager`, `WorkerToFabricEdmSender`, packet headers, `fabric_set_unicast_route`, `tt::tt_fabric::linear::to_noc_unicast_write`, `safe_get_noc_addr`) plus the donor helpers below, which are themselves fabric API.
- **Feature compatibility:** GREEN (no gate fired).

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `address_offset` set |
  | GlobalSemaphore in a compute kernel | N/A | no compute kernel |
  | Caller GlobalSemaphore not re-applied on a cache hit | N/A | caller semaphore excluded from the hash (`device/all_to_all_async_device_operation.cpp:254`) and re-applied by the override (`device/all_to_all_async_program_factory.cpp:619, 640, 664`) |
  | Fabric mux | N/A | direct `FabricConnectionManager::build_from_args` (`interleaved_all_to_all_writer.cpp:65`); `minimal_ccl_common.hpp` includes `tt_fabric_mux_interface.hpp` but the op calls no mux function (false-positive guard) |
  | RPCM with per-route or auto-selected links | N/A | no RPCM |
  | MeshSocket | N/A | |
  | Fused CCL op signalers | N/A | |

- **CB endpoints (GATE-free):** see Port-work summary. Nothing blocks.
- **Offset base pointers:** GREEN. Every tensor address reaches a kernel as a bare `Buffer*` runtime arg (`device/all_to_all_async_program_factory.cpp:478, 540-541, 566-567, 588`) with no host-side offset. The op is not in the offset-base-pointer triage tables.
- **TensorParameter analysis (Case 2):** GREEN. No Case 2 binding (inventory below).
- **TensorAccessor 3rd argument:** N/A. No accessor in the op passes a 3rd argument.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding). All arrive as `Buffer*` runtime args (framework-patched on a cache hit today, and also patched by the override). Every kernel use goes through a `TensorAccessor`:
  - `input`: **Case 1**. Sender reader RTA 0 → `TensorAccessor(TensorAccessorArgs<7>, …)` (`interleaved_all_to_all_reader.cpp:32, 42-43`); receiver reader RTA 1 → `TensorAccessorArgs<10>` (`interleaved_all_to_all_receiver_reader.cpp:33, 51-52`).
  - `intermediate`: **Case 1**. Sender writer RTA 0 → `TensorAccessorArgs<13>` (`interleaved_all_to_all_writer.cpp:50, 92-93`), consumed by the fabric write helpers as an address generator. Receiver reader RTA 0 → accessor chained after `input` (`:32, 53-54`), read through `get_noc_addr(first_id, …)`.
  - `output`: **Case 1**. Receiver writer RTA 0 → `TensorAccessorArgs<8>` (`interleaved_all_to_all_receiver_writer.cpp:28, 45-46`). The sender writer also receives it (RTA 1, accessor at `interleaved_all_to_all_writer.cpp:94-95`) but never uses the accessor (see Misc anomalies).
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:**
  - Sender cores, `c_in0` (data): reader FIFO-produces, writer FIFO-consumes → legal 1P+1C.
  - Sender cores, `c_in1` (packet headers, `device/all_to_all_async_program_factory.cpp:86-94`): writer only (`reserve_back`/`push_back` + `get_write_ptr`, `interleaved_all_to_all_writer.cpp:76-84`) → **self-loop** on the writer.
  - Receiver cores, `c_in0` (data, `:106-114`): receiver reader FIFO-produces, receiver writer FIFO-consumes → legal 1P+1C.
  - The sender and receiver core ranges are disjoint (`:265`), so the two `c_in0` CBs are two separate DFBs that happen to share an index.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none.
- **Cross-op / shared kernels:** none. All four kernels live in the op directory and only this factory binds them (grep of `ttnn/`, `tests/`, `models/`): convert in place, no `_metal2` fork.
- **Indexed-collection args:** none. Every RTA and CTA is a fixed, named field. The fabric connection args after RTA 12 of the sender writer are read by `FabricConnectionManager::build_from_args` and become fabric bindings, not an array.
- **CCL constructs in use** (per `ccl_port.md`):
  - **One caller-supplied `GlobalSemaphore`** (`operation_attributes.semaphore`): bound by the sender writer (RTA 2, used to address the remote receiver's semaphore in a fabric atomic-inc) and by the receiver reader (RTA 2, raw `noc_semaphore_wait_min` / `noc_semaphore_set`). It is excluded from the hash and re-applied in the override: the *Caller-supplied: rebind it in the override* case.
  - **`FabricConnectionManager` with forward/backward present-flags**, one link (`num_links` must be 1, `device/all_to_all_async_device_operation.cpp:137-140`), one sender core: the *One sender, or a FabricConnectionManager* case → two one-target specs, `forward` and `backward`. The op requires a `Ring` of ≥3 devices, so both sides exist on every device.
  - No RPCM, no op-allocated semaphore, no mux.
- **`FABRIC_2D`:** the writer's own source has no `FABRIC_2D` branch. Its includes have some (`tt_fabric_api.h:22` static_assert only; `edm_fabric_worker_adapters.hpp:32` the binding word layout; `linear/tt-1xx/api_impl.h:27,40` RPCM-only helpers the writer doesn't call). Every test of the op uses `FABRIC_1D_RING`.

## Team-only

- **Out-of-directory coupling & donor shape:** `✓ clean`.

  | Op kernel | Donor file | Class | Calls | Status |
  |---|---|---|---|---|
  | `interleaved_all_to_all_writer.cpp` | `ttnn/cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp` | in-family shared (ccl) | `perform_atomic_fabric_write`, `perform_payload_send` | ✓ |
  | `interleaved_all_to_all_writer.cpp` | `tt_metal/fabric/hw/inc/…` (`fabric_connection_manager.hpp`, `noc_addr.h`, `tt_fabric_api.h`) | `tt_metal/*` | fabric API | ✓ |

  Per-call: `perform_atomic_fabric_write(pkt_hdr, dest_id, AddrGenType addrgen, WorkerToFabricEdmSender&, size_t&, uint32_t, uint64_t semaphore_noc_addr, …)` takes a `TensorAccessor` (Shape 1 ✓). Its `uint64_t semaphore_noc_addr` is a `GlobalSemaphore` address composed through `safe_get_noc_addr`, which `ccl_port.md` keeps unchanged. Read literally, the shape table's `uint64_t NOC-encoded sem` row would mark it ✗ (see Recipe notes). `perform_payload_send(WorkerToFabricEdmSender&, …)`: fabric API ✓.
  Borrowed kernel files: none. Reverse coupling: nothing outside the op directory references its kernels or factory (only `experimental/ccl/{CMakeLists.txt,sources.cmake}` and the family nanobind include).
- **Relaxation candidates:** none noticed. The custom hash pins `tensor_args` whole. It drops `semaphore` and replaces `sub_device_id` with the sub-device's worker core range set (`device/all_to_all_async_device_operation.cpp:250-263`). Neither is a tensor property.
- **TTNN factory analysis:** concept `descriptor (per-coord)` → `MeshWorkloadSpecFactoryConcept`. Per-coordinate entry point `AllToAllAsyncProgram::create_descriptor` (`device/all_to_all_async_program_factory.cpp:149`). What varies per coordinate:
  - `ring_index`: CTA 0 of all four kernels.
  - Forward/backward fabric peers (`:173-193`).
  - The receiver core the sender's semaphore increment targets (`receiver_core_x/y`, `:462-463`).
  - The per-device input strides and offsets (`:437-438`).

  The override (`:607`) refreshes input/intermediate/output addresses and the semaphore address on every kernel that receives them. Custom hash at `device/all_to_all_async_device_operation.cpp:246`. No `get_dynamic_runtime_args`, no pybound `create_descriptor`, no op-owned tensors.

## Misc anomalies  *(team-only, non-gating)*

- **The op cannot run on 2 devices.** `device/all_to_all_async_device_operation.cpp:295-298` turns `Ring` into `Linear` for 2 devices, then `device/all_to_all_async_program_factory.cpp:164-167` `TT_FATAL`s on anything but `Ring`. On 2 devices `forward_coord == backward_coord` also trips `:196-200`.
- **Receiver reader hardcodes the packet width.** `interleaved_all_to_all_receiver_reader.cpp:112` computes `first_id` with `2 * N_DRAM_BANKS`, while the sender writer uses `contig_pages_advanced * N_DRAM_BANKS` with `contig_pages_advanced = pages_per_packet` (`interleaved_all_to_all_writer.cpp:146-147`, `device/all_to_all_async_program_factory.cpp:289`). It agrees only when a packet holds 2 pages. The receiver's own `contig_pages_advanced` CTA (8) is used only for the payload size.
- **Dead kernel args.** Read but never used:
  - Sender reader: `in_row_tiles`, `out_row_offset`, `out_col_offset`.
  - Sender writer: `num_packet_headers_storable`, `out_row_tiles`, `out_col_tiles`, `wait_output_semaphore`, `reset_global_semaphore`, the `output` accessor (`:94-95`), `arg_for_fab` (`:63`).
  - Receiver reader: `chunk_num_tiles`, `num_chunks_per_shard`, `in_row_tiles`, `out_row_tiles`, `out_col_tiles`.
  - Receiver writer: `ring_size`, `chunk_granularity`, `chunk_num_tiles`, `num_chunks_per_shard`, `in_row_tiles`, `in_col_tiles`, `input_row_device_stride`, `input_col_device_stride`, `out_row_tiles`.
- **Shadowed CTA.** Both receiver kernels read an RTA named `num_pages_per_packet` (`interleaved_all_to_all_receiver_reader.cpp:45`, `interleaved_all_to_all_receiver_writer.cpp:39`) that shadows a CTA of the same name (`:18`). The factory passes the same value to both (`pages_per_packet`).
- **Remote receiver addressing.** The sender addresses the remote receiver core by this device's physical coordinates for `receiver_worker_cores[ring_index]` (`device/all_to_all_async_program_factory.cpp:462-463`). That assumes identical worker-core translation on every device.

## Questions for the user

1. **Verification hardware.** The op's functional tests need a ring of ≥3 devices (`tests/nightly/t3000/ccl/test_all_to_all.py` on 1×8; the Blackhole CI box tests skip on Wormhole and on ≤2 devices). `tests/nightly/t3000/ccl/test_ccl_topology_labels.py::test_all_to_all_async_whole_mesh[1x2]` is parametrized for an N300, but per the anomaly above it can only skip or fail identically before and after a port. On an N300 the port can be compiled, and its 2-device failure mode compared, but not functionally tested.

## Recipe notes

1. **`ccl_port.md` vs. the audit's donor shape table on GlobalSemaphore addresses.** The out-of-directory coupling table marks `uint32_t sem_addr (L1) or uint64_t NOC-encoded sem` as "✗ not OK — No clean Metal-2.0 → donor bridge today". `ccl_port.md` § *Kernel side* says a `GlobalSemaphore` address flows unchanged into "helpers in or outside the op directory". Read literally, the table marks `perform_atomic_fabric_write` ✗ for its `uint64_t semaphore_noc_addr`. I followed `ccl_port.md` (newer, specific) and marked it ✓. The table row should say it covers program semaphores, and that a `GlobalSemaphore` address now has a bridge (`GlobalSemaphore(global_sem::x).address()`).
2. **`FABRIC_2D` scope in `ccl_port.md`** ("a kernel with `#if defined(FABRIC_2D)` code that the legacy factory compiled *without* the define … stop"). It doesn't say whether included headers count. Fabric headers every CCL writer includes have `FABRIC_2D` branches (the binding word layout in `edm_fabric_worker_adapters.hpp` among them), so read transitively the rule would stop every CCL port on a 2D fabric. I read it as "the kernel's executed code paths" and checked the headers by hand. Suggest saying "branches the kernel actually executes", and noting that the binding-layout branch is expected.
3. **The readiness fetch's saved-result format.** `ttnn_op_porting_readiness.md` step 2 says the saved file is JSON with a base64 `content` field. This session's harness saved the CSV itself (`…-blob-….csv`), so step 3's `jq | base64 -d` would fail. I copied the file directly. Suggest: "if the saved file is already CSV, copy it."
4. **Provenance command assumes the recipe and the op share a checkout.** The docs live on `akertesz/op-porting-recipe`; the port branch is off `main`, which has no `metal_2.0/ai/` tree. The provenance command run from the port checkout prints nothing, which the recipe reads as "untracked docs". I ran it in the recipe worktree.
5. **Unrunnable-op verification isn't anticipated.** The audit has no slot for "the op can't run on the hardware the port will be verified on". I put it in *Result* as a caveat and in *Questions*. A heads-up in the brief template would help the porter plan verification.
