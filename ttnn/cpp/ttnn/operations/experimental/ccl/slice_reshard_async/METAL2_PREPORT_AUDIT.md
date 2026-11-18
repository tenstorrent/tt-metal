# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/ccl/slice_reshard_async`

**Audited at `c210a7093f6`.** No `[Prereq]` commit was necessary: every gate clears on the code as it is.

- **`SliceReshardAsyncDeviceOperation`** (`device/slice_reshard_async_device_operation.{hpp,cpp}`)
  - `SliceReshardAsyncProgramFactory` (`device/slice_reshard_async_program_factory.{hpp,cpp}`) — one factory with
    `create_workload_descriptor` (`slice_reshard_async_program_factory.cpp:286`) and `override_runtime_arguments`
    (`:302`). It builds one program per mesh coordinate (`build_slice_reshard_program_descriptor`, `:34`).
- Kernels (both in `device/kernels/`, both referenced, both op-owned):
  - `minimal_default_reader.cpp` — one instance per worker core (reads the input, fills the CB, waits on the final semaphore)
  - `minimal_default_writer.cpp` — one instance per worker core (fabric sender to one neighbor, local writer, barrier semaphore)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead .id form` — from
`git log -1 … origin/akertesz/op-porting-recipe -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`. The recipe
docs are not in this checkout; the line was taken from the recipe branch ref.

**Branch-specific overrides applied** (this branch is newer than the recipe; copied from the `point_to_point` brief):

1. GlobalSemaphore is supported in Metal 2.0 on this branch. The Appendix A "GlobalSemaphore — UNSUPPORTED" entry is not applied.
2. Op-owned GlobalSemaphores go in `MeshWorkloadArtifacts::global_semaphores` (`ttnn/api/ttnn/metal_v2_artifacts.hpp:45`).
   This op owns none: both semaphores are caller-supplied (see TTNN factory analysis).
3. Fabric connections are supported. They replace `tt::tt_fabric::append_fabric_connection_rt_args`.
4. A sanctioned kernel-side exception: the port may replace (a) the raw GlobalSemaphore address RTA and its raw-pointer
   uses with a `global_sem::` binding and (b) the `FabricConnectionManager::build_from_args` RTAs (incl. the
   forward/backward flags) with a `fabric::` binding. Other non-whitelisted kernel changes follow the recipe.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/ccl/slice_reshard_async` |
| **Overall** | **GREEN** (TTNN readiness derived from code; sheet not read — see Questions) |
| **DOps / Factories** | `SliceReshardAsyncDeviceOperation` → `SliceReshardAsyncProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes.** The only Device 1.0 sites left are the GlobalSemaphore and fabric-connection sites that override 4 replaces. |
| *Prereqs* — Cross-op escapes | Ok — `✓ clean` |
| *Feature Support* — overall | GREEN — no Appendix A entry fires after override 1 |
| *Feature Support* — Variadic-CTA | Ok — no variable-index CTA reads |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Not read — sheet not reachable.** Derived from code: no blocking signal. |
| *TTNN Readiness* — Concept (current) | `WorkloadDescriptor` (`slice_reshard_async_program_factory.hpp:41`) |
| *TTNN Readiness* — Execution Model | `SPMD (+ per-device args)` — every coordinate has the same kernels and CB; the CTAs `is_first_chip` / `is_last_chip`, the RTAs and the fabric targets depend on the ring index |
| *TTNN Readiness* — Custom hash | **Yes** (not a gate; port leaves it intact): `slice_reshard_async_device_operation.cpp:50`. It leaves out both GlobalSemaphores. |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | **Yes** (not a gate; the porter translates it): `slice_reshard_async_program_factory.cpp:302` |
| *TTNN Readiness* — Pybind `create_descriptor` | No (`slice_reshard_async_nanobind.cpp` binds only the op) |
| *TTNN Readiness* — Op-owned tensors | No. Op-owned GlobalSemaphores: no (caller-supplied, in `operation_attributes_t`) |
| *TTNN Readiness* — Target concept | `MeshWorkloadSpecFactoryConcept` with the optional `override_runtime_arguments` (derived from code; sheet not read) |
| *Port work* — Offset base pointer | none |
| *Port work* — Tensor bindings (per binding) | Case 1 for both bindings; one dead address RTA (writer `input_tensor_address`) |
| *TTNN Readiness* — TensorParameter relaxation | Not read — sheet not reachable. Code shows no need (the custom hash keys on the input `TensorSpec`; the output spec follows from it and from hashed attributes). |
| *Port work* — TensorAccessor 3rd arg | drop (Class 2) at 1 site |
| *Port work* — CB endpoints | 1P+1C on the one CB; no self-loop, no multi-binding, no dead CB |

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`).

Every gate-bearing subject clears: Device 2.0 (with override 4), feature compatibility (with override 1), offset base
pointers, and the TensorAccessor 3rd argument (one Class 2 site). The TTNN factory concept gate is cleared **from the code
only**: this session could not read the readiness sheet, so its `Is able to port?` cell is unconfirmed (Question 1). If the
sheet says `no`, that cell overrides this verdict.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **Not read.** The readiness sheet needs a Google Drive download tool that
  this session does not have. Values from the code:
  - `Concept` = `WorkloadDescriptor`: `SliceReshardAsyncProgramFactory::create_workload_descriptor`
    (`slice_reshard_async_program_factory.hpp:41`, `.cpp:286`). No `create_descriptor`.
  - `Custom hash`: yes, `SliceReshardAsyncDeviceOperation::compute_program_hash`
    (`slice_reshard_async_device_operation.cpp:50`). It hashes `dim`, `output_dim_offset`, `output_dim_shape`,
    `cluster_axis`, `num_links`, `output_mem_config`, `topology`, `ring_size` and the input `TensorSpec`. It leaves out
    `final_semaphore` and `barrier_semaphore`, which `override_runtime_arguments` re-applies on every hit.
  - `get_dynamic_runtime_args`: none. `override_runtime_arguments`: yes (`.cpp:302`). `Pybind descriptor`: none.
  - `Op-owned tensors?`: no (`WorkloadDescriptor::buffers` not set). `WorkloadDescriptor::semaphores` not set.
  - `Secretly SPMD Workload?`: no — `is_first_chip` / `is_last_chip` are CTAs derived from the coordinate (`.cpp:157-158`,
    `:199-200`). Why mesh-shaped: per-coordinate CTAs, RTAs and fabric targets.
  - `Smuggled pointer`: none for tensors (both are `Buffer*` RTAs). The two GlobalSemaphore addresses are raw RTAs, marked
    `smuggled-rta-ok` and refreshed by the override; override 4a replaces them.
  - Factory set: one factory.
- **Device 2.0 (every kernel used):** **GREEN** (with override 4). All CB access goes through `CircularBuffer` objects and
  all tensor transfers through `Noc::async_read` / `async_write` / `async_write_zeros`. Sites left on purpose, all replaced by
  the port under override 4:

  | File | Line | Call | Override |
  |---|---|---|---|
  | `minimal_default_reader.cpp` | 52, 109, 112 | raw GlobalSemaphore address RTA, `noc_semaphore_wait_min` / `noc_semaphore_set` | 4a |
  | `minimal_default_writer.cpp` | 54, 58, 109, 111 | raw GlobalSemaphore address RTAs, `noc_semaphore_wait_min` / `noc_semaphore_set` | 4a |
  | `minimal_default_writer.cpp` | 85, 156 | `safe_get_noc_addr(x, y, <sem address>, 0)` for the fabric atomic-inc headers | 4a |
  | `minimal_default_writer.cpp` | 59-60, 76, 90-104, 131-142, 160-170, 217 | `FabricConnectionManager::build_from_args`, `open`, forward/backward pick, `close` | 4b |

  No sanctioned free functions are in use. `PacketHeaderPool::allocate_header()` and the `(uint32_t)pkt_hdr` addresses are
  fabric API usage, not Device 2.0 holdovers.
- **Feature compatibility:** GREEN — no gate fired.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no GCB type, no `.global_circular_buffer` field, no remote CB |
  | CBDescriptor `address_offset` (non-zero) | N/A | no CB sets `address_offset` |
  | GlobalSemaphore | **Supported on this branch (override 1)** — not RED | `GlobalSemaphore final_semaphore, barrier_semaphore` in `SliceReshardAsyncParams` (`slice_reshard_async_device_operation_types.hpp:29-30`); include at `:14`; addresses at `slice_reshard_async_program_factory.cpp:133-134`. Under the recipe as written this entry would be RED. |

  Not in Appendix A, so supported: `append_fabric_connection_rt_args` (`slice_reshard_async_program_factory.cpp:240,256`) →
  fabric connections (override 3).
- **CB endpoints (GATE-free):** see Port-work summary. Nothing blocks.
- **Offset base pointers:** GREEN. Every address RTA is a clean base passed as `Buffer*`: reader RTA0 input (`.cpp:172`);
  writer RTA0 input (`:212`, dead — see Misc anomalies); writer RTA1 output (`:213`). The op is not in the offset-base-pointer
  triage tables.
- **TensorAccessor 3rd argument:** GREEN — the one site is Class 2.

  | Site | Tensor | 3rd arg (host value) | Tensor page | Class | Transfer size |
  |---|---|---|---|---|---|
  | `minimal_default_writer.cpp:66` | output | writer RTA `stick_size` = `input_buffer->page_size()` (`.cpp:214`) | output page = input page (ROW_MAJOR, same dtype and width; only dim 0 changes, `slice_reshard_async_device_operation.cpp:37`) | **2** — equal, interleaved | `stick_size` RTA (lines 130, 134, 183-188, 207) |

  The input is never sharded (`slice_reshard_async_device_operation.cpp:23`). The output memory config defaults to the
  input's (`slice_reshard_async.cpp:39`) and every test uses DRAM interleaved. The writer never reads `get_aligned_page_size()`.
  The reader accessor (`minimal_default_reader.cpp:56`) takes no 3rd argument.

  The 3rd-arg triage lists `slice_reshard_async` as Class 1 (its own footnote says the page is constant across the slice).
  It is not Class 1 here: the custom hash keys on the input `TensorSpec`, so no page size varies across cache hits. Do not set
  `dynamic_tensor_shape`.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding; clean bases, fed to a `TensorAccessor`):
  - input → reader (`minimal_default_reader.cpp:56`, RTA0) — **Case 1**
  - output → writer (`minimal_default_writer.cpp:66`, RTA1) — **Case 1**
  - writer RTA0 `input_tensor_address` — read at `minimal_default_writer.cpp:40` and never used. It drops with no binding.
- **TensorParameter relaxation:** none (code-derived; sheet not read).
- **TensorAccessor 3rd arg:** drop at `minimal_default_writer.cpp:66` (Class 2). `stick_size` stays as a named RTA: it is
  still the transfer size.
- **CB endpoints** (each worker core `{core_idx, 0}` has its own reader and writer instance, same census on every core):

  | CB | Touchers on a node | Disposition |
  |---|---|---|
  | `c_in0` sender CB (`slice_reshard_async_program_factory.cpp:122`) | reader FIFO-produces (`reserve_back`/`push_back`, incl. the zero-pad entry); writer FIFO-consumes (`wait_front`/`pop_front`, `get_read_ptr` peek) | 1P+1C (legal) |

  No hidden second writer. No dead CB.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes):** none.
- **Cross-op / shared kernels:** both kernels are op-owned and no other factory or test binds them (`grep -rn
  slice_reshard_async/device/kernels`: only this factory and the CMake glob). Convert them in place; no `_metal2` fork.
- **RTA varargs:** none after override 4 (the fabric-connection tail read by `build_from_args` is replaced by a `fabric::`
  binding).
- **Per-core kernel instances:** the factory pushes one reader and one writer `KernelDescriptor` per worker core, with
  per-direction CTAs. Keep one `KernelSpec` per legacy descriptor.
- **GlobalSemaphore, fabric connection, override translation:** see the brief's Construct and Watch-for sections.

## Team-only

- **Out-of-directory coupling & donor shape:** `✓ clean`.

  | Op kernel | Donor | Class | Shape |
  |---|---|---|---|
  | both | `ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp`, `ccl_host_types.hpp` | cross-family (ccl) | included, no function called by these kernels |
  | writer | `ccl/kernel_common/worker_routing_utils.hpp`, `sharding_addrgen.hpp`, `ccl/common/kernels/minimal_ccl_common.hpp` | cross-family (ccl) | included, no function called |
  | writer | `tt_metal/fabric/hw/inc/...` (`fabric_connection_manager.hpp`, `noc_addr.h` `safe_get_noc_addr`, `packet_header_pool.h`) | `tt_metal/*` | no concern |

  Borrowed kernel files: none.
- **Relaxation candidates:** none.
- **TTNN factory analysis:** target `MeshWorkloadSpecFactoryConcept` with the optional `override_runtime_arguments`.
  Ported-from entry point: `create_workload_descriptor` (`slice_reshard_async_program_factory.cpp:286`), one program per
  coordinate in `tensor_coords`. Per coordinate: ring index, neighbors, the derived outer-dim counts (RTAs), the
  `is_first_chip` / `is_last_chip` CTAs and the fabric targets. Ported-from override (`.cpp:302`): writes the final
  semaphore address into reader RTA 9 and writer RTA 14 and the barrier semaphore address into writer RTA 18, on every
  core, for every program. No op-owned tensors or semaphores.

## Misc anomalies  *(team-only, non-gating)*

- **Dead RTA.** Writer RTA0 `input_tensor_address` (`slice_reshard_async_program_factory.cpp:212`) is read at
  `minimal_default_writer.cpp:40` and never used.
- **Kernels on cores without the CB.** `split_work_to_cores(core_grid, num_sticks_per_outer_dim * 2)` (`.cpp:108`) can use
  fewer cores than `core_grid` when `num_sticks_per_outer_dim < num_links`. The CB is created only on `worker_core_ranges`
  (`:124`), but the loop (`:137`) creates a reader and a writer on every core of `core_grid`. On an unused core the count is
  `num_sticks_per_core_group_2` (0), but the reader's zero-pad path (`minimal_default_reader.cpp:82-87`) still reserves a CB
  entry. Untested edge case.
- **Constant RTA.** Writer `use_barrier_semaphore` is always `1u` (`.cpp:229`).
- **Topology attribute unused by the factory.** Neighbors are always linear (no wrap-around), whatever `topology` is; it is
  only in the hash.
- **Validation message.** `"Error, neighbor pad currently only supports sharding dim 0"`
  (`slice_reshard_async_device_operation.cpp:21`) names another op.
- **Unused includes.** The factory includes several CCL host headers (`ccl_worker_builder.hpp`, `command_lowering.hpp`, …)
  and the writer several kernel headers (`sharding_addrgen.hpp`, `worker_routing_utils.hpp`, …) that nothing uses.

## Questions for the user

1. **Readiness sheet.** Please confirm the `experimental/ccl/slice_reshard_async` row: `Is able to port?`,
   `TensorParameter relaxation`, `Known op issues`, `Porting Target`. This audit could not read them; the GREEN verdict
   assumes they agree with the code.

## Recipe notes

- **Caller-supplied GlobalSemaphores.** `ttnn_factory.md` "Feasibility gate" covers only op-owned GlobalSemaphores. A
  caller-supplied semaphore in `operation_attributes_t`, left out of the custom hash and refreshed by
  `override_runtime_arguments`, maps onto `AdvancedProgramRunArgs::global_semaphore_args` in the override. The recipe has no
  entry for it.
- **3rd-arg triage disagrees with the code.** The triage puts this op in Class 1, but the custom hash pins the page size per
  cache entry, so it is Class 2. Same pattern as `point_to_point`.
- **Unreachable readiness sheet.** As for `point_to_point`: the primary columns were derived from code and the derived
  verdict is unconfirmed.
- **Provenance command** assumes the recipe docs are in the checkout; here the line came from the recipe branch ref.
