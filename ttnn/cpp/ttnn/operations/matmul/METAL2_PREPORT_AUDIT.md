# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/matmul`

This directory hosts **two independent device operations** that share the `device/kernels/` pool and several program factories' kernel files. They are bundled into this single report with per-DeviceOperation attribution; the matmul kernel pool is shared, so they form one porting unit.

- **`ttnn::prim::MatmulDeviceOperation`** (`device/matmul_device_operation.{hpp,cpp}`)
  - `MatmulMultiCoreProgramFactory` (`factory/matmul_multicore_program_factory.cpp`) — **PD**
  - `MatmulMultiCoreReuseOptimizedProgramFactory` (`factory/matmul_multicore_reuse_optimized_program_factory.cpp`) — **PD**
  - `MatmulMultiCoreReuseMcast1DProgramFactory` (`factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp`) — **PD** (`create_descriptor`, non-gather)
  - `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory` (same file) — **imperative** (gather_in0 path; `create_mesh_workload`)
  - `MatmulMultiCoreReuseMcast2DProgramFactory` (`factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp`) — **PD**
  - `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory` (`factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp`) — **PD**
  - `MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory` (`factory/matmul_multicore_reuse_batched_hs_dram_sharded_program_factory.cpp`) — **PD**
- **`ttnn::prim::SparseMatmulDeviceOperation`** (`device/sparse/sparse_matmul_device_operation.{hpp,cpp}`)
  - `SparseMatmulMultiCoreReuseMcast1DProgramFactory` (`device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp`) — **imperative**

**Unreferenced kernel files in the directory (out of scope — no factory references them):** `device/kernels/dataflow/reader_bmm_tile_layout.cpp`, `reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp`, `reader_bmm_tile_layout_in1_receiver_writer_padding.cpp`, `writer_bmm_tile_layout.cpp`, and `device/kernels/compute/bmm_large_block_zm.cpp`.

**Referenced kernels (in scope, 16 distinct files):**
- compute: `bmm.cpp`, `bmm_large_block_zm_fused_bias_activation.cpp`, `bmm_large_block_zm_fused_bias_activation_gathered.cpp` (+ headers `bmm_fused_activation.hpp`).
- dataflow: `reader_bmm_8bank_output_tiles_partitioned.cpp`, `reader_bmm_tile_layout_in0.cpp`, `reader_bmm_tile_layout_in0_receiver.cpp`, `reader_bmm_tile_layout_in0_ring_all_gather.cpp`, `reader_bmm_tile_layout_in0_sender_dram_sharded.cpp`, `reader_bmm_tile_layout_in0_sender_dram_sharded_height.cpp`, `reader_bmm_tile_layout_in0_sender_padding.cpp`, `reader_bmm_tile_layout_in1_ring_all_gather.cpp`, `reader_bmm_tile_layout_in1_sender_dram_sharded.cpp`, `reader_bmm_tile_layout_in1_sender_dram_sharded_height.cpp`, `reader_bmm_tile_layout_in1_sender_writer_padding.cpp`, `reader_writer_bmm_tile_layout_in1.cpp`, `writer_unary_interleaved_start_id.cpp`.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/matmul` |
| **Overall** | **RED** (at op level) |
| **DOps / Factories** | `MatmulDeviceOperation` → 7 factories (5 clean PD, 1 PD/clean-subject-to-feature-gate, 1 imperative MeshWorkload); `SparseMatmulDeviceOperation` → 1 imperative factory |
| *Prereqs* — ProgramDescriptor | **No** (mixed: 5 PD factories clean; `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory` and the entire `SparseMatmulDeviceOperation` are imperative `host_api.hpp`) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes-with-holdovers** (YELLOW — `reader_bmm_tile_layout_in0_receiver.cpp` raw sem-address read; fix on D2.0 track first) |
| *Prereqs* — Cross-op escapes | Ok (CCL `worker_sync_utils.hpp` is a header-only include; no donor-kernel file instantiation, no cross-family addr-gen donor) |
| *Feature Support* — overall | **RED** |
| *Feature Support* — Variadic-CTA | Ok (no runtime-varying-index CTA loops; `std::vector<Tensor>` input list is fixed-shape at port time per path) |
| *TTNN Readiness* — Op-owned tensors | No (output tensors created in `compute_output_specs`/`create_output_tensors`, not factory-owned scratch) |
| *TTNN Readiness* — MeshWorkload needed | **Yes (legacy artifact, not genuine)** — `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory` is the gather_in0 path; it is on the MeshWorkload path because `create_descriptor` "not yet supported" for gather (`matmul_device_operation.cpp:701`), not a genuine cross-device need |
| *TTNN Readiness* — Pybind `create_descriptor` | **Yes** — 6 factory classes bind `create_descriptor` (`matmul_nanobind.cpp:1210-1310`), each with the tell-tale extra `core_range_set` param |
| *TTNN Readiness* — Other risky pybind | **Yes** — `nb::class_<MatmulDeviceOperation>` exposes `create_output_tensors`/`compute_output_specs`/`compute_program_hash` (`matmul_nanobind.cpp:1192-1207`) |
| *TTNN Readiness* — Custom hash | **Yes → delete** (`MatmulDeviceOperation::compute_program_hash` @ `matmul_device_operation.cpp:2127`). Sparse: none (commented out @ `sparse_matmul_device_operation.cpp:256`) |
| *TTNN Readiness* — Custom override-RTA | **Yes** — `MatmulMultiCoreReuseMcast1DProgramFactory::override_runtime_arguments`, `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory::override_runtime_arguments`, `MatmulMultiCoreReuseMcast2DProgramFactory::override_runtime_arguments`, `SparseMatmul...::override_runtime_arguments` |
| *TTNN Readiness* — Fake CBs (address-only) | None observed |

**Fake CBs** = CBs used purely as an address source. No fake CBs found: every CB in the audited factories has a producer and consumer; sharded/borrowed CBs are real DFBs (`.tensor`-backed / `set_globally_allocated_address`).

## Result

**RED at op level** — blocked on two gates that fire on a *subset* of the bundled work, while a large clean subset is fully GREEN-feasible:

1. **ProgramDescriptor prerequisite (GATE):** the gather_in0 MeshWorkload path (`MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory`) and the entire `SparseMatmulDeviceOperation` are still on the imperative `host_api.hpp` builder. Routed to the **ProgramDescriptor migration team**. This is the expected outcome for legacy paths; it unblocks when those paths' ProgramDescriptor migration lands.
2. **GlobalCircularBuffer feature (GATE, UNSUPPORTED):** the `global_cb` / "remote CB" path used by `MatmulMultiCoreReuseMcast1DProgramFactory`'s imperative helper and by `SparseMatmulMultiCoreReuseMcast1DProgramFactory`; surfaces in the kernel `reader_bmm_tile_layout_in1_ring_all_gather.cpp` (`ENABLE_GLOBAL_CB`). Routed to the **wait-for-feature** track (GlobalDataflowBuffer not yet implemented).

**Clean subset that IS portable today (offer a scoped-subset port):** the five fully-PD factories of `MatmulDeviceOperation` whose `create_descriptor` path takes no `global_cb` —
`MatmulMultiCoreProgramFactory`, `MatmulMultiCoreReuseOptimizedProgramFactory`, `MatmulMultiCoreReuseMcast2DProgramFactory`, `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory`, `MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory` — plus `MatmulMultiCoreReuseMcast1DProgramFactory`'s **non-gather, non-global_cb** `create_descriptor` path. These are blocked only by the YELLOW Device-2.0 holdover in `reader_bmm_tile_layout_in0_receiver.cpp` (which is shared with the imperative paths). A subset port is feasible once that holdover is cleared — *but no `METAL2_PORT_BRIEF.md` is issued*, because the op-level gates (ProgramDescriptor on the gather/sparse paths, GlobalCircularBuffer) are RED; a scoped subset port requires explicit user scoping and is evaluated case-by-case.

## Gate detail

- **ProgramDescriptor:** **RED (subset).**
  - **PD-clean (live device-op path):** `MatmulMultiCoreProgramFactory`, `MatmulMultiCoreReuseOptimizedProgramFactory`, `MatmulMultiCoreReuseMcast2DProgramFactory`, `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory`, `MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory` all populate a `ProgramDescriptor` via `create_descriptor` and use `KernelDescriptor`/`CBDescriptor`/`SemaphoreDescriptor`. `MatmulMultiCoreReuseMcast1DProgramFactory::create_descriptor` (`matmul_multicore_reuse_mcast_1d_program_factory.cpp:5085`) dispatches to PD builders `create_program_mcast_in0_descriptor`/`_in1_descriptor` (lines 2889, 3906).
  - **RED — imperative `host_api.hpp` (live device-op paths):**
    - `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory::create_mesh_workload` (`matmul_multicore_reuse_mcast_1d_program_factory.cpp:5359`) calls the imperative `matmul_multi_core_reuse_mcast_1d_optimized_` which uses `tt_metal::CreateSemaphore` (`:314-315`, `:1278-1279`, `:2094`), `tt_metal::CreateKernel` (`:585`, `:610`, `:629`, `:649`, `:662`, `:740`, `:1512`, `:1527`, `:1545`, `:1628`, `:2350`, `:2361`, `:2377`), and `UpdateDynamicCircularBufferAddress` / `GetRuntimeArgs` in `override_program_parameters` (`:2705-2818`). Selected when `program_config.gather_in0` (`matmul_device_operation.cpp:701-703`).
    - `SparseMatmulMultiCoreReuseMcast1DProgramFactory` (`sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp`) is wholly imperative: `tt_metal::CreateSemaphore` (`:256-257`), `tt_metal::CreateKernel` (`:427`, `:444`, `:457`, `:512`), `tt_metal::CreateCircularBuffer` (`:538`, `:553`, `:584-585`, `:603`, `:622`), `tt_metal::SetRuntimeArgs` (`:673`, `:686`, `:753`).
  - *Note:* the imperative `matmul_multi_core_reuse_mcast_1d_optimized_helper` / `create_program_mcast_in0_in1` helpers in the mcast_1d/mcast_2d files (with `set_globally_allocated_address`, `CreateKernel`, etc.) are **legacy/CCL-fusion entry points** *not* reached by `MatmulDeviceOperation`'s `create_descriptor` — but they are exported (`*_helper` signatures in the headers) and called by CCL ops; that cross-op sharing is recorded under Out-of-directory coupling.
  - Framing: ProgramDescriptor migration is a substantial standalone effort; a RED here on the gather/sparse paths is the expected outcome for legacy code, and those paths unblock once their migration lands.

- **Device 2.0 (every kernel used):** **YELLOW** — one isolated holdover; all other 15 referenced kernels are GREEN.

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | `device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp` | 27, 41-42, 57 | `uint32_t in0_mcast_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(5));` → `reinterpret_cast<volatile tt_l1_ptr uint32_t*>(...)` → `*ptr == VALID` | `Semaphore<> receiver_sem(get_compile_time_arg_val(5))` @ `:39` (same CTA index 5) |

  The kernel is otherwise structurally Device 2.0 (`Noc`, `CircularBuffer`, `Semaphore<>` used throughout — `:36-39`, `:51-83`). The holdover is a single raw-L1-sem-address read of the sparsity-valid flag, used only in the `get_batch_from_reader` (unstructured-sparsity) branch; the `receiver_sem` wrapper for the same semaphore is already in scope, so the value read should go through it. **Route to the Device 2.0 effort to be cleaned first; not part of the port diff.** All other referenced kernels — including the DRAM-sharded readers using `AllocatorBank` and the ring/all-gather readers using `Noc`/`CircularBuffer`/`Semaphore<>`/`TensorAccessor` — are GREEN with no Device 1.0 idioms.

  > **Nuance (RED-leaning — flagged on verification).** This holdover is tiered YELLOW because the `Semaphore<>` wrapper *is* in scope (so Metal-2.0 binding tokens attach — the recipe's structural feasibility test passes). But two facts pull it toward RED rather than the clean "isolated CB-index holdover" the YELLOW tier is scoped to: (1) the recipe's YELLOW carve-out is specifically the *CB-index-keyed free-function family* (`get_read_ptr`/`get_write_ptr`); a **raw semaphore-address read** is listed under the **RED** Device-1.0 idioms ("raw sem addresses"). (2) `Semaphore<>` (`tt_metal/hw/inc/api/dataflow/noc_semaphore.h`) exposes `up/down/wait/wait_min/set/...` but **no public member to read the current value** — so the `*ptr == VALID` read has *no member-form equivalent today* and is **not** the presumed 1-line swap. Net effect on tiering is nil (matmul is RED at op level regardless), but the Device-2.0 team should treat this as a genuine Device-1.0 idiom needing a (possibly new) value-read member, not a mechanical holdover cleanup.

- **Feature compatibility:** every Appendix A entry, in order.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | **RED** | UNSUPPORTED, in use on the `global_cb`/gather path — see detail below |
  | Dynamic CircularBuffer (borrowed memory) | GREEN | In use (LANDED); port uses `borrowed_from`. PD: `CBDescriptor::tensor = &sharded_tensor` (`optimized:507,558`; `mcast_dram_sharded:549,606`; `batched_hs:257,329`; `mcast_2d:945,961,977,1007,1039`). Imperative: `set_globally_allocated_address(...)` (`mcast_1d:774,792,855,875,1647,...`; `mcast_2d:2388,2405,2422,2485`) |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset`/`set_address_offset`/4-arg `UpdateDynamicCircularBufferAddress` found |
  | Aliased Circular Buffers | GREEN | In use (LANDED); port uses `advanced_options.alias_with`. PD: shared output+interm CB with two `CBFormatDescriptor` (c_4 + c_5) — `optimized:548-557`, `mcast_2d:1028-1037` ("share buffer"). Imperative also at `mcast_1d` ~`:840` per Appendix A |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` / `CreateGlobalSemaphore` / `global_semaphore.hpp` anywhere |
  | Non-zero semaphore initial value | GREEN | In use (LANDED, deprecated). `INVALID` sentinel (non-zero) in PD `SemaphoreDescriptor::initial_value` (`mcast_2d:1075-1081`, `mcast_1d:3704-3706,4567-4569`) and imperative `CreateSemaphore(..., INVALID)` (`mcast_1d:314-315`, `mcast_2d:1820-1823`, `sparse:256-257`) — heads-up only |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | `grep ArgConfig::Runtime` over the op returns nothing |
  | `UpdateCircularBuffer*` | N/A | No `UpdateCircularBufferTotalSize`/`PageSize`/`AddressAndTotalSize`. (`UpdateDynamicCircularBufferAddress` 3-arg form @ `mcast_1d:2741-2818` is the supported address-rebind, *not* this rule) |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` carries `std::vector<Tensor> input_tensors` (`matmul_device_operation_types.hpp:33`), but no kernel reads CTAs with a runtime-varying index; the gathered compute kernel iterates `fill_named_cb_array<batch>` over a **constexpr** `batch`. Fixed-shape per path |

#### RED detail — GlobalCircularBuffer (UNSUPPORTED)

- **Signal fired:** construction-by-consumption — `tt_metal::experimental::CreateCircularBuffer(program, all_cores, remote_cb_config, *global_cb)` with `remote_cb_config.remote_index(remote_cb_index)`, in `matmul_multicore_reuse_mcast_1d_program_factory.cpp:2119-2127` (`use_global_cb` branch). Type `std::optional<...GlobalCircularBuffer>& global_cb` threaded through the factory signatures (`.hpp:70,87,93`), the op attribute `MatmulParams::global_cb` (`matmul_device_operation_types.hpp:28`), and the sparse op (`sparse_matmul_device_operation_types.hpp:25`, factory `:43`).
- **Kernel-side confirmation:** `reader_bmm_tile_layout_in1_ring_all_gather.cpp` — `#include "...remote_circular_buffer.h"` (`:9`), `cb_remote` named CTA (`:112`), `experimental::remote_cb_wait_front` (`:142`), `remote_cb_pop_front` (`:202`), `update_remote_cb_config_in_l1` (`:214`), all under `#ifdef ENABLE_GLOBAL_CB`.
- **Affected factories (ProgramFactory granularity):** `MatmulMultiCoreReuseMcast1DProgramFactory`'s imperative helper + the `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory` gather path; and `SparseMatmulMultiCoreReuseMcast1DProgramFactory`. The five other `MatmulDeviceOperation` factories and the **non-global_cb `create_descriptor`** mcast_1d path do not touch `global_cb`.
- **Expected resolution:** not yet supported in Metal 2.0 — GlobalCircularBuffer maps by lifetime to the unimplemented user-managed `GlobalDataflowBuffer`; do not map onto any current DFB variant. The port of the affected factories becomes possible once GlobalDataflowBuffer lands.

## Port-work summary  *(mirrors the brief — applies to the clean-subset port only; no brief issued at op level)*

- **Tensor bindings** (per binding, across the PD-clean factories):
  - `multicore` (`MatmulMultiCoreProgramFactory`): in0/in1/output via `TensorAccessorArgs(...).append_to(reader/writer_compile_time_args)` (`matmul_multicore_program_factory.cpp:137,138,151`) → **Case 1** (kernel builds `TensorAccessor` from CTA-baked args; bind as `TensorParameter`). Kernels `reader_bmm_8bank_output_tiles_partitioned.cpp` / `writer_unary_interleaved_start_id.cpp` consume RTA address into `TensorAccessor` → Case 1.
  - `optimized` (`MatmulMultiCoreReuseOptimizedProgramFactory`): `emplace_runtime_args(core, {in0_buffer, ...})` `Buffer*`-binding form (`matmul_multicore_reuse_optimized_program_factory.cpp:413`) → **Case 2** (kernel consumes raw `uint32_t` base; framework-patched today; bind as `TensorParameter`, bridge via `get_bank_base_address`). in1/output via `TensorAccessorArgs` CTA (`:287-288`) → Case 1. Kernels are dataflow (bridge available).
  - `mcast_2d` (`MatmulMultiCoreReuseMcast2DProgramFactory`): `in0_tensor.address()`/`in1_tensor.address()`/`out_tensor.address()` into PD `runtime_args` (`:1194,1246,1259,...`), consumed via `TensorAccessor` → **Case 1**.
  - `mcast_1d` non-gather PD path: `in0/in1/out_tensor.address()` into PD `runtime_args` (`matmul_multicore_reuse_mcast_1d_program_factory.cpp:3774,3813,3826,...`), consumed via `TensorAccessor` → **Case 1**.
  - `mcast_dram_sharded` (`MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory`): `in1_tensor.address()` + `bias->address()` into RTA ("will be replaced by Buffer*", `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:763-764`); kernel `reader_bmm_tile_layout_in1_sender_dram_sharded.cpp` walks DRAM banks via `AllocatorBank` from the RTA base (`:22,24`) → **Case 2** (dataflow → bridge available). in0 reads a resident sharded CB → clean.
  - `batched_hs` (`MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory`): kernel `reader_bmm_tile_layout_in1_sender_dram_sharded_height.cpp` — in1 + bias DRAM base from RTA via `AllocatorBank` (`:26,28`) → **Case 2** (dataflow → bridge available). in0 height-sharded reader pulls a remote-L1 shard → clean.
- **Custom hash:** **delete** `MatmulDeviceOperation::compute_program_hash` → default (sanctioned exception). `SparseMatmulDeviceOperation`: none (already default; commented-out declaration).

## Heads-ups  *(mirrors the brief — clean-subset only)*

- **Notable LANDED constructs:**
  - **Aliased CB** → `advanced_options.alias_with`: shared output+interm CB at `optimized:548-557`, `mcast_2d:1028-1037` (and imperative mcast_1d ~`:840`).
  - **Borrowed-memory DFB** → `borrowed_from`: `CBDescriptor::tensor = &sharded_tensor` in every sharded PD factory (sites listed in the feature table); the kernels reading these sharded CBs by L1 read-ptr (`reader_bmm_tile_layout_in0.cpp` IN0_SHARDED branch, `in0_sender_dram_sharded.cpp`) are the clean borrowed-DFB reads.
  - **Non-zero semaphore initial value** → `advanced_options.initial_value` (`[[deprecated]]`, Gen2-unsupported): `INVALID` sentinel in mcast_1d/mcast_2d PD `SemaphoreDescriptor` and imperative `CreateSemaphore` (sites in feature table). Expected; not a blocker on Gen1.
- **Fake CBs (address-only):** none.
- **Cross-op / shared kernels:** the matmul kernel pool (`device/kernels/`) is shared between `MatmulDeviceOperation` and `SparseMatmulDeviceOperation` (e.g. `reader_bmm_tile_layout_in0_receiver.cpp`, `reader_bmm_tile_layout_in0_sender_padding.cpp`, `bmm_large_block_zm_fused_bias_activation.cpp` are instantiated by both). They form a **port-together set** — any Metal 2.0 rewrite of these shared kernels must land in both DeviceOperations' factory updates together. The mcast_1d/mcast_2d/sparse imperative `*_helper` functions are also consumed by external CCL fusion ops (see Team-only coupling).
- **RTA varargs:** none observed (no `num_runtime_varargs`, no runtime-varying-index RTA loop).
- **TTNN factory analysis (porter-relevant):**
  - **Pybind `create_descriptor`** to delete: `matmul_nanobind.cpp:1213,1233,1249,1265,1282,1299` (six factory classes, each `nb::class_<...ProgramFactory>().def_static("create_descriptor", ...)` with the extra `core_range_set` param).
  - **Other risky pybind** to delete: `nb::class_<MatmulDeviceOperation>` with `create_output_tensors`/`compute_output_specs`/`compute_program_hash` (`matmul_nanobind.cpp:1192-1207`); `nb::class_<MatmulParams>`/`<MatmulInputs>` attribute structs (`:1173,1185`); `default_core_range` def (`:1226`).
  - **Custom `override_runtime_arguments`** present on: `MatmulMultiCoreReuseMcast1DProgramFactory` (`.hpp:28`), `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory` (`.hpp:52`), `MatmulMultiCoreReuseMcast2DProgramFactory` (`.hpp:34`), `SparseMatmulMultiCoreReuseMcast1DProgramFactory` (`.hpp:31,48`). (The fully-PD multicore/optimized/dram_sharded/batched_hs factories define no override hook.)

## Team-only

- **Out-of-directory coupling & donor shape:**
  - **Op-level roll-up: `✓ clean` (function-call escapes) + `⚠` (file-path: shared kernel pool, broadly co-borrowed).** No cross-family donor *kernel file* is instantiated; no donor function with a pre-Device-2.0 addr-gen signature (Shape 4) or `CircularBuffer&` signature is called. The only out-of-dir function-call escape is CCL `worker_sync_utils.hpp`.
  - **Summary table** (op kernel → donor):

    | Op kernel | Donor | Class | Shape |
    |---|---|---|---|
    | `reader_bmm_tile_layout_in0_sender_padding.cpp` | `ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp` | Cross-family (CCL) | header-only `MatmulOpReceiver` / `OpSignaler` helpers — ✓ no resource-handle shape concern |
    | `reader_bmm_tile_layout_in1_sender_writer_padding.cpp` | `ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp` | Cross-family (CCL) | same — ✓ |
    | `reader_bmm_8bank_output_tiles_partitioned.cpp`, `reader_bmm_tile_layout_in0.cpp`, `reader_bmm_tile_layout_in0_sender_dram_sharded.cpp` | `ttnn/operations/kernel_helper_functions/pad_tile.hpp` | shared utility pool (#4) | ✓ lib pool, handled by lib team |
    | all referenced kernels | `tt_metal/*` (`api/dataflow/*`, `api/compute/*`, `api/tensor/*`, `hostdevcommon/*`, `internal/*`, `ckernel*`) | LLK/HAL/firmware | ✓ no concern |
    | `bmm_large_block_zm_fused_bias_activation*.cpp`, `bmm_fused_activation.hpp` | `ttnn/operations/matmul/shared_with_host/activation_type.hpp`, `bmm_fused_activation.hpp` | in-family | ✓ port the family together |

  - **Host-side cross-op sharing (file-path / function-call escape):** the imperative `matmul_multi_core_reuse_mcast_1d_optimized_helper`, `matmul_multi_core_reuse_mcast_1d_optimized_`, and `create_program_mcast_in0_in1` are exported from the mcast_1d/mcast_2d factory headers and called by CCL fusion ops (e.g. `experimental/ccl/llama_all_gather_matmul_async/`). These are the legacy entry points; their ProgramDescriptor migration is coupled with the CCL ports, not just this op. Tracked as a port-together coordination item.
  - **Borrowed kernel files (file-path instantiation):** none borrowed from outside the matmul family. The matmul kernel pool is shared *internally* between the two matmul DeviceOperations (the port-together set noted in Heads-ups).

- **Relaxation candidates** (mined from `MatmulDeviceOperation::compute_program_hash` before deletion): **FALLIBLE — candidates to verify; default strict.** The custom hash keys on `attributes`, `factory.index()`, `input_tensor_a`, `input_tensor_b`, and each optional input/output tensor (`matmul_device_operation.cpp:2135-2148`). It does **not** distinguish padded vs logical shape, so no obvious `match_padded_shape_only` signal. **Anomaly (also a correctness note):** for the gather_in0 path the device-op carries *multiple* `b` weight tensors in `input_tensors` (indices ≥ 2), but the hash only incorporates `input_tensors.at(1)` — the extra weight tensors are absent from the key. Flag for the op owner; the default reflection-based hash (which keys on all tensor args) would fix this on deletion.

- **TTNN factory analysis** (six questions, `MatmulDeviceOperation` unless noted):
  1. **Op-owned tensors?** **No.** Outputs are produced via `compute_output_specs` + `create_output_tensors` (`matmul_device_operation.cpp:2122` `create_device_tensor` builds the declared *output*, not factory-owned scratch). Sparse: same (`sparse_matmul_device_operation.cpp:242`). No intermediate/scratch device tensors allocated in factories.
  2. **MeshWorkload needed?** **No (legacy artifact).** `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory` uses `create_mesh_workload`/`cached_mesh_workload_t` (`.hpp:44-46`), but `select_program_factory` routes to it only because `create_descriptor` does not yet support `gather_in0` (`matmul_device_operation.cpp:701` comment "gather_in0 uses the legacy MeshWorkload path"). No genuine cross-device coordination — it builds one program per mesh coord identically. Once the gather path gains a `create_descriptor`, it is morally single-program. (Open question for the user — see below.)
  3. **Pybind `create_descriptor`?** **Yes** — six factory classes (`matmul_nanobind.cpp:1210-1310`). Delete on port.
  4. **Other risky pybind?** **Yes** — `nb::class_<MatmulDeviceOperation>` exposing device-op methods, plus `MatmulParams`/`MatmulInputs` structs and `default_core_range` (`matmul_nanobind.cpp:1173-1207,1226`). Delete on port.
  5. **Custom hash?** **Yes** (`MatmulDeviceOperation` @ `matmul_device_operation.cpp:2127`). Treatment in Custom program hash subject (delete → default). Sparse: **No**.
  6. **Custom `override_runtime_arguments`?** **Yes** — mcast_1d, mcast_1d-MeshWorkload, mcast_2d factories, and sparse factory (sites in Heads-ups). The fully-PD multicore/optimized/dram_sharded/batched_hs factories: **No**.

## Misc anomalies  *(team-only, non-gating)*

- **Custom-hash omits extra gather weights:** `MatmulDeviceOperation::compute_program_hash` hashes only `input_tensors.at(1)` for `b`, not the additional weight tensors present in the gather_in0 path (`matmul_device_operation.cpp:2130-2136`). Latent cache-collision risk; resolved on hash deletion. Routes to op owner.
- **`allowed_worker_cores` auto-population warnings:** multiple factories `log_warning` and auto-populate `program_config.allowed_worker_cores` when callers bypass `normalize_program_config()` ("will become a hard error in a future release") — e.g. `matmul_multicore_reuse_mcast_1d_program_factory.cpp:5163-5175,5312-5322,5371-5382`. Not port work; flagged for the op owner's planned hard-error transition.

## Per-DeviceOperation attribution

| DeviceOperation | ProgramDescriptor | Device 2.0 | GlobalCircularBuffer | Custom hash | Overall |
|---|---|---|---|---|---|
| `MatmulDeviceOperation` (5 PD factories + mcast_1d non-gather PD path) | Yes | Yes-with-holdover (shared receiver kernel) | N/A (clean path) | Yes → delete | **GREEN-feasible subset** (blocked only by D2.0 holdover) |
| `MatmulDeviceOperation` — `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory` (gather_in0) | **No** (imperative) | Yes-with-holdover | **RED** (global_cb path) | (shared) | **RED** |
| `SparseMatmulDeviceOperation` | **No** (imperative) | Yes-with-holdover (shares receiver kernel) | **RED** (global_cb path) | No (default) | **RED** |

## Questions for the user

1. **Scoped-subset port:** Do you want a scoped port of the clean subset (the 5 fully-PD `MatmulDeviceOperation` factories + the non-gather, non-global_cb mcast_1d `create_descriptor` path), leaving the gather_in0 MeshWorkload path and `SparseMatmulDeviceOperation` on their current paths until the ProgramDescriptor migration + GlobalCircularBuffer feature land? The shared `reader_bmm_tile_layout_in0_receiver.cpp` Device-2.0 holdover would need clearing first regardless. (Context: `matmul_device_operation.cpp:701-705`.)
2. **MeshWorkload classification (Q2):** I classified the gather_in0 MeshWorkload path as a *legacy artifact* (not a genuine multi-program need) — it builds one identical program per mesh coord and exists only because `create_descriptor` doesn't yet support gather. Please confirm there is no genuine cross-device coordination in the gather path that would make it a true MeshWorkload need. (Context: `matmul_multicore_reuse_mcast_1d_program_factory.cpp:5358-5425`.)

## Recipe notes

- **Mixed-API single factory file.** The mcast_1d and mcast_2d *files* each contain both a live PD `create_descriptor` path **and** a legacy imperative `*_helper`/`create_program_mcast_*` path that is exported for external CCL callers (not reached by this op's device-op). The recipe's ProgramDescriptor check ("op uses ProgramDescriptor vs imperative") is framed per-op/per-factory, but here the *same file* and even the *same factory struct* (`MatmulMultiCoreReuseMcast1DProgramFactory` has both `create_descriptor` PD and exported imperative helpers) straddles both. I resolved it by gating on the **live device-op dispatch path** (`select_program_factory` → `create_descriptor` / `create_mesh_workload`) and recording the exported imperative helpers as cross-op coupling. A note in the ProgramDescriptor-check section about "judge by the live device-op path, not by any imperative symbol present in the file" would help the next auditor of a CCL-shared factory.
