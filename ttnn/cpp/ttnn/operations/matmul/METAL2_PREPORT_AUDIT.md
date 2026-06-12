# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/matmul`

This directory bundles two DeviceOperations:

- **`MatmulDeviceOperation`** (primary — `device/matmul_device_operation.hpp/.cpp`)
  - `MatmulMultiCoreProgramFactory` (`factory/matmul_multicore_program_factory.cpp`)
  - `MatmulMultiCoreReuseOptimizedProgramFactory` (`factory/matmul_multicore_reuse_optimized_program_factory.cpp`)
  - `MatmulMultiCoreReuseMcast1DProgramFactory` (`factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp`)
  - `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory` (`factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp` — legacy path, gather_in0 only)
  - `MatmulMultiCoreReuseMcast2DProgramFactory` (`factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp`)
  - `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory` (`factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp`)
  - `MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory` (`factory/matmul_multicore_reuse_batched_hs_dram_sharded_program_factory.cpp`)
- **`SparseMatmulDeviceOperation`** (secondary — `device/sparse/sparse_matmul_device_operation.hpp`)
  - `SparseMatmulMultiCoreReuseMcast1DProgramFactory` (`device/sparse/sparse_matmul_device_operation.hpp`)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/matmul` |
| **Overall** | RED |
| **DOps / Factories** | `MatmulDeviceOperation` → 7 factories (5 PD, 1 MeshWorkload-legacy, 1 imperative fragment); `SparseMatmulDeviceOperation` → 1 factory (legacy imperative) |
| *Prereqs* — ProgramDescriptor | Yes (for all `MatmulDeviceOperation` non-gather_in0 paths); No (for `SparseMatmulDeviceOperation` and the gather_in0 legacy path in `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory`) |
| *Prereqs* — Device 2.0 (every kernel used) | Yes (all PD-path kernels are D2.0-clean) |
| *Prereqs* — Cross-op escapes | Ok — the `worker_sync_utils.hpp` CCL include has legacy functions, but matmul kernels call only `MatmulOpReceiver` (no legacy noc calls exercised) |
| *Feature Support* — overall | RED |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — Op-owned tensors | No (output tensor created in `create_output_tensors` is the standard declaration; no scratch/intermediate op-owned tensors) |
| *TTNN Readiness* — MeshWorkload needed | No (op-owned-tensor artifact only) — `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory` exists solely as a plumbing artifact for the legacy gather_in0 path, not for genuine cross-device coordination |
| *TTNN Readiness* — Pybind `create_descriptor` | Yes: `matmul_nanobind.cpp:1205-1305` — all 6 PD-factory classes pybbound with `create_descriptor` |
| *TTNN Readiness* — Other risky pybind | Yes: `MatmulInputs` struct pybound (`matmul_nanobind.cpp:1185`); `MatmulDeviceOperation` pybound with `create_output_tensors` + `compute_output_specs` (`matmul_nanobind.cpp:1192-1202`); `select_program_factory` exposed as `matmul_select_program_factory` (`matmul_nanobind.cpp:1308`); `create_matmul_attributes` helper (`matmul_nanobind.cpp:1315`) |
| *TTNN Readiness* — Custom hash | Yes → delete (see Custom program hash) — `matmul_device_operation.cpp:2029` |
| *TTNN Readiness* — Custom override-RTA | Yes: `MatmulMultiCoreReuseMcast1DProgramFactory` (`matmul_multicore_reuse_mcast_1d_program_factory.cpp:5162`), `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory` (`matmul_multicore_reuse_mcast_1d_program_factory.cpp:5512`), `MatmulMultiCoreReuseMcast2DProgramFactory` (`matmul_multicore_reuse_mcast_2d_program_factory.cpp:3315`) |
| *TTNN Readiness* — Fake CBs (address-only) | None observed in PD-path factories |

## Result

**RED — blocked on GlobalCircularBuffer (UNSUPPORTED feature).**

`MatmulParams` carries `std::optional<tt::tt_metal::experimental::GlobalCircularBuffer> global_cb` (`matmul_device_operation_types.hpp:35`). Metal 2.0 does not yet support `GlobalCircularBuffer` on `KernelSpec` / `DataflowBufferSpec`. The port is blocked until GlobalCircularBuffer support lands in the Metal 2.0 framework.

**Scoped subset note:** The five non-gather_in0, non-GCB PD factories — `MatmulMultiCoreProgramFactory`, `MatmulMultiCoreReuseOptimizedProgramFactory`, `MatmulMultiCoreReuseMcast2DProgramFactory`, `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory`, `MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory` — never touch `global_cb` in their `create_descriptor` paths (the 1D factory explicitly `TT_FATAL`s on gather_in0 at line 5205). These paths are structurally clear of the GCB blocker. A scoped port of those five factories (omitting the 1D + MeshWorkload gather_in0 path) would be feasible once the GCB field is decoupled from `MatmulParams`. Route this to the team for a possible partial port.

**Secondary prerequisites gap (also gated):** `SparseMatmulDeviceOperation` is on the legacy imperative API — it does not implement `create_descriptor`. It blocks separately on the ProgramDescriptor prerequisite.

## Gate detail

### ProgramDescriptor

**GREEN (with exceptions)** for the `MatmulDeviceOperation` non-gather_in0 paths. All five non-1D factories and `MatmulMultiCoreReuseMcast1DProgramFactory` implement `create_descriptor` returning `ProgramDescriptor`. Confirmed via inspection of all factory `.hpp` / `.cpp` files.

**RED** for:
- `SparseMatmulDeviceOperation` / `SparseMatmulMultiCoreReuseMcast1DProgramFactory` — uses `create` + `CachedProgram<>` pattern; no `create_descriptor`. Requires a separate ProgramDescriptor migration effort; expected outcome for a legacy op.
- The gather_in0 code path handled by `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory::create_mesh_workload` — uses the legacy mesh-workload path. The `create_descriptor` in `MatmulMultiCoreReuseMcast1DProgramFactory` explicitly `TT_FATAL(!gather_in0)` at line 5205. This path is the known in-flight work.

These are expected blockers for legacy paths; unblock by landing their ProgramDescriptor migrations.

### Device 2.0 (every kernel used)

**GREEN** for all kernels used by PD-path factories. Every kernel file under `device/kernels/dataflow/` and `device/kernels/compute/` includes `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/tensor/noc_traits.h` and uses `Noc noc`, `TensorAccessor`, `CircularBuffer cb_*(id)` — the Device 2.0 wrappers. No `InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedAddrGenFast`, or raw `noc_async_read` / `noc_async_write` calls found in any PD-path kernel.

Notable findings:
- `writer_unary_interleaved_start_id.cpp:19` uses `get_local_cb_interface(cb_id_out).fifo_page_size` — this is a **sanctioned** Device 2.0 free function, not a holdover.
- `reader_bmm_tile_layout_in0_ring_all_gather.cpp` and `reader_bmm_tile_layout_in1_ring_all_gather.cpp` are used only by the legacy gather_in0 imperative path, not by any PD factory.
- Cross-op include `ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp` (in 4 kernels) contains legacy `noc_semaphore_set` / `get_noc_addr` calls, but matmul kernels use only the `MatmulOpReceiver` struct. Those legacy functions are dead code from matmul's perspective. Comment in that file: `TODO(#45846): refactor to take an id-based interface.`

### Feature compatibility

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | RED | `MatmulParams::global_cb` field + helper function signatures in `matmul_multicore_reuse_mcast_1d_program_factory.hpp:70-88`; also `matmul_device_operation.cpp:13` include |
| Dynamic CircularBuffer (borrowed memory) | N/A | No borrowed-memory DFB pattern in PD-path factories |
| CBDescriptor `address_offset` (non-zero) | N/A | No `address_offset` usage found |
| Aliased Circular Buffers | GREEN | Present in `matmul_multicore_reuse_mcast_1d_program_factory.cpp:3687-3705`, `:4581-4598` and `matmul_multicore_reuse_mcast_2d_program_factory.cpp:1026-1044` — single `CBDescriptor` with two `CBFormatDescriptor` entries (c_4 + c_5 sharing buffer); port uses `DataflowBufferSpec::advanced_options.alias_with` |
| GlobalSemaphore | N/A | No `GlobalSemaphore` usage |
| Non-zero semaphore initial value | N/A | Semaphore descriptors use `initial_value = INVALID`; `INVALID = 0` per `common_values.hpp:13` |
| Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | No `ArgConfig::Runtime*` usage |
| `UpdateCircularBuffer*` | N/A | No `UpdateCircularBuffer` calls |
| Variable-count compile-time arguments (CTA varargs) | N/A | Fixed-count CTAs throughout; `tensor_args_t = MatmulInputs` with fixed-count vectors; CTA arrays indexed by CTA-known `num_x`/`num_y` counts, not runtime-variable |

#### GlobalCircularBuffer (RED)

**Signal:**
- `device/matmul_device_operation_types.hpp:9` — `#include "tt-metalium/global_circular_buffer.hpp"`
- `device/matmul_device_operation_types.hpp:35` — `std::optional<tt::tt_metal::experimental::GlobalCircularBuffer> global_cb = std::nullopt` in `MatmulParams`
- `device/factory/matmul_multicore_reuse_mcast_1d_program_factory.hpp:70-73,87-88` — helper function signatures take `const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb`
- `device/matmul_device_operation.cpp:13` — `#include "tt-metalium/experimental/global_circular_buffer.hpp"`
- `device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp:2160` — `tt_metal::experimental::CreateCircularBuffer(program, all_cores, remote_cb_config, *global_cb)` in the legacy path

**Scope:** `global_cb` is only consumed in the legacy gather_in0 path (`process_gather_in0_program_and_create_override_variables`). The PD `create_descriptor` implementations for all five non-1D factories never touch `global_cb`. `MatmulMultiCoreReuseMcast1DProgramFactory::create_descriptor` explicitly `TT_FATAL(!gather_in0)` (line 5205), so the non-gather_in0 PD path never exercises GCB. **However**, `GlobalCircularBuffer` is part of `MatmulParams` (the op-level attributes type), so the Metal 2.0 framework will see the type at the op boundary even if the non-gather_in0 factories don't exercise it.

**Expected resolution:** GlobalCircularBuffer support is not yet in Metal 2.0 (`KernelSpec` / `DataflowBufferSpec`). The port becomes possible for the non-gather_in0 subset once either (a) GCB support lands in Metal 2.0, or (b) the `global_cb` field is moved out of `MatmulParams` into a gather_in0-specific struct / factory. The latter is a refactor that could unblock the five clean factories immediately. Route to team.

## Port-work summary *(mirrors the brief — N/A since RED)*

- **Tensor bindings:** All PD-path factories use `MeshTensor` references in `emplace_runtime_args` (Case 1 — re-express via `TensorParameter` / `TensorBinding`). Full inventory below.
  - `MatmulMultiCoreProgramFactory`: `a`, `b`, `output` — Case 1 (`matmul_multicore_program_factory.cpp:174-188`)
  - `MatmulMultiCoreReuseOptimizedProgramFactory`: `in0_buffer`, `in1_buffer`, `output` — Case 1 (`matmul_multicore_reuse_optimized_program_factory.cpp:405,409`)
  - `MatmulMultiCoreReuseMcast1DProgramFactory`: multiple `MeshTensor` ref variants per core — Case 1 (`matmul_multicore_reuse_mcast_1d_program_factory.cpp:3832-3858,3934-3942`)
  - `MatmulMultiCoreReuseMcast2DProgramFactory`: `in0_tensor`, `in1_tensor`, `out_tensor`, bias — Case 1 (`matmul_multicore_reuse_mcast_2d_program_factory.cpp:1241-1410`)
  - `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory`: `in1_tensor` (+ optional bias) — Case 1 (`matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:870-876`); `in0` via shard path (no explicit addr RTA for in0 sender side)
  - `MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory`: `in0_tensor`, `in1_tensor`, `out_tensor` — Case 1 (`matmul_multicore_reuse_batched_hs_dram_sharded_program_factory.cpp:563-579`)
- **Custom hash:** delete `MatmulDeviceOperation::compute_program_hash` at `matmul_device_operation.cpp:2029` → default (sanctioned exception)

**Note on dram_sharded kernel:** `reader_bmm_tile_layout_in1_sender_dram_sharded.cpp:22` reads `in1_tensor_addr = get_arg_val<uint32_t>(1)` and uses it via `AllocatorBank<DRAM>` with a raw bank_id + addr pattern (no `TensorAccessor`). This is Case 1 — the factory now passes `in1_tensor` reference at position [1] of the variant args (line 872 of the dram_sharded factory). The kernel-side address arithmetic will need to be re-expressed or bridged via `TensorAccessor::get_bank_base_address` at port time. **Do not self-classify as Case 2** — this is likely addressable via the standard bridge mechanism; user confirmation recommended.

## Heads-ups *(mirrors the brief — N/A since RED)*

- **Aliased CBs (LANDED):** `matmul_multicore_reuse_mcast_1d_program_factory.cpp:3687-3705,4581-4598` and `matmul_multicore_reuse_mcast_2d_program_factory.cpp:1026-1044` — single `CBDescriptor` with two `CBFormatDescriptor` entries sharing one buffer (c_4 = output, c_5 = interm0). Port uses `DataflowBufferSpec::advanced_options.alias_with`.

- **Cross-op kernel includes:** Four PD-path kernels include `ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp` for `MatmulOpReceiver`. This file has legacy functions (not called from matmul paths). The `MatmulOpReceiver` struct takes `uint32_t sem_id` and `uint32_t cb_id` — workable in Metal 2.0. See Out-of-directory coupling for donor shape detail.

- **Custom `override_runtime_arguments`:** Three factories define it — `MatmulMultiCoreReuseMcast1DProgramFactory` (line 5162), `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory` (line 5512), `MatmulMultiCoreReuseMcast2DProgramFactory` (line 3315). These are device-op-class edits during the port.

- **Migration-risky pybind:** Extensive (see TTNN factory analysis). All six PD factories are pybbound via `create_descriptor`; `MatmulDeviceOperation` is pybbound with `create_output_tensors`, `compute_output_specs`; `MatmulInputs` is pybbound; `select_program_factory` and `create_matmul_attributes` are exposed as module-level functions. All require deletion / migration during the port.

- **gather_in0 / MeshWorkload legacy path:** `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory` is a plumbing artifact for the gather_in0 path. It is NOT a genuine MeshWorkload need. Ports of the clean PD factories should treat this factory as out-of-scope (separate migration track).

## Team-only

### TensorAccessor convertibility — dram_sharded factory (Case 1)

`reader_bmm_tile_layout_in1_sender_dram_sharded.cpp` uses `AllocatorBank<DRAM>` with `{.bank_id = dram_bank_id, .addr = in1_tensor_addr}` (lines 77-79). The host factory now passes `in1_tensor` MeshTensor ref in the variant args (dram_sharded factory line 872), which registers the buffer binding. The kernel side still manually walks bank_id + addr. Classification: **Case 1** (the binding channel is in place; the kernel-side address arithmetic needs conversion to `TensorAccessor` or the bridge pattern). This is awkward-but-convertible, not genuinely exotic — the DRAM bank walk pattern is well-understood. Candidate for a future `TensorAccessor` enhancement; issue recommended.

### Out-of-directory coupling and donor shape

**Op-level roll-up:** ⚠ workable — cross-op includes are present but all have workable Metal 2.0 shapes.

**Summary table:**

| Op kernel | Donor file | Donor class | Status |
|---|---|---|---|
| `reader_bmm_tile_layout_in0.cpp` | `ttnn/operations/kernel_helper_functions/pad_tile.hpp` | shared utility pool | ✓ clean — pure computation, no resource handles |
| `reader_bmm_8bank_output_tiles_partitioned.cpp` | `ttnn/operations/kernel_helper_functions/pad_tile.hpp` | shared utility pool | ✓ clean |
| `reader_bmm_tile_layout_in0_sender_dram_sharded.cpp` | `ttnn/operations/kernel_helper_functions/pad_tile.hpp` | shared utility pool | ✓ clean |
| `reader_bmm_tile_layout_in0_sender_padding.cpp` | `ttnn/operations/kernel_helper_functions/pad_tile.hpp` | shared utility pool | ✓ clean |
| `reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp` | `ttnn/operations/kernel_helper_functions/pad_tile.hpp` | shared utility pool | ✓ clean |
| `reader_bmm_tile_layout_in0_sender_padding.cpp` | `ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp` | cross-family donor (CCL) | ⚠ `MatmulOpReceiver(uint32_t sem_id, uint32_t cb_id)` — `uint32_t sem_id` is suboptimal but workable; `uint32_t cb_id` maps cleanly |
| `reader_bmm_tile_layout_in1_sender_writer_padding.cpp` | `ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp` | cross-family donor (CCL) | ⚠ same as above |
| `reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp` | `ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp` | cross-family donor (CCL) | ⚠ same |
| `reader_bmm_tile_layout_in1_receiver_writer_padding.cpp` | `ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp` | cross-family donor (CCL) | ⚠ same |

**Per-call detail — `worker_sync_utils.hpp`:**

The only function matmul kernels call from this file is the `MatmulOpReceiver` constructor and its member methods. The struct takes `uint32_t sem_id` and `uint32_t cb_id`. Under Metal 2.0, `sem::name` provides a constexpr cast to `uint32_t` and `dfb::name` provides a constexpr cast to `uint32_t` — both handle these parameter shapes. The `master_sync_slaves` and `slave_sync_with_master` functions in `worker_sync_utils.hpp` use legacy `noc_semaphore_set` / `get_noc_addr` but are NOT called by any matmul kernel — they are dead code from matmul's perspective.

The CCL file has a TODO comment (issue #45846) to refactor to id-based interface. Until that lands, the `uint32_t sem_id` path is the workable path. This is not a blocker.

**Borrowed kernel files (file-path kernel instantiation):**

Matmul owns all its kernel `.cpp` files under `device/kernels/`. No cross-family file-path borrowing detected in the PD-path factories. The helper header `pad_tile.hpp` in `kernel_helper_functions/` is an in-tree shared utility (not a borrowed kernel file in the `.cpp` sense). No port-together set needed beyond matmul itself.

### Relaxation candidates (mined from custom hash — FALLIBLE, candidates only)

The custom `compute_program_hash` at `matmul_device_operation.cpp:2029-2052`:
1. Hashes `attributes` + `factory.index()` + full `input_tensor_a` + `input_tensor_b` + optional tensors.
2. The explicit `factory.index()` bake-in signals the author wanted factory dispatch to be part of the hash key. Under the default hash, factory selection flows from the spec, so this is implicit.
3. No obvious shape-only relaxation candidates — the hash is conservative and roughly correct in structure.
4. **Potential anomaly:** The hash hashes optional input tensors only when they `has_value()` — skips `nullopt` bias. This means cache hits for two calls where one has bias and one doesn't would require the optional-present flag to differ in `attributes` (which likely includes the program config with bias flag). Verify this doesn't cause a subtle miss when optional bias presence changes between invocations with otherwise identical attributes.

Default strict; no relaxation recommended without explicit verification.

### TTNN factory concept analysis

**Q1 — Op-owned tensors:** No. `create_output_tensors` in `matmul_device_operation.cpp:2024` creates the declared output tensor via `create_device_tensor` — that is the standard output-tensor allocation, not an op-owned scratch tensor. No intermediate tensors allocated in any factory.

**Q2 — MeshWorkload concept needed:** No (op-owned-tensor artifact only). `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory` has `create_mesh_workload` (`matmul_multicore_reuse_mcast_1d_program_factory.cpp:5443`) but it is a plumbing artifact for the legacy gather_in0 path. `select_program_factory` dispatches gather_in0 to this factory specifically because the `create_descriptor` path `TT_FATAL`s on gather_in0 (line 5205). There is no genuine cross-device coordination requirement. The five clean PD factories need no MeshWorkload concept.

**Q3 — Pybind `create_descriptor`:** Yes. Six factories pybbound with `create_descriptor` in `matmul_nanobind.cpp`:
- `MatmulMultiCoreReuseOptimizedProgramFactory::create_descriptor` (line 1205-1223, note extra `default_core_range` also pybbound)
- `MatmulMultiCoreProgramFactory::create_descriptor` (line 1225-1239)
- `MatmulMultiCoreReuseMcast1DProgramFactory::create_descriptor` (line 1241-1255)
- `MatmulMultiCoreReuseMcast2DProgramFactory::create_descriptor` (line 1257-1271)
- `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory::create_descriptor` (line 1273-1288)
- `MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory::create_descriptor` (line 1290-1305)

The port deletes all six. Sanctioned device-op-class edit per `port_op_to_metal2_ttnn_factory.md`.

**Q4 — Other migration-risky pybind:** Yes. Several DeviceOperation internals are exposed:
- `nb::class_<ttnn::prim::MatmulInputs>` pybbound with `input_tensors`, `optional_input_tensors`, `optional_output_tensors` (`matmul_nanobind.cpp:1185-1189`)
- `nb::class_<ttnn::prim::MatmulDeviceOperation>` pybbound with `create_output_tensors` and `compute_output_specs` (`matmul_nanobind.cpp:1192-1202`)
- `matmul_select_program_factory` — exposes `MatmulDeviceOperation::select_program_factory` as a module-level function (`matmul_nanobind.cpp:1308-1312`)
- `create_matmul_attributes` helper exposed as a module-level function (`matmul_nanobind.cpp:1315-1321`)

These represent Python-accessible introspection into the DeviceOperation and factory dispatch internals. All are migration-risky and require deletion or migration during the port.

**Q5 — Custom hash:** Yes → delete. `matmul_device_operation.cpp:2029`. Cross-reference: Custom program hash section above.

**Q6 — Custom `override_runtime_arguments`:** Yes:
- `MatmulMultiCoreReuseMcast1DProgramFactory::override_runtime_arguments` (`matmul_multicore_reuse_mcast_1d_program_factory.cpp:5162`)
- `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory::override_runtime_arguments` (`matmul_multicore_reuse_mcast_1d_program_factory.cpp:5512`, delegates to the above)
- `MatmulMultiCoreReuseMcast2DProgramFactory::override_runtime_arguments` (`matmul_multicore_reuse_mcast_2d_program_factory.cpp:3315`, delegates to `override_runtime_arguments_impl` at line 2989)

All three represent device-op-class edits at port time.

## Misc anomalies

- **`SparseMatmulDeviceOperation` in scope:** This device operation is in the same directory but is entirely on the legacy API — `create` + `CachedProgram<>` — with no `create_descriptor`. It also has its own `GlobalCircularBuffer` usage. It should be tracked as a separate migration item; its presence here does not affect the `MatmulDeviceOperation` audit, but planners should be aware the two are bundled.
- **`gather_in0` path and `GlobalCircularBuffer` coupling:** The `global_cb` field in `MatmulParams` is solely exercised by the gather_in0 legacy path. Decoupling `global_cb` from the main `MatmulParams` struct (e.g., into a gather_in0-specific config sub-struct) would unblock the five clean PD factories from the GCB gate. Worth tracking as a prerequisite refactor.

## Questions for the user

1. **GCB decoupling refactor:** The five clean PD factories (`MatmulMultiCoreProgramFactory`, `MatmulMultiCoreReuseOptimizedProgramFactory`, `MatmulMultiCoreReuseMcast2DProgramFactory`, `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory`, `MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory`) never touch `global_cb`. Would the team accept decoupling `global_cb` from `MatmulParams` as a prerequisite refactor to unblock a scoped partial port of those five factories? Or should the port wait for full Metal 2.0 GCB support?

2. **dram_sharded kernel tensor accessor:** `reader_bmm_tile_layout_in1_sender_dram_sharded.cpp` uses `AllocatorBank<DRAM>` with raw `addr` + `bank_id` (no `TensorAccessor`). This has been classified as Case 1 (re-express), but the bank-id walk pattern is unusual. Please confirm the intent: is this straightforward to express via `TensorAccessor`, or does the bank_id selection require the `get_bank_base_address` bridge (Case 1 bridge approach)?

## Recipe notes

- The "MeshWorkload false-positive trap" note in the recipe is well-targeted — `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory` would easily be misread as a genuine MeshWorkload need without the recipe's guidance. The recipe's explicit framing ("gather_in0 legacy path") matches what the dispatch code says.
- The recipe's "do not self-classify into Case 2" rule applied to the dram_sharded `AllocatorBank<DRAM>` pattern. This is left as a user question per the rule.
