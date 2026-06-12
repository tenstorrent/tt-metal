# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/paged_cache`

This directory bundles three independent DeviceOperations sharing a common op family. They do not share program factories or kernels with each other, but share the audit here per the recipe's "bundle closely-related ops in one directory" guidance. Per-DeviceOperation attribution is preserved throughout.

- **`PagedFillCacheDeviceOperation`**
  - `PagedFillCacheProgramFactory` (`device/fill_cache/paged_fill_cache_program_factory.cpp`)
  - `PagedFillCacheMeshWorkloadFactory` (`device/fill_cache/paged_fill_cache_program_factory.cpp`)
- **`PagedUpdateCacheDeviceOperation`**
  - `PagedUpdateCacheProgramFactory` (`device/update_cache/paged_update_cache_program_factory.cpp`)
  - `PagedUpdateCacheMeshWorkloadFactory` (`device/update_cache/paged_update_cache_program_factory.cpp`)
- **`PagedFusedUpdateCacheDeviceOperation`**
  - `PagedTiledFusedUpdateCacheProgramFactory` (`device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp`)
  - `PagedTiledFusedUpdateCacheMeshWorkloadFactory` (`device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp`)
  - `PagedRowMajorFusedUpdateCacheProgramFactory` (`device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp`)
  - `PagedRowMajorFusedUpdateCacheMeshWorkloadFactory` (`device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp`)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/paged_cache` |
| **Overall** | GREEN |
| **DOps / Factories** | `PagedFillCacheDeviceOperation` → `PagedFillCacheProgramFactory`, `PagedFillCacheMeshWorkloadFactory`; `PagedUpdateCacheDeviceOperation` → `PagedUpdateCacheProgramFactory`, `PagedUpdateCacheMeshWorkloadFactory`; `PagedFusedUpdateCacheDeviceOperation` → `PagedTiledFusedUpdateCacheProgramFactory`, `PagedTiledFusedUpdateCacheMeshWorkloadFactory`, `PagedRowMajorFusedUpdateCacheProgramFactory`, `PagedRowMajorFusedUpdateCacheMeshWorkloadFactory` |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | Yes |
| *Prereqs* — Cross-op escapes | Ok |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | N/A |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — MeshWorkload needed | No (mesh-coord-filtering plumbing artifact only; no genuine multi-program or cross-device coordination) |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | Yes → delete (see Custom program hash) — all three DeviceOperations |
| *TTNN Readiness* — Custom override-RTA | No (`get_dynamic_runtime_args` used instead, which is a different interface and not `override_runtime_arguments`) |
| *TTNN Readiness* — Fake CBs (address-only) | None |

## Result

**GREEN → brief issued.** All gates cleared. The port can proceed after explicit user go-ahead. Three custom `compute_program_hash` overrides must be deleted as part of the port (PORT WORK). Multiple tensor bindings use the `Buffer*`-binding form (Case 1 re-express) and two factories use aliased CBs (LANDED, FYI-P). All kernels are Device 2.0 compliant. No UNSUPPORTED features in use.

## Gate detail

- **ProgramDescriptor:** GREEN — all three DeviceOperations and all factories include `<tt-metalium/program_descriptors.hpp>` and build `ProgramDescriptor` via `CBDescriptor`, `KernelDescriptor`, `SemaphoreDescriptor`, and `emplace_runtime_args`. No imperative `host_api.hpp` usage found.

- **Device 2.0 (every kernel used):** GREEN — all eight kernel files (6 dataflow, 2 compute) use Device 2.0 idioms throughout: `Noc noc;`, `CircularBuffer cb_...(cb_id)` wrappers, `TensorAccessor(args, addr)`, `Semaphore<>(sem_id)`. No legacy `InterleavedAddrGen`, `ShardedAddrGen`, raw `noc_async_read`, raw semaphore addresses, or manual CB-index management found in any kernel.

- **Feature compatibility:**

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | Not used in any factory or kernel |
  | Dynamic CircularBuffer (borrowed memory) | GREEN | Multiple `CBDescriptor` with `.buffer = non-null` — see Heads-ups for sites |
  | CBDescriptor `address_offset` (non-zero) | N/A | `.address_offset` not set on any `CBDescriptor` |
  | Aliased Circular Buffers | GREEN | Two factories have a `CBDescriptor` with two `CBFormatDescriptor` entries — see Heads-ups |
  | GlobalSemaphore | N/A | Not used |
  | Non-zero semaphore initial value | N/A | All `SemaphoreDescriptor` use `.initial_value = 0` explicitly |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | No `ArgConfig::Runtime` token found in any factory |
  | `UpdateCircularBuffer*` | N/A | No calls to `UpdateCircularBufferTotalSize`, `UpdateCircularBufferPageSize`, or `UpdateDynamicCircularBufferAddressAndTotalSize` |
  | Variable-count compile-time arguments (CTA varargs) | N/A | All `tensor_args_t` structs carry a fixed set of named `Tensor` fields; no `std::vector<Tensor>`; no `get_compile_time_arg_val(i)` loop |

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, all three DeviceOperations):
  - `PagedFillCacheDeviceOperation`:
    - `input_tensor` (src_buffer) — **Case 1** (re-express). Factory: `reader_desc.emplace_runtime_args(core, {src_buffer, ...})` + `TensorAccessorArgs(src_buffer).append_to(reader_compile_time_args)`. Kernel: `TensorAccessor(src_args, src_addr)` where `src_addr = get_arg_val<uint32_t>(0)`.
    - `cache_tensor` (dst_buffer) — **Case 1** (re-express). Factory: `writer_args.push_back(dst_buffer)` + `TensorAccessorArgs(dst_buffer).append_to(writer_compile_time_args)`. Kernel: `TensorAccessor(s0_args, dst_addr)`.
    - `page_table` (page_table_buffer) — **Case 1** (re-express). Factory: `writer_args.push_back(page_table_buffer)` + `TensorAccessorArgs(page_table_buffer).append_to(writer_compile_time_args)`. Kernel: `TensorAccessor(page_table_args, page_table_addr)`.
    - `batch_idx_tensor` (optional, batch_idx_tensor->buffer()) — **Case 1** (re-express). Factory: `writer_args.push_back(batch_idx_tensor->buffer())` + `TensorAccessorArgs(batch_idx_tensor->buffer()).append_to(writer_compile_time_args)`. Kernel: `TensorAccessor(batch_idx_tensor_args, batch_arg)`.
  - `PagedUpdateCacheDeviceOperation`:
    - `cache_tensor` (dst_buffer) — **Case 1** (re-express). Factory: `rargs.push_back(dst_buffer)` (reader + writer) + `TensorAccessorArgs(dst_buffer).append_to(reader_compile_time_args)` + `TensorAccessorArgs(dst_buffer).append_to(writer_compile_time_args)`. Kernels: `TensorAccessor(s0_args, cache_addr)`.
    - `input_tensor` (in1_buffer) — borrowed-memory DFB (`CBDescriptor{ .buffer = in1_buffer }`). Kernel reads via CB pointer (`cb_input` borrowed-memory sharded CB). **clean** (borrowed-memory DFB; causal-link gate applies — sharded input tensor CB has sharding as the producer/consumer mechanism).
    - `update_idxs_tensor` (optional, index_buffer_for_rt) — **Case 1** (re-express). Factory: `rargs.push_back(index_buffer_for_rt)`. Kernel: `TensorAccessor(index_tensor_args, index_tensor_addr)`.
    - `page_table` (optional, page_table_buffer_for_rt) — **Case 1** (re-express). Factory: `rargs.push_back(page_table_buffer_for_rt)`. Kernel: `TensorAccessor(page_table_args, page_table_tensor_addr)`.
  - `PagedFusedUpdateCacheDeviceOperation` (both tiled and row-major factories):
    - `cache_tensor1` (dst1_buffer) — **Case 1** (re-express). Factory: `rargs.push_back(dst1_buffer)` + `TensorAccessorArgs(dst1_buffer).append_to(compile_time_args)`. Kernel: `TensorAccessor(s0_args, cache_addr)`.
    - `cache_tensor2` (dst2_buffer) — **Case 1** (re-express). Factory: `rargs.push_back(dst2_buffer)` for core2. No separate TensorAccessorArgs — uses the same `s0_args` CTA slot as `cache_tensor1` on the writer side.
    - `input_tensor1` (in1_buffer) — borrowed-memory DFB. **clean**.
    - `input_tensor2` (in2_buffer) — borrowed-memory DFB. **clean**.
    - `update_idxs_tensor` (optional, index_buffer_for_rt) — **Case 1** (re-express). Factory: `rargs.push_back(index_buffer_for_rt)`. Kernel: `TensorAccessor(index_tensor_args, index_tensor_addr)`.
    - `page_table` (optional, page_table_buffer_for_rt) — **Case 1** (re-express, when not sharded). Factory: `rargs.push_back(page_table_buffer_for_rt)`. When sharded: borrowed-memory DFB (`CBDescriptor{ .buffer = page_table_buffer_ptr }`).

- **Custom hash:** Delete `compute_program_hash` → default (sanctioned exception) — **all three DeviceOperations**:
  - `PagedFillCacheDeviceOperation::compute_program_hash` at `device/fill_cache/paged_fill_cache_device_operation.cpp:182`
  - `PagedUpdateCacheDeviceOperation::compute_program_hash` at `device/update_cache/paged_update_cache_device_operation.cpp:313`
  - `PagedFusedUpdateCacheDeviceOperation::compute_program_hash` at `device/fused_update_cache/paged_fused_update_cache_device_operation.cpp:247`

## Heads-ups  *(mirrors the brief)*

- **Notable LANDED constructs:**
  - **Dynamic CircularBuffer (borrowed-memory DFB)** — multiple sites. These CBs use `.buffer = non-null` to place input/index/page-table tensors as borrowed-memory DFBs. Port uses `DataflowBufferSpec::borrowed_from`. Sites:
    - `paged_update_cache_program_factory.cpp:200`: `src1_cb_index` (c_1) — `CBDescriptor{ .buffer = in1_buffer }` (input sharded tensor)
    - `paged_tiled_fused_update_cache_program_factory.cpp:210`: `src1_cb_index` (c_1) — `CBDescriptor{ .buffer = in1_buffer }` (input1 sharded)
    - `paged_tiled_fused_update_cache_program_factory.cpp:218`: `src2_cb_index` (c_2) — `CBDescriptor{ .buffer = in2_buffer }` (input2 sharded)
    - `paged_tiled_fused_update_cache_program_factory.cpp:276` (conditional, index_tensor): `cb_index_id` (c_3) — `CBDescriptor{ .buffer = index_buffer_ptr }` (sharded index tensor)
    - `paged_tiled_fused_update_cache_program_factory.cpp:287` (conditional, page_table): `cb_pagetable_id` (c_4) — `CBDescriptor{ .buffer = page_table_buffer_ptr }` (sharded page table)
    - `paged_row_major_fused_update_cache_program_factory.cpp:216`: `src1_cb_index` (c_1) — `CBDescriptor{ .buffer = in1_buffer }`
    - `paged_row_major_fused_update_cache_program_factory.cpp:224`: `src2_cb_index` (c_2) — `CBDescriptor{ .buffer = in2_buffer }`
    - `paged_row_major_fused_update_cache_program_factory.cpp:265` (conditional, index): `cb_index_id` (c_3) — `CBDescriptor{ .buffer = index_buffer_ptr }`
    - `paged_row_major_fused_update_cache_program_factory.cpp:277` (conditional, page_table): `cb_pagetable_id` (c_4) — `CBDescriptor{ .buffer = page_table_buffer_ptr }`
  - **Aliased Circular Buffers** — two factories create a single `CBDescriptor` with two `CBFormatDescriptor` entries (indices `intermed0_cb_index`/`intermed1_cb_index`). Port uses `DataflowBufferSpec::advanced_options.alias_with`. Sites:
    - `paged_tiled_fused_update_cache_program_factory.cpp:226-239`: one `CBDescriptor` with `intermed0_cb_index` (c_24) and `intermed1_cb_index` (c_25)
    - `paged_row_major_fused_update_cache_program_factory.cpp:232-245`: one `CBDescriptor` with `intermed0_cb_index` (c_5) and `intermed1_cb_index` (c_6)
    - Note: `paged_update_cache_program_factory.cpp:206-218` also has this pattern (`intermed0_cb_index` c_24 + `intermed1_cb_index` c_25 in one descriptor).

- **Fake CBs (address-only):** None. All borrowed-memory CBs have genuine sharded producers/consumers (sharded input tensors write to these CBs; compute kernels consume them).

- **Cross-op / shared kernels:** Compute kernels include `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` and `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` (the official shared kernel library). Functions called use `uint32_t cb_id` template parameters — the `dfb::name` constexpr cast to `uint32_t` handles this cleanly. No file-path kernel instantiation from other op families. All kernel `.cpp` files are owned by this op.

- **RTA varargs:** None. No `get_vararg` calls found in any kernel.

- **TTNN factory analysis (porter-relevant):**
  - **Pybind `create_descriptor`:** None. `paged_cache_nanobind.cpp` uses `ttnn::bind_function<>` to bind the top-level op functions only. No `nb::class_<...ProgramFactory>` with `create_descriptor` exposed.
  - **Other risky pybind:** None. No device-operation class methods or factory internals exposed.
  - **Custom `override_runtime_arguments`:** None. All three DeviceOperations use `get_dynamic_runtime_args` (the newer patching interface), not `override_runtime_arguments`.

## Team-only

### TensorAccessor convertibility (Case-2 bindings)

No Case-2 bindings identified. All tensor bindings classified Case 1 or clean.

### Out-of-directory coupling & donor shape

**Op-level roll-up:** `✓ clean` — all out-of-directory includes are from the official shared kernel library (`ttnn/cpp/ttnn/kernel_lib/`) and call functions via `uint32_t cb_id` template parameters, which Metal 2.0 handles cleanly via the constexpr cast.

**Summary table:**

| Op kernel | Donor file | Donor class | Functions called | Shape | Status |
|---|---|---|---|---|---|
| `compute/update_cache.cpp` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` | Shared kernel lib | `compute_kernel_lib::untilize<Wt, cb_id, cb_id>(n)` | `uint32_t cb_id` NTTP | ✓ |
| `compute/update_cache.cpp` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` | Shared kernel lib | `compute_kernel_lib::tilize<Wt, cb_id, cb_id>(n)` | `uint32_t cb_id` NTTP | ✓ |
| `compute/paged_fused_update_cache.cpp` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` | Shared kernel lib | `compute_kernel_lib::untilize<Wt, cb_id, cb_id>(n)` | `uint32_t cb_id` NTTP | ✓ |
| `compute/paged_fused_update_cache.cpp` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` | Shared kernel lib | `compute_kernel_lib::tilize<Wt, cb_id, cb_id>(n)` | `uint32_t cb_id` NTTP | ✓ |
| `compute/paged_row_major_fused_update_cache.cpp` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` | Shared kernel lib | `compute_kernel_lib::untilize<Wt, cb_id, cb_id>(n)` | `uint32_t cb_id` NTTP | ✓ |
| `compute/paged_row_major_fused_update_cache.cpp` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` | Shared kernel lib | `compute_kernel_lib::tilize<Wt, cb_id, cb_id>(n)` | `uint32_t cb_id` NTTP | ✓ |

**Borrowed kernel files:** None. All kernel `.cpp` files are owned by this op directory. No file-path kernel instantiation from shared pools or other op families.

### Relaxation candidates (mined from custom hashes before deletion)

**FALLIBLE — candidates to verify, default remains strict.**

- `PagedFillCacheDeviceOperation::compute_program_hash` (`paged_fill_cache_device_operation.cpp:182-191`) hashes: `mesh_coords`, `block_size_override`, `cache_position_modulo`, `tensor_args`, `program_factory.index()`. Excludes: `batch_idx_fallback`, `noop`. Observation: `tensor_args` is hashed in full (all four tensors). No relaxation on tensor shape is encoded. No candidates apparent.
- `PagedUpdateCacheDeviceOperation::compute_program_hash` (`paged_update_cache_device_operation.cpp:313-337`) excludes `update_idxs` (runtime-only), `batch_offset` (always 0). Hashes `compute_kernel_config`, `share_cache`, `mesh_coords`, `block_size_override`, `num_kv_heads_override`, `cache_position_modulo`, `tensor_args`. The exclusion of `update_idxs` suggests the op tolerates per-dispatch position changes (already handled by `get_dynamic_runtime_args`). No shape relaxation encoded. No candidates apparent.
- `PagedFusedUpdateCacheDeviceOperation::compute_program_hash` (`paged_fused_update_cache_device_operation.cpp:247-265`) excludes `update_idxs`, `batch_offset`. Similar to above. No candidates apparent.

### TTNN factory analysis

**Q1 — Op-owned tensors?** No. All three DeviceOperations are in-place: `create_output_tensors` returns `tensor_args.cache_tensor` (or cache_tensor1/cache_tensor2) directly without allocating new device tensors.

**Q2 — MeshWorkload concept needed?** No. The `*MeshWorkloadFactory` variants exist for the `mesh_coords` filtering feature: when `operation_attributes.mesh_coords.has_value()`, the factory dispatches a noop program (or empty descriptor) to coordinates outside the set. This is a plumbing artifact of the coordinate-filtering design — not genuine cross-program or cross-device coordination. There is no `create_mesh_workload` / `create_workload_descriptor` / `cached_mesh_workload_t` in any DeviceOperation. The `program_factory_t` variant that holds a `*MeshWorkloadFactory` type is the TTNN-level dispatch selector, not a Metal-level MeshWorkload need. → Not needed.

**Q3 — Pybind `create_descriptor`?** No. `paged_cache_nanobind.cpp` uses `ttnn::bind_function<"paged_update_cache", ...>`, `ttnn::bind_function<"paged_fused_update_cache", ...>`, and `ttnn::bind_function<"paged_fill_cache", ...>` — all top-level user-facing functions. No `nb::class_<...ProgramFactory>` with `def_static("create_descriptor")` found.

**Q4 — Other migration-risky pybind?** No. The nanobind file exposes only the three user-facing functions with their documented parameters (tensors, indices, config). No device-op methods, no factory parameter structs, no introspection entry points.

**Q5 — Custom hash?** Yes — all three DeviceOperations. See Custom program hash subject above.

**Q6 — Custom override-runtime-args?** No `override_runtime_arguments` defined in any factory. The DeviceOperations use `get_dynamic_runtime_args` for the per-dispatch patching of position-derived values (`cache_start_id`, `tile_update_offset_B`, `batch_idx_fallback`). This is a different interface from `override_runtime_arguments` and does not require a port-time deletion.

## Misc anomalies  *(team-only, non-gating)*

- `paged_tiled_fused_update_cache_program_factory.cpp:198`: `PagedUpdateCacheProgramFactory` also has the aliased-CB pattern on `intermed0_cb_index`/`intermed1_cb_index` (two-element `format_descriptors`). This is correctly listed under Heads-ups above. (Reminder to the porter: three factories have this pattern, not two.)

- `reader_update_cache_interleaved_start_id.cpp:29`: `const uint32_t log_base_2_of_page_size = get_compile_time_arg_val(6);` — declared non-`constexpr` despite being a compile-time arg. Minor style inconsistency; not a functional issue.

- `reader_paged_fused_update_cache_interleaved_start_id.cpp:43`: Similarly `const uint32_t log_base_2_of_page_size = get_compile_time_arg_val(8)` is non-`constexpr`.

## Questions for the user  *(omit if none)*

None. All findings are unambiguous.

## Recipe notes  *(omit if none)*

- The recipe's TensorAccessor handling section covers the `Buffer*`-binding form (framework auto-registers and patches) as "still enumerate it — Case 1." This op uses that pattern pervasively (`emplace_runtime_args` with `Buffer*` objects plus `TensorAccessorArgs().append_to(CTA)` on the same buffer). The combination of CTA-baked TensorAccessorArgs + `Buffer*` RTA is the standard modern ProgramDescriptor pattern and translates cleanly to `TensorParameter` + `TensorBinding`. The audit recipe could benefit from a note that this combined pattern (CTA accessor args + Buffer* RTA = framework-managed base) is the expected form for ProgramDescriptor-API ops, so auditors don't hesitate classifying it as Case 1.

- The "MeshWorkload false-positive trap" section of the recipe focuses on op-owned tensors forcing ops onto the MeshWorkload path. This op's `*MeshWorkloadFactory` variants arise from a different cause: a `mesh_coords` filtering feature (dispatch noop programs to non-target coordinates). This isn't exactly the op-owned-tensor artifact the recipe describes, but the conclusion is the same (→ No genuine MeshWorkload need). The recipe could add a brief note that the mesh-coord-filtering pattern is another common false positive.
