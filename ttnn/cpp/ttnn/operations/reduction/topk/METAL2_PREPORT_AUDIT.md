# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/reduction/topk`

Single device operation, two program factories:

- **`ttnn::prim::TopKDeviceOperation`**
  - `TopKSingleCoreProgramFactory` (`device/topk_single_core_program_factory.cpp`)
  - `TopKMultiCoreProgramFactory` (`device/topk_multi_core_program_factory.cpp`)

**Kernels referenced (all in-directory, all Device-2.0):**
- Single-core: `device/kernels/dataflow/reader_create_index_tensor.cpp`, `device/kernels/dataflow/writer_binary_interleaved.cpp`, `device/kernels/compute/topk.cpp`
- Multi-core: `device/kernels/dataflow/reader_create_index_local_topk.cpp`, `device/kernels/dataflow/reader_final_topk.cpp`, `device/kernels/dataflow/writer_local_topk.cpp`, `device/kernels/dataflow/writer_final_topk.cpp`, `device/kernels/compute/topk_local.cpp`, `device/kernels/compute/topk_final.cpp`
- Shared in-directory headers: `device/kernels/dataflow/topk_dataflow_common.hpp` (index-tile generation), `device/kernels/compute/topk_common_funcs.hpp` (bitonic merge helpers)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/reduction/topk` |
| **Overall** | GREEN |
| **DOps / Factories** | `TopKDeviceOperation` → `TopKSingleCoreProgramFactory`, `TopKMultiCoreProgramFactory` |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | Yes |
| *Prereqs* — Cross-op escapes | Ok (no out-of-directory kernel includes) |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — MeshWorkload needed | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | None |

**Fake CBs** = CBs used purely as an address source. The multi-core `gathered_values_cb` (c_4) / `gathered_indices_cb` (c_5) on the final core *look* address-only (remote cores write into them by raw L1 pointer in `writer_local_topk.cpp`), but each has a real producer–consumer FIFO: `reader_final_topk.cpp` reserves/pushes (producer side) and `topk_final.cpp` `wait_front`/`pop_front`s (consumer). Producer + consumer present → genuine DFB, **not** a fake CB. No fake CBs anywhere in the op.

## Result

**GREEN → brief issued.** Both program factories are on the `ProgramDescriptor` API and every kernel they exercise is Device 2.0 compliant. No UNSUPPORTED Appendix A feature fires. All tensor bindings are Case 1 (read/written exclusively through `TensorAccessor`); no Case 2 raw-pointer bindings and no compute-kernel tensor binding. No custom program hash, no op-owned tensors, no MeshWorkload need, no risky pybind. The port is straightforward, mechanical work: convert `TensorAccessorArgs` CTA plumbing + the buffer-address RTAs into typed `TensorParameter` / `TensorBinding`s.

## Gate detail

- **ProgramDescriptor:** GREEN. Both factories return `tt::tt_metal::ProgramDescriptor` via `create_descriptor(...)` and populate `desc.cbs` (`CBDescriptor` / `CBFormatDescriptor`), `desc.kernels` (`KernelDescriptor` with `SourceType::FILE_PATH`, `ReaderConfigDescriptor`/`WriterConfigDescriptor`/`ComputeConfigDescriptor`), and `desc.semaphores` (`SemaphoreDescriptor`). No `CreateProgram` / `CreateKernel` / `CreateCircularBuffer` / `SetRuntimeArgs` imperative calls. (`device/topk_device_operation.hpp:24-38`, `device/topk_single_core_program_factory.cpp:18,33,95,203,287-289`, `device/topk_multi_core_program_factory.cpp:66,78,171,322,535-540`.)

- **Device 2.0 (every kernel used):** GREEN. Every kernel uses the Device-2.0 wrapper surface throughout and contains no Device-1.0 idioms:
  - Data movement: `Noc noc; noc.async_read(...)` / `noc.async_write(...)` / `noc.async_read_barrier()` (e.g. `reader_create_index_tensor.cpp:42,55-56`, `writer_binary_interleaved.cpp:33,46`, `reader_create_index_local_topk.cpp:35,52`, `writer_final_topk.cpp:33,44`).
  - CBs: `CircularBuffer` wrapper objects with member-form methods — `cb.reserve_back/push_back/wait_front/pop_front`, `cb.get_tile_size()`, `cb.get_write_ptr()`, `cb.get_read_ptr()` (e.g. `reader_create_index_tensor.cpp:43-47`, `writer_local_topk.cpp:35-46`, `topk.cpp:141-146`, `topk_common_funcs.hpp:18-21`, `topk_dataflow_common.hpp:32-36`). These are the sanctioned **wrapper-method** forms, not the CB-index free-function holdovers.
  - Tensors: `TensorAccessorArgs<N>()` + `TensorAccessor(args, addr)` (e.g. `reader_create_index_tensor.cpp:27,40`, `writer_binary_interleaved.cpp:23-31`, `reader_create_index_local_topk.cpp:32-33`, `writer_final_topk.cpp:22-31`).
  - Semaphores: `Semaphore<>` wrapper with `.set/.wait/.up/.set_multicast` (e.g. `reader_final_topk.cpp:26-52`, `writer_local_topk.cpp:32-96`); cross-core via `UnicastEndpoint remote` (`writer_local_topk.cpp:34,64-69`) and `CoreLocalMem<volatile T>` (`topk_dataflow_common.hpp:36`).
  - No `noc_async_read`/`noc_async_write` raw forms, no `InterleavedAddrGen`/`ShardedAddrGen`/`InterleavedAddrGenFast`, no bare CB-index `get_read_ptr(cb_id)`/`get_write_ptr(cb_id)` free functions, no raw semaphore addresses. Confirmed by directory-wide grep.
  - No out-of-directory donor kernels are instantiated or `#include`d (all kernel includes are `api/*` LLK/HAL + the two in-directory shared headers), so there is no donor-side Device 2.0 dependency to gate on.

- **Feature compatibility:** every Appendix A entry, in order.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CBDescriptor::global_circular_buffer` field set, no `remote_index`/`remote_cb_*` idiom. |
  | Dynamic CircularBuffer (borrowed memory) | N/A | No `CBDescriptor::.buffer` set; no `set_globally_allocated_address`. All CBs are static (`.total_size` + `format_descriptors` only). |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` set on any `CBDescriptor`; no `set_address_offset`. |
  | Aliased Circular Buffers | N/A | Every `CBDescriptor::format_descriptors` is single-element (`{{CBFormatDescriptor{...}}}`). |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` type, no `CreateGlobalSemaphore`. Only plain `SemaphoreDescriptor` (multi-core factory lines 322-333). |
  | Non-zero semaphore initial value | N/A | `SemaphoreDescriptor::initial_value = INVALID` (`topk_multi_core_program_factory.cpp:326,332`); `INVALID` resolves to `0` (`tt_metal/hostdevcommon/api/hostdevcommon/common_values.hpp:13`), i.e. explicit zero → false-positive guard applies, no heads-up. |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | All `TensorAccessorArgs(tensor)` are the single-argument (static) form; no `ArgConfig::Runtime*` token anywhere. |
  | `UpdateCircularBuffer*` | N/A | No `UpdateCircularBuffer*` / `UpdateDynamicCircularBuffer*` calls; no `override_runtime_arguments` hook exists. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` is a fixed-shape `TopkInputs` (one input + optional indices + optional preallocated outputs); no `std::vector<Tensor>`. Kernels read CTAs at fixed positions (`get_compile_time_arg_val(0..N)`) and `TensorAccessorArgs<N>()` at a fixed offset — no runtime-varying CTA index. |

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, all **Case 1** — via `TensorAccessor`):
  - **Single-core** (`TopKSingleCoreProgramFactory`):
    - `input` — RTA arg 0 is the `input_tensor` object (`topk_single_core_program_factory.cpp:262`; auto-registered `BufferBinding`, framework-patched today); consumed as `TensorAccessor(inout_tensor_args, src_addr)` (`reader_create_index_tensor.cpp:40`). Case 1.
    - `indices` (optional input) — RTA arg 3 is `tensor_args.indices->mesh_tensor().address()` (`topk_single_core_program_factory.cpp:266`); consumed as `TensorAccessor(indices_args, src_indices_addr)` (`reader_create_index_tensor.cpp:33`). This is a raw `.address()`-via-RTA smuggle feeding a `TensorAccessor` → Case 1; the port replaces it with a typed `TensorParameter` (eliminates the fast-path-cache stale-address hazard). *(Note: only the `GENERATE_INDICES=1` path is compiled today — see Misc anomalies — but the binding is enumerated regardless.)*
    - `values` output — RTA arg 0 is the `value_tensor` object (`:272`); consumed as `TensorAccessor(values_tensor_args, dst_addr0)` (`writer_binary_interleaved.cpp:30`). Case 1.
    - `indices` output — RTA arg 1 is the `index_tensor` object (`:273`); consumed as `TensorAccessor(indices_tensor_args, dst_addr1)` (`writer_binary_interleaved.cpp:31`). Case 1.
  - **Multi-core** (`TopKMultiCoreProgramFactory`):
    - `input` — RTA arg 0 is the `input_tensor` object (`topk_multi_core_program_factory.cpp:501`); consumed as `TensorAccessor(s_args, src_addr)` (`reader_create_index_local_topk.cpp:33`). Case 1.
    - `indices` (optional input) — RTA arg 4 is `input_indices_tensor->address()` (`:505`); consumed as `TensorAccessor(indices_args, src_indices_addr)` (`reader_create_index_local_topk.cpp:44`). Raw `.address()`-via-RTA feeding a `TensorAccessor` → Case 1.
    - `values` output — RTA arg 0 is the `value_tensor` object (`:531`); consumed as `TensorAccessor(interleaved_accessor0_args, dst_addr0)` (`writer_final_topk.cpp:30`). Case 1.
    - `indices` output — RTA arg 1 is the `index_tensor` object (`:532`); consumed as `TensorAccessor(interleaved_accessor1_args, dst_addr1)` (`writer_final_topk.cpp:31`). Case 1.
  - **No Case 2 (raw-pointer) bindings.** Every tensor read/write goes through a `TensorAccessor`. The compute kernels (`topk.cpp`, `topk_local.cpp`, `topk_final.cpp`) and the inter-core transfer kernels (`writer_local_topk.cpp`, `reader_final_topk.cpp`) touch only CB L1 memory, never tensor memory — out of scope for this subject (no compute-kernel `TensorBinding` blocker).
- **Custom hash:** none — the op uses the default reflection-based hash.

## Heads-ups  *(mirrors the brief)*

- **Notable LANDED constructs:** none. No aliased CB, no borrowed-memory DFB, no dynamic `TensorAccessor`, no non-zero semaphore init (the `INVALID` sentinel is `0`).
- **Fake CBs (address-only):** none (see Status-summary note — the gathered CBs are genuine producer/consumer DFBs).
- **Cross-op / shared kernels:** none. The op owns all of its kernels; no kernel `.cpp` is instantiated from a shared pool or another family, and no kernel `#include`s anything outside the op directory except `api/*` (LLK/HAL). No port-together coupling.
- **RTA varargs:** none. No kernel reads `num_runtime_varargs` or loops over `get_arg_val` with a runtime-varying index.
- **TTNN factory analysis (porter-relevant):** none — no pybind `create_descriptor`, no other migration-risky pybind, no custom `override_runtime_arguments`.

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ✓ clean.** No function-call escapes and no file-path kernel borrows.

- **Function-call escape inventory:** every kernel `#include` resolves to `api/*` (LLK/HAL/firmware — donor class 1, no concern) or to the two in-directory shared headers (`topk_dataflow_common.hpp`, `topk_common_funcs.hpp`, both in-family / same op). No `ttnn/cpp/ttnn/kernel_lib/`, `ttnn/cpp/ttnn/kernel/`, `kernel_helper_functions/`, or cross-family includes. Summary table and per-call detail omitted (all rolls ✓).
- **Borrowed kernel files (file-path instantiation):** none. All nine kernel `.cpp` files instantiated by the two factories live in `device/kernels/` of this op directory. No shared-pool or cross-family borrow → no port-together set.

### Relaxation candidates (mined from custom hash before deletion)

N/A — no custom `compute_program_hash`.

### TTNN factory analysis (six questions)

1. **Op-owned tensors? No.** The only device-tensor creation is `create_device_tensor(...)` in `TopKDeviceOperation::create_output_tensors` (`device/topk_device_operation.cpp:292-293`) for the op's two *declared output* tensors (`tensor_return_value`). No intermediate/scratch tensors beyond inputs and outputs are allocated in either factory.
2. **MeshWorkload concept needed? No.** Neither factory provides `create_mesh_workload` / `create_workload_descriptor`; the device-operation carries no `cached_mesh_workload_t`. Both factories return a single `ProgramDescriptor`. The multi-core factory coordinates many cores within one program via semaphores (`SemaphoreDescriptor` + `Semaphore<>` kernels), which is single-program multi-core, not multi-program/cross-device.
3. **Pybind `create_descriptor`? No.** `topk_nanobind.cpp` binds only the op function (`ttnn::bind_function<"topk">`, line 90). No `nb::class_<...ProgramFactory>` / `def_static("create_descriptor", ...)`.
4. **Other migration-risky pybind? None.** No `DeviceOperation`/factory/param class is exposed to Python; no `compute_program_hash` / `create_output_tensors` / `compute_output_specs` / `select_program_factory` binding.
5. **Custom hash? No.** `TopKDeviceOperation` declares no `compute_program_hash` (uses the default reflection hash). See Custom program hash subject — nothing to delete.
6. **Custom override-runtime-args? No.** Neither factory defines `override_runtime_arguments`; runtime args are emitted directly in `create_descriptor` (`emplace_runtime_args` / `runtime_args.emplace_back`).

## Misc anomalies  *(team-only, non-gating)*

- **Dead `GENERATE_INDICES` branch in single-core (`reader_create_index_tensor.cpp:29-34,66-73`).** The single-core factory hardcodes `{"GENERATE_INDICES", "1"}` unconditionally (`topk_single_core_program_factory.cpp:198-200`) with the comment `// tensor_args.indices.has_value() ? "0" : "1" - GH issue: #36329`. So the `#if not GENERATE_INDICES` precomputed-indices read path is never compiled in the single-core kernel, even though the factory still pushes the optional indices `TensorAccessorArgs` (`:195-197`) and the `indices->address()` RTA (`:266`). The multi-core factory, by contrast, sets the define from `tensor_args.indices.has_value()` (`topk_multi_core_program_factory.cpp:347-349`) and does compile the read path. Net: in the single-core path the optional-indices CTA + RTA are effectively dead. Routes to the op owner (GH #36329); the port enumerates the indices binding either way.

## Recipe notes  *(none)*
