# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/reduction/topk`

Single `DeviceOperation` in this directory with two factories:

- **`TopKDeviceOperation`** (`topk_device_operation.cpp`, `topk_device_operation.hpp`)
  - `TopKSingleCoreProgramFactory` (`topk_single_core_program_factory.cpp`)
  - `TopKMultiCoreProgramFactory` (`topk_multi_core_program_factory.cpp`)

Kernels owned by this op (all in `device/kernels/`):

- Dataflow: `reader_create_index_tensor.cpp`, `writer_binary_interleaved.cpp` (single-core)
- Dataflow: `reader_create_index_local_topk.cpp`, `reader_final_topk.cpp`, `writer_local_topk.cpp`, `writer_final_topk.cpp` (multi-core)
- Compute: `topk.cpp` (single-core), `topk_local.cpp`, `topk_final.cpp` (multi-core)
- Shared in-op headers: `topk_dataflow_common.hpp`, `topk_common_funcs.hpp`

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/reduction/topk` |
| **Overall** | GREEN |
| **DOps / Factories** | `TopKDeviceOperation` → `TopKSingleCoreProgramFactory`, `TopKMultiCoreProgramFactory` |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | Yes |
| *Prereqs* — Cross-op escapes | Ok |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — MeshWorkload needed | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | None |

## Result

**GREEN → brief issued.** All gates cleared. Port can proceed once user gives explicit go-ahead.

## Gate detail

- **ProgramDescriptor:** GREEN. Both factories populate a `tt::tt_metal::ProgramDescriptor` and use `KernelDescriptor`, `CBDescriptor`, `SemaphoreDescriptor`. No imperative `host_api.hpp` builder calls (`CreateProgram`, `CreateKernel`, `SetRuntimeArgs`, etc.).

- **Device 2.0 (every kernel used):** GREEN. All dataflow kernels use Device 2.0 wrappers throughout: `Noc`, `CircularBuffer`, `Semaphore<>`, `TensorAccessor`. No legacy idioms (`InterleavedAddrGen`, `ShardedAddrGen`, raw `noc_async_read`, raw semaphore-address integers, etc.). All compute kernels use Device 2.0 compute API (`api/compute/...`) and `CircularBuffer` wrappers. No CB-index-keyed free-function holdovers found.

- **Feature compatibility:** See table below. No UNSUPPORTED features fired. All Appendix A entries either N/A or clean GREEN.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` reference anywhere in the op |
  | Dynamic CircularBuffer (borrowed memory) | N/A | No `CBDescriptor::buffer` set to non-null anywhere |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `address_offset` field used |
  | Aliased Circular Buffers | N/A | All `format_descriptors` initializers are single-element |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` reference anywhere |
  | Non-zero semaphore initial value | N/A | Both `SemaphoreDescriptor::initial_value = INVALID` = 0 (`hostdevcommon/common_values.hpp:13`) |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | No `ArgConfig::Runtime` token anywhere |
  | `UpdateCircularBuffer*` | N/A | No `UpdateCircularBuffer*` call anywhere |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` has fixed-count named members; no CTA loop |

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per factory):

  | Binding | Factory | Shape | Case |
  |---|---|---|---|
  | `input` | SingleCore | `MeshTensor&` in `emplace_runtime_args` (buffer-binding form) | Case 1 — re-express |
  | `value` (output) | SingleCore | `MeshTensor&` in `emplace_runtime_args` (buffer-binding form) | Case 1 — re-express |
  | `index` (output) | SingleCore | `MeshTensor&` in `emplace_runtime_args` (buffer-binding form) | Case 1 — re-express |
  | `indices` (optional input) | SingleCore | raw `.address()` RTA: `tensor_args.indices->mesh_tensor().address()` cast to `uint32_t` — **correctness hazard** | Case 1 — re-express |
  | `input` | MultiCore | `MeshTensor&` in `emplace_runtime_args` (buffer-binding form) | Case 1 — re-express |
  | `value` (output) | MultiCore | `MeshTensor&` in `emplace_runtime_args` (buffer-binding form) | Case 1 — re-express |
  | `index` (output) | MultiCore | `MeshTensor&` in `emplace_runtime_args` (buffer-binding form) | Case 1 — re-express |
  | `indices` (optional input) | MultiCore | raw `.address()` RTA: `input_indices_tensor->address()` cast to `uint32_t` — **correctness hazard** | Case 1 — re-express |

  **Notes on the `indices` binding:** In both factories the optional `indices` input tensor's address is extracted via `.address()` and cast to `uint32_t` before injection into `emplace_runtime_args`. This is the **silent-wrong hazard** pattern: on a fast-path cache hit the framework patches the typed-binding channel but leaves this RTA at its original value, so the kernel reads a stale buffer address on cache hits with different tensor storage. The kernel consumes it as a `TensorAccessor` base (`src_indices_addr → TensorAccessor(indices_args, src_indices_addr)` in both `reader_create_index_tensor.cpp:33` and `reader_create_index_local_topk.cpp:44`). Re-expressing via `TensorParameter` / `TensorBinding` removes both the hazard and the explicit `.address()` extraction.

  Specific host-side sites:
  - `topk_single_core_program_factory.cpp:266` — `static_cast<uint32_t>(tensor_args.indices->mesh_tensor().address())`
  - `topk_multi_core_program_factory.cpp:505` — `static_cast<uint32_t>(input_indices_tensor->address())`

- **Custom hash:** None — no `compute_program_hash` override.

## Heads-ups  *(mirrors the brief)*

- **Notable LANDED constructs:** None that require porter heads-ups. No aliased CBs, no borrowed-memory DFBs, no dynamic TensorAccessors, no non-zero semaphore init values.
- **Fake CBs (address-only):** None found. All CBs have real producer/consumer pairs.
- **Cross-op / shared kernels:** None. All kernel files are owned by this op. All kernel `#include`s resolve to `tt_metal/*` LLK/HAL or in-op headers.
- **RTA varargs:** None found in either factory or any kernel.
- **TTNN factory analysis (porter-relevant):** No pybind `create_descriptor`, no other migration-risky pybind, no custom `override_runtime_arguments`.

## Team-only

### TensorAccessor convertibility

Both `indices` bindings are Case 1 (not Case 2). The access pattern is page-by-page (`page_id = row * Wt + w` or `page_id = i * Wt + j`) — standard interleaved iteration well within `TensorAccessor` support. No exotic addressing. Marked convertible / non-exotic; no Case 2 flag needed.

### Out-of-directory coupling & donor shape

**Op-level roll-up:** ✓ clean. No cross-family or in-family escapes. Zero out-of-op kernel function calls.

**Summary table:**

| Op kernel | Donor file | Class | Status |
|---|---|---|---|
| All kernel `.cpp` files | `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, etc. | `tt_metal/*` LLK/HAL | ✓ No concern |
| All kernel `.cpp` files | `api/compute/compute_kernel_api.h`, etc. | `tt_metal/*` LLK/HAL | ✓ No concern |
| `reader_create_index_tensor.cpp`, `reader_create_index_local_topk.cpp` | `topk_dataflow_common.hpp` | In-op | ✓ No escape |
| `topk_local.cpp`, `topk_final.cpp` | `topk_common_funcs.hpp` | In-op | ✓ No escape |

**Borrowed kernel files (file-path kernel instantiation):** None. Both factories instantiate only kernels from their own `device/kernels/` subdirectory. No shared-pool or cross-family kernel files.

### Relaxation candidates

No custom `compute_program_hash` was present, so no hash to mine. Default strict hash applies throughout.

### TTNN factory analysis (six questions)

1. **Op-owned tensors?** No. The factories receive `tensor_return_value` (a `std::tuple<Tensor, Tensor>`) from the device-op framework and write to those tensors. No `create_device_tensor` or `allocate_tensor_on_device` calls inside either factory. `create_output_tensors` in `topk_device_operation.cpp:281` does call `create_device_tensor`, but that is the device-op's standard output-allocation hook, not an intermediate allocation inside a factory.

2. **MeshWorkload concept needed?** No. Neither factory provides `create_mesh_workload` / `create_workload_descriptor`. No `cached_mesh_workload_t` in the device-op. Single-program path.

3. **Pybind `create_descriptor`?** No. `topk_nanobind.cpp` binds only `ttnn::topk` via `ttnn::bind_function<"topk">` — the normal op-function binding. No `nb::class_<…ProgramFactory>` or `.def_static("create_descriptor", …)` present.

4. **Other migration-risky pybind?** None. The nanobind file exposes only the user-facing `topk` function with standard argument types. No device-op methods, no factory or param classes, no `compute_program_hash` exposure.

5. **Custom hash?** No. `TopKDeviceOperation` has no `compute_program_hash` member or override. `ProgramDescriptor::custom_program_hash` is not set in either factory. The `ProgramDescriptor` struct has a `std::optional<uint64_t> custom_program_hash` field (per `program_descriptors.hpp:220`) but it is left unset here.

6. **Custom override-runtime-args?** No. Neither factory defines `override_runtime_arguments`. Confirmed by full search across the op directory.

## Misc anomalies

- `topk_single_core_program_factory.cpp:199`: The `reader_defines_map` always sets `GENERATE_INDICES = "1"` with the comment `"tensor_args.indices.has_value() ? "0" : "1" - GH issue: #36329"`. The optional-indices read path (the `#if not GENERATE_INDICES` branch in the reader kernel) is therefore currently dead — the kernel always generates indices rather than reading a pre-supplied index tensor, even when `tensor_args.indices` is provided. The indices tensor address is still injected as RTA arg 3, but the kernel never reads it on this factory path. This is the single-core factory only; the multi-core factory correctly conditions the define on `tensor_args.indices.has_value()`. The anomaly is the mismatch, not the individual behaviors.

## Recipe notes

None. The audit recipe was followed without friction or ambiguity.
