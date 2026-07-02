# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama`

Single `DeviceOperation` in the directory:

- **`RotaryEmbeddingLlamaDeviceOperation`** (`device/rotary_embedding_llama_device_operation.cpp` / `.hpp`)
  - `RotaryEmbeddingLlamaMultiCore` (`device/rotary_embedding_llama_multi_core_program_factory.cpp`)
    — interleaved-input prefill path; own reader/writer/compute kernels
  - `RotaryEmbeddingLlamaMultiCoreSharded` (`device/rotary_embedding_llama_sharded_program_factory.cpp`)
    — fully-sharded decode path; compute-only kernel (no reader/writer)
  - `RotaryEmbeddingLlamaMultiCorePrefillSharded` (`device/rotary_embedding_llama_multi_core_prefill_sharded_program_factory.cpp`)
    — prefill path with optional HEIGHT_SHARDED cos/sin/trans_mat; own reader/writer/compute kernels

**Kernels in scope (all op-owned; no borrowed file-path kernels):**

| Kernel file | Used by factory |
|---|---|
| `device/kernels/dataflow/reader_rotary_embedding_llama_interleaved_start_id.cpp` | `RotaryEmbeddingLlamaMultiCore`, `RotaryEmbeddingLlamaMultiCorePrefillSharded` (writer re-used) |
| `device/kernels/dataflow/writer_rotary_embedding_llama_interleaved_start_id.cpp` | `RotaryEmbeddingLlamaMultiCore`, `RotaryEmbeddingLlamaMultiCorePrefillSharded` |
| `device/kernels/dataflow/reader_rotary_embedding_llama_prefill_sharded.cpp` | `RotaryEmbeddingLlamaMultiCorePrefillSharded` |
| `device/kernels/compute/rotary_embedding_llama.cpp` | `RotaryEmbeddingLlamaMultiCore`, `RotaryEmbeddingLlamaMultiCorePrefillSharded` |
| `device/kernels/compute/rotary_embedding_llama_sharded.cpp` | `RotaryEmbeddingLlamaMultiCoreSharded` |

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama` |
| **Overall** | GREEN |
| **DOps / Factories** | `RotaryEmbeddingLlamaDeviceOperation` → `RotaryEmbeddingLlamaMultiCore`, `RotaryEmbeddingLlamaMultiCoreSharded`, `RotaryEmbeddingLlamaMultiCorePrefillSharded` |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | Yes |
| *Prereqs* — Cross-op escapes | Ok |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — MeshWorkload needed | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | Yes → delete (see Custom program hash) |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | None |

---

## Result

**GREEN → brief issued.** All gates cleared. Every prerequisite is met, no UNSUPPORTED feature fires, Device 2.0 compliance is clean across all five kernel files, and the nanobind surface is minimal and normal. The port can proceed with user go-ahead; the `METAL2_PORT_BRIEF.md` has been written alongside this file.

---

## Gate detail

### ProgramDescriptor

**GREEN.** All three factories (`RotaryEmbeddingLlamaMultiCore`, `RotaryEmbeddingLlamaMultiCoreSharded`, `RotaryEmbeddingLlamaMultiCorePrefillSharded`) already return a `tt::tt_metal::ProgramDescriptor`. Their headers declare `create_descriptor` returning `ProgramDescriptor` and `#include <tt-metalium/program_descriptors.hpp>`. No legacy `host_api.hpp` builder calls (`CreateProgram`, `CreateKernel`, `CreateCircularBuffer`, `SetRuntimeArgs`) are present in the factory `.cpp` files.

### Device 2.0 (every kernel used)

**GREEN.** All five kernel files are Device 2.0 compliant:

- All dataflow kernels use `Noc noc;` + `noc.async_read(...)` / `noc.async_write(...)` + `noc.async_read_barrier()` / `noc.async_write_barrier()` — `experimental::Noc` wrapper throughout.
- All kernels use `CircularBuffer cb_foo(cb_id);` wrapper objects for every CB interaction (`reserve_back`, `push_back`, `wait_front`, `pop_front`, `get_write_ptr`, `get_read_ptr`).
- All kernels use `TensorAccessor` / `TensorAccessorArgs` for tensor page access.
- `get_tile_size(cb_id)` appears in all three dataflow kernels (reader interleaved: lines 46, 49, 52, 70; reader prefill sharded: lines 48, 74, 85, 86, 142, 145; writer: lines 36, 37) — this is a **sanctioned** Device 2.0 free function per the audit recipe, not a holdover.
- No `InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedAddrGenFast`, `InterleavedPow2AddrGen*`, raw `noc_async_read` / `noc_async_write` free functions, or raw semaphore-address manipulation anywhere.

### Feature compatibility

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | No `experimental::GlobalCircularBuffer`, no `global_circular_buffer` field set on any `CBDescriptor`. |
| Dynamic CircularBuffer (borrowed memory) | GREEN | Used in `RotaryEmbeddingLlamaMultiCoreSharded` and conditionally in `RotaryEmbeddingLlamaMultiCorePrefillSharded`. `CBDescriptor::buffer` set for input/cos/sin/trans_mat/output CBs. Port uses `DataflowBufferSpec::borrowed_from`. |
| CBDescriptor `address_offset` (non-zero) | N/A | No `address_offset` field set on any `CBDescriptor`; no `UpdateDynamicCircularBufferAddress` 4-arg overload. |
| Aliased Circular Buffers | N/A | Every `CBDescriptor::format_descriptors` initializer is single-element. |
| GlobalSemaphore | N/A | No semaphores declared at all; no `GlobalSemaphore` references. |
| Non-zero semaphore initial value | N/A | No semaphores used anywhere in the op. |
| Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | No `ArgConfig::Runtime` tokens in any factory. All `TensorAccessorArgs` use the single-argument form `TensorAccessorArgs(*buffer)`. |
| `UpdateCircularBuffer*` | N/A | No `UpdateCircularBufferTotalSize`, `UpdateCircularBufferPageSize`, or `UpdateDynamicCircularBufferAddressAndTotalSize` calls anywhere. |
| Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` is a struct with four fixed-name `Tensor` fields. No variable-count container; no `get_compile_time_arg_val(i)` loop with a runtime-varying index. |

**Feature compatibility overall: GREEN — no gate fired.**

Dynamic CircularBuffer is in use (LANDED); see Heads-ups for the borrowed-memory DFB details.

---

## Port-work summary *(mirrors the brief)*

### Tensor bindings (per binding)

The host-side Buffer* objects are passed via `emplace_runtime_args` into `KernelDescriptor::RTArgList` — the "Buffer*-binding form" where the framework auto-registers `BufferBinding`s. This is not the silent-wrong hazard (the framework patches them on cache hits); it is still **Case 1 (re-express)** port work.

| Binding | Factory | Case | Notes |
|---|---|---|---|
| `input_tensor` (`src_buffer`) | `RotaryEmbeddingLlamaMultiCore` | **Case 1** | `Buffer*` in reader RTA; kernel reads `src_addr` as `uint32_t` → `TensorAccessor`. |
| `cos_cache` (`cos_buffer`) | `RotaryEmbeddingLlamaMultiCore` | **Case 1** | `Buffer*` in reader RTA; kernel reads `cos_addr` → `TensorAccessor`. |
| `sin_cache` (`sin_buffer`) | `RotaryEmbeddingLlamaMultiCore` | **Case 1** | `Buffer*` in reader RTA; kernel reads `sin_addr` → `TensorAccessor`. |
| `trans_mat` (`trans_mat_buffer`) | `RotaryEmbeddingLlamaMultiCore` | **Case 1** | `Buffer*` in reader RTA; kernel reads `trans_mat_addr` → `TensorAccessor`. |
| `output` (`dst_buffer`) | `RotaryEmbeddingLlamaMultiCore` | **Case 1** | `Buffer*` in writer RTA; kernel reads `dst_addr` → `TensorAccessor`. |
| `input_tensor` (`src_buffer`) | `RotaryEmbeddingLlamaMultiCoreSharded` | clean | `CBDescriptor::buffer = src_buffer` → borrowed-memory DFB; compute kernel accesses via CB pointer only. |
| `cos_cache` (`cos_buffer`) | `RotaryEmbeddingLlamaMultiCoreSharded` | clean | `CBDescriptor::buffer = cos_buffer` → borrowed-memory DFB. |
| `sin_cache` (`sin_buffer`) | `RotaryEmbeddingLlamaMultiCoreSharded` | clean | `CBDescriptor::buffer = sin_buffer` → borrowed-memory DFB. |
| `trans_mat` (`trans_mat_buffer`) | `RotaryEmbeddingLlamaMultiCoreSharded` | clean | `CBDescriptor::buffer = trans_mat_buffer` → borrowed-memory DFB. |
| `output` (`dst_buffer`) | `RotaryEmbeddingLlamaMultiCoreSharded` | clean | `CBDescriptor::buffer = dst_buffer` → borrowed-memory DFB. |
| `input_tensor` (`src_buffer`) | `RotaryEmbeddingLlamaMultiCorePrefillSharded` | **Case 1** | `Buffer*` in reader RTA; kernel reads `src_addr` → `TensorAccessor`. (Input itself is always interleaved in prefill.) |
| `cos_cache` (`cos_buffer`) | `RotaryEmbeddingLlamaMultiCorePrefillSharded` | **Case 1** (interleaved path) / clean (sharded fast-path) | When `cos_sin_sharded && !cos_sin_sharded_reload`: `CBDescriptor::buffer = cos_buffer` → borrowed-memory DFB (clean). Otherwise `Buffer*` in reader RTA → `TensorAccessor` (Case 1). |
| `sin_cache` (`sin_buffer`) | `RotaryEmbeddingLlamaMultiCorePrefillSharded` | **Case 1** (interleaved path) / clean (sharded fast-path) | Same split as cos_cache. |
| `trans_mat` (`trans_mat_buffer`) | `RotaryEmbeddingLlamaMultiCorePrefillSharded` | **Case 1** (non-global-cb path) / clean (global-cb path) | When `trans_mat_use_global_cb`: `CBDescriptor::buffer = trans_mat_buffer` → borrowed-memory DFB (clean). Otherwise `Buffer*` in reader RTA → `TensorAccessor` (Case 1). |
| `output` (`dst_buffer`) | `RotaryEmbeddingLlamaMultiCorePrefillSharded` | **Case 1** | `Buffer*` in writer RTA; kernel reads `dst_addr` → `TensorAccessor`. Output is always interleaved in prefill. |

**Op-level roll-up: `⚠ port work` — Case 1 bindings in the two prefill factories.**

### Custom hash

**Delete custom `compute_program_hash` → default (sanctioned exception).**

`device/rotary_embedding_llama_device_operation.cpp:228` — the override calls `tt::tt_metal::operation::hash_operation<RotaryEmbeddingLlamaDeviceOperation>(operation_attributes, tensor_args)`, which is the framework default hash. It is effectively a no-op override that spells out the default explicitly. Per the recipe, the port deletes it regardless — the Metal 2.0 factory concept has no role for a custom hash, and the default is correct-by-construction. Simple deletion; no relaxation candidates are present.

---

## Heads-ups *(mirrors the brief)*

### Notable LANDED constructs

**Borrowed-memory DFB (Dynamic CircularBuffer)** — present in `RotaryEmbeddingLlamaMultiCoreSharded` (all five working CBs) and conditionally in `RotaryEmbeddingLlamaMultiCorePrefillSharded` (cos/sin sharded fast-path and trans_mat global-CB path):

- `device/rotary_embedding_llama_sharded_program_factory.cpp:87-88` — `input_cb` `.buffer = src_buffer`
- `device/rotary_embedding_llama_sharded_program_factory.cpp:98-100` — `cos_cb` `.buffer = cos_buffer`
- `device/rotary_embedding_llama_sharded_program_factory.cpp:110-112` — `sin_cb` `.buffer = sin_buffer`
- `device/rotary_embedding_llama_sharded_program_factory.cpp:125-126` — `trans_mat_cb` `.buffer = trans_mat_buffer`
- `device/rotary_embedding_llama_sharded_program_factory.cpp:171-172` — `output_cb` `.buffer = dst_buffer`
- `device/rotary_embedding_llama_multi_core_prefill_sharded_program_factory.cpp:174-175` — cos_cb sharded fast-path `.buffer = cos_buffer`
- `device/rotary_embedding_llama_multi_core_prefill_sharded_program_factory.cpp:183-185` — sin_cb sharded fast-path `.buffer = sin_buffer`
- `device/rotary_embedding_llama_multi_core_prefill_sharded_program_factory.cpp:260-261` — trans_mat global-CB path `.buffer = trans_mat_buffer`

Port declares these as `DataflowBufferSpec` with `borrowed_from = <tensor_parameter_name>`.

All borrowed-memory CBs are confirmed real producer+consumer: in `RotaryEmbeddingLlamaMultiCoreSharded`, the compute kernel does `reserve_back` / `push_back` / `wait_front` / `pop_front` on every CB; the "self-serve" pattern on the same core is the canonical legit sharded case.

### Cross-op / shared kernels

None. All five kernel `.cpp` files are op-owned; no file-path kernel instantiation borrows from outside the directory. All kernel `#include` statements resolve to `tt_metal/*` framework headers (`api/dataflow/`, `api/compute/`, `api/tensor/`, `api/core_local_mem.h`) only.

### RTA varargs

None. All kernels read a small, fixed number of positional `get_arg_val<uint32_t>` entries (8 in the reader kernels, 5 in the writer, 4 in the compute kernels). No `num_runtime_varargs` reads and no loop with a runtime-varying index.

### TTNN factory analysis (porter-relevant)

- **Pybind `create_descriptor`:** None. `rotary_embedding_llama_nanobind.cpp` binds only `ttnn::bind_function<"rotary_embedding_llama", ...>` — the normal op-function surface. No `nb::class_<...ProgramFactory>` or `.def_static("create_descriptor", ...)`.
- **Other risky pybind:** None. No `DeviceOperation` methods, factory internals, or `ProgramDescriptor` introspection entry points exposed.
- **Custom `override_runtime_arguments`:** None. No `static void <Factory>::override_runtime_arguments(...)` in any factory.

---

## Team-only

### TensorAccessor convertibility (per Case-2 binding)

No Case-2 bindings. All non-clean bindings are Case 1 (page-by-page interleaved or interleaved-fallback iteration; `TensorAccessor` is the natural fit).

### Out-of-directory coupling & donor shape

**Op-level roll-up: ✓ clean.**

No cross-directory function-call escapes. Every kernel `#include` is a `tt_metal/*` framework header (class 1: LLK / HAL / firmware — no concern). No function calls into other op families or shared utility headers outside `tt_metal/`.

No borrowed file-path kernels. All five kernel `.cpp` files are owned by this op directory.

**Summary table:**

| Op kernel | Donor file | Class | Shape | Status |
|---|---|---|---|---|
| all kernels | `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/core_local_mem.h`, `api/tensor/noc_traits.h`, `api/compute/common.h`, `api/compute/eltwise_binary.h`, `api/compute/bcast.h`, `api/compute/matmul.h` | LLK/HAL | — | ✓ no concern |

No per-call detail section needed (all ✓).

**Borrowed kernel files:** None.

### Relaxation candidates (mined from the custom hash before deletion — FALLIBLE)

The custom `compute_program_hash` calls `hash_operation<RotaryEmbeddingLlamaDeviceOperation>(operation_attributes, tensor_args)` — it is effectively the default hash and hashes everything. No selective attribute exclusions to mine. No relaxation candidates surface from it.

### TTNN factory analysis (full six-question record)

1. **Op-owned tensors?** No. `create_output_tensors` (`device/rotary_embedding_llama_device_operation.cpp:222`) calls `create_device_tensor(compute_output_specs(...)[0], ...)` — this creates the op's single declared output tensor, not an internal intermediate. No factory allocates or manages additional device tensors.

2. **MeshWorkload concept needed?** No. No `create_mesh_workload`, `create_workload_descriptor`, or `cached_mesh_workload_t` anywhere. The op has no op-owned tensors (Q1 is No), so the MeshWorkload artifact path does not apply.

3. **Pybind `create_descriptor`?** No. `rotary_embedding_llama_nanobind.cpp:17-46` exposes only `bind_function<"rotary_embedding_llama", "ttnn.experimental.">` — the op function, its argument types, and defaults. No `nb::class_<...ProgramFactory>` binding.

4. **Other migration-risky pybind?** No. The nanobind file is minimal: one `bind_function` call. No `DeviceOperation` methods, factory parameter classes, or ProgramDescriptor introspection.

5. **Custom hash?** Yes. `device/rotary_embedding_llama_device_operation.cpp:228–232`. Treated as PORT WORK — delete and revert to default. See Custom program hash in Port-work summary.

6. **Custom override-runtime-args?** No. No `override_runtime_arguments` in any of the three factory `.cpp` files.

---

## Misc anomalies *(none)*

No latent code issues noticed during the audit that are not already captured above.

---

## Questions for the user *(none)*

All audit subjects resolved without ambiguity.

---

## Recipe notes *(none)*

The recipe worked cleanly for this op. No steps were unclear or contradictory; no recognition rules false-fired; no cases outside recipe scope were encountered.
