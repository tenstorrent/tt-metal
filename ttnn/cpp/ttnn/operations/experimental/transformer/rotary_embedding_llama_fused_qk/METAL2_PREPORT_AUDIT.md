# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk`

- **`RotaryEmbeddingLlamaFusedQKDeviceOperation`**
  - `RotaryEmbeddingLlamaFusedQKProgramFactory` (`device/rotary_embedding_llama_fused_qk_program_factory.cpp`)
    - Compute kernel (tiled path): `device/kernels/compute/rotary_embedding_llama_sharded.cpp`
    - Compute kernel (row-major path): `device/kernels/compute/rotary_embedding_llama_sharded_row_major.cpp`

The factory selects between the two compute kernels based on `operation_attributes.row_major_QK`. Both kernels have the same structure; differences are noted where they exist.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk` |
| **Overall** | GREEN |
| **DOps / Factories** | `RotaryEmbeddingLlamaFusedQKDeviceOperation` → `RotaryEmbeddingLlamaFusedQKProgramFactory` |
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
| *TTNN Readiness* — Fake CBs (address-only) | present: `(cos c_2, compute)`, `(sin c_3, compute)`, `(trans_mat c_4, compute)`, `(q_output c_16, compute)`, `(k_output c_17, compute)` (workaround) |

**Fake CBs** = CBs used purely as an address source. **Litmus: does the CB have a producer *and* a consumer?** (Same core may be both.) The five CBs listed above fail this test — see Gate detail / Feature compatibility and Heads-ups for the per-CB breakdown and the port workaround reference.

## Result

**GREEN → brief issued.** All gates cleared. The port can proceed after explicit user go-ahead. Five fake CBs (address-only, no producer+consumer pair) are present; the port resolves each with the sanctioned fake-CB workaround (see the porting recipe). No gate failures.

## Gate detail

- **ProgramDescriptor:** GREEN — the factory populates a `ProgramDescriptor` with `CBDescriptor`, `KernelDescriptor`, and compile-time/runtime args. No imperative `host_api.hpp` calls (`CreateKernel`, `SetRuntimeArgs`, etc.) anywhere in the op. (`device/rotary_embedding_llama_fused_qk_program_factory.cpp:18`)

- **Device 2.0 (every kernel used):** GREEN — both compute kernels use the Device 2.0 `CircularBuffer` wrapper objects for all FIFO-active circular buffers. The kernels include `api/dataflow/circular_buffer.h` and instantiate `CircularBuffer obj(cb_id)` wrappers, calling `.reserve_back()`, `.push_back()`, `.wait_front()`, `.pop_front()` on them. No legacy dataflow idioms are present: no `InterleavedAddrGen`, no `ShardedAddrGen`, no `InterleavedAddrGenFast`, no `InterleavedPow2AddrGen*`, no raw `noc_async_read`/`noc_async_write`, no raw semaphore addresses. The kernels are **pure compute kernels** — they perform no dataflow (NoC reads/writes); all tensor data arrives pre-loaded in sharded CBs backed by borrowed memory.

  Three `CircularBuffer` wrapper objects are instantiated in `rotary_embedding_llama_sharded.cpp` (lines 60–62: `cos_cb_obj`, `sin_cb_obj`, `trans_mat_cb_obj`) but their methods are never called. The raw `uint32_t` CB indices for these three are used only as arguments to LLK compute functions (`mm_init`, `matmul_tiles`, `mul_tiles_bcast`, etc.) from `api/compute/*`. LLK compute calls taking a raw CB index are not dataflow free functions and are not Device 2.0 holdovers — they are the correct API surface for the compute engine. No action required on Device 2.0.

  Dead comment blocks in `rotary_embedding_llama_sharded.cpp` (lines 70–84 and 146–153) contain legacy-style `cb_reserve_back(trans_mat_cb, ...)` / `cb_push_back(...)` / `cb_wait_front(...)` / `cb_pop_front(...)` free-function calls, but these are fully commented out (`/* ... */`) and do not appear in any active code path.

- **Feature compatibility:** every Appendix A entry:

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `experimental::CreateGlobalCircularBuffer`, no `global_circular_buffer` field set on any `CBDescriptor` |
  | Dynamic CircularBuffer (borrowed memory) | GREEN | Seven `CBDescriptor`s have `.buffer` set to non-null `Buffer*`: `q_src_buffer` (c_0), `k_src_buffer` (c_1), `cos_buffer` (c_2), `sin_buffer` (c_3), `trans_mat_buffer` (c_4), `q_dst_buffer` (c_16), `k_dst_buffer` (c_17). All seven borrow sharded tensor memory. Port uses `DataflowBufferSpec::borrowed_from` for each. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` field set anywhere in the factory |
  | Aliased Circular Buffers | N/A | Every `CBDescriptor::format_descriptors` is single-element; no aliased CBs |
  | GlobalSemaphore | N/A | No semaphores at all in this op (no `SemaphoreDescriptor`, no `CreateSemaphore`) |
  | Non-zero semaphore initial value | N/A | No semaphores in this op |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | No `TensorAccessorArgs` anywhere; op uses sharded/borrowed-memory CB pattern throughout |
  | `UpdateCircularBuffer*` | N/A | No `UpdateCircularBufferTotalSize`, no `UpdateCircularBufferPageSize`, no `UpdateDynamicCircularBufferAddressAndTotalSize` |
  | Variable-count compile-time arguments (CTA varargs) | N/A | Fixed count of 13 CTAs (indices 0–12), all read via `constexpr ... = get_compile_time_arg_val(N)` with statically-known N. `tensor_args_t` contains exactly 5 named `Tensor` members (no `std::vector<Tensor>`). |

## Port-work summary *(mirrors the brief)*

- **Tensor bindings:** N/A — both kernels are pure compute; they access tensor data through borrowed-memory CBs, not `TensorAccessor`. The causal-link gate applies to the input CBs (c_0, c_1) which have genuine producer+consumer cycles. The output and cos/sin/trans_mat CBs are fake CBs resolved by workaround (see Heads-ups).
- **Custom hash:** none.

## Heads-ups *(mirrors the brief)*

- **Notable LANDED constructs — borrowed-memory DFBs:** Seven CBs use `CBDescriptor::buffer` (Dynamic CircularBuffer feature, LANDED). Port uses `DataflowBufferSpec::borrowed_from` for each. `device/rotary_embedding_llama_fused_qk_program_factory.cpp` lines 95–211:
  - `q_input_cb` (c_0) `.buffer = q_src_buffer` @ line 103
  - `k_input_cb` (c_1) `.buffer = k_src_buffer` @ line 115
  - `cos_cb` (c_2) `.buffer = cos_buffer` @ line 127
  - `sin_cb` (c_3) `.buffer = sin_buffer` @ line 139
  - `trans_mat_cb` (c_4) `.buffer = trans_mat_buffer` @ line 153
  - `q_output_cb` (c_16) `.buffer = q_dst_buffer` @ line 199
  - `k_output_cb` (c_17) `.buffer = k_dst_buffer` @ line 210

- **Fake CBs (address-only):** Five of the seven borrowed-memory CBs are fake — the compute kernel does not form a genuine producer+consumer FIFO on them:
  - `cos_cb` (c_2): `CircularBuffer cos_cb_obj` instantiated at `rotary_embedding_llama_sharded.cpp:60` but no wrapper methods ever called; raw `cos_cb` uint32_t used only in LLK calls. No push/wait cycle. `rotary_embedding_llama_sharded_row_major.cpp`: not even instantiated as a wrapper object. Both kernels: **fake**.
  - `sin_cb` (c_3): same pattern. `rotary_embedding_llama_sharded.cpp:61`; `rotary_embedding_llama_sharded_row_major.cpp`: no wrapper. Both kernels: **fake**.
  - `trans_mat_cb` (c_4): same pattern. `rotary_embedding_llama_sharded.cpp:62`; `rotary_embedding_llama_sharded_row_major.cpp`: no wrapper. Both kernels: **fake**.
  - `q_output_cb` (c_16): `out_cb_obj.reserve_back(Wt)` and `out_cb_obj.push_back(Wt)` called but no `wait_front` / `pop_front`. Producer only; no consumer. `rotary_embedding_llama_sharded.cpp:90,141`. Both kernels: **fake** (same pattern in row_major variant).
  - `k_output_cb` (c_17): same as q_output_cb. Both kernels: **fake**.

  The two non-fake borrowed-memory CBs — `q_input_cb` (c_0) and `k_input_cb` (c_1) — do have real FIFO cycles: the compute kernel calls `.reserve_back(Wt)`, `.push_back(Wt)`, `.wait_front(Wt)`, `.pop_front(Wt)` on them in sequence (same core acts as both producer and consumer). These translate as genuine borrowed-memory DFBs via `DataflowBufferSpec::borrowed_from`.

  **The port resolves each fake CB with the sanctioned fake-CB workaround** (see the porting recipe). This does **not** gate the port.

- **Cross-op / shared kernels:** None. Both kernel files are owned by this op; all `#include`s resolve to `api/compute/*` (LLK/HAL) and `api/dataflow/circular_buffer.h` (LLK/HAL). No borrowed kernel files. No out-of-directory function calls.

- **RTA varargs:** None. One runtime arg per core (the `is_q` flag, a single `uint32_t`), set via `compute_desc.runtime_args.emplace_back(core, std::vector<uint32_t>{is_q_or_k_arg})`. Fixed count, not a vararg pattern.

- **TTNN factory analysis (porter-relevant):**
  - Pybind `create_descriptor`: None. The nanobind file (`rotary_embedding_llama_fused_qk_nanobind.cpp`) uses `ttnn::bind_function<"rotary_embedding_llama_fused_qk", ...>` — standard op-function binding only. No `nb::class_<...ProgramFactory>(...).def_static("create_descriptor", ...)`.
  - Other migration-risky pybind: None.
  - Custom `override_runtime_arguments`: None.

## Team-only

### TensorAccessor convertibility
Not applicable. Both kernels are pure compute; tensor memory access is entirely through borrowed-memory CBs. No `TensorAccessor` present in either kernel. The causal-link gate applies to c_0 and c_1 (genuine borrowed-memory DFBs); the five fake CBs are resolved by the port workaround, not via TensorAccessor.

### Out-of-directory coupling & donor shape

**Op-level roll-up:** `✓ clean` — all cross-directory includes resolve to the `tt_metal/*` / LLK/HAL tier. No cross-family or in-family function-call escapes.

**Summary table:**

| Op kernel | Included path | Donor class | Status |
|---|---|---|---|
| `rotary_embedding_llama_sharded.cpp` | `api/compute/common.h` | LLK/HAL | ✓ no concern |
| `rotary_embedding_llama_sharded.cpp` | `api/compute/eltwise_binary.h` | LLK/HAL | ✓ no concern |
| `rotary_embedding_llama_sharded.cpp` | `api/compute/bcast.h` | LLK/HAL | ✓ no concern |
| `rotary_embedding_llama_sharded.cpp` | `api/compute/matmul.h` | LLK/HAL | ✓ no concern |
| `rotary_embedding_llama_sharded.cpp` | `api/dataflow/circular_buffer.h` | LLK/HAL | ✓ no concern |
| `rotary_embedding_llama_sharded_row_major.cpp` | `api/compute/common.h` | LLK/HAL | ✓ no concern |
| `rotary_embedding_llama_sharded_row_major.cpp` | `api/compute/eltwise_binary.h` | LLK/HAL | ✓ no concern |
| `rotary_embedding_llama_sharded_row_major.cpp` | `api/compute/bcast.h` | LLK/HAL | ✓ no concern |
| `rotary_embedding_llama_sharded_row_major.cpp` | `api/compute/matmul.h` | LLK/HAL | ✓ no concern |
| `rotary_embedding_llama_sharded_row_major.cpp` | `api/dataflow/circular_buffer.h` | LLK/HAL | ✓ no concern |

**Borrowed kernel files (file-path kernel instantiation):** None — both kernel `.cpp` files are owned by this op. No shared-pool or cross-family borrowed kernels.

### Relaxation candidates
No custom `compute_program_hash` — no relaxation candidates to mine.

### TTNN factory analysis

1. **Op-owned tensors?** No. `create_output_tensors` calls `create_device_tensor` for the two outputs (q and k), but these are the declared outputs of the op, not intermediate op-owned tensors. There are no intermediate tensors allocated and owned by the factory beyond the declared `tensor_return_value`. (`device/rotary_embedding_llama_fused_qk_device_operation.cpp:134–136`)

2. **MeshWorkload concept needed?** No. Single-program op; no `create_mesh_workload`, no `create_workload_descriptor`, no `cached_mesh_workload_t`. Q1 is No (no op-owned tensors), so the MeshWorkload path is not induced as an artifact either.

3. **Pybind `create_descriptor`?** No. `rotary_embedding_llama_fused_qk_nanobind.cpp:18` binds the op function via `ttnn::bind_function<"rotary_embedding_llama_fused_qk", "ttnn.experimental.">`. No `nb::class_<RotaryEmbeddingLlamaFusedQKProgramFactory>` binding; `create_descriptor` is not exposed to Python.

4. **Other migration-risky pybind?** No. The nanobind file binds only the top-level op function. No `DeviceOperation` methods, no factory/param structs, no introspection entry points are pybind-exposed.

5. **Custom hash?** No `compute_program_hash` declaration or override anywhere in the op's host code.

6. **Custom override-runtime-args?** No `override_runtime_arguments` static method in any factory.

## Misc anomalies *(team-only, non-gating)*

- **Dead `CircularBuffer` wrapper objects in `rotary_embedding_llama_sharded.cpp`:** `cos_cb_obj` (line 60), `sin_cb_obj` (line 61), and `trans_mat_cb_obj` (line 62) are instantiated but never used — their wrapper methods are never called. The raw uint32_t identifiers (`cos_cb`, `sin_cb`, `trans_mat_cb`) are used directly in LLK calls instead. These are harmless dead instantiations; the port's fake-CB workaround will replace them. Worth noting for the op owner as a minor style oddity.

- **Dead comment blocks with legacy `cb_*` free-function calls:** `rotary_embedding_llama_sharded.cpp` lines 70–84 and 146–153 are commented-out code blocks containing `cb_reserve_back`, `cb_push_back`, `cb_wait_front`, `cb_pop_front` calls on `trans_mat_cb`, `sin_cb`, `cos_cb`. Labeled "Unnecessary CB APIs (comment out for code size)." Not active; the port can ignore or delete them.

- **TODO comment re: batch_per_core:** `device/rotary_embedding_llama_fused_qk_program_factory.cpp:80` — `const uint32_t batch_per_core = 1;  // TODO: To make general, add support for batch_per_core > 1`. Not an audit finding, but the hardcoded value is an open generalization item for the op owner.

## Questions for the user *(none)*

## Recipe notes

- The fake-CB litmus ("does the CB have a producer *and* a consumer?") is applied per (CB, endpoint) edge. For pure compute-only ops with no dataflow kernels, *all* borrowed-memory CBs that aren't explicitly looped through a push/wait cycle by the compute kernel end up fake. This op has 5 of 7 borrowed-memory CBs classified as fake. The recipe's examples for the fake-CB pattern reference "a sharded reader's fake-push satisfying a waiting compute consumer" — the inverse (compute kernel driving both a push and wait on itself for an input CB) is the "real" case here, and it is easy to identify. No friction with the recipe on this front, but noting it for any auditor who encounters a similarly compute-only op.
