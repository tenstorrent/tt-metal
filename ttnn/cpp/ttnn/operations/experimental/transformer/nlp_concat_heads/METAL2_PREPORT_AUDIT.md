# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads`

- **`NLPConcatHeadsDeviceOperation`**
  - `NLPConcatHeadsProgramFactory` (`device/nlp_concat_heads_program_factory.cpp`)
    - Interleaved path (non-sharded input): reader kernel `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads.cpp` + borrowed writer kernel `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
    - Sharded path (sharded input): both reader and writer slots use `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_sharded.cpp`

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads` |
| **Overall** | GREEN |
| **DOps / Factories** | `NLPConcatHeadsDeviceOperation` → `NLPConcatHeadsProgramFactory` |
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
| *TTNN Readiness* — Fake CBs (address-only) | present: `(cb_in0, sharded reader)` and `(cb_out0, sharded writer)` in `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_sharded.cpp` (workaround) |

**Fake CBs** = CBs used purely as an address source. In the sharded path, both `cb_in0` (index 0) and `cb_out0` (index 16) are borrowed-memory CBs (`CBDescriptor::buffer` set) but have no genuine FIFO producer-consumer pair — `cb_in0.reserve_back` is called on the input CB (marked "Redundant" in a comment) with no `wait_front`/`pop_front`, and `cb_out0.push_back` is commented out (`// cb_out0.push_back(block_size);`). Both CBs are used only to obtain base L1 addresses. Port resolves these with the sanctioned fake-CB workaround (see the porting recipe); does **not** gate.

## Result

**GREEN → brief issued.** All gates cleared. Two Case-1 tensor bindings require port work (interleaved path only). Two fake CBs in the sharded path require the fake-CB workaround. The borrowed eltwise/unary writer kernel induces a port-together coupling with other ops that share it.

## Gate detail

- **ProgramDescriptor:** GREEN — `NLPConcatHeadsProgramFactory::create_descriptor` returns a `ProgramDescriptor` populated with `KernelDescriptor`, `CBDescriptor` structs. No imperative `host_api.hpp` builder calls. (`device/nlp_concat_heads_program_factory.cpp`)

- **Device 2.0 (every kernel used):** GREEN — all three kernels use the Device 2.0 wrapper API exclusively:
  - `reader_tm_tile_layout_nlp_concat_heads.cpp`: includes `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, uses `Noc noc`, `CircularBuffer cb_in0(cb_id_in0)`, `noc.async_read(...)`, `cb_in0.reserve_back(...)` / `cb_in0.push_back(...)`. `get_tile_size(cb_id_in0)` at line 30 is a sanctioned free function. No legacy Device 1.0 idioms.
  - `reader_tm_tile_layout_nlp_concat_heads_sharded.cpp`: includes `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/dataflow/endpoints.h`, uses `Noc noc`, `CircularBuffer cb_in0(...)`, `CircularBuffer cb_out0(...)`, `noc.async_read(...)`. `get_tile_size(cb_id_in0)` at line 30 is sanctioned. `cb_in0.get_read_ptr()` (line 43) and `cb_out0.get_write_ptr()` (line 44) are member-form calls on wrapper objects — Device 2.0 compliant.
  - `writer_unary_interleaved_start_id.cpp` (eltwise/unary cross-family donor): includes `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, uses `Noc noc`, `CircularBuffer cb(cb_id_out)`, `noc.async_write(...)`, `cb.wait_front(...)` / `cb.pop_front(...)`. `get_local_cb_interface(cb_id_out)` at line 19 is a sanctioned free function per the Device 2.0 migration guide. No legacy Device 1.0 idioms.

- **Feature compatibility:**

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type or `CreateGlobalCircularBuffer` call found |
  | Dynamic CircularBuffer (borrowed memory) | GREEN | `CBDescriptor::buffer` set to `in0_buffer` (sharded input) and `out_buffer` (sharded output) in `nlp_concat_heads_program_factory.cpp:150,163`. Port uses `DataflowBufferSpec::borrowed_from`. |
  | CBDescriptor `address_offset` (non-zero) | N/A | `.address_offset` field not set anywhere; defaults to zero |
  | Aliased Circular Buffers | N/A | Both `CBDescriptor` initializers use single-element `format_descriptors` (`{{CBFormatDescriptor{...}}}`) |
  | GlobalSemaphore | N/A | No semaphores of any kind in this op |
  | Non-zero semaphore initial value | N/A | No semaphores present |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | No `ArgConfig::Runtime*` usage; both `TensorAccessorArgs` calls are single-argument form (`nlp_concat_heads_program_factory.cpp:117,119`) |
  | `UpdateCircularBuffer*` | N/A | No `UpdateCircularBuffer*` calls found |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t = Tensor` (single fixed input); all `get_compile_time_arg_val` calls use fixed indices |

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, interleaved path only):
  - `input` (`in0_buffer`) — **Case 1** (re-express via `TensorParameter` / `TensorBinding`). Factory: `in0_buffer` pushed as `Buffer*` into `reader_desc.emplace_runtime_args` (line 201). Kernel: `in0_tensor_addr = get_arg_val<uint32_t>(0)` used as `TensorAccessor(in0_args, in0_tensor_addr)` in `reader_tm_tile_layout_nlp_concat_heads.cpp:17,31`. CTA layout metadata injected via `TensorAccessorArgs(*in0_buffer).append_to(...)` at factory line 117 and `TensorAccessorArgs<4>()` at kernel line 27 — both disappear on Metal 2.0 port.
  - `output` (`out_buffer`) — **Case 1** (re-express via `TensorParameter` / `TensorBinding`). Factory: `out_buffer` pushed as `Buffer*` into `writer_desc.emplace_runtime_args` (line 210). Kernel: `dst_addr = get_arg_val<uint32_t>(0)` used as `TensorAccessor(dst_args, dst_addr)` in `writer_unary_interleaved_start_id.cpp:11,31`. CTA layout metadata injected via `TensorAccessorArgs(*out_buffer).append_to(...)` at factory line 119 and `TensorAccessorArgs<1>()` at kernel line 16 — both disappear on Metal 2.0 port.
  - Sharded path: input and output accessed through fake borrowed-memory CBs (see Fake CBs below). No `TensorParameter` bindings needed there; the fake-CB workaround handles them.
- **Custom hash:** none

## Heads-ups  *(mirrors the brief)*

- **Notable LANDED constructs:**
  - Borrowed-memory DFB: `CBDescriptor::buffer = in0_buffer` (sharded-input CB, cb_id=0) at `nlp_concat_heads_program_factory.cpp:150` and `CBDescriptor::buffer = out_buffer` (sharded-output CB, cb_id=16) at line 163. Both are in the sharded code path only. Port declares these as `DataflowBufferSpec::borrowed_from` naming the appropriate `TensorParameter`. **Note:** these are also fake CBs (see below) — the borrowed-from declaration still applies to the spec; the fake-CB workaround addresses the missing producer/consumer pair.

- **Fake CBs (address-only):** Sharded path in `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_sharded.cpp`:
  - `(cb_in0=0, reader endpoint)` at lines 35,43: `cb_in0.reserve_back(block_size);  // Redundant` with no `wait_front`/`pop_front`; used only as `cb_in0.get_read_ptr()` address source.
  - `(cb_out0=16, writer endpoint)` at lines 36,44,62: `cb_out0.reserve_back(block_size)` with `// cb_out0.push_back(block_size);` commented out; used only as `cb_out0.get_write_ptr()` address source.
  - Port resolves both with the sanctioned fake-CB workaround (see the porting recipe). Does **not** gate.

- **Cross-op / shared kernels:**
  - Interleaved path instantiates `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (cross-family, owned by eltwise/unary). This is a broadly shared kernel used by at least: `eltwise/unary_backward/tanh_bw`, `eltwise/unary_backward/gelu_bw`, `experimental/transformer/nlp_concat_heads_boltz`, `experimental/matmul/attn_matmul`, `embedding`, `kv_cache/fill_cache_multi_core`, `reduction/prod`, `reduction/generic` (reduce variants), `data_movement/concat`, `examples/example`. Any Metal 2.0 rewrite of this kernel must be coordinated as a single change across all co-borrowers.

- **RTA varargs:** none
- **TTNN factory analysis (porter-relevant):** pybind `create_descriptor` — none; other migration-risky pybind — none; custom `override_runtime_arguments` — none.

## Team-only

### TensorAccessor convertibility

Both input and output bindings are **Case 1** (re-express). The interleaved reader uses standard page-by-page `TensorAccessor` iteration (one tile at a time, sequential page IDs). The writer uses the same standard tile-sequential pattern. Neither is exotic — both are straightforwardly convertible via `TensorParameter` / `TensorBinding`.

### Out-of-directory coupling & donor shape analysis

**Op-level roll-up:** `⚠ workable` — one cross-family kernel file instantiation; no function-call escapes outside the op's own kernel files.

**Summary table:**

| Op kernel | Donor file | Type | Status |
|---|---|---|---|
| `nlp_concat_heads_program_factory.cpp` (interleaved path) | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | Cross-family file-path instantiation | ⚠ port-together coupling |
| `reader_tm_tile_layout_nlp_concat_heads.cpp` | (no external includes beyond `api/dataflow/*`, `api/core_local_mem.h`, `api/tensor/noc_traits.h`) | LLK / HAL / Device 2.0 API | ✓ clean |
| `reader_tm_tile_layout_nlp_concat_heads_sharded.cpp` | (no external includes beyond `api/dataflow/*`, `api/core_local_mem.h`) | LLK / HAL / Device 2.0 API | ✓ clean |
| `writer_unary_interleaved_start_id.cpp` | (no external includes beyond `api/dataflow/*`, `api/tensor/noc_traits.h`) | LLK / HAL / Device 2.0 API | ✓ clean |

**Borrowed kernel files (file-path instantiation):**
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
  - Owning family: `eltwise/unary`
  - Broadly shared: yes — confirmed other users include `eltwise/unary_backward/tanh_bw`, `eltwise/unary_backward/gelu_bw`, `experimental/transformer/nlp_concat_heads_boltz`, `experimental/matmul/attn_matmul`, `embedding`, `kv_cache/fill_cache_multi_core`, `reduction/prod`, `reduction/generic` (multiple factories), `data_movement/concat`, `examples/example`.
  - Port-together implication: the Metal 2.0 rewrite of `writer_unary_interleaved_start_id.cpp` (CB→DFB, named-token bindings) is a **single rewrite** that all co-borrowers must adopt simultaneously. Coordinate with the eltwise/unary porter and all other families above before touching this kernel.
  - The kernel is Device 2.0 compliant and has no function-call escapes outside Device 2.0 API headers. No coupling issues beyond the port-together requirement.

**Per-call detail:** Omitted — all kernel includes resolve to Device 2.0 API headers or LLK/HAL (`api/dataflow/*`, `api/core_local_mem.h`, `api/tensor/noc_traits.h`, `api/dataflow/endpoints.h`). No function-call escapes into donor op-family headers with ⚠/✗/⭐ shapes. The eltwise/unary kernel's `TensorAccessorArgs<1>` (line 16) uses the standard single-argument form, translating cleanly to `TensorParameter` / `TensorBinding` — Shape 1 (✓ excellent).

### Relaxation candidates

No custom `compute_program_hash` present; no relaxation candidates to mine.

### TTNN factory analysis

1. **Op-owned tensors?** No. `create_output_tensors` calls `create_device_tensor(compute_output_specs(...), input_tensor.device())` — this creates the op's declared output tensor, not an intermediate/scratch tensor. No additional device tensors allocated in the factory or device-op `.cpp`. (`device/nlp_concat_heads_device_operation.cpp:93`)

2. **MeshWorkload concept needed?** No. No `create_mesh_workload`, `create_workload_descriptor`, or `cached_mesh_workload_t` present. Single-program op. Q1 is also No, so no op-owned-tensor artifact either.

3. **Pybind `create_descriptor`?** No. `nlp_concat_heads_nanobind.cpp` only binds the op function itself via `bind_function<"nlp_concat_heads", "ttnn.experimental.">` — normal op surface. No `nb::class_<…ProgramFactory>` or `create_descriptor` binding.

4. **Other migration-risky pybind?** None. The nanobind file is minimal: one `bind_function` call wrapping the user-facing `ttnn::experimental::nlp_concat_heads`. No DeviceOperation methods, factory internals, or param structs exposed.

5. **Custom hash?** No. `NLPConcatHeadsDeviceOperation` does not declare `compute_program_hash`. Default TTNN hash applies.

6. **Custom override-runtime-args?** No. `NLPConcatHeadsProgramFactory` does not define `override_runtime_arguments`.

## Misc anomalies  *(team-only, non-gating)*

- `nlp_concat_heads_program_factory.cpp:5`: `#include <tt-metalium/host_api.hpp>` is present but no imperative builder API calls are made. The include is likely a legacy residue from before the ProgramDescriptor migration; it can be removed.
- `reader_tm_tile_layout_nlp_concat_heads_sharded.cpp:62`: `// cb_out0.push_back(block_size);` is commented out. This is the root cause of the fake-CB pattern on `cb_out0`. The comment is silent on why it was disabled. Op owner should review whether this was intentional or a latent bug (if the sharded output is a real FIFO consumer somewhere, the missing push_back may affect correctness; if purely address-based, the fake-CB workaround at port time is correct and the comment should be made explicit).
- `reader_tm_tile_layout_nlp_concat_heads_sharded.cpp:35`: `cb_in0.reserve_back(block_size);  // Redundant` — calling `reserve_back` on the input CB (which in the sharded model is pre-populated by the runtime before the kernel runs) is indeed redundant, and confirms the fake-CB shape. This line should be removed or left as-is with the fake-CB workaround acknowledgment.

## Questions for the user  *(omit if none)*

None — all findings have clear dispositions.

## Recipe notes  *(omit if none)*

None.
