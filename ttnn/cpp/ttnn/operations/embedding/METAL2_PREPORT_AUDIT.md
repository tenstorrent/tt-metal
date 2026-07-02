# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/embedding`

Single DeviceOperation with three ProgramFactories:

- **`EmbeddingsDeviceOperation`**
  - `EmbeddingsRMProgramFactory` (`embeddings_rm_program_factory.cpp`) — row-major, non-tilized output
  - `EmbeddingsFusedProgramFactory` (`embeddings_fused_program_factory.cpp`) — fused reader + tilize; tilized output
  - `EmbeddingsTilizedIndicesProgramFactory` (`embeddings_tilized_indices_program_factory.cpp`) — tilized-index (TILE_LAYOUT input)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/embedding` |
| **Overall** | RED |
| **DOps / Factories** | `EmbeddingsDeviceOperation` → `EmbeddingsRMProgramFactory`, `EmbeddingsFusedProgramFactory`, `EmbeddingsTilizedIndicesProgramFactory` |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | No — `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` is on Device 1.0 idioms; used by `EmbeddingsRMProgramFactory` (non-chunked non-sharded path) and `EmbeddingsTilizedIndicesProgramFactory` |
| *Prereqs* — Cross-op escapes | issue — `writer_unary_stick_layout_interleaved_start_id.cpp` (Device 1.0); see Device 2.0 gate |
| *Feature Support* — overall | GREEN (no UNSUPPORTED features in use) |
| *Feature Support* — Variadic-CTA | N/A |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — MeshWorkload needed | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | present: `src1_cb_index` (c_1) in all three factories — indices scratch; `src2_cb_index` (c_2/c_3) in PADDED and BINARY modes — pad/binary cache scratch (workaround) |

## Result

**RED → blocked on Device 2.0 gate**, routed to the Device 2.0 migration team.

The borrowed shared kernel `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` uses Device 1.0 idioms throughout (`cb_wait_front`, `get_read_ptr`, `noc_async_write`, `noc_async_write_barrier`). It is instantiated by `EmbeddingsRMProgramFactory` (non-chunked, non-sharded writer path) and `EmbeddingsTilizedIndicesProgramFactory` (always-on writer). The port is blocked on this kernel until its Device 2.0 migration lands. The `EmbeddingsFusedProgramFactory` uses a different borrowed kernel (`writer_unary_interleaved_start_id.cpp` from eltwise/unary) that is Device 2.0 compliant, and `EmbeddingsRMProgramFactory`'s chunked-writer path uses its own `embeddings_rm_writer_chunked.cpp` which is also Device 2.0 compliant — so those sub-paths are structurally clear once the blocking kernel migrates.

Future path: op can be ported to Metal 2.0 once the Device 2.0 migration of `writer_unary_stick_layout_interleaved_start_id.cpp` lands. That kernel is broadly shared (also used by `data_movement/concat`, `data_movement/slice`, `data_movement/copy`, `data_movement/indexed_fill`) — its migration is shared infrastructure that unblocks multiple ops simultaneously.

## Gate detail

- **ProgramDescriptor:** GREEN — all three factories declare `create_descriptor(...)` returning `tt::tt_metal::ProgramDescriptor`. Headers confirm this: `embeddings_rm_program_factory.hpp:13`, `embeddings_fused_program_factory.hpp:13`, `embeddings_tilized_indices_program_factory.hpp:13`.

- **Device 2.0 (every kernel used):**

  **Own kernels** — GREEN for all embedding-owned kernels:
  - `kernels/dataflow/embeddings.cpp` — `Noc`, `CircularBuffer` wrappers, `TensorAccessor` — Device 2.0 compliant.
  - `kernels/dataflow/embeddings_tilize.cpp` — same Device 2.0 pattern.
  - `kernels/dataflow/embedding_ind_tilized.cpp` — same.
  - `kernels/dataflow/embeddings_rm_writer_chunked.cpp` — `Noc`, `CircularBuffer`, `TensorAccessor` — Device 2.0 compliant.
  - `kernels/dataflow/embeddings_common.hpp` — Device 2.0 helpers used by all three reader kernels (`Noc`, `CircularBuffer`, `TensorAccessor`).
  - `kernels/compute/tilize_chunked.cpp` — compute kernel using `api/compute/tilize.h` and `kernel_lib/tilize_helpers.hpp` — Device 2.0 compliant.

  **Borrowed kernels** — two compliant, one RED:

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | 26 | `cb_wait_front(cb_id_out0, 1)` | None — no `CircularBuffer` wrapper created |
  | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | 27 | `get_read_ptr(cb_id_out0)` | None — same |
  | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | 29 | `noc_async_write(l1_read_addr, dst_noc_addr, stick_size)` | No `Noc` object |
  | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | 30 | `noc_async_write_barrier()` | No `Noc` object |

  `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — Device 2.0 compliant (`Noc`, `CircularBuffer` wrappers; `get_local_cb_interface` is a sanctioned free function). Used only by `EmbeddingsFusedProgramFactory` (non-sharded path).

  `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize.cpp` — Device 2.0 compliant; compute-only, no NoC. Used by `EmbeddingsFusedProgramFactory` (non-chunked path).

- **Feature compatibility:**

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` or `global_circular_buffer` field usage |
  | Dynamic CircularBuffer (borrowed memory) | GREEN | `EmbeddingsRMProgramFactory` (sharded path, line 134) and `EmbeddingsFusedProgramFactory` (sharded path, line 173) set `CBDescriptor::buffer = out_buffer` — output CB backed by the output buffer. Port uses `borrowed_from`. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `address_offset` field set anywhere in the factories |
  | Aliased Circular Buffers | N/A | All `format_descriptors` initializers are single-element `{{CBFormatDescriptor{...}}}` |
  | GlobalSemaphore | N/A | No semaphores used anywhere in the op |
  | Non-zero semaphore initial value | N/A | No semaphores |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | All `TensorAccessorArgs(buffer)` calls are single-arg; no `ArgConfig::Runtime*` tokens |
  | `UpdateCircularBuffer*` | N/A | No `UpdateCircularBuffer` calls anywhere |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` holds fixed-count fields (`input_tensor_arg`, `weight_arg`, `optional_output_tensor`); all CTA reads use fixed-index `get_compile_time_arg_val(N)` |

## Port-work summary *(mirrors the brief)*

*(Not issued — port is RED on Device 2.0 gate. Recorded here as forward-looking context for when the prereq clears.)*

- **Tensor bindings** (per binding, all three factories):
  - `input` — **Case 1** (re-express): host passes `a_buffer` (Buffer*) as RTA arg 0, accessor metadata as CTAs; kernel reconstructs via `TensorAccessor(input_args, input_buffer_src_addr)`.
  - `weights` — **Case 1** (re-express): same pattern as `input`; host passes `weights_buffer` as RTA arg 1.
  - `output` — **Case 1** (re-express): host passes `output_buffer` as RTA arg 0 in writers; kernel reconstructs via `TensorAccessor(dst0_args, dst_addr)`. In the sharded path, output is the borrowed-memory DFB — clean via `borrowed_from`.
- **Custom hash:** none.

## Heads-ups *(mirrors the brief)*

*(Not issued — port is RED. Forward-looking context only.)*

- **Notable LANDED constructs:**
  - Dynamic (borrowed-memory) DFB: `EmbeddingsRMProgramFactory:embeddings_rm_program_factory.cpp:134` (sharded output path, `out_cb_desc.buffer = out_buffer`) and `EmbeddingsFusedProgramFactory:embeddings_fused_program_factory.cpp:173` (same pattern). Port declares these `DataflowBufferSpec`s with `borrowed_from = output`. In the sharded path there is no separate writer kernel; the borrowed-memory DFB IS the output.
- **Fake CBs (address-only):**
  - `(src1_cb_index / c_1, embeddings.cpp)` — created via `cb_in1.reserve_back(1)` + `get_write_ptr()` as indices scratch; never `push_back`'d; no consumer. Fake CB in all three factories (reader kernels all use this same pattern).
  - `(src2_cb_index / c_2 or c_3, embeddings_common.hpp:prepare_local_cache, PADDED mode)` — `cb.reserve_back(1)` + `get_write_ptr()` for pad-token cache; never pushed. Fake CB.
  - `(src2_cb_index / c_2 or c_3, embeddings_common.hpp:prepare_local_cache, BINARY mode)` — `cb.reserve_back(2)` + `get_write_ptr()` for zero/one embedding cache; never pushed. Fake CB.
  - Port resolves each via the sanctioned fake-CB workaround (see port recipe).
- **Cross-op / shared kernels:** see Out-of-directory coupling below. The `writer_unary_stick_layout_interleaved_start_id.cpp` borrow is the Device 2.0 blocker; its migration forms a port-together set with `data_movement/concat`, `data_movement/slice`, `data_movement/copy`, `data_movement/indexed_fill`.
- **RTA varargs:** none.
- **TTNN factory analysis (porter-relevant):** no `create_descriptor` pybind; no other risky pybind; no custom `override_runtime_arguments`.

## Team-only

### TensorAccessor convertibility

No Case-2 bindings — all bindings are Case 1 (Buffer* RTA + CTA metadata). No convertibility annotation needed.

### Out-of-directory coupling & donor shape

**Op-level roll-up:** ⭐ blocked (one borrowed kernel pre-Device-2.0; Device 2.0 GATE recorded above).

**Summary table:**

| Op kernel | Donor file | Donor class | Shape | Status |
|---|---|---|---|---|
| `embeddings_rm_program_factory.cpp` (non-chunked, non-sharded writer) | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | Shared pool (`ttnn/cpp/ttnn/kernel/`) | Pre-Device-2.0 — no `Noc`/`CircularBuffer` wrappers | ⭐ Device 2.0 GATE |
| `embeddings_tilized_indices_program_factory.cpp` (writer) | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | Shared pool (`ttnn/cpp/ttnn/kernel/`) | Pre-Device-2.0 | ⭐ Device 2.0 GATE |
| `embeddings_fused_program_factory.cpp` (non-sharded writer) | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | Cross-family (eltwise/unary) | Device 2.0 — `Noc`, `CircularBuffer`, `TensorAccessor`, sanctioned `get_local_cb_interface` | ✓ OK |
| `embeddings_fused_program_factory.cpp` (non-chunked compute) | `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize.cpp` | Cross-family (data_movement/tilize) | Compute-only; `uint32_t cb_id` via CTAs | ✓ OK |
| `embeddings_fused_program_factory.cpp` (chunked compute) + own `tilize_chunked.cpp` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` | Official shared kernel library | kernel_lib — handled internally | ✓ OK |

**Per-call detail (blocked donor):**

`ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` — `kernel_main()` uses Device 1.0 throughout:
- `cb_wait_front(cb_id_out0, 1)` — free-function CB FIFO poll (no wrapper in scope)
- `get_read_ptr(cb_id_out0)` — free-function CB pointer read
- `noc_async_write(l1_read_addr, dst_noc_addr, stick_size)` — raw NoC write
- `noc_async_write_barrier()` — raw NoC barrier
- Also uses `s0.get_noc_addr(i)` via `TensorAccessor` — the one Device-2.0 piece, but the surrounding infrastructure is pre-Device-2.0.

**Borrowed kernel files — file-path instantiation:**

| Borrowed kernel | Owner family | Broadly shared? |
|---|---|---|
| `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | Shared pool (`ttnn/cpp/ttnn/kernel/`) | Yes — also instantiated by `data_movement/concat`, `data_movement/slice/embeddings_rm_program_factory`, `data_movement/copy`, `data_movement/indexed_fill` (grep-verified) |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `eltwise/unary` | Broadly used by eltwise ops; shared rewrite required when eltwise/unary ports |
| `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize.cpp` | `data_movement/tilize` | Broadly used; shared rewrite when data_movement/tilize ports |

**Port-together sets induced:**
- The Device 2.0 migration of `writer_unary_stick_layout_interleaved_start_id.cpp` (a single-file fix) unblocks all five ops that borrow it — embedding, concat, slice, copy, indexed_fill. The Device 2.0 migration team should treat this as one unit.
- Once Metal 2.0 port proceeds: any Metal 2.0 rewrite of `writer_unary_interleaved_start_id.cpp` requires coordinating with all eltwise/unary borrowers; similarly for `tilize.cpp` and data_movement/tilize borrowers.

### Relaxation candidates

None (no custom `compute_program_hash` to mine).

### TTNN factory analysis

1. **Op-owned tensors?** No. `create_output_tensors` (`embedding_device_operation.cpp:129`) creates the output tensor via `create_device_tensor` — this is the standard output, not an intermediate. No factory allocates intermediate scratch tensors.
2. **MeshWorkload concept needed?** No. No `create_mesh_workload`, `create_workload_descriptor`, or `cached_mesh_workload_t` anywhere. No op-owned tensors, so no MeshWorkload artifact either.
3. **Pybind `create_descriptor`?** No. `embedding_nanobind.cpp` binds only the op function (`bind_function<"embedding">`) and exports `EmbeddingsType` enum. No `nb::class_<...ProgramFactory>` binding.
4. **Other migration-risky pybind?** None. The nanobind file contains only the user-facing function binding and enum export.
5. **Custom hash?** No. No `compute_program_hash` override in `embedding_device_operation.cpp` or `.hpp`.
6. **Custom override-runtime-args?** No. No `override_runtime_arguments` in any factory file.

## Misc anomalies *(non-gating)*

- `embeddings_rm_program_factory.cpp`: the `weight_stick_size` compile-time arg (CTA slot 4) is skipped — slots 3 and 5 are used but slot 4 is not extracted in `embeddings_rm_writer_chunked.cpp`. In `embeddings.cpp` slot 4 maps to `weight_stick_size` — this is correct but asymmetric; the writer kernel re-derives page info differently. Not a bug; just potentially confusing if both CTA layouts are read side-by-side.
- `embedding_ind_tilized.cpp:51`: `input_page_size` is obtained via `input.get_aligned_page_size()` at runtime, even though the page size is also available as a CTA (slot 3 `input_page_size` exists but is noted as unused by comparing the kernel's `get_compile_time_arg_val` calls — slots 0, 1, 2, 4, 5, 6 are used; slot 3 labeled `input_page_size` in `embeddings_tilized_indices_program_factory.cpp:139` is not extracted in the kernel). The kernel instead derives it at runtime from the accessor. Not a bug (the accessor value is correct), but the CTA slot is dead code.

## Questions for the user *(none)*

No open questions. The blocking is unambiguous.

## Recipe notes *(none)*

The audit recipe was clear and applicable throughout. No guidance conflicts or unanticipated cases.
