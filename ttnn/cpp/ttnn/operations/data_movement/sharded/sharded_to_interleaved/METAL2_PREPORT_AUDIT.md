# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/sharded/sharded_to_interleaved`

- **`ShardedToInterleavedDeviceOperation`**
  - `ShardedToInterleavedProgramFactory` (`device/sharded_to_interleaved_program_factory.cpp`)

Host-side files: `device/sharded_to_interleaved_device_operation.hpp`, `device/sharded_to_interleaved_device_operation_types.hpp`, `device/sharded_to_interleaved_device_operation.cpp`, `sharded_to_interleaved.hpp`, `sharded_to_interleaved.cpp`, `sharded_to_interleaved_nanobind.cpp`.

Kernels exercised by the factory (all DataflowBuffer kernels in scope for Device 2.0; compute kernel out of scope for Device 2.0 dataflow check):

| Role | Path | Owner |
|---|---|---|
| Reader | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` | eltwise/unary (cross-family donor) |
| Writer (tile layout) | `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp` | data_movement/sharded (in-family) |
| Writer (stick layout) | `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | data_movement/sharded (in-family) |
| Compute (optional; convert_df path only) | `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | shared kernel library (`ttnn/kernel/`) |

Unreferenced in this op's factory but present in the sibling directory `data_movement/sharded/device/kernels/`:
`eltwise_copy.cpp` (compute), `reader_unary_nd_sharded_blocks.cpp`, `reader_unary_sharded.cpp`, `reader_unary_sharded_blocks_interleaved_start_id.cpp`, `reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp`, `reshard_reader.cpp`, `reshard_reader_diff_width.cpp`, `reshard_same_height_reader.cpp`, `reshard_same_height_writer.cpp`, `reshard_same_width_reader.cpp`, `reshard_same_width_writer.cpp`, `writer_unary_sharded.cpp`, `writer_unary_sharded_blocks_start_id.cpp`, `writer_unary_sharded_stick_layout_start_id.cpp` — these belong to other ops in the sharded family and are out of scope for this audit.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/sharded/sharded_to_interleaved` |
| **Overall** | GREEN |
| **DOps / Factories** | `ShardedToInterleavedDeviceOperation` → `ShardedToInterleavedProgramFactory` |
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

---

## Result

**GREEN → brief issued.** All gates cleared. The op is on the `ProgramDescriptor` API, all dataflow kernels are Device 2.0 compliant, no UNSUPPORTED features are in use. Port work is limited to one Case-1 TensorAccessor binding (the output tensor, `dst_buffer`) across both writer kernels, plus the borrowed-memory DFB for the input CB. One incidental question is surfaced about the stick-layout writer's offset-base TensorAccessor pattern; it is tentatively Case 1 pending user confirmation.

---

## Gate detail

### ProgramDescriptor

GREEN — `ShardedToInterleavedProgramFactory::create_descriptor` returns a `ProgramDescriptor` populated with `CBDescriptor`, `KernelDescriptor` (reader, writer, optional compute), and emplace'd runtime args. No `host_api.hpp` imperative-builder calls (`CreateKernel`, `CreateCircularBuffer`, `SetRuntimeArgs`, etc.). File: `device/sharded_to_interleaved_program_factory.hpp:15`, `device/sharded_to_interleaved_program_factory.cpp:47`.

### Device 2.0 (every kernel used)

GREEN — all three dataflow kernels use Device 2.0 idioms throughout; the optional compute kernel is out of scope for the Device 2.0 dataflow check.

- **`reader_unary_sharded.cpp`** (eltwise/unary): includes `api/dataflow/circular_buffer.h`; uses `CircularBuffer cb(cb_id_in0); cb.push_back(num_tiles_per_core);`. No NoC, no legacy addr-gen. Fully Device 2.0.
- **`writer_unary_sharded_blocks_interleaved_start_id.cpp`** (data_movement/sharded): includes `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/tensor/noc_traits.h`; uses `Noc noc; CircularBuffer cb_out(cb_id_out); noc.async_write(cb_out, s, ...); noc.async_write_barrier(); cb_out.wait_front(...); cb_out.pop_front(...)`. `get_tile_size(cb_id_out)` at line 26 is a sanctioned Device 2.0 free function. Fully Device 2.0.
- **`writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp`** (data_movement/sharded): same includes and Noc/CircularBuffer wrapper usage. Fully Device 2.0.
- **`ttnn/kernel/compute/eltwise_copy.cpp`** (shared compute): uses `api/compute/common.h`, `api/compute/tile_move_copy.h`, `api/compute/eltwise_unary/eltwise_unary.h` plus `cb_wait_front`/`cb_pop_front`/`cb_push_back`/`cb_reserve_back` free functions with CB index. This is a **compute kernel** — the Device 2.0 migration guide covers data movement APIs only; compute-kernel CB free functions are a separate axis not covered by the Device 2.0 gate. Per the tilize/untilize precedent in existing audits, compute kernels are out of scope for the Device 2.0 dataflow check.

No legacy `InterleavedAddrGen`, `ShardedAddrGen`, `noc_async_read`/`noc_async_write` free functions, or raw semaphore addresses in any in-scope kernel.

### Feature compatibility

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | No `GlobalCircularBuffer`, no `global_circular_buffer` field set, no `CreateGlobalCircularBuffer` |
| Dynamic CircularBuffer (borrowed memory) | GREEN | `CBDescriptor::buffer = src_buffer` in `push_s2i_cb_pair`, called at factory line 147 with `bound_buffer = src_buffer`. LANDED; port uses `DataflowBufferSpec::borrowed_from` for the input CB |
| CBDescriptor `address_offset` (non-zero) | N/A | `address_offset` never set on any CBDescriptor in the factory |
| Aliased Circular Buffers | N/A | All `format_descriptors` initializers are single-element |
| GlobalSemaphore | N/A | No semaphores of any kind in the factory |
| Non-zero semaphore initial value | N/A | No `SemaphoreDescriptor` or `CreateSemaphore` calls |
| Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | `TensorAccessorArgs(*dst_buffer).append_to(...)` uses the single-argument form; no `ArgConfig::Runtime*` token anywhere |
| `UpdateCircularBuffer*` | N/A | No `UpdateCircularBufferTotalSize`, `UpdateCircularBufferPageSize`, or `UpdateDynamicCircularBufferAddressAndTotalSize` |
| Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t = ShardedToInterleavedInputs` has fixed tensors (no `std::vector<Tensor>`); all kernels read CTAs at fixed indices |

---

## Port-work summary *(mirrors the brief)*

**Tensor bindings:**

- `input_tensor` (src_buffer, CB `c_0`) — **clean** via borrowed-memory DFB. The sharded input CB is `CBDescriptor::buffer = src_buffer` (factory `push_s2i_cb_pair`, line 41–42); `reader_unary_sharded.cpp` calls `cb.push_back(num_tiles_per_core)` as the producer, and the writer kernel calls `cb_out.wait_front(...)` as the consumer. Real producer + consumer → genuine borrowed-memory DFB. Port: `DataflowBufferSpec::borrowed_from = input`.
- `output_tensor` (dst_buffer) — **Case 1** (re-express via `TensorParameter` / `TensorBinding`). The factory pushes `dst_buffer` (the `Buffer*` object itself, not `->address()`) as RTA arg 0 on every core; the framework auto-registers it as a `BufferBinding` and patches it on cache hits (correct-on-cache-hit today; not the silent-wrong hazard). Both writer kernels consume it via `get_arg_val<uint32_t>(0)` and pass it to `TensorAccessor(dst_args, dst_addr)` (tile layout) or `TensorAccessor(dst_args, dst_addr + input_width_offset_bytes)` (stick layout). Port: declare `TensorParameter output_tensor`, inject via `TensorBinding`. See question Q1 below about the stick-layout offset-base form.

**Custom hash:** none.

---

## Heads-ups *(mirrors the brief)*

### Notable LANDED constructs

- **Borrowed-memory DFB (Dynamic CircularBuffer):** `CBDescriptor::buffer = src_buffer` at `device/sharded_to_interleaved_program_factory.cpp:41` (inside `push_s2i_cb_pair`), instantiated at factory line 147. Port: `DataflowBufferSpec::borrowed_from = input`. The reader's `cb.push_back(num_tiles_per_core)` is the producer; writer's `cb_out.wait_front(...)` is the consumer — genuine DFB, not a fake CB.

### Fake CBs (address-only)

None — the input CB has a real producer (`reader_unary_sharded.cpp:16: cb.push_back(num_tiles_per_core)`) and a real consumer (both writer kernels call `cb_out.wait_front(...)`). Not a fake CB.

### Cross-op / shared kernels

**Port-together sets — the porter must coordinate these:**

1. **`reader_unary_sharded.cpp`** (cross-family borrow, owned by `eltwise/unary`): also used by `untilize_with_unpadding`, `untilize`, `tilize`, `transpose_wh_sharded`, `slice_write` (experimental), and others. Any Metal 2.0 rewrite of this kernel must land as a single change covering all co-borrowers. This kernel has no tensor address in it (pure sharded-CB reader — `cb.push_back` only; no NoC, no TensorAccessor), so its Metal 2.0 rewrite is trivial.

2. **`writer_unary_sharded_blocks_interleaved_start_id.cpp`** and **`writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp`** (in-family, owned by `data_movement/sharded`): also used by `sharded_to_interleaved_partial`. Rewrite of these two kernels must cover both `sharded_to_interleaved` and `sharded_to_interleaved_partial` in the same change.

3. **`ttnn/kernel/compute/eltwise_copy.cpp`** (shared kernel library, `ttnn/kernel/`): also used by `untilize_with_unpadding`, `sharded_to_interleaved_partial`, `copy` (copy_same_memory_config). Only active on the `convert_df=true` path. Rewrite must cover all co-borrowers.

### RTA varargs

None — all RT arg arrays are fixed-length at factory construction time.

### TTNN factory analysis (porter-relevant)

- **Pybind `create_descriptor`:** none — `sharded_to_interleaved_nanobind.cpp` binds only the op function itself via `bind_function<"sharded_to_interleaved">`.
- **Other risky pybind:** none.
- **Custom `override_runtime_arguments`:** none.

---

## Team-only

### TensorAccessor convertibility

N/A — only one Case-1 binding (`output_tensor`, `dst_buffer`); no Case-2 bindings.

The stick-layout writer's offset-base pattern (`TensorAccessor(dst_args, dst_addr + input_width_offset_bytes)`) is noted in Q1. If confirmed exotic, it would become Case 2 with the bridge: bind as `TensorParameter`, extract base via `get_bank_base_address`, apply the offset. This seems convertible (the offset is a simple per-core `curr_idx_w` byte offset into the output buffer), but the user should confirm. The tile-layout writer's `TensorAccessor(dst_args, dst_addr)` is straightforward Case 1.

### Out-of-directory coupling and donor shape

**Op-level roll-up:** ✓ clean — all cross-directory borrows are either LLK/HAL headers (no concern) or file-path borrowed kernels; no function-call escapes with problematic handle shapes.

**Summary table:**

| Op kernel | Donor file | Bucket |
|---|---|---|
| reader | `eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` | ✓ File-path borrow; cross-family donor (eltwise/unary) |
| writer (tile layout) | `data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp` | ✓ File-path borrow; in-family (data_movement/sharded) |
| writer (stick layout) | `data_movement/sharded/device/kernels/dataflow/writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | ✓ File-path borrow; in-family (data_movement/sharded) |
| compute (convert_df path) | `ttnn/kernel/compute/eltwise_copy.cpp` | ✓ Shared kernel library; compute kernel, dataflow-out-of-scope |

**Function-call escapes:** none — the kernels `#include` only `api/dataflow/` and `api/compute/` (LLK/HAL, class 1) and `api/tensor/noc_traits.h` (LLK). No cross-op helper functions called.

**Borrowed kernel files:**

- `reader_unary_sharded.cpp` — owned by `eltwise/unary`; broadly shared across `untilize_with_unpadding`, `untilize`, `tilize`, `transpose_wh_sharded`, `slice_write` (experimental), and potentially others. Port-together coupling is trivial for this kernel (no TensorAccessor, no tensor address RTAs — pure CB push).
- `writer_unary_sharded_blocks_interleaved_start_id.cpp` — owned by `data_movement/sharded`; also used by `sharded_to_interleaved_partial`. Port-together coupling for the two ops.
- `writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` — same ownership and co-borrower (`sharded_to_interleaved_partial`).
- `ttnn/kernel/compute/eltwise_copy.cpp` — owned by the shared kernel library (`ttnn/kernel/`); also used by `untilize_with_unpadding`, `sharded_to_interleaved_partial`, and `copy` (copy_same_memory_config). Port-together coupling for the convert_df code path.

### Relaxation candidates

No custom hash; no relaxation candidates to mine.

### TTNN factory analysis

1. **Op-owned tensors?** No. `create_output_tensors` returns the `preallocated_output` if provided, or calls `create_device_tensor(spec, device)` for the standard output. That is the declared `tensor_return_value_t` — not an intermediate scratch tensor the factory allocates for itself. No `create_device_tensor` calls inside `create_descriptor` (the factory itself allocates no tensors).

2. **MeshWorkload needed?** No. The op has no op-owned tensors (Q1 = No), so the MeshWorkload false-positive trap does not apply. The factory provides no `create_mesh_workload` / `create_workload_descriptor`, and the device-operation carries no `cached_mesh_workload_t`. Single-program op; standard program concept.

3. **Pybind `create_descriptor`?** No. `sharded_to_interleaved_nanobind.cpp:46` uses `bind_function<"sharded_to_interleaved">` — the normal op-function surface only. No `nb::class_<...ProgramFactory>` anywhere in the nanobind file.

4. **Other migration-risky pybind?** No. The nanobind file exposes nothing from `ShardedToInterleavedDeviceOperation` or `ShardedToInterleavedProgramFactory`. Only the wrapper function `sharded_to_interleaved_wrapper` is bound.

5. **Custom hash?** No. No `compute_program_hash` declaration in `ShardedToInterleavedDeviceOperation` or `ShardedToInterleavedProgramFactory`.

6. **Custom override-runtime-args?** No. No `override_runtime_arguments` in any factory.

---

## Misc anomalies *(team-only, non-gating)*

- **Dead RTA in stick-layout writer:** `device/sharded_to_interleaved_program_factory.cpp:294` pushes `num_units_per_row` as RTA arg index 1 (`writer_rt.push_back(num_units_per_row)`), but `writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` never reads arg 1 (reads args 0, 2, 3, 4, 5, 6 — skips index 1). The extra pushed value wastes one slot but is otherwise harmless. Routes to op owner; not porter-actionable.

---

## Questions for the user

1. **Stick-layout writer offset-base TensorAccessor:** `writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp:22` initializes `TensorAccessor(dst_args, dst_addr + input_width_offset_bytes)` where `input_width_offset_bytes = curr_idx_w` (a per-core byte offset within the output row, for width-sharded RM layout). The standard Metal 2.0 `TensorAccessor(ta::name)` construct binds to the buffer base; adding a constant per-core byte offset before construction is non-standard. Is this pattern expressible via standard `TensorParameter` / `TensorBinding` with `page_id` adjustment (Case 1 re-express), or does the width-within-stick byte-offset require the `get_bank_base_address` bridge (Case 2)? The audit tentatively classifies it as Case 1 (the offset appears to be a simple `curr_idx_w` scalar per core, not exotic address arithmetic), but user confirmation before porting is requested.

---

## Recipe notes

- The audit recipe (Check 2) says Device 2.0 applies to "every kernel the op uses" but does not explicitly carve out compute kernels. Existing audits (tilize, untilize_with_unpadding) treat compute kernels as out of scope for the Device 2.0 dataflow check, which is consistent with the migration guide's title ("Data Movement API Migration Guide"). The recipe's wording could benefit from an explicit "compute kernels are out of scope for this gate — the Device 2.0 check covers data movement / NoC / semaphore APIs only" to avoid auditor confusion.
