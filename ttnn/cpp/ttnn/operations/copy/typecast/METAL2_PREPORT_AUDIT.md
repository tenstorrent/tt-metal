# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/copy/typecast`

Single device operation in this directory. Four program factories sharing the same device-operation class.

- **`TypecastDeviceOperation`**
  - `TypecastProgramFactory` (`typecast_program_factory.cpp`) — interleaved/tiled and interleaved/row-major; instantiates donor kernels from `eltwise/unary/`
  - `TypecastSubgridProgramFactory` (`typecast_program_factory.cpp`) — same file; sub-core-grid variant; same donor kernels
  - `TypecastShardedProgramFactory` (`typecast_sharded_program_factory.cpp`) — L1-sharded, optimized path; borrowed-memory CBs; donor kernels from `eltwise/unary/`
  - `TypecastRowMajorChunkedProgramFactory` (`typecast_rm_chunked_program_factory.cpp`) — row-major chunked path; uses **op-owned** kernels `reader_typecast_rm_chunked.cpp` / `writer_typecast_rm_chunked.cpp`

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/copy/typecast` |
| **Overall** | RED |
| **DOps / Factories** | `TypecastDeviceOperation` → `TypecastProgramFactory`, `TypecastSubgridProgramFactory`, `TypecastShardedProgramFactory`, `TypecastRowMajorChunkedProgramFactory` |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | No |
| *Prereqs* — Cross-op escapes | Ok (donor kernels are Device 2.0 compliant; file-path coupling noted) |
| *Feature Support* — overall | GREEN (no UNSUPPORTED features; one LANDED feature — borrowed-memory DFB) |
| *Feature Support* — Variadic-CTA | N/A |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — MeshWorkload needed | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | Yes → delete (see Custom program hash) |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | Present: output CB in `TypecastShardedProgramFactory` — compute writes, no consumer kernel (workaround) |

## Result

**RED → blocked on Device 2.0 migration**, routed to the Device 2.0 team.

The op's own chunked kernels (`reader_typecast_rm_chunked.cpp`, `writer_typecast_rm_chunked.cpp`) are written using Device 1.0 idioms throughout — raw `noc_async_read`/`noc_async_write` free functions, `cb_reserve_back`/`cb_push_back`/`cb_wait_front`/`cb_pop_front`/`get_write_ptr`/`get_read_ptr` free functions, no `Noc` or `CircularBuffer` wrapper objects. These kernels are used exclusively by `TypecastRowMajorChunkedProgramFactory`. The remaining three factories use cross-family donor kernels (`reader_unary_interleaved_start_id.cpp`, `writer_unary_interleaved_start_id.cpp`, `reader_unary_sharded.cpp`) that are already Device 2.0 compliant.

**Subset opportunity:** `TypecastProgramFactory`, `TypecastSubgridProgramFactory`, and `TypecastShardedProgramFactory` are structurally unblocked — all kernels they use are Device 2.0 compliant. However, because all four factories share the same `TypecastDeviceOperation` class and the port is a unit, a subset-only port (excluding `TypecastRowMajorChunkedProgramFactory`) would require either splitting the device operation or leaving that factory on the old API — which may or may not be acceptable to the team. **Noted as a potential scope option, not a recommendation.**

The `ProgramDescriptor` prerequisite is cleared; the feature surface is clean. The only blocker is Device 2.0 migration of the two chunked kernels.

## Gate detail

- **ProgramDescriptor:** GREEN — all four factories populate `tt::tt_metal::ProgramDescriptor` with `KernelDescriptor`, `CBDescriptor`, and `emplace_runtime_args` / `runtime_args`. No `host_api.hpp` imperative-builder calls observed.

- **Device 2.0 (every kernel used):** RED (GATE).

  The op's own chunked dataflow kernels use Device 1.0 idioms exclusively — raw free-function NoC calls and CB management, no `Noc`/`CircularBuffer` wrapper objects:

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | `device/kernels/dataflow/reader_typecast_rm_chunked.cpp` | 29 | `get_write_ptr(cb_id_in)` | None |
  | `device/kernels/dataflow/reader_typecast_rm_chunked.cpp` | 28 | `cb_reserve_back(cb_id_in, onepage)` | None |
  | `device/kernels/dataflow/reader_typecast_rm_chunked.cpp` | 33 | `noc_async_read(chunk_noc_addr, l1_write_addr, full_chunk_size_bytes)` | None |
  | `device/kernels/dataflow/reader_typecast_rm_chunked.cpp` | 34 | `noc_async_read_barrier()` | None |
  | `device/kernels/dataflow/reader_typecast_rm_chunked.cpp` | 36 | `cb_push_back(cb_id_in, onepage)` | None |
  | `device/kernels/dataflow/reader_typecast_rm_chunked.cpp` | 44-46 | same pattern (partial chunk path) | None |
  | `device/kernels/dataflow/writer_typecast_rm_chunked.cpp` | 29 | `cb_wait_front(cb_id_out, onepage)` | None |
  | `device/kernels/dataflow/writer_typecast_rm_chunked.cpp` | 30 | `get_read_ptr(cb_id_out)` | None |
  | `device/kernels/dataflow/writer_typecast_rm_chunked.cpp` | 33 | `noc_async_write(l1_read_addr, chunk_noc_addr, full_chunk_size_bytes)` | None |
  | `device/kernels/dataflow/writer_typecast_rm_chunked.cpp` | 35 | `noc_async_writes_flushed()` | None |
  | `device/kernels/dataflow/writer_typecast_rm_chunked.cpp` | 36 | `cb_pop_front(cb_id_out, onepage)` | None |
  | `device/kernels/dataflow/writer_typecast_rm_chunked.cpp` | 43-49 | same pattern (partial chunk path) | None |
  | `device/kernels/dataflow/writer_typecast_rm_chunked.cpp` | 52 | `noc_async_write_barrier()` | None |

  These kernels include only `"api/dataflow/dataflow_api.h"` and do not include `"api/dataflow/noc.h"` or `"api/dataflow/circular_buffer.h"`. Every NoC and CB operation is the legacy free-function form — this is a wholesale Device 1.0 kernel, not an isolated holdover.

  **The port is blocked until `reader_typecast_rm_chunked.cpp` and `writer_typecast_rm_chunked.cpp` are migrated to Device 2.0** on the Device 2.0 track. The kernels are owned exclusively by this op.

  Donor kernels used by the other three factories — `reader_unary_interleaved_start_id.cpp`, `writer_unary_interleaved_start_id.cpp`, `reader_unary_sharded.cpp` (all in `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/`) — are Device 2.0 compliant. They use `Noc noc;`, `CircularBuffer cb(...)`, `noc.async_read(...)`, `noc.async_write(...)` wrapper forms.

- **Feature compatibility:** The following table covers all Appendix A entries.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `experimental::GlobalCircularBuffer` reference anywhere |
  | Dynamic CircularBuffer (borrowed memory) | GREEN | `TypecastShardedProgramFactory` sets `CBDescriptor::buffer` on both input and output CBs; input CB is a genuine DFB (producer + consumer), output CB is a fake CB addressed separately |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` field set in any factory |
  | Aliased Circular Buffers | N/A | All `CBDescriptor::format_descriptors` have exactly one `CBFormatDescriptor` element |
  | GlobalSemaphore | N/A | No semaphores of any kind in this op |
  | Non-zero semaphore initial value | N/A | No semaphores |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | All `TensorAccessorArgs(buffer)` calls use the single-argument form with no `Runtime*` flag |
  | `UpdateCircularBuffer*` | N/A | No `UpdateCircularBuffer*` calls in any factory or callback |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` (`TypecastInputs`) holds exactly two tensors (`input`, `preallocated_output`); no `std::vector<Tensor>`; all `get_compile_time_arg_val` calls use fixed `constexpr` indices |

## Port-work summary *(mirrors the brief)*

No brief is issued (RED). Recorded here for forward-looking planning.

- **Tensor bindings:**
  - `TypecastProgramFactory` — `input` (reader): `Buffer*`-binding form via `emplace_runtime_args(core, {src_buffer, ...})` — **Case 1** (re-express via `TensorParameter`/`TensorBinding`). `output` (writer): same form, **Case 1**.
  - `TypecastSubgridProgramFactory` — same as `TypecastProgramFactory`: both `input` and `output` are **Case 1**.
  - `TypecastRowMajorChunkedProgramFactory` — `input` (reader): `Buffer*`-binding form, **Case 1**. `output` (writer): same, **Case 1**. (Moot until Device 2.0 migration of the chunked kernels lands.)
  - `TypecastShardedProgramFactory` — input CB: `CBDescriptor::buffer = input.buffer()`, genuine borrowed-memory DFB (reader kernel produces via `cb.push_back`, compute kernel consumes via `cb_wait_front`/`cb_pop_front`) — **clean**. Output CB: `CBDescriptor::buffer = output.buffer()`, fake CB (compute kernel produces but no consumer kernel; see Fake CBs below) — **Case 1** with fake-CB workaround at port time.
- **Custom hash:** delete `TypecastDeviceOperation::compute_program_hash` at `device/typecast_device_op.cpp:156` → default (sanctioned exception).

## Heads-ups *(mirrors the brief)*

No brief is issued (RED). Recorded for forward-looking planning.

- **Notable LANDED constructs:**
  - `typecast_sharded_program_factory.cpp:86–95` — input `CBDescriptor::buffer = input.buffer()` (borrowed-memory DFB, LANDED). Port uses `DataflowBufferSpec::borrowed_from`. Input CB is a genuine DFB (producer + consumer present).
  - `typecast_sharded_program_factory.cpp:103–112` — output `CBDescriptor::buffer = output.buffer()` (borrowed-memory DFB field set, LANDED). However, see Fake CB entry below.

- **Fake CBs (address-only):** Output CB in `TypecastShardedProgramFactory` — `CBDescriptor::buffer = output.buffer()` at `typecast_sharded_program_factory.cpp:103`. The compute kernel (`eltwise_typecast.cpp`) produces into it (`cb_reserve_back`/`pack_tile`/`cb_push_back`) but no kernel consumes (no `cb_wait_front`/`cb_pop_front` on this CB index). Litmus fails: producer present, no consumer. This is a fake CB used as an address-only output target for the sharded write. The port resolves it with the fake-CB workaround (see the porting recipe); **it does not gate**. Record as **(output_cb, compute endpoint)**.

- **Cross-op / shared kernels:** `TypecastProgramFactory`, `TypecastSubgridProgramFactory`, and `TypecastShardedProgramFactory` instantiate kernels from `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/`. These kernels are broadly shared — approximately 50 other ops in `ttnn/cpp/ttnn/operations/` also instantiate them. A Metal 2.0 rewrite of those kernel files is a **port-together set** rewrite: every op that instantiates them must adopt it in a single coordinated change, or co-borrowers break. See Out-of-directory coupling section for detail.

- **RTA varargs:** None observed.
- **TTNN factory analysis (porter-relevant):** No `create_descriptor` pybind, no other risky pybind, no custom `override_runtime_arguments`. Custom hash: YES — delete at `device/typecast_device_op.cpp:156` (PORT WORK).

## Team-only

### TensorAccessor convertibility (per Case-2 binding)

No Case-2 bindings identified. All tensor bindings are Case 1 (routine re-express). The RM chunked kernels use `TensorAccessor` properly on the kernel side (`TensorAccessorArgs<5>()`, `TensorAccessor(src_args, src_addr)`, `s.get_noc_addr(row_id, byte_offset)`) — the pattern is page-accessible with byte-offset support, clearly Case 1 once the Device 2.0 prerequisite clears.

### Out-of-directory coupling & donor shape

**Op-level roll-up:** ✓ clean for function-call escapes (no cross-op `#include` calls in any kernel; all `#include`s are `api/...` LLK framework paths). File-path coupling is present and broadly shared.

**Summary table — kernel instantiation (file-path coupling):**

| Op kernel | Donor file | Donor pool | Scope |
|---|---|---|---|
| `TypecastProgramFactory` reader | `eltwise/unary/.../reader_unary_interleaved_start_id.cpp` | Cross-family (eltwise/unary) | ~50 co-borrowers |
| `TypecastProgramFactory` writer | `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` | Cross-family (eltwise/unary) | ~50 co-borrowers |
| `TypecastSubgridProgramFactory` reader | `eltwise/unary/.../reader_unary_interleaved_start_id.cpp` | Cross-family (eltwise/unary) | ~50 co-borrowers |
| `TypecastSubgridProgramFactory` writer | `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` | Cross-family (eltwise/unary) | ~50 co-borrowers |
| `TypecastShardedProgramFactory` reader | `eltwise/unary/.../reader_unary_sharded.cpp` | Cross-family (eltwise/unary) | ~50 co-borrowers |
| `TypecastRowMajorChunkedProgramFactory` reader | `typecast/device/kernels/dataflow/reader_typecast_rm_chunked.cpp` | Op-owned | This op only |
| `TypecastRowMajorChunkedProgramFactory` writer | `typecast/device/kernels/dataflow/writer_typecast_rm_chunked.cpp` | Op-owned | This op only |

**Per-call detail:** All function-call escapes are N/A (no function-call coupling). The donor files export no callable functions consumed by these kernels; they are instantiated as complete `kernel_main()` translations.

**Device 2.0 status of donors:**
- `reader_unary_interleaved_start_id.cpp`: Device 2.0 compliant (`Noc noc;`, `CircularBuffer cb(cb_id_in0);`, `noc.async_read(s, cb, page_bytes, ...)` — all wrapper-based).
- `writer_unary_interleaved_start_id.cpp`: Device 2.0 compliant (`Noc noc;`, `CircularBuffer cb(...)`, `noc.async_write(...)` — all wrapper-based).
- `reader_unary_sharded.cpp`: Device 2.0 compliant (`CircularBuffer cb(cb_id_in0);`, `cb.push_back(...)` — wrapper-based).

**Port-together coupling:** The three donor kernels are instantiated by ~50 ops. A Metal 2.0 port of any one op that changes these files (CB → DFB naming, etc.) requires all ~50 co-borrowers to adopt the change in the same diff. The shared-kernel rewrite should be planned as a coordinated family-level action, not a per-op action.

**Op-owned kernels (`reader_typecast_rm_chunked.cpp`, `writer_typecast_rm_chunked.cpp`):** Unique to this op. No port-together coupling for these; their Device 2.0 migration and subsequent Metal 2.0 rewrite are contained within this op.

### Relaxation candidates (mined from custom hash — FALLIBLE)

The custom `compute_program_hash` at `typecast_device_op.cpp:156–183` hashes:
- For TILE layout: `program_factory.index()`, `input_dtype`, `input.memory_config()`, `input_shape.volume()`, `input.layout()`, plus op args (which include `output_dtype`, `output_memory_config`, `fp32_dest_acc_en`, etc.).
- For ROW_MAJOR layout: same but with `input_shape` (full shape) instead of `input_shape.volume()`.

Relaxation candidate: **for TILE layout only** — the hash uses volume rather than per-dimension shape, suggesting the op is shape-agnostic in tile layout (number of tiles matters, not their arrangement). The `dynamic_tensor_shape` relaxation on the input `TensorParameter` *may* be safe for tile layout if `TensorAccessor` iteration is purely page-id based (which it is via `reader_unary_interleaved_start_id.cpp`). **FALLIBLE — verify** before applying: the hash was written independently and may omit relevant attributes.

### TTNN factory analysis

1. **Op-owned tensors:** No. `create_output_tensors` at `typecast_device_op.cpp:149` calls `create_device_tensor(...)` for the standard op output — this is the output `tensor_return_value_t`, not a factory-managed intermediate. `TypecastInputs::preallocated_output` is a caller-provided optional; it is not factory-allocated. No intermediate scratch tensors found in any factory.

2. **MeshWorkload concept needed:** No. No `create_mesh_workload`, `create_workload_descriptor`, or `cached_mesh_workload_t` in any device-op file. Single-program op.

3. **Pybind `create_descriptor`:** No. The nanobind registration in `ttnn/cpp/ttnn-nanobind/operations/copy.cpp` binds only the top-level `ttnn::typecast` function via `bind_function<"typecast">`. No `nb::class_<...ProgramFactory>(...).def_static("create_descriptor", ...)` binding present.

4. **Other migration-risky pybind:** None. No `DeviceOperation` or factory/param class exposed to Python. Standard op-function binding only.

5. **Custom hash:** Yes — `ttsl::hash::hash_t TypecastDeviceOperation::compute_program_hash(...)` at `device/typecast_device_op.cpp:156–184`. PORT WORK: delete and revert to default (sanctioned exception). Cross-reference the Custom program hash subject.

6. **Custom override-runtime-args:** No. No `override_runtime_arguments` static method in any factory file.

## Misc anomalies *(team-only, non-gating)*

- `typecast_device_op.cpp:161`: `auto program_factory = select_program_factory(args, tensor_args);` is called inside `compute_program_hash` — the factory selection is re-run purely to get `program_factory.index()` for the hash key, which redundantly re-evaluates all the factory selection predicates. This is a latent inefficiency (the hash is computed at cache-miss time, so it runs when the factory is already being selected anyway). Not a correctness issue; the custom hash deletion at port time resolves it automatically.

- `typecast_device_op.cpp:168–172` (TILE branch of custom hash): `input_tensor.dtype()` is hashed but `input_tensor.dtype()` should equal `args.input_dtype` per the contract (the op params capture dtype at call time). If they can diverge (e.g., preallocated_output path), this is a subtle discrepancy. Non-actionable at port time since the hash is being deleted.

## Questions for the user

1. **Subset port scope:** `TypecastProgramFactory`, `TypecastSubgridProgramFactory`, and `TypecastShardedProgramFactory` are structurally unblocked (all their kernels are Device 2.0 compliant). Only `TypecastRowMajorChunkedProgramFactory` is blocked on Device 2.0. Is splitting the port — porting the three unblocked factories first and adding the chunked factory once its kernels are migrated — acceptable? Or should the port wait until all four factories are unblocked and proceed as a unit? This determines how the Device 2.0 work item for the chunked kernels should be sequenced relative to the Metal 2.0 port.

## Recipe notes

- The audit recipe's instructions for the "Fake CB" litmus are clear, but applying them to the borrowed-memory sharded output CB required careful reading: the compute kernel is both the LLK-level "producer" (it writes tile data into the output CB) and there is genuinely no consumer kernel. The recipe says "does the CB have a producer and a consumer?" — producer exists (compute kernel pushes), consumer does not (nobody pops). Marking it fake was the correct call, but the recipe could benefit from an explicit note covering the "compute kernel writes to borrowed-memory output CB in a reader-only sharded factory" pattern, since it will recur for all sharded ops with no writer.
- The recipe says "Buffer*-binding form (framework-patched) → still enumerate it as Case 1". The `emplace_runtime_args(core, {buffer_ptr, ...})` pattern matches exactly; recorded as Case 1. No ambiguity here, but worth noting for completeness.
