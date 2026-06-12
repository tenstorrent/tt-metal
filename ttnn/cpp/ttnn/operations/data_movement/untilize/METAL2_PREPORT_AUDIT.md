# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/untilize`

One `DeviceOperation` with eight program factories:

- **`UntilizeDeviceOperation`** (`ttnn::prim::UntilizeDeviceOperation`)
  - `UntilizeSingleCoreProgramFactory` (`device/factories/untilize_single_core_program_factory.cpp`)
  - `UntilizeMultiCoreSubCoreGridsProgramFactory` (`device/factories/untilize_multi_core_sub_core_grids_program_factory.cpp`)
  - `UntilizeMultiCoreBlockProgramFactory` (`device/factories/untilize_multi_core_block_program_factory.cpp`)
  - `UntilizeMultiCoreInputAndOutputShardTypeAndShardSpecIdenticalProgramFactory` (`device/factories/untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical_program_factory.cpp`)
  - `UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory` (`device/factories/untilize_multi_core_input_and_output_nd_shard_type_and_shard_spec_identical_program_factory.cpp`)
  - `UntilizeMultiCoreParallelizeColumnProgramFactory` (`device/factories/untilize_multi_core_parallelize_column_program_factory.cpp`)
  - `UntilizeMultiCoreProgramFactory` (`device/factories/untilize_multi_core_program_factory.cpp`)
  - `UntilizeMultiCoreNDShardInputProgramFactory` (`device/factories/untilize_multi_core_nd_shard_input_program_factory.cpp`)

Kernel files (own):
- `device/kernels/dataflow/reader_unary_start_id.cpp`
- `device/kernels/dataflow/reader_unary_sharded_blocks.cpp`
- `device/kernels/dataflow/writer_unary_stick_layout_split_rows_single_core.cpp`
- `device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`
- `device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core_nd_shard.cpp`
- `device/kernels/dataflow/writer_unary_stick_layout_split_rows_interleaved_parallel_columns.cpp`
- `device/kernels/compute/untilize.cpp`
- `device/kernels/compute/untilize_variable_num_blocks.cpp`
- `device/kernels/compute/untilize_w.cpp` (referenced by `UntilizeMultiCoreParallelizeColumnProgramFactory` — see Misc anomalies)
- `device/kernels/compute/untilize_wh.cpp`

Borrowed kernel files (file-path kernel instantiation, in-scope for Device 2.0 gate):
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` (eltwise/unary family)
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` (eltwise/unary family)
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_wh_multicore.cpp` (eltwise/unary family)
- `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` (data_movement/sharded family)
- `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reader_unary_nd_sharded_blocks.cpp` (data_movement/sharded family)
- `ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/writer_unary_stick_layout_wh_multicore.cpp` (data_movement/untilize_with_unpadding family)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/untilize` |
| **Overall** | GREEN |
| **DOps / Factories** | `UntilizeDeviceOperation` → 8 factories (see above) |
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

**GREEN → brief issued.** All gates cleared. The op is ready for the Metal 2.0 port (pending explicit user go-ahead). The port's main construction work is the tensor-binding re-expression (Case 1) for the six interleaved/DRAM-path factories; the two identical-shard-spec factories use borrowed-memory DFBs (already LANDED) and contribute no binding case work.

## Gate detail

- **ProgramDescriptor:** GREEN — all eight factories return a `tt::tt_metal::ProgramDescriptor` from `create_descriptor`. Every factory includes `<tt-metalium/program_descriptors.hpp>` and populates `ProgramDescriptor`, `CBDescriptor`, `KernelDescriptor` structs. No imperative `host_api.hpp` builder calls (`CreateProgram` / `CreateKernel` / `CreateCircularBuffer` / `SetRuntimeArgs`) are present.

- **Device 2.0 (every kernel used):** GREEN — every kernel file (own and borrowed) uses the Device 2.0 API surface consistently:
  - Dataflow kernels: `Noc noc`, `CircularBuffer cb(cb_id)`, `TensorAccessor`, `noc.async_read(...)` / `noc.async_write(...)` with typed endpoints.
  - Free functions present: `get_tile_size(cb_id)` (sanctioned per Device 2.0 migration guide), `get_local_cb_interface(cb_id_in0)` in `reader_unary_interleaved_start_id.cpp:20` (sanctioned). No unsanctioned CB-index free functions observed.
  - Compute kernels: `compute_kernel_lib::untilize<...>` from `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` — Device 2.0 kernel lib, parameters named `input_dfb` / `output_dfb`.
  - Legacy Device 1.0 patterns absent: no `InterleavedAddrGen` / `ShardedAddrGen` usage, no raw `noc_async_read` (only `noc.async_read` member form), no manual CB index operations. The `sharding_addrgen.hpp` is `#include`d as a utility header but its legacy `ShardedAddrGen` / `get_noc_addr` APIs are not called by any untilize kernel; the kernels use only `TensorAccessor::shard_pages()`.

- **Feature compatibility:** No UNSUPPORTED features found. Full feature table:

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | Not used |
  | Dynamic CircularBuffer (borrowed memory) | GREEN | `CBDescriptor::buffer` set to `src0_buffer` / `dst_buffer` in `UntilizeMultiCoreProgramFactory` (even-sharding path), `UntilizeMultiCoreInputAndOutputShardTypeAndShardSpecIdenticalProgramFactory`, and `UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory`. Port uses `borrowed_from` on the affected `DataflowBufferSpec`s. All have genuine producer + consumer (sharded reader pushes, compute waits). |
  | CBDescriptor `address_offset` (non-zero) | N/A | `.address_offset` not set in any factory |
  | Aliased Circular Buffers | N/A | All `format_descriptors` initializers are single-element |
  | GlobalSemaphore | N/A | No semaphores present in this op |
  | Non-zero semaphore initial value | N/A | No semaphores |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | No `ArgConfig::Runtime*` tokens in any factory |
  | `UpdateCircularBuffer*` | N/A | Not used |
  | Variable-count compile-time arguments (CTA varargs) | N/A | Fixed input count (1 tensor); no variable-count CTA loops |

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding):

  | Factory | Binding | Classification | Notes |
  |---|---|---|---|
  | `UntilizeSingleCoreProgramFactory` | `src0_buffer` (input) | Case 1 — re-express | CTA-baked `TensorAccessorArgs`; RTA: `Buffer*` |
  | `UntilizeSingleCoreProgramFactory` | `dst_buffer` (output) | Case 1 — re-express | CTA-baked `TensorAccessorArgs`; RTA: `Buffer*` |
  | `UntilizeMultiCoreSubCoreGridsProgramFactory` | `src0_buffer` (input) | Case 1 — re-express | CTA-baked `TensorAccessorArgs`; RTA: `Buffer*` |
  | `UntilizeMultiCoreSubCoreGridsProgramFactory` | `dst_buffer` (output) | Case 1 — re-express | CTA-baked `TensorAccessorArgs`; RTA: `Buffer*` |
  | `UntilizeMultiCoreBlockProgramFactory` | `src0_buffer` (input) | Case 1 — re-express | CTA-baked `TensorAccessorArgs`; RTA: `Buffer*` |
  | `UntilizeMultiCoreBlockProgramFactory` | `dst_buffer` (output) | Case 1 — re-express | CTA-baked `TensorAccessorArgs`; RTA: `Buffer*` |
  | `UntilizeMultiCoreInputAndOutputShardTypeAndShardSpecIdenticalProgramFactory` | `src0_buffer` (input CB) | clean — borrowed-memory DFB | `CBDescriptor::buffer = src0_buffer`; genuine producer (sharded reader) + consumer (compute) |
  | `UntilizeMultiCoreInputAndOutputShardTypeAndShardSpecIdenticalProgramFactory` | `dst_buffer` (output CB) | clean — borrowed-memory DFB | `CBDescriptor::buffer = dst_buffer`; producer (compute) + consumer (sharded writer) |
  | `UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory` | `src0_buffer` (input CB) | clean — borrowed-memory DFB | Same pattern as above; ND-shard variant |
  | `UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory` | `dst_buffer` (output CB) | clean — borrowed-memory DFB | Same pattern |
  | `UntilizeMultiCoreParallelizeColumnProgramFactory` | `src0_buffer` (input) | Case 1 — re-express | CTA-baked `TensorAccessorArgs`; RTA: `Buffer*` |
  | `UntilizeMultiCoreParallelizeColumnProgramFactory` | `dst_buffer` (output) | Case 1 — re-express | CTA-baked `TensorAccessorArgs`; RTA: `Buffer*` |
  | `UntilizeMultiCoreProgramFactory` | `src0_buffer` (input) | Case 1 (interleaved path) / clean (even-sharding borrowed-memory DFB path) | Interleaved path: CTA-baked `TensorAccessorArgs`; RTA: `Buffer*`. Even-sharding path: `CBDescriptor::buffer = src0_buffer` — clean borrowed-memory DFB. Block-reader sharding path: CTA-baked `TensorAccessorArgs`; RTA: `Buffer*`. |
  | `UntilizeMultiCoreProgramFactory` | `dst_buffer` (output) | Case 1 — re-express | CTA-baked `TensorAccessorArgs`; RTA: `Buffer*` in all paths |
  | `UntilizeMultiCoreNDShardInputProgramFactory` | `src0_buffer` (input) | Case 1 — re-express | CTA-baked `TensorAccessorArgs`; RTA: `Buffer*` (reader + writer) |
  | `UntilizeMultiCoreNDShardInputProgramFactory` | `dst_buffer` (output) | Case 1 — re-express | CTA-baked `TensorAccessorArgs`; RTA: `Buffer*` |

- **Custom hash:** None — no `compute_program_hash` override in `UntilizeDeviceOperation`. No port action needed.

## Heads-ups  *(mirrors the brief)*

- **Notable LANDED constructs:** Dynamic CircularBuffer (borrowed memory) — `CBDescriptor::buffer` set in three factories:
  - `UntilizeMultiCoreProgramFactory` even-sharding path: `device/factories/untilize_multi_core_program_factory.cpp:118–129` — input CB only
  - `UntilizeMultiCoreInputAndOutputShardTypeAndShardSpecIdenticalProgramFactory`: `device/factories/untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical_program_factory.cpp:54–79` — both input and output CBs
  - `UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory`: `device/factories/untilize_multi_core_input_and_output_nd_shard_type_and_shard_spec_identical_program_factory.cpp:68–97` — both input and output CBs

  Port uses `DataflowBufferSpec::borrowed_from` naming the appropriate `TensorParameter`. These CBs have genuine producer + consumer pairs.

- **Fake CBs (address-only):** None.

- **Cross-op / shared kernels:** Six borrowed kernel files instantiated by the factories — see borrowed kernel list in the Identifying section. These files are shared with other ops; the Metal 2.0 rewrite of each borrowed kernel is a single rewrite that all borrowing ops must adopt together:
  - `reader_unary_sharded.cpp` (eltwise/unary) — also used by at least: `UntilizeMultiCoreInputAndOutputShardTypeAndShardSpecIdenticalProgramFactory`, `UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory`, `UntilizeMultiCoreProgramFactory` (even-sharding path). Other ops in the data_movement and eltwise families use this too; the porter should coordinate a shared rewrite.
  - `reader_unary_interleaved_start_id.cpp` (eltwise/unary) — used by `UntilizeMultiCoreSubCoreGridsProgramFactory` and `UntilizeMultiCoreParallelizeColumnProgramFactory`. Broadly shared.
  - `reader_unary_interleaved_wh_multicore.cpp` (eltwise/unary) — used by `UntilizeMultiCoreBlockProgramFactory`. Shared with other data_movement ops.
  - `writer_unary_sharded.cpp` (data_movement/sharded) — used by identical-shard-spec factories.
  - `reader_unary_nd_sharded_blocks.cpp` (data_movement/sharded) — used by `UntilizeMultiCoreNDShardInputProgramFactory`.
  - `writer_unary_stick_layout_wh_multicore.cpp` (data_movement/untilize_with_unpadding) — used by `UntilizeMultiCoreBlockProgramFactory`.

- **RTA varargs:** None.

- **TTNN factory analysis (porter-relevant):**
  - Pybind `create_descriptor`: No — `untilize_nanobind.cpp` uses only `ttnn::bind_function<"untilize">` (standard op-function surface).
  - Other risky pybind: None.
  - Custom `override_runtime_arguments`: None.

## Team-only

### TensorAccessor convertibility (Case-2 candidates)

None — all non-clean bindings are Case 1. The access patterns are straightforward page-by-page or shard-page iteration via `TensorAccessor`. No exotic address arithmetic or sub-page walk patterns observed.

### Out-of-directory coupling & donor shape

**Op-level roll-up:** ⚠ workable — all borrowed kernels are Device 2.0 compliant; the coupling creates port-together constraints for each shared file but does not block the port.

**Summary table:**

| Op kernel (factory) | Borrowed file | Donor family | D2.0 status | Shape notes |
|---|---|---|---|---|
| All even-sharding / identical-shard-spec | `eltwise/unary/.../reader_unary_sharded.cpp` | eltwise/unary | ✓ D2.0 | CB-only (no tensor access); `CircularBuffer cb(cb_id)` — `uint32_t cb_id` shape |
| SubCoreGrids, ParallelizeColumn | `eltwise/unary/.../reader_unary_interleaved_start_id.cpp` | eltwise/unary | ✓ D2.0 | `TensorAccessor`; `get_local_cb_interface(cb_id_in0)` (sanctioned) |
| Block | `eltwise/unary/.../reader_unary_interleaved_wh_multicore.cpp` | eltwise/unary | ✓ D2.0 | `TensorAccessor` with CTA offset |
| Identical-shard factories, NDshard | `data_movement/sharded/.../writer_unary_sharded.cpp` | data_movement/sharded | ✓ D2.0 | CB-only; `CircularBuffer cb_out(cb_id)` — `uint32_t cb_id` shape |
| NDShardInput | `data_movement/sharded/.../reader_unary_nd_sharded_blocks.cpp` | data_movement/sharded | ✓ D2.0 | `TensorAccessor::shard_pages()`; `get_tile_size(cb_id)` (sanctioned) |
| Block | `data_movement/untilize_with_unpadding/.../writer_unary_stick_layout_wh_multicore.cpp` | data_movement/untilize_with_unpadding | ✓ D2.0 | `TensorAccessor`, `Noc`, `CircularBuffer`; member-form `cb_out0.get_write_ptr()` |

**Per-call detail — borrowed files:**

All borrowed files are already Device 2.0 compliant. No ✗ / ⭐ shape entries. Key observations:
- `reader_unary_sharded.cpp`: trivial — only calls `cb.push_back(num_tiles)` with `uint32_t cb_id` shape. The `uint32_t cb_id` → `dfb::name`'s constexpr cast handles this.
- `writer_unary_sharded.cpp`: trivial — only calls `cb_out.wait_front(num_units)` with `uint32_t cb_id` shape. Same handling.
- `reader_unary_interleaved_start_id.cpp`: uses `TensorAccessorArgs<0>` (Shape 1 / `TensorAccessor`) — excellent; porter constructs `TensorAccessor(ta::name)`.
- `reader_unary_interleaved_wh_multicore.cpp`: uses `TensorAccessorArgs<3>()` (Shape 1) — excellent.
- `reader_unary_nd_sharded_blocks.cpp`: uses `TensorAccessorArgs<4>()` (Shape 1) — excellent.
- `writer_unary_stick_layout_wh_multicore.cpp`: uses `TensorAccessorArgs<4>()` (Shape 1) — excellent. The `cb_out0.get_write_ptr()` is the Device 2.0 member form, not a legacy holdover.

**Port-together coupling:** The six borrowed files form coupling constraints. Most are broadly shared across the data_movement and eltwise families — coordinating their Metal 2.0 rewrite as shared-file units will be the main planning consideration for this port.

### Relaxation candidates

Not applicable — no custom `compute_program_hash` to mine.

### TTNN factory analysis

1. **Op-owned tensors?** No. `create_output_tensors` calls `create_device_tensor(compute_output_specs(...), ...)` — standard output allocation, not an intermediate. No factory allocates its own intermediate/scratch tensors (`device/untilize_device_operation.cpp:276`).

2. **MeshWorkload needed?** No. Single-program op. No `create_mesh_workload` / `cached_mesh_workload_t`.

3. **Pybind `create_descriptor`?** No. `untilize_nanobind.cpp` uses only `ttnn::bind_function<"untilize">` — the normal op-function surface, not factory internals.

4. **Other migration-risky pybind?** None. The nanobind file does not expose `UntilizeDeviceOperation`, any factory class, or any `ProgramDescriptor`-returning entry point.

5. **Custom hash?** No `compute_program_hash` override in `UntilizeDeviceOperation` or any factory.

6. **Custom override-runtime-args?** No `override_runtime_arguments` defined anywhere in the op directory.

## Misc anomalies  *(team-only, non-gating)*

- `device/kernels/compute/untilize_w.cpp` is present in the op's kernel directory but is not referenced by any factory `KernelDescriptor::kernel_source` in the eight current factories. It appears to be dead code (or residual from an earlier factory). Its contents are Device 2.0 compliant. If the op was migrated away from the factory that used it, consider removing the file to avoid confusion for future readers.
