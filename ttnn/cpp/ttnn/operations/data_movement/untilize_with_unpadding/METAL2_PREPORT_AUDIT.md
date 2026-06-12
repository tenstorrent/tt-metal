# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding`

Single DeviceOperation in this directory:

- **`UntilizeWithUnpaddingDeviceOperation`**
  - `UntilizeWithUnpaddingSingleCoreProgramFactory` (`factories/untilize_with_unpadding_single_core_program_factory.cpp`)
  - `UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory` (`factories/untilize_with_unpadding_multi_core_interleaved_program_factory.cpp`)
  - `UntilizeWithUnpaddingMultiCoreShardedProgramFactory` (`factories/untilize_with_unpadding_multi_core_sharded_program_factory.cpp`)
  - `UntilizeWithUnpaddingMultiCoreColInterleavedProgramFactory` (`factories/untilize_with_unpadding_multi_core_col_interleaved_program_factory.cpp`)
  - `UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory` (`factories/untilize_with_unpadding_multi_core_block_interleaved_program_factory.cpp`)
  - `UntilizeWithUnpaddingMultiCoreNDShardedProgramFactory` (`factories/untilize_with_unpadding_multi_core_nd_sharded_program_factory.cpp`)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding` |
| **Overall** | RED |
| **DOps / Factories** | `UntilizeWithUnpaddingDeviceOperation` → 6 factories (see above) |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | No |
| *Prereqs* — Cross-op escapes | issue: two shared-pool kernels on Device 1.0 (detail below) |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | N/A |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — MeshWorkload needed | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | None |

## Result

**RED → blocked on Device 2.0 prerequisite**, routed to the Device 2.0 migration team.

Two shared-pool kernels exercised by `UntilizeWithUnpaddingMultiCoreShardedProgramFactory` are on Device 1.0 idioms:
- `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` (used in the non-out_sharded path) — legacy `cb_wait_front`/`get_read_ptr`/`noc_async_write`/`noc_async_write_barrier`/`cb_pop_front` free functions.
- `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` (used in the `unpad_tensor_w_16` path) — legacy `cb_wait_front`/`cb_reserve_back`/`cb_pop_front`/`cb_push_back` free functions.

Both are shared-pool kernels (`ttnn/cpp/ttnn/kernel/`) used by other ops. Their Device 2.0 migration is a shared-pool effort, not scoped to this op alone.

**Scoped subset:** The five non-sharded factories — `SingleCore`, `MultiCoreInterleaved`, `MultiCoreColInterleaved`, `MultiCoreBlockInterleaved`, `MultiCoreNDSharded` — are clear of this specific block. The `MultiCoreShardedProgramFactory` is the sole factory that exercises the two blocked kernels. A partial port of the five non-sharded factories is structurally possible if the shared-pool migration is staged, but the sharded factory will need to wait.

## Gate detail

- **ProgramDescriptor:** GREEN — all six factories populate a `ProgramDescriptor` and use `KernelDescriptor`, `CBDescriptor`, `SemaphoreDescriptor`-style constructs exclusively. None use the imperative `host_api.hpp` builder API (`CreateProgram`/`CreateKernel`/`SetRuntimeArgs`/etc.).

- **Device 2.0 (every kernel used):** RED — two shared-pool kernels used by `UntilizeWithUnpaddingMultiCoreShardedProgramFactory` are on Device 1.0 idioms. All other kernels (the op's own writer kernels, the eltwise/unary donor reader kernels, and the untilize compute library kernels) are Device 2.0 compliant.

  **Blocking violations:**

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` | 24 | `cb_wait_front(cb_id_out0, block_width_ntiles)` | No `CircularBuffer` wrapper constructed |
  | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` | 25 | `get_read_ptr(cb_id_out0)` | No `CircularBuffer` wrapper constructed |
  | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` | 31 | `noc_async_write(l1_read_addr, dst_noc_addr, ...)` | No `Noc` wrapper constructed |
  | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` | 35 | `noc_async_write_barrier()` | No `Noc` wrapper constructed |
  | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` | 36 | `cb_pop_front(cb_id_out0, block_width_ntiles)` | No `CircularBuffer` wrapper constructed |
  | `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | 20 | `cb_wait_front(tt::CBIndex::c_0, 1)` | No `CircularBuffer` wrapper constructed |
  | `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | 21 | `cb_reserve_back(tt::CBIndex::c_16, 1)` | No `CircularBuffer` wrapper constructed |
  | `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | 28 | `cb_pop_front(tt::CBIndex::c_0, 1)` | No `CircularBuffer` wrapper constructed |
  | `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | 29 | `cb_push_back(tt::CBIndex::c_16, 1)` | No `CircularBuffer` wrapper constructed |

  Note: a Device 2.0 compliant variant of `eltwise_copy.cpp` exists at `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/compute/eltwise_copy.cpp` (uses `CircularBuffer` wrapper methods) — the sharded factory should be redirected to that path once the dataflow writer is also migrated.

  **Note on `get_tile_size` in `reader_unary_interleaved_start_id.cpp` and `reader_unary_interleaved_col_multicore.cpp`:** `get_tile_size(cb_id)` is a sanctioned Device 2.0 free function per the migration guide — not flagged.

  **Note on `get_local_cb_interface(cb_id_in0)` in `reader_unary_interleaved_start_id.cpp` (line 20):** Also sanctioned per the Device 2.0 migration guide — not flagged.

- **Feature compatibility:** every Appendix A entry scanned below.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `experimental::GlobalCircularBuffer` references in any factory or kernel |
  | Dynamic CircularBuffer (borrowed memory) | GREEN | `MultiCoreShardedProgramFactory`: `CBDescriptor::buffer = a.buffer()` (line 97) and `CBDescriptor::buffer = output.buffer()` (line 127). Port uses `DataflowBufferSpec::borrowed_from`. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` set to non-zero anywhere in the op |
  | Aliased Circular Buffers | N/A | All `format_descriptors` initializers are single-element |
  | GlobalSemaphore | N/A | No semaphores used in any factory |
  | Non-zero semaphore initial value | N/A | No semaphores used |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | No `ArgConfig::Runtime*` references anywhere in host code |
  | `UpdateCircularBuffer*` | N/A | No calls to `UpdateCircularBufferTotalSize`/`UpdateCircularBufferPageSize`/`UpdateDynamicCircularBufferAddressAndTotalSize` |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t = Tensor` (single fixed input); no CTA loop over runtime-varying index in any kernel |

## Port-work summary *(mirrors the brief)*

No brief issued (RED). Forward-looking port-work inventory for use after the Device 2.0 prerequisite clears:

**Tensor bindings** (per binding, per factory):

*`UntilizeWithUnpaddingSingleCoreProgramFactory`:*
- `input` (src0_buffer) — **Case 1** (re-express): `emplace_runtime_args(core_0, {src0_buffer, ...})` passes a `Buffer*` via the framework's auto-registration path. Kernel `reader_unary_interleaved_start_id.cpp` receives `src_addr` as RTA arg 0 and already uses `TensorAccessor(src_args, src_addr)`.
- `output` (dst_buffer) — **Case 1** (re-express): `emplace_runtime_args(core_0, {dst_buffer, ...})` passes a `Buffer*`. Kernel `writer_unary_unpad_dims_split_rows.cpp` receives `dst_addr` as RTA arg 0 and already uses `TensorAccessor(dst_args, dst_addr)`.

*`UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory`:*
- `input` (src0_buffer) — **Case 1**: `emplace_runtime_args(core, {src0_buffer, ...})` passes `Buffer*`. Kernel already uses `TensorAccessor`.
- `output` (dst_buffer) — **Case 1**: `emplace_runtime_args(core, writer_rt_args)` where `writer_rt_args[0] = dst_buffer` (Buffer*). Kernel `writer_unary_stick_layout_split_rows_multicore.cpp` receives `dst_addr` arg 0 and uses `TensorAccessor(dst_args, dst_addr)`.

*`UntilizeWithUnpaddingMultiCoreShardedProgramFactory`:*
- `input` (a.buffer()) — **clean** (sharded path): bound as borrowed-memory DFB via `CBDescriptor::buffer = a.buffer()`. No TensorAccessor involved in the reader (`reader_unary_sharded.cpp` just calls `cb.push_back`). Port: `DataflowBufferSpec::borrowed_from = input`.
- `output` (dst_buffer) — **Case 1** (non-out_sharded path): `emplace_runtime_args(core, {dst_buffer, ...})` passes `Buffer*`. Kernel `writer_unary_stick_layout_interleaved_blocks.cpp` receives `dst_addr` arg 0 and uses `TensorAccessor(dst_args, dst_addr)`. *This path is also Device 2.0 blocked.*
- `output` (output.buffer()) — **clean** (out_sharded path): bound as borrowed-memory DFB via `CBDescriptor::buffer = output.buffer()`. The `writer_unary_unpad_batch_rows_sharded.cpp` and `writer_unary_unpad_width_16_sharded.cpp` kernels read from `cb_out` (the borrowed CB) without TensorAccessor — correct per causal-link gate.

*`UntilizeWithUnpaddingMultiCoreColInterleavedProgramFactory`:*
- `input` (src0_buffer) — **Case 1** (address-as-RTA bypass): `reader_desc.runtime_args.emplace_back(core, {src0_buffer->address(), ...})` passes raw address at factory line 177. Kernel `reader_unary_interleaved_col_multicore.cpp` receives `src_addr` at arg 0 and uses `TensorAccessor(src_args, src_addr)`.
- `output` (dst_buffer) — **Case 1** (address-as-RTA bypass): `writer_desc.runtime_args.emplace_back(core, {dst_buffer->address(), ...})` passes raw address at factory line 168. Kernel `writer_unary_stick_layout_col_multicore.cpp` receives `dst_addr` at arg 0 and uses `TensorAccessor(dst_args, dst_addr)`.

*`UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory`:*
- `input` (src0_buffer) — **Case 1** (address-as-RTA bypass): `reader_desc.runtime_args.emplace_back(core, {src0_buffer->address(), ...})` at factory line 308. Kernel `reader_unary_interleaved_wh_multicore.cpp` receives `src_addr` at arg 0 and uses `TensorAccessor(src_args, src_addr)`.
- `output` (dst_buffer) — **Case 1** (address-as-RTA bypass): `std::vector<uint32_t> writer_rt_args = {dst_buffer->address(), ...}` at factory line 295. Kernel `writer_unary_stick_layout_wh_multicore.cpp` receives `dst_addr` at arg 0 and uses `TensorAccessor(dst_args, dst_addr)`.

*`UntilizeWithUnpaddingMultiCoreNDShardedProgramFactory`:*
- `input` (src0_buffer) — **Case 1**: `reader_desc.emplace_runtime_args(core, {src0_buffer, start_shard_id})` passes `Buffer*`. Kernel `reader_unary_nd_sharded_blocks.cpp` receives `src_addr` at arg 0 and uses `TensorAccessor(src_args, src_addr)`.
- `output` (dst_buffer) — **Case 1**: `writer_desc.emplace_runtime_args(core, {dst_buffer, src0_buffer, start_shard_id})` passes `Buffer*` objects. Kernel `writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp` receives `dst_addr` at arg 0 and `src0_addr` at arg 1, uses `TensorAccessor(dst_args, dst_addr)` and `TensorAccessor(src0_args, src0_addr)`.

**Custom hash:** None.

## Heads-ups *(mirrors the brief)*

- **Notable LANDED constructs:** Dynamic CircularBuffer (borrowed memory) in `MultiCoreShardedProgramFactory`:
  - `device/factories/untilize_with_unpadding_multi_core_sharded_program_factory.cpp:97` — `CBDescriptor::buffer = src_sharded ? a.buffer() : nullptr` (input CB in sharded input path). Port: `DataflowBufferSpec::borrowed_from = input`.
  - `device/factories/untilize_with_unpadding_multi_core_sharded_program_factory.cpp:127` — `CBDescriptor::buffer = output.buffer()` (output CB in out_sharded path). Port: `DataflowBufferSpec::borrowed_from = output`.
- **Fake CBs (address-only):** None. All borrowed-memory CBs have genuine producer–consumer pairs.
- **Cross-op / shared kernels:** See Team-only section below. Summary: two shared-pool kernels block the sharded factory (Device 2.0 RED); five reader kernels from `eltwise/unary/` are borrowed (Device 2.0 compliant); four compute kernels from `data_movement/untilize/` are borrowed (Device 2.0 compliant).
- **RTA varargs:** `writer_unary_stick_layout_split_rows_multicore.cpp` reads a variable number of `BlockRep` entries from RTAs using a runtime-varying loop. The host packs `(n_data, n_mixed, n_pads, times, count_repeated)` tuples per block group starting at RTA offset 4. At port time, choose between named RTAs or the Metal 2.0 RTA vararg mechanism.
- **TTNN factory analysis (porter-relevant):** No pybind `create_descriptor`, no other risky pybind, no custom `override_runtime_arguments`.

## Team-only

### TensorAccessor convertibility

All Case 1 bindings above are routine re-express: the kernels already use `TensorAccessor` internally and the change is replacing the raw uint32 address-RTA with the typed binding channel. None are genuinely exotic (Case 2 does not apply to any binding).

### Out-of-directory coupling & donor shape

**Op-level roll-up:** `⭐ blocked` — two shared-pool kernels (`ttnn/kernel/`) are on Device 1.0 idioms and gate the `MultiCoreShardedProgramFactory`. All other donor kernels are Device 2.0 compliant and produce workable or excellent coupling shapes.

**Summary table:**

| Op kernel (factory) | Donor file | Donor class | Status |
|---|---|---|---|
| All interleaved readers | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | Cross-family donor (eltwise/unary) | ✓ Device 2.0 clean |
| Col reader | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_col_multicore.cpp` | Cross-family donor (eltwise/unary) | ✓ Device 2.0 clean |
| WH/block reader | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_wh_multicore.cpp` | Cross-family donor (eltwise/unary) | ✓ Device 2.0 clean |
| Sharded reader | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` | Cross-family donor (eltwise/unary) | ✓ Device 2.0 clean |
| ND sharded reader | `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reader_unary_nd_sharded_blocks.cpp` | In-family donor (data_movement/sharded) | ✓ Device 2.0 clean |
| Untilize compute (all interleaved factories) | `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp` | In-family donor (data_movement/untilize) | ✓ Device 2.0 clean (kernel_lib-based) |
| Untilize-W compute (col factory) | `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_w.cpp` | In-family donor (data_movement/untilize) | ✓ Device 2.0 clean |
| Untilize-WH compute (block factory) | `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_wh.cpp` | In-family donor (data_movement/untilize) | ✓ Device 2.0 clean |
| Variable untilize compute (ND sharded factory) | `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp` | In-family donor (data_movement/untilize) | ✓ Device 2.0 clean |
| Sharded interleaved writer (non-out_sharded path) | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` | Shared kernel pool (`ttnn/kernel/`) | ⭐ **Device 1.0 — RED GATE** |
| Sharded eltwise-copy compute (w16 path) | `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | Shared kernel pool (`ttnn/kernel/`) | ⭐ **Device 1.0 — RED GATE** |

**Per-call detail for blocked donors:**

*`ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp`* (used by `MultiCoreShardedProgramFactory`, non-out_sharded path):
- Function `write_tiles_in_block`: calls `cb_wait_front(cb_id_out0, ...)`, `get_read_ptr(cb_id_out0)`, `noc_async_write(...)`, `noc_async_write_barrier()`, `cb_pop_front(cb_id_out0, ...)` — all Device 1.0 free functions. The `TensorAccessor<DSpec>& s` parameter shape is ✓ excellent (Device 2.0 native). The CB management and NoC calls need Device 2.0 rewrite.
- Called-from: `MultiCoreShardedProgramFactory`, non-out_sharded writer path (`factory line 164`).
- Other users of this shared file: only this op (grep shows single user).

*`ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp`* (used by `MultiCoreShardedProgramFactory`, `unpad_tensor_w_16` path):
- Uses `cb_wait_front(tt::CBIndex::c_0, 1)`, `cb_reserve_back(tt::CBIndex::c_16, 1)`, `cb_pop_front(tt::CBIndex::c_0, 1)`, `cb_push_back(tt::CBIndex::c_16, 1)` — Device 1.0 free functions.
- A Device 2.0 compliant variant exists at `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/compute/eltwise_copy.cpp` (uses `CircularBuffer` wrapper objects). The `MultiCoreShardedProgramFactory` should redirect to that path after the shared-pool kernel is confirmed migrated.
- Other users of `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp`: `sharded_to_interleaved_program_factory.cpp`, `sharded_to_interleaved_partial_program_factory.cpp`, `copy_same_memory_config_program_factory.cpp` — migration affects those ops as well.

**Borrowed kernel files (file-path instantiation outside the op's own directory):**

| Kernel file | Owning family / pool | Broadly shared? |
|---|---|---|
| `ttnn/.../eltwise/unary/.../reader_unary_interleaved_start_id.cpp` | eltwise/unary | Yes — widely shared across data_movement and others |
| `ttnn/.../eltwise/unary/.../reader_unary_interleaved_col_multicore.cpp` | eltwise/unary | Shared across data_movement ops |
| `ttnn/.../eltwise/unary/.../reader_unary_interleaved_wh_multicore.cpp` | eltwise/unary | Shared across data_movement ops |
| `ttnn/.../eltwise/unary/.../reader_unary_sharded.cpp` | eltwise/unary | Shared across data_movement ops |
| `ttnn/.../data_movement/sharded/.../reader_unary_nd_sharded_blocks.cpp` | data_movement/sharded | Shared within data_movement family |
| `ttnn/.../data_movement/untilize/.../untilize.cpp` | data_movement/untilize | Shared within data_movement family |
| `ttnn/.../data_movement/untilize/.../untilize_w.cpp` | data_movement/untilize | Shared within data_movement family |
| `ttnn/.../data_movement/untilize/.../untilize_wh.cpp` | data_movement/untilize | Shared within data_movement family |
| `ttnn/.../data_movement/untilize/.../untilize_variable_num_blocks.cpp` | data_movement/untilize | Shared within data_movement family |
| `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` | `ttnn/kernel/` pool | Only this op (sole user) |
| `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | `ttnn/kernel/` pool | Shared: sharded_to_interleaved, sharded_to_interleaved_partial, copy |
| `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` (alt D2.0 version) | `ttnn/kernel/` via sharded/device/kernels/ | Shared within data_movement |

Port-together coupling: the reader kernels in `eltwise/unary/` are broadly shared — their Metal 2.0 rewrite (CB→DFB token names, named bindings) must be coordinated as one change affecting all ops that instantiate them. Similarly the untilize compute kernels are shared across untilize and untilize_with_unpadding; those families must port together.

### Relaxation candidates

No custom `compute_program_hash` present — no candidates to mine.

### TTNN factory analysis

1. **Op-owned tensors?** No. `create_output_tensors` calls `create_device_tensor(output_spec, input.device())` and returns the result directly as the op's output tensor — this is the standard output allocation, not an op-owned intermediate.
2. **MeshWorkload concept needed?** No. No `create_mesh_workload` / `create_workload_descriptor` / `cached_mesh_workload_t` anywhere. No op-owned tensors (Q1 is No), so no MeshWorkload artifact either.
3. **Pybind `create_descriptor`?** No. `untilize_with_unpadding_nanobind.cpp` only binds the top-level op function via `bind_function<"untilize_with_unpadding">` — normal op surface, not factory innards.
4. **Other migration-risky pybind?** No. The nanobind file binds nothing from `DeviceOperation` or factory classes.
5. **Custom hash?** No `compute_program_hash` defined in `UntilizeWithUnpaddingDeviceOperation` or any factory.
6. **Custom override-RTA?** No `override_runtime_arguments` defined in any factory.

## Misc anomalies *(team-only, non-gating)*

- `untilize_with_unpadding_multi_core_interleaved_program_factory.cpp:149` — `full_compute_idx` and `cliff_compute_idx` are assigned, immediately cast to `(void)`, and never used. Dead variables left over from an earlier implementation. No correctness impact.
- `untilize_with_unpadding_multi_core_shared_variables.hpp` — defines `UntilizeWithUnpaddingMultiCoreSharedVariables` struct containing `KernelHandle` fields and a `cores` vector. This struct is never referenced by any factory in the current code. Appears to be a dead artifact of a refactored factory approach. Safe to delete.
- `device/factories/untilize_with_unpadding_multi_core_sharded_program_factory.cpp:50` — `auto out_shard_spec = output.shard_spec().has_value() ? output.shard_spec().value() : shard_spec;` falls back to `shard_spec` (the *input* shard spec) when the output has no shard spec. The comment says "I am not sure it is correct to ever use the shard_spec here" — this uncertainty is worth resolving before or during the port, not during the audit.

## Questions for the user

1. **Device 2.0 compliant `eltwise_copy` variant:** A Device 2.0 compliant `eltwise_copy.cpp` already exists at `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/compute/eltwise_copy.cpp`. Could the `MultiCoreShardedProgramFactory`'s `unpad_tensor_w_16` path be redirected to use that file now (replacing the `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` reference), or is there a reason the shared-pool version is preferred? This would resolve one of the two Device 2.0 blockers without a full migration of the shared pool.

2. **Scoped subset port:** The five non-sharded factories are clear of the shared-pool Device 2.0 block. Would a partial Metal 2.0 port of those five factories (leaving `MultiCoreShardedProgramFactory` on the `ProgramDescriptor` path temporarily) be acceptable as an interim deliverable? This requires clarifying how the TTNN framework handles a mixed-API factory variant.

## Recipe notes

- The recipe's Device 2.0 check examples focus on dataflow (NoC) idioms. The `eltwise_copy.cpp` case involves compute-kernel CB management (`cb_wait_front`, `cb_push_back`) rather than NoC — the recipe's RED criteria list "manual CB index management" as a Device 1.0 indicator, which applies, but the framing is ambiguous for compute-only kernels that don't touch NoC at all. The alternative D2.0-compliant version confirms the intent is for compute kernels to use wrapper methods too. No ambiguity in the finding, but a note in the recipe's recognition bullets distinguishing dataflow vs. compute Device 2.0 idioms would be helpful for future auditors.
- The `writer_unary_stick_layout_interleaved_blocks.cpp` is the sole user of that shared-pool file (one op), making its Device 2.0 migration straightforward to schedule. Noting this as a pattern: shared-pool kernels in `ttnn/kernel/` with a single user might be better moved into the owning op's directory during migration rather than upgraded in place.
