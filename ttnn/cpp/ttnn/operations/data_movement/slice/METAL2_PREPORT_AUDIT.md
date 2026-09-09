# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/slice`

One DeviceOperation, five program factories:

- **`ttnn::prim::SliceDeviceOperation`** ([device/slice_device_operation.hpp:31-53](device/slice_device_operation.hpp#L31-L53))
  - `SliceRmProgramFactory` ([device/slice_program_factory_rm.cpp:316-432](device/slice_program_factory_rm.cpp#L316-L432))
  - `SliceRmShardedProgramFactory` ([device/slice_program_factory_rm_sharded.cpp:212-348](device/slice_program_factory_rm_sharded.cpp#L212-L348))
  - `SliceRmStrideProgramFactory` ([device/slice_program_factory_rm_stride.cpp:22-176](device/slice_program_factory_rm_stride.cpp#L22-L176))
  - `SliceTileProgramFactory` ([device/slice_program_factory_tile.cpp:23-187](device/slice_program_factory_tile.cpp#L23-L187))
  - `SliceTileTensorArgsProgramFactory` ([device/slice_program_factory_tile_tensor_args.cpp:23-193](device/slice_program_factory_tile_tensor_args.cpp#L23-L193))

Kernels each factory instantiates (all data movement; the op has no compute kernel):

| Factory | Reader | Writer |
|---|---|---|
| `SliceRmProgramFactory` | [slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp](device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp#L13-L119) | [slice_writer_unary_stick_layout_interleaved_start_id.cpp](device/kernels/dataflow/slice_writer_unary_stick_layout_interleaved_start_id.cpp#L12-L82) |
| `SliceRmShardedProgramFactory` | [slice_reader_unary_unpad_dims_rm_sharded.cpp](device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded.cpp#L13-L90) | *(none: the factory creates one kernel)* |
| `SliceRmStrideProgramFactory`, rank ≤ 4 | [reader_multicore_slice_4d.cpp](device/kernels/dataflow/reader_multicore_slice_4d.cpp#L50-L198) | [writer_multicore_slice_4d.cpp](device/kernels/dataflow/writer_multicore_slice_4d.cpp#L50-L96) |
| `SliceRmStrideProgramFactory`, rank > 4 | [reader_multicore_slice_nd.cpp](device/kernels/dataflow/reader_multicore_slice_nd.cpp#L56-L191) | [writer_multicore_slice_nd.cpp](device/kernels/dataflow/writer_multicore_slice_nd.cpp#L55-L103) |
| `SliceTileProgramFactory` | [reader_unary_unpad_dims_interleaved_start_id.cpp](device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id.cpp#L11-L53) | [writer_unary_interleaved_start_id.cpp](device/kernels/dataflow/writer_unary_interleaved_start_id.cpp#L14-L52) *(slice-owned copy)* |
| `SliceTileTensorArgsProgramFactory` | [reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp](device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp#L11-L133) | [../../eltwise/unary/…/writer_unary_interleaved_start_id.cpp](../../eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp#L18-L55) **(borrowed)** |

**Unreferenced files in the op directory** (no factory names them; their contents are out of audit scope, listed so a reader does not mistake them for live code): [device/kernels/dataflow/strided_slice_reader_rm_interleaved_nd.cpp](device/kernels/dataflow/strided_slice_reader_rm_interleaved_nd.cpp#L1-L109) and [device/kernels/dataflow/strided_slice_writer_rm_interleaved.cpp](device/kernels/dataflow/strided_slice_writer_rm_interleaved.cpp#L1-L35). A repository-wide grep for both basenames returns only the files themselves.

**Scope:** TTNN op, Gen1 (WH/BH) target, within scope of `audit/metal2_audit.md`.

**Recipe docs:** `f9451e2a21d 2026-09-09 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/slice` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `SliceDeviceOperation` → `SliceRmProgramFactory`, `SliceRmShardedProgramFactory`, `SliceRmStrideProgramFactory`, `SliceTileProgramFactory`, `SliceTileTensorArgsProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes.** All 10 referenced kernels are structurally Device 2.0 (`Noc`, `DataflowBuffer`, `CoreLocalMem`, `UnicastEndpoint`). No holdover found. |
| *Prereqs* — Cross-op escapes | Ok. One borrowed kernel file (already has a `_metal2` fork) and one in-family header of `Noc`-native helpers. Separately: a host-side consumer, `ccl/mesh_partition`, calls this op's factory entry points (see Heads-ups). |
| *Feature Support* — overall | **GREEN.** All three Appendix A entries are `N/A`. |
| *Feature Support* — Variadic-CTA | Ok. No `get_compile_time_arg_val` at a varying index anywhere in the op. |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes**, on all five factory rows. |
| *TTNN Readiness* — Concept (current) | `descriptor` (all five rows) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A. Concept is `descriptor`; the sheet's `Execution Model` column reads `SPMD`. |
| *TTNN Readiness* — Custom hash | **Yes** (not a gate; the port leaves it intact): [device/slice_device_operation.cpp:348-432](device/slice_device_operation.cpp#L348-L432) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No.** The hook is absent from the device operation; a repository grep over the op directory returns nothing. |
| *TTNN Readiness* — `override_runtime_arguments` | **Yes** on all five factories (not a gate; it selects `CustomProgramSpecFactoryConcept`). Sites listed under Gate detail. |
| *TTNN Readiness* — Pybind `create_descriptor` | **Yes** (not a gate; the port deletes the binding): [slice_nanobind.cpp:168-179](slice_nanobind.cpp#L168-L179) |
| *TTNN Readiness* — Op-owned tensors | No. The `Op-owned tensors?` column is blank on all five rows, consistent with the `descriptor` concept. |
| *TTNN Readiness* — Target concept | **`CustomProgramSpecFactoryConcept`** (no op-owned tensors) |
| *Port work* — Offset base pointer | **none.** No address argument folds a host-side offset into a base. The row-major width offset is already split out as a separate scalar. |
| *Port work* — Tensor bindings (per binding) | Case 1 on every non-borrowed binding; clean (borrowed-memory DFB) on both `SliceRmShardedProgramFactory` bindings. No Case 2. |
| *TTNN Readiness* — TensorParameter relaxation | `none` on all five rows (clears; the port applies no relaxation) |
| *Port work* — TensorAccessor 3rd arg | **none.** No accessor in the op passes a 3rd argument. |
| *Port work* — CB endpoints | 5 CBs legal 1:1, 3 CBs self-loop. No multi-binding, no dead CB, no conditional DFB. |

**CB endpoints** are port-time resolutions, not gates (see the audit recipe's *CB endpoints* subject): every out-of-window CB has a port-time resolution. Each entry below is recorded per `(CB, config)`.

## Result

**GREEN → brief issued.** All five gate-bearing subjects clear: Device 2.0, Feature compatibility, TTNN factory concept, Offset base pointers, and TensorAccessor 3rd argument. The porter brief is at `METAL2_PORT_BRIEF.md` in this directory.

Two dated triage analyses list this op and **both rows are stale**; the current code no longer matches either. Details are under Gate detail, and the two owners should be told so the analyses can be corrected:

- The offset-base-pointer analysis lists `slice` / `slice_program_factory_rm.cpp` as **Type 2**. The offset has since been split out; the base is now clean.
- The `TensorAccessor` 3rd-argument analysis lists `slice` as **Class 1 + Special**. No accessor in the op passes a 3rd argument at all, and the `Special` row's stated concern is the base offset, which is the first bullet above.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** The readiness sheet's `Is able to port?` column reads `yes` on all five factory rows. The lightweight cross-check came back clean on every column that has a code-visible counterpart:

  | Column | Sheet value | Code evidence | Agrees |
  |---|---|---|---|
  | `Concept` | `descriptor` (×5) | each factory declares `static ProgramDescriptor create_descriptor(...)`, e.g. [device/slice_program_factory_rm.hpp:26-27](device/slice_program_factory_rm.hpp#L26-L27) | yes |
  | `Custom hash (compute_program_hash)` | `yes` (×5) | [device/slice_device_operation.cpp:348-432](device/slice_device_operation.cpp#L348-L432) | yes |
  | `Backdoor custom hash (attribute_values / to_hash)` | `no` (×5) | no `attribute_values` or `to_hash` in the op directory | yes |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` (×5) | no such hook on `SliceDeviceOperation` ([device/slice_device_operation.hpp:31-53](device/slice_device_operation.hpp#L31-L53)) | yes |
  | `Override runtime args method? (PD only)` | `yes` (×5) | five definitions, listed below | yes |
  | `Pybind descriptor (nb::class_ of device op)` | `PR` (×5) | the binding is still present at [slice_nanobind.cpp:168-179](slice_nanobind.cpp#L168-L179); `PR` records that an in-flight PR addresses it, so the two are consistent | yes |
  | `Smuggled pointer (raw buffer addr in RTA/CRTA)` | `no` (×5) | every buffer argument is delivered as a `Buffer*` binding, never as `->address()`; see *Tensor bindings* below | yes |
  | `Op-owned tensors?` | blank (×5) | no `WorkloadDescriptor`, no `buffers` vector | yes |
  | `TensorParameter relaxation` | `none` (×5) | not code-visible; read, not vetted | n/a |
  | Factory-set match | 5 sheet rows | 5 factories in `program_factory_t` ([device/slice_device_operation.hpp:36-42](device/slice_device_operation.hpp#L36-L42)) | one-to-one, no phantom or missing row |

  Cross-column invariants hold: `get_dynamic_runtime_args` is `no` (it would only be possible on `descriptor` / `WorkloadDescriptor`, which is satisfied anyway), and `Op-owned tensors?` is blank on a `descriptor` concept.

  The five `override_runtime_arguments` definitions, each a one-line delegation to the shared address patcher:
  [device/slice_program_factory_rm.cpp:425-432](device/slice_program_factory_rm.cpp#L425-L432) ·
  [device/slice_program_factory_rm_sharded.cpp:415-422](device/slice_program_factory_rm_sharded.cpp#L415-L422) ·
  [device/slice_program_factory_rm_stride.cpp:178-185](device/slice_program_factory_rm_stride.cpp#L178-L185) ·
  [device/slice_program_factory_tile.cpp:189-196](device/slice_program_factory_tile.cpp#L189-L196) ·
  [device/slice_program_factory_tile_tensor_args.cpp:195-202](device/slice_program_factory_tile_tensor_args.cpp#L195-L202).
  The shared implementation is `patch_slice_program_addresses` at [device/slice_program_factory_rm_sharded.cpp:354-413](device/slice_program_factory_rm_sharded.cpp#L354-L413), declared in [device/slice_device_operation.hpp:71-78](device/slice_device_operation.hpp#L71-L78).

- **Device 2.0 (every kernel used):** **GREEN.** All 10 kernels the five factories instantiate, plus the one shared header they call into, are structurally Device 2.0. Every NoC call goes through a `Noc` object method, every CB access goes through a `DataflowBuffer` object method, and remote L1 addressing goes through `CoreLocalMem` plus `UnicastEndpoint` ([slice_reader_unary_unpad_dims_rm_sharded.cpp:60-66](device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded.cpp#L60-L66)).

  Scans that returned nothing, across all 10 kernels and the shared header: `InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedAddrGenFast`, `InterleavedPow2AddrGen*`; free-function `noc_async_read` / `noc_async_write` / `noc_async_*_barrier` / `noc_semaphore_*` (the only textual matches are inside comments describing what `noc_async_read_sharded` does internally); free-function `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front`; free-function `get_write_ptr(cb_id)` / `get_read_ptr(cb_id)`; `evil_set_write_ptr` / `evil_set_read_ptr`; raw semaphore addresses; and the stale `api/dataflow/circular_buffer.h` include.

  One CB-index free function is present and is **sanctioned**, so it is not a holdover and does not affect the gate:

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | [../../eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp](../../eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp#L27) | 27 | `get_local_cb_interface(cb_id_out).fifo_page_size` | `DataflowBuffer dfb(cb_id_out)` is constructed at line 30 |

  `get_local_cb_interface(cb_id)` is on the Device 2.0 sanctioned list, and the list does not turn on what object is in scope. The `_metal2` fork of the same file already reads the value off the object instead (`dfb.get_entry_size()`), which is where the port's version of this line lands.

  All calls into the in-family helper header use the **non-deprecated** overloads (the ones taking a leading `Noc`): `noc_async_read_sharded`, `noc_async_write_sharded` and `tt_memmove` are each called with `noc` as the first argument at every one of the 10 call sites.

- **Feature compatibility:** every Appendix A entry, in order. All three are absent from the op, so all three are `N/A` and no gate fired.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `global_circular_buffer` field on any `CBDescriptor`, no `remote_index` / `remote_cb_*` / `remote_circular_buffer.h`, no `<tt-metalium/global_circular_buffer.hpp>` in either spelling. The op builds 6 plain `CBDescriptor`s across its five factories. |
  | CBDescriptor `address_offset` (non-zero) | N/A | The token `address_offset` does not appear anywhere in the op directory. No `set_address_offset`, no four-argument `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. The two borrowed-memory CBs in `SliceRmShardedProgramFactory` set `.buffer` and leave `address_offset` at its default zero ([device/slice_program_factory_rm_sharded.cpp:282-303](device/slice_program_factory_rm_sharded.cpp#L282-L303)). |
  | GlobalSemaphore | N/A | The op uses no semaphore of any kind. A case-insensitive grep for `semaphore` over the whole op directory returns nothing. |

- **CB endpoints (GATE-free):** every CB is either legal (one producer, one consumer) or resolves with a self-loop. No multi-binding, no dead CB, no conditional DFB. Endpoints are counted per CB per node; both kernels of every two-kernel factory cover the same core range, so every node sees the same endpoint count.

  | Factory | CB | Config | Touching kernels | Roles | Resolution |
  |---|---|---|---|---|---|
  | `SliceRmProgramFactory` | `src0_cb_index` = 0 ([device/slice_program_factory_rm.cpp:351-359](device/slice_program_factory_rm.cpp#L351-L359)) | all | reader, writer | locked producer + locked consumer | legal 1:1 |
  | `SliceRmShardedProgramFactory` | `src0_cb_index` = 0, borrowed from the input buffer ([device/slice_program_factory_rm_sharded.cpp:282-291](device/slice_program_factory_rm_sharded.cpp#L282-L291)) | all | reader only, raw peek `dfb_in.get_write_ptr()` ([slice_reader_unary_unpad_dims_rm_sharded.cpp:41](device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded.cpp#L41)) | touched by one kernel, role-free | **self-loop** |
  | `SliceRmShardedProgramFactory` | `c_16`, borrowed from the output buffer ([device/slice_program_factory_rm_sharded.cpp:294-303](device/slice_program_factory_rm_sharded.cpp#L294-L303)) | all | reader only, `reserve_back` / `get_write_ptr` / `push_back` ([slice_reader_unary_unpad_dims_rm_sharded.cpp:40](device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded.cpp#L40), [:42](device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded.cpp#L42), [:89](device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded.cpp#L89)) | touched by one kernel, locked producer | **self-loop** |
  | `SliceRmStrideProgramFactory` | `in_cb` = 0 ([device/slice_program_factory_rm_stride.cpp:69-77](device/slice_program_factory_rm_stride.cpp#L69-L77)) | rank ≤ 4 | `reader_multicore_slice_4d`, `writer_multicore_slice_4d` | locked producer + locked consumer | legal 1:1 |
  | `SliceRmStrideProgramFactory` | `in_cb` = 0 | rank > 4 | `reader_multicore_slice_nd`, `writer_multicore_slice_nd` | locked producer + locked consumer | legal 1:1 |
  | `SliceTileProgramFactory` | `src0_cb_index` = 0 ([device/slice_program_factory_tile.cpp:53-60](device/slice_program_factory_tile.cpp#L53-L60)) | all | reader, writer | locked producer + locked consumer | legal 1:1 |
  | `SliceTileTensorArgsProgramFactory` | `src0_cb_index` = 0 ([device/slice_program_factory_tile_tensor_args.cpp:56-64](device/slice_program_factory_tile_tensor_args.cpp#L56-L64)) | all | reader, borrowed writer | locked producer + locked consumer | legal 1:1 |
  | `SliceTileTensorArgsProgramFactory` | `tensor_cb_index` = 1 ([device/slice_program_factory_tile_tensor_args.cpp:65-73](device/slice_program_factory_tile_tensor_args.cpp#L65-L73)) | all | reader only, which both fills and drains it ([reader_…_tensor_args.cpp:52-59](device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp#L52-L59), [:66](device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp#L66)) | touched by one kernel, locked to both FIFO roles | **self-loop** |

  Notes on the hunt for a missed toucher:
  - The hidden-second-writer pattern cannot apply here. The op has no semaphore of any kind, so there is no coordinating wait/post pair for a raw co-fill to use, and no CB is written by a kernel that is not its FIFO producer.
  - The dual-instance work-split pattern does not apply. No factory pushes the same `kernel_source` into two `KernelDescriptor`s.
  - No CB reaches zero endpoints in any config. Every `buffer_index` the six `CBDescriptor`s declare is referenced by at least one bound kernel, either through a named compile-time argument (`dfb_id_in`, `dfb_id_out`), a positional compile-time argument, or a literal `tt::CBIndex` constant.

- **Offset base pointers:** **GREEN.** Every buffer address argument in every factory is a clean base, delivered as a `Buffer*` binding rather than an `->address()` expression, so there is no host-folded offset to lose.

  This op is the catalogued example in the dated offset-base-pointer analysis, and reconciliation lands on that analysis's third outcome: **no fold present, op in the tables, so the analysis is stale.** The ops team has split the offset out. Concretely, for the factory the analysis names:

  - The analysis records reader RTA[0] as `input->address() + begins_bytes − misalignment`.
  - The current factory pushes the bare `Buffer*` at reader slot 0 ([device/slice_program_factory_rm.cpp:406](device/slice_program_factory_rm.cpp#L406)) and passes `begins_bytes - misalignment` as a **separate scalar** in the same argument list ([device/slice_program_factory_rm.cpp:99](device/slice_program_factory_rm.cpp#L99)).
  - The kernel reads that scalar as `src_offset_bytes` ([slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp:29](device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp#L29)), builds its accessor on the unshifted base ([:40](device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp#L40)), and applies the offset per read through the accessor's own byte-offset parameter ([:98](device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp#L98), and [:64](device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp#L64) on the chunked path). A comment in the kernel states the reason ([:38-39](device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp#L38-L39)).

  The scan covered every address-bearing argument in all five factories, not only the one the analysis names:

  | Factory | Argument | Value passed | Fold? |
  |---|---|---|---|
  | `SliceRmProgramFactory` | reader slot 0 | `Buffer*` `src0_buffer` ([:406](device/slice_program_factory_rm.cpp#L406)) | no |
  | `SliceRmProgramFactory` | writer slot 0 | `Buffer*` `dst_buffer` ([:414](device/slice_program_factory_rm.cpp#L414)) | no |
  | `SliceRmShardedProgramFactory` | none | the factory passes no buffer address; both tensors reach the kernel as borrowed-memory CBs | no |
  | `SliceRmStrideProgramFactory` | reader slot 0 | `Buffer*` `input_buffer` ([:128](device/slice_program_factory_rm_stride.cpp#L128), [:147](device/slice_program_factory_rm_stride.cpp#L147)) | no |
  | `SliceRmStrideProgramFactory` | writer slot 0 | `Buffer*` `output_buffer` ([:136](device/slice_program_factory_rm_stride.cpp#L136), [:160](device/slice_program_factory_rm_stride.cpp#L160)) | no |
  | `SliceTileProgramFactory` | reader common slot 0 | `Buffer*` `src0_buffer` ([:143](device/slice_program_factory_tile.cpp#L143)) | no |
  | `SliceTileProgramFactory` | writer slot 0 | `Buffer*` `dst_buffer` ([:180](device/slice_program_factory_tile.cpp#L180)) | no |
  | `SliceTileTensorArgsProgramFactory` | reader common slots 0, 1, 2 | `Buffer*` `src_buffer`, `start_buffer`, `end_buffer` ([:182-184](device/slice_program_factory_tile_tensor_args.cpp#L182-L184)) | no |
  | `SliceTileTensorArgsProgramFactory` | writer slot 0 | `Buffer*` `dst_buffer` ([:151](device/slice_program_factory_tile_tensor_args.cpp#L151), [:168](device/slice_program_factory_tile_tensor_args.cpp#L168)) | no |

  The five `->address()` calls in the op directory are all in `patch_slice_program_addresses`, the cache-hit refresh path, and every one is a bare `buffer()->address()` with no arithmetic ([device/slice_program_factory_rm_sharded.cpp:381](device/slice_program_factory_rm_sharded.cpp#L381), [:389](device/slice_program_factory_rm_sharded.cpp#L389), [:395](device/slice_program_factory_rm_sharded.cpp#L395), [:398-399](device/slice_program_factory_rm_sharded.cpp#L398-L399)).

  The two tile factories add a `start_offset` to their per-core `start_id`, but `start_id` is a **tile index**, not an address: it is consumed as `{.page_id = src_tile_id}` on the accessor ([reader_unary_unpad_dims_interleaved_start_id.cpp:40](device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id.cpp#L40)). The row-major factories do the same with a **stick index**. Neither is an offset base pointer.

  Type 3 (`address_offset`) is covered by the Appendix A row above and does not fire. Type 4 (`narrow`) does not appear.

  **Route to the offset-base-pointer analysis owner:** the `slice` row in the Type 2 table is stale and should be retired or moved to a "resolved" list.

- **TensorAccessor 3rd argument:** **N/A.** No accessor in the op passes a 3rd argument, so the subject never fires. This is a *no sites found* result, not *sites found and classified redundant*.

  A multi-line-aware parse of every `TensorAccessor(...)` construction in all 10 referenced kernels, the two unreferenced kernels, and the in-family helper header found 14 construction sites, all with exactly **two** arguments (`args`, `base_addr`). The helper header constructs no accessor of its own; it takes one as a template parameter.

  The dated 3rd-argument analysis lists `slice` twice and **both rows are stale**:
  - The `Class 1 — Dynamic page size` row does not apply: there is no 3rd argument to be dynamic. The row-major kernels take the aligned page size that `TensorAccessorArgs` bakes into the compile-time arguments, and the factory asserts the equivalence host-side in `check_accessor_page_size` ([device/slice_program_factory_rm.cpp:292-308](device/slice_program_factory_rm.cpp#L292-L308), called at [:337-340](device/slice_program_factory_rm.cpp#L337-L340)).
  - The `Special — sub-page base offset` row describes the base offset, which that analysis itself flags as "a *2nd-arg* concern, separate from the page-size 3rd arg". It belongs to the Offset base pointers subject above, where it resolves GREEN because the offset is now a separate argument.

  **Route to the 3rd-argument analysis owner:** both `slice` rows should be retired.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per factory). Every binding is either **Case 1** (the kernel feeds the base into a `TensorAccessor` and does all access through it) or **clean** (a borrowed-memory DFB, resolved by `DataflowBufferSpec::borrowed_from`). **There is no Case 2 anywhere in the op**: no kernel does hand-rolled address arithmetic on a buffer base.

  | Factory | Binding | Delivery today | Kernel use | Case |
  |---|---|---|---|---|
  | `SliceRmProgramFactory` | `input` | `Buffer*` at reader slot 0 | `TensorAccessor(src_args, src_addr)` ([kernel:40](device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp#L40)) | **Case 1** |
  | `SliceRmProgramFactory` | `output` | `Buffer*` at writer slot 0 | `TensorAccessor(dst_args, dst_addr)` ([kernel:29](device/kernels/dataflow/slice_writer_unary_stick_layout_interleaved_start_id.cpp#L29)) | **Case 1** |
  | `SliceRmShardedProgramFactory` | `input` | `CBDescriptor.buffer` ([:290](device/slice_program_factory_rm_sharded.cpp#L290)) | borrowed-memory DFB, base read via `get_write_ptr()` | **clean** |
  | `SliceRmShardedProgramFactory` | `output` | `CBDescriptor.buffer` ([:302](device/slice_program_factory_rm_sharded.cpp#L302)) | borrowed-memory DFB, filled through the FIFO | **clean** |
  | `SliceRmStrideProgramFactory` | `input` | `Buffer*` at reader slot 0 | `TensorAccessor(src_args, src_addr)` ([4d kernel:89](device/kernels/dataflow/reader_multicore_slice_4d.cpp#L89), [nd kernel:94](device/kernels/dataflow/reader_multicore_slice_nd.cpp#L94)) | **Case 1** |
  | `SliceRmStrideProgramFactory` | `output` | `Buffer*` at writer slot 0 | `TensorAccessor(dst_args, dst_addr)` ([4d kernel:72](device/kernels/dataflow/writer_multicore_slice_4d.cpp#L72), [nd kernel:79](device/kernels/dataflow/writer_multicore_slice_nd.cpp#L79)) | **Case 1** |
  | `SliceTileProgramFactory` | `input` | `Buffer*` at reader **common** slot 0 | `TensorAccessor(src_args, src_addr)` ([kernel:26](device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id.cpp#L26)) | **Case 1** |
  | `SliceTileProgramFactory` | `output` | `Buffer*` at writer slot 0 | `TensorAccessor(dst_args, dst_addr)` ([kernel:36](device/kernels/dataflow/writer_unary_interleaved_start_id.cpp#L36)) | **Case 1** |
  | `SliceTileTensorArgsProgramFactory` | `input` | `Buffer*` at reader common slot 0 | `TensorAccessor(src_args, src_addr)` ([kernel:33](device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp#L33)) | **Case 1** |
  | `SliceTileTensorArgsProgramFactory` | `start_tensor` | `Buffer*` at reader common slot 1 | `TensorAccessor(start_args, start_addr)` ([kernel:44](device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp#L44)) | **Case 1** |
  | `SliceTileTensorArgsProgramFactory` | `end_tensor` | `Buffer*` at reader common slot 2 | `TensorAccessor(end_args, end_addr)` ([kernel:45](device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp#L45)) | **Case 1** |
  | `SliceTileTensorArgsProgramFactory` | `output` | `Buffer*` at writer slot 0 | `TensorAccessor(dst_args, dst_addr)` ([donor kernel:39](../../eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp#L39)) | **Case 1** |

  Op-level roll-up: **⚠ port work** (10 Case-1 bindings; 2 clean).

  All twelve bindings arrive by the `Buffer*` form, which the framework already patches on cache hits, so none of them is the silently-wrong stale-address hazard. They are enumerated because each is a pointer argument the port replaces with a typed binding.

- **TensorParameter relaxation:** `none` (the readiness sheet's value, verbatim, on all five rows). No relaxation analysis document exists for this op, and none is expected: the `none` value clears without one.

- **TensorAccessor 3rd arg:** none. No site to drop.

- **CB endpoints:** self-loop on `SliceRmShardedProgramFactory` `src0_cb_index`=0 and `c_16` (all configs) and on `SliceTileTensorArgsProgramFactory` `tensor_cb_index`=1 (all configs). All five other CBs are legal 1:1.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding patterns to watch):** none. No CB in this op has more than two touching kernels, and no CB has two kernels locked to the same FIFO role.

- **Cross-op / shared kernels.**

  *Borrowed kernel file, one item.* `SliceTileTensorArgsProgramFactory` instantiates [../../eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp](../../eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp#L18-L55) by file path ([device/slice_program_factory_tile_tensor_args.cpp:132-133](device/slice_program_factory_tile_tensor_args.cpp#L132-L133)). A **`_metal2` fork already exists beside it**, at `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp` (a real sibling in the same directory, not a quasar-tree file), so this is a **rung 1: reuse the existing fork**. Its interface is `dfb::out`, `tensor::dst`, `args::num_pages`, `args::start_id`, which matches exactly what the slice factory supplies today ([device/slice_program_factory_tile_tensor_args.cpp:168](device/slice_program_factory_tile_tensor_args.cpp#L168)). The fork carries `#ifdef OUT_SHARDED` and `#ifdef BACKWARDS` branches; the slice factory defines neither, so no `compiler_options.defines` entry is needed.

  *Sunset list for that fork* (**a coordination and sunset list, not authorization to convert the legacy file in place**). About 30 factories still bind the legacy copy. Tracked in issue [#52228](https://github.com/tenstorrent/tt-metal/issues/52228), which the file's own header comment names as the record of the full consumer list and the sunset plan. The consumers, from a repository-wide filename grep filtered to factories that bind the `eltwise/unary` path: `data_movement/bcast`, `data_movement/concat`, `data_movement/pad`, `data_movement/permute`, `data_movement/reshape_on_device`, `data_movement/slice` (this op), `data_movement/tilize` (5 factories), `data_movement/tilize_with_val_padding` (2), `data_movement/transpose` (2), `eltwise/unary_backward/gelu_bw`, `eltwise/unary_backward/tanh_bw`, `embedding`, `examples/example` (2), `experimental/matmul/attn_matmul`, `experimental/transformer/nlp_concat_heads`, `experimental/transformer/nlp_concat_heads_boltz`, `experimental/unary_backward/gelu_backward`, `kv_cache`, `reduction/generic` (4), `reduction/prod` (2). A second, redundant `_metal2` fork of the same kernel also exists under `copy/typecast`; the one beside the original is the one to bind.

  *Slice's own kernels are not lent.* None of the nine kernel files this op owns is instantiated by any other non-quasar op; a filename grep confirms each is bound only by a slice factory. No fork is needed for any of them.

  *Function-call escapes.* Six of the kernels include [../common/kernels/common.hpp](../common/kernels/common.hpp#L34-L419), the in-family `tt::data_movement::common` helper pool, and call three functions from it: `noc_async_read_sharded`, `noc_async_write_sharded` and `tt_memmove`. All three take a leading `Noc` and a plain `uint32_t` L1 address, and the two sharded helpers take the accessor as `AddrGenType tensor`, which is the `TensorAccessor<DSpec>` case. Full inventory under Team-only.

- **RTA varargs.** Six kernels read a genuine variable-count argument block whose length tracks the tensor rank, so the port reaches for the vararg mechanism rather than naming each element. In every case the fixed scalars come **first** and the variable-count block comes last, so there are no nameable trailing scalars riding the varargs.

  | Kernel | Recognition site | What varies |
  |---|---|---|
  | [slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp:31-33](device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp#L31-L33) | `get_arg_addr(13)` cast to a pointer, then three blocks of `num_dims` read in loops at [:76-83](device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp#L76-L83) | RTA: `num_unpadded_sticks`, `num_padded_sticks`, `id_per_dim`, each `num_dims` long |
  | [slice_reader_unary_unpad_dims_rm_sharded.cpp:26-30](device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded.cpp#L26-L30) | four pointers into the RTA region, two of them at indices **computed from `num_cores_read`** (`get_arg_addr(1 + num_cores_read * 2)` and `… * 3`) | RTA: the entire tail after slot 0 is one variable-length stream whose internal layout depends on a runtime value. This is the strongest vararg case in the op. |
  | [reader_multicore_slice_nd.cpp:73-87](device/kernels/dataflow/reader_multicore_slice_nd.cpp#L73-L87) | five pointers, each advanced by `rt_args_idx += tensor_rank` | RTA: `input_dims`, `output_dims`, `slice_starts`, `slice_ends`, `slice_steps` |
  | [writer_multicore_slice_nd.cpp:73](device/kernels/dataflow/writer_multicore_slice_nd.cpp#L73) | one pointer into the RTA region | RTA: `output_dims[rank]` |
  | [reader_unary_unpad_dims_interleaved_start_id.cpp:17-23](device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id.cpp#L17-L23) | `get_common_arg_addr(1)` for two blocks of `num_dims`, plus `get_arg_addr(2)` for `id_per_dim` | CRTA **and** RTA. `num_dims` is a compile-time argument here, which still varies across instantiations, so it is a vararg and not an unrolled name set. |
  | [reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp:25-31](device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp#L25-L31), [:91-92](device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp#L91-L92) | `get_common_arg_addr(3)`, then `get_common_arg_addr(3 + 2 * num_dims)`, plus `get_arg_addr(2)` | CRTA **and** RTA. Three `num_dims`-long common blocks plus `id_per_dim`. |

  Not varargs, so the porter names each argument: the two `SliceRmStrideProgramFactory` rank ≤ 4 kernels read a **fixed** run of 25 and 9 arguments through a running `rt_args_idx++` ([reader_multicore_slice_4d.cpp:52-77](device/kernels/dataflow/reader_multicore_slice_4d.cpp#L52-L77), [writer_multicore_slice_4d.cpp:52-61](device/kernels/dataflow/writer_multicore_slice_4d.cpp#L52-L61)); so do both stick-layout writers and the two tile writers.

  **CTA varargs: none.** No kernel reads `get_compile_time_arg_val` at a varying index. Every compile-time read is at a literal index or a `constexpr` `TensorAccessorArgs<N>` offset.

- **The `id_per_dim` blocks are written by the kernel, not just read.** Four readers use their runtime-argument region as mutable per-core scratch: `id_per_dim[j]++` and `id_per_dim[j] = 0` write straight back into the argument memory ([slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp:77-79](device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp#L77-L79), [reader_unary_unpad_dims_interleaved_start_id.cpp:45-47](device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id.cpp#L45-L47), [reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp:124-127](device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp#L124-L127)). The vararg the port produces has to be writable, and the host has to keep re-emitting the initial values on every dispatch. The current code does re-emit them: `slice_tile_dynamic_args` rewrites each `id_per_dim` slot on every cache hit ([device/slice_program_factory_tile.cpp:268-271](device/slice_program_factory_tile.cpp#L268-L271)).

- **A host-side cross-op consumer depends on this op's factory entry points.** `ccl/mesh_partition` builds its MeshWorkload out of slice's factories: it calls `SliceOp::validate_on_program_cache_miss`, `SliceOp::select_program_factory` and `Factory::create_descriptor` in `create_at`, and `ttnn::prim::patch_slice_program_addresses` in its own `override_runtime_arguments` ([../../ccl/mesh_partition/device/mesh_partition_program_factory.cpp:117-158](../../ccl/mesh_partition/device/mesh_partition_program_factory.cpp#L117-L158)). It also stores a `prim::SliceDeviceOperation::program_factory_t` in its cached shared variables ([../../ccl/mesh_partition/device/mesh_partition_device_operation.hpp:49](../../ccl/mesh_partition/device/mesh_partition_device_operation.hpp#L49)). The readiness sheet records `ccl/mesh_partition` as `legacy (MeshWorkload)` with `Is able to port? = no`, so it will still be on the legacy path when slice ports. Renaming or re-signing `create_descriptor` and `override_runtime_arguments` breaks its build.

- **Two host helper functions in this op's header are used by four other ops.** `get_rm_start_offset` and `get_tiled_start_offset` are declared in [device/slice_device_operation.hpp:23-25](device/slice_device_operation.hpp#L23-L25) and defined in [device/slice_device_operation.cpp:59-91](device/slice_device_operation.cpp#L59-L91). They are pure host-side index arithmetic with no Metal 2.0 content, and they are called from `experimental/padded_slice` (2 factories), `experimental/slice_write` (3 factories) and `experimental/transformer/nlp_kv_cache_load_slice`. They stay where they are.

- **The quasar copy of this op is out of bounds.** `ttnn/cpp/ttnn/operations/experimental/quasar/slice/` holds a full pre-port copy of all five factories and all of these kernels, and the readiness sheet marks its rows `MetalV2 / N/A (done)`. It is not a template, not a naming source, and not evidence that any construct ports. A grep for any slice kernel basename will hit it; skip those hits.

## Team-only

### Out-of-directory coupling and donor signatures

**Op-level roll-up: ✓ clean.** No donor function the op calls has a signature the Metal 2.0 named tokens cannot reach, and no donor is on pre-Device-2.0 idioms.

*Summary table, one row per (op kernel, donor file):*

| Op kernel | Donor file | Donor class | Status |
|---|---|---|---|
| `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp` | `data_movement/common/kernels/common.hpp` | in-family shared | ✓ |
| `slice_writer_unary_stick_layout_interleaved_start_id.cpp` | `data_movement/common/kernels/common.hpp` | in-family shared | ✓ |
| `reader_multicore_slice_4d.cpp` | `data_movement/common/kernels/common.hpp` | in-family shared | ✓ |
| `writer_multicore_slice_4d.cpp` | `data_movement/common/kernels/common.hpp` | in-family shared | ✓ |
| `reader_multicore_slice_nd.cpp` | `data_movement/common/kernels/common.hpp` | in-family shared | ✓ |
| `writer_multicore_slice_nd.cpp` | `data_movement/common/kernels/common.hpp` | in-family shared | ✓ |
| every kernel | `tt_metal/hw/inc/api/**` (`dataflow_api.h`, `noc.h`, `dataflow_buffer.h`, `core_local_mem.h`, `endpoints.h`, `noc_traits.h`) | LLK / HAL / firmware | ✓ no concern |

`slice_reader_unary_unpad_dims_rm_sharded.cpp`, `reader_unary_unpad_dims_interleaved_start_id.cpp`, `reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp` and the slice-owned `writer_unary_interleaved_start_id.cpp` include only framework headers, so they have no function-call escape at all.

*Per-call detail.* All three functions called across the one shared-library boundary:

| Function | Signature (resource handles) | Handle form | Status |
|---|---|---|---|
| `tt::data_movement::common::noc_async_read_sharded` ([../common/kernels/common.hpp:370-405](../common/kernels/common.hpp#L370-L405)) | `(Noc noc, uint32_t l1_addr, AddrGenType tensor, uint32_t src_id, uint32_t offset, uint32_t size)` | `Noc` is Device 2.0 native; `AddrGenType` resolves to `TensorAccessor<DSpec>`, which the audit recipe classifies as its `Shape 1`; `l1_addr` is a plain L1 address, not a CB handle | ✓ excellent |
| `tt::data_movement::common::noc_async_write_sharded` ([../common/kernels/common.hpp:320-356](../common/kernels/common.hpp#L320-L356)) | same signature, write direction | as above | ✓ excellent |
| `tt::data_movement::common::tt_memmove` ([../common/kernels/common.hpp:138-204](../common/kernels/common.hpp#L138-L204)) | `(Noc noc, uint32_t dst_l1_addr, uint32_t src_l1_addr, uint32_t bytes)` | `Noc` native, plain addresses, no CB or semaphore handle | ✓ excellent |

No signature in the donor takes a `uint32_t sem_id`, a raw semaphore address, a `TensorAccessorArgs<N>`, a tensor CTA offset as a template parameter, an old-style address generator, or a legacy `CircularBuffer`. Both sharded helpers also have a `[[deprecated]]` no-`Noc` overload beside them; **no slice kernel calls the deprecated form**, verified at all 10 call sites.

*Borrowed kernel files (file-path instantiation).* One item, covered in full under Heads-ups:

| Kernel file | Owning family | Also instantiated by | `_metal2` fork beside it? |
|---|---|---|---|
| `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `eltwise/unary` | ~30 factories across `data_movement`, `reduction`, `eltwise`, `embedding`, `kv_cache`, `examples`, `experimental` (issue [#52228](https://github.com/tenstorrent/tt-metal/issues/52228)) | **yes**, `writer_unary_interleaved_start_id_metal2.cpp`, same directory |

### Relaxation candidates (fallible, candidates to verify; default strict)

**None found.** The custom `compute_program_hash` ([device/slice_device_operation.cpp:348-432](device/slice_device_operation.cpp#L348-L432)) *widens* the cache key rather than narrowing it: on top of the default it mixes in the input's full logical shape, padded shape, layout, dtype and memory config, the same six fields for the computed output spec, and the same six again for the start tensor, the end tensor and the preallocated output when each is present. A hash that adds fields reveals nothing about which tensor properties the op could safely ignore, so there is no candidate to mine here. The header comment states the reason for the widening (weak distribution of the default hash over small-integer shape sequences, issue #47602).

Two properties the hash deliberately omits are guarded by validation instead, which is worth recording because it explains why their absence is not a relaxation candidate either: the tile geometry is rejected unless it is 32×32 ([device/slice_device_operation.cpp:196-203](device/slice_device_operation.cpp#L196-L203)), and a preallocated output's full tensor layout must equal the computed one ([device/slice_device_operation.cpp:168-180](device/slice_device_operation.cpp#L168-L180)).

### TTNN factory analysis

Sheet-derived facts with code evidence. The gate conjuncts are recorded under Gate detail; the entries here are the non-gating facts that inform the port's TTNN ProgramFactory wiring (see `ttnn_factory.md`).

- **Current concept:** `descriptor`, all five factories.
- **Execution model:** the sheet's `Execution Model` column reads `SPMD`, and `Porting Target` reads `CustomProgramSpecFactoryConcept`, which matches the target derived independently from `Override runtime args method? = yes`.
- **Op-owned tensors:** none. No factory returns a `WorkloadDescriptor`, so there is no `buffers` vector.
- **MeshWorkload need:** none for this op. `SliceDeviceOperation` returns a plain `ProgramDescriptor` per factory. The MeshWorkload wrapping happens one level up, in `ccl/mesh_partition`, which is a separate device operation with its own sheet rows.
- **Pybind `create_descriptor`:** present at [slice_nanobind.cpp:168-179](slice_nanobind.cpp#L168-L179), binding `SliceTileProgramFactory::create_descriptor` only. The surrounding `bind_slice_descriptor` also exposes `SliceParams`, `SliceInputs`, and two `SliceDeviceOperation` statics ([slice_nanobind.cpp:137-167](slice_nanobind.cpp#L137-L167)); only the `create_descriptor` binding is the one the port deletes, and its removal is a user-visible API change that belongs in the port report.
- **Custom hash:** present, left intact by the port. See *Relaxation candidates* above for what it covers.
- **`get_dynamic_runtime_args`:** absent. Note that `slice_tile_dynamic_args` ([device/slice_program_factory_tile.cpp:198-281](device/slice_program_factory_tile.cpp#L198-L281)) returns `std::vector<DynamicRuntimeArg>` and could be mistaken for the deprecated hook. It is not: it is a helper the factory's own `override_runtime_arguments` calls through `apply_dynamic_runtime_args`, and the device operation declares no `get_dynamic_runtime_args` static.
- **`override_runtime_arguments`:** present on all five factories, each delegating to one shared implementation. Sites under Gate detail.
- **Informational sheet columns, recorded verbatim and not acted on:** `Op Classification` = `PD Op (custom)`; `Diego validation` = `no`; `Model` = `llama`; `ProgramFactory used in llama?` = `yes` for `SliceRmProgramFactory` and `SliceTileProgramFactory`, `no` for the other three; `Uses llama kernels? (primary or shared)` = `yes` for the two tile factories, `no` for the three row-major ones; `Provisional relaxation finding (Edwin)` = `needs fix, then none` on `SliceRmProgramFactory` and `SliceTileProgramFactory`, blank on the other three (see Questions).

## Misc anomalies  *(team-only, non-gating, not porter-actionable)*

These route to the ops team. The port does not act on any of them.

1. **A whole tensor read is dead.** `SliceTileTensorArgsProgramFactory`'s reader reads the end tensor from device into a staging CB and unpacks it into `end_indices`, and then never uses the value; the variable carries a `[[maybe_unused]]` attribute ([reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp:49](device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp#L49), written at [:80-82](device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp#L80-L82)). The `end_tensor` binding, its `TensorAccessorArgs` in the reader's compile-time arguments ([device/slice_program_factory_tile_tensor_args.cpp:84](device/slice_program_factory_tile_tensor_args.cpp#L84)), its common runtime argument ([:184](device/slice_program_factory_tile_tensor_args.cpp#L184)), its per-hit address patch ([device/slice_program_factory_rm_sharded.cpp:399](device/slice_program_factory_rm_sharded.cpp#L399)) and one full CB staging read all exist to produce a value nothing consumes. Either the end bound is meant to be enforced on device and the check is missing, or the read is vestigial. The port keeps the binding either way, because removing it would be a functional change.

2. **Dead runtime arguments in the rank ≤ 4 stride kernels.** The host emits them and the kernel never reads them. Reader: `input_n`, `output_h`, `output_d`, `output_n` ([reader_multicore_slice_4d.cpp:58-62](device/kernels/dataflow/reader_multicore_slice_4d.cpp#L58-L62)), and the locally computed `output_bytes_per_row` ([:86](device/kernels/dataflow/reader_multicore_slice_4d.cpp#L86)). Writer: `tensor_rank`, `output_h`, `output_d`, `output_n` ([writer_multicore_slice_4d.cpp:54-58](device/kernels/dataflow/writer_multicore_slice_4d.cpp#L54-L58)). Each appears exactly once in its file, at the declaration.

3. **A dead compile-time argument in all four stride kernels.** `compile_time_element_size = get_compile_time_arg_val(1)` is declared and never used; the kernels use the runtime `element_size` instead ([reader_multicore_slice_4d.cpp:81](device/kernels/dataflow/reader_multicore_slice_4d.cpp#L81), [writer_multicore_slice_4d.cpp:65](device/kernels/dataflow/writer_multicore_slice_4d.cpp#L65), [reader_multicore_slice_nd.cpp:67](device/kernels/dataflow/reader_multicore_slice_nd.cpp#L67), [writer_multicore_slice_nd.cpp:66](device/kernels/dataflow/writer_multicore_slice_nd.cpp#L66)). The host still supplies the value ([device/slice_program_factory_rm_stride.cpp:79](device/slice_program_factory_rm_stride.cpp#L79), [:82](device/slice_program_factory_rm_stride.cpp#L82)). The slot itself is occupied, because `TensorAccessorArgs<2>` starts after it, so removing the value means renumbering the accessor offset.

4. **A dead output vector in the sharded runtime-argument builder.** `get_slice_runtime_args_rm_sharded` builds an empty `writer_kernel_args` and stores it as the second half of every returned pair ([device/slice_program_factory_rm_sharded.cpp:199-200](device/slice_program_factory_rm_sharded.cpp#L199-L200)), but the factory creates only a reader kernel ([:345](device/slice_program_factory_rm_sharded.cpp#L345)), so the second half is never read. The whole return type could be a single vector.

5. **An attribute the factory ignores is still fed to the cache key.** `SliceRmShardedProgramFactory` logs a warning and ignores `sub_core_grids` ([device/slice_program_factory_rm_sharded.cpp:234-236](device/slice_program_factory_rm_sharded.cpp#L234-L236)), yet `sub_core_grids` is mixed into `compute_program_hash` unconditionally ([device/slice_device_operation.cpp:363](device/slice_device_operation.cpp#L363)). Two calls that differ only in an ignored `sub_core_grids` therefore compile two identical programs. Correct, but wasteful of cache entries.

6. **Dead preprocessor branches.** Both tile writers carry `#ifdef OUT_SHARDED` and `#ifdef BACKWARDS` branches ([writer_unary_interleaved_start_id.cpp:29-51](device/kernels/dataflow/writer_unary_interleaved_start_id.cpp#L29-L51) and the borrowed copy). No slice factory sets any `defines`, so neither branch can be reached from this op.

7. **The shared address patcher lives in an unexpected file.** `patch_slice_program_addresses` handles all five factories and is shared with `ccl/mesh_partition`, but it is defined inside `slice_program_factory_rm_sharded.cpp` ([:354-413](device/slice_program_factory_rm_sharded.cpp#L354-L413)) rather than in `slice_device_operation.cpp`, where its declaration sits. Nothing is wrong; it is just hard to find.

8. **A compile-time argument read into a non-`constexpr` local.** In the tensor-args reader, `tile_width` and `tile_height` are declared `const uint32_t` from `get_compile_time_arg_val`, while every neighbouring compile-time read in the same function is `constexpr` ([reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp:12-19](device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp#L12-L19)). This one is also relevant to the port, so it is repeated in the brief.

9. **Two kernel files in the op directory are referenced by nothing** ([device/kernels/dataflow/strided_slice_reader_rm_interleaved_nd.cpp](device/kernels/dataflow/strided_slice_reader_rm_interleaved_nd.cpp#L1-L109), [device/kernels/dataflow/strided_slice_writer_rm_interleaved.cpp](device/kernels/dataflow/strided_slice_writer_rm_interleaved.cpp#L1-L35)). A repository-wide grep for either basename returns only the files themselves. They look like superseded versions of the stride path. Candidates for deletion, outside the port.

## Per-DeviceOperation attribution

Not applicable. The directory holds exactly one DeviceOperation, `SliceDeviceOperation`. Every finding above is already attributed to the factory it applies to.

## Questions for the user

1. **A provisional relaxation column disagrees with the gating one.** The gating `TensorParameter relaxation` column reads `none` on all five rows, so the relaxation gate clears. The adjacent `Provisional relaxation finding (Edwin)` column reads **`needs fix, then none`** on `SliceRmProgramFactory` and `SliceTileProgramFactory`, and is blank on the other three. That column is not one the audit recipe reads, and I have not treated it as a gate. Worth confirming with the readiness-sheet owner whether the "needs fix" it refers to is already done (which would make the two columns consistent) or is still open.

2. **Is the end tensor's device read supposed to do something?** Misc anomaly 1 above: `SliceTileTensorArgsProgramFactory` reads the end tensor from device on every dispatch and discards the values. This is an ops-team question, not a port blocker, but the port will carry a tensor binding whose only purpose today is to feed a dead variable.

## Recipe notes

Friction with the audit recipe itself, for the recipe maintainer.

1. **The causal-link gate in *TensorParameter analysis* is worded for reads only.** It defines the clean case as "a borrowed-memory DFB read: it reads tensor data through `cb_*.wait_front` / `cb_*.get_read_ptr` from a CB that is itself a borrowed-memory CB." `SliceRmShardedProgramFactory` has a borrowed-memory CB the kernel **writes**: the CB is bound to `output.buffer()` and the kernel fills it with `reserve_back` / `get_write_ptr` / `push_back` ([device/slice_program_factory_rm_sharded.cpp:294-303](device/slice_program_factory_rm_sharded.cpp#L294-L303), [slice_reader_unary_unpad_dims_rm_sharded.cpp:40](device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded.cpp#L40), [:42](device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded.cpp#L42)). The mechanism and the `borrowed_from` translation are identical, so I marked the binding clean by analogy, but the rule as written does not cover it. Suggest widening to "a borrowed-memory DFB access" and listing the write-side signals (`reserve_back` / `get_write_ptr` / `push_back`) alongside the read-side ones. The false-positive guard list has the same gap: it names "Kernel reading from a borrowed-memory CB via `cb.get_read_ptr()`" and no write equivalent.

2. **A `Special` row in the 3rd-argument triage analysis is not a 3rd-argument finding.** The analysis's `Special` table lists `slice` with the explanation "sub-page base offset … a *2nd-arg* concern, separate from the page-size 3rd arg". An auditor using the analysis as the accelerator the recipe recommends would carry an `S` verdict, which the taxonomy marks GATE, into the *TensorAccessor 3rd argument* subject, where it does not belong: the concern is the accessor's **base**, which is the *Offset base pointers* subject and resolves GREEN here. Suggest the 3rd-argument subject say explicitly that a `Special` row in that analysis may encode a base-offset concern, and that such a row routes to *Offset base pointers* rather than gating the 3rd-argument subject.

3. **A host-side cross-op consumer has no home in the recipe.** *Out-of-directory coupling* covers two escape types, both kernel-level: a kernel calling another op's helper function, and a factory instantiating another op's kernel file. This op has a third kind: another op's host factory (`ccl/mesh_partition`) calls slice's `create_descriptor`, `select_program_factory` and `patch_slice_program_addresses`, and stores slice's `program_factory_t` in its own cached state. Nothing in the recipe asks the auditor to look for it, yet it is the coupling most likely to break a build, because the ported entry points change signature and the consumer cannot co-migrate (its own `Is able to port?` reads `no`). I filed it under the brief's open "anything else the porter needs" bullet, which is where the recipe says such things go, but a named category or a one-line prompt in *Out-of-directory coupling* would stop the next auditor from missing it.

4. **The `Smuggled pointer` cross-check would benefit from one clarifying line.** The recipe explains the `Buffer*`-binding form thoroughly under *TensorParameter analysis*, and separately lists a "smuggled RTA pointer" as a blocking signal in the *TTNN factory concept* table. This op pushes a `Buffer*` into runtime-argument slot 0 of every kernel while the sheet's `Smuggled pointer` column reads `no`, which is correct but takes a moment to confirm. A one-line note in the cross-check list, saying that the column means an `->address()` value and not a `Buffer*` binding, would remove the doubt.
