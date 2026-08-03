# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/reduction/topk`

One device operation shares this directory, with two program factories:

- **`TopKDeviceOperation`** (`device/topk_device_operation.hpp`, `device/topk_device_operation.cpp`)
  - `TopKSingleCoreProgramFactory` (`device/topk_single_core_program_factory.cpp`)
  - `TopKMultiCoreProgramFactory` (`device/topk_multi_core_program_factory.cpp`)

Nine kernel files are referenced, all owned by this op (no borrowed kernel sources):

| Kernel | Bound by |
|---|---|
| `device/kernels/dataflow/reader_create_index_tensor.cpp` | single-core |
| `device/kernels/dataflow/writer_binary_interleaved.cpp` | single-core |
| `device/kernels/compute/topk.cpp` | single-core |
| `device/kernels/dataflow/reader_create_index_local_topk.cpp` | multi-core (local cores) |
| `device/kernels/dataflow/reader_final_topk.cpp` | multi-core (final core) |
| `device/kernels/dataflow/writer_local_topk.cpp` | multi-core (local cores) |
| `device/kernels/dataflow/writer_final_topk.cpp` | multi-core (final core) |
| `device/kernels/compute/topk_local.cpp` | multi-core (local cores) |
| `device/kernels/compute/topk_final.cpp` | multi-core (final core) |

Two in-directory kernel headers are pulled in by those files: `device/kernels/compute/topk_common_funcs.hpp`
(also included by two other ops, see *Out-of-directory coupling*) and
`device/kernels/dataflow/topk_dataflow_common.hpp` (private to this op). No unreferenced kernel files
are present. `topk.cpp` / `topk.hpp` at the op root are the composite front end (transpose, pad, slice
around the device op) and carry no kernels, buffers, or bindings.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `ccf3df7c4ab 2026-08-03 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/reduction/topk` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `TopKDeviceOperation` → `TopKSingleCoreProgramFactory`, `TopKMultiCoreProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes.** All nine kernels are structurally Device 2.0 (`Noc`, `DataflowBuffer`, `Semaphore<>`, `UnicastEndpoint`, `CoreLocalMem`). No CB-index free-function holdovers, no legacy addr-gen. |
| *Prereqs* — Cross-op escapes | Ok. Kernel includes are all `api/*` (framework) plus two in-directory headers. One inbound coupling: `topk_common_funcs.hpp` is *lent* to two other ops. |
| *Feature Support* — overall | GREEN (no Appendix A entry fires) |
| *Feature Support* — Variadic-CTA | Ok — every CTA is read at a constant index |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes**, both factory rows |
| *TTNN Readiness* — Concept (current) | `descriptor` (both factories) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes (both rows) |
| *TTNN Readiness* — Custom hash | No (also `Backdoor custom hash` = no; confirmed by grep) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No (confirmed by grep of the device op) |
| *TTNN Readiness* — `override_runtime_arguments` | No (confirmed by grep) |
| *TTNN Readiness* — Pybind `create_descriptor` | No (`topk_nanobind.cpp` binds only the user-facing op) |
| *TTNN Readiness* — Op-owned tensors | No (cell blank; `descriptor` concept cannot carry them) |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` |
| *Port work* — Offset base pointer | none (no host-folded offset reaches any kernel arg) |
| *Port work* — Tensor bindings (per binding) | 8 bindings across the two factories, **all Case 1** |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none — no accessor passes a third argument |
| *Port work* — CB endpoints | legal / self-loop only; **plus two zero-endpoint per-node instances that must NOT be dropped** (see below) |

**CB endpoints** are per-CB actions, not gates: every out-of-window CB has a port-time answer (a
self-loop for one toucher, a 1P+1C assignment for two, the multi-binding advanced option for a
genuine excess, or a dead-CB drop for zero endpoints). This op needs **no multi-binding flag**
anywhere. It does have two CBs (`c_0`, `c_1`) that are fully live on the multi-core local cores and
have **zero endpoints on the final core**, which is *not* the dead-CB case: see
*CB endpoints* under Gate detail, and Question 1.

## Result

**GREEN → brief issued.** Both factories clear every gate. `TopKDeviceOperation` is already on the
`ProgramDescriptor` API with no custom hash, no runtime-arg-update hook, and no pybound descriptor;
all nine kernels are already Device 2.0; no Appendix A feature is in use; no tensor address has a
host-folded offset; and no `TensorAccessor` passes an explicit page size. All eight tensor bindings
are the straightforward Case 1 form (address fed to a `TensorAccessor`), and they already ride the
framework's patched `emplace_runtime_args(MeshTensor)` channel rather than a raw `->address()` RTA.

Two items are not gates but deserve the porter's attention before construction, and both are recorded
as questions for the ops and framework owners:

1. The multi-core factory declares `c_0` and `c_1` over the union of local and final cores, yet no
   final-core kernel references either index. Whether a Metal 2.0 `DataflowBufferSpec` may span a core
   range where some nodes hold no binding decides whether the porter keeps the range as-is or narrows
   it to the local cores.
2. The multi-core path depends on a **cross-core SRAM address assumption**: a local core reads the
   write pointer of its *own* `c_4` / `c_5` instance and uses that value as the destination address on
   the final core. This is correct today only because a CB declared over a core range set is placed at
   one common address across that whole range.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** The readiness sheet carries exactly two
  rows for `reduction/topk`, one per factory, and both read `Is able to port? = yes`. Every conjunct
  is `no`/clear: `Concept = descriptor`, `Custom hash = no`, `Backdoor custom hash = no`,
  `Runtime-args update (get_dynamic_runtime_args) = no`, `Override runtime args method? = no`,
  `Pybind descriptor = no`, `Is safe to port? = yes`, `Smuggled pointer = no`. The sheet also records
  `Op Classification = PD Op (pointer-patching)`, `Pointer patching perf issue? = OK`,
  `Formerly custom hashed? = no`, `TensorParameter relaxation = none`, and
  `Porting Target = ProgramSpecFactoryConcept`.

  Cross-check against the code, all clean:

  | Column | Code evidence | Agrees? |
  |---|---|---|
  | `Concept` | both factories define `create_descriptor()` returning `tt::tt_metal::ProgramDescriptor` (`device/topk_device_operation.hpp:25`, `:32`) | yes |
  | `Custom hash` | no `compute_program_hash` anywhere in the op directory | yes |
  | `get_dynamic_runtime_args` | absent from `TopKDeviceOperation` (`device/topk_device_operation.hpp:38-44` lists the full hook set) | yes |
  | `Override runtime args method?` | no `override_runtime_arguments` in the op directory | yes |
  | `Pybind descriptor` | `topk_nanobind.cpp` registers only the user-facing `ttnn::topk`; no `create_descriptor` binding, no `nb::class_` of the device op | yes |
  | `Op-owned tensors?` | no `WorkloadDescriptor`, so no `buffers` vector; cell is blank, read as `no` | yes |
  | Factory-set match | `program_factory_t` (`device/topk_device_operation.hpp:38`) holds exactly the two factories the sheet rows name; no phantom and no missing row | yes |

  Cross-column invariants hold: `get_dynamic_runtime_args = no` on a `descriptor` concept is valid,
  and op-owned tensors are absent on a `descriptor` row as required.

- **Device 2.0 (every kernel used):** **GREEN.** All nine kernels are on Device 2.0 data-movement
  idioms throughout. Evidence, by kernel:

  | Kernel | Device 2.0 objects in use |
  |---|---|
  | `reader_create_index_tensor.cpp` | `Noc noc` + `noc.async_read(accessor, dfb, …)` (`:42-56`), `DataflowBuffer` (`:43-44`) |
  | `writer_binary_interleaved.cpp` | `Noc` + `noc.async_write(dfb, accessor, …)` (`:33-47`), `DataflowBuffer` (`:34-35`) |
  | `reader_create_index_local_topk.cpp` | `Noc`, `DataflowBuffer` (`:35-37`) |
  | `reader_final_topk.cpp` | `Noc`, `Semaphore<>` (`:26-27`), `DataflowBuffer` (`:28-29`) |
  | `writer_local_topk.cpp` | `Noc`, `Semaphore<>`, `UnicastEndpoint` (`:31-38`) |
  | `writer_final_topk.cpp` | `Noc`, `DataflowBuffer` (`:33-35`) |
  | `topk.cpp` (compute) | `DataflowBuffer` objects for every buffer operation (`:141-146`, `:306-307`) |
  | `topk_local.cpp` / `topk_final.cpp` (+ `topk_common_funcs.hpp`) | `DataflowBuffer` objects for every buffer operation (`topk_local.cpp:129-132`, `topk_common_funcs.hpp:23-26`) |
  | `topk_dataflow_common.hpp` | `DataflowBuffer` + `CoreLocalMem<volatile T>` over `dfb.get_write_ptr()` (`:32-36`) |

  A targeted scan for holdovers found **none**: no `cb_wait_front` / `cb_push_back` / `cb_pop_front` /
  `cb_reserve_back` calls, no raw `noc_async_read` / `noc_async_write` / `noc_semaphore_*`, no
  `get_noc_addr*`, no `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedPow2AddrGen*`, no
  `get_local_cb_interface`, no `evil_set_*_ptr`, and no free-function `get_write_ptr(cb_id)` /
  `get_read_ptr(cb_id)` / `get_tile_size(cb_id)` (every such call is a method on a `DataflowBuffer`
  object). The single textual match for a legacy name is prose in a comment:
  `device/kernels/dataflow/writer_local_topk.cpp:71` says "`cb_pop_front` only advances the read
  pointer" while the code beneath it calls the Device 2.0 `pop_front` method. Cosmetic only, recorded
  under Misc anomalies.

  Compute kernels pass CB indices to the compute LLK entry points (`transpose_tile(idx, …)`,
  `pack_tile(reg, idx)`, `reconfig_data_format_srca(idx)`, `copy_tile(idx, …)`,
  `compute_kernel_hw_startup(…)`). Those are the standard compute API, not the Device 2.0
  data-movement boundary, so they are not holdovers. Every *buffer* operation in those kernels
  (`reserve_back` / `push_back` / `wait_front` / `pop_front`) is already a `DataflowBuffer` method.

- **Feature compatibility:** every Appendix A entry, in order. All absent.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `global_circular_buffer` field on any `CBDescriptor`, no `remote_index` / `remote_cb_*` idiom, no `global_circular_buffer.hpp` include. All ten `CBDescriptor` literals across the two factories set only `total_size`, `core_ranges`, and `format_descriptors`. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset`, no `set_address_offset`, no four-argument `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. The only offset arithmetic in the op is kernel-side and applied to a CB write pointer, not to a `CBDescriptor` (`writer_local_topk.cpp:49-50`), which the entry's false-positive guard excludes. |
  | GlobalSemaphore | N/A | Both semaphores are plain `SemaphoreDescriptor` entries (`topk_multi_core_program_factory.cpp:322-333`), consumed kernel-side as `Semaphore<>` objects. No `GlobalSemaphore` type, no `CreateGlobalSemaphore`, no `global_semaphore.hpp` include. The single-core factory declares no semaphores. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` is a fixed set of named tensors (`TopkInputs`: one input, one optional indices, one optional output pair) with no variable-count container. Every kernel reads its CTAs at constant indices via `get_compile_time_arg_val(<literal>)`; no `get_compile_time_arg_val(i)` inside a loop and no kernel templated over a runtime-derived count. |

- **CB endpoints (GATE-free):** no multi-binding flag is needed anywhere in this op, and there is no
  hidden second writer on any node. Counted per CB per node, per config.

  **Single-core factory** (one config; eight CBs, all over the same core range; every core in that
  range runs reader, writer, and compute). Note that despite the factory's name it spreads `Ht` rows
  across several cores via `split_work_to_cores` (`topk_single_core_program_factory.cpp:70`), so this
  tally applies to each of those cores identically.

  | CB | Index | Touchers on the node | Verdict / action |
  |---|---|---|---|
  | `input_cb` | `c_0` | reader produces, compute consumes | legal 1P+1C |
  | `index_cb` | `c_1` | reader produces, compute consumes | legal 1P+1C |
  | `transposed_val_cb` | `c_2` | compute only (produces and consumes) | **self-loop** |
  | `transposed_ind_cb` | `c_3` | compute only | **self-loop** |
  | `result_prep_val_cb` | `c_4` | compute only | **self-loop** |
  | `result_prep_ind_cb` | `c_5` | compute only | **self-loop** |
  | `output_val_cb` | `c_6` | compute produces, writer consumes | legal 1P+1C |
  | `output_ind_cb` | `c_7` | compute produces, writer consumes | legal 1P+1C |

  **Multi-core factory, local-core nodes** (`local_cores_range_set`; reader_local, writer_local,
  compute_local resident).

  | CB | Index | Touchers on the node | Verdict / action |
  |---|---|---|---|
  | `input_cb` | `c_0` | reader_local produces, compute_local consumes | legal 1P+1C |
  | `index_cb` | `c_1` | reader_local produces, compute_local consumes | legal 1P+1C |
  | `input_transposed_cb` | `c_2` | compute_local only | **self-loop** |
  | `index_transposed_cb` | `c_3` | compute_local only | **self-loop** |
  | `gathered_values_cb` | `c_4` | writer_local only, and only as a raw pointer peek (`final_values_dfb.get_write_ptr()`, `writer_local_topk.cpp:45`) | **self-loop** |
  | `gathered_indices_cb` | `c_5` | writer_local only, raw pointer peek (`writer_local_topk.cpp:46`) | **self-loop** |
  | `values_cb` | `c_8` | compute_local produces, writer_local consumes | legal 1P+1C |
  | `output_ind_cb` | `c_9` | compute_local produces, writer_local consumes | legal 1P+1C |

  **Multi-core factory, final-core node** (`final_cores_range_set`; reader_final, writer_final,
  compute_final resident).

  | CB | Index | Touchers on the node | Verdict / action |
  |---|---|---|---|
  | `input_cb` | `c_0` | **none** | **zero endpoints — do NOT drop** (below) |
  | `index_cb` | `c_1` | **none** | **zero endpoints — do NOT drop** (below) |
  | `gathered_values_cb` | `c_4` | reader_final produces, compute_final consumes | legal 1P+1C |
  | `gathered_indices_cb` | `c_5` | reader_final produces, compute_final consumes | legal 1P+1C |
  | `final_values_cb` | `c_6` | compute_final only | **self-loop** |
  | `final_indices_cb` | `c_7` | compute_final only | **self-loop** |
  | `values_cb` | `c_8` (second descriptor, final range, output dtype) | compute_final produces, writer_final consumes | legal 1P+1C |
  | `output_ind_cb` | `c_9` | compute_final produces, writer_final consumes | legal 1P+1C |

  **The hidden-second-writer face was hunted and resolved.** The op does contain the recognition
  pattern (a raw write into a CB, gated by a semaphore pair rather than FIFO sync), but in a
  cross-core form: `writer_local_topk.cpp:64-92` writes tile bytes straight into `c_4` / `c_5` while
  `reader_final_topk.cpp:34-57` performs the FIFO bookkeeping (`reserve_back` / `push_back`) for those
  same buffers, coordinated by `sender_sem` / `receiver_sem`. The raw writer sits on a **different
  node** than the instance it fills, so it does not add an endpoint to the final core's tally: on its
  own node it binds `c_4` / `c_5` only to read the write pointer (a role-free peek, hence the
  self-loop above), and the bytes travel over the NoC. The final core therefore stays at one locked
  producer (reader_final, doing accounting for data it does not itself write) plus one locked consumer
  (compute_final). No multi-binding.

  **The two zero-endpoint instances are not dead CBs.** `c_0` and `c_1` are declared over
  `all_cores_range_set` (`topk_multi_core_program_factory.cpp:171-191`) but no final-core kernel
  references either index: reader_final uses `c_4` / `c_5` (CTAs 9 and 10,
  `topk_multi_core_program_factory.cpp:377-378`), writer_final uses `c_8` / `c_9`, and compute_final's
  CTAs 0 and 1 are also `c_4` / `c_5` (`:468-469`). Both indices *are* live on every local core, so
  the recipe's dead-CB drop does not apply: dropping the allocation would remove a buffer two
  local-core kernels use. What is genuinely in question is only whether the DFB's declared range may
  keep covering the final core when no final-core kernel binds it, or whether the porter must narrow
  the range to `local_cores_range_set`. Narrowing is not free: it changes the final core's SRAM
  layout, which the factory deliberately arranged (`:158-168`) so that the shared CBs are placed
  before the core-specific ones. Raised as Question 1 rather than resolved here. A confirmed dead CB
  would resurface loudly at the spec validator; a wrongly narrowed range would not, so the
  conservative reading is the right one.

- **Offset base pointers:** **GREEN.** No address argument anywhere folds a host-computed offset into
  a tensor base. There is no `->address()` expression at all in the op directory: every tensor base
  reaches a kernel through `KernelDescriptor::emplace_runtime_args(core, {mesh_tensor, …})`, which
  registers the tensor as a framework-patched binding and writes a clean base address into the arg
  slot (`tt_metal/api/tt-metalium/program_descriptors.hpp:164-202`). The six address-bearing call
  sites are `topk_single_core_program_factory.cpp:260-279` (input, optional indices, values, indices)
  and `topk_multi_core_program_factory.cpp:500-512` and `:533-538` (input, optional indices, values,
  indices). Each passes the tensor itself, with no arithmetic. `reduction/topk` does not appear in the
  offset-base-pointer triage analysis (a dated prior), and the scan agrees with that silence: no fold,
  op not in the tables, so every address is a clean base handed to TensorParameter analysis. Type 3
  (`address_offset`) is absent; Type 4 (`narrow`) does not appear.

  For completeness, the one offset expression that *looks* similar is not this finding: at
  `writer_local_topk.cpp:49-50` a kernel adds `start_wt * tile_bytes * Kt` to a **CB write pointer**
  it read on-device. That is kernel-side SRAM arithmetic on a buffer pointer, not a host fold into a
  tensor base, and no `TensorAccessor` is built from it.

- **TensorAccessor 3rd argument:** **GREEN — the subject does not fire.** Every `TensorAccessor`
  construction in this op passes exactly two arguments (`args`, `base_addr`), so there is no explicit
  page size to classify and nothing for the porter to drop. All eight sites:
  `reader_create_index_tensor.cpp:33` and `:40`; `writer_binary_interleaved.cpp:30` and `:31`;
  `reader_create_index_local_topk.cpp:33` and `:44`; `writer_final_topk.cpp:30` and `:31`.
  `reduction/topk` does not appear in the 3rd-arg triage analysis, and the scan agrees.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per factory) — **all Case 1**: the base address is fed to a
  `TensorAccessor` and every memory access goes through the accessor. Each becomes a `TensorParameter`
  / `TensorBinding`, the kernel builds `TensorAccessor(tensor::name)`, and both the address arg and
  its `TensorAccessorArgs` plumbing disappear.

  | Factory | Binding | Delivered at | Consumed at | Case |
  |---|---|---|---|---|
  | single-core | `input` | reader RTA 0 (`topk_single_core_program_factory.cpp:263`) | `reader_create_index_tensor.cpp:40` | 1 |
  | single-core | `indices` (optional) | reader RTA 3 (`topk_single_core_program_factory.cpp:266-270`) | `reader_create_index_tensor.cpp:33`, currently unreachable (see Misc anomaly 2) | 1 |
  | single-core | `values` output | writer RTA 0 (`topk_single_core_program_factory.cpp:275`) | `writer_binary_interleaved.cpp:30` | 1 |
  | single-core | `indices` output | writer RTA 1 (`topk_single_core_program_factory.cpp:276`) | `writer_binary_interleaved.cpp:31` | 1 |
  | multi-core | `input` | reader_local RTA 0 (`topk_multi_core_program_factory.cpp:503`) | `reader_create_index_local_topk.cpp:33` | 1 |
  | multi-core | `indices` (optional) | reader_local RTA 4 (`topk_multi_core_program_factory.cpp:507-511`) | `reader_create_index_local_topk.cpp:44` | 1 |
  | multi-core | `values` output | writer_final RTA 0 (`topk_multi_core_program_factory.cpp:536`) | `writer_final_topk.cpp:30` | 1 |
  | multi-core | `indices` output | writer_final RTA 1 (`topk_multi_core_program_factory.cpp:537`) | `writer_final_topk.cpp:31` | 1 |

  No Case 2 (raw pointer) binding exists: no kernel does hand-rolled address arithmetic on a tensor
  base, so the `get_bank_base_address` bridge is not needed anywhere. No borrowed-memory DFB exists
  either (no `set_globally_allocated_address` in the op), so no binding is clean-by-causal-link.
  Because the addresses already ride the patched `emplace_runtime_args(MeshTensor)` channel rather
  than a bare `->address()` RTA, none of these is a live stale-address hazard today; the port replaces
  the delivery mechanism with the typed binding, which is routine work.

- **TensorParameter relaxation:** none. The sheet lists `none` for both factories, consistent with
  the absence of a custom hash.
- **TensorAccessor 3rd arg:** none — no site passes one.
- **CB endpoints:**
  - **self-loop**: single-core `c_2`, `c_3`, `c_4`, `c_5` (compute-only scratch); multi-core local-core
    `c_2`, `c_3` (compute-only scratch) and `c_4`, `c_5` (pointer-peek only); multi-core final-core
    `c_6`, `c_7` (compute-only scratch).
  - **legal 1P+1C, no action**: single-core `c_0`, `c_1`, `c_6`, `c_7`; multi-core local-core `c_0`,
    `c_1`, `c_8`, `c_9`; multi-core final-core `c_4`, `c_5`, `c_8`, `c_9`.
  - **multi-binding advanced option**: not needed anywhere.
  - **dead-CB drop**: none. `c_0` / `c_1` have zero endpoints on the final-core node but are live on
    the local cores, so they are not droppable (Question 1).

## Heads-ups  *(mirrors the brief)*

- **Cross-core SRAM address assumption (the highest-value thing to verify first).** The multi-core
  path moves data between cores by having a local core read the write pointer of its **own** `c_4` /
  `c_5` instance and use that value as the destination address on the final core
  (`writer_local_topk.cpp:45-50`, then `:69` and `:89`). This is correct only while a CB declared over
  a core range set is placed at one common address on every core in that range. The factory documents
  the legacy allocator behaviour it relies on and orders its allocations for it
  (`topk_multi_core_program_factory.cpp:158-168`). If the Metal 2.0 DFB allocator does not offer the
  same guarantee, the multi-core factory mis-addresses silently rather than failing loudly. Raised as
  Question 2.
- **CB naming is actively misleading across the multi-core kernels; take binding names from the
  factory, not from the kernel locals.** Three collisions to watch:
  - `reader_final_topk.cpp:22-23` and `writer_local_topk.cpp:25-26` call CTAs 9 and 10
    `final_values_dfb_index` / `final_indices_dfb_index`, but the factory passes
    `gathered_values_cb_index` (`c_4`) and `gathered_indices_cb_index` (`c_5`) there
    (`topk_multi_core_program_factory.cpp:377-378`, `:403-404`). The factory's *actual* `final_*` CBs
    are `c_6` / `c_7`, which those kernels never touch.
  - `topk_final.cpp:47-50` calls CTAs 0 and 1 `input_dfb_index` / `index_dfb_index` (they are `c_4` /
    `c_5`, the gathered buffers) and CTAs 2 and 3 `input_transposed_dfb_index` /
    `index_transposed_dfb_index` (they are `c_6` / `c_7`, the final workspaces). Neither `c_0` nor
    `c_1` reaches this kernel despite the names.
  - `topk_local.cpp` and `topk_final.cpp` share `topk_common_funcs.hpp` with the same parameter names
    bound to different CBs in each caller.
- **Two CBs with one index and two descriptors.** `values_cb_index` (`c_8`) is declared twice in the
  multi-core factory, over disjoint core ranges and with **different data formats**: the local-core
  copy uses `compute_cb_data_format` (`topk_multi_core_program_factory.cpp:273-281`) and the
  final-core copy uses `value_cb_data_format` (`:283-291`). The comment at `:261-271` explains why
  (bf16 on the local side preserves values through the transposed-layout NoC transfer; the output
  dtype on the final side matches the DRAM write). Both must survive the port as two separate specs.
- **Cross-op / shared kernels:** `device/kernels/compute/topk_common_funcs.hpp` is **lent** to two
  other ops (below). No `_metal2` fork exists beside any topk kernel, and there is no
  `experimental/quasar` copy of this op, so nothing to reuse and nothing to be misled by.
- **RTA varargs:** none. Every kernel reads a fixed set of args at constant indices, so all runtime
  args become named args.

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ✓ clean** for outbound escapes, with one inbound coupling to coordinate.

- Every `#include` in every topk kernel resolves either to `api/*` under the framework
  (`api/dataflow/dataflow_api.h`, `noc.h`, `dataflow_buffer.h`, `noc_semaphore.h`, `endpoints.h`,
  `api/tensor/noc_traits.h`, `api/core_local_mem.h`, `api/compute/*`) or to one of the two
  in-directory headers. Framework headers are donor class 1 (LLK / HAL / firmware): no concern, never
  forked, not the porter's to modify.
- **No borrowed kernel files.** Both factories instantiate only kernel sources under
  `ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/`. There is no cross-family or shared-pool
  kernel instantiation, so the per-call donor shape table has no ⚠ / ✗ / ⭐ rows and the per-call
  detail section is omitted.
- **One inbound coupling: topk *lends* `topk_common_funcs.hpp`.** Two other ops include this op's
  compute header into their own kernels:
  - `ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_grouped_gate/device/kernels/compute/deepseek_grouped_gate.cpp:13`
  - `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_grouped_topk/device/kernels/compute/moe_gate_common_compute.hpp:24`

  Neither binds it as a kernel *source*; both `#include` it, so this is a function-call coupling in
  the lent direction. Its public functions (`process_and_sort_tiles`, `process_tile_pair`,
  `process_tiles`, `process_iteration`, `transpose_and_pack`) take CB indices as `uint32_t`
  parameters, which is the ✓ **OK** donor shape: a `dfb::name` token carries a constexpr cast to
  `uint32_t` in both runtime and template-parameter position. So the port can pass its named DFB
  tokens into the existing signatures **without editing the header**, and the two consumer ops keep
  compiling. No fork is required, and none should be created. The two consumers are a coordination
  and sunset list, not authorization to convert the header in place.

  Worth noting for planning: both consumers are themselves gated today
  (`experimental/reduction/deepseek_grouped_gate` and
  `experimental/deepseek_prefill/moe_grouped_topk` are `legacy device-op` concept rows on the
  readiness sheet with `Is able to port? = no`), so a bundled conversion of the header is not
  available even if it were desirable.

- `device/kernels/dataflow/topk_dataflow_common.hpp` is included only by this op's own two readers, so
  it carries no cross-op coupling.

### Relaxation candidates

None. There is no custom hash to mine, and the sheet proposes `none` for both factories.

### TTNN factory analysis

- **Current concept:** `descriptor` for both factories. Each defines
  `static tt::tt_metal::ProgramDescriptor create_descriptor(...)`
  (`device/topk_device_operation.hpp:25-36`) and returns a populated `ProgramDescriptor`.
- **Op-owned tensors:** none. No `create_workload_descriptor`, no `WorkloadDescriptor`, so no
  `buffers` vector. The op allocates its outputs through `create_output_tensors`
  (`device/topk_device_operation.cpp:293-307`), which is ordinary output allocation, not op-owned
  tensors.
- **MeshWorkload need:** none. Single program per dispatch; the sheet records
  `Execution Model = SPMD`.
- **Custom hash:** absent, so the framework's default reflection hash over `TopkParams` applies. Note
  the dead-but-hashed attribute under Misc anomalies.
- **`get_dynamic_runtime_args` / `override_runtime_arguments`:** both absent. The device op declares
  only `select_program_factory`, `validate_on_program_cache_miss`, `compute_output_specs`, and
  `create_output_tensors` (`device/topk_device_operation.hpp:40-43`).
- **Pybind `create_descriptor` / other risky pybind:** none. `topk_nanobind.cpp` registers the
  user-facing operation only.
- **Target concept:** `ProgramSpecFactoryConcept`, matching the sheet's `Porting Target` cell.
- **Runtime-arg delivery today:** the factories use `KernelDescriptor::emplace_runtime_args` with
  `MeshTensor` references, so tensor bases are already framework-registered bindings patched on cache
  hits, not raw values baked into an RTA vector. The Metal 2.0 typed binding supersedes this
  mechanism.

## Per-DeviceOperation attribution

Only one device operation shares this directory, and every gate verdict is identical across its two
factories, so no split attribution is needed. Where findings differ *between factories* (CB endpoint
actions, binding sites, the cross-core address assumption, which is multi-core only) the tables above
name the factory.

## Misc anomalies  *(team-only, non-gating)*

1. **The `sorted` attribute is dead in every path, yet it is part of the program hash.**
   `TopkParams::sorted` (`device/topk_device_operation_types.hpp`) is never consumed. The single-core
   factory does not pass it to any kernel (`topk_single_core_program_factory.cpp:230-244` omits it),
   and the multi-core factory passes it as CTA 13 to both compute kernels
   (`topk_multi_core_program_factory.cpp:451`, `:481`), where it is declared and never read
   (`topk_local.cpp:108`, `topk_final.cpp:60`). Because the op has no custom hash, two calls that
   differ only in `sorted` compile and cache two separate programs with identical behaviour. Either
   the flag should drive something or it should stop reaching the factory.
2. **The single-core path silently ignores a caller-supplied `indices` tensor.**
   `topk_single_core_program_factory.cpp:198-200` hardcodes `GENERATE_INDICES` to `"1"` with the
   commented-out intent `tensor_args.indices.has_value() ? "0" : "1"` and a reference to GH issue
   36329. Consequences: the reader always generates indices; the RTA at index 3 (the indices base
   address) is dead, since `reader_create_index_tensor.cpp:32` reads it only under
   `#if not GENERATE_INDICES`; and the `TensorAccessorArgs` appended for the indices tensor at
   `topk_single_core_program_factory.cpp:196` are dead CTAs. The multi-core factory does honour the
   flag (`topk_multi_core_program_factory.cpp:348`), so the same call behaves differently depending on
   which factory is selected.
3. **Dead CTAs.** `Ht` is declared and never used in `reader_create_index_tensor.cpp:23` (CTA 2) and
   `writer_binary_interleaved.cpp:20` (CTA 2). Both kernels drive their loops from `work_per_core` and
   `total_number_of_cores` instead.
4. **`uint16_output` boundary is computed two ways.** `compute_output_specs` uses `<=`
   (`device/topk_device_operation.cpp:279`) while `select_program_factory`
   (`device/topk_device_operation.cpp:70`) and the single-core factory
   (`topk_single_core_program_factory.cpp:31`) use `<`. At a padded last dimension of exactly 65535
   they disagree: the index output tensor would be `UINT16` while the factory would set
   `fp32_dest_acc_en` and tell the reader to generate 32-bit index tiles into a `UInt16`-format CB.
   Unreachable in practice, because a tile-layout padded last dimension is always a multiple of 32 and
   65535 is not, but the two expressions should still be made to agree.
5. **`total_size` and `page_size` are derived from different dtypes on the multi-core input CB.**
   `topk_multi_core_program_factory.cpp:171-179` sizes `c_0` as `cb_in_units * value_tile_size` while
   its page size is `input_tile_size`. The two are equal today because the values output dtype always
   equals the input dtype (`device/topk_device_operation.cpp:282-283`), so the CB is correctly sized;
   the expression would silently mis-size the buffer if that ever stopped holding. Compare the
   single-core factory, which uses `input_tile_size` for both (`:95-103`).
6. **Stale API name in a comment.** `writer_local_topk.cpp:71` explains the barrier in terms of
   `cb_pop_front`, but the code calls the Device 2.0 `pop_front` method.
7. **Pack format reconfigured for the wrong buffer, harmlessly.** `topk_local.cpp:213` calls
   `pack_reconfig_data_format(index_transposed_dfb_index)` immediately before packing into
   `output_ind_dfb_index` (`:223`). Correct only because both CBs carry `index_cb_data_format`. The
   parallel value path two blocks earlier reconfigures for its actual destination
   (`pack_reconfig_data_format(values_dfb_index)`, `:189`), which is the pattern the index path
   should follow.

## Questions for the user

1. **Zero-endpoint CB instances on the multi-core final core.** `c_0` (`input_cb`) and `c_1`
   (`index_cb`) are declared over `all_cores_range_set`
   (`topk_multi_core_program_factory.cpp:171-191`) but no final-core kernel references either index,
   while both are fully live on every local core. Can a Metal 2.0 `DataflowBufferSpec` keep a declared
   core range that includes nodes where no kernel binds it, or must the porter narrow these two to
   `local_cores_range_set`? Narrowing changes the final core's SRAM layout, which the factory
   deliberately arranged (`:158-168`); keeping the range means the spec covers nodes with no binding.
   Neither reading is a dead-CB drop, so this is not a gate, but the porter needs the answer before
   writing the specs.
2. **Does Metal 2.0 guarantee one common address for a DFB across its whole declared core range?**
   `writer_local_topk.cpp:45-50` reads the write pointer of the local core's own `c_4` / `c_5`
   instance and uses it as the destination address on the final core (`:69`, `:89`). The legacy
   allocator makes this valid by assigning a CB one address across its core ranges, which
   `topk_multi_core_program_factory.cpp:158-168` documents and orders its allocations around. If the
   Metal 2.0 DFB allocator does not offer the same guarantee, this cross-core transfer needs a
   different way to learn the remote address, and the failure mode is silent mis-addressing rather
   than a validator error.

## Recipe notes

1. **Readiness-sheet column name drift.** `analyses/ttnn_op_porting_readiness.md` and
   `audit/metal2_audit.md` both name the column
   `Override runtime args method? (PD and legacy)`, but the live sheet's header reads
   `Override runtime args method? (PD only)`. The doc states that existing column names never change,
   so either the sheet was renamed or the docs anticipated a name it never had. No effect here (the
   cell is `no` for both rows), but a future auditor matching the header text literally would not find
   it.
2. **A sheet column is missing from the gate derivation.** The live sheet carries
   `Backdoor custom hash (attribute_values / to_hash)`, which the `Is able to port?` derivation in
   `audit/metal2_audit.md` does not list among its conjuncts. Both topk rows are `no`, so the verdict
   is unaffected, but if that column feeds the verdict the formula should say so, and if it does not,
   a note explaining why would save the next auditor the same detour.
3. **The CB endpoint table has no row for "live on one class of node, unreferenced on another."** The
   subject counts endpoints per CB per node and maps 0 endpoints to "dead CB, porter drops it." This
   op hits a case that reading does not cover: `c_0` / `c_1` are declared over a range spanning two
   kinds of core, with real producers and consumers on one kind and no reference at all on the other.
   Dropping the CB would be plainly wrong (it is used), and the real decision is about the DFB's
   declared *range*, not about the buffer's existence. The dead-CB text warns hard against
   over-calling, which is what steered this to a question rather than a drop, but an explicit row or
   sentence for the per-node-partial case would make that conclusion direct instead of inferred.
4. **The hidden-second-writer face assumes co-resident kernels.** Face (a) is described as a second
   kernel co-filling a CB by raw write, coordinated by semaphores, and the examples are all
   same-node. This op has the same pattern across nodes: the raw filler runs on a local core, the CB
   instance it fills lives on the final core, and the FIFO bookkeeping is done by a third kernel on
   the receiving node. The per-node counting rule gives the right answer, but working out that a
   remote NoC write is not an endpoint on the receiving node took a deliberate detour. One sentence
   stating that only local accesses enter a node's tally, and that a remote writer counts on the node
   whose instance it binds to obtain the address, would settle it quickly.
5. **Naming inference can be actively wrong, not merely unhelpful.** The RTA varargs subject tells
   the porter to infer each named arg from the kernel variable a `get_arg_val` unpacks into, and the
   same instinct applies to naming DFBs from kernel-side CTA variable names. In this op that instinct
   produces crossed bindings: `reader_final_topk.cpp` and `writer_local_topk.cpp` name CTAs 9 and 10
   `final_values_dfb_index` / `final_indices_dfb_index`, but the factory passes the *gathered* buffers
   (`c_4` / `c_5`) there, and the factory's own `final_*` CBs are `c_6` / `c_7`. A caution that the
   factory is the authority when factory and kernel vocabularies disagree would be worth adding
   wherever name inference is prescribed.
