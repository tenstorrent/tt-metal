# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/full`

One device operation, three program factories:

- **`FullDeviceOperation`** ([device/full_device_operation.hpp:20](device/full_device_operation.hpp#L20))
  - `FullInterleavedProgramFactory` ([device/full_program_factory_interleaved.cpp:17](device/full_program_factory_interleaved.cpp#L17)) — binds `writer_full.cpp` **twice** over the same core range (a Writer-config instance and a Reader-config instance)
  - `FullShardedProgramFactory` ([device/full_program_factory_sharded.cpp:20](device/full_program_factory_sharded.cpp#L20)) — binds `writer_full_sharded.cpp`
  - `FullNDShardedProgramFactory` ([device/full_program_factory_nd_sharded.cpp:20](device/full_program_factory_nd_sharded.cpp#L20)) — binds `writer_full_nd_sharded.cpp`

Factory selection is on the output's memory config ([device/full_device_operation.cpp:13-22](device/full_device_operation.cpp#L13-L22)): sharded with an explicit `shard_spec` picks the sharded factory, sharded without one picks the ND-sharded factory, interleaved picks the interleaved factory.

Kernels in scope (all three are op-owned, none borrowed, none unreferenced):

- [device/kernels/writer_full.cpp](device/kernels/writer_full.cpp)
- [device/kernels/writer_full_sharded.cpp](device/kernels/writer_full_sharded.cpp)
- [device/kernels/writer_full_nd_sharded.cpp](device/kernels/writer_full_nd_sharded.cpp)
- [device/kernels/full_kernel_common.hpp](device/kernels/full_kernel_common.hpp) — shared header, included by all three

Only one `DeviceOperation` lives in this directory, so no bundling and no per-DeviceOperation split is needed. There are no unreferenced kernel files. No copy of this op exists under `ttnn/cpp/ttnn/operations/experimental/quasar/`, so no out-of-bounds material was in reach.

Notable structural fact: **the op takes no input tensors.** `tensor_args_t` is an empty struct ([device/full_device_operation_types.hpp:22](device/full_device_operation_types.hpp#L22)) and the op is invoked with `tensor_args_t{}` ([device/full_device_operation.cpp:90](device/full_device_operation.cpp#L90)). The output tensor it creates is its only tensor, so there is exactly one tensor binding to reason about in every factory.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `4d6e9518bf5 2026-07-30 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/full` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `FullDeviceOperation` → `FullInterleavedProgramFactory`, `FullShardedProgramFactory`, `FullNDShardedProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes.** All three kernels are structurally Device 2.0 (`Noc`, `CircularBuffer`, `Semaphore`-free, `TensorAccessor`); no CB-index free-function holdovers |
| *Prereqs* — Cross-op escapes | **Ok.** Every kernel `#include` resolves to `tt_metal/hw/inc/api/*` (donor class 1, no concern) or to the op's own in-directory header. No borrowed kernel files, no lent kernel files |
| *Feature Support* — overall | **GREEN** (all four Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | Ok. Every `get_compile_time_arg_val` index is a literal constant |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes**, all three factory rows |
| *TTNN Readiness* — Concept (current) | `descriptor` (all three factories) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A (concept is `descriptor`) |
| *TTNN Readiness* — Is safe to port? | Yes (all three rows) |
| *TTNN Readiness* — Custom hash | No. No `compute_program_hash` anywhere in the op |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No. Hook absent from the device operation |
| *TTNN Readiness* — `override_runtime_arguments` | No. Method absent |
| *TTNN Readiness* — Pybind `create_descriptor` | No. [full_nanobind.cpp](full_nanobind.cpp) binds only the `moreh_full` free function |
| *TTNN Readiness* — Op-owned tensors | No (impossible on a `descriptor` concept; sheet cell is blank) |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (no op-owned tensors) |
| *Port work* — Offset base pointer | **none.** No `->address()` call exists in the op; the output buffer travels as a `Buffer*` and every page offset is already a separate scalar arg |
| *Port work* — Tensor bindings (per binding) | `output` → **Case 1** (`TensorAccessor`) in all three factories |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none. All three accessor constructions are 2-argument |
| *Port work* — CB endpoints | **self-loop** on every CB, in every config (five `(CB, config)` pairs) |

**CB endpoints** are dispositions, not gates: every out-of-window CB has a port-time resolution. For this op every CB has exactly one toucher, so every one takes the self-loop.

## Result

**GREEN → brief issued.** All five gate-bearing subjects cleared: Device 2.0, Feature compatibility, TTNN factory concept, Offset base pointers, TensorAccessor 3rd argument. No code path is blocked, so no subset scoping applies.

The port work is small and mechanical: one tensor binding per factory (all Case 1), a self-loop on each of the five `(CB, config)` pairs, and no relaxation, no page-size override, no offset split, no dead CB. The porter brief is at `METAL2_PORT_BRIEF.md` in this directory.

## Gate detail

- **TTNN factory concept (`Is able to port?`): GREEN.** The readiness sheet ("Operations analysis", fetched fresh this run) carries exactly three rows for `full`, one per factory. All three read: `Concept` = `descriptor`, `Custom hash` = `no`, `Runtime-args update (get_dynamic_runtime_args)` = `no`, `Override runtime args method?` = `no`, `Pybind descriptor` = `no`, `Smuggled pointer` = `no`, `Is safe to port?` = `yes`, **`Is able to port?` = `yes`**, `TensorParameter relaxation` = `none`. The sheet's `Porting Target` column independently states `ProgramSpecFactoryConcept`, and `Op Classification` reads `PD Op (pointer-patching)`, matching the `Buffer*`-binding delivery described under Port work below.

  Cross-check against the code, clean on every cheaply-checkable column:

  | Column | Sheet | Code evidence |
  |---|---|---|
  | `Concept` | `descriptor` | All three factories declare `static ProgramDescriptor create_descriptor(...)` ([interleaved](device/full_program_factory_interleaved.hpp#L13), [sharded](device/full_program_factory_sharded.hpp#L13), [nd_sharded](device/full_program_factory_nd_sharded.hpp#L13)). No `create_workload_descriptor`, no mesh-workload return, no `create()` + `override_runtime_arguments()` pair |
  | `Custom hash` | `no` | `grep -rn compute_program_hash` over the op directory: zero hits |
  | `get_dynamic_runtime_args` | `no` | Zero hits. `FullDeviceOperation` ([device/full_device_operation.hpp:20-32](device/full_device_operation.hpp#L20-L32)) declares only `select_program_factory`, `validate_inputs`, `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors` |
  | `Override runtime args method?` | `no` | Zero hits for `override_runtime_arguments` |
  | `Pybind descriptor` | `no` | [full_nanobind.cpp:43-53](full_nanobind.cpp#L43-L53) binds `ttnn::moreh_full` via `ttnn::bind_function`. No `nb::class_` of the device operation, no `create_descriptor` binding |
  | `Op-owned tensors?` | (blank) | Consistent with `descriptor` (which cannot carry them). No `WorkloadDescriptor`, no `buffers` vector |
  | Factory-set match | 3 rows | 3 factories in the code, one-to-one. No phantom row, no missing row |

  Cross-column invariants hold: `get_dynamic_runtime_args` is `no` on a `descriptor` concept (legal), and no op-owned tensors on a `descriptor` row (legal). Per the recipe, `Is safe to port?` was **not** re-derived; that axis is the sheet owner's judgment and it reads `yes`.

- **Device 2.0 (every kernel used): GREEN.** All three kernels are written against the Device 2.0 data-movement surface throughout. No violations to route. Per-kernel evidence:

  | Kernel | Device 2.0 idioms in use |
  |---|---|
  | [writer_full.cpp](device/kernels/writer_full.cpp) | `Noc noc;` ([:26](device/kernels/writer_full.cpp#L26)) · `CircularBuffer cb(cb_value);` ([:27](device/kernels/writer_full.cpp#L27)) · `cb.reserve_back` / `cb.get_write_ptr()` / `cb.push_back` / `cb.wait_front` / `cb.pop_front` ([:29-:67](device/kernels/writer_full.cpp#L29-L67)) · `noc.async_write(cb, s, ...)` ([:64](device/kernels/writer_full.cpp#L64)) · `noc.async_writes_flushed()` / `noc.async_write_barrier()` · `TensorAccessor(dst_args, output_addr)` ([:58](device/kernels/writer_full.cpp#L58)) |
  | [writer_full_sharded.cpp](device/kernels/writer_full_sharded.cpp) | Same set; `noc.async_write(cb, dst_accessor, ...)` at [:67](device/kernels/writer_full_sharded.cpp#L67) |
  | [writer_full_nd_sharded.cpp](device/kernels/writer_full_nd_sharded.cpp) | Same set, plus the Device 2.0 accessor iterator `dst_accessor.shard_pages(shard_id)` ([:64-:68](device/kernels/writer_full_nd_sharded.cpp#L64-L68)) |
  | [full_kernel_common.hpp](device/kernels/full_kernel_common.hpp) | `zero_buffer` uses `Noc` + `CircularBuffer` + `noc.async_write_zeros(cb, bytes)` + `noc.write_zeros_l1_barrier()` ([:15-:20](device/kernels/full_kernel_common.hpp#L15-L20)), which is the pattern the Device 2.0 migration guide documents under *Zeroing Memory* |

  Explicitly checked and **absent** from all three kernels: `noc_async_read` / `noc_async_write` free functions, `InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedAddrGenFast`, `InterleavedPow2AddrGen*`, raw semaphore addresses, `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front` free functions, `get_read_ptr(cb_id)` / `get_write_ptr(cb_id)` free functions, `get_local_cb_interface`. Every `get_write_ptr` in the op is the wrapper method `cb.get_write_ptr()`, not the CB-index free function.

  One shape worth naming so a reader does not mistake it for a holdover: the op's own kernel helper `zero_buffer(uint32_t cb_id, uint32_t bytes)` ([full_kernel_common.hpp:15](device/kernels/full_kernel_common.hpp#L15)) takes a CB index and constructs a `CircularBuffer` from it internally, and is called at sites where a `CircularBuffer` wrapper is already in scope ([writer_full.cpp:34](device/kernels/writer_full.cpp#L34) and the two siblings). It has the *outward form* the holdover rule keys on, but it is an **op-owned function, not a Device 2.0 API free function**, and there is no wrapper-method replacement for it. It is therefore not a Device 2.0 holdover. It is a small port item and is carried into the brief. (See Recipe notes.)

- **Feature compatibility:** every Appendix A entry, in order. All four are absent, so all four are `N/A`.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `global_circular_buffer.hpp` include. The three `CBDescriptor` literals ([interleaved :38 and :69](device/full_program_factory_interleaved.cpp#L38), [sharded :41](device/full_program_factory_sharded.cpp#L41), [nd_sharded :41](device/full_program_factory_nd_sharded.cpp#L41)) set only `total_size`, `core_ranges`, and `format_descriptors`; the `global_circular_buffer` field is never named. No `remote_index`, no `remote_cb_*` identifier, no `remote_circular_buffer.h` |
  | CBDescriptor `address_offset` (non-zero) | N/A | The token `address_offset` does not appear anywhere in the op. No `set_address_offset`, no 4-argument `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor` call. All three `CBDescriptor` literals leave the field at its default zero |
  | GlobalSemaphore | N/A | The op uses no semaphores of any kind. `grep -rn 'semaphore\|Semaphore'` over the whole op directory: zero hits |
  | Variable-count compile-time arguments (CTA varargs) | N/A | Op-level cue absent: `tensor_args_t` is an *empty* struct ([device/full_device_operation_types.hpp:22](device/full_device_operation_types.hpp#L22)), so there is no variable-count tensor container. Kernel-level decider absent: every `get_compile_time_arg_val` call in all three kernels uses a literal index (0, 1, 2, 4, 5), and each `TensorAccessorArgs<N>` uses a literal base (`<3>`, `<5>`, `<6>`). No CTA is read at a runtime-varying index |

- **CB endpoints (GATE-free): every CB is single-ended, so every one takes a self-loop.** The endpoint count is one toucher per CB in every config. Each toucher is simultaneously the FIFO producer (`reserve_back` / `push_back`) and the FIFO consumer (`wait_front` / `pop_front`) of that CB, inside one kernel instance, which is the textbook self-loop.

  | CB | Config | Touchers | Verdict | Resolution |
  |---|---|---|---|---|
  | `c_0` (fill-value page) | interleaved, reader present (`num_pages > num_cores`) | 1 — the Writer-config instance of `writer_full.cpp` | single-ended | self-loop |
  | `c_0` (fill-value page) | interleaved, no reader (`num_pages <= num_cores`) | 1 — same instance | single-ended | self-loop |
  | `c_1` (fill-value page, reader copy) | interleaved, reader present only | 1 — the Reader-config instance of `writer_full.cpp` | single-ended | self-loop |
  | `c_24` (fill-value page) | sharded | 1 — `writer_full_sharded.cpp` | single-ended | self-loop |
  | `c_24` (fill-value page) | ND-sharded | 1 — `writer_full_nd_sharded.cpp` | single-ended | self-loop |

  The interleaved factory **is** the dual-instance work-split pattern the recipe warns about: it pushes the same `kernel_source` into two `KernelDescriptor`s that differ only by `WriterConfigDescriptor` / `ReaderConfigDescriptor` and their per-instance work-split args, both over the same `all_cores` range ([device/full_program_factory_interleaved.cpp:55-88](device/full_program_factory_interleaved.cpp#L55-L88), page split at [:101-:111](device/full_program_factory_interleaved.cpp#L101-L111)). It nonetheless produces **no co-touched CB**, because each instance receives its *own* CB index through CTA 0: the Writer instance gets `c_0` ([:52](device/full_program_factory_interleaved.cpp#L52)) and the Reader instance gets `c_1` ([:79](device/full_program_factory_interleaved.cpp#L79)), with a separate `CBDescriptor` allocated for each ([:38](device/full_program_factory_interleaved.cpp#L38) and [:69](device/full_program_factory_interleaved.cpp#L69)). So no 1P+1C assignment is required and the multi-binding advanced option is not needed anywhere in this op.

  Hidden-second-writer scan (the face that hides from a FIFO trace): run over all three kernels and clean. There is no `get_local_cb_interface`, no `fifo_wr_ptr` / `fifo_rd_ptr` access, no `evil_set_write_ptr` / `evil_set_read_ptr`, and no semaphore anywhere in the op, so there is no coordinated raw co-fill to find. The only raw pointer access to a CB is `cb.get_write_ptr()` by that CB's own single FIFO producer ([writer_full.cpp:31](device/kernels/writer_full.cpp#L31) and siblings), which is a peek on a binding that kernel already holds and does not add a toucher. `zero_buffer` writes into the same CB but from within that same kernel instance, so it likewise does not add one.

  No dead CB: every allocated `buffer_index` is read by the bound kernel through CTA 0 and used to construct the `CircularBuffer`.

- **Offset base pointers: GREEN.** No fold exists, and none can: **the op contains no `->address()` call at all.** The output buffer is handed to the kernels as a `Buffer*` pushed into the runtime-arg list (`emplace_runtime_args(core, {output.buffer(), ...})` at [interleaved :104/:108/:110](device/full_program_factory_interleaved.cpp#L104), [sharded :89-90](device/full_program_factory_sharded.cpp#L89-L90), [nd_sharded :69](device/full_program_factory_nd_sharded.cpp#L69)), which the framework registers as a `BufferBinding` and patches per dispatch. Host arithmetic cannot be folded into a `Buffer*`.

  Every page offset the kernels need is passed as a **separate scalar arg** and applied on the device side, which is exactly the shape the ops team refactors offset-folding ops *into*:

  - interleaved: `reader_page_start` / `writer_page_start` are separate args ([:102-:108](device/full_program_factory_interleaved.cpp#L102-L108)), consumed as `start_id` and turned into a `page_id` in the kernel ([writer_full.cpp:62-65](device/kernels/writer_full.cpp#L62-L65))
  - sharded: `first_page_id` is a separate arg ([:79-90](device/full_program_factory_sharded.cpp#L79-L90)), walked as `curr_page_id` in the kernel ([writer_full_sharded.cpp:64-71](device/kernels/writer_full_sharded.cpp#L64-L71))
  - ND-sharded: `start_shard_id` is a separate arg ([:69](device/full_program_factory_nd_sharded.cpp#L69)), resolved through `dst_accessor.shard_pages(shard_id)` ([writer_full_nd_sharded.cpp:63-68](device/kernels/writer_full_nd_sharded.cpp#L63-L68))

  Type 3 (`address_offset`) is N/A per Appendix A above. Type 4 (`narrow`) does not apply. Cross-check against the dated triage `analyses/2026-07-19_offset_base_pointers.md`: `full` appears in none of its tables, matching this scan. This is a "no fold, op not in the tables" outcome, so the single tensor binding drops through to ordinary TensorParameter analysis.

- **TensorAccessor 3rd argument: GREEN, no site.** All three accessor constructions pass exactly two arguments: `TensorAccessor(dst_args, output_addr)` at [writer_full.cpp:58](device/kernels/writer_full.cpp#L58), [writer_full_sharded.cpp:60](device/kernels/writer_full_sharded.cpp#L60), [writer_full_nd_sharded.cpp:59](device/kernels/writer_full_nd_sharded.cpp#L59). No explicit page-size override anywhere, so no site to classify. The kernels *query* the page size off the accessor (`get_aligned_page_size()`), which is the auto-supplied value and the direction the port wants anyway. Cross-check against the dated triage `analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md`: `full` appears in none of its tables, matching this scan.

  One related observation, in the vicinity of this subject but *not* a 3rd-arg finding: the host passes an `aligned_page_size` CTA that the sharded and ND-sharded kernels never read. Recorded under Misc anomalies.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding): `output` — **Case 1** (`TensorAccessor`), identically in all three factories. Delivered today by the `Buffer*`-binding form (`emplace_runtime_args(core, {output.buffer(), ...})`), so the base arrives as a raw `uint32_t` at RTA index 0 and is fed straight into a `TensorAccessor` constructor. This is *not* the silent-wrong `->address()`-on-an-RTA hazard: the framework patches `BufferBinding`s on cache hits, and the sheet's `Smuggled pointer` column agrees (`no`). It is still a pointer argument to convert. Under Metal 2.0 the `Buffer*` RTA, the host-side `TensorAccessorArgs(output.buffer()).append_to(...)` CTA plumbing ([interleaved :53](device/full_program_factory_interleaved.cpp#L53) and [:80](device/full_program_factory_interleaved.cpp#L80), [sharded :57](device/full_program_factory_sharded.cpp#L57), [nd_sharded :57](device/full_program_factory_nd_sharded.cpp#L57)), and the kernel-side `TensorAccessorArgs<N>` / `output_addr` unpack all disappear in favour of a `TensorParameter` and `TensorAccessor(tensor::name)`.
- **TensorParameter relaxation:** none. The sheet says `none` on all three rows, and this is consistent: the op has no custom hash, and a relaxation is expressed as one.
- **TensorAccessor 3rd arg:** none. No override to drop.
- **CB endpoints:** self-loop on all five `(CB, config)` pairs, per the table above. No 1P+1C assignment, no multi-binding advanced option, no dead-CB drop.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. The interleaved factory looks like the dual-instance work-split pattern but gives each instance its own CB, so no CB in this op has more than one toucher. There is no hidden second writer and no semaphore in the op.
- **Cross-op / shared kernels:** none in either direction. The op owns all three of its kernel sources and no other op instantiates them (`grep -rn writer_full` across the tree returns hits only inside this op). No `_metal2` fork exists beside any of them, so if a fork were ever needed this port would be the one to create it. No fork is needed here, because nothing outside the op binds these files and no *other factory of this op* binds the same file either: each factory has its own kernel, and the interleaved factory's two instances of `writer_full.cpp` both convert in the same change.
- **Shared in-directory header:** [device/kernels/full_kernel_common.hpp](device/kernels/full_kernel_common.hpp) is included by all three kernels and holds `zero_buffer(uint32_t cb_id, uint32_t bytes)` ([:15](device/kernels/full_kernel_common.hpp#L15)), the `value` union, and the `onepage` constant. Its `uint32_t cb_id` parameter is the "✓ OK" donor form, so a `dfb::` handle passes through it unchanged via the constexpr cast and no signature change is forced. If the porter does change it, all three consumers are in this port's scope and convert together.
- **RTA varargs:** none. Every runtime arg in all three kernels is read at a distinct literal index (`writer_full.cpp` 0-3, `writer_full_sharded.cpp` 0-4, `writer_full_nd_sharded.cpp` 0-2). No counted loop over runtime args, no running `arg_index++`, no data-selected index. Every one is nameable.

## Team-only

- **Out-of-directory coupling & donor shape — op-level roll-up: ✓ clean.** No function-call escape and no file-path escape.
  - *Function-call escape:* every `#include` in the op's kernels resolves either to the framework surface or to the op's own directory. `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, and `api/tensor/noc_traits.h` all resolve under `tt_metal/hw/inc/api/`, which is donor class 1 (LLK / HAL / firmware, no concern, never forked, out of porter scope). `full_kernel_common.hpp` is in-directory. There is no `ttnn/cpp/ttnn/kernel_lib/` include, no `ttnn/cpp/ttnn/kernel/` include, no `kernel_helper_functions/` include, no in-family shared kernel, and no cross-family donor. The summary table and per-call detail are omitted because every roll is ✓.
  - *Borrowed kernel files (file-path instantiation):* none. All three `KernelDescriptor::kernel_source` paths point inside this op's own `device/kernels/`.
  - *Lent kernel files:* none. No factory outside this op instantiates `writer_full.cpp`, `writer_full_sharded.cpp`, or `writer_full_nd_sharded.cpp`.
  - *Host-side out-of-directory use, for completeness (not a kernel escape):* the two sharded factories call `get_optimal_worker_cores_for_sharded_tensor` from `ttnn/api/ttnn/tensor/tensor_utils.hpp` ([sharded :31](device/full_program_factory_sharded.cpp#L31), [nd_sharded :32](device/full_program_factory_nd_sharded.cpp#L32)), a widely-used host helper (`data_movement/pad`, `data_movement/untilize`, and others call it too). It returns a core list, touches no kernel handle, and has no bearing on any gate.
- **Relaxation candidates (mined from a custom hash):** not applicable. The op has no custom hash, so there is nothing to mine and no candidate to record.
- **TTNN factory analysis:** the sheet-derived facts with code evidence are in the Gate detail cross-check table above. Summarizing the non-gating half: **op-owned tensors — none** (the concept is `descriptor`, which cannot carry them; there is no `WorkloadDescriptor` and no `buffers` vector anywhere in the op). **MeshWorkload need — none**; all three factories return a plain `ProgramDescriptor`, and the sheet's `Execution Model` column reads `SPMD`. **Pybind `create_descriptor` — absent**; [full_nanobind.cpp](full_nanobind.cpp) exposes only the `moreh_full` free function, with no `nb::class_` of the device operation and no other migration-risky binding. **Custom hash, `get_dynamic_runtime_args`, `override_runtime_arguments` — all absent** (zero grep hits each). **Target concept: `ProgramSpecFactoryConcept`**, which the sheet's `Porting Target` column states directly for all three rows.

## Misc anomalies  *(team-only, non-gating, not porter work)*

1. **Dead CTA in the sharded factory.** [device/full_program_factory_sharded.cpp:55-56](device/full_program_factory_sharded.cpp#L55-L56) pushes `aligned_page_size` as compile-time arg index 3, but [writer_full_sharded.cpp](device/kernels/writer_full_sharded.cpp) reads indices 0, 1, 2, and 4 and then starts its accessor args at `TensorAccessorArgs<5>()` ([:19-23](device/kernels/writer_full_sharded.cpp#L19-L23)). Index 3 is never read. Harmless today because the accessor base index still matches, but it is a live trap: anyone renumbering these CTAs has to notice the gap.
2. **Same dead CTA in the ND-sharded factory.** [device/full_program_factory_nd_sharded.cpp:55-56](device/full_program_factory_nd_sharded.cpp#L55-L56) pushes `aligned_page_size` at index 3; [writer_full_nd_sharded.cpp](device/kernels/writer_full_nd_sharded.cpp) reads 0, 1, 2, 4, 5 and starts at `TensorAccessorArgs<6>()` ([:17-22](device/kernels/writer_full_nd_sharded.cpp#L17-L22)). Index 3 is never read.
3. **`elems_per_page` is computed two different ways across factories.** Interleaved uses `page_size / output.element_size()` ([:31](device/full_program_factory_interleaved.cpp#L31)); both sharded factories use `page_size / datum_size(data_format)` ([sharded :54](device/full_program_factory_sharded.cpp#L54), [nd_sharded :54](device/full_program_factory_nd_sharded.cpp#L54)). The two agree for the three dtypes `validate_inputs` allows (BFLOAT16, INT32, FLOAT32, per [device/full_device_operation.cpp:26-30](device/full_device_operation.cpp#L26-L30)), so there is no bug today. They would diverge for a block-float dtype, where `element_size()` and `datum_size()` differ, so this is a hazard if the dtype allowlist is ever widened.
4. **The CB is sized `page_size` but the kernels source `aligned_page_size` bytes out of it.** All three `CBDescriptor` literals set `total_size = page_size` (the *unaligned* logical page), while the NoC write passes `get_aligned_page_size()` as the transfer size ([writer_full.cpp:64](device/kernels/writer_full.cpp#L64), [writer_full_sharded.cpp:67](device/kernels/writer_full_sharded.cpp#L67), [writer_full_nd_sharded.cpp:67](device/kernels/writer_full_nd_sharded.cpp#L67)); the zero path likewise zeros only `page_size` bytes ([writer_full.cpp:34](device/kernels/writer_full.cpp#L34)). For TILE layout the two sizes match, so nothing is amiss. For ROW_MAJOR layout with a narrow last dimension they diverge (the nightly test exercises exactly this, e.g. `[1, 3]` int32 row-major in `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_full.py`, where the logical page is 12 bytes and the aligned DRAM page is 32 or 64). My guess is that this is benign in practice: the excess bytes come from adjacent SRAM and land in the destination page's padding region, which `to_torch` discards, which is why the tests pass. Worth a look from the ops team regardless, since it means the fill kernel sources uninitialized SRAM past the end of its own CB allocation.
5. **The reader instance can be launched with zero work.** The interleaved factory turns the reader on globally when `num_pages > num_cores` ([:66](device/full_program_factory_interleaved.cpp#L66)), then gives each core `num_pages_per_core / 2` reader pages ([:103](device/full_program_factory_interleaved.cpp#L103)). A core in the smaller work group with a single page gets a reader instance with `num_pages_per_reader == 0`, which still reserves a CB page, fills it, and pushes it before writing nothing. Correct, just wasted work on those cores.
6. **`write_addr` is computed on a path that does not use it.** All three kernels call `cb.get_write_ptr()` unconditionally ([writer_full.cpp:31](device/kernels/writer_full.cpp#L31) and siblings) but only read it in the non-zero fill branch; the `val.u == 0` branch calls `zero_buffer`, which re-derives the pointer itself. Cosmetic.

## Questions for the user

None. Every gate resolved from the code without ambiguity, and no site needed a conservative default.

## Recipe notes

1. **Live sheet column name differs from the recipe's.** The recipe (and `analyses/ttnn_op_porting_readiness.md`) names the column `Override runtime args method? (PD and legacy)`. The live sheet header is `Override runtime args method?` / `(PD only)`. The readiness doc states that "existing column names never change", so this drift is worth reconciling, especially since the recipe leans on the name to distinguish this gate from the `Concept`-level legacy signature check. Same value either way here (`no`), so it changed no finding.
2. **`Pybind descriptor` header describes a different check than the recipe's cross-check.** The live header is `Pybind descriptor` / `(nb::class_ of device op)`, but the recipe's cross-check instruction is "grep the op's `*_nanobind.cpp` for a `create_descriptor` binding". Those are two different greps (a device-op class binding versus a `create_descriptor` method binding). Both come back absent for this op, so there was no conflict to escalate, but on an op where only one fires an auditor would not know which one the sheet's `yes` meant.
3. **Blank cells versus `no`.** For all three `full` rows, `Op-owned tensors?` and `Secretly SPMD Workload?` are **blank**, not `no`. The recipe's `Is able to port?` derivation and the status-summary template treat these as yes/no values, and the spreadsheet-broken routing is strict, so a literal reading could gate on "blank is not `no`". I read blank as "not applicable on a `descriptor` concept" (op-owned tensors are structurally impossible there, and the SPMD question only arises for `WorkloadDescriptor`), which is clearly what the sheet intends. An explicit rule for blank cells on the not-applicable columns would remove the judgment call.
4. **The sheet now answers the TTNN porting shape subject directly.** A `Porting Target` column exists (reading `ProgramSpecFactoryConcept` for all three rows), alongside new `Op Classification`, `Execution Model`, `Known op issues`, and `Pointer patching perf issue?` columns. The *TTNN porting shape* subject asks the auditor to derive the target from `Concept` plus `Op-owned tensors?`; that derivation is still the right one to run, but `Porting Target` is now a free cross-check on it and the recipe could say so.
5. **The Device 2.0 holdover rule has no verdict for an op-local helper with the holdover form.** `full`'s own `zero_buffer(uint32_t cb_id, uint32_t bytes)` ([device/kernels/full_kernel_common.hpp:15](device/kernels/full_kernel_common.hpp#L15)) matches the *cue* the RED bullet describes (a free function taking a `uint32_t` CB index, called where a `CircularBuffer` wrapper is already in scope), but neither bullet covers it: the Green bullet enumerates *sanctioned Device 2.0 API* free functions (`get_tile_size`, `get_local_cb_interface`), and the RED bullet's definition requires "a wrapper-method replacement exists", which for an op-authored helper it never does. I resolved it as not a holdover. A sentence scoping the holdover rule to the Device 2.0 API surface, and pointing op-local `uint32_t cb_id` helpers at the coupling table's "✓ OK" row instead, would make that explicit.
6. **A dead CTA is team-only by the routing table but lands in the porter's lap anyway.** *Incidental anomalies* classifies a dead compile-time arg as FYI-U, "not porter-actionable", so anomalies 1 and 2 above stay out of the brief. But a dead *CTA* is unlike a dead RTA or a suspicious constant: the port converts positional CTAs into a named `compile_time_args` schema, so the porter must decide, at the keyboard, whether to carry a name for an arg no kernel reads. That decision is made harder, not easier, by the finding being withheld. The recipe already carves out an analogous case for dead CBs (the porter drops the allocation and any dead CTA carrying its index); a dead non-CB-index CTA seems to want either the same treatment or an explicit line saying "carry it forward unread, do not clean it up."
