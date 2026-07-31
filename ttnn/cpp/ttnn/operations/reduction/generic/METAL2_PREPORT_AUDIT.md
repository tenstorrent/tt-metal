# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/reduction/generic/`

Two `DeviceOperation`s share this directory. They share the op's `common.hpp` / `common.cpp` helpers, the
`reduce_op_utils::get_defines` dispatcher, one dataflow reader (`reader_unary_reduce_universal_start_id.cpp`),
one more reader (`reader_unary_transpose_wh_universal_input_cols_partitioned.cpp`), and the borrowed
`writer_unary_interleaved_start_id.cpp` — so they are audited together as one porting unit, with
per-DeviceOperation attribution below where findings differ.

- **`ReduceDeviceOperation`** (`device/reduce_op_device_operation.hpp`)
  - `ReduceSingleCoreHwProgramFactory` (`device/reduce_op_single_core_hw_program_factory.cpp`)
  - `ReduceMultiCoreHProgramFactory` (`device/reduce_op_multi_core_h_program_factory.cpp`)
  - `ReduceMultiCoreWProgramFactory` (`device/reduce_op_multi_core_w_program_factory.cpp`)
- **`WelfordReduceDeviceOperation`** (`device/welford_reduce_device_operation.hpp`)
  - `WelfordReduceProgramFactory` (`device/welford_reduce_program_factory.cpp`)

Kernels audited (all referenced by a `KernelDescriptor::kernel_source` in one of the four factories):

| Kind | File | Owner |
|---|---|---|
| dataflow | `device/kernels/dataflow/reader_unary_reduce_rm.cpp` | this op |
| dataflow | `device/kernels/dataflow/reader_unary_reduce_universal_start_id.cpp` | this op |
| dataflow | `device/kernels/dataflow/reader_unary_transpose_wh_universal_input_cols_partitioned.cpp` | this op |
| dataflow | `device/kernels/dataflow/reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp` | this op |
| dataflow | `device/kernels/dataflow/writer_reduce_rm_scalar.cpp` | this op |
| dataflow | `device/kernels/dataflow/writer_welford_hw.cpp` | this op |
| dataflow (hdr) | `device/kernels/dataflow/reduce_rm_dataflow_common.hpp` | this op (included by the two RM kernels only) |
| compute | `device/kernels/compute/reduce.cpp` | this op |
| compute | `device/kernels/compute/reduce_h_neg.cpp` · `reduce_w_neg.cpp` · `reduce_hw_neg.cpp` | this op |
| compute | `device/kernels/compute/reduce_rm.cpp` | this op |
| compute | `device/kernels/compute/welford_reduce_h.cpp` · `welford_reduce_w.cpp` · `welford_reduce_hw.cpp` | this op |
| **borrowed** dataflow | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `eltwise/unary` |
| **borrowed** dataflow | `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` | `data_movement/sharded` |

No unreferenced kernel files sit in the op's kernel directories — every file listed above is bound by a factory.

> **Note on `experimental/quasar/reduction/generic/`.** A whole-op quasar copy of this op exists at
> `ttnn/cpp/ttnn/operations/experimental/quasar/reduction/generic/`, with its own duplicated kernels and its own
> readiness-sheet rows. It is **out of bounds** — it is a deliberately-shortcut pre-port, not a template, and it
> binds its own copies of every kernel (verified: its factories' `kernel_source` strings all point inside
> `experimental/quasar/...`). It induces **no** coupling on the op audited here. Do not read it, and do not treat
> its `*_metal2`-style code as precedent.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `7e91046b794 2026-07-31 docs(metal_2.0): add the op-porting recipe set`
*(squashed import of `origin/akertesz/op-porting-recipe` onto `main`; the branch's own history carries the
per-change detail if a reviewer needs to pin a finer version.)*

**Readiness sheet:** fetched live this run via the Google Drive MCP (`Operations analysis`, file id
`1KUMj8SyBGlNMZlLFgs1MbAZlO2g6EoUc4KaxSlcy8jw`); four `reduction/generic` rows read by header name.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/reduction/generic/` |
| **Overall** | **GREEN** — all gates cleared, brief issued |
| **DOps / Factories** | `ReduceDeviceOperation` → `ReduceSingleCoreHwProgramFactory`, `ReduceMultiCoreHProgramFactory`, `ReduceMultiCoreWProgramFactory` · `WelfordReduceDeviceOperation` → `WelfordReduceProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 14 own kernels + both borrowed writers are structurally Device 2.0 (`Noc`, `DataflowBuffer` / `CircularBuffer` wrappers, `TensorAccessor`). No holdovers. One judgment call recorded under Gate detail (`get_dataformat(cb_id)`, ×2). |
| *Prereqs* — Cross-op escapes | Ok — two borrowed writer kernels, both Device 2.0; no `_metal2` fork exists for either (this port creates the first). Function-call escapes are all kernel-lib / framework. |
| *Feature Support* — overall | **GREEN** — all four Appendix A entries `N/A` |
| *Feature Support* — Variadic-CTA | Ok — every `get_compile_time_arg_val` index in every kernel is a compile-time constant; `tensor_args_t` is a single `Tensor` |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** — all four factory rows |
| *TTNN Readiness* — Concept (current) | `descriptor` (all four factories) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A — no `WorkloadDescriptor` factory |
| *TTNN Readiness* — Is safe to port? | Yes (all four rows); `Smuggled pointer` = `no` — confirmed by code, see Tensor bindings |
| *TTNN Readiness* — Custom hash | No — no `compute_program_hash` on either device-op (sheet agrees; `Formerly custom hashed? = yes` is historical) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No — hook absent from both device-ops |
| *TTNN Readiness* — `override_runtime_arguments` | No — absent from all four factories and both device-ops |
| *TTNN Readiness* — Pybind `create_descriptor` | No — `std_var_reductions_nanobind.cpp` binds only the user-facing `ttnn.std` / `ttnn.var` entry points |
| *TTNN Readiness* — Op-owned tensors | No — both device-ops return a plain `ProgramDescriptor`; no `buffers` vector |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (all four; matches the sheet's `Porting Target`) |
| *Port work* — Offset base pointer | none — no `->address()` reaches a runtime-arg context anywhere in the op |
| *Port work* — Tensor bindings (per binding) | Case 1 (`TensorAccessor`) on input + output in every factory/config **except** the width-sharded H config, where both are `clean` (borrowed-memory DFB) |
| *Port work* — TensorParameter relaxation | none (sheet: `none` on all four rows; no custom hash to reconcile against) |
| *Port work* — TensorAccessor 3rd arg | none — all five `TensorAccessor(...)` construction sites are two-argument |
| *Port work* — CB endpoints | legal / self-loop only — **no** multi-binding, **no** dead CB, across every `(CB, config)` |

**CB endpoints** are dispositions, not gates: every out-of-window CB here resolves to a **self-loop** (one
toucher). Nothing in this op needs the multi-binding advanced option, and nothing is dead. Full per-`(CB, config)`
census below.

## Result

**GREEN → brief issued.** All five gates clear: Device 2.0 ✓ · Feature compatibility ✓ · TTNN factory concept ✓ ·
Offset base pointers ✓ · TensorAccessor 3rd argument ✓. No factory is blocked; no subset scoping is needed.

The op is unusually well-positioned for this port: it is already fully on the `ProgramDescriptor` API with
per-factory `MeshTensor` bindings (no smuggled addresses), its kernels are already Device 2.0 and already speak
`DataflowBuffer` in most files, and it uses none of the Appendix A features. The substantive port work is the
mechanical CB→DFB / named-binding rewrite across 14 own kernels plus one forked borrowed writer, spread over
four factories with several config branches each — **volume, not difficulty**.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** All four rows read `yes`. Cross-check against the code
  is clean on every cheaply-checkable column:
  - `Concept = descriptor` — confirmed: each factory declares exactly
    `static tt::tt_metal::ProgramDescriptor create_descriptor(...)`
    (`reduce_op_device_operation.hpp:25,32,39`; `welford_reduce_device_operation.hpp:24`). No mesh-workload return.
  - `Custom hash = no` — confirmed: `grep -rn 'compute_program_hash\|attribute_values\|to_hash'` over the op
    directory returns nothing.
  - `Runtime-args update (get_dynamic_runtime_args) = no` — confirmed: hook absent from both device-ops.
  - `Override runtime args method? (PD only) = no` — confirmed: no `override_runtime_arguments` anywhere.
  - `Pybind descriptor = no` — confirmed: `std_var_reductions_nanobind.cpp` contains no `create_descriptor`
    binding and no `nb::class_` of a device-op.
  - **Factory-set match** — the sheet's four rows correspond one-to-one with the four factories in the code; no
    phantom row, no missing row. The `program_factory_t` variants list exactly
    `{ReduceSingleCoreHw, ReduceMultiCoreH, ReduceMultiCoreW}` and `{WelfordReduce}`.
  - **Cross-column invariants** — `Op-owned tensors?` blank on `descriptor` rows (consistent);
    `get_dynamic_runtime_args = no` (no invariant to violate).
  - `Is safe to port? = yes` — the readiness-sheet owner's call, not re-derived. Consistent with our own
    observation that no buffer address is smuggled through an RTA (see Tensor bindings).

  One **column-name discrepancy** worth flagging to the recipe maintainer, not a gate: the live sheet's header
  reads `Override runtime args method? (PD only)`, while `metal2_audit.md` and `ttnn_op_porting_readiness.md`
  both name it `Override runtime args method? (PD and legacy)`. Same column, same `no` value here — recorded
  under Recipe notes.

- **Device 2.0 (every kernel used):** **GREEN.** Every kernel the op instantiates — its own 14 and both borrowed
  writers — is structurally Device 2.0. A full scan for Device 1.0 idioms
  (`noc_async_read(` / `noc_async_write(` free functions, `InterleavedAddrGen`, `ShardedAddrGen`,
  `InterleavedAddrGenFast`, `InterleavedPow2AddrGen*`, `cb_reserve_back` / `cb_push_back` / `cb_wait_front` /
  `cb_pop_front`, `get_noc_addr_from_bank_id`, raw semaphore addresses) over all 16 files returns **zero** hits.
  All data movement goes through `Noc` + `TensorAccessor` / `UnicastEndpoint`; all CB access goes through
  `DataflowBuffer` or `CircularBuffer` wrapper objects; every `get_write_ptr()` / `get_read_ptr()` in the op is a
  **method call on a wrapper object**, never the free function.

  Free functions taking a CB index, and why none is a holdover:

  | File | Line | Call | Verdict |
  |---|---|---|---|
  | `kernels/dataflow/reader_unary_reduce_universal_start_id.cpp` | 26 | `get_tile_size(dfb_id_in0)` | **sanctioned** (Green bullet) |
  | `kernels/dataflow/reader_unary_transpose_wh_universal_input_cols_partitioned.cpp` | 42 | `get_tile_size(dfb_id_in0)` | **sanctioned** |
  | `kernels/dataflow/reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp` | 44 | `get_tile_size(dfb_id_in0)` | **sanctioned** |
  | `kernels/dataflow/writer_reduce_rm_scalar.cpp` | 55 | `get_tile_size(dfb_id_tile)` | **sanctioned** |
  | `kernels/dataflow/writer_welford_hw.cpp` | 146, 147 | `get_tile_size(dfb_partial)`, `get_tile_size(dfb_out)` | **sanctioned** |
  | `kernels/dataflow/reader_unary_reduce_rm.cpp` | 75 | `get_tile_size(cb_id_clear_value)` | **sanctioned** |
  | `kernels/dataflow/reader_unary_reduce_rm.cpp` | 82 | `get_local_cb_interface(cb_id_rm).fifo_page_size` | **sanctioned** (Green bullet) |
  | *(borrowed)* `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` | 19 | `get_local_cb_interface(cb_id_out).fifo_page_size` | **sanctioned** |
  | `kernels/dataflow/reader_unary_transpose_wh_universal_input_cols_partitioned.cpp` | 35 | `constexpr DataFormat reduce_format = get_dataformat(dfb_id_in0);` | **not a holdover** — see below |
  | `kernels/dataflow/reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp` | 38 | `constexpr DataFormat reduce_format = get_dataformat(dfb_id_in0);` | **not a holdover** — see below |

  **The `get_dataformat(cb_id)` judgment call.** This free function is *not* on the audit's sanctioned list
  (which names only `get_tile_size` and `get_local_cb_interface`), so I checked it against the holdover
  definition rather than the list. A holdover requires that *a wrapper-method replacement exists* at the call
  site. `CircularBuffer::get_dataformat()` exists (`tt_metal/hw/inc/api/dataflow/circular_buffer.h:115`) but is
  **not `constexpr`**, whereas both call sites are `constexpr` initializers whose result is then used as a
  **template argument** (`is_sfpu_reduce_path<REDUCE_OP, REDUCE_DIM, reduce_format, fp32_mode>()`). The
  Device 2.0 wrapper therefore offers no replacement usable here, and at the sharded reader the wrapper object
  is not even constructed until line 47. **Verdict: not a Device 2.0 holdover — GREEN.** It *is* Metal 2.0 port
  work (whitelist rule 7), and the port has a clean answer: `DataflowBuffer::get_dataformat()` **is** `constexpr`
  (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:241`). Carried to the brief as a construct note.

  The same reasoning covers the `unpack_src_format[cb_id]` array lookups in `compute/reduce.cpp:41`,
  `compute/reduce_h_neg.cpp:38`, `compute/reduce_w_neg.cpp:40` — an array index is not a CB-index free function,
  and `metal2_port.md` rule 7 explicitly owns it as port-time work.

- **Feature compatibility:** all four Appendix A entries scanned against host code, factory code, descriptors,
  and kernel code.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `global_circular_buffer` field on any `CBDescriptor`, no `remote_index` / `remote_cb_*` identifiers, no `global_circular_buffer.hpp` include. The op's only Buffer-backed CBs are the ordinary borrowed-memory pattern (`.tensor = &a` / `.tensor = &output`, `reduce_op_multi_core_h_program_factory.cpp:195,243`) — a mechanical porting-recipe translation to `DataflowBufferSpec::borrowed_from`, explicitly not this entry. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `CBDescriptor` in any of the four factories sets `.address_offset`; it defaults to 0. No `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor` call. |
  | GlobalSemaphore | N/A | The op uses **no semaphores at all** — `grep -rn 'emaphore'` over the whole op directory returns nothing. No `global_semaphore.hpp` include, no `CreateGlobalSemaphore`. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | Op-level cue absent: `tensor_args_t = Tensor` (a single tensor), not a variable-count container. Kernel-level decider absent: every `get_compile_time_arg_val(...)` index across all 14 kernels is a compile-time constant. The four indices that *look* computed — `reader_unary_reduce_rm.cpp:133` and `writer_reduce_rm_scalar.cpp:102,103,105` — are `constexpr` `(DIM == ckernel::ReduceDim::REDUCE_COL) ? N : 0` selectors written specifically so the discarded `if constexpr` branch does not eagerly instantiate an out-of-range slot. Fixed-count, not variadic. |

- **CB endpoints (GATE-free):** every CB in every config is either a plain 1:1 (one locked producer + one locked
  consumer) or a **self-loop** (single toucher). **No multi-binding CB, no dead CB, anywhere in this op.**
  The census below is per `(CB, config)`; the Device 2.0 gate is GREEN so nothing is deferred.

  Legend for touchers: `R` = reader kernel, `W` = writer kernel, `C` = compute kernel.

  **`ReduceMultiCoreHProgramFactory`**

  | Config | CB | Touchers | Census | Disposition |
  |---|---|---|---|---|
  | interleaved-tiled | `c_0` src0 | R produces / C consumes | 1P + 1C | 1:1 |
  | interleaved-tiled | `c_2` scaler | R produces (`prepare_reduce_scaler`) / C consumes (`compute_kernel_lib::reduce` waits) | 1P + 1C | 1:1 |
  | interleaved-tiled | `c_3` out | C produces / W consumes (borrowed `writer_unary_interleaved_start_id`) | 1P + 1C | 1:1 |
  | + fused negate | `c_4` acc | C only (`reduce_h_neg.cpp:166,178,204,206,212,217,240`) | 1 toucher | **self-loop** |
  | + fused negate | `c_5` ineg | C only (`reduce_h_neg.cpp:166–201`) | 1 toucher | **self-loop** |
  | width-sharded | `c_0` src0 | R produces / C consumes | 1P + 1C | 1:1 |
  | width-sharded | `c_1` src1 (borrowed from input) | R only — `reserve_back(num_tiles)` + `get_write_ptr()` self-read (`reader_..._sharded.cpp:50,51`) | 1 toucher | **self-loop** |
  | width-sharded | `c_2` scaler | R produces / C consumes | 1P + 1C | 1:1 |
  | width-sharded | `c_3` out (borrowed from output) | C produces / W consumes (borrowed `writer_unary_sharded`, a readiness handshake) | 1P + 1C | 1:1 |
  | width-sharded + negate | `c_4`, `c_5` | C only | 1 toucher each | **self-loop** ×2 |
  | dense RM | `c_24` cb_rm | R produces (`reader_unary_reduce_rm.cpp:104`) / C consumes (`tilize<…, cb_rm, cb_tile_in>`) | 1P + 1C | 1:1 |
  | dense RM | `c_4` clear_value | R only — fills via `get_write_ptr()`, `push_back`, then re-reads via `get_read_ptr()` as a NoC source (`reader_unary_reduce_rm.cpp:77,78,79`) | 1 toucher | **self-loop** |
  | dense RM | `c_5` cb_acc | C only — `compute_kernel_lib::Accumulate::at(cb_acc, chunk_idx)` packs and reads back inside the compute kernel | 1 toucher | **self-loop** |
  | dense RM | `c_0` cb_tile_in | C only — `tilize` writes it, `reduce` drains it, both inside `reduce_rm.cpp` | 1 toucher | **self-loop** |
  | dense RM | `c_2` scaler | R produces / C consumes | 1P + 1C | 1:1 |
  | dense RM | `c_3` out | C produces / W consumes (`writer_reduce_rm_scalar`) | 1P + 1C | 1:1 |

  **`ReduceMultiCoreWProgramFactory`**

  | Config | CB | Touchers | Census | Disposition |
  |---|---|---|---|---|
  | interleaved-tiled | `c_0`, `c_2`, `c_3` | R→C, R→C, C→W | 1P + 1C each | 1:1 |
  | + fused negate | `c_4` acc, `c_5` inv | C only (`reduce_w_neg.cpp:139–169`) | 1 toucher each | **self-loop** ×2 |
  | dense RM | `c_24`, `c_4`, `c_5`, `c_0`, `c_2`, `c_3` | identical shape to the H dense-RM column above (same three kernels, `REDUCE_DIM` differs) | — | 1:1 / self-loop as above |

  **`ReduceSingleCoreHwProgramFactory`**

  | Config | CB | Touchers | Census | Disposition |
  |---|---|---|---|---|
  | default | `c_0`, `c_2`, `c_3` | R→C, R→C, C→W | 1P + 1C each | 1:1 |
  | negate | `c_4` acc, `c_5` ineg | C only (`reduce_hw_neg.cpp`) | 1 toucher each | **self-loop** ×2 |

  **`WelfordReduceProgramFactory`**

  | Config | CB | Touchers | Census | Disposition |
  |---|---|---|---|---|
  | W-reduce | `c_0` in | R produces / C consumes | 1P + 1C | 1:1 |
  | W-reduce | `c_2` scalar | **R produces only** — `prepare_reduce_scaler` in `reader_unary_reduce_universal_start_id.cpp:21`; no welford compute kernel reads `c_2` (they apply the user scalar via `WELFORD_POST_MUL` / `mul_unary_tile` instead) | 1 toucher | **self-loop** (see Misc anomaly 2) |
  | W-reduce | `c_16` out | C produces / W consumes (borrowed writer) | 1P + 1C | 1:1 |
  | W-reduce | `c_19` cb_var | C only — packs (`welford_reduce_w.cpp:149–154`) then `wait_front`/`transpose_tile`/`pop_front` (156–172) | 1 toucher | **self-loop** |
  | H-reduce | `c_0`, `c_16` | R→C, C→W | 1P + 1C each | 1:1 |
  | H-reduce | `c_2` scalar | R produces only (`reader_..._cols_partitioned.cpp:46`); `welford_reduce_h.cpp` never reads it | 1 toucher | **self-loop** (see Misc anomaly 2) |
  | HW-reduce | `c_0` in | R produces / C consumes | 1P + 1C | 1:1 |
  | HW-reduce | `c_2` scalar | R produces only | 1 toucher | **self-loop** (see Misc anomaly 2) |
  | HW-reduce | `c_21` partial | C produces (`welford_reduce_hw.cpp:154–159`) / W consumes (`writer_welford_hw.cpp:177,179,219`, incl. a `get_read_ptr()` peek on its own consumer binding) | 1P + 1C | 1:1 |
  | HW-reduce | `c_22` combined | W produces (`writer_welford_hw.cpp:266,268/276,282`, incl. a `get_write_ptr()` peek on its own producer binding) / C consumes (`welford_reduce_hw.cpp:169–188`) | 1P + 1C | 1:1 |
  | HW-reduce | `c_16` out | C produces / W consumes (`writer_welford_hw.cpp:285–289`) | 1P + 1C | 1:1 |

  **Hidden-second-writer hunt (face (a)) — run, negative.** Every `get_write_ptr()` / `get_read_ptr()` /
  `get_local_cb_interface(...)` raw access in the op was traced to the kernel that already holds the FIFO role on
  that CB (a peek on its own binding), and there is no semaphore anywhere in the op to coordinate a co-fill — so
  no CB has a raw co-filler that a FIFO trace would miss. **Face (b)** (multiple readers): no CB is read by two
  co-resident kernels. **Face (c)** (dual-instance work-split): no factory pushes the same `kernel_source` into
  two `KernelDescriptor`s over one core range — the only same-source pairs are `compute_desc_g1` / `compute_desc_g2`,
  which cover **disjoint** core groups (`core_group_1` / `core_group_2`), i.e. the ordinary per-group split, not
  the dual-instance shape. Each node sees exactly one compute instance.

- **Offset base pointers:** **GREEN.** No address RTA in this op folds a host-side offset into a base — in fact
  no `->address()` expression reaches a runtime-arg context at all. `grep -rn 'address()'` over the op directory
  returns only `device->lowest_occupied_compute_l1_address()` (an L1-budget query, `reduce_op_multi_core_h_program_factory.cpp:302`
  and `common.cpp:350`) and two `tensor_args.buffer() != nullptr` validation checks. All tensor addresses reach
  the kernels through the framework's `MeshTensor` binding overload of `emplace_runtime_args` (see Tensor
  bindings below), which delivers a clean base and nothing else. `reduction/generic` is not in the
  `2026-07-19_offset_base_pointers.md` tables, and the scan agrees — **"no fold, op not in the tables → clean."**
  No Type 3 (`address_offset` is unset everywhere) and no Type 4 (`ttnn::narrow` / interior-base `MeshBuffer::create`)
  construct appears.

- **TensorAccessor 3rd argument:** **GREEN.** The subject does not fire. All five `TensorAccessor` construction
  sites in the op pass exactly two arguments — `TensorAccessor(args, addr)`:
  `reader_unary_reduce_universal_start_id.cpp:28`, `reader_unary_transpose_wh_universal_input_cols_partitioned.cpp:49`,
  `reader_unary_reduce_rm.cpp:81`, `writer_reduce_rm_scalar.cpp:56`, `writer_welford_hw.cpp:154`. The borrowed
  `writer_unary_interleaved_start_id.cpp:31` is likewise two-argument. No explicit page size is overridden
  anywhere, so there is no value to classify and nothing for the port to drop.
  `reduction/generic` does not appear in `2026-07-06_tensor_accessor_3rd_arg_triage.md` (its `reduce`-matching
  rows are `deepseek_moe_fast_reduce_nc_fused` and `deepseek_moe_post_combine_reduce`, different ops) — the
  table's silence and our read agree.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per factory). Every tensor address is delivered by the framework's
  `MeshTensor`-binding overload — `KernelDescriptor::emplace_runtime_args(core, {tensor, …})`
  (`tt_metal/api/tt-metalium/program_descriptors.hpp:195-197`), which auto-registers a `BufferBinding` and
  patches it on cache hits. This is the *delivery* mechanism, not the silent-wrong `->address()`-on-an-RTA
  hazard, and it matches the sheet's `Smuggled pointer = no`. Classification is therefore by what the kernel
  does with the base it receives:

  | Factory / config | Binding | Delivery site | Kernel use | Case |
  |---|---|---|---|---|
  | `ReduceMultiCoreH` interleaved-tiled | input | `…multi_core_h…:602-603` (`{a, …}`) | `TensorAccessor(tensor_args, src_addr)` @ `reader_..._cols_partitioned.cpp:49` | **Case 1** |
  | `ReduceMultiCoreH` interleaved-tiled | output | `…multi_core_h…:605-611` (`{output, …}`) | `TensorAccessor(dst_args, dst_addr)` @ borrowed `writer_unary_interleaved_start_id.cpp:31` | **Case 1** |
  | `ReduceMultiCoreH` width-sharded | input | *(none — no address RTA)* | read via borrowed-memory DFB `c_1` (`.tensor = &a`, `…multi_core_h…:195`) | **clean** |
  | `ReduceMultiCoreH` width-sharded | output | *(none — no address RTA)* | written in place via borrowed-memory DFB `c_3` (`.tensor = &output`, `…:243`) | **clean** |
  | `ReduceMultiCoreH` dense RM | input / output | `…multi_core_h…:540-553` | `TensorAccessor` @ `reader_unary_reduce_rm.cpp:81` / `writer_reduce_rm_scalar.cpp:56` | **Case 1** ×2 |
  | `ReduceMultiCoreW` (both configs) | input / output | `…multi_core_w…:366-379` (RM), `383-397` (tiled) | `TensorAccessor` @ the respective reader/writer | **Case 1** ×2 |
  | `ReduceSingleCoreHw` | input / output | `…single_core_hw…:203`, `:213` | `TensorAccessor` @ `reader_unary_reduce_universal_start_id.cpp:28` / borrowed writer | **Case 1** ×2 |
  | `WelfordReduce` (W / H / HW) | input / output | `…welford…:513,516` (W), `547-559` (HW), `579-584` (H) | `TensorAccessor` @ the respective reader / `writer_welford_hw.cpp:154` / borrowed writer | **Case 1** ×2 |

  **Op-level roll-up: ⚠ port work** — every binding is Case 1 (express as `TensorParameter` / `TensorBinding`;
  the kernel builds `TensorAccessor(tensor::name)` and the `TensorAccessorArgs(...).append_to(...)` CT-arg
  plumbing plus the address RTA both disappear), except the width-sharded H config's two bindings, which are
  `clean` (borrowed-memory DFB → `DataflowBufferSpec::borrowed_from`). **No Case 2 anywhere** — no kernel in this
  op does hand-rolled address arithmetic on a tensor base, so the `get_bank_base_address` bridge is not needed.

  *Per-factory split to preserve:* the same `input` / `output` `TensorParameter` is **clean** in
  `ReduceMultiCoreH`'s width-sharded config and **Case 1** in its other three configs. Do not flatten this to a
  single verdict.

- **TensorParameter relaxation:** none. The sheet lists `none` on all four rows, and there is no custom hash to
  reconcile a relaxation against.
- **TensorAccessor 3rd arg:** none — no site to drop.
- **CB endpoints:** self-loop on `c_4` / `c_5` (fused-negate configs, all three Reduce factories), `c_4` / `c_5` /
  `c_0` (dense-RM configs, H and W), `c_1` (width-sharded H), `c_19` (Welford W), and `c_2` (all three Welford
  configs). Everything else is a plain 1:1. **No multi-binding flag. No dead-CB drop.**

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. The hidden-second-writer, multiple-reader, and
  dual-instance-work-split hunts were all run and all came back negative (detail in Gate detail above). The
  porter does not need to re-run them.
- **Cross-op / shared kernels:** two borrowed writer kernels, no fork yet for either — this port creates the
  first `_metal2` fork of each, beside the original:
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
    (owner: `eltwise/unary`; bound by **~34** op factories tree-wide) — used by `ReduceMultiCoreH`
    (interleaved-tiled), `ReduceMultiCoreW` (tiled), `ReduceSingleCoreHw`, and `WelfordReduce` (W and H).
  - `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp`
    (owner: `data_movement/sharded`; bound by **~11** op factories tree-wide) — used by `ReduceMultiCoreH`
    width-sharded only.

  Both consumer counts are **sunset / coordination lists, not authorization to convert in place**. Neither
  directory contains a `_metal2` sibling today (checked locationally, per rung 1). Note that
  `writer_unary_interleaved_start_id.cpp` is one of the most widely-bound kernels in the tree, so its fork's
  binding names will be inherited by many later ports — name them from the kernel's own vocabulary
  (`dfb::out`, `tensor::dst`), not from reduce's locals.

  All of the op's **own** kernels are exclusively bound by its own four factories — nothing here is *lent*. (A
  naive filename grep suggests `compute/reduce.cpp` is shared with `experimental/ssm/hc_sum_reduce` and
  `experimental/reduction/deepseek_moe_fast_reduce_nc`; both are false positives — those factories bind
  `ssm_1d_sum_reduce.cpp` and `deepseek_moe_fast_reduce_nc_reduce.cpp` respectively, their own files.)

- **RTA varargs:** none. Every kernel reads a fixed set of runtime args at constant indices
  (`get_arg_val<uint32_t>(0)` … `(5)` at most, in `reader_..._sharded.cpp`), with no counted loop over an arg
  index and no data-selected read. The porter names every RTA.

## Team-only

- **Out-of-directory coupling & donor shape.**

  **Op-level roll-up: ✓ clean.** No donor function-call escape is blocked or needs donor-side work. No `⭐`
  entries — no donor is pre-Device-2.0, and no donor function takes a `CircularBuffer&`.

  *Summary table — one row per (op kernel, donor file):*

  | Op kernel | Donor file | Donor class | Status |
  |---|---|---|---|
  | all readers/writers | `tt_metal/hw/inc/api/dataflow/{dataflow_api.h,noc.h,dataflow_buffer.h,circular_buffer.h,endpoints.h}` | 1 — `tt_metal/*` framework | ✓ |
  | all compute kernels | `tt_metal/hw/inc/api/compute/*`, `llk_defs.h`, `llk_math_eltwise_binary.h` | 1 — `tt_metal/*` framework | ✓ |
  | `reader_*`, `writer_reduce_rm_scalar`, all compute | `ttnn/cpp/ttnn/kernel_lib/{reduce_helpers_common,reduce_helpers_dataflow,reduce_helpers_compute,dest_helpers,tilize_helpers}.hpp` | 2 — official shared kernel library | ✓ |
  | `reader_unary_reduce_rm.cpp`, `reduce_rm_dataflow_common.hpp` | `ttnn/cpp/ttnn/operations/pool/device/kernels/experimental_device_api.hpp` | 6 — **cross-family donor** (`pool`) | ✓ |

  *Per-call detail (the one non-trivial donor):*

  `pool/device/kernels/experimental_device_api.hpp` is a 91-line convenience header — it is **not**
  pre-Device-2.0; it is a thin Device 2.0 alias/wrapper layer, and it is broadly shared (conv2d, fold,
  convert_to_chw/hwc, conv3d, padded_slice, and others include it). The op uses exactly two things from it:

  | Symbol | Shape | Status |
  |---|---|---|
  | `experimental::CB` (`experimental_device_api.hpp:24`) | a `using` **type alias** for the Device 2.0 `CircularBuffer` — not a function call, no handle crosses a signature | ✓ — the port simply declares a `DataflowBuffer` / `dfb::name` instead; the alias disappears from the op's kernels |
  | `experimental::local_addr(uint32_t addr, uint8_t noc_id)` (`:37`) | takes a **raw `uint32_t` L1 address** and returns a `noc_traits_t<UnicastEndpoint>::src_args_type` | ✓ — the address handed in is `cb_clear_value.get_read_ptr()`, an in-op peek on the op's own buffer; a `dfb::name`-derived read pointer feeds it unchanged |

  The op's own helper `rm_fill_page_with_clear_template(Noc&, experimental::CB&, …)`
  (`reduce_rm_dataflow_common.hpp:108-122`) *does* take a `CircularBuffer&` — but the file is the op's own
  (included only by `reader_unary_reduce_rm.cpp`, no external consumer), so it is inside the porter's writeable
  surface and the parameter type changes with the port. It is **not** the `⭐ CircularBuffer&` donor-boundary
  flag, which concerns donor signatures the porter may not rewrite.

  Every kernel-lib entry point the op calls takes CB ids as **template non-type parameters** or plain runtime
  `uint32_t` — `compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, cb_in, cb_scaler, cb_out, …>`,
  `compute_kernel_lib::tilize<wt_tiles_per_chunk, cb_rm, cb_tile_in>`,
  `compute_kernel_lib::Accumulate::at(cb_acc, chunk_idx)`,
  `dataflow_kernel_lib::prepare_reduce_scaler<dfb_id, REDUCE_OP, REDUCE_DIM>` — all of which `dfb::name`'s
  constexpr cast covers in both runtime and template-parameter position. Per the recipe, kernel-lib is out of
  porter scope and is never forked.

  *Borrowed kernel files (file-path instantiation):* the two writers listed under Heads-ups. Both Device 2.0,
  neither forked, consumer counts ~34 and ~11.

- **Relaxation candidates:** none to mine — the op carries no custom hash (present or renamed), so there is no
  hash logic to read a candidate relaxation out of.

- **TTNN factory analysis** (sheet-derived facts with `file:line` evidence):
  - **Op-owned tensors:** none. Both device-ops' factories return `tt::tt_metal::ProgramDescriptor` directly
    (`reduce_op_device_operation.hpp:25,32,39`; `welford_reduce_device_operation.hpp:24`); there is no
    `WorkloadDescriptor` and therefore no `buffers` vector.
  - **MeshWorkload need:** none — genuine single-program SPMD; the sheet's `Execution Model` column reads `SPMD`.
  - **Pybind `create_descriptor`:** absent. `std_var_reductions_nanobind.cpp` /
    `generic_reductions_nanobind.hpp` bind only the user-level ops.
  - **Other risky pybind:** none surfaced (`Is safe to port? = yes`, no `warning`).
  - **Custom hash:** absent (`Formerly custom hashed? = yes` is historical — a hash that has already been removed).
  - **`get_dynamic_runtime_args`:** absent from both device-ops.
  - **`override_runtime_arguments`:** absent from all four factories.
  - **Target concept:** `ProgramSpecFactoryConcept` for all four factories, with no op-owned tensors — matches
    the sheet's `Porting Target` column exactly.

## Misc anomalies  *(team-only, non-gating; route to the ops team — the port does not act on these)*

1. **`MULTI_CORE_HW` never selects a distinct factory.** `get_parallelization_strategy` returns
   `ReduceOpParallelizationStrategy::MULTI_CORE_HW` when `num_tiles > 1` (`device/common.cpp:157-159`), but
   `select_program_factory` maps `MULTI_CORE_HW` and `SINGLE_CORE_HW` to the *same*
   `ReduceSingleCoreHwProgramFactory` (`device/reduce_op_device_operation.cpp:23-25`). The multi-tile HW case
   never reaches that factory in practice, because `reduce_op.cpp:215` intercepts `is_multicore_hw` and
   decomposes it into a W-then-H pair of `ttnn::prim::reduce` calls. The enum value is therefore dead as a
   *selector* and reads as if a multi-core HW factory exists.

2. **The Welford factory allocates and fills a scalar CB that no consumer reads.** `c_2` is declared at
   `welford_reduce_program_factory.cpp:189-198` and filled by the reader on all three reduce dims
   (`prepare_reduce_scaler`, `reader_unary_reduce_universal_start_id.cpp:21` /
   `reader_unary_transpose_wh_universal_input_cols_partitioned.cpp:46`), but **none** of
   `welford_reduce_w.cpp` / `welford_reduce_h.cpp` / `welford_reduce_hw.cpp` references `c_2` — the user scalar
   is applied post-reduction via `WELFORD_POST_MUL` / `mul_unary_tile` instead
   (`welford_reduce_program_factory.cpp:241-244,296-298`). Cost is one tile of L1 plus one zero-fill +
   `flush`/`push_back` per core per launch. Not a dead CB in the audit sense (the reader is a real toucher, so
   it ports as a self-loop), but it is dead *work* that could be dropped by the ops team.

3. **`dst_full_sync_en` from the user's compute-kernel config is silently dropped by three of the four
   factories.** All four destructure it from `get_compute_kernel_config_args`, but only
   `ReduceMultiCoreHProgramFactory` forwards it into `ComputeConfigDescriptor`
   (`reduce_op_multi_core_h_program_factory.cpp:474,506`). `ReduceMultiCoreWProgramFactory` (`:298-302,325-329`),
   `ReduceSingleCoreHwProgramFactory` (`:198-201`), and `WelfordReduceProgramFactory` (`:459-463,474-478`) omit
   it, so those compute kernels run at the default sync mode regardless of what the caller asked for. A related
   latent trap in the Welford factory: it *does* pass `DST_SYNC_FULL = dst_full_sync_en` as a **define** to both
   reader and compute (`:292`), while the compute kernel's `DEST_AUTO_LIMIT` resolves from the JIT-generated
   `DST_SYNC_MODE` (which wins — `kernel_lib/dest_helpers.hpp:71-79`). Today the two cannot disagree in a way
   that matters, because the Welford readers never consume `DEST_AUTO_LIMIT` (`use_welford=1` pins
   `row_chunk = 1`, `reader_..._cols_partitioned.cpp:37-39`). If a future change lets a Welford reader use
   `DEST_AUTO_LIMIT`, reader and compute would size DEST from different sources.

4. **Dead compute RTA on the H dense-RM path.** `reduce_op_multi_core_h_program_factory.cpp:555,557` passes
   `{num_output_tiles_local, output_tiles_seen}` to the compute kernel, but `reduce_rm.cpp` reads only
   `get_arg_val<uint32_t>(0)` and its own comment states arg 1 is unused
   (`compute/reduce_rm.cpp:121-124`). Arg 1 is dead plumbing.

5. **Stale comment contradicts the code.** `reduce_op_multi_core_w_program_factory.cpp:365` reads
   *"Use raw addresses (not Buffer\*) so mesh program-cache fast paths re-apply per-core args"*, but the code
   immediately below passes the `MeshTensor`s `a` / `output` into `emplace_runtime_args` — the binding overload,
   not a raw address. The comment describes a shape the code no longer has and would mislead a reader into
   thinking an address is being smuggled here.

6. **Unused destructured config fields.** `math_approx_mode` and `packer_l1_acc` are bound by the structured
   binding from `get_compute_kernel_config_args` in all four factories and never used. Harmless, but they make
   the `dst_full_sync_en` drop in anomaly 3 harder to spot by eye.

## Per-DeviceOperation attribution

| Field | `ReduceDeviceOperation` | `WelfordReduceDeviceOperation` |
|---|---|---|
| Factories | 3 (`SingleCoreHw`, `MultiCoreH`, `MultiCoreW`) | 1 (`WelfordReduce`) |
| `Is able to port?` | Yes (all 3 rows) | Yes |
| Concept → target | `descriptor` → `ProgramSpecFactoryConcept` | `descriptor` → `ProgramSpecFactoryConcept` |
| Device 2.0 | GREEN | GREEN |
| Feature compatibility | GREEN (all N/A) | GREEN (all N/A) |
| Offset base pointers | GREEN | GREEN |
| TensorAccessor 3rd arg | GREEN (no sites) | GREEN (no sites) |
| Tensor bindings | Case 1 ×2 per config; **clean ×2** in `MultiCoreH` width-sharded (borrowed DFB) | Case 1 ×2 in all three reduce dims |
| CB endpoints | 1:1 + self-loops (`c_4`/`c_5` negate, `c_0`/`c_4`/`c_5` dense-RM, `c_1` width-sharded) | 1:1 + self-loops (`c_19` W-reduce, `c_2` all dims) |
| Borrowed kernels | `writer_unary_interleaved_start_id.cpp` (H tiled, W tiled, HW), `writer_unary_sharded.cpp` (H width-sharded) | `writer_unary_interleaved_start_id.cpp` (W, H) |
| Distinct anomalies | 1, 3 (partial), 4, 5, 6 | 2, 3 (partial), 6 |
| **Verdict** | **GREEN** | **GREEN** |

## Questions for the user

1. **Welford's unread scalar CB:** `c_2` is allocated and filled but read by no Welford compute kernel
   (Misc anomaly 2). Confirming with the op owner that this is vestigial — rather than a half-landed feature —
   would let a later cleanup drop the allocation and the reader's `prepare_reduce_scaler` call. **This is not a
   port-blocking question**; the port carries the CB forward faithfully as a self-loop either way.

## Recipe notes

1. **Readiness-sheet column name drift.** `metal2_audit.md` (§TTNN factory concept prerequisite) and
   `analyses/ttnn_op_porting_readiness.md` both name the column
   `Override runtime args method? (PD and legacy)`. The live sheet's header, fetched this run, reads
   `Override runtime args method? (PD only)`. The readiness doc states as a standing guarantee that *"existing
   column names never change"* — that guarantee did not hold here. The value was `no` on all four rows, so
   nothing turned on it, but a stricter auditor reading the column strictly by the documented header name would
   have found no match and could have escalated it as "spreadsheet is broken." Recommend the doc either track
   the current header or say the match is by prefix.

2. **The sanctioned CB-index free-function list is two entries short of what real Device-2.0 kernels use.**
   The Device 2.0 Green bullet names `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)` as sanctioned.
   This op also uses `get_dataformat(cb_id)` (2 sites) and the `unpack_src_format[cb_id]` descriptor array
   (3 sites) — both in **`constexpr`** position feeding template arguments. I cleared them (reasoning in Gate
   detail: the `CircularBuffer` wrapper's `get_dataformat()` is *not* `constexpr`, so no wrapper-method
   replacement exists at those call sites, which is the holdover test's own precondition). But this required a
   non-trivial read of two framework headers to resolve what the list presents as a lookup, and the same two
   constructs will appear in essentially every reduce-shaped op. Suggest either adding `get_dataformat(cb_id)`
   to the sanctioned list explicitly, or adding a sentence to the holdover bullet noting that a **`constexpr`
   use site** is exempt whenever the wrapper's method is not itself `constexpr`.

3. **A "compute-internal CB" is a common self-loop shape the CB-endpoints section doesn't name.** Several CBs
   here (`c_5` cb_acc and `c_0` cb_tile_in on the dense-RM path, `c_19` on Welford W, `c_4`/`c_5` on every
   fused-negate path) are touched only by the compute kernel, which both packs into them and unpacks back out —
   a scratch/accumulator round-trip through L1, not a producer/consumer pipe. The classification table handles
   it correctly as "1 toucher → self-loop," but the section's prose frames one-toucher CBs as *"single-ended /
   sync-free"* pointer-only access, which reads as a dataflow-kernel phenomenon. Naming the compute-scratch
   shape explicitly would save the next auditor a moment's doubt that a genuine FIFO producer *and* consumer,
   both inside one compute kernel, really is the one-toucher row.

4. **Multi-device-op bundling worked well, and the shared-code test was unambiguous here** — the two device-ops
   share `common.hpp`/`common.cpp`, the defines helper, and two dataflow kernels, so bundling was clearly right.
   No friction; noting it as a positive data point for the rule as written.
