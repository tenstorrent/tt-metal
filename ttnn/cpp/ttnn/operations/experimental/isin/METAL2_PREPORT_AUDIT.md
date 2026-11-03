# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/isin`

- **`IsInDeviceOperation`** (`device/isin_device_operation.hpp`, `device/isin_device_operation.cpp`; types in `device/isin_device_operation_types.hpp`)
  - `IsInProgramFactory` (`device/isin_program_factory.hpp`, `device/isin_program_factory.cpp`) — the only factory (`program_factory_t = std::variant<IsInProgramFactory>`, `isin_device_operation.hpp:17`)
    - reader `device/kernels/dataflow/isin_reader.cpp` (`ReaderConfigDescriptor`)
    - writer `device/kernels/dataflow/isin_writer.cpp` (`WriterConfigDescriptor`)
    - kernel-side helper header used by both: `device/kernels/dataflow/isin_common.hpp`
- Host wrapper `isin.cpp` / `isin.hpp` (`ttnn::experimental::isin`): converts both inputs to ROW_MAJOR, flattens them to 1-D, sizes the L1 subchunk, and restores the output's shape and layout. Host constants live in `isin_common.hpp`. The Python binding `isin_nanobind.cpp` binds the free function only.
- No unreferenced kernel files. No copy of this op exists under `experimental/quasar/`.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `` c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead `.id` form ``

**Code audited at:** `977421c3a1c` on `iwrosz/isin_metal2.0` (recipe tip `c41f7f4ef0a` + `origin/main` `a1ce48e3fe8`). **Readiness sheet:** fresh fetch taken by the orchestrating session at 2026-11-03 13:06 UTC (28 columns, 477 data rows). It has exactly one row for this op.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/isin` |
| **Overall** | **RED**: the readiness sheet is stale for this op ("spreadsheet is broken", routed to the readiness-sheet owner). Every gate that depends on the op's code clears. |
| **DOps / Factories** | `IsInDeviceOperation` → `IsInProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes.** `isin_reader.cpp`, `isin_writer.cpp` and `isin_common.hpp` are all on `Noc` / `CircularBuffer` / `TensorAccessor`. Two judgment calls are recorded, neither counted as a violation: `get_dataformat(cb)` used as a template argument (`isin_reader.cpp:91`), and 4 raw L1 casts on wrapper-supplied pointers. See Gate detail and Question 2. |
| *Prereqs* — Cross-op escapes | **Ok.** No include resolves outside `tt_metal/*`. Both kernels are op-owned and bound only by this factory. |
| *Feature Support* — overall | **GREEN** (all three Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | **Ok.** No CTA varargs: fixed CTA indices 3–11 plus chained `TensorAccessorArgs`. Appendix A has no such entry today (Recipe note 2). |
| *TTNN Readiness* — `Is able to port?` (the gate) | Sheet: `yes (with PD step)` → **RED**. Attributed to a **spreadsheet-broken** conflict on `Concept`: the PD step the cell asks for is already done. Routed to the readiness-sheet owner. |
| *TTNN Readiness* — Concept (current) | Code: **`descriptor`** (since `d67dde5ad66`, #57541, 2026-09-29). Sheet: `legacy device-op` (**stale**). |
| *TTNN Readiness* — Execution Model | `SPMD` (not mesh-shaped, so `Secretly SPMD Workload?` is N/A) |
| *TTNN Readiness* — Custom hash | No — default hash. Sheet `no` / backdoor `no`; matches the code. |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No (matches) |
| *TTNN Readiness* — `override_runtime_arguments` | No in code. The sheet shows `n/a` because the column is "PD only" and the row is classed legacy, so it follows from the stale `Concept`. |
| *TTNN Readiness* — Pybind `create_descriptor` | No (matches) |
| *TTNN Readiness* — Op-owned tensors | No (matches) |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (the sheet's `Porting Target`). Consistent with the code; re-read it after the row is reconciled. |
| *Port work* — Offset base pointer | none |
| *Port work* — Tensor bindings (per binding) | `elements` Case 1 · `test_elements` Case 1 · `output` Case 1. All three are `Buffer*`-binding RTAs today. |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | drop (Class 2) at 3 sites |
| *Port work* — CB endpoints | self-loop on `c_0` ELEMENTS and `c_1` TEST_ELEMENTS (reader) · `c_2` OUTPUT is a legal 1:1 |

## Result

**RED → blocked on the TTNN factory concept gate, routed to the readiness-sheet owner.** The sheet still classifies `IsInProgramFactory` as `legacy device-op`, so its derived `Is able to port?` cell reads `yes (with PD step)`. The code contradicts it: `d67dde5ad66` ("[ttnn] Describe experimental isin with ProgramDescriptor (#57541)", 2026-09-29, already on `origin/main`) migrated the factory to `ProgramDescriptor`. That is a primary-column conflict. Per the recipe, the sheet is therefore broken for this op, and the gate holds until its owner reconciles the row. **`RED at op level; no portable subset`**: the op has a single factory.

Every gate that depends on the op's code clears: Device 2.0 ✓ · Feature compatibility ✓ (all N/A) · Offset base pointers ✓ · TensorAccessor 3rd arg ✓ (3 sites, Class 2, drop). All of the sheet's other cross-checked columns match the code.

**Path forward:** reconcile the row by setting `Concept` → `descriptor` and `Override runtime args method?` → `no`, then let the derived cells recompute. If `Is able to port?` then reads `yes`, the op needs **no code change** before the port. A re-audit only has to confirm the row and check that the code hasn't moved; it can then issue the brief from this report's *Port-work summary*, *Heads-ups* and *TTNN ProgramFactory* sections. If the cell reads `no` instead, route it per the recipe's blocking table.

**Scoping:** the blocker clears off the op's code, through a sheet edit. Per the Red-outcome exception (*a RED that clears without touching the op's code: run them anyway*), all seven informational subjects were run in full and none were skipped. I did **not** apply the legacy-API short-circuit. Its premise, that the op is still on the imperative API and needs a PD port, is contradicted by the code, and following it would send finished work to the PD-migration track (Recipe note 1).

No `METAL2_PORT_BRIEF.md` is issued because the verdict is RED.

## Gate detail

### TTNN factory concept (`Is able to port?`) — RED (spreadsheet broken)

Sheet row, cells verbatim: `Op` = `experimental/isin` · `Device operation` = `IsInDeviceOperation` · `Factory (variant)` = `IsInProgramFactory` · `Concept` = `legacy device-op` · `Op Classification` = `Legacy Op` · `Is able to port?` = `yes (with PD step)` · `Porting Target` = `ProgramSpecFactoryConcept` · `Execution Model` = `SPMD` · `TensorParameter relaxation` = `none` · `Known op issues` = *(blank)* · `Diego validation` = `yes`.

Cross-check of the primary columns against the code:

| Column | Sheet | Code | Evidence |
|---|---|---|---|
| `Concept` | `legacy device-op` | **`descriptor` — CONFLICT** | The only hook is `static ProgramDescriptor create_descriptor(const IsinParams&, const IsinInputs&, Tensor&)` (`device/isin_program_factory.hpp:13`; body at `isin_program_factory.cpp:42-155`). It takes no `MeshCoordinate` parameter, and there is no `create_workload_descriptor` and no `cached_program_t` / `create` / `override_runtime_arguments`. The factory satisfies `ProgramDescriptorFactoryConcept` (`ttnn/api/ttnn/operation_concepts.hpp:73-74`), not the legacy `ProgramFactoryConcept` (`:26-33`). `d67dde5ad66` deleted the legacy `create()` and `override_runtime_arguments()`. |
| `Custom hash` / `Backdoor custom hash` | `no` / `no` | none | `IsInDeviceOperation` has no `compute_program_hash` (`isin_device_operation.hpp:12-23`). `IsinParams` / `IsinInputs` have no `attribute_values` / `to_hash` (`isin_device_operation_types.hpp:13-23`). |
| `Runtime-args update (get_dynamic_runtime_args)` | `no` | none | Not on the device-op (`isin_device_operation.hpp:12-23`). |
| `Override runtime args method? (PD only)` | `n/a` | none (`no`) | Removed in `d67dde5ad66`. Its only job was refreshing the three buffer addresses, which the `Buffer*` bindings now do. The `n/a` follows from the stale `Concept`, so fix the two cells together. |
| `Pybind descriptor` | `no` | none | `isin_nanobind.cpp:36-45` binds only the free function, via `bind_function`. There is no `nb::class_` and no `create_descriptor` binding. |
| `Op-owned tensors?` | `no` | none | `descriptor` concept with no `buffers`. |
| `Secretly SPMD Workload?` / `Why secretly SPMD?` | blank | N/A | Not a mesh-shaped concept. |
| Factory set | 1 row: `IsInProgramFactory` | 1 factory: `IsInProgramFactory` | `isin_device_operation.hpp:17`. One-to-one; no phantom or missing row. |

The cross-column invariants hold (`get_dynamic_runtime_args` = `no`; `Op-owned tensors?` = `no`). `Smuggled pointer` = `no` also agrees with the code: the address RTAs are `Buffer*` bindings (see *Tensor bindings*).

**Attribution:** `Is able to port?` = `yes (with PD step)` and `Op Classification` = `Legacy Op` both follow from the stale `Concept`. None of the blocking columns fires (`get_dynamic_runtime_args`, smuggled pointer, `Known op issues`, relaxation). **Route:** the readiness-sheet owner (Diego, per `ttnn_op_porting_readiness.md`), to reconcile the row. The row carries `Diego validation` = `yes`; presumably it was validated before the 2026-09-29 migration landed.

### Device 2.0 (every kernel used) — GREEN

The op exercises `isin_reader.cpp` and `isin_writer.cpp`, plus the helper header `isin_common.hpp` that both include. All three are op-owned; there are no donor kernels.

- **NoC:** all DRAM traffic uses `Noc::async_read(TensorAccessor → CircularBuffer)` and `Noc::async_write(CircularBuffer → TensorAccessor)` with `{.page_id, .offset_bytes}` args (`isin_common.hpp:109-115`, `:134-140`), plus `noc.async_read_barrier()` / `async_write_barrier()`. The zero-fill uses `noc.async_write_zeros(cb, bytes)` + `noc.write_zeros_l1_barrier()` (`isin_reader.cpp:46-49`). There is no raw `noc_async_*`, no `get_noc_addr`, no addrgen, and no semaphore.
- **CBs:** every FIFO op and pointer peek is a `CircularBuffer` method (`isin_common.hpp:104-117`, `:129-142`; `isin_reader.cpp:62`, `:105-107`, `:126-130`, `:147-148`, `:160-165`). No `cb_*` free functions, and no `get_read_ptr(cb_id)` / `get_write_ptr(cb_id)`.

Two sites were considered and **not** counted as violations:

| File | Line | Call | Wrapper in scope | Verdict |
|---|---|---|---|---|
| `isin_reader.cpp` | 91 | `using elements_number_type = std_type_t<get_dataformat(ctas.elements_cb)>;` | No. `elements_cb` is constructed later, at `:105`. | **Not a holdover.** The call is a template argument, so it must be a constant expression. The only wrapper equivalent, `CircularBuffer::get_dataformat()` (`tt_metal/hw/inc/api/dataflow/circular_buffer.h:115`), is not `constexpr`, and neither is `explicit CircularBuffer(uint32_t)` (`:27`). So no wrapper method can replace the call here. The free function itself is `constexpr` (`dataflow_api.h:302`), and Device 2.0 has no other form for this use. |
| `isin_reader.cpp` | 28-29, 30-31, 33-34, 63-64 | `reinterpret_cast<volatile tt_l1_ptr T*>(addr)` on addresses from `CircularBuffer::get_read_ptr()` / `get_write_ptr()`: the compare loop in `isin_subchunks` and the invert pre-fill in `prefill_output` | Yes (the CB wrappers) | **Judged not gate-bearing.** The Device 2.0 guide's *Memory Access* section shows this cast as the legacy form of `CoreLocalMem<T>`. Against that: (a) it is neither a CB-index free function nor a data-movement idiom, so it fits neither Red sub-category. (b) The CB wrappers that Metal 2.0's `dfb::` tokens bind to are intact; the cast sits downstream of `get_*_ptr()`. (c) The Metal 2.0 CB→DFB whitelist marks exactly this shape as correct (`cb_dfb_api_whitelist.md:41-44`), and the post-port style doc lists it among the shapes ported kernels carry (`post_port/style/sync_free_dfbs.md:209-218`). (d) The Device 2.0 migration of these kernels (`7ff84036118`, #49334) left the casts in place. If the Device 2.0 track counts `CoreLocalMem` adoption as part of compliance, this becomes an **isolated** RED (4 sites, one kernel, op-code side). See Question 2. |

### Feature compatibility — GREEN (no gate fired)

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | Nothing in the op references the `GlobalCircularBuffer` type, `CreateGlobalCircularBuffer`, the `global_circular_buffer` field, or the `remote_index` / remote-CB idiom. The three `CBDescriptor`s (`isin_program_factory.cpp:31-39`) set only `total_size`, `core_ranges` and `format_descriptors`. |
| CBDescriptor `address_offset` (non-zero) | N/A | `address_offset` is never set, so it defaults to 0. There is no `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`, and no buffer-backed CB. |
| GlobalSemaphore | N/A | The op uses no semaphores of any kind. |

### CB endpoints (GATE-free)

All three CBs are 1-page and unconditional, over `all_cores` (`isin_program_factory.cpp:108-111`). The census is the same in every config: `invert` and dtype change *how* the reader fills `c_2`, not *who* touches it.

| CB | Touchers per node | Disposition |
|---|---|---|
| `c_0` `ELEMENTS` | **1**, the reader only. It FIFO-produces in `load_to_cb` (`isin_common.hpp:104-117`, called at `isin_reader.cpp:124-125`) and FIFO-consumes (`:126` `wait_front`, `:129` `get_read_ptr`, `:165` `pop_front`). The writer never touches it. | **self-loop** (reader as PRODUCER and CONSUMER) |
| `c_1` `TEST_ELEMENTS` | **1**, the reader only: `load_to_cb` (`isin_reader.cpp:141-146`), then `:147` `wait_front`, `:148` `get_read_ptr`, `:160` `pop_front`. | **self-loop** (reader) |
| `c_2` `OUTPUT` | **2**. The reader is the locked producer: `:128` `reserve_back`, `:130` / `:62` `get_write_ptr`, the raw fill / `async_write_zeros` at `:61-70`, the raw mask writes at `:33-39`, and `:164` `push_back`. The writer is the locked consumer, in `write_to_dram` (`isin_common.hpp:129-142`): `wait_front`, `noc.async_write(cb_obj, …)`, `pop_front`. | **legal 1:1**, no action |

Hunt results:
- No hidden second writer: there are no semaphores, and the writer makes no raw CB writes.
- No dual-instance work-split: the reader and writer are different sources.
- No dead CB, and no conditional DFB.

Both self-loops are DM self-loops. They are legal on Gen1 and rejected only at Quasar uplift, so they are recorded for that audit and do not block here.

### Offset base pointers — GREEN

There are three address RTAs. Each is a bare `Buffer*` base with no offset folded in (`isin_program_factory.cpp:145-147`): reader RTA0 `elements_buffer`, reader RTA1 `test_elements_buffer`, and writer RTA0 `output_buffer`. The per-core position travels in a separate scalar RTA (`subchunks_offset`), and the kernel applies it as the accessor's `offset_bytes` (`isin_common.hpp:108`/`:113`, `:133`/`:139`). Base and offset are already split.

isin is not in the triage tables (`analyses/2026-07-19_offset_base_pointers.md`), so the outcome is *no fold, op not in the tables → clean*. There is no Type 3 (`address_offset`) and no Type 4 (`narrow`).

### TensorAccessor 3rd argument — GREEN (3 sites, all Class 2 → drop)

The two classification questions resolve the same way for every site:

1. **Interleaved.** `elements` and `test_elements` are validated `!is_sharded()` (`isin_device_operation.cpp:25`, `:29`). The default output inherits `elements.memory_config()` (`:41`).
2. **Never used (`page_id ≡ 0`).** The op flattens every tensor to a single 1-D ROW_MAJOR page (`isin.cpp:76-97`; output spec at `isin_device_operation.cpp:40-41`). Every access is `page_id = FIRST_STICK = 0` plus a byte offset (`isin_common.hpp:52`, `:113`, `:139`).
   - So the stride never multiplies a nonzero page index. `InterleavedAddrGen::get_addr` computes `bank_offset_index * aligned_page_size + …`, and `bank_offset_index` is 0 (`tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h:289-294`, `:299`).
   - The hardware address-generator path is Quasar-only (`internal/tensor/generated_noc_addr.h:21-23`).
   - Point 2 holds for the sharded specialization too, so it does not depend on point 1.

| Site | Accessor | 3rd arg (host value) | True page | Class |
|---|---|---|---|---|
| `isin_reader.cpp:97-98` | `elements` | `ctas.elements_size * elements_element_size` = `volume × element_size` (CTAs 6, 10) | `volume × element_size` | **2**: (b) correct magnitude on an interleaved accessor, and (c) `page_id ≡ 0` |
| `isin_reader.cpp:99-102` | `test_elements` | `ctas.test_elements_size * test_elements_element_size`. `test_elements_element_size` is the *elements* datum size (`:94`), which equals the test-elements datum size because of the dtype-equality check (`isin_device_operation.cpp:19-23`). | `volume × element_size` | **2**: (b) + (c) |
| `isin_writer.cpp:19` | `output` | `ctas.elements_size`, an element **count** (CTA 6), not bytes | `4 × volume` B (uint32 mask) | **2, by (c) only.** The units are wrong (4× too low, and realignment can't repair that), but the stride is structurally never used. |

This is not Class 1. The values are compile-time (CTA-derived), and the relaxation is `none`. The kernel comments claiming the override guards against a stale `AlignedPageSize` on cache hits (`isin_reader.cpp:95-96`, `isin_writer.cpp:17-18`) are inaccurate. The arg comes from CTAs, not RTAs, and the default hash covers the tensor specs, because a `Tensor` hashes as `(storage, tensor_spec)` (`ttnn/api/ttnn/tensor/tensor.hpp:252-253`). Do not set `dynamic_tensor_shape`.

*Cross-check against `analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md`:* the doc lists `isin` as Class 2, "(b) … ¶ additionally (c)", and calls the sites "all interleaved + correct-magnitude". The class agrees. The "(b) correct-magnitude" reasoning does not hold for the writer site, though: it has passed an element count since the op landed (`0f2f1c09ac9`). That site is Class 2 by (c) alone. FYI for the triage-doc owner.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings.** All three are `Buffer*` entries in `emplace_runtime_args` (`isin_program_factory.cpp:145-147`). The framework patches those on cache hits today (`tt_metal/api/tt-metalium/program_descriptors.hpp:116-124`), so they are not the stale-pointer hazard.
  - `elements` — **Case 1.** Reader RTA0 feeds `TensorAccessor(ctas.elements_accessor_args, …)` (`isin_reader.cpp:86`, `:97-98`). All access goes through `noc.async_read`.
  - `test_elements` — **Case 1.** Reader RTA1 (`isin_reader.cpp:87`, `:99-102`).
  - `output` — **Case 1.** Writer RTA0 (`isin_writer.cpp:12`, `:19`). All access goes through `noc.async_write`. This binds the device-op's return tensor, which is `*optional_out` when the caller passes `out` (`isin_device_operation.cpp:47-51`).
  - These disappear with the bindings: the three `get_arg_val` address reads, and the `TensorAccessorArgs(*buffer).append_to(...)` plumbing (`isin_program_factory.cpp:90-92`; kernel side `isin_common.hpp:77-79`).
- **TensorParameter relaxation:** `none`. Strict `TensorSpec` matching on all three bindings.
- **TensorAccessor 3rd arg:** drop it at `isin_reader.cpp:97-98`, `isin_reader.cpp:99-102` and `isin_writer.cpp:19`. All three are Class 2; do **not** set `dynamic_tensor_shape`.
- **CB endpoints:**
  - Self-loop `c_0` `ELEMENTS` and `c_1` `TEST_ELEMENTS`, with the reader as both PRODUCER and CONSUMER, in all configs.
  - `c_2` `OUTPUT` is a legal 1:1: reader PRODUCER, writer CONSUMER.
  - No multi-binding, no dead-CB drop, no conditional DFB.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes):** none.
- **Cross-op / shared kernels:** none. Both kernel sources are op-owned and bound only by `IsInProgramFactory`; a filename census finds them referenced only at `isin_program_factory.cpp:114` and `:116`. No `_metal2` fork exists, and none is needed, because no other factory binds them. There is no sunset list.
- **RTA varargs:** none.
  - RTAs are read at fixed indices: reader `0-3` (`isin_reader.cpp:86-89`), writer `0-2` (`isin_writer.cpp:12-14`).
  - Once the address RTAs become tensor bindings, each kernel keeps two named per-node RTAs, `subchunks_per_core` and `subchunks_offset`. Their values vary by core group and running offset (`isin_program_factory.cpp:135-150`), so they stay RTAs.
  - CTAs are read at fixed indices (`isin_common.hpp:81-89`), so there are no CTA varargs.
- **`get_dataformat` in a template argument** (`isin_reader.cpp:91`): keep the free function and pass it the binding token, `get_dataformat(dfb::<elements>)`.
  - The `DataflowBuffer::get_dataformat()` getter (`dataflow_buffer.h:348`) can't produce a constant expression, because no `DataflowBuffer` can be `constexpr` (`cb_dfb_api_whitelist.md:93-107`).
  - The line has no `constexpr` keyword, but its template-argument position is exactly the constant-expression context that rule protects (Recipe note 4).
  - Record the site in the port report as Gen1-only token-form debt.
- **Shared kernel-side CTA decoder:** both kernels call `IsInCTAs` / `get_ctas()` (`isin_common.hpp:56-93`; called at `isin_reader.cpp:83` and `isin_writer.cpp:10`). It decodes the entire positional block: all three CB indices (CTAs 3-5) and all three `TensorAccessorArgs` (from CTA 12).
  - After the port, each kernel binds only its own resources. The reader binds DFBs `elements` / `test_elements` / `output` and tensors `elements` / `test_elements`; the writer binds DFB `output` and tensor `output`.
  - So the helper must not reference a token the calling kernel doesn't bind. The file is op-private, so reshaping it affects no other op.
- **Dead CTA placeholders:** CTA slots 0-2 are `0u` (`isin_program_factory.cpp:75-80`). They are never read and exist only to preserve the legacy positional offset. Named CTAs dissolve them with no functional change. The "index 11" comments at `:75` and `isin_common.hpp:54-55` are stale; the accessor args start at 12 (`isin_common.hpp:77`).
- **Helpers re-wrap the CB on every call:** `load_to_cb` / `write_to_dram` (`isin_common.hpp:96-143`) take a `uint32_t` CB index and build their own `CircularBuffer`, while `kernel_main` holds a second wrapper over the same CB (`isin_reader.cpp:105-107`).
  - On Gen1 a `DataflowBuffer` is a view over the shared `LocalCBInterface` (`tt_metal/hw/inc/internal/tt-1xx/dataflow_buffer.inl:25-26`). Building it from the token on each call and passing `kernel_main`'s object by reference therefore behave identically.
  - On Quasar, DFBs carry per-object state and drain in their destructor (`dataflow_buffer.h:158-162`). That makes this Quasar-uplift debt only.
- **Raw L1 casts stay as they are** (`isin_reader.cpp:28-34`, `:63-64`, on `get_read_ptr()` / `get_write_ptr()` results). Device 2.0 idioms are off the port's whitelist, and the shape is correct on a DFB (`cb_dfb_api_whitelist.md:41-44`).
- **Stale include and handle names:** the kernels include `api/dataflow/circular_buffer.h` (`isin_common.hpp:7`, `isin_reader.cpp:8`) and use `*_cb` / `cb_obj` handle names. The port recipe replaces both; don't carry either into the ported files.
- **Kernel configs:** `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` (`isin_program_factory.cpp:123`, `:130`) preserve the legacy reader/writer RISC and NOC choice (#57541). Carry the same assignment into the port.

## TTNN ProgramFactory

### Concept
`ProgramSpecFactoryConcept` (the sheet's `Porting Target`). There is no `override_runtime_arguments` to translate.

### Fit
- Execution model: `SPMD` — one program, not mesh-shaped.
- Op-owned device resources: none (no op-owned tensors, no GlobalSemaphores).
- Tensor-arg matching: strict (relaxation `none`).
- Ported-from shape: 1:1. One `descriptor` (`create_descriptor`, `isin_program_factory.cpp:42`) becomes one `ProgramSpec`.

### Custom compute_program_hash
None. The op uses the default reflection-based hash over `operation_attributes` + `tensor_args` (`ttnn/api/ttnn/device_operation.hpp:83-87`).

### Stop signals
None on the TTNN-integration side. The overall RED comes from the stale readiness row, not from a missing framework capability.

**Device-op-class edits the port forces:** none. There is no pybound `create_descriptor`, no pybind-only factory parameter, and this is not a direct-descriptor op (it has a `program_factory_t`).

## Team-only

### Out-of-directory coupling & donor shape — ✓ clean
- **Roll-up:** ✓ clean. Every include outside the op resolves to `tt_metal/*` (class 1: LLK/HAL/firmware, no concern). Nothing comes from `kernel_lib/`, `kernel/`, `kernel_helper_functions/`, or an in-family or cross-family donor.
- **Summary table:**

  | Op kernel | Donor file | Class | Status |
  |---|---|---|---|
  | `isin_reader.cpp` | `api/dataflow/dataflow_api.h`, `api/dataflow/circular_buffer.h` | 1 `tt_metal/*` | ✓ |
  | `isin_writer.cpp` | `api/dataflow/dataflow_api.h` | 1 `tt_metal/*` | ✓ |
  | `isin_common.hpp` (in-op, both kernels) | `api/dataflow/circular_buffer.h`, `api/dataflow/noc.h`, `api/core_local_mem.h`, `api/tensor/noc_traits.h` | 1 `tt_metal/*` | ✓ |

- **Per-call detail:** omitted, since every row is ✓.
- **Borrowed kernel files:** none. Both kernel sources are op-owned and not lent to any other op, and no `_metal2` fork sits beside them.

### Relaxation candidates
None: the op has no custom hash to mine.

### TTNN factory analysis
The sheet-derived facts and their code evidence are in the cross-check table under *Gate detail*. In short: the concept is `descriptor` (the sheet is stale); the execution model is SPMD; there are no op-owned tensors, no custom hash, no `get_dynamic_runtime_args`, no `override_runtime_arguments` and no pybound `create_descriptor`; the target is `ProgramSpecFactoryConcept`.

## Misc anomalies  *(team-only, non-gating → ops team)*

- **Dead-but-hashed attribute `assume_unique`.** The factory never reads `IsinParams::assume_unique` (`isin_device_operation_types.hpp:14`); it reads only `invert` and `single_fetch_subchunk_size` (`isin_program_factory.cpp:52-53`). The attribute is still part of the default program hash (`device_operation.hpp:83-87`), so calls that differ only in `assume_unique` build duplicate programs. The binding's own docstring calls it a no-op (`isin_nanobind.cpp:24`).
- **Wrong-units 3rd arg (inert).** `isin_writer.cpp:19` passes `ctas.elements_size`, an element count, as the output page size; the true page is `4 × volume` bytes. It is harmless only because `page_id ≡ 0`. This is the same family of units slip that #57656 (`73fb8710860`) fixed elsewhere in this op. The Metal 2.0 port drops the arg anyway.
- **Preallocated `out` is barely validated.** `create_output_tensors` checks only `out.dtype() == UINT32` (`isin_device_operation.cpp:47-51`), and the wrapper passes `opt_out` through without flattening it (`isin.cpp:133-139`).
  - The kernels assume a 1-D, single-page, interleaved ROW_MAJOR output: the writer puts every mask word at `page_id 0` plus a byte offset (`isin_common.hpp:134-139`).
  - A multi-D (e.g. original-shape), TILE or sharded `out` would therefore be mis-addressed. The result would be wrong output, and possibly writes past page 0 in bank 0.
  - No test passes `out=` (`tests/ttnn/unit_tests/operations/experimental/test_isin.py`, `tests/ttnn/nightly/unit_tests/operations/experimental/test_isin.py`).
  - This is a static finding; I did not reproduce it.
- **Misleading comments.**
  - `isin_reader.cpp:95-96` and `isin_writer.cpp:17-18` say the 3rd arg comes "from runtime args" and guards against a stale `AlignedPageSize` on cache hits. It actually comes from CTAs, and the default hash rules out the staleness.
  - The "TensorAccessorArgs remains at index 11" comments (`isin_program_factory.cpp:75`, `isin_common.hpp:54-55`) have been stale since `73fb8710860` added CTA 11. The accessor args start at 12.

## Questions for the user

1. **Reconciling the readiness sheet:** the `experimental/isin` row still reads `Concept` = `legacy device-op` and `Is able to port?` = `yes (with PD step)`, but the factory has been `descriptor` since `d67dde5ad66` (#57541, 2026-09-29). Can you raise this with Diego (the sheet owner), or update the row yourself if you have edit access? Once `Is able to port?` reads `yes`, I expect a quick re-audit to clear and issue the brief, because the code-side subjects in this report are complete.
2. **Do raw L1 casts fall under Device 2.0?** You authored the Device 2.0 migration of these kernels (#49334). Does the Device 2.0 track treat `reinterpret_cast<volatile tt_l1_ptr T*>` over `CircularBuffer::get_*_ptr()` as compliant? I assumed yes; the reasons are under *Gate detail → Device 2.0*. If adopting `CoreLocalMem<T>` is part of compliance, the Device 2.0 gate becomes an isolated, op-code-side RED on `isin_reader.cpp:28-34` and `:63-64`.

## Recipe notes

1. **`yes (with PD step)` alongside a `Concept` conflict.** The routing list has one row for `Is able to port?` == `yes (with PD step)` (→ legacy-API short-circuit, skip the informational subjects) and another for a primary-column conflict (→ spreadsheet broken → sheet owner). Here both match at once: the sheet says legacy, but the code is already `descriptor`.
   - I took the conflict row. It rests on code evidence, and the short-circuit's premise ("the op isn't on the `ProgramDescriptor` API yet") is false here. I then ran the informational subjects under the "RED that clears without touching the op's code" exception.
   - Suggestion: state explicitly in the routing list that a legacy cell contradicted by a `descriptor` factory is spreadsheet-broken, not a short-circuit case.
   - Related: on such a row, the "PD only" `Override runtime args method?` column reads `n/a`. That is a knock-on of the same staleness, not a second conflict, and the recipe could say so.
2. **The "Feature Support — Variadic-CTA" template row has no Appendix A entry.** Appendix A lists only GlobalCircularBuffer, `address_offset` and GlobalSemaphore, and the *RTA varargs* subject now treats CTA varargs as ordinary port work (`KernelAdvancedOptions::compile_time_varargs`). The template row looks stale; I filled it in as "Ok".
3. **Does the Device 2.0 gate include the guide's *Memory Access* section?** The gate defers to the Device 2.0 migration guide for what compliance means. That guide lists a raw `reinterpret_cast<volatile uint32_t*>` as the legacy form of `CoreLocalMem<T>`. However, neither of the gate's Red sub-categories mentions it, and the Metal 2.0 docs treat the cast on a DFB pointer as correct (`cb_dfb_api_whitelist.md`'s access-control ✅; `sync_free_dfbs.md`'s "ported kernels carry … raw cast"). Please state whether raw local-L1 casts on wrapper-supplied pointers are in gate scope.
4. **CB-index free function in a constant-expression context.** The holdover test ("wrapper in scope *and* a wrapper-method replacement exists") needed interpretation for `get_dataformat(cb)` used as a template argument: the wrapper getter exists but is not `constexpr`.
   - Suggestion: add "a replacement that can't appear in the call's context (template argument, array bound, `if constexpr`) does not count as a replacement".
   - The same gap affects port rule 7 and `cb_dfb_api_whitelist.md` §A, which make "`constexpr` on the line you are replacing" the entire test for token form vs getter. A call used directly as a template argument (`using T = std_type_t<get_dataformat(cb)>;`) carries no `constexpr` keyword, yet the getter can't compile there. Suggestion: key the test on a constant-expression context, not on the keyword.
5. **Two places prescribe the TTNN section.** `ttnn_factory.md` ("Audit report deliverable") asks for a `## TTNN ProgramFactory` section, while this recipe's template puts a "TTNN factory analysis" bullet under *Team-only*. I emitted both, with the bullet pointing back to the cross-check table. The template could reference the `ttnn_factory.md` section directly.
