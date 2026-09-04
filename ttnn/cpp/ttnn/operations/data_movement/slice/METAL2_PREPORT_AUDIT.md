# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/slice`

One DeviceOperation, five program factories, all in `device/`:

- **`ttnn::prim::SliceDeviceOperation`** (`slice_device_operation.hpp/.cpp`)
  - `SliceRmProgramFactory` (`slice_program_factory_rm.cpp`)
  - `SliceRmShardedProgramFactory` (`slice_program_factory_rm_sharded.cpp`)
  - `SliceRmStrideProgramFactory` (`slice_program_factory_rm_stride.cpp`)
  - `SliceTileProgramFactory` (`slice_program_factory_tile.cpp`)
  - `SliceTileTensorArgsProgramFactory` (`slice_program_factory_tile_tensor_args.cpp`)

Eleven kernels are referenced across the five factories (ten owned by slice, one borrowed from `eltwise/unary`). Two kernel files in `device/kernels/dataflow/` are **referenced by no factory** and are therefore out of scope — `strided_slice_reader_rm_interleaved_nd.cpp` and `strided_slice_writer_rm_interleaved.cpp` (no `kernel_source` in the tree names either; their contents were not audited).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `1167faf7b42 2026-09-04 docs(metal_2.0): binary_ng relaxation analysis; invariant checks over commit stamps`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/slice/` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `SliceDeviceOperation` → `SliceRmProgramFactory`, `SliceRmShardedProgramFactory`, `SliceRmStrideProgramFactory`, `SliceTileProgramFactory`, `SliceTileTensorArgsProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 11 referenced kernels are structurally Device 2.0 (`Noc`, `DataflowBuffer`, `CoreLocalMem`, `TensorAccessor`). One free-function site is on the sanctioned list; see Gate detail. |
| *Prereqs* — Cross-op escapes | Ok — one in-family header (`data_movement/common/kernels/common.hpp`, Shape 1 / ✓) and one borrowed cross-family kernel file (`eltwise/unary/.../writer_unary_interleaved_start_id.cpp`, `_metal2` fork already exists). |
| *Feature Support* — overall | **GREEN** (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | Ok — no `get_compile_time_arg_val` at a varying index anywhere; not an Appendix A entry in any case |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** — `yes` on all five factory rows |
| *TTNN Readiness* — Concept (current) | `descriptor` (all five) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A (concept is `descriptor`) |
| *TTNN Readiness* — Custom hash | **Yes** (not a gate; port leaves it intact): `slice_device_operation.cpp:348` |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No — absent from the device-op (grep clean); sheet agrees |
| *TTNN Readiness* — `override_runtime_arguments` | **Yes** (not a gate; selects `CustomProgramSpecFactoryConcept`) — one per factory: `slice_program_factory_rm.cpp:424`, `slice_program_factory_rm_sharded.cpp:415`, `slice_program_factory_rm_stride.cpp:178`, `slice_program_factory_tile.cpp:189`, `slice_program_factory_tile_tensor_args.cpp:195`. All five delegate to the shared `patch_slice_program_addresses` (`slice_program_factory_rm_sharded.cpp:354`). |
| *TTNN Readiness* — Pybind `create_descriptor` | **Yes** (not a gate; port deletes the binding): `slice_nanobind.cpp:168-179` (`SliceTileProgramFactory::create_descriptor`). Sheet cell reads `PR`. |
| *TTNN Readiness* — Op-owned tensors | No (cell blank; `descriptor` concept cannot carry them — invariant holds) |
| *TTNN Readiness* — Target concept | **`CustomProgramSpecFactoryConcept`** (all five factories) |
| *Port work* — Offset base pointer | **none** — no address RTA folds a host-side offset into its base. The fold the 2026-07-19 triage catalogued has already been split out by the ops team; see Gate detail. |
| *Port work* — Tensor bindings (per binding) | Case 1 ×8 · clean (borrowed-DFB) ×2 — no Case 2 anywhere |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) on all five rows |
| *Port work* — TensorAccessor 3rd arg | **none** — no accessor in the op passes a 3rd argument (12 construction sites, all two-arg) |
| *Port work* — CB endpoints | legal ×4 · self-loop ×3 — no multi-binding, no dead CB, no conditional DFB |

**CB endpoints** are dispositions, not gates (see the recipe's *CB endpoints* subject): every out-of-window CB has a port-time resolution. Recorded per `(CB, config)` below. Slice's CB census does **not** flip with config within a factory — the RM factory's chunked-vs-unchunked branch changes CB *sizing* only, not endpoint count — so each factory has a single disposition set.

## Result

**GREEN → brief issued.** All five gate-bearing subjects clear: Device 2.0, Feature compatibility, TTNN factory concept, Offset base pointers, TensorAccessor 3rd argument. `METAL2_PORT_BRIEF.md` is written alongside this file.

Two findings deserve the reader's attention even though neither blocks:

1. **Both dated triage analyses are stale for `slice`, in the same direction.** The 2026-07-19 offset-base-pointer doc names `slice_program_factory_rm.cpp` as "the canonical case" of a Type-2 accessor-fed offset, and the 2026-07-06 3rd-arg doc classes slice as "1 — Dynamic page size (+ **S** base-offset)". Neither survives contact with current `main`: the offset is now a separate scalar arg added per-read inside the kernel, and no accessor passes a page-size argument. Per the recipe's reconciliation rules (*"No fold, op in the tables → the doc is stale → GREEN"*, and *"trust your read"* for the 3rd arg), both resolve GREEN. Both docs should be updated for slice.
2. **`ccl/mesh_partition` drives these factories directly** and shares `patch_slice_program_addresses`. That is a host-side cross-op coupling the port will break if it is not carried along. It is the single highest-risk item for the porter and is called out prominently in the brief.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** All five factory rows read `yes`. Cross-check against the code came back clean on every checkable column:

  | Column | Sheet | Code | Verdict |
  |---|---|---|---|
  | `Concept` | `descriptor` ×5 | each factory defines `create_descriptor(...) -> ProgramDescriptor` | ✓ |
  | `Custom hash (compute_program_hash)` | `yes` ×5 | `slice_device_operation.cpp:348` | ✓ |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` ×5 | absent from the device-op (grep over the whole op dir) | ✓ |
  | `Override runtime args method? (PD only)` | `yes` ×5 | one per factory (sites in Status summary) | ✓ |
  | `Pybind descriptor (nb::class_ of device op)` | `PR` ×5 | present: `slice_nanobind.cpp:168-179` | ✓ (consistent — `PR` = in-flight PR, and a binding exists) |
  | `Secretly SPMD Workload?` | blank | N/A on a `descriptor` concept | ✓ |
  | **Factory-set match** | 5 rows | 5 factories in `SliceDeviceOperation::program_factory_t` (`slice_device_operation.hpp:36-41`) | ✓ one-to-one, no phantom or missing row |

  Cross-column invariants hold: no `get_dynamic_runtime_args` on any row (so the `descriptor`-only constraint is vacuous), and `Op-owned tensors?` is blank on every `descriptor` row.

- **Device 2.0 (every kernel used):** **GREEN.** All 11 referenced kernels use `Noc`, `DataflowBuffer`, `CoreLocalMem<uint32_t>`, `UnicastEndpoint` and `TensorAccessor` throughout. A scan for legacy idioms across all 11 files returned **zero** hits for `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedPow2AddrGen*`, bare `cb_wait_front` / `cb_push_back` / `cb_pop_front` / `cb_reserve_back`, free-function `noc_async_read(` / `noc_async_write(` / `noc_async_*_barrier(`, `noc_semaphore_*`, `get_noc_addr_from_bank_id`, or a stale `api/dataflow/circular_buffer.h` include.

  One CB-index free-function site exists, and it is **sanctioned** — not a violation, not a holdover:

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | 27 | `get_local_cb_interface(cb_id_out).fifo_page_size` | `DataflowBuffer dfb(cb_id_out)` (line 30) |

  This is precisely the misfire the recipe warns about — a `DataflowBuffer` is in scope, and a wrapper replacement demonstrably exists (slice's own near-identical copy at `device/kernels/dataflow/writer_unary_interleaved_start_id.cpp:26` already uses `dfb_out.get_entry_size()`). Per the Green bullet, `get_local_cb_interface(cb_id)` is on the sanctioned list and the list is the whole test, so this does **not** RED the gate. It is instead a **port-stage** change under kernel-side whitelist rule 7 — carried to the brief as a breadcrumb.

- **Feature compatibility:** every Appendix A entry, in order. All absent.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | Zero hits across the op for `GlobalCircularBuffer`, `CreateGlobalCircularBuffer`, `.global_circular_buffer`, `remote_index(`, `remote_cb`, `remote_circular_buffer.h`, or the 3-arg `UpdateDynamicCircularBufferAddress`. The two buffer-backed CBs in `SliceRmShardedProgramFactory` (`slice_program_factory_rm_sharded.cpp:290,302`) set `.buffer` only — the ordinary borrowed-memory pattern, which is a mechanical porting-recipe translation via `DataflowBufferSpec::borrowed_from`, not this entry. |
  | CBDescriptor `address_offset` (non-zero) | N/A | Zero hits for `address_offset`, `set_address_offset`, `cb_descriptor_from_sharded_tensor`, `cb_descriptor_from_overlapped_tensor`, or the 4-arg `UpdateDynamicCircularBufferAddress`. No `CBDescriptor` in the op sets the field, so it defaults to zero. |
  | GlobalSemaphore | N/A | Zero hits for `GlobalSemaphore` / `CreateGlobalSemaphore` / `global_semaphore.hpp`. Slice uses **no semaphores at all** — no `CreateSemaphore`, no `SemaphoreDescriptor`, no `.semaphores` on any `ProgramDescriptor`. |

- **CB endpoints (GATE-free):** every CB is either a plain legal 1:1 or a one-toucher self-loop. No multi-binding, no dead CB, no config-conditional DFB.

  | Factory | CB | Backing | Touchers on a node | Verdict | Disposition |
  |---|---|---|---|---|---|
  | `SliceRmProgramFactory` | `c_0` (`src0_cb_index`) | local (`rm.cpp:350`) | reader = locked PRODUCER (`reserve_back`/`push_back`, `slice_reader_..._rm_interleaved_start_id.cpp:60,71,90,117`); writer = locked CONSUMER (`wait_front`/`pop_front`, `slice_writer_..._interleaved_start_id.cpp:47,57,67,80`) | plain 1:1 | none — legal |
  | `SliceRmShardedProgramFactory` | `c_0` (`src0_cb_index`) | **borrowed** from `input.buffer()` (`rm_sharded.cpp:290`) | reader only, **raw peek** (`dfb_in.get_write_ptr()`, `slice_reader_..._rm_sharded.cpp:41`) — no FIFO ops → role-free | 1 toucher | **self-loop** (bind reader PRODUCER **and** CONSUMER) |
  | `SliceRmShardedProgramFactory` | `c_16` (`output_cb_index`) | **borrowed** from `output.buffer()` (`rm_sharded.cpp:302`) | reader only, locked PRODUCER (`reserve_back`/`push_back`, `slice_reader_..._rm_sharded.cpp:40,89`) — nothing drains it | 1 toucher | **self-loop** |
  | `SliceRmStrideProgramFactory` | `c_0` (`in_cb`) | local (`rm_stride.cpp:69`) | reader = locked PRODUCER; writer = locked CONSUMER (4D: `reader_multicore_slice_4d.cpp:152,179` / `writer_multicore_slice_4d.cpp:81,93`; ND: `reader_multicore_slice_nd.cpp:137,166` / `writer_multicore_slice_nd.cpp:88,100`) | plain 1:1 | none — legal (both rank branches) |
  | `SliceTileProgramFactory` | `c_0` (`src0_cb_index`) | local (`tile.cpp:53-60`) | reader = locked PRODUCER (`reader_unary_unpad_dims_interleaved_start_id.cpp:39,42`); writer = locked CONSUMER (`writer_unary_interleaved_start_id.cpp:45,48`) | plain 1:1 | none — legal |
  | `SliceTileTensorArgsProgramFactory` | `c_0` (`src0_cb_index`) | local (`tile_tensor_args.cpp:56`) | reader = locked PRODUCER (`reader_..._tensor_args.cpp:117,120`); writer (borrowed eltwise kernel) = locked CONSUMER | plain 1:1 | none — legal |
  | `SliceTileTensorArgsProgramFactory` | `c_1` (`tensor_cb_index`) | local (`tile_tensor_args.cpp:65`) | reader only, and it drives **both** FIFO roles — `reserve_back`/`push_back`/`wait_front`/`pop_front` on the same kernel (`reader_..._tensor_args.cpp:52,58,59,66` and again `69,75,76,83`) | 1 toucher | **self-loop** |

  The `c_1` case is the textbook single-kernel staging buffer — the kernel's own comment calls it "the producer/consumer handshake (reserve -> push -> wait -> pop)". Both `SliceRmShardedProgramFactory` self-loops land on a **DM** kernel (`ReaderConfigDescriptor`), which is legal on Gen1; record as Quasar-uplift debt for that later audit, not a Gen1 concern.

  No CB anywhere in the op has a zero-toucher census, and no raw co-fill by a second kernel was found: every kernel that calls `get_write_ptr()` / `get_read_ptr()` on a DFB is that DFB's own bound endpoint (a peek on a binding it already holds), and there are no semaphores to coordinate a hidden co-fill with.

- **Offset base pointers:** **GREEN** — no address RTA in any factory folds a host-side offset into its base.

  The 2026-07-19 triage doc lists `slice` / `slice_program_factory_rm.cpp` as a **Type 2** (accessor-fed offset), "the canonical case", with the offset expression `input->address() + begins_bytes − misalignment`. **That fold is no longer present.** Current `main`:

  - The input base reaches the reader as a plain `Buffer*` binding at RTA slot 0 — `reader_args.push_back(src0_buffer)` (`slice_program_factory_rm.cpp:405`), with no arithmetic. The output base likewise: `writer_args.push_back(dst_buffer)` (`:413`).
  - The W-begin shift is now a **separate scalar** RTA: `begins_bytes - misalignment` is the last element of `common_reader_kernel_args` (`slice_program_factory_rm.cpp:99`), arriving as kernel arg 12, `src_offset_bytes`.
  - The kernel keeps the accessor on the clean base and adds the shift per-read: `TensorAccessor(src_args, src_addr)` (`slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp:40`), then `noc_async_read_sharded(noc, …, s0, src_stick_id, /*offset=*/src_offset_bytes, …)` (`:98`, and `:64,67` on the chunked path).
  - The kernel carries an explicit comment recording the Metal 2.0 rationale for the split (`:38-39`): *"The accessor base stays the unshifted buffer base: Metal 2.0 supplies it from the tensor binding and offers no seam for a pre-offset base. The W-begin shift rides each read as `src_offset_bytes`."*

  This is the recipe's **"No fold, op in the tables → the doc is stale"** outcome, and the fix has landed on exactly the shape the Type-2 remedy discussion contemplated (base binding + kernel-side offset). The RTA drops to ordinary Case-1 tensor-binding port work.

  The other four factories were scanned on the same question and are clean for independent reasons:

  | Factory | Base delivery | Offset carrier | Verdict |
  |---|---|---|---|
  | `SliceRmProgramFactory` | `Buffer*` RTA[0] (reader + writer) | separate RTA `src_offset_bytes` (arg 12) | no fold |
  | `SliceRmShardedProgramFactory` | none — both tensors reach the kernel as **borrowed-memory DFBs** | `begins_bytes` is a **CTA** (`rm_sharded.cpp:310`), added to an L1 offset kernel-side (`slice_reader_..._rm_sharded.cpp:69`) | no device pointer to fold into |
  | `SliceRmStrideProgramFactory` | `Buffer*` RTA[0] (reader + writer) | `slice_start[*]` passed as plain scalars; row index computed kernel-side | no fold |
  | `SliceTileProgramFactory` | `Buffer*` CRTA[0] (reader) / RTA[0] (writer) | `get_tiled_start_offset()` result folded into `start_id`, a **tile index**, not an address (`tile.cpp:97,119,125`) | no fold (Type-note: this is the clean tile-index scalar the triage doc itself calls out as unaffected) |
  | `SliceTileTensorArgsProgramFactory` | `Buffer*` CRTA[0..2] (reader) / RTA[0] (writer) | `start_offset` is `constexpr 0` on the host (`tile_tensor_args.cpp:129`); the real offset is computed **on device** from the start/end tensors (`reader_..._tensor_args.cpp:85-112`) | no fold |

  Type 3 (`address_offset`) is absent — see the Appendix A row. Type 4 (`narrow`) does not appear in this op.

- **TensorAccessor 3rd argument:** **N/A** — no accessor in the op passes a 3rd argument. All **12** construction sites across the 11 referenced kernels are the two-arg form `TensorAccessor(args, addr)`; the subject never fires.

  This contradicts the 2026-07-06 triage doc, which lists `slice` (interleaved RM path) as "1 — Dynamic page size (+ **S** base-offset)" and, in its Special table, as *"sub-page base offset … stays on raw addressing."* Per the recipe's *"trust your read"* contract for this dated doc, my read stands. The change is documented in the factory itself (`slice_program_factory_rm.cpp:283-290`): *"Both RM kernels build their TensorAccessor from the two-arg form, so each takes the aligned page size `TensorAccessorArgs` bakes into the compile-time args."* The factory pins the equivalence with a runtime assertion — `check_accessor_page_size()` (`:291-307`), called for input and output at `:336-339` — which fails loudly if a caller route reaches the factory with a per-shard page size that disagrees with the accessor's aligned page size. The doc's Special "base-offset" concern is the same one resolved under *Offset base pointers* above.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per factory) — **eight Case 1, two clean, zero Case 2**:

  | Factory | Binding | Delivery today | Kernel use | Case |
  |---|---|---|---|---|
  | `SliceRmProgramFactory` | `input` | `Buffer*` RTA[0] (`rm.cpp:405`) | `TensorAccessor(src_args, src_addr)` | **1** |
  | `SliceRmProgramFactory` | `output` | `Buffer*` RTA[0] (`rm.cpp:413`) | `TensorAccessor(dst_args, dst_addr)` | **1** |
  | `SliceRmShardedProgramFactory` | `input` | borrowed-memory DFB `c_0` (`rm_sharded.cpp:290`) | `dfb_in.get_write_ptr()` | **clean** |
  | `SliceRmShardedProgramFactory` | `output` | borrowed-memory DFB `c_16` (`rm_sharded.cpp:302`) | `dfb_out` FIFO | **clean** |
  | `SliceRmStrideProgramFactory` | `input` | `Buffer*` RTA[0] (`rm_stride.cpp:128,147`) | `TensorAccessor(src_args, src_addr)` | **1** |
  | `SliceRmStrideProgramFactory` | `output` | `Buffer*` RTA[0] (`rm_stride.cpp:136,160`) | `TensorAccessor(dst_args, dst_addr)` | **1** |
  | `SliceTileProgramFactory` | `input` | `Buffer*` **CRTA**[0] (`tile.cpp:143`) | `TensorAccessor(src_args, src_addr)` | **1** |
  | `SliceTileProgramFactory` | `output` | `Buffer*` RTA[0] (`tile.cpp:180`) | `TensorAccessor(dst_args, dst_addr)` | **1** |
  | `SliceTileTensorArgsProgramFactory` | `input` | `Buffer*` **CRTA**[0] (`tile_tensor_args.cpp:182`) | `TensorAccessor(src_args, src_addr)` | **1** |
  | `SliceTileTensorArgsProgramFactory` | `start_tensor` | `Buffer*` **CRTA**[1] (`tile_tensor_args.cpp:183`) | `TensorAccessor(start_args, start_addr)` | **1** |
  | `SliceTileTensorArgsProgramFactory` | `end_tensor` | `Buffer*` **CRTA**[2] (`tile_tensor_args.cpp:184`) | `TensorAccessor(end_args, end_addr)` | **1** |
  | `SliceTileTensorArgsProgramFactory` | `output` | `Buffer*` RTA[0] (`tile_tensor_args.cpp:151,168`) | `TensorAccessor(dst_args, dst_addr)` (borrowed eltwise writer) | **1** |

  Note the delivery mechanism: **every** address in this op travels as a `Buffer*` pushed into an `RTArgList` / `emplace_runtime_args`, never as a bare `->address()` in a descriptor's arg list. Per the recipe, that shape is the framework's interim binding-injection hack — *correct on cache hits today*, not the silent-wrong hazard. So the per-binding work below is **routine port work with no correctness urgency**; the classification still matters because the kernel-side rewrite differs by case, and here every non-clean binding is the mechanical Case 1.

  The `SliceRmShardedProgramFactory` split is the recipe's *"classification can vary per factory"* case: the same `input` / `output` tensors are clean (borrowed-memory DFB) in the sharded factory and Case 1 everywhere else.

- **TensorParameter relaxation:** `none` on all five rows — clears; the port applies no relaxation. (No `analyses/relaxations/data_movement_slice.md` exists, and none is required for a `none` cell.)
- **TensorAccessor 3rd arg:** none — no site passes one.
- **CB endpoints:** self-loop `(c_0, SliceRmShardedProgramFactory)`, `(c_16, SliceRmShardedProgramFactory)`, `(c_1, SliceTileTensorArgsProgramFactory)`; all four remaining CBs are legal 1:1. No multi-binding flag, no dead-CB drop, no conditional DFB.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. No hidden second writer exists — the op has no semaphores, and every raw `get_write_ptr()` / `get_read_ptr()` call is by the DFB's own bound endpoint.
- **Cross-op / shared kernels:**
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — cross-family borrow, instantiated by `SliceTileTensorArgsProgramFactory` (`tile_tensor_args.cpp:133`). A **`_metal2` fork already exists beside it** (`writer_unary_interleaved_start_id_metal2.cpp`, same directory, *not* under `experimental/quasar/`) — the porter binds the existing fork rather than creating one. The legacy file's own header comment records the fork and points at issue **#52228** for the consumer list and sunset plan.
  - **Same-basename trap:** slice *also owns* a file called `writer_unary_interleaved_start_id.cpp` in its own `device/kernels/dataflow/`, used by `SliceTileProgramFactory` (`tile.cpp:157`). The two are different files — slice's copy takes its DFB index from a **named** CTA (`get_named_compile_time_arg_val("dfb_id_out")`) and uses `dfb_out.get_entry_size()`; the eltwise copy takes a **positional** CTA 0 and uses `get_local_cb_interface(...)`. Don't cross the wires.
- **RTA varargs:** six kernels carry genuine variable-count blocks — detail in the brief.
- **Anything else the porter needs:**
  - **`ccl/mesh_partition` drives slice's factories directly.** `mesh_partition_program_factory.cpp:126-134` calls `SliceOp::validate_on_program_cache_miss` + `SliceOp::select_program_factory` + `Factory::create_descriptor(...)` under a `std::visit`, stores the chosen `SliceDeviceOperation::program_factory_t` in its own `shared_variables_t` (`mesh_partition_device_operation.hpp:47-50`), and on a cache hit calls `ttnn::prim::patch_slice_program_addresses` (`mesh_partition_program_factory.cpp:155`). Porting slice's factories changes both entry points that MeshPartition consumes.
  - **`patch_slice_program_addresses` is the real `override_runtime_arguments` body** for all five factories (`slice_program_factory_rm_sharded.cpp:354-413`) — a single `std::visit`-dispatched function with per-factory branches. The `CustomProgramSpecFactoryConcept` translation is one function, not five.
  - The RM reader **hardcodes** its DFB index in the kernel (`constexpr uint32_t dfb_id_in0 = 0;`, `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp:42`) rather than taking it from a CTA.

## Team-only

- **Out-of-directory coupling & donor shape.**

  **Op-level roll-up: ✓ clean.** One in-family donor header, both of whose consumed functions are ✓ shapes; one borrowed cross-family kernel file with a `_metal2` fork already in place. No ⚠, ✗ or ⭐ entries.

  **Summary table** — one row per (op kernel, donor file):

  | Op kernel | Donor file | Class | Status |
  |---|---|---|---|
  | `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp` | `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp` | 5 — in-family shared | ✓ |
  | `slice_writer_unary_stick_layout_interleaved_start_id.cpp` | same | 5 — in-family shared | ✓ |
  | `reader_multicore_slice_4d.cpp` | same | 5 — in-family shared | ✓ |
  | `writer_multicore_slice_4d.cpp` | same | 5 — in-family shared | ✓ |
  | `reader_multicore_slice_nd.cpp` | same | 5 — in-family shared | ✓ |
  | `writer_multicore_slice_nd.cpp` | same | 5 — in-family shared | ✓ |
  | all 11 kernels | `api/dataflow/{dataflow_api.h,noc.h,dataflow_buffer.h,endpoints.h}`, `api/tensor/noc_traits.h`, `api/core_local_mem.h` | 1 — `tt_metal/*` LLK/HAL | ✓ no concern |

  **Per-call detail** — omitted, all rolls ✓. For the record, the three functions actually called from `common.hpp` and their signature shapes:

  | Function | Signature shape | Status |
  |---|---|---|
  | `noc_async_read_sharded(Noc, uint32_t l1_addr, AddrGenType tensor, uint32_t src_id, uint32_t offset, uint32_t size)` (`common.hpp:375`) | `AddrGenType` instantiated as `TensorAccessor<DSpec>` — **Shape 1** | ✓ excellent |
  | `noc_async_write_sharded(...)` (`common.hpp:325`) | same — **Shape 1** | ✓ excellent |
  | `tt_memmove<bool,bool,bool,uint32_t>(Noc, uint32_t dst_l1_addr, uint32_t src_l1_addr, uint32_t bytes)` (`common.hpp:143`) | plain L1 addresses + `Noc` — no CB or semaphore handle | ✓ |

  No call site passes a `uint32_t sem_id`, a sem address, a `TensorAccessorArgs<N>`, a CTA-offset NTTP, a `uint32_t cb_id`, a `DataflowBuffer`, or a `CircularBuffer`. Every slice call site uses the **non-deprecated** `Noc`-leading overloads of the two sharded helpers (`common.hpp:362-372, 411-421` carry `[[deprecated]]` legacy overloads that slice does **not** use). The donor is itself already Device 2.0, so the Shape-4 donor-side gate does not arise.

  **Borrowed kernel files (file-path instantiation).** One:

  | File | Owning family | Also instantiated by | `_metal2` fork beside it? |
  |---|---|---|---|
  | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `eltwise/unary` (shared pool in practice) | broadly shared — 14 other consumers found: `experimental/transformer/nlp_concat_heads_boltz`, `experimental/transformer/nlp_concat_heads`, `experimental/matmul/attn_matmul`, `embedding` (fused), `examples/example` (multi-core + single-core), `data_movement/reshape_on_device`, `data_movement/tilize` (×4 factories), `data_movement/concat`, `eltwise/unary_backward/tanh_bw` | **Yes** — `writer_unary_interleaved_start_id_metal2.cpp`, same directory, non-quasar. Bind it; do not re-fork. |

  That consumer list is a **sunset list** (when the legacy copy can be retired), not a must-port-together bundle and not authorization to convert the file in place. Issue #52228 already tracks it.

  **Host-side cross-op coupling (not covered by the two escape types above).** `ttnn/cpp/ttnn/operations/ccl/mesh_partition/` reuses slice's *factories* directly rather than its kernels — see the Heads-ups bullet. This is a real coupling with a real break risk and has no bucket in the coupling taxonomy; recorded here and raised in *Recipe notes*.

- **Relaxation candidates** (noticed in the custom hash while auditing): **FALLIBLE — candidates to verify; default strict; the ops team owns the real analysis.** `compute_program_hash` (`slice_device_operation.cpp:348-432`) keys on `logical_shape`, `padded_shape`, `layout`, `dtype` and `memory_config` for the input, output spec, and each optional tensor — i.e. essentially the full `TensorSpec` minus the tile. Two observations:
  - The hash deliberately **omits `tile`**, and the op compensates with a hard `TT_FATAL` restricting tile layout to 32×32 (`slice_device_operation.cpp:198-203`), with an in-code comment explaining that a non-standard tile would both mis-size the program and alias onto a cached 32×32 one. This is a *narrowing* of the supported space, not a relaxation candidate.
  - `preallocated_output` is hashed as a full spec (`:419-429`) *and* validated against the computed spec (`:144-181`), including a `tensor_layout()` equality check whose comment explains the writer's `TensorAccessorArgs` bakes the destination's aligned page size in as a compile-time word. If anything, this suggests the op depends on **more** of the destination spec than a default hash would key on — the opposite of a relaxation opportunity.

  Net: no relaxation candidate surfaced. Consistent with the sheet's `none`.

- **TTNN factory analysis:** all sheet-derived facts with `file:line` evidence are in the *Gate detail* cross-check table above. Summary of the non-gating facts that inform the port's TTNN ProgramFactory wiring: **custom hash present** (`slice_device_operation.cpp:348`) — leave it exactly as it is; **`override_runtime_arguments` present on all five factories**, all delegating to one shared function — this selects `CustomProgramSpecFactoryConcept` and is the substantive TTNN-side translation work; **pybound `create_descriptor`** on `SliceTileProgramFactory` (`slice_nanobind.cpp:168-179`) — the port deletes it, a user-visible API change for the port report; **no op-owned tensors**; **no `get_dynamic_runtime_args`**; **not a MeshWorkload** (plain `descriptor` concept, no mesh-workload need of any kind).

## Misc anomalies  *(team-only, non-gating)*

1. **Dead compile-time arg in all four strided-slice kernels.** `constexpr uint32_t compile_time_element_size = get_compile_time_arg_val(1);` is declared and never referenced — every kernel takes `element_size` from an RTA instead. Sites: `reader_multicore_slice_4d.cpp:81`, `writer_multicore_slice_4d.cpp:65`, `reader_multicore_slice_nd.cpp:67`, `writer_multicore_slice_nd.cpp:66`. The host emits it at `slice_program_factory_rm_stride.cpp:79,82` (`{in_cb, element_size}`). Note for whoever cleans this up: the arg sits at index 1 with `TensorAccessorArgs<2>` immediately after, so removing it requires reindexing the accessor args — not a one-line delete. Not porter work.
2. **Dead runtime args in the 4D strided-slice kernels.** `reader_multicore_slice_4d.cpp:60-62` reads `output_h` / `output_d` / `output_n` and never uses them; `writer_multicore_slice_4d.cpp:56-58` does the same. The host emits them at `slice_program_factory_rm_stride.cpp:129` (reader: `output_shape[-2..-4]`) and `:138-141` (writer). Six dead RTA slots per core.
3. **Dead local in two readers.** `output_bytes_per_row` is computed and never used at `reader_multicore_slice_4d.cpp:86` and `reader_multicore_slice_nd.cpp:91`. (The *writers* of both ranks do use their equivalent.)
4. **Belt-and-braces address patching, worth a confirming look by the ops team.** For `SliceRmProgramFactory` and `SliceRmStrideProgramFactory`, the input/output bases are declared as `Buffer*` bindings — which the framework patches on cache hits, per the factories' own comments (`slice_program_factory_rm.cpp:401-402,409-410`; `slice_program_factory_tile.cpp:165-166`) — *and* `patch_slice_program_addresses` re-patches the same slot 0 by hand (`slice_program_factory_rm_sharded.cpp:372-389`). The most likely explanation is that MeshPartition builds `Program{descriptor}` directly (`mesh_partition_program_factory.cpp:131-132`) and may not receive the framework's binding injection, making the manual patch load-bearing on that path and redundant on slice's own. Flagging as an observation, not a defect — I could not confirm which path relies on which, and the values written are identical either way. It matters for the port because the `ProgramRunArgs` translation has to decide which of the two mechanisms survives.
5. **`#ifdef OUT_SHARDED` is unreachable for slice** in both `writer_unary_interleaved_start_id.cpp` copies — no slice factory sets kernel `defines`, so the `wait_front(num_pages)`-only branch is dead on every slice instantiation. Relevant only because a reader of the borrowed kernel might assume otherwise.

## Per-DeviceOperation attribution

Single DeviceOperation (`SliceDeviceOperation`); no bundling. All per-factory variation is recorded in-line in the tables above — the material split is `SliceRmShardedProgramFactory`, which alone uses borrowed-memory DFBs, has no writer kernel, has no `TensorAccessor`, and takes both self-loop dispositions.

## Questions for the user

1. **`Provisional relaxation finding (Edwin)` says `needs fix, then none`.** On the `SliceRmProgramFactory` and `SliceTileProgramFactory` rows, that column reads `needs fix, then none`, while the gating `TensorParameter relaxation` column reads `none` on all five rows. The audit gates on the latter, so this does not block — but "needs fix" suggests an ops-team item that may want to land before or alongside the port. The column is not documented in `ttnn_op_porting_readiness.md` and I could not tell what fix is meant. Worth a check with Edwin / the sheet owner.
2. **Both dated triage docs need a slice update.** `2026-07-19_offset_base_pointers.md:63` (Type 2, "the canonical case") and `2026-07-06_tensor_accessor_3rd_arg_triage.md:75,139` (Class 1 + Special) both describe code that no longer exists. I trusted my read per the recipe's contract for these docs, but the entries will mislead the next reader — and the audit recipe's own *Offset base pointers → Code-path scope* paragraph is written on top of the stale entry (see *Recipe notes* 1). Who owns updating them?

## Recipe notes

1. **The recipe pre-commits to a slice RED, and the code says otherwise.** `metal2_audit.md` → *Offset base pointers* → **Code-path scope** reads: *"The wall is a **row-major-layout** phenomenon… So a slice-family RED applies Code-path scope — RED the RM factory and name the tiled factories as a clean subset (`RED at op level; subset <tiled factories> is clear`)."* Read cold, that paragraph is a verdict, not a conditional: it names this op's factories and pre-writes the Result string. It is also **wrong for current `main`** — the fold has been split out and the RM factory is clean. The recipe's own reconciliation table three paragraphs earlier gets this right (*"No fold, op in the tables → the doc is stale → GREEN"*), so the two passages pull in opposite directions on the same op, and the more specific-looking one is the stale one. An auditor who read the Code-path-scope paragraph first and pattern-matched would RED a GREEN op. **Suggested fix:** make it conditional and de-name the verdict — e.g. *"If a slice-family fold is present, it is factory-scoped: RED the RM factory and name the tiled factories as a clean subset. Confirm against current code first; the catalogued slice fold may already have been split out."*
2. **Both dated priors are stale for this op, in the same direction.** Independently of (1): `slice` is stale in *both* triage docs, and in both cases because the ops team fixed it after the analysis date. That is the recipe's anticipated drift direction, and the *"trust your scan / trust your read"* contracts handled it cleanly — the guidance worked. Flagging only because two-for-two on one op suggests a coordinated slice cleanup landed after 2026-07-19, and the docs may be stale for the sibling ops the same commit touched (`padded_slice`, `slice_write`) — worth a sweep rather than op-by-op discovery.
3. **The `Buffer*`-binding form has no roll-up vocabulary, and the op-level verdict overstates it.** *TensorParameter analysis* is written around `->address()`-in-an-RTA as "the legacy hazard", with the `Buffer*` form as one bullet in the detection list that explicitly says it is *"correct-on-cache-hit today… not the silent-wrong hazard"* and *"don't over-state the urgency of this one."* In slice, **every** address is a `Buffer*` binding and there is not one `->address()` RTA in any `create_descriptor`. The prescribed op-level roll-up is then `⚠ port work` — the same label an op with eight genuinely-stale RTA addresses would get. The subject's own guidance says not to overstate it, but the roll-up vocabulary offers no way to comply. **Suggested fix:** add a roll-up value distinguishing the two, e.g. `⚠ port work (Buffer*-binding delivery — routine, no correctness hazard)`.
4. **Out-of-directory coupling has no bucket for host-side factory reuse.** The subject covers exactly two escape types, both kernel-side: function-call escape (`#include` + call) and file-path kernel instantiation. `ccl/mesh_partition` does neither — it calls slice's `select_program_factory`, every factory's `create_descriptor`, and the shared `patch_slice_program_addresses`, and stores `SliceDeviceOperation::program_factory_t` in its own shared variables. This was the single highest-risk item I found for the porter (both consumed entry points change under the port), and the recipe gave me nowhere to put it except the open *"Anything else the porter needs"* bullet. **Suggested fix:** add a third escape type — *host-side factory reuse by another op* — with a recognition cue (grep the op's factory type names and any exported helpers outside its own directory; `ttnn::prim::` free functions declared in the device-op header are the tell). It is cheap to scan and, unlike the kernel-side escapes, it breaks at compile time in someone else's op.
5. **The readiness sheet has columns the reading guide doesn't document, including one that duplicates a derivation the recipe asks the auditor to perform.** The header row now carries `Porting Target` (value here: `CustomProgramSpecFactoryConcept`) and `Execution Model` (`SPMD`), neither listed in `ttnn_op_porting_readiness.md`'s column set. `Porting Target` is exactly what *TTNN porting shape* asks the auditor to derive from `Concept` + `Op-owned tensors?` + `Override runtime args method?`. Mine agreed with the sheet, but the recipe doesn't say which wins on a disagreement, or whether the column should be cross-checked at all. Also undocumented: `Provisional relaxation finding (Edwin)` (see *Questions* 1), `Diego validation`, `Pointer patching perf issue?`, `Formerly custom hashed?`. **Suggested fix:** add them to the reading guide, and say whether `Porting Target` is a cross-check target for the derived shape or another read-don't-vet derived cell.
6. **Device 2.0 sanctioned-list guidance worked exactly as written — no change needed.** Recording the positive: the borrowed eltwise writer has `get_local_cb_interface(cb_id_out)` three lines from a `DataflowBuffer dfb(cb_id_out)`, and slice's own near-identical copy of the same kernel already uses `dfb_out.get_entry_size()` — so a replacement demonstrably exists and the "wrapper in scope + replacement exists" holdover cue fires hard. The Green bullet's explicit override (*"the list is the whole test… Kernels already on `DataflowBuffer` are where that cue misfires hardest"*) resolved it without ambiguity, and the breadcrumb about whitelist rule 7 told me where it does belong. Worth keeping verbatim.
7. **Minor — "unreferenced kernel files" guidance is clear but the disambiguation case is worth naming.** The recipe says to mention unreferenced files *"if their presence could confuse a reader."* Slice has a sharper version of that: two files with the **same basename** in different directories, one owned and one borrowed, both referenced, and behaving differently (`writer_unary_interleaved_start_id.cpp`). Nothing in the recipe prompts a check for basename collisions across the referenced set, and it is a plausible porter footgun. Might be worth a line in the borrowed-kernel-files bullet.
