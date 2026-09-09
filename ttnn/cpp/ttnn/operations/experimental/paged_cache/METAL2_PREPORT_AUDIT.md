# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/paged_cache`

The directory holds **three** DeviceOperations. They are audited together and reported as one unit
because they share the `device/kernels/` tree, the same kernel idioms, and the same host-side
descriptor/override structure. Per-DeviceOperation attribution is given wherever a finding differs.

- **`PagedFillCacheDeviceOperation`** (declared in [`device/fill_cache/paged_fill_cache_device_operation.hpp:21-38`](device/fill_cache/paged_fill_cache_device_operation.hpp#L21-L38))
  - `PagedFillCacheProgramFactory` ([`device/fill_cache/paged_fill_cache_program_factory.cpp:340-346`](device/fill_cache/paged_fill_cache_program_factory.cpp#L340-L346))
  - `PagedFillCacheMeshWorkloadFactory` ([`device/fill_cache/paged_fill_cache_program_factory.cpp:348-359`](device/fill_cache/paged_fill_cache_program_factory.cpp#L348-L359))
  - Kernels: `reader_fill_cache_interleaved.cpp`, `writer_fill_cache_interleaved.cpp` (no compute kernel)
- **`PagedUpdateCacheDeviceOperation`** (declared in [`device/update_cache/paged_update_cache_device_operation.hpp:20-41`](device/update_cache/paged_update_cache_device_operation.hpp#L20-L41))
  - `PagedUpdateCacheProgramFactory` ([`device/update_cache/paged_update_cache_program_factory.cpp:89-441`](device/update_cache/paged_update_cache_program_factory.cpp#L89-L441))
  - `PagedUpdateCacheMeshWorkloadFactory` ([`device/update_cache/paged_update_cache_program_factory.cpp:443-455`](device/update_cache/paged_update_cache_program_factory.cpp#L443-L455))
  - Kernels: `reader_update_cache_interleaved_start_id.cpp`, `writer_update_cache_interleaved_start_id.cpp`, `compute/update_cache.cpp`
- **`PagedFusedUpdateCacheDeviceOperation`** (declared in [`device/fused_update_cache/paged_fused_update_cache_device_operation.hpp:21-46`](device/fused_update_cache/paged_fused_update_cache_device_operation.hpp#L21-L46))
  - `PagedTiledFusedUpdateCacheProgramFactory` ([`device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp:79-537`](device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp#L79-L537))
  - `PagedTiledFusedUpdateCacheMeshWorkloadFactory` ([`device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp:539-552`](device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp#L539-L552))
  - `PagedRowMajorFusedUpdateCacheProgramFactory` ([`device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp:79-534`](device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp#L79-L534))
  - `PagedRowMajorFusedUpdateCacheMeshWorkloadFactory` ([`device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp:536-554`](device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp#L536-L554))
  - Kernels: `reader_paged_fused_update_cache_interleaved_start_id.cpp`, `writer_paged_fused_update_cache_interleaved_start_id.cpp`, `compute/paged_fused_update_cache.cpp` (tiled) and `reader_paged_row_major_fused_update_cache_interleaved_start_id.cpp`, `writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp`, `compute/paged_row_major_fused_update_cache.cpp` (row-major)

**All 11 kernel files under `device/kernels/` are referenced by a factory.** There are no unreferenced
kernel files in the directory, and no other op in the tree instantiates any of them.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `f9451e2a21d 2026-09-09 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/paged_cache` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `PagedFillCacheDeviceOperation` → `PagedFillCacheProgramFactory`, `PagedFillCacheMeshWorkloadFactory` · `PagedUpdateCacheDeviceOperation` → `PagedUpdateCacheProgramFactory`, `PagedUpdateCacheMeshWorkloadFactory` · `PagedFusedUpdateCacheDeviceOperation` → `PagedTiledFusedUpdateCacheProgramFactory`, `PagedTiledFusedUpdateCacheMeshWorkloadFactory`, `PagedRowMajorFusedUpdateCacheProgramFactory`, `PagedRowMajorFusedUpdateCacheMeshWorkloadFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 11 op kernels are structurally Device 2.0 (`Noc`, `CircularBuffer`, `Semaphore<>`, `CoreLocalMem`, `UnicastEndpoint`); no CB-index free-function holdovers; both `kernel_lib` donors are already on `DataflowBuffer` |
| *Prereqs* — Cross-op escapes | **Ok** — only `tt_metal/*` LLK headers and two `ttnn/cpp/ttnn/kernel_lib/` helpers; no cross-family donor, no borrowed kernel file |
| *Feature Support* — overall | **GREEN** — every Appendix A entry is `N/A` |
| *Feature Support* — Variadic-CTA | Ok — no compile-time arg is read at a varying index |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes**, on all 8 factory rows |
| *TTNN Readiness* — Concept (current) | `descriptor` (all 8 rows; confirmed in code — every factory defines `create_descriptor` returning `tt::tt_metal::ProgramDescriptor`) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A — `Concept` is `descriptor`, not `WorkloadDescriptor`, so the column does not apply (the sheet leaves it blank) |
| *TTNN Readiness* — Custom hash | **Yes** (not a gate; port leaves it intact): [`device/fill_cache/paged_fill_cache_device_operation.cpp:216-225`](device/fill_cache/paged_fill_cache_device_operation.cpp#L216-L225) · [`device/update_cache/paged_update_cache_device_operation.cpp:314-338`](device/update_cache/paged_update_cache_device_operation.cpp#L314-L338) · [`device/fused_update_cache/paged_fused_update_cache_device_operation.cpp:372-390`](device/fused_update_cache/paged_fused_update_cache_device_operation.cpp#L372-L390). No backdoor `attribute_values` / `to_hash`. |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** — grep over the whole op directory returns zero hits |
| *TTNN Readiness* — `override_runtime_arguments` | **Yes** (not a gate; selects `CustomProgramSpecFactoryConcept`): [`device/fill_cache/paged_fill_cache_program_factory.cpp:361-428`](device/fill_cache/paged_fill_cache_program_factory.cpp#L361-L428) · [`device/update_cache/paged_update_cache_program_factory.cpp:457-530`](device/update_cache/paged_update_cache_program_factory.cpp#L457-L530) · [`device/fused_update_cache/paged_fused_update_cache_device_operation.cpp:396-447`](device/fused_update_cache/paged_fused_update_cache_device_operation.cpp#L396-L447) |
| *TTNN Readiness* — Pybind `create_descriptor` | **No** — [`paged_cache_nanobind.cpp:23-165`](paged_cache_nanobind.cpp#L23-L165) binds only the three user-facing op functions; no `nb::class_` of a device op, no `create_descriptor` binding |
| *TTNN Readiness* — Op-owned tensors | **No** — no factory returns a `WorkloadDescriptor`, so there is no `buffers` vector |
| *TTNN Readiness* — Target concept | **`CustomProgramSpecFactoryConcept`** (all 8 rows; the sheet's own `Porting Target` column agrees) |
| *Port work* — Offset base pointer | **none** — every `->address()` in the op is a bare base; no host-side offset fold anywhere |
| *Port work* — Tensor bindings (per binding) | 15 distinct bindings: 10 Case 1, 3 clean (borrowed-memory DFB), 2 that flip between Case 1 and clean with the DRAM-vs-L1-sharded config. Inventory below. |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears — the port applies no relaxation) |
| *Port work* — TensorAccessor 3rd arg | **none — no accessor in the op passes a 3rd argument.** All 17 construction sites are the 2-argument `TensorAccessor(args, addr)` form. |
| *Port work* — CB endpoints | legal (1P+1C) for 24 of 27 `(CB, config)` instances · **self-loop** for 3 in `fill_cache` · **conditional DFB** for 8 (present only under a config) · no dead CB · no multi-binding flag needed |
| *Port work* — fused kernel placement | narrow each `WorkUnitSpec` to one input shard grid (forced by derived DFB placement); the `unused_cores` instances drop out — 20 cores in llama3-70b galaxy decode, each freeing ~59–112 KB |

**CB endpoints** are dispositions, not gates: every out-of-window CB has a port-time resolution — a
**self-loop** (one toucher: single-ended / sync-free), a **1P+1C assignment** (two touchers), the
**multi-binding advanced-option flag** (genuine ≥2 of a role on a node the census can't relabel), or a
**dead-CB drop** (zero endpoints). The disposition is recorded per `(CB, config)` below, classified per
instantiation.

## Result

**GREEN → brief issued.** All five gate-bearing subjects clear:

- Device 2.0 ✓ — the op's own kernels and both shared-library donors are Device 2.0 compliant.
- Feature compatibility ✓ — no `GlobalCircularBuffer`, no non-zero `CBDescriptor::address_offset`, no `GlobalSemaphore`.
- TTNN factory concept ✓ — `Is able to port? = yes` on all 8 factory rows; the cheaply-checkable columns match the code; the sheet's 8 rows map one-to-one onto the 8 factories in the code.
- Offset base pointers ✓ — no address RTA folds a host-side offset into its base.
- TensorAccessor 3rd argument ✓ — the subject never fires; no accessor passes a 3rd argument.

The port is substantial but structural rather than blocked. Two shapes account for most of the work and
are called out prominently in the brief: **conditionally-bound resources** (eight optional CBs and four
optional tensors that the kernels reference unconditionally at C++ scope while gating *use* behind
`if constexpr`), and the fused ops' **runtime-selected input CB index** on a kernel placed over a core
range wider than the CB it binds. Both have documented Metal 2.0 patterns; neither is a gate.

**One question goes to the readiness-sheet owner**, and it is narrow: on the three mesh variants that
return an empty `ProgramDescriptor` for an excluded coordinate, is it acceptable for an excluded chip
to receive a program whose kernels launch and immediately return? Answering yes ports all eight
factories; answering no leaves those three on the legacy descriptor path, which is supported but costs
code duplication. Recommended default and full reasoning in *Question for the user*.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** All 8 rows read `yes`. Cross-check against
  the code came back clean on every cheaply-checkable column:

  | Column | Sheet value | Code evidence | Match |
  |---|---|---|---|
  | `Concept` | `descriptor` (×8) | every factory defines `create_descriptor` returning `ProgramDescriptor`; no `create_workload_descriptor`, no `create()` + legacy `override_runtime_arguments` pair, no `MetalV2` factory | ✓ |
  | `Custom hash (compute_program_hash)` | `yes` (×8) | a `compute_program_hash` override on each of the three device-ops (sites in the status summary) | ✓ |
  | `Backdoor custom hash (attribute_values / to_hash)` | `no` (×8) | no `attribute_values` or `to_hash` anywhere in the op | ✓ |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` (×8) | zero hits for `get_dynamic_runtime_args` in the op directory | ✓ |
  | `Override runtime args method? (PD only)` | `yes` (×8) | all 8 factories declare `override_runtime_arguments` (sites in the status summary) | ✓ |
  | `Pybind descriptor (nb::class_ of device op)` | `no` (×8) | [`paged_cache_nanobind.cpp:23`](paged_cache_nanobind.cpp#L23) binds three free functions only | ✓ |
  | `Smuggled pointer (raw buffer addr in RTA/CRTA)` | `no` (×8) | no `->address()` reaches a runtime-arg list at descriptor-build time — see the tensor-binding inventory below | ✓ |
  | `Op-owned tensors?` | blank (×8) | no `WorkloadDescriptor`, no `buffers` vector | ✓ |
  | `Secretly SPMD Workload?` | blank (×8) | not applicable — the column is only read when `Concept == WorkloadDescriptor` | ✓ (N/A) |
  | `TensorParameter relaxation` | `none` (×8) | clears — no relaxation applied by the port | ✓ |
  | `Known op issues` | blank (×8) | — | ✓ |
  | Factory-set match | 8 rows | 8 factories in the code, one-to-one, no phantom row and no missing row | ✓ |

  Cross-column invariants hold: `get_dynamic_runtime_args == no` is consistent with the `descriptor`
  concept, and `Op-owned tensors?` is not `yes` on a `descriptor` row.

  *Recorded, not gating:* the sheet's `Diego validation` column reads `no` on all 8 rows. That column is
  not one the audit reads or gates on; it is noted here only so a reader of the row does not mistake it
  for the verdict cell. The gate cell is `Is able to port?`, which reads `yes`.

- **Device 2.0 (every kernel used):** **GREEN.** Every kernel the op instantiates is structurally
  Device 2.0, and so is every donor it calls into.

  *The op's own 11 kernels.* Each dataflow kernel opens with `Noc noc;` and does all memory movement
  through `noc.async_read` / `noc.async_write` / `noc.async_write_zeros`-family methods with
  `CoreLocalMem<uint32_t>` and `UnicastEndpoint` destinations, all CB access through `CircularBuffer`
  *objects* (`cb.reserve_back`, `cb.push_back`, `cb.wait_front`, `cb.pop_front`, `cb.get_write_ptr()`,
  `cb.get_read_ptr()`, `cb.get_tile_size()`), and all semaphore access through `Semaphore<>` objects
  (`.wait`, `.set`, `.up`). A scan for the Device 1.0 idiom set — `noc_async_read` / `noc_async_write`,
  `InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedAddrGenFast`, `InterleavedPow2AddrGen*`,
  `get_noc_addr_from_bank_id`, raw `noc_semaphore_*`, `get_semaphore`, and the free-function
  `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front` — returns **zero** code hits
  across the op (the only two textual matches are the words `noc_semaphore_inc` inside explanatory
  comments at [`device/kernels/dataflow/writer_paged_fused_update_cache_interleaved_start_id.cpp:178`](device/kernels/dataflow/writer_paged_fused_update_cache_interleaved_start_id.cpp#L178)
  and [`device/kernels/dataflow/writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp:180`](device/kernels/dataflow/writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp#L180)).

  **No CB-index-keyed free-function holdovers.** All 37 `get_write_ptr` / `get_read_ptr` /
  `get_tile_size` call sites in the op are *methods* on a `CircularBuffer` object; none is the
  free-function form taking a `uint32_t` CB index. No `get_local_cb_interface`, no
  `evil_set_write_ptr` / `evil_set_read_ptr`.

  *Donors.* The two compute kernel-library helpers the compute kernels call —
  `compute_kernel_lib::untilize` ([`../../../../../../ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp:145-154`](../../../../../../ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp#L145-L154))
  and `compute_kernel_lib::tilize` ([`../../../../../../ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:153-163`](../../../../../../ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp#L153-L163))
  — are **already on `DataflowBuffer`** internally
  ([`../../../../../../ttnn/cpp/ttnn/kernel_lib/untilize_helpers.inl:199-200`](../../../../../../ttnn/cpp/ttnn/kernel_lib/untilize_helpers.inl#L199-L200),
  [`../../../../../../ttnn/cpp/ttnn/kernel_lib/tilize_helpers.inl:149-150`](../../../../../../ttnn/cpp/ttnn/kernel_lib/tilize_helpers.inl#L149-L150)).
  Their one free-function CB-index use, `get_local_cb_interface`
  ([`../../../../../../ttnn/cpp/ttnn/kernel_lib/untilize_helpers.inl:207`](../../../../../../ttnn/cpp/ttnn/kernel_lib/untilize_helpers.inl#L207)),
  is on the sanctioned list and is not a holdover. Everything else the kernels include is `tt_metal/*`
  LLK / HAL (`api/dataflow/*`, `api/compute/*`, `api/core_local_mem.h`, `api/tensor/noc_traits.h`) — no
  concern.

  No violations table is given because there are no violations.

  *One idiom recorded so the porter does not mistake it for a holdover:* the three writer kernels form a
  local NoC address for an intra-core L1→L1 copy out of `my_x[noc_id]` / `my_y[noc_id]` with
  `noc_id = noc.get_noc_id()`, fed to a `UnicastEndpoint`
  ([`device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp:114-131`](device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp#L114-L131)).
  `my_x` / `my_y` are HAL globals for the core's own coordinates, not CB-index free functions, and the
  surrounding call is the Device 2.0 `UnicastEndpoint` path. Not a Device 2.0 violation, and not
  something the Metal 2.0 port touches.

- **Feature compatibility:** every Appendix A entry, in order. Every entry is UNSUPPORTED, so a per-row
  status of `N/A` means the feature is absent from the op.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | **N/A** | Zero hits for the type `GlobalCircularBuffer` (either include spelling), `experimental::CreateGlobalCircularBuffer`, the `CBDescriptor::global_circular_buffer` field, the 4-argument `experimental::CreateCircularBuffer(..., global_cb)` form, `CircularBufferConfig::remote_index`, any `remote_cb_*` identifier, or the 3-argument `UpdateDynamicCircularBufferAddress(program, handle, const GlobalCircularBuffer&)` overload. The op's two `UpdateDynamicCircularBufferAddress` call sites are both the two-argument `Buffer&` form, which is unrelated: [`device/fused_update_cache/paged_fused_update_cache_device_operation.cpp:70`](device/fused_update_cache/paged_fused_update_cache_device_operation.cpp#L70) and [`device/update_cache/paged_update_cache_program_factory.cpp:518-519`](device/update_cache/paged_update_cache_program_factory.cpp#L518-L519). |
  | CBDescriptor `address_offset` (non-zero) | **N/A** | Zero hits for `address_offset`, `set_address_offset`, `cb_descriptor_from_sharded_tensor`, or the 4-argument `UpdateDynamicCircularBufferAddress(program, handle, buffer, offset)` form anywhere in the op. No `CBDescriptor` literal sets the field, so it defaults to zero on all 26 CB descriptors. No runtime-team consultation is needed. |
  | GlobalSemaphore | **N/A** | Zero hits for the type `GlobalSemaphore`, `experimental::CreateGlobalSemaphore`, or `#include <tt-metalium/global_semaphore.hpp>`. The op's three semaphores are plain `SemaphoreDescriptor` literals ([`device/update_cache/paged_update_cache_program_factory.cpp:246-252`](device/update_cache/paged_update_cache_program_factory.cpp#L246-L252), [`device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp:259-265`](device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp#L259-L265), [`device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp:255-261`](device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp#L255-L261)), read kernel-side as `Semaphore<>` — the supported `SemaphoreSpec` path. `fill_cache` declares no semaphore at all. |

- **CB endpoints (GATE-free):** every CB is either a plain 1P+1C FIFO or carries a port-time
  disposition. **No CB needs the multi-binding advanced option, and no CB is dead.** The full
  per-`(CB, config)` census is in *Port-work summary* below. Highlights:

  - **24 of 27 `(CB, config)` instances are plain 1:1** — one locked FIFO producer plus one locked FIFO
    consumer on every node. Nothing to do.
  - **3 self-loops, all in `fill_cache`** — `c_1` (page table), `c_2` (batch_idx) and `c_3`
    (valid_seq_len) are touched by the **writer only**: it `reserve_back(1)`s, takes
    `get_write_ptr()`, NoC-reads the metadata into that address and then reads it back through an L1
    pointer, never `push_back`ing and with no other kernel involved
    ([`device/kernels/dataflow/writer_fill_cache_interleaved.cpp:148-150`](device/kernels/dataflow/writer_fill_cache_interleaved.cpp#L148-L150)).
    One toucher → bind the writer both PRODUCER and CONSUMER.
  - **8 conditional DFBs** — the index CB and the page-table CB in `update_cache` and in each of the two fused
    factories, plus the batch_idx and valid_seq_len CBs in `fill_cache`, are pushed into `desc.cbs`
    only under their config flag. The legacy factories already gate the *allocation*, so no CB is dead
    in a config where it is allocated. The port keeps each `DataflowBufferSpec` conditional on the same
    flag — and must gate the kernel-side references at the preprocessor level, because the kernels
    declare the `CircularBuffer` objects unconditionally at function scope (see *Heads-ups*).
  - **The `(is_paged_cache && !use_index_tensor)` config, which would have left the page-table CB with
    zero touchers, is unreachable.** All three device-ops assert that paged mode requires an index
    tensor: [`device/update_cache/paged_update_cache_device_operation.cpp:149`](device/update_cache/paged_update_cache_device_operation.cpp#L149)
    and [`device/fused_update_cache/paged_fused_update_cache_device_operation.cpp:213`](device/fused_update_cache/paged_fused_update_cache_device_operation.cpp#L213).
    (`fill_cache` takes the page table as a required, non-optional tensor, so the question does not
    arise there.) This is the one place a `(0, 0)` census looked possible, and validation rules it out.
  - **No hidden second writer.** Every CB access in every kernel was enumerated, FIFO and raw-pointer
    alike. The only raw writes into a CB another kernel also touches are the ones the FIFO producer
    makes into its *own* buffer before `push_back` — one toucher, not two. Nothing writes through
    `get_write_ptr()` into a CB it does not FIFO-produce, and no semaphore in the op coordinates a CB
    co-fill (all three semaphores serialize the *share_cache* cache-read ordering between cores, which
    is not a CB endpoint).
  - **The `c_1` / `c_2` input CBs of the fused factories are not co-resident.** The device-op asserts
    the two input shard grids do not overlap
    ([`device/fused_update_cache/paged_fused_update_cache_device_operation.cpp:352`](device/fused_update_cache/paged_fused_update_cache_device_operation.cpp#L352)),
    so on any given node exactly one of the two exists. That is why the runtime CB select (below) is a
    binding-shape problem and not an endpoint-count problem.

- **Offset base pointers:** **GREEN.** All 12 `->address()` expressions in the op are bare bases with
  no host arithmetic folded in — see the list below. They all live in the cache-hit
  `override_runtime_arguments` patch; at descriptor-build time the factories push the `Buffer*` itself
  and never an address expression. The tile-index offsets the kernels need (`cache_start_id`,
  `start_tile_id`, `start_row_num`, `tile_update_offset_B`) travel as **separate scalar runtime args**
  and are added on-device, which is exactly the already-split shape the subject calls GREEN.

  | Site | Expression | Fold? |
  |---|---|---|
  | [`device/update_cache/paged_update_cache_program_factory.cpp:488`](device/update_cache/paged_update_cache_program_factory.cpp#L488) | `cache_tensor.buffer()->address()` | no |
  | [`device/update_cache/paged_update_cache_program_factory.cpp:490-493`](device/update_cache/paged_update_cache_program_factory.cpp#L490-L493) | `update_idxs_tensor…->address()`, `page_table…->address()` | no |
  | [`device/fill_cache/paged_fill_cache_program_factory.cpp:384-396`](device/fill_cache/paged_fill_cache_program_factory.cpp#L384-L396) | `input_tensor`, `cache_tensor`, `page_table`, `valid_seq_len_tensor_opt`, `batch_idx_tensor_opt` bases | no |
  | [`device/fused_update_cache/paged_fused_update_cache_device_operation.cpp:83-88`](device/fused_update_cache/paged_fused_update_cache_device_operation.cpp#L83-L88) | `cache_tensor1`, `cache_tensor2`, `update_idxs_tensor`, `page_table` bases | no |

  The op appears in **neither** table of the dated offset-base-pointer triage analysis
  (`analyses/2026-07-19_offset_base_pointers.md`), which is consistent with the scan: *no fold, op not
  in the tables* → clean, and every address RTA is handed to TensorParameter analysis below. No
  `ttnn::narrow` and no `MeshBuffer::create(…, parent_base + offset)` interior-base view anywhere in the
  op, so Type 4 does not arise either.

- **TensorAccessor 3rd argument:** **N/A — no accessor in the op passes a 3rd argument, so the subject
  never fires.** All 17 `TensorAccessor(...)` construction sites across the 8 dataflow kernels use the
  2-argument `TensorAccessor(args, base_addr)` form; none supplies an explicit page size. This is not
  "every site classified Class 2" — there are no sites to classify. The op appears in no row of the
  dated 3rd-argument triage analysis (`analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md`), which
  agrees with the scan.

## Port-work summary  *(mirrors the brief)*

### Tensor bindings (per binding)

Fifteen distinct tensor bindings across the three device-ops (the table below pairs the fused op's `1`/`2` tensors on one row where their treatment is identical). Every legacy address delivery is the
`Buffer*`-binding form — the factories push a `Buffer*` into `KernelDescriptor::RTArgList` /
`emplace_runtime_args` and the framework auto-registers a `BufferBinding` — so none of them is the
silent-wrong stale-address hazard; each is routine port work, classified by what the kernel does with
the `uint32_t` base it receives. The op's own `override_runtime_arguments` additionally re-applies every
address slot on each cache hit
([`device/fill_cache/paged_fill_cache_program_factory.cpp:405-416`](device/fill_cache/paged_fill_cache_program_factory.cpp#L405-L416),
[`device/update_cache/paged_update_cache_program_factory.cpp:500-516`](device/update_cache/paged_update_cache_program_factory.cpp#L500-L516),
[`device/fused_update_cache/paged_fused_update_cache_device_operation.cpp:100-124`](device/fused_update_cache/paged_fused_update_cache_device_operation.cpp#L100-L124)).
No address RTA is left unpatched.

| DeviceOperation | Binding | Class | Evidence |
|---|---|---|---|
| `PagedFillCache` | `input_tensor` | **Case 1** | `Buffer*` at [`paged_fill_cache_program_factory.cpp:302`](device/fill_cache/paged_fill_cache_program_factory.cpp#L302) → `TensorAccessor(src_args, src_addr)` at [`reader_fill_cache_interleaved.cpp:29`](device/kernels/dataflow/reader_fill_cache_interleaved.cpp#L29) |
| `PagedFillCache` | `cache_tensor` (in-place output) | **Case 1** | `Buffer*` at [`paged_fill_cache_program_factory.cpp:311`](device/fill_cache/paged_fill_cache_program_factory.cpp#L311) → `TensorAccessor(s0_args, dst_addr)` at [`writer_fill_cache_interleaved.cpp:145`](device/kernels/dataflow/writer_fill_cache_interleaved.cpp#L145) |
| `PagedFillCache` | `page_table` | **Case 1** | `Buffer*` at [`paged_fill_cache_program_factory.cpp:312`](device/fill_cache/paged_fill_cache_program_factory.cpp#L312) → `TensorAccessor(page_table_args, page_table_addr)` at [`writer_fill_cache_interleaved.cpp:146`](device/kernels/dataflow/writer_fill_cache_interleaved.cpp#L146) |
| `PagedFillCache` | `batch_idx_tensor_opt` *(optional)* | **Case 1** | `Buffer*` at [`paged_fill_cache_program_factory.cpp:316`](device/fill_cache/paged_fill_cache_program_factory.cpp#L316) → `TensorAccessor(batch_idx_tensor_args, batch_arg)` at [`writer_fill_cache_interleaved.cpp:101`](device/kernels/dataflow/writer_fill_cache_interleaved.cpp#L101) |
| `PagedFillCache` | `valid_seq_len_tensor_opt` *(optional)* | **Case 1** | `Buffer*` at [`paged_fill_cache_program_factory.cpp:324`](device/fill_cache/paged_fill_cache_program_factory.cpp#L324) → `TensorAccessor(valid_seq_len_tensor_args, valid_seq_len_addr)` at [`writer_fill_cache_interleaved.cpp:122`](device/kernels/dataflow/writer_fill_cache_interleaved.cpp#L122) |
| `PagedUpdateCache` | `cache_tensor` (in-place output) | **Case 1** | `Buffer*` at [`paged_update_cache_program_factory.cpp:406`](device/update_cache/paged_update_cache_program_factory.cpp#L406) and `:426` → `TensorAccessor(s0_args, cache_addr)` at [`reader_update_cache_interleaved_start_id.cpp:69`](device/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp#L69) and [`writer_update_cache_interleaved_start_id.cpp:55`](device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp#L55) |
| `PagedUpdateCache` | `update_idxs_tensor` *(optional)* | **Case 1** | `Buffer*` at [`paged_update_cache_program_factory.cpp:409`](device/update_cache/paged_update_cache_program_factory.cpp#L409) → `TensorAccessor(index_tensor_args, index_tensor_addr)` at [`reader_update_cache_interleaved_start_id.cpp:74`](device/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp#L74). No config split: this CB is never buffer-backed here. |
| `PagedUpdateCache` | `page_table` *(optional)* | **Case 1** | `Buffer*` at [`paged_update_cache_program_factory.cpp:415`](device/update_cache/paged_update_cache_program_factory.cpp#L415) → `TensorAccessor(page_table_args, page_table_tensor_addr)` at [`reader_update_cache_interleaved_start_id.cpp:95`](device/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp#L95) |
| `PagedUpdateCache` | `input_tensor` | **clean** | Borrowed-memory CB: `.buffer = in1_buffer` on the `c_1` descriptor at [`paged_update_cache_program_factory.cpp:200-209`](device/update_cache/paged_update_cache_program_factory.cpp#L200-L209). The reader `reserve_back`/`push_back`s without writing and the compute kernel untilizes straight out of it — the borrowed DFB *is* the tensor access. Port via `DataflowBufferSpec::borrowed_from`. |
| `PagedFusedUpdateCache` | `cache_tensor1`, `cache_tensor2` (in-place outputs) | **Case 1** | `Buffer*` at [`paged_tiled_fused_update_cache_program_factory.cpp:438`](device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp#L438) / `:483` (and `:457`/`:503` for the writer) → `TensorAccessor(s0_args, cache_addr)` at [`reader_paged_fused_update_cache_interleaved_start_id.cpp:82`](device/kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp#L82) and [`writer_paged_fused_update_cache_interleaved_start_id.cpp:60`](device/kernels/dataflow/writer_paged_fused_update_cache_interleaved_start_id.cpp#L60); same shape in the row-major pair. Both cache tensors reach the same `TensorAccessorArgs` slot, one per core group. |
| `PagedFusedUpdateCache` | `input_tensor1`, `input_tensor2` | **clean** | Borrowed-memory CBs `c_1` / `c_2`: `.buffer = in1_buffer` / `in2_buffer` at [`paged_tiled_fused_update_cache_program_factory.cpp:203-222`](device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp#L203-L222) and [`paged_row_major_fused_update_cache_program_factory.cpp:208-227`](device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp#L208-L227). Port via `borrowed_from`. |
| `PagedFusedUpdateCache` | `update_idxs_tensor` *(optional)* | **Case 1** *(DRAM)* / **clean** *(L1-sharded)* | Per-config split. When the tensor is L1-sharded the factory sets `.buffer = index_buffer_ptr` on the `c_3` CB ([`paged_tiled_fused_update_cache_program_factory.cpp:267-279`](device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp#L267-L279)) and the reader reads it straight out of the borrowed CB without a NoC read ([`reader_paged_fused_update_cache_interleaved_start_id.cpp:90-96`](device/kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp#L90-L96)) → clean. When it is DRAM the pointer is null, the CB is plain, and the reader NoC-reads through `TensorAccessor(index_tensor_args, index_tensor_addr)` → Case 1. |
| `PagedFusedUpdateCache` | `page_table` *(optional)* | **Case 1** *(DRAM)* / **clean** *(L1-sharded)* | Same split, driven by `page_table_is_dram`: borrowed `c_4` CB at [`paged_tiled_fused_update_cache_program_factory.cpp:280-291`](device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp#L280-L291), and the reader either NoC-reads via `TensorAccessor(page_table_args, …)` or just offsets the borrowed CB's write pointer ([`reader_paged_fused_update_cache_interleaved_start_id.cpp:108-119`](device/kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp#L108-L119)). |

**Op-level roll-up: ⚠ port work** — 10 Case-1 bindings (mechanical: express as `TensorParameter` /
`TensorBinding`, build `TensorAccessor(tensor::name)` kernel-side, and the address RTA plus its
`TensorAccessorArgs` plumbing both disappear), 3 clean borrowed-memory bindings, and 2 bindings whose
classification flips with the DRAM-vs-L1-sharded config. **No Case 2 anywhere**: no kernel in the op
uses a tensor base address for hand-rolled NoC arithmetic, so the `get_bank_base_address` bridge is
never needed.

**TensorParameter relaxation:** `none` — the port applies no relaxation. No analysis doc is expected or
required.

**TensorAccessor 3rd arg:** none — no accessor passes a 3rd argument.

### CB endpoints — per `(CB, config)` census

Every count below is per node, and each `(CB, config)` was classified separately. `P` = locked FIFO
producer, `C` = locked FIFO consumer.

**`PagedFillCacheProgramFactory` / `PagedFillCacheMeshWorkloadFactory`** (identical CB layout; the
mesh variant differs only in the `noop` runtime arg):

| CB | Config | Touchers | Verdict | Port-time resolution |
|---|---|---|---|---|
| `c_0` input tiles | all | reader P, writer C | plain 1:1 | none |
| `c_1` page table | all | writer only (`reserve_back` + `get_write_ptr`, no `push_back`) | single-ended | **self-loop** — bind the writer PRODUCER *and* CONSUMER |
| `c_2` batch_idx | `use_batch_idx_tensor` | writer only | single-ended | **self-loop**; **conditional DFB** (absent when no `batch_idx_tensor`) |
| `c_3` valid_seq_len | `use_valid_seq_len` | writer only | single-ended | **self-loop**; **conditional DFB** (absent when no `valid_seq_len_tensor`) |

**`PagedUpdateCacheProgramFactory` / `PagedUpdateCacheMeshWorkloadFactory`**:

| CB | Config | Touchers | Verdict | Port-time resolution |
|---|---|---|---|---|
| `c_0` cache tiles | all | reader P, compute C | plain 1:1 | none |
| `c_1` input shard *(borrowed)* | all | reader P, compute C | plain 1:1 | `borrowed_from` the `input_tensor` parameter |
| `c_2` index | `use_index_tensor` | reader P, writer C | plain 1:1 | **conditional DFB** |
| `c_3` page table | `is_paged_cache` | reader P, writer C | plain 1:1 | **conditional DFB** |
| `c_24` untilized cache | all | compute P, writer C | plain 1:1 | none |
| `c_25` untilized cache 2 | all | writer P, compute C | plain 1:1 | none |
| `c_26` untilized input | all | compute P, writer C | plain 1:1 | none |
| `c_16` output tiles | all | compute P, writer C | plain 1:1 | none |

**`PagedTiledFusedUpdateCacheProgramFactory` / `…MeshWorkloadFactory`** (census taken on a node in
`input1_cores`; a node in `input2_cores` is the mirror image with `c_2` in place of `c_1`):

| CB | Config | Touchers | Verdict | Port-time resolution |
|---|---|---|---|---|
| `c_0` cache tiles | all | reader P, compute C | plain 1:1 | none |
| `c_1` / `c_2` input shard *(borrowed)* | all | reader P, compute C | plain 1:1 | `borrowed_from`; see the runtime-CB-select heads-up |
| `c_3` index | `use_index_tensor` | reader P, writer C | plain 1:1 | **conditional DFB**; `borrowed_from` only when L1-sharded |
| `c_4` page table | `is_paged_cache` | reader P, writer C | plain 1:1 | **conditional DFB**; `borrowed_from` only when L1-sharded |
| `c_24` untilized cache | all | compute P, writer C | plain 1:1 | none |
| `c_25` untilized cache 2 | all | writer P, compute C | plain 1:1 | none |
| `c_26` untilized input | all | compute P, writer C | plain 1:1 | none |
| `c_16` output tiles | all | compute P, writer C | plain 1:1 | none |

**`PagedRowMajorFusedUpdateCacheProgramFactory` / `…MeshWorkloadFactory`** (row-major input needs no
untilize step, so the input CB is drained by the *writer*, not by compute):

| CB | Config | Touchers | Verdict | Port-time resolution |
|---|---|---|---|---|
| `c_0` cache tiles | all | reader P, compute C | plain 1:1 | none |
| `c_1` / `c_2` input shard *(borrowed)* | all | reader P, writer C | plain 1:1 | `borrowed_from`; see the runtime-CB-select heads-up |
| `c_3` index | `use_index_tensor` | reader P, writer C (`wait_front` with no `pop_front`) | plain 1:1 | **conditional DFB**; `borrowed_from` only when L1-sharded |
| `c_4` page table | `is_paged_cache` | reader P, writer C (`wait_front` with no `pop_front`) | plain 1:1 | **conditional DFB**; `borrowed_from` only when L1-sharded |
| `c_5` untilized cache | all | compute P, writer C | plain 1:1 | none |
| `c_6` untilized cache 2 | all | writer P, compute C | plain 1:1 | none |
| `c_7` output tiles | all | compute P, writer C | plain 1:1 | none |

### Fused kernel placement — the `unused_cores` instances drop out

**This is forced port work, not a choice, and it is a net gain.** Both fused factories place reader,
writer and compute over `all_cores_bb`, the bounding box of the two input shard grids, and give each
core in `unused_cores` (inside the box, in neither grid) a one-element runtime-arg vector
`{!has_work}` so its kernels return at once
([`paged_tiled_fused_update_cache_program_factory.cpp:524-530`](device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp#L524-L530),
[`paged_row_major_fused_update_cache_program_factory.cpp:519-525`](device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp#L519-L525)).
Six of the tiled factory's eight CB descriptors span the box, and five of the row-major factory's
seven (four and three of them unconditionally), so those cores reserve SRAM they never touch.

Metal 2.0 cannot express that shape. A DFB has no core range of its own — its node set is **derived**
from the union of its bound kernels' `WorkUnitSpec::target_nodes`
([`../../../../../../tt_metal/api/tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp:57-58`](../../../../../../tt_metal/api/tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp#L57-L58))
— and the two input DFBs are `borrowed_from` tensors sharded on `input1_cores` / `input2_cores` only.
A kernel placed over `all_cores_bb` that binds `c_1` would extend that borrow onto cores where the
input tensor has **no allocation at all**, and the spec requires the bound object to have L1 storage
large enough for the DFB. A third `WorkUnitSpec` for the idle cores does not rescue it either: a
kernel must bind every DFB its source names. So narrowing each `WorkUnitSpec` to one shard grid is the
only expressible shape, and `unused_cores` necessarily falls outside both.

**What that changes:** no output value moves — those cores early-exit today and are absent tomorrow.
The op dispatches to fewer cores, and stops reserving roughly **59–112 KB on each core of
`unused_cores`**. In the llama3-70b galaxy decode configuration that set holds **20 cores** (derivation
below, in *Team-only → `unused_cores` in a shipping configuration*). The only observable differences
are in tooling: a profiler trace shows fewer cores and an SRAM report shows less reserved. **Record
both in the port report**; neither needs a decision.

## Heads-ups  *(mirrors the brief)*

- **Conditionally-bound resources are the largest single piece of port work, and the gate has to move
  to the preprocessor.** Eight CBs and four tensors exist only under a host-side config flag, while the
  kernels reference their names *unconditionally at C++ scope* and gate only the *use* behind
  `if constexpr`. Because `if constexpr` in a non-template function still performs name lookup on the
  discarded branch, the generated `dfb::` / `tensor::` token has to exist at parse time — so each of
  these gates must be promoted to an `#ifdef` fed from `KernelSpec::compiler_options.defines`. The
  unconditional declarations to fix:
  - [`device/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp:56-57`](device/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp#L56-L57) — `cb_index`, `cb_page_table` declared before the `if constexpr (use_index_tensor)` at `:73` and the `if constexpr (is_paged_cache)` at `:94`
  - [`device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp:61-62`](device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp#L61-L62) — same pair
  - [`device/kernels/dataflow/writer_fill_cache_interleaved.cpp:92-93`](device/kernels/dataflow/writer_fill_cache_interleaved.cpp#L92-L93) — `cb_batch_idx`, `cb_valid_seq_len`
  - [`device/kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp:69-70`](device/kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp#L69-L70) and [`writer_paged_fused_update_cache_interleaved_start_id.cpp:66-67`](device/kernels/dataflow/writer_paged_fused_update_cache_interleaved_start_id.cpp#L66-L67)
  - [`device/kernels/dataflow/reader_paged_row_major_fused_update_cache_interleaved_start_id.cpp:69-70`](device/kernels/dataflow/reader_paged_row_major_fused_update_cache_interleaved_start_id.cpp#L69-L70) and [`writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp:73-74`](device/kernels/dataflow/writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp#L73-L74)

  The four optional **tensors** (`update_idxs_tensor`, `page_table`, `batch_idx_tensor_opt`,
  `valid_seq_len_tensor_opt`) are the mandatory case: the factories pass `TensorAccessorArgs(nullptr)`
  for an absent one ([`device/fill_cache/paged_fill_cache_program_factory.cpp:261-264`](device/fill_cache/paged_fill_cache_program_factory.cpp#L261-L264),
  [`device/update_cache/paged_update_cache_program_factory.cpp:309-311`](device/update_cache/paged_update_cache_program_factory.cpp#L309-L311)),
  and in Metal 2.0 an absent tensor produces no `tensor::` token at all, so there is nothing to bind
  even in principle.

- **The fused factories select their input CB index at runtime, from a kernel placed over a core range
  wider than the CB itself.** Both fused factories put reader, writer and compute over
  `all_cores_bb` (the bounding box of the two input shard grids) while the input CBs `c_1` and `c_2`
  are scoped to `input1_cores` and `input2_cores` respectively:
  - tiled: kernels at [`device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp:370`](device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp#L370), `:380`, `:389`; CBs at `:205` and `:215`
  - row-major: kernels at [`device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp:366`](device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp#L366), `:376`, `:386`; CBs at `:210` and `:220`

  Each kernel then picks which of the two to use from a per-core runtime arg, `is_input1`:
  [`reader_paged_fused_update_cache_interleaved_start_id.cpp:32-35`](device/kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp#L32-L35),
  [`compute/paged_fused_update_cache.cpp:21-25`](device/kernels/compute/paged_fused_update_cache.cpp#L21-L25) and `:39-55`,
  [`reader_paged_row_major_fused_update_cache_interleaved_start_id.cpp:32-35`](device/kernels/dataflow/reader_paged_row_major_fused_update_cache_interleaved_start_id.cpp#L32-L35),
  [`writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp:58-61`](device/kernels/dataflow/writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp#L58-L61).
  Cores in `all_cores_bb` that belong to neither shard grid get a single-element runtime-arg vector
  `{!has_work}` and early-exit
  ([`paged_tiled_fused_update_cache_program_factory.cpp:524-530`](device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp#L524-L530),
  [`paged_row_major_fused_update_cache_program_factory.cpp:519-525`](device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp#L519-L525)),
  so one kernel carries two different runtime-arg-vector lengths across cores.

  A Metal 2.0 `dfb::` token is a compile-time name, so the runtime select has to become structural. The
  documented translation is the one the *Anti-pattern: Demoting per-group CTA to RTA* entry in
  `shared/port_patterns.md` states in reverse — two `WorkUnitSpec`s over the disjoint node sets, each
  with its own reader / writer / compute `KernelSpec` of the same source, differing only in which input
  DFB it binds. That makes the CB choice compile-time, dissolves the `is_input1` runtime arg into
  structure, and drops the ragged arg vectors. **Why that placement is the only expressible one, and
  what it does to the `unused_cores` instances, is in *Port-work summary → Fused kernel placement*** —
  do not re-derive it here.

- **Cross-op / shared kernels:** none. The op owns all 11 of its kernel files, no other op in the tree
  instantiates any of them, and no `_metal2` fork exists beside any of them — this port creates the
  first for each. There is also **no `paged_cache` copy under `ttnn/cpp/ttnn/operations/experimental/quasar/`**,
  so there is no shortcut-port lookalike to be misled by. The only out-of-directory code the kernels
  call is `compute_kernel_lib::untilize` / `compute_kernel_lib::tilize`, both of which take their CB
  handles as `uint32_t` non-type template parameters already named `input_dfb` / `output_dfb` and
  construct `DataflowBuffer` objects internally — pass the `dfb::` tokens straight through; no
  donor-side change and no fork needed.

- **RTA varargs:** none. Every runtime-arg read in the op is a distinct field at a fixed position —
  either a constant index (`get_arg_val<uint32_t>(0)` … `(6)` in the `fill_cache` and `update_cache`
  kernels) or a fixed run of `get_arg_val<uint32_t>(rt_args_idx++)` at the top of the kernel (the six
  fused kernels). There is no counted loop over runtime args, no data-selected index, and no
  `get_common_arg_val` / `common_runtime_args` use anywhere. All of them get names.

- **CTA varargs:** none. No `get_compile_time_arg_val` is read at a varying index; every compile-time
  arg is read at a literal constant offset.

- **The Metal 2.0 `DataflowBuffer` exposes its own tile/format metadata accessors.** The kernels
  currently read tile size through the Device 2.0 `CircularBuffer` wrapper (`cb_cache.get_tile_size()`,
  8 sites). Moving those onto the DFB object is a port-stage change, not a Device 2.0 one; confirm the
  DFB equivalent rather than swapping blind. The `#include "api/dataflow/circular_buffer.h"` in each
  kernel becomes the DFB header at the same time.

- **The `noop` / `has_work` runtime gates carry the mesh-coordinate filtering, and the three factories
  do it two different ways.** `fill_cache` builds a full program for an excluded coordinate and sets a
  `noop` runtime arg that both kernels test
  ([`device/fill_cache/paged_fill_cache_program_factory.cpp:33-40`](device/fill_cache/paged_fill_cache_program_factory.cpp#L33-L40),
  [`reader_fill_cache_interleaved.cpp:23-25`](device/kernels/dataflow/reader_fill_cache_interleaved.cpp#L23-L25)),
  and its `override_runtime_arguments` re-derives that flag per coordinate on every cache hit
  ([`paged_fill_cache_program_factory.cpp:399`](device/fill_cache/paged_fill_cache_program_factory.cpp#L399)) — which
  rides cleanly on a per-coordinate `ProgramRunArgs`. `update_cache` and both fused factories instead
  return an **empty `ProgramDescriptor`** for an excluded coordinate
  ([`paged_update_cache_program_factory.cpp:448-453`](device/update_cache/paged_update_cache_program_factory.cpp#L448-L453),
  [`paged_tiled_fused_update_cache_program_factory.cpp:544-549`](device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp#L544-L549),
  [`paged_row_major_fused_update_cache_program_factory.cpp:542-548`](device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp#L542-L548)) —
  a *structurally* different program per coordinate. See *Question for the user*.

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ✓ clean.** No donor needs work, no fork, no cross-team coordination.

Every `#include` in every kernel of the op resolves to one of exactly two donor classes:

| Donor class | Headers | Concern |
|---|---|---|
| 1 — `tt_metal/*` LLK / HAL / firmware | `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/dataflow/endpoints.h`, `api/dataflow/noc_semaphore.h`, `api/core_local_mem.h`, `api/tensor/noc_traits.h`, `api/compute/common.h`, `api/compute/pack_untilize.h`, `api/compute/tilize.h` | none |
| 2 — `ttnn/cpp/ttnn/kernel_lib/` official shared kernel library | `untilize_helpers.hpp`, `tilize_helpers.hpp` | lib team handles internally |

There is **no** class-3 (`ttnn/cpp/ttnn/kernel/`), class-4 (`kernel_helper_functions/`), in-family
shared, or cross-family donor.

**Summary table** — one row per (op kernel, donor file) pair:

| Op kernel | Donor file | Class | Status |
|---|---|---|---|
| `compute/update_cache.cpp` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` | 2 | ✓ |
| `compute/update_cache.cpp` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` | 2 | ✓ |
| `compute/paged_fused_update_cache.cpp` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` | 2 | ✓ |
| `compute/paged_fused_update_cache.cpp` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` | 2 | ✓ |
| `compute/paged_row_major_fused_update_cache.cpp` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` | 2 | ✓ |
| `compute/paged_row_major_fused_update_cache.cpp` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` | 2 | ✓ |
| all 8 dataflow kernels | `tt_metal/hw/inc/api/dataflow/*`, `api/core_local_mem.h`, `api/tensor/noc_traits.h` | 1 | ✓ |
| all 3 compute kernels | `tt_metal/hw/inc/api/compute/*` | 1 | ✓ |

**Per-call detail** — omitted for the class-1 LLK donors. For the two class-2 donors:

| Donor function | Handle shape in signature | Status | Notes |
|---|---|---|---|
| `compute_kernel_lib::untilize<block_width_tiles, input_dfb, output_dfb, …>(uint32_t num_blocks)` | `uint32_t` CB/DFB index as a non-type template parameter | ✓ OK | `dfb::name`'s constexpr cast covers template-parameter position. The donor builds `DataflowBuffer in_dfb(input_dfb)` / `out_dfb(output_dfb)` internally — the ✓ *`DataflowBuffer`* row, not the ⭐ `CircularBuffer` one. No donor-side change, no fork. |
| `compute_kernel_lib::tilize<block_width_tiles, input_dfb, output_dfb, …>(uint32_t num_blocks, std::optional<uint32_t>)` | same | ✓ OK | same |
| `compute_kernel_hw_startup(uint32_t icb0, uint32_t ocb)` | `uint32_t` CB index, runtime position | ✓ OK | class-1 LLK. Called with a *runtime-selected* index in `compute/paged_fused_update_cache.cpp:36` — see the runtime-CB-select heads-up; the resolution there makes the argument compile-time again. |

**Borrowed kernel files (file-path kernel instantiation):** **none.** All 11 `kernel_source` paths the four
program-building factories name point inside `device/kernels/` of this op. No other op instantiates any of them
(a repo-wide grep for `paged_cache/device/kernels` finds referrers only inside this op directory), so
there is no sunset list and no cross-op coordination cost. No `_metal2` fork exists beside any of the
11 files; this port creates the first one for each.

### `unused_cores` in a shipping configuration — 20 cores, derived

Backs the *Fused kernel placement* item in Port-work. The set is **non-empty in the llama3-70b galaxy
decode path**, and the chain has no unverified step:

1. `nlp_create_qkv_heads_decode` places the two tensors as **two consecutive `batch`-length row-major
   runs** over the create-head output grid: `v` (sharing q's grid) takes the first `batch` cores, `k`
   takes `batch` cores starting one core past q's run
   ([`nlp_create_qkv_heads_decode_device_operation.cpp:134-151`](../../../../../../ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/nlp_create_qkv_heads_decode_device_operation.cpp#L134-L151)).
   The walk fills each `CoreRange` left-to-right then row-by-row, ranges in declaration order
   ([`work_split.cpp:150-200`](../../../../../../tt_metal/common/work_split.cpp#L150-L200)).
2. `rotary_embedding_llama` defaults its output memory config to the **input's**
   ([`rotary_embedding_llama_device_operation.cpp:249-256`](../../../../../../ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/rotary_embedding_llama_device_operation.cpp#L249-L256)),
   and the model calls it without one
   ([`llama_attention.py:857-863`](../../../../../../models/demos/llama3_70b_galaxy/tt/llama_attention.py#L857-L863)),
   so `k` reaches the fused op still on the create-head grid.
3. That grid is `CREATE_HEAD_OUTPUT_MEMCFG`'s: the ranges `(1,0)`–`(3,9)` and `(5,0)`–`(6,9)`
   ([`model_config.py:2987-2992`](../../../../../../models/demos/llama3_70b_galaxy/tt/model_config.py#L2987-L2992),
   used at [`model_config.py:3014-3025`](../../../../../../models/demos/llama3_70b_galaxy/tt/model_config.py#L3014-L3025)).
4. `v` does **not** stay on it: the model reshards `v` to `(5,0)`–`(6,3)`, 8 cores
   ([`llama_attention.py:884-895`](../../../../../../models/demos/llama3_70b_galaxy/tt/llama_attention.py#L884-L895)).
   That reshard also pins `batch`: `v` carries `num_kv_heads_padded × batch = 32 × batch` rows, and 8
   cores of `[32, head_dim]` hold 256, so **`batch = 8`**.

Walking step 1 with `batch = 8` puts `k` on `(3,2)`, `(1,3)`, `(2,3)`, `(3,3)`, `(1,4)`, `(2,4)`,
`(3,4)`, `(1,5)` — disjoint from `v`'s `(5,0)`–`(6,3)` and the same core count, so both of the op's own
asserts pass
([`paged_fused_update_cache_device_operation.cpp:352-357`](device/fused_update_cache/paged_fused_update_cache_device_operation.cpp#L352-L357)),
an independent check that this is a real configuration. Union = 16 cores; bounding box `(1,0)`–`(6,5)`
= 36 cores; **`unused_cores` = 20 cores**, including all of column 4, which the create-head grid
excludes outright. `Wt = head_dim / 32 = 4` and `B = batch = 8` there, the same arithmetic as the
unit-test shapes, so the per-core reservation is **~59–112 KB** across the row-major/tiled factories
and a `bfloat8_b`/`bfloat16` cache.

**Why no local test shows this.** When *both* grids come straight from `nlp_create_qkv_heads_decode`
the union is a contiguous row-major prefix of length `2 × batch`, so on a full-width rectangular grid
`unused_cores` is empty whenever `2 × batch` is a multiple of the grid width — true of every
power-of-two batch on an 8-wide grid. Computed over the walk above: an 8×8 Wormhole grid is empty at
`batch` 4, 8, 16 and 32, which is why the unit tests show nothing. The set turns non-empty on a
**non-rectangular sub-core grid** (the galaxy grid gives 1 core at `batch = 4`, 2 at `batch = 8`) or
when a caller **reshards one input to a separate block** — the latter is what makes llama3-70b galaxy
an order of magnitude worse at 20 cores.

### Relaxation candidates (FYI-U — fallible)

**None found.** All three custom hashes feed `tensor_args` wholesale into
`operation::hash_operation<…>`, so the full `TensorSpec` of every input participates in the key — the
hashes narrow nothing on the tensor side. What they exclude is scalar operation attributes
(`update_idxs`, `batch_offset`, `batch_idx_fallback`, `noop`), which the factories then re-apply on
every cache hit through `override_runtime_arguments`. That is a runtime-arg exclusion, not a
tensor-property relaxation, so it yields no relaxation candidate. Consistent with the sheet's
`TensorParameter relaxation = none` and its blank `Provisional relaxation finding (Edwin)` cell.

### TTNN factory analysis

Sheet-derived facts with `file:line` evidence. All of them are the non-gating kind; the gate conjuncts
(`get_dynamic_runtime_args`, a relaxation that neither clears nor points at an analysis, genuine
multi-program) are all absent.

- **Current concept:** `descriptor` on all 8 factory rows. Confirmed in code: each factory defines
  `create_descriptor` returning `tt::tt_metal::ProgramDescriptor`. The four `*MeshWorkloadFactory`
  variants take an extra `const std::optional<ttnn::MeshCoordinate>&` and are dispatched per
  coordinate by the framework's descriptor adapter, which has explicit support for that signature
  ([`../../../../../../ttnn/api/ttnn/mesh_device_operation_adapter.hpp:427-444`](../../../../../../ttnn/api/ttnn/mesh_device_operation_adapter.hpp#L427-L444)).
  They are still `descriptor`, not `WorkloadDescriptor`.
- **Op-owned tensors:** none. No factory returns a `WorkloadDescriptor`, so no `buffers` vector exists.
- **MeshWorkload need:** none of the artifact kind — the four `*MeshWorkloadFactory` variants exist to
  filter mesh coordinates, not to carry op-owned tensors.
- **Custom hash:** present on all three device-ops (sites in the status summary). The port leaves each
  exactly as it is. Each documents what it excludes and why, and each factory's
  `override_runtime_arguments` re-applies the excluded values on every cache hit — the two halves are
  paired by design, so the porter must keep them paired.
- **`get_dynamic_runtime_args`:** absent.
- **`override_runtime_arguments`:** present on all 8 factories. The porter translates each into one
  returning a `ProgramRunArgs`; the four `*MeshWorkloadFactory` bodies simply delegate to their
  single-device sibling, so there are **four** distinct patch bodies to translate, not eight — and the
  two fused ones are thin wrappers over a single shared `patch_runtime_args` helper.
- **Pybind `create_descriptor` / other risky pybind:** none. [`paged_cache_nanobind.cpp:23-165`](paged_cache_nanobind.cpp#L23-L165)
  exposes only the three user-facing op functions and their keyword arguments. No user-visible API
  change falls out of this port.
- **Target concept:** `CustomProgramSpecFactoryConcept`, derived from `Concept == descriptor` plus
  `Override runtime args method? == yes`, and matching the sheet's own `Porting Target` column on all 8
  rows. See *Question for the user* for the one wrinkle in that mapping.

## Misc anomalies  *(team-only, non-gating; the port does not act on these)*

These are latent code issues noticed while reading every line of the op. They route to the ops team;
the port diff should not touch them.

- **Dead compile-time arg `max_blocks_per_seq`, in six kernels.** Declared and never used:
  [`reader_update_cache_interleaved_start_id.cpp:37`](device/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp#L37),
  [`writer_update_cache_interleaved_start_id.cpp:40`](device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp#L40),
  [`reader_paged_fused_update_cache_interleaved_start_id.cpp:51`](device/kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp#L51),
  [`writer_paged_fused_update_cache_interleaved_start_id.cpp:46`](device/kernels/dataflow/writer_paged_fused_update_cache_interleaved_start_id.cpp#L46),
  [`reader_paged_row_major_fused_update_cache_interleaved_start_id.cpp:51`](device/kernels/dataflow/reader_paged_row_major_fused_update_cache_interleaved_start_id.cpp#L51),
  [`writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp:48`](device/kernels/dataflow/writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp#L48).
  The kernels bound the page-table index with `virtual_block_id` arithmetic and never range-check it,
  so the value looks like a bounds check that was intended and never wired up.
- **Dead compile-time arg `log_base_2_of_page_size`, in three readers**, always fed the literal `0` by
  the host: declared at [`reader_update_cache_interleaved_start_id.cpp:29`](device/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp#L29),
  [`reader_paged_fused_update_cache_interleaved_start_id.cpp:43`](device/kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp#L43),
  [`reader_paged_row_major_fused_update_cache_interleaved_start_id.cpp:43`](device/kernels/dataflow/reader_paged_row_major_fused_update_cache_interleaved_start_id.cpp#L43);
  host side never assigns the variable it comes from
  ([`paged_update_cache_program_factory.cpp:116`](device/update_cache/paged_update_cache_program_factory.cpp#L116),
  [`paged_tiled_fused_update_cache_program_factory.cpp:111`](device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp#L111),
  [`paged_row_major_fused_update_cache_program_factory.cpp:111`](device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp#L111)).
- **Dead compile-time arg `log2_page_table_stick_size`, in four kernels**
  ([`reader_update_cache_interleaved_start_id.cpp:38`](device/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp#L38),
  [`writer_fill_cache_interleaved.cpp:47`](device/kernels/dataflow/writer_fill_cache_interleaved.cpp#L47),
  [`reader_paged_fused_update_cache_interleaved_start_id.cpp:52`](device/kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp#L52),
  [`reader_paged_row_major_fused_update_cache_interleaved_start_id.cpp:52`](device/kernels/dataflow/reader_paged_row_major_fused_update_cache_interleaved_start_id.cpp#L52)).
  In `update_cache` and both fused factories the host leaves it at `0`. In `fill_cache` the host
  actually computes it with `std::log2` at
  [`paged_fill_cache_program_factory.cpp:119`](device/fill_cache/paged_fill_cache_program_factory.cpp#L119)
  — a dead host-side floating-point call. The `TT_FATAL` immediately above it
  ([`:116-118`](device/fill_cache/paged_fill_cache_program_factory.cpp#L116-L118)) requires the page
  table's page size to be a multiple of 32; its stated reason is address alignment, so it may well be a
  genuine guard independent of the dead log2 value — worth a look by whoever owns the file rather than
  removing both together.
- **Dead compile-time args and a dead runtime arg in the row-major fused compute kernel.** CTAs 0 and 1
  (`in1_cb`, `in2_cb`) exist only to compute `in_cb`, which is `[[maybe_unused]]` and never read; the
  `is_input1` runtime arg exists only to choose between them
  ([`compute/paged_row_major_fused_update_cache.cpp:19-25`](device/kernels/compute/paged_row_major_fused_update_cache.cpp#L19-L25)).
  In the row-major path the input CB is drained by the writer, not by compute, so this whole block is a
  leftover from the tiled kernel it was derived from. The host still emits both CTAs
  ([`paged_row_major_fused_update_cache_program_factory.cpp:349-357`](device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp#L349-L357))
  and the `is_input1` runtime arg on every core.
- **Shadowed-and-dead `update_idx` in three writers.** An outer `uint32_t update_idx = 0;` is declared
  and never read, then shadowed by an inner `const uint32_t update_idx` that does all the work:
  [`writer_update_cache_interleaved_start_id.cpp:67`](device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp#L67) vs `:84`,
  [`writer_paged_fused_update_cache_interleaved_start_id.cpp:72`](device/kernels/dataflow/writer_paged_fused_update_cache_interleaved_start_id.cpp#L72) vs `:80`,
  [`writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp:79`](device/kernels/dataflow/writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp#L79) vs `:87`.
  Harmless today; a shadowing pair like this is the shape a real read-of-the-wrong-variable bug takes.
- **Dead runtime arg and dead accessor in the L1-sharded index / page-table configs of the fused
  factories.** The reader is handed the index tensor's address whether or not the tensor is sharded
  ([`paged_fused_update_cache_device_operation.cpp:85-88`](device/fused_update_cache/paged_fused_update_cache_device_operation.cpp#L85-L88)
  says so explicitly), and it builds `TensorAccessor(index_tensor_args, index_tensor_addr)`
  unconditionally at
  [`reader_paged_fused_update_cache_interleaved_start_id.cpp:87`](device/kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp#L87)
  while using it only inside `if constexpr (index_is_dram)`. In the sharded config both the argument and
  the accessor are dead.
- **`paged_fill_cache` accepts a `compute_kernel_config` argument and discards it.** The parameter is
  exposed all the way out to Python ([`paged_cache_nanobind.cpp:23-165`](paged_cache_nanobind.cpp#L23-L165)) and then
  dropped with an explicit `(void)` cast at
  [`paged_cache.cpp:80-81`](paged_cache.cpp#L80-L81); `PagedFillCacheParams` has no such field, so it
  never reaches the device-op or the hash. A caller passing one gets no error and no effect.
- **The `share_cache` sequencing semaphore is allocated unconditionally**
  ([`paged_update_cache_program_factory.cpp:246-252`](device/update_cache/paged_update_cache_program_factory.cpp#L246-L252),
  [`paged_tiled_fused_update_cache_program_factory.cpp:259-265`](device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp#L259-L265),
  [`paged_row_major_fused_update_cache_program_factory.cpp:255-261`](device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp#L255-L261)),
  even though every `wait_to_start` / `send_signal` flag is `false` when `share_cache` is `false`, and
  `share_cache` is part of the program hash. A conditional allocation would be expressible; today the
  non-shared path pays for a semaphore it never touches.
- **Asymmetric CB drain between the two fused writers.** The tiled writer `pop_front`s the index and
  page-table CBs after reading them
  ([`writer_paged_fused_update_cache_interleaved_start_id.cpp:112`](device/kernels/dataflow/writer_paged_fused_update_cache_interleaved_start_id.cpp#L112),
  `:122`); the row-major writer `wait_front`s the same two CBs and never pops
  ([`writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp:84`](device/kernels/dataflow/writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp#L84),
  `:95`). Both work because each CB is filled and read exactly once per program execution, so the
  FIFO never wraps — but the two siblings disagree on an invariant, which makes either one harder to
  change safely.

## Per-DeviceOperation attribution

Findings are uniform across the three DeviceOperations except where noted. The differences:

| Field | `PagedFillCacheDeviceOperation` | `PagedUpdateCacheDeviceOperation` | `PagedFusedUpdateCacheDeviceOperation` |
|---|---|---|---|
| `Is able to port?` | yes (2 rows) | yes (2 rows) | yes (4 rows) |
| Overall | GREEN | GREEN | GREEN |
| Compute kernel | none | `compute/update_cache.cpp` | one per layout variant |
| Semaphores | none | 1 (`share_cache` sequencing) | 1 per factory (`share_cache` sequencing) |
| Borrowed-memory CBs | none | 1 (`c_1`, input shard) | 2 unconditional (`c_1`, `c_2`) + 2 conditional (index, page table — only when L1-sharded) |
| Tensor bindings | 5, all Case 1 | 3 Case 1 + 1 clean | 2 Case 1 + 2 clean + 2 config-split |
| Conditional DFBs | 2 (batch_idx, valid_seq_len) | 2 (index, page table) | 2 per factory (index, page table) |
| Self-loop CBs | 3 | none | none |
| Runtime-selected CB index | no | no | **yes** — both factories |
| `unused_cores` (bounding-box placement) | none — no bounding box | none — placement is the input shard grid | **yes** — 20 cores in llama3-70b galaxy decode |
| Mesh-coordinate exclusion | `noop` runtime arg on a full program — already the recommended shape | empty `ProgramDescriptor` — subject of the open question | empty `ProgramDescriptor` — subject of the open question |
| `cache_position_modulo` support | yes | yes | no |

## Question for the user

**`mesh_coords` exclusion on the three empty-descriptor mesh variants — one narrow question, with a
   recommended default.**

   **The situation.** `PagedUpdateCacheMeshWorkloadFactory` and the two mesh variants of the fused op
   return an **empty `ProgramDescriptor`** for a coordinate outside `mesh_coords` — no kernels, no CBs
   ([`paged_update_cache_program_factory.cpp:448-453`](device/update_cache/paged_update_cache_program_factory.cpp#L448-L453),
   [`paged_tiled_fused_update_cache_program_factory.cpp:544-549`](device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp#L544-L549),
   [`paged_row_major_fused_update_cache_program_factory.cpp:542-548`](device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp#L542-L548)).
   That is a structurally different program per coordinate, while both Metal 2.0 spec concepts are
   SPMD-shaped — one `ProgramSpec` replicated across the mesh
   ([`../../../../../../ttnn/api/ttnn/operation_concepts.hpp:134-140`](../../../../../../ttnn/api/ttnn/operation_concepts.hpp#L134-L140)).
   `PagedFillCacheMeshWorkloadFactory` is unaffected: its exclusion already rides on a `noop` runtime
   arg its `override_runtime_arguments` re-derives per coordinate
   ([`paged_fill_cache_program_factory.cpp:399`](device/fill_cache/paged_fill_cache_program_factory.cpp#L399)),
   which is exactly what a replicated spec plus per-coordinate `ProgramRunArgs` expresses.

   **`mesh_coords` is a public, exposed parameter on all three ops**
   ([`paged_cache_nanobind.cpp:70`](paged_cache_nanobind.cpp#L70), [`:115`](paged_cache_nanobind.cpp#L115),
   [`:161`](paged_cache_nanobind.cpp#L161)), so the feature has to keep working regardless of what this
   repo happens to call. In-repo usage is confined to `paged_update_cache` and `paged_fill_cache` from
   DeepSeek V3's MLA path
   ([`mla1d.py:2351-2412`](../../../../../../models/demos/deepseek_v3/tt/mla/mla1d.py#L2351-L2412),
   [`mla1d.py:2133-2146`](../../../../../../models/demos/deepseek_v3/tt/mla/mla1d.py#L2133-L2146)); no
   in-repo caller passes `mesh_coords` to the fused op. **That is a fact about this repo, not a basis
   for treating the fused mesh variants as dead** — an external consumer can reach them.

   **Recommended: translate the exclusion into a runtime gate, and port all eight factories.** On an
   excluded coordinate, build the full program and set an all-cores `has_work = 0`. The kernels need no
   change — each already reads `has_work` first and returns on 0
   ([`reader_paged_fused_update_cache_interleaved_start_id.cpp:16-20`](device/kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp#L16-L20))
   — and `fill_cache` is the working precedent in this same directory. Output is unchanged on every
   coordinate, so the zero-functional-change contract holds.

   **The question, and it is only this one:** an excluded chip goes from receiving nothing to receiving
   a program whose kernels launch and immediately return, plus that program's DFB reservation for the
   duration of its execution. Is that acceptable on the `mesh_coords` path?

   **Two things that make it easier to accept.** First, the SRAM half is removable without touching the
   kernels: `ProgramRunArgs` carries **per-execution DFB size overrides** (`entry_size`, `num_entries`
   on `DFBRunOverrides` —
   [`program_run_args.hpp:118-133`](../../../../../../tt_metal/api/tt-metalium/experimental/metal2_host_api/program_run_args.hpp#L118-L133)),
   and the custom concept's `override_runtime_arguments` runs per coordinate, so an excluded coordinate
   can shrink every DFB to minimum. **That is not port work** — it is new host logic the legacy op
   never had, so it belongs as a follow-up if measurement shows the cost matters. Two caveats for
   whoever does it: the overrides are *stateful across executions* (an included coordinate must set
   sizes explicitly, not rely on the spec value), and a borrowed-memory DFB needs its backing
   `tensor_arg` supplied alongside any size update — which the fused op's input DFBs are. Second, the
   residual dispatch cost is measurable rather than a matter of judgement.

   **Fallback if the answer is no.** Port the plain factories and **leave these three mesh variants on
   the legacy `ProgramDescriptor` path**. This is supported: the framework dispatches **per variant
   alternative**, one concept-constrained lambda each, so a `program_factory_t` may legitimately mix a
   `ProgramDescriptorFactoryConcept` alternative with a `CustomProgramSpecFactoryConcept` one
   ([`device_operation.hpp:239-269`](../../../../../../ttnn/api/ttnn/device_operation.hpp#L239-L269)).
   It preserves the `mesh_coords` path byte-for-byte at zero runtime cost, and matches the recipe's
   Code-path scope pattern. Its cost is **code duplication**: each mesh variant currently *delegates*
   its build to the plain factory
   ([`paged_update_cache_program_factory.cpp:454`](device/update_cache/paged_update_cache_program_factory.cpp#L454),
   [`paged_tiled_fused_update_cache_program_factory.cpp:550-551`](device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp#L550-L551),
   [`paged_row_major_fused_update_cache_program_factory.cpp:552-553`](device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp#L552-L553)),
   so once the plain factory returns a `ProgramSpec` the legacy `create_descriptor` body has to be
   retained separately — losing the "the two paths cannot drift" property those factories' own comments
   exist to protect. It also leaves 3 of 8 factories unported.

   **Not recommended: naming `MeshWorkloadSpecFactoryConcept` as the target.** It is the concept that
   fits the current shape, is fully implemented in the adapter
   ([`mesh_device_operation_adapter.hpp:1033-1102`](../../../../../../ttnn/api/ttnn/mesh_device_operation_adapter.hpp#L1033-L1102))
   and backed by a range-keyed `MeshWorkloadArtifacts`
   ([`metal_v2_artifacts.hpp:38-46`](../../../../../../ttnn/api/ttnn/metal_v2_artifacts.hpp#L38-L46)).
   But **no op in the tree uses it** and **no port procedure covers it** — a brief naming it makes the
   porter stop
   ([`metal2_port.md:90`](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#L90),
   [`ttnn_factory.md:179`](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/ttnn_factory.md#L179)).
   It leaves the same 3 factories unported as the fallback, without the fallback's benefit of keeping
   them working on a supported path.

   One argument bears on all of this: the TTNN-side guidance holds that a per-coordinate program shape
   is not by itself genuine multi-program — *"If every per-coord program is structurally identical …
   and the only thing pushing it multi-program was a resource workaround, the op is* morally
   *single-program and **ports cleanly**"*
   ([`ttnn_factory.md:181`](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/ttnn_factory.md#L181)).
   Here the per-coordinate difference is a *skip*, not different work.

## Recipe notes

- **The *TTNN porting shape* mapping has no row for a per-coordinate `create_descriptor`.** The
  subject derives the target from `Concept`, `Op-owned tensors?` and
  `Override runtime args method?` only. This op's four `*MeshWorkloadFactory` variants are
  `Concept == descriptor` and yet take a `const std::optional<ttnn::MeshCoordinate>&` and build a
  *different* descriptor per coordinate — a first-class shape the descriptor adapter supports
  explicitly ([`../../../../../../ttnn/api/ttnn/mesh_device_operation_adapter.hpp:427-444`](../../../../../../ttnn/api/ttnn/mesh_device_operation_adapter.hpp#L427-L444)),
  and one the recipe's three bullets do not distinguish from a plain single-program `descriptor`. The
  mapping is not *ambiguous* here — `descriptor` + override → `CustomProgramSpecFactoryConcept`, which
  the sheet's `Porting Target` column independently confirms — so this audit did not treat it as a
  recipe gap in the sense the section defines. But the mapping arrives at a concept the concept header
  calls SPMD-only, for a factory that is not, which is worth a fourth bullet: *`descriptor` whose
  `create_descriptor` takes a mesh coordinate and varies structurally per coordinate → look up whether
  `MeshWorkloadSpecFactoryConcept` is the target.* The open question above weighs that concept and does
  **not** recommend it, precisely because it has no in-tree consumer and no port procedure — which is
  the fact the lookup does not surface.

- **`ttnn_factory.md`'s multi-program lookup is now stale, and the stale line is the one that decides
  a RED.** That subject tells the auditor to grep `operation_concepts.hpp` for a Metal 2.0 concept
  admitting per-coordinate programs, then says *"As of 2026-08-31 there was none … If that is still
  the picture, record **target concept: none yet** and RED"*
  ([`ttnn_factory.md:177`](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/ttnn_factory.md#L177)).
  Run today, that grep finds `MeshWorkloadSpecFactoryConcept`
  ([`../../../../../../ttnn/api/ttnn/operation_concepts.hpp:118-129`](../../../../../../ttnn/api/ttnn/operation_concepts.hpp#L118-L129)),
  fully implemented in the adapter
  ([`mesh_device_operation_adapter.hpp:1033-1102`](../../../../../../ttnn/api/ttnn/mesh_device_operation_adapter.hpp#L1033-L1102)).
  So the picture has changed and the doc's own instruction correctly produces a target rather than a
  RED — the lookup design worked. Worth updating the dated sentence, and worth adding the fact the
  lookup does not surface: the concept has **zero in-tree consumers**, and no port procedure covers
  it, so naming it as a target stops the port rather than shaping it. An auditor who follows the
  lookup literally records a target and reads it as progress; the cost only becomes visible on
  reaching `metal2_port.md:90`. One clause at the lookup would close that gap.

- **The *CB endpoints* census asks "how many kernels touch this CB", but the harder question in this
  op was "which CB does this kernel touch".** The fused factories bind one kernel over a core range
  wider than the CB it uses and pick the CB index from a runtime arg. The endpoint count is perfectly
  legal (1P+1C on every node) and the census subject reports it as clean — yet the binding is the
  single largest piece of port work in the op, because a `dfb::` token is compile-time and a DFB's node
  set is derived from its bound kernels. A one-line prompt in the CB-endpoints recognition step —
  *"also check whether any kernel's CB index is a runtime value, or whether a kernel is placed over a
  wider node set than a CB it binds"* — would have surfaced it as a subject finding rather than as an
  ad-hoc heads-up. The resolution is already documented (the *Demoting per-group CTA to RTA*
  anti-pattern entry states the two-`WorkUnitSpec` shape), so the gap is in recognition, not in
  guidance.

- **Ragged per-core runtime-arg vectors have no home in any subject.** Both fused factories emplace an
  8- or 9-element arg vector on working cores and a 1-element `{!has_work}` vector on unused cores from
  the *same* `KernelDescriptor`
  ([`paged_tiled_fused_update_cache_program_factory.cpp:524-530`](device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp#L524-L530)).
  That is not RTA varargs by the subject's definition — every read is a distinct field at a fixed
  position — and it is not a CB or tensor finding either, so it ended up in *Heads-ups* by
  judgement. If `KernelSpec::runtime_arg_schema` requires one uniform shape per kernel, this is worth
  an explicit recognition line somewhere; if it does not, saying so would save the next auditor the
  same detour.

- **The self-loop recognition wording assumes the single toucher is sync-free.** The *CB endpoints*
  table's 1-toucher row is labelled "single-ended / sync-free", and the surrounding prose describes the
  single toucher as doing "pointer-only access". `fill_cache`'s three metadata CBs are a slightly
  different shape: the writer *is* the only toucher, but it issues a real FIFO op (`reserve_back(1)`)
  and then never `push_back`s
  ([`writer_fill_cache_interleaved.cpp:148-150`](device/kernels/dataflow/writer_fill_cache_interleaved.cpp#L148-L150)).
  The classification table settles it — one toucher is one toucher, so it is a self-loop regardless of
  role-locking — but the "sync-free" label in the row heading pulled the other way on first reading.
  Naming the row "one toucher" and leaving "sync-free" to the prose would remove the hesitation.
