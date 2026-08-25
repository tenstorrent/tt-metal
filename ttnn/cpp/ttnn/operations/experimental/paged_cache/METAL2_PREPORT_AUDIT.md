# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/paged_cache`

Three `DeviceOperation`s share this directory. They share **no** kernels and **no** program factories with each other (each family owns its own), so by the recipe's shared-code test they are independent; they are audited together here because the output contract is one report per *op directory*, they are one `Op` row-group (`experimental/paged_cache`) on the readiness sheet, and every finding below happens to be structurally common to all three. **Per-DeviceOperation attribution is given for every finding** (see the dedicated section) so a downstream consumer can extract per-DeviceOperation status. See *Recipe notes*.

- **`PagedUpdateCacheDeviceOperation`** (`device/update_cache/`)
  - `PagedUpdateCacheProgramFactory` (`paged_update_cache_program_factory.cpp:89`)
  - `PagedUpdateCacheMeshWorkloadFactory` (`paged_update_cache_program_factory.cpp:443`)
  - Kernels: `kernels/dataflow/reader_update_cache_interleaved_start_id.cpp`, `kernels/dataflow/writer_update_cache_interleaved_start_id.cpp`, `kernels/compute/update_cache.cpp`
- **`PagedFillCacheDeviceOperation`** (`device/fill_cache/`)
  - `PagedFillCacheProgramFactory` (`paged_fill_cache_program_factory.cpp:340`)
  - `PagedFillCacheMeshWorkloadFactory` (`paged_fill_cache_program_factory.cpp:348`)
  - Kernels: `kernels/dataflow/reader_fill_cache_interleaved.cpp`, `kernels/dataflow/writer_fill_cache_interleaved.cpp` (no compute kernel)
- **`PagedFusedUpdateCacheDeviceOperation`** (`device/fused_update_cache/`)
  - `PagedTiledFusedUpdateCacheProgramFactory` (`paged_tiled_fused_update_cache_program_factory.cpp:79`)
  - `PagedTiledFusedUpdateCacheMeshWorkloadFactory` (`paged_tiled_fused_update_cache_program_factory.cpp:539`)
  - `PagedRowMajorFusedUpdateCacheProgramFactory` (`paged_row_major_fused_update_cache_program_factory.cpp:79`)
  - `PagedRowMajorFusedUpdateCacheMeshWorkloadFactory` (`paged_row_major_fused_update_cache_program_factory.cpp:538`)
  - Kernels: `kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp`, `kernels/dataflow/writer_paged_fused_update_cache_interleaved_start_id.cpp`, `kernels/compute/paged_fused_update_cache.cpp`, `kernels/dataflow/reader_paged_row_major_fused_update_cache_interleaved_start_id.cpp`, `kernels/dataflow/writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp`, `kernels/compute/paged_row_major_fused_update_cache.cpp`

**All 11 kernel files in `device/kernels/` are referenced by a factory — none is unreferenced/dead.**

> **Naming caution for any reader:** the four `*MeshWorkloadFactory` types do **not** return a `WorkloadDescriptor`. Each declares `create_descriptor(...) -> ProgramDescriptor` with an extra `mesh_dispatch_coordinate` parameter and delegates to its single-device sibling, returning an empty (or noop) `ProgramDescriptor` for coordinates outside `operation_attributes.mesh_coords`. The concept is `descriptor`, not `WorkloadDescriptor`; the name is a legacy holdover. Do not go looking for `create_workload_descriptor` — there is none in this directory (verified by grep).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `50e992a8ec2 2026-08-21 docs(metal_2.0): a run in flight freezes the kernel sources`
*(The working checkout `/localdev/edwinlee/Paged_Cache_Port` does not carry `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`; the provenance command was run in the sibling doc-branch checkout `/localdev/edwinlee/Port_Recipe`, whose `ai/audit/metal2_audit.md` is byte-identical to the recipe supplied for this run.)*

**Readiness sheet:** fetched fresh this session from the *"Operations analysis"* sheet (file id `1KUMj8SyBGlNMZlLFgs1MbAZlO2g6EoUc4KaxSlcy8jw`), 8 rows for `experimental/paged_cache`.

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/paged_cache` |
| **Overall** | **RED** — blocked on the **Device 2.0** prerequisite. `RED at op level; no portable subset.` |
| **DOps / Factories** | `PagedUpdateCacheDeviceOperation` → {`PagedUpdateCacheProgramFactory`, `PagedUpdateCacheMeshWorkloadFactory`} · `PagedFillCacheDeviceOperation` → {`PagedFillCacheProgramFactory`, `PagedFillCacheMeshWorkloadFactory`} · `PagedFusedUpdateCacheDeviceOperation` → {`PagedTiledFusedUpdateCacheProgramFactory`, `PagedTiledFusedUpdateCacheMeshWorkloadFactory`, `PagedRowMajorFusedUpdateCacheProgramFactory`, `PagedRowMajorFusedUpdateCacheMeshWorkloadFactory`} |
| *Prereqs* — Device 2.0 (every kernel used) | **No (RED — routed to the Device 2.0 track). Isolated holdovers**, not broad Device 1.0: 6 × `get_dataformat(cb_id)` free-function calls, one line each, every one of them assigned to a variable that is never read (so the fix is a *deletion*, not even a swap). Idioms otherwise fully Device 2.0. |
| *Prereqs* — Cross-op escapes | **Ok** — donors are `tt_metal` `api/*` only, plus `ttnn/cpp/ttnn/kernel_lib/{tilize,untilize}_helpers.hpp` (official shared kernel library). No cross-family donors. **No borrowed kernel files** — every kernel this op instantiates lives in this op's own directory and is instantiated by no other op. |
| *Feature Support* — overall | **GREEN** — every Appendix A entry is `N/A` (feature absent). |
| *Feature Support* — GlobalCircularBuffer | N/A |
| *Feature Support* — CBDescriptor `address_offset` (non-zero) | N/A |
| *Feature Support* — GlobalSemaphore | N/A |
| *TTNN Readiness* — `Is able to port?` (the gate) | Sheet says **`no`** on all 8 rows. Attributable to exactly one blocking column: `TensorParameter relaxation`. **Cleared out-of-band** — the launching user reports the legality analysis is complete and the decision is *strict `TensorParameter`s, no relaxation*. See *Gate detail* → the sheet cell is stale on this op and needs a refresh from the sheet owner. |
| *TTNN Readiness* — Concept (current) | `descriptor` — all 8 factories. **Cross-check: confirmed.** |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A (`Concept == descriptor`). Sheet cell blank — consistent. |
| *TTNN Readiness* — Custom hash | **Yes** (not a gate; port leaves it intact): `device/update_cache/paged_update_cache_device_operation.cpp:313`, `device/fill_cache/paged_fill_cache_device_operation.cpp:207`, `device/fused_update_cache/paged_fused_update_cache_device_operation.cpp:371`. All three hash `tensor_args` in full. **Cross-check: confirmed.** |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** — grep over `device/` finds no such hook. **Cross-check: confirmed.** |
| *TTNN Readiness* — `override_runtime_arguments` | **Yes** (not a gate; selects `CustomProgramSpecFactoryConcept`) — all 8 factories. Bodies at `paged_update_cache_program_factory.cpp:457` & `:522`; `paged_fill_cache_program_factory.cpp:361` & `:420`; `paged_fused_update_cache_device_operation.cpp:395`, `:409`, `:428`, `:438`. **Cross-check: confirmed.** |
| *TTNN Readiness* — Pybind `create_descriptor` | **No** — `paged_cache_nanobind.cpp` has no `nb::class_` and no `create_descriptor` binding; only `ttnn::bind_function` wrappers. **Cross-check: confirmed.** |
| *TTNN Readiness* — Op-owned tensors | **No** — no `WorkloadDescriptor` and no `buffers` vector anywhere in `device/` (grep). Sheet cell blank — no cross-column invariant violated. |
| *TTNN Readiness* — Target concept | **`CustomProgramSpecFactoryConcept`** (all 8 factories). Matches the sheet's own `Porting Target` column. |
| *Port work* — Offset base pointer | **none** — GREEN. All 12 `->address()` sites in the op are bare bases with no host arithmetic; every scalar offset (`cache_start_id`, `tile_update_offset_B`, `start_tile_id`, `start_row_num`) already travels as a *separate* arg. |
| *Port work* — Tensor bindings (per binding) | 11 distinct bindings across the 3 DOps: 9 **Case 1** (`TensorAccessor`), 2 **clean** (borrowed-memory DFB), 2 of which are **config-split** (Case 1 on the DRAM-interleaved path, clean on the L1-sharded path). No Case 2 anywhere — the op never uses a tensor base pointer for hand-rolled NoC arithmetic. |
| *TTNN Readiness* — TensorParameter relaxation | Sheet, verbatim: **`(legality - pending analysis)`** on all 8 rows → this is the column that explains the `no`. **User-supplied resolution: analysis complete, go strict — no relaxation.** Under that resolution the conjunct is `none` and clears. |
| *Port work* — TensorAccessor 3rd arg | **N/A — no accessor in the op passes a 3rd argument.** All 16 `TensorAccessor(...)` constructions across the 11 kernels are 2-arg. The subject never fires. |
| *Port work* — CB endpoints | **All legal or self-loop.** 29 `(CB, config)` instances across the 4 factory bodies: 26 plain **1:1**, 3 **self-loop** (one-toucher, in `fill_cache`). No multi-binding, no dead CB, no conditional-DFB drop. |

**CB endpoints** are dispositions, not gates: every out-of-window CB here resolves as a **self-loop** (one toucher). Recorded per `(CB, config)` below.

---

## Result

**RED** — the port cannot start. `RED at op level; no portable subset.`

**Primary blocker: the Device 2.0 prerequisite.** Six kernel lines call the CB-index-keyed free function `get_dataformat(cb_id)` where the Device 2.0 `CircularBuffer` wrapper is available in the same function and `CircularBuffer::get_dataformat()` exists as its replacement (`tt_metal/hw/inc/api/dataflow/circular_buffer.h`, alongside `get_tile_size()` / `get_tile_hw()` under the same `DATA_FORMATS_DEFINED` guard). `get_dataformat` is **not** on the audit's sanctioned free-function list — that list is `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)`, and the recipe states the list is the whole test. Routed to the **Device 2.0 migration team**.

This is the *cheapest* class of Device 2.0 gate the recipe describes — *isolated holdovers*, not broad Device 1.0. It is cheaper still than the recipe's canonical example: in all six cases the returned `DataFormat` is stored in a `const` local (`cache_data_format` / `data_format`) that **nothing in the kernel ever reads**, so the correct Device 2.0 change is to delete the six lines outright rather than swap them onto the wrapper. Everything else in these kernels is already structurally Device 2.0: `Noc`, `CircularBuffer`, `Semaphore<>`, `CoreLocalMem<>`, `TensorAccessor`, `UnicastEndpoint` — zero `noc_async_read`/`noc_async_write`, zero `cb_reserve_back`/`cb_push_back`/`cb_wait_front`/`cb_pop_front`, zero `InterleavedAddrGen`/`ShardedAddrGen`/`InterleavedPow2AddrGen*`, zero raw semaphore addresses, zero `get_read_ptr(cb_id)`/`get_write_ptr(cb_id)` free calls (all six kernels already use the wrapper methods).

**No portable subset.** The six sites are spread so that **every** factory family touches at least one affected kernel — `fill_cache` (its writer), `update_cache` (both its dataflow kernels), tiled-fused (its reader), row-major-fused (its reader and its writer). There is no factory whose complete kernel set is clean, so `Code-path scope` yields nothing to carve out.

**Secondary blocker, cleared out-of-band: the readiness sheet's `Is able to port? == no`.** All 8 rows read `no`, and the only blocking column that explains it is `TensorParameter relaxation == (legality - pending analysis)`. The user launching this audit reports that the legality analysis has since been completed and the decision is to proceed with **strict (no relaxation) `TensorParameter`s** — which resolves that conjunct to the clearing value. This audit therefore treats the TTNN factory-concept gate as **cleared**, and routes a *sheet-refresh request* (not a blocker) to the readiness-sheet owner. My own reading of the three custom hashes is consistent with the strict decision and is recorded under *Team-only → Relaxation candidates* (fallible, non-authoritative).

**Path forward.** One Device 2.0 change (6 line deletions, no behavioural effect), then a cheap re-audit. Everything else in this audit is GREEN or clean: features, offsets, 3rd-arg, CB endpoints, cross-op coupling, RTA varargs. If the sanctioned-free-function list is extended to cover `get_dataformat` (see *Questions* #1 and *Recipe notes* #1 — this single name is what the whole gate turns on), the op goes GREEN with no code change at all.

---

## Gate detail

### TTNN factory concept (`Is able to port?`) — **cleared out-of-band**

Sheet values, identical on all 8 rows (quoted verbatim):

| Column | Value |
|---|---|
| `Concept` | `descriptor` |
| `Op Classification` | `PD Op (custom)` |
| `Execution Model` | `SPMD` |
| `Porting Target` | `CustomProgramSpecFactoryConcept` |
| `Custom hash (compute_program_hash)` | `yes` |
| `Backdoor custom hash (attribute_values / to_hash)` | `no` |
| `Runtime-args update (get_dynamic_runtime_args)` | `no` |
| `Override runtime args method? (PD only)` | `yes` |
| `Pybind descriptor (nb::class_ of device op)` | `no` |
| `Smuggled pointer (raw buffer addr in RTA/CRTA)` | `no` |
| `Known op issues` | *(empty)* |
| `Diego validation` | `no` |
| **`Is able to port?`** | **`no`** |
| **`TensorParameter relaxation`** | **`(legality - pending analysis)`** |
| `Op-owned tensors?` | *(empty)* |
| `Secretly SPMD Workload?` | *(empty)* |
| `Model` | `other` (update_cache, fill_cache rows) / `llama` (fused rows) |
| `ProgramFactory used in llama?` | `yes` (update_cache, fill_cache) / `no` (fused) |
| `Uses llama kernels? (primary or shared)` | `yes` (update_cache, fill_cache) / `no` (fused) |

**Attribution of the `no`.** Exactly one blocking column fires: `TensorParameter relaxation != none`. `get_dynamic_runtime_args` is `no`; `Smuggled pointer` is `no`; `Known op issues` is empty; `Op Classification` does not read as broken; `Concept` is `descriptor`, not `legacy device-op` or `WorkloadDescriptor`. So the `no` is **fully attributed**, not unattributed.

**Disposition.** The user reports the pending legality analysis is complete with the outcome *strict `TensorParameter`s, no relaxation*. That resolution turns the blocking conjunct into the clearing value, so this gate is recorded as **cleared**. Two consequences, both non-blocking:
- **To the readiness-sheet owner (Diego), as a refresh request:** `TensorParameter relaxation` and `Is able to port?` are stale for `experimental/paged_cache` on all 8 rows; please update to `none` / `yes` so the sheet and the ops-team decision agree. This is *not* a "spreadsheet is broken" finding — none of the four triggers fired (no primary-column conflict, no violated cross-column invariant, no missing row, no phantom/missing factory row). It is a value that reality has moved past.
- **Nothing routes to the ops team** for a relaxation; there is no relaxation to design.

**Lightweight cross-check — every checkable column agrees with the code.**

| Column | Sheet | Code evidence | Verdict |
|---|---|---|---|
| `Concept` | `descriptor` | All 8 factories declare `static tt::tt_metal::ProgramDescriptor create_descriptor(...)`: `paged_update_cache_program_factory.hpp:17,35`; `paged_fill_cache_program_factory.hpp:17,36`; `paged_tiled_fused_update_cache_program_factory.hpp:19,50`; `paged_row_major_fused_update_cache_program_factory.hpp:19,50`. No `create_workload_descriptor`, no mesh-workload return type (grep over `device/`). | ✓ agrees |
| `Custom hash` | `yes` | `compute_program_hash` overridden on all 3 DOps (`paged_update_cache_device_operation.cpp:313`, `paged_fill_cache_device_operation.cpp:207`, `paged_fused_update_cache_device_operation.cpp:371`). Not a pybound-rename case (no pybind of internals). | ✓ agrees |
| `Runtime-args update (get_dynamic_runtime_args)` | `no` | No `get_dynamic_runtime_args` token anywhere under `device/`. | ✓ agrees |
| `Override runtime args method?` | `yes` | All 8 factories declare and define `override_runtime_arguments`. On a `descriptor` op this is the target-concept signal (not the legacy-concept signature), since `create_descriptor` is present alongside it — so it selects `CustomProgramSpecFactoryConcept`, it does not gate. | ✓ agrees |
| `Pybind descriptor` | `no` | `paged_cache_nanobind.cpp`: no `nb::class_`, no `create_descriptor` binding — only `ttnn::bind_function<"paged_update_cache">` / `<"paged_fill_cache">` / `<"paged_fused_update_cache">`. Nothing for the port to delete here. | ✓ agrees |
| `Smuggled pointer` | `no` | `create_descriptor` never emplaces a raw `->address()`. Every buffer arg is emplaced as an **annotated `Buffer*`** (`KernelDescriptor::RTArgList::push_back(Buffer*)` / `emplace_runtime_args`), which the framework auto-registers as a `BufferBinding`. Raw addresses appear only inside `override_runtime_arguments`, which is the sanctioned patch path. | ✓ agrees |
| `Secretly SPMD Workload?` | *(blank)* | Only meaningful when `Concept == WorkloadDescriptor`; it isn't. | ✓ N/A |
| **Factory-set match** | 8 rows | Code has exactly 8 factory structs, names matching the sheet's `Factory (variant)` cells one-for-one (`grep -rn 'struct Paged.*Factory' device/`). No phantom row, no missing row. | ✓ agrees |

**Cross-column invariants.** `Runtime-args update == no` on a `descriptor` concept — legal. `Op-owned tensors?` is blank (not `yes`) on a `descriptor` concept — legal. No invariant violated.

### Device 2.0 (every kernel used) — **RED**

**Class: isolated holdovers** (idioms structurally intact; each fix is one line). Route to the **Device 2.0 migration team**. All six kernels are owned by this op — no borrowed/donor kernel contributes a violation, so there is no external family dependency to schedule.

| File (under `device/kernels/`) | Line | Call | Wrapper in scope | Note |
|---|---|---|---|---|
| `dataflow/reader_update_cache_interleaved_start_id.cpp` | 64 | `get_dataformat(cache_cb_id)` | `CircularBuffer cb_cache(cache_cb_id)` @ 54 | result stored in `cache_data_format`, **never read** |
| `dataflow/writer_update_cache_interleaved_start_id.cpp` | 54 | `get_dataformat(cache_cb_id)` | `CircularBuffer cb_cache(cache_cb_id)` @ **60** (declared *after* the call, same function) | result stored in `cache_data_format`, **never read** |
| `dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp` | 77 | `get_dataformat(cache_cb_id)` | `CircularBuffer cb_cache(cache_cb_id)` @ 68 | result stored in `cache_data_format`, **never read** |
| `dataflow/reader_paged_row_major_fused_update_cache_interleaved_start_id.cpp` | 77 | `get_dataformat(cache_cb_id)` | `CircularBuffer cb_cache(cache_cb_id)` @ 68 | result stored in `cache_data_format`, **never read** |
| `dataflow/writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp` | 66 | `get_dataformat(cache_cb_id)` | `CircularBuffer cb_cache(cache_cb_id)` @ **72** (declared *after* the call, same function) | result stored in `cache_data_format`, **never read** |
| `dataflow/writer_fill_cache_interleaved.cpp` | 144 | `get_dataformat(cb_id_in)` | `CircularBuffer cb_in(cb_id_in)` @ 90 | result stored in `data_format`, **never read** |

Two of the six (marked above) call the free function *before* the wrapper's declaration line, so a mechanical swap would also need a one-line reorder — which is moot, since the recommended fix for all six is deletion of the dead assignment.

**Why this is a violation and not a sanctioned free function.** The recipe's sanctioned list is exactly `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)`, and it states plainly that the list is the whole test and does not turn on what object is in scope. `get_dataformat` is absent from it. Checking the Device 2.0 surface itself (the test the recipe *does* name for unseating a name): `tt_metal/hw/inc/api/dataflow/circular_buffer.h` exposes `get_dataformat()` as a wrapper method (adjacent to `get_tile_size()` and `get_tile_hw()`, all three under `#ifdef DATA_FORMATS_DEFINED`), i.e. the surface provides the replacement rather than keeping the free function — which points the same way. And the Device 2.0 migration guide's own migrated example (`device_api_migration_guide.md`, *Complete Migration Examples → Example 1*) keeps `get_tile_size(cb_id)` as a free function but never mentions `get_dataformat`. So no evidence sanctions it. See *Questions* #1 — this one name is the entire gate.

**Kernels verified clean of any other Device 2.0 idiom** (all 11): the exhaustive grep for `noc_async_read|noc_async_write|noc_async_*_barrier|get_noc_addr|get_noc_addr_from_bank_id|get_semaphore|noc_semaphore_*|InterleavedAddrGen*|ShardedAddrGen|InterleavedPow2AddrGen*|cb_reserve_back|cb_push_back|cb_wait_front|cb_pop_front|get_read_ptr(|get_write_ptr(|evil_set_*_ptr` over `device/kernels/` returns **zero** legacy hits; every `get_read_ptr`/`get_write_ptr` occurrence is a `cb_obj.` wrapper-method call.

*Not counted as violations (judgement recorded for transparency):*
- **`reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_*_ptr())`** — 16 sites across 6 kernels (e.g. `reader_update_cache_interleaved_start_id.cpp:83,108`; `writer_fill_cache_interleaved.cpp:113,128,151`). The migration guide's *Memory Access* section does offer `CoreLocalMem<T>` as the modern form, and these kernels already use `CoreLocalMem<uint32_t>` for their NoC endpoints — so this is a *style* residue. It is not a CB-index-keyed free function, and more to the point the gate exists because Metal 2.0's binding tokens must have a Device 2.0 wrapper object to attach to: here the CB *is* already a `CircularBuffer`, the cast is applied to the pointer it returns, and `dfb::name` binds regardless. Not a blocker; noted for the Device 2.0 team as optional cleanup they may fold in while touching these files.
- **`get_tile_size(cb_id)`** — 8 sites. Explicitly sanctioned; not flagged.
- **`my_x[noc_id]` / `my_y[noc_id]`** alongside `noc.get_noc_id()` (3 writer kernels) — the standard way to obtain own coordinates; no wrapper replacement exists and it is not a CB-index free function.

### Feature compatibility — GREEN (no entry fires)

Every Appendix A entry is UNSUPPORTED, so an absent feature is `N/A`, not a pass.

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | **N/A** | No `GlobalCircularBuffer` type, `using` alias, or `experimental::CreateGlobalCircularBuffer` call. No `#include <tt-metalium/global_circular_buffer.hpp>` (or the `experimental/` spelling) — the full host include set is `buffer.hpp`, `circular_buffer.hpp`, `constants.hpp`, `core_coord.hpp`, `host_api.hpp`, `program.hpp`, `program_descriptors.hpp`, `tensor_accessor_args.hpp`, `work_split.hpp`. **Descriptor-API attachment checked by field name:** no `CBDescriptor` literal in the four factory bodies sets `.global_circular_buffer` (the 26 `CBDescriptor` literals set only `.total_size`, `.core_ranges`, `.format_descriptors`, and — for the 6 borrowed-memory ones — `.buffer`). No `.remote_index(`, no `remote_cb_*` identifier, no `remote_circular_buffer.h`. **Imperative attachment checked:** the two `UpdateDynamicCircularBufferAddress` sites (`paged_fused_update_cache_device_operation.cpp:70`, `paged_update_cache_program_factory.cpp:518`) both take a `Buffer&` (`*buffer` / `*tensor_args.input_tensor.buffer()`), i.e. the unrelated 3-arg `Buffer&` overload — explicitly the false-positive guard, not the `const GlobalCircularBuffer&` overload. No factory signature takes `std::optional<const GlobalCircularBuffer>&`. |
| CBDescriptor `address_offset` (non-zero) | **N/A** | The token `address_offset` does not appear anywhere in the op (grep, case-insensitive, host + kernels). No `.address_offset` on any `CBDescriptor`, no `CircularBufferConfig::set_address_offset`, no 4-argument `UpdateDynamicCircularBufferAddress(program, handle, buffer, offset)` — both sites are the 3-arg form, so the offset defaults to zero. No `cb_descriptor_from_sharded_tensor` call. The borrowed-memory CBs bind at base (`.buffer = <ptr>` with no offset field), which is the ordinary, supported pattern. Nothing to escalate to the runtime team. |
| GlobalSemaphore | **N/A** | No `GlobalSemaphore` type or `using` alias, no `experimental::CreateGlobalSemaphore`, no `#include <tt-metalium/global_semaphore.hpp>`, no factory signature taking `const GlobalSemaphore&` / `std::optional<GlobalSemaphore>`. The op's only semaphores are three plain `SemaphoreDescriptor{...}` literals (`paged_update_cache_program_factory.cpp:247`, `paged_tiled_fused_update_cache_program_factory.cpp:260`, `paged_row_major_fused_update_cache_program_factory.cpp:256`) — the regular path, supported as `SemaphoreSpec`. Kernel side uses `Semaphore<>` (`noc_semaphore.h`), which is the guarded non-Global form. |

### CB endpoints (GATE-free) — all legal or self-loop

Run in full: the Device 2.0 gate is RED **only** on isolated holdovers, so the Device-2.0 idioms this scan keys on (`CircularBuffer` wrapper methods, `Semaphore` objects) are intact — the recipe's precondition for running rather than deferring is met.

Census method: per CB, per node, counting every toucher — FIFO produce (`reserve_back`/`push_back`), FIFO consume (`wait_front`/`pop_front`), and raw pointer access (`get_write_ptr`/`get_read_ptr`). A kernel that raw-peeks a CB it already FIFO-owns is counted **once** (the peek rides its own binding). The `compute_kernel_lib::untilize<Wt, in_dfb, out_dfb>` / `tilize<Wt, in_dfb, out_dfb>` helpers FIFO-consume their input DFB and FIFO-produce their output DFB, so the compute kernel is a locked consumer of the first template CB and a locked producer of the second.

**Hidden-second-writer hunt (face (a)): performed and came back empty.** No kernel in the op writes another kernel's CB through `get_write_ptr()`/`fifo_wr_ptr` + offset without owning the FIFO. The only semaphore in the op (`in0_sequential_mode`, one per factory) coordinates *share_cache* ordering between a writer and the *next core's* reader — `Semaphore<>::up(noc, send_core_x, send_core_y, 1)` at `writer_update_cache_interleaved_start_id.cpp:165` / `writer_paged_fused_...:176` / `writer_paged_row_major_...:179`, awaited at `reader_*:127`/`:153`/`:155` — not a raw CB co-fill handshake. **Multiple-readers hunt (face (b)): empty** — no CB has base-pointer read sites in 2+ co-resident kernels. **Dual-instance work-split hunt (face (c)): empty** — no factory pushes the same `kernel_source` into two `KernelDescriptor`s; each factory emits exactly one reader, one writer, and (except `fill_cache`) one compute descriptor, each over one core range. The fused factories *look* adjacent to this shape because one reader instance serves both input1 and input2 cores, but that is one instance over disjoint node sets with per-core args — ordinary 1:1 per node, not two touchers.

**`PagedUpdateCacheProgramFactory` / `PagedUpdateCacheMeshWorkloadFactory`** (identical bodies; `all_cores` = the input shard grid, one user per core):

| CB | Index | Config | Touchers on a node | Verdict |
|---|---|---|---|---|
| `src0_cb_index` (cache tiles in) | `c_0` | all | reader locked-P (`reader:133,144`) · compute locked-C (`untilize<Wt, cache_cb, untilized_cache_cb>`, `compute/update_cache.cpp:48`) | **1:1** |
| `src1_cb_index` (input shard, **borrowed** `.buffer = in1_buffer`) | `c_1` | all | reader locked-P (`reader:60-61`) · compute locked-C (`untilize<Wt, in_cb, untilized_in_cb>`, `compute:39-45`) | **1:1** |
| `intermed0_cb_index` | `c_24` | all | compute locked-P · writer locked-C (`writer:123,135`) | **1:1** |
| `intermed1_cb_index` | `c_25` | all | writer locked-P (`writer:124,134`) · compute locked-C (`tilize<Wt, untilized_cache2_cb, out_cb>`, `compute:51`) | **1:1** |
| `intermed2_cb_index` | `c_26` | all | compute locked-P (`compute:39-45` output) · writer locked-C (`writer:114,161`) | **1:1** |
| `output_cb_index` | `c_16` | all | compute locked-P (`compute:51` output) · writer locked-C (`writer:138,149`) | **1:1** |
| `cb_index_id` | `c_2` | `use_index_tensor` only (CB not allocated otherwise) | reader locked-P (`reader:77,82`; its `get_write_ptr` peek @78 rides that binding) · writer locked-C (`writer:73,111`) | **1:1** |
| `cb_pagetable_id` | `c_3` | `is_paged_cache` only | reader locked-P (`reader:98,106`; peek @98) · writer locked-C (`writer:88,101`) | **1:1** |

**`PagedFillCacheProgramFactory` / `PagedFillCacheMeshWorkloadFactory`** (no compute kernel):

| CB | Index | Config | Touchers on a node | Verdict |
|---|---|---|---|---|
| `src0_cb_index` (input tiles) | `c_0` | all | reader locked-P (`reader_fill:38,46`) · writer locked-C (`writer_fill:197-198, 232-236, 245`) | **1:1** |
| `page_table_cb_index` | `c_1` | all | **writer only** — `reserve_back(1)` @149 + raw write via `get_write_ptr()` @150 + repeated `noc.async_read` into that pointer @211; never pushed, never popped, no other kernel references the index | **self-loop** (one toucher) |
| `cb_batch_idx_id` | `c_2` | `use_batch_idx_tensor` only | **writer only** — `reserve_back(1)` @102 + raw @103-113 | **self-loop** |
| `cb_valid_seq_len_id` | `c_3` | `use_valid_seq_len` only | **writer only** — `reserve_back(1)` @123 + raw @124-128 | **self-loop** |

**`PagedTiledFusedUpdateCacheProgramFactory` / `...MeshWorkloadFactory`** (kernels over `all_cores_bb`; `c_1` allocated only over `input1_cores`, `c_2` only over `input2_cores`, the two disjoint):

| CB | Index | Core range | Config | Touchers on a node | Verdict |
|---|---|---|---|---|---|
| `cache_cb_index` | `c_0` | `all_cores_bb` | all | reader locked-P (`reader:159,170`) · compute locked-C (`paged_fused_update_cache.cpp:59`) | **1:1** |
| `src1_cb_index` (**borrowed** `in1_buffer`) | `c_1` | `input1_cores` | all | reader locked-P (`reader:73-74`) · compute locked-C (`compute:48-54`) | **1:1** |
| `src2_cb_index` (**borrowed** `in2_buffer`) | `c_2` | `input2_cores` | all | reader locked-P (`reader:73-74`) · compute locked-C (`compute:40-46`) | **1:1** |
| `intermed0_cb_index` | `c_24` | `all_cores_bb` | all | compute locked-P · writer locked-C (`writer:134,146`) | **1:1** |
| `intermed1_cb_index` | `c_25` | `all_cores_bb` | all | writer locked-P (`writer:135,145`) · compute locked-C (`compute:62`) | **1:1** |
| `intermed2_cb_index` | `c_26` | `all_cores_bb` | all | compute locked-P (`compute:40-54` output) · writer locked-C (`writer:125,172`) | **1:1** |
| `output_cb_index` | `c_16` | `all_cores_bb` | all | compute locked-P (`compute:62` output) · writer locked-C (`writer:149,160`) | **1:1** |
| `cb_index_id` (**borrowed** when the index tensor is L1-sharded) | `c_3` | `all_cores_bb` | `use_index_tensor` only | reader locked-P (`reader:89,96`) · writer locked-C (`writer:77,122`) | **1:1** |
| `cb_pagetable_id` (**borrowed** when the page table is L1-sharded) | `c_4` | `all_cores_bb` | `is_paged_cache` only | reader locked-P (`reader:106,122`) · writer locked-C (`writer:88,112`) | **1:1** |

**`PagedRowMajorFusedUpdateCacheProgramFactory` / `...MeshWorkloadFactory`** (different CB indices; the RM writer consumes the row-major input shard directly instead of an untilized intermediate):

| CB | Index | Core range | Config | Touchers on a node | Verdict |
|---|---|---|---|---|---|
| `cache_cb_index` | `c_0` | `all_cores_bb` | all | reader locked-P (`reader:161,172`) · compute locked-C (`paged_row_major_fused_update_cache.cpp:39`) | **1:1** |
| `src1_cb_index` (**borrowed** `in1_buffer`) | `c_1` | `input1_cores` | all | reader locked-P (`reader:73-74`) · **writer** locked-C (`writer:128,175`) — RM compute does *not* touch it | **1:1** |
| `src2_cb_index` (**borrowed** `in2_buffer`) | `c_2` | `input2_cores` | all | reader locked-P · writer locked-C | **1:1** |
| `intermed0_cb_index` | `c_5` | `all_cores_bb` | all | compute locked-P (`compute:39` output) · writer locked-C (`writer:137,149`) | **1:1** |
| `intermed1_cb_index` | `c_6` | `all_cores_bb` | all | writer locked-P (`writer:138,148`) · compute locked-C (`compute:42`) | **1:1** |
| `output_cb_index` | `c_7` | `all_cores_bb` | all | compute locked-P (`compute:42` output) · writer locked-C (`writer:152,163`) | **1:1** |
| `cb_index_id` | `c_3` | `all_cores_bb` | `use_index_tensor` only | reader locked-P (`reader:90,98`) · writer locked-C (`writer:85` — `wait_front` with **no matching `pop_front`**; see *Misc anomalies*) | **1:1** |
| `cb_pagetable_id` | `c_4` | `all_cores_bb` | `is_paged_cache` only | reader locked-P (`reader:110,125`) · writer locked-C (`writer:96` — **no matching `pop_front`**) | **1:1** |

**No dead CB anywhere.** Every allocated `buffer_index` is referenced by at least one bound kernel in the config that allocates it, confirmed across all instantiations (index-tensor on/off, paged on/off, page-table DRAM vs L1-sharded, batch-idx tensor on/off, valid-seq-len on/off, share_cache on/off, tiled vs row-major). The two configuration-optional CBs in each factory (`cb_index`, `cb_pagetable`, plus `fill_cache`'s `cb_batch_idx` / `cb_valid_seq_len`) are **already conditionally allocated on the host side** — so there is no "dead in some configs, live in others" case needing a new conditional DFB spec; the existing `if (use_index_tensor)` / `if (is_paged_cache)` / `if (use_batch_idx_tensor)` / `if (use_valid_seq_len)` guards translate directly.

**One node-level nuance, non-blocking:** in the fused factories the kernels span `all_cores_bb` (the bounding box of `input1_cores ∪ input2_cores`), and any node in `unused_cores = all_cores_bb − all_cores` receives a single runtime arg `{!has_work}` and early-returns, so it touches nothing. That is *runtime* control flow, not a config difference — the `buffer_index` is still referenced by the kernels bound over that range — so those CBs are **not** dead there and must not be dropped. The porter-facing consequence (a per-core runtime-arg count that varies within one kernel) is recorded under *Heads-ups*.

### Offset base pointers — GREEN

Not listed in the offset-base-pointer triage analysis (a dated prior, not an authority) — and, per the recipe, "not in the tables" is not allowed to stand in for "scanned and clean", so the recognition scan was run on **every** address argument in the op.

**Every address that reaches a kernel is a clean base.** All 12 `->address()` expressions in the op, exhaustively:

| Site | Expression | Fold? |
|---|---|---|
| `update_cache/paged_update_cache_program_factory.cpp:488` | `tensor_args.cache_tensor.buffer()->address()` | none |
| `update_cache/paged_update_cache_program_factory.cpp:491` | `...update_idxs_tensor.value().buffer()->address()` | none |
| `update_cache/paged_update_cache_program_factory.cpp:493` | `...page_table.value().buffer()->address()` | none |
| `fill_cache/paged_fill_cache_program_factory.cpp:384` | `tensor_args.input_tensor.buffer()->address()` | none |
| `fill_cache/paged_fill_cache_program_factory.cpp:385` | `tensor_args.cache_tensor.buffer()->address()` | none |
| `fill_cache/paged_fill_cache_program_factory.cpp:386` | `tensor_args.page_table.buffer()->address()` | none |
| `fill_cache/paged_fill_cache_program_factory.cpp:390` | `...valid_seq_len_tensor_opt->buffer()->address()` | none |
| `fill_cache/paged_fill_cache_program_factory.cpp:396` | `...batch_idx_tensor_opt->buffer()->address()` | none |
| `fused_update_cache/paged_fused_update_cache_device_operation.cpp:83` | `tensor_args.cache_tensor1.buffer()->address()` | none |
| `fused_update_cache/paged_fused_update_cache_device_operation.cpp:84` | `tensor_args.cache_tensor2.buffer()->address()` | none |
| `fused_update_cache/paged_fused_update_cache_device_operation.cpp:87` | `update_idxs_tensor->buffer()->address()` | none |
| `fused_update_cache/paged_fused_update_cache_device_operation.cpp:88` | `page_table->buffer()->address()` | none |

Every `create_descriptor` address argument is an **annotated `Buffer*`**, which makes a fold structurally impossible at the descriptor site (there is no expression to add to).

**The offsets that exist all already travel separately** — which is the shape the ops team's Type-1 fix *produces*, present here by original design:
- `cache_start_id` (a **tile index**, not an address) — computed host-side as `cache_batch_tile_offset + (update_idx / TILE_HEIGHT) * Wt` (`paged_update_cache_program_factory.cpp:80`; tiled `:72`; RM `:74`) and consumed kernel-side purely as a `TensorAccessor` `{.page_id = ...}` (`reader_update_cache:138`, `writer_update_cache:143`, and the fused equivalents). A page index, never added to a base address.
- `tile_update_offset_B` / `cache_tile_offset_B` — a **byte offset into an L1 CB**, added kernel-side to `cb_untilized_cache.get_read_ptr()` (`writer_update_cache:126`; fused `:137`; RM `:140`). Not a tensor base.
- `start_tile_id` = `num_blocks_written * Wt` and `start_row_num` (`paged_fill_cache_program_factory.cpp:303,313`) — work-split scalars, consumed as loop bounds and `{.page_id}` values.
- Kernel-side pointer arithmetic on borrowed CBs — `page_table_cb_wr_ptr += my_batch_idx * page_table_stick_size` (`reader_paged_fused:119`, `reader_paged_row_major:123`) and the read-pointer mirrors (`writer_paged_fused:91`, `writer_paged_row_major:99`). These offset an **L1 CB pointer inside the kernel**, not a host-folded device base, so they are outside this subject entirely (and are preserved verbatim by the borrowed-DFB translation).

**Type 3** (`address_offset`) — absent; see the Appendix A row. **Type 4** (`ttnn::narrow` / interior-base `MeshBuffer::create`) — absent (grep). Reconciliation outcome for every RTA: *no fold, op not in the tables* → clean, handed to *TensorParameter analysis* below.

### TensorAccessor 3rd argument — N/A

**No accessor in this op passes a 3rd argument, so the subject never fires.** This is a *no sites* finding, not "sites found and classified redundant". All 16 `TensorAccessor(...)` constructions across the 11 kernels are the 2-argument form `TensorAccessor(args, addr)`:

`reader_fill_cache_interleaved.cpp:31` · `writer_fill_cache_interleaved.cpp:101,122,146,147` · `reader_update_cache_interleaved_start_id.cpp:70,75,96` · `writer_update_cache_interleaved_start_id.cpp:58` · `reader_paged_fused_...:83,88,110` · `writer_paged_fused_...:62` · `reader_paged_row_major_...:83,88,114` · `writer_paged_row_major_...:70`.

The op is not in the 3rd-arg triage analysis (a dated prior), and the syntactic scan for a newly-added site — which a dated table could not have caught — likewise comes back empty. Nothing to drop at port time, no `dynamic_tensor_shape` to set, no Class 3/4/Special to escalate.

---

## Port-work summary  *(would mirror the brief; no brief is issued on this RED)*

- **Tensor bindings** (per binding, per DeviceOperation):

  **`PagedUpdateCacheDeviceOperation`** — 4 bindings:
  | Binding | Delivery in legacy | Kernel use | Case |
  |---|---|---|---|
  | `cache_tensor` | `Buffer*` → reader RTA[0] & writer RTA[0] (`factory:406,426`); raw address re-applied by `override_runtime_arguments` (`:504,509`) | `TensorAccessor(s0_args, cache_addr)` — `reader:70`, `writer:58`; all access via `{.page_id}` | **Case 1** |
  | `input_tensor` | borrowed-memory CB `c_1` (`.buffer = in1_buffer`, `factory:208`); re-pointed on cache hit by `UpdateDynamicCircularBufferAddress` (`:518`) | reader `reserve_back`/`push_back` then compute untilizes from it; no `TensorAccessor`, no address RTA | **clean** (causal-link gate: borrowed-memory DFB read → `DataflowBufferSpec::borrowed_from`) |
  | `update_idxs_tensor` (optional) | `Buffer*` → reader RTA[2] (`factory:409`) | `TensorAccessor(index_tensor_args, index_tensor_addr)` — `reader:75` | **Case 1** |
  | `page_table` (optional) | `Buffer*` → reader RTA[4] (`factory:415`) | `TensorAccessor(page_table_args, page_table_tensor_addr)` — `reader:96` | **Case 1** |

  **`PagedFillCacheDeviceOperation`** — 5 bindings, all Case 1:
  | Binding | Delivery in legacy | Kernel use | Case |
  |---|---|---|---|
  | `cache_tensor` | `Buffer*` → writer RTA[0] (`factory:311`) | `TensorAccessor(s0_args, dst_addr)` — `writer_fill:146` | **Case 1** |
  | `input_tensor` | `Buffer*` → reader RTA[0] (`factory:302`) | `TensorAccessor(src_args, src_addr)` — `reader_fill:31` | **Case 1** |
  | `page_table` | `Buffer*` → writer RTA[1] (`factory:312`) | `TensorAccessor(page_table_args, page_table_addr)` — `writer_fill:147` | **Case 1** |
  | `batch_idx_tensor` (optional) | `Buffer*` → writer RTA[4] **or** the plain scalar `batch_idx_fallback` in the same slot (`factory:315-319`) | `TensorAccessor(batch_idx_tensor_args, batch_arg)` — `writer_fill:101` | **Case 1** + overloaded-slot split (see *Heads-ups*) |
  | `valid_seq_len_tensor` (optional) | `Buffer*` → writer RTA[6] **or** literal `0` (`factory:323-327`) | `TensorAccessor(valid_seq_len_tensor_args, valid_seq_len_addr)` — `writer_fill:122` | **Case 1** + overloaded-slot split |

  **`PagedFusedUpdateCacheDeviceOperation`** — 6 bindings (identical in both the tiled and row-major factories):
  | Binding | Delivery in legacy | Kernel use | Case |
  |---|---|---|---|
  | `cache_tensor1` | `Buffer*` → reader RTA[2] & writer RTA[1], **on `cores1` only** (tiled `:438,460`; RM `:435,457`) | `TensorAccessor(s0_args, cache_addr)` — `reader:83`, `writer:62`/`:70` | **Case 1** |
  | `cache_tensor2` | `Buffer*` → the **same** reader RTA[2] / writer RTA[1] slots, on `cores2` (tiled `:483,505`; RM `:481,503`) | same accessor | **Case 1** + per-core binding split (see *Heads-ups*) |
  | `input_tensor1` | borrowed CB `c_1` over `input1_cores` (tiled `:211`; RM `:216`) | reader FIFO-produces; compute (tiled) or writer (RM) consumes | **clean** (borrowed DFB) |
  | `input_tensor2` | borrowed CB `c_2` over `input2_cores` (tiled `:221`; RM `:226`) | same | **clean** (borrowed DFB) |
  | `update_idxs_tensor` (optional) | `Buffer*` → reader RTA[4]; **and** borrowed CB `c_3` when the tensor is L1-sharded (`.buffer = index_buffer_ptr`, tiled `:276`; RM `:272`) | DRAM-interleaved path: `TensorAccessor(index_tensor_args, index_tensor_addr)` + `noc.async_read` (`reader:93`). L1-sharded path: the read is compiled out (`if constexpr (index_is_dram)`) and the value is read straight out of the borrowed CB (`reader:97`). | **config-split** — **Case 1** on DRAM-interleaved · **clean** on L1-sharded |
  | `page_table` (optional) | `Buffer*` → reader RTA[6]; **and** borrowed CB `c_4` when L1-sharded (tiled `:289`; RM `:285`) | DRAM: `TensorAccessor(page_table_args, ...)` + `noc.async_read` (`reader:110-117`). L1-sharded: pointer walk inside the borrowed CB (`reader:119`, `writer:91`/`:99`). | **config-split** — **Case 1** on DRAM · **clean** on L1-sharded |

  **No Case 2 anywhere in the op.** Every tensor base that reaches a kernel is fed to a `TensorAccessor`; nothing does hand-rolled NoC arithmetic on a tensor base, so no `get_bank_base_address` bridge is needed.

  **Urgency note.** None of these 11 bindings is the *silent-wrong* fast-path-cache hazard. `create_descriptor` never emplaces a raw `->address()`; every buffer arrives as an annotated `Buffer*` (auto-registered `BufferBinding`, patched on cache hits), and on top of that all 8 factories define `override_runtime_arguments`, which supersedes `resolve_bindings` and re-applies every address explicitly. The Metal 2.0 typed binding supersedes both mechanisms; this is **routine port work**, not a correctness fix.
- **TensorParameter relaxation:** **none** — per the user-supplied resolution of the sheet's `(legality - pending analysis)`. Strict `TensorSpec` match. My independent reading of the three custom hashes is consistent with that (see *Team-only*), but the ops-team decision is the authority.
- **TensorAccessor 3rd arg:** **none** — no site exists.
- **CB endpoints:** self-loop `(page_table_cb_index c_1, fill_cache — all configs)`, `(cb_batch_idx_id c_2, fill_cache — use_batch_idx_tensor)`, `(cb_valid_seq_len_id c_3, fill_cache — use_valid_seq_len)`. All other 26 `(CB, config)` instances are plain 1:1. No multi-binding flag, no dead-CB drop, no new conditional DFB.

---

## Heads-ups  *(would mirror the brief; recorded here for the re-audit and the team)*

- **CB endpoints (multi-binding shapes to watch):** **none.** All three faces were hunted (details in *Gate detail → CB endpoints*) and came back empty.
- **Cross-op / shared kernels:** **none — and this is worth stating positively, because a grep by basename says otherwise.** All 11 kernels this op instantiates live under `device/kernels/` in this directory, and `grep -rl 'operations/experimental/paged_cache/device/kernels'` shows the only instantiators are this op's own four factories. **However**, `ttnn/cpp/ttnn/operations/kv_cache/device/kernels/` contains three files with *identical basenames* — `dataflow/reader_update_cache_interleaved_start_id.cpp`, `dataflow/writer_update_cache_interleaved_start_id.cpp`, `compute/update_cache.cpp` — which are **separate copies** that `kv_cache`'s own factory instantiates by its own path (`update_cache_multi_core_program_factory.cpp:298,317,346,357`). They are *not* shared files: editing them does nothing for this op, and editing this op's copies does nothing for `kv_cache`. A porter grepping by filename will land in the wrong directory. There is no `_metal2` fork beside any of this op's kernels, and none is needed — no fork, no sunset list, no cross-op coordination cost. (For planning context only: `kv_cache`'s `UpdateCacheMultiCoreProgramFactory` and `FillCacheMultiCoreProgramFactory` are `Is able to port? == yes` with `TensorParameter relaxation == none` on the sheet, so that op is a separate, independently portable workstream — not a dependency of this one.)
- **RTA varargs:** **none.** No kernel reads an argument at a loop-variable or data-selected index. Every read is either at a constant index (`reader_fill_cache:18-21`, `writer_fill_cache:71-78`, `reader_update_cache:16-21`, `writer_update_cache:17-23`) or a **fixed run** of `rt_args_idx++` at the top of the kernel (the five fused kernels), which the recipe classifies as legacy positional plumbing that dissolves into named args — not a vararg. No `get_common_arg_val` anywhere. **No CTA varargs either:** the only computed compile-time-arg offsets are `TensorAccessorArgs<N>()` chained through `constexpr` `next_compile_time_args_offset()`, which is a fixed set at constexpr offsets, so those get names too.
- **The aliased two-format intermediate DFB is load-bearing — do not split it.** In three of the four factories a *single* `CBDescriptor` carries **two** `CBFormatDescriptor`s, so two buffer indices alias one L1 allocation: `intermed0`/`intermed1` = `c_24`/`c_25` in `update_cache` (`paged_update_cache_program_factory.cpp:210-225`) and in tiled-fused (`paged_tiled_fused_update_cache_program_factory.cpp:223-238`), and `c_5`/`c_6` in RM-fused (`paged_row_major_fused_update_cache_program_factory.cpp:228-243`). The aliasing *is* the algorithm: the writer takes `cb_untilized_cache.get_read_ptr() + cache_tile_offset_B` (index 0), NoC-writes the new row into that L1 region in place, then publishes the *same* memory through index 1 via `cb_untilized_cache2.push_back(Wt)` for compute to re-tilize (`writer_update_cache:126,134`; `writer_paged_fused:137,145`; `writer_paged_row_major:140,148`). Port this as **one** `DataflowBufferSpec` with two format descriptors. Splitting it into two independent DFBs compiles, validates, and silently produces wrong numerics.
- **Runtime-selected DFB index across two DFBs with disjoint core ranges (fused factories).** Three fused kernels pick which input CB to touch from a *runtime* arg: `reader_paged_fused_...:32-35`, `reader_paged_row_major_...:32-35`, `writer_paged_row_major_...:59-62` (`input_cb_id = is_input1 ? input1_cb_id : input2_cb_id`), and `compute/paged_fused_update_cache.cpp:23-26,36` does the same for `compute_kernel_hw_startup(in_cb, ...)` before branching to compile-time-parameterised `untilize<Wt, in1_cb, ...>` / `untilize<Wt, in2_cb, ...>`. Since a kernel cannot touch a DFB it has not bound, each of these KernelSpecs must bind **both** `src1` and `src2` — yet `src1`'s `CBDescriptor` covers only `input1_cores` and `src2`'s only `input2_cores`, and the two are validated **disjoint** (`paged_fused_update_cache_device_operation.cpp:350-351`) while the KernelSpecs span `all_cores_bb`. So on any given node exactly one of the two bound DFBs actually exists. This is the single most consequential porting shape in the op; resolve how the spec expresses it (both bound with per-node existence, or per-core-set KernelSpecs) *before* writing the fused specs. Nothing in Appendix A covers it, so it is not a gate — but it is not mechanical either.
- **One RTA slot, two different tensors, selected by core (fused factories).** Reader RTA[2] and writer RTA[1] carry `cache_tensor1` on `cores1` and `cache_tensor2` on `cores2` (tiled `:438,460` vs `:483,505`; RM `:435,457` vs `:481,503`), and `override_runtime_arguments` patches them the same way (`paged_fused_update_cache_device_operation.cpp:100-124`). A `tensor::name` binding is per-KernelSpec, so one reader spec would need *two* `TensorParameter`s reaching one arg position by node. Same structural question as the bullet above, on the tensor-binding channel rather than the DFB channel.
- **One RTA slot, tensor-or-scalar by config (`fill_cache`).** Writer RTA[4] is a `Buffer*` (batch-idx tensor) when `use_batch_idx_tensor`, and the plain scalar `operation_attributes.batch_idx_fallback` otherwise (`paged_fill_cache_program_factory.cpp:315-319`, patched at `:413`); writer RTA[6] is likewise a `Buffer*` or literal `0` (`:323-327`, patched at `:415`). In Metal 2.0 the binding channel and the named-scalar channel are different, so this one slot has to split into a config-conditional `TensorParameter` **plus** a named scalar arg. The kernel side already keys off the same CTA (`writer_fill_cache_interleaved.cpp:56,100-116`), so the branch exists — it just has to move onto two channels.
- **Per-core runtime-arg count varies within one KernelSpec (fused factories).** Working cores get 8 reader args / 8–9 writer args / 2 compute args, while every node in `unused_cores = all_cores_bb − all_cores` gets a **single** arg `{!has_work}` (tiled `:524-530`, RM `:523-529`) and early-returns on it. A `runtime_arg_schema` is one schema for the whole KernelSpec, so decide how the short-arg nodes are expressed (supply the full named set with don't-care values, or narrow the KernelSpec's core range) rather than discovering it at validation time. Note that `unused_cores` is non-empty only when `input1_cores ∪ input2_cores` is not itself a rectangle.
- **`override_runtime_arguments` is the translation unit of real work here, and it is index-addressed.** All 8 factories carry one, so the target is `CustomProgramSpecFactoryConcept` and each must be translated into a `ProgramRunArgs` producer. The three bodies (`paged_update_cache_program_factory.cpp:457`; `paged_fill_cache_program_factory.cpp:361`; `paged_fused_update_cache_device_operation.cpp:55-125`, shared by all four fused factories) reach args by **hard-coded positional index** (`reader_args[0]`, `writer_args[4]`, the `kReader*Arg` / `kWriter*Arg` constants) *and* reach CBs by **hard-coded position in `desc.cbs`** (`kInputCbPos = 1`; `kSrc1CbPos = 1`, `kSrc2CbPos = 2`, `kFirstOptionalCbPosTiled = 6`, `kFirstOptionalCbPosRowMajor = 5`). Those positional constants are exactly what named args and named DFB bindings replace, so the translation deletes them — but any mismatch is silent. The `kFirstOptionalCbPos*` constants in particular already encode the tiled-vs-RM CB-count difference by hand (`paged_fused_update_cache_device_operation.cpp:44-45`) and are walked with a post-increment (`:75-81`); read them together with the `if (use_index_tensor)` / `if (is_paged_cache)` push order in each factory before rewriting.
- **Kernels declare CB wrapper objects unconditionally for conditionally-allocated CBs.** e.g. `CircularBuffer cb_index(cb_index_id)` / `cb_page_table(page_table_cb_id)` at `reader_update_cache:56-57`, `writer_update_cache:64-65`, and the fused equivalents — constructed even when `use_index_tensor` / `is_paged_cache` is false and the CB was never allocated. Harmless today (every *access* is behind `if constexpr`), but a Metal 2.0 binding is not a no-op the way an unused `CircularBuffer(id)` is: check whether the binding must be made conditional alongside the `DataflowBufferSpec`.
- **`share_cache` cross-core semaphore chain.** One plain `SemaphoreDescriptor` per factory (`paged_update_cache_program_factory.cpp:247`; tiled `:260`; RM `:256`), used only when `share_cache` is set: writer *i* signals reader *i+1* via `Semaphore<>::up(noc, send_core_x, send_core_y, 1)` (`writer_update_cache:165`; `writer_paged_fused:176`; `writer_paged_row_major:179`), awaited at `reader_*:127`/`:153`/`:155`, with the `send_core_x/y` **physical** coordinates baked into RTAs host-side (`worker_core_from_logical_core`, `paged_update_cache_program_factory.cpp:395`; tiled `:422,427`; RM `:419,424`). Ports as an ordinary `SemaphoreSpec` + `sem::name`; note the trailing `noc.async_atomic_barrier()` (`writer_paged_fused:182`, `writer_paged_row_major:185`, `writer_update_cache:166`) whose comment records a real Watcher NOC-idle race — keep it.

---

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ✓ clean.** No `⚠`, no `✗`, no `⭐`. Every donor is either `tt_metal` firmware/LLK or the official shared kernel library, and every donor function this op calls takes its CB handles as `uint32_t` NTTPs — the ✓ row of the shape table. There is no cross-family donor, no in-family shared kernel, no `CircularBuffer&` in any donor signature, no `uint32_t sem_id` / `sem_addr` escape, no `TensorAccessorArgs<N>` or CTA-offset-NTTP donor parameter, and no pre-Device-2.0 addr-gen donor (Shape 4) — so no donor contributes to the Device 2.0 gate.

**Summary table** (one row per (op kernel, donor file)):

| Op kernel | Donor file | Bucket | Status |
|---|---|---|---|
| all 8 dataflow kernels | `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/core_local_mem.h`, `api/tensor/noc_traits.h` | 1 — `tt_metal/*` (LLK/HAL/firmware) | ✓ |
| `reader_update_cache`, `reader_paged_fused`, `reader_paged_row_major`, `writer_update_cache`, `writer_paged_fused`, `writer_paged_row_major` | `api/dataflow/noc_semaphore.h` | 1 | ✓ |
| `writer_update_cache`, `writer_paged_fused`, `writer_paged_row_major` | `api/dataflow/endpoints.h` | 1 | ✓ |
| `compute/update_cache`, `compute/paged_fused_update_cache`, `compute/paged_row_major_fused_update_cache` | `api/compute/common.h`, `api/compute/pack_untilize.h`, `api/compute/tilize.h` | 1 | ✓ |
| `compute/update_cache`, `compute/paged_fused_update_cache`, `compute/paged_row_major_fused_update_cache` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` | 2 — official shared kernel library | ✓ |
| `compute/update_cache`, `compute/paged_fused_update_cache`, `compute/paged_row_major_fused_update_cache` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` | 2 | ✓ |

**Per-call detail** — omitted for the bucket-1 donors (all ✓). The two bucket-2 donors are the only non-firmware escape, and both are ✓, so this is recorded for completeness rather than as an action:

| Donor | Function called | Signature shape | Status |
|---|---|---|---|
| `kernel_lib/untilize_helpers.hpp:145-154` | `compute_kernel_lib::untilize<block_width_tiles, input_dfb, output_dfb, ...>(uint32_t num_blocks)` | CB handles are `uint32_t` **NTTPs** — the `uint32_t cb_id` row; `dfb::name`'s constexpr cast covers template-parameter position. Parameters are already *named* `input_dfb` / `output_dfb`, i.e. the donor is DFB-aware. | ✓ OK — no donor-side change, no fork |
| `kernel_lib/tilize_helpers.hpp:153-163` | `compute_kernel_lib::tilize<block_width_tiles, input_dfb, output_dfb, ...>(uint32_t num_blocks, std::optional<uint32_t>)` | same | ✓ OK |

**Borrowed kernel files (file-path instantiation): none.** All 11 kernel `.cpp` files the four factories instantiate are owned by this op directory, and no other op instantiates any of them (`grep -rl 'operations/experimental/paged_cache/device/kernels'` → only this op's own four factories). No `_metal2` fork exists beside any of them and none is needed. **There is no sunset list and no cross-op coordination cost for this port.** See *Heads-ups* for the same-basename trap in `ttnn/cpp/ttnn/operations/kv_cache/device/kernels/`.

### Relaxation candidates — **FALLIBLE, candidates to verify; default strict**

The ops team owns the real analysis, and the answer for this op is already in (strict, no relaxation). Recorded only because the recipe asks for what a custom hash reveals, and because it happens to *support* rather than challenge the strict decision:

All three `compute_program_hash` overrides pass **`tensor_args` in full** to `operation::hash_operation<...>` (`paged_update_cache_device_operation.cpp:328-336`; `paged_fill_cache_device_operation.cpp:214-215`; `paged_fused_update_cache_device_operation.cpp:383-388`). What each one *narrows* relative to the default is only **`operation_attributes`** members, never tensor properties:

| DeviceOperation | Attributes excluded from the key | Where the excluded value is re-supplied on a cache hit |
|---|---|---|
| `PagedUpdateCacheDeviceOperation` | `update_idxs` (a `std::vector<uint32_t>` of runtime positions), `batch_offset` (validated `== 0` at `:298`) | `update_idxs` → `compute_update_cache_offsets()` re-derives `cache_start_id` / `tile_update_offset_B` and `override_runtime_arguments` re-patches them (`paged_update_cache_program_factory.cpp:498,511-515`) |
| `PagedFillCacheDeviceOperation` | `batch_idx_fallback`, `noop` | both re-patched (`paged_fill_cache_program_factory.cpp:395-399,413-414`) |
| `PagedFusedUpdateCacheDeviceOperation` | `update_idxs`, `batch_offset` (validated `== 0` at `:332`) | re-patched via `compute_{tiled,row_major}_fused_offsets` + `patch_runtime_args` (`paged_fused_update_cache_device_operation.cpp:100-124`) |

Since no tensor property is dropped from the key, two calls that cache-hit necessarily carry identical `TensorSpec`s for every tensor argument — which is what a strict `TensorParameter` match requires. **No relaxation candidate is visible in these hashes.** (Fallible: I did not audit `hash_operation`'s reflection over `tensor_args`, and this is not the ops-team analysis.)

### TTNN factory analysis (sheet-derived facts with `file:line` evidence)

- **Concept (current):** `descriptor`, all 8 factories. Evidence: the `create_descriptor` declarations listed in the identifying section. **Not** `WorkloadDescriptor` despite four `*MeshWorkloadFactory` names.
- **Target concept:** `CustomProgramSpecFactoryConcept` — driven by `Override runtime args method? == yes`, agreeing with the sheet's own `Porting Target` cell.
- **Op-owned tensors:** none. No `WorkloadDescriptor`, no `buffers` vector under `device/`.
- **MeshWorkload need:** none genuine. The mesh variants exist purely for **per-coordinate filtering**, and they implement it two different ways worth recording: `update_cache` and the two fused families return an **empty `ProgramDescriptor`** for an excluded coordinate (`paged_update_cache_program_factory.cpp:448-453`; tiled `:544-549`; RM `:547-552`), whereas `fill_cache` returns a **fully-built descriptor whose kernels early-exit** on a `noop` RTA (`paged_fill_cache_program_factory.cpp:33-40,348-359`, consumed at `reader_fill_cache_interleaved.cpp:23-25` and `writer_fill_cache_interleaved.cpp:80-82`) so the cache slot is still populated for that coordinate. The `override_runtime_arguments` hooks mirror the same choice (`paged_update_cache_program_factory.cpp:472-475`; `paged_fused_update_cache_device_operation.cpp:129-134`; `paged_fill_cache_program_factory.cpp:399`). Both idioms have to survive the port unchanged.
- **Pybind `create_descriptor`:** none. Nothing for the port to delete, so no user-visible API change from this port.
- **Other risky pybind:** none. `paged_cache_nanobind.cpp` binds only the three public entry points through `ttnn::bind_function`; no device-op or factory internals are exposed.
- **Custom hash:** present on all 3 DOps (sites above). The port leaves each exactly as it is.
- **`get_dynamic_runtime_args`:** absent.
- **`override_runtime_arguments`:** present on all 8 factories; the four fused ones funnel into one shared `patch_runtime_args` template (`paged_fused_update_cache_device_operation.cpp:54-125`) and the two mesh variants of `update_cache` / `fill_cache` each delegate to their single-device sibling (`paged_update_cache_program_factory.cpp:522-530`; `paged_fill_cache_program_factory.cpp:420-428`). One translation per program body, four bodies, eight factories.
- **Gate conjuncts confirmed absent:** `get_dynamic_runtime_args`, genuine multi-program `WorkloadDescriptor`. The third (`TensorParameter relaxation != none`) is the sheet cell resolved out-of-band to `none`.

---

## Misc anomalies  *(team-only, non-gating, not porter-actionable)*

Noticed while reading; these route to the ops team and the port must not act on them.

1. **Six dead `DataFormat` locals.** `cache_data_format` / `data_format` is assigned from `get_dataformat(cb_id)` and never read, in `reader_update_cache_interleaved_start_id.cpp:64`, `writer_update_cache_interleaved_start_id.cpp:54`, `reader_paged_fused_update_cache_interleaved_start_id.cpp:77`, `reader_paged_row_major_fused_update_cache_interleaved_start_id.cpp:77`, `writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp:66`, `writer_fill_cache_interleaved.cpp:144`. Deleting all six simultaneously clears the Device 2.0 gate above — this is the one anomaly whose fix is also the unblock.
2. **`log_base_2_of_page_size` is a dead CTA that is always 0.** Read at `reader_update_cache_interleaved_start_id.cpp:29` (CTA 6), `reader_paged_fused_...:43` (CTA 8), `reader_paged_row_major_...:43` (CTA 8) and never used. Host-side it is a local initialised to `0` and never assigned: `paged_update_cache_program_factory.cpp:116`, `paged_tiled_fused_update_cache_program_factory.cpp:111`, `paged_row_major_fused_update_cache_program_factory.cpp:111` (the last is even declared `const ... = 0`). Three CTA slots that could be removed from all three readers.
3. **`log2_page_table_stick_size` is a dead CTA in four kernels, and in `fill_cache` it is a *computed* dead value with a latent precision bug.** Read and unused at `reader_update_cache_...:38` (CTA 13), `reader_paged_fused_...:52` (CTA 15), `reader_paged_row_major_...:52` (CTA 15), `writer_fill_cache_interleaved.cpp:47` (CTA 6). In the first three the host value is a hardcoded `0`. In `fill_cache` it is `std::log2(page_table_stick_size_B)` truncated to `uint32_t` (`paged_fill_cache_program_factory.cpp:119`), and the adjacent `TT_FATAL` only requires `page_table_stick_size_B % 32 == 0` (`:116-118`), **not** a power of two — so for e.g. 96 bytes the value would be 6, not a valid shift. Harmless *only* because no kernel reads it. If anyone ever wires it up, the assertion has to be tightened first.
4. **`PagedFillCacheParams::noop` is never set `true` by any caller.** Declared at `paged_fill_cache_device_operation_types.hpp:16`; the sole public entry point hardcodes `.noop = false` (`paged_fill_cache_device_operation.cpp:237`). The functional noop path is driven entirely by the mesh-coordinate test inside `paged_fill_cache_noop` (`paged_fill_cache_program_factory.cpp:33-40`), which ORs in this attribute as a second, unused source. Correctly excluded from the hash; the field itself is dead API surface.
5. **`paged_row_major_fused_update_cache.cpp` carries three dead inputs.** CTAs 0 and 1 (`in1_cb`, `in2_cb`) and RTA[1] (`is_input1`) exist only to compute `in_cb` at `:23-26`, which is marked `[[maybe_unused]]` and never used — the RM compute kernel genuinely does not touch the input CBs (the RM writer consumes them directly). The author already acknowledged this with the attribute; the args could be dropped from `compute_kernel_args` (`paged_row_major_fused_update_cache_program_factory.cpp:350-351`) and from the per-core `compute_desc.emplace_runtime_args` (`:468-473`, `:514-519`).
6. **Two unbalanced FIFO `wait_front`s in the row-major fused writer.** `cb_index.wait_front(1)` at `writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp:85` and `cb_page_table.wait_front(num_pages_to_read)` at `:96` have **no** matching `pop_front`. Both sibling writers do pop: `writer_paged_fused_...:112` (`cb_page_table`) and `:122` (`cb_index`), and `writer_update_cache_...:101,111` likewise. Benign today because each CB is filled once per dispatch and never re-waited, but it is an asymmetry against two siblings that do it the other way, and it is the kind of thing a later change to the loop structure turns into a hang.
7. **`max_blocks_per_seq` is read but unused in all six fused/update dataflow kernels.** `reader_update_cache_...:37`, `writer_update_cache_...:40`, `reader_paged_fused_...:51`, `writer_paged_fused_...:46`, `reader_paged_row_major_...:51`, `writer_paged_row_major_...:48`. The host computes it from `page_table.padded_shape()[1]` and it is genuinely load-bearing in `validate_on_program_cache_miss` as a bound, but no kernel range-checks `virtual_block_id` against it before indexing `page_table_ptr[virtual_block_id]`. Not a port concern; worth the ops team's attention as a missing on-device bound check.
8. **`fill_cache`'s `batch_idx_stick_size_B` default is a hardcoded `4` with a comment admitting the assumption** (`paged_fill_cache_program_factory.cpp:128`). Only reached when `use_batch_idx_tensor` is false, in which case the CB is not allocated and the CTA is unused, so it is inert — but the constant sits next to code that does the right thing (`tensor.element_size()`, `:134`) in the live branch.
9. **`PagedFillCacheDeviceOperation::validate_on_program_cache_miss` has a suspicious `||` chain** (`paged_fill_cache_device_operation.cpp:33-36`): the predicate mixes `input_tensor.dtype()` for the first two alternatives and `cache_tensor.dtype()` for the last two, so a `BFLOAT8_B` *cache* satisfies the check regardless of the input dtype (and the message says "input tensor"). Looks like a copy-paste slip rather than intent; the port does not touch validation.

---

## Per-DeviceOperation attribution

Findings are structurally common across the three DeviceOperations; the table records where they differ.

| Field | `PagedUpdateCacheDeviceOperation` | `PagedFillCacheDeviceOperation` | `PagedFusedUpdateCacheDeviceOperation` |
|---|---|---|---|
| Factories | 2 | 2 | 4 (tiled ×2, row-major ×2) |
| **Overall** | **RED** — Device 2.0 | **RED** — Device 2.0 | **RED** — Device 2.0 |
| Device 2.0 violations | 2 (`reader:64`, `writer:54`) | 1 (`writer_fill:144`) | 3 (`reader_paged_fused:77`, `reader_paged_rm:77`, `writer_paged_rm:66`) |
| Feature compatibility | GREEN (all N/A) | GREEN (all N/A) | GREEN (all N/A) |
| Sheet `Is able to port?` | `no` → relaxation column → cleared out-of-band | `no` → same | `no` → same |
| Concept / target | `descriptor` → `CustomProgramSpecFactoryConcept` | same | same |
| Custom hash | yes (`device_operation.cpp:313`) | yes (`device_operation.cpp:207`) | yes (`device_operation.cpp:371`) |
| Offset base pointers | GREEN (3 clean bases) | GREEN (5 clean bases) | GREEN (4 clean bases) |
| Tensor bindings | 3 Case 1, 1 clean | 5 Case 1 (2 with an overloaded tensor-or-scalar slot) | 2 Case 1, 2 clean, 2 config-split (Case 1 / clean) |
| TensorAccessor 3rd arg | N/A (no site) | N/A (no site) | N/A (no site) |
| CB endpoints | 8 `(CB, config)` — all 1:1 | 4 `(CB, config)` — 1 × 1:1, **3 self-loop** | 9 tiled + 8 RM `(CB, config)` — all 1:1 |
| Semaphores | 1 (`share_cache` chain) | none | 1 per factory (`share_cache` chain) |
| Borrowed-memory CBs | 1 (`c_1` input shard) | none | tiled: up to 4 (`c_1`, `c_2`, `c_3`†, `c_4`†) · RM: same († only when the tensor is L1-sharded) |
| Distinctive porting shapes | aliased two-format intermediate DFB | tensor-or-scalar overloaded RTA slots; noop-program mesh idiom | runtime-selected DFB index over disjoint core ranges; one RTA slot ↔ two cache tensors by core; variable per-core RTA count; aliased two-format intermediate DFB |
| RTA / CTA varargs | none | none | none |
| Out-of-directory coupling | ✓ clean | ✓ clean | ✓ clean |

---

## Questions for the user

1. **Is `get_dataformat(cb_id)` intended to be a sanctioned Device 2.0 free function?** This single question decides the whole verdict. The audit's sanctioned list is exactly `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)`, and the recipe says the list is the whole test — so I gated. But the three sit side by side in the same Device 2.0 header under one `#ifdef DATA_FORMATS_DEFINED` block (`tt_metal/hw/inc/api/dataflow/circular_buffer.h`: `get_tile_size()`, `get_tile_hw()`, `get_dataformat()`, each a one-line forward to `::<name>(cb_id_)`), and the rationale the recipe gives for sanctioning `get_tile_size` — the migration guide's own migrated example uses it — is silent on `get_dataformat` only because that example has no need for a data format. If the Device 2.0 owners consider `get_dataformat` sanctioned, this op is **GREEN today with zero code change** and a brief can be issued on a re-run. If not, the fix is 6 line deletions (all 6 results are unused) on the Device 2.0 track. Evidence at `METAL2_PREPORT_AUDIT.md` → *Gate detail → Device 2.0*.
2. **Should the readiness sheet be refreshed before the port is scheduled?** All 8 `experimental/paged_cache` rows still read `TensorParameter relaxation == (legality - pending analysis)` and `Is able to port? == no`. I have treated the relaxation conjunct as cleared on your instruction (strict, no relaxation), but the sheet is what downstream planning reads. Worth asking Diego to update both cells so the next auditor of this op doesn't re-raise it — and so a re-audit after the Device 2.0 fix doesn't have to carry this same out-of-band caveat.
3. **How should the fused factories' runtime-selected input DFB be expressed in a `ProgramSpec`?** (`reader_paged_fused_...:32-35` and siblings — see *Heads-ups*.) Three kernels choose between `src1` and `src2` from a runtime arg, so both must be bound, yet each DFB is allocated over only one of two disjoint core subsets while the KernelSpec spans their bounding box. Nothing in Appendix A covers it, so I did not gate — but a decision here (bind both and rely on per-node existence, vs. split into per-core-set KernelSpecs, vs. something else) belongs upstream of the port rather than inside it, and it will shape the fused specs more than any other single choice.

---

## Recipe notes

1. **The sanctioned-free-function list is the entire gate for this op, and its rationale doesn't generalise cleanly.** `metal2_audit.md` → *Device 2.0 prerequisite* → Green bullet: *"Currently sanctioned (do **not** flag): `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)`. Each is grounded in the Device 2.0 surface itself"*, followed by *"The list is the whole test"*. I applied it literally and REDed on `get_dataformat(cb_id)`. But the grounding argument offered for `get_tile_size` is that the migration guide's migrated example keeps it — and `get_dataformat` is absent from that example only because the example never needs a data format, not because it was considered and rejected. Meanwhile the Device 2.0 header wraps all three identically (`circular_buffer.h`: `get_tile_size()`, `get_tile_hw()`, `get_dataformat()`, one `#ifdef DATA_FORMATS_DEFINED` block). Suggestion: either add `get_dataformat` (and `get_tile_hw`) to the sanctioned list, or add one sentence saying the omission is deliberate. As written, a whole op's verdict turns on which of three sibling accessors happened to appear in an example — and the six offending calls here are dead code, which makes the gate feel especially incidental.
2. **The status-summary template has a stale feature row.** The `METAL2_PREPORT_AUDIT.md` template includes `| *Feature Support* — Variadic-CTA | Ok / Unsupported |`, but Appendix A has no Variadic-CTA entry (its three entries are GlobalCircularBuffer, `address_offset`, GlobalSemaphore) — and *RTA varargs* explicitly says CTA varargs port onto `KernelAdvancedOptions::compile_time_varargs` and do **not** gate. The row looks left over from an earlier Appendix A. I replaced it with one row per current Appendix A entry.
3. **"Multiple device-operations in one op directory" and the output contract can pull apart.** The bundling test is *shared factories or kernels* → one report; independent → *"audit each separately"*. Here the three DeviceOperations share **no** kernels and **no** factories, so the test says separate — but *Output: the two documents* specifies one `METAL2_PREPORT_AUDIT.md` per **op directory**, and the readiness sheet keys on `Op == experimental/paged_cache`, so three files would have nowhere to live and no consumer. I produced one bundled report with full per-DeviceOperation attribution. Suggestion: say what to do when the shared-code test says "separate" but the directory (and the sheet's op granularity) says "one" — I think one file with mandatory per-DOp attribution is right, but the recipe currently implies three.
4. **The Red-outcome scoping rule needs a tie-break when a RED of each kind is present.** This op has two: a Device 2.0 RED (op-code side → *skip* the seven informational subjects) and a readiness-verdict RED (clears with the op untouched → *run them anyway*). The rule is written as if one RED sets the policy. I ran all seven and said so, reasoning that (a) the exception's justification is staleness, and one of my two REDs demonstrably won't cause any, and (b) the specific Device 2.0 fix is six deletions of dead locals, which cannot invalidate a single finding in those seven subjects. Suggestion: add a sentence — "if any RED is of the clears-without-touching-the-code kind, run them" — or, better, let the auditor judge by *how much* the op-code fix would actually change, since "a Device 2.0 migration" spans deleting six dead lines to rewriting every kernel and the staleness risk differs by orders of magnitude.
5. **Provenance is assumed to be recoverable from the working checkout.** The command in *Output: the two documents* is to be run "from the checkout root", with the fallback being "the docs aren't from a tracked doc-branch checkout — record that instead". Neither case fit: the recipe was handed to me as a standalone file, the working repo has no `metal_2.0` doc tree, and a sibling checkout has both the tree *and* a byte-identical copy of the recipe. I pinned the hash from the sibling and said so. Suggestion: allow "run it wherever the doc tree lives, and name that path", since a hash from a verified-identical checkout is strictly better than "can't be pinned".
6. **Small thing that cost a few minutes:** *TensorParameter analysis* → *Detection — host side* → `Buffer*`-binding form says the framework auto-registers these and patches them on cache hits, which is correct — but every factory here *also* defines `override_runtime_arguments`, which the code comments say "supersedes `resolve_bindings`, so all addresses are ours". Both mechanisms are live in the same op and the recipe discusses them separately. A half-sentence noting that a `CustomProgramSpecFactoryConcept` op's override takes precedence over `Buffer*` auto-patching (so the `Buffer*` form's "correct-on-cache-hit today" reassurance is doubly true there) would have saved me re-reading the override bodies to check for a conflict.
