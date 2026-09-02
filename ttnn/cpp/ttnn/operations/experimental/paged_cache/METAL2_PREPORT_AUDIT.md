# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/paged_cache`

Three `DeviceOperation`s share this directory. They share **no** kernels and **no** program factories with each other (each family owns its own), so by the recipe's shared-code test they are independent; they are audited together here because the output contract is one report per *op directory*, they are one `Op` row-group (`experimental/paged_cache`) on the readiness sheet, and every gate finding below is structurally common to all three. **Per-DeviceOperation attribution is given for every finding** (see the dedicated section) so a downstream consumer can extract per-DeviceOperation status. See *Recipe notes*.

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

**All 11 kernel files in `device/kernels/` are referenced by a factory — none is unreferenced/dead.** (Verified against the four factories' `kernel_source` strings, which are assembled from two literals and so are invisible to a single-line path grep.)

> **Naming caution for any reader:** the four `*MeshWorkloadFactory` types do **not** return a `WorkloadDescriptor`. Each declares `create_descriptor(...) -> ProgramDescriptor` with an extra `mesh_dispatch_coordinate` parameter and delegates to its single-device sibling, returning an empty `ProgramDescriptor` (`update_cache` / both fused) or a fully-built noop descriptor (`fill_cache`) for coordinates outside `operation_attributes.mesh_coords`. The concept is `descriptor`, not `WorkloadDescriptor`; the name is a legacy holdover. There is no `create_workload_descriptor` in this directory (verified by grep).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `50e992a8ec2 2026-08-21 docs(metal_2.0): a run in flight freezes the kernel sources`
*(The working checkout `/localdev/edwinlee/Paged_Cache_Port` does not carry `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`; the provenance command was run in the sibling doc-branch checkout `/localdev/edwinlee/Port_Recipe`, whose `ai/audit/metal2_audit.md` is byte-identical (`diff -q`) to the recipe supplied for this run.)*

**Readiness sheet:** fetched fresh this session from the *"Operations analysis"* sheet (file id `1KUMj8SyBGlNMZlLFgs1MbAZlO2g6EoUc4KaxSlcy8jw`), 28 columns, 8 rows for `experimental/paged_cache`.

**Re-audit note.** This op was audited previously and REDed on the Device 2.0 prerequisite (six `get_dataformat(cb_id)` CB-index free-function holdovers). Two things have changed since, and **both blockers are now cleared**:

1. **Device 2.0 migration landed** at `47a266001ad` *"[Cleanup] Port Paged Cache to Device 2.0 (#54598)"* — eight dataflow kernels, 12 insertions / 18 deletions. It deleted all six dead `DataFormat` locals and additionally moved every `get_tile_size(cb_id)` free call onto `CircularBuffer::get_tile_size()` (which was already sanctioned, so that half was optional). This audit re-ran the full Device 2.0 scan from scratch on the current tree; see *Gate detail*.
2. **The readiness sheet was updated.** All 8 rows now read `TensorParameter relaxation == none` and `Is able to port? == yes`. The previous audit had to treat the relaxation conjunct as cleared out-of-band on the launching user's word; the sheet and the ops-team decision now agree, so no out-of-band caveat is carried forward.

`git diff 47a266001ad~1..HEAD` over the op directory shows the Device 2.0 commit as the **only** code change to the op since the previous audit (plus that audit's own `.md`). Host-side line numbers are therefore unchanged; kernel line numbers shifted by 1–2 lines and have all been re-resolved against the current tree.

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/paged_cache` |
| **Overall** | **GREEN** — every gate cleared. Brief issued (`METAL2_PORT_BRIEF.md`). |
| **DOps / Factories** | `PagedUpdateCacheDeviceOperation` → {`PagedUpdateCacheProgramFactory`, `PagedUpdateCacheMeshWorkloadFactory`} · `PagedFillCacheDeviceOperation` → {`PagedFillCacheProgramFactory`, `PagedFillCacheMeshWorkloadFactory`} · `PagedFusedUpdateCacheDeviceOperation` → {`PagedTiledFusedUpdateCacheProgramFactory`, `PagedTiledFusedUpdateCacheMeshWorkloadFactory`, `PagedRowMajorFusedUpdateCacheProgramFactory`, `PagedRowMajorFusedUpdateCacheMeshWorkloadFactory`} |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes — GREEN.** All 11 kernels structurally Device 2.0; **zero** CB-index-keyed free functions of any name, zero Device 1.0 idioms. Cleared by `47a266001ad`. |
| *Prereqs* — Cross-op escapes | **Ok** — donors are `tt_metal` `api/*` only, plus `ttnn/cpp/ttnn/kernel_lib/{tilize,untilize}_helpers.hpp` (+ their `.inl`) — the official shared kernel library, whose CB handles are `uint32_t` NTTPs (the ✓ row). No cross-family donors. **No borrowed kernel files** — every kernel this op instantiates lives in this op's own directory and is instantiated by no other op. |
| *Feature Support* — overall | **GREEN** — every Appendix A entry is `N/A` (feature absent). |
| *Feature Support* — GlobalCircularBuffer | N/A |
| *Feature Support* — CBDescriptor `address_offset` (non-zero) | N/A |
| *Feature Support* — GlobalSemaphore | N/A |
| *TTNN Readiness* — `Is able to port?` (the gate) | **`yes`** on all 8 rows. Cross-check of every checkable primary column: clean. **GATE CLEARED.** |
| *TTNN Readiness* — Concept (current) | `descriptor` — all 8 factories. **Cross-check: confirmed.** |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A (`Concept == descriptor`). Sheet cell blank — consistent. |
| *TTNN Readiness* — Custom hash | **Yes** (not a gate; port leaves it intact): `device/update_cache/paged_update_cache_device_operation.cpp:313`, `device/fill_cache/paged_fill_cache_device_operation.cpp:207`, `device/fused_update_cache/paged_fused_update_cache_device_operation.cpp:371`. All three hash `tensor_args` in full. **Cross-check: confirmed.** Backdoor route (`attribute_values` / `to_hash`) absent — sheet says `no`, grep agrees. |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** — grep over `device/` finds no such hook. **Cross-check: confirmed.** |
| *TTNN Readiness* — `override_runtime_arguments` | **Yes** (not a gate; selects `CustomProgramSpecFactoryConcept`) — all 8 factories. Bodies at `paged_update_cache_program_factory.cpp:457` & `:522`; `paged_fill_cache_program_factory.cpp:361` & `:420`; `paged_fused_update_cache_device_operation.cpp:395`, `:409`, `:428`, `:438`. **Cross-check: confirmed.** |
| *TTNN Readiness* — Pybind `create_descriptor` | **No** — `paged_cache_nanobind.cpp` has no `nb::class_` and no `create_descriptor` binding; only three `ttnn::bind_function` wrappers (`:48`, `:89`, `:134`). Nothing for the port to delete. **Cross-check: confirmed.** |
| *TTNN Readiness* — Op-owned tensors | **No** — no `WorkloadDescriptor` and no `buffers` vector anywhere in `device/` (grep). Sheet cell blank — no cross-column invariant violated. |
| *TTNN Readiness* — Target concept | **`CustomProgramSpecFactoryConcept`** (all 8 factories). Matches the sheet's own `Porting Target` column. |
| *Port work* — Offset base pointer | **none** — GREEN. All 12 `->address()` sites in the op are bare bases with no host arithmetic; every scalar offset (`cache_start_id`, `tile_update_offset_B`, `start_tile_id`, `start_row_num`) already travels as a *separate* arg. |
| *Port work* — Tensor bindings (per binding) | **15 distinct bindings** across the 3 DOps: **10 Case 1** (`TensorAccessor`), **3 clean** (borrowed-memory DFB), **2 config-split** (Case 1 on the DRAM-interleaved path, clean on the L1-sharded path). **No Case 2 anywhere** — the op never uses a *tensor* base pointer for hand-rolled NoC arithmetic. |
| *TTNN Readiness* — TensorParameter relaxation | Sheet, verbatim: **`none`** on all 8 rows → the conjunct clears. Nothing routes to the ops team. |
| *Port work* — TensorAccessor 3rd arg | **N/A — no accessor in the op passes a 3rd argument.** All **17** `TensorAccessor(...)` constructions across the 11 kernels are 2-arg. The subject never fires. |
| *Port work* — CB endpoints | **All legal or self-loop.** 29 `(CB, config)` instances across the 4 factory bodies: **26 plain 1:1**, **3 self-loop** (one-toucher, all in `fill_cache`). No multi-binding, no dead CB, no new conditional DFB. |

**CB endpoints** are dispositions, not gates: every out-of-window CB here resolves as a **self-loop** (one toucher). Recorded per `(CB, config)` below.

---

## Result

**GREEN → brief issued** at `ttnn/cpp/ttnn/operations/experimental/paged_cache/METAL2_PORT_BRIEF.md`.

Every gate-bearing subject clears: Device 2.0 ✓ · Feature compatibility ✓ (all three Appendix A entries `N/A`) · TTNN factory concept ✓ (`Is able to port? == yes`, cross-check clean) · Offset base pointers ✓ (no host-folded offset) · TensorAccessor 3rd argument ✓ (no site exists). `TensorParameter relaxation` is `none`, the only clearing value.

**Both blockers from the previous audit have been resolved on their respective tracks** — the Device 2.0 migration in `47a266001ad`, and the readiness-sheet relaxation cell now reading `none` / `yes`. No blocker replaced them; no clean-subset carve-out is needed because nothing is blocked.

**The port is not, however, mechanical.** Four structural shapes will dominate the porter's effort, none of them an Appendix A gate and none resolvable by a syntax swap. They are recorded in full under *Heads-ups* and carried into the brief; in descending order of consequence:

1. **A runtime-selected DFB index across two DFBs whose core ranges are disjoint** (both fused factories). Three kernels choose between `src1` and `src2` from a runtime arg, so each `KernelSpec` must bind *both* — yet each DFB is allocated over only one of two validated-disjoint core subsets while the `KernelSpec` spans their bounding box. Decide how the spec expresses this before writing the fused specs.
2. **An aliased two-format intermediate DFB that is load-bearing** (3 of 4 factories). One `CBDescriptor` carrying two `CBFormatDescriptor`s is the mechanism by which the writer updates untilized cache rows in place; splitting it into two DFBs compiles, validates, and silently produces wrong numerics.
3. **`override_runtime_arguments` is the real translation unit**, and it is index-addressed — hard-coded positional arg indices *and* hard-coded positions in `desc.cbs`. Four program bodies serve all eight factories.
4. **One RTA slot carrying two different tensors selected by core** (fused), and **one RTA slot carrying a tensor-or-scalar selected by config** (`fill_cache`).

---

## Gate detail

### TTNN factory concept (`Is able to port?`) — **GREEN**

Sheet values, identical on all 8 rows (quoted verbatim; column names resolved against the header row fetched this session):

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
| **`Is able to port?`** | **`yes`** |
| **`TensorParameter relaxation`** | **`none`** |
| `Provisional relaxation finding (Edwin)` | *(empty)* |
| `Op-owned tensors?` | *(empty)* |
| `Secretly SPMD Workload?` | *(empty)* |
| `Why secretly SPMD?` | *(empty)* |
| `Pointer patching perf issue?` | *(empty)* |
| `Formerly custom hashed?` | *(empty)* |
| `Model` | `other` (update_cache, fill_cache rows) / `llama` (fused rows) |
| `ProgramFactory used in llama?` | `yes` (update_cache, fill_cache) / `no` (fused) |
| `Uses llama kernels? (primary or shared)` | `yes` (update_cache, fill_cache) / `no` (fused) |
| `Factory definition path` / `Declared in` | the op's `*_program_factory.hpp` / `*_device_operation.hpp`, per family |

**No blocking column fires.** `TensorParameter relaxation` is `none`; `get_dynamic_runtime_args` is `no`; `Smuggled pointer` is `no`; `Known op issues` is empty; `Op Classification` does not read as broken; `Concept` is `descriptor`, not `legacy device-op` or `WorkloadDescriptor`. The gate clears on its own terms, not by out-of-band resolution.

*(One combination I record but do not vet, per the recipe's visibility rule: `Diego validation == no` alongside `Is able to port? == yes`. `Diego validation` is not one of the columns the audit's blocking table names, and `Is able to port?` is derived, so I read the verdict and leave the derivation alone. Flagging it only so a reader who expects the two to move together isn't surprised.)*

**Lightweight cross-check — every checkable column agrees with the code.**

| Column | Sheet | Code evidence | Verdict |
|---|---|---|---|
| `Concept` | `descriptor` | All 8 factories declare `static tt::tt_metal::ProgramDescriptor create_descriptor(...)`: `paged_update_cache_program_factory.hpp:17,35`; `paged_fill_cache_program_factory.hpp:17,37`; `paged_tiled_fused_update_cache_program_factory.hpp:29,55`; `paged_row_major_fused_update_cache_program_factory.hpp:29,55`. No `create_workload_descriptor`, no mesh-workload return type (grep over `device/`). | ✓ agrees |
| `Custom hash` | `yes` | `compute_program_hash` overridden on all 3 DOps (`paged_update_cache_device_operation.cpp:313`, `paged_fill_cache_device_operation.cpp:207`, `paged_fused_update_cache_device_operation.cpp:371`). Not a pybound-rename case (no pybind of internals). | ✓ agrees |
| `Backdoor custom hash` | `no` | No `attribute_values` and no `to_hash` token anywhere in the op. | ✓ agrees |
| `Runtime-args update (get_dynamic_runtime_args)` | `no` | No `get_dynamic_runtime_args` / `DynamicRuntimeArg` token anywhere in the op. | ✓ agrees |
| `Override runtime args method?` | `yes` | All 8 factories declare (`*_program_factory.hpp` — update `:25,42`; fill `:25,44`; tiled `:42,62`; RM `:42,62`) and define `override_runtime_arguments`. On a `descriptor` op this is the target-concept signal, not the legacy-concept signature, since `create_descriptor` is present alongside it — so it selects `CustomProgramSpecFactoryConcept` and does not gate. | ✓ agrees |
| `Pybind descriptor` | `no` | `paged_cache_nanobind.cpp`: no `nb::class_`, no `create_descriptor` binding — only `ttnn::bind_function<"paged_update_cache">` (`:48`), `<"paged_fused_update_cache">` (`:89`), `<"paged_fill_cache">` (`:134`). | ✓ agrees |
| `Smuggled pointer` | `no` | `create_descriptor` never emplaces a raw `->address()`. Every buffer arg is emplaced as an **annotated `Buffer*`** (`KernelDescriptor::RTArgList::push_back(Buffer*)` / `emplace_runtime_args`), which the framework auto-registers as a `BufferBinding`. All 12 `->address()` expressions in the op sit inside `override_runtime_arguments` / `patch_runtime_args`, which is the sanctioned patch path. | ✓ agrees |
| `Secretly SPMD Workload?` | *(blank)* | Only meaningful when `Concept == WorkloadDescriptor`; it isn't. | ✓ N/A |
| **Factory-set match** | 8 rows | Code has exactly 8 factory structs, names matching the sheet's `Factory (variant)` cells one-for-one (`paged_fill_cache_program_factory.hpp:16,33`; `paged_update_cache_program_factory.hpp:16,33`; `paged_tiled_fused_update_cache_program_factory.hpp:19,50`; `paged_row_major_fused_update_cache_program_factory.hpp:19,50`). No phantom row, no missing row. | ✓ agrees |

**Cross-column invariants.** `Runtime-args update == no` on a `descriptor` concept — legal. `Op-owned tensors?` is blank (not `yes`) on a `descriptor` concept — legal. No invariant violated.

### Device 2.0 (every kernel used) — **GREEN**

All 11 kernels are structurally Device 2.0, and the scan finds **no violation of any class** — not a broad Device 1.0 residue, not a single isolated CB-index-keyed free-function holdover.

**Positive evidence.** The kernels use `Noc` (8 dataflow kernels), the `CircularBuffer` wrapper with method-form FIFO and pointer calls, `Semaphore<>` (6 kernels), `CoreLocalMem<uint32_t>` for every NoC endpoint, `UnicastEndpoint` (3 writers), and `TensorAccessor` for every tensor access.

**Negative evidence — two exhaustive greps over `device/kernels/`, both empty:**

1. Legacy Device 1.0 idioms — `noc_async_read|noc_async_write|noc_async_*barrier|get_noc_addr|get_noc_addr_from_bank_id|noc_semaphore_*|get_semaphore|InterleavedAddrGen|ShardedAddrGen|InterleavedPow2AddrGen|cb_reserve_back|cb_push_back|cb_wait_front|cb_pop_front|evil_set_*` → **zero code hits.** (Two matches are prose: the word `noc_semaphore_inc` inside the explanatory comments at `writer_paged_fused_update_cache_interleaved_start_id.cpp:178` and `writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp:180`.)
2. CB-index-keyed free functions — bare (non-member) `get_tile_size|get_dataformat|get_tile_hw|get_read_ptr|get_write_ptr|get_local_cb_interface|get_cb_interface|get_local_cb_addr` → **zero hits.** Every one of the 30 `get_read_ptr()` / `get_write_ptr()` occurrences is a `cb_obj.` wrapper-method call, and all 8 `get_tile_size()` calls are `cb_obj.get_tile_size()` (e.g. `reader_update_cache_interleaved_start_id.cpp:63`, `writer_fill_cache_interleaved.cpp:143`).

The six `get_dataformat(cb_id)` holdovers that gated the previous audit are gone: `47a266001ad` deleted each one together with the dead `const DataFormat` local it fed. That commit also converted the eight `get_tile_size(cb_id)` free calls to the wrapper method — not required (the free function is on the audit's sanctioned list) but consistent, and it means **no CB-index free function of any name survives** in this op, so the `get_dataformat`-vs-sanctioned-list question that decided the previous verdict no longer bears on it. (The recipe question it raised is still open; see *Recipe notes* #1.)

*Not counted as violations (judgement recorded for transparency):*
- **`reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_*_ptr())`** — 16 sites across 6 kernels (e.g. `reader_update_cache_interleaved_start_id.cpp:82,106`; `writer_fill_cache_interleaved.cpp:113,128,150`). The migration guide's *Memory Access* section offers `CoreLocalMem<T>` as the modern form, and these kernels already use `CoreLocalMem<uint32_t>` for their NoC endpoints — so this is a *style* residue. It is not a CB-index-keyed free function, and the gate exists because Metal 2.0's binding tokens need a Device 2.0 wrapper object to attach to: here the CB *is* already a `CircularBuffer`, the cast is applied to the pointer it returns, and `dfb::name` binds regardless. Not a blocker; noted for the Device 2.0 team as optional cleanup.
- **`my_x[noc_id]` / `my_y[noc_id]`** alongside `noc.get_noc_id()` (3 writer kernels, e.g. `writer_update_cache_interleaved_start_id.cpp:114-116`) — the standard way to obtain own coordinates; no wrapper replacement exists and it is not a CB-index free function.

### Feature compatibility — GREEN (no entry fires)

Every Appendix A entry is UNSUPPORTED, so an absent feature is `N/A`, not a pass.

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | **N/A** | No `GlobalCircularBuffer` type, `using` alias, or `experimental::CreateGlobalCircularBuffer` call. No `#include <tt-metalium/global_circular_buffer.hpp>` (or the `experimental/` spelling) — the complete host include set is `buffer.hpp`, `circular_buffer.hpp`, `constants.hpp`, `core_coord.hpp`, `host_api.hpp`, `program.hpp`, `program_descriptors.hpp`, `tensor_accessor_args.hpp`, `work_split.hpp`. **Descriptor-API attachment checked by field name:** no `CBDescriptor` literal in the four factory bodies sets `.global_circular_buffer` (the 26 `CBDescriptor` literals set only `.total_size`, `.core_ranges`, `.format_descriptors`, and — for the 6 borrowed-memory ones — `.buffer`). No `.remote_index(`, no `remote_cb_*` identifier, no `remote_circular_buffer.h`, no `num_global_cb_receivers`. **Imperative attachment checked:** the two `UpdateDynamicCircularBufferAddress` sites (`paged_update_cache_program_factory.cpp:518`, `paged_fused_update_cache_device_operation.cpp:70`) both take a `Buffer&` (`*tensor_args.input_tensor.buffer()` / `*buffer`), i.e. the unrelated 3-arg `Buffer&` overload — explicitly the false-positive guard, not the `const GlobalCircularBuffer&` overload. No factory signature takes `std::optional<const GlobalCircularBuffer>&`. |
| CBDescriptor `address_offset` (non-zero) | **N/A** | The token `address_offset` does not appear anywhere in the op (grep, case-insensitive, host + kernels). No `.address_offset` on any `CBDescriptor`, no `CircularBufferConfig::set_address_offset`, no 4-argument `UpdateDynamicCircularBufferAddress(program, handle, buffer, offset)` — both sites are the 3-arg form, so the offset defaults to zero. No `cb_descriptor_from_sharded_tensor` call. The six borrowed-memory CBs bind at base (`.buffer = <ptr>` with no offset field), which is the ordinary, supported pattern. Nothing to escalate to the runtime team. |
| GlobalSemaphore | **N/A** | No `GlobalSemaphore` type or `using` alias, no `experimental::CreateGlobalSemaphore`, no `#include <tt-metalium/global_semaphore.hpp>`, no factory signature taking `const GlobalSemaphore&` / `std::optional<GlobalSemaphore>`. The op's only semaphores are three plain `SemaphoreDescriptor{...}` literals (`paged_update_cache_program_factory.cpp:247`, `paged_tiled_fused_update_cache_program_factory.cpp:260`, `paged_row_major_fused_update_cache_program_factory.cpp:256`) — the regular path, supported as `SemaphoreSpec`. Kernel side uses `Semaphore<>` (`api/dataflow/noc_semaphore.h`), the guarded non-Global form. |

### CB endpoints (GATE-free) — all legal or self-loop

Run in full: the Device 2.0 gate is GREEN, so the Device-2.0 idioms this scan keys on (`CircularBuffer` wrapper methods, `Semaphore` objects) are intact and no deferral applies.

**Census method.** Per CB, per node, counting every toucher — FIFO produce (`reserve_back`/`push_back`), FIFO consume (`wait_front`/`pop_front`), and raw pointer access (`get_write_ptr`/`get_read_ptr`). A kernel that raw-peeks a CB it already FIFO-owns is counted **once** (the peek rides its own binding). The `compute_kernel_lib::untilize<Wt, input_dfb, output_dfb>` / `tilize<Wt, input_dfb, output_dfb>` helpers were read to confirm their FIFO behaviour (`untilize_helpers.inl:199-268`, `tilize_helpers.inl:149-191`): each builds a `DataflowBuffer` from the index, `wait_front`/`pop_front`s its input and `reserve_back`/`push_back`es its output — so the compute kernel is a **locked consumer** of the first template CB and a **locked producer** of the second. `compute_kernel_hw_startup(a, b)` reads CB *format* config, not memory, and in every case here `a` and `b` are CBs the same kernel already touches through a helper, so it adds no toucher.

**Every CB-index CTA was resolved through the factory's argument list to the actual `CBIndex`** rather than trusting the kernel's local name; several are deliberately counter-intuitive (in all three writers the CTA named `cache_cb_id` is the *output* CB, `c_16` / `c_7`, not the cache CB `c_0`).

**Hidden-second-writer hunt (face (a)): performed and came back empty.** No kernel in the op writes another kernel's CB through `get_write_ptr()` / `fifo_wr_ptr` + offset without owning the FIFO. The only semaphore in the op (`in0_sequential_mode`, one per factory) coordinates *share_cache* ordering between a writer and the *next core's* reader — `Semaphore<>::up(noc, send_core_x, send_core_y, 1)` at `writer_update_cache_interleaved_start_id.cpp:164`, `writer_paged_fused_...:176`, `writer_paged_row_major_...:178`, awaited at `reader_update_cache_...:126-128`, `reader_paged_fused_...:152-154`, `reader_paged_row_major_...:154-156` — not a raw CB co-fill handshake. **Multiple-readers hunt (face (b)): empty** — no CB has base-pointer read sites in 2+ co-resident kernels. **Dual-instance work-split hunt (face (c)): empty** — no factory pushes the same `kernel_source` into two `KernelDescriptor`s; each factory emits exactly one reader, one writer, and (except `fill_cache`) one compute descriptor, each over one core range. The fused factories *look* adjacent to that shape because one reader instance serves both `cores1` and `cores2`, but that is one instance over disjoint node sets with per-core args — ordinary 1:1 per node, not two touchers.

**`PagedUpdateCacheProgramFactory` / `PagedUpdateCacheMeshWorkloadFactory`** (identical bodies; `all_cores` = the input shard grid, one user per core):

| CB | Index | Config | Touchers on a node | Verdict |
|---|---|---|---|---|
| `src0_cb_index` (cache tiles in) | `c_0` | all | reader locked-P (`reader:132,143`; peek @134) · compute locked-C (`untilize<Wt, cache_cb, untilized_cache_cb>`, `compute/update_cache.cpp:48`) | **1:1** |
| `src1_cb_index` (input shard, **borrowed** `.buffer = in1_buffer` @`factory:208`) | `c_1` | all | reader locked-P (`reader:60-61`) · compute locked-C (`untilize<Wt, in_cb, untilized_in_cb>`, `compute:39-45`) | **1:1** |
| `intermed0_cb_index` | `c_24` | all | compute locked-P (`compute:48` output) · writer locked-C (`writer:122,134`; peek @125) | **1:1** |
| `intermed1_cb_index` | `c_25` | all | writer locked-P (`writer:123,133`) · compute locked-C (`tilize<Wt, untilized_cache2_cb, out_cb>`, `compute:51`) | **1:1** |
| `intermed2_cb_index` | `c_26` | all | compute locked-P (`compute:39-45` output) · writer locked-C (`writer:113,160`; peek @117) | **1:1** |
| `output_cb_index` | `c_16` | all | compute locked-P (`compute:51` output) · writer locked-C (`writer:137,148`; peek @139) | **1:1** |
| `cb_index_id` | `c_2` | `use_index_tensor` only (CB not allocated otherwise, `factory:254-264`) | reader locked-P (`reader:76,81`; peek @77) · writer locked-C (`writer:72,110`; peek @73) | **1:1** |
| `cb_pagetable_id` | `c_3` | `is_paged_cache` only (`factory:266-276`) | reader locked-P (`reader:96,105`; peek @97) · writer locked-C (`writer:87,100`; peek @88) | **1:1** |

**`PagedFillCacheProgramFactory` / `PagedFillCacheMeshWorkloadFactory`** (no compute kernel):

| CB | Index | Config | Touchers on a node | Verdict |
|---|---|---|---|---|
| `src0_cb_index` (input tiles) | `c_0` | all | reader locked-P (`reader_fill:38,46`; peek @39) · writer locked-C (`writer_fill:196-197, 231-232, 235-236, 244`) | **1:1** |
| `page_table_cb_index` | `c_1` | all | **writer only** — `reserve_back(1)` @148 + raw write via `get_write_ptr()` @149 + repeated `noc.async_read` into that pointer @210-216; never pushed, never popped, no other kernel references the index | **self-loop** (one toucher) |
| `cb_batch_idx_id` | `c_2` | `use_batch_idx_tensor` only (`factory:199-211`) | **writer only** — `reserve_back(1)` @102 + raw @103-113 | **self-loop** |
| `cb_valid_seq_len_id` | `c_3` | `use_valid_seq_len` only (`factory:212-222`) | **writer only** — `reserve_back(1)` @123 + raw @124-128 | **self-loop** |

**`PagedTiledFusedUpdateCacheProgramFactory` / `...MeshWorkloadFactory`** (kernels over `all_cores_bb`; `c_1` allocated only over `input1_cores`, `c_2` only over `input2_cores`, the two validated disjoint at `paged_fused_update_cache_device_operation.cpp:350-351`):

| CB | Index | Core range | Config | Touchers on a node | Verdict |
|---|---|---|---|---|---|
| `cache_cb_index` | `c_0` | `all_cores_bb` | all | reader locked-P (`reader:158,169`; peek @160) · compute locked-C (`compute/paged_fused_update_cache.cpp:59`) | **1:1** |
| `src1_cb_index` (**borrowed** `in1_buffer` @`factory:211`) | `c_1` | `input1_cores` | all | reader locked-P (`reader:73-74`) · compute locked-C (`compute:48-54`) | **1:1** |
| `src2_cb_index` (**borrowed** `in2_buffer` @`factory:221`) | `c_2` | `input2_cores` | all | reader locked-P (`reader:73-74`) · compute locked-C (`compute:40-46`) | **1:1** |
| `intermed0_cb_index` | `c_24` | `all_cores_bb` | all | compute locked-P (`compute:59` output) · writer locked-C (`writer:134,146`; peek @137) | **1:1** |
| `intermed1_cb_index` | `c_25` | `all_cores_bb` | all | writer locked-P (`writer:135,145`) · compute locked-C (`compute:62`) | **1:1** |
| `intermed2_cb_index` | `c_26` | `all_cores_bb` | all | compute locked-P (`compute:40-54` output) · writer locked-C (`writer:125,172`; peek @129) | **1:1** |
| `output_cb_index` | `c_16` | `all_cores_bb` | all | compute locked-P (`compute:62` output) · writer locked-C (`writer:149,160`; peek @151) | **1:1** |
| `cb_index_id` (**borrowed** when the index tensor is L1-sharded, `.buffer = index_buffer_ptr` @`factory:276`) | `c_3` | `all_cores_bb` | `use_index_tensor` only | reader locked-P (`reader:88,95`; peek @89) · writer locked-C (`writer:77,122`; peek @78) | **1:1** |
| `cb_pagetable_id` (**borrowed** when the page table is L1-sharded, @`factory:289`) | `c_4` | `all_cores_bb` | `is_paged_cache` only | reader locked-P (`reader:105,121`; peek @106) · writer locked-C (`writer:88,112`; peek @89) | **1:1** |

**`PagedRowMajorFusedUpdateCacheProgramFactory` / `...MeshWorkloadFactory`** (different CB indices — no `intermed2`; the RM writer consumes the row-major input shard directly instead of an untilized intermediate, via CTAs 3/4 = `src1_cb_index`/`src2_cb_index` at `factory:327-328`):

| CB | Index | Core range | Config | Touchers on a node | Verdict |
|---|---|---|---|---|---|
| `cache_cb_index` | `c_0` | `all_cores_bb` | all | reader locked-P (`reader:160,171`; peek @162) · compute locked-C (`compute/paged_row_major_fused_update_cache.cpp:39`) | **1:1** |
| `src1_cb_index` (**borrowed** `in1_buffer` @`factory:216`) | `c_1` | `input1_cores` | all | reader locked-P (`reader:73-74`) · **writer** locked-C (`writer:127,174`; peek @131) — RM compute does *not* touch it (`compute:23`, `[[maybe_unused]]`) | **1:1** |
| `src2_cb_index` (**borrowed** `in2_buffer` @`factory:226`) | `c_2` | `input2_cores` | all | reader locked-P (`reader:73-74`) · writer locked-C (`writer:127,174`) | **1:1** |
| `intermed0_cb_index` | `c_5` | `all_cores_bb` | all | compute locked-P (`compute:39` output) · writer locked-C (`writer:136,148`; peek @139) | **1:1** |
| `intermed1_cb_index` | `c_6` | `all_cores_bb` | all | writer locked-P (`writer:137,147`) · compute locked-C (`compute:42`) | **1:1** |
| `output_cb_index` | `c_7` | `all_cores_bb` | all | compute locked-P (`compute:42` output) · writer locked-C (`writer:151,162`; peek @153) | **1:1** |
| `cb_index_id` | `c_3` | `all_cores_bb` | `use_index_tensor` only | reader locked-P (`reader:89,97`; peek @90) · writer locked-C (`writer:84` — `wait_front` with **no matching `pop_front`**; see *Misc anomalies* #5) | **1:1** |
| `cb_pagetable_id` | `c_4` | `all_cores_bb` | `is_paged_cache` only | reader locked-P (`reader:109,124`; peek @110) · writer locked-C (`writer:95` — **no matching `pop_front`**) | **1:1** |

**No dead CB anywhere.** Every allocated `buffer_index` is referenced by at least one bound kernel in the config that allocates it, confirmed across all instantiations (index-tensor on/off, paged on/off, page-table DRAM vs L1-sharded, index tensor DRAM vs L1-sharded, batch-idx tensor on/off, valid-seq-len on/off, share_cache on/off, tiled vs row-major, `cache_position_modulo` zero/non-zero). The configuration-optional CBs in each factory (`cb_index`, `cb_pagetable`, plus `fill_cache`'s `cb_batch_idx` / `cb_valid_seq_len`) are **already conditionally allocated on the host side** — so there is no "dead in some configs, live in others" case needing a new conditional DFB spec; the existing `if (use_index_tensor)` / `if (is_paged_cache)` / `if (use_batch_idx_tensor)` / `if (use_valid_seq_len)` guards translate directly.

**Two node-level nuances, both non-blocking:**

1. In the fused factories the kernels span `all_cores_bb` (the bounding box of `input1_cores ∪ input2_cores`), and any node in `unused_cores = all_cores_bb − all_cores` receives a single runtime arg `{!has_work}` and early-returns, so it touches nothing at runtime. That is *runtime* control flow, not a config difference — the `buffer_index` is still statically referenced by the kernels bound over that range — so those CBs are **not** dead there and must not be dropped. The porter-facing consequence (a per-core runtime-arg count that varies within one `KernelSpec`) is recorded under *Heads-ups*.
2. Also in the fused factories, `c_1` exists only on `input1_cores` and `c_2` only on `input2_cores`, while every `KernelSpec` spans `all_cores_bb` and must bind both (the index is runtime-selected). So on any given node exactly one of the two bound DFBs actually exists. This is a *spec-expression* question, not a census question — the census on each node is still 1P+1C for whichever CB exists there — and it is the op's most consequential porting shape; see *Heads-ups*.

### Offset base pointers — GREEN

`experimental/paged_cache` is **not** listed in the offset-base-pointer triage analysis (`2026-07-19_offset_base_pointers.md`, a dated prior, not an authority) — and, per the recipe, "not in the tables" is not allowed to stand in for "scanned and clean", so the recognition scan was run on **every** address argument in the op.

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

- `cache_start_id` (a **tile index**, not an address) — computed host-side as `cache_batch_tile_offset + (update_idx / TILE_HEIGHT) * Wt` (`paged_update_cache_program_factory.cpp:80`; tiled `:72`; RM `:72`) and consumed kernel-side purely as a `TensorAccessor` `{.page_id = ...}` (`reader_update_cache:137`, `writer_update_cache:142`, and the fused equivalents). A page index, never added to a base address.
- `tile_update_offset_B` / `cache_tile_offset_B` — a **byte offset into an L1 CB**, added kernel-side to `cb_untilized_cache.get_read_ptr()` (`writer_update_cache:125`; fused `:137`; RM `:139`). Not a tensor base.
- `start_tile_id` = `num_blocks_written * Wt` and `start_row_num` (`paged_fill_cache_program_factory.cpp:303,313`) — work-split scalars, consumed as loop bounds and `{.page_id}` values.
- Kernel-side pointer arithmetic on borrowed CBs — `page_table_cb_wr_ptr += my_batch_idx * page_table_stick_size` (`reader_paged_fused:118`, `reader_paged_row_major:122`) and the read-pointer mirrors (`writer_paged_fused:91`, `writer_paged_row_major:98`). These offset an **L1 CB pointer inside the kernel**, not a host-folded device base, so they are outside this subject entirely (and are preserved verbatim by the borrowed-DFB translation).

**Type 3** (`address_offset`) — absent; see the Appendix A row. **Type 4** (`ttnn::narrow` / interior-base `MeshBuffer::create`) — absent (grep). Reconciliation outcome for every RTA: *no fold, op not in the tables* → clean, handed to *TensorParameter analysis* below.

### TensorAccessor 3rd argument — N/A

**No accessor in this op passes a 3rd argument, so the subject never fires.** This is a *no sites* finding, not "sites found and classified redundant". All **17** `TensorAccessor(...)` constructions across the 11 kernels are the 2-argument form `TensorAccessor(args, addr)`:

`reader_fill_cache_interleaved.cpp:29` · `writer_fill_cache_interleaved.cpp:101,122,145,146` · `reader_update_cache_interleaved_start_id.cpp:69,74,95` · `writer_update_cache_interleaved_start_id.cpp:55` · `reader_paged_fused_...:82,87,109` · `writer_paged_fused_...:60` · `reader_paged_row_major_...:82,87,113` · `writer_paged_row_major_...:67`.

The op is not in the 3rd-arg triage analysis (`2026-07-06_tensor_accessor_3rd_arg_triage.md`, a dated prior), and the syntactic scan for a newly-added site — which a dated table could not have caught — likewise comes back empty (a regex for a 3-comma-separated-argument `TensorAccessor(` over `device/kernels/` returns nothing). Nothing to drop at port time, no `dynamic_tensor_shape` to set, no Class 3/4/Special to escalate.

---

## Port-work summary  *(mirrors the brief)*

### Tensor bindings — 15 bindings, per DeviceOperation

**`PagedUpdateCacheDeviceOperation`** — 4 bindings:

| Binding | Delivery in legacy | Kernel use | Case |
|---|---|---|---|
| `cache_tensor` | `Buffer*` → reader RTA[0] & writer RTA[0] (`factory:406,426`); raw address re-applied by `override_runtime_arguments` (`:504,509`) | `TensorAccessor(s0_args, cache_addr)` — `reader:69`, `writer:55`; all access via `{.page_id}` | **Case 1** |
| `input_tensor` | borrowed-memory CB `c_1` (`.buffer = in1_buffer`, `factory:200-209`); re-pointed on cache hit by `UpdateDynamicCircularBufferAddress` (`:518`) | reader `reserve_back`/`push_back` only (`reader:60-61`) — no NoC read; compute untilizes from it. No `TensorAccessor`, no address RTA | **clean** (causal-link gate: borrowed-memory DFB read → `DataflowBufferSpec::borrowed_from`) |
| `update_idxs_tensor` (optional) | `Buffer*` → reader RTA[2] (`factory:409`) | `TensorAccessor(index_tensor_args, index_tensor_addr)` — `reader:74`, read at `:79` | **Case 1** |
| `page_table` (optional) | `Buffer*` → reader RTA[4] (`factory:415`) | `TensorAccessor(page_table_args, page_table_tensor_addr)` — `reader:95`, read at `:98-103` | **Case 1** |

**`PagedFillCacheDeviceOperation`** — 5 bindings, all Case 1:

| Binding | Delivery in legacy | Kernel use | Case |
|---|---|---|---|
| `input_tensor` | `Buffer*` → reader RTA[0] (`factory:302`) | `TensorAccessor(src_args, src_addr)` — `reader_fill:29` | **Case 1** |
| `cache_tensor` | `Buffer*` → writer RTA[0] (`factory:311`) | `TensorAccessor(s0_args, dst_addr)` — `writer_fill:145` | **Case 1** |
| `page_table` | `Buffer*` → writer RTA[1] (`factory:312`) | `TensorAccessor(page_table_args, page_table_addr)` — `writer_fill:146` | **Case 1** |
| `batch_idx_tensor` (optional) | `Buffer*` → writer RTA[4] **or** the plain scalar `operation_attributes.batch_idx_fallback` in the same slot (`factory:315-319`, patched `:413`) | `TensorAccessor(batch_idx_tensor_args, batch_arg)` — `writer_fill:101` | **Case 1** + overloaded-slot split (see *Heads-ups*) |
| `valid_seq_len_tensor` (optional) | `Buffer*` → writer RTA[6] **or** literal `0` (`factory:323-327`, patched `:415`) | `TensorAccessor(valid_seq_len_tensor_args, valid_seq_len_addr)` — `writer_fill:122` | **Case 1** + overloaded-slot split |

**`PagedFusedUpdateCacheDeviceOperation`** — 6 bindings (identical in both the tiled and row-major factories):

| Binding | Delivery in legacy | Kernel use | Case |
|---|---|---|---|
| `cache_tensor1` | `Buffer*` → reader RTA[2] & writer RTA[1], **on `cores1` only** (tiled `:438,456-467`; RM `:435,453-465`) | `TensorAccessor(s0_args, cache_addr)` — `reader:82`, `writer:60` (tiled) / `:67` (RM) | **Case 1** |
| `cache_tensor2` | `Buffer*` → the **same** reader RTA[2] / writer RTA[1] slots, on `cores2` (tiled `:483,501-512`; RM `:481,499-511`) | same accessor | **Case 1** + per-core binding split (see *Heads-ups*) |
| `input_tensor1` | borrowed CB `c_1` over `input1_cores` (tiled `:203-212`; RM `:208-217`); re-pointed at `paged_fused_update_cache_device_operation.cpp:73` | reader FIFO-produces (`reader:73-74`); compute (tiled) or writer (RM) consumes | **clean** (borrowed DFB) |
| `input_tensor2` | borrowed CB `c_2` over `input2_cores` (tiled `:213-222`; RM `:218-227`); re-pointed at `:74` | same | **clean** (borrowed DFB) |
| `update_idxs_tensor` (optional) | `Buffer*` → reader RTA[4] (tiled `:441`; RM `:438`); **and** borrowed CB `c_3` when the tensor is L1-sharded (`.buffer = index_buffer_ptr`, tiled `:276`; RM `:272`; `index_buffer_ptr` is `nullptr` when not sharded — tiled `:117`, RM `:117`) | DRAM-interleaved path: `TensorAccessor(index_tensor_args, index_tensor_addr)` + `noc.async_read` (`reader:87,92`). L1-sharded path: the read is compiled out (`if constexpr (index_is_dram)`) and the value is read straight out of the borrowed CB (`reader:96,98`). | **config-split** — **Case 1** on DRAM-interleaved · **clean** on L1-sharded |
| `page_table` (optional) | `Buffer*` → reader RTA[6] (tiled `:447`; RM `:444`); **and** borrowed CB `c_4` when L1-sharded (tiled `:289`; RM `:285`) | DRAM: `TensorAccessor(page_table_args, ...)` + `noc.async_read` (`reader:109-116` tiled / `:113-120` RM). L1-sharded: pointer walk inside the borrowed CB (`reader:118` / `:122`, `writer:91` / `:98`). | **config-split** — **Case 1** on DRAM · **clean** on L1-sharded |

**No Case 2 anywhere in the op.** Every *tensor* base that reaches a kernel is fed to a `TensorAccessor`; nothing does hand-rolled NoC arithmetic on a tensor base, so no `get_bank_base_address` bridge is needed. The pointer arithmetic that does exist (`page_table_cb_wr_ptr += my_batch_idx * page_table_stick_size`) walks an **L1 CB** pointer, which the borrowed-DFB translation preserves verbatim.

**Urgency note.** None of these 15 bindings is the *silent-wrong* fast-path-cache hazard. `create_descriptor` never emplaces a raw `->address()`; every buffer arrives as an annotated `Buffer*` (auto-registered `BufferBinding`, patched on cache hits), and on top of that all 8 factories define `override_runtime_arguments`, which supersedes `resolve_bindings` and re-applies every address explicitly. The Metal 2.0 typed binding supersedes both mechanisms; this is **routine port work**, not a correctness fix.

### Other port work

- **TensorParameter relaxation:** **none** — the sheet's clearing value, on all 8 rows. Strict `TensorSpec` match. My independent reading of the three custom hashes is consistent with that (see *Team-only*).
- **TensorAccessor 3rd arg:** **none** — no site exists.
- **CB endpoints:** self-loop `(page_table_cb_index c_1, fill_cache — all configs)`, `(cb_batch_idx_id c_2, fill_cache — use_batch_idx_tensor)`, `(cb_valid_seq_len_id c_3, fill_cache — use_valid_seq_len)`. All other 26 `(CB, config)` instances are plain 1:1. No multi-binding flag, no dead-CB drop, no new conditional DFB.

---

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** **none.** All three faces were hunted (details in *Gate detail → CB endpoints*) and came back empty.

- **The aliased two-format intermediate DFB is load-bearing — do not split it.** In three of the four factories a *single* `CBDescriptor` carries **two** `CBFormatDescriptor`s, so two buffer indices alias one L1 allocation: `intermed0`/`intermed1` = `c_24`/`c_25` in `update_cache` (`paged_update_cache_program_factory.cpp:210-225`) and in tiled-fused (`paged_tiled_fused_update_cache_program_factory.cpp:223-238`), and `c_5`/`c_6` in RM-fused (`paged_row_major_fused_update_cache_program_factory.cpp:228-243`). The aliasing *is* the algorithm: compute untilizes a cache block and publishes it through index 0; the writer takes `cb_untilized_cache.get_read_ptr() + cache_tile_offset_B` (index 0), NoC-writes the new row into that L1 region **in place**, then publishes the *same* memory through index 1 via `cb_untilized_cache2.push_back(Wt)` for compute to re-tilize (`writer_update_cache:125,133`; `writer_paged_fused:137,145`; `writer_paged_row_major:139,147`). Port this as **one** `DataflowBufferSpec` with two format descriptors. Splitting it into two independent DFBs compiles, validates, and silently produces wrong numerics.

- **Runtime-selected DFB index across two DFBs with disjoint core ranges (fused factories) — the single most consequential shape in the op.** Three fused kernels pick which input CB to touch from a *runtime* arg: `reader_paged_fused_...:30-35`, `reader_paged_row_major_...:30-35`, `writer_paged_row_major_...:59-62` (`input_cb_id = is_input1 ? input1_cb_id : input2_cb_id`), and `compute/paged_fused_update_cache.cpp:21-26,36` does the same for `compute_kernel_hw_startup(in_cb, ...)` before branching at `:39-55` to two compile-time-parameterised `untilize<Wt, in1_cb, ...>` / `untilize<Wt, in2_cb, ...>` instantiations (both of which are compiled into every node's binary). Since a kernel cannot touch a DFB it has not bound, each of these `KernelSpec`s must bind **both** `src1` and `src2` — yet `src1`'s `CBDescriptor` covers only `input1_cores` and `src2`'s only `input2_cores`, the two are validated **disjoint** (`paged_fused_update_cache_device_operation.cpp:350-351`), and the `KernelSpec`s span `all_cores_bb`. So on any given node exactly one of the two bound DFBs actually exists. Resolve how the spec expresses this (both bound with per-node existence, or per-core-set `KernelSpec`s) *before* writing the fused specs. Nothing in Appendix A covers it, so it is not a gate — but it is not mechanical either. See *Questions* #1.

- **One RTA slot, two different tensors, selected by core (fused factories).** Reader RTA[2] and writer RTA[1] carry `cache_tensor1` on `cores1` and `cache_tensor2` on `cores2` (tiled `:438` vs `:483`, `:456-467` vs `:501-512`; RM `:435` vs `:481`, `:453-465` vs `:499-511`), and `patch_runtime_args` patches them the same way (`paged_fused_update_cache_device_operation.cpp:100-124`, via `patch_core(cores1[i], dst1_addr)` / `patch_core(cores2[i], dst2_addr)`). A `tensor::name` binding is per-`KernelSpec`, so one reader spec would need *two* `TensorParameter`s reaching one arg position by node. Same structural question as the bullet above, on the tensor-binding channel rather than the DFB channel.

- **One RTA slot, tensor-or-scalar by config — and only one of the three instances is hard.** Three factories overload an RTA slot between a `Buffer*` and a scalar, but they differ in kind:
  - **`fill_cache` — the hard one.** Writer RTA[4] is a `Buffer*` (batch-idx tensor) when `use_batch_idx_tensor`, and the *meaningful* scalar `operation_attributes.batch_idx_fallback` otherwise (`paged_fill_cache_program_factory.cpp:315-319`, patched at `:413`); writer RTA[6] is likewise a `Buffer*` or literal `0` (`:323-327`, patched at `:415`). In Metal 2.0 the binding channel and the named-scalar channel are different, so RTA[4] has to split into a config-conditional `TensorParameter` **plus** a named scalar arg. The kernel already keys off the same CTA (`writer_fill_cache_interleaved.cpp:56,100-116`), so the branch exists — it just has to move onto two channels.
  - **`update_cache` and both fused factories — the easy ones.** The scalar alternative is a literal `0` that the kernel never reads (the access is behind `if constexpr (use_index_tensor)` / `(is_paged_cache)`): `paged_update_cache_program_factory.cpp:408-418`, tiled `:440-450`, RM `:437-447`. These collapse to a conditional `TensorParameter` with no scalar counterpart.

- **Per-core runtime-arg count varies within one `KernelSpec` (fused factories).** Working cores get 8 reader args / 8 writer args (tiled) or 9 (RM, which appends `is_input1`) / 2 compute args, while every node in `unused_cores = all_cores_bb − all_cores` gets a **single** arg `{!has_work}` (tiled `:524-530`, RM `:523-529`) and early-returns on it (`reader:17-20`, `writer:18-21`, `compute:15-18`). A `runtime_arg_schema` is one schema for the whole `KernelSpec`, so decide how the short-arg nodes are expressed (supply the full named set with don't-care values, or narrow the `KernelSpec`'s core range) rather than discovering it at validation time. `unused_cores` is non-empty only when `input1_cores ∪ input2_cores` is not itself a rectangle.

- **`override_runtime_arguments` is the translation unit of real work here, and it is index-addressed.** All 8 factories carry one, so the target is `CustomProgramSpecFactoryConcept` and each must be translated into a `ProgramRunArgs` producer. There are only **four program bodies** behind the eight factories: `paged_update_cache_program_factory.cpp:457` (with `:522` delegating), `paged_fill_cache_program_factory.cpp:361` (`:420` delegating), and `paged_fused_update_cache_device_operation.cpp:54-125` — one shared `patch_runtime_args` template reached by all four fused hooks (`:395`, `:409`, `:428`, `:438`). Each body reaches args by **hard-coded positional index** (`reader_args[0]`, `writer_args[4]`, the `kReader*Arg` / `kWriter*Arg` constants at `paged_fused_update_cache_device_operation.cpp:28-37`) *and* reaches CBs by **hard-coded position in `desc.cbs`** (`kInputCbPos = 1` at `paged_update_cache_program_factory.cpp:485`; `kSrc1CbPos = 1`, `kSrc2CbPos = 2`, `kFirstOptionalCbPosTiled = 6`, `kFirstOptionalCbPosRowMajor = 5` at `paged_fused_update_cache_device_operation.cpp:42-45`). Those positional constants are exactly what named args and named DFB bindings replace, so the translation deletes them — but any mismatch is silent. The `kFirstOptionalCbPos*` pair in particular encodes the tiled-vs-RM CB-count difference by hand (tiled has an extra `intermed2` descriptor) and is walked with a post-increment (`:75-81`); read it together with the `if (use_index_tensor)` / `if (is_paged_cache)` push order in each factory before rewriting.

- **Kernels declare CB wrapper objects unconditionally for conditionally-allocated CBs.** e.g. `CircularBuffer cb_index(cb_index_id)` / `cb_page_table(page_table_cb_id)` at `reader_update_cache:56-57`, `writer_update_cache:61-62`, `reader_paged_fused:69-70`, `writer_paged_fused:66-67`, `reader_paged_row_major:69-70`, `writer_paged_row_major:73-74`, and `writer_fill_cache:92-93` — constructed even when `use_index_tensor` / `is_paged_cache` / `use_batch_idx_tensor` / `use_valid_seq_len` is false and the CB was never allocated. Harmless today (every *access* is behind `if constexpr`), but a Metal 2.0 binding is not a no-op the way an unused `CircularBuffer(id)` is: check whether the binding must be made conditional alongside the `DataflowBufferSpec`.

- **`share_cache` cross-core semaphore chain.** One plain `SemaphoreDescriptor` per factory (`paged_update_cache_program_factory.cpp:247`; tiled `:260`; RM `:256`), used only when `share_cache` is set: writer *i* signals reader *i+1* via `Semaphore<>::up(noc, send_core_x, send_core_y, 1)` (`writer_update_cache:164`; `writer_paged_fused:176`; `writer_paged_row_major:178`), awaited at `reader_update_cache:126-128` / `reader_paged_fused:152-154` / `reader_paged_row_major:154-156`, with the `send_core_x/y` **physical** coordinates baked into RTAs host-side (`worker_core_from_logical_core`, `paged_update_cache_program_factory.cpp:394`; tiled `:422,427`; RM `:419,424`). Ports as an ordinary `SemaphoreSpec` + `sem::name`; note the trailing `noc.async_atomic_barrier()` (`writer_update_cache:165`, `writer_paged_fused:182`, `writer_paged_row_major:184`) whose comment records a real Watcher NOC-idle race — keep it.

- **Two mesh-filtering idioms, both of which must survive unchanged.** `update_cache` and both fused families return an **empty `ProgramDescriptor`** for a coordinate outside `mesh_coords` (`paged_update_cache_program_factory.cpp:448-453`; tiled `:544-549`; RM `:547-552`), and their override hooks early-return on the same test (`paged_update_cache_program_factory.cpp:472-475`; `paged_fused_update_cache_device_operation.cpp:129-134`, called at `:402`, `:416`). `fill_cache` instead returns a **fully-built descriptor whose kernels early-exit** on a `noop` RTA (`paged_fill_cache_program_factory.cpp:33-40,348-359`, consumed at `reader_fill_cache_interleaved.cpp:21,23-25` and `writer_fill_cache_interleaved.cpp:77,80-82`) so the cache slot is still populated for that coordinate, and its override re-patches `noop` (`:399,408,414`). Two different mechanisms; neither is the port's to normalise.

- **Cross-op / shared kernels: none — and this is worth stating positively, because a grep by basename says otherwise.** All 11 kernels this op instantiates live under `device/kernels/` in this directory, and `grep -rl 'operations/experimental/paged_cache/device/kernels' ttnn/ tests/` shows the only instantiators are this op's own four factories. **However**, `ttnn/cpp/ttnn/operations/kv_cache/device/kernels/` contains three files with *identical basenames* — `dataflow/reader_update_cache_interleaved_start_id.cpp`, `dataflow/writer_update_cache_interleaved_start_id.cpp`, `compute/update_cache.cpp` — which are **separate private copies** that `kv_cache`'s own factories instantiate by their own path (`update_cache_multi_core_program_factory.cpp:291,319,374`; `fill_cache_multi_core_program_factory.cpp:197`). They are *not* shared files: editing them does nothing for this op, and editing this op's copies does nothing for `kv_cache`. A porter grepping by filename will land in the wrong directory. There is **no `_metal2` fork** beside any of this op's kernels and none is needed — no fork, no sunset list, no cross-op coordination cost.

- **Intra-op kernel sharing is real but self-resolving.** Each `*MeshWorkloadFactory` delegates its `create_descriptor` to its single-device sibling, so the two members of each pair bind the *same* kernel sources. That is technically the shared-kernel caution's **intra-op** shape — but because the pair shares one program body and one `override_runtime_arguments`, converting the body converts both factories in the same change, so no fork is required. The practical shape of the port is **four program bodies for eight factories**. What this does mean: do not port one member of a pair without the other.

- **RTA varargs: none.** No kernel reads an argument at a loop-variable or data-selected index. Every read is either at a constant index (`reader_fill_cache:18-21`, `writer_fill_cache:71-78`, `reader_update_cache:16-21`, `writer_update_cache:17-23`) or a **fixed run** of `rt_args_idx++` at the top of the kernel (the six fused kernels, 8–9 reads each), which the recipe classifies as legacy positional plumbing that dissolves into named args — not a vararg. No `get_common_arg_val` anywhere. **No CTA varargs either:** every `get_compile_time_arg_val` index is a literal, and the only computed compile-time-arg offsets are `TensorAccessorArgs<N>()` chained through `constexpr next_compile_time_args_offset()` (`reader_update_cache:48-50`, `writer_fill_cache:84-88`, and the fused equivalents), which is a fixed set at constexpr offsets — so those get names too.

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
| the same 3 compute kernels | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` (+ `untilize_helpers.inl`) | 2 — official shared kernel library | ✓ |
| the same 3 compute kernels | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` (+ `tilize_helpers.inl`) | 2 | ✓ |

**Per-call detail** — omitted for the bucket-1 donors (all ✓). The two bucket-2 donors are the only non-firmware escape, and both are ✓, so this is recorded for completeness rather than as an action:

| Donor | Function called | Signature shape | Status |
|---|---|---|---|
| `kernel_lib/untilize_helpers.hpp:145-154` | `compute_kernel_lib::untilize<block_width_tiles, input_dfb, output_dfb, …>(uint32_t num_blocks)` | CB handles are `uint32_t` **NTTPs** — the `uint32_t cb_id` row; `dfb::name`'s constexpr cast covers template-parameter position. Parameters are already *named* `input_dfb` / `output_dfb`, and the impl builds `DataflowBuffer in_dfb(input_dfb)` / `out_dfb(output_dfb)` internally (`untilize_helpers.inl:199-200`), i.e. the donor is fully DFB-aware. | ✓ OK — no donor-side change, no fork |
| `kernel_lib/tilize_helpers.hpp:153-163` | `compute_kernel_lib::tilize<block_width_tiles, input_dfb, output_dfb, …>(uint32_t num_blocks, std::optional<uint32_t>)` | same (`tilize_helpers.inl:149-150`) | ✓ OK |

**Borrowed kernel files (file-path instantiation): none.** All 11 kernel `.cpp` files the four factories instantiate are owned by this op directory, and no other op instantiates any of them. No `_metal2` fork exists beside any of them (checked locationally — `ls` of `device/kernels/*/`), and none is needed. **There is no sunset list and no cross-op coordination cost for this port.** See *Heads-ups* for the same-basename trap in `ttnn/cpp/ttnn/operations/kv_cache/device/kernels/` and for the intra-op factory-pair sharing.

### Relaxation candidates — **FALLIBLE, candidates to verify; default strict**

The sheet now reads `TensorParameter relaxation == none` on all 8 rows, so the ops team's answer is in and there is nothing to route. Recorded only because the recipe asks for what a custom hash reveals, and because it happens to *support* rather than challenge the strict decision.

All three `compute_program_hash` overrides pass **`tensor_args` in full** to `operation::hash_operation<...>` (`paged_update_cache_device_operation.cpp:327-336`; `paged_fill_cache_device_operation.cpp:214-215`; `paged_fused_update_cache_device_operation.cpp:383-388`). What each one *narrows* relative to the default is only **`operation_attributes`** members, never tensor properties:

| DeviceOperation | Attributes excluded from the key | Where the excluded value is re-supplied on a cache hit |
|---|---|---|
| `PagedUpdateCacheDeviceOperation` | `update_idxs` (a `std::vector<uint32_t>` of runtime positions), `batch_offset` (validated `== 0`) | `update_idxs` → `compute_update_cache_offsets()` re-derives `cache_start_id` / `tile_update_offset_B` and `override_runtime_arguments` re-patches them (`paged_update_cache_program_factory.cpp:498,511-515`) |
| `PagedFillCacheDeviceOperation` | `batch_idx_fallback`, `noop` | both re-patched (`paged_fill_cache_program_factory.cpp:395-399,413-414`) |
| `PagedFusedUpdateCacheDeviceOperation` | `update_idxs`, `batch_offset` (validated `== 0` at `:332`) | re-patched via `compute_{tiled,row_major}_fused_offsets` + `patch_runtime_args` (`paged_fused_update_cache_device_operation.cpp:100-124`) |

All three hashes *include* every attribute that reaches a compile-time arg — `compute_kernel_config`, `share_cache`, `mesh_coords`, `block_size_override`, `num_kv_heads_override`, `cache_position_modulo`, and `program_factory.index()`. Since no tensor property is dropped from the key, two calls that cache-hit necessarily carry identical `TensorSpec`s for every tensor argument — which is what a strict `TensorParameter` match requires. **No relaxation candidate is visible in these hashes.** (Fallible: I did not audit `hash_operation`'s reflection over `tensor_args`, and this is not the ops-team analysis.)

### TTNN factory analysis (sheet-derived facts with `file:line` evidence)

- **Concept (current):** `descriptor`, all 8 factories. Evidence: the `create_descriptor` declarations listed in the cross-check table. **Not** `WorkloadDescriptor` despite four `*MeshWorkloadFactory` names.
- **Target concept:** `CustomProgramSpecFactoryConcept` — driven by `Override runtime args method? == yes`, agreeing with the sheet's own `Porting Target` cell.
- **Op-owned tensors:** none. No `WorkloadDescriptor`, no `buffers` vector under `device/`.
- **MeshWorkload need:** none genuine. The mesh variants exist purely for **per-coordinate filtering**, implemented two different ways (see the mesh-idiom bullet under *Heads-ups*). Both idioms have to survive the port unchanged.
- **Pybind `create_descriptor`:** none. Nothing for the port to delete, so no user-visible API change from this port.
- **Other risky pybind:** none. `paged_cache_nanobind.cpp` binds only the three public entry points through `ttnn::bind_function` (`:48`, `:89`, `:134`); no device-op or factory internals are exposed.
- **Custom hash:** present on all 3 DOps (sites above). The port leaves each exactly as it is.
- **`get_dynamic_runtime_args`:** absent.
- **`override_runtime_arguments`:** present on all 8 factories; the four fused ones funnel into one shared `patch_runtime_args` template (`paged_fused_update_cache_device_operation.cpp:54-125`) and the two mesh variants of `update_cache` / `fill_cache` each delegate to their single-device sibling (`paged_update_cache_program_factory.cpp:522-530`; `paged_fill_cache_program_factory.cpp:420-428`). **One translation per program body: four bodies, eight factories.**
- **Gate conjuncts confirmed absent:** a non-`none` `TensorParameter relaxation`, `get_dynamic_runtime_args`, genuine multi-program `WorkloadDescriptor`.

---

## Misc anomalies  *(team-only, non-gating, not porter-actionable)*

Noticed while reading; these route to the ops team and the port must not act on them. The previous audit's first anomaly — six dead `DataFormat` locals — was fixed by `47a266001ad` and is dropped from this list.

1. **`log_base_2_of_page_size` is a dead CTA that is always 0.** Read at `reader_update_cache_interleaved_start_id.cpp:29` (CTA 6), `reader_paged_fused_...:43` (CTA 8), `reader_paged_row_major_...:43` (CTA 8) and never used. Host-side it is a local initialised to `0` and never assigned: `paged_update_cache_program_factory.cpp:116`, `paged_tiled_fused_update_cache_program_factory.cpp:111`, `paged_row_major_fused_update_cache_program_factory.cpp:111` (the last is even declared `const … = 0`). Three CTA slots that could be removed from all three readers.
2. **`log2_page_table_stick_size` is a dead CTA in four kernels, and in `fill_cache` it is a *computed* dead value with a latent precision bug.** Read and unused at `reader_update_cache_...:38` (CTA 13), `reader_paged_fused_...:52` (CTA 15), `reader_paged_row_major_...:52` (CTA 15), `writer_fill_cache_interleaved.cpp:47` (CTA 6). In the first three the host value is a hardcoded `0` (`paged_update_cache_program_factory.cpp:131`, tiled `:129`, RM `:129`). In `fill_cache` it is `std::log2(page_table_stick_size_B)` truncated to `uint32_t` (`paged_fill_cache_program_factory.cpp:119`), and the adjacent `TT_FATAL` only requires `page_table_stick_size_B % 32 == 0` (`:116-118`), **not** a power of two — so for e.g. 96 bytes the value would be 6, not a valid shift. Harmless *only* because no kernel reads it. If anyone ever wires it up, the assertion has to be tightened first.
3. **`PagedFillCacheParams::noop` is never set `true` by any caller.** Declared at `paged_fill_cache_device_operation_types.hpp:16`; the sole public entry point hardcodes `.noop = false` (`paged_fill_cache_device_operation.cpp:237`). The functional noop path is driven entirely by the mesh-coordinate test inside `paged_fill_cache_noop` (`paged_fill_cache_program_factory.cpp:33-40`), which ORs in this attribute as a second, unused source. Correctly excluded from the hash; the field itself is dead API surface.
4. **`paged_row_major_fused_update_cache.cpp` carries three dead inputs.** CTAs 0 and 1 (`in1_cb`, `in2_cb`) and RTA[1] (`is_input1`) exist only to compute `in_cb` at `:23-26`, which is marked `[[maybe_unused]]` and never used — the RM compute kernel genuinely does not touch the input CBs (the RM writer consumes them directly). The author already acknowledged this with the attribute; the args could be dropped from `compute_kernel_args` (`paged_row_major_fused_update_cache_program_factory.cpp:350-351`) and from the per-core `compute_desc.emplace_runtime_args` (`:468-473`, `:514-519`).
5. **Two unbalanced FIFO `wait_front`s in the row-major fused writer.** `cb_index.wait_front(1)` at `writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp:84` and `cb_page_table.wait_front(num_pages_to_read)` at `:95` have **no** matching `pop_front`. Both sibling writers do pop: `writer_paged_fused_...:112` (`cb_page_table`) and `:122` (`cb_index`), and `writer_update_cache_...:100,110` likewise. Benign today because each CB is filled once per dispatch and never re-waited, but it is an asymmetry against two siblings that do it the other way, and it is the kind of thing a later change to the loop structure turns into a hang.
6. **`max_blocks_per_seq` is read but unused in all six fused/update dataflow kernels.** `reader_update_cache_...:37`, `writer_update_cache_...:40`, `reader_paged_fused_...:51`, `writer_paged_fused_...:46`, `reader_paged_row_major_...:51`, `writer_paged_row_major_...:48`. The host computes it from `page_table.padded_shape()[1]` and it is genuinely load-bearing in `validate_on_program_cache_miss` as a bound, but no kernel range-checks `virtual_block_id` against it before indexing `page_table_ptr[virtual_block_id]`. Not a port concern; worth the ops team's attention as a missing on-device bound check.
7. **`fill_cache`'s `batch_idx_stick_size_B` default is a hardcoded `4` with a comment admitting the assumption** (`paged_fill_cache_program_factory.cpp:128`). Only reached when `use_batch_idx_tensor` is false, in which case the CB is not allocated and the CTA is unused, so it is inert — but the constant sits next to code that does the right thing (`tensor.element_size()`, `:134`) in the live branch. `valid_seq_len_stick_size_B` has the same shape (`:156` default `4`, `:158` `element_size()`).
8. **`PagedFillCacheDeviceOperation::validate_on_program_cache_miss` has a suspicious `||` chain** (`paged_fill_cache_device_operation.cpp:32-36`): the predicate mixes `input_tensor.dtype()` for the first two alternatives and `cache_tensor.dtype()` for the last two, so a `BFLOAT8_B` *cache* satisfies the check regardless of the input dtype (and the message says "input tensor"). Looks like a copy-paste slip rather than intent; the port does not touch validation.

---

## Per-DeviceOperation attribution

Findings are structurally common across the three DeviceOperations; the table records where they differ.

| Field | `PagedUpdateCacheDeviceOperation` | `PagedFillCacheDeviceOperation` | `PagedFusedUpdateCacheDeviceOperation` |
|---|---|---|---|
| Factories | 2 (1 program body) | 2 (1 program body) | 4 (2 program bodies: tiled, row-major) |
| **Overall** | **GREEN** | **GREEN** | **GREEN** |
| Device 2.0 | GREEN (3 kernels) | GREEN (2 kernels) | GREEN (6 kernels) |
| Feature compatibility | GREEN (all N/A) | GREEN (all N/A) | GREEN (all N/A) |
| Sheet `Is able to port?` | `yes` (both rows) | `yes` (both rows) | `yes` (all four rows) |
| Concept / target | `descriptor` → `CustomProgramSpecFactoryConcept` | same | same |
| Custom hash | yes (`…_device_operation.cpp:313`) | yes (`…_device_operation.cpp:207`) | yes (`…_device_operation.cpp:371`) |
| Offset base pointers | GREEN (3 clean bases) | GREEN (5 clean bases) | GREEN (4 clean bases) |
| Tensor bindings | 3 Case 1, 1 clean | 5 Case 1 (2 with an overloaded tensor-or-scalar slot) | 2 Case 1, 2 clean, 2 config-split (Case 1 / clean) |
| TensorAccessor 3rd arg | N/A (no site) | N/A (no site) | N/A (no site) |
| CB endpoints | 8 `(CB, config)` — all 1:1 | 4 `(CB, config)` — 1 × 1:1, **3 self-loop** | 9 tiled + 8 RM `(CB, config)` — all 1:1 |
| Semaphores | 1 (`share_cache` chain) | none | 1 per factory (`share_cache` chain) |
| Borrowed-memory CBs | 1 (`c_1` input shard) | none | tiled: up to 4 (`c_1`, `c_2`, `c_3`†, `c_4`†) · RM: same († only when the tensor is L1-sharded) |
| Distinctive porting shapes | aliased two-format intermediate DFB; 2 easy tensor-or-`0` slots | tensor-or-*meaningful*-scalar overloaded RTA slots; noop-program mesh idiom | runtime-selected DFB index over disjoint core ranges; one RTA slot ↔ two cache tensors by core; variable per-core RTA count; aliased two-format intermediate DFB |
| RTA / CTA varargs | none | none | none |
| Out-of-directory coupling | ✓ clean | ✓ clean | ✓ clean |

---

## Questions for the user

1. **How should the fused factories' runtime-selected input DFB be expressed in a `ProgramSpec`?** (`reader_paged_fused_...:30-35` and siblings — see *Heads-ups*.) Three kernels choose between `src1` and `src2` from a runtime arg, so both must be bound, yet each DFB is allocated over only one of two validated-disjoint core subsets while the `KernelSpec` spans their bounding box. Nothing in Appendix A covers it, so I did not gate — but a decision here (bind both and rely on per-node existence, vs. split into per-core-set `KernelSpec`s, vs. something else) belongs upstream of the port rather than inside it, and it will shape the fused specs more than any other single choice. This is the same question the previous audit raised and it is the one open design item on an otherwise-GREEN op.

2. **Is a `Diego validation == no` row expected to be portable?** All 8 rows read `Is able to port? == yes` with `Diego validation == no`. `Diego validation` is not among the columns the audit's blocking table names, and `Is able to port?` is derived, so I read the verdict and did not vet it. Flagging it only in case the two cells are meant to move together and this op's `yes` landed ahead of a validation pass — worth a one-line confirmation from the readiness-sheet owner before the port is scheduled.

---

## Recipe notes

1. **The `get_dataformat` sanctioned-list question is now moot for this op but still open in the recipe.** The previous audit REDed this op solely on six `get_dataformat(cb_id)` calls, on the grounds that the sanctioned list (`get_tile_size(cb_id)`, `get_local_cb_interface(cb_id)`) "is the whole test". The op-side Device 2.0 fix landed, so the question no longer bears on this verdict — but the recipe text is unchanged, and the next op that uses `get_dataformat` (or `get_tile_hw`) will hit the same fork. The three sit side by side in one `#ifdef DATA_FORMATS_DEFINED` block in `tt_metal/hw/inc/api/dataflow/circular_buffer.h:113-115`, each a one-line forward to `::<name>(cb_id_)`, and the grounding argument offered for `get_tile_size` (the migration guide's migrated example keeps it) is silent on the other two only because that example never needs a data format. Suggestion unchanged: either add `get_dataformat` and `get_tile_hw` to the sanctioned list, or add one sentence saying the omission is deliberate.

2. **There is no guidance for re-auditing an op that already has a `METAL2_PREPORT_AUDIT.md` in its directory.** This audit found one and read it — deliberately, and it was the single highest-leverage thing I did: `git diff <prior-basis>..HEAD` over the op directory showed the only code change was the Device 2.0 commit, which told me exactly which findings could be carried and which had to be re-derived (all kernel line numbers, the whole Device 2.0 scan). I still re-ran every gate from the code, and I found two small errors in the prior report (a `TensorAccessor` count of 16 for 17 listed sites; a binding roll-up of "11 bindings" that didn't add up against its own per-DOp tables — the real count is 15). Suggestion: say explicitly that a prior audit in the op directory is **evidence about what changed, not a source of findings** — read it, diff the tree against its basis, re-derive every gate, and correct it where it is wrong. Without that sentence an auditor has to choose between ignoring a highly relevant document and treating it as authority, and neither is right.

3. **The status-summary template has a stale feature row.** The `METAL2_PREPORT_AUDIT.md` template includes `| *Feature Support* — Variadic-CTA | Ok / Unsupported |`, but Appendix A has no Variadic-CTA entry (its three entries are GlobalCircularBuffer, `address_offset`, GlobalSemaphore) — and *RTA varargs* explicitly says CTA varargs port onto `KernelAdvancedOptions::compile_time_varargs` and do **not** gate. The row looks left over from an earlier Appendix A. I replaced it with one row per current Appendix A entry.

4. **"Multiple device-operations in one op directory" and the output contract can pull apart.** The bundling test is *shared factories or kernels* → one report; independent → *"audit each separately"*. Here the three DeviceOperations share **no** kernels and **no** factories, so the test says separate — but *Output: the two documents* specifies one `METAL2_PREPORT_AUDIT.md` per **op directory**, and the readiness sheet keys on `Op == experimental/paged_cache`, so three files would have nowhere to live and no consumer. I produced one bundled report with full per-DeviceOperation attribution. Suggestion: say what to do when the shared-code test says "separate" but the directory (and the sheet's op granularity) says "one" — I think one file with mandatory per-DOp attribution is right, but the recipe currently implies three.

5. **Provenance is assumed to be recoverable from the working checkout.** The command in *Output: the two documents* is to be run "from the checkout root", with the fallback being "the docs aren't from a tracked doc-branch checkout — record that instead". Neither case fit: the recipe was handed to me as a standalone file, the working repo has no `metal_2.0` doc tree, and a sibling checkout has both the tree *and* a byte-identical copy of the recipe (verified with `diff -q`). I pinned the hash from the sibling and said so. Suggestion: allow "run it wherever the doc tree lives, and name that path" — a hash from a verified-identical checkout is strictly better than "can't be pinned".

6. **A half-sentence about `Buffer*` auto-patching vs. `override_runtime_arguments` precedence would save re-reading the override bodies.** *TensorParameter analysis* → *Detection — host side* → `Buffer*`-binding form says the framework auto-registers these and patches them on cache hits, which is correct — but every factory here *also* defines `override_runtime_arguments`, whose own comments say it "supersedes `resolve_bindings`, so all addresses are ours". Both mechanisms are live in the same op and the recipe discusses them separately, so I had to read all four override bodies to confirm there was no conflict. Noting that a `CustomProgramSpecFactoryConcept` op's override takes precedence over `Buffer*` auto-patching (making the "correct-on-cache-hit today" reassurance doubly true there) would cover it.

7. **The readiness sheet has grown four columns the recipe doesn't mention, one of which looks audit-adjacent.** Today's header carries `Provisional relaxation finding (Edwin)`, `Why secretly SPMD?`, `Pointer patching perf issue?`, and `Formerly custom hashed?` (all blank for this op). The recipe's rule to resolve columns by header name handled them fine, so this is not friction — but `Pointer patching perf issue?` sounds like it may relate to the *TensorParameter analysis* subject's fast-path-cache binding-injection discussion. If it does, `ttnn_op_porting_readiness.md`'s column legend is the place to say so; if it doesn't, a line saying which columns the audit deliberately ignores would stop the next auditor wondering.

8. **The shared-kernel caution's "intra-op" rung fires on the delegating-mesh-sibling pattern, which is common and benign.** Every one of this op's four `*MeshWorkloadFactory` types binds the same kernel sources as its single-device sibling, which matches *port_patterns.md* → *Caution: Porting a shared kernel* → **Intra-op** ("two factories of *your own* op bind the same kernel and you are porting one of them"). But because the pair shares one `create_descriptor` body and one `override_runtime_arguments`, converting the body converts both factories in the same change and no fork is needed. The entry as written sends the porter to the fork rungs. Suggestion: add a guard clause — a factory pair that shares one program body ports as one unit, so the intra-op rung applies only where the two factories build *different* bodies from the same kernel source.
