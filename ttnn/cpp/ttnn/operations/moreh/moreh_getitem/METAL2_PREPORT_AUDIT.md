# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/moreh/moreh_getitem`

One DeviceOperation, two program factories (single combined report):

- **`MorehGetItemOperation`** (`device/moreh_getitem_device_operation.{hpp,cpp}`)
  - `MorehGetItemRmFactory` (`device/moreh_getitem_rm_factory.cpp`)
  - `MorehGetItemTilizedFactory` (`device/moreh_getitem_tilized_factory.cpp`)

Factory selection is by input layout (`moreh_getitem_device_operation.cpp:69-77`): `ROW_MAJOR` → Rm, otherwise → Tilized.

**Three program shapes, not two — this matters throughout.** `MorehGetItemTilizedFactory` branches internally on `is_w_index_exist` (`moreh_getitem_tilized_factory.cpp:74-79, 87`) and emits **two entirely different programs** with different kernels and different CB sets. Per the recipe's *classify per instantiation* rule, they are audited as separate configs:

| Shape | Condition | Reader | Writer |
|---|---|---|---|
| **RM** | `MorehGetItemRmFactory` | `moreh_getitem_kernels/reader_moreh_getitem.cpp` | `moreh_getitem_kernels/writer_moreh_getitem.cpp` |
| **Tilized-W** | Tilized, `is_w_index_exist == true` | `moreh_getitem_tilized_kernels/reader_moreh_getitem_tilize_w.cpp` | `…/writer_moreh_getitem_tilize_w.cpp` |
| **Tilized-noW** | Tilized, `is_w_index_exist == false` | `moreh_getitem_tilized_kernels/reader_moreh_getitem_tilize.cpp` | `…/writer_moreh_getitem_tilize.cpp` |

Each shape further sub-configures on the index layout via a `ROW_MAJOR_INDEX` / `TILIZE_INDEX` define (tilized only, `moreh_getitem_tilized_factory.cpp:183-187, 433-437`).

**Kernels:** 6 `.cpp` files plus one shared header (`moreh_getitem_tilized_kernels/common.hpp`). All 6 are referenced; there are no unreferenced kernel files. The op has **no compute kernels** — it is pure data movement (reader → CB → writer).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `b72c35b810e 2026-08-04 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/moreh/moreh_getitem` |
| **Overall** | **GREEN — all five gates cleared. Brief issued.** |
| **DOps / Factories** | `MorehGetItemOperation` → `MorehGetItemRmFactory`, `MorehGetItemTilizedFactory` (the latter in two configs) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 6 kernels Device 2.0 native; zero holdovers |
| *Prereqs* — Cross-op escapes | **Ok** — *none*. Kernels include only `api/*` (LLK/HAL) and the op's own `common.hpp` |
| *Feature Support* — overall | **GREEN** — all four Appendix A entries `N/A` |
| *Feature Support* — Variadic-CTA | **Ok** — op-level cue fires (`std::vector<Tensor>`), kernel-level decider does **not**; see Gate detail |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** — both factory rows `yes`; cross-check clean |
| *TTNN Readiness* — Concept (current) | `descriptor` — sheet and code agree |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A — not a `WorkloadDescriptor` op |
| *TTNN Readiness* — Is safe to port? | **Yes** (both rows) |
| *TTNN Readiness* — Custom hash | **No** — no `compute_program_hash` anywhere in the op |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** |
| *TTNN Readiness* — `override_runtime_arguments` | **No** |
| *TTNN Readiness* — Pybind `create_descriptor` | **No** — `moreh_getitem_nanobind.cpp:18` binds only `ttnn::moreh_getitem` |
| *TTNN Readiness* — Op-owned tensors | **No** |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (no op-owned tensors) |
| *Port work* — Offset base pointer | **none** — no `->address()` anywhere; kernel-side `.offset_bytes` is page-relative NoC addressing, not a host fold |
| *Port work* — Tensor bindings (per binding) | **Case 1** × up to 7 (`input`, `output`, `index0..4`) — the index bindings are **optional/absent-capable** |
| *Port work* — TensorParameter relaxation | **`dynamic_tensor_shape` proposed** (RM factory) — **but its safety check is unrunnable; see Question 1** |
| *Port work* — TensorAccessor 3rd arg | **Class 1** × 7 sites, **RM factory only** → drop + relaxation (see Question 1). Tilized: none |
| *Port work* — CB endpoints | **`c_16` dead-CB drop in all three shapes** · self-loops · 1 config-scoped dead-CB **question** |

**CB endpoints** are dispositions, not gates. Recorded per `(CB, config)` below.

---

## Result

**GREEN → brief issued.** All five gates cleared, for both factories and all three program shapes. No subset scoping needed.

Two findings dominate the port work, and neither gates:

1. **`c_16` — the "output CB" — is allocated in all three program shapes and referenced by no kernel at all.** Confirmed dead. Every writer instead drains `c_0`, the input CB. Three allocations to drop.
2. **The `TensorAccessor` 3rd argument is Class 1 (dynamic page size) on the RM factory** — 7 sites, exactly as the triage doc predicts. The mechanical action is "drop the override and set `dynamic_tensor_shape`," but **the recipe's safety check for that relaxation cannot be run on this op** (it presupposes a custom hash; there is none), and the two candidate readings differ in cache semantics. Raised as **Question 1** — the porter must not choose unilaterally.

Beyond those, the port's real design work is the **optional/absent index-tensor bindings**: the op declares up to five index tensors of which only some exist per call, delivered today as `nullptr` `Buffer*` RTAs that the framework deliberately lowers to a literal `0u` with no binding. Metal 2.0's typed binding channel needs an equivalent story. Not an Appendix A feature, so not a gate — but it is the largest single item in the port and is flagged prominently to the porter.

---

## Gate detail

### TTNN factory concept (`Is able to port?`) — **GREEN**

Readiness rows as supplied by the user (the claude.ai Google Drive connector is unavailable in this session — same limitation as recorded in the `moreh_sum` audit; see *Recipe notes* §1):

| Device operation | Factory (variant) | Concept | Is safe to port | Is able to port? |
|---|---|---|---|---|
| `MorehGetItemOperation` | `MorehGetItemRmFactory` | `descriptor` | yes | **yes** |
| `MorehGetItemOperation` | `MorehGetItemTilizedFactory` | `descriptor` | yes | **yes** |

**Gate cleared for both factories.** `Is safe to port? == yes` is the sheet owner's correctness call and is taken as given.

Conjuncts not present in the supplied columns are **entailed** by `Is able to port? == yes` (which can only hold when `Custom hash`, `get_dynamic_runtime_args`, `override_runtime_arguments` and `Pybind descriptor` are all `no` and `Concept == descriptor`) and each is **independently confirmed in the code**:

| Conjunct | Code evidence | Verdict |
|---|---|---|
| `Concept` | `moreh_getitem_device_operation.hpp:34,41` — two `static ProgramDescriptor create_descriptor(...)` | `descriptor` ✓ |
| `Custom hash` | no `compute_program_hash` in the op directory | `no` ✓ |
| `Runtime-args update (get_dynamic_runtime_args)` | hook absent from `MorehGetItemOperation` (`…hpp:49-53`) | `no` ✓ |
| `Override runtime args method?` | no `override_runtime_arguments` in either factory | `no` ✓ |
| `Pybind descriptor` | `moreh_getitem_nanobind.cpp:18` binds `"moreh_getitem"` only | `no` ✓ |
| `Op-owned tensors?` | `descriptor` concept; no `buffers` vector | `no` ✓ (invariant holds) |
| **Factory-set match** | two sheet rows ↔ two code factories, one-to-one | ✓ |

**No code-vs-sheet disagreement on any column.** Cross-column invariants hold.

**Note — the sheet's factory granularity is coarser than the code's.** `MorehGetItemTilizedFactory` emits two structurally distinct programs (see the identifying section). The sheet carries one row for it, which is correct at the `ProgramFactory` level the sheet models, but a reader tracking port progress should know that clearing "the Tilized factory" means clearing two program shapes. Not a defect; recorded so the accounting is not surprising.

Supporting evidence for `Is safe to port?` (corroboration only): the op has **no un-annotated pointer arguments**. Every buffer reaches a kernel via `emplace_runtime_args` with a `Buffer*`, which the framework auto-registers as a `BufferBinding` — the sanctioned annotation, the opposite of the smuggled-pointer shape.

### Device 2.0 (every kernel used) — **GREEN**

All 6 kernels are structurally Device 2.0: `Noc` objects for every transfer, `DataflowBuffer` objects for every FIFO operation and every L1 pointer fetch, `TensorAccessor` for every address generation. A scan of all 6 files for `InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedPow2AddrGen*`, free-function `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front`, **free-function** `get_write_ptr` / `get_read_ptr` (pattern anchored to exclude method calls), `noc_async_read` / `noc_async_write`, `get_noc_addr_from_bank_id`, `CircularBuffer`, and `evil_set_*_ptr` returns **zero violations**.

Every pointer fetch in the op is a wrapper **method** — e.g. `dfb_in1_obj.get_write_ptr()` (`reader_moreh_getitem.cpp:162`), `dfb_out0_obj.get_read_ptr()` (`writer_moreh_getitem_tilize_w.cpp:49`). No `get_tile_size(cb_id)` sanctioned-free-function use arises (the op sizes its transfers from runtime args, not tile metadata).

**No donor code to assess.** Unusually for this family, the kernels include nothing outside `api/*` (LLK/HAL, class 1 — no concern) and the op's own `moreh_getitem_tilized_kernels/common.hpp`. There is no `moreh_common.hpp` or `kernel_lib` dependency, so the donor-shape question does not arise at all.

### Feature compatibility — **GREEN** (all entries `N/A`)

A scan of the whole op directory for every Appendix A recognition signal — `GlobalCircularBuffer`, `CreateGlobalCircularBuffer`, the `global_circular_buffer` field, `address_offset`, `set_address_offset`, `UpdateDynamicCircularBufferAddress`, `cb_descriptor_from_sharded_tensor`, `remote_index` / `remote_cb`, `GlobalSemaphore`, `CreateGlobalSemaphore`, `set_globally_allocated_address` — returns **zero hits**.

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | All CBDescriptors across both factories are plain. |
| CBDescriptor `address_offset` (non-zero) | N/A | Field never set. **Not to be confused with** the kernel-side `.offset_bytes` page-address field used throughout the tilized kernels — that is an unrelated NoC addressing feature, and the entry's own false-positive guard excludes it. |
| GlobalSemaphore | N/A | The op uses **no semaphores at all**. |
| Variable-count compile-time arguments (CTA varargs) | **N/A** | The op-level cue fires; the kernel-level decider does not. See below. |

#### CTA varargs — the op-level cue fires, the decider does not

This is the one Appendix A entry that needed real work, because the op-level signal is present in its textbook form: `tensor_args_t` carries **`const std::vector<Tensor>& index_tensors`** (`moreh_getitem_device_operation.hpp:26`) — a variable-count container of input tensors. Per the entry, that is "a prompt to inspect the kernel, not a verdict on its own."

Inspecting the kernels resolves it to **N/A**, on three independent grounds:

1. **The host flattens the variable list into a fixed 5-slot array before any CTA is emitted.** Both factories declare `IndexInfo index_info[5]` (`moreh_getitem_rm_factory.cpp:67`; `…tilized_factory.cpp:89, 356`), scatter the user's index tensors into it by dimension, and then append accessor args for **all five slots unconditionally** — `for (auto& dim : index_info) { dim.args.append_to(reader_compile_time_args); }` (`rm_factory.cpp:146-148`; `tilized_factory.cpp:191-193, 441-443`). The number of appended blocks is 1 (input) + 5 (index), always, regardless of how many index tensors the caller passed.
2. **Every kernel reads its accessor args at a `constexpr` offset.** The six blocks are chained through `next_compile_time_args_offset()` (`reader_moreh_getitem.cpp:66-71`, `reader_moreh_getitem_tilize.cpp:77-82`, `reader_moreh_getitem_tilize_w.cpp:78-83`). Computed, but compile-time — explicitly the entry's false-positive guard.
3. **No kernel calls `get_compile_time_arg_val` at all.** Verified across all 6 files: zero occurrences. There is no CTA loop to be runtime-indexed.

The per-index metadata that *does* vary — `isN_defined`, `indexN_stick_size`, the addresses — all rides **runtime** args, which is the entry's named non-firing case ("a variable-count input list whose per-input data rides RTAs … is **not** CTA varargs"). `N/A`, with no residual uncertainty.

### TensorAccessor 3rd argument — **Class 1, RM factory only** (PORT WORK, not a gate)

**7 sites, all in the RM shape**, all passing an explicit page size:

| Site | Accessor | 3rd arg | Host expression |
|---|---|---|---|
| `reader_moreh_getitem.cpp:75` | input | `stick_size` | `input_unit_size = input_5d_shape[-1] * input_5d.element_size()` (`rm_factory.cpp:81`) |
| `reader_moreh_getitem.cpp:79-83` | `index0`–`index4` | `indexN_stick_size` | `index.padded_shape()[-1] * index.element_size()` (`rm_factory.cpp:76`) |
| `writer_moreh_getitem.cpp:27` | output | `output_stick_size` | `output_unit_size = input_unit_size` (`rm_factory.cpp:82`) |

The **tilized shapes pass no 3rd argument** — all 13 accessor constructions there are 2-arg (`reader_moreh_getitem_tilize.cpp:84-90`, `…_tilize_w.cpp:85-91`, both tilized writers). The subject therefore scopes cleanly to the RM factory.

Classified against the two questions, independently of the table:

1. **Sharded or interleaved?** Interleaved row-major is the design point — the op's whole RM path addresses by stick (`{.page_id = noc_id}` with a per-row page). No sharding-specific path exists in either factory. Interleaved ⇒ the accessor realigns the passed value, so only a *wrong-magnitude* value could mis-address.
2. **Correct or wrong magnitude?** All three expressions resolve to `last_dim × element_size` — exactly the true logical page of a row-major stick, i.e. `buffer->page_size()`. **Correct magnitude.** No Class 3, Class 4, or Special case; nothing gates.

The value **genuinely varies with row width across shapes** that would otherwise reuse one compiled program — the kernel comments say so outright (`reader_moreh_getitem.cpp:73-74, 77-78`: *"Third argument page_size from runtime args overrides TensorAccessorArgs::AlignedPageSize, which may be stale on program cache hits"*) — and the op carries **all five input and all five output extents as runtime args**, which is precisely the "no coarseness hazard" property the triage doc names as qualifying an op for the all-dims relaxation. → **Class 1**, agreeing with the triage doc's row for `moreh_getitem`.

**But the port action cannot be settled here — see Question 1.** Class 1's action is "set `dynamic_tensor_shape` and drop the manual override," cross-referenced to [TensorParameter relaxations]. That subject's safety check is *"confirm the existing custom hash's logic matches the listed relaxation"* — and **this op has no custom hash**, so the check has nothing to run against. The two readings that remain differ in cache semantics, so the choice is a real decision, not a formality. Routed to the ops team / relaxation-design owner.

### Offset base pointers — **GREEN**

The op contains **no `->address()` or `.address()` expression at all** — zero hits across the directory. Every tensor base rides the descriptor API's **`Buffer*`-binding form**: the factories pass `input.buffer()`, `output.buffer()` and `index_info[0..4].buffer` directly into `emplace_runtime_args` (`rm_factory.cpp:187-250`; `tilized_factory.cpp:259-344, 507-589`). No host arithmetic touches any of them. No `ttnn::narrow` (Type 4), no `address_offset` (Type 3).

**One construct deserves an explicit non-finding, because it looks like the gate and is not.** The tilized kernels pass a byte offset alongside the page id on both reads and writes — e.g. `noc.async_read(s0, dfb_in0_obj, stick_size, {.page_id = noc_id, .offset_bytes = noc_offset}, …)` (`reader_moreh_getitem_tilize.cpp:292-293`) and the matching write at `writer_moreh_getitem_tilize.cpp:61-62`, with `noc_offset` from `get_noc_offset_in_tile(...)` (`common.hpp:13-33`). This is **kernel-side, page-relative NoC addressing** — the offset is computed on-device from tile-face geometry and applied to a page the accessor resolves. It is not a host-folded base (`buffer()->address() + offset`), which is what Types 1 and 2 recognize, and the accessor's base stays clean. **Not a gate.**

Reconciled against `2026-07-19_offset_base_pointers.md`: **no fold present, op not in the tables** → clean. The RTAs hand off to TensorParameter analysis as clean bases.

### CB endpoints (GATE-free) — dispositions recorded, one confirmed drop, one question

Counting method (same as the `moreh_sum` audit, restated so it can be checked): an endpoint is counted where a **code reference** to the CB exists in a kernel, not where a runtime branch happens to execute — Metal 2.0 bindings are static. A reference removed by `#ifdef` or `if constexpr` is **not** a toucher; one under a plain `if` **is**.

The op has **no compute kernels**, so every CB is touched only by a reader and/or a writer, one instance of each per node.

| Shape | CB | Census | Disposition |
|---|---|---|---|
| **RM** | `c_0` | reader P (`reader:211,214`), writer C (`writer:34,37`) | legal 1P+1C |
| | `c_1`–`c_4` (defined dims) | reader only — full FIFO cycle within one kernel (`reserve_back:161` … `push_back:184` … `wait_front:189` … `pop_front:190`) | **self-loop** |
| | `c_5` (defined dim 4) | **0 touchers** — reader has no `dfb` for `c_5`, loop is `dim = 3 … 0` | **question — see below** |
| | **`c_16`** | **0 touchers** | **dead-CB drop — confirmed** |
| **Tilized-W** | `c_0` | reader P (`reader_…_w:331,371`), writer C (`writer_…_w:83,101`) | legal 1P+1C |
| | `c_1`–`c_5` (defined dims) | reader only — locked producer, `reserve_back` + `get_write_ptr`, **no** `push_back` (`reader_…_w:176-201, 247-288`) | **self-loop** |
| | **`c_16`** | **0 touchers** | **dead-CB drop — confirmed** |
| | `c_17` | writer only — **role-free**: `get_read_ptr` (`writer_…_w:50`), raw stores (`:91,98`), NoC source (`:104`); no FIFO ops | **self-loop** |
| **Tilized-noW** | `c_0` | reader P (`reader_…_tilize:276,295`), writer C (`writer_…_tilize:43,64`) | legal 1P+1C |
| | `c_1`–`c_4` (defined dims) | reader only — locked producer, `reserve_back` + `get_write_ptr`, no `push_back` | **self-loop** |
| | **`c_16`** | **0 touchers** | **dead-CB drop — confirmed** |

**No multi-binding anywhere.** No CB on any node has ≥3 distinct touchers or two kernels locked to the same FIFO role. All three faces were hunted: there is no hidden second writer (no semaphores exist in the op, so no semaphore-gated co-fill is possible), no multi-reader CB, and no dual-instance work-split (each kernel source is instantiated once per factory).

#### Confirmed dead CB — `c_16`, in **all three** program shapes

`CBIndex::c_16` is allocated as the output CB in every shape — `rm_factory.cpp:129-138` (named `out_cb_index`), `tilized_factory.cpp:156-166` (`out_cb0_index`), `tilized_factory.cpp:418-427` (`out_cb_index`) — and **no kernel in the op references index 16 anywhere.** A grep for `c_16`, `= 16;`, `(16)` and `16u` across all six kernel `.cpp` files and `common.hpp` returns **zero hits**.

Every writer instead drains **`c_0`**, the input CB, directly:

- `writer_moreh_getitem.cpp:22` — `constexpr uint32_t cb_id_out = tt::CBIndex::c_0;`
- `writer_moreh_getitem_tilize.cpp:33` — same
- `writer_moreh_getitem_tilize_w.cpp:37` — `cb_id_out0 = tt::CBIndex::c_0` (and stages through `c_17`)

The op is a pure gather: the reader fills `c_0` with the selected stick and the writer immediately drains it to the output tensor. No second buffer is needed, and `c_16` was never wired up. The `out_cb_index` naming shows the intent; the wiring never followed.

Ruling out every indirect path, per the recipe's *distrust a `(0,0)` result*: no CTA carries the index (both readers' and both writers' compile-time args are `TensorAccessorArgs` blocks only — `rm_factory.cpp:144-148,159-160`; `tilized_factory.cpp:189-193,205-206,439-443,455-456`); no RTA carries it (the arg lists are enumerated in full above); the only included header, `common.hpp`, contains no CB index at all; and the finding is uniform across **all** configs rather than config-dependent.

**Disposition: PORT WORK — drop all three allocations.** A dead CB has no behavior, so removing it changes none; and a bindingless DFB cannot be expressed in Metal 2.0, so it must go. Effect: one page of L1 saved per core in each shape. Confirmation requested as Question 2, given the recipe's warning about wrongly dropping a live CB.

#### Question, not a drop — `c_5` in the RM factory

`moreh_getitem_rm_factory.cpp:111-127` allocates `CBIndex::c_1 + dim` for **every** defined index dimension `dim ∈ [0,5)`, so a defined index at normalized dim 4 allocates `c_5`. But `reader_moreh_getitem.cpp` declares `DataflowBuffer` objects only for `c_0`–`c_4` (`:133-137`) and its dimension loop runs `for (int32_t dim = 3; dim >= 0; dim--)` (`:146`) — it never reaches dim 4. So when `index_info[4].is_defined`, `c_5` is allocated with **zero endpoints**.

It is filed as a question rather than a drop because the same code path raises a **correctness** concern that the ops team, not the porter, must settle — and the answer determines whether `c_5` is dead-in-a-reachable-config or dead-in-an-unreachable-one:

> The RM guard tests the **user-space** dimension index — `TT_FATAL(dim != 4, "getitem for ROW_MAJOR layout not support W index tensor!")` (`moreh_getitem_device_operation.cpp:47-51`) — while the factory and kernel work in **5-D-normalized** dimensions, `dim = index_dims[i] + input_dim_offset` with `input_dim_offset = 5 - rank` (`rm_factory.cpp:50, 70`). For an input of rank < 5 the two disagree: a rank-4 ROW_MAJOR input with `index_dims = {3}` passes the guard (user dim is 3, not 4) and yields normalized dim 4. That instantiation would allocate `c_5` and — because the reader's loop stops at dim 3 — never consult `index_is_defined[4]`, silently ignoring the index tensor rather than erroring.
>
> Note the tilized factory computes the *same* predicate correctly, in normalized space: `if (dim + input_dim_offset == 4) is_w_index_exist = true;` (`tilized_factory.cpp:75-79`), and routes to a reader that genuinely handles dim 4 (`reader_moreh_getitem_tilize_w.cpp:144` declares `dfb_in5_obj`; its loop runs `dim = 4 … 0`). The RM path has no equivalent.

If the case is reachable, this is a silent-wrong-answer bug that predates and is independent of the port, and `c_5` is a live-config dead CB the port must handle conditionally. If it is unreachable for a reason not visible in this directory, `c_5` never allocates and there is nothing to do. Either way the porter should not guess. → **Question 3.**

---

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** — up to **7 per program**, all **Case 1** (every access goes through a `TensorAccessor`):
  - `input` — Case 1. Base delivered as a `Buffer*` RTA (arg 0); all reads via `noc.async_read(s0, …)`.
  - `output` — Case 1, same shape via `noc.async_write(…, s0, …)`.
  - `index0`–`index4` — Case 1, **but optional**: only the slots the caller supplied are real; the rest are `nullptr`. See the heads-up below — this is the port's main design item.
  - **No Case 2 anywhere.** The raw L1 pointer work in the readers (`index_l1_ptr`, `reader_moreh_getitem.cpp:186-190`) and in `writer_moreh_getitem_tilize_w.cpp:86-99` operates on **CB memory** reached through `dfb.get_write_ptr()` / `get_read_ptr()`, never on tensor memory. No `get_bank_base_address` bridge needed.
  - No borrowed-memory DFB reads (no `set_globally_allocated_address`), so no binding is "clean" by the causal-link gate.
- **TensorParameter relaxation:** `dynamic_tensor_shape` proposed for the RM factory's interleaved-RM parameters — **blocked on Question 1**, not on a gate.
- **TensorAccessor 3rd arg:** drop the page-size argument at all **7 RM sites** (`reader_moreh_getitem.cpp:75,79,80,81,82,83` and `writer_moreh_getitem.cpp:27`); pair with the relaxation decision from Question 1. Tilized sites: none.
- **CB endpoints:**
  - **dead-CB drop:** `c_16` @ `rm_factory.cpp:129-138`, `tilized_factory.cpp:156-166`, `tilized_factory.cpp:418-427` — confirmed, all three shapes
  - **self-loop:** RM `c_1`–`c_4` · Tilized-W `c_1`–`c_5` and `c_17` · Tilized-noW `c_1`–`c_4`
  - **1P+1C / legal:** `c_0` in all three shapes
  - **multi-binding flag:** none
  - **open:** RM `c_5` (Question 3)

---

## Heads-ups  *(mirrors the brief)*

- **Optional / absent tensor bindings — the port's biggest design item.** Both factories pass `index_info[N].buffer` for all five slots, and undefined slots are `nullptr` (`rm_factory.cpp:192-196`; `tilized_factory.cpp:264-268, 512-516`). The framework handles this deliberately — `emplace_runtime_args_impl` (`tt_metal/impl/program/program_descriptors.cpp:239-247`) comments *"nullptr Buffer\* represents an absent optional tensor. Emit 0u with no binding so the fast cache-hit path is not invalidated by optional inputs"* — and the kernels construct all five `TensorAccessor`s unconditionally, using only the ones `index_is_defined[dim]` selects. Metal 2.0's typed binding channel needs an equivalent for "declared but absent this instantiation." Not an Appendix A feature, so **not a gate**, but it is the item most likely to need a framework conversation; surface it early rather than inventing a shape.
- **Runtime-selected DFB handles.** All three readers pick which DFB to act on from a runtime dimension index, via a chain of `if (dim == N)` blocks over five distinct objects (`reader_moreh_getitem.cpp:159-182`, `reader_moreh_getitem_tilize.cpp:162-185, 202-241`, `reader_moreh_getitem_tilize_w.cpp:176-201, 247-288`). `dfb::name` tokens are static, so these do not translate one-for-one. The RM reader goes further and holds a **pointer** to the selected buffer — `DataflowBuffer* index_dfb_obj = nullptr;` (`:158`), dereferenced at `:184, 189, 190` — which has no direct binding-token analogue at all. Expect to keep the `if`-chain and bind all five.
- **The tilized factory is two programs behind one name.** Porting "the tilized factory" means porting both the `is_w_index_exist` branch and the `else` branch — different kernels, different CB sets (`c_17` exists only in the W branch), different runtime-arg lists. Read `moreh_getitem_tilized_factory.cpp:87` and `:354` as the split point.
- **Cross-op / shared kernels:** **none.** The op owns all 6 kernels and its one shared header; no other op instantiates them, and they include nothing outside `api/*` and their own directory. No `_metal2` fork question, no sunset list, no donor coordination.
- **RTA varargs:** none. Every kernel reads a **fixed** run of args through a running `i++` counter at the top (`reader_moreh_getitem.cpp:11-57` and equivalents) — the recipe's explicit non-signal. Every RTA is nameable, and there are many (up to 38 in the tilized-W reader), so naming them is the bulk of the mechanical work.
- **`experimental/quasar/` holds no copy of this op** — checked.

---

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ✓ clean — and unusually so.** Both escape types are absent:

- **No file-path kernel instantiation escape.** Every `kernel_source` in both factories points inside `moreh_getitem/device/`. No other op instantiates a `moreh_getitem` kernel.
- **No function-call escape outside LLK/HAL.** The complete include set across all 6 kernels is `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/dataflow_buffer.h`, `api/tensor/noc_traits.h`, `api/core_local_mem.h` (class 1 — LLK/HAL, no concern), `<algorithm>`, and the op's own `moreh_getitem_tilized_kernels/common.hpp`.

There is no donor summary table or per-call detail because there are no donors. Notably this op does **not** depend on `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`, unlike most of the `moreh` family.

### Relaxation candidates

None mineable — no custom hash exists. The `dynamic_tensor_shape` proposal for the RM factory comes from the 3rd-arg triage, not from hash archaeology, and is unresolved per Question 1.

### TTNN factory analysis

Sheet rows and the code cross-check agree (tabulated under *Gate detail*). What the port's TTNN ProgramFactory wiring needs: current concept `descriptor`, no op-owned tensors, no MeshWorkload need, no pybound internals, no custom hash, neither runtime-arg-update hook → target **`ProgramSpecFactoryConcept`**, plain, for both factories.

---

## Misc anomalies  *(team-only, non-gating, not porter work)*

- **Dead `c_16` allocation in all three shapes** — `rm_factory.cpp:129-138`, `tilized_factory.cpp:156-166`, `:418-427`. Also PORT WORK above, since the port must drop it.
- **Dead `index_cbs[5]` array in all three readers** — declared at `reader_moreh_getitem.cpp:93-99`, `reader_moreh_getitem_tilize.cpp:100-106`, `reader_moreh_getitem_tilize_w.cpp:101-107`. In the two tilized readers it is **never read at all**. In the RM reader its only use is `tt::CBIndex idx_cb = index_cbs[dim];` (`:151`), and `idx_cb` is itself never used — a dead store feeding a dead array. `cb_in5` in the RM and tilize-noW readers is reachable only through it, so it too is effectively dead there.
- **RM `dim != 4` guard operates in the wrong index space** — `moreh_getitem_device_operation.cpp:47-51` vs. the normalized `dim` used by `rm_factory.cpp:70` and the reader's `dim = 3 … 0` loop. Detailed under *CB endpoints → Question 3*; routed as a possible silent-wrong-answer bug independent of the port.
- **Index CB page size hardcoded to `1024 * 4`** — `tilized_factory.cpp:144, 406`, mirrored by `#define INDEX_TILE_SIZE (4096)` in `common.hpp:11` and used as the read size at `reader_moreh_getitem_tilize.cpp:166` and friends. Correct for a 32×32 INT32 tile (the only index dtype the op validates, `moreh_getitem_device_operation.cpp:32`), but the constant is duplicated across host and device with nothing tying them together, and neither is derived from the tile spec.
- **Tile geometry hardcoded in `common.hpp`** — `TILE_HEIGHT 32`, `TILE_WIDTH 32`, `FACE_WIDTH 16` as `#define`s (`:7-9`), plus literal `16`, `32`, `256` face offsets in `get_noc_offset_in_tile` (`:22-30`) and `reader_moreh_getitem_tilize.cpp:191-195`. Correct for the only supported layout; no assert guards a different tile size.
- **`num_elements_per_alignment == 8` special case is unexplained** — `writer_moreh_getitem_tilize_w.cpp:77-79` and `reader_moreh_getitem_tilize_w.cpp` add a conditional `NOC_MINIMUM_READ_SIZE` offset only when the value is exactly 8 (i.e. 4-byte elements at 32-byte alignment). Reads as a targeted fix for one dtype rather than a general rule; worth a comment at minimum.
- **`output_dim_offset` computed from the input rank** — `tilized_factory.cpp:65`, `auto output_dim_offset = 5 - input_shape.rank();` then used to index `new_output_shape` while iterating over `output_shape.rank()` (`:66-69`). The RM factory uses the output's own rank for the same job (`rm_factory.cpp:54`). Benign where the tilized path keeps input and output rank equal (which `compute_output_specs` does for `Layout::TILE`), but the asymmetry is fragile and reads as a copy-paste slip.
- **`moreh_getitem.cpp:15-21` throws on an absent input** that the signature declares optional, with a comment calling it a decorator-infra workaround. Pre-existing; noted only because a reader of the pybind signature (`nb::arg("input") = nb::none()`) would reasonably expect the argument to be omittable.

---

## Per-DeviceOperation attribution

Not applicable — one DeviceOperation. Findings that differ **per factory / per shape** are attributed inline throughout: the 3rd-arg Class-1 sites and the `c_5` question are **RM-only**; `c_17` exists **only** in Tilized-W; the dead `c_16` is **universal**.

---

## Questions for the user

1. **Is the RM factory's page-size override load-bearing, and should the port set `dynamic_tensor_shape`?** *(Blocks a port decision, not the port.)*
   The 7 RM sites are Class 1 by both the triage doc and my own read, and the kernel comments (`reader_moreh_getitem.cpp:73-74`) assert the override exists because `AlignedPageSize` "may be stale on program cache hits." But the recipe's safety check for applying a relaxation is *"confirm the existing custom hash's logic matches the listed relaxation"* — and **this op has no custom hash**, so there is nothing to confirm against. That leaves two readings with different consequences:
   - **(a) The override is load-bearing.** The default program hash does *not* separate two calls whose row widths differ, so a cache hit can reuse a program whose compile-time `AlignedPageSize` is wrong — which is exactly what the comment describes. Then the port must set `dynamic_tensor_shape` to preserve today's behavior, and dropping the arg without it would reintroduce the staleness bug.
   - **(b) The override is redundant.** The default hash already distinguishes those shapes (so the program is rebuilt and `AlignedPageSize` recomputed), the comment is defensive, and this is really Class 2 — drop the arg, add no relaxation. Setting `dynamic_tensor_shape` here would *broaden* cache reuse beyond what the legacy op does — a semantic change in cache behavior, which the port is not supposed to make.

   Deciding needs knowledge of what the default PD hash keys on for a row-major tensor, which is the ops / relaxation-design owner's call, not the auditor's or the porter's. **Recommendation: settle this before the port starts**, since it changes both the kernel diff and the `TensorParameter` declaration.

2. **Confirm the `c_16` dead-CB drop in all three shapes.**
   The evidence is unambiguous — index 16 appears nowhere in any of the six kernels or `common.hpp`, no CTA or RTA carries it, and all three writers demonstrably drain `c_0` instead. Flagging it only because the recipe treats a wrongly-dropped live CB as the worst outcome a port can produce, and because a dead CB in *every* shape of *both* factories is unusual enough to be worth a second pair of eyes. Confirming that the reader→`c_0`→writer handoff is the intended design (rather than `c_16` being a half-finished double-buffering change) closes it.

3. **Is a rank-4 `ROW_MAJOR` input with an index on its last dimension reachable — and if so, is it wrong today?**
   `moreh_getitem_device_operation.cpp:47-51` rejects a W index for ROW_MAJOR by testing the user-space `dim != 4`, but `rm_factory.cpp:70` normalizes to `dim = index_dims[i] + (5 - rank)`. For rank 4 and `index_dims = {3}` the guard passes and the normalized dim is 4 — which allocates `c_5` (`rm_factory.cpp:111-127`) while `reader_moreh_getitem.cpp:146` loops only `dim = 3 … 0`, so the index tensor is silently ignored. The tilized factory computes the same predicate in normalized space and handles dim 4 properly (`tilized_factory.cpp:75-79`), which is what makes the RM path look like an oversight rather than a deliberate restriction. This is independent of the port; it decides whether RM `c_5` is a reachable dead CB the port must handle or a non-case.

---

## Recipe notes

1. **§TTNN factory concept prerequisite — the connector-unavailable branch, second occurrence.** As logged in the `moreh_sum` audit, the claude.ai Google Drive connector is absent from this session, so the sheet cannot be fetched by the documented procedure; the user pasted the rows instead, which worked cleanly again. The suggestion stands: add an explicit branch for *connector unavailable* (complete the audit, mark the gate INDETERMINATE rather than RED-blocked, ask the human to authorize or paste), and state whether user-pasted rows are acceptable substitutes and what provenance to record. Two consecutive audits have now taken this path.

2. **§TensorAccessor 3rd argument × §TensorParameter relaxations — Class 1 has no safety check on a no-custom-hash op.** This is the sharpest gap I hit. The relaxations subject says a relaxation-bearing op *has* a custom hash ("the relaxation **is** the hash excluding the relaxed property"), and builds its whole verification step on comparing the two. `moreh_getitem` is a **Class 1 op with `Custom hash == no`** — a combination the recipe does not anticipate, and one that is now *reachable*, since the custom-hash gate that used to make Class 1 co-occur with a RED is not firing here. The result is that the recipe hands the porter a mechanical-sounding instruction ("set `dynamic_tensor_shape` and drop the manual override") whose correctness cannot be established by any check the recipe provides, and where the wrong choice is a silent cache-semantics change in one direction or a reintroduced staleness bug in the other. I declined to pick and raised it as Question 1. Suggest: give §TensorParameter relaxations an explicit *no-custom-hash* branch — either a substitute check (e.g. "confirm the default hash's treatment of the relaxed property") or an instruction to route it as a question, as I did.

3. **§Feature compatibility / CTA varargs — the op-level cue's flattened-to-fixed-array case is worth naming.** The entry's false-positive guard covers "a variable-count input list whose per-input data rides RTAs." `moreh_getitem` is a *stronger* and structurally different non-case: the host flattens `std::vector<Tensor>` into a **fixed-size array** (`IndexInfo index_info[5]`) and emits a constant number of CTA blocks for all slots, defined or not, so the CTA count is fixed by construction rather than by where the metadata rides. That pattern — max-rank array, unconditional emission, `is_defined` flags on RTAs — seems likely to recur in indexing/gather ops. Adding it to the guard would let a future auditor dismiss the cue in one step instead of three.

4. **§CB endpoints / §Output — no place to record that one ProgramFactory emits several distinct programs.** `MorehGetItemTilizedFactory` branches on `is_w_index_exist` into two programs with different kernels *and* different CB sets. The recipe's *Classify per instantiation* rule covers the CB census correctly, but the identifying section, the status summary's "DOps / Factories" row, and the readiness sheet's one-row-per-factory model all assume factory ≈ program. I handled it by introducing a "program shape" table in the identifying section and threading the three shapes through every subject, but that is my invention, not the recipe's. A sentence in §Output sanctioning a per-shape breakdown when a factory is internally branched would make this consistent across auditors — and would flag to the sheet owner that "the Tilized factory is ported" is a two-part claim.

5. **Minor — "dead CB should be exceedingly rare" is not matching observation in this family.** The recipe warns that a `(0,0)` result is "more likely a gap in your own analysis than a real dead CB." That framing was useful and I applied the full rule-out procedure both times, but this is now **two consecutive moreh ops** with a confirmed dead CB (`moreh_sum`'s `c_24` in one factory; `moreh_getitem`'s `c_16` in all three shapes), plus two config-scoped ones raised as questions. The caution is still right — over-calling is the dangerous direction — but auditors of this family should expect to find them, and the recipe's "flagrant waste you would not expect to find" phrasing may cause someone to talk themselves out of a real one.
