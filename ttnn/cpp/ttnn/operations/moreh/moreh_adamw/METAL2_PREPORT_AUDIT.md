# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/moreh/moreh_adamw`

Single device operation, single factory. `create_descriptor` and `override_runtime_arguments` are both defined in the
factory file; there is no separate `ProgramFactory` class.

- **`MorehAdamWDeviceOperation`**
  - `MorehAdamWDeviceOperation` (single-descriptor) — `device/multi_core_program_factory.cpp`
    (`create_descriptor` @ `:58`, `override_runtime_arguments` @ `:353`), declared in
    `device/moreh_adamw_device_operation.hpp`

Kernels referenced by the factory — all three owned by this op, none shared with any other op:

- `device/kernels/reader_moreh_adamw.cpp` (reader / DM)
- `device/kernels/writer_moreh_adamw.cpp` (writer / DM)
- `device/kernels/moreh_adamw.cpp` (compute; instantiated **twice**, over the two disjoint work-split core groups)

No unreferenced kernel files in the directory. *(Name-collision warning for anyone re-running the shared-kernel census:
the compute kernel `device/kernels/moreh_adamw.cpp` and the host wrapper `moreh_adamw.cpp` share a filename. A
filename grep hits `moreh/sources.cmake`, which lists the **host** file — it is not a second binder of the kernel.)*

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `086a669ff5e 2026-08-15 docs(metal_2.0): two porter-facing gaps a blind cold read turned up`

> ### Readiness-sheet gate — held `no` deliberately; audited as `yes` on the maintainer's instruction
>
> The readiness sheet's `Is able to port?` cell for `moreh/moreh_adamw` reads **`no`**. That is **not** an op defect:
> this op belongs to the family targeting `CustomProgramSpecFactoryConcept` (selected by
> `Override runtime args method? == yes`), whose audit/port-recipe support is newly added and **still under test**.
> The sheet rows are held red on purpose, to stop the porting team starting these ops before that testing completes.
>
> This audit was run with the cell treated as **`yes`**, on the recipe maintainer's explicit instruction, as part of
> that testing. The verdict below is **GREEN**, and a porter brief is issued.
>
> **Downstream readers: this op is not yet released for porting.** The GREEN certifies that every audit gate clears on
> the code; it does not lift the family-wide hold. Fingerprint of the held row: `Concept` = `descriptor`,
> `Override runtime args method?` = `yes`, `Porting Target` = `CustomProgramSpecFactoryConcept`,
> `Is able to port?` = `no`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/moreh/moreh_adamw` |
| **Overall** | **GREEN** (gate audited as `yes` per the note above; sheet row held `no`) |
| **DOps / Factories** | `MorehAdamWDeviceOperation` → `MorehAdamWDeviceOperation (single-descriptor)` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all three kernels, and every donor function they call, are Device 2.0 clean. In fact they are already **past** Device 2.0: they use `DataflowBuffer` objects, not the `CircularBuffer` wrapper. |
| *Prereqs* — Cross-op escapes | **Ok** — two donor headers, both `ttnn/cpp/ttnn/kernel/` shared-lib class, all calls take `DataflowBuffer` (see the shape note below) |
| *Feature Support* — overall | **GREEN** (all four Appendix A entries `N/A`) |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (as audited — see the note above; sheet cell held `no`) |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Custom hash | `compute_program_hash`: **No**. Backdoor (`attribute_values`): **Yes** — `device/moreh_adamw_device_operation.hpp:35-40` (not a gate; port leaves it intact) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | **Yes** (not a gate; selects `CustomProgramSpecFactoryConcept`): `device/multi_core_program_factory.cpp:353` |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `CustomProgramSpecFactoryConcept` |
| *Port work* — Offset base pointer | **none** (GREEN — all nine bases clean; page offset travels as a separate scalar) |
| *Port work* — Tensor bindings (per binding) | **9 bindings, all Case 1** (`TensorAccessor`); two of them `amsgrad`-only |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | **none** — all nine accessors are 2-arg |
| *Port work* — CB endpoints | 19 CBs: 11 legal 1:1 · 5 **self-loop** · **3 dead-CB drops, config-scoped to `amsgrad == false`** |

**CB endpoints** are dispositions, not gates. The `amsgrad` axis flips three CBs between legal and dead, so they are
recorded per `(CB, config)` below — this is the load-bearing finding of the audit.

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`, beside this file), subject to the family-wide hold above.

Every gate clears on evidence: Device 2.0 complete (and then some), no Appendix A feature, no offset base pointer, no
`TensorAccessor` 3rd argument, `TensorParameter relaxation == none`.

**The port is mostly mechanical, with one structural item.** The kernels have already made the CB→DFB *API* move (they
use `DataflowBuffer`, `Noc`, and `TensorAccessor` throughout), so the remaining work is the *binding* layer —
`dfb::name`, `tensor::name`, named args — not an idiom rewrite. The one thing the porter cannot do mechanically:

> **Three of the op's 19 CBs are dead when `amsgrad == false`, and a dead DFB cannot be expressed in Metal 2.0 at
> all** (the spec validator rejects a DFB with no producer and no consumer binding). The factory currently allocates
> all 19 unconditionally, so the port must make the `c_4` / `c_19` / `c_27` DFB specs **conditional on `amsgrad`**.
> This is not a branch the legacy code has — it is new structure the port must add. Details under *Gate detail → CB
> endpoints* and *Port-work summary*.

## Gate detail

- **TTNN factory concept (`Is able to port?`): GREEN — audited as `yes`.** See the note at the head of this document.
  Sheet fetched fresh this session; one row for this op, matching the code's one factory (factory-set match ✓).
  The cross-check of every cheaply-checkable factual column against the code is **clean**.

  Sheet row, verbatim:

  | Column | Value |
  |---|---|
  | `Op` | `moreh/moreh_adamw` |
  | `Device operation` | `MorehAdamWDeviceOperation` |
  | `Factory (variant)` | `MorehAdamWDeviceOperation (single-descriptor)` |
  | `Concept` | `descriptor` |
  | `Op Classification` | `PD Op (custom)` |
  | `Execution Model` | `SPMD` |
  | `Porting Target` | `CustomProgramSpecFactoryConcept` |
  | `Custom hash (compute_program_hash)` | `no` |
  | `Backdoor custom hash (attribute_values / to_hash)` | `yes` |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` |
  | `Override runtime args method? (PD only)` | `yes` |
  | `Pybind descriptor (nb::class_ of device op)` | `no` |
  | `Smuggled pointer (raw buffer addr in RTA/CRTA)` | `no` |
  | `Known op issues` | *(empty)* |
  | `Diego validation` | `yes` |
  | **`Is able to port?`** | **`no`** — *deliberate family-wide hold; audited as `yes`, see the note above* |
  | `TensorParameter relaxation` | `none` |
  | `Op-owned tensors?` | *(empty)* |
  | `Secretly SPMD Workload?` | *(empty)* |
  | `Factory definition path` | `ttnn/cpp/ttnn/operations/moreh/moreh_adamw/device/multi_core_program_factory.cpp` |
  | `Declared in` | `ttnn/cpp/ttnn/operations/moreh/moreh_adamw/device/moreh_adamw_device_operation.hpp` |

  Cross-check against the code:

  | Column | Sheet | Code evidence | Agrees |
  |---|---|---|---|
  | `Concept` | `descriptor` | `create_descriptor()` returning `ProgramDescriptor` @ `device/multi_core_program_factory.cpp:58` | ✓ |
  | `Custom hash` | `no` | no `compute_program_hash` anywhere in the op dir | ✓ |
  | `Backdoor custom hash` | `yes` | hand-written `attribute_names` / `attribute_values` @ `device/moreh_adamw_device_operation.hpp:35-40`, excluding `lr` and `step` from the hash (comment @ `:31-34` states why) | ✓ |
  | `Runtime-args update` | `no` | no `get_dynamic_runtime_args` hook on the device-op | ✓ |
  | `Override runtime args method?` | `yes` | `MorehAdamWDeviceOperation::override_runtime_arguments` @ `device/multi_core_program_factory.cpp:353`, declared `device/moreh_adamw_device_operation.hpp:76-81` | ✓ |
  | `Pybind descriptor` | `no` | `moreh_adamw_nanobind.cpp` uses only `ttnn::bind_function<"moreh_adamw">` @ `:43`; no `nb::class_`, no `create_descriptor` | ✓ |
  | `Smuggled pointer` | `no` | all eight/nine addresses are passed as **annotated** `Buffer*` via `emplace_runtime_args` @ `device/multi_core_program_factory.cpp:305, 324`, not bare `->address()` | ✓ |
  | `Op-owned tensors?` | *(empty)* | `descriptor` concept, no `buffers` vector. The optional outputs created by `create_output_tensors` (`device/moreh_adamw_device_operation.cpp:99-133`) are ordinary op outputs, not op-owned scratch | ✓ |
  | Factory-set match | 1 row | 1 factory in code | ✓ |

  Cross-column invariants: `get_dynamic_runtime_args == no` on a `descriptor` concept ✓; `Op-owned tensors?` not `yes`
  on a `descriptor` concept ✓. No invariant violated.

- **Device 2.0 (every kernel used): GREEN — and the kernels are already past it.** All three kernels use
  `DataflowBuffer` (the Metal 2.0 spec-layer object exposed kernel-side) rather than the Device 2.0 `CircularBuffer`
  wrapper, plus `Noc` and `TensorAccessor` throughout. A targeted scan for every Device 1.0 idiom
  (`noc_async_read` / `noc_async_write`, `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front`,
  free-function `get_write_ptr(` / `get_read_ptr(`, `InterleavedAddrGen` / `ShardedAddrGen` /
  `InterleavedPow2AddrGen*`, raw semaphore addresses) over all three kernels **and** both donor headers returned
  **zero** hits.

  | Kernel | Device 2.0 evidence | Verdict |
  |---|---|---|
  | `device/kernels/reader_moreh_adamw.cpp` | `Noc noc` @ 80 · `DataflowBuffer` ×9 @ 62-65, 81-86 · `noc.async_read(accessor, dfb, bytes, {.page_id}, {.offset_bytes})` @ 101, 106, 111, 116, 122 · `TensorAccessor` @ 51-54, 59 | ✓ |
  | `device/kernels/writer_moreh_adamw.cpp` | `Noc noc` @ 38 · `DataflowBuffer` ×4 @ 39-43 · `noc.async_write(dfb, accessor, bytes, {.offset_bytes}, {.page_id})` @ 57, 62, 67, 73 · `TensorAccessor` @ 28-30, 35 | ✓ |
  | `device/kernels/moreh_adamw.cpp` (compute) | `DataflowBuffer` ×19 @ 21-65, driven through `.wait_front` / `.pop_front` / `.reserve_back` / `.push_back` | ✓ |

  The only CB-index free function anywhere is `get_tile_size(cb_id)` — reader `:89-94`, writer `:46-50`, and inside
  the donor headers. It is **sanctioned** by the recipe's Green bullet and is not flagged. (One wrinkle recorded in
  *Recipe notes* #2: the recipe's stated reason for sanctioning it — the `CircularBuffer` wrapper method forwards to
  the free function — does not apply to these kernels, whose in-scope object is a `DataflowBuffer` with an
  independently-implemented `get_tile_size()`. The sanction still holds on the recipe's explicit list.)

- **Feature compatibility: GREEN** — all four entries `N/A`. Scan covered host code, the device-op, the factory, all
  three kernels, and both donor headers.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` / `CreateGlobalCircularBuffer` / `global_circular_buffer` field / `remote_index` / `remote_cb_*` / `experimental::CreateCircularBuffer(…, global_cb)`. All 19 CBs are plain `CBDescriptor` literals @ `device/multi_core_program_factory.cpp:111-187`. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `CBDescriptor` sets `address_offset` (every literal omits the field → default 0). No `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. |
  | GlobalSemaphore | N/A | The op uses **no** semaphores at all — `grep -i semaphore` over the whole op directory returns nothing. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | See the detail below — this is the one entry that needed real work to clear. |

  **CTA varargs — why `N/A` despite a variable-length CTA list.** The factory appends **four or five**
  `TensorAccessorArgs` blocks to the reader's compile-time args and **three or four** to the writer's, depending on
  `amsgrad` (`device/multi_core_program_factory.cpp:192-207`). The *list length* therefore varies. That is not this
  entry, which fires on a **runtime-varying CTA index**:
  - The kernels read every CTA at a **constexpr** offset — `TensorAccessorArgs<0>()` then
    `TensorAccessorArgs<prev.next_compile_time_args_offset()>()`, a compile-time chain
    (`reader_moreh_adamw.cpp:46-49, 58`; `writer_moreh_adamw.cpp:24-26, 34`). No loop, no runtime index.
  - The variable block is selected by `#ifdef AMSGRAD`, i.e. resolved **at compile time**, and the define is emitted
    iff `amsgrad` (`device/multi_core_program_factory.cpp:211-214`).
  - `amsgrad` **is** in the program hash (`device/moreh_adamw_device_operation.hpp:36`), so a change in `amsgrad`
    forces a cache miss and a recompile. The CTA count is fixed per compiled instantiation.
  - Op-level cue also absent: `tensor_args_t` (`device/moreh_adamw_device_operation.hpp:43-54`) carries nine
    individually-named tensors, four of them `std::optional<Tensor>` — a fixed-count shape, not a
    variable-count container.

- **CB endpoints (GATE-free): 19 CBs; three flip to dead under `amsgrad == false`.** All CBs are declared over
  `all_cores`. The compute kernel is instantiated **twice** — `compute_desc_1` on `core_group_1`, `compute_desc_2` on
  `core_group_2` (`device/multi_core_program_factory.cpp:247-264`) — but the two core ranges are **disjoint**, so every
  node sees exactly one compute instance. This is the *per-group-CTA* shape (`num_units_per_core_group_N` as CTA 0),
  **not** the dual-instance work-split; there is no assignment question and no multi-toucher census anywhere in the op.
  Max touchers on any node for any CB is **2**.

  The config axis is `amsgrad` (on/off). `fp32_dest_acc_en` changes the intermediate CBs' *data format* only
  (`:105-107`) and does not move any endpoint, so it is not a census axis.

  | CB | Index | Producer | Consumer | `amsgrad == true` | `amsgrad == false` |
  |---|---|---|---|---|---|
  | `param_in` | `c_0` | reader | compute | legal 1:1 | legal 1:1 |
  | `grad` | `c_1` | reader | compute | legal 1:1 | legal 1:1 |
  | `exp_avg_in` | `c_2` | reader | compute | legal 1:1 | legal 1:1 |
  | `exp_avg_sq_in` | `c_3` | reader | compute | legal 1:1 | legal 1:1 |
  | `max_exp_avg_sq_in` | `c_4` | reader | compute | legal 1:1 | **DEAD — drop** |
  | `scalar_args` (lr/β1/β2/eps/wd) | `c_5` | reader (`fill_cb_with_value` ×5) | compute | legal 1:1 | legal 1:1 |
  | `one` | `c_6` | reader (`fill_cb_with_value`) | compute | legal 1:1 | legal 1:1 |
  | `param_out` | `c_16` | compute | writer | legal 1:1 | legal 1:1 |
  | `exp_avg_out` | `c_17` | compute | writer | legal 1:1 | legal 1:1 |
  | `exp_avg_sq_out` | `c_18` | compute | writer | legal 1:1 | legal 1:1 |
  | `max_exp_avg_sq_out` | `c_19` | compute | writer | legal 1:1 | **DEAD — drop** |
  | `tmp_param` | `c_24` | compute | compute | **self-loop** | **self-loop** |
  | `tmp_exp_avg` | `c_25` | compute | compute | **self-loop** | **self-loop** |
  | `tmp_exp_avg_sq` | `c_26` | compute | compute | **self-loop** | **self-loop** |
  | `tmp_max_exp_avg_sq` | `c_27` | compute | compute | **self-loop** | **DEAD — drop** |
  | `beta1_exponent` | `c_28` | reader (`fill_cb_with_value`) | compute | legal 1:1 | legal 1:1 |
  | `beta2_exponent` | `c_29` | reader (`fill_cb_with_value`) | compute | legal 1:1 | legal 1:1 |
  | `tmp1` | `c_30` | compute | compute | **self-loop** | **self-loop** |
  | `tmp2` | `c_31` | compute | compute | **self-loop** | **self-loop** |

  **Self-loops (5, both configs; `c_27` amsgrad-only).** Each is a compute-kernel scratch tile that the *same* kernel
  both fills and drains — one toucher, so bind compute PRODUCER **and** CONSUMER. Legal on Gen1 for a compute kernel.
  Representative evidence: `c_24` is produced at `moreh_adamw.cpp:102-103` (`sub_tiles_to_cb(..., tmp_dfb_param_obj)`)
  and consumed at `:276` (`sub_tiles_to_cb(tmp_dfb_param_obj, ...)`); `c_30` is reserved/pushed at `:134-141`,
  `:175-184`, `:211-226`, `:236-246` and waited/popped at `:210`, `:225`, `:235`, `:245` — all inside the compute
  kernel. No reader or writer kernel references any of `c_24`–`c_27`, `c_30`, `c_31`.

  **Dead CBs (3, `amsgrad == false` only) — positively confirmed, per the recipe's "distrust a `(0, 0)` result."**
  The recipe is right that a genuinely dead CB is rare, so this was checked hard rather than assumed. Confirmation is
  unusually clean here because the deadness is *compile-time* rather than incidental:
  - Every reference to `c_4`, `c_19`, `c_27` in all three kernels sits inside an `#ifdef AMSGRAD` block —
    `reader_moreh_adamw.cpp:56-60, 85-87, 93-95, 120-126`; `writer_moreh_adamw.cpp:32-36, 42-44, 49-51, 71-77`;
    `moreh_adamw.cpp:28-31, 43-46, 54-57, 90-92, 187-206, 212-214, 227-229, 282-284`.
  - The `AMSGRAD` define is emitted **iff** `amsgrad`, to both the DM and the compute kernels
    (`device/multi_core_program_factory.cpp:211-214`), so under `amsgrad == false` these indices do not exist in the
    compiled kernel at all.
  - **No indirect path.** The three indices are `constexpr auto` locals declared *inside* the `#ifdef` blocks; none is
    threaded through a CTA, computed, offset, or aliased from another value. There is no helper that could receive one.
  - Both configs were inspected; `amsgrad` is the only axis that moves them.
  - Meanwhile the *allocations* are unconditional: `c_4` @ `device/multi_core_program_factory.cpp:135-140`, `c_19` @
    `:182-187`, `c_27` from the `c_24..c_31` loop @ `:155-162`.

  So under `amsgrad == false` the op allocates three CBs (2 × `data_tile_size` + 1 × `intermed_tile_size` per core)
  that no kernel touches. Dropping them changes L1 footprint and nothing else. **This is forced, not optional:** a DFB
  with neither a producer nor a consumer binding is rejected by the spec validator, so the port cannot carry these
  across — it must make the three DFB specs conditional on `amsgrad`. See *Port-work summary*.

  **No multi-binding, no hidden second writer.** The op declares no semaphores, so the semaphore-gated raw co-fill
  shape cannot occur. No kernel takes a raw pointer into any CB (`get_write_ptr` / `get_read_ptr` / `fifo_*_ptr`
  appear nowhere in the three kernels), so there are no non-FIFO touchers to miss — every endpoint in the table above
  is a FIFO endpoint, and the census is exhaustive.

- **Offset base pointers: GREEN.** Nine address-bearing arguments, every one a clean base. `moreh_adamw` appears in
  the `2026-07-19_offset_base_pointers.md` tables in neither direction — scan run independently, per the recipe's
  "never let *not in the tables* stand in for *scanned and clean*."

  | Site | Expression | Fold? |
  |---|---|---|
  | `device/multi_core_program_factory.cpp:269-281` (cache miss, both DM kernels) | eight `tensor.buffer()` / `.value().buffer()` `Buffer*` captures, no arithmetic; the two optional ones resolve to `nullptr` when absent | no |
  | `device/multi_core_program_factory.cpp:374-384` (cache hit, `override_runtime_arguments`) | nine `…buffer()->address()` expressions, each assigned straight into an array element — no `+` on any | no |

  The per-core page offset is deliberately **not** folded into any address: it travels as a separate scalar
  (`tile_offset` → reader RTA 15 / writer RTA 5 → `start_id`) and is applied on-device as a page index,
  `{.page_id = i}` (`reader_moreh_adamw.cpp:101-123`, `writer_moreh_adamw.cpp:57-74`). Type 3 (`address_offset`) N/A;
  Type 4 (`narrow`) N/A.

- **TensorAccessor 3rd argument: GREEN — N/A, the subject does not fire.** All nine `TensorAccessor` constructions in
  the op pass **two** arguments — `reader_moreh_adamw.cpp:51-54, 59` and `writer_moreh_adamw.cpp:28-30, 35`. There is
  no explicit page-size override anywhere, so there is no site to classify. (`moreh_adamw` is likewise absent from
  `2026-07-06_tensor_accessor_3rd_arg_triage.md`, consistent with the read.) The `*_tile_bytes` values passed to
  `noc.async_read` / `noc.async_write` are transfer-size arguments, not accessor constructor arguments — not this
  subject.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings — 9 bindings, all Case 1** (fed into a `TensorAccessor`, all memory access through it). Two are
  `amsgrad`-only. No Case 2 anywhere: no kernel does raw address arithmetic on a base.

  | Binding | Kernel | Case | Miss-path delivery | Hit-path delivery |
  |---|---|---|---|---|
  | `param_in` | reader | 1 | `Buffer*` @ `:307` | `->address()` @ `:375` |
  | `grad` | reader | 1 | `Buffer*` @ `:308` | `->address()` @ `:376` |
  | `exp_avg_in` | reader | 1 | `Buffer*` @ `:309` | `->address()` @ `:377` |
  | `exp_avg_sq_in` | reader | 1 | `Buffer*` @ `:310` | `->address()` @ `:378` |
  | `max_exp_avg_sq_in` *(amsgrad)* | reader | 1 | `Buffer*` or `nullptr` @ `:311` | `->address()` or `0u` @ `:379` |
  | `param_out` | writer | 1 | `Buffer*` @ `:326` | `->address()` @ `:381` |
  | `exp_avg_out` | writer | 1 | `Buffer*` @ `:327` | `->address()` @ `:382` |
  | `exp_avg_sq_out` | writer | 1 | `Buffer*` @ `:328` | `->address()` @ `:383` |
  | `max_exp_avg_sq_out` *(amsgrad)* | writer | 1 | `Buffer*` or `nullptr` @ `:329` | `->address()` or `0u` @ `:384` |

  Neither path is the silent-wrong `->address()`-on-an-RTA hazard: the miss path uses the annotated `Buffer*` form
  (auto-registered as a `BufferBinding` and patched on cache hits), and the hit path is the sanctioned
  `override_runtime_arguments`, which supersedes binding resolution. The two agree on the absent-optional case —
  `emplace_runtime_args` emits `0u` with no binding for a `nullptr` `Buffer*`
  (`tt_metal/impl/program/program_descriptors.cpp:243-251`) and the override writes `0u` too. Under Metal 2.0 both
  paths collapse into nine `TensorParameter` / `TensorBinding` declarations; RTA slots 0-4 (reader) and 0-3 (writer),
  plus all the `TensorAccessorArgs(...).append_to(...)` CTA plumbing @ `:192-207`, disappear.

- **TensorParameter relaxation:** `none`.

- **TensorAccessor 3rd arg:** none — all nine accessors are already 2-arg.

- **CB endpoints:**
  - **Self-loop** `c_24`, `c_25`, `c_26`, `c_30`, `c_31` (both configs) and `c_27` (`amsgrad == true` only) — bind the
    compute kernel PRODUCER and CONSUMER.
  - **Dead-CB drop, config-scoped:** `c_4` @ `device/multi_core_program_factory.cpp:135-140`, `c_19` @ `:182-187`,
    `c_27` (from the loop @ `:155-162`) are dead when `amsgrad == false`.
    **The port must make these three DFB specs conditional on `amsgrad`** — declare them only on the `amsgrad` path.
    This is the one piece of new structure the port adds; the legacy factory has no such branch. Note `c_27` is *not*
    an unconditional drop: it is a live self-loop when `amsgrad` is on. There is no dead CTA to drop alongside them —
    none of the three indices is threaded to a kernel as a compile-time arg.
  - Everything else: legal 1:1, ordinary translation.

- **TTNN factory wiring (target concept):** `CustomProgramSpecFactoryConcept`. `override_runtime_arguments` @
  `device/multi_core_program_factory.cpp:353-428` is *translated* into a `ProgramRunArgs`-returning method, not
  deleted. The backdoor hash (`device/moreh_adamw_device_operation.hpp:35-40`) is left exactly as it is — it excludes
  `lr` and `step` from the program hash, which is safe **only because** the override re-applies them (plus the two
  β-exponents derived from `step`) on every cache hit. The two are a matched pair and must stay in sync.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. No CB on any node has ≥3 touchers or two kernels locked to
  the same FIFO role; max is 2. The two compute `KernelDescriptor`s cover **disjoint** core ranges
  (`device/multi_core_program_factory.cpp:250`, `:260`), so this is the per-group-CTA shape, not a dual-instance
  work-split — no 1P+1C assignment question arises.

- **Cross-op / shared kernels: none.** All three kernels are owned exclusively by this op; a filename census
  (`grep -rl <filename> ttnn/cpp/ttnn/operations/`) finds no other binder. No `_metal2` fork exists or is needed, and
  there is no sunset list. *(The one census hit worth discarding: `moreh/sources.cmake` lists the host-side
  `moreh_adamw/moreh_adamw.cpp`, which shares a filename with the compute kernel
  `device/kernels/moreh_adamw.cpp`. Check the bound path, not the filename.)*

- **RTA varargs: none — name every argument.** Both DM kernels read a fixed run via a running `i++` at the top of the
  kernel (`reader_moreh_adamw.cpp:14-32`, `writer_moreh_adamw.cpp:11-18`), which the recipe classifies as legacy
  positional plumbing rather than a loop; the compute kernel reads a single constant index
  (`moreh_adamw.cpp:17`). No counted loop, no data-selected index. Names are legible from the kernel locals:
  - reader (16 args) → `param_addr`, `grad_addr`, `exp_avg_addr`, `exp_avg_sq_addr`, `max_exp_avg_sq_addr` *(the five
    become tensor bindings)*, `lr`, `beta1`, `beta2`, `eps`, `weight_decay`, `beta1_exponent`, `beta2_exponent`,
    `step`, `amsgrad`, `num_tiles_per_core`, `start_id`
  - writer (6 args) → `param_addr`, `exp_avg_addr`, `exp_avg_sq_addr`, `max_exp_avg_sq_addr` *(bindings)*,
    `num_tiles_per_core`, `start_id`
  - compute (1 arg) → `step`

  *(Three of those are dead on arrival — see Misc anomalies. They are still ported as-is: removing them is an ops-team
  change, not port work.)*

- **The kernels are already half-ported — read them before planning.** All three use `DataflowBuffer` objects
  constructed from `constexpr` CB indices (e.g. `DataflowBuffer dfb_param(cb_id_param)`,
  `reader_moreh_adamw.cpp:81`), so the CB→DFB *API* move (`cb_dfb_api_whitelist.md` section A) is already done. What
  remains is the binding layer: `DataflowBuffer dfb_param(dfb::param)`. Two consequences worth knowing up front:
  - **The compute kernel carries both forms for the same buffer** — a `constexpr auto cb_one = tt::CBIndex::c_6`
    *and* a `DataflowBuffer dfb_one_obj(cb_one)` (`moreh_adamw.cpp:35-36`) — and uses each in different places:
    object-taking donor helpers (`sub_tiles_init_with_dt(dfb_one_obj, …)` @ `:135`) alongside raw-index LLK calls
    (`sub_tiles(cb_one, cb_scalar_args, …)` @ `:136`). Both collapse onto the binding: the object is constructed from
    `dfb::one`, and the raw-index LLK call takes the token directly.
  - **`get_tile_size(cb_id)` @ `reader_moreh_adamw.cpp:89-94` and `writer_moreh_adamw.cpp:46-50` is declared `const`,
    not `constexpr`** — so by `cb_dfb_api_whitelist.md`'s "the legacy declaration is the entire test," these take the
    **member getter** (`dfb_param.get_tile_size()`), not the `get_tile_size(dfb::param)` token form.

## Team-only

- **Out-of-directory coupling & donor shape: ✓ clean** (function-call escapes), **none** (file-path escapes).

  | Op kernel | Donor include | Donor class | Status |
  |---|---|---|---|
  | `reader_moreh_adamw.cpp` | `ttnn/kernel/dataflow/moreh_common.hpp` | **3** — `ttnn/cpp/ttnn/kernel/` shared pool | ✓ |
  | `reader_moreh_adamw.cpp` | `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/dataflow_buffer.h`, `api/tensor/noc_traits.h` | 1 — `tt_metal/*` | ✓ |
  | `writer_moreh_adamw.cpp` | `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/dataflow_buffer.h`, `api/tensor/noc_traits.h` | 1 — `tt_metal/*` | ✓ |
  | `moreh_adamw.cpp` (compute) | `ttnn/kernel/compute/moreh_common.hpp` | **3** — `ttnn/cpp/ttnn/kernel/` shared pool | ✓ |
  | `moreh_adamw.cpp` (compute) | `api/compute/compute_kernel_api.h`, `api/compute/eltwise_binary.h`, `api/compute/eltwise_unary/{eltwise_unary,recip,sqrt}.h`, `api/compute/tile_move_copy.h`, `api/dataflow/dataflow_buffer.h` | 1 — `tt_metal/*` | ✓ |

  **Per-call shape analysis — every donor function this op calls takes `DataflowBuffer` by value.** That shape is
  **not in the recipe's shape table** (see *Recipe notes* #1); assessed directly instead, and it is the best case
  available:

  | Donor file | Functions called | Handle shape | Status |
  |---|---|---|---|
  | `ttnn/kernel/dataflow/moreh_common.hpp` | `fill_cb_with_value` @ `:98` | `DataflowBuffer` (by value) | ✓ excellent |
  | `ttnn/kernel/compute/moreh_common.hpp` | `mul_tiles_to_cb` @ `:139`, `copy_tile_to_cb` @ `:453`, `add_tiles_to_cb` @ `:500`, `sub_tiles_to_cb` @ `:656`, `pack_tile_with_dt` @ `:28`, `copy_tile_init_with_dt` @ `:35`, `add_tiles_init_with_dt` @ `:42`, `sub_tiles_init_with_dt` @ `:70`, `mul_tiles_init_with_dt` | `DataflowBuffer` (by value) | ✓ excellent |

  `DataflowBuffer` has a **non-explicit converting constructor from the binding token** —
  `DataflowBuffer(DFBBindingToken token)`, `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:72` — so a `dfb::name`
  token converts implicitly at a donor call site (one user-defined conversion). Both idioms work: pass the token
  directly, or keep the op's existing local objects and construct them from tokens. **No donor-side change is
  required**, and no donor here needs a `uint32_t`-cast bridge, a `TensorAccessorArgs` hand-off, or a
  `Semaphore`-shape workaround. Per-call detail otherwise omitted (all rolls ✓).

  **File-path kernel instantiation: none.** The op instantiates only its own three kernels, and no other op binds any
  of them.

- **Relaxation candidates (FALLIBLE — candidates to verify; the ops team owns the real analysis):** the backdoor hash
  @ `device/moreh_adamw_device_operation.hpp:35-40` excludes only `lr` and `step` — both scalar op attributes
  re-applied by `override_runtime_arguments`, which is the intended design. Tensor shape/dtype/layout still reach the
  hash via `tensor_args`. It reveals **no** tensor-property independence, so there is **no candidate** here.
  Consistent with the sheet's `TensorParameter relaxation == none`.

  One adjacent observation for the roadmap, not a relaxation candidate: `beta1`, `beta2`, `eps` and `weight_decay`
  *are* hashed (`:36`) even though they reach the device purely as runtime args filled into CBs
  (`reader_moreh_adamw.cpp:66-70`) and change nothing about program structure. Hashing them costs a recompile per
  distinct hyperparameter set. Whether that is deliberate conservatism or an oversight is an ops-team question — the
  comment @ `:31-34` calls them "rarely-varying," which reads deliberate.

- **TTNN factory analysis:** current concept `descriptor` (`create_descriptor` @
  `device/multi_core_program_factory.cpp:58`) · no op-owned tensors · no MeshWorkload need (sheet `Execution Model` ==
  `SPMD`; the op returns a plain `ProgramDescriptor`) · no pybound `create_descriptor` and no other risky pybind
  (`moreh_adamw_nanobind.cpp` uses only `ttnn::bind_function<"moreh_adamw">` @ `:43`) · no `compute_program_hash`,
  backdoor hash present @ `device/moreh_adamw_device_operation.hpp:35-40` · no `get_dynamic_runtime_args` ·
  `override_runtime_arguments` @ `device/multi_core_program_factory.cpp:353` → target
  `CustomProgramSpecFactoryConcept`. Gate conjuncts all clear: relaxation `none` ✓, `get_dynamic_runtime_args`
  absent ✓, not multi-program ✓.

## Misc anomalies  *(team-only, non-gating; route to the ops team — the port does not act on these)*

- **Three dead runtime args, one of which is a kernel's entire RTA list.** All are read into a named local and then
  never referenced:
  - `reader_moreh_adamw.cpp:29` — `step` (reader RTA index 12). The host precomputes both β-exponents
    (`device/multi_core_program_factory.cpp:282-283`), so the reader has no use for `step`.
  - `reader_moreh_adamw.cpp:30` — `amsgrad` (reader RTA index 13). The kernel branches on the `AMSGRAD` **define**,
    never on this runtime value.
  - `moreh_adamw.cpp:17` — `step` (compute RTA index 0, and the compute kernel's **only** runtime arg). `step`
    appears nowhere else in the file except comments (`:170-171`, `:249-250`) explaining that the exponents come from
    the host.

  Two follow-on costs. First, `override_runtime_arguments` spends per-core work every cache hit re-writing two of
  these dead slots — `a[kReaderStepIdx] = step` @ `device/multi_core_program_factory.cpp:404` and the whole compute
  loop @ `:419-427`, which exists solely to write the dead compute `step`. Second, and more brittle: the override's
  liveness guard for the reader keys on a **dead** argument — `if (a.size() <= kReaderStepIdx) continue;` @ `:395`,
  with `kReaderStepIdx = 12` @ `:372`. It works today, but it is anchored to an index that could be removed as
  cleanup without anyone connecting the two.

- **Two unused includes.** `#include <tt-metalium/experimental/program_descriptor_patching.hpp>` @
  `device/moreh_adamw_device_operation.hpp:15` — nothing from it (`resolve_bindings`, `apply_resolved_bindings`,
  `apply_dynamic_runtime_args`, `assert_fastpath_parity`) is referenced in the header or the factory; plausibly a
  leftover from the `get_dynamic_runtime_args` → `override_runtime_arguments` migration. And
  `#include "ttnn/operations/moreh/moreh_helper_functions.hpp"` @ `device/multi_core_program_factory.cpp:13` — nothing
  from it is used in that file (`check_tensor` is used in `moreh_adamw_device_operation.cpp`, which includes it
  separately).

- **`packer_l1_acc` and `dst_full_sync_en` are destructured from the compute-kernel config and dropped.**
  `device/multi_core_program_factory.cpp:98-99` unpacks all five fields, but the `ComputeConfigDescriptor` @ `:241-245`
  sets only `math_fidelity`, `fp32_dest_acc_en` and `math_approx_mode`. Since the whole `compute_kernel_config`
  attribute is hashed (`device/moreh_adamw_device_operation.hpp:36`), two user-settable fields participate in
  cache-key distinctions while having no effect on the generated program — two cached programs for behaviourally
  identical configs. (Unlike the `dst_full_sync_en` case, `packer_l1_acc` has no corresponding descriptor field, so
  the drop may be unavoidable; `dst_full_sync_en` does exist on `ComputeConfigDescriptor` and is simply not set here.)

- **Under `amsgrad == false` the op burns three tiles of L1 per core on CBs nothing touches** — `c_4` and `c_19` at
  `data_tile_size` each, `c_27` at `intermed_tile_size`. This is the same finding as the dead-CB port item above,
  noted here because it is a live inefficiency in the *legacy* op too, independent of the port: the allocations at
  `device/multi_core_program_factory.cpp:135-140`, `:155-162`, `:182-187` are unconditional while every use is
  `#ifdef AMSGRAD`.

## Recipe notes

1. **The per-call donor shape table has no row for `DataflowBuffer` — the shape every donor in this op actually
   uses.** `#out-of-directory-coupling`'s table covers `CircularBuffer` / `CircularBuffer&` (⭐ flag) and
   `uint32_t cb_id` (✓ OK), but not `DataflowBuffer` / `DataflowBuffer&`. Both shared-pool headers this op consumes
   (`ttnn/kernel/dataflow/moreh_common.hpp`, `ttnn/kernel/compute/moreh_common.hpp`) take `DataflowBuffer` by value in
   every function it calls, so the table was silent on the entire coupling analysis. I resolved it from the code —
   `DataflowBuffer(DFBBindingToken)` is a non-explicit converting constructor
   (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:72`), so a `dfb::name` token converts implicitly at a call site
   and the shape is ✓ excellent — but that is a framework-header read the recipe should not require of every auditor.
   Suggest adding: **`DataflowBuffer` / `DataflowBuffer&` → ✓ excellent — `dfb::name` converts implicitly; no donor
   change needed.** Worth doing soon: as more shared kernel-lib code migrates to DFB, this becomes the *common* donor
   shape, and an auditor who finds no matching row may reach for the adjacent `CircularBuffer&` row and wrongly
   ⭐-flag a clean donor.

2. **The `get_tile_size(cb_id)` sanction is stated with a rationale that doesn't cover DFB-based kernels.** The
   Device 2.0 Green bullet sanctions it because "the `CircularBuffer` wrapper's `get_tile_size()` just forwards to the
   free function," and instructs the auditor to "check the current Device 2.0 surface rather than assuming the shape
   alone makes it a holdover." These kernels have no `CircularBuffer` in scope at all — the object is a
   `DataflowBuffer`, whose `get_tile_size()` does **not** forward but indexes `unpack_tile_size[logical_dfb_id_]`
   directly (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:167-174`). So the *holdover* test's two conditions are
   both met by shape — wrapper object in scope, member replacement exists — and only the explicit sanctioned-list
   entry saves it. That is the right answer, but I got there by overriding the stated reasoning rather than applying
   it. Suggest either restating the sanction unconditionally ("`get_tile_size(cb_id)` is sanctioned regardless of what
   buffer object is in scope") or extending the rationale to the DFB case. Same issue, same fix, as note #4 in the
   `uniform` audit — this pair of free functions accounts for two of my four Device 2.0 judgment calls across both
   ops.

3. **The dead-CB machinery is written for accidental deadness, not config-gated deadness, and the difference changes
   the port action.** `#dead-cb-0-0` frames the finding as a hazard of the auditor's own analysis ("more likely a gap
   in your own analysis than a real dead CB") and prescribes "the porter drops the allocation." Here the deadness is
   *config-gated and compile-time explicit* — three CBs sit behind `#ifdef AMSGRAD` in every kernel — so confirming it
   was easy, but the action isn't a drop: the CBs are **live** under `amsgrad == true`, so the port must **add a
   conditional** the legacy factory never had. The recipe's *Classify per instantiation* paragraph anticipates that a
   disposition flips with config, but the dead-CB section's resolution text does not, and a porter reading only
   "drop it" would break the amsgrad path. Suggest a sentence in the dead-CB resolution: *"if the CB is dead in one
   config and live in another, the port makes the DFB spec conditional rather than dropping it — and the brief must
   name the config."*

4. **Minor: the CB-endpoints table's "0 touchers" row does not say what to do when the count is config-dependent.**
   Related to #3 but narrower: the census table maps a count to a verdict, with no column for "which config." The
   surrounding prose says to classify per `(CB, config)`, so the information is there — it just isn't in the artifact
   an auditor is most likely to read as the decision procedure. A `(per config)` note in the table caption would
   close it.

5. **Minor: "fetch a fresh copy of the sheet every run" is ambiguous when one session audits several ops.**
   `#ttnn-factory-concept-prerequisite` says "Pull a fresh copy of the sheet every run." This was the second audit in
   one session, so I re-fetched and re-verified the row rather than reusing the copy from twenty minutes earlier.
   Cheap here, but on a session auditing a batch of ops it is a repeated ~260 KB download for data that cannot
   realistically have changed. Suggest clarifying to *"once per session is enough; never reuse a checked-in or
   previous-session copy"* — which is what the rule is actually defending against.

6. **Minor: the audit report template has no natural home for "the kernels are already partially modernized."** This
   op's kernels have already made the CB→DFB API move, which materially changes what the port involves — it is a
   binding-layer change, not an idiom rewrite — and it also creates a specific trap (the compute kernel carries both a
   `constexpr` CB index and a `DataflowBuffer` object for the same buffer, used in different call positions). It is
   squarely porter-facing, but it is not a CB-endpoint, a shared-kernel, or an RTA-vararg finding, so *Watch for* has
   no slot for it. I added a fourth bullet, as in the `uniform` audit. If a partially-modernized kernel is expected to
   become common — and with `_metal2` forks accumulating in-tree, it should — the template may want a standing
   *"kernel starting state"* line.
