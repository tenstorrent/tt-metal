# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/moreh/moreh_sum`

One DeviceOperation, six program factories, all in the same directory (single combined report):

- **`MorehSumOperation`** (`device/moreh_sum_device_operation.{hpp,cpp}`)
  - `MorehSumHFactory` (`device/moreh_sum_h_program_factory.cpp`)
  - `MorehSumWFactory` (`device/moreh_sum_w_program_factory.cpp`)
  - `MorehSumNCFactory` (`device/moreh_sum_nc_program_factory.cpp`)
  - `MorehSumHIntFactory` (`device/moreh_int_sum_h_program_factory.cpp`)
  - `MorehSumWIntFactory` (`device/moreh_int_sum_w_program_factory.cpp`)
  - `MorehSumNCIntFactory` (`device/moreh_int_sum_nc_program_factory.cpp`)

Factory selection is by dtype × reduced-dim (`moreh_sum_device_operation.cpp:17-39`): `INT32` picks the `*Int` factories, otherwise the float ones; `dim == rank-1` → W, `dim == rank-2` → H, else → NC.

**Kernels:** 16 files under `device/moreh_sum_{h,w,nc}_impl_kernels/`. **All 16 are referenced** by a factory — there are no unreferenced/dead kernel files in the directory. `reader_moreh_sum_nc.cpp` and `writer_moreh_sum_nc.cpp` are each shared by two factories (`MorehSumNCFactory` and `MorehSumNCIntFactory`); every other kernel is used by exactly one.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `20c1692eb08 2026-08-03 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/moreh/moreh_sum` |
| **Overall** | **GREEN — all five gates cleared. Brief issued.** |
| **DOps / Factories** | `MorehSumOperation` → `MorehSumHFactory`, `MorehSumWFactory`, `MorehSumNCFactory`, `MorehSumHIntFactory`, `MorehSumWIntFactory`, `MorehSumNCIntFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 16 op kernels and every donor function they call are Device 2.0 native. No holdovers. |
| *Prereqs* — Cross-op escapes | **Ok** — header-only function-call escape into shared pools; all call shapes ✓. No borrowed kernel `.cpp` files. |
| *Feature Support* — overall | **GREEN** — all four Appendix A entries `N/A` |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** — all six factory rows `yes`; cross-check clean |
| *TTNN Readiness* — Concept (current) | `descriptor` — sheet and code agree (six `create_descriptor` returning `ProgramDescriptor`) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A — not a `WorkloadDescriptor` op |
| *TTNN Readiness* — Is safe to port? | **Yes** (all six rows) |
| *TTNN Readiness* — Custom hash | **No** — no `compute_program_hash` anywhere in the op |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** — hook absent from the device-op |
| *TTNN Readiness* — `override_runtime_arguments` | **No** — method absent from all six factories |
| *TTNN Readiness* — Pybind `create_descriptor` | **No** — `moreh_sum_nanobind.cpp` binds only `ttnn::moreh_sum` |
| *TTNN Readiness* — Op-owned tensors | **No** — `descriptor` concept, no `buffers` vector |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (no op-owned tensors) |
| *Port work* — Offset base pointer | **none** — no `->address()` expression exists anywhere in the op |
| *Port work* — Tensor bindings (per binding) | **Case 1** × 2 (`input`, `output`) in all six factories |
| *Port work* — TensorParameter relaxation | **none** — entailed by `Custom hash == no` (see Gate detail) |
| *Port work* — TensorAccessor 3rd arg | **none** — all 10 accessor sites are 2-arg |
| *Port work* — CB endpoints | self-loop (several) · **1 confirmed dead-CB drop** · 1 config-scoped dead-CB **question** |

**CB endpoints** are dispositions, not gates: every out-of-window CB has a port-time resolution. Recorded per `(CB, config)` below.

---

## Result

**GREEN → brief issued.** All five gates cleared, for all six factories. No subset scoping needed — the op is uniformly clear.

- **TTNN factory concept** — all six readiness rows read `Is able to port? == yes` / `Is safe to port? == yes`; the code cross-check agrees on every checkable column.
- **Device 2.0** — clean across all 16 kernels and every donor function they call.
- **Feature compatibility** — all four Appendix A features absent.
- **Offset base pointers** — the op contains no `->address()` expression at all; every tensor base reaches its kernel through the framework's `Buffer*`-binding form.
- **TensorAccessor 3rd argument** — no site passes one.

Port work is light and mechanical: two Case-1 tensor bindings per factory, a set of compute-internal self-loop CBs, and **one confirmed dead-CB drop** (`c_24` in `MorehSumNCFactory`). Two things need a decision before or during the port — a config-scoped zero-endpoint CB in `MorehSumHFactory` (Question 1) and three runtime-selected DFB handles that `dfb::name`'s static tokens do not express one-for-one (Heads-ups).

**Provenance of the readiness data.** The claude.ai Google Drive MCP connector was **not available in this session** (tool absent from the toolset; OAuth cannot run non-interactively), so the sheet could not be fetched by the documented procedure. The six rows were **supplied directly by the user** and are recorded verbatim under *Gate detail* below. They are treated as authoritative, exactly as a fetched copy would be. See *Recipe notes* §1.

---

## Gate detail

### TTNN factory concept (`Is able to port?`) — **GREEN**

Readiness rows as supplied (see *Result* for how they were obtained):

| Device operation | Factory (variant) | Concept | Is safe to port | Is able to port? | DFB Notes |
|---|---|---|---|---|---|
| `MorehSumOperation` | `MorehSumHFactory` | `descriptor` | yes | **yes** | AGREES with 'yes': already kernel-ported to DataflowBuffer by #49430; Class 1 linear FIFO. GREEN. |
| `MorehSumOperation` | `MorehSumHIntFactory` | `descriptor` | yes | **yes** | *(same)* |
| `MorehSumOperation` | `MorehSumNCFactory` | `descriptor` | yes | **yes** | *(same)* |
| `MorehSumOperation` | `MorehSumNCIntFactory` | `descriptor` | yes | **yes** | *(same)* |
| `MorehSumOperation` | `MorehSumWFactory` | `descriptor` | yes | **yes** | *(same)* |
| `MorehSumOperation` | `MorehSumWIntFactory` | `descriptor` | yes | **yes** | *(same)* |

**Gate cleared for all six factories.** `Is safe to port? == yes` is the sheet owner's correctness call and is taken as given.

**Conjuncts not present in the supplied columns are entailed by the verdict.** `Is able to port? == yes` can only hold when `Custom hash`, `get_dynamic_runtime_args`, `override_runtime_arguments` and `Pybind descriptor` are all `no` and `Concept == descriptor`. Every one of those is **independently confirmed in the code** by the cross-check below, so the entailment is corroborated rather than assumed. Two further consequences follow:

- **`TensorParameter relaxation` = `none`.** A relaxation-bearing op has a custom hash (the relaxation *is* the hash excluding the relaxed property); `Custom hash == no` is entailed by the gate and confirmed in code, so no relaxation can be pending. Nothing for the porter to apply, and no mismatch to check.
- **`Op-owned tensors?` = `no`.** The `descriptor` concept cannot carry them (cross-column invariant), consistent with the code.

**Cross-column invariants:** all hold. `get_dynamic_runtime_args` on a `descriptor` concept is permissible in principle and is absent here; `Op-owned tensors == yes` would require `WorkloadDescriptor`, which does not apply.

**Factory-set match:** exact. The sheet's six rows map one-to-one onto the six factories in `moreh_sum_device_operation.hpp:75-81` — no phantom row (every named factory exists in code) and no missing row (every factory in code has a row). The sheet is current for this op.

**Note on the `DFB Notes` column** (informational, not a gate conjunct — and not one of the columns the recipe asks the auditor to verify): its "already kernel-ported to DataflowBuffer by #49430" corroborates the Device 2.0 GREEN below, arrived at independently. Its "Class 1 linear FIFO" characterization is accurate for the reader→compute→writer spine but does not capture two things this audit found — the compute-internal accumulator CBs that are touched by a single kernel (self-loops, five of six factories) and the dead `c_24` in `MorehSumNCFactory`. Not a conflict (different column, different question), but the porter should work from the endpoint census below rather than from that phrase.

The **lightweight cross-check** the recipe pairs with the lookup, run in full:

| Conjunct | Code evidence | Verdict |
|---|---|---|
| `Concept` | `moreh_sum_device_operation.hpp:34,41,48,55,62,69` — six `static ProgramDescriptor create_descriptor(...)`; no mesh-workload return, no `create()`/`override_runtime_arguments()` pair | `descriptor` ✓ |
| `Custom hash` | no `compute_program_hash` in the op directory | `no` ✓ |
| `Runtime-args update (get_dynamic_runtime_args)` | hook absent from `MorehSumOperation` (`moreh_sum_device_operation.hpp:83-86` declares only `select_program_factory`, `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`) | `no` ✓ |
| `Override runtime args method? (PD and legacy)` | no `override_runtime_arguments` in any factory | `no` ✓ |
| `Pybind descriptor` | `moreh_sum_nanobind.cpp:19-29` binds `&ttnn::moreh_sum` only — no `create_descriptor` binding | `no` ✓ |
| `Op-owned tensors?` | `descriptor` concept; no `buffers` vector | `no` ✓ (invariant holds) |
| `Secretly SPMD Workload?` | N/A — not `WorkloadDescriptor` | — |
| **Factory-set match** | six sheet rows ↔ six code factories, one-to-one | ✓ |
| **`Is safe to port?`** | sheet owner's call: `yes` (not auditor-derivable by recipe rule) | ✓ |

**No code-vs-sheet disagreement on any column.** The sheet is not stale for this op.

Supporting evidence for the owner's `Is safe to port?` call (corroboration only — the call is theirs): the op has **no un-annotated pointer arguments**. Every buffer reaches a kernel via `emplace_runtime_args(core, {input_buf, …})` with a `Buffer*`, which `program_descriptors.hpp:194` accepts as the `std::variant<uint32_t, Buffer*>` overload and auto-registers as a `BufferBinding`. That is the framework's sanctioned annotation, i.e. the *opposite* of the smuggled-pointer shape that drives a `no` on this column.

### Device 2.0 (every kernel used) — **GREEN**

All 16 op kernels are structurally Device 2.0: `Noc` objects for every transfer (`noc.async_read` / `noc.async_write` / barriers), `DataflowBuffer` objects for every FIFO operation, `TensorAccessor` for every address generation. A scan of all 16 files for `InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedPow2AddrGen*`, free-function `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front`, free-function `get_write_ptr(cb_id)` / `get_read_ptr(cb_id)`, `get_noc_addr_from_bank_id`, raw semaphore addresses, `CircularBuffer`, and `evil_set_*_ptr` returns **zero violations**.

Two apparent hits are not violations:

| File | Line | Call | Why it is not a holdover |
|---|---|---|---|
| `device/moreh_sum_h_impl_kernels/writer_moreh_int_sum_h.cpp` | 32 | `dfb_out_obj.get_read_ptr()` | **Method** on the Device 2.0 wrapper, not the free function |
| `device/moreh_sum_w_impl_kernels/writer_moreh_int_sum_w.cpp` | 31 | `dfb_out_obj.get_read_ptr()` | Same |

`get_tile_size(cb_id)` appears in every dataflow kernel (e.g. `reader_moreh_sum_nc.cpp:41`, `writer_moreh_sum_h.cpp:25`). This is a **sanctioned** CB-index free function per the Green bullet — the Device 2.0 migration guide keeps it in its own migrated examples (`device_api_migration_guide.md:605,630`). Not flagged.

Compute kernels use LLK free functions that take a CB index (`copy_tile`, `pack_tile`, `matmul_tiles`, `binary_op_init_common`, `reconfig_data_format`, …). These are **compute-engine LLK APIs, outside the scope of the Device 2.0 *data-movement* migration** — the guide covers `noc_*`, CB FIFO functions, pointer getters, semaphores and addr-gens, and never addresses them. No `DataflowBuffer` method replacement exists for them. Not flagged.

**Donor code is equally clean.** Every donor function these kernels call was inspected; all `get_write_ptr` occurrences inside them are **methods on a `DataflowBuffer` parameter**, not free functions:

| Donor | Sites | Shape |
|---|---|---|
| `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` | 57, 65, 79, 196, 236, 277, … | `cb.get_write_ptr()` — method ✓ |
| `ttnn/cpp/ttnn/kernel/dataflow/generate_mm_scaler.hpp` | 15 | `cb.get_write_ptr()` — method ✓ |
| `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl` | 164 | `dfb.get_write_ptr()` — method ✓ |
| `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | — | zero hits |
| `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp` | — | zero hits |
| `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl` | — | zero hits |

### Feature compatibility — **GREEN** (all entries `N/A`)

A scan of the whole op directory for every Appendix A recognition signal — `GlobalCircularBuffer`, `CreateGlobalCircularBuffer`, the `global_circular_buffer` field on a `CBDescriptor`, `address_offset`, `set_address_offset`, `UpdateDynamicCircularBufferAddress`, `cb_descriptor_from_sharded_tensor`, `remote_index` / `remote_cb`, `GlobalSemaphore`, `CreateGlobalSemaphore`, `set_globally_allocated_address` — returns **zero hits**.

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | No GCB type, no `CreateCircularBuffer(…, global_cb)`, no `.global_circular_buffer` field. All 27 `CBDescriptor`s across the six factories are plain. |
| CBDescriptor `address_offset` (non-zero) | N/A | Field never set; no `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. |
| GlobalSemaphore | N/A | The op uses **no semaphores at all** — no `SemaphoreDescriptor`, no `CreateSemaphore`. Synchronization is entirely CB FIFO. |
| Variable-count compile-time arguments (CTA varargs) | N/A | See below. |

**CTA varargs, resolved to the kernel-level signal.** The op-level cue does not fire: `tensor_args_t` (`moreh_sum_device_operation.hpp:25-28`) is a fixed pair — `const Tensor& input` and `const std::optional<Tensor>& output` — not a variable-count container. Reading the kernels anyway, all 26 `get_compile_time_arg_val` call sites use either a **literal constant** index or one **constexpr-computed** offset (`reader_moreh_sum_w.cpp:17`, `get_compile_time_arg_val(src_args.next_compile_time_args_offset())`). No index depends on a runtime value, and no kernel loops over CTAs. Constexpr-computed offsets are explicitly the false-positive guard, not the rule. `N/A`.

### CB endpoints (GATE-free) — dispositions recorded, one question

Counting method used, stated so it can be checked: an endpoint is counted where a **code reference** to the CB exists in a kernel, not where a runtime branch happens to execute — Metal 2.0 bindings are static, so a compiled-but-untaken `wait_front` still requires a binding. Consequently a reference removed by `#ifdef` or `if constexpr` is **not** a toucher, while one under a plain `if` on a constexpr bool **is**. This distinction is what separates the two findings below.

The compute kernels are instantiated twice per factory over **disjoint** core groups (`core_group_1` / `core_group_2`), so each node sees exactly one compute instance — ordinary 1:1, not the dual-instance work-split shape. Helper semantics: `copy_tile_to_dst(DataflowBuffer, …)` does `wait_front` + `pop_front` → **locked consumer**; `pack_tile_from_dst(DataflowBuffer, …)` does `reserve_back` + `push_back` → **locked producer** (`ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp:1172-1191`). `compute_kernel_lib::reduce<…, input, scaler, output>` consumes input/scaler and produces output; `Accumulate::at(cb, n)` consumes the accumulator (`reduce_helpers_compute.inl:196,217`).

| Factory | CB | Config | Census | Disposition |
|---|---|---|---|---|
| `MorehSumHFactory` | `c_0` | all | reader P, compute C | legal 1P+1C |
| | `c_2` scaler | all | reader P, compute C | legal 1P+1C |
| | `c_3` mask_h | `do_mask_h` | reader P, compute C | legal 1P+1C |
| | `c_3` mask_h | `!do_mask_h` | compute only (locked C; reader ref is `#ifdef`-excluded) | **self-loop** |
| | `c_24` accum | all | compute only (P via `reduce` output + C via `Accumulate`) | **self-loop** |
| | `c_25` masked_input | `do_mask_h` | compute only (P+C) | **self-loop** |
| | `c_25` masked_input | `!do_mask_h` | **0 endpoints** (`if constexpr`, `moreh_sum_h.cpp:54`) | **question — see below** |
| | `c_16` | all | compute P, writer C | legal 1P+1C |
| `MorehSumWFactory` | `c_0`, `c_2`, `c_16` | all | 1P+1C | legal |
| | `c_3` mask_w | `do_mask_w` | reader P, compute C | legal 1P+1C |
| | `c_3` mask_w | `!do_mask_w` | compute only (plain `if`, `moreh_sum_w.cpp:37,131`) | **self-loop** |
| | `c_24` accum | all | compute only (P 61/68, C 103/126) | **self-loop** |
| | `c_25` masked_input | all | compute only (P 84/91, C 98/124; plain `if` → always compiled) | **self-loop** |
| `MorehSumNCFactory` | `c_0`, `c_1` zero, `c_16` | all | 1P+1C | legal |
| | **`c_24`** | **all** | **0 endpoints** | **dead-CB drop — confirmed** |
| `MorehSumHIntFactory` | `c_0`, `c_16` | all | 1P+1C | legal |
| | `c_1` mask_h | `do_mask_h` / `!do_mask_h` | 1P+1C / compute only | legal / **self-loop** |
| | `c_24` intermed0 | all | compute only (P 63/81, C 75/87) | **self-loop** |
| `MorehSumWIntFactory` | `c_0`, `c_16` | all | 1P+1C | legal |
| | `c_1` mask_w | `do_mask_w` / `!do_mask_w` | 1P+1C / compute only | legal / **self-loop** |
| | `c_24` intermed0 | all | compute only | **self-loop** |
| `MorehSumNCIntFactory` | `c_0`, `c_16` | all | 1P+1C | legal |
| | `c_24` intermed0 | all | compute only (P via `DataflowBuffer(cb_out)` line 40, C line 32) | **self-loop** |

**No multi-binding anywhere.** No CB on any node has ≥3 distinct touchers or two kernels locked to the same FIFO role. The hidden-second-writer hunt was run over every CB: the only raw-pointer writes into a CB are the in-place int32 sub-tile folds at `writer_moreh_int_sum_h.cpp:32-40` and `writer_moreh_int_sum_w.cpp:31-39`, both performed by the kernel that is already `c_16`'s FIFO consumer — a peek on a binding it already holds, so **one** toucher, not two. No semaphore-gated co-fill exists (the op has no semaphores).

#### Confirmed dead CB — `c_24` in `MorehSumNCFactory`

`device/moreh_sum_nc_program_factory.cpp:95-103` allocates a 1-tile `CBIndex::c_24` (`intermed0_t * intermed_tile_size`, named "accumulated sum" at line 60) that **no kernel in that factory references, in any config.** Verified against the recipe's "distrust a `(0,0)` result" instruction by ruling out every indirect path:

- The factory's three kernels — `reader_moreh_sum_nc.cpp`, `writer_moreh_sum_nc.cpp`, `moreh_sum_nc.cpp` — contain **zero** occurrences of `24`, `c_24` or `intermed` (the only textual hits under `moreh_sum_nc_impl_kernels/` are in `moreh_int_sum_nc.cpp`, which belongs to the *other* factory).
- No CTA carries the index: the reader's and writer's compile-time args are `TensorAccessorArgs` only (lines 117-118, 131-132); the compute kernel's are `{num_cols_per_core_group_N, num_reduce_input_tile}` (lines 159, 175).
- No RTA carries it (lines 203-213).
- `unpack_to_dest_mode` is left all-`Default` (line 150) — unlike the H/W float factories, which do index `unpack_to_dest_mode[CBIndex::c_24]`. That asymmetry is itself corroboration.
- The shared headers the three kernels include (`kernel/dataflow/moreh_common.hpp`, `kernel/compute/moreh_common.hpp`, `kernel_lib/l1_helpers.hpp`) contain no hardcoded `c_24`.

The cause is coherent, which raises confidence: `moreh_sum_nc.cpp` accumulates **in DST** (`add_tiles(..., acc_to_dest = true)`, lines 23/30/36) and never needs an L1 intermediate. Its INT32 sibling `moreh_int_sum_nc.cpp` *does* need one and uses `c_24` — the float factory carries the allocation as a copy-paste leftover.

**Disposition: PORT WORK — drop it.** A dead CB has no behavior, so removing it changes none; and a bindingless DFB cannot be expressed in Metal 2.0 at all, so it *must* go. The only effect is a one-tile L1 saving on `MorehSumNCFactory`. Drop site: `moreh_sum_nc_program_factory.cpp:95-103` (and the now-unused `intermed_data_format` / `intermed_tile_size` locals at lines 70/72, plus `intermed0_t` at line 60).

#### Question, not a drop — `c_25` in `MorehSumHFactory` under `!do_mask_h`

`moreh_sum_h_program_factory.cpp:120-128` allocates `CBIndex::c_25` (`cb_masked_input`) **unconditionally**, but the compute kernel guards every *access* to it behind `if constexpr (do_mask_h)` (`moreh_sum_h.cpp:54`, uses at 67/70/72/74/79). When `origin_H % 32 == 0` the accesses are discarded at compile time, leaving **zero endpoints**.

This is *not* filed as a confirmed drop, per the recipe's instruction to raise a question on any residual doubt. Two things distinguish it from the `c_24` case:

1. It is **config-scoped**, not unconditional — under `do_mask_h` the CB is a live self-loop, so the DFB cannot simply be deleted from the spec; the spec entry would have to become conditional on `do_mask_h`, which the factory already computes at line 39.
2. The kernel still **constructs the object unconditionally** — `DataflowBuffer dfb_masked_input_obj(cb_masked_input);` at `moreh_sum_h.cpp:23`, outside the guard. A Metal 2.0 port that drops the DFB from the spec without also guarding line 23 will fail to compile, because the `dfb::` token would not exist.

Note the **asymmetry with the W-float twin**: `moreh_sum_w.cpp` guards the same construct with a plain `if (do_mask_w)` (line 71) rather than `if constexpr`, so its `c_25` references survive compilation and it has a live toucher in every config. Same logical structure, different disposition — worth knowing before assuming the two mirror each other. → **Question for the ops team** (below).

### Offset base pointers — **GREEN**

The op contains **no `->address()`, `.address()` or `(*buffer).address()` expression at all** — the scan across the whole directory returns zero hits, so there is no address RTA into which an offset could have been folded, and no Type 1 or Type 2 site exists. Nor is there any `ttnn::narrow` (Type 4) or `address_offset` (Type 3).

Every tensor base instead rides the descriptor API's **`Buffer*`-binding form**: the factories hold `auto* const input_buf = input.buffer();` / `output_buf` and pass the pointer itself into `emplace_runtime_args` (`moreh_sum_h_program_factory.cpp:236-250`, `moreh_sum_w_program_factory.cpp:246-259`, `moreh_sum_nc_program_factory.cpp:203-213`, `moreh_int_sum_h_program_factory.cpp:224-238`, `moreh_int_sum_w_program_factory.cpp:229-242`, `moreh_int_sum_nc_program_factory.cpp:189-199`). The framework registers these as `BufferBinding`s and patches them on cache hits.

Reconciled against the triage doc `2026-07-19_offset_base_pointers.md`: **no fold present, op not in the tables** → clean. The RTAs hand off to TensorParameter analysis as clean bases.

### TensorAccessor 3rd argument — **GREEN**

All **10** `TensorAccessor` construction sites in the op pass exactly two arguments:

`reader_moreh_sum_h.cpp:41` · `writer_moreh_sum_h.cpp:21` · `reader_moreh_int_sum_h.cpp:33` · `writer_moreh_int_sum_h.cpp:22` · `reader_moreh_sum_w.cpp:34` · `writer_moreh_sum_w.cpp:21` · `reader_moreh_int_sum_w.cpp:26` · `writer_moreh_int_sum_w.cpp:22` · `reader_moreh_sum_nc.cpp:37` · `writer_moreh_sum_nc.cpp:23`

No explicit page-size override exists, so no site needs classifying and nothing gates. Consistent with the triage doc `2026-07-06_tensor_accessor_3rd_arg_triage.md`, which lists `moreh_fold` and `moreh_getitem` from this family but not `moreh_sum` — here the doc's silence is *confirmed* by the scan rather than relied upon.

---

## Port-work summary  *(would mirror the brief)*

- **Tensor bindings** — two per factory, identical across all six, both **Case 1**:
  - `input` — the base is passed as a `Buffer*` RTA (arg 0) and consumed **only** through `TensorAccessor(src_args, src_addr)`; all reads go via `noc.async_read(s, dfb, …, {.page_id = …})`. → express as `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(tensor::input)`, and the `TensorAccessorArgs(*input.buffer()).append_to(...)` CTA plumbing plus the address RTA both disappear.
  - `output` — same shape via `TensorAccessor(dst_args, dst_addr)` and `noc.async_write`. → `TensorAccessor(tensor::output)`.
  - **No Case 2 anywhere.** The two int32 writers do raw pointer arithmetic (`writer_moreh_int_sum_h.cpp:32-40`, `writer_moreh_int_sum_w.cpp:31-39`), but on **CB memory** via `dfb_out_obj.get_read_ptr()`, never on tensor memory — the tensor write still goes through the accessor. No `get_bank_base_address` bridge is needed.
  - No borrowed-memory DFB reads exist (no `set_globally_allocated_address`), so no binding is "clean" by the causal-link gate.
- **TensorParameter relaxation:** **none** — entailed by `Custom hash == no` and confirmed in code. Nothing to apply.
- **TensorAccessor 3rd arg:** none — nothing to drop.
- **CB endpoints:**
  - **self-loop:** `MorehSumHFactory` (`c_24` all configs; `c_25` masked; `c_3` unmasked) · `MorehSumWFactory` (`c_24`, `c_25` all configs; `c_3` unmasked) · `MorehSumHIntFactory` / `MorehSumWIntFactory` (`c_24` all configs; `c_1` unmasked) · `MorehSumNCIntFactory` (`c_24` all configs)
  - **dead-CB drop:** `c_24` @ `moreh_sum_nc_program_factory.cpp:95-103` (`MorehSumNCFactory`, all configs) — confirmed
  - **1P+1C / legal:** everything else
  - **multi-binding flag:** none

---

## Heads-ups  *(would mirror the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. All three faces were hunted and no CB reaches ≥3 touchers or FIFO-doubles. The in-place raw writes at `writer_moreh_int_sum_{h,w}.cpp` are peeks by `c_16`'s existing FIFO consumer, not second touchers.
- **Runtime-selected DFB handles — the sharpest thing in this op.** Three kernels choose *which* CB to act on at runtime rather than binding a fixed handle. Metal 2.0's `dfb::name` tokens are static, so these do not translate one-for-one and need a deliberate decision (a branch on two bound tokens rather than a mutable index):
  - `moreh_sum_w.cpp:15,46,94` — `cb_input` is a **mutable** variable reassigned from `c_0` to `cb_masked_input` (`c_25`) mid-loop, then used through temporaries `DataflowBuffer(cb_input)` at lines 51, 58, 73, 93, 98, 124. The same expression denotes two different DFBs at different points.
  - `moreh_int_sum_nc.cpp:39-40` — `uint32_t cb_out = last_out ? cb_out0 : cb_intermed0;` then `pack_tile_from_dst(DataflowBuffer(cb_out), dst0)`; one call site produces into either `c_16` or `c_24`.
  - `moreh_int_sum_h.cpp:14` — `auto cb_in0 = tt::CBIndex::c_0;` is declared non-`constexpr` (unlike its siblings). Not reassigned, so cosmetic, but it will read as mutable to the porter.
- **`if constexpr` vs plain `if` guards diverge between the H and W float compute kernels** — `moreh_sum_h.cpp:54` uses `if constexpr (do_mask_h)`, `moreh_sum_w.cpp:71` uses a plain `if (do_mask_w)`. This changes which CBs have compile-time-visible endpoints (see the `c_25` question). Do not assume the two kernels mirror each other.
- **Cross-op / shared kernels:** the op **owns all 16 of its kernel files** and instantiates no borrowed kernel `.cpp` — so there is no `_metal2` fork question and no sunset list. Coupling is header-only (function-call escape), and every call shape is clean; see Team-only below.
- **RTA varargs:** **none.** `ArgFetcher` (`kernel/dataflow/moreh_common.hpp:44-53`) is a running `arg_idx++` counter, used for a **fixed** run of reads at the top of `reader_moreh_sum_nc.cpp` (7) and `writer_moreh_sum_nc.cpp` (3). That is the recipe's explicit non-signal, not a vararg block. All other kernels read literal indices `get_arg_val<uint32_t>(0..4)`. Every RTA is nameable — prefer named args throughout.

---

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ✓ clean.** Two independent reasons the coupling is unusually cheap here:

- **No file-path kernel instantiation escape at all.** Every `kernel_source` in all six factories points inside `moreh_sum/device/`. The op borrows no kernel `.cpp` from any shared pool or other family, and no other op instantiates a `moreh_sum` kernel (verified by grep across `ttnn/`). The `_metal2`-fork coordination problem does not arise.
- **Function-call escape is header-only and every shape is supported.** No Shape 4 (old addr-gen), no semaphore shapes, no `CircularBuffer&`.

| Op kernel(s) | Donor file | Class | Status |
|---|---|---|---|
| all 6 readers, `reader_moreh_sum_nc.cpp` | `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` | 3 — shared pool (`ttnn/cpp/ttnn/kernel/`) | ✓ |
| all 6 compute kernels | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | 3 — shared pool | ✓ |
| `reader_moreh_sum_w.cpp` | `ttnn/cpp/ttnn/kernel/dataflow/generate_mm_scaler.hpp` | 3 — shared pool | ✓ |
| `reader_moreh_sum_nc.cpp` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp` | 2 — official kernel lib | ✓ |
| `reader_moreh_sum_h.cpp` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` | 2 — official kernel lib | ✓ |
| `moreh_sum_h.cpp` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` | 2 — official kernel lib | ✓ |
| all kernels | `api/dataflow/*`, `api/tensor/*`, `api/compute/*` | 1 — LLK/HAL | ✓ no concern |

**Per-call detail** (no ⚠ / ✗ / ⭐ entries, included because the `DataflowBuffer`-by-value shape is not in the recipe's table and is worth recording):

| Donor function | Signature shape | Status |
|---|---|---|
| `generate_mask_h(DataflowBuffer, uint32_t)` — `moreh_common.hpp:183` | `DataflowBuffer` by value | ✓ excellent — see note |
| `generate_mask_w(DataflowBuffer, uint32_t)` — `:223` | `DataflowBuffer` by value | ✓ excellent |
| `generate_mm_scaler(DataflowBuffer, uint32_t)` — `generate_mm_scaler.hpp:12` | `DataflowBuffer` by value | ✓ excellent |
| `copy_tile_to_dst(DataflowBuffer, …)` / `pack_tile_from_dst(DataflowBuffer, …)` — `compute/moreh_common.hpp:1172,1185` | `DataflowBuffer` by value | ✓ excellent |
| `dataflow_kernel_lib::prepare_zero_tile<dfb_id>()` — `l1_helpers.hpp:57-58` | `uint32_t` CB id as NTTP | ✓ OK |
| `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<dfb_id, pool, dim>()` — `reduce_helpers_dataflow.hpp:84` | `uint32_t` CB id as NTTP | ✓ OK |
| `compute_kernel_lib::reduce<…, input_dfb_id, scaler_dfb_id, output_dfb_id, …>(…)` — `reduce_helpers_compute.hpp:381` | `uint32_t` CB ids as NTTPs; `Accumulate::at(uint32_t, …)` runtime | ✓ OK |
| `ArgFetcher::get_next_arg_val<T>()` — `moreh_common.hpp:44` | no resource handles | ✓ |

**Note on the `DataflowBuffer`-by-value shape** (absent from the recipe's shape table, which lists only `uint32_t cb_id` and `CircularBuffer&`): it bridges natively and needs **no** donor change. `DataflowBuffer` has a non-explicit `DataflowBuffer(DFBBindingToken)` constructor (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:72`), so a `dfb::name` token converts implicitly at the call site. This is a *better* case than the table's `uint32_t cb_id` row, not a worse one.

**Consequence worth stating for planning:** these donors are broadly shared (`kernel/dataflow/moreh_common.hpp` ≈ 64 consumers, `kernel/compute/moreh_common.hpp` ≈ 45, `kernel_lib/reduce_helpers_*` ≈ 53 each). Because every shape bridges natively, **this port should require zero donor-side edits** — the porter must not "helpfully" modernize a shared header, which would ripple across dozens of ops. No `_metal2` fork of any of these headers exists today (checked; none under `ttnn/cpp/ttnn/kernel*`).

### Relaxation candidates

None. The op has no custom hash to mine.

### TTNN factory analysis

Sheet rows and the code cross-check are tabulated under *Gate detail → TTNN factory concept*; both agree. What the port's TTNN ProgramFactory wiring needs: current concept `descriptor`, no op-owned tensors, no MeshWorkload need (genuine or artifact), no pybound internals or other migration-risky pybind, no custom hash, neither runtime-arg-update hook, no relaxation → target **`ProgramSpecFactoryConcept`**, plain. All six factories are identical in this respect, so one wiring pattern covers the op.

---

## Misc anomalies  *(team-only, non-gating, not porter work)*

- **Dead CB allocation** — `moreh_sum_nc_program_factory.cpp:95-103` allocates `c_24` that no kernel touches; the locals feeding it (`intermed0_t` line 60, `intermed_data_format` line 70, `intermed_tile_size` line 72) are dead with it. Also covered as PORT WORK above, since the port must drop it.
- **Unconditional mask/intermediate CB allocation** — `c_3` (H and W float, `c_1` in the int factories) and `c_25` are allocated regardless of whether masking is active, though every factory computes `do_mask_h` / `do_mask_w` before the CB block (e.g. `moreh_sum_h_program_factory.cpp:39`). Under `!do_mask_*` this is one to two tiles of L1 per core with no consumer. Minor waste, pre-existing.
- **Hardcoded tile geometry in the int32 writers** — `writer_moreh_int_sum_h.cpp:35-38` and `writer_moreh_int_sum_w.cpp:34-37` fold sub-tile faces with literal `16`, `4`/`8`, `256`/`512` strides, assuming a 32×32 int32 tile and a specific face layout. Correct for the only shape the op supports today, but silently wrong for any other tile size — no assert guards it.
- **`fp32_dest_acc_en` is forced but still hashed** — the three int factories override the caller's setting (`moreh_int_sum_h_program_factory.cpp:54-57`, `moreh_int_sum_w_program_factory.cpp:56-59`, `moreh_int_sum_nc_program_factory.cpp:52-55` all do `if (!fp32_dest_acc_en) { log_warning(...); fp32_dest_acc_en = true; }`). The un-forced value still rides `operation_attributes.compute_kernel_config` into the default program hash, so two INT32 calls differing *only* in `fp32_dest_acc_en` produce two distinct cache entries for what is the same program. Harmless but wasteful; also the classic shape of a relaxation candidate should the op ever gain a custom hash.
- **Unsigned underflow if `dims` is empty** — `moreh_sum.cpp:32`, `for (uint32_t i = dims.size() - 1; i > 0; i--)`. Unreachable in practice: `get_dim` (`moreh_helper_functions.cpp:452-467`) fills the vector with the full range when the input is absent or empty, so it is non-empty for any rank ≥ 1. Only a rank-0 input could reach it. Noted for completeness, not a live bug.
- **Non-`constexpr` CTA reads in the float H/W compute kernels** — `moreh_sum_h.cpp:10-12` and `moreh_sum_w.cpp:10-12` read `Ht` / `Wt` / `NC` into plain `uint32_t` while their int siblings use `constexpr`. This demotes compile-time-known values to runtime branches (`is_h_single_tile`, `is_w_single_tile`). Pre-existing; a small missed optimization, and it is what keeps those CBs' endpoints compile-time visible.

---

## Per-DeviceOperation attribution

Not applicable — one DeviceOperation. Findings that differ **per factory** are attributed inline throughout (the dead `c_24` is `MorehSumNCFactory`-only; the `c_25` question is `MorehSumHFactory`-only).

---

## Questions for the user

1. **`c_25` under `!do_mask_h` in `MorehSumHFactory` — drop conditionally, or align with the W kernel?**
   `moreh_sum_h_program_factory.cpp:120-128` allocates `c_25` unconditionally, but `moreh_sum_h.cpp:54` guards every access with `if constexpr (do_mask_h)`, leaving zero endpoints in the unmasked config — which Metal 2.0 cannot express. Two clean resolutions, and the choice is the ops team's, not the porter's: **(a)** make the DFB spec conditional on `do_mask_h` (already computed at `moreh_sum_h_program_factory.cpp:39`) *and* move the `DataflowBuffer dfb_masked_input_obj(...)` declaration at `moreh_sum_h.cpp:23` inside the guard; or **(b)** relax the guard to a plain `if`, matching `moreh_sum_w.cpp:71`, so the endpoint stays compile-time visible in both configs. (a) saves a tile of L1; (b) is the smaller diff and makes the H/W twins consistent.

2. **Confirm the dead `c_24` drop in `MorehSumNCFactory`.**
   The evidence is as strong as this check gets (no reference in any of the three kernels, no CTA, no RTA, no header, and the DST-accumulation design explains why it was never needed). Flagging it only because the recipe treats a wrongly-dropped live CB as the worst outcome a port can produce. Confirming that `moreh_sum_nc.cpp`'s DST accumulation is the intended design — and that no in-flight change reintroduces an L1 intermediate on that path — closes it.

---

## Recipe notes

1. **§Feasibility audit / §TTNN factory concept prerequisite — no procedure for "the sheet is unreachable."** The recipe covers the sheet being *wrong* or *silent* for an op (→ spreadsheet-broken, route to the owner), but not the connector being unavailable, which is a different failure with a different fix (the human authorizes access, or pastes the rows; nobody edits the sheet). The recipe also says "Fetch and locate. Pull a fresh copy of the sheet every run" without stating what to do when that fails, and separately labels the reference data "**recommended**" in the *Reference data* paragraph while the gate treats it as mandatory — those two framings pull in opposite directions when the fetch fails.

   What happened here: the connector was absent, so I ran every gate-bearing subject **and** all informational subjects (the op is `descriptor`-concept, so the *Red* scoping rule's "audit it in full" branch applies), reported the audit as RED-indeterminate on this one gate, and withheld the brief. The user then **pasted the six rows into the session**, which closed the gate and flipped the audit to GREEN with no other section changing. That worked well and cost one round-trip.

   Two suggestions: **(a)** add an explicit branch — *connector unavailable → complete the audit in full, mark this gate INDETERMINATE (not RED-blocked, which misroutes to a team), withhold the brief, and ask the human either to authorize the connector or to paste the op's rows.* **(b)** State whether user-pasted rows are acceptable in place of a fetched copy, and if so what to record about their provenance. I treated them as authoritative and noted the provenance in the Result; the recipe's insistence on a fresh fetch every run is aimed at staleness, and a paste from the live sheet has the same freshness property — but the recipe doesn't say so, and a stricter auditor might have refused it.

2. **§Device 2.0 prerequisite — compute-side LLK CB-index free functions are unaddressed.** The Green bullet sanctions `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)` by name, and the holdover definition keys on "a free function taking a `uint32_t` CB index where the wrapper object is in scope and a wrapper-method replacement exists." Compute kernels are full of `copy_tile(cb, …)`, `pack_tile(dst, cb)`, `matmul_tiles(cb_a, cb_b, …)`, `binary_op_init_common(cb, cb, cb)` — CB-index free functions with the wrapper in scope. I judged them out of scope, since Device 2.0 is the *data-movement* migration and the guide never mentions them, and no `DataflowBuffer` **method** replaces them. But this op muddies it: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp:1172,1185` provides `copy_tile_to_dst(DataflowBuffer, …)` / `pack_tile_from_dst(DataflowBuffer, …)`, the int kernels use them, and the float H/W kernels call raw `copy_tile`/`pack_tile` — so a *DFB-taking alternative demonstrably exists* and a stricter reading of the holdover rule could RED this op. One sentence in the Green bullet ("compute LLK APIs taking a CB index are outside the Device 2.0 data-movement boundary, regardless of whether a DFB-taking kernel_lib wrapper exists") would remove the ambiguity. This was the single highest-stakes judgment call in the audit.

3. **§CB endpoints — the census needs a stated rule for compile-time-eliminated references.** Whether an endpoint is counted by *code reference* or by *runtime execution* decides real dispositions, and the recipe does not say. In this op the answer flips `c_25`: `moreh_sum_h.cpp:54` (`if constexpr`) eliminates the reference, while `moreh_sum_w.cpp:71` (plain `if`) keeps it — same logical construct, different census. I adopted "a code reference is a toucher, because Metal 2.0 bindings are static; `#ifdef` / `if constexpr` elimination removes the reference, a plain `if` does not," and stated it in the report. Worth promoting into the *endpoint census* paragraph, since `#ifdef`-guarded CB access is extremely common in dataflow kernels.

4. **§Out-of-directory coupling — the shape table has no row for `DataflowBuffer` by value.** It lists `uint32_t cb_id` (✓ OK) and `CircularBuffer` / `CircularBuffer&` (⭐ flag), but every donor in this op takes `DataflowBuffer` **by value**. It reads like the `CircularBuffer` row (the DFB-era name for the same wrapper) and would be flagged by a literal-minded auditor — yet it is the *best* case: `dataflow_buffer.h:72` has a non-explicit `DataflowBuffer(DFBBindingToken)` constructor, so `dfb::name` converts implicitly with no donor change. Suggest adding a `DataflowBuffer` / `DataflowBuffer&` row marked ✓ excellent, with that constructor as the reason.

5. **Minor — the audit template has no natural home for a "runtime-selected DFB handle" finding.** `moreh_sum_w.cpp` reassigns a mutable `cb_input` between two CB indices and constructs `DataflowBuffer(cb_input)` from it; `moreh_int_sum_nc.cpp:39-40` picks its output CB with a ternary. Neither is in Appendix A (so, correctly, not a gate) and neither is a CB-endpoint, tensor-binding, or vararg finding — but both are exactly the kind of thing a porter needs warned about, since `dfb::name` tokens are static. I put them under Heads-ups. A sentence in §RTA varargs' neighbourhood, or a named Heads-ups bullet ("runtime-selected resource handles"), would give this a home.
