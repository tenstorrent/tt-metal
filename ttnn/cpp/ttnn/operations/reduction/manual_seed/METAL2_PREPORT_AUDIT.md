# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/reduction/manual_seed`

One DeviceOperation, four program factories, all defined in one file:

- **`ManualSeedDeviceOperation`** (`device/manual_seed_operation.hpp` / `.cpp`)
  - `ManualSeedSingleSeedToAllCoresProgramFactory` (`device/manual_seed_program_factory.cpp:58`) — scalar seed, no user IDs
  - `ManualSeedSingleSeedSingleCoreProgramFactory` (`device/manual_seed_program_factory.cpp:82`) — scalar seed, scalar user ID
  - `ManualSeedSingleSeedSetCoresProgramFactory` (`device/manual_seed_program_factory.cpp:108`) — scalar seed, `user_ids` tensor
  - `ManualSeedSetSeedsSetCoresProgramFactory` (`device/manual_seed_program_factory.cpp:176`) — `seeds` tensor, `user_ids` tensor

Five kernel files, all owned by this op and all referenced by a factory (no unreferenced/dead kernel files in the directory):

| Kernel | Used by |
|---|---|
| `device/kernels/compute/manual_seed_set_seed.cpp` | factories 1, 2 |
| `device/kernels/dataflow/reader_manual_seed_read_user_id.cpp` | factory 3 |
| `device/kernels/compute/manual_seed_single_seed_receive_user_id.cpp` | factory 3 |
| `device/kernels/dataflow/reader_manual_seed_read_all_data.cpp` | factory 4 |
| `device/kernels/compute/manual_seed_receive_all_data.cpp` | factory 4 |

Single DeviceOperation, so no per-DeviceOperation bundling is needed. Findings that differ *per factory* are attributed inline throughout.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `4d6e9518bf5 2026-07-30 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/reduction/manual_seed` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `ManualSeedDeviceOperation` → `SingleSeedToAllCores`, `SingleSeedSingleCore`, `SingleSeedSetCores`, `SetSeedsSetCores` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all five kernels are structurally Device 2.0 (`Noc`, `DataflowBuffer`, `CoreLocalMem`, `TensorAccessor`). No holdovers. See Gate detail for one considered-and-cleared free function. |
| *Prereqs* — Cross-op escapes | **Ok** — none. Every kernel is owned by this op; every `#include` resolves to `tt_metal/*`. |
| *Feature Support* — overall | **GREEN** — all four Appendix A entries `N/A` |
| *Feature Support* — Variadic-CTA | **Ok** — every `get_compile_time_arg_val` read is at a literal constexpr offset |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** — `yes` on all four factory rows |
| *TTNN Readiness* — Concept (current) | `descriptor` (all four factories) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A — concept is `descriptor` |
| *TTNN Readiness* — Is safe to port? | Yes (all four rows) |
| *TTNN Readiness* — Custom hash | No — no `compute_program_hash` anywhere in the op |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No — hook absent from the device-op |
| *TTNN Readiness* — `override_runtime_arguments` | No — method absent |
| *TTNN Readiness* — Pybind `create_descriptor` | No — `manual_seed_nanobind.cpp:73-81` binds only the top-level `manual_seed` function |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` |
| *Port work* — Offset base pointer | **none** — the op contains no `->address()` expression at all |
| *Port work* — Tensor bindings (per binding) | Case 1 × 3 (`user_ids` in factories 3 and 4; `seeds` in factory 4). Factories 1 and 2 have no tensor bindings. |
| *Port work* — TensorParameter relaxation | none (sheet: `none` on all four rows) |
| *Port work* — TensorAccessor 3rd arg | **none** — every accessor is constructed with two arguments |
| *Port work* — CB endpoints | self-loop × 3 · legal 1:1 × 2 · no dead CBs · no multi-binding |

**CB endpoints** are dispositions, not gates: every out-of-window CB has a port-time resolution. The dispositions here are recorded per `(CB, factory)` in the Gate detail section below.

## Result

**GREEN → brief issued.**

All five gates cleared. `manual_seed` is a small, structurally simple op that is already on the `ProgramDescriptor` API with fully Device-2.0 kernels, uses no Appendix A feature, folds no offsets into any pointer, and passes no page-size override to any `TensorAccessor`. The readiness sheet independently reports `Is able to port? = yes` for all four factories, and the cross-check against the code agrees on every column.

Port work is light and mechanical: three Case-1 tensor bindings, three self-loop DFBs, and two legal 1:1 DFBs. There is no code-path scoping to do — no factory is blocked, so the whole op is portable as one unit.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** The readiness sheet carries exactly four rows for `reduction/manual_seed`, one per factory, and all four read `Is able to port? = yes`. Every conjunct is `no`/clear: `Custom hash = no`, `Runtime-args update (get_dynamic_runtime_args) = no`, `Override runtime args method? = no`, `Pybind descriptor = no`, `Smuggled pointer = no`, `Is safe to port? = yes`, `Concept = descriptor`.

  Cross-check against the code — clean on every cheaply-checkable column:

  | Column | Sheet | Code evidence |
  |---|---|---|
  | `Concept` | `descriptor` | all four factories declare `static ProgramDescriptor create_descriptor(...)` @ `device/manual_seed_program_factory.hpp:19,24,29,34` |
  | `Custom hash` | `no` | no `compute_program_hash` in the op directory (grep clean) |
  | `get_dynamic_runtime_args` | `no` | hook absent from `ManualSeedDeviceOperation` @ `device/manual_seed_operation.hpp:16-34` |
  | `Override runtime args method?` | `no` | no `override_runtime_arguments` in the op directory |
  | `Pybind descriptor` | `no` | `manual_seed_nanobind.cpp:73-81` binds `&manual_seed_wrapper` only — no `create_descriptor` binding, no `nb::class_` of the device-op |
  | `Op-owned tensors?` | blank (no) | consistent with the `descriptor` concept — no `WorkloadDescriptor`, no `buffers` vector |
  | `Secretly SPMD Workload?` | blank | N/A — only meaningful on `WorkloadDescriptor` |
  | **Factory-set match** | 4 rows | 4 factories in `program_factory_t` @ `device/manual_seed_operation.hpp:21-25`; names match one-to-one, no phantom and no missing row |

  Cross-column invariants hold: `get_dynamic_runtime_args = no` on a `descriptor` concept is legal, and no op-owned tensors on a `descriptor` row is the required state. `Is safe to port?` was not re-derived (the readiness-sheet owner's judgment axis).

  The sheet's `Porting Target` column independently reads `ProgramSpecFactoryConcept`, agreeing with the target derived in the TTNN porting shape subject below.

- **Device 2.0 (every kernel used):** **GREEN.** All five kernels are structurally Device 2.0. No violations, so no table of holdovers.

  Evidence per kernel:

  | Kernel | Device 2.0 idioms present |
  |---|---|
  | `reader_manual_seed_read_user_id.cpp` | `Noc noc` @ `:32`; `DataflowBuffer` @ `:33-34`; `TensorAccessor` @ `:30`; `CoreLocalMem<volatile uint32_t>` @ `:45,55`; `noc.async_read(...)` / `noc.async_read_barrier()` @ `:39-41` |
  | `reader_manual_seed_read_all_data.cpp` | `Noc noc` @ `:39`; `DataflowBuffer` @ `:40-42`; `TensorAccessor` @ `:34,37`; `CoreLocalMem` @ `:60,61,72`; `noc.async_read` @ `:47,54` |
  | `manual_seed_single_seed_receive_user_id.cpp` | `DataflowBuffer` @ `:22`; `wait_front` / `read_tile_value` / `pop_front` @ `:25,28,38` |
  | `manual_seed_receive_all_data.cpp` | `DataflowBuffer` @ `:21`; `wait_front` / `read_tile_value` / `pop_front` @ `:24,27,32,42` |
  | `manual_seed_set_seed.cpp` | no data movement at all — one CTA read and `rand_tile_init` @ `:10-16`; nothing for Device 2.0 to migrate |

  No legacy Device 1.0 idiom appears anywhere in the op: no `noc_async_read`, no `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedAddrGenFast` / `InterleavedPow2AddrGen*`, no raw semaphore addresses, no manual CB index management, no `CircularBuffer` wrapper, and no `api/dataflow/circular_buffer.h` include.

  **One free function considered and cleared — `get_dataformat(<cb_index>)`,** at `reader_manual_seed_read_user_id.cpp:29` and `reader_manual_seed_read_all_data.cpp:33,36`. This is a CB-index-keyed free function that is *not* on the audit's explicitly sanctioned list (which names only `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)`), so it needed a call. Not flagged as a holdover, for four independent reasons:

  1. **The holdover shape test is not met.** The rule requires the Device-2.0 wrapper object to be *already in scope at the call site*. At all three sites the `DataflowBuffer` is declared **after** the call (`:33` vs. the call at `:29`; `:40-41` vs. the calls at `:33,36`).
  2. **Its relationship to the wrapper is identical to `get_tile_size`'s** — the sanctioned case. `CircularBuffer::get_dataformat()` @ `tt_metal/hw/inc/api/dataflow/circular_buffer.h:115` is a one-line forward to `::get_dataformat(cb_id_)`, exactly as the audit describes for `get_tile_size`. The audit's instruction is to "check the current Device 2.0 surface rather than assuming the shape alone makes it a holdover"; the current surface treats the two the same way.
  3. **Device 2.0's own migrated code uses it.** The free form appears at ~91 sites across `ttnn/cpp/ttnn/operations` kernels, including many already on `api/dataflow/noc.h` + `api/dataflow/dataflow_buffer.h`. Per the Green bullet's "if Device 2.0 allows the free function, so do we," that is the evidence standard.
  4. **The Device 2.0 migration guide never mentions it** — zero hits for `get_dataformat` in `device_api_migration_guide.md`, so it is not in the guide's migrate-these set. (The guide *does* use `get_tile_size(cb_id)` in its own migrated examples, at `:605` and `:630`.)

  Independently of the gate call: **all three of these `constexpr DataFormat` variables are dead** — declared and never read. Recorded under Misc anomalies. See also Recipe notes, which asks for the sanctioned list to name `get_dataformat` explicitly so the next auditor doesn't have to re-derive this.

  `get_tile_size(<cb_index>)` also appears, at `reader_manual_seed_read_user_id.cpp:40` and `reader_manual_seed_read_all_data.cpp:48,54`. Explicitly sanctioned at the Device 2.0 stage → not a violation. (A Metal 2.0 port does move these onto the object per kernel-side whitelist rule 7 — carried to the brief as port work.)

- **Feature compatibility:** all four Appendix A entries are absent. Grep across the whole op directory for every recognition signal — `GlobalCircularBuffer`, `CreateGlobalCircularBuffer`, the `global_circular_buffer` field on a `CBDescriptor`, `remote_index` / `remote_cb` identifiers, `address_offset`, `set_address_offset`, `UpdateDynamicCircularBufferAddress`, `cb_descriptor_from_sharded_tensor`, `GlobalSemaphore`, `CreateGlobalSemaphore` — returns zero matches. The op declares no semaphores of any kind.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No reference to the type, the factory call, or the `global_circular_buffer` field. The three `CBDescriptor`s built via `push_tensor_circular_buffer` @ `device/manual_seed_program_factory.cpp:45-53` set only `total_size`, `core_ranges`, and one `CBFormatDescriptor` — the plain supported path. |
  | CBDescriptor `address_offset` (non-zero) | N/A | The field is never set on any descriptor (defaults to 0). No imperative `set_address_offset`, no four-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor` call. |
  | GlobalSemaphore | N/A | The op creates no semaphores at all — `ProgramDescriptor::semaphores` is never touched in any of the four factories. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | Both op-level and kernel-level signals are negative. `tensor_args_t` is `ManualSeedInputs` @ `device/manual_seed_device_operation_types.hpp:19-22` — two `std::optional<Tensor>` fields, a fixed-count set, not a variable-count container. And the decider is clean: all 11 `get_compile_time_arg_val` reads across the five kernels are at **literal constexpr offsets** (`(0)`, `(1)`, `(2)`, `(3)`), never a runtime-varying index. The `TensorAccessorArgs<N>` templates @ `reader_manual_seed_read_user_id.cpp:23` and `reader_manual_seed_read_all_data.cpp:25-27` are constexpr-offset chained (`next_compile_time_args_offset()`), which is the standard fixed-count accessor-args pattern, not a CTA loop. |

- **CB endpoints (GATE-free):** no gate; every CB has a port-time disposition. Census run per CB, per node, per factory. All CBs span the whole `core_grid`, so every node sees the same census — no per-node variation within a factory.

  | Factory | CB | Touchers on a node | Verdict | Disposition |
  |---|---|---|---|---|
  | `SingleSeedToAllCores` | — | no CBs declared | — | — |
  | `SingleSeedSingleCore` | — | no CBs declared | — | — |
  | `SingleSeedSetCores` | `c_0` `user_ids_cb_index` @ `program_factory.cpp:127-128` | **1** — reader only (`reserve_back` @ `reader_manual_seed_read_user_id.cpp:37`, `get_write_ptr` @ `:38`, NoC read destination @ `:39-40`) | single-ended | **self-loop** |
  | `SingleSeedSetCores` | `c_1` `kernel_communication_cb_index` @ `program_factory.cpp:130-131` | **2** — reader locked producer (`reserve_back` @ `:54`, `push_back` @ `:59`); compute locked consumer (`wait_front` @ `manual_seed_single_seed_receive_user_id.cpp:25`, `pop_front` @ `:38`) | plain 1:1 | **legal — no action** |
  | `SetSeedsSetCores` | `c_0` `user_ids_cb_index` @ `program_factory.cpp:200-201` | **1** — reader only (`reserve_back` @ `reader_manual_seed_read_all_data.cpp:45`, `get_write_ptr` @ `:46`, NoC read destination @ `:47-48`) | single-ended | **self-loop** |
  | `SetSeedsSetCores` | `c_1` `seeds_cb_index` @ `program_factory.cpp:203-204` | **1** — reader only (`reserve_back` @ `:52`, `get_write_ptr` @ `:53`, NoC read destination @ `:54`) | single-ended | **self-loop** |
  | `SetSeedsSetCores` | `c_2` `kernel_communication_cb_index` @ `program_factory.cpp:206-207` | **2** — reader locked producer (`reserve_back` @ `:71`, `push_back` @ `:77`); compute locked consumer (`wait_front` @ `manual_seed_receive_all_data.cpp:24`, `pop_front` @ `:42`) | plain 1:1 | **legal — no action** |

  **No dead CBs.** Every declared CB index is positively referenced by a kernel: the factory threads each index through the reader's and compute kernel's CTA lists, and each is consumed at a named `constexpr uint32_t ..._dfb_index = get_compile_time_arg_val(n)` that then constructs a `DataflowBuffer`. Verified in every factory and every kernel — no index reaches a kernel only through indirection, and there is no config in which a CB goes unreferenced.

  **No multi-binding anywhere.** All three faces of the hunt come back negative:

  - *(a) Hidden second writer* — actively scanned. Only two kernels exist per factory (one reader, one compute), and the compute kernel touches **only** the `kernel_communication` CB: `manual_seed_single_seed_receive_user_id.cpp` declares exactly two CTAs (the communication index and the seed) and `manual_seed_receive_all_data.cpp` exactly one (the communication index). Neither compute kernel receives the `user_ids` or `seeds` CB index at all, so it structurally cannot raw-write into them. There are no semaphores in the op, so the semaphore-gated co-fill pattern has no coordination mechanism to hide behind.
  - *(b) Multiple readers* — negative. No CB is read by two co-resident kernels. Each `user_ids` / `seeds` CB is written and then read back via `CoreLocalMem` **within the same reader kernel** (`reader_manual_seed_read_all_data.cpp:60-61`), which is one toucher's peek on its own buffer, not a second endpoint. None of these CBs is borrowed-memory.
  - *(c) Dual-instance work-split* — negative. Each factory pushes each `kernel_source` into exactly **one** `KernelDescriptor`. No kernel source is instantiated twice.

  A note on the three self-loops, since the shape recurs: each is a NoC read landing area that the same reader kernel immediately reads back through `CoreLocalMem` at the write pointer, with no FIFO handoff to any other kernel — the canonical single-ended / sync-free scratchpad. The reader is a locked producer (`reserve_back`), which the self-loop's PRODUCER-and-CONSUMER binding satisfies. Legal on Gen1 for DM kernels; the Gen2 DM-self-loop restriction is Quasar-uplift's concern, not a Gen1 blocker.

- **Offset base pointers:** **GREEN — no fold, and no address RTA to fold into.** The op contains **zero** `->address()` / `.address()` expressions (grep clean across the whole directory), so there is no host-side pointer arithmetic that could hide an offset. Tensor bases reach the kernels through the framework instead: `emplace_runtime_args(core, {user_ids_mesh, core_id})` @ `device/manual_seed_program_factory.cpp:157` and `emplace_runtime_args(core, {user_ids_mesh, seeds_mesh, core_id})` @ `:235` pass `MeshTensor` references, which the `RTArgList` overload (`tt_metal/api/tt-metalium/program_descriptors.hpp:176,200-202`) auto-registers as buffer bindings — the framework resolves the base, so the factory never sees or manipulates an address.

  Reconciling against the dated triage: `manual_seed` does not appear in `2026-07-19_offset_base_pointers.md`. That is the *no fold, op not in the tables* outcome — clean, and confirmed by scan rather than inferred from the doc's silence. Types 3 and 4 are likewise absent (no `address_offset`, no `ttnn::narrow`, no interior-base `MeshBuffer::create`).

- **TensorAccessor 3rd argument:** **GREEN — the subject does not fire.** Every `TensorAccessor` in the op is constructed with exactly **two** arguments, so no page-size override exists to classify:

  - `reader_manual_seed_read_user_id.cpp:30` — `TensorAccessor(user_ids_tensor_accessor_args, user_ids_tensor_buffer_addr)`
  - `reader_manual_seed_read_all_data.cpp:34` — `TensorAccessor(user_ids_tensor_accessor_args, user_ids_tensor_buffer_addr)`
  - `reader_manual_seed_read_all_data.cpp:37` — `TensorAccessor(seeds_tensor_accessor_args, seeds_tensor_buffer_addr)`

  Consistent with the dated triage doc, which does not list `manual_seed` — but established by direct read of all three construction sites, not by the doc's silence.

  **Guard against a false match:** the third argument to `noc.async_read(accessor, dfb, get_tile_size(...), {.page_id = 0}, {.offset_bytes = 0})` @ `reader_manual_seed_read_user_id.cpp:39-40` and `reader_manual_seed_read_all_data.cpp:47-48,54` is a **transfer size on the NoC call**, not a page size on an accessor constructor. Not this subject. (The magnitude of that size argument is a separate observation — see Misc anomalies.)

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per factory):
  - `SingleSeedToAllCores`, `SingleSeedSingleCore` — **no tensor bindings.** Both build a single compute kernel with one CTA and no CBs; `tensor_args` is unused in both (`device/manual_seed_program_factory.cpp:59,83`).
  - `SingleSeedSetCores` — `user_ids` → **Case 1**. Delivered as a `MeshTensor` reference in the RTA list @ `:157`; the kernel reads the base at `get_arg_val<uint32_t>(0)` @ `reader_manual_seed_read_user_id.cpp:16` and feeds it straight into `TensorAccessor` @ `:30`, doing all memory access through the accessor.
  - `SetSeedsSetCores` — `user_ids` → **Case 1** (base at `get_arg_val<uint32_t>(0)` @ `reader_manual_seed_read_all_data.cpp:16` → `TensorAccessor` @ `:34`); `seeds` → **Case 1** (base at `get_arg_val<uint32_t>(1)` @ `:17` → `TensorAccessor` @ `:37`).

  All three are the `Buffer*`-binding delivery shape (via the `MeshTensor` overload), so they are **correct on cache hits today** — the framework patches these bindings, unlike a raw `->address()` RTA. This is routine port work, not a correctness hazard. The port replaces the delivery with a typed `TensorParameter` and the kernel-side plumbing (`TensorAccessorArgs` CTAs + the address RTA) disappears.

  No binding is clean-via-borrowed-DFB: the causal-link gate does not apply anywhere here. The `user_ids` / `seeds` CBs are plain L1 CBs used as NoC read destinations, not borrowed-memory CBs — `set_globally_allocated_address` appears nowhere in the op, and no `CBDescriptor` names a backing buffer.

- **TensorParameter relaxation:** none. The sheet reports `none` on all four rows, consistent with `Custom hash = no` (a relaxation co-occurs with a custom hash, and there is none).
- **TensorAccessor 3rd arg:** none — no site passes one.
- **CB endpoints:** self-loop on `(c_0, SingleSeedSetCores)`, `(c_0, SetSeedsSetCores)`, `(c_1, SetSeedsSetCores)`. Legal 1:1 on `(c_1, SingleSeedSetCores)` and `(c_2, SetSeedsSetCores)` — no action. No dead-CB drops, no multi-binding flags.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. All three faces scanned negative — see Gate detail for the reasoning per face.
- **Cross-op / shared kernels:** no *cross-op* sharing — the op owns all five kernel files, borrows none, and no other op binds any of them, so there is no sunset list. **But one kernel is shared intra-op:** `device/kernels/compute/manual_seed_set_seed.cpp` is bound by **two** factories — `SingleSeedToAllCores` @ `device/manual_seed_program_factory.cpp:68-69` and `SingleSeedSingleCore` @ `:94-95`. That is the *intra-op* shape of the shared-kernel caution, so which rung applies depends on how the port is scoped:
  - **Porting all four factories in one change** (the natural unit — nothing is blocked, so there is no reason to split): both binders convert together, so the kernel converts **in place**, no fork. The two `KernelSpec`s differ only in core ranges and both pass the same single CTA (`seed`), so one converted kernel serves both without divergence.
  - **Porting the factories piecemeal**: the kernel would need an `_metal2` fork beside the original, since the un-ported factory keeps binding the legacy copy.

  No `_metal2` sibling exists for any of the five kernels today (checked locationally in both kernel directories), so nothing is available to reuse.
- **RTA varargs:** none. Every RTA is read at a fixed literal index and maps to a nameable field: `reader_manual_seed_read_user_id.cpp:16-17` reads indices 0 and 1 into `user_ids_tensor_buffer_addr` and `core_id`; `reader_manual_seed_read_all_data.cpp:16-18` reads indices 0, 1, 2 into `user_ids_tensor_buffer_addr`, `seeds_tensor_buffer_addr`, `core_id`. No counted loop over args, no data-selected index, no running `arg_index++`. This is the preferred non-signal case — the porter names each. The op sets no common runtime args.

## Team-only

- **Out-of-directory coupling & donor shape:** **✓ clean** — the strongest possible result on this subject. Nothing to inventory:
  - *Function-call escape:* none. Every `#include` in all five kernels resolves to `tt_metal/*` (donor class 1, "no concern"): `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/dataflow_buffer.h`, `api/core_local_mem.h`, `api/tensor/noc_traits.h`, `api/compute/compute_kernel_api.h`, `api/compute/common.h`, `api/compute/eltwise_unary/rand.h`, `ckernel.h`, `ckernel_defs.h`, `<tt-metalium/constants.hpp>`, `<cstdint>`. No include reaches `ttnn/cpp/ttnn/kernel_lib/`, `ttnn/cpp/ttnn/kernel/`, `operations/kernel_helper_functions/`, another reduction-family op, or another op family. So there are no donor functions to classify by handle shape, and the summary table and per-call detail are omitted per the report format (all rolls ✓).
  - *File-path kernel instantiation:* none borrowed. All five `kernel_source` paths point inside `ttnn/cpp/ttnn/operations/reduction/manual_seed/device/kernels/`. Census run per kernel filename across `ttnn/cpp/ttnn/operations/`: the only binding hit for each of the five is this op's own `manual_seed_program_factory.cpp` (the other hits are prose in `docs/Manual_seed.md` and this audit file, both discarded per the census rule). So there is **no cross-op coordination cost and no sunset list**. No `_metal2` sibling exists in either kernel directory.

    The one sharing relationship is **intra-op**: `manual_seed_set_seed.cpp` is bound by two of this op's own factories (`SingleSeedToAllCores` @ `:68-69` and `SingleSeedSingleCore` @ `:94-95`). Detailed in Heads-ups above — it decides between in-place conversion and an intra-op fork depending on whether the port covers all four factories at once.
- **Relaxation candidates** (mined from a custom hash): N/A — the op has no custom hash to mine.
- **TTNN factory analysis:** the sheet-derived facts with code evidence are in the Gate detail cross-check table above. Summarized: concept `descriptor` on all four factories; no op-owned tensors; no MeshWorkload (so the genuine-vs-artifact question does not arise); no pybind of internals — `manual_seed_nanobind.cpp` exposes only the top-level `manual_seed` function via `ttnn::bind_function<"manual_seed">` @ `:73`, with a plain wrapper @ `:24-30`; no custom hash; no `get_dynamic_runtime_args`; no `override_runtime_arguments`. Target concept `ProgramSpecFactoryConcept`, which the sheet's `Porting Target` column independently confirms.

  **TTNN porting shape (FYI-P):** target is `ProgramSpecFactoryConcept`, derived from `Concept == descriptor` with no op-owned tensors — the common case. All four factories map to the same target, so the port's TTNN ProgramFactory wiring is uniform across them.

## Misc anomalies  *(team-only, non-gating, not porter-actionable)*

These are latent code issues noticed while reading the op. They route to the ops team; the port should not act on them.

1. **Three dead `constexpr DataFormat` variables.** `user_ids_tensor_data_format` @ `device/kernels/dataflow/reader_manual_seed_read_user_id.cpp:29`; `user_ids_tensor_data_format` @ `device/kernels/dataflow/reader_manual_seed_read_all_data.cpp:33`; `seeds_tensor_data_format` @ `device/kernels/dataflow/reader_manual_seed_read_all_data.cpp:36`. Each is computed from `get_dataformat(<dfb_index>)` and never read. Harmless (constexpr, no codegen), but deleting them would also remove the only reason the `get_dataformat` free-function question came up in the Device 2.0 gate.

2. **NoC read size is a tile size, but the tensors are row-major — likely over-read, and wrong for any tensor spanning more than one page.** The reads @ `reader_manual_seed_read_user_id.cpp:39-40` and `reader_manual_seed_read_all_data.cpp:47-48,54` transfer `get_tile_size(<dfb_index>)` bytes — 4096 for UINT32 (32×32×4) — from `page_id = 0`. But `validate_on_program_cache_hit` @ `device/manual_seed_operation.cpp:66,71,82` requires both tensors to be **rank-1 ROW_MAJOR UINT32**, so a tensor of, say, 32 elements is a single 128-byte page. Two consequences:
   - The read pulls ~3968 bytes past the tensor. Benign in effect (the destination CB is sized `tensor_tile_size` @ `device/manual_seed_program_factory.cpp:46`, so nothing overflows, and the kernel only consumes `number_of_ids` elements), but it is a real over-read of DRAM.
   - If a tensor ever exceeded one page, this would silently read **wrong** data, not just extra data: for a row-major interleaved tensor, the bytes following page 0 in its bank belong to non-adjacent logical pages, so elements past the first page would come back scrambled. Nothing in the current validation caps the tensor at one page — the rank-1 check does not bound the volume. Worth an explicit bound or a paged read loop.

3. **`out_num_cores` goes stale when `sub_core_grids` is supplied.** `compute_core_grid` @ `device/manual_seed_program_factory.cpp:20-36` sets `out_num_cores` from the **full** device grid @ `:24`, then overrides `core_grid` with `sub_core_grids` @ `:31-33` **without updating `out_num_cores`**. All three callers that use the count then pass the stale value to `corerange_to_cores(core_grid, num_cores, true)` @ `:88,119,189`. Benign today only because `max_cores` in `corerange_to_cores` (`tt_metal/common/core_coord.cpp:599-613`) only ever *truncates* — and a sub-grid is by definition no larger than the full grid, so the truncation never fires. The two values are nonetheless describing different grids, which is fragile.

4. **Unhandled `std::out_of_range` on the scalar-user-id path.** `cores.at(operation_attributes.user_ids.value_or(0))` @ `device/manual_seed_program_factory.cpp:89`. Validation caps the scalar `user_ids` at 31 @ `device/manual_seed_operation.cpp:92-94`, but does not check it against the *available* core count. With a `sub_core_grids` of fewer than 32 cores, `.at(31)` throws a raw `std::out_of_range` instead of a `TT_FATAL` with a useful message. The nanobind docstring @ `manual_seed_nanobind.cpp:46` already tells users a custom core range set is expected for multi-user execution, which makes the small-sub-grid case reachable in normal use.

5. **Raw `MeshDevice*` in the hashed attributes struct.** `ManualSeedParams::device` @ `device/manual_seed_device_operation_types.hpp:13` is a raw pointer living in `operation_attributes_t`, so it feeds the default program hash by pointer value. Used host-side only (for `compute_with_storage_grid_size` @ `device/manual_seed_program_factory.cpp:23`), so it is not a smuggled device pointer and the readiness sheet's `Smuggled pointer = no` is correct. Flagged only because a pointer in the cache key is unusual and the field is not otherwise part of the op's identity.

6. **Silent fallback on an unmatched factory selection.** `select_program_factory` @ `device/manual_seed_operation.cpp:44-48` logs a warning and defaults to `ManualSeedSingleSeedToAllCoresProgramFactory` when no case matches, describing its own state as a "Logic error". The `validate_*` methods should already make the unmatched combination unreachable, which suggests this branch is either dead or masking a validation gap — a `TT_THROW` would be more honest than a warning plus a factory that would then read a `seeds` value that may be `nullopt`.

## Questions for the user  *(none)*

Every finding resolved from the code; nothing needed to be gated conservatively for lack of information.

## Recipe notes

Two points of friction, both minor and both in the same spirit — an explicit list that a real op fell just outside of.

1. **The Device 2.0 sanctioned-free-function list reads as closed, but is missing `get_dataformat(cb_id)`.** The [Device 2.0 prerequisite](metal2_audit.md#device-20-prerequisite) Green bullet says *"Currently sanctioned (do **not** flag): `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)`"* — a two-item list, phrased definitively. This op uses `get_dataformat(<cb_index>)` at three sites, which is not on that list but is *structurally indistinguishable* from `get_tile_size`: same single-CB-index signature, and `CircularBuffer::get_dataformat()` @ `tt_metal/hw/inc/api/dataflow/circular_buffer.h:115` forwards to the free function exactly as the bullet's own breadcrumb describes for `get_tile_size`. It also has ~91 callers across ttnn kernels, many already Device 2.0.

   The recipe *does* give the escape hatch to resolve this — "check the current Device 2.0 surface rather than assuming the shape alone makes it a holdover," plus "if Device 2.0 allows the free function, so do we" — and the holdover shape test independently fails here (the wrapper is declared after the call site at all three sites). So I reached GREEN with reasonable confidence. But the call was mine to make, on a **gate**, which the RED/GREEN-boundary note in the intro asks me to surface. Suggestion: either add `get_dataformat(cb_id)` to the sanctioned list, or replace the two-item enumeration with the *criterion* (a free function whose Device-2.0 wrapper method is a pure forward, and which Device 2.0's own migrated code still uses) and keep the two names as examples. The criterion generalizes; the list will keep getting overtaken by real ops.

2. **Readiness-sheet column header drift on the `override_runtime_arguments` gate.** Both `metal2_audit.md` and `ttnn_op_porting_readiness.md` name the column **`Override runtime args method? (PD and legacy)`**. The live sheet's header today reads **`Override runtime args method? (PD only)`**. Same gate, same meaning, parenthetical changed. It cost nothing here (the value is `no` on all four rows, and header-name lookup found it unambiguously), and it is not a "spreadsheet is broken" conflict — the *value* agrees with the code. Flagging it only because `ttnn_op_porting_readiness.md:48` states as a standing guarantee that *"existing column names never change"*, and a strict name-match lookup would miss this column. Worth reconciling the doc with the sheet, or loosening the guarantee to "the leading part of the name is stable."

   The live sheet also carries several columns neither doc enumerates — `Op Classification`, `Execution Model`, `Porting Target`, `Known op issues`, `Pointer patching perf issue?`. That is expected and covered ("The sheet may carry other, informational columns"), and two were genuinely useful: `Porting Target` independently confirmed `ProgramSpecFactoryConcept`, and `Op Classification = PD Op (pointer-patching)` corroborated the `MeshTensor`-binding RTA delivery I found in the code. Both might be worth naming in the readiness doc's column list as confirmatory cross-checks.
