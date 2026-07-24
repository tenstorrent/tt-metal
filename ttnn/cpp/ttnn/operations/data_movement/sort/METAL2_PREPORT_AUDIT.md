# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/sort`

**Device operation:** one — `ttnn::prim::SortDeviceOperation` — with three program factories (selected at runtime by `Wt` in `select_program_factory`):

- **`SortDeviceOperation`**
  - `SortProgramFactorySingleRowSingleCore` — `create_descriptor()` → `ProgramDescriptor` (`sort_program_factory.cpp:21`). Kernels: `reader_single_row_single_core.cpp`, `writer_single_row_single_core.cpp`, `sort_single_row_single_core.cpp`.
  - `SortProgramFactorySingleRowMultiCore` — `create_descriptor()` → `ProgramDescriptor` (`sort_program_factory.cpp:969`). Kernels: `coordinator_single_row_multi_core.cpp`, `reader_single_row_multi_core.cpp`, `writer_single_row_multi_core.cpp`, `sort_single_row_multi_core.cpp`.
  - `SortProgramFactoryCrossCoreDataExchange` — `create_workload_descriptor()` → `WorkloadDescriptor` (`sort_program_factory.cpp:902`), op-owned lookup-table tensor. Kernels: `reader_cross_core_data_exchange.cpp`, `writer_cross_core_data_exchange.cpp`, `sort_cross_core_data_exchange.cpp`.

Shared kernel headers (in-directory): `compute/sort_common.hpp`, `dataflow/sort_dataflow_common.hpp`, `dataflow/cross_core_data_exchange_common.hpp`. No unreferenced kernel files in the directory.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `44da718b06b 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/sort` |
| **Overall** | **RED** (config-scoped — one factory blocked; two-factory subset clear) |
| **DOps / Factories** | `SortDeviceOperation` → `SingleRowSingleCore`, `SingleRowMultiCore`, `CrossCoreDataExchange` |
| *Prereqs* — Device 2.0 (every kernel used) | **No (RED)** — one isolated holdover in `CrossCoreDataExchange`'s writer; the other two factories' kernels are clean → Device 2.0 track |
| *Prereqs* — Cross-op escapes | Ok — no function-call or file-path escape; every kernel `#include` is a framework `api/*` header or an in-directory header |
| *Feature Support* — overall | GREEN (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | Ok — all `get_compile_time_arg_val` indices are `constexpr` |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** for all three factories (readiness sheet) |
| *TTNN Readiness* — Concept (current) | `descriptor` (SingleRowSingleCore, SingleRowMultiCore) · `WorkloadDescriptor` (CrossCoreDataExchange) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | Yes — `CrossCoreDataExchange` replicates one `ProgramDescriptor` across the mesh coords |
| *TTNN Readiness* — Is safe to port? | Yes (all three) |
| *TTNN Readiness* — Custom hash | No (all three) |
| *TTNN Readiness* — Runtime-args update | No (all three) |
| *TTNN Readiness* — Pybind `create_descriptor` | No — nanobind binds `ttnn::sort` only (`sort_nanobind.cpp:74`) |
| *TTNN Readiness* — Op-owned tensors | Yes for `CrossCoreDataExchange` (physical-core lookup table, `sort_program_factory.cpp:919`); No for the other two |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` (all three; `CrossCoreDataExchange` carries op-owned tensors natively) |
| *Port work* — Offset base pointer | none (no address fold; every buffer arrives as a clean base) |
| *Port work* — Tensor bindings (per binding) | all **Case 1** (via `TensorAccessor`) across all factories |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none — every `TensorAccessor(...)` is 2-arg |
| *Port work* — CB endpoints | subset {SingleRowSingleCore, SingleRowMultiCore}: self-loop / 1P+1C / config-dependent dead-CB drops (below). CrossCoreDataExchange: deferred (factory blocked) |

## Result

**RED at op level; subset {`SortProgramFactorySingleRowSingleCore`, `SortProgramFactorySingleRowMultiCore`} is clear.**

The single blocker is a **Device 2.0 holdover** confined to `SortProgramFactoryCrossCoreDataExchange`'s writer kernel: a direct kernel-side `get_semaphore(sem_id)` raw-semaphore-address call at `device/kernels/dataflow/writer_cross_core_data_exchange.cpp:26`. It is the Device 1.0 "before" idiom for semaphores (replaced by the `Semaphore<>` object in Device 2.0), it is *not* in the sanctioned free-function set, and Device 2.0 cleanup is off the port's kernel-side whitelist — so it must be cleared on the **Device 2.0 track** before that factory can port, then a (cheap) re-audit. It is a single, **dead** line ("unused - for future improvements"); the rest of the op — including this same writer — is fully Device 2.0. **The other two factories use entirely separate kernels that are Device-2.0 clean and clear every gate, so they are offered as a scoped-subset port now** (see [Code-path scope](#code-path-scope-per-factory) and the porter brief).

Every other gate is clear for **all three** factories: no unsupported Appendix A feature, TTNN factory concept `Is able to port? = yes` ×3, no offset-base-pointer fold, no `TensorAccessor` 3rd-arg override. So `CrossCoreDataExchange` is blocked **only** on the one dead line; nothing else about it needs work before it ports.

> **Path forward (reassurance).** This RED is not a design problem. It is one dead line of Device-1.0-style code. The Device 2.0 team deletes it (or completes the intended `Semaphore<>` migration the comment gestures at), the op re-audits, and `CrossCoreDataExchange` clears immediately. Meanwhile the two-factory subset can be ported in parallel.

## Code-path scope (per factory)

| Factory | Concept | Device 2.0 | Other gates | Port status |
|---|---|---|---|---|
| `SingleRowSingleCore` | `descriptor` | clean | all clear | **portable now (subset)** |
| `SingleRowMultiCore` | `descriptor` | clean | all clear | **portable now (subset)** |
| `CrossCoreDataExchange` | `WorkloadDescriptor` (SPMD, op-owned tensors) | **RED** — 1 holdover in writer | all clear | blocked on the one Device 2.0 line; re-audit after fix |

The three factories are separate structs in one `program_factory_t` variant selected by `Wt` at runtime, so a factory-scoped subset is well-defined (the matmul / GlobalCircularBuffer precedent in the recipe). See Recipe notes for one caveat about heterogeneous-variant porting.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN for all three factories. The readiness sheet (*"Operations analysis"*, fetched this run) reports `Is able to port? = yes` on each row. Cross-check against the code is clean on every cheaply-checkable column:

  | Factory | `Concept` (sheet) | code | `Custom hash` | `Runtime-args update` | `Pybind desc` | `Op-owned tensors?` | `Secretly SPMD?` |
  |---|---|---|---|---|---|---|---|
  | `SingleRowSingleCore` | `descriptor` | `create_descriptor()`→`ProgramDescriptor` ✓ | no ✓ | no ✓ | no ✓ | (blank) ✓ | n/a |
  | `SingleRowMultiCore` | `descriptor` | `create_descriptor()`→`ProgramDescriptor` ✓ | no ✓ | no ✓ | no ✓ | (blank) ✓ | n/a |
  | `CrossCoreDataExchange` | `WorkloadDescriptor` | `create_workload_descriptor()`→`WorkloadDescriptor` ✓ | no ✓ | no ✓ | no ✓ | **yes** ✓ | **yes** ✓ |

  Cross-column invariants hold: `Op-owned tensors? = yes` occurs only on the `WorkloadDescriptor` factory; `Runtime-args update = no` throughout. Code confirmations: no `compute_program_hash` override in the device op; no `get_dynamic_runtime_args` / `override_runtime_arguments` in any factory; `sort_nanobind.cpp` binds `ttnn::sort`, not `create_descriptor`. Op-owned tensor confirmed at `sort_program_factory.cpp:919` (`wd.buffers.push_back({lookup_owner, lookup_buffer})`). Secretly-SPMD confirmed at `sort_program_factory.cpp:924-930` — the same `ProgramDescriptor` is replicated across every entry of `tensor_coords.ranges()` (one program across the mesh; multi-program only as a coordinate-set artifact). The sheet's `Op Classification = PD (pointer-patching)` matches the observed `Buffer*`-in-runtime-args binding form (`Smuggled pointer = no`, `Is safe to port? = yes`).

- **Device 2.0 (every kernel used):** **RED** — one isolated holdover.

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | `device/kernels/dataflow/writer_cross_core_data_exchange.cpp` | 26 | `get_semaphore(get_compile_time_arg_val(10))` (raw sem address → unused `sem_exchange_addr`) | none — no `Semaphore<>` object is constructed in this kernel |

  This is the only Device-1.0 idiom in the entire op. It is used by `SortProgramFactoryCrossCoreDataExchange` only (the writer kernel it lives in is referenced by no other factory). **Incompleteness sizing: isolated, trivial.** The `device_api_migration_guide.md` §"Semaphore Operations" (lines 405-451) lists exactly this call — `get_semaphore(sem_id)` → raw L1 address — as the Device 1.0 "before" form, with `Semaphore<> sem(sem_id)` as the Device 2.0 replacement; the op's *reader* cross-core kernel already uses `Semaphore<>` (`reader_cross_core_data_exchange.cpp:85-86`). The line is dead (the result is never read; the comment says "unused - for future improvements"), so the fix is a one-line deletion — or, if the "future improvements" are intended, completing the `Semaphore<>` migration. Either way it is a Device 2.0-track change (off the port's kernel-side whitelist, which forbids the port from removing even a one-line Device-1.0 idiom), followed by a cheap re-audit. **Route to: Device 2.0 migration team.** (See Recipe notes — the recipe's isolated-holdover template describes a *live CB-index* free-function with its wrapper in scope; a *dead sem-address* free-function is a boundary case I resolved conservatively as a gate.)

  All other kernels — and all data-movement in this same writer — are structurally Device 2.0: `Noc` object; `DataflowBuffer` / `CircularBuffer` wrappers; `Semaphore<>` objects with `.up`/`.wait`/`.set`/`.set_multicast`; `CoreLocalMem` + `UnicastEndpoint` for L1→L1 NoC writes; `TensorAccessor` (2-arg). Sanctioned free functions in use and **not** flagged: `get_tile_size(cb_id)` (per the Green bullet). No raw `noc_async_read`/`noc_async_write`, no `noc_semaphore_*` free functions, no `InterleavedAddrGen`/`ShardedAddrGen`, no raw `cb_*` FIFO free functions, no raw `get_read_ptr`/`get_write_ptr` free functions anywhere.

- **Feature compatibility:** all Appendix A entries N/A (clean scan — no `GREEN` row exists; N/A means the feature is absent).

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer` type, no `.global_circular_buffer` field, no `remote_cb`/`.remote_index`/`CreateGlobalCircularBuffer`. Plain `CBDescriptor` throughout. |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `.address_offset`, `set_address_offset`, 4-arg `UpdateDynamicCircularBufferAddress`, or `cb_descriptor_from_sharded_tensor` anywhere. |
  | GlobalSemaphore | N/A | no `GlobalSemaphore` type or `CreateGlobalSemaphore`. Semaphores are plain `SemaphoreDescriptor` (ids 0-2) → `Semaphore<>`. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | op input is a single fixed `const Tensor&` (`tensor_args_t = SortInputs`), not a `std::vector<Tensor>`. Every `get_compile_time_arg_val(N)` uses a `constexpr` index — literal or a `constexpr` offset from `TensorAccessorArgs<>::next_compile_time_args_offset()` (the recipe's explicit false-positive guard). No runtime-varying CTA index. |

- **CB endpoints (GATE-free):** run for the surviving subset only (below). `CrossCoreDataExchange` deferred (factory blocked on the Device 2.0 gate — its detail is regenerated at re-audit).

- **Offset base pointers:** GREEN. No `->address()` appears anywhere in the op (host or kernel). Every tensor buffer is delivered to its kernel as a `Buffer*` runtime arg (auto-registered `BufferBinding`) and the kernel reconstructs addresses through a `TensorAccessor` from the clean base — no host-folded `base + offset`. Not in the offset-base-pointers triage doc (`2026-07-19_offset_base_pointers.md`), and the scan confirms no fold was introduced since. Type 3/4 absent.

- **TensorAccessor 3rd argument:** GREEN. Every `TensorAccessor(...)` call in the op is 2-argument (`TensorAccessor(args, base_addr)`) — 14 sites across the dataflow kernels, none passing a page-size 3rd arg. Not in the 3rd-arg triage doc (`2026-07-06_tensor_accessor_3rd_arg_triage.md`), and the scan confirms no 3rd-arg site was added. Nothing to drop; no Class 3/4/Special hazard.

## Port-work summary  *(mirrors the brief; applies to the clean subset)*

- **Tensor bindings** (per binding, per factory) — **all Case 1** (fed into a `TensorAccessor`; today delivered via the `Buffer*`-binding form, patched on cache hits, so correct-on-cache-hit and not the silent-wrong hazard):
  - `SingleRowSingleCore`: `input` (reader), `index`-output (reader), `value`-output (writer) — all Case 1.
  - `SingleRowMultiCore`: `input` (coordinator + reader), `value`-output (coordinator + writer), `index`-output (coordinator + writer) — all Case 1.
  - (`CrossCoreDataExchange`, for reference once unblocked: `input`, `index`-out, `value`-out, plus the op-owned `physical_core_lookup_table` — all Case 1.)
  - Port action: express each as `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(tensor::name)`; the address-via-RTA and its `TensorAccessorArgs` plumbing both disappear.
- **TensorParameter relaxation:** none (sheet `= none`; no custom hash, so no relaxation candidate to mine either).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints (subset):** see the census below — self-loops, 1P+1C assignments, and config-dependent dead-CB drops. No multi-binding anywhere.

### CB endpoints census — subset {SingleRowSingleCore, SingleRowMultiCore}

Dispositions are per `(CB, config)` and, for the multi-core factory, per node-type (the coordinator runs a different kernel than the workers). **No CB in either factory reaches ≥3 touchers or two locked producers/consumers, so nothing needs the multi-binding advanced option.** The scan actively looked for a hidden raw co-filler (a `get_write_ptr` write by a non-FIFO-producer gated by a semaphore) and found none in these two factories.

**`SortProgramFactorySingleRowSingleCore`** — all CBs on one uniform core range; kernels reader / writer / compute.

| CB | TILE config | ROW_MAJOR config |
|---|---|---|
| c_0 `input_tensor` | 1P+1C (reader→compute) | **self-loop** (compute tilizes, sorts, untilizes) |
| c_1 `index_tensor` | 1P+1C (writer→compute) | 1P+1C (writer→compute) |
| c_2 `input_tensor_transposed` | **self-loop** (compute) | **self-loop** (compute) |
| c_3 `index_tensor_transposed` | **self-loop** (compute) | **self-loop** (compute) |
| c_4 `value_tensor` | 1P+1C (compute→writer) | **dead — drop** (0 touchers) |
| c_5 `index_tensor_output` | 1P+1C (compute→reader) | **dead — drop** (0 touchers) |
| c_6 `synchronization` | **self-loop** (compute) | **self-loop** (compute) |
| c_7 `rm_input` | (not allocated) | 1P+1C (reader→compute) |
| c_8 `rm_value_output` | (not allocated) | 1P+1C (compute→writer) |
| c_9 `rm_index_output` | (not allocated) | 1P+1C (compute→reader) |
| c_10 `rm_post_sort_index` | (not allocated) | **self-loop** (compute) |

- **Dead-in-RM drop (confirmed):** in the ROW_MAJOR instantiation, c_4 (`value_tensor`, produced only at `sort_single_row_single_core.cpp:297`, inside `if constexpr (!is_row_major)`) and c_5 (`index_tensor_output`, produced only at `:299`) have **no toucher** — the RM path routes value/index output through c_8/c_9 instead. The factory still allocates c_4/c_5 unconditionally (`:140-160`), so the RM program would carry two unbound DFBs → spec-validator rejection. **Porter must not emit c_4/c_5 as DFBs in the RM instantiation** (guard their allocation on `!is_row_major`, mirroring the existing `if (is_row_major)` guard on c_7-c_10). Behavior-neutral (a CB no kernel touches has no behavior).

**`SortProgramFactorySingleRowMultiCore`** — CBs c_0-c_5 declared on `all_core_set` (coordinator + workers); c_6/c_7 aliased (coordinator-only `rm_coord_*` **and** worker-only `rm_worker_input_*` share the index, on disjoint node sets — safe, each node sees one instance); c_8/c_9 worker-only. Coordinator node runs `coordinator_single_row_multi_core.cpp`; worker nodes run reader / writer / compute.

*Worker nodes:*

| CB | TILE config | ROW_MAJOR config |
|---|---|---|
| c_0 `input_tensor` | 1P+1C (reader→compute) | **self-loop** (compute tilizes c_6→c_0, sorts) |
| c_1 `index_tensor` | 1P+1C (reader→compute) | **self-loop** (compute tilizes c_7→c_1, sorts) |
| c_2 `input_tensor_transposed` | **self-loop** (compute) | **self-loop** (compute) |
| c_3 `index_tensor_transposed` | **self-loop** (compute) | **self-loop** (compute) |
| c_4 `input_tensor_output` (value) | 1P+1C (compute→writer) | **self-loop** (compute packs, untilizes c_4→c_8) |
| c_5 `index_tensor_output` | 1P+1C (compute→writer) | **self-loop** (compute packs, untilizes c_5→c_9) |
| c_6 `rm_worker_input_value` | (not allocated) | 1P+1C (reader→compute) |
| c_7 `rm_worker_input_index` | (not allocated) | 1P+1C (reader→compute) |
| c_8 `rm_worker_output_value` | (not allocated) | 1P+1C (compute→writer) |
| c_9 `rm_worker_output_index` | (not allocated) | 1P+1C (compute→writer) |

*Coordinator node* (only the coordinator kernel runs): c_0/c_1 → **self-loop** in TILE (the coordinator both fills from DRAM and drains back, streaming); c_6/c_7 (`rm_coord_value_row`/`rm_coord_index_row`) → **self-loop** in RM. **c_2, c_3, c_4, c_5 are untouched on the coordinator node** (the coordinator kernel never references them), and c_0/c_1 are untouched on the coordinator node in the RM config (it uses c_6/c_7 there). Because the factory scopes c_0-c_5 to `all_core_set`, each of those is an unbound DFB on the coordinator node → validator rejection.

- **Over-scoped core range (porter action):** the porter should narrow c_2-c_5 (and c_0/c_1 for the RM instantiation) to the worker `core_range` rather than `all_core_set`, so the coordinator node carries only the CBs it touches. Behavior-neutral (dead on that node). This is the multi-core analogue of the single-core dead-in-RM drop, and the same spec-validator rule forces it.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no factory in the clean subset has a hidden second writer or a multi-reader that reaches ≥3 touchers. The dispositions to apply are self-loops, 1P+1C assignments, and the dead-CB / over-scoped-range drops above.
- **Cross-op / shared kernels:** none borrowed. All 10 kernel sources are in `sort/device/kernels/`; every `#include` resolves to a framework `api/*` header or one of the three in-directory shared headers. No file-path escape, no function-call escape → no port-together coupling with other ops.
- **RTA varargs:** none. Every runtime arg is read at a fixed offset (`get_arg_val<uint32_t>(0..N)`); no counted `arg_index++` loop, no `get_common_arg_val`. All args are nameable — no vararg mechanism needed.

## Team-only

- **Out-of-directory coupling & donor shape:** `✓ clean`. No donor function-call escape and no file-path kernel instantiation. Kernel `#include`s are exclusively framework LLK/HAL (`api/dataflow/*`, `api/compute/*`, `api/tensor/*`, `api/core_local_mem.h`, `ckernel.h` — donor class 1, "no concern"), plus `<cstdint>`/`<utility>` and the op's own `sort_common.hpp` / `sort_dataflow_common.hpp` / `cross_core_data_exchange_common.hpp`. No summary table or per-call detail needed (all rolls ✓).
- **Relaxation candidates (from a custom hash):** none — no factory has a custom hash.
- **TTNN factory analysis:** sheet-derived facts with code evidence — current concepts `descriptor` ×2 / `WorkloadDescriptor` ×1; op-owned tensors only on `CrossCoreDataExchange` (the physical-core lookup table, built once on cache-miss in `create_workload_descriptor` and parked on `wd.buffers` so it survives cache hits, address patched into reader RTAs via `BufferBinding` — `sort_program_factory.cpp:475-497, 915-919`); `WorkloadDescriptor` need is a pure op-owned-tensor / SPMD artifact (single replicated program), not genuine multi-program; no pybind `create_descriptor`, no custom hash, no custom `override_runtime_arguments`. Target concept `MetalV2FactoryConcept` for all three (carrying op-owned tensors natively for `CrossCoreDataExchange`).

## Misc anomalies  *(team-only, non-gating; noted in passing, not from a dedicated hunt)*

- **Dead raw-sem-address line** — `writer_cross_core_data_exchange.cpp:26`: `sem_exchange_addr = get_semaphore(...)` is computed and never used ("unused - for future improvements"). Same site as the Device 2.0 gate above; also a plain dead computation.
- **Dead CTAs in the cross-core writer** — `writer_cross_core_data_exchange.cpp`: arg 9 `number_of_cores_used` and arg 10 `sem_exchange` are both marked "unused - for future improvements" and are not used in kernel logic (arg 10 only feeds the dead line above).
- **Stray `push_back` on the lookup-table CB** — `writer_cross_core_data_exchange.cpp:102`: the writer calls `physical_core_lookup_table_dfb.push_back(one_tile)` on c_10 without ever `reserve_back`-ing or otherwise producing it; the reader is the CB's real producer/consumer. This makes the cross-core writer an extra toucher of c_10 (relevant to that factory's deferred CB census). Looks vestigial.
- **Config-dependent dead CB allocations** — the dead-in-RM c_4/c_5 (single-core) and the over-scoped c_0-c_5 on the coordinator node (multi-core), detailed in the CB census. Legacy over-allocation: harmless L1 waste under Device 1.0, but a spec-validator rejection under Metal 2.0, so the porter must drop/narrow them.
- **Placeholder semaphore** — `sort_program_factory.cpp:746-760`: `SortProgramFactoryCrossCoreDataExchange` allocates a `semaphore_unused` (id 1) purely to keep the id numbering stable; no kernel uses it. (Deferred detail — factory blocked.)
- **`get_core_physical_coordinates` hardcoded `tile_size = 1024`** — `cross_core_data_exchange_common.hpp:137`: the bound check `if (2 * core_id >= tile_size)` uses a hardcoded 1024 (the UInt32-tile element count) rather than a value derived from the lookup-table CB; loosely coupled to the actual page size. (Deferred detail — factory blocked.)
- **Behaviorally-inert hashed / passed attributes** — `stable` is asserted `false` at the `ttnn::sort` entry (`sort.cpp:232`) yet is still threaded to the compute CTA (`descending`/`stable`) and left in the hashed `SortParams`; `dim` is constrained to the last axis (`-1` or `rank-1 = 3`) at the device op, so the two legal values hash distinctly for identical behavior. Minor cache/plumbing cruft; not porter work.

## Per-DeviceOperation attribution

Single `DeviceOperation` (`SortDeviceOperation`); attribution above is per-factory. Gate-relevant split: `SingleRowSingleCore` and `SingleRowMultiCore` clear all gates (portable subset); `CrossCoreDataExchange` clears every gate **except** Device 2.0 (one dead holdover line) and additionally carries op-owned tensors + secretly-SPMD `WorkloadDescriptor` shape.

## Questions for the user  *(none blocking)*

1. **Heterogeneous-variant subset port:** the two clean factories share a `program_factory_t` variant with the blocked `CrossCoreDataExchange`. Porting {SingleRowSingleCore, SingleRowMultiCore} to `MetalV2FactoryConcept` while `CrossCoreDataExchange` stays on `WorkloadDescriptor` yields a mixed-concept variant. Given the blocker is one dead line, do you prefer (a) the scoped-subset port now, or (b) clearing the one line on the Device 2.0 track first and porting all three together at re-audit? The brief is written for (a); (b) is likely less total work.

## Recipe notes  *(friction with the audit recipe itself)*

- **Dead Device-1.0 idiom at the Device 2.0 gate.** The isolated-holdover template in [Device 2.0 prerequisite](audit/metal2_audit.md#device-20-prerequisite) is written for a *live* **CB-index** free function "where the corresponding Device-2.0 wrapper object is already in scope at the call site." My finding is a *dead* **semaphore-id** free function (`get_semaphore`) with **no** wrapper in scope — it matches the "raw sem addresses" signal in the *Broad Device 1.0* bullet, but it is a single isolated line, not broad. Neither sub-category fits cleanly. I resolved it as a gate (a kernel containing a Device-1.0 raw-sem-address idiom is "not fully Device 2.0 compliant," and the port whitelist can't remove it), but the recipe doesn't explicitly say whether a **dead** Device-1.0 idiom gates or is merely an Incidental anomaly. A one-line clarification ("dead Device-1.0 idioms still gate, because the port cannot remove them off-whitelist") would remove the judgment call.
- **Config-scoped subset brief vs. Device 2.0 "no partial pass".** [Device 2.0 prerequisite](audit/metal2_audit.md#device-20-prerequisite) says "there is no partial pass" and "the op is re-audited once the cleanup lands," while [Output: the two documents](audit/metal2_audit.md#output-the-two-documents) says a config-scoped GATE "still issues a brief for the clean subset." Here the Device 2.0 RED is confined to one factory's kernel while two other factories use clean kernels, so both rules seem to apply with opposite conclusions. I followed the Code-path-scope rule (issue a subset brief) on the matmul/GlobalCircularBuffer precedent, reading "no partial pass" as "no half-migrated *kernel*," not "no clean-*factory* subset." Worth an explicit statement of which rule wins when a Device 2.0 holdover is factory-confined.
