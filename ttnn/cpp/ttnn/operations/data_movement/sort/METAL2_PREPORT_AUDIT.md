# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/sort`

**Device operation:** one — `ttnn::prim::SortDeviceOperation` — with three program factories (selected at runtime by `Wt` in `select_program_factory`):

- **`SortDeviceOperation`**
  - `SortProgramFactorySingleRowSingleCore` — `create_descriptor()` → `ProgramDescriptor` (`sort_program_factory.cpp:21`). Kernels: `reader_single_row_single_core.cpp`, `writer_single_row_single_core.cpp`, `sort_single_row_single_core.cpp`.
  - `SortProgramFactorySingleRowMultiCore` — `create_descriptor()` → `ProgramDescriptor` (`sort_program_factory.cpp:969`). Kernels: `coordinator_single_row_multi_core.cpp`, `reader_single_row_multi_core.cpp`, `writer_single_row_multi_core.cpp`, `sort_single_row_multi_core.cpp`.
  - `SortProgramFactoryCrossCoreDataExchange` — `create_workload_descriptor()` → `WorkloadDescriptor` (`sort_program_factory.cpp:902`), op-owned lookup-table tensor. Kernels: `reader_cross_core_data_exchange.cpp`, `writer_cross_core_data_exchange.cpp`, `sort_cross_core_data_exchange.cpp`.

Shared kernel headers (in-directory): `compute/sort_common.hpp`, `dataflow/sort_dataflow_common.hpp`, `dataflow/cross_core_data_exchange_common.hpp`. No unreferenced kernel files in the directory.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `44da718b06b 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

> **Re-audit note.** An earlier pass RED'd this op on a single Device 2.0 holdover — a dead `get_semaphore(sem_id)` raw-sem-address call in `writer_cross_core_data_exchange.cpp`. That line has since been removed (the kernel now carries only a `// CT arg 10 unused - for future improvements` comment; the factory still passes the id at writer CTA slot 10, so downstream CTA positions are unchanged). This report reflects the current, fixed state: **all gates clear — GREEN.**

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/sort` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `SortDeviceOperation` → `SingleRowSingleCore`, `SingleRowMultiCore`, `CrossCoreDataExchange` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — no Device-1.0 idiom anywhere; the sole prior holdover is removed |
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
| *Port work* — CB endpoints | self-loop / 1P+1C / dead-CB drop (config-dependent) / over-scoped-range narrow · one **multi-binding** (CrossCoreDataExchange c_10) |

## Result

**GREEN → brief issued** for all three factories (`METAL2_PORT_BRIEF.md`).

Every gate clears for every factory: Device 2.0 ✓ (no idiom left after the holdover was removed), Feature compatibility ✓ (no Appendix A feature), TTNN factory concept ✓ (`Is able to port? = yes` ×3, cross-checked), Offset base pointers ✓ (no fold), TensorAccessor 3rd arg ✓ (all 2-arg). Port work is uniform and low-risk: every tensor binding is Case 1 (via `TensorAccessor`), no relaxation, no 3rd-arg drop; the only non-mechanical CB action is a single multi-binding advanced-option flag on `CrossCoreDataExchange`'s lookup-table CB, plus config-dependent dead-CB drops the Metal 2.0 spec validator will otherwise force.

## Per-factory scope

| Factory | Concept | Target concept | Op-owned tensors | Notable |
|---|---|---|---|---|
| `SingleRowSingleCore` | `descriptor` | `MetalV2FactoryConcept` | none | dead-in-RM CBs (c_4, c_5) |
| `SingleRowMultiCore` | `descriptor` | `MetalV2FactoryConcept` | none | coordinator over-scoped CBs; c_6/c_7 aliased across disjoint node sets |
| `CrossCoreDataExchange` | `WorkloadDescriptor` (SPMD) | `MetalV2FactoryConcept` + op-owned tensors | physical-core lookup table | multi-binding on c_10; dead-in-RM CBs (c_4, c_5) |

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN for all three factories. The readiness sheet (*"Operations analysis"*, fetched this run) reports `Is able to port? = yes` on each row. Cross-check against the code is clean on every cheaply-checkable column:

  | Factory | `Concept` (sheet) | code | `Custom hash` | `Runtime-args update` | `Pybind desc` | `Op-owned tensors?` | `Secretly SPMD?` |
  |---|---|---|---|---|---|---|---|
  | `SingleRowSingleCore` | `descriptor` | `create_descriptor()`→`ProgramDescriptor` ✓ | no ✓ | no ✓ | no ✓ | (blank) ✓ | n/a |
  | `SingleRowMultiCore` | `descriptor` | `create_descriptor()`→`ProgramDescriptor` ✓ | no ✓ | no ✓ | no ✓ | (blank) ✓ | n/a |
  | `CrossCoreDataExchange` | `WorkloadDescriptor` | `create_workload_descriptor()`→`WorkloadDescriptor` ✓ | no ✓ | no ✓ | no ✓ | **yes** ✓ | **yes** ✓ |

  Cross-column invariants hold: `Op-owned tensors? = yes` occurs only on the `WorkloadDescriptor` factory; `Runtime-args update = no` throughout. Code confirmations: no `compute_program_hash` override in the device op; no `get_dynamic_runtime_args` / `override_runtime_arguments` in any factory; `sort_nanobind.cpp` binds `ttnn::sort`, not `create_descriptor`. Op-owned tensor confirmed at `sort_program_factory.cpp:919` (`wd.buffers.push_back({lookup_owner, lookup_buffer})`). Secretly-SPMD confirmed at `sort_program_factory.cpp:924-930` — the same `ProgramDescriptor` is replicated across every entry of `tensor_coords.ranges()` (one program across the mesh; multi-program only as a coordinate-set artifact). The sheet's `Op Classification = PD (pointer-patching)` matches the observed `Buffer*`-in-runtime-args binding form (`Smuggled pointer = no`, `Is safe to port? = yes`).

- **Device 2.0 (every kernel used):** **GREEN.** A whole-op sweep finds no Device-1.0 idiom: no `get_semaphore`, no `noc_semaphore_*` / raw `noc_async_read`/`noc_async_write` free functions, no `InterleavedAddrGen`/`ShardedAddrGen`/`get_noc_addr_from_bank_id`, no raw `cb_*` FIFO free functions, no raw `get_read_ptr`/`get_write_ptr` free functions. Every kernel is structurally Device 2.0: `Noc` object; `DataflowBuffer` / `CircularBuffer` wrappers; `Semaphore<>` objects with `.up`/`.wait`/`.set`/`.set_multicast`; `CoreLocalMem` + `UnicastEndpoint` for L1→L1 NoC writes; `TensorAccessor` (2-arg). Sanctioned free functions in use and correctly not flagged: `get_tile_size(cb_id)` (per the Green bullet). The prior blocker — a dead `get_semaphore(get_compile_time_arg_val(10))` in `writer_cross_core_data_exchange.cpp` — has been removed.

- **Feature compatibility:** all Appendix A entries N/A (clean scan — no `GREEN` row exists; N/A means the feature is absent).

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer` type, no `.global_circular_buffer` field, no `remote_cb`/`.remote_index`/`CreateGlobalCircularBuffer`. Plain `CBDescriptor` throughout. |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `.address_offset`, `set_address_offset`, 4-arg `UpdateDynamicCircularBufferAddress`, or `cb_descriptor_from_sharded_tensor` anywhere. |
  | GlobalSemaphore | N/A | no `GlobalSemaphore` type or `CreateGlobalSemaphore`. Semaphores are plain `SemaphoreDescriptor` → `Semaphore<>`. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | op input is a single fixed `const Tensor&` (`tensor_args_t = SortInputs`), not a `std::vector<Tensor>`. Every `get_compile_time_arg_val(N)` uses a `constexpr` index — literal or a `constexpr` offset from `TensorAccessorArgs<>::next_compile_time_args_offset()` (the recipe's explicit false-positive guard). No runtime-varying CTA index. |

- **CB endpoints (GATE-free):** full census below (all three factories). Every out-of-window CB has a port-time disposition; one CB needs the multi-binding advanced option; nothing blocks the port.

- **Offset base pointers:** GREEN. No `->address()` appears anywhere in the op (host or kernel). Every tensor buffer is delivered to its kernel as a `Buffer*` runtime arg (auto-registered `BufferBinding`) and the kernel reconstructs addresses through a `TensorAccessor` from the clean base — no host-folded `base + offset`. Not in the offset-base-pointers triage doc (`2026-07-19_offset_base_pointers.md`), and the scan confirms no fold was introduced since. Type 3/4 absent.

- **TensorAccessor 3rd argument:** GREEN. Every `TensorAccessor(...)` call in the op is 2-argument (`TensorAccessor(args, base_addr)`) — 14 sites across the dataflow kernels, none passing a page-size 3rd arg. Not in the 3rd-arg triage doc (`2026-07-06_tensor_accessor_3rd_arg_triage.md`), and the scan confirms no 3rd-arg site was added. Nothing to drop; no Class 3/4/Special hazard.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per factory) — **all Case 1** (fed into a `TensorAccessor`; today delivered via the `Buffer*`-binding form, patched on cache hits, so correct-on-cache-hit and not the silent-wrong hazard). Port action for each: express as `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(tensor::name)`; the address-via-RTA and its `TensorAccessorArgs<N>()` plumbing both disappear.
  - `SingleRowSingleCore`: `input` (reader), `index`-output (reader), `value`-output (writer).
  - `SingleRowMultiCore`: `input` (coordinator + reader), `value`-output (coordinator + writer), `index`-output (coordinator + writer).
  - `CrossCoreDataExchange`: `input` (reader), `index`-output (reader), `value`-output (writer), plus the op-owned `physical_core_lookup_table` (reader) — all Case 1. The lookup table is the op-owned tensor and binds as a `TensorParameter` carried on the `MetalV2FactoryConcept`.
- **TensorParameter relaxation:** none (sheet `= none`; no custom hash, so no relaxation candidate to mine either).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** census below — self-loops, 1P+1C, one multi-binding (CrossCoreDataExchange c_10), and config-dependent dead-CB drops / over-scoped-range narrows.

### CB endpoints census

Dispositions are per `(CB, config)` and, for the multi-core factory, per node-type (the coordinator runs a different kernel than the workers). The scan actively looked for hidden raw co-fillers; the one found is the visible stray `push_back` on `CrossCoreDataExchange`'s lookup-table CB (below).

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

- **Dead-in-RM drop (confirmed):** in the ROW_MAJOR instantiation, c_4 (`value_tensor`, produced only at `sort_single_row_single_core.cpp:297`, inside `if constexpr (!is_row_major)`) and c_5 (`index_tensor_output`, produced only at `:299`) have **no toucher** — the RM path routes output through c_8/c_9. The factory still allocates c_4/c_5 unconditionally (`:140-160`), so the RM program would carry two unbound DFBs → spec-validator rejection. **Porter must not emit c_4/c_5 as DFBs in the RM instantiation** (guard their allocation on `!is_row_major`, mirroring the existing `if (is_row_major)` guard on c_7-c_10). Behavior-neutral.

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

*Coordinator node* (only the coordinator kernel runs): c_0/c_1 → **self-loop** in TILE (the coordinator both fills from DRAM and drains back); c_6/c_7 (`rm_coord_*`) → **self-loop** in RM. **c_2, c_3, c_4, c_5 are untouched on the coordinator node** (the coordinator kernel never references them), and c_0/c_1 are untouched on the coordinator node in the RM config. Because the factory scopes c_0-c_5 to `all_core_set`, each of those is an unbound DFB on the coordinator node → validator rejection.

- **Over-scoped core range (porter action):** narrow c_2-c_5 (and c_0/c_1 for the RM instantiation) to the worker `core_range` rather than `all_core_set`, so the coordinator node carries only the CBs it touches. Behavior-neutral (dead on that node).

**`SortProgramFactoryCrossCoreDataExchange`** — all CBs on one uniform worker core range (the "leader" is core 0 running the same reader kernel — no separate coordinator kernel). Kernels reader / writer / compute; the reader↔peer NoC exchange runs in **both** configs (the sort is always TILE-format internally; RM tilizes first).

| CB | TILE config | ROW_MAJOR config |
|---|---|---|
| c_0 `input_tensor` | 1P+1C (reader→compute) | **self-loop** (compute tilizes c_12→c_0, sorts) |
| c_1 `index_tensor` | 1P+1C (writer→compute) | 1P+1C (writer→compute) |
| c_2 `input_tensor_transposed` | **self-loop** (compute) | **self-loop** (compute) |
| c_3 `index_tensor_transposed` | **self-loop** (compute) | **self-loop** (compute) |
| c_4 `value_tensor` | 1P+1C (compute→writer) | **dead — drop** (0 touchers) |
| c_5 `index_tensor_output` | 1P+1C (compute→reader) | **dead — drop** (0 touchers) |
| c_6 `value_tensor_intermediate` | 1P+1C (compute→reader) | 1P+1C (compute→reader) |
| c_7 `index_tensor_intermediate` | 1P+1C (compute→reader) | 1P+1C (compute→reader) |
| c_8 `value_tensor_peer` | 1P+1C (reader→compute) | 1P+1C (reader→compute) |
| c_9 `index_tensor_peer` | 1P+1C (reader→compute) | 1P+1C (reader→compute) |
| c_10 `physical_core_lookup_table` | **multi-binding** (2 locked producers) | **multi-binding** (2 locked producers) |
| c_11 `packer_unpacker_sync` | **self-loop** (compute) | **self-loop** (compute) |
| c_12 `rm_input` | (not allocated) | 1P+1C (reader→compute) |
| c_13 `rm_value_output` | (not allocated) | 1P+1C (compute→writer) |
| c_14 `rm_index_output` | (not allocated) | 1P+1C (compute→reader) |
| c_15 `rm_post_sort_index` | (not allocated) | **self-loop** (compute) |

- **Multi-binding — c_10 `physical_core_lookup_table`:** two kernels drive the producer/write cursor. The **reader** is the genuine producer (`reserve_back` at `reader_cross_core_data_exchange.cpp:75`, `push_back` at `:216`) and also self-reads it via `get_read_ptr` in `get_core_physical_coordinates`. The **writer** does a lone `push_back(one_tile)` at `writer_cross_core_data_exchange.cpp:102` with no matching `reserve_back` — a second producer-role access. Two locked producers on one node ⇒ the census cannot fit 1P+1C, so the porter **sets the DFB multi-binding advanced option** on c_10 (records Quasar debt). *Note:* the writer's `push_back` is vestigial (see Misc anomalies); if the ops team removes it, c_10 collapses to a clean reader **self-loop** — but that is a functional change, out of port scope, so the port sets the flag as-is.
- **Dead-in-RM drop (confirmed):** c_4 (`value_tensor`) and c_5 (`index_tensor_output`) have no toucher in the RM instantiation (RM routes output through c_13/c_14 and the sort keeps value/index in the transposed/intermediate CBs). Same disposition as the single-core factory — guard their allocation on `!is_row_major`.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** one — `CrossCoreDataExchange` c_10 (details above). The single-core and multi-core factories have **no** multi-binding (no hidden second writer, no multi-reader reaching ≥3 touchers). Before setting the c_10 flag, confirm the writer's lone `push_back` is the only extra producer (it is — the writer never `reserve_back`s c_10).
- **Cross-op / shared kernels:** none borrowed. All 10 kernel sources are in `sort/device/kernels/`; every `#include` resolves to a framework `api/*` header or one of the three in-directory shared headers. No file-path escape, no function-call escape → no port-together coupling with other ops.
- **RTA varargs:** none. Every runtime arg is read at a fixed offset (`get_arg_val<uint32_t>(0..N)`); no counted `arg_index++` loop, no `get_common_arg_val`. All args nameable — no vararg mechanism needed.

## Team-only

- **Out-of-directory coupling & donor shape:** `✓ clean`. No donor function-call escape and no file-path kernel instantiation. Kernel `#include`s are exclusively framework LLK/HAL (`api/dataflow/*`, `api/compute/*`, `api/tensor/*`, `api/core_local_mem.h`, `ckernel.h` — donor class 1, "no concern"), plus `<cstdint>`/`<utility>` and the op's own three in-directory shared headers. No summary table or per-call detail needed (all rolls ✓).
- **Relaxation candidates (from a custom hash):** none — no factory has a custom hash.
- **TTNN factory analysis:** current concepts `descriptor` ×2 / `WorkloadDescriptor` ×1; op-owned tensors only on `CrossCoreDataExchange` (the physical-core lookup table, built once on cache-miss in `create_workload_descriptor` and parked on `wd.buffers` so it survives cache hits, address patched into reader RTAs via `BufferBinding` — `sort_program_factory.cpp:475-497, 915-919`); `WorkloadDescriptor` need is a pure op-owned-tensor / SPMD artifact (single replicated program), not genuine multi-program; no pybind `create_descriptor`, no custom hash, no custom `override_runtime_arguments`. Target concept `MetalV2FactoryConcept` for all three (carrying op-owned tensors natively for `CrossCoreDataExchange`).

## Misc anomalies  *(team-only, non-gating; noted in passing, not from a dedicated hunt)*

- **Vestigial `push_back` on the lookup-table CB** — `writer_cross_core_data_exchange.cpp:102`: the writer calls `physical_core_lookup_table_dfb.push_back(one_tile)` on c_10 without ever `reserve_back`-ing or otherwise producing it; the reader is the CB's real producer. This is what forces c_10 into a multi-binding (above). Looks like leftover code; removing it (ops team; functional change) would let the port self-loop c_10 instead of flagging it.
- **Declared-but-unused CTAs in the cross-core writer** — `writer_cross_core_data_exchange.cpp`: arg 4 `value_tensor_peer_cb_index` (declared, no `DataflowBuffer` built from it), arg 9 `number_of_cores_used` ("unused - for future improvements"), and arg 10 (now only a `// CT arg 10 unused` comment, still passed by the factory to preserve slot alignment). Harmless plumbing cruft.
- **Config-dependent dead CB allocations** — the dead-in-RM c_4/c_5 (single-core and cross-core) and the over-scoped c_0-c_5 on the multi-core coordinator node, detailed in the CB census. Legacy over-allocation: harmless L1 waste under Device 1.0, but a spec-validator rejection under Metal 2.0, so the porter must drop/narrow them.
- **Placeholder semaphore** — `sort_program_factory.cpp:746-760`: `SortProgramFactoryCrossCoreDataExchange` allocates a `semaphore_unused` (id 1) purely to keep the id numbering stable; no kernel uses it. Ports as an inert `SemaphoreSpec` unless the ops team removes it.
- **`get_core_physical_coordinates` hardcoded `tile_size = 1024`** — `cross_core_data_exchange_common.hpp:137`: the bound check `if (2 * core_id >= tile_size)` uses a hardcoded 1024 (the UInt32-tile element count) rather than a value derived from the lookup-table CB; loosely coupled to the actual page size.
- **Behaviorally-inert hashed / passed attributes** — `stable` is asserted `false` at the `ttnn::sort` entry (`sort.cpp:232`) yet is still threaded to the compute CTA and left in the hashed `SortParams`; `dim` is constrained to the last axis (`-1` or `rank-1 = 3`) at the device op, so the two legal values hash distinctly for identical behavior. Minor cache/plumbing cruft; not porter work.

## Per-DeviceOperation attribution

Single `DeviceOperation` (`SortDeviceOperation`); attribution above is per-factory. All three factories clear all gates. Distinguishing facts: `CrossCoreDataExchange` is the `WorkloadDescriptor` + op-owned-tensor + secretly-SPMD factory and carries the sole multi-binding CB; the other two are plain `descriptor` factories.

## Questions for the user  *(none)*

## Recipe notes  *(friction with the audit recipe itself)*

- **Dead Device-1.0 idiom at the Device 2.0 gate (now resolved).** The earlier RED turned on whether a *dead* `get_semaphore(sem_id)` raw-sem-address line gates the Device 2.0 prerequisite. The recipe's isolated-holdover template is written for a *live* **CB-index** free function with its wrapper in scope; a *dead* **semaphore-id** free function with no wrapper in scope fits neither the isolated-holdover nor the broad-Device-1.0 sub-category cleanly. It was resolved conservatively as a gate (a kernel containing a Device-1.0 raw-sem-address idiom is not fully Device 2.0 compliant, and the port whitelist cannot remove it), and the fix confirmed that reading was actionable — the line was simply deleted. A one-line clarification in the recipe ("a dead Device-1.0 idiom still gates, because the port cannot remove it off-whitelist") would remove the judgment call for the next auditor.
