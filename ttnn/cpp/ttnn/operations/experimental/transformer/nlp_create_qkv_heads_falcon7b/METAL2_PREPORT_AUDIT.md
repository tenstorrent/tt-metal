# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_falcon7b`

- **`NlpCreateHeadsFalcon7BDeviceOperation`** (`device/nlp_create_qkv_heads_falcon7b_device_operation.hpp:17`)
  - *(no named factory)*: the op is in the **direct-descriptor** shape. `create_descriptor` is a static member of the device-op itself (`device/nlp_create_qkv_heads_falcon7b_device_operation.hpp:27-30`, body `device/nlp_create_qkv_heads_falcon7b_program_factory.cpp:19-155`). There is no `program_factory_t`, so the framework drives it through `MeshDeviceOperationAdapter::DirectDescriptorFactory` (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:170`, selected by `HasDirectDescriptor`, `ttnn/api/ttnn/operation_concepts.hpp:158`).
  - Kernels used:
    - reader, **borrowed** from `eltwise/unary`: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
    - writer, op-owned: `device/kernels/dataflow/writer_tm_tile_layout_nlp_create_qkv_heads_falcon7b.cpp`
  - No compute kernel. No unreferenced kernel files in the op directory.

The op splits a Falcon-7B fused QKV tensor `[B, 1, S, 4672]` (TILE, interleaved output only) into Q `[B, 71, S, 64]`, K `[B, 1, S, 64]` and V `[B, 1, S, 64]`. This is a fresh audit: no earlier `METAL2_PREPORT_AUDIT.md` existed. The op's last two commits are PD batch #57409 (`f5093e705ae`, which moved it to the direct descriptor and deleted `nlp_create_qkv_heads_falcon7b_program_factory.hpp`) and Device 2.0 cleanup #58858 (`8e04962ad72`, which removed one line from the writer).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(hash from the `Port_Recipe` checkout)*

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_falcon7b` |
| **Overall** | **GREEN (user waiver)**. The original finding was RED: the TTNN factory concept gate fails because the readiness sheet is stale (spreadsheet-broken). The user waived it on 2026-10-07. Every other gate is GREEN. |
| **DOps / Factories** | `NlpCreateHeadsFalcon7BDeviceOperation` → direct `create_descriptor` (no factory struct; the sheet still names `NlpCreateQkvHeadsFalcon7BProgramFactory`) |
| *Prereqs* — Device 2.0 (every kernel used) | Yes. The op's writer and the borrowed `eltwise/unary` reader are both Device 2.0. |
| *Prereqs* — Cross-op escapes | Ok. No function-call escapes outside `tt_metal/`. One borrowed kernel file (the reader) already has a `_metal2` fork. |
| *Feature Support* — overall | GREEN (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | Ok (none; every CTA is read at a fixed offset) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Waived by the user (2026-10-07).** Original finding: No, **spreadsheet-broken**. The cell reads `yes (with PD step)`, but the row conflicts with the code on `Concept` and names a phantom factory. See Gate detail. |
| *TTNN Readiness* — Concept (current) | Code: `descriptor` (direct-descriptor shape). Sheet: `legacy device-op` (stale). |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Custom hash | No (sheet `no`; no `compute_program_hash` in the device-op) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No (sheet `no`; no hook in the device-op) |
| *TTNN Readiness* — `override_runtime_arguments` | No (sheet `n/a`; none in the device-op) |
| *TTNN Readiness* — Pybind `create_descriptor` | No (sheet `no`). `nlp_create_qkv_heads_falcon7b_nanobind.cpp:19-28` binds only the user function. |
| *TTNN Readiness* — Op-owned tensors | No (sheet `no`; a plain `ProgramDescriptor` is returned) |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (sheet `Porting Target` agrees) |
| *Port work* — Offset base pointer | none (all four address RTAs are clean `Buffer*` bases) |
| *Port work* — Tensor bindings (per binding) | `input` Case 1 · `q` Case 1 · `k` Case 1 · `v` Case 1 |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | none: no accessor passes a 3rd arg |
| *Port work* — CB endpoints | legal: CB 0 is a plain 1:1 (reader PRODUCER, writer CONSUMER) |

## Result

**GREEN (user waiver) → brief issued.** On 2026-10-07 the user waived the TTNN factory concept gate: "Okay waive it if it's just the stale sheet". The original finding is kept below. The sheet refresh is still owed to the readiness-sheet owner.

*Original finding:* **RED → blocked on the TTNN factory concept gate (spreadsheet-broken), routed to the readiness-sheet owner.** The live readiness sheet (fetched 2026-10-07) still describes this op as it was before PD batch #57409:

- `Concept` = `legacy device-op`
- factory `NlpCreateQkvHeadsFalcon7BProgramFactory`, defined at the now-deleted `device/nlp_create_qkv_heads_falcon7b_program_factory.hpp`
- `Is able to port?` = `yes (with PD step)`

The code is already on a direct `create_descriptor`. This is the only RED. Device 2.0, Feature compatibility, Offset base pointers and TensorAccessor 3rd argument are all GREEN, and the informational subjects found nothing that needs more than routine port work. The blocker clears **without touching the op's code** (a sheet refresh), so a single row update unblocks the whole op. RED at op level; there is no subset to offer because the op has one code path.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **Waived by the user (2026-10-07)**, original finding kept. **RED: spreadsheet-broken**, routed to the readiness-sheet owner (Diego) to reconcile. Sheet row (`Op` = `experimental/transformer/nlp_create_qkv_heads_falcon7b`, `Device operation` = `NlpCreateHeadsFalcon7BDeviceOperation`, `Factory (variant)` = `NlpCreateQkvHeadsFalcon7BProgramFactory`). Two of the recipe's four broken-sheet triggers fire:
  1. **Primary-column conflict on `Concept`.** The sheet says `legacy device-op`. The code says `descriptor`: the device-op declares `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` (`device/nlp_create_qkv_heads_falcon7b_device_operation.hpp:27-30`), and nothing in the directory has `create()` + `override_runtime_arguments()`.
  2. **Phantom factory row.** No `NlpCreateQkvHeadsFalcon7BProgramFactory` type exists in the code. The sheet's `Factory definition path` (`device/nlp_create_qkv_heads_falcon7b_program_factory.hpp`) was deleted by #57409 (`f5093e705ae`: "…program_factory.hpp | 35 ------"). The only factory today is the framework shim `DirectDescriptorFactory`, which has no row.

  The `Is able to port?` cell reads `yes (with PD step)`, which fits a pre-PD-migration row. Per the routing rules, the conflict, not the cell, decides the gate. The other primary columns cross-check clean against the code: `Custom hash` `no`, `Backdoor custom hash` `no`, `Runtime-args update (get_dynamic_runtime_args)` `no`, `Override runtime args method?` `n/a`, `Pybind descriptor` `no`, `Op-owned tensors?` `no`. `Smuggled pointer` `no` also agrees: all four addresses ride as `Buffer*` bindings. `Known op issues` is empty, and `TensorParameter relaxation` is `none`. No cross-column invariant is violated. **Path forward:** refresh the row to `Concept` = `descriptor` with the post-#57409 factory and definition path. This is the same stale-row pattern seen on the other #57409 ops (`concatenate_heads`, `create_qkv_heads_from_separate_tensors`, `nlp_concat_heads_boltz`, `topk_router_gpt`).
- **Device 2.0 (every kernel used):** **GREEN.**
  - Writer (op-owned): uses `Noc noc` (`:14`), the `CircularBuffer cb_out0(cb_id_out0)` wrapper (`:42`, with `get_read_ptr`/`wait_front`/`pop_front` as methods, `:52,60,77,92,115`), `TensorAccessor` + `noc.async_write(CoreLocalMem…, accessor, …)` (`:38-40,62-67,79-84,94-99`), and `noc.async_write_barrier()` (`:114`). Its only CB-index free function is `get_tile_size(cb_id_out0)` (`:37`), which is sanctioned.
  - Reader (borrowed, `eltwise/unary`): uses `Noc`, `DataflowBuffer dfb(cb_id_in0)` and `noc.async_read(s, dfb, …)` (`reader_unary_interleaved_start_id.cpp:30-46`). Its only free function is `get_local_cb_interface(cb_id_in0).fifo_page_size` (`:25`), which is sanctioned.
  - There are no raw `noc_async_*` calls, no `*AddrGen*`, and no raw semaphore addresses in either kernel.
- **Feature compatibility:** **GREEN.** No gate fired. A grep for `GlobalCircularBuffer`, `GlobalSemaphore`, `address_offset` and `global_circular_buffer` over the op and the borrowed reader returned 0 hits.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | The one `CBDescriptor` (`program_factory.cpp:108-116`) has no `global_circular_buffer` field set. |
  | CBDescriptor `address_offset` (non-zero) | N/A | Not set (default 0). |
  | GlobalSemaphore | N/A | The op uses no semaphores at all. |

- **CB endpoints (GATE-free):** **legal.** There is one CB, index 0 (`program_factory.cpp:106-116`; `2 × per_tensor_tiles` = 292 tiles of `tile_size(dtype)`, page = one tile, over `all_cores`). The census on every node shows two touchers. The reader is a locked **producer** (`reserve_back`/`push_back`, reader `:43,46`). The writer is a locked **consumer** (`wait_front`/`pop_front`, writer `:60,77,92,115`); its `get_read_ptr()` peek (`:52`) is covered by its consumer binding. The writer's `cb_id_out0 = 0` is the same index as the reader's `cb_id_in0` (writer `:36`). Plain 1:1, no flag. There is only one config: output is always interleaved (`device_operation.cpp:34-37`), and the input path does not branch.
- **Offset base pointers:** **GREEN.** There are four address RTAs, each a bare `Buffer*` with no host-side arithmetic: reader RTA 0 `in0_buffer` (`program_factory.cpp:129-130`), and writer RTAs 0-2 `q_buffer`/`k_buffer`/`v_buffer` (`:139-141`). Per-core offsets travel as separate tile-index scalars (`num_blocks_written * per_tensor_tiles`, `q_out_tensor_tile_id`, `kv_out_tensor_tile_id`). The op is not in `2026-07-19_offset_base_pointers.md`; that doc's `nlp_create_qkv_heads` / `_boltz` entries are different ops. Outcome: no fold, op not in the tables, so it is clean.
- **TensorAccessor 3rd argument:** **N/A.** No accessor passes a 3rd argument (writer `:38-40` and reader `:30` are all 2-arg), so the subject never fires. The op is not in `2026-07-06_tensor_accessor_3rd_arg_triage.md`. (Commit `69eb869a7bf`, "Remove redundant page_size arg from TensorAccessor", appears in the op's history and presumably dropped an earlier one.)

## Port-work summary  *(mirrors the brief; produced because the RED clears off-code)*

The only blocker is cleared on the sheet side, so re-audit will read this same code. Per the scoping rule's "Elsewhere → run them" exception, the informational subjects were run in full.

- **Factory shape (forced, `ttnn_factory.md` §3).** The direct-descriptor shape must be converted to a conventional program factory: add a nested factory struct (the pre-#57409 name `NlpCreateQkvHeadsFalcon7BProgramFactory` fits) with `create_program_artifacts`, add `using program_factory_t = std::variant<…>;`, and remove the device-op-level `create_descriptor` (`device_operation.hpp:23-30`). The body stays in `device/nlp_create_qkv_heads_falcon7b_program_factory.cpp`, which is already listed in `experimental/transformer/sources.cmake:36`.
- **Tensor bindings** (per binding):
  - `input`: **Case 1**. A `Buffer*` RTA (`program_factory.cpp:130`) feeds `TensorAccessor(src_args, src_addr)` in the reader (`reader_unary_interleaved_start_id.cpp:16,20,30`). It becomes the fork's `tensor::src`.
  - `q`, `k`, `v`: **Case 1**. `Buffer*` RTAs (`:139-141`) feed `TensorAccessor(q_args/k_args/v_args, …)` in the writer (writer `:17-19,32-34,38-40`). Their `TensorAccessorArgs` CTAs (`:84-86`) disappear into the bindings.
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** all legal. CB 0 has the reader as PRODUCER (the fork's `dfb::in`) and the writer as CONSUMER.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. There is no hidden second writer, no semaphores, and no third toucher.
- **Cross-op / shared kernels:** the reader is borrowed from `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`. A `_metal2` fork **already exists** beside it (`reader_unary_interleaved_start_id_metal2.cpp`): rung 1, so bind it and do not re-fork. Its interface is `dfb::in`, `tensor::src`, `args::num_pages`, `args::start_id` and an optional `BACKWARDS` define (fork `:20-35`). It fits this op exactly: the legacy reader RTAs 1/2 are `num_blocks_per_core * per_tensor_tiles` and `num_blocks_written * per_tensor_tiles` (`program_factory.cpp:130`), and the factory sets no defines. The page size moves from `fifo_page_size` to `dfb.get_entry_size()`, which is the same value (one tile). Legacy-reader **sunset list** (not authorization to convert it in place), from a cheap path grep excluding quasar: `examples/example` (multi- and single-core), `examples/example_multiple_return`, `reduction/topk` (`topk_route_prep`), and this op. The writer is op-owned and bound by no other op.
- **RTA varargs:** none. The writer reads 7 RTAs at fixed indices (`:17-23`), and the fork's reader reads 2 named args.
- **Writer is on `CircularBuffer`, not `DataflowBuffer`.** The writer includes `api/dataflow/circular_buffer.h` (`:9`) and builds `CircularBuffer cb_out0(cb_id_out0)` (`:42`). The port's whitelisted CB→DFB swap applies (`DataflowBuffer` from `dfb::<name>`), and `get_tile_size(cb_id_out0)` (`:37`) moves to the DFB's tile-size accessor (whitelist rule 7).
- **Writer RTAs are mutated locally.** `q_out_h_dim`, `q_out_tensor_tile_id` and `kv_out_tensor_tile_id` are updated inside the block loop (`:104-112`). When converting them to named args, bind them to non-`const` locals.
- **Node order.** The factory walks cores column-major (`CoreCoord core = {i / num_cores_y, i % num_cores_y}`, `program_factory.cpp:119`), matching `split_work_to_cores`'s default. Keep that order for the per-node RTA values. The grid is the device's full `compute_with_storage_grid_size` (`:48`), so the tests run on the 8×8 Wormhole here (`tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads.py:99-114`, including a program-cache test).

## Team-only

- **Out-of-directory coupling & donor shape:** **✓ clean.**
  - *Function-call escapes:* none outside `tt_metal/`. The writer includes `api/dataflow/dataflow_api.h`, `noc.h`, `circular_buffer.h`, `api/core_local_mem.h` and `api/tensor/noc_traits.h` (all `tt_metal/*`, class 1). The reader includes `dataflow_api.h`, `noc.h`, `dataflow_buffer.h` and `noc_traits.h` (class 1).

    | Op kernel | Donor file | Class | Status |
    |---|---|---|---|
    | writer | `tt_metal/hw/inc/api/...` headers | 1 | ✓ |
    | reader (borrowed) | `tt_metal/hw/inc/api/...` headers | 1 | ✓ |

  - *Borrowed kernel files:* `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`, owned by `eltwise/unary`. It is broadly shared, and the `_metal2` fork beside it already has 8 Metal 2.0 consumers: `copy/typecast`, `data_movement/copy` (default tilized), `data_movement/pad` (tile), `data_movement/untilize` (two factories), `data_movement/untilize_with_unpadding` (two factories), and `reduction/prod` (all). Legacy consumers are listed in Heads-ups.
- **Relaxation candidates:** none (no custom hash).
- **TTNN factory analysis:** current concept `descriptor` (direct-descriptor, `device_operation.hpp:27-30`). No op-owned tensors and no MeshWorkload. No pybound `create_descriptor` (`nlp_create_qkv_heads_falcon7b_nanobind.cpp:19-28` binds only `ttnn.experimental.nlp_create_qkv_heads_falcon7b`), and no other risky pybind. No custom hash, no `get_dynamic_runtime_args`, no `override_runtime_arguments`. The device-op comment (`device_operation.hpp:23-26`) records why none is needed: the only per-dispatch state is the four buffer addresses. Target: `ProgramSpecFactoryConcept`, which matches the sheet's `Porting Target`.

## Questions for the user

1. **Waive the stale-sheet gate?** *Answered 2026-10-07: waived ("Okay waive it if it's just the stale sheet").* The only RED is the stale #57409 readiness-sheet row (`Concept` = `legacy device-op`, phantom `NlpCreateQkvHeadsFalcon7BProgramFactory`). If you waive it, this audit becomes "GREEN (user waiver)" with the original finding kept, and the brief is written with a waiver paragraph. The sheet refresh would still be owed to the readiness-sheet owner.

## Recipe notes

- **`Is able to port?` = `yes (with PD step)` is outside the documented vocabulary.** The routing rules cover `yes` and `no`, but this cell is a third value. Here the broken-sheet triggers decide the gate regardless, so it didn't matter. On a row whose primary columns *agreed* with a legacy concept, though, it is unclear whether `yes (with PD step)` should read as a clearing `yes` or as "blocked pending PD migration". Suggest listing it in `ttnn_op_porting_readiness.md`.
- **Recurring stale-row pattern for #57409 ops.** This is at least the fifth op from that batch to hit the same spreadsheet-broken RED. A note in the recipe that sheet refreshes lag PD batches might save auditors and launchers a round-trip.
