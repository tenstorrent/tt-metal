# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding`

One `DeviceOperation` in this directory, with five program factories:

- **`ttnn::prim::UntilizeWithUnpaddingDeviceOperation`** (`device/untilize_with_unpadding_device_operation.{hpp,cpp}`)
  - `UntilizeWithUnpaddingSingleCoreProgramFactory` (`device/factories/untilize_with_unpadding_single_core_program_factory.cpp`)
  - `UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory` (`device/factories/untilize_with_unpadding_multi_core_interleaved_program_factory.cpp`)
  - `UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory` (`device/factories/untilize_with_unpadding_multi_core_block_interleaved_program_factory.cpp`)
  - `UntilizeWithUnpaddingMultiCoreNDShardedProgramFactory` (`device/factories/untilize_with_unpadding_multi_core_nd_sharded_program_factory.cpp`)
  - `UntilizeWithUnpaddingMultiCoreShardedProgramFactory` (`device/factories/untilize_with_unpadding_multi_core_sharded_program_factory.cpp`)

Also present in the directory but **not referenced by any factory**:
`device/factories/untilize_with_unpadding_multi_core_shared_variables.hpp` — a legacy `KernelHandle`-holding shared-variables struct from the pre-`ProgramDescriptor` era; it is listed in `ttnn/cpp/ttnn/operations/data_movement/CMakeLists.txt:324` but included by nothing. Out of audit scope; see *Misc anomalies*.

A separate, independent copy of this op exists at `ttnn/cpp/ttnn/operations/experimental/quasar/untilize_with_unpadding/` (a legacy-Quasar port). It is **not** part of this audit — porting legacy Quasar tests/ops is explicitly out of scope for `audit/metal2_audit.md`. It is mentioned only because greps for this op's kernels hit it.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `66ac84052d4 2026-07-27 docs(metal_2.0): split the runtime-args porting gate into its two sheet columns`

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding` |
| **Overall** | **RED at op level; subset of 4 factories is clear** (all but `MultiCoreSharded`) |
| **DOps / Factories** | `UntilizeWithUnpaddingDeviceOperation` → SingleCore · MultiCoreInterleaved · MultiCoreBlockInterleaved · MultiCoreNDSharded · **MultiCoreSharded (blocked)** |
| *Prereqs* — Device 2.0 (every kernel used) | **No (RED)** — 2 donor kernels still on Device 1.0 idioms, both instantiated **only** by `MultiCoreSharded`. Routed to the Device 2.0 track. Not isolated CB-index holdovers: neither kernel has any Device 2.0 wrapper object in scope, so each needs a small **full** migration (~10 call sites each). |
| *Prereqs* — Cross-op escapes | Ok — every function-call escape is ✓ (Device 2.0-native `Noc` + `TensorAccessor` shapes, or `uint32_t` DFB-index NTTPs). Heavy **file-path** borrowing though: 9 of the op's 12 distinct kernels are borrowed, several shared with 15+ other factories. See *Team-only*. |
| *Feature Support* — overall | **GREEN** — no Appendix A entry fires |
| *Feature Support* — Variadic-CTA | Ok — every `get_compile_time_arg_val` index is a literal or a `constexpr` accessor offset; `tensor_args_t` is a single `Tensor` |
| *TTNN Readiness* — `Is able to port?` (the gate) | **NOT VERIFIED — readiness sheet could not be fetched this run** (Google Drive MCP connector is unauthorized in this non-interactive session). Every *cheaply-checkable* conjunct was verified against the code and is clean; the sheet-owned `Is safe to port?` axis and `TensorParameter relaxation` are unread. See *Gate detail* and *Questions*. |
| *TTNN Readiness* — Concept (current) | `descriptor` — all five factories expose `static ProgramDescriptor create_descriptor(params, const Tensor&, Tensor&)` (each `*_program_factory.hpp:14`) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A — not a `WorkloadDescriptor` op |
| *TTNN Readiness* — Is safe to port? | **Unknown** — sheet-owner judgment, not fetchable this run. Code-side note: the op carries **no** `->address()`-in-RTA smuggled pointers (see *Port-work summary*), which is the most common source of a `no` here. |
| *TTNN Readiness* — Custom hash | No — no `compute_program_hash` anywhere in `device/` |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No — hook absent from the device-op |
| *TTNN Readiness* — `override_runtime_arguments` | No — method absent from every factory and from the device-op |
| *TTNN Readiness* — Pybind `create_descriptor` | No — `untilize_with_unpadding_nanobind.cpp` binds only the user-facing `ttnn::untilize_with_unpadding` free function |
| *TTNN Readiness* — Op-owned tensors | No — `descriptor` concept, no `buffers` vector |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` (no op-owned tensors) |
| *Port work* — Offset base pointer | **none — cleared.** No `->address()` fold anywhere; every tensor reaches a kernel as a bare `Buffer*` RTA, and every offset rides its own scalar arg |
| *Port work* — Tensor bindings (per binding) | **Case 1** for every binding in the clean subset (all consumed through a `TensorAccessor`); no Case 2 |
| *Port work* — TensorParameter relaxation | Unknown (sheet unread). No custom hash exists, so no relaxation can be active today |
| *Port work* — TensorAccessor 3rd arg | **drop (Class 2)** at both sites — one in the clean subset, one in the blocked factory |
| *Port work* — CB endpoints | **all legal (1P + 1C)** across the clean subset; one **self-loop** in the blocked `MultiCoreSharded` factory (`c_17`) |

**CB endpoints** are dispositions, not gates: every out-of-window CB has a port-time resolution. Recorded per `(CB, config)` below.

---

## Result

**RED at op level; subset `{SingleCore, MultiCoreInterleaved, MultiCoreBlockInterleaved, MultiCoreNDSharded}` is clear.**

The single blocker is the **Device 2.0 prerequisite**, and it is confined to one factory. `UntilizeWithUnpaddingMultiCoreShardedProgramFactory` instantiates two kernels that are still on Device 1.0 free-function idioms:

- `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` — the WIDTH/BLOCK-sharded-input → interleaved-output writer
- `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` — the compute kernel for the `unpad_tensor_w_16` fast path

Both live in the shared `ttnn/cpp/ttnn/kernel/` pool (a second shared-kernel pool, treated as shared-lib class). **Routed to the Device 2.0 migration team.** Neither is a one-line holdover — no Device 2.0 wrapper object exists in either file — but both are small: ~6 call sites in the writer, ~4 in the compute kernel. A directly analogous, already-migrated sibling exists for the compute kernel at `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/compute/eltwise_copy.cpp`, which shows exactly the target shape.

**Path forward:** the gate lifts as soon as those two kernels land on Device 2.0, at which point the whole op is re-audited (cheaply — everything else is already green) and ports as one unit. In the meantime the four clean factories can be ported today; a scoped brief for them is issued in `METAL2_PORT_BRIEF.md`.

**One open item that is not a code finding:** the per-factory readiness sheet could not be fetched in this session (the claude.ai Google Drive connector is not authorized here, and it cannot be authorized from inside a session). Every conjunct the recipe asks the auditor to *cross-check against code* was checked and is clean, but the sheet's own `Is safe to port?` call and the `TensorParameter relaxation` column are unread. **The sheet lookup must be completed before the subset port begins** — see *Questions for the user*.

---

## Gate detail

### TTNN factory concept (`Is able to port?`) — **not verified (sheet unavailable)**

The recipe requires this verdict to come from the TTNN team's *"Operations analysis"* sheet, fetched fresh via the claude.ai Google Drive MCP connector. In this session that connector is **unauthorized and unauthorizable** (non-interactive; the tool schema does not even resolve via `ToolSearch`), and no local CSV copy exists under `metal_2.0/analyses/`. This is neither a "conflict" nor a "missing op row" — it is a *fetch failure*, a case the routing table does not enumerate. It is recorded here rather than silently resolved either way.

What **was** verified, directly against the code (the recipe's "cheaply-checkable factual columns"):

| Column | Code evidence | Verdict |
|---|---|---|
| `Concept` | `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` on all five factories — `untilize_with_unpadding_single_core_program_factory.hpp:14`, `..._multi_core_interleaved_...hpp:14`, `..._multi_core_block_interleaved_...hpp:14`, `..._multi_core_nd_sharded_...hpp:14`, `..._multi_core_sharded_...hpp:14` | `descriptor` |
| `Custom hash` | no `compute_program_hash` in `device/` | `no` |
| `Runtime-args update (get_dynamic_runtime_args)` | hook absent from `untilize_with_unpadding_device_operation.hpp` (which declares exactly `select_program_factory`, `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`, `create_op_performance_model`) | `no` |
| `Override runtime args method? (PD and legacy)` | no `override_runtime_arguments` anywhere in the op | `no` |
| `Pybind descriptor` | `untilize_with_unpadding_nanobind.cpp:41-50` binds only `ttnn::untilize_with_unpadding`; no `create_descriptor` binding | `no` |
| `Secretly SPMD Workload?` | N/A (`descriptor`, not `WorkloadDescriptor`) | N/A |
| `Op-owned tensors?` | N/A — the `descriptor` concept cannot carry them; no `buffers` vector | `no` |
| **Factory-set match** | five factories in `program_factory_t` (`untilize_with_unpadding_device_operation.hpp:24-29`), five factory files, five `create_descriptor` definitions — self-consistent. **Cannot be checked against the sheet's row set** (that is the staleness check the fetch failure specifically defeats) | unchecked |

So five of the six shape conjuncts are code-confirmed clean. The two things only the sheet can supply — `Is safe to port?` and `TensorParameter relaxation` — remain open. Since the op has **no custom hash**, a real relaxation value cannot be active today (a relaxation *is* a custom hash excluding a property), so the practical residual risk is confined to `Is safe to port?`.

### Device 2.0 (every kernel used) — **RED**

Twelve distinct kernels are instantiated across the five factories. Ten are Device 2.0 compliant (`Noc`, `DataflowBuffer`/`CircularBuffer`, `CoreLocalMem`, endpoint structs, `TensorAccessor`). Two are not, and **both are instantiated only by `UntilizeWithUnpaddingMultiCoreShardedProgramFactory`**.

#### Violation 1 — `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp`

Owning pool: `ttnn/cpp/ttnn/kernel/dataflow/` (shared kernel pool). Instantiated at `untilize_with_unpadding_multi_core_sharded_program_factory.cpp:236`, on the branch `!cross_shard_type && !out_sharded && input layout != HEIGHT_SHARDED` — i.e. **WIDTH_SHARDED or BLOCK_SHARDED input → INTERLEAVED output**.

| File | Line | Call | Wrapper in scope |
|---|---|---|---|
| `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` | 24 | `cb_wait_front(cb_id_out0, block_width_ntiles)` | none |
| `…/writer_unary_stick_layout_interleaved_blocks.cpp` | 25 | `get_read_ptr(cb_id_out0)` | none |
| `…/writer_unary_stick_layout_interleaved_blocks.cpp` | 30 | `s.get_noc_addr(block_row_id, block_row_offset)` → raw `uint64_t` NoC address | none |
| `…/writer_unary_stick_layout_interleaved_blocks.cpp` | 31 | `noc_async_write(l1_read_addr, dst_noc_addr, block_row_size_unpadded)` | none |
| `…/writer_unary_stick_layout_interleaved_blocks.cpp` | 35 | `noc_async_write_barrier()` | none |
| `…/writer_unary_stick_layout_interleaved_blocks.cpp` | 36 | `cb_pop_front(cb_id_out0, block_width_ntiles)` | none |

**Sizing for the Device 2.0 team:** *broad Device 1.0, but small in absolute terms.* It is not the isolated-holdover shape (no wrapper object is in scope at any call site, so no 1-line substitution applies), yet it is also not a heavy migration: the kernel is 89 lines, uses **no** legacy addr-gen (`InterleavedAddrGen` / `ShardedAddrGen` / …) — it is already on `TensorAccessor` — and needs only a `Noc` + `DataflowBuffer`/`CircularBuffer` object introduction plus the six substitutions above. The `s.get_noc_addr(...) → noc_async_write(...)` pair becomes a single `noc.async_write(CoreLocalMem<uint32_t>(l1_read_addr), s, size, {}, {.page_id = …, .offset_bytes = …})`, exactly as this op's own `writer_unary_unpad_sharded_to_interleaved.cpp:58-63` already does.

**Co-borrower note:** within non-Quasar code this file is instantiated **only** by this op, so its migration has no cross-op coordination cost. (The Quasar clone at `experimental/quasar/untilize_with_unpadding/.../untilize_with_unpadding_multi_core_sharded_program_factory.cpp:203` also names it, but that tree is out of scope.)

#### Violation 2 — `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp`

Owning pool: `ttnn/cpp/ttnn/kernel/compute/` (shared kernel pool). Instantiated at `untilize_with_unpadding_multi_core_sharded_program_factory.cpp:265`, on the `unpad_tensor_w_16` fast path (`untilize_with_unpadding_multi_core_sharded_program_factory.cpp:47-48`).

| File | Line | Call | Wrapper in scope |
|---|---|---|---|
| `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | 20 | `cb_wait_front(tt::CBIndex::c_0, 1)` | none |
| `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | 21 | `cb_reserve_back(tt::CBIndex::c_16, 1)` | none |
| `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | 28 | `cb_pop_front(tt::CBIndex::c_0, 1)` | none |
| `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | 29 | `cb_push_back(tt::CBIndex::c_16, 1)` | none |

**Sizing:** trivial. An already-migrated sibling of this exact kernel exists at `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/compute/eltwise_copy.cpp:15-36` — same body, with `DataflowBuffer dfb_in(tt::CBIndex::c_0)` / `dfb_out(tt::CBIndex::c_16)` and the four calls moved onto the objects. The migration is a copy of that shape.

**Co-borrower note (this one *does* need coordination):** three other factories instantiate the same file and must be re-tested with it — `data_movement/copy/device/copy_same_memory_config_program_factory.cpp:40`, `data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_program_factory.cpp:191`, and `data_movement/sharded_partial/sharded_to_interleaved_partial/device/sharded_to_interleaved_partial_program_factory.cpp:192`. The migration is behaviour-preserving, so this is a test-coverage note, not a design coupling.

#### Kernels confirmed Device 2.0 compliant

| Kernel | Owner | Used by |
|---|---|---|
| `untilize_with_unpadding/device/kernels/dataflow/writer_unary_unpad_dims_split_rows.cpp` | this op | SingleCore |
| `untilize_with_unpadding/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multicore.cpp` | this op | MultiCoreInterleaved |
| `untilize_with_unpadding/device/kernels/dataflow/writer_unary_stick_layout_wh_multicore.cpp` | this op | MultiCoreBlockInterleaved |
| `untilize_with_unpadding/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp` | this op | MultiCoreNDSharded |
| `untilize_with_unpadding/device/kernels/dataflow/writer_unary_unpad_batch_rows_sharded.cpp` | this op | MultiCoreSharded |
| `untilize_with_unpadding/device/kernels/dataflow/writer_unary_unpad_width_16_sharded.cpp` | this op | MultiCoreSharded |
| `untilize_with_unpadding/device/kernels/dataflow/writer_unary_unpad_cross_sharded.cpp` | this op | MultiCoreSharded |
| `untilize_with_unpadding/device/kernels/dataflow/writer_unary_unpad_sharded_to_interleaved.cpp` | this op | MultiCoreSharded |
| `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | `eltwise/unary` | SingleCore, MultiCoreInterleaved |
| `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_wh_multicore.cpp` | `eltwise/unary` | MultiCoreBlockInterleaved |
| `eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` | `eltwise/unary` | MultiCoreSharded |
| `data_movement/sharded/device/kernels/dataflow/reader_unary_nd_sharded_blocks.cpp` | `data_movement/sharded` | MultiCoreNDSharded |
| `data_movement/untilize/device/kernels/compute/untilize.cpp` | `data_movement/untilize` | SingleCore, MultiCoreInterleaved, MultiCoreSharded |
| `data_movement/untilize/device/kernels/compute/untilize_wh.cpp` | `data_movement/untilize` | MultiCoreBlockInterleaved |
| `data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp` | `data_movement/untilize` | MultiCoreNDSharded |

The three `untilize*.cpp` compute kernels hold no CB calls of their own — they delegate entirely to `compute_kernel_lib::untilize<...>` from `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` (bucket 2, lib-team owned), passing DFB indices as `uint32_t` NTTPs.

**Sanctioned free functions observed — deliberately not flagged**, per the Green bullet's whitelist:

- `get_local_cb_interface(cb_id_in0).fifo_page_size` — `reader_unary_interleaved_start_id.cpp:20`
- `get_tile_size(cb_id_in0)` — `reader_unary_interleaved_wh_multicore.cpp:25`, `reader_unary_nd_sharded_blocks.cpp:21`
- `get_tile_size(cb_id_out)` — `writer_unary_unpad_width_16_sharded.cpp:22`

### Feature compatibility — **GREEN**

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `global_circular_buffer.hpp` include, no `.global_circular_buffer` field on any `CBDescriptor`, no `remote_index` / `remote_cb_*` / `remote_circular_buffer.h` idiom, no 4-arg `experimental::CreateCircularBuffer(..., global_cb)`. The op's two Buffer-backed CBs use the plain `.buffer` field (`untilize_with_unpadding_multi_core_sharded_program_factory.cpp:150, 179`), which is the ordinary borrowed-memory pattern → a mechanical `DataflowBufferSpec::borrowed_from` translation, explicitly *not* this entry. |
| CBDescriptor `address_offset` (non-zero) | N/A | No `CBDescriptor` in this op sets `.address_offset` (all four factories that build CBs leave it defaulted to 0). No `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. The only textual hit on the string is a *comment* at `untilize_with_unpadding_multi_core_sharded_program_factory.cpp:141` describing the framework's cache-hit re-apply of `UpdateDynamicCircularBufferAddress`, not a call. |
| GlobalSemaphore | N/A | The op uses **no semaphores at all** — `Semaphore`, `GlobalSemaphore`, `CreateSemaphore`, `global_semaphore.hpp` are all absent from the entire directory. |
| Variable-count compile-time arguments (CTA varargs) | N/A | Op-level cue does not fire: `tensor_args_t = Tensor` (`untilize_with_unpadding_device_operation.hpp:20`) — a single tensor, no `std::vector<Tensor>`. Kernel-level decider does not fire either: every `get_compile_time_arg_val` in every kernel the op uses takes a **literal** index, and the one computed accessor offset — `TensorAccessorArgs<dst_args.next_compile_time_args_offset()>()` at `writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp:42` — is `constexpr`, which the entry's guard explicitly excludes. The ND-sharded writer's one genuine variable-count loop reads **common runtime args**, not CTAs (`writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp:100-105`) — CRTA varargs are supported; see *Heads-ups*. |

### CB endpoints (GATE-free) — every CB legal or self-loopable

Counted **per CB, per node**, per instantiation. The op has **no semaphores anywhere**, which rules out the hidden-second-writer face (face (a)) structurally: a raw semaphore-gated co-fill cannot exist without a coordinating semaphore. It also instantiates no kernel source twice over one core range, ruling out the dual-instance work-split face (face (c)).

| Factory | CB | Config | Touchers on a node | Verdict |
|---|---|---|---|---|
| SingleCore | `c_0` | all | `reader_unary_interleaved_start_id` P (`:38,41`) · `untilize.cpp` C | **legal 1:1** |
| SingleCore | `c_16` | all | `untilize.cpp` P · `writer_unary_unpad_dims_split_rows` C (`:50-51,56-57,70`) | **legal 1:1** |
| MultiCoreInterleaved | `c_0` | all | reader P · exactly one compute instance C | **legal 1:1** |
| MultiCoreInterleaved | `c_16` | all | one compute P · `writer_unary_stick_layout_split_rows_multicore` C (`:41-42,50-52,61`) | **legal 1:1** |
| MultiCoreBlockInterleaved | `c_0` | all | `reader_unary_interleaved_wh_multicore` P · one compute C | **legal 1:1** |
| MultiCoreBlockInterleaved | `c_16` | all | one compute P · `writer_unary_stick_layout_wh_multicore` C (`:37-38,58`) | **legal 1:1** |
| MultiCoreNDSharded | `c_0` | all | `reader_unary_nd_sharded_blocks` P (`:32,40`) · `untilize_variable_num_blocks` C | **legal 1:1** |
| MultiCoreNDSharded | `c_16` | all | compute P · ND writer C (`:53,55,95`) | **legal 1:1** |
| MultiCoreSharded | `c_0` (borrowed, `.buffer = a.buffer()` @ `:150`) | all sharded | `reader_unary_sharded` P — `dfb.push_back` (`:16`) · compute C | **legal 1:1** |
| MultiCoreSharded | `c_16` | all sharded | compute P · the selected writer C (all four writer variants `wait_front`/`pop_front` on `c_16`) | **legal 1:1** |
| MultiCoreSharded | `c_17` (borrowed, `.buffer = output.buffer()` @ `:179`) | `out_sharded && !cross_shard_type` only | writer only — `reserve_back` + `get_write_ptr` + `push_back` (`writer_unary_unpad_batch_rows_sharded.cpp:29-30,53`, `writer_unary_unpad_width_16_sharded.cpp:32-33,102`) | **1 toucher → self-loop** (bind the writer PRODUCER *and* CONSUMER) |

Two notes on the census, both worth carrying to the porter:

- **The compute-kernel instances are node-disjoint by construction, so no node ever sees two.** `split_blocks_for_tilize` (`ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp:~120`) and `split_blocks_for_tilize_wh` (`…:171`) both build their sub-ranges by a monotonically advancing `core_index` over one enumerated core list, with `all_cores` defined as the union. So in MultiCoreInterleaved (2 compute descriptors) and MultiCoreBlockInterleaved (up to 4), every node carries exactly one. This is the check that would otherwise look like a same-role doubling.
- **`c_17` is genuinely single-ended.** It is the borrowed output buffer; the writer fills it and nothing drains it (the data is resident in L1 at kernel exit). It is *not* a dead CB — the writer's `reserve_back`/`get_write_ptr`/`push_back` are real accesses.

### Offset base pointers — **GREEN, cleared**

The op is **not** in the tables of `analyses/2026-07-19_offset_base_pointers.md`, and an independent scan of every address-bearing runtime arg confirms the clean-base finding rather than resting on that absence.

There is **no `->address()` expression anywhere in the op.** Every tensor address reaches a kernel through the `Buffer*`-binding form: the factory pushes the `Buffer*` object itself into `KernelDescriptor::RTArgList` and the framework auto-registers a `BufferBinding`. Sites:

- `untilize_with_unpadding_single_core_program_factory.cpp:187` (`src0_buffer`), `:190` (`dst_buffer`)
- `untilize_with_unpadding_multi_core_interleaved_program_factory.cpp:196` (`dst_buffer`), `:231` (`src0_buffer`)
- `untilize_with_unpadding_multi_core_block_interleaved_program_factory.cpp:295` (`src0_buffer`), `:300` (`dst_buffer`)
- `untilize_with_unpadding_multi_core_nd_sharded_program_factory.cpp:272` (`src0_buffer`), `:275` (`dst_buffer` **and** `src0_buffer`)
- `untilize_with_unpadding_multi_core_sharded_program_factory.cpp:308, 354, 413` (`dst_buffer`)

A `Buffer*` push carries no arithmetic — there is nowhere for a host-folded offset to hide. And the op consistently uses the *already-split* shape the triage doc describes as the fixed state: where an interior offset is needed, it rides its **own scalar RTA** and is added on the device side. Three examples, all clean:

- `untilize_with_unpadding_multi_core_sharded_program_factory.cpp:314` passes `info.col_shard_id * block_row_size` as a separate `col_byte_offset` arg; `writer_unary_unpad_cross_sharded.cpp:51` feeds it to `noc_async_write_sharded`'s `offset` parameter, while the accessor's base at `:35` is the bare `dst_addr`.
- `untilize_with_unpadding_multi_core_sharded_program_factory.cpp:422` passes `block_start_row_offset` separately; consumed as `block_row_offset` in the writer's page-offset argument.
- `untilize_with_unpadding_multi_core_block_interleaved_program_factory.cpp:303` passes `start_column_id` separately; `writer_unary_stick_layout_wh_multicore.cpp:51` uses it as `{.offset_bytes = start_column_id}`.

Type 3 (`address_offset`) is N/A per Appendix A above; Type 4 (`ttnn::narrow`) does not appear.

### TensorAccessor 3rd argument — **GREEN, both sites Class 2 (drop)**

`untilize_with_unpadding` does not appear in the op→class table of `analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md`. Both sites below post-date that analysis, so each is classified from first principles using the two questions. (The doc does note, at line 114, that the sibling `data_movement/untilize` op passes no 3rd arg at all — consistent, but not evidence about this op.)

Every other `TensorAccessor` in the op is 2-arg and needs no classification.

**Site 1 — `writer_unary_stick_layout_split_rows_multicore.cpp:34`** — `TensorAccessor(dst_args, dst_addr, writer_page_size)`, in the **clean subset** (`MultiCoreInterleaved`).

1. *Sharded or interleaved?* **Both are reachable.** `select_program_factory` (`untilize_with_unpadding_device_operation.cpp:58-60`) routes every sharded-output request to this factory, and interleaved output also lands here on the default path.
2. *Correct or wrong magnitude?* The value is computed at `untilize_with_unpadding_multi_core_interleaved_program_factory.cpp:108-113`. For interleaved and HEIGHT_SHARDED output it is `unpadded_row_size_bytes` = one full output row = the buffer's logical page. For WIDTH/BLOCK-sharded output it is `out_mem_config.shard_spec().value().shape[1] * output.element_size()` = one shard row = the buffer's logical page. **Correct magnitude in every config.**

The sharded sub-case is the one with no realignment safety net, so it is worth stating why the *exact* value holds there: the output shard spec this reads is the one `compute_output_specs` derives at `untilize_with_unpadding_device_operation.cpp:436-451`, where every shard width is `tt::round_up(..., tile_width)`. A tile-width-multiple element count gives ≥64 bytes in multiples of 64 for both bf16 (32×2) and fp32 (32×4) — already conformant to the strictest Blackhole DRAM alignment of 64, so `page_size == aligned_page_size` and the verbatim sharded stride is exact. → **Class 2, drop the arg.**

**Site 2 — `writer_unary_unpad_cross_sharded.cpp:35`** — `TensorAccessor(dst_args, dst_addr, writer_page_size)`, in the **blocked** `MultiCoreSharded` factory (`cross_shard_type` branch only).

1. *Sharded or interleaved?* **Sharded** — the branch exists only for WIDTH↔BLOCK-sharded output.
2. *Correct or wrong magnitude?* The value is `shard_spec.shape[1] * output.element_size()` (`untilize_with_unpadding_multi_core_sharded_program_factory.cpp:203`), taken from the **input's** shard spec. That is legitimate here rather than a mix-up: `validate_on_program_cache_miss` requires the input and output column shard widths to match on both cross directions (`untilize_with_unpadding_device_operation.cpp:165-171` for BLOCK→WIDTH, `:284-290` for WIDTH→BLOCK), so the input shard width *is* the output shard width. The input is TILE-layout, so its shard width is a tile multiple and the same 64-byte-alignment argument applies. → **Class 2, drop the arg.**

Neither site is Class 1 (the value does not vary with row width across cache-reused shapes — it is a shard-geometry constant, and there is no custom hash relaxing anything), so no `dynamic_tensor_shape` relaxation is implied.

---

## Port-work summary  *(mirrors the brief; scoped to the clean 4-factory subset)*

- **Tensor bindings** (per binding) — **every binding is Case 1**; there is no Case 2 and no borrowed-DFB `clean` binding in the subset:

  | Factory | Binding | Delivery today | Kernel consumption | Case |
  |---|---|---|---|---|
  | SingleCore | `input` | `Buffer*` RTA 0 @ `:187` | `TensorAccessor(src_args, src_addr)` — `reader_unary_interleaved_start_id.cpp:25` | 1 |
  | SingleCore | `output` | `Buffer*` RTA 0 @ `:190` | `TensorAccessor(dst_args, dst_addr)` — `writer_unary_unpad_dims_split_rows.cpp:44` | 1 |
  | MultiCoreInterleaved | `input` | `Buffer*` RTA 0 @ `:231` | `reader_unary_interleaved_start_id.cpp:25` | 1 |
  | MultiCoreInterleaved | `output` | `Buffer*` RTA 0 @ `:196` | `writer_unary_stick_layout_split_rows_multicore.cpp:34` | 1 |
  | MultiCoreBlockInterleaved | `input` | `Buffer*` RTA 0 @ `:295` | `reader_unary_interleaved_wh_multicore.cpp:27` | 1 |
  | MultiCoreBlockInterleaved | `output` | `Buffer*` RTA 0 @ `:300` | `writer_unary_stick_layout_wh_multicore.cpp:25` | 1 |
  | MultiCoreNDSharded | `input` | `Buffer*` RTA 0 @ `:272` **and** RTA 1 of the writer @ `:275` | `reader_unary_nd_sharded_blocks.cpp:27`; `writer_…_nd_sharded.cpp:43` (`accessor_src.shard_pages(...)`) | 1 (both consumers) |
  | MultiCoreNDSharded | `output` | `Buffer*` RTA 0 @ `:275` | `writer_…_nd_sharded.cpp:41` | 1 |

  The `input` binding in `MultiCoreNDSharded` is bound by **two** kernels (reader and writer) — both via `TensorAccessor`, so it is one `TensorParameter` with two `TensorBinding` consumers, not a Case-2 escape.

  Note on urgency: the `Buffer*` form is the framework's interim binding hack, patched correctly on cache hits today. This is **routine port work, not a correctness hazard** — there are no `->address()`-in-RTA smuggled pointers in this op to fix.

- **TensorParameter relaxation:** none applicable — the op has no custom hash, so no relaxation can be active. *(The sheet's column is unread this run; see Questions.)*

- **TensorAccessor 3rd arg:** drop the redundant page-size arg at `writer_unary_stick_layout_split_rows_multicore.cpp:34` (Class 2, in-subset). The second site, `writer_unary_unpad_cross_sharded.cpp:35`, is also Class 2 but sits in the blocked factory — it becomes port work when that factory unblocks. Neither is Class 1, so no `dynamic_tensor_shape`.

- **CB endpoints:** all legal 1P+1C across the subset — nothing to set, nothing to drop. *(Out-of-subset, for the eventual full port: self-loop `c_17` in `MultiCoreSharded` under `out_sharded && !cross_shard_type`.)*

---

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. The op has no semaphores (rules out the hidden-second-writer face structurally) and instantiates no kernel source twice over one core range (rules out the dual-instance work-split face). The only non-trivial disposition anywhere in the op is the `c_17` self-loop, and that is outside the clean subset.

- **Cross-op / shared kernels:** 9 of the op's 12 distinct kernels are borrowed by file path, and three of them are broadly shared — `reader_unary_sharded.cpp` (~19 factories), `reader_unary_interleaved_start_id.cpp` (~17), `untilize.cpp` (~10). Their Metal 2.0 rewrite is a **single** rewrite that every co-borrower must adopt in the same change. Full inventory in *Team-only*.

- **RTA varargs — two genuine cases**, both in the clean subset:

  1. **`writer_unary_stick_layout_split_rows_multicore.cpp:73-86`** (RTA, `MultiCoreInterleaved`). `n_block_reps` (RTA 3) bounds a loop that pulls a 5-tuple `{n_data, n_mixed, n_pads, times, repeat_count}` per group through a running `rt_arg_idx` advanced **inside** the loop (`:82`). Recognition shape (a). The producing side is `untilize_with_unpadding_multi_core_interleaved_program_factory.cpp:195-226`, where the group count varies per core with the block assignment. → **vararg block.** The four leading args (`dst_addr`, `padded_X_size`, `start_stick_id`, `n_block_reps`, `:19-22`) are ordinary fixed fields and should be **named**, not swept into the varargs.
  2. **`writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp:100-105`** (**CRTA**, `MultiCoreNDSharded`). Two `get_common_arg_val<uint32_t>(i)` loops bounded by the `tensor_rank` CTA read the output shape dims then the input shape dims. A CTA-bounded count still varies across instantiations, so per the recipe this is a vararg, not an unrolled name set. The producing side is `untilize_with_unpadding_multi_core_nd_sharded_program_factory.cpp:175-183`. → **CRTA vararg block** (the kernel-side vararg mechanism supports CRTAs).

  **Non-signal, called out so it is not mis-flagged:** `writer_unary_stick_layout_wh_multicore.cpp:65-70` re-reads args 2–7 inside the `third_dim` loop, but at **constant** indices — a fixed set of distinct fields read repeatedly, not a loop-indexed block. Name each.

---

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ⭐ blocked** — on file-path escape, not on function-call escape.

- Function-call escapes: **✓ clean.** Two donors, both with Device 2.0-native or DFB-index signatures.
- File-path kernel instantiation: **⭐** — the op owns 8 of its 12 distinct kernels' *sources* and file-path-instantiates 9 borrowed ones (one kernel, `untilize.cpp`, is borrowed by three factories). Two of those borrows carry the Device 2.0 gate; three are broadly shared enough to constitute a real port-together set.

#### Summary table — function-call escapes (`#include` outside the op directory)

| Op kernel | Donor file | Donor class | Status |
|---|---|---|---|
| `writer_unary_stick_layout_split_rows_multicore.cpp:12` | `ttnn/operations/data_movement/common/kernels/common.hpp` | 5 — in-family shared | ✓ |
| `writer_unary_unpad_cross_sharded.cpp:10` | `ttnn/operations/data_movement/common/kernels/common.hpp` | 5 — in-family shared | ✓ |
| *(all op + donor kernels)* | `tt_metal/hw/inc/api/**` (`dataflow_api.h`, `noc.h`, `dataflow_buffer.h`, `circular_buffer.h`, `endpoints.h`, `core_local_mem.h`, `noc_traits.h`, `dprint.h`) | 1 — LLK / HAL | ✓ |
| `untilize.cpp:7`, `untilize_wh.cpp:6`, `untilize_variable_num_blocks.cpp:7` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` | 2 — official shared kernel library | ✓ |

#### Per-call detail

| Donor | Function called | Handle shapes in signature | Status |
|---|---|---|---|
| `data_movement/common/kernels/common.hpp:294` | `tt::data_movement::common::noc_async_write_sharded(Noc noc, uint32_t l1_addr, AddrGenType tensor, uint32_t dest_id, uint32_t offset, uint32_t size)` | `Noc` (Device 2.0 native) + `TensorAccessor<DSpec>` by value (**Shape 1**). `l1_addr` is a plain L1 address obtained from `dfb.get_read_ptr()`, not a resource handle. | ✓ excellent — porter passes `TensorAccessor(tensor::name)` |
| `ttnn/kernel_lib/untilize_helpers.hpp` | `compute_kernel_lib::untilize<block_width_tiles, input_dfb, output_dfb, InitUninitMode, WaitMode, ReconfigureRegisterDatatypeMode>(num_blocks)` | CB identity as `uint32_t` **non-type template parameters** | ✓ OK — `dfb::name`'s constexpr cast covers template-parameter position |

Note the *deprecated* overload at `common.hpp:329` (same name, no leading `Noc`) — this op calls only the current `Noc`-first form, so there is nothing to migrate off.

#### Borrowed kernel files (file-path instantiation) — the port-together sets

| Kernel file | Owner | Also instantiated by | Coupling |
|---|---|---|---|
| `eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` | `eltwise/unary` (cross-family) | ~19 factories (typecast sharded, slice_write ×2, tilize ×2, …) | **large port-together set** |
| `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | `eltwise/unary` (cross-family) | ~17 factories (prod, typecast, nlp_create_qkv_heads_falcon7b, pad, …) | **large port-together set** |
| `data_movement/untilize/device/kernels/compute/untilize.cpp` | `data_movement/untilize` (in-family) | ~10 factories (upsample, fold, untilize ×4, …) | medium |
| `data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp` | `data_movement/untilize` (in-family) | ~6 factories | medium |
| `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_wh_multicore.cpp` | `eltwise/unary` (cross-family) | `data_movement/untilize` block factory | small |
| `data_movement/untilize/device/kernels/compute/untilize_wh.cpp` | `data_movement/untilize` (in-family) | `data_movement/untilize` block factory | small |
| `data_movement/sharded/device/kernels/dataflow/reader_unary_nd_sharded_blocks.cpp` | `data_movement/sharded` (in-family) | `data_movement/untilize` ND factory | small |
| `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | shared pool (class 3) | `copy`, `sharded_to_interleaved`, `sharded_to_interleaved_partial` | small — **also a Device 2.0 gate** |
| `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` | shared pool (class 3) | none (non-Quasar) | none — **but a Device 2.0 gate** |

The practical planning consequence: `untilize_with_unpadding` and `data_movement/untilize` share **five** kernels and should be sequenced as one unit; the two `eltwise/unary` readers pull a much wider set (~30 factories combined) whose Metal 2.0 rewrite has to be coordinated as a single change.

### Relaxation candidates mined from a custom hash

None — the op has no custom hash to mine.

### TTNN factory analysis

Sheet-derived facts could not be retrieved this run (see *Gate detail*). Code-side facts, with evidence:

- **Concept:** `descriptor`, uniformly across all five factories (`create_descriptor` returning `tt::tt_metal::ProgramDescriptor`).
- **Op-owned tensors:** none — structurally impossible on the `descriptor` concept, and no `buffers` vector exists.
- **MeshWorkload need:** none — no `WorkloadDescriptor` return anywhere.
- **Pybind `create_descriptor`:** absent. `untilize_with_unpadding_nanobind.cpp:41-50` binds only the public entry point with plain scalar/optional kwargs.
- **Other risky pybind:** none observed — no factory or device-op internals are exposed.
- **Custom hash:** absent (default hash over `UntilizeWithUnpaddingParams`).
- **`get_dynamic_runtime_args`:** absent.
- **`override_runtime_arguments`:** absent.
- **Target concept:** `MetalV2FactoryConcept`, no op-owned tensors.

---

## Misc anomalies  *(team-only, non-gating; route to the ops team — the port does not act on these)*

1. **Latent correctness bug — the `unpad_tensor_w_16` fast path is not guarded against interleaved output.** `untilize_with_unpadding_multi_core_sharded_program_factory.cpp:47-48` sets `unpad_tensor_w_16` from `!cross_shard_type && output W == 16 && output H % 32 == 0`, with no `out_sharded` condition. The **writer** selection consumes it only inside the `else if (out_sharded)` branch (`:210-218`), but the **compute** selection at `:263-267` swaps in `eltwise_copy.cpp` (a tiled face-preserving copy, *not* untilize) whenever the flag is set — including when output is interleaved. On that combination the writer (`writer_unary_unpad_sharded_to_interleaved.cpp` for HEIGHT_SHARDED input, or `writer_unary_stick_layout_interleaved_blocks.cpp` otherwise) would consume still-tiled data as if it were untilized row-major. The path looks reachable — e.g. HEIGHT_SHARDED tiled input of padded width 32 unpadded to width 16 with interleaved output passes every `validate_on_program_cache_miss` check. **Not hardware-verified in this audit**; flagged for the ops team to confirm and, if real, extend the guard to `!cross_shard_type && out_sharded && …` (matching the intent already stated in the comment at `:44-46`).
2. **Dead compile-time args in the ND-sharded writer.** `untilize_with_unpadding_multi_core_nd_sharded_program_factory.cpp:155-174` pushes 17 CTAs, but `writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp:25-39` reads only indices 0, 2–7 and 9–16. **CTA 1 (`output_stick_size`) and CTA 8 (`input_single_tile_size`) are never read.** Both are still computed on the host (`:154`, and `input_single_tile_size` at `:33`), so this is dead plumbing, not a mis-indexing bug — the surviving reads all use literal indices that match.
3. **Dead compile-time arg in the width-16 sharded writer.** `untilize_with_unpadding_multi_core_sharded_program_factory.cpp:211` pushes `{output_cb_index, sharded_output_cb_index, aligned_page_size}` for both same-shard-type writers, but `writer_unary_unpad_width_16_sharded.cpp:19-20` reads only CTAs 0 and 1 — `aligned_page_size` (CTA 2) is unused on that path. (The sibling `writer_unary_unpad_batch_rows_sharded.cpp:21-23` does read all three, which is why the shared arg list exists.)
4. **Dead local variables.** `untilize_with_unpadding_multi_core_interleaved_program_factory.cpp:147-148, 160-161, 175-176`: `full_compute_idx` and `cliff_compute_idx` are computed and immediately `(void)`-cast; nothing reads them. Leftovers from an earlier descriptor-ordering approach, superseded by the `desc.kernels.insert(begin, …)` calls at `:238-239`.
5. **Unreferenced header shipped in the build file list.** `device/factories/untilize_with_unpadding_multi_core_shared_variables.hpp` declares `UntilizeWithUnpaddingMultiCoreSharedVariables` (holding `KernelHandle`s and a core vector — the pre-`ProgramDescriptor` `override_runtime_arguments` idiom). No factory includes it; it survives only in `ttnn/cpp/ttnn/operations/data_movement/CMakeLists.txt:324`. Safe to delete.
6. **Author-flagged uncertainty left in the code.** `untilize_with_unpadding_multi_core_sharded_program_factory.cpp:60-61`: `// I am not sure it is correct to ever use the shard_spec here.` — immediately above `out_shard_spec = output.shard_spec().has_value() ? output.shard_spec().value() : shard_spec;`, which silently falls back to the *input's* shard spec for an unsharded output. Worth an owner's confirmation, since `out_shard_spec` feeds `block_row_size`, `num_rows_block` and `last_block_row_size_unpadded` for every writer on that factory.
7. **Unused debug includes.** `#include "api/debug/dprint.h"` with no `DPRINT` in the file: `writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp:12`, `reader_unary_sharded.cpp:9` (donor), `untilize_wh.cpp:5` (donor). Cosmetic.

---

## Per-DeviceOperation attribution

Not applicable — a single `DeviceOperation` in this directory. The per-**factory** split that matters is captured throughout: `MultiCoreSharded` is RED on Device 2.0; the other four are clear.

---

## Questions for the user

1. **The readiness sheet was not consulted — please complete this lookup before the subset port starts.** The audit recipe makes the sheet's `Is able to port?` cell the gate verdict, but the claude.ai Google Drive MCP connector is unauthorized in this session and cannot be authorized from inside one (`ToolSearch` does not even resolve `mcp__claude_ai_Google_Drive__download_file_content`), and there is no local CSV under `metal_2.0/analyses/`. All five *code-checkable* conjuncts were verified clean (see *Gate detail*); the two sheet-owned values — **`Is safe to port?`** and **`TensorParameter relaxation`** — are unread, as is the factory-set staleness cross-check. Authorizing the connector and re-running just this subject is a few minutes of work and would close the last open item on the clean subset. Since the op has no custom hash, a live relaxation is very unlikely; `Is safe to port?` is the one that could still change the verdict.
2. **Is the `unpad_tensor_w_16` + interleaved-output combination intended to be unreachable?** *(Misc anomaly 1, `untilize_with_unpadding_multi_core_sharded_program_factory.cpp:47-48, 263-267.)* If some caller-side or validation constraint I did not find rules it out, the finding is void; if not, it is a live wrong-data path independent of the port. Either way it is the ops team's, not the porter's — but it sits in the same factory the Device 2.0 gate blocks, so the two could be fixed on one pass.
3. **Should the two Device 2.0 kernel migrations be requested as one ticket?** Both live in the `ttnn/cpp/ttnn/kernel/` shared pool and both are small (~6 and ~4 call sites). `eltwise_copy.cpp` has three co-borrowers to re-test; the writer has none. Bundling them would let this op be re-audited once rather than twice.

---

## Recipe notes

1. **The brief-emission rule contradicts itself on config-scoped gates, and this op lands exactly on the seam.** The GATE role bullet says *"(A config-scoped GATE — e.g. GlobalCircularBuffer confined to one factory — still issues a brief for the clean subset; see Code-path scope.)"*, while both `Output: the two documents` and the `METAL2_PORT_BRIEF.md` section state flatly that the brief is *"emitted only on a fully GREEN audit … On any RED there is no brief."* I followed the more specific carve-out and issued a subset-scoped brief with a prominent scope banner, but the two statements should be reconciled — a porter reading only the second rule would treat the brief's existence as evidence the op is fully GREEN.
2. **Compute kernels are inside the Device 2.0 gate, but nothing in the recipe or the migration guide says so.** The gate text says *"every kernel this op exercises"*, yet it links a doc titled *"Device 2.0 **Data Movement** API Migration Guide"* whose examples are all dataflow, and the CB-endpoints subject's precondition lists only dataflow idioms. I resolved this empirically: ~20+ compute kernels in `ttnn/cpp/ttnn/operations/**/kernels/compute/` already use `DataflowBuffer`/`CircularBuffer` objects, and there is a migrated sibling of the very kernel I flagged (`data_movement/sharded/.../compute/eltwise_copy.cpp`). That was decisive but it took a codebase survey to establish. One sentence in the Device 2.0 subject — *"compute kernels count; the target idiom is a `DataflowBuffer`/`CircularBuffer` object around the `cb_*` calls"* — would settle it for the next auditor.
3. **There is no routing for "the readiness sheet could not be fetched."** The recipe enumerates `yes`, `no`, conflict, and missing-op-row, all of which presuppose a successful fetch. Fetch failure is a distinct and, for headless/non-interactive sessions, entirely predictable state — the connector is documented as authorizable only from an interactive main session. I recorded it as an explicit unresolved conjunct rather than defaulting it to RED (which would misroute a healthy op to a prereq team) or GREEN (which would launder an unchecked gate). A short bullet in the routing list saying which of those two the recipe prefers would remove the judgment call.
4. **The `Buffer*`-binding form deserves a mention in the *Offset base pointers* subject, not only in *TensorParameter analysis*.** That subject's recognition text is written entirely around `…->address() + <offset>` expressions. This op pushes bare `Buffer*` objects into the RTA list, which makes an offset fold *structurally* impossible — a stronger and cheaper clearing argument than scanning arithmetic. One line noting "a `Buffer*` push carries no arithmetic and is automatically clean-base" would let a future auditor dispose of this whole subject in a sentence for the growing number of ops on that form.
5. **Minor:** the *Dead CB* subject warns hard against over-calling a dead CB, but there is no equivalent note for the adjacent case I hit — a CB with exactly one toucher that *looks* dead because nothing drains it (`c_17` here: filled by the writer, resident at exit). The distinction (accessed-but-not-drained → self-loop; never-referenced → drop) is derivable from the classification table, but naming the resident-output shape explicitly under *Single-ended / sync-free* would speed it up.
