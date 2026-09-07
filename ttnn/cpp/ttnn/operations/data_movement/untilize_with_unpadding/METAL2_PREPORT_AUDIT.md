# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding`

One device-operation, five program factories, all on the `descriptor` concept (each declares
`static tt::tt_metal::ProgramDescriptor create_descriptor(...)` at its header's line 14):

- **`UntilizeWithUnpaddingDeviceOperation`** (`device/untilize_with_unpadding_device_operation.{hpp,cpp}`; params in `device/untilize_with_unpadding_device_operation_types.hpp`)
  - `UntilizeWithUnpaddingSingleCoreProgramFactory` — `device/factories/untilize_with_unpadding_single_core_program_factory.cpp`
  - `UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory` — `device/factories/untilize_with_unpadding_multi_core_interleaved_program_factory.cpp`
  - `UntilizeWithUnpaddingMultiCoreShardedProgramFactory` — `device/factories/untilize_with_unpadding_multi_core_sharded_program_factory.cpp`
  - `UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory` — `device/factories/untilize_with_unpadding_multi_core_block_interleaved_program_factory.cpp`
  - `UntilizeWithUnpaddingMultiCoreNDShardedProgramFactory` — `device/factories/untilize_with_unpadding_multi_core_nd_sharded_program_factory.cpp`

Dispatch is by memory layout plus a size heuristic (`select_program_factory`,
`device/untilize_with_unpadding_device_operation.cpp:40`): sharded input with a legacy 2D shard spec →
Sharded, without one → NDSharded; **any** sharded output from interleaved input → MultiCoreInterleaved
(forced, `:58`); `!use_multicore` → SingleCore; `!enough_space_height` → BlockInterleaved; the wide-row
heuristic (`:84`) → BlockInterleaved; otherwise MultiCoreInterleaved.

All **8** writer kernels under `device/kernels/dataflow/` are referenced by some factory — none is dead
code. The op owns **no** readers and **no** compute kernels; every one is borrowed (see
*Out-of-directory coupling*).

| Factory | Reader (borrowed) | Writer (op-owned) | Compute (borrowed) |
|---|---|---|---|
| SingleCore | `eltwise/unary/…/reader_unary_interleaved_start_id.cpp` | `writer_unary_unpad_dims_split_rows.cpp` | `untilize/…/compute/untilize.cpp` |
| MultiCoreInterleaved | `eltwise/unary/…/reader_unary_interleaved_start_id.cpp` | `writer_unary_stick_layout_split_rows_multicore.cpp` | `untilize/…/compute/untilize.cpp` |
| MultiCoreSharded | `eltwise/unary/…/reader_unary_sharded.cpp` | one of four, by config (below) | `untilize/…/compute/untilize.cpp`, or `ttnn/kernel/compute/eltwise_copy.cpp` on the W=16 fast path |
| MultiCoreBlockInterleaved | `eltwise/unary/…/reader_unary_interleaved_wh_multicore.cpp` | `writer_unary_stick_layout_wh_multicore.cpp` | `untilize/…/compute/untilize_wh.cpp` |
| MultiCoreNDSharded | `data_movement/sharded/…/reader_unary_nd_sharded_blocks.cpp` | `writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp` | `untilize/…/compute/untilize_variable_num_blocks.cpp` |

The Sharded factory picks its writer by configuration
(`…_multi_core_sharded_program_factory.cpp:204-240`):

| Config | Writer |
|---|---|
| cross-shard-type (WIDTH↔BLOCK, matching column shard width) | `writer_unary_unpad_cross_sharded.cpp` (`:208`) |
| same-type sharded output, W=16 fast path | `writer_unary_unpad_width_16_sharded.cpp` (`:216`) |
| same-type sharded output, general | `writer_unary_unpad_batch_rows_sharded.cpp` (`:218`) |
| HEIGHT_SHARDED → interleaved | `writer_unary_unpad_sharded_to_interleaved.cpp` (`:229`) |
| WIDTH/BLOCK_SHARDED → interleaved | `ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` (`:238`, borrowed) |

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `9c1a0466220 2026-09-07 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `UntilizeWithUnpaddingDeviceOperation` → SingleCore · MultiCoreInterleaved · MultiCoreSharded · MultiCoreBlockInterleaved · MultiCoreNDSharded |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 17 kernel files (8 own + 9 borrowed) are structurally Device 2.0; no holdovers |
| *Prereqs* — Cross-op escapes | **Ok** — one in-family function-call escape, ✓ excellent shape; 9 borrowed kernel files (coordination cost, not a gate) |
| *Feature Support* — overall | **GREEN** — every Appendix A entry `N/A` |
| *Feature Support* — Variadic-CTA | Ok — no `get_compile_time_arg_val` at a varying index anywhere |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes**, all five factory rows (value supplied by the launching user — see *Gate detail*) |
| *TTNN Readiness* — Concept (current) | `descriptor` (all five factories) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A — no `create_workload_descriptor` in the op |
| *TTNN Readiness* — Custom hash | **No** — no `compute_program_hash`, no backdoor `attribute_values` / `to_hash` |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** — hook absent from the device-op |
| *TTNN Readiness* — `override_runtime_arguments` | **No** — absent; base concept applies |
| *TTNN Readiness* — Pybind `create_descriptor` | **No** — `untilize_with_unpadding_nanobind.cpp` binds only `ttnn::untilize_with_unpadding`; nothing for the port to delete |
| *TTNN Readiness* — Op-owned tensors | **No** — a `descriptor`-concept op cannot carry them |
| *TTNN Readiness* — Target concept | **`ProgramSpecFactoryConcept`** |
| *Port work* — Offset base pointer | **none** — there is no `->address()` expression anywhere in the op; every base is delivered as a `Buffer*` binding |
| *Port work* — Tensor bindings (per binding) | Case 1 throughout, plus two `clean` borrowed-memory DFB reads — see *Port-work summary* |
| *TTNN Readiness* — TensorParameter relaxation | **`none`** on all five rows (clears; value supplied by the launching user) |
| *Port work* — TensorAccessor 3rd arg | **drop (Class 2)** at 2 sites — `writer_unary_stick_layout_split_rows_multicore.cpp:36`, `writer_unary_unpad_cross_sharded.cpp:35` |
| *Port work* — CB endpoints | **legal** everywhere except one **self-loop** (`c_17`, Sharded / same-shard-type sharded output), which is also a **conditional DFB** |

**CB endpoints** are dispositions, not gates. Every CB in this op is either an ordinary 1-producer /
1-consumer FIFO or the single-toucher `c_17` that takes a self-loop. No multi-binding flag is needed
anywhere, and no CB is dead.

## Result

**GREEN → brief issued.** Every gate clears: Device 2.0 ✓ · Feature compatibility ✓ · TTNN factory
concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd argument ✓ (both sites Class 2 — port work,
not a block). The porter brief is `METAL2_PORT_BRIEF.md`, beside this file.

Nothing is scoped out: all five factories and all of their configurations are portable.

## Gate detail

- **TTNN factory concept (`Is able to port?`): GREEN.**

  The gate value and the relaxation value were **supplied directly by the launching user** — *"Is able
  to port column is yes and tensor parameter relaxation column is none for all factories under this"* —
  rather than fetched from the readiness sheet in-session (this session has no Drive connector, so the
  sheet could not be pulled; recorded here so a reader knows the provenance of these two cells). The
  cross-checkable **primary** columns were verified against the code independently and all agree:

  | Column | Value | Code evidence | Agrees |
  |---|---|---|---|
  | `Concept` | `descriptor` | Five `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` declarations, one per factory header at `device/factories/*_program_factory.hpp:14`; each defined in its `.cpp`. No `create_workload_descriptor`, no `create()` + `override_runtime_arguments()` legacy pair, no `MetalV2` factory. | ✓ |
  | `Custom hash` | `no` | No `compute_program_hash` override on the device-op (`device/untilize_with_unpadding_device_operation.hpp:31-43` declares only `select_program_factory`, `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`, `create_op_performance_model`), and no backdoor `attribute_values` / `to_hash` anywhere in the op. | ✓ |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | Hook absent from the device-op; a grep of the whole op directory returns no `get_dynamic_runtime_args`. | ✓ |
  | `Override runtime args method?` | `no` | No `override_runtime_arguments` method on any factory. The only textual occurrence is a *comment* at `…_multi_core_block_interleaved_program_factory.cpp:292` noting the factory deliberately defines none. → base `ProgramSpecFactoryConcept`, nothing to translate. | ✓ |
  | `Pybind descriptor` | `no` | `untilize_with_unpadding_nanobind.cpp:38-48` binds only `ttnn::untilize_with_unpadding`; no `create_descriptor` binding. Nothing for the port to delete ⇒ **no user-visible API change from this column**. | ✓ |
  | `Secretly SPMD Workload?` | N/A | Only applies at `Concept == WorkloadDescriptor`, which this op is not. | ✓ |
  | `Op-owned tensors?` | `no` | Structurally impossible on the `descriptor` concept, and no `buffers` vector is constructed anywhere. Cross-column invariant (op-owned tensors ⇒ `WorkloadDescriptor`) holds. | ✓ |

  Cross-column invariants hold: `get_dynamic_runtime_args == no` is consistent with a `descriptor`
  concept, and `Op-owned tensors? == no` is required on it.

  **Factory-set match — partially checkable.** The code has exactly five factories, listed above. The
  user's statement covers *"all factories under this"*, so the row set is asserted to match; the
  individual sheet rows were not in hand to confirm one-to-one (no phantom row, no missing row) by
  inspection. This is a provenance limitation, not a finding — nothing observed in the code suggests a
  stale row, and the op's factory list is stable and fully enumerated in the device-op's
  `program_factory_t` variant (`device/untilize_with_unpadding_device_operation.hpp:24-29`). Flagged so a
  reader with sheet access can close it in one glance.

- **Device 2.0 (every kernel used): GREEN.** Every kernel this op instantiates — its own 8 writers and
  the 9 it borrows — is structurally Device 2.0: `Noc`, `DataflowBuffer` / `CircularBuffer` wrappers,
  `CoreLocalMem`, `UnicastEndpoint`, `TensorAccessor`. There is **no** raw `noc_async_read` /
  `noc_async_write`, no `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedAddrGenFast` /
  `InterleavedPow2AddrGen*`, no manual CB index management, and no raw semaphore address (the op uses
  no semaphores at all). There is no holdover table below because there are no holdovers.

  Three call sites *look* like CB-index-keyed holdovers and are **not** — each is on the sanctioned
  list, so none knocks the op out of Green:

  | File | Line | Call | Why it is not a violation |
  |---|---|---|---|
  | `eltwise/unary/…/reader_unary_interleaved_start_id.cpp` | 25 | `get_local_cb_interface(cb_id_in0).fifo_page_size` | Sanctioned free function — Device 2.0's own `CircularBuffer` wrapper implements `get_read_ptr()` / `get_write_ptr()` by calling it. |
  | `eltwise/unary/…/reader_unary_interleaved_wh_multicore.cpp` | 27 | `get_tile_size(cb_id_in0)` | Sanctioned free function. |
  | `device/kernels/dataflow/writer_unary_unpad_width_16_sharded.cpp` | 22 | `get_tile_size(cb_id_out)` | Sanctioned free function; a `DataflowBuffer` *is* in scope at the call site (`:30`), but per the recipe the sanctioned list does not turn on what object is in scope. |

  Confirmed against the Device 2.0 surface rather than the shape of the call site: the free
  `get_tile_size(operand)` (`tt_metal/hw/inc/api/dataflow/dataflow_api.h:280`) and
  `DataflowBuffer::get_tile_size()` (`.../dataflow_buffer.h:248`) read the *same* JIT
  `unpack_tile_size[]` descriptor array, so the port's move onto the object is a faithful swap, not a
  behavioural change. (That move is Metal 2.0 port work under kernel-side whitelist rule 7 — it does
  not shift the Device 2.0 boundary here.)

  Two kernels are on the Device 2.0 `CircularBuffer` wrapper rather than `DataflowBuffer`
  (`device/kernels/dataflow/writer_unary_unpad_sharded_to_interleaved.cpp:44` and the borrowed
  `ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp:77`, the latter also passing
  `CircularBuffer&` to its own file-local helper at `:15`). `CircularBuffer` is a Device 2.0 wrapper, so
  this is **not** a Device 2.0 gap — it is CB→DFB conversion work the Metal 2.0 port does anyway.
  Carried to the brief as a heads-up, not a blocker.

- **Feature compatibility:** every Appendix A entry, in order. All `N/A` — the feature is *absent*, so
  the entry cannot fire.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | **N/A** | No `GlobalCircularBuffer` type, `using` alias, or `CreateGlobalCircularBuffer` call. No `CBDescriptor` in any factory sets a `.global_circular_buffer` field (the arcane descriptor-API signal) — the five factories' `CBDescriptor` literals set only `total_size`, `core_ranges`, `format_descriptors` and, in the Sharded factory, `buffer`. No `remote_index(`, no `remote_cb_*` identifier, no `remote_circular_buffer.h` include, no `num_global_cb_receivers`. |
  | CBDescriptor `address_offset` (non-zero) | **N/A** | The field is never set on any `CBDescriptor` — it stays at its default zero everywhere. No `set_address_offset`, no four-argument `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. **Note the near-miss:** the Sharded factory *does* set `.buffer` (`…_multi_core_sharded_program_factory.cpp:152` and `:181`) — that is the **borrowed-memory** pattern, a mechanical porting-recipe translation via `DataflowBufferSpec::borrowed_from`, and explicitly not this entry. The only textual occurrence of `UpdateDynamicCircularBufferAddress` in the op is a comment at `:143`, not a call. |
  | GlobalSemaphore | **N/A** | The op uses no semaphores of any kind. A grep for `Semaphore` / `semaphore` across the op's `device/` tree and its kernel_lib donor returns zero hits — so neither `GlobalSemaphore` nor even a plain `CreateSemaphore` is present. |

- **CB endpoints (GATE-free): every CB is legal or carries a port-time disposition.** Census taken per
  CB, per node, per configuration. Endpoints counted include raw-pointer access, not just FIFO ops.

  | Factory | CB | Config | Census on a node | Verdict | Disposition |
  |---|---|---|---|---|---|
  | SingleCore | `c_0` | all | reader `push_back` (locked P) + compute `pop_front` (locked C) | plain 1:1 | none |
  | SingleCore | `c_16` | all | compute `push_back` (locked P) + writer `pop_front` (locked C) | plain 1:1 | none |
  | MultiCoreInterleaved | `c_0` | all | reader (locked P) + compute (locked C) | plain 1:1 | none |
  | MultiCoreInterleaved | `c_16` | all | compute (locked P) + writer (locked C) | plain 1:1 | none |
  | BlockInterleaved | `c_0` / `c_16` | full set | reader (locked P) + compute (locked C) / compute (locked P) + writer (locked C) | plain 1:1 | none |
  | BlockInterleaved | `c_2` / `c_17` | cliffrow set | same pair, on the cliffrow cores | plain 1:1 | none |
  | Sharded | `c_0` | all | reader `push_back` (locked P) + compute `pop_front` (locked C) | plain 1:1 | **borrowed-memory** → `borrowed_from` |
  | Sharded | `c_16` | all | compute (locked P) + writer (locked C) | plain 1:1 | none |
  | Sharded | `c_17` | same-shard-type sharded output only | writer only — `reserve_back` + `get_write_ptr` + `push_back` (locked P); **nothing drains it** | **single-ended** | **self-loop** + **borrowed-memory** + **conditional DFB** |
  | NDSharded | `c_0` | all | reader (locked P) + compute (locked C) | plain 1:1 | none |
  | NDSharded | `c_16` | all | compute (locked P) + writer (locked C) | plain 1:1 | none |

  Three things worth stating explicitly, because each is where a census usually goes wrong:

  1. **No hidden second writer anywhere.** Every kernel that touches a CB was scanned for a raw
     `get_write_ptr()` / `fifo_wr_ptr` co-fill by a non-FIFO-producer. There is none, and the shape
     that would coordinate one (a `reserve_done` / `write_done` semaphore pair) cannot exist here — the
     op declares no semaphores at all.
  2. **The two same-source kernel-instance pairs are *not* the dual-instance work-split.** The
     BlockInterleaved factory pushes `reader_unary_interleaved_wh_multicore.cpp` and
     `writer_unary_stick_layout_wh_multicore.cpp` into two `KernelDescriptor`s each (full set and
     cliffrow set, `…_multi_core_block_interleaved_program_factory.cpp:168-171`), and pushes the
     compute kernel four times (`:221-232`) — but every one of those covers a **disjoint** core set, and
     each set binds its **own** CB indices (`c_0`/`c_16` for full, `c_2`/`c_17` for cliffrow, assigned in
     `make_block_plan`, `ttnn/cpp/ttnn/operations/data_movement/common/common.cpp:906-923`). Each node
     therefore sees exactly one reader, one writer and one compute instance → ordinary 1:1, no
     assignment question. `buffer_set_for_core` (`common.cpp:928`) asserts the disjointness at build
     time. Similarly the MultiCoreInterleaved factory's full/cliff compute split
     (`…_multi_core_interleaved_program_factory.cpp:155-184`) is over disjoint core ranges.
  3. **`c_17` in the Sharded factory is single-ended, not dead.** Its one toucher is the writer, which
     produces into it and never drains it — the CB *is* the output buffer (`.buffer = output.buffer()`,
     `…_multi_core_sharded_program_factory.cpp:181`), so the data leaving is the point. One toucher →
     self-loop (bind the writer PRODUCER **and** CONSUMER; legal on Gen1 for DM). Both writers that
     reach it do the same thing: `writer_unary_unpad_batch_rows_sharded.cpp:29,30,53` and
     `writer_unary_unpad_width_16_sharded.cpp:32,33,102`.

  `c_17` is also **conditional**: the factory allocates it only under
  `out_sharded && !cross_shard_type` (`…_multi_core_sharded_program_factory.cpp:169`). In the other three
  Sharded configurations it does not exist at all. This is the easy case of a conditional DFB — the
  legacy factory *already* gates the allocation host-side, so the port translates an existing
  conditional rather than inventing new structure. It is **not** a dead-CB drop: in the configuration
  where it is allocated it is live.

- **Offset base pointers: GREEN.** No fold exists to split out — and, more strongly, **this op contains
  no address RTA at all**. A grep for `address()` across the entire op directory returns zero hits.
  Every tensor base reaches its kernel through the descriptor-API **`Buffer*`-binding form**: the
  factories push the `Buffer*` object itself into `emplace_runtime_args` / an `RTArgList`
  (e.g. `…_single_core_program_factory.cpp:187,190`; `…_multi_core_interleaved_program_factory.cpp:202,237`;
  `…_multi_core_block_interleaved_program_factory.cpp:294,300`; `…_multi_core_sharded_program_factory.cpp:310,356,415`;
  `…_multi_core_nd_sharded_program_factory.cpp:272,275`). There is therefore no expression into which a
  host offset *could* be folded, and no host arithmetic is performed on any base.

  Type 1 (raw offset arg) and Type 2 (accessor-fed offset arg): **absent**. Type 3 (`address_offset`) is
  the Appendix A row above and is absent. Type 4 (`ttnn::narrow` / `MeshBuffer::create(…, parent_base +
  offset)`): absent, and would not gate regardless.

  The one construct that superficially resembles a fold is **already the split-out form** the recipe
  describes as the fixed shape: the cross-shard writer receives a *clean* base as a `Buffer*` and a
  **separate scalar** `col_byte_offset = info.col_shard_id * block_row_size`
  (`…_multi_core_sharded_program_factory.cpp:316`), which the kernel passes as
  `noc_async_write_sharded`'s `offset` argument (`writer_unary_unpad_cross_sharded.cpp:51`) — never
  added into the accessor's base. Green.

  Cross-reference: the offset-base-pointer triage analysis
  (`analyses/2026-07-19_offset_base_pointers.md`, a dated prior) has **no entry** for this op. That
  agrees with the scan, and the scan is what decides it.

- **TensorAccessor 3rd argument: GREEN — 2 sites, both Class 2 (redundant → drop).** Sites found and
  classified, not "no sites": this op does pass a 3rd argument, in two kernels.

  The op is **not in** the triage doc's lookup table (`analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md`).
  Note the near-collision: that doc's line 114 records the *main* `data_movement/untilize` op as having
  no 3rd arg at all — a **different op**, and not a statement about this one. Both sites below are
  therefore classified from first principles, per the two questions.

  | # | Site | Config | Sharded or interleaved? | Value passed | Implicit `aligned_page_size` | Class |
  |---|---|---|---|---|---|---|
  | A | `device/kernels/dataflow/writer_unary_stick_layout_split_rows_multicore.cpp:36` | interleaved output | **interleaved** | `unpadded_row_size_bytes` = `output.padded_shape()[-1] * element_size` (`…_multi_core_interleaved_program_factory.cpp:110`) | identical value, rounded up | **2** |
  | A | same site | HEIGHT_SHARDED output | **sharded** | `dst_buffer->aligned_page_size()` verbatim (`:117`) | identical | **2** |
  | A | same site | BLOCK/WIDTH_SHARDED output | **sharded** | `out_mem_config.shard_spec().value().shape[1] * element_size` (`:115`) | identical | **2** |
  | B | `device/kernels/dataflow/writer_unary_unpad_cross_sharded.cpp:35` | cross-shard-type | **sharded** | `shard_spec.shape[1] * output.element_size()` (`…_multi_core_sharded_program_factory.cpp:205`) | identical | **2** |

  **Question 1 — sharded or interleaved.** Site A's accessor is on the output buffer, whose layout
  varies by config (this factory is forced for *any* sharded output from interleaved input,
  `device/untilize_with_unpadding_device_operation.cpp:58`), so all three rows were resolved separately.
  Site B's is always sharded (cross-type is WIDTH↔BLOCK by definition).

  **Question 2 — magnitude.** For a ROW_MAJOR tensor the page shape is `(1, physical_shard_width)` when
  a WIDTH/BLOCK shard spec is present and `(1, physical_width)` otherwise
  (`tt_metal/impl/tensor/spec/layout/page_config.cpp:101-126`), so `buffer->page_size()` is exactly
  `shard_width * element_size` for a W/B-sharded output and exactly the row size otherwise. Each value
  above therefore matches the true logical page:

  - *Interleaved row (Site A row 1):* equals `buffer->page_size()`, and the interleaved specialization
    realigns it via `InterleavedAddrGen` anyway — inert twice over.
  - *`aligned_page_size()` (Site A row 2):* literally the value the framework would supply.
  - *Shard width in bytes (Site A row 3, Site B):* equals `buffer->page_size()`. The sharded
    specialization uses the value **verbatim**, so this one deserves the extra step —
    `page_size == aligned_page_size` only if the value is already aligned. It always is: the output
    shard width is rounded up to `tile_width` (32) by `compute_output_specs`
    (`device/untilize_with_unpadding_device_operation.cpp:438` and `:446-448`) for Site A, and by the
    tiled input's own shard-shape requirement for Site B (where `validate_on_program_cache_miss`
    additionally pins input and output shard widths equal, `:166-171` and `:284-290`). The narrowest
    output element size is 2 bytes (BFLOAT8_B is converted to BFLOAT16 on output, `:418`), so the page
    is at least `32 × 2 = 64` bytes and always a multiple of 64 — aligned under **Blackhole DRAM (64)**,
    the strictest target, and *a fortiori* under L1 (16).

  Both sites are therefore genuinely redundant. Neither is Class 1 (nothing varies across cache-reused
  shapes that the framework would not also supply), neither is Class 3/4 (no wrong magnitude in any
  config), and neither is Special (no sharded raw-pack page, no sub-page base offset). **Port action:
  drop the argument at both sites**, and drop the now-dead CTA that feeds it — CTA index 2 in the
  MultiCoreInterleaved writer (`…_multi_core_interleaved_program_factory.cpp:124`) and CTA index 0 in
  the cross-sharded writer (`…_multi_core_sharded_program_factory.cpp:206`), with the kernels' shifted
  `TensorAccessorArgs<N>` offsets adjusted to match.

  Nine further `TensorAccessor` constructions in the op and its donors pass **no** 3rd argument and are
  outside this subject: `writer_unary_unpad_dims_split_rows.cpp:44`,
  `writer_unary_stick_layout_wh_multicore.cpp:26`, `writer_unary_unpad_sharded_to_interleaved.cpp:42`,
  `writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp:41,43`,
  `reader_unary_interleaved_start_id.cpp:30`, `reader_unary_interleaved_wh_multicore.cpp:29`,
  `reader_unary_nd_sharded_blocks.cpp:32`, `writer_unary_stick_layout_interleaved_blocks.cpp:74`.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per factory — classification varies by config in the Sharded factory):

  | Factory | Binding | Delivery today | Kernel use | Case |
  |---|---|---|---|---|
  | SingleCore | input | `Buffer*` RTA (`:187`) | `TensorAccessor` (`reader_unary_interleaved_start_id.cpp:30`) | **Case 1** |
  | SingleCore | output | `Buffer*` RTA (`:190`) | `TensorAccessor` (`writer_unary_unpad_dims_split_rows.cpp:44`) | **Case 1** |
  | MultiCoreInterleaved | input | `Buffer*` RTA (`:237`) | `TensorAccessor` | **Case 1** |
  | MultiCoreInterleaved | output | `Buffer*` RTA (`:202`) | `TensorAccessor` (`writer_unary_stick_layout_split_rows_multicore.cpp:36`) | **Case 1** |
  | BlockInterleaved | input | `Buffer*` RTA (`:295`) | `TensorAccessor` (`reader_unary_interleaved_wh_multicore.cpp:29`) | **Case 1** |
  | BlockInterleaved | output | `Buffer*` RTA (`:300`) | `TensorAccessor` (`writer_unary_stick_layout_wh_multicore.cpp:26`) | **Case 1** |
  | Sharded | input | **borrowed-memory CB** `c_0` (`.buffer = a.buffer()`, `:152`) | `reader_unary_sharded.cpp` only `push_back`s — the DFB *is* the tensor access | **clean** (causal-link gate) |
  | Sharded | output — same-shard-type sharded | **borrowed-memory CB** `c_17` (`.buffer = output.buffer()`, `:181`) | writer fills it by `get_write_ptr` | **clean** (causal-link gate) |
  | Sharded | output — cross-shard-type | `Buffer*` RTA (`:310`) | `TensorAccessor` (`writer_unary_unpad_cross_sharded.cpp:35`) | **Case 1** |
  | Sharded | output — HEIGHT_SHARDED → interleaved | `Buffer*` RTA (`:356`) | `TensorAccessor` (`writer_unary_unpad_sharded_to_interleaved.cpp:42`) | **Case 1** |
  | Sharded | output — W/B_SHARDED → interleaved | `Buffer*` RTA (`:415`) | `TensorAccessor` (`writer_unary_stick_layout_interleaved_blocks.cpp:74`) | **Case 1** |
  | NDSharded | input (reader) | `Buffer*` RTA (`:272`) | `TensorAccessor` (`reader_unary_nd_sharded_blocks.cpp:32`) | **Case 1** |
  | NDSharded | input (writer, for shard geometry) | `Buffer*` RTA (`:275`) | `TensorAccessor` (`…_nd_sharded.cpp:43`, used for `shard_pages()`) | **Case 1** |
  | NDSharded | output | `Buffer*` RTA (`:275`) | `TensorAccessor` (`…_nd_sharded.cpp:41`) | **Case 1** |

  **No Case 2 anywhere** — no kernel does hand-rolled NoC arithmetic on a tensor base. Every raw
  pointer in these kernels is a *CB* pointer (`get_read_ptr()` / `get_write_ptr()`), not a tensor base;
  the two borrowed-memory reads above are clean by the causal-link gate.

  **Urgency note.** Every base here is delivered as a `Buffer*`, never as `->address()`, so the
  framework already registers these as `BufferBinding`s and patches them on cache hits. This op does
  **not** carry the silent-wrong stale-pointer hazard; the Case-1 conversion is routine port work, not
  a correctness repair.

- **TensorParameter relaxation:** `none` (the sheet value supplied by the user). No relaxation is
  applied and no analysis doc is referenced.

- **TensorAccessor 3rd arg:** drop the redundant page-size argument at
  `writer_unary_stick_layout_split_rows_multicore.cpp:36` and `writer_unary_unpad_cross_sharded.cpp:35`,
  and drop the two host CTAs that feed them (`…_multi_core_interleaved_program_factory.cpp:124`,
  `…_multi_core_sharded_program_factory.cpp:206`). No site is Class 1, so no `dynamic_tensor_shape` is
  set.

- **CB endpoints:**
  - self-loop `c_17` (Sharded factory, same-shard-type sharded output — both the general and the W=16
    writer configurations)
  - borrowed-memory → `DataflowBufferSpec::borrowed_from`: `c_0` (Sharded, all configs) and `c_17`
    (Sharded, same-shard-type sharded output)
  - conditional DFB: `c_17` — exists only under `out_sharded && !cross_shard_type`; **not** a drop
  - all other CBs across all five factories: legal 1P+1C, no action
  - no multi-binding flag anywhere; no dead CB anywhere

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. The census found no CB with ≥3 touchers and
  no CB with two kernels locked to the same FIFO role, in any factory or configuration.
- **Cross-op / shared kernels:** the op owns none of its readers and none of its compute kernels — 9 of
  the 17 kernel files it instantiates are borrowed. Four already have a checked-in `_metal2` fork to
  bind; five do not and this port would create the first. Full inventory in *Team-only*.
- **RTA varargs:** two genuine variable-count blocks —
  `writer_unary_stick_layout_split_rows_multicore.cpp:75-88` (RTA) and
  `writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp:100-105` (CRTA). Detail below.
- **CB→DFB conversion in two kernels still on the `CircularBuffer` wrapper:**
  `writer_unary_unpad_sharded_to_interleaved.cpp:44` and the borrowed
  `writer_unary_stick_layout_interleaved_blocks.cpp:77` (which also passes `CircularBuffer&` to its own
  file-local helper, `:15`). Device 2.0-legal today; ordinary port work, not a donor-coupling issue.

### RTA varargs — detail

Both are **FYI-P**; Metal 2.0 supports RTA and CRTA varargs, so neither gates.

1. **`writer_unary_stick_layout_split_rows_multicore.cpp:75-88` — RTA vararg (shape (a), variable-count
   loop).** The kernel loops `n_block_reps` times (itself RTA 3, a genuine runtime value) and pulls five
   args per group through a running `rt_arg_idx` advanced **inside** the loop body (`:84`). The host
   side builds exactly this variable-length tail: `…_multi_core_interleaved_program_factory.cpp:218-232`
   emits five values per distinct `BlockRep` run, a count that varies per core and per shape. There are
   no per-argument names to infer → port the block with the vararg mechanism.

   **The four leading args are *not* part of the block** and must be named:
   `dst_addr` (0, and it becomes the tensor binding), `padded_X_size` (1), `start_stick_id` (2),
   `n_block_reps` (3) — all read at constant indices at the top of the kernel (`:19-22`), before the
   loop begins. This is the reverse of the pathology the recipe warns about (nameable scalars at the
   *tail* riding the varargs); here the fixed prefix simply precedes the block.

2. **`writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp:100-105` — CRTA vararg (shape (a),
   CTA-bounded loop).** Two back-to-back loops read the output shape then the input shape via
   `get_common_arg_val<uint32_t>(i)` and `get_common_arg_val<uint32_t>(i + tensor_rank)`, bounded by the
   CTA `tensor_rank` (`:39`). A CTA-bounded count still varies across instantiations, so this is a
   vararg, not an unrolled name set. Host side: `…_multi_core_nd_sharded_program_factory.cpp:175-183`
   pushes `output.padded_shape()` followed by `input.padded_shape()` into
   `writer_desc.common_runtime_args`.

**Non-signal, do not convert:** `writer_unary_stick_layout_wh_multicore.cpp:66-71` re-reads RTA indices
2-7 inside a `third_dim` loop, but the indices are **constant** — a fixed set of distinct fields read
repeatedly, which dissolves into named args. Likewise every other kernel in the op reads each RTA a
fixed number of times at a constant index.

**CTA varargs:** none. No kernel in the op or its donors calls `get_compile_time_arg_val` at a varying
index; every compile-time read is at a literal constant offset.

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ✓ clean.** Exactly one function-call escape exists across all 8 op-owned kernels,
and its shape is ✓ excellent. All other includes are `tt_metal/*` (`api/dataflow/*`, `api/tensor/*`,
`api/core_local_mem.h`, `api/debug/dprint.h`) — donor class 1, no concern. There is **no** ⚠, ✗ or ⭐
entry, so the per-call detail section is omitted.

**Summary table — function-call escapes (one row per op kernel × donor file):**

| Op kernel | Donor file | Donor class | Functions called | Shape | Status |
|---|---|---|---|---|---|
| `writer_unary_stick_layout_split_rows_multicore.cpp:12` | `ttnn/operations/data_movement/common/kernels/common.hpp` | 5 — in-family shared | `tt::data_movement::common::noc_async_write_sharded(Noc, uint32_t, AddrGenType, uint32_t, uint32_t, uint32_t)` | `Noc` by value (Device 2.0 native) + `TensorAccessor<DSpec>` by value (**Shape 1**) | **✓ excellent** |
| `writer_unary_unpad_cross_sharded.cpp:10` | same | 5 — in-family shared | same | same | **✓ excellent** |

The donor is on the Device 2.0 `Noc` object and takes the accessor by value, so the porter constructs
`TensorAccessor(tensor::name)` and passes it — no donor-side change, no fork of the header, no
`uint32_t sem_id` / `sem_addr` bridging problem, no old-style addr-gen (Shape 4). The remaining raw
`uint32_t l1_addr` parameter is a CB read pointer, not a resource handle, so it is outside the shape
table.

**Borrowed kernel files (file-path kernel instantiation).** The op instantiates 9 kernel files it does
not own. `_metal2` fork status is a locational test; `experimental/quasar/**` copies are excluded and do
not count as forks.

| Kernel file | Owner | Also instantiated by (sunset list) | `_metal2` fork beside it? |
|---|---|---|---|
| `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | eltwise/unary (cross-family) | `examples/example`, `examples/example_multiple_return`, `experimental/transformer/nlp_create_qkv_heads_falcon7b`, `reduction/topk` | **Yes** — `reader_unary_interleaved_start_id_metal2.cpp` |
| `eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` | eltwise/unary (cross-family) | `data_movement/tilize`, `data_movement/untilize`, `data_movement/sharded_partial/sharded_to_interleaved_partial`, `experimental/slice_write` | **Yes** — `reader_unary_sharded_metal2.cpp` |
| `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_wh_multicore.cpp` | eltwise/unary (cross-family) | `data_movement/untilize` | **No** — this port creates the first |
| `data_movement/sharded/device/kernels/dataflow/reader_unary_nd_sharded_blocks.cpp` | data_movement/sharded (in-family) | *(none — this op is the only consumer)* | **Yes** — `reader_unary_nd_sharded_blocks_metal2.cpp` |
| `ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` | `ttnn/cpp/ttnn/kernel/` (shared pool, class 3) | *(none — this op is the only consumer)* | **No** — this port creates the first |
| `data_movement/untilize/device/kernels/compute/untilize.cpp` | data_movement/untilize (in-family) | `data_movement/fold` | **Yes** — `untilize_metal2.cpp` |
| `data_movement/untilize/device/kernels/compute/untilize_wh.cpp` | data_movement/untilize (in-family) | `data_movement/untilize` | **No** — this port creates the first |
| `data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp` | data_movement/untilize (in-family) | `data_movement/untilize` | **Yes** — `untilize_variable_num_blocks_metal2.cpp` |
| `ttnn/kernel/compute/eltwise_copy.cpp` | `ttnn/cpp/ttnn/kernel/` (shared pool, class 3) | `data_movement/copy`, `data_movement/sharded/interleaved_to_sharded`, `data_movement/sharded_partial/interleaved_to_sharded_partial`, `data_movement/sharded_partial/sharded_to_interleaved_partial` | **Yes** — `eltwise_copy_metal2.cpp` |

The co-borrower column is the **coordination and sunset list**, not a must-port-together bundle: the
fork convention lets each op migrate independently, and the legacy copy is deleted when its last
consumer migrates. Four of the nine kernels already have a fork this port simply binds; five would be
forked for the first time here, which is where the cross-op cost of this port actually sits.

Also inventoried, though not borrowed: the compute kernels reach `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`
(donor class 2 — official shared kernel library, lib team's own). It is already fully `DataflowBuffer`-based
(`untilize_helpers.inl:198-268`), so it needs nothing from this port.

**A negative pointer, deliberately.** `ttnn/cpp/ttnn/operations/experimental/quasar/untilize_with_unpadding/`
contains a copy of this op carrying a deliberately hacky shortcut port, and quasar copies appear as
apparent co-borrowers of six of the nine kernels above. That tree is **out of bounds** — it is not a
production port, its `_metal2` files are whole-op pre-port copies that do **not** count as forks to
reuse, and nothing in it is evidence about the op in front of us. It has been excluded from every table
in this report; the porter should not read it either.

### Relaxation candidates

None. The op has no custom `compute_program_hash`, so there is no hash from which a candidate
relaxation could be mined.

### TTNN factory analysis

Sheet-derived facts with `file:line` evidence, in the form the TTNN ProgramFactory wiring consumes.
Every one of these is a **non-gating** fact here (the gate conjuncts are in *Gate detail*):

- **Current concept:** `descriptor` — one `static ProgramDescriptor create_descriptor(...)` per factory,
  declared at `device/factories/*_program_factory.hpp:14`.
- **Op-owned tensors:** none — structurally impossible on the `descriptor` concept.
- **MeshWorkload need:** none — no `create_workload_descriptor`, no `WorkloadDescriptor`, so the
  genuine-vs-op-owned-tensor-artifact question does not arise.
- **Custom hash:** absent → the port has no hash to leave alone.
- **`get_dynamic_runtime_args`:** absent.
- **`override_runtime_arguments`:** absent → base `ProgramSpecFactoryConcept`, no method to translate.
- **Pybind `create_descriptor`:** absent → nothing for the port to delete, and therefore **no
  user-visible API change** from this port.
- **Other risky pybind:** none. `untilize_with_unpadding_nanobind.cpp` exposes only the op function and
  its documented arguments; no descriptor internals are reachable from Python.
- **Target concept:** **`ProgramSpecFactoryConcept`** (`descriptor` + no op-owned tensors +
  `Override runtime args method? == no`).

## Misc anomalies  *(team-only, non-gating — route to the ops team; the port does not act on these)*

1. **Dead file — an orphaned shared-variables struct.**
   `device/factories/untilize_with_unpadding_multi_core_shared_variables.hpp` defines
   `UntilizeWithUnpaddingMultiCoreSharedVariables` (reader/writer `KernelHandle`s, a core vector, an
   `ncores`), and **nothing in the repository references either the header or the type**. It is a
   leftover from the pre-`ProgramDescriptor` cached-program era: the `descriptor` concept has no
   shared-variables channel. It also drags in `<tt-metalium/host_api.hpp>` (`:7`) for no consumer. Safe
   to delete on the ops track.

2. **Two dead compile-time args in the ND-sharded writer.** The factory emits 17 CTAs
   (`…_multi_core_nd_sharded_program_factory.cpp:155-174`), but the kernel never reads index **1**
   (`output_stick_size`, `:157`) or index **8** (`input_single_tile_size`, `:164`) — see
   `writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp:25-39`, which reads 0, 2-7 and 9-16 and
   skips exactly those two. They are computed on the host and consume CTA slots for nothing. Harmless
   today (the `TensorAccessorArgs<17>` offset already accounts for them), but removing them means
   renumbering the kernel's CTA reads, which is why this is ops-team work rather than something the port
   should scoop up.

3. **A dead compile-time arg on the W=16 sharded fast path.** The Sharded factory builds one CTA vector
   `{output_cb_index, sharded_output_cb_index, aligned_page_size}` for both same-shard-type writers
   (`…_multi_core_sharded_program_factory.cpp:213`), but `writer_unary_unpad_width_16_sharded.cpp` reads
   only indices 0 and 1 (`:19-20`) — `aligned_page_size` is dead in that configuration. It is live in
   the sibling `writer_unary_unpad_batch_rows_sharded.cpp:23`, so this is shared-vector convenience, not
   a bug.

4. **Dead local variables in the MultiCoreInterleaved factory.** `full_compute_idx` and
   `cliff_compute_idx` are declared (`…_multi_core_interleaved_program_factory.cpp:153-154`), assigned
   (`:166`, `:181`) and immediately discarded with `(void)` casts (`:167`, `:182`). Nothing consumes
   them. Their accompanying comment about descriptor positions is still accurate for the `insert`-at-
   front logic at `:244-245`, but the variables themselves are inert.

5. **Unused debug include.** `writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp:12` includes
   `api/debug/dprint.h` with zero `DPRINT` uses in the file. Same in the borrowed
   `eltwise/unary/…/reader_unary_sharded.cpp:14` and `untilize/…/compute/untilize_wh.cpp:5`.

6. **A self-flagged uncertainty left in the code.** `…_multi_core_sharded_program_factory.cpp:62` reads
   *"I am not sure it is correct to ever use the shard_spec here"*, immediately above the
   `out_shard_spec` fallback at `:63` that substitutes the **input's** shard spec when the output has
   none. The fallback is reachable — the interleaved-output branches of this factory take it — and
   `out_shard_spec` then drives `num_rows_block`, `block_row_size` and `last_block_row_size_unpadded`
   (`:76-84`). Worth a deliberate answer from the op's owner rather than leaving the comment standing.
   Not a portability issue: the port preserves the behaviour either way.

## Per-DeviceOperation attribution

Not applicable — the directory holds a single `DeviceOperation`. The per-factory and per-configuration
splits that *do* vary (the tensor-binding classification in the Sharded factory, the `c_17` disposition,
the two 3rd-arg sites) are attributed inline in the tables above.

## Questions for the user

1. **Readiness-sheet provenance:** the two gate-bearing cells (`Is able to port?` = `yes`,
   `TensorParameter relaxation` = `none`, all five factory rows) were taken from your launching message
   rather than fetched — no Drive connector is available in this session. Every cross-checkable primary
   column was verified against the code and agrees. If you have the sheet open, the one check I could
   not run is the **factory-set match**: confirming the sheet carries exactly five rows for this op and
   that they name the five factories listed at the top of this report, with no phantom or missing row.

## Recipe notes

1. **The `Buffer*`-binding form makes "Offset base pointers" and "TensorParameter analysis" scan for a
   construct that no longer exists in a fully-migrated op.** Both subjects are written around resolving
   `->address()` RTAs — *"You are already scanning address RTAs for TensorParameter analysis; the extra
   question here is whether an offset is folded into the base."* This op has **zero** `->address()`
   expressions; every base is a `Buffer*` pushed into an `RTArgList`. TensorParameter analysis does cover
   that shape explicitly (the `Buffer*`-binding form bullet), but the Offset base pointers subject does
   not mention it, so its recognition scan has nothing to run against and its "four outcomes"
   reconciliation table has no rows to fill. The verdict is unambiguous — a `Buffer*` is structurally
   incapable of carrying a folded offset, which is arguably a *stronger* green than "scanned and clean" —
   but the recipe leaves the auditor to reason that out. A sentence in Offset base pointers noting that
   the `Buffer*`-binding form forecloses the fold by construction would close it.

2. **The 3rd-arg subject's Class 2 test wants one more step for a *sharded* accessor, and the recipe's
   own wording nearly hides it.** Class 2 is *"`== aligned_page_size`, **or** a correct-magnitude value
   on an *interleaved* accessor"*. Three of the four config-rows I classified were sharded accessors
   fed the buffer's **unaligned** `page_size()` — correct magnitude, but on a specialization the recipe
   says uses the value *verbatim*, where correct-magnitude is explicitly not enough. Resolving them
   required going a step past the taxonomy: establishing that the op's own validation and output-spec
   derivation force the shard width to a tile multiple, hence the page to a 64-byte multiple, hence
   `page_size == aligned_page_size` on Blackhole DRAM. That is the right analysis and the two questions
   do get you there, but the Class 2 row reads as a lookup and this was not one. Worth an explicit
   sub-case: *a sharded accessor passed `buffer->page_size()` is Class 2 only if that value is already
   alignment-aligned — show the alignment, don't assume it.*

3. **"Two same-source `KernelDescriptor`s" needs its disjoint-node case stated where the census is
   taken, not only in the anti-pattern cross-reference.** The CB endpoints subject describes the
   dual-instance work-split (face (c)) as *"the dominant two-toucher shape"* and points at the
   demoting-per-group-CTA anti-pattern for the disjoint variant. This op hits the disjoint variant three
   times over (two reader instances, two writer instances, four compute instances in the
   BlockInterleaved factory) and it is emphatically *not* a two-toucher shape — each node sees one
   instance, and the two buffer sets even use different CB indices. The recognition cue the recipe gives
   for face (c) (*"the same `kernel_source` into two `KernelDescriptor`s… over one `core_ranges`"*) does
   disambiguate on the "one `core_ranges`" clause, but it is easy to match on the first half and stop.
   Given how cheaply a factory can be checked for disjointness, a one-line "first, check whether the
   instances' core ranges intersect" ahead of the three faces would prevent the false positive
   outright.
