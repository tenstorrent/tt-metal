# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/untilize`

This directory contains **two independent `DeviceOperation`s**, both dispatched from the single host entry `ttnn::untilize` ([untilize.cpp:127](untilize.cpp)). They share **no** factories or kernels, but they are two implementations of one op, so they are audited together with per-DeviceOperation attribution.

- **`ttnn::prim::UntilizeDeviceOperation`** (native) — `device/untilize_device_operation.hpp`, 8 program factories:
  - `UntilizeSingleCoreProgramFactory` (`device/factories/untilize_single_core_program_factory.cpp`)
  - `UntilizeMultiCoreProgramFactory` (`device/factories/untilize_multi_core_program_factory.cpp`)
  - `UntilizeMultiCoreNDShardInputProgramFactory` (`device/factories/untilize_multi_core_nd_shard_input_program_factory.cpp`)
  - `UntilizeMultiCoreParallelizeColumnProgramFactory` (`device/factories/untilize_multi_core_parallelize_column_program_factory.cpp`)
  - `UntilizeMultiCoreSubCoreGridsProgramFactory` (`device/factories/untilize_multi_core_sub_core_grids_program_factory.cpp`)
  - `UntilizeMultiCoreInputAndOutputShardTypeAndShardSpecIdenticalProgramFactory` (`..._shard_type_and_shard_spec_identical_program_factory.cpp`)
  - `UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory` (`..._nd_shard_type_and_shard_spec_identical_program_factory.cpp`)
  - **`UntilizeMultiCoreBlockProgramFactory`** (`device/factories/untilize_multi_core_block_program_factory.cpp`) — **BLOCKED** (see gate detail)
- **`ttnn::prim::UntilizeCodegenDeviceOperation`** (codegen) — `codegen/untilize_codegen_device_operation.hpp`, 1 program factory:
  - `UntilizeCodegenProgramFactory` (`codegen/untilize_codegen_program_factory.cpp`) — **BLOCKED** (no readiness-sheet coverage; see gate detail)

**Host dispatch:** `ttnn::untilize` routes to the **codegen** prim when `use_multicore && !sub_core_grids && supported_by_codegen(...) && !is_demoted(...)`, else the **native** prim. `supported_by_codegen` = TILE layout, interleaved (non-sharded) input+output, dtype bf16/bf8_b, within an L1 chunking threshold ([codegen/untilize_codegen_supported.cpp:32](codegen/untilize_codegen_supported.cpp)); `is_demoted` is hardwired `false` ([:129](codegen/untilize_codegen_supported.cpp)), so **codegen is the default path for its supported class** and native handles everything else.

**Unreferenced kernel files in the op directory (out of scope — no in-scope factory instantiates them):**
- `device/kernels/compute/untilize_w.cpp` — referenced only by an `experimental/quasar/*` factory (out of bounds); dead in this op.
- `device/kernels/compute/untilize_metal2.cpp` — the checked-in **Metal 2.0 fork** of the shared compute kernel `untilize.cpp`; currently bound by *other* ops (`data_movement/fold`), not by any untilize factory. Relevant to the port as the fork to reuse (see Team-only → shared kernels), but its contents are not audited here.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `548e18500b3 2026-08-18 docs(metal_2.0): a direct-descriptor op converts to a real program factory`

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/untilize` |
| **Overall** | **RED** (op level) — clean subset available |
| **DOps / Factories** | `UntilizeDeviceOperation` → 8 factories (7 clear, 1 RED) · `UntilizeCodegenDeviceOperation` → 1 factory (RED: no sheet row) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes (GREEN)** — all own + donor kernels Device-2.0 compliant (already DFB-aware) |
| *Prereqs* — Cross-op escapes | Ok — file-path borrows only; function-call escapes are `kernel_lib` (lib-team owned) |
| *Feature Support* — overall | **GREEN** |
| *Feature Support* — GlobalCircularBuffer / address_offset / GlobalSemaphore | N/A / N/A / N/A (none present) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Native: RED (config-scoped)** — 7 factories `yes`, `UntilizeMultiCoreBlockProgramFactory` `no` (`Known op issues = "Per-node CB size"`, tracked by [tenstorrent/tt-metal#51305](https://github.com/tenstorrent/tt-metal/issues/51305)). **Codegen: RED — no sheet row** (coverage gap → readiness-sheet owner) |
| *TTNN Readiness* — Concept (current) | `descriptor` (all 8 native factories + codegen factory; verified in code) |
| *TTNN Readiness* — Secretly SPMD | N/A (concept is `descriptor`, not `WorkloadDescriptor`) |
| *TTNN Readiness* — Custom hash | No (sheet `no` ×8; no `compute_program_hash` in either device op) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No (sheet `no` ×8; none in code) |
| *TTNN Readiness* — `override_runtime_arguments` | No (sheet `no` ×8; none in code) |
| *TTNN Readiness* — Pybind `create_descriptor` | No (nanobind binds only `untilize` + two verification-only `untilize_force_*`) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (sheet Porting Target; `Override runtime args = no`) |
| *Port work* — Offset base pointer | **none (GREEN)** — no `->address()` folds; all `Buffer*`-binding form |
| *Port work* — Tensor bindings (per binding) | Case 1 (`TensorAccessor`) or clean (borrowed-memory DFB); **no Case 2** |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | drop (Class 2) — 3 codegen sites only; native uses 2-arg form |
| *Port work* — CB endpoints | all legal 1P+1C (some borrowed-memory); no self-loop/multi-binding/dead |

**CB endpoints** are dispositions, not gates. Every CB in every factory is a plain 1-producer/1-consumer pair on each node (`c_0` reader→compute, `c_16` compute→writer); backed CBs resolve to borrowed-memory 1P+1C. No self-loop, multi-binding, or dead-CB dispositions arise.

---

## Result

**RED at op level.** Two independent blockers, each config/DeviceOperation-scoped; a substantial clean subset ports today:

- **Native `UntilizeDeviceOperation` — RED at op level; subset {`UntilizeSingleCoreProgramFactory`, `UntilizeMultiCoreProgramFactory`, `UntilizeMultiCoreNDShardInputProgramFactory`, `UntilizeMultiCoreParallelizeColumnProgramFactory`, `UntilizeMultiCoreSubCoreGridsProgramFactory`, `UntilizeMultiCoreInputAndOutputShardTypeAndShardSpecIdenticalProgramFactory`, `UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory`} is clear.** Only `UntilizeMultiCoreBlockProgramFactory` is blocked, on the readiness sheet's `Is able to port? = no` / `Known op issues = "Per-node CB size"` — the concrete prereq is [tenstorrent/tt-metal#51305](https://github.com/tenstorrent/tt-metal/issues/51305) (assigned bbradelTT, OPEN). All other gates (Device 2.0, features, offset base pointers, 3rd-arg) are GREEN for every factory. **A brief is issued for the clean 7-factory subset** (config-scoped GATE per Code-path scope).
- **Codegen `UntilizeCodegenDeviceOperation` — RED**, because the readiness sheet has **no row** for this DeviceOperation (it postdates the sheet's untilize analysis). This is a coverage gap → routed to the readiness-sheet owner to onboard it. The cheap code cross-check is clean (descriptor concept, no custom hash / dynamic RTA / override / pybind / op-owned tensors), and all *other* audit gates are GREEN for the codegen factory, so the codegen path is very likely portable once the sheet's authoritative verdict is available. **No brief is issued for codegen** until that verdict lands.

**Path forward:** the block factory unblocks when the "Per-node CB size" capability is available (route via the readiness-sheet `Known op issues` owner — ops team / port-recipe); the codegen op unblocks when the readiness-sheet owner adds its row. Neither is a code defect in the op.

---

## Gate detail

### TTNN factory concept (`Is able to port?`)

Readiness sheet ("TTNN Operations analysis", fetched live this session). **8 rows for `data_movement/untilize`, all `UntilizeDeviceOperation`, all `Concept = descriptor`.** Cross-check against code passed on every cheaply-checkable column:

| Column | Sheet (all 8 rows) | Code cross-check |
|---|---|---|
| `Concept` | `descriptor` | ✓ every factory defines `create_descriptor(...) → ProgramDescriptor` |
| `Custom hash` | `no` | ✓ no `compute_program_hash` in `device/` or `codegen/` |
| `Runtime-args update (get_dynamic_runtime_args)` | `no` | ✓ absent |
| `Override runtime args method?` | `no` | ✓ absent → target `ProgramSpecFactoryConcept` |
| `Pybind descriptor` | `no` | ✓ `untilize_nanobind.cpp` binds no `create_descriptor` |
| `Smuggled pointer` | `no` | ✓ (all pointer args are `Buffer*`-binding form — framework-patched) |
| `TensorParameter relaxation` | `none` | ✓ clears |
| `Op-owned tensors?` / `Secretly SPMD?` | empty / N/A | ✓ (`descriptor` concept) |

**Per-factory `Is able to port?`:**

| Factory | `Is able to port?` | Note |
|---|---|---|
| `UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory` | **yes** | clear |
| `UntilizeMultiCoreInputAndOutputShardTypeAndShardSpecIdenticalProgramFactory` | **yes** | clear |
| `UntilizeMultiCoreNDShardInputProgramFactory` | **yes** | clear |
| `UntilizeMultiCoreParallelizeColumnProgramFactory` | **yes** | clear |
| `UntilizeMultiCoreProgramFactory` | **yes** | clear |
| `UntilizeMultiCoreSubCoreGridsProgramFactory` | **yes** | clear |
| `UntilizeSingleCoreProgramFactory` | **yes** | clear |
| **`UntilizeMultiCoreBlockProgramFactory`** | **no** | `Known op issues = "Per-node CB size"` → **GATE** |

**Block factory GATE — attribution (code evidence):** `UntilizeMultiCoreBlockProgramFactory` allocates the **same CB `buffer_index`** (`c_0` input, `c_16` output) with **different `total_size` on different core ranges** — four `push_cb_pair` calls, one per core group with a distinct per-core tile count ([untilize_multi_core_block_program_factory.cpp:125-164](device/factories/untilize_multi_core_block_program_factory.cpp)): `core_range` uses `single_sub_block_size`; `cliff_col_row_core_range` and `cliff_row_core_range` use `single_block_size_cliff_row`; `cliff_col_core_range` uses `single_sub_block_size`. Metal 2.0's `DataflowBufferSpec` maps one buffer index to one spec whose size is uniform per node, so a per-node-varying CB size is what "Per-node CB size" names.

**Tracked by [tenstorrent/tt-metal#51305](https://github.com/tenstorrent/tt-metal/issues/51305)** — "Per-region-DFB-sizing decision in the multi-core block tilize/untilize family" (assigned **bbradelTT**, OPEN). The issue confirms this exact mechanism across the whole `_multi_core_block[_interleaved]` family (tilize, **untilize**, tilize_with_val_padding, untilize_with_unpadding) and adds that buffer size is a **correctness** property in these factories (a "hybrid FIFO-scratchpad chimera"), so a single max-sized DFB is *functionally incorrect*, not merely wasteful. It calls for a **prereq refactor** before the Metal 2.0 port — option 1 (split the DM kernels into separate kernels each with its own correctly-sized DFB — what quasar's tilize did) or option 2 (convert the hybrid to pure FIFO / pure scratchpad). The issue also warns that a quasar porter already shipped a **silent data corruption** improvising this sizing — do not attempt to size it by hand in the port. **Route:** ops team via #51305; the block factory is re-audited once that refactor lands. **No code defect in the current op** — the sheet cell is authoritative and read as-is. This affects **only** the block factory; the other 7 native factories and the codegen op size their CBs uniformly per core range and are unaffected.

**Codegen GATE — missing sheet row.** The sheet has **zero rows** for `UntilizeCodegenDeviceOperation` / `UntilizeCodegenProgramFactory` (grep for `codegen` returns only `data_movement/repeat/codegen`). Per the recipe's factory-set-match check, a factory present in code with no sheet row is a staleness/coverage signal → **spreadsheet is stale for this DeviceOperation → GATE → readiness-sheet owner** to onboard it. This is the mild, administrative kind: the codegen device op is brand new (2026, "phase 3/7" comments) and simply postdates the untilize analysis. Code cross-check is clean (see table above; verified against `codegen/untilize_codegen_program_factory.cpp` + `codegen/untilize_codegen_device_operation.hpp`). See *Recipe notes* for the two-DeviceOperations-in-one-dir wrinkle this exposes.

### Device 2.0 (every kernel used) — GREEN

Every kernel this op exercises — its own DM kernels, its compute kernels, and all borrowed/donor kernels — is Device-2.0 compliant. The DM kernels are in fact **already DFB-aware** (`DataflowBuffer dfb(cb_id)`, `Noc`, `TensorAccessor`), which is beyond baseline Device 2.0.

Free-function idioms observed, all **sanctioned** (not violations):

| File | Line | Call | Sanctioned? |
|---|---|---|---|
| `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_wh_multicore.cpp` | 25 | `get_tile_size(cb_id_in0)` | ✓ sanctioned |
| `data_movement/sharded/device/kernels/dataflow/reader_unary_nd_sharded_blocks.cpp` | 21 | `get_tile_size(cb_id_in0)` | ✓ sanctioned |
| `device/kernels/dataflow/reader_unary_sharded_blocks.cpp` | 49 | `get_tile_size(cb_id_in0)` | ✓ sanctioned |
| `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | 25 | `get_local_cb_interface(cb_id_in0)` | ✓ sanctioned |
| `codegen/kernels/reader_tile_interleaved_unified.cpp` | 75 | `get_local_cb_interface(cb_id)` | ✓ sanctioned |

The writer kernels' `dfb_out.get_read_ptr()` calls are **member methods on a `DataflowBuffer` object**, not the CB-index free function — not violations. Compute kernels (`untilize.cpp`, `untilize_wh.cpp`, `untilize_variable_num_blocks.cpp`, codegen `compute_untilize.cpp`) use the portable compute API (`compute_kernel_lib::untilize`, `pack_untilize_*`) with no DM idioms.

### Feature compatibility — GREEN

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | no `GlobalCircularBuffer`, `CreateCircularBuffer(..., global_cb)`, `.remote_index`, `remote_cb*`, or `.global_circular_buffer` field anywhere |
| CBDescriptor `address_offset` (non-zero) | N/A | no `.address_offset`, `set_address_offset`, or 4-arg `UpdateDynamicCircularBufferAddress` |
| GlobalSemaphore | N/A | no `GlobalSemaphore` / `CreateGlobalSemaphore` (no semaphores at all) |

A repo-wide grep of the op directory for all recognition signals returned zero matches.

### CB endpoints (GATE-free) — all legal 1P+1C

Every factory declares exactly two CBs — input `c_0` and output `c_16`. Per node, per config:

- **`c_0` (input):** reader **produces** (`dfb.reserve_back`/`push_back`, or a bare `push_back` when backed) + compute **consumes** (`wait_front`/`pop_front`) = 1 locked producer + 1 locked consumer → **plain 1P+1C**.
- **`c_16` (output):** compute **produces** + writer **consumes** (the writer's `dfb_out.get_read_ptr()` is a peek on its own consumer binding — one toucher) → **plain 1P+1C**.
- **Backed (borrowed-memory) CBs** — `UntilizeMultiCoreInputAndOutputShardTypeAndShardSpecIdentical...` (`c_0`←src0, `c_16`←dst), `UntilizeMultiCoreInputAndOutputNDShardType...` (both), and `UntilizeMultiCoreProgramFactory`'s even-sharded config (`c_0`←src0, [untilize_multi_core_program_factory.cpp:118](device/factories/untilize_multi_core_program_factory.cpp)) — resolve to the same 1P+1C, translated via `DataflowBufferSpec::borrowed_from` (mechanical port step).
- **Cliff/split-compute factories** (`UntilizeMultiCoreProgramFactory`, `...ParallelizeColumn`, block[RED], and codegen `build_main_split`/`build_with_unpadding`) place reader+writer over the **union** core range and per-group compute over **disjoint** ranges. On any single node only one compute instance is resident, so the per-node census is still 1P+1C. (The disjoint-core-group *placement* is a WorkUnitSpec-per-group port concern — see Heads-ups, not CB endpoints.)

No self-loop, multi-binding, dead-CB, or conditional-DFB dispositions arise.

### Offset base pointers — GREEN

No address RTA folds a host-side offset into a base. No `->address()` / `buffer()->address()` appears in any host factory; every factory passes the tensor as a `Buffer*` slot (framework `BufferBinding`, e.g. [untilize_single_core...cpp:185-188](device/factories/untilize_single_core_program_factory.cpp), [block...cpp:294-307](device/factories/untilize_multi_core_block_program_factory.cpp)) or, for backed-CB factories, no buffer slot at all. All offsets (`start_id`, `tile_start_id`, `start_shard_id`) are separate scalar args consumed as page indices through a `TensorAccessor`, never folded into a device address. (The offset-base-pointer triage doc lists no untilize entry; scan confirms clean.)

### TensorAccessor 3rd argument — GREEN (Class 2 drops, codegen only)

Native kernels all use the 2-arg `TensorAccessor(args, addr)` form — the subject does not fire for the native op. The **codegen** kernels pass a 3rd argument at three sites, all **interleaved** (codegen is non-sharded by construction) and all equal to `get_aligned_page_size()`:

| Site | 3rd arg expression | Class | Action |
|---|---|---|---|
| `codegen/kernels/reader_tile_interleaved_unified.cpp:77` | `source_page_size` (= `src_args.get_aligned_page_size()`, since the factory's `src_page_pitch` named CTA is hardwired `0` — [codegen program factory:148](codegen/untilize_codegen_program_factory.cpp)) | **2** (redundant) | drop the arg |
| `codegen/kernels/writer_untilize_interleaved.cpp:49` | `dst_args.get_aligned_page_size()` | **2** (redundant) | drop the arg |
| `codegen/kernels/writer_untilize_col_parallel.cpp:37` | `dst_args.get_aligned_page_size()` | **2** (redundant) | drop the arg |

All three pass exactly the value Metal 2.0 supplies implicitly → mechanical drop, **PORT WORK, not a gate**. (Porter nuance: the reader keeps the `source_page_size` local for its `source_read_size` clamp; only the accessor's 3rd argument and the now-vestigial `src_page_pitch` CTA go away.) The dated 3rd-arg triage doc lists no untilize entry; classification done from first principles per the two questions.

---

## Port-work summary  *(mirrors the brief; applies to the clean subset)*

- **Tensor bindings** (per binding, per factory): **all Case 1 (`TensorAccessor`) or clean (borrowed-memory DFB); no Case 2.**
  - Interleaved factories (`SingleCore`; `MultiCore` interleaved config; `ParallelizeColumn`; `SubCoreGrids`; `NDShardInput`; codegen): `src0` + `dst` → **Case 1**.
  - `MultiCore` even-sharded config: `c_0`←src0 backed → **clean**; `dst` → Case 1.
  - `...ShardTypeAndShardSpecIdentical` + `...NDShardType...Identical`: both `c_0`←src0 and `c_16`←dst backed → **both clean (borrowed-memory)**.
  - `NDShardInput` writer binds **both** `dst` (Case 1) and `src0` (Case 1 — it reads the input buffer for ND-shard page mapping, [writer_unary_stick_layout_split_rows_multi_core_nd_shard.cpp:34-37](device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core_nd_shard.cpp)).
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** drop the redundant page-size arg at the 3 codegen sites above (all Class 2; no `dynamic_tensor_shape` needed).
- **CB endpoints:** all legal 1P+1C. Translate the backed CBs (identical-shard, identical-nd-shard, multi_core even-sharded) via `DataflowBufferSpec::borrowed_from`.

## Heads-ups  *(mirrors the brief)*

- **WorkUnitSpec-per-group (cliff / split-compute):** `UntilizeMultiCoreProgramFactory` (full + cliff compute), `UntilizeMultiCoreParallelizeColumnProgramFactory` (full + cliff compute), and codegen `build_main_split` / `build_with_unpadding` (cg1 + cg2 compute) all run reader/writer over the union core range and per-group compute over disjoint ranges. The port must give **each core group its own `WorkUnitSpec`** containing reader + writer + that group's compute (a shared kernel is listed in each group's WU), never one WU over the union plus a narrower WU for a sub-range — else the disjointness invariant fires at test time (`program_spec.cpp` "overlap in target nodes").
- **Shared-kernel forks — two already exist; bind them, don't re-fork:**
  - `device/kernels/compute/untilize.cpp` → **`untilize_metal2.cpp` already checked in** (created by the `data_movement/fold` port). Used by clean-subset factories `SingleCore`, `SubCoreGrids`, `ParallelizeColumn`, `...ShardTypeAndShardSpecIdentical`.
  - `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` → **`reader_unary_interleaved_start_id_metal2.cpp` already checked in.** Used by `ParallelizeColumn`, `SubCoreGrids`.
- **Shared donors without a fork yet (this port creates the first `_metal2` fork; the co-borrower set is a *sunset list*, not a port-together bundle):** `data_movement/sharded/.../writer_unary_sharded.cpp` (9 co-borrower families), `eltwise/unary/.../reader_unary_sharded.cpp` (7), `data_movement/sharded/.../reader_unary_nd_sharded_blocks.cpp` (2), plus own kernels `reader_unary_start_id.cpp` (shared w/ copy, tilize) and `compute/untilize_variable_num_blocks.cpp` (shared w/ untilize_with_unpadding).
- **RTA varargs:** none. Every kernel reads a fixed set of RTAs at constant indices (or a fixed `ArgsBase` struct in the codegen reader); no loop-indexed or data-selected varargs. All args are nameable.
- **Codegen op owns all its kernels locally** (`codegen/kernels/*`) — no cross-op kernel coupling; the shared "unified reader" it derives from lives elsewhere (`data_movement/common/kernels/codegen/`, PR #52806) and untilize keeps a local simplified copy.

---

## Team-only

### Out-of-directory coupling & donor shape

**Function-call escapes:** the only out-of-directory `#include` that resolves to code the op *calls into* is `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` (compute kernels `untilize.cpp`, `untilize_variable_num_blocks.cpp`) — the **official shared kernel library** (`kernel_lib`), handled by the lib team; ✓ no concern. Op-level roll-up: **✓ clean.**

**Borrowed kernel files (file-path instantiation).** Summary — one row per (kernel file, disposition). `_metal2` fork column is the locational check (excludes quasar copies):

| Kernel file | Owner | Used by (untilize factories) | `_metal2` fork | Other non-quasar instantiators (sunset list) |
|---|---|---|---|---|
| `device/kernels/dataflow/reader_unary_start_id.cpp` | own | SingleCore, MultiCore (interleaved) | no | copy, tilize |
| `device/kernels/dataflow/reader_unary_sharded_blocks.cpp` | own | MultiCore (block-reader) | no | — (untilize only) |
| `device/kernels/dataflow/writer_unary_stick_layout_split_rows_single_core.cpp` | own | SingleCore | no | — |
| `device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp` | own | MultiCore | no | — |
| `device/kernels/dataflow/writer_unary_stick_layout_split_rows_interleaved_parallel_columns.cpp` | own | ParallelizeColumn, SubCoreGrids | no | — |
| `device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core_nd_shard.cpp` | own | NDShardInput | no | — |
| `device/kernels/compute/untilize.cpp` | own (shared) | SingleCore, SubCoreGrids, ParallelizeColumn, IdenticalShard | **YES** (`untilize_metal2.cpp`, bound by fold) | untilize_with_unpadding, pool/upsample, fold |
| `device/kernels/compute/untilize_variable_num_blocks.cpp` | own (shared) | MultiCore, NDShardInput, IdenticalNDShard | no | untilize_with_unpadding |
| `device/kernels/compute/untilize_wh.cpp` | own (shared) | **Block (RED)** | no | untilize_with_unpadding |
| `eltwise/unary/.../reader_unary_sharded.cpp` | cross-family | MultiCore (even-shard), IdenticalShard, IdenticalNDShard | no | sharded, sharded_partial, tilize, transpose, untilize_with_unpadding, slice_write |
| `eltwise/unary/.../reader_unary_interleaved_start_id.cpp` | cross-family | ParallelizeColumn, SubCoreGrids | **YES** (`..._metal2.cpp`) | copy, pad, untilize_with_unpadding, transformer, examples |
| `eltwise/unary/.../reader_unary_interleaved_wh_multicore.cpp` | cross-family | **Block (RED)** | no | untilize_with_unpadding |
| `data_movement/sharded/.../writer_unary_sharded.cpp` | cross-op | IdenticalShard, IdenticalNDShard, MultiCore(even-shard) | no | sharded, sharded_partial, tilize, tilize_with_val_padding, transpose, padded_slice, transformer, reduction/generic |
| `data_movement/sharded/.../reader_unary_nd_sharded_blocks.cpp` | cross-op | NDShardInput | no | untilize_with_unpadding |
| `untilize_with_unpadding/.../writer_unary_stick_layout_wh_multicore.cpp` | cross-op | **Block (RED)** | no | untilize_with_unpadding |

All donor kernels are Device-2.0 compliant (see Device 2.0 gate), so donor coupling induces **no scheduling block** — only the fork-coordination cost above. Per-call shape roll-up: **✓ clean / ⚠ workable** (shared kernels need `_metal2` forks; `DataflowBuffer`-typed handles cross cleanly). No `CircularBuffer&` / old addr-gen / `uint32_t sem_addr` donor signatures.

### TTNN factory analysis

Sheet-derived facts (all 8 native rows), with code evidence:
- **Target concept:** `ProgramSpecFactoryConcept` (`Override runtime args = no`, no op-owned tensors). Applies to the codegen factory too (`Override` absent in code).
- **No op-owned tensors, no MeshWorkload, no pybound `create_descriptor`, no custom hash, no `get_dynamic_runtime_args`, no `override_runtime_arguments`** — all verified against `device/` and `codegen/` sources; nothing gates on the TTNN wiring beyond the two blockers in the Result.
- `Op Classification = "PD Op (pointer-patching)"` on all native rows — the op relies on the framework's `Buffer*` binding-injection (patched on cache hit); the Metal 2.0 typed binding supersedes it. Not a defect.

### Relaxation candidates
None. (No custom hash to mine; relaxation column is `none`.)

---

## Misc anomalies  *(team-only, non-gating — route to ops team; the port does not act on these)*

1. **Latent cliff-core RTA arg-count mismatch in `UntilizeMultiCoreParallelizeColumnProgramFactory`.** The writer kernel `writer_unary_stick_layout_split_rows_interleaved_parallel_columns.cpp` reads exactly **6** runtime args (`dst_addr, num_sticks, num_tiles_per_core, tile_width_size, start_stick_id, offset_within_stick` — [:17-22](device/kernels/dataflow/writer_unary_stick_layout_split_rows_interleaved_parallel_columns.cpp)). Full cores are given 6 ([factory:204-211](device/factories/untilize_multi_core_parallelize_column_program_factory.cpp)) — correct. The **cliff core is given 7** ([factory:227-235](device/factories/untilize_multi_core_parallelize_column_program_factory.cpp)) — an extra `stick_size` (`block_size_nbytes`) inserted at index 2, shifting every subsequent arg, so the cliff core would read `num_tiles_per_core = stick_size`, `tile_width_size = ntiles_per_core_cliff`, `start_stick = tile_width_size`, `offset = 0`, and never read the real offset. Likely unhit today (this factory is selected only for single-tile-row wide-interleaved shapes via `get_pf_type()==0`, and a cliff requires an uneven `nblocks` split), which is why it survives, but it is a genuine latent bug independent of any port.
2. **Unreferenced kernels in the op directory:** `device/kernels/compute/untilize_w.cpp` (only an `experimental/quasar` factory references it) and `device/kernels/compute/untilize_metal2.cpp` (the fork, bound by other ops). Neither is dead per se, but neither is instantiated by any in-scope untilize factory; a reader scanning `device/kernels/compute/` may mistake them for untilize-native kernels.

---

## Per-DeviceOperation attribution

| DeviceOperation | Overall | Blocking gate | Clean subset | Other gates |
|---|---|---|---|---|
| `UntilizeDeviceOperation` (native) | **RED** (config-scoped) | `UntilizeMultiCoreBlockProgramFactory`: `Is able to port? = no` (`Per-node CB size`) | 7 non-block factories → **brief issued** | Device 2.0 ✓ · Features ✓ · Offset ✓ · 3rd-arg ✓ |
| `UntilizeCodegenDeviceOperation` (codegen) | **RED** (coverage gap) | No readiness-sheet row → readiness-sheet owner | none yet (whole-DOp gate) → **no brief** until sheet reconciled | Device 2.0 ✓ · Features ✓ · Offset ✓ · 3rd-arg ✓ (Class 2 drops) · code cross-check ✓ |

---

## Questions for the user

1. **Codegen sheet coverage:** `UntilizeCodegenDeviceOperation` has no readiness-sheet row (it postdates the untilize analysis). Its code cross-check is clean and every other gate is GREEN. Do you want to (a) treat the codegen op as portable and proceed once the sheet owner confirms, or (b) hold codegen until the sheet is reconciled, and port only the clean 7-factory native subset now? (The brief is written for the native subset regardless.)
2. **Bundling:** both device ops are audited in this one report with per-DeviceOperation attribution (they share a directory and one host entry, though no code). Confirm this is the accounting you want, vs. two separate reports.

## Recipe notes

- **Two independent `DeviceOperation`s in one op directory, and the sheet tracks only one.** The recipe's *Multiple device-operations in one op directory* rule covers the bundling decision, but the *factory-set-match* staleness check (`ttnn_factory.md`) is framed per single DeviceOperation ("every factory in the code has a row"). Here an **entire new DeviceOperation** (`UntilizeCodegenDeviceOperation`) is untracked, which the check treats as a "missing row → spreadsheet-broken → GATE." That routing is defensible, but it reads more alarmingly than the situation warrants — a brand-new device op the sheet simply hasn't onboarded is different in kind from a renamed/dropped factory of a *tracked* device op. Suggest the recipe distinguish "a factory of a tracked DeviceOperation lacks a row" (genuine staleness) from "an entire new DeviceOperation is untracked" (coverage gap / onboarding), so an auditor doesn't have to choose between over-blocking and improvising.
- The `descriptor`-op RED-with-clean-subset path worked cleanly here (block factory is one of eight siblings), consistent with the recipe's note that a `descriptor`-op concept-gate failure usually leaves a portable subset.
