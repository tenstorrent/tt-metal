# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/untilize`

**Device operation (single):** `ttnn::prim::UntilizeDeviceOperation` — new TMP device-operation framework (`select_program_factory` + `program_factory_t` variant of 8 `create_descriptor` factories). All 8 factories share the device op and its kernel pool, so they are audited together as one porting unit.

- **`UntilizeDeviceOperation`** (`device/untilize_device_operation.hpp` / `.cpp`)
  - `UntilizeSingleCoreProgramFactory` (`factories/untilize_single_core_program_factory.cpp`)
  - `UntilizeMultiCoreSubCoreGridsProgramFactory` (`factories/untilize_multi_core_sub_core_grids_program_factory.cpp`)
  - `UntilizeMultiCoreParallelizeColumnProgramFactory` (`factories/untilize_multi_core_parallelize_column_program_factory.cpp`)
  - `UntilizeMultiCoreProgramFactory` (`factories/untilize_multi_core_program_factory.cpp`)
  - `UntilizeMultiCoreNDShardInputProgramFactory` (`factories/untilize_multi_core_nd_shard_input_program_factory.cpp`)
  - `UntilizeMultiCoreInputAndOutputShardTypeAndShardSpecIdenticalProgramFactory` (`factories/untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical_program_factory.cpp`)
  - `UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory` (`factories/untilize_multi_core_input_and_output_nd_shard_type_and_shard_spec_identical_program_factory.cpp`)
  - **`UntilizeMultiCoreBlockProgramFactory`** (`factories/untilize_multi_core_block_program_factory.cpp`) — **GATED** (see Result)

**Unreferenced kernel file (out of scope):** `device/kernels/compute/untilize_w.cpp` — no factory sets `kernel_source` to it (dead code). Not audited.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `53e5e16e8d0 2026-07-24 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/untilize` |
| **Overall** | **RED at op level; subset of 7 factories is clear** (Block factory gated) |
| **DOps / Factories** | `UntilizeDeviceOperation` → 8 `descriptor` factories (7 clear, 1 gated) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** (GREEN) — all 15 referenced kernels are Device 2.0 compliant |
| *Prereqs* — Cross-op escapes | Ok — 5 borrowed kernels, all Device 2.0 compliant; port-together coupling only |
| *Feature Support* — overall | **GREEN** (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | Ok (N/A) — single input tensor; all CTAs read at constexpr indices |
| *TTNN Readiness* — `Is able to port?` (the gate) | **7 = Yes; `UntilizeMultiCoreBlockProgramFactory` = `NO (confer with ops)`** |
| *TTNN Readiness* — Concept (current) | `descriptor` (all 8 factories) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes (all rows) |
| *TTNN Readiness* — Custom hash | No (all rows; confirmed — no `compute_program_hash` override) |
| *TTNN Readiness* — Runtime-args update | No (all rows; confirmed — no `get_dynamic_runtime_args` / `override_runtime_arguments`) |
| *TTNN Readiness* — Pybind `create_descriptor` | No (confirmed — nanobind binds `ttnn::untilize` only) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` (no op-owned tensors) |
| *Port work* — Offset base pointer | none (GREEN) — no `->address()` fold; addresses via `Buffer*` binding |
| *Port work* — Tensor bindings (per binding) | Case 1 (interleaved / ND-sharded paths) + clean (identical-shard backed CBs) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none — all `TensorAccessor` are 2-arg |
| *Port work* — CB endpoints | all legal 1:1 (2 CBs/factory: `c_0` in, `c_16` out); identical-shard CBs backed → `borrowed_from` |

**CB endpoints** are dispositions, not gates. Every clean-subset CB is a plain 1:1 FIFO (one locked producer + one locked consumer per node); the two identical-shard factories back their CBs with the shard buffer (borrowed-memory → `DataflowBufferSpec::borrowed_from`), still 1P+1C. No self-loops, no multi-binding, no dead CBs found.

## Result

**RED at op level; subset of 7 factories is clear.**

- **Primary blocker:** the **TTNN factory-concept gate** REDs one factory — `UntilizeMultiCoreBlockProgramFactory` has `Is able to port? = "NO (confer with ops)"` on Diego's readiness sheet. The other 7 factories are `yes`. This is a per-factory (config-scoped) gate, so **Code-path scope** applies: RED the op, and the clean subset — the **7 non-Block factories** — is portable now. A porter brief is issued for that subset (`METAL2_PORT_BRIEF.md`).
- **Routing:** route the Block factory to the **ops team** (the sheet cell literally says "confer with ops"). Note that the sheet's `NO` **diverges from its own documented derivation** — every gate conjunct for the Block row is a passing value (`Concept=descriptor`, `Custom hash=no`, `Runtime-args update=no`, `Pybind descriptor=no`, `Smuggled pointer=no`, `Is safe to port?=yes`), which the derivation formula would compose to `yes`. The `NO` is therefore a manual ops hold whose reason is not captured in the factual columns. Also surface to the **readiness-sheet owner** to confirm the override is intended (see Questions).
- The Block factory's kernels (`reader_unary_interleaved_wh_multicore.cpp`, `writer_unary_stick_layout_wh_multicore.cpp`, `untilize_wh.cpp`) are **not** used by any of the 7 clean factories, so the clean subset is independently portable regardless of how the Block hold resolves.
- Every other gate (Device 2.0, Feature compatibility, Offset base pointers, TensorAccessor 3rd arg) is **GREEN** for all factories, including the Block factory — the Block factory's only blocker is the sheet hold.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **RED (per-factory).** All 8 rows on the readiness sheet are `Concept=descriptor`, `Custom hash=no`, `Runtime-args update=no`, `Override runtime args method=no`, `Pybind descriptor=no`, `Smuggled pointer=no`, `Is safe to port?=yes`. Cross-check against code confirms every factual column: all factories expose `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` (descriptor concept); no `compute_program_hash` override anywhere in the op; no `get_dynamic_runtime_args` / `override_runtime_arguments`; `untilize_nanobind.cpp` binds the `ttnn::untilize` free function only (no `create_descriptor` pybind). The `Is able to port?` verdicts: **7 factories = `yes` (cleared)**; **`UntilizeMultiCoreBlockProgramFactory` = `NO (confer with ops)` (blocked)** → ops team + readiness-sheet owner. Gate lifts when ops resolves the hold and the sheet flips to `yes`.

- **Device 2.0 (every kernel used):** **GREEN.** Every kernel any factory instantiates is Device 2.0 compliant — modern `Noc`, `DataflowBuffer`, `TensorAccessor`, `CoreLocalMem` object idioms, `noc.async_read/write` + `dfb.reserve_back/push_back/wait_front/pop_front`, and wrapper-method `dfb.get_read_ptr()` (not the free-function holdover). The only free functions present are the **sanctioned** `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)` (Green-bullet exemptions). No `noc_async_read`, no `InterleavedAddrGen`/`ShardedAddrGen`, no raw CB-index `get_read_ptr(cb_id)`, no raw semaphore addresses anywhere. Kernels audited (own + donor + compute):

  | Kernel file | Owner | Used by | Device 2.0 |
  |---|---|---|---|
  | `.../untilize/.../compute/untilize.cpp` | untilize | single, sub_core_grids, parallelize_column, identical-shard | ✓ (kernel_lib) |
  | `.../untilize/.../compute/untilize_variable_num_blocks.cpp` | untilize | multi_core, nd_shard_input, nd-identical-shard | ✓ (kernel_lib) |
  | `.../untilize/.../compute/untilize_wh.cpp` | untilize | Block (gated) | ✓ (kernel_lib) |
  | `.../untilize/.../dataflow/reader_unary_start_id.cpp` | untilize | single, multi_core (interleaved) | ✓ |
  | `.../untilize/.../dataflow/reader_unary_sharded_blocks.cpp` | untilize | multi_core (block-reader) | ✓ |
  | `.../untilize/.../dataflow/writer_unary_stick_layout_split_rows_single_core.cpp` | untilize | single | ✓ |
  | `.../untilize/.../dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp` | untilize | multi_core | ✓ |
  | `.../untilize/.../dataflow/writer_unary_stick_layout_split_rows_multi_core_nd_shard.cpp` | untilize | nd_shard_input | ✓ |
  | `.../untilize/.../dataflow/writer_unary_stick_layout_split_rows_interleaved_parallel_columns.cpp` | untilize | parallelize_column, sub_core_grids | ✓ |
  | `eltwise/unary/.../dataflow/reader_unary_interleaved_start_id.cpp` | eltwise/unary | sub_core_grids, parallelize_column | ✓ |
  | `eltwise/unary/.../dataflow/reader_unary_sharded.cpp` | eltwise/unary | identical-shard, multi_core (even-shard) | ✓ |
  | `eltwise/unary/.../dataflow/reader_unary_interleaved_wh_multicore.cpp` | eltwise/unary | Block (gated) | ✓ |
  | `data_movement/sharded/.../dataflow/writer_unary_sharded.cpp` | data_movement/sharded | identical-shard | ✓ |
  | `data_movement/sharded/.../dataflow/reader_unary_nd_sharded_blocks.cpp` | data_movement/sharded | nd_shard_input | ✓ |
  | `data_movement/untilize_with_unpadding/.../dataflow/writer_unary_stick_layout_wh_multicore.cpp` | untilize_with_unpadding | Block (gated) | ✓ |

- **Feature compatibility:** every Appendix A entry scanned; **all N/A** (clean scan).

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer` type, no `.global_circular_buffer` field, no `remote_index`/`remote_cb` idiom, no 4-arg `CreateCircularBuffer` |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `.address_offset` set, no `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor` |
  | GlobalSemaphore | N/A | no `GlobalSemaphore` type, no `CreateGlobalSemaphore`; op uses no semaphores at all |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` carries a single `Tensor` (no `std::vector<Tensor>`); every `get_compile_time_arg_val(N)` uses a constexpr literal index — no runtime-varying CTA loop |

- **CB endpoints (GATE-free):** all CBs legal. Each clean-subset factory declares exactly **two** CBs — input `c_0` and output `c_16` — and no intermediates. Per node: `c_0` = reader (FIFO producer) + compute (FIFO consumer) → **1P+1C legal**; `c_16` = compute (FIFO producer) + writer (FIFO consumer) → **1P+1C legal**. The two identical-shard factories set `cb_src0.buffer = src0_buffer` and `cb_output.buffer = dst_buffer` (borrowed-memory backed CBs) — these port via `DataflowBufferSpec::borrowed_from` and remain 1P+1C (reader `push_back` only / writer `wait_front`+`pop_front` only). `UntilizeMultiCoreProgramFactory` uses two compute kernels (full-core + cliff-core) over **disjoint** core sets — each node sees one instance → ordinary 1:1, not multi-binding. No self-loops, no multi-binding, no dead CBs.
- **Offset base pointers:** **GREEN.** No address RTA folds a host-side offset. No `->address()` expression appears in any factory `.cpp`; addresses reach kernels via the **`Buffer*` binding form** (`emplace_runtime_args(core, {src0_buffer, ...})` / `{dst_buffer, ...}`), which the framework auto-registers as a `BufferBinding` and patches on cache hits (the readiness sheet's `Smuggled pointer=no`, `PD (pointer-patching)` classification). Clean bases only.
- **TensorAccessor 3rd argument:** **GREEN.** Every `TensorAccessor` construction in every referenced kernel is the 2-arg form `TensorAccessor(args, addr)` — no explicit page-size third argument at any site. Subject does not fire.

## Port-work summary  *(mirrors the brief — applies to the 7-factory clean subset)*

- **Tensor bindings** (per binding, per factory):
  - **Interleaved / ND-sharded input factories** (`single`, `sub_core_grids`, `parallelize_column`, `multi_core` interleaved path, `nd_shard_input`): input (`src`) and output (`dst`) → **Case 1** — the `Buffer*`-bound base is fed into `TensorAccessor(args, addr)` in the reader/writer; express as `TensorParameter`/`TensorBinding`, kernel builds `TensorAccessor(tensor::name)`.
  - **Identical-shard factories** (`InputAndOutputShardType…Identical`, `InputAndOutputNDShardType…Identical`): input and output → **clean** — CBs backed by the shard buffers (borrowed-memory DFB); port via `DataflowBufferSpec::borrowed_from`. Readers/writers take only tile counts (no `Buffer*` slot).
  - **`multi_core` even-sharded path:** input → **clean** (backed CB via `cb_backing_buffer = src0_buffer`); output → **Case 1**. (Per-config split within this factory — record via Per-DeviceOperation attribution if the port needs it.)
  - No **Case 2** (raw-pointer) bindings anywhere — every tensor access is either via `TensorAccessor` or via a borrowed-memory CB.
- **TensorParameter relaxation:** none (sheet: `none` for all rows).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** all legal 1:1. Port `c_0`/`c_16` as ordinary 1P+1C DFBs; the identical-shard factories' `c_0`/`c_16` port via `borrowed_from`.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no hidden second writer, no multi-reader, no dual-instance same-core-range split found in the clean subset.
- **Cross-op / shared kernels (port-together set):** the coupling runs **both directions** — kernels untilize borrows, and untilize's own kernel borrowed by another op. Every co-borrower listed instantiates the *same* kernel file, so its Metal 2.0 rewrite is a single change all must adopt together. (Quasar `experimental/quasar/*` copies of these ops instantiate the same file paths too — an out-of-scope Gen2 coupling to keep in view.)

  **(A) Kernels untilize borrows (outbound) — co-borrowers enumerated:**
  - `eltwise/unary/.../dataflow/reader_unary_interleaved_start_id.cpp` — used by untilize (`parallelize_column`, `sub_core_grids`). Mainline co-borrowers: `copy/typecast`, `data_movement/copy`, `data_movement/pad`, **`data_movement/untilize_with_unpadding`** (2 factories), `reduction/prod`, `experimental/transformer/nlp_create_qkv_heads_falcon7b`, `examples/example(+_multiple_return)`.
  - `eltwise/unary/.../dataflow/reader_unary_sharded.cpp` — used by untilize (identical-shard ×2, `multi_core` even-shard). Mainline co-borrowers: `copy/typecast`(sharded), `sharded/sharded_to_interleaved`, `sharded_partial/sharded_to_interleaved_partial`, `data_movement/tilize` (2), `data_movement/transpose`, **`untilize_with_unpadding`**, `experimental/slice_write` (2).
  - `data_movement/sharded/.../dataflow/writer_unary_sharded.cpp` — used by untilize (identical-shard ×2). Mainline co-borrowers: `sharded/interleaved_to_sharded`, `sharded_partial/interleaved_to_sharded_partial`, `data_movement/tilize` (2), `tilize_with_val_padding`, `data_movement/transpose`, `reduction/generic` (reduce_h), `experimental/padded_slice`, `experimental/transformer/nlp_kv_cache_load_slice`.
  - `data_movement/sharded/.../dataflow/reader_unary_nd_sharded_blocks.cpp` — used by untilize (`nd_shard_input`). Mainline co-borrower: **`untilize_with_unpadding`** (nd_sharded) only (narrow).
  - (Block factory, gated: `eltwise/unary/.../reader_unary_interleaved_wh_multicore.cpp` and `data_movement/untilize_with_unpadding/.../writer_unary_stick_layout_wh_multicore.cpp` — both co-borrowed only by `untilize_with_unpadding` (block). Note untilize *borrows the writer back* from the `untilize_with_unpadding` directory.)

  **(B) untilize's OWN kernels borrowed by other ops (inbound) — port-together in the other direction.** Porting any of these kernels to Metal 2.0 (CB→DFB, named-token bindings) breaks every listed co-borrower unless it adopts the rewrite in the same change. Confirmed by resolving each reference to its target path (mainline, non-quasar):

  | untilize-owned kernel | Also instantiated by (mainline) | Site |
  |---|---|---|
  | `device/kernels/compute/untilize.cpp` | `data_movement/fold` | `fold_multi_core_dram_program_factory.cpp:171` |
  | *(same)* | `data_movement/untilize_with_unpadding` (3 factories: interleaved, sharded, single-core) | `..._multi_core_interleaved:142`, `..._multi_core_sharded:262`, `..._single_core:176` |
  | *(same)* | `pool/upsample` | `upsample_program_factory_multicore_interleaved.cpp:198,214` |
  | `device/kernels/compute/untilize_wh.cpp` | `data_movement/untilize_with_unpadding` (block) | `..._multi_core_block_interleaved:253` |
  | `device/kernels/compute/untilize_variable_num_blocks.cpp` | `data_movement/untilize_with_unpadding` (nd_sharded) | `..._multi_core_nd_sharded:218` |
  | `device/kernels/dataflow/reader_unary_start_id.cpp` | `data_movement/tilize` (retile) | `tilize_multi_core_retile_program_factory.cpp:169` |

  `untilize.cpp` is the hub — its Metal 2.0 rewrite must be coordinated across **fold, untilize_with_unpadding (3), and upsample** simultaneously. (Not co-borrowers, despite fuzzy name matches: `padded_slice` and `sliding_window/halo` use `pack_untilize.cpp`; `deepseek_prefill/combine` uses `reader_untilize.cpp`/`writer_untilize.cpp`; `data_movement/copy` uses its own `reader_unary_start_id.cpp`.) Quasar `experimental/quasar/*` variants of untilize / untilize_with_unpadding / fold / halo reference the same-named kernels (mostly their own copies) — an out-of-scope Gen2 coupling to keep in view.

  **Planning note — the shared-kernel graph is large.** untilize and **`untilize_with_unpadding`** are effectively a co-port pair (they share nearly every kernel in *both* directions — untilize borrows uwu's block writer, uwu borrows untilize's three compute kernels). The full Metal 2.0 port-together set around untilize's kernels spans at least: **untilize, untilize_with_unpadding, fold, pool/upsample, tilize** (via untilize's own kernels) plus **typecast, transpose, pad, copy, sharded↔interleaved conversions, slice_write, padded_slice, reduction/{prod,generic}, nlp_create_qkv_heads_falcon7b, nlp_kv_cache_load_slice** (via the shared eltwise/unary + data_movement/sharded donor kernels). Sequencing this as isolated per-op ports is not viable; planners should treat the shared kernels as the unit of migration.
- **RTA varargs:** none — every kernel reads its runtime args at fixed positional indices as distinct fields; all are nameable in the port.

## Team-only

- **Out-of-directory coupling & donor shape:**
  - **Function-call escapes:**
    - `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` (**class 2 — official shared kernel library**): compute kernels call `compute_kernel_lib::untilize<...>()` and `compute_kernel_hw_startup(...)`. Lib team handles internally; no per-op action, no gate.
    - `ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp` (**class 6 — cross-family, ccl**): included by 5 own dataflow kernels. Used to supply `noc_traits` / sharded-page-iteration support for `TensorAccessor` (`accessor.shard_pages(...)`); no legacy free-function call. Device 2.0 native — `✓`, no gate.
  - **Borrowed kernel files (file-path instantiation):** the 5 (+2 gated) donor kernels above, plus the inbound `untilize.cpp`→`fold` borrow — full co-borrower enumeration in the Heads-ups "Cross-op / shared kernels (port-together set)" section. All are Device 2.0 compliant, so no Device 2.0 donor gate. `reader_unary_interleaved_start_id.cpp`, `reader_unary_sharded.cpp`, and `writer_unary_sharded.cpp` are broadly shared (≈8–10 mainline co-borrowers each); `reader_unary_nd_sharded_blocks.cpp` and the two block kernels are narrow (untilize_with_unpadding only). Roll-up: **✓ clean** (all donor shapes Device 2.0 native; no `⭐`/`✗` shapes) — coupling is port-together sequencing only, not a gate.
- **Relaxation candidates (mined from custom hash):** none — the op has no custom hash.
- **TTNN factory analysis:** current concept `descriptor` (all 8); no op-owned tensors; no MeshWorkload; no pybind `create_descriptor`; no custom hash; no custom `override_runtime_arguments`. Target concept `MetalV2FactoryConcept`.

## Misc anomalies  *(team-only, non-gating)*

- **Dead-but-hashed attribute `pf_type`** — `UntilizeOperationAttributes::pf_type` (`device/untilize_device_operation_types.hpp:31`) is set by `invoke` (`device/untilize_device_operation.cpp:416`) but never read by `select_program_factory`, which recomputes the value fresh via `get_pf_type(...)` (`untilize_device_operation.cpp:354`). The stored field is dead yet still enters the default program hash. (Value is deterministic from inputs, so no incorrect caching — just a redundant hashed field.) Route to ops team.
- **Dead-but-hashed attribute `enough_space_width`** — `UntilizeOperationAttributes::enough_space_width` (`untilize_device_operation_types.hpp:29`) is set by `invoke` but never read anywhere in the device op or factories; only `enough_space_height` is consulted (`untilize_device_operation.cpp:303`). Dead field, still hashed. Route to ops team.
- **Unreferenced kernel file `device/kernels/compute/untilize_w.cpp`** — no factory references it (dead code). Candidate for removal.

## Questions for the user

1. **Block-factory hold reason:** `UntilizeMultiCoreBlockProgramFactory` is marked `Is able to port? = "NO (confer with ops)"` on the readiness sheet, yet every gate conjunct in the sheet (and confirmed in code) is a passing value, and every audit gate (Device 2.0, features, offset, 3rd-arg) is GREEN for it too. The `NO` is a manual ops hold with no reason captured in the factual columns. What is the hold about (a known Block-factory correctness/perf concern? an in-flight change?), and is the override intended? This is the only thing blocking a full-op GREEN — the other 7 factories are clear now. (`untilize_multi_core_block_program_factory.cpp`; readiness sheet row `UntilizeMultiCoreBlockProgramFactory`.)

## Recipe notes

- **`Is able to port?` diverging from its documented derivation.** The recipe presents `Is able to port?` as `Is safe && Custom hash==no && Runtime-args update==no && Pybind==no && Concept∈{descriptor, SPMD WorkloadDescriptor}`, and says "that cell *is* the gate." For the Block row all inputs compose to `yes`, but the cell is `"NO (confer with ops)"` — a free-text manual override the formula can't express. The recipe's `no`-routing ("name which conjunct failed") assumes a conjunct-attributable failure; it has no branch for "verdict overridden to NO with no failing conjunct." I treated the cell as authoritative (per "that cell *is* the gate"), gated the Block factory, and routed to ops per the cell's own text — but a short rule for "manual-override NO that no conjunct explains" (route to ops/sheet-owner, don't try to attribute a conjunct) would remove the judgment call. The cheaply-checkable factual columns all matched code, so this is not the "spreadsheet is broken / cross-check conflict" case as written.
