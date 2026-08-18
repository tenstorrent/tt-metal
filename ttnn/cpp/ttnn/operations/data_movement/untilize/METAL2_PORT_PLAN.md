# Port Plan — untilize (native `UntilizeDeviceOperation`, 7-factory GREEN subset)

Port plan for `ttnn/cpp/ttnn/operations/data_movement/untilize`, ported from the TTNN
`ProgramDescriptor` factory API (`create_descriptor`) to Metal 2.0 (`create_program_artifacts` /
`ProgramSpecFactoryConcept`). Written during the inventory and planning steps; committed alongside
the port for review.

**Scope (from `METAL2_PORT_BRIEF.md`):** the 7 GREEN factories of `ttnn::prim::UntilizeDeviceOperation`.
**Outcome note (post-verification):** 6 of the 7 ported; `UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory`
capitulated (reverted to legacy) due to a Metal 2.0 borrowed-DFB spec-validator over-strictness on
**uneven ND sharding** — see `METAL2_PORT_REPORT.md` Handoff #2. This plan documents the original
7-factory intent; the report records the realized outcome.
**Not ported (blocked):** `UntilizeMultiCoreBlockProgramFactory` (per-node CB size, #51305) and the
whole `UntilizeCodegenDeviceOperation` (no readiness-sheet row). The device-op `program_factory_t`
variant stays mixed: the block factory keeps `create_descriptor` (ProgramDescriptorFactoryConcept),
the 7 subset factories get `create_program_artifacts` (ProgramSpecFactoryConcept). The framework
dispatches per-factory; the half-ported op builds and runs.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — each factory is a struct with a static
  `create_descriptor(...) -> tt::tt_metal::ProgramDescriptor`. Methods live in a factory struct
  (not direct-descriptor), held in `UntilizeDeviceOperation::program_factory_t` variant. **Not**
  direct-descriptor → exception 3 does not apply.
- Variants: the device-op has one `program_factory_t` variant with 8 alternatives; each is a
  separate factory (not W/H/HW-style multi-variant). Ported one-by-one.
- Custom `compute_program_hash`: none (default reflection hash). No `override_runtime_arguments`,
  no `get_dynamic_runtime_args`, no op-owned tensors, no pybound `create_descriptor`.

### Kernels (per factory)

Every factory: input DFB `c_0` (reader→compute), output DFB `c_16` (compute→writer). All DM kernels
are already Device-2.0/DFB-aware (`DataflowBuffer dfb(cb_id)`, `Noc`, `TensorAccessor`). Compute uses
`compute_kernel_lib::untilize`. The port is a binding-layer change, not an idiom rewrite.

**1. SingleCore** (interleaved only, core {0,0}):
- reader `reader_unary_start_id.cpp` — CTA[0]=cb, +TA(src0); RTA {src_addr, num_tiles, start_page_id}; binds INPUT.
- writer `writer_unary_stick_layout_split_rows_single_core.cpp` — CTA {out_cb, output_stick_size, tile_height, num_blocks_across_height, num_columns_of_blocks, num_blocks_per_column_row, num_tiles_per_block, output_single_block_width_size}, +TA(dst); RTA {dst_addr}; binds OUTPUT.
- compute `untilize.cpp` — CTA {num_blocks, num_tiles_per_block, src_cb, out_cb}; RTA none.

**2. MultiCore** (interleaved + sharded; runtime reader selection; cliff compute):
- reader (3 sources selected at runtime):
  - block-reader `reader_unary_sharded_blocks.cpp` (own) — CTA {cb, num_tiles_per_input_block, +TA(src0)}; RTA {src_addr, block_idx(i), num_blocks}; binds INPUT.
  - even-sharded `reader_unary_sharded.cpp` (eltwise/unary) — CTA {cb}; RTA {num_tiles}; c_0 borrowed_from INPUT (no TensorAccessor).
  - interleaved `reader_unary_start_id.cpp` (own) — CTA {cb, +TA(src0)}; RTA {src_addr, num_tiles, tile_start}; binds INPUT.
- writer `writer_unary_stick_layout_split_rows_multi_core.cpp` (own) — CTA {out_cb, output_stick_size(dead,idx1), tile_height, num_tiles_per_input_block, output_num_blocks_across_width, output_element_size, num_cols_per_input_block, num_cols_per_output_block, +TA(dst)}; RTA {dst, num_input_blocks_to_process, height_wise_start, num_unpadded_cols, width_wise_out_start, num_cols_already_processed}; binds OUTPUT.
- compute `untilize_variable_num_blocks.cpp` (full + cliff) — CTA {num_tiles_per_input_block, src_cb, out_cb}; RTA {num_input_blocks_to_process}.
- CB c_0: even-sharded → borrowed_from INPUT; else regular. c_16: regular.

**3. NDShardInput** (ND-sharded input):
- reader `reader_unary_nd_sharded_blocks.cpp` (data_movement/sharded) — CTA {cb, num_tiles_per_input_block, num_shards, num_cores, +TA(src0)}; RTA {src_addr, start_shard_id}; binds INPUT.
- writer `writer_unary_stick_layout_split_rows_multi_core_nd_shard.cpp` (own) — CTA {out_cb, output_stick_size(dead,idx1), tile_height, num_tiles_per_input_block, output_num_blocks_across_width, output_element_size, num_cols_per_input_block, num_cols_per_output_block, input_single_tile_size(dead,idx8), num_shards, num_cores, num_tiles_per_input_row, tile_width, output_tensor_width, output_tensor_height, +TA(dst), +TA(src0)}; RTA {dst, src0, start_shard_id}; binds OUTPUT and INPUT.
- compute `untilize_variable_num_blocks.cpp` — CTA {num_tiles_per_input_block, src_cb, out_cb}; RTA {num_input_blocks_to_process}.
- CB c_0, c_16: regular.

**4. ParallelizeColumn** (interleaved, cliff compute):
- reader `reader_unary_interleaved_start_id.cpp` (eltwise/unary) — CTA {+TA(src0)}; RTA {src_addr, ntiles, tile_start}; binds INPUT.
- writer `writer_unary_stick_layout_split_rows_interleaved_parallel_columns.cpp` (own) — CTA {stick_size(dead), +TA(dst)}; RTA full {dst, nsticks, ntiles_per_core, tile_width_size, start_stick=0, offset}; **cliff RTA has 7 (bug — extra stick_size at idx2)**; binds OUTPUT.
- compute `untilize.cpp` (full + cliff) — CTA {nblocks_per_core, ntiles_per_block, src_cb, out_cb}; RTA none.
- CB c_0, c_16: regular.

**5. SubCoreGrids** (interleaved, sub_core_grids, no cliff):
- reader `reader_unary_interleaved_start_id.cpp` — CTA {+TA(src0)}; RTA {src_addr, ntiles, tile_start}; binds INPUT.
- writer `writer_unary_stick_layout_split_rows_interleaved_parallel_columns.cpp` (own) — CTA {stick_size(dead), +TA(dst)}; RTA {dst, nsticks, ntiles_per_core, tile_width_size, start_stick=0, offset}; binds OUTPUT.
- compute `untilize.cpp` — CTA {nblocks_per_core, ntiles_per_block, src_cb, out_cb}; RTA none.
- CB c_0, c_16: regular.

**6. IdenticalShard** (input & output shard identical):
- reader `reader_unary_sharded.cpp` (eltwise/unary) — CTA {cb}; RTA {num_tiles_to_read}; c_0 borrowed_from INPUT.
- writer `writer_unary_sharded.cpp` (data_movement/sharded) — CTA {out_cb}; RTA {num_tiles_to_write}; c_16 borrowed_from OUTPUT.
- compute `untilize.cpp` — CTA {num_blocks_per_core, num_tiles_per_block, src_cb, out_cb}; RTA none.
- CB c_0 borrowed_from INPUT, c_16 borrowed_from OUTPUT. Both tensor params are borrow-only.

**7. IdenticalNDShard** (ND input & output shard identical):
- reader `reader_unary_sharded.cpp` — CTA {cb}; RTA {num_tiles_to_process}; c_0 borrowed_from INPUT.
- writer `writer_unary_sharded.cpp` — CTA {out_cb}; RTA {num_tiles_to_process}; c_16 borrowed_from OUTPUT.
- compute `untilize_variable_num_blocks.cpp` — CTA {num_tiles_per_block, src_cb, out_cb}; RTA {num_blocks_to_process}.
- CB c_0 borrowed_from INPUT, c_16 borrowed_from OUTPUT.

### Semaphores
none (no semaphores in any factory).

### Work split
- SingleCore: single core {0,0}.
- MultiCore: `split_blocks_for_tilize` → full compute range + cliff compute range (interleaved); sharded → single grid, no cliff. Reader/writer over union (compute_core_range); compute split per group.
- NDShardInput: `get_optimal_worker_cores_for_sharded_tensor` grid, single group.
- ParallelizeColumn: `split_blocks_for_tilize` → full + cliff. Reader/writer over `all_cores` (union); compute split per group.
- SubCoreGrids: `num_cores_to_corerangeset_in_subcoregrids` all_cores, single group.
- IdenticalShard/IdenticalNDShard: shard grid, single group.

### Shared kernels
| kernel | rung | note |
|---|---|---|
| compute/untilize.cpp | REUSE fork `untilize_metal2.cpp` | dfb::src, dfb::out; args per_core_block_cnt, per_core_block_tile_cnt |
| eltwise/unary reader_unary_interleaved_start_id.cpp | REUSE fork `..._metal2.cpp` | dfb::in, tensor::src; args num_pages, start_id |
| dataflow/reader_unary_start_id.cpp | CREATE fork | co-borrowers copy, tilize |
| compute/untilize_variable_num_blocks.cpp | CREATE fork | co-borrower untilize_with_unpadding |
| eltwise/unary reader_unary_sharded.cpp | CREATE fork | ~7 families |
| data_movement/sharded writer_unary_sharded.cpp | CREATE fork | ~9 families |
| data_movement/sharded reader_unary_nd_sharded_blocks.cpp | CREATE fork | co-borrower untilize_with_unpadding |
| own DM writers (single_core, multi_core, interleaved_parallel_columns, multi_core_nd_shard), reader_unary_sharded_blocks.cpp | CONVERT IN PLACE | untilize-only; no external / out-of-scope binder |

### Flags
- Unreferenced in op dir (not audited, untouched): `compute/untilize_w.cpp` (quasar-only), `compute/untilize_metal2.cpp` (the reused fork, owned by fold).
- ParallelizeColumn cliff-writer 7-vs-6 RTA latent bug (audit Misc anomaly #1) — carried, flagged, not replicated.
- Dead CTAs in writers (output_stick_size at idx1; input_single_tile_size at idx8 in nd_shard writer) — kernel never reads them; dropped on port.

## TTNN ProgramFactory
- Concept (inherited from audit): `ProgramSpecFactoryConcept` for all 7.
- Custom `compute_program_hash`: none — leave default.
- Implementation notes: device-op class untouched; each factory struct's `.hpp`/`.cpp` swaps
  `create_descriptor` → `create_program_artifacts`. `program_factory_t` variant unchanged (still lists
  all 8; block stays on the descriptor concept).

## Planned Spec Shape (per factory, default 1:1 with legacy)
- KernelSpecs: reader + writer + compute; compute gets 2 KernelSpecs where legacy had full+cliff
  (MultiCore, ParallelizeColumn) — preserved multiplicity.
- DataflowBufferSpecs: INPUT (c_0) + OUTPUT (c_16). borrowed_from set for backed CBs
  (MultiCore even-sharded c_0; IdenticalShard/IdenticalNDShard both).
- SemaphoreSpecs: none.
- TensorParameters: INPUT, OUTPUT. INPUT bound by reader (and NDShardInput writer); OUTPUT by writer.
  Borrow-only params where a CB borrows (no TensorBinding, resolved via borrowed_from).
- WorkUnitSpecs: one per compute core group (MultiCore/ParallelizeColumn: full + cliff, reader+writer
  in both; others: single).

## Preserved Multiplicity
| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (roles) |
|---|---|---|---|
| MultiCore: compute full + cliff (untilize_variable_num_blocks) | COMPUTE_FULL, COMPUTE_CLIFF | wu_full, wu_cliff | c_0 (CONSUMER), c_16 (PRODUCER) each over disjoint node set |
| ParallelizeColumn: compute full + cliff (untilize.cpp) | COMPUTE_FULL, COMPUTE_CLIFF | wu_full, wu_cliff | c_0 (CONSUMER), c_16 (PRODUCER) each over disjoint node set |
| all others | single compute | single wu | — |

## Dropped Plumbing
| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| every reader/writer CTA carrying a CB index (`src0_cb_index`, `output_cb_index`) | positional/`(uint32_t)CBIndex::c_*` | `DFBBinding` |
| every reader/writer `TensorAccessorArgs(*buf).append_to(cta)` | CTA plumbing | `TensorBinding` + kernel `TensorAccessor(tensor::name)` |
| reader RTA slot 0 = `src0_buffer` (Buffer*) | BufferBinding | `TensorBinding(INPUT)` |
| writer RTA slot 0 = `dst_buffer` (Buffer*) | BufferBinding | `TensorBinding(OUTPUT)` |
| NDShardInput writer RTA slot 1 = `src0_buffer` | BufferBinding | `TensorBinding(INPUT)` |
| writer CTA `output_stick_size` (multi_core, nd_shard idx1), `input_single_tile_size` (nd_shard idx8) | dead CTA (kernel skips) | dropped |
| ParallelizeColumn/SubCoreGrids writer CTA `stick_size` | dead CTA (kernel reads-but-unused) | kept as named CTA to stay faithful |
| positional CTAs everywhere | positional | named CTAs |

## Applied Patterns
- [Two-toucher / disjoint-node work split (Demoting per-group CTA to RTA anti-pattern avoidance)]: MultiCore & ParallelizeColumn compute full+cliff → 2 KernelSpecs over disjoint WUs, per-group CTA kept.
- [Borrowed-memory DFBs]: MultiCore even-sharded (c_0), IdenticalShard/IdenticalNDShard (c_0, c_16) via `borrowed_from`.
- [Pass DFB handles directly to LLKs]: compute kernels pass `dfb::src`/`dfb::out` into `compute_kernel_lib::untilize`.
- [Porting a shared kernel]: reuse 2 existing forks; create 5 new forks; convert 5 own untilize-only kernels in place.
- [DFB metadata via object/token]: `get_tile_size` — `const` sites → member getter; `constexpr` sites (reader_unary_sharded_blocks) → `get_tile_size(dfb::in)` token form.

## Deferred / Flagged
- ParallelizeColumn cliff-writer latent RTA bug: named-arg model drops the erroneous extra `stick_size`;
  documented, not deliberately fixed, not replicated. See report.
- No new structural blockers uncovered during planning.
