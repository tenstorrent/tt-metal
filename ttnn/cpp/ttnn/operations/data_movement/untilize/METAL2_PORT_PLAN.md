# Port Plan — `untilize` (Phase 1: shard-spec-identical + single-core)

## Scope and phasing

`UntilizeDeviceOperation` has 8 factories. The framework adapter dispatches each variant arm independently against its concept, so mixing `ProgramDescriptorFactoryConcept` and `ProgramSpecFactoryConcept` factories is supported.

This port proceeds in phases. After each phase, the user builds and tests so issues don't propagate across many files.

| Phase | Factories | Kernel forks needed | Status |
|---|---|---|---|
| 1a | `UntilizeMultiCoreInputAndOutputShardTypeAndShardSpecIdenticalProgramFactory` (sharded), `UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory` (nd-sharded) | `untilize_metal2.cpp` (new — includes early-return for nd-sharded idle cores); reuses `reader_unary_sharded_metal2.cpp` (s2i) and `writer_unary_sharded_metal2.cpp` (tilize) | **DONE — builds green** |
| 1b | `UntilizeSingleCoreProgramFactory` | `reader_unary_start_id_metal2.cpp` (new), `writer_unary_stick_layout_split_rows_single_core_metal2.cpp` (new); reuses `untilize_metal2.cpp` from 1a | **DONE — builds green** |
| 2 | `UntilizeMultiCoreSubCoreGridsProgramFactory`, `UntilizeMultiCoreParallelizeColumnProgramFactory`, `UntilizeMultiCoreNDShardInputProgramFactory` | `reader_unary_interleaved_start_id_metal2.cpp` (new, eltwise/unary), `writer_unary_stick_layout_split_rows_interleaved_parallel_columns_metal2.cpp` (new), `reader_unary_nd_sharded_blocks_metal2.cpp` (new), `writer_unary_stick_layout_split_rows_multi_core_nd_shard_metal2.cpp` (new) | **DONE — pending build/test** |
| 3a | `UntilizeMultiCoreProgramFactory` (multi-path: block-reader / sharded-backed / interleaved) | `reader_unary_sharded_blocks_metal2.cpp` (new), `writer_unary_stick_layout_split_rows_multi_core_metal2.cpp` (new); reuses other readers from prior phases | **DONE — pending build/test** |
| 3b | `UntilizeMultiCoreBlockProgramFactory` | n/a — DEFERRED, see below | **DEFERRED** |

## Phase 1 design notes

### Unified `untilize_metal2.cpp` compute kernel

Legacy `untilize.cpp` and `untilize_variable_num_blocks.cpp` differed in two ways:
- `untilize.cpp` had `per_core_block_cnt` as CTA[0]; `untilize_variable_num_blocks.cpp` moved it to RTA[0].
- `untilize_variable_num_blocks.cpp` early-returned when `per_core_block_cnt == 0` (idle nd-shard cores).

The Metal 2.0 fork unifies both: `per_core_block_cnt` is always an RTA (per the DFB-invariant pattern from tilize), and the early-return is always present. This lets all untilize factories use a single compute kernel fork — both regular and nd-sharded variants are served by `untilize_metal2.cpp`.

### `KernelSpec::CompilerOptions::Defines` for `DST_ACCUM_MODE`

Legacy used `KernelDescriptor::Defines` to set `DST_ACCUM_MODE=1` for INT32 / UINT32 / FLOAT32 inputs. Metal 2.0 has this on `KernelSpec::compiler_options.defines`. Same payload, different field location. The defines vector is `std::vector<std::pair<std::string, std::string>>`.

### Dead-CTA-slot dropped in single-core writer

`writer_unary_stick_layout_split_rows_single_core.cpp` had `output_stick_size` at CTA[1] but never read it in the kernel. Dropped in the metal2 fork (same dead-slot pattern as the tilize+s2i ports).

## Phase 2 / 3 work-up (preserved for continuation)

### Reused kernels (cumulative)

These metal2 kernel forks are already in tree and can be reused across all remaining untilize phases:

- `reader_unary_sharded_metal2.cpp` — sharded reader (PRODUCER on borrowed-memory DFB)
- `writer_unary_sharded_metal2.cpp` — sharded writer (CONSUMER on borrowed-memory DFB)
- `untilize_metal2.cpp` — compute (RTA-driven block count, idle-core safe)

### Anticipated Phase 2/3 forks

Likely need:
- `reader_unary_sharded_blocks_metal2.cpp` (sharded reader with block-iteration)
- `writer_unary_stick_layout_split_rows_multi_core_metal2.cpp`
- `writer_unary_stick_layout_split_rows_multi_core_nd_shard_metal2.cpp`
- `writer_unary_stick_layout_split_rows_interleaved_parallel_columns_metal2.cpp`
- Possibly `untilize_w_metal2.cpp` / `untilize_wh_metal2.cpp` if any factory uses them

### Phase 3b — `UntilizeMultiCoreBlockProgramFactory` — DEFERRED

Confirmed the same structural blocker as `tilize_multi_core_block`. The factory builds 4 distinct compute kernels (one per core-range type: full, cliff_col, cliff_row, cliff_col_row), each with different CTAs `{block_size_col, block_size_row, third_dim}`. The first arg is a **template parameter** of `compute_kernel_lib::untilize<...>` — a constexpr CTA, not a runtime value. The RTA-pattern (which worked for `per_core_block_cnt` in untilize_metal2) doesn't extend to template args.

Four compute KernelSpecs all consuming from one SRC_DFB would violate `dataflow_buffer_spec.hpp:46`'s "one producer / one consumer" invariant.

**Same two unblock paths as for tilize_multi_core_block:**
1. Add a runtime-block-width variant to `compute_kernel_lib::untilize` (then a single compute KernelSpec serves all core ranges, RTA-driven).
2. Relax the DFB invariant for disjoint-node WU consumers.

The framework adapter (`mesh_device_operation_adapter.hpp:204`) dispatches each `program_factory_t` variant arm independently, so leaving `UntilizeMultiCoreBlockProgramFactory` on `ProgramDescriptorFactoryConcept` while the other 7 are on `ProgramSpecFactoryConcept` is a valid intermediate state.

## Doc-evolution suggestions

- **Compute kernel unification opportunity**: `untilize.cpp` vs `untilize_variable_num_blocks.cpp` were two near-identical kernels distinguished only by which args were compile-time vs runtime. Metal 2.0's RTA-based pattern naturally unifies them. Worth mentioning in the migration guide that "split kernels driven by CT-vs-runtime arg distinctions can collapse to one in Metal 2.0" as a pattern.
- The DST_ACCUM_MODE define is set conditionally based on input dtype — useful for the migration guide's defines section as a real-world example.
