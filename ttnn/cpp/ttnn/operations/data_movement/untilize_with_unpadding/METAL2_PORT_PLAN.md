# Port Plan — `data_movement/untilize_with_unpadding`

Port plan for `untilize_with_unpadding`, ported from `ProgramDescriptor` to Metal 2.0. It covers
**all five factories**. Written during the inventory and planning steps and committed alongside the
port for review.

> **Revision, 2026-09-23 — five of five, after merging `main`.** An earlier revision planned all five
> factories. Owners then narrowed the port to three, because untilize codegen's
> `build_native_equivalent` called `create_descriptor` on two of them. PR #56280 deleted that
> caller, so this revision is five of five again. It **supersedes** the three-of-five plan (still in
> branch history). What changed since the five-factory design was first written:
>
> - **Three forks this port used to create now already exist on `main`** (created by #56280 for
>   `data_movement/untilize`). The BlockInterleaved factory **reuses** them (rung 1) instead of
>   creating its own (rung 2). Their binding vocabulary is identical to what this factory emits. See
>   [Shared kernels](#shared-kernels).
> - **The shared-kernel sunset is in this diff.** Five legacy kernels now have no binder anywhere,
>   and the invoker decided to delete them here. The recipe does not assign that to the porter; see
>   [Edits outside the op directory](#edits-outside-the-op-directory).
> - **The `BACKWARDS` correction to a shared fork is in this diff**, also by invoker decision. Same
>   section.
> - **Two more dead compile-time args are dropped**, in the NDSharded writer. The audit brief asked
>   for this; see [Dropped Plumbing](#dropped-plumbing).
>
> The legacy factories themselves are **unchanged on `main`** since the five-factory design:
> `git diff` over the op's five factory files between the original port's base and today's merge
> base is empty. So the legacy inventory below still describes the ported-from code line for line.

Recipe docs: `port/metal2_port.md` and its companions from `akertesz/op-porting-recipe` at
`4bd4bf42bfe` (blob-identical, extracted outside the repo, not on this branch). Recipe sections
are cited by name below, not linked.

---

## Legacy Inventory

### Legacy factory shape

- **Concept:** `ProgramDescriptorFactoryConcept`. Each factory has one
  `static tt::tt_metal::ProgramDescriptor create_descriptor(const UntilizeWithUnpaddingParams&, const Tensor&, Tensor&)`
  at `device/factories/*_program_factory.hpp:14`.
- **Where the factory methods live:** in a five-alternative `program_factory_t` variant on
  `UntilizeWithUnpaddingDeviceOperation` (`device/untilize_with_unpadding_device_operation.hpp:24-29`).
  This is **not** the direct-descriptor shape, so `ttnn_factory.md` exception 3 does not apply.
- **Variants:** five factories, each with its own program shape:

  | # | Factory | File (`device/factories/`) |
  |---|---|---|
  | 1 | `UntilizeWithUnpaddingSingleCoreProgramFactory` | `…_single_core_program_factory.cpp` |
  | 2 | `UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory` | `…_multi_core_interleaved_program_factory.cpp` |
  | 3 | `UntilizeWithUnpaddingMultiCoreShardedProgramFactory` | `…_multi_core_sharded_program_factory.cpp` |
  | 4 | `UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory` | `…_multi_core_block_interleaved_program_factory.cpp` |
  | 5 | `UntilizeWithUnpaddingMultiCoreNDShardedProgramFactory` | `…_multi_core_nd_sharded_program_factory.cpp` |

- **Custom `compute_program_hash`:** none. The op uses the default reflection-based hash, and there
  is no backdoor `attribute_values` / `to_hash` either. Nothing to preserve and nothing to touch.
- **`override_runtime_arguments`:** none on any factory, so the base concept applies.
- **`get_dynamic_runtime_args`:** none.
- **Pybound `create_descriptor`:** none. `untilize_with_unpadding_nanobind.cpp` binds only the
  user-facing op, so no pybind deletion is forced and there is **no user-visible API change**.
- **External callers of the factories:** none since #56280. No code outside the op directory names
  any `UntilizeWithUnpadding*ProgramFactory`. The only mention anywhere is a documentation table
  (`models/experimental/ops/quasar/tests/qwen3_vl_ops/PROGRAM_FACTORIES.md`) that lists two of them
  by name; it calls nothing.

*(The Metal 2.0 target concept was chosen during the audit; see the brief's TTNN factory analysis.
It is carried forward in [TTNN ProgramFactory](#ttnn-programfactory) below.)*

`opt_level`: `git grep opt_level` over the five legacy factory files returns **0 hits**. So every
resolved level is the legacy per-kernel-type default: **O2** for data movement, **O3** for compute.

---

### Variant 1: SingleCore

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|
| reader | `eltwise/unary/…/dataflow/reader_unary_interleaved_start_id.cpp` | `core` (1 node) | `TensorAccessorArgs(*src0_buffer)` only (`:130`) | `{src0_buffer, num_tiles, 0}` (`:187`) | — | O2 | `ReaderConfigDescriptor{}` |
| writer | `…/dataflow/writer_unary_unpad_dims_split_rows.cpp` | `core` | `{FLOAT32_DTYPE, unpadded_stick_size}` (`:136`) + `TensorAccessorArgs(*dst_buffer)` (`:137`) | 15 values from `dst_buffer` (`:190`) | — | O2 | `WriterConfigDescriptor{}` |
| compute | `data_movement/untilize/…/compute/untilize.cpp` | `core` | `{num_tiles/num_tiles_per_block, num_tiles_per_block, c_0, c_16}` (`:161-162`) | — | `DST_ACCUM_MODE=1` for Int32/UInt32/Float32 input | O3 | `ComputeConfigDescriptor{.fp32_dest_acc_en, .unpack_to_dest_mode}` |

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|
| `c_0` | `num_tiles_per_block * input_single_tile_size` | `core` | input format | `input_single_tile_size` | unset |
| `c_16` | `num_tiles_per_block * output_single_tile_size` | `core` | output format | `output_single_tile_size` | unset |

#### Semaphores

None. **No factory in this op declares a `SemaphoreDescriptor`.**

#### Tensor accessors

| host site | originating Tensor | RTA slot |
|---|---|---|
| `:130` | input | reader RTA 0 |
| `:137` | output | writer RTA 0 |

#### Work split

Not applicable: a single core (`sub_core_grids ? first core : (0,0)`).

---

### Variant 2: MultiCoreInterleaved

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | RTAs | opt_level | config |
|---|---|---|---|---|---|---|
| reader | `eltwise/unary/…/reader_unary_interleaved_start_id.cpp` | `all_cores` | `TensorAccessorArgs(*src0_buffer)` (`:93`) | `{src0_buffer, num_tiles_per_core, tile_start_id}` per core (`:237`) | O2 | `ReaderConfigDescriptor{}` |
| writer | `…/dataflow/writer_unary_stick_layout_split_rows_multicore.cpp` | `all_cores` | `{FLOAT32_DTYPE, unpadded_row_size_bytes, writer_page_size}` (`:120-124`) + `TensorAccessorArgs(*dst_buffer)` (`:125`) | `{dst_buffer, padded_row_size_bytes, row_start_id, assignment.size()}` then **5 per `BlockRep` run**, run count varying per core (`:201-232`) | O2 | `WriterConfigDescriptor{}` |
| compute (full) | `data_movement/untilize/…/compute/untilize.cpp` | `core_range` | `{nblocks_per_core, num_tiles_per_row, c_0, c_16}` (`:160`) | — | O3 | `ComputeConfigDescriptor{…}` |
| compute (cliff) | same source | `core_range_cliff` | `{nblocks_per_core_cliff, num_tiles_per_row, c_0, c_16}` (`:175`) | — | O3 | `ComputeConfigDescriptor{…}` |

`row_start_id` is pushed **before** the inner loop that advances it past the core's own blocks
(`:204` vs `:213`). The port must capture it at the same point.

#### CBs

| index | total_size | core_ranges | data_format | page_size |
|---|---|---|---|---|
| `c_0` | `num_tiles_per_row * input_single_tile_size` | `all_cores` | input format | `input_single_tile_size` |
| `c_16` | `num_tiles_per_row * output_single_tile_size` | `all_cores` | output format | `output_single_tile_size` |

#### Tensor accessors

| host site | originating Tensor | RTA slot |
|---|---|---|
| `:93` | input | reader RTA 0 |
| `:125` | output | writer RTA 0 |

#### Work split

- Driver: `ttnn::split_blocks_for_tilize(available_grid, num_blocks)` (`:48-49`) →
  `(ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff)`,
  with `has_cliff = !core_range_cliff.empty()`.
- Per-core writer payload from `ttnn::distribute_work(...)` (`:187`) → `std::vector<BlockRep>` per
  core, run-length compressed into 5-tuples.
- Runtime-arg loop over `corerange_to_cores(available_grid)[0..ncores)` (`:193-241`).

---

### Variant 3: MultiCoreSharded

Four mutually exclusive configurations (`:204-240`):

| # | Condition | Writer source | Compute source |
|---|---|---|---|
| a | `cross_shard_type` | `…/dataflow/writer_unary_unpad_cross_sharded.cpp` | `untilize.cpp` |
| b | `out_sharded && !cross_shard_type && !unpad_tensor_w_16` | `…/dataflow/writer_unary_unpad_batch_rows_sharded.cpp` | `untilize.cpp` |
| b′ | `out_sharded && !cross_shard_type && unpad_tensor_w_16` | `…/dataflow/writer_unary_unpad_width_16_sharded.cpp` | `ttnn/kernel/compute/eltwise_copy.cpp` |
| c | `!out_sharded && HEIGHT_SHARDED` | `…/dataflow/writer_unary_unpad_sharded_to_interleaved.cpp` | `untilize.cpp` |
| d | `!out_sharded` (W/B sharded) | `ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` | `untilize.cpp` |

In every configuration the reader is `eltwise/unary/…/reader_unary_sharded.cpp` over `all_cores`
(`= shard_spec.grid`), with CTA `{src0_cb_index}` (`:191`) and RTA
`{ntiles_per_block * nblocks_per_core}`.

#### CBs

| index | total_size | core_ranges | data_format | page_size | `.buffer` |
|---|---|---|---|---|---|
| `c_0` | `ntiles_per_block * nblocks_per_core * input_single_tile_size` | `all_cores` | input | `input_single_tile_size` | **`a.buffer()`** when `src_sharded` (`:152`) |
| `c_16` | `num_output_tiles * output_single_tile_size` | `all_cores` | output | `output_single_tile_size` | — |
| `c_17` | `num_output_rows_unpadded * aligned_page_size` | `all_cores` | output | `aligned_page_size` | **`output.buffer()`** (`:181`), allocated **only** under `out_sharded && !cross_shard_type` (`:169`) |

Both `.buffer` sites leave `address_offset` at its default of zero. They are borrowed memory,
**not** the gated `address_offset` feature.

#### Tensor accessors

| host site | originating Tensor | RTA slot | configs |
|---|---|---|---|
| `:207` | output | writer RTA 0 (`:310`) | a |
| `:227` | output | writer RTA 0 (`:356`) | c |
| `:237` | output | writer RTA 0 (`:415`) | d |

Configs b and b′ construct no `TensorAccessor` at all: the writer moves L1 to L1 through `c_17`.

#### Work split

Not applicable. `all_cores = shard_spec.grid`, and every core runs the same kernel set.

---

### Variant 4: MultiCoreBlockInterleaved

#### Kernels

`make_block_plan(BlockDirection::Untilize, BlockCoreOrder::ColumnMajor, …)` (`:61-70`) yields two
`BlockBufferSet`s, `full` and `cliffrow`. The factory builds one reader and one writer **per
non-empty set**, plus up to **four** compute instances, each over its own core range and bound to
the set that matches its cores' block width.

| unique_id | source | core_ranges | CTAs (positional) | RTAs | opt_level |
|---|---|---|---|---|---|
| reader ×(1–2) | `eltwise/unary/…/reader_unary_interleaved_wh_multicore.cpp` | `set.core_ranges` | `{num_tiles_2d, third_dim, total_tiles_per_row, set.input_index}` + `TensorAccessorArgs(*src0_buffer)` (`:140-142`) | `{src0_buffer, tile_start_id, single_block_size_row_arg, single_block_size_col_arg}` (`:294-295`) | O2 |
| writer ×(1–2) | `…/dataflow/writer_unary_stick_layout_wh_multicore.cpp` | `set.core_ranges` | `{total_num_rows, third_dim, TILE_HEIGHT, unpadded_row_size_bytes, set.output_index}` + `TensorAccessorArgs(*dst_buffer)` (`:154-156`) | `{dst_buffer, width_size, start_row_id, start_column_id, single_block_size_row_arg, single_block_size_col_arg, sub_block_width_size, single_sub_block_size_row_arg}` (`:298-307`) | O2 |
| compute ×(≤4) | `data_movement/untilize/…/compute/untilize_wh.cpp` | per region, below | `{block_size_col, block_size_row, third_dim, set.input_index, set.output_index}` (`:211-212`) | — | O3 |

Compute instances (`:221-232`):

| core range | buffer set | `block_size_col` | `block_size_row` | guard |
|---|---|---|---|---|
| `core_range` | full | `single_sub_block_size_wh` | `single_sub_block_size` | `!core_range.empty()` |
| `cliff_col_row_core_range` | cliffrow | `single_block_size_cliff_col` | `single_block_size_cliff_row` | `has_cliff_col && has_cliff_row` |
| `cliff_row_core_range` | cliffrow | `single_block_size` | `single_block_size_cliff_row` | `has_cliff_row` |
| `cliff_col_core_range` | full | `single_sub_block_size_cliff_col_wh` | `single_sub_block_size` | `has_cliff_col` |

In `data_movement/common/common.cpp` `make_block_plan`:
`full.core_ranges = core_range ∪ (has_cliff_col ? cliff_col_core_range : ∅)` and
`cliffrow.core_ranges = has_cliff_row ? cliff_row_core_range ∪ (has_cliff_col ? cliff_col_row_core_range : ∅) : ∅`.
So a set's cores are exactly the union of the compute regions bound to it.
`buffer_set_for_core` asserts that every core falls in exactly one set.

#### CBs

Pushed by the **shared** helper `push_buffer_set` (`data_movement/common/common.cpp`), which takes a
`ProgramDescriptor&`. An untilize set has no `staging_index`, so the helper emits exactly two CBs
per non-empty set, and this factory passes no `tile`:

| index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|
| `full.input_index` = `c_0` | `full.block_tiles * input_single_tile_size` | `full.core_ranges` | input | `input_single_tile_size` | unset |
| `full.output_index` = `c_16` | `full.block_tiles * output_single_tile_size` | `full.core_ranges` | output | `output_single_tile_size` | unset |
| `cliffrow.input_index` = `c_2` | `cliffrow.block_tiles * input_single_tile_size` | `cliffrow.core_ranges` | input | `input_single_tile_size` | unset |
| `cliffrow.output_index` = `c_17` | `cliffrow.block_tiles * output_single_tile_size` | `cliffrow.core_ranges` | output | `output_single_tile_size` | unset |

#### Tensor accessors

| host site | originating Tensor | RTA slot |
|---|---|---|
| `:142` (per reader instance) | input | reader RTA 0 |
| `:156` (per writer instance) | output | writer RTA 0 |

#### Work split

`make_block_plan(...).split` is a `ttnn::BlockSplitWH`. The runtime-arg loop walks
`corerange_to_cores(available_grid)` (`:235-321`), and `ColumnMajor` must match that walk.
`make_block_plan` reads live L1 occupancy, so it is valid only on a program-cache miss, which is
the only time the factory runs.

---

### Variant 5: MultiCoreNDSharded

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | RTAs | CRTAs | opt_level |
|---|---|---|---|---|---|---|
| reader | `data_movement/sharded/…/dataflow/reader_unary_nd_sharded_blocks.cpp` | `compute_core_range` | `{src0_cb_index, num_tiles_per_input_block, num_shards, num_compute_cores}` (`:122`) + `TensorAccessorArgs(*src0_buffer)` (`:126`) | `{src0_buffer, start_shard_id}` (`:272`) | — | O2 |
| writer | `…/dataflow/writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp` | `compute_core_range` | 17 values (`:155-174`) + `TensorAccessorArgs(*dst_buffer)` (`:185`) + `TensorAccessorArgs(*src0_buffer)` (`:187`) | `{dst_buffer, src0_buffer, start_shard_id}` (`:275`) | output padded shape then input padded shape, 2 × rank (`:175-183`) | O2 |
| compute | `data_movement/untilize/…/compute/untilize_variable_num_blocks.cpp` | `compute_core_range` | `{num_tiles_per_input_block, src0_cb_index, output_cb_index}` (`:221`) | `{num_input_blocks_to_process}` | — | O3 |

Writer CTA slots 0–16: `output_cb_index`, `output_stick_size` (`:157`, **never read**),
`tile_height`, `num_tiles_per_input_block`, `output_num_blocks_across_width`,
`output_element_size`, `num_cols_per_input_block`, `num_cols_per_output_block`,
`input_single_tile_size` (`:164`, **never read**), `num_shards`, `num_cores`,
`num_tiles_per_input_row`, `num_tiles_per_output_row`, `tile_width`, `output_tensor_width`,
`output_tensor_height`, `tensor_rank`.

#### CBs

| index | total_size | core_ranges | data_format | page_size |
|---|---|---|---|---|
| `c_0` | `input_cb_num_tiles * input_single_tile_size` | `compute_core_range` | input | `input_single_tile_size` |
| `c_16` | `output_cb_num_tiles * output_single_tile_size` | `compute_core_range` | output | `output_single_tile_size` |

Neither CB is borrowed: the ND reader NOC-reads the input into `c_0` page by page.

#### Tensor accessors

| host site | originating Tensor | RTA slot |
|---|---|---|
| `:126` | input | reader RTA 0 |
| `:185` | output | writer RTA 0 |
| `:187` | input | writer RTA 1 (shard geometry only: `accessor_src.shard_pages(shard_id)`) |

#### Work split

No `split_work_to_cores`. The cores come from `input.buffer()->buffer_distribution_spec()`:
`ordered_cores_with_data` becomes `compute_core_range`. Each core's `start_shard_id` is its
enumeration index, and the compute RTA comes from `page_mapping.core_host_page_indices`.

---

### Shared kernels

Census: `git grep -F <filename>` over the whole tree, then disambiguated by the **bound path**.
Build files, same-named private copies, and comment mentions were discarded.

**`experimental/quasar/**` is excluded by rule.** No file there was read. A path-literal check
confirms none of its 382 `"ttnn/cpp/ttnn/…"` kernel paths points into any of the five shared
directories below. It binds its own private copies, so it is not a consumer of anything here.

| kernel the legacy factory bound | relation | `_metal2` fork | rung | legacy copy's remaining binders |
|---|---|---|---|---|
| `eltwise/unary/…/reader_unary_interleaved_start_id.cpp` | borrowed | existed | **1 — reuse** | 5 (`reduction/topk`, `…/nlp_create_qkv_heads_falcon7b`, `examples/example` ×2, `examples/example_multiple_return`) |
| `eltwise/unary/…/reader_unary_sharded.cpp` | borrowed | existed | **1 — reuse** | 4 (`experimental/slice_write` ×2, `untilize` ND-identical factory, `sharded_to_interleaved_partial`) |
| `data_movement/sharded/…/reader_unary_nd_sharded_blocks.cpp` | borrowed | existed | **1 — reuse** | **0 → deleted here** (sunset) |
| `data_movement/untilize/…/compute/untilize.cpp` | borrowed | existed | **1 — reuse** | 0 factories, **but** `test_parallel_sequential.py` reads its source text, so it is **kept** |
| `data_movement/untilize/…/compute/untilize_variable_num_blocks.cpp` | borrowed | existed | **1 — reuse** | 1 (`untilize` ND-identical factory) |
| `ttnn/kernel/compute/eltwise_copy.cpp` | borrowed | existed | **1 — reuse** | 3 (`copy`, `sharded_to_interleaved_partial`, and `interleaved_to_sharded_partial` via a different copy) |
| `eltwise/unary/…/reader_unary_interleaved_wh_multicore.cpp` | borrowed | **created by #56280** | **1 — reuse** | **0 → deleted here** (sunset) |
| `data_movement/untilize/…/compute/untilize_wh.cpp` | borrowed | **created by #56280** | **1 — reuse** | **0 → deleted here** (sunset) |
| `…/untilize_with_unpadding/…/writer_unary_stick_layout_wh_multicore.cpp` | **lent** (to `untilize`) | **created by #56280** | **1 — reuse** | **0 → deleted here** (sunset) |
| `ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` | borrowed | **created by this branch** | **2 — create** | **0 → deleted here** (sunset) |

The other seven op-owned writers are bound only by this op, so they are converted **in place**. No
fork is needed:
`writer_unary_stick_layout_split_rows_multicore.cpp`, `…_nd_sharded.cpp`,
`writer_unary_unpad_dims_split_rows.cpp`, `writer_unary_unpad_cross_sharded.cpp`,
`writer_unary_unpad_batch_rows_sharded.cpp`, `writer_unary_unpad_width_16_sharded.cpp`,
`writer_unary_unpad_sharded_to_interleaved.cpp`.

**Binding vocabulary inherited at rung 1.** This is now a constraint, not a free choice. Every fork
below also has consumers in other ops: between 1 and 7 other factories each, including `untilize`,
`fold` and `upsample`. So **any rename breaks another op**. The one fork only this op binds is the
one this branch created, `writer_unary_stick_layout_interleaved_blocks_metal2.cpp`.

| fork | `dfb::` | `tensor::` | named args | `#ifdef`s |
|---|---|---|---|---|
| `reader_unary_interleaved_start_id_metal2.cpp` | `in` (P) | `src` | RTA `num_pages`, `start_id` | — |
| `reader_unary_sharded_metal2.cpp` | `in` (P) | — | RTA `num_tiles_per_core` | — |
| `reader_unary_nd_sharded_blocks_metal2.cpp` | `in` (P) | `src` | CTA `num_tiles_per_input_block`, `num_shards`, `num_cores`; RTA `start_shard_id` | — |
| `reader_unary_interleaved_wh_multicore_metal2.cpp` | `in` (P) | `src` | CTA `num_tiles_per_2d`, `third_dim`, `total_tiles_per_row`; RTA `start_id`, `single_block_size_row_arg`, `single_block_size_col_arg` | `BACKWARDS` (defined by no consumer) |
| `writer_unary_stick_layout_wh_multicore_metal2.cpp` | `out` (C) | `dst` | CTA `total_num_rows`, `third_dim`, `tile_height`, `unpadded_X_size`; RTA `width_size`, `start_row_id`, `start_column_id`, `single_block_size_row_arg`, `single_block_size_col_arg`, `sub_block_width_size`, `single_sub_block_size_row_arg` | — |
| `untilize_metal2.cpp` | `src` (C), `out` (P) | — | CTA `per_core_block_cnt`, `per_core_block_tile_cnt` | — |
| `untilize_wh_metal2.cpp` | `src` (C), `out` (P) | — | CTA `block_size_col`, `block_size_row`, `third_dim` | — |
| `untilize_variable_num_blocks_metal2.cpp` | `src` (C), `out` (P) | — | CTA `per_core_block_tile_cnt`; RTA `per_core_block_cnt` | — |
| `eltwise_copy_metal2.cpp` | `in` (C), `out` (P) | — | CTA `per_core_tile_cnt` | — |

**Fit check for the three #56280 forks.** Each was compared against the fork this branch
originally created for the same kernel (branch history, before the narrowing revert) and against
its legacy original. The kernel bodies are behaviourally identical to what the earlier five-factory
run verified. There are only two differences:
- The reader fork's `BACKWARDS` walk. See [Edits outside the op directory](#edits-outside-the-op-directory).
- `untilize_wh_metal2.cpp` declares its three CTAs `const uint32_t`, where this branch's copy had
  used `constexpr auto`. The legacy `untilize_wh.cpp` declared them `const uint32_t`, so #56280's
  form is the faithful one.

### Flags

1. **`device/factories/untilize_with_unpadding_multi_core_shared_variables.hpp` is dead code.**
   Nothing references the header or its struct. It is listed only in `data_movement/CMakeLists.txt`.
   Out of scope; reported.
2. **Dead compile-time args.** The host emits these and the kernel never reads them:
   - NDSharded writer: `output_stick_size` and `input_single_tile_size`. **Dropped** in this
     revision, per the brief (see Dropped Plumbing).
   - Sharded, config d writer: `output_row_size`. **Dropped** earlier on this branch (`20469fc7248`).
   - SingleCore writer: `unpadded_stick_size`. **Kept**: the brief did not flag it.
   - Sharded, config b′ writer: `aligned_page_size`. **Kept**, because one CTA vector feeds both the
     b and b′ writers and only b reads it.
   - SingleCore writer RTA `num_blocks_w_input` is read into an unused local. It occupies a dispatch
     slot, so it is kept as a named RTA.
3. **No unreferenced kernel files** in `device/kernels/`. All seven remaining op-owned writers and
   the one `_metal2` fork are bound.
4. **No descriptor type outside the audit's scan.** Only `CBDescriptor` and `KernelDescriptor`
   appear. There is no `SemaphoreDescriptor`, no `WorkloadDescriptor`, and nothing GlobalCircularBuffer-shaped.
5. **No `->address()` expression anywhere in the op.** Every tensor base arrived as a `Buffer*`,
   which the descriptor framework already patched on cache hits. The conversion to typed bindings
   therefore repairs no stale-pointer hazard.

---

## TTNN ProgramFactory

- **Concept (inherited from audit):** `ProgramSpecFactoryConcept` (plain) on **all five**
  factories. No op-owned tensors, and no `override_runtime_arguments` to translate.
- **Custom `compute_program_hash`:** none; the op uses the default reflection-based hash.
- **Implementation notes:**
  - `tensor_args_t` is a bare `Tensor`, so the entry point is
    `create_program_artifacts(const UntilizeWithUnpaddingParams&, const Tensor& input, Tensor& output)`.
    Each factory extracts the `MeshTensor` once, at the top, via `.mesh_tensor()`.
  - Each `.hpp` swaps `create_descriptor` for `create_program_artifacts`, and
    `<tt-metalium/program_descriptors.hpp>` for `ttnn/metal_v2_artifacts.hpp`.
  - **Unity-build hygiene:** all five factory `.cpp`s share one unity translation unit, so every
    anonymous-namespace spec-name constant is prefixed per factory (`SC_`, `MCI_`, `SH_`, `BI_`,
    `ND_`).

---

## Planned Spec Shape

DFB spec names carry no `cb`. Accessor names are per binding; where a rung-1 fork dictates one
(`src`, `in`, `out`, `dst`), it is used verbatim.

### Variant 1: SingleCore

- **KernelSpecs (3):** `SC_READER` (`reader_unary_interleaved_start_id_metal2.cpp`), `SC_WRITER`
  (`writer_unary_unpad_dims_split_rows.cpp`, converted in place), `SC_COMPUTE` (`untilize_metal2.cpp`).
- **DataflowBufferSpecs (2):** `SC_IN`, `SC_OUT`. Sizes match the legacy CBs;
  `tile_format_metadata` is unset, as the legacy `tile` was.
- **TensorParameters (2):** `SC_INPUT`, `SC_OUTPUT`. **WorkUnitSpecs (1):** all three kernels on `core`.

### Variant 2: MultiCoreInterleaved

- **KernelSpecs (2 + up to 2):** `MCI_READER`, `MCI_WRITER`, plus `MCI_COMPUTE_FULL` (if
  `!core_range.empty()`) and `MCI_COMPUTE_CLIFF` (if `has_cliff`). Both computes come from one lambda
  (`…_multi_core_interleaved…:163-187`).
- **DataflowBufferSpecs (2):** `MCI_IN`, `MCI_OUT`, each with `num_entries = num_tiles_per_row`.
- **TensorParameters (2):** `MCI_INPUT`, `MCI_OUTPUT`.
- **WorkUnitSpecs (≤2):** `full` = `{READER, WRITER, COMPUTE_FULL}` on `core_range`;
  `cliff` = `{READER, WRITER, COMPUTE_CLIFF}` on `core_range_cliff`. The reader's and writer's node
  set is their union, which equals the legacy `all_cores`.

### Variant 3: MultiCoreSharded

- **KernelSpecs (3):** `SH_READER` (`reader_unary_sharded_metal2.cpp`), `SH_WRITER` (source chosen by
  configuration), `SH_COMPUTE` (`untilize_metal2.cpp`, or `eltwise_copy_metal2.cpp` under
  `unpad_tensor_w_16`).
- **DataflowBufferSpecs (2 or 3):** `SH_IN` (`borrowed_from = SH_INPUT` when `src_sharded`), `SH_OUT`,
  and `SH_SHARDED_OUT`. The last is **conditional** (only under `out_sharded && !cross_shard_type`)
  and has `borrowed_from = SH_OUTPUT`, `entry_size = aligned_page_size`, and
  `num_entries = num_output_rows_unpadded`.
- **TensorParameters (1–2):** `SH_OUTPUT` always. `SH_INPUT` only under `src_sharded`; it is
  borrow-only (no kernel binds it), which is legal because `borrowed_from` names it.
- **WorkUnitSpecs (1):** all three kernels on `all_cores`.

### Variant 4: MultiCoreBlockInterleaved

- **KernelSpecs (up to 8):** `BI_READER_FULL` / `BI_WRITER_FULL` (if `!full_set.empty()`),
  `BI_READER_CLIFFROW` / `BI_WRITER_CLIFFROW` (if `!cliffrow_set.empty()`), and up to four computes:
  `BI_COMPUTE_FULL`, `BI_COMPUTE_CLIFF_COL_ROW`, `BI_COMPUTE_CLIFF_ROW`, `BI_COMPUTE_CLIFF_COL`.
- **DataflowBufferSpecs (up to 4):** `BI_IN_FULL` / `BI_OUT_FULL` and `BI_IN_CLIFFROW` /
  `BI_OUT_CLIFFROW`, one pair per non-empty set, with `num_entries = set.block_tiles`. They are built
  inline (`…_block_interleaved…:120-144`) with `push_buffer_set`'s sizing rules, because that helper
  emits into a `ProgramDescriptor&`. #56280's port of `untilize`'s block factory made the same
  choice.
- **TensorParameters (2):** `BI_INPUT`, `BI_OUTPUT`.
- **WorkUnitSpecs (up to 4):** one per compute instance, each pairing it with its set's reader and writer:

  | WorkUnit | kernels | target_nodes |
  |---|---|---|
  | `bi_compute_full` | `BI_READER_FULL`, `BI_WRITER_FULL`, `BI_COMPUTE_FULL` | `core_range` |
  | `bi_compute_cliff_col_row` | `BI_READER_CLIFFROW`, `BI_WRITER_CLIFFROW`, `BI_COMPUTE_CLIFF_COL_ROW` | `cliff_col_row_core_range` |
  | `bi_compute_cliff_row` | `BI_READER_CLIFFROW`, `BI_WRITER_CLIFFROW`, `BI_COMPUTE_CLIFF_ROW` | `cliff_row_core_range` |
  | `bi_compute_cliff_col` | `BI_READER_FULL`, `BI_WRITER_FULL`, `BI_COMPUTE_CLIFF_COL` | `cliff_col_core_range` |

  Each set's reader and writer therefore cover exactly `set.core_ranges`, as in legacy.

### Variant 5: MultiCoreNDSharded

- **KernelSpecs (2 or 3):** `ND_READER` (`reader_unary_nd_sharded_blocks_metal2.cpp`), `ND_WRITER`
  (converted in place), and `ND_COMPUTE` (`untilize_variable_num_blocks_metal2.cpp`, iff `has_compute`).
- **DataflowBufferSpecs (2):** `ND_IN`, `ND_OUT`. **TensorParameters (2):** `ND_INPUT`, `ND_OUTPUT`.
- **WorkUnitSpecs (1):** on `compute_core_range`.

---

## Preserved Multiplicity

```
Variant 2 — Legacy KernelDescriptors [compute-full, compute-cliff] of untilize.cpp
  → KernelSpecs [MCI_COMPUTE_FULL, MCI_COMPUTE_CLIFF] of untilize_metal2.cpp
  → in WorkUnitSpecs [full (core_range), cliff (core_range_cliff)]
  → sharing MCI_IN (CONSUMER on each) and MCI_OUT (PRODUCER on each)
  Per-group CTA preserved: per_core_block_cnt = nblocks_per_core vs nblocks_per_core_cliff.

Variant 4 — Legacy KernelDescriptors [full_reader, cliffrow_reader] of reader_unary_interleaved_wh_multicore.cpp
  → KernelSpecs [BI_READER_FULL, BI_READER_CLIFFROW] of the #56280 _metal2 fork
  → in WorkUnitSpecs [bi_compute_full + bi_compute_cliff_col] and [bi_compute_cliff_row + bi_compute_cliff_col_row]
  → binding DIFFERENT DFBs (BI_IN_FULL vs BI_IN_CLIFFROW) as PRODUCER — no shared DFB at all.

Variant 4 — Legacy KernelDescriptors [full_writer, cliffrow_writer] of writer_unary_stick_layout_wh_multicore.cpp
  → KernelSpecs [BI_WRITER_FULL, BI_WRITER_CLIFFROW] of the #56280 _metal2 fork
  → same WorkUnitSpec pairing as the readers
  → binding BI_OUT_FULL vs BI_OUT_CLIFFROW as CONSUMER — disjoint DFBs.

Variant 4 — Legacy KernelDescriptors [4 × compute] of untilize_wh.cpp
  → KernelSpecs [BI_COMPUTE_FULL, BI_COMPUTE_CLIFF_COL_ROW, BI_COMPUTE_CLIFF_ROW, BI_COMPUTE_CLIFF_COL]
  → in WorkUnitSpecs [bi_compute_full, bi_compute_cliff_col_row, bi_compute_cliff_row, bi_compute_cliff_col]
  → BI_IN_FULL CONSUMER on {FULL, CLIFF_COL}; BI_IN_CLIFFROW CONSUMER on {CLIFF_COL_ROW, CLIFF_ROW};
    BI_OUT_FULL PRODUCER on {FULL, CLIFF_COL}; BI_OUT_CLIFFROW PRODUCER on {CLIFF_COL_ROW, CLIFF_ROW}.
  Per-group CTAs preserved: block_size_col / block_size_row per region (never demoted to RTA).
```

**Every one of these is the disjoint-node work split, not the same-grid two-toucher.** Each node
sees exactly one reader, one writer and one compute instance. `dataflow_buffer_spec.hpp`'s endpoint
invariant lets several same-kind `KernelSpec`s share an endpoint role when their node coverage does
not overlap and their binding parameters are identical; all of that holds here. Two `TT_FATAL`s
from the legacy factory keep the per-region CTAs and the buffer sizes from drifting apart:
`block_size_row == set.block_tiles` (`…_block_interleaved…:253-258`) and
`single_sub_block_size_row_arg == set.block_tiles` (`:409-415`). Both are kept.
**`allow_instance_multi_binding` appears nowhere in this port.**

---

## Dropped Plumbing

### Buffer-address RTAs → `TensorBinding`

Every one was a `Buffer*` pushed into `emplace_runtime_args` or an `RTArgList`. None used `->address()`.

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| `…_single_core…:187` reader RTA 0 | `src0_buffer` | `TensorBinding{SC_INPUT, "src"}` |
| `…_single_core…:190` writer RTA 0 | `dst_buffer` | `TensorBinding{SC_OUTPUT, "dst"}` |
| `…_multi_core_interleaved…:237` reader RTA 0 | `src0_buffer` | `TensorBinding{MCI_INPUT, "src"}` |
| `…_multi_core_interleaved…:202` writer RTA 0 | `dst_buffer` | `TensorBinding{MCI_OUTPUT, "dst"}` |
| `…_multi_core_sharded…:310` / `:356` / `:415` writer RTA 0 (configs a / c / d) | `dst_buffer` | `TensorBinding{SH_OUTPUT, "dst"}` |
| `…_multi_core_block_interleaved…:295` reader RTA 0 | `src0_buffer` | `TensorBinding{BI_INPUT, "src"}` |
| `…_multi_core_block_interleaved…:300` writer RTA 0 | `dst_buffer` | `TensorBinding{BI_OUTPUT, "dst"}` |
| `…_multi_core_nd_sharded…:272` reader RTA 0 | `src0_buffer` | `TensorBinding{ND_INPUT, "src"}` |
| `…_multi_core_nd_sharded…:275` writer RTA 0, 1 | `dst_buffer`, `src0_buffer` | `TensorBinding{ND_OUTPUT, "dst"}`, `TensorBinding{ND_INPUT, "src"}` (geometry only) |

All are **Case 1**, used through `TensorAccessor`. **There is no Case 2 anywhere**, so no kernel
needs the `get_bank_base_address` bridge.

### `TensorAccessorArgs` plumbing → binding mechanism

| host site | kernel-side chain |
|---|---|
| `…_single_core…:130`, `:137` | fork already converted; `writer_unary_unpad_dims_split_rows.cpp` `TensorAccessorArgs<2>()` |
| `…_multi_core_interleaved…:93`, `:125` | fork already converted; `writer_unary_stick_layout_split_rows_multicore.cpp` `TensorAccessorArgs<3>()` |
| `…_multi_core_sharded…:207`, `:227`, `:237` | `writer_unary_unpad_cross_sharded.cpp` `<1>`; `writer_unary_unpad_sharded_to_interleaved.cpp` `<0>`; the interleaved-blocks fork (was `<2>`) |
| `…_multi_core_block_interleaved…:142`, `:156` | both forks already converted by #56280 (were `<4>` and `<5>`) |
| `…_multi_core_nd_sharded…:126`, `:185`, `:187` | reader fork already converted; the writer's chained `TensorAccessorArgs<17>()` + `<dst_args.next_compile_time_args_offset()>()` collapses to `TensorAccessor(tensor::dst)` / `TensorAccessor(tensor::src)` |

### Page-size 3rd-argument CTAs (audit-cleared Class 2, dropped)

| kernel site | host CTA dropped |
|---|---|
| `writer_unary_stick_layout_split_rows_multicore.cpp` `TensorAccessor(dst_args, dst_addr, writer_page_size)` | `…_multi_core_interleaved…:124` (`writer_page_size`, CTA 2), together with its only computation (`:110-119`) |
| `writer_unary_unpad_cross_sharded.cpp` `TensorAccessor(dst_args, dst_addr, writer_page_size)` | `…_multi_core_sharded…:206` (`cross_writer_page_size`, CTA 0) and its computation at `:205`. That writer is left with no CTAs at all |

**No `dynamic_tensor_shape` is set anywhere.**

### Magic CB indices in CTAs → `DFBBinding`

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| `…_single_core…:161-162` compute CTA 2, 3 | `src0_cb_index`, `output_cb_index` | `dfb::src` / `dfb::out` on `SC_COMPUTE` |
| `writer_unary_unpad_dims_split_rows.cpp` | hardcoded `16` | `dfb::out` on `SC_WRITER` |
| `…_multi_core_interleaved…:160`, `:175` compute CTA 2, 3 | `c_0`, `c_16` | `dfb::src` / `dfb::out` |
| `writer_unary_stick_layout_split_rows_multicore.cpp` | hardcoded `dfb_id_out0 = 16` | `dfb::out` (the buffer-role comment is moved onto the construction) |
| `…_multi_core_sharded…:191` reader CTA 0 | `src0_cb_index` | `dfb::in` |
| `…_multi_core_sharded…:213` writer CTA 0, 1 | `output_cb_index`, `sharded_output_cb_index` | `dfb::untilize_out` / `dfb::out` (configs b, b′) |
| `writer_unary_unpad_cross_sharded.cpp` / `…_sharded_to_interleaved.cpp` / interleaved-blocks | hardcoded `16` / `c_16` | `dfb::untilize_out` / `dfb::out` / `dfb::out` |
| `…_multi_core_block_interleaved…:141`, `:155`, `:212` | `set.input_index` / `set.output_index` | per-set `DFBBinding`s: `BI_IN_*` / `BI_OUT_*` |
| `…_multi_core_nd_sharded…:122`, `:156`, `:221` | `src0_cb_index`, `output_cb_index` | `dfb::in` / `dfb::out` / `dfb::src` + `dfb::out` |

### Semaphore-ID RTAs

None: the op declares no semaphores.

### Positional CTAs → named CTAs

Every surviving CTA is named. Names for rung-1 forks are inherited, not chosen.

- `writer_unary_unpad_dims_split_rows.cpp`: `float32_dtype`, `unpadded_stick_size` (dead, kept).
- `writer_unary_stick_layout_split_rows_multicore.cpp`: `float32_dtype`, `unpadded_X_size`.
- `writer_unary_unpad_batch_rows_sharded.cpp` / `…_width_16_sharded.cpp`: `aligned_page_size` (dead in the width-16 writer, kept).
- `writer_unary_unpad_cross_sharded.cpp`, `writer_unary_unpad_sharded_to_interleaved.cpp`: **none left**.
- `writer_unary_stick_layout_interleaved_blocks_metal2.cpp`: `float32_dtype` (`output_row_size` dropped as dead, `20469fc7248`).
- Rung-1 forks: see the vocabulary table in [Shared kernels](#shared-kernels).
- `writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp`: 14 names: `tile_height`,
  `num_tiles_per_input_block`, `num_output_blocks_across_width`, `output_element_size`,
  `num_cols_per_input_block`, `num_cols_per_output_block`, `num_shards`, `num_cores`,
  `num_tiles_per_input_row`, `num_tiles_per_output_row`, `tile_width`, `output_tensor_width`,
  `output_tensor_height`, `tensor_rank`. The two unread slots (`output_stick_size`,
  `input_single_tile_size`) are **dropped, not named**. A positional slot the kernel never reads has
  no kernel-side variable to name, and the brief asked for the drop. Dropping them made the
  factory's `output_stick_size` local dead too, so that is removed as well.

---

## Applied Patterns

- **Sync-free / single-ended CB → self-loop:** `SH_SHARDED_OUT` in configs b / b′. Per node, the
  writer is the **only** toucher: it `reserve_back`s, fills via `get_write_ptr()`, and `push_back`s,
  and nothing drains it, because the buffer *is* the output shard. That makes `SH_WRITER` both
  PRODUCER and CONSUMER under one accessor name. This was re-derived from the census and agrees with
  the brief.
- **Conditional / optional resource bindings:** `SH_SHARDED_OUT` exists only under
  `out_sharded && !cross_shard_type`, and the BlockInterleaved cliffrow pair only when the split
  produced a cliff row. **No `#ifdef` is needed** in either case. Each conditional DFB is bound only
  by `KernelSpec`s that exist on the same condition, and the binding `KernelSpec` uses the same
  accessor name the unconditional one does. So no kernel build ever name-looks-up a token it does not
  bind.
- **Borrowed-memory DFBs:** `SH_IN.borrowed_from = SH_INPUT` and
  `SH_SHARDED_OUT.borrowed_from = SH_OUTPUT`. No `dfb_run_overrides` are needed.
- **Multi-variant factory:** the Sharded factory's four configurations branch inside
  `create_program_artifacts`, exactly where the legacy `create_descriptor` branched.
- **Pass DFB handles directly to LLKs / kernel-lib helpers:** the untilize compute forks pass
  `dfb::src` / `dfb::out` in both call-argument and non-type-template-parameter positions.
- **`constexpr` metadata carve-out (CB→DFB whitelist §A):** `writer_unary_unpad_width_16_sharded.cpp`
  keeps `constexpr uint32_t … = get_tile_size(dfb::out)`, because it feeds a `static_assert` and
  `NOC_MAX_BURST_SIZE` template arguments. This is Gen1-only token usage, recorded for Quasar uplift.
- **Caution: Porting a shared kernel:** nine forks reused at rung 1, one created at rung 2. No
  kernel another op binds is converted in place.
- **Caution: Avoid varargs:** two genuine vararg blocks are retained (below); everything else is named.
- **Unity-build hygiene:** per-factory prefixes on every anonymous-namespace spec name.

### Varargs — retained, with justification

1. **`writer_unary_stick_layout_split_rows_multicore.cpp` — runtime varargs.** The loop runs
   `n_block_reps` times, a runtime value, and advances `rt_arg_idx` by 5 **inside** the loop to
   pull one 5-tuple per `BlockRep` run. The run count varies **per core and per shape**. The three
   leading scalars (`padded_X_size`, `start_stick_id`, `n_block_reps`) are distinct fields read
   once, so they are **named**. `dst_addr` became the tensor binding, and `rt_arg_idx` now starts at
   0 because varargs live in their own section. Since the count differs per core, the schema uses
   `advanced_options.num_runtime_varargs_per_node`. That field is documented for removal, but it is
   the only construct that reproduces the legacy per-core RTA layout exactly.
2. **`writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp` — common runtime varargs.** Two
   loops, bounded by the CTA `tensor_rank`, read the output shape and then the input shape. The
   count is `2 * rank`, identical on every node, so it uses `num_common_runtime_varargs`.

**Not varargs:** `writer_unary_stick_layout_wh_multicore_metal2.cpp` re-reads the same named RTAs
inside a `third_dim` loop. That is a fixed set of distinct fields, so it stays named. There are no
compile-time varargs anywhere.

---

## Hardware configuration and compiler options

- **DM kernels.** Every legacy reader is a plain `ReaderConfigDescriptor{}` and every writer a plain
  `WriterConfigDescriptor{}`, i.e. the default triples. There is no custom triple and no
  `DM_DYNAMIC_NOC`, so the port uses `create_reader_datamovement_config(arch)` /
  `create_writer_datamovement_config(arch)`, which reproduce those triples byte for byte on Gen1.
- **Compute kernels — Style B.** Every factory sets a Metal `ComputeConfigDescriptor` directly; there
  is no TTNN `ComputeKernelConfig` in the op. So each factory builds `ComputeGen1Config` directly.
  Legacy sets only two fields:

  | legacy field | value in this op | `ComputeGen1Config` | action |
  |---|---|---|---|
  | `math_fidelity` | unset → `HiFi4` | default `HiFi4` | none |
  | `math_approx_mode` | unset → `false` | default `Precise` | none |
  | `bfp8_pack_precise` | unset → `false` | default `Approximate` | none |
  | `dst_full_sync_en` | unset → `false` | `double_buffer_dest` default `true` (= `!false`) | none |
  | `fp32_dest_acc_en` | `operation_attributes.fp32_dest_acc_en` | `enable_32_bit_dest` | **set** |
  | `unpack_to_dest_mode` | `UnpackToDestFp32` on the compute kernel's input CB iff `fp32_dest_acc_en` | `unpack_modes` | **set**: `{{<that kernel's input DFB>, UnpackToDest}}` iff `fp32_dest_acc_en` |

  `unpack_modes` is gated on the same flag as `enable_32_bit_dest`, so every entry is the
  "`UnpackToDest`, consumer, `enable=true` → accepted" case of the validator. `fp32_dest_acc_en` is
  itself set only for Int32/UInt32/Float32 inputs (`untilize_with_unpadding.cpp:79`). In
  BlockInterleaved, legacy marked `unpack_to_dest_mode[set.input_index]` **per compute instance**, so
  each `KernelSpec` names only its own set's input DFB.
- **`opt_level`.** Every compute `KernelSpec` gets an explicit `O3`: the one construction site per
  factory (a lambda in MultiCoreInterleaved and BlockInterleaved). No DM spec sets one, since legacy
  `O2` equals the Metal 2.0 default.
- **Gen2:** not populated, and no `if (arch == QUASAR)` branch is added.
- **`defines`:** `DST_ACCUM_MODE=1` is carried across unchanged, under the same condition, on every
  compute spec.

---

## Edits outside the op directory

The recipe sanctions exactly one kind of write outside the op directory: creating a `_metal2` fork
beside its original, plus the pointer comment in the original. This diff contains more than that,
each piece by **explicit invoker decision** (2026-09-23), recorded here so reviewers see the scope
in one place.

| edit | files | recipe status | why it is here |
|---|---|---|---|
| **Fork created (rung 2)** | `ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks_metal2.cpp` | sanctioned | The Sharded factory's config d writer. |
| **Fork install entries** | `ttnn/sources.cmake` (two lines) | necessary, not covered | `ttnn/cpp/ttnn/kernel/` is installed from an explicit list (`TTNN_CORE_JIT_API_HEADERS`), not a glob. Without an entry, installed builds ship without the fork. Two forks this op binds live there: the one created here (`writer_unary_stick_layout_interleaved_blocks_metal2.cpp`, whose entry replaces the retired original's) and the reused `eltwise_copy_metal2.cpp`. The port newly binds the latter on the Sharded W=16 path in place of the installed `eltwise_copy.cpp`, so it gets an entry beside that one. The recipe assumes a glob covers every fork directory; here none does. All 17 kernel sources the factories bind are now covered by an install rule. |
| **Sunset: delete 5 orphaned legacy kernels** | `eltwise/unary/…/reader_unary_interleaved_wh_multicore.cpp`, `…/writer_unary_stick_layout_wh_multicore.cpp` (in this op), `untilize/…/untilize_wh.cpp`, `sharded/…/reader_unary_nd_sharded_blocks.cpp`, `ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` | **deviation**: `port_patterns.md` says the sunset "is not the porter's to perform" | Invoker decision. #56280's description names this port as the sunset point. A repo-wide sweep (every file type, including tests and build files) finds no remaining reference to any of the five except the forks' own history notes. |
| **Sunset: build files** | `data_movement/CMakeLists.txt` (redundant explicit entry for the deleted writer removed); `ttnn/sources.cmake` (legacy entry **replaced** by the fork's) | deviation (follows the sunset) | A stale explicit entry is a hard CMake error. |
| **Sunset: fork header comments** | the 5 surviving forks (`…_interleaved_blocks_metal2`, `…_nd_sharded_blocks_metal2`, `untilize_wh_metal2`, `…_wh_multicore_metal2` ×2) and `untilize_metal2.cpp` | deviation: these forks have other consumers, which the recipe treats as read-only | Their "the original serves the legacy consumers" note is false once the original is gone. `untilize_metal2.cpp` keeps its original because `test_parallel_sequential.py` reads it, and its note now says so by full path. The forks keep their `_metal2` names: the recipe's full sunset would rename each fork over its original, which would touch other ops' factories. |
| **`BACKWARDS` correction** | `eltwise/unary/…/reader_unary_interleaved_wh_multicore_metal2.cpp` | **deviation**: a bug fix (the port preserves bugs) in a fork another op binds | Invoker decision. The bug is real: `dim` is `uint32_t`, so `dim > -third_dim` never enters the loop, and `-start_id` wraps. It is latent: no consumer defines `BACKWARDS`. The live (non-`BACKWARDS`) path is arithmetically unchanged. `data_movement/untilize` owners should be on the review. |

Not in the diff, and deliberately left alone: `untilize/…/compute/untilize.cpp`, which
`TestCrossOpCompilation` still reads.

---

## Deferred / Flagged

- **Dead code, reported:** `…_multi_core_shared_variables.hpp` (Flags 1); the remaining dead CTAs and
  RTA (Flags 2).
- **Structural note:** the BlockInterleaved factory reproduces `push_buffer_set`'s sizing inline,
  and so does `untilize`'s ported block factory. Two spec factories now duplicate the rule the shared
  helper exists to keep single-sourced. A spec-side twin in `data_movement/common` would restore
  that. That is shared-code work outside any port's scope.
- No feature gate fired during planning, and no construct needed a workaround.
