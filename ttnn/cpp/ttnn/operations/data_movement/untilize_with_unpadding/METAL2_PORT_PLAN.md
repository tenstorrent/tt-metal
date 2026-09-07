# Port Plan — `data_movement/untilize_with_unpadding`

Port plan for `untilize_with_unpadding`, ported from `ProgramDescriptor` to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

Scope: **all five factories, every configuration.** Nothing deferred.

---

## Legacy Inventory

### Legacy factory shape

- **Concept:** `ProgramDescriptorFactoryConcept` — one `static tt::tt_metal::ProgramDescriptor
  create_descriptor(const UntilizeWithUnpaddingParams&, const Tensor&, Tensor&)` per factory,
  declared at `device/factories/*_program_factory.hpp:14`.
- **Where the factory methods live:** in a `program_factory_t` variant on
  `UntilizeWithUnpaddingDeviceOperation` (`device/untilize_with_unpadding_device_operation.hpp:24-29`),
  five alternatives. **Not** the direct-descriptor shape, so `ttnn_factory.md` exception 3 does not apply.
- **Variants:** five factories, each its own program shape:

  | # | Factory | File (`device/factories/`) |
  |---|---|---|
  | 1 | `UntilizeWithUnpaddingSingleCoreProgramFactory` | `…_single_core_program_factory.cpp` |
  | 2 | `UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory` | `…_multi_core_interleaved_program_factory.cpp` |
  | 3 | `UntilizeWithUnpaddingMultiCoreShardedProgramFactory` | `…_multi_core_sharded_program_factory.cpp` |
  | 4 | `UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory` | `…_multi_core_block_interleaved_program_factory.cpp` |
  | 5 | `UntilizeWithUnpaddingMultiCoreNDShardedProgramFactory` | `…_multi_core_nd_sharded_program_factory.cpp` |

- **Custom `compute_program_hash`:** none — default reflection-based hash. No backdoor
  `attribute_values` / `to_hash` either (`device/untilize_with_unpadding_device_operation.cpp` declares
  neither). Nothing to preserve, nothing to touch.
- **`override_runtime_arguments`:** none on any factory. → base concept.
- **`get_dynamic_runtime_args`:** none.
- **Pybound `create_descriptor`:** none — `untilize_with_unpadding_nanobind.cpp` binds only the
  user-facing op. No pybind deletion is forced; **no user-visible API change**.

*(The Metal 2.0 factory concept the port targets was chosen during the audit — see the brief's TTNN
factory analysis section. Carried forward in [TTNN ProgramFactory](#ttnn-programfactory) below.)*

---

### Variant 1: SingleCore

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `eltwise/unary/…/dataflow/reader_unary_interleaved_start_id.cpp` | `core` (1 node) | `TensorAccessorArgs(*src0_buffer)` only | — | `{src0_buffer, num_tiles, 0}` | — | — | absent → **O2** | `ReaderConfigDescriptor{}` |
| writer | `…/untilize_with_unpadding/device/kernels/dataflow/writer_unary_unpad_dims_split_rows.cpp` | `core` | `{FLOAT32_DTYPE, unpadded_stick_size}` + `TensorAccessorArgs(*dst_buffer)` | — | 15 values, `{dst_buffer, output_w, padded_W_diff_blocks, output_z, padded_Z_diff_blocks, output_y, padded_Y_diff_blocks, num_leftover_Y, output_x, padded_stick_size, num_blocks_w_input, num_blocks_w_output, num_blocks_w_diff, block_row_size, block_row_leftover_size}` | — | — | absent → **O2** | `WriterConfigDescriptor{}` |
| compute | `data_movement/untilize/…/compute/untilize.cpp` | `core` | `{num_tiles/num_tiles_per_block, num_tiles_per_block, c_0, c_16}` | — | — | — | `DST_ACCUM_MODE=1` if input format is Int32/UInt32/Float32 | absent → **O3** | `ComputeConfigDescriptor{.fp32_dest_acc_en, .unpack_to_dest_mode}` |

`opt_level`: `grep -n opt_level device/` returns **0 hits across all 5 factory `.cpp` files** — no factory
sets it. Resolved levels are therefore the legacy per-kernel-type defaults, as recorded above.

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` | `num_tiles_per_block * input_single_tile_size` | `core` | `input_cb_data_format` | `input_single_tile_size` | unset |
| `c_16` | `num_tiles_per_block * output_single_tile_size` | `core` | `output_cb_data_format` | `output_single_tile_size` | unset |

No GlobalCircularBuffer anywhere in this op (all five factories). No `.buffer` set in this variant.

#### Semaphores

none — **the op declares no `SemaphoreDescriptor` in any factory.**

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `…_single_core_program_factory.cpp:130` (`TensorAccessorArgs(*src0_buffer)`) | input | reader RTA 0 (`src0_buffer`) |
| `…_single_core_program_factory.cpp:137` (`TensorAccessorArgs(*dst_buffer)`) | output | writer RTA 0 (`dst_buffer`) |

#### Work split

n/a — single core (`core` = `sub_core_grids ? corerange_to_cores(...).at(0) : CoreRange({0,0},{0,0})`).

---

### Variant 2: MultiCoreInterleaved

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | RTAs | opt_level | config |
|---|---|---|---|---|---|---|
| reader | `eltwise/unary/…/reader_unary_interleaved_start_id.cpp` | `all_cores` | `TensorAccessorArgs(*src0_buffer)` | `{src0_buffer, num_tiles_per_core, tile_start_id}` per core | O2 | `ReaderConfigDescriptor{}` |
| writer | `…/dataflow/writer_unary_stick_layout_split_rows_multicore.cpp` | `all_cores` | `{FLOAT32_DTYPE, unpadded_row_size_bytes, writer_page_size}` + `TensorAccessorArgs(*dst_buffer)` | `{dst_buffer, padded_row_size_bytes, row_start_id, assignment.size()}` + **5 per `BlockRep` run** (variable count per core) | O2 | `WriterConfigDescriptor{}` |
| compute (full) | `data_movement/untilize/…/compute/untilize.cpp` | `core_range` | `{nblocks_per_core, num_tiles_per_row, c_0, c_16}` | — | **O3** | `ComputeConfigDescriptor{…}` |
| compute (cliff) | same source | `core_range_cliff` | `{nblocks_per_core_cliff, num_tiles_per_row, c_0, c_16}` | — | **O3** | `ComputeConfigDescriptor{…}` |

#### CBs

| index | total_size | core_ranges | data_format | page_size |
|---|---|---|---|---|
| `c_0` | `num_tiles_per_row * input_single_tile_size` | `all_cores` | `input_cb_data_format` | `input_single_tile_size` |
| `c_16` | `num_tiles_per_row * output_single_tile_size` | `all_cores` | `output_cb_data_format` | `output_single_tile_size` |

#### Tensor accessors

| host site | originating Tensor | RTA slot |
|---|---|---|
| `…_multi_core_interleaved_program_factory.cpp:93` | input | reader RTA 0 |
| `…_multi_core_interleaved_program_factory.cpp:125` | output | writer RTA 0 |

#### Work split

- Driver: `ttnn::split_blocks_for_tilize(available_grid, num_blocks)` (`:48-49`).
- `(ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff)`;
  `has_cliff = !core_range_cliff.empty()`.
- Per-core writer args come from `ttnn::distribute_work(...)` (`:187`) → `std::vector<BlockRep>` per core.

---

### Variant 3: MultiCoreSharded

Four mutually-exclusive configurations, selected at `:204-240`:

| # | Condition | Writer source | Compute source |
|---|---|---|---|
| a | `cross_shard_type` | `…/dataflow/writer_unary_unpad_cross_sharded.cpp` | `untilize.cpp` |
| b | `out_sharded && !cross_shard_type && !unpad_tensor_w_16` | `…/dataflow/writer_unary_unpad_batch_rows_sharded.cpp` | `untilize.cpp` |
| b′ | `out_sharded && !cross_shard_type && unpad_tensor_w_16` | `…/dataflow/writer_unary_unpad_width_16_sharded.cpp` | `ttnn/kernel/compute/eltwise_copy.cpp` |
| c | `!out_sharded && HEIGHT_SHARDED` | `…/dataflow/writer_unary_unpad_sharded_to_interleaved.cpp` | `untilize.cpp` |
| d | `!out_sharded` (W/B sharded) | `ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` | `untilize.cpp` |

Reader is `eltwise/unary/…/reader_unary_sharded.cpp` in every configuration, over `all_cores`
(`= shard_spec.grid`), CTA `{src0_cb_index}`, RTA `{ntiles_per_block * nblocks_per_core}` broadcast
per core.

#### CBs

| index | total_size | core_ranges | data_format | page_size | `.buffer` |
|---|---|---|---|---|---|
| `c_0` | `ntiles_per_block * nblocks_per_core * input_single_tile_size` | `all_cores` | `input_cb_data_format` | `input_single_tile_size` | **`a.buffer()`** when `src_sharded` (`:152`) |
| `c_16` | `num_output_tiles * output_single_tile_size` | `all_cores` | `output_cb_data_format` | `output_single_tile_size` | — |
| `c_17` | `num_output_rows_unpadded * aligned_page_size` | `all_cores` | `output_cb_data_format` | `aligned_page_size` | **`output.buffer()`** (`:181`) — allocated **only** under `out_sharded && !cross_shard_type` (`:169`) |

Both `.buffer` sites carry `address_offset` at its default zero: borrowed memory, **not** the gated
`address_offset` feature.

#### Tensor accessors

| host site | originating Tensor | RTA slot | configs |
|---|---|---|---|
| `…_multi_core_sharded_program_factory.cpp:207` | output | writer RTA 0 | a |
| `…:227` | output | writer RTA 0 | c |
| `…:237` | output | writer RTA 0 | d |

Configs b / b′ construct no `TensorAccessor` at all — the writer moves L1→L1 through `c_17`.

#### Work split

n/a — `all_cores = shard_spec.grid`; every core runs the same kernel set. Per-core writer args in
configs a / c / d; broadcast args in b / b′.

---

### Variant 4: MultiCoreBlockInterleaved

#### Kernels

Two `BlockBufferSet`s (`full`, `cliffrow`) from
`ttnn::operations::data_movement::make_block_plan(BlockDirection::Untilize, BlockCoreOrder::ColumnMajor, …)`
(`:61-70`). One reader + one writer **per non-empty set**; up to **four** compute instances, each over
its own core range and bound to the set matching its cores' block width.

| unique_id | source | core_ranges | CTAs (positional) | RTAs | opt_level |
|---|---|---|---|---|---|
| reader ×2 (per set) | `eltwise/unary/…/reader_unary_interleaved_wh_multicore.cpp` | `set.core_ranges` | `{num_tiles_2d, third_dim, total_tiles_per_row, set.input_index}` + `TensorAccessorArgs(*src0_buffer)` | `{src0_buffer, tile_start_id, single_block_size_row_arg, single_block_size_col_arg}` | O2 |
| writer ×2 (per set) | `…/dataflow/writer_unary_stick_layout_wh_multicore.cpp` | `set.core_ranges` | `{total_num_rows, third_dim, TILE_HEIGHT, unpadded_row_size_bytes, set.output_index}` + `TensorAccessorArgs(*dst_buffer)` | `{dst_buffer, width_size, start_row_id, start_column_id, single_block_size_row_arg, single_block_size_col_arg, sub_block_width_size, single_sub_block_size_row_arg}` | O2 |
| compute ×≤4 | `data_movement/untilize/…/compute/untilize_wh.cpp` | see below | `{block_size_col, block_size_row, third_dim, set.input_index, set.output_index}` | — | **O3** |

Compute instances (`:221-232`), each paired with a buffer set:

| core range | buffer set | `block_size_col` | `block_size_row` | guard |
|---|---|---|---|---|
| `core_range` | full | `single_sub_block_size_wh` | `single_sub_block_size` | `!core_range.empty()` |
| `cliff_col_row_core_range` | cliffrow | `single_block_size_cliff_col` | `single_block_size_cliff_row` | `has_cliff_col && has_cliff_row` |
| `cliff_row_core_range` | cliffrow | `single_block_size` | `single_block_size_cliff_row` | `has_cliff_row` |
| `cliff_col_core_range` | full | `single_sub_block_size_cliff_col_wh` | `single_sub_block_size` | `has_cliff_col` |

`full.core_ranges = core_range ∪ cliff_col_core_range`;
`cliffrow.core_ranges = cliff_row_core_range ∪ cliff_col_row_core_range`
(`data_movement/common/common.cpp:906-923`), so each compute instance's cores are a subset of its
set's cores, and the two sets are disjoint (`buffer_set_for_core`, `:928`).

#### CBs

Pushed by the **shared** helper `push_buffer_set` (`data_movement/common/common.cpp:795-850`), which
takes a `ProgramDescriptor&`. Two CBs per non-empty set (untilize sets carry no `staging_index`):

| index | total_size | core_ranges | data_format | page_size |
|---|---|---|---|---|
| `full.input_index` = `c_0` | `full.block_tiles * input_single_tile_size` | `full.core_ranges` | `input_cb_data_format` | `input_single_tile_size` |
| `full.output_index` = `c_16` | `full.block_tiles * output_single_tile_size` | `full.core_ranges` | `output_cb_data_format` | `output_single_tile_size` |
| `cliffrow.input_index` = `c_2` | `cliffrow.block_tiles * input_single_tile_size` | `cliffrow.core_ranges` | `input_cb_data_format` | `input_single_tile_size` |
| `cliffrow.output_index` = `c_17` | `cliffrow.block_tiles * output_single_tile_size` | `cliffrow.core_ranges` | `output_cb_data_format` | `output_single_tile_size` |

`push_buffer_set`'s optional `tile` argument is **not** passed by this factory → `tile` is `nullopt`.

#### Tensor accessors

| host site | originating Tensor | RTA slot |
|---|---|---|
| `…_multi_core_block_interleaved_program_factory.cpp:142` (per reader instance) | input | reader RTA 0 |
| `…:156` (per writer instance) | output | writer RTA 0 |

#### Work split

`make_block_plan(...).split` → `ttnn::BlockSplitWH`: `(ncores, all_cores, core_range,
cliff_row_core_range, cliff_col_core_range, cliff_col_row_core_range, nblocks_per_core,
single_block_size, single_block_size_cliff_row, single_block_size_cliff_col, has_cliff_row,
has_cliff_col, full_cores_per_row, full_cores_per_col, single_sub_block_size)`.
Runtime-arg loop walks `corerange_to_cores(available_grid)` (`ColumnMajor` order — must not change).

---

### Variant 5: MultiCoreNDSharded

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | RTAs | CRTAs | opt_level |
|---|---|---|---|---|---|---|
| reader | `data_movement/sharded/…/dataflow/reader_unary_nd_sharded_blocks.cpp` | `compute_core_range` | `{src0_cb_index, num_tiles_per_input_block, num_shards, num_compute_cores}` + `TensorAccessorArgs(*src0_buffer)` | `{src0_buffer, start_shard_id}` | — | O2 |
| writer | `…/dataflow/writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp` | `compute_core_range` | 17 values (see below) + `TensorAccessorArgs(*dst_buffer)` + `TensorAccessorArgs(*src0_buffer)` | `{dst_buffer, src0_buffer, start_shard_id}` | `output.padded_shape()` dims then `input.padded_shape()` dims (2 × rank) | O2 |
| compute | `data_movement/untilize/…/compute/untilize_variable_num_blocks.cpp` | `compute_core_range` | `{num_tiles_per_input_block, src0_cb_index, output_cb_index}` | `{num_input_blocks_to_process}` | — | **O3** |

Writer CTA slots 0-16: `output_cb_index`, `output_stick_size`, `tile_height`,
`num_tiles_per_input_block`, `output_num_blocks_across_width`, `output_element_size`,
`num_cols_per_input_block`, `num_cols_per_output_block`, `input_single_tile_size`, `num_shards`,
`num_cores`, `num_tiles_per_input_row`, `num_tiles_per_output_row`, `tile_width`,
`output_tensor_width`, `output_tensor_height`, `tensor_rank`.

#### CBs

| index | total_size | core_ranges | data_format | page_size |
|---|---|---|---|---|
| `c_0` | `input_cb_num_tiles * input_single_tile_size` | `compute_core_range` | `input_cb_data_format` | `input_single_tile_size` |
| `c_16` | `output_cb_num_tiles * output_single_tile_size` | `compute_core_range` | `output_cb_data_format` | `output_single_tile_size` |

Neither is borrowed — the ND reader NOC-reads the input into `c_0` page by page.

#### Tensor accessors

| host site | originating Tensor | RTA slot |
|---|---|---|
| `…_multi_core_nd_sharded_program_factory.cpp:126` | input | reader RTA 0 |
| `…:185` | output | writer RTA 0 |
| `…:187` | input | writer RTA 1 (geometry only — `accessor_src.shard_pages(shard_id)`) |

#### Work split

n/a in the `split_work_to_cores` sense. Cores come from
`input.buffer()->buffer_distribution_spec()`: `ordered_cores_with_data` →
`compute_core_range`; per-core `start_shard_id` is the enumeration index, and the compute RTA
`num_input_blocks_to_process` is derived from `page_mapping.core_host_page_indices`.

---

### Shared kernels

Census run with `grep -rl <filename> ttnn/cpp/ttnn/operations/`, hits disambiguated by checking the
*bound path* (build files, comment mentions, and same-named private copies discarded).

**`experimental/quasar/**` is excluded by rule.** Verified by directory listing only (no file read)
that `experimental/quasar/untilize_with_unpadding/device/kernels/dataflow/` holds its **own private
copies** of these kernel filenames — so the quasar tree binds its copies, not this op's files, and is
not a consumer of anything here.

| kernel | relation | `_metal2` fork? | rung |
|---|---|---|---|
| `eltwise/unary/…/reader_unary_interleaved_start_id.cpp` | borrowed | **exists** | **1 — reuse** |
| `eltwise/unary/…/reader_unary_sharded.cpp` | borrowed | **exists** | **1 — reuse** |
| `data_movement/sharded/…/reader_unary_nd_sharded_blocks.cpp` | borrowed | **exists** | **1 — reuse** |
| `data_movement/untilize/…/compute/untilize.cpp` | borrowed | **exists** | **1 — reuse** |
| `data_movement/untilize/…/compute/untilize_variable_num_blocks.cpp` | borrowed | **exists** | **1 — reuse** |
| `ttnn/kernel/compute/eltwise_copy.cpp` | borrowed | **exists** | **1 — reuse** |
| `eltwise/unary/…/reader_unary_interleaved_wh_multicore.cpp` | borrowed | none | **2 — create** |
| `data_movement/untilize/…/compute/untilize_wh.cpp` | borrowed | none | **2 — create** |
| `ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` | borrowed | none | **2 — create** |
| `…/untilize_with_unpadding/…/writer_unary_stick_layout_wh_multicore.cpp` | **LENT** (see Flags) | none | **2 — create** |

Binding vocabulary inherited at rung 1 (this is now a **constraint**, not a free choice):

| fork | `dfb::` | `tensor::` | named args |
|---|---|---|---|
| `reader_unary_interleaved_start_id_metal2.cpp` | `in` (PRODUCER) | `src` | RTA `num_pages`, `start_id` |
| `reader_unary_sharded_metal2.cpp` | `in` (PRODUCER) | — | RTA `num_tiles_per_core` |
| `reader_unary_nd_sharded_blocks_metal2.cpp` | `in` (PRODUCER) | `src` | CTA `num_tiles_per_input_block`, `num_shards`, `num_cores`; RTA `start_shard_id` |
| `untilize_metal2.cpp` | `src` (CONSUMER), `out` (PRODUCER) | — | CTA `per_core_block_cnt`, `per_core_block_tile_cnt` |
| `untilize_variable_num_blocks_metal2.cpp` | `src` (CONSUMER), `out` (PRODUCER) | — | CTA `per_core_block_tile_cnt`; RTA `per_core_block_cnt` |
| `eltwise_copy_metal2.cpp` | `in` (CONSUMER), `out` (PRODUCER) | — | CTA `per_core_tile_cnt` |

None of the six gates on an `#ifdef`, so no `defines` are forced on this side.

### Flags

1. **The brief's shared-kernel table is incomplete — `writer_unary_stick_layout_wh_multicore.cpp` is
   *lent*.** The brief lists it among the "8 op-owned writers … none is dead code" and does not flag
   it as shared, but `data_movement/untilize`'s block factory binds it by full path
   (`untilize/device/factories/untilize_multi_core_block_program_factory.cpp:150-152`). Converting it
   in place would break that op. Treated as a rung-2 shared kernel: fork beside the original (which
   happens to be inside this op's directory), pointer comment in the original. Recorded in the port
   report under Friction.
2. **`device/factories/untilize_with_unpadding_multi_core_shared_variables.hpp` is dead code.** It
   defines `UntilizeWithUnpaddingMultiCoreSharedVariables` (reader/writer `KernelHandle`s, a core
   vector, an `ncores`) — a leftover from the pre-`ProgramDescriptor` `ProgramFactoryConcept` era.
   Nothing in the tree references the struct; the header is only listed in
   `data_movement/CMakeLists.txt:333`. Left untouched (out of the port's scope); reported.
3. **Dead compile-time args** — emitted by the host, never read by the kernel. Carried across as
   named CTAs (a named CTA the kernel never reads costs nothing: it lowers to an unused
   `constexpr experimental::CtaVal<uint32_t>` in the generated header) so the port stays a syntax
   swap. Reported, not fixed:
   - `writer_unary_unpad_dims_split_rows.cpp`: CTA 1 `unpadded_stick_size` (host `…_single_core…:136`).
   - `writer_unary_stick_layout_interleaved_blocks.cpp`: CTA 1 `output_row_size` (host
     `…_multi_core_sharded…:236`).
   - `writer_unary_unpad_width_16_sharded.cpp`: CTA 2 `aligned_page_size` (host `…:213`) — the same
     `writer_ct_args` vector feeds both the b and b′ writers, and only b reads it.
   - `writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp`: CTA 1 `output_stick_size` and
     CTA 8 `input_single_tile_size` (host `…_multi_core_nd_sharded…:157` and `:164`).
   - `writer_unary_unpad_dims_split_rows.cpp:29`: RTA 10 `num_blocks_w_input` is read into a local
     that the kernel body never uses. Kept as a named RTA (it occupies a dispatch slot today).
4. **No unreferenced kernel files** in `device/kernels/`: all 8 op-owned writers are bound.
5. **No descriptor type outside the audit's scan.** The five factories use only `CBDescriptor` and
   `KernelDescriptor`; no `SemaphoreDescriptor`, no `WorkloadDescriptor`, no
   `GlobalCircularBuffer` / `remote_cb_config` / `.global_circular_buffer` anywhere.
6. **Not a single `->address()` expression in the op.** Every tensor base arrives as a `Buffer*`
   pushed through `emplace_runtime_args`, so the descriptor framework already registers these as
   `BufferBinding`s. This port converts them to the typed channel; it does not repair a stale-pointer
   hazard.

---

## TTNN ProgramFactory

- **Concept (inherited from audit):** `ProgramSpecFactoryConcept` — plain, for **all five** factories.
  No op-owned tensors, no `override_runtime_arguments` to translate.
- **Custom `compute_program_hash`:** none — default reflection-based hash. Nothing to preserve.
- **Implementation notes:**
  - `tensor_args_t` is a bare `Tensor` (not a struct), so the entry point is
    `create_program_artifacts(const UntilizeWithUnpaddingParams&, const Tensor& input, Tensor& output)`.
    The `MeshTensor` is extracted once at the top of each factory via `.mesh_tensor()`.
  - Each factory's `.hpp` swaps `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` for
    `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(...)`. `<tt-metalium/program_descriptors.hpp>`
    is replaced by `ttnn/metal_v2_artifacts.hpp` in the headers.
  - **Unity-build hygiene** ([catalog](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols)):
    all five factory `.cpp`s live in one CMake target, so every anonymous-namespace spec-name constant
    is prefixed per factory (`SC_`, `MCI_`, `SH_`, `BI_`, `ND_`) rather than sharing a bare `READER` /
    `IN` / `OUT`.

---

## Planned Spec Shape

DFB spec names carry no `cb` (post-port the op has no CBs). Accessor names are per-binding; where a
rung-1 fork dictates one (`src`, `in`, `out`), that name is used verbatim.

### Variant 1: SingleCore

- **KernelSpecs (3):** `SC_READER` (`reader_unary_interleaved_start_id_metal2.cpp`),
  `SC_WRITER` (`writer_unary_unpad_dims_split_rows.cpp`, converted in place — not shared),
  `SC_COMPUTE` (`untilize_metal2.cpp`).
- **DataflowBufferSpecs (2):** `SC_IN` (`"in"`, `entry_size = input_single_tile_size`,
  `num_entries = num_input_tiles`), `SC_OUT` (`"out"`, `entry_size = output_single_tile_size`,
  `num_entries = num_output_tiles`). Both carry `data_format_metadata`; `tile_format_metadata` stays
  unset (legacy `CBFormatDescriptor::tile` unset).
- **SemaphoreSpecs:** none.
- **TensorParameters (2):** `SC_INPUT` (input), `SC_OUTPUT` (output).
- **WorkUnitSpecs (1):** `{SC_READER, SC_WRITER, SC_COMPUTE}` on `core`.

### Variant 2: MultiCoreInterleaved

- **KernelSpecs (2 + up to 2):** `MCI_READER`, `MCI_WRITER`, plus `MCI_COMPUTE_FULL` (when
  `!core_range.empty()`) and `MCI_COMPUTE_CLIFF` (when `has_cliff`).
- **DataflowBufferSpecs (2):** `MCI_IN` (`num_entries = num_tiles_per_row`), `MCI_OUT` (same count).
- **TensorParameters (2):** `MCI_INPUT`, `MCI_OUTPUT`.
- **WorkUnitSpecs (up to 2):** `{MCI_READER, MCI_WRITER, MCI_COMPUTE_FULL}` on `core_range`;
  `{MCI_READER, MCI_WRITER, MCI_COMPUTE_CLIFF}` on `core_range_cliff`.

### Variant 3: MultiCoreSharded

- **KernelSpecs (3):** `SH_READER` (`reader_unary_sharded_metal2.cpp`), `SH_WRITER` (source selected
  by configuration), `SH_COMPUTE` (`untilize_metal2.cpp`, or `eltwise_copy_metal2.cpp` when
  `unpad_tensor_w_16`).
- **DataflowBufferSpecs (2 or 3):**
  - `SH_IN` (`"in"`) — `borrowed_from = SH_INPUT` when `src_sharded`.
  - `SH_OUT` (`"out"`) — always.
  - `SH_SHARDED_OUT` (`"sharded_out"`) — **conditional**, built only under
    `out_sharded && !cross_shard_type`; `borrowed_from = SH_OUTPUT`,
    `entry_size = aligned_page_size`, `num_entries = num_output_rows_unpadded`.
- **TensorParameters (2):** `SH_INPUT`, `SH_OUTPUT` — both always declared. `SH_INPUT` is
  **borrow-only** (no kernel binds it); that is legal, since a parameter named by a
  `borrowed_from` counts as used.
- **WorkUnitSpecs (1):** `{SH_READER, SH_WRITER, SH_COMPUTE}` on `all_cores`.

### Variant 4: MultiCoreBlockInterleaved

- **KernelSpecs (up to 8):** `BI_READER_FULL` / `BI_WRITER_FULL` (if `!full_set.empty()`),
  `BI_READER_CLIFFROW` / `BI_WRITER_CLIFFROW` (if `!cliffrow_set.empty()`), and up to four computes
  `BI_COMPUTE_FULL`, `BI_COMPUTE_CLIFF_COL_ROW`, `BI_COMPUTE_CLIFF_ROW`, `BI_COMPUTE_CLIFF_COL`.
- **DataflowBufferSpecs (up to 4):** `BI_IN_FULL` / `BI_OUT_FULL` (`num_entries = full.block_tiles`)
  and `BI_IN_CLIFFROW` / `BI_OUT_CLIFFROW` (`num_entries = cliffrow.block_tiles`), each pair built
  only for a non-empty set. Sizes copied from `push_buffer_set`, which this factory can no longer
  call (it takes a `ProgramDescriptor&`); the two DFBs are built inline in the factory instead.
- **TensorParameters (2):** `BI_INPUT`, `BI_OUTPUT`.
- **WorkUnitSpecs (up to 4):** one per compute instance, pairing it with its set's reader and writer:

  | WorkUnit | kernels | target_nodes |
  |---|---|---|
  | `wu_full` | `BI_READER_FULL`, `BI_WRITER_FULL`, `BI_COMPUTE_FULL` | `core_range` |
  | `wu_cliff_col_row` | `BI_READER_CLIFFROW`, `BI_WRITER_CLIFFROW`, `BI_COMPUTE_CLIFF_COL_ROW` | `cliff_col_row_core_range` |
  | `wu_cliff_row` | `BI_READER_CLIFFROW`, `BI_WRITER_CLIFFROW`, `BI_COMPUTE_CLIFF_ROW` | `cliff_row_core_range` |
  | `wu_cliff_col` | `BI_READER_FULL`, `BI_WRITER_FULL`, `BI_COMPUTE_CLIFF_COL` | `cliff_col_core_range` |

### Variant 5: MultiCoreNDSharded

- **KernelSpecs (2 or 3):** `ND_READER` (`reader_unary_nd_sharded_blocks_metal2.cpp`), `ND_WRITER`,
  `ND_COMPUTE` (`untilize_variable_num_blocks_metal2.cpp`, present iff `has_compute`).
- **DataflowBufferSpecs (2):** `ND_IN` (`num_entries = input_cb_num_tiles`),
  `ND_OUT` (`num_entries = output_cb_num_tiles`).
- **TensorParameters (2):** `ND_INPUT`, `ND_OUTPUT`.
- **WorkUnitSpecs (1):** `{ND_READER, ND_WRITER}` ∪ `{ND_COMPUTE}` on `compute_core_range`.

---

## Preserved Multiplicity

```
Variant 2 — Legacy KernelDescriptors [compute-full, compute-cliff] of untilize.cpp
  → KernelSpecs [MCI_COMPUTE_FULL, MCI_COMPUTE_CLIFF] of untilize_metal2.cpp
  → in WorkUnitSpecs [wu_full (core_range), wu_cliff (core_range_cliff)]
  → sharing MCI_IN (CONSUMER on each) and MCI_OUT (PRODUCER on each)
  Per-group CTA preserved: per_core_block_cnt = nblocks_per_core vs nblocks_per_core_cliff.

Variant 4 — Legacy KernelDescriptors [full_reader, cliffrow_reader] of reader_unary_interleaved_wh_multicore.cpp
  → KernelSpecs [BI_READER_FULL, BI_READER_CLIFFROW] of the _metal2 fork
  → in WorkUnitSpecs [wu_full + wu_cliff_col] and [wu_cliff_row + wu_cliff_col_row]
  → binding DIFFERENT DFBs (BI_IN_FULL vs BI_IN_CLIFFROW) as PRODUCER — not a shared DFB at all.

Variant 4 — Legacy KernelDescriptors [full_writer, cliffrow_writer] of writer_unary_stick_layout_wh_multicore.cpp
  → KernelSpecs [BI_WRITER_FULL, BI_WRITER_CLIFFROW] of the new _metal2 fork
  → same WorkUnitSpec pairing as the readers
  → binding BI_OUT_FULL vs BI_OUT_CLIFFROW as CONSUMER — again disjoint DFBs.

Variant 4 — Legacy KernelDescriptors [4 × compute] of untilize_wh.cpp
  → KernelSpecs [BI_COMPUTE_FULL, BI_COMPUTE_CLIFF_COL_ROW, BI_COMPUTE_CLIFF_ROW, BI_COMPUTE_CLIFF_COL]
  → in WorkUnitSpecs [wu_full, wu_cliff_col_row, wu_cliff_row, wu_cliff_col]
  → BI_IN_FULL CONSUMER on {FULL, CLIFF_COL}; BI_IN_CLIFFROW CONSUMER on {CLIFF_COL_ROW, CLIFF_ROW};
    BI_OUT_FULL PRODUCER on {FULL, CLIFF_COL}; BI_OUT_CLIFFROW PRODUCER on {CLIFF_COL_ROW, CLIFF_ROW}.
  Per-group CTAs preserved: block_size_col / block_size_row per region (never demoted to RTA).
```

**Every one of these is the disjoint-node work split**, not the same-grid two-toucher shape: each node
sees exactly one reader, one writer and one compute instance. Per
[`dataflow_buffer_spec.hpp`](../../../../../../tt_metal/api/tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp)'s
endpoint invariant, several `KernelSpec`s may share one endpoint role when their node coverage is
non-overlapping, they are the same kernel kind, and their binding-site parameters are identical — all
three hold here. **No `allow_instance_multi_binding` anywhere in this port.**

---

## Dropped Plumbing

### Buffer-address RTAs → `TensorBinding`

Every one is a `Buffer*` pushed into `emplace_runtime_args` / an `RTArgList` (never `->address()`).

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `…_single_core…:187` reader RTA 0 | `src0_buffer` | `TensorBinding{SC_INPUT, "src"}` |
| `…_single_core…:190` writer RTA 0 | `dst_buffer` | `TensorBinding{SC_OUTPUT, "dst"}` |
| `…_multi_core_interleaved…:237` reader RTA 0 | `src0_buffer` | `TensorBinding{MCI_INPUT, "src"}` |
| `…_multi_core_interleaved…:202` writer RTA 0 | `dst_buffer` | `TensorBinding{MCI_OUTPUT, "dst"}` |
| `…_multi_core_sharded…:310` writer RTA 0 (config a) | `dst_buffer` | `TensorBinding{SH_OUTPUT, "dst"}` |
| `…_multi_core_sharded…:356` writer RTA 0 (config c) | `dst_buffer` | `TensorBinding{SH_OUTPUT, "dst"}` |
| `…_multi_core_sharded…:415` writer RTA 0 (config d) | `dst_buffer` | `TensorBinding{SH_OUTPUT, "dst"}` |
| `…_multi_core_block_interleaved…:295` reader RTA 0 | `src0_buffer` | `TensorBinding{BI_INPUT, "src"}` |
| `…_multi_core_block_interleaved…:300` writer RTA 0 | `dst_buffer` | `TensorBinding{BI_OUTPUT, "dst"}` |
| `…_multi_core_nd_sharded…:272` reader RTA 0 | `src0_buffer` | `TensorBinding{ND_INPUT, "src"}` |
| `…_multi_core_nd_sharded…:275` writer RTA 0 | `dst_buffer` | `TensorBinding{ND_OUTPUT, "dst"}` |
| `…_multi_core_nd_sharded…:275` writer RTA 1 | `src0_buffer` | `TensorBinding{ND_INPUT, "src"}` (geometry-only accessor) |

All are **Case 1** (used through `TensorAccessor`). **No Case 2 anywhere** → the
`get_bank_base_address` bridge is never reached, and no compute kernel needs a raw L1 base address.

### `TensorAccessorArgs` plumbing → binding mechanism

| host site | kernel-side chain |
|---|---|
| `…_single_core…:130`, `:137` | `reader_unary_interleaved_start_id.cpp:20` `TensorAccessorArgs<0>()`; `writer_unary_unpad_dims_split_rows.cpp:38` `TensorAccessorArgs<2>()` |
| `…_multi_core_interleaved…:93`, `:125` | fork already converted; `writer_unary_stick_layout_split_rows_multicore.cpp:32` `TensorAccessorArgs<3>()` |
| `…_multi_core_sharded…:207`, `:227`, `:237` | `writer_unary_unpad_cross_sharded.cpp:32` `<1>`; `writer_unary_unpad_sharded_to_interleaved.cpp:38` `<0>`; `writer_unary_stick_layout_interleaved_blocks.cpp:62` `<2>` |
| `…_multi_core_block_interleaved…:142`, `:156` | `reader_unary_interleaved_wh_multicore.cpp:23` `<4>`; `writer_unary_stick_layout_wh_multicore.cpp:22` `<5>` |
| `…_multi_core_nd_sharded…:126`, `:185`, `:187` | fork already converted; `writer_…_nd_sharded.cpp:40` `TensorAccessorArgs<17>()` **and** `:42` `TensorAccessorArgs<dst_args.next_compile_time_args_offset()>()` — the chained pair collapses to two independent `TensorAccessor(tensor::dst)` / `TensorAccessor(tensor::src)` constructions |

### Page-size 3rd-argument CTAs (audit-cleared, Class 2 — redundant/inert)

| kernel site | host CTA dropped |
|---|---|
| `writer_unary_stick_layout_split_rows_multicore.cpp:36` `TensorAccessor(dst_args, dst_addr, writer_page_size)` | `…_multi_core_interleaved…:124` (`writer_page_size`, CTA 2). Its only consumer, so the whole computation at `:110-119` (incl. the `out_mem_config` local and the `dst_buffer->aligned_page_size()` call) becomes dead and is removed. Kernel `TensorAccessorArgs<3>` → the binding. |
| `writer_unary_unpad_cross_sharded.cpp:35` `TensorAccessor(dst_args, dst_addr, writer_page_size)` | `…_multi_core_sharded…:206` (`cross_writer_page_size`, CTA 0) and its computation at `:205`. Kernel `TensorAccessorArgs<1>` → the binding; this writer is then left with **no** compile-time args at all. |

Both are Class 2 per the audit — the value each computes by hand equals the `aligned_page_size` the
binding supplies. **No `dynamic_tensor_shape` is set anywhere in this port.**

### Magic CB indices in CTAs → `DFBBinding`

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| `…_single_core…:161-162` compute CTA 2, 3 | `src0_cb_index`, `output_cb_index` | `dfb::src` / `dfb::out` bindings on `SC_COMPUTE` |
| `…_single_core…:16` (kernel) | hardcoded `dfb_id_out0 = 16` | `dfb::out` binding on `SC_WRITER` |
| `…_multi_core_interleaved…:160`, `:175` compute CTA 2, 3 | `tt::CBIndex::c_0`, `c_16` | `dfb::src` / `dfb::out` |
| `writer_unary_stick_layout_split_rows_multicore.cpp:16` | hardcoded `dfb_id_out0 = 16` | `dfb::out` |
| `…_multi_core_sharded…:191` reader CTA 0 | `src0_cb_index` | `dfb::in` on `SH_READER` |
| `…_multi_core_sharded…:213` writer CTA 0, 1 | `output_cb_index`, `sharded_output_cb_index` | `dfb::untilize_out` / `dfb::out` on `SH_WRITER` (configs b, b′) |
| `…_multi_core_sharded…:250-251` compute CTA 2, 3 | `src0_cb_index`, `output_cb_index` | `dfb::src` / `dfb::out` (`dfb::in` / `dfb::out` on the `eltwise_copy` path) |
| `writer_unary_unpad_cross_sharded.cpp:33` | hardcoded `dfb_id_untilize_out = 16` | `dfb::untilize_out` |
| `writer_unary_unpad_sharded_to_interleaved.cpp:39` | hardcoded `cb_id_out0 = tt::CBIndex::c_16` | `dfb::out` |
| `writer_unary_stick_layout_interleaved_blocks.cpp:65` | hardcoded `cb_id_out0 = tt::CBIndex::c_16` | `dfb::out` (in the fork) |
| `…_multi_core_block_interleaved…:141`, `:155`, `:212` | `set.input_index` / `set.output_index` CTAs | `dfb::in` / `dfb::out` bindings, per set |
| `…_multi_core_nd_sharded…:122` reader CTA 0 | `src0_cb_index` | `dfb::in` |
| `…_multi_core_nd_sharded…:156` writer CTA 0 | `output_cb_index` | `dfb::out` |
| `…_multi_core_nd_sharded…:221` compute CTA 1, 2 | `src0_cb_index`, `output_cb_index` | `dfb::src` / `dfb::out` |

### Semaphore-ID RTAs

none — the op declares no semaphores.

### Positional CTAs → named CTAs

Every surviving CTA is named. Names, per kernel (rung-1 forks' names are inherited, not chosen):

- `writer_unary_unpad_dims_split_rows.cpp`: `float32_dtype`, `unpadded_stick_size` (dead — kept).
- `writer_unary_stick_layout_split_rows_multicore.cpp`: `float32_dtype`, `unpadded_X_size`.
- `writer_unary_unpad_batch_rows_sharded.cpp` / `writer_unary_unpad_width_16_sharded.cpp`:
  `aligned_page_size` (dead in the width-16 writer — kept).
- `writer_unary_unpad_cross_sharded.cpp`: **none left**.
- `writer_unary_unpad_sharded_to_interleaved.cpp`: **none left**.
- `writer_unary_stick_layout_interleaved_blocks_metal2.cpp`: `float32_dtype`, `output_row_size` (dead — kept).
- `reader_unary_interleaved_wh_multicore_metal2.cpp`: `num_tiles_per_2d`, `third_dim`, `total_tiles_per_row`.
- `writer_unary_stick_layout_wh_multicore_metal2.cpp`: `total_num_rows`, `third_dim`, `tile_height`, `unpadded_X_size`.
- `untilize_wh_metal2.cpp`: `block_size_col`, `block_size_row`, `third_dim`.
- `writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp`: `output_stick_size` (dead — kept),
  `tile_height`, `num_tiles_per_input_block`, `num_output_blocks_across_width`, `output_element_size`,
  `num_cols_per_input_block`, `num_cols_per_output_block`, `input_single_tile_size` (dead — kept),
  `num_shards`, `num_cores`, `num_tiles_per_input_row`, `num_tiles_per_output_row`, `tile_width`,
  `output_tensor_width`, `output_tensor_height`, `tensor_rank`.

---

## Applied Patterns

- **[Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb):**
  `SH_SHARDED_OUT` in the Sharded factory, configs b / b′. Census on a node: the writer is the
  **only** kernel that touches `c_17` — it `reserve_back`s, fills via `get_write_ptr()`, and
  `push_back`s; nothing drains it, because the buffer *is* the output shard. One toucher → self-loop:
  `SH_WRITER` bound both PRODUCER and CONSUMER, one accessor name. Legal on Gen1 for a DM kernel;
  kernel code untouched. (Re-derived from the census; agrees with the brief.)
- **[Conditional / optional resource bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-resource-bindings):**
  `SH_SHARDED_OUT` exists only under `out_sharded && !cross_shard_type`. **No `#ifdef` is needed**:
  the host already selects a *different kernel source* per configuration
  (`…_multi_core_sharded…:204-240`), and only the b / b′ writers reference `dfb::out`. Each writer
  source is compiled against exactly the bindings its own `KernelSpec` declares, so no kernel ever
  name-looks-up a token its build does not bind. The conditional lives entirely on the host, where the
  legacy branch already is.
- **[Borrowed-memory DFBs](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/migration_guide.md#dataflowbufferspec):**
  `SH_IN.borrowed_from = SH_INPUT` (legacy `.buffer = a.buffer()`), and
  `SH_SHARDED_OUT.borrowed_from = SH_OUTPUT` (legacy `.buffer = output.buffer()`). No
  `dfb_run_overrides` entry is needed — the backing L1 address resolves from `tensor_args`.
- **[Multi-variant factory](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-multi-variant-factories):**
  the Sharded factory's four configurations branch inside `create_program_artifacts`, as the legacy
  `create_descriptor` already did.
- **[Pass DFB handles directly to LLKs](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers):**
  `compute_kernel_hw_startup(dfb::src, dfb::out)` and `compute_kernel_lib::untilize<…, dfb::src, dfb::out, …>`
  in the `untilize_wh` fork — the token flows into both call-argument and non-type-template-parameter
  positions via the `constexpr operator uint32_t()`.
- **[CB→DFB whitelist §A `constexpr` carve-out](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/cb_dfb_api_whitelist.md#tile--format-metadata-jit-descriptors):**
  `writer_unary_unpad_width_16_sharded.cpp:22` declares `constexpr uint32_t tile_size_in_bytes = get_tile_size(cb_id_out);`
  and feeds it to a `static_assert` and to `NOC_MAX_BURST_SIZE` template arguments → keeps the free-function
  form with the token: `get_tile_size(dfb::out)`. By contrast
  `reader_unary_interleaved_wh_multicore.cpp:27` declares `const uint32_t tile_bytes` (not `constexpr`)
  → moves onto the object: `dfb.get_tile_size()`.
- **[Caution: Porting a shared kernel](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel):**
  6 forks reused at rung 1; **4 forks created at rung 2**, each beside its original with a pointer
  comment added to the original.
- **[Caution: Avoid varargs](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-avoid-varargs-unless-absolutely-necessary):**
  two genuine vararg blocks retained (below); everything else named.

### Varargs — retained, with justification

1. **`writer_unary_stick_layout_split_rows_multicore.cpp:75-88` — runtime varargs.** The loop runs
   `n_block_reps` times (a *runtime* value, RTA 3) and advances `rt_arg_idx` by 5 **inside** the loop
   (`:84`), pulling a 5-tuple per `BlockRep` run. The host emits five values per run
   (`…_multi_core_interleaved…:218-232`), and the run count varies **per core and per shape** — an
   indexed-collection element with a runtime-bounded count. The four leading args are distinct fields
   read once at constant indices before the loop (`:19-22`) and are **named**: `dst_addr` becomes the
   tensor binding; `padded_X_size`, `start_stick_id`, `n_block_reps` become named RTAs. Kernel-side
   `rt_arg_idx` starts at **0** instead of 4 (varargs live in their own section).
   *Because the vararg count differs per core, the schema needs
   `advanced_options.num_runtime_varargs_per_node` — the per-node override, which is `[[deprecated]]`.
   It is used deliberately: it is the only construct that reproduces the legacy per-core RTA layout
   exactly. Reported.*
2. **`writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp:100-105` — common runtime
   varargs.** Two loops read the output shape then the input shape via `get_common_arg_val`, bounded
   by the CTA `tensor_rank` (`:39`). A CTA-bounded count still varies across instantiations →
   vararg. Host side `…_multi_core_nd_sharded…:175-183`; count is `2 * rank`, identical on every node
   → `num_common_runtime_varargs`, no per-node override.

**Not converted to varargs** (distinct fields read a fixed number of times, at constant indices):
`writer_unary_stick_layout_wh_multicore.cpp:66-71` re-reads RTA indices 2-7 inside a `third_dim` loop
but at **constant** indices → named. Every other kernel in the op reads each RTA once at a constant
index. **No compile-time varargs anywhere** — no kernel calls `get_compile_time_arg_val` at a varying
index.

---

## Hardware configuration and compiler options

- **DM kernels.** Every reader is `ReaderConfigDescriptor{}` and every writer is
  `WriterConfigDescriptor{}` — the resolved triples are exactly the reader
  (`RISCV_1`, `NOC_0`, `DM_DEDICATED_NOC`) and writer (`RISCV_0`, `NOC_1`, `DM_DEDICATED_NOC`)
  defaults. No custom triple, no `DM_DYNAMIC_NOC` anywhere → the arch-agnostic TTNN helpers
  `create_reader_datamovement_config(device->arch())` / `create_writer_datamovement_config(device->arch())`
  reproduce them byte-for-byte on Gen1 and supply the Gen2 branch for free.
- **Compute kernels — Style B.** Every factory sets a Metal `ComputeConfigDescriptor` directly; there
  is no TTNN `ComputeKernelConfig` in the op (`fp32_dest_acc_en` is a plain bool on
  `UntilizeWithUnpaddingParams`). So `ComputeGen1Config` is built directly, **not** through
  `to_compute_hardware_config` (whose defaults lean the other way). Field-by-field:

  | legacy `ComputeConfigDescriptor` | value in this op | `ComputeGen1Config` | action |
  |---|---|---|---|
  | `math_fidelity` | unset → `HiFi4` | `fpu_math_fidelity` default `HiFi4` | none |
  | `math_approx_mode` | unset → `false` | `sfpu_precision_mode` default `Precise` | none |
  | `bfp8_pack_precise` | unset → `false` | `bfp_pack_precision_mode` default `Approximate` | none |
  | `dst_full_sync_en` | unset → `false` | `double_buffer_dest` = `!false` = `true` (default) | none |
  | `fp32_dest_acc_en` | `operation_attributes.fp32_dest_acc_en` | `enable_32_bit_dest` | **set explicitly** |
  | `unpack_to_dest_mode` | `UnpackToDestFp32` on the input CB iff `fp32_dest_acc_en`, `Default` elsewhere | `unpack_modes` | **set explicitly** |

  `unpack_modes` reindexes from CB id to DFB name and translates value: the single non-`Default`
  entry is on the compute kernel's **input** DFB → `{{<that kernel's input DFB>, UnpackMode::UnpackToDest}}`
  when `fp32_dest_acc_en`, otherwise an empty table (`Default` → `UnpackToSrc`, expressed by omission).
  Because `enable_32_bit_dest` and the entry are gated on the *same* flag, the
  `UnpackToDest + enable_32_bit_dest=true` case is the always-accepted one
  (`program_spec.cpp:1078-1080`), and the Float32-consumer required-entry rule is satisfied wherever
  it fires. In the BlockInterleaved factory the legacy marks `unpack_to_dest_mode[set.input_index]`
  **per compute instance** (`:201-205`) — each `KernelSpec` therefore names only *its own* set's input
  DFB, never both.
- **`opt_level`.** No factory sets one, so: **every compute `KernelSpec` gets an explicit
  `compiler_options = {.opt_level = KernelBuildOptLevel::O3}`** (legacy `ComputeConfigDescriptor`
  resolves to `O3`; Metal 2.0 defaults to `O2`). DM `KernelSpec`s are left alone — legacy `O2` already
  matches the Metal 2.0 default. Compute specs needing the line: `SC_COMPUTE`, `MCI_COMPUTE_FULL`,
  `MCI_COMPUTE_CLIFF`, `SH_COMPUTE`, all four `BI_COMPUTE_*`, `ND_COMPUTE` — **10 in total**.
- **Gen2:** not populated anywhere; no `if (arch == QUASAR)` branch is added.
- **`defines`:** `DST_ACCUM_MODE=1` carried across unchanged, as a
  `KernelSpec::compiler_options.defines` entry on the compute spec, under the same condition.

---

## Deferred / Flagged

- **New finding (audit gap):** `writer_unary_stick_layout_wh_multicore.cpp` is a **lent** shared
  kernel that the brief did not flag — see Flags 1. Handled by forking (rung 2); no factory is
  deferred because of it.
- **New finding:** five dead compile-time args and one dead runtime arg — see Flags 3. Carried across
  verbatim; reported, not fixed.
- **New finding:** `untilize_with_unpadding_multi_core_shared_variables.hpp` is unreferenced dead
  code — see Flags 2. Left in place.
- **Structural note (not a blocker):** the BlockInterleaved factory can no longer call the shared
  `push_buffer_set` helper, which is `ProgramDescriptor`-typed and lives outside this op's writeable
  surface. The two DFBs per set are built inline in the factory, copying the helper's sizing rules
  exactly (`entry_size` = single tile size, `num_entries` = `set.block_tiles`). The helper stays
  untouched for its remaining `ProgramDescriptor` callers (tilize, tilize_with_val_padding, untilize).
  Reported so a future Metal 2.0 sibling port can decide whether the helper should grow a spec-side
  twin rather than each factory re-deriving it.
- **Construction addendum (recorded after the fact, for the reviewer):** two sweeps the
  [anti-pattern self-audit](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#anti-pattern-self-audit)'s
  `cb`-name check forced, both inside the five factory bodies:
  - `input_cb_data_format` / `output_cb_data_format` → `input_dfb_data_format` / `output_dfb_data_format`
    (the documented `cb_*` → `dfb_*` API rename; these locals name the format of what is now a DFB).
  - `#include "ttnn/operations/cb_utils.hpp"` dropped from all five factories — it provides only the
    legacy `create_cb` / `calculate_total_cb_size` helpers, none of which a spec factory can call.
  Three stale `CB` comments remain in `device/untilize_with_unpadding_device_operation.cpp` and one in
  `untilize_with_unpadding.cpp`; both files are off-limits to the port, so they are reported rather
  than edited.
- Nothing else. No feature gate fired during planning; no construct required a workaround.
