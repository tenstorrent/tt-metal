# Port Plan — tilize (`TilizeMultiCoreBlockProgramFactory`)

Port plan for `data_movement/tilize`'s **block** factory, ported from the legacy
`ProgramDescriptor` (descriptor) concept to Metal 2.0 `CustomProgramSpecFactoryConcept`.
Written during the inventory and planning steps; committed alongside the port for review.

Scope: the **one** remaining factory — `TilizeMultiCoreBlockProgramFactory`. The other five tilize
factories are already ported (`CustomProgramSpecFactoryConcept`, PR #54805) and are out of scope.

Working template: the inverse op's block factory on the same `BlockBufferSet` model —
`data_movement/untilize/device/factories/untilize_multi_core_block_program_factory.cpp` (#56280) —
and the five ported tilize siblings (custom-concept `override` shape).

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `create_descriptor` returns `ProgramDescriptor`
  (`tilize_multi_core_block_program_factory.hpp:17`). Methods live on the factory struct
  `TilizeMultiCoreBlockProgramFactory`, which is a `program_factory_t` variant of `TilizeDeviceOperation`
  (not a direct-descriptor op — no exception-3 restructure needed).
- Also defines `override_runtime_arguments` (void return) at `...block_program_factory.cpp:380` — this
  selects the target concept.
- Custom `compute_program_hash`: none (default reflection-based hash).

*(Target Metal 2.0 concept — `CustomProgramSpecFactoryConcept` — chosen in the audit; carried forward
in [TTNN ProgramFactory](#ttnn-programfactory) below.)*

### Kernels
All three are already Device-2.0 / DFB-based (verified in the audit); the forks are binding-layer swaps,
not idiom rewrites. Compile-time-arg slots read from the host factory's emission order (authoritative).

| unique_id | source | core_ranges | CTAs (positional) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|
| reader (×2, per set) | `data_movement/tilize_with_val_padding/.../reader_unary_pad_multicore_both_dims.cpp` (**borrowed**) | `set.core_ranges` | 0 total_num_rows, 1 third_dim, 2 tile_height, 3 element_size, 4 unpadded_X_size(=row_size_bytes), 5 dram_alignment, 6 dfb_id_in0(=input_index), 7 dfb_id_in1(=staging_index), 8+ `TensorAccessorArgs(*src0_buffer)` | 0 src_addr(=`src0_buffer`), 1 pad_value(0), 2 width_size, 3 start_row_id, 4 start_column_id, 5 single_block_size_row_arg, 6 single_block_size_col_arg, 7 sub_block_width_size, 8 single_sub_block_size_row_arg | first reader carries `dm_kernel_metadata` (`{num_pairs, reader0, writer0, [reader1, writer1]}`) — **host-side only, the kernel reads no common args** | none | `ReaderConfigDescriptor{}` |
| writer (×2, per set) | `eltwise/unary/.../writer_unary_interleaved_start_id_wh.cpp` (**borrowed**) | `set.core_ranges` | 0 cb_id_out(=output_index), 1 num_tiles_2d, 2 third_dim, 3 total_tiles_per_row, 4+ `TensorAccessorArgs(*dst_buffer)` | 0 dst_addr(=`dst_buffer`), 1 tile_start_id(=start_id), 2 single_block_size_row_arg, 3 single_block_size_col_arg | none | none | none | `WriterConfigDescriptor{}` |
| compute (×≤4, per region) | `data_movement/tilize/.../compute/tilize_wh.cpp` (**lent** — own dir) | see regions below | 0 block_size_col, 1 block_size_row, 2 third_dim, 3 dfb_id_in(=input_index), 4 dfb_id_out(=output_index) | none | none | none | O3 (resolved; `ComputeConfigDescriptor` has no `opt_level` field → compute default) | `ComputeConfigDescriptor{.fp32_dest_acc_en=fp32_llk_acc, .unpack_to_dest_mode=<vec>}` |

- reader/writer `opt_level`: absent on the `KernelDescriptor` → resolves to `O2` (DM default). No action.
- compute `opt_level`: absent → resolves to `O3` (compute default). **Metal 2.0 defaults to O2 → must set
  `O3` explicitly on every compute `KernelSpec`.**
- reader/writer `config`: plain `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` (resolved reader
  & writer *defaults*) → port to `ttnn::create_reader_datamovement_config(arch)` /
  `create_writer_datamovement_config(arch)`. **No** `disable_dfb_implicit_sync_for_all` (that is a knob
  the *default* sibling factory sets; the legacy block factory does not, and neither does the untilize
  block precedent).

Compute regions (4 possible, emitted conditionally — `...block_program_factory.cpp:225-240`):
| region | core range | buffer set | block_size_col (CTA0) | block_size_row (CTA1) |
|---|---|---|---|---|
| full | `core_range` | full | `single_sub_block_wh` | `single_sub_block_size` |
| cliff_col_row | `cliff_col_row_core_range` | cliffrow | `single_block_size_cliff_col` | `single_block_size_cliff_row` |
| cliff_row | `cliff_row_core_range` | cliffrow | `single_block_size` | `single_block_size_cliff_row` |
| cliff_col | `cliff_col_core_range` | full | `single_sub_block_cliff_col_wh` | `single_sub_block_size` |

### CBs
Six CBs across two buffer sets, created by `push_buffer_set` (`common.cpp:808-863`). Each `buffer_index`
is pushed exactly once at one size over that set's disjoint cores (`common.cpp:917-918`) → uniform
size per named DFB. Sizes (per set; `block_tiles` = the set's width in tiles):

| index | total_size | page_size (→ entry_size) | num_entries | data_format | tile |
|---|---|---|---|---|---|
| staging c_1 (full) / c_3 (cliffrow) | `(input_single_tile_size/tile_height)*block_tiles + 2*dram_alignment` | = total_size | 1 | input | (unset) |
| input c_0 (full) / c_2 (cliffrow) | `block_tiles*input_single_tile_size` | input_single_tile_size | block_tiles | input | `operation_attributes.tile` |
| output c_16 (full) / c_17 (cliffrow) | `block_tiles*output_single_tile_size` | output_single_tile_size | block_tiles | output | `operation_attributes.tile` |

No `GlobalCircularBuffer`, no `address_offset`, no borrowed-memory CB (all plain L1 scratch).

### Semaphores
none.

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| reader `TensorAccessor(src_args, src_addr)` | input (`src0_buffer`) | reader slot 0 + `TensorAccessorArgs(*src0_buffer)` CTA (`...:152`) |
| writer `TensorAccessor(dst_args, dst_addr)` | output (`dst_buffer`) | writer slot 0 + `TensorAccessorArgs(*dst_buffer)` CTA (`...:168`) |

Both **Case 1** (via `TensorAccessor`). Both slot-0 addresses are clean bases (`->address()`, no host fold).

### Work split
- Driver: `make_block_plan(BlockDirection::Tilize, BlockCoreOrder::ColumnMajor, a, output, …, sub_core_grids)`
  → `BlockPlan{split, full, cliffrow}` (`common.cpp:865-938`). `split` is `ttnn::BlockSplitWH`.
- Two buffer sets: `full` (core_range ∪ cliff_col) and `cliffrow` (cliff_row ∪ cliff_col_row). Either may be empty.
- RTA loop walks `corerange_to_cores(available_grid)` in column-major order (matching `ColumnMajor`), routing
  each core's args to its set via `buffer_set_for_core(plan, core)`.

### Shared kernels
All three, **rung 2 (create the first `_metal2` fork)** — no non-quasar fork exists (verified: the only
`_metal2` copies are under `experimental/quasar/**`, which do not count). Consumer set of each, after
disambiguating (quasar tree + CMakeLists + comment/substring hits discarded):

| kernel | owner | remaining consumer after this port | fork to create |
|---|---|---|---|
| `reader_unary_pad_multicore_both_dims.cpp` | tilize_with_val_padding (borrowed) | tilize_with_val_padding block | `reader_unary_pad_multicore_both_dims_metal2.cpp` (beside original) |
| `writer_unary_interleaved_start_id_wh.cpp` | eltwise/unary (borrowed) | tilize_with_val_padding block | `writer_unary_interleaved_start_id_wh_metal2.cpp` (beside original) |
| `tilize_wh.cpp` | data_movement/tilize (lent) | tilize_with_val_padding block | `tilize_wh_metal2.cpp` (beside original) |

Sunset set for all three = {tilize block (this factory), tilize_with_val_padding block}. Both still on
`descriptor`; TTNN-side gating means they cannot co-migrate implicitly → rung 2 is the expected path.
The `_wh` writer keeps its `#ifdef BACKWARDS` (faithful fork; block factory does not define it).

### Flags
- Unreferenced-by-this-factory kernels in the compute dir: `tilize.cpp`, `retile.cpp` — bound by *other*
  tilize factories, out of scope.
- `patch_tilize_kernel_slot0` (`tilize_device_operation.{hpp:45,cpp:372}`) — this factory
  (`...:435`) is its **only** caller. Dead after the port → remove decl + def (device-op cleanup in the
  same change). Confirmed by grep: all other references are comments.

## TTNN ProgramFactory
- **Concept (inherited from audit)**: `CustomProgramSpecFactoryConcept` (because `override_runtime_arguments`
  is present). Same target as the five ported siblings.
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: `create_descriptor`→`create_program_artifacts` (returns `ProgramArtifacts`);
  `override_runtime_arguments` return type `void`→`ProgramRunArgs`. Device-op edit forced: remove the now-dead
  `patch_tilize_kernel_slot0`. No pybind `create_descriptor` to remove (nanobind binds `ttnn::tilize` only).

## Planned Spec Shape
- **KernelSpecs** (≤8): `READER_FULL`, `WRITER_FULL` (if full non-empty); `READER_CLIFFROW`,
  `WRITER_CLIFFROW` (if cliffrow non-empty); `COMPUTE_FULL`, `COMPUTE_CLIFF_COL_ROW`, `COMPUTE_CLIFF_ROW`,
  `COMPUTE_CLIFF_COL` (each per its region conditional).
- **DataflowBufferSpecs** (≤6): `IN_FULL`, `STAGING_FULL`, `OUT_FULL` (full set); `IN_CLIFFROW`,
  `STAGING_CLIFFROW`, `OUT_CLIFFROW` (cliffrow set). Each set emitted only when non-empty (carry the
  legacy `for set in {full, cliffrow}: if set.empty() continue` conditional).
- **SemaphoreSpecs**: none.
- **TensorParameters**: `INPUT`, `OUTPUT`.
- **WorkUnitSpecs** (≤4): one per compute region — `wu_full`{READER_FULL,WRITER_FULL,COMPUTE_FULL}@core_range;
  `wu_cliff_col_row`{READER_CLIFFROW,WRITER_CLIFFROW,COMPUTE_CLIFF_COL_ROW}@cliff_col_row;
  `wu_cliff_row`{READER_CLIFFROW,WRITER_CLIFFROW,COMPUTE_CLIFF_ROW}@cliff_row;
  `wu_cliff_col`{READER_FULL,WRITER_FULL,COMPUTE_CLIFF_COL}@cliff_col.
- **Op-owned tensors**: none.

## Preserved Multiplicity
| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| 2 readers of `reader_unary_pad_multicore_both_dims.cpp` (full, cliffrow), disjoint core ranges | `READER_FULL`, `READER_CLIFFROW` | full readers listed in `wu_full`+`wu_cliff_col`; cliffrow reader in `wu_cliff_row`+`wu_cliff_col_row` | `IN_*` PRODUCER, `STAGING_*` self-loop (PRODUCER+CONSUMER) |
| 2 writers of `writer_unary_interleaved_start_id_wh.cpp` (full, cliffrow), disjoint | `WRITER_FULL`, `WRITER_CLIFFROW` | same work units as their readers | `OUT_*` CONSUMER |
| ≤4 computes of `tilize_wh.cpp` (2 bind full set, 2 bind cliffrow set), disjoint sub-ranges | `COMPUTE_FULL`, `COMPUTE_CLIFF_COL`, `COMPUTE_CLIFF_ROW`, `COMPUTE_CLIFF_COL_ROW` | one work unit each | `IN_*` CONSUMER, `OUT_*` PRODUCER |

Per node the census is **1P+1C** for every input/output DFB (the two computes binding a set run on
**disjoint** sub-ranges, so each node sees exactly one). This is the disjoint-node work-split (shared
reader/writer listed in each work unit over disjoint `target_nodes`), **NOT** multi-binding — do not set
`allow_instance_multi_binding`.

## Dropped Plumbing
| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA slot 0 (`...:303`) | `src0_buffer` (`Buffer*`) | `TensorBinding(INPUT, "src")` |
| reader CTA slot 6 (`...:150`) | `set.input_index` | `DFBBinding(IN_*, "in", PRODUCER)` |
| reader CTA slot 7 (`...:151`) | `*set.staging_index` | `DFBBinding(STAGING_*, "staging", PRODUCER)+(…,CONSUMER)` (self-loop) |
| reader CTA slot 8+ (`...:152`) | `TensorAccessorArgs(*src0_buffer)` | (folded into `TensorBinding`) |
| reader CTAs 0-5 | positional | named: total_num_rows, third_dim, tile_height, element_size, unpadded_X_size, dram_alignment |
| reader RTAs 1-8 | positional | named: pad_value, width_size, start_row_id, start_column_id, single_block_size_row_arg, single_block_size_col_arg, sub_block_width_size, single_sub_block_size_row_arg |
| writer RTA slot 0 (`...:315`) | `dst_buffer` (`Buffer*`) | `TensorBinding(OUTPUT, "dst")` |
| writer CTA slot 0 (`...:167`) | `set.output_index` | `DFBBinding(OUT_*, "out", CONSUMER)` |
| writer CTA slot 4+ (`...:168`) | `TensorAccessorArgs(*dst_buffer)` | (folded into `TensorBinding`) |
| writer CTAs 1-3 | positional | named: num_tiles_per_2d, third_dim, total_tiles_per_row |
| writer RTAs 1-3 | positional | named: start_id, single_block_size_row_arg, single_block_size_col_arg |
| compute CTA slot 3 (`...:215`) | `set.input_index` | `DFBBinding(IN_*, "in", CONSUMER)` |
| compute CTA slot 4 (`...:215`) | `set.output_index` | `DFBBinding(OUT_*, "out", PRODUCER)` |
| compute CTAs 0-2 | positional | named: block_size_col, block_size_row, third_dim |
| `override_runtime_arguments` (`...:380-442`) + `dm_kernel_metadata` CRTA (`...:366-371`) + width checks (`...:418-441`) + `patch_tilize_kernel_slot0` | slot-0 `Buffer*` re-point on cache hit | `override` returning `ProgramRunArgs{.tensor_args = {{INPUT, input}, {OUTPUT, output}}}` |
| writer `get_tile_size(cb_id_out)` (kernel line 24) | free fn on CB id | `dfb.get_tile_size()` (member getter; value is `const`, whitelist rule 7) |

## Applied Patterns
- [Self-loop DFB binding](../../../../../../docs/.../port_patterns.md): `STAGING_*` bound PRODUCER+CONSUMER on
  the set's single reader — single-ended (FIFO producer `reserve_back`/`push_back`, no consumer), one toucher.
- [Two-toucher / preserved-multiplicity (disjoint-node work-split)]: `IN_*`/`OUT_*` produced/consumed by two
  same-source compute KernelSpecs over disjoint sub-ranges → per-node 1P+1C, not multi-binding.
- [Conditional bindings]: each buffer set (and its 3 DFBs + reader/writer KernelSpecs + work units) is emitted
  only when the set is non-empty — carried as host-side `if (!set.empty())`, no kernel-side `#ifdef` needed
  (the whole KernelSpec is omitted, not a binding within a bound kernel).
- [Shared kernel — rung 2]: three first `_metal2` forks created beside their originals, pointer comments added.
- Custom-concept `override` translation (slot-0 patch → tensor_args).

## Deferred / Flagged
- none new. The reader's per-row DRAM-alignment fix-ups (`start_column_id` offset, `s.get_noc_addr(page)` +
  `.offset_bytes`) are kernel-side computations off the clean base, not a host-folded `base + offset` — no
  offset-base wall. Confirmed against the audit.
