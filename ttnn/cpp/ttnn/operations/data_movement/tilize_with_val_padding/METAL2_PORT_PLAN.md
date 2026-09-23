# Port Plan — `tilize_with_val_padding` · `TilizeWithValPaddingMultiCoreBlockInterleavedFactory`

Port plan for the **block-interleaved** factory of `data_movement/tilize_with_val_padding`, ported from
the `ProgramDescriptor` (`descriptor`) concept to Metal 2.0 `ProgramSpecFactoryConcept`.
Written during the inventory and planning steps; committed alongside the port for review.

Scope: **only** `TilizeWithValPaddingMultiCoreBlockInterleavedFactory`. The three sibling factories
(SingleCore / MultiCoreDefault / MultiCoreSharded) stay on the `descriptor` concept — the
`program_factory_t` variant dispatches per-factory, so a half-ported op builds and runs.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `create_descriptor()` returns a `ProgramDescriptor`
  (`factories/…_multi_core_block_interleaved_program_factory.hpp:14-17`). Methods live in a
  `program_factory_t` **variant** struct (`device/tilize_with_val_padding_device_operation.hpp:27-31`),
  not directly on the device-op → **not** the direct-descriptor shape; no `exception 3` edit needed.
- Variants: single (this device-op has one attribute set; four *factories* in the variant, of which
  this is one).
- Custom `compute_program_hash`: none — default reflection-based hash (device-op has no override,
  no `attribute_values`/`to_hash`).

*(Target concept `ProgramSpecFactoryConcept` inherited from the audit brief — carried forward below.)*

### Kernels
One reader + one writer per non-empty buffer set (2 sets), and up to four compute instances over
disjoint core ranges. All three kernel **sources** are shared (see Shared kernels).

| unique_id | source | core_ranges | CTAs (positional, legacy) | RTAs (legacy) | config | opt_level |
|---|---|---|---|---|---|---|
| reader (×2: full, cliffrow) | `…/tilize_with_val_padding/device/kernels/dataflow/reader_unary_pad_multicore_both_dims.cpp` | `set.core_ranges` | `total_num_rows, third_dim, tile_height, element_size, unpadded_row_size_bytes, dram_alignment, set.input_index, *set.staging_index`, then `TensorAccessorArgs(*src0_buffer)` | `src0_buffer`(slot0), `packed_pad_value`, `width_size`, `start_row_id`, `start_column_id`, `single_block_size_row_arg`, `single_block_size_col_arg`, `sub_block_width_size`, `single_sub_block_size_row_arg` | `ReaderConfigDescriptor{}` | O2 (DM default) |
| writer (×2: full, cliffrow) | `…/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_wh.cpp` | `set.core_ranges` | `set.output_index, num_tiles_2d, third_dim, total_tiles_per_row`, then `TensorAccessorArgs(*dst_buffer)` | `dst_buffer`(slot0), `tile_start_id`, `single_block_size_row_arg`, `single_block_size_col_arg` | `WriterConfigDescriptor{}` | O2 (DM default) |
| compute (×4) | `…/data_movement/tilize/device/kernels/compute/tilize_wh.cpp` | see regions ↓ | `block_size_col, block_size_row, third_dim, set.input_index, set.output_index` | none | `ComputeConfigDescriptor{.fp32_dest_acc_en = fp32_llk_acc, .unpack_to_dest_mode = unpack_to_dest_mode}` | **O3** (compute default) |

Compute regions (source `tilize_wh.cpp`, `factory.cpp:217-232`) — preserved as 4 distinct `KernelSpec`s:

| region | cores | set | block_size_col | block_size_row (= set.block_tiles) | present when |
|---|---|---|---|---|---|
| COMPUTE_FULL | `core_range` | full | `single_sub_block_wh` | `single_sub_block_size` | `!core_range.empty()` |
| COMPUTE_CLIFF_COL_ROW | `cliff_col_row_core_range` | cliffrow | `single_block_size_cliff_col` | `single_block_size_cliff_row` | `has_cliff_col && has_cliff_row` |
| COMPUTE_CLIFF_ROW | `cliff_row_core_range` | cliffrow | `single_block_size` | `single_block_size_cliff_row` | `has_cliff_row` |
| COMPUTE_CLIFF_COL | `cliff_col_core_range` | full | `single_sub_block_cliff_col_wh` | `single_sub_block_size` | `has_cliff_col` |

`opt_level` resolved: reader/writer descriptors set no `opt_level` → **O2** (DM default, unchanged on
Metal 2.0). Compute `ComputeConfigDescriptor` sets none → **O3** (legacy compute default); Metal 2.0
defaults `CompilerOptions.opt_level` to O2, so **each** compute `KernelSpec` must set
`opt_level = O3` explicitly.

### CBs (from `push_buffer_set`, `data_movement/common/common.cpp:808-863`; block factory passes `tile=nullopt`)
Per non-empty set, three CBs, each sized once over the whole set from that set's scalar `block_tiles`
(this is the #51305 fix — no CB index allocated at two sizes across nodes):

| index (full / cliffrow) | role | total_size | page_size (→ entry_size) | num_entries | data_format |
|---|---|---|---|---|---|
| c_1 / c_3 | staging | `input_row_bytes*block_tiles + 2*dram_alignment` | == total_size | 1 | input_cb_data_format |
| c_0 / c_2 | input | `block_tiles*input_single_tile_size` | `input_single_tile_size` | `block_tiles` | input_cb_data_format |
| c_16 / c_17 | output | `block_tiles*output_single_tile_size` | `output_single_tile_size` | `block_tiles` | output_cb_data_format |

`input_row_bytes = input_single_tile_size / TILE_HEIGHT`. `full.block_tiles = single_sub_block_size`;
`cliffrow.block_tiles = single_block_size_cliff_row`. No aliasing, no borrowed memory, no global CB.

### Semaphores
none.

### Tensor accessors
| host site | originating Tensor | legacy RTA slot |
|---|---|---|
| reader `TensorAccessor(src_args, src_addr)` | input (`a` / `src0_buffer`) | slot 0 (`Buffer*`) |
| writer `TensorAccessor(dst_args, dst_addr)` | output (`dst_buffer`) | slot 0 (`Buffer*`) |

Both **Case 1** (consumed only via `TensorAccessor`). Address arrives today as a `Buffer*` pushed at
RTA slot 0 (framework `BufferBinding` auto-registration), not `->address()+offset` — clean base.

### Work split
- Driver: `make_block_plan(BlockDirection::Tilize, BlockCoreOrder::ColumnMajor, a, output, …)` →
  `plan.split` (a `BlockSplitWH`) + `plan.full` / `plan.cliffrow` `BlockBufferSet`s.
- Core order **ColumnMajor** → runtime-arg loop walks `corerange_to_cores(available_grid)` (NOT
  `grid_to_cores`; contrast the untilize sibling which is RowMajor). This ordering is
  correctness-relevant and is preserved verbatim.
- `full.core_ranges = core_range ∪ (has_cliff_col ? cliff_col_core_range : {})`;
  `cliffrow.core_ranges = has_cliff_row ? (cliff_row_core_range ∪ (has_cliff_col ? cliff_col_row_core_range : {})) : {}`.

### Shared kernels
All three sources are shared with the **still-legacy tilize block factory**
(`data_movement/tilize/device/tilize_multi_core_block_program_factory.cpp`). Census
(`grep -rl <filename> ttnn/cpp/ttnn/operations/`, quasar + build-file + `.md` hits discarded;
`untilize_wh` hits on `tilize_wh` are substring false positives):

| kernel | ownership | co-binders (non-quasar) | `_metal2` fork beside original? | rung |
|---|---|---|---|---|
| `reader_unary_pad_multicore_both_dims.cpp` | **lent** (in this op's dir; tilize also binds it) | {this factory, tilize block} | none (quasar copy doesn't count) | **create** |
| `writer_unary_interleaved_start_id_wh.cpp` | **borrowed** (eltwise/unary) | {this factory, tilize block} | none — the adjacent `writer_unary_interleaved_start_id_metal2.cpp` is the **non-`_wh`** variant, do not reuse | **create** |
| `tilize_wh.cpp` | **borrowed** (data_movement/tilize) | {this factory, tilize block} | none — `kernel/compute/tilize_metal2.cpp` is a different tilize kernel | **create** |

The three sibling factories of *this* op bind **none** of these (they use
`reader_unary_pad_dims_split_rows*.cpp` / `…_height_width_sharded.cpp`), so no intra-op coupling.
Each fork is created **beside its original** (rung 2), a pointer comment added to each original, and
the co-borrower `tilize` block factory keeps binding the legacy copies until it ports (sunset set =
{this factory, tilize block}).

### Flags
- Writer carries a `#ifdef BACKWARDS` block (untilize direction); this factory sets no such define
  (forward path). The fork **must preserve** the `#ifdef` for the co-borrowers.
- Stale comment `// Assuming bfloat16 dataformat` on `unpadded_row_size_bytes`/`padded_row_size_bytes`
  (`factory.cpp:82-83`) — computed with `a.element_size()`, correct for any dtype. Not a bug; not
  touched (routed to report).
- Reader re-reads RTA slots 3-8 each `third_dim` iteration at **constant** indices — same named args
  read in a loop; harmless, named once each.

## TTNN ProgramFactory
- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` (base — the factory declares no
  `override_runtime_arguments`; framework refreshes tensor bindings on cache hit).
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: method swap `create_descriptor` → `create_program_artifacts` inside the
  existing `TilizeWithValPaddingMultiCoreBlockInterleavedFactory` struct. No device-op-class edit
  (variant already declares the factory struct; no pybound `create_descriptor`; no custom hash). The
  other three factories remain on `descriptor` in the same variant.

## Planned Spec Shape
- **KernelSpecs (8)**: `READER_FULL`, `WRITER_FULL`, `READER_CLIFFROW`, `WRITER_CLIFFROW` (each per
  non-empty set), `COMPUTE_FULL`, `COMPUTE_CLIFF_COL_ROW`, `COMPUTE_CLIFF_ROW`, `COMPUTE_CLIFF_COL`
  (one per present region). Sources point at the three new `_metal2` forks.
- **DataflowBufferSpecs (6, per non-empty set ×3)**: `STAGE_FULL`/`STAGE_CLIFFROW` (self-loop),
  `IN_FULL`/`IN_CLIFFROW`, `OUT_FULL`/`OUT_CLIFFROW`. Sizes mirror `push_buffer_set`. Kept on
  **distinct** names — collapsing them reintroduces the #51305 corruption.
- **SemaphoreSpecs**: none.
- **TensorParameters (2)**: `INPUT` (`a.tensor_spec()`), `OUTPUT` (`output.tensor_spec()`), relaxation
  `none` (strict).
- **WorkUnitSpecs (up to 4)**: one per present compute region, each = {that set's reader, that set's
  writer, the region's compute} over the region's cores. Shared reader/writer are listed per region
  (disjoint `target_nodes` → union placement), the preserved-multiplicity wiring.
- **Op-owned tensors**: none.

## Preserved Multiplicity
Legacy emits 4 compute `KernelDescriptor`s of one source over disjoint core ranges (+ a reader/writer
per set). Preserved 1:1 — no CTA→RTA demotion.

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| 4× `tilize_wh.cpp` (disjoint cores) | COMPUTE_FULL, COMPUTE_CLIFF_COL_ROW, COMPUTE_CLIFF_ROW, COMPUTE_CLIFF_COL | wu_full, wu_cliff_col_row, wu_cliff_row, wu_cliff_col | full-set: IN_FULL (CONSUMER), OUT_FULL (PRODUCER) by COMPUTE_FULL+COMPUTE_CLIFF_COL over disjoint nodes; cliffrow-set: IN_CLIFFROW/OUT_CLIFFROW by COMPUTE_CLIFF_ROW+COMPUTE_CLIFF_COL_ROW |
| 2× reader, 2× writer (one per set, over the set's union of regions) | READER_FULL/WRITER_FULL, READER_CLIFFROW/WRITER_CLIFFROW | listed in each of their set's region WUs | full: IN_FULL PRODUCER (reader), OUT_FULL CONSUMER (writer), STAGE_FULL self-loop (reader P+C); cliffrow: analogous |

Per node: exactly 1 reader + 1 compute + 1 writer on each I/O DFB → validator's per-node 1P+1C holds.
Multiple same-role bindings (e.g. two compute CONSUMERs of IN_FULL) are legal because their node sets
are **disjoint**, same kernel kind, identical binding-site params. **Not** the multi-binding flag.

## Dropped Plumbing
| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA slot 0 (`factory.cpp:294`) + kernel `src_addr = get_arg_val<uint32_t>(0)` (reader:34) | `Buffer* src0_buffer` | `TensorParameter INPUT` + `TensorBinding{INPUT,"src"}`; kernel `TensorAccessor(tensor::src)` |
| reader CTA `TensorAccessorArgs(*src0_buffer)` (`factory.cpp:145`) + kernel `TensorAccessorArgs<8>()` (reader:32) | accessor-args plumbing | dropped — binding token supplies layout/page size |
| reader CTA slot 6 `set.input_index` (reader:28 `dfb_id_in0`) | magic CB index | `DFBBinding IN_* → dfb::in` |
| reader CTA slot 7 `*set.staging_index` (reader:29 `dfb_id_in1`) | magic CB index | `DFBBinding STAGE_* (self-loop) → dfb::stage` |
| writer RTA slot 0 (`factory.cpp:306`) + kernel `dst_addr = get_arg_val<uint32_t>(0)` (writer:11) | `Buffer* dst_buffer` | `TensorParameter OUTPUT` + `TensorBinding{OUTPUT,"dst"}`; kernel `TensorAccessor(tensor::dst)` |
| writer CTA `TensorAccessorArgs(*dst_buffer)` (`factory.cpp:161`) + kernel `TensorAccessorArgs<4>()` (writer:20) | accessor-args plumbing | dropped |
| writer CTA slot 0 `set.output_index` (writer:16 `cb_id_out`) | magic CB index | `DFBBinding OUT_* → dfb::out` |
| compute CTA slots 3,4 `set.input_index/output_index` (compute:19-20) | magic CB indices | `DFBBinding IN_*/OUT_* → dfb::in/dfb::out` |
| all positional CTAs/RTAs | positional | named CTAs (`compile_time_args` map) + named RTAs (`runtime_arg_schema` / `get_arg(args::…)`) |

No page-size 3rd-arg CTA/RTA (both accessors are 2-arg). No semaphore-ID RTAs.

## Applied Patterns
- **[Self-loop DFB binding]** — STAGE_FULL/STAGE_CLIFFROW on each reader `KernelSpec` (PRODUCER +
  CONSUMER, shared accessor `stage`). Single-toucher DM scratchpad → self-loop (Gen1-legal DM
  self-loop; a Quasar-uplift item, not a Gen1 blocker).
- **[Two-toucher / disjoint-node preserved multiplicity]** — 4 compute + 2 reader/writer over disjoint
  node sets binding shared DFBs; each node sees one instance (1P+1C). *Not* the multi-binding flag.
- **[Pass DFB handles directly to LLKs/kernel-lib]** — compute passes `dfb::in`/`dfb::out` as both
  NTTP (`is_fp32_input_format<dfb::in>()`, `tilize<…,dfb::in,dfb::out,…>`) and call args
  (`compute_kernel_hw_startup`); constexpr `operator uint32_t()` bridges both positions.
- **[Porting a shared kernel]** — rung 2 (create fork) for all three sources.
- **[Removing pybound legacy factory entry points]** — N/A (no pybound `create_descriptor`).

## Deferred / Flagged
- **unpack_modes / enable_32_bit_dest**: `fp32_llk_acc` (input FLOAT32/FP8_E4M3, or output
  FP8_E4M3/BFLOAT8_B) drives both `enable_32_bit_dest` and the legacy
  `unpack_to_dest_mode[input_index]=UnpackToDestFp32`. Ported per compute `KernelSpec` as
  `enable_32_bit_dest = fp32_llk_acc` + (when true) `unpack_modes.insert({its input DFB, UnpackToDest})`.
  Keyed only to the DFB **that kernel binds** (validator rejects a foreign-DFB entry; the legacy
  shared vector's extra entry for the other set's input CB is dropped). Verified validator-safe:
  `UnpackToDest` with `enable_32_bit_dest=true` is always permitted (`program_spec.cpp:1515-1516`);
  the ≤16-bit-format rejection only fires when `enable_32_bit_dest=false`, which never coincides with
  setting `UnpackToDest` here.
- No new findings that change the audit's GREEN verdict.
