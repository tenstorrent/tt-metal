# Port Plan — experimental/quasar/grid_sample

Port plan for `ttnn/cpp/ttnn/operations/experimental/quasar/grid_sample` (the quasar copy of `pool/grid_sample`), ported from the `ProgramDescriptor` API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

**Scope (user instruction):** port the quasar copy only. Nothing under the main tree (`pool/…`) may be added to or changed. The borrowed compute kernel is copied into this op's directory and converted there.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — both factories are structs in the `program_factory_t` variant (`device/grid_sample_device_operation.hpp:23-38`), each with `create_descriptor(...)`. Not the direct-descriptor shape.
- Variants: two factories, selected by `mode` (`device/grid_sample_device_operation.cpp:15-22`). Each has runtime configs: Bilinear {interleaved grid, sharded grid without split reader, sharded grid with split reader}, Nearest {interleaved grid, sharded grid} (the nearest split reader is always on, `device/grid_sample_utils.cpp:15-17`).
- Custom `compute_program_hash`: none. Default reflection hash over `GridSampleParams::attribute_values` (`device/grid_sample_device_operation_types.hpp:21-38`) — left intact.

### Variant: Bilinear (`device/grid_sample_bilinear_program_factory.cpp`, legacy line numbers)

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|
| reader0 | `kernels/dataflow/reader_grid_sample_interleaved_start_id.cpp` (interleaved) / `reader_grid_sample_sharded.cpp` (sharded) | all_cores | 0 input_cb, 1 grid_cb, 2 scalar_cb, 3 input_stick_size, 4 grid_stick_size, 5 input_batch, 6 input_height, 7 input_width, 8 grid_batching_factor, 9 grid_dtype, 10 grid_hw, 11 use_precomputed_grid, 12 align_corners, 13 in_nblocks_c, 14 input_chunk_nbytes, 15 last_chunk_partial; sharded adds 16 enable_split_reader, 17 reader_id=0, 18 grid_nsticks_per_core; then `TensorAccessorArgs(input)` (+ `(grid)` when interleaved) | interleaved: [input `Buffer*`, grid `Buffer*`, grid_sticks, grid_processed]; sharded: [input `Buffer*`, i*grid_nsticks_per_core] | none | O2 (unset, DM) | interleaved `ReaderConfigDescriptor{}` (RISCV_1/NOC_0); sharded `DataMovementConfigDescriptor{RISCV_0, RISCV_0_default}` |
| reader1 | `reader_grid_sample_sharded.cpp` | all_cores | as reader0 with 0 = input_cb_1, 2 = scalar_cb_1, 17 = 1 | [input `Buffer*`, i*grid_nsticks_per_core] | none | O2 | `{RISCV_1, RISCV_1_default}` — only when sharded && split |
| compute ×1 (sharded) or ×2 (interleaved, core_group_1 / core_group_2) | `pool/generic/device/kernels/compute/compute_pool_2d.cpp` (**borrowed**) | all_cores / core_group_1 / core_group_2 | 0 in_ntiles_c, 1 REDUCTION_SIZE, 2 enable_split_reader, 3 total_interpolations (**per group**), 4 channels, 5 in_nblocks_c, 6 MAX_ROWS_FOR_REDUCTION, 7 input_cb_0, 8 input_cb_1, 9 scalar_cb_0, 10 scalar_cb_1, 11 output_cb, 12 ONE_SCALAR_PER_CORE, 13 pre_tilize_cb=32, 14 is_output_tiled=0, 15 is_output_block_format=0, 16 force_4_tile_chunk, 17–37 mpwi/pool-only placeholders (not read by compute_pool_2d), 38 fast_tilize_cb=32 | none | `get_defines(AVG_POOL2D)` | O3 (unset, compute) | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, math_approx_mode}` from `get_compute_kernel_config_args` |
| writer | `kernels/dataflow/writer_grid_sample_interleaved.cpp` | all_cores | 0 output_cb, 1 output_stick_size, 2 out_ntiles_c, then `TensorAccessorArgs(output)` | [output `Buffer*`, output_sticks, output_processed] | none | O2 | `WriterConfigDescriptor{}` (RISCV_0/NOC_1) — interleaved only |

#### CBs
| index | total_size | core_ranges | data_format | page_size | face_geometry / tile | buffer |
|---|---|---|---|---|---|---|
| grid | pages × grid_stick_size (pages = grid_nsticks_per_core sharded, else 1) | all_cores | grid | grid_stick_size | — | grid buffer when sharded |
| input_0 | 2 × input_cb_page_size | all_cores | input | max_tiles_per_iter·TILE_HW·elem | {4, 2} | — |
| input_1 | same | all_cores | input | same | {4, 2} | — (sharded && split only) |
| scalar_0 | 2 × tile_size | all_cores | input | tile_size | {1, 2} | — |
| scalar_1 | same | all_cores | input | same | {1, 2} | — (sharded && split only) |
| output | pages × FACE_WIDTH·elem | all_cores | output | FACE_WIDTH·elem | {1, 1 or 2}; tile {1,16} when single partial | output buffer when sharded |

#### Work split
- Interleaved: `split_work_to_cores(compute_with_storage_grid_size, total_grid_nsticks)` → (num_cores, all_cores, core_group_1, core_group_2, num_sticks_1, num_sticks_2). Sharded: grid shard spec (grid, num_cores, shard height).

### Variant: Nearest (`device/grid_sample_nearest_program_factory.cpp`)

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs | opt_level | config |
|---|---|---|---|---|---|---|
| writer0 | `kernels/dataflow/writer_grid_sample_nearest_sharded.cpp` | all_cores | 0 grid_cb_0, 1 output_cb, 2 input_stick_size, 3 grid_stick_size, 4 input_height, 5 input_width, 6 grid_batching_factor, 7 grid_dtype, 8 grid_hw, 9 use_precomputed_grid, 10 align_corners, 11 split, 12 reader_id=0, 13 grid_nsticks_per_core, 14 is_sharded, 15 fill_cb, 16 batch; then `TensorAccessorArgs(input)` and `(grid)` interleaved / `(input)` again sharded | sharded: [input `Buffer*`, i*grid_nsticks_per_core]; interleaved: [input `Buffer*`, grid `Buffer*`, grid_sticks (**unread**), grid_processed] | O2 | `{RISCV_0, RISCV_0_default}` |
| writer1 | same source | all_cores | as writer0 with 0 = grid_cb_1 (interleaved) / grid_cb_0 (sharded), 12 = 1 | same as writer0 | O2 | `{RISCV_1, RISCV_1_default}` (whenever split; always for nearest) |

#### CBs
| index | total_size | data_format | page_size | buffer |
|---|---|---|---|---|
| grid_0 | pages × grid_stick_size | grid | grid_stick_size | grid buffer when sharded |
| grid_1 | grid_stick_size | grid | grid_stick_size | — (split && interleaved only) |
| fill | output_cb_page_size | output | output_cb_page_size | — |
| output | output_nsticks_per_core × page | output | output_cb_page_size | output buffer (always) |

#### Work split
- Sharded: grid shard spec. Interleaved grid: the output shard spec (output is always sharded for nearest).

### Semaphores
none

### Tensor accessors
| host site (legacy) | originating Tensor | RTA slot |
|---|---|---|
| bilinear reader interleaved `:247,249,435-436` | input, grid | 0, 1 |
| bilinear reader sharded `:247,409,416` | input | 0 |
| bilinear writer `:387,445` | output | 0 |
| nearest writers `:162-167,208-243` | input; grid (interleaved) | 0; 1 |

### Shared kernels
- `pool/generic/device/kernels/compute/compute_pool_2d.cpp` — **borrowed**. Other binders: `pool/generic/device/pool_multi_core_program_factory.cpp:939`, `pool/grid_sample/device/grid_sample_bilinear_program_factory.cpp`, `pool/rotate/device/rotate_bilinear_program_factory.cpp`, `experimental/quasar/rotate/device/rotate_bilinear_program_factory.cpp:294`. No `_metal2` copy beside it. **Rung: none of the recipe's three.** Per user decision (main tree is off limits) the port copies it into `device/kernels/compute/compute_pool_2d.cpp` in this op directory and converts the copy in place. Bound by this op and by quasar rotate (joint port).
- **Lent:** `writer_grid_sample_interleaved.cpp` is also bound by quasar rotate's bilinear factory, and rotate's bilinear reader includes `grid_sample_reader_common.hpp` (missed by the first audit pass; found by the re-check). Both convert in place because rotate is ported in the same change.
- Donor headers `pool/device/kernels/{pool_kernels_common,experimental_device_api}.hpp` are include-only; no change (`zero_out_tiles<uint32_t cb_id>` takes the `dfb::` token as a template argument).

### Flags
- none (no unreferenced kernel files).

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` (both factories).
- **Custom `compute_program_hash`**: none — leave intact.
- **Implementation notes**: rename `create_descriptor` → `create_program_artifacts` on both factory structs (`device/grid_sample_device_operation.hpp`). No pybind `create_descriptor` exists, so no binding removal.

## Planned Spec Shape

### Variant: Bilinear
- **TensorParameters**: `input`, `grid`, `output` (`tensor_spec()` of each).
- **DataflowBufferSpecs**: `grid` (borrowed_from `grid` when sharded), `in_0`, `in_1` (sharded+split only), `scalar_0`, `scalar_1` (sharded+split only), `out` (borrowed_from `output` when sharded). Legacy `face_geometry` → `tile_format_metadata` = the tile the build derived from it: {4,2} → `Tile({4,32})`, {1,2} → `Tile({1,32})`, {1,1} → `Tile({1,16})` (legacy JIT used exactly this tile for every unpack/pack metadata value and the tile size, `tt_metal/jit_build/jit_build_options.cpp:29-37,100-131`).
- **KernelSpecs**: `reader_0`, `reader_1` (sharded+split), `compute` (sharded) or `compute_group_1` / `compute_group_2` (interleaved), `writer` (interleaved).
- **WorkUnitSpecs**: sharded: one on all_cores {reader_0, [reader_1], compute}. Interleaved: {reader_0, compute_group_1, writer} on core_group_1 and {reader_0, compute_group_2, writer} on core_group_2.

### Variant: Nearest
- **TensorParameters**: `input`, `grid`, `output`.
- **DataflowBufferSpecs**: `grid_0` (borrowed_from `grid` when sharded), `grid_1` (interleaved+split), `fill`, `out` (borrowed_from `output`).
- **KernelSpecs**: `writer_0`, `writer_1` (split).
- **WorkUnitSpecs**: one on all_cores.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| bilinear compute ×2 (core_group_1/2, CTA 3 differs) | `compute_group_1`, `compute_group_2` | `…_group_1`, `…_group_2` (disjoint nodes) | `in_0` CONSUMER, `scalar_0` CONSUMER, `out` PRODUCER — each on its own nodes, no flag |
| bilinear sharded reader0/reader1 (same grid) | `reader_0`, `reader_1` | one | `grid`: reader_0 PRODUCER, reader_1 CONSUMER (two role-free touchers → 1P+1C) |
| nearest writer0/writer1 (same grid) | `writer_0`, `writer_1` | one | `fill`, `out`: writer_0 PRODUCER, writer_1 CONSUMER; `grid_0` (sharded) same; interleaved: `grid_0` self-loop on writer_0, `grid_1` self-loop on writer_1 |

## Dropped Plumbing

| legacy location (legacy file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| bilinear reader RTA 0 (`:409,416,435`) | `input_tensor.buffer()` | `TensorBinding(input)` → `TensorAccessor(tensor::input)` |
| bilinear interleaved reader RTA 1 (`:436`) | `grid_tensor.buffer()` | `TensorBinding(grid)` → `TensorAccessor(tensor::grid)` |
| bilinear writer RTA 0 (`:445`) | `output_tensor.buffer()` | `TensorBinding(output)` → `TensorAccessor(tensor::output)` |
| nearest writers RTA 0 (`:208,215,231,242`) | `input_tensor.buffer()` | `TensorBinding(input)` |
| nearest writers RTA 1 interleaved (`:232,243`) | `grid_tensor.buffer()` | `TensorBinding(grid)` (interleaved only; `GRID_INTERLEAVED` define) |
| nearest writers RTA 2 interleaved (`:233,244`) | `grid_sticks` — **never read by the kernel** | dropped (dead arg; zero behavior change) |
| bilinear reader CTAs 0,1,2 / reader1 0,2 | input/grid/scalar CB indices | `DFBBinding`s `in`, `grid`, `scalar` (reader_1 binds `in_1`/`scalar_1` under the same kernel names) |
| bilinear reader CTA 5 (interleaved source) | `input_batch` — not read by the interleaved reader | not declared for the interleaved reader (still passed to the sharded one, which reads it) |
| bilinear compute CTAs 7–11, 13, 38 | CB indices (13 and 38 = dummy 32; 8, 10 = dummy 32 unless split) | `DFBBinding`s `in_0`, `in_1`, `scalar_0`, `scalar_1`, `out`; pre_tilize / fast_tilize never bound (`OUTPUT_TILED` never defined) |
| bilinear compute CTAs 17–37 | pool/mpwi-only placeholders, not read by compute_pool_2d | dropped |
| bilinear writer CTA 0 | output CB index | `DFBBinding` `out` |
| nearest writer CTAs 0, 1, 15 | grid / output / fill CB indices | `DFBBinding`s `grid`, `out`, `fill` |
| all `TensorAccessorArgs(...).append_to(...)` (`bilinear:247,249,387`, `nearest:162-167`) and kernel `TensorAccessorArgs<N>()` chains | accessor plumbing | the binding mechanism |
| nearest sharded second `TensorAccessorArgs(input)` + kernel's `grid_addr = 0` accessor | unused placeholder accessor | not built when the grid is sharded (`#ifdef GRID_INTERLEAVED`) |
| every remaining positional CTA / RTA | positional | named, using the kernel's own local variable names |

## Applied Patterns

- Self-loop DFB (sync-free / single-ended): bilinear `grid` (interleaved, or sharded without split) on reader_0; bilinear `out` (sharded) on compute; nearest `grid_0` / `grid_1` (interleaved) on their writer.
- Two-toucher DFB → 1P+1C: bilinear `grid` (sharded+split); nearest `fill`, `out`, `grid_0` (sharded).
- Conditional / optional resource bindings: compute `in_1`/`scalar_1` (`SPLIT_READER`), `pre_tilize`/`fast_tilize` (`OUTPUT_TILED`, never set by this op); nearest `tensor::grid` (`GRID_INTERLEAVED`, a promoted `is_sharded` CTA gate, also gating the RTA read that differs between configs).
- Pass DFB handles directly to LLKs / helpers: compute LLK calls; `zero_out_tiles<dfb::in>`, `process_grid_point<…, dfb::in, dfb::scalar>`, `push_noop_sticks<dfb::scalar, …>`.
- Multi-variant factories: config selection stays inside each `create_program_artifacts`.
- Preserved per-group compute KernelSpecs (anti-pattern "Demoting per-group CTA to RTA" avoided).

## Deferred / Flagged

- **`face_geometry` has no `DataflowBufferSpec` field.** Resolved by carrying the equivalent tile (see Planned Spec Shape); no behavior change because the legacy build used the same tile. Recorded as a doc gap in the report.
- **Compute runtime-arg fallback.** `compute_pool_2d` reads RTA 0 only when `max_out_sticks_per_core == 0` (pool's path; grid_sample always passes a non-zero count). In the copy that read is gated behind `OUT_STICKS_PER_CORE_FROM_RTA`, which this op never defines, so no runtime arg is declared for compute (none existed in legacy).
- **Pre-existing test failures**: 2 of 944 baseline tests fail on the unported quasar copy (see report).
