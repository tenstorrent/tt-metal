# Metal 2.0 Port Plan — `experimental/quasar/untilize`

Audit: GREEN. Recipe @ `origin/akertesz/metal2-documentation`. Clean reader→compute→writer op (no scratch CBs — verified). Strong reference: ported sibling `untilize_with_unpadding`.
**Pass 1 unit: `UntilizeSingleCoreProgramFactory`.** Other 7 factories stay on ProgramDescriptor concept (per-factory dispatch); enumerated as remaining work.

## Legacy Inventory (single_core)
- Factory concept: ProgramDescriptorFactoryConcept (`create_descriptor` → ProgramDescriptor).
- Custom `compute_program_hash`: NONE on UntilizeDeviceOperation. No device-op edit.
- Kernels: reader `reader_unary_start_id.cpp` (shared w/ multi_core → FORK); writer `writer_unary_stick_layout_split_rows_single_core.cpp` (single_core-only → in place); compute `untilize.cpp` (shared w/ 3 factories → FORK).
- CBs: c_0 (in, num_tiles_per_block) reader→compute; c_16 (out, num_tiles_per_block) compute→writer. Clean SPSC, no scratch.
- Tensors: input (Case 1 TensorAccessor), output (Case 1).
- Dead CTAs dropped: reader has none; writer CTA[1] `output_stick_size` unread → dropped.

## Forks
- `compute/untilize.cpp` → `compute/untilize_metal2.cpp` (NEW; per_core_block_cnt/tile_cnt + dfb::in/out — identical to uwu's untilize_metal2). Legacy retained for parallelize_column / input_and_output_shard_*_identical / sub_core_grids.
- `reader_unary_start_id.cpp` → `reader_unary_start_id_metal2.cpp` (NEW; dfb::in/tensor::input). Legacy retained for multi_core.

## Spec shape
- DFBs IN/OUT; TensorParameters INPUT/OUTPUT; KernelSpecs READER/WRITER/COMPUTE on 1 core; WorkUnit wu.
- Dropped: buffer-addr RTAs→TensorBinding; TensorAccessorArgs→bindings; CB indices→DFBBinding; positional CTAs→named; dead writer CTA.

## Remaining factories (next passes)
multi_core, multi_core_block, multi_core_parallelize_column, multi_core_sub_core_grids, multi_core_nd_shard_input, multi_core_input_and_output_{shard,nd_shard}_type_and_shard_spec_identical. **Check each for scratch CBs first** (untilize readers are single-CB clean, so likely all portable; sharded variants TBD).

## Pass 2: parallelize_column (PORTED)
Full+cliff compute; forks reader_unary_interleaved_start_id_metal2 + writer_..._parallel_columns_metal2 (both shared w/ sub_core_grids); reuses untilize_metal2 compute fork. LATENT CLIFF BUG corrected+flagged (see REPORT). sub_core_grids is now the cheapest next factory (reuses all 3 forks).
