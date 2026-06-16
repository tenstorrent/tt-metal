# Metal 2.0 Port Report — tilize (single-core factory)

## Status: PORTED

## Factory chosen
`TilizeSingleCoreProgramFactory` — the simplest single-program factory in the op
(one core, three kernels, two CBs, no work-split). The other four factories
(`TilizeMultiCoreDefaultProgramFactory`, `TilizeMultiCoreBlockProgramFactory`,
`TilizeMultiCoreShardedProgramFactory`, `TilizeMultiCoreWidthShardedProgramFactory`)
remain on the legacy `ProgramDescriptor` concept. The framework's
`program_factory_t` variant supports a mix of `ProgramSpecFactoryConcept` and
`ProgramDescriptorFactoryConcept` alternatives (per `ttnn/api/ttnn/operation_concepts.hpp`),
and dispatches per-factory at runtime, so the op continues to build and run with one
factory ported.

## Legacy inventory (ported factory)
- Concept: `ProgramDescriptorFactoryConcept` (`create_descriptor` → `ProgramDescriptor`).
- Custom `compute_program_hash`: none on `TilizeDeviceOperation`. Nothing to delete.
- CBs: c_0 (`src0`, input format, `num_tiles_per_block` entries) and c_16 (`output`,
  output format, `num_tiles_per_block` entries).
- Kernels:
  - reader `reader_unary_stick_layout_split_rows_singlecore.cpp` (op-local, used only by
    this factory) — `ReaderConfigDescriptor`. RTAs: src_addr(0), num_sticks(1),
    stick_size(2, UNUSED by kernel), num_tiles_per_block(3), block_width_size(4),
    num_full_blocks_in_row(5), num_leftover_tiles(6, UNUSED), leftover_width_in_row(7, UNUSED),
    row_start_id(8). CTA: stick_size + TensorAccessorArgs(src).
  - writer `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
    (SHARED across ~dozens of ops) — `WriterConfigDescriptor`. RTAs: dst_addr(0),
    num_tiles(1), start_id(2). CTA: output_cb_index(0) + TensorAccessorArgs(dst).
  - compute `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` (SHARED across many tilize/
    tilize_with_val_padding factories) — `ComputeConfigDescriptor` with `fp32_dest_acc_en`
    and `unpack_to_dest_mode[c_0]=UnpackToDestFp32` when fp32. CTAs: per_core_block_cnt(0),
    per_core_block_tile_cnt(1).
- Work split: none (single core, fixed at `(0,0)` or the first core of `sub_core_grids`).

## Spec shape (post-port)
- 2 DataflowBufferSpecs: `src0`, `output` (1:1 with legacy CBs).
- 3 KernelSpecs: `reader`, `writer`, `compute`.
- 2 TensorParameters: `input`, `output`; matching TensorArguments via `mesh_tensor()`.
- 1 WorkUnitSpec hosting all three kernels on `core_ranges` (Local-DFB rule:
  reader produces `src0`, compute consumes `src0` + produces `output`, writer consumes
  `output` — all three share the one WorkUnitSpec, so every DFB's producer and consumer
  are co-located).

## Kernels: ported vs forked
- **reader** `reader_unary_stick_layout_split_rows_singlecore.cpp` — **ported in place**
  (op-local, single user). CircularBuffer→DataflowBuffer; CB id `c_0`→`dfb::src0`;
  `TensorAccessorArgs<1>`+addr RTA→`TensorAccessor(ta::input)`; positional RTAs→named
  (`args::num_sticks`, `args::num_tiles_per_block`, `args::block_width_size`,
  `args::num_full_blocks_in_row`, `args::row_start_id`).
- **writer** — **FORKED** to op-local
  `device/kernels/dataflow/writer_unary_interleaved_start_id_m2.cpp` (the eltwise source is
  shared by many other ops; not editable in place). Same mechanical conversion;
  `OUT_SHARDED`/`BACKWARDS` `#ifdef`s preserved (neither is defined for tilize single-core,
  so only the default interleaved path is active — behavior unchanged).
- **compute** — **FORKED** to op-local `device/kernels/compute/tilize_m2.cpp` (the
  `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` source is shared by many factories). CB ids
  `c_0`/`c_16`→`dfb::src0`/`dfb::output` (assigned to `constexpr uint32_t` locals first,
  matching the matmul `bmm_m2` idiom, so they flow as non-type template args to
  `is_fp32_input_format<>` and `tilize<>`); CTAs→`args::per_core_block_cnt` /
  `args::per_core_block_tile_cnt`. `tilize_helpers.hpp` (kernel_lib) left untouched —
  it takes CB ids as `uint32_t`, which `dfb::` converts to implicitly.

## Dropped plumbing
- Reader `src_addr` RTA(0) → `TensorBinding{input}`.
- Reader unused RTAs `stick_size`(2), `num_leftover_tiles`(6), `leftover_width_in_row`(7):
  dropped — the kernel never read them. The host stops emitting them.
- Writer `dst_addr` RTA(0) → `TensorBinding{output}`; `output_cb_index` CTA(0) → `dfb::output`.
- Compute magic CB ids `c_0`/`c_16` → DFB bindings.
- All positional CTAs/RTAs → named args. All `TensorAccessorArgs` host plumbing removed.

## Applied patterns
- Single WorkUnitSpec with all three kernels (Local-DFB rule).
- `unpack_to_dest_mode` carried via `ComputeHardwareConfig::unpack_to_dest_mode`
  (`Table<DFBSpecName, UnpackToDestMode>`), keyed on `src0` (the fp32 compute-consumer DFB),
  only when `fp32_llk_acc` — preserving the legacy `unpack_to_dest_mode[c_0]` behavior.

## Device-op-class edits
- `tilize_single_core_program_factory.hpp`: `create_descriptor`→`create_program_spec`,
  return type `ProgramDescriptor`→`ttnn::device_operation::ProgramArtifacts`; include
  `ttnn/metal2_artifacts.hpp` instead of `program_descriptors.hpp`.
- No change needed to `tilize_device_operation.hpp/.cpp` — the variant already lists
  `TilizeSingleCoreProgramFactory`, and `select_program_factory` already routes to it.
- nanobind (`tilize_nanobind.cpp`): no `create_descriptor` hook for this factory; nothing to remove.

## Findings / open items (routed, not changed)
- The legacy reader emitted three RTAs the kernel never reads (slots 2/6/7). Left the
  device-op/other-factory code untouched; only this factory's emission was trimmed as part
  of the named-arg conversion.
- `validate_on_program_cache_miss` comment "Assuming bfloat16 dataformat" on `stick_size`
  predates the multi-dtype support now present — flagging for owner review (not touched).

## Build / test
NOT built, NOT committed (per instructions — this worktree has no build dir and the
deliverable is a patch + report).

## Assumptions to verify at build time
- `dfb::name` usable as a `constexpr uint32_t` non-type template argument: confirmed by the
  `dgomez/rand-metal2` matmul `bmm_m2.cpp` exemplar (`constexpr uint32_t cb_in0 = dfb::in0;`).
- `get_local_cb_interface(dfb::output)` in the forked writer: `dfb::output` converts to
  `uint32_t`, matching the legacy `get_local_cb_interface(cb_id_out)` call. Verify the DFB
  interface is populated identically to the legacy local-CB interface.
