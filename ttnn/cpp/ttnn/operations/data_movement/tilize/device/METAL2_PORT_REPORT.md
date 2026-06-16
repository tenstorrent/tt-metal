# Metal 2.0 Port Report — tilize (single-core + multi-core default factories)

## Status: PORTED

---

# Multi-core default factory (added port)

## Status: PORTED

## Factory chosen
`TilizeMultiCoreDefaultProgramFactory` — the default-selected factory for normal
(interleaved, non-sharded) inputs (`select_program_factory` returns it as the fall-through
case; `TilizeMultiCoreBlockProgramFactory` is the only alternative for tall/wide blocks).
Interleaved multi-core, three kernel roles, two DFBs, work-split across the compute grid
with a full core group + optional cliff core.

## Legacy inventory (this factory)
- Concept: `ProgramDescriptorFactoryConcept` (`create_descriptor` → `ProgramDescriptor`).
- Custom `compute_program_hash`: none on `TilizeDeviceOperation`. Nothing to delete/keep.
- CBs: c_0 (`src0`, input format, `ntiles_per_block` entries) and c_16 (`output`, output
  format, `ntiles_per_block` entries), both on `all_cores`.
- Work-split: `split_blocks_for_tilize(available_grid, nblocks)` → `core_range` (full cores,
  `nblocks_per_core` each) + `core_range_cliff` (≤1 cliff core, `nblocks_per_core_cliff`).
  Reader/writer placed on `all_cores`; compute on `core_range`, compute_cliff on
  `core_range_cliff`.
- Kernels:
  - reader `reader_unary_stick_layout_split_rows_multicore.cpp` (op-local, used only by this
    factory) — CTs `{aligned_page_size(0, UNUSED), num_pages_in_row(1), size_of_valid_data(2)}`
    + `TensorAccessorArgs<3>`. RTAs: src_addr(0), num_rows(1), page_size(2, UNUSED),
    ntiles_per_block(3), block_width_size(4), num_full_blocks_in_row(5), num_leftover_tiles(6,
    UNUSED), leftover_width_in_row(7, UNUSED), start_page_id(8).
  - writer `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` (SHARED) — CT
    `output_cb_index` + TensorAccessorArgs(dst). RTAs dst_addr(0), num_tiles(1), start_id(2).
  - compute `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` (SHARED) — `fp32_dest_acc_en` +
    `unpack_to_dest_mode[c_0]=UnpackToDestFp32` when fp32. CTAs per_core_block_cnt(0),
    per_core_block_tile_cnt(1). Full vs cliff differ only in per_core_block_cnt.

## Spec shape (post-port)
- 2 DataflowBufferSpecs: `src0`, `output`.
- KernelSpecs: `reader`, `writer`, plus `compute` (full group) and/or `compute_cliff`.
- 2 TensorParameters: `input`, `output`.
- WorkUnitSpecs (Local-DFB rule, matmul-multicore pattern): one per core group, each hosting
  reader + writer + that group's compute. `full` = {reader, writer, compute} on `core_range`;
  `cliff` = {reader, writer, compute_cliff} on `core_range_cliff`. Reader/writer are shared
  across both WorkUnitSpecs (per-core RTAs routed by core coord); the per-group compute differs
  only in its `per_core_block_cnt` CTA (per-group-CTA pattern — NOT demoted to RTA). Groups
  guarded by `has_full`/`has_cliff` so an all-cliff or no-cliff split emits only the needed units.

## Kernels: ported vs forked (this factory)
- **reader** `reader_unary_stick_layout_split_rows_multicore.cpp` — **ported in place**
  (op-local, single user; same treatment as the single-core reader). CircularBuffer→
  DataflowBuffer; `c_0`→`dfb::src0`; `TensorAccessorArgs<3>`+addr RTA→`TensorAccessor(ta::input)`;
  CTs→named (`args::num_pages_in_row`, `args::size_of_valid_data_in_last_page_in_row`); RTAs→named
  (`args::num_rows`, `args::num_tiles_per_block`, `args::block_width_size`,
  `args::num_full_blocks_in_row`, `args::start_page_id`). Unused CT slot 0 (aligned_page_size)
  and RTA slots 2/6/7 dropped — the kernel never read them.
- **writer** — **REUSED** the single-core port's fork
  `device/kernels/dataflow/writer_unary_interleaved_start_id_m2.cpp` (identical logic; legacy
  multi-core writer RTAs `{num_tiles, start_id}` map 1:1 to the fork's `{num_pages, start_id}`).
- **compute** — **REUSED** the single-core port's fork `device/kernels/compute/tilize_m2.cpp`
  (identical CTAs `per_core_block_cnt`/`per_core_block_tile_cnt`; the multi-core legacy used the
  same shared `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` source).

## Dropped plumbing (this factory)
- Reader `src_addr` RTA(0)→`TensorBinding{input}`; unused RTAs 2/6/7 and CT0 dropped.
- Writer `dst_addr` RTA(0)→`TensorBinding{output}`; `output_cb_index` CT→`dfb::output`.
- All positional CTAs/RTAs→named args; all `TensorAccessorArgs` host plumbing removed.

## Device-op-class edits (this factory)
- `tilize_multi_core_default_program_factory.hpp`: `create_descriptor`→`create_program_spec`,
  return type `ProgramDescriptor`→`ttnn::device_operation::ProgramArtifacts`; include
  `ttnn/metal2_artifacts.hpp` + metal2 host-api headers instead of `program_descriptors.hpp`.
- No change to `tilize_device_operation.hpp/.cpp` — variant already lists the factory and
  `select_program_factory` already routes to it. nanobind: no hook for this factory.

## Blockers
- None. No op-owned tensors, no GlobalSemaphore, no per-mesh-coord variation, no live-allocator
  introspection, no raw `buffer->address()` threaded to a kernel. Clean port.

---

# Single-core factory (original port)

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
