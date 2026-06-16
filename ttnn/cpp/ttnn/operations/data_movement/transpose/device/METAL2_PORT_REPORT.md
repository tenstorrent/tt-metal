# Metal 2.0 Port Report — transpose (CN factory)

## Summary

- **Status:** PORTED (one factory of eight).
- **Factory chosen:** `TransposeCNProgramFactory` — the simplest single-program factory in the
  transpose device-op (147-line legacy `.cpp`, two data-movement kernels, no compute kernel).
- **Concept:** legacy `ProgramDescriptorFactoryConcept` (`create_descriptor` → `ProgramDescriptor`)
  → Metal 2.0 `ProgramSpecFactoryConcept` (`create_program_spec` → `ProgramArtifacts`).

## Why CN

Factory `.cpp` line counts (all under `transpose/device/`):

| factory | lines |
|---|---|
| transpose_cn | 147 |
| transpose_wh_sharded | 167 |
| transpose_hc_rm | 202 |
| transpose_hc_tiled | 208 |
| transpose_hc_tiled_interleaved | 240 |
| transpose_wh_sharded_rm | 253 |
| transpose_wh | 343 |
| transpose_hc_sharded | 429 |

CN is the smallest and structurally the cleanest: one DFB, a reader + writer (no compute),
two tensor parameters, a single work-split with one WorkUnitSpec.

## Legacy inventory

- **Legacy factory shape:** `ProgramDescriptorFactoryConcept` (returned `tt::tt_metal::ProgramDescriptor`).
- **Custom `compute_program_hash`:** none on `TransposeDeviceOperation` — nothing to delete.
- **CBs:** 1 — `src0` (legacy index 0), `entry_size = stick_size`, `num_entries = 2`,
  `data_format = datatype_to_dataformat_converter(input.dtype())`.
- **Kernels:**
  - reader `kernels/dataflow/reader_unary_transpose_cn_interleaved_start_id.cpp`,
    `core_ranges = total_cores` (full grid), CTAs `{src0_cb_index, aligned_page_size, stick_size}`
    + `TensorAccessorArgs(src0_buffer, RuntimeTensorShape)`, define `CN_RM` when row-major,
    `ReaderConfigDescriptor`.
  - writer `kernels/dataflow/writer_unary_transpose_cn_interleaved_start_id.cpp`,
    `core_ranges = total_cores`, CTAs `{src0_cb_index, aligned_page_size, stick_size}`
    + `TensorAccessorArgs(dst_buffer, RuntimeTensorShape)`, define `CN_RM` when row-major,
    `WriterConfigDescriptor`.
- **Tensor accessors:** input `src0_buffer` (reader), output `dst_buffer` (writer) — both with
  `tensor_accessor::ArgConfig::RuntimeTensorShape`.
- **Work split:** `split_work_to_cores(compute_with_storage_grid_size, num_tensor_pages)`.
  Kernels launch on the full grid; no-op cores get `num_pages = 0`.
- **Cross-op kernels:** none. Both kernels are referenced ONLY by this factory
  (`grep -rl <kernel> ttnn/cpp/ttnn/operations` returns just this file), so they were ported in
  place (no fork needed).

## Spec shape (ported)

- **DataflowBufferSpec:** `src0` (entry_size = stick_size, num_entries = 2, format = input dtype).
- **KernelSpecs:** `reader` (PRODUCER of src0; tensor binding `input`) and `writer` (CONSUMER of
  src0; tensor binding `output`).
- **TensorParameters:** `input` (= `input_tensor.tensor_spec()`), `output` (= `output_tensor.tensor_spec()`).
- **WorkUnitSpec:** one, `{reader, writer}` on `total_cores` (the full grid) — the src0 DFB's
  producer and consumer must share the same WorkUnitSpec.

## Dropped plumbing

- **Buffer-address RTAs:** legacy reader RTA slot 0 (`src0_buffer`) and writer RTA slot 0
  (`dst_buffer`) → replaced by `TensorBinding`s (`input`, `output`). Kernel-side
  `TensorAccessor(src_args, src_addr)` → `TensorAccessor(ta::input)` / `(ta::output)`.
- **`TensorAccessorArgs` plumbing:** host `TensorAccessorArgs(...).append_to(cta, common_rta)` and
  kernel `TensorAccessorArgs<3>()` removed — the binding mechanism owns accessor config end-to-end.
- **Magic CB index in CTA:** `src0_cb_index` CTA → `dfb::src0` binding.
- **Positional CTAs/RTAs → named:** `page_size`, `read_size`/`write_size` (CTAs); reader RTAs
  `num_pages`, `start_id`, `hw`, `n` and CRTAs `N`, `C`, `HtWt`, `batch_step`, `channel_step`;
  writer RTAs `num_pages`, `start_id`.

## Faithful-mapping note (constant-per-core args → CRTA)

The legacy factory emitted `N`, `C`, `HtWt`, `batch_step`, `channel_step` as per-core runtime args
even though they are identical on every core. The port expresses them as **common** runtime args
(broadcast), matching the matmul exemplar's treatment of its uniform args. Values are identical;
only the dispatch channel (CRTA vs per-core RTA) changed. The kernel reads them via
`get_arg(args::N)` etc. regardless, so this is invisible to the kernel.

## Kernel edits (whitelist only)

Both kernels were already on Device 2.0 APIs (`Noc`, `CircularBuffer`, `TensorAccessor`). Edits:

1. Added `#include "experimental/kernel_args.h"`.
2. `CircularBuffer cb(cb_id_*)` → `DataflowBuffer cb(dfb::src0)`.
3. `TensorAccessor(src_args, addr)` → `TensorAccessor(ta::input)` / `(ta::output)`;
   removed the `TensorAccessorArgs<3>()` line and the buffer-address `get_arg_val` reads.
4. All positional `get_compile_time_arg_val` / `get_arg_val` → `get_arg(args::<name>)`.
5. `#ifdef CN_RM` blocks, loop bounds, and the `noc_async_read_sharded` / `noc_async_write_sharded`
   calls are **unchanged**; comments preserved verbatim.

## Open items / friction for downstream

1. **`RuntimeTensorShape` accessor config (the only real semantic question).** The legacy factory
   built both TensorAccessors with `tensor_accessor::ArgConfig::RuntimeTensorShape` — the tensor's
   shape is supplied as a runtime arg so one cached program serves inputs of varying shape. The
   Metal 2.0 `TensorParameter` binding bakes the shape into the accessor's compile-time args by
   default. `TensorParameterAdvancedOptions::dynamic_tensor_shape` exists and, per its header doc,
   makes "shape an implicit runtime argument" — but **only for sharded tensors** ("for an
   interleaved tensor, TensorAccessor configuration is unchanged"). CN is reached both for
   interleaved and (via the `CN_RM` sharded path) for sharded buffers. Two consequences to confirm
   on hardware:
   - For interleaved inputs this is behavior-neutral (shape doesn't affect the interleaved
     accessor's addressing), but it changes the **program-cache key**: the legacy cached program
     was shape-agnostic; the default Metal 2.0 binding will key on shape and rebuild for each new
     shape. This may regress cache-hit rate for callers that previously shared one entry across
     shapes. If that matters, set `dynamic_tensor_shape = true` on both TensorParameters.
   - For the `CN_RM` sharded path, `dynamic_tensor_shape = true` is likely **required** to
     reproduce the legacy runtime-shape accessor; left at the default the sharded accessor would be
     statically shaped. I left both TensorParameters at the default (no advanced option) to keep
     the port minimal and behavior-preserving for the common interleaved case, and flag the sharded
     case here rather than guessing. **Recommend the op/kernel owner confirm the intended
     accessor-shape semantics before relying on the CN sharded path under Metal 2.0.**
2. **No-op cores in the WorkUnitSpec.** The legacy factory launches both kernels on the full grid
   (`total_cores`) and gives no-op cores `num_pages = 0`. The port preserves this by targeting the
   WorkUnitSpec at `total_cores` and emitting RTAs for every core (including zero-page cores). If
   the framework prefers WorkUnitSpecs scoped to only the working cores (`all_cores`), that would
   be a behavior-equivalent simplification — left as-is to preserve exact legacy placement.

## Remaining factories (not ported this pass)

`TransposeWHProgramFactory`, `TransposeWHShardedProgramFactory`, `TransposeWHShardedRMProgramFactory`,
`TransposeHCTiledInterleavedProgramFactory`, `TransposeHCTiledProgramFactory`,
`TransposeHCRMProgramFactory`, `TransposeHCShardedProgramFactory` remain on
`ProgramDescriptorFactoryConcept`. The op builds and runs with the variant mixed (per-factory
dispatch).

## Build / test

Not built, not run (worktree has no build dir, per instructions).
