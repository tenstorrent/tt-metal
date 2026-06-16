# Port Plan — reshard (NdReshardCopyPagesFactory)

Port plan for the `NdReshardCopyPagesFactory` variant of `reshard`, ported from the
`ProgramDescriptorFactoryConcept` (`create_descriptor`) to Metal 2.0
(`ProgramSpecFactoryConcept` / `create_program_spec`).

Scope note: `reshard`'s `program_factory_t` holds **eight** factory variants. Per the
recipe's atomic-unit rule, this port targets **one** factory — the simplest single-program
one, `NdReshardCopyPagesFactory` (DRAM→DRAM ND reshard, 124-line factory, one CB, two DM
kernels). The other seven stay on the legacy concept; the op continues to build and run
(framework dispatches per-factory).

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (`create_descriptor` returns `ProgramDescriptor`).
- Variants (in `program_factory_t`): ReshardSameWidthFactory<true/false>, ReshardSameHeightFactory<true/false>,
  ReshardGenericFactory, **NdReshardCopyPagesFactory** (this port), NdReshardCopyLocalShardFactory<true/false>.
- Custom `compute_program_hash`: none — `ReshardDeviceOperation` uses the default reflection-based hash.

### Kernels (NdReshardCopyPagesFactory)
| unique_id | source | core_ranges | CTAs (positional) | RTAs | CRTAs | config |
|---|---|---|---|---|---|---|
| reader | nd_reshard_copy_pages_reader.cpp | full grid | input TensorAccessorArgs..., cb_in0_idx(=c_0), aligned_page_size | start_page, end_page | input_buffer addr (CommonBufferBinding) | ReaderConfigDescriptor |
| writer | nd_reshard_copy_pages_writer.cpp | full grid | output TensorAccessorArgs..., cb_in0_idx(=c_0), aligned_page_size | start_page, end_page | output_buffer addr (CommonBufferBinding) | WriterConfigDescriptor |

### CBs
| index | total_size | core_ranges | data_format | page_size |
|---|---|---|---|---|
| c_0 | aligned_page_size * 1 | full grid | input dtype | aligned_page_size |

Single-entry CB (num_tiles_in_cb = 1). Real producer (reader) / consumer (writer) FIFO — not a fake CB.

### Semaphores
none

### Tensor accessors
| host site | originating Tensor | RTA/CRTA slot | kernel CTA chain |
|---|---|---|---|
| `TensorAccessorArgs(*input_buffer)` | input | base addr via CRTA slot 0 | `TensorAccessorArgs<0,0>()` in reader |
| `TensorAccessorArgs(*output_buffer)` | output | base addr via CRTA slot 0 | `TensorAccessorArgs<0,0>()` in writer |

Both are plain interleaved-style page accessors (`.pages(start,end)`) over ND-sharded DRAM buffers.
Case-1 bindings: standard page access, no exotic address arithmetic.

### Work split
- Driver: manual page split — `num_dev_pages = tensor_shape_in_pages().volume()`; `n_pages_per_core = num_dev_pages / cores.size()`,
  with the remainder distributed one-page-per-core to the first `remainder` cores.
- num_cores: all cores in the full compute grid (`corerange_to_cores`, orientation from `nd_shard_spec.orientation`).
- Single core group (per-core page count varies by +1 for the remainder cores, expressed via RTAs not CTAs — correct, page counts are data not loop-unroll dims).

### Cross-op kernels
none — both kernels live in this op's `device/kernels/` and are used only by this factory
(`grep -rl nd_reshard_copy_pages_{reader,writer} ttnn/cpp/ttnn/operations` → only this factory).
Ported in place.

### Flags
- The other 7 factory variants are untouched and remain on the legacy concept.

## TTNN ProgramFactory
- **Concept**: `MetalV2FactoryConcept` (`ProgramSpecFactoryConcept`) — `create_program_spec` returning `ttnn::device_operation::ProgramArtifacts`.
- **Custom `compute_program_hash`**: none to delete.
- **Implementation notes**: `select_program_factory` already returns `NdReshardCopyPagesFactory{}`; the variant entry is unchanged. The framework's `ProgramSpecMeshWorkloadFactoryAdapter` routes it automatically once the factory exposes `create_program_spec` instead of `create_descriptor`.

## Planned Spec Shape
- KernelSpecs: 2 — `reader` (DM, role READER, DFB PRODUCER), `writer` (DM, role WRITER, DFB CONSUMER).
- DataflowBufferSpecs: 1 — `cb_in0` (entry_size = aligned_page_size, num_entries = 1, data_format = input dtype).
- SemaphoreSpecs: none.
- TensorParameters: 2 — `input` (input.tensor_spec()), `output` (output.tensor_spec()).
- WorkUnitSpecs: 1 — `{reader, writer}` on the full grid (local-DFB rule: producer + consumer share the WorkUnitSpec).

## Preserved Multiplicity
none — single core group; the per-core page-count variation is a runtime value (RTA), never a per-group CTA.

## Dropped Plumbing
| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader CRTA slot 0 | `emplace_common_runtime_args({input_buffer})` (base addr) | `TensorBinding`(input) via `ta::input` |
| writer CRTA slot 0 | `emplace_common_runtime_args({output_buffer})` (base addr) | `TensorBinding`(output) via `ta::output` |
| reader/writer CTA: `input/output_accessor_args.get_compile_time_args()` + `TensorAccessorArgs<0,0>()` chain | positional TensorAccessorArgs plumbing | `TensorBinding` end-to-end (auto-injected CTA/CRTA) |
| reader/writer CTA: `cb_in0_idx` (= c_0) | magic CB index in CTA | `DFBBinding`(cb_in0) → `dfb::cb_in0` |
| reader/writer CTA: `aligned_page_size` | positional CTA | named CTA `page_size` |
| reader/writer RTA: positional slot 0/1 (`start_page`/`end_page`) | positional RTA | named RTAs `start_page`, `end_page` |

## Applied Patterns
- Plain reader-produces / writer-consumes DFB — standard local DFB, both endpoints in one WorkUnitSpec (matmul exemplar shape).
- No self-loop, no conditional binding, no aliasing.

## Deferred / Flagged
- No audit (`METAL2_PREPORT_AUDIT.md`) was present; this port was performed directly per invoker instruction.
  The TensorAccessor bindings were self-classified Case-1 (standard page access) — no user override on file.
- The other 7 reshard factory variants remain to be ported; several involve L1 shard-spec geometry
  (block/width-sharded shard grids) and may surface shard-shape handling not present in this simplest variant.
