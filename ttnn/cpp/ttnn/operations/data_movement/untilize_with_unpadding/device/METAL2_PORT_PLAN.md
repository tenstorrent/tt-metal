# Port Plan — untilize_with_unpadding (single-core factory)

Port plan for the `UntilizeWithUnpaddingSingleCoreProgramFactory`, ported from the
`ProgramDescriptorFactoryConcept` (`create_descriptor`) API to Metal 2.0
(`ProgramSpecFactoryConcept` / `create_program_spec`).

Only the **single-core** factory is ported. The device-operation's other five
factories (multi-core interleaved / sharded / col-interleaved / block-interleaved /
nd-sharded) remain on the legacy `ProgramDescriptorFactoryConcept` and continue to
build and run; the framework dispatches per-factory.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (`create_descriptor(params, const Tensor&, Tensor&)`).
- Variants: single (this factory). The device-op `program_factory_t` variant holds 6 factories total.
- Custom `compute_program_hash`: none — device-op uses the default reflection-based hash.

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | eltwise/unary/.../reader_unary_interleaved_start_id.cpp | single core | TensorAccessorArgs(src0) | src_addr, num_tiles, start_id(=0) | — | ReaderConfigDescriptor |
| writer | untilize_with_unpadding/.../writer_unary_unpad_dims_split_rows.cpp | single core | [0]=FLOAT32_DTYPE, [1]=unpadded_stick_size (UNUSED by kernel), TensorAccessorArgs(dst) | dst_addr + 14 dims | — | WriterConfigDescriptor |
| compute | untilize/.../compute/untilize.cpp | single core | per_core_block_cnt, per_core_block_tile_cnt, src_cb_id(0), out_cb_id(16) | — | DST_ACCUM_MODE (int32/uint32/fp32) | ComputeConfigDescriptor{fp32_dest_acc_en, unpack_to_dest_mode} |

### CBs
| index | total_size | core_ranges | data_format | page_size |
|---|---|---|---|---|
| 0 (src0) | num_tiles_per_block * input_single_tile_size | core | input fmt | input_single_tile_size |
| 16 (c_16, out) | num_tiles_per_block * output_single_tile_size | core | output fmt | output_single_tile_size |

### Semaphores
none

### Tensor accessors
| host site | originating Tensor | RTA slot (host) |
|---|---|---|
| reader_compile_time_args TensorAccessorArgs(*src0_buffer) | input | reader RTA 0 (src_addr) |
| writer_compile_time_args TensorAccessorArgs(*dst_buffer) | output | writer RTA 0 (dst_addr) |

### Work split
n/a — single core (core 0,0 or first core of sub_core_grids).

### Cross-op kernels
- `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` — shared by ~12 op directories. **FORKED** (see report).
- `data_movement/untilize/device/kernels/compute/untilize.cpp` — shared by ~9 op directories. **FORKED** (see report).
- Writer (`writer_unary_unpad_dims_split_rows.cpp`) lives in this op's dir and is used only by this factory — ported **in place**.

### Flags
- Writer host CTA slot 1 (`unpadded_stick_size`) is emitted by the legacy host but never read by the kernel (it reads only CTA slot 0 and `TensorAccessorArgs<2>()`). It is dropped in the port (no named CTA), which is behavior-preserving.
- Writer reads `num_unpadded_X` / `padded_X_size` RTAs but does not use them in the loop body; preserved as named RTAs to match legacy emission.

## TTNN ProgramFactory
- **Concept**: `MetalV2FactoryConcept` (`ProgramSpecFactoryConcept`), returning `ttnn::device_operation::ProgramArtifacts`.
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: `tensor_args_t = Tensor` (the single input), `tensor_return_value_t = Tensor` (the output). `create_program_spec(params, const Tensor& input, Tensor& output)`.

## Planned Spec Shape
- KernelSpecs: reader (DM READER), writer (DM WRITER), compute. 1:1 with legacy.
- DataflowBufferSpecs: IN (legacy CB 0), OUT (legacy CB 16). `data_format_metadata` set on both (consumed/produced by compute).
- SemaphoreSpecs: none.
- TensorParameters: SRC (input), DST (output).
- WorkUnitSpecs: one — {reader, writer, compute} on the single core. Satisfies the local-DFB rule (IN: reader PRODUCER → compute CONSUMER; OUT: compute PRODUCER → writer CONSUMER; all on the same WU/node).

## Preserved Multiplicity
none — no work-split multiplicity in legacy (single core).

## Dropped Plumbing
| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA 0 | `src0_buffer` address | `TensorBinding(SRC)` → `TensorAccessor(ta::src)` |
| reader CTA (TensorAccessorArgs) | `TensorAccessorArgs(*src0_buffer)` | binding mechanism |
| reader CTA implicit cb id 0 | `cb_id_in0 = 0` | `DFBBinding(IN, "in", PRODUCER)` → `dfb::in` |
| reader RTA 1,2 | num_tiles, start_id | named RTAs `num_pages`, `start_id` |
| writer RTA 0 | `dst_buffer` address | `TensorBinding(DST)` → `TensorAccessor(ta::dst)` |
| writer CTA (TensorAccessorArgs) | `TensorAccessorArgs(*dst_buffer)` | binding mechanism |
| writer CTA slot 0 | FLOAT32_DTYPE (a flag, not a CB id) | named CTA `float32_dtype` |
| writer CTA slot 1 | unpadded_stick_size | DROPPED (kernel never reads it) |
| writer magic cb id 16 | `cb_id_out0 = 16` | `DFBBinding(OUT, "out", CONSUMER)` → `dfb::out` |
| writer RTA 1..14 | positional dims | named RTAs |
| compute CTA 2,3 | src_cb_id(0), out_cb_id(16) | `DFBBinding(IN CONSUMER)`, `DFBBinding(OUT PRODUCER)` → `dfb::in`, `dfb::out` |
| compute CTA 0,1 | positional | named CTAs `per_core_block_cnt`, `per_core_block_tile_cnt` |

## Applied Patterns
- Pass DFB handles directly to LLKs / kernel-lib helpers: compute kernel passes `dfb::in` / `dfb::out` into `compute_kernel_hw_startup` and `compute_kernel_lib::untilize<...>` (template-parameter position via the constexpr implicit conversion).
- Caution: Modifying a shared dataflow kernel — reader and compute FORKED to `_m2` copies (see report).

## Deferred / Flagged
- New finding: writer's unused `unpadded_stick_size` CTA dropped; writer's unused `num_unpadded_X`/`padded_X_size` RTAs preserved (behavior-preserving). Flagged to op owner.
- The other 5 factories of this device-op remain on the legacy concept (out of scope for this single-factory port).

---

# Port Plan — untilize_with_unpadding (multi-core interleaved factory)

Port plan for `UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory`, ported from
`ProgramDescriptorFactoryConcept` (`create_descriptor`) to Metal 2.0
(`create_program_spec` → `ttnn::device_operation::ProgramArtifacts`). This is the
**default-selected** factory for non-sharded multicore untilize_with_unpadding.

## Legacy Inventory (multi-core)
### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | eltwise/unary/.../reader_unary_interleaved_start_id.cpp | all_cores | TensorAccessorArgs(src0) | src_addr, num_tiles_per_core, tile_start_id | — | ReaderConfigDescriptor |
| writer | untilize_with_unpadding/.../writer_unary_stick_layout_split_rows_multicore.cpp | all_cores | [0]=FLOAT32_DTYPE, [1]=unpadded_row_size_bytes, TensorAccessorArgs(dst) | dst_addr, padded_X_size, start_stick_id, n_block_reps, + variable-length 5-tuples (n_data,n_mixed,n_pads,times,repeat_count) | — | WriterConfigDescriptor |
| compute (full) | untilize/.../compute/untilize.cpp | core_range | nblocks_per_core, num_tiles_per_row, c_0, c_16 | — | DST_ACCUM_MODE (int32/uint32/fp32) | Compute{fp32_dest_acc_en, unpack_to_dest_mode} |
| compute (cliff) | untilize/.../compute/untilize.cpp | core_range_cliff | nblocks_per_core_cliff, num_tiles_per_row, c_0, c_16 | — | DST_ACCUM_MODE | Compute{...} |

### CBs
| index | total_size | core_ranges | data_format | page_size |
|---|---|---|---|---|
| 0 (src0) | num_tiles_per_row * input_single_tile_size | all_cores | input fmt | input_single_tile_size |
| 16 (out) | num_tiles_per_row * output_single_tile_size | all_cores | output fmt | output_single_tile_size |

### Work split
`split_blocks_for_tilize(available_grid, num_blocks)` →
`(ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff)`.
Reader/writer placed on `all_cores`; compute split into a full group (`core_range`,
`nblocks_per_core`) and a cliff group (`core_range_cliff`, `nblocks_per_core_cliff`).
`distribute_work(...)` produces per-core BlockRep assignments driving the writer RTAs.

### Cross-op / shared kernels
- reader `reader_unary_interleaved_start_id.cpp` — cross-op (~12 consumers). REUSES the existing
  `reader_unary_interleaved_start_id_m2.cpp` fork in this op dir (already created for single-core).
- compute `untilize/.../compute/untilize.cpp` — cross-op (~9 consumers). REUSES the existing
  `untilize_m2.cpp` fork in this op dir.
- writer `writer_unary_stick_layout_split_rows_multicore.cpp` — op-local, single consumer (this
  factory only). Ported **in place**.

## Planned Spec Shape (multi-core)
- KernelSpecs: reader (DM READER, all_cores), writer (DM WRITER, all_cores), compute_full,
  compute_cliff (1 KernelSpec per work-split group — preserves multiplicity, no CTA→RTA demotion).
- DataflowBufferSpecs: IN (legacy c_0), OUT (legacy c_16). num_entries = num_tiles_per_row.
- TensorParameters: SRC (input), DST (output).
- WorkUnitSpecs: uwu_full {reader, writer, compute_full} on core_range; uwu_cliff
  {reader, writer, compute_cliff} on core_range_cliff. Reader/writer appear in BOTH WUs
  (their target node set = core_range ∪ core_range_cliff = all_cores), satisfying the local-DFB
  rule (IN: reader PRODUCER → compute CONSUMER; OUT: compute PRODUCER → writer CONSUMER, same WU).

## Preserved Multiplicity (multi-core)
```
Legacy KernelDescriptors [compute(full), compute(cliff)] of source untilize.cpp
  → KernelSpecs [compute_full, compute_cliff] of same source (untilize_m2.cpp)
  → in WorkUnitSpecs [uwu_full, uwu_cliff]
  → sharing IN (CONSUMER) / OUT (PRODUCER) DFBs; reader/writer shared across both WUs.
```

## Dropped Plumbing (multi-core)
| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA 0 | src0_buffer address | TensorBinding(SRC) → ta::src |
| reader CTA TensorAccessorArgs | TensorAccessorArgs(*src0_buffer) | binding mechanism |
| reader implicit cb id 0 | cb_id_in0=0 | DFBBinding(IN PRODUCER) → dfb::in |
| reader RTA 1,2 | num_tiles_per_core, tile_start_id | named RTAs num_pages, start_id |
| writer RTA 0 | dst_buffer address | TensorBinding(DST) → ta::dst |
| writer CTA TensorAccessorArgs | TensorAccessorArgs(*dst_buffer) | binding mechanism |
| writer CTA 0 | FLOAT32_DTYPE | named CTA float32_dtype |
| writer CTA 1 | unpadded_row_size_bytes | named CTA unpadded_X_size (READ by kernel — unlike single-core writer) |
| writer magic cb id 16 | cb_id_out0=16 | DFBBinding(OUT CONSUMER) → dfb::out |
| writer RTA 1,2,3 | padded_X_size, start_stick_id, n_block_reps | named RTAs |
| writer RTA 4+ (variable) | run-length-compressed BlockRep 5-tuples | runtime varargs (get_vararg), per-node count |
| compute CTA 2,3 | c_0, c_16 | DFBBinding(IN CONSUMER), DFBBinding(OUT PRODUCER) → dfb::in, dfb::out |
| compute CTA 0,1 | nblocks_per_core, num_tiles_per_row | named CTAs per_core_block_cnt, per_core_block_tile_cnt |

## Applied Patterns (multi-core)
- **Anti-pattern avoided — Demoting per-group CTA to RTA**: full/cliff compute kept as two
  KernelSpecs in two WorkUnitSpecs (matmul-multicore exemplar shape), not one KernelSpec with the
  per-group block count demoted to an RTA.
- **Caution: shared dataflow kernel** — reader + compute REUSE the cross-op `_m2` forks created
  for single-core; writer ported in place (op-local).
- **Pass DFB handles directly** — compute feeds dfb::in/dfb::out into kernel-lib helpers.
- **Varargs (Caution: avoid varargs unless necessary)** — writer's per-core block-rep tuples are
  genuinely variable-length per node (kernel reads them in a loop bounded by the n_block_reps RTA),
  so varargs are the correct fit. Count varies per core → uses KernelAdvancedOptions
  `num_runtime_varargs_per_node` (see report Open items: this field is `[[deprecated]]`).

## Deferred / Flagged (multi-core)
- `num_runtime_varargs_per_node` is marked `[[deprecated]]` in advanced_options.hpp (slated for
  removal). It is the only API supporting per-node-VARYING vararg counts; the non-deprecated scalar
  `num_runtime_varargs` requires a uniform count across all nodes. Documented in the report.
- Remaining 4 factories (sharded / col-interleaved / block-interleaved / nd-sharded) stay legacy.
