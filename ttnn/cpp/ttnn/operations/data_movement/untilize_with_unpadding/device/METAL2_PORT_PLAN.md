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
