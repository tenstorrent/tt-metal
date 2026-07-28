# Port Plan — `data_movement/fold`

Port plan for `ttnn/cpp/ttnn/operations/data_movement/fold`, ported from the `ProgramDescriptor` API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

**Recipe docs:** `b5b801a923d 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
**Audit docs (inherited):** `b5b801a923d 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

Single `DeviceOperation` (`Fold`) with two `descriptor`-concept factories, ported together:
- **`MultiCore`** — height-sharded row-major path (`fold_multi_core_program_factory.cpp`)
- **`MultiCoreDRAMFold`** — interleaved DRAM path (`fold_multi_core_dram_program_factory.cpp`); forks at runtime on input layout into two structurally distinct programs (**tiled** / **row-major**)

## Legacy Inventory

### Legacy factory shape
- Concept: **`ProgramDescriptorFactoryConcept`** (both factories expose `create_descriptor` returning `ProgramDescriptor`).
- Variants: two (`MultiCore`, `MultiCoreDRAMFold`); `MultiCoreDRAMFold` forks internally into tiled / row-major sub-programs.
- Custom `compute_program_hash`: **none** — already default reflection-based hash (confirmed: no override in the op dir).

*(Target Metal 2.0 concept `MetalV2FactoryConcept`, inherited from the audit — see TTNN ProgramFactory below.)*

### Variant: MultiCore (sharded, `fold_multi_core_program_factory.cpp`)

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs | config | opt_level |
|---|---|---|---|---|---|---|
| writer-inst | `writer_cb2s_row_major.cpp` | `all_cores` (shard grid) | 14 (see below), `is_reader`=1 | none | WriterConfigDescriptor | **Os** |
| reader-inst | `writer_cb2s_row_major.cpp` (same source) | `all_cores` | 14, `is_reader`=0 | none | ReaderConfigDescriptor | **Os** |

CTA order (0..13): `cb_src0_index`(c_0), `cb_dst0_index`(c_16), `pixel_size`, `aligned_pixel_size`, `aligned_dst_pixel_size`, `stride_w*aligned_pixel_size`, `width*aligned_pixel_size`, `stride_h`, `stride_w`, `num_dst_rows`, `width/stride_w`, `pixels_per_dst_row*aligned_pixel_size`, `element_size`, `is_reader`.

Dual-instance work-split: one kernel source, two `KernelDescriptor`s over the **same** `all_cores`, differing only by Reader/Writer config and the `is_reader` CTA (splits output columns: `cols_per_core = num_dst_cols/2`). Both instances raw-touch both CBs. No FIFO ops, no semaphores, no RTAs.

#### CBs
| index | total_size | core_ranges | data_format | page_size | buffer (borrowed) |
|---|---|---|---|---|---|
| c_0 (src0) | `num_pixels * aligned_pixel_size` | all_cores | input dtype | `aligned_pixel_size` | `src_buffer` (input) |
| c_16 (dst0) | `num_dst_pixels * aligned_dst_pixel_size` | all_cores | input dtype (`cb_data_format`) | `aligned_dst_pixel_size` | `dst_buffer` (output) |

Both CBs are **borrowed-memory** (`cb.buffer = ...`). `aligned_pixel_size = align(pixel_size, l1_alignment)`, `aligned_dst_pixel_size = align(dst_pixel_size, l1_alignment)`.

#### Tensor accessors
None. Both input and output reach the kernel through borrowed-memory CBs (raw `get_read_ptr`/`get_write_ptr`), not `TensorAccessor`.

#### Work split
`all_cores = input.shard_spec()->grid`; each core owns one shard. No `split_work_to_cores`.

### Variant: MultiCoreDRAMFold → tiled sub-program (`fold_multi_core_tiled_interleaved`)

#### Kernels
| unique_id | source | core_ranges | CTAs | RTAs | config |
|---|---|---|---|---|---|
| reader | `reader_dram2cb_tiled.cpp` | all_cores | `tiles_per_channel_dim`, `tiles_per_width_dim`, `src0_cb_index` + `TensorAccessorArgs(src)` | `src0_buffer`(Buffer*), `block_start_id`, `nblocks_per_core` | ReaderConfigDescriptor |
| writer | `writer_cb2dram_for_tiled_input.cpp` | all_cores | `input_width`, `stride_h`, `stride_w`, `stick_nbytes`, `aligned_stick_nbytes`, `tiles_per_channel_dim`, `tiles_per_width_dim`, `datum_size(out)`, `src1_cb_index` + `TensorAccessorArgs(dst)` | `dst_buffer`, `block_start_id`, `nblocks_per_core`, `patch_height_offset`, `output_offset` | WriterConfigDescriptor |
| compute | **donor** `untilize/device/kernels/compute/untilize.cpp` | `core_range` (full group) | `nblocks_per_core*tiles_per_width_dim`, `tiles_per_channel_dim`, `src0_cb_index`, `src1_cb_index` | none | ComputeConfigDescriptor{`fp32_dest_acc_en = (input==Float32)`} |
| compute-cliff (optional) | same donor | `core_range_cliff` | `nblocks_per_core_cliff*tiles_per_width_dim`, … | none | same |

#### CBs
| index | total_size | data_format | page_size |
|---|---|---|---|
| c_0 (src0) | `tiles_per_channel_dim * single_tile_size` | input dtype | `single_tile_size` |
| c_1 (src1) | `tiles_per_channel_dim * out_single_tile_size` | output dtype | `out_single_tile_size` |

#### Tensor accessors
`src` (input) → reader (Case 1); `dst` (output) → writer (Case 1). Delivered today as `Buffer*` BufferBindings in RTA slot 0.

#### Work split
`split_blocks_for_tilize(grid, num_blocks)` → `(ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff)`.

#### CB endpoints
`c_0`: reader(P) + compute(C) → legal 1:1. `c_1`: compute(P) + writer(C) → legal 1:1.

### Variant: MultiCoreDRAMFold → row-major sub-program (`fold_multi_core_row_major_interleaved`)

#### Kernels
| unique_id | source | core_ranges | CTAs (0..8 common) | RTAs | config |
|---|---|---|---|---|---|
| reader | `reader_dram2cb_for_rm_input.cpp` | all_cores | `stick_nbytes`, `cb_src0_index`, `aligned_stick_nbytes`, `stride_h`, `stride_w`, `input_width`, `patches_per_core`, `cb_src1_index`, `is_l1_aligned` + `TensorAccessorArgs(src)` | `src0_buffer`, `src_idx`, `src_col_offset` | ReaderConfigDescriptor |
| writer | `writer_cb2dram_for_rm_input.cpp` | all_cores | same 0..8 common + `TensorAccessorArgs(dst)` | `dst_buffer`, `dst_idx` | WriterConfigDescriptor |

#### CBs
| index | total_size | data_format | page_size | condition |
|---|---|---|---|---|
| c_0 (src0) | `2 * aligned_stick_nbytes * stride_w * stride_h` | input dtype | `aligned_stick_nbytes * stride_w * stride_h` | always |
| c_1 (src1) | `stick_nbytes * stride_w * stride_h` | input dtype | same | **only `!is_l1_aligned`** |

`is_l1_aligned = (stick_nbytes == aligned_stick_nbytes)`, `aligned_stick_nbytes = align(stick_nbytes, dram_alignment)`.

#### Tensor accessors
`src` → reader (Case 1); `dst` → writer (Case 1).

#### Work split
`patches_per_core = div_up(total_patches, num_cores_total)`; all grid cores.

#### CB endpoints
`c_0`: reader(P) + writer(C) → legal 1:1. `c_1`: writer-only raw `get_write_ptr` (sync-free, sole toucher) → **self-loop**; conditional (allocated only `!is_l1_aligned`).

### Cross-op kernels
- `untilize/device/kernels/compute/untilize.cpp` — donor compute kernel from the **untilize** op, file-path instantiated by the tiled sub-program ("borrowed" shared kernel). Cannot be Metal-2.0-ified in place without breaking untilize's still-legacy binders. **Census** (binders of the original, all still legacy): untilize `single_core` / `multi_core_parallelize_column` / `multi_core_sub_core_grids` / `multi_core_input_and_output_shard_type_and_shard_spec_identical`; untilize_with_unpadding `single_core` / `multi_core_interleaved` / `multi_core_sharded`; `pool/upsample`. No existing production `_metal2` fork beside the original at audit time → **rung 2: create the fork beside the original** as `untilize/device/kernels/compute/untilize_metal2.cpp` (+ pointer comment in the original), bind the tiled factory to it. Thin wrapper over `compute_kernel_lib::untilize` (kernel_lib, unchanged). Recorded in the port report under Handoff points / Open items.

### Flags
- `writer_cb2dram_for_rm_input.cpp:33` calls `cb_in1.get_write_ptr()` **unconditionally**, but `c_1` is allocated only when `!is_l1_aligned`. In Metal 2.0 a kernel cannot touch an unbound DFB — the touch must match the binding. **Resolution: `#ifdef`-gate (conditional binding)** — bind `c_1` and emit its use only when `!is_l1_aligned`, and gate the `get_write_ptr()` behind the same condition (dead in the aligned config anyway). See Applied Patterns.
- All five own dataflow kernels + the untilize compute donor + the pool `experimental_device_api.hpp` helper are Device 2.0 (audit GREEN). No unreferenced kernel files.

## TTNN ProgramFactory
- **Concept (inherited from audit):** `MetalV2FactoryConcept` (both factories).
- **Custom `compute_program_hash`:** none.
- **Implementation notes:** Both factories rename `create_descriptor` → `create_program_artifacts` (fixed signature: `operation_attributes_t`, `tensor_args_t`, `tensor_return_value_t&`). `select_program_factory`, `validate_*`, `compute_output_specs`, `create_output_tensors` are unchanged (out of port scope). No pybind `create_descriptor` to remove (`fold_nanobind.cpp` binds via `bind_function<"fold">`). No device-op-class edits forced beyond the factory bodies + headers.

## Planned Spec Shape

### Variant: MultiCore (sharded)
- **KernelSpecs (2):** `WRITER_INST` (WriterGen1 DM config, `is_reader`=1), `READER_INST` (ReaderGen1 DM config, `is_reader`=0). Same source, opt_level **Os**. 13 named CTAs each (drop the `is_reader` CTA — it becomes a per-instance literal value; the two magic CB-index CTAs become DFB bindings).
- **DataflowBufferSpecs (2):** `SRC0` (borrowed_from `INPUT`, entry `aligned_pixel_size`, num `num_pixels`), `DST0` (borrowed_from `OUTPUT`, entry `aligned_dst_pixel_size`, num `num_dst_pixels`). `data_format_metadata = cb_data_format`.
- **TensorParameters (2):** `INPUT` (input spec), `OUTPUT` (output spec). No `TensorBinding`s — a borrowed-memory DFB draws its backing L1 address from the paired `TensorArgument` via `borrowed_from` (`dataflow_buffer_spec.hpp`; migration guide — *Borrowed-memory DFBs*), which the spec validator accepts as the parameter's binding. The audit's causal-link gate already classified these bindings **clean** (borrowed-DFB, no `TensorAccessor`). Confirmed at runtime (tests pass with no `TensorBinding` on the borrowed tensors).
- **SemaphoreSpecs:** none.
- **WorkUnitSpecs (1):** `{WRITER_INST, READER_INST}` over `all_cores`.
- No RTAs → no `KernelRunArgs`.

### Variant: MultiCoreDRAMFold — tiled
- **KernelSpecs (3–4):** `READER`, `WRITER`, `COMPUTE` (over `core_range`), optional `COMPUTE_CLIFF` (over `core_range_cliff`). Named CTAs. Reader/Writer DM Gen1 configs; compute `ComputeGen1Config{.enable_32_bit_dest = fp32_dest_acc_en}` (+ `unpack_modes` entry for SRC0 when it consumes FP32 with 32-bit dest — see hw-config note).
- **DataflowBufferSpecs (2):** `SRC0` (entry `single_tile_size`, num `tiles_per_channel_dim`, fmt input), `SRC1` (entry `out_single_tile_size`, num `tiles_per_channel_dim`, fmt output).
- **TensorParameters (2):** `INPUT` (reader TensorBinding "src"), `OUTPUT` (writer TensorBinding "dst"). Both Case 1.
- **WorkUnitSpecs:** `WU_MAIN` `{READER, WRITER, COMPUTE}` over `core_range`; if cliff, `WU_CLIFF` `{READER, WRITER, COMPUTE_CLIFF}` over `core_range_cliff`. (Reader/Writer span all cores via membership in both WUs.)
- RTAs per-core: reader `{block_start_id, nblocks}`, writer `{block_start_id, nblocks, patch_height_offset, output_offset}`.

### Variant: MultiCoreDRAMFold — row-major
- **KernelSpecs (2):** `READER`, `WRITER`. Named CTAs. DM Gen1 configs.
- **DataflowBufferSpecs (1–2):** `SRC0` (always), `SRC1` (only `!is_l1_aligned`, conditional binding).
- **TensorParameters (2):** `INPUT` (reader "src"), `OUTPUT` (writer "dst"). Both Case 1.
- **WorkUnitSpecs (1):** `{READER, WRITER}` over `all_cores`.
- RTAs per-core: reader `{src_idx, src_col_offset}`, writer `{dst_idx}`.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| MultiCore: 2× `writer_cb2s_row_major.cpp` over one grid | `WRITER_INST`, `READER_INST` | one WU (both) | SRC0: WRITER_INST=CONSUMER, READER_INST=PRODUCER · DST0: WRITER_INST=PRODUCER, READER_INST=CONSUMER (1P+1C each; roles cosmetic — both raw-touch) |
| MultiCoreDRAMFold tiled: `compute` + optional `compute-cliff` (disjoint node sets) | `COMPUTE`, `COMPUTE_CLIFF` | `WU_MAIN` / `WU_CLIFF` | SRC0/SRC1 each bound by one compute per disjoint node set — ordinary 1:1, no assignment question |

The sharded pair is the **dual-instance work-split** (two-toucher 1P+1C, *not* multi-binding). The tiled compute+cliff pair is the disjoint-node work-split (each node sees one compute instance) — no flag either.

## Dropped Plumbing

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| sharded CTA[0]/[1] | `cb_src0_index` / `cb_dst0_index` magic indices | `DFBBinding` SRC0 / DST0 |
| sharded CTA[13] | `is_reader` positional | per-instance CTA value baked into each KernelSpec (still a named CTA `is_reader`, distinct value per instance) |
| sharded kernel | `experimental::CB cb(cb_id)` | `DataflowBuffer dfb(dfb::name)` |
| tiled reader RTA[0] / CTA `TensorAccessorArgs` | `src0_buffer` Buffer* + `TensorAccessorArgs(src)` CTA tail + kernel `TensorAccessorArgs<3>()` | `TensorParameter INPUT` / `TensorBinding "src"` → `TensorAccessor(tensor::src)` |
| tiled writer RTA[0] / CTA `TensorAccessorArgs` | `dst_buffer` Buffer* + `TensorAccessorArgs(dst)` | `TensorParameter OUTPUT` / `TensorBinding "dst"` → `TensorAccessor(tensor::dst)` |
| tiled reader CTA[2] `src0_cb_index` | magic index | `DFBBinding SRC0` |
| tiled writer CTA[8] `src1_cb_index` | magic index | `DFBBinding SRC1` |
| tiled compute CTA[2]/[3] | `src0_cb_index` / `src1_cb_index` | `DFBBinding SRC0` (consumer) / `SRC1` (producer) → `dfb::src` / `dfb::out` |
| RM reader RTA[0] / CTA `TensorAccessorArgs` | `src0_buffer` + `TensorAccessorArgs(src)` | `TensorParameter INPUT` / `TensorBinding "src"` |
| RM writer RTA[0] / CTA `TensorAccessorArgs` | `dst_buffer` + `TensorAccessorArgs(dst)` | `TensorParameter OUTPUT` / `TensorBinding "dst"` |
| RM reader CTA[1]/[7] | `cb_src0_index` / `cb_src1_index` | `DFBBinding SRC0` / conditional `SRC1` |
| all kernels | positional `get_compile_time_arg_val(N)` / `get_arg_val<uint32_t>(N)` | `get_arg(args::name)` |
| reader_dram2cb_tiled | `get_tile_size(cb_id_in0)` free fn | `dfb_in0.get_tile_size()` (rule 7) |

No page-size 3rd-argument CTAs (audit confirmed no 3rd arg anywhere). No semaphore-ID RTAs. No offset-folded pointers.

## Applied Patterns

- **[Two-toucher DFB → assign 1P+1C](port_patterns.md):** sharded SRC0 & DST0 — two same-source instances over one grid, both raw-touch each borrowed DFB; assign one PRODUCER + one CONSUMER per DFB. **Not** multi-binding.
- **[Borrowed-memory DFB]:** sharded SRC0/DST0 `borrowed_from` INPUT/OUTPUT; RM `c_1` is a plain (non-borrowed) L1 scratch.
- **[Sync-free / single-ended → self-loop DFB](port_patterns.md):** RM `SRC1` scratch — writer is the sole toucher (raw `get_write_ptr`), bind it PRODUCER + CONSUMER.
- **[Conditional / optional DFB bindings](port_patterns.md):** RM `SRC1` exists only when `!is_l1_aligned`; conditionally bind + emit a `NOT_L1_ALIGNED` define, and `#ifdef`-gate the kernel's `dfb::src1` alias and its `get_write_ptr()` use (which was unconditional in legacy — a latent no-op in the aligned config; gating makes the touch match the allocation).
- **[Multi-variant factory]:** `MultiCoreDRAMFold::create_program_artifacts` selects tiled vs row-major sub-program by input layout (runtime), same as legacy.
- **[Caution: Porting a shared kernel] (rung 2, borrowed):** donor `untilize.cpp` forked *beside the original* as `untilize/device/kernels/compute/untilize_metal2.cpp` (not copied into fold's tree); tiled factory binds the fork.

## Deferred / Flagged
- **Compute unpack_modes (tiled):** the untilize compute consumes SRC0. When input dtype is `Float32`, `fp32_dest_acc_en` (→ `enable_32_bit_dest`) is true and SRC0 carries a Float32 format → the validator requires an explicit `unpack_modes` entry for SRC0. Legacy set no unpack mode (default) → `UnpackMode::UnpackToSrc`. Add `{{SRC0, UnpackToSrc}}` gated on that condition. (SRC1 is produced, not consumed → no entry.)
- **No new structural blockers** surfaced during planning. The `c_1`-unconditional-`get_write_ptr` anomaly (flagged in the audit) is handled by the conditional-binding pattern above.
