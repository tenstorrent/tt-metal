# Port Plan — `reduction/generic`

Port plan for the four reduction-family program factories, ported from `ProgramDescriptor` to Metal 2.0:

- `ReduceMultiCoreHProgramFactory` (`reduce_op_multi_core_h_program_factory.cpp`)
- `ReduceMultiCoreWProgramFactory` (`reduce_op_multi_core_w_program_factory.cpp`)
- `ReduceSingleCoreHwProgramFactory` (`reduce_op_single_core_hw_program_factory.cpp`)
- `WelfordReduceProgramFactory` (`welford_reduce_program_factory.cpp`)

The two device-operations (`ReduceDeviceOperation` and `WelfordReduceDeviceOperation`) are co-located and share kernel sources; the port treats them as one bundle. Welford is multi-variant (W/H/HW); the Reduce factories are each single-variant per file but Reduce-W and Reduce-H carry a `negate` and `use_post_mul` configuration option that conditionally adds DFBs.

## Legacy Inventory

### Factory shape
- Concept: `ProgramDescriptorFactoryConcept` (each factory has `create_descriptor(...)` returning `tt::tt_metal::ProgramDescriptor`).
- Variants:
  - `ReduceMultiCoreHProgramFactory` — single-variant per factory; runtime configuration: `use_width_sharding`, `negate`, `use_post_mul`.
  - `ReduceMultiCoreWProgramFactory` — single-variant per factory; runtime configuration: `negate`, `use_post_mul`.
  - `ReduceSingleCoreHwProgramFactory` — single-variant per factory; runtime configuration: `negate`, `use_post_mul`.
  - `WelfordReduceProgramFactory` — **multi-variant** (W / H / HW selected from `attrs.reduce_dim`).

### Cross-op kernels

Two cross-op writer kernels referenced from outside the op directory:

- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — used by W, single-core HW, the interleaved branch of H, and the W + H variants of Welford. Many other ops also use this writer.
- `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` — used by the width-sharded branch of H. Many other ops also use this writer.

Both fork to `_metal2`-suffixed copies in their respective sibling directories, per the [Caution: Modifying a shared dataflow kernel](../../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#caution-modifying-a-shared-dataflow-kernel) entry — bulk-port-window strategy. Sunset is when the last unmigrated consumer ports.

### Flags

- The Welford HW writer (`writer_welford_hw.cpp`) is op-local — modified in place.
- The five op-local dataflow / compute kernels all have small `get_tile_size(cb_id)` holdovers that get swapped to `cb_obj.get_tile_size()` form during the port (port-time cleanup).
- All compute kernels source CB ids via `tt::CBIndex::c_N` constants and construct `CircularBuffer cb_NAME_obj(cb_NAME)` wrappers. The port swaps each to `DataflowBuffer cb_NAME_obj(dfb::name)` form.

---

### Variant: Reduce Multi-Core H

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader (interleaved) | `device/kernels/dataflow/reader_unary_transpose_wh_universal_input_cols_partitioned.cpp` | `all_cores` | `{Ht, Wt, HtWt, scaler_bits, /*use_welford=*/0}` + `TensorAccessorArgs(input).append_to(...)` | per-core: `{input.buffer(), col_start, curr_col_in_batch, num_cols_per_core}` | from `reduce_op_utils::get_defines(math_op, H)` plus `ENABLE_FP32_DEST_ACC`, `DST_SYNC_FULL`, optionally `REDUCE_POST_MUL` | `ReaderConfigDescriptor{}` |
| reader (width-sharded) | `device/kernels/dataflow/reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp` | `all_cores` | `{src0_cb_index, src1_cb_index, scaler_cb_index, scaler_bits}` | per-core: `{num_tiles, shard_Wt, Ht, NC, shard_row_size, shard_batch_size}` | base + `REDUCE_SCALER`, `ENABLE_FP32_DEST_ACC`, `DST_SYNC_FULL` | `ReaderConfigDescriptor{}` |
| writer (interleaved) | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (cross-op; forked) | `all_cores` | `{output_cb_index}` + `TensorAccessorArgs(output).append_to(...)` | per-core: `{output.buffer(), num_cols_per_core, num_cols_read}` | (none) | `WriterConfigDescriptor{}` |
| writer (width-sharded) | `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` (cross-op; forked) | `all_cores` | `{output_cb_index}` | per-core: `{num_cols_per_core_group_1}` | (none) | `WriterConfigDescriptor{}` |
| compute_g1 | `device/kernels/compute/reduce.cpp` or `reduce_h_neg.cpp` | `core_group_1` | `{Ht, compute_Wt_g1, compute_NC, post_mul_scaler_bits}` | (none) | `reduce_defines` (+ `REDUCE_POST_MUL` if `use_post_mul`) | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en}` |
| compute_g2 | same as g1 | `core_group_2` (if non-empty) | `{Ht, compute_Wt_g2, compute_NC, post_mul_scaler_bits}` | (none) | same as g1 | same as g1 |

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) | borrowed? |
|---|---|---|---|---|---|---|
| `CBIndex::c_0` (src0) | `num_input_tiles * src0_single_tile_size` | `all_cores` | `src0_cb_data_format` (input dtype) | `src0_single_tile_size` | — | no |
| `CBIndex::c_1` (src1, only width-sharded) | `num_shard_tiles * src0_single_tile_size` | `all_cores` | `src0_cb_data_format` | `src0_single_tile_size` | — | **yes** (`a.buffer()`) |
| `CBIndex::c_2` (scaler) | `scaler_single_tile_size` | `all_cores` | `scaler_cb_data_format` (FP32 or Float16_b) | `scaler_single_tile_size` | — | no |
| `CBIndex::c_3` (output) | `num_output_tiles * dst_single_tile_size` (width-sharded: shard size); interleaved: `2 * dst_single_tile_size` (or `chunk_size * dst_single_tile_size` if negate) | `all_cores` | `dst_cb_data_format` | `dst_single_tile_size` | — | width-sharded path: **yes** (`output.buffer()`); interleaved: no |
| `CBIndex::c_4` (acc, only negate) | `Ht * lcm(Wt_g1, Wt_g2) * dst_single_tile_size` | `all_cores` | `dst_cb_data_format` | `dst_single_tile_size` | — | no |
| `CBIndex::c_5` (ineg, only negate) | same as c_4 | `all_cores` | `dst_cb_data_format` | `dst_single_tile_size` | — | no |

`num_input_tiles` is `chunk_size` when negate else `2`. `chunk_size` is `1` for width-sharded and `ttnn::get_dest_reg_count(...)` otherwise.

#### Semaphores

None.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) | CTA offset (kernel) |
|---|---|---|---|
| reader_h:283 (`TensorAccessorArgs(*src0_buffer).append_to(...)`) (interleaved branch) | input `a` | RTA 0: `a.buffer()` (address) | CTA 5 onward (5 preceding CTAs) |
| reader_h:391 (no TA used — sharded reads via borrowed-memory CB) | input `a` | n/a | n/a |
| writer_h:314 (`TensorAccessorArgs(*dst_buffer).append_to(...)`) (interleaved branch) | output | RTA 0: `output.buffer()` | CTA 1 onward |
| writer_h:307 (no TA used — sharded writes via borrowed-memory CB) | output | n/a | n/a |

#### Work split

- Driver: `split_work_to_cores(grid, NC * Wt)` (interleaved); `all_cores = a.shard_spec().value().grid` (width-sharded).
- num_cores: dispatch-time
- core_group_1, num_cols_per_core_group_1
- core_group_2, num_cols_per_core_group_2 (may be empty)
- Width-sharded: `core_group_1 = all_cores`, `core_group_2 = empty`.

---

### Variant: Reduce Multi-Core W

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_unary_reduce_universal_start_id.cpp` | `all_cores` | `{scaler_bits}` + `TensorAccessorArgs(input).append_to(...)` | per-core: `{a.buffer(), num_tensor_tiles_per_core, num_tiles_read}` | from `reduce_op_utils::get_defines(math_op, W)` (+ `REDUCE_POST_MUL` if `use_post_mul`) | `ReaderConfigDescriptor{}` |
| writer | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (cross-op; forked) | `all_cores` | `{output_cb_index}` + `TensorAccessorArgs(output).append_to(...)` | per-core: `{output.buffer(), num_tiles_per_core_output, num_tiles_read_output}` | reduce_defines | `WriterConfigDescriptor{}` |
| compute_g1 | `device/kernels/compute/reduce.cpp` or `reduce_w_neg.cpp` | `core_group_1` | `{num_rows_per_core_group_1, Wt, 1, post_mul_scaler_bits}` | (none) | reduce_defines | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en}` |
| compute_g2 | same as g1 | `core_group_2` (if non-empty) | `{num_rows_per_core_group_2, Wt, 1, post_mul_scaler_bits}` | (none) | same as g1 | same as g1 |

#### CBs

| index | total_size | core_ranges | data_format | page_size | borrowed? |
|---|---|---|---|---|---|
| `0` (src0) | `2 * src0_single_tile_size` | `all_cores` | `src0_cb_data_format` | `src0_single_tile_size` | no |
| `CBIndex::c_2` (scaler) | `scaler_single_tile_size` | `all_cores` | `scaler_cb_data_format` | `scaler_single_tile_size` | no |
| `CBIndex::c_3` (output) | `2 * dst_single_tile_size` | `all_cores` | `dst_cb_data_format` | `dst_single_tile_size` | no |
| `CBIndex::c_4` (acc, only negate) | `1 * dst_single_tile_size` | `all_cores` | `dst_cb_data_format` | `dst_single_tile_size` | no |
| `CBIndex::c_5` (ineg, only negate) | `1 * dst_single_tile_size` | `all_cores` | `dst_cb_data_format` | `dst_single_tile_size` | no |

#### Semaphores

None.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) | CTA offset (kernel) |
|---|---|---|---|
| reader_w:107 (`TensorAccessorArgs(*src_buffer).append_to(...)`) | input `a` | RTA 0: `a.buffer()` | CTA 1 onward (1 preceding CTA: `scaler_bits`) |
| reader_w:110 (`TensorAccessorArgs(*dst_buffer).append_to(...)`) | output | RTA 0: `output.buffer()` | CTA 1 onward |

#### Work split

- Driver: `split_work_to_cores(grid, NC * Ht)`.
- num_cores: dispatch-time
- core_group_1, num_rows_per_core_group_1
- core_group_2, num_rows_per_core_group_2 (may be empty)

---

### Variant: Reduce Single-Core HW

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_unary_reduce_universal_start_id.cpp` | one core (`selected_core_coord`) | `{sqrt(scaler)_bits}` + `TensorAccessorArgs(input).append_to(...)` | `{a.buffer(), num_tensor_tiles, 0u}` | `reduce_op_utils::get_defines(math_op, HW)` (+ `REDUCE_POST_MUL`) | `ReaderConfigDescriptor{}` |
| writer | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (cross-op; forked) | one core | `{output_cb_index}` + `TensorAccessorArgs(output).append_to(...)` | `{output.buffer(), num_tensor_tiles / (Ht*Wt), 0u}` | (none) | `WriterConfigDescriptor{}` |
| compute | `device/kernels/compute/reduce.cpp` or `reduce_hw_neg.cpp` | one core | `{Ht, Wt, NC, post_mul_scaler_bits}` | (none) | reduce_defines | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en}` |

#### CBs

| index | total_size | core_ranges | data_format | page_size | borrowed? |
|---|---|---|---|---|---|
| `0` (src0) | `2 * src0_single_tile_size` | one core | `src0_cb_data_format` | `src0_single_tile_size` | no |
| `CBIndex::c_2` (scaler) | `scaler_single_tile_size` | one core | `scaler_cb_data_format` | `scaler_single_tile_size` | no |
| `CBIndex::c_3` (output) | `2 * dst_single_tile_size` | one core | `dst_cb_data_format` | `dst_single_tile_size` | no |
| `CBIndex::c_4` (acc, only negate) | `1 * dst_single_tile_size` | one core | `dst_cb_data_format` | `dst_single_tile_size` | no |
| `CBIndex::c_5` (ineg, only negate) | `1 * dst_single_tile_size` | one core | `dst_cb_data_format` | `dst_single_tile_size` | no |

#### Semaphores

None.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) | CTA offset (kernel) |
|---|---|---|---|
| reader_hw:131 (`TensorAccessorArgs(*src0_buffer).append_to(...)`) | input `a` | RTA 0: `a.buffer()` | CTA 1 onward |
| reader_hw:160 (`TensorAccessorArgs(*dst_buffer).append_to(...)`) | output | RTA 0: `output.buffer()` | CTA 1 onward |

#### Work split

- n/a — single core.

---

### Variant: Welford W

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_unary_reduce_universal_start_id.cpp` | `all_cores` | `{scaler_bits}` + `TensorAccessorArgs(input).append_to(...)` | per-core: `{input.buffer(), num_input_tiles, input_tiles_offset}` | from `reduce_op_utils::get_defines(math_op, W)` + `ENABLE_FP32_DEST_ACC`, `DST_SYNC_FULL` | `ReaderConfigDescriptor{}` |
| writer | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (cross-op; forked) | `all_cores` | `{output_cb_index}` + `TensorAccessorArgs(output).append_to(...)` | per-core: `{output.buffer(), num_output_tiles, output_tiles_offset}` | reduce_defines | `WriterConfigDescriptor{}` |
| compute_g1 | `device/kernels/compute/welford_reduce_w.cpp` | `core_group_1` | `{Wt, W, tile_width, do_scale, correction, is_std}` | per-core: `{num_work_units_per_core_g1}` (NCHt) | reduce_defines | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en}` |
| compute_g2 | same as g1 | `core_group_2` (if non-empty) | same as g1 | per-core: `{num_work_units_per_core_g2}` | same as g1 | same as g1 |

#### CBs

| index | total_size | core_ranges | data_format | page_size | borrowed? |
|---|---|---|---|---|---|
| `CBIndex::c_0` (in) | `2 * input_single_tile_size` | `all_cores` | input dtype | `input_single_tile_size` | no |
| `CBIndex::c_2` (scalar) | `scalar_single_tile_size` (BFloat16) | `all_cores` | Float16_b | `scalar_single_tile_size` | no |
| `CBIndex::c_16` (out) | `2 * dst_single_tile_size` | `all_cores` | output dtype | `dst_single_tile_size` | no |
| `CBIndex::c_19` (var scratch) | `1 * Float32_or_Float16_b tile_size` | `all_cores` | Float32 if fp32_dest_acc else Float16_b | per data format | no |
| `CBIndex::c_20` (scaled, only `do_scale`) | `1 * input_single_tile_size` | `all_cores` | input dtype | `input_single_tile_size` | no |

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) | CTA offset (kernel) |
|---|---|---|---|
| welford_reduce_program_factory.cpp (reader W branch) | input | RTA 0: `tensor_arg.buffer()` | CTA 1 onward |
| welford_reduce_program_factory.cpp (writer) | output | RTA 0: `tensor_return_value.buffer()` | CTA 1 onward |

#### Work split

- Driver: `split_work_to_cores(grid, NC * Ht)`.
- num_work_units = `NC * Ht`
- num_work_units_per_core_group_1 / _2

---

### Variant: Welford H

#### Kernels

Same shape as Welford W, with reader = `reader_unary_transpose_wh_universal_input_cols_partitioned.cpp` and compute = `welford_reduce_h.cpp`.

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_unary_transpose_wh_universal_input_cols_partitioned.cpp` | `all_cores` | `{Ht, Wt, HtWt, scaler_bits, /*use_welford=*/1}` + `TensorAccessorArgs(input).append_to(...)` | per-core: `{input.buffer(), col_start_tile_id, curr_col_in_batch, num_cols_per_core}` | reduce_defines + `ENABLE_FP32_DEST_ACC` + `DST_SYNC_FULL` | `ReaderConfigDescriptor{}` |
| writer | `writer_unary_interleaved_start_id.cpp` (cross-op; forked) | `all_cores` | `{output_cb_index}` + `TensorAccessorArgs(output).append_to(...)` | per-core: `{output.buffer(), num_cols_per_core, num_cols_read}` | reduce_defines | `WriterConfigDescriptor{}` |
| compute_g1 | `device/kernels/compute/welford_reduce_h.cpp` | `core_group_1` | `{Ht, H, tile_height, do_scale, correction, is_std}` | per-core: `{num_work_units_per_core_g1}` (NCWt) | reduce_defines | `ComputeConfigDescriptor` |
| compute_g2 | same as g1 | `core_group_2` (if non-empty) | same as g1 | per-core: `{num_work_units_per_core_g2}` | same as g1 | same as g1 |

#### CBs

Same as Welford W, **minus** `cb_var` (c_19) and `cb_scaled` (c_20). H-reduce doesn't need the transpose-scratch nor the scaled intermediate (transpose is unneeded for H).

| index | total_size | core_ranges | data_format | page_size | borrowed? |
|---|---|---|---|---|---|
| `CBIndex::c_0` (in) | `2 * input_single_tile_size` | `all_cores` | input dtype | `input_single_tile_size` | no |
| `CBIndex::c_2` (scalar) | `scalar_single_tile_size` | `all_cores` | Float16_b | `scalar_single_tile_size` | no |
| `CBIndex::c_16` (out) | `2 * dst_single_tile_size` | `all_cores` | output dtype | `dst_single_tile_size` | no |

---

### Variant: Welford HW

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_unary_transpose_wh_universal_input_cols_partitioned.cpp` | `all_cores` | `{Ht, Wt, HtWt, scaler_bits, /*use_welford=*/1}` + `TensorAccessorArgs(input).append_to(...)` | per-core: `{input.buffer(), col_start_tile_id, /*curr_col_in_batch=*/0u, num_cols}` | reduce_defines + `ENABLE_FP32_DEST_ACC` + `DST_SYNC_FULL` | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/dataflow/writer_welford_hw.cpp` (op-local) | `all_cores` | `{Wt, W, tile_width, H, correction, reduce_batch_size}` + `TensorAccessorArgs(output).append_to(...)` | per-core: `{output.buffer(), nc_slices_per_core, output_offset}` | (none) | `WriterConfigDescriptor{}` |
| compute_g1 | `device/kernels/compute/welford_reduce_hw.cpp` | `core_group_1` | `{Ht, H, tile_height, Wt, do_scale, reduce_batch_size, is_std}` | per-core: `{nc_slices_per_core_g1}` | reduce_defines | `ComputeConfigDescriptor` |
| compute_g2 | same as g1 | `core_group_2` (if non-empty) | same as g1 | per-core: `{nc_slices_per_core_g2}` | same as g1 | same as g1 |

#### CBs

Same as Welford H, **plus** `cb_partial` (c_21, Float32 mean+var pairs from compute) and `cb_combined` (c_22, Float32 scalar from writer).

| index | total_size | core_ranges | data_format | page_size | borrowed? |
|---|---|---|---|---|---|
| `CBIndex::c_0` (in) | `2 * input_single_tile_size` | `all_cores` | input dtype | `input_single_tile_size` | no |
| `CBIndex::c_2` (scalar) | `scalar_single_tile_size` | `all_cores` | Float16_b | `scalar_single_tile_size` | no |
| `CBIndex::c_16` (out) | `2 * dst_single_tile_size` | `all_cores` | output dtype | `dst_single_tile_size` | no |
| `CBIndex::c_21` (partial) | `4 * Float32 tile_size` | `all_cores` | Float32 | Float32 tile_size | no |
| `CBIndex::c_22` (combined) | `1 * Float32 tile_size` | `all_cores` | Float32 | Float32 tile_size | no |

---

## Planned Spec Shape

### Reduce H

- KernelSpecs: `reader`, `writer`, `compute_g1`, optional `compute_g2`. Variants: interleaved vs. width-sharded — selected inside the helper function for this factory.
- DataflowBufferSpecs:
  - `in_dfb` (c_0)
  - `scaler_dfb` (c_2)
  - `out_dfb` (c_3)
  - Width-sharded only: `in_shard_dfb` (c_1, `borrowed_from = "input"`), `out_dfb` is `borrowed_from = "output"`.
  - Negate only: `acc_dfb` (c_4), `ineg_dfb` (c_5).
- SemaphoreSpecs: none.
- TensorParameters: `input`, `output`. (Always declared, even on width-sharded path — needed for borrowed_from.)
- WorkUnitSpecs:
  - `wu_g1`: `{reader, writer, compute_g1}` on `core_group_1`.
  - `wu_g2` (if g2 non-empty): `{reader, writer, compute_g2}` on `core_group_2`.
  - Width-sharded: single `wu` on `all_cores` (`core_group_1 = all_cores`).

### Reduce W

- KernelSpecs: `reader`, `writer`, `compute_g1`, optional `compute_g2`.
- DataflowBufferSpecs: `in_dfb` (c_0), `scaler_dfb` (c_2), `out_dfb` (c_3). Negate adds `acc_dfb` (c_4), `ineg_dfb` (c_5).
- SemaphoreSpecs: none.
- TensorParameters: `input`, `output`.
- WorkUnitSpecs: `wu_g1`, optional `wu_g2`. Reader + writer share both WUs (so they get bound to both groups via shared kernel listing).

### Reduce HW (single-core)

- KernelSpecs: `reader`, `writer`, `compute`.
- DataflowBufferSpecs: `in_dfb` (c_0), `scaler_dfb` (c_2), `out_dfb` (c_3). Negate adds `acc_dfb` (c_4), `ineg_dfb` (c_5).
- SemaphoreSpecs: none.
- TensorParameters: `input`, `output`.
- WorkUnitSpecs: `wu` (single core).

### Welford W

- KernelSpecs: `reader`, `writer`, `compute_g1`, optional `compute_g2`.
- DataflowBufferSpecs: `in_dfb` (c_0), `scalar_dfb` (c_2), `out_dfb` (c_16), `var_dfb` (c_19). `do_scale` adds `scaled_dfb` (c_20).
- SemaphoreSpecs: none.
- TensorParameters: `input`, `output`.
- WorkUnitSpecs: `wu_g1`, optional `wu_g2`.

### Welford H

- KernelSpecs: `reader`, `writer`, `compute_g1`, optional `compute_g2`.
- DataflowBufferSpecs: `in_dfb` (c_0), `scalar_dfb` (c_2), `out_dfb` (c_16).
- SemaphoreSpecs: none.
- TensorParameters: `input`, `output`.
- WorkUnitSpecs: `wu_g1`, optional `wu_g2`.

### Welford HW

- KernelSpecs: `reader`, `writer`, `compute_g1`, optional `compute_g2`.
- DataflowBufferSpecs: `in_dfb` (c_0), `scalar_dfb` (c_2), `out_dfb` (c_16), `partial_dfb` (c_21), `combined_dfb` (c_22).
- SemaphoreSpecs: none.
- TensorParameters: `input`, `output`.
- WorkUnitSpecs: `wu_g1`, optional `wu_g2`.

---

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (multi-binding) |
|---|---|---|---|
| Reduce H: compute_desc_g1 + compute_desc_g2 (same source, different `compute_Wt` CTA) | `compute_g1`, `compute_g2` | `wu_g1`, `wu_g2` | `in_dfb` (CONSUMER), `scaler_dfb` (CONSUMER), `out_dfb` (PRODUCER); negate-only: `acc_dfb` (self-loop), `ineg_dfb` (self-loop) |
| Reduce W: compute_desc_g1 + compute_desc_g2 (same source, different `num_rows_per_core_group_N` CTA) | `compute_g1`, `compute_g2` | `wu_g1`, `wu_g2` | `in_dfb` (CONSUMER), `scaler_dfb` (CONSUMER), `out_dfb` (PRODUCER); negate-only: `acc_dfb` (self-loop), `ineg_dfb` (self-loop) |
| Welford W: compute_desc_g1 + compute_desc_g2 (same source, same CTAs but different RTA `NCHt`) | `compute_g1`, `compute_g2` | `wu_g1`, `wu_g2` | `in_dfb` (CONSUMER), `scalar_dfb` (CONSUMER), `var_dfb` (self-loop), `out_dfb` (PRODUCER); do_scale-only: `scaled_dfb` (self-loop) |
| Welford H: compute_desc_g1 + compute_desc_g2 (same source, same CTAs but different RTA `NCWt`) | `compute_g1`, `compute_g2` | `wu_g1`, `wu_g2` | `in_dfb` (CONSUMER), `scalar_dfb` (CONSUMER), `out_dfb` (PRODUCER) |
| Welford HW: compute_desc_g1 + compute_desc_g2 (same source, same CTAs but different RTA `NC_per_core`) | `compute_g1`, `compute_g2` | `wu_g1`, `wu_g2` | `in_dfb` (CONSUMER), `scalar_dfb` (CONSUMER), `partial_dfb` (PRODUCER), `combined_dfb` (CONSUMER), `out_dfb` (PRODUCER) |

For the Welford variants, the CTAs are identical across g1 and g2 but the RTA differs — so the two `KernelSpec`s are structurally the same but bind to distinct work units. In principle one `KernelSpec` could span both groups, but to keep the multi-`KernelDescriptor` ↔ multi-`KernelSpec` 1:1 mapping clean (and avoid odd RTA-per-core layouts), the port preserves the same shape as Reduce H/W: one `KernelSpec` per work-split group.

Note: Welford H + HW reader RTAs (e.g. `col_start_tile_id`, `curr_col_in_batch`, `num_cols`) differ per work split because the reader sees different ranges of columns. The reader is single-`KernelSpec` (no per-group CTA difference); RTAs are emitted per-node.

---

## Dropped Plumbing

The buffer-address RTAs and the TensorAccessor CTA plumbing get replaced by `TensorBinding`s. The CB-index CTAs get replaced by `DFBBinding`s.

| legacy location (file) | legacy form | Metal 2.0 replacement |
|---|---|---|
| **Reduce H reader (interleaved)** RTA slot 0 | `a.buffer()->address()` | `TensorBinding(input)` |
| **Reduce H reader (interleaved)** CTAs 5+: `TensorAccessorArgs(input).append_to(...)` | TensorAccessorArgs plumbing | `TensorBinding(input)` — host packs the layout metadata; kernel uses `TensorAccessor(ta::input)` |
| **Reduce H reader (sharded)** CTAs 0–2 (`src0_cb_index`, `src1_cb_index`, `scaler_cb_index`) | magic CB indices | `DFBBinding`s on the reader (`in_dfb`, `in_shard_dfb`, `scaler_dfb`) |
| **Reduce H writer (interleaved)** RTA slot 0 | `output.buffer()->address()` | `TensorBinding(output)` |
| **Reduce H writer (interleaved)** CTA 0 (`output_cb_index`) + TensorAccessorArgs plumbing | magic CB index + TA plumbing | `DFBBinding(out_dfb, CONSUMER)` + `TensorBinding(output)` |
| **Reduce H writer (sharded)** CTA 0 (`output_cb_index`) | magic CB index | `DFBBinding(out_dfb, CONSUMER)` |
| **Reduce W reader** RTA slot 0 + TensorAccessorArgs | `a.buffer()->address()` + TA plumbing | `TensorBinding(input)` |
| **Reduce W writer** RTA slot 0 + CTA 0 + TensorAccessorArgs | `output.buffer()->address()` + CB index + TA plumbing | `TensorBinding(output)` + `DFBBinding(out_dfb, CONSUMER)` |
| **Reduce HW reader** RTA slot 0 + TensorAccessorArgs | `a.buffer()->address()` + TA plumbing | `TensorBinding(input)` |
| **Reduce HW writer** RTA slot 0 + CTA 0 + TensorAccessorArgs | `output.buffer()->address()` + CB index + TA plumbing | `TensorBinding(output)` + `DFBBinding(out_dfb, CONSUMER)` |
| **All Reduce compute kernels** — positional CTAs `Ht`, `Wt`, `NC`, `post_mul_scaler_bits` | positional CTAs | Named CTAs: `Ht`, `Wt`, `NC`, `post_mul_scaler_bits` |
| **All Reduce compute kernels** — implicit CB index references `tt::CBIndex::c_0` / `c_2` / `c_3` / `c_4` / `c_5` | magic CB indices in source | `DFBBinding`s on each compute kernel; kernel constructs wrappers as `DataflowBuffer cb_input_obj(dfb::in_dfb)`, etc. |
| **Welford reader (W)** RTA slot 0 + CTA 0 (scaler_bits) + TensorAccessorArgs | `tensor_arg.buffer()->address()` + scaler + TA plumbing | `TensorBinding(input)` + named CTA `scaler_bits` |
| **Welford reader (H/HW)** RTA slot 0 + CTAs 0–4 (Ht, Wt, HtWt, scaler_bits, use_welford) + TensorAccessorArgs | `tensor_arg.buffer()->address()` + CTAs + TA plumbing | `TensorBinding(input)` + named CTAs (`Ht`, `Wt`, `HtWt`, `scaler_bits`, `use_welford`) |
| **Welford W/H writer** (cross-op writer) | as Reduce W writer | as Reduce W writer |
| **Welford HW writer (op-local)** RTA slot 0 + CTAs (Wt, W, tile_width, H, correction, reduce_batch_size) + TensorAccessorArgs | `tensor_return_value.buffer()->address()` + CTAs + TA plumbing | `TensorBinding(output)` + named CTAs + `DFBBinding(partial_dfb, CONSUMER)`, `DFBBinding(combined_dfb, PRODUCER)`, `DFBBinding(out_dfb, CONSUMER)` |
| **Welford compute kernels** positional CTAs | positional CTAs | Named CTAs (`Wt`/`W`/`tile_width`/`do_scale`/`correction`/`is_std` for W; `Ht`/`H`/`tile_height`/... for H/HW) |
| **Welford compute kernels** implicit CB index references | magic CB indices in source | `DFBBinding`s; wrappers via `dfb::name` |

---

## Applied Patterns

- **[Multi-variant factory](../../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-multi-variant-factories)** — Welford W/H/HW selected from `attrs.reduce_dim` inside `create_program_spec`.
- **[Self-loop DFB binding](../../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-self-loop-dfb-binding)** — Reduce negate compute kernels bind `acc_dfb` and `ineg_dfb` as both PRODUCER and CONSUMER. Welford W compute binds `scaled_dfb` and `var_dfb` as self-loops.
- **[Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers)** — every kernel-lib / LLK call passes `dfb::name` directly; `DFBAccessor::operator uint32_t()` does the conversion. No `.id` extraction, no temp wrappers.
- **[Unity-build hygiene for anonymous-namespace symbols](../../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols)** — four factory `.cpp` files in the same `ttnn_op_reduction` target need disambiguated anonymous-namespace symbols. Plan: prefix per-factory constants and helpers with factory tag (e.g., `H_READER`, `W_READER`, `HW_READER`, `WELFORD_READER`).

### Conditional / optional DFB bindings — unconditional bind + `if constexpr`

Per the [Pattern: Conditional / optional DFB bindings](../../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-conditional--optional-dfb-bindings), the recommended port shape is to **bind the DFB unconditionally on the host** and gate **only the uses** inside the kernel with `if constexpr` on a CTA.

For this port:

- **Reduce H/W/HW with `negate`**: `acc_dfb` (c_4) and `ineg_dfb` (c_5) are used only when `attrs.negate == true`. The legacy code allocates these CBs only on the negate path.
- **Welford W with `do_scale`**: `scaled_dfb` (c_20) is used only when `do_scale == true`. The legacy code allocates this CB only when `do_scale`.

**Decision per Reduce factories:** the Reduce-negate paths use **separate compute kernel sources** (`reduce_h_neg.cpp` vs. `reduce.cpp`, etc.) — the negate vs. non-negate paths are not flag-gated within one kernel, they're entirely separate kernel files. So the host doesn't need a CTA-gated `if constexpr` here at all: the negate kernel always uses `acc_dfb` / `ineg_dfb`, and the non-negate kernel never references them. **The DFBs are bound unconditionally in the negate-path branch of the factory** (and not bound at all in the non-negate branch). This is the host's natural construction; no `if constexpr` gating needed on the kernel side.

**Decision per Welford W:** `welford_reduce_w.cpp` has both code paths (`do_scale=true` and `do_scale=false`) and gates them with `if constexpr (do_scale)`. The L1 cost of binding `scaled_dfb` when `do_scale=false` is ~1 input-tile per core. The Welford W kernel already has the `do_scale` CTA — bind `scaled_dfb` unconditionally on the host; in the kernel, the existing `if constexpr (do_scale)` block keeps the wrapper declaration + uses inside, and changes the wrapper construction from `CircularBuffer cb_scaled_obj(cb_scaled)` to `DataflowBuffer cb_scaled_obj(dfb::scaled_dfb)`. Per the pattern: declaring the wrapper inside the `if constexpr` block is required (the alternative — declaring at top level — would compile-fail because the lookup of `dfb::scaled_dfb` happens at parse time and would need an unconditional binding... wait, since we ARE binding unconditionally, `dfb::scaled_dfb` IS declared. So the wrapper CAN be declared at top level. Hmm — the migration-guide's "Optional resources" section says "wrapper inside the gate" for the conditional-host-binding case; but for the unconditional-bind case, the kernel can declare the wrapper anywhere. Going with the migration-guide's example shape: wrapper declared inside the `if constexpr` block to avoid paying construction cost when unused. The wrapper construction itself is `constexpr`-free runtime code, so it does cost a (tiny) couple of instructions when invoked; better to put it inside the gate.)

Per the patterns catalog [Pattern: Conditional / optional DFB bindings](../../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-conditional--optional-dfb-bindings) "Correct port" example: wrapper declared at top level, uses gated by `if constexpr`. The existing welford_reduce_w.cpp already has the wrapper at top level (line 63: `CircularBuffer cb_scaled_obj(cb_scaled);`). Keeping that shape and converting to `DataflowBuffer cb_scaled_obj(dfb::scaled_dfb);` at top level matches the pattern. The uses (lines 144–157) are already gated by `if constexpr (do_scale)` in the existing code. Good — minimal kernel disruption.

The host binds `scaled_dfb` unconditionally (i.e., for the Welford W variant, the DFB is always in the spec, always bound). When `do_scale=false`, the L1 is allocated but unused; the `if constexpr (do_scale)` block elides the uses. Per the migration guide's stated cost: ~1 input-tile per core in waste when `do_scale=false`.

---

## Deferred / Flagged

- The audit's YELLOW Device 2.0 holdover list is folded into the port as port-time cleanup. Each `get_tile_size(cb_id)` → `cb_obj.get_tile_size()` swap is one line per site.
- The cross-op writer `get_local_cb_interface(cb_id).fifo_page_size` site in the forked writer is the trickier swap. Plan: the `LocalCBInterface` accessor's `fifo_page_size` is an internal CB runtime accessor. The DFB wrapper exposes `get_page_size()` (verify in the DFB API header during construction). Equivalent on Gen1.
- **REDUCE_POST_MUL kernel define stays.** Per the audit's Question 1 self-answer: the `#ifdef REDUCE_POST_MUL` blocks in the Reduce compute kernels are load-bearing legacy `#ifdef`s that gate code logic (not DFB use); preserving them is sanctioned. The host's `KernelSpec::compiler_options.defines` carries the define when `use_post_mul == true`.
- **`scaler_bits` CTA on Reduce H interleaved reader is no longer needed in CTA form** once it's a named CTA, but the legacy kernel constructs `scaler_f = bit_cast<float>(scaler_bits)` and uses it. The named CTA preserves the semantics.
