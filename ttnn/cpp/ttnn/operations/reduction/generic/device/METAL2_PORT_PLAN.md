# Port Plan — reduction/generic op family

Port plan for the four program factories under `ttnn/cpp/ttnn/operations/reduction/generic/device/`, ported from `ProgramDescriptor` to Metal 2.0. Audit cleared GREEN (see `METAL2_PREPORT_AUDIT.md`).

Two device-operations share this directory; the port treats them as one family because they share kernel sources:

- `ReduceDeviceOperation`: `ReduceMultiCoreWProgramFactory`, `ReduceMultiCoreHProgramFactory`, `ReduceSingleCoreHwProgramFactory`.
- `WelfordReduceDeviceOperation`: `WelfordReduceProgramFactory` (multi-variant W/H/HW).

## Legacy Inventory

### Factory shape
- Concept: `ProgramDescriptorFactoryConcept` (all four — `create_descriptor` returning `ProgramDescriptor`).
- Variants:
  - `ReduceMultiCoreWProgramFactory` — single variant (operates on W reduction; conditional negate kernel via `negate` attribute).
  - `ReduceMultiCoreHProgramFactory` — two structural variants (interleaved vs. `use_width_sharding`); plus conditional negate.
  - `ReduceSingleCoreHwProgramFactory` — single variant (operates on HW reduction; conditional negate).
  - `WelfordReduceProgramFactory` — three variants by `reduce_dim` (W / H / HW), each with its own kernels and CB set.

### Cross-op kernels

Two kernel sources are referenced from outside the op's directory:

- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — used as writer by W, H (interleaved), HW factories, and Welford W/H. This is a peer-op shared dataflow kernel. Cautioned per [Modifying a shared dataflow kernel](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#caution-modifying-a-shared-dataflow-kernel).
- `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` — used as writer by H factory's `use_width_sharding` branch only. Reads from a (borrowed-memory) output CB; does not touch tensor memory.

Both are also touched by many other ops. Modifying them is in scope per the recipe (with caution).

### Flags

- `reduce_op_device_operation.hpp:39-44` declares `ReduceMultiCoreWProgramFactory::create_program_spec` returning `ttnn::device_operation::ProgramArtifacts`, but the implementation file `reduce_op_multi_core_w_program_factory.cpp` defines `create_descriptor` returning `ProgramDescriptor`. Header/impl mismatch from a previous-attempt revert; resolved by reimplementing the W factory as `create_program_spec` (which the header already expects).

---

### Variant: W (ReduceMultiCoreWProgramFactory)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | `kernels/dataflow/reader_unary_reduce_universal_start_id.cpp` | `all_cores` | `{scaler_bits, TensorAccessorArgs(*src)}` | `{src_buffer, num_tiles, start_id}` | `REDUCE_OP/REDUCE_DIM` (+ `REDUCE_POST_MUL` if `use_post_mul`) | ReaderConfig |
| writer | `<eltwise/unary>/.../writer_unary_interleaved_start_id.cpp` | `all_cores` | `{output_cb_index, TensorAccessorArgs(*dst)}` | `{dst_buffer, num_tiles, start_id}` | same as reader | WriterConfig |
| compute_g1 | `kernels/compute/reduce[_w_neg].cpp` | `core_group_1` | `{num_rows_g1, Wt, 1 /*NC*/, post_mul_scaler_bits}` | none | same | Compute |
| compute_g2 | (same) | `core_group_2` (if non-empty) | `{num_rows_g2, Wt, 1, post_mul_scaler_bits}` | none | same | Compute |

#### CBs

| index | total_size | core_ranges | data_format | page_size |
|---|---|---|---|---|
| 0 (c_0) | `2 * src0_tile_size` | `all_cores` | `src0_cb_data_format` | `src0_single_tile_size` |
| 2 (c_2) | `scaler_single_tile_size` | `all_cores` | `scaler_cb_data_format` | `scaler_single_tile_size` |
| 3 (c_3) | `2 * dst_tile_size` | `all_cores` | `dst_cb_data_format` | `dst_single_tile_size` |
| 4 (c_4, if `negate`) | `dst_tile_size` | `all_cores` | `dst_cb_data_format` | `dst_single_tile_size` |
| 5 (c_5, if `negate`) | `dst_tile_size` | `all_cores` | `dst_cb_data_format` | `dst_single_tile_size` |

#### Tensor accessors

| host site | originating Tensor | RTA slot | kernel-side accessor |
|---|---|---|---|
| `reader_unary_reduce_universal_start_id.cpp:13/28` | input `a` | RTA 0 (`src_addr`) → `TensorAccessor(tensor_args, src_addr)` | reader |
| `writer_unary_interleaved_start_id.cpp:11` | output | RTA 0 (`dst_addr`) → `TensorAccessor(dst_args, dst_addr)` | writer |

#### Work split

- Driver: `split_work_to_cores(grid, NC * Ht)` (rows of tile grid).
- num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_g1, num_rows_per_core_g2.

---

### Variant: H (ReduceMultiCoreHProgramFactory)

H has two structural sub-variants. The factory branches on `use_width_sharding` (input + output memory layout WIDTH_SHARDED).

#### H, interleaved (no width sharding)

##### Kernels

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | `kernels/dataflow/reader_unary_transpose_wh_universal_input_cols_partitioned.cpp` | `all_cores` | `{Ht, Wt, HtWt, scaler_bits, /*use_welford=*/0, TensorAccessorArgs(*src)}` | `{src, col_start_tile_id, curr_col_in_batch, num_cols}` | `REDUCE_OP/REDUCE_DIM` (+ `REDUCE_POST_MUL`) + `ENABLE_FP32_DEST_ACC`, `DST_SYNC_FULL` | ReaderConfig |
| writer | `<eltwise/unary>/.../writer_unary_interleaved_start_id.cpp` | `all_cores` | `{output_cb_index, TensorAccessorArgs(*dst)}` | `{dst, num_tiles, start_id}` | same | WriterConfig |
| compute_g1 | `kernels/compute/reduce[_h_neg].cpp` | `core_group_1` | `{Ht, compute_Wt_g1, compute_NC=1, post_mul_scaler_bits}` | none | same | Compute (`dst_full_sync_en`) |
| compute_g2 | (same) | `core_group_2` (if non-empty) | `{Ht, compute_Wt_g2, 1, post_mul_scaler_bits}` | none | same | Compute |

##### CBs

| index | total_size | core_ranges | data_format | page_size |
|---|---|---|---|---|
| 0 (c_0) | `num_input_tiles * src0_tile_size`, `num_input_tiles = negate ? chunk_size : 2` | `all_cores` | `src0_cb_data_format` | `src0_single_tile_size` |
| 2 (c_2) | `scaler_tile_size` | `all_cores` | `scaler_cb_data_format` | `scaler_single_tile_size` |
| 3 (c_3) | `num_output_tiles * dst_tile_size`, `num_output_tiles = negate ? chunk_size : 2` | `all_cores` | `dst_cb_data_format` | `dst_single_tile_size` |
| 4 (c_4, if `negate`) | `per_cb_total_size` (= `Ht * per_nc_advance * dst_tile_size`) | `all_cores` | `dst_cb_data_format` | `dst_single_tile_size` |
| 5 (c_5, if `negate`) | `per_cb_total_size` | `all_cores` | `dst_cb_data_format` | `dst_single_tile_size` |

#### H, width-sharded

##### Kernels

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | `kernels/dataflow/reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp` | `all_cores` | `{src0_cb_index, src1_cb_index, scaler_cb_index, scaler_bits}` | `{num_tiles, shard_Wt, Ht, NC, shard_row_size, shard_batch_size}` | `REDUCE_SCALER=1`, `ENABLE_FP32_DEST_ACC`, `DST_SYNC_FULL`, + reduce defines | ReaderConfig |
| writer | `<data_movement/sharded>/.../writer_unary_sharded.cpp` | `all_cores` | `{output_cb_index}` | `{num_units}` | (none) | WriterConfig |
| compute_g1 | `kernels/compute/reduce[_h_neg].cpp` | `core_group_1` | `{Ht, compute_Wt_g1 = num_cols_g1 / NC, compute_NC_g1 = NC, post_mul_scaler_bits}` | none | reduce defines | Compute |

##### CBs

| index | total_size | core_ranges | data_format | page_size | borrowed_from |
|---|---|---|---|---|---|
| 0 (c_0) | `2 * src0_tile_size` | `all_cores` | `src0_cb_data_format` | `src0_single_tile_size` | (none — local) |
| 1 (c_1) | `num_shard_tiles * src0_tile_size` | `all_cores` | `src0_cb_data_format` | `src0_single_tile_size` | **input** (`a.buffer()`) |
| 2 (c_2) | `scaler_tile_size` | `all_cores` | `scaler_cb_data_format` | `scaler_single_tile_size` | (none) |
| 3 (c_3) | `num_output_tiles * dst_tile_size` (`num_output_tiles = shard.numel() / tile_hw`) | `all_cores` | `dst_cb_data_format` | `dst_single_tile_size` | **output** (`output.buffer()`) |
| (no c_4/c_5 in width-sharded path — no negate combination supported here) | | | | | |

#### Tensor accessors (H, both)

| host site | originating Tensor | RTA slot | kernel-side accessor |
|---|---|---|---|
| interleaved reader (`reader_unary_transpose_wh_universal_input_cols_partitioned.cpp:41`) | input | RTA 0 → `TensorAccessor(tensor_args, src_addr)` | reader |
| interleaved writer (`writer_unary_interleaved_start_id.cpp`) | output | RTA 0 → `TensorAccessor(dst_args, dst_addr)` | writer |
| width-sharded reader | **none** — reads from borrowed-memory cb_in1 (causal-link gate) | n/a | n/a |
| width-sharded writer | **none** — reads from borrowed-memory cb_out | n/a | n/a |

#### Work split (H)

- Interleaved: `split_work_to_cores(grid, NC * Wt)` (columns of tile grid). `core_group_1`, `core_group_2`.
- Width-sharded: `all_cores = shard.grid`, `core_group_1 = all_cores`, `core_group_2 = empty`. Per-core `num_cols = NC * shard_Wt`.

---

### Variant: HW (ReduceSingleCoreHwProgramFactory)

Single-core, single-variant (with conditional negate).

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | `kernels/dataflow/reader_unary_reduce_universal_start_id.cpp` | `core_set` | `{sqrt(scaler)_bits, TensorAccessorArgs(*src)}` | `{src, num_tensor_tiles, 0}` | `REDUCE_OP/REDUCE_DIM` (+ `REDUCE_POST_MUL`) | ReaderConfig |
| writer | `<eltwise/unary>/.../writer_unary_interleaved_start_id.cpp` | `core_set` | `{output_cb_index, TensorAccessorArgs(*dst)}` | `{dst, num_tiles, 0}` | same | WriterConfig |
| compute | `kernels/compute/reduce[_hw_neg].cpp` | `core_set` | `{Ht, Wt, NC, post_mul_scaler_bits}` | none | same | Compute |

#### CBs

| index | total_size | data_format | page_size |
|---|---|---|---|
| 0 (c_0) | `2 * src0_tile_size` | `src0_cb_data_format` | `src0_single_tile_size` |
| 2 (c_2) | `scaler_tile_size` | `scaler_cb_data_format` | `scaler_single_tile_size` |
| 3 (c_3) | `2 * dst_tile_size` | `dst_cb_data_format` | `dst_single_tile_size` |
| 4 (c_4, if `negate`) | `dst_tile_size` | `dst_cb_data_format` | `dst_single_tile_size` |
| 5 (c_5, if `negate`) | `dst_tile_size` | `dst_cb_data_format` | `dst_single_tile_size` |

#### Work split

- n/a — single core. Optional `sub_core_grids` selects one core; default `{0, 0}`.

---

### Variant: Welford (WelfordReduceProgramFactory)

Multi-variant: branches on `reduce_dim` (W / H / HW). Each variant has its own kernels and CB set.

#### Welford W (`reduce_dim = W`)

##### Kernels

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | `kernels/dataflow/reader_unary_reduce_universal_start_id.cpp` | `all_cores` | `{scaler_bits, TensorAccessorArgs(*src)}` | `{src, num_input_tiles, input_tiles_offset}` | reduce defines, `ENABLE_FP32_DEST_ACC`, `DST_SYNC_FULL` | ReaderConfig |
| writer | `<eltwise/unary>/.../writer_unary_interleaved_start_id.cpp` | `all_cores` | `{output_cb_index (c_16), TensorAccessorArgs(*dst)}` | `{dst, num_output_tiles, output_tiles_offset}` | reduce defines | WriterConfig |
| compute_g1 | `kernels/compute/welford_reduce_w.cpp` | `core_group_1` | `{Wt, W, tile_width, do_scale, correction, is_std}` | `{NCHt = num_work_units_per_core}` | same | Compute |
| compute_g2 | (same) | `core_group_2` (if non-empty) | same shape | `{NCHt}` | same | Compute |

##### CBs

| index | total_size | data_format | page_size |
|---|---|---|---|
| 0 (c_0) | `2 * input_tile_size` | `input_cb_data_format` | `input_single_tile_size` |
| 2 (c_2) | `scalar_tile_size` | `Float16_b` | `scalar_single_tile_size` |
| 16 (c_16) | `2 * dst_tile_size` | `dst_cb_data_format` | `dst_single_tile_size` |
| 19 (c_19) | `scratch_tile_size` | `Float32` if `fp32_dest_acc_en` else `Float16_b` | `scratch_single_tile_size` |
| 20 (c_20, if `do_scale`) | `input_tile_size` | `input_cb_data_format` | `input_single_tile_size` |

##### Tensor accessors (Welford W)
Same as W reduce: reader has `TensorAccessor(input)`; writer has `TensorAccessor(output)`.

##### Work split (Welford W)
- `split_work_to_cores(grid, NC * Ht)`.

#### Welford H (`reduce_dim = H`)

##### Kernels

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | `kernels/dataflow/reader_unary_transpose_wh_universal_input_cols_partitioned.cpp` | `all_cores` | `{Ht, Wt, HtWt, scaler_bits, /*use_welford=*/1, TensorAccessorArgs(*src)}` | `{src, col_start, curr_col_in_batch=0, num_cols=num_cols_per_core}` | reduce defines (+ ENABLE/SYNC) | ReaderConfig |
| writer | `<eltwise/unary>/.../writer_unary_interleaved_start_id.cpp` | `all_cores` | `{output_cb_index, TensorAccessorArgs(*dst)}` | `{dst, num_cols_per_core, num_cols_read}` | reduce defines | WriterConfig |
| compute_g1 | `kernels/compute/welford_reduce_h.cpp` | `core_group_1` | `{Ht, H, tile_height, do_scale, correction, is_std}` | `{NCWt = num_cols_per_core}` | same | Compute |
| compute_g2 | (same) | `core_group_2` (if non-empty) | same shape | `{NCWt}` | same | Compute |

##### CBs

| index | total_size | data_format | page_size |
|---|---|---|---|
| 0 (c_0) | `2 * input_tile_size` | `input_cb_data_format` | `input_single_tile_size` |
| 2 (c_2) | `scalar_tile_size` | `Float16_b` | `scalar_single_tile_size` |
| 16 (c_16) | `2 * dst_tile_size` | `dst_cb_data_format` | `dst_single_tile_size` |

##### Work split (Welford H)
- `split_work_to_cores(grid, NC * Wt)`.

#### Welford HW (`reduce_dim = HW`)

##### Kernels

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | `kernels/dataflow/reader_unary_transpose_wh_universal_input_cols_partitioned.cpp` | `all_cores` | `{Ht, Wt, HtWt, scaler_bits, /*use_welford=*/1, TensorAccessorArgs(*src)}` | `{src, col_start_tile_id, curr_col_in_batch=0, num_cols = Wt*nc_slices}` | reduce defines (+ ENABLE/SYNC) | ReaderConfig |
| writer | `kernels/dataflow/writer_welford_hw.cpp` | `all_cores` | `{Wt, W, tile_width, H, correction, reduce_batch_size, TensorAccessorArgs(*dst)}` | `{dst, NC_per_core = nc_slices_per_core, output_tile_start_id}` | (none, matches original) | WriterConfig |
| compute_g1 | `kernels/compute/welford_reduce_hw.cpp` | `core_group_1` | `{Ht, H, tile_height, Wt, do_scale, reduce_batch_size, is_std}` | `{NC_per_core = nc_slices_per_core}` | reduce defines | Compute |
| compute_g2 | (same) | `core_group_2` (if non-empty) | same shape | `{NC_per_core}` | same | Compute |

##### CBs

| index | total_size | data_format | page_size |
|---|---|---|---|
| 0 (c_0) | `2 * input_tile_size` | `input_cb_data_format` | `input_single_tile_size` |
| 2 (c_2) | `scalar_tile_size` | `Float16_b` | `scalar_single_tile_size` |
| 16 (c_16) | `2 * dst_tile_size` | `dst_cb_data_format` | `dst_single_tile_size` |
| 21 (c_21) | `4 * partial_tile_size_fp32` | `Float32` | `partial_single_tile_size` |
| 22 (c_22) | `combined_tile_size_fp32` | `Float32` | `combined_single_tile_size` |

##### Work split (Welford HW)
- `split_work_to_cores(grid, NC / reduce_batch_size)`.

---

## Planned Spec Shape

### Variant: W reduce (ReduceMultiCoreWProgramFactory)

- KernelSpecs: `reader`, `writer`, `compute_g1`, optional `compute_g2`.
- DataflowBufferSpecs: `input_dfb` (c_0), `scaler_dfb` (c_2), `output_dfb` (c_3); conditionally `acc_dfb` (c_4) and `ineg_dfb` (c_5) when `negate`.
- SemaphoreSpecs: none.
- TensorParameters: `input` (input tensor), `output` (output tensor).
- WorkUnitSpecs: `wu_g1` (kernels: reader, writer, compute_g1 — `target_nodes = core_group_1`); optional `wu_g2` (kernels: reader, writer, compute_g2 — `target_nodes = core_group_2`). The reader/writer are listed in both work units because they run on all cores.

### Variant: H reduce, interleaved branch

- KernelSpecs: `reader`, `writer`, `compute_g1`, optional `compute_g2`.
- DataflowBufferSpecs: `input_dfb` (c_0), `scaler_dfb` (c_2), `output_dfb` (c_3); conditionally `acc_dfb` (c_4) and `ineg_dfb` (c_5) when `negate`.
- TensorParameters: `input`, `output`.
- WorkUnitSpecs: `wu_g1`, optional `wu_g2`.

### Variant: H reduce, width-sharded branch

- KernelSpecs: `reader`, `writer`, `compute_g1` (no g2 here — all cores in group_1).
- DataflowBufferSpecs: `input_dfb` (c_0, local), `input_sharded_dfb` (c_1, **borrowed_from = input**), `scaler_dfb` (c_2), `output_dfb` (c_3, **borrowed_from = output**).
- TensorParameters: `input`, `output`.
- WorkUnitSpecs: `wu_main` (kernels: reader, writer, compute_g1 — `target_nodes = all_cores`).

### Variant: HW reduce (ReduceSingleCoreHwProgramFactory)

- KernelSpecs: `reader`, `writer`, `compute`.
- DataflowBufferSpecs: `input_dfb`, `scaler_dfb`, `output_dfb`; conditionally `acc_dfb`, `ineg_dfb` when `negate`.
- TensorParameters: `input`, `output`.
- WorkUnitSpecs: `wu_main` (single core).

### Variant: Welford (multi-variant factory inside `create_program_spec`)

Branches on `reduce_dim`:

- **Welford W**: KernelSpecs = `reader`, `writer`, `compute_g1`, optional `compute_g2`. DFBs = `input_dfb` (c_0), `scaler_dfb` (c_2), `output_dfb` (c_16), `scratch_dfb` (c_19); conditionally `scaled_dfb` (c_20) when `do_scale`. TensorParameters = `input`, `output`. WorkUnitSpecs = `wu_g1`, optional `wu_g2`.
- **Welford H**: KernelSpecs = `reader`, `writer`, `compute_g1`, optional `compute_g2`. DFBs = `input_dfb`, `scaler_dfb`, `output_dfb`. TensorParameters = `input`, `output`. WorkUnitSpecs = `wu_g1`, optional `wu_g2`.
- **Welford HW**: KernelSpecs = `reader`, `writer`, `compute_g1`, optional `compute_g2`. DFBs = `input_dfb`, `scaler_dfb`, `output_dfb`, `partial_dfb` (c_21), `combined_dfb` (c_22). TensorParameters = `input`, `output`. WorkUnitSpecs = `wu_g1`, optional `wu_g2`.

Pattern: [Multi-variant factories](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-multi-variant-factories).

## Preserved Multiplicity

All three Reduce factories (W, H interleaved, Welford W/H/HW) use `split_work_to_cores` and create two compute `KernelDescriptor`s with different per-group CTAs (`num_rows_g1` vs `num_rows_g2` for W; `Ht`/`compute_Wt_*` etc.). Per [Demoting per-group CTA to RTA](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta), this maps 1:1 to two compute `KernelSpec`s sharing reader/writer/DFBs in two `WorkUnitSpec`s.

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (multi-binding) |
|---|---|---|---|
| `compute_desc_g1`, `compute_desc_g2` (W) | `compute_g1`, `compute_g2` | `wu_g1`, `wu_g2` | `input_dfb` (CONSUMER ×2 via compute), `output_dfb` (PRODUCER ×2), `scaler_dfb` (CONSUMER ×2), `acc_dfb`/`ineg_dfb` (self-loop ×2 if negate) — reader/writer themselves run on all cores; in our model reader and writer go into **both** work units, so input/output DFBs gain one consumer/producer (compute) per variant. |
| same for H interleaved, Welford W, Welford H, Welford HW | same | same | same |
| H, width-sharded path | single `compute_g1` only (no work-split) | single `wu_main` | no multiplicity |
| HW (single core) | single `compute` | single `wu_main` | no multiplicity |

For all multi-group variants, both `compute_g1` and `compute_g2` bind the same DFBs (`input` CONSUMER, `output` PRODUCER, `scaler` CONSUMER). The framework's multi-PRODUCER / multi-CONSUMER relaxation handles this.

## Dropped Plumbing

Across the family:

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| Reader CTA slot for `scaler_bits` | positional CTA `{scaler_bits, ...}` | Named CTA `{"scaler_bits", scaler_bits}` |
| Reader CTA `TensorAccessorArgs(*src).append_to(cta)` and kernel `TensorAccessorArgs<N>()` | `TensorAccessorArgs` plumbing | `TensorBinding(input, "input")` + `TensorAccessor(ta::input)` in kernel |
| Writer CTA `{output_cb_index, ...}` | positional CTA for CB id | `DFBBinding(output_dfb, "out", PRODUCER)` + named CTA absent (DFB id implicit via `dfb::out`) |
| Writer CTA `TensorAccessorArgs(*dst).append_to(cta)` and kernel `TensorAccessorArgs<1>()` | `TensorAccessorArgs` plumbing | `TensorBinding(output, "output")` + `TensorAccessor(ta::output)` in kernel |
| Reader RTA slot 0 `a.buffer()` (passed as `src_addr` to kernel `get_arg_val<uint32_t>(0)`) | buffer-address RTA | `TensorBinding(input, "input")` (auto-injected) |
| Writer RTA slot 0 `output.buffer()` | buffer-address RTA | `TensorBinding(output, "output")` (auto-injected) |
| H width-sharded reader CTAs `{src0_cb_index, src1_cb_index, scaler_cb_index, scaler_bits}` | magic CB indices in positional CTAs | DFB bindings + named CTA for `scaler_bits` only |
| H width-sharded writer CTA `{output_cb_index}` | magic CB index | DFB binding (output via `borrowed_from`) |
| H width-sharded input CB c_1 with `.buffer = a.buffer()` | borrowed-memory CB (dynamic CB) | `DataflowBufferSpec{.borrowed_from = "input", ...}` |
| H width-sharded output CB c_3 with `.buffer = output.buffer()` | borrowed-memory CB | `DataflowBufferSpec{.borrowed_from = "output", ...}` |
| All compute CTAs `{Ht, Wt, NC, post_mul_scaler_bits}` (W/H/HW reduce); `{Wt, W, tile_width, do_scale, correction, is_std}` etc. (Welford) | positional CTAs | Named CTAs `{"Ht", Ht}`, etc. |
| Reduce W compute RTAs (none in W; H/HW each have RTA-shaped `num_cols_per_core` etc.) | positional RTAs `get_arg_val<uint32_t>(N)` | Named RTAs `args::num_work_units_per_core` etc. |
| Reduce W `c_4`/`c_5` (acc/ineg) — both PRODUCER and CONSUMER from compute kernel | magic CB index ×2 used self-loop | `DFBBinding ×2` per kernel: same DFB with PRODUCER + CONSUMER endpoint_type. See [Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-self-loop-dfb-binding). |
| Welford W `c_20` (scaled) conditional on `do_scale` | unconditionally declared CB, kernel gates uses via `if constexpr` | Conditional DFB binding on compute kernel only when `do_scale`; kernel wraps decl + uses in `if constexpr`. See [Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-conditional--optional-dfb-bindings). |
| Welford HW `c_21`/`c_22` — written by compute, read by writer (and vice versa for c_22) | magic CB indices | DFBs with PRODUCER/CONSUMER bindings; c_22 is writer→compute, c_21 is compute→writer |
| Defines map `{REDUCE_OP, REDUCE_DIM, REDUCE_POST_MUL?, ENABLE_FP32_DEST_ACC?, DST_SYNC_FULL?}` | `KernelDescriptor::defines` map | `KernelSpec::compiler_options::defines` vector of `{name, value}` pairs |

## Applied Patterns

- [Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers): kernel-lib calls (`dataflow_kernel_lib::prepare_reduce_scaler<cb_id_in2, REDUCE_OP, REDUCE_DIM>`) and LLK calls (`reduce_init`, `reduce_tile`, `compute_kernel_hw_startup`, `pack_tile`, `copy_tile`, etc.) take CB-id `uint32_t`. Replace `cb_id_in2`-style constants with `dfb::name` — the implicit `DFBAccessor::operator uint32_t()` carries them.
- [Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-self-loop-dfb-binding): `acc_dfb` / `ineg_dfb` on `reduce_*_neg` compute kernels — both PRODUCER and CONSUMER on the same compute KernelSpec.
- [Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-conditional--optional-dfb-bindings): Welford W's `scaled_dfb` (c_20), gated by `do_scale` named CTA. Also: `acc_dfb`/`ineg_dfb` for negate variants — host conditionally adds the DFB and the binding, kernel wraps decl+uses in `if constexpr`.

  *Negate handling — implementation choice*: today the negate compute kernels (`reduce_w_neg.cpp` etc.) are *separate kernel sources* selected by the host based on `operation_attributes.negate`. The non-negate kernel does not reference `c_4`/`c_5`. Therefore we keep the legacy kernel-selection approach — host picks `reduce.cpp` vs `reduce_*_neg.cpp` — and binds `acc_dfb`/`ineg_dfb` only on the negate path. No `if constexpr` gating needed for the negate kernels themselves; the unconditional reference inside the negate kernels is fine. This keeps the port mechanical and avoids reshaping the kernel sources.
- [Multi-variant factories](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-multi-variant-factories): Welford factory branches on `reduce_dim` inside `create_program_spec`. H factory branches on `use_width_sharding` (also a multi-variant case).
- [Unity-build hygiene for anonymous-namespace symbols](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols): four `.cpp` files in the same TU; will prefix per-factory constants (`W_READER_KERNEL`, `H_READER_KERNEL`, etc.) or use unnamed namespaces in headers.

## Deferred / Flagged

- None from audit YELLOW — the `get_tile_size(cb_id)` holdovers fold into port-time cleanup.
- New finding during planning: Welford HW writer kernel `writer_welford_hw.cpp` reads from `cb_partial` (c_21) and writes to `cb_combined` (c_22). Both are program-scope DFBs. The writer KernelSpec binds `c_21` as CONSUMER (reads `cb_partial` partials from compute) and `c_22` as PRODUCER (writes combined scalar tile back to compute); the compute KernelSpec binds `c_21` PRODUCER and `c_22` CONSUMER. This is a 2-way data flow between writer and compute on the same node, which the local-DFB invariant (one PRODUCER, one CONSUMER per DFB, same WorkUnitSpec membership) supports cleanly — these are two separate DFBs, each with a single producer / single consumer.
