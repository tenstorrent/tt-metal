# Port Plan — SamplingProgramFactory

Port plan for `SamplingProgramFactory`, ported from `ProgramDescriptor` to Metal 2.0.

## Legacy Inventory

### Factory shape

- Concept: `ProgramDescriptorFactoryConcept` (`create_descriptor` returns `ProgramDescriptor`).
- Variants: single factory, single mode (interleaved tensors only). Per-core specialization is via `core_id` CTA on the writer kernel.

### Kernels

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader (1 KernelDescriptor) | `kernels/dataflow/reader_values_indices_tensor.cpp` | full `core_grid` | `[input_values_cb_index, input_indices_cb_index, cb_intermed_index, Ht, Wt, input_indices_page_size, tile_height, TAArgs(values), TAArgs(indices)]` | `[values_addr, indices_addr]` per core | none | `ReaderConfigDescriptor` |
| writer (N KernelDescriptors, one per core) | `kernels/dataflow/writer_interleaved.cpp` | `single_core` (1×1 per core) | `[TAArgs(out), TAArgs(temp), TAArgs(k), TAArgs(p), cb_out, cb_mask, scaler_max, scaler_sum, final_idx_rm, local_vals, local_indices, final_idx_stick, out_stick, rand_tile, cb_k, cb_p, cb_temp, core_id, tile_width, num_cores]` | `[dst, temp, k, p]` | none | `WriterConfigDescriptor` |
| compute (N KernelDescriptors, one per core; identical CTAs) | `kernels/compute/sampling.cpp` | `single_core` | `[input_cb, index_cb, input_T_cb, index_T_cb, values_cb, output_ind_cb, topk_mask_cb, scaler_max_cb, scaler_sum_cb, cb_cur_max, cb_cur_sum, Ht, Wt, log2(Wt), rand_tile, seed, cb_local_vals, temp_cb, tile_width]` | none | none | `ComputeConfigDescriptor` |

### CBs

16 standard CBs, all `core_ranges = core_grid`, no `.buffer` set (none borrowed):

| index | symbol | size | data_format |
|---|---|---|---|
| `c_0`  | `input_values_cb`     | `cb_in_units * input_values_tile_size`    | `input_values_cb_data_format` (BFLOAT16) |
| `c_1`  | `cb_local_vals`       | `num_cb_unit * input_values_tile_size`    | `input_values_cb_data_format` |
| `c_2`  | `index_cb`            | `cb_in_units * index_tile_size`           | `UInt16` |
| `c_3`  | `scaler_max_cb`       | `scale_tiles * scalar_tile_size`          | `scalar_df` (BF16 or F32) |
| `c_4`  | `topk_mask_cb`        | `cb_in_units * input_values_tile_size`    | `input_values_cb_data_format` |
| `c_5`  | `input_transposed_cb` | `Wt * input_values_tile_size`             | `input_values_cb_data_format` |
| `c_6`  | `index_transposed_cb` | `Wt * index_tile_size`                    | `UInt16` |
| `c_7`  | `values_cb`           | `num_cb_unit * input_values_tile_size`    | `input_values_cb_data_format` |
| `c_8`  | `output_ind_cb`       | `num_cb_unit * index_tile_size`           | `UInt16` |
| `c_9`  | `cb_cur_max`          | `num_out_tiles * input_values_tile_size`  | `input_values_cb_data_format` |
| `c_10` | `cb_cur_sum`          | `num_out_tiles * input_values_tile_size`  | `input_values_cb_data_format` |
| `c_11` | `rand_tile`           | `rand_tile_size`                          | `Float16_b` |
| `c_12` | `final_indices_rm_cb` | `Ht * tile_height * aligned_size`         | `input_indices_cb_data_format` (UInt32/Int32) |
| `c_13` | `output_cb`           | `aligned_out0_unit_size`                  | `UInt16` |
| `c_14` | `cb_k`                | `num_cores * sizeof(uint32_t)`            | `k_cb_data_format` (UInt32) |
| `c_15` | `cb_p`                | `num_cores * sizeof(uint16_t)`            | `p_cb_data_format` (BFloat16) |
| `c_16` | `cb_temp`             | `num_cores * sizeof(uint16_t)`            | `temp_cb_data_format` (BFloat16) |
| `c_17` | `scaler_sum_cb`       | `scale_tiles * scalar_tile_size`          | `scalar_df` |

### Semaphores

none

### Tensor accessors

| host site | originating Tensor | RTA slot (host) | CTA offset (kernel) |
|---|---|---|---|
| reader CT `TensorAccessorArgs(input_values_buffer)` | `tensor_args.input_values` | reader RTA[0] = `values_addr` | reader CTA[7..] = `s0_args` |
| reader CT `TensorAccessorArgs(input_indices_buffer)` | `tensor_args.input_indices` | reader RTA[1] = `indices_addr` | reader CTA = `s1_args` |
| writer CT `TensorAccessorArgs(output_buffer)` | output | writer RTA[0] = `dst_addr` | writer CTA[0..] = `dst_args` |
| writer CT `TensorAccessorArgs(temp_buffer)` | `tensor_args.temp` | writer RTA[1] = `temp_addr` | writer CTA = `temp_args` |
| writer CT `TensorAccessorArgs(k_buffer)` | `tensor_args.k` | writer RTA[2] = `k_addr` | writer CTA = `k_args` |
| writer CT `TensorAccessorArgs(p_buffer)` | `tensor_args.p` | writer RTA[3] = `p_addr` | writer CTA = `p_args` |

### Work split

- Driver: `num_cores = input_shape[0] * input_shape[1] * input_shape[2]` (= 32 users by validation). Each core handles one user.
- Cores from `num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size, true)` or `sub_core_grids` override.

### Cross-op kernels

None — every kernel `source` is under `device/kernels/`.

### Flags

The `core_id` CTA varies per-writer-core (legacy creates 32 writer + 32 compute KernelDescriptors). The compute kernels are functionally identical across cores (no `core_id` CTA), so the multi-`KernelDescriptor` compute construction is legacy redundancy — we collapse to a single compute `KernelSpec`. For the writer, `core_id` can be either preserved as N `KernelSpec`s or promoted to a per-node named RTA. We choose the **named-RTA** route here — `core_id` is used only as a runtime array index and a runtime branch on `>= FACE_WIDTH` (no compile-time loop unrolling depends on it), so the demoted form is benign per the [Demoting per-group CTA to RTA](metal2_port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta) caveat (only fires when CTA participates in loop unrolling).

## Planned Spec Shape

- **KernelSpecs**: three — `reader`, `writer`, `compute`. One `KernelSpec` each.
- **DataflowBufferSpecs**: 16, all standard (no `borrowed_from`).
- **SemaphoreSpecs**: none.
- **TensorParameters**: six — `INPUT_VALUES`, `INPUT_INDICES`, `K`, `P`, `TEMP`, `OUTPUT`.
- **WorkUnitSpecs**: one (`main`) covering `{reader, writer, compute}` on `core_grid`.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (multi-binding) |
|---|---|---|---|

(legacy created N writer + N compute `KernelDescriptor`s of identical sources, but the per-core variation was a single CTA — `core_id` on the writer. We collapse to one `KernelSpec` per kernel and promote `core_id` to a named RTA on the writer.)

## Dropped Plumbing

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader CTA slots (after positional): `TensorAccessorArgs(input_values_buffer)`, `TensorAccessorArgs(input_indices_buffer)` | TensorAccessorArgs plumbing | `TensorBinding(INPUT_VALUES, "values")` + `TensorBinding(INPUT_INDICES, "indices")` on reader; `TensorAccessor(ta::values)` / `TensorAccessor(ta::indices)` in kernel |
| writer CTA slots: 4× `TensorAccessorArgs(...)` for output/temp/k/p | TensorAccessorArgs plumbing | 4 `TensorBinding`s on writer + `TensorAccessor(ta::name)` in kernel |
| reader RTAs `[values_addr, indices_addr]` | buffer-address RTAs | gone (auto-injected by TensorBindings) |
| writer RTAs `[dst_addr, temp_addr, k_addr, p_addr]` | buffer-address RTAs | gone (auto-injected by TensorBindings) |
| Per-core writer/compute KernelDescriptors (N=32 each) | per-core kernel specialization via positional CTAs | one writer `KernelSpec` with `core_id` as a per-node named RTA; one compute `KernelSpec` (CTAs are uniform across cores) |
| All positional CTAs (16 indices + scalars on writer/compute) | positional CTAs | Named CTAs throughout |

## Applied Patterns

- **Single KernelSpec per source** (collapse compute multi-`KernelDescriptor` to one `KernelSpec`; promote writer's `core_id` from CTA to named RTA).
- **Multi-variant factory** is N/A — the op has a single mode.

## Deferred / Flagged

None.
