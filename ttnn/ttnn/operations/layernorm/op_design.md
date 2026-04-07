# Operation Design: layer_norm

## Overview

| Field | Value |
|-------|-------|
| Classification | fused normalization/reduction device op |
| Goal | Implement a real C++ host/device `ttnn::layer_norm(...)` under `ttnn/cpp/ttnn/operations/normalization/layernorm` with dedicated reader/compute/writer kernels, normalization build integration, and nanobind registration. |
| Math | `y = (((x + residual?) - mean_last_dim(x + residual?)) * rsqrt(var_last_dim(x + residual?) + epsilon)) * weight? + bias?`; shared primitive plumbing keeps room for `RMSNORM` by skipping the mean-subtraction phase. |
| Mode | Hybrid |
| References | `ttnn/ttnn/operations/layernorm/architecture.md`; `ttnn/ttnn/operations/layernorm/design_journal.jsonl`; `ttnn/cpp/ttnn/operations/normalization/CMakeLists.txt:14-35`; `ttnn/CMakeLists.txt:367-368`; `ttnn/cpp/ttnn/operations/normalization/normalization_nanobind.cpp:9-13`; `ttnn/cpp/ttnn/operations/experimental/transformer/dit_rms_norm_unary_fused/dit_rms_norm_unary_fused.hpp:11-29`; `ttnn/cpp/ttnn/operations/experimental/transformer/dit_rms_norm_unary_fused/dit_rms_norm_unary_fused.cpp:38-52` |

## Compatibility Corrections

| Item | Architect Said | Actual | Engineering Resolution |
|------|----------------|--------|------------------------|
| Common type surface | `layernorm_common.hpp` is sufficient | Current workspace already includes `layernorm_types.hpp` from `dit_rms_norm_unary_fused.hpp:11-29` and `layernorm_common.hpp` from `dit_rms_norm_unary_fused.cpp:6-7` | Add both `device/layernorm_types.hpp` and `device/layernorm_common.hpp`; `layernorm_common.hpp` includes helper declarations and re-exports the type surface. |
| Nanobind build wiring | normalization CMake update is enough | Runtime sources live in `ttnn/cpp/ttnn/operations/normalization/CMakeLists.txt:14-35`, but nanobind TUs are enumerated separately in `ttnn/CMakeLists.txt:367-368` | Add layernorm runtime/device `.cpp` files to normalization CMake, add `layernorm_nanobind.cpp` to `ttnn/CMakeLists.txt`, and register it in `normalization_nanobind.cpp:9-13`. |
| Stats combine helper | Prefer `combine_welford_partials(...)` from `combine_welford.h:24-155` | The baseline never splits one logical row across cores (`work_split.hpp:46-52`, `moreh_norm_program_factory_w_other.cpp:52-59,151-189`), so there is no cross-core partial merge in Phase 2b | Use direct row-local `mean` and `mean(x^2)` accumulation with `reduce<..., Accumulate>` to keep CBs O(1) and avoid preloading whole rows. |
| TDD op name | Directory-derived `layernorm` would be fine | `tdd_orchestrator.py` derives `op_name = op_path.name` during `init` (`tt_metal/third_party/tt_ops_code_gen/scripts/tdd-pipeline/tdd_orchestrator.py:357-365`), but the public symbol is `layer_norm` | Keep `op_path = .../layernorm`, but set `.tdd_state.json["op_name"] = "layer_norm"` so future generated tests call the correct public op. |

## Planned Source Layout

| Planned file | Responsibility |
|--------------|----------------|
| `ttnn/cpp/ttnn/operations/normalization/layernorm/layernorm.hpp` | Public declarations for `ttnn::layer_norm(...)` and the compatibility sibling `ttnn::rms_norm(...)`. |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/layernorm.cpp` | Host wrappers, zero-volume fast path, option normalization, primitive forwarding. |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/layernorm_nanobind.hpp` | Binder declaration. |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/layernorm_nanobind.cpp` | `ttnn::bind_function` bindings for `layer_norm` and `rms_norm`, plus program-config/type nanobind exposure as needed. |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_types.hpp` | Shared enums/config structs: `LayerNormType`, `DistributedLayerNormStage`, `LayerNormDefaultProgramConfig`, `LayerNormShardedMultiCoreProgramConfig`, `LayerNormProgramConfig`, and helper declarations required by existing includes. |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_common.hpp` | Common utility declarations, `LayerNormParams`, `LayerNormInputs`, compatibility aliases, and primitive helper declarations (`create_layernorm_program_config`, default compute-config helpers). |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_utils.hpp` | Helper declarations for compute-kernel-config defaults and program-config resolution. |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_utils.cpp` | Implement `layernorm_default_compute_config`, `rmsnorm_default_compute_config`, and `create_layernorm_program_config`. |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_device_operation.hpp` | `LayerNormOperation` declaration, factory variant, validation/output/hash interfaces, primitive declarations. |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_device_operation.cpp` | Validation, output spec/allocation, explicit custom program hash, `ttnn::prim::layer_norm(...)` launch path. |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_program_factory.cpp` | Interleaved last-dim program construction, CB layout, kernel creation, runtime-arg override. |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_layernorm.cpp` | Two-pass reread reader: pass0 stats stream, pass1 normalize stream, optional residual/affine rereads, helper tile creation. |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/writer_layernorm.cpp` | Output writeback. |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_kernel.cpp` | Baseline interleaved compute kernel. |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_sfpu_kernel.cpp` | Same CB/arg contract as `layernorm_kernel.cpp`; compiled/selected for fp32-dest or later SFPU-specific paths. |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `epsilon` | `float` | yes | `> 0` | `1e-5f` | RT host attr, reader RT helper tile seed |
| `norm_type` | `LayerNormType` | primitive-only | `{LAYERNORM, RMSNORM}` | `LAYERNORM` in `ttnn::layer_norm`, `RMSNORM` in `ttnn::rms_norm` | CT compute flag |
| `distributed_stage` | `DistributedLayerNormStage` | primitive-only | Phase 2b accepts `NOT_DISTRIBUTED` only | `NOT_DISTRIBUTED` | host attr only |
| `memory_config` | `MemoryConfig` | no | output must be `INTERLEAVED` in Phase 2b | `input.memory_config()` | host attr only |
| `program_config` | `LayerNormProgramConfig` | no | Phase 2b accepts `LayerNormDefaultProgramConfig` only; sharded configs validate-reject | `create_layernorm_program_config(input.shard_spec())` | host attr only |
| `compute_kernel_config` | `DeviceComputeKernelConfig` | no | device-valid config only | `layernorm_default_compute_config(arch)` / `rmsnorm_default_compute_config(arch)` | host attr only |
| `dtype` | `std::optional<DataType>` | primitive-only | `BFLOAT16` or `FLOAT32` | `input.dtype()` | host attr only |

## Tensors

### Inputs

| Tensor | Required | Shape | Dtype | Layout | Memory | Notes |
|--------|----------|-------|-------|--------|--------|-------|
| `input` | yes | rank `> 0`; padded last two dims tile-aligned | `BFLOAT16` or `FLOAT32` | `TILE_LAYOUT` | device, allocated, `INTERLEAVED` | Logical last dim is the normalization axis. |
| `residual_input_tensor` | no | same logical and padded shape as `input` | same as `input` | `TILE_LAYOUT` | device, allocated, `INTERLEAVED` | Added before statistics and normalization. |
| `weight` | no | logical shape broadcastable to last dim; practical Phase 2b form is `[1,...,1,W]` with padded last dim matching input padded W | same as `input` | `TILE_LAYOUT` | device, allocated, `INTERLEAVED` | Streamed per width tile during pass1. |
| `bias` | no | same rule as `weight` | same as `input` | `TILE_LAYOUT` | device, allocated, `INTERLEAVED` | Applied after optional weight. |
| `recip_tensor` | no | reserved compatibility slot only | same as `input` | `TILE_LAYOUT` | device, allocated, `INTERLEAVED` | Phase 2b validates `recip_tensor == None`; keep the slot stable for future Welford/distributed extensions because `models/experimental/ops/descriptors/normalization/_utils.py:68-101` already expects it. |

### Output

| Property | Value |
|----------|-------|
| Shape | Same logical shape as `input`; stage-specific TDD reductions temporarily use one-tile-wide output, but final op returns full-shape output. |
| Dtype | `dtype.value_or(input.dtype())` |
| Layout | `TILE_LAYOUT` |
| Memory | `memory_config.value_or(input.memory_config())` |

## Validation and Hash Policy

| Area | Decision |
|------|----------|
| Tensor validation | Follow `BatchNormOperation`-style layered validation (`batch_norm_device_operation.cpp:33-96`) but replace rank-4/channel rules with last-dim rules: all tensors on-device, allocated, tiled, interleaved; optional tensors shape-compatible with input last dim; Phase 2b rejects sharded input/output and non-default distributed stages. |
| Output creation | Mirror `BatchNormOperation::compute_output_specs/create_output_tensors` (`batch_norm_device_operation.cpp:102-119`) with same-shape tiled output. |
| Program hash | Use a custom hash, not the default batch-norm tuple. Include `epsilon`, `norm_type`, `distributed_stage`, `memory_config`, `program_config`, `compute_kernel_config`, output dtype, input padded shape, input dtype/memory config, and optional presence + padded-shape/dtype/memory metadata for residual/weight/bias/recip. This avoids the shape ambiguity called out in the architecture and is more explicit than the generic runtime fallback in `device_operation.hpp:56-67`. |
| Program-factory selection | Keep an explicit `select_program_factory(...)` boundary returning `InterleavedLastDimFactory` in Phase 2b. Future sharded/distributed factories plug in here without disturbing the primitive surface. |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | One tile row across the full last dimension (`1 x Wt` tiles). |
| Grid | `device->compute_with_storage_grid_size()` |
| Per-core work | `num_units = outer_volume * Ht`; split via `tt::tt_metal::split_work_to_cores(grid, num_units)` (`work_split.hpp:46-52`). |
| Remainder | Use `core_group_1`/`core_group_2` from `split_work_to_cores`, exactly like `moreh_norm_program_factory_w_other.cpp:52-59,151-189`. |
| Runtime tile base | `tile_offset += num_rows_per_core * Wt` |
| Invariants | Never split one logical row across cores; both stats pass and normalize pass consume the same row ownership; last width tile is masked out of statistics and final output. |

## Circular Buffers

| CB ID | Name | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|-------|------|-----------|-----------|--------|----------|----------|----------|
| `c_0` | `input_pass0` | `tile_size(input_df)` | 2 | `input_df` | reader | compute | row-pass0 streaming |
| `c_1` | `residual_pass0` | `tile_size(input_df)` | 2 | `input_df` | reader | compute | row-pass0 streaming, optional |
| `c_2` | `input_pass1` | `tile_size(input_df)` | 2 | `input_df` | reader | compute | row-pass1 streaming |
| `c_3` | `residual_pass1` | `tile_size(input_df)` | 2 | `input_df` | reader | compute | row-pass1 streaming, optional |
| `c_4` | `weight_pass1` | `tile_size(input_df)` | 2 | `input_df` | reader | compute | row-pass1 streaming, optional |
| `c_5` | `bias_pass1` | `tile_size(input_df)` | 2 | `input_df` | reader | compute | row-pass1 streaming, optional |
| `c_6` | `reduce_scaler_inv_width` | `tile_size(bfloat16)` | 1 | `BFLOAT16` | reader | compute | whole-kernel persistent |
| `c_7` | `epsilon_scalar` | `tile_size(intermed_df)` | 1 | `intermed_df` | reader | compute | whole-kernel persistent |
| `c_8` | `mask_w` | `tile_size(input_df)` | 1 | `input_df` | reader | compute | whole-kernel persistent when `logical_W % 32 != 0` |
| `c_16` | `output` | `tile_size(output_df)` | 2 | `output_df` | compute | writer | row-pass1 streaming |
| `c_24` | `effective_tile` | `tile_size(intermed_df)` | 1 | `intermed_df` | compute | compute | tile-local |
| `c_25` | `square_tile` | `tile_size(intermed_df)` | 1 | `intermed_df` | compute | compute | pass0 tile-local |
| `c_26` | `mean_accum` | `tile_size(intermed_df)` | 1 | `intermed_df` | compute | compute | row-persistent until pass1 centering completes |
| `c_27` | `sqmean_accum` | `tile_size(intermed_df)` | 1 | `intermed_df` | compute | compute | row-persistent until `invstd` is derived |
| `c_28` | `invstd` | `tile_size(intermed_df)` | 1 | `intermed_df` | compute | compute | row-persistent through pass1 |
| `c_29` | `scratch0` | `tile_size(intermed_df)` | 1 | `intermed_df` | compute | compute | tile-local / stat-finalize scratch |
| `c_30` | `scratch1` | `tile_size(intermed_df)` | 1 | `intermed_df` | compute | compute | tile-local / affine scratch |

`input_df = datatype_to_dataformat_converter(input.dtype())`; `intermed_df = fp32_dest_acc_en ? Float32 : input_df`; `output_df = datatype_to_dataformat_converter(output_dtype)`. Two pages are only used on streamed reader/writer boundaries; every compute-owned intermediate stays O(1).

## API Mapping

| Phase | Type | Function | File:Line | Template Params / Args | CB In | CB Out | Requirements |
|-------|------|----------|-----------|-------------------------|-------|--------|--------------|
| host wrapper | helper | zero-volume clone pattern | `ttnn/cpp/ttnn/operations/normalization/batch_norm/batch_norm.cpp:50-57` | `ttnn::clone(input, std::nullopt, memory_config.value_or(input.memory_config()), std::nullopt)` | n/a | n/a | Preserve current normalization fast path for `logical_volume() == 0`. |
| primitive launch | helper | `ttnn::device_operation::launch` via batch-norm pattern | `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_device_operation.cpp:163-181` | resolved attrs + tensor args | n/a | n/a | Use a dedicated device op, not composite reductions. |
| cache identity | raw_api | default/custom device-operation hash contract | `ttnn/api/ttnn/device_operation.hpp:56-67` | custom layernorm hash still selected explicitly | n/a | n/a | Custom hash must include shape-sensitive fields for this op. |
| core split | helper | `split_work_to_cores` | `tt_metal/api/tt-metalium/work_split.hpp:46-52` | `(grid, num_units)` | n/a | n/a | One logical row stays on one core. |
| tensor accessors | raw_api | `TensorAccessorArgs::append_to(...)` | `tt_metal/api/tt-metalium/tensor_accessor_args.hpp:39-43` | append accessors last in stable slot order | n/a | n/a | Optional tensor slots stay stable even when absent. |
| stats reduce | helper | `compute_kernel_lib::reduce` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:424-438` | `PoolType::SUM`, `ReduceDim::REDUCE_ROW`, `ReduceInputBlockShape::single()`, `Accumulate::at(cb_accum, tile_idx)` | `c_24`/`c_25`, `c_6` | `c_26`/`c_27` | Streaming accumulation avoids `WaitUpfrontNoPop` whole-row CB sizing. |
| reduce accumulation | helper | `Accumulate::at` / accumulator reload | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:197-210`; `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl:68-81` | accumulator CB equals output CB for row-local rolling accumulation | `c_26`/`c_27` | same | Valid because accumulator tile is popped then replaced each iteration. |
| binary math | helper | `compute_kernel_lib::add/sub/mul/square` | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:225-314` | `BroadcastDim::{NONE,COL,ROW,SCALAR}` with per-input policies | `c_0..c_8`, `c_24..c_30` | `c_24..c_30`, `c_16` | Call `binary_op_init_common(...)` once up front. |
| reduce scaler tile | helper | `generate_reduce_scaler` | `ttnn/cpp/ttnn/kernel/dataflow/generate_reduce_scaler.hpp:12-40` | packed BF16 `1 / logical_W` | n/a | `c_6` | Keep scaler CB BF16 as required by the helper/TDD checklist. |
| epsilon helper tile | helper | `fill_cb_with_value` | `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp:128-139` | `epsilon_bits` repeated across tile | n/a | `c_7` | Use intermed format because the add+rsqrt phase happens in intermed precision. |
| tail mask tile | helper | `generate_mask_w` | `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp:253-290` | `mask_w = logical_W % 32` | n/a | `c_8` | Used twice: stats masking and final output masking. |
| copy / pack with fp32-dest safety | raw_api | `ckernel::copy_tile_init_with_dt` / `ckernel::pack_tile_with_dt` | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp:39-51` | single-tile copy fallback when optional paths are disabled | `c_24..c_30` | `c_24..c_30`, `c_16` | Required for safe copy paths under `FP32_DEST_ACC_EN`. |

## Compute Phases

| # | Operation | Helper? | Input CB (tiles, state) | Output CB (tiles) | CB State After |
|---|-----------|---------|-------------------------|-------------------|----------------|
| 1 | Seed `reduce_scaler`, `epsilon_scalar`, optional `mask_w` | helper | reader RT scalars | `c_6`, `c_7`, `c_8` | `c_6`/`c_7` persist for whole kernel; `c_8` persists only when tail exists. |
| 2 | Pass0 effective tile: `input (+ residual)` | helper/raw copy | `c_0`, optional `c_1` | `c_24` | `c_24` holds one effective tile. |
| 3 | Pass0 tail-mask last width tile | helper + copy fallback | `c_24`, optional `c_8` | `c_29` | `c_29` is the masked stats source. |
| 4 | Mean accumulation | helper | `c_29`, `c_6` | `c_26` | `c_26` always contains the rolling row mean tile after each width tile. |
| 5 | Square masked tile | helper | `c_29` | `c_25` | `c_25` holds one `x^2` tile. |
| 6 | Mean-square accumulation | helper | `c_25`, `c_6` | `c_27` | `c_27` always contains the rolling row `mean(x^2)` tile after each width tile. |
| 7 | Finalize row stats (`var = mean(x^2) - mean^2` or `sqmean` for RMS; then `rsqrt(var + eps)`) | helper + raw post-op | `c_26`, `c_27`, `c_7` | `c_28`, `c_29`, `c_30` | `c_26` keeps `mean`; `c_28` keeps `invstd`; `c_27` can be released once `invstd` is packed. |
| 8 | Pass1 effective tile reread: `input (+ residual)` | helper/raw copy | `c_2`, optional `c_3` | `c_24` | `c_24` holds the normalize-pass source tile. |
| 9 | Pass1 pre-mask last width tile | helper + copy fallback | `c_24`, optional `c_8` | `c_29` | `c_29` is the source for centering/RMS scaling. |
| 10 | Optional centering (`x - mean`) | helper | `c_29`, persistent `c_26` | `c_24` | Skip entirely for RMS-compatible primitive calls. |
| 11 | Multiply by `invstd` | helper | centered/source tile, persistent `c_28` | `c_29` | `c_29` holds normalized tile. |
| 12 | Optional weight row-broadcast | helper + copy fallback | `c_29`, optional `c_4` | `c_30` | `c_30` holds post-weight tile. |
| 13 | Optional bias row-broadcast | helper + copy fallback | `c_30`, optional `c_5` | `c_29` | `c_29` holds post-bias tile. |
| 14 | Final tail mask | helper + copy fallback | `c_29`, optional `c_8` | `c_16` | Padded columns are zeroed after all broadcasts. |
| 15 | Writer drains same-order output tiles | helper | `c_16` | output tensor | Row traversal order matches input traversal exactly. |

## Kernel Arguments

### Compile-Time

| Kernel | Index | Name | Type | Source/Formula |
|--------|-------|------|------|----------------|
| reader | 0 | `has_residual` | `uint32_t` | `tensor_args.residual_input_tensor.has_value()` |
| reader | 1 | `has_weight` | `uint32_t` | `tensor_args.weight.has_value()` |
| reader | 2 | `has_bias` | `uint32_t` | `tensor_args.bias.has_value()` |
| reader | 3.. | `input_accessor` | `TensorAccessorArgs` | `TensorAccessorArgs(*input.buffer())` |
| reader | next | `residual_accessor` | `TensorAccessorArgs` | `TensorAccessorArgs(residual ? residual->buffer().get() : nullptr)` |
| reader | next | `weight_accessor` | `TensorAccessorArgs` | `TensorAccessorArgs(weight ? weight->buffer().get() : nullptr)` |
| reader | next | `bias_accessor` | `TensorAccessorArgs` | `TensorAccessorArgs(bias ? bias->buffer().get() : nullptr)` |
| compute | 0 | `has_residual` | `uint32_t` | same as reader |
| compute | 1 | `has_weight` | `uint32_t` | same as reader |
| compute | 2 | `has_bias` | `uint32_t` | same as reader |
| compute | 3 | `is_rmsnorm` | `uint32_t` | `operation_attributes.norm_type == LayerNormType::RMSNORM` |
| writer | 0.. | `output_accessor` | `TensorAccessorArgs` | `TensorAccessorArgs(*output.buffer())` |

### Runtime

| Kernel | Index | Name | Type | Source/Formula |
|--------|-------|------|------|----------------|
| reader | 0 | `input_addr` | `uint32_t` | `input.buffer()->address()` |
| reader | 1 | `residual_addr` | `uint32_t` | `residual ? residual->buffer()->address() : 0` |
| reader | 2 | `weight_addr` | `uint32_t` | `weight ? weight->buffer()->address() : 0` |
| reader | 3 | `bias_addr` | `uint32_t` | `bias ? bias->buffer()->address() : 0` |
| reader | 4 | `num_rows_per_core` | `uint32_t` | `num_units_per_core` from `split_work_to_cores(...)` |
| reader | 5 | `Wt` | `uint32_t` | `input.padded_shape()[-1] / TILE_WIDTH` |
| reader | 6 | `tile_offset` | `uint32_t` | running base tile for this core |
| reader | 7 | `logical_W` | `uint32_t` | `input.logical_shape()[-1]` |
| reader | 8 | `inv_width_bf16_packed` | `uint32_t` | packed BF16 pair for `1.0f / logical_W` |
| reader | 9 | `epsilon_bits` | `uint32_t` | `bit_cast<uint32_t>(epsilon)` |
| compute | 0 | `num_rows_per_core` | `uint32_t` | same as reader |
| compute | 1 | `Wt` | `uint32_t` | same as reader |
| compute | 2 | `logical_W` | `uint32_t` | same as reader |
| writer | 0 | `output_addr` | `uint32_t` | `output.buffer()->address()` |
| writer | 1 | `num_rows_per_core` | `uint32_t` | same as reader |
| writer | 2 | `Wt` | `uint32_t` | same as reader |
| writer | 3 | `tile_offset` | `uint32_t` | same as reader |

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|----|-------------------|-------------------|---------------|
| pass0 tail mask | `mul` | All | All (`mask_w` is a full 0/1 tile) | `NONE` |
| centering | `sub` | All | `REDUCE_ROW` result in `c_26` => Col0 | `COL` |
| invstd apply | `mul` | All | `REDUCE_ROW` result in `c_28` => Col0 | `COL` |
| affine weight | `mul` | All | weight tile => Row0 | `ROW` |
| affine bias | `add` | All | bias tile => Row0 | `ROW` |
| final tail mask | `mul` | All | All (`mask_w` tile) | `NONE` |

## Kernel Implementations

### Reader

```cpp
// reader_layernorm.cpp
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/kernel/dataflow/moreh_common.hpp"

void kernel_main() {
    constexpr bool has_residual = get_compile_time_arg_val(0) == 1;
    constexpr bool has_weight = get_compile_time_arg_val(1) == 1;
    constexpr bool has_bias = get_compile_time_arg_val(2) == 1;

    constexpr auto input_args = TensorAccessorArgs<3>();
    constexpr auto residual_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto weight_args = TensorAccessorArgs<residual_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();

    int i = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(i++);
    const uint32_t residual_addr = get_arg_val<uint32_t>(i++);
    const uint32_t weight_addr = get_arg_val<uint32_t>(i++);
    const uint32_t bias_addr = get_arg_val<uint32_t>(i++);
    const uint32_t num_rows_per_core = get_arg_val<uint32_t>(i++);
    const uint32_t Wt = get_arg_val<uint32_t>(i++);
    const uint32_t tile_offset = get_arg_val<uint32_t>(i++);
    const uint32_t logical_W = get_arg_val<uint32_t>(i++);
    const uint32_t inv_width_bf16_packed = get_arg_val<uint32_t>(i++);
    const uint32_t epsilon_bits = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_input_p0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_residual_p0 = tt::CBIndex::c_1;
    constexpr uint32_t cb_input_p1 = tt::CBIndex::c_2;
    constexpr uint32_t cb_residual_p1 = tt::CBIndex::c_3;
    constexpr uint32_t cb_weight = tt::CBIndex::c_4;
    constexpr uint32_t cb_bias = tt::CBIndex::c_5;
    constexpr uint32_t cb_reduce_scaler = tt::CBIndex::c_6;
    constexpr uint32_t cb_epsilon = tt::CBIndex::c_7;
    constexpr uint32_t cb_mask_w = tt::CBIndex::c_8;

    const uint32_t input_tile_bytes = get_tile_size(cb_input_p0);
    const auto input = TensorAccessor(input_args, input_addr, input_tile_bytes);
    const auto residual = TensorAccessor(residual_args, residual_addr, input_tile_bytes);
    const auto weight = TensorAccessor(weight_args, weight_addr, input_tile_bytes);
    const auto bias = TensorAccessor(bias_args, bias_addr, input_tile_bytes);

    generate_reduce_scaler(cb_reduce_scaler, inv_width_bf16_packed);
    fill_cb_with_value(cb_epsilon, epsilon_bits);

    constexpr uint32_t TILE_W = 32;
    const bool do_mask_w = (logical_W % TILE_W) != 0;
    const uint32_t mask_w = do_mask_w ? (logical_W % TILE_W) : TILE_W;
    if (do_mask_w) {
        generate_mask_w(cb_mask_w, mask_w);
    }

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        const uint32_t row_tile_base = tile_offset + row * Wt;

        // Pass0: stats reread
        for (uint32_t col = 0; col < Wt; ++col) {
            noc_async_read_tile_helper(cb_input_p0, 1, row_tile_base + col, input);
            if constexpr (has_residual) {
                noc_async_read_tile_helper(cb_residual_p0, 1, row_tile_base + col, residual);
            }
        }

        // Pass1: normalize reread + optional affine rereads
        for (uint32_t col = 0; col < Wt; ++col) {
            noc_async_read_tile_helper(cb_input_p1, 1, row_tile_base + col, input);
            if constexpr (has_residual) {
                noc_async_read_tile_helper(cb_residual_p1, 1, row_tile_base + col, residual);
            }
            if constexpr (has_weight) {
                noc_async_read_tile_helper(cb_weight, 1, col, weight);
            }
            if constexpr (has_bias) {
                noc_async_read_tile_helper(cb_bias, 1, col, bias);
            }
        }
    }
}
```

### Compute

```cpp
// layernorm_kernel.cpp / layernorm_sfpu_kernel.cpp
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace {
ALWI void copy_tile_to_cb(uint32_t src_cb, uint32_t dst_cb, bool pop_src = true) {
    constexpr uint32_t onetile = 1;
    tile_regs_acquire();
    cb_wait_front(src_cb, onetile);
    cb_reserve_back(dst_cb, onetile);
    ckernel::copy_tile_init_with_dt(src_cb);
    copy_tile(src_cb, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    ckernel::pack_tile_with_dt(0, dst_cb);
    tile_regs_release();
    if (pop_src) {
        cb_pop_front(src_cb, onetile);
    }
    cb_push_back(dst_cb, onetile);
}
}  // namespace

void kernel_main() {
    constexpr bool has_residual = get_compile_time_arg_val(0) == 1;
    constexpr bool has_weight = get_compile_time_arg_val(1) == 1;
    constexpr bool has_bias = get_compile_time_arg_val(2) == 1;
    constexpr bool is_rmsnorm = get_compile_time_arg_val(3) == 1;

    int i = 0;
    const uint32_t num_rows_per_core = get_arg_val<uint32_t>(i++);
    const uint32_t Wt = get_arg_val<uint32_t>(i++);
    const uint32_t logical_W = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_input_p0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_residual_p0 = tt::CBIndex::c_1;
    constexpr uint32_t cb_input_p1 = tt::CBIndex::c_2;
    constexpr uint32_t cb_residual_p1 = tt::CBIndex::c_3;
    constexpr uint32_t cb_weight = tt::CBIndex::c_4;
    constexpr uint32_t cb_bias = tt::CBIndex::c_5;
    constexpr uint32_t cb_reduce_scaler = tt::CBIndex::c_6;
    constexpr uint32_t cb_epsilon = tt::CBIndex::c_7;
    constexpr uint32_t cb_mask_w = tt::CBIndex::c_8;
    constexpr uint32_t cb_output = tt::CBIndex::c_16;
    constexpr uint32_t cb_effective = tt::CBIndex::c_24;
    constexpr uint32_t cb_square = tt::CBIndex::c_25;
    constexpr uint32_t cb_mean = tt::CBIndex::c_26;
    constexpr uint32_t cb_sqmean = tt::CBIndex::c_27;
    constexpr uint32_t cb_invstd = tt::CBIndex::c_28;
    constexpr uint32_t cb_scratch0 = tt::CBIndex::c_29;
    constexpr uint32_t cb_scratch1 = tt::CBIndex::c_30;

    compute_kernel_hw_startup(cb_input_p0, cb_reduce_scaler, cb_output);
    binary_op_init_common(cb_input_p0, cb_input_p0, cb_output);

    constexpr uint32_t TILE_W = 32;
    const bool do_mask_w = (logical_W % TILE_W) != 0;

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        // Pass0: accumulate mean and mean(x^2) tile-by-tile with O(1) CBs.
        for (uint32_t col = 0; col < Wt; ++col) {
            if constexpr (has_residual) {
                compute_kernel_lib::add(
                    cb_input_p0, cb_residual_p0, cb_effective, compute_kernel_lib::BinaryInputBlockShape::single());
            } else {
                copy_tile_to_cb(cb_input_p0, cb_effective);
            }

            if (do_mask_w && col == Wt - 1) {
                compute_kernel_lib::mul<
                    compute_kernel_lib::BroadcastDim::NONE,
                    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
                    cb_effective, cb_mask_w, cb_scratch0, compute_kernel_lib::BinaryInputBlockShape::single());
            } else {
                copy_tile_to_cb(cb_effective, cb_scratch0);
            }

            compute_kernel_lib::reduce<
                PoolType::SUM,
                ReduceDim::REDUCE_ROW,
                compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile>(
                cb_scratch0,
                cb_reduce_scaler,
                cb_mean,
                compute_kernel_lib::ReduceInputBlockShape::single(),
                {},
                compute_kernel_lib::Accumulate::at(cb_mean, col));

            compute_kernel_lib::square(cb_scratch0, cb_square, compute_kernel_lib::BinaryInputBlockShape::single());
            compute_kernel_lib::reduce<
                PoolType::SUM,
                ReduceDim::REDUCE_ROW,
                compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile>(
                cb_square,
                cb_reduce_scaler,
                cb_sqmean,
                compute_kernel_lib::ReduceInputBlockShape::single(),
                {},
                compute_kernel_lib::Accumulate::at(cb_sqmean, col));
        }

        // Finalize row statistics.
        if constexpr (!is_rmsnorm) {
            compute_kernel_lib::square<
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
                cb_mean, cb_scratch0, compute_kernel_lib::BinaryInputBlockShape::single());
            compute_kernel_lib::sub<
                compute_kernel_lib::BroadcastDim::NONE,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
                cb_sqmean, cb_scratch0, cb_scratch1, compute_kernel_lib::BinaryInputBlockShape::single());
        } else {
            copy_tile_to_cb(cb_sqmean, cb_scratch1);
        }

        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_scratch1,
            cb_epsilon,
            cb_invstd,
            compute_kernel_lib::BinaryInputBlockShape::single(),
            [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // Pass1: reread, normalize, affine, and emit output.
        for (uint32_t col = 0; col < Wt; ++col) {
            if constexpr (has_residual) {
                compute_kernel_lib::add(
                    cb_input_p1, cb_residual_p1, cb_effective, compute_kernel_lib::BinaryInputBlockShape::single());
            } else {
                copy_tile_to_cb(cb_input_p1, cb_effective);
            }

            if (do_mask_w && col == Wt - 1) {
                compute_kernel_lib::mul<
                    compute_kernel_lib::BroadcastDim::NONE,
                    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
                    cb_effective, cb_mask_w, cb_scratch0, compute_kernel_lib::BinaryInputBlockShape::single());
            } else {
                copy_tile_to_cb(cb_effective, cb_scratch0);
            }

            if constexpr (!is_rmsnorm) {
                compute_kernel_lib::sub<
                    compute_kernel_lib::BroadcastDim::COL,
                    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
                    cb_scratch0, cb_mean, cb_effective, compute_kernel_lib::BinaryInputBlockShape::single());
            } else {
                copy_tile_to_cb(cb_scratch0, cb_effective);
            }

            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::COL,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
                cb_effective, cb_invstd, cb_scratch0, compute_kernel_lib::BinaryInputBlockShape::single());

            if constexpr (has_weight) {
                compute_kernel_lib::mul<
                    compute_kernel_lib::BroadcastDim::ROW>(
                    cb_scratch0, cb_weight, cb_scratch1, compute_kernel_lib::BinaryInputBlockShape::single());
            } else {
                copy_tile_to_cb(cb_scratch0, cb_scratch1);
            }

            if constexpr (has_bias) {
                compute_kernel_lib::add<
                    compute_kernel_lib::BroadcastDim::ROW>(
                    cb_scratch1, cb_bias, cb_scratch0, compute_kernel_lib::BinaryInputBlockShape::single());
            } else {
                copy_tile_to_cb(cb_scratch1, cb_scratch0);
            }

            if (do_mask_w && col == Wt - 1) {
                compute_kernel_lib::mul<
                    compute_kernel_lib::BroadcastDim::NONE,
                    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
                    cb_scratch0, cb_mask_w, cb_output, compute_kernel_lib::BinaryInputBlockShape::single());
            } else {
                copy_tile_to_cb(cb_scratch0, cb_output);
            }
        }

        // Row-local persistent stats consumed; pop once here before the next row starts.
        cb_wait_front(cb_mean, 1);
        cb_pop_front(cb_mean, 1);
        cb_wait_front(cb_invstd, 1);
        cb_pop_front(cb_invstd, 1);
    }

    cb_wait_front(cb_reduce_scaler, 1);
    cb_pop_front(cb_reduce_scaler, 1);
    cb_wait_front(cb_epsilon, 1);
    cb_pop_front(cb_epsilon, 1);
    if (do_mask_w) {
        cb_wait_front(cb_mask_w, 1);
        cb_pop_front(cb_mask_w, 1);
    }
}
```

### Writer

```cpp
// writer_layernorm.cpp
#include "ttnn/kernel/dataflow/moreh_common.hpp"

void kernel_main() {
    constexpr auto output_args = TensorAccessorArgs<0>();

    int i = 0;
    const uint32_t output_addr = get_arg_val<uint32_t>(i++);
    const uint32_t num_rows_per_core = get_arg_val<uint32_t>(i++);
    const uint32_t Wt = get_arg_val<uint32_t>(i++);
    const uint32_t tile_offset = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_output = tt::CBIndex::c_16;
    const uint32_t output_tile_bytes = get_tile_size(cb_output);
    const auto output = TensorAccessor(output_args, output_addr, output_tile_bytes);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        const uint32_t row_tile_base = tile_offset + row * Wt;
        for (uint32_t col = 0; col < Wt; ++col) {
            noc_async_write_tile_helper(cb_output, 1, row_tile_base + col, output);
        }
    }
}
```

## TDD Stages

| # | Name | Type | Scope (kernels modified) | Reference Expression | Shapes | rtol | atol |
|---|------|------|--------------------------|---------------------|--------|------|------|
| 1 | `data_pipeline` | `implementation` | `reader_layernorm.cpp`, `writer_layernorm.cpp`, `layernorm_kernel.cpp`, `layernorm_sfpu_kernel.cpp` | `return input_tensor` | `(1, 1, 32, 32)`, `(1, 1, 32, 96)`, `(2, 3, 64, 70)` | `0.0` | `0.0` |
| 2 | `mean_reduce` | `implementation` | `layernorm_kernel.cpp`, `layernorm_sfpu_kernel.cpp` | `return input_tensor.float().mean(dim=-1, keepdim=True)` | `(1, 1, 32, 32)`, `(1, 1, 32, 96)`, `(2, 3, 64, 70)` | `0.05` | `0.2` |
| 3 | `invstd_reduce` | `implementation` | `layernorm_kernel.cpp`, `layernorm_sfpu_kernel.cpp` | `mean = input.mean(-1, keepdim=True); var = ((input - mean) ** 2).mean(-1, keepdim=True); return torch.rsqrt(var + 1e-5)` | `(1, 1, 32, 32)`, `(1, 1, 32, 96)`, `(2, 3, 64, 70)` | `0.05` | `0.2` |
| 4 | `normalize` | `implementation` | `layernorm_kernel.cpp`, `layernorm_sfpu_kernel.cpp` | `return torch.nn.functional.layer_norm(input.float(), (input.shape[-1],), weight=None, bias=None, eps=1e-5)` | `(1, 1, 32, 32)`, `(1, 1, 32, 96)`, `(2, 3, 64, 70)` | `0.05` | `0.2` |
| 5 | `residual_affine` | `implementation` | `reader_layernorm.cpp`, `layernorm_kernel.cpp`, `layernorm_sfpu_kernel.cpp` | `x = input + residual; return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight=weight, bias=bias, eps=1e-5)` | `(1, 1, 32, 32)`, `(1, 1, 32, 96)`, `(2, 3, 64, 70)` | `0.05` | `0.2` |
| 6 | `acceptance` | `acceptance` | none | cross-product over residual/weight/bias/tail-width/epsilon on the final public `layer_norm` wrapper | `(1, 1, 32, 32)`, `(1, 1, 32, 96)`, `(2, 3, 64, 70)`, `(1, 2, 96, 128)` | `0.05` | `0.2` |

## Assumptions

| Assumption | Why it is acceptable in Phase 2b |
|------------|----------------------------------|
| Interleaved TILE tensors only | Matches the nearest local normalization pattern (`batch_norm_device_operation.cpp:64-92`) and keeps the first kernel family bounded. |
| Two-pass reread baseline | Avoids `WaitUpfrontNoPop` whole-row CB sizing for wide hidden dims while preserving row-local correctness. |
| `LayerNormDefaultProgramConfig` owns Phase 2b | Existing demos/models already reference both default and sharded program-config symbols, but only the default interleaved path is engineered in this phase. |
| Compatibility type surface can be broader than the initial executable subset | Existing includes and descriptors already reference missing type/program-config symbols; adding them now avoids another API churn later. |

## Risks / Unknowns

| Risk | Impact | Mitigation |
|------|--------|------------|
| Row-by-row two-pass reread doubles input/residual bandwidth | Lower throughput than a future staged-row or sharded implementation | Accept for baseline correctness; preserve explicit factory split for later optimization. |
| Descriptor-facing compatibility surface is wider than the runtime op scope (`models/experimental/ops/descriptors/normalization/_utils.py:16-114`) | Some descriptor-only callers may still need follow-on bindings/helpers even after the runtime op lands | Keep the type/helper names stable now; treat descriptor plumbing as an adjacent follow-on if it is not folded into the same patch. |
| Sharded program configs are referenced broadly (`models/common/modules/rmsnorm/rmsnorm_1d.py:604-621`, `models/demos/stable_diffusion_xl_base/refiner/tt/model_configs/model_configs_1024x1024.py:1083`) | Users may expect more than the baseline default config | Expose the symbols in `layernorm_types.hpp`, but validate-reject non-default configs in `LayerNormOperation` until a sharded factory exists. |

## Hardware Constraints

- [x] CB sync: push count = wait count for every CB
- [x] Reduce scaler CB is bfloat16
- [x] DEST: max 8 tiles (bf16) / 4 tiles (f32)
- [x] Sequential helper intermediates sized to a single row-unit tile, not a full-width preload
- [x] Page sizes aligned to tile size
- [x] RM CBs are not used in this Phase 2b kernel family
- [x] Optional tensor accessor slots remain stable even when tensors are absent
