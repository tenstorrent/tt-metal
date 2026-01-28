// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "update_cache_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

using namespace tt::constants;

UpdateKVCacheOperation::program_factory_t UpdateKVCacheOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& /*tensor_args*/) {
    if (args.op_type == UpdateCacheOpType::FILL) {
        return FillCacheMultiCoreProgramFactory{};
    }
    return UpdateCacheMultiCoreProgramFactory{};
}

void UpdateKVCacheOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& cache_tensor = tensor_args.cache;
    const auto& input_tensor = tensor_args.input;
    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE and cache_tensor.storage_type() == StorageType::DEVICE,
        "Operands to update_cache need to be on device!");
    TT_FATAL(input_tensor.device() == cache_tensor.device(), "Operands to update_cache need to be on the same device!");
    TT_FATAL(
        input_tensor.buffer() != nullptr and cache_tensor.buffer() != nullptr,
        "Operands to update_cache need to be allocated in buffers on device!");
    TT_FATAL(
        (input_tensor.layout() == Layout::TILE && cache_tensor.layout() == Layout::TILE),
        "Inputs to update_cache must be tilized");
    TT_FATAL(
        input_tensor.dtype() == DataType::FLOAT32 || input_tensor.dtype() == DataType::BFLOAT16 ||
            input_tensor.dtype() == DataType::BFLOAT8_B,
        "Input tensor dtype must be FLOAT32, BFLOAT16, or BFLOAT8_B but got {}",
        input_tensor.dtype());
    TT_FATAL(
        cache_tensor.dtype() == DataType::FLOAT32 || cache_tensor.dtype() == DataType::BFLOAT16 ||
            cache_tensor.dtype() == DataType::BFLOAT8_B,
        "Cache tensor dtype must be FLOAT32, BFLOAT16, or BFLOAT8_B but got {}",
        cache_tensor.dtype());

    TT_FATAL(
        input_tensor.padded_shape()[-1] == cache_tensor.padded_shape()[-1],
        "Input tensor width ({}) must equal cache tensor width ({})",
        input_tensor.padded_shape()[-1],
        cache_tensor.padded_shape()[-1]);
    TT_FATAL(
        input_tensor.padded_shape()[0] == 1,
        "Input tensor batch size must be 1 but got {}",
        input_tensor.padded_shape()[0]);
    TT_FATAL(
        input_tensor.padded_shape()[1] == cache_tensor.padded_shape()[1],
        "Input tensor height ({}) must equal cache tensor height ({})",
        input_tensor.padded_shape()[1],
        cache_tensor.padded_shape()[1]);
    TT_FATAL(
        cache_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Cache tensor memory layout must be INTERLEAVED but got {}",
        cache_tensor.memory_config().memory_layout());
    if (args.op_type == UpdateCacheOpType::FILL) {
        // TODO: If we want to support mixed precision like decode, we need to add simple compute kernel for conversion
        TT_FATAL(input_tensor.dtype() == cache_tensor.dtype(), "Input and cache tensors must have same dtype!");

        // TODO: For interleaved, assume each core handles 1 tile of seq_len if kv_heads > 1
        // For 56 cores and 2 heads, this effectively caps max seq len at 56 / 2 * 32 = 896
        // Can generalize interleaved to infer and check arbitrary number of tiles along seq_len per core; or, add more
        // robust logic in reader/writer loops to handle generic blocking of work For sharded, we infer number of tiles
        // each core handles from shard so no issues there
        if (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED and
            input_tensor.padded_shape()[1] > 1) {
            const uint32_t num_blocks_of_work =
                input_tensor.padded_shape()[1] * input_tensor.padded_shape()[-2] / TILE_HEIGHT;
            const auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
            TT_FATAL(
                (num_blocks_of_work <= compute_with_storage_grid_size.x * compute_with_storage_grid_size.y),
                "Number of work blocks ({}) must be <= total grid size ({})",
                num_blocks_of_work,
                compute_with_storage_grid_size.x * compute_with_storage_grid_size.y);
        }

        if (input_tensor.is_sharded()) {
            TT_FATAL(
                input_tensor.memory_config().memory_layout() != TensorMemoryLayout::WIDTH_SHARDED,
                "Input tensor memory layout must not be WIDTH_SHARDED but got {}",
                input_tensor.memory_config().memory_layout());
            TT_FATAL(
                input_tensor.shard_spec().value().shape[1] == input_tensor.padded_shape()[-1],
                "Input tensor shard width ({}) must equal padded width ({})",
                input_tensor.shard_spec().value().shape[1],
                input_tensor.padded_shape()[-1]);
            // Require even work division along seq_len and also only 1 head per core
            TT_FATAL(
                input_tensor.padded_shape()[-2] % input_tensor.shard_spec().value().shape[0] == 0,
                "Seq len must be divisible by shard height!");
        }

        TT_FATAL(
            args.batch_idx < cache_tensor.padded_shape()[0],
            "Batch index ({}) must be less than cache tensor batch size ({})",
            args.batch_idx,
            cache_tensor.padded_shape()[0]);
        TT_FATAL(
            input_tensor.padded_shape()[-2] <= cache_tensor.padded_shape()[-2],
            "Input tensor height ({}) must be <= cache tensor height ({})",
            input_tensor.padded_shape()[-2],
            cache_tensor.padded_shape()[-2]);
    } else if (args.op_type == UpdateCacheOpType::UPDATE) {
        if (input_tensor.device()->arch() == tt::ARCH::GRAYSKULL) {
            TT_FATAL(
                cache_tensor.dtype() == DataType::BFLOAT16,
                "#12931: Update Cache currently produces non-deterministic output on GS when converting data types for "
                "cache tensor");
        }
        if (input_tensor.is_sharded()) {
            TT_FATAL(
                input_tensor.memory_config().memory_layout() != TensorMemoryLayout::WIDTH_SHARDED,
                "Input tensor memory layout must not be WIDTH_SHARDED but got {}",
                input_tensor.memory_config().memory_layout());
            TT_FATAL(
                input_tensor.shard_spec().value().shape[1] == input_tensor.padded_shape()[-1],
                "Input tensor shard width ({}) must equal padded width ({})",
                input_tensor.shard_spec().value().shape[1],
                input_tensor.padded_shape()[-1]);
            // Require even work division for now
            TT_FATAL(
                (input_tensor.physical_volume() / input_tensor.padded_shape()[-1]) %
                        input_tensor.shard_spec().value().shape[0] ==
                    0,
                "Error");
        } else {
            TT_FATAL(
                input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
                "Input tensor memory layout must be INTERLEAVED but got {}",
                input_tensor.memory_config().memory_layout());
        }
        TT_FATAL(
            cache_tensor.padded_shape()[0] <= input_tensor.padded_shape()[-2],
            "Cache tensor batch size ({}) must be <= input tensor height ({})",
            cache_tensor.padded_shape()[0],
            input_tensor.padded_shape()[-2]);
        // batch offset is only valid if num_user less than 32 and batch_offset + num_user <= 32
        if (cache_tensor.padded_shape()[0] < 32) {
            TT_FATAL(
                args.batch_offset + cache_tensor.padded_shape()[0] <= 32,
                "Batch offset ({}) + cache tensor batch size ({}) must be <= 32",
                args.batch_offset,
                cache_tensor.padded_shape()[0]);
        } else {
            TT_FATAL(args.batch_offset == 0, "Batch offset must be 0 when cache tensor batch size >= 32");
        }
    }
}

void UpdateKVCacheOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

TensorSpec UpdateKVCacheOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    // Do nothing because it's an in-place operation. Cache Tensor is the output tensor.
    return tensor_args.cache.tensor_spec();
}

Tensor UpdateKVCacheOperation::create_output_tensors(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    // Do nothing because it's an in-place operation. Cache Tensor is the output tensor.
    return tensor_args.cache;
}

tt::tt_metal::operation::Hash UpdateKVCacheOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<UpdateKVCacheOperation>(
        args.op_type, std::vector<Tensor>{tensor_args.cache, tensor_args.input});
}

Tensor update_cache(
    const Tensor& cache,
    const Tensor& input,
    const uint32_t batch_idx,
    const uint32_t update_index,
    const uint32_t batch_offset,
    const UpdateCacheOpType op_type,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    return ttnn::device_operation::launch<UpdateKVCacheOperation>(
        KvCacheParams{
            .batch_idx = batch_idx,
            .update_idx = update_index,
            .batch_offset = batch_offset,
            .op_type = op_type,
            .compute_kernel_config = compute_kernel_config},
        KvCacheInputs{.cache = cache, .input = input});
}

}  // namespace ttnn::prim
