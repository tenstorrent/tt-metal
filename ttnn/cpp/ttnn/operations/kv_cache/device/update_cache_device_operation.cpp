// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
        cache_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
            cache_tensor.memory_config().memory_layout() == TensorMemoryLayout::ND_SHARDED ||
            (cache_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
             cache_tensor.memory_config().created_with_nd_shard_spec() &&
             cache_tensor.memory_config()
                 .is_dram()),  // ND_SHARDED layout can collapse to HEIGHT_SHARDED when each bank holds single shard
        "Cache tensor memory layout must be INTERLEAVED, ND_SHARDED, or HEIGHT_SHARDED (created from ND shard spec) "
        "but got {}",
        cache_tensor.memory_config().memory_layout());
    if (args.op_type == UpdateCacheOpType::FILL) {
        // TODO: If we want to support mixed precision like decode, we need to add simple compute kernel for conversion
        TT_FATAL(input_tensor.dtype() == cache_tensor.dtype(), "Input and cache tensors must have same dtype!");

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
            args.update_idx % TILE_HEIGHT == 0,
            "Fill update_idx ({}) must be a multiple of TILE_HEIGHT ({})",
            args.update_idx,
            TILE_HEIGHT);
        const uint32_t fill_rows =
            tensor_args.input_row_offset.has_value() ? args.num_input_rows : input_tensor.padded_shape()[-2];
        TT_FATAL(
            args.update_idx <= cache_tensor.padded_shape()[-2] &&
                fill_rows <= cache_tensor.padded_shape()[-2] - args.update_idx,
            "Fill update_idx ({}) + rows filled ({}) must be <= cache tensor height ({})",
            args.update_idx,
            fill_rows,
            cache_tensor.padded_shape()[-2]);
        if (tensor_args.input_row_offset.has_value()) {
            const auto& offset = tensor_args.input_row_offset.value();
            TT_FATAL(
                offset.storage_type() == StorageType::DEVICE && offset.buffer() != nullptr,
                "fill_cache: input_row_offset_tensor must be allocated on device");
            TT_FATAL(
                (offset.dtype() == DataType::INT32 || offset.dtype() == DataType::UINT32) &&
                    offset.layout() == Layout::ROW_MAJOR && offset.logical_shape().volume() == 1 &&
                    offset.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
                "fill_cache: input_row_offset_tensor must be one INT32 / UINT32 row-major interleaved value");
            TT_FATAL(
                !input_tensor.is_sharded() && input_tensor.padded_shape()[1] == 1,
                "fill_cache: input_row_offset_tensor needs an interleaved input with one head, got {}",
                input_tensor.padded_shape());
            TT_FATAL(
                args.num_input_rows > 0 && args.num_input_rows % TILE_HEIGHT == 0 &&
                    args.num_input_rows <= input_tensor.padded_shape()[-2],
                "fill_cache: num_input_rows ({}) must be a positive multiple of {} and at most the input height ({})",
                args.num_input_rows,
                TILE_HEIGHT,
                input_tensor.padded_shape()[-2]);
        } else {
            TT_FATAL(args.num_input_rows == 0, "fill_cache: num_input_rows needs input_row_offset_tensor");
        }
    } else if (args.op_type == UpdateCacheOpType::UPDATE) {
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

tt::tt_metal::TensorSpec UpdateKVCacheOperation::compute_output_specs(
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
        args.op_type,
        args.num_input_rows,
        std::vector<Tensor>{tensor_args.cache, tensor_args.input},
        tensor_args.input_row_offset);
}

Tensor update_cache(
    const Tensor& cache,
    const Tensor& input,
    const uint32_t batch_idx,
    const uint32_t update_index,
    const uint32_t batch_offset,
    const UpdateCacheOpType op_type,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<Tensor>& input_row_offset,
    uint32_t num_input_rows) {
    return ttnn::device_operation::launch<UpdateKVCacheOperation>(
        KvCacheParams{
            .batch_idx = batch_idx,
            .update_idx = update_index,
            .batch_offset = batch_offset,
            .op_type = op_type,
            .compute_kernel_config = compute_kernel_config,
            .num_input_rows = num_input_rows},
        KvCacheInputs{.cache = cache, .input = input, .input_row_offset = input_row_offset});
}

}  // namespace ttnn::prim
