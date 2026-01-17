// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_concat_heads_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::nlp_concat_heads {

NLPConcatHeadsDeviceOperation::program_factory_t NLPConcatHeadsDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return program::NLPConcatHeadsProgramFactory{};
}

void NLPConcatHeadsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void NLPConcatHeadsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;

    TT_FATAL(input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor.dtype() == tt::tt_metal::DataType::FLOAT32 ||
            input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format");
    TT_FATAL(
        input_tensor.layout() == tt::tt_metal::Layout::TILE,
        "Input tensor layout must be TILE but got {}",
        input_tensor.layout());

    if (input_tensor.is_sharded()) {
        TT_FATAL(
            input_tensor.memory_config().memory_layout() != tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
            "Input tensor memory layout must not be WIDTH_SHARDED but got {}",
            input_tensor.memory_config().memory_layout());
        auto shard_spec = input_tensor.shard_spec().value();
        TT_FATAL(
            shard_spec.shape[1] == input_tensor.padded_shape()[-1],
            "Input tensor shard width ({}) must equal padded width ({})",
            shard_spec.shape[1],
            input_tensor.padded_shape()[-1]);
        TT_FATAL(
            shard_spec.shape[0] % input_tensor.padded_shape()[-2] == 0,
            "Input tensor shard height ({}) must be divisible by padded height ({})",
            shard_spec.shape[0],
            input_tensor.padded_shape()[-2]);
        TT_FATAL(
            input_tensor.padded_shape()[1] % (shard_spec.shape[0] / input_tensor.padded_shape()[-2]) == 0,
            "Input tensor padded height ({}) must be divisible by shard height / padded height ({} / {})",
            input_tensor.padded_shape()[1],
            shard_spec.shape[0],
            input_tensor.padded_shape()[-2]);
        TT_FATAL(
            args.output_mem_config.memory_layout() != tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
            "Output memory config layout must not be HEIGHT_SHARDED but got {}",
            args.output_mem_config.memory_layout());
    } else {
        TT_FATAL(
            args.output_mem_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
            "Output memory config layout must be INTERLEAVED but got {}",
            args.output_mem_config.memory_layout());
    }
}

spec_return_value_t NLPConcatHeadsDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_shape = input_tensor.logical_shape();

    auto num_heads = input_shape[1];
    auto sequence_length = input_shape[2];
    auto head_dim = input_shape[3];

    auto hidden_dim = num_heads * head_dim;

    Shape output_shape({input_shape[0], 1, sequence_length, hidden_dim});

    if (args.output_mem_config.is_sharded()) {
        tt::tt_metal::ShardSpec shard_spec = input_tensor.shard_spec().value();
        uint32_t heads_per_shard = shard_spec.shape[0] / input_tensor.padded_shape()[-2];
        shard_spec.shape = {shard_spec.shape[0] / heads_per_shard, shard_spec.shape[1] * heads_per_shard};
        auto mem_config = args.output_mem_config.with_shard_spec(shard_spec);
        return TensorSpec(
            output_shape,
            tt::tt_metal::TensorLayout(
                input_tensor.dtype(), tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), mem_config));
    }

    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), args.output_mem_config));
}

tensor_return_value_t NLPConcatHeadsDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), input_tensor.device());
}

}  // namespace ttnn::operations::experimental::nlp_concat_heads

namespace ttnn::prim {

ttnn::operations::experimental::nlp_concat_heads::tensor_return_value_t nlp_concat_heads(
    const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    using OperationType = ttnn::operations::experimental::nlp_concat_heads::NLPConcatHeadsDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .output_mem_config = memory_config.value_or(input_tensor.memory_config()),
    };
    auto tensor_args = OperationType::tensor_args_t{.input = input_tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
