// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_concat_heads_boltz_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::nlp_concat_heads_boltz {

NLPConcatHeadsBoltzDeviceOperation::program_factory_t NLPConcatHeadsBoltzDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return NLPConcatHeadsBoltzProgramFactory{};
}

void NLPConcatHeadsBoltzDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void NLPConcatHeadsBoltzDeviceOperation::validate_on_program_cache_miss(
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

    if (tensor_args.preallocated_output.has_value()) {
        const auto computed_output_spec = compute_output_specs(args, tensor_args);
        const auto& preallocated_output = tensor_args.preallocated_output.value();
        TT_FATAL(
            preallocated_output.logical_shape() == computed_output_spec.logical_shape(),
            "Preallocated output shape must match computed output shape");
    }
}

NLPConcatHeadsBoltzDeviceOperation::spec_return_value_t NLPConcatHeadsBoltzDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    const auto& input_tensor = tensor_args.input;
    const auto& input_shape = input_tensor.logical_shape();

    auto num_heads = input_shape[0];
    auto sequence_length = input_shape[2];
    auto head_dim = input_shape[3];

    auto hidden_dim = num_heads * head_dim;

    Shape output_shape({1, sequence_length, sequence_length, hidden_dim});

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

NLPConcatHeadsBoltzDeviceOperation::tensor_return_value_t NLPConcatHeadsBoltzDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input.device());
}

}  // namespace ttnn::operations::experimental::nlp_concat_heads_boltz

namespace ttnn::prim {

ttnn::operations::experimental::nlp_concat_heads_boltz::tensor_return_value_t nlp_concat_heads_boltz(
    const Tensor& input_tensor,
    const tt::tt_metal::MemoryConfig& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    using OperationType = ttnn::operations::experimental::nlp_concat_heads_boltz::NLPConcatHeadsBoltzDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{memory_config};
    auto tensor_args = OperationType::tensor_args_t{input_tensor, std::move(optional_output_tensor)};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
